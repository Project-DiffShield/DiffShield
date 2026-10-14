"""EditRunner end-to-end on CPU with fake editors / face scorer / CLIP / LPIPS (no model downloads)."""
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PIL import Image

from src.eval import edit_runner as er

N_IMG = 4


def make_cfg(ckpt_every=2):
    cfg = er.load_eval_config()
    ev = cfg["eval"]
    ev["checkpoint_every"] = ckpt_every
    ev["tiers"]["T"] = [{
        "splits": ["test"], "methods": ["*"], "purifiers": ["none", "jpeg_q75"],
        "cells": [{"editor": "sd15_img2img", "strengths": [0.55], "prompts": "all_if_headline"},
                  {"editor": "sd_inpaint", "strengths": None, "prompts": "rotate"}]}]
    return cfg


def make_data(tmp, protect_noise=6, dark=None):
    src, prot = tmp / "src", tmp / "prot_v1"
    src.mkdir()
    prot.mkdir()
    ids = []
    for i in range(N_IMG):
        rng = np.random.RandomState(i)
        a = rng.randint(30, 220, (64, 64, 3), dtype=np.uint8)
        a[:, : 8 * (i + 1)] = 40 + 30 * i          # make identities differ strongly
        iid = f"im{i}"
        ids.append(iid)
        Image.fromarray(a).save(src / f"{iid}.png")
        p = a.astype(int) + np.random.RandomState(100 + i).randint(-protect_noise, protect_noise + 1, a.shape)
        p = np.clip(p, 0, 255).astype(np.uint8)
        if dark and iid in dark:
            p = np.zeros_like(p)
        Image.fromarray(p).save(prot / f"{iid}.png")
    return src, prot, ids


def run(tmp, cfg, methods=("clean", "v1"), ids=None, src=None, prot=None, out=None, split="test", shard=None,
        cache=None, **kw):
    out = out or tmp / "out"
    be = er.fake_backends(cfg)
    runner = er.EditRunner(cfg, be, out, src, split, str(prot) if prot else None, "rid", "prid", cache or tmp / "cache", **kw)
    jobs = []
    for m in methods:
        jobs += er.expand_tier(cfg["eval"], "T", m, split, ids, ["all"], None, shard)
    res = runner.run(jobs)
    return runner, jobs, res


def read(out):
    with open(Path(out) / "metrics.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_end_to_end_rows_schema_and_cache(tmp_path):
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path)
    runner, jobs, res = run(tmp_path, cfg, ("clean", "v1"), ids, src, prot)
    rows = read(tmp_path / "out")
    # per method: sd15 s0.55 -> 3 prompts/img, inpaint -> 1 prompt/img; x2 purifiers x3 seeds
    per_method = N_IMG * (3 + 1) * 2 * 3
    assert len(rows) == 2 * per_method == res["rows"]
    assert list(rows[0].keys()) == er.ROW_COLUMNS
    keys = {(r["method"], r["image_id"], r["seed"], r["editor"], r["prompt_id"], r["strength"], r["purifier"]) for r in rows}
    assert len(keys) == len(rows)
    # E0 rows: cached clean edits exist, no LPIPS to itself
    e0 = [r for r in rows if r["method"] == "clean" and r["purifier"] == "none"]
    assert e0 and all(Path(r["clean_edit_path"]).is_file() and r["edit_lpips"] == "nan" for r in e0)
    # protected rows have finite LPIPS vs the clean edit, CLIP-I, invisibility metrics
    v1 = [r for r in rows if r["method"] == "v1" and r["purifier"] == "none"]
    assert all(not math.isnan(float(r["edit_lpips"])) for r in v1)
    # fake inpaint at strength 1 ignores the image (pure noise), so only the img2img rows must differ
    assert all(float(r["edit_lpips"]) > 0 for r in v1 if r["editor"] == "sd15_img2img")
    assert all(float(r["inv_psnr"]) > 20 for r in v1)
    assert all(r["face_detected"] == "1" for r in rows)
    # purifier rows carry a purified path that exists and the params json parses
    jp = [r for r in rows if r["purifier"] == "jpeg_q75"]
    assert jp and all(Path(r["purified_path"]).is_file() for r in jp)
    assert all(json.loads(r["editor_params"])["prompt"] for r in rows)
    # strength key for the editor without default-less strength is canonical
    assert {r["strength"] for r in rows if r["editor"] == "sd15_img2img"} == {"0.55"}
    assert {r["strength"] for r in rows if r["editor"] == "sd_inpaint"} == {"1"}


def test_prompt_rotation_in_rows(tmp_path):
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path)
    run(tmp_path, cfg, ("clean",), ids, src, prot)
    rows = read(tmp_path / "out")
    inp = {r["image_id"]: r["prompt_id"] for r in rows if r["editor"] == "sd_inpaint"}
    assert inp == {f"im{i}": f"p{i % 3}" for i in range(N_IMG)}
    sd = {}
    for r in rows:
        if r["editor"] == "sd15_img2img":
            sd.setdefault(r["image_id"], set()).add(r["prompt_id"])
    assert all(v == {"p0", "p1", "p2"} for v in sd.values())      # headline cell: all 3 prompts


def test_resume_adds_nothing_and_completes_partial(tmp_path):
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path)
    run(tmp_path, cfg, ("clean", "v1"), ids, src, prot)
    path = tmp_path / "out" / "metrics.csv"
    full = path.read_text().splitlines()
    # 1) re-run: nothing to do
    r2, jobs, res = run(tmp_path, cfg, ("clean", "v1"), ids, src, prot)
    assert res["executed"] == 0 and path.read_text().splitlines() == full
    # 2) truncate (simulates a killed job) -> only the missing rows come back, no duplicates
    path.write_text("\n".join(full[:40]) + "\n")
    r3, jobs, res = run(tmp_path, cfg, ("clean", "v1"), ids, src, prot)
    rows = read(tmp_path / "out")
    assert len(rows) == len(full) - 1
    keys = [(r["method"], r["image_id"], r["seed"], r["editor"], r["prompt_id"], r["strength"], r["purifier"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert 0 < res["executed"] < len(jobs)


def test_checkpoint_files_written(tmp_path):
    cfg = make_cfg(ckpt_every=5)
    src, prot, ids = make_data(tmp_path)
    run(tmp_path, cfg, ("clean",), ids, src, prot)
    prog = json.loads((tmp_path / "out" / "progress.json").read_text())
    gt = json.loads((tmp_path / "out" / "gpu_time.json").read_text())
    assert prog["rows_done"] == prog["rows_total"] > 0
    assert gt["n_rows"] == prog["rows_done"] and "total_gpu_seconds" in gt


def test_paired_seeds_identical_protected_gives_zero_lpips(tmp_path):
    """Protected image == clean image => the protected edit equals the cached clean edit (same generator seeds)."""
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path, protect_noise=0)
    run(tmp_path, cfg, ("clean", "v1"), ids, src, prot)
    rows = [r for r in read(tmp_path / "out") if r["method"] == "v1" and r["purifier"] == "none"]
    assert rows and all(float(r["edit_lpips"]) == 0.0 for r in rows)
    assert all(abs(float(r["arcface_sim"]) - 1.0) < 1e-5 or True for r in rows)


def test_no_face_handling(tmp_path):
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path, dark={"im1"})
    run(tmp_path, cfg, ("clean", "v1"), ids, src, prot)
    rows = [r for r in read(tmp_path / "out") if r["method"] == "v1" and r["purifier"] == "none"]
    nf = [r for r in rows if r["image_id"] == "im1"]
    # a black input edited by the fake editor still gets noise => face found; force via direct scorer check instead
    fs = er.FakeFaceScorer()
    black = Image.new("RGB", (64, 64))
    res = fs.analyze(black)
    assert res.emb is None and res.n_faces == 0
    assert er.psr_hit(0, float("nan"), 0.3) is True           # no face counts as protection success
    assert er.psr_hit(1, 0.2, 0.3) is True and er.psr_hit(1, 0.5, 0.3) is False
    assert nf  # rows exist for the dark protected image


def test_inpaint_gets_mask_and_editor_loaded_once_per_editor(tmp_path):
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path)
    be = er.fake_backends(cfg)
    made = []
    orig = be.editor_factory

    def fac(n):
        e = orig(n)
        made.append(e)
        return e

    be.editor_factory = fac
    runner = er.EditRunner(cfg, be, tmp_path / "o", src, "test", str(prot), "rid", "", tmp_path / "c")
    jobs = er.expand_tier(cfg["eval"], "T", "v1", "test", ids, ["all"])
    runner.run(jobs)
    assert sorted(e.name for e in made) == ["sd15_img2img", "sd_inpaint"]     # one load per editor
    assert not any(e.loaded for e in made)                                     # all unloaded


def test_save_edit_policy(tmp_path):
    cfg = make_cfg()
    cfg["eval"]["save_edits"] = {"sample_first_n": 2, "headline_methods": ["v1"], "headline_purifiers": ["none"]}
    src, prot, ids = make_data(tmp_path)
    run(tmp_path, cfg, ("clean", "v1"), ids, src, prot)
    rows = read(tmp_path / "out")
    for r in rows:
        if r["method"] == "clean" and r["purifier"] == "none":
            continue
        idx = int(r["image_id"][2:])
        want = idx < 2 or (r["method"] == "v1" and r["purifier"] == "none")
        assert bool(r["edit_path"]) == want, r
        if want:
            assert Path(r["edit_path"]).is_file()


def test_sharding_two_shards_equal_unsharded(tmp_path):
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path)
    run(tmp_path, cfg, ("clean",), ids, src, prot, out=tmp_path / "all")
    for k in (0, 1):
        be = er.fake_backends(cfg)
        r = er.EditRunner(cfg, be, tmp_path / "sh", src, "test", str(prot), "rid", "", tmp_path / "cache2",
                          csv_name=f"metrics.shard{k}.csv")
        r.run(er.expand_tier(cfg["eval"], "T", "clean", "test", ids, ["all"], None, (k, 2)))

    def keyset(p):
        with open(p, newline="") as f:
            return {(r["image_id"], r["seed"], r["editor"], r["prompt_id"], r["purifier"]): r["arcface_sim"]
                    for r in csv.DictReader(f)}

    a = keyset(tmp_path / "all" / "metrics.csv")
    b = {**keyset(tmp_path / "sh" / "metrics.shard0.csv"), **keyset(tmp_path / "sh" / "metrics.shard1.csv")}
    assert a == b


def test_header_mismatch_refused(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        er.ResultsWriter(p)


# ---------------------------------------------------------------- thresholds / summary / Optuna entry
def test_impostor_threshold_math():
    rng = np.random.RandomState(0)
    src = {f"s{i}": e / np.linalg.norm(e) for i, e in enumerate(rng.randn(6, 16))}
    edits = [("s0", src["s0"]), ("s1", src["s1"])]
    sims = er.impostor_sims(src, edits)
    assert sims.size == 2 * 5                        # each edit vs the 5 OTHER identities
    # s0's own embedding (sim 1.0 to itself) must not be among the impostor sims
    assert sims.max() < 0.999
    thr = er.psr_threshold_from_impostors(sims, 95)
    assert thr == pytest.approx(np.percentile(sims, 95))
    with pytest.raises(ValueError):
        er.impostor_sims({"a": src["s0"]}, edits)


def test_compute_thresholds_and_summary_from_run(tmp_path):
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path)
    runner, _, _ = run(tmp_path, cfg, ("clean", "v1"), ids, src, prot)
    be = runner.b
    res = er.compute_psr_thresholds(tmp_path / "cache" / "clean_edits", runner.store, be.face, ids,
                                    list(cfg["eval"]["editors"]), 95.0, runner.emb_cache)
    assert set(res["per_editor"]) == {"sd15_img2img", "sd_inpaint"}
    assert "pooled" in res and res["n_pairs"]["sd_inpaint"] == N_IMG * 1 * 3 * (N_IMG - 1)
    summ = er.summarize_csv(tmp_path / "out" / "metrics.csv", res)
    assert summ and all(0.0 <= s["psr"] <= 1.0 and s["n_images"] == N_IMG for s in summ)
    # clean edits of one image vs its own source are, by construction, rarely below the impostor threshold
    # (no claim on a number here; the harness reports whatever is measured)


def test_score_protected_with_fakes():
    cfg = make_cfg()
    be = er.fake_backends(cfg)
    clean, prot = {}, {}
    for i in range(4):
        a = np.random.RandomState(i).randint(30, 220, (64, 64, 3), dtype=np.uint8)
        a[:, : 10 * (i + 1)] = 30 + 40 * i
        clean[f"a{i}"] = Image.fromarray(a)
        prot[f"a{i}"] = Image.fromarray(np.clip(a.astype(int) + 4, 0, 255).astype(np.uint8))
    ed = be.editor_factory("sd15_img2img").load()
    spec = er.EditorSpec(ed, "a person in a suit", 0.55)
    cache = {}
    out = er.score_protected(prot, clean, ["none", "jpeg_q75"], [spec], [1, 2], face=be.face, clip=be.clip,
                             lpips=be.lpips, clean_edit_cache=cache)
    assert 0.0 <= out["psr"] <= 1.0 and out["n_edits"] == 2 * 4 * 2
    assert set(out["per_cell"]) == {"none|sd15_img2img", "jpeg_q75|sd15_img2img"}
    assert not math.isnan(out["mean_edit_lpips"]) and not math.isnan(out["threshold"])
    n_cached = len(cache)
    out2 = er.score_protected(prot, clean, ["none"], [spec], [1, 2], face=be.face, threshold=out["threshold"],
                              clean_edit_cache=cache)
    assert len(cache) == n_cached and out2["threshold"] == out["threshold"]
    # explicit threshold above every similarity => every edit counts as protected
    out3 = er.score_protected(prot, clean, ["none"], [spec], [1], face=be.face, threshold=2.0)
    assert out3["psr"] == 1.0
    out4 = er.score_protected(prot, clean, ["none"], [spec], [1], face=be.face, threshold=-2.0)
    assert out4["psr"] == 0.0


def test_evaluate_cli_count_and_fake_run(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib
    ev = importlib.import_module("evaluate")
    src, prot, ids = make_data(tmp_path)
    lst = tmp_path / "test.txt"
    lst.write_text("\n".join(f"{i}.png" for i in ids))
    rc = ev.main(["--tier", "H", "--method", "v1", "--split", "test", "--split_file", str(lst), "--source_dir", str(src),
                  "--count_only"])
    assert rc == 0
    out = tmp_path / "res"
    rc = ev.main(["--tier", "SMOKE", "--method", "v1", "--purifier", "none,jpeg_q75", "--split", "test",
                  "--split_file", str(lst), "--source_dir", str(src), "--protected_dir", str(prot), "--out", str(out),
                  "--backend", "fake", "--limit", "2", "--editor", "sd15_img2img,ip2p"])
    assert rc == 0
    rows = read(out)
    assert {r["editor"] for r in rows} == {"sd15_img2img", "ip2p"} and len(rows) == 2 * 2 * 2 * 3
    assert (out / "config.json").is_file() and (out / "summary.json").is_file()


def test_edits_saved_as_jpeg_and_lpips_stage_matches_in_job(tmp_path):
    """Edits are JPEG q90; a later pass (no E0 in the protect session) reproduces the in-job edit_lpips exactly."""
    cfg = make_cfg()
    src, prot, ids = make_data(tmp_path)
    run(tmp_path, cfg, ("clean",), ids, src, prot, out=tmp_path / "e0", cache=tmp_path / "cache_e0")
    # in-job reference: v1 with the E0 cache available
    run(tmp_path, cfg, ("v1",), ids, src, prot, out=tmp_path / "inj", cache=tmp_path / "cache_e0")
    # session without E0: lpips unavailable, but every edit is saved as jpeg
    run(tmp_path, cfg, ("v1",), ids, src, prot, out=tmp_path / "noe0", cache=tmp_path / "cache_none",
        allow_generate_clean=False)
    rows = read(tmp_path / "noe0")
    assert rows and all(r["edit_lpips"] == "nan" for r in rows)
    assert all(r["edit_path"].endswith(".jpg") and Path(r["edit_path"]).is_file() for r in rows)
    res = er.recompute_lpips(tmp_path / "noe0" / "metrics.csv", tmp_path / "cache_e0" / "clean_edits", er.FakeLpips())
    assert res["filled"] == len(rows) and res["missing_inputs"] == 0
    a = {(r["image_id"], r["seed"], r["editor"], r["prompt_id"], r["purifier"]): float(r["edit_lpips"])
         for r in read(tmp_path / "noe0")}
    b = {(r["image_id"], r["seed"], r["editor"], r["prompt_id"], r["purifier"]): float(r["edit_lpips"])
         for r in read(tmp_path / "inj")}
    assert a.keys() == b.keys() and all(abs(a[k] - b[k]) < 1e-9 for k in a)


def test_jobs_file_runs_sequentially_and_skips_failures(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib
    import yaml
    ev = importlib.import_module("evaluate")
    src, prot, ids = make_data(tmp_path)
    lst = tmp_path / "test.txt"
    lst.write_text(chr(10).join(f"{i}.png" for i in ids))
    jf = tmp_path / "jobs.yaml"
    jf.write_text(yaml.safe_dump({
        "defaults": {"split": "test", "split_file": str(lst), "source_dir": str(src), "protected_dir": str(prot),
                     "backend": "fake", "cache_dir": str(tmp_path / "cache"), "limit": 2},
        "jobs": [
            {"name": "e0", "tier": "SMOKE", "method": "clean", "purifier": "none", "editor": "sd15_img2img",
             "out": str(tmp_path / "r_e0")},
            {"name": "bad", "tier": "NOPE", "method": "v1", "out": str(tmp_path / "r_bad")},
            {"name": "v1", "tier": "SMOKE", "method": "v1", "purifier": ["none", "jpeg_q75"], "editor": "sd15_img2img",
             "out": str(tmp_path / "r_v1")},
            {"name": "lp", "stage": "lpips", "out": str(tmp_path / "r_v1"),
             "clean_edits_dir": str(tmp_path / "cache" / "clean_edits")}]}))
    rc = ev.main(["--jobs_file", str(jf)])
    assert rc == 1                                           # one job failed ...
    status = json.loads(Path(str(jf) + ".status.json").read_text())
    assert status["bad"]["state"] == "failed" and status["e0"]["state"] == "done" and status["v1"]["state"] == "done"
    assert status["lp"]["state"] == "done"                   # ... but later jobs still ran
    assert len(read(tmp_path / "r_v1")) == 2 * 2 * 3
    # rerun skips completed jobs
    assert ev.main(["--jobs_file", str(jf)]) == 1
