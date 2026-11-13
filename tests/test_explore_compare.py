"""CPU test of scripts/explore_compare.py with tiny fake CSVs (no torch needed)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("explore_compare", REPO / "scripts" / "explore_compare.py")
ec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ec)

PHOTOS = [str(100 + i) for i in range(8)]
CELLS = [("sd15_img2img", "none"), ("sd15_img2img", "jpeg_q75"), ("ipadapter_faceid", "none"), ("ipadapter_faceid", "jpeg_q75")]


def make_edits(method, hit_prob, rng, no_face_photo=None):
    rows = []
    for ph in PHOTOS:
        for ed, pu in CELLS:
            for seed in (1, 2):
                hit = rng.random() < hit_prob
                rows.append({"method": method, "image_id": ph, "editor": ed, "purifier": pu, "strength": "0.55",
                             "prompt_id": "0", "seed": seed, "face_detected": 0 if ph == no_face_photo else 1,
                             "arcface_sim": 0.05 if hit else 0.5})
    return pd.DataFrame(rows)


def write_variant(root, name, method, p, rng, psnr, ssim, **kw):
    d = root / name / "results" / f"eval_{method}_EXPL"
    d.mkdir(parents=True)
    make_edits(method, p, rng, **kw).to_csv(d / "metrics.csv", index=False)
    pd.DataFrame({"image_id": PHOTOS, "psnr": psnr, "ssim": ssim}).to_csv(d / "psnr_ssim.csv", index=False)
    return d


def test_compare_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    ok = [40.0] * 8
    write_variant(tmp_path, "X0", "v2_next", 0.1, rng, ok, [0.97] * 8)
    bad_psnr = [40.0] * 6 + [37.0, 36.0]
    write_variant(tmp_path, "X1", "v2_x_eps8", 0.9, rng, bad_psnr, [0.97] * 7 + [0.94])
    # a protect-style CSV in the tree (no arcface_sim) must be ignored
    pr = tmp_path / "X0" / "results" / "prot_v2_next_tune_search"
    pr.mkdir(parents=True)
    pd.DataFrame({"method": ["v2_next"], "image_id": ["100"], "png_psnr": [40.0]}).to_csv(pr / "metrics.csv", index=False)

    v = ec.load_variants([tmp_path / "X0", tmp_path / "X1"])
    assert sorted(v) == ["X0", "X1"]
    cells, summ = ec.compare(v, 0.1005, "X0", 2000, 2026)
    assert len(cells) == 8 and set(cells["variant"]) == {"X0", "X1"}
    s = summ.set_index("variant")
    assert s.loc["X1", "psr_all"] > 0.6 > 0.4 > s.loc["X0", "psr_all"]
    assert abs(s.loc["X0", "psr_all"] - cells[cells.variant == "X0"]["psr"].mean()) < 1e-12
    assert s.loc["X1", "psnr_min"] == 36.0 and s.loc["X1", "ssim_min"] == 0.94
    assert s.loc["X1", "n_violations"] == 2 and s.loc["X0", "n_violations"] == 0
    assert s.loc["X1", "diff_lo"] > 0 and s.loc["X1", "diff_lo"] <= s.loc["X1", "diff_mean"] <= s.loc["X1", "diff_hi"]
    assert np.isnan(s.loc["X0", "diff_mean"]) and s.loc["X1", "diff_n_photos"] == 8
    # deterministic bootstrap
    _, summ2 = ec.compare(v, 0.1005, "X0", 2000, 2026)
    assert summ2.loc[summ2.variant == "X1", "diff_lo"].iloc[0] == s.loc["X1", "diff_lo"]


def test_sd15_and_ipadapter_means_and_no_face_counts_as_hit():
    rng = np.random.default_rng(1)
    e = make_edits("v2_next", 0.0, rng)                      # nothing below threshold
    e.loc[(e.editor == "ipadapter_faceid") & (e.image_id == "100"), "face_detected"] = 0   # one photo with no face on ipadapter
    v = {"X0": {"edits": e, "invis": None, "method": "v2_next"}}
    _, summ = ec.compare(v, 0.1005)
    r = summ.iloc[0]
    assert r["psr_sd15"] == 0.0 and abs(r["psr_ipadapter"] - 1 / 8) < 1e-12
    assert abs(r["psr_all"] - (0.0 + 1 / 8) / 2) < 1e-12
    assert np.isnan(r["psnr_mean"]) and r["n_violations"] is None


def test_cli_writes_csv_and_markdown_and_invis_mode(tmp_path, capsys):
    rng = np.random.default_rng(2)
    write_variant(tmp_path, "X0", "v2_next", 0.2, rng, [41.0] * 8, [0.98] * 8)
    write_variant(tmp_path, "X1", "v2_x_eps8", 0.5, rng, [39.0] * 8, [0.96] * 8)
    out = tmp_path / "out"
    assert ec.main([str(tmp_path / "X0"), str(tmp_path / "X1"), "--out_dir", str(out)]) == 0
    assert (out / "explore_cells.csv").exists() and (out / "explore_summary.csv").exists()
    md = (out / "explore_compare.md").read_text(encoding="utf-8")
    assert "cannot be compared with the other variants at equal invisibility" in md
    assert "not a test-set claim" in md.lower() or "NOT a test-set claim" in md
    assert "sd15_img2img / jpeg_q75" in md and "ipadapter_faceid / none" in md and "2000 draws, seed 2026" in md
    # X0 alone: no diff column values
    assert ec.main([str(tmp_path / "X0"), "--out_dir", str(tmp_path / "out0")]) == 0
    # invisibility mode
    run = tmp_path / "prot"
    run.mkdir()
    pd.DataFrame({"image_id": PHOTOS, "png_psnr": np.arange(8) + 40.0, "png_ssim": [0.97] * 8}).to_csv(run / "metrics.csv", index=False)
    assert ec.main(["--invis_from", str(run), "--invis_out", str(tmp_path / "x" / "psnr_ssim.csv")]) == 0
    got = pd.read_csv(tmp_path / "x" / "psnr_ssim.csv")
    assert list(got.columns) == ["image_id", "psnr", "ssim"] and len(got) == 8
