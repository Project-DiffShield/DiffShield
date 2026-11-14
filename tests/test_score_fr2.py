"""CPU tests for scripts/score_fr2.py with tiny fake images and a fake embedder (no downloads, no torch)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("score_fr2", REPO / "scripts" / "score_fr2.py")
fr = importlib.util.module_from_spec(_spec)
sys.modules["score_fr2"] = fr
_spec.loader.exec_module(fr)

N = 12


def person_img(i, noise=0.0, blank=False, seed=0):
    """16x16 image whose 'identity' is a fixed random pattern per person; blank = no face."""
    rng = np.random.default_rng(1000 + i)
    base = rng.uniform(0, 255, (16, 16, 3))
    if noise:
        base = base + np.random.default_rng(seed).normal(0, noise, base.shape)
    if blank:
        base = np.zeros((16, 16, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


class FakeEmbedder:
    def embed_paths(self, paths):
        out = []
        for p in paths:
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32).ravel()
            if a.std() < 1:
                out.append(None)
                continue
            a = a - a.mean()
            out.append((a / np.linalg.norm(a)).astype(np.float32))
        return out


@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data" / "test"
    data.mkdir(parents=True)
    for i in range(N):
        person_img(i).save(data / f"{100 + i}.png")
    unit = tmp_path / "in" / "T04"
    res = unit / "results" / "eval_x"
    (res / "edits").mkdir(parents=True)
    (unit / "cache" / "clean_edits").mkdir(parents=True)
    rows = []

    def add(method, i, kind, purifier="none", editor="sd15_img2img", seed=1, in_cache=False):
        sub = unit / "cache" / "clean_edits" if in_cache else res / "edits"
        name = f"{method}_{editor}_{purifier}_{i}_{seed}.png"
        if kind == "same":       # edit still looks like the person
            img = person_img(i, noise=5, seed=seed)
        elif kind == "other":    # edit looks like someone else
            img = person_img((i + 5) % N, noise=5, seed=seed)
        elif kind == "noface":
            img = person_img(i, blank=True)
        if kind != "missing":
            img.save(sub / name)
        orig = f"/kaggle/working/T04/{'cache/clean_edits' if in_cache else 'results/eval_x/edits'}/{name}"
        rows.append({"run_id": "r", "method": method, "image_id": str(100 + i), "split": "test", "seed": str(seed),
                     "editor": editor, "prompt_id": "p0", "strength": "0.55", "purifier": purifier,
                     "edit_path": orig, "face_detected": "0" if kind == "noface" else "1",
                     "arcface_sim": "nan" if kind == "noface" else "0.5"})

    for i in range(N):
        add("clean", i, "same", in_cache=True)
        add("clean", i, "same", in_cache=True, seed=2)
        add("old", i, "same")
        add("new", i, "other")
    add("new", 0, "noface", seed=2)
    add("new", 1, "missing", seed=2)
    pd.DataFrame(rows).to_csv(res / "metrics.shard0.csv", index=False)
    # a protect-style csv must be ignored
    pr = unit / "results" / "prot_x"
    pr.mkdir()
    pd.DataFrame({"method": ["old"], "image_id": ["100"], "png_psnr": [40.0]}).to_csv(pr / "metrics.csv", index=False)
    return tmp_path


def test_discovery_ignores_protect_csv_and_names_unit(world):
    units = fr.discover_units(input_roots=[world / "in"])
    assert list(units) == [world / "in" / "T04"]
    assert len(units[world / "in" / "T04"]) == 1


def test_path_resolver_suffix(world):
    unit = world / "in" / "T04"
    r = fr.PathResolver([unit, unit.parent])
    p = r("/kaggle/working/T04/results/eval_x/edits/old_sd15_img2img_none_3_1.png")
    assert p is not None and p.is_file()
    assert r("/kaggle/working/T04/results/eval_x/edits/nope.png") is None


def test_end_to_end(world):
    units = fr.discover_units(input_roots=[world / "in"])
    out = world / "out"
    res = fr.run(units, world / "data", out, FakeEmbedder(), percentile=95.0)
    sc = res["scored"]
    assert (out / "fr2_T04.csv").is_file() and (out / "fr2_threshold.json").is_file()
    cal = json.loads((out / "fr2_threshold.json").read_text())
    assert cal["basis_summary"] == "clean_edits"
    assert "clean edits" in cal["pooled"]["basis"]
    thr = cal["pooled"]["threshold"]
    # impostor similarity of random patterns is small compared with the same-person similarity
    assert thr < 0.5
    st = sc.set_index(["method", "image_id", "seed"])["fr2_status"]
    assert st[("new", "100", "2")] == "edit_no_face"
    assert st[("new", "101", "2")] == "edit_missing"
    assert st[("old", "100", "1")] == "ok"
    old = sc[(sc.method == "old")]
    assert (pd.to_numeric(old["fr2_sim"]) > 0.9).all()
    t = res["table"].set_index("method")
    assert t.loc["old", "fr2_psr"] == 0.0           # same person: above the threshold
    # 'new' edits look like another person (sim ~ impostor level) plus one no-face edit
    assert t.loc["new", "fr2_psr"] > 0.5
    assert t.loc["new", "n_photos"] == N
    assert t.loc["old", "arcface_psr"] == 0.0       # arcface_sim 0.5 copied from the CSV, above 0.1005
    assert (out / "fr2_psr_table.md").read_text().count("NOTE: X3-X5") == 1
    # original metrics untouched
    orig = pd.read_csv(world / "in" / "T04" / "results" / "eval_x" / "metrics.shard0.csv")
    assert "fr2_sim" not in orig.columns


def test_threshold_falls_back_to_sources_without_clean_rows(world):
    csv = world / "in" / "T04" / "results" / "eval_x" / "metrics.shard0.csv"
    df = pd.read_csv(csv, dtype=str)
    df[df.method != "clean"].to_csv(csv, index=False)
    units = fr.discover_units(input_roots=[world / "in"])
    res = fr.run(units, world / "data", world / "out2", FakeEmbedder())
    cal = res["calibration"]
    assert cal["basis_summary"] == "sources_only"
    assert "distinct source photos" in cal["pooled"]["basis"]
    assert cal["sources_only"]["n_pairs"] == N * (N - 1) // 2


def test_impostor_excludes_same_person():
    S = np.eye(3, dtype=np.float32)
    E = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    v = fr._impostor_sims(E, ["a", "b"], ["a", "b", "c"], S)
    assert v.size == 4 and (v == 0).all()


def test_max_rows_and_missing_source(world):
    for f in (world / "data" / "test").glob("100.png"):
        f.unlink()
    units = fr.discover_units(input_roots=[world / "in"])
    res = fr.run(units, world / "data", world / "out3", FakeEmbedder(), max_rows=20)
    sc = res["scored"]
    assert len(sc) == 20
    assert (sc[sc.image_id == "100"]["fr2_status"] == "source_missing").all()
