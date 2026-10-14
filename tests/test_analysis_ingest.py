"""ingest_results: validation, de-duplication, coverage, thresholds, safe unzip; compute_edit_lpips (fake backend)."""
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.eval import analysis as A

REPO = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ing = _load("ingest_results")
clp = _load("compute_edit_lpips")


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("kaggle_outputs")
    info = A.write_synthetic_tree(root, n_images=10)
    rep = ing.ingest(root, root / "combined", log=lambda *a: None)
    return root, info, rep


def test_ingest_dedups_validates_and_reports(tree):
    root, info, rep = tree
    comb = root / "combined"
    e = pd.read_csv(comb / "edits.csv", dtype=str, keep_default_na=False)
    assert len(e) == info["rows_written"]                       # the 50 duplicated rows were dropped
    assert rep["edits"]["duplicates_dropped"] == info["dup_rows"]
    assert not e.duplicated(list(A.KEY_COLS)).any()
    assert list(e.columns[:len(A.EDIT_COLUMNS)]) == A.EDIT_COLUMNS
    # junk and broken csv are rejected with a reason, not merged
    reasons = {Path(r["file"]).name: r["reason"] for r in rep["rejected"]}
    assert "metrics.junk.csv" in reasons and "unknown header" in reasons["metrics.junk.csv"]
    assert "metrics.broken.csv" in reasons and "missing columns" in reasons["metrics.broken.csv"]
    # duplicates keep the copy that has LPIPS (the re-run copy had NaN and an older timestamp)
    v1 = e[(e["method"] == "v1") & (e["purifier"] == "none") & (e["editor"] == "sd15_img2img") & (e["strength"] == "0.55")]
    assert pd.to_numeric(v1["edit_lpips"], errors="coerce").notna().all()
    p = pd.read_csv(comb / "protect.csv", dtype=str)
    assert len(p) == info["protect_rows"]


def test_ingest_coverage_cells_images_seeds(tree):
    root, info, _ = tree
    cov = pd.read_csv(root / "combined" / "coverage.csv", dtype=str)
    row = cov[(cov["method"] == "v2_final") & (cov["purifier"] == "none") & (cov["editor"] == "sd15_img2img")]
    assert len(row) == 1 and row.iloc[0]["n_images"] == "10" and row.iloc[0]["seeds"] == "1,2,3"
    assert row.iloc[0]["strengths"] == "0.3,0.55,0.75" and row.iloc[0]["min_images_per_strength"] == "10"
    assert (root / "combined" / "coverage.md").read_text().startswith("# Coverage")
    plan = pd.read_csv(root / "combined" / "coverage_vs_plan.csv")
    assert {"present", "missing"} >= set(plan["status"])
    assert (plan["status"] == "missing").any()                   # the synthetic tree does not run every planned cell


def test_ingest_thresholds_from_npz(tree):
    root, _, rep = tree
    thr = json.loads((root / "combined" / "psr_thresholds.json").read_text())
    assert "sd15_img2img" in thr["per_editor"] and "pooled" in thr
    assert thr["n_pairs"]["sd15_img2img"] == 10 * 9 * 15         # 5 (prompt, strength) combos x 3 seeds per image, 9 impostors each


def test_reingest_is_idempotent_and_keeps_lpips_cache(tmp_path):
    A.write_synthetic_tree(tmp_path, n_images=6)
    ing.ingest(tmp_path, tmp_path / "combined", log=lambda *a: None)
    comb = tmp_path / "combined"
    e = pd.read_csv(comb / "edits.csv", dtype=str, keep_default_na=False)
    # blank one LPIPS value as if it had never been computed, put the true value into the cache file
    i = e.index[(e["method"] == "mist") & (e["purifier"] == "none")][0]
    key = e.loc[i, list(A.KEY_COLS)].to_dict()
    cache = pd.DataFrame([{**key, "edit_lpips": "0.123456"}])
    cache.to_csv(comb / "edit_lpips_cache.csv", index=False)
    for shard in tmp_path.rglob("metrics.shard*.csv"):            # make the source CSVs lack that value
        d = pd.read_csv(shard, dtype=str, keep_default_na=False)
        m = (d["method"] == key["method"]) & (d["image_id"] == key["image_id"]) & (d["seed"] == key["seed"]) & \
            (d["editor"] == key["editor"]) & (d["prompt_id"] == key["prompt_id"]) & (d["strength"] == key["strength"]) & (d["purifier"] == key["purifier"])
        if m.any():
            d.loc[m, "edit_lpips"] = "nan"
            d.to_csv(shard, index=False)
    rep = ing.ingest(tmp_path, comb, log=lambda *a: None)
    e2 = pd.read_csv(comb / "edits.csv", dtype=str, keep_default_na=False)
    assert rep["edits"]["lpips_filled_from_cache"] == 1
    assert e2.loc[(e2[list(A.KEY_COLS)] == pd.Series(key)).all(axis=1), "edit_lpips"].iloc[0] == "0.123456"
    assert len(e2) == len(e)


def test_unzip_extracts_data_only_and_blocks_zip_slip(tmp_path):
    z = tmp_path / "J_outputs.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("results/run/metrics.csv", "a,b\n1,2\n")
        f.writestr("results/run/images/x.png", b"\x89PNG")
        f.writestr("../evil.csv", "x")
        f.writestr("/abs/evil.csv", "x")
    info = ing.safe_extract(z, tmp_path / "J_outputs")
    assert (tmp_path / "J_outputs" / "results" / "run" / "metrics.csv").is_file()
    assert not (tmp_path / "J_outputs" / "results" / "run" / "images" / "x.png").exists()   # images only with --with_images
    assert not (tmp_path / "evil.csv").exists() and not Path("/abs/evil.csv").exists()
    assert info["extracted"] == 1 and info["skipped"] == 3


# ---------------------------------------------------------------- compute_edit_lpips (fake backend)
def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), (value, value, value)).save(path, format="JPEG", quality=90)


@pytest.mark.parametrize("workers", [1, 2])
def test_compute_edit_lpips_fills_nan_resumable(tmp_path, workers):
    comb = tmp_path / "combined"
    comb.mkdir()
    rows = []
    for i, m in enumerate(["clean", "v1", "v1", "v2_final"]):
        iid = f"im{i}"
        r = {c: "" for c in A.EDIT_COLUMNS}
        r.update(method=m, image_id=iid, seed="1", editor="sd15_img2img", prompt_id="p0", strength="0.55", purifier="none",
                 split="test", face_detected="1", arcface_sim="0.5", edit_lpips="nan")
        rows.append(r)
        if m != "clean":
            _save(tmp_path / "jobA" / "edits" / A.edit_rel_path(m, "none", "sd15_img2img", "p0", "0.55", 1, iid).__add__(".jpg"), 100 + 10 * i)
        _save(tmp_path / "jobE0" / "cache" / "clean_edits" / (A.clean_rel_path("sd15_img2img", "p0", "0.55", 1, iid) + ".jpg"), 100)
    # the third row's edit image is missing on disk on purpose
    (tmp_path / "jobA" / "edits" / (A.edit_rel_path("v1", "none", "sd15_img2img", "p0", "0.55", 1, "im2") + ".jpg")).unlink()
    pd.DataFrame(rows).to_csv(comb / "edits.csv", index=False)
    args = ["--combined", str(comb), "--results", str(tmp_path), "--workers", str(workers), "--chunk", "1", "--backend", "fake"]
    assert clp.main(args) == 0
    e = pd.read_csv(comb / "edits.csv", dtype=str, keep_default_na=False)
    vals = pd.to_numeric(e["edit_lpips"], errors="coerce")
    assert np.isnan(vals[0])                                       # E0 clean rows stay NaN by definition
    assert vals[1] == pytest.approx(10 / 255, abs=0.01)            # fake distance: mean abs diff / 255 (jpeg round trip tolerance)
    assert np.isnan(vals[2]) and vals[3] > vals[1]                 # missing image -> stays NaN, listed below
    miss = pd.read_csv(comb / "edit_lpips_missing.csv")
    assert list(miss["image_id"].astype(str)) == ["im2"]
    cache = pd.read_csv(comb / "edit_lpips_cache.csv", dtype=str)
    assert len(cache) == 2
    # second run: nothing left to compute except the missing one; cache not duplicated
    assert clp.main(args) == 0
    assert len(pd.read_csv(comb / "edit_lpips_cache.csv", dtype=str)) == 2
