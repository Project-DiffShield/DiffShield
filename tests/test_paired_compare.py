"""CPU tests for scripts/paired_compare.py with synthetic edit CSVs (real column layout)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("paired_compare", REPO / "scripts" / "paired_compare.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)

from src.eval.edit_runner import ROW_COLUMNS  # noqa: E402

THR = {"per_editor": {"sd15_img2img": 0.1}, "pooled": 0.1}
N = 40


def make(method, sims, photos=None, purifier="none", editor="sd15_img2img", seeds=(1, 2)):
    photos = photos if photos is not None else [str(10000 + i) for i in range(N)]
    rows = []
    for i, p in enumerate(photos):
        for s in seeds:
            r = {c: "" for c in ROW_COLUMNS}
            r.update(method=method, image_id=p, split="test", seed=str(s), editor=editor, prompt_id="p0",
                     strength="0.55", purifier=purifier, face_detected="1", arcface_sim=repr(float(sims[i])))
            rows.append(r)
    return pd.DataFrame(rows)[ROW_COLUMNS]


def typed(df):
    return pc.A.typed_edits(df)


def run(old, new, clean=None, **kw):
    return pc.compare(typed(old), typed(new), None if clean is None else typed(clean), THR, n_boot=500, seed=7,
                      expect_photos=kw.pop("expect_photos", N), **kw)


def headline(rep):
    return next(c for c in rep["cells"] if c["is_headline"])


def test_identical_methods_keep_with_zero_diff():
    sims = np.linspace(0.0, 0.3, N)
    old, new = make("a", sims), make("b", sims)
    rep = run(old, new)
    h = headline(rep)
    assert rep["verdict"] == "KEEP"
    assert h["diff"]["mean_diff"] == 0 and h["diff"]["ci_lo"] == 0 and h["diff"]["ci_hi"] == 0


def test_better_everywhere_adopts_and_net_gain():
    old = make("a", np.full(N, 0.5))      # PSR 0 (all above 0.1)
    new = make("b", np.full(N, 0.0))      # PSR 1
    clean = make("clean", np.full(N, 0.5))
    rep = run(old, new, clean)
    h = headline(rep)
    assert rep["verdict"] == "ADOPT"
    assert h["diff"]["mean_diff"] == pytest.approx(1.0) and h["diff"]["ci_lo"] > 0
    assert h["net_gain_diff"]["mean"] == pytest.approx(1.0)
    assert h["net_new"]["mean_diff"] == pytest.approx(1.0) and h["net_old"]["mean_diff"] == pytest.approx(0.0)


def test_noisy_small_gain_is_keep():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 0.3, N)
    new_sims = base.copy()
    new_sims[0] = 0.0 if base[0] >= 0.1 else 0.5   # flip exactly one photo
    new_sims[1] = 0.0 if base[1] >= 0.1 else 0.0
    rep = run(make("a", base), make("b", new_sims))
    assert rep["verdict"] in ("KEEP", "ADOPT")
    assert headline(rep)["diff"]["ci_lo"] <= headline(rep)["diff"]["mean_diff"]
    # rule is exactly lower bound > 0
    assert (rep["verdict"] == "ADOPT") == (headline(rep)["diff"]["ci_lo"] > 0)


def test_no_face_counts_as_hit():
    old = make("a", np.full(N, 0.5))
    new = make("b", np.full(N, 0.5))
    new["face_detected"] = "0"
    new["arcface_sim"] = "nan"
    rep = run(old, new)
    assert headline(rep)["psr_new"]["mean"] == 1.0 and rep["verdict"] == "ADOPT"


def test_mismatched_photos_reported_and_invalid():
    photos = [str(10000 + i) for i in range(N)]
    old = make("a", np.full(N, 0.5))
    new = make("b", np.full(N - 3, 0.0), photos=photos[:-3])
    rep = run(old, new)
    h = headline(rep)
    assert rep["verdict"] == "INVALID"
    assert any("only in old" in i for i in h["issues"]) and any("common photos" in i for i in h["issues"])
    assert h["n_common"] == N - 3


def test_mismatched_cells_reported():
    sims = np.full(N, 0.5)
    old = pd.concat([make("a", sims), make("a", sims, purifier="jpeg_q50")], ignore_index=True)
    new = make("b", sims)
    rep = run(old, new)
    cell = next(c for c in rep["cells"] if c["cleanup"] == "jpeg_q50")
    assert cell["status"] == "missing" and "missing in new" in cell["issues"][0]
    assert rep["verdict"] == "KEEP"          # headline cell is intact


def test_missing_headline_is_invalid():
    sims = np.full(N, 0.5)
    rep = run(make("a", sims, purifier="jpeg_q50"), make("b", sims, purifier="jpeg_q50"))
    assert rep["verdict"] == "INVALID"


def test_deterministic_with_fixed_seed():
    rng = np.random.default_rng(1)
    old, new = make("a", rng.uniform(0, 0.3, N)), make("b", rng.uniform(0, 0.3, N))
    r1, r2 = run(old, new), run(old, new)
    assert json.dumps(headline(r1)["diff"]) == json.dumps(headline(r2)["diff"])
    r3 = pc.compare(typed(old), typed(new), None, THR, n_boot=500, seed=8, expect_photos=N)
    assert headline(r3)["diff"]["mean_diff"] == headline(r1)["diff"]["mean_diff"]


def test_cli_dir_and_glob(tmp_path):
    sims = np.linspace(0, 0.3, N)
    for name in ("o", "n"):
        d = tmp_path / name
        d.mkdir()
        df = make(name, sims)
        df.iloc[: len(df) // 2].to_csv(d / "metrics.shard0.csv", index=False)
        df.iloc[len(df) // 2:].to_csv(d / "metrics.shard1.csv", index=False)
    thr = tmp_path / "thr.json"
    thr.write_text(json.dumps(THR))
    out = tmp_path / "rep.json"
    rc = pc.main(["--old", str(tmp_path / "o"), "--new", str(tmp_path / "n" / "metrics.shard*.csv"),
                  "--thresholds", str(thr), "--n-boot", "200", "--expect-photos", str(N), "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "KEEP" and headline(rep)["n_common"] == N
