"""CPU tests for scripts/select_tuning_photos.py on synthetic clustered features (no images, no models)."""
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts import select_tuning_photos as st  # noqa: E402

D = 16
K = 4                        # clusters
SIZES = [300, 300, 300, 150]
KW = dict(n_search=3, n_confirm_protos=3, n_confirm_crit=2, n_random=50, n_cov_random=10, bandwidth_n=400,
          chunk=256, log=lambda *_: None)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(K, D)))
    clip, lab = [], []
    for k, n in enumerate(SIZES):
        clip.append(centers[k] + 0.08 * rng.normal(size=(n, D)))
        lab += [k] * n
    clip = _unit(np.concatenate(clip)).astype(np.float32)
    lab = np.asarray(lab)
    N = len(lab)
    arc = _unit(rng.normal(size=(N, 32))).astype(np.float32)      # distinct random identities (cos ~ 0)
    attr = (rng.random((N, 40)) < 0.3 + 0.1 * lab[:, None]).astype(np.int8)
    pose = rng.normal(size=(N, 3)).astype(np.float32)
    names = np.asarray([f"{i}.jpg" for i in range(N)])
    return dict(names=names, clip=clip, arcface=arc, has_face=np.ones(N, bool), attr=attr,
                attr_names=np.asarray([f"a{i}" for i in range(40)]), pose=pose), lab


def _ids(names):
    return [int(n.split(".")[0]) for n in names]


def test_prototypes_hit_every_cluster():
    data, lab = make_data()
    res = st.run_selection(data, [], [], **{**KW, "n_search": 4})
    idx = _ids(res["tune_search"])
    assert sorted(set(lab[idx])) == list(range(K))               # one prototype per cluster
    assert len(set(idx)) == len(idx)


def test_identity_gate_blocks_same_person():
    data, lab = make_data()
    a = data["arcface"].copy()
    a[lab == 0] = a[0]                                           # whole cluster 0 = one person
    data["arcface"] = a
    res = st.run_selection(data, [], [], **{**KW, "n_search": 4, "n_confirm_protos": 4})
    allsel = _ids(res["tune_search"] + res["tune_confirm"])
    assert sum(lab[i] == 0 for i in allsel) <= 1
    assert res["max_pairwise_arcface_cosine_among_selected"] < st.ID_THR


def test_exclusions_leak_gate_and_confirm_disjoint():
    data, lab = make_data()
    a = data["arcface"].copy()
    excl = [f"{i}.jpg" for i in range(0, 50)]                    # excluded by name
    a[700] = a[10]                                               # same person as an excluded image
    data["arcface"] = a
    hf = data["has_face"].copy()
    hf[701] = False                                              # no face
    data["has_face"] = hf
    pool, counts = st.build_pool(list(data["names"]), a, hf, excl)
    assert counts["n_identity_leak_removed"] >= 1 and counts["n_no_face"] == 1
    assert 700 not in pool and 701 not in pool and not set(range(50)) & set(pool.tolist())
    res = st.run_selection(data, excl, [], **KW)
    sel = set(res["tune_search"]) | set(res["tune_confirm"])
    assert not sel & set(excl) and "700.jpg" not in sel and "701.jpg" not in sel
    assert not set(res["tune_search"]) & set(res["tune_confirm"])
    assert len(res["tune_confirm"]) == 5 and res["confirm_criticisms"] == res["tune_confirm"][3:]


def test_criticisms_come_from_undercovered_region():
    _, lab = make_data()
    rng = np.random.default_rng(1)
    X = (6.0 * rng.normal(size=(K, D)))[lab] + 0.3 * rng.normal(size=(len(lab), D))   # well separated clusters
    X = X.astype(np.float32)
    s = st.median_bandwidth(X, 400, np.random.default_rng(0))
    colm = st.colmeans(X, X, s, 256)
    protos, _ = st.greedy_prototypes(X, colm, s, 3, np.ones(len(X), bool))
    assert sorted(lab[protos]) == [0, 1, 2]                      # the three big clusters; cluster 3 is not covered
    crit, trace = st.greedy_criticisms(X, colm, s, protos, 1, np.ones(len(X), bool), None)
    assert lab[crit[0]] == 3                                     # the under-covered cluster has the largest |witness|
    w = np.abs(colm - st.rbf(X, X[protos], s).mean(1))
    assert crit[0] == int(np.argmax(w))                          # first pick = max |witness| (log-det gain is 0)
    crit3, _ = st.greedy_criticisms(X, colm, s, protos, 3, np.ones(len(X), bool), None)
    assert len(set(crit3)) == 3


def test_confirm_criticisms_not_same_identity_as_search():
    data, lab = make_data()
    res = st.run_selection(data, [], [], **KW)
    ids = _ids(res["tune_search"] + res["tune_confirm"])
    A = data["arcface"][ids]
    G = A @ A.T
    np.fill_diagonal(G, -1)
    assert G.max() < st.ID_THR


def test_determinism_and_report_shapes():
    data, _ = make_data()
    val = [f"{i}.jpg" for i in (1, 2, 3, 4)]
    r1 = st.run_selection(data, [], val, **KW)
    r2 = st.run_selection(data, [], val, **KW)
    assert r1["tune_search"] == r2["tune_search"] and r1["tune_confirm"] == r2["tune_confirm"]
    assert r1["evidence"]["old_val"]["mmd2_clip"]["ours"] == pytest.approx(r2["evidence"]["old_val"]["mmd2_clip"]["ours"], abs=1e-6)
    ev = r1["evidence"]
    assert set(ev) == {"tune_search", "tune_confirm_prototypes", "tune_confirm", "old_val"}
    b = ev["tune_search"]["mmd2_clip"]
    assert b["n_random"] == 50 and b["random_p05"] <= b["random_p50"] <= b["random_p95"]
    assert 1 / 51 <= b["p_rank"] <= 1.0
    assert set(ev["tune_search"]["mmd2_attr_pose"]) == set(b)
    assert len(ev["tune_search"]["marginals"]["attributes"]) == 40
    assert set(ev["tune_search"]["coverage_clip"]) == {"mean_dist", "max_dist"}
    assert b["ours"] < b["random_p05"]                           # optimised set beats random in the selection space


def test_mmd2_and_colmeans_match_dense():
    data, _ = make_data()
    X = data["clip"]
    s = st.median_bandwidth(X, 400, np.random.default_rng(0))
    colm = st.colmeans(X, X, s, 100)
    Kd = st.rbf(X, X, s)
    assert np.allclose(colm, Kd.mean(1), atol=1e-5)
    idx = [0, 400, 700, 900]
    dense = Kd[np.ix_(idx, idx)].mean() - 2 * Kd[idx].mean() + Kd.mean()
    assert st.mmd2(X, idx, colm, float(colm.mean()), s) == pytest.approx(dense, abs=1e-5)


def test_cli_end_to_end_and_refuses_overwrite(tmp_path):
    data, _ = make_data()
    npz = tmp_path / "pool.npz"
    np.savez(npz, **data)
    sp = tmp_path / "splits"
    sp.mkdir()
    for nm, lst in {"test": ["0.jpg"], "messy": ["1.jpg"], "ae_pool": ["2.jpg"], "v1_calib": ["3.jpg"],
                    "val": ["5.jpg", "6.jpg"]}.items():
        (sp / f"{nm}.txt").write_text("\n".join(lst) + "\n")
    out, rep = tmp_path / "out", tmp_path / "rep"
    argv = ["--npz", str(npz), "--splits_dir", str(sp), "--out_splits_dir", str(out), "--report_dir", str(rep),
            "--n_search", "3", "--n_confirm_protos", "3", "--n_confirm_crit", "2", "--n_random", "20",
            "--n_cov_random", "5", "--bandwidth_n", "300"]
    assert st.main(argv) == 0
    assert len((out / "tune_search.txt").read_text().split()) == 3
    assert len((out / "tune_confirm.txt").read_text().split()) == 5
    meta = json.loads((out / "tune_meta.json").read_text())
    assert len(meta["sha256"]["tune_search"]) == 64 and len(meta["npz_sha256"]) == 64
    assert (rep / "selection_report.json").exists() and (rep / "selection_report.md").exists()
    assert st.main(argv) == 2                                    # refuses to overwrite
    assert st.main(argv + ["--force"]) == 0


def test_embed_pool_imports_without_models():
    import importlib
    m = importlib.import_module("scripts.embed_pool")
    assert callable(m.main) and m.DEFAULT_CLIP_ID.startswith("openai/")
