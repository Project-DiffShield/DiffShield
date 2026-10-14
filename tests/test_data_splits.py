import hashlib
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MS = _load("make_splits")

N_IMG = 700
N_IDENT = 500   # 200 identities appear twice -> a naive split would leak identities
ATTR_NAMES = ["Blurry", "Wearing_Hat", "Eyeglasses", "Male", "Smiling", "Young", "Bald", "Pale_Skin"] + [f"A{i}" for i in range(32)]


def synth():
    rng = np.random.RandomState(0)
    fns = [f"{i}.jpg" for i in range(N_IMG)]
    attr = (rng.rand(N_IMG, len(ATTR_NAMES)) < 0.3).astype(np.float32)
    ident = {f: str(i % N_IDENT) for i, f in enumerate(fns)}
    calib = [fns[i] for i in (5, 17, 33, 60)]
    # v1-like eval list: 70 images with distinct identities, disjoint from calib
    v1, seen = [], {ident[c] for c in calib}
    for f in fns[100:]:
        if ident[f] not in seen and len(v1) < 70:
            v1.append(f)
            seen.add(ident[f])
    return fns, attr, ident, calib, v1


def stats_fn(fname):
    h = int(hashlib.sha256(fname.encode()).hexdigest()[:8], 16)
    r = np.random.RandomState(h)
    return {"luma_mean": float(r.rand()), "luma_std": float(0.05 + 0.3 * r.rand()), "lap_var": float(r.rand() * 0.01)}


def pose_anno(fns):
    r = np.random.RandomState(3)
    return {f: (float(r.randn() * 20), float(r.randn() * 10), float(r.randn() * 5)) for f in fns}


def identity_of(f, ident):
    return ident[f]


def run(gate, **kw):
    fns, attr, ident, calib, v1 = synth()
    return MS.build_splits(attr, ATTR_NAMES, fns, calib, v1, gate, seed=2026, stats_fn=stats_fn,
                           pose_anno=pose_anno(fns), **kw), (fns, ident, calib, v1)


def test_label_identity_splits_disjoint_and_deterministic():
    fns, attr, ident, calib, v1 = synth()
    (s1, m1), _ = run(MS.LabelIdentity(ident))
    (s2, m2), _ = run(MS.LabelIdentity(ident))
    assert s1 == s2  # deterministic
    assert {k: len(v) for k, v in s1.items()} == MS.SIZES
    assert s1["test"][:70] == v1                 # all v1 eval images kept first, in v1 order
    allf = [f for v in s1.values() for f in v]
    assert len(allf) == len(set(allf))           # file disjoint
    assert not set(calib) & set(allf)            # calibration excluded
    ids = [ident[f] for f in allf]
    assert len(ids) == len(set(ids))             # identity disjoint across ALL splits (one image per person)
    assert m1["identity"]["mode"] == "labels"
    assert m1["v1_eval70_identity_clashes"] == 0


def test_different_seed_changes_selection():
    fns, attr, ident, calib, v1 = synth()
    a, _ = MS.build_splits(attr, ATTR_NAMES, fns, calib, v1, MS.LabelIdentity(ident), seed=2026, stats_fn=stats_fn, pose_anno=pose_anno(fns))
    b, _ = MS.build_splits(attr, ATTR_NAMES, fns, calib, v1, MS.LabelIdentity(ident), seed=7, stats_fn=stats_fn, pose_anno=pose_anno(fns))
    assert a["test"][:70] == b["test"][:70]
    assert a["ae_pool"] != b["ae_pool"]


def test_arcface_dedup_gate_synthetic_embeddings():
    fns, attr, ident, calib, v1 = synth()
    rng = np.random.RandomState(1)
    centers = rng.randn(N_IDENT, 64)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def analyze(f):
        c = centers[int(ident[f])] + 0.02 * rng.randn(64) / 8
        return {"emb": c / np.linalg.norm(c), "kps": None}

    faces = MS.FaceCache(analyze)
    gate = MS.ArcFaceIdentity(faces, 0.5)
    (s, meta) = MS.build_splits(attr, ATTR_NAMES, fns, calib, v1, gate, seed=2026, stats_fn=stats_fn, pose_anno=pose_anno(fns))
    allf = [f for v in s.values() for f in v]
    ids = [ident[f] for f in allf]
    assert len(ids) == len(set(ids))
    assert meta["identity"]["mode"] == "arcface" and meta["identity"]["threshold_verified"] is False


def test_arcface_noface_candidates_rejected():
    faces = MS.FaceCache(lambda f: {"emb": None, "kps": None} if f == "0.jpg" else {"emb": np.eye(8)[int(f[:-4]) % 8], "kps": None})
    g = MS.ArcFaceIdentity(faces, 0.5)
    assert g.try_add("0.jpg") is False
    assert g.try_add("1.jpg") is True
    assert g.try_add("9.jpg") is False  # same direction as 1.jpg (9 % 8 == 1)
    assert g.try_add("2.jpg") is True


def test_messy_strata_and_pose_rule():
    fns, attr, ident, calib, v1 = synth()
    pa = pose_anno(fns)
    (s, meta) = MS.build_splits(attr, ATTR_NAMES, fns, calib, v1, MS.LabelIdentity(ident), seed=2026, stats_fn=stats_fn, pose_anno=pa)
    recs = meta["messy"]["records"]
    assert len(recs) == 20 == len(s["messy"])
    assert {r["file"] for r in recs} == set(s["messy"])
    strata = {r["stratum"] for r in recs}
    assert strata <= {"blur", "lighting", "occlusion", "pose"} and "pose" in strata and "blur" in strata
    score = {f: max(abs(v[0]), abs(v[1])) for f, v in pa.items()}
    cut = np.quantile(list(score.values()), 0.85)  # pose picks come from the (pool) top decile -> well above the median
    assert all(score[r["file"]] >= np.median(list(score.values())) for r in recs if r["stratum"] == "pose")
    assert cut > np.median(list(score.values()))
    assert meta["messy"]["pose_source"] == "pose_anno"


def test_messy_without_stats_or_pose_redistributes():
    fns, attr, ident, calib, v1 = synth()
    s, meta = MS.build_splits(attr, ATTR_NAMES, fns, calib, v1, MS.LabelIdentity(ident), seed=2026)
    assert len(s["messy"]) == 20
    assert any("no pose annotation" in n for n in meta["notes"])


def test_check_splits_catches_leaks():
    fns, attr, ident, calib, v1 = synth()
    (s, _), _ = run(MS.LabelIdentity(ident))
    bad = {k: list(v) for k, v in s.items()}
    bad["val"][0] = bad["train"][0]
    with pytest.raises(AssertionError):
        MS.check_splits(bad, calib, v1)
    bad = {k: list(v) for k, v in s.items()}
    bad["val"][0] = calib[0]
    with pytest.raises(AssertionError):
        MS.check_splits(bad, calib, v1)


def test_parsers(tmp_path):
    attr = tmp_path / "attr.txt"
    attr.write_text("2\nA B C\n0.jpg 1 -1 1\n1.jpg -1 -1 1\n", encoding="utf-8")
    fns, names, m = MS.parse_attr_file(str(attr))
    assert fns == ["0.jpg", "1.jpg"] and names == ["A", "B", "C"]
    assert m.tolist() == [[1, 0, 1], [0, 0, 1]]
    mp = tmp_path / "map.txt"
    mp.write_text("idx orig_idx orig_file\n0 119613 119614.jpg\n1 5 000006.jpg\n", encoding="utf-8")
    idf = tmp_path / "id.txt"
    idf.write_text("119614.jpg 77\n000006.jpg 78\n", encoding="utf-8")
    assert MS.parse_hq_identity_map(str(mp), str(idf)) == {"0.jpg": "77", "1.jpg": "78"}
    pose = tmp_path / "pose.txt"
    pose.write_text("2\n0.jpg 1.5 -2.0 0.3\n1 -30 4 1\nbad line\n", encoding="utf-8")
    pa = MS.parse_pose_anno(str(pose))
    assert pa["0.jpg"] == (1.5, -2.0, 0.3) and pa["1.jpg"] == (-30.0, 4.0, 1.0)


def test_write_splits_is_frozen(tmp_path):
    (s, meta), _ = run(MS.LabelIdentity(synth()[2]))
    MS.write_splits(s, meta, str(tmp_path))
    assert set(os.listdir(tmp_path)) == {"test.txt", "messy.txt", "train.txt", "val.txt", "ae_pool.txt", "splits_meta.json"}
    assert MS.read_list(str(tmp_path / "test.txt")) == s["test"]
    MS.write_splits(s, meta, str(tmp_path))  # same content: fine
    s2 = {k: list(v) for k, v in s.items()}
    s2["val"] = s2["val"][::-1]
    with pytest.raises(FileExistsError):
        MS.write_splits(s2, meta, str(tmp_path))
    MS.write_splits(s2, meta, str(tmp_path), force=True)
