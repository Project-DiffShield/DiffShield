#!/usr/bin/env python
"""Build the frozen v2 splits from CelebAMask-HQ (job J02, CPU).

Outputs (one CelebA-HQ file name per line, e.g. ``13117.jpg``) in ``--out_dir`` (default data/splits):
    test.txt (100)  messy.txt (20)  train.txt (20)  val.txt (20)  ae_pool.txt (120)  splits_meta.json

Design (decisions documented in splits_meta.json):
  * test = ALL 70 v1 evaluation images (kept verbatim, in v1 order, so paired v1-vs-v2 comparisons exist) + 30 new
    attribute-diverse images. The 4 v1 calibration images (5319, 1715, 6385, 8932) are excluded from every split.
  * messy, val, train, ae_pool are disjoint from each other, from test and from calibration - by FILE and by
    IDENTITY (see below). Train / val / AE pool are never used for test-time numbers.
  * Selection is attribute-diverse like v1: KMeans (random_state = split seed, k-means++, n_init=10) on the 40 binary
    CelebA attributes, one image per cluster = the member closest to the centroid that passes the identity gate.
  * Identity disjointness (one image per person across ALL splits):
      - labels  : CelebA-HQ-to-CelebA-mapping.txt + identity_CelebA.txt (true identities), if both are available;
      - arcface : otherwise greedy ArcFace de-duplication, two images are the same person if cosine >= 0.5
                  (threshold UNVERIFIED), insightface buffalo_l. Only SELECTED images are embedded (a few hundred);
      - none    : only with --allow_no_dedup (recorded loudly in the meta).
    The 70 v1 images are forced in even if two of them share an identity; such pairs are counted in the meta.
  * messy (20): harder conditions by stratum - blur (Blurry attribute or low Laplacian variance), lighting (extreme
    mean luma / low contrast measured on 128 px thumbnails), occlusion (Wearing_Hat / Eyeglasses), pose (top 10% by
    max(|yaw|,|pitch|) from CelebAMask-HQ-pose-anno.txt, rank-based because units are unverified; falls back to a
    landmark yaw proxy when that file is absent). prepare_data.py then applies
    JPEG (q 50-70) + a mild down-up resize to these 20 (parameters recorded there).
Deterministic given the inputs and --seed (default 2026).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SEED = 2026
SIZES = {"test": 100, "messy": 20, "train": 20, "val": 20, "ae_pool": 120}
CALIB_DEFAULT = ["5319.jpg", "1715.jpg", "6385.jpg", "8932.jpg"]
MESSY_QUOTA = {"blur": 5, "lighting": 5, "occlusion": 4, "pose": 6}
ARCFACE_SAME_PERSON_SIM = 0.5   # UNVERIFIED threshold (plan decision)
YAW_THRESHOLD = 0.25            # |nose offset| / inter-ocular distance; frontal ~ 0 (heuristic, UNVERIFIED)


# ----------------------------------------------------------------------------------------------- parsing


def parse_attr_file(path: str) -> Tuple[List[str], List[str], np.ndarray]:
    """CelebAMask-HQ-attribute-anno.txt -> (filenames, attribute names, float32 {0,1} matrix). Same as v1 cell 2."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    names = lines[1].split()
    fns, rows = [], []
    for ln in lines[2:]:
        parts = ln.split()
        fns.append(parts[0])
        rows.append([1 if int(v) == 1 else 0 for v in parts[1:]])
    return fns, names, np.asarray(rows, dtype=np.float32)


def parse_pose_anno(path: str) -> Dict[str, Tuple[float, float, float]]:
    """CelebAMask-HQ-pose-anno.txt -> {file name: (yaw, pitch, roll)}.

    Expected rows ``<name>.jpg yaw pitch roll`` (an optional count header and unparsable lines are skipped; units
    are NOT verified, so callers must use RANKS / quantiles, never absolute angle thresholds). A bare numeric name
    ('12') is mapped to '12.jpg'.
    """
    out: Dict[str, Tuple[float, float, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 4:
                continue
            try:
                vals = tuple(float(v) for v in p[-3:])
            except ValueError:
                continue
            name = p[0]
            if name.isdigit():
                name += ".jpg"
            out[name] = vals  # type: ignore[assignment]
    return out


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_hq_identity_map(mapping_path: str, identity_path: str) -> Dict[str, str]:
    """HQ file name ('0.jpg') -> CelebA identity id, via the HQ->CelebA mapping and identity_CelebA.txt."""
    ident: Dict[str, str] = {}
    with open(identity_path, "r", encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 2:
                ident[p[0]] = p[1]
    out: Dict[str, str] = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for ln in f:
            p = ln.split()
            if len(p) < 3 or not p[0].isdigit():  # header or junk
                continue
            hq, orig = f"{p[0]}.jpg", p[-1]
            if orig in ident:
                out[hq] = ident[orig]
    return out


# ----------------------------------------------------------------------------------------------- identity gates


class IdentityGate:
    """Greedy one-image-per-person gate. ``register`` forces an image in (returns True if it clashed)."""

    mode = "none"

    def conflicts(self, fname: str) -> bool:
        return False

    def register(self, fname: str) -> bool:
        return False

    def try_add(self, fname: str) -> bool:
        if self.conflicts(fname):
            return False
        self.register(fname)
        return True

    def describe(self) -> dict:
        return {"mode": self.mode}


class LabelIdentity(IdentityGate):
    mode = "labels"

    def __init__(self, labels: Dict[str, str]):
        self.labels = labels
        self.used: set = set()
        self.n_unlabeled = 0

    def _id(self, fname: str) -> Optional[str]:
        return self.labels.get(fname)

    def conflicts(self, fname: str) -> bool:
        i = self._id(fname)
        return i is not None and i in self.used

    def register(self, fname: str) -> bool:
        i = self._id(fname)
        if i is None:
            self.n_unlabeled += 1
            return False
        clash = i in self.used
        self.used.add(i)
        return clash

    def describe(self) -> dict:
        return {"mode": self.mode, "n_labels": len(self.labels), "n_unlabeled_registered": self.n_unlabeled}


class FaceCache:
    """fname -> {'emb': np.ndarray|None, 'kps': (5,2)|None}; ``analyze_fn(fname)`` does the (lazy, cached) work."""

    def __init__(self, analyze_fn: Callable[[str], Optional[dict]]):
        self.fn = analyze_fn
        self.cache: Dict[str, Optional[dict]] = {}

    def get(self, fname: str) -> Optional[dict]:
        if fname not in self.cache:
            self.cache[fname] = self.fn(fname)
        return self.cache[fname]


class ArcFaceIdentity(IdentityGate):
    mode = "arcface"

    def __init__(self, faces: FaceCache, threshold: float = ARCFACE_SAME_PERSON_SIM):
        self.faces, self.thr = faces, float(threshold)
        self.embs: List[np.ndarray] = []
        self.n_noface_forced = 0

    def _emb(self, fname: str) -> Optional[np.ndarray]:
        r = self.faces.get(fname)
        if not r or r.get("emb") is None:
            return None
        e = np.asarray(r["emb"], dtype=np.float64)
        return e / max(np.linalg.norm(e), 1e-12)

    def _max_sim(self, e: np.ndarray) -> float:
        return max((float(e @ o) for o in self.embs), default=-1.0)

    def conflicts(self, fname: str) -> bool:
        e = self._emb(fname)
        if e is None:
            return True  # an image with no detectable face is useless for the identity metrics
        return self._max_sim(e) >= self.thr

    def register(self, fname: str) -> bool:
        e = self._emb(fname)
        if e is None:
            self.n_noface_forced += 1
            return False
        clash = self._max_sim(e) >= self.thr
        self.embs.append(e)
        return clash

    def describe(self) -> dict:
        return {"mode": self.mode, "same_person_cosine_threshold": self.thr, "threshold_verified": False,
                "n_noface_forced": self.n_noface_forced}


# ----------------------------------------------------------------------------------------------- selection


def kmeans_select(attr: np.ndarray, pool_idx: np.ndarray, k: int, seed: int, gate: IdentityGate,
                  filenames: Sequence[str], max_try: int = 60) -> List[int]:
    """Attribute-diverse pick of k indices (into ``filenames``): per KMeans cluster, the member nearest the centroid
    that passes ``gate``; leftover shortfall is filled by globally nearest-to-centroid candidates."""
    from sklearn.cluster import KMeans
    pool_idx = np.asarray(pool_idx)
    if len(pool_idx) < k:
        raise ValueError(f"pool of {len(pool_idx)} images is smaller than k={k}")
    X = attr[pool_idx]
    km = KMeans(n_clusters=k, init="k-means++", random_state=seed, n_init=10).fit(X)
    d = ((X - km.cluster_centers_[km.labels_]) ** 2).sum(1)
    picks: List[int] = []
    taken = set()
    for c in range(k):
        mem = np.where(km.labels_ == c)[0]
        for j in mem[np.argsort(d[mem], kind="stable")][:max_try]:
            gi = int(pool_idx[j])
            if gate.try_add(filenames[gi]):
                picks.append(gi)
                taken.add(int(j))
                break
    if len(picks) < k:  # shortfall (clusters whose members all clashed)
        for j in np.argsort(d, kind="stable"):
            if len(picks) >= k:
                break
            if int(j) in taken:
                continue
            gi = int(pool_idx[j])
            if gate.try_add(filenames[gi]):
                picks.append(gi)
                taken.add(int(j))
    if len(picks) < k:
        raise RuntimeError(f"could only select {len(picks)}/{k} identity-disjoint images")
    return picks


def laplacian_var(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def make_image_stats_fn(img_dir: str) -> Callable[[str], dict]:
    """Thumbnail statistics: mean / std of luma and Laplacian variance (128 px, bilinear)."""
    from PIL import Image

    def fn(fname: str) -> dict:
        im = Image.open(os.path.join(img_dir, fname))
        im.draft("RGB", (256, 256))
        g = np.asarray(im.convert("L").resize((128, 128), Image.BILINEAR), dtype=np.float64) / 255.0
        return {"luma_mean": float(g.mean()), "luma_std": float(g.std()), "lap_var": laplacian_var(g)}
    return fn


def yaw_proxy(kps: np.ndarray) -> float:
    """(nose_x - mid-eye x) / inter-ocular distance from 5 landmarks [l-eye, r-eye, nose, l-mouth, r-mouth]."""
    k = np.asarray(kps, dtype=np.float64)
    iod = max(abs(k[1, 0] - k[0, 0]), 1e-6)
    return float((k[2, 0] - 0.5 * (k[0, 0] + k[1, 0])) / iod)


def messy_select(attr: np.ndarray, attr_names: Sequence[str], pool_idx: np.ndarray, filenames: Sequence[str],
                 n: int, seed: int, gate: IdentityGate, stats_fn: Optional[Callable[[str], dict]] = None,
                 faces: Optional[FaceCache] = None, n_stat_sample: int = 800, n_pose_try: int = 200,
                 quota: Optional[Dict[str, int]] = None,
                 pose_anno: Optional[Dict[str, Tuple[float, float, float]]] = None, pose_top_frac: float = 0.10) -> Tuple[List[int], List[dict]]:
    """Pick ``n`` harder-condition images by stratum (see module docstring). Returns (indices, per-image records)."""
    quota = dict(quota or MESSY_QUOTA)
    scale = n / max(sum(quota.values()), 1)
    quota = {k: int(round(v * scale)) for k, v in quota.items()}
    rng = np.random.RandomState(seed)
    pool_idx = np.asarray(pool_idx)
    col = {nm: i for i, nm in enumerate(attr_names)}

    def attr_set(*names):
        m = np.zeros(len(pool_idx), dtype=bool)
        for nm in names:
            if nm in col:
                m |= attr[pool_idx, col[nm]] == 1
        return pool_idx[m]

    cands: Dict[str, List[int]] = {}
    cands["blur"] = [int(i) for i in rng.permutation(attr_set("Blurry"))]
    cands["occlusion"] = [int(i) for i in rng.permutation(attr_set("Wearing_Hat", "Eyeglasses"))]
    stats: Dict[int, dict] = {}
    thresholds: Dict[str, float] = {}
    if stats_fn is not None:
        sample = rng.choice(pool_idx, size=min(len(pool_idx), n_stat_sample), replace=False)
        for gi in sample:
            stats[int(gi)] = stats_fn(filenames[int(gi)])
        lm = np.array([s["luma_mean"] for s in stats.values()])
        ls = np.array([s["luma_std"] for s in stats.values()])
        lv = np.array([s["lap_var"] for s in stats.values()])
        lo_m, hi_m, lo_s, lo_v = (float(np.quantile(lm, .07)), float(np.quantile(lm, .93)),
                                  float(np.quantile(ls, .07)), float(np.quantile(lv, .07)))
        thresholds = {"luma_mean_low": lo_m, "luma_mean_high": hi_m, "luma_std_low": lo_s, "lap_var_low": lo_v}
        light = [gi for gi, s in stats.items() if s["luma_mean"] <= lo_m or s["luma_mean"] >= hi_m or s["luma_std"] <= lo_s]
        cands["lighting"] = [int(i) for i in rng.permutation(light)]
        soft = [gi for gi, s in stats.items() if s["lap_var"] <= lo_v]
        cands["blur"] = cands["blur"] + [int(i) for i in rng.permutation(soft)]  # attribute-blurry first
    pose_order: List[int] = []
    pose_src = "none"
    if pose_anno:
        # preferred: dataset pose annotation. Candidates = top ``pose_top_frac`` of the pool by max(|yaw|, |pitch|).
        sc = {int(i): max(abs(pose_anno[filenames[int(i)]][0]), abs(pose_anno[filenames[int(i)]][1]))
              for i in pool_idx if filenames[int(i)] in pose_anno}
        if sc:
            cut = float(np.quantile(list(sc.values()), 1.0 - pose_top_frac))
            top = [i for i, v in sc.items() if v >= cut]
            pose_order = [int(i) for i in rng.permutation(sorted(top))]
            pose_src = "pose_anno"
    if not pose_order and faces is not None:
        pose_order = [int(i) for i in rng.permutation(pool_idx)[:n_pose_try]]
        pose_src = "landmark_yaw_proxy"

    picks: List[int] = []
    recs: List[dict] = []
    chosen: set = set()
    order = ["blur", "lighting", "occlusion", "pose"]

    def candidates(stratum: str):
        if stratum == "pose" and pose_src == "pose_anno":
            for gi in pose_order:
                yaw, pitch, roll = pose_anno[filenames[gi]]  # type: ignore[index]
                yield gi, {"yaw": yaw, "pitch": pitch, "roll": roll}
        elif stratum == "pose":
            for gi in pose_order:
                r = faces.get(filenames[gi]) if faces is not None else None
                if not r or r.get("kps") is None:
                    continue
                yaw = yaw_proxy(r["kps"])
                if abs(yaw) >= YAW_THRESHOLD:
                    yield gi, {"yaw_proxy": yaw}
        else:
            for gi in cands.get(stratum, []):
                yield gi, {}

    gens = {s: candidates(s) for s in order}

    def take(stratum: str, k: int) -> int:
        got = 0
        if k <= 0:
            return 0
        for gi, extra in gens[stratum]:
            if gi in chosen or not gate.try_add(filenames[gi]):
                continue
            chosen.add(gi)
            picks.append(gi)
            recs.append({"file": filenames[gi], "stratum": stratum, **extra, **stats.get(gi, {})})
            got += 1
            if got >= k:
                break
        return got

    for s in order:
        take(s, quota.get(s, 0))
    # redistribute any shortfall round-robin over strata that still have candidates
    while len(picks) < n:
        if not sum(take(s, 1) for s in order if len(picks) < n):
            break
    if len(picks) < n:
        raise RuntimeError(f"messy: only {len(picks)}/{n} images found (stats available: {stats_fn is not None}, "
                           f"faces available: {faces is not None})")
    return picks, recs


def build_splits(attr: np.ndarray, attr_names: Sequence[str], filenames: Sequence[str], v1_calib: Sequence[str],
                 v1_eval70: Sequence[str], gate: IdentityGate, seed: int = SEED, sizes: Optional[Dict[str, int]] = None,
                 stats_fn: Optional[Callable[[str], dict]] = None, faces: Optional[FaceCache] = None,
                 pose_anno: Optional[Dict[str, Tuple[float, float, float]]] = None) -> Tuple[Dict[str, List[str]], dict]:
    sizes = dict(sizes or SIZES)
    fidx = {f: i for i, f in enumerate(filenames)}
    for f in list(v1_calib) + list(v1_eval70):
        if f not in fidx:
            raise ValueError(f"v1 image {f} not found in the attribute file")
    if len(set(v1_eval70)) != len(v1_eval70):
        raise ValueError("v1_eval70 has duplicates")
    if set(v1_calib) & set(v1_eval70):
        raise ValueError("v1 calibration and evaluation lists overlap")
    if sizes["test"] < len(v1_eval70):
        raise ValueError("test size smaller than the v1 evaluation list")

    used = set(v1_calib) | set(v1_eval70)
    meta: dict = {"seed": seed, "sizes": sizes, "identity": None, "notes": []}
    clash_calib = sum(bool(gate.register(f)) for f in v1_calib)
    clash_v1 = sum(bool(gate.register(f)) for f in v1_eval70)
    meta["v1_calib_identity_clashes_among_themselves"] = clash_calib
    meta["v1_eval70_identity_clashes"] = clash_v1  # counted against calib + earlier v1 images; kept anyway
    if clash_v1:
        meta["notes"].append(f"{clash_v1} v1 eval images share an identity with another v1/calibration image; kept for pairing.")

    def pool():
        return np.array([i for i, f in enumerate(filenames) if f not in used])

    out: Dict[str, List[str]] = {}
    extra_test = sizes["test"] - len(v1_eval70)
    picks = kmeans_select(attr, pool(), extra_test, seed + 1, gate, filenames) if extra_test else []
    out["test"] = list(v1_eval70) + [filenames[i] for i in picks]
    used |= set(out["test"])

    mp, mrecs = messy_select(attr, attr_names, pool(), filenames, sizes["messy"], seed + 2, gate, stats_fn, faces,
                          pose_anno=pose_anno)
    out["messy"] = [filenames[i] for i in mp]
    used |= set(out["messy"])
    meta["messy"] = {"quota": MESSY_QUOTA, "records": mrecs, "image_stats_used": stats_fn is not None,
                     "pose_source": ("pose_anno" if pose_anno else ("landmark_yaw_proxy" if faces is not None else "none")),
                     "pose_rule": "top 10% of pool by max(|yaw|,|pitch|) (rank-based; units unverified)" if pose_anno else
                     f"|nose offset|/inter-ocular >= {YAW_THRESHOLD}"}
    if stats_fn is None:
        meta["notes"].append("messy: no image statistics available (lighting stratum empty; blur from attribute only).")
    if faces is None and not pose_anno:
        meta["notes"].append("messy: no pose annotation / face detector available (pose stratum empty; quota redistributed).")

    for j, name in enumerate(["val", "train", "ae_pool"]):
        pk = kmeans_select(attr, pool(), sizes[name], seed + 3 + j, gate, filenames)
        out[name] = [filenames[i] for i in pk]
        used |= set(out[name])
    meta["identity"] = gate.describe()
    if gate.mode == "none":
        meta["notes"].append("WARNING: no identity de-duplication was performed; splits may share people.")
    check_splits(out, v1_calib, v1_eval70, sizes)
    return out, meta


def check_splits(splits: Dict[str, List[str]], v1_calib: Sequence[str], v1_eval70: Sequence[str],
                 sizes: Optional[Dict[str, int]] = None) -> None:
    sizes = sizes or SIZES
    seen: Dict[str, str] = {}
    for name, lst in splits.items():
        if len(lst) != sizes[name]:
            raise AssertionError(f"{name}: {len(lst)} != {sizes[name]}")
        if len(set(lst)) != len(lst):
            raise AssertionError(f"{name}: duplicates")
        for f in lst:
            if f in seen:
                raise AssertionError(f"{f} in both {seen[f]} and {name}")
            seen[f] = name
    bad = set(v1_calib) & set(seen)
    if bad:
        raise AssertionError(f"calibration images leaked into splits: {sorted(bad)}")
    if not set(v1_eval70) <= set(splits["test"]):
        raise AssertionError("test must contain all v1 eval images")


# ----------------------------------------------------------------------------------------------- IO / CLI


def sha256_lines(lst: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(lst) + "\n").encode()).hexdigest()


def write_splits(splits: Dict[str, List[str]], meta: dict, out_dir: str, force: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, lst in splits.items():
        p = os.path.join(out_dir, f"{name}.txt")
        txt = "\n".join(lst) + "\n"
        if os.path.exists(p) and not force:
            with open(p, "r", encoding="utf-8") as f:
                if f.read() != txt:
                    raise FileExistsError(f"{p} exists and differs; splits are frozen. Use --force to overwrite deliberately.")
            continue
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
    meta = dict(meta)
    meta["sha256"] = {k: sha256_lines(v) for k, v in splits.items()}
    meta["counts"] = {k: len(v) for k, v in splits.items()}
    mp = os.path.join(out_dir, "splits_meta.json")
    if not (os.path.exists(mp) and not force):
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)


def find_celebahq(root: Optional[str]) -> Tuple[str, str]:
    """-> (image dir, attribute file). Same discovery as v1: any dir holding CelebA-HQ-img + the attribute file."""
    roots = [root] if root else ["/kaggle/input"]
    for r in roots:
        for d, dirs, files in os.walk(r):
            if "CelebA-HQ-img" in dirs and "CelebAMask-HQ-attribute-anno.txt" in files:
                return os.path.join(d, "CelebA-HQ-img"), os.path.join(d, "CelebAMask-HQ-attribute-anno.txt")
    raise FileNotFoundError("CelebA-HQ-img/ + CelebAMask-HQ-attribute-anno.txt not found; pass --celebahq_root")


def find_file(name: str, roots: Sequence[Optional[str]]) -> Optional[str]:
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for d, _, files in os.walk(r):
            if name in files:
                return os.path.join(d, name)
    return None


def make_face_cache(img_dir: str) -> Optional[FaceCache]:
    """ArcFace / landmarks via A3's FaceScorer if importable, else insightface directly; None if unavailable."""
    try:
        from src.eval.faces import FaceScorer
        scorer = FaceScorer(prefer_gpu=False)
        scorer.load()

        def analyze(fname):
            from PIL import Image
            im = Image.open(os.path.join(img_dir, fname)).convert("RGB").resize((512, 512), Image.BICUBIC)
            r = scorer.analyze(im)
            return {"emb": r.emb, "kps": getattr(r, "kps", None)}
        return FaceCache(analyze)
    except Exception as e1:
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=-1, det_size=(640, 640))

            def analyze(fname):
                from PIL import Image
                im = Image.open(os.path.join(img_dir, fname)).convert("RGB").resize((512, 512), Image.BICUBIC)
                faces = app.get(np.asarray(im)[:, :, ::-1].copy())
                if not faces:
                    return {"emb": None, "kps": None}
                f = max(faces, key=lambda z: (z.bbox[2] - z.bbox[0]) * (z.bbox[3] - z.bbox[1]))
                return {"emb": np.asarray(f.normed_embedding), "kps": np.asarray(f.kps)}
            return FaceCache(analyze)
        except Exception as e2:
            print(f"[make_splits] no face analysis available ({e1!r} / {e2!r})", file=sys.stderr)
            return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celebahq_root", default=None, help="dir containing CelebA-HQ-img/ and the attribute file (default: search /kaggle/input)")
    ap.add_argument("--mapping", default=None, help="CelebA-HQ-to-CelebA-mapping.txt (optional)")
    ap.add_argument("--pose_anno", default=None, help="CelebAMask-HQ-pose-anno.txt (auto-discovered next to the attribute file)")
    ap.add_argument("--identity", default=None, help="identity_CelebA.txt (optional)")
    ap.add_argument("--identity_mode", choices=["auto", "labels", "arcface", "none"], default="auto")
    ap.add_argument("--arcface_threshold", type=float, default=ARCFACE_SAME_PERSON_SIM)
    ap.add_argument("--allow_no_dedup", action="store_true")
    ap.add_argument("--v1_calib", default=os.path.join(ROOT, "data", "splits", "v1_calib.txt"))
    ap.add_argument("--v1_eval70", default=os.path.join(ROOT, "data", "splits", "v1_eval70.txt"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "splits"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no_image_stats", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing differing split files (splits are meant to be frozen)")
    a = ap.parse_args(argv)

    img_dir, attr_path = find_celebahq(a.celebahq_root)
    fns, names, attr = parse_attr_file(attr_path)
    print(f"[make_splits] {len(fns)} images, {len(names)} attributes from {attr_path}")
    calib = read_list(a.v1_calib) if os.path.isfile(a.v1_calib) else list(CALIB_DEFAULT)
    if not os.path.isfile(a.v1_eval70):
        print(f"missing v1 eval list {a.v1_eval70}; pass --v1_eval70 (read-only copy of evaluation_70_manifest Filename column)", file=sys.stderr)
        return 2
    v1 = read_list(a.v1_eval70)

    search = [a.celebahq_root, os.path.dirname(attr_path), "/kaggle/input"]
    mapping = a.mapping or find_file("CelebA-HQ-to-CelebA-mapping.txt", search)
    ident = a.identity or find_file("identity_CelebA.txt", search)
    pose_path = a.pose_anno or find_file("CelebAMask-HQ-pose-anno.txt", search)
    pose_anno = parse_pose_anno(pose_path) if pose_path else None
    print(f"[make_splits] pose annotation: {pose_path} ({len(pose_anno) if pose_anno else 0} rows)")
    mode = a.identity_mode
    faces = None
    if mode == "auto":
        mode = "labels" if (mapping and ident) else "arcface"
    if mode == "labels":
        if not (mapping and ident):
            print("identity_mode=labels needs --mapping and --identity", file=sys.stderr)
            return 2
        labels = parse_hq_identity_map(mapping, ident)
        gate: IdentityGate = LabelIdentity(labels)
        print(f"[make_splits] true identity labels for {len(labels)} HQ images")
    elif mode == "arcface":
        faces = make_face_cache(img_dir)
        if faces is None:
            if not a.allow_no_dedup:
                print("ArcFace dedup unavailable (insightface not importable) and no identity labels; "
                      "install insightface or pass --allow_no_dedup", file=sys.stderr)
                return 2
            gate = IdentityGate()
        else:
            gate = ArcFaceIdentity(faces, a.arcface_threshold)
    else:
        if not a.allow_no_dedup:
            print("--identity_mode none requires --allow_no_dedup", file=sys.stderr)
            return 2
        gate = IdentityGate()

    stats_fn = None if a.no_image_stats else make_image_stats_fn(img_dir)
    if faces is None and mode == "labels" and not pose_anno:
        faces = make_face_cache(img_dir)  # optional fallback for the pose stratum of messy
    splits, meta = build_splits(attr, names, fns, calib, v1, gate, a.seed, SIZES, stats_fn, faces, pose_anno)
    meta.update({"celebahq_attr_file": attr_path, "n_images_total": len(fns), "v1_calib": calib,
                 "v1_eval70_file": a.v1_eval70, "argv": sys.argv[1:]})
    write_splits(splits, meta, a.out_dir, a.force)
    print("[make_splits] wrote", {k: len(v) for k, v in splits.items()}, "->", a.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
