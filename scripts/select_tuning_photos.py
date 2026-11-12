#!/usr/bin/env python
"""Pick representative tuning photos with MMD-critic (CPU; reads the npz written by scripts/embed_pool.py).

Method: MMD-critic, B. Kim, R. Khanna, O. Koyejo, "Examples are not enough, learn to criticize! Criticism for
interpretability", NeurIPS 2016 (reference code: github.com/BeenKim/MMD-critic). Re-implemented in plain numpy:
  * kernel k(a,b) = exp(-||a-b||^2 / (2 s^2)), s = median pairwise distance of ``--bandwidth_n`` random pool images;
  * prototypes: greedy maximisation of J(S) = 2/(n|S|) sum_{i in S, j in P} k(i,j) - 1/|S|^2 sum_{i,j in S} k(i,j)
    (= greedy minimisation of MMD^2(S, P) up to a constant); column means over the pool are computed once, chunked;
  * criticisms: greedy maximisation of sum_{l in C} |witness(x_l)| + w * log det K_CC (the paper's LogDet diversity
    regulariser, weight ``--crit_reg``, default 1), witness(x) = mean_j k(x,j) - mean_{s in S} k(x,s).
Selection space: CLIP ViT-L/14 image embeddings (npz ``clip``). Independent check space (never used for selection):
40 attributes + pose, standardised.

Pool P = all npz images with an ArcFace face, minus the names in test / messy / ae_pool / v1_calib, minus any image whose
ArcFace cosine to one of those is >= ``--id_thr`` (0.5, the project's UNVERIFIED same-person value). A candidate is
skipped if its ArcFace cosine to an already selected photo is >= ``--id_thr`` (one photo per identity).
Sets: tune_search = ``--n_search`` prototypes on P; tune_confirm = ``--n_confirm_protos`` prototypes of an independent
greedy run (candidates = P minus the search identities; target distribution still P) + ``--n_confirm_crit`` criticisms
of the SEARCH set (identity-gated against everything selected).

Outputs: ``tune_search.txt``, ``tune_confirm.txt``, ``tune_meta.json`` in ``--out_splits_dir`` (refuses to overwrite
without ``--force``) and ``selection_report.json`` / ``selection_report.md`` in ``--report_dir`` (numbers only):
MMD^2 to P vs ``--n_random`` random same-size subsets in CLIP space and in attribute+pose space, rank-based
p = (1 + #{random <= ours}) / (n_random + 1), nearest-selected coverage vs random, attribute marginals. The old
``val.txt`` list is evaluated for context. All randomness is seeded (``--seed``, 2026).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ID_THR = 0.5   # UNVERIFIED project value (make_splits.ARCFACE_SAME_PERSON_SIM)
EXCLUDE_LISTS = ("test", "messy", "ae_pool", "v1_calib")


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


# ------------------------------------------------------------------------------------------------ kernel maths


def _sqnorm(X: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", X, X)


def rbf(A: np.ndarray, B: np.ndarray, s: float) -> np.ndarray:
    """RBF kernel matrix k(A_i, B_j), float32 compute."""
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    d2 = _sqnorm(A)[:, None] + _sqnorm(B)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-d2 / (2.0 * float(s) ** 2))


def median_bandwidth(X: np.ndarray, n: int, rng: np.random.Generator) -> float:
    """Median heuristic: median pairwise Euclidean distance among ``n`` random rows (distinct pairs)."""
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Z = np.asarray(X[idx], np.float32)
    d2 = _sqnorm(Z)[:, None] + _sqnorm(Z)[None, :] - 2.0 * (Z @ Z.T)
    iu = np.triu_indices(len(Z), k=1)
    d = np.sqrt(np.maximum(d2[iu], 0.0))
    med = float(np.median(d))
    return med if med > 0 else 1.0


def colmeans(Xc: np.ndarray, Xp: np.ndarray, s: float, chunk: int = 1024) -> np.ndarray:
    """mean_j k(Xc_i, Xp_j) for every row of Xc, in row chunks (never holds more than chunk x |Xp| floats)."""
    out = np.empty(len(Xc), np.float64)
    for a in range(0, len(Xc), chunk):
        out[a:a + chunk] = rbf(Xc[a:a + chunk], Xp, s).mean(axis=1, dtype=np.float64)
    return out


def mmd2(X: np.ndarray, idx: Sequence[int], colm: np.ndarray, pp_mean: float, s: float) -> float:
    """MMD^2(S, P) = mean_SS k - 2 mean_SP k + mean_PP k, with colm = colmeans(X, X_P) and pp_mean = mean over P of colm."""
    idx = np.asarray(idx, int)
    return float(rbf(X[idx], X[idx], s).mean(dtype=np.float64) - 2.0 * colm[idx].mean() + pp_mean)


# ------------------------------------------------------------------------------------------------ MMD-critic


def greedy_prototypes(X: np.ndarray, colm: np.ndarray, s: float, k: int, allowed: np.ndarray,
                      arc: Optional[np.ndarray] = None, thr: float = ID_THR):
    """Greedy MMD-critic prototypes. X [n,d] pool features, colm [n] = mean_j k(x_i, x_j) over the pool,
    allowed [n] bool candidates, arc [n,512] L2-normalised ArcFace for the identity gate.
    Returns (local indices in selection order, J after each step). Ties -> lowest index (deterministic)."""
    allowed = allowed.copy()
    n = len(X)
    sumK = np.zeros(n)
    cm_sum, ss = 0.0, 0.0
    sel: List[int] = []
    trace: List[float] = []
    for m in range(1, k + 1):
        cand = np.flatnonzero(allowed)
        if len(cand) == 0:
            break
        ss_new = ss + 2.0 * sumK[cand] + 1.0                     # k(c,c) = 1
        J = 2.0 * (cm_sum + colm[cand]) / m - ss_new / m ** 2
        b = int(np.argmax(J))
        c = int(cand[b])
        ss, cm_sum = float(ss_new[b]), cm_sum + float(colm[c])
        sumK += rbf(X, X[c:c + 1], s)[:, 0]
        sel.append(c)
        trace.append(float(J[b]))
        allowed[c] = False
        if arc is not None:
            allowed &= (arc @ arc[c]) < thr
    return sel, trace


def greedy_criticisms(X: np.ndarray, colm: np.ndarray, s: float, protos: Sequence[int], k: int, allowed: np.ndarray,
                      arc: Optional[np.ndarray] = None, thr: float = ID_THR, reg: float = 1.0, jitter: float = 1e-6):
    """Greedy criticisms of the prototype set ``protos`` (local indices): maximise |witness| + reg * logdet gain,
    where the log-det gain of adding c to C is log(1 - k_Cc^T K_CC^-1 k_Cc) (k(c,c)=1; Schur complement)."""
    allowed = allowed.copy()
    Kxp = rbf(X, X[list(protos)], s)
    absw = np.abs(colm - Kxp.mean(axis=1, dtype=np.float64))
    sel: List[int] = []
    trace: List[dict] = []
    for _ in range(k):
        cand = np.flatnonzero(allowed)
        if len(cand) == 0:
            break
        if sel:
            Kxc = rbf(X[cand], X[sel], s).astype(np.float64)     # [cand, |C|]
            Kcc = rbf(X[sel], X[sel], s).astype(np.float64) + jitter * np.eye(len(sel))
            q = np.einsum("ij,ij->i", Kxc, np.linalg.solve(Kcc, Kxc.T).T)
            gain = np.log(np.clip(1.0 - q, 1e-12, None))
        else:
            gain = np.zeros(len(cand))
        score = absw[cand] + reg * gain
        b = int(np.argmax(score))
        c = int(cand[b])
        sel.append(c)
        trace.append({"abs_witness": float(absw[c]), "logdet_gain": float(gain[b])})
        allowed[c] = False
        if arc is not None:
            allowed &= (arc @ arc[c]) < thr
    return sel, trace


# ------------------------------------------------------------------------------------------------ evidence


def attr_pose_features(attr: np.ndarray, pose: np.ndarray, pool_idx: np.ndarray) -> np.ndarray:
    """Independent space: [N, 40 + 3] standardised with pool statistics; missing pose -> 0 (= pool mean)."""
    A = np.asarray(attr, np.float64)
    mu, sd = A[pool_idx].mean(0), A[pool_idx].std(0)
    A = (A - mu) / np.where(sd > 0, sd, 1.0)
    P = np.asarray(pose, np.float64)
    pmu, psd = np.nanmean(P[pool_idx], 0), np.nanstd(P[pool_idx], 0)
    P = (P - pmu) / np.where(psd > 0, psd, 1.0)
    P = np.nan_to_num(P, nan=0.0)
    return np.concatenate([A, P], axis=1).astype(np.float32)


def _pct(a, q):
    return float(np.percentile(a, q))


def random_baseline(ours: float, rand: np.ndarray) -> dict:
    n = len(rand)
    return {"ours": float(ours), "random_mean": float(np.mean(rand)), "random_p05": _pct(rand, 5),
            "random_p50": _pct(rand, 50), "random_p95": _pct(rand, 95), "n_random": int(n),
            "p_rank": float((1 + int(np.sum(rand <= ours))) / (n + 1))}


def mmd_vs_random(X: np.ndarray, colm: np.ndarray, pp_mean: float, s: float, idx: Sequence[int],
                  pool_idx: np.ndarray, n_random: int, seed: int) -> dict:
    m = len(idx)
    rng = np.random.default_rng([seed, m])
    rand = np.empty(n_random)
    for r in range(n_random):
        rand[r] = mmd2(X, rng.choice(pool_idx, size=m, replace=False), colm, pp_mean, s)
    return random_baseline(mmd2(X, idx, colm, pp_mean, s), rand)


def _nearest_dist(Xp: np.ndarray, Xs: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """1 - max cosine to the selected rows (rows are L2-normalised CLIP embeddings)."""
    out = np.empty(len(Xp), np.float32)
    Xs = np.asarray(Xs, np.float32)
    for a in range(0, len(Xp), chunk):
        out[a:a + chunk] = 1.0 - (np.asarray(Xp[a:a + chunk], np.float32) @ Xs.T).max(axis=1)
    return out


def coverage_vs_random(Xclip: np.ndarray, idx: Sequence[int], pool_idx: np.ndarray, n_random: int, seed: int) -> dict:
    Xp = Xclip[pool_idx]
    d = _nearest_dist(Xp, Xclip[np.asarray(idx, int)])
    m = len(idx)
    rng = np.random.default_rng([seed, m, 1])
    rm, rx = np.empty(n_random), np.empty(n_random)
    for r in range(n_random):
        dr = _nearest_dist(Xp, Xclip[rng.choice(pool_idx, size=m, replace=False)])
        rm[r], rx[r] = dr.mean(), dr.max()
    return {"mean_dist": random_baseline(d.mean(), rm), "max_dist": random_baseline(d.max(), rx)}


def attr_marginals(attr: np.ndarray, pose: np.ndarray, attr_names: Sequence[str], idx: Sequence[int],
                   pool_idx: np.ndarray) -> dict:
    idx = np.asarray(idx, int)
    pool_share, set_share = attr[pool_idx].mean(0), attr[idx].mean(0)
    pm, ps = np.nanmean(pose[pool_idx], 0), np.nanstd(pose[pool_idx], 0)
    sm, ss_ = np.nanmean(pose[idx], 0), np.nanstd(pose[idx], 0)
    return {"attributes": {str(n): {"pool": float(a), "set": float(b)} for n, a, b in zip(attr_names, pool_share, set_share)},
            "pose": {n: {"pool_mean": float(a), "set_mean": float(b), "pool_std": float(c), "set_std": float(d)}
                     for n, a, b, c, d in zip(("yaw", "pitch", "roll"), pm, sm, ps, ss_)}}


# ------------------------------------------------------------------------------------------------ pipeline


def build_pool(names: Sequence[str], arc: np.ndarray, has_face: np.ndarray, exclude_names: Sequence[str],
               thr: float = ID_THR):
    """-> (pool global indices, counts dict). Order follows the npz order."""
    name_to_i = {n: i for i, n in enumerate(names)}
    has = np.asarray(has_face, bool) & np.isfinite(arc).all(axis=1)
    ex_idx = sorted({name_to_i[n] for n in exclude_names if n in name_to_i})
    in_pool = has.copy()
    for i in ex_idx:
        in_pool[i] = False
    ref = [i for i in ex_idx if has[i]]
    leak = np.zeros(len(names), bool)
    if ref:
        safe = np.nan_to_num(np.asarray(arc, np.float32))          # no-face rows are NaN: they never leak
        R = safe[ref]
        for a in range(0, len(names), 4096):
            leak[a:a + 4096] = (safe[a:a + 4096] @ R.T).max(axis=1) >= thr
    n_leak = int((in_pool & leak).sum())
    in_pool &= ~leak
    counts = {"n_npz": len(names), "n_no_face": int((~has).sum()), "n_excluded_names_in_npz": len(ex_idx),
              "n_excluded_names_missing_from_npz": len(set(exclude_names) - set(names)),
              "n_identity_leak_removed": n_leak, "n_pool": int(in_pool.sum())}
    return np.flatnonzero(in_pool), counts


def run_selection(data: dict, exclude_names: Sequence[str], old_val: Sequence[str], n_search: int = 28,
                  n_confirm_protos: int = 8, n_confirm_crit: int = 4, seed: int = 2026, n_random: int = 1000,
                  n_cov_random: int = 200, bandwidth_n: int = 5000, id_thr: float = ID_THR, crit_reg: float = 1.0,
                  chunk: int = 1024, log=print) -> dict:
    names = [str(n) for n in data["names"]]
    clip = np.asarray(data["clip"], np.float32)
    clip /= np.clip(np.linalg.norm(clip, axis=1, keepdims=True), 1e-12, None)
    arc_raw = np.asarray(data["arcface"], np.float32)
    has = np.asarray(data["has_face"], bool)
    arc = arc_raw / np.clip(np.linalg.norm(arc_raw, axis=1, keepdims=True), 1e-12, None)
    attr = np.asarray(data["attr"], np.float32)
    attr_names = [str(a) for a in data["attr_names"]]
    pose = np.asarray(data["pose"], np.float32)

    pool_idx, counts = build_pool(names, arc, has, exclude_names, id_thr)
    n = len(pool_idx)
    log(f"[select] pool {n} of {len(names)}: {counts}")
    if n < n_search + n_confirm_protos + n_confirm_crit:
        raise ValueError(f"pool of {n} is too small for the requested set sizes")
    Xp, arcp = clip[pool_idx], arc[pool_idx]

    rng = np.random.default_rng(seed)
    s = median_bandwidth(Xp, bandwidth_n, rng)
    log(f"[select] CLIP bandwidth {s:.6f}; computing kernel column means")
    colm_all = colmeans(clip, Xp, s, chunk)                       # all npz rows vs pool (rows of val may lie outside P)
    colm_p = colm_all[pool_idx]
    pp_clip = float(colm_p.mean())

    # prototypes: search set
    s_loc, s_trace = greedy_prototypes(Xp, colm_p, s, n_search, np.ones(n, bool), arcp, id_thr)
    # confirm prototypes: independent greedy run, candidates = pool minus search identities (target distribution still P)
    allowed = np.ones(n, bool)
    for c in s_loc:
        allowed &= (arcp @ arcp[c]) < id_thr
    allowed[s_loc] = False
    n_removed = int(n - allowed.sum())
    cp_loc, cp_trace = greedy_prototypes(Xp, colm_p, s, n_confirm_protos, allowed, arcp, id_thr)
    for c in cp_loc:
        allowed &= (arcp @ arcp[c]) < id_thr
    allowed[cp_loc] = False
    cc_loc, cc_trace = greedy_criticisms(Xp, colm_p, s, s_loc, n_confirm_crit, allowed, arcp, id_thr, crit_reg)

    G = lambda loc: [int(pool_idx[i]) for i in loc]                # local -> global rows
    sel_search, sel_cprot, sel_ccrit = G(s_loc), G(cp_loc), G(cc_loc)
    sets: Dict[str, List[int]] = {"tune_search": sel_search, "tune_confirm_prototypes": sel_cprot,
                                  "tune_confirm": sel_cprot + sel_ccrit}
    name_to_i = {nm: i for i, nm in enumerate(names)}
    val_idx = [name_to_i[v] for v in old_val if v in name_to_i]
    val_missing = [v for v in old_val if v not in name_to_i]
    if val_idx:
        sets["old_val"] = val_idx

    # independent space
    F = attr_pose_features(attr, pose, pool_idx)
    Fp = F[pool_idx]
    sF = median_bandwidth(Fp, bandwidth_n, np.random.default_rng(seed + 1))
    colm_F = colmeans(F, Fp, sF, chunk)
    pp_F = float(colm_F[pool_idx].mean())

    evidence = {}
    for nm, idx in sets.items():
        log(f"[select] evidence for {nm} ({len(idx)} photos)")
        evidence[nm] = {
            "n": len(idx),
            "mmd2_clip": mmd_vs_random(clip, colm_all, pp_clip, s, idx, pool_idx, n_random, seed),
            "mmd2_attr_pose": mmd_vs_random(F, colm_F, pp_F, sF, idx, pool_idx, n_random, seed),
            "coverage_clip": coverage_vs_random(clip, idx, pool_idx, n_cov_random, seed),
            "marginals": attr_marginals(attr, pose, attr_names, idx, pool_idx),
        }
    # identity sanity: max pairwise ArcFace cosine inside / across the selected sets
    sel_all = sel_search + sel_cprot + sel_ccrit
    A = arc[sel_all]
    G_ = A @ A.T
    np.fill_diagonal(G_, -1.0)
    return {
        "counts": counts, "bandwidth_clip": s, "bandwidth_attr_pose": sF, "n_confirm_candidates_removed_by_search_identity": n_removed,
        "tune_search": [names[i] for i in sel_search],
        "tune_confirm": [names[i] for i in sel_cprot + sel_ccrit],
        "confirm_prototypes": [names[i] for i in sel_cprot], "confirm_criticisms": [names[i] for i in sel_ccrit],
        "traces": {"search_J": s_trace, "confirm_proto_J": cp_trace, "confirm_criticisms": cc_trace},
        "max_pairwise_arcface_cosine_among_selected": float(G_.max()),
        "old_val_missing_from_npz": val_missing, "evidence": evidence, "attr_names": attr_names,
        "n_random": n_random, "n_cov_random": n_cov_random,
    }


def _fmt(x, p=5):
    return f"{x:.{p}g}"


def report_markdown(res: dict, meta: dict) -> str:
    L = ["# Tuning photo selection report", "",
         "Numbers only. MMD^2 is to the pool P (lower = more representative). p = (1 + #{random <= ours}) / (n_random + 1).",
         "CLIP space is the optimised space (an extreme p is expected); attribute+pose space is the independent check.", "",
         f"Pool: {json.dumps(res['counts'])}", f"Bandwidths: CLIP {_fmt(res['bandwidth_clip'])}, attribute+pose {_fmt(res['bandwidth_attr_pose'])}",
         f"Random subsets: {res['n_random']} (MMD), {res['n_cov_random']} (coverage). Seed {meta['args']['seed']}.", "",
         "## MMD^2 vs random same-size subsets", "",
         "| set | n | space | ours | rand mean | rand p05 | rand p50 | rand p95 | p |", "|---|---|---|---|---|---|---|---|---|"]
    for nm, ev in res["evidence"].items():
        for sp in ("mmd2_clip", "mmd2_attr_pose"):
            b = ev[sp]
            L.append(f"| {nm} | {ev['n']} | {sp[5:]} | {_fmt(b['ours'])} | {_fmt(b['random_mean'])} | {_fmt(b['random_p05'])} | "
                     f"{_fmt(b['random_p50'])} | {_fmt(b['random_p95'])} | {_fmt(b['p_rank'])} |")
    L += ["", "## Coverage: CLIP cosine distance from each pool image to the nearest selected photo", "",
          "| set | stat | ours | rand mean | rand p05 | rand p50 | rand p95 | p |", "|---|---|---|---|---|---|---|---|"]
    for nm, ev in res["evidence"].items():
        for st in ("mean_dist", "max_dist"):
            b = ev["coverage_clip"][st]
            L.append(f"| {nm} | {st} | {_fmt(b['ours'])} | {_fmt(b['random_mean'])} | {_fmt(b['random_p05'])} | "
                     f"{_fmt(b['random_p50'])} | {_fmt(b['random_p95'])} | {_fmt(b['p_rank'])} |")
    names = list(res["evidence"])
    L += ["", "## Attribute shares (pool vs set)", "", "| attribute | pool | " + " | ".join(names) + " |",
          "|---|---|" + "---|" * len(names)]
    for a in res["attr_names"]:
        row = [_fmt(res["evidence"][names[0]]["marginals"]["attributes"][a]["pool"], 3)]
        row += [_fmt(res["evidence"][nm]["marginals"]["attributes"][a]["set"], 3) for nm in names]
        L.append(f"| {a} | " + " | ".join(row) + " |")
    L += ["", "## Pose (mean / std; raw annotation units, unverified)", "", "| axis | pool | " + " | ".join(names) + " |",
          "|---|---|" + "---|" * len(names)]
    for ax in ("yaw", "pitch", "roll"):
        p0 = res["evidence"][names[0]]["marginals"]["pose"][ax]
        row = [f"{_fmt(p0['pool_mean'], 3)} / {_fmt(p0['pool_std'], 3)}"]
        row += [f"{_fmt(res['evidence'][nm]['marginals']['pose'][ax]['set_mean'], 3)} / "
                f"{_fmt(res['evidence'][nm]['marginals']['pose'][ax]['set_std'], 3)}" for nm in names]
        L.append(f"| {ax} | " + " | ".join(row) + " |")
    L += ["", f"Max pairwise ArcFace cosine among all selected photos: {_fmt(res['max_pairwise_arcface_cosine_among_selected'])}",
          f"Confirm criticisms: {', '.join(res['confirm_criticisms'])}", ""]
    if res["old_val_missing_from_npz"]:
        L.append(f"Old val names missing from the npz: {len(res['old_val_missing_from_npz'])}")
    return "\n".join(L) + "\n"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_lines(lst: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(lst) + "\n").encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True, help="pool_embeddings.npz from scripts/embed_pool.py")
    ap.add_argument("--splits_dir", default=os.path.join(ROOT, "data", "splits"), help="holds test/messy/ae_pool/v1_calib/val lists")
    ap.add_argument("--out_splits_dir", default=None, help="where tune_*.txt / tune_meta.json go (default --splits_dir)")
    ap.add_argument("--report_dir", default=os.path.join(ROOT, "reports", "step1_selection"))
    ap.add_argument("--n_search", type=int, default=28)
    ap.add_argument("--n_confirm_protos", type=int, default=8)
    ap.add_argument("--n_confirm_crit", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n_random", type=int, default=1000, help="random subsets for the MMD baselines")
    ap.add_argument("--n_cov_random", type=int, default=200, help="random subsets for the coverage baselines")
    ap.add_argument("--bandwidth_n", type=int, default=5000)
    ap.add_argument("--id_thr", type=float, default=ID_THR, help="ArcFace same-person cosine (UNVERIFIED 0.5)")
    ap.add_argument("--crit_reg", type=float, default=1.0, help="weight of the log-det regulariser for criticisms")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--force", action="store_true", help="overwrite existing tune_*.txt / tune_meta.json")
    args = ap.parse_args(argv)

    out_dir = args.out_splits_dir or args.splits_dir
    targets = [os.path.join(out_dir, f) for f in ("tune_search.txt", "tune_confirm.txt", "tune_meta.json")]
    existing = [t for t in targets if os.path.exists(t)]
    if existing and not args.force:
        print(f"refusing to overwrite {existing}; use --force", file=sys.stderr)
        return 2

    exclude: List[str] = []
    for nm in EXCLUDE_LISTS:
        p = os.path.join(args.splits_dir, f"{nm}.txt")
        if not os.path.exists(p):
            print(f"missing exclusion list {p}", file=sys.stderr)
            return 2
        exclude += read_list(p)
    val_path = os.path.join(args.splits_dir, "val.txt")
    old_val = read_list(val_path) if os.path.exists(val_path) else []

    z = np.load(args.npz, allow_pickle=False)
    data = {k: z[k] for k in ("names", "clip", "arcface", "has_face", "attr", "attr_names", "pose")}
    res = run_selection(data, exclude, old_val, args.n_search, args.n_confirm_protos, args.n_confirm_crit, args.seed,
                        args.n_random, args.n_cov_random, args.bandwidth_n, args.id_thr, args.crit_reg, args.chunk)

    meta = {"args": vars(args), "npz_sha256": sha256_file(args.npz), "pool": res["counts"],
            "identity_threshold": args.id_thr, "identity_threshold_verified": False,
            "bandwidth_clip": res["bandwidth_clip"], "bandwidth_attr_pose": res["bandwidth_attr_pose"],
            "selection_order": {"tune_search": res["tune_search"], "tune_confirm": res["tune_confirm"]},
            "confirm_prototypes": res["confirm_prototypes"], "confirm_criticisms": res["confirm_criticisms"],
            "sha256": {"tune_search": sha256_lines(res["tune_search"]), "tune_confirm": sha256_lines(res["tune_confirm"])},
            "counts": {"tune_search": len(res["tune_search"]), "tune_confirm": len(res["tune_confirm"])},
            "traces": res["traces"],
            "reference": "Kim, Khanna, Koyejo, NeurIPS 2016 (MMD-critic)"}
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    for nm in ("tune_search", "tune_confirm"):
        with open(os.path.join(out_dir, f"{nm}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(res[nm]) + "\n")
    with open(os.path.join(out_dir, "tune_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(args.report_dir, "selection_report.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {k: v for k, v in meta.items() if k != "traces"}, **{k: v for k, v in res.items() if k not in ("tune_search", "tune_confirm")}},
                  f, indent=1)
    with open(os.path.join(args.report_dir, "selection_report.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(report_markdown(res, meta))
    print(f"[select] wrote lists + meta to {out_dir}, report to {args.report_dir}")
    print(f"[select] tune_search {len(res['tune_search'])}, tune_confirm {len(res['tune_confirm'])} "
          f"({len(res['confirm_criticisms'])} criticisms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
