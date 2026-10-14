"""Analysis layer: results CSVs -> numbers, tables and gate verdicts (CPU only; pandas/numpy/scipy).

Owner: agent A5. Half B only (never imports src.protect). Works on the ``edits.csv`` / ``protect.csv`` written by
``scripts/ingest_results.py`` (schema = docs/V2_PLAN.md section 4.1 = ``src.eval.schema``). Nothing here invents a
number: a cell with no rows is reported as ``not run`` and a statistic that cannot be computed is NaN.

Statistical conventions (handoff "Honesty and statistics")
  * unit = image: every metric is averaged over edit seeds and prompts WITHIN an image first;
  * mean +- 95% percentile-bootstrap CI over images (10,000 resamples, fixed seed 2026; identical to
    ``src.eval.stats.bootstrap_ci`` for the same input);
  * paired differences vs another method are aligned BY ``image_id`` (inner join), CI from a paired image bootstrap,
    plus a two-sided Wilcoxon signed-rank p (uncorrected for multiplicity);
  * PSR hit = no face found in the edit (counted as success, also reported as the no-face rate) or
    ArcFace sim(source, edit) < threshold; thresholds = 95th percentile of the impostor ArcFace similarity of the
    E0 clean edits (per editor, plus pooled).

Sections: constants, csv io and de-duplication, LPIPS cache, bootstrap, thresholds, ``Analysis`` (per-image frame,
cells, paired tests), invisibility, gates (E3/E4/E5/E6 + invisibility), coverage, table writers, template
rendering, LPIPS workers, synthetic data (for tests and dry runs; always labelled SYNTHETIC).
"""
from __future__ import annotations

import csv
import fnmatch
import io
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .schema import EDIT_COLUMNS, PROTECT_COLUMNS

# ------------------------------------------------------------------------------------------------ constants
KEY_COLS: Tuple[str, ...] = ("method", "image_id", "seed", "editor", "prompt_id", "strength", "purifier")
PROTECT_KEY: Tuple[str, ...] = ("run_id", "method", "image_id", "split")
REQUIRED_EDIT = KEY_COLS + ("split", "face_detected", "arcface_sim")
REQUIRED_PROTECT = ("method", "image_id", "png_psnr", "png_ssim")
NUM_EDIT = ("seed", "eps_255", "face_detected", "n_faces", "det_score", "arcface_sim", "fr2_sim", "edit_lpips",
            "clip_i", "inv_psnr", "inv_ssim", "inv_lpips", "gpu_seconds_edit", "gpu_seconds_purify")
NUM_PROTECT = ("eps_255", "step_255", "iters", "v1_psnr", "v1_ssim", "v1_lpips", "v1_linf_model", "png_psnr",
               "png_ssim", "png_lpips", "png_maxabs_u8", "png_frac_at_bound", "gpu_seconds", "peak_mem_mb")

HEAD_EDITOR = "sd15_img2img"
HEAD_STRENGTH = "0.55"
DEFAULT_STRENGTH = {"sd15_img2img": "0.55", "sdxl_img2img": "0.55", "ip2p": "", "sd_inpaint": "1",
                    "ipadapter_faceid": "0.55"}
EDITOR_ORDER = ["sd15_img2img", "sdxl_img2img", "ip2p", "sd_inpaint", "ipadapter_faceid"]
PURIFIER_ORDER = ["none", "jpeg_q90", "jpeg_q75", "jpeg_q50", "jpeg_other", "downup_0.5", "downup_other", "vae_rt",
                  "diffpure_0.05", "diffpure_0.1", "diffpure_0.2", "diffpure_0.3", "restore_swin2sr", "gridpure",
                  "adaptive_ae"]
NOT_RUN = "not run"
PENDING = "PENDING"
SYNTH_PREFIX = "SYNTHETIC"


def canon_strength(s) -> str:
    """Canonical string of the strength key part ('' = editor has none). Matches edit_runner.fmt_strength."""
    if s is None:
        return ""
    if isinstance(s, float) and math.isnan(s):
        return ""
    s = str(s).strip()
    if s == "" or s.lower() in ("nan", "none", "na"):
        return ""
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip().lower() in ("", "nan"))


# ------------------------------------------------------------------------------------- csv io and schema
def classify_columns(cols: Sequence[str]) -> str:
    """'edit' | 'protect' | 'unknown' from a CSV header."""
    c = set(cols)
    if {"arcface_sim", "editor", "face_detected"} <= c:
        return "edit"
    if {"png_psnr", "png_ssim", "method", "image_id"} <= c:
        return "protect"
    return "unknown"


def validate_schema(cols: Sequence[str], kind: str) -> List[str]:
    need = REQUIRED_EDIT if kind == "edit" else REQUIRED_PROTECT
    return [c for c in need if c not in set(cols)]


def read_results_csv(path) -> pd.DataFrame:
    """All columns as strings, no NA parsing (keeps keys byte-exact; numbers are coerced later)."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def typed_edits(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in EDIT_COLUMNS:
        if c not in d:
            d[c] = "" if c not in NUM_EDIT else np.nan
    for c in NUM_EDIT:
        d[c] = _num(d[c])
    d["strength"] = d["strength"].map(canon_strength)
    for c in ("method", "image_id", "split", "editor", "prompt_id", "purifier"):
        d[c] = d[c].astype(str)
    return d


def typed_protect(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in PROTECT_COLUMNS:
        if c not in d:
            d[c] = "" if c not in NUM_PROTECT else np.nan
    for c in NUM_PROTECT:
        d[c] = _num(d[c])
    for c in ("method", "image_id", "split", "run_id"):
        d[c] = d[c].astype(str)
    return d


def dedup_edits(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """De-duplicate string-typed edit rows by the row key (method, image_id, seed, editor, prompt_id, strength,
    purifier). Preference: a row with a finite ``edit_lpips``, then the latest ``created_utc``, then the later row.
    Returns (rows sorted by key, stats). ``conflicts`` counts duplicate keys whose arcface_sim differ by > 1e-4
    (GPU non-determinism; the preferred row is kept)."""
    d = df.copy()
    d["strength"] = d["strength"].map(canon_strength)
    key = list(KEY_COLS)
    n0 = len(d)
    if "created_utc" not in d:
        d["created_utc"] = ""
    d["_o_l"] = _num(d["edit_lpips"] if "edit_lpips" in d else pd.Series(np.nan, index=d.index)).notna().astype(int)
    d["_o_t"] = d["created_utc"].astype(str)
    d["_o_i"] = np.arange(len(d))
    d = d.sort_values(["_o_l", "_o_t", "_o_i"], kind="stable")
    dup = d.duplicated(key, keep=False)
    conflicts = 0
    if dup.any():
        sims = _num(d.loc[dup, "arcface_sim"])
        g = pd.DataFrame({**{k: d.loc[dup, k] for k in key}, "s": sims}).groupby(key)["s"]
        spread = (g.max() - g.min()).dropna()
        conflicts = int((spread > 1e-4).sum())
    kept = d.drop_duplicates(key, keep="last").drop(columns=["_o_l", "_o_t", "_o_i"])
    kept = kept.sort_values(["method", "purifier", "editor", "strength", "prompt_id", "image_id", "seed"],
                            kind="stable").reset_index(drop=True)
    return kept, {"rows_in": n0, "rows_out": len(kept), "duplicates_dropped": n0 - len(kept), "conflicts": conflicts}


def dedup_protect(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    d = df.copy()
    n0 = len(d)
    if "created_utc" not in d:
        d["created_utc"] = ""
    d["_o_i"] = np.arange(len(d))
    d = d.sort_values(["created_utc", "_o_i"], kind="stable")
    kept = d.drop_duplicates([k for k in PROTECT_KEY if k in d], keep="last").drop(columns=["_o_i"])
    kept = kept.sort_values(["method", "image_id"], kind="stable").reset_index(drop=True)
    return kept, {"rows_in": n0, "rows_out": len(kept), "duplicates_dropped": n0 - len(kept)}


# ------------------------------------------------------------------------------------------- LPIPS cache
LPIPS_CACHE_COLS = list(KEY_COLS) + ["edit_lpips"]


def load_lpips_cache(path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame(columns=LPIPS_CACHE_COLS)
    d = pd.read_csv(p, dtype=str, keep_default_na=False)
    d["strength"] = d["strength"].map(canon_strength)
    return d[LPIPS_CACHE_COLS].drop_duplicates(list(KEY_COLS), keep="last")


def apply_lpips_cache(df: pd.DataFrame, cache: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Fill missing ``edit_lpips`` of string-typed rows from the cache (key match). Returns (df, n_filled)."""
    if len(cache) == 0 or len(df) == 0:
        return df, 0
    d = df.copy()
    d["strength"] = d["strength"].map(canon_strength)
    c = cache.rename(columns={"edit_lpips": "_cache_lpips"})
    m = d.merge(c, on=list(KEY_COLS), how="left")
    miss = _num(m["edit_lpips"]).isna()
    have = m["_cache_lpips"].notna() & (_num(m["_cache_lpips"]).notna())
    fill = miss & have
    m.loc[fill, "edit_lpips"] = m.loc[fill, "_cache_lpips"]
    return m.drop(columns=["_cache_lpips"]), int(fill.sum())


def edit_rel_path(method, purifier, editor, prompt_id, strength, seed, image_id) -> str:
    """Relative path of a saved edit under <root>/edits/ (see EditRunner.edit_save_path)."""
    return f"{method}/{purifier}/{editor}/{prompt_id}_s{canon_strength(strength) or 'na'}_seed{seed}/{image_id}"


def clean_rel_path(editor, prompt_id, strength, seed, image_id) -> str:
    """Relative path of an E0 clean edit under clean_edits/ (see CleanEditCache.path)."""
    return f"{editor}/{prompt_id}_s{canon_strength(strength) or 'na'}_seed{seed}/{image_id}"


def find_with_ext(roots: Sequence[Path], rel: str, exts=(".jpg", ".jpeg", ".png")) -> Optional[Path]:
    for r in roots:
        for e in exts:
            p = Path(r) / (rel + e)
            if p.is_file():
                return p
    return None


# ------------------------------------------------------------------------------------------- bootstrap
_IDX: Dict[tuple, np.ndarray] = {}


def _boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    k = (n, n_boot, seed)
    if k not in _IDX:
        if len(_IDX) >= 6:
            _IDX.clear()
        _IDX[k] = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    return _IDX[k]


def _finite(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a[np.isfinite(a)]


def boot_ci(x, n_boot: int = 10000, ci: float = 0.95, seed: int = 2026) -> Tuple[float, float, float]:
    """(mean, lo, hi); same resampling as ``src.eval.stats.bootstrap_ci`` (identical output for the same input)."""
    a = _finite(x)
    if a.size == 0:
        return (float("nan"),) * 3
    means = a[_boot_idx(a.size, n_boot, seed)].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(a.mean()), float(lo), float(hi)


def wilcoxon_p(d) -> Tuple[float, int]:
    """Two-sided Wilcoxon signed-rank p on paired differences (zeros dropped). (p, n_nonzero); p=1 if all zero."""
    a = _finite(d)
    nz = a[a != 0]
    if nz.size == 0:
        return (1.0 if a.size else float("nan")), 0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(nz, zero_method="wilcox").pvalue), int(nz.size)
    except Exception:
        return float("nan"), int(nz.size)


def paired_stats(a: pd.Series, b: pd.Series, n_boot=10000, ci=0.95, seed=2026) -> Optional[dict]:
    """Paired comparison of per-image values, aligned by index (image_id); images missing in either are dropped."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    n = len(j)
    if n == 0:
        return None
    d = (j["a"] - j["b"]).to_numpy(dtype=np.float64)
    m, lo, hi = boot_ci(d, n_boot, ci, seed)
    p, nnz = wilcoxon_p(d)
    return {"n_pairs": int(n), "mean_a": float(j["a"].mean()), "mean_b": float(j["b"].mean()), "mean_diff": m,
            "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p, "n_nonzero": nnz, "share_a_greater": float((d > 0).mean()),
            "share_a_less": float((d < 0).mean())}


def paired_ratio(a: pd.Series, b: pd.Series, n_boot=10000, ci=0.95, seed=2026) -> Optional[dict]:
    """Ratio of means mean(a)/mean(b) over the SAME images, CI from a joint image bootstrap."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    n = len(j)
    if n == 0:
        return None
    av, bv = j["a"].to_numpy(np.float64), j["b"].to_numpy(np.float64)
    idx = _boot_idx(n, n_boot, seed)
    ma, mb = av[idx].mean(axis=1), bv[idx].mean(axis=1)
    ok = mb > 0
    r = ma[ok] / mb[ok]
    pt = float(av.mean() / bv.mean()) if bv.mean() > 0 else float("nan")
    if r.size == 0:
        return {"n_pairs": int(n), "ratio": pt, "ci_lo": float("nan"), "ci_hi": float("nan")}
    lo, hi = np.percentile(r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"n_pairs": int(n), "ratio": pt, "ci_lo": float(lo), "ci_hi": float(hi)}


def binom_ci(k: int, n: int, ci: float = 0.95) -> Tuple[float, float, float]:
    """(p, lo, hi) Clopper-Pearson interval."""
    if n <= 0:
        return (float("nan"),) * 3
    try:
        from scipy.stats import binomtest
        r = binomtest(int(k), int(n)).proportion_ci(confidence_level=ci, method="exact")
        return k / n, float(r.low), float(r.high)
    except Exception:
        return k / n, float("nan"), float("nan")


# --------------------------------------------------------------------------------------------- thresholds
def load_npz_sources(paths: Sequence[Path]) -> Dict[str, Optional[np.ndarray]]:
    """{image_id: embedding | None} from SourceStore npz files (ids, emb, ok)."""
    out: Dict[str, Optional[np.ndarray]] = {}
    for p in paths:
        z = np.load(p, allow_pickle=False)
        for i, k in enumerate(z["ids"]):
            out.setdefault(str(k), z["emb"][i].astype(np.float32) if bool(z["ok"][i]) else None)
    return out


def load_npz_edit_embs(paths: Sequence[Path]) -> Dict[str, Optional[np.ndarray]]:
    """{'editor/<prompt>_s<strength>_seed<k>/image_id': embedding | None} from EmbCache npz files (keys, emb, ok)."""
    out: Dict[str, Optional[np.ndarray]] = {}
    for p in paths:
        z = np.load(p, allow_pickle=False)
        for k, e, ok in zip(z["keys"], z["emb"], z["ok"]):
            out.setdefault(str(k), e.astype(np.float32) if bool(ok) else None)
    return out


def _impostor_sims(src: Mapping[str, np.ndarray], edits: Iterable[Tuple[str, np.ndarray]]) -> np.ndarray:
    try:
        from .edit_runner import impostor_sims
        return impostor_sims(src, edits)
    except ImportError:  # pragma: no cover
        ids = [i for i, e in src.items() if e is not None]
        S = np.stack([src[i] for i in ids]).astype(np.float64)
        S /= np.clip(np.linalg.norm(S, axis=1, keepdims=True), 1e-12, None)
        pos = {i: k for k, i in enumerate(ids)}
        out = []
        for iid, e in edits:
            if e is None:
                continue
            v = np.asarray(e, np.float64)
            v = v / max(np.linalg.norm(v), 1e-12)
            s = S @ v
            m = np.ones(len(ids), bool)
            if iid in pos:
                m[pos[iid]] = False
            out.append(s[m])
        return np.concatenate(out) if out else np.zeros(0)


def thresholds_from_embeddings(sources: Mapping[str, Optional[np.ndarray]], edit_embs: Mapping[str, Optional[np.ndarray]],
                               percentile: float = 95.0, restrict_ids: Optional[set] = None) -> dict:
    """PSR thresholds: 95th percentile of ArcFace sim between each E0 clean edit and every OTHER identity's source
    (V2_PLAN 4.1). Per editor (all prompts/strengths/seeds of that editor pooled) and pooled over editors.
    ``restrict_ids`` limits both sources and edits to those image ids (e.g. the test split)."""
    src = {i: e for i, e in sources.items() if e is not None and (restrict_ids is None or i in restrict_ids)}
    res: dict = {"percentile": percentile, "per_editor": {}, "n_pairs": {}, "n_sources_with_face": len(src),
                 "computed_by": "src.eval.analysis.thresholds_from_embeddings"}
    by_ed: Dict[str, list] = {}
    for key, emb in edit_embs.items():
        parts = key.split("/", 2)
        if len(parts) != 3:
            continue
        ed, _, iid = parts
        if restrict_ids is not None and iid not in restrict_ids:
            continue
        by_ed.setdefault(ed, []).append((iid, emb))
    pooled = []
    for ed in sorted(by_ed):
        if len(src) < 2:
            break
        s = _impostor_sims(src, by_ed[ed])
        if s.size:
            res["per_editor"][ed] = float(np.percentile(s, percentile))
            res["n_pairs"][ed] = int(s.size)
            pooled.append(s)
    if pooled:
        allp = np.concatenate(pooled)
        res["pooled"] = float(np.percentile(allp, percentile))
        res["n_pairs"]["pooled"] = int(allp.size)
    return res


def thresholds_from_npz(source_npz: Sequence[Path], edit_npz: Sequence[Path], percentile: float = 95.0,
                        restrict_ids: Optional[set] = None) -> dict:
    return thresholds_from_embeddings(load_npz_sources(source_npz), load_npz_edit_embs(edit_npz), percentile, restrict_ids)


def threshold_for(thresholds: Optional[dict], editor: str, scope: str = "per_editor") -> float:
    if not thresholds:
        return float("nan")
    pooled = thresholds.get("pooled", float("nan"))
    if scope == "pooled":
        return float(pooled)
    return float((thresholds.get("per_editor") or {}).get(editor, pooled))


# ----------------------------------------------------------------------------------------------- analysis
class Spec(NamedTuple):
    """One evaluation cell. ``strength=None`` -> the editor's default strength."""
    method: str
    purifier: str = "none"
    editor: str = HEAD_EDITOR
    strength: Optional[str] = None
    split: str = "test"


METRICS = ("psr", "sim", "no_face", "edit_lpips", "clip_i")
CELL_KEYS = ["split", "method", "purifier", "editor", "strength"]


class Analysis:
    """Per-image frame + memoised cell statistics and paired tests over a typed edits frame."""

    def __init__(self, edits: pd.DataFrame, thresholds: Optional[dict] = None, scope: str = "per_editor",
                 n_boot: int = 10000, ci: float = 0.95, seed: int = 2026):
        self.edits = edits
        self.thresholds = thresholds or {}
        self.scope = scope
        self.n_boot, self.ci, self.seed = n_boot, ci, seed
        self.pi = self._per_image()
        self._cells = {k: g.set_index("image_id") for k, g in self.pi.groupby(CELL_KEYS, sort=False)}
        self._memo: Dict[tuple, Optional[dict]] = {}
        self._pmemo: Dict[tuple, Optional[dict]] = {}

    # ---- construction
    def _per_image(self) -> pd.DataFrame:
        d = self.edits
        thr = d["editor"].map(lambda e: threshold_for(self.thresholds, e, self.scope)).astype(float)
        nf = d["face_detected"] == 0
        sim = d["arcface_sim"]
        hit = np.where(nf, 1.0, np.where(sim.notna() & thr.notna(), (sim < thr).astype(float), np.nan))
        t = d.assign(_hit=hit, _nf=nf.astype(float))
        gb = t.groupby(CELL_KEYS + ["image_id"], sort=False)
        pi = gb.agg(psr=("_hit", "mean"), sim=("arcface_sim", "mean"), no_face=("_nf", "mean"),
                    edit_lpips=("edit_lpips", "mean"), clip_i=("clip_i", "mean"), inv_psnr=("inv_psnr", "mean"),
                    inv_ssim=("inv_ssim", "mean"), inv_lpips=("inv_lpips", "mean"), n_edits=("seed", "size"),
                    n_seeds=("seed", "nunique"), n_prompts=("prompt_id", "nunique")).reset_index()
        return pi

    # ---- lookup
    def methods(self, split: str = "test") -> List[str]:
        return sorted(self.pi.loc[self.pi["split"] == split, "method"].unique())

    def resolve(self, spec: Spec) -> Spec:
        st = spec.strength
        if st is None:
            st = DEFAULT_STRENGTH.get(spec.editor, HEAD_STRENGTH)
        else:
            st = canon_strength(st)
        if (spec.split, spec.method, spec.purifier, spec.editor, st) not in self._cells and spec.strength is None:
            avail = sorted({k[4] for k in self._cells
                            if k[:4] == (spec.split, spec.method, spec.purifier, spec.editor)})
            if len(avail) == 1:
                st = avail[0]
        return spec._replace(strength=st)

    def frame(self, spec: Spec) -> Optional[pd.DataFrame]:
        s = self.resolve(spec)
        return self._cells.get((s.split, s.method, s.purifier, s.editor, s.strength))

    def series(self, spec: Spec, metric: str = "psr") -> Optional[pd.Series]:
        g = self.frame(spec)
        return None if g is None else g[metric]

    def cell(self, spec: Spec) -> Optional[dict]:
        """Statistics of one cell, or None if not run. Keys: n_images, n_edits, seeds_min/max, prompts_max,
        threshold, and for each metric {mean, lo, hi, n}."""
        s = self.resolve(spec)
        key = tuple(s)
        if key in self._memo:
            return self._memo[key]
        g = self.frame(s)
        if g is None:
            self._memo[key] = None
            return None
        out = {"n_images": int(len(g)), "n_edits": int(g["n_edits"].sum()), "seeds_min": int(g["n_seeds"].min()),
               "seeds_max": int(g["n_seeds"].max()), "prompts_max": int(g["n_prompts"].max()),
               "threshold": threshold_for(self.thresholds, s.editor, self.scope), "spec": s}
        for m in METRICS + ("inv_psnr", "inv_ssim", "inv_lpips"):
            mean, lo, hi = boot_ci(g[m].to_numpy(), self.n_boot, self.ci, self.seed)
            out[m] = {"mean": mean, "lo": lo, "hi": hi, "n": int(np.isfinite(g[m].to_numpy(dtype=float)).sum())}
        self._memo[key] = out
        return out

    def paired(self, a: Spec, b: Spec, metric: str = "psr") -> Optional[dict]:
        ka, kb = tuple(self.resolve(a)), tuple(self.resolve(b))
        key = (ka, kb, metric)
        if key in self._pmemo:
            return self._pmemo[key]
        sa, sb = self.series(a, metric), self.series(b, metric)
        r = None if sa is None or sb is None else paired_stats(sa, sb, self.n_boot, self.ci, self.seed)
        self._pmemo[key] = r
        return r

    def ratio(self, a: Spec, b: Spec, metric: str = "psr") -> Optional[dict]:
        sa, sb = self.series(a, metric), self.series(b, metric)
        return None if sa is None or sb is None else paired_ratio(sa, sb, self.n_boot, self.ci, self.seed)

    def reference_n(self, split: str = "test") -> int:
        sub = self.pi[self.pi["split"] == split]
        if len(sub) == 0:
            return 0
        return int(sub.groupby(CELL_KEYS[1:], sort=False).size().max())

    def has_thresholds(self) -> bool:
        return bool(self.thresholds) and (bool(self.thresholds.get("per_editor")) or "pooled" in self.thresholds)


# --------------------------------------------------------------------------------------- number formatting
def fmt_num(x, nd: int = 2, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def fmt_ci(mean, lo, hi, nd: int = 2, signed: bool = False) -> str:
    if mean is None or not math.isfinite(mean):
        return "n/a"
    return f"{fmt_num(mean, nd, signed)} [{fmt_num(lo, nd, signed)}, {fmt_num(hi, nd, signed)}]"


def fmt_p(p) -> str:
    if p is None or not math.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def cell_text(st: Optional[dict], metric: str = "psr", nd: int = 2, ref_n: Optional[int] = None) -> str:
    if st is None:
        return NOT_RUN
    m = st[metric]
    if not math.isfinite(m["mean"]):
        return "n/a"
    t = fmt_ci(m["mean"], m["lo"], m["hi"], nd)
    if ref_n and st["n_images"] < ref_n:
        t += f" (n={st['n_images']})"
    return t


# -------------------------------------------------------------------------------------------- invisibility
def invisibility_by_method(protect: Optional[pd.DataFrame], edits: Optional[pd.DataFrame] = None,
                           n_boot: int = 10000, seed: int = 2026, psnr_min: float = 38.0, ssim_min: float = 0.95
                           ) -> pd.DataFrame:
    """PSNR/SSIM/LPIPS of protected vs clean per method (published-PNG convention). Source: protect.csv, with a
    per-image fallback to the ``inv_*`` columns of the edit rows for methods missing there.
    If several protect runs exist for a (method, split), the run with the most images is used (note in column)."""
    rows = []
    seen = set()
    if protect is not None and len(protect):
        p = protect[protect["method"].astype(str) != ""]
        for (m, sp), g in p.groupby(["method", "split"], sort=True):
            runs = g.groupby("run_id").size().sort_values(ascending=False)
            run = runs.index[0]
            gg = g[g["run_id"] == run]
            rows.append(_inv_row(m, sp, gg["png_psnr"], gg["png_ssim"], gg["png_lpips"], "protect.csv",
                                 f"run {run}" + (f" (+{len(runs) - 1} other runs ignored)" if len(runs) > 1 else ""),
                                 n_boot, seed, psnr_min, ssim_min, gg.get("eps_255")))
            seen.add((m, sp))
    if edits is not None and len(edits):
        e = edits[edits["method"] != "clean"]
        for (m, sp), g in e.groupby(["method", "split"], sort=True):
            if (m, sp) in seen:
                continue
            gi = g.groupby("image_id")[["inv_psnr", "inv_ssim", "inv_lpips"]].first()
            if gi["inv_psnr"].notna().sum() == 0:
                continue
            rows.append(_inv_row(m, sp, gi["inv_psnr"], gi["inv_ssim"], gi["inv_lpips"], "edits.csv", "inv_* columns",
                                 n_boot, seed, psnr_min, ssim_min, g.get("eps_255")))
    cols = ["method", "split", "n_images", "eps_255", "psnr_mean", "psnr_lo", "psnr_hi", "psnr_min", "n_below_psnr",
            "ssim_mean", "ssim_lo", "ssim_hi", "ssim_min", "n_below_ssim", "lpips_mean", "lpips_lo", "lpips_hi",
            "source", "note"]
    return pd.DataFrame(rows, columns=cols)


def _inv_row(m, sp, psnr, ssim, lp, source, note, n_boot, seed, psnr_min, ssim_min, eps):
    ps, ss, ls = (_finite(x) for x in (psnr, ssim, lp))
    pm = boot_ci(ps, n_boot, 0.95, seed)
    sm = boot_ci(ss, n_boot, 0.95, seed)
    lm = boot_ci(ls, n_boot, 0.95, seed)
    eps_v = float(pd.to_numeric(eps, errors="coerce").dropna().iloc[0]) if eps is not None and len(pd.to_numeric(eps, errors="coerce").dropna()) else float("nan")
    return {"method": m, "split": sp, "n_images": int(ps.size), "eps_255": eps_v, "psnr_mean": pm[0], "psnr_lo": pm[1],
            "psnr_hi": pm[2], "psnr_min": float(ps.min()) if ps.size else float("nan"),
            "n_below_psnr": int((ps < psnr_min).sum()), "ssim_mean": sm[0], "ssim_lo": sm[1], "ssim_hi": sm[2],
            "ssim_min": float(ss.min()) if ss.size else float("nan"), "n_below_ssim": int((ss < ssim_min).sum()),
            "lpips_mean": lm[0], "lpips_lo": lm[1], "lpips_hi": lm[2], "source": source, "note": note}


# ------------------------------------------------------------------------------------------------- gates
@dataclass
class GateParams:
    """Operationalisation of the handoff's gates. The gate VALUES (2x, half) are the handoff's; the E3 cut-offs and
    the purifier sets are analyst choices (the handoff gives none) and are printed with every verdict."""
    psnr_min: float = 38.0
    ssim_min: float = 0.95
    e3_core: Tuple[str, ...] = ("jpeg_q75", "jpeg_q50", "downup_0.5", "vae_rt", "diffpure_0.1", "diffpure_0.2")
    e3_collapse_rel: float = 0.5   # mean relative PSR drop >= this  -> "collapses" -> continue with Phase 5
    e3_barely_rel: float = 0.2     # mean relative PSR drop <  this  -> "barely drops" -> recommend skipping Phase 5
    e4_method: str = "v2_cheap"
    e4_purifiers: Tuple[str, ...] = ("jpeg_q75", "jpeg_q50", "downup_0.5", "vae_rt")
    e4_info: Tuple[str, ...] = ("jpeg_other", "downup_other")
    e4_factor: float = 2.0
    degenerate_hi: float = 0.5     # v1 PSR above this: doubling impossible (V2_PLAN R20)
    trivial_lo: float = 0.02       # v1 PSR below this: doubling is trivial (R20)
    e5_method: str = "v2_main"
    e5_extra: Tuple[str, ...] = ("v2_final",)
    e5_purifier: str = "diffpure_0.1"
    e5_frac: float = 0.5
    e5_info: Tuple[str, ...] = ("diffpure_0.2", "diffpure_0.3", "gridpure", "restore_swin2sr")
    e6_method: str = "v2_antipure"
    e6_ref: str = "v2_main"
    e6_purifiers: Tuple[str, ...] = ("diffpure_0.1", "diffpure_0.2", "diffpure_0.3")


def parse_gate_params(overrides: Sequence[str]) -> GateParams:
    gp = GateParams()
    for ov in overrides or ():
        k, v = ov.split("=", 1)
        cur = getattr(gp, k)
        if isinstance(cur, tuple):
            setattr(gp, k, tuple(x for x in v.split(",") if x))
        elif isinstance(cur, float):
            setattr(gp, k, float(v))
        else:
            setattr(gp, k, v)
    return gp


@dataclass
class GateResult:
    id: str
    title: str
    rule: str
    verdict: str = "NOT RUN"
    rows: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self):
        return {"id": self.id, "title": self.title, "rule": self.rule, "verdict": self.verdict, "rows": self.rows,
                "notes": self.notes}


def _psr(an: Analysis, spec: Spec):
    st = an.cell(spec)
    return None if st is None else st["psr"]


def gate_e3(an: Analysis, gp: GateParams) -> GateResult:
    g = GateResult("E3", "v1 under purification (decision rule for Phase 5)",
                   f"mean relative PSR drop of v1 over {','.join(gp.e3_core)} (SD1.5 s0.55): >= {gp.e3_collapse_rel:.0%} "
                   f"collapses -> continue with Phase 5; < {gp.e3_barely_rel:.0%} barely drops -> recommend skipping "
                   "Phase 5 and ask the user; in between -> ask the user. The cut-offs are analyst proposals "
                   "(the handoff gives none).")
    base = Spec("v1", "none")
    b = _psr(an, base)
    if b is None or not math.isfinite(b["mean"]):
        g.notes.append("v1 without purifier (SD1.5 s0.55) not run or no PSR threshold available")
        return g
    drops = []
    for pur in PURIFIER_ORDER[1:]:
        sp = Spec("v1", pur)
        s = _psr(an, sp)
        if s is None:
            continue
        pr = an.paired(sp, base)
        rel = 1 - s["mean"] / b["mean"] if b["mean"] > 0 else float("nan")
        g.rows.append({"purifier": pur, "psr_none": b["mean"], "psr": s["mean"], "psr_lo": s["lo"], "psr_hi": s["hi"],
                       "diff": pr["mean_diff"] if pr else float("nan"), "diff_lo": pr["ci_lo"] if pr else float("nan"),
                       "diff_hi": pr["ci_hi"] if pr else float("nan"), "p": pr["p_wilcoxon"] if pr else float("nan"),
                       "rel_drop": rel, "n_pairs": pr["n_pairs"] if pr else 0, "core": pur in gp.e3_core})
        if pur in gp.e3_core and math.isfinite(rel):
            drops.append(rel)
    if b["mean"] < 0.1:
        g.notes.append(f"v1 PSR without purifier is only {b['mean']:.2f}: a relative drop is hard to interpret")
    if len(drops) < 3:
        g.verdict = "PARTIAL" if drops else "NOT RUN"
        g.notes.append(f"only {len(drops)} of {len(gp.e3_core)} core purifiers available")
        return g
    md = float(np.mean(drops))
    g.notes.append(f"mean relative PSR drop over {len(drops)} core purifiers = {md:.2f}")
    if md >= gp.e3_collapse_rel:
        g.verdict = "DEGRADES (continue with Phase 5)"
    elif md < gp.e3_barely_rel:
        g.verdict = "BARELY DROPS (recommend skipping Phase 5; ask the user)"
    else:
        g.verdict = "AMBIGUOUS (ask the user)"
    return g


def _inv_mean(an: Analysis, method: str) -> Optional[Tuple[float, int, int]]:
    """(mean inv_psnr over images, n_images, n_below_38) of a method from its edit rows (any cell)."""
    sub = an.pi[(an.pi["method"] == method) & (an.pi["split"] == "test") & an.pi["inv_psnr"].notna()]
    if len(sub) == 0:
        return None
    per = sub.groupby("image_id")["inv_psnr"].mean()
    return float(per.mean()), int(len(per)), int((per < 38).sum())


def gate_e4(an: Analysis, gp: GateParams) -> GateResult:
    g = GateResult("E4", f"{gp.e4_method} vs v1 under cheap purifiers",
                   f"PSR({gp.e4_method}) >= {gp.e4_factor:g} x PSR(v1) under {','.join(gp.e4_purifiers)} (SD1.5 s0.55) "
                   f"with PSNR >= {gp.psnr_min:g}. Point estimate decides; the paired ratio CI is shown. "
                   f"Degenerate if v1 PSR >= {gp.degenerate_hi:g} (doubling impossible), trivial if < {gp.trivial_lo:g}.")
    stat = []
    for pur in tuple(gp.e4_purifiers) + tuple(gp.e4_info):
        primary = pur in gp.e4_purifiers
        sv, s1 = Spec(gp.e4_method, pur), Spec("v1", pur)
        a, b = _psr(an, sv), _psr(an, s1)
        row = {"purifier": pur, "primary": primary, "psr_new": None, "psr_v1": None, "ratio": float("nan"),
               "ratio_lo": float("nan"), "ratio_hi": float("nan"), "diff": float("nan"), "diff_lo": float("nan"),
               "diff_hi": float("nan"), "p": float("nan"), "status": NOT_RUN}
        if a is not None and b is not None and math.isfinite(a["mean"]) and math.isfinite(b["mean"]):
            r = an.ratio(sv, s1)
            pr = an.paired(sv, s1)
            row.update(psr_new=a["mean"], psr_v1=b["mean"], ratio=r["ratio"], ratio_lo=r["ci_lo"], ratio_hi=r["ci_hi"],
                       diff=pr["mean_diff"], diff_lo=pr["ci_lo"], diff_hi=pr["ci_hi"], p=pr["p_wilcoxon"])
            ok = a["mean"] >= gp.e4_factor * b["mean"]
            if b["mean"] >= gp.degenerate_hi:
                row["status"] = "DEGENERATE (v1 PSR >= %.2f; doubling impossible)" % gp.degenerate_hi
            elif b["mean"] < gp.trivial_lo:
                row["status"] = "PASS (trivial: v1 PSR < %.2f)" % gp.trivial_lo if ok else "FAIL"
            else:
                row["status"] = "PASS" if ok else "FAIL"
        stat.append(row)
    g.rows = stat
    prim = [r for r in stat if r["primary"]]
    inv = _inv_mean(an, gp.e4_method)
    if inv is None:
        g.notes.append(f"invisibility of {gp.e4_method} not available (no inv_psnr)")
    else:
        g.notes.append(f"{gp.e4_method} PSNR mean {inv[0]:.2f} dB over {inv[1]} images; {inv[2]} below {gp.psnr_min:g}")
    if not any(r["status"] != NOT_RUN for r in prim):
        g.verdict = "NOT RUN"
        return g
    statuses = [r["status"] for r in prim]
    psnr_ok = inv is not None and inv[0] >= gp.psnr_min
    if any(s == "FAIL" for s in statuses) or (inv is not None and not psnr_ok):
        g.verdict = "FAIL"
    elif any(s.startswith("DEGENERATE") for s in statuses):
        g.verdict = "DEGENERATE (ask the user; V2_PLAN R20)"
    elif any(s == NOT_RUN for s in statuses) or inv is None:
        g.verdict = "PARTIAL (some cells not run)"
    else:
        g.verdict = "PASS"
    return g


def gate_e5(an: Analysis, gp: GateParams) -> GateResult:
    g = GateResult("E5", f"{gp.e5_method}: held-out DiffPure t*=0.1 vs no purifier",
                   f"PSR under {gp.e5_purifier} >= {gp.e5_frac:g} x PSR without purifier (SD1.5 s0.55, same images). "
                   "Point estimate decides; 'robust' if the paired ratio CI lower bound also >= the fraction.")
    verdicts = {}
    for meth in (gp.e5_method,) + tuple(gp.e5_extra):
        base = Spec(meth, "none")
        b = _psr(an, base)
        for pur in (gp.e5_purifier,) + tuple(gp.e5_info):
            sp = Spec(meth, pur)
            s = _psr(an, sp)
            row = {"method": meth, "purifier": pur, "psr_none": None, "psr": None, "ratio": float("nan"),
                   "ratio_lo": float("nan"), "ratio_hi": float("nan"), "status": NOT_RUN,
                   "primary": meth == gp.e5_method and pur == gp.e5_purifier}
            if b is not None and s is not None and math.isfinite(b["mean"]) and math.isfinite(s["mean"]):
                r = an.ratio(sp, base)
                row.update(psr_none=b["mean"], psr=s["mean"], ratio=r["ratio"], ratio_lo=r["ci_lo"], ratio_hi=r["ci_hi"])
                if pur == gp.e5_purifier:
                    ok = s["mean"] >= gp.e5_frac * b["mean"]
                    row["status"] = ("PASS (robust)" if r["ci_lo"] >= gp.e5_frac else "PASS") if ok else "FAIL"
                else:
                    row["status"] = "info"
            g.rows.append(row)
            if row["primary"]:
                verdicts[meth] = row["status"]
    v = verdicts.get(gp.e5_method, NOT_RUN)
    g.verdict = "NOT RUN" if v == NOT_RUN else v
    for meth in gp.e5_extra:
        if meth in verdicts:
            g.notes.append(f"{meth} (informational): {verdicts[meth]}")
    return g


def gate_e6(an: Analysis, gp: GateParams) -> GateResult:
    g = GateResult("E6", f"{gp.e6_method} vs {gp.e6_ref} (anti-purification terms)",
                   "go if the paired PSR gain under higher-t* DiffPure has a 95% CI above 0; 'no gain (within CI)' "
                   "otherwise (handoff Phase 5: drop it if gains are within the CI).")
    gains = []
    for pur in gp.e6_purifiers:
        a, r = Spec(gp.e6_method, pur), Spec(gp.e6_ref, pur)
        pr = an.paired(a, r)
        if pr is None:
            g.rows.append({"purifier": pur, "status": NOT_RUN})
            continue
        gain = pr["ci_lo"] > 0
        gains.append(gain)
        g.rows.append({"purifier": pur, "psr_new": pr["mean_a"], "psr_ref": pr["mean_b"], "diff": pr["mean_diff"],
                       "diff_lo": pr["ci_lo"], "diff_hi": pr["ci_hi"], "p": pr["p_wilcoxon"],
                       "status": "gain" if gain else "no gain (within CI)" if pr["ci_hi"] >= 0 else "worse"})
    if not gains:
        g.verdict = "NOT RUN"
    elif any(gains):
        g.verdict = "GO (gain on %d of %d purifiers)" % (sum(gains), len(gains))
    else:
        g.verdict = "NO-GO (gains within CI)"
    return g


def gate_invisibility(inv: pd.DataFrame, gp: GateParams, methods: Sequence[str],
                      human: Optional[dict] = None) -> GateResult:
    g = GateResult("INV", "Invisibility", f"PSNR >= {gp.psnr_min:g} and SSIM >= {gp.ssim_min:g} (per image); human 2AFC "
                   "accuracy not far above 50%.")
    sub = inv[(inv["method"].isin(methods)) & (inv["split"] == "test")] if len(inv) else inv
    if len(sub) == 0:
        return g
    allok = True
    for r in sub.itertuples(index=False):
        ok = r.psnr_mean >= gp.psnr_min and r.ssim_mean >= gp.ssim_min and r.n_below_psnr == 0 and r.n_below_ssim == 0
        allok &= bool(ok)
        g.rows.append({"method": r.method, "n": r.n_images, "psnr": r.psnr_mean, "ssim": r.ssim_mean,
                       "n_below_psnr": r.n_below_psnr, "n_below_ssim": r.n_below_ssim,
                       "status": "PASS" if ok else "FAIL" if (r.psnr_mean < gp.psnr_min or r.ssim_mean < gp.ssim_min) else "MARGINAL"})
    g.verdict = "PASS" if allok else "FAIL/MARGINAL (see rows)"
    if human and human.get("task_a"):
        a = human["task_a"]
        g.notes.append(f"human 2AFC accuracy {a.get('accuracy', float('nan')):.2f} "
                       f"[{a.get('ci_lo', float('nan')):.2f}, {a.get('ci_hi', float('nan')):.2f}], "
                       f"{a.get('n_raters', '?')} raters" + (" (PILOT)" if human.get("pilot") else ""))
    else:
        g.notes.append("human 2AFC result not available (E10 not run)")
    return g


def all_gates(an: Analysis, inv: pd.DataFrame, gp: Optional[GateParams] = None, human: Optional[dict] = None
              ) -> Dict[str, GateResult]:
    gp = gp or GateParams()
    out = {"E3": gate_e3(an, gp), "E4": gate_e4(an, gp), "E5": gate_e5(an, gp), "E6": gate_e6(an, gp)}
    meths = [m for m in an.methods() if m != "clean"]
    out["INV"] = gate_invisibility(inv, gp, meths, human)
    return out


# ------------------------------------------------------------------------------------------------ coverage
def coverage(edits: pd.DataFrame) -> pd.DataFrame:
    """One row per (split, method, purifier, editor): images, rows, seeds, strengths, prompts, face rate, LPIPS fill."""
    g = edits.groupby(["split", "method", "purifier", "editor"], sort=True)
    rows = []
    for (sp, m, p, e), d in g:
        rows.append({"split": sp, "method": m, "purifier": p, "editor": e, "n_images": d["image_id"].nunique(),
                     "n_rows": len(d), "seeds": ",".join(str(int(s)) for s in sorted(d["seed"].dropna().unique())),
                     "n_seeds": d["seed"].nunique(), "strengths": ",".join(sorted(d["strength"].unique())) or "-",
                     "n_prompts": d["prompt_id"].nunique(), "face_rate": float((d["face_detected"] == 1).mean()),
                     "lpips_filled": float(d["edit_lpips"].notna().mean()),
                     "min_images_per_strength": int(d.groupby("strength")["image_id"].nunique().min())})
    return pd.DataFrame(rows)


def expected_cells(ev: dict, observed_methods: Sequence[str], tiers: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Cells planned by ``eval.tiers`` (config/parts/eval.yaml) as (tier, split, method, purifier, editor, strength).
    Method patterns (e.g. 'eps*') are expanded against the observed method names; if none match, the pattern itself
    is listed. The SMOKE tier is skipped."""
    from .edit_runner import expand_tier
    rows = []
    for tier, rules in (ev.get("tiers") or {}).items():
        if tier == "SMOKE" or (tiers and tier not in tiers):
            continue
        names = set()
        for r in rules:
            for pat in r["methods"]:
                if any(c in pat for c in "*?["):
                    hit = [m for m in observed_methods if fnmatch.fnmatchcase(m, pat)]
                    names.update(hit if hit else [pat])
                else:
                    names.add(pat)
        for sp in ("test", "messy"):
            for m in sorted(names):
                for j in expand_tier(ev, tier, m, sp, ["_"]):
                    rows.append({"tier": tier, "split": sp, "method": m, "purifier": j.purifier, "editor": j.editor,
                                 "strength": j.strength})
    d = pd.DataFrame(rows, columns=["tier", "split", "method", "purifier", "editor", "strength"])
    return d.drop_duplicates().reset_index(drop=True)


def coverage_vs_plan(edits: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Expected cells with the number of images found (0 = missing)."""
    if len(expected) == 0:
        return expected.assign(n_images=[], status=[])
    have = edits.groupby(["split", "method", "purifier", "editor", "strength"])["image_id"].nunique().rename("n_images")
    m = expected.merge(have.reset_index(), on=["split", "method", "purifier", "editor", "strength"], how="left")
    m["n_images"] = m["n_images"].fillna(0).astype(int)
    m["status"] = np.where(m["n_images"] > 0, "present", "missing")
    return m


# ------------------------------------------------------------------------------------------ table writers
@dataclass
class Table:
    name: str
    title: str
    df: pd.DataFrame
    long: Optional[pd.DataFrame] = None
    note: str = ""
    synthetic: bool = False


SYNTH_BANNER = ("SYNTHETIC DATA - generated by src.eval.analysis.synthetic_results for software tests. "
                "These are NOT experimental results.")


def _md_escape(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def to_markdown(t: Table) -> str:
    cols = [str(c) for c in t.df.columns]
    out = []
    if t.synthetic:
        out.append(f"> **{SYNTH_BANNER}**\n")
    out.append(f"**{t.title}**\n")
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    for row in t.df.itertuples(index=False):
        out.append("| " + " | ".join(_md_escape(v) for v in row) + " |")
    if t.note:
        out.append("")
        out.append(t.note)
    return "\n".join(out) + "\n"


_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
            "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(s) -> str:
    return "".join(_TEX_ESC.get(c, c) for c in str(s))


def to_latex(t: Table) -> str:
    cols = list(t.df.columns)
    n = len(cols)
    align = "l" + "c" * (n - 1)
    lines = ["\\begin{table}[t]", "\\centering", "\\small"]
    if n > 7:
        lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(tex_escape(c) for c in cols) + " \\\\")
    lines.append("\\midrule")
    for row in t.df.itertuples(index=False):
        lines.append(" & ".join(tex_escape(v) for v in row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if n > 7:
        lines.append("}")
    cap = tex_escape(t.title)
    if t.synthetic:
        cap = "SYNTHETIC DATA, not results. " + cap
    lines.append(f"\\caption{{{cap}}}")
    lines.append(f"\\label{{tab:{t.name}}}")
    if t.note:
        lines.append("\\par\\footnotesize " + tex_escape(t.note.replace("\n", " ")))
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write_table(t: Table, outdir) -> List[Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    p = out / f"{t.name}.md"
    p.write_text(to_markdown(t), encoding="utf-8")
    paths.append(p)
    p = out / f"{t.name}.csv"
    t.df.to_csv(p, index=False)
    paths.append(p)
    p = out / f"{t.name}.tex"
    p.write_text(to_latex(t), encoding="utf-8")
    paths.append(p)
    if t.long is not None:
        p = out / f"{t.name}_long.csv"
        t.long.to_csv(p, index=False)
        paths.append(p)
    return paths


def gate_to_markdown(g: GateResult) -> str:
    out = [f"**{g.id}: {g.title}**  verdict: **{g.verdict}**", "", f"Rule: {g.rule}", ""]
    if g.rows:
        cols = list(g.rows[0].keys())
        for r in g.rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join("---" for _ in cols) + "|")
        for r in g.rows:
            out.append("| " + " | ".join(_md_escape(fmt_num(r.get(c)) if isinstance(r.get(c), float) else r.get(c, ""))
                                         for c in cols) + " |")
        out.append("")
    for n in g.notes:
        out.append(f"- {n}")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------- templates
_PH = re.compile(r"\{\{\s*(table|num|gate)\s*:\s*([^}|]+?)\s*(?:\|\s*([^}]*?))?\s*\}\}")


def render_template(text: str, tables_dir, numbers: Mapping[str, Any], gates_dir=None) -> Tuple[str, List[str]]:
    """Replace ``{{table:NAME}}`` with tables/NAME.md, ``{{num:KEY}}`` with numbers[KEY], ``{{gate:ID}}`` with the
    gate verdict (numbers['gate.ID']). Anything unavailable becomes ``PENDING (...)`` and is listed in the second
    return value, so a rendered report never contains an invented number."""
    missing: List[str] = []
    tdir = Path(tables_dir)

    def sub(m):
        kind, key, default = m.group(1), m.group(2).strip(), m.group(3)
        if kind == "table":
            p = tdir / f"{key}.md"
            if p.is_file():
                return p.read_text(encoding="utf-8").rstrip("\n")
            missing.append(f"table:{key}")
            return f"PENDING (table `{key}` not generated; run scripts/make_tables.py)"
        k = key if kind == "num" else f"gate.{key}"
        if k in numbers and numbers[k] not in (None, ""):
            return str(numbers[k])
        missing.append(f"{kind}:{key}")
        return default.strip() if default else f"PENDING ({k})"

    return _PH.sub(sub, text), missing


# --------------------------------------------------------------------------------------------- workers
_W: dict = {}


def _fake_distance(a, b) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float32)
    y = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float32)
    return float(np.abs(x - y).mean() / 255.0)


def lpips_worker_init(backend: str = "real", threads: int = 1, quality: int = 90) -> None:
    """Pool initialiser: load the LPIPS(VGG) model once per process (CPU)."""
    _W["quality"] = quality
    if backend == "fake":
        _W["fn"] = _fake_distance
        return
    import torch
    torch.set_num_threads(threads)
    import lpips
    m = lpips.LPIPS(net="vgg", verbose=False).eval()

    def fn(a, b):
        def t(im):
            x = torch.from_numpy(np.asarray(im.convert("RGB"), dtype=np.float32)).permute(2, 0, 1)[None] / 127.5 - 1.0
            return x
        with torch.no_grad():
            return float(m(t(a), t(b)).item())
    _W["fn"] = fn


def _load_jpeg_domain(path: str):
    """Open an edit image; non-JPEG files are round-tripped through JPEG q90 (same definition as the in-job LPIPS)."""
    from PIL import Image
    im = Image.open(path)
    im.load()
    im = im.convert("RGB")
    if str(path).lower().endswith(".png"):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(_W.get("quality", 90)))
        buf.seek(0)
        im = Image.open(buf)
        im.load()
        im = im.convert("RGB")
    return im


def lpips_chunk(chunk: Sequence[Tuple[int, str, str]]) -> List[Tuple[int, float]]:
    """chunk = [(row_index, clean_edit_path, protected_edit_path)] -> [(row_index, lpips or nan)]."""
    out = []
    for idx, cp, ep in chunk:
        try:
            out.append((idx, float(_W["fn"](_load_jpeg_domain(cp), _load_jpeg_domain(ep)))))
        except Exception:
            out.append((idx, float("nan")))
    return out


# --------------------------------------------------------------------------------------- synthetic data
SYN_BASE = {("sd15_img2img", "0.3"): 0.72, ("sd15_img2img", "0.55"): 0.52, ("sd15_img2img", "0.75"): 0.36,
            ("sdxl_img2img", "0.55"): 0.47, ("ip2p", ""): 0.58, ("sd_inpaint", "1"): 0.66,
            ("ipadapter_faceid", "0.55"): 0.62}
SYN_EFFECT = {"clean": 0.0, "noise_uniform": 0.02, "noise_gauss_psnr": 0.03, "photoguard": 0.22, "mist": 0.25,
              "v1": 0.30, "v2_cheap": 0.31, "v2_main": 0.34, "v2_antipure": 0.34, "v2_final": 0.40,
              "abl_vis_only": 0.26, "abl_den_only": 0.28, "abl_no_sem": 0.33, "abl_no_pur": 0.30,
              "eps2": 0.20, "eps6": 0.46, "eps8": 0.52}
SYN_PSNR = {"clean": float("nan"), "noise_uniform": 42.0, "noise_gauss_psnr": 41.9, "photoguard": 40.5, "mist": 39.8,
            "v1": 41.9, "v2_cheap": 40.8, "v2_main": 40.2, "v2_antipure": 39.9, "v2_final": 40.0, "abl_vis_only": 41.0,
            "abl_den_only": 40.5, "abl_no_sem": 40.2, "abl_no_pur": 41.0, "eps2": 46.0, "eps6": 38.5, "eps8": 36.8}
SYN_EPS = {"eps2": 2, "eps6": 6, "eps8": 8}


def _syn_survival(method: str, purifier: str) -> float:
    if purifier in ("none", "adaptive_ae"):
        base = 1.0 if purifier == "none" else (0.55 if method.startswith("v2") else 0.1)
        return base
    fam = "jpeg" if purifier.startswith("jpeg") else "downup" if purifier.startswith("downup") else \
        "vae" if purifier == "vae_rt" else "dp" if purifier.startswith("diffpure") else "other"
    table = {"v1": {"jpeg": 0.35, "downup": 0.3, "vae": 0.3, "dp": 0.08, "other": 0.15},
             "v2": {"jpeg": 0.85, "downup": 0.8, "vae": 0.85, "dp": 0.55, "other": 0.5}}
    k = "v1" if method in ("v1", "photoguard", "mist", "noise_uniform", "noise_gauss_psnr") else "v2"
    if method == "v2_cheap":
        return table["v2"][fam] * (0.4 if fam == "dp" else 1.0)
    return table[k][fam]


def synthetic_results(n_images: int = 30, seeds: Sequence[int] = (1, 2, 3), rng_seed: int = 0,
                      run_prefix: str = SYNTH_PREFIX, emb_dim: int = 64):
    """Synthetic edits/protect frames (string-typed like ``read_results_csv``) + E0 embeddings, following the real
    schema and the tier structure of config/parts/eval.yaml (subset). Effects are invented ON PURPOSE to exercise the
    code; every run_id starts with 'SYNTHETIC'. Returns (edits_df, protect_df, sources{id: emb}, edit_embs{key: emb})."""
    rng = np.random.default_rng(rng_seed)
    ids = [f"{1000 + 7 * i}" for i in range(n_images)]
    off = rng.normal(0, 0.07, n_images)                                   # per-image difficulty, shared by methods
    src = {i: (lambda v: v / np.linalg.norm(v))(rng.normal(size=emb_dim)).astype(np.float32) for i in ids}
    cells: List[Tuple[str, str, str, str, str]] = []                      # (method, purifier, editor, strength, tier)
    meths_h = ["clean", "noise_uniform", "noise_gauss_psnr", "photoguard", "mist", "v1", "v2_cheap", "v2_main", "v2_final"]
    for m in meths_h:
        for (ed, st) in SYN_BASE:
            cells.append((m, "none", ed, st, "H"))
    for m in ("clean", "v1", "v2_cheap", "v2_main", "v2_antipure", "v2_final"):
        for p in PURIFIER_ORDER[1:14]:
            cells.append((m, p, HEAD_EDITOR, HEAD_STRENGTH, "P"))
    for m in ("photoguard", "mist"):
        for p in ("jpeg_q75", "jpeg_q50", "downup_0.5", "vae_rt", "diffpure_0.1"):
            cells.append((m, p, HEAD_EDITOR, HEAD_STRENGTH, "P"))
    cells.append(("v2_antipure", "none", HEAD_EDITOR, HEAD_STRENGTH, "H"))
    for m in ("abl_vis_only", "abl_den_only", "abl_no_sem", "abl_no_pur"):
        for p in ("none", "jpeg_q50", "vae_rt", "diffpure_0.1", "diffpure_0.2"):
            cells.append((m, p, HEAD_EDITOR, HEAD_STRENGTH, "A"))
        for (ed, st) in (("ipadapter_faceid", "0.55"), ("ip2p", ""), ("sd_inpaint", "1")):
            cells.append((m, "none", ed, st, "A"))
    for m in ("eps2", "eps6", "eps8"):
        for p in ("none", "jpeg_q75", "diffpure_0.1"):
            cells.append((m, p, HEAD_EDITOR, HEAD_STRENGTH, "E9"))
        cells.append((m, "none", "ipadapter_faceid", "0.55", "E9"))
    for m in ("clean", "v1", "v2_final"):
        for ed, st in ((HEAD_EDITOR, HEAD_STRENGTH), ("ipadapter_faceid", "0.55")):
            cells.append((m, "adaptive_ae", ed, st, "E8"))
    seen, uniq = set(), []
    for c in cells:
        if c[:4] not in seen:
            seen.add(c[:4])
            uniq.append(c)
    rows = []
    edit_embs: Dict[str, np.ndarray] = {}
    t0 = "2026-10-15T08:00:00Z"
    for (m, pur, ed, st, tier) in uniq:
        base = SYN_BASE[(ed, st)]
        eff = SYN_EFFECT.get(m, 0.3) * _syn_survival(m, pur)
        mult = 1.0
        if ed == "ipadapter_faceid" and m not in ("clean", "v2_final", "eps6", "eps8"):
            mult = 0.25
        elif ed == "ipadapter_faceid":
            mult = 0.8
        if ed == "sd_inpaint":
            mult *= 0.5
        headline = ed == HEAD_EDITOR and st == HEAD_STRENGTH
        for ii, iid in enumerate(ids):
            plist = [0, 1, 2] if (headline and tier == "H" and pur == "none") else [ii % 3]
            for pi_ in plist:
                for s in seeds:
                    mu = base + off[ii] - eff * mult
                    sim = float(np.clip(mu + rng.normal(0, 0.05) + rng.normal(0, 0.03), -0.2, 0.95))
                    nf = int(rng.random() < min(0.03, 0.06 * eff * mult))
                    if m == "clean" and pur == "none":
                        e = src[iid] * (base + off[ii]) + rng.normal(size=emb_dim).astype(np.float32) * 0.1
                        e = e / np.linalg.norm(e)
                        edit_embs[f"{ed}/p{pi_}_s{st or 'na'}_seed{s}/{iid}"] = e.astype(np.float32)
                        sim = float(np.dot(e, src[iid]))
                    inv_p = SYN_PSNR.get(m, 40.0) + rng.normal(0, 0.4) if m != "clean" else float("nan")
                    rows.append({
                        "run_id": f"{run_prefix}_{tier}", "method": m, "method_cfg_hash": "synthetic0000",
                        "image_id": iid, "split": "test", "seed": str(s), "editor": ed, "prompt_id": f"p{pi_}",
                        "strength": st, "purifier": pur, "protect_run_id": f"{run_prefix}_P", "eps_255": SYN_EPS.get(m, 4) if m != "clean" else 0,
                        "purifier_params": "{}", "editor_params": "{}", "source_path": f"/kaggle/input/{iid}.png",
                        "protected_path": f"/kaggle/working/{m}/{iid}.png", "purified_path": "",
                        "edit_path": f"/kaggle/working/edits/{m}/{pur}/{ed}/p{pi_}_s{st or 'na'}_seed{s}/{iid}.jpg",
                        "clean_edit_path": "", "code_git_hash": "synthetic", "created_utc": t0,
                        "face_detected": 0 if nf else 1, "n_faces": 0 if nf else 1, "det_score": 0.9,
                        "arcface_sim": float("nan") if nf else sim, "fr2_sim": float("nan"),
                        "edit_lpips": float("nan") if (m == "clean" and pur == "none") else float(np.clip(0.15 + 0.6 * eff * mult + rng.normal(0, 0.03), 0, 1)),
                        "clip_i": float(np.clip(0.85 - 0.2 * eff + rng.normal(0, 0.03), 0, 1)),
                        "inv_psnr": inv_p, "inv_ssim": 0.975 + rng.normal(0, 0.003) if m != "clean" else float("nan"),
                        "inv_lpips": 0.23 + rng.normal(0, 0.03) if m != "clean" else float("nan"),
                        "gpu_seconds_edit": 1.0, "gpu_seconds_purify": 0.0})
    edits = pd.DataFrame(rows, columns=EDIT_COLUMNS)
    prot_rows = []
    for m in sorted({c[0] for c in uniq if c[0] != "clean"}):
        for iid in ids:
            prot_rows.append({"run_id": f"{run_prefix}_P_{m}", "method": m, "method_cfg_hash": "synthetic0000",
                              "image_id": iid, "split": "test", "seed": "1", "editor": "none", "prompt_id": "none",
                              "strength": "", "purifier": "none", "src_file": f"{iid}.png", "out_file": f"{iid}.png",
                              "eps_255": SYN_EPS.get(m, 4), "step_255": 0.5, "iters": 40, "v1_psnr": np.nan,
                              "v1_ssim": np.nan, "v1_lpips": np.nan, "v1_linf_model": np.nan,
                              "png_psnr": SYN_PSNR.get(m, 40.0) + rng.normal(0, 0.4),
                              "png_ssim": 0.975 + rng.normal(0, 0.003), "png_lpips": 0.23 + rng.normal(0, 0.03),
                              "png_maxabs_u8": SYN_EPS.get(m, 4), "png_frac_at_bound": 0.3, "gpu_seconds": 60.0,
                              "peak_mem_mb": 9000.0, "created_utc": t0})
    prot = pd.DataFrame(prot_rows, columns=PROTECT_COLUMNS)

    def as_str(df):
        out = df.copy()
        for c in out.columns:
            out[c] = out[c].map(lambda v: "nan" if (isinstance(v, float) and math.isnan(v)) else
                                (repr(v) if isinstance(v, float) else str(v)))
        return out

    return as_str(edits), as_str(prot), src, edit_embs


def write_synthetic_tree(root, n_images: int = 24, seeds: Sequence[int] = (1, 2, 3), rng_seed: int = 0) -> dict:
    """Write a fake Kaggle-output tree (job dirs with sharded metrics CSVs, one duplicated block, some missing LPIPS,
    npz caches, a protect CSV) under ``root`` to exercise ``scripts/ingest_results.py``. Returns a summary dict."""
    root = Path(root)
    edits, prot, src, eemb = synthetic_results(n_images, seeds, rng_seed)
    jobs = {"SYN_S07": ["clean"], "SYN_S08": ["v1", "noise_uniform"], "SYN_S09": ["v2_main", "v2_cheap", "v2_final"],
            "SYN_S10": ["photoguard", "mist", "noise_gauss_psnr", "v2_antipure"]}
    other = set(edits["method"]) - {m for v in jobs.values() for m in v}
    jobs["SYN_S11"] = sorted(other)
    n_rows = 0
    for job, methods in jobs.items():
        sub = edits[edits["method"].isin(methods)]
        d = root / job / "results" / f"eval_{job}"
        d.mkdir(parents=True, exist_ok=True)
        half = np.arange(len(sub)) % 2
        for k in (0, 1):
            sub[half == k].to_csv(d / f"metrics.shard{k}.csv", index=False)
        n_rows += len(sub)
        (root / job / "DONE").write_text("synthetic")
    # duplicate block (a re-run of some v1 rows), second copy without LPIPS and with an older timestamp
    dup = edits[edits["method"] == "v1"].head(50).copy()
    dup["edit_lpips"] = "nan"
    dup["created_utc"] = "2026-10-14T00:00:00Z"
    dd = root / "SYN_S12" / "results" / "rerun"
    dd.mkdir(parents=True, exist_ok=True)
    dup.to_csv(dd / "metrics.csv", index=False)
    for m, g in prot.groupby("method"):
        pd_ = root / ("SYN_S08" if m in jobs["SYN_S08"] else "SYN_S09" if m in jobs["SYN_S09"] else "SYN_S11") / "results" / f"protect_{m}"
        pd_.mkdir(parents=True, exist_ok=True)
        g.to_csv(pd_ / "metrics.csv", index=False)
    cache = root / "SYN_S07" / "results" / "eval_SYN_S07" / "cache" / "arcface"
    cache.mkdir(parents=True, exist_ok=True)
    ids = sorted(src)
    np.savez(cache / "test_sources.npz", ids=np.array(ids, dtype=str), emb=np.stack([src[i] for i in ids]),
             ok=np.ones(len(ids), bool), bbox=np.zeros((len(ids), 4), np.float32), det=np.ones(len(ids), np.float32),
             nf=np.ones(len(ids), np.int32))
    keys = sorted(eemb)
    np.savez(cache / "edit_embeddings.npz", keys=np.array(keys, dtype=str), emb=np.stack([eemb[k] for k in keys]),
             ok=np.ones(len(keys), bool))
    # a CSV that is neither an edit nor a protect table (must be skipped), and one with a broken schema
    (root / "SYN_S07" / "results" / "eval_SYN_S07" / "metrics.junk.csv").write_text("a,b\n1,2\n")
    (root / "SYN_S12" / "results" / "rerun" / "metrics.broken.csv").write_text(
        "method,image_id,editor,face_detected,arcface_sim\nv1,1,sd15_img2img,1,0.3\n")
    return {"rows_written": n_rows, "dup_rows": len(dup), "n_images": n_images, "methods": sorted(set(edits["method"])),
            "protect_rows": len(prot)}
