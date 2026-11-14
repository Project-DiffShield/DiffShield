#!/usr/bin/env python
"""Paired comparison of two methods' PSR on the SAME photos (pre-registered adoption rule). CPU only.

Rule: the NEW method replaces the OLD (current final) method only if PSR_new - PSR_old on the headline cell
(sd15_img2img, no cleanup, strength 0.55) has a paired-bootstrap 95% interval whose LOWER bound is > 0.

  python scripts/paired_compare.py --old OLD_DIR_OR_CSV [...] --new NEW_DIR_OR_CSV [...] \
        [--clean CLEAN_CSV ...] [--old-method M] [--new-method M] [--clean-method clean] \
        [--thresholds results/combined/psr_thresholds.json] [--split test] [--out report.json|report.md]

Nothing is re-implemented from the harness: rows are typed / de-duplicated with ``src.eval.analysis`` (typed_edits,
dedup_edits), the per-photo PSR comes from ``src.eval.analysis.Analysis`` (which uses ``edit_runner.psr_hit``'s rule:
no face = hit, else sim < per-editor impostor threshold; seeds and prompts averaged within a photo first), and the
bootstrap is ``analysis.paired_stats`` / ``analysis.boot_ci`` (same resampling as ``src.eval.stats.bootstrap_ci``;
n_boot / ci / seed from config ``stats``). The paired difference is taken per photo, then the photos are resampled
once and the same resample is used for both methods.

SECONDARY (reported alongside, never part of the ADOPT/KEEP rule): the paired bootstrap (same photo resampling, same seed) of
the difference in per-photo mean ``arcface_sim`` (new minus old; negative = better protection), for every cell, for the
headline cell, and pooled over strengths per (editor, cleanup); the same for ``fr2_sim`` when that column has values
(``scripts/score_fr2.py`` / job F1). Edits without a face have no similarity and are ignored in those means.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval import analysis as A  # noqa: E402

HEADLINE = ("sd15_img2img", "none", "0.55")  # (editor, purifier, strength)
CELL = ["editor", "purifier", "strength"]


# ------------------------------------------------------------------------------------------------ loading
def expand_paths(items: Sequence[str]) -> List[Path]:
    """Files, globs, or directories (-> metrics*.csv, else edits.csv, inside)."""
    out: List[Path] = []
    for it in items:
        p = Path(it)
        hits: List[str]
        if p.is_dir():
            hits = sorted(glob.glob(str(p / "metrics*.csv"))) or sorted(glob.glob(str(p / "edits*.csv")))
        elif any(ch in it for ch in "*?["):
            hits = sorted(glob.glob(it))
        else:
            hits = [it] if p.is_file() else []
        if not hits:
            raise FileNotFoundError(f"no CSV found for {it!r}")
        out.extend(Path(h) for h in hits)
    return out


def load_edits(items: Sequence[str], method: Optional[str], split: Optional[str], label: str, notes: List[str]) -> pd.DataFrame:
    files = expand_paths(items)
    frames = []
    for f in files:
        df = A.read_results_csv(f)
        missing = A.validate_schema(df.columns, "edit")
        if missing:
            raise ValueError(f"{label}: {f} lacks required columns {missing}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    notes.append(f"{label}: {len(files)} file(s), {len(df)} rows")
    if method is not None:
        df = df[df["method"] == method]
        if df.empty:
            raise ValueError(f"{label}: no rows with method == {method!r}")
    methods = sorted(df["method"].unique())
    if len(methods) != 1:
        raise ValueError(f"{label}: files contain several methods {methods}; pick one with --{label}-method")
    if split is not None:
        df = df[df["split"] == split]
        if df.empty:
            raise ValueError(f"{label}: no rows with split == {split!r}")
    df, st = A.dedup_edits(df)
    if st["duplicates_dropped"]:
        notes.append(f"{label}: dropped {st['duplicates_dropped']} duplicate rows ({st['conflicts']} with conflicting sim)")
    return A.typed_edits(df)


def load_thresholds(path: Optional[Path]) -> dict:
    p = Path(path) if path else REPO / "results" / "combined" / "psr_thresholds.json"
    if not p.is_file():
        raise FileNotFoundError(f"PSR thresholds file not found: {p} (pass --thresholds)")
    return json.loads(p.read_text())


def load_stats_cfg(path: Optional[Path] = None) -> dict:
    import yaml
    p = Path(path) if path else REPO / "config" / "config.yaml"
    st = (yaml.safe_load(p.read_text()) or {}).get("stats", {}) if p.is_file() else {}
    return {"n_boot": int(st.get("n_boot", 10000)), "ci": float(st.get("ci", 0.95)),
            "seed": int(st.get("bootstrap_seed", 2026))}


# ------------------------------------------------------------------------------------------------ comparison
def _row_keys(edits: pd.DataFrame) -> Dict[tuple, set]:
    """{cell: set of (image_id, prompt_id, seed)}"""
    out: Dict[tuple, set] = {}
    for k, g in edits.groupby(CELL, sort=False):
        out[k] = set(zip(g["image_id"], g["prompt_id"], g["seed"].astype(int).astype(str)))
    return out


def per_photo_psr(edits: pd.DataFrame, thresholds: dict, n_boot: int, ci: float, seed: int) -> Dict[tuple, pd.Series]:
    an = A.Analysis(edits, thresholds, "per_editor", n_boot, ci, seed)
    out = {}
    for k, g in an.pi.groupby(CELL, sort=False):
        out[k] = g.set_index("image_id")["psr"].sort_index()
    return out


SIM_METRICS = ("arcface_sim", "fr2_sim")


def per_photo_sim(edits: pd.DataFrame, col: str, by: Sequence[str]) -> Dict[tuple, pd.Series]:
    """{cell: per-photo mean of ``col`` over that photo's edits (seeds, prompts, and any strengths/rows pooled by ``by``)}.
    Edits without a face have NaN similarity and are ignored in the mean (the PSR counts them as hits; the similarity
    cannot). A photo with no finite value in the cell is NaN and dropped from the pairing (``n_pairs`` says how many)."""
    out: Dict[tuple, pd.Series] = {}
    if col not in edits or not edits[col].notna().any():
        return out
    g = edits.groupby(list(by) + ["image_id"], sort=False)[col].mean().reset_index()
    for k, gg in g.groupby(list(by), sort=False):
        out[k if isinstance(k, tuple) else (k,)] = gg.set_index("image_id")[col].sort_index()
    return out


def sim_diffs(old: pd.DataFrame, new: pd.DataFrame, by: Sequence[str], kw: dict) -> Dict[tuple, dict]:
    """{cell: {metric: paired stats of mean sim new - old}}; SECONDARY continuous identity metric (negative = better
    protection). Same paired bootstrap as the PSR (``analysis.paired_stats``: same photo resampling, same seed)."""
    res: Dict[tuple, dict] = {}
    for m in SIM_METRICS:
        so, sn = per_photo_sim(old, m, by), per_photo_sim(new, m, by)
        for c in sorted(set(so) & set(sn)):
            ps = A.paired_stats(sn[c], so[c], **kw)
            if ps is not None:
                res.setdefault(c, {})[m] = _stat(ps)
    return res


def _stat(s: Optional[dict]) -> Optional[dict]:
    if s is None:
        return None
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in s.items()}


def compare(old: pd.DataFrame, new: pd.DataFrame, clean: Optional[pd.DataFrame], thresholds: dict,
            n_boot: int = 10000, ci: float = 0.95, seed: int = 2026, expect_photos: Optional[int] = 100,
            headline: tuple = HEADLINE) -> dict:
    kw = dict(n_boot=n_boot, ci=ci, seed=seed)
    psr = {"old": per_photo_psr(old, thresholds, **kw), "new": per_photo_psr(new, thresholds, **kw)}
    keys = {"old": _row_keys(old), "new": _row_keys(new)}
    if clean is not None:
        psr["clean"] = per_photo_psr(clean, thresholds, **kw)
        keys["clean"] = _row_keys(clean)
    issues: List[str] = []
    for ed in sorted({k[0] for k in list(psr["old"]) + list(psr["new"])}):
        if ed not in thresholds.get("per_editor", {}):
            issues.append(f"no per-editor PSR threshold for editor {ed!r}; pooled threshold used")
    cells = sorted(set(psr["old"]) | set(psr["new"]))
    sim_cell = sim_diffs(old, new, CELL, kw)
    sim_ep = sim_diffs(old, new, ["editor", "purifier"], kw)
    res_cells = []
    for c in cells:
        r: dict = {"editor": c[0], "cleanup": c[1], "strength": c[2], "is_headline": tuple(c) == tuple(headline)}
        cissues: List[str] = []
        so, sn = psr["old"].get(c), psr["new"].get(c)
        if so is None or sn is None:
            cissues.append("cell missing in " + ("old" if so is None else "new"))
            r["issues"] = cissues
            r["status"] = "missing"
            res_cells.append(r)
            continue
        io_, in_ = set(so.index), set(sn.index)
        common = sorted(io_ & in_)
        if io_ - in_:
            cissues.append(f"{len(io_ - in_)} photos only in old: {sorted(io_ - in_)[:5]}")
        if in_ - io_:
            cissues.append(f"{len(in_ - io_)} photos only in new: {sorted(in_ - io_)[:5]}")
        ko, kn = keys["old"][c], keys["new"][c]
        if ko != kn:
            cissues.append(f"row keys (photo,prompt,seed) differ: {len(ko - kn)} only in old, {len(kn - ko)} only in new")
        if expect_photos is not None and len(common) != expect_photos:
            cissues.append(f"{len(common)} common photos, expected {expect_photos}")
        nan_o, nan_n = int(so.loc[common].isna().sum()), int(sn.loc[common].isna().sum())
        if nan_o or nan_n:
            cissues.append(f"photos with undefined PSR (threshold/sim missing): old {nan_o}, new {nan_n}")
        a, b = sn.loc[common], so.loc[common]
        d = A.paired_stats(a, b, **kw)
        r.update(n_common=len(common), psr_old=_stat(dict(zip(("mean", "lo", "hi"), A.boot_ci(b.to_numpy(), **kw)))),
                 psr_new=_stat(dict(zip(("mean", "lo", "hi"), A.boot_ci(a.to_numpy(), **kw)))),
                 diff=_stat(d))
        if "clean" in psr:
            sc = psr["clean"].get(c)
            if sc is None:
                cissues.append("clean cell missing")
            else:
                miss = [p for p in common if p not in sc.index]
                if miss:
                    cissues.append(f"{len(miss)} photos missing from clean: {miss[:5]}")
                if keys["clean"][c] != ko:
                    cissues.append("clean row keys (photo,prompt,seed) differ from old")
                cm = [p for p in common if p in sc.index]
                cl = sc.loc[cm]
                r["psr_clean"] = _stat(dict(zip(("mean", "lo", "hi"), A.boot_ci(cl.to_numpy(), **kw))))
                r["net_old"] = _stat(A.paired_stats(b.loc[cm], cl, **kw))
                r["net_new"] = _stat(A.paired_stats(a.loc[cm], cl, **kw))
                nd = (a.loc[cm] - cl) - (b.loc[cm] - cl)
                r["net_gain_diff"] = _stat(dict(zip(("mean", "lo", "hi"), A.boot_ci(nd.to_numpy(), **kw))))
                r["net_gain_n"] = len(cm)
        r["sim_secondary"] = sim_cell.get(tuple(c), {})
        r["issues"] = cissues
        r["status"] = "ok" if not cissues else "mismatch"
        res_cells.append(r)

    head = next((r for r in res_cells if r["is_headline"]), None)
    if head is None:
        verdict, reason = "INVALID", f"headline cell {headline} not present in both methods"
    elif head["status"] == "missing":
        verdict, reason = "INVALID", "headline cell missing in one method: " + "; ".join(head["issues"])
    elif head["issues"]:
        verdict, reason = "INVALID", "headline cell not comparable: " + "; ".join(head["issues"])
    elif head["diff"] is None or head["diff"]["ci_lo"] is None:
        verdict, reason = "INVALID", "headline difference could not be computed"
    elif head["diff"]["ci_lo"] > 0:
        verdict, reason = "ADOPT", f"lower bound {head['diff']['ci_lo']:+.4f} > 0"
    else:
        verdict, reason = "KEEP", f"lower bound {head['diff']['ci_lo']:+.4f} <= 0"
    sim_note = ("SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity "
                "new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; "
                "fr2_sim only where the column has values")
    return {"rule": "ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0",
            "headline_cell": {"editor": headline[0], "cleanup": headline[1], "strength": headline[2]},
            "bootstrap": {"n_boot": n_boot, "ci": ci, "seed": seed, "unit": "photo, paired"},
            "thresholds": {"per_editor": thresholds.get("per_editor"), "pooled": thresholds.get("pooled")},
            "verdict": verdict, "reason": reason, "global_issues": issues, "cells": res_cells,
            "similarity_secondary_note": sim_note,
            "similarity_headline": (head or {}).get("sim_secondary", {}),
            "similarity_by_editor_purifier": [
                {"editor": k[0], "purifier": k[1], **v} for k, v in sorted(sim_ep.items())],
            "per_cleanup_verdicts_informational": {
                f"{r['editor']}|{r['cleanup']}|{r['strength']}":
                    ("n/a" if r["status"] == "missing" or not r.get("diff") or r["diff"]["ci_lo"] is None
                     else ("ADOPT" if r["diff"]["ci_lo"] > 0 else "KEEP"))
                    for r in res_cells}}


# ------------------------------------------------------------------------------------------------ output
def _f(s, key="mean", nd=3, signed=False):
    return "n/a" if not s or s.get(key) is None else (f"{s[key]:+.{nd}f}" if signed else f"{s[key]:.{nd}f}")


def _ci(s, signed=False, nd=3):
    if not s:
        return "n/a"
    mk, lo, hi = ("mean_diff", "ci_lo", "ci_hi") if "ci_lo" in s else ("mean", "lo", "hi")
    if s.get(mk) is None:
        return "n/a"
    return f"{_f(s, mk, nd, signed)} [{_f(s, lo, nd, signed)}, {_f(s, hi, nd, signed)}]"


def to_markdown(rep: dict, notes: Sequence[str] = ()) -> str:
    h = rep["headline_cell"]
    L = [f"# Paired PSR comparison (new minus old)", "",
         f"Headline cell: {h['editor']} x cleanup {h['cleanup']} x strength {h['strength']}",
         f"Bootstrap: {rep['bootstrap']}", ""]
    L += [f"- {n}" for n in notes]
    L += [f"- WARNING: {i}" for i in rep["global_issues"]]
    L += ["", f"## VERDICT: {rep['verdict']}  ({rep['reason']})", "", rep["rule"], "",
          "## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)", "",
          "| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["cells"]:
        key = f"{r['editor']}|{r['cleanup']}|{r['strength']}"
        if r["status"] == "missing":
            L.append(f"| {r['editor']} | {r['cleanup']} | {r['strength']} | - | - | - | - | n/a | - | - | - | - | {'; '.join(r['issues'])} |")
            continue
        nets = lambda s: _ci(s) if s else "n/a"
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["editor"], r["cleanup"] + (" (HEADLINE)" if r["is_headline"] else ""), r["strength"], r["n_common"],
            _f(r["psr_old"]), _f(r["psr_new"]),
            _ci(r["diff"], signed=True), rep["per_cleanup_verdicts_informational"][key], _f(r.get("psr_clean")),
            _f(r["net_old"], "mean_diff") if r.get("net_old") else "n/a",
            _f(r["net_new"], "mean_diff") if r.get("net_new") else "n/a",
            _ci(r.get("net_gain_diff"), signed=True), "; ".join(r["issues"]) or "-"))
    L += ["", "## Secondary: mean identity similarity, paired difference new - old (negative = better)", "",
          rep.get("similarity_secondary_note", ""), ""]
    hs = rep.get("similarity_headline") or {}
    L.append("Headline cell: " + (", ".join(f"{m}: {_ci(v, signed=True)} (n={v['n_pairs']})" for m, v in hs.items())
                                  or "n/a"))
    L += ["", "| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |",
          "|---|---|---|---|---|---|---|"]
    for r in rep["cells"]:
        ss = r.get("sim_secondary") or {}
        a, f = ss.get("arcface_sim"), ss.get("fr2_sim")
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            r["editor"], r["cleanup"], r["strength"], _f(a, "mean_b"), _f(a, "mean_a"),
            _ci(a, signed=True), _ci(f, signed=True)))
    L += ["", "Pooled over strengths, per (editor, purifier):", "",
          "| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |", "|---|---|---|---|"]
    for r in rep.get("similarity_by_editor_purifier", []):
        L.append(f"| {r['editor']} | {r['purifier']} | {_ci(r.get('arcface_sim'), signed=True)} | {_ci(r.get('fr2_sim'), signed=True)} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", nargs="+", required=True, help="CSV file(s), glob(s) or dir(s) of the current method")
    ap.add_argument("--new", nargs="+", required=True, help="CSV file(s), glob(s) or dir(s) of the candidate method")
    ap.add_argument("--clean", nargs="+", help="CSV file(s) of the clean (unprotected) method, for net gain")
    ap.add_argument("--old-method"), ap.add_argument("--new-method")
    ap.add_argument("--clean-method", default="clean", help="method name to select inside --clean files (default clean)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--thresholds", type=Path, help="psr_thresholds.json (default results/combined/psr_thresholds.json)")
    ap.add_argument("--config", type=Path, help="config.yaml holding stats: {n_boot, ci, bootstrap_seed}")
    ap.add_argument("--n-boot", type=int), ap.add_argument("--seed", type=int)
    ap.add_argument("--expect-photos", type=int, default=100, help="photos required in the headline cell (0 = no check)")
    ap.add_argument("--out", type=Path, help=".json or .md report")
    a = ap.parse_args(argv)

    st = load_stats_cfg(a.config)
    if a.n_boot:
        st["n_boot"] = a.n_boot
    if a.seed is not None:
        st["seed"] = a.seed
    notes: List[str] = []
    try:
        old = load_edits(a.old, a.old_method, a.split, "old", notes)
        new = load_edits(a.new, a.new_method, a.split, "new", notes)
        clean = load_edits(a.clean, a.clean_method, a.split, "clean", notes) if a.clean else None
        rep = compare(old, new, clean, load_thresholds(a.thresholds), expect_photos=a.expect_photos or None, **st)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    rep["inputs"] = notes
    md = to_markdown(rep, notes)
    print(md)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rep, indent=2, default=str) if a.out.suffix == ".json" else md)
    return 2 if rep["verdict"] == "INVALID" else 0


if __name__ == "__main__":
    sys.exit(main())
