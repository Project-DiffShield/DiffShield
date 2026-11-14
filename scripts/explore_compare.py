#!/usr/bin/env python
"""Compare the exploration variants X0..X5 (docs/EXPLORATION.md) from their tier-EXPL result folders. CPU only.

  python scripts/explore_compare.py DIR [DIR ...] [--out_dir results/explore] [--base X0] [--psr_threshold 0.1005]
         [--n_boot 2000] [--seed 2026]

Each DIR is a result folder (or any ancestor, e.g. the unzipped X3_outputs) holding ``metrics*.csv`` edit rows
(column ``arcface_sim``; protect-run CSVs without it are ignored) and, next to it, ``psnr_ssim.csv`` (per-photo PSNR/SSIM,
written by the job). The variant is recognised from the ``method`` column (v2_next = X0, v2_x_eps8 = X1, v2_x_it100 = X2,
v2_x_id_direct_a = X3, v2_x_id_direct_b = X4, v2_x_id_recon = X5). Any subset of the variants works; the difference to X0 is
reported only if X0 is among them.

PSR rule (same as src.eval.edit_runner.psr_hit): an edit is a hit if no face was detected or arcface_sim < threshold.
Seeds / prompts are averaged within a photo first; a cell is (editor, cleanup); cell PSR = mean over photos.
"all-cell mean" = mean over cells of the cell PSR. Difference to X0: per-photo all-cell score (mean over that photo's cells),
paired by photo, photos resampled with replacement (n_boot draws, seed) -> percentile 95% interval.

--invis_from RUN_DIR --invis_out CSV : instead write per-photo PSNR/SSIM of a protect run (metrics.csv columns png_psnr,
png_ssim = protected PNG vs clean photo) and exit. Used by the Kaggle jobs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LABELS = {"v2_next": "X0", "v2_x_eps8": "X1", "v2_x_it100": "X2", "v2_x_id_direct_a": "X3",
          "v2_x_id_direct_b": "X4", "v2_x_id_recon": "X5", "v2_x_eps8_tuned": "X6"}
NAMES = {"X0": "base (v2_next)", "X1": "eps8", "X2": "it100", "X3": "id_direct_a", "X4": "id_direct_b", "X5": "id_recon",
         "X6": "eps8_tuned"}
PSNR_MIN, SSIM_MIN = 38.0, 0.95
SD15, FID = "sd15_img2img", "ipadapter_faceid"
DEDUP = ["image_id", "purifier", "editor", "strength", "prompt_id", "seed"]


# ------------------------------------------------------------------------------------------- per-photo invisibility
def write_invisibility(run_dir, out_csv) -> pd.DataFrame:
    p = Path(run_dir) / "metrics.csv"
    df = pd.read_csv(p, dtype={"image_id": str})
    out = df[["image_id", "png_psnr", "png_ssim"]].rename(columns={"png_psnr": "psnr", "png_ssim": "ssim"})
    out = out.drop_duplicates("image_id", keep="last").sort_values("image_id")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


# ------------------------------------------------------------------------------------------- loading
def load_variants(paths) -> Dict[str, dict]:
    """{label: {"edits": DataFrame, "invis": DataFrame|None, "method": str}} from result folders."""
    out: Dict[str, dict] = {}
    for root in paths:
        root = Path(root)
        files = [root] if root.is_file() else sorted(root.rglob("metrics*.csv"))
        for f in files:
            head = pd.read_csv(f, nrows=0)
            if "arcface_sim" not in head.columns:
                continue
            df = pd.read_csv(f, dtype={"image_id": str, "strength": str, "prompt_id": str}, keep_default_na=True)
            for method, g in df.groupby("method"):
                if method not in LABELS:
                    continue
                lab = LABELS[method]
                ent = out.setdefault(lab, {"edits": [], "invis": None, "method": method})
                ent["edits"].append(g)
                inv = f.parent / "psnr_ssim.csv"
                if inv.exists() and ent["invis"] is None:
                    ent["invis"] = pd.read_csv(inv, dtype={"image_id": str})
    for lab, ent in out.items():
        e = pd.concat(ent["edits"], ignore_index=True)
        cols = [c for c in DEDUP if c in e.columns]
        ent["edits"] = e.drop_duplicates(cols, keep="last")
    return out


# ------------------------------------------------------------------------------------------- statistics
def hit(face_detected, sim, thr) -> np.ndarray:
    fd = pd.to_numeric(face_detected, errors="coerce").fillna(0).to_numpy() > 0
    s = pd.to_numeric(sim, errors="coerce").to_numpy(dtype=float)
    return (~fd) | np.isnan(s) | (s < thr)


def per_photo_cells(edits: pd.DataFrame, thr: float) -> pd.DataFrame:
    """rows = image_id, columns = 'editor|cleanup', value = PSR of that photo in that cell (seeds/prompts averaged)."""
    d = edits.copy()
    d["hit"] = hit(d["face_detected"], d["arcface_sim"], thr).astype(float)
    d["cell"] = d["editor"].astype(str) + "|" + d["purifier"].astype(str)
    return d.groupby(["image_id", "cell"])["hit"].mean().unstack("cell")


def paired_boot(diff: np.ndarray, n_boot: int, seed: int):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    m = diff[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def compare(variants: Dict[str, dict], thr: float = 0.1005, base: str = "X0", n_boot: int = 2000, seed: int = 2026):
    """Returns (cells_df, summary_df). cells_df: variant, editor, cleanup, psr, n_photos."""
    cell_rows, sums, photo_scores = [], [], {}
    for lab in sorted(variants):
        ent = variants[lab]
        pc = per_photo_cells(ent["edits"], thr)
        for cell in pc.columns:
            ed, pu = cell.split("|", 1)
            cell_rows.append({"variant": lab, "editor": ed, "cleanup": pu, "psr": float(pc[cell].mean()),
                              "n_photos": int(pc[cell].notna().sum())})
        cols_sd = [c for c in pc.columns if c.startswith(SD15 + "|")]
        cols_fi = [c for c in pc.columns if c.startswith(FID + "|")]
        cm = {c: float(pc[c].mean()) for c in pc.columns}
        row = {"variant": lab, "name": NAMES.get(lab, lab), "n_photos": int(len(pc)), "n_cells": len(pc.columns),
               "psr_all": float(np.mean(list(cm.values()))),
               "psr_sd15": float(np.mean([cm[c] for c in cols_sd])) if cols_sd else float("nan"),
               "psr_ipadapter": float(np.mean([cm[c] for c in cols_fi])) if cols_fi else float("nan")}
        inv = ent.get("invis")
        if inv is not None:
            inv = inv[inv["image_id"].isin(pc.index)] if len(pc) else inv
            ps, ss = inv["psnr"].to_numpy(float), inv["ssim"].to_numpy(float)
            row.update({"psnr_mean": float(ps.mean()), "ssim_mean": float(ss.mean()),
                        "psnr_min": float(ps.min()), "ssim_min": float(ss.min()),
                        "n_violations": int(((ps < PSNR_MIN) | (ss < SSIM_MIN)).sum()), "n_invis": int(len(inv))})
        else:
            row.update({k: float("nan") for k in ("psnr_mean", "ssim_mean", "psnr_min", "ssim_min")})
            row.update({"n_violations": None, "n_invis": 0})
        photo_scores[lab] = pc.mean(axis=1)   # per-photo all-cell score
        sums.append(row)
    summary = pd.DataFrame(sums)
    for c in ("diff_mean", "diff_lo", "diff_hi", "diff_n_photos"):
        summary[c] = np.nan
    if base in photo_scores:
        for i, r in summary.iterrows():
            if r["variant"] == base:
                continue
            a, b = photo_scores[r["variant"]], photo_scores[base]
            common = a.index.intersection(b.index)
            m, lo, hi = paired_boot((a.loc[common] - b.loc[common]).to_numpy(), n_boot, seed)
            summary.loc[i, ["diff_mean", "diff_lo", "diff_hi", "diff_n_photos"]] = [m, lo, hi, len(common)]
    return pd.DataFrame(cell_rows), summary


# ------------------------------------------------------------------------------------------- output
def _f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def to_markdown(cells: pd.DataFrame, summary: pd.DataFrame, thr: float, base: str, n_boot: int, seed: int) -> str:
    L: List[str] = ["# Exploration comparison (X0..X5)", "",
                    "EXPLORATION on the 28 tune_search photos. NOT a test-set claim; adopting any variant needs a fresh "
                    "100-photo test run (docs/EXPLORATION.md).", "",
                    f"PSR = share of edits whose ArcFace similarity to the source is below {thr} (or no face found); higher is better. "
                    "Seeds/prompts are averaged within a photo; a cell is editor x cleanup.", "",
                    f"Difference to {base}: paired by photo, all-cell mean per photo, photos resampled with replacement "
                    f"({n_boot} draws, seed {seed}), percentile 95% interval.", "",
                    "**X1 (eps8) changes the threat / visibility trade-off (a larger perturbation budget is visibly stronger "
                    "noise); it cannot be compared with the other variants at equal invisibility.** Compare it by its own "
                    "PSNR/SSIM columns, not as a free gain.", ""]
    L += ["## Summary", "",
          "| variant | name | photos | PSR all-cell | PSR sd15 | PSR ipadapter | diff to " + base + " [95% CI] | mean PSNR | mean SSIM | min PSNR | min SSIM | photos violating PSNR>=38 and SSIM>=0.95 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        diff = "-" if r["variant"] == base or np.isnan(r["diff_mean"]) else \
            f"{r['diff_mean']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}] (n={int(r['diff_n_photos'])})"
        viol = "n/a" if r["n_violations"] is None or (isinstance(r["n_violations"], float) and np.isnan(r["n_violations"])) \
            else f"{int(r['n_violations'])} of {int(r['n_invis'])}"
        L.append(f"| {r['variant']} | {r['name']} | {int(r['n_photos'])} | {_f(r['psr_all'])} | {_f(r['psr_sd15'])} | "
                 f"{_f(r['psr_ipadapter'])} | {diff} | {_f(r['psnr_mean'], 2)} | {_f(r['ssim_mean'], 4)} | "
                 f"{_f(r['psnr_min'], 2)} | {_f(r['ssim_min'], 4)} | {viol} |")
    L += ["", "## PSR per variant x editor x cleanup", ""]
    if len(cells):
        piv = cells.pivot_table(index="variant", columns=["editor", "cleanup"], values="psr")
        head = ["variant"] + [f"{e} / {c}" for e, c in piv.columns]
        L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for v, row in piv.iterrows():
            L.append("| " + " | ".join([str(v)] + [_f(row[c]) for c in piv.columns]) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--out_dir", default="results/explore")
    ap.add_argument("--base", default="X0")
    ap.add_argument("--psr_threshold", type=float, default=0.1005)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--invis_from", default=None)
    ap.add_argument("--invis_out", default=None)
    a = ap.parse_args(argv)
    if a.invis_from:
        if not a.invis_out:
            ap.error("--invis_out is required with --invis_from")
        df = write_invisibility(a.invis_from, a.invis_out)
        print(f"wrote {a.invis_out}: {len(df)} photos, mean PSNR {df['psnr'].mean():.2f}, mean SSIM {df['ssim'].mean():.4f}")
        return 0
    if not a.dirs:
        ap.error("give at least one result folder")
    variants = load_variants(a.dirs)
    if not variants:
        print("no EXPL edit rows (metrics*.csv with arcface_sim and a known method) found", file=sys.stderr)
        return 1
    cells, summary = compare(variants, a.psr_threshold, a.base, a.n_boot, a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "explore_cells.csv", index=False)
    summary.to_csv(out / "explore_summary.csv", index=False)
    md = to_markdown(cells, summary, a.psr_threshold, a.base, a.n_boot, a.seed)
    (out / "explore_compare.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
