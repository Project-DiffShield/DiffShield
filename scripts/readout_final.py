"""One-command readout of the final comparison (SD1.5 img2img strength 0.55 unless stated).

python scripts/readout_final.py [--combined results/combined] [--split test] [--out results/tables/final_readout.md]
        [--methods clean,v1,v2_main,v2_cheap,v2_final,v2_tgt_zero,v2_tgt_zero_mid,v2_tgt_zero_hi,photoguard,mist]

Per method x purifier: PSR (share of edits with no face or ArcFace similarity below the per-editor E0 threshold), averaged over
seeds/prompts FIRST so the unit is the photo (n photos), 95% bootstrap CI over photos; NET = method minus clean photos under the
same cleanup (paired); VS V1 = paired difference to v1. A '*' marks a CI that excludes 0. Invisibility per method comes from the
protection runs (published PNG convention: PSNR >= 38 dB, SSIM >= 0.95 on EVERY photo). Nothing is invented: a cell without rows
prints '-'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_METHODS = "clean,v1,v2_main,v2_cheap,v2_final,v2_tgt_zero,v2_tgt_zero_mid,v2_tgt_zero_hi,photoguard,mist"
PURIFIERS = ["none", "jpeg_q75", "jpeg_q50", "downup_0.5", "vae_rt", "diffpure_0.1", "diffpure_0.2", "diffpure_0.3"]


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = 2000):
    if len(x) == 0:
        return float("nan"), float("nan")
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def per_photo_psr(e: pd.DataFrame, method: str, editor: str, purifier: str, thr: float, strength: float | None) -> pd.Series:
    d = e[(e["method"] == method) & (e["editor"] == editor) & (e["purifier"] == purifier)]
    if strength is not None:
        d = d[(d["strength"].astype(str) == "") | ((pd.to_numeric(d["strength"], errors="coerce") - strength).abs() < 1e-9)]
    if d.empty:
        return pd.Series(dtype=float)
    sim = pd.to_numeric(d["arcface_sim"], errors="coerce")
    prot = ((sim < thr) | sim.isna()).astype(float)      # no face (NaN similarity) counts as protected
    return prot.groupby(d["image_id"].astype(str)).mean()


def cell(a: pd.Series, base: pd.Series | None, rng) -> str:
    if a.empty:
        return "-"
    s = f"{a.mean():.3f}"
    if base is not None and not base.empty:
        ids = a.index.intersection(base.index)
        if len(ids):
            lo, hi = boot_ci((a[ids] - base[ids]).to_numpy(), rng)
            s += f" ({(a[ids] - base[ids]).mean():+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''})"
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default=str(REPO / "results" / "combined"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--out", default=str(REPO / "results" / "tables" / "final_readout.md"))
    a = ap.parse_args(argv)
    comb = Path(a.combined)
    e = pd.read_csv(comb / "edits.csv", low_memory=False)
    e = e[e["split"] == a.split].copy()
    thr = json.loads((comb / "psr_thresholds.json").read_text())["per_editor"]
    rng = np.random.default_rng(2026)
    methods = [m for m in a.methods.split(",") if m in set(e["method"])]
    ed = "sd15_img2img"
    lines = [f"# Final readout, split = {a.split} (SD1.5 img2img, strength 0.55)", "",
             "Cell = PSR (mean over photos); in brackets: paired difference to CLEAN photos under the same cleanup (net protection) with 95% CI over photos; '*' = CI excludes 0.", ""]
    cols = [p for p in PURIFIERS if ((e["editor"] == ed) & (e["purifier"] == p)).any()]
    lines.append("| method | " + " | ".join(cols) + " |")
    lines.append("|---|" + "---|" * len(cols))
    base = {p: per_photo_psr(e, "clean", ed, p, thr[ed], 0.55) for p in cols}
    for m in methods:
        row = []
        for p in cols:
            a_ = per_photo_psr(e, m, ed, p, thr[ed], 0.55)
            row.append(cell(a_, None if m == "clean" else base[p], rng))
        n = max((len(per_photo_psr(e, m, ed, p, thr[ed], 0.55)) for p in cols), default=0)
        lines.append(f"| {m} (n={n}) | " + " | ".join(row) + " |")
    lines += ["", "## Paired difference to v1 (same photos, same cleanup)", "", "| method | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    v1 = {p: per_photo_psr(e, "v1", ed, p, thr[ed], 0.55) for p in cols}
    for m in methods:
        if m in ("clean", "v1"):
            continue
        row = []
        for p in cols:
            a_ = per_photo_psr(e, m, ed, p, thr[ed], 0.55)
            ids = a_.index.intersection(v1[p].index)
            if len(ids) == 0:
                row.append("-")
                continue
            d = (a_[ids] - v1[p][ids]).to_numpy()
            lo, hi = boot_ci(d, rng)
            row.append(f"{d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]{'*' if lo > 0 or hi < 0 else ''}")
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    lines += ["", "## Other editors and strengths, no cleanup (PSR; net over clean in brackets)", "",
              "| method | SD1.5 s0.3 | SD1.5 s0.75 | IP-Adapter FaceID | InstructPix2Pix | inpainting | SDXL |", "|---|---|---|---|---|---|---|"]
    spec = [(ed, 0.3), (ed, 0.75), ("ipadapter_faceid", None), ("ip2p", None), ("sd_inpaint", None), ("sdxl_img2img", 0.55)]
    cb = [per_photo_psr(e, "clean", x, "none", thr[x], s) for x, s in spec]
    for m in methods:
        row = [cell(per_photo_psr(e, m, x, "none", thr[x], s), None if m == "clean" else cb[i], rng) for i, (x, s) in enumerate(spec)]
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    # invisibility
    pr = pd.read_csv(comb / "protect.csv", low_memory=False)
    pr = pr[pr["split"] == a.split].drop_duplicates(["method", "image_id"])
    lines += ["", "## Invisibility of the protected photos (published PNG; rule: PSNR >= 38 dB and SSIM >= 0.95 on every photo)", "",
              "| method | n | PSNR mean (min) | SSIM mean (min) | photos < 38 dB | photos SSIM < 0.95 | max pixel change |", "|---|---|---|---|---|---|---|"]
    for m in methods:
        d = pr[pr["method"] == m]
        if d.empty:
            continue
        ps, ss = pd.to_numeric(d["png_psnr"]), pd.to_numeric(d["png_ssim"])
        lines.append(f"| {m} | {len(d)} | {ps.mean():.2f} ({ps.min():.2f}) | {ss.mean():.4f} ({ss.min():.4f}) | {(ps < 38).sum()} | {(ss < 0.95).sum()} | {pd.to_numeric(d['png_maxabs_u8']).max():.0f} |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
