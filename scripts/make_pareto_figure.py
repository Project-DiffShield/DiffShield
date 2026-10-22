"""Headline figure: protection (PSR, no cleanup, SD1.5 s0.55) against invisibility (PSNR of the protected photo), all methods, test photos.

python scripts/make_pareto_figure.py [--combined results/combined] [--out results/figures/pareto_psr_vs_psnr]
Hollow marker = the method violates the per-photo invisibility rule (SSIM < 0.95 or PSNR < 38 on at least one test photo). Error bars: 95%
bootstrap CI over photos. Only methods with rows are drawn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STYLE = {"clean": ("#999999", "o"), "noise_uniform": ("#E69F00", "X"), "noise_gauss_psnr": ("#E69F00", "P"), "v1": ("#D55E00", "D"),
         "v2_main": ("#0072B2", "s"), "v2_cheap": ("#56B4E9", "s"), "v2_final": ("#CC79A7", "^"), "v2_antipure": ("#CC79A7", "v"),
         "v2_tgt_zero": ("#009E73", "o"), "v2_tgt_zero_mid": ("#009E73", "*"), "v2_tgt_zero_hi": ("#009E73", "h"),
         "eps2": ("#999999", "<"), "eps8": ("#999999", ">")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined", default=str(REPO / "results" / "combined"))
    ap.add_argument("--out", default=str(REPO / "results" / "figures" / "pareto_psr_vs_psnr"))
    a = ap.parse_args(argv)
    comb = Path(a.combined)
    e = pd.read_csv(comb / "edits.csv", low_memory=False)
    e = e[(e["split"] == "test") & (e["editor"] == "sd15_img2img") & (e["purifier"] == "none")]
    e = e[(pd.to_numeric(e["strength"], errors="coerce") - 0.55).abs() < 1e-9]
    thr = json.loads((comb / "psr_thresholds.json").read_text())["per_editor"]["sd15_img2img"]
    pr = pd.read_csv(comb / "protect.csv", low_memory=False)
    pr = pr[pr["split"] == "test"].drop_duplicates(["method", "image_id"])
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.axvspan(30, 38, color="#dddddd", alpha=0.6, lw=0)
    ax.text(37.8, 0.02, "PSNR < 38 dB:\nvisible", ha="right", fontsize=7, color="#555555")
    for m, (c, mk) in STYLE.items():
        d = e[e["method"] == m]
        q = pr[pr["method"] == m]
        if d.empty or q.empty:
            continue
        sim = pd.to_numeric(d["arcface_sim"], errors="coerce")
        psr = (((sim < thr) | sim.isna()).astype(float)).groupby(d["image_id"].astype(str)).mean().to_numpy()
        bs = rng.choice(psr, size=(1500, len(psr))).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ps, ss = pd.to_numeric(q["png_psnr"]), pd.to_numeric(q["png_ssim"])
        bad = bool((ss < 0.95).any() or (ps < 38).any())
        ax.errorbar(ps.mean(), psr.mean(), yerr=[[psr.mean() - lo], [hi - psr.mean()]], fmt=mk, ms=9 if mk in "*h" else 7, color=c,
                    mfc="white" if bad else c, mec=c, capsize=2, lw=1)
        ax.annotate(m.replace("noise_", "noise ").replace("v2_tgt_zero", "tgt").replace("_", " "), (ps.mean(), psr.mean()), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)
    ax.set_xlabel("PSNR of the protected photo (dB; higher = less visible)")
    ax.set_ylabel("Protection success rate (SD1.5, strength 0.55, no cleanup)")
    ax.set_title("Protection vs invisibility, 100 test photos (hollow = breaks the per-photo rule)")
    ax.invert_xaxis()
    ax.set_ylim(0, 0.75)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    print("wrote", str(out) + ".png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
