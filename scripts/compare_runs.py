"""Paired per-image comparison of two runs, or of a run against v1's original per-image CSV.

python scripts/compare_runs.py --a results/<runA> --b results/<runB>
python scripts/compare_runs.py --a results/<runA> --v1_csv data/v1_ref/stage1_per_image_metrics.csv
Writes compare_<name>.json/.md into run A's directory. With --v1_csv images are paired by Original_Filename
(= src_file of the run) and the proposed Phase 0 tolerance gate is evaluated:
PSNR +-0.5 dB, SSIM +-0.005, LPIPS +-0.03 (on the mean paired difference), Linf == 8/255 on [-1,1],
PNG max abs diff == 4. The gate values are proposals (docs/V2_PLAN.md 3.11); nothing is tuned to pass them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.eval.stats import paired_bootstrap_diff  # noqa: E402

PAIRS_V1 = [("v1_psnr", "PSNR_dB", 0.5), ("v1_ssim", "SSIM", 0.005), ("v1_lpips", "LPIPS", 0.03)]
PAIRS_RUN = [("v1_psnr", "v1_psnr", None), ("v1_ssim", "v1_ssim", None), ("v1_lpips", "v1_lpips", None),
             ("png_psnr", "png_psnr", None), ("png_ssim", "png_ssim", None)]


def compare(a_df, b_df, a_key, b_key, pairs):
    m = a_df.merge(b_df, left_on=a_key, right_on=b_key, suffixes=("_a", "_b"))
    out = {"n_paired": int(len(m)), "n_a": int(len(a_df)), "n_b": int(len(b_df)), "metrics": {}}
    for ca, cb, tol in pairs:
        xa = ca + "_a" if ca + "_a" in m else ca
        xb = cb + "_b" if cb + "_b" in m else cb
        d = paired_bootstrap_diff(m[xa].to_numpy(float), m[xb].to_numpy(float))
        d["mean_a"] = float(m[xa].mean())
        d["mean_b"] = float(m[xb].mean())
        d["tolerance"] = tol
        d["within_tolerance"] = None if tol is None else abs(d["mean_diff"]) <= tol
        out["metrics"][ca] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None)
    ap.add_argument("--v1_csv", default=None)
    a = ap.parse_args(argv)
    A = pd.read_csv(Path(a.a) / "metrics.csv")
    if a.v1_csv:
        B = pd.read_csv(a.v1_csv)
        res = compare(A, B, "src_file", "Original_Filename", PAIRS_V1)
        res["linf_all_8_255"] = bool((A["v1_linf_model"].sub(8 / 255).abs() < 1e-4).all())
        res["png_max_abs_diff"] = int(A["png_maxabs_u8"].max())
        res["png_max_is_4"] = res["png_max_abs_diff"] == 4
        tols = [v["within_tolerance"] for v in res["metrics"].values()]
        res["gate_pass"] = bool(all(tols) and res["linf_all_8_255"] and res["png_max_is_4"])
        res["gate_note"] = ("Proposed tolerance, not yet accepted by the user. Note the paired set may be a subset "
                            "of v1's 70 images; v1 per-image numbers use its own RNG/image order.")
        name = "v1"
    elif a.b:
        B = pd.read_csv(Path(a.b) / "metrics.csv")
        res = compare(A, B, "image_id", "image_id", PAIRS_RUN)
        name = Path(a.b).name
    else:
        ap.error("need --b or --v1_csv")
    outp = Path(a.a)
    (outp / f"compare_{name}.json").write_text(json.dumps(res, indent=2))
    lines = [f"# Comparison {Path(a.a).name} vs {name}", "", f"paired images: {res['n_paired']}", "",
             "| metric | mean A | mean B | mean diff A-B | 95% CI | tolerance | within |", "|---|---|---|---|---|---|---|"]
    for k, v in res["metrics"].items():
        lines.append(f"| {k} | {v['mean_a']:.4f} | {v['mean_b']:.4f} | {v['mean_diff']:.4f} | "
                     f"[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}] | {v['tolerance']} | {v['within_tolerance']} |")
    if "gate_pass" in res:
        lines += ["", f"Linf == 8/255 for all images: {res['linf_all_8_255']}; PNG max abs diff: "
                      f"{res['png_max_abs_diff']}", f"**Gate (proposed tolerance): {'PASS' if res['gate_pass'] else 'FAIL'}**",
                  res["gate_note"]]
    (outp / f"compare_{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
