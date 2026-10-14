"""Measure the effective L-inf budget on the standard 0-255 scale (CPU only).

python scripts/budget_check.py --clean_dir <dir> --protected_dir <dir> [--pairs_csv stage1_per_image_metrics.csv]
       [--run_id ID --output_root results | --out_dir DIR]
Pairs by file stem by default. --pairs_csv pairs by (Original_Filename -> Protected_Filename), which is how v1's
shuffled output names must be matched (docs/V2_PLAN.md F5).
Writes budget_check.json + budget_check.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.io import load_u8  # noqa: E402

EXTS = (".png", ".jpg", ".jpeg")


def collect_pairs(clean_dir, protected_dir, pairs_csv=None):
    clean_dir, protected_dir = Path(clean_dir), Path(protected_dir)
    pairs = []
    if pairs_csv:
        df = pd.read_csv(pairs_csv)
        for _, r in df.iterrows():
            pairs.append((clean_dir / r["Original_Filename"], protected_dir / r["Protected_Filename"]))
    else:
        prot = {p.stem: p for p in protected_dir.iterdir() if p.suffix.lower() in EXTS}
        for c in sorted(clean_dir.iterdir()):
            if c.suffix.lower() in EXTS and c.stem in prot:
                pairs.append((c, prot[c.stem]))
    return pairs


def check(pairs, max_level=16):
    per, hist = [], np.zeros(max_level + 1, dtype=np.int64)
    gmax, jpeg_clean = 0, 0
    diffs = []
    for c, p in pairs:
        if c.suffix.lower() in (".jpg", ".jpeg"):
            jpeg_clean += 1
        a, b = load_u8(c), load_u8(p)
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch {c.name} {a.shape} vs {p.name} {b.shape}")
        d = np.abs(a.astype(np.int64) - b.astype(np.int64))
        m = int(d.max())
        per.append({"clean": c.name, "protected": p.name, "max_abs_diff": m,
                    "frac_at_own_max": float((d == m).mean())})
        hist += np.bincount(np.minimum(d.ravel(), max_level), minlength=max_level + 1)
        gmax = max(gmax, m)
        diffs.append(d)
    at_max = sum(int((d == gmax).sum()) for d in diffs)
    total = sum(d.size for d in diffs)
    return {"n_pairs": len(pairs), "global_max_abs_diff": gmax, "inferred_eps_255": gmax,
            "share_of_values_at_global_max": at_max / total if total else None,
            "histogram_abs_diff_0_to_16plus": {int(i): int(hist[i]) for i in range(max_level + 1)},
            "n_images_with_max_equal_global": sum(1 for r in per if r["max_abs_diff"] == gmax),
            "clean_files_are_jpeg": jpeg_clean, "per_image": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--protected_dir", required=True)
    ap.add_argument("--pairs_csv", default=None)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--output_root", default=str(REPO / "results"))
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args(argv)
    pairs = collect_pairs(a.clean_dir, a.protected_dir, a.pairs_csv)
    if not pairs:
        print("no pairs found", file=sys.stderr)
        return 1
    res = check(pairs)
    out = Path(a.out_dir) if a.out_dir else Path(a.output_root) / (a.run_id or "budget_check")
    out.mkdir(parents=True, exist_ok=True)
    (out / "budget_check.json").write_text(json.dumps(res, indent=2))
    g = res["global_max_abs_diff"]
    concl = f"max gray-level change = {g} => effective eps = {g}/255 (standard 0-255 scale)"
    lines = ["# Budget check", "", f"- pairs: {res['n_pairs']}", f"- {concl}",
             f"- images whose own max equals the global max: {res['n_images_with_max_equal_global']}",
             f"- share of pixel values exactly at the global max: {res['share_of_values_at_global_max']:.4f}",
             "", "| abs diff | count |", "|---|---|"]
    lines += [f"| {k}{'+' if k == 16 else ''} | {v} |" for k, v in res["histogram_abs_diff_0_to_16plus"].items()]
    if res["clean_files_are_jpeg"]:
        lines += ["", f"Note: {res['clean_files_are_jpeg']} clean files are JPEG (expected for the v1 pipeline; "
                      "docs/V2_PLAN.md F2)."]
    (out / "budget_check.md").write_text("\n".join(lines) + "\n")
    print(concl)
    return 0


if __name__ == "__main__":
    sys.exit(main())
