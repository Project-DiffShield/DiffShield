"""Merge metrics.shard*.csv (and shard gpu_time/summary) into metrics.csv / summary.json / gpu_time.json.

python scripts/merge_shards.py --run_dir results/<run_id> [--list data/splits/v1_eval70.txt] [--also <other run dirs>]
`--also` merges shards from other run directories with the same run_id (e.g. jobs on different accounts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.eval.stats import summarize  # noqa: E402

METRIC_COLS = ("v1_psnr", "v1_ssim", "v1_lpips", "v1_linf_model", "png_psnr", "png_ssim", "png_lpips",
               "png_maxabs_u8", "gpu_seconds")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--list", default=None)
    ap.add_argument("--also", nargs="*", default=[])
    a = ap.parse_args(argv)
    run = Path(a.run_dir)
    dirs = [run] + [Path(p) for p in a.also]
    files = sorted(f for d in dirs for f in d.glob("metrics.shard*.csv"))
    if not files:
        files = [run / "metrics.csv"] if (run / "metrics.csv").exists() else []
    if not files:
        print("no shard files found", file=sys.stderr)
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df.duplicated(["image_id", "method", "editor", "prompt_id", "strength", "purifier"], keep="last")
    if dup.any():
        print(f"dropping {int(dup.sum())} duplicate rows (keeping the last)")
        df = df[~dup]
    df = df.sort_values("image_id").reset_index(drop=True)
    if a.list:
        want = {Path(l.strip()).stem for l in Path(a.list).read_text(encoding="utf-8").splitlines() if l.strip()}
        have = set(df.image_id.astype(str))
        missing = sorted(want - have)
        extra = sorted(have - want)
        if missing:
            print(f"WARNING: {len(missing)} ids missing from merged results: {missing[:10]}")
        if extra and a.list:
            print(f"note: {len(extra)} ids not in list (e.g. --limit runs): {extra[:5]}")
    df.to_csv(run / "metrics.csv", index=False)
    gts = [json.loads(f.read_text()) for d in dirs for f in sorted(d.glob("gpu_time*.json"))]
    gts = [g for g in gts if "total_gpu_seconds" in g]
    total = float(df["gpu_seconds"].sum()) if "gpu_seconds" in df else float("nan")
    gt = {"total_gpu_seconds": total, "n_images": int(len(df)),
          "per_image_mean": total / len(df) if len(df) else None,
          "peak_mem_mb": max([g.get("peak_mem_mb") or 0 for g in gts] or [0]),
          "gpu_name": next((g.get("gpu_name") for g in gts if g.get("gpu_name")), None),
          "n_shard_files": len(files),
          "max_shard_wall_seconds": max([g.get("wall_seconds") or 0 for g in gts] or [0])}
    (run / "gpu_time.json").write_text(json.dumps(gt, indent=2))
    summ = {"run_id": run.name, "n_images": int(len(df)),
            "metrics": {c: summarize(df[c].to_numpy(dtype=float)) for c in METRIC_COLS if c in df}}
    (run / "summary.json").write_text(json.dumps(summ, indent=2))
    print(f"merged {len(files)} file(s) -> {run / 'metrics.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
