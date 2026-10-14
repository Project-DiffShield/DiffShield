"""Summarise measured cost per method/editor from smoke (or full) result folders.

python scripts/smoke_report.py --root /kaggle/working/S04/results [--out smoke_report.md]
Protect runs (metrics.csv with gpu_seconds + iters): s/image, s/image/iteration, peak memory, invisibility.
Eval runs (metrics*.csv with gpu_seconds_edit): s/edit by editor and purifier, rows, no-face rate.
Numbers are measurements of this run only; extrapolate to full runs by multiplying (iters 40, images 100/120).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def md(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "
".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    lines = ["# Cost report (measured)", ""]
    prot = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / "metrics.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "gpu_seconds" in df and "png_psnr" in df:
            it = float(df["iters"].iloc[0]) if "iters" in df else float("nan")
            prot.append({"run": d.name, "method": df["method"].iloc[0], "n": len(df), "iters": it,
                         "s_per_img": df["gpu_seconds"].mean(), "s_per_img_per_iter": df["gpu_seconds"].mean() / it,
                         "peak_mem_mb": df["peak_mem_mb"].max(), "psnr": df["png_psnr"].mean(),
                         "ssim": df["png_ssim"].mean(), "maxabs_u8": int(df["png_maxabs_u8"].max())})
        elif "gpu_seconds_edit" in df:
            g = df.groupby(["method", "editor", "purifier"]).agg(
                n=("image_id", "size"), s_edit=("gpu_seconds_edit", "mean"), s_pur=("gpu_seconds_purify", "mean"),
                no_face=("face_detected", lambda x: float((x == 0).mean()))).reset_index()
            g.insert(0, "run", d.name)
            lines += [f"## Evaluation {d.name}", "", md(g), ""]
    if prot:
        lines += ["## Protection", "", md(pd.DataFrame(prot)), ""]
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
