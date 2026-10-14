"""Summarise GPU time per run from results/*/gpu_time*.json and kaggle_session*.json.

python scripts/gpu_ledger.py [--results results] -> results/gpu_ledger.csv (+ weekly totals printed)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    a = ap.parse_args(argv)
    root = Path(a.results)
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        gpu = sum(json.loads(f.read_text()).get("total_gpu_seconds") or 0 for f in d.glob("gpu_time*.json"))
        sess = [json.loads(f.read_text()) for f in d.glob("kaggle_session*.json")]
        sec = sum(s.get("session_seconds") or 0 for s in sess)
        ts = dt.datetime.fromtimestamp(d.stat().st_mtime)
        iso = ts.isocalendar()
        gname = next((json.loads(f.read_text()).get("gpu_name") for f in d.glob("gpu_time*.json")), None)
        rows.append({"run_id": d.name, "iso_week": f"{iso[0]}-W{iso[1]:02d}", "session_seconds": sec,
                     "gpu_seconds": gpu, "gpu_name": gname})
    out = root / "gpu_ledger.csv"
    root.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "iso_week", "session_seconds", "gpu_seconds", "gpu_name"])
        w.writeheader()
        w.writerows(rows)
    wk = defaultdict(float)
    for r in rows:
        wk[r["iso_week"]] += r["gpu_seconds"]
    for k, v in sorted(wk.items()):
        print(f"{k}: {v / 3600:.2f} GPU-hours (sum of per-image protect time; session time is in the csv)")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
