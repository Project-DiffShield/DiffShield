"""SMOKE-ONLY split lists built from the v1 evaluation list (used when the bundle has no frozen data/splits yet).

python scripts/make_smoke_splits.py --out_dir <dir>
Never use these for reported numbers: the real splits come from scripts/make_splits.py (job S03).
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLICES = {"test": (0, 10), "val": (10, 20), "messy": (20, 30), "train": (30, 40), "ae_pool": (40, 70)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--source", default=str(REPO / "data" / "splits" / "v1_eval70.txt"))
    a = ap.parse_args(argv)
    names = [l.strip() for l in Path(a.source).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, (lo, hi) in SLICES.items():
        (out / f"{k}.txt").write_text("\n".join(names[lo:hi]) + "\n", encoding="utf-8")
    (out / "SMOKE_ONLY.txt").write_text("These lists are for smoke tests only.\n", encoding="utf-8")
    print(f"wrote smoke splits to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
