#!/usr/bin/env python
"""Restore an earlier protection run (folder ``prot_<method>_test`` with images/*.png) into this job's results folder,
so an edit job can evaluate it without re-protecting. Used by Kaggle jobs W1/W2 (docs/KAGGLE_LAUNCH.md).

  python scripts/restore_prot.py --src /kaggle/input/<dataset or output> --name prot_v2_next_tuned_test \
         --dst /kaggle/working/W1/results/prot_v2_next_tuned_test --n_expect 100 [--fallback_root /kaggle/input]

--src is searched recursively for a folder called --name (e.g. the T04 output ``T04/results/prot_v2_next_tuned_test``
or the S28 output dataset). If --src is empty the --fallback_root is searched instead. Several hits: the one with the most
PNGs is used (ties: sorted path). Fails loudly (exit 1) if no folder is found or it holds fewer than --n_expect PNGs in
``images/`` (never silently evaluates a partial set).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def count_pngs(d: Path) -> int:
    img = d / "images"
    return len([f for f in os.listdir(img) if f.lower().endswith(".png")]) if img.is_dir() else 0


def find_candidates(root: str, name: str) -> List[Tuple[int, Path]]:
    hits = []
    for r, dirs, _ in os.walk(root):
        if os.path.basename(r) == name:
            hits.append((count_pngs(Path(r)), Path(r)))
            dirs[:] = []  # do not descend into the match
    return sorted(hits, key=lambda t: (-t[0], str(t[1])))


def restore(src: str, name: str, dst: str, n_expect: int, fallback_root: Optional[str] = None) -> int:
    root = src or fallback_root or ""
    if not root or not os.path.isdir(root):
        print(f"ERROR: --src is empty or not a folder ({src!r}) and no usable fallback ({fallback_root!r}); "
              f"pass the folder that holds {name}", file=sys.stderr)
        return 1
    hits = find_candidates(root, name)
    for n, p in hits:
        print(f"candidate: {p} ({n} PNGs)")
    if not hits:
        print(f"ERROR: no folder named {name} under {root}", file=sys.stderr)
        return 1
    n, best = hits[0]
    if n < n_expect:
        print(f"ERROR: best candidate {best} has {n} PNGs in images/, expected at least {n_expect}", file=sys.stderr)
        return 1
    shutil.copytree(best, dst, dirs_exist_ok=True)
    got = count_pngs(Path(dst))
    if got < n_expect:
        print(f"ERROR: after copy {dst} has {got} PNGs, expected at least {n_expect}", file=sys.stderr)
        return 1
    print(f"restored {got} protected PNGs from {best} to {dst}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--n_expect", type=int, default=100)
    ap.add_argument("--fallback_root", default="/kaggle/input")
    a = ap.parse_args(argv)
    return restore(a.src, a.name, a.dst, a.n_expect, a.fallback_root)


if __name__ == "__main__":
    sys.exit(main())
