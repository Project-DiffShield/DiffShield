#!/usr/bin/env python
"""Continuous identity analysis of the exploration runs (docs/EXPLORATION.md). CPU only, reads existing outputs.

PSR is a yes/no count against a strict threshold (0.1005), so it cannot see a variant that lowers identity similarity
without crossing the threshold. This script reports, per variant and editor:
  * mean ArcFace similarity (edit vs source) with a paired bootstrap 95% interval of the difference to X0
    (per-photo means over seeds/prompts/cleanups, photos resampled, 2000 draws, seed 2026),
  * the lowest per-edit similarity (how close any edit came to the threshold),
  * for identity-loss variants, the surrogate (FaceNet) cosine at the first and last PGD iteration,
    read from the protect run's info/*.json ``terms.id`` trace (what the optimiser actually achieved).

  python scripts/explore_similarity.py results/X0_run1 results/X3_run1 ... [--out results/explore/similarity.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_compare import LABELS, NAMES  # noqa: E402

THRESH = 0.1005


def load_edits(d: str) -> pd.DataFrame:
    fs = [f for f in glob.glob(os.path.join(d, "**", "metrics*.csv"), recursive=True)
          if "eval_" in f and "arcface_sim" in open(f, encoding="utf-8").readline()]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def id_trace(d: str):
    first, last = [], []
    for f in glob.glob(os.path.join(d, "**", "prot_*", "info", "*.json"), recursive=True):
        t = (json.load(open(f, encoding="utf-8")).get("terms") or {}).get("id") or []
        if t:
            first.append(t[0])
            last.append(t[-1])
    return (float(np.mean(first)), float(np.mean(last)), len(last)) if last else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results/explore/similarity.md")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args(argv)

    runs, traces = {}, {}
    for d in a.dirs:
        e = load_edits(d)
        if e.empty:
            continue
        lab = LABELS.get(str(e["method"].iloc[0]), str(e["method"].iloc[0]))
        runs[lab] = e
        traces[lab] = id_trace(d)
    if "X0" not in runs:
        sys.exit("X0 (v2_next) is needed as the reference")

    rng = np.random.default_rng(a.seed)
    L = ["# Identity similarity of the exploration variants (continuous, complements PSR)", "",
         f"ArcFace (buffalo_l) similarity between edit and source; lower = less like the person. PSR counts an edit as "
         f"protected only below {THRESH}. Exploration on the 28 tune_search photos, not a test-set claim. "
         f"Difference to X0: paired by photo, {a.n_boot} bootstrap draws, seed {a.seed}, percentile 95% interval.", "",
         "| variant | name | editor | mean sim | diff to X0 [95% CI] | lowest edit sim | surrogate FaceNet cos first -> last iter |",
         "|---|---|---|---|---|---|---|"]
    for ed in sorted(runs["X0"]["editor"].unique()):
        b = runs["X0"][runs["X0"].editor == ed].groupby("image_id").arcface_sim.mean()
        for lab in sorted(runs):
            x = runs[lab][runs[lab].editor == ed]
            v = x.groupby("image_id").arcface_sim.mean()
            if lab == "X0":
                ci = "-"
            else:
                diff = (v - b).dropna().values
                bs = [rng.choice(diff, len(diff)).mean() for _ in range(a.n_boot)]
                ci = f"{diff.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (n={len(diff)})"
            tr = traces.get(lab)
            trs = f"{tr[0]:.2f} -> {tr[1]:.2f} (n={tr[2]})" if tr else "-"
            L.append(f"| {lab} | {NAMES.get(lab, lab)} | {ed} | {v.mean():.3f} | {ci} | {x.arcface_sim.min():.3f} | {trs} |")
    txt = "\n".join(L) + "\n"
    print(txt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
