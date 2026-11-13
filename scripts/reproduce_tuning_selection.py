#!/usr/bin/env python
"""Re-derive the committed tuning photo lists from the saved T01 embeddings and check they are identical.

    python scripts/reproduce_tuning_selection.py --npz <path to pool_embeddings.npz>

The npz is the output of Kaggle job T01 (T01/pool/pool_embeddings.npz inside the downloaded T01_outputs zip). It is
derived data and is not stored in git; its sha256 is recorded in data/splits/tune_meta.json and is checked first.

What it does (nothing is written into the repo):
  1. copies the committed test and messy lists into a temp dir; ae_pool and v1_calib are written EMPTY, i.e. the final
     rule "exclude only test and messy" (the AE pool and the 4 calibration photos are free);
  2. runs scripts/select_tuning_photos.py with the committed arguments (28 search, 8 + 4 confirm, seed 2026,
     1000 random subsets, 200 coverage subsets);
  3. compares the produced tune_search.txt / tune_confirm.txt with data/splits/ and prints MATCH or DIFFERENT.
Takes about a minute on a laptop CPU (about 1 min measured on the dev machine; yours may differ).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def lines(p: Path):
    return [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    a = ap.parse_args()
    meta = json.load(open(SPLITS / "tune_meta.json", encoding="utf-8"))
    got = sha256_file(Path(a.npz))
    print("npz sha256:", got)
    print("expected  :", meta["npz_sha256"])
    if got != meta["npz_sha256"]:
        print("DIFFERENT INPUT: the embeddings file is not the one the lists were made from.")
        return 2
    args = meta["args"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "in").mkdir()
        for nm in ("test", "messy", "val", "train"):
            if (SPLITS / f"{nm}.txt").exists():
                shutil.copy(SPLITS / f"{nm}.txt", td / "in" / f"{nm}.txt")
        for nm in ("ae_pool", "v1_calib"):          # released: nothing excluded for these two
            (td / "in" / f"{nm}.txt").write_text("", encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / "select_tuning_photos.py"), "--npz", a.npz,
               "--splits_dir", str(td / "in"), "--out_splits_dir", str(td / "out"), "--report_dir", str(td / "rep"),
               "--n_search", str(args["n_search"]), "--n_confirm_protos", str(args["n_confirm_protos"]),
               "--n_confirm_crit", str(args["n_confirm_crit"]), "--seed", str(args["seed"]),
               "--n_random", str(args["n_random"]), "--n_cov_random", str(args["n_cov_random"]),
               "--bandwidth_n", str(args["bandwidth_n"]), "--id_thr", str(args["id_thr"]),
               "--crit_reg", str(args["crit_reg"]), "--chunk", str(args["chunk"])]
        print("running:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print("selection script failed, rc =", rc)
            return rc
        ok = True
        for nm in ("tune_search", "tune_confirm"):
            same = lines(td / "out" / f"{nm}.txt") == lines(SPLITS / f"{nm}.txt")
            print(f"{nm}: {'MATCH' if same else 'DIFFERENT'}")
            ok &= same
    print("RESULT:", "MATCH - the committed lists are exactly reproduced" if ok else "DIFFERENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
