"""Unpack any new Kaggle output zips from Downloads and run the whole analysis chain in one command.

python scripts/finalize_all.py [--downloads C:/Users/karth/Downloads] [--dry_run]

1. For every S<number>_outputs*.zip in --downloads: read <S>/kaggle_session.json from the archive; skip it when the session failed
   (ok = false) and a successful unpack of that session already exists; otherwise extract it to results/<S>/ (data files and images
   stay local, never committed).
2. scripts/ingest_results.py  (smoke sessions excluded)
3. scripts/make_tables.py     -> results/tables/
4. scripts/readout_final.py   -> results/tables/final_readout.md
5. scripts/make_figures.py    -> results/figures/ (photo panels use local copies of the images; they contain faces: keep local)
Nothing is deleted. A failing step is reported and the next one still runs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
PAT = re.compile(r"^S\d+_outputs(?: \(\d+\))?\.zip$")
SMOKE = {"S03", "S04", "S05"}   # throw-away validation sessions: never unpacked, never analysed


def session_ok(zf: zipfile.ZipFile, top: str):
    try:
        return bool(json.loads(zf.read(f"{top}/kaggle_session.json")).get("ok"))
    except Exception:
        return None


def dest_ok(dest: Path, top: str):
    f = dest / top / "kaggle_session.json"
    if not f.exists():
        return None
    try:
        return bool(json.loads(f.read_text()).get("ok"))
    except Exception:
        return None


def unpack(downloads: Path, results: Path, dry: bool) -> list:
    done = []
    for z in sorted(downloads.glob("S*_outputs*.zip"), key=lambda p: p.stat().st_mtime):
        if not PAT.match(z.name):
            continue
        with zipfile.ZipFile(z) as zf:
            tops = sorted({n.split("/")[0] for n in zf.namelist() if "/" in n})
            top = next((t for t in tops if re.fullmatch(r"S\d+", t)), None)
            if top is None:
                print(f"skip {z.name}: no S<number>/ folder inside")
                continue
            if top in SMOKE:
                print(f"skip {z.name}: smoke session")
                continue
            ok = session_ok(zf, top)
            have = dest_ok(results / top, top)
            if ok is False and have is True:
                print(f"skip {z.name}: failed session, a successful {top} is already unpacked")
                continue
            if ok is True and have is True:
                print(f"skip {z.name}: {top} already unpacked")
                continue
            print(f"{'would unpack' if dry else 'unpacking'} {z.name} -> results/{top}  (session ok={ok}, existing ok={have})")
            if not dry:
                zf.extractall(results / top)
            done.append((z.name, top, ok))
    return done


def run(cmd: list) -> int:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=REPO)
    if rc:
        print(f"!! step failed with exit code {rc}", flush=True)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downloads", default=str(Path.home() / "Downloads"))
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    results = REPO / "results"
    done = unpack(Path(a.downloads), results, a.dry_run)
    print(f"\n{len(done)} archive(s) {'to unpack' if a.dry_run else 'unpacked'}")
    if a.dry_run:
        return 0
    rcs = [run([PY, "scripts/ingest_results.py", "--results", "results"]),
           run([PY, "scripts/make_tables.py", "--combined", "results/combined", "--out", "results/tables"]),
           run([PY, "scripts/readout_final.py", "--split", "test"])]
    roots = []
    for d in ("S08/S08/results/eval_v1_H", "S09/S09/results/eval_v2_main_H", "S16/S16/results/eval_v2_final_H",
              "S26/S26/results/eval_v2_tgt_zero_hi_TGT", "S27/S27/results/eval_v2_tgt_zero_TGT", "S28/S28/results/eval_v2_tgt_zero_mid_TGT"):
        if (results / d).exists():
            roots += ["--edits_root", f"results/{d}"]
    figs = [PY, "scripts/make_figures.py", "--combined", "results/combined", "--out", "results/figures", "--source_dir", "results/S06/S06/data/test",
            "--clean_root", "results/S07/S07/cache/clean_edits", "--optuna_db", "results/S06/S06/optuna/v2_final.db", "--results", "results", "--n_boot", "1000"] + roots
    rcs.append(run(figs))
    print("\nSUMMARY: " + ", ".join(f"{n}={'ok' if r == 0 else 'FAILED'}" for n, r in zip(["ingest", "tables", "readout", "figures"], rcs)))
    return 0 if not any(rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
