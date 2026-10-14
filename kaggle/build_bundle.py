"""Build the single code zip that is uploaded once as a Kaggle Dataset (no git push needed).

python kaggle/build_bundle.py [--allow_dirty] [--out kaggle/dist]
-> kaggle/dist/diffshield-code-<hash7>.zip  (+ kaggle/dist/diffshield-code-latest.txt with the file name)

Contents: src/ scripts/ config/ data/splits data/v1_ref kaggle/{jobs.yaml,kaggle_job.py} requirements*.txt
requirements.d pytest.ini tests/ BUNDLE_INFO.json. Never includes images, results, .git, .venv, data/v1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = ["src", "scripts", "config", "tests", "requirements.d", "data/splits", "data/v1_ref"]
INCLUDE_FILES = ["requirements.txt", "requirements-dev.txt", "pytest.ini", "kaggle/jobs.yaml",
                 "kaggle/kaggle_job.py", "kaggle/README.md", "kaggle/job_runner.ipynb"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv", ".git"}
SKIP_SUFFIX = {".pyc", ".png", ".jpg", ".jpeg", ".zip", ".pt", ".pth", ".safetensors", ".bin", ".npz"}


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_dirty", action="store_true")
    ap.add_argument("--out", default=str(REPO / "kaggle" / "dist"))
    a = ap.parse_args(argv)
    h = git("rev-parse", "HEAD") or "unknown"
    dirty = bool(git("status", "--porcelain"))
    if dirty and not a.allow_dirty:
        print("working tree is dirty; commit first or pass --allow_dirty (recorded in BUNDLE_INFO.json)",
              file=sys.stderr)
        return 1
    files = []
    for d in INCLUDE_DIRS:
        for p in sorted((REPO / d).rglob("*")) if (REPO / d).exists() else []:
            if p.is_file() and not (set(p.parts) & SKIP_PARTS) and p.suffix.lower() not in SKIP_SUFFIX:
                files.append(p)
    for f in INCLUDE_FILES:
        if (REPO / f).exists():
            files.append(REPO / f)
    rel = sorted({p.relative_to(REPO).as_posix() for p in files})
    info = {"git_hash": h, "dirty": dirty, "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": {r: hashlib.sha256((REPO / r).read_bytes()).hexdigest() for r in rel}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"diffshield-code-{h[:7]}{'-dirty' if dirty else ''}.zip"
    with zipfile.ZipFile(out / name, "w", zipfile.ZIP_DEFLATED) as z:
        for r in rel:
            z.write(REPO / r, r)
        z.writestr("BUNDLE_INFO.json", json.dumps(info, indent=2))
    (out / "diffshield-code-latest.txt").write_text(name + "\n")
    print(f"wrote {out / name} ({len(rel)} files, git {h[:7]}, dirty={dirty})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
