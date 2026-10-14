"""Merge requirements.d/*.txt into requirements.txt (integration step).

python scripts/merge_requirements.py [--write]
Adds every requirement from requirements.d/<agent>.txt that is not already in requirements.txt (by package name)
under a '# added from requirements.d' block. Conflicting version specs for one package are reported and not merged.
Dry run unless --write.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def name_of(line):
    m = re.match(r"^([A-Za-z0-9_.\-]+)", line.strip())
    return re.sub(r"[-_.]+", "-", m.group(1)).lower() if m else None


def read_reqs(path):
    out = {}
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        s = l.split("#")[0].strip()
        if s and not s.startswith("-"):
            out[name_of(s)] = s
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    main_p = REPO / "requirements.txt"
    base = read_reqs(main_p)
    add, conflicts = {}, []
    for f in sorted((REPO / "requirements.d").glob("*.txt")):
        for n, spec in read_reqs(f).items():
            if n in base:
                if base[n] != spec:
                    conflicts.append((f.name, spec, base[n]))
            elif n in add and add[n][1] != spec:
                conflicts.append((f.name, spec, add[n][1]))
            else:
                add[n] = (f.name, spec)
    for c in conflicts:
        print("CONFLICT:", c)
    if add:
        block = ["", "# added from requirements.d (merge_requirements.py)"] + [f"{s}  # {f}" for f, s in add.values()]
        print("\n".join(block))
        if a.write:
            with open(main_p, "a", encoding="utf-8") as fh:
                fh.write("\n".join(block) + "\n")
            print("written")
    else:
        print("nothing to add")
    return 1 if conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
