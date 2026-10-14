"""Compare installed package versions with the `==` pins in requirements.txt; print CUDA/GPU info.

python scripts/check_env.py [--requirements requirements.txt] [--allow_mismatch]
Exits non-zero when torch/torchvision/diffusers/transformers/kornia differ from a pin (unless --allow_mismatch).
Lines that are not `name==version` (ranges, comments, options) are ignored.
"""
from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CRITICAL = {"torch", "torchvision", "diffusers", "transformers", "kornia"}


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


def parse_pins(path):
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
        if m:
            pins[norm(m.group(1))] = m.group(2)
    return pins


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default=str(REPO / "requirements.txt"))
    ap.add_argument("--allow_mismatch", action="store_true")
    a = ap.parse_args(argv)
    bad = []
    for name, want in sorted(parse_pins(a.requirements).items()):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "not-installed"
        have_base = have.split("+")[0]
        ok = have_base == want.split("+")[0]
        print(f"{'OK ' if ok else 'DIFF'} {name}: pinned {want}, installed {have}")
        if not ok and name in CRITICAL:
            bad.append(name)
    try:
        import torch

        print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
            print("arch list:", torch.cuda.get_arch_list())
    except Exception as e:
        print("torch not importable:", e)
    if bad and not a.allow_mismatch:
        print("MISMATCH in critical packages:", bad, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
