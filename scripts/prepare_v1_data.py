"""Prepare the v1 evaluation inputs: 512 px JPEGs of the exact v1 image lists.

python scripts/prepare_v1_data.py --from_manifest [DIR] [--celebahq_root DIR] --out_dir <dir>
       [--which eval|calib|both] [--write_splits]

--from_manifest reads evaluation_70_manifest.csv and calibration_subset_manifest.csv (column: Filename) from DIR
(default: data/v1_ref in the repo; falls back to data/v1/phase1_zip/kaggle/working/metrics), skips KMeans, and with
--write_splits writes data/splits/v1_eval70.txt and v1_calib.txt.
Images are resized exactly like v1: PIL convert('RGB') -> resize((512,512), BICUBIC) -> saved under the original
name as JPEG with PIL's default quality 75 (docs/V2_PLAN.md F2). Nothing under data/v1/ is ever written.
KMeans re-selection from the attribute file is NOT implemented (the manifest makes it unnecessary).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402


def find_celebahq(root="/kaggle/input"):
    for r, dirs, files in os.walk(root):
        if "CelebA-HQ-img" in dirs and "CelebAMask-HQ-attribute-anno.txt" in files:
            return Path(r)
    raise FileNotFoundError(f"CelebA-HQ-img + CelebAMask-HQ-attribute-anno.txt not found under {root}")


def _manifest_dir(arg):
    d = Path(arg) if arg and arg is not True else REPO / "data" / "v1_ref"
    if (d / "evaluation_70_manifest.csv").exists():
        return d
    alt = REPO / "data" / "v1" / "phase1_zip" / "kaggle" / "working" / "metrics"
    if (alt / "evaluation_70_manifest.csv").exists():
        return alt
    raise FileNotFoundError(f"evaluation_70_manifest.csv not found in {d} or {alt}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_manifest", nargs="?", const=True, default=True)
    ap.add_argument("--celebahq_root", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--which", default="eval", choices=["eval", "calib", "both"])
    ap.add_argument("--write_splits", action="store_true", help="also (re)write data/splits/v1_*.txt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--size", type=int, default=512, help="v1 uses 512; other values only for tests")
    a = ap.parse_args(argv)

    md = _manifest_dir(a.from_manifest)
    ev = pd.read_csv(md / "evaluation_70_manifest.csv")["Filename"].tolist()
    ca = pd.read_csv(md / "calibration_subset_manifest.csv")["Filename"].tolist()
    if a.write_splits:
        sp = REPO / "data" / "splits"
        sp.mkdir(parents=True, exist_ok=True)
        (sp / "v1_eval70.txt").write_text("\n".join(ev) + "\n")
        (sp / "v1_calib.txt").write_text("\n".join(ca) + "\n")
    names = {"eval": ev, "calib": ca, "both": ev + ca}[a.which]
    if a.limit:  # same ordering as src.utils.io.list_images (sorted by stem) so --limit matches run_protect
        names = sorted(names, key=lambda n: (Path(n).stem, n))[: a.limit]

    root = Path(a.celebahq_root) if a.celebahq_root else find_celebahq()
    img_dir = root / "CelebA-HQ-img" if (root / "CelebA-HQ-img").is_dir() else root
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for fname in names:
        src = img_dir / fname
        if not src.exists():
            raise FileNotFoundError(src)
        img = Image.open(src).convert("RGB").resize((a.size, a.size), Image.Resampling.BICUBIC)
        img.save(out / fname)  # .jpg => PIL default quality 75, as in v1
        n += 1
    print(f"wrote {n} images (manifest dir {md}) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
