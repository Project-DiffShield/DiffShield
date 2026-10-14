#!/usr/bin/env python
"""Materialise the split images as 512 px PNGs in a working dir (job J02, CPU) + optional caches.

    python scripts/prepare_data.py --celebahq_root /kaggle/input/<x> --splits_dir data/splits --out_dir /kaggle/working/data_v2

For every file listed in data/splits/{test,messy,train,val,ae_pool}.txt:
  * open the 1024 px CelebA-HQ jpg, resize to 512x512 with PIL BICUBIC (same resampling as v1), save LOSSLESS PNG
    (v1 re-saved its inputs as JPEG with PIL's default quality 75; v2 does not - all v1 numbers must be re-run on these PNGs
    for paired comparisons);
  * messy split ONLY: additionally apply deterministic pre-compression and a mild resize BEFORE saving, i.e. the
    "messy" photo is what a user would upload after social-media style re-encoding:
        512 -> bicubic down to s*512 (s in [0.60, 0.85]) -> JPEG (quality in [50, 70]) -> decode -> bicubic up to 512 -> PNG
    parameters are drawn per image from sha256(seed | file) and recorded in messy_degradation.json.
Outputs: <out_dir>/<split>/<stem>.png, <out_dir>/manifest.csv (split, file, png sha256, params), prepare_meta.json.

Optional (lazy; skipped with a warning if the A3 helpers / insightface are not importable):
  * <cache_dir>/arcface/<split>_sources.npz   ids, emb [N,512] (NaN if no face), bbox [N,4], kps [N,5,2], det_score, n_faces
  * <cache_dir>/masks/<id>.png                inpaint keep-face masks via src.eval.masks.MaskProvider (needs the cached bboxes;
                                              uses the CelebAMask-HQ parsing masks when --parsing_dir is given)
Originals are only read, never modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SPLITS = ["test", "messy", "train", "val", "ae_pool"]
SIZE = 512
MESSY_SCALE = (0.60, 0.85)
MESSY_JPEG_Q = (50, 70)


def _rng_for(seed: int, key: str) -> np.random.RandomState:
    h = hashlib.sha256(f"{seed}|messy|{key}".encode()).digest()
    return np.random.RandomState(int.from_bytes(h[:4], "little"))


def messy_params(seed: int, fname: str) -> dict:
    r = _rng_for(seed, fname)
    return {"scale": float(round(r.uniform(*MESSY_SCALE), 3)), "jpeg_quality": int(r.randint(MESSY_JPEG_Q[0], MESSY_JPEG_Q[1] + 1))}


def load_512(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)


def degrade_messy(img512, params: dict):
    """Down-up resize around a JPEG round trip (all PIL, deterministic given params)."""
    from PIL import Image
    s = max(8, int(round(SIZE * params["scale"])))
    small = img512.resize((s, s), Image.BICUBIC)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=int(params["jpeg_quality"]))
    buf.seek(0)
    dec = Image.open(buf).convert("RGB")
    return dec.resize((SIZE, SIZE), Image.BICUBIC)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_split(split: str, files: List[str], img_dir: str, out_dir: str, seed: int) -> List[dict]:
    os.makedirs(os.path.join(out_dir, split), exist_ok=True)
    rows = []
    for fn in files:
        stem = os.path.splitext(fn)[0]
        dst = os.path.join(out_dir, split, stem + ".png")
        params = messy_params(seed, fn) if split == "messy" else {}
        if not os.path.exists(dst):  # never overwrite
            im = load_512(os.path.join(img_dir, fn))
            if split == "messy":
                im = degrade_messy(im, params)
            tmp = dst + ".tmp"
            im.save(tmp, format="PNG")
            os.replace(tmp, dst)
        rows.append({"split": split, "file": fn, "png": os.path.relpath(dst, out_dir).replace("\\", "/"),
                     "sha256": sha256_file(dst), "params": json.dumps(params) if params else ""})
    return rows


def cache_arcface(split: str, rows: List[dict], out_dir: str, cache_dir: str, scorer) -> Dict[str, Optional[tuple]]:
    from PIL import Image
    n = len(rows)
    emb = np.full((n, 512), np.nan, dtype=np.float32)
    bbox = np.full((n, 4), np.nan, dtype=np.float32)
    kps = np.full((n, 5, 2), np.nan, dtype=np.float32)
    det = np.full((n,), np.nan, dtype=np.float32)
    nf = np.zeros((n,), dtype=np.int32)
    ids, bboxes = [], {}
    for i, r in enumerate(rows):
        ids.append(os.path.splitext(r["file"])[0])
        res = scorer.analyze(Image.open(os.path.join(out_dir, r["png"])).convert("RGB"))
        nf[i] = int(getattr(res, "n_faces", 0))
        if res.emb is not None:
            emb[i], det[i] = res.emb, res.det_score
            bbox[i] = res.bbox
            bboxes[ids[-1]] = tuple(float(v) for v in res.bbox)
            if getattr(res, "kps", None) is not None:
                kps[i] = res.kps
    p = os.path.join(cache_dir, "arcface", f"{split}_sources.npz")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, ids=np.array(ids), emb=emb, bbox=bbox, kps=kps, det_score=det, n_faces=nf)
    return bboxes


def cache_masks(rows: List[dict], cache_dir: str, parsing_dir: Optional[str], bboxes: Dict[str, tuple]) -> int:
    from src.eval.masks import MaskProvider
    mp = MaskProvider(cache_dir=os.path.join(cache_dir, "masks"), parsing_dir=parsing_dir, size=SIZE,
                      face_bbox_fn=lambda iid: bboxes.get(os.path.splitext(iid)[0]))
    done = 0
    for r in rows:
        try:
            mp(os.path.splitext(r["file"])[0])
            done += 1
        except Exception as e:  # e.g. no parsing mask and no face bbox
            print(f"[prepare_data] mask failed for {r['file']}: {e}", file=sys.stderr)
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celebahq_root", default=None)
    ap.add_argument("--splits_dir", default=os.path.join(ROOT, "data", "splits"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cache_dir", default=None, help="default: <out_dir>/cache")
    ap.add_argument("--parsing_dir", default=None, help="CelebAMask-HQ-mask-anno dir (for inpaint masks)")
    ap.add_argument("--no_cache", action="store_true", help="only write the PNGs")
    a = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from make_splits import find_celebahq, read_list

    img_dir, attr_path = find_celebahq(a.celebahq_root)
    parsing_dir = a.parsing_dir
    if parsing_dir is None:
        cand = os.path.join(os.path.dirname(img_dir), "CelebAMask-HQ-mask-anno")
        parsing_dir = cand if os.path.isdir(cand) else None
    cache_dir = a.cache_dir or os.path.join(a.out_dir, "cache")
    os.makedirs(a.out_dir, exist_ok=True)

    manifest: List[dict] = []
    by_split: Dict[str, List[dict]] = {}
    for sp in a.splits:
        files = read_list(os.path.join(a.splits_dir, f"{sp}.txt"))
        rows = prepare_split(sp, files, img_dir, a.out_dir, a.seed)
        by_split[sp] = rows
        manifest += rows
        print(f"[prepare_data] {sp}: {len(rows)} PNGs")
    with open(os.path.join(a.out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "file", "png", "sha256", "params"])
        w.writeheader()
        w.writerows(manifest)
    if "messy" in by_split:
        with open(os.path.join(a.out_dir, "messy_degradation.json"), "w", encoding="utf-8") as f:
            json.dump({"scale_range": MESSY_SCALE, "jpeg_quality_range": MESSY_JPEG_Q, "seed": a.seed,
                       "per_image": {r["file"]: json.loads(r["params"]) for r in by_split["messy"]}}, f, indent=1)

    meta = {"size": SIZE, "format": "png (lossless; v1 used JPEG q75)", "resample": "PIL BICUBIC", "seed": a.seed,
            "celebahq_img_dir": img_dir, "parsing_dir": parsing_dir, "caches": {}}
    if not a.no_cache:
        try:
            from src.eval.faces import FaceScorer
            scorer = FaceScorer(prefer_gpu=False).load()
        except Exception as e:
            print(f"[prepare_data] ArcFace cache skipped (A3 FaceScorer / insightface unavailable): {e!r}", file=sys.stderr)
            scorer = None
        if scorer is not None:
            bboxes: Dict[str, tuple] = {}
            for sp, rows in by_split.items():
                bboxes.update(cache_arcface(sp, rows, a.out_dir, cache_dir, scorer))
                print(f"[prepare_data] cached ArcFace sources for {sp}")
            meta["caches"]["arcface"] = os.path.join(cache_dir, "arcface")
            try:
                n = sum(cache_masks(rows, cache_dir, parsing_dir, bboxes) for rows in by_split.values())
                meta["caches"]["masks"] = {"dir": os.path.join(cache_dir, "masks"), "n": n}
            except Exception as e:
                print(f"[prepare_data] mask cache skipped (src.eval.masks unavailable): {e!r}", file=sys.stderr)
    with open(os.path.join(a.out_dir, "prepare_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
