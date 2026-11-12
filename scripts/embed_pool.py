#!/usr/bin/env python
"""Embed the CelebAMask-HQ pool for tuning-photo selection (job T01, Kaggle GPU).

Writes ``pool_embeddings.npz`` (derived features only, NO images; keep it private, CelebA terms apply):
    names      [N]      CelebA-HQ file names ('1234.jpg'), numerically sorted
    clip       [N,768]  float16, L2-normalised CLIP image embeddings (``models.clip_eval`` in config/config.yaml)
    arcface    [N,512]  float16, L2-normalised ArcFace (insightface buffalo_l); NaN row if no face was found
    has_face   [N]      bool
    attr       [N,40]   int8 in {0,1} (CelebAMask-HQ-attribute-anno.txt, via make_splits.parse_attr_file)
    attr_names [40]
    pose       [N,3]    float32 (yaw, pitch, roll from CelebAMask-HQ-pose-anno.txt; NaN if missing)

Evaluation convention (matched on purpose): each 1024 px JPEG is converted to RGB and resized to 512x512 BICUBIC
first, exactly as ``make_splits.make_face_cache`` does for ArcFace; the SAME 512 px image is then given to
  * ArcFace through ``src.eval.faces.FaceScorer.analyze`` (largest face, 0.4 border padding, buffalo_l), and
  * CLIP through ``src.eval.clip_metric.ClipScorer.embed`` (CLIPProcessor, fp16 on GPU, projected + L2-normalised
    image embedding), i.e. the model/pipeline used for the CLIP-I metric in evaluation.
Neither scoring rule is reimplemented here.

Resumable: every ``--chunk`` images are saved to ``--work_dir/chunk_<start>.npz`` and re-used on a rerun.
``--limit N`` embeds only the first N images (smoke test).
insightface / transformers / torch are imported only when ``main`` runs the models, so the module imports on CPU.
Not exercised locally (GPU path and model downloads): UNVERIFIED until the Kaggle smoke run.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CLIP_DIM, ARC_DIM = 768, 512
DEFAULT_CLIP_ID = "openai/clip-vit-large-patch14"   # config/config.yaml models.clip_eval


def _name_key(n: str):
    stem = os.path.splitext(n)[0]
    return (0, int(stem), n) if stem.isdigit() else (1, 0, n)


def clip_id_from_config(config_path: Optional[str] = None) -> str:
    """CLIP id from config/config.yaml ``models.clip_eval.id`` (falls back to the documented default)."""
    path = config_path or os.path.join(ROOT, "config", "config.yaml")
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return str(cfg["models"]["clip_eval"]["id"])
    except Exception:
        return DEFAULT_CLIP_ID


def merge_chunks(chunk_paths: List[str]) -> dict:
    parts = [dict(np.load(p, allow_pickle=False)) for p in chunk_paths]
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--celebahq_root", default=None, help="root searched for CelebA-HQ-img/ (default /kaggle/input)")
    ap.add_argument("--out", default="pool_embeddings.npz", help="output npz")
    ap.add_argument("--work_dir", default=None, help="chunk directory (default <out>.chunks)")
    ap.add_argument("--chunk", type=int, default=2000, help="images per resumable chunk")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N images (smoke); 0 = all")
    ap.add_argument("--batch", type=int, default=32, help="CLIP batch size")
    ap.add_argument("--workers", type=int, default=4, help="image-loading threads")
    ap.add_argument("--config", default=None, help="config.yaml (for models.clip_eval.id)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (slow; debugging only)")
    args = ap.parse_args(argv)

    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from scripts.make_splits import find_celebahq, find_file, parse_attr_file, parse_pose_anno

    img_dir, attr_path = find_celebahq(args.celebahq_root)
    roots = [args.celebahq_root] if args.celebahq_root else ["/kaggle/input"]
    pose_path = find_file("CelebAMask-HQ-pose-anno.txt", roots + [os.path.dirname(attr_path)])
    fns, attr_names, attr = parse_attr_file(attr_path)
    pose_map = parse_pose_anno(pose_path) if pose_path else {}
    if not pose_path:
        print("[embed_pool] pose annotation file not found: pose = NaN everywhere", file=sys.stderr)
    on_disk = set(os.listdir(img_dir))
    order = sorted((i for i, f in enumerate(fns) if f in on_disk), key=lambda i: _name_key(fns[i]))
    if args.limit > 0:
        order = order[:args.limit]
    names = [fns[i] for i in order]
    attr = attr[order].astype(np.int8)
    pose = np.asarray([pose_map.get(n, (np.nan,) * 3) for n in names], dtype=np.float32).reshape(-1, 3)
    n = len(names)
    print(f"[embed_pool] {n} images from {img_dir}; pose rows found: {int(np.isfinite(pose[:, 0]).sum())}")

    work = args.work_dir or (args.out + ".chunks")
    os.makedirs(work, exist_ok=True)

    import torch
    from src.eval.clip_metric import ClipScorer
    from src.eval.faces import FaceScorer
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    clip = ClipScorer(clip_id_from_config(args.config), device=dev)
    face = FaceScorer(prefer_gpu=(dev == "cuda"))
    clip.load()
    face.load()

    def load(name):
        im = Image.open(os.path.join(img_dir, name)).convert("RGB")
        return im.resize((512, 512), Image.BICUBIC)

    chunk_paths = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for start in range(0, n, args.chunk):
            cp = os.path.join(work, f"chunk_{start:06d}.npz")
            chunk_paths.append(cp)
            if os.path.exists(cp):
                print(f"[embed_pool] chunk {start} present, skipping")
                continue
            sub = names[start:start + args.chunk]
            clips, arcs, has = [], [], []
            for b in range(0, len(sub), args.batch):
                ims = list(ex.map(load, sub[b:b + args.batch]))
                clips.append(clip.embed(ims))
                for im in ims:
                    emb, _, _ = face.embed(im)
                    has.append(emb is not None)
                    if emb is None:
                        arcs.append(np.full(ARC_DIM, np.nan, np.float32))
                    else:
                        e = np.asarray(emb, np.float32).ravel()
                        arcs.append(e / max(float(np.linalg.norm(e)), 1e-12))
            tmp = cp + ".tmp.npz"
            np.savez(tmp, clip=np.concatenate(clips).astype(np.float16), arcface=np.stack(arcs).astype(np.float16),
                     has_face=np.asarray(has, bool))
            os.replace(tmp, cp)
            done = min(start + args.chunk, n)
            print(f"[embed_pool] {done}/{n} ({time.time() - t0:.0f} s)")

    m = merge_chunks(chunk_paths)
    assert m["clip"].shape == (n, CLIP_DIM) and m["arcface"].shape == (n, ARC_DIM), (m["clip"].shape, m["arcface"].shape)
    np.savez(args.out, names=np.asarray(names), clip=m["clip"], arcface=m["arcface"], has_face=m["has_face"],
             attr=attr, attr_names=np.asarray(attr_names), pose=pose)
    print(f"[embed_pool] wrote {args.out}: {n} images, {int(m['has_face'].sum())} with a face")
    return 0


if __name__ == "__main__":
    sys.exit(main())
