#!/usr/bin/env python
"""Train the E8 adaptive attacker (LightShed-style learned perturbation remover). Job J42, ~1 GPU-hour.

Inputs: a directory of PROTECTED images and a directory of the matching CLEAN images (same file stems),
both from the 120-image ``ae_pool`` split (disjoint from test / messy / train / val / calibration).
Output: ``--out`` weights (default results/adaptive_ae/ae_final.pt) + training log json next to it.

    python scripts/train_adaptive_ae.py --protected_dir results/J42/protected --clean_dir data/work/ae_pool \
        --out results/adaptive_ae/ae_final.pt --minutes 60

Resumable: pass --resume (loads <out>.ckpt). Safe to rerun after a Kaggle session stop.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protected_dir", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--out", default=None, help="final weights path (default: adaptive_ae.weights_path from config)")
    ap.add_argument("--config", default=None, help="optional YAML (e.g. config/parts/baselines.yaml) for defaults")
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lpips_weight", type=float, default=None)
    ap.add_argument("--base_channels", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--val_fraction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    import torch
    from src.eval import adaptive_ae as AE

    cfg = {}
    cpath = a.config or os.path.join(ROOT, "config", "parts", "baselines.yaml")
    if os.path.isfile(cpath):
        import yaml
        with open(cpath, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    ac = cfg.get("adaptive_ae", {}) or {}
    tc = ac.get("train", {}) or {}

    def pick(v, d, *fallback):
        return v if v is not None else d

    out = a.out or ac.get("weights_path", "results/adaptive_ae/ae_final.pt")
    base = pick(a.base_channels, ac.get("base_channels", 32))
    depth = pick(a.depth, ac.get("depth", 3))
    patch = pick(a.patch, ac.get("patch", 128))
    minutes = a.minutes if a.minutes is not None else (None if a.steps is not None else tc.get("minutes", 60))
    bs = pick(a.batch_size, tc.get("batch_size", 16))
    lr = pick(a.lr, tc.get("lr", 2e-4))
    lw = pick(a.lpips_weight, tc.get("lpips_weight", 0.0))
    vf = pick(a.val_fraction, tc.get("val_fraction", 0.1))
    device = AE._pick_device(a.device)

    pairs = AE.list_pairs(a.protected_dir, a.clean_dir)
    if not pairs:
        print("no matching (protected, clean) file stems found", file=sys.stderr)
        return 2
    stems, prot, clean = AE.load_pairs(pairs)
    tr, va = AE.split_pairs(len(stems), vf, a.seed)
    print(f"[ae] {len(stems)} pairs: {len(tr)} train / {len(va)} val; device={device}; patch={patch}")

    sampler = AE.PatchSampler([prot[i] for i in tr], [clean[i] for i in tr], patch=patch, seed=a.seed)
    val = ([prot[i] for i in va], [clean[i] for i in va]) if va else None
    model = AE.AdaptiveAE(base, depth)

    ckpt = out + ".ckpt"
    resume_state = None
    if a.resume and os.path.isfile(ckpt):
        resume_state = torch.load(ckpt, map_location="cpu")
        print(f"[ae] resuming from {ckpt} at step {resume_state.get('step')}")

    meta = {"git": _git_hash(), "seed": a.seed, "n_pairs": len(stems), "train_stems": [stems[i] for i in tr],
            "val_stems": [stems[i] for i in va], "patch": patch, "batch_size": bs, "lr": lr, "lpips_weight": lw,
            "protected_dir": a.protected_dir, "clean_dir": a.clean_dir}
    t0 = time.time()
    res = AE.train_ae(model, sampler, device, steps=a.steps, minutes=minutes, batch_size=bs, lr=lr, lpips_weight=lw,
                      log_every=tc.get("log_every", 50), ckpt_every=tc.get("ckpt_every", 500), ckpt_path=ckpt, val=val,
                      resume_state=resume_state, meta=meta, seed=a.seed)
    AE.save_checkpoint(out, model, None, res["steps"], res["elapsed_s"], meta)
    log = {"out": out, "wall_s": time.time() - t0, "steps": res["steps"], "elapsed_s": res["elapsed_s"],
           "val": res.get("val"), "arch": model.arch, "meta": meta, "history": res["history"]}
    with open(os.path.splitext(out)[0] + "_train_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=1)
    print(f"[ae] done: {res['steps']} steps; val={res.get('val')}; weights -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
