"""Protect a folder of images (folder in -> protected PNG folder out), with checkpoint/resume.

python scripts/run_protect.py --config config/config.yaml --method v1 --input_dir <dir>
       [--list data/splits/v1_eval70.txt] [--limit 5] --run_id <id> [--output_root results]
       [--shard K/N] [--resume] [--resume_from <prev_run_dir>] [--checkpoint_every 5]
       [--backend real|tiny] [--device auto|cuda|cpu] [--no_lpips] [--set budget.iters=2 ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.eval.metrics import LPIPSScorer, png_quality_metrics, v1_quality_metrics  # noqa: E402
from src.eval.schema import PROTECT_COLUMNS  # noqa: E402
from src.eval.stats import summarize  # noqa: E402
from src.protect.registry import build_protector  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.io import list_images, load_tensor_v1, parse_shard, save_png_v1, shard_items  # noqa: E402
from src.utils.runlog import finish_run, make_run_id, start_run, utc_now  # noqa: E402
from src.utils.seeding import derive_seed  # noqa: E402

CSV = "metrics.csv"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO / "config" / "config.yaml"))
    p.add_argument("--method", default="v1")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--list", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run_id", default=None)
    p.add_argument("--output_root", default=str(REPO / "results"))
    p.add_argument("--shard", default=None, help="K/N")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_from", default=None)
    p.add_argument("--checkpoint_every", type=int, default=None)
    p.add_argument("--backend", default="real", choices=["real", "tiny"])
    p.add_argument("--device", default="auto")
    p.add_argument("--no_lpips", action="store_true")
    p.add_argument("--split", default="")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    return p.parse_args(argv)


def process_one(item, protector, ctx, lpips, cfg, split=""):
    image_id = item.stem
    seed = derive_seed(cfg["seed"], "protect", image_id)
    x = load_tensor_v1(item, cfg["data"]["image_size"])
    dev = getattr(protector, "device", "cpu")
    with ctx.gpu_timer() as t:
        x_adv, info = protector.protect(x.to(dev), seed)
    out_path = ctx.images_dir / f"{image_id}.png"
    save_png_v1(x_adv, out_path)
    x_c, x_p = x.to(dev), x_adv.to(dev)
    m1 = v1_quality_metrics(x_c, x_p, lpips)
    m2 = png_quality_metrics(item, out_path, lpips)
    b = cfg["budget"]
    row = {
        "run_id": ctx.run_id, "method": cfg.get("_method") or "", "method_cfg_hash": ctx.cfg_hash,
        "image_id": image_id, "split": split, "seed": seed, "editor": "none", "prompt_id": "none",
        "strength": "", "purifier": "none",
        "src_file": item.name, "out_file": out_path.name, "eps_255": b["eps_255"], "step_255": b["step_255"],
        "iters": b["iters"],
        "v1_psnr": m1["PSNR"], "v1_ssim": m1["SSIM"], "v1_lpips": m1["LPIPS"],
        "v1_linf_model": m1["Linf"],
        "png_psnr": m2["psnr"], "png_ssim": m2["ssim"], "png_lpips": m2["lpips"],
        "png_maxabs_u8": m2["maxabs_u8"], "png_frac_at_bound": m2["frac_at_bound"],
        "gpu_seconds": t.seconds, "peak_mem_mb": t.peak_mem_mb, "created_utc": utc_now(),
    }
    assert list(row.keys()) == PROTECT_COLUMNS
    return row


def _summary(ctx, csv_path):
    df = pd.read_csv(csv_path)
    s = {"run_id": ctx.run_id, "n_images": int(len(df)), "git": ctx.git, "metrics": {}}
    for c in ("v1_psnr", "v1_ssim", "v1_lpips", "v1_linf_model", "png_psnr", "png_ssim", "png_lpips",
              "png_maxabs_u8", "gpu_seconds"):
        if c in df:
            s["metrics"][c] = summarize(df[c].to_numpy(dtype=float))
    return s


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # frozen v1 code prints unicode (L-inf); Windows consoles are cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = parse_args(argv)
    argv_list = list(sys.argv[1:] if argv is None else argv)
    overrides = list(args.overrides) + [f"runtime.backend={args.backend}"]
    cfg = load_config(args.config, method=args.method, overrides=overrides)
    if args.checkpoint_every:
        cfg["runtime"]["checkpoint_every"] = args.checkpoint_every
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = bool(cfg["runtime"].get("cudnn_benchmark", False))
    shard = parse_shard(args.shard)

    run_id = args.run_id or make_run_id("p0", args.method, "run", "nogit")
    out_root = Path(args.output_root)
    run_dir = out_root / run_id
    if args.resume_from:
        prev = Path(args.resume_from)
        if prev.resolve() != run_dir.resolve():
            (run_dir / "images").mkdir(parents=True, exist_ok=True)
            for f in prev.glob("metrics*.csv"):
                if not (run_dir / f.name).exists():
                    shutil.copy2(f, run_dir / f.name)
            if (prev / "config.json").exists() and not (run_dir / "config.json").exists():
                shutil.copy2(prev / "config.json", run_dir / "config.json")  # start_run checks hash equality
            if (prev / "images").is_dir():
                for f in (prev / "images").glob("*.png"):
                    if not (run_dir / "images" / f.name).exists():
                        shutil.copy2(f, run_dir / "images" / f.name)
    ctx = start_run(cfg, run_id, out_root, "scripts/run_protect.py", argv_list, cfg["seed"], shard=shard,
                    repo_root=REPO)
    ctx.log(f"run {run_id} method={args.method} device={device} backend={args.backend} shard={shard} "
            f"git={ctx.git['hash'][:7]} dirty={ctx.git['dirty']}")

    items = list_images(args.input_dir, args.list)
    if args.limit:
        items = items[: args.limit]
    my_items = shard_items(items, shard)
    done = ctx.done_ids(CSV)
    todo = [it for it in my_items if not (it.stem in done and (ctx.images_dir / f"{it.stem}.png").exists())]
    ctx.log(f"{len(my_items)} images in this shard ({len(items)} total), {len(my_items) - len(todo)} already done")

    if todo:
        protector = build_protector(args.method, cfg, device, backend=args.backend)
        lpips = None if args.no_lpips else LPIPSScorer(device)
        ck = int(cfg["runtime"].get("checkpoint_every", 5))
        t0 = dt.datetime.now()
        for i, it in enumerate(todo, 1):
            row = process_one(it, protector, ctx, lpips, cfg, args.split)
            ctx.append_row(CSV, row)
            if i % ck == 0 or i == len(todo):
                el = (dt.datetime.now() - t0).total_seconds()
                ctx.write_progress(done=len(my_items) - len(todo) + i, total=len(my_items),
                                   gpu_seconds_total=ctx.gpu_seconds_total, last_image=it.stem,
                                   eta_s=el / i * (len(todo) - i))
                ctx.log(f"[{len(my_items) - len(todo) + i}/{len(my_items)}] {it.stem} "
                        f"psnr={row['v1_psnr']:.2f} ssim={row['v1_ssim']:.4f} gpu_s={row['gpu_seconds']:.1f}")
        del protector
    p = ctx.csv_path(CSV)
    if p.exists():
        ctx.gpu_seconds_total = float(pd.read_csv(p)["gpu_seconds"].sum())  # all sessions incl. resumed rows
        finish_run(ctx, _summary(ctx, p))
    ctx.log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
