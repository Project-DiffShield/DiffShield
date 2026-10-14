"""Demo: protect ONE photo and show what happens when someone edits it.

  python scripts/demo.py --image my_face.jpg --out demo_out                 # method v2_final (falls back to v1), SD1.5 img2img
  python scripts/demo.py --image my_face.jpg --method v1 --editor sd15_img2img --strength 0.55 --prompt "a person at a beach"
  python scripts/demo.py --image any.png --cpu-dry-run --out demo_dry       # tiny/fake models, tests the wiring on CPU

Writes <out>/source.png, protected.png (the image you would post), edit_of_original.png, edit_of_protected.png,
panel.png (source | protected | edit of original | edit of protected, with ArcFace similarity to the source) and demo.json.
Both edits use the same generator seed, so the only difference between them is the protection.

Needs a GPU for the real models (SD1.5, insightface); protection takes ~2-3 minutes per image on a T4 (planner estimate,
unmeasured). On CPU only --cpu-dry-run is practical: it uses a tiny random VAE/CLIP for protection and fake editor/face
models, so its numbers are meaningless; it only proves the pipeline runs.
Lower ArcFace similarity after editing = the edit no longer looks like the person = protection worked. This demo is a
single example, NOT evidence: see the tables for measured results. Use only photos you have the right to use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def prepare_source(path: Path, out_dir: Path, size: int) -> Path:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)                                       # centre square crop, like portrait datasets
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.Resampling.BICUBIC)
    p = out_dir / "source.png"
    im.save(p)
    return p


def make_panel(out_dir: Path, sims: dict, psnr, ssim, title: str, dry: bool, protected_note: str = ""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [("source.png", "original"), ("protected.png", f"protected ({protected_note})"),
             ("edit_of_original.png", f"edit of original\nArcFace sim {sims['clean']}"),
             ("edit_of_protected.png", f"edit of protected\nArcFace sim {sims['protected']}")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, (f, t) in zip(axes, names):
        ax.imshow(Image.open(out_dir / f).convert("RGB"))
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    sub = f"protected vs original: PSNR {psnr:.1f} dB, SSIM {ssim:.3f}. " if psnr is not None else ""
    fig.suptitle(title + ("   [DRY RUN: fake models, numbers are meaningless]" if dry else ""), fontsize=10)
    fig.text(0.5, 0.02, sub + "Lower ArcFace similarity after editing = the edit no longer looks like the person (single example, not evidence).",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_dir / "panel.png", dpi=150)
    plt.close(fig)


def _sim(face, src_emb, img):
    r = face.analyze(img)
    if r.emb is None:
        return None
    return float(np.dot(r.emb, src_emb) / (np.linalg.norm(r.emb) * np.linalg.norm(src_emb) + 1e-12))


def _fmt(s):
    return "no face found (counts as protected)" if s is None else f"{s:.2f}"


def run(args) -> dict:
    from src.eval import edit_runner as er
    from src.eval.metrics import png_quality_metrics
    from src.protect.registry import build_protector
    from src.utils.config import load_config
    from src.utils.io import load_tensor_v1, save_png_v1
    from src.utils.seeding import derive_seed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dry = args.cpu_dry_run
    size = 64 if dry else 512
    device = "cpu" if dry else (args.device if args.device != "auto" else _auto_device())
    src_png = prepare_source(Path(args.image), out, size)
    stem = Path(args.image).stem
    cfg_path = args.config or str(REPO / "config" / "config.yaml")

    method = args.method
    used = method
    note = ""
    x = None
    for attempt in ([method] if dry else [method, "v1"] if method != "v1" else ["v1"]):
        try:
            if dry:
                ov = ["runtime.backend=tiny", "budget.iters=2", f"data.image_size={size}"]
                cfg = load_config(cfg_path, method="v1", overrides=ov)
                protector, used, note = build_protector("v1", cfg, device, backend="tiny"), "v1 (tiny dry run)", "tiny random VAE/CLIP"
            elif attempt == "v1":
                cfg = load_config(cfg_path, method="v1", overrides=list(args.set))
                protector = build_protector("v1", cfg, device, backend="real")
                used = "v1"
            else:
                cfg = load_config(cfg_path, method=None, overrides=list(args.set))
                protector = build_protector(attempt, cfg, device, backend="real")
                used = attempt
            x = load_tensor_v1(src_png, size)
            seed = derive_seed(cfg["seed"], "protect", stem)
            x_adv, info = protector.protect(x.to(device), seed)
            break
        except Exception as e:  # v2_final not tuned / weights unavailable -> v1 (documented fallback)
            print(f"[demo] method {attempt} failed ({e!r})", file=sys.stderr)
            if attempt == "v1" or dry:
                raise
            print("[demo] falling back to v1", file=sys.stderr)
    save_png_v1(x_adv.cpu(), out / "protected.png")
    del protector
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
    q = png_quality_metrics(src_png, out / "protected.png", None)

    ecfg = er.load_eval_config(cfg_path, [], REPO)
    if dry:
        be = er.fake_backends(ecfg)
        face, editor = be.face, be.editor_factory(args.editor)
    else:
        from src.eval.editors import build_editor
        from src.eval.faces import FaceScorer

        face = FaceScorer.from_config(ecfg)
        editor = build_editor(args.editor, ecfg, face_scorer=face)
        if getattr(editor, "needs_mask", False):
            raise SystemExit("the inpainting editor needs a face mask; use sd15_img2img, sdxl_img2img, ip2p or ipadapter_faceid")
    editor.load("cpu" if dry else device)
    gseed = er.derive_seed(int(cfg["seed"]), "edit", stem, 1)
    src_img = Image.open(src_png).convert("RGB")
    prot_img = Image.open(out / "protected.png").convert("RGB")
    strength = args.strength if args.strength is not None else ecfg["eval"]["editors"][args.editor].get("default_strength")
    e_clean = editor.edit_batch(src_img, args.prompt, strength, [gseed])[0]
    e_prot = editor.edit_batch(prot_img, args.prompt, strength, [gseed])[0]
    editor.unload()
    e_clean.save(out / "edit_of_original.png")
    e_prot.save(out / "edit_of_protected.png")
    s_emb = face.analyze(src_img).emb
    if s_emb is None:
        print("[demo] WARNING: no face detected in the source photo; similarities are unavailable", file=sys.stderr)
        sims = {"clean": None, "protected": None}
    else:
        sims = {"clean": _sim(face, s_emb, e_clean), "protected": _sim(face, s_emb, e_prot)}
    res = {"image": str(args.image), "method_requested": method, "method_used": used, "editor": args.editor, "strength": strength,
           "prompt": args.prompt, "edit_generator_seed": gseed, "dry_run": dry, "psnr": q["psnr"], "ssim": q["ssim"],
           "max_abs_gray_level_change": q["maxabs_u8"], "arcface_sim_to_source": sims}
    if args.thresholds and Path(args.thresholds).is_file() and not dry:
        from src.eval.analysis import threshold_for

        thr = threshold_for(json.loads(Path(args.thresholds).read_text()), args.editor)
        res["psr_threshold"] = thr
        res["protected_edit_counts_as_protected"] = bool(sims["protected"] is None or sims["protected"] < thr)
    (out / "demo.json").write_text(json.dumps(res, indent=2))
    show = {"clean": _fmt(sims["clean"]), "protected": _fmt(sims["protected"])}
    make_panel(out, show, q["psnr"], q["ssim"], f"DiffShield demo: {used}, {args.editor}", dry,
               f"max change {q['maxabs_u8']}/255")
    print(f"method {used}: PSNR {q['psnr']:.1f} dB, max change {q['maxabs_u8']} gray levels; ArcFace sim to source: "
          f"edit of original {show['clean']}, edit of protected {show['protected']}")
    print(f"wrote {out / 'panel.png'}")
    return res


def _auto_device():
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--method", default="v2_final", help="v2_final (default; falls back to v1 if unavailable) | v1 | v2_main | ...")
    ap.add_argument("--editor", default="sd15_img2img", choices=["sd15_img2img", "sdxl_img2img", "ip2p", "ipadapter_faceid"])
    ap.add_argument("--strength", type=float, default=None, help="default: the editor's default (0.55 for SD1.5 img2img)")
    ap.add_argument("--prompt", default="a person in a suit")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v", help="config overrides for the protector, e.g. budget.iters=10")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--thresholds", default=None, help="psr_thresholds.json to state whether the edit counts as protected")
    ap.add_argument("--cpu-dry-run", dest="cpu_dry_run", action="store_true", help="tiny/fake models (wiring test only)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # the frozen v1 code prints unicode (L-inf); Windows consoles are cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
