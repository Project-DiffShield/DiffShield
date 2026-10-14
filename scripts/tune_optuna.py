#!/usr/bin/env python
"""Phase 6 protection-aware tuning of v2_final with Optuna (V2_PLAN_48H section 2).

    python scripts/tune_optuna.py --storage sqlite:///results/optuna/v2_final.db --study v2_final \
        --n-trials 16 --n-images 10 --worker-id w1            # run the same command in a 2nd process = 2 workers
    python scripts/tune_optuna.py ... --confirm-top 3 --confirm-images 20   # after the search

Plan: 16 trials x 10 validation images, PROXY objective (20 PGD iters, K=2 EoT draws, 1 edit seed,
editors SD1.5 img2img s=0.55 + IP-Adapter FaceID, purifiers {none, jpeg_q75, diffpure_t0.1}); the top 3
feasible trials (plus the hand-set default, trial 0) are then re-run on all 20 validation images with the
final protection settings (budget.iters / eot.k_draws of the resolved v2_final config).
Objective = mean PSR (higher is better) over editors x purifiers; constraints (Optuna TPE
``constraints_func``): mean PSNR >= 38 and mean SSIM >= 0.95 of the protected uint8 images vs the
clean images (PSNR/SSIM computed here with src/protect/losses; the SSIM is the 11x11 Gaussian-window
SSIM on [0,1] RGB, which may differ slightly from the harness's skimage SSIM -> re-check the winner with
the harness metric).

Searched: alpha, eta (log-uniform 0.05-1), beta = L_sem weight in [0, 1.5] (0 switches L_sem off),
gamma (log-uniform 20-150), antipure on/off and, if on, one shared weight for w_hf = w_mono in
[0.1, 1.0] (our choice). NOT searched: eps (fixed by the budget), lambda_id (kept 0; --tune-id adds it
in [0, 1] only if a face embedder is installed), K, iterations.

Resumable: study uses ``load_if_exists``; storage is SQLite with a long busy timeout so two worker
processes can share it; crashed (heartbeat-stale) trials are failed and retried once; the global trial
budget counts COMPLETE + RUNNING + WAITING trials, so workers stop together. The hand-set default
(v2_final_default values) is enqueued as trial 0 so "tuned beats hand-set" is a measured comparison.

------------------------------------------------------------------------------------------------
SCORER INTERFACE. The study calls a generic ``score_fn(sources, protected, *, spec, device) -> dict``
    sources/protected : lists of PIL RGB images (clean / protected, uint8 round-tripped), same order
    spec              : {"editors": [{"name","strength"}], "purifiers": [...], "seeds": [...],
                         "image_names": [...], "cache_dir": ...}
    returns           : {"psr": float (objective), "per": {...optional breakdown...}}
The default (``--score-fn harness``) is ``HarnessScorer`` below, an adapter over A3's
``src.eval.edit_runner.score_protected(images_protected, images_clean, purifiers, editors, seeds, *, face,
threshold, clean_edit_cache, ...)`` (read from branch v2/eval, not yet merged here). It lazily builds the
FaceScorer (``src.eval.faces.FaceScorer.from_config``) and the loaded editors
(``src.eval.editors.build_editor(name, cfg, face).load(device)``, first prompt of the editor's prompt list), keeps them
and the clean-edit cache alive across trials, and reports ``psr`` and ``per_cell`` from the harness.
PSR threshold: pass ``--psr-threshold`` (precompute once from E0 clean edits on val; the harness's fallback
re-derives it from only n val images and is noisy => unstable objective). Purifiers requiring weights that are
missing (e.g. diffpure) raise NotAvailable from the harness; the trial then fails (no silent drop).
Any other scorer: ``--score-fn module:function``. ``--stub-score`` = plumbing test only (psr 0.0, meaningless).
------------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SPEC = {
    "editors": [{"name": "sd15_img2img", "strength": 0.55}, {"name": "ipadapter_faceid"}],
    "purifiers": ["none", "jpeg_q75", "diffpure_t0.1"],
    "seeds": [1],
    "split": "val",
}
PROTECTOR_COMPONENTS = None   # tests inject tiny stand-in models here (v2_registry `components`)
IMAGE_SIZE = 512
HAND_SET = {"alpha": 0.27, "eta": 0.27, "beta": 0.90, "gamma": 60.5, "antipure": False, "ap_w": 0.5}
PSNR_MIN, SSIM_MIN = 38.0, 0.95


# --------------------------------------------------------------------------------------------
# config / data helpers
# --------------------------------------------------------------------------------------------
def load_cfg(path: str) -> dict:
    """config/config.yaml deep-merged with config/parts/*.yaml (sorted), same rule as A1's load_config.
    Tries A1's loader first, falls back to a local equivalent."""
    for mod, fn in (("src.utils.config", "load_config"), ("src.config", "load_config"), ("src.utils", "load_config")):
        try:
            return dict(getattr(importlib.import_module(mod), fn)(path))
        except Exception:
            continue
    import yaml

    from src.protect.v2_registry import deep_merge

    p = Path(path)
    cfg = yaml.safe_load(open(p, encoding="utf-8")) if p.exists() else {}
    for part in sorted((p.parent / "parts").glob("*.yaml")):
        cfg = deep_merge(cfg, yaml.safe_load(open(part, encoding="utf-8")) or {})
    return cfg


def image_seed(study: str, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{study}:{name}".encode()).digest()[:8], "big") & 0x7FFFFFFF


def load_val_images(list_path: str, image_dir: str, n: int, size: int = 512):
    """First n names of the val list -> [(name, PIL RGB image at size x size)]."""
    from PIL import Image

    names = [ln.strip() for ln in open(list_path, encoding="utf-8") if ln.strip() and not ln.startswith("#")]
    out = []
    for nm in names[:n]:
        pth = Path(image_dir) / nm
        if not pth.exists():  # split lists hold CelebA-HQ names (.jpg); prepare_data writes <stem>.png
            for ext in (".png", ".jpg", ".jpeg"):
                if (Path(image_dir) / (Path(nm).stem + ext)).exists():
                    pth = Path(image_dir) / (Path(nm).stem + ext)
                    break
        im = Image.open(pth).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        out.append((nm, im))
    if len(out) < n:
        raise SystemExit(f"val list {list_path} has only {len(out)} images, need {n}")
    return out


def pil_to_tensor(im, device):
    import numpy as np
    import torch

    a = torch.from_numpy(np.asarray(im).astype("float32")).permute(2, 0, 1)[None]
    return (a / 127.5 - 1.0).to(device)


def tensor_to_pil(x):
    import numpy as np
    from PIL import Image

    u = ((x[0].detach().float().cpu() + 1.0) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(np.ascontiguousarray(u))


# --------------------------------------------------------------------------------------------
# trial -> protection config
# --------------------------------------------------------------------------------------------
def suggest(trial, tune_id: bool = False) -> dict:
    p = {
        "alpha": trial.suggest_float("alpha", 0.05, 1.0, log=True),
        "eta": trial.suggest_float("eta", 0.05, 1.0, log=True),
        "beta": trial.suggest_float("beta", 0.0, 1.5),
        "gamma": trial.suggest_float("gamma", 20.0, 150.0, log=True),
        "antipure": trial.suggest_categorical("antipure", [False, True]),
    }
    p["ap_w"] = trial.suggest_float("ap_w", 0.1, 1.0) if p["antipure"] else 0.5
    if tune_id:
        p["lambda_id"] = trial.suggest_float("lambda_id", 0.0, 1.0)
    return p


def params_to_override(p: dict, iters=None, k=None) -> dict:
    ov = {"loss": {"alpha": p["alpha"], "eta": p["eta"], "beta": p["beta"], "gamma": p["gamma"],
                   "lambda_id": p.get("lambda_id", 0.0),
                   "anti_purify": {"enabled": bool(p["antipure"]), "w_hf": p["ap_w"], "w_mono": p["ap_w"]}}}
    if iters is not None:
        ov["budget"] = {"iters": int(iters)}
    if k is not None:
        ov["eot"] = {"k_draws": int(k)}
    return ov


def evaluate_params(p, *, cfg, images, device, study, score_fn, spec, iters, k, protector_cache=None):
    """Protect all images with params p, score them. Returns dict with psr, psnr, ssim, per, time."""
    from src.protect.v2_registry import build_v2_protector, deep_merge

    run_cfg = deep_merge(cfg, params_to_override(p, iters, k))
    # the override must beat both the tuned layer and the default: pass it via methods.v2_final
    run_cfg["methods"] = deep_merge(run_cfg.get("methods") or {}, {"v2_final": params_to_override(p, iters, k)})
    prot = build_v2_protector("v2_final", run_cfg, device, PROTECTOR_COMPONENTS)
    sources, outs, psnrs, ssims, t0 = [], [], [], [], time.time()
    for name, im in images:
        x = pil_to_tensor(im, device)
        x_adv, info = prot.protect(x, image_seed(study, name))
        sources.append(im)
        outs.append(tensor_to_pil(x_adv))  # uint8 roundtrip == what is saved as PNG
        psnrs.append(info["psnr"])
        ssims.append(info["ssim"])
    t_prot = time.time() - t0
    sp = dict(spec, image_names=[n for n, _ in images], cache_dir=(cfg.get("paths") or {}).get("cache_dir", "results/cache"))
    sc = score_fn(sources, outs, spec=sp, device=device)
    return {"psr": float(sc["psr"]), "per": sc.get("per", {}), "psnr": sum(psnrs) / len(psnrs),
            "ssim": sum(ssims) / len(ssims), "psnr_min": min(psnrs), "ssim_min": min(ssims),
            "t_protect_s": t_prot, "t_total_s": time.time() - t0}


def constraints(psnr: float, ssim: float):
    return [PSNR_MIN - psnr, SSIM_MIN - ssim]   # <= 0 means feasible


class HarnessScorer:
    """Adapter over A3's src.eval.edit_runner.score_protected (see module docstring)."""

    def __init__(self, cfg: dict, threshold=None):
        self.cfg, self.threshold = cfg, threshold
        self.face = None
        self.editors = {}
        self.clean_cache = {}

    def _editor_specs(self, spec, device):
        from src.eval.edit_runner import EditorSpec
        from src.eval.editors import build_editor

        ev = self.cfg["eval"]
        out = []
        for e in spec["editors"]:
            if e["name"] not in self.editors:
                self.editors[e["name"]] = build_editor(e["name"], self.cfg, self.face).load(device)
            ed = self.editors[e["name"]]
            prompt = ev["prompts"][ev["editors"][e["name"]]["prompts"]][0]
            st = e.get("strength", ev["editors"][e["name"]].get("default_strength"))
            out.append(EditorSpec(ed, prompt, st, name=e["name"]))
        return out

    def __call__(self, sources, protected, *, spec, device):
        from src.eval.edit_runner import score_protected
        from src.eval.faces import FaceScorer

        if self.face is None:
            self.face = FaceScorer.from_config(self.cfg).load()
        names = spec["image_names"]
        res = score_protected(dict(zip(names, protected)), dict(zip(names, sources)), spec["purifiers"],
                              self._editor_specs(spec, device), spec["seeds"], face=self.face,
                              threshold=self.threshold, clean_edit_cache=self.clean_cache)
        return {"psr": res["psr"], "per": {k: v.get("psr") for k, v in res.get("per_cell", {}).items()},
                "threshold": res.get("threshold"), "no_face_rate": res.get("no_face_rate")}


def resolve_score_fn(spec_str: str, stub: bool, cfg: dict = None, threshold=None):
    if spec_str == "harness" and not stub:
        return HarnessScorer(cfg or {}, threshold)
    if stub:
        def _stub(sources, protected, *, spec, device):
            print("[tune_optuna] STUB scorer: psr=0.0, no meaning", file=sys.stderr)
            return {"psr": 0.0, "per": {}}
        return _stub
    mod, fn = spec_str.split(":")
    try:
        return getattr(importlib.import_module(mod), fn)
    except Exception as e:
        raise SystemExit(f"cannot import scorer {spec_str!r}: {e}\nSee the INTERFACE section at the top of {__file__}")


# --------------------------------------------------------------------------------------------
def make_study(args):
    import random
    import time

    import optuna

    # S06: two workers creating the SQLite tables at the same instant raised an OperationalError in one of them
    # (that worker died, leaving one GPU idle). Retry the whole creation with jitter; the second attempt sees the tables.
    last = None
    for attempt in range(8):
        try:
            storage = optuna.storages.RDBStorage(
                args.storage, engine_kwargs={"connect_args": {"timeout": 120}},
                heartbeat_interval=60, grace_period=600,
                failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=1))
            sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup, constant_liar=True,
                                                 constraints_func=lambda t: t.user_attrs.get("constraints", [0.0, 0.0]))
            return optuna.create_study(study_name=args.study, storage=storage, sampler=sampler,
                                       direction="maximize", load_if_exists=True)
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[{getattr(args, 'worker_id', '?')}] study creation failed ({type(e).__name__}); retry {attempt + 1}/8", flush=True)
            time.sleep(2 + 3 * attempt + random.random() * 3)
    raise last


def n_started(study) -> int:
    import optuna

    S = optuna.trial.TrialState
    return sum(1 for t in study.get_trials(deepcopy=False) if t.state in (S.COMPLETE, S.RUNNING, S.WAITING))


def search(args, cfg, score_fn):
    import optuna

    study = make_study(args)
    if n_started(study) == 0:
        study.enqueue_trial({**HAND_SET, **({"lambda_id": 0.0} if args.tune_id else {})}, skip_if_exists=True)
    images = load_val_images(args.val_list, args.image_dir, args.n_images, IMAGE_SIZE)
    spec = dict(DEFAULT_SPEC, seeds=list(args.edit_seeds))

    def objective(trial):
        p = suggest(trial, args.tune_id)
        res = evaluate_params(p, cfg=cfg, images=images, device=args.device, study=args.study,
                              score_fn=score_fn, spec=spec, iters=args.iters, k=args.k_draws)
        trial.set_user_attr("constraints", constraints(res["psnr"], res["ssim"]))
        for k in ("psnr", "ssim", "psnr_min", "ssim_min", "t_protect_s", "t_total_s"):
            trial.set_user_attr(k, res[k])
        trial.set_user_attr("per", res["per"])
        trial.set_user_attr("worker", args.worker_id)
        trial.set_user_attr("feasible", all(c <= 0 for c in trial.user_attrs["constraints"]))
        print(f"[{args.worker_id}] trial {trial.number}: psr={res['psr']:.4f} psnr={res['psnr']:.2f} "
              f"ssim={res['ssim']:.4f} params={p}", flush=True)
        return res["psr"]

    while n_started(study) < args.n_trials or any(
            t.state == optuna.trial.TrialState.WAITING for t in study.get_trials(deepcopy=False)):
        study.optimize(objective, n_trials=1)
        if n_started(study) >= args.n_trials and not any(
                t.state == optuna.trial.TrialState.WAITING for t in study.get_trials(deepcopy=False)):
            break
    return study


def feasible_complete(study):
    import optuna

    return sorted([t for t in study.get_trials(deepcopy=False)
                   if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("feasible")],
                  key=lambda t: -t.value)


def confirm(args, cfg, score_fn, study):
    """Re-run the top-k feasible trials and the hand-set default on the full val set with the final
    protection settings; write results + a final.yaml candidate (NOT copied into config/parts)."""
    out_dir = Path(args.out_dir) / args.study
    out_dir.mkdir(parents=True, exist_ok=True)
    images = load_val_images(args.val_list, args.image_dir, args.confirm_images, IMAGE_SIZE)
    spec = dict(DEFAULT_SPEC, seeds=list(args.edit_seeds))
    top = feasible_complete(study)[: args.confirm_top]
    cands = [(f"trial_{t.number}", {**HAND_SET, **t.params}) for t in top]
    if not any(t.number == 0 for t in top):
        cands.append(("hand_set", dict(HAND_SET)))
    results = {}
    rfile = out_dir / "confirm.json"
    if rfile.exists():
        results = json.load(open(rfile))
    for name, p in cands:
        if name in results:
            continue
        r = evaluate_params(p, cfg=cfg, images=images, device=args.device, study=args.study, score_fn=score_fn,
                            spec=spec, iters=args.confirm_iters, k=args.confirm_k)
        r["params"] = p
        r["feasible"] = r["psnr"] >= PSNR_MIN and r["ssim"] >= SSIM_MIN
        results[name] = r
        json.dump(results, open(rfile, "w"), indent=1)
        print(f"confirm {name}: psr={r['psr']:.4f} psnr={r['psnr']:.2f} ssim={r['ssim']:.4f} feasible={r['feasible']}")
    ok = {k: v for k, v in results.items() if v["feasible"]}
    if not ok:
        print("no confirmed candidate satisfies PSNR>=38 and SSIM>=0.95; no final.yaml candidate written")
        return results
    best = max(ok, key=lambda k: ok[k]["psr"])
    p = ok[best]["params"]
    ov = params_to_override(p)
    import yaml

    with open(out_dir / "final_candidate.yaml", "w", encoding="utf-8") as f:
        f.write(f"# candidate for config/parts/final.yaml ({best}; confirmed on {args.confirm_images} val images; "
                f"hand-set psr={results.get('hand_set', results.get('trial_0', {})).get('psr')})\n")
        yaml.safe_dump({"methods": {"v2_final": ov}}, f, sort_keys=False)
    print(f"best confirmed: {best} psr={ok[best]['psr']:.4f}; wrote {out_dir / 'final_candidate.yaml'}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--storage", default="sqlite:///results/optuna/v2_final.db")
    ap.add_argument("--study", default="v2_final")
    ap.add_argument("--n-trials", type=int, default=16, help="global trial budget across all workers")
    ap.add_argument("--n-images", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--k-draws", type=int, default=2)
    ap.add_argument("--edit-seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--val-list", default="data/splits/val.txt")
    ap.add_argument("--image-dir", default="data/images")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worker-id", default="w1")
    ap.add_argument("--seed", type=int, default=2026, help="sampler seed")
    ap.add_argument("--startup", type=int, default=5, help="random TPE startup trials")
    ap.add_argument("--tune-id", action="store_true", help="also search lambda_id in [0,1] (needs a face embedder)")
    ap.add_argument("--score-fn", default="harness", help="harness | module:function")
    ap.add_argument("--psr-threshold", type=float, default=None, help="ArcFace-sim PSR threshold from E0 (recommended)")
    ap.add_argument("--stub-score", action="store_true", help="plumbing test only: psr=0.0 stub scorer")
    ap.add_argument("--confirm-top", type=int, default=0, help=">0: run the confirmation stage instead of the search")
    ap.add_argument("--confirm-images", type=int, default=20)
    ap.add_argument("--confirm-iters", type=int, default=None, help="default: resolved v2_final budget.iters")
    ap.add_argument("--confirm-k", type=int, default=None, help="default: resolved v2_final eot.k_draws")
    ap.add_argument("--out-dir", default="results/optuna")
    ap.add_argument("--image-size", type=int, default=512)
    args = ap.parse_args(argv)
    global IMAGE_SIZE
    IMAGE_SIZE = args.image_size

    if args.storage.startswith("sqlite:///"):
        Path(args.storage[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(args.config)
    score_fn = resolve_score_fn(args.score_fn, args.stub_score, cfg, args.psr_threshold)
    if args.confirm_top > 0:
        study = make_study(args)
        confirm(args, cfg, score_fn, study)
    else:
        study = search(args, cfg, score_fn)
        out_dir = Path(args.out_dir) / args.study
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
        except Exception as e:  # pandas missing
            print("could not write trials.csv:", e)
        feas = feasible_complete(study)
        print(f"done: {n_started(study)} trials; feasible={len(feas)}; "
              f"best feasible={'none' if not feas else (feas[0].number, round(feas[0].value, 4))}")


if __name__ == "__main__":
    main()
