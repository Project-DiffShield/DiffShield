"""Half B CLI: edit + score the protected images of one or more methods, per tier.

Examples (see docs of each flag below):
  # count what a tier would do (no models, no GPU)
  python scripts/evaluate.py --tier H --method v1 --split test --split_file data/splits/test.txt --count_only

  # E0 clean edits (also fills the clean-edit cache used by every other job)
  python scripts/evaluate.py --tier H --method clean --source_dir data/images/test --split test \
      --split_file data/splits/test.txt --out results/e0_H_shard0 --shard 0/2

  # v1 under tier H, no purifier
  python scripts/evaluate.py --tier H --method v1 --purifier none --protected_dir results/p1_v1/images \
      --source_dir data/images/test --split test --split_file data/splits/test.txt --out results/h_v1

  # purifier grid; --purifier all expands to the purifiers each matching rule of tier P lists
  python scripts/evaluate.py --tier P --method v1 --purifier all ...

  # PSR thresholds from the E0 clean-edit cache (95th pct of impostor ArcFace sims)
  python scripts/evaluate.py --stage thresholds --source_dir ... --split test --split_file ... --cache_dir <dir with clean_edits>

  # CPU dry run with fake models (no downloads): checks wiring, resume, CSV
  python scripts/evaluate.py --backend fake ...

Stages: all (edit+score, default) | thresholds | summary | lpips (fill NaN edit_lpips from saved JPEG edits).

Several jobs in one session (models stay loaded; failures are skipped and logged):
  python scripts/evaluate.py --jobs_file kaggle/eval_jobs.yaml   # YAML: defaults + jobs (name, tier, method, purifier, out, ...)

"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval import edit_runner as er  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config/config.yaml (config/parts/*.yaml are merged)")
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v", help="dotted config overrides")
    ap.add_argument("--tier", default="H", help="comma list of tiers defined in eval.tiers (H,M,P,A,E9,E8,SMOKE)")
    ap.add_argument("--method", default="clean", help="comma list of method names ('clean' = E0 reference)")
    ap.add_argument("--purifier", default="none", help="none | all | comma list of purifier names")
    ap.add_argument("--editor", default=None, help="optional comma list to restrict editors")
    ap.add_argument("--split", default="test")
    ap.add_argument("--split_file", default=None, help="text file with one image file name per line (default data/splits/<split>.txt)")
    ap.add_argument("--source_dir", default=None, help="clean 512px PNGs (default <data_root>/images/<split>)")
    ap.add_argument("--protected_dir", default=None, help="protected PNGs; may contain {method}")
    ap.add_argument("--protect_run_id", default="")
    ap.add_argument("--parsing_dir", default=None, help="CelebAMask-HQ-mask-anno dir for keep-face masks (optional)")
    ap.add_argument("--out", default=None, help="output dir (default results/<run_id>)")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--cache_dir", default=None, help="shared cache (clean_edits, arcface, masks, purified)")
    ap.add_argument("--shard", default=None, help="k/n: process images with index %% n == k")
    ap.add_argument("--limit", type=int, default=None, help="first N images of the split only")
    ap.add_argument("--resume_from", default=None, help="earlier run dir whose metrics*.csv is copied in before starting")
    ap.add_argument("--stage", default="all", choices=["all", "thresholds", "summary", "lpips"])
    ap.add_argument("--clean_edits_dir", default=None, help="stage lpips: dir with E0 clean edits (<editor>/<prompt>_s<s>_seed<k>/<id>.jpg)")
    ap.add_argument("--jobs_file", default=None, help="YAML {defaults: {...}, jobs: [{name, tier, method, ...}]} run sequentially in one session")
    ap.add_argument("--rerun", action="store_true", help="jobs file: do not skip jobs marked done")
    ap.add_argument("--fail_fast", action="store_true", help="jobs file: stop at the first failing job")
    ap.add_argument("--thresholds", default=None, help="psr_thresholds.json to use in the summary")
    ap.add_argument("--no_generate_clean", action="store_true", help="never generate missing clean edits (LPIPS -> NaN)")
    ap.add_argument("--backend", default="real", choices=["real", "fake"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--count_only", action="store_true")
    return ap.parse_args(argv)


def read_split(args, cfg) -> list:
    sf = Path(args.split_file) if args.split_file else Path(cfg.get("paths", {}).get("splits_dir", "data/splits")) / f"{args.split}.txt"
    if sf.is_file():
        ids = [Path(l.strip()).stem for l in sf.read_text().splitlines() if l.strip() and not l.startswith("#")]
        ids = sorted(ids)  # same order as src.utils.io.list_images (run_protect), so --limit N picks the same images (S05)
    else:
        src = Path(args.source_dir)
        ids = sorted({p.stem for p in src.iterdir() if p.suffix.lower() in er.IMG_EXTS})
    if args.limit:
        ids = ids[: args.limit]
    return ids


def _pick_device(d):
    if d != "auto":
        return d
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _start_run(cfg, run_id, out: Path, argv, shard):
    """Use A1's start_run/finish_run if present, else write a minimal provenance set."""
    try:  # pragma: no cover - depends on A1
        from src.utils.runlog import start_run
        # start_run writes into output_root/<name>, so pass the output folder's own name
        return start_run(cfg, out.name, str(out.parent), "scripts/evaluate.py", list(argv), int(cfg.get("seed", 2026)), shard)
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        if not (out / "config.json").exists():
            (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        (out / "git_hash.txt").write_text(er._git_hash())
        (out / "command.txt").write_text(" ".join(argv))
        return None


def _job_namespace(base, entry: dict):
    """Namespace for one jobs-file entry: CLI args, then file `defaults`, then the entry itself."""
    ns = argparse.Namespace(**vars(base))
    for k, v in entry.items():
        if k == "name":
            continue
        if k not in vars(ns):
            raise KeyError(f"unknown jobs-file key '{k}'")
        if isinstance(v, (list, tuple)) and k != "set":
            v = ",".join(str(x) for x in v)
        setattr(ns, k, v)
    return ns


def run_job(args, cfg, argv, shared: dict) -> int:
    ev = cfg["eval"]
    shard = None
    if args.shard:
        k, n = args.shard.split("/")
        shard = (int(k), int(n))
    data_root = Path(cfg.get("paths", {}).get("data_root", "data"))
    source_dir = Path(args.source_dir) if args.source_dir else data_root / "images" / args.split
    args.source_dir = str(source_dir)
    ids = read_split(args, cfg)
    methods = [m for m in args.method.split(",") if m]
    tiers = [t for t in args.tier.split(",") if t]
    pfilt = [p for p in args.purifier.split(",") if p]
    efilt = args.editor.split(",") if args.editor else None

    jobs = []
    for t in tiers:
        for m in methods:
            jobs += er.expand_tier(ev, t, m, args.split, ids, pfilt, efilt, shard)
    seeds = ev["seeds"]
    if args.count_only:
        counts = er.count_rows(jobs, len(seeds))
        print(json.dumps({"tiers": tiers, "methods": methods, "split": args.split, "n_images": len(ids),
                          "jobs": len(jobs), "rows_by_editor": counts}, indent=2))
        return 0

    run_id = args.run_id or f"eval_{'-'.join(tiers)}_{'-'.join(methods)}_{er._git_hash()[:7]}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(args.out) if args.out else Path(cfg.get("paths", {}).get("results_root", "results")) / run_id
    cache_dir = Path(args.cache_dir) if args.cache_dir else out / "cache"
    device = _pick_device(args.device)
    csv_name = f"metrics.shard{shard[0]}.csv" if shard else "metrics.csv"

    def get_face_and_lpips():
        key = ("fl", args.backend, device)
        if key not in shared:
            if args.backend == "fake":
                b = er.fake_backends(cfg)
                shared[key] = (b.face, b.lpips)
            else:
                from src.eval.clip_metric import LpipsScorer
                from src.eval.faces import FaceScorer
                shared[key] = (FaceScorer.from_config(cfg), LpipsScorer.from_config(cfg, device))
        return shared[key]

    if args.stage == "thresholds":
        face, _ = get_face_and_lpips()
        store = er.SourceStore(source_dir, args.split, cache_dir, face)
        embc = er.EmbCache(cache_dir / "arcface" / "edit_embeddings.npz")
        res = er.compute_psr_thresholds(cache_dir / "clean_edits", store, face, ids, list(ev["editors"]),
                                        float(ev["psr"]["percentile"]), embc)
        store.save()
        embc.save()
        out.mkdir(parents=True, exist_ok=True)
        (out / "psr_thresholds.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return 0

    if args.stage == "lpips":
        _, lp = get_face_and_lpips()
        ei = ev.get("edit_images", {}) or {}
        clean_root = Path(args.clean_edits_dir) if args.clean_edits_dir else cache_dir / "clean_edits"
        for pth in sorted(out.glob("metrics*.csv")):
            res = er.recompute_lpips(pth, clean_root, lp, int(ei.get("quality", 90)),
                                     ".png" if ei.get("format") == "png" else ".jpg")
            print(json.dumps({"file": str(pth), **res}))
        return 0

    if args.stage == "summary":
        thr = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
        for p in sorted(out.glob("metrics*.csv")):
            for r in er.summarize_csv(p, thr):
                print(json.dumps(r))
        return 0

    _start_run(cfg, run_id, out, argv, shard)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume_from:
        import shutil
        for p in Path(args.resume_from).glob("metrics*.csv"):
            if not (out / p.name).exists():
                shutil.copy2(p, out / p.name)

    bkey = ("be", args.backend, device, str(cache_dir), str(source_dir), args.split, args.parsing_dir)
    if bkey not in shared:       # models (face, CLIP, LPIPS) are loaded once per session and reused by later jobs
        if args.backend == "fake":
            shared[bkey] = er.fake_backends(cfg)
        else:
            store_for_masks = er.SourceStore(source_dir, args.split, cache_dir, None)
            be0 = er.default_backends(cfg, device, args.parsing_dir, str(cache_dir), store_for_masks)
            store_for_masks.face = be0.face
            shared[bkey] = be0
    be = shared[bkey]
    runner = er.EditRunner(cfg, be, out, source_dir, args.split, args.protected_dir, run_id, args.protect_run_id,
                           cache_dir, csv_name, allow_generate_clean=not args.no_generate_clean)
    res = runner.run(jobs)
    thr = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
    summ = er.summarize_csv(out / csv_name, thr) if runner.writer.n_rows else []
    (out / f"summary{('.shard' + str(shard[0])) if shard else ''}.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({"run": res, "summary": summ}, indent=2))
    return 0


def run_jobs_file(args, cfg, argv) -> int:
    """Run a list of jobs sequentially in one session (Kaggle). Models stay loaded between jobs; a failing job is
    logged and skipped (unless --fail_fast); finished jobs are recorded in <jobs_file>.status.json and skipped on rerun
    (row-level resume inside each job still applies)."""
    import traceback
    import yaml
    jf = Path(args.jobs_file)
    spec = yaml.safe_load(jf.read_text())
    entries = spec["jobs"] if isinstance(spec, dict) else spec
    defaults = spec.get("defaults", {}) if isinstance(spec, dict) else {}
    base = _job_namespace(args, defaults)
    base.jobs_file = None
    status_path = jf.with_suffix(jf.suffix + ".status.json")
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    shared: dict = {}
    failed = 0
    for i, ent in enumerate(entries):
        name = ent.get("name", f"job{i}")
        if status.get(name, {}).get("state") == "done" and not args.rerun:
            print(f"[jobs] {name}: already done, skipping")
            continue
        t0 = time.time()
        print(f"[jobs] {name}: start")
        try:
            ns = _job_namespace(base, ent)
            job_cfg = er.load_eval_config(ns.config, ns.set, REPO) if (ns.set != args.set or ns.config != args.config) else cfg
            rc = run_job(ns, job_cfg, argv, shared)
            status[name] = {"state": "done" if rc == 0 else "failed", "rc": rc, "seconds": time.time() - t0}
        except Exception as e:  # keep going: overnight sessions are limited
            traceback.print_exc()
            status[name] = {"state": "failed", "error": repr(e), "seconds": time.time() - t0}
            failed += 1
            if args.fail_fast:
                status_path.write_text(json.dumps(status, indent=2))
                return 1
        status_path.write_text(json.dumps(status, indent=2))
    return 1 if failed else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    cfg = er.load_eval_config(args.config, args.set, REPO)
    if args.jobs_file:
        return run_jobs_file(args, cfg, argv)
    return run_job(args, cfg, argv, {})


if __name__ == "__main__":
    sys.exit(main())
