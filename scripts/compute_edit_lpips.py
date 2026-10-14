"""Fill the NaN ``edit_lpips`` column locally on CPU (48h plan, fallback rule 6).

Edit LPIPS is NOT computed inside the Kaggle edit jobs. Definition used everywhere (same for every method):
  LPIPS(VGG) between the clean-image edit and the protected-image edit, BOTH saved as JPEG quality 90, same editor /
  prompt / strength / seed. Files that are not JPEG are round-tripped through JPEG q90 first.

  python scripts/compute_edit_lpips.py --combined results/combined --workers 4
      [--results results]                 # auto-discovers <root>/edits/ (protected edits) and clean_edits/ (E0) under here
      [--edits_root DIR ...] [--clean_root DIR ...]   # explicit roots instead
      [--chunk 64] [--limit N] [--backend real|fake] [--dry_run]

Resumable and safe: every finished chunk is appended to results/combined/edit_lpips_cache.csv; a re-run skips keys already
there; scripts/ingest_results.py re-applies the cache, so re-ingesting never loses values. At the end the values are written
back into results/combined/edits.csv (atomic replace). Rows whose images cannot be found are listed in
results/combined/edit_lpips_missing.csv (nothing is invented). Estimated cost (planner, not measured): ~0.2 s per pair per core.
Alternative: ``python scripts/evaluate.py --stage lpips`` (single process) or the Kaggle fallback job J50.
Edit images are CelebA-HQ derivatives: they are read locally and never copied.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from src.eval import analysis as A  # noqa: E402


def find_dirs(root: Path, name: str):
    out = []
    if not root.is_dir():
        return out
    for dp, dn, _ in os.walk(root):
        if "combined" in Path(dp).parts:
            dn[:] = []
            continue
        if name in dn:
            out.append(Path(dp) / name)
            dn.remove(name)
    return out


def build_tasks(df: pd.DataFrame, edit_roots, clean_roots, done_keys: set, limit=None):
    """Rows needing LPIPS -> ([(row_index, clean_path, edit_path)], [missing rows])."""
    need = pd.to_numeric(df["edit_lpips"], errors="coerce").isna() & ~((df["method"] == "clean") & (df["purifier"] == "none"))
    tasks, missing = [], []
    edit_dirs = [Path(r) / "edits" for r in edit_roots]
    for idx in df.index[need]:
        r = df.loc[idx]
        key = tuple(str(r[k]) if k != "strength" else A.canon_strength(r[k]) for k in A.KEY_COLS)
        if key in done_keys:
            continue
        st = A.canon_strength(r["strength"])
        ep = A.find_with_ext(edit_dirs, A.edit_rel_path(r["method"], r["purifier"], r["editor"], r["prompt_id"], st, r["seed"], r["image_id"]))
        cp = A.find_with_ext(clean_roots, A.clean_rel_path(r["editor"], r["prompt_id"], st, r["seed"], r["image_id"]))
        if ep is None or cp is None:
            missing.append({**{k: r[k] for k in A.KEY_COLS}, "reason": ("edit image " if ep is None else "") + ("clean edit " if cp is None else "") + "not found"})
            continue
        tasks.append((int(idx), str(cp), str(ep)))
        if limit and len(tasks) >= limit:
            break
    return tasks, missing


def run_pool(tasks, workers: int, chunk: int, backend: str, quality: int, on_chunk, log=print):
    """Compute LPIPS for the tasks; ``on_chunk([(idx, value)])`` is called after every finished chunk."""
    chunks = [tasks[i:i + chunk] for i in range(0, len(tasks), chunk)]
    t0 = time.time()
    done = 0
    if workers <= 1:
        A.lpips_worker_init(backend, 1, quality)
        for c in chunks:
            res = A.lpips_chunk(c)
            on_chunk(res)
            done += len(c)
            _progress(log, done, len(tasks), t0)
        return
    with cf.ProcessPoolExecutor(max_workers=workers, initializer=A.lpips_worker_init, initargs=(backend, 1, quality)) as ex:
        futs = [ex.submit(A.lpips_chunk, c) for c in chunks]
        try:
            for f in cf.as_completed(futs):
                res = f.result()
                on_chunk(res)
                done += len(res)
                _progress(log, done, len(tasks), t0)
        except KeyboardInterrupt:  # finished chunks are already in the cache; a re-run continues
            for f in futs:
                f.cancel()
            raise


def _progress(log, done, total, t0):
    el = time.time() - t0
    eta = el / max(done, 1) * (total - done)
    log(f"  {done}/{total} pairs  ({el / 60:.1f} min elapsed, ~{eta / 60:.1f} min left)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combined", default=str(REPO / "results" / "combined"))
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--edits_root", action="append", default=[], help="folder that contains an edits/ directory")
    ap.add_argument("--clean_root", action="append", default=[], help="clean_edits folder (E0 cache)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only the first N missing pairs (testing)")
    ap.add_argument("--backend", default="real", choices=["real", "fake"], help="fake = L1 distance, for tests only")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--dry_run", action="store_true", help="count what would be computed, write nothing")
    a = ap.parse_args(argv)

    comb = Path(a.combined)
    path = comb / "edits.csv"
    if not path.is_file():
        print(f"{path} not found: run scripts/ingest_results.py first", file=sys.stderr)
        return 2
    df = A.read_results_csv(path)
    cache_path = comb / "edit_lpips_cache.csv"
    cache = A.load_lpips_cache(cache_path)
    done_keys = {tuple(r) for r in cache[list(A.KEY_COLS)].itertuples(index=False, name=None)
                 if True} if len(cache) else set()
    done_keys = {tuple(str(x) for x in k) for k in done_keys}
    res = Path(a.results)
    edit_roots = [Path(p) for p in a.edits_root] or [p.parent for p in find_dirs(res, "edits")]
    clean_roots = [Path(p) for p in a.clean_root] or find_dirs(res, "clean_edits")
    print(f"edit roots: {len(edit_roots)}, clean roots: {len(clean_roots)}, already cached: {len(cache)}")
    tasks, missing = build_tasks(df, edit_roots, clean_roots, done_keys, a.limit)
    n_need = int((pd.to_numeric(df['edit_lpips'], errors='coerce').isna() & ~((df['method'] == 'clean') & (df['purifier'] == 'none'))).sum())
    print(f"rows without edit_lpips: {n_need}; to compute now: {len(tasks)}; images not found: {len(missing)}")
    if a.dry_run:
        return 0
    if missing:
        pd.DataFrame(missing).to_csv(comb / "edit_lpips_missing.csv", index=False)
    cols = A.LPIPS_CACHE_COLS
    new_file = not cache_path.is_file() or cache_path.stat().st_size == 0
    f = open(cache_path, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(cols)
    n_ok = n_bad = 0

    def on_chunk(results):
        nonlocal n_ok, n_bad
        for idx, val in results:
            if val != val:       # NaN (unreadable image): not cached, retried next time
                n_bad += 1
                continue
            r = df.loc[idx]
            w.writerow([A.canon_strength(r[k]) if k == "strength" else r[k] for k in A.KEY_COLS] + [repr(float(val))])
            n_ok += 1
        f.flush()
        os.fsync(f.fileno())

    try:
        run_pool(tasks, a.workers, a.chunk, a.backend, a.quality, on_chunk)
    finally:
        f.close()
    cache = A.load_lpips_cache(cache_path)
    df2, n_fill = A.apply_lpips_cache(df, cache)
    tmp = path.with_name(path.name + ".tmp")
    df2.to_csv(tmp, index=False)
    os.replace(tmp, path)
    left = int((pd.to_numeric(df2["edit_lpips"], errors="coerce").isna() & ~((df2["method"] == "clean") & (df2["purifier"] == "none"))).sum())
    print(f"computed {n_ok} ({n_bad} unreadable); wrote {n_fill} values into {path}; rows still without edit_lpips: {left}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
