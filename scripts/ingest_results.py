"""Combine Kaggle job outputs into one validated, de-duplicated results set.

Typical use (after you downloaded the <JOB>_outputs.zip files and unzipped each into its own folder under results/):

  python scripts/ingest_results.py --results results            # scan results/**/metrics*.csv
  python scripts/ingest_results.py --results results --unzip    # also extract CSV/JSON/NPZ from *_outputs.zip first

Writes results/combined/
  edits.csv            all edit rows, de-duplicated by (method, image_id, seed, editor, prompt_id, strength, purifier)
  protect.csv          all protection rows (invisibility metrics), de-duplicated by (run_id, method, image_id, split)
  coverage.csv / .md   which (split, method, purifier, editor) cells exist: n images, n seeds, strengths, LPIPS filled
  coverage_vs_plan.csv planned cells of eval.tiers (config/parts/eval.yaml) with n images found (0 = missing)
  psr_thresholds.json  95th percentile of impostor ArcFace similarity (from the E0 npz caches; see below)
  arcface/             merged npz caches used for the thresholds
  ingest_report.json   files read / rejected, row counts, duplicates dropped, conflicts

Duplicate rule: keep the row that has edit_lpips, then the latest created_utc. A pre-computed LPIPS cache
(results/combined/edit_lpips_cache.csv, written by scripts/compute_edit_lpips.py) is re-applied, so running ingest
again never loses LPIPS values. Files that are not valid edit/protect tables are listed in the report and skipped
(``--strict`` makes that an error). Nothing is deleted; images and zips are never copied.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval import analysis as A  # noqa: E402
from src.eval.schema import EDIT_COLUMNS, PROTECT_COLUMNS  # noqa: E402

SAFE_SUFFIXES = {".csv", ".json", ".npz", ".txt", ".yaml", ".yml", ".md", ".log"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def safe_extract(zip_path: Path, dest: Path, with_images: bool = False, max_member_mb: int = 512) -> dict:
    """Extract only data members (CSV/JSON/NPZ/...; images optionally) into a NEW folder, with a zip-slip guard.
    Archives are untrusted data: members with absolute paths or '..' are skipped, nothing is executed."""
    dest.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    root = dest.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            suf = Path(info.filename).suffix.lower()
            if suf not in SAFE_SUFFIXES and not (with_images and suf in IMAGE_SUFFIXES):
                n_skip += 1
                continue
            target = (dest / info.filename).resolve()
            if root not in target.parents or info.file_size > max_member_mb * 2 ** 20:
                n_skip += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            n_ok += 1
    return {"zip": str(zip_path), "extracted": n_ok, "skipped": n_skip, "dest": str(dest)}


def find_files(root: Path, patterns, exclude_dirs=("combined", "_to_delete")) -> list:
    out = []
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file() and not any(part in exclude_dirs for part in p.relative_to(root).parts):
                out.append(p)
    return sorted(set(out))


def merge_edit_embeddings(paths, out_path: Path) -> int:
    emb = A.load_npz_edit_embs(paths)
    keys = sorted(emb)
    if not keys:
        return 0
    dim = next(v.shape[0] for v in emb.values() if v is not None)
    E = np.stack([emb[k] if emb[k] is not None else np.zeros(dim, np.float32) for k in keys])
    np.savez(out_path, keys=np.array(keys, dtype=str), emb=E, ok=np.array([emb[k] is not None for k in keys]))
    return len(keys)


def merge_sources(paths, out_path: Path) -> int:
    src = A.load_npz_sources(paths)
    ids = sorted(src)
    if not ids:
        return 0
    dim = next(v.shape[0] for v in src.values() if v is not None)
    E = np.stack([src[i] if src[i] is not None else np.zeros(dim, np.float32) for i in ids])
    np.savez(out_path, ids=np.array(ids, dtype=str), emb=E, ok=np.array([src[i] is not None for i in ids]),
             bbox=np.zeros((len(ids), 4), np.float32), det=np.zeros(len(ids), np.float32), nf=np.ones(len(ids), np.int32))
    return len(ids)


def ingest(results: Path, out: Path, unzip: bool = False, with_images: bool = False, strict: bool = False,
           patterns=("metrics*.csv",), percentile: float = 95.0, plan: bool = True, log=print) -> dict:
    report: dict = {"results_root": str(results), "files": [], "rejected": [], "unzipped": [], "warnings": []}
    results = Path(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if unzip:
        for z in sorted(results.glob("*.zip")) + sorted(results.glob("*/*.zip")):
            d = z.with_suffix("")
            if d.exists():
                report["warnings"].append(f"{z.name}: target folder {d.name} exists, not extracting again")
                continue
            report["unzipped"].append(safe_extract(z, d, with_images))
    edit_parts, prot_parts = [], []
    for p in find_files(results, patterns):
        try:
            d = A.read_results_csv(p)
        except Exception as e:  # unreadable csv
            report["rejected"].append({"file": str(p), "reason": f"unreadable: {e!r}"})
            continue
        kind = A.classify_columns(list(d.columns))
        if kind == "unknown":
            report["rejected"].append({"file": str(p), "reason": "not an edit/protect results table (unknown header)"})
            continue
        miss = A.validate_schema(list(d.columns), kind)
        if miss:
            report["rejected"].append({"file": str(p), "reason": f"{kind} table missing columns {miss}"})
            continue
        d = d[(d["method"] != "") & (d["image_id"] != "")]
        n_bad_sim = int((pd.to_numeric(d["arcface_sim"], errors="coerce").isna() & (d["face_detected"] == "1")).sum()) \
            if kind == "edit" else 0
        report["files"].append({"file": str(p), "kind": kind, "rows": int(len(d)),
                                "face_detected_without_sim": n_bad_sim})
        (edit_parts if kind == "edit" else prot_parts).append(d)
    if strict and report["rejected"]:
        raise SystemExit(f"--strict: {len(report['rejected'])} rejected files, see report: {report['rejected'][:3]}")

    # ---- edits
    if edit_parts:
        e = pd.concat(edit_parts, ignore_index=True)
        for c in EDIT_COLUMNS:
            if c not in e:
                e[c] = ""
        extra = [c for c in e.columns if c not in EDIT_COLUMNS]
        e, st = A.dedup_edits(e)
        cache = A.load_lpips_cache(out / "edit_lpips_cache.csv")
        e, n_fill = A.apply_lpips_cache(e, cache)
        st["lpips_filled_from_cache"] = n_fill
        e = e[EDIT_COLUMNS + extra]
        e.to_csv(out / "edits.csv", index=False)
        report["edits"] = st
        te = A.typed_edits(e)
        cov = A.coverage(te)
        cov.to_csv(out / "coverage.csv", index=False)
        (out / "coverage.md").write_text(coverage_markdown(cov, te, st), encoding="utf-8")
        report["edits"]["n_cells"] = int(len(cov))
        report["edits"]["lpips_missing"] = int(te["edit_lpips"].isna().sum())
        if plan:
            try:
                from src.eval.edit_runner import load_eval_config
                ev = load_eval_config(None, [], REPO)["eval"]
                exp = A.expected_cells(ev, sorted(te["method"].unique()))
                cvp = A.coverage_vs_plan(te, exp)
                cvp.to_csv(out / "coverage_vs_plan.csv", index=False)
                report["plan"] = {"expected_cells": int(len(cvp)), "missing_cells": int((cvp["status"] == "missing").sum())}
            except Exception as ex:  # config missing: coverage_vs_plan is optional
                report["warnings"].append(f"coverage_vs_plan skipped: {ex!r}")
    else:
        report["warnings"].append("no edit tables found")

    # ---- protect
    if prot_parts:
        p = pd.concat(prot_parts, ignore_index=True)
        for c in PROTECT_COLUMNS:
            if c not in p:
                p[c] = ""
        p, st = A.dedup_protect(p)
        p[PROTECT_COLUMNS].to_csv(out / "protect.csv", index=False)
        report["protect"] = st
    else:
        report["warnings"].append("no protect tables found")

    # ---- PSR thresholds from the E0 caches (computed here from embeddings; the job's own json is kept for cross-check)
    (out / "arcface").mkdir(exist_ok=True)
    src_npz = find_files(results, ["*_sources.npz"])
    emb_npz = find_files(results, ["edit_embeddings.npz"])
    job_json = find_files(results, ["psr_thresholds.json"])
    if job_json:
        best = max(job_json, key=lambda q: json.loads(q.read_text()).get("n_pairs", {}).get("pooled", 0)
                   if isinstance(json.loads(q.read_text()).get("n_pairs"), dict) else 0)
        (out / "psr_thresholds.from_job.json").write_text(best.read_text())
        report["thresholds_from_job"] = str(best)
    if src_npz and emb_npz:
        ns = merge_sources(src_npz, out / "arcface" / "sources.npz")
        ne = merge_edit_embeddings(emb_npz, out / "arcface" / "edit_embeddings.npz")
        restrict = None
        if edit_parts and "split" in e:
            test_ids = set(e.loc[e["split"] == "test", "image_id"])
            restrict = test_ids or None
        thr = A.thresholds_from_npz([out / "arcface" / "sources.npz"], [out / "arcface" / "edit_embeddings.npz"],
                                    percentile, restrict)
        thr["n_source_embeddings_merged"], thr["n_edit_embeddings_merged"] = ns, ne
        thr["restricted_to"] = "test split image ids" if restrict else "all sources"
        (out / "psr_thresholds.json").write_text(json.dumps(thr, indent=2))
        report["thresholds"] = {"per_editor": thr["per_editor"], "pooled": thr.get("pooled"), "n_pairs": thr["n_pairs"]}
    elif job_json:
        (out / "psr_thresholds.json").write_text((out / "psr_thresholds.from_job.json").read_text())
        report["warnings"].append("thresholds copied from the job's psr_thresholds.json (npz caches not found)")
    else:
        report["warnings"].append("no PSR thresholds: need E0 (clean) edit embeddings + source embeddings "
                                  "(cache/arcface/*.npz) or psr_thresholds.json; PSR will show 'n/a' until then")
    (out / "ingest_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def coverage_markdown(cov: pd.DataFrame, te: pd.DataFrame, st: dict) -> str:
    lines = ["# Coverage of combined results", "",
             f"Rows after de-duplication: {st['rows_out']} (dropped {st['duplicates_dropped']} duplicates, "
             f"{st['conflicts']} duplicate keys with differing ArcFace sim).", "",
             f"Cells: {len(cov)}; methods: {', '.join(sorted(te['method'].unique()))}; "
             f"splits: {', '.join(sorted(te['split'].unique()))}.", "",
             "| split | method | purifier | editor | images | seeds | strengths | prompts | face rate | LPIPS filled |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cov.itertuples(index=False):
        lines.append(f"| {r.split} | {r.method} | {r.purifier} | {r.editor} | {r.n_images} | {r.seeds} | {r.strengths} | "
                     f"{r.n_prompts} | {r.face_rate:.2f} | {r.lpips_filled:.2f} |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "results"), help="folder holding the unzipped job outputs")
    ap.add_argument("--out", default=None, help="default <results>/combined")
    ap.add_argument("--unzip", action="store_true", help="extract data files of results/*_outputs.zip into new folders first")
    ap.add_argument("--with_images", action="store_true", help="with --unzip: also extract images (large)")
    ap.add_argument("--strict", action="store_true", help="fail if any csv is rejected")
    ap.add_argument("--glob", action="append", default=None, help="csv name pattern(s), default metrics*.csv")
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--no_plan", action="store_true", help="skip the comparison with eval.tiers")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    res = Path(a.results)
    out = Path(a.out) if a.out else res / "combined"
    rep = ingest(res, out, a.unzip, a.with_images, a.strict, tuple(a.glob or ["metrics*.csv"]), a.percentile, not a.no_plan)
    ed, pr = rep.get("edits"), rep.get("protect")
    print(f"edit files: {sum(f['kind'] == 'edit' for f in rep['files'])}, protect files: "
          f"{sum(f['kind'] == 'protect' for f in rep['files'])}, rejected: {len(rep['rejected'])}")
    if ed:
        print(f"edits.csv: {ed['rows_out']} rows ({ed['duplicates_dropped']} duplicates dropped, {ed['conflicts']} conflicts, "
              f"{ed['lpips_missing']} rows without edit_lpips, {ed['n_cells']} cells)")
    if pr:
        print(f"protect.csv: {pr['rows_out']} rows ({pr['duplicates_dropped']} duplicates dropped)")
    if "plan" in rep:
        print(f"plan: {rep['plan']['missing_cells']} of {rep['plan']['expected_cells']} planned cells have no rows")
    for w in rep["warnings"]:
        print("WARNING:", w)
    for r in rep["rejected"]:
        print("REJECTED:", r["file"], "-", r["reason"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
