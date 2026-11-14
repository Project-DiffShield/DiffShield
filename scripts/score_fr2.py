#!/usr/bin/env python
"""Post-hoc second face recogniser (fr2): FaceNet (facenet-pytorch InceptionResnetV1 'vggface2') scored on saved edits.

The harness column ``fr2_sim`` is NaN in every run (the planned second recogniser was never implemented in the edit
runner). This script fills it AFTER the fact, without touching the original metrics files:

  fr2_sim = cosine( FaceNet(source photo), FaceNet(edit) )

Faces are detected and aligned by facenet-pytorch's own MTCNN (image_size 160, margin 0, largest face), so neither the
detector nor the recogniser is insightface / buffalo_l (the ArcFace judge). No face in the edit -> fr2_sim NaN and
fr2_face = 0 (counts as a hit in fr2-PSR, like the ArcFace PSR).

CAVEAT (independence): the exploration variants X3/X4/X5 were TRAINED against this same FaceNet (surrogate identity
loss), so fr2 is NOT an independent judge for them. For every other method (v2_tgt_zero_mid, v2_next_tuned, clean, ...)
FaceNet was not used as a loss. Do not read an fr2 gain of X3-X5 as independent evidence.

  python scripts/score_fr2.py --input_roots /kaggle/input --data_dir <prepare_data out>/data --out_dir OUT
  python scripts/score_fr2.py --folders results/T04 results/T05 --data_dir data --out_dir OUT

Inputs
  --folders / --input_roots   result "units". A unit is the folder above a ``results`` directory (e.g. .../T04) in which
                              metrics*.csv edit rows (columns edit_path, arcface_sim) are found; with --input_roots the
                              roots are searched recursively, with --folders each folder is a unit root. Protect-run CSVs
                              (no edit_path column) are ignored.
  --data_dir                  prepare_data output: either the folder with the photos (<id>.png) or its parent holding
                              <split>/<id>.png. Sources are looked up by image_id (the CSV paths are Kaggle paths of the
                              ORIGINAL session and do not exist here).
Paths in the CSVs (e.g. /kaggle/working/T04/results/eval_x/edits/...jpg) are resolved by trying progressively shorter
suffixes of the path under the unit folder, its parent and the search roots.

Outputs (in --out_dir)
  fr2_<unit>.csv              one row per edit row: key columns, face_detected / arcface_sim (copied), fr2_sim, fr2_face,
                              fr2_status (ok | edit_no_face | source_no_face | edit_missing | source_missing)
  fr2_threshold.json          calibrated threshold: 95th percentile of FaceNet similarity between DIFFERENT people
                              (basis recorded per editor and pooled; see below)
  fr2_psr_table.csv / .md     fr2-PSR per method x editor x purifier (x strength): share of edits with fr2_sim < threshold
                              or no face (per-photo mean first, then mean over photos), next to the ArcFace PSR at 0.1005

Threshold basis. Preferred: edits of the ``clean`` method (edit of person i vs SOURCE of person j != i) with
purifier none, per editor; an editor without such rows uses the pooled threshold (all clean purifier-none edits; else
all clean edits of any purifier; else pairs of distinct source photos). The source-photo-only threshold is always
written too (``sources_only``) for reference. Each image_id is treated as a different person (the splits are de-duplicated).
CPU-capable; uses cuda if present. Nothing is downloaded except the vggface2 weights of InceptionResnetV1 (GitHub release).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
IMG_EXTS = (".png", ".jpg", ".jpeg")
KEY_COLS = ["method", "image_id", "seed", "editor", "prompt_id", "strength", "purifier"]
OUT_COLS = ["unit", "run_id", "method", "image_id", "split", "seed", "editor", "prompt_id", "strength", "purifier",
            "edit_path", "face_detected", "arcface_sim", "fr2_sim", "fr2_face", "fr2_status"]
ARCFACE_THRESHOLD = 0.1005
MIN_PAIRS_PER_EDITOR = 200


# ------------------------------------------------------------------------------------------------ embedders
class FaceNetEmbedder:
    """MTCNN (facenet-pytorch) + InceptionResnetV1 vggface2. ``embed_paths`` -> list of L2-normalised 512-d float32
    vectors, None where MTCNN finds no face (or the file cannot be opened)."""

    def __init__(self, device: Optional[str] = None, batch: int = 64):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1  # lazy: optional dependency (pip install --no-deps)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch = int(batch)
        self.mtcnn = MTCNN(image_size=160, margin=0, post_process=True, select_largest=True, keep_all=False,
                           device=self.device)
        self.net = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def embed_paths(self, paths: Sequence[Path]) -> List[Optional[np.ndarray]]:
        from PIL import Image
        torch = self.torch
        out: List[Optional[np.ndarray]] = [None] * len(paths)
        faces, idx = [], []

        def flush():
            if not faces:
                return
            with torch.no_grad():
                e = self.net(torch.stack(faces).to(self.device)).float().cpu().numpy()
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
            for i, v in zip(idx, e):
                out[i] = v.astype(np.float32)
            faces.clear()
            idx.clear()

        for i, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
                with torch.no_grad():
                    f = self.mtcnn(img)
            except Exception as ex:  # unreadable file counts as missing at the caller, not as a face
                print(f"[fr2] cannot read {p}: {ex!r}", file=sys.stderr)
                f = None
            if f is not None:
                faces.append(f)
                idx.append(i)
                if len(faces) >= self.batch:
                    flush()
        flush()
        return out


# ------------------------------------------------------------------------------------------------ path resolution
def find_source(data_dir: Path, split: str, image_id: str) -> Optional[Path]:
    for base in (data_dir / split, data_dir):
        for ext in IMG_EXTS:
            p = base / f"{image_id}{ext}"
            if p.is_file():
                return p
    return None


class PathResolver:
    """Map a path recorded in the ORIGINAL session (absolute Kaggle path) to a file under ``bases``: try the longest
    suffix first; the (base, suffix-length) that worked last time is tried first."""

    def __init__(self, bases: Sequence[Path]):
        self.bases = [Path(b) for b in bases]
        self.last: Optional[Tuple[int, int]] = None
        self.cache: Dict[str, Optional[Path]] = {}

    def __call__(self, recorded: str) -> Optional[Path]:
        if not recorded:
            return None
        if recorded in self.cache:
            return self.cache[recorded]
        parts = [p for p in recorded.replace("\\", "/").split("/") if p and not p.endswith(":")]
        hit = None
        if os.path.isfile(recorded):
            hit = Path(recorded)
        else:
            order = [(bi, k) for bi in range(len(self.bases)) for k in range(len(parts))]
            if self.last is not None and self.last[1] < len(parts):
                order.insert(0, self.last)
            for bi, k in order:
                cand = self.bases[bi].joinpath(*parts[k:])
                if cand.is_file():
                    hit, self.last = cand, (bi, k)
                    break
        self.cache[recorded] = hit
        return hit


# ------------------------------------------------------------------------------------------------ discovery / reading
def _is_edit_csv(f: Path) -> bool:
    try:
        head = pd.read_csv(f, nrows=0).columns
    except Exception:
        return False
    return "edit_path" in head and "arcface_sim" in head and "method" in head


def unit_of(csv: Path) -> Path:
    """The folder above the ``results`` directory that holds the CSV (else the CSV's own folder)."""
    parts = csv.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "results":
            return Path(*parts[:i]) if i > 0 else Path(".")
    return csv.parent


PRUNE_DIRS = {"edits", "images", "cache", "info", "data", "purified", "clean_edits", "__pycache__", ".git"}


def find_metrics_csvs(root: Path) -> List[Path]:
    """metrics*.csv under ``root`` without descending into image folders or dataset-looking folders (a CelebAMask-HQ
    input holds hundreds of thousands of files; the result CSVs live at <unit>/results/<run>/metrics*.csv)."""
    out: List[Path] = []
    for r, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in PRUNE_DIRS and "celeba" not in d.lower())
        out.extend(Path(r) / f for f in sorted(files) if f.startswith("metrics") and f.endswith(".csv"))
    return out


def discover_units(folders: Iterable[Path] = (), input_roots: Iterable[Path] = ()) -> Dict[Path, List[Path]]:
    """{unit folder: [edit CSV files]}"""
    units: Dict[Path, List[Path]] = {}
    for fo in folders:
        fo = Path(fo)
        for f in find_metrics_csvs(fo):
            if _is_edit_csv(f):
                units.setdefault(fo, []).append(f)
    for root in input_roots:
        for f in find_metrics_csvs(Path(root)):
            if _is_edit_csv(f):
                units.setdefault(unit_of(f), []).append(f)
    return units


def read_unit_rows(files: Sequence[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    key = [c for c in KEY_COLS if c in df.columns]
    return df.drop_duplicates(key, keep="last").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ scoring
def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def score_unit(name: str, rows: pd.DataFrame, resolve: PathResolver, data_dir: Path, embedder,
               src_cache: Dict[Tuple[str, str], Optional[np.ndarray]], max_rows: int = 0
               ) -> Tuple[pd.DataFrame, Dict[int, np.ndarray]]:
    """Returns (result frame, {row index: edit embedding}) (embeddings only for edits with a face)."""
    if max_rows:
        rows = rows.iloc[:max_rows].reset_index(drop=True)
    # sources
    need = sorted({(r.get("split", ""), r["image_id"]) for _, r in rows.iterrows()} - set(src_cache))
    paths, keys = [], []
    for sp, iid in need:
        p = find_source(data_dir, sp, iid)
        if p is None:
            src_cache[(sp, iid)] = "missing"  # type: ignore[assignment]
        else:
            paths.append(p)
            keys.append((sp, iid))
    for k, e in zip(keys, embedder.embed_paths(paths) if paths else []):
        src_cache[k] = e
    # edits
    epaths, eidx, status = [], [], ["edit_missing"] * len(rows)
    for i, r in rows.iterrows():
        p = resolve(r.get("edit_path", ""))
        if p is not None:
            epaths.append(p)
            eidx.append(i)
    emb = embedder.embed_paths(epaths) if epaths else []
    edit_emb: Dict[int, np.ndarray] = {}
    sim = [float("nan")] * len(rows)
    face = [float("nan")] * len(rows)
    for i, e in zip(eidx, emb):
        r = rows.iloc[i]
        s = src_cache.get((r.get("split", ""), r["image_id"]))
        if isinstance(s, str):
            status[i] = "source_missing"
            continue
        if e is None:
            face[i], status[i] = 0.0, "edit_no_face"
            continue
        edit_emb[i] = e
        if s is None:
            face[i], status[i] = 1.0, "source_no_face"
            continue
        face[i], status[i] = 1.0, "ok"
        sim[i] = cos(e, s)
    out = rows.copy()
    out["unit"] = name
    out["fr2_sim"] = sim
    out["fr2_face"] = face
    out["fr2_status"] = status
    for c in OUT_COLS:
        if c not in out:
            out[c] = ""
    return out[OUT_COLS], edit_emb


# ------------------------------------------------------------------------------------------------ threshold
def _impostor_sims(edit_embs: np.ndarray, edit_ids: Sequence[str], src_ids: Sequence[str], src_embs: np.ndarray) -> np.ndarray:
    """cosine between each edit and the SOURCE of every OTHER person; flat array."""
    if len(edit_embs) == 0 or len(src_embs) == 0:
        return np.zeros(0)
    S = edit_embs @ src_embs.T
    same = np.asarray(edit_ids)[:, None] == np.asarray(src_ids)[None, :]
    return S[~same]


def calibrate(clean_rows: pd.DataFrame, clean_emb: Dict[int, np.ndarray], src_by_id: Dict[str, np.ndarray],
              percentile: float = 95.0, min_pairs: int = MIN_PAIRS_PER_EDITOR) -> dict:
    """clean_rows: scored clean rows (columns image_id, editor, purifier; index = key of clean_emb)."""
    ids = sorted(src_by_id)
    S = np.stack([src_by_id[i] for i in ids]) if ids else np.zeros((0, 1))
    res: dict = {"percentile": percentile, "n_source_photos": len(ids)}
    # sources only: distinct pairs i<j
    if len(ids) >= 2:
        M = S @ S.T
        iu = np.triu_indices(len(ids), 1)
        v = M[iu]
        res["sources_only"] = {"threshold": float(np.percentile(v, percentile)), "n_pairs": int(v.size)}
    else:
        res["sources_only"] = None

    def thr_for(sub: pd.DataFrame) -> Optional[dict]:
        idx = [i for i in sub.index if i in clean_emb]
        if not idx:
            return None
        E = np.stack([clean_emb[i] for i in idx])
        v = _impostor_sims(E, sub.loc[idx, "image_id"].tolist(), ids, S)
        return {"threshold": float(np.percentile(v, percentile)), "n_pairs": int(v.size), "n_clean_edits": len(idx)} if v.size else None

    pooled, pbasis = None, None
    if len(clean_rows):
        none = clean_rows[clean_rows["purifier"] == "none"]
        for sub, b in ((none, "clean edits, purifier none, all editors"), (clean_rows, "clean edits, any purifier, all editors")):
            pooled = thr_for(sub) if len(sub) else None
            if pooled:
                pbasis = b
                break
    if pooled is None and res["sources_only"]:
        pooled, pbasis = dict(res["sources_only"]), "pairs of distinct source photos (no clean edit rows with faces in the inputs)"
    res["pooled"] = dict(pooled, basis=pbasis) if pooled else None
    per_editor: dict = {}
    if len(clean_rows):
        none = clean_rows[clean_rows["purifier"] == "none"]
        for ed, g in none.groupby("editor"):
            t = thr_for(g)
            if t and t["n_pairs"] >= min_pairs:
                per_editor[ed] = dict(t, basis="clean edits of this editor, purifier none")
    res["per_editor"] = per_editor
    res["used_when_editor_missing"] = "pooled"
    res["basis_summary"] = ("clean_edits" if (per_editor or (pbasis or "").startswith("clean")) else
                            ("sources_only" if pooled else "none"))
    return res


def threshold_for(cal: dict, editor: str) -> Optional[float]:
    t = (cal.get("per_editor") or {}).get(editor) or cal.get("pooled")
    return None if not t else float(t["threshold"])


# ------------------------------------------------------------------------------------------------ PSR table
def psr_table(scored: pd.DataFrame, cal: dict, arcface_thr: float = ARCFACE_THRESHOLD) -> pd.DataFrame:
    d = scored[scored["fr2_status"].isin(["ok", "edit_no_face"])].copy()
    if d.empty:
        return pd.DataFrame(columns=["method", "editor", "purifier", "strength", "n_photos", "n_edits", "fr2_psr",
                                     "fr2_threshold", "mean_fr2_sim", "arcface_psr", "mean_arcface_sim"])
    d["thr"] = pd.to_numeric(d["editor"].map(lambda e: threshold_for(cal, e)), errors="coerce")
    d["fr2_sim"] = pd.to_numeric(d["fr2_sim"])
    nf = d["fr2_face"].astype(float) == 0
    d["hit"] = np.where(nf, 1.0, (d["fr2_sim"] < d["thr"]).astype(float))
    a_sim = pd.to_numeric(d["arcface_sim"], errors="coerce")
    a_fd = pd.to_numeric(d["face_detected"], errors="coerce")
    d["a_hit"] = np.where(a_fd == 0, 1.0, np.where(a_sim.notna(), (a_sim < arcface_thr).astype(float), np.nan))
    d["arcface_sim_n"] = a_sim
    g = d.groupby(["method", "editor", "purifier", "strength", "image_id"], sort=True)
    pp = g.agg(hit=("hit", "mean"), a_hit=("a_hit", "mean"), sim=("fr2_sim", "mean"), asim=("arcface_sim_n", "mean"),
               n=("hit", "size"), thr=("thr", "first")).reset_index()
    t = pp.groupby(["method", "editor", "purifier", "strength"], sort=True).agg(
        n_photos=("image_id", "nunique"), n_edits=("n", "sum"), fr2_psr=("hit", "mean"), fr2_threshold=("thr", "first"),
        mean_fr2_sim=("sim", "mean"), arcface_psr=("a_hit", "mean"), mean_arcface_sim=("asim", "mean")).reset_index()
    return t


def to_markdown(t: pd.DataFrame, cal: dict, notes: Sequence[str] = ()) -> str:
    L = ["# fr2-PSR (FaceNet vggface2 + MTCNN), per method x editor x purifier x strength", "",
         "fr2-PSR = share of edits with FaceNet similarity (source vs edit) below the calibrated threshold or no face "
         "(per-photo mean over seeds/prompts, then mean over photos). Higher = better protection. ArcFace PSR (fixed "
         f"threshold {ARCFACE_THRESHOLD}) shown next to it. Threshold basis: {cal.get('basis_summary')}; "
         f"pooled {json.dumps(cal.get('pooled'))}.", "",
         "NOTE: X3-X5 were trained against this FaceNet, so fr2 is not independent for them.", ""]
    L += [f"- {n}" for n in notes]
    L += ["", "| method | editor | purifier | strength | photos | edits | fr2-PSR | fr2 thr | mean fr2 sim | ArcFace PSR | mean ArcFace sim |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(x, nd=3):
        return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"

    for _, r in t.iterrows():
        L.append(f"| {r['method']} | {r['editor']} | {r['purifier']} | {r['strength']} | {int(r['n_photos'])} | {int(r['n_edits'])} | "
                 f"{f(r['fr2_psr'])} | {f(r['fr2_threshold'], 4)} | {f(r['mean_fr2_sim'])} | {f(r['arcface_psr'])} | {f(r['mean_arcface_sim'])} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ main
def run(units: Dict[Path, List[Path]], data_dir: Path, out_dir: Path, embedder, search_roots: Sequence[Path] = (),
        max_rows: int = 0, percentile: float = 95.0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_cache: Dict[Tuple[str, str], Optional[np.ndarray]] = {}
    scored_all: List[pd.DataFrame] = []
    clean_parts: List[pd.DataFrame] = []
    clean_emb: Dict[int, np.ndarray] = {}
    used_names: Dict[str, int] = {}
    notes: List[str] = []
    offset = 0
    for unit, files in sorted(units.items(), key=lambda kv: str(kv[0])):
        name = unit.name or "unit"
        used_names[name] = used_names.get(name, 0) + 1
        if used_names[name] > 1:
            name = f"{name}_{used_names[name]}"
        rows = read_unit_rows(files)
        bases = [unit, unit.parent, *search_roots]
        resolve = PathResolver(bases)
        sc, emb = score_unit(name, rows, resolve, data_dir, embedder, src_cache, max_rows)
        sc.to_csv(out_dir / f"fr2_{name}.csv", index=False)
        cnt = sc["fr2_status"].value_counts().to_dict()
        notes.append(f"{name}: {len(sc)} rows from {len(files)} file(s); status {cnt}")
        print(f"[fr2] {notes[-1]}", flush=True)
        sc = sc.copy()
        sc.index = sc.index + offset
        scored_all.append(sc)
        cl = sc[sc["method"] == "clean"]
        clean_parts.append(cl)
        clean_emb.update({i + offset: e for i, e in emb.items() if (i + offset) in cl.index})
        offset += len(sc)
    scored = pd.concat(scored_all, ignore_index=False) if scored_all else pd.DataFrame(columns=OUT_COLS)
    clean_rows = pd.concat(clean_parts) if clean_parts else pd.DataFrame(columns=OUT_COLS)
    src_by_id = {}
    for (sp, iid), e in src_cache.items():
        if isinstance(e, np.ndarray) and iid not in src_by_id:
            src_by_id[iid] = e
    cal = calibrate(clean_rows, clean_emb, src_by_id, percentile)
    cal["units"] = notes
    (out_dir / "fr2_threshold.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    # the same output attached twice (e.g. as dataset and as notebook output) must not double the rows
    t = psr_table(scored.drop_duplicates([c for c in KEY_COLS if c in scored.columns], keep="first"), cal)
    t.to_csv(out_dir / "fr2_psr_table.csv", index=False)
    (out_dir / "fr2_psr_table.md").write_text(to_markdown(t, cal, notes), encoding="utf-8")
    return {"calibration": cal, "table": t, "scored": scored}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folders", nargs="*", default=[], help="result unit folders (each searched recursively for metrics*.csv)")
    ap.add_argument("--input_roots", nargs="*", default=[], help="roots searched recursively for result units (e.g. /kaggle/input)")
    ap.add_argument("--search_roots", nargs="*", default=[], help="extra roots for resolving image paths")
    ap.add_argument("--data_dir", required=True, help="prepare_data output (photos by image_id, or <split>/<id>.png)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_rows", type=int, default=0, help="smoke: only the first N rows per unit (0 = all)")
    ap.add_argument("--percentile", type=float, default=95.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    units = discover_units([Path(p) for p in a.folders], [Path(p) for p in a.input_roots])
    if not units:
        print("ERROR: no edit-row CSVs (metrics*.csv with edit_path and arcface_sim) found", file=sys.stderr)
        return 2
    print(f"[fr2] units: {[str(u) for u in units]}", flush=True)
    emb = FaceNetEmbedder(a.device)
    res = run(units, Path(a.data_dir), Path(a.out_dir), emb, [Path(p) for p in a.search_roots], a.max_rows, a.percentile)
    print(to_markdown(res["table"], res["calibration"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
