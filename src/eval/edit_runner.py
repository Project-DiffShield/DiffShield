"""Core edit-and-score loop of the Half B harness (edit + score in one job, 48h plan section 3).

For every (image, method, purifier, editor, strength, prompt, seed) of a *tier* it
  1. loads the published (protected) PNG, applies the eval purifier (cached PNG on disk),
  2. edits it with the editor (3 seeds batched, per-seed generators),
  3. obtains the clean-image edit of the same (editor, prompt, strength, seed) from the E0 cache (generating it if
     missing and allowed), and
  4. writes ONE CSV row per edit with ArcFace similarity to the source, CLIP-I, edit LPIPS, invisibility metrics.

Resume is by row key ``(method, image_id, seed, editor, prompt_id, strength, purifier)``; progress and gpu_time.json
are checkpointed every ``eval.checkpoint_every`` (50) rows. Heavy dependencies (torch, diffusers, insightface, clip,
lpips) enter only through *backends*, so the whole loop is testable on CPU with fakes (tests/test_eval_*.py).

Also exposes ``score_protected`` (callable for Optuna), PSR threshold helpers and a small summary.

CSV schema = docs/V2_PLAN.md section 4.1 (the order of ``ROW_COLUMNS`` below).
"""
from __future__ import annotations

import csv
import fnmatch
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------------------- schema
ROW_COLUMNS: List[str] = [
    # keys
    "run_id", "method", "method_cfg_hash", "image_id", "split", "seed", "editor", "prompt_id", "strength", "purifier",
    # provenance
    "protect_run_id", "eps_255", "purifier_params", "editor_params", "source_path", "protected_path", "purified_path",
    "edit_path", "clean_edit_path", "code_git_hash", "created_utc",
    # identity
    "face_detected", "n_faces", "det_score", "arcface_sim", "fr2_sim",
    # edit
    "edit_lpips", "clip_i",
    # invisibility
    "inv_psnr", "inv_ssim", "inv_lpips",
    # cost
    "gpu_seconds_edit", "gpu_seconds_purify",
]
KEY_COLUMNS: Tuple[str, ...] = ("method", "image_id", "seed", "editor", "prompt_id", "strength", "purifier")
NAN = float("nan")


def fmt_strength(s) -> str:
    """Canonical string for the strength key part ('' when the editor has none)."""
    if s is None or (isinstance(s, str) and s == ""):
        return ""
    return f"{float(s):g}"


def derive_seed(base: int, *parts) -> int:
    """sha256-based seed (never Python hash()). Uses src.utils.seeding.derive_seed if available (A1)."""
    try:  # pragma: no cover - depends on A1's module
        from src.utils.seeding import derive_seed as _ds
        return int(_ds(base, *parts))
    except Exception:
        h = hashlib.sha256("|".join([str(base)] + [str(p) for p in parts]).encode()).hexdigest()
        return int(h, 16) % (2 ** 31)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_hash(repo: Optional[Path] = None) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo or Path(__file__).resolve().parents[2]),
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    for p in (Path("BUNDLE_INFO.json"), Path(__file__).resolve().parents[2] / "BUNDLE_INFO.json"):
        try:
            return json.loads(p.read_text()).get("hash", "unknown")
        except Exception:
            continue
    return "unknown"


def _cfg_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------- config helpers
def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_eval_config(config_path: Optional[str] = None, overrides: Sequence[str] = (), repo_root: Optional[Path] = None) -> dict:
    """Merged config containing at least ``eval:``. Uses A1's ``src.utils.config.load_config`` when importable,
    else a local fallback: config/config.yaml (if present) deep-merged with config/parts/*.yaml (sorted)."""
    import yaml
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    cfg_path = Path(config_path) if config_path else root / "config" / "config.yaml"
    try:  # pragma: no cover - depends on A1
        from src.utils.config import load_config
        return load_config(cfg_path, None, list(overrides))
    except Exception:
        pass
    cfg: dict = {}
    if cfg_path.is_file():
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    for p in sorted((cfg_path.parent / "parts").glob("*.yaml")):
        cfg = deep_merge(cfg, yaml.safe_load(p.read_text()) or {})
    for ov in overrides:
        k, v = ov.split("=", 1)
        d = cfg
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    return cfg


# ------------------------------------------------------------------------------------------- tier expansion
@dataclass(frozen=True)
class Job:
    tier: str
    method: str
    split: str
    image_id: str
    image_index: int
    purifier: str
    editor: str
    strength: str          # canonical key string ('' = none)
    prompt_idx: int

    @property
    def prompt_id(self) -> str:
        return f"p{self.prompt_idx}"

    def strength_value(self) -> Optional[float]:
        return None if self.strength == "" else float(self.strength)

    def row_keys(self, seeds: Sequence[int]) -> List[tuple]:
        return [(self.method, self.image_id, str(s), self.editor, self.prompt_id, self.strength, self.purifier)
                for s in seeds]


def _matches(method: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(method, p) for p in patterns)


def canonical_purifier(name: str, ev: dict) -> str:
    from .purify_attacks import canonical_name
    return canonical_name(name, {"eval": ev})


def expand_tier(ev: dict, tier: str, method: str, split: str, image_ids: Sequence[str],
                purifier_filter: Optional[Sequence[str]] = None, editor_filter: Optional[Sequence[str]] = None,
                shard: Optional[Tuple[int, int]] = None) -> List[Job]:
    """All jobs of ``tier`` for one method on one split. ``image_ids`` is the FULL split list (order defines
    ``image_index`` and hence prompt rotation); ``shard=(k, n)`` keeps images with ``image_index % n == k``.

    purifier_filter: None / ['all'] = every purifier of the matching rules; else restrict to these names.
    """
    rules = ev["tiers"][tier]
    headline = ev.get("headline", {})
    pfilt = None
    if purifier_filter and "all" not in purifier_filter:
        pfilt = {canonical_purifier(p, ev) for p in purifier_filter}
    efilt = set(editor_filter) if editor_filter else None
    jobs: Dict[tuple, Job] = {}
    for rule in rules:
        if split not in rule.get("splits", [split]) or not _matches(method, rule["methods"]):
            continue
        purs = [canonical_purifier(p, ev) for p in rule["purifiers"]]
        if pfilt is not None:
            purs = [p for p in purs if p in pfilt]
        for cell in rule["cells"]:
            ed = cell["editor"]
            if efilt is not None and ed not in efilt:
                continue
            edcfg = ev["editors"][ed]
            strengths = cell.get("strengths")
            if strengths is None:
                strengths = [edcfg.get("default_strength")]
            n_prompts = len(ev["prompts"][edcfg["prompts"]])
            for s in strengths:
                sk = fmt_strength(s)
                is_headline = (ed == headline.get("editor") and sk == fmt_strength(headline.get("strength")))
                for pur in purs:
                    for idx, iid in enumerate(image_ids):
                        if shard is not None and idx % shard[1] != shard[0]:
                            continue
                        if cell.get("prompts") == "all_if_headline" and is_headline:
                            pidxs = list(range(n_prompts))
                        else:
                            pidxs = [idx % n_prompts]
                        for pi in pidxs:
                            j = Job(tier, method, split, iid, idx, pur, ed, sk, pi)
                            jobs[(method, split, iid, pur, ed, sk, pi)] = j
    return list(jobs.values())


def count_rows(jobs: Sequence[Job], n_seeds: int) -> Dict[str, int]:
    """Row counts by editor (edits = jobs x seeds)."""
    out: Dict[str, int] = {}
    for j in jobs:
        out[j.editor] = out.get(j.editor, 0) + n_seeds
    out["total"] = sum(out.values())
    return out


def get_prompt(ev: dict, editor: str, prompt_idx: int) -> str:
    return ev["prompts"][ev["editors"][editor]["prompts"]][prompt_idx]


# ------------------------------------------------------------------------------------------------ results io
def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "nan" if math.isnan(v) else repr(v)
    if isinstance(v, (np.floating,)):
        return _fmt(float(v))
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, default=str)
    return str(v)


class ResultsWriter:
    """Append-only CSV with fsync per row; resume by key tuple. Header mismatch on an existing file is an error."""

    def __init__(self, path: Path, columns: Sequence[str] = ROW_COLUMNS, key_cols: Sequence[str] = KEY_COLUMNS):
        self.path = Path(path)
        self.columns = list(columns)
        self.key_cols = tuple(key_cols)
        self.done: set = set()
        self.n_rows = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_file() and self.path.stat().st_size > 0:
            with open(self.path, newline="") as f:
                rd = csv.DictReader(f)
                if rd.fieldnames != self.columns:
                    raise ValueError(f"{self.path}: header differs from ROW_COLUMNS; refusing to append")
                for r in rd:
                    self.done.add(tuple(r[k] for k in self.key_cols))
                    self.n_rows += 1
        else:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(self.columns)

    def key_of(self, row: Mapping) -> tuple:
        return tuple(str(row[k]) for k in self.key_cols)

    def append(self, row: Mapping) -> None:
        k = self.key_of(row)
        if k in self.done:
            return
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([_fmt(row.get(c)) for c in self.columns])
            f.flush()
            os.fsync(f.fileno())
        self.done.add(k)
        self.n_rows += 1


def _atomic_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------- caches
IMG_EXTS = (".png", ".jpg", ".jpeg")


def find_image(directory: Path, image_id: str) -> Optional[Path]:
    for ext in IMG_EXTS:
        p = Path(directory) / f"{image_id}{ext}"
        if p.is_file():
            return p
    return None


def _open_rgb(path):
    from PIL import Image
    im = Image.open(path)
    im.load()
    return im.convert("RGB")


def _save_png(img, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def jpeg_roundtrip(img, quality: int = 90):
    """Encode/decode through libjpeg (what a saved edit looks like)."""
    from PIL import Image
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def _save_jpeg(img, path: Path, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.jpg")
    img.convert("RGB").save(tmp, format="JPEG", quality=int(quality))
    os.replace(tmp, path)


class EmbCache:
    """Persistent {key: embedding | None} (npz). Safe to lose; merged with what is on disk at save time."""

    def __init__(self, path: Optional[Path]):
        self.path = Path(path) if path else None
        self.data: Dict[str, Optional[np.ndarray]] = {}
        self._dirty = False
        self._load_into(self.data)

    def _load_into(self, d):
        if self.path and self.path.is_file():
            try:
                z = np.load(self.path, allow_pickle=False)
                for k, e, ok in zip(z["keys"], z["emb"], z["ok"]):
                    d.setdefault(str(k), e.astype(np.float32) if ok else None)
            except Exception:
                pass

    def get(self, key: str):
        return self.data.get(key, "MISSING")

    def put(self, key: str, emb: Optional[np.ndarray]):
        self.data[key] = None if emb is None else np.asarray(emb, dtype=np.float32)
        self._dirty = True

    def save(self):
        if not (self.path and self._dirty):
            return
        merged: Dict[str, Optional[np.ndarray]] = {}
        self._load_into(merged)
        merged.update(self.data)
        keys = sorted(merged)
        dim = next((v.shape[0] for v in merged.values() if v is not None), 512)
        emb = np.stack([v if v is not None else np.zeros(dim, np.float32) for v in (merged[k] for k in keys)]) \
            if keys else np.zeros((0, dim), np.float32)
        ok = np.array([merged[k] is not None for k in keys], dtype=bool)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, keys=np.array(keys, dtype=str), emb=emb, ok=ok)
        os.replace(tmp, self.path)
        self._dirty = False


class SourceStore:
    """Source (clean) images and their cached face results (embedding, det score, bbox, n_faces)."""

    def __init__(self, source_dir: Path, split: str, cache_dir: Path, face):
        self.source_dir, self.split, self.face = Path(source_dir), split, face
        self.path = Path(cache_dir) / "arcface" / f"{split}_sources.npz"
        self.mem: Dict[str, tuple] = {}
        self._load()

    def _load(self):
        if self.path.is_file():
            try:
                z = np.load(self.path, allow_pickle=False)
                for i, k in enumerate(z["ids"]):
                    emb = z["emb"][i] if z["ok"][i] else None
                    bb = tuple(float(v) for v in z["bbox"][i]) if z["ok"][i] else None
                    self.mem[str(k)] = (emb, float(z["det"][i]), bb, int(z["nf"][i]))
            except Exception:
                self.mem = {}

    def image_path(self, image_id: str) -> Path:
        p = find_image(self.source_dir, image_id)
        if p is None:
            raise FileNotFoundError(f"source image {image_id} not found in {self.source_dir}")
        return p

    def image(self, image_id: str):
        return _open_rgb(self.image_path(image_id))

    def face_of(self, image_id: str) -> tuple:
        """(emb | None, det_score, bbox | None, n_faces)"""
        if image_id not in self.mem:
            r = self.face.analyze(self.image(image_id))
            self.mem[image_id] = (r.emb, r.det_score, r.bbox, r.n_faces)
        return self.mem[image_id]

    def bbox_of(self, image_id: str):
        return self.face_of(image_id)[2]

    def save(self):
        ids = sorted(self.mem)
        if not ids:
            return
        dim = next((self.mem[i][0].shape[0] for i in ids if self.mem[i][0] is not None), 512)
        emb = np.stack([self.mem[i][0] if self.mem[i][0] is not None else np.zeros(dim, np.float32) for i in ids])
        ok = np.array([self.mem[i][0] is not None for i in ids], dtype=bool)
        bbox = np.array([self.mem[i][2] if self.mem[i][2] is not None else (0, 0, 0, 0) for i in ids], dtype=np.float32)
        det = np.array([self.mem[i][1] for i in ids], dtype=np.float32)
        nf = np.array([self.mem[i][3] for i in ids], dtype=np.int32)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez(tmp, ids=np.array(ids, dtype=str), emb=emb, ok=ok, bbox=bbox, det=det, nf=nf)
        os.replace(tmp, self.path)


class CleanEditCache:
    """cache/clean_edits/<editor>/<prompt_id>_s<strength>_seed<k>/<image_id>.jpg (E0, shared by all methods).

    Stored as JPEG q90 (small). ``edit_lpips`` is defined on JPEG-q90 versions of BOTH edits, so the in-job value and
    the later ``--stage lpips`` recomputation from saved files are identical."""

    def __init__(self, root: Path, ext: str = ".jpg", quality: int = 90):
        self.root, self.ext, self.quality = Path(root), ext, int(quality)

    def path(self, editor: str, prompt_id: str, strength: str, seed: int, image_id: str) -> Path:
        return self.root / editor / f"{prompt_id}_s{strength or 'na'}_seed{seed}" / f"{image_id}{self.ext}"

    def get(self, *a):
        p = self.path(*a)
        return _open_rgb(p) if p.is_file() else None

    def put(self, img, *a) -> Path:
        p = self.path(*a)
        if self.ext == ".png":
            _save_png(img, p)
        else:
            _save_jpeg(img, p, self.quality)
        return p


# ------------------------------------------------------------------------------------ simple image metrics
def psnr_pil(a, b) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse == 0 else float(20 * math.log10(255.0) - 10 * math.log10(mse))


def ssim_pil(a, b) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return NAN
    x = np.asarray(a.convert("RGB"))
    y = np.asarray(b.convert("RGB"))
    return float(structural_similarity(x, y, channel_axis=2, data_range=255))


# ----------------------------------------------------------------------------------------------- backends
@dataclass
class Backends:
    editor_factory: Callable[[str], Any]            # name -> Editor (unloaded)
    purifier_factory: Callable[[str], Any]          # canonical name -> EvalPurifier
    face: Any                                       # .analyze(img) -> FaceResult
    clip: Any = None                                # .similarity(a, b)
    lpips: Any = None                               # .distance(a, b)
    mask_provider: Optional[Callable[[str], Any]] = None
    device: str = "cuda"


def default_backends(cfg: dict, device: str = "cuda", parsing_dir: Optional[str] = None, cache_dir: Optional[str] = None,
                     source_store: Optional[SourceStore] = None, with_clip: bool = True, with_lpips: bool = True) -> Backends:
    """Real backends (need GPU libs). Not exercised in the CPU test-suite."""
    from .clip_metric import ClipScorer, LpipsScorer
    from .editors import build_editor
    from .faces import FaceScorer
    from .masks import MaskProvider
    from .purify_attacks import build_purifier

    face = FaceScorer.from_config(cfg)
    mc = cfg["eval"].get("masks", {}) or {}
    masks = MaskProvider(os.path.join(cache_dir, "masks") if cache_dir else None, parsing_dir,
                         int(cfg["eval"].get("image_size", 512)),
                         face_bbox_fn=(source_store.bbox_of if source_store else None),
                         **({"keep_parts": mc["keep_parts"]} if "keep_parts" in mc else {}),
                         dilate_px=int(mc.get("dilate_px", 8)))
    return Backends(
        editor_factory=lambda n: build_editor(n, cfg, face_scorer=face),
        purifier_factory=lambda n: build_purifier(n, cfg, device),
        face=face,
        clip=ClipScorer.from_config(cfg, device) if with_clip else None,
        lpips=LpipsScorer.from_config(cfg, device) if with_lpips else None,
        mask_provider=masks, device=device)


# --- deterministic CPU fakes (dry-run backend and tests) ---
class FakeEditor:
    """Deterministic editor: blends the input with seeded noise. Depends on image, prompt, strength, seed."""
    needs_mask = False

    def __init__(self, name="fake", needs_mask=False, default_strength=0.55):
        self.name, self.needs_mask, self.default_strength = name, needs_mask, default_strength
        self.loaded = False
        self.calls = 0
        self.last_info: dict = {}

    def load(self, device="cpu"):
        self.loaded = True
        return self

    def unload(self):
        self.loaded = False

    def params(self):
        return {"kind": "fake"}

    def edit_batch(self, img, prompt, strength, seeds, mask=None):
        from PIL import Image
        if not self.loaded:
            raise RuntimeError("not loaded")
        if self.needs_mask and mask is None:
            raise ValueError("mask required")
        self.calls += 1
        s = self.default_strength if strength is None else strength
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        outs = []
        for sd in seeds:
            h = int(hashlib.sha256(f"{self.name}|{prompt}|{sd}".encode()).hexdigest()[:8], 16)
            noise = np.random.RandomState(h).uniform(0, 255, arr.shape).astype(np.float32)
            outs.append(Image.fromarray(np.clip((1 - s) * arr + s * noise, 0, 255).astype(np.uint8)))
        return outs


class FakeFaceScorer:
    """Embedding = fixed random projection of a 16x16 grey thumbnail. 'No face' if the image is almost black."""

    def __init__(self, dim=64, dark_thresh=2.0):
        self.proj = np.random.RandomState(0).randn(256, dim).astype(np.float32)
        self.dark_thresh = dark_thresh

    def analyze(self, img):
        from .faces import FaceResult
        g = np.asarray(img.convert("L").resize((16, 16)), dtype=np.float32)
        if g.mean() < self.dark_thresh:
            return FaceResult(None, NAN, None, 0, None)
        v = (g.reshape(-1) - g.mean()) @ self.proj
        n = np.linalg.norm(v)
        w, h = img.size
        return FaceResult((v / max(n, 1e-9)).astype(np.float32), 0.9, (w * 0.25, h * 0.2, w * 0.75, h * 0.8), 1, None)

    def embed(self, img):
        return self.analyze(img).as_tuple()


class FakeClip:
    def similarity(self, a, b):
        x = np.asarray(a.convert("L").resize((8, 8)), dtype=np.float32).ravel()
        y = np.asarray(b.convert("L").resize((8, 8)), dtype=np.float32).ravel()
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


class FakeLpips:
    def distance(self, a, b):
        x = np.asarray(a.convert("RGB"), dtype=np.float32)
        y = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float32)
        return float(np.abs(x - y).mean() / 255.0)


def _fake_mask(image_id):
    from PIL import Image
    return Image.new("L", (64, 64), 255)


def fake_backends(cfg: dict) -> Backends:
    from .purify_attacks import build_purifier
    ev = cfg["eval"]
    return Backends(
        editor_factory=lambda n: FakeEditor(n, needs_mask=bool(ev["editors"][n].get("needs_mask", False)),
                                            default_strength=ev["editors"][n].get("default_strength") or 0.55),
        purifier_factory=lambda n: build_purifier(n, cfg, "cpu"),
        face=FakeFaceScorer(), clip=FakeClip(), lpips=FakeLpips(),
        mask_provider=_fake_mask, device="cpu")


# --------------------------------------------------------------------------------------------- the runner
def _sync():
    if "torch" in sys.modules:
        try:
            t = sys.modules["torch"]
            if t.cuda.is_available():
                t.cuda.synchronize()
        except Exception:
            pass


def _gpu_name() -> str:
    if "torch" in sys.modules:
        try:
            t = sys.modules["torch"]
            if t.cuda.is_available():
                return t.cuda.get_device_name(0)
        except Exception:
            pass
    return "cpu"


class EditRunner:
    """Runs jobs, writes rows. See module docstring."""

    def __init__(self, cfg: dict, backends: Backends, out_dir: Path, source_dir: Path, split: str,
                 protected_dir: Optional[Path] = None, run_id: str = "run", protect_run_id: str = "",
                 cache_dir: Optional[Path] = None, csv_name: str = "metrics.csv", log: Callable[[str], None] = print,
                 allow_generate_clean: bool = True, shard_tag: str = ""):
        self.cfg, self.ev, self.b = cfg, cfg["eval"], backends
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else self.out_dir / "cache"
        self.source_dir, self.split = Path(source_dir), split
        self.protected_dir = str(protected_dir) if protected_dir else None   # may contain '{method}'
        self.run_id, self.protect_run_id = run_id, protect_run_id
        self.log = log
        self.allow_generate_clean = allow_generate_clean
        self.seeds: List[int] = [int(s) for s in self.ev["seeds"]]
        self.base_seed = int(cfg.get("seed", 2026))
        self.writer = ResultsWriter(self.out_dir / csv_name)
        self.store = SourceStore(self.source_dir, split, self.cache_dir, backends.face)
        ei = self.ev.get("edit_images", {}) or {}
        self.img_fmt = ei.get("format", "jpg")
        self.img_q = int(ei.get("quality", 90))
        self.clean = CleanEditCache(self.cache_dir / "clean_edits", ".png" if self.img_fmt == "png" else ".jpg", self.img_q)
        self.emb_cache = EmbCache(self.cache_dir / "arcface" / "edit_embeddings.npz")
        self.git = _git_hash()
        self.ckpt_every = int(self.ev.get("checkpoint_every", 50))
        self.purify_s: Dict[tuple, float] = {}
        self.edit_s_total = 0.0
        self.rows_written = 0
        self.edits_by_editor: Dict[str, float] = {}
        self._inv: Dict[tuple, tuple] = {}
        self._t_start = time.time()
        self._total_rows = 0

    # ---- paths
    def protected_path(self, method: str, image_id: str) -> Path:
        if method == "clean":
            return self.store.image_path(image_id)
        if self.protected_dir is None:
            raise ValueError(f"protected_dir required for method {method}")
        d = Path(self.protected_dir.format(method=method))
        p = find_image(d, image_id)
        if p is None:
            raise FileNotFoundError(f"protected image for {image_id} not found in {d}")
        return p

    def purified_path(self, method: str, purifier: str, image_id: str) -> Path:
        # keyed by protect run AND method: one runner may handle several methods that share a protect_run_id
        tag = f"{self.protect_run_id}/{method}" if self.protect_run_id else method
        return self.cache_dir / "purified" / tag / purifier / f"{image_id}.png"

    def input_path(self, job: Job) -> Path:
        return self.protected_path(job.method, job.image_id) if job.purifier == "none" \
            else self.purified_path(job.method, job.purifier, job.image_id)

    def edit_save_path(self, job: Job, seed: int) -> Path:
        return self.out_dir / "edits" / job.method / job.purifier / job.editor / \
            f"{job.prompt_id}_s{job.strength or 'na'}_seed{seed}" / f"{job.image_id}{self.clean.ext}"

    def should_save(self, job: Job) -> bool:
        se = self.ev.get("save_edits", {})
        if se.get("all", False):
            return True
        if job.image_index < int(se.get("sample_first_n", 10)):
            return True
        return job.method in se.get("headline_methods", []) and job.purifier in se.get("headline_purifiers", ["none"])

    def gen_seed(self, image_id: str, k: int) -> int:
        return derive_seed(self.base_seed, "edit", image_id, k)

    def purify_seed(self, image_id: str, purifier: str) -> int:
        return derive_seed(self.base_seed, "purify", image_id, purifier)

    # ---- stages
    def run(self, jobs: Sequence[Job]) -> dict:
        todo = [j for j in jobs if any(k not in self.writer.done for k in j.row_keys(self.seeds))]
        self._total_rows = len(jobs) * len(self.seeds)
        self.log(f"[runner] {len(jobs)} jobs, {len(todo)} to do ({len(jobs) - len(todo)} already complete)")
        # prefetch source faces (cached on disk)
        for iid in sorted({j.image_id for j in todo}):
            self.store.face_of(iid)
        self.store.save()
        self.unavailable_purifiers = set()
        self._purify_stage(todo)
        if self.unavailable_purifiers:
            skipped = [j for j in todo if j.purifier in self.unavailable_purifiers]
            self.log(f"[runner] SKIPPING {len(skipped)} jobs with unavailable purifiers: {sorted(self.unavailable_purifiers)}")
            todo = [j for j in todo if j.purifier not in self.unavailable_purifiers]
        by_editor: Dict[str, List[Job]] = {}
        for j in todo:
            by_editor.setdefault(j.editor, []).append(j)
        failed_editors: Dict[str, str] = {}
        for ed_name in sorted(by_editor):
            jl = sorted(by_editor[ed_name], key=lambda j: (j.strength, j.prompt_idx, j.purifier, j.method != "clean", j.method, j.image_index))
            # one broken editor must not take the other editors (and a night of GPU time) down with it (S05: FaceID load
            # failure aborted every editor after it). Rows already written are kept; resume re-runs only what is missing.
            editor = None
            try:
                editor = self.b.editor_factory(ed_name)
                editor.load(self.b.device)
                for j in jl:
                    self._process(editor, j)
            except Exception as e:  # noqa: BLE001
                import traceback
                self.log(f"[runner] EDITOR FAILED: {ed_name}: {type(e).__name__}: {e}" + chr(10) + traceback.format_exc())
                failed_editors[ed_name] = f"{type(e).__name__}: {e}"
            finally:
                if editor is not None:
                    try:
                        editor.unload()
                    except Exception:  # noqa: BLE001
                        pass
        self.finish()
        if failed_editors:
            (Path(self.out_dir) / "failed_editors.json").write_text(json.dumps(failed_editors, indent=2), encoding="utf-8")
            raise RuntimeError(f"editors failed (other editors completed): {sorted(failed_editors)}")
        return {"jobs": len(jobs), "executed": len(todo), "rows": self.rows_written}

    def _purify_stage(self, todo: Sequence[Job]):
        from .purify_attacks import NotAvailable
        if not hasattr(self, 'unavailable_purifiers'):
            self.unavailable_purifiers = set()
        need: Dict[str, set] = {}
        for j in todo:
            if j.purifier != "none":
                need.setdefault(j.purifier, set()).add((j.method, j.image_id))
        names = sorted(need)
        built = {}
        for n in names:
            try:
                built[n] = self.b.purifier_factory(n)
            except NotAvailable as e:
                self.unavailable_purifiers.add(n)
                self.log(f"[purify] {n} NOT AVAILABLE, its jobs are skipped: {e}")
        names = [n for n in names if n in built]
        groups: Dict[str, List[str]] = {}
        for n in names:
            groups.setdefault(getattr(built[n], "share_key", None) or n, []).append(n)
        for _, grp in groups.items():
            loaded = []
            try:
                for n in grp:
                    pairs = [(m, i) for (m, i) in sorted(need[n]) if not self.purified_path(m, n, i).is_file()]
                    if not pairs:
                        continue
                    try:
                        built[n].load(self.b.device)
                    except NotAvailable as e:
                        self.unavailable_purifiers.add(n)
                        self.log(f"[purify] {n} NOT AVAILABLE, its jobs are skipped: {e}")
                        continue
                    loaded.append(n)
                    for (m, iid) in pairs:
                        t0 = time.perf_counter()
                        img = _open_rgb(self.protected_path(m, iid))
                        out = built[n].apply(img, self.purify_seed(iid, n))
                        _sync()
                        _save_png(out, self.purified_path(m, n, iid))
                        self.purify_s[(m, n, iid)] = time.perf_counter() - t0
            finally:
                for n in grp:
                    try:
                        built[n].unload()
                    except Exception:
                        pass

    def _inv_metrics(self, job: Job):
        key = (job.method, job.image_id)
        if key not in self._inv:
            if job.method == "clean":
                self._inv[key] = (NAN, NAN, NAN)
            else:
                s = self.store.image(job.image_id)
                p = _open_rgb(self.protected_path(job.method, job.image_id))
                lp = self.b.lpips.distance(s, p) if self.b.lpips is not None else NAN
                self._inv[key] = (psnr_pil(s, p), ssim_pil(s, p), lp)
        return self._inv[key]

    def _method_eps(self, method: str) -> float:
        if method == "clean":
            return 0.0
        m = (self.cfg.get("methods") or {}).get(method) or {}
        v = (m.get("budget") or {}).get("eps_255", (self.cfg.get("budget") or {}).get("eps_255"))
        return NAN if v is None else float(v)

    def _clean_edits(self, editor, job: Job, mask, prompt: str, sval) -> Optional[list]:
        got = [self.clean.get(job.editor, job.prompt_id, job.strength, k, job.image_id) for k in self.seeds]
        if all(g is not None for g in got):
            return got
        if not self.allow_generate_clean:
            return None
        src = self.store.image(job.image_id)
        outs = editor.edit_batch(src, prompt, sval, [self.gen_seed(job.image_id, k) for k in self.seeds], mask=mask)
        for k, o in zip(self.seeds, outs):
            self.clean.put(o, job.editor, job.prompt_id, job.strength, k, job.image_id)
        return [self.clean.get(job.editor, job.prompt_id, job.strength, k, job.image_id) for k in self.seeds]

    def _process(self, editor, job: Job):
        sval = job.strength_value()
        prompt = get_prompt(self.ev, job.editor, job.prompt_idx)
        x_path = self.input_path(job)
        x_in = _open_rgb(x_path)
        mask = self.b.mask_provider(job.image_id) if getattr(editor, "needs_mask", False) else None
        is_e0 = job.method == "clean" and job.purifier == "none"
        gseeds = [self.gen_seed(job.image_id, k) for k in self.seeds]
        _sync()
        t0 = time.perf_counter()
        edits = editor.edit_batch(x_in, prompt, sval, gseeds, mask=mask)
        _sync()
        dt = time.perf_counter() - t0
        info = dict(getattr(editor, "last_info", {}) or {})
        self.edit_s_total += dt
        self.edits_by_editor[job.editor] = self.edits_by_editor.get(job.editor, 0.0) + dt
        if is_e0:
            clean_edits = None
            for k, o in zip(self.seeds, edits):
                self.clean.put(o, job.editor, job.prompt_id, job.strength, k, job.image_id)
        else:
            clean_edits = self._clean_edits(editor, job, mask, prompt, sval)
        src_img = self.store.image(job.image_id)
        s_emb, _, _, _ = self.store.face_of(job.image_id)
        inv = self._inv_metrics(job)
        save = self.should_save(job) and not is_e0
        try:
            ed_params = dict(editor.params())
        except Exception:
            ed_params = {}
        ed_params.update({"prompt": prompt, "seed_gen": dict(zip(map(str, self.seeds), gseeds))})
        ed_params.update(info)
        pur_params = {} if job.purifier == "none" else self._pur_params(job.purifier)
        pur_s = self.purify_s.get((job.method, job.purifier, job.image_id), 0.0)
        for i, k in enumerate(self.seeds):
            row_key = (job.method, job.image_id, str(k), job.editor, job.prompt_id, job.strength, job.purifier)
            if row_key in self.writer.done:
                continue
            e_img = edits[i]
            fr = self.b.face.analyze(e_img)
            if is_e0:
                self.emb_cache.put(f"{job.editor}/{job.prompt_id}_s{job.strength or 'na'}_seed{k}/{job.image_id}", fr.emb)
            sim = NAN if (fr.emb is None or s_emb is None) else float(np.dot(fr.emb, s_emb) /
                                                                    (np.linalg.norm(fr.emb) * np.linalg.norm(s_emb) + 1e-12))
            clip_i = self.b.clip.similarity(src_img, e_img) if self.b.clip is not None else NAN
            if is_e0 or clean_edits is None or self.b.lpips is None:
                lp = NAN
            else:
                lp = self.b.lpips.distance(clean_edits[i], jpeg_roundtrip(e_img, self.img_q) if self.img_fmt != "png" else e_img)
            e_path = ""
            if save:
                e_path = str(self.edit_save_path(job, k))
                if self.img_fmt == "png":
                    _save_png(e_img, Path(e_path))
                else:
                    _save_jpeg(e_img, Path(e_path), self.img_q)
            c_path = str(self.clean.path(job.editor, job.prompt_id, job.strength, k, job.image_id)) \
                if (clean_edits is not None or is_e0) else ""
            if is_e0:
                e_path = c_path
            row = {
                "run_id": self.run_id, "method": job.method,
                "method_cfg_hash": _cfg_hash((self.cfg.get("methods") or {}).get(job.method, {})),
                "image_id": job.image_id, "split": job.split, "seed": k, "editor": job.editor,
                "prompt_id": job.prompt_id, "strength": job.strength, "purifier": job.purifier,
                "protect_run_id": self.protect_run_id, "eps_255": self._method_eps(job.method),
                "purifier_params": pur_params, "editor_params": ed_params,
                "source_path": str(self.store.image_path(job.image_id)),
                "protected_path": str(self.protected_path(job.method, job.image_id)),
                "purified_path": str(self.purified_path(job.method, job.purifier, job.image_id)) if job.purifier != "none" else "",
                "edit_path": e_path, "clean_edit_path": c_path, "code_git_hash": self.git, "created_utc": _now(),
                "face_detected": int(fr.emb is not None), "n_faces": fr.n_faces, "det_score": fr.det_score,
                "arcface_sim": sim, "fr2_sim": NAN, "edit_lpips": lp, "clip_i": clip_i,
                "inv_psnr": inv[0], "inv_ssim": inv[1], "inv_lpips": inv[2],
                "gpu_seconds_edit": dt / len(self.seeds), "gpu_seconds_purify": pur_s / len(self.seeds),
            }
            self.writer.append(row)
            self.rows_written += 1
            if self.rows_written % self.ckpt_every == 0:
                self.checkpoint()

    def _pur_params(self, name: str) -> dict:
        return dict((self.ev.get("purifiers") or {}).get(name) or {})

    # ---- checkpoint / finish
    def checkpoint(self):
        self.store.save()
        self.emb_cache.save()
        el = time.time() - self._t_start
        done = self.writer.n_rows
        eta = (el / max(self.rows_written, 1)) * max(self._total_rows - done, 0)
        _atomic_json(self.out_dir / "progress.json", {
            "rows_done": done, "rows_total": self._total_rows, "elapsed_s": el, "eta_s": eta,
            "updated_utc": _now()})
        self._write_gpu_time()

    def _write_gpu_time(self):
        n = max(self.rows_written, 1)
        _atomic_json(self.out_dir / "gpu_time.json", {
            "total_gpu_seconds": self.edit_s_total + sum(self.purify_s.values()),
            "edit_seconds": self.edit_s_total, "purify_seconds": sum(self.purify_s.values()),
            "n_rows": self.rows_written, "per_edit_mean_s": self.edit_s_total / n,
            "edit_seconds_by_editor": self.edits_by_editor, "gpu_name": _gpu_name(),
            "note": "wall-clock around edit/purify calls (cuda-synchronised); excludes model load and scoring"})

    def finish(self):
        self.checkpoint()


def recompute_lpips(metrics_csv: Path, clean_root: Path, lpips, quality: int = 90, ext: str = ".jpg") -> dict:
    """Cheap later pass: fill ``edit_lpips`` (and ``clean_edit_path``) for rows where it is NaN, from the saved edit
    images (``edit_path``) and the E0 clean-edit images under ``clean_root`` (<editor>/<prompt>_s<strength>_seed<k>/<id>).
    Rewrites the CSV atomically. Rows of E0 itself (method=clean, purifier=none) stay NaN."""
    metrics_csv = Path(metrics_csv)
    with open(metrics_csv, newline="") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames
        rows = list(rd)
    cc = CleanEditCache(clean_root, ext, quality)
    filled = missing = 0
    for r in rows:
        if r["method"] == "clean" and r["purifier"] == "none":
            continue
        if r["edit_lpips"] not in ("nan", "", None):
            continue
        cp = cc.path(r["editor"], r["prompt_id"], r["strength"], int(r["seed"]), r["image_id"])
        ep = r.get("edit_path", "")
        if not (ep and Path(ep).is_file() and cp.is_file()):
            missing += 1
            continue
        r["edit_lpips"] = _fmt(float(lpips.distance(_open_rgb(cp), _open_rgb(ep))))
        r["clean_edit_path"] = str(cp)
        filled += 1
    tmp = metrics_csv.with_name(metrics_csv.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
    os.replace(tmp, metrics_csv)
    return {"filled": filled, "missing_inputs": missing, "rows": len(rows)}


# ------------------------------------------------------------------------------- thresholds and summaries
def psr_threshold_from_impostors(sims: Sequence[float], percentile: float = 95.0) -> float:
    s = np.asarray([x for x in sims if not (x is None or math.isnan(x))], dtype=np.float64)
    if s.size == 0:
        raise ValueError("no impostor similarities")
    return float(np.percentile(s, percentile))


def impostor_sims(src_embs: Mapping[str, np.ndarray], edit_embs: Iterable[Tuple[str, np.ndarray]]) -> np.ndarray:
    """ArcFace sim between each edit (of source ``image_id``) and every OTHER identity's source embedding."""
    ids = [i for i, e in src_embs.items() if e is not None]
    if len(ids) < 2:
        raise ValueError("need >= 2 sources with a detected face")
    S = np.stack([src_embs[i] for i in ids]).astype(np.float64)
    S /= np.clip(np.linalg.norm(S, axis=1, keepdims=True), 1e-12, None)
    pos = {i: k for k, i in enumerate(ids)}
    out = []
    for iid, e in edit_embs:
        if e is None:
            continue
        v = np.asarray(e, np.float64)
        v = v / max(np.linalg.norm(v), 1e-12)
        sims = S @ v
        mask = np.ones(len(ids), bool)
        if iid in pos:
            mask[pos[iid]] = False
        out.append(sims[mask])
    return np.concatenate(out) if out else np.zeros(0)


def compute_psr_thresholds(clean_root: Path, store: SourceStore, face, image_ids: Sequence[str],
                           editors: Sequence[str], percentile: float = 95.0, emb_cache: Optional[EmbCache] = None) -> dict:
    """Per-editor and pooled thresholds from E0 clean edits (different identities = different image ids; assumes one
    image per identity in the split, see V2_PLAN R19). Writes nothing; caller saves JSON."""
    src = {i: store.face_of(i)[0] for i in image_ids}
    res = {"percentile": percentile, "per_editor": {}, "n_pairs": {}, "n_sources_with_face": sum(v is not None for v in src.values())}
    pooled: List[np.ndarray] = []
    for ed in editors:
        base = Path(clean_root) / ed
        items = []
        if base.is_dir():
            for sub in sorted(os.listdir(base)):
                for iid in image_ids:
                    p = find_image(base / sub, iid)
                    if p is None:
                        continue
                    key = f"{ed}/{sub}/{iid}"
                    c = emb_cache.get(key) if emb_cache else "MISSING"
                    if isinstance(c, str):
                        c = face.analyze(_open_rgb(p)).emb
                        if emb_cache:
                            emb_cache.put(key, c)
                    items.append((iid, c))
        if not items:
            continue
        s = impostor_sims(src, items)
        if s.size:
            res["per_editor"][ed] = psr_threshold_from_impostors(s, percentile)
            res["n_pairs"][ed] = int(s.size)
            pooled.append(s)
    if pooled:
        allp = np.concatenate(pooled)
        res["pooled"] = psr_threshold_from_impostors(allp, percentile)
        res["n_pairs"]["pooled"] = int(allp.size)
    return res


def psr_hit(face_detected, sim, thr: float) -> bool:
    """Protection success: no face found, or similarity below the impostor threshold."""
    if not face_detected or sim is None or (isinstance(sim, float) and math.isnan(sim)):
        return True
    return bool(sim < thr)


def summarize_csv(csv_path: Path, thresholds: Optional[dict] = None, by: Sequence[str] = ("method", "purifier", "editor", "strength")) -> list:
    """Minimal summary (n and mean; CIs belong to analysis/A5). Averages seeds/prompts per image first."""
    import pandas as pd
    df = pd.read_csv(csv_path, keep_default_na=True, dtype={"strength": str, "image_id": str}).fillna({"strength": ""})
    out = []
    for key, g in df.groupby(list(by), dropna=False):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        thr = None
        if thresholds:
            thr = (thresholds.get("per_editor") or {}).get(row.get("editor"), thresholds.get("pooled"))
        g = g.copy()
        g["no_face"] = (g["face_detected"] == 0).astype(float)
        if thr is not None:
            g["psr_hit"] = [float(psr_hit(fd, s, thr)) for fd, s in zip(g["face_detected"], g["arcface_sim"])]
        per_img = g.groupby("image_id").mean(numeric_only=True)
        row.update({"n_images": int(len(per_img)), "n_edits": int(len(g)),
                    "mean_arcface_sim": float(per_img["arcface_sim"].mean()),
                    "no_face_rate": float(g["no_face"].mean()),
                    "mean_clip_i": float(per_img["clip_i"].mean()), "mean_edit_lpips": float(per_img["edit_lpips"].mean())})
        if thr is not None:
            row["psr"] = float(per_img["psr_hit"].mean())
            row["psr_threshold"] = float(thr)
        out.append(row)
    return out


# ------------------------------------------------------------------------------------- Optuna entry point
@dataclass
class EditorSpec:
    """One editing configuration for ``score_protected``. ``editor`` must already be ``load()``-ed."""
    editor: Any
    prompt: str
    strength: Optional[float] = None
    name: str = ""

    def label(self) -> str:
        return self.name or getattr(self.editor, "name", "editor")


def _mean(xs):
    xs = [x for x in xs if not (x is None or (isinstance(x, float) and math.isnan(x)))]
    return float(np.mean(xs)) if xs else NAN


def score_protected(images_protected: Mapping[str, Any], images_clean: Mapping[str, Any],
                    purifiers: Sequence[Any], editors: Sequence[EditorSpec], seeds: Sequence[int], *,
                    face, threshold: Optional[float] = None, clip=None, lpips=None, mask_provider=None,
                    clean_edit_cache: Optional[dict] = None, percentile: float = 95.0, purify_seed: int = 0,
                    base_seed: int = 2026) -> dict:
    """Programmatic scoring for Optuna (proxy objective) without any files or CSV.

    Arguments
      images_protected / images_clean : {image_id: PIL.Image} (same keys). Protected images must be the published
          uint8 images (quantise before calling).
      purifiers : names (resolved with ``build_purifier``) or ``EvalPurifier`` instances; 'none' allowed. Purifiers
          are loaded/unloaded here; editors are NOT (pass loaded ones, so Optuna trials reuse them).
      editors : list of ``EditorSpec`` (loaded editor, prompt, strength).
      seeds : edit seeds (ints); generator seed = derive_seed(base_seed, "edit", image_id, seed), identical for the
          clean and protected edit.
      face : object with ``analyze(img)``. clip / lpips optional.
      threshold : PSR threshold on ArcFace similarity. If None it is computed from the impostor distribution of the
          CLEAN edits of these same images (needs >= 3 images; noisy for small n: precompute once from E0 instead
          and pass it for a stable objective).
      clean_edit_cache : dict reused across calls ({(spec label, image_id, seed): PIL}); pass the same dict every trial
          so clean edits are generated once.

    Returns dict with
      psr            : mean over images of the per-image PSR (an edit is a hit if no face or sim < threshold),
                       averaged over purifiers x editors x seeds
      mean_sim       : mean ArcFace similarity over edits with a face (NaN if none)
      no_face_rate   : share of edits with no detected face
      mean_edit_lpips, mean_clip_i : means (NaN if the scorer is absent)
      threshold, n_edits
      per_cell       : {"<purifier>|<editor label>": {"psr":..., "mean_sim":..., "no_face_rate":..., "n":...}}
    """
    from .purify_attacks import build_purifier, EvalPurifier
    ids = sorted(images_protected)
    if cache_missing := [i for i in ids if i not in images_clean]:
        raise KeyError(f"images_clean lacks {cache_missing}")
    cache = clean_edit_cache if clean_edit_cache is not None else {}
    src = {i: face.analyze(images_clean[i]).emb for i in ids}

    def gseeds(iid):
        return [derive_seed(base_seed, "edit", iid, s) for s in seeds]

    def clean_edits(spec, iid):
        key = (spec.label(), iid, tuple(seeds))
        if key not in cache:
            mask = mask_provider(iid) if (mask_provider and getattr(spec.editor, "needs_mask", False)) else None
            cache[key] = spec.editor.edit_batch(images_clean[iid], spec.prompt, spec.strength, gseeds(iid), mask=mask)
        return cache[key]

    if threshold is None:
        if len(ids) < 3:
            raise ValueError("need >= 3 images to derive a threshold; pass threshold=")
        items = []
        for spec in editors:
            for iid in ids:
                for o in clean_edits(spec, iid):
                    items.append((iid, face.analyze(o).emb))
        threshold = psr_threshold_from_impostors(impostor_sims(src, items), percentile)

    cells: Dict[str, dict] = {}
    all_hits, all_sims, all_nf, all_lp, all_clip = [], [], [], [], []
    for pur in purifiers:
        p = pur if isinstance(pur, EvalPurifier) or hasattr(pur, "apply") else build_purifier(pur)
        pname = getattr(p, "name", str(pur))
        p.load()
        try:
            xin = {i: (p.apply(images_protected[i], derive_seed(base_seed, "purify", i, pname) + purify_seed)
                       if pname != "none" else images_protected[i]) for i in ids}
        finally:
            p.unload()
        for spec in editors:
            hits, sims, nfs = {}, [], []
            for iid in ids:
                mask = mask_provider(iid) if (mask_provider and getattr(spec.editor, "needs_mask", False)) else None
                outs = spec.editor.edit_batch(xin[iid], spec.prompt, spec.strength, gseeds(iid), mask=mask)
                ce = clean_edits(spec, iid) if lpips is not None else None
                h = []
                for si, o in enumerate(outs):
                    fr = face.analyze(o)
                    sim = NAN if (fr.emb is None or src[iid] is None) else float(
                        np.dot(fr.emb, src[iid]) / (np.linalg.norm(fr.emb) * np.linalg.norm(src[iid]) + 1e-12))
                    h.append(psr_hit(fr.emb is not None, sim, threshold))
                    nfs.append(float(fr.emb is None))
                    if fr.emb is not None and not math.isnan(sim):
                        sims.append(sim)
                    if lpips is not None:
                        all_lp.append(lpips.distance(ce[si], o))
                    if clip is not None:
                        all_clip.append(clip.similarity(images_clean[iid], o))
                hits[iid] = float(np.mean(h))
            cells[f"{pname}|{spec.label()}"] = {"psr": float(np.mean(list(hits.values()))), "mean_sim": _mean(sims),
                                                "no_face_rate": float(np.mean(nfs)) if nfs else NAN,
                                                "n": len(nfs)}
            all_hits.extend(hits.values())
            all_sims.extend(sims)
            all_nf.extend(nfs)
    return {"psr": float(np.mean(all_hits)) if all_hits else NAN, "mean_sim": _mean(all_sims),
            "no_face_rate": float(np.mean(all_nf)) if all_nf else NAN, "mean_edit_lpips": _mean(all_lp),
            "mean_clip_i": _mean(all_clip), "threshold": float(threshold), "n_edits": len(all_nf), "per_cell": cells}
