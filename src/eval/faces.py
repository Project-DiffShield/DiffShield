"""Face identity scoring (Half B only). ArcFace via insightface buffalo_l.

Contract (frozen): ``FaceScorer.embed(img) -> (emb | None, det_score, bbox)``.
``analyze`` additionally reports the number of detected faces and 5 landmarks.

No-face policy (handoff): an edit with no detectable face counts as a protection success, but the
no-face rate is reported separately. This module only returns ``None``; the policy is applied in
``edit_runner.psr_hit``.

insightface / onnxruntime are imported lazily so that the rest of the harness (and the CPU tests with
fake scorers) never needs them. Not exercised locally (insightface is not installed here): unverified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import contextlib
import random
import time

import numpy as np


@dataclass
class FaceResult:
    emb: Optional[np.ndarray]            # L2-normalised 512-d ArcFace embedding of the largest face, or None
    det_score: float                     # detector score of that face (NaN if no face)
    bbox: Optional[Tuple[float, float, float, float]]   # x1, y1, x2, y2 in pixels of the input image
    n_faces: int = 0
    kps: Optional[np.ndarray] = None     # (5, 2) landmarks

    def as_tuple(self):
        return self.emb, self.det_score, self.bbox


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (robust to un-normalised input)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """All-pairs cosine similarity, shape (len(A), len(B))."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
    B = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
    return A @ B.T


@contextlib.contextmanager
def _init_lock(path: str = "/tmp/diffshield_insightface_init.lock"):
    """Cross-process lock around FaceAnalysis init (no-op where fcntl is missing, e.g. Windows tests)."""
    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def choose_providers(prefer_gpu: bool = True, available: Optional[Sequence[str]] = None) -> list:
    """onnxruntime provider list: CUDA first if present, else CPU. ``available`` is injectable for tests."""
    if available is None:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except Exception:
            available = ["CPUExecutionProvider"]
    provs = []
    if prefer_gpu and "CUDAExecutionProvider" in available:
        provs.append("CUDAExecutionProvider")
    provs.append("CPUExecutionProvider")
    return provs


class FaceScorer:
    """insightface FaceAnalysis wrapper (detection + ArcFace recognition only).

    Largest face (by bbox area) is used, as the datasets are portraits with one subject.
    """

    def __init__(self, pack: str = "buffalo_l", det_size: Sequence[int] = (640, 640), prefer_gpu: bool = True,
                 root: Optional[str] = None, log=print, pad_frac: float = 0.4):
        # pad_frac: border added on every side (fraction of the image size) before detection. S03/S05 found that the
        # insightface detector misses 38 of 100 tightly cropped CelebA-HQ test photos (face fills the frame); the same
        # happens on edits. Bbox / keypoints are shifted back, so callers see original-image coordinates.
        self.pad_frac = float(pad_frac)
        self.pack = pack
        self.det_size = tuple(det_size)
        self.prefer_gpu = prefer_gpu
        self.root = root
        self.log = log
        self._app: Any = None
        self.providers: list = []

    @classmethod
    def from_config(cls, cfg: dict, **kw) -> "FaceScorer":
        a = (cfg.get("eval", {}).get("models", {}) or {}).get("arcface", {}) or {}
        return cls(pack=a.get("pack", "buffalo_l"), det_size=a.get("det_size", (640, 640)),
                   pad_frac=float(a.get("pad_frac", 0.4)), **kw)

    def load(self):
        if self._app is not None:
            return self
        from insightface.app import FaceAnalysis  # lazy
        try:  # onnxruntime >= 1.21: find the CUDA/cuDNN libs shipped with torch's nvidia-* wheels
            import onnxruntime as _ort
            if hasattr(_ort, "preload_dlls"):
                _ort.preload_dlls()
        except Exception as e:  # pragma: no cover - environment dependent
            self.log(f"[faces] onnxruntime.preload_dlls failed ({e}); CPU fallback is possible")
        self.providers = choose_providers(self.prefer_gpu)
        kw = {"name": self.pack, "providers": self.providers, "allowed_modules": ["detection", "recognition"]}
        if self.root:
            kw["root"] = self.root
        app = None
        # Two shard processes download/unpack the same model pack. S05: one hit FileExistsError, retried and then loaded a
        # half-unpacked pack WITHOUT the recognition model (embeddings came out as NaN 0-d arrays). So: serialise the init
        # with a file lock and refuse to continue unless detection AND recognition are loaded.
        with _init_lock():
            for attempt in range(6):
                try:
                    app = FaceAnalysis(**kw)
                    if "recognition" not in getattr(app, "models", {}) or "detection" not in app.models:
                        raise RuntimeError(f"insightface pack incomplete, models loaded: {sorted(getattr(app, 'models', {}))}")
                    break
                except (FileExistsError, OSError, RuntimeError) as e:
                    if attempt == 5:
                        raise
                    self.log(f"[faces] FaceAnalysis init failed ({e}); retry {attempt + 1}/5")
                    time.sleep(3 + 3 * attempt + random.random() * 3)
        app.prepare(ctx_id=0 if "CUDAExecutionProvider" in self.providers else -1, det_size=self.det_size)
        self._app = app
        self.log(f"[faces] insightface {self.pack} providers={self.providers}")
        return self

    def analyze(self, img) -> FaceResult:
        self.load()
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()   # RGB -> BGR for insightface/cv2
        pad = int(round(self.pad_frac * max(arr.shape[:2])))
        if pad > 0:
            arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=127)
        faces = self._app.get(arr)
        if not faces:
            return FaceResult(None, float("nan"), None, 0, None)
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        f = faces[int(np.argmax(areas))]
        emb = getattr(f, "normed_embedding", None)
        if emb is None and getattr(f, "embedding", None) is None:
            raise RuntimeError("insightface returned a face without an embedding: recognition model not loaded")
        if emb is None:
            e = np.asarray(f.embedding, dtype=np.float64)
            emb = e / max(np.linalg.norm(e), 1e-12)
        bbox = tuple(float(v) - pad for v in f.bbox[:4])
        kps = (np.asarray(f.kps) - pad) if getattr(f, "kps", None) is not None else None
        return FaceResult(np.asarray(emb, dtype=np.float32), float(f.det_score), bbox, len(faces), kps)

    def embed(self, img):
        """Frozen contract: (emb | None, det_score, bbox)."""
        return self.analyze(img).as_tuple()

    @staticmethod
    def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return float("nan")
        return cosine(a, b)

    def unload(self):
        self._app = None
