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
                 root: Optional[str] = None, log=print):
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
        return cls(pack=a.get("pack", "buffalo_l"), det_size=a.get("det_size", (640, 640)), **kw)

    def load(self):
        if self._app is not None:
            return self
        from insightface.app import FaceAnalysis  # lazy
        self.providers = choose_providers(self.prefer_gpu)
        kw = {"name": self.pack, "providers": self.providers, "allowed_modules": ["detection", "recognition"]}
        if self.root:
            kw["root"] = self.root
        app = FaceAnalysis(**kw)
        app.prepare(ctx_id=0 if "CUDAExecutionProvider" in self.providers else -1, det_size=self.det_size)
        self._app = app
        self.log(f"[faces] insightface {self.pack} providers={self.providers}")
        return self

    def analyze(self, img) -> FaceResult:
        self.load()
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()   # RGB -> BGR for insightface/cv2
        faces = self._app.get(arr)
        if not faces:
            return FaceResult(None, float("nan"), None, 0, None)
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        f = faces[int(np.argmax(areas))]
        emb = getattr(f, "normed_embedding", None)
        if emb is None:
            e = np.asarray(f.embedding, dtype=np.float64)
            emb = e / max(np.linalg.norm(e), 1e-12)
        bbox = tuple(float(v) for v in f.bbox[:4])
        kps = np.asarray(f.kps) if getattr(f, "kps", None) is not None else None
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
