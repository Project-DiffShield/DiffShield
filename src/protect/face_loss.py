"""L_id: identity loss through a DIFFERENTIABLE face embedder (Phase 6; weight ``loss.lambda_id`` = 0
by default, so nothing here is loaded unless lambda_id > 0).

    L_id = cos( emb(source face), emb( face recovered from VAE_decode(VAE_encode(x + delta)) ) )   (minimised)

Embedder choice (documented decision, see V2_PLAN R16)
-------------------------------------------------------
* Evaluation uses insightface ``buffalo_l`` (ArcFace ``w600k_r50``, ONNX, not differentiable) and a
  second recognizer. Training against the SAME network as the judge would inflate PSR (white-box on the
  judge), so the loss uses a different, torch-native network:
  ``facenet_pytorch.InceptionResnetV1(pretrained='vggface2')`` (FaceNet embedding, 512-d, NOT ArcFace).
  Weights: downloaded by facenet-pytorch from its GitHub release on first use (availability not
  verified here). Install with ``pip install --no-deps facenet-pytorch`` (its metadata pins old torch).
* Alternative backend ``onnx2torch``: converts an ONNX recognizer to a differentiable torch module (e.g.
  a user-supplied ArcFace ONNX). If that ONNX is the evaluation model the gaming risk above applies;
  select it only knowingly (``face.backend: onnx2torch``, ``face.onnx_path``).
* No face detector runs inside the loop. The face box is given once on the clean image (``bbox`` =
  (x0, y0, x1, y1) in pixels; default = central 60% crop) and the same crop is applied to the
  reconstruction (a fixed crop, no gradient through alignment). Landmark alignment is left to the
  caller (pass a bbox from Half B's FaceScorer on the clean image).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _crop_resize(x: torch.Tensor, bbox: Optional[Sequence[int]], size: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    if bbox is None:
        bbox = (int(0.2 * W), int(0.2 * H), int(0.8 * W), int(0.8 * H))
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, max(x1, x0 + 8)), min(H, max(y1, y0 + 8))
    return F.interpolate(x[..., y0:y1, x0:x1], size=(size, size), mode="bilinear", align_corners=False)


class FaceEmbedder(nn.Module):
    """Wraps a torch face-recognition net: input image in [-1, 1] (any size) -> L2-normalised embedding."""

    def __init__(self, net: nn.Module, input_size: int = 160, device="cpu"):
        super().__init__()
        self.net = net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.input_size = input_size
        self.device = torch.device(device)

    def embed(self, x: torch.Tensor, bbox: Optional[Sequence[int]] = None) -> torch.Tensor:
        # facenet / ArcFace-style nets normalise to roughly [-1, 1] already ((x*255-127.5)/128)
        e = self.net(_crop_resize(x.float().to(self.device), bbox, self.input_size))
        return F.normalize(e, dim=1)


def load_face_embedder(cfg: dict, device="cuda") -> FaceEmbedder:
    face = cfg.get("face") or {}
    backend = face.get("backend", "facenet")
    if backend == "facenet":
        from facenet_pytorch import InceptionResnetV1  # lazy: optional dependency

        return FaceEmbedder(InceptionResnetV1(pretrained=face.get("pretrained", "vggface2")), 160, device)
    if backend == "onnx2torch":
        import onnx2torch  # lazy: optional dependency

        return FaceEmbedder(onnx2torch.convert(face["onnx_path"]), int(face.get("input_size", 112)), device)
    raise KeyError(f"unknown face backend {backend!r}")


class IdentityLoss:
    """cos(source embedding, embedding of the VAE reconstruction of x+delta); minimised by PGD.

    ``recon`` is a callable image -> VAE reconstruction (differentiable), typically
    ``lambda z: models.decode(models.encode_mean(z, scaled=False), scaled=False)``.
    """

    def __init__(self, embedder, recon: Callable[[torch.Tensor], torch.Tensor], bbox: Optional[Sequence[int]] = None):
        self.emb, self.recon, self.bbox = embedder, recon, bbox
        self.src: Optional[torch.Tensor] = None

    def prepare(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            self.src = self.emb.embed(x, self.bbox)

    def __call__(self, x_pub: torch.Tensor) -> torch.Tensor:
        e = self.emb.embed(self.recon(x_pub), self.bbox)
        return F.cosine_similarity(e, self.src.expand_as(e), dim=1).mean()
