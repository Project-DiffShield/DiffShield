"""CLIP-I (source vs edit image similarity) and edit LPIPS (clean-edit vs protected-edit). Half B only.

CLIP-I uses a *different* CLIP than the protection loss (ViT-L/14 vs the loss's B/32) on purpose, so that L_sem
cannot game the metric (V2_PLAN R11). torch / transformers / lpips are imported lazily.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _resize_like(img, ref):
    if img.size != ref.size:
        from PIL import Image
        return img.resize(ref.size, Image.BICUBIC)
    return img


class ClipScorer:
    """Cosine similarity of CLIP image embeddings. ``similarity(a, b) -> float`` in [-1, 1]."""

    def __init__(self, model_id: str = "openai/clip-vit-large-patch14", device: Optional[str] = None,
                 revision: Optional[str] = None, fp16: bool = True):
        self.model_id, self.device, self.revision, self.fp16 = model_id, device, revision, fp16
        self.model = None
        self.processor = None

    @classmethod
    def from_config(cls, cfg: dict, device=None) -> "ClipScorer":
        m = cfg["eval"]["models"]["clip_eval"]
        return cls(m["id"], device, m.get("revision"))

    def load(self):
        if self.model is not None:
            return self
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        dt = torch.float16 if (self.fp16 and dev.startswith("cuda")) else torch.float32
        self.model = CLIPModel.from_pretrained(self.model_id, revision=self.revision, torch_dtype=dt).to(dev).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id, revision=self.revision)
        return self

    def embed(self, imgs) -> np.ndarray:
        import torch
        self.load()
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        inp = self.processor(images=[i.convert("RGB") for i in imgs], return_tensors="pt")
        pix = inp["pixel_values"].to(self.device, dtype=next(self.model.parameters()).dtype)
        with torch.no_grad():
            f = self.model.get_image_features(pixel_values=pix)
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        return f.cpu().numpy()

    def similarity(self, a, b) -> float:
        e = self.embed([a, b])
        return float(np.dot(e[0], e[1]))

    def unload(self):
        self.model = None
        self.processor = None


class LpipsScorer:
    """LPIPS distance between two PIL images (same size; the second is resized to the first if needed)."""

    def __init__(self, net: str = "vgg", device: Optional[str] = None):
        self.net, self.device = net, device
        self.fn = None

    @classmethod
    def from_config(cls, cfg: dict, device=None) -> "LpipsScorer":
        return cls(cfg["eval"]["models"].get("lpips_net", "vgg"), device)

    def load(self):
        if self.fn is not None:
            return self
        import torch
        import lpips
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        self.fn = lpips.LPIPS(net=self.net, verbose=False).to(dev).eval()
        return self

    @staticmethod
    def _to_tensor(img, device):
        import torch
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    def distance(self, a, b) -> float:
        import torch
        self.load()
        b = _resize_like(b, a)
        with torch.no_grad():
            d = self.fn(self._to_tensor(a, self.device), self._to_tensor(b, self.device))
        return float(d.reshape(-1)[0].item())

    # alias used by the runner
    __call__ = distance

    def unload(self):
        self.fn = None


def edit_lpips(scorer, clean_edit, protected_edit) -> float:
    """LPIPS(clean-image edit, protected-image edit) with the same editor/prompt/strength/seed."""
    return scorer.distance(clean_edit, protected_edit)
