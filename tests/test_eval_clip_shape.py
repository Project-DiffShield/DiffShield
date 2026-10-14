"""CLIP-I embedding must work whether the vision tower returns a tensor-like pooled output (transformers 4 and 5)."""
import types

import numpy as np
import torch
from PIL import Image

from src.eval.clip_metric import ClipScorer


class _Vis(torch.nn.Module):
    def forward(self, pixel_values):  # BaseModelOutputWithPooling-like (what transformers 5 returns from get_image_features)
        return types.SimpleNamespace(pooler_output=pixel_values.flatten(1)[:, :8])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _Vis()
        self.visual_projection = torch.nn.Identity()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def get_image_features(self, **kw):  # the S05 failure: not a tensor
        return types.SimpleNamespace(pooler_output=None)


def test_embed_does_not_depend_on_get_image_features_return_type():
    c = ClipScorer.__new__(ClipScorer)
    c.model, c.device = _Model(), "cpu"
    c.load = lambda: c
    c.processor = lambda images, return_tensors: {"pixel_values": torch.rand(len(images), 3, 4, 4)}
    e = c.embed([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])
    assert e.shape == (2, 8) and np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
