"""Perturbation-budget conversions.

All budgets in config are on the STANDARD 0-255 scale. Tensors in code are in [-1, 1]
(range 2), so a budget of v/255 standard is v*2/255 on the tensor scale.
"""
from __future__ import annotations

import numpy as np


def to_model_scale(v_255: float) -> float:
    return float(v_255) * 2.0 / 255.0


def to_standard_scale(v_model: float) -> float:
    return float(v_model) * 255.0 / 2.0


def _to_hwc_float(x) -> np.ndarray:
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    if a.ndim == 4:
        a = a[0]
    return a.transpose(1, 2, 0)


def tensor_to_u8_round(x) -> np.ndarray:
    """Published convention (what torchvision.save_image writes: x*255+0.5 then cast): round half up. HWC uint8."""
    a = (_to_hwc_float(x).astype(np.float64) + 1.0) * 127.5
    return np.clip(np.floor(a + 0.5), 0, 255).astype(np.uint8)


def tensor_to_u8_trunc_v1(x) -> np.ndarray:
    """Exact v1 conversion in compute_image_quality_metrics: float32 -> astype(uint8) (truncation)."""
    a = (_to_hwc_float(x) + 1.0) * 127.5
    return a.astype(np.uint8)
