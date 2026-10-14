"""Invisibility metrics. Must never import src.protect.

Two conventions are reported (see docs/V2_PLAN.md F3):
  v1  : tensors -> uint8 by TRUNCATION (exact replica of phase1_v1 compute_image_quality_metrics)
  png : decode the saved files as uint8 (what an attacker/user actually receives)
"""
from __future__ import annotations

import math

import numpy as np

from src.utils.budget import tensor_to_u8_trunc_v1
from src.utils.io import load_u8


def psnr_u8(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float("inf")


def ssim_u8(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    return float(structural_similarity(a, b, channel_axis=-1, data_range=255, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


def max_abs_diff_u8(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())


def diff_histogram_u8(a: np.ndarray, b: np.ndarray, max_level: int = 16) -> dict:
    d = np.abs(a.astype(np.int64) - b.astype(np.int64)).ravel()
    h = np.bincount(np.minimum(d, max_level), minlength=max_level + 1)
    return {int(i): int(h[i]) for i in range(max_level + 1)}


class LPIPSScorer:
    """Lazy lpips.LPIPS(net='vgg'); inputs are [1,3,H,W] tensors in [-1,1]."""

    def __init__(self, device="cpu", net="vgg"):
        self.device, self.net, self._m = device, net, None

    def __call__(self, t1, t2) -> float:
        import torch

        if self._m is None:
            import lpips

            self._m = lpips.LPIPS(net=self.net).to(self.device)
        with torch.no_grad():
            return float(self._m(t1.to(self.device), t2.to(self.device)).item())


def v1_quality_metrics(clean_t, prot_t, lpips=None) -> dict:
    a, b = tensor_to_u8_trunc_v1(clean_t), tensor_to_u8_trunc_v1(prot_t)
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    out = {"MSE": mse, "PSNR": psnr_u8(a, b), "SSIM": ssim_u8(a, b),
           "LPIPS": float(lpips(clean_t, prot_t)) if lpips is not None else float("nan"),
           "Linf": float((prot_t - clean_t).abs().max().item())}
    return out


def _u8_to_tensor(a: np.ndarray):
    import torch

    return torch.from_numpy(a.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0


def png_quality_metrics(clean_path, prot_path, lpips=None) -> dict:
    a, b = load_u8(clean_path), load_u8(prot_path)
    m = max_abs_diff_u8(a, b)
    d = np.abs(a.astype(np.int64) - b.astype(np.int64))
    frac = float((d == m).mean()) if m > 0 else 0.0
    return {"psnr": psnr_u8(a, b), "ssim": ssim_u8(a, b),
            "lpips": float(lpips(_u8_to_tensor(a), _u8_to_tensor(b))) if lpips is not None else float("nan"),
            "maxabs_u8": m, "frac_at_bound": frac}
