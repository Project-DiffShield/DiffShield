"""Noise controls (E1) and shared budget helpers for the baselines.

Protector interface (frozen, 48h plan section 4):
    protect(x: Tensor[1,3,H,W] in [-1,1], seed: int) -> (Tensor, dict)

Budget convention: ``eps_255`` is on the STANDARD 0-255 scale. Published images are uint8 PNGs, so every
protector here returns a tensor that lies exactly on the uint8 grid (``u8 / 127.5 - 1``) and whose uint8
difference to ``round((x + 1) * 127.5)`` is at most ``floor(eps_255)`` gray levels (R12 in V2_PLAN.md).

Controls:
  * ``uniform_linf``   - integer-grid uniform noise, L_inf <= eps (exact in uint8 terms).
  * ``sign_linf``      - +/- eps noise (extra, highest-energy L_inf control).
  * ``gaussian_matched_psnr`` - Gaussian noise whose sigma is bisected per image so the PSNR of the
    uint8 result vs the uint8 clean image hits a target (typically the image's own v1 / v2 protected PSNR).
    Its L_inf is NOT bounded by eps (Gaussian tails); that is the point of the control (same PSNR, any L_inf).
CPU only.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, Optional, Tuple

import torch

# --------------------------------------------------------------------------------------------- helpers


def to_model_scale(v_255: float) -> float:
    """Standard-scale gray levels -> [-1, 1] scale (x * 2 / 255)."""
    return float(v_255) * 2.0 / 255.0


def to_u8(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] float -> float tensor holding integer values 0..255 (published rounding)."""
    return ((x.detach().float() + 1.0) * 127.5).round().clamp(0, 255)


def from_u8(u: torch.Tensor) -> torch.Tensor:
    return u / 127.5 - 1.0


def int_budget(eps_255: float) -> int:
    """Largest integer gray-level change allowed by a (possibly fractional) standard-scale budget."""
    return int(math.floor(float(eps_255) + 1e-9))


def quantize_to_budget(x_clean: torch.Tensor, x_adv: torch.Tensor, eps_255: float) -> torch.Tensor:
    """Round ``x_adv`` to the uint8 grid and clip to |u8(x_adv) - u8(x_clean)| <= floor(eps_255)."""
    u0 = to_u8(x_clean)
    ua = to_u8(x_adv)
    e = float(int_budget(eps_255))
    ua = torch.minimum(torch.maximum(ua, u0 - e), u0 + e).clamp(0, 255)
    return from_u8(ua)


def linf_u8(x_clean: torch.Tensor, x_pub: torch.Tensor) -> int:
    return int((to_u8(x_pub) - to_u8(x_clean)).abs().max().item())


def psnr_u8(a_u8: torch.Tensor, b_u8: torch.Tensor) -> float:
    """PSNR (dB) between two uint8-valued tensors (peak 255). inf if identical."""
    mse = float(((a_u8.double() - b_u8.double()) ** 2).mean().item())
    return float("inf") if mse <= 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def psnr_tensors(x_clean: torch.Tensor, x_pub: torch.Tensor) -> float:
    return psnr_u8(to_u8(x_clean), to_u8(x_pub))


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) % (2 ** 63))
    return g


def _check(x: torch.Tensor) -> None:
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"expected [B,3,H,W] in [-1,1], got {tuple(x.shape)}")


# --------------------------------------------------------------------------------------------- functions


def uniform_linf(x: torch.Tensor, eps_255: float, seed: int) -> torch.Tensor:
    """Uniform noise U(-eps, eps) in gray levels, rounded to the uint8 grid (so |delta| <= floor(eps))."""
    _check(x)
    e = float(int_budget(eps_255))
    u0 = to_u8(x).cpu()
    n = (torch.rand(u0.shape, generator=_gen(seed)) * 2.0 - 1.0) * e
    ua = (u0 + n).round()
    ua = torch.minimum(torch.maximum(ua, u0 - e), u0 + e).clamp(0, 255)
    return from_u8(ua).to(x.device)


def sign_linf(x: torch.Tensor, eps_255: float, seed: int) -> torch.Tensor:
    """+/- eps noise with random signs (extra control: maximum L2 energy at the L_inf budget)."""
    _check(x)
    e = float(int_budget(eps_255))
    u0 = to_u8(x).cpu()
    s = torch.randint(0, 2, u0.shape, generator=_gen(seed)).float() * 2.0 - 1.0
    ua = (u0 + s * e).clamp(0, 255)
    return from_u8(ua).to(x.device)


def _noisy_u8(u0: torch.Tensor, n: torch.Tensor, sigma: float) -> torch.Tensor:
    return (u0 + sigma * n).round().clamp(0, 255)


def gaussian_matched_psnr(x: torch.Tensor, target_psnr: float, seed: int, tol: float = 0.02,
                          max_iter: int = 60) -> Tuple[torch.Tensor, Dict]:
    """Gaussian noise (in gray levels) with sigma chosen per image so that PSNR(u8 clean, u8 noisy) ~= target.

    Bisection on sigma AFTER clipping to [0,255] and rounding to uint8 (clipping at saturated pixels lowers
    the effective noise power, so sigma > 255 * 10^(-psnr/20)). Returns (tensor on the uint8 grid, info).
    """
    _check(x)
    u0 = to_u8(x).cpu()
    n = torch.randn(u0.shape, generator=_gen(seed))
    target = float(target_psnr)

    def psnr_at(s: float) -> float:
        return psnr_u8(u0, _noisy_u8(u0, n, s))

    lo, hi = 0.0, max(1.0, 255.0 * 10 ** (-target / 20.0) * 3.0)
    while psnr_at(hi) > target and hi < 4096.0:  # grow until PSNR drops below target
        hi *= 2.0
    best_s, best_err = hi, abs(psnr_at(hi) - target)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = psnr_at(mid)
        err = abs(p - target) if math.isfinite(p) else float("inf")
        if err < best_err:
            best_s, best_err = mid, err
        if err <= tol:
            break
        if p > target:  # too little noise
            lo = mid
        else:
            hi = mid
    ua = _noisy_u8(u0, n, best_s)
    achieved = psnr_u8(u0, ua)
    info = {"sigma_gray": float(best_s), "target_psnr": target, "achieved_psnr": float(achieved),
            "psnr_abs_err": float(abs(achieved - target)), "linf_u8": int((ua - u0).abs().max().item())}
    return from_u8(ua).to(x.device), info


# --------------------------------------------------------------------------------------------- protectors


class UniformNoiseProtector:
    """E1 control: uniform random noise at the SAME L_inf budget as the real methods."""

    name = "noise_uniform"

    def __init__(self, eps_255: float = 4.0, variant: str = "uniform"):
        if variant not in ("uniform", "sign"):
            raise ValueError("variant must be 'uniform' or 'sign'")
        self.eps_255 = float(eps_255)
        self.variant = variant

    def protect(self, x: torch.Tensor, seed: int, **_kw) -> Tuple[torch.Tensor, Dict]:
        fn = uniform_linf if self.variant == "uniform" else sign_linf
        # per-sample seeds so batch order cannot change a sample's noise
        outs = [fn(x[i:i + 1], self.eps_255, int(seed) + i) for i in range(x.shape[0])]
        y = torch.cat(outs, 0)
        info = {"method": self.name, "variant": self.variant, "eps_255": self.eps_255, "seed": int(seed),
                "linf_u8": linf_u8(x, y), "psnr": psnr_tensors(x, y)}
        return y, info


def _stem(s: str) -> str:
    return os.path.splitext(os.path.basename(str(s)))[0]


def load_psnr_map(csv_path: str, key_col: str = "Original_Filename", val_col: str = "PSNR_dB") -> Dict[str, float]:
    """image stem -> target PSNR. Default columns match v1's ``stage1_per_image_metrics.csv``."""
    out: Dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[_stem(row[key_col])] = float(row[val_col])
    return out


class GaussianPSNRProtector:
    """E1 control: Gaussian noise matched in PSNR to a target.

    Target resolution order: ``protect(..., target_psnr=)`` > ``psnr_map[image_id]`` > constant ``target_psnr``.
    The constant default (41.911 dB) is the v1 mean over the 70 eval images (V2_PLAN_48H section 0); for the paired
    control, pass ``image_id`` (via ``protect(..., image_id=...)`` or ``protector.image_id = ...``) and a
    ``psnr_map`` built from the protected-image PSNRs of the method being controlled.
    """

    name = "noise_gauss_psnr"

    def __init__(self, target_psnr: float = 41.911, psnr_map: Optional[Dict[str, float]] = None):
        self.target_psnr = float(target_psnr)
        self.psnr_map = {_stem(k): float(v) for k, v in (psnr_map or {}).items()}
        self.image_id: Optional[str] = None

    def resolve_target(self, image_id: Optional[str], target_psnr: Optional[float]) -> Tuple[float, str]:
        if target_psnr is not None:
            return float(target_psnr), "arg"
        iid = image_id if image_id is not None else self.image_id
        if iid is not None and _stem(iid) in self.psnr_map:
            return self.psnr_map[_stem(iid)], "map"
        return self.target_psnr, "constant"

    def protect(self, x: torch.Tensor, seed: int, image_id: Optional[str] = None,
                target_psnr: Optional[float] = None, **_kw) -> Tuple[torch.Tensor, Dict]:
        tgt, src = self.resolve_target(image_id, target_psnr)
        outs, infos = [], []
        for i in range(x.shape[0]):
            y, inf = gaussian_matched_psnr(x[i:i + 1], tgt, int(seed) + i)
            outs.append(y)
            infos.append(inf)
        y = torch.cat(outs, 0)
        info = {"method": self.name, "seed": int(seed), "target_source": src, "psnr": psnr_tensors(x, y)}
        info.update(infos[0] if len(infos) == 1 else {"per_sample": infos})
        return y, info
