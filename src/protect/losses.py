"""Loss terms for the v2 protection objective (all return fp32 scalars; PGD ascends on

    total = alpha*L_vis + eta*L_den - beta*L_sem - gamma*L_str - lambda_id*cos_id [+ anti-purify terms]

so L_vis, L_den are MAXIMISED, L_sem (1-cos) and L_str (1-SSIM) are MINIMISED, as in v1 / the handoff.

  L_vis  MSE between the VAE latent mean (UNSCALED, keeps v1's alpha scale) of the purified
         protected image and of the clean image (latent of the clean image computed once).
  L_den  MSE between true noise and the UNet's noise prediction on the noised latent of the purified
         image, empty prompt, t ~ U[t_min, t_max]. If the drawn purifier is the one-step purifier its
         own UNet pass is reused (aux), otherwise one extra UNet pass is made. Latents are SCALED
         (vae scaling factor from config) for the UNet (R1).
  L_sem  1 - cos(CLIP image embedding, text embedding of 'a potted plant'); optional (beta may be 0).
  L_str  1 - windowed SSIM(x, x_published), 11x11 Gaussian window sigma 1.5, fp32. It is computed on
         the PUBLISHED image x + delta, not on the purified one.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from .models import SDModels, g_randint, g_randn_like

TERMS = ("vis", "den", "sem", "str", "id", "hf", "mono")


def l_vis(models: SDModels, xp: torch.Tensor, z_clean: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(models.encode_mean(xp, scaled=False), z_clean)


def l_den(models: SDModels, xp: Optional[torch.Tensor], gen: torch.Generator, aux: Optional[dict] = None,
          t_range: Sequence[int] = (50, 300)) -> torch.Tensor:
    """Denoiser error. ``aux`` (from SDOneStepPurifier.forward_aux) is reused when given."""
    if aux is not None:
        return F.mse_loss(aux["eps"], aux["noise"])
    z = models.encode_mean(xp, scaled=True)
    t = g_randint(t_range[0], t_range[1], gen, n=z.shape[0]).to(z.device)
    n = g_randn_like(z, gen)
    ab = models.alpha_bar(t).to(z.dtype)
    zt = ab.sqrt() * z + (1.0 - ab).sqrt() * n
    return F.mse_loss(models.unet_eps(zt, t), n)


def l_sem(clip_sem, xp: torch.Tensor) -> torch.Tensor:
    return clip_sem.sem_loss(xp)


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(1)
    return (g @ g.t()).expand(channels, 1, window_size, window_size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean windowed SSIM of two images in [-1,1] (same formula as v1's structure term; fp32)."""
    x01 = ((x.float() + 1.0) / 2.0)
    y01 = ((y.float() + 1.0) / 2.0)
    C = x01.shape[1]
    w = _gaussian_window(window, sigma, C, x01.device, x01.dtype)
    pad = window // 2
    mu_x = F.conv2d(x01, w, padding=pad, groups=C)
    mu_y = F.conv2d(y01, w, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    s_x = F.conv2d(x01 * x01, w, padding=pad, groups=C) - mu_x2
    s_y = F.conv2d(y01 * y01, w, padding=pad, groups=C) - mu_y2
    s_xy = F.conv2d(x01 * y01, w, padding=pad, groups=C) - mu_xy
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    smap = ((2 * mu_xy + C1) * (2 * s_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (s_x + s_y + C2))
    return smap.mean()


def l_str(x: torch.Tensor, x_pub: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    return 1.0 - ssim(x, x_pub, window, sigma)


def psnr_uint8(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR of two images in [-1,1], both quantised to uint8 first (what the saved PNGs contain)."""
    qa = ((a.float() + 1.0) * 127.5).round().clamp(0, 255)
    qb = ((b.float() + 1.0) * 127.5).round().clamp(0, 255)
    mse = float(((qa - qb) ** 2).mean())
    return float("inf") if mse == 0 else float(10.0 * torch.log10(torch.tensor(255.0 ** 2 / mse)))


class TermScales:
    """Per-term normalisation (V2_PLAN Phase 6: 'normalize all loss terms to comparable scales').

    scale[term] divides the term before the weight is applied, so with scales measured as the mean
    |term| at the random start the weights are comparable. Default (no scales) leaves the terms
    unchanged, i.e. v1's weight semantics. Scales are stored in config ``loss_norm``.
    """

    def __init__(self, scales: Optional[Dict[str, float]] = None):
        self.scales = {k: float(v) for k, v in (scales or {}).items() if v}

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        s = self.scales.get(name)
        return value / s if s else value


def scales_from_infos(infos: Sequence[dict]) -> Dict[str, float]:
    """Calibrate term scales from the ``info['terms_init']`` entries of calibration runs made with
    unit scales: mean absolute value per term (terms whose mean is 0 are skipped)."""
    acc: Dict[str, list] = {}
    for inf in infos:
        for k, v in (inf.get("terms_init") or {}).items():
            acc.setdefault(k, []).append(abs(float(v)))
    return {k: sum(v) / len(v) for k, v in acc.items() if sum(v) > 0}


class SingleVis:
    """L_vis on one VAE (the SD1.5 VAE of ``models``)."""

    def __init__(self, models: SDModels):
        self.m = models

    def prepare(self, x: torch.Tensor):
        with torch.no_grad():
            return self.m.encode_mean(x, scaled=False)

    def __call__(self, xp: torch.Tensor, state) -> torch.Tensor:
        return l_vis(self.m, xp, state)
