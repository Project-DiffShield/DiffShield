"""PhotoGuard encoder attack (Salman et al. 2023, "Raising the Cost of Malicious AI-Powered Image Editing").

THIS IS A REIMPLEMENTATION (label kept) of the public MadryLab/photoguard encoder attack, now checked against
the official code. source_verified: https://github.com/MadryLab/photoguard @ 686bea75c786cb46c88fc396a0cd0ee3d7d28c2e
(2023-02-27), notebooks/demo_simple_attack_img2img.ipynb cell 7 (pgd, img2img: loss = latent_dist.mean.norm(), i.e.
zero-latent target) and notebooks/demo_simple_attack_inpainting.ipynb cell 7 (pgd with target =
vae.encode(target_image).latent_dist.mean, loss = (mean - target).norm()).

Official hyper-parameters (on [-1,1] scale) vs ours here:
  * PGD, L_inf, sign gradient DESCENT, uniform random start in [-eps, eps] (no projection of the start), project to
    the eps box then clamp to [-1,1] each step: matches.
  * step size decays linearly step -> step/100: matches the official pgd().
  * iters: official demos use 1000 (we now default to 1000; was 200).
  * step: official 0.02 (img2img demo) / 0.01 (inpainting demo) = 2.55 / 1.275 in /255 standard units; we default to
    the img2img value 2.55 (was 1.0).
  * eps: official 0.06 on [-1,1] = 7.65/255 standard. NOT adopted: eps is OUR budget (4/255).
  * loss: L2 norm (not squared) of the latent MEAN (latent_dist.mean, unscaled) minus target; ours is the same norm
    (the sign of the gradient is identical to the squared version).
  * target: img2img demo = zero latent; inpainting demo = latent of a target image (the repo's assets/ image, not
    vendored; pass ``target_image``). Default is now ``zero_latent`` (was gray); ``gray`` stays as an extra option.
  * Official fp16 autocast; ours runs the VAE in fp16 on cuda with a loss scale (does not change the sign).

Deviations / out of scope (documented, not hidden):
  * The "diffusion" attack variant of PhotoGuard (backprop through the whole sampling chain, ~10x cost) is OUT
    OF SCOPE (48h plan); only the encoder attack is run. Report it as "PhotoGuard (encoder attack)".
  * Final image is quantized to the uint8 grid inside the eps ball (published PNG), see noise_control.quantize_to_budget.
  * Inpainting-mask-aware variant of the repo (attack only the unmasked region) is not implemented: the
    protection is applied to the whole image, as for every method in this study.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from .noise_control import from_u8, int_budget, linf_u8, psnr_tensors, quantize_to_budget, to_model_scale, to_u8

SOURCE_VERIFIED = ("https://github.com/MadryLab/photoguard@686bea75c786cb46c88fc396a0cd0ee3d7d28c2e "
                   "notebooks/demo_simple_attack_{img2img,inpainting}.ipynb cell 7")

EncodeFn = Callable[[torch.Tensor], torch.Tensor]  # x[B,3,H,W] in [-1,1] -> latent mean (differentiable, float)


def make_generator(seed: int, device) -> torch.Generator:
    dev = torch.device(device)
    g = torch.Generator(device=dev.type if dev.type != "cuda" else "cuda")
    g.manual_seed(int(seed) % (2 ** 63))
    return g


def project(x_adv: torch.Tensor, x_ref: torch.Tensor, eps_m: float) -> torch.Tensor:
    """Project onto the L_inf ball around x_ref (model scale) and onto [-1, 1]."""
    x_adv = torch.minimum(torch.maximum(x_adv, x_ref - eps_m), x_ref + eps_m)
    return x_adv.clamp(-1.0, 1.0)


def step_size_at(step: float, i: int, iters: int, decay: bool = True) -> float:
    """Linear decay step -> step/100 (official pgd(), verified)."""
    if not decay:
        return step
    return step - (step - step / 100.0) / max(iters, 1) * i


class EncoderAttackProtector:
    """PhotoGuard encoder attack as a Protector (``protect(x, seed) -> (x_pub, info)``)."""

    name = "photoguard"

    def __init__(self, encode_fn: EncodeFn, eps_255: float = 4.0, step_255: float = 2.55, iters: int = 1000,
                 target: str = "zero_latent", target_image: Optional[torch.Tensor] = None, step_decay: bool = True,
                 random_start: bool = True, loss_scale: float = 1.0):
        if target not in ("gray", "zero_latent", "image"):
            raise ValueError("target must be gray | zero_latent | image")
        if target == "image" and target_image is None:
            raise ValueError("target='image' needs target_image [1,3,H,W] in [-1,1]")
        self.encode_fn = encode_fn
        self.eps_255 = float(eps_255)
        self.step_255 = float(step_255)
        self.iters = int(iters)
        self.target = target
        self.target_image = target_image
        self.step_decay = bool(step_decay)
        self.random_start = bool(random_start)
        self.loss_scale = float(loss_scale)

    # -- target latent ---------------------------------------------------------------------------
    def target_latent(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.target == "zero_latent":
                return torch.zeros_like(self.encode_fn(x[:1]).float())
            if self.target == "gray":
                return self.encode_fn(torch.zeros_like(x)).float()
            ti = self.target_image.to(x.device, x.dtype)
            if ti.shape[-2:] != x.shape[-2:]:
                ti = torch.nn.functional.interpolate(ti, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(-1, 1)
            return self.encode_fn(ti.expand(x.shape[0], -1, -1, -1)).float()

    # -- attack ----------------------------------------------------------------------------------
    def protect(self, x: torch.Tensor, seed: int, **_kw) -> Tuple[torch.Tensor, Dict]:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,H,W], got {tuple(x.shape)}")
        x_ref = from_u8(to_u8(x)).to(x.dtype)        # grid-aligned clean image the budget is measured against
        eps_m = to_model_scale(int_budget(self.eps_255))
        step_m = to_model_scale(self.step_255)
        g = make_generator(seed, x.device)
        z_t = self.target_latent(x_ref)

        x_adv = x_ref.clone()
        if self.random_start:
            noise = (torch.rand(x_ref.shape, generator=g, device=x_ref.device, dtype=x_ref.dtype) * 2 - 1) * eps_m
            x_adv = project(x_ref + noise, x_ref, eps_m)
        loss_first = loss_last = float("nan")
        for i in range(self.iters):
            x_adv = x_adv.detach().requires_grad_(True)
            z = self.encode_fn(x_adv).float()
            loss = (z - z_t).norm()                   # official: (mean - target).norm()
            grad, = torch.autograd.grad(loss * self.loss_scale, x_adv)
            if i == 0:
                loss_first = float(loss.detach())
            loss_last = float(loss.detach())
            s = step_size_at(step_m, i, self.iters, self.step_decay)
            x_adv = project(x_adv.detach() - s * grad.sign(), x_ref, eps_m)
        x_pub = quantize_to_budget(x_ref, x_adv.detach(), self.eps_255).to(x.dtype)
        with torch.no_grad():
            loss_pub = float((self.encode_fn(x_pub).float() - z_t).norm())
        info = {"method": self.name, "attack": "encoder", "target": self.target, "eps_255": self.eps_255,
                "step_255": self.step_255, "iters": self.iters, "seed": int(seed),
                "implementation": "reimplementation", "source_verified": SOURCE_VERIFIED,
                "loss_first": loss_first, "loss_last_iter": loss_last, "loss_published": loss_pub,
                "linf_u8": linf_u8(x_ref, x_pub), "psnr": psnr_tensors(x_ref, x_pub)}
        return x_pub, info
