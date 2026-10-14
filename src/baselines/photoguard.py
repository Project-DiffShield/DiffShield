"""PhotoGuard encoder attack (Salman et al. 2023, "Raising the Cost of Malicious AI-Powered Image Editing").

THIS IS A REIMPLEMENTATION from the published method description and from memory of the public repo
(MadryLab/photoguard); the repo was NOT available in this workspace (no network access from the coding agent).
Everything marked UNVERIFIED must be checked against the official code before it is cited as "PhotoGuard".

Method (encoder attack): PGD on the image so that the SD VAE *encoder* maps the protected image close to a
target latent, i.e. minimize ||E(x + delta) - z_T||_2 over ||delta||_inf <= eps. Editing pipelines
(img2img / inpainting) first encode the input, so a corrupted latent corrupts every downstream edit.

Implemented as in the paper / repo demo:
  * PGD with sign gradient *descent* on the latent distance, random start in the eps ball, projection onto
    the eps ball and onto [-1, 1] each step.
  * 200 iterations (task spec; the paper/demo values are UNVERIFIED), eps at OUR budget (default 4/255 standard
    scale) instead of the official default (UNVERIFIED, the official demo uses a larger L_inf budget of
    roughly 0.06-0.1 on [-1,1], i.e. 3-6/255 standard; check before quoting).
  * Step size decays linearly from ``step`` to ``step/100`` over the run (my recollection of the repo's pgd();
    UNVERIFIED).
  * Target latent: ``target="gray"`` encodes a constant mid-gray image (task spec); ``"zero_latent"`` uses
    z_T = 0, which is the "minimize ||E(x)||" variant I recall from the repo's simple demo (UNVERIFIED);
    ``"image"`` encodes a user-supplied target image (the paper uses a specific structured target image; the
    official asset is not vendored here).
  * Latent = posterior MEAN of the VAE encoder, unscaled (the scale factor does not change the sign gradient
    for this single-term loss).

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
    """Linear decay step -> step/100 (UNVERIFIED recollection of the official pgd())."""
    if not decay:
        return step
    return step - (step - step / 100.0) / max(iters, 1) * i


class EncoderAttackProtector:
    """PhotoGuard encoder attack as a Protector (``protect(x, seed) -> (x_pub, info)``)."""

    name = "photoguard"

    def __init__(self, encode_fn: EncodeFn, eps_255: float = 4.0, step_255: float = 1.0, iters: int = 200,
                 target: str = "gray", target_image: Optional[torch.Tensor] = None, step_decay: bool = True,
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
            loss = (z - z_t).pow(2).sum()             # sign step is invariant to the sqrt / scale of this loss
            grad, = torch.autograd.grad(loss * self.loss_scale, x_adv)
            if i == 0:
                loss_first = float(loss.detach())
            loss_last = float(loss.detach())
            s = step_size_at(step_m, i, self.iters, self.step_decay)
            x_adv = project(x_adv.detach() - s * grad.sign(), x_ref, eps_m)
        x_pub = quantize_to_budget(x_ref, x_adv.detach(), self.eps_255).to(x.dtype)
        with torch.no_grad():
            loss_pub = float((self.encode_fn(x_pub).float() - z_t).pow(2).sum())
        info = {"method": self.name, "attack": "encoder", "target": self.target, "eps_255": self.eps_255,
                "step_255": self.step_255, "iters": self.iters, "seed": int(seed),
                "loss_first": loss_first, "loss_last_iter": loss_last, "loss_published": loss_pub,
                "linf_u8": linf_u8(x_ref, x_pub), "psnr": psnr_tensors(x_ref, x_pub)}
        return x_pub, info
