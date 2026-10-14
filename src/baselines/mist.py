"""Mist (Liang & Wu 2023, "Mist: Towards Improved Adversarial Examples for Diffusion Models").

THIS IS A REIMPLEMENTATION from the published method description and from memory of the public code
(psyker-team/mist); the official repo was NOT available in this workspace (no network access from the coding
agent) and has NOT been run. Report it as "Mist (reimplementation)", never as the official tool.

Method as implemented (fused loss, gradient steps on the image under an L_inf budget):
    maximize   w_sem * L_sem(x')  -  w_tex * L_tex(x')
  * L_sem  (semantic loss)  = E_{t,eps} || eps_theta( sqrt(abar_t) * s*E(x') + sqrt(1-abar_t) * eps, t, c ) - eps ||^2
            i.e. the SD UNet denoising error on the protected image (AdvDM-style), MAXIMIZED, with s the VAE
            scaling factor and c the text conditioning (default: empty prompt).
  * L_tex  (textural loss)  = || E(x') - E(T) ||^2 over VAE-encoder latent means, MINIMIZED, where T is a
            high-contrast target pattern image. Pulls the encoder output toward a texture so edits inherit
            visible patterns.
  PGD: random start, sign step, projection onto the eps ball and [-1,1]; quantized to the uint8 grid at the end.

UNVERIFIED / deviations (must be stated wherever results are reported):
  * Loss weights (w_sem=1, w_tex=1), iteration count (100), step size (1/255), timestep range ([1,1000)),
    number of noise draws per step, and the sign conventions are my reading of the paper, NOT checked against
    the official code (whose default weighting / "rate" and mode flag I could not verify).
  * Target pattern: the official repo ships its own target image (a blue/white textured pattern, "MIST.png").
    That asset is not vendored here; ``make_texture_target`` builds a deterministic procedural high-contrast
    pattern instead, or pass ``target_image`` / config ``baselines.mist.target_image`` (path) to use the
    official asset when you have it. This changes the visual signature of the perturbed edits.
  * Official Mist v2 uses a larger budget by default (UNVERIFIED, ~16/255); here the budget is OURS (4/255
    standard scale) for a same-budget comparison.
  * No Mist-specific preprocessing (e.g. fixed 512 resize, their pipeline wrappers) is reproduced.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch

from .noise_control import from_u8, int_budget, linf_u8, psnr_tensors, quantize_to_budget, to_model_scale, to_u8
from .photoguard import make_generator, project, step_size_at

EncodeFn = Callable[[torch.Tensor], torch.Tensor]
# unet_fn(z_t [B,4,h,w], t [B] long, cond) -> eps_pred [B,4,h,w]
UNetFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_texture_target(h: int, w: int, seed: int = 0, device="cpu") -> torch.Tensor:
    """Deterministic procedural high-contrast pattern in [-1,1], shape [1,3,h,w].

    Stand-in for Mist's official target image: diagonal stripes of two frequencies modulated by a checker, in
    saturated blue/white/orange tones. Content is irrelevant for the attack; contrast and structure are what matter.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    ph = torch.rand(3, generator=g) * 2 * math.pi
    ys = torch.linspace(0, 1, h).view(h, 1)
    xs = torch.linspace(0, 1, w).view(1, w)
    s1 = torch.sin(2 * math.pi * (14 * xs + 9 * ys))
    s2 = torch.sin(2 * math.pi * (23 * xs - 17 * ys))
    checker = torch.sign(torch.sin(2 * math.pi * 8 * xs) * torch.sin(2 * math.pi * 8 * ys))
    chans = [torch.tanh(2.5 * (s1 * 0.6 + s2 * 0.4 * checker) + ph[c].sin()) for c in range(3)]
    img = torch.stack(chans, 0).unsqueeze(0)
    return img.clamp(-1, 1).to(device)


class MistProtector:
    """Mist (reimplementation) as a Protector: ``protect(x, seed) -> (x_pub, info)``."""

    name = "mist"

    def __init__(self, encode_fn: EncodeFn, unet_fn: UNetFn, alphas_cumprod: torch.Tensor, cond: torch.Tensor,
                 eps_255: float = 4.0, step_255: float = 1.0, iters: int = 100, w_sem: float = 1.0, w_tex: float = 1.0,
                 t_min: int = 1, t_max: Optional[int] = None, n_noise: int = 1, scaling_factor: float = 0.18215,
                 target_image: Optional[torch.Tensor] = None, step_decay: bool = False, random_start: bool = True,
                 loss_scale: float = 1.0):
        self.encode_fn = encode_fn
        self.unet_fn = unet_fn
        self.ab = alphas_cumprod.detach().float()
        self.cond = cond
        self.eps_255, self.step_255, self.iters = float(eps_255), float(step_255), int(iters)
        self.w_sem, self.w_tex = float(w_sem), float(w_tex)
        self.t_min = int(t_min)
        self.t_max = int(t_max) if t_max is not None else int(self.ab.numel())
        self.n_noise = int(n_noise)
        self.sf = float(scaling_factor)
        self.target_image = target_image
        self.step_decay, self.random_start, self.loss_scale = bool(step_decay), bool(random_start), float(loss_scale)

    # -- losses ----------------------------------------------------------------------------------
    def semantic_loss(self, z_mean: torch.Tensor, g: torch.Generator) -> torch.Tensor:
        """UNet denoising error at random t (to be MAXIMIZED); averaged over ``n_noise`` draws."""
        total = z_mean.new_zeros(())
        z = z_mean * self.sf
        ab = self.ab.to(z.device)
        for _ in range(self.n_noise):
            t = torch.randint(self.t_min, self.t_max, (z.shape[0],), generator=g, device=z.device)
            n = torch.randn(z.shape, generator=g, device=z.device, dtype=z.dtype)
            a = ab[t].view(-1, 1, 1, 1)
            zt = a.sqrt() * z + (1 - a).sqrt() * n
            pred = self.unet_fn(zt, t, self.cond).float()
            total = total + (pred - n).pow(2).mean()
        return total / self.n_noise

    def textural_loss(self, z_mean: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        return (z_mean - z_target).pow(2).mean()

    def _target_latent(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        ti = self.target_image
        if ti is None:
            ti = make_texture_target(h, w, 0, x.device)
        ti = ti.to(x.device, x.dtype)
        if ti.shape[-2:] != (h, w):
            ti = torch.nn.functional.interpolate(ti, size=(h, w), mode="bicubic", align_corners=False).clamp(-1, 1)
        with torch.no_grad():
            return self.encode_fn(ti.expand(x.shape[0], -1, -1, -1)).float()

    # -- attack ----------------------------------------------------------------------------------
    def protect(self, x: torch.Tensor, seed: int, **_kw) -> Tuple[torch.Tensor, Dict]:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"expected [B,3,H,W], got {tuple(x.shape)}")
        x_ref = from_u8(to_u8(x)).to(x.dtype)
        eps_m = to_model_scale(int_budget(self.eps_255))
        step_m = to_model_scale(self.step_255)
        g = make_generator(seed, x.device)
        z_tgt = self._target_latent(x_ref)

        x_adv = x_ref.clone()
        if self.random_start:
            noise = (torch.rand(x_ref.shape, generator=g, device=x_ref.device, dtype=x_ref.dtype) * 2 - 1) * eps_m
            x_adv = project(x_ref + noise, x_ref, eps_m)
        first = last = None
        for i in range(self.iters):
            x_adv = x_adv.detach().requires_grad_(True)
            z = self.encode_fn(x_adv).float()
            l_sem = self.semantic_loss(z, g)
            l_tex = self.textural_loss(z, z_tgt)
            obj = self.w_sem * l_sem - self.w_tex * l_tex      # ASCENT objective
            grad, = torch.autograd.grad(obj * self.loss_scale, x_adv)
            rec = (float(obj.detach()), float(l_sem.detach()), float(l_tex.detach()))
            first = rec if first is None else first
            last = rec
            s = step_size_at(step_m, i, self.iters, self.step_decay)
            x_adv = project(x_adv.detach() + s * grad.sign(), x_ref, eps_m)
        x_pub = quantize_to_budget(x_ref, x_adv.detach(), self.eps_255).to(x.dtype)
        info = {"method": self.name, "implementation": "reimplementation (unverified vs official)",
                "eps_255": self.eps_255, "step_255": self.step_255, "iters": self.iters, "seed": int(seed),
                "w_sem": self.w_sem, "w_tex": self.w_tex,
                "obj_first": first[0] if first else float("nan"), "obj_last": last[0] if last else float("nan"),
                "l_sem_first": first[1] if first else float("nan"), "l_sem_last": last[1] if last else float("nan"),
                "l_tex_first": first[2] if first else float("nan"), "l_tex_last": last[2] if last else float("nan"),
                "linf_u8": linf_u8(x_ref, x_pub), "psnr": psnr_tensors(x_ref, x_pub)}
        return x_pub, info
