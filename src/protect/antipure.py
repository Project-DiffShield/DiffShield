"""Anti-purification terms (Phase 5, AntiPure idea; OFF by default, ``antipure.enabled: false``).

Both terms are MAXIMISED and use the one-step purifier's reconstruction, so they are only evaluated
on EoT draws where the one-step purifier was drawn (the draw's UNet pass is reused through ``aux``;
the monotonic term costs one extra one-step pass per such draw).

  l_hf    patch-wise (8x8, non-overlapping) DCT; mean squared high-frequency coefficient energy
          (u + v >= hf_cut) of the one-step reconstruction of the published image, divided by the same
          quantity on the clean image (a detached constant), i.e. a dimensionless ratio ~1 at start.
          Rationale: push the purified output to carry high-frequency artifacts.
  l_mono  reconstruction error (MSE vs the clean image) at two timesteps t_lo < t_hi sharing the same
          noise: 0.5*(e_lo + e_hi) - relu(e_lo - e_hi), i.e. large error that does not shrink as t grows.
          This is OUR reading of 'keep the reconstruction error growing with t' (the AntiPure paper's
          exact formulation was not checked; unverified). The handoff's starting weights are 0.5 / 0.5.
"""
from __future__ import annotations

from typing import Dict, Optional

import math

import torch
import torch.nn.functional as F

from .models import g_randint
from .purifiers import SDOneStepPurifier


def dct_matrix(n: int = 8, device="cpu", dtype=torch.float32) -> torch.Tensor:
    k = torch.arange(n, dtype=torch.float64).view(-1, 1)
    i = torch.arange(n, dtype=torch.float64).view(1, -1)
    d = torch.cos(math.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2.0 / n)
    d[0] = d[0] / math.sqrt(2.0)
    return d.to(device=device, dtype=dtype)


def hf_energy(x: torch.Tensor, patch: int = 8, hf_cut: int = 8) -> torch.Tensor:
    """Mean squared high-frequency DCT energy of x in [-1,1] (luma, per 8x8 patch)."""
    g = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3] + 1.0) / 2.0
    h, w = g.shape[-2] // patch * patch, g.shape[-1] // patch * patch
    g = g[..., :h, :w]
    blocks = g.unfold(2, patch, patch).unfold(3, patch, patch)  # B,1,h/p,w/p,p,p
    D = dct_matrix(patch, g.device, g.dtype)
    coef = D @ blocks @ D.t()
    u = torch.arange(patch, device=g.device).view(-1, 1)
    v = torch.arange(patch, device=g.device).view(1, -1)
    mask = ((u + v) >= hf_cut).to(g.dtype)
    return ((coef ** 2) * mask).sum(dim=(-1, -2)).mean() / mask.sum()


class AntiPure:
    def __init__(self, onestep: SDOneStepPurifier, w_hf: float = 0.5, w_mono: float = 0.5,
                 patch: int = 8, hf_cut: int = 8):
        self.os = onestep
        self.w_hf, self.w_mono = float(w_hf), float(w_mono)
        self.patch, self.hf_cut = patch, hf_cut

    def prepare(self, x: torch.Tensor) -> dict:
        with torch.no_grad():
            return {"x": x.detach(), "hf_ref": hf_energy(x, self.patch, self.hf_cut).clamp_min(1e-8)}

    def terms(self, aux: Optional[dict], x_pub: torch.Tensor, state: dict, gen: torch.Generator) -> Dict[str, torch.Tensor]:
        """Returns {'hf': ..., 'mono': ...} (weights NOT applied); empty dict if the draw was not
        a one-step purification."""
        if aux is None:
            return {}
        out = {}
        rec1, t1 = aux["rec"], aux["t"]
        if self.w_hf:
            out["hf"] = hf_energy(rec1, self.patch, self.hf_cut) / state["hf_ref"]
        if self.w_mono:
            mid = (self.os.t_min + self.os.t_max) // 2
            if int(t1.max()) <= mid:
                t2 = g_randint(mid + 1, self.os.t_max, gen, n=t1.numel()).to(t1.device)
                rec_lo = rec1
                rec_hi, _ = self.os.reconstruct(x_pub, t2, aux["noise"])
            else:
                t2 = g_randint(self.os.t_min, mid, gen, n=t1.numel()).to(t1.device)
                rec_lo, _ = self.os.reconstruct(x_pub, t2, aux["noise"])
                rec_hi = rec1
            e_lo = F.mse_loss(rec_lo, state["x"])
            e_hi = F.mse_loss(rec_hi, state["x"])
            out["mono"] = 0.5 * (e_lo + e_hi) - F.relu(e_lo - e_hi)
        return out
