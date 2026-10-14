"""Encoder ensemble for L_vis (Phase 6): average the unscaled latent-mean MSE over several VAEs
(SD1.5 VAE from ``SDModels`` + SDXL VAE). Same ``prepare`` / ``__call__`` interface as
``losses.SingleVis`` so the PGD loop does not care.

Note: the SDXL VAE is a different (larger-latent-variance) network, so its MSE has a different scale
from the SD1.5 one; ``weights`` lets the caller rebalance (default: equal, plain average as in the
plan). Both are unscaled posterior means.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import SDModels


class EnsembleVis:
    def __init__(self, models: SDModels, extra_vaes: Dict[str, nn.Module],
                 weights: Optional[Dict[str, float]] = None):
        self.m = models
        self.extra = dict(extra_vaes)
        names = ["sd15"] + list(self.extra)
        w = {n: float((weights or {}).get(n, 1.0)) for n in names}
        tot = sum(w.values())
        self.w = {n: v / tot for n, v in w.items()}

    def _enc(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name == "sd15":
            return self.m.encode_mean(x, scaled=False)
        return self.extra[name].encode(x.float()).latent_dist.mean

    def prepare(self, x: torch.Tensor):
        with torch.no_grad():
            return {n: self._enc(n, x) for n in self.w}

    def __call__(self, xp: torch.Tensor, state) -> torch.Tensor:
        return sum(self.w[n] * F.mse_loss(self._enc(n, xp), state[n]) for n in self.w)
