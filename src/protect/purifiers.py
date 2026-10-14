"""Differentiable training purifiers (Half A). Interface (frozen, V2_PLAN_48H section 4):

    TrainPurifier(nn.Module).forward(x, gen) -> x        x in [-1, 1], same shape, differentiable

Eval-side (real PIL / diffusers) purifiers live in Half B (``src/eval``) and never share code with this
file.

JPEG note (R7). Primary implementation is ``kornia.enhance.jpeg_codec_differentiable`` (checked with
kornia 0.8.3: accepts a float quality tensor of shape (B,), any H/W, gradients non-zero). It only
approximates libjpeg's rounding, so evaluation must use real PIL JPEG. Fallback ``impl: pil_bpda``
runs the *real* PIL codec in the forward pass and uses the identity as backward
(straight-through / BPDA); it is also what ``DiffJPEG`` uses automatically if kornia cannot be imported.

``forward_aux(x, gen) -> (x_p, aux)`` additionally returns the tensors of the purifier's internal
UNet pass (``SDOneStepPurifier`` only) so L_den / the anti-purification terms can reuse it instead
of paying for a second UNet pass (V2_PLAN section 5, Phase 4).
"""
from __future__ import annotations

import io
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import SDModels, g_randint, g_randn_like, g_uniform


def ste_clamp(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """clamp in the forward pass, identity gradient (a hard clamp zeroes gradients at saturated
    pixels; R2)."""
    return x + (x.clamp(lo, hi) - x).detach()


class TrainPurifier(nn.Module):
    name = "base"

    def forward(self, x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward_aux(self, x: torch.Tensor, gen: torch.Generator) -> Tuple[torch.Tensor, Optional[dict]]:
        return self(x, gen), None


class Identity(TrainPurifier):
    name = "none"

    def forward(self, x, gen):
        return x


class DiffJPEG(TrainPurifier):
    name = "jpeg"

    def __init__(self, q_range: Sequence[float] = (50, 95), impl: str = "kornia_diff"):
        super().__init__()
        self.q_range = (float(q_range[0]), float(q_range[1]))
        self.impl = impl
        self._kornia = None
        if impl == "kornia_diff":
            try:
                from kornia.enhance import jpeg_codec_differentiable

                self._kornia = jpeg_codec_differentiable
            except Exception:  # kornia missing / API moved: documented BPDA fallback
                self.impl = "pil_bpda"

    def forward(self, x, gen):
        q = g_uniform(self.q_range[0], self.q_range[1], gen, shape=(x.shape[0],)).to(x.device)
        x01 = ((x.float() + 1.0) / 2.0)
        if self.impl == "kornia_diff":
            y01 = self._kornia(x01.clamp(0, 1), q)
        else:
            y01 = x01 + (self._pil_jpeg(x01.detach().clamp(0, 1), q) - x01).detach()
        return y01 * 2.0 - 1.0

    @staticmethod
    def _pil_jpeg(x01: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        import numpy as np
        from PIL import Image

        outs = []
        for i in range(x01.shape[0]):
            arr = (x01[i].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=int(round(float(q[i]))))
            buf.seek(0)
            dec = np.asarray(Image.open(buf).convert("RGB")).astype("float32") / 255.0
            outs.append(torch.from_numpy(dec).permute(2, 0, 1))
        return torch.stack(outs).to(x01.device)


class DownUp(TrainPurifier):
    """Bicubic/bilinear down-up at a random scale (all of H and W scaled by s, then back)."""

    name = "downup"

    def __init__(self, scale_range: Sequence[float] = (0.5, 0.9), modes: Sequence[str] = ("bilinear", "bicubic"),
                 antialias: bool = True):
        super().__init__()
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.modes = tuple(modes)
        self.antialias = bool(antialias)

    def forward(self, x, gen):
        s = float(g_uniform(self.scale_range[0], self.scale_range[1], gen))
        mode = self.modes[int(torch.randint(0, len(self.modes), (1,), generator=gen, device=gen.device))]
        h, w = x.shape[-2:]
        hs, ws = max(8, int(round(h * s))), max(8, int(round(w * s)))
        down = F.interpolate(x.float(), size=(hs, ws), mode=mode, align_corners=False, antialias=self.antialias)
        up = F.interpolate(down, size=(h, w), mode=mode, align_corners=False)
        return ste_clamp(up)


class VAERoundTrip(TrainPurifier):
    name = "vae_roundtrip"

    def __init__(self, models: SDModels):
        super().__init__()
        self.m = models

    def forward(self, x, gen):
        return self.m.decode(self.m.encode_mean(x, scaled=False), scaled=False)


class SDOneStepPurifier(TrainPurifier):
    """One-step SD1.5 'DiffPure' surrogate: encode -> noise to t -> one UNet eps-prediction ->
    z0 estimate -> decode. All weights frozen; gradients flow to the input pixels only.

    Fixes vs the handoff reference (R2): the scaling factor comes from the VAE config, alphas_cumprod
    lives on the device and sqrt / division run in fp32, every random draw uses the supplied
    generator, t is drawn from the *inclusive* range, the UNet condition is the real null-prompt
    embedding (cast to the UNet dtype in ``SDModels``), the encoder path is differentiable
    (``latent_dist.mean``), the output clamp is a straight-through clamp, UNet call is
    dtype-safe (fp16 UNet / fp32 VAE).
    """

    name = "sd_onestep"

    def __init__(self, models: SDModels, t_min: int = 50, t_max: int = 300):
        super().__init__()
        self.m = models
        self.t_min, self.t_max = int(t_min), int(t_max)

    def reconstruct(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """Deterministic given (t, noise). Returns (rec, aux)."""
        z0 = self.m.encode_mean(x, scaled=True)
        ab = self.m.alpha_bar(t).to(z0.dtype)
        zt = ab.sqrt() * z0 + (1.0 - ab).sqrt() * noise
        eps = self.m.unet_eps(zt, t)
        z0h = (zt - (1.0 - ab).sqrt() * eps) / ab.sqrt()
        rec = self.m.decode(z0h, scaled=True)
        return rec, {"t": t, "noise": noise, "eps": eps, "z0": z0, "zt": zt, "ab": ab, "rec": rec}

    def sample_t_noise(self, x: torch.Tensor, gen: torch.Generator, t_range: Optional[Tuple[int, int]] = None):
        lo, hi = t_range if t_range is not None else (self.t_min, self.t_max)
        t = g_randint(lo, hi, gen, n=x.shape[0]).to(x.device)
        lat_shape = (x.shape[0], 4, x.shape[-2] // 8, x.shape[-1] // 8)
        noise = torch.randn(lat_shape, generator=gen, device=gen.device).to(x.device)
        return t, noise

    def forward_aux(self, x, gen):
        t, noise = self.sample_t_noise(x, gen)
        rec, aux = self.reconstruct(x, t, noise)
        return rec, aux

    def forward(self, x, gen):
        return self.forward_aux(x, gen)[0]


class PurifierMixture:
    """Draws one purifier per EoT draw. ``exclude`` removes held-out variants from the draw; the
    remaining probabilities are renormalised."""

    def __init__(self, purifiers: Dict[str, TrainPurifier], probs: Optional[Dict[str, float]] = None,
                 exclude: Iterable[str] = ()):
        excl = set(exclude)
        names = [n for n in purifiers if n not in excl and (probs is None or probs.get(n, 0.0) > 0.0)]
        self.purifiers = {n: purifiers[n] for n in names}
        if not names:
            self.purifiers = {"none": Identity()}
            names = ["none"]
            ps = [1.0]
        else:
            ps = [float(probs[n]) if probs is not None else 1.0 for n in names]
        tot = sum(ps)
        self.names = names
        self.probs = torch.tensor([p / tot for p in ps], dtype=torch.float32)
        self.excluded = sorted(excl)

    def to_modules(self) -> nn.ModuleList:
        return nn.ModuleList(self.purifiers.values())

    def sample(self, x: torch.Tensor, gen: torch.Generator):
        """-> (purified x, aux or None, purifier name)."""
        if len(self.names) == 1:
            idx = 0
        else:
            idx = int(torch.multinomial(self.probs.to(gen.device), 1, generator=gen))
        name = self.names[idx]
        xp, aux = self.purifiers[name].forward_aux(x, gen)
        return ste_clamp(xp), aux, name


def _p(cfg: dict, key: str) -> dict:
    return ((cfg.get("purifier_train_params") or {}).get(key)) or {}


def build_purifier(name: str, cfg: dict, models: Optional[SDModels]) -> TrainPurifier:
    p = _p(cfg, name)
    if name == "jpeg":
        return DiffJPEG(p.get("quality", (50, 95)), impl=p.get("impl", "kornia_diff"))
    if name in ("downup", "downup_heldout"):
        return DownUp(p.get("scale", (0.5, 0.9)), p.get("modes", ("bilinear", "bicubic")),
                      p.get("antialias", True))
    if name == "vae_roundtrip":
        return VAERoundTrip(models)
    if name == "sd_onestep":
        return SDOneStepPurifier(models, p.get("t_min", 50), p.get("t_max", 300))
    raise KeyError(f"unknown training purifier {name!r}")


def build_mixture(cfg: dict, models: Optional[SDModels]) -> PurifierMixture:
    """Build from cfg keys ``purifiers_train`` (list), ``purifier_train_probs``, ``purifier_train_params``,
    ``purifier_train_exclude`` (held-out names removed from the draw, default ['downup_heldout']).
    An empty ``purifiers_train`` gives an identity-only mixture (ablation abl_no_pur)."""
    names = list(cfg.get("purifiers_train") or [])
    purs = {n: build_purifier(n, cfg, models) for n in names}
    return PurifierMixture(purs, cfg.get("purifier_train_probs"),
                           exclude=cfg.get("purifier_train_exclude", ["downup_heldout"]))


def build_heldout(cfg: dict) -> Dict[str, TrainPurifier]:
    """Differentiable held-out variant(s) that are never in the training mixture (early check only;
    the real held-out evaluation is Half B). ``downup_heldout``: scale range disjoint from the
    training down-up range (configured in purifier_train_params.downup_heldout)."""
    return {"downup_heldout": build_purifier("downup_heldout", cfg, None)}
