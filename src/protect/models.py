"""Frozen model bundle for the v2 protection loop (Half A only).

Everything here is lazy: nothing is downloaded or imported from diffusers /
transformers until a ``load_*`` function is called, so the rest of the package
(and the CPU unit tests) import without those libraries.

Model ids come from the config (``models.sd15.id`` etc.). Status of the default ids,
checked 2026-10-14 only by an HTTP 200 on ``huggingface.co/api/models/<id>``
(NOT by loading weights, so revisions are still unpinned; pin ``revision`` after the first
Kaggle env probe):
  stable-diffusion-v1-5/stable-diffusion-v1-5   200 (mirror; ``runwayml/...`` now answers 307)
  stabilityai/stable-diffusion-xl-base-1.0      200
  madebyollin/sdxl-vae-fp16-fix                 200
  openai/clip-vit-base-patch32                  200

Conventions (see V2_PLAN.md R1, R5, R6):
  * images are tensors in [-1, 1];
  * ``encode_mean(x, scaled=False)`` is the unscaled VAE posterior mean (what L_vis uses, so
    alpha keeps v1's scale); ``scaled=True`` multiplies by ``vae.config.scaling_factor``
    (what the UNet expects). The factor is read from the VAE config, never hard-coded;
  * the VAE runs in fp32 for protection, the UNet in fp16 / fp32 / fp32+autocast
    (``unet_dtype``), outputs are always returned in fp32.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn

DEFAULT_IDS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sdxl_vae_fp16": "madebyollin/sdxl-vae-fp16-fix",
    "clip_sem": "openai/clip-vit-base-patch32",
}


def sd15_alphas_cumprod(n: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012) -> torch.Tensor:
    """SD1.5 'scaled_linear' DDPM schedule. Used as a fallback / for tests; the loader prefers the
    scheduler shipped with the checkpoint."""
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=torch.float64) ** 2
    return torch.cumprod(1.0 - betas, dim=0).float()


def _dtype(name) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    return {"fp16": torch.float16, "float16": torch.float16, "fp32": torch.float32,
            "float32": torch.float32, "autocast": torch.float32}[str(name)]


def _freeze(m: nn.Module) -> nn.Module:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class SDModels:
    """Frozen VAE + UNet + schedule + null-prompt condition.

    ``vae`` must expose ``encode(x).latent_dist.mean`` and ``decode(z).sample`` (diffusers
    AutoencoderKL signature) and ``unet(zt, t, encoder_hidden_states=c).sample`` (UNet2DConditionModel
    signature). Tests inject tiny stand-ins with those signatures.
    """

    def __init__(self, vae, unet, alphas_cumprod: torch.Tensor, null_cond: torch.Tensor,
                 device="cpu", unet_dtype="fp32", scaling_factor: Optional[float] = None,
                 grad_checkpointing: bool = False):
        self.device = torch.device(device)
        self.unet_mode = str(unet_dtype)
        self.unet_dtype = _dtype(unet_dtype)
        self.vae = _freeze(vae).to(self.device).float()
        self.unet = _freeze(unet).to(self.device).to(self.unet_dtype)
        sf = scaling_factor
        if sf is None:
            sf = getattr(getattr(vae, "config", None), "scaling_factor", None)
        if sf is None:
            raise ValueError("scaling_factor not given and not found in vae.config (never hard-coded; R1)")
        self.scaling_factor = float(sf)
        self.alphas_cumprod = alphas_cumprod.to(self.device).float()
        self.num_train_timesteps = int(self.alphas_cumprod.numel())
        self.null_cond = null_cond.to(self.device).to(self.unet_dtype)
        self.grad_checkpointing = False
        self.set_grad_checkpointing(grad_checkpointing)

    # -- switches ---------------------------------------------------------------------------
    def set_grad_checkpointing(self, on: bool) -> None:
        """Gradient checkpointing for UNet and VAE. In some diffusers versions checkpointing is only
        active when ``module.training`` is True (R6), so the modules are put in train() mode when it
        is on. SD1.5 UNet/VAE have no dropout / batch-norm, so outputs are unchanged."""
        self.grad_checkpointing = bool(on)
        for m in (self.vae, self.unet):
            if on and hasattr(m, "enable_gradient_checkpointing"):
                m.enable_gradient_checkpointing()
                m.train()
            elif not on and hasattr(m, "disable_gradient_checkpointing"):
                m.disable_gradient_checkpointing()
                m.eval()

    # -- VAE --------------------------------------------------------------------------------
    def encode_mean(self, x: torch.Tensor, scaled: bool = False) -> torch.Tensor:
        z = self.vae.encode(x.float()).latent_dist.mean
        return z * self.scaling_factor if scaled else z

    def decode(self, z: torch.Tensor, scaled: bool = True) -> torch.Tensor:
        """Decode latents. ``scaled=True``: ``z`` is in UNet scale (divides by the scaling factor)."""
        if scaled:
            z = z / self.scaling_factor
        return self.vae.decode(z.float()).sample

    # -- UNet -------------------------------------------------------------------------------
    def unet_eps(self, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Noise prediction with the empty-prompt condition; fp32 in / fp32 out."""
        cond = self.null_cond.expand(zt.shape[0], -1, -1)
        ctx = (torch.autocast(self.device.type, dtype=torch.float16)
               if self.unet_mode == "autocast" and self.device.type == "cuda" else nullcontext())
        with ctx:
            out = self.unet(zt.to(self.unet_dtype), t, encoder_hidden_states=cond).sample
        return out.float()

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        return self.alphas_cumprod[t.to(self.device)].view(-1, 1, 1, 1)


# ---------------------------------------------------------------------------------------------
# Real loaders (lazy imports)
# ---------------------------------------------------------------------------------------------
def _model_cfg(cfg: dict, key: str) -> tuple:
    m = (cfg.get("models") or {}).get(key) or {}
    return m.get("id") or DEFAULT_IDS[key], m.get("revision")


def load_sd15(cfg: dict, device="cuda") -> SDModels:
    """Load SD1.5 VAE, UNet, scheduler alphas and the null-prompt text condition from HF.

    cfg keys used: models.sd15.{id,revision}; runtime.{unet_dtype,grad_checkpointing}.
    The text encoder is used once for the empty prompt and then freed.
    """
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    mid, rev = _model_cfg(cfg, "sd15")
    kw = {"revision": rev} if rev else {}
    rt = cfg.get("runtime") or {}
    unet_mode = rt.get("unet_dtype", "fp16")
    if torch.device(device).type == "cpu" and unet_mode == "fp16":
        unet_mode = "fp32"  # fp16 on CPU is slow / unsupported for several ops

    vae = AutoencoderKL.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32, **kw)
    unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet", torch_dtype=_dtype(unet_mode), **kw)
    try:
        ac = DDPMScheduler.from_pretrained(mid, subfolder="scheduler", **kw).alphas_cumprod.float()
    except Exception:  # scheduler config unreadable: SD1.5 scaled_linear schedule
        ac = sd15_alphas_cumprod()
    tok = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer", **kw)
    te = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=torch.float32, **kw).eval()
    ids = tok([""], padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        null_cond = te(ids.input_ids)[0]  # (1, 77, 768)
    del te
    return SDModels(vae, unet, ac, null_cond, device=device, unet_dtype=unet_mode,
                    grad_checkpointing=bool(rt.get("grad_checkpointing", True)))


def load_sdxl_vae(cfg: dict, device="cuda") -> nn.Module:
    """SDXL VAE (fp16-fix variant, run in fp32 here). Returns the diffusers module; its own
    ``config.scaling_factor`` (0.13025) is NOT used by L_vis (unscaled latent mean)."""
    from diffusers import AutoencoderKL

    mid, rev = _model_cfg(cfg, "sdxl_vae_fp16")
    kw = {"revision": rev} if rev else {}
    vae = AutoencoderKL.from_pretrained(mid, torch_dtype=torch.float32, **kw)
    vae = _freeze(vae).to(device).float()
    if (cfg.get("runtime") or {}).get("grad_checkpointing", True) and hasattr(vae, "enable_gradient_checkpointing"):
        vae.enable_gradient_checkpointing()
        vae.train()
    return vae


class ClipSem:
    """CLIP image encoder + fixed text target for L_sem = 1 - cos(img_emb, txt_emb)."""

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, vision, text_embed: torch.Tensor, device="cpu"):
        self.vision = _freeze(vision).to(device)
        self.text_embed = text_embed.detach().to(device).float()
        self.device = torch.device(device)
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1)

    def image_embed(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x + 1.0) / 2.0
        x224 = torch.nn.functional.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
        out = self.vision((x224 - self.mean) / self.std)
        return out.image_embeds if hasattr(out, "image_embeds") else out

    def sem_loss(self, x: torch.Tensor) -> torch.Tensor:
        e = self.image_embed(x)
        return (1.0 - torch.nn.functional.cosine_similarity(e, self.text_embed.expand_as(e), dim=1)).mean()


def load_clip_sem(cfg: dict, device="cuda") -> ClipSem:
    """CLIP B/32 vision tower + text embedding of ``loss.sem_target_text`` (v1: "a potted plant")."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

    mid, rev = _model_cfg(cfg, "clip_sem")
    vision = CLIPVisionModelWithProjection.from_pretrained(mid).eval()
    text = CLIPTextModelWithProjection.from_pretrained(mid).eval()
    tok = CLIPTokenizer.from_pretrained(mid)
    prompt = (cfg.get("loss") or {}).get("sem_target_text", "a potted plant")
    with torch.no_grad():
        emb = text(**tok([prompt], return_tensors="pt", padding=True, truncation=True)).text_embeds
    return ClipSem(vision, emb, device=device)


# Process-wide cache so several presets / Optuna trials in one process share weights.
_CACHE: dict = {}


def get_sd15(cfg: dict, device="cuda") -> SDModels:
    mid, rev = _model_cfg(cfg, "sd15")
    key = ("sd15", mid, rev, str(device), (cfg.get("runtime") or {}).get("unet_dtype", "fp16"))
    if key not in _CACHE:
        _CACHE[key] = load_sd15(cfg, device)
    return _CACHE[key]


def get_sdxl_vae(cfg: dict, device="cuda"):
    mid, rev = _model_cfg(cfg, "sdxl_vae_fp16")
    key = ("sdxl_vae", mid, rev, str(device))
    if key not in _CACHE:
        _CACHE[key] = load_sdxl_vae(cfg, device)
    return _CACHE[key]


def get_clip_sem(cfg: dict, device="cuda") -> ClipSem:
    mid, rev = _model_cfg(cfg, "clip_sem")
    key = ("clip_sem", mid, rev, str(device), (cfg.get("loss") or {}).get("sem_target_text", "a potted plant"))
    if key not in _CACHE:
        _CACHE[key] = load_clip_sem(cfg, device)
    return _CACHE[key]


# ---------------------------------------------------------------------------------------------
# Generator-aware random helpers: every random draw in the loop goes through one torch.Generator
# (created on the working device, seeded per image) so a protection run is reproducible (R2, R15).
# ---------------------------------------------------------------------------------------------
def make_generator(seed: int, device="cpu") -> torch.Generator:
    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


def g_randn_like(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return torch.randn(x.shape, generator=gen, device=gen.device, dtype=x.dtype).to(x.device)


def g_uniform(lo: float, hi: float, gen: torch.Generator, shape=()) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, generator=gen, device=gen.device)


def g_randint(lo: int, hi_inclusive: int, gen: torch.Generator, n: int = 1) -> torch.Tensor:
    """Integers in [lo, hi_inclusive] (torch.randint excludes the upper bound; R2)."""
    return torch.randint(int(lo), int(hi_inclusive) + 1, (n,), generator=gen, device=gen.device)
