"""Baseline registry: ``build_baseline(name, cfg, device)`` for noise_uniform, noise_gauss_psnr, photoguard, mist.

Dispatched lazily from ``src/protect/registry.py`` (A1). Every returned object satisfies the frozen Protector
contract: ``protect(x: Tensor[1,3,H,W] in [-1,1], seed: int) -> (Tensor, dict)``.

Config (``config/parts/baselines.yaml``, key ``baselines``); budget defaults to ``budget.eps_255`` (standard 0-255
scale). The SD loader here is deliberately minimal and self-contained (A2 has its own); it loads only what a
baseline needs (VAE; plus UNet + text encoder for Mist) and caches them per process.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

NAMES = ("noise_uniform", "noise_gauss_psnr", "photoguard", "mist")
_ALIASES = {"noise": "noise_uniform", "noise_gauss": "noise_gauss_psnr", "gauss_psnr": "noise_gauss_psnr",
            "photoguard_encoder": "photoguard"}

_SD_CACHE: Dict[tuple, Dict[str, Any]] = {}

_DEFAULTS = {
    "noise_uniform": {"variant": "uniform"},
    "noise_gauss_psnr": {"target_psnr": 41.911, "psnr_csv": None, "psnr_key_col": "Original_Filename",
                         "psnr_val_col": "PSNR_dB"},
    "photoguard": {"step_255": 2.55, "iters": 1000, "target": "zero_latent", "step_decay": True, "loss_scale": 1024.0},
    "mist": {"step_255": 1.0, "iters": 100, "w_sem": 1e4, "w_tex": 1.0, "t_min": 0, "t_max": 1000,
             "n_noise": 1, "prompt": "a painting", "target_image": None, "loss_scale": 1024.0, "grad_checkpointing": True},
}


def canonical(name: str) -> str:
    n = _ALIASES.get(name, name)
    if n not in NAMES:
        raise KeyError(f"unknown baseline '{name}' (known: {', '.join(NAMES)})")
    return n


def resolve_params(name: str, cfg: Optional[dict]) -> dict:
    """Defaults < cfg['baselines']['_common'] < cfg['baselines'][name]; eps_255 falls back to budget.eps_255 (else 4)."""
    cfg = cfg or {}
    b = cfg.get("baselines", {}) or {}
    p = dict(_DEFAULTS.get(name, {}))
    p.update(b.get("_common", {}) or {})
    p.update(b.get(name, {}) or {})
    if p.get("eps_255") is None:
        p["eps_255"] = float((cfg.get("budget", {}) or {}).get("eps_255", 4))
    return p


def _model_cfg(cfg: dict) -> dict:
    m = ((cfg.get("models", {}) or {}).get("sd15")) or {}
    return {"id": m.get("id", "stable-diffusion-v1-5/stable-diffusion-v1-5"), "revision": m.get("revision")}


def load_sd_components(cfg: dict, device, need_unet: bool = False, dtype: str = "auto",
                       grad_checkpointing: bool = True, prompt: str = "") -> Dict[str, Any]:
    """Minimal lazy SD1.5 loader. UNVERIFIED here: weights were never downloaded in the coding workspace."""
    import torch
    dev = torch.device(device)
    if dtype == "auto":
        dtype = "fp16" if dev.type == "cuda" else "fp32"
    td = {"fp16": torch.float16, "fp32": torch.float32}[dtype]
    mc = _model_cfg(cfg)
    key = (mc["id"], mc["revision"], str(dev), dtype, bool(need_unet), prompt)
    if key in _SD_CACHE:
        return _SD_CACHE[key]
    from diffusers import AutoencoderKL
    kw = {"revision": mc["revision"]} if mc["revision"] else {}
    vae = AutoencoderKL.from_pretrained(mc["id"], subfolder="vae", torch_dtype=td, **kw).to(dev).eval()
    vae.requires_grad_(False)
    comp: Dict[str, Any] = {"vae": vae, "dtype": td, "scaling_factor": float(vae.config.scaling_factor)}
    if need_unet:
        from diffusers import DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        unet = UNet2DConditionModel.from_pretrained(mc["id"], subfolder="unet", torch_dtype=td, **kw).to(dev)
        unet.requires_grad_(False)
        if grad_checkpointing:
            unet.enable_gradient_checkpointing()
            unet.train()  # R6: some diffusers versions only checkpoint in training mode (dropout is 0 in SD1.5)
        else:
            unet.eval()
        tok = CLIPTokenizer.from_pretrained(mc["id"], subfolder="tokenizer", **kw)
        te = CLIPTextModel.from_pretrained(mc["id"], subfolder="text_encoder", torch_dtype=td, **kw).to(dev).eval()
        with torch.no_grad():
            ids = tok(prompt, padding="max_length", max_length=tok.model_max_length, truncation=True,
                      return_tensors="pt").input_ids.to(dev)
            cond = te(ids)[0].to(td)
        sched = DDPMScheduler.from_pretrained(mc["id"], subfolder="scheduler", **kw)
        comp.update({"unet": unet, "cond": cond, "alphas_cumprod": sched.alphas_cumprod.to(dev).float()})
        del te
    _SD_CACHE[key] = comp
    return comp


def vae_mean_encode_fn(vae):
    """x [B,3,H,W] in [-1,1] -> posterior mean latent (float32, unscaled). Differentiable."""
    def f(x):
        return vae.encode(x.to(vae.dtype)).latent_dist.mean.float()
    return f


def unet_eps_fn(unet):
    def f(zt, t, cond):
        c = cond.expand(zt.shape[0], -1, -1)
        return unet(zt.to(unet.dtype), t, encoder_hidden_states=c).sample.float()
    return f


def _load_target_image(path: str):
    import numpy as np
    import torch
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def build_baseline(name: str, cfg: Optional[dict], device="cpu"):
    """Return a Protector for ``name`` (see module docstring)."""
    name = canonical(name)
    p = resolve_params(name, cfg)
    if name == "noise_uniform":
        from .noise_control import UniformNoiseProtector
        return UniformNoiseProtector(eps_255=p["eps_255"], variant=p.get("variant", "uniform"))
    if name == "noise_gauss_psnr":
        from .noise_control import GaussianPSNRProtector, load_psnr_map
        pm = None
        if p.get("psnr_csv"):
            pm = load_psnr_map(p["psnr_csv"], p["psnr_key_col"], p["psnr_val_col"])  # raises if missing
        return GaussianPSNRProtector(target_psnr=p["target_psnr"], psnr_map=pm)
    cfg = cfg or {}
    if name == "photoguard":
        from .photoguard import EncoderAttackProtector
        comp = load_sd_components(cfg, device, need_unet=False, dtype=p.get("dtype", "auto"))
        ti = _load_target_image(p["target_image"]) if p.get("target_image") else None
        prot = EncoderAttackProtector(
            vae_mean_encode_fn(comp["vae"]), eps_255=p["eps_255"], step_255=p["step_255"], iters=p["iters"],
            target="image" if ti is not None else p["target"], target_image=ti, step_decay=p["step_decay"],
            loss_scale=p["loss_scale"] if comp["dtype"].itemsize == 2 else 1.0)
        prot.device = str(device)  # run_protect moves the input image to protector.device (S04 failure: cpu image, cuda VAE)
        return prot
    if name == "mist":
        from .mist import MistProtector
        comp = load_sd_components(cfg, device, need_unet=True, dtype=p.get("dtype", "auto"),
                                  grad_checkpointing=bool(p["grad_checkpointing"]), prompt=p["prompt"])
        ti = _load_target_image(p["target_image"]) if p.get("target_image") else None
        prot = MistProtector(
            vae_mean_encode_fn(comp["vae"]), unet_eps_fn(comp["unet"]), comp["alphas_cumprod"], comp["cond"],
            eps_255=p["eps_255"], step_255=p["step_255"], iters=p["iters"], w_sem=p["w_sem"], w_tex=p["w_tex"],
            t_min=p["t_min"], t_max=p["t_max"], n_noise=p["n_noise"], scaling_factor=comp["scaling_factor"],
            target_image=ti, loss_scale=p["loss_scale"] if comp["dtype"].itemsize == 2 else 1.0)
        prot.device = str(device)
        return prot
    raise KeyError(name)  # pragma: no cover
