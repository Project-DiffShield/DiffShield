"""Held-out + test purifiers (Half B only). Never imported from ``src.protect``.

Contract (frozen): ``EvalPurifier.apply(img: PIL.Image, seed: int) -> PIL.Image``.

Registry names (canonical):
  none, jpeg_q90, jpeg_q75, jpeg_q50, jpeg_other, downup_0.5, downup_other, vae_rt,
  diffpure_0.05, diffpure_0.1, diffpure_0.2, diffpure_0.3, restore_swin2sr, gridpure, adaptive_ae
Aliases from config.yaml / the handoff (jpeg_q_other, vae_rt_sd15, diffpure_t0.1, ...) are resolved by
``canonical_name``.

Purifiers are *real* codecs / models on saved PNG-quality PIL images, never the differentiable training
approximations. ``adaptive_ae`` is a lazy plug-in: ``src/eval/adaptive_ae.py`` (agent A4) must expose
``build_purifier(cfg: dict, device) -> EvalPurifier``.

Heavy imports (torch, diffusers, transformers) are lazy; JPEG / down-up / none work with PIL only.
"""
from __future__ import annotations

import importlib
import io
import re
from typing import Callable, Dict, List, Optional

import numpy as np

_ALIASES = {
    "jpeg_q_other": "jpeg_other",
    "vae_rt_sd15": "vae_rt",
    "diffpure_t0.05": "diffpure_0.05",
    "diffpure_t0.1": "diffpure_0.1",
    "diffpure_t0.2": "diffpure_0.2",
    "diffpure_t0.3": "diffpure_0.3",
}


class NotAvailable(RuntimeError):
    """A purifier/model cannot be used in this environment (documented limitation, not a silent drop)."""


class EvalPurifier:
    """Base class. ``apply`` must be deterministic given (img, seed)."""
    name = "base"
    needs_gpu = False
    share_key: Optional[str] = None     # purifiers with the same key may be kept loaded together

    def load(self, device: Optional[str] = None):
        return self

    def apply(self, img, seed: int):
        raise NotImplementedError

    def unload(self):
        pass

    def params(self) -> dict:
        return {}


def _pil_resample(name: str):
    from PIL import Image
    return {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "lanczos": Image.LANCZOS,
            "nearest": Image.NEAREST, "box": Image.BOX}[name]


# ----------------------------------------------------------------------------------------------- cheap (PIL)
class NonePurifier(EvalPurifier):
    name = "none"

    def apply(self, img, seed):
        return img.convert("RGB")


class JpegPurifier(EvalPurifier):
    """Real libjpeg round trip through PIL (quality q; subsampling None = PIL default 4:2:0, 0 = 4:4:4)."""

    def __init__(self, name: str, quality: int, subsampling: Optional[int] = None):
        self.name, self.quality, self.subsampling = name, int(quality), subsampling

    def apply(self, img, seed):
        from PIL import Image
        buf = io.BytesIO()
        kw = {"quality": self.quality}
        if self.subsampling is not None:
            kw["subsampling"] = self.subsampling
        img.convert("RGB").save(buf, format="JPEG", **kw)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out.convert("RGB")

    def params(self):
        return {"quality": self.quality, "subsampling": self.subsampling}


class DownUpPurifier(EvalPurifier):
    """Resize by ``scale`` then back to the original size with the same resample filter."""

    def __init__(self, name: str, scale: float, resample: str = "bicubic"):
        self.name, self.scale, self.resample = name, float(scale), resample

    def apply(self, img, seed):
        img = img.convert("RGB")
        w, h = img.size
        rs = _pil_resample(self.resample)
        small = img.resize((max(1, int(round(w * self.scale))), max(1, int(round(h * self.scale)))), rs)
        return small.resize((w, h), rs)

    def params(self):
        return {"scale": self.scale, "resample": self.resample}


# ------------------------------------------------------------------------------------------- model-based
def _pick_device(device: Optional[str]) -> str:
    if device:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pad8(arr: np.ndarray):
    h, w = arr.shape[:2]
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    return arr, h, w


class VaeRoundTrip(EvalPurifier):
    """SD VAE encode -> decode with a separately loaded VAE instance (never shared with the protection code)."""
    name = "vae_rt"
    needs_gpu = True

    def __init__(self, model_id: str, revision: Optional[str] = None, latent: str = "mean"):
        self.model_id, self.revision, self.latent = model_id, revision, latent
        self.vae = None
        self.device = None
        self.share_key = f"vae:{model_id}"

    def load(self, device=None):
        if self.vae is not None:
            return self
        import torch
        from diffusers import AutoencoderKL
        self.device = _pick_device(device)
        dt = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae", revision=self.revision,
                                                 torch_dtype=dt).to(self.device).eval()
        return self

    def apply(self, img, seed):
        import torch
        from PIL import Image
        self.load()
        arr, h, w = _pad8(np.asarray(img.convert("RGB")))
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        x = x.to(self.device, dtype=self.vae.dtype)
        with torch.no_grad():
            post = self.vae.encode(x).latent_dist
            if self.latent == "mean":
                z = post.mean
            else:
                g = torch.Generator(device=self.device).manual_seed(int(seed))
                z = post.sample(generator=g)
            y = self.vae.decode(z).sample
        y = ((y.float().clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(y[:h, :w], mode="RGB")

    def unload(self):
        self.vae = None
        _free_cuda()

    def params(self):
        return {"model": self.model_id, "latent": self.latent}


class DiffPurePurifier(EvalPurifier):
    """Multi-step DiffPure-style cleanup with SD1.5: forward-noise to t* (= ``strength``), then deterministic DDIM
    denoise with an empty prompt, guidance 1.0 (so no CFG). Implemented with the diffusers img2img pipeline, so the
    output also passes through the SD VAE (inherent to SD-based DiffPure; documented). t* in {0.05, 0.1, 0.2, 0.3}.
    Shares the pipeline between instances (``share_key``)."""
    needs_gpu = True
    _POOL: Dict[str, list] = {}      # share_key -> [pipe, refcount]

    def __init__(self, name: str, model_id: str, strength: float, steps: int = 50, guidance: float = 1.0,
                 revision: Optional[str] = None):
        self.name, self.model_id, self.strength, self.steps, self.guidance = name, model_id, float(strength), int(steps), float(guidance)
        self.revision = revision
        self.share_key = f"diffpure:{model_id}"
        self.pipe = None
        self.device = None

    def load(self, device=None):
        if self.pipe is not None:
            return self
        import torch
        self.device = _pick_device(device)
        entry = DiffPurePurifier._POOL.get(self.share_key)
        if entry is None:
            from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
            dt = torch.float16 if self.device.startswith("cuda") else torch.float32
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id, revision=self.revision, torch_dtype=dt, safety_checker=None,
                requires_safety_checker=False).to(self.device)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.set_progress_bar_config(disable=True)
            entry = [pipe, 0]
            DiffPurePurifier._POOL[self.share_key] = entry
        entry[1] += 1
        self.pipe = entry[0]
        return self

    def apply(self, img, seed):
        import torch
        self.load()
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.pipe(prompt="", image=img.convert("RGB"), strength=self.strength, num_inference_steps=self.steps,
                        guidance_scale=self.guidance, generator=g, output_type="pil").images[0]
        return out.convert("RGB")

    def unload(self):
        if self.pipe is None:
            return
        entry = DiffPurePurifier._POOL.get(self.share_key)
        self.pipe = None
        if entry is not None:
            entry[1] -= 1
            if entry[1] <= 0:
                DiffPurePurifier._POOL.pop(self.share_key, None)
                _free_cuda()

    def params(self):
        return {"model": self.model_id, "strength": self.strength, "steps": self.steps, "guidance": self.guidance}


class Swin2SRRestore(EvalPurifier):
    """Restoration-style purifier: bicubic down by ``down_scale`` then Swin2SR x2 super-resolution back up.
    HF id caidas/swin2SR-classical-sr-x2-64 is UNVERIFIED (not loaded in this workspace)."""
    name = "restore_swin2sr"
    needs_gpu = True

    def __init__(self, model_id: str, down_scale: float = 0.5, revision: Optional[str] = None):
        self.model_id, self.down_scale, self.revision = model_id, float(down_scale), revision
        self.model = None
        self.proc = None
        self.device = None

    def load(self, device=None):
        if self.model is not None:
            return self
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
        self.device = _pick_device(device)
        self.proc = AutoImageProcessor.from_pretrained(self.model_id, revision=self.revision)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(self.model_id, revision=self.revision).to(self.device).eval()
        return self

    def apply(self, img, seed):
        import torch
        from PIL import Image
        self.load()
        img = img.convert("RGB")
        w, h = img.size
        small = img.resize((max(1, int(round(w * self.down_scale))), max(1, int(round(h * self.down_scale)))), Image.BICUBIC)
        inp = self.proc(small, return_tensors="pt").to(self.device)
        with torch.no_grad():
            rec = self.model(**inp).reconstruction
        arr = rec[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        sw, sh = small.size
        arr = arr[: sh * 2, : sw * 2]               # processor may pad to the window size; crop back
        out = Image.fromarray(np.rint(arr * 255).astype(np.uint8), mode="RGB")
        if out.size != (w, h):
            out = out.resize((w, h), Image.BICUBIC)
        return out

    def unload(self):
        self.model = None
        self.proc = None
        _free_cuda()

    def params(self):
        return {"model": self.model_id, "down_scale": self.down_scale}


def _free_cuda():
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------------------------- registry
def canonical_name(name: str, cfg: Optional[dict] = None) -> str:
    aliases = dict(_ALIASES)
    if cfg:
        aliases.update(((cfg.get("eval") or {}).get("purifier_aliases") or {}))
    return aliases.get(name, name)


def _ev(cfg: Optional[dict]) -> dict:
    return (cfg or {}).get("eval", {}) or {}


def _model(cfg, key, default_id):
    m = (_ev(cfg).get("models") or {}).get(key) or {}
    return m.get("id", default_id), m.get("revision")


def _params(cfg, name) -> dict:
    return dict((_ev(cfg).get("purifiers") or {}).get(name) or {})


def _build_jpeg(name, cfg, device):
    p = {"quality": {"jpeg_q90": 90, "jpeg_q75": 75, "jpeg_q50": 50, "jpeg_other": 65}[name]}
    if name == "jpeg_other":
        p["subsampling"] = 0
    p.update(_params(cfg, name))
    return JpegPurifier(name, p["quality"], p.get("subsampling"))


def _build_downup(name, cfg, device):
    p = {"downup_0.5": {"scale": 0.5, "resample": "bicubic"},
         "downup_other": {"scale": 0.375, "resample": "lanczos"}}[name]
    p.update(_params(cfg, name))
    return DownUpPurifier(name, p["scale"], p["resample"])


def _build_vae(name, cfg, device):
    p = {"model": "sd15", "latent": "mean"}
    p.update(_params(cfg, name))
    mid, rev = _model(cfg, p["model"], "stable-diffusion-v1-5/stable-diffusion-v1-5")
    return VaeRoundTrip(mid, rev, p["latent"])


def _build_diffpure(name, cfg, device):
    t = float(name.split("_")[1])
    p = {"model": "sd15", "strength": t, "steps": 50, "guidance": 1.0}
    p.update(_params(cfg, name))
    mid, rev = _model(cfg, p["model"], "stable-diffusion-v1-5/stable-diffusion-v1-5")
    pur = DiffPurePurifier(name, mid, p["strength"], p["steps"], p["guidance"], rev)
    return pur


def _build_swin(name, cfg, device):
    p = {"down_scale": 0.5, "model": "swin2sr"}
    p.update(_params(cfg, name))
    mid, rev = _model(cfg, p["model"], "caidas/swin2SR-classical-sr-x2-64")
    return Swin2SRRestore(mid, p["down_scale"], rev)


def _build_gridpure(name, cfg, device):
    from .gridpure import GridPurePurifier   # lazy
    return GridPurePurifier(_params(cfg, "gridpure"), device)


_BUILDERS: Dict[str, Callable] = {"none": lambda n, c, d: NonePurifier()}
for _n in ("jpeg_q90", "jpeg_q75", "jpeg_q50", "jpeg_other"):
    _BUILDERS[_n] = _build_jpeg
for _n in ("downup_0.5", "downup_other"):
    _BUILDERS[_n] = _build_downup
_BUILDERS["vae_rt"] = _build_vae
for _n in ("diffpure_0.05", "diffpure_0.1", "diffpure_0.2", "diffpure_0.3"):
    _BUILDERS[_n] = _build_diffpure
_BUILDERS["restore_swin2sr"] = _build_swin
_BUILDERS["gridpure"] = _build_gridpure

# Plug-ins resolved lazily by module name (src/eval/<name>.py must expose build_purifier(cfg, device)).
_PLUGINS = {"adaptive_ae": "adaptive_ae"}


def available_names() -> List[str]:
    return sorted(set(_BUILDERS) | set(_PLUGINS))


def register(name: str, builder: Callable) -> None:
    """Register a purifier builder ``builder(name, cfg, device) -> EvalPurifier`` (tests / extensions)."""
    _BUILDERS[name] = builder


def build_purifier(name: str, cfg: Optional[dict] = None, device: Optional[str] = None) -> EvalPurifier:
    """Registry lookup. Raises NotAvailable for unknown names / missing plug-ins / missing weights."""
    name = canonical_name(name, cfg)
    if name in _BUILDERS:
        return _BUILDERS[name](name, cfg, device)
    if name in _PLUGINS:
        try:
            mod = importlib.import_module(f"{__package__}.{_PLUGINS[name]}")
        except Exception as e:
            raise NotAvailable(f"purifier '{name}' needs src/eval/{_PLUGINS[name]}.py (not importable: {e})")
        if not hasattr(mod, "build_purifier"):
            raise NotAvailable(f"src/eval/{_PLUGINS[name]}.py must define build_purifier(cfg, device)")
        return mod.build_purifier(cfg or {}, device)
    raise NotAvailable(f"unknown purifier '{name}'; known: {available_names()}")


def purifier_params(p: EvalPurifier) -> dict:
    try:
        return p.params()
    except Exception:
        return {}
