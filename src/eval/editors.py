"""Editors an attacker would use on a (protected, possibly purified) face photo. Half B only.

Interface (frozen): ``Editor.load(device)``, ``Editor.edit_batch(img, prompt, strength, seeds) -> list[PIL]``,
``Editor.unload()``. We add one optional keyword, ``mask`` (PIL 'L', 255 = repaint), used only by the inpaint editor.

Common settings (48h plan): DPM-Solver++ multistep, ``eval.num_inference_steps`` (30) steps, fp16, one
``torch.Generator`` per seed, seeds batched (``max_batch`` per editor), no safety checker, no negative prompt.

Verification status (no GPU / no downloads in the authoring workspace; everything below is UNVERIFIED by execution):
  * pipeline classes / ``load_ip_adapter(..., image_encoder_folder=None)`` / ``ip_adapter_image_embeds`` recipe for
    IP-Adapter FaceID follow the current diffusers IP-Adapter documentation page (read, not executed);
  * model ids: see config/parts/eval.yaml (``unverified`` comments);
  * IP-Adapter FaceID is run as *img2img* (init image = input photo, strength 0.55) with identity injected through
    the ArcFace embedding. A pure text2img FaceID editor would not depend on the input pixels at all, so the
    pixel-space protection could not act on it; img2img keeps the VAE path attackable, the embedding path measures
    what L_id (Phase 6) is for. A no-face input uses an all-zero identity embedding (``last_info['no_face']``).
  * the SD1.5 FaceID LoRA (``ip-adapter-faceid_sd15_lora.safetensors``) is NOT loaded (docs example works without it).
"""
from __future__ import annotations

import contextlib
from typing import Callable, Dict, List, Optional, Sequence

PILImage = object


def _lazy_torch():
    import torch
    return torch


class Editor:
    """Base class / interface."""
    name = "base"
    needs_mask = False
    default_strength: Optional[float] = 0.55
    last_info: Dict = {}

    def load(self, device: str = "cuda"):
        raise NotImplementedError

    def edit_batch(self, img, prompt: str, strength: Optional[float], seeds: Sequence[int], mask=None) -> list:
        raise NotImplementedError

    def unload(self):
        pass

    def params(self) -> dict:
        return {}


def _to_dtype(name: str):
    torch = _lazy_torch()
    return {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}[name]


def set_dpmpp(pipe, kind: str = "dpmpp"):
    """DPM-Solver++ (multistep, 2nd order) by default; ``kind='ddim'`` switches to DDIM."""
    if kind == "ddim":
        from diffusers import DDIMScheduler
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++",
                                                                 solver_order=2)
    return pipe


class _PipeEditor(Editor):
    kind = "base"

    def __init__(self, ed_cfg: dict, ev_cfg: dict):
        self.ed_cfg = dict(ed_cfg or {})
        self.ev = ev_cfg
        m = self.ev["models"][self.ed_cfg["model"]]
        self.model_id, self.revision = m["id"], m.get("revision")
        self.steps = int(self.ev.get("num_inference_steps", 30))
        self.guidance = float(self.ev.get("guidance_scale", 7.5))
        self.dtype_name = self.ev.get("dtype", "float16")
        self.max_batch = int(self.ed_cfg.get("max_batch", 1))
        self.default_strength = self.ed_cfg.get("default_strength")
        self.needs_mask = bool(self.ed_cfg.get("needs_mask", False))
        self.pipe = None
        self.device = None
        self.last_info = {}

    # -- to be provided by subclasses
    def _build(self, device):
        raise NotImplementedError

    def _call(self, imgs: list, prompts: list, strength, gens: list, mask):
        raise NotImplementedError

    def _prep_image(self, img):
        return img.convert("RGB")

    def load(self, device="cuda"):
        if self.pipe is not None:
            return self
        self.device = device
        self.pipe = self._build(device)
        with contextlib.suppress(Exception):
            self.pipe.set_progress_bar_config(disable=True)
        return self

    def unload(self):
        self.pipe = None
        try:
            import gc
            gc.collect()
            torch = _lazy_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def edit_batch(self, img, prompt, strength, seeds, mask=None):
        if self.pipe is None:
            raise RuntimeError("call load() first")
        torch = _lazy_torch()
        if strength is None:
            strength = self.default_strength
        seeds = [int(s) for s in seeds]
        img = self._prep_image(img)
        outs: list = []
        for i in range(0, len(seeds), self.max_batch):
            chunk = seeds[i:i + self.max_batch]
            gens = [torch.Generator(device=self.device).manual_seed(s) for s in chunk]
            with torch.inference_mode():
                res = self._call([img] * len(chunk), [prompt] * len(chunk), strength, gens, mask)
            outs.extend(r.convert("RGB") for r in res)
        return outs

    def params(self):
        return {"kind": self.kind, "model": self.model_id, "steps": self.steps, "guidance": self.guidance,
                "sampler": "dpmpp_2m", "dtype": self.dtype_name}


class SD15Img2Img(_PipeEditor):
    name = kind = "sd15_img2img"

    def _build(self, device):
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=_to_dtype(self.dtype_name), safety_checker=None,
            requires_safety_checker=False).to(device)
        return set_dpmpp(pipe)

    def _call(self, imgs, prompts, strength, gens, mask):
        return self.pipe(prompt=prompts, image=imgs, strength=float(strength), guidance_scale=self.guidance,
                         num_inference_steps=self.steps, generator=gens, output_type="pil").images


class SDXLImg2Img(_PipeEditor):
    """SDXL img2img at 1024: the 512 input is upsampled bicubically (recorded in params; itself a mild purification,
    V2_PLAN R21). fp16-fix VAE avoids fp16 NaNs."""
    name = kind = "sdxl_img2img"

    def _build(self, device):
        torch = _lazy_torch()
        from diffusers import AutoencoderKL, StableDiffusionXLImg2ImgPipeline
        dt = _to_dtype(self.dtype_name)
        vae = None
        vm = self.ev["models"].get("sdxl_vae_fp16")
        if vm:
            vae = AutoencoderKL.from_pretrained(vm["id"], revision=vm.get("revision"), torch_dtype=dt)
        kw = dict(revision=self.revision, torch_dtype=dt, use_safetensors=True)
        if vae is not None:
            kw["vae"] = vae
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(self.model_id, **kw)
        set_dpmpp(pipe)
        if self.ed_cfg.get("cpu_offload", False):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        return pipe

    def _prep_image(self, img):
        from PIL import Image
        r = int(self.ed_cfg.get("resolution", 1024))
        img = img.convert("RGB")
        return img if img.size == (r, r) else img.resize((r, r), Image.BICUBIC)

    def _call(self, imgs, prompts, strength, gens, mask):
        return self.pipe(prompt=prompts, image=imgs, strength=float(strength), guidance_scale=self.guidance,
                         num_inference_steps=self.steps, generator=gens, output_type="pil").images

    def params(self):
        d = super().params()
        d.update({"resolution": int(self.ed_cfg.get("resolution", 1024)), "upsample": "bicubic",
                  "vae": (self.ev["models"].get("sdxl_vae_fp16") or {}).get("id")})
        return d


class InstructPix2Pix(_PipeEditor):
    """Instruction editing; no ``strength`` (image guidance is the analogue, 1.5)."""
    name = kind = "ip2p"

    def _build(self, device):
        from diffusers import StableDiffusionInstructPix2PixPipeline
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=_to_dtype(self.dtype_name), safety_checker=None,
            requires_safety_checker=False).to(device)
        return set_dpmpp(pipe)

    def _call(self, imgs, prompts, strength, gens, mask):
        return self.pipe(prompt=prompts, image=imgs, num_inference_steps=self.steps, guidance_scale=self.guidance,
                         image_guidance_scale=float(self.ev.get("ip2p_image_guidance", 1.5)), generator=gens,
                         output_type="pil").images

    def params(self):
        d = super().params()
        d["image_guidance"] = float(self.ev.get("ip2p_image_guidance", 1.5))
        return d


class SDInpaint(_PipeEditor):
    """SD inpainting with a keep-face mask: repaint = background/clothes (255), never the face. ``mask`` is mandatory.
    No pixel composite of the original is pasted back (the face is reproduced by the model from the masked-image
    latents, which is exactly the path the protection perturbs)."""
    name = kind = "sd_inpaint"
    needs_mask = True

    def _build(self, device):
        from diffusers import StableDiffusionInpaintPipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id, revision=self.revision, torch_dtype=_to_dtype(self.dtype_name), safety_checker=None,
            requires_safety_checker=False).to(device)
        return set_dpmpp(pipe)

    def _call(self, imgs, prompts, strength, gens, mask):
        if mask is None:
            raise ValueError("SDInpaint needs a keep-face mask (see src/eval/masks.py)")
        m = mask.convert("L")
        w, h = imgs[0].size
        if m.size != (w, h):
            m = m.resize((w, h))
        return self.pipe(prompt=prompts, image=imgs, mask_image=[m] * len(imgs), strength=float(strength),
                         guidance_scale=self.guidance, num_inference_steps=self.steps, generator=gens,
                         height=h, width=w, output_type="pil").images


class IPAdapterFaceID(_PipeEditor):
    """SD1.5 img2img + IP-Adapter FaceID (ArcFace embedding of the *input* photo as identity prompt)."""
    name = kind = "ipadapter_faceid"

    def __init__(self, ed_cfg, ev_cfg, face_scorer=None):
        super().__init__(ed_cfg, ev_cfg)
        self._face = face_scorer
        # base model is SD1.5 (not the FaceID repo)
        base = self.ed_cfg.get("base_model", "sd15")
        bm = self.ev["models"][base]
        self.base_id, self.base_rev = bm["id"], bm.get("revision")
        self.weight_name = self.ev["models"][self.ed_cfg["model"]].get("weight_name", "ip-adapter-faceid_sd15.bin")
        self.scale = float(self.ed_cfg.get("ip_adapter_scale", 0.6))

    def _build(self, device):
        from diffusers import StableDiffusionImg2ImgPipeline
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.base_id, revision=self.base_rev, torch_dtype=_to_dtype(self.dtype_name), safety_checker=None,
            requires_safety_checker=False).to(device)
        set_dpmpp(pipe, self.ed_cfg.get("scheduler", "dpmpp"))
        pipe.load_ip_adapter(self.model_id, subfolder=None, weight_name=self.weight_name, image_encoder_folder=None)
        pipe.set_ip_adapter_scale(self.scale)
        if self._face is None:
            from .faces import FaceScorer
            self._face = FaceScorer.from_config({"eval": self.ev})
        return pipe

    def _id_embeds(self, img):
        torch = _lazy_torch()
        emb, _, _ = self._face.embed(img)
        self.last_info = {"no_face": emb is None}
        if emb is None:
            ref = torch.zeros(1, 512)
        else:
            ref = torch.from_numpy(emb).float().reshape(1, 512)
        ref = ref.unsqueeze(0).unsqueeze(0)               # (1, 1, 1, 512) as in the diffusers docs recipe
        neg = torch.zeros_like(ref)
        return torch.cat([neg, ref]).to(dtype=_to_dtype(self.dtype_name), device=self.device)

    def _call(self, imgs, prompts, strength, gens, mask):
        # max_batch is 1 for this editor: the embedding batching semantics of ip_adapter_image_embeds are unverified
        # for B > 1 (docs only show one reference face), so seeds are looped.
        emb = self._id_embeds(imgs[0])
        return self.pipe(prompt=prompts, image=imgs, strength=float(strength), guidance_scale=self.guidance,
                         num_inference_steps=self.steps, generator=gens, ip_adapter_image_embeds=[emb],
                         output_type="pil").images

    def params(self):
        d = super().params()
        d.update({"adapter": self.model_id, "weight": self.weight_name, "ip_adapter_scale": self.scale,
                  "base": self.base_id, "scheduler": self.ed_cfg.get("scheduler", "dpmpp"), "lora": None})
        return d


EDITOR_KINDS: Dict[str, Callable] = {
    "sd15_img2img": SD15Img2Img,
    "sdxl_img2img": SDXLImg2Img,
    "ip2p": InstructPix2Pix,
    "sd_inpaint": SDInpaint,
    "ipadapter_faceid": IPAdapterFaceID,
}


def build_editor(name: str, cfg: dict, face_scorer=None) -> Editor:
    """Build an (unloaded) editor from ``cfg['eval']['editors'][name]``."""
    ev = cfg["eval"]
    ed = ev["editors"][name]
    cls = EDITOR_KINDS[ed["kind"]]
    if cls is IPAdapterFaceID:
        return cls(ed, ev, face_scorer=face_scorer)
    return cls(ed, ev)
