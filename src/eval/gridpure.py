"""GrIDPure purifier (Zhao et al., CVPR 2024, "Can Protective Perturbation Safeguard Personal Data from Being
Exploited by Stable Diffusion?"). Half B only.

Source of this implementation: the official repository's README and ``gridpure.py`` as summarised from
https://github.com/ZhengyueZhao/GrIDPure (read through a web summariser, not cloned: details are UNVERIFIED
until compared with the cloned code in the J03b smoke job):

  * the purifier network is DiffPure on the *unconditional ImageNet 256x256 guided-diffusion* model
    (``256x256_diffusion_uncond.pt``, OpenAI blob storage, not on Hugging Face);
  * the image is tiled with 256x256 windows at stride 128 (plus a final window flush with the right/bottom
    edge if the tiling does not reach it); the four 128x128 corner patches are packed into one 256x256 canvas
    and purified once;
  * every window is purified (DiffPure noise-and-denoise with ``pure_steps`` diffusion steps; official example
    ``--pure_steps 10``), purified windows are written back and averaged where they overlap;
  * the whole thing is iterated ``pure_iter_num`` times (official example 20); each iteration blends
    ``img = (1 - gamma) * purified + gamma * input`` (official example ``--gamma 0.1``). Whether the blend
    partner is the current input or the original image is UNVERIFIED; the current input is used here.

The grid logic (tiling, corner packing, merging, iteration, blending) is implemented natively and unit-tested
with a fake patch purifier. The patch purifier needs the ``guided_diffusion`` package from the official repo on
``PYTHONPATH`` (config ``eval.purifiers.gridpure.repo_path``) and the checkpoint file
(``eval.purifiers.gridpure.checkpoint``); otherwise ``NotAvailable`` is raised with a clear message. The reverse
process here is plain DDPM ancestral sampling from timestep ``pure_steps`` (an approximation of DiffPure's
reverse VP-SDE solver).

Cost warning (estimate, not measured): 10 windows x 20 iterations x 10 steps = 2000 ADM-256 forward passes per
image. Reduce ``iters`` in config if the J03b smoke job shows it does not fit the budget.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .purify_attacks import EvalPurifier, NotAvailable

Box = Tuple[int, int, int, int]   # x0, y0, x1, y1 (exclusive)


def get_crop_boxes(h: int, w: int, window: int = 256, stride: int = 128) -> List[Box]:
    """Window tiling; the last row/column is flush with the image edge so every pixel is covered."""
    if h < window or w < window:
        raise ValueError(f"image {w}x{h} smaller than window {window}")

    def starts(n):
        s = list(range(0, n - window + 1, stride))
        if s[-1] != n - window:
            s.append(n - window)
        return s

    return [(x, y, x + window, y + window) for y in starts(h) for x in starts(w)]


def get_corner_boxes(h: int, w: int, half: int = 128) -> List[Box]:
    """Four ``half`` x ``half`` corner patches (TL, TR, BL, BR)."""
    return [(0, 0, half, half), (w - half, 0, w, half), (0, h - half, half, h), (w - half, h - half, w, h)]


def pack_corners(arr: np.ndarray, boxes: Sequence[Box]) -> np.ndarray:
    """Pack the 4 corner patches of ``arr`` (H, W, C) into one (2*half, 2*half, C) canvas, row-major."""
    half = boxes[0][2] - boxes[0][0]
    out = np.zeros((2 * half, 2 * half, arr.shape[2]), dtype=arr.dtype)
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        r, c = divmod(k, 2)
        out[r * half:(r + 1) * half, c * half:(c + 1) * half] = arr[y0:y1, x0:x1]
    return out


def unpack_corners(canvas: np.ndarray, boxes: Sequence[Box]) -> List[np.ndarray]:
    half = boxes[0][2] - boxes[0][0]
    return [canvas[(k // 2) * half:(k // 2 + 1) * half, (k % 2) * half:(k % 2 + 1) * half] for k in range(len(boxes))]


PatchFn = Callable[[np.ndarray], np.ndarray]   # (N, 256, 256, 3) float32 in [0, 1] -> same shape


def grid_pure_pass(arr: np.ndarray, patch_fn: PatchFn, window: int = 256, stride: int = 128) -> np.ndarray:
    """One GrIDPure pass: purify all windows + the packed-corner canvas, average overlaps. arr (H, W, 3) in [0, 1]."""
    h, w, _ = arr.shape
    boxes = get_crop_boxes(h, w, window, stride)
    cboxes = get_corner_boxes(h, w, window // 2)
    batch = [arr[y0:y1, x0:x1] for (x0, y0, x1, y1) in boxes] + [pack_corners(arr, cboxes)]
    out = patch_fn(np.stack(batch).astype(np.float32))
    acc = np.zeros_like(arr, dtype=np.float64)
    cnt = np.zeros((h, w, 1), dtype=np.float64)
    for (x0, y0, x1, y1), p in zip(boxes, out[:len(boxes)]):
        acc[y0:y1, x0:x1] += p
        cnt[y0:y1, x0:x1] += 1
    for (x0, y0, x1, y1), p in zip(cboxes, unpack_corners(out[-1], cboxes)):
        acc[y0:y1, x0:x1] += p
        cnt[y0:y1, x0:x1] += 1
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


def gridpure(arr: np.ndarray, patch_fn: PatchFn, iters: int = 20, gamma: float = 0.1, window: int = 256,
             stride: int = 128) -> np.ndarray:
    """Iterated grid purification with blending toward the current input. arr (H, W, 3) float in [0, 1]."""
    cur = arr.astype(np.float32)
    for _ in range(iters):
        pur = grid_pure_pass(cur, patch_fn, window, stride)
        cur = ((1.0 - gamma) * pur + gamma * cur).astype(np.float32)
    return np.clip(cur, 0.0, 1.0)


class GuidedDiffusionPatchFn:
    """DiffPure-style noise-and-denoise with the OpenAI ImageNet-256 unconditional ADM model (UNVERIFIED API use)."""

    def __init__(self, checkpoint: str, repo_path: Optional[str], pure_steps: int, device: str, fp16: bool = True,
                 seed: int = 0):
        if not checkpoint or not os.path.isfile(checkpoint):
            raise NotAvailable(
                "GrIDPure checkpoint missing: download 256x256_diffusion_uncond.pt "
                "(https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) "
                "and set eval.purifiers.gridpure.checkpoint")
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        except Exception as e:  # pragma: no cover
            raise NotAvailable(f"guided_diffusion package not importable ({e}); clone "
                               "github.com/ZhengyueZhao/GrIDPure and set eval.purifiers.gridpure.repo_path")
        import torch
        args = model_and_diffusion_defaults()
        args.update(dict(attention_resolutions="32,16,8", class_cond=False, diffusion_steps=1000, image_size=256,
                         learn_sigma=True, noise_schedule="linear", num_channels=256, num_head_channels=64,
                         num_res_blocks=2, resblock_updown=True, use_fp16=fp16 and device.startswith("cuda"),
                         use_scale_shift_norm=True, timestep_respacing=""))
        model, diffusion = create_model_and_diffusion(**args)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.requires_grad_(False).eval().to(device)
        if args["use_fp16"]:
            model.convert_to_fp16()
        self.model, self.diffusion, self.device, self.t = model, diffusion, device, int(pure_steps)
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def reseed(self, seed: int):
        self.gen.manual_seed(int(seed))

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        import torch
        x0 = torch.from_numpy(patches).permute(0, 3, 1, 2).to(self.device) * 2 - 1
        b = x0.shape[0]
        t = torch.full((b,), self.t - 1, device=self.device, dtype=torch.long)
        noise = torch.randn(x0.shape, generator=self.gen, device=self.device)
        x = self.diffusion.q_sample(x0, t, noise=noise)
        with torch.no_grad():
            for i in reversed(range(self.t)):
                tt = torch.full((b,), i, device=self.device, dtype=torch.long)
                x = self.diffusion.p_sample(self.model, x, tt, clip_denoised=True)["sample"]
        return ((x.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).float().cpu().numpy()

    def unload(self):
        self.model = None


class GridPurePurifier(EvalPurifier):
    name = "gridpure"
    needs_gpu = True

    def __init__(self, params: dict, device: Optional[str] = None, patch_fn: Optional[PatchFn] = None):
        self.params = dict(params or {})
        self.device = device
        self._patch_fn = patch_fn      # injectable for tests

    def load(self, device: Optional[str] = None):
        if self._patch_fn is not None:
            return self
        dev = device or self.device
        if dev is None:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        p = self.params
        self._patch_fn = GuidedDiffusionPatchFn(p.get("checkpoint"), p.get("repo_path"), p.get("pure_steps", 10), dev)
        return self

    def apply(self, img, seed: int):
        from PIL import Image
        self.load()
        if hasattr(self._patch_fn, "reseed"):
            self._patch_fn.reseed(seed)
        p = self.params
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        out = gridpure(arr, self._patch_fn, iters=int(p.get("iters", 20)), gamma=float(p.get("gamma", 0.1)),
                       window=int(p.get("window", 256)), stride=int(p.get("stride", 128)))
        return Image.fromarray(np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8), mode="RGB")

    def unload(self):
        if hasattr(self._patch_fn, "unload"):
            self._patch_fn.unload()
        self._patch_fn = None
