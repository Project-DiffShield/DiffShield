"""Keep-face inpainting masks (Half B only).

Convention (diffusers inpainting): mask value 255 = REPAINT (edit this), 0 = KEEP. The threat model of the
SD-inpaint editor is an attacker who masks the background / clothes and never the face (if the face were
masked its pixels would be discarded and no perturbation could protect it, HANDOFF risk table).

Two sources:
  * CelebAMask-HQ parsing annotations (``<anno_dir>/<idx//2000>/<idx:05d>_<part>.png``): KEEP = face parts (skin,
    brows, eyes, nose, mouth, lips, ears) + glasses/earring/neck, dilated by ``dilate_px``; repaint = everything else
    (background, cloth, hat, hair). The keep list is configurable (``eval.masks.keep_parts``).
  * Face-bbox ellipse fallback (messy photos / non-CelebA ids): keep an ellipse around the face bbox.

``ensure_face_unmasked`` is applied on top of both as a hard guarantee.
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# CelebAMask-HQ part names (19 classes in the annotation, without 'background').
PARTS = ("skin", "nose", "eye_g", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip",
         "l_lip", "hair", "hat", "ear_r", "neck_l", "neck", "cloth")
FACE_PARTS = ("skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip")
REPAINT_PARTS = ("cloth",)


def _ellipse(shape: Tuple[int, int], cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ((xx - cx) / max(rx, 1e-6)) ** 2 + ((yy - cy) / max(ry, 1e-6)) ** 2 <= 1.0


def face_ellipse(shape_hw: Tuple[int, int], bbox: Sequence[float], scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Boolean (H, W) ellipse inscribed in the bbox, optionally scaled about its centre."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return _ellipse(shape_hw, (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2 * scale_x, (y2 - y1) / 2 * scale_y)


def repaint_mask_from_bbox(size: Tuple[int, int], bbox: Sequence[float], keep_scale_x: float = 1.45,
                           keep_scale_y: float = 1.75, up_shift: float = 0.12) -> np.ndarray:
    """uint8 (H, W) mask, 255 = repaint, 0 = keep. ``size`` is (width, height).

    Keeps a generous ellipse (covers forehead/hair/chin) around the detector bbox, shifted up a little because
    detector boxes usually start below the hairline.
    """
    w, h = size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    dy = up_shift * (y2 - y1)
    keep = face_ellipse((h, w), (x1, y1 - dy, x2, y2 - dy), keep_scale_x, keep_scale_y)
    keep |= face_ellipse((h, w), bbox)   # never lose the true face ellipse
    return np.where(keep, 0, 255).astype(np.uint8)


def ensure_face_unmasked(mask: np.ndarray, bbox: Optional[Sequence[float]], scale: float = 1.0) -> np.ndarray:
    """Force the face-bbox ellipse (times ``scale``) to KEEP (0). Returns a new array."""
    out = mask.copy()
    if bbox is not None:
        out[face_ellipse(mask.shape[:2], bbox, scale, scale)] = 0
    return out


def parsing_masks_dir_index(image_id: str) -> Optional[int]:
    """CelebAMask-HQ images are named ``<idx>.jpg`` (0..29999); returns idx or None for other ids."""
    stem = os.path.splitext(os.path.basename(str(image_id)))[0]
    return int(stem) if stem.isdigit() else None


# Default KEEP set (never repainted): all face parts, glasses, earring, neck. Everything else is repainted:
# unlabeled background, cloth, hat, hair (coordinator decision; configurable via eval.masks.keep_parts).
KEEP_PARTS_DEFAULT = FACE_PARTS + ("eye_g", "ear_r", "neck", "neck_l")


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    from PIL import Image, ImageFilter
    im = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MaxFilter(2 * int(px) + 1))
    return np.asarray(im) > 0


def repaint_mask_from_parsing(anno_dir: str, index: int, size: int = 512,
                              keep_parts: Iterable[str] = KEEP_PARTS_DEFAULT, dilate_px: int = 0) -> Optional[np.ndarray]:
    """uint8 mask (255 = repaint) from CelebAMask-HQ per-part PNGs (``<anno_dir>/<idx//2000>/<idx:05d>_<part>.png``).

    KEEP = union of ``keep_parts`` dilated by ``dilate_px`` pixels; everything else (unlabeled background, cloth, hat,
    hair, ... any part not in keep_parts) is repainted. None if the annotation folder or all part files are missing."""
    from PIL import Image
    sub = os.path.join(anno_dir, str(index // 2000))
    if not os.path.isdir(sub):
        return None
    keep_parts = set(keep_parts)
    keep = np.zeros((size, size), dtype=bool)
    found = False
    for p in PARTS:
        f = os.path.join(sub, f"{index:05d}_{p}.png")
        if not os.path.isfile(f):
            continue
        found = True
        if p in keep_parts:
            keep |= np.asarray(Image.open(f).convert("L").resize((size, size), Image.NEAREST)) > 0
    if not found:
        return None
    keep = _dilate(keep, dilate_px)
    return np.where(keep, 0, 255).astype(np.uint8)


class MaskProvider:
    """image_id -> PIL 'L' mask (255 = repaint). Cached on disk so clean and protected edits share one mask.

    ``face_bbox_fn(image_id) -> bbox | None`` supplies the bbox for the fallback and the safety clamp
    (the runner passes the cached source-face bbox).
    """

    def __init__(self, cache_dir: Optional[str] = None, parsing_dir: Optional[str] = None, size: int = 512,
                 face_bbox_fn=None, keep_parts: Iterable[str] = KEEP_PARTS_DEFAULT, dilate_px: int = 8):
        self.keep_parts, self.dilate_px = tuple(keep_parts), int(dilate_px)
        self.cache_dir = cache_dir
        self.parsing_dir = parsing_dir
        self.size = size
        self.face_bbox_fn = face_bbox_fn
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __call__(self, image_id: str):
        from PIL import Image
        cpath = os.path.join(self.cache_dir, f"{image_id}.png") if self.cache_dir else None
        if cpath and os.path.isfile(cpath):
            return Image.open(cpath).convert("L")
        bbox = self.face_bbox_fn(image_id) if self.face_bbox_fn else None
        arr = None
        idx = parsing_masks_dir_index(image_id)
        if self.parsing_dir and idx is not None:
            arr = repaint_mask_from_parsing(self.parsing_dir, idx, self.size, self.keep_parts, self.dilate_px)
        if arr is None:
            if bbox is None:
                raise RuntimeError(f"no parsing mask and no face bbox for {image_id}: cannot build keep-face mask")
            arr = repaint_mask_from_bbox((self.size, self.size), bbox)
        arr = ensure_face_unmasked(arr, bbox, 0.9)
        m = Image.fromarray(arr, mode="L")
        if cpath:
            m.save(cpath)
        return m
