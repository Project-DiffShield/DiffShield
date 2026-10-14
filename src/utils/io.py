"""Image / file helpers."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


def list_images(input_dir, list_file=None, exts=(".png", ".jpg", ".jpeg")):
    """Images in input_dir sorted by stem. If list_file is given, only the names listed there
    (names with or without extension), in sorted order; missing files raise FileNotFoundError."""
    input_dir = Path(input_dir)
    if list_file is not None:
        names = [l.strip() for l in Path(list_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        out = []
        for n in names:
            p = input_dir / n
            if not p.exists():
                cands = [input_dir / (Path(n).stem + e) for e in exts]
                cands = [c for c in cands if c.exists()]
                if not cands:
                    raise FileNotFoundError(f"{n} (from {list_file}) not found in {input_dir}")
                p = cands[0]
            out.append(p)
    else:
        out = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(out, key=lambda p: (p.stem, p.name))


def shard_items(items, shard):
    """shard=(k, n) -> items[k::n]; None -> all."""
    if shard is None:
        return list(items)
    k, n = shard
    if not (0 <= k < n):
        raise ValueError(f"bad shard {shard}")
    return list(items)[k::n]


def parse_shard(s):
    if s is None or s == "":
        return None
    k, n = str(s).split("/")
    return (int(k), int(n))


def load_tensor_v1(path, size=512):
    """Exactly src.data's transform: Resize((size,size)) -> ToTensor -> Normalize(.5,.5). [1,3,H,W]."""
    import torchvision.transforms as T

    tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize([0.5], [0.5])])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)


def save_png_v1(x, path) -> None:
    """torchvision.utils.save_image((x+1)/2) (rounds to uint8), written atomically."""
    from torchvision.utils import save_image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    save_image(((x.detach().cpu() + 1.0) / 2.0).squeeze(0), str(tmp))
    os.replace(tmp, path)


def load_u8(path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))
