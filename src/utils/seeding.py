"""Deterministic seeding helpers. Never use Python hash() (randomized per process)."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def derive_seed(base: int, *parts) -> int:
    """sha256 of 'base|part|part...' reduced to 31 bits. Stable across processes/machines."""
    s = "|".join([str(int(base))] + [str(p) for p in parts])
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_generator(seed: int, device="cpu"):
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g
