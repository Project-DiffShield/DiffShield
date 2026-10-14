"""Method name -> Protector. Heavy imports are lazy so each owner's code loads only when needed.

Contract: Protector.protect(x [1,3,H,W] in [-1,1], seed) -> (Tensor, dict)
"""
from __future__ import annotations


def _is_v2(name: str) -> bool:
    return name.startswith(("v2_", "abl_", "eps"))


def _is_baseline(name: str) -> bool:
    return name.startswith("noise_") or name in ("photoguard", "mist")


def build_protector(name: str, cfg: dict, device: str, backend: str = None):
    backend = backend or cfg.get("runtime", {}).get("backend", "real")
    cfg.setdefault("runtime", {})["backend"] = backend
    if name == "v1":
        from src.protect.v1 import build_v1

        return build_v1(cfg, device, backend)
    if _is_v2(name):
        from src.protect.v2_registry import build_v2_protector

        return build_v2_protector(name, cfg, device)
    if _is_baseline(name):
        from src.baselines.registry import build_baseline

        return build_baseline(name, cfg, device)
    raise KeyError(f"unknown method {name!r}")
