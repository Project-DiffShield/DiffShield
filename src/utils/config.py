"""Config loading: config/config.yaml deep-merged with config/parts/*.yaml (sorted), then an
optional method preset (cfg['methods'][method]) and `--set a.b.c=val` overrides."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

_MISSING = object()


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(cfg: dict, overrides: Sequence[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"override must look like a.b.c=value, got {ov!r}")
        key, val = ov.split("=", 1)
        parsed = yaml.safe_load(val)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into non-dict at {p!r} in {key!r}")
        node[parts[-1]] = parsed
    return cfg


def cfg_get(cfg: dict, dotted: str, default: Any = _MISSING):
    node = cfg
    for p in dotted.split("."):
        if isinstance(node, dict) and p in node:
            node = node[p]
        elif default is not _MISSING:
            return default
        else:
            raise KeyError(dotted)
    return node


def validate_config(cfg: dict) -> None:
    for k in ("seed", "paths", "budget", "loss", "models", "data", "runtime"):
        if k not in cfg:
            raise KeyError(f"config missing required section {k!r}")
    b = cfg["budget"]
    if b.get("scale") != "standard_0_255":
        raise ValueError("budget.scale must be 'standard_0_255'")
    if not (0 < float(b["eps_255"]) <= 16):
        raise ValueError(f"budget.eps_255 out of range (0,16]: {b['eps_255']}")
    if not (0 < float(b["step_255"]) <= float(b["eps_255"])):
        raise ValueError("budget.step_255 must be in (0, eps_255]")
    if int(b["iters"]) < 1:
        raise ValueError("budget.iters must be >= 1")
    if cfg.get("purifiers_train"):
        probs = cfg.get("purifier_train_probs", {})
        tot = sum(float(probs.get(p, 0.0)) for p in cfg["purifiers_train"])
        # allow non-normalised subsets (Phase 3 renormalises) but total must be positive
        if tot <= 0:
            raise ValueError("purifier_train_probs has no mass on purifiers_train")


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return d or {}


def load_config(path, method: str | None = None, overrides: Sequence[str] = (),
                parts_dir: str | Path | None = None) -> dict:
    path = Path(path)
    cfg = _load_yaml(path)
    pdir = Path(parts_dir) if parts_dir is not None else path.parent / "parts"
    if pdir.is_dir():
        for p in sorted(pdir.glob("*.yaml")):
            cfg = deep_merge(cfg, _load_yaml(p))
    if method is not None:
        presets = cfg.get("methods", {}) or {}
        if method == "v2_final" and method not in presets and "v2_final_default" in presets:
            method_key = "v2_final_default"  # tuned layer (config/parts/final.yaml) not present yet
        else:
            method_key = method
        if method_key not in presets:
            raise KeyError(f"method preset {method!r} not found in config (have: {sorted(presets)})")
        cfg = deep_merge(cfg, presets[method_key])
    cfg = apply_overrides(cfg, overrides)
    cfg["_method"] = method
    cfg["_overrides"] = list(overrides or ())
    validate_config(cfg)
    return cfg
