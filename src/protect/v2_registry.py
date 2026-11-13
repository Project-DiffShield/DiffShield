"""v2 protector factory (called lazily by src/protect/registry.py, A1):

    protector = build_v2_protector(name, cfg, device)       # no model loading yet
    x_adv, info = protector.protect(x, seed)                # x: [1,3,H,W] in [-1,1]

Presets (names): v2_cheap, v2_main, v2_antipure, v2_final, abl_vis_only, abl_den_only, abl_no_sem,
abl_no_pur, eps2, eps6, eps8 (defined in ``config/parts/protect.yaml`` under ``methods:``).

``v2_final`` and ``eps*`` resolve as layers, lowest to highest priority:
    methods.v2_final_default  (hand-set start, protect.yaml)
    methods.v2_final          (tuned values from config/parts/final.yaml, optional; B3)
    methods.<name>            (eps2/eps6/eps8: budget only)
because final.yaml sorts before protect.yaml in load_config's sorted deep merge.

``cfg`` may be the full merged config dict or already method-merged; missing pieces (``methods``
presets, protect.yaml top-level keys) are filled from config/parts/protect.yaml next to the repo.
Models load on first ``protect`` call (cached process-wide in ``models._CACHE``). ``components`` may
inject ready objects (tests, shared weights): keys ``models`` (SDModels), ``sem`` (ClipSem-like),
``extra_vaes`` (dict name->vae), ``face`` (embedder with .embed).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch

from .pgd import params_from_cfg, pgd_protect

PRESETS = ("v2_cheap", "v2_main", "v2_antipure", "v2_final", "abl_vis_only", "abl_den_only",
           "abl_no_sem", "abl_no_pur", "eps2", "eps6", "eps8", "v2_tgt_zero", "v2_tgt_zero_mid", "v2_tgt_zero_hi", "v2_tgt_gray", "v2_next",
           "v2_x_eps8", "v2_x_it100", "v2_x_id_direct_a", "v2_x_id_direct_b", "v2_x_id_recon")
_CHAIN = {
    "v2_final": ["v2_final_default", "v2_final"],
    "v2_next_tuned": ["v2_next", "v2_next_tuned"],   # methods.v2_next_tuned = tune_optuna final_candidate.yaml
    **{k: ["v2_next", k] for k in ("v2_x_eps8", "v2_x_it100", "v2_x_id_direct_a", "v2_x_id_direct_b", "v2_x_id_recon")},
    "eps2": ["v2_final_default", "v2_final", "eps2"],
    "eps6": ["v2_final_default", "v2_final", "eps6"],
    "eps8": ["v2_final_default", "v2_final", "eps8"],
}
_PART = Path(__file__).resolve().parents[2] / "config" / "parts" / "protect.yaml"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _as_dict(cfg) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    for attr in ("to_dict", "as_dict", "_asdict"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)()
    if hasattr(cfg, "items"):
        return dict(cfg.items())
    return dict(vars(cfg))


def _part_file() -> dict:
    try:
        import yaml

        with open(_PART, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


ONESTEP_BASE_PROBS = {"jpeg": 0.35, "downup": 0.35, "vae_roundtrip": 0.30}


def probs_from_p_onestep(p_onestep: float) -> dict:
    """Training-mixture probabilities from the single knob ``p_onestep`` (v2_next):
    jpeg : downup : vae_roundtrip = 0.35 : 0.35 : 0.30 scaled to 1 - p_onestep, sd_onestep = p_onestep.
    p_onestep = 0 gives exactly {jpeg 0.35, downup 0.35, vae_roundtrip 0.3, sd_onestep 0.0}
    (= v2_tgt_zero_mid's mixture; sd_onestep then is neither drawn nor built)."""
    p = float(p_onestep)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p_onestep must be in [0, 1], got {p_onestep!r}")
    out = {k: v * (1.0 - p) for k, v in ONESTEP_BASE_PROBS.items()}
    out["sd_onestep"] = p
    return out


def resolve_cfg(name: str, cfg) -> dict:
    """Merged config dict for preset ``name`` (see module doc)."""
    cfg = _as_dict(cfg)
    part = _part_file()
    methods = deep_merge(part.get("methods") or {}, cfg.get("methods") or {})
    base = deep_merge({k: v for k, v in part.items() if k != "methods"},
                      {k: v for k, v in cfg.items() if k != "methods"})
    chain = _CHAIN.get(name, [name])
    if chain[0] not in methods:
        raise KeyError(f"unknown v2 preset {name!r}; known: {', '.join(PRESETS)}")
    out = base
    for layer in chain:
        if layer in methods:
            out = deep_merge(out, methods[layer])
    if out.get("p_onestep") is not None:       # v2_next: one knob drives the mixture (overrides purifier_train_probs)
        out["purifier_train_probs"] = probs_from_p_onestep(out["p_onestep"])
    cap = int((out.get("budget") or {}).get("iters_cap") or 0)   # smoke only (--set budget.iters_cap=N): presets that fix budget.iters (v2_x_it100) would otherwise ignore --set budget.iters
    if cap and int((out.get("budget") or {}).get("iters", cap)) > cap:
        out["budget"]["iters"] = cap
    out["method_name"] = name
    return out


class V2Protector:
    def __init__(self, name: str, cfg: dict, device="cuda", components: Optional[dict] = None):
        self.name = name
        self.cfg = resolve_cfg(name, cfg)
        self.device = torch.device(device)
        self.params = params_from_cfg(self.cfg)
        self.components = dict(components or {})
        self._built = False

    # -- lazy build ---------------------------------------------------------------------------
    def _build(self):
        from . import models as M
        from .antipure import AntiPure
        from .ensemble import EnsembleVis
        from .losses import SingleVis
        from .purifiers import SDOneStepPurifier, build_mixture

        c, cfg, p = self.components, self.cfg, self.params
        self.models = c.get("models") or M.get_sd15(cfg, self.device)
        self.mixture = build_mixture(cfg, self.models)

        ens = (cfg.get("protect") or {}).get("ensemble") or {}
        if p.alpha and ens.get("enabled"):
            extra = c.get("extra_vaes")
            if extra is None:
                extra = {k: M.get_sdxl_vae({"models": cfg.get("models"), "runtime": cfg.get("runtime")}, self.device)
                         for k in ens.get("vaes", ["sdxl_vae_fp16"])}
            self.vis = EnsembleVis(self.models, extra, ens.get("weights") or None)
        elif (cfg.get("loss") or {}).get("target"):
            from .losses import TargetVis

            t = cfg["loss"]["target"]
            self.vis = TargetVis(self.models, t.get("kind", "zero"), t.get("w_tgt", 1.0), t.get("w_dev", 0.0))
        else:
            self.vis = SingleVis(self.models)

        self.sem = None
        if p.beta:
            self.sem = c.get("sem") or M.get_clip_sem(cfg, self.device)

        self.id_loss = None
        if p.lambda_id:
            from .face_loss import IdentityLoss, load_face_embedder

            emb = c.get("face") or load_face_embedder(cfg, self.device)
            m = self.models
            face = cfg.get("face") or {}
            self.id_loss = IdentityLoss(emb, lambda z: m.decode(m.encode_mean(z, scaled=False), scaled=False),
                                        face.get("bbox"), mode=face.get("id_mode", "recon"))

        self.antipure = None
        ap = (cfg.get("loss") or {}).get("anti_purify") or {}
        if ap.get("enabled"):
            os_ = self.mixture.purifiers.get("sd_onestep")
            if os_ is None:  # anti-purify terms need the one-step surrogate even if not in the mixture
                from .purifiers import build_purifier

                os_ = build_purifier("sd_onestep", cfg, self.models)
            self.antipure = AntiPure(os_, p.w_hf, p.w_mono)
        self._built = True

    # -- API ----------------------------------------------------------------------------------
    def protect(self, x: torch.Tensor, seed: int, face_bbox=None):
        if not self._built:
            self._build()
        if self.id_loss is not None and face_bbox is not None:
            self.id_loss.bbox = face_bbox
        x = x.to(self.device)
        x_adv, info = pgd_protect(x, self.models, self.mixture, self.params, vis=self.vis, sem=self.sem,
                                  id_loss=self.id_loss, antipure=self.antipure, seed=seed)
        info["method"] = self.name
        return x_adv, info

    def unload(self):
        from . import models as M

        M._CACHE.clear()
        self._built = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_v2_protector(name: str, cfg, device="cuda", components: Optional[dict] = None) -> V2Protector:
    return V2Protector(name, cfg, device, components)
