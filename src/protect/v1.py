"""Adapter exposing the frozen v1 algorithm through the shared Protector interface."""
from __future__ import annotations

from typing import Protocol

import torch

from src.utils.budget import to_model_scale
from src.utils.seeding import seed_everything


class Protector(Protocol):
    name: str

    def protect(self, x: torch.Tensor, seed: int):  # -> (Tensor, dict)
        ...


class V1Protector:
    name = "v1"

    def __init__(self, cfg: dict, device: str, backend: str = "real"):
        from src.optimizer import PGDOptimizer

        b = cfg["budget"]
        self.eps = to_model_scale(b["eps_255"])
        self.step = to_model_scale(b["step_255"])
        if cfg.get("_method") == "v1" and cfg.get("v1_strict", True):
            if abs(self.eps - 8 / 255) > 1e-12 or abs(self.step - 1 / 255) > 1e-12:
                raise ValueError("method v1 requires eps 4/255 and step 0.5/255 (standard scale) = "
                                 "8/255 and 1/255 on [-1,1]; set v1_strict=false to override")
        self.device = device
        if b.get("random_start", "uniform_full") != "uniform_full":
            raise ValueError("v1 protector only supports random_start: uniform_full")
        if backend == "tiny":
            from src.utils.tiny_models import tiny_target_embedding, tiny_v1_loss

            loss_fn = tiny_v1_loss(device)
            target = tiny_target_embedding().to(device)
        else:
            from src.losses import DiffShieldLoss

            m = cfg["models"][cfg.get("model_key", "sd15_v1")]
            loss_fn = DiffShieldLoss(device=device, sd_model_id=m["id"], revision=m.get("revision"),
                                     clip_model_id=cfg["models"]["clip_sem"]["id"])
            target = loss_fn.encode_target_text([cfg["loss"]["sem_target_text"]])
        self.loss_fn = loss_fn
        self.target = target
        self.w = (float(cfg["loss"]["alpha"]), float(cfg["loss"]["beta"]), float(cfg["loss"]["gamma"]))
        self.opt = PGDOptimizer(epsilon=self.eps, alpha=self.step, iters=int(b["iters"]), device=device,
                                loss_fn=loss_fn)
        if backend == "tiny":  # tests only: EoT crop output size follows the (small) test image size
            from src.eot import EoTLayer

            self.opt.eot = EoTLayer(image_size=int(cfg["data"]["image_size"])).to(device)

    def protect(self, x, seed: int):
        seed_everything(seed)  # v1 uses the global RNG (delta.uniform_, kornia EoT params)
        x = x.to(self.device)
        x_adv = self.opt.optimize(x, self.target, w_alpha=self.w[0], w_beta=self.w[1], w_gamma=self.w[2])
        return x_adv, {"linf_model": float((x_adv - x).abs().max())}


def build_v1(cfg: dict, device: str, backend: str = "real") -> V1Protector:
    return V1Protector(cfg, device, backend)
