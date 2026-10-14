"""PGD with Expectation-over-Transformation over purifier draws (the v2 protection loop).

Per iteration, with x_pub = clip(x + delta) the PUBLISHED image:

    grad  =  d/d delta [ -gamma * L_str(x, x_pub) - lambda_id * cos_id(x_pub) ]                (once)
          +  (1/K) sum_k d/d delta [ alpha*L_vis + eta*L_den - beta*L_sem + antipure terms ](P_k(x_pub))

where P_k is drawn from the training mixture for every draw (own backward per draw => memory of one
draw). delta <- clip_[-eps, eps](delta + step * sign(grad)), then x + delta clipped to [-1, 1].

Budget: eps and step are given on the STANDARD 0-255 scale; the model-space value is
eps_255 * 2 / 255 (images live in [-1, 1]). With ``quantize=True`` the input is snapped to the uint8 grid
and the result is returned on the uint8 grid with |x_adv_u8 - x_u8| <= floor(eps_255), so that the
saved PNG respects the budget exactly (rounding cannot add a gray level).

Randomness: one ``torch.Generator`` seeded with ``seed`` (random start, purifier choice, JPEG quality,
timesteps, noise) => the same (image, seed) gives the same output on the same device/backend (GPU
kernels may still be nondeterministic, R15).

fp16: the UNet may run in fp16; the scalar loss is multiplied by ``loss_scale`` before autograd
to avoid gradient underflow (sign() is scale-invariant); non-finite per-draw gradients are skipped and
counted; the fraction of exactly-zero gradient pixels is logged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from .losses import TermScales, l_den, l_str, psnr_uint8, ssim
from .models import SDModels, make_generator
from .purifiers import PurifierMixture


@dataclass
class PGDParams:
    eps_255: float = 4.0
    step_255: float = 0.5
    iters: int = 40
    k_draws: int = 4
    random_start: str = "uniform_full"   # uniform_full | uniform_half | zero
    alpha: float = 0.27
    beta: float = 0.90
    gamma: float = 60.5
    eta: float = 0.27
    lambda_id: float = 0.0
    w_hf: float = 0.5
    w_mono: float = 0.5
    loss_scale: float = 1.0
    t_range: tuple = (50, 300)
    quantize: bool = True
    loss_norm: dict = field(default_factory=dict)


def params_from_cfg(cfg: dict) -> PGDParams:
    b, l, e = cfg.get("budget") or {}, cfg.get("loss") or {}, cfg.get("eot") or {}
    ap = l.get("anti_purify") or {}
    os_ = ((cfg.get("purifier_train_params") or {}).get("sd_onestep")) or {}
    d = PGDParams()
    return PGDParams(
        eps_255=float(b.get("eps_255", d.eps_255)), step_255=float(b.get("step_255", d.step_255)),
        iters=int(b.get("iters", d.iters)), k_draws=int(e.get("k_draws", d.k_draws)),
        random_start=b.get("random_start", d.random_start),
        alpha=float(l.get("alpha", d.alpha)), beta=float(l.get("beta", d.beta)),
        gamma=float(l.get("gamma", d.gamma)), eta=float(l.get("eta", d.eta)),
        lambda_id=float(l.get("lambda_id", 0.0)),
        w_hf=float(ap.get("w_hf", d.w_hf)), w_mono=float(ap.get("w_mono", d.w_mono)),
        loss_scale=float((cfg.get("runtime") or {}).get("loss_scale", d.loss_scale)),
        t_range=(int(os_.get("t_min", 50)), int(os_.get("t_max", 300))),
        quantize=bool((cfg.get("protect") or {}).get("quantize", True)),
        loss_norm=(dict(cfg.get("loss_norm") or {}) if l.get("normalize_terms") else {}),
    )


def to_u8(x: torch.Tensor) -> torch.Tensor:
    """[-1,1] float -> uint8-valued float tensor, same convention as the evaluation harness."""
    return ((x.float() + 1.0) * 127.5).round().clamp(0, 255)


def from_u8(u: torch.Tensor) -> torch.Tensor:
    return u * (2.0 / 255.0) - 1.0


def pgd_protect(x: torch.Tensor, models: SDModels, mixture: PurifierMixture, params: PGDParams, *,
                vis, sem=None, id_loss=None, antipure=None, seed: int = 0):
    """Protect one image batch ``x`` [B,3,H,W] in [-1,1] (B=1 expected). Returns (x_adv, info).

    vis        object with prepare(x)->state and __call__(xp, state)->scalar (SingleVis / EnsembleVis)
    sem        ClipSem (or any object with sem_loss(x)); only used if params.beta != 0
    id_loss    face_loss.IdentityLoss; only used if params.lambda_id != 0
    antipure   antipure.AntiPure; only used if given
    """
    t0 = time.perf_counter()
    dev = x.device
    p = params
    gen = make_generator(seed, dev.type)
    S = TermScales(p.loss_norm)
    eps, step = p.eps_255 * 2.0 / 255.0, p.step_255 * 2.0 / 255.0
    x0 = x.detach().float()
    if p.quantize:
        x0_u8 = to_u8(x0)
        x0 = from_u8(x0_u8)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    z_state = vis.prepare(x0) if p.alpha else None
    ap_state = antipure.prepare(x0) if antipure is not None else None
    if id_loss is not None and p.lambda_id:
        id_loss.prepare(x0)

    if p.random_start == "zero":
        delta = torch.zeros_like(x0)
    else:
        u = torch.rand(x0.shape, generator=gen, device=gen.device).to(dev) * 2 - 1
        delta = u * eps * (0.5 if p.random_start == "uniform_half" else 1.0)
    delta = (x0 + delta).clamp(-1, 1) - x0

    hist = {k: [] for k in ("vis", "den", "sem", "str", "id", "hf", "mono")}
    counts: dict = {}
    zero_fracs, nonfinite = [], 0

    for _ in range(p.iters):
        delta = delta.detach().requires_grad_(True)
        g_tot = torch.zeros_like(x0)
        rec = {k: [] for k in hist}

        def acc(loss):
            nonlocal nonfinite
            g = torch.autograd.grad(loss * p.loss_scale, delta)[0]
            if not torch.isfinite(g).all():
                nonfinite += 1
                return torch.zeros_like(g)
            return g / p.loss_scale

        # published-image terms (once per iteration)
        x_pub = (x0 + delta).clamp(-1, 1)
        v_str = l_str(x0, x_pub)
        rec["str"].append(float(v_str.detach()))
        g_tot = g_tot + acc(-p.gamma * S("str", v_str))
        if id_loss is not None and p.lambda_id:
            v_id = id_loss((x0 + delta).clamp(-1, 1))
            rec["id"].append(float(v_id.detach()))
            g_tot = g_tot + acc(-p.lambda_id * S("id", v_id))

        # EoT draws
        for _k in range(p.k_draws):
            x_pub = (x0 + delta).clamp(-1, 1)
            xp, aux, name = mixture.sample(x_pub, gen)
            counts[name] = counts.get(name, 0) + 1
            total = None

            def add(w, key, val):
                nonlocal total
                rec[key].append(float(val.detach()))
                term = w * S(key, val)
                total = term if total is None else total + term

            if p.alpha:
                add(p.alpha, "vis", vis(xp, z_state))
            if p.eta:
                add(p.eta, "den", l_den(models, xp, gen, aux, p.t_range))
            if p.beta and sem is not None:
                add(-p.beta, "sem", sem.sem_loss(xp))
            if antipure is not None and aux is not None:
                for key, val in antipure.terms(aux, x_pub, ap_state, gen).items():
                    add(p.w_hf if key == "hf" else p.w_mono, key, val)
            if total is not None and total.requires_grad:
                g_tot = g_tot + acc(total / p.k_draws)

        for k in hist:
            if rec[k]:
                hist[k].append(sum(rec[k]) / len(rec[k]))
        zero_fracs.append(float((g_tot == 0).float().mean()))
        with torch.no_grad():
            d = (delta + step * g_tot.sign()).clamp(-eps, eps)
            delta = (x0 + d).clamp(-1, 1) - x0

    delta = delta.detach()
    if p.quantize:
        E = float(int(p.eps_255 + 1e-6))
        d_u8 = (delta / (2.0 / 255.0)).round().clamp(-E, E)
        x_adv = from_u8((x0_u8 + d_u8).clamp(0, 255))
    else:
        x_adv = (x0 + delta).clamp(-1, 1)

    ref_u8 = to_u8(x if not p.quantize else x0)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    info = {
        "seed": int(seed), "eps_255": p.eps_255, "step_255": p.step_255, "iters": p.iters, "k_draws": p.k_draws,
        "linf_u8": float((to_u8(x_adv) - ref_u8).abs().max()),
        "psnr": psnr_uint8(x_adv, x0), "ssim": float(ssim(x0, from_u8(to_u8(x_adv)))),
        "terms": hist,
        "terms_init": {k: v[0] for k, v in hist.items() if v},
        "terms_final": {k: v[-1] for k, v in hist.items() if v},
        "purifier_counts": counts,
        "zero_grad_frac_mean": sum(zero_fracs) / max(1, len(zero_fracs)),
        "nonfinite_draws": nonfinite,
        "time_s": time.perf_counter() - t0,
        "peak_mem_mb": (torch.cuda.max_memory_allocated(dev) / 2 ** 20) if dev.type == "cuda" else None,
    }
    return x_adv, info
