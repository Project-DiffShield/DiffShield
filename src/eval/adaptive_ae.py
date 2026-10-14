"""E8 adaptive attacker: LightShed-style learned perturbation remover (small conv U-Net autoencoder).

The attacker KNOWS the protection method (adaptive mode, honest stress test): it protects its own pool of
images (the 120-image ``ae_pool`` split, disjoint from test / messy / train / val / calibration), then trains a
network ``f`` on (protected -> clean) pairs with 128 px random patches and applies ``f`` fully-convolutionally to
full 512 px protected images before editing. Budget ~1 GPU-hour (config ``adaptive_ae.train.minutes``).

This is a WEAK attacker by design (tiny network, 120 images, ~1 h): a drop in protection measured against it is a
lower bound on the damage an adaptive attacker can do. It is NOT the LightShed implementation (the paper's
architecture / training recipe was not available here; only the idea is reproduced).

Contract: ``AdaptiveAEPurifier.apply(img: PIL.Image, seed: int) -> PIL.Image`` (EvalPurifier), and
``build_purifier(cfg, device)`` for the A3 plug-in registry (``src/eval/purify_attacks.py``).
Only used in Half B (evaluation); never imported from ``src.protect``.
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # A3's base class when available (merged tree); duck-typed fallback otherwise
    from .purify_attacks import EvalPurifier as _Base, NotAvailable as _NotAvailable
except Exception:  # pragma: no cover - depends on the other agent's module being merged
    class _Base:  # type: ignore
        name = "base"
        needs_gpu = False
        share_key = None

        def load(self, device=None):
            return self

        def unload(self):
            return None

    class _NotAvailable(RuntimeError):  # type: ignore
        pass


# ---------------------------------------------------------------------------------------------- model


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, c) if c % 8 else 8, c)


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), _gn(cout), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class AdaptiveAE(nn.Module):
    """Residual U-Net: out = x + f(x) (identity at init because the last conv is zero-initialised).

    Fully convolutional. Input is reflect-padded to a multiple of 2**depth and cropped back, so any H, W work.
    """

    def __init__(self, base_channels: int = 32, depth: int = 3, in_channels: int = 3):
        super().__init__()
        self.depth = int(depth)
        ch = [base_channels * (2 ** i) for i in range(self.depth + 1)]
        self.inc = _Block(in_channels, ch[0])
        self.downs = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], ch[i + 1], 3, stride=2, padding=1),
                                                  _Block(ch[i + 1], ch[i + 1])) for i in range(self.depth)])
        self.ups = nn.ModuleList([_Block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(self.depth))])
        self.outc = nn.Conv2d(ch[0], in_channels, 3, padding=1)
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)
        self.arch = {"base_channels": int(base_channels), "depth": int(depth), "in_channels": int(in_channels)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        m = 2 ** self.depth
        ph, pw = (-h) % m, (-w) % m
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        feats = [self.inc(xp)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        y = feats[-1]
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="nearest")
            y = up(torch.cat([y, skip], 1))
        out = xp + self.outc(y)
        return out[..., :h, :w]


# ---------------------------------------------------------------------------------------------- data


def pil_to_u8(img) -> torch.Tensor:
    a = np.asarray(img.convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(a.copy()).permute(2, 0, 1).contiguous()  # CHW uint8


def u8_to_unit(u: torch.Tensor) -> torch.Tensor:
    return u.float() / 127.5 - 1.0


def list_pairs(protected_dir: str, clean_dir: str, exts=(".png", ".jpg", ".jpeg")) -> List[Tuple[str, str, str]]:
    """Match files by stem -> [(stem, protected_path, clean_path)] sorted by stem. Unmatched files are skipped."""
    def index(d):
        out = {}
        for f in sorted(os.listdir(d)):
            s, e = os.path.splitext(f)
            if e.lower() in exts:
                out[s] = os.path.join(d, f)
        return out
    p, c = index(protected_dir), index(clean_dir)
    return [(s, p[s], c[s]) for s in sorted(set(p) & set(c))]


def load_pairs(pairs: Sequence[Tuple[str, str, str]]) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
    from PIL import Image
    stems, prot, clean = [], [], []
    for s, pp, cp in pairs:
        a, b = pil_to_u8(Image.open(pp)), pil_to_u8(Image.open(cp))
        if a.shape != b.shape:
            raise ValueError(f"pair {s}: protected {tuple(a.shape)} != clean {tuple(b.shape)}")
        stems.append(s)
        prot.append(a)
        clean.append(b)
    return stems, prot, clean


def split_pairs(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """Deterministic train / val index split (at least 1 train item; val may be empty when n is tiny)."""
    idx = list(np.random.RandomState(seed).permutation(n))
    nv = int(round(n * val_fraction)) if n > 1 else 0
    nv = min(nv, n - 1)
    return sorted(idx[nv:]), sorted(idx[:nv])


class PatchSampler:
    """Random aligned patches (+ random flips / 90-degree rotations) from (protected, clean) uint8 CHW tensors."""

    def __init__(self, protected: Sequence[torch.Tensor], clean: Sequence[torch.Tensor], patch: int = 128, seed: int = 0):
        assert len(protected) == len(clean) and len(protected) > 0
        self.p, self.c, self.patch = list(protected), list(clean), int(patch)
        self.rng = np.random.RandomState(seed)

    def batch(self, bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(bs):
            i = int(self.rng.randint(len(self.p)))
            _, H, W = self.p[i].shape
            ps = min(self.patch, H, W)
            y0, x0 = int(self.rng.randint(H - ps + 1)), int(self.rng.randint(W - ps + 1))
            a = self.p[i][:, y0:y0 + ps, x0:x0 + ps]
            b = self.c[i][:, y0:y0 + ps, x0:x0 + ps]
            if self.rng.rand() < 0.5:
                a, b = a.flip(-1), b.flip(-1)
            k = int(self.rng.randint(4))
            if k:
                a, b = torch.rot90(a, k, (1, 2)), torch.rot90(b, k, (1, 2))
            xs.append(u8_to_unit(a))
            ys.append(u8_to_unit(b))
        return torch.stack(xs), torch.stack(ys)


# ---------------------------------------------------------------------------------------------- training


def _psnr_unit(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR on the uint8 grid between [-1,1] tensors."""
    qa = ((a.clamp(-1, 1) + 1) * 127.5).round()
    qb = ((b.clamp(-1, 1) + 1) * 127.5).round()
    mse = float(((qa - qb) ** 2).mean())
    return float("inf") if mse <= 0 else 10 * math.log10(255.0 ** 2 / mse)


@torch.no_grad()
def evaluate_ae(model: nn.Module, protected: Sequence[torch.Tensor], clean: Sequence[torch.Tensor], device,
                max_images: int = 8) -> Dict[str, float]:
    """Mean PSNR(protected, clean) vs PSNR(AE(protected), clean) on whole images (honest gain report)."""
    was = model.training
    model.eval()
    pin, pout = [], []
    for a, b in list(zip(protected, clean))[:max_images]:
        x = u8_to_unit(a).unsqueeze(0).to(device)
        y = u8_to_unit(b).unsqueeze(0).to(device)
        pin.append(_psnr_unit(x, y))
        pout.append(_psnr_unit(model(x), y))
    model.train(was)
    if not pin:
        return {}
    fin = [v for v in pin if math.isfinite(v)]
    fout = [v for v in pout if math.isfinite(v)]
    return {"n": len(pin), "psnr_protected_vs_clean": float(np.mean(fin)) if fin else float("inf"),
            "psnr_ae_vs_clean": float(np.mean(fout)) if fout else float("inf")}


def save_checkpoint(path: str, model: AdaptiveAE, opt, step: int, elapsed: float, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"state_dict": model.state_dict(), "arch": model.arch, "step": int(step), "elapsed_s": float(elapsed),
                "opt": opt.state_dict() if opt is not None else None, "meta": meta or {}}, tmp)
    os.replace(tmp, path)


def train_ae(model: AdaptiveAE, sampler: PatchSampler, device, steps: Optional[int] = None,
             minutes: Optional[float] = None, batch_size: int = 16, lr: float = 2e-4, lpips_weight: float = 0.0,
             log_every: int = 50, ckpt_every: int = 500, ckpt_path: Optional[str] = None,
             val: Optional[Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]] = None,
             resume_state: Optional[dict] = None, log: Callable[[str], None] = print, meta: Optional[dict] = None,
             seed: int = 0) -> Dict:
    """Train until ``steps`` or ``minutes`` (whichever hits first; at least one is required). L1 (+ optional LPIPS)."""
    if steps is None and minutes is None:
        raise ValueError("give steps and/or minutes")
    torch.manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    step0, elapsed0 = 0, 0.0
    if resume_state:
        model.load_state_dict(resume_state["state_dict"])
        if resume_state.get("opt"):
            opt.load_state_dict(resume_state["opt"])
        step0, elapsed0 = int(resume_state.get("step", 0)), float(resume_state.get("elapsed_s", 0.0))
    lp = None
    if lpips_weight > 0:
        import lpips  # optional dependency, lazy
        lp = lpips.LPIPS(net="vgg").to(device).eval()
        lp.requires_grad_(False)
    t0 = time.time()
    hist: List[Dict] = []
    step = step0
    while True:
        elapsed = elapsed0 + time.time() - t0
        if steps is not None and step >= steps:
            break
        if minutes is not None and elapsed >= minutes * 60:
            break
        x, y = sampler.batch(batch_size)
        x, y = x.to(device), y.to(device)
        out = model(x)
        l1 = F.l1_loss(out, y)
        loss = l1
        if lp is not None:
            loss = loss + lpips_weight * lp(out.clamp(-1, 1), y).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % log_every == 0 or step == step0 + 1:
            rec = {"step": step, "loss": float(loss.detach()), "l1": float(l1.detach()), "elapsed_s": elapsed}
            hist.append(rec)
            log(f"[ae] step {step} loss {rec['loss']:.5f} l1 {rec['l1']:.5f} t={elapsed:.0f}s")
        if ckpt_path and ckpt_every and step % ckpt_every == 0:
            save_checkpoint(ckpt_path, model, opt, step, elapsed, meta)
    elapsed = elapsed0 + time.time() - t0
    if ckpt_path:
        save_checkpoint(ckpt_path, model, opt, step, elapsed, meta)
    res = {"steps": step, "elapsed_s": elapsed, "history": hist}
    if val is not None and len(val[0]):
        res["val"] = evaluate_ae(model, val[0], val[1], device)
    return res


# ---------------------------------------------------------------------------------------------- purifier


def _pick_device(device) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(weights_path: str, device="cpu", base_channels: int = 32, depth: int = 3) -> AdaptiveAE:
    if not os.path.isfile(weights_path):
        raise _NotAvailable(f"adaptive AE weights not found: {weights_path} (train with scripts/train_adaptive_ae.py)")
    ck = torch.load(weights_path, map_location="cpu")
    arch = ck.get("arch", {}) if isinstance(ck, dict) else {}
    m = AdaptiveAE(arch.get("base_channels", base_channels), arch.get("depth", depth), arch.get("in_channels", 3))
    m.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    return m.to(device).eval()


class AdaptiveAEPurifier(_Base):
    """EvalPurifier: apply the trained AE fully-convolutionally to the (full-size) protected image."""

    name = "adaptive_ae"
    needs_gpu = False  # runs on CPU too (slowly); uses cuda when available

    def __init__(self, weights_path: str, device="auto", base_channels: int = 32, depth: int = 3,
                 model: Optional[AdaptiveAE] = None):
        self.weights_path = weights_path
        self._device_arg = device
        self.base_channels, self.depth = base_channels, depth
        self.model = model
        self.device = _pick_device(device)

    def load(self, device=None):
        if device is not None:
            self.device = _pick_device(device)
        if self.model is None:
            self.model = load_model(self.weights_path, self.device, self.base_channels, self.depth)
        else:
            self.model.to(self.device).eval()
        return self

    def unload(self):
        self.model = None

    def params(self) -> dict:
        return {"weights_path": self.weights_path, "base_channels": self.base_channels, "depth": self.depth,
                "deterministic": True}

    @torch.no_grad()
    def apply(self, img, seed: int = 0):
        from PIL import Image
        self.load()
        x = u8_to_unit(pil_to_u8(img)).unsqueeze(0).to(self.device)
        y = self.model(x).clamp(-1, 1)
        u = ((y[0] + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(u, "RGB")


def build_purifier(cfg: Optional[dict], device=None) -> AdaptiveAEPurifier:
    """A3 plug-in entry point. Reads ``cfg['adaptive_ae']``; ``cfg['eval']['purifiers']['adaptive_ae']`` overrides."""
    cfg = cfg or {}
    a = dict(cfg.get("adaptive_ae", {}) or {})
    ov = ((cfg.get("eval", {}) or {}).get("purifiers", {}) or {}).get("adaptive_ae", {}) or {}
    a.update(ov)
    wp = a.get("weights_path", "results/adaptive_ae/ae_final.pt")
    p = AdaptiveAEPurifier(wp, device if device is not None else a.get("device", "auto"),
                           a.get("base_channels", 32), a.get("depth", 3))
    if not os.path.isfile(wp):
        raise _NotAvailable(f"adaptive AE weights not found: {wp} (train with scripts/train_adaptive_ae.py)")
    return p
