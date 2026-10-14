import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import noise_control as nc  # noqa: E402
from src.baselines.mist import MistProtector, make_texture_target  # noqa: E402
from src.baselines.photoguard import EncoderAttackProtector, project, step_size_at  # noqa: E402


class TinyEnc(nn.Module):
    """Stand-in for the SD VAE encoder: 8x down, 4 latent channels, non-linear."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c1 = nn.Conv2d(3, 8, 4, stride=4)
        self.c2 = nn.Conv2d(8, 4, 2, stride=2)

    def forward(self, x):
        return self.c2(torch.tanh(self.c1(x)))


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.c1 = nn.Conv2d(5, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 4, 3, padding=1)

    def forward(self, zt, t, cond):
        tt = (t.float() / 1000.0).view(-1, 1, 1, 1).expand(-1, 1, *zt.shape[-2:])
        return self.c2(torch.relu(self.c1(torch.cat([zt, tt], 1))))


def make_x(seed=0, h=64, w=64):
    g = torch.Generator().manual_seed(seed)
    u = torch.randint(30, 220, (1, 3, h, w), generator=g).float()
    return nc.from_u8(u)


def test_helpers():
    assert abs(step_size_at(1.0, 0, 10) - 1.0) < 1e-9
    assert step_size_at(1.0, 9, 10) < 0.2
    assert step_size_at(1.0, 5, 10, decay=False) == 1.0
    ref = torch.zeros(1, 3, 4, 4)
    p = project(torch.full_like(ref, 5.0), ref, 0.1)
    assert torch.allclose(p, torch.full_like(ref, 0.1))


@pytest.mark.parametrize("target", ["gray", "zero_latent"])
def test_photoguard_budget_gradient_and_progress(target):
    enc = TinyEnc()
    x = make_x()
    pg = EncoderAttackProtector(enc, eps_255=4, step_255=1.0, iters=25, target=target)
    zt = pg.target_latent(x)
    y, info = pg.protect(x, seed=11)
    # budget: exact in uint8 terms, on the uint8 grid
    d = nc.to_u8(y) - nc.to_u8(x)
    assert 0 < d.abs().max().item() <= 4
    assert torch.allclose(y, nc.from_u8(nc.to_u8(y)))
    assert info["linf_u8"] <= 4 and info["iters"] == 25
    # progress: closer to the target latent than clean and than same-budget uniform noise
    with torch.no_grad():
        noise = nc.uniform_linf(x, 4, 1)
        dist_y = (enc(y) - zt).norm().item()
        dist_x = (enc(x) - zt).norm().item()
        dist_n = (enc(noise) - zt).norm().item()
    assert dist_y < dist_x
    assert dist_y < dist_n
    assert abs(info["loss_published"] - dist_y) < 1e-3 * max(1.0, dist_y)
    # determinism
    y2, _ = pg.protect(x, seed=11)
    assert torch.equal(y, y2)
    y3, _ = pg.protect(x, seed=12)
    assert not torch.equal(y, y3)


def test_photoguard_gradient_flows():
    enc = TinyEnc()
    x = make_x().requires_grad_(True)
    loss = enc(x).pow(2).sum()
    g, = torch.autograd.grad(loss, x)
    assert g.abs().sum() > 0


def test_photoguard_image_target_and_validation():
    enc = TinyEnc()
    with pytest.raises(ValueError):
        EncoderAttackProtector(enc, target="image")
    with pytest.raises(ValueError):
        EncoderAttackProtector(enc, target="bogus")
    x = make_x()
    ti = make_texture_target(32, 32, 1)
    pg = EncoderAttackProtector(enc, iters=3, target="image", target_image=ti)
    y, info = pg.protect(x, 0)
    assert y.shape == x.shape and info["linf_u8"] <= 4


def test_texture_target():
    t = make_texture_target(48, 64, seed=0)
    assert t.shape == (1, 3, 48, 64) and t.min() >= -1 and t.max() <= 1 and t.std() > 0.3
    assert torch.equal(t, make_texture_target(48, 64, seed=0))


def make_mist(**kw):
    enc, un = TinyEnc(), TinyUNet()
    ab = torch.cumprod(1 - torch.linspace(1e-4, 2e-2, 1000), 0)
    for p in list(enc.parameters()) + list(un.parameters()):
        p.requires_grad_(False)
    args = dict(eps_255=4, step_255=1.0, iters=30, t_min=1, t_max=1000)
    args.update(kw)
    return MistProtector(enc, un, ab, torch.zeros(1, 77, 8), **args)


def test_mist_budget_progress_determinism():
    x = make_x(2)
    m = make_mist()
    y, info = m.protect(x, seed=5)
    d = nc.to_u8(y) - nc.to_u8(x)
    assert 0 < d.abs().max().item() <= 4
    assert torch.allclose(y, nc.from_u8(nc.to_u8(y)))
    assert info["l_tex_last"] < info["l_tex_first"]          # textural term is being minimised
    assert info["implementation"].startswith("reimplementation")
    y2, _ = make_mist().protect(x, seed=5)
    assert torch.equal(y, y2)


def test_mist_semantic_term_gradient_flows():
    m = make_mist()
    x = make_x(3).requires_grad_(True)
    g = torch.Generator().manual_seed(0)
    z = m.encode_fn(x)
    l_sem = m.semantic_loss(z, g)
    grad, = torch.autograd.grad(l_sem, x)
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_mist_ascent_on_semantic_only_raises_loss():
    x = make_x(4)
    m = make_mist(w_tex=0.0, iters=40, n_noise=4)
    y, _ = m.protect(x, seed=1)
    with torch.no_grad():
        before = m.semantic_loss(m.encode_fn(x), torch.Generator().manual_seed(123)).item()
        after = m.semantic_loss(m.encode_fn(y), torch.Generator().manual_seed(123)).item()
    assert after > before
