import os
import sys

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import adaptive_ae as AE  # noqa: E402


def synth_pairs(n=4, size=64, amp=4, seed=0):
    """Smooth 'clean' images and 'protected' = clean + sign noise of +/-amp gray levels (uint8 CHW)."""
    g = np.random.RandomState(seed)
    clean, prot = [], []
    for _ in range(n):
        yy, xx = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij")
        base = np.stack([128 + 80 * np.sin(6 * xx + g.rand() * 3), 128 + 80 * np.cos(5 * yy + g.rand() * 3),
                         128 + 60 * np.sin(4 * (xx + yy))], 0)
        c = np.clip(base, 0, 255).round().astype(np.uint8)
        noise = g.choice([-amp, amp], size=c.shape)
        p = np.clip(c.astype(int) + noise, 0, 255).astype(np.uint8)
        clean.append(torch.from_numpy(c))
        prot.append(torch.from_numpy(p))
    return prot, clean


def test_forward_shape_fully_convolutional_and_identity_init():
    m = AE.AdaptiveAE(base_channels=8, depth=3)
    for shape in [(1, 3, 128, 128), (2, 3, 50, 70), (1, 3, 512, 512)]:
        x = torch.randn(*shape).clamp(-1, 1)
        with torch.no_grad():
            y = m(x)
        assert y.shape == x.shape
        assert torch.allclose(y, x, atol=1e-6)  # last conv zero-initialised -> identity


def test_patch_sampler_shapes_and_alignment():
    prot, clean = synth_pairs(3, 64)
    s = AE.PatchSampler(prot, clean, patch=32, seed=1)
    x, y = s.batch(5)
    assert x.shape == y.shape == (5, 3, 32, 32)
    assert x.min() >= -1 and x.max() <= 1
    # aligned: protected - clean is at most 2*amp gray levels (~ 8/127.5) everywhere
    assert (x - y).abs().max() <= 8 / 127.5 + 1e-6
    s2 = AE.PatchSampler(prot, clean, patch=32, seed=1)
    x2, _ = s2.batch(5)
    assert torch.equal(x, x2)


def test_three_step_training_smoke_on_random_tensors():
    g = torch.Generator().manual_seed(0)
    clean = [torch.randint(0, 256, (3, 64, 64), generator=g, dtype=torch.uint8) for _ in range(3)]
    prot = [(c.int() + torch.randint(-4, 5, c.shape, generator=g)).clamp(0, 255).to(torch.uint8) for c in clean]
    m = AE.AdaptiveAE(8, 2)
    s = AE.PatchSampler(prot, clean, patch=32, seed=0)
    res = AE.train_ae(m, s, "cpu", steps=3, batch_size=2, lr=1e-3, log_every=1, log=lambda *_: None)
    assert res["steps"] == 3 and len(res["history"]) >= 3
    assert all(np.isfinite(h["loss"]) for h in res["history"])
    # weights actually moved (last conv was zero)
    assert m.outc.weight.abs().sum() > 0


def test_training_reduces_error_and_checkpoint_roundtrip(tmp_path):
    prot, clean = synth_pairs(6, 64, amp=4)
    m = AE.AdaptiveAE(8, 2)
    s = AE.PatchSampler(prot, clean, patch=32, seed=0)
    x, y = s.batch(8)
    with torch.no_grad():
        before = (m(x) - y).abs().mean().item()
    ck = str(tmp_path / "ae.pt")
    res = AE.train_ae(m, s, "cpu", steps=30, batch_size=4, lr=3e-3, ckpt_path=ck, ckpt_every=15, log=lambda *_: None,
                      val=(prot[:2], clean[:2]))
    with torch.no_grad():
        after = (m(x) - y).abs().mean().item()
    assert after < before
    assert res["val"]["n"] == 2
    assert os.path.isfile(ck)
    m2 = AE.load_model(ck, "cpu")
    xx = AE.u8_to_unit(prot[0]).unsqueeze(0)
    with torch.no_grad():
        assert torch.allclose(m(xx), m2(xx), atol=1e-5)
    # resume continues the step counter
    st = torch.load(ck, map_location="cpu")
    res2 = AE.train_ae(AE.AdaptiveAE(8, 2), s, "cpu", steps=st["step"] + 2, batch_size=4, resume_state=st,
                       log=lambda *_: None)
    assert res2["steps"] == st["step"] + 2


def test_purifier_apply_and_build_from_config(tmp_path):
    prot, clean = synth_pairs(2, 48)
    m = AE.AdaptiveAE(8, 2)
    ck = str(tmp_path / "w.pt")
    AE.save_checkpoint(ck, m, None, 0, 0.0)
    cfg = {"adaptive_ae": {"weights_path": ck, "base_channels": 8, "depth": 2}}
    p = AE.build_purifier(cfg, "cpu")
    assert p.name == "adaptive_ae"
    img = Image.fromarray(prot[0].permute(1, 2, 0).numpy(), "RGB")
    out = p.apply(img, 0)
    assert out.size == img.size and out.mode == "RGB"
    assert np.array_equal(np.asarray(out), np.asarray(img))  # identity at init
    assert np.array_equal(np.asarray(p.apply(img, 1)), np.asarray(p.apply(img, 2)))  # deterministic


def test_missing_weights_is_not_available(tmp_path):
    cfg = {"adaptive_ae": {"weights_path": str(tmp_path / "nope.pt")}}
    with pytest.raises(RuntimeError):
        AE.build_purifier(cfg, "cpu")


def test_list_pairs_and_split(tmp_path):
    (tmp_path / "p").mkdir()
    (tmp_path / "c").mkdir()
    for n in ["a", "b", "c"]:
        Image.new("RGB", (8, 8)).save(tmp_path / "c" / f"{n}.png")
    for n in ["a", "c", "z"]:
        Image.new("RGB", (8, 8)).save(tmp_path / "p" / f"{n}.png")
    pairs = AE.list_pairs(str(tmp_path / "p"), str(tmp_path / "c"))
    assert [p[0] for p in pairs] == ["a", "c"]
    tr, va = AE.split_pairs(10, 0.2, 1)
    assert len(va) == 2 and not set(tr) & set(va)
    assert AE.split_pairs(10, 0.2, 1) == (tr, va)
    tr1, va1 = AE.split_pairs(1, 0.5, 1)
    assert len(tr1) == 1 and va1 == []
