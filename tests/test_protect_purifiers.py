import pytest
import torch

from test_protect_fakes import base_cfg, make_image, make_models  # noqa: F401  (also fixes sys.path)

from src.protect import purifiers as P
from src.protect.models import make_generator


def _grad_check(pur, x, seed=0):
    x = x.clone().requires_grad_(True)
    gen = make_generator(seed)
    y = pur(x, gen)
    assert y.shape == x.shape
    (y ** 2).mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    return y.detach()


def test_gradient_flows_through_every_purifier():
    m = make_models()
    x = make_image()
    purs = [P.Identity(), P.DiffJPEG((50, 95), "kornia_diff"), P.DiffJPEG((50, 95), "pil_bpda"),
            P.DownUp((0.5, 0.9)), P.VAERoundTrip(m), P.SDOneStepPurifier(m, 50, 300)]
    for pur in purs:
        _grad_check(pur, x)


def test_jpeg_changes_image_and_quality_is_random():
    x = make_image()
    j = P.DiffJPEG((10, 30))
    y1 = j(x, make_generator(1))
    y2 = j(x, make_generator(2))
    assert (y1 - x).abs().max() > 1e-3
    assert not torch.allclose(y1, y2)
    assert torch.allclose(y1, j(x, make_generator(1)))  # same seed -> same output


def test_downup_losing_detail_and_heldout_disjoint():
    cfg = base_cfg()
    import yaml
    import os

    part = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config", "parts", "protect.yaml")))
    cfg["purifier_train_params"] = part["purifier_train_params"]
    tr = cfg["purifier_train_params"]["downup"]["scale"]
    ho = cfg["purifier_train_params"]["downup_heldout"]["scale"]
    assert ho[1] < tr[0] or ho[0] > tr[1]  # disjoint scale ranges
    x = make_image()
    y = P.build_heldout(cfg)["downup_heldout"](x, make_generator(0))
    assert (y - x).abs().max() > 0


def test_onestep_aux_and_scaling_factor_from_config():
    m1 = make_models(scaling_factor=0.18215)
    m2 = make_models(scaling_factor=0.13025)  # same weights (seed), different VAE scaling factor
    x = make_image()
    o1, o2 = P.SDOneStepPurifier(m1), P.SDOneStepPurifier(m2)
    r1, a1 = o1.forward_aux(x, make_generator(7))
    r2, a2 = o2.forward_aux(x, make_generator(7))
    assert set(a1) >= {"t", "noise", "eps", "z0", "zt", "ab", "rec"}
    assert 50 <= int(a1["t"]) <= 300
    assert torch.allclose(a1["z0"], m1.encode_mean(x) * 0.18215)
    assert torch.allclose(a2["z0"], m2.encode_mean(x) * 0.13025)
    assert not torch.allclose(r1, r2)  # the factor really enters the pipeline
    # deterministic given generator state
    r1b, _ = o1.forward_aux(x, make_generator(7))
    assert torch.allclose(r1, r1b)


def test_onestep_t_range_inclusive_and_alpha_on_device():
    m = make_models()
    o = P.SDOneStepPurifier(m, 50, 52)
    x = make_image()
    gen = make_generator(0)
    seen = set()
    for _ in range(60):
        t, _n = o.sample_t_noise(x, gen)
        seen.add(int(t))
    assert seen == {50, 51, 52}
    assert m.alphas_cumprod.device == x.device and m.alphas_cumprod.dtype == torch.float32


def test_mixture_renormalises_excludes_and_empty():
    m = make_models()
    purs = {"jpeg": P.DiffJPEG(), "downup": P.DownUp(), "sd_onestep": P.SDOneStepPurifier(m)}
    mix = P.PurifierMixture(purs, {"jpeg": 0.25, "downup": 0.25, "sd_onestep": 0.3}, exclude=["downup"])
    assert mix.names == ["jpeg", "sd_onestep"]
    assert abs(float(mix.probs.sum()) - 1.0) < 1e-6
    assert abs(float(mix.probs[1]) - 0.3 / 0.55) < 1e-6
    x = make_image()
    gen = make_generator(0)
    cnt = {}
    for _ in range(300):
        _, aux, name = mix.sample(x, gen)
        cnt[name] = cnt.get(name, 0) + 1
        assert (aux is not None) == (name == "sd_onestep")
    assert set(cnt) == {"jpeg", "sd_onestep"}
    assert abs(cnt["sd_onestep"] / 300 - 0.3 / 0.55) < 0.12
    empty = P.PurifierMixture({}, None)
    xp, aux, name = empty.sample(x, gen)
    assert name == "none" and aux is None and torch.allclose(xp, x)


def test_build_mixture_from_cfg_excludes_heldout():
    cfg = base_cfg()
    cfg["purifiers_train"] = ["jpeg", "downup", "downup_heldout"]
    cfg["purifier_train_params"] = {"downup_heldout": {"scale": [0.3, 0.45]}}
    mix = P.build_mixture(cfg, None)
    assert "downup_heldout" not in mix.names and set(mix.names) == {"jpeg", "downup"}


def test_pil_bpda_is_real_codec_forward_identity_backward():
    x = make_image()
    j = P.DiffJPEG((20, 20), "pil_bpda")
    xx = x.clone().requires_grad_(True)
    y = j(xx, make_generator(0))
    y.sum().backward()
    assert torch.allclose(xx.grad, torch.full_like(xx, 1.0))  # d(sum y)/dx = 1 (straight-through, x01*2-1 chain)
    assert (y - x).abs().max() > 1e-3


def test_ste_clamp():
    x = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
    y = P.ste_clamp(x)
    assert torch.allclose(y.detach(), torch.tensor([-1.0, 0.5, 1.0]))
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones(3))
