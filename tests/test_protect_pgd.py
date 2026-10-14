import pytest
import torch

from test_protect_fakes import (TinyFace, TinySem, base_cfg, make_image, make_models)  # noqa: F401

from src.protect import losses as L
from src.protect.antipure import AntiPure, hf_energy
from src.protect.ensemble import EnsembleVis
from src.protect.face_loss import IdentityLoss
from src.protect.losses import SingleVis
from src.protect.pgd import PGDParams, from_u8, params_from_cfg, pgd_protect, to_u8
from src.protect.purifiers import SDOneStepPurifier, build_mixture


def run(cfg=None, seed=0, x=None, sem=True, id_loss=None, antipure=False, extra_vae=None, models=None, **over):
    cfg = cfg or base_cfg()
    m = models or make_models()
    p = params_from_cfg(cfg)
    for k, v in over.items():
        setattr(p, k, v)
    mix = build_mixture(cfg, m)
    vis = EnsembleVis(m, {"b": extra_vae}) if extra_vae is not None else SingleVis(m)
    ap = AntiPure(mix.purifiers["sd_onestep"]) if antipure else None
    x = make_image() if x is None else x
    return pgd_protect(x, m, mix, p, vis=vis, sem=TinySem() if sem else None, id_loss=id_loss,
                       antipure=ap, seed=seed), x


@pytest.mark.parametrize("eps", [2, 4, 8])
def test_projection_respects_eps_in_uint8_terms(eps):
    cfg = base_cfg(iters=6)
    cfg["budget"]["eps_255"] = eps
    (xa, info), x = run(cfg)
    diff = (to_u8(xa) - to_u8(x)).abs()
    assert diff.max() <= eps
    assert info["linf_u8"] <= eps
    assert diff.max() >= 1  # the perturbation is not empty
    assert xa.min() >= -1 and xa.max() <= 1
    # output lies exactly on the uint8 grid
    u = (xa + 1) * 127.5
    assert torch.allclose(u, u.round(), atol=1e-3)


def test_budget_saturates_with_enough_steps():
    cfg = base_cfg(iters=12)
    cfg["budget"].update(eps_255=3, step_255=1.0)
    (xa, info), x = run(cfg)
    assert info["linf_u8"] == 3


def test_unquantized_still_within_budget_when_saved_as_png():
    cfg = base_cfg(iters=6)
    (xa, info), x = run(cfg, quantize=False)
    assert (to_u8(xa) - to_u8(x)).abs().max() <= 4  # rounding on save cannot add a gray level
    assert (xa - x).abs().max() <= 4 * 2 / 255 + 1e-6


def test_clamp_at_range_edges():
    x = make_image()
    x[..., :8, :8] = -1.0
    x[..., -8:, -8:] = 1.0
    (xa, info), _ = run(base_cfg(iters=8), x=x)
    assert xa.min() >= -1 and xa.max() <= 1
    assert (to_u8(xa) - to_u8(x)).abs().max() <= 4


def test_determinism_by_seed():
    (a, ia), _ = run(seed=11)
    (b, ib), _ = run(seed=11)
    (c, _), _ = run(seed=12)
    assert torch.equal(a, b)
    assert ia["purifier_counts"] == ib["purifier_counts"]
    assert not torch.equal(a, c)


def test_info_contents_and_all_terms_logged():
    (xa, info), _ = run(base_cfg(iters=3, k=4))
    for k in ("vis", "den", "sem", "str"):
        assert len(info["terms"][k]) == 3
    assert info["terms_init"].keys() >= {"vis", "den", "sem", "str"}
    assert sum(info["purifier_counts"].values()) == 3 * 4
    assert info["time_s"] > 0 and info["psnr"] > 30 and 0 < info["ssim"] <= 1
    assert info["nonfinite_draws"] == 0 and 0 <= info["zero_grad_frac_mean"] <= 1


def test_ascent_increases_vis_when_it_is_the_only_term():
    cfg = base_cfg(iters=10)
    cfg["budget"].update(random_start="uniform_half", step_255=1.0)
    cfg["loss"].update(alpha=1.0, eta=0.0, beta=0.0, gamma=0.0)
    cfg["purifiers_train"] = []
    (xa, info), _ = run(cfg, sem=False)
    v = info["terms"]["vis"]
    assert v[-1] > v[0]


def test_loss_scale_does_not_change_the_result():
    cfg1, cfg2 = base_cfg(iters=4), base_cfg(iters=4)
    cfg1["runtime"]["loss_scale"] = 1.0
    cfg2["runtime"]["loss_scale"] = 4096.0
    (a, _), _ = run(cfg1, seed=3)
    (b, _), _ = run(cfg2, seed=3)
    assert torch.equal(a, b)


def test_no_purifier_and_zero_weights_paths():
    cfg = base_cfg(iters=2)
    cfg["purifiers_train"] = []
    (xa, info), _ = run(cfg)
    assert info["purifier_counts"] == {"none": 2 * 2}
    cfg = base_cfg(iters=2)
    cfg["loss"].update(alpha=0.0, beta=0.0, eta=0.27)  # den-only
    (xa, info), _ = run(cfg, sem=False)
    assert "vis" not in info["terms_init"] and "den" in info["terms_init"]


def test_antipure_terms_and_gradients():
    m = make_models()
    mix = build_mixture(base_cfg(), m)
    o = mix.purifiers["sd_onestep"]
    x = make_image()
    ap = AntiPure(o)
    st = ap.prepare(x)
    xx = x.clone().requires_grad_(True)
    from src.protect.models import make_generator

    gen = make_generator(0)
    rec, aux = o.forward_aux(xx, gen)
    terms = ap.terms(aux, xx, st, gen)
    assert set(terms) == {"hf", "mono"}
    (terms["hf"] + terms["mono"]).backward()
    assert xx.grad.abs().sum() > 0
    assert ap.terms(None, xx, st, gen) == {}
    # pipeline with antipure on logs the terms
    cfg = base_cfg(iters=3, k=4)
    cfg["purifier_train_probs"] = {"jpeg": 0.0, "downup": 0.0, "vae_roundtrip": 0.0, "sd_onestep": 1.0}
    (_, info), _ = run(cfg, antipure=True)
    assert "hf" in info["terms"] and "mono" in info["terms"]


def test_hf_energy_higher_for_noisy_image():
    x = make_image()
    noisy = (x + 0.3 * torch.randn_like(x)).clamp(-1, 1)
    assert hf_energy(noisy) > 5 * hf_energy(x)


def test_l_str_and_ssim():
    x = make_image()
    assert abs(float(L.ssim(x, x)) - 1.0) < 1e-5
    n1 = (x + 0.02 * torch.randn_like(x)).clamp(-1, 1)
    n2 = (x + 0.2 * torch.randn_like(x)).clamp(-1, 1)
    assert L.l_str(x, n1) < L.l_str(x, n2)


def test_l_den_reuses_aux_and_has_grad_otherwise():
    from src.protect.models import make_generator
    from src.protect.purifiers import SDOneStepPurifier

    m = make_models()
    x = make_image().requires_grad_(True)
    o = SDOneStepPurifier(m)
    gen = make_generator(0)
    _, aux = o.forward_aux(x, gen)
    d = L.l_den(m, None, gen, aux)
    assert torch.allclose(d, torch.nn.functional.mse_loss(aux["eps"], aux["noise"]))
    d2 = L.l_den(m, x, make_generator(1))
    d2.backward()
    assert x.grad.abs().sum() > 0


def test_ensemble_equals_single_when_vaes_identical_and_grad_flows():
    m = make_models()
    x = make_image()
    single, ens = SingleVis(m), EnsembleVis(m, {"b": m.vae})
    xp = (x + 0.05).clamp(-1, 1).requires_grad_(True)
    v1 = single(xp, single.prepare(x))
    v2 = ens(xp, ens.prepare(x))
    assert torch.allclose(v1, v2, atol=1e-7)
    v2.backward()
    assert xp.grad.abs().sum() > 0
    assert abs(sum(ens.w.values()) - 1) < 1e-9


def test_ensemble_in_pgd_with_second_distinct_vae():
    from test_protect_fakes import TinyVAE

    torch.manual_seed(9)
    other = TinyVAE(0.13025)
    (xa, info), x = run(base_cfg(iters=3), extra_vae=other)
    assert info["linf_u8"] <= 4 and "vis" in info["terms"]


def test_identity_loss_gradient_and_logged():
    m = make_models()
    idl = IdentityLoss(TinyFace(), lambda z: m.decode(m.encode_mean(z, scaled=False), scaled=False))
    x = make_image()
    idl.prepare(x)
    xp = (x + 0.1).clamp(-1, 1).requires_grad_(True)
    idl(xp).backward()
    assert xp.grad.abs().sum() > 0
    cfg = base_cfg(iters=3)
    cfg["loss"]["lambda_id"] = 1.0
    (_, info), _ = run(cfg, id_loss=idl, models=m)
    assert len(info["terms"]["id"]) == 3


def test_term_scales_calibration_roundtrip():
    (_, info), _ = run(base_cfg(iters=2))
    sc = L.scales_from_infos([info])
    assert set(sc) >= {"vis", "den", "str"}
    cfg = base_cfg(iters=2)
    cfg["loss"]["normalize_terms"] = True
    cfg["loss_norm"] = sc
    assert params_from_cfg(cfg).loss_norm == sc
    (_, info2), _ = run(cfg)
    assert info2["linf_u8"] <= 4
