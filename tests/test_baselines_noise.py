import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.baselines import noise_control as nc  # noqa: E402
from src.baselines.registry import build_baseline, canonical, resolve_params  # noqa: E402


def make_x(h=64, w=64, seed=0, saturate=True):
    g = torch.Generator().manual_seed(seed)
    u = torch.randint(0, 256, (1, 3, h, w), generator=g).float()
    if saturate:  # include clipped pixels so the budget logic is exercised at the 0/255 edges
        u[0, :, :4, :4] = 0
        u[0, :, -4:, -4:] = 255
    return nc.from_u8(u)


@pytest.mark.parametrize("eps", [2, 4, 8])
def test_uniform_budget_exact_in_uint8(eps):
    x = make_x()
    y = nc.uniform_linf(x, eps, seed=1)
    d = nc.to_u8(y) - nc.to_u8(x)
    assert d.abs().max().item() == eps          # reached but never exceeded
    assert torch.allclose(y, nc.from_u8(nc.to_u8(y)))  # on the uint8 grid
    assert y.min() >= -1 and y.max() <= 1
    assert nc.to_u8(y).min() >= 0 and nc.to_u8(y).max() <= 255


def test_uniform_deterministic_and_seed_sensitive():
    x = make_x()
    a, b, c = nc.uniform_linf(x, 4, 7), nc.uniform_linf(x, 4, 7), nc.uniform_linf(x, 4, 8)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_sign_noise_budget():
    x = make_x(saturate=False)
    u = nc.to_u8(x)
    keep = (u >= 4) & (u <= 251)
    y = nc.sign_linf(x, 4, 3)
    d = (nc.to_u8(y) - u).abs()
    assert d.max().item() == 4
    assert (d[keep] == 4).all()


def test_protector_interface_uniform():
    p = build_baseline("noise_uniform", {"budget": {"eps_255": 4}}, "cpu")
    x = make_x()
    y, info = p.protect(x, 5)
    assert y.shape == x.shape and info["linf_u8"] <= 4 and info["eps_255"] == 4.0
    assert info["method"] == "noise_uniform"


def test_budget_config_override():
    assert resolve_params("noise_uniform", {"budget": {"eps_255": 6}})["eps_255"] == 6
    cfg = {"budget": {"eps_255": 6}, "baselines": {"noise_uniform": {"eps_255": 2}}}
    assert resolve_params("noise_uniform", cfg)["eps_255"] == 2
    assert resolve_params("photoguard", {})["eps_255"] == 4
    with pytest.raises(KeyError):
        canonical("nope")


@pytest.mark.parametrize("target", [35.0, 41.9, 46.0])
def test_gaussian_psnr_matching(target):
    x = make_x(128, 128, seed=3)
    y, info = nc.gaussian_matched_psnr(x, target, seed=4)
    assert abs(nc.psnr_tensors(x, y) - target) < 0.05
    assert abs(info["achieved_psnr"] - target) < 0.05
    assert nc.to_u8(y).min() >= 0 and nc.to_u8(y).max() <= 255


def test_gaussian_per_image_target_resolution(tmp_path):
    csvp = tmp_path / "m.csv"
    csvp.write_text("Original_Filename,PSNR_dB\n13117.jpg,40.0\n1172.jpg,44.0\n", encoding="utf-8")
    cfg = {"baselines": {"noise_gauss_psnr": {"psnr_csv": str(csvp), "target_psnr": 41.9}}}
    p = build_baseline("noise_gauss_psnr", cfg, "cpu")
    x = make_x(96, 96)
    _, i1 = p.protect(x, 1, image_id="13117.jpg")
    _, i2 = p.protect(x, 1, image_id="1172")
    _, i3 = p.protect(x, 1)
    _, i4 = p.protect(x, 1, image_id="13117.jpg", target_psnr=38.0)
    assert (i1["target_source"], i2["target_source"], i3["target_source"], i4["target_source"]) == ("map", "map", "constant", "arg")
    for inf, t in ((i1, 40.0), (i2, 44.0), (i3, 41.9), (i4, 38.0)):
        assert abs(inf["psnr"] - t) < 0.06


def test_missing_psnr_csv_raises():
    cfg = {"baselines": {"noise_gauss_psnr": {"psnr_csv": "/no/such.csv"}}}
    with pytest.raises(FileNotFoundError):
        build_baseline("noise_gauss_psnr", cfg, "cpu")
