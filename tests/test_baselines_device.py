"""S04 regression: PhotoGuard/Mist must expose .device so run_protect moves the image next to the (cuda) model."""
import torch

from src.baselines import registry


class _Vae(torch.nn.Module):
    dtype = torch.float32


def _fake_components(cfg, device, **kw):
    comp = {"vae": _Vae(), "dtype": torch.float32, "scaling_factor": 0.18215}
    if kw.get("need_unet"):
        comp.update({"unet": torch.nn.Identity(), "cond": torch.zeros(1, 1, 1), "alphas_cumprod": torch.ones(1000)})
    return comp


def test_photoguard_and_mist_expose_device(monkeypatch):
    monkeypatch.setattr(registry, "load_sd_components", _fake_components)
    for name in ("photoguard", "mist"):
        prot = registry.build_baseline(name, {"budget": {"eps_255": 4}}, "cpu")
        assert prot.device == "cpu", name
