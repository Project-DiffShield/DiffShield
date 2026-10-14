"""Cross-agent wiring: every method name resolves through src.protect.registry with the real merged config
(config/config.yaml + config/parts/*.yaml) exactly as scripts/run_protect.py does it, without downloads."""
import pytest
import torch

from src.protect.registry import build_protector
from src.utils.config import load_config

V2 = ["v2_cheap", "v2_main", "v2_antipure", "v2_final", "abl_vis_only", "abl_den_only", "abl_no_sem", "abl_no_pur",
      "eps2", "eps6", "eps8"]
BASELINES = ["noise_uniform", "noise_gauss_psnr", "photoguard", "mist"]


@pytest.fixture
def fake_sd(monkeypatch):
    import src.baselines.registry as br
    from src.utils.tiny_models import tiny_vae

    def fake_load(cfg, device, need_unet=False, dtype="auto", grad_checkpointing=True, prompt=""):
        comp = {"vae": tiny_vae().eval(), "dtype": torch.float32, "scaling_factor": 0.18215}
        if need_unet:
            comp.update({"unet": torch.nn.Identity(), "cond": torch.zeros(1, 77, 8),
                         "alphas_cumprod": torch.linspace(0.99, 0.01, 1000)})
        return comp

    monkeypatch.setattr(br, "load_sd_components", fake_load)


@pytest.mark.parametrize("name", V2)
def test_v2_names_dispatch_with_real_config(name, repo_root):
    cfg = load_config(repo_root / "config" / "config.yaml", method=name)
    p = build_protector(name, cfg, "cpu")
    assert type(p).__module__.endswith("v2_registry")
    # handoff step size is explicit for every v2 variant, v1's own value stays 0.5
    assert p.cfg["budget"]["step_255"] == 1.0
    want_eps = {"eps2": 2, "eps6": 6, "eps8": 8}.get(name, 4)
    assert p.cfg["budget"]["eps_255"] == want_eps
    assert p.cfg["loss"]["alpha"] in (0.0, 0.2699475469647097)


def test_v1_preset_keeps_true_step(repo_root):
    cfg = load_config(repo_root / "config" / "config.yaml", method="v1")
    assert cfg["budget"]["step_255"] == 0.5 and cfg["budget"]["eps_255"] == 4


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_names_dispatch(name, repo_root, fake_sd):
    cfg = load_config(repo_root / "config" / "config.yaml")
    p = build_protector(name, cfg, "cpu")
    assert type(p).__module__.startswith("src.baselines")
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    if name.startswith("noise"):
        y, info = p.protect(x, 1)
        assert y.shape == x.shape and isinstance(info, dict)


def test_unknown_name_raises(repo_root):
    cfg = load_config(repo_root / "config" / "config.yaml")
    with pytest.raises(KeyError):
        build_protector("nope", cfg, "cpu")


def test_adaptive_ae_build_purifier_signature(repo_root):
    from src.eval.adaptive_ae import build_purifier

    cfg = load_config(repo_root / "config" / "config.yaml")
    with pytest.raises(Exception) as e:  # weights are produced by train_adaptive_ae.py: a clear "not available" error
        build_purifier(cfg, "cpu")
    assert "weights" in str(e.value).lower()


def test_final_yaml_layer_is_picked_up(tmp_path, repo_root):
    import shutil

    shutil.copytree(repo_root / "config", tmp_path / "config")
    (tmp_path / "config" / "parts" / "final.yaml").write_text(
        "methods:\n  v2_final:\n    loss: {beta: 0.0}\n")
    cfg = load_config(tmp_path / "config" / "config.yaml", method="v2_final")
    p = build_protector("v2_final", cfg, "cpu")
    assert p.cfg["loss"]["beta"] == 0.0
    p8 = build_protector("eps8", load_config(tmp_path / "config" / "config.yaml", method="eps8"), "cpu")
    assert p8.cfg["loss"]["beta"] == 0.0 and p8.cfg["budget"]["eps_255"] == 8
