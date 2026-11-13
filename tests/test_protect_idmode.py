"""Exploration X3-X5: IdentityLoss modes (recon = original, direct = new) and the v2_x_* presets. CPU, fake embedder, no downloads."""
import pytest

torch = pytest.importorskip("torch")

from test_protect_fakes import TinyFace, TinySem, base_cfg, make_image, make_models  # noqa: E402,F401

from src.protect import face_loss as FL  # noqa: E402
from src.protect.face_loss import IdentityLoss  # noqa: E402
from src.protect.v2_registry import V2Protector, deep_merge, resolve_cfg  # noqa: E402
from test_protect_registry import full_cfg  # noqa: E402


def _recon(m):
    return lambda z: m.decode(m.encode_mean(z, scaled=False), scaled=False)


def test_default_mode_is_recon_and_unchanged():
    m, f, x = make_models(), TinyFace(), make_image()
    default, explicit = IdentityLoss(f, _recon(m)), IdentityLoss(f, _recon(m), mode="recon")
    assert default.mode == "recon"
    for idl in (default, explicit):
        idl.prepare(x)
    xp = (x + 0.05).clamp(-1, 1)
    manual = torch.nn.functional.cosine_similarity(f.embed(_recon(m)(xp)), f.embed(x), dim=1).mean()
    assert torch.allclose(default(xp), manual) and torch.allclose(explicit(xp), manual)


def test_direct_mode_embeds_x_pub_itself_without_vae():
    f, x = TinyFace(), make_image()

    def boom(_):
        raise AssertionError("recon must not be called in direct mode")

    idl = IdentityLoss(f, boom, mode="direct")
    idl.prepare(x)
    xp = (x + 0.05).clamp(-1, 1)
    manual = torch.nn.functional.cosine_similarity(f.embed(xp), f.embed(x), dim=1).mean()
    assert torch.allclose(idl(xp), manual)
    IdentityLoss(f, None, mode="direct")          # recon not needed
    with pytest.raises(ValueError):
        IdentityLoss(f, None, mode="recon")
    with pytest.raises(ValueError):
        IdentityLoss(f, None, mode="nope")


def test_direct_mode_differentiates_through_x_pub_only():
    f, x = TinyFace(), make_image().requires_grad_(True)
    idl = IdentityLoss(f, None, mode="direct")
    idl.prepare(x)                                # source embedding must be detached
    assert not idl.src.requires_grad
    xp = (x.detach() + 0.1).clamp(-1, 1).requires_grad_(True)
    idl(xp).backward()
    assert xp.grad is not None and xp.grad.abs().sum() > 0
    assert x.grad is None                         # nothing flowed into the source image


def test_direct_loss_is_minimal_cos_one_for_identical_image():
    f, x = TinyFace(), make_image()
    idl = IdentityLoss(f, None, mode="direct")
    idl.prepare(x)
    assert float(idl(x)) == pytest.approx(1.0, abs=1e-5)


def test_lambda_zero_loads_nothing(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("embedder must not be loaded when lambda_id == 0")

    monkeypatch.setattr(FL, "load_face_embedder", boom)
    p = V2Protector("v2_next", full_cfg(), "cpu", {"models": make_models(), "sem": TinySem()})
    p._build()
    assert p.id_loss is None and p.params.lambda_id == 0.0


@pytest.mark.parametrize("name,mode,lam", [("v2_x_id_direct_a", "direct", 2.0), ("v2_x_id_direct_b", "direct", 10.0),
                                           ("v2_x_id_recon", "recon", 2.0)])
def test_id_presets_wire_mode_and_logging(name, mode, lam):
    c = full_cfg(budget={"iters_cap": 2})
    r = resolve_cfg(name, c)
    assert r["face"]["id_mode"] == mode and r["loss"]["lambda_id"] == lam and r["face"]["backend"] == "facenet"
    p = V2Protector(name, c, "cpu", {"models": make_models(), "sem": TinySem(), "face": TinyFace()})
    xa, info = p.protect(make_image(), seed=1)
    assert p.id_loss.mode == mode and len(info["terms"]["id"]) == 2


def test_presets_differ_from_v2_next_by_exactly_one_thing():
    c = full_cfg()
    base = resolve_cfg("v2_next", c)
    assert base["face"]["id_mode"] == "recon" and base["loss"]["lambda_id"] == 0.0 and base["budget"]["iters"] == 2          # full_cfg() base iters; v2_next itself does not set it
    assert resolve_cfg("v2_x_eps8", c)["budget"]["eps_255"] == 8 and base["budget"]["eps_255"] == 4
    assert resolve_cfg("v2_x_eps8", c)["budget"]["step_255"] == 1.0
    assert resolve_cfg("v2_x_it100", c)["budget"]["iters"] == 100
    for n in ("v2_x_eps8", "v2_x_it100", "v2_x_id_direct_a", "v2_x_id_direct_b", "v2_x_id_recon"):
        r = resolve_cfg(n, c)
        assert r["loss"]["alpha"] == 0.55 and r["loss"]["target"]["kind"] == "zero" and r["eot"]["k_draws"] == 4
        assert r["purifier_train_probs"] == base["purifier_train_probs"]


def test_iters_cap_only_lowers():
    c = full_cfg(budget={"iters_cap": 3})
    assert resolve_cfg("v2_x_it100", c)["budget"]["iters"] == 3
    assert resolve_cfg("v2_next", full_cfg(budget={"iters_cap": 0}))["budget"]["iters"] == 2
    assert resolve_cfg("v2_next", full_cfg(budget={"iters_cap": 99}))["budget"]["iters"] == 2
    assert resolve_cfg("v2_next", full_cfg(budget={"iters_cap": 1}))["budget"]["iters"] == 1


def test_onnx2torch_guard_blocks_the_evaluation_arcface():
    for path in ("/x/buffalo_l/w600k_r50.onnx", "arcface.onnx"):
        with pytest.raises(ValueError):
            FL._guard_not_eval_model({"onnx_path": path})
    FL._guard_not_eval_model({"onnx_path": "/x/some_other_net.onnx"})
    FL._guard_not_eval_model({"onnx_path": "/x/w600k_r50.onnx", "allow_eval_model": True})
    with pytest.raises(ValueError):
        FL.load_face_embedder({"face": {"backend": "onnx2torch", "onnx_path": "w600k_r50.onnx"}}, "cpu")
