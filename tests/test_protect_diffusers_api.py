"""API-compat check against REAL diffusers classes with tiny randomly initialised weights (no downloads).
Skipped if diffusers is not installed."""
import pytest
import torch

pytest.importorskip("diffusers")

from test_protect_fakes import make_image  # noqa: E402

from diffusers import AutoencoderKL, UNet2DConditionModel  # noqa: E402

from src.protect.losses import l_den  # noqa: E402
from src.protect.models import SDModels, make_generator, sd15_alphas_cumprod  # noqa: E402
from src.protect.purifiers import SDOneStepPurifier, VAERoundTrip  # noqa: E402


def tiny_real(ckpt):
    torch.manual_seed(0)
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8),
                        layers_per_block=1, latent_channels=4, norm_num_groups=8, sample_size=64,
                        scaling_factor=0.18215)
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                                block_out_channels=(16, 32), down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
                                up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16,
                                attention_head_dim=4, norm_num_groups=8)
    return SDModels(vae, unet, sd15_alphas_cumprod(), torch.randn(1, 77, 16), device="cpu", unet_dtype="fp32",
                    grad_checkpointing=ckpt)


@pytest.mark.parametrize("ckpt", [False, True])
def test_real_diffusers_modules_gradient_flow_and_checkpointing_equivalence(ckpt):
    m = tiny_real(ckpt)
    assert m.scaling_factor == pytest.approx(0.18215)  # read from vae.config
    x = make_image().requires_grad_(True)
    o = SDOneStepPurifier(m, 50, 300)
    rec, aux = o.forward_aux(x, make_generator(0))
    (rec.pow(2).mean() + l_den(m, None, make_generator(0), aux)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    VAERoundTrip(m)(make_image().requires_grad_(True), make_generator(0)).mean().backward()


def test_checkpointing_does_not_change_outputs():
    a, b = tiny_real(False), tiny_real(True)
    x = make_image()
    ra, _ = SDOneStepPurifier(a).forward_aux(x, make_generator(1))
    rb, _ = SDOneStepPurifier(b).forward_aux(x, make_generator(1))
    assert torch.allclose(ra, rb, atol=1e-5)
