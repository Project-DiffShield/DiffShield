"""Shared CPU stand-ins for the protect tests (no model downloads). Not a test module itself
(the single test below only checks that the fakes behave like the diffusers call signatures)."""
import os
import sys
from types import SimpleNamespace

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from src.protect.models import SDModels, sd15_alphas_cumprod


class TinyVAE(nn.Module):
    """Same signatures as diffusers AutoencoderKL: encode(x).latent_dist.mean (stride 8, 4 ch),
    decode(z).sample, config.scaling_factor."""

    def __init__(self, scaling_factor=0.18215):
        super().__init__()
        self.enc = nn.Conv2d(3, 4, 8, stride=8)
        self.dec = nn.ConvTranspose2d(4, 3, 8, stride=8)
        self.config = SimpleNamespace(scaling_factor=scaling_factor)

    def encode(self, x):
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=torch.tanh(self.enc(x))))

    def decode(self, z):
        return SimpleNamespace(sample=self.dec(z))


class TinyUNet(nn.Module):
    """unet(zt, t, encoder_hidden_states=c).sample"""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 4, 3, padding=1)
        self.temb = nn.Embedding(1000, 8)

    def forward(self, zt, t, encoder_hidden_states=None):
        h = torch.tanh(self.c1(zt) + self.temb(t).view(-1, 8, 1, 1))
        return SimpleNamespace(sample=self.c2(h))


class TinySem:
    """ClipSem stand-in: L_sem = 1 - cos(random projection of the image, fixed target)."""

    def __init__(self, seed=3):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(3 * 4 * 4, 16, generator=g)
        self.tgt = torch.randn(1, 16, generator=g)

    def sem_loss(self, x):
        f = torch.nn.functional.adaptive_avg_pool2d(x, 4).flatten(1) @ self.W
        return (1 - torch.nn.functional.cosine_similarity(f, self.tgt.expand_as(f), dim=1)).mean()


class TinyFace(nn.Module):
    """FaceEmbedder stand-in with .embed(x, bbox)."""

    def __init__(self, seed=5):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(3 * 4 * 4, 8, generator=g), requires_grad=False)

    def embed(self, x, bbox=None):
        f = torch.nn.functional.adaptive_avg_pool2d(x, 4).flatten(1) @ self.W
        return torch.nn.functional.normalize(f, dim=1)


def make_models(device="cpu", seed=0, scaling_factor=0.18215, **kw):
    torch.manual_seed(seed)
    vae, unet = TinyVAE(scaling_factor), TinyUNet()
    null = torch.randn(1, 77, 16)
    return SDModels(vae, unet, sd15_alphas_cumprod(), null, device=device, unet_dtype="fp32", **kw)


def make_image(size=64, seed=0):
    """Image already on the uint8 grid, as if read from a PNG, in [-1,1]."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 3, size, size, generator=g)
    smooth = torch.nn.functional.avg_pool2d(base, 5, 1, 2)
    u8 = (smooth * 255).round().clamp(0, 255)
    return u8 * (2.0 / 255.0) - 1.0


def base_cfg(iters=3, k=2):
    return {
        "budget": {"eps_255": 4, "step_255": 0.5, "iters": iters, "random_start": "uniform_full"},
        "loss": {"alpha": 0.27, "beta": 0.9, "gamma": 60.5, "eta": 0.27, "lambda_id": 0.0,
                 "sem_target_text": "a potted plant", "normalize_terms": False,
                 "anti_purify": {"enabled": False, "w_hf": 0.5, "w_mono": 0.5}},
        "eot": {"k_draws": k},
        "runtime": {"unet_dtype": "fp32", "grad_checkpointing": False, "loss_scale": 1024.0},
        "purifiers_train": ["jpeg", "downup", "vae_roundtrip", "sd_onestep"],
        "purifier_train_probs": {"jpeg": 0.25, "downup": 0.25, "vae_roundtrip": 0.2, "sd_onestep": 0.3},
        "models": {"sd15": {"id": "x/sd15", "revision": None}},
    }


def test_fakes_signatures():
    m = make_models()
    x = make_image()
    z = m.encode_mean(x)
    assert z.shape == (1, 4, 8, 8)
    assert m.decode(z, scaled=False).shape == x.shape
    t = torch.tensor([100])
    assert m.unet_eps(z, t).shape == z.shape
