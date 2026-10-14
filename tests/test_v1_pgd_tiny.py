import torch

from src.eot import EoTLayer
from src.optimizer import PGDOptimizer
from src.utils.tiny_models import tiny_target_embedding, tiny_v1_loss


def _run(seed, iters=3):
    torch.manual_seed(seed)
    loss_fn = tiny_v1_loss()
    tgt = tiny_target_embedding()
    opt = PGDOptimizer(epsilon=8 / 255, alpha=1 / 255, iters=iters, device="cpu", loss_fn=loss_fn)
    opt.eot = EoTLayer(image_size=64)  # default is 512 which makes CPU tests slow
    g = torch.Generator().manual_seed(123)
    x = torch.rand(1, 3, 64, 64, generator=g) * 1.6 - 0.8
    torch.manual_seed(seed)
    return x, opt.optimize(x, tgt, w_alpha=0.27, w_beta=0.9, w_gamma=60.5)


def test_pgd_tiny_bounds_and_determinism():
    x, a = _run(0)
    _, b = _run(0)
    _, c = _run(1)
    assert a.min() >= -1 and a.max() <= 1
    assert (a - x).abs().max() <= 8 / 255 + 1e-6
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert (a - x).abs().max() > 0


def test_gradients_nonzero():
    loss_fn = tiny_v1_loss()
    tgt = tiny_target_embedding()
    torch.manual_seed(0)
    x = torch.rand(1, 3, 64, 64) * 1.6 - 0.8
    y = (x + 0.03 * torch.randn_like(x)).requires_grad_(True)
    total, *_ = loss_fn(x, y, tgt, alpha=0.27, beta=0.9, gamma=60.5)
    total.backward()
    assert y.grad.abs().sum() > 0
