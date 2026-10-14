import math

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_func

from src.eval.metrics import (diff_histogram_u8, max_abs_diff_u8, png_quality_metrics, psnr_u8, ssim_u8,
                              v1_quality_metrics)
from src.utils.io import save_png_v1


def test_psnr_constant_diff_4():
    a = np.full((16, 16, 3), 100, np.uint8)
    b = a + 4
    assert abs(psnr_u8(a, b) - 20 * math.log10(255 / 4)) < 1e-9
    assert psnr_u8(a, a) == float("inf") and abs(ssim_u8(a, a) - 1.0) < 1e-9
    assert max_abs_diff_u8(a, b) == 4 and diff_histogram_u8(a, b)[4] == a.size


def _oracle(clean_tensor, immunized_tensor):
    """Verbatim copy of phase1_v1 compute_image_quality_metrics, LPIPS removed."""
    clean_np = ((clean_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    immunized_np = ((immunized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
    mse = np.mean((clean_np.astype(np.float64) - immunized_np.astype(np.float64)) ** 2)
    psnr = 20 * math.log10(255.0 / math.sqrt(mse)) if mse > 0 else float("inf")
    ssim = ssim_func(clean_np, immunized_np, channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5,
                     use_sample_covariance=False)
    linf = (immunized_tensor - clean_tensor).abs().max().item()
    return {"MSE": mse, "PSNR": psnr, "SSIM": ssim, "Linf": linf}


def test_v1_metrics_equal_oracle():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
    y = (x + (torch.rand(1, 3, 64, 64, generator=g) * 2 - 1) * 8 / 255).clamp(-1, 1)
    ours = v1_quality_metrics(x, y, lpips=lambda a, b: 0.5)
    ref = _oracle(x, y)
    for k in ref:
        assert ours[k] == ref[k], k
    assert ours["LPIPS"] == 0.5


def test_png_metrics(tmp_path):
    g = torch.Generator().manual_seed(1)
    x = torch.rand(1, 3, 32, 32, generator=g) * 2 - 1
    y = (x + 0.03).clamp(-1, 1)
    save_png_v1(x, tmp_path / "a.png")
    save_png_v1(y, tmp_path / "b.png")
    m = png_quality_metrics(tmp_path / "a.png", tmp_path / "b.png", None)
    assert m["maxabs_u8"] in (3, 4) and m["psnr"] > 30 and math.isnan(m["lpips"])
