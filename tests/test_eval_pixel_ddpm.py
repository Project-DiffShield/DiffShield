"""DiffPure-style pixel-space face DDPM purifier + downup_256 control, with tiny random-weight stand-ins (CPU, no downloads)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("diffusers")
import torch
from diffusers import DDPMScheduler, UNet2DModel

from src.eval import purify_attacks as pa


def tiny():
    torch.manual_seed(0)
    unet = UNet2DModel(sample_size=16, in_channels=3, out_channels=3, layers_per_block=1, block_out_channels=(8, 16),
                       down_block_types=("DownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "UpBlock2D"),
                       norm_num_groups=4).eval()
    return unet, DDPMScheduler(num_train_timesteps=1000)


def img(w=32, h=32, seed=0):
    rng = np.random.RandomState(seed)
    base = np.linspace(0, 255, w)[None, :, None].repeat(h, 0).repeat(3, 2)
    return Image.fromarray(np.clip(base + rng.randn(h, w, 3) * 10, 0, 255).astype(np.uint8))


def build(name, **over):
    unet, sched = tiny()
    p = {"size": 16, "_inject": {"unet": unet, "scheduler": sched}}
    p.update(over)
    return pa.build_purifier(name, {"eval": {"purifiers": {name: p}}})


def test_registered_and_aliases():
    names = set(pa.available_names())
    assert {"downup_256", "diffpure_ddpm256_t100", "diffpure_ddpm256_t200"} <= names
    for n in ("downup_256", "diffpure_ddpm256_t100", "diffpure_ddpm256_t200"):
        assert pa.canonical_name(n) == n
    p = build("diffpure_ddpm256_t100")
    assert p.t_star == 100 and pa.build_purifier("diffpure_ddpm256_t200").t_star == 200
    assert pa.build_purifier("diffpure_ddpm256_t100").model_id == "google/ddpm-celebahq-256"


def test_output_shape_dtype_range_and_determinism():
    x = img()
    p = build("diffpure_ddpm256_t100", t_star=8)
    a = p.apply(x, 3)
    assert a.size == x.size and a.mode == "RGB"
    arr = np.asarray(a)
    assert arr.dtype == np.uint8 and arr.shape == (32, 32, 3)
    assert np.array_equal(arr, np.asarray(p.apply(x, 3)))
    assert np.array_equal(arr, np.asarray(build("diffpure_ddpm256_t100", t_star=8).apply(x, 3)))
    assert not np.array_equal(arr, np.asarray(p.apply(x, 4)))        # seed matters


def test_tstar_zero_is_pure_resize_and_matches_control():
    x = img()
    ctrl = build("downup_256", size=16).apply(x, 0)
    ref = x.resize((16, 16), Image.BICUBIC).resize((32, 32), Image.BICUBIC)
    assert np.array_equal(np.asarray(ctrl), np.asarray(ref))
    z = build("diffpure_ddpm256_t100", t_star=0).apply(x, 0)
    assert np.array_equal(np.asarray(z), np.asarray(ref))


def test_different_tstar_and_strided_steps_differ():
    x = img()
    a = np.asarray(build("diffpure_ddpm256_t100", t_star=5).apply(x, 1))
    b = np.asarray(build("diffpure_ddpm256_t100", t_star=50).apply(x, 1))
    c = np.asarray(build("diffpure_ddpm256_t100", t_star=50, steps=5).apply(x, 1))
    assert not np.array_equal(a, b) and not np.array_equal(b, c)
    assert c.shape == (32, 32, 3)


def test_unet_call_count():
    p = build("diffpure_ddpm256_t100", t_star=7)
    n = {"c": 0}
    p.unet.register_forward_hook(lambda *a: n.__setitem__("c", n["c"] + 1))
    p.apply(img(), 0)
    assert n["c"] == 7
    p2 = build("diffpure_ddpm256_t100", t_star=20, steps=4)
    n2 = {"c": 0}
    p2.unet.register_forward_hook(lambda *a: n2.__setitem__("c", n2["c"] + 1))
    p2.apply(img(), 0)
    assert n2["c"] == 4


def test_not_available_when_weights_missing():
    pa.PixelDDPMPurifier._POOL.clear()
    p = pa.build_purifier("diffpure_ddpm256_t100", {"eval": {"purifiers": {
        "diffpure_ddpm256_t100": {"model_id": "/nonexistent/path/ddpm"}}}})
    with pytest.raises(pa.NotAvailable):
        p.load("cpu")
    with pytest.raises(pa.NotAvailable):
        p.apply(img(), 0)


def test_held_tier_expands_with_new_names():
    from src.eval import edit_runner as er
    ev = er.load_eval_config()["eval"]
    jobs = er.expand_tier(ev, "HELD", "v2_final", "test", ["a", "b"], ["diffpure_ddpm256_t100"], None)
    assert jobs and {j.purifier for j in jobs} == {"diffpure_ddpm256_t100"}
    allp = {j.purifier for j in er.expand_tier(ev, "HELD", "clean", "test", ["a"], None, None)}
    assert allp == {"downup_256", "diffpure_ddpm256_t100", "diffpure_ddpm256_t200"}
