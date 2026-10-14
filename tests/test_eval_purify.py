"""Cheap purifiers are deterministic and well-behaved; registry/aliases/plug-in contract. No models."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PIL import Image

from src.eval import purify_attacks as pa


def make_img(w=64, h=64, seed=0):
    rng = np.random.RandomState(seed)
    base = np.linspace(0, 255, w)[None, :, None].repeat(h, 0).repeat(3, 2)
    return Image.fromarray(np.clip(base + rng.randn(h, w, 3) * 10, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("name", ["jpeg_q90", "jpeg_q75", "jpeg_q50", "jpeg_other", "downup_0.5", "downup_other", "none"])
def test_cheap_purifiers_deterministic_and_same_size(name):
    img = make_img()
    p = pa.build_purifier(name)
    a, b = p.apply(img, 1), p.apply(img, 999)       # seed-independent
    assert a.size == img.size and a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert np.array_equal(np.asarray(a), np.asarray(pa.build_purifier(name).apply(img, 1)))


def test_jpeg_quality_ordering_and_changes_pixels():
    img = make_img(128, 128)
    err = {}
    for n in ("jpeg_q90", "jpeg_q75", "jpeg_q50"):
        out = pa.build_purifier(n).apply(img, 0)
        err[n] = np.abs(np.asarray(out, float) - np.asarray(img, float)).mean()
    assert err["jpeg_q90"] < err["jpeg_q75"] < err["jpeg_q50"]
    assert err["jpeg_q90"] > 0


def test_jpeg_matches_pil_directly():
    import io
    img = make_img()
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    ref = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    assert np.array_equal(np.asarray(ref), np.asarray(pa.build_purifier("jpeg_q75").apply(img, 0)))


def test_downup_loses_high_frequencies():
    rng = np.random.RandomState(0)
    noise = Image.fromarray(rng.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    out = np.asarray(pa.build_purifier("downup_0.5").apply(noise, 0), float)
    assert out.std() < np.asarray(noise, float).std()
    assert pa.build_purifier("downup_other").apply(noise, 0).size == (64, 64)


def test_aliases_and_unknown():
    assert pa.canonical_name("jpeg_q_other") == "jpeg_other"
    assert pa.canonical_name("diffpure_t0.2") == "diffpure_0.2"
    assert pa.canonical_name("vae_rt_sd15") == "vae_rt"
    assert pa.build_purifier("jpeg_q_other").name == "jpeg_other"
    with pytest.raises(pa.NotAvailable):
        pa.build_purifier("does_not_exist")


def test_registry_contains_required_names():
    names = set(pa.available_names())
    need = {"none", "jpeg_q90", "jpeg_q75", "jpeg_q50", "jpeg_other", "downup_0.5", "downup_other", "vae_rt",
            "diffpure_0.05", "diffpure_0.1", "diffpure_0.2", "diffpure_0.3", "restore_swin2sr", "gridpure", "adaptive_ae"}
    assert need <= names


def test_model_purifiers_construct_without_loading():
    cfg = {"eval": {"models": {"sd15": {"id": "x/sd15"}, "swin2sr": {"id": "x/swin"}},
                    "purifiers": {"diffpure_0.1": {"steps": 10}}}}
    d = pa.build_purifier("diffpure_0.1", cfg)
    assert d.strength == 0.1 and d.steps == 10 and d.model_id == "x/sd15" and d.pipe is None
    assert pa.build_purifier("diffpure_t0.3", cfg).strength == 0.3
    assert pa.build_purifier("vae_rt", cfg).model_id == "x/sd15"
    assert pa.build_purifier("restore_swin2sr", cfg).model_id == "x/swin"
    # diffpure instances share a pipeline key (kept loaded together by the runner)
    assert d.share_key == pa.build_purifier("diffpure_0.2", cfg).share_key


def test_adaptive_ae_is_lazy_plugin_not_available_without_module():
    # src/eval/adaptive_ae.py is built by another agent; without it we get a clear NotAvailable
    import importlib
    try:
        importlib.import_module("src.eval.adaptive_ae")
        pytest.skip("adaptive_ae present")
    except ImportError:
        pass
    with pytest.raises(pa.NotAvailable, match="adaptive_ae"):
        pa.build_purifier("adaptive_ae", {})


def test_adaptive_ae_plugin_contract(monkeypatch):
    import types
    mod = types.ModuleType("src.eval.adaptive_ae")

    class P(pa.EvalPurifier):
        name = "adaptive_ae"

        def apply(self, img, seed):
            return img

    mod.build_purifier = lambda cfg, device: P()
    monkeypatch.setitem(sys.modules, "src.eval.adaptive_ae", mod)
    assert pa.build_purifier("adaptive_ae", {}).name == "adaptive_ae"
