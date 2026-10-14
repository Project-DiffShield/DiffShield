import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from PIL import Image

from src.eval import gridpure as G
from src.eval import purify_attacks as pa


def test_crop_boxes_cover_image():
    for hw in [(512, 512), (300, 400), (256, 256)]:
        boxes = G.get_crop_boxes(*hw)
        cov = np.zeros(hw, int)
        for x0, y0, x1, y1 in boxes:
            assert x1 - x0 == 256 and y1 - y0 == 256
            cov[y0:y1, x0:x1] += 1
        assert (cov > 0).all()
    assert len(G.get_crop_boxes(512, 512)) == 9


def test_corner_pack_roundtrip():
    arr = np.random.RandomState(0).rand(512, 512, 3).astype(np.float32)
    cb = G.get_corner_boxes(512, 512)
    canvas = G.pack_corners(arr, cb)
    assert canvas.shape == (256, 256, 3)
    for (x0, y0, x1, y1), p in zip(cb, G.unpack_corners(canvas, cb)):
        assert np.array_equal(p, arr[y0:y1, x0:x1])


def test_identity_patch_fn_is_identity():
    arr = np.random.RandomState(1).rand(512, 512, 3).astype(np.float32)
    out = G.gridpure(arr, lambda x: x, iters=3, gamma=0.1)
    assert np.allclose(out, arr, atol=1e-5)


def test_patch_fn_called_on_windows_plus_corner_canvas_and_blend():
    seen = []

    def fn(batch):
        seen.append(batch.shape)
        return np.zeros_like(batch)

    arr = np.ones((512, 512, 3), np.float32)
    out = G.gridpure(arr, fn, iters=2, gamma=0.1)
    assert seen == [(10, 256, 256, 3)] * 2        # 9 windows + 1 packed corner canvas
    # each iteration: cur = 0.1 * cur (purified is zero) -> 0.01 after two
    assert np.allclose(out, 0.01, atol=1e-6)


def test_purifier_class_with_fake_patch_fn_and_not_available():
    img = Image.fromarray(np.random.RandomState(0).randint(0, 256, (512, 512, 3), dtype=np.uint8))
    p = G.GridPurePurifier({"iters": 2, "gamma": 0.1}, patch_fn=lambda x: x)
    out = p.apply(img, 0)
    assert out.size == img.size
    assert np.abs(np.asarray(out, int) - np.asarray(img, int)).max() <= 1
    real = pa.build_purifier("gridpure", {"eval": {"purifiers": {"gridpure": {"checkpoint": None}}}})
    with pytest.raises(pa.NotAvailable, match="checkpoint"):
        real.load("cpu")
