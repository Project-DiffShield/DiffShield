import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture
def repo_root():
    return REPO


@pytest.fixture
def synth_dir(tmp_path):
    """3 synthetic 64px JPEG 'faces' (smooth gradients + texture)."""
    d = tmp_path / "clean"
    d.mkdir()
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    for i in range(3):
        base = np.stack([xx * 3 + i * 20, yy * 3 + i * 10, (xx + yy) * 2], axis=-1).astype(np.float32)
        base += rng.normal(0, 4, base.shape)
        Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(d / f"img{i}.jpg")
    return d
