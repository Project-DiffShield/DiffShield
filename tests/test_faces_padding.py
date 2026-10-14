"""S05: insightface misses full-frame faces; FaceScorer pads the image and maps boxes back."""
import types

import numpy as np
from PIL import Image

from src.eval.faces import FaceScorer


class _App:
    def __init__(self):
        self.seen = None

    def get(self, arr):
        self.seen = arr.shape
        f = types.SimpleNamespace(bbox=np.array([140.0, 150.0, 340.0, 350.0]), det_score=0.9,
                                  normed_embedding=np.ones(4, dtype=np.float32) / 2, kps=np.array([[200.0, 220.0]]))
        return [f]


def test_pad_then_shift_back():
    sc = FaceScorer(pad_frac=0.25)
    sc._app = _App()
    r = sc.analyze(Image.new("RGB", (100, 100)))
    assert sc._app.seen[:2] == (150, 150)                    # 100 + 2 * 25
    assert r.bbox == (115.0, 125.0, 315.0, 325.0)            # padded coords minus 25
    assert np.allclose(r.kps, [[175.0, 195.0]])
