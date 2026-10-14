import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from src.eval import masks as M


def test_bbox_mask_keeps_face_and_repaints_corners():
    bbox = (180, 150, 330, 330)
    m = M.repaint_mask_from_bbox((512, 512), bbox)
    assert m.shape == (512, 512) and set(np.unique(m)) <= {0, 255}
    face = M.face_ellipse((512, 512), bbox)
    assert (m[face] == 0).all()                # face never repainted
    assert m[0, 0] == 255 and m[-1, -1] == 255 and m[0, -1] == 255 and m[-1, 0] == 255
    assert (m == 255).mean() > 0.5             # most of the portrait background is repainted


def test_ensure_face_unmasked_overrides():
    m = np.full((100, 100), 255, np.uint8)
    out = M.ensure_face_unmasked(m, (30, 30, 70, 70))
    assert out[50, 50] == 0 and out[0, 0] == 255
    assert (m == 255).all()                    # input untouched


def test_parsing_mask(tmp_path):
    d = tmp_path / "0"
    d.mkdir()
    sk = np.zeros((512, 512), np.uint8); sk[100:400, 150:350] = 255
    hair = np.zeros((512, 512), np.uint8); hair[50:100, 150:350] = 255
    cloth = np.zeros((512, 512), np.uint8); cloth[400:512, :] = 255
    for n, a in (("skin", sk), ("hair", hair), ("cloth", cloth)):
        Image.fromarray(a).save(d / f"00007_{n}.png")
    m = M.repaint_mask_from_parsing(str(tmp_path), 7, 512)
    assert m[250, 250] == 0           # skin kept
    assert m[75, 250] == 255          # hair is repainted by default (not a face part)
    m2 = M.repaint_mask_from_parsing(str(tmp_path), 7, 512, keep_parts=M.KEEP_PARTS_DEFAULT + ("hair",))
    assert m2[75, 250] == 0           # configurable: hair kept
    m3 = M.repaint_mask_from_parsing(str(tmp_path), 7, 512, dilate_px=10)
    assert m3[95, 250] == 0 and m3[75, 250] == 255   # dilation grows the keep region around skin
    assert m[450, 250] == 255         # cloth repainted
    assert m[10, 10] == 255           # unlabeled background repainted
    assert M.repaint_mask_from_parsing(str(tmp_path), 4001, 512) is None   # folder 2 missing


def test_mask_provider_caches_and_falls_back(tmp_path):
    calls = []

    def bbox(i):
        calls.append(i)
        return (180, 150, 330, 330)

    mp = M.MaskProvider(str(tmp_path / "masks"), parsing_dir=None, size=512, face_bbox_fn=bbox)
    m1 = mp("abc")
    m2 = mp("abc")
    assert np.array_equal(np.asarray(m1), np.asarray(m2)) and len(calls) == 1   # second call from disk cache
    a = np.asarray(m1)
    assert a[240, 250] == 0 and a[5, 5] == 255
