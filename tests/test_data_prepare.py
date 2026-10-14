import importlib.util
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PD = _load("prepare_data")


def make_jpgs(d, names, size=96):
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(0)
    for n in names:
        a = (rng.rand(size, size, 3) * 255).astype(np.uint8)
        Image.fromarray(a).save(os.path.join(d, n), quality=95)


def test_messy_params_deterministic_and_in_range():
    a, b = PD.messy_params(2026, "12.jpg"), PD.messy_params(2026, "12.jpg")
    assert a == b
    assert PD.MESSY_SCALE[0] <= a["scale"] <= PD.MESSY_SCALE[1]
    assert PD.MESSY_JPEG_Q[0] <= a["jpeg_quality"] <= PD.MESSY_JPEG_Q[1]
    assert PD.messy_params(2026, "13.jpg") != a or PD.messy_params(2026, "14.jpg") != a


def test_degrade_changes_image_but_keeps_size():
    img = Image.fromarray((np.random.RandomState(1).rand(512, 512, 3) * 255).astype(np.uint8))
    out = PD.degrade_messy(img, {"scale": 0.7, "jpeg_quality": 55})
    assert out.size == (512, 512)
    assert np.abs(np.asarray(out).astype(int) - np.asarray(img).astype(int)).mean() > 1
    out2 = PD.degrade_messy(img, {"scale": 0.7, "jpeg_quality": 55})
    assert np.array_equal(np.asarray(out), np.asarray(out2))


def test_prepare_split_png_512_lossless_and_idempotent(tmp_path):
    src, out = str(tmp_path / "src"), str(tmp_path / "out")
    make_jpgs(src, ["1.jpg", "2.jpg"])
    rows = PD.prepare_split("test", ["1.jpg", "2.jpg"], src, out, 2026)
    p = os.path.join(out, "test", "1.png")
    im = Image.open(p)
    assert im.size == (512, 512) and im.format == "PNG" and im.mode == "RGB"
    ref = np.asarray(Image.open(os.path.join(src, "1.jpg")).convert("RGB").resize((512, 512), Image.BICUBIC))
    assert np.array_equal(np.asarray(im), ref)
    rows2 = PD.prepare_split("test", ["1.jpg", "2.jpg"], src, out, 2026)
    assert [r["sha256"] for r in rows] == [r["sha256"] for r in rows2]
    assert rows[0]["params"] == ""


def test_prepare_messy_records_params(tmp_path):
    src, out = str(tmp_path / "src"), str(tmp_path / "out")
    make_jpgs(src, ["7.jpg"])
    rows = PD.prepare_split("messy", ["7.jpg"], src, out, 2026)
    assert '"jpeg_quality"' in rows[0]["params"]
    clean_ref = np.asarray(Image.open(os.path.join(src, "7.jpg")).convert("RGB").resize((512, 512), Image.BICUBIC))
    got = np.asarray(Image.open(os.path.join(out, "messy", "7.png")))
    assert not np.array_equal(got, clean_ref)


def test_main_end_to_end_no_cache(tmp_path):
    root = tmp_path / "ds"
    img = root / "CelebA-HQ-img"
    make_jpgs(str(img), [f"{i}.jpg" for i in range(6)])
    (root / "CelebAMask-HQ-attribute-anno.txt").write_text("6\nA\n" + "".join(f"{i}.jpg 1\n" for i in range(6)), encoding="utf-8")
    sp = tmp_path / "splits"
    sp.mkdir()
    for k, v in {"test": [0, 1], "messy": [2], "train": [3], "val": [4], "ae_pool": [5]}.items():
        (sp / f"{k}.txt").write_text("".join(f"{i}.jpg\n" for i in v), encoding="utf-8")
    out = tmp_path / "out"
    rc = PD.main(["--celebahq_root", str(root), "--splits_dir", str(sp), "--out_dir", str(out), "--no_cache"])
    assert rc == 0
    assert (out / "manifest.csv").is_file() and (out / "messy_degradation.json").is_file()
    assert (out / "ae_pool" / "5.png").is_file()
