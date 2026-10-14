"""demo.py --cpu-dry-run: tiny/fake models, checks the wiring and the written artefacts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def test_demo_cpu_dry_run(tmp_path):
    spec = importlib.util.spec_from_file_location("demo", REPO / "scripts" / "demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    yy, xx = np.mgrid[0:90, 0:120]
    img = tmp_path / "face.jpg"
    Image.fromarray(np.stack([xx * 2, yy * 2, xx + yy], -1).astype(np.uint8)).save(img)
    out = tmp_path / "out"
    assert demo.main(["--image", str(img), "--out", str(out), "--cpu-dry-run"]) == 0
    for f in ("source.png", "protected.png", "edit_of_original.png", "edit_of_protected.png", "panel.png", "demo.json"):
        assert (out / f).is_file(), f
    res = json.loads((out / "demo.json").read_text())
    assert res["dry_run"] is True and res["method_requested"] == "v2_final" and res["method_used"].startswith("v1")
    assert res["max_abs_gray_level_change"] <= 4                    # eps 4/255 on the published uint8 image
    a, b = np.asarray(Image.open(out / "source.png")), np.asarray(Image.open(out / "protected.png"))
    assert a.shape == b.shape and a.shape[0] == 64
