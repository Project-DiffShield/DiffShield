"""S05 regression: run_protect (list_images) and evaluate (read_split) must pick the same first N images."""
import types

from scripts.evaluate import read_split
from src.utils.io import list_images


def test_limit_picks_same_images(tmp_path):
    names = ["13117", "1172", "16383", "862", "4334", "10001", "999"]
    d = tmp_path / "imgs"
    d.mkdir()
    for n in names:
        (d / f"{n}.png").write_bytes(b"x")
    lst = tmp_path / "test.txt"
    lst.write_text("\n".join(n + ".jpg" for n in names))
    a = types.SimpleNamespace(split_file=str(lst), split="test", source_dir=str(d), limit=4)
    ev = read_split(a, {})
    prot = [p.stem for p in list_images(d, lst, limit=4)]
    assert ev == prot
