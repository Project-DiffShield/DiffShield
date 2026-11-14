"""scripts/restore_prot.py: finds prot_<method>_test under a root, copies it, fails loudly when short."""
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("restore_prot", REPO / "scripts" / "restore_prot.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)


def make(root, n, name="prot_v2_next_tuned_test", extra=0):
    d = root / "T04" / "results" / name / "images"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"{i}.png").write_bytes(b"x")
    for i in range(extra):
        (d / f"{i}.txt").write_bytes(b"x")
    (root / "T04" / "results" / name / "metrics.csv").write_text("a\n")


def test_restore_ok(tmp_path):
    make(tmp_path / "in", 100)
    dst = tmp_path / "W1" / "results" / "prot_v2_next_tuned_test"
    assert rp.restore(str(tmp_path / "in"), "prot_v2_next_tuned_test", str(dst), 100) == 0
    assert len(list((dst / "images").glob("*.png"))) == 100 and (dst / "metrics.csv").is_file()


def test_fewer_than_expected_fails(tmp_path, capsys):
    make(tmp_path / "in", 99)
    assert rp.restore(str(tmp_path / "in"), "prot_v2_next_tuned_test", str(tmp_path / "o"), 100) == 1
    assert "expected at least 100" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_missing_folder_and_empty_src_fail(tmp_path):
    (tmp_path / "in").mkdir()
    assert rp.restore(str(tmp_path / "in"), "prot_v2_next_tuned_test", str(tmp_path / "o"), 5) == 1
    assert rp.restore("", "prot_x", str(tmp_path / "o"), 5, fallback_root=str(tmp_path / "nope")) == 1


def test_empty_src_uses_fallback_and_smoke_minimum(tmp_path):
    make(tmp_path / "in", 100)
    assert rp.restore("", "prot_v2_next_tuned_test", str(tmp_path / "o"), 5, fallback_root=str(tmp_path / "in")) == 0
