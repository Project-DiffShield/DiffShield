import os
import subprocess
import sys

from src.utils.seeding import derive_seed


def test_derive_seed_properties():
    assert derive_seed(2026, "protect", "a") == derive_seed(2026, "protect", "a")
    assert derive_seed(2026, "protect", "a") != derive_seed(2026, "protect", "b")
    assert derive_seed(2026, "protect", "a") != derive_seed(2026, "purify", "a")
    assert 0 <= derive_seed(1, "x") < 2 ** 31


def test_derive_seed_across_processes(repo_root):
    code = "from src.utils.seeding import derive_seed; print(derive_seed(2026,'protect','1600'))"
    outs = set()
    for h in (1, 2):
        env = dict(os.environ, PYTHONHASHSEED=str(h), PYTHONPATH=str(repo_root))
        outs.add(subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True,
                                env=env).stdout.strip())
    assert outs == {str(derive_seed(2026, "protect", "1600"))}
