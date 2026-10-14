"""Half A (src/protect) and Half B (src/eval) must never import each other; baselines import only src.utils."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"


def _imports(py):
    tree = ast.parse(py.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                yield a.name
        elif isinstance(n, ast.ImportFrom) and n.module:
            yield ("." * n.level) + n.module


def _scan(pkg, forbidden):
    bad = []
    d = ROOT / pkg
    for py in (d.rglob("*.py") if d.exists() else []):
        for imp in _imports(py):
            if any(imp == f or imp.startswith(f + ".") for f in forbidden):
                bad.append((py.name, imp))
    return bad


def test_eval_never_imports_protect():
    assert _scan("eval", ["src.protect"]) == []


def test_protect_never_imports_eval():
    assert _scan("protect", ["src.eval"]) == []


def test_baselines_only_utils():
    assert _scan("baselines", ["src.eval", "src.protect", "src.losses", "src.optimizer", "src.eot"]) == []
