"""Half B (attack-and-score harness). Must never import src.protect.

`Evaluator` and `calculate_psnr` live in src.legacy_eval (moved from src/eval.py)
and are re-exported lazily so that importing this package does not import
diffusers/cv2.
"""

_LEGACY = ("Evaluator", "calculate_psnr")


def __getattr__(name):
    if name in _LEGACY:
        from src import legacy_eval
        return getattr(legacy_eval, name)
    raise AttributeError(f"module 'src.eval' has no attribute {name!r}")
