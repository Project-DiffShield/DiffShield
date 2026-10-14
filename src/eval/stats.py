"""Bootstrap statistics over images."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n_boot: int = 10000, ci: float = 0.95, seed: int = 2026):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(x.mean()), float(lo), float(hi)


def summarize(x, **kw) -> dict:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan")}
    m, lo, hi = bootstrap_ci(a, **kw)
    return {"n": int(a.size), "mean": m, "std": float(a.std(ddof=0)), "min": float(a.min()),
            "max": float(a.max()), "ci_lo": lo, "ci_hi": hi}


def paired_bootstrap_diff(a, b, **kw) -> dict:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must be aligned (same shape)")
    d = a - b
    ok = np.isfinite(d)
    d = d[ok]
    m, lo, hi = bootstrap_ci(d, **kw)
    return {"n": int(d.size), "mean_diff": m, "ci_lo": lo, "ci_hi": hi,
            "share_a_greater": float((d > 0).mean()) if d.size else float("nan")}


def per_image_aggregate(df, value_col, by=("image_id",)):
    """Average over seeds/prompts first, so the bootstrap unit is the image."""
    return df.groupby(list(by))[value_col].mean()
