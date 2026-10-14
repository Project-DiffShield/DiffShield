import numpy as np

from src.eval.stats import bootstrap_ci, paired_bootstrap_diff, summarize


def test_ci_properties():
    x = np.random.default_rng(0).normal(5, 1, 50)
    m, lo, hi = bootstrap_ci(x, n_boot=500)
    assert lo <= m <= hi and (m, lo, hi) == bootstrap_ci(x, n_boot=500)
    m, lo, hi = bootstrap_ci(np.ones(10), n_boot=100)
    assert lo == hi == m == 1.0
    d = paired_bootstrap_diff(x, x.copy(), n_boot=100)
    assert d["mean_diff"] == 0 and d["ci_lo"] == 0 and d["ci_hi"] == 0
    s = summarize(x, n_boot=100)
    assert s["n"] == 50 and abs(s["std"] - x.std()) < 1e-12
