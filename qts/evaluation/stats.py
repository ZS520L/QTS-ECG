"""Aggregation and paired statistics over seeds."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_std(values) -> tuple[float, float]:
    """Mean and *sample* standard deviation (ddof=1)."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def paired_comparison(a, b) -> dict:
    """Compare paired per-seed values ``a`` (proposed) and ``b`` (baseline).

    Returns the mean difference with a 95% t-interval, the number of seeds in
    which ``a > b``, and Wilcoxon signed-rank p-values (exact for n <= 25).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    out = {"n": int(n), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
           "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}
    if n > 1:
        se = d.std(ddof=1) / np.sqrt(n)
        h = stats.t.ppf(0.975, n - 1) * se
        out.update({"ci95_low": float(d.mean() - h), "ci95_high": float(d.mean() + h)})
    if n > 0 and np.any(d != 0):
        out["p_wilcoxon_two_sided"] = float(stats.wilcoxon(a, b, alternative="two-sided").pvalue)
        out["p_wilcoxon_greater"] = float(stats.wilcoxon(a, b, alternative="greater").pvalue)
    else:
        out["p_wilcoxon_two_sided"] = out["p_wilcoxon_greater"] = 1.0
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (family = the given dict)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj
