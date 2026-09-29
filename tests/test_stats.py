import numpy as np
import pytest

from qts.evaluation.stats import holm, mean_std, paired_comparison


def test_mean_std_uses_sample_sd():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0)
    assert s == pytest.approx(1.0)


def test_paired_comparison_exact_wilcoxon():
    a = np.array([0.9, 0.8, 0.85, 0.87, 0.91])
    b = a - np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    r = paired_comparison(a, b)
    assert r["wins"] == 5 and r["losses"] == 0
    assert r["mean_diff"] == pytest.approx(0.03)
    # exact test, n = 5, all positive: one-sided 1/32, two-sided 2/32
    assert r["p_wilcoxon_greater"] == pytest.approx(1 / 32)
    assert r["p_wilcoxon_two_sided"] == pytest.approx(2 / 32)
    assert r["ci95_low"] < 0.03 < r["ci95_high"]


def test_holm_monotone_and_capped():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)      # monotone: max(0.04*1, 0.06)
    assert max(holm({"x": 0.9, "y": 0.8}).values()) <= 1.0
