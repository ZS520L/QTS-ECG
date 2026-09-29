import numpy as np

from qts.evaluation.metrics import evaluate_scores


def test_perfect_and_trivial():
    y = np.array([0, 0, 1, 1, 1])
    m = evaluate_scores(y, np.array([0.1, 0.2, 0.8, 0.9, 1.0]))
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0 and abs(m["best_f1"] - 1) < 1e-6
    assert abs(m["prevalence"] - 0.6) < 1e-12
    assert abs(m["f1_trivial"] - 0.75) < 1e-12


def test_constant_scores_give_chance_auroc_and_trivial_f1():
    y = np.array([0, 1, 1, 0, 1, 1])
    m = evaluate_scores(y, np.zeros(6))
    assert m["auroc"] == 0.5
    assert abs(m["best_f1"] - m["f1_trivial"]) < 1e-6


def test_nonfinite_scores_are_ranked_most_anomalous():
    y = np.array([0, 0, 1, 1])
    m = evaluate_scores(y, np.array([0.1, np.nan, 0.5, 0.6]))
    assert m["n_nonfinite_scores"] == 1
    assert m["auroc"] < 1.0
