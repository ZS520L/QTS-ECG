"""Threshold-free and best-threshold detection metrics.

* **AUROC** - primary metric.
* **AUPRC** - average precision with *abnormal* as the positive class. Its
  chance level equals the abnormal prevalence of the test set (about 0.57
  on PTB-XL and 0.86 on CPSC 2018), which is reported alongside.
* **Best F1** - maximum F1 over all thresholds (an optimistic, oracle-threshold
  statistic). Because abnormal records are the majority class, predicting
  "abnormal" for everything already gives ``F1 = 2p / (1 + p)``; this
  trivial level is reported as ``f1_trivial``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def best_f1(labels: np.ndarray, scores: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    i = int(np.argmax(f1))
    thr = thresholds[i] if i < len(thresholds) else 0.0
    return float(f1[i]), float(thr), float(precision[i]), float(recall[i])


def evaluate_scores(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    out = {"n_test": int(labels.size), "n_test_abnormal": int(labels.sum()),
           "n_nonfinite_scores": int((~finite).sum())}
    if not finite.all():
        # Non-finite scores are ranked as most anomalous so a diverged model
        # cannot silently look good.
        fill = scores[finite].max() + 1.0 if finite.any() else 0.0
        scores = np.where(finite, scores, fill)
    p = labels.mean()
    f1, thr, prec, rec = best_f1(labels, scores)
    out.update({
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "best_f1": f1,
        "threshold": thr,
        "precision": prec,
        "recall": rec,
        "prevalence": float(p),
        "f1_trivial": float(2 * p / (1 + p)),
    })
    return out
