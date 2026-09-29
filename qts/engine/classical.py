"""Classical one-class baselines on hand-crafted features (OC-SVM, Isolation Forest).

Features (288-d): for each of the 12 leads - mean, standard deviation,
skewness, excess kurtosis and the magnitudes of FFT bins 1-20 (0.1-2 Hz
at 10 s / 500 Hz). Features are standardised with statistics of the
training set only.
"""

from __future__ import annotations

import time

import numpy as np

from ..utils import set_seed


def extract_features(loader) -> tuple[np.ndarray, np.ndarray]:
    feats_all, labels_all = [], []
    for x, y in loader:
        x = x.numpy()
        b, c, _ = x.shape
        feats = np.zeros((b, c * 24), dtype=np.float32)
        for lead in range(c):
            sig = x[:, lead, :]
            base = lead * 24
            mu = sig.mean(axis=1)
            std = sig.std(axis=1) + 1e-8
            z = (sig - mu[:, None]) / std[:, None]
            feats[:, base + 0] = mu
            feats[:, base + 1] = std
            feats[:, base + 2] = (z ** 3).mean(axis=1)
            feats[:, base + 3] = (z ** 4).mean(axis=1) - 3.0
            feats[:, base + 4:base + 24] = np.abs(np.fft.rfft(sig, axis=1))[:, 1:21]
        feats_all.append(np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0))
        labels_all.append(y.numpy())
    return np.concatenate(feats_all), np.concatenate(labels_all)


def run_classical(method: str, params: dict, train_loader, test_loader, seed: int) -> dict:
    """Fit on training features and score the test set.

    Returns a dict with ``scores``, ``labels``, ``fit_time_sec`` and
    ``inference_time_sec``. Note that the training loader drops its last
    incomplete batch, exactly as for the deep methods.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    set_seed(seed)
    x_train, _ = extract_features(train_loader)
    x_test, y_test = extract_features(test_loader)
    scaler = StandardScaler().fit(x_train)
    x_train = np.nan_to_num(scaler.transform(x_train), nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(scaler.transform(x_test), nan=0.0, posinf=0.0, neginf=0.0)

    t0 = time.time()
    if method == "ocsvm":
        model = OneClassSVM(**params)
    elif method == "isolation_forest":
        model = IsolationForest(random_state=seed, n_jobs=-1, **params)
    else:
        raise ValueError(method)
    model.fit(x_train)
    fit_time = time.time() - t0
    t1 = time.time()
    scores = -model.decision_function(x_test).astype(np.float32)
    return {"scores": scores, "labels": y_test, "fit_time_sec": fit_time,
            "inference_time_sec": time.time() - t1}
