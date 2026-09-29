"""Re-implementations of the 12 unsupervised baselines used in the paper.

All deep baselines expose the same interface as QTS::

    loss  = model.compute_loss(x)       # scalar, minimised on normal data
    score = model.anomaly_score(x)      # (B,), higher = more anomalous
    model.initialize(loader, device)    # optional data-dependent init

Several baselines are *simplified* adaptations of the original methods to
10-s, 12-lead, 500-Hz recordings; the differences are documented in each
module docstring and in ``docs/BASELINES.md``.
"""
