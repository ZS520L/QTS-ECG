import numpy as np

from qts.data.splits import patient_level_split


def _toy(n_patients=300, seed=0):
    rng = np.random.default_rng(seed)
    recs = rng.integers(1, 5, n_patients)
    pids = np.repeat(np.arange(n_patients), recs)
    labels = rng.integers(0, 2, len(pids))
    return labels, pids


def test_partitions_are_patient_disjoint_and_normal_only_for_training():
    labels, pids = _toy()
    tr, va, te = patient_level_split(labels, pids, seed=3)
    assert set(pids[tr]).isdisjoint(pids[va])
    assert set(pids[tr]).isdisjoint(pids[te])
    assert set(pids[va]).isdisjoint(pids[te])
    assert (labels[tr] == 0).all() and (labels[va] == 0).all()
    # the test set keeps every record of its patients (normal and abnormal)
    test_patients = set(pids[te])
    assert set(te) == {i for i in range(len(pids)) if pids[i] in test_patients}


def test_ratios_and_determinism():
    labels, pids = _toy(1000)
    a = patient_level_split(labels, pids, seed=7)
    b = patient_level_split(labels, pids, seed=7)
    c = patient_level_split(labels, pids, seed=8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[2], c[2])
    n_test_patients = len(set(pids[a[2]]))
    assert abs(n_test_patients / 1000 - 0.15) < 0.01


def test_string_patient_ids():
    labels = np.array([0, 1, 0, 0, 1, 0] * 50)
    pids = np.array([f"A{i:04d}" for i in range(len(labels))])
    tr, va, te = patient_level_split(labels, pids, seed=0)
    assert len(tr) + len(va) + len(te) <= len(labels)
    assert len(set(tr) | set(va) | set(te)) == len(tr) + len(va) + len(te)
