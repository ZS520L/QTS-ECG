"""Convert the raw PhysioNet releases into fixed-size NumPy arrays.

Output (per dataset) in ``<processed_root>/<dataset>/``::

    signals.npy      float32 (N, 12, 5000)  band-passed, 10 s, per-lead z-scored
    labels.npy       int64   (N,)           0 = normal, 1 = abnormal
    patient_ids.npy          (N,)           grouping key for the patient-level split
    meta.csv                                one row per record (ids, fs, diagnoses)
    prepare_info.json                       settings, counts and SHA-256 checksums

Usage::

    python scripts/prepare_data.py --dataset ptbxl cpsc2018 \
        --ptbxl-dir /path/to/ptb-xl-1.0.3 --cpsc-dir /path/to/cpsc_2018
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.config import load_config, resolve_path  # noqa: E402
from qts.data.preprocessing import preprocess_record  # noqa: E402
from qts.data.readers import list_records, load_signal  # noqa: E402
from qts.utils import save_json  # noqa: E402

_PP: dict = {}


def _init_worker(pp: dict) -> None:
    _PP.update(pp)


def _process(task: dict):
    sig, fs = load_signal(task)
    if abs(fs - _PP["fs"]) > 1e-6:
        raise ValueError(f"{task['path']}: fs={fs}, expected {_PP['fs']}")
    x = preprocess_record(sig, fs=fs, target_len=_PP["seq_len"], low=_PP["low"],
                          high=_PP["high"], normalize=_PP["normalize"])
    return x, int(sig.shape[1]), bool(np.isfinite(x).all())


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def prepare(name: str, raw_dir: str, out_root: Path, pp: dict, workers: int) -> None:
    t0 = time.time()
    tasks = list_records(name, raw_dir)
    n = len(tasks)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] {n} records from {raw_dir}", flush=True)

    signals = np.lib.format.open_memmap(out / "signals.npy", mode="w+", dtype=np.float32,
                                        shape=(n, 12, pp["seq_len"]))
    meta_rows = []
    n_bad = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(pp,)) as ex:
        for i, (x, n_samples, finite) in enumerate(ex.map(_process, tasks, chunksize=32)):
            signals[i] = x
            row = dict(tasks[i]["meta"])
            row["n_samples_raw"] = n_samples
            meta_rows.append(row)
            n_bad += int(not finite)
            if (i + 1) % 2000 == 0 or i + 1 == n:
                print(f"  {i + 1}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    signals.flush()
    del signals

    meta = pd.DataFrame(meta_rows)
    labels = meta["label"].to_numpy(dtype=np.int64)
    if name == "ptbxl":
        patient_ids = meta["patient_id"].to_numpy(dtype=np.int64)
    else:
        patient_ids = np.array(meta["patient_id"].astype(str).tolist())
    np.save(out / "labels.npy", labels)
    np.save(out / "patient_ids.npy", patient_ids)
    meta.to_csv(out / "meta.csv", index=False)

    info = {
        "dataset": name,
        "raw_dir": str(raw_dir),
        "preprocessing": pp,
        "n_records": n,
        "n_normal": int((labels == 0).sum()),
        "n_abnormal": int((labels == 1).sum()),
        "n_patients": int(len(np.unique(patient_ids))),
        "n_nonfinite_records": n_bad,
        "seconds": round(time.time() - t0, 1),
        "sha256": {f: sha256_file(out / f) for f in
                   ("signals.npy", "labels.npy", "patient_ids.npy")},
    }
    save_json(info, out / "prepare_info.json")
    print(f"[{name}] done: normal={info['n_normal']} abnormal={info['n_abnormal']} "
          f"patients={info['n_patients']} non-finite={n_bad} ({info['seconds']}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", nargs="+", default=["ptbxl", "cpsc2018"])
    ap.add_argument("--ptbxl-dir", default=None, help="overrides data.raw_dirs.ptbxl")
    ap.add_argument("--cpsc-dir", default=None, help="overrides data.raw_dirs.cpsc2018")
    ap.add_argument("--processed-root", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))["data"]
    raw = dict(cfg["raw_dirs"])
    if args.ptbxl_dir:
        raw["ptbxl"] = args.ptbxl_dir
    if args.cpsc_dir:
        raw["cpsc2018"] = args.cpsc_dir
    out_root = resolve_path(args.processed_root or cfg["processed_root"])
    pp = {k: cfg["preprocessing"][k] for k in ("fs", "seq_len", "low", "high", "normalize")}

    for name in args.dataset:
        prepare(name, str(resolve_path(raw[name])), out_root, pp, args.workers)


if __name__ == "__main__":
    main()
