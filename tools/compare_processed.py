"""Compare two preprocessed dataset directories array by array.

Used to verify that ``scripts/prepare_data.py`` reproduces a previously
generated set of arrays (e.g. the ones used for the paper)::

    python tools/compare_processed.py data/processed/ptbxl /old/preprocessed/ptbxl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def compare(a_dir: Path, b_dir: Path, chunk: int = 512) -> dict:
    out = {}
    for name in ("labels.npy", "patient_ids.npy"):
        a = np.load(a_dir / name, allow_pickle=True)
        b = np.load(b_dir / name, allow_pickle=True)
        out[name] = {"shape_a": list(a.shape), "shape_b": list(b.shape),
                     "equal": bool(a.shape == b.shape and np.array_equal(a.astype(str), b.astype(str)))}
    a = np.load(a_dir / "signals.npy", mmap_mode="r")
    b = np.load(b_dir / "signals.npy", mmap_mode="r")
    res = {"shape_a": list(a.shape), "shape_b": list(b.shape),
           "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}
    if a.shape == b.shape:
        max_abs, n_diff = 0.0, 0
        for i in range(0, a.shape[0], chunk):
            d = np.abs(np.asarray(a[i:i + chunk], np.float64) - np.asarray(b[i:i + chunk], np.float64))
            max_abs = max(max_abs, float(d.max()))
            n_diff += int((d > 0).sum())
        res.update({"max_abs_diff": max_abs, "n_differing_values": n_diff,
                    "bitwise_equal": n_diff == 0})
    out["signals.npy"] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("reference_dir", type=Path)
    args = ap.parse_args()
    res = compare(args.new_dir, args.reference_dir)
    print(json.dumps(res, indent=2))
    ok = res["labels.npy"]["equal"] and res["patient_ids.npy"]["equal"] and \
        res["signals.npy"].get("max_abs_diff", 1.0) < 1e-5
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
