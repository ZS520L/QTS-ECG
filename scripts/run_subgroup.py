"""Per-diagnosis AUROC: normal test records vs. records carrying each label.

Reads the saved test scores of the benchmark (no retraining) and the per-record
diagnoses in ``meta.csv``. A record with several diagnoses contributes to every
matching sub-group. Output: ``results/subgroup/subgroup_auroc.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.config import load_config, resolve_path  # noqa: E402
from qts.data.readers import SUBGROUPS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--processed-root", help="override data.processed_root")
    ap.add_argument("--bench", default=None, help="benchmark results directory")
    ap.add_argument("--out", default="results/subgroup")
    ap.add_argument("--min-count", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    bench = resolve_path(args.bench or cfg["benchmark"]["output_dir"])
    processed = resolve_path(args.processed_root or cfg["data"]["processed_root"])
    rows = []
    for ds, groups in SUBGROUPS.items():
        meta_path = processed / ds / "meta.csv"
        if not meta_path.exists():
            print(f"skip {ds}: {meta_path} missing")
            continue
        meta = pd.read_csv(meta_path)
        for score_file in sorted((bench / ds).glob("*/seed_*/scores.npz")):
            method, seed = score_file.parts[-3], int(score_file.parent.name.split("_")[1])
            z = np.load(score_file)
            scores, labels, idx = z["scores"], z["labels"].astype(int), z["test_idx"]
            m = meta.iloc[idx]
            normal = labels == 0
            for g, kind in groups.items():
                pos = (m[g].to_numpy() == 1) & (labels == 1)
                if pos.sum() < args.min_count:
                    continue
                mask = normal | pos
                rows.append({"dataset": ds, "method": method, "seed": seed, "subgroup": g,
                             "type": kind, "n_normal": int(normal.sum()), "n_group": int(pos.sum()),
                             "auroc": roc_auc_score(labels[mask], scores[mask])})
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "subgroup_auroc.csv", index=False)
    print(f"wrote {len(df)} rows to {out / 'subgroup_auroc.csv'}")


if __name__ == "__main__":
    main()
