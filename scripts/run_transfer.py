"""Zero-shot cross-dataset transfer (no fine-tuning).

For every seed, a model trained on dataset A (checkpoint from the benchmark)
scores the *test split of dataset B for the same seed*. The in-domain result
on exactly the same test records is the benchmark run trained on B, which
makes the two numbers directly comparable.

Output: ``results/transfer/transfer.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.config import load_config, resolve_path  # noqa: E402
from qts.data import build_subsets, load_processed  # noqa: E402
from qts.engine import predict_scores  # noqa: E402
from qts.evaluation import evaluate_scores  # noqa: E402
from qts.models import build_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--processed-root", help="override data.processed_root")
    ap.add_argument("--methods", nargs="+", default=["qts", "deep_svdd", "vae"])
    ap.add_argument("--bench", default=None)
    ap.add_argument("--out", default="results/transfer")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    bench = resolve_path(args.bench or cfg["benchmark"]["output_dir"])
    seeds = cfg["benchmark"]["seeds"]
    datasets = cfg["benchmark"]["datasets"]
    processed = resolve_path(args.processed_root or cfg["data"]["processed_root"])
    ratios = tuple(cfg["data"]["split"]["ratios"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = []
    for tgt in datasets:
        ds = load_processed(processed, tgt)
        for seed in seeds:
            subsets, _ = build_subsets(ds, seed, ratios)
            test_ld = DataLoader(subsets["test"], batch_size=64, shuffle=False)
            for src in datasets:
                for m in args.methods:
                    ckpt = bench / src / m / f"seed_{seed}" / "model.pt"
                    if not ckpt.exists():
                        continue
                    mcfg = cfg["methods"][m]
                    model = build_model(mcfg.get("arch", m), mcfg["params"])
                    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                    model.to(device)
                    torch.manual_seed(seed)  # VAE scoring samples the latent
                    scores, labels, _ = predict_scores(model, test_ld, device)
                    met = evaluate_scores(labels, scores)
                    rows.append({"method": m, "source": src, "target": tgt, "seed": seed,
                                 "in_domain": src == tgt, "auroc": met["auroc"],
                                 "auprc": met["auprc"], "best_f1": met["best_f1"]})
                    print(json.dumps(rows[-1]), flush=True)
            del subsets
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "transfer.csv", index=False)


if __name__ == "__main__":
    main()
