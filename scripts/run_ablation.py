"""Template-length / aggregation ablation of QTS (Table 5 of the paper).

Uses exactly the training protocol, seeds and splits of the main benchmark.
The single-bank K=155 configuration *is* the main QTS model, so its results
are linked from ``results/benchmark`` instead of being recomputed.

    python scripts/run_ablation.py [--variants qts_k35 qts_ms_sum] [--seeds 0 1]
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.config import load_config, resolve_path  # noqa: E402
from qts.data import build_subsets, load_processed  # noqa: E402
from qts.engine import run_experiment  # noqa: E402
from qts.utils import environment_info, save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ablation.yaml")
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--processed-root", help="override data.processed_root")
    ap.add_argument("--max-epochs", type=int)
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    acfg = cfg["ablation"]
    reuse = acfg.get("reuse_from_benchmark", {})
    variants = args.variants or [v for v in acfg["variants"] if v not in reuse]
    datasets = args.datasets or acfg["datasets"]
    seeds = args.seeds if args.seeds is not None else acfg["seeds"]
    out_root = resolve_path(args.out or acfg["output_dir"])
    tcfg = dict(cfg["training"])
    if args.max_epochs:
        tcfg["max_epochs"] = args.max_epochs
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                          else "cpu")
    base = cfg["methods"][acfg["base"]]

    out_root.mkdir(parents=True, exist_ok=True)
    save_json({"config": cfg, "variants": variants, "seeds": seeds, "datasets": datasets,
               "training": tcfg, "environment": environment_info(),
               "command": " ".join(sys.argv)}, out_root / "run_info.json")
    processed_root = resolve_path(args.processed_root or cfg["data"]["processed_root"])
    ratios = tuple(cfg["data"]["split"]["ratios"])

    for ds_name in datasets:
        ds = load_processed(processed_root, ds_name)
        for seed in seeds:
            todo = [v for v in variants
                    if not (out_root / ds_name / v / f"seed_{seed}" / "metrics.json").exists()]
            if not todo:
                continue
            subsets, indices = build_subsets(ds, seed, ratios)
            for v in todo:
                spec = dict(acfg["variants"][v])
                mcfg = copy.deepcopy(base)
                mcfg["arch"] = "qts"
                mcfg["label"] = spec.pop("label", v)
                mcfg["params"].update(spec)
                if args.max_epochs:
                    mcfg["max_epochs"] = args.max_epochs
                print(f"[{time.strftime('%H:%M:%S')}] {v} / {ds_name} / seed {seed}", flush=True)
                run_experiment(v, mcfg, tcfg, subsets, indices, ds_name, seed, device,
                               out_root / ds_name / v / f"seed_{seed}", save_checkpoint=False,
                               num_workers=cfg["data"]["loader"]["num_workers"])
            del subsets


if __name__ == "__main__":
    main()
