"""Main benchmark: every method x dataset x seed (Table 3 of the paper).

Runs are resumable - completed runs (with ``metrics.json``) are skipped - so
the sweep can be interrupted and restarted at any time. Seeds are the outer
loop, so paired comparisons become available early.

Examples::

    python scripts/run_benchmark.py                                  # full sweep
    python scripts/run_benchmark.py --methods qts deep_svdd --seeds 0 1
    python scripts/run_benchmark.py --datasets cpsc2018 --max-epochs 2 --out results/smoke
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out", help="output directory (default: benchmark.output_dir)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--processed-root", help="override data.processed_root")
    ap.add_argument("--max-epochs", type=int, help="override (smoke tests only)")
    ap.add_argument("--batch-size", type=int, help="override (smoke tests only)")
    ap.add_argument("--no-checkpoints", action="store_true")
    args = ap.parse_args()

    cfg = load_config(resolve_path(args.config))
    bcfg = cfg["benchmark"]
    methods = args.methods or bcfg["methods"]
    datasets = args.datasets or bcfg["datasets"]
    seeds = args.seeds if args.seeds is not None else bcfg["seeds"]
    out_root = resolve_path(args.out or bcfg["output_dir"])
    tcfg = dict(cfg["training"])
    if args.max_epochs:
        tcfg["max_epochs"] = args.max_epochs
    dev = args.device or bcfg.get("device", "cuda")
    device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
    unknown = [m for m in methods if m not in cfg["methods"]]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")

    out_root.mkdir(parents=True, exist_ok=True)
    save_json({"config": cfg, "methods": methods, "datasets": datasets, "seeds": seeds,
               "training": tcfg, "environment": environment_info(),
               "command": " ".join(sys.argv)}, out_root / "run_info.json")

    processed_root = resolve_path(args.processed_root or cfg["data"]["processed_root"])
    ratios = tuple(cfg["data"]["split"]["ratios"])
    t_all = time.time()
    for ds_name in datasets:
        ds = load_processed(processed_root, ds_name)
        print(f"\n=== {ds_name}: {len(ds.labels)} records, normal={(ds.labels == 0).sum()}, "
              f"abnormal={(ds.labels == 1).sum()} ===", flush=True)
        for seed in seeds:
            todo = [m for m in methods
                    if not (out_root / ds_name / m / f"seed_{seed}" / "metrics.json").exists()]
            if not todo:
                continue
            subsets, indices = build_subsets(ds, seed, ratios)
            print(f"\n--- {ds_name} seed {seed}: train={len(subsets['train'])} (normal) "
                  f"val={len(subsets['val'])} (normal) test={len(subsets['test'])} "
                  f"(abnormal={int(subsets['test'].labels.sum())}) ---", flush=True)
            for m in todo:
                mcfg = dict(cfg["methods"][m])
                if args.max_epochs and mcfg["kind"] == "deep":
                    mcfg["max_epochs"] = args.max_epochs
                if args.batch_size:
                    mcfg["batch_size"] = args.batch_size
                print(f"[{time.strftime('%H:%M:%S')}] {m} / {ds_name} / seed {seed}", flush=True)
                run_experiment(m, mcfg, tcfg, subsets, indices, ds_name, seed, device,
                               out_root / ds_name / m / f"seed_{seed}",
                               save_checkpoint=bcfg.get("save_checkpoints", True)
                               and not args.no_checkpoints,
                               num_workers=cfg["data"]["loader"]["num_workers"])
            del subsets
    print(f"\nAll done in {(time.time() - t_all) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
