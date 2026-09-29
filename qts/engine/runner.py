"""Run one (method, dataset, seed) experiment and persist every artefact.

Output directory layout::

    <out_dir>/
        metrics.json     metrics + cost (params, time, epochs) + provenance
        scores.npz       scores, labels and dataset row indices of the test set
        train_log.csv    per-epoch losses / lr / timing       (deep methods)
        model.pt         best-validation state_dict           (optional)
"""

from __future__ import annotations

import csv
import gc
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from ..data.datasets import build_loaders
from ..evaluation.metrics import evaluate_scores
from ..models import build_model
from ..utils import save_json, set_seed
from .classical import run_classical
from .trainer import predict_scores, train_model


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_experiment(name: str, mcfg: dict, tcfg: dict, subsets: dict, indices: dict,
                   dataset: str, seed: int, device: torch.device, out_dir: Path,
                   save_checkpoint: bool = True, num_workers: int = 0) -> dict | None:
    """Train/evaluate ``name`` and write results to ``out_dir``.

    Args:
        name: experiment name (method key or ablation variant).
        mcfg: method configuration (``kind``, ``arch``, ``lr``, ``params`` ...).
        tcfg: shared training protocol (``configs/methods.yaml: training``).
        subsets / indices: output of :func:`qts.data.build_subsets` for ``seed``.
    Returns the metrics dict, or ``None`` if the run failed (``error.txt``).
    """
    out_dir = Path(out_dir)
    if (out_dir / "metrics.json").exists():
        print(f"  [skip] {out_dir} already complete", flush=True)
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error.txt").unlink(missing_ok=True)
    t0 = time.time()
    try:
        set_seed(seed)
        batch_size = mcfg.get("batch_size", tcfg["batch_size"])
        train_ld, val_ld, test_ld = build_loaders(subsets, batch_size, num_workers=num_workers,
                                                  pin_memory=device.type == "cuda")
        common = {
            "method": name, "dataset": dataset, "seed": seed,
            "n_train": len(subsets["train"]), "n_val": len(subsets["val"]),
            "batch_size": batch_size,
        }
        if mcfg["kind"] == "classical":
            r = run_classical(name, mcfg.get("params", {}), train_ld, test_ld, seed)
            scores, labels = r["scores"], r["labels"]
            extra = {"param_count": 0, "train_time_sec": r["fit_time_sec"],
                     "inference_time_sec": r["inference_time_sec"],
                     "best_epoch": None, "epochs_run": None}
            log, state = [], None
        else:
            model = build_model(mcfg.get("arch", name), mcfg.get("params", {}))
            res = train_model(
                model, train_ld, val_ld, device, seed,
                lr=mcfg["lr"],
                weight_decay=mcfg.get("weight_decay", tcfg["weight_decay"]),
                max_epochs=mcfg.get("max_epochs", tcfg["max_epochs"]),
                patience=mcfg.get("patience", tcfg["patience"]),
                min_delta=mcfg.get("min_delta", tcfg["min_delta"]),
                eta_min=mcfg.get("eta_min", tcfg["eta_min"]),
                grad_clip=mcfg.get("grad_clip", tcfg["grad_clip"]),
                init_kwargs=mcfg.get("init"),
            )
            scores, labels, t_inf = predict_scores(res.model, test_ld, device)
            extra = {"param_count": res.param_count, "train_time_sec": res.train_time_sec,
                     "inference_time_sec": t_inf, "best_epoch": res.best_epoch,
                     "epochs_run": res.epochs_run}
            if hasattr(res.model, "scale_weights") and len(getattr(res.model, "banks", [])) > 1:
                extra["scale_weights"] = res.model.scale_weights()
            log, state = res.log, res.best_state

        assert np.array_equal(labels, subsets["test"].labels), "test order mismatch"
        metrics = evaluate_scores(labels, scores)
        metrics.update(common)
        metrics.update(extra)
        metrics["throughput_rec_per_sec"] = len(labels) / max(extra["inference_time_sec"], 1e-9)
        metrics["wall_time_sec"] = time.time() - t0

        np.savez_compressed(out_dir / "scores.npz", scores=scores.astype(np.float32),
                            labels=labels.astype(np.int8), test_idx=indices["test"])
        _write_csv(log, out_dir / "train_log.csv")
        if save_checkpoint and state is not None:
            torch.save(state, out_dir / "model.pt")
        save_json(metrics, out_dir / "metrics.json")   # written last = completion marker
        print(f"  -> AUROC={metrics['auroc']:.4f}  AUPRC={metrics['auprc']:.4f}  "
              f"F1={metrics['best_f1']:.4f}  params={metrics['param_count']:,}  "
              f"train={metrics['train_time_sec']:.1f}s", flush=True)
        return metrics
    except Exception:  # keep the sweep going; the failure is recorded on disk
        (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"  [error] {name}/{dataset}/seed {seed}:\n{traceback.format_exc()}", flush=True)
        return None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
