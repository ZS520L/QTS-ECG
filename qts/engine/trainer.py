"""Generic one-class training loop shared by QTS and every deep baseline.

Protocol (identical for all deep methods, Section 4 of the paper):

* AdamW, cosine annealing over ``max_epochs`` (to ``eta_min``), gradient
  clipping at norm 1.0;
* validation loss (same objective, normal-only validation patients) after
  every epoch; the weights with the lowest validation loss are restored at
  the end; training stops after ``patience`` epochs without an improvement
  larger than ``min_delta``;
* **no test data is used for any decision** (model selection, early
  stopping, thresholds or seeds).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..utils import count_parameters, set_seed


class EarlyStopping:
    """Stop after ``patience`` epochs without ``val < best - min_delta``."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


@dataclass
class TrainResult:
    model: torch.nn.Module
    best_state: dict | None
    log: list = field(default_factory=list)
    best_epoch: int = 0
    epochs_run: int = 0
    train_time_sec: float = 0.0
    param_count: int = 0


@torch.no_grad()
def predict_scores(model: torch.nn.Module, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(scores, labels, seconds)`` for every record of ``loader``."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    scores, labels = [], []
    for x, y in loader:
        scores.append(model.anomaly_score(x.to(device)).float().cpu().numpy())
        labels.append(y.numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(scores), np.concatenate(labels), time.time() - t0


def _mean_loss(model, loader, device) -> float:
    total, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            total += model.compute_loss(x.to(device)).item()
            n += 1
    return total / max(n, 1)


def train_model(model: torch.nn.Module, train_loader, val_loader, device, seed: int,
                lr: float, weight_decay: float = 1e-5, max_epochs: int = 100,
                patience: int = 10, min_delta: float = 1e-4, eta_min: float = 0.0,
                grad_clip: float = 1.0, init_kwargs: dict | None = None,
                log_every: int = 5, verbose: bool = True) -> TrainResult:
    """Train ``model`` on normal data and restore the best-validation weights.

    ``seed`` is re-applied here, *after* the model has been constructed (the
    caller seeds before construction), so that data-dependent initialisation
    and the shuffling order do not depend on the architecture.
    """
    set_seed(seed)
    model = model.to(device)
    n_params = count_parameters(model)
    if hasattr(model, "initialize"):
        model.initialize(train_loader, device, **(init_kwargs or {}))

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)
    stopper = EarlyStopping(patience, min_delta)

    log, best_val, best_state, best_epoch = [], float("inf"), None, 0
    t_start = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        t_ep = time.time()
        model.train()
        total, n = 0.0, 0
        for x, _ in train_loader:
            loss = model.compute_loss(x.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += loss.item()
            n += 1
        train_loss = total / max(n, 1)

        model.eval()
        val_loss = _mean_loss(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        log.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "lr": round(lr_now, 8),
            "epoch_time_sec": round(time.time() - t_ep, 2),
            "cumulative_time_sec": round(time.time() - t_start, 2),
        })
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose and (epoch == 1 or epoch % log_every == 0):
            print(f"    epoch {epoch:3d}  train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={lr_now:.6f}  ({log[-1]['epoch_time_sec']:.1f}s)", flush=True)
        stopper.step(val_loss)
        if stopper.should_stop:
            if verbose:
                print(f"    early stop at epoch {epoch} (best epoch {best_epoch})", flush=True)
            break

    train_time = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, best_state=best_state, log=log, best_epoch=best_epoch,
                       epochs_run=epoch, train_time_sec=train_time, param_count=n_params)
