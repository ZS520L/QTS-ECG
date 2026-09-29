"""Produce all paper figures from the result directories (PDF + PNG).

Figures (written to ``results/figures/``):
    auroc_per_seed      per-seed AUROC of every method (box + points)
    paired_differences  per-seed AUROC(QTS) - AUROC(baseline)
    params_vs_auroc     model size vs. mean AUROC
    training_curves     validation loss of the deep methods (mean +- s.d. over seeds)
    templates           learned QTS templates (lead II, seed 0)
    ablation            template-length ablation
    subgroup            AUROC per diagnostic sub-group
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.config import load_config, resolve_path  # noqa: E402
from qts.results import collect_metrics  # noqa: E402

DS_LABEL = {"ptbxl": "PTB-XL", "cpsc2018": "CPSC 2018"}
QTS_COLOR, SVDD_COLOR, VAE_COLOR = "#c0392b", "#2471a3", "#e67e22"
BASE_COLOR = "#7f8c8d"
HIGHLIGHT = {"qts": QTS_COLOR, "deep_svdd": SVDD_COLOR, "vae": VAE_COLOR}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 200, "pdf.fonttype": 42})


def save(fig, out: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print("  ", name)


def method_order(df, ds, methods):
    means = df[df.dataset == ds].groupby("method").auroc.mean()
    return [m for m in means.sort_values().index if m in methods]


def fig_auroc_per_seed(df, labels, datasets, out):
    """Per-seed AUROC: seed dots + mean diamond, styled like the other paper figures."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.9))
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, ds in zip(axes, datasets):
        order = method_order(df, ds, labels)[::-1]          # best method on top
        for i, m in enumerate(order):
            v = df[(df.dataset == ds) & (df.method == m)].auroc.values
            c = HIGHLIGHT.get(m, BASE_COLOR)
            ax.plot([v.min(), v.max()], [i, i], lw=0.9, color=c, alpha=0.45,
                    solid_capstyle="round", zorder=1)
            ax.scatter(v, i + rng.uniform(-0.16, 0.16, len(v)), s=8, linewidths=0,
                       color=BASE_COLOR if m not in HIGHLIGHT else c, alpha=0.75, zorder=2)
            ax.scatter(v.mean(), i, marker="D", s=16, color=c, zorder=3,
                       edgecolors="white", linewidths=0.5)
        ax.set_yticks(range(len(order)), [labels[m] for m in order], fontsize=8)
        ax.axvline(0.5, ls=":", c="0.6", lw=0.8, zorder=0)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlabel("AUROC")
        ax.set_title(DS_LABEL[ds], fontsize=10)
    fig.tight_layout()
    save(fig, out, "auroc_per_seed")


def fig_paired(df, labels, datasets, out):
    """Paired per-seed differences: seed dots + mean with 95% CI, same visual language
    as the other paper figures (thin spines, muted palette, small markers)."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.9), sharex=False)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(1)
    for ax, ds in zip(axes, datasets):
        piv = df[df.dataset == ds].pivot(index="seed", columns="method", values="auroc")
        if "qts" not in piv:
            continue
        diffs = {m: (piv["qts"] - piv[m]).dropna().values for m in piv.columns if m != "qts"}
        order = sorted(diffs, key=lambda m: np.mean(diffs[m]), reverse=True)
        for i, m in enumerate(order):
            d = diffs[m]
            c = SVDD_COLOR if m == "deep_svdd" else QTS_COLOR
            ax.scatter(d, i + rng.uniform(-0.15, 0.15, len(d)), s=8, linewidths=0,
                       color=BASE_COLOR, alpha=0.75, zorder=2)
            ci = 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
            ax.errorbar(d.mean(), i, xerr=ci, fmt="D", color=c, ms=4, lw=1.0, capsize=2.5,
                        capthick=0.9, zorder=3, markeredgecolor="white", markeredgewidth=0.5)
        ax.axvline(0, c="0.35", lw=0.9, zorder=1)
        ax.set_yticks(range(len(order)), [labels[m] for m in order], fontsize=8)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xlabel("AUROC(QTS) − AUROC(baseline)")
        ax.set_title(DS_LABEL[ds], fontsize=10)
    fig.tight_layout()
    save(fig, out, "paired_differences")


def fig_params(df, labels, datasets, out):
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.2))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        g = df[(df.dataset == ds) & (df.param_count > 0)].groupby("method").agg(
            p=("param_count", "first"), m=("auroc", "mean"), s=("auroc", "std"))
        for m, r in g.iterrows():
            c = QTS_COLOR if m == "qts" else BASE_COLOR
            ax.errorbar(r.p, r.m, yerr=r.s, fmt="o", color=c, ms=5, capsize=2)
            ax.annotate(labels[m], (r.p, r.m), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("Trainable parameters")
        ax.set_ylabel("AUROC")
        ax.set_title(DS_LABEL[ds])
    fig.tight_layout()
    save(fig, out, "params_vs_auroc")


def fig_curves(bench_root: Path, labels, datasets, out, seed=None):
    """Min-max normalised validation loss, mean +- s.d. over seeds.

    Runs that stop early are padded with their last value so that all seeds
    contribute to every epoch.
    """
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.6 * len(datasets), 3.0))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        for m in sorted(labels, key=lambda k: k in HIGHLIGHT):
            files = sorted((bench_root / ds / m).glob("seed_*/train_log.csv"))
            if seed is not None:
                files = [f for f in files if f.parent.name == f"seed_{seed}"]
            curves = []
            for f in files:
                v = pd.read_csv(f).val_loss.values.astype(float)
                if len(v) < 2 or not np.all(np.isfinite(v)) or np.ptp(v) == 0:
                    continue
                curves.append((v - v.min()) / np.ptp(v))
            if not curves:
                continue
            n = max(len(c) for c in curves)
            arr = np.stack([np.pad(c, (0, n - len(c)), mode="edge") for c in curves])
            mu, sd = arr.mean(0), arr.std(0)
            ep = np.arange(1, n + 1)
            c = HIGHLIGHT.get(m, "0.65")
            ax.plot(ep, mu, color=c, lw=1.5 if m in HIGHLIGHT else 0.7,
                    label=labels[m] if m in HIGHLIGHT else None, zorder=3 if m in HIGHLIGHT else 1)
            if m in HIGHLIGHT and len(curves) > 1:
                ax.fill_between(ep, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalised validation loss")
        ax.set_title(DS_LABEL[ds])
    axes[-1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, out, "training_curves")


def fig_templates(bench_root: Path, datasets, out, seed=0, lead=1, n_show=8, fs=500):
    import torch
    fig, axes = plt.subplots(len(datasets), n_show, figsize=(1.3 * n_show, 1.4 * len(datasets)),
                             squeeze=False)
    for row, ds in enumerate(datasets):
        f = bench_root / ds / "qts" / f"seed_{seed}" / "model.pt"
        if not f.exists():
            continue
        w = torch.load(f, map_location="cpu")["banks.0.templates"].numpy()   # (M, L, K)
        energy = (w ** 2).sum(axis=(1, 2))
        idx = np.argsort(energy)[::-1][:n_show]
        t = np.arange(w.shape[-1]) / fs * 1000
        for j, k in enumerate(idx):
            ax = axes[row, j]
            ax.plot(t, w[k, lead], color=QTS_COLOR, lw=0.9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            if j == 0:
                ax.set_ylabel(DS_LABEL[ds], fontsize=8)
    fig.suptitle(f"Learned QTS templates, lead II ({w.shape[-1]} samples = {w.shape[-1] / fs * 1000:.0f} ms), "
                 "8 highest-energy of 64", fontsize=8)
    fig.tight_layout()
    save(fig, out, "templates")


def fig_ablation(tables: Path, out):
    f = tables / "ablation.csv"
    if not f.exists():
        return
    a = pd.read_csv(f)
    ds_cols = [c[:-len("_auroc_mean")] for c in a.columns if c.endswith("_auroc_mean")]
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(a))
    w = 0.8 / max(len(ds_cols), 1)
    for i, ds in enumerate(ds_cols):
        ax.bar(x + (i - (len(ds_cols) - 1) / 2) * w, a[f"{ds}_auroc_mean"], w,
               yerr=a[f"{ds}_auroc_std"], capsize=2, label=DS_LABEL.get(ds, ds))
    ax.set_xticks(x, a["label"], rotation=25, ha="right", fontsize=7)
    lo = np.nanmin([a[f"{d}_auroc_mean"] - a[f"{d}_auroc_std"] for d in ds_cols])
    ax.set_ylim(max(0.0, lo - 0.03), 1.0)
    ax.set_ylabel("AUROC")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, out, "ablation")


def fig_subgroup(tables: Path, labels, out, top=5):
    f = tables / "subgroup.csv"
    if not f.exists():
        return
    s = pd.read_csv(f)
    dss = list(s.dataset.unique())
    fig, axes = plt.subplots(len(dss), 1, figsize=(7, 2.6 * len(dss)), squeeze=False)
    for ax, ds in zip(axes[:, 0], dss):
        a = s[s.dataset == ds]
        best = a.groupby("method").auroc_mean.mean().sort_values(ascending=False).index[:top].tolist()
        if "qts" in a.method.values and "qts" not in best:
            best = best[:-1] + ["qts"]
        groups = a.groupby("subgroup").n_group.mean().sort_values(ascending=False).index.tolist()
        x = np.arange(len(groups))
        w = 0.8 / len(best)
        for i, m in enumerate(best):
            v = a[a.method == m].set_index("subgroup").reindex(groups)
            ax.bar(x + (i - (len(best) - 1) / 2) * w, v.auroc_mean, w, yerr=v.auroc_std, capsize=1.5,
                   color=QTS_COLOR if m == "qts" else None, label=labels.get(m, m))
        ax.set_xticks(x, groups, rotation=30, ha="right", fontsize=7)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUROC")
        ax.set_title(DS_LABEL.get(ds, ds))
        ax.legend(frameon=False, fontsize=6, ncol=len(best))
    fig.tight_layout()
    save(fig, out, "subgroup")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0, help="seed whose QTS checkpoint is visualised")
    args = ap.parse_args()
    root = resolve_path(args.results)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(resolve_path("configs/benchmark.yaml"))
    labels = {m: cfg["methods"][m].get("label", m) for m in cfg["benchmark"]["methods"]}

    df = collect_metrics(root / "benchmark")
    datasets = [d for d in cfg["benchmark"]["datasets"] if len(df) and d in set(df.dataset)]
    if len(df):
        fig_auroc_per_seed(df, labels, datasets, out)
        fig_paired(df, labels, datasets, out)
        fig_params(df, labels, datasets, out)
        fig_curves(root / "benchmark", labels, datasets, out)
        fig_templates(root / "benchmark", datasets, out, args.seed)
    fig_ablation(root / "tables", out)
    fig_subgroup(root / "tables", labels, out)


if __name__ == "__main__":
    main()
