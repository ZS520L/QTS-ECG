"""Aggregate all results into paper-ready tables (CSV, Markdown and LaTeX).

Reads ``results/benchmark``, ``results/ablation``, ``results/subgroup`` and
``results/transfer`` (whichever exist) and writes ``results/tables/``.
Mean +- sample standard deviation over seeds; paired Wilcoxon signed-rank
tests over seeds with Holm correction across baselines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qts.config import load_config, resolve_path  # noqa: E402
from qts.evaluation.stats import holm, mean_std, paired_comparison  # noqa: E402
from qts.results import collect_errors, collect_metrics  # noqa: E402

DS_LABEL = {"ptbxl": "PTB-XL", "cpsc2018": "CPSC 2018"}
METRICS = ["auroc", "auprc", "best_f1"]
METRIC_LABEL = {"auroc": "AUROC", "auprc": "AUPRC", "best_f1": "F1"}


def fmt(m, s, d=3):
    return f"{m:.{d}f} ± {s:.{d}f}"


def human(n):
    if n is None or (isinstance(n, float) and np.isnan(n)) or n == 0:
        return "—"
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main_table(df, cfg, out: Path):
    methods = [m for m in cfg["benchmark"]["methods"] if m in set(df["method"])]
    datasets = [d for d in cfg["benchmark"]["datasets"] if d in set(df["dataset"])]
    labels = {m: cfg["methods"][m].get("label", m) for m in methods}

    rows, per_seed = [], []
    for m in methods:
        row = {"method": m, "label": labels[m]}
        for ds in datasets:
            sub = df[(df.method == m) & (df.dataset == ds)].sort_values("seed")
            row[f"{ds}_n_seeds"] = len(sub)
            for k in METRICS:
                if len(sub):
                    row[f"{ds}_{k}_mean"], row[f"{ds}_{k}_std"] = mean_std(sub[k])
            if len(sub):
                row[f"{ds}_train_time_sec"] = float(sub["train_time_sec"].mean())
                row[f"{ds}_epochs"] = float(sub["epochs_run"].mean()) if sub["epochs_run"].notna().any() else np.nan
                row[f"{ds}_throughput"] = float(sub["throughput_rec_per_sec"].mean())
                row[f"{ds}_prevalence"] = float(sub["prevalence"].mean())
                row[f"{ds}_f1_trivial"] = float(sub["f1_trivial"].mean())
            for _, r in sub.iterrows():
                per_seed.append({"method": m, "dataset": ds, "seed": int(r.seed),
                                 **{k: r[k] for k in METRICS}})
        row["param_count"] = int(df[df.method == m]["param_count"].iloc[0])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "main_results.csv", index=False)
    pd.DataFrame(per_seed).to_csv(out / "per_seed_metrics.csv", index=False)

    # ---- markdown / latex -------------------------------------------------
    best = {}
    for ds in datasets:
        for k in METRICS:
            col = f"{ds}_{k}_mean"
            if col in summary:
                order = summary[col].round(3).rank(ascending=False, method="min")
                best[(ds, k)] = order
    md_rows, tex_rows = [], []
    for i, r in summary.iterrows():
        md = {"Method": r["label"]}
        tex = [r["label"].replace("(ours)", "(Ours)")]
        for ds in datasets:
            for k in METRICS:
                if f"{ds}_{k}_mean" not in r or pd.isna(r.get(f"{ds}_{k}_mean")):
                    md[f"{DS_LABEL[ds]} {METRIC_LABEL[k]}"] = "—"
                    tex.append("--")
                    continue
                s = fmt(r[f"{ds}_{k}_mean"], r[f"{ds}_{k}_std"])
                rank = best[(ds, k)][i]
                md[f"{DS_LABEL[ds]} {METRIC_LABEL[k]}"] = f"**{s}**" if rank == 1 else (f"_{s}_" if rank == 2 else s)
                t = f"{r[f'{ds}_{k}_mean']:.3f}$\\pm${r[f'{ds}_{k}_std']:.3f}"
                tex.append(f"\\textbf{{{t}}}" if rank == 1 else (f"\\underline{{{t}}}" if rank == 2 else t))
        md["Params"] = human(r["param_count"])
        tex.append(human(r["param_count"]).replace("—", "---"))
        md_rows.append(md)
        tex_rows.append(" & ".join(tex) + r" \\")
    (out / "main_results.md").write_text(to_markdown(pd.DataFrame(md_rows)), encoding="utf-8")
    n_seeds = int(df.groupby(["method", "dataset"]).size().min())
    header = " & ".join(["Method"] + [f"{DS_LABEL[d]} {METRIC_LABEL[k]}" for d in datasets for k in METRICS] + ["Params"])
    tex = ("\\begin{table}[t]\n\\centering\n"
           f"\\caption{{Anomaly detection performance (mean $\\pm$ s.d. over {n_seeds} seeds). "
           "Best in bold, second underlined.}\n\\label{tab:main}\n"
           "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l" + "c" * (len(datasets) * 3 + 1) + "}\n\\hline\n"
           + header + r" \\" + "\n\\hline\n" + "\n".join(tex_rows) + "\n\\hline\n\\end{tabular}%\n}\n\\end{table}\n")
    (out / "main_results.tex").write_text(tex, encoding="utf-8")

    # ---- cost table -------------------------------------------------------
    cost = []
    for _, r in summary.iterrows():
        c = {"Method": r["label"], "Params": human(r["param_count"])}
        for ds in datasets:
            c[f"{DS_LABEL[ds]} train (s)"] = f"{r.get(f'{ds}_train_time_sec', np.nan):.0f}"
            ep = r.get(f"{ds}_epochs", np.nan)
            c[f"{DS_LABEL[ds]} epochs"] = "—" if pd.isna(ep) else f"{ep:.1f}"
            c[f"{DS_LABEL[ds]} rec/s"] = f"{r.get(f'{ds}_throughput', np.nan):.0f}"
        cost.append(c)
    (out / "cost.md").write_text(to_markdown(pd.DataFrame(cost)), encoding="utf-8")

    # ---- paired statistics: QTS vs every baseline -------------------------
    stats_rows = []
    if "qts" in methods:
        for ds in datasets:
            for k in ("auroc", "auprc"):
                q = df[(df.method == "qts") & (df.dataset == ds)].set_index("seed")[k]
                pv = {}
                tmp = []
                for m in methods:
                    if m == "qts":
                        continue
                    b = df[(df.method == m) & (df.dataset == ds)].set_index("seed")[k]
                    common = q.index.intersection(b.index)
                    if len(common) < 2:
                        continue
                    res = paired_comparison(q.loc[common].values, b.loc[common].values)
                    res.update({"dataset": ds, "metric": k, "baseline": m})
                    pv[m] = res["p_wilcoxon_two_sided"]
                    tmp.append(res)
                adj = holm(pv)
                for res in tmp:
                    res["p_holm_two_sided"] = adj[res["baseline"]]
                stats_rows += tmp
        st = pd.DataFrame(stats_rows)
        st.to_csv(out / "stats_qts_vs_baselines.csv", index=False)
        if len(st):
            view = st[st.metric == "auroc"].copy()
            view["Baseline"] = view["baseline"].map(labels)
            view["Dataset"] = view["dataset"].map(DS_LABEL)
            view["ΔAUROC (QTS − baseline)"] = view.apply(
                lambda r: f"{r.mean_diff:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]", axis=1)
            view["W/T/L"] = view.apply(lambda r: f"{r.wins}/{r.ties}/{r.losses}", axis=1)
            view["p (Wilcoxon)"] = view["p_wilcoxon_two_sided"].map(lambda p: f"{p:.4f}")
            view["p (Holm)"] = view["p_holm_two_sided"].map(lambda p: f"{p:.4f}")
            (out / "stats_qts_vs_baselines.md").write_text(
                to_markdown(view[["Dataset", "Baseline", "ΔAUROC (QTS − baseline)", "W/T/L",
                                  "p (Wilcoxon)", "p (Holm)"]]), encoding="utf-8")

    # ---- per-seed ranks ---------------------------------------------------
    rk = []
    for ds in datasets:
        piv = df[df.dataset == ds].pivot(index="seed", columns="method", values="auroc")
        ranks = piv.rank(axis=1, ascending=False, method="min")
        for m in piv.columns:
            rk.append({"dataset": ds, "method": m, "mean_rank": ranks[m].mean(),
                       "times_first": int((ranks[m] == 1).sum()), "n_seeds": int(ranks[m].notna().sum())})
    pd.DataFrame(rk).sort_values(["dataset", "mean_rank"]).to_csv(out / "ranks_auroc.csv", index=False)
    return summary


def ablation_table(abl: pd.DataFrame, bench: pd.DataFrame, cfg, out: Path):
    acfg = cfg["ablation"]
    reuse = acfg.get("reuse_from_benchmark", {})
    parts = [abl]
    for v, src in reuse.items():
        b = bench[bench.method == src].copy()
        b["method"] = v
        parts.append(b)
    df = pd.concat(parts, ignore_index=True)
    rows, md = [], []
    for v, spec in acfg["variants"].items():
        r = {"variant": v, "label": spec.get("label", v)}
        mrow = {"Configuration": spec.get("label", v)}
        for ds in acfg["datasets"]:
            sub = df[(df.method == v) & (df.dataset == ds)]
            if len(sub):
                r[f"{ds}_auroc_mean"], r[f"{ds}_auroc_std"] = mean_std(sub["auroc"])
                r[f"{ds}_n"] = len(sub)
                r[f"{ds}_train_time_sec"] = float(sub["train_time_sec"].mean())
                mrow[f"{DS_LABEL[ds]} AUROC"] = fmt(r[f"{ds}_auroc_mean"], r[f"{ds}_auroc_std"]) + f" (n={len(sub)})"
            else:
                mrow[f"{DS_LABEL[ds]} AUROC"] = "—"
        pc = df[df.method == v]["param_count"]
        r["param_count"] = int(pc.iloc[0]) if len(pc) else None
        mrow["Params"] = human(r["param_count"]) if r["param_count"] else "—"
        rows.append(r)
        md.append(mrow)
    pd.DataFrame(rows).to_csv(out / "ablation.csv", index=False)
    (out / "ablation.md").write_text(to_markdown(pd.DataFrame(md)), encoding="utf-8")


def subgroup_table(path: Path, cfg, out: Path):
    sg = pd.read_csv(path)
    agg = sg.groupby(["dataset", "subgroup", "type", "method"]).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", lambda x: x.std(ddof=1)),
        n_group=("n_group", "mean"), n_seeds=("seed", "nunique")).reset_index()
    agg.to_csv(out / "subgroup.csv", index=False)
    labels = {m: c.get("label", m) for m, c in cfg["methods"].items()}
    md = []
    for ds in agg.dataset.unique():
        a = agg[agg.dataset == ds]
        top = (a.groupby("method").auroc_mean.mean().sort_values(ascending=False).index[:6])
        piv = a[a.method.isin(top)].pivot(index=["subgroup", "type"], columns="method", values="auroc_mean")
        piv = piv[list(top)]
        md.append(f"### {DS_LABEL[ds]}\n")
        tbl = piv.reset_index()
        tbl.columns = ["Sub-group", "Type"] + [labels.get(m, m) for m in top]
        for c in tbl.columns[2:]:
            tbl[c] = tbl[c].map(lambda x: f"{x:.3f}")
        md.append(to_markdown(tbl))
    (out / "subgroup.md").write_text("\n".join(md), encoding="utf-8")


def transfer_table(path: Path, cfg, out: Path):
    tr = pd.read_csv(path)
    agg = tr.groupby(["method", "source", "target"]).agg(
        auroc_mean=("auroc", "mean"), auroc_std=("auroc", lambda x: x.std(ddof=1)),
        n=("seed", "nunique")).reset_index()
    agg.to_csv(out / "transfer.csv", index=False)
    labels = {m: c.get("label", m) for m, c in cfg["methods"].items()}
    md = []
    for _, r in agg.iterrows():
        md.append({"Method": labels.get(r.method, r.method), "Train on": DS_LABEL[r.source],
                   "Test on": DS_LABEL[r.target], "AUROC": fmt(r.auroc_mean, r.auroc_std),
                   "Seeds": r.n})
    (out / "transfer.md").write_text(to_markdown(pd.DataFrame(md)), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    root = resolve_path(args.results)
    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    bcfg = load_config(resolve_path("configs/benchmark.yaml"))
    acfg = load_config(resolve_path("configs/ablation.yaml"))

    bench = collect_metrics(root / "benchmark")
    errs = collect_errors(root / "benchmark")
    if errs:
        print("WARNING - failed runs:\n  " + "\n  ".join(errs))
    if len(bench):
        main_table(bench, bcfg, out)
        print("main table:", len(bench), "runs")
    if (root / "ablation").exists():
        abl = collect_metrics(root / "ablation")
        if len(abl):
            ablation_table(abl, bench, acfg, out)
            print("ablation table:", len(abl), "runs")
    if (root / "subgroup" / "subgroup_auroc.csv").exists():
        subgroup_table(root / "subgroup" / "subgroup_auroc.csv", bcfg, out)
        print("subgroup table done")
    if (root / "transfer" / "transfer.csv").exists():
        transfer_table(root / "transfer" / "transfer.csv", bcfg, out)
        print("transfer table done")


if __name__ == "__main__":
    main()
