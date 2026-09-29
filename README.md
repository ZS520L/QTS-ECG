# QTS-ECG — Quadratic Template Scoring for Unsupervised 12-lead ECG Anomaly Detection

Official code for

> **Quadratic Template Scoring: A Parameter-Efficient Approach to Unsupervised ECG Anomaly Detection**
> Jing Shen, Huashun Li, Jian Gao

QTS learns a bank of 64 multi-lead templates, each as long as the QT interval
(155 samples = 310 ms at 500 Hz), from **normal ECGs only**. A recording is
scored by how far its sliding windows are from their nearest template. The
whole model is two 1-D convolutions and a `min`, with 119 K parameters and no
beat segmentation.

```
d[k,t] = ||x_t||² + 2⟨w_k, x_t⟩ + ||w_k||²      (conv with ones-kernel, conv with templates, constant)
S(x)   = mean_t  min_k  d[k,t]                   (anomaly score; trained by minimising S on normal data)
```

The implementation is in [`qts/models/qts.py`](qts/models/qts.py) (≈150 lines).

---

## Repository layout

```
configs/            all hyper-parameters (data, 13 methods, benchmark, ablation) - YAML
qts/
  data/             WFDB readers, preprocessing, patient-level splits, datasets
  models/           QTS + 11 deep / 2 classical baselines, build_model() registry
  engine/           training loop, early stopping, classical pipeline, experiment runner
  evaluation/       AUROC / AUPRC / F1, paired statistics (Wilcoxon + Holm)
scripts/            one entry point per experiment (see below)
tools/              helpers (e.g. bit-wise comparison of preprocessed arrays)
tests/              unit tests (pytest)
docs/               REPRODUCIBILITY.md - protocol, environment, expected run times
```

## Installation

```bash
git clone <this repository> qts-ecg && cd qts-ecg
python -m venv .venv && source .venv/bin/activate        # optional
pip install torch --index-url https://download.pytorch.org/whl/cu118   # pick your CUDA build
pip install -r requirements.txt
pytest -q                                                 # 36 tests, < 1 min on CPU
```

## Data

| Dataset | Source | Records used | Normal / abnormal |
|---|---|---|---|
| PTB-XL v1.0.3 | <https://physionet.org/content/ptb-xl/1.0.3/> | 21,799 | 9,069 / 12,730 |
| CPSC 2018 | <http://2018.icbeb.org/Challenge.html> | 6,877 | 918 / 5,959 |

Download both datasets and point `configs/data.yaml` (or the command-line flags) to them:

```bash
python scripts/prepare_data.py --ptbxl-dir /path/to/ptb-xl --cpsc-dir /path/to/cpsc2018
```

This band-pass filters (0.5–45 Hz, 4th-order zero-phase Butterworth), pads or
truncates to 10 s and z-scores every lead, writing `data/processed/<dataset>/`
(`signals.npy`, `labels.npy`, `patient_ids.npy`, `meta.csv`).

## Reproducing the paper

Everything — main table, ablation, cross-dataset transfer, sub-group analysis,
tables and figures — is produced by

```bash
bash scripts/run_all.sh            # ~17 h on a single RTX 3060; resumable
```

or stage by stage:

| Step | Command | Output |
|---|---|---|
| Main benchmark (13 methods × 2 datasets × 10 seeds) | `python scripts/run_benchmark.py` | `results/benchmark/` |
| Template-length ablation | `python scripts/run_ablation.py` | `results/ablation/` |
| Cross-dataset transfer | `python scripts/run_transfer.py` | `results/transfer/` |
| Diagnostic sub-groups | `python scripts/run_subgroup.py` | `results/subgroup/` |
| Tables (CSV / Markdown / LaTeX) | `python scripts/make_tables.py` | `results/tables/` |
| Figures (PDF / PNG) | `python scripts/make_figures.py` | `results/figures/` |

Useful flags: `--methods qts deep_svdd`, `--datasets ptbxl`, `--seeds 0 1 2`,
`--device cpu`. Each run writes `metrics.json`, `scores.npz` (test scores,
labels and record indices), `train_log.csv` and `model.pt` to
`results/<experiment>/<dataset>/<method>/seed_<k>/`; finished runs are skipped
on restart.

### Evaluation protocol

* **Seeds** `0–9`, fixed in advance and shared by all methods. A seed determines
  the patient-level split *and* the model initialisation, so results are paired.
* **Split** 70 / 15 / 15 % by patient; train and validation contain normal
  recordings only, the test set contains all test-patient recordings.
* **Training** identical for every deep method: AdamW (weight decay 1e-5),
  cosine schedule, ≤ 100 epochs, early stopping on validation loss (patience 10),
  gradient clipping 1.0, batch size 64 (32 for TranAD / Anomaly Transformer).
* **Metrics** threshold-free AUROC and AUPRC; best-threshold F1 is reported for
  completeness together with the trivial "all abnormal" F1, which is high
  because abnormal records are the majority class.
* **Statistics** mean ± sample s.d. over seeds; QTS vs. each baseline with the
  paired Wilcoxon signed-rank test, Holm-corrected over baselines.

## Results

`results/tables/main_results.md` after `run_all.sh`. The numbers reported in the
paper are those produced by this code with seeds 0–9.

## Baselines

All baselines are re-implemented in a common framework with the same input,
split and training protocol. Where an implementation departs from the original
paper (e.g. the simplified TranAD), this is stated in the module docstring
under *Simplification*. Parameter counts are measured, not quoted.

## Citation

```bibtex
@article{shen2026qts,
  title  = {Quadratic Template Scoring: A Parameter-Efficient Approach to Unsupervised ECG Anomaly Detection},
  author = {Shen, Jing and Li, Huashun and Gao, Jian},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). PTB-XL and CPSC 2018 are distributed under their own licenses.
