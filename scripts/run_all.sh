#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper.
#
#   bash scripts/run_all.sh [--device cuda]
#
# Each stage is resumable: finished runs (metrics.json present) are skipped,
# so the script can simply be restarted after an interruption.
# Expected wall time on one RTX 3060 (12 GB): ~11 h benchmark, ~6 h ablation,
# a few minutes for transfer / sub-group / tables / figures.
set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="cuda"
[[ "${1:-}" == "--device" ]] && DEVICE="$2"
PY="${PYTHON:-python}"
mkdir -p logs

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
stage() { echo; echo "[$(stamp)] ===== $* ====="; }

if [[ ! -f data/processed/ptbxl/signals.npy || ! -f data/processed/cpsc2018/signals.npy ]]; then
  stage "0/6 preprocessing"
  "$PY" scripts/prepare_data.py 2>&1 | tee logs/prepare_data.log
fi

stage "1/6 main benchmark (13 methods x 2 datasets x 10 seeds)"
"$PY" scripts/run_benchmark.py --device "$DEVICE" 2>&1 | tee -a logs/benchmark.log

stage "2/6 template-length ablation"
"$PY" scripts/run_ablation.py --device "$DEVICE" 2>&1 | tee -a logs/ablation.log

stage "3/6 cross-dataset transfer"
"$PY" scripts/run_transfer.py --device "$DEVICE" 2>&1 | tee logs/transfer.log

stage "4/6 diagnostic sub-groups"
"$PY" scripts/run_subgroup.py 2>&1 | tee logs/subgroup.log

stage "5/6 tables"
"$PY" scripts/make_tables.py 2>&1 | tee logs/tables.log

stage "6/6 figures"
"$PY" scripts/make_figures.py 2>&1 | tee logs/figures.log

stage "done - see results/tables and results/figures"
