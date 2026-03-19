#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/data/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/mask_ablation_st_full_vs_local_$(date +%Y%m%d_%H%M%S).log"

echo "Running mask ablation: spatiotemporal_local vs spatiotemporal_full" | tee -a "$LOG_FILE"
stdbuf -oL -eL python3 -u -m st_decoder.benchmarks.mask_ablation 2>&1 | tee -a "$LOG_FILE"

echo "Summarizing mask ablation CSVs..." | tee -a "$LOG_FILE"
stdbuf -oL -eL python3 -u analysis/compare_st_mask_vs_nomask_d5.py \
  --save-dir data/checkpoints/exp_st_vs_nomask_d5_r357 \
  --out-csv data/exp_st_vs_nomask_d5_r357_compare.csv \
  2>&1 | tee -a "$LOG_FILE"

echo "run_mask_ablation_st_full_vs_local.sh complete." | tee -a "$LOG_FILE"
