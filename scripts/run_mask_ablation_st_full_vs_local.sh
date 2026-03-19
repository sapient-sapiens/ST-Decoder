#!/usr/bin/env bash
# Mask ablation: train spatiotemporal_local vs spatiotemporal_full for d=5, r in {3,5,7}
# (same grid as paper Appendix C5). Equivalent to looping finetune with different --model-type.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/data/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="${LOG_DIR}/mask_ablation_st_full_vs_local_$(date +%Y%m%d_%H%M%S).log"
SAVE_DIR="${ROOT_DIR}/data/checkpoints/exp_st_vs_nomask_d5_r357"
mkdir -p "$SAVE_DIR"

echo "Mask ablation: spatiotemporal_local vs spatiotemporal_full (d=5, r in 3,5,7)" | tee -a "$LOG_FILE"

for r in 3 5 7; do
  for mt in spatiotemporal_local spatiotemporal_full; do
    echo "=== finetune model_type=${mt} d=5 r=${r} ===" | tee -a "$LOG_FILE"
    stdbuf -oL -eL python3 -u -m train_utils.cli.finetune \
      --d 5 --r "$r" --p 0.005 \
      --epochs 30 \
      --train-size 1000000 \
      --batch-size 1024 \
      --val-ratio 0.1 \
      --num-workers 0 \
      --lr 1e-4 \
      --weight-decay 0.0 \
      --save-dir "$SAVE_DIR" \
      --model-type "$mt" \
      2>&1 | tee -a "$LOG_FILE"
  done
done

echo "Summarizing finetune CSVs..." | tee -a "$LOG_FILE"
stdbuf -oL -eL python3 -u analysis/compare_st_mask_vs_nomask_d5.py \
  --save-dir "$SAVE_DIR" \
  --out-csv data/exp_st_vs_nomask_d5_r357_compare.csv \
  2>&1 | tee -a "$LOG_FILE"

echo "run_mask_ablation_st_full_vs_local.sh complete." | tee -a "$LOG_FILE"
