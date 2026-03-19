#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPOCHS="${EPOCHS:-250}"
TRAIN_SIZE_TOTAL="${TRAIN_SIZE_TOTAL:-1000000}"
VAL_TRAIN_SIZE="${VAL_TRAIN_SIZE:-50000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
LOG_DIR="${ROOT_DIR}/data/logs"

mkdir -p "$LOG_DIR"

declare -A SAVE_DIRS=(
  [spatiotemporal_local]="${ROOT_DIR}/data/checkpoints/general_local"
  [vanilla]="${ROOT_DIR}/data/checkpoints/general_vanilla"
  [gnn]="${ROOT_DIR}/data/checkpoints/general_gnn"
  [3d_cnn]="${ROOT_DIR}/data/checkpoints/general_3dcnn"
)

for model in spatiotemporal_local vanilla gnn 3d_cnn; do
  save_dir="${SAVE_DIRS[$model]}"
  mkdir -p "$save_dir"
  log_file="${LOG_DIR}/general_${model}_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "General training: ${model}"
  echo "  epochs=${EPOCHS} train_size_total=${TRAIN_SIZE_TOTAL}"
  echo "  save_dir=${save_dir}"
  echo "  log=${log_file}"
  echo "============================================================"

  stdbuf -oL -eL python3 -u -m train_utils.cli.general \
    --model-type "$model" \
    --epochs "$EPOCHS" \
    --train-size-total "$TRAIN_SIZE_TOTAL" \
    --val-train-size "$VAL_TRAIN_SIZE" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --save-dir "$save_dir" \
    2>&1 | tee -a "$log_file"
done

echo "train_general_all.sh complete."
