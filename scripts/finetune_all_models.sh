#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

D="${D:-5}"
R="${R:-5}"
P="${P:-0.005}"
TRAIN_SIZE="${TRAIN_SIZE:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-2e-5}"
EPOCHS_DEPOL="${EPOCHS_DEPOL:-20}"
EPOCHS_NOISE="${EPOCHS_NOISE:-10}"
BIAS="${BIAS:-10}"
ANGLE="${ANGLE:-0.1}"
SPECTATOR="${SPECTATOR:-0.01}"

SAVE_DIR="${ROOT_DIR}/data/checkpoints/finetune_more_d5r5"
LOG_DIR="${ROOT_DIR}/data/logs"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

pick_best_ckpt() {
  local ckpt_dir="$1"
  CKPT_DIR="$ckpt_dir" python3 - <<'PY'
import glob
import os
import torch

ckpt_dir = os.environ["CKPT_DIR"]
paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
if not paths:
    raise SystemExit(f"No checkpoints found in {ckpt_dir}")

def score(path: str):
    try:
        ckpt = torch.load(path, map_location="cpu")
        val_acc = ckpt.get("val_acc")
        val_loss = ckpt.get("val_loss")
        if val_acc is not None:
            return (2, float(val_acc), 0.0)
        if val_loss is not None:
            return (1, -float(val_loss), 0.0)
    except Exception:
        pass
    return (0, 0.0, os.path.getmtime(path))

print(max(paths, key=score))
PY
}

pick_latest_depol_best() {
  local model="$1"
  local save_dir="$2"
  local d="$3"
  local r="$4"
  local p="$5"
  SAVE_DIR="$save_dir" MODEL="$model" D="$d" R="$r" P="$p" python3 - <<'PY'
import glob
import os

save_dir = os.environ["SAVE_DIR"]
model = os.environ["MODEL"]
d = os.environ["D"]
r = os.environ["R"]
p = os.environ["P"]
pattern = os.path.join(save_dir, f"{model}_d{d}_r{r}_p{p}_*_best.pt")
paths = [p for p in glob.glob(pattern) if all(x not in os.path.basename(p) for x in ("_biased_", "_coherent_", "_spectator_"))]
if not paths:
    raise SystemExit(f"No depolarizing checkpoint found for {model}")
print(max(paths, key=os.path.getmtime))
PY
}

run_finetune() {
  local model_type="$1"
  local noise_model="$2"
  local init_ckpt="$3"
  local epochs="$4"
  local extra_args="$5"

  local log="${LOG_DIR}/finetune_all_${model_type}_${noise_model}_d${D}_r${R}_p${P}_$(date +%Y%m%d_%H%M%S).log"
  echo "Model=${model_type} Noise=${noise_model} Epochs=${epochs} Init=${init_ckpt}"

  stdbuf -oL -eL python3 -u -m st_decoder.cli.finetune \
    --model-type "$model_type" \
    --d "$D" --r "$R" --p "$P" \
    --epochs "$epochs" --train-size "$TRAIN_SIZE" --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --save-dir "$SAVE_DIR" --init-ckpt "$init_ckpt" \
    --noise-model "$noise_model" \
    ${extra_args} \
    2>&1 | tee -a "$log"
}

# 1) Depolarizing finetune for all four models at (d=5,r=5,p=0.005).
declare -A GENERAL_DIRS=(
  [spatiotemporal_local]="${ROOT_DIR}/data/checkpoints/general_local"
  [vanilla]="${ROOT_DIR}/data/checkpoints/general_vanilla"
  [gnn]="${ROOT_DIR}/data/checkpoints/general_gnn"
  [3d_cnn]="${ROOT_DIR}/data/checkpoints/general_3dcnn"
)

for model in spatiotemporal_local vanilla gnn 3d_cnn; do
  init_ckpt="$(pick_best_ckpt "${GENERAL_DIRS[$model]}")"
  run_finetune "$model" "depolarizing" "$init_ckpt" "$EPOCHS_DEPOL" ""
done

# 2) Noise-model finetunes for all four models.
noise_models=(biased coherent spectator)
noise_args=("--noise-bias ${BIAS}" "--noise-angle ${ANGLE}" "--noise-spectator ${SPECTATOR}")

for model in spatiotemporal_local vanilla gnn 3d_cnn; do
  base_ckpt="$(pick_latest_depol_best "$model" "$SAVE_DIR" "$D" "$R" "$P")"
  for i in "${!noise_models[@]}"; do
    run_finetune "$model" "${noise_models[$i]}" "$base_ckpt" "$EPOCHS_NOISE" "${noise_args[$i]}"
  done
done

echo "finetune_all_models.sh complete."
