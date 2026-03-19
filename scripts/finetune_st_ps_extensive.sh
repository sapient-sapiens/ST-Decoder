#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAVE_DIR="${ROOT_DIR}/data/checkpoints/finetuned_local"
LOG_DIR="${ROOT_DIR}/data/logs"
mkdir -p "$SAVE_DIR" "$LOG_DIR"

TRAIN_SIZE="${TRAIN_SIZE:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-2e-5}"

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

INIT_CKPT="${INIT_CKPT:-$(pick_best_ckpt "${ROOT_DIR}/data/checkpoints/general_local")}"

epochs_for_pair() {
  local d="$1"
  local r="$2"
  if [[ "$d" == "3" && "$r" == "7" ]]; then
    echo 10
  elif [[ "$d" == "5" && "$r" == "5" ]]; then
    echo 20
  elif [[ "$d" == "7" && "$r" == "3" ]]; then
    echo 30
  else
    echo 20
  fi
}

for pair in "3:7" "5:5" "7:3"; do
  d="${pair%%:*}"
  r="${pair##*:}"
  epochs="$(epochs_for_pair "$d" "$r")"

  for i in $(seq -w 1 8); do
    p="$(printf '0.%03d' "$i")"
    log="${LOG_DIR}/finetune_st_ext_d${d}_r${r}_p${p}.log"

    stdbuf -oL -eL python3 -u -m train_utils.cli.finetune \
      --model-type spatiotemporal_local \
      --d "$d" --r "$r" --p "$p" \
      --epochs "$epochs" --train-size "$TRAIN_SIZE" --batch-size "$BATCH_SIZE" --lr "$LR" \
      --save-dir "$SAVE_DIR" --init-ckpt "$INIT_CKPT" \
      2>&1 | tee -a "$log"
  done
done

echo "finetune_st_ps_extensive.sh complete."
