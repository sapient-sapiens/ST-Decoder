#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/data/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/latency_suite_$(date +%Y%m%d_%H%M%S).log"

echo "Running latency benchmarks and plots..." | tee -a "$LOG_FILE"

stdbuf -oL -eL python3 -u -m latency.fc_vs_st_rounds 2>&1 | tee -a "$LOG_FILE"
stdbuf -oL -eL python3 -u -m latency.decode_latency 2>&1 | tee -a "$LOG_FILE"
stdbuf -oL -eL python3 -u analysis/plot_fc_st_rounds.py 2>&1 | tee -a "$LOG_FILE"
stdbuf -oL -eL python3 -u analysis/plot_latency.py 2>&1 | tee -a "$LOG_FILE"

echo "latency_suite.sh complete." | tee -a "$LOG_FILE"
