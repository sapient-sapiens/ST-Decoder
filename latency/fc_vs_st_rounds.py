"""FC vs ST-local latency vs measurement rounds (single distance)."""

from __future__ import annotations

import csv
import json
import os
import statistics

from .decode_latency import time_ml_model
from st_decoder.paths import REPO_ROOT

OUTPUT = "data/fc_vs_st_rounds_d11.csv"

D = 11
R_LIST = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
P = 0.002
BATCH_SIZE = 1
NUM_TRIALS = 50


def _summarize(trials: list) -> tuple:
    mean_val = sum(trials) / len(trials)
    std_val = statistics.stdev(trials) if len(trials) > 1 else 0.0
    return mean_val, std_val, json.dumps(trials)


def main() -> None:
    path = os.path.join(str(REPO_ROOT), OUTPUT)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "d",
                "r",
                "p",
                "batch_size",
                "fc_transformer",
                "fc_transformer_std",
                "fc_transformer_trials",
                "st_local",
                "st_local_std",
                "st_local_trials",
            ]
        )
        for r in R_LIST:
            fc_trials = time_ml_model(D, r, P, BATCH_SIZE, "fc_transformer", trials=NUM_TRIALS)
            st_trials = time_ml_model(D, r, P, BATCH_SIZE, "st_local", trials=NUM_TRIALS)
            fc_mean, fc_std, fc_json = _summarize(fc_trials)
            st_mean, st_std, st_json = _summarize(st_trials)
            writer.writerow([D, r, P, BATCH_SIZE, fc_mean, fc_std, fc_json, st_mean, st_std, st_json])
            f.flush()


if __name__ == "__main__":
    main()
