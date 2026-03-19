"""Train one model across all PAIRS × PS (even mix per epoch)."""

from __future__ import annotations

import argparse
import os
import warnings

from train_utils.config import REPO_ROOT
from train_utils.training import run_multitask_training

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description="General model training across multiple (d,r,p)")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--train-size-total", type=int, default=100_000)
    ap.add_argument("--val-train-size", type=int, default=50_000, help="Samples per (d,r,p) for validation")
    ap.add_argument("--checkpoint-every", type=int, default=10, help="Save epoch checkpoint; 0 to disable")
    ap.add_argument("--save-dir", type=str, default=None, help="Output directory")
    ap.add_argument(
        "--model-type",
        type=str,
        default="spatiotemporal_local",
        choices=["spatiotemporal_local", "vanilla", "3d_cnn", "gnn"],
    )
    args = ap.parse_args()

    out_dir = args.save_dir or os.path.join(str(REPO_ROOT), "data", "checkpoints", "general_local")
    run_multitask_training(
        epochs=args.epochs,
        train_size_total=args.train_size_total,
        save_dir=out_dir,
        model_type=args.model_type,
        val_train_size=args.val_train_size,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
