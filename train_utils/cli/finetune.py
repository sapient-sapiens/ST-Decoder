"""Fine-tune a decoder on a single (d, r, p) with configurable noise."""

from __future__ import annotations

import argparse

from train_utils.config import (
    DROPOUT,
    ST_LOCAL_D_MODEL,
    ST_LOCAL_FFN_DIM,
    ST_LOCAL_N_LAYERS,
    ST_LOCAL_S_NHEAD,
    ST_LOCAL_T_NHEAD,
)
from train_utils.models import build_model
from train_utils.training import configure_finetune_runtime, run_one


def main() -> None:
    configure_finetune_runtime()

    ap = argparse.ArgumentParser(description="Finetune model on a single (d,r,p) with configurable noise")
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--p", type=float, default=None, help="Physical error rate (sets both p_gate and p_measurement)")
    ap.add_argument("--p-gate", type=float, default=None, help="Gate error rate (overrides --p for gates)")
    ap.add_argument("--p-measurement", type=float, default=None, help="Measurement error rate (overrides --p for measurements)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--train-size", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--init-ckpt", type=str, default=None)
    ap.add_argument("--save-dir", type=str, required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for optimizer (default: 0.0, use 0.01-0.1 for vanilla transformer)",
    )
    ap.add_argument(
        "--model-type",
        type=str,
        default="spatiotemporal_local",
        choices=["spatiotemporal_local", "spatiotemporal_full", "vanilla", "3d_cnn", "gnn"],
    )
    ap.add_argument(
        "--noise-model",
        type=str,
        default="depolarizing",
        choices=["depolarizing", "biased", "coherent", "spectator"],
        help="Noise: depolarizing (default), biased, coherent, spectator",
    )
    ap.add_argument("--noise-bias", type=float, default=100.0, help="Z-bias ratio for biased noise")
    ap.add_argument("--noise-angle", type=float, default=0.1, help="Coherent rotation angle (rad)")
    ap.add_argument("--noise-spectator", type=float, default=0.01, help="Spectator error probability")
    ap.add_argument("--precomputed-data", type=str, default=None, help="Path to precomputed .pt data")

    args = ap.parse_args()

    use_3d = args.model_type == "3d_cnn"
    use_padding = args.model_type in ("spatiotemporal_local", "spatiotemporal_full", "vanilla")

    p_val = args.p if args.p is not None else 0.005
    p_gate = args.p_gate if args.p_gate is not None else p_val
    p_measurement = args.p_measurement if args.p_measurement is not None else p_val

    base_d_model = ST_LOCAL_D_MODEL
    base_n_layers = ST_LOCAL_N_LAYERS
    base_s_nhead = ST_LOCAL_S_NHEAD
    base_t_nhead = ST_LOCAL_T_NHEAD
    base_ffn = ST_LOCAL_FFN_DIM
    base_dropout = DROPOUT

    model = build_model(
        args.d,
        args.r,
        d_model=base_d_model,
        n_layers=base_n_layers,
        s_nhead=base_s_nhead,
        t_nhead=base_t_nhead,
        ffn=base_ffn,
        dropout=base_dropout,
        init_ckpt=args.init_ckpt,
        model_type=args.model_type,
    )
    best, csvp = run_one(
        model,
        args.d,
        args.r,
        p_val,
        args.epochs,
        args.train_size,
        args.batch_size,
        args.val_ratio,
        args.num_workers,
        args.save_dir,
        tag=args.model_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_3d=use_3d,
        use_padding=use_padding,
        p_gate=p_gate,
        p_measurement=p_measurement,
        noise_model=args.noise_model,
        noise_bias=args.noise_bias,
        noise_angle=args.noise_angle,
        noise_spectator=args.noise_spectator,
        precomputed_data=args.precomputed_data,
    )
    print(best)
    print(csvp)


if __name__ == "__main__":
    main()
