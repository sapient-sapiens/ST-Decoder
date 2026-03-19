"""Model and PyMatching latency sweeps (writes CSV)."""

from __future__ import annotations

import csv
import gc
import json
import os
import statistics
from time import perf_counter
from typing import List, Optional

import pymatching
import stim
import torch
from train_utils.config import (
    DROPOUT,
    ST_LOCAL_D_MODEL,
    ST_LOCAL_FFN_DIM,
    ST_LOCAL_N_LAYERS,
    ST_LOCAL_S_NHEAD,
    ST_LOCAL_T_NHEAD,
    ST_LOCAL_SPATIAL_RADIUS,
    ST_TEMPORAL_WINDOW,
    VANILLA_D_MODEL,
    VANILLA_FFN_DIM,
    VANILLA_N_HEAD,
    VANILLA_N_LAYERS,
)
from train_utils.config import REPO_ROOT
from src.circuit import get_data_surface, get_circuit_surface, pad_surface
from src.transformer_models import SpatioTemporalLocalTransformer, VanillaTransformer


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _cuda_clear() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def time_pymatching(d: int, r: int, p: float, batch_size: int, trials: int = 1) -> List[float]:
    circuit = get_circuit_surface(d, r, p)
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = circuit.compile_detector_sampler()
    syndrome, flips = sampler.sample(batch_size, separate_observables=True)
    times = []
    for _ in range(max(1, int(trials))):
        t0 = perf_counter()
        matching.decode_batch(syndrome)
        t1 = perf_counter()
        times.append((t1 - t0) / batch_size)
    del circuit, model, matching, sampler, syndrome, flips
    _cuda_clear()
    return times


def time_ml_model(
    d: int,
    r: int,
    p: float,
    batch_size: int,
    model_type: str = "st_local",
    quantize: Optional[str] = None,
    trials: int = 1,
    *,
    use_compile: bool = False,
) -> List[float]:
    device = get_device()
    _cuda_clear()

    circuit = get_circuit_surface(d, r, p)
    seq_len = (r + 1) * (d * d - 1)
    effective_batch = batch_size
    max_attn_elems = 20_000_000
    if model_type == "fc_transformer":
        max_batch = max(1, max_attn_elems // (seq_len * seq_len))
        effective_batch = min(batch_size, int(max_batch))
    elif model_type == "st_local":
        s = d * d - 1
        t = r + 1
        max_batch = max(1, max_attn_elems // (t * s * s))
        effective_batch = min(batch_size, int(max_batch))
    detections, _, _ = get_data_surface(circuit, effective_batch)

    expected_seq_len = seq_len
    if detections.shape[1] != expected_seq_len:
        pad_fn = pad_surface(d, r)
        detections = pad_fn(detections)
    tokens = ((detections + 1) / 2).to(dtype=torch.long, device=device)

    if model_type == "st_local":
        model = SpatioTemporalLocalTransformer(
            d_model=ST_LOCAL_D_MODEL,
            n_layers=ST_LOCAL_N_LAYERS,
            n_head=ST_LOCAL_S_NHEAD,
            ffn=ST_LOCAL_FFN_DIM,
            dropout=DROPOUT,
            s_nhead=ST_LOCAL_S_NHEAD,
            t_nhead=ST_LOCAL_T_NHEAD,
            t_window_size=ST_TEMPORAL_WINDOW,
            configs=[(d, r)],
            spatial_radius=ST_LOCAL_SPATIAL_RADIUS,
        )
    elif model_type == "fc_transformer":
        model = VanillaTransformer(
            d_model=VANILLA_D_MODEL,
            n_layers=VANILLA_N_LAYERS,
            n_head=VANILLA_N_HEAD,
            ffn=VANILLA_FFN_DIM,
            dropout=DROPOUT,
            configs=[(d, r)],
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model.to(device)

    if quantize == "bf16":
        model = model.to(dtype=torch.bfloat16)
        tokens = tokens.to(dtype=torch.long)
    elif quantize == "fp16":
        model = model.to(dtype=torch.float16)
        tokens = tokens.to(dtype=torch.long)

    if use_compile:
        model = torch.compile(model)
    model.eval()

    with torch.no_grad():
        _ = model(tokens)
        _cuda_sync()

    times = []
    for _ in range(max(1, int(trials))):
        _cuda_sync()
        t0 = perf_counter()
        with torch.no_grad():
            _ = model(tokens)
        _cuda_sync()
        t1 = perf_counter()
        times.append((t1 - t0) / effective_batch)

    del model, tokens, detections, circuit
    _cuda_clear()
    return times


MAX_FC_SEQ_LEN = 8000
DEFAULT_SAVE = "data/latency_r25.csv"


def _summarize(trials: List[float]) -> tuple:
    if not trials:
        return float("nan"), float("nan"), "[]"
    mean_val = sum(trials) / len(trials)
    std_val = statistics.stdev(trials) if len(trials) > 1 else 0.0
    return mean_val, std_val, json.dumps(trials)


def main() -> None:
    d_list = list(range(3, 26, 2))
    r_list = [25]
    p_list = [0.001, 0.002, 0.005]
    save_dir = os.path.join(str(REPO_ROOT), DEFAULT_SAVE)
    batch_size = 64
    num_trials = 50
    use_compile = False

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    with open(save_dir, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "d",
                "r",
                "p",
                "pymatching",
                "pymatching_std",
                "pymatching_trials",
                "fc_transformer",
                "fc_transformer_std",
                "fc_transformer_trials",
                "st_local",
                "st_local_std",
                "st_local_trials",
                "bf16_fc_transformer",
                "bf16_fc_transformer_std",
                "bf16_fc_transformer_trials",
                "bf16_st_local",
                "bf16_st_local_std",
                "bf16_st_local_trials",
                "fp16_fc_transformer",
                "fp16_fc_transformer_std",
                "fp16_fc_transformer_trials",
                "fp16_st_local",
                "fp16_st_local_std",
                "fp16_st_local_trials",
            ]
        )
        for d in d_list:
            for r in r_list:
                for p in p_list:
                    results = []
                    seq_len = (r + 1) * (d * d - 1)
                    fc_ok = seq_len <= MAX_FC_SEQ_LEN
                    _cuda_clear()

                    trials = time_pymatching(d, r, p, batch_size, trials=num_trials)
                    results.extend(_summarize(trials))
                    _cuda_clear()

                    if fc_ok:
                        trials = time_ml_model(
                            d, r, p, batch_size, "fc_transformer", trials=num_trials, use_compile=use_compile
                        )
                        results.extend(_summarize(trials))
                    else:
                        results.extend([float("nan"), float("nan"), "[]"])
                    _cuda_clear()

                    trials = time_ml_model(d, r, p, batch_size, "st_local", trials=num_trials, use_compile=use_compile)
                    results.extend(_summarize(trials))
                    _cuda_clear()

                    if fc_ok:
                        trials = time_ml_model(
                            d, r, p, batch_size, "fc_transformer", "bf16", trials=num_trials, use_compile=use_compile
                        )
                        results.extend(_summarize(trials))
                    else:
                        results.extend([float("nan"), float("nan"), "[]"])
                    _cuda_clear()

                    trials = time_ml_model(
                        d, r, p, batch_size, "st_local", "bf16", trials=num_trials, use_compile=use_compile
                    )
                    results.extend(_summarize(trials))
                    _cuda_clear()

                    if fc_ok:
                        trials = time_ml_model(
                            d, r, p, batch_size, "fc_transformer", "fp16", trials=num_trials, use_compile=use_compile
                        )
                        results.extend(_summarize(trials))
                    else:
                        results.extend([float("nan"), float("nan"), "[]"])
                    _cuda_clear()

                    trials = time_ml_model(
                        d, r, p, batch_size, "st_local", "fp16", trials=num_trials, use_compile=use_compile
                    )
                    results.extend(_summarize(trials))

                    writer.writerow([d, r, p, *results])
                    f.flush()
                    _cuda_clear()


if __name__ == "__main__":
    main()
