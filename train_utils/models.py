"""Model construction, checkpoint loading, and multi-(d,r) factory."""

from __future__ import annotations

import os
from typing import Optional, Set, Tuple

import torch
from torch import nn

from train_utils.config import (
    DROPOUT,
    GNN_DROPOUT,
    GNN_HIDDEN_GCN,
    GNN_HIDDEN_MLP,
    PAIRS,
    ST_LOCAL_SPATIAL_RADIUS,
    ST_TEMPORAL_WINDOW,
    ST_LOCAL_D_MODEL,
    ST_LOCAL_FFN_DIM,
    ST_LOCAL_N_LAYERS,
    ST_LOCAL_S_NHEAD,
    ST_LOCAL_T_NHEAD,
    VANILLA_D_MODEL,
    VANILLA_FFN_DIM,
    VANILLA_N_HEAD,
    VANILLA_N_LAYERS,
)
from src.other_models import Conv3DDecoder, GNNDecoderWrapper, MultiConfigGNNWrapper
from src.transformer_models import (
    SpatioTemporalFullAttentionTransformer,
    SpatioTemporalLocalTransformer,
    VanillaTransformer,
)

# --- Checkpoint helpers ---


def infer_configs_from_state(state: dict) -> Optional[list[tuple[int, int]]]:
    """Extract all (d, r) configs from PE bank keys in a state dict."""
    configs: Set[Tuple[int, int]] = set()
    for k in state.keys():
        if "pe_bank.alpha." in k or "pe_bank.mods." in k:
            parts = str(k).split(".")
            for part in parts:
                if part.startswith("d") and "_r" in part:
                    try:
                        d_str, r_str = part.split("_")
                        d = int(d_str[1:])
                        r = int(r_str[1:])
                        configs.add((d, r))
                    except (ValueError, IndexError):
                        continue
    return sorted(configs) if configs else None


def filter_state_by_shape(model: nn.Module, state: dict, tag: str) -> dict:
    """Drop keys with mismatched tensor shapes to avoid load_state_dict errors."""
    model_state = model.state_dict()
    filtered: dict = {}
    skipped: list = []
    for k, v in state.items():
        if k not in model_state:
            continue
        try:
            if model_state[k].shape != v.shape:
                skipped.append(k)
                continue
        except Exception:
            skipped.append(k)
            continue
        filtered[k] = v
    if skipped:
        preview = ", ".join(skipped[:5])
        suffix = "..." if len(skipped) > 5 else ""
        print(f"[{tag}] Skipping {len(skipped)} keys with shape mismatch: {preview}{suffix}")
    return filtered


def normalize_state_dict(model: nn.Module, state: dict) -> dict:
    """Handle compiled checkpoints by stripping _orig_mod prefix when needed."""
    model_keys = set(model.state_dict().keys())
    if any(k in model_keys for k in state.keys()):
        return state
    stripped = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    if any(k in model_keys for k in stripped.keys()):
        return stripped
    return state


def _try_compile(model: nn.Module) -> nn.Module:
    try:
        return torch.compile(model)  # type: ignore[attr-defined]
    except Exception:
        return model


def build_model(
    d: int,
    r: int,
    *,
    d_model: int,
    n_layers: int,
    s_nhead: int,
    t_nhead: int,
    ffn: int,
    dropout: float,
    init_ckpt: Optional[str] = None,
    model_type: str = "spatiotemporal_local",
) -> nn.Module:
    """Single-(d,r) model for fine-tuning / evaluation."""

    if model_type == "3d_cnn":
        model = Conv3DDecoder()
        model = _try_compile(model)
        if init_ckpt and os.path.isfile(init_ckpt):
            state = torch.load(init_ckpt, map_location="cpu").get("model_state")
            if state:
                model_keys = set(model.state_dict().keys())
                ckpt_keys = set(state.keys())
                if model_keys == ckpt_keys:
                    model.load_state_dict(state, strict=True)
                else:
                    new_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
                    model.load_state_dict(new_state, strict=False)
                print(f"[3d_cnn] Loaded checkpoint from {init_ckpt}")
        return model

    if model_type == "vanilla":
        if init_ckpt and os.path.isfile(init_ckpt):
            ckpt = torch.load(init_ckpt, map_location="cpu")
            cfg = ckpt.get("config", {}) or {}
            v_d_model = int(cfg.get("d_model", VANILLA_D_MODEL))
            v_n_layers = int(cfg.get("n_layers", VANILLA_N_LAYERS))
            v_n_head = int(cfg.get("n_head", VANILLA_N_HEAD))
            v_ffn = int(cfg.get("ffn_dim", VANILLA_FFN_DIM) or VANILLA_FFN_DIM)
            v_dropout = float(cfg.get("dropout", dropout) or dropout)
            configs = ckpt.get("configs") or [(d, r)]
        else:
            v_d_model, v_n_layers, v_n_head, v_ffn, v_dropout = (
                VANILLA_D_MODEL,
                VANILLA_N_LAYERS,
                VANILLA_N_HEAD,
                VANILLA_FFN_DIM,
                dropout,
            )
            configs = [(d, r)]
        model = VanillaTransformer(
            d=d,
            r=r,
            d_model=v_d_model,
            n_layers=v_n_layers,
            n_head=v_n_head,
            ffn=v_ffn,
            dropout=v_dropout,
            configs=configs,
        )
        model = _try_compile(model)
        if init_ckpt and os.path.isfile(init_ckpt):
            state = torch.load(init_ckpt, map_location="cpu").get("model_state")
            if state:
                normalized = normalize_state_dict(model, state)
                filtered = filter_state_by_shape(model, normalized, "vanilla")
                model.load_state_dict(filtered, strict=False)
        return model

    if model_type == "spatiotemporal_local":
        if init_ckpt and os.path.isfile(init_ckpt):
            ckpt = torch.load(init_ckpt, map_location="cpu")
            cfg = ckpt.get("config", {}) or {}
            l_d_model = int(cfg.get("d_model", d_model))
            l_n_layers = int(cfg.get("n_layers", n_layers))
            l_s_nhead = int(cfg.get("s_nhead", s_nhead))
            l_t_nhead = int(cfg.get("t_nhead", t_nhead))
            l_ffn = int(cfg.get("ffn_dim", ffn) or ffn)
            l_dropout = float(cfg.get("dropout", dropout) or dropout)
            spatial_radius = int(cfg.get("spatial_radius", ST_LOCAL_SPATIAL_RADIUS))
            configs = infer_configs_from_state(ckpt.get("model_state", {})) or ckpt.get("configs") or [(d, r)]
            ckpt_d = ckpt.get("d", d)
            ckpt_r = ckpt.get("rounds", r)
            model = SpatioTemporalLocalTransformer(
                d=ckpt_d,
                r=ckpt_r,
                d_model=l_d_model,
                n_layers=l_n_layers,
                n_head=l_s_nhead,
                ffn=l_ffn,
                dropout=l_dropout,
                s_nhead=l_s_nhead,
                t_nhead=l_t_nhead,
                t_window_size=ST_TEMPORAL_WINDOW,
                configs=configs,
                spatial_radius=spatial_radius,
            )
        else:
            model = SpatioTemporalLocalTransformer(
                d=d,
                r=r,
                d_model=d_model,
                n_layers=n_layers,
                n_head=s_nhead,
                ffn=ffn,
                dropout=dropout,
                s_nhead=s_nhead,
                t_nhead=t_nhead,
                t_window_size=ST_TEMPORAL_WINDOW,
                configs=[(d, r)],
                spatial_radius=ST_LOCAL_SPATIAL_RADIUS,
            )
        model = _try_compile(model)
        if init_ckpt and os.path.isfile(init_ckpt):
            state = torch.load(init_ckpt, map_location="cpu").get("model_state")
            if state:
                normalized = normalize_state_dict(model, state)
                filtered = filter_state_by_shape(model, normalized, "spatiotemporal_local")
                model.load_state_dict(filtered, strict=False)
        return model

    if model_type == "spatiotemporal_full":
        if init_ckpt and os.path.isfile(init_ckpt):
            ckpt = torch.load(init_ckpt, map_location="cpu")
            cfg = ckpt.get("config", {}) or {}
            f_d_model = int(cfg.get("d_model", d_model))
            f_n_layers = int(cfg.get("n_layers", n_layers))
            f_s_nhead = int(cfg.get("s_nhead", s_nhead))
            f_t_nhead = int(cfg.get("t_nhead", t_nhead))
            f_ffn = int(cfg.get("ffn_dim", ffn) or ffn)
            f_dropout = float(cfg.get("dropout", dropout) or dropout)
            configs = infer_configs_from_state(ckpt.get("model_state", {})) or ckpt.get("configs") or [(d, r)]
            ckpt_d = ckpt.get("d", d)
            ckpt_r = ckpt.get("rounds", r)
            model = SpatioTemporalFullAttentionTransformer(
                d=ckpt_d,
                r=ckpt_r,
                d_model=f_d_model,
                n_layers=f_n_layers,
                n_head=f_s_nhead,
                ffn=f_ffn,
                dropout=f_dropout,
                s_nhead=f_s_nhead,
                t_nhead=f_t_nhead,
                configs=configs,
            )
        else:
            model = SpatioTemporalFullAttentionTransformer(
                d=d,
                r=r,
                d_model=d_model,
                n_layers=n_layers,
                n_head=s_nhead,
                ffn=ffn,
                dropout=dropout,
                s_nhead=s_nhead,
                t_nhead=t_nhead,
                configs=[(d, r)],
            )
        model = _try_compile(model)
        if init_ckpt and os.path.isfile(init_ckpt):
            state = torch.load(init_ckpt, map_location="cpu").get("model_state")
            if state:
                normalized = normalize_state_dict(model, state)
                filtered = filter_state_by_shape(model, normalized, "spatiotemporal_full")
                model.load_state_dict(filtered, strict=False)
        return model

    if model_type == "gnn":
        if init_ckpt and os.path.isfile(init_ckpt):
            ckpt = torch.load(init_ckpt, map_location="cpu")
            cfg = ckpt.get("config", {}) or {}
            g_hidden_gcn = cfg.get("hidden_gcn", GNN_HIDDEN_GCN)
            g_hidden_mlp = cfg.get("hidden_mlp", GNN_HIDDEN_MLP)
            g_dropout = float(cfg.get("dropout", GNN_DROPOUT) or GNN_DROPOUT)
        else:
            g_hidden_gcn, g_hidden_mlp, g_dropout = GNN_HIDDEN_GCN, GNN_HIDDEN_MLP, GNN_DROPOUT
        model = GNNDecoderWrapper(
            d=d,
            r=r,
            hidden_channels_gcn=g_hidden_gcn,
            hidden_channels_mlp=g_hidden_mlp,
            dropout=g_dropout,
        )
        if init_ckpt and os.path.isfile(init_ckpt):
            state = torch.load(init_ckpt, map_location="cpu").get("model_state")
            if state:
                model.load_state_dict(state, strict=False)
        return model

    raise ValueError(f"Unknown model_type: {model_type}")


def build_model_multitask(model_type: str) -> nn.Module:
    """Model with multi-(d,r) heads/banks for `general` training over PAIRS."""

    if model_type == "spatiotemporal_local":
        model = SpatioTemporalLocalTransformer(
            d_model=ST_LOCAL_D_MODEL,
            n_layers=ST_LOCAL_N_LAYERS,
            n_head=ST_LOCAL_S_NHEAD,
            ffn=ST_LOCAL_FFN_DIM,
            dropout=DROPOUT,
            s_nhead=ST_LOCAL_S_NHEAD,
            t_nhead=ST_LOCAL_T_NHEAD,
            t_window_size=ST_TEMPORAL_WINDOW,
            configs=PAIRS,
            spatial_radius=ST_LOCAL_SPATIAL_RADIUS,
        )
    elif model_type == "vanilla":
        model = VanillaTransformer(
            d_model=VANILLA_D_MODEL,
            n_layers=VANILLA_N_LAYERS,
            n_head=VANILLA_N_HEAD,
            ffn=VANILLA_FFN_DIM,
            dropout=DROPOUT,
            configs=PAIRS,
        )
    elif model_type == "3d_cnn":
        model = Conv3DDecoder()
    elif model_type == "gnn":
        model = MultiConfigGNNWrapper(
            configs=PAIRS,
            hidden_channels_gcn=GNN_HIDDEN_GCN,
            hidden_channels_mlp=GNN_HIDDEN_MLP,
            dropout=GNN_DROPOUT,
        )
        return model
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return _try_compile(model)


__all__ = [
    "build_model",
    "build_model_multitask",
    "infer_configs_from_state",
    "filter_state_by_shape",
    "normalize_state_dict",
]
