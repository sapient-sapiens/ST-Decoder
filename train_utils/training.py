"""Training loops, single-run fine-tuning, multitask general training, and CUDA runtime tweaks."""

from __future__ import annotations

import csv
import os
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.circuit import transform_3D
from train_utils.config import (
    BATCH_SIZE,
    DROPOUT,
    GNN_DROPOUT,
    GNN_HIDDEN_GCN,
    GNN_HIDDEN_MLP,
    LR,
    NUM_WORKERS,
    PAIRS,
    PS,
    ST_LOCAL_D_MODEL,
    ST_LOCAL_FFN_DIM,
    ST_LOCAL_N_LAYERS,
    ST_LOCAL_S_NHEAD,
    ST_LOCAL_T_NHEAD,
    ST_LOCAL_SPATIAL_RADIUS,
    VAL_RATIO,
    VANILLA_D_MODEL,
    VANILLA_FFN_DIM,
    VANILLA_N_HEAD,
    VANILLA_N_LAYERS,
    WEIGHT_DECAY,
)
from train_utils.data import build_loaders
from train_utils.models import build_model_multitask

warnings.filterwarnings("ignore")

# --- Optional PyTorch runtime (finetune) ---


def configure_training_warnings() -> None:
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=FutureWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)


def configure_cuda_sdp() -> None:
    """Prefer math SDP to avoid rare CUDA kernel configuration issues."""
    import torch as _torch

    try:
        _torch.backends.cuda.enable_flash_sdp(False)  # type: ignore[attr-defined]
        _torch.backends.cuda.enable_mem_efficient_sdp(False)  # type: ignore[attr-defined]
        _torch.backends.cuda.enable_math_sdp(True)  # type: ignore[attr-defined]
        _torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        _torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass


def configure_finetune_runtime() -> None:
    configure_training_warnings()
    configure_cuda_sdp()


# --- Epoch loops ---


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    n_samples = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for tokens, labels in pbar:
        tokens = tokens.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        seq_logits = model(tokens)
        logits = seq_logits.mean(dim=1)
        loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(labels.dtype)
            running_correct += (preds == labels).sum().item()
        running_loss += loss.item() * labels.size(0)
        n_samples += labels.size(0)
        pbar.set_postfix(
            loss=f"{running_loss / max(1, n_samples):.4f}",
            acc=f"{running_correct / max(1, n_samples):.4f}",
        )
    return running_loss / max(1, n_samples), running_correct / max(1, n_samples)


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_correct = 0
    n_samples = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for tokens, labels in pbar:
            tokens = tokens.to(device)
            labels = labels.to(device)
            seq_logits = model(tokens)
            logits = seq_logits.mean(dim=1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).to(labels.dtype)
            running_correct += (preds == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            pbar.set_postfix(
                loss=f"{running_loss / max(1, n_samples):.4f}",
                acc=f"{running_correct / max(1, n_samples):.4f}",
            )
    return running_loss / max(1, n_samples), running_correct / max(1, n_samples)


# --- Single (d,r,p) run ---


def run_one(
    model: nn.Module,
    d: int,
    r: int,
    p: float,
    epochs: int,
    train_size: int,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    save_dir: str,
    tag: str = "st",
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    use_3d: bool = False,
    use_padding: bool = True,
    p_gate: Optional[float] = None,
    p_measurement: Optional[float] = None,
    noise_model: str = "depolarizing",
    noise_bias: float = 100.0,
    noise_angle: float = 0.1,
    noise_spectator: float = 0.01,
    precomputed_data: Optional[str] = None,
) -> Tuple[str, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    noise_suffix = "" if noise_model == "depolarizing" else f"_{noise_model}"
    run_tag = f"{tag}_d{d}_r{r}_p{p}{noise_suffix}_{ts}"
    ckpt_best = os.path.join(save_dir, f"{run_tag}_best.pt")
    ckpt_final = os.path.join(save_dir, f"{run_tag}.pt")
    csv_path = os.path.join(save_dir, f"{run_tag}.csv")

    noise_kwargs: Dict[str, Any] = dict(
        noise_model=noise_model,
        noise_bias=noise_bias,
        noise_angle=noise_angle,
        noise_spectator=noise_spectator,
        precomputed_data=precomputed_data,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    with open(csv_path, "w", newline="") as csv_f:
        w = csv.writer(csv_f)
        w.writerow(
            [
                "timestamp",
                "model",
                "d",
                "r",
                "p",
                "noise_model",
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
            ]
        )
        csv_f.flush()

        best_acc = 0.0
        val_loss, val_acc = float("nan"), float("nan")

        tfm = (lambda det: transform_3D(det, d, r)) if use_3d else None
        _, val_loader = build_loaders(
            d=d,
            r=r,
            p=p,
            train_size=train_size,
            batch_size=batch_size,
            val_ratio=val_ratio,
            num_workers=num_workers,
            use_padding=use_padding,
            transform=tfm,
            p_gate=p_gate,
            p_measurement=p_measurement,
            **noise_kwargs,
        )

        ep_bar = tqdm(range(1, epochs + 1), desc=f"{tag} epochs")
        for ep in ep_bar:
            train_loader, _ = build_loaders(
                d=d,
                r=r,
                p=p,
                train_size=train_size,
                batch_size=batch_size,
                val_ratio=val_ratio,
                num_workers=num_workers,
                use_padding=use_padding,
                transform=tfm,
                p_gate=p_gate,
                p_measurement=p_measurement,
                **noise_kwargs,
            )
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                scaler=None,
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            ep_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
            )
            w.writerow(
                [
                    datetime.now().isoformat(),
                    tag,
                    d,
                    r,
                    p,
                    noise_model,
                    ep,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                ]
            )
            csv_f.flush()
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "d": d,
                        "rounds": r,
                        "configs": [(d, r)],
                        "config": {
                            "d_model": getattr(model, "d_model", None),
                            "n_layers": getattr(model, "n_layers", None),
                            "s_nhead": getattr(model, "s_nhead", None),
                            "t_nhead": getattr(model, "t_nhead", None),
                            "ffn_dim": getattr(model, "ffn", None),
                            "dropout": getattr(model, "dropout", None),
                            "model": tag,
                        },
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    ckpt_best,
                )

        torch.save(
            {
                "model_state": model.state_dict(),
                "d": d,
                "rounds": r,
                "final_val_loss": val_loss,
                "final_val_acc": val_acc,
            },
            ckpt_final,
        )
    return ckpt_best, csv_path


# --- Multitask general training ---


def epoch_even_mix_train(
    model: nn.Module,
    device: torch.device,
    train_size_total: int,
    optimizer: torch.optim.Optimizer,
    *,
    use_3d: bool = False,
    use_padding: bool = True,
) -> Tuple[float, float]:
    weights: List[Tuple[int, int, float, int]] = []
    total_w = 0
    for d, r in PAIRS:
        for p in PS:
            w = 1
            weights.append((d, r, p, w))
            total_w += w

    alloc: List[Tuple[int, int, float, int]] = []
    remainder = train_size_total
    for d, r, p, w in weights:
        n = max(1, int(train_size_total * (w / max(1, total_w))))
        alloc.append((d, r, p, n))
        remainder -= n
    i = 0
    while remainder != 0 and alloc:
        d, r, p, n = alloc[i % len(alloc)]
        if remainder > 0:
            n += 1
            remainder -= 1
        else:
            if n > 1:
                n -= 1
                remainder += 1
        alloc[i % len(alloc)] = (d, r, p, n)
        i += 1

    train_loaders = []
    for d, r, p, n_samples in alloc:
        tfm = (lambda det, _d=d, _r=r: transform_3D(det, _d, _r)) if use_3d else None
        tl, _ = build_loaders(
            d=d,
            r=r,
            p=p,
            train_size=n_samples,
            batch_size=BATCH_SIZE,
            val_ratio=VAL_RATIO,
            num_workers=NUM_WORKERS,
            use_padding=use_padding,
            transform=tfm,
        )
        train_loaders.append(tl)

    criterion = nn.BCEWithLogitsLoss()
    model.train()
    running_loss = 0.0
    running_correct = 0
    n_samples = 0

    iters = [iter(tl) for tl in train_loaders]
    active = list(range(len(iters)))
    total_batches = sum(len(tl) for tl in train_loaders)
    pbar = tqdm(total=total_batches, desc="train", leave=False)
    while active:
        next_active = []
        for idx in active:
            it = iters[idx]
            try:
                tokens, labels = next(it)
            except StopIteration:
                continue
            tokens = tokens.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            seq_logits = model(tokens)
            logits = seq_logits.mean(dim=1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).to(labels.dtype)
                running_correct += (preds == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            avg_loss = running_loss / max(1, n_samples)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")
            pbar.update(1)
            next_active.append(idx)
        active = next_active
    pbar.close()
    return running_loss / max(1, n_samples), running_correct / max(1, n_samples)


def eval_average(
    model: nn.Module,
    device: torch.device,
    val_train_size: int,
    *,
    use_3d: bool = False,
    use_padding: bool = True,
) -> Tuple[float, float, list]:
    vals, accs = [], []
    details = []
    criterion = nn.BCEWithLogitsLoss()
    total_cfgs = len(PAIRS) * len(PS)
    cfg_bar = tqdm(total=total_cfgs, desc="val", leave=False)
    for d, r in PAIRS:
        for p in PS:
            tfm = (lambda det, _d=d, _r=r: transform_3D(det, _d, _r)) if use_3d else None
            _, val_loader = build_loaders(
                d=d,
                r=r,
                p=p,
                train_size=val_train_size,
                batch_size=BATCH_SIZE,
                val_ratio=VAL_RATIO,
                num_workers=NUM_WORKERS,
                use_padding=use_padding,
                transform=tfm,
            )
            vloss, vacc = evaluate(model, val_loader, criterion, device)
            vals.append(vloss)
            accs.append(vacc)
            details.append((d, r, p, vloss, vacc))
            cfg_bar.set_postfix(
                avg_val_loss=f"{(sum(vals) / len(vals)):.4f}",
                avg_val_acc=f"{(sum(accs) / len(accs)):.4f}",
            )
            cfg_bar.update(1)
    cfg_bar.close()
    return float(sum(vals) / len(vals)), float(sum(accs) / len(accs)), details


def run_multitask_training(
    *,
    epochs: int,
    train_size_total: int,
    save_dir: str,
    model_type: str,
    val_train_size: int = 50_000,
    checkpoint_every: int = 10,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_3d = model_type == "3d_cnn"
    use_padding = model_type in ("spatiotemporal_local", "vanilla")
    model = build_model_multitask(model_type).to(device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type == "vanilla":
        run_tag = (
            f"vanilla_dmodel{VANILLA_D_MODEL}_L{VANILLA_N_LAYERS}_H{VANILLA_N_HEAD}_FFN{VANILLA_FFN_DIM}_{ts}"
        )
        model_config = {
            "d_model": VANILLA_D_MODEL,
            "n_layers": VANILLA_N_LAYERS,
            "n_head": VANILLA_N_HEAD,
            "ffn_dim": VANILLA_FFN_DIM,
            "dropout": DROPOUT,
            "model": "vanilla",
        }
    elif model_type == "gnn":
        gcn_depth = len(GNN_HIDDEN_GCN)
        gcn_width = GNN_HIDDEN_GCN[0] if GNN_HIDDEN_GCN else 0
        run_tag = f"gnn_gcn{gcn_depth}x{gcn_width}_mlp{GNN_HIDDEN_MLP}_{ts}"
        model_config = {
            "hidden_gcn": GNN_HIDDEN_GCN,
            "hidden_mlp": GNN_HIDDEN_MLP,
            "dropout": GNN_DROPOUT,
            "model": "gnn",
        }
    elif model_type == "3d_cnn":
        run_tag = f"3d_cnn_{ts}"
        model_config = {"model": "3d_cnn"}
    elif model_type == "spatiotemporal_local":
        run_tag = (
            f"st_local_dmodel{ST_LOCAL_D_MODEL}_L{ST_LOCAL_N_LAYERS}_S{ST_LOCAL_S_NHEAD}_T{ST_LOCAL_T_NHEAD}_FFN{ST_LOCAL_FFN_DIM}_R{ST_LOCAL_SPATIAL_RADIUS}_{ts}"
        )
        model_config = {
            "d_model": ST_LOCAL_D_MODEL,
            "n_layers": ST_LOCAL_N_LAYERS,
            "s_nhead": ST_LOCAL_S_NHEAD,
            "t_nhead": ST_LOCAL_T_NHEAD,
            "ffn_dim": ST_LOCAL_FFN_DIM,
            "dropout": DROPOUT,
            "spatial_radius": ST_LOCAL_SPATIAL_RADIUS,
            "model": model_type,
        }
    else:
        raise ValueError(f"Unhandled model type: {model_type}")

    os.makedirs(save_dir, exist_ok=True)
    ckpt_best_path = os.path.join(save_dir, f"{run_tag}_best.pt")
    ckpt_final_path = os.path.join(save_dir, f"{run_tag}.pt")
    csv_path = os.path.join(save_dir, f"{run_tag}.csv")
    cfg_csv_path = os.path.join(save_dir, f"{run_tag}_per_config.csv")

    with open(csv_path, "w", newline="") as csv_f, open(cfg_csv_path, "w", newline="") as cfg_f:
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["timestamp", "model", "epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        cfg_w = csv.writer(cfg_f)
        cfg_w.writerow(["timestamp", "model", "epoch", "d", "r", "p", "val_loss", "val_acc"])
        csv_f.flush()
        cfg_f.flush()

        best_val = float("inf")
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        last_val_loss, last_val_acc = float("nan"), float("nan")
        epoch_pbar = tqdm(range(1, epochs + 1), desc="epochs")
        for epoch in epoch_pbar:
            train_loss, train_acc = epoch_even_mix_train(
                model,
                device,
                train_size_total=train_size_total,
                optimizer=optimizer,
                use_3d=use_3d,
                use_padding=use_padding,
            )
            val_loss, val_acc, val_details = eval_average(
                model, device, val_train_size=val_train_size, use_3d=use_3d, use_padding=use_padding
            )
            scheduler.step()
            last_val_loss, last_val_acc = val_loss, val_acc
            epoch_pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                train_acc=f"{train_acc:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.4f}",
            )
            csv_w.writerow([datetime.now().isoformat(), model_type, epoch, train_loss, train_acc, val_loss, val_acc])
            csv_f.flush()
            now_ts = datetime.now().isoformat()
            for d, r, p, vloss, vacc in val_details:
                cfg_w.writerow([now_ts, model_type, epoch, d, r, p, vloss, vacc])
            cfg_f.flush()
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "configs": PAIRS,
                        "ps": PS,
                        "config": model_config,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    ckpt_best_path,
                )
            if checkpoint_every > 0 and epoch % checkpoint_every == 0:
                ckpt_epoch_path = os.path.join(save_dir, f"{run_tag}_epoch{epoch}.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "configs": PAIRS,
                        "ps": PS,
                        "config": model_config,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                    ckpt_epoch_path,
                )
        torch.save(
            {
                "model_state": model.state_dict(),
                "configs": PAIRS,
                "ps": PS,
                "config": model_config,
                "final_val_loss": last_val_loss,
                "final_val_acc": last_val_acc,
            },
            ckpt_final_path,
        )


__all__ = [
    "configure_training_warnings",
    "configure_cuda_sdp",
    "configure_finetune_runtime",
    "train_one_epoch",
    "evaluate",
    "run_one",
    "epoch_even_mix_train",
    "eval_average",
    "run_multitask_training",
]
