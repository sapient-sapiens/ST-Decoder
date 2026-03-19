"""Datasets and DataLoaders for syndrome / flip-label training."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.circuit import (
    get_circuit_surface,
    get_circuit_surface_biased as get_circuit_biased,
    get_circuit_surface_coherent as get_circuit_coherent,
    get_circuit_surface_spectator as get_circuit_spectator,
    get_data_surface as get_data,
    pad_surface as pad,
)


class DetectionDataset(Dataset):
    """Syndrome tokens (or 3D CNN tensor) with logical-flip labels."""

    def __init__(self, detections: torch.Tensor, flips: torch.Tensor) -> None:
        if detections.ndim == 5:
            self.tokens = detections
        else:
            self.tokens = ((detections + 1) / 2).to(dtype=torch.long)
        self.labels = flips.to(dtype=torch.float32)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.labels[idx]


def build_loaders(
    d: int,
    r: int,
    p: float,
    train_size: int,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    *,
    use_padding: bool = False,
    transform=None,
    p_gate: float | None = None,
    p_measurement: float | None = None,
    noise_model: str = "depolarizing",
    noise_bias: float = 100.0,
    noise_angle: float = 0.1,
    noise_spectator: float = 0.01,
    precomputed_data: str | None = None,
) -> Tuple[DataLoader, DataLoader]:
    if precomputed_data is not None:
        print(f"Loading precomputed data from {precomputed_data}")
        data = torch.load(precomputed_data)
        detections = data["detections"]
        flips = data["flips"]
        print(f"Loaded {len(detections)} samples")
    else:
        if noise_model == "biased":
            circuit = get_circuit_biased(d=d, r=r, p=p, bias=noise_bias)
        elif noise_model == "coherent":
            circuit = get_circuit_coherent(d=d, r=r, p=p, angle=noise_angle)
        elif noise_model == "spectator":
            circuit = get_circuit_spectator(d=d, r=r, p=p, p_spectator=noise_spectator)
        else:
            pg = p_gate if p_gate is not None else p
            pm = p_measurement if p_measurement is not None else p
            circuit = get_circuit_surface(d=d, r=r, p_gate=pg, p_measurement=pm)
        detections, flips, _ = get_data(circuit, train_size=train_size)

    if use_padding:
        detections = pad(d, r)(detections)
    if transform is not None:
        detections = transform(detections)

    dataset = DetectionDataset(detections, flips)
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    split_generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=split_generator)

    pin = torch.cuda.is_available()
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )
    return (
        DataLoader(train_set, shuffle=True, **kwargs),
        DataLoader(val_set, shuffle=False, **kwargs),
    )


__all__ = ["DetectionDataset", "build_loaders"]
