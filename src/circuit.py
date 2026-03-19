import math
import re
from functools import lru_cache
from typing import Any

import numpy as np
import stim
import torch


def get_circuit_surface(d: int = 5, r: int = 5, p_gate: float = 0.005, p_measurement: float = 0.005):
    circuit_cls: Any = getattr(stim, "Circuit")
    return circuit_cls.generated(
        "surface_code:rotated_memory_z",
        rounds=r,
        distance=d,
        after_clifford_depolarization=p_gate,
        before_measure_flip_probability=p_measurement,
        before_round_data_depolarization=p_gate,
    )


def get_circuit_surface_biased(d: int = 5, r: int = 5, p: float = 0.005, bias: float = 100.0):
    """Surface code memory circuit with Z-biased noise."""
    p_z = p * bias / (bias + 1)
    p_xy = p / (2 * (bias + 1))

    circuit_cls: Any = getattr(stim, "Circuit")
    base = circuit_cls.generated(
        "surface_code:rotated_memory_z",
        rounds=r,
        distance=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )

    circuit_str = str(base)
    circuit_str = re.sub(
        r"DEPOLARIZE1\([^)]+\)",
        f"PAULI_CHANNEL_1({p_xy}, {p_xy}, {p_z})",
        circuit_str,
    )
    return stim.Circuit(circuit_str)


def get_circuit_surface_coherent(d: int = 5, r: int = 5, p: float = 0.005, angle: float = 0.1):
    """Surface code memory circuit with coherent over-rotation errors."""
    p_coherent = math.sin(angle / 2) ** 2
    circuit_cls: Any = getattr(stim, "Circuit")
    base = circuit_cls.generated(
        "surface_code:rotated_memory_z",
        rounds=r,
        distance=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p_coherent,
    )

    circuit_str = str(base)
    circuit_str = re.sub(
        r"DEPOLARIZE1\([^)]+\)",
        f"PAULI_CHANNEL_1(0, 0, {p_coherent})",
        circuit_str,
    )
    return stim.Circuit(circuit_str)


def get_circuit_surface_spectator(d: int = 5, r: int = 5, p: float = 0.005, p_spectator: float = 0.01):
    """Surface code with spectator qubit errors (2Q gate crosstalk)."""
    circuit_cls: Any = getattr(stim, "Circuit")
    base = circuit_cls.generated(
        "surface_code:rotated_memory_z",
        rounds=r,
        distance=d,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )

    circuit_str = str(base)

    # Parse qubit coordinates
    pattern_qubit = r"QUBIT_COORDS\((\d+), (\d+)\) (\d+)"
    matches = re.findall(pattern_qubit, circuit_str)
    qubit_coords = {int(q): (int(x), int(y)) for x, y, q in matches}

    # Find data qubits from R instruction (qubits that are reset)
    r_match = re.search(r"^R (.+)$", circuit_str, re.MULTILINE)
    if not r_match:
        return stim.Circuit(circuit_str)
    all_qubits = [int(q) for q in r_match.group(1).split()]

    # Find ancilla qubits from MR instruction
    mr_match = re.search(r"^MR (.+)$", circuit_str, re.MULTILINE)
    if not mr_match:
        return stim.Circuit(circuit_str)
    ancilla_qubits = set(int(q) for q in mr_match.group(1).split())

    data_qubits = [q for q in all_qubits if q not in ancilla_qubits]
    if len(data_qubits) < 3:
        return stim.Circuit(circuit_str)

    def manhattan_dist(q1, q2):
        if q1 not in qubit_coords or q2 not in qubit_coords:
            return float("inf")
        x1, y1 = qubit_coords[q1]
        x2, y2 = qubit_coords[q2]
        return abs(x1 - x2) + abs(y1 - y2)

    # Find next-nearest neighbor pairs (distance 4)
    spectator_pairs = []
    for i, q1 in enumerate(data_qubits):
        for q2 in data_qubits[i + 1 :]:
            dist = manhattan_dist(q1, q2)
            if dist == 4:
                spectator_pairs.append((q1, q2))

    if not spectator_pairs:
        # Fallback: use distance 2 pairs
        for i, q1 in enumerate(data_qubits):
            for q2 in data_qubits[i + 1 :]:
                dist = manhattan_dist(q1, q2)
                if dist == 2:
                    spectator_pairs.append((q1, q2))

    if not spectator_pairs:
        return stim.Circuit(circuit_str)

    corr_lines = []
    for q1, q2 in spectator_pairs[:12]:
        corr_lines.append(f"CORRELATED_ERROR({p_spectator}) Z{q1} Z{q2}")
    corr_block = "\n" + "\n".join(corr_lines)

    # Insert before each MR line (end of each round)
    circuit_str = re.sub(r"\nMR ", corr_block + "\nMR ", circuit_str)

    return stim.Circuit(circuit_str)


def get_data_surface(circuit, train_size: int = 5 * 10**6):
    sampler = circuit.compile_detector_sampler()
    detections, flips = sampler.sample(train_size, separate_observables=True)
    detections = detections.astype(int) * 2 - 1
    detections, flips = torch.Tensor(detections), torch.Tensor(flips.astype(int).flatten())
    num_input = detections.shape[1]
    return detections, flips, num_input


def get_info_surface(text: str, d: int = 5, r: int = 5):
    qubit_to_coord = {}
    coord_to_qubit = {}
    pattern_qubit = r"QUBIT_COORDS\((\d+), (\d+)\) (\d+)"
    matches_qubit = re.findall(pattern_qubit, text)
    for x_str, y_str, z_str in matches_qubit:
        x = int(x_str)
        y = int(y_str)
        z = int(z_str)
        qubit_to_coord[z] = (x, y)
        coord_to_qubit[(x, y)] = z

    pattern_det = r"DETECTOR\((-?\d+), (-?\d+), (-?\d+)\)"
    matches_det = re.findall(pattern_det, text)
    xy = [(int(x), int(y)) for (x, y, _t) in matches_det]
    if len(xy) == 0:
        raise RuntimeError("No DETECTOR coordinates parsed from circuit string")

    min_x = min(x for x, _ in xy)
    min_y = min(y for _, y in xy)
    xy_shifted = [(x - min_x, y - min_y) for (x, y) in xy]

    group1_xy, group2_xy, group3_xy = [], [], []
    for idx, (x, y) in enumerate(xy_shifted):
        if idx < (d**2 - 1) // 2:
            group1_xy.append((x, y))
        elif idx < (d**2 - 1) + (d**2 - 1) // 2:
            group2_xy.append((x, y))
        else:
            group3_xy.append((x, y))
    return group1_xy, group2_xy, group3_xy, qubit_to_coord, coord_to_qubit

def get_spatial_graph(d): 
    circuit = get_circuit_surface(d=d, r=3)
    _, g2, _, _, _ = get_info_surface(str(circuit), d=d, r=3)
    


@lru_cache(maxsize=None)
def get_3D_surface(d: int = 5, r: int = 5):
    circuit = get_circuit_surface(d=d, r=r)
    g1, g2, g3, _, _ = get_info_surface(str(circuit), d=d, r=r)

    coords = []
    for i in range(len(g1)):
        coords.append((g1[i][0], g1[i][1], 0))
    for j in range(1, r):
        for i in range(len(g2)):
            coords.append((g2[i][0], g2[i][1], j))
    for i in range(len(g3)):
        coords.append((g3[i][0], g3[i][1], r))

    X = max([c[0] for c in coords]) + 1
    Y = max([c[1] for c in coords]) + 1
    T = r + 1
    return coords, X, Y, T


def transform_3D(detections, d, r):
    coords, X, Y, T = get_3D_surface(d, r)
    coords_t = torch.as_tensor(coords, device=detections.device, dtype=torch.long)
    x = coords_t[:, 0]
    y = coords_t[:, 1]
    t = coords_t[:, 2]

    out = torch.zeros((detections.shape[0], 1, X, Y, T), dtype=torch.float32, device=detections.device)
    out[:, 0, x, y, t] = detections.float()
    return out


def maps_surface(qubit_to_coord, coord_to_qubit, ratio=2):
    mappings = []
    for i in range(ratio):
        for j in range(ratio):
            mapping = {}
            dirx = i * 4
            diry = j * 4
            for qubit, coord in qubit_to_coord.items():
                mapping[qubit] = coord_to_qubit[tuple(a + b for a, b in zip(coord, (dirx, diry)))]
            mappings.append(mapping)
    return mappings


def meta_data_surface(d: int = 7, r: int = 5):
    circuit = get_circuit_surface(d=d, r=r)
    group1_xy, group2_xy, group3_xy, _qubit_to_coord, _coord_to_qubit = get_info_surface(str(circuit), d=d, r=r)
    ans = []

    def append_group(t: int, group_xy):
        for (x, y) in group_xy:
            ans.append((int(x), int(y), t))

    append_group(0, group1_xy)
    for t in range(1, r):
        append_group(t, group2_xy)
    append_group(r, group3_xy)

    assert len(ans) == r * (d**2 - 1), f"Unexpected metadata length: {len(ans)} vs {r * (d**2 - 1)}"
    return ans


def stabilizer_labels_surface(d: int, r: int) -> torch.Tensor:
    md = meta_data_surface(d=d, r=r)
    labels = torch.empty(len(md), dtype=torch.long)
    for i, (x, y, _t) in enumerate(md):
        labels[i] = int((x + y) & 1)
    return labels


def get_data_surface_with_labels(d: int, r: int, p: float, train_size: int = 5 * 10**6):
    circ = get_circuit_surface(d=d, r=r, p_gate=p, p_measurement=p)
    detections, flips, num_input = get_data_surface(circ, train_size=train_size)
    labels = stabilizer_labels_surface(d=d, r=r)
    return detections, flips, labels, num_input


def pad_surface(d, r):
    circuit = get_circuit_surface(d=d, r=r)
    g1, g2, g3, _, _ = get_info_surface(str(circuit), d=d, r=r)

    full = d * d - 1
    half = full // 2
    pos = {n: i for i, n in enumerate(g2)}

    idx1 = torch.tensor([pos[n] for n in g1], dtype=torch.long)
    idx2 = torch.tensor([pos[n] for n in g3], dtype=torch.long)

    def pad_transform(detections: torch.Tensor) -> torch.Tensor:
        B = detections.size(0)
        dev = detections.device
        left, mid, right = detections[:, :half], detections[:, half:-half], detections[:, -half:]

        i1 = idx1.to(dev)
        i2 = idx2.to(dev)
        outL = detections.new_zeros(B, full).scatter_(1, i1.unsqueeze(0).expand(B, -1), left)
        outR = detections.new_zeros(B, full).scatter_(1, i2.unsqueeze(0).expand(B, -1), right)

        return torch.cat([outL, mid, outR], dim=1)

    return pad_transform
