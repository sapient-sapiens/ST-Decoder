import torch
import torch.nn as nn
import numpy as np
from functools import lru_cache

try:
    from .circuit import get_3D_surface, meta_data_surface, stabilizer_labels_surface  # type: ignore
except Exception:  # pragma: no cover
    from circuit import get_3D_surface, meta_data_surface, stabilizer_labels_surface  # type: ignore

try:
    from torch_geometric.nn import GraphConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

class Conv3DDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.d = d
        self.r = r
        _, X, Y, T = get_3D_surface(self.d, self.r)
        self.input_shape = (1, X, Y, T)
        '''
        self.net = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv3d(256, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x).flatten(1)
        x = self.decoder(x)
        return x

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# GNN Decoder (Lange et al. style)
# ============================================================================

@lru_cache(maxsize=32)
def _build_detector_metadata(d: int, r: int):
    """Precompute detector (x, y, t) coordinates and stabilizer types for a given (d, r).
    
    Returns:
        coords: np.ndarray of shape [num_detectors, 3] with (x, y, t)
        stab_types: np.ndarray of shape [num_detectors] with 0 (Z) or 1 (X)
        edge_index_full: np.ndarray of shape [2, num_edges] for full connectivity within distance threshold
    """
    md = meta_data_surface(d=d, r=r)
    coords = np.array(md, dtype=np.float32)  # [N, 3] with (x, y, t)
    
    # Normalize coordinates to [0, 1] range for better NN learning
    coords_norm = coords.copy()
    for dim in range(3):
        max_val = coords[:, dim].max()
        if max_val > 0:
            coords_norm[:, dim] = coords[:, dim] / max_val
    
    stab_types = stabilizer_labels_surface(d=d, r=r).numpy().astype(np.float32)
    
    # Precompute edges based on spatial-temporal proximity
    # Connect detectors within distance threshold (Manhattan distance ≤ 4 in original coords)
    N = len(md)
    src_list, dst_list = [], []
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = abs(coords[i, 0] - coords[j, 0])
            dy = abs(coords[i, 1] - coords[j, 1])
            dt = abs(coords[i, 2] - coords[j, 2])
            # Connect if close in space-time (Manhattan distance based)
            if dx <= 2 and dy <= 2 and dt <= 1:
                src_list.extend([i, j])
                dst_list.extend([j, i])
    
    edge_index_full = np.array([src_list, dst_list], dtype=np.int64)
    
    return coords_norm, stab_types, edge_index_full


def detections_to_graph_batch(detections: torch.Tensor, d: int, r: int, device: torch.device = None):
    """Convert a batch of detection tensors to a PyG Batch object.
    
    Args:
        detections: Tensor of shape [B, num_detectors] with values in {-1, 1} or {0, 1}
        d: code distance
        r: number of rounds
        device: target device
        
    Returns:
        PyG Batch object with node features [x, y, t, stab_type, detection_value]
    """
    if not HAS_PYG:
        raise RuntimeError("torch_geometric is required for GNN decoder. Install with: pip install torch_geometric")
    
    if device is None:
        device = detections.device
    
    coords_norm, stab_types, edge_index_full = _build_detector_metadata(d, r)
    coords_t = torch.from_numpy(coords_norm).to(device)  # [N, 3]
    stab_t = torch.from_numpy(stab_types).to(device)  # [N]
    edge_index_t = torch.from_numpy(edge_index_full).to(device)  # [2, E]
    
    B, N = detections.shape
    
    # Normalize detection values to [0, 1] if they're in {-1, 1}
    det_vals = detections.float()
    if det_vals.min() < 0:
        det_vals = (det_vals + 1) / 2  # Map {-1, 1} -> {0, 1}
    det_vals = det_vals.to(device)
    
    # Build list of Data objects for batching
    data_list = []
    for b in range(B):
        # Node features: [x, y, t, stab_type, detection_value]
        node_features = torch.cat([
            coords_t,  # [N, 3]
            stab_t.unsqueeze(1),  # [N, 1]
            det_vals[b].unsqueeze(1),  # [N, 1]
        ], dim=1)  # [N, 5]
        
        # Edge weights based on detection values at endpoints (optional)
        edge_attr = None
        
        data = Data(x=node_features, edge_index=edge_index_t, edge_attr=edge_attr)
        data_list.append(data)
    
    batch = Batch.from_data_list(data_list)
    return batch


class GNNDecoder(nn.Module):
    """Graph Neural Network decoder for quantum error correction.
    
    Based on Lange et al. (Phys. Rev. Research 7, 023181, 2025).
    Uses GraphConv layers followed by global mean pooling and MLP classifier.
    """
    
    def __init__(
        self,
        hidden_channels_gcn: list[int] = None,
        hidden_channels_mlp: list[int] = None,
        num_node_features: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not HAS_PYG:
            raise RuntimeError("torch_geometric is required for GNN decoder")
        
        if hidden_channels_gcn is None:
            hidden_channels_gcn = [32, 128, 256, 256, 128]
        if hidden_channels_mlp is None:
            hidden_channels_mlp = [128, 64]
        
        # GCN layers
        channels = [num_node_features] + hidden_channels_gcn
        self.graph_layers = nn.ModuleList([
            GraphConv(in_ch, out_ch)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])
        self.graph_norms = nn.ModuleList([
            nn.LayerNorm(out_ch) for out_ch in hidden_channels_gcn
        ])
        
        # MLP classifier
        channels = hidden_channels_gcn[-1:] + hidden_channels_mlp
        self.dense_layers = nn.ModuleList([
            nn.Linear(in_ch, out_ch)
            for in_ch, out_ch in zip(channels[:-1], channels[1:])
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_channels_mlp[-1], 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_attr: torch.Tensor = None):
        """Forward pass for PyG-style input.
        
        Args:
            x: Node features [total_nodes, num_features]
            edge_index: Edge connectivity [2, total_edges]
            batch: Batch assignment for each node [total_nodes]
            edge_attr: Optional edge attributes
        """
        # Graph convolution layers
        for conv, norm in zip(self.graph_layers, self.graph_norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Global pooling to get graph-level embedding
        x = global_mean_pool(x, batch)  # [B, hidden]
        
        # MLP classifier
        for layer in self.dense_layers:
            x = layer(x)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # Output
        out = self.output_layer(x)  # [B, 1]
        return out
    
    def forward_batch(self, batch_data):
        """Convenience method for PyG Batch objects."""
        return self.forward(batch_data.x, batch_data.edge_index, batch_data.batch, batch_data.edge_attr)


class GNNDecoderWrapper(nn.Module):
    """Wrapper that accepts raw detection tensors and handles graph construction internally.
    
    This allows the GNN to be used with the same training infrastructure as other models.
    """
    
    def __init__(
        self,
        d: int,
        r: int,
        hidden_channels_gcn: list[int] = None,
        hidden_channels_mlp: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d = d
        self.r = r
        self.gnn = GNNDecoder(
            hidden_channels_gcn=hidden_channels_gcn,
            hidden_channels_mlp=hidden_channels_mlp,
            num_node_features=5,
            dropout=dropout,
        )
    
    def forward(self, detections: torch.Tensor):
        """Forward pass from raw detections.
        
        Args:
            detections: Tensor of shape [B, num_detectors] with detection values
            
        Returns:
            logits: Tensor of shape [B, 1] 
        """
        batch_data = detections_to_graph_batch(detections, self.d, self.r, device=detections.device)
        out = self.gnn.forward_batch(batch_data)
        # Return shape [B, 1] to match other models that return [B, seq] then take mean
        # The training code does seq_logits.mean(dim=1), so we need [B, 1] 
        return out


class MultiConfigGNNWrapper(nn.Module):
    """Multi-config GNN that shares weights across different (d, r) configurations.
    
    For general training across multiple surface code sizes. The GNN weights are shared;
    only the graph structure differs per (d, r) config. Infers config from input shape.
    """
    
    def __init__(
        self,
        configs: list[tuple[int, int]],
        hidden_channels_gcn: list[int] = None,
        hidden_channels_mlp: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.configs = configs
        # Single shared GNN (weights don't depend on d, r)
        self.gnn = GNNDecoder(
            hidden_channels_gcn=hidden_channels_gcn,
            hidden_channels_mlp=hidden_channels_mlp,
            num_node_features=5,
            dropout=dropout,
        )
        # Build lookup from sequence length to (d, r)
        # Actual seq_len from surface code data is r * (d*d - 1)
        self._seqlen_to_config: dict[int, tuple[int, int]] = {}
        for d, r in configs:
            seq_len = r * (d * d - 1)
            self._seqlen_to_config[seq_len] = (d, r)
    
    def forward(self, detections: torch.Tensor):
        """Forward pass from raw detections.
        
        Args:
            detections: Tensor of shape [B, num_detectors] with detection values
            
        Returns:
            logits: Tensor of shape [B, 1] 
        """
        seq_len = detections.shape[1]
        if seq_len not in self._seqlen_to_config:
            raise ValueError(f"Unknown sequence length {seq_len}. Valid configs: {self.configs}")
        d, r = self._seqlen_to_config[seq_len]
        batch_data = detections_to_graph_batch(detections, d, r, device=detections.device)
        out = self.gnn.forward_batch(batch_data)
        return out
