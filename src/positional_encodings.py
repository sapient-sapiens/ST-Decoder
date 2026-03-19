import torch
import torch.nn as nn
import math

try:
    from .circuit import meta_data_surface as meta_data, get_circuit_surface, get_info_surface  # type: ignore
except Exception:  # pragma: no cover
    from circuit import meta_data_surface as meta_data, get_circuit_surface, get_info_surface  # type: ignore

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import networkx as nx
import numpy as np


class SinusoidalEncoding(nn.Module):
    def __init__(self, maxlen, d_model):
        super().__init__()
        pe = torch.zeros(maxlen, d_model)
        idx = torch.arange(maxlen)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(idx[:, None] * div_term[None, :])
        pe[:, 1::2] = torch.cos(idx[:, None] * div_term[: pe[:, 1::2].shape[1]][None, :])
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :].to(dtype=x.dtype, device=x.device)


class PositionalEncoding(nn.Module):
    def __init__(self, d, r, d_model: int = 256):
        super().__init__()
        q, rem = divmod(d_model, 3)
        dx, dy, dz = q + (rem > 0), q + (rem > 1), q
        maxlen = (d * 2 - 1) * (r + 1)
        pe_x = SinusoidalEncoding(maxlen, dx).pe
        pe_y = SinusoidalEncoding(maxlen, dy).pe
        pe_z = SinusoidalEncoding(maxlen, dz).pe
        info = meta_data(d, r)
        assert len(info) == r * (d ** 2 - 1)
        pe = torch.zeros((d ** 2 - 1) * r, d_model)
        for i, (x, y, t) in enumerate(info):
            pe[i, :] = torch.cat([pe_x[x, :], pe_y[y, :], pe_z[t, :]], dim=0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :].to(dtype=x.dtype, device=x.device)


class SpatioTemporalPositionalEncoding(nn.Module):
    def __init__(self, d, r, d_model=256, dtype=torch.float32, device=None):
        super().__init__()
        self.d, self.r, self.d_model = d, r, d_model
        self.k = min(128, d_model)
        self.dtype, self.device = dtype, device
        self.proj = nn.Linear(self.k, d_model, bias=False)
        self.register_buffer("base_pe", None, persistent=False)
        self.build_pe()
        self._init_projection()

    def forward(self, x):
        base = self.base_pe[: x.size(1), :].to(dtype=x.dtype, device=x.device)
        enc = self.proj(base)
        return x + enc

    def build_pe(self):
        G = self._build_detector_graph()
        L = G.number_of_nodes()
        idx = {n: i for i, n in enumerate(G.nodes())}
        r, c, w = [], [], []
        for u, v, a in G.edges(data=True):
            i, j = idx[u], idx[v]
            ww = float(a.get("weight", 1.0))
            r += [i, j]
            c += [j, i]
            w += [ww, ww]
        W = sp.csr_matrix((w, (r, c)), shape=(L, L))
        deg = np.asarray(W.sum(1)).ravel()
        dinv2 = np.zeros_like(deg)
        m = deg > 0
        dinv2[m] = 1.0 / np.sqrt(deg[m])
        Dm = sp.diags(dinv2)
        Lsym = sp.eye(L) - Dm @ W @ Dm
        k_eff = max(self.k + 1, 2)
        if L - 1 > 0:
            k_eff = min(k_eff, L - 2)
        vals, vecs = eigsh(Lsym, k=k_eff, which="SM")
        o = np.argsort(vals)
        vals, vecs = vals[o], vecs[:, o]
        candidate_indices = [i for i, v in enumerate(vals) if v > 1e-9]
        keep = candidate_indices[: self.k] if candidate_indices else list(range(1, min(k_eff, self.k + 1)))
        U = vecs[:, keep]
        for j in range(U.shape[1]):
            col = U[:, j]
            if abs(col.min()) > abs(col.max()):
                U[:, j] = -col
        V = (Dm @ sp.csr_matrix(U)).toarray()
        Vt = torch.from_numpy(V).to(dtype=self.dtype, device=self.device)
        Vt = Vt / (Vt.std(0, keepdim=True) + 1e-8)
        self.base_pe = Vt.detach()

    def _init_projection(self):
        in_feats = int(self.base_pe.shape[1]) if self.base_pe is not None else self.k
        if self.proj.in_features != in_feats or self.proj.out_features != self.d_model:
            self.proj = nn.Linear(in_feats, self.d_model, bias=False)
        with torch.no_grad():
            weight = self.proj.weight
            weight.zero_()
            rows, cols = weight.shape
            n = min(rows, cols)
            if n > 0:
                eye = torch.eye(n, dtype=weight.dtype, device=weight.device)
                weight[:n, :n].copy_(eye)
            if rows > cols:
                nn.init.kaiming_uniform_(weight[n:, :], a=math.sqrt(5))

    def _build_detector_graph(self) -> nx.Graph:
        G = nx.Graph()
        circuit_text = str(get_circuit_surface(d=self.d, r=self.r))
        _g1, group, _g3, _q2c, _c2q = get_info_surface(circuit_text, d=self.d, r=self.r)
        space_edges = []
        for i in range(len(group)):
            for j in range(len(group)):
                x1, y1 = group[i]
                x2, y2 = group[j]
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dx == 2 and dy == 2:
                    space_edges.append((i, j, 4))
                elif max(dx, dy) == 2:
                    space_edges.append((i, j, 2))
        t = self.r + 1
        d = len(group)
        for i in range(t):
            for j in range(d):
                if i != 0:
                    G.add_edge(i * d + j, (i - 1) * d + j, weight=3)
            for edge in space_edges:
                G.add_edge(edge[0] + i * d, edge[1] + i * d, weight=edge[2])
        return G
