import torch
import torch.nn as nn
from functools import lru_cache

try:
    from . import positional_encodings as posenc  # type: ignore
except Exception:  # pragma: no cover
    import positional_encodings as posenc  # type: ignore

from math import sqrt
from .circuit import get_circuit_surface, get_info_surface


class BaseTransformer(nn.Module):
    def __init__(self, d: int, r: int, d_model: int = 256, n_layers: int = 6, n_head: int = 8, ffn: int = 512, dropout: float = 0.1, pe: nn.Module | None = None):
        super().__init__()
        self.d = d
        self.r = r
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_head = n_head
        self.ffn = ffn
        self.dropout = dropout
        encoder_layer = self._build_encoder_layer(d_model, n_head, ffn, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.PE = pe if pe is not None else self._build_positional_encoding(d, r, d_model)
        self.embedding = nn.Embedding(2, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, 1),
        )

    def _build_encoder_layer(self, d_model, n_head, ffn, dropout):
        return nn.TransformerEncoderLayer(d_model, n_head, ffn, dropout=dropout)

    def _build_positional_encoding(self, d, r, d_model):
        return posenc.PositionalEncoding(d, r, d_model)

    def forward(self, x, src_key_padding_mask=None):
        x = x.long()
        emb = self.embedding(x)
        emb = self.emb_norm(emb)
        emb = self.PE(emb)
        src = emb.transpose(0, 1)
        encoded = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        encoded = encoded.transpose(0, 1)
        out = self.decoder(encoded).squeeze(-1)
        return out


class PositionalEncodingBank(nn.Module):
    def __init__(self, d_model: int, configs: list[tuple[int, int]]):
        super().__init__()
        self.d_model = d_model
        self._keys: list[tuple[int, int]] = []
        self._len_map: dict[int, tuple[int, int]] = {}
        self.alpha = nn.ParameterDict()
        self.register_buffer("_dummy", torch.empty(0))
        self.mods = nn.ModuleDict()
        for (d, r) in configs:
            key = (int(d), int(r))
            if key in self._keys:
                continue
            name = f"d{d}_r{r}"
            pe = posenc.SpatioTemporalPositionalEncoding(d, r, d_model)
            self.mods[name] = pe
            self.alpha[name] = nn.Parameter(torch.tensor(1.0))
            self._keys.append(key)
            self._len_map[(r + 1) * (d * d - 1)] = key

    def spatial_for_length(self, seq_len: int) -> int:
        if seq_len not in self._len_map:
            raise RuntimeError(f"No PE for sequence length {seq_len}; available lengths: {list(self._len_map.keys())}")
        d, r = self._len_map[seq_len]
        return d * d - 1

    def pe_for_length(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if seq_len not in self._len_map:
            raise RuntimeError(f"No PE cached for seq_len={seq_len}")
        d, r = self._len_map[seq_len]
        name = f"d{d}_r{r}"
        pe_mod: nn.Module = self.mods[name]
        a: nn.Parameter = self.alpha[name]
        base = pe_mod.base_pe[:seq_len, :].to(device=device, dtype=dtype)
        enc = pe_mod.proj(base)
        return a * enc


class SpatioTemporalLocalTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        s_nhead,
        t_nhead,
        *,
        latent_size: int = 64,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        window_size: int = 3,
        spatial_radius: int = 4,
    ):
        super().__init__()
        self.s_nhead = s_nhead
        self.t_nhead = t_nhead
        self.latent_size = int(latent_size)
        self.dropout = dropout
        self.d = d_model
        self.d_ff = dim_feedforward
        self.window_size = int(window_size)
        self.spatial_radius = int(spatial_radius)
        self.t_att = nn.MultiheadAttention(self.d, t_nhead, dropout=dropout, batch_first=True)
        self.s_att = nn.MultiheadAttention(self.d, s_nhead, dropout=dropout, batch_first=True)
        self.n1 = nn.RMSNorm(d_model)
        self.n2 = nn.RMSNorm(d_model)
        self.n3 = nn.RMSNorm(d_model)
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.do3 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
        )

    @torch._dynamo.disable
    @lru_cache(maxsize=None)
    def _spatial_mask(self, s, device: torch.device):
        d = int(sqrt(s + 1))
        circuit = get_circuit_surface(d=d, r=3)
        _, g2, _, _, _ = get_info_surface(str(circuit), d=d, r=3)
        coords = torch.as_tensor(g2, dtype=torch.long, device="cpu")
        x = coords[:, 0].unsqueeze(1)
        y = coords[:, 1].unsqueeze(1)
        dist = torch.maximum((x - x.t()).abs(), (y - y.t()).abs())
        allowed = dist <= max(0, int(self.spatial_radius))
        # attn_mask expects True where attention is NOT allowed
        return (~allowed).to(device=device)

    @lru_cache(maxsize=None)
    def _temporal_mask(self, t, device: torch.device):
        win = max(1, int(self.window_size))
        idx = torch.arange(t, device=device)
        dist = idx.unsqueeze(1) - idx.unsqueeze(0)
        allowed = (dist >= 0) & (dist <= win)
        mask = ~allowed
        return mask

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal: bool = False, spatial_size: int | None = None):
        b, seq, d = x.shape
        if spatial_size is None or spatial_size <= 0:
            raise RuntimeError("spatial_size must be provided to SpatioTemporalLocalTransformerEncoderLayer.forward")
        if seq % int(spatial_size) != 0:
            raise RuntimeError(f"Sequence length {seq} not divisible by spatial size s={spatial_size}")
        s = int(spatial_size)
        t = seq // s
        x4 = x.reshape(b, t, s, d)
        x_s = x4.reshape(b * t, s, d)
        xs_norm = self.n1(x_s)
        attn_mask = self._spatial_mask(s, x.device)
        y_s, _ = self.s_att(query=xs_norm, key=xs_norm, value=xs_norm, attn_mask=attn_mask, need_weights=False)
        x4 = x4 + self.do1(y_s.reshape(b, t, s, d))
        x_t = x4.permute(0, 2, 1, 3).contiguous().reshape(b * s, t, d)
        attn_mask = self._temporal_mask(t, x.device)
        xt_norm = self.n2(x_t)
        y_t, _ = self.t_att(xt_norm, xt_norm, xt_norm, attn_mask=attn_mask, need_weights=False)
        x4 = x4 + self.do2(y_t.reshape(b, s, t, d).permute(0, 2, 1, 3).contiguous())
        y = self.ff(self.n3(x4))
        x4 = x4 + self.do3(y)
        return x4.reshape(b, seq, d)


class SpatialTemporalFullAttentionEncoderLayer(nn.Module):
    """Spatiotemporal encoder layer with full temporal attention and no latents."""

    def __init__(self, d_model, s_nhead, t_nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.s_nhead = s_nhead
        self.t_nhead = t_nhead
        self.dropout = dropout
        self.d = d_model
        self.d_ff = dim_feedforward
        self.s_att = nn.MultiheadAttention(self.d, s_nhead, dropout=dropout, batch_first=True)
        self.t_att = nn.MultiheadAttention(self.d, t_nhead, dropout=dropout, batch_first=True)
        self.n1 = nn.RMSNorm(d_model)
        self.n2 = nn.RMSNorm(d_model)
        self.n3 = nn.RMSNorm(d_model)
        self.do1 = nn.Dropout(dropout)
        self.do2 = nn.Dropout(dropout)
        self.do3 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
        )

    def forward(self, x, src_mask=None, src_key_padding_mask=None, is_causal: bool = False, spatial_size: int | None = None):
        b, seq, d = x.shape
        if spatial_size is None or spatial_size <= 0:
            raise RuntimeError("spatial_size must be provided to SpatialTemporalFullAttentionEncoderLayer.forward")
        if seq % int(spatial_size) != 0:
            raise RuntimeError(f"Sequence length {seq} not divisible by spatial size s={spatial_size}")
        s = int(spatial_size)
        t = seq // s
        x4 = x.reshape(b, t, s, d)
        x_s = x4.reshape(b * t, s, d)
        xs_norm = self.n1(x_s)
        y_s, _ = self.s_att(xs_norm, xs_norm, xs_norm)
        x4 = x4 + self.do1(y_s.reshape(b, t, s, d))
        x_t = x4.permute(0, 2, 1, 3).reshape(b * s, t, d)
        xt_norm = self.n2(x_t)
        y_t, _ = self.t_att(xt_norm, xt_norm, xt_norm)
        x4 = x4 + self.do2(y_t.reshape(b, s, t, d).permute(0, 2, 1, 3).reshape(b, t, s, d))
        y = self.ff(self.n3(x4))
        x4 = x4 + self.do3(y)
        return x4.reshape(b, seq, d)


class VanillaTransformer(BaseTransformer):
    """Standard full-attention transformer with sinusoidal positional encoding.

    Supports multiple (d, r) configs like the spatio-temporal variants.
    Uses a simple 1D sinusoidal PE over the flattened sequence.
    ~4.78M params with default settings (matched to spatio-temporal model).
    """

    def __init__(
        self,
        d: int | None = None,
        r: int | None = None,
        d_model: int = 272,
        n_layers: int = 5,
        n_head: int = 8,
        ffn: int = 1152,
        dropout: float = 0.1,
        pe: nn.Module | None = None,
        configs: list[tuple[int, int]] | None = None,
    ):
        base_d = d if d is not None else (configs[0][0] if configs else 5)
        base_r = r if r is not None else (configs[0][1] if configs else 5)
        super().__init__(base_d, base_r, d_model, n_layers, n_head, ffn, dropout, pe)
        # Compute max sequence length from configs to size the sinusoidal PE
        cfgs = configs if configs is not None else [(base_d, base_r)]
        max_len = max((rv + 1) * (dv * dv - 1) for dv, rv in cfgs)
        self.sinusoidal_pe = posenc.SinusoidalEncoding(max_len, d_model)
        self._cfg_set = set(cfgs)

    def _build_positional_encoding(self, d, r, d_model):
        # Defer to sinusoidal_pe in forward; return identity here
        return nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        x = x.long()
        emb = self.embedding(x)
        emb = self.emb_norm(emb)
        emb = self.sinusoidal_pe(emb)
        src = emb.transpose(0, 1)
        encoded = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        encoded = encoded.transpose(0, 1)
        out = self.decoder(encoded).squeeze(-1)
        return out


class SpatioTemporalFullAttentionTransformer(BaseTransformer):
    """Spatiotemporal transformer without sliding window or latents."""

    def __init__(
        self,
        d: int | None = None,
        r: int | None = None,
        d_model: int = 256,
        n_layers: int = 6,
        n_head: int = 8,
        ffn: int = 512,
        dropout: float = 0.1,
        pe: nn.Module | None = None,
        s_nhead: int | None = None,
        t_nhead: int | None = None,
        configs: list[tuple[int, int]] | None = None,
    ):
        self.s_nhead = s_nhead if s_nhead is not None else n_head
        self.t_nhead = t_nhead if t_nhead is not None else n_head
        base_d = d if d is not None else (configs[0][0] if configs else 5)
        base_r = r if r is not None else (configs[0][1] if configs else 5)
        super().__init__(base_d, base_r, d_model, n_layers, n_head, ffn, dropout, pe)
        self.encoder = nn.ModuleList(
            [
                SpatialTemporalFullAttentionEncoderLayer(
                    d_model,
                    self.s_nhead,
                    self.t_nhead,
                    dim_feedforward=ffn,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        cfgs = configs if configs is not None else [(base_d, base_r)]
        self.pe_bank = PositionalEncodingBank(d_model, cfgs)

    def _build_positional_encoding(self, d, r, d_model):
        return nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        x = x.long()
        emb = self.embedding(x)
        emb = self.emb_norm(emb)
        seq_len = emb.size(1)
        pe = self.pe_bank.pe_for_length(seq_len, emb.device, emb.dtype)
        emb = emb + pe
        spatial_size = self.pe_bank.spatial_for_length(seq_len)
        encoded = emb
        for layer in self.encoder:
            encoded = layer(encoded, src_key_padding_mask=src_key_padding_mask, spatial_size=spatial_size)
        out = self.decoder(encoded).squeeze(-1)
        return out


class SpatioTemporalLocalTransformer(BaseTransformer):
    """Spatiotemporal transformer with local spatial attention mask."""

    def __init__(
        self,
        d: int | None = None,
        r: int | None = None,
        d_model: int = 256,
        n_layers: int = 6,
        n_head: int = 8,
        ffn: int = 512,
        dropout: float = 0.1,
        pe: nn.Module | None = None,
        s_nhead: int | None = None,
        t_nhead: int | None = None,
        t_window_size: int = 3,
        configs: list[tuple[int, int]] | None = None,
        spatial_radius: int = 4,
    ):
        self.s_nhead = s_nhead if s_nhead is not None else n_head
        self.t_nhead = t_nhead if t_nhead is not None else n_head
        base_d = d if d is not None else (configs[0][0] if configs else 5)
        base_r = r if r is not None else (configs[0][1] if configs else 5)
        super().__init__(base_d, base_r, d_model, n_layers, n_head, ffn, dropout, pe)
        self.encoder = nn.ModuleList(
            [
                SpatioTemporalLocalTransformerEncoderLayer(
                    d_model,
                    self.s_nhead,
                    self.t_nhead,
                    dim_feedforward=ffn,
                    dropout=dropout,
                    window_size=int(t_window_size),
                    spatial_radius=int(spatial_radius),
                )
                for _ in range(n_layers)
            ]
        )
        cfgs = configs if configs is not None else [(base_d, base_r)]
        self.pe_bank = PositionalEncodingBank(d_model, cfgs)

    def _build_positional_encoding(self, d, r, d_model):
        return nn.Identity()

    def forward(self, x, src_key_padding_mask=None):
        x = x.long()
        emb = self.embedding(x)
        emb = self.emb_norm(emb)
        seq_len = emb.size(1)
        pe = self.pe_bank.pe_for_length(seq_len, emb.device, emb.dtype)
        emb = emb + pe
        spatial_size = self.pe_bank.spatial_for_length(seq_len)
        encoded = emb
        for layer in self.encoder:
            encoded = layer(encoded, src_key_padding_mask=src_key_padding_mask, spatial_size=spatial_size)
        out = self.decoder(encoded).squeeze(-1)
        return out
