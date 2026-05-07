"""
Module 3b — Spatio-Temporal GNN
=================================

Dual-output model that predicts, for each of Bengaluru's 198 BBMP wards:
  * heat_stress_score : 0–100 Urban Heat Island intensity
  * flood_risk_score  : 0–100 flash-flood risk

Architecture
─────────────
  Node features at timestep t  (N, F_in)
       │
       ├─ GraphSAGE Encoder  ──────────────────────────────────────────
       │  3 × SAGEConv(F → H=128) + BatchNorm + ReLU + Dropout(0.1)
       │  Spatial message passing across ward adjacency edges
       │  → (N, H) spatial embedding per ward
       │
       ├─ Temporal Attention  ──────────────────────────────────────────
       │  Input: history window of T=8 timesteps → (N, T, H)
       │  Multi-head self-attention across the time axis (heads=4)
       │  LayerNorm + residual
       │  → (N, H) temporally-aware representation
       │
       └─ Dual Output Heads  ──────────────────────────────────────────
          Shared FC(H→64) → GELU
          ┌─ heat_head:  FC(64→1) + Sigmoid × 100  → heat_stress  (N,)
          └─ flood_head: FC(64→1) + Sigmoid × 100  → flood_risk   (N,)

Node features (F_in = 12 by default)
──────────────────────────────────────
  From Module 1 / Delta Lake snapshot:
    0  ndvi                  1  ndbi
    2  impervious_pct/100    3  lst_celsius (normalised)
    4  rainfall_mm_24h       5  pm25_ugm3 (log-norm)
    6  population_density    7  area_km2
    8  industrial_weight     9  centroid_lon (normalised)
   10  centroid_lat (norm)  11  ward_embedding[:1]  (CNN-ViT dim-reduced)

  The last feature slot is optionally filled with the CNN-ViT embedding
  (from Module 2) projected down to a single scalar; the full 128-dim
  embedding is concatenated at the end of the SAGEConv stack when
  use_cnn_vit_embedding=True.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default dims
NODE_FEAT_DIM   = 12      # tabular features per ward per timestep
HIDDEN_DIM      = 256     # SAGEConv hidden dim
T_WINDOW        = 8       # temporal history length
ATTN_HEADS      = 4
N_SAGE_LAYERS   = 3
EMBED_DIM_VIT   = 128     # CNN-ViT embedding dim (Module 2 output)


# ---------------------------------------------------------------------------
# SAGEConv (pure PyTorch — no torch-geometric dependency for portability)
# ---------------------------------------------------------------------------

class SAGEConv(nn.Module):
    """
    Graph SAGE convolution (Hamilton et al. 2017).

    h_v^{(k)} = W_1 · h_v^{(k-1)}  +  W_2 · MEAN_{u∈N(v)} h_u^{(k-1)}

    Falls back to a plain linear layer when edge_index is not provided
    (useful for testing without a graph).
    """

    def __init__(self, in_dim: int, out_dim: int, normalize: bool = True) -> None:
        super().__init__()
        self.lin_self   = nn.Linear(in_dim,  out_dim, bias=False)
        self.lin_neigh  = nn.Linear(in_dim,  out_dim, bias=False)
        self.bias       = nn.Parameter(torch.zeros(out_dim))
        self.normalize  = normalize

    def forward(
        self,
        x: torch.Tensor,                    # (N, in_dim)
        edge_index: Optional[torch.Tensor], # (2, E) int64 or None
        edge_weight: Optional[torch.Tensor] = None,  # (E,) float32 or None
    ) -> torch.Tensor:
        N = x.size(0)

        if edge_index is None or edge_index.numel() == 0:
            # No graph — identity aggregation
            return F.relu(self.lin_self(x) + self.bias)

        src, dst = edge_index[0], edge_index[1]   # E,  E

        # Weighted mean aggregation: for each node aggregate neighbour features
        if edge_weight is not None:
            # Scatter weighted sum
            weighted = x[src] * edge_weight.unsqueeze(-1)  # (E, in_dim)
        else:
            weighted = x[src]

        aggr = torch.zeros(N, x.size(-1), device=x.device, dtype=x.dtype)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)

        # Normalise by degree
        deg = torch.zeros(N, device=x.device, dtype=x.dtype)
        ones = torch.ones(src.size(0), device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, dst, ones)
        deg = deg.clamp(min=1).unsqueeze(-1)
        aggr = aggr / deg

        out = self.lin_self(x) + self.lin_neigh(aggr) + self.bias
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


# ---------------------------------------------------------------------------
# GraphSAGE Encoder
# ---------------------------------------------------------------------------

class GraphSAGEEncoder(nn.Module):
    """Stack of SAGEConv layers with BN, ReLU, and dropout."""

    def __init__(
        self,
        in_dim: int = NODE_FEAT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_SAGE_LAYERS,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1]) for i in range(n_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(dims[i + 1]) for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            residual = x
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
            # Residual connection for layers after the first (dims match)
            if i > 0:
                x = x + residual
        return x   # (N, hidden_dim)


# ---------------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Multi-head self-attention over the time dimension of a node's history.

    Input:  (N, T, H)  — N nodes, T timesteps, H hidden dim
    Output: (N, H)     — temporally compressed representation

    Uses a learnable [CLS] query token that attends to the T timestep tokens
    and summarises them into a single vector per node.
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        n_heads: int = ATTN_HEADS,
        t_window: int = T_WINDOW,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # Sinusoidal positional encoding for time steps
        self.register_buffer(
            "pos_embed", self._sinusoidal_pe(t_window + 1, hidden_dim)
        )

        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Fixed sinusoidal positional encoding (1, max_len, d_model)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, H) hidden states across time window.
        Returns:
            (N, H) — CLS-token summarising the full history.
        """
        N = x.size(0)
        cls = self.cls_token.expand(N, -1, -1)          # (N, 1, H)
        tokens = torch.cat([cls, x], dim=1)              # (N, T+1, H)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        # Self-attention (all tokens attend to each other)
        residual = tokens
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(residual + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        return tokens[:, 0]   # (N, H) — CLS token


# ---------------------------------------------------------------------------
# CNN-ViT embedding projection (optional)
# ---------------------------------------------------------------------------

class CNNViTProjection(nn.Module):
    """Projects the 128-dim CNN-ViT ward embedding to hidden_dim."""

    def __init__(self, vit_dim: int = EMBED_DIM_VIT, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(vit_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.proj(emb)))


# ---------------------------------------------------------------------------
# Dual-output prediction heads
# ---------------------------------------------------------------------------

class _DualHead(nn.Module):
    """Shared trunk + two sigmoid output heads."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(0.3)
        )
        self.heat_head  = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.flood_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.shared(x)
        heat  = self.heat_head(z).squeeze(-1)  * 100   # (N,) → 0..100
        flood = self.flood_head(z).squeeze(-1) * 100   # (N,) → 0..100
        return heat, flood


# ---------------------------------------------------------------------------
# Main model — SpatioTemporalGNN
# ---------------------------------------------------------------------------

class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal GNN for dual-output ward climate risk prediction.

    Args:
        node_feat_dim:         Number of tabular features per ward per timestep.
        hidden_dim:            SAGEConv / attention hidden dimension.
        t_window:              Number of historical timesteps in each sample.
        n_sage_layers:         Number of GraphSAGE convolution layers.
        attn_heads:            Number of temporal attention heads.
        use_cnn_vit_embedding: Whether to fuse CNN-ViT embeddings as extra node features.
        vit_embed_dim:         Dimension of incoming CNN-ViT embedding.
        dropout:               Dropout probability.
    """

    def __init__(
        self,
        node_feat_dim: int    = NODE_FEAT_DIM,
        hidden_dim: int       = HIDDEN_DIM,
        t_window: int         = T_WINDOW,
        n_sage_layers: int    = N_SAGE_LAYERS,
        attn_heads: int       = ATTN_HEADS,
        use_cnn_vit_embedding: bool = False,
        vit_embed_dim: int    = EMBED_DIM_VIT,
        dropout: float        = 0.3,
    ) -> None:
        super().__init__()
        self.t_window = t_window
        self.use_vit  = use_cnn_vit_embedding

        # Input projection (tabular → hidden)
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # Optional CNN-ViT embedding fusion
        if use_cnn_vit_embedding:
            self.vit_proj = CNNViTProjection(vit_embed_dim, hidden_dim)
            sage_in = hidden_dim * 2   # concat tabular + vit projections
        else:
            sage_in = hidden_dim

        # Spatial encoder (applied per-timestep)
        self.sage = GraphSAGEEncoder(sage_in, hidden_dim, n_sage_layers, dropout)

        # Temporal aggregator (across T timesteps)
        self.temporal = TemporalAttention(hidden_dim, attn_heads, t_window, dropout)

        # Prediction heads
        self.head = _DualHead(hidden_dim)

        self._node_feat_dim = node_feat_dim
        self._hidden_dim    = hidden_dim

    # ------------------------------------------------------------------
    def forward(
        self,
        x_seq: torch.Tensor,                    # (N, T, F)
        edge_index: Optional[torch.Tensor],     # (2, E)
        edge_weight: Optional[torch.Tensor] = None,  # (E,)
        vit_embeddings: Optional[torch.Tensor] = None,  # (N, vit_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_seq:          (N, T, F) node features across T timesteps.
            edge_index:     (2, E) ward adjacency in COO format.
            edge_weight:    (E,) optional normalised edge weights.
            vit_embeddings: (N, 128) CNN-ViT embeddings (optional).

        Returns:
            dict with:
              'heat_stress'  : (N,) float, 0–100
              'flood_risk'   : (N,) float, 0–100
              'node_embedding': (N, hidden_dim) — can be used by RL module
        """
        N, T, F = x_seq.shape
        sage_outputs: list[torch.Tensor] = []

        # Optional VIT projection (constant across time)
        vit_feat: Optional[torch.Tensor] = None
        if self.use_vit and vit_embeddings is not None:
            vit_feat = self.vit_proj(vit_embeddings)   # (N, H)

        for t in range(T):
            xt = self.input_proj(x_seq[:, t, :])   # (N, H)
            if vit_feat is not None:
                xt = torch.cat([xt, vit_feat], dim=-1)   # (N, 2H)
            xt = self.sage(xt, edge_index, edge_weight)  # (N, H)
            sage_outputs.append(xt)

        # Stack along time → (N, T, H)
        temporal_seq = torch.stack(sage_outputs, dim=1)
        node_emb = self.temporal(temporal_seq)   # (N, H)

        heat, flood = self.head(node_emb)

        return {
            "heat_stress":    heat,
            "flood_risk":     flood,
            "node_embedding": node_emb,
        }

    # ------------------------------------------------------------------
    def predict_single_snapshot(
        self,
        x: torch.Tensor,                       # (N, F) — one timestep
        edge_index: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor] = None,
        vit_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience: replicate a single timestep T times to fill the window.
        Useful for real-time inference when full history is unavailable.
        """
        x_seq = x.unsqueeze(1).expand(-1, self.t_window, -1)   # (N, T, F)
        return self.forward(x_seq, edge_index, edge_weight, vit_embeddings)

    # ------------------------------------------------------------------
    def parameter_count(self) -> Dict[str, int]:
        def _n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            "input_proj": _n(self.input_proj),
            "sage":       _n(self.sage),
            "temporal":   _n(self.temporal),
            "head":       _n(self.head),
            "total":      _n(self),
        }


# ---------------------------------------------------------------------------
# __main__ — architecture smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loguru import logger

    N, T, F = 198, T_WINDOW, NODE_FEAT_DIM
    logger.info(f"ST-GNN smoke test | N={N}  T={T}  F={F}")

    model = SpatioTemporalGNN()
    params = model.parameter_count()
    logger.info(
        f"Parameters | SAGE={params['sage']:,}  Temporal={params['temporal']:,}  "
        f"Total={params['total']:,}"
    )

    # Dummy inputs
    x_seq      = torch.randn(N, T, F)
    edge_index = torch.randint(0, N, (2, N * 6))   # ~6 edges per node

    out = model(x_seq, edge_index)
    assert out["heat_stress"].shape    == (N,), "heat_stress shape"
    assert out["flood_risk"].shape     == (N,), "flood_risk shape"
    assert out["node_embedding"].shape == (N, HIDDEN_DIM), "node_embedding shape"
    assert 0 <= out["heat_stress"].min() and out["heat_stress"].max() <= 100
    assert 0 <= out["flood_risk"].min()  and out["flood_risk"].max()  <= 100

    # Single-snapshot path
    out2 = model.predict_single_snapshot(x_seq[:, 0, :], edge_index)
    assert out2["heat_stress"].shape == (N,)

    # With CNN-ViT embeddings
    model_vit = SpatioTemporalGNN(use_cnn_vit_embedding=True)
    vit_emb   = torch.randn(N, EMBED_DIM_VIT)
    out3 = model_vit(x_seq, edge_index, vit_embeddings=vit_emb)
    assert out3["heat_stress"].shape == (N,)

    logger.success(
        f"heat_stress range=[{out['heat_stress'].min():.1f}, {out['heat_stress'].max():.1f}]  "
        f"flood_risk range=[{out['flood_risk'].min():.1f}, {out['flood_risk'].max():.1f}]"
    )
    logger.success("model.py smoke test passed.")
