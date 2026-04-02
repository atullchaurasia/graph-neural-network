"""
models/stgnn.py
Spatio-Temporal Graph Neural Network (STGNN).

Architecture:
  Input [B, T_in, N, F]
    → Spatial GAT block  (per time step, shared weights)
    → Temporal LSTM+Attn block
    → Multi-step prediction head
  Output [B, T_out, N]
"""

import torch
import torch.nn as nn
from .gat_conv import TrafficGATConv
from .temporal import TemporalEncoder


class SpatialBlock(nn.Module):
    """
    Apply TrafficGATConv to every time step independently.
    Input  : [B, T, N, in_dim]
    Output : [B, T, N, out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat  = TrafficGATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.norm = nn.LayerNorm(out_dim)
        self.act  = nn.ELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        B, T, N, D = x.shape

        # Flatten batch × time for graph ops
        x_flat = x.reshape(B * T, N, D)                    # [B*T, N, D]

        # Expand edge_index/weight for the batch dimension
        # PyG processes one graph at a time — we create a disconnected union graph
        offsets     = torch.arange(B * T, device=x.device).unsqueeze(1) * N  # [B*T, 1]
        src_batch   = (edge_index[0].unsqueeze(0) + offsets).reshape(-1)
        dst_batch   = (edge_index[1].unsqueeze(0) + offsets).reshape(-1)
        ei_batch    = torch.stack([src_batch, dst_batch])                      # [2, B*T*E]

        if edge_weight is not None:
            ew_batch = edge_weight.unsqueeze(0).expand(B * T, -1).reshape(-1)
        else:
            ew_batch = None

        x_node = x_flat.reshape(B * T * N, D)              # [B*T*N, D]
        out    = self.gat(x_node, ei_batch, ew_batch)       # [B*T*N, out_dim]
        out    = out.reshape(B, T, N, -1)
        out    = self.act(self.norm(out))
        return out


class STGNN(nn.Module):
    """
    Full Spatio-Temporal GNN.

    Parameters
    ----------
    num_nodes   : int   — number of sensor nodes (207 for METR-LA)
    in_features : int   — input features per node per step (1 = speed only)
    hidden_dim  : int   — internal channel width
    t_out       : int   — prediction horizon (steps)
    gat_heads   : int   — attention heads in GAT layers
    lstm_layers : int   — LSTM depth
    dropout     : float — dropout rate
    """

    def __init__(
        self,
        num_nodes:   int   = 207,
        in_features: int   = 1,
        hidden_dim:  int   = 64,
        t_out:       int   = 12,
        gat_heads:   int   = 4,
        lstm_layers: int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.t_out     = t_out

        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)

        # Stacked spatial blocks
        self.spatial1 = SpatialBlock(hidden_dim, hidden_dim, gat_heads, dropout)
        self.spatial2 = SpatialBlock(hidden_dim, hidden_dim, gat_heads, dropout)

        # Temporal encoder
        self.temporal = TemporalEncoder(
            spatial_dim=hidden_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            attn_heads=4,
            dropout=dropout,
        )

        # Prediction head: [B, N, hidden] → [B, N, t_out]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, t_out),
        )

    def forward(
        self,
        x:           torch.Tensor,       # [B, T_in, N, F]
        edge_index:  torch.Tensor,       # [2, E]
        edge_weight: torch.Tensor = None,# [E]
    ) -> torch.Tensor:                   # [B, T_out, N]

        # Project input features
        h = self.input_proj(x)           # [B, T, N, hidden]

        # Spatial message passing (2 rounds)
        h = self.spatial1(h, edge_index, edge_weight)
        h = self.spatial2(h, edge_index, edge_weight)

        # Temporal encoding → node embeddings
        z = self.temporal(h)             # [B, N, hidden]

        # Multi-step prediction
        out = self.head(z)               # [B, N, T_out]
        out = out.permute(0, 2, 1)       # [B, T_out, N]

        return out


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, N, F = 4, 12, 207, 1
    model = STGNN(num_nodes=N, in_features=F, hidden_dim=64, t_out=12)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    x          = torch.randn(B, T, N, F)
    src        = torch.randint(0, N, (N * 3,))
    dst        = torch.randint(0, N, (N * 3,))
    edge_index = torch.stack([src, dst])
    edge_weight = torch.rand(N * 3)

    out = model(x, edge_index, edge_weight)
    print(f"Output shape: {out.shape}")  # [4, 12, 207]
