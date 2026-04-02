"""
models/temporal.py
Temporal module: encodes time-series node features using LSTM + attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Temporal self-attention across T time steps.
    Applied per node independently.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B*N, T, hidden_dim]
        Returns : [B*N, T, hidden_dim]
        """
        residual = x
        out, _ = self.attn(x, x, x)
        return self.norm(out + residual)


class TemporalEncoder(nn.Module):
    """
    Bi-directional LSTM followed by temporal attention.
    Input  : [B, T, N, spatial_dim]
    Output : [B, N, hidden_dim]   (last step representation)
    """

    def __init__(
        self,
        spatial_dim: int,
        hidden_dim: int  = 64,
        lstm_layers: int = 2,
        attn_heads: int  = 4,
        dropout: float   = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=spatial_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        # BiLSTM doubles hidden → project back
        self.proj  = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn  = TemporalAttention(hidden_dim, attn_heads)
        self.norm  = nn.LayerNorm(hidden_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, N, spatial_dim]
        """
        B, T, N, D = x.shape

        # Reshape: process each node's time series independently
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)   # [B*N, T, D]

        lstm_out, _ = self.lstm(x_flat)                         # [B*N, T, 2*hidden]
        lstm_out    = self.proj(lstm_out)                        # [B*N, T, hidden]
        lstm_out    = self.drop(lstm_out)

        attn_out = self.attn(lstm_out)                           # [B*N, T, hidden]

        # Take last time step as node representation
        out = attn_out[:, -1, :]                                 # [B*N, hidden]
        out = self.norm(out)

        return out.view(B, N, -1)                                # [B, N, hidden]
