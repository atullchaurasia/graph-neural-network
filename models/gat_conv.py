"""
models/gat_conv.py
Custom Graph Attention Convolution for traffic networks.
Supports multi-head attention and edge weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class TrafficGATConv(MessagePassing):
    """
    Graph Attention Convolution tailored for traffic:
    - Multi-head attention
    - Edge weight incorporation
    - Residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        negative_slope: float = 0.2,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.heads        = heads
        self.concat       = concat
        self.dropout      = dropout

        # Linear projections
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention vectors [1, heads, 2*out_channels]
        self.att_src = nn.Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_channels))

        self.leaky = nn.LeakyReLU(negative_slope)

        # Residual projection (if dimensions differ)
        out_dim = heads * out_channels if concat else out_channels
        self.residual = nn.Linear(in_channels, out_dim, bias=False) if in_channels != out_dim else nn.Identity()

        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """
        x           : [N, in_channels]
        edge_index  : [2, E]
        edge_weight : [E]   optional
        Returns     : [N, heads*out_channels] or [N, out_channels]
        """
        H, C = self.heads, self.out_channels

        # Project all nodes → [N, H, C]
        x_proj = self.lin(x).view(-1, H, C)

        # Attention scores
        alpha_src = (x_proj * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1)

        out = self.propagate(
            edge_index,
            x=x_proj,
            alpha=(alpha_src, alpha_dst),
            edge_weight=edge_weight,
            size=None,
        )                                                  # [N, H, C]

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        out = out + self.bias + self.residual(x)
        return out

    def message(self, x_j, alpha_j, alpha_i, edge_weight, index, ptr, size_i):
        """Compute attention-weighted messages."""
        alpha = self.leaky(alpha_i + alpha_j)             # [E, H]
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_weight is not None:
            alpha = alpha * edge_weight.unsqueeze(-1)

        return x_j * alpha.unsqueeze(-1)                  # [E, H, C]
