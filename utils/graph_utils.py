"""
utils/graph_utils.py
Helpers for graph normalization and edge construction.
"""

import torch
import numpy as np
import scipy.sparse as sp


def normalize_adjacency(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
    """
    Symmetric normalized adjacency: D^{-1/2} A D^{-1/2}
    Returns normalized edge_weight (same edge_index).
    """
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, row, edge_weight)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

    norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return norm_weight


def add_self_loops(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
    """Add self-loops with weight 1.0 to edge_index."""
    self_loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    self_weight = torch.ones(num_nodes, dtype=torch.float)

    edge_index  = torch.cat([edge_index, self_loops], dim=1)
    edge_weight = torch.cat([edge_weight, self_weight])
    return edge_index, edge_weight


def laplacian(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
    """Normalized graph Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    norm_w = normalize_adjacency(edge_index, edge_weight, num_nodes)
    # L = I - A_norm  → use in spectral methods
    row, col = edge_index
    diag_mask = row == col
    norm_w[diag_mask] = 1.0 - norm_w[diag_mask]
    return norm_w


def coords_to_edge_distances(sensor_locs_csv: str):
    """
    Read sensor lat/lon CSV and return pairwise haversine distances (km).
    Columns expected: sensor_id, latitude, longitude
    """
    import pandas as pd

    df = pd.read_csv(sensor_locs_csv)
    lats = np.radians(df["latitude"].values)
    lons = np.radians(df["longitude"].values)

    n = len(df)
    dist = np.zeros((n, n))
    R = 6371.0  # earth radius km

    for i in range(n):
        dlat = lats - lats[i]
        dlon = lons - lons[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lats[i]) * np.cos(lats) * np.sin(dlon / 2) ** 2
        dist[i] = 2 * R * np.arcsin(np.sqrt(a))

    return df["sensor_id"].values, dist
