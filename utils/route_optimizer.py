"""
utils/route_optimizer.py
A* shortest path on the road graph using GNN-predicted travel times as edge weights.
"""

import heapq
import math
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in km (heuristic for A*)."""
    R    = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


import networkx as nx

def build_networkx_graph(
    edge_index:  torch.Tensor,   # [2, E]
    edge_weight: torch.Tensor,   # [E]  predicted travel times (seconds)
) -> nx.DiGraph:
    """Build NetworkX DiGraph for ultra-fast bidirectional search."""
    G = nx.DiGraph()
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    weights  = edge_weight.tolist()
    # Bulk loading edges is exceptionally fast in NetworkX
    G.add_weighted_edges_from(zip(src, dst, weights))
    return G


class RouteOptimizer:
    """
    High-level interface: given predicted node speeds, compute
    edge travel times and find the optimal route.
    """

    def __init__(
        self,
        edge_index:   torch.Tensor,        # [2, E]
        edge_lengths: np.ndarray,          # [E] road segment lengths in metres
        sensor_coords: Optional[np.ndarray] = None,  # [N, 2] lat/lon
    ):
        self.edge_index    = edge_index
        self.edge_lengths  = edge_lengths
        self.sensor_coords = sensor_coords
        self.num_nodes     = edge_index.max().item() + 1

    def speeds_to_travel_times(self, predicted_speeds: torch.Tensor) -> torch.Tensor:
        """
        predicted_speeds : [N]  in km/h
        Returns edge travel times [E] in seconds.
        Uses average of src + dst node speed for each edge.
        """
        src, dst = self.edge_index[0], self.edge_index[1]
        avg_speed_ms = (predicted_speeds[src] + predicted_speeds[dst]) / 2.0  # km/h
        avg_speed_ms = avg_speed_ms * (1000.0 / 3600.0)                        # → m/s
        avg_speed_ms = avg_speed_ms.clamp(min=1.0)                             # avoid /0

        lengths_t    = torch.tensor(self.edge_lengths, dtype=torch.float)
        return lengths_t / avg_speed_ms                                        # seconds

    def find_route(
        self,
        start:            int,
        goal:             int,
        predicted_speeds: torch.Tensor,    # [N]
    ) -> Tuple[List[int], float, List[Tuple[float, float]]]:
        """
        Returns:
          path_nodes  : list of node indices
          travel_time : total seconds
          path_coords : list of (lat, lon) if sensor_coords available
        """
        travel_times = self.speeds_to_travel_times(predicted_speeds)
        
        # Super-fast bidirectional optimization using NetworkX
        nx_graph = build_networkx_graph(self.edge_index, travel_times)
        try:
            cost, path = nx.bidirectional_dijkstra(nx_graph, start, goal)
        except nx.NetworkXNoPath:
            path, cost = [], float("inf")

        path_coords = []
        if self.sensor_coords is not None and path:
            path_coords = [(self.sensor_coords[n, 0], self.sensor_coords[n, 1]) for n in path]

        return path, cost, path_coords
