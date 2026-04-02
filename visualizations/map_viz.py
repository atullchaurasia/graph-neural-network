"""
visualizations/map_viz.py
Folium-based map visualization:
  - Congestion heatmap (predicted speed as colour)
  - Optimal route overlay
  - Sensor node markers
"""

import os
import numpy as np
import torch
import folium
from folium.plugins import HeatMap


# ── Colour scale (speed → congestion colour) ─────────────────────────────────

def speed_to_color(speed: float, min_spd: float = 0.0, max_spd: float = 70.0) -> str:
    """Maps speed (km/h) to hex colour: red (slow) → yellow → green (fast)."""
    ratio = max(0.0, min(1.0, (speed - min_spd) / (max_spd - min_spd + 1e-8)))
    if ratio < 0.5:
        r = 255
        g = int(255 * ratio * 2)
    else:
        r = int(255 * (1 - ratio) * 2)
        g = 255
    return f"#{r:02x}{g:02x}00"


# ── Main visualizer class ─────────────────────────────────────────────────────

class TrafficMapVisualizer:
    """
    Parameters
    ----------
    sensor_coords : np.ndarray [N, 2]  — (lat, lon) per node
    edge_index    : torch.Tensor [2, E]
    center        : (lat, lon) map centre; auto-computed if None
    zoom          : initial zoom level
    """

    def __init__(
        self,
        sensor_coords: np.ndarray,
        edge_index:    torch.Tensor,
        center:        tuple = None,
        zoom:          int   = 12,
    ):
        self.coords     = sensor_coords
        self.edge_index = edge_index
        self.center     = center or (sensor_coords[:, 0].mean(), sensor_coords[:, 1].mean())
        self.zoom       = zoom

    def _base_map(self) -> folium.Map:
        return folium.Map(
            location    = self.center,
            zoom_start  = self.zoom,
            tiles       = "OpenStreetMap",   # Standard detailed street map
        )

    # ── Congestion heatmap ────────────────────────────────────────────────────

    def congestion_heatmap(
        self,
        predicted_speeds: np.ndarray,    # [N]  km/h per node
        output_path: str = "visualizations/congestion_map.html",
    ) -> folium.Map:
        """
        Renders a heatmap where intensity = inverse of speed (high = congested).
        """
        m = self._base_map()

        # HeatMap data: [lat, lon, intensity]
        max_spd  = predicted_speeds.max() + 1e-8
        heat_data = [
            [self.coords[i, 0], self.coords[i, 1], 1.0 - predicted_speeds[i] / max_spd]
            for i in range(len(self.coords))
        ]

        HeatMap(
            heat_data,
            radius          = 20,
            blur            = 15,
            gradient        = {0.0: "blue", 0.3: "cyan", 0.6: "yellow", 1.0: "red"},
            name            = "Congestion",
        ).add_to(m)

        # Edge overlays coloured by speed
        src, dst = self.edge_index[0].tolist(), self.edge_index[1].tolist()
        for s, d in zip(src, dst):
            avg_spd = (predicted_speeds[s] + predicted_speeds[d]) / 2.0
            color   = speed_to_color(avg_spd)
            folium.PolyLine(
                locations = [
                    [self.coords[s, 0], self.coords[s, 1]],
                    [self.coords[d, 0], self.coords[d, 1]],
                ],
                color   = color,
                weight  = 3,
                opacity = 0.7,
                tooltip = f"Avg speed: {avg_spd:.1f} km/h",
            ).add_to(m)

        # Sensor markers
        for i, (lat, lon) in enumerate(self.coords):
            folium.CircleMarker(
                location  = [lat, lon],
                radius    = 4,
                color     = speed_to_color(predicted_speeds[i]),
                fill      = True,
                fill_opacity = 0.9,
                tooltip   = f"Sensor {i} | {predicted_speeds[i]:.1f} km/h",
            ).add_to(m)

        folium.LayerControl().add_to(m)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        m.save(output_path)
        print(f"Saved congestion map -> {output_path}")
        return m

    # ── Route overlay ─────────────────────────────────────────────────────────

    def route_overlay(
        self,
        predicted_speeds: np.ndarray,
        path_nodes:       list,
        path_coords:      list,           # [(lat, lon), ...]
        travel_time_s:    float,
        output_path: str = "visualizations/route_map.html",
        show_heatmap: bool = False,
    ) -> folium.Map:
        """
        Renders highlighted optimal route, optionally with congestion background.
        """
        if show_heatmap:
            m = self.congestion_heatmap(predicted_speeds, output_path="/dev/null")
        else:
            m = self._base_map()

        if path_coords:
            # Route polyline
            folium.PolyLine(
                locations   = path_coords,
                color       = "#00CFFF",
                weight      = 6,
                opacity     = 0.95,
                tooltip     = f"Optimal route | ETA: {travel_time_s/60:.1f} min",
            ).add_to(m)

            # Start / end markers
            folium.Marker(
                location  = path_coords[0],
                tooltip   = "Start",
                icon      = folium.Icon(color="green", icon="play"),
            ).add_to(m)
            folium.Marker(
                location  = path_coords[-1],
                tooltip   = f"Destination | ETA {travel_time_s/60:.1f} min",
                icon      = folium.Icon(color="red", icon="flag"),
            ).add_to(m)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        m.save(output_path)
        print(f"Saved route map -> {output_path}")
        return m

    # ── Time-series snapshot ──────────────────────────────────────────────────

    def multi_step_snapshot(
        self,
        speeds_sequence: np.ndarray,   # [T_out, N]
        step:            int   = 0,    # which prediction step to show
        output_path: str = "visualizations/snapshot_map.html",
    ) -> folium.Map:
        """Render a snapshot map for a specific prediction horizon."""
        return self.congestion_heatmap(speeds_sequence[step], output_path=output_path)


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Synthetic demo with LA-area bounding box
    N = 207
    rng = np.random.default_rng(42)
    lats   = rng.uniform(33.70, 34.35, N)
    lons   = rng.uniform(-118.70, -118.10, N)
    coords = np.stack([lats, lons], axis=1)

    src    = rng.integers(0, N, N * 3)
    dst    = rng.integers(0, N, N * 3)
    ei     = torch.tensor(np.stack([src, dst]))
    speeds = rng.uniform(5, 65, N).astype(np.float32)

    viz = TrafficMapVisualizer(coords, ei, zoom=11)
    viz.congestion_heatmap(speeds, "visualizations/demo_congestion.html")

    # Fake route
    path   = list(range(10))
    p_coords = [(coords[i, 0], coords[i, 1]) for i in path]
    viz.route_overlay(speeds, path, p_coords, travel_time_s=720,
                      output_path="visualizations/demo_route.html")
    print("Demo maps saved in visualizations/")
