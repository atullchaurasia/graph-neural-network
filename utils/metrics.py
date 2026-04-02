"""
utils/metrics.py
Standard traffic prediction metrics.
"""

import numpy as np
import torch


def mae(pred: torch.Tensor, target: torch.Tensor, mask_zero: bool = True) -> float:
    """Mean Absolute Error, optionally masking near-zero ground truth."""
    if mask_zero:
        mask = target > 1.0
        pred, target = pred[mask], target[mask]
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred: torch.Tensor, target: torch.Tensor, mask_zero: bool = True) -> float:
    """Root Mean Square Error."""
    if mask_zero:
        mask = target > 1.0
        pred, target = pred[mask], target[mask]
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def mape(pred: torch.Tensor, target: torch.Tensor, mask_zero: bool = True) -> float:
    """Mean Absolute Percentage Error (%)."""
    if mask_zero:
        mask = target > 1.0
        pred, target = pred[mask], target[mask]
    return (torch.mean(torch.abs((pred - target) / target)) * 100).item()


def all_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Return dict of all metrics."""
    return {
        "MAE":  mae(pred, target),
        "RMSE": rmse(pred, target),
        "MAPE": mape(pred, target),
    }


def horizon_metrics(pred: torch.Tensor, target: torch.Tensor, horizons=(3, 6, 12)):
    """
    Metrics at specific prediction horizons.
    pred/target: [B, T_out, N]
    Returns dict keyed by step index (0-based).
    """
    results = {}
    for h in horizons:
        if h <= pred.shape[1]:
            results[f"h{h}"] = all_metrics(pred[:, h - 1, :], target[:, h - 1, :])
    return results
