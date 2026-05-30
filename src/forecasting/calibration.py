"""Bias correction and ensemble weighting from OOS performance."""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_oos_bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean prediction error (pred - actual); positive = overestimate."""
    common = y_true.index.intersection(y_pred.index)
    if len(common) == 0:
        return 0.0
    return float((y_pred.loc[common] - y_true.loc[common]).mean())


def apply_bias_correction(forecast: pd.Series, bias: float) -> pd.Series:
    """Subtract systematic OOS bias."""
    return forecast - bias


def inverse_rmse_weights(metrics_df: pd.DataFrame, models: list[str]) -> dict[str, float]:
    """Weights proportional to 1/RMSE from backtest table."""
    weights = {}
    for m in models:
        if m not in metrics_df.index:
            continue
        rmse = metrics_df.loc[m, "rmse_mean"] if "rmse_mean" in metrics_df.columns else metrics_df.loc[m, "rmse"]
        weights[m] = 1.0 / max(rmse, 1e-6)
    total = sum(weights.values())
    if total == 0:
        n = len(weights)
        return {k: 1 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def weighted_ensemble(predictions: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    """Combine model forecasts with normalized weights."""
    common_idx = None
    for name, s in predictions.items():
        if name not in weights:
            continue
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)
    if common_idx is None or len(common_idx) == 0:
        raise ValueError("No overlapping predictions for ensemble.")

    out = pd.Series(0.0, index=common_idx)
    w_sum = 0.0
    for name, w in weights.items():
        if name not in predictions:
            continue
        out += w * predictions[name].reindex(common_idx)
        w_sum += w
    return out / w_sum if w_sum else out
