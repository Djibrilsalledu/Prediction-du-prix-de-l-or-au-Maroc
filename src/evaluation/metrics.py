"""Forecast evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _align(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    common = y_true.index.intersection(y_pred.index)
    if len(common) == 0:
        raise ValueError("No overlapping indices between actual and predicted.")
    yt = y_true.loc[common].astype(float).values
    yp = y_pred.loc[common].astype(float).values
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return {k: float("nan") for k in ("rmse", "mae", "mape", "r2", "bias", "directional_accuracy")}

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    mape = float(np.mean(np.abs((yt - yp) / np.where(yt == 0, np.nan, yt))) * 100)
    r2 = float(r2_score(yt, yp))
    bias = float(np.mean(yp - yt))

    if len(yt) > 1:
        actual_dir = np.sign(np.diff(yt))
        pred_dir = np.sign(np.diff(yp))
        directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
    else:
        directional_accuracy = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "bias": bias,
        "directional_accuracy": directional_accuracy,
    }


def metrics_to_frame(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).T.sort_values("rmse")
