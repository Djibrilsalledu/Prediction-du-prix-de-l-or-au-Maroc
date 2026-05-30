"""Bootstrap confidence intervals for forecast metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.config import BOOTSTRAP_N_SAMPLES, RANDOM_SEED


def _align_arrays(y_true: pd.Series, y_pred: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    common = y_true.index.intersection(y_pred.index)
    yt = y_true.loc[common].astype(float).values
    yp = y_pred.loc[common].astype(float).values
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


def bootstrap_metric_ci(
    y_true: pd.Series,
    y_pred: pd.Series,
    metric: str = "rmse",
    n_samples: int = BOOTSTRAP_N_SAMPLES,
    seed: int = RANDOM_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    yt, yp = _align_arrays(y_true, y_pred)
    n = len(yt)
    if n < 4:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan")}

    rng = np.random.default_rng(seed)

    def _metric(a: np.ndarray, b: np.ndarray) -> float:
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(a, b)))
        if metric == "mae":
            return float(mean_absolute_error(a, b))
        if metric == "r2":
            return float(r2_score(a, b))
        raise ValueError(f"Unknown metric: {metric}")

    point = _metric(yt, yp)
    boots = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        boots.append(_metric(yt[idx], yp[idx]))

    low, high = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "low": float(low), "high": float(high)}


def bootstrap_all_models(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for name, pred in predictions.items():
        if name.startswith("Baseline"):
            continue
        for metric in ("rmse", "mae", "r2"):
            ci = bootstrap_metric_ci(y_true, pred, metric=metric)
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "point": ci["point"],
                    "ci_low": ci["low"],
                    "ci_high": ci["high"],
                }
            )
    return pd.DataFrame(rows)
