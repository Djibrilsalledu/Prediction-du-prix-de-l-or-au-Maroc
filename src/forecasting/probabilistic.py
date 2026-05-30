"""Prediction intervals via model-native or residual bootstrap."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastInterval:
    point: pd.Series
    lower: pd.Series
    upper: pd.Series


def residual_bootstrap_intervals(
    point_forecast: pd.Series,
    in_sample_residuals: pd.Series,
    *,
    n_samples: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> ForecastInterval:
    """
    Fan chart: add cumulative residual noise; uncertainty grows with horizon.
    """
    rng = np.random.default_rng(seed)
    resid = in_sample_residuals.dropna().values
    if len(resid) < 5:
        z = 1.96
        std = float(np.std(resid)) if len(resid) else 0.0
        lower = point_forecast - z * std * np.sqrt(np.arange(1, len(point_forecast) + 1))
        upper = point_forecast + z * std * np.sqrt(np.arange(1, len(point_forecast) + 1))
        return ForecastInterval(point_forecast, pd.Series(lower, index=point_forecast.index),
                                pd.Series(upper, index=point_forecast.index))

    steps = len(point_forecast)
    sims = np.zeros((n_samples, steps))
    for i in range(n_samples):
        noise = rng.choice(resid, size=steps, replace=True)
        cum_noise = np.cumsum(noise) * np.sqrt(np.arange(1, steps + 1) / steps)
        sims[i] = point_forecast.values + cum_noise

    lo = np.percentile(sims, 100 * alpha / 2, axis=0)
    hi = np.percentile(sims, 100 * (1 - alpha / 2), axis=0)
    return ForecastInterval(
        point_forecast,
        pd.Series(lo, index=point_forecast.index),
        pd.Series(hi, index=point_forecast.index),
    )


def arima_confidence_intervals(model, steps: int, index: pd.DatetimeIndex, alpha: float = 0.05) -> ForecastInterval | None:
    """Use pmdarima native confidence intervals if available."""
    try:
        fc, conf = model.predict(n_periods=steps, return_conf_int=True, alpha=alpha)
        return ForecastInterval(
            pd.Series(fc, index=index[:steps]),
            pd.Series(conf[:, 0], index=index[:steps]),
            pd.Series(conf[:, 1], index=index[:steps]),
        )
    except Exception:
        return None
