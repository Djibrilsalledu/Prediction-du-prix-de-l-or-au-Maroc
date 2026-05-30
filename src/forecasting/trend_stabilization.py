"""Trend damping, mean-reversion blend, and horizon stabilization."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import TREND_DAMPING_PHI, TREND_MAX_MONTHLY_PCT


def damp_forecast(
    forecast: pd.Series,
    history: pd.Series,
    *,
    phi: float = TREND_DAMPING_PHI,
    max_monthly_pct: float = TREND_MAX_MONTHLY_PCT,
) -> pd.Series:
    """
    Stabilize long-horizon forecasts:
    - Damp trend component exponentially with horizon (phi^h)
    - Blend toward recent 12-month mean growth path
    - Cap month-over-month changes to economically plausible bounds
    """
    y = history.astype(float).dropna()
    if len(y) < 13:
        return forecast

    recent = y.iloc[-12:]
    monthly_growth = (recent.iloc[-1] / recent.iloc[0]) ** (1 / 12) - 1
    monthly_growth = float(np.clip(monthly_growth, -max_monthly_pct, max_monthly_pct))
    long_run = float(recent.mean())

    stabilized = []
    prev = float(y.iloc[-1])
    raw = forecast.astype(float).values

    for h, f in enumerate(raw):
        decay = phi ** (h + 1)
        trend_path = prev * (1 + monthly_growth)
        mean_revert = long_run + (f - long_run) * decay
        blended = decay * f + (1 - decay) * 0.5 * (trend_path + mean_revert)

        cap_up = prev * (1 + max_monthly_pct)
        cap_dn = prev * (1 - max_monthly_pct)
        blended = float(np.clip(blended, cap_dn, cap_up))
        stabilized.append(blended)
        prev = blended

    return pd.Series(stabilized, index=forecast.index)


def apply_stabilization_if_enabled(
    forecast: pd.Series,
    history: pd.Series,
    model_name: str,
    stabilize_models: frozenset[str],
) -> pd.Series:
    if model_name not in stabilize_models:
        return forecast
    return damp_forecast(forecast, history)
