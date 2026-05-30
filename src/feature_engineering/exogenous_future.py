"""
Future exogenous variable construction for multi-step monthly forecasts.

Methodology:
- Moroccan events: seasonal repetition from historical monthly calendar patterns
  (same month across years averaged, then blended with last observed year).
- USD/MAD and macro: linear trend extrapolation on last 24 months + bounds,
  with fallback to last observed value if trend unstable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import DATE_COLUMN, EVENT_COLUMNS, FORECAST_END, USD_MAD_COL


def _seasonal_profile(history: pd.Series) -> pd.Series:
    """Average intensity by calendar month across all years."""
    tmp = history.copy()
    tmp = tmp.to_frame("v")
    tmp["month"] = tmp.index.month
    profile = tmp.groupby("month")["v"].mean()
    return profile


def extend_events(history: pd.DataFrame, future_index: pd.DatetimeIndex) -> pd.DataFrame:
    future = pd.DataFrame(index=future_index)
    for col in EVENT_COLUMNS:
        if col not in history.columns:
            future[col] = 0.0
            continue
        profile = _seasonal_profile(history[col])
        last_year = history[col].iloc[-12:] if len(history) >= 12 else history[col]
        last_profile = last_year.copy()
        last_profile.index = last_profile.index.month
        blended = []
        for dt in future_index:
            m = dt.month
            seasonal = 0.7 * profile.get(m, 0.0) + 0.3 * last_profile.get(m, profile.get(m, 0.0))
            blended.append(float(np.clip(seasonal, 0.0, 1.0)))
        future[col] = blended
    future.index.name = DATE_COLUMN
    return future


def _extrapolate_series_damped(history: pd.Series, steps: int, phi: float = 0.95) -> pd.Series:
    """Linear extrapolation with horizon-dependent trend decay toward last value."""
    y = history.dropna().astype(float)
    if len(y) < 6:
        last = y.iloc[-1] if len(y) else 0.0
        return pd.Series([last] * steps)

    window = min(24, len(y))
    x = np.arange(window)
    coef = np.polyfit(x, y.values[-window:], deg=1)
    last = float(y.iloc[-1])
    preds = []
    for h in range(1, steps + 1):
        linear = np.polyval(coef, len(y) - window + window - 1 + h)
        decay = phi ** h
        preds.append(decay * linear + (1 - decay) * last)
    cap_up = last * 1.12
    cap_dn = last * 0.88
    return pd.Series(np.clip(preds, cap_dn, cap_up))


def extend_numeric_exog(
    history: pd.DataFrame, columns: list[str], future_index: pd.DatetimeIndex
) -> pd.DataFrame:
    steps = len(future_index)
    out = pd.DataFrame(index=future_index)
    for col in columns:
        if col not in history.columns:
            continue
        preds = _extrapolate_series_damped(history[col], steps)
        preds.index = future_index
        out[col] = preds.values
    out.index.name = DATE_COLUMN
    return out


def build_future_exogenous(
    history: pd.DataFrame,
    last_observed: pd.Timestamp,
    forecast_end: str = FORECAST_END,
    exog_cols: list[str] | None = None,
    allow_extrapolation: bool | None = None,
) -> pd.DataFrame:
    """
    Build monthly exogenous matrix from month after last_observed through forecast_end.

    Events: seasonal profile (calendar-known; justified for Morocco monthly intensities).
    FX/macro: linear trend extrapolation ONLY if allow_extrapolation=True.

    Set allow_extrapolation=False (STRICT_EVALUATION_MODE) to hold FX/macro at last
    observed train value — avoids unverified future macro assumptions.
    """
    from src.utils.config import ALLOW_EXOG_EXTRAPOLATION

    if allow_extrapolation is None:
        allow_extrapolation = ALLOW_EXOG_EXTRAPOLATION
    start = (last_observed + pd.offsets.MonthBegin(1)).normalize()
    end = pd.Timestamp(forecast_end)
    future_index = pd.date_range(start, end, freq="MS")
    if len(future_index) == 0:
        return pd.DataFrame()

    event_future = extend_events(history, future_index)
    numeric_cols = [c for c in (exog_cols or []) if c not in EVENT_COLUMNS and c in history.columns]
    if numeric_cols:
        if allow_extrapolation:
            num_future = extend_numeric_exog(history, numeric_cols, future_index)
        else:
            num_future = pd.DataFrame(index=future_index)
            for col in numeric_cols:
                last_val = float(history[col].dropna().iloc[-1]) if history[col].notna().any() else 0.0
                num_future[col] = last_val
            num_future.index.name = DATE_COLUMN
        future = event_future.join(num_future, how="outer")
    else:
        future = event_future

    if USD_MAD_COL in history.columns and USD_MAD_COL not in future.columns:
        if allow_extrapolation:
            fx = extend_numeric_exog(history, [USD_MAD_COL], future_index)
        else:
            fx = pd.DataFrame(
                {USD_MAD_COL: float(history[USD_MAD_COL].dropna().iloc[-1])},
                index=future_index,
            )
            fx.index.name = DATE_COLUMN
        future = future.join(fx)

    return future.sort_index()
