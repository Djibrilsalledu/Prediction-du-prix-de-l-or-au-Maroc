"""Regime shift detection: changepoints, volatility, rolling trend strength."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.utils.config import FIGURES_DIR, TARGET_COL


def detect_changepoints_rolling(y: pd.Series, window: int = 24, threshold: float = 2.0) -> pd.DatetimeIndex:
    """
    Detect structural breaks via rolling mean shift z-scores.
    Returns dates where |z| > threshold (no future data used at each point).
    """
    y = y.astype(float)
    roll_mean = y.rolling(window, min_periods=window // 2).mean()
    roll_std = y.rolling(window, min_periods=window // 2).std()
    z = (y - roll_mean) / roll_std.replace(0, np.nan)
    breaks = y.index[abs(z) > threshold]
    return pd.DatetimeIndex(breaks)


def rolling_volatility_regime(y: pd.Series, window: int = 12) -> pd.Series:
    """High/low volatility regime (0=low, 1=high) from rolling return std."""
    ret = y.pct_change(fill_method=None)
    vol = ret.rolling(window, min_periods=6).std()
    median_vol = vol.expanding(min_periods=window).median()
    return (vol > median_vol).astype(float).rename("volatility_regime")


def rolling_trend_strength(y: pd.Series, window: int = 12) -> pd.Series:
    """Absolute 12-month return over rolling window — trend intensity indicator."""
    past = y.shift(1)
    strength = (past - past.shift(window)) / past.shift(window).replace(0, np.nan)
    return strength.abs().rename("trend_strength")


def adf_by_regime(y: pd.Series, changepoints: pd.DatetimeIndex) -> pd.DataFrame:
    """ADF test on segments between changepoints."""
    dates = sorted(set([y.index.min()] + list(changepoints) + [y.index.max()]))
    rows = []
    for i in range(len(dates) - 1):
        seg = y.loc[dates[i] : dates[i + 1]].dropna()
        if len(seg) < 12:
            continue
        stat, pval, *_ = adfuller(seg, autolag="AIC")
        rows.append({"start": dates[i], "end": dates[i + 1], "adf_stat": stat, "adf_pvalue": pval, "n": len(seg)})
    return pd.DataFrame(rows)


def run_regime_analysis(y: pd.Series, output_dir: Path = FIGURES_DIR) -> dict:
    """Save regime visualizations and return summary stats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y = y.dropna()

    cps = detect_changepoints_rolling(y)
    vol_reg = rolling_volatility_regime(y)
    trend = rolling_trend_strength(y)
    adf_seg = adf_by_regime(y, cps)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    y.plot(ax=axes[0], color="black", label=TARGET_COL)
    for cp in cps:
        axes[0].axvline(cp, color="red", alpha=0.4, linestyle="--")
    axes[0].set_title("Price with detected changepoints")
    axes[0].legend()

    vol_reg.plot(ax=axes[1], color="orange", label="Volatility regime")
    axes[1].set_title("Volatility regime (1=high)")
    axes[1].legend()

    trend.plot(ax=axes[2], color="steelblue", label="Trend strength")
    axes[2].set_title("Rolling trend strength")
    axes[2].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "regime_detection.png", dpi=150)
    plt.close()

    adf_seg.to_csv(output_dir.parent / "tables" / "regime_adf_segments.csv", index=False)

    return {
        "n_changepoints": len(cps),
        "changepoint_dates": [str(d) for d in cps],
        "current_volatility_regime": float(vol_reg.iloc[-1]) if len(vol_reg) else np.nan,
        "current_trend_strength": float(trend.iloc[-1]) if len(trend) else np.nan,
    }
