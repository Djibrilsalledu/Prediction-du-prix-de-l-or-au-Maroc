"""Leak-safe feature engineering with cyclical time encoding."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import EVENT_COLUMNS, GOLD_USD_COL, MAX_LAG, TARGET_COL, USD_MAD_COL

RAW_EXOG_FOR_ML = {GOLD_USD_COL, USD_MAD_COL}


def _add_lags(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def _add_rolling(df: pd.DataFrame, col: str, windows: list[int]) -> pd.DataFrame:
    base = df[col].shift(1)
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = base.rolling(w, min_periods=1).mean()
        df[f"{col}_roll_std_{w}"] = base.rolling(w, min_periods=2).std()

    return df


def _cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    month = df.index.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def _lagged_interaction(
    df: pd.DataFrame, col_a: str, col_b: str, lag: int = 1, name: str | None = None
) -> pd.DataFrame:
    fname = name or f"{col_a}_lag{lag}_x_{col_b}_lag{lag}"
    df[fname] = df[col_a].shift(lag) * df[col_b].shift(lag)
    return df


def build_features(df: pd.DataFrame, include_macro: bool = True) -> pd.DataFrame:
    """Features at t use only information from t-1 and earlier."""
    out = df.copy()
    target = TARGET_COL
    y_past = out[target].shift(1)

    out = _add_lags(out, target, [1, 3, 6, 12])
    out = _add_rolling(out, target, [3, 6, 12])
    out[f"{target}_pct_change_1"] = y_past.pct_change(1, fill_method=None)
    out[f"{target}_pct_change_12"] = y_past.pct_change(12, fill_method=None)
    out[f"{target}_momentum_3"] = y_past - out[target].shift(4)
    out[f"{target}_momentum_12"] = y_past - out[target].shift(13)

    out = _cyclical_month(out)
    out["quarter"] = out.index.quarter
    out["year"] = out.index.year

    for ev in EVENT_COLUMNS:
        if ev not in out.columns:
            continue
        out = _add_lags(out, ev, [1, 3])
        ev_past = out[ev].shift(1)
        out[f"{ev}_roll_mean_3"] = ev_past.rolling(3, min_periods=1).mean()
        out[f"{ev}_roll_mean_12"] = ev_past.rolling(12, min_periods=1).mean()
        out[f"{ev}_cumulative_12"] = ev_past.rolling(12, min_periods=1).sum()

    if GOLD_USD_COL in out.columns:
        out = _add_lags(out, GOLD_USD_COL, [1])
        out["gold_price_usd_lag1"] = out[GOLD_USD_COL].shift(1)
    if USD_MAD_COL in out.columns:
        out = _add_lags(out, USD_MAD_COL, [1])
        out["usd_mad_lag1"] = out[USD_MAD_COL].shift(1)
    if GOLD_USD_COL in out.columns and USD_MAD_COL in out.columns:
        _lagged_interaction(
            out, GOLD_USD_COL, USD_MAD_COL, lag=1, name="gold_price_usd_lag1_x_usd_mad_lag1"
        )
        fx_past = out[USD_MAD_COL].shift(1)
        out[f"{USD_MAD_COL}_pct_change_1"] = fx_past.pct_change(1, fill_method=None)
        out = _add_rolling(out, USD_MAD_COL, [3, 6, 12])

    macro_like = [c for c in (
        "inflation_morocco",
        "policy_rate_bam",
        "oil_brent_usd",
        "fed_funds_rate",
        "dxy_index",
    ) if c in out.columns]

    if include_macro:
        for mc in macro_like:
            mc_past = out[mc].shift(1)
            out[f"{mc}_pct_change_1"] = mc_past.pct_change(1, fill_method=None)
            out[f"{mc}_roll_mean_6"] = mc_past.rolling(6, min_periods=1).mean()
            out[f"{mc}_roll_vol_6"] = mc_past.pct_change(1, fill_method=None).rolling(6, min_periods=2).std()
            if GOLD_USD_COL in out.columns:
                _lagged_interaction(
                    out, GOLD_USD_COL, mc, lag=1, name=f"gold_price_usd_lag1_x_{mc}_lag1"
                )

    # Rolling volatility & trend strength (lagged target only)
    ret_past = y_past.pct_change(1, fill_method=None)
    out[f"{target}_roll_vol_6"] = ret_past.rolling(6, min_periods=2).std()
    out[f"{target}_roll_vol_12"] = ret_past.rolling(12, min_periods=3).std()
    out[f"{target}_trend_strength_12"] = (
        (y_past - out[target].shift(13)) / out[target].shift(13).replace(0, np.nan)
    ).abs()
    out[f"{target}_roll_return_3"] = y_past.pct_change(3, fill_method=None)

    return out


def get_ml_feature_columns(
    df: pd.DataFrame,
    frozen: list[str] | None = None,
) -> list[str]:
    if frozen is not None:
        return [c for c in frozen if c in df.columns]
    exclude = {TARGET_COL, GOLD_USD_COL, USD_MAD_COL} | RAW_EXOG_FOR_ML
    return [
        c
        for c in df.columns
        if c not in exclude
        and df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        and not c.startswith("Unnamed")
    ]


def training_row_mask(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    """
    Rows usable for ML training: skip warmup from lags only (first MAX_LAG rows),
    not full dropna on all columns.
    """
    n = len(feature_df)
    warmup = pd.Series(False, index=feature_df.index)
    if n > MAX_LAG:
        warmup.iloc[MAX_LAG:] = True
    else:
        return pd.Series(False, index=feature_df.index)

    available = feature_df[feature_cols].notna().all(axis=1)
    return warmup & available


def train_test_split_chronological(
    df: pd.DataFrame, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0 or test_size >= len(df):
        raise ValueError("Invalid test_size for chronological split.")
    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()
    return train, test
