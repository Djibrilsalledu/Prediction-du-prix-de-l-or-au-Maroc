"""Target transformations for forecasting experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import TARGET_COL, TARGET_LOG_COL


def add_log_target(df: pd.DataFrame, col: str = TARGET_COL) -> pd.DataFrame:
    out = df.copy()
    out[TARGET_LOG_COL] = np.log(out[col].clip(lower=1e-6))
    return out


def transform_series(y: pd.Series, use_log: bool) -> pd.Series:
    if use_log:
        return pd.Series(np.log(y.clip(lower=1e-6)), index=y.index, name=TARGET_LOG_COL)
    return y


def inverse_transform_series(y: pd.Series, use_log: bool) -> pd.Series:
    if use_log:
        return pd.Series(np.exp(y), index=y.index, name=TARGET_COL)
    return y


def inverse_transform_values(values: np.ndarray | float, use_log: bool) -> np.ndarray | float:
    if use_log:
        return np.exp(values)
    return values
