"""Naive forecasting baselines for benchmark comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_last_value(y_train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Naïve : répéter la dernière valeur observée."""
    last = float(y_train.iloc[-1])
    return pd.Series(last, index=test_index, name="baseline_last_value")


def baseline_train_mean(y_train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Moyenne historique du train."""
    mean = float(y_train.mean())
    return pd.Series(mean, index=test_index, name="baseline_train_mean")


def baseline_drift(y_train: pd.Series, test_index: pd.DatetimeIndex) -> pd.Series:
    """Drift linéaire : dernier niveau + pente moyenne par pas."""
    y = y_train.astype(float)
    if len(y) < 2:
        return baseline_last_value(y_train, test_index)

    slope = (y.iloc[-1] - y.iloc[0]) / (len(y) - 1)
    preds = []
    for h in range(1, len(test_index) + 1):
        preds.append(float(y.iloc[-1] + slope * h))
    return pd.Series(preds, index=test_index, name="baseline_drift")


def compute_all_baselines(y_train: pd.Series, test_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    return {
        "Baseline_LastValue": baseline_last_value(y_train, test_index),
        "Baseline_TrainMean": baseline_train_mean(y_train, test_index),
        "Baseline_Drift": baseline_drift(y_train, test_index),
    }
