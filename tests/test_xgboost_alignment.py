"""XGBoost must train with aligned train fold only (no combined index)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.features import build_features
from src.models.xgboost_model import XGBoostForecaster
from src.utils.config import GOLD_USD_COL, TARGET_COL, USD_MAD_COL


def _frame(n: int, start: str = "2010-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    rng = np.random.default_rng(0)
    g = 3000 + np.cumsum(rng.normal(5, 2, n))
    fx = 9.0 + rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            GOLD_USD_COL: g,
            USD_MAD_COL: fx,
            TARGET_COL: g * fx,
            "ramadan": 0.0,
        },
        index=idx,
    )


def test_xgboost_fit_train_only_no_keyerror():
    train = _frame(100)
    test = _frame(12, start="2018-05-01")
    y_train = train[TARGET_COL]

    model = XGBoostForecaster(save_artifacts=False)
    model.fit(y_train, train)

    combined = pd.concat([train, test])
    pred = model.predict(len(test), exog=combined, index=test.index)
    assert len(pred) == len(test)


def test_xgboost_rejects_mismatched_index():
    train = _frame(50)
    y_wrong = train[TARGET_COL].copy()
    y_wrong.index = pd.date_range("2000-01-01", periods=len(y_wrong), freq="MS")

    model = XGBoostForecaster(save_artifacts=False)
    with pytest.raises(ValueError, match="index mismatch"):
        model.fit(y_wrong, train)


def test_xgboost_rejects_combined_frame_at_fit():
    train = _frame(80)
    test = _frame(12, start="2016-09-01")
    combined = pd.concat([train, test])
    y_train = train[TARGET_COL]

    model = XGBoostForecaster(save_artifacts=False)
    with pytest.raises(ValueError, match="length mismatch"):
        model.fit(y_train, combined)
