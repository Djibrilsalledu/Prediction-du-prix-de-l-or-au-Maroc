"""Unit tests: leakage prevention and backtest structure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtesting import generate_backtest_splits
from src.feature_engineering.features import build_features, get_ml_feature_columns
from src.preprocessing.fold_preprocessor import FoldPreprocessor, prepare_fold_frames
from src.utils.config import GOLD_USD_COL, TARGET_COL, USD_MAD_COL


def _synthetic_raw(n: int = 240) -> pd.DataFrame:
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(42)
    gold = 2000 + np.cumsum(rng.normal(10, 5, n))
    fx = 9.5 + rng.normal(0, 0.05, n)
    df = pd.DataFrame(
        {
            GOLD_USD_COL: gold,
            USD_MAD_COL: fx,
            TARGET_COL: gold * fx,
            "ramadan": rng.random(n),
            "oil_brent_usd": np.nan,
        },
        index=idx,
    )
    df.loc[df.index[50:55], "oil_brent_usd"] = 70.0
    return df


def test_backtest_splits_are_chronological():
    splits = generate_backtest_splits(300, initial_train=180, test_horizon=12, step=12)
    assert len(splits) >= 1
    for sp in splits:
        assert sp.train_end == sp.test_start
        assert sp.test_end > sp.test_start
        assert sp.train_start < sp.train_end


def test_imputer_uses_train_stats_only():
    raw = _synthetic_raw()
    mid = len(raw) // 2
    train_raw = raw.iloc[:mid]
    test_raw = raw.iloc[mid : mid + 24]

    prep = FoldPreprocessor()
    prep.fit(train_raw)
    train_t = prep.transform(train_raw, split_role="train")
    test_t = prep.transform(test_raw, split_role="test")

    train_median = train_raw["oil_brent_usd"].median()
    assert prep.fill_values_["oil_brent_usd"] == pytest.approx(train_median, nan_ok=True)

    # NaN at start of test must not be filled from a value appearing only at end of test
    test_raw2 = test_raw.copy()
    test_raw2["oil_brent_usd"] = np.nan
    test_raw2.iloc[-1, test_raw2.columns.get_loc("oil_brent_usd")] = 999.0
    test_t2 = prep.transform(test_raw2, split_role="test")
    assert test_t2["oil_brent_usd"].iloc[0] != 999.0
    assert test_t2["oil_brent_usd"].iloc[-1] == 999.0


def test_no_target_leakage_in_features():
    raw = _synthetic_raw(120)
    train_raw = raw.iloc[:100]
    test_raw = raw.iloc[100:110]
    train_df, test_df, _ = prepare_fold_frames(train_raw, test_raw)

    feat_cols = get_ml_feature_columns(train_df)
    assert TARGET_COL not in feat_cols
    assert GOLD_USD_COL not in feat_cols
    assert USD_MAD_COL not in feat_cols
    assert "gold_price_usd_lag1_x_usd_mad_lag1" in train_df.columns
    assert f"{GOLD_USD_COL}_x_{USD_MAD_COL}" not in train_df.columns

    # Lag-1 target feature at t equals y_{t-1}
    t = train_df.index[20]
    assert train_df.loc[t, f"{TARGET_COL}_lag_1"] == pytest.approx(
        train_df[TARGET_COL].shift(1).loc[t]
    )


def test_train_test_preprocessing_independent():
    raw = _synthetic_raw()
    train_raw = raw.iloc[:150]
    test_raw = raw.iloc[150:162]

    prep = FoldPreprocessor()
    prep.fit(train_raw)
    t1 = prep.transform(train_raw, split_role="train")
    t2 = prep.transform(test_raw, split_role="test")

    assert t1["oil_brent_usd"].isna().sum() == 0
    assert t2["oil_brent_usd"].isna().sum() == 0
    assert prep.last_train_value_["oil_brent_usd"] == pytest.approx(
        t1["oil_brent_usd"].dropna().iloc[-1]
    )


def test_feature_consistency_across_folds():
    raw = _synthetic_raw()
    cols_a = get_ml_feature_columns(build_features(raw.iloc[:160]))
    cols_b = get_ml_feature_columns(build_features(raw.iloc[:200]))
    assert set(cols_a) == set(cols_b)
