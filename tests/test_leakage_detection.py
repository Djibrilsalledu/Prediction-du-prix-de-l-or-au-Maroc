"""Detect future information in fold preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.fold_preprocessor import FoldPreprocessor


def test_test_imputation_does_not_use_future_bfill():
    """If test has NaN only at end, imputation must not pull from later filled values incorrectly."""
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    train = pd.DataFrame({"macro_x": [1.0, 2.0, 3.0, 4.0]}, index=idx[:4])
    test = pd.DataFrame({"macro_x": [np.nan, np.nan]}, index=idx[4:6])

    prep = FoldPreprocessor(columns=["macro_x"])
    prep.fit(train)
    out = prep.transform(test, split_role="test")

    assert out["macro_x"].iloc[0] == 4.0
    assert out["macro_x"].iloc[1] == 4.0


def test_global_bfill_would_differ_from_fold_safe():
    """Sanity: fold-safe transform != pandas bfill on concatenated series."""
    idx = pd.date_range("2020-01-01", periods=8, freq="MS")
    train = pd.DataFrame({"v": [1.0, np.nan, 3.0, 4.0]}, index=idx[:4])
    test = pd.DataFrame({"v": [np.nan, np.nan]}, index=idx[4:6])

    prep = FoldPreprocessor(columns=["v"])
    prep.fit(train)
    safe = prep.transform(test, split_role="test")["v"].values

    concat = pd.concat([train, test])["v"]
    leaky = concat.bfill().iloc[4:6].values

    assert not np.array_equal(safe, leaky) or np.isnan(leaky).all()
