"""
Per-fold preprocessing: fit on train only, transform train/test without future information.

Rules:
- Train: forward-fill within train, then fill remaining NaNs with train-only statistics.
- Test: forward-fill within test using carry from last train value; never bfill from future.
- No global interpolation across the full dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.config import EVENT_COLUMNS, MACRO_COLUMNS


@dataclass
class FoldPreprocessor:
    """Imputer fit on a single training fold."""

    columns: list[str] = field(default_factory=list)
    fill_values_: dict[str, float] = field(default_factory=dict)
    last_train_value_: dict[str, float] = field(default_factory=dict)
    fold_id: int | None = None

    @classmethod
    def default_columns(cls, df: pd.DataFrame) -> list[str]:
        cols = [c for c in MACRO_COLUMNS + EVENT_COLUMNS if c in df.columns]
        return cols

    def fit(self, train: pd.DataFrame, columns: list[str] | None = None) -> "FoldPreprocessor":
        if columns is not None:
            self.columns = columns
        elif not self.columns:
            self.columns = self.default_columns(train)
        self.fill_values_ = {}
        self.last_train_value_ = {}

        for col in self.columns:
            if col not in train.columns:
                continue
            series = train[col].astype(float)
            if col in EVENT_COLUMNS:
                self.fill_values_[col] = float(series.fillna(0).median())
            else:
                self.fill_values_[col] = float(series.median()) if series.notna().any() else 0.0

            valid = series.dropna()
            self.last_train_value_[col] = (
                float(valid.iloc[-1]) if len(valid) else self.fill_values_[col]
            )
        return self

    def transform(self, df: pd.DataFrame, *, split_role: str) -> pd.DataFrame:
        """
        split_role: 'train' | 'test'

        Test rows never use statistics computed on test; only train-fitted fills.
        """
        if split_role not in ("train", "test"):
            raise ValueError("split_role must be 'train' or 'test'")

        out = df.copy()
        for col in self.columns:
            if col not in out.columns:
                continue

            if split_role == "train":
                s = out[col].astype(float).ffill()
                s = s.fillna(self.fill_values_[col])
                out[col] = s
            else:
                vals = out[col].astype(float).to_numpy(copy=True)
                if len(vals) == 0:
                    continue
                if np.isnan(vals[0]):
                    vals[0] = self.last_train_value_.get(col, self.fill_values_.get(col, np.nan))
                for i in range(1, len(vals)):
                    if np.isnan(vals[i]) and not np.isnan(vals[i - 1]):
                        vals[i] = vals[i - 1]
                fill = self.fill_values_.get(col, np.nan)
                vals = np.where(np.isnan(vals), fill, vals)
                out[col] = pd.Series(vals, index=out.index)

        return out

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "columns": self.columns,
            "fill_values": self.fill_values_,
            "last_train_value": self.last_train_value_,
        }


def prepare_fold_frames(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    fold_id: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, FoldPreprocessor]:
    """
    Impute per fold, then build lag features on train+test concat (lags may use train history).
    Returns processed train and test slices.
    """
    from src.feature_engineering.features import build_features

    prep = FoldPreprocessor(fold_id=fold_id)
    prep.fit(train_raw)
    train_imp = prep.transform(train_raw, split_role="train")
    test_imp = prep.transform(test_raw, split_role="test")

    combined = pd.concat([train_imp, test_imp])
    featured = build_features(combined)
    train_out = featured.loc[train_imp.index].copy()
    test_out = featured.loc[test_imp.index].copy()
    return train_out, test_out, prep
