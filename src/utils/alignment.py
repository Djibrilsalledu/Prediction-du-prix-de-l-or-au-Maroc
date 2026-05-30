"""Train/test alignment checks for fold backtesting (no global .loc reindexing)."""
from __future__ import annotations

import pandas as pd


def assert_series_frame_aligned(y: pd.Series, frame: pd.DataFrame, *, context: str = "") -> None:
    """y and frame must be same length and share the same index (chronological split)."""
    prefix = f"{context}: " if context else ""
    if len(y) != len(frame):
        raise ValueError(
            f"{prefix}length mismatch — y={len(y)}, frame={len(frame)}"
        )
    if not y.index.equals(frame.index):
        raise ValueError(
            f"{prefix}index mismatch — align via train_df[TARGET_COL], not external .loc"
        )


def assert_xy_lengths(X, y, *, context: str = "") -> None:
    prefix = f"{context}: " if context else ""
    n_x = len(X)
    n_y = len(y)
    if n_x != n_y:
        raise ValueError(f"{prefix}len(X)={n_x} != len(y)={n_y}")


def validate_fold_frames(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> None:
    """Sanity checks after fold preprocessing."""
    if train_df.index.intersection(test_df.index).size > 0:
        raise ValueError("Train and test indices must not overlap.")
    if target_col not in train_df.columns:
        raise ValueError(f"Missing {target_col} in train fold.")
    assert len(train_df) == len(train_df[target_col])
    assert len(test_df) == len(test_df[target_col])
