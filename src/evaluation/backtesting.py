"""Fold-aware rolling / expanding window backtesting."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.baselines import compute_all_baselines
from src.evaluation.metrics import compute_metrics
from src.preprocessing.fold_preprocessor import prepare_fold_frames
from src.utils.alignment import validate_fold_frames
from src.utils.config import (
    BACKTEST_INITIAL_TRAIN_MONTHS,
    BACKTEST_ROLLING_TRAIN_WINDOW,
    BACKTEST_STEP_MONTHS,
    BACKTEST_STRATEGY,
    BACKTEST_TEST_MONTHS,
    MODELS_DIR,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


@dataclass
class BacktestSplit:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def generate_backtest_splits(
    n_obs: int,
    initial_train: int = BACKTEST_INITIAL_TRAIN_MONTHS,
    test_horizon: int = BACKTEST_TEST_MONTHS,
    step: int = BACKTEST_STEP_MONTHS,
    strategy: str = BACKTEST_STRATEGY,
    rolling_train_window: int | None = BACKTEST_ROLLING_TRAIN_WINDOW,
) -> list[BacktestSplit]:
    splits: list[BacktestSplit] = []
    test_start = initial_train
    fold = 0

    while test_start + test_horizon <= n_obs:
        if strategy == "rolling" and rolling_train_window is not None:
            train_start = max(0, test_start - rolling_train_window)
        else:
            train_start = 0

        splits.append(
            BacktestSplit(
                fold=fold,
                train_start=train_start,
                train_end=test_start,
                test_start=test_start,
                test_end=test_start + test_horizon,
            )
        )
        fold += 1
        test_start += step

    if not splits:
        raise ValueError(
            f"Cannot build backtest splits (n={n_obs}, initial_train={initial_train}, "
            f"test_horizon={test_horizon})."
        )
    return splits


def aggregate_fold_metrics(fold_metrics: list[dict[str, dict[str, float]]]) -> pd.DataFrame:
    models = fold_metrics[0].keys()
    rows = []
    for model in models:
        keys = fold_metrics[0][model].keys()
        row = {"model": model}
        for k in keys:
            vals = [fm[model][k] for fm in fold_metrics if model in fm]
            row[f"{k}_mean"] = float(np.nanmean(vals))
            row[f"{k}_std"] = float(np.nanstd(vals))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").sort_values("rmse_mean")


def run_rolling_backtest(
    df_raw: pd.DataFrame,
    fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], dict[str, pd.Series]],
    *,
    include_baselines: bool = True,
    save_fold_metadata: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict[str, pd.Series], list[dict]]:
    """
    Per-fold workflow:
      1. Chronological train/test split on raw data
      2. Fit FoldPreprocessor on train only; transform train & test
      3. Build features (train history available for test lags — no future target leakage)
      4. fit_predict_fn(train_processed, test_processed)
      5. Aggregate metrics

    fit_predict_fn receives fold-processed frames including TARGET_COL and exogenous columns.
    """
    splits = generate_backtest_splits(len(df_raw))
    fold_metrics: list[dict[str, dict[str, float]]] = []
    pooled_actual: list[pd.Series] = []
    pooled_preds: dict[str, list[pd.Series]] = {}
    fold_metadata: list[dict] = []

    for sp in splits:
        train_raw = df_raw.iloc[sp.train_start : sp.train_end]
        test_raw = df_raw.iloc[sp.test_start : sp.test_end]

        train_df, test_df, prep = prepare_fold_frames(
            train_raw, test_raw, fold_id=sp.fold
        )
        validate_fold_frames(train_df, test_df, TARGET_COL)

        meta = {
            "fold": sp.fold,
            "train_start": str(train_raw.index.min()),
            "train_end": str(train_raw.index.max()),
            "test_start": str(test_raw.index.min()),
            "test_end": str(test_raw.index.max()),
            "preprocessing": prep.to_dict(),
            "n_train": len(train_df),
            "n_test": len(test_df),
        }

        preds = fit_predict_fn(train_df, test_df)
        meta["models"] = list(preds.keys())

        y_test = test_df[TARGET_COL]
        y_train = train_df[TARGET_COL]

        if include_baselines:
            preds.update(compute_all_baselines(y_train, test_df.index))

        fold_m: dict[str, dict[str, float]] = {}
        for name, pred in preds.items():
            fold_m[name] = compute_metrics(y_test, pred)
            pooled_preds.setdefault(name, []).append(pred)

        fold_metrics.append(fold_m)
        pooled_actual.append(y_test)
        fold_metadata.append(meta)

        logger.info(
            "Fold %d | train %s→%s | test %s→%s | models=%s",
            sp.fold,
            meta["train_start"],
            meta["train_end"],
            meta["test_start"],
            meta["test_end"],
            meta["models"],
        )

    metrics_df = aggregate_fold_metrics(fold_metrics)

    y_oos = pd.concat(pooled_actual).sort_index()
    y_oos = y_oos[~y_oos.index.duplicated(keep="last")]

    oos_predictions = {
        name: pd.concat(series_list).sort_index()
        for name, series_list in pooled_preds.items()
    }
    for name in oos_predictions:
        s = oos_predictions[name]
        oos_predictions[name] = s[~s.index.duplicated(keep="last")]

    if save_fold_metadata:
        path = Path(MODELS_DIR) / "backtest_fold_metadata.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fold_metadata, f, indent=2, default=str)

    return metrics_df, y_oos, oos_predictions, fold_metadata, fold_metrics
