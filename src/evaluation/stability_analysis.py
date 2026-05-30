"""Per-fold stability metrics and cross-window analysis."""
from __future__ import annotations

import pandas as pd

from src.utils.config import TABLES_DIR


def export_fold_stability(
    fold_metrics: list[dict[str, dict[str, float]]],
    fold_metadata: list[dict],
) -> pd.DataFrame:
    """Long-format table: one row per model per fold."""
    rows = []
    for fold_i, fm in enumerate(fold_metrics):
        meta = fold_metadata[fold_i] if fold_i < len(fold_metadata) else {}
        for model, metrics in fm.items():
            rows.append(
                {
                    "fold": fold_i,
                    "test_start": meta.get("test_start"),
                    "test_end": meta.get("test_end"),
                    "model": model,
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / "rolling_cv_by_fold.csv", index=False)

    if len(df) == 0:
        return df

    stability = (
        df.groupby("model")[["rmse", "mae", "r2", "bias"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    stability.to_csv(TABLES_DIR / "model_stability_summary.csv")
    return df
