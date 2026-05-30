"""Model comparison, baselines, ranking, and visualization."""
from __future__ import annotations

import pandas as pd

from src.evaluation.baselines import compute_all_baselines
from src.evaluation.metrics import compute_metrics, metrics_to_frame
from src.utils.config import FIGURES_DIR, TABLES_DIR
from src.utils.plotting import plot_residuals, plot_series


def evaluate_predictions(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    y_train: pd.Series | None = None,
    *,
    include_baselines: bool = True,
) -> pd.DataFrame:
    """
    Compute RMSE, MAE, MAPE, R², bias, directional accuracy.

    If y_train is provided and include_baselines=True, adds naive baselines.
    """
    preds = dict(predictions)
    if include_baselines and y_train is not None:
        preds.update(compute_all_baselines(y_train, y_true.index))

    results = {name: compute_metrics(y_true, pred) for name, pred in preds.items()}
    metrics_df = metrics_to_frame(results)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(TABLES_DIR / "model_ranking.csv")
    return metrics_df


def save_backtest_metrics(metrics_df: pd.DataFrame, filename: str = "model_ranking_backtest.csv") -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(TABLES_DIR / filename)


def plot_model_comparisons(
    y_true: pd.Series,
    predictions: dict[str, pd.Series],
    title_prefix: str = "Backtest",
) -> None:
    for name, pred in predictions.items():
        plot_series(
            y_true,
            pred,
            f"{title_prefix}: {name} — Actual vs Predicted",
            FIGURES_DIR / f"{title_prefix.lower()}_{name.lower().replace(' ', '_')}_actual_vs_pred.png",
        )
        resid = y_true - pred.reindex(y_true.index)
        plot_residuals(
            resid.dropna(),
            name,
            FIGURES_DIR / f"{title_prefix.lower()}_{name.lower().replace(' ', '_')}_residuals.png",
        )

    import matplotlib.pyplot as plt

    fig_path = FIGURES_DIR / f"{title_prefix.lower()}_all_models_comparison.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    y_true.plot(ax=ax, label="Actual", color="black", linewidth=2)
    for name, pred in predictions.items():
        pred.reindex(y_true.index).plot(ax=ax, label=name, alpha=0.8)
    ax.legend(fontsize=8)
    ax.set_title(f"{title_prefix} — Model comparison")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def identify_best_model(metrics_df: pd.DataFrame, rmse_col: str = "rmse") -> str:
    col = rmse_col if rmse_col in metrics_df.columns else "rmse_mean"
    return str(metrics_df.sort_values(col).index[0])
