"""Static matplotlib plots for analysis and evaluation (no dashboards)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_figure(path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_forecast_with_intervals(
    history: pd.Series,
    point: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    history.plot(ax=ax, label="History", color="black", linewidth=1.5)
    point.plot(ax=ax, label="Forecast", color="steelblue", linestyle="--")
    ax.fill_between(
        point.index,
        lower.reindex(point.index),
        upper.reindex(point.index),
        alpha=0.25,
        color="steelblue",
        label="95% interval",
    )
    ax.set_title(title)
    ax.legend()
    save_figure(path)


def plot_series(
    actual: pd.Series,
    predicted: pd.Series | None,
    title: str,
    path: Path,
    label_pred: str = "Predicted",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    actual.plot(ax=ax, label="Actual", color="black", linewidth=1.5)
    if predicted is not None:
        predicted.plot(ax=ax, label=label_pred, color="steelblue", linestyle="--")
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("Date")
    save_figure(path)


def plot_residuals(residuals: pd.Series, title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    residuals.plot(ax=axes[0], title=f"{title} — Residuals")
    residuals.hist(ax=axes[1], bins=20, edgecolor="black")
    axes[1].set_title("Residual distribution")
    save_figure(path)


def plot_feature_importance(
    names: list[str], values: list[float], title: str, path: Path, top_n: int = 20
) -> None:
    idx = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:top_n]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    ax.barh([names[i] for i in idx][::-1], [values[i] for i in idx][::-1])
    ax.set_title(title)
    save_figure(path)
