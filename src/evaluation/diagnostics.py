"""Exploratory and econometric diagnostics saved to results/."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from src.utils.config import EVENT_COLUMNS, FIGURES_DIR, TARGET_COL


def run_eda_diagnostics(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    y = df[TARGET_COL].astype(float)

    # Correlation heatmap
    num = df.select_dtypes(include="number")
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im)
    ax.set_title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close()

    # Decomposition
    if len(y) >= 24:
        decomp = seasonal_decompose(y, model="additive", period=12, extrapolate_trend="freq")
        fig = decomp.plot()
        fig.set_size_inches(10, 8)
        plt.savefig(output_dir / "seasonal_decomposition.png", dpi=150)
        plt.close()

    # ADF
    adf_stat, pval, *_ = adfuller(y.dropna())
    with open(output_dir / "adf_target.txt", "w", encoding="utf-8") as f:
        f.write(f"ADF stat: {adf_stat}\nADF p-value: {pval}\n")

    # ACF target
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(y.dropna(), lags=36, ax=ax)
    ax.set_title("Target autocorrelation")
    plt.tight_layout()
    plt.savefig(output_dir / "target_acf.png", dpi=150)
    plt.close()

    # Moroccan events impact
    fig, axes = plt.subplots(len(EVENT_COLUMNS), 1, figsize=(10, 2.5 * len(EVENT_COLUMNS)))
    if len(EVENT_COLUMNS) == 1:
        axes = [axes]
    for ax, col in zip(axes, EVENT_COLUMNS):
        ax2 = ax.twinx()
        y.plot(ax=ax, color="black", alpha=0.6, label=TARGET_COL)
        df[col].plot(ax=ax2, color="darkorange", alpha=0.7, label=col)
        ax.set_title(f"{TARGET_COL} vs {col}")
    plt.tight_layout()
    plt.savefig(output_dir / "moroccan_events_impact.png", dpi=150)
    plt.close()

    # Macro influence if present
    macro_cols = [c for c in df.columns if c in (
        "inflation_morocco", "policy_rate_bam", "oil_brent_usd", "fed_funds_rate", "dxy_index"
    )]
    if macro_cols:
        fig, axes = plt.subplots(len(macro_cols), 1, figsize=(10, 2.5 * len(macro_cols)))
        if len(macro_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, macro_cols):
            ax2 = ax.twinx()
            y.plot(ax=ax, color="black", alpha=0.6)
            df[col].plot(ax=ax2, color="green", alpha=0.7)
            ax.set_title(f"{TARGET_COL} vs {col}")
        plt.tight_layout()
        plt.savefig(output_dir / "macro_influence.png", dpi=150)
        plt.close()
