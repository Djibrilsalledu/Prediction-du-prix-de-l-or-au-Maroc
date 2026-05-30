"""Publication-quality summary report from pipeline outputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import MODELS_DIR, RESULTS_DIR, TABLES_DIR


def generate_analysis_report(
    *,
    best_model: str,
    regime_summary: dict,
    metrics_bt: pd.DataFrame,
    log_comp: pd.DataFrame | None = None,
) -> Path:
    """Write markdown interpretation to results/models/analysis_report.md"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "analysis_report.md"

    model_only = metrics_bt[~metrics_bt.index.str.startswith("Baseline")]
    lines = [
        "# Moroccan Gold Price Forecasting — Analysis Report",
        "",
        "## Recommended production model",
        f"**{best_model}** (lowest mean RMSE on expanding-window rolling backtest, excl. baselines).",
        "",
        "## Regime context",
        f"- Detected changepoints: {regime_summary.get('n_changepoints', 'N/A')}",
        f"- Current volatility regime (1=high): {regime_summary.get('current_volatility_regime', 'N/A')}",
        f"- Current trend strength: {regime_summary.get('current_trend_strength', 'N/A')}",
        "",
        "## Model ranking (rolling CV mean RMSE)",
        "",
        model_only[["rmse_mean", "mae_mean", "r2_mean", "bias_mean", "directional_accuracy_mean"]]
        .round(3)
        .to_string(),
        "",
        "## Interpretation",
        "",
        "### SARIMAX / ARIMA",
        "- Capture linear dynamics and exogenous FX/events well; highest R² on in-sample and short horizons.",
        "- Risk: unconstrained trend extrapolation on long horizons → addressed via trend damping post-processing.",
        "",
        "### XGBoost",
        "- More conservative recursive forecasts; better economic plausibility at long horizons.",
        "- Captures nonlinear interactions via lagged features; sensitive to recursive error accumulation.",
        "",
        "### Prophet / LSTM_univariate",
        "- Underperform in rolling CV: Prophet misses sharp regime shifts; LSTM is univariate-only benchmark.",
        "",
        "### Hybrid ARIMA + XGBoost",
        "- Combines ARIMA structure with ML residual correction; useful when systematic bias patterns exist.",
        "",
        "## Strengths / weaknesses",
        "",
        "| Model | Strength | Weakness |",
        "|-------|----------|----------|",
        "| SARIMAX | Exogenous interpretability, strong short-horizon fit | Long-horizon trend over-extrapolation |",
        "| ARIMA | Simple, stable baseline | No exogenous drivers |",
        "| XGBoost | Nonlinear, conservative long path | Recursive drift, needs careful lags |",
        "| Hybrid | Bias/residual correction | Two-stage complexity |",
        "| Ensemble | Robustness across regimes | Less interpretable |",
        "",
        "## Methodology notes",
        "- Fold-aware preprocessing (no global imputation).",
        "- Strict anti-leakage feature engineering (shift ≥ 1).",
        "- Expanding-window walk-forward validation.",
        "- Trend damping + optional OOS bias correction for production forecasts.",
        "",
    ]

    if log_comp is not None and len(log_comp):
        lines.extend(["## Log vs level target", "", log_comp.round(3).to_string(), ""])

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
