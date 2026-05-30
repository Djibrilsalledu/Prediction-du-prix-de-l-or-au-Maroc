"""Fold-aware pipeline with stabilization, ensembles, and probabilistic outputs."""
from __future__ import annotations

import json
import logging
import shutil

import pandas as pd

from src.analysis.final_report import generate_analysis_report
from src.analysis.regime_detection import run_regime_analysis
from src.evaluation.backtesting import run_rolling_backtest
from src.evaluation.bootstrap import bootstrap_all_models
from src.evaluation.compare_models import (
    evaluate_predictions,
    identify_best_model,
    plot_model_comparisons,
    save_backtest_metrics,
)
from src.evaluation.diagnostics import run_eda_diagnostics
from src.evaluation.stability_analysis import export_fold_stability
from src.feature_engineering.exogenous_future import build_future_exogenous
from src.forecasting.calibration import (
    apply_bias_correction,
    estimate_oos_bias,
    inverse_rmse_weights,
    weighted_ensemble,
)
from src.forecasting.probabilistic import residual_bootstrap_intervals
from src.forecasting.trend_stabilization import apply_stabilization_if_enabled
from src.models.arima_model import ARIMAForecaster
from src.models.ensemble_model import ARIMAXGBHybrid
from src.models.lstm_model import LSTMForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.sarima_model import SARIMAForecaster
from src.models.sarimax_model import SARIMAXForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.fold_preprocessor import FoldPreprocessor, prepare_fold_frames
from src.utils.config import (
    ENABLE_BIAS_CORRECTION,
    ENSEMBLE_MODELS,
    EVENT_COLUMNS,
    FIGURES_DIR,
    FORECAST_END,
    FORECASTS_DIR,
    LOG_TARGET_MODELS,
    MACRO_COLUMNS,
    MODELS_DIR,
    PREDICTION_INTERVAL_ALPHA,
    STABILIZE_MODELS,
    STRICT_EVALUATION_MODE,
    TABLES_DIR,
    TARGET_COL,
    USD_MAD_COL,
)
from src.utils.pipeline_metadata import build_run_metadata, save_run_metadata
from src.utils.plotting import plot_forecast_with_intervals, plot_series
from src.utils.reproducibility import set_global_seed
from src.utils.target_transform import inverse_transform_series, transform_series

logger = logging.getLogger(__name__)


def _exog_cols(df: pd.DataFrame) -> list[str]:
    macro = [c for c in MACRO_COLUMNS if c in df.columns]
    return [USD_MAD_COL] + EVENT_COLUMNS + macro


def _post_process_point(
    name: str,
    forecast: pd.Series,
    history: pd.Series,
    oos_bias: float = 0.0,
) -> pd.Series:
    fc = apply_stabilization_if_enabled(forecast, history, name, STABILIZE_MODELS)
    if ENABLE_BIAS_CORRECTION and oos_bias != 0.0:
        fc = apply_bias_correction(fc, oos_bias)
    return fc


def _fit_predict_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    models: tuple[str, ...] | None = None,
    log_models: frozenset[str] = frozenset(),
    save_arima_diagnostics: bool = False,
    save_xgb_artifacts: bool = False,
) -> dict[str, pd.Series]:
    all_models = models or (
        "ARIMA",
        "SARIMA",
        "SARIMAX",
        "Prophet",
        "XGBoost",
        "Hybrid_ARIMA_XGBoost",
        "LSTM_univariate",
    )
    y_train = train_df[TARGET_COL]
    test_index = test_df.index
    steps = len(test_index)
    exog_train = train_df[_exog_cols(train_df)]
    exog_test = test_df[_exog_cols(test_df)]
    preds: dict[str, pd.Series] = {}

    def _y_for_model(name: str) -> pd.Series:
        key = "ARIMA" if name == "Hybrid_ARIMA_XGBoost" else name
        if key in log_models or (name == "XGBoost" and "XGBoost" in log_models):
            return transform_series(y_train, use_log=True)
        return y_train

    def _to_level(name: str, series: pd.Series) -> pd.Series:
        key = "XGBoost" if name == "XGBoost" else ("ARIMA" if name == "Hybrid_ARIMA_XGBoost" else name)
        if key in log_models:
            return inverse_transform_series(series, use_log=True)
        return series

    if "ARIMA" in all_models:
        m = ARIMAForecaster(save_diagnostics=save_arima_diagnostics)
        m.fit(_y_for_model("ARIMA"))
        raw = _to_level("ARIMA", m.predict(steps, index=test_index))
        preds["ARIMA"] = _post_process_point("ARIMA", raw, y_train)

    if "SARIMA" in all_models:
        m = SARIMAForecaster(save_diagnostics=save_arima_diagnostics)
        m.fit(y_train)
        raw = m.predict(steps, index=test_index)
        preds["SARIMA"] = _post_process_point("SARIMA", raw, y_train)

    if "SARIMAX" in all_models:
        m = SARIMAXForecaster()
        m.fit(y_train, exog_train)
        raw = m.predict(steps, exog=exog_test, index=test_index)
        preds["SARIMAX"] = _post_process_point("SARIMAX", raw, y_train)

    if "Prophet" in all_models:
        m = ProphetForecaster()
        m.fit(_y_for_model("Prophet"), exog_train)
        preds["Prophet"] = _to_level("Prophet", m.predict(steps, exog=exog_test, index=test_index))

    if "XGBoost" in all_models:
        m = XGBoostForecaster(
            use_log_target="XGBoost" in log_models,
            save_artifacts=save_xgb_artifacts,
        )
        m.fit(_y_for_model("XGBoost"), train_df)
        history_for_predict = pd.concat([train_df, test_df])
        preds["XGBoost"] = m.predict(steps, exog=history_for_predict, index=test_index)

    if "Hybrid_ARIMA_XGBoost" in all_models:
        m = ARIMAXGBHybrid(use_log_arima="ARIMA" in log_models)
        m.fit(y_train, train_df)
        history_for_predict = pd.concat([train_df, test_df])
        raw = m.predict(steps, exog=history_for_predict, index=test_index)
        preds["Hybrid_ARIMA_XGBoost"] = _post_process_point("Hybrid_ARIMA_XGBoost", raw, y_train)

    if "LSTM_univariate" in all_models:
        m = LSTMForecaster()
        m.fit(y_train)
        preds["LSTM_univariate"] = m.predict(steps, index=test_index)

    return preds


def _archive_previous_metrics() -> None:
    ranking = TABLES_DIR / "model_ranking_backtest.csv"
    if ranking.exists():
        archive = TABLES_DIR / "model_ranking_backtest_pre_refactor.csv"
        shutil.copy(ranking, archive)


def _run_log_target_backtest(df_raw: pd.DataFrame) -> pd.DataFrame:
    log_set = frozenset(LOG_TARGET_MODELS)

    def level_fn(train_df, test_df):
        return _fit_predict_fold(train_df, test_df, models=LOG_TARGET_MODELS, log_models=frozenset())

    def log_fn(train_df, test_df):
        return _fit_predict_fold(train_df, test_df, models=LOG_TARGET_MODELS, log_models=log_set)

    metrics_level, _, _, _, _ = run_rolling_backtest(df_raw, level_fn, include_baselines=False)
    metrics_log, _, _, _, _ = run_rolling_backtest(df_raw, log_fn, include_baselines=False)

    comparison = []
    for model in LOG_TARGET_MODELS:
        if model not in metrics_level.index or model not in metrics_log.index:
            continue
        comparison.append(
            {
                "model": model,
                "rmse_level": metrics_level.loc[model, "rmse_mean"],
                "rmse_log": metrics_log.loc[model, "rmse_mean"],
                "mae_level": metrics_level.loc[model, "mae_mean"],
                "mae_log": metrics_log.loc[model, "mae_mean"],
                "r2_level": metrics_level.loc[model, "r2_mean"],
                "r2_log": metrics_log.loc[model, "r2_mean"],
                "best_transform": (
                    "log"
                    if metrics_log.loc[model, "rmse_mean"] < metrics_level.loc[model, "rmse_mean"]
                    else "level"
                ),
            }
        )
    comp_df = pd.DataFrame(comparison).set_index("model")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(TABLES_DIR / "log_vs_level_comparison.csv")
    return comp_df


def _prepare_full_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    from src.feature_engineering.features import build_features

    prep = FoldPreprocessor()
    prep.fit(df_raw)
    df = prep.transform(df_raw, split_role="train")
    return build_features(df)


def _forecast_future_all_models(
    df_raw: pd.DataFrame,
    log_models: frozenset[str] = frozenset(),
    metrics_bt: pd.DataFrame | None = None,
    oos_bias: dict[str, float] | None = None,
) -> pd.DataFrame:
    df = _prepare_full_frame(df_raw)
    last_date = df.index.max()
    future_exog = build_future_exogenous(
        df,
        last_observed=last_date,
        forecast_end=FORECAST_END,
        exog_cols=_exog_cols(df),
        allow_extrapolation=not STRICT_EVALUATION_MODE,
    )
    future_index = future_exog.index
    steps = len(future_index)

    y_full = df[TARGET_COL]
    exog_full = df[_exog_cols(df)]
    oos_bias = oos_bias or {}
    forecasts: dict[str, pd.Series] = {}

    arima = ARIMAForecaster(save_diagnostics=False)
    arima.fit(transform_series(y_full, "ARIMA" in log_models))
    raw_arima = inverse_transform_series(
        arima.predict(steps, index=future_index), "ARIMA" in log_models
    )
    forecasts["ARIMA"] = _post_process_point("ARIMA", raw_arima, y_full, oos_bias.get("ARIMA", 0))

    sarima = SARIMAForecaster(save_diagnostics=False)
    sarima.fit(y_full)
    forecasts["SARIMA"] = _post_process_point(
        "SARIMA", sarima.predict(steps, index=future_index), y_full, oos_bias.get("SARIMA", 0)
    )

    sarimax = SARIMAXForecaster()
    sarimax.fit(y_full, exog_full)
    forecasts["SARIMAX"] = _post_process_point(
        "SARIMAX",
        sarimax.predict(steps, exog=future_exog, index=future_index),
        y_full,
        oos_bias.get("SARIMAX", 0),
    )

    prophet = ProphetForecaster()
    prophet.fit(transform_series(y_full, "Prophet" in log_models), exog_full)
    forecasts["Prophet"] = inverse_transform_series(
        prophet.predict(steps, exog=future_exog, index=future_index), "Prophet" in log_models
    )

    xgb = XGBoostForecaster(use_log_target="XGBoost" in log_models, save_artifacts=True)
    xgb.fit(y_full, df)
    bundle = pd.concat([df, future_exog])
    bundle = bundle[~bundle.index.duplicated(keep="last")].sort_index()
    forecasts["XGBoost"] = xgb.predict(steps, exog=bundle, index=future_index)

    hybrid = ARIMAXGBHybrid(use_log_arima="ARIMA" in log_models)
    hybrid.fit(y_full, df)
    forecasts["Hybrid_ARIMA_XGBoost"] = _post_process_point(
        "Hybrid_ARIMA_XGBoost",
        hybrid.predict(steps, exog=bundle, index=future_index),
        y_full,
        oos_bias.get("Hybrid_ARIMA_XGBoost", 0),
    )

    lstm = LSTMForecaster()
    lstm.fit(y_full)
    forecasts["LSTM_univariate"] = lstm.predict(steps, index=future_index)

    if metrics_bt is not None:
        avail = [m for m in ENSEMBLE_MODELS if m in forecasts and m in metrics_bt.index]
        if len(avail) >= 2:
            w = inverse_rmse_weights(metrics_bt, avail)
            forecasts["Ensemble_Weighted"] = weighted_ensemble(
                {k: forecasts[k] for k in avail}, w
            )

    fc_df = pd.DataFrame(forecasts)
    fc_df.index.name = "date"
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    end_tag = pd.Timestamp(FORECAST_END).strftime("%Y_%m")
    fc_df.to_csv(FORECASTS_DIR / f"future_forecasts_to_{end_tag}.csv")

    # Probabilistic fan chart for best statistical model
    try:
        resid = arima.in_sample_residuals()
        interval = residual_bootstrap_intervals(
            forecasts["SARIMAX"] if "SARIMAX" in fc_df else forecasts["ARIMA"],
            resid,
            alpha=PREDICTION_INTERVAL_ALPHA,
        )
        interval_df = pd.DataFrame(
            {"point": interval.point, "lower": interval.lower, "upper": interval.upper}
        )
        interval_df.to_csv(FORECASTS_DIR / f"forecast_intervals_{end_tag}.csv")
        plot_forecast_with_intervals(
            y_full,
            interval.point,
            interval.lower,
            interval.upper,
            "Production forecast with uncertainty bands",
            FIGURES_DIR / "forecast_with_intervals.png",
        )
    except Exception as exc:
        logger.warning("Could not build prediction intervals: %s", exc)

    return fc_df


def run_forecast_only(log_models: frozenset[str] | None = None) -> pd.DataFrame:
    df_raw = load_raw_data()
    if log_models is None:
        log_models = frozenset()
    return _forecast_future_all_models(df_raw, log_models=log_models)


def run_pipeline(build_macro: bool = False, forecast_only: bool = False) -> dict:
    set_global_seed()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _archive_previous_metrics()

    if build_macro:
        from src.preprocessing.build_macro import build_macro_indicators

        build_macro_indicators()

    df_raw = load_raw_data()
    logger.info(
        "Loaded %d raw monthly rows (%s to %s) | strict_mode=%s",
        len(df_raw),
        df_raw.index.min(),
        df_raw.index.max(),
        STRICT_EVALUATION_MODE,
    )

    if forecast_only:
        fc = run_forecast_only()
        logger.info("Forecast-only run: %d months through %s", len(fc), FORECAST_END)
        return {
            "forecast_end": str(FORECAST_END),
            "n_future_steps": len(fc),
            "forecast_file": f"future_forecasts_to_{pd.Timestamp(FORECAST_END).strftime('%Y_%m')}.csv",
        }

    run_eda_diagnostics(df_raw)
    regime_summary = run_regime_analysis(df_raw[TARGET_COL])

    def backtest_fn(train_df, test_df):
        return _fit_predict_fold(train_df, test_df, save_xgb_artifacts=False)

    metrics_bt, y_oos, oos_preds, fold_meta, fold_metrics = run_rolling_backtest(
        df_raw, backtest_fn, include_baselines=True
    )
    save_backtest_metrics(metrics_bt, "model_ranking_backtest.csv")
    export_fold_stability(fold_metrics, fold_meta)

    pooled_metrics = evaluate_predictions(
        y_oos, oos_preds, y_train=None, include_baselines=False
    )
    pooled_metrics.to_csv(TABLES_DIR / "model_ranking_pooled_oos.csv")

    bootstrap_df = bootstrap_all_models(y_oos, oos_preds)
    bootstrap_df.to_csv(TABLES_DIR / "metrics_bootstrap_ci.csv", index=False)

    plot_model_comparisons(y_oos, oos_preds, title_prefix="RollingBacktest")
    model_only = metrics_bt[~metrics_bt.index.str.startswith("Baseline")]
    best = identify_best_model(model_only, rmse_col="rmse_mean")
    logger.info("Best model (fold-aware backtest, excl. baselines): %s", best)

    oos_bias = {
        name: estimate_oos_bias(y_oos, pred)
        for name, pred in oos_preds.items()
        if not name.startswith("Baseline")
    }
    pd.Series(oos_bias, name="bias").to_csv(TABLES_DIR / "oos_bias_by_model.csv")

    log_comp = _run_log_target_backtest(df_raw)
    log_for_future = frozenset(
        m for m in LOG_TARGET_MODELS
        if m in log_comp.index and log_comp.loc[m, "best_transform"] == "log"
    )
    future_fc = _forecast_future_all_models(
        df_raw, log_models=log_for_future, metrics_bt=model_only, oos_bias=oos_bias
    )

    prod_model = "Ensemble_Weighted" if "Ensemble_Weighted" in future_fc.columns else best
    if prod_model in future_fc.columns:
        plot_series(
            df_raw[TARGET_COL],
            future_fc[prod_model],
            f"Future forecast — {prod_model}",
            FIGURES_DIR / f"future_forecast_{prod_model.lower()}.png",
        )

    generate_analysis_report(
        best_model=prod_model,
        regime_summary=regime_summary,
        metrics_bt=metrics_bt,
        log_comp=log_comp,
    )

    xgb_final = XGBoostForecaster(
        use_log_target="XGBoost" in log_for_future,
        save_artifacts=True,
    )
    train_full, _, _ = prepare_fold_frames(df_raw, df_raw.iloc[-1:])
    xgb_final.fit(train_full[TARGET_COL], train_full)

    metadata = build_run_metadata(
        best_model=prod_model,
        feature_lists={"XGBoost_final": xgb_final.feature_columns},
        fold_preprocessors=[m.get("preprocessing", {}) for m in fold_meta],
        evaluation_mode="fold_aware_expanding_backtest_v2",
        extra={
            "strict_evaluation_mode": STRICT_EVALUATION_MODE,
            "trend_stabilization": list(STABILIZE_MODELS),
            "oos_bias": oos_bias,
            "regime": regime_summary,
        },
    )
    save_run_metadata(metadata)

    summary = {
        "best_model_backtest": best,
        "recommended_production_model": prod_model,
        "backtest_metrics": metrics_bt.reset_index().to_dict(orient="records"),
        "log_vs_level": log_comp.reset_index().to_dict(orient="records"),
        "forecast_end": str(FORECAST_END),
        "strict_evaluation_mode": STRICT_EVALUATION_MODE,
        "evaluation": "fold_aware_expanding_backtest_v2",
    }
    with open(MODELS_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
