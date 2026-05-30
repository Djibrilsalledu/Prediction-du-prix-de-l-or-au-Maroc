"""
Moroccan gold price forecasting — main entry point.

Usage:
    python main.py
    python main.py --build-macro
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.forecasting.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Moroccan gold price monthly forecasting (academic pipeline)."
    )
    parser.add_argument(
        "--build-macro",
        action="store_true",
        help="Download/build optional macro_indicators.csv before running.",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Skip backtest; regenerate future forecasts only (uses FORECAST_END).",
    )
    args = parser.parse_args()
    summary = run_pipeline(build_macro=args.build_macro, forecast_only=args.forecast_only)
    print("\n=== Pipeline complete ===")
    if "best_model_backtest" in summary:
        print(f"Best model (rolling backtest RMSE): {summary['best_model_backtest']}")
    if "recommended_production_model" in summary:
        print(f"Recommended production model: {summary['recommended_production_model']}")
    print(f"Future forecasts saved through: {summary['forecast_end']}")
    if "forecast_file" in summary:
        print(f"Forecast file: results/forecasts/{summary['forecast_file']}")
    print("See results/figures, results/tables, results/forecasts")


if __name__ == "__main__":
    main()
