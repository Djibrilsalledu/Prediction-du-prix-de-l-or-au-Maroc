"""Central configuration for paths, columns, and forecasting horizons."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
FORECASTS_DIR = RESULTS_DIR / "forecasts"
MODELS_DIR = RESULTS_DIR / "models"

GOLD_FILE = DATA_DIR / "gold_prices.csv"
USD_MAD_FILE = DATA_DIR / "usd_mad.csv"
EVENTS_FILE = DATA_DIR / "moroccan_events.csv"
MACRO_FILE = DATA_DIR / "macro_indicators.csv"

DATE_COLUMN = "date"
TARGET_COL = "gold_price_mad"
GOLD_USD_COL = "gold_price_usd"
USD_MAD_COL = "usd_mad"

EVENT_COLUMNS = [
    "ramadan",
    "eid_alfitr",
    "eid_aladha",
    "wedding_season",
    "mre_season",
]

MACRO_COLUMNS = [
    "inflation_morocco",
    "policy_rate_bam",
    "oil_brent_usd",
    "fed_funds_rate",
    "dxy_index",
]

FORECAST_END = "2027-12-01"
TEST_SIZE_MONTHS = 24  # legacy reference; evaluation uses rolling backtest
RANDOM_SEED = 42

# Rolling / expanding backtest (replaces single holdout)
BACKTEST_INITIAL_TRAIN_MONTHS = 180
BACKTEST_TEST_MONTHS = 12
BACKTEST_STEP_MONTHS = 12
BACKTEST_STRATEGY = "expanding"  # "expanding" | "rolling"
BACKTEST_ROLLING_TRAIN_WINDOW = 180  # used when BACKTEST_STRATEGY == "rolling"

# Log-target experiment
TARGET_LOG_COL = "log_gold_price_mad"
LOG_TARGET_MODELS = ("ARIMA", "Prophet", "XGBoost")

# Minimum history required after feature lags
MAX_LAG = 12

# Strict evaluation: backtest uses observed test exog only; forecast skips FX/macro extrapolation
STRICT_EVALUATION_MODE = True
ALLOW_EXOG_EXTRAPOLATION = not STRICT_EVALUATION_MODE

# Bootstrap confidence intervals
BOOTSTRAP_N_SAMPLES = 500

# Trend stabilization (SARIMA/SARIMAX/ARIMA long-horizon realism)
TREND_DAMPING_PHI = 0.92
TREND_MAX_MONTHLY_PCT = 0.04  # max 4% monthly move after damping
STABILIZE_MODELS = frozenset({"ARIMA", "SARIMA", "SARIMAX"})

# Bias correction from rolling OOS errors
ENABLE_BIAS_CORRECTION = True

# Ensemble members for production blend
ENSEMBLE_MODELS = ("SARIMAX", "XGBoost", "Hybrid_ARIMA_XGBoost")

# Probabilistic forecast alpha
PREDICTION_INTERVAL_ALPHA = 0.05
