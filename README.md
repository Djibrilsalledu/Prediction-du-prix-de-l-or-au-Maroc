# Moroccan Gold Price Forecasting

Academic-grade **monthly** time series forecasting system for Moroccan gold prices (MAD), combining econometrics, statistical modeling, and machine learning. No dashboards or web UI — outputs are static figures, tables, and CSV forecasts.

## Objective

- Predict **Moroccan gold price in MAD** (`gold_price_usd × usd_mad`)
- Compare **ARIMA, SARIMA, SARIMAX, Prophet, XGBoost, LSTM**
- Quantify **Moroccan socio-cultural** (event intensity) and **macro** effects
- Forecast through **December 2026** (monthly `MS` frequency)

## Data

| File | Description |
|------|-------------|
| `data/gold_prices.csv` | International gold quote (USD) |
| `data/usd_mad.csv` | USD/MAD exchange rate |
| `data/moroccan_events.csv` | Monthly intensity: ramadan, eid_alfitr, eid_aladha, wedding_season, mre_season |
| `data/macro_indicators.csv` | Optional macro regressors (build with `--build-macro`) |

All series: **monthly**, ~2000-08 to 2026-02.

## Structure

```
project/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── models/
│   ├── evaluation/
│   ├── forecasting/
│   └── utils/
├── results/
├── requirements.txt
├── main.py
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Run

```bash
python main.py
python main.py --build-macro   # optional: fetch Brent/DXY proxies via yfinance
```

## Methodology

### Target

`gold_price_mad = gold_price_usd * usd_mad` (local Moroccan price in MAD).

### Train / test

- Chronological split — **last 24 months** holdout (no shuffle).
- Scalers fit **only on training** data (XGBoost, LSTM).

### Future exogenous variables

| Variable type | Strategy |
|---------------|----------|
| Moroccan events | Seasonal profile (month-of-year mean) blended with last-year pattern |
| USD/MAD & macro | Linear trend on last 24 months, clipped to ±15% of last value |

Documented in `src/feature_engineering/exogenous_future.py`.

### Models

| Model | Role |
|-------|------|
| ARIMA | Univariate; ADF + ACF/PACF diagnostics |
| SARIMA | Seasonal (m=12) |
| SARIMAX | Exog: FX, events, optional macro |
| Prophet | Yearly + custom monthly seasonality; event regressors |
| XGBoost | Engineered lags/rolls; SHAP when available |
| LSTM | Sequence lookback=12; dropout, early stopping, LR schedule |

### Metrics

RMSE, MAE, MAPE, R², bias, directional accuracy — saved to `results/tables/model_ranking.csv`.

## Outputs

- `results/figures/` — EDA, diagnostics, actual vs predicted, residuals
- `results/tables/` — model ranking, SARIMAX coefficients
- `results/forecasts/future_forecasts_to_2026_12.csv`
- `results/models/pipeline_summary.json`

## Notebook

`notebooks/01_exploratory_analysis.ipynb` — interactive EDA aligned with the pipeline.

## Reproducibility

Fixed seed (`RANDOM_SEED=42`) in `src/utils/reproducibility.py`.

## License

Academic / research use.
