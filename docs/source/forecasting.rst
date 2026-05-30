.. _forecasting:

Forecasting
===========

This section covers the seven individual models, the hybrid and ensemble strategies,
trend stabilisation, bias correction, and probabilistic interval estimation.

All model code lives in ``src/models/``.  Calibration and post-processing helpers are
in ``src/forecasting/``.

----

Base interface
--------------

Every model inherits from ``BaseForecaster`` (``src/models/base.py``):

.. code-block:: python

   class BaseForecaster(ABC):
       name: str

       def fit(self, y: pd.Series, exog: pd.DataFrame | None = None) -> "BaseForecaster":
           ...

       def predict(
           self, steps: int,
           exog: pd.DataFrame | None = None,
           index: pd.DatetimeIndex | None = None,
       ) -> pd.Series:
           ...

``fit()`` always receives the **training target** ``y`` and optionally the full training
frame ``exog`` (required for SARIMAX and XGBoost).
``predict()`` returns a ``pd.Series`` indexed by the requested future ``DatetimeIndex``.

----

ARIMA — ``arima_model.py``
--------------------------

``ARIMAForecaster`` wraps ``pmdarima.auto_arima`` with ADF pre-check and ACF/PACF
diagnostics.

**Order selection:**

.. code-block:: python

   auto_arima(y, seasonal=False, stepwise=True,
              max_p=3, max_q=3, d=None)

``d`` is determined automatically from the ADF test result.  Given the ADF statistic of
+2.30 and p-value of 0.999, ``d = 1`` is selected in every fold.

**Diagnostics saved on fit:**

* ``results/figures/arima_adf.txt`` — ADF stat and p-value
* ``results/figures/arima_acf_pacf.png`` — ACF and PACF to lag 24

**Trend stabilisation:** applied (``STABILIZE_MODELS`` contains ``"ARIMA"``).

**Log-target experiment:** ARIMA is the best log-space model (RMSE 1 368 vs 1 536
in level space); used in log space in the final production run.

----

SARIMA — ``sarima_model.py``
-----------------------------

``SARIMAForecaster`` extends ``ARIMAForecaster`` with ``seasonal=True, m=12``:

.. code-block:: python

   auto_arima(y, seasonal=True, m=12,
              max_P=2, max_Q=2, stepwise=True)

The ACF plot confirms slow decay and PACF shows a single significant spike at lag 1,
consistent with an ARIMA(1,1,0) × (1,1,0)₁₂ structure.

**Trend stabilisation:** applied.

----

SARIMAX — ``sarimax_model.py``
-------------------------------

``SARIMAXForecaster`` is the best individual model (mean RMSE 1 478 MAD over 10 folds).

It augments SARIMA with exogenous regressors selected in this priority order:

1. ``usd_mad``
2. All five Moroccan event columns
3. Available macro columns

**Coefficient estimates (final full-sample fit):**

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Estimate
     - Interpretation
   * - ``usd_mad`` (x1)
     - +1 048.5
     - Dominant FX effect: 1 MAD weakening → +1 048 MAD/oz
   * - ``eid_aladha`` (x3)
     - +70.7
     - Aïd Al-Adha demand spike
   * - ``wedding_season`` (x4)
     - +48.1
     - Bridal gold demand
   * - ``ramadan`` (x2)
     - +22.0
     - Pre-Aïd jewellery purchases
   * - ``dxy_index`` (x11)
     - −79.2
     - Strong USD reduces gold in MAD terms
   * - ``oil_brent_usd`` (x10)
     - −17.7
     - Risk-on rotation away from gold
   * - ``fed_funds_rate`` (x8)
     - −0.29
     - Opportunity cost of holding gold
   * - ``ma.L1``
     - +0.166
     - MA(1) residual correction term

**Bootstrap R² CI (OOS pooled):** 0.859 [0.827 – 0.892]

**Test exogenous in backtesting:** When ``STRICT_EVALUATION_MODE = True``, the test-fold
exogenous values are taken from the **observed data**, not extrapolated.  This avoids
inflating backtest performance with future macro knowledge.

----

Prophet — ``prophet_model.py``
-------------------------------

``ProphetForecaster`` uses Meta's Prophet library with:

* ``yearly_seasonality = True``
* Custom monthly seasonality (Fourier order 5)
* All five Moroccan event columns added as additive regressors

**Weaknesses observed in backtest:**

Prophet achieves a mean MAPE of **12.3 %** — nearly double that of SARIMAX.  The main
failure mode is inability to track the sharp regime shifts in the gold price (the 2020
COVID shock, the 2022 commodity spike, the 2024–2026 bull run).  Prophet's piecewise-linear
trend cannot adapt quickly enough at fold boundaries.

**Log-target experiment:** Prophet benefits slightly from log space (RMSE 2 292 vs 2 406);
used in log space in production.

----

XGBoost — ``xgboost_model.py``
-------------------------------

``XGBoostForecaster`` performs **recursive multi-step forecasting**: at each future step,
the predicted value is appended to the working history and features are recomputed before
predicting the next step.

**Model hyperparameters:**

.. code-block:: python

   XGBRegressor(
       n_estimators=300,
       max_depth=5,
       learning_rate=0.05,
       subsample=0.9,
       colsample_bytree=0.8,
       random_state=42,
   )

**Scaler:** ``sklearn.preprocessing.StandardScaler`` fit on training features only.

**Feature set:** ~85 columns (see :ref:`feature_engineering`).  Features are frozen after
the final full-sample fit to guarantee identical column ordering at inference time.

**SHAP explanations:** If ``shap`` is installed, a SHAP summary plot is saved to
``results/figures/xgboost_shap_summary.png`` after fitting.

**Key weaknesses:**

* Recursive drift accumulates with horizon — reliability decreases beyond 6 months.
* Log-target transformation degrades performance significantly (RMSE 16 628 vs 1 759
  in level space); XGBoost is always trained in level space.

**Test alignment guard:**

.. code-block:: python

   pytest tests/test_xgboost_alignment.py -v

Verifies that (a) XGBoost raises ``ValueError`` if ``y`` and ``exog`` index are
mismatched, and (b) the combined train+test frame is never passed to ``fit()``.

----

LSTM — ``lstm_model.py``
-------------------------

``LSTMForecaster`` is a **univariate deep-learning benchmark** — it uses only the target
series (no exogenous regressors).

**Architecture:**

.. code-block:: text

   LSTM(64, return_sequences=True)
   Dropout(0.2)
   LSTM(32, return_sequences=False)
   Dropout(0.2)
   Dense(16, relu)
   Dense(1)

**Training:**

* Lookback window: 12 months
* Optimiser: Adam (lr = 0.001)
* Loss: MSE
* Callbacks: ``EarlyStopping(patience=12)``, ``ReduceLROnPlateau(factor=0.5, patience=5)``
* Train / validation split: 85 % / 15 % (time-ordered, no shuffle)
* Scaler: ``MinMaxScaler`` fit on training values only

**Forecast mode:** Recursive (auto-regressive), like XGBoost.

**Interpretation:** The LSTM achieves a bootstrap R² of 0.750 — respectable for a
univariate model with no macro information, but clearly below the exogenous statistical
models.

----

Hybrid ARIMA + XGBoost — ``ensemble_model.py``
-----------------------------------------------

``ARIMAXGBHybrid`` implements **residual modelling**:

1. ARIMA is fit on the training target to capture linear structure.
2. In-sample residuals ``y - ŷ_ARIMA`` are computed.
3. XGBoost is fit on the residuals, using the full feature matrix as input.
4. Forecast = ARIMA point forecast + XGBoost residual correction.

This design reduces the systematic bias of ARIMA (mean bias −865 → −772 MAD)
while keeping the interpretable ARIMA trend structure.

----

Ensemble_Weighted (production) — ``ensemble_model.py``
------------------------------------------------------

``WeightedEnsembleForecaster`` blends three component models:

* **SARIMAX** (~36 % weight)
* **XGBoost** (~31 % weight)
* **Hybrid ARIMA+XGBoost** (~33 % weight)

Weights are proportional to **1 / RMSE** from the rolling backtest:

.. code-block:: python

   from src.forecasting.calibration import inverse_rmse_weights, weighted_ensemble

   weights = inverse_rmse_weights(metrics_bt, ["SARIMAX", "XGBoost", "Hybrid_ARIMA_XGBoost"])
   ensemble_fc = weighted_ensemble(component_forecasts, weights)

The ensemble is not evaluated in the rolling backtest (it is constructed *after* individual
model evaluation).  It is the **recommended production model** because it averages out
model-specific failure modes across volatility regimes.

----

Trend stabilisation — ``trend_stabilization.py``
-------------------------------------------------

Long-horizon forecasts from ARIMA-family models tend to extrapolate the most recent
momentum indefinitely, producing economically implausible paths (e.g. +30 % per year).
The ``damp_forecast()`` function applies three corrections:

1. **Exponential trend decay** — the linear trend component is multiplied by
   :math:`\phi^h` where :math:`\phi = 0.92` and *h* is the horizon in months.
2. **Mean-reversion blend** — each step is blended toward the recent 12-month mean-growth
   path weighted by :math:`(1 - \phi^h)`.
3. **Month-over-month cap** — the predicted change is clipped to ±4 % of the previous
   step (``TREND_MAX_MONTHLY_PCT = 0.04``).

**Applied to:** ``ARIMA``, ``SARIMA``, ``SARIMAX`` (``STABILIZE_MODELS`` frozenset).

----

Bias correction — ``calibration.py``
--------------------------------------

After rolling backtest, the mean out-of-sample prediction error (bias) is computed for
each model:

.. code-block:: python

   from src.forecasting.calibration import estimate_oos_bias, apply_bias_correction

   bias = estimate_oos_bias(y_oos, y_pred)   # positive = overestimate
   corrected = apply_bias_correction(forecast, bias)

OOS bias values (MAD per ounce):

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Model
     - Bias (MAD)
     - Direction
   * - SARIMAX
     - −797
     - Under-forecast
   * - Hybrid ARIMA+XGB
     - −772
     - Under-forecast (lowest)
   * - ARIMA
     - −865
     - Under-forecast
   * - SARIMA
     - −927
     - Under-forecast
   * - XGBoost
     - −1 438
     - Under-forecast (strongest)
   * - LSTM
     - −1 573
     - Under-forecast
   * - Prophet
     - −1 100
     - Under-forecast

All models systematically under-forecast during the 2024–2025 gold bull run.
Bias correction shifts the production forecast upward by the estimated bias value.

----

Probabilistic intervals — ``probabilistic.py``
------------------------------------------------

Fan-chart confidence intervals are produced by **residual bootstrap**:

1. Sample *n_samples = 500* paths from the in-sample ARIMA residuals.
2. For each path, add cumulative noise scaled by :math:`\sqrt{h / H}` to the point forecast.
3. Report the 2.5th and 97.5th percentiles as the 95 % prediction interval.

.. code-block:: python

   from src.forecasting.probabilistic import residual_bootstrap_intervals

   interval = residual_bootstrap_intervals(
       point_forecast=fc,
       in_sample_residuals=arima.in_sample_residuals(),
       n_samples=500,
       alpha=0.05,
   )

The interval widens with horizon, reflecting growing uncertainty in recursive forecasts.
Results are saved to ``results/forecasts/forecast_intervals_2027_12.csv``.
