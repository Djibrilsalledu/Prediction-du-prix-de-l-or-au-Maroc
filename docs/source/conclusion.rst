.. _conclusion:

Conclusion & Perspectives
=========================

Summary of findings
--------------------

This project built an academic-grade, fully reproducible pipeline for forecasting Moroccan
gold prices at monthly frequency.  The main conclusions are:

**1. SARIMAX is the best individual model**

With a mean RMSE of **1 478 MAD**, mean MAPE of **6.49 %** and directional accuracy of
**60.9 %** across 10 expanding-window folds, SARIMAX outperforms all other single models.
The key drivers of this performance are:

* Direct inclusion of the USD/MAD exchange rate (coefficient +1 048) — the dominant
  short-run price driver.
* Moroccan event regressors that capture predictable demand seasonality invisible to
  univariate models.
* The ``auto_arima`` order-selection procedure, which adapts ARIMA orders per fold.

**2. Ensemble_Weighted is recommended for production**

The inverse-RMSE weighted blend of SARIMAX, XGBoost and Hybrid ARIMA+XGBoost achieves
lower sensitivity to any single model's failure mode.  It converges smoothly toward
~51 000 MAD/oz by December 2027 — an economically plausible path given trend damping.

**3. Moroccan socio-cultural events have a quantifiable effect**

Event coefficients from the full-sample SARIMAX fit:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Event
     - Coefficient (MAD)
     - Interpretation
   * - Ramadan (x2)
     - +22.0
     - Pre-Aïd jewellery purchasing
   * - Aïd Al-Adha (x3)
     - +70.7
     - Peak bridal gold demand
   * - Wedding season (x4)
     - +48.1
     - 6-month peak season
   * - Aïd Al-Fitr (x5)
     - +28.1
     - Gift-giving demand

These effects persist after controlling for FX, macro and trend, confirming that
cultural seasonality is a genuine, independent price driver.

**4. Deep learning (LSTM) and Prophet underperform on this dataset**

* Prophet achieves MAPE 12.3 % — nearly double SARIMAX.  Its piecewise-linear trend
  cannot track the sharp regime shifts in the gold price.
* LSTM achieves R² = 0.750 as a univariate benchmark, which is respectable but clearly
  below the exogenous statistical models.

**5. Anti-leakage methodology is critical**

The three test suites (``test_leakage_detection.py``, ``test_backtest_integrity.py``,
``test_xgboost_alignment.py``) demonstrate that naive global imputation or combined-frame
fitting produces materially different (and optimistic) backtest results.  Every
performance figure in this project is leak-free.

----

Limitations
-----------

**1. High-volatility regime underforecasting**

All models systematically underestimate prices during the 2024–2026 bull run
(mean bias −772 to −1 573 MAD).  The recent regime is characterised by exceptionally
fast monthly price increases that fall outside the range of the historical training
distribution.

**2. FX and macro assumptions in long-horizon forecasting**

Beyond 12 months, the pipeline holds FX and macro regressors at their last observed
value (``STRICT_EVALUATION_MODE = True``).  Real forecasts would require macro
projections from BAM or IMF, which carry their own uncertainty.

**3. LSTM is univariate**

The LSTM implementation uses only ``gold_price_mad`` as input.  A multivariate LSTM
incorporating FX and event regressors might perform comparably to SARIMAX; this
extension is identified as future work.

**4. Data availability**

The ``macro_indicators.csv`` file relies on proxy constructions when primary BAM and
FRED feeds are unavailable.  Any proxy mismatch introduces noise into the macro
regressors.

**5. Prediction interval underestimation**

The residual bootstrap intervals are very tight (< 10 MAD at 22-month horizon).  This
is because they sample from in-sample residuals, which do not reflect regime-switching
or structural uncertainty.  A proper predictive interval should incorporate parameter
uncertainty and scenario analysis.

----

Future work
-----------

The following extensions would strengthen the pipeline:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Extension
     - Description
   * - **Multivariate LSTM**
     - Add FX, event and macro regressors to the LSTM input sequence
   * - **Transformer / Temporal Fusion**
     - Explore TFT (Lim et al., 2021) for long-horizon multi-step forecasting with
       interpretable attention weights
   * - **Scenario-based forecasting**
     - Run the pipeline under BAM rate scenarios (stable / hawkish / dovish) and
       USD/MAD scenarios (appreciation / depreciation paths)
   * - **Real-time data integration**
     - Connect to BAM API and World Gold Council feeds for automatic monthly updates
   * - **Conformal prediction intervals**
     - Replace residual bootstrap with conformal prediction for distribution-free
       coverage guarantees
   * - **Moroccan retail price survey**
     - Add a direct observation of Moroccan physical gold retail prices to validate
       the ``gold_price_mad = gold_usd × usd_mad`` identity against market frictions
       (import taxes, dealer margins)
   * - **GARCH volatility modelling**
     - Fit a GARCH(1,1) on SARIMAX residuals to improve the volatility regime signal
       and produce heteroskedastic prediction intervals

----

References
----------

* Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*,
  3rd edition. OTexts.
* Taylor, S.J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.
* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
  *ACM KDD*, 785–794.
* Lim, B., Arık, S.Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers
  for interpretable multi-horizon time series forecasting.
  *International Journal of Forecasting*, 37(4), 1748–1764.
* Box, G.E.P., Jenkins, G.M., Reinsel, G.C., & Ljung, G.M. (2015).
  *Time Series Analysis: Forecasting and Control*, 5th edition. Wiley.
* Bank Al-Maghrib (2024). *Rapport annuel sur la situation économique, monétaire et
  financière*. BAM.

----

Reproducibility checklist
--------------------------

Before re-running the pipeline:

.. code-block:: bash

   # 1. Verify environment
   pip install -r requirements.txt

   # 2. Run unit tests
   pytest tests/ -v

   # 3. Run full pipeline
   python main.py

   # 4. Verify outputs
   ls results/tables/model_ranking_backtest.csv
   ls results/forecasts/future_forecasts_to_2027_12.csv
   ls results/models/pipeline_run_metadata.json

The ``pipeline_run_metadata.json`` file records the exact package versions, best model,
feature lists and fold preprocessing parameters used in the run.
