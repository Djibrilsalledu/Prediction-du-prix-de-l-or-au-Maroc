.. _results:

Results
=======

This section presents the quantitative outcomes of the pipeline: rolling backtest
performance, regime analysis, bootstrap confidence intervals, and future forecasts.

All output files are written to ``results/``.

----

Rolling backtest summary
-------------------------

The pipeline evaluates all models on **10 expanding-window folds** (2015–2025).
Each fold uses a fresh ``FoldPreprocessor`` fitted on the training half only.

Mean metrics across folds
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 13 13 12 15 13 8

   * - Model
     - RMSE
     - MAE
     - MAPE %
     - Dir. Acc. %
     - Bias
     - R²
   * - **SARIMAX ★**
     - **1 478**
     - **1 230**
     - **6.49**
     - 60.9
     - −797
     - −1.46
   * - Baseline Drift
     - 1 487
     - 1 260
     - 6.83
     - 60.9
     - −756
     - −1.95
   * - ARIMA
     - 1 536
     - 1 296
     - 6.92
     - 55.5
     - −865
     - −1.87
   * - Hybrid ARIMA+XGB
     - 1 544
     - 1 319
     - 7.26
     - 58.2
     - −772
     - −2.41
   * - SARIMA
     - 1 616
     - 1 363
     - 7.42
     - 54.5
     - −927
     - −2.63
   * - Baseline LastValue
     - 1 683
     - 1 430
     - 7.66
     - 0.0
     - −1 098
     - −2.46
   * - XGBoost
     - 1 759
     - 1 525
     - 7.92
     - 60.9
     - −1 438
     - −2.36
   * - LSTM (univariate)
     - 1 979
     - 1 741
     - 9.10
     - 47.3
     - −1 573
     - −3.88
   * - Prophet
     - 2 406
     - 2 026
     - 12.30
     - 46.4
     - −1 100
     - −16.87
   * - Baseline TrainMean
     - 7 954
     - 7 894
     - 44.78
     - 0.0
     - −7 894
     - −114.1

.. note::

   The **rolling backtest R²** is negative for all models because it is computed relative
   to the OOS mean — a non-trivial benchmark for a strongly trending series.  The
   **bootstrap R²** reported below (computed on pooled OOS predictions) is positive and
   in the range 0.71 – 0.87, reflecting strong out-of-sample explanatory power.

Saved to: ``results/tables/model_ranking_backtest.csv``

----

Bootstrap confidence intervals (OOS pooled)
--------------------------------------------

500-sample bootstrap on the pooled out-of-sample predictions (all 10 folds concatenated):

.. list-table::
   :header-rows: 1
   :widths: 30 30 20 30

   * - Model
     - RMSE [95 % CI]
     - R²
     - R² [95 % CI]
   * - **SARIMAX**
     - 1 881 [1 506 – 2 272]
     - **0.859**
     - [0.827 – 0.892]
   * - Hybrid ARIMA+XGB
     - 1 821 [1 506 – 2 129]
     - 0.868
     - [0.840 – 0.894]
   * - ARIMA
     - 1 905 [1 544 – 2 289]
     - 0.856
     - [0.824 – 0.887]
   * - SARIMA
     - 1 938 [1 587 – 2 315]
     - 0.851
     - [0.818 – 0.882]
   * - XGBoost
     - 2 270 [1 818 – 2 724]
     - 0.795
     - [0.749 – 0.837]
   * - LSTM
     - 2 508 [2 118 – 2 938]
     - 0.750
     - [0.696 – 0.789]
   * - Prophet
     - 2 693 [2 324 – 3 078]
     - 0.712
     - [0.612 – 0.774]

Saved to: ``results/tables/metrics_bootstrap_ci.csv``

----

Log vs level target comparison
--------------------------------

Three models were re-evaluated with a log-transformed target:

.. list-table::
   :header-rows: 1
   :widths: 20 18 18 18 18 14

   * - Model
     - RMSE (level)
     - RMSE (log)
     - MAE (level)
     - MAE (log)
     - Best
   * - ARIMA
     - 1 536
     - **1 368**
     - 1 296
     - **1 131**
     - Log ✓
   * - Prophet
     - 2 406
     - **2 292**
     - 2 026
     - **1 993**
     - Log ✓
   * - XGBoost
     - **1 759**
     - 16 628
     - **1 525**
     - 16 599
     - Level ✓

Key finding: Log-space XGBoost fails catastrophically (RMSE ×9.5) because the recursive
back-transformation amplifies small forecast errors exponentially.

Saved to: ``results/tables/log_vs_level_comparison.csv``

----

Regime analysis
----------------

Changepoint detection
~~~~~~~~~~~~~~~~~~~~~~

The ``run_regime_analysis()`` function (``src/analysis/regime_detection.py``) applies
a rolling z-score detector (window = 24 months, threshold = 2.0 standard deviations):

* **57 structural changepoints** detected over the full series.
* Major clusters: 2005–2006 gold bull run, 2008 financial crisis, 2011 all-time high,
  2013 correction, 2016 consolidation, 2019–2020 COVID, 2022 commodity shock,
  2024–2026 new bull market.

Current regime (as of February 2026)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 20 40

   * - Indicator
     - Value
     - Interpretation
   * - Volatility regime (0=low, 1=high)
     - **1.0**
     - High volatility — recent monthly swings exceed the historical median
   * - Trend strength (12-month)
     - **0.578**
     - Moderate-to-strong uptrend

ADF by regime segment
~~~~~~~~~~~~~~~~~~~~~~

ADF tests on the 10 largest segments between consecutive changepoints confirm that the
series is non-stationary within every sub-period (p-values > 0.05 in all segments ≥ 12
observations), validating the universal d = 1 choice.

Saved to: ``results/tables/regime_adf_segments.csv``

----

Model stability across folds
------------------------------

``src/analysis/stability_analysis.py`` exports per-fold RMSE, MAE, R² and bias:

**Per-fold RMSE range (SARIMAX):**

* Best fold: ~400 MAD (stable period 2016–2018)
* Worst fold: ~3 100 MAD (volatile period 2024–2025)
* Standard deviation: 1 165 MAD

The high standard deviation reflects the heteroskedastic nature of the series — gold
price volatility increases proportionally with the price level.

Saved to: ``results/tables/rolling_cv_by_fold.csv``, ``results/tables/model_stability_summary.csv``

----

Future forecasts 2026–2027
---------------------------

All models are re-fit on the **full sample** (Aug 2000 – Feb 2026) and forecast through
December 2027 (22 months).

.. list-table::
   :header-rows: 1
   :widths: 16 15 15 15 15 15 9

   * - Date
     - Ensemble
     - SARIMAX
     - XGBoost
     - ARIMA
     - Prophet
     - LSTM
   * - Mar 2026
     - 45 870
     - 45 929
     - 43 980
     - 45 205
     - 34 840
     - 32 333
   * - Jun 2026
     - 47 261
     - 49 253
     - 43 889
     - 46 142
     - 37 258
     - 33 977
   * - Sep 2026
     - 48 569
     - 51 672
     - 43 885
     - 46 548
     - 38 757
     - 34 454
   * - Dec 2026
     - 49 592
     - 53 635
     - 43 584
     - 46 559
     - 40 127
     - 33 750
   * - Mar 2027
     - 50 351
     - 54 479
     - 43 586
     - 46 240
     - 32 706
     - 32 254
   * - Jun 2027
     - 50 684
     - 54 266
     - 43 576
     - 45 679
     - 38 638
     - 32 085
   * - Sep 2027
     - 50 862
     - 53 386
     - 43 584
     - 44 958
     - 43 798
     - 31 568
   * - Dec 2027
     - **50 906**
     - 52 182
     - 43 588
     - 44 151
     - 41 828
     - 30 974

*All values in MAD per troy ounce.*

Key observations:

* **Ensemble_Weighted** converges smoothly toward ~51 000 MAD by December 2027,
  with a decelerating growth rate — an economically plausible path given trend damping.
* **SARIMAX** peaks around March 2027 (~54 500 MAD) then gradually reverts toward
  52 000 MAD as the mean-reversion component of trend stabilisation takes hold.
* **XGBoost** is the most conservative model, holding near 43 500 MAD throughout.
  This is consistent with its recursive design — without strong recent exogenous signals,
  the model defaults to a near-flat trajectory.
* **Prophet** produces the widest inter-period variance, reflecting its inability to
  model the current high-volatility regime.
* **LSTM** declines over the forecast horizon — a known artefact of univariate
  recursive LSTM models when the series mean-reverts relative to the training distribution.

Saved to: ``results/forecasts/future_forecasts_to_2027_12.csv``

Prediction intervals (95 %)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Residual bootstrap intervals for SARIMAX:

* Mar 2026: 45 929 MAD [tight band ≈ ±0.03]
* Dec 2027: 52 182 MAD [band ≈ ±8 MAD]

.. note::

   The narrow bootstrap intervals arise because the residual bootstrap samples from
   in-sample ARIMA residuals, which have low variance relative to the recent regime.
   True predictive uncertainty at 22-month horizon is substantially larger and should be
   interpreted with caution.

Saved to: ``results/forecasts/forecast_intervals_2027_12.csv``
