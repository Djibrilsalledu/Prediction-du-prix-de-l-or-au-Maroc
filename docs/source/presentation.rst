.. _presentation:

Project Presentation
====================

Context & Motivation
--------------------

Gold is universally regarded as a safe-haven asset and inflation hedge.  Its price in
international markets (USD per troy ounce) has risen more than **750 %** between 2000 and
early 2026.  In Morocco, however, the effective price paid by consumers and traders is not
the USD spot price but rather its **MAD equivalent**:

.. math::

   \text{gold\_price\_mad} = \text{gold\_price\_usd} \times \text{usd\_mad}

This double exposure — to the international commodity market **and** to the USD/MAD exchange
rate — creates a forecasting problem that is specific to the Moroccan market and has not been
studied in a dedicated academic pipeline.

Why Morocco?
~~~~~~~~~~~~

Morocco presents three distinctive features that make gold-price forecasting particularly
interesting:

1. **Socio-cultural demand cycles** — Ramadan, Aïd Al-Fitr, Aïd Al-Adha, the wedding season
   (April–September) and the MRE (Moroccan Residents abroad) return season generate
   predictable monthly spikes in jewellery demand that feed through to local gold prices.

2. **FX sensitivity** — The MAD is managed against a currency basket dominated by the EUR and
   USD.  A strong dollar simultaneously lifts the USD gold price and weakens the MAD, creating
   compounding effects on ``gold_price_mad``.

3. **Macro policy context** — Bank Al-Maghrib (BAM) interest-rate decisions and domestic
   inflation interact with import-cost dynamics in ways that differ from advanced economies.

Research objectives
-------------------

The project answers four questions:

* Can classical econometric models (ARIMA, SARIMA, SARIMAX) outperform machine-learning
  baselines (XGBoost, LSTM) for monthly Moroccan gold prices?
* Do Moroccan socio-cultural event indicators provide statistically significant predictive
  power when added as exogenous regressors?
* What is the optimal ensemble strategy for a 12-to-24-month production forecast?
* How does trend stabilisation (damping, mean reversion) affect long-horizon economic
  plausibility?

Academic framework
------------------

This project was carried out at the **École Nationale Supérieure d'Arts et Métiers (ENSAM)**
under the supervision of **Pr. Tawfik Masrour**.

* **Author**: Djibril SALL
* **Level**: Engineering student (Élève-Ingénieur)
* **Scope**: Academic research / reproducible pipeline

Project structure
-----------------

.. code-block:: text

   project/
   ├── data/
   │   ├── gold_prices.csv
   │   ├── usd_mad.csv
   │   ├── moroccan_events.csv
   │   └── macro_indicators.csv
   ├── notebooks/
   │   └── 01_exploratory_analysis.ipynb
   ├── src/
   │   ├── preprocessing/        # data loading, cleaning, fold preprocessor
   │   ├── feature_engineering/  # lag/roll features, event encoding, exog future
   │   ├── models/               # ARIMA, SARIMA, SARIMAX, Prophet, XGBoost, LSTM, Ensemble
   │   ├── evaluation/           # backtesting, metrics, bootstrap CI, stability
   │   ├── forecasting/          # pipeline orchestrator, calibration, trend stabilisation
   │   └── utils/                # config, alignment checks, plotting, reproducibility
   ├── results/
   │   ├── figures/
   │   ├── tables/
   │   ├── forecasts/
   │   └── models/
   ├── tests/
   │   ├── test_leakage_detection.py
   │   ├── test_backtest_integrity.py
   │   └── test_xgboost_alignment.py
   ├── main.py
   ├── requirements.txt
   └── README.md

Pipeline overview
-----------------

The pipeline runs in a single command (``python main.py``) and executes the following stages:

.. list-table::
   :header-rows: 1
   :widths: 5 25 45 25

   * - #
     - Stage
     - Description
     - Output
   * - 1
     - **Data loading**
     - Merge gold, FX, events and macro; compute ``gold_price_mad``; no global imputation
     - Raw ``DataFrame``
   * - 2
     - **EDA diagnostics**
     - Correlation matrix, seasonal decomposition, ADF test, ACF/PACF, event impact plots
     - ``results/figures/``
   * - 3
     - **Regime detection**
     - Rolling z-score changepoints (57 detected), volatility regime, trend strength
     - ``regime_adf_segments.csv``
   * - 4
     - **Rolling backtest**
     - 10 expanding-window folds; fold-aware preprocessing; all models + baselines
     - ``rolling_cv_by_fold.csv``
   * - 5
     - **Model comparison**
     - RMSE/MAE/MAPE/R²/bias/directional accuracy; bootstrap 95 % CI (500 samples)
     - ``model_ranking_backtest.csv``, ``metrics_bootstrap_ci.csv``
   * - 6
     - **Log vs level**
     - Backtest ARIMA/Prophet/XGBoost in log-target space; pick best transform per model
     - ``log_vs_level_comparison.csv``
   * - 7
     - **Future forecast**
     - Full-sample re-fit; trend damping φ=0.92; bias correction; Ensemble_Weighted
     - ``future_forecasts_to_2027_12.csv``
   * - 8
     - **Probabilistic intervals**
     - Residual bootstrap fan chart (95 % PI)
     - ``forecast_intervals_2027_12.csv``
   * - 9
     - **Report**
     - Markdown analysis report + JSON pipeline metadata
     - ``analysis_report.md``, ``pipeline_run_metadata.json``

Reproducibility
---------------

A fixed random seed (``RANDOM_SEED = 42``) is applied to NumPy, Python's ``random`` module,
and TensorFlow at the start of every run via ``src/utils/reproducibility.py``.

The full package-version fingerprint is saved to ``results/models/pipeline_run_metadata.json``
on each run.

Run the test suite to verify pipeline integrity before any experiment:

.. code-block:: bash

   pytest tests/ -v
