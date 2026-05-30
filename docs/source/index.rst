.. Moroccan Gold Price Forecasting documentation master file

Moroccan Gold Price Forecasting
================================

.. image:: https://img.shields.io/badge/python-3.10%2B-blue
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/license-Academic-green
   :alt: Academic License

.. image:: https://img.shields.io/badge/status-Research-orange
   :alt: Research

**Academic-grade monthly time series forecasting system for Moroccan gold prices (MAD).**

This project combines classical econometrics, modern machine learning and deep learning to
forecast the price of gold in Moroccan Dirhams (MAD) at monthly frequency from August 2000
through December 2027.  The pipeline is fully reproducible, strictly prevents data leakage,
and produces static figures, tables, and CSV forecasts — no dashboard or web UI.

.. note::

   Author: **Djibril SALL** — École Nationale Supérieure d'Arts et Métiers (ENSAM)

   Supervisor: **Pr. Tawfik Masrour**

----

Quick start
-----------

.. code-block:: bash

   # Clone and install
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # source .venv/bin/activate  # Linux / macOS
   pip install -r requirements.txt

   # Run the full pipeline
   python main.py

   # Optional: download macro indicators via yfinance
   python main.py --build-macro

   # Skip backtesting, regenerate forecasts only
   python main.py --forecast-only

After a successful run, results are written to:

* ``results/figures/``  — EDA, diagnostics, actual-vs-predicted, residuals
* ``results/tables/``   — model rankings, bootstrap CIs, fold stability
* ``results/forecasts/`` — future forecasts to 2027-12 (CSV)
* ``results/models/``   — pipeline metadata JSON, analysis report

----

.. toctree::
   :maxdepth: 2
   :caption: Project overview

   presentation

.. toctree::
   :maxdepth: 2
   :caption: User guide

   datasets
   preprocessing
   feature_engineering
   forecasting
   results

.. toctree::
   :maxdepth: 2
   :caption: Reference

   metrics/index
   models/index
   conclusion

----

Key results at a glance
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 22 15 15 12 16 15

   * - Model
     - RMSE mean
     - MAE mean
     - MAPE %
     - Dir. Acc. %
     - Bias mean
   * - **SARIMAX ★**
     - **1 478**
     - **1 230**
     - **6.49**
     - 60.9
     - −797
   * - ARIMA
     - 1 536
     - 1 296
     - 6.92
     - 55.5
     - −865
   * - Hybrid ARIMA+XGB
     - 1 544
     - 1 319
     - 7.26
     - 58.2
     - −772
   * - SARIMA
     - 1 616
     - 1 363
     - 7.42
     - 54.5
     - −927
   * - XGBoost
     - 1 759
     - 1 525
     - 7.92
     - 60.9
     - −1 438
   * - LSTM (univariate)
     - 1 979
     - 1 741
     - 9.10
     - 47.3
     - −1 573
   * - Prophet
     - 2 406
     - 2 026
     - 12.30
     - 46.4
     - −1 100

*Metrics averaged over 10 expanding-window folds (2015–2025, 12-month test horizon each).*

**Recommended production model:** ``Ensemble_Weighted``
(inverse-RMSE blend of SARIMAX + XGBoost + Hybrid ARIMA-XGBoost).
