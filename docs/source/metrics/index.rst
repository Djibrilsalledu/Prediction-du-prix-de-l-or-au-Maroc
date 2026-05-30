.. _metrics:

Metrics Reference
=================

All evaluation metrics are implemented in ``src/evaluation/metrics.py``.

Alignment
---------

Before computing any metric, ``_align(y_true, y_pred)`` intersects the two series on
their index, converts to ``float64``, and drops any row where either value is ``NaN``.
This prevents silent errors from index mismatches between fold outputs.

----

Core metrics
------------

RMSE — Root Mean Squared Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}

Penalises large errors quadratically.  The primary ranking metric for model selection
because it is sensitive to the occasional large miss that matters most in practice.
Expressed in MAD per troy ounce.

MAE — Mean Absolute Error
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|

More robust to outliers than RMSE.  Expressed in MAD.

MAPE — Mean Absolute Percentage Error
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{MAPE} = \frac{100}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right|

Scale-free; useful for comparing across periods with different price levels.
The best model (SARIMAX) achieves **6.49 %** mean MAPE, meaning typical monthly
forecast errors are about 6.5 % of the actual price.

R² — Coefficient of Determination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   R^2 = 1 - \frac{\sum (y_t - \hat{y}_t)^2}{\sum (y_t - \bar{y})^2}

.. warning::

   For rolling backtest evaluation, R² is computed relative to the **OOS mean** (the
   mean of the test-fold actuals), not the global mean.  Because the gold price trends
   strongly upward, the OOS mean is always far from the actual values, yielding **negative
   R²** for all models even when forecasts are accurate.

   The **bootstrap R²** (computed on pooled OOS predictions) is the more meaningful
   measure: SARIMAX achieves 0.859 [0.827 – 0.892].

Bias
~~~~~

.. math::

   \text{Bias} = \frac{1}{n} \sum_{t=1}^{n} (\hat{y}_t - y_t)

Signed average prediction error.  Positive = systematic over-forecast;
negative = under-forecast.  All models exhibit negative bias (−772 to −1 573 MAD),
reflecting the difficulty of capturing the 2024–2025 bull run momentum.

Directional Accuracy
~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{DA} = \frac{100}{n-1} \sum_{t=2}^{n}
   \mathbf{1}\!\left[\text{sign}(y_t - y_{t-1}) = \text{sign}(\hat{y}_t - \hat{y}_{t-1})\right]

Percentage of months where the model correctly predicts the direction of price movement
(up or down).  A random forecast scores ~50 %.

SARIMAX and XGBoost both achieve **60.9 %** directional accuracy — roughly 11 percentage
points above random.

----

Bootstrap confidence intervals
--------------------------------

Implemented in ``src/evaluation/bootstrap.py`` using 500 resampling iterations with
``np.random.default_rng(42)`` for reproducibility.

.. code-block:: python

   from src.evaluation.bootstrap import bootstrap_metric_ci, bootstrap_all_models

   ci = bootstrap_metric_ci(y_true, y_pred, metric="rmse", n_samples=500)
   # Returns {"point": ..., "low": ..., "high": ...}

   df = bootstrap_all_models(y_true, predictions_dict)
   # Returns a long-format DataFrame with columns: model, metric, point, ci_low, ci_high

----

Naive baselines
---------------

Three naive baselines are included in every backtest fold to contextualise model
performance (``src/evaluation/baselines.py``):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Baseline
     - Description
   * - ``Baseline_LastValue``
     - Repeats the last observed training value for all *h* steps.
       Achieves 0 % directional accuracy by construction.
   * - ``Baseline_TrainMean``
     - Uses the training-fold mean for all *h* steps.
       Worst overall model (MAPE 44.8 %).
   * - ``Baseline_Drift``
     - Projects a constant linear trend from the first to the last training value.
       Competitive with SARIMAX on RMSE (1 487) and directional accuracy (60.9 %) —
       highlighting how much of gold's forecastability comes from its strong trend.

----

Pooled vs fold-averaged metrics
---------------------------------

Two aggregation modes are supported:

**Fold-averaged** (``model_ranking_backtest.csv``):
   Compute the metric independently for each fold's 12-month window, then take the
   arithmetic mean and standard deviation across folds.
   Robust to fold-level heteroskedasticity; the primary ranking metric.

**Pooled OOS** (``model_ranking_pooled_oos.csv``):
   Concatenate all 10 × 12 = 120 test observations into a single series and compute
   the metric once.  More sensitive to recent high-volatility folds (which dominate the
   pooled residuals) but provides a single-number performance summary.
