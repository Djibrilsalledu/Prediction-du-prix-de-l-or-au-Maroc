.. _feature_engineering:

Feature Engineering
===================

All feature-engineering code lives in ``src/feature_engineering/``.
The two key modules are ``features.py`` (in-sample features) and
``exogenous_future.py`` (out-of-sample exogenous construction).

The cardinal rule is **no look-ahead**: every feature computed at time *t* uses only
information available up to and including *t − 1*.

----

In-sample features — ``features.py``
--------------------------------------

Entry point
~~~~~~~~~~~

.. code-block:: python

   from src.feature_engineering.features import (
       build_features,
       get_ml_feature_columns,
       training_row_mask,
   )

   featured_df = build_features(df)         # adds ~85 columns
   feat_cols   = get_ml_feature_columns(featured_df)   # list of ML-safe columns
   mask        = training_row_mask(featured_df, feat_cols)  # warmup filter

``build_features(df)``
~~~~~~~~~~~~~~~~~~~~~~

Builds the full feature matrix from a preprocessed ``DataFrame``.  All operations shift
the source series by at least 1 step before computing derived columns.

**Target lags**

.. code-block:: text

   gold_price_mad_lag_1, _lag_3, _lag_6, _lag_12

**Rolling statistics** (applied to ``gold_price_mad.shift(1)``):

.. code-block:: text

   gold_price_mad_roll_mean_3,  _roll_std_3
   gold_price_mad_roll_mean_6,  _roll_std_6
   gold_price_mad_roll_mean_12, _roll_std_12

**Momentum and returns**:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Feature
     - Formula
   * - ``gold_price_mad_pct_change_1``
     - ``y_{t-1} / y_{t-2} - 1``
   * - ``gold_price_mad_pct_change_12``
     - ``y_{t-1} / y_{t-13} - 1``
   * - ``gold_price_mad_momentum_3``
     - ``y_{t-1} - y_{t-4}``
   * - ``gold_price_mad_momentum_12``
     - ``y_{t-1} - y_{t-13}``
   * - ``gold_price_mad_roll_vol_6``
     - Rolling std of monthly returns, window = 6
   * - ``gold_price_mad_roll_vol_12``
     - Rolling std of monthly returns, window = 12
   * - ``gold_price_mad_trend_strength_12``
     - ``|y_{t-1} - y_{t-13}| / y_{t-13}``
   * - ``gold_price_mad_roll_return_3``
     - 3-month lagged percentage return

**Cyclical time encoding** (month-of-year, no ordinal bias):

.. code-block:: text

   month_sin = sin(2π × month / 12)
   month_cos = cos(2π × month / 12)
   quarter, year

**Moroccan event features** (per event: ``ramadan``, ``eid_alfitr``, ``eid_aladha``,
``wedding_season``, ``mre_season``):

.. code-block:: text

   <event>_lag_1, _lag_3
   <event>_roll_mean_3, _roll_mean_12
   <event>_cumulative_12

**FX and USD gold lags / interactions**:

.. code-block:: text

   gold_price_usd_lag_1, gold_price_usd_lag1
   usd_mad_lag_1, usd_mad_lag1
   gold_price_usd_lag1_x_usd_mad_lag1        ← interaction term
   usd_mad_pct_change_1
   usd_mad_roll_mean_3 / _6 / _12
   usd_mad_roll_std_3  / _6 / _12

**Macro interaction terms** (per macro variable ``m``):

.. code-block:: text

   <m>_pct_change_1
   <m>_roll_mean_6
   <m>_roll_vol_6
   gold_price_usd_lag1_x_<m>_lag1

This yields **~85 features** in total (exact count varies with macro column availability).

``get_ml_feature_columns(df, frozen=None)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns the list of columns safe for ML models by excluding:

* ``gold_price_mad`` (target — would be leakage)
* ``gold_price_usd``, ``usd_mad`` (contemporaneous raw predictors — leakage)
* Any non-numeric or ``Unnamed`` columns

Optionally accepts a ``frozen`` list (the feature set from the final full-sample fit)
to guarantee identical column ordering at inference time.

``training_row_mask(feature_df, feature_cols)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Returns a boolean ``Series`` that is ``True`` only for rows where:

1. The row index is beyond the first ``MAX_LAG = 12`` rows (warmup period).
2. All feature columns are non-``NaN``.

This prevents gradient-boosting models from training on partially populated lag windows.

----

Future exogenous construction — ``exogenous_future.py``
---------------------------------------------------------

When generating forecasts beyond the last observed date, exogenous columns must be
projected forward.  The module ``src/feature_engineering/exogenous_future.py`` provides
two strategies:

Entry point
~~~~~~~~~~~

.. code-block:: python

   from src.feature_engineering.exogenous_future import build_future_exogenous

   future_exog = build_future_exogenous(
       history,
       last_observed=df.index.max(),
       forecast_end="2027-12-01",
       exog_cols=["usd_mad", "ramadan", ...],
       allow_extrapolation=False,   # STRICT_EVALUATION_MODE
   )

Event projection (``extend_events``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Moroccan event columns are **calendar-deterministic**: their future values can be
approximated from the Islamic calendar.  The blending formula is:

.. math::

   \hat{e}_{t} = 0.7 \times \bar{e}_{\text{month}} + 0.3 \times e_{\text{last year, month}}

where :math:`\bar{e}_{\text{month}}` is the historical average intensity for that
calendar month.  The result is clipped to [0, 1].

FX & macro projection (``extend_numeric_exog``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ``usd_mad``, ``oil_brent_usd``, ``dxy_index``, etc., a **damped linear trend** is used:

.. math::

   \hat{x}_{t+h} = \phi^h \cdot \hat{x}^{\text{linear}}_{t+h} + (1 - \phi^h) \cdot x_t

with :math:`\phi = 0.95` and a ±12 % cap around the last observed value.

.. warning::

   FX/macro extrapolation is **disabled by default** when
   ``STRICT_EVALUATION_MODE = True`` (``src/utils/config.py``).
   In strict mode, FX and macro columns are held at their last observed training value
   throughout the forecast horizon.  This avoids unverified assumptions about future
   macroeconomic conditions during backtesting.

   Set ``ALLOW_EXOG_EXTRAPOLATION = True`` (or pass ``allow_extrapolation=True``) only
   for the final production forecast.

Configuration
~~~~~~~~~~~~~

All feature-engineering parameters are centralised in ``src/utils/config.py``:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Constant
     - Default
     - Description
   * - ``MAX_LAG``
     - ``12``
     - Minimum warmup rows for training mask
   * - ``STRICT_EVALUATION_MODE``
     - ``True``
     - Disables FX/macro extrapolation during backtesting
   * - ``ALLOW_EXOG_EXTRAPOLATION``
     - ``False``
     - Derived from ``not STRICT_EVALUATION_MODE``
   * - ``EVENT_COLUMNS``
     - (list)
     - Names of the five Moroccan event columns
   * - ``MACRO_COLUMNS``
     - (list)
     - Names of the five macro indicator columns
