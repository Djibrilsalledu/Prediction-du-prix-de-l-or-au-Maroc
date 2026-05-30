.. _preprocessing:

Preprocessing
=============

The preprocessing layer is responsible for:

1. Loading and merging the four raw CSV files.
2. Enforcing a clean monthly ``DatetimeIndex``.
3. Applying **fold-aware imputation** — no global statistics, no future-data leakage.

All modules live in ``src/preprocessing/``.

----

Data loading — ``data_loader.py``
----------------------------------

Entry points
~~~~~~~~~~~~

.. code-block:: python

   from src.preprocessing.data_loader import load_raw_data, load_and_prepare_data

   df = load_raw_data()          # preferred — returns raw merged frame, NaNs intact
   df = load_and_prepare_data()  # backward-compatible alias for load_raw_data()

``load_raw_data()`` performs the following steps in order:

1. **Load gold prices** (``_load_gold``) — reads ``data/gold_prices.csv``, renames columns,
   coerces to numeric, enforces monthly index.
2. **Load USD/MAD** (``_load_usd_mad``) — applies European-decimal parsing, enforces monthly
   index.
3. **Load events** (``_load_events``) — reads intensity columns; coerces to numeric.
4. **Inner-join** gold × FX × events on the monthly index.
5. **Compute target**: ``df["gold_price_mad"] = df["gold_price_usd"] * df["usd_mad"]``.
6. **Left-join macro** (``_load_macro_optional``) if ``data/macro_indicators.csv`` exists;
   missing macro columns silently skipped.

.. warning::

   ``load_raw_data()`` intentionally leaves ``NaN`` values in place.
   Do **not** apply ``fillna`` or ``interpolate`` on the merged frame.
   All imputation must happen inside ``FoldPreprocessor.fit()`` / ``transform()``
   to prevent look-ahead bias.

Cleaning — ``cleaners.py``
---------------------------

.. code-block:: python

   from src.preprocessing.cleaners import (
       parse_european_decimal,
       to_monthly_datetime,
       enforce_monthly_index,
   )

``parse_european_decimal(value)``
   Converts a string like ``"9,85"`` or ``"  9.85 "`` to a Python ``float``.
   Handles spaces, commas, non-numeric characters and ``NaN`` inputs.

``to_monthly_datetime(series)``
   Converts a ``pd.Series`` of date strings to a ``pd.DatetimeIndex`` normalised to
   month-start (``MS`` frequency) via ``pd.Period("M").to_timestamp()``.

``enforce_monthly_index(df, date_col)``
   * Parses the date column.
   * Sorts by date and drops duplicate months (keeps last observation).
   * Reindexes to a complete monthly range so that any missing months appear as ``NaN``
     rather than being silently absent.

----

Fold-aware imputation — ``fold_preprocessor.py``
-------------------------------------------------

The ``FoldPreprocessor`` is the single most important anti-leakage component in the
pipeline.  It is a stateful, scikit-learn-style transformer that is **fit on the training
fold only** and then applied separately to train and test.

.. code-block:: python

   from src.preprocessing.fold_preprocessor import FoldPreprocessor, prepare_fold_frames

Design rules
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Context
     - Rule
   * - **Training half**
     - Forward-fill within the training window, then fill any remaining ``NaN`` with the
       fold-level **median** of that column (event columns use the median of the
       forward-filled series; macro columns use the numeric median).
   * - **Test half**
     - Fill the *first* missing value with the **last observed value from training**
       (``last_train_value_``), then carry forward within the test window.
       Any remaining ``NaN`` is filled with the train-level fill value.
       **Back-filling from the future is never allowed.**
   * - **Events**
     - Use median of the forward-filled training series (not mean, to be robust to zeros).
   * - **Macro**
     - Use numeric median of training observations.

API
~~~

.. code-block:: python

   prep = FoldPreprocessor(fold_id=0)
   prep.fit(train_raw)                              # learns fill_values_, last_train_value_
   train_clean = prep.transform(train_raw, split_role="train")
   test_clean  = prep.transform(test_raw,  split_role="test")

   # Serialise for audit trail
   meta = prep.to_dict()   # stored in backtest_fold_metadata.json

``prepare_fold_frames(train_raw, test_raw, fold_id)``
   Convenience wrapper that:

   1. Instantiates and fits a ``FoldPreprocessor``.
   2. Transforms both halves.
   3. Concatenates them and runs ``build_features()`` so that lag columns at the start
      of the test window can look back into training history.
   4. Slices the featured frame back into clean train and test ``DataFrame`` objects.

   Returns ``(train_df, test_df, prep)``.

Leakage tests
~~~~~~~~~~~~~

Three unit tests validate the imputation guarantees:

.. code-block:: bash

   pytest tests/test_leakage_detection.py -v
   pytest tests/test_backtest_integrity.py -v

Key assertions:

* ``test_test_imputation_does_not_use_future_bfill`` — test-set ``NaN`` values must be
  filled with the last training value, not a future test observation.
* ``test_global_bfill_would_differ_from_fold_safe`` — demonstrates that naive global
  ``bfill`` produces different (leaky) results.
* ``test_imputer_uses_train_stats_only`` — verifies ``fill_values_`` are computed from
  training data only.
* ``test_no_target_leakage_in_features`` — target column and raw ``gold_price_usd`` /
  ``usd_mad`` do not appear in the ML feature set.

----

Fold structure — ``backtesting.py`` (split generation)
-------------------------------------------------------

Splits are generated by ``generate_backtest_splits()``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Parameter
     - Value (default)
   * - ``initial_train``
     - 180 months (15 years)
   * - ``test_horizon``
     - 12 months per fold
   * - ``step``
     - 12 months (annual walk-forward)
   * - ``strategy``
     - ``"expanding"`` — train window grows with each fold

With 307 observations and an initial train of 180, **10 folds** are produced:

.. list-table::
   :header-rows: 1
   :widths: 10 25 25 25 15

   * - Fold
     - Train start
     - Train end
     - Test period
     - N train
   * - 0
     - 2000-08
     - 2015-07
     - 2015-08 → 2016-07
     - 180
   * - 1
     - 2000-08
     - 2016-07
     - 2016-08 → 2017-07
     - 192
   * - 2
     - 2000-08
     - 2017-07
     - 2017-08 → 2018-07
     - 204
   * - 3
     - 2000-08
     - 2018-07
     - 2018-08 → 2019-07
     - 216
   * - 4
     - 2000-08
     - 2019-07
     - 2019-08 → 2020-07
     - 228
   * - 5
     - 2000-08
     - 2020-07
     - 2020-08 → 2021-07
     - 240
   * - 6
     - 2000-08
     - 2021-07
     - 2021-08 → 2022-07
     - 252
   * - 7
     - 2000-08
     - 2022-07
     - 2022-08 → 2023-07
     - 264
   * - 8
     - 2000-08
     - 2023-07
     - 2023-08 → 2024-07
     - 276
   * - 9
     - 2000-08
     - 2024-07
     - 2024-08 → 2025-07
     - 288

Fold metadata (train/test dates, imputation fill values, last training values) is
persisted to ``results/models/backtest_fold_metadata.json`` for full reproducibility.
