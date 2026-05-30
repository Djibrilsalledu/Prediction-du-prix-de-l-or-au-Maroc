.. _datasets:

Datasets
========

All four datasets share a **monthly frequency** with date stamps normalised to
month-start (``MS``).  The common index runs from **August 2000 to February 2026**
(307 observations).  No global imputation is applied at load time — missing values
remain ``NaN`` until fold-level preprocessing.

.. note::

   The data-loading entry point is ``src/preprocessing/data_loader.py``.
   Call ``load_raw_data()`` to obtain the merged ``DataFrame`` with the target column.

----

gold_prices.csv
---------------

**International gold spot price in USD per troy ounce.**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``date``
     - ``datetime``
     - Month-start timestamp (e.g. ``2000-08-01``)
   * - ``gold_price_usd``
     - ``float``
     - Gold price in USD per troy ounce (XAU/USD)

**Source:** World Gold Council / Yahoo Finance (``GC=F``).

**Key characteristics:**

* Strong upward trend from ~$270 (Aug 2000) to ~$2 900 (Feb 2026).
* 57 structural changepoints detected by the rolling z-score regime detector.
* Non-stationary; ADF statistic = +2.30, p-value = 0.999 → differencing required (d = 1).

----

usd_mad.csv
-----------

**USD / Moroccan Dirham exchange rate.**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``date``
     - ``datetime``
     - Month-start timestamp
   * - ``usd_mad``
     - ``float``
     - Monthly average USD/MAD rate (number of MAD per 1 USD)

**Source:** Bank Al-Maghrib (BAM) / World Bank.

**Parsing note:** The raw file may use European decimal notation (comma separator).
The loader applies ``parse_european_decimal()`` from ``src/preprocessing/cleaners.py``.

**Role in the pipeline:**

The target variable is defined as:

.. math::

   \text{gold\_price\_mad} = \text{gold\_price\_usd} \times \text{usd\_mad}

``usd_mad`` is also used directly as an exogenous regressor in SARIMAX, with a
SARIMAX coefficient of **+1 048** — the largest exogenous effect in the model.

----

moroccan_events.csv
-------------------

**Monthly intensity of Moroccan socio-cultural events (scale 0 – 1).**

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Column
     - Range
     - Description
   * - ``date``
     - —
     - Month-start timestamp
   * - ``ramadan``
     - [0, 1]
     - Fractional overlap of the Ramadan lunar month with the calendar month.
       Jewellery purchases peak before Aïd Al-Fitr; SARIMAX coefficient = **+70.7**.
   * - ``eid_alfitr``
     - [0, 1]
     - Aïd Al-Fitr intensity.  Concentration of gift-giving drives short gold-demand
       spikes; coefficient = **+22.0**.
   * - ``eid_aladha``
     - [0, 1]
     - Aïd Al-Adha intensity.  Spending redirected toward livestock; weaker gold impact
       (coefficient = **+48.1** for ``wedding_season``).
   * - ``wedding_season``
     - [0, 1]
     - April–September Moroccan wedding season.  Bridal gold jewellery constitutes a
       significant fraction of the marriage gift (*mahr*).
   * - ``mre_season``
     - [0, 1]
     - Return season of Moroccan Residents Abroad (June–August). Cash remittances
       converted to MAD boost domestic gold demand.

**Source:** Islamic calendar computation + local market surveys.

**Usage in future forecasts:**

Event columns are forward-projected using a *seasonal profile* approach
(``src/feature_engineering/exogenous_future.py``):

* Compute the historical average intensity per calendar month (month-of-year mean).
* Blend 70 % long-run average with 30 % last-year observed pattern.
* Clip to [0, 1].

----

macro_indicators.csv
--------------------

**Optional macroeconomic regressors — built on demand.**

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Column
     - Unit
     - Description
   * - ``oil_brent_usd``
     - USD/bbl
     - Brent crude monthly average.  Positive co-movement with gold as a commodity.
       SARIMAX coefficient = **−17.7** (substitution / risk-on effect).
   * - ``dxy_index``
     - Index
     - US Dollar Index (DXY).  Strong negative correlation with USD gold price
       (−0.75 in the correlation matrix); SARIMAX coefficient = **−79.2**.
   * - ``fed_funds_rate``
     - %
     - Effective Federal Funds Rate.  Higher US rates increase the opportunity cost of
       holding gold; coefficient = **−0.29**.
   * - ``inflation_morocco``
     - % YoY
     - Moroccan CPI inflation.  Used as a demand-side pressure indicator.
   * - ``policy_rate_bam``
     - %
     - Bank Al-Maghrib policy rate.  Endogenous response to inflation; correlated with
       ``inflation_morocco`` (see correlation matrix).

**Building the macro file:**

.. code-block:: bash

   python main.py --build-macro

This runs ``src/preprocessing/build_macro.py``, which fetches Brent and DXY from
``yfinance`` and constructs proxy series for Fed Funds, Moroccan inflation, and BAM rate.
If ``yfinance`` is unavailable, all macro columns are filled with ``NaN`` and the pipeline
gracefully falls back to event-only exogenous regressors.

**Missing-value treatment:**

Macro columns are **not** globally imputed.  Missing values at fold boundaries are handled
by ``FoldPreprocessor`` (see :ref:`preprocessing`):

* *Training half*: forward-fill within fold, then fill remaining gaps with the
  fold-level median.
* *Test half*: carry-forward from the last observed training value; never back-fill
  from future test observations.

----

Derived target column
---------------------

After merging the four sources, the loader computes:

.. code-block:: python

   df["gold_price_mad"] = df["gold_price_usd"] * df["usd_mad"]

This column is the **sole forecasting target**.  All features and model outputs are expressed
in MAD per troy ounce.

Summary statistics (full sample)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 18 16

   * - Series
     - Min
     - Mean
     - Max
     - Obs.
   * - ``gold_price_usd``
     - ~268
     - ~1 100
     - ~2 940
     - 307
   * - ``usd_mad``
     - ~8.5
     - ~9.6
     - ~10.8
     - 307
   * - ``gold_price_mad``
     - ~2 300
     - ~10 800
     - ~43 200
     - 307
   * - ``oil_brent_usd``
     - ~18
     - ~76
     - ~130
     - 307
   * - ``dxy_index``
     - ~72
     - ~91
     - ~115
     - 307
