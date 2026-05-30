"""Load and merge raw monthly datasets — NO global imputation."""
from __future__ import annotations

import pandas as pd

from src.preprocessing.cleaners import enforce_monthly_index, parse_european_decimal
from src.utils.config import (
    DATE_COLUMN,
    EVENT_COLUMNS,
    EVENTS_FILE,
    GOLD_FILE,
    GOLD_USD_COL,
    MACRO_COLUMNS,
    MACRO_FILE,
    TARGET_COL,
    USD_MAD_COL,
    USD_MAD_FILE,
)


def _load_gold() -> pd.DataFrame:
    df = pd.read_csv(GOLD_FILE)
    date_col = "Date" if "Date" in df.columns else DATE_COLUMN
    df = df.rename(columns={date_col: DATE_COLUMN, df.columns[-1]: GOLD_USD_COL})
    df[GOLD_USD_COL] = pd.to_numeric(df[GOLD_USD_COL], errors="coerce")
    return enforce_monthly_index(df[[DATE_COLUMN, GOLD_USD_COL]], DATE_COLUMN)


def _load_usd_mad() -> pd.DataFrame:
    df = pd.read_csv(USD_MAD_FILE)
    df[DATE_COLUMN] = df.iloc[:, 0]
    rate_col = [c for c in df.columns if c.lower() != DATE_COLUMN][0]
    df[USD_MAD_COL] = df[rate_col].map(parse_european_decimal)
    return enforce_monthly_index(df[[DATE_COLUMN, USD_MAD_COL]], DATE_COLUMN)


def _load_events() -> pd.DataFrame:
    df = pd.read_csv(EVENTS_FILE)
    df[DATE_COLUMN] = df.iloc[:, 0]
    for col in EVENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return enforce_monthly_index(df[[DATE_COLUMN] + EVENT_COLUMNS], DATE_COLUMN)


def _load_macro_optional() -> pd.DataFrame | None:
    if not MACRO_FILE.exists():
        return None
    df = pd.read_csv(MACRO_FILE)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    cols = [c for c in MACRO_COLUMNS if c in df.columns]
    if not cols:
        return None
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return enforce_monthly_index(df[[DATE_COLUMN] + cols], DATE_COLUMN)


def load_raw_data() -> pd.DataFrame:
    """
    Merge mandatory series and optional macro WITHOUT imputation.

    Missing values remain NaN until fold-level preprocessing.
  Target: gold_price_mad = gold_price_usd * usd_mad (where both are observed).
    """
    gold = _load_gold()
    fx = _load_usd_mad()
    events = _load_events()

    merged = gold.join(fx, how="inner").join(events, how="inner")
    merged[TARGET_COL] = merged[GOLD_USD_COL] * merged[USD_MAD_COL]

    macro = _load_macro_optional()
    if macro is not None:
        merged = merged.join(macro, how="left")

    merged.index.name = DATE_COLUMN
    return merged.sort_index()


def load_and_prepare_data() -> pd.DataFrame:
    """Backward-compatible alias: returns raw merged data (no global fill)."""
    return load_raw_data()
