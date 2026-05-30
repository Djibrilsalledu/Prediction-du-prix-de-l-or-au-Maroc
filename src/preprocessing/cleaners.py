"""Parsing and cleaning raw monthly series."""
import re

import pandas as pd


def parse_european_decimal(value) -> float:
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    return float(s) if s else float("nan")


def to_monthly_datetime(series: pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(series, errors="coerce")
    return pd.DatetimeIndex(dt).to_period("M").to_timestamp()


def enforce_monthly_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = to_monthly_datetime(out[date_col])
    out = out.sort_values(date_col).drop_duplicates(date_col, keep="last")
    out = out.set_index(date_col)
    full_range = pd.date_range(out.index.min(), out.index.max(), freq="MS")
    out = out.reindex(full_range)
    return out
