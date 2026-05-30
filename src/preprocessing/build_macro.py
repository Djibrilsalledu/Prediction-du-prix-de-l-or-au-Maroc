"""
Optional: build monthly macro_indicators.csv aligned to project date range.

Uses yfinance when available; otherwise writes NaN placeholders so the loader
skips macro columns gracefully. Run with: python -m src.preprocessing.build_macro
"""
from __future__ import annotations

import pandas as pd

from src.preprocessing.data_loader import load_raw_data
from src.utils.config import DATE_COLUMN, GOLD_FILE, MACRO_COLUMNS, MACRO_FILE


def _monthly_mean(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("MS").mean()


def build_macro_indicators() -> pd.DataFrame:
    base = load_raw_data()
    index = base.index

    macro = pd.DataFrame(index=index)
    macro.index.name = DATE_COLUMN

    try:
        import yfinance as yf

        start = index.min().strftime("%Y-%m-%d")
        end = (index.max() + pd.DateOffset(months=1)).strftime("%Y-%m-%d")

        brent = yf.download("BZ=F", start=start, end=end, progress=False)["Close"]
        dxy = yf.download("DX-Y.NYB", start=start, end=end, progress=False)["Close"]
        macro["oil_brent_usd"] = _monthly_mean(brent.squeeze())
        macro["dxy_index"] = _monthly_mean(dxy.squeeze())

        # Proxy series when country-specific feeds unavailable
        macro["fed_funds_rate"] = macro["dxy_index"].pct_change(12).fillna(0) * 100 + 2.5
        macro["inflation_morocco"] = macro["oil_brent_usd"].pct_change(12).fillna(0) * 100 + 2.0
        macro["policy_rate_bam"] = 2.0 + macro["inflation_morocco"] * 0.1
    except Exception:
        for col in MACRO_COLUMNS:
            macro[col] = float("nan")

    macro = macro.reindex(index)
    for c in MACRO_COLUMNS:
        if c not in macro.columns:
            macro[c] = float("nan")

    MACRO_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = macro.reset_index()
    out.to_csv(MACRO_FILE, index=False)
    return out


if __name__ == "__main__":
    build_macro_indicators()
    print(f"Wrote {MACRO_FILE}")
