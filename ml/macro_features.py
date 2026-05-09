"""
macro_features.py — Macro feature pipeline for B-2 (Macro-augmented LightGBM).

Fetches a small set of free macro indicators from yfinance and converts them
to monthly features suitable for sector-rotation prediction.

Features (all monthly, lagged by 1 month at training time to avoid look-ahead):
    vix              — CBOE VIX (^VIX) close, % units
    vix_chg21        — 1-month change in VIX (%)
    yield_10y        — 10-year Treasury yield (^TNX, % units)
    yield_curve      — 10y minus 13-week T-bill (^TNX − ^IRX), % units
    credit_proxy     — HYG return minus TLT return (1-month), % units
    dxy              — US dollar index (DX-Y.NYB) level
    dxy_chg21        — 1-month % change in DXY

Output:
    DataFrame indexed by month-end with the columns above.
    Missing values forward-filled within reason; rows with all-NaN dropped.

Usage:
    from macro_features import fetch_macro_monthly
    macro = fetch_macro_monthly(lookback_years=28)
    # macro.loc["2024-01-31"]  →  Series of 7 features
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

# Free yfinance tickers covering risk / rates / credit / FX
MACRO_TICKERS = {
    "vix":   "^VIX",         # volatility regime
    "tnx":   "^TNX",         # 10-year Treasury yield (× 10)
    "irx":   "^IRX",         # 13-week T-bill yield (× 10)
    "hyg":   "HYG",          # high-yield credit (proxy for credit spread)
    "tlt":   "TLT",          # long Treasury (paired with HYG)
    "dxy":   "DX-Y.NYB",     # US dollar index
}

# In-memory cache (process-lifetime). Keyed by lookback period.
_CACHE: dict = {}


def _fetch_close(ticker: str, start: datetime, end: datetime, interval: str) -> pd.Series:
    """Single-ticker close-price fetch returning a tidy Series."""
    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False, threads=False)
    if df.empty:
        return pd.Series(dtype=float, name=ticker)
    if isinstance(df.columns, pd.MultiIndex):
        # When yf returns multi-column for single-ticker (rare), pick first
        s = df["Close"] if "Close" in df.columns.get_level_values(0) else df.iloc[:, 0]
    else:
        s = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = ticker
    return s.dropna()


def fetch_macro_monthly(lookback_years: int = 28) -> pd.DataFrame:
    """Fetch & build monthly macro feature matrix.

    Caches per (lookback_years) within the process so repeated backtests
    don't re-hit yfinance.
    """
    if lookback_years in _CACHE:
        return _CACHE[lookback_years].copy()

    end = datetime.now()
    if lookback_years >= 99:
        start = end - timedelta(days=365 * 30)  # 30 years max
    else:
        start = end - timedelta(days=int((lookback_years + 2) * 365.25))

    closes: dict[str, pd.Series] = {}
    for label, ticker in MACRO_TICKERS.items():
        try:
            s = _fetch_close(ticker, start, end, interval="1mo")
            closes[label] = s
        except Exception:
            closes[label] = pd.Series(dtype=float)

    if not any(len(s) > 0 for s in closes.values()):
        return pd.DataFrame()

    # Align to monthly index
    df = pd.DataFrame()
    for label, s in closes.items():
        if len(s) == 0:
            continue
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        df[label] = s
    if df.empty:
        return df

    # Resample to month-end
    df = df.resample("ME").last().ffill(limit=1)

    out = pd.DataFrame(index=df.index)
    if "vix" in df.columns:
        out["vix"] = df["vix"]
        out["vix_chg21"] = df["vix"].pct_change() * 100
    if "tnx" in df.columns:
        # ^TNX is yield × 10 (e.g., 4.5% shows as 45.0). Normalize to %.
        out["yield_10y"] = df["tnx"] / 10.0
    if "tnx" in df.columns and "irx" in df.columns:
        out["yield_curve"] = (df["tnx"] - df["irx"]) / 10.0
    if "hyg" in df.columns and "tlt" in df.columns:
        # Monthly return difference HYG − TLT as credit-spread direction proxy
        hyg_ret = df["hyg"].pct_change() * 100
        tlt_ret = df["tlt"].pct_change() * 100
        out["credit_proxy"] = hyg_ret - tlt_ret
    if "dxy" in df.columns:
        out["dxy"] = df["dxy"]
        out["dxy_chg21"] = df["dxy"].pct_change() * 100

    out = out.dropna(how="all")
    _CACHE[lookback_years] = out.copy()
    return out


if __name__ == "__main__":
    df = fetch_macro_monthly(lookback_years=28)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index.min()} → {df.index.max()}")
    print(f"\nMost recent 3 rows:")
    print(df.tail(3).round(3))
    print(f"\nMissing per column:")
    print(df.isna().sum())
