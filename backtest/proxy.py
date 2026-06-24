# -*- coding: utf-8 -*-
"""proxy.py — deterministic PM-proxy ticker selector.

For each historical date D, computes a simplified quant score from price data
alone (no LLM) and selects top-20 LONG + top-20 SHORT per asset type.

Score components (cross-sectional percentile rank):
  - mom_21d            : 21-day price return (momentum)
  - mom_63d            : 63-day price return (longer momentum)
  - trend_strength     : (close - SMA50) / SMA50  → position above/below trend
  - rs_21d             : relative strength vs SPY (21d ticker return - 21d SPY return)
  - vol_adj_mom        : mom_21d / 21d realized vol (Sharpe-style)

Composite proxy_score = weighted avg of the 5 percentile ranks (50/50 long/short
context). Higher score = stronger LONG candidate; lower = stronger SHORT candidate.

Bullish classification proxy: close > SMA50 AND mom_21d > 0
Bearish classification proxy: close < SMA50 AND mom_21d < 0

Sector concentration cap: max 5 per GICS sector per bucket (mirrors PM agent).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd


WEIGHTS = {
    "mom_21d":         0.25,
    "mom_63d":         0.20,
    "trend_strength":  0.25,
    "rs_21d":          0.15,
    "vol_adj_mom":     0.15,
}


def _pct_rank(values: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank (0-100), ignores NaN."""
    return values.rank(pct=True) * 100


def compute_proxy_score_one_date(
    data: dict[str, pd.DataFrame],
    universe: list[dict],
    as_of: pd.Timestamp,
    spy_ticker: str = "SPY",
    min_history_days: int = 120,
    max_stale_days: int = 7,
) -> Optional[pd.DataFrame]:
    """Compute proxy score for every ticker as of `as_of`.

    Bias mitigations (Fix A):
      - min_history_days (default 120): exclude recent IPOs without enough history
      - max_stale_days (default 7): exclude tickers where the latest close
        is > N days old (likely delisted / data stopped)
    """
    # Get SPY 21d return as benchmark
    if spy_ticker not in data:
        return None
    spy = data[spy_ticker]
    spy = spy[spy.index <= as_of]
    if len(spy) < 22:
        return None
    spy_21d = (spy["Close"].iloc[-1] / spy["Close"].iloc[-22]) - 1

    stale_cutoff = as_of - pd.Timedelta(days=max_stale_days)

    rows = []
    skipped_short_history = 0
    skipped_stale = 0
    for u in universe:
        t = u["ticker"]
        df = data.get(t)
        if df is None or df.empty:
            continue
        sub = df[df.index <= as_of]
        if len(sub) < 64:   # need ≥ 63 trading days for mom_63d + SMA50
            skipped_short_history += 1
            continue
        # Fix A: enforce minimum history (IPO filter — fresh listings noisy)
        first_date = df.index[0]
        history_days = (as_of - first_date).days
        if history_days < min_history_days:
            skipped_short_history += 1
            continue
        # Fix A: skip stale data (last close older than max_stale_days = likely delisted)
        last_date = sub.index[-1]
        if last_date < stale_cutoff:
            skipped_stale += 1
            continue
        close = sub["Close"].iloc[-1]
        if pd.isna(close) or close <= 0:
            continue

        mom_21d = (close / sub["Close"].iloc[-22]) - 1
        mom_63d = (close / sub["Close"].iloc[-64]) - 1
        sma50   = sub["Close"].iloc[-50:].mean()
        trend_strength = (close - sma50) / sma50 if sma50 > 0 else 0
        rs_21d  = mom_21d - spy_21d
        rets_21 = sub["Close"].pct_change().iloc[-21:].dropna()
        vol = rets_21.std() if len(rets_21) >= 5 else np.nan
        vol_adj_mom = mom_21d / vol if (vol and vol > 0) else 0

        rows.append({
            "ticker": t, "name": u["name"], "sector": u["sector"],
            "asset_type": u["asset_type"],
            "close": close,
            "mom_21d": mom_21d, "mom_63d": mom_63d,
            "trend_strength": trend_strength,
            "rs_21d": rs_21d, "vol_adj_mom": vol_adj_mom,
            "_sma50": sma50,
        })

    if not rows:
        return None
    df = pd.DataFrame(rows)

    # Cross-sectional percentile rank for each component
    for col in WEIGHTS:
        df[f"{col}_rank"] = _pct_rank(df[col])

    # Weighted composite
    df["proxy_score"] = sum(df[f"{col}_rank"] * w for col, w in WEIGHTS.items())

    # Classification proxy
    def _cls(row):
        if row["close"] > row["_sma50"] and row["mom_21d"] > 0:
            return "LONG_OK"
        if row["close"] < row["_sma50"] and row["mom_21d"] < 0:
            return "SHORT_OK"
        return "NEUTRAL"
    df["classification"] = df.apply(_cls, axis=1)
    return df


def select_picks_one_date(
    scored_df: pd.DataFrame,
    top_n: int = 20,
    max_per_sector: int = 5,
) -> dict[str, list[dict]]:
    """From a scored DataFrame, return 4 buckets of top-N picks with sector cap."""

    def _bucket(rows: pd.DataFrame, side: str) -> list[dict]:
        # Side determines sort direction: long=desc by score, short=asc
        rows = rows.sort_values("proxy_score", ascending=(side == "short"))
        selected: list[dict] = []
        sector_count: Counter = Counter()
        for _, r in rows.iterrows():
            if sector_count[r["sector"]] >= max_per_sector:
                continue
            selected.append({
                "ticker": r["ticker"], "name": r["name"], "sector": r["sector"],
                "asset_type": r["asset_type"], "side": side,
                "proxy_score": round(float(r["proxy_score"]), 2),
                "mom_21d": round(float(r["mom_21d"]) * 100, 2),
                "mom_63d": round(float(r["mom_63d"]) * 100, 2),
                "trend_strength": round(float(r["trend_strength"]) * 100, 2),
                "rs_21d": round(float(r["rs_21d"]) * 100, 2),
                "classification": r["classification"],
                "close": round(float(r["close"]), 4),
                "rank": len(selected) + 1,
            })
            sector_count[r["sector"]] += 1
            if len(selected) >= top_n:
                break
        return selected

    long_pool  = scored_df[scored_df["classification"] == "LONG_OK"]
    short_pool = scored_df[scored_df["classification"] == "SHORT_OK"]
    stocks_long  = long_pool[long_pool["asset_type"]  == "Stock"]
    etfs_long    = long_pool[long_pool["asset_type"]  == "ETF"]
    stocks_short = short_pool[short_pool["asset_type"] == "Stock"]
    etfs_short   = short_pool[short_pool["asset_type"] == "ETF"]

    return {
        "long_stocks":  _bucket(stocks_long,  "long"),
        "long_etfs":    _bucket(etfs_long,    "long"),
        "short_stocks": _bucket(stocks_short, "short"),
        "short_etfs":   _bucket(etfs_short,   "short"),
    }


if __name__ == "__main__":
    from data import load_universe, download_prices

    universe = load_universe()
    tickers = [u["ticker"] for u in universe]
    # Ensure SPY in universe
    if "SPY" not in tickers:
        tickers.append("SPY")
        universe.append({"ticker":"SPY","name":"SPDR S&P 500 ETF","sector":"Broad","asset_type":"ETF"})

    print("Loading prices (will use cache if present)…")
    data = download_prices(tickers, start="2025-08-01", end="2026-06-08")

    as_of = pd.Timestamp("2026-03-13")
    print(f"\n=== Proxy selection as of {as_of.date()} ===")
    scored = compute_proxy_score_one_date(data, universe, as_of)
    if scored is None:
        print("Insufficient data")
    else:
        print(f"Scored {len(scored)} tickers (LONG_OK={sum(scored['classification']=='LONG_OK')}, "
              f"SHORT_OK={sum(scored['classification']=='SHORT_OK')})")
        picks = select_picks_one_date(scored)
        for bucket in ["long_stocks", "long_etfs", "short_stocks", "short_etfs"]:
            print(f"\n--- {bucket} (top {len(picks[bucket])}) ---")
            for p in picks[bucket][:10]:
                print(f"  #{p['rank']:2}  {p['ticker']:10} {p['name'][:24]:24} "
                      f"Score {p['proxy_score']:>6}  "
                      f"mom21d {p['mom_21d']:+5.1f}%  RS {p['rs_21d']:+5.1f}%  {p['sector'][:14]}")
