"""
compute_ytd.py — One-shot YTD return enrichment.

Fetches end-of-prior-year close + current close for every ticker in
.scan_cache.pkl (single yfinance batch, ~30-60s for 756 tickers) and writes
YTD return % to .ytd_returns.json for the API to merge into df.

Use this between scans to refresh YTD without running the full scan.
Once price_discovery.py is re-run, ret_ytd is included natively in the cache
and this enrichment becomes redundant (api.py will prefer cache value).

Run:  python3 compute_ytd.py
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
import yfinance as yf

CACHE_PATH = ".scan_cache.pkl"
OUT_PATH = ".ytd_returns.json"


def fetch_ytd_returns(tickers: List[str]) -> Dict[str, float]:
    """Bulk-fetch close prices since (current_year-1)-12-15 → today.
    Returns {ticker: ytd_pct_return}.

    EOY price = last close on or before current_year-1-12-31.
    Current price = last close in fetched window.
    """
    if not tickers:
        return {}

    today = datetime.now()
    current_year = today.year
    # Start a couple of weeks before year-end to ensure we capture the last
    # trading day of the prior year (and tolerate holidays).
    start = datetime(current_year - 1, 12, 15)
    end = today + timedelta(days=1)

    print(f"Fetching {len(tickers)} tickers from {start.date()} → {end.date()} (Adj Close)")
    df = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    out: Dict[str, float] = {}

    # Two layouts: single-ticker → flat columns; multi → MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # MultiIndex: (ticker, field)
        for tk in tickers:
            try:
                if tk not in df.columns.get_level_values(0):
                    continue
                close = df[tk]["Close"].dropna()
                if close.empty:
                    continue
                # EOY price: last close in prior year
                prior_year_mask = close.index.year < current_year
                if not prior_year_mask.any():
                    continue
                eoy_price = float(close[prior_year_mask].iloc[-1])
                cur_price = float(close.iloc[-1])
                if eoy_price > 0:
                    out[tk] = round((cur_price / eoy_price - 1.0) * 100, 3)
            except Exception:
                pass
    else:
        # Single ticker
        if "Close" in df.columns:
            close = df["Close"].dropna()
            if not close.empty and tickers:
                prior_year_mask = close.index.year < current_year
                if prior_year_mask.any():
                    eoy_price = float(close[prior_year_mask].iloc[-1])
                    cur_price = float(close.iloc[-1])
                    if eoy_price > 0:
                        out[tickers[0]] = round((cur_price / eoy_price - 1.0) * 100, 3)

    return out


def main():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    tickers = sorted({r["ticker"] for r in cache.get("results", []) if r.get("ticker")})
    print(f"Universe: {len(tickers)} tickers")

    # yfinance has a soft limit per batch — chunk to be safe
    CHUNK = 200
    ytd: Dict[str, float] = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        print(f"\nBatch {i // CHUNK + 1} ({i + 1}-{i + len(chunk)} of {len(tickers)})...")
        ytd.update(fetch_ytd_returns(chunk))
        print(f"  cumulative: {len(ytd)}/{len(tickers)}")

    out = {
        "as_of": datetime.utcnow().isoformat(),
        "n_total": len(tickers),
        "n_with_ytd": len(ytd),
        "current_year": datetime.now().year,
        "tickers": ytd,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n✓ Wrote {OUT_PATH}  ·  {out['n_with_ytd']}/{out['n_total']} tickers ({out['n_with_ytd']/out['n_total']*100:.1f}%)")

    # Print top movers
    sorted_ytd = sorted(ytd.items(), key=lambda x: -x[1])
    print(f"\nTop 5 YTD: {sorted_ytd[:5]}")
    print(f"Bottom 5 YTD: {sorted_ytd[-5:]}")


if __name__ == "__main__":
    main()
