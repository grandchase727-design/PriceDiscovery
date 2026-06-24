# -*- coding: utf-8 -*-
"""data.py — historical price downloader for backtest.

Downloads OHLCV from yfinance for the 770 ticker universe and caches locally.
Reused across backtest runs to avoid repeated network calls.

Cache file: .backtest_price_cache.pkl  (gitignored)
"""
from __future__ import annotations

import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

CACHE_PATH = Path(".backtest_price_cache.pkl")


def load_universe() -> list[dict]:
    """Return list of {ticker, name, sector, asset_type} from scan_cache."""
    sc = pickle.load(open(".scan_cache.pkl", "rb"))
    results = sc.get("results") or []
    out = []
    for r in results:
        t = r.get("ticker")
        if not t:
            continue
        asset_type = r.get("asset_type") or ("Stock" if (r.get("category") or "").startswith("STK_") else "ETF")
        sector = r.get("sector") or r.get("category", "Other")
        if isinstance(sector, str) and sector.startswith(("STK_", "EQ_", "FI_", "MA_", "ETF_")):
            sector = sector.split("_", 1)[1]
        out.append({
            "ticker": t,
            "name": r.get("name", ""),
            "sector": sector,
            "asset_type": asset_type,
        })
    return out


def _save_cache(payload: dict) -> None:
    CACHE_PATH.write_bytes(pickle.dumps(payload))


def _load_cache() -> Optional[dict]:
    if not CACHE_PATH.exists():
        return None
    try:
        return pickle.loads(CACHE_PATH.read_bytes())
    except Exception:
        return None


def download_prices(
    tickers: list[str],
    start: str = "2025-08-01",
    end: str = "2026-06-08",
    batch_size: int = 50,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download adjusted close + volume for all tickers, batched.

    Returns dict[ticker] -> DataFrame with columns: ['Close','Volume']
    """
    if use_cache:
        cached = _load_cache()
        if cached and cached.get("start") == start and cached.get("end") == end:
            existing = set(cached.get("data", {}).keys())
            missing = [t for t in tickers if t not in existing]
            if not missing:
                print(f"[data] cache hit: {len(existing)} tickers loaded from {CACHE_PATH}")
                return cached["data"]
            print(f"[data] cache partial hit: {len(existing)} existing, {len(missing)} new to download")
            tickers_to_dl = missing
            data = dict(cached["data"])
        else:
            print(f"[data] cache miss or date range changed; downloading all {len(tickers)} tickers")
            tickers_to_dl = tickers
            data = {}
    else:
        tickers_to_dl = tickers
        data = {}

    fails: list[str] = []
    n = len(tickers_to_dl)
    for i in range(0, n, batch_size):
        batch = tickers_to_dl[i:i + batch_size]
        print(f"[data] batch {i//batch_size + 1}/{(n + batch_size - 1)//batch_size}: {len(batch)} tickers")
        try:
            df = yf.download(
                tickers=batch, start=start, end=end,
                progress=False, group_by="ticker",
                auto_adjust=True, threads=True,
            )
        except Exception as e:
            print(f"  ERROR: batch failed entirely — {e}")
            fails.extend(batch)
            continue

        if len(batch) == 1:
            t = batch[0]
            try:
                sub = df[["Close", "Volume"]].dropna()
                if len(sub) >= 60:
                    data[t] = sub
                else:
                    fails.append(t)
            except Exception:
                fails.append(t)
        else:
            for t in batch:
                try:
                    sub = df[t][["Close", "Volume"]].dropna()
                    if len(sub) >= 60:
                        data[t] = sub
                    else:
                        fails.append(t)
                except Exception:
                    fails.append(t)

        time.sleep(0.5)   # gentle rate limit

    if use_cache:
        _save_cache({"start": start, "end": end, "data": data,
                     "fetched_at": datetime.utcnow().isoformat()})

    print(f"[data] success: {len(data)} / {len(tickers)} tickers   |   fails: {len(fails)}")
    return data


def get_trading_fridays(year: int = 2026, end_date: str = "2026-06-07") -> list[pd.Timestamp]:
    """Return list of Friday dates from Jan 1 of `year` to end_date."""
    end = pd.Timestamp(end_date)
    start = pd.Timestamp(f"{year}-01-01")
    dates = pd.date_range(start, end, freq="W-FRI")
    return list(dates)


if __name__ == "__main__":
    uni = load_universe()
    print(f"universe: {len(uni)} tickers ({sum(1 for u in uni if u['asset_type']=='Stock')} stocks, "
          f"{sum(1 for u in uni if u['asset_type']=='ETF')} etfs)")
    tickers = [u["ticker"] for u in uni]
    data = download_prices(tickers[:30])   # smoke test with 30
    print(f"data shape: {len(data)} tickers")
    for t, df in list(data.items())[:3]:
        print(f"  {t}: {len(df)} days,  {df.index[0].date()} → {df.index[-1].date()}")
