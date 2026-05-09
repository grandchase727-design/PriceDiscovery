"""
finnhub_fundamentals.py — Enrich fundamentals_cache with Finnhub data.

Loads `.fundamentals_cache.pkl` (built by fundamentals_pipeline.py from yfinance),
fetches Finnhub data for every non-Korean ticker, and enriches each entry with:

  finnhub_metrics    : full Finnhub ratio dict (gross/op/profit margin, ROE/ROA,
                       PE/PB/PS, debt ratios, growth ratios — 70+ fields)
  rec_history        : monthly recommendation trend (~4 months of buy/hold/sell)
  eps_surprises      : quarterly EPS actual/estimate/surprise (~4 quarters)

Plus derived leading signals:
  bullish_change_3m  : current bullish_ratio − 3-months-ago bullish_ratio
                       (positive = analysts becoming more bullish over time)
  eps_beat_rate      : fraction of recent quarters that beat estimate
  eps_surprise_avg   : average surprise % across quarters
  rec_total_now      : total #analysts (latest month)
  bullish_ratio_now  : (strongBuy + buy) / total (latest month)

Korean tickers (.KS) are skipped — Finnhub free tier returns 403 for them.

Usage:
  python3 finnhub_fundamentals.py            # enrich full cache
  python3 finnhub_fundamentals.py --workers 4
"""

from __future__ import annotations

import argparse
import os, sys, pickle, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# This file lives in pipelines/ — project root is one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.finnhub_client import FinnhubClient
from pipelines.fundamentals_pipeline import (
    load_fundamentals_cache, CACHE_PATH as FUND_CACHE_PATH,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def is_supported_symbol(ticker: str) -> bool:
    """Finnhub free tier covers US-listed only. Skip Korean .KS / Japan local etc."""
    if "." in ticker:
        suffix = ticker.split(".")[-1].upper()
        # Whitelist: blank suffix or .US is OK; anything else (KS/T/HK/SS/SZ) blocked
        return suffix == "US"
    return True  # plain ticker = US


def _bullish_ratio(rec: dict) -> Optional[float]:
    sb = rec.get("strongBuy", 0) or 0
    b = rec.get("buy", 0) or 0
    h = rec.get("hold", 0) or 0
    s = rec.get("sell", 0) or 0
    ss = rec.get("strongSell", 0) or 0
    total = sb + b + h + s + ss
    if total == 0:
        return None
    return (sb + b) / total


def _derive_signals(rec_history: list, eps_surprises: list,
                    news_items: Optional[list] = None) -> dict:
    """Compute derived leading signals from raw history."""
    out = {
        "rec_total_now": 0,
        "bullish_ratio_now": None,
        "bullish_change_3m": None,
        "eps_beat_rate": None,
        "eps_surprise_avg": None,
        "eps_n_quarters": 0,
        # News-derived (Catalyst Agent input)
        "news_count_7d": 0,
        "news_count_3d": 0,
        "news_recency": None,    # fraction of 7d news in last 3d → 0..1, >0.43 means accelerating
    }

    # Recommendation trend
    if rec_history:
        # Sorted descending by period typically (latest first), but ensure
        sorted_rec = sorted(rec_history, key=lambda r: r.get("period", ""), reverse=True)
        latest = sorted_rec[0]
        latest_ratio = _bullish_ratio(latest)
        out["bullish_ratio_now"] = latest_ratio
        out["rec_total_now"] = (
            (latest.get("strongBuy", 0) or 0) + (latest.get("buy", 0) or 0)
            + (latest.get("hold", 0) or 0) + (latest.get("sell", 0) or 0)
            + (latest.get("strongSell", 0) or 0)
        )
        if len(sorted_rec) >= 4 and latest_ratio is not None:
            # 3 months ago = index 3 (4th most recent)
            prev_ratio = _bullish_ratio(sorted_rec[3])
            if prev_ratio is not None:
                out["bullish_change_3m"] = latest_ratio - prev_ratio
        elif len(sorted_rec) >= 2 and latest_ratio is not None:
            # Fallback: use earliest available
            prev_ratio = _bullish_ratio(sorted_rec[-1])
            if prev_ratio is not None:
                out["bullish_change_3m"] = latest_ratio - prev_ratio

    # Earnings surprises
    if eps_surprises:
        valid = [e for e in eps_surprises
                 if e.get("actual") is not None and e.get("estimate") is not None]
        if valid:
            out["eps_n_quarters"] = len(valid)
            beats = sum(1 for e in valid if e["actual"] > e["estimate"])
            out["eps_beat_rate"] = beats / len(valid)
            surprises = [e.get("surprisePercent") for e in valid
                         if e.get("surprisePercent") is not None]
            if surprises:
                out["eps_surprise_avg"] = sum(surprises) / len(surprises)

    # News (catalyst input)
    if news_items:
        import time as _time
        now_epoch = _time.time()
        cutoff_3d = now_epoch - (3 * 86400)
        out["news_count_7d"] = len(news_items)
        cnt_3d = sum(1 for n in news_items if n.get("datetime", 0) >= cutoff_3d)
        out["news_count_3d"] = cnt_3d
        if out["news_count_7d"] > 0:
            out["news_recency"] = cnt_3d / out["news_count_7d"]

    return out


# ──────────────────────────────────────────────────────────────────────
# Per-ticker fetch
# ──────────────────────────────────────────────────────────────────────

def fetch_one(client: FinnhubClient, ticker: str, asset_type: str,
              fetch_news: bool = True, news_window_days: int = 7) -> dict:
    """Fetch all relevant Finnhub data for one ticker. Returns enrichment dict."""
    import datetime as _dt
    out = {
        "ticker": ticker,
        "asset_type": asset_type,
        "finnhub_metrics": None,
        "rec_history": None,
        "eps_surprises": None,
        "news_items_meta": None,   # lightweight: just datetime list (full payload too heavy)
        "derived": None,
        "fetch_ok": False,
        "error": None,
    }
    try:
        # Strip .US suffix if present (Finnhub accepts both bare and .US for US tickers)
        sym = ticker.replace(".US", "")

        # 1. Metrics (always try — works for US stocks AND ETFs)
        m = client.metrics(sym)
        if m and isinstance(m, dict) and "metric" in m:
            out["finnhub_metrics"] = m["metric"]

        # 2. Recommendation history + earnings (only stocks; ETFs return empty)
        if asset_type == "Stock":
            rec = client.recommendation_trends(sym)
            if rec and isinstance(rec, list) and len(rec) > 0:
                out["rec_history"] = rec

            es = client.earnings_surprises(sym)
            if es and isinstance(es, list) and len(es) > 0:
                out["eps_surprises"] = es

            # 3. News (Catalyst Agent input — last N days)
            if fetch_news:
                to_d = _dt.date.today().isoformat()
                from_d = (_dt.date.today() - _dt.timedelta(days=news_window_days)).isoformat()
                news = client.company_news(sym, from_d, to_d)
                if news and isinstance(news, list):
                    # Store only datetime for compactness; we just need timing/count
                    out["news_items_meta"] = [
                        {"datetime": n.get("datetime", 0)} for n in news
                    ]

        # Derived signals (compute even if some sources missing)
        out["derived"] = _derive_signals(
            out["rec_history"] or [],
            out["eps_surprises"] or [],
            out["news_items_meta"] or [],
        )

        # Mark OK if we got at least the metrics
        out["fetch_ok"] = out["finnhub_metrics"] is not None
    except Exception as e:
        out["error"] = str(e)[:200]

    return out


# ──────────────────────────────────────────────────────────────────────
# Batch enrichment
# ──────────────────────────────────────────────────────────────────────

def enrich_cache(cache_path: str = FUND_CACHE_PATH,
                 max_workers: int = 4,
                 progress_every: int = 25) -> dict:
    """Load fundamentals cache, fetch Finnhub for each US ticker, save back."""
    cache = load_fundamentals_cache(cache_path)
    if cache is None:
        raise FileNotFoundError(f"No cache at {cache_path}. Run fundamentals_pipeline.py first.")

    tickers = cache.get("tickers", {})
    targets = [(tk, t.get("asset_type", "Stock"))
               for tk, t in tickers.items()
               if is_supported_symbol(tk)]
    skipped = [tk for tk in tickers if not is_supported_symbol(tk)]

    print(f"[finnhub] enriching cache: {len(targets)} US-supported / "
          f"{len(skipped)} skipped (Korean/local)")

    client = FinnhubClient()

    enriched_count = 0
    failed_count = 0
    t0 = time.time()
    done = 0

    # Use sequential with small worker pool — Finnhub rate limit is 60/min
    # 4 workers ≈ 240 req/min burst, but client.get_json self-throttles
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_one, client, tk, at): tk for tk, at in targets}
        for fut in as_completed(futures):
            tk = futures[fut]
            try:
                result = fut.result()
                if result.get("fetch_ok"):
                    # Merge into existing ticker entry
                    existing = tickers.get(tk, {})
                    existing["finnhub_metrics"] = result["finnhub_metrics"]
                    existing["rec_history"] = result["rec_history"]
                    existing["eps_surprises"] = result["eps_surprises"]
                    existing["news_items_meta"] = result.get("news_items_meta")
                    existing["finnhub_derived"] = result["derived"]
                    existing["finnhub_ok"] = True
                    tickers[tk] = existing
                    enriched_count += 1
                else:
                    failed_count += 1
                    if tk in tickers:
                        tickers[tk]["finnhub_ok"] = False
                        tickers[tk]["finnhub_error"] = result.get("error")
            except Exception as e:
                failed_count += 1
            done += 1
            if done % progress_every == 0 or done == len(targets):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(targets) - done) / rate if rate else 0
                print(f"[finnhub] {done}/{len(targets)} ({100*done/len(targets):.0f}%) "
                      f"| {rate:.1f} tk/s | ETA {eta:.0f}s "
                      f"| enriched={enriched_count} failed={failed_count}")

    duration = time.time() - t0

    # Update top-level stats
    cache["finnhub_enriched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cache.setdefault("stats", {}).update({
        "finnhub_enriched": enriched_count,
        "finnhub_failed": failed_count,
        "finnhub_skipped_korean": len(skipped),
        "finnhub_duration_sec": round(duration, 1),
    })

    # Persist
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"\n[finnhub] DONE — enriched {enriched_count} tickers in {duration:.0f}s")
    print(f"  Failed: {failed_count} | Korean skipped: {len(skipped)}")
    print(f"  Cache saved to {cache_path}")

    return cache


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich fundamentals cache with Finnhub data")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache-path", default=FUND_CACHE_PATH)
    args = parser.parse_args()
    enrich_cache(cache_path=args.cache_path, max_workers=args.workers)
