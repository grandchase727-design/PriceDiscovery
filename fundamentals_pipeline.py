"""
fundamentals_pipeline.py — Daily fundamental + analyst-consensus cache

Fetches per-ticker fundamentals via yfinance for the full universe
(GLOBAL_ETF_UNIVERSE + STOCK_UNIVERSE) and writes to `.fundamentals_cache.pkl`.

Key data captured:
  - info:        trailingPE, forwardPE, priceToBook, gross/operating/profit margins,
                 ROE, debt/equity, market cap, beta, dividend yield,
                 earningsGrowth, revenueGrowth
  - estimates:   forward EPS / revenue consensus (avg/low/high/n_analysts/growth)
                 across 4 horizons: 0q (current qtr), +1q, 0y (current yr), +1y
  - revisions:   upLast7days/upLast30days/downLast30days/downLast7Days
                 → derived: net_30d, ratio_30d (key leading signal)

Run:
  python3 fundamentals_pipeline.py                   # full universe
  python3 fundamentals_pipeline.py --tickers AAPL    # one ticker
  python3 fundamentals_pipeline.py --max-age-h 12    # skip if cache fresh
"""

import os, sys, pickle, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price_discovery import GLOBAL_ETF_UNIVERSE, STOCK_UNIVERSE

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fundamentals_cache.pkl")

# ── Fields to extract from yfinance .info ──
# (left = our snake_case, right = yfinance camelCase)
INFO_FIELDS_STOCK = {
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "price_to_book": "priceToBook",
    "price_to_sales": "priceToSalesTrailing12Months",
    "peg": "pegRatio",
    "gross_margin": "grossMargins",
    "operating_margin": "operatingMargins",
    "profit_margin": "profitMargins",
    "roe": "returnOnEquity",
    "roa": "returnOnAssets",
    "debt_to_equity": "debtToEquity",
    "current_ratio": "currentRatio",
    "quick_ratio": "quickRatio",
    "market_cap": "marketCap",
    "enterprise_value": "enterpriseValue",
    "beta": "beta",
    "dividend_yield": "dividendYield",
    "earnings_growth": "earningsGrowth",       # YoY
    "revenue_growth": "revenueGrowth",         # YoY
    "earnings_q_growth": "earningsQuarterlyGrowth",
    "revenue_q_growth": "revenueQuarterlyGrowth",
    "free_cash_flow": "freeCashflow",
    "operating_cf": "operatingCashflow",
}
INFO_FIELDS_ETF = {
    "total_assets": "totalAssets",
    "nav_price": "navPrice",
    "ytd_return": "ytdReturn",
    "expense_ratio": "annualReportExpenseRatio",
    "trailing_pe": "trailingPE",          # ETF aggregate when present
    "dividend_yield": "dividendYield",
    "beta": "beta",
}

ESTIMATE_PERIODS = ["0q", "+1q", "0y", "+1y"]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _safe(v):
    """Coerce yfinance numerics to JSON-safe Python types."""
    try:
        if v is None:
            return None
        if isinstance(v, float):
            return v if pd.notna(v) and abs(v) < 1e30 else None
        if isinstance(v, (int,)):
            return int(v)
        if hasattr(v, "item"):
            return _safe(v.item())
        return v
    except Exception:
        return None


def _extract_info(info: dict, fields: dict) -> dict:
    out = {}
    for k, src in fields.items():
        out[k] = _safe(info.get(src))
    return out


def _extract_estimates(t: yf.Ticker) -> Optional[dict]:
    """Extract earnings + revenue estimates across 4 horizons."""
    out = {}
    try:
        ee = t.earnings_estimate
        if ee is None or ee.empty:
            return None
        for period in ESTIMATE_PERIODS:
            if period not in ee.index:
                continue
            row = ee.loc[period]
            out[period] = {
                "eps_avg": _safe(row.get("avg")),
                "eps_low": _safe(row.get("low")),
                "eps_high": _safe(row.get("high")),
                "year_ago_eps": _safe(row.get("yearAgoEps")),
                "n_analysts": _safe(row.get("numberOfAnalysts")),
                "growth": _safe(row.get("growth")),
            }
    except Exception:
        return None

    # Add revenue estimates if available
    try:
        re = t.revenue_estimate
        if re is not None and not re.empty:
            for period in ESTIMATE_PERIODS:
                if period not in re.index or period not in out:
                    continue
                row = re.loc[period]
                out[period]["rev_avg"] = _safe(row.get("avg"))
                out[period]["rev_growth"] = _safe(row.get("growth"))
    except Exception:
        pass

    return out if out else None


def _extract_revisions(t: yf.Ticker) -> Optional[dict]:
    """Extract EPS revision counts (key leading signal)."""
    try:
        er = t.eps_revisions
        if er is None or er.empty:
            return None
        # Use 0q (current quarter) as primary — most actionable
        if "0q" not in er.index:
            return None
        row = er.loc["0q"]
        up7 = _safe(row.get("upLast7days")) or 0
        up30 = _safe(row.get("upLast30days")) or 0
        dn7 = _safe(row.get("downLast7Days")) or 0
        dn30 = _safe(row.get("downLast30days")) or 0

        total30 = up30 + dn30
        ratio_30d = (up30 / total30) if total30 > 0 else None  # 1.0 = all upgrades, 0.0 = all downgrades

        # Also capture +1q and 0y for trend
        out_periods = {}
        for period in ["0q", "+1q", "0y"]:
            if period in er.index:
                r = er.loc[period]
                out_periods[period] = {
                    "up_7d": int(_safe(r.get("upLast7days")) or 0),
                    "up_30d": int(_safe(r.get("upLast30days")) or 0),
                    "down_7d": int(_safe(r.get("downLast7Days")) or 0),
                    "down_30d": int(_safe(r.get("downLast30days")) or 0),
                }
        return {
            "up_7d": int(up7), "up_30d": int(up30),
            "down_7d": int(dn7), "down_30d": int(dn30),
            "net_30d": int(up30 - dn30),
            "ratio_30d": ratio_30d,
            "by_period": out_periods,
        }
    except Exception:
        return None


def _extract_recommendations(t: yf.Ticker) -> Optional[dict]:
    """Most recent recommendation summary (strongBuy/buy/hold/sell counts)."""
    try:
        rec = t.recommendations
        if rec is None or rec.empty:
            return None
        # yfinance returns months of history; take most recent row
        latest = rec.iloc[0]
        out = {
            "strong_buy": int(_safe(latest.get("strongBuy")) or 0),
            "buy": int(_safe(latest.get("buy")) or 0),
            "hold": int(_safe(latest.get("hold")) or 0),
            "sell": int(_safe(latest.get("sell")) or 0),
            "strong_sell": int(_safe(latest.get("strongSell")) or 0),
        }
        total = sum(out.values())
        out["total"] = total
        if total > 0:
            # Bullish ratio: (strong_buy + buy) / total
            out["bullish_ratio"] = (out["strong_buy"] + out["buy"]) / total
            out["bearish_ratio"] = (out["sell"] + out["strong_sell"]) / total
        return out
    except Exception:
        return None


def _extract_price_targets(t: yf.Ticker, current_price: Optional[float]) -> Optional[dict]:
    """Analyst consensus price target."""
    try:
        info = t.info or {}
        target = _safe(info.get("targetMeanPrice"))
        if target is None:
            return None
        return {
            "mean": target,
            "median": _safe(info.get("targetMedianPrice")),
            "low": _safe(info.get("targetLowPrice")),
            "high": _safe(info.get("targetHighPrice")),
            "n_analysts": _safe(info.get("numberOfAnalystOpinions")),
            "upside_pct": ((target / current_price - 1.0) * 100) if (current_price and target) else None,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Per-ticker fetch
# ──────────────────────────────────────────────────────────────────────

def fetch_one(ticker: str, asset_type: str) -> dict:
    """Fetch fundamentals for a single ticker. Returns a dict (never raises)."""
    t0 = time.time()
    out = {
        "ticker": ticker,
        "asset_type": asset_type,
        "info": None,
        "estimates": None,
        "revisions": None,
        "recommendations": None,
        "price_targets": None,
        "fetch_ok": False,
        "error": None,
        "elapsed_sec": 0.0,
    }
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        if not info or "symbol" not in info and "shortName" not in info and "longName" not in info:
            # Empty/unrecognized — likely delisted or bad ticker
            out["error"] = "no_info"
            out["elapsed_sec"] = time.time() - t0
            return out

        fields = INFO_FIELDS_ETF if asset_type == "ETF" else INFO_FIELDS_STOCK
        out["info"] = _extract_info(info, fields)
        current_price = _safe(info.get("currentPrice") or info.get("regularMarketPrice"))

        if asset_type == "Stock":
            out["estimates"] = _extract_estimates(t)
            out["revisions"] = _extract_revisions(t)
            out["recommendations"] = _extract_recommendations(t)
            out["price_targets"] = _extract_price_targets(t, current_price)

        out["fetch_ok"] = True
    except Exception as e:
        out["error"] = str(e)[:200]
    finally:
        out["elapsed_sec"] = round(time.time() - t0, 2)

    return out


# ──────────────────────────────────────────────────────────────────────
# Batch pipeline
# ──────────────────────────────────────────────────────────────────────

def _region_of(ticker: str) -> str:
    """Coarse region bucket by listing suffix (diagnostics + fair-ordering)."""
    t = str(ticker).upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        return "KR"
    if t.endswith(".T"):
        return "JP"
    if t.endswith(".HK") or t.endswith(".SS") or t.endswith(".SZ"):
        return "CN"
    if t.endswith(".NS") or t.endswith(".BO"):
        return "IN"
    if t.endswith((".L", ".PA", ".DE", ".AS", ".SW", ".MI", ".MC", ".ST", ".BR", ".LS")):
        return "EU"
    return "US"


def build_universe(interleave: bool = True) -> list[tuple[str, str]]:
    """Returns list of (ticker, asset_type) covering the full universe.

    ETFs first, then stocks, in each dict's category order — then DEDUPED by ticker
    (first occurrence wins, so a ticker present in both universes is fetched as ETF).

    When interleave=True (default) the deduped list is round-robin merged across its
    source categories so no category is structurally submitted last. This fixes the
    rate-limit tail bias where STOCK_UNIVERSE's trailing sections (Korea / Japan /
    China / Europe / India) absorbed ~100% of the 429 failures under the un-throttled
    batch. Set interleave=False for the raw dict-order stream (kept for reproducibility).
    Signature stays list[tuple[str,str]] so both callers (run_pipeline, retry_failed)
    are unaffected.
    """
    from collections import OrderedDict

    # 1) ordered (category, ticker, asset_type) in dict order
    ordered: list[tuple[str, str, str]] = []
    for cat, data in GLOBAL_ETF_UNIVERSE.items():
        for tk in data["tickers"].keys():
            ordered.append((cat, tk, "ETF"))
    for cat, data in STOCK_UNIVERSE.items():
        for tk in data["tickers"].keys():
            ordered.append((cat, tk, "Stock"))

    # 2) dedupe by ticker (first occurrence wins), grouped back by source category
    seen: set = set()
    by_cat: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    for cat, tk, at in ordered:
        if tk in seen:
            continue
        seen.add(tk)
        by_cat.setdefault(cat, []).append((tk, at))

    if not interleave:
        return [pair for pairs in by_cat.values() for pair in pairs]

    # 3) round-robin across categories → each pass takes one ticker per non-empty
    #    category (in category order), so every region/sector is spread evenly through
    #    the stream and no group is ever the structural tail of the submission queue.
    queues = [list(pairs) for pairs in by_cat.values() if pairs]
    merged: list[tuple[str, str]] = []
    while queues:
        next_round = []
        for q in queues:
            merged.append(q.pop(0))
            if q:
                next_round.append(q)
        queues = next_round
    return merged


def _merge_results(existing: dict, fresh: dict) -> dict:
    """Non-destructive merge of a fresh fetch batch into existing cache records.

    Rule: a fresh record replaces the prior ONLY if it is a successful fetch, OR the
    prior record was itself a failure/absent (so we keep the latest error for retry
    visibility). A fresh FAILED fetch against a prior GOOD (fetch_ok=True) record is
    DISCARDED — the good data is kept. This makes any re-run strictly monotonic: it can
    upgrade a record or re-record an error on an already-bad slot, but can NEVER null out
    previously-good fundamentals. This is the guard that stops a rate-limited full run
    from re-destroying good (e.g. Korean) data.
    """
    merged = dict(existing or {})
    for tk, r in (fresh or {}).items():
        prior = merged.get(tk)
        if r.get("fetch_ok"):
            merged[tk] = r                               # success always wins
        elif not (prior and prior.get("fetch_ok")):
            merged[tk] = r                               # prior bad/absent → keep latest error
        # else: fresh failed but prior was good → keep prior (discard fresh)
    return merged


def _recompute_stats(tickers_map: dict, total_attempted: int, duration: float) -> dict:
    """Derive the cache stats block from the (possibly merged) full ticker map."""
    vals = list(tickers_map.values())
    failed_tk = [tk for tk, r in tickers_map.items() if not r.get("fetch_ok")]
    return {
        "total_attempted": total_attempted,
        "stock_ok": sum(1 for r in vals if r.get("fetch_ok") and r.get("asset_type") == "Stock"),
        "etf_ok": sum(1 for r in vals if r.get("fetch_ok") and r.get("asset_type") == "ETF"),
        "has_estimates": sum(1 for r in vals if r.get("estimates")),
        "has_revisions": sum(1 for r in vals if r.get("revisions")),
        "failed_count": len(failed_tk),
        "failed_tickers": failed_tk,
        "duration_sec": round(duration, 1),
    }


def run_pipeline(
    tickers: Optional[list[tuple[str, str]]] = None,
    max_workers: int = 4,
    cache_path: str = CACHE_PATH,
    progress_every: int = 25,
    chunk_size: int = 100,
    chunk_cooldown: float = 15.0,
    submit_delay: float = 0.05,
    auto_retry: bool = True,
    auto_retry_threshold: int = 30,
) -> dict:
    """Fetch fundamentals and MERGE into the cache (non-destructive).

    Throttled + chunked: the interleaved universe is fetched in region-mixed chunks of
    `chunk_size`, each on its own ThreadPoolExecutor(max_workers) pool with `submit_delay`
    spacing between submissions and a `chunk_cooldown` pause between chunks. This lets
    Yahoo's per-window rate budget recover between bursts and stops the mid-batch 429
    cascade that used to dump ~100% of failures on the tail (Korea/Japan/China/Europe/
    India). Results are MERGE-written via _merge_results so a rate-limited fetch can never
    overwrite previously-good data. On a full run, if the post-merge failure count exceeds
    `auto_retry_threshold`, retry_failed() is auto-invoked (non-destructive) so recovery
    needs no manual --retry-failed.
    """
    is_full_run = tickers is None
    if tickers is None:
        tickers = build_universe()   # interleaved by default

    started = datetime.now(timezone.utc)
    print(f"[fundamentals] starting batch: {len(tickers)} tickers, {max_workers} workers, "
          f"chunk={chunk_size}, cooldown={chunk_cooldown}s")

    results: dict[str, dict] = {}
    failed: list[str] = []
    t0 = time.time()
    done = 0
    total = len(tickers)

    for ci in range(0, total, max(1, chunk_size)):
        chunk = tickers[ci:ci + chunk_size]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_tk = {}
            for tk, at in chunk:
                future_to_tk[pool.submit(fetch_one, tk, at)] = (tk, at)
                if submit_delay > 0:
                    time.sleep(submit_delay)   # space out submissions
            for future in as_completed(future_to_tk):
                tk, at = future_to_tk[future]
                try:
                    r = future.result()
                    results[tk] = r
                    if not r["fetch_ok"]:
                        failed.append(tk)
                except Exception as e:
                    failed.append(tk)
                    results[tk] = {
                        "ticker": tk, "asset_type": at, "fetch_ok": False, "error": str(e)[:200],
                    }
                done += 1
                if done % progress_every == 0 or done == total:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed else 0
                    eta = (total - done) / rate if rate else 0
                    print(f"[fundamentals] {done}/{total} "
                          f"({100*done/total:.0f}%) "
                          f"| {rate:.1f} tk/s | ETA {eta:.0f}s "
                          f"| failed={len(failed)}")
        # inter-chunk cooldown (skip after the final chunk)
        if ci + chunk_size < total and chunk_cooldown > 0:
            time.sleep(chunk_cooldown)

    duration = time.time() - t0

    # ── MERGE-SAFE WRITE: fold this batch into the existing cache; a fresh failure
    #    never overwrites a prior good record (see _merge_results).
    prior = load_fundamentals_cache(cache_path)
    prior_tickers = prior.get("tickers", {}) if isinstance(prior, dict) else {}
    merged_tickers = _merge_results(prior_tickers, results)

    cache = {
        "fetched_at": started.isoformat(),
        "tickers": merged_tickers,
        "stats": _recompute_stats(merged_tickers, total, duration),
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    st = cache["stats"]
    print(f"\n[fundamentals] DONE — merge-saved to {cache_path}")
    print(f"  This batch: {total} attempted | fresh-failed: {len(failed)}")
    print(f"  Cache total: Stock OK: {st['stock_ok']} | ETF OK: {st['etf_ok']} "
          f"| failed (all): {st['failed_count']} | Duration: {duration:.0f}s")

    # ── AUTO-HEAL: on a full run, if too many failures remain, run the non-destructive
    #    rate-limit retry so recovery is human-free.
    if is_full_run and auto_retry:
        rl_failed = [tk for tk, r in merged_tickers.items()
                     if not r.get("fetch_ok") and _is_rate_limited(r.get("error", ""))]
        if len(rl_failed) >= auto_retry_threshold:
            print(f"[fundamentals] auto-retry: {len(rl_failed)} rate-limited failures "
                  f">= {auto_retry_threshold} → running retry_failed() (non-destructive)")
            cache = retry_failed(cache_path=cache_path, cooldown_sec=90,
                                 max_workers=2, per_request_delay=0.3, max_attempts=3)

    return cache


# ──────────────────────────────────────────────────────────────────────
# Cache loader (for downstream consumers — Pre-Mom / Momentum / API)
# ──────────────────────────────────────────────────────────────────────

def load_fundamentals_cache(cache_path: str = CACHE_PATH) -> Optional[dict]:
    """Load cached fundamentals. Returns None if missing."""
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[fundamentals] cache load failed: {e}")
        return None


def cache_age_hours(cache_path: str = CACHE_PATH) -> Optional[float]:
    """Returns age of cache in hours, or None if missing."""
    if not os.path.exists(cache_path):
        return None
    try:
        mtime = os.path.getmtime(cache_path)
        return (time.time() - mtime) / 3600.0
    except Exception:
        return None


def _is_rate_limited(err_msg: str) -> bool:
    if not err_msg:
        return False
    s = err_msg.lower()
    return ("rate limit" in s) or ("too many requests" in s) or ("429" in s)


def retry_failed(
    cache_path: str = CACHE_PATH,
    cooldown_sec: int = 300,
    max_workers: int = 2,
    per_request_delay: float = 0.3,
    max_attempts: int = 3,
    include_all_failures: bool = False,
) -> dict:
    """Reload cache, retry failed fetches with conservative settings (non-destructive merge).

    Iterates up to `max_attempts` times — each round waits `cooldown_sec` before retrying.
    By default targets only rate-limited failures. Set `include_all_failures=True` to also
    re-attempt non-rate-limit failures (e.g. a transient "NoneType is not iterable"), which
    catches stragglers the rate-limit filter would skip — genuine delisted/no_info tickers
    simply fail again harmlessly and are left with their latest error.
    """
    universe = dict(build_universe())

    for attempt in range(1, max_attempts + 1):
        cache = load_fundamentals_cache(cache_path)
        if cache is None:
            print("[fundamentals] no cache to retry from")
            return {}

        rate_limited = [tk for tk, r in cache["tickers"].items()
                        if not r.get("fetch_ok") and _is_rate_limited(r.get("error", ""))]
        if include_all_failures:
            failed = [tk for tk, r in cache["tickers"].items() if not r.get("fetch_ok")]
        else:
            failed = rate_limited
        if not failed:
            print("[fundamentals] no failures remaining to retry")
            return cache

        print(f"\n[retry attempt {attempt}/{max_attempts}] {len(failed)} tickers "
              f"({len(rate_limited)} rate-limited)")
        print(f"  cooldown: {cooldown_sec}s | workers: {max_workers} | delay: {per_request_delay}s")
        time.sleep(cooldown_sec)

        targets = [(tk, universe.get(tk, "Stock")) for tk in failed]

        # Sequential-ish with small worker pool + per-request jitter
        results = {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = []
            for tk, at in targets:
                futures.append(pool.submit(fetch_one, tk, at))
                time.sleep(per_request_delay)  # space out submissions
            for i, fut in enumerate(as_completed(futures)):
                try:
                    r = fut.result()
                    results[r["ticker"]] = r
                except Exception:
                    pass
                if (i + 1) % 20 == 0:
                    print(f"  [retry] {i+1}/{len(targets)} | "
                          f"recovered: {sum(1 for x in results.values() if x.get('fetch_ok'))}")

        # Merge into main cache
        recovered = 0
        for tk, r in results.items():
            if r.get("fetch_ok"):
                cache["tickers"][tk] = r
                recovered += 1
            else:
                # Keep latest error for transparency
                cache["tickers"][tk] = r

        # Update stats + persist
        all_results = cache["tickers"]
        cache["stats"]["stock_ok"] = sum(1 for r in all_results.values()
                                          if r.get("fetch_ok") and r.get("asset_type") == "Stock")
        cache["stats"]["etf_ok"] = sum(1 for r in all_results.values()
                                        if r.get("fetch_ok") and r.get("asset_type") == "ETF")
        cache["stats"]["has_estimates"] = sum(1 for r in all_results.values() if r.get("estimates"))
        cache["stats"]["has_revisions"] = sum(1 for r in all_results.values() if r.get("revisions"))
        cache["stats"]["failed_tickers"] = [tk for tk, r in all_results.items() if not r.get("fetch_ok")]
        cache["stats"]["failed_count"] = len(cache["stats"]["failed_tickers"])

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)

        elapsed = time.time() - t0
        print(f"  [retry attempt {attempt}] recovered {recovered}/{len(targets)} in {elapsed:.0f}s")
        print(f"  total OK: stock={cache['stats']['stock_ok']} etf={cache['stats']['etf_ok']} "
              f"failed={cache['stats']['failed_count']}")

        # Abort ONLY when a round made no progress AND nothing is still rate-limited.
        # If the round recovered nothing but tickers remain throttled (cooldown too
        # short / Yahoo still 429ing), keep going — the next round's cooldown may clear
        # it. This fixes the old behaviour of quitting after one throttled round.
        round_rate_limited = sum(1 for r in results.values()
                                 if not r.get("fetch_ok") and _is_rate_limited(r.get("error", "")))
        if recovered == 0 and round_rate_limited == 0:
            print("  no progress and no rate-limit errors remain — aborting further retries")
            break

    return cache


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Fundamentals cache builder")
    parser.add_argument("--tickers", nargs="*", help="Specific tickers (default: full universe)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    parser.add_argument("--max-age-h", type=float, default=None,
                        help="Skip refresh if cache is younger than this many hours")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry failures from the existing cache (non-destructive merge)")
    parser.add_argument("--retry-cooldown", type=int, default=300,
                        help="Seconds to wait before retrying failed tickers (default 300)")
    parser.add_argument("--include-all-failures", action="store_true",
                        help="With --retry-failed, also retry non-rate-limit failures (catches stragglers)")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Tickers per throttled chunk on a full run (default 100)")
    parser.add_argument("--chunk-cooldown", type=float, default=15.0,
                        help="Seconds to pause between chunks (default 15)")
    parser.add_argument("--no-auto-retry", action="store_true",
                        help="Disable auto retry_failed() at the tail of a full run")
    parser.add_argument("--cache-path", default=CACHE_PATH)
    args = parser.parse_args()

    if args.retry_failed:
        retry_failed(cache_path=args.cache_path, cooldown_sec=args.retry_cooldown,
                     include_all_failures=args.include_all_failures)
        return

    if args.max_age_h is not None:
        age = cache_age_hours(args.cache_path)
        if age is not None and age < args.max_age_h:
            print(f"[fundamentals] cache is {age:.1f}h old (< {args.max_age_h}h) — skipping refresh")
            return

    if args.tickers:
        # Look up asset_type from universe (targeted run: no chunking overhead, no auto-retry)
        full = dict(build_universe())
        targeted = [(tk, full.get(tk, "Stock")) for tk in args.tickers]
        run_pipeline(tickers=targeted, max_workers=args.workers, cache_path=args.cache_path,
                     chunk_size=max(len(targeted), 1), chunk_cooldown=0.0, auto_retry=False)
    else:
        run_pipeline(tickers=None, max_workers=args.workers, cache_path=args.cache_path,
                     chunk_size=args.chunk_size, chunk_cooldown=args.chunk_cooldown,
                     auto_retry=not args.no_auto_retry)


if __name__ == "__main__":
    _cli()
