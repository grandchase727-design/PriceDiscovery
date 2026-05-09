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

# This file lives in pipelines/ — project root is one level up.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
from price_discovery import GLOBAL_ETF_UNIVERSE, STOCK_UNIVERSE

CACHE_PATH = os.path.join(_PROJECT_ROOT, ".fundamentals_cache.pkl")

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

def build_universe() -> list[tuple[str, str]]:
    """Returns list of (ticker, asset_type) covering full universe."""
    universe = []
    for cat, data in GLOBAL_ETF_UNIVERSE.items():
        for tk in data["tickers"].keys():
            universe.append((tk, "ETF"))
    for cat, data in STOCK_UNIVERSE.items():
        for tk in data["tickers"].keys():
            universe.append((tk, "Stock"))
    # Dedupe (in case a ticker appears in both)
    seen = set()
    uniq = []
    for tk, at in universe:
        if tk not in seen:
            seen.add(tk)
            uniq.append((tk, at))
    return uniq


def run_pipeline(
    tickers: Optional[list[tuple[str, str]]] = None,
    max_workers: int = 8,
    cache_path: str = CACHE_PATH,
    progress_every: int = 25,
) -> dict:
    """Fetch fundamentals in parallel and save cache."""
    if tickers is None:
        tickers = build_universe()

    started = datetime.now(timezone.utc)
    print(f"[fundamentals] starting batch: {len(tickers)} tickers, {max_workers} workers")

    results: dict[str, dict] = {}
    failed: list[str] = []
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_tk = {pool.submit(fetch_one, tk, at): tk for tk, at in tickers}
        for future in as_completed(future_to_tk):
            tk = future_to_tk[future]
            try:
                r = future.result()
                results[tk] = r
                if not r["fetch_ok"]:
                    failed.append(tk)
            except Exception as e:
                failed.append(tk)
                results[tk] = {
                    "ticker": tk, "fetch_ok": False, "error": str(e)[:200],
                }
            done += 1
            if done % progress_every == 0 or done == len(tickers):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(tickers) - done) / rate if rate else 0
                print(f"[fundamentals] {done}/{len(tickers)} "
                      f"({100*done/len(tickers):.0f}%) "
                      f"| {rate:.1f} tk/s | ETA {eta:.0f}s "
                      f"| failed={len(failed)}")

    duration = time.time() - t0
    stock_ok = sum(1 for r in results.values() if r.get("fetch_ok") and r.get("asset_type") == "Stock")
    etf_ok = sum(1 for r in results.values() if r.get("fetch_ok") and r.get("asset_type") == "ETF")
    has_estimates = sum(1 for r in results.values() if r.get("estimates"))
    has_revisions = sum(1 for r in results.values() if r.get("revisions"))

    cache = {
        "fetched_at": started.isoformat(),
        "tickers": results,
        "stats": {
            "total_attempted": len(tickers),
            "stock_ok": stock_ok,
            "etf_ok": etf_ok,
            "has_estimates": has_estimates,
            "has_revisions": has_revisions,
            "failed_count": len(failed),
            "failed_tickers": failed,
            "duration_sec": round(duration, 1),
        },
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"\n[fundamentals] DONE — saved to {cache_path}")
    print(f"  Total: {len(tickers)} | Stock OK: {stock_ok} | ETF OK: {etf_ok}")
    print(f"  With estimates: {has_estimates} | With revisions: {has_revisions}")
    print(f"  Failed: {len(failed)} | Duration: {duration:.0f}s")
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
) -> dict:
    """Reload cache, retry rate-limited failures with conservative settings.

    Iterates up to `max_attempts` times — each round waits `cooldown_sec`
    before retrying any tickers still rate-limited.
    """
    universe = dict(build_universe())

    for attempt in range(1, max_attempts + 1):
        cache = load_fundamentals_cache(cache_path)
        if cache is None:
            print("[fundamentals] no cache to retry from")
            return {}

        failed = [tk for tk, r in cache["tickers"].items()
                  if not r.get("fetch_ok") and _is_rate_limited(r.get("error", ""))]
        if not failed:
            print("[fundamentals] no rate-limited failures remaining")
            return cache

        print(f"\n[retry attempt {attempt}/{max_attempts}] {len(failed)} rate-limited tickers")
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

        if recovered == 0:
            print("  no progress — aborting further retries")
            break

    return cache


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Fundamentals cache builder")
    parser.add_argument("--tickers", nargs="*", help="Specific tickers (default: full universe)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default 8)")
    parser.add_argument("--max-age-h", type=float, default=None,
                        help="Skip refresh if cache is younger than this many hours")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry only the rate-limited failures from the existing cache")
    parser.add_argument("--retry-cooldown", type=int, default=300,
                        help="Seconds to wait before retrying rate-limited tickers (default 300)")
    parser.add_argument("--cache-path", default=CACHE_PATH)
    args = parser.parse_args()

    if args.retry_failed:
        retry_failed(cache_path=args.cache_path, cooldown_sec=args.retry_cooldown)
        return

    if args.max_age_h is not None:
        age = cache_age_hours(args.cache_path)
        if age is not None and age < args.max_age_h:
            print(f"[fundamentals] cache is {age:.1f}h old (< {args.max_age_h}h) — skipping refresh")
            return

    if args.tickers:
        # Look up asset_type from universe
        full = dict(build_universe())
        targeted = [(tk, full.get(tk, "Stock")) for tk in args.tickers]
    else:
        targeted = None

    run_pipeline(tickers=targeted, max_workers=args.workers, cache_path=args.cache_path)


if __name__ == "__main__":
    _cli()
