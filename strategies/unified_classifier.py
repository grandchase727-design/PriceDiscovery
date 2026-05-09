"""
unified_classifier.py — Cross-border stock classification (Phase Y).

Unifies US + KR + (and globally any yfinance-supported) tickers under a
single 4-level taxonomy:

    Level 1 (Asset class):    Equity / ETF / etc.   (already in cache.category)
    Level 2 (Geo):            US / KR / etc.        (derived from suffix)
    Level 3 (GICS Sector):    11 sectors            (NEW — auto from yfinance)
    Level 4 (SubTheme):       105 themes            (already in cache.theme)

Adds two new dimensions:
    GICS_Industry             — yfinance industry label (74 industries)
    Cap_Tier                  — MEGA / LARGE / MID / SMALL (USD market cap)

Strategy: yfinance.info as the unified source. Works for both US and KR
listings (KOSPI 200 + most KOSDAQ tickers). Pyfx normalization layer
converts yfinance sector names to GICS standard.

Output: .unified_classification.json
    {
        "as_of": "...",
        "n_total": ...,
        "n_success": ...,
        "tickers": {
            "AAPL":     {gics_sector, gics_industry, country, cap_tier, mktcap_usd_b, name, ...},
            "005930.KS": {...},
            ...
        }
    }

Run:
    python3 unified_classifier.py                 # fetch all 756 tickers
    python3 unified_classifier.py --tickers AAPL,NVDA,005930.KS   # subset
    python3 unified_classifier.py --validate       # cross-check vs curated
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import yfinance as yf

CACHE_PATH = ".scan_cache.pkl"
OUTPUT_PATH = ".unified_classification.json"

# ───────────────────────────────────────────────────────────────
# Mapping tables
# ───────────────────────────────────────────────────────────────

# yfinance returns slightly different sector names than GICS standard.
# Normalize to GICS 11-sector taxonomy.
YFINANCE_TO_GICS: Dict[str, str] = {
    "Technology":             "Information Technology",
    "Communication Services": "Communication Services",
    "Consumer Cyclical":      "Consumer Discretionary",
    "Consumer Defensive":     "Consumer Staples",
    "Energy":                 "Energy",
    "Financial Services":     "Financials",
    "Healthcare":             "Health Care",
    "Industrials":            "Industrials",
    "Basic Materials":        "Materials",
    "Real Estate":            "Real Estate",
    "Utilities":              "Utilities",
}

# Approximate FX rates for converting non-USD market cap to USD.
# (Rough — refresh quarterly via FRED if precision matters.)
FX_TO_USD: Dict[str, float] = {
    "USD": 1.0,
    "KRW": 1.0 / 1380.0,    # ~1380 KRW/USD
    "JPY": 1.0 / 155.0,
    "EUR": 1.10,
    "GBP": 1.27,
    "HKD": 1.0 / 7.85,
    "CNY": 1.0 / 7.20,
    "INR": 1.0 / 83.0,
}

# Cap tier thresholds (USD)
CAP_TIER_THRESHOLDS = [
    (200e9, "MEGA"),      # ≥ $200B
    (10e9,  "LARGE"),     # $10B - $200B
    (2e9,   "MID"),       # $2B - $10B
    (0,     "SMALL"),     # < $2B
]


def _yfinance_to_gics(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    return YFINANCE_TO_GICS.get(label.strip(), label.strip())


def _to_usd(amount: float, currency: str) -> float:
    rate = FX_TO_USD.get(currency.upper(), 1.0) if currency else 1.0
    return float(amount) * rate


def _cap_tier(mktcap_usd: float) -> str:
    for threshold, label in CAP_TIER_THRESHOLDS:
        if mktcap_usd >= threshold:
            return label
    return "MICRO"


def _country_from_ticker(ticker: str, country_field: str = "") -> str:
    """Infer geographic listing country from ticker suffix."""
    if ticker.endswith(".KS") or ticker.endswith(".KQ"):
        return "KR"
    if ticker.endswith(".T"):
        return "JP"
    if ticker.endswith(".HK"):
        return "HK"
    if ticker.endswith(".SS") or ticker.endswith(".SZ"):
        return "CN"
    if ticker.endswith(".PA") or ticker.endswith(".DE") or ticker.endswith(".AS") \
            or ticker.endswith(".MI") or ticker.endswith(".L") or ticker.endswith(".SW"):
        return "EU"
    # Fallback to yfinance country field
    if country_field:
        if "Korea" in country_field:
            return "KR"
        if "United States" in country_field:
            return "US"
        if "Japan" in country_field:
            return "JP"
        if "China" in country_field:
            return "CN"
    return "US"


# ───────────────────────────────────────────────────────────────
# Single-ticker classification
# ───────────────────────────────────────────────────────────────

def classify_ticker(ticker: str, retry: int = 1) -> Dict[str, Any]:
    """Fetch + normalize a single ticker. Returns Dict; on failure all
    classification fields are None but ticker/source are populated."""
    out: Dict[str, Any] = {
        "ticker": ticker,
        "name": None,
        "country": _country_from_ticker(ticker),
        "currency": None,
        "yfinance_sector": None,
        "gics_sector": None,
        "gics_industry": None,
        "mktcap_usd_b": None,
        "cap_tier": None,
        "exchange": None,
        "source": "yfinance.info",
        "ok": False,
    }
    for _ in range(retry + 1):
        try:
            info = yf.Ticker(ticker).info
            if not info or len(info) < 5:
                continue
            yf_sec = info.get("sector")
            ind = info.get("industry")
            cur = info.get("currency", "USD") or "USD"
            mc = info.get("marketCap") or 0
            mc_usd = _to_usd(mc, cur) if mc else 0
            out.update({
                "name": info.get("longName") or info.get("shortName"),
                "country": _country_from_ticker(ticker, info.get("country", "")),
                "currency": cur,
                "yfinance_sector": yf_sec,
                "gics_sector": _yfinance_to_gics(yf_sec),
                "gics_industry": ind,
                "mktcap_usd_b": round(mc_usd / 1e9, 2) if mc_usd else None,
                "cap_tier": _cap_tier(mc_usd) if mc_usd else None,
                "exchange": info.get("exchange"),
                "ok": True,
            })
            return out
        except Exception as e:
            out["error"] = str(e)[:100]
            time.sleep(0.5)
    return out


# ───────────────────────────────────────────────────────────────
# Batch classification
# ───────────────────────────────────────────────────────────────

def classify_universe(tickers: List[str], throttle_sec: float = 0.4,
                       progress_every: int = 25) -> Dict[str, Any]:
    """Fetch all tickers with throttling. Returns full classification dict."""
    n = len(tickers)
    print(f"Classifying {n} tickers (throttle {throttle_sec}s between calls)...")
    results: Dict[str, Dict[str, Any]] = {}
    n_ok = 0
    t0 = time.time()
    for i, tk in enumerate(tickers, 1):
        c = classify_ticker(tk)
        results[tk] = c
        if c.get("ok"):
            n_ok += 1
        if i % progress_every == 0 or i == n:
            elapsed = time.time() - t0
            eta = elapsed / i * (n - i)
            print(f"  [{i}/{n}]  ok {n_ok}/{i}  ({i/elapsed*60:.0f} t/min)  ETA {eta/60:.1f}m")
        if i < n:
            time.sleep(throttle_sec)

    return {
        "as_of": datetime.utcnow().isoformat(),
        "n_total": n,
        "n_success": n_ok,
        "n_failure": n - n_ok,
        "throttle_sec": throttle_sec,
        "tickers": results,
    }


# ───────────────────────────────────────────────────────────────
# Validation against curated taxonomy
# ───────────────────────────────────────────────────────────────

def _curated_sector_from_cache(r: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of curated 'sector' label from existing scan
    record. Falls back to category prefix mapping for STK_ entries."""
    s = r.get("sector")
    if s:
        return s
    cat = (r.get("category") or "")
    if cat.startswith("ETF") or not cat.startswith("STK"):
        return None
    # Map STK_<X> → human-readable sector (rough)
    suffix = cat.replace("STK_", "")
    return {
        "Technology": "Information Technology",
        "Healthcare": "Health Care",
        "Financials": "Financials",
        "Discretionary": "Consumer Discretionary",
        "Staples": "Consumer Staples",
        "Industrials": "Industrials",
        "CommServices": "Communication Services",
        "Energy": "Energy",
        "Materials": "Materials",
        "Utilities": "Utilities",
        "RealEstate": "Real Estate",
        "Korea": None,  # Korean stocks have varied sectors — defer to auto
    }.get(suffix)


def validate(classification: Dict[str, Any], cache_path: str = CACHE_PATH
             ) -> Dict[str, Any]:
    """Compare auto-classification vs curated cache labels.

    Returns:
        {
            "n_total": ...,
            "n_compared": ...,
            "n_agree": ...,
            "agreement_pct": ...,
            "mismatches": [ {ticker, curated, auto, ...}, ... ],
            "by_country": {"US": {...}, "KR": {...}}
        }
    """
    if not os.path.exists(cache_path):
        return {"error": "scan cache not found"}
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    by_ticker = {r["ticker"]: r for r in cache.get("results", [])}

    tickers_data = classification.get("tickers", {})
    rows: List[Dict[str, Any]] = []
    n_compared = 0
    n_agree = 0
    by_country: Dict[str, Dict[str, int]] = {}

    for tk, c in tickers_data.items():
        cached = by_ticker.get(tk)
        if not cached or not c.get("ok"):
            continue
        curated = _curated_sector_from_cache(cached)
        auto = c.get("gics_sector")
        if curated is None:
            continue  # KR stocks (no curated sector) — skip strict comparison
        n_compared += 1
        country = c.get("country", "?")
        bc = by_country.setdefault(country, {"compared": 0, "agree": 0})
        bc["compared"] += 1
        if curated == auto:
            n_agree += 1
            bc["agree"] += 1
        else:
            rows.append({
                "ticker": tk,
                "name": c.get("name") or cached.get("name"),
                "country": country,
                "curated": curated,
                "auto_gics": auto,
                "auto_industry": c.get("gics_industry"),
                "category": cached.get("category"),
                "subtheme": cached.get("theme"),
                "mktcap_usd_b": c.get("mktcap_usd_b"),
                "cap_tier": c.get("cap_tier"),
            })

    agree_pct = round(100.0 * n_agree / max(n_compared, 1), 1)
    for country, d in by_country.items():
        d["agree_pct"] = round(100.0 * d["agree"] / max(d["compared"], 1), 1)

    return {
        "n_total": len(tickers_data),
        "n_compared": n_compared,
        "n_agree": n_agree,
        "agreement_pct": agree_pct,
        "by_country": by_country,
        "mismatches": sorted(rows, key=lambda r: -(r.get("mktcap_usd_b") or 0)),
    }


# ───────────────────────────────────────────────────────────────
# CLI / entry
# ───────────────────────────────────────────────────────────────

def get_universe_from_cache(cache_path: str = CACHE_PATH,
                              skip_etf: bool = False) -> List[str]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found.")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    tickers = []
    for r in cache.get("results", []):
        cat = r.get("category", "")
        if skip_etf and (not cat.startswith("STK")):
            continue
        tickers.append(r["ticker"])
    return tickers


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", help="Comma-separated subset (default: all from cache)")
    p.add_argument("--cache", default=CACHE_PATH)
    p.add_argument("--output", default=OUTPUT_PATH)
    p.add_argument("--throttle", type=float, default=0.4)
    p.add_argument("--validate", action="store_true",
                    help="Run validation against curated cache after fetch")
    p.add_argument("--skip-etf", action="store_true",
                    help="Stocks only (no ETFs)")
    p.add_argument("--reuse", action="store_true",
                    help="Reuse existing .unified_classification.json (skip fetch)")
    args = p.parse_args()

    if args.tickers:
        universe = [tk.strip() for tk in args.tickers.split(",") if tk.strip()]
    else:
        universe = get_universe_from_cache(args.cache, skip_etf=args.skip_etf)
    print(f"Universe size: {len(universe)}")

    if args.reuse and os.path.exists(args.output):
        with open(args.output) as f:
            classification = json.load(f)
        print(f"Loaded existing {args.output}")
    else:
        classification = classify_universe(universe, throttle_sec=args.throttle)
        with open(args.output, "w") as f:
            json.dump(classification, f, indent=2, default=str)
        print(f"\n✓ Wrote {args.output}")
        print(f"  Total: {classification['n_total']}  ·  "
              f"OK: {classification['n_success']}  ·  "
              f"Failed: {classification['n_failure']}")

    if args.validate:
        report = validate(classification, args.cache)
        print(f"\n══ Validation: curated vs auto-GICS ══")
        print(f"  Total:    {report['n_total']}")
        print(f"  Compared: {report['n_compared']} (excluding KR / non-stock)")
        print(f"  Agree:    {report['n_agree']}  ({report['agreement_pct']}%)")
        print(f"\n  By country:")
        for c, d in report["by_country"].items():
            print(f"    {c}: {d['agree']}/{d['compared']} ({d.get('agree_pct', 0)}%)")
        print(f"\n  Top 20 mismatches (by market cap):")
        for r in report["mismatches"][:20]:
            cap_s = f"${r['mktcap_usd_b']:.1f}B" if r.get("mktcap_usd_b") else "?"
            print(f"    {r['ticker']:<8} {r.get('name', '')[:30]:<30} "
                  f"curated={r['curated']:<25} auto={r['auto_gics']:<25} ({cap_s})")
