"""
etf_holdings_pipeline.py — ETF holdings cache builder (Phase A Hybrid Bottom-up).

Fetches top-N holdings for equity sector/thematic ETFs via yfinance's
`Ticker.get_funds_data().top_holdings` (~top 10 per ETF, with weights).

Output: .etf_holdings_cache.json
  {
      "fetched_at": "2026-05-13T...",
      "etfs": {
          "SMH": {
              "name": "VanEck Semiconductor",
              "holdings": [
                  {"ticker": "NVDA", "name": "NVIDIA Corp", "weight": 0.181},
                  {"ticker": "TSM",  "name": "Taiwan Semi...","weight": 0.106},
                  ...
              ],
              "n_holdings": 10,
              "top10_weight_sum": 0.872,
              "as_of": "2026-05-13"
          }, ...
      }
  }

The api.py layer joins this with /api/table to compute breadth/weighted/
concentration metrics per ETF.

CLI:
  python3 etf_holdings_pipeline.py            # full refresh
  python3 etf_holdings_pipeline.py --etfs SMH,XLK,ARKK   # subset
  python3 etf_holdings_pipeline.py --skip-existing       # incremental
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

OUTPUT_PATH = ".etf_holdings_cache.json"

# Equity sector / thematic ETFs where bottom-up makes sense.
# Excluded: Fixed Income (FI_*), Macro (Commodities/Currency/REITs partly),
#           Leveraged/Inverse, Currency-Hedged, International broad index.
# International country ETFs (EWY/EWG/etc.) — holdings exist but constituents
# rarely overlap with our 538-stock universe → low information yield, skipped.
ETF_TARGETS = [
    # ── SPDR Select Sector ETFs (11) ────────────────────────────────────
    "XLK",  # Technology
    "XLF",  # Financials
    "XLV",  # Healthcare
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLI",  # Industrials
    "XLE",  # Energy
    "XLB",  # Materials
    "XLU",  # Utilities
    "XLRE", # Real Estate
    "XLC",  # Communication Services
    # ── iShares Sector ETFs ──────────────────────────────────────────────
    "IYR",  # Real Estate
    # ── Semiconductors / Tech sub-themes ─────────────────────────────────
    "SMH",  # VanEck Semi
    "SOXX", # iShares Semi
    "XSD",  # SPDR Semi (equal weight)
    "IGV",  # Software
    "SKYY", # Cloud
    "CLOU", # Cloud
    "FINX", # Fintech
    "AIQ",  # AI & Big Data
    "ROBO", # Robotics
    "ARKW", # Next Gen Internet
    "ARKK", # Disruptive Innovation
    "ARKG", # Genomic Revolution
    # ── Energy / Clean Energy ────────────────────────────────────────────
    "TAN",  # Solar
    "ICLN", # Clean Energy
    "QCLN", # Green Energy
    "LIT",  # Lithium & Battery
    "BATT", # Battery
    "URA",  # Uranium
    "URNM", # Uranium Miners
    "NUKZ", # Nuclear
    "OIH",  # Oil Services
    "XOP",  # Oil & Gas E&P
    "FCG",  # Natural Gas E&P
    # ── Mining / Materials ───────────────────────────────────────────────
    "PICK", # Metal Mining
    "GDX",  # Gold Miners
    "GDXJ", # Junior Gold Miners
    "SIL",  # Silver Miners
    "SILJ", # Junior Silver Miners
    "COPX", # Copper Miners
    "REMX", # Rare Earth
    "XME",  # Metals & Mining
    "GUNR", # Natural Resources
    # ── Defense / Aerospace ──────────────────────────────────────────────
    "SHLD", # Global Defense
    # ── Factor / Style ETFs ──────────────────────────────────────────────
    "MTUM", # Momentum
    "QUAL", # Quality
    "VLUE", # Value Factor
    "USMV", # Min Vol
    "SCHD", # Dividend Growth
    "NOBL", # Dividend Aristocrats
    "MOAT", # Wide Moat
    "COWZ", # Free Cash Flow Yield
    "SPHQ", # S&P 500 Quality
    "SPMO", # S&P 500 Momentum
    # ── Broad market ─────────────────────────────────────────────────────
    "SPY", "VOO", "IVV",  # S&P 500
    "QQQ", "QQQM",        # Nasdaq-100
    "IWM",                # Russell 2000
    "DIA",                # Dow
    "VTI",                # Total Market
    "RSP",                # S&P 500 Equal Weight
    # ── International / Country (limited overlap with our stock universe) ─
    "EWY",  # Korea (overlaps with our KR stocks)
    "EWJ",  # Japan
    "INDA", # India
    "MCHI", # China
    "KWEB", # China Internet
    "EWT",  # Taiwan
]


def fetch_holdings(ticker: str, retries: int = 1) -> Optional[Dict]:
    """Fetch top holdings for a single ETF via yfinance.

    Returns dict with `holdings` list and metadata, or None on failure.
    """
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            # ETF metadata
            try:
                info = t.info or {}
                name = info.get("longName") or info.get("shortName") or ticker
            except Exception:
                name = ticker

            # Top holdings (typically ~top 10)
            fd = t.get_funds_data()
            if fd is None or fd.top_holdings is None:
                return None
            df = fd.top_holdings
            if df is None or df.empty:
                return None

            holdings = []
            top10_sum = 0.0
            for sym, row in df.iterrows():
                # Holding Percent is 0-1 decimal
                w = float(row.get("Holding Percent", 0) or 0)
                if w <= 0:
                    continue
                holdings.append({
                    "ticker": str(sym),
                    "name": str(row.get("Name", "")),
                    "weight": round(w, 5),
                })
                top10_sum += w
            return {
                "name": name,
                "holdings": holdings,
                "n_holdings": len(holdings),
                "top10_weight_sum": round(top10_sum, 4),
                "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(2)
                continue
            return None
    return None


def fetch_universe(
    tickers: List[str],
    out_path: str = OUTPUT_PATH,
    throttle_sec: float = 0.6,
    skip_existing: bool = False,
) -> Dict:
    """Fetch holdings for all tickers, save to JSON cache."""
    existing: Dict[str, Dict] = {}
    if skip_existing and os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = (json.load(f) or {}).get("etfs", {}) or {}
        except Exception:
            existing = {}

    out: Dict[str, Dict] = dict(existing)
    n_success, n_fail, n_skip = 0, 0, 0
    print(f"Fetching holdings for {len(tickers)} ETFs (skip_existing={skip_existing})...")

    for i, tk in enumerate(tickers, 1):
        if skip_existing and tk in existing and existing[tk].get("n_holdings", 0) > 0:
            n_skip += 1
            continue
        result = fetch_holdings(tk)
        if result and result["n_holdings"] > 0:
            out[tk] = result
            n_success += 1
            print(f"  [{i:>3}/{len(tickers)}] {tk:<8} ✓ {result['n_holdings']} holdings, top10={result['top10_weight_sum']*100:.1f}%")
        else:
            n_fail += 1
            print(f"  [{i:>3}/{len(tickers)}] {tk:<8} ✗ no data")
        time.sleep(throttle_sec)

    payload = {
        "fetched_at": datetime.utcnow().isoformat(),
        "n_success": n_success,
        "n_fail": n_fail,
        "n_skip": n_skip,
        "etfs": out,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Wrote {out_path} — success={n_success}, fail={n_fail}, skip={n_skip}, total_cached={len(out)}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etfs", type=str, default=None,
                        help="Comma-separated tickers (default: built-in ETF_TARGETS)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip tickers already in cache")
    parser.add_argument("--throttle", type=float, default=0.6,
                        help="Seconds between API calls")
    args = parser.parse_args()

    if args.etfs:
        tickers = [t.strip().upper() for t in args.etfs.split(",") if t.strip()]
    else:
        tickers = ETF_TARGETS

    fetch_universe(tickers, throttle_sec=args.throttle, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
