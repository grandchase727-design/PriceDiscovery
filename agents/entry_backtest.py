# -*- coding: utf-8 -*-
"""entry_backtest.py — Backtest entry methodologies on historical data.

================================================================================
PURPOSE
================================================================================

Compare three entry methodologies on the buy_list universe:
  1. AGGRESSIVE (Composite-based, current price entry)
  2. PRIMARY (CAN SLIM pivot point)
  3. CONSERVATIVE (SMA50 pullback)

For each, simulate:
  - Entry on trigger
  - Hold until O'Neil 7% stop OR +20% target OR N days

Compute per-methodology statistics:
  - Hit rate (% trades profitable)
  - Avg gain, avg loss, expectancy
  - Sharpe-like ratio
  - Days to outcome distribution

================================================================================
USAGE
================================================================================

from agents.entry_backtest import backtest_entry_methods

results = backtest_entry_methods(
    tickers=["AAPL","MSFT","NVDA"],
    lookback_days=180,
    horizon="core",
)
# returns {AGGRESSIVE: {hit_rate, avg_gain, ...}, PRIMARY: {...}, CONSERVATIVE: {...}}
"""
from __future__ import annotations

import json
import math
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

CACHE_PATH = Path(".entry_backtest_cache.json")

# Trade outcome rules (O'Neil-aligned)
STOP_LOSS_PCT = -0.07        # -7% cut-loss
TARGET_GAIN_PCT = 0.20        # +20% target
MAX_HOLD_DAYS = 63             # ~3 months (strategic horizon limit)
TIGHT_HOLD_DAYS = 5            # tactical = 1 week
CORE_HOLD_DAYS = 21            # core = 1 month


def _simulate_trade(entry_idx: int, entry_price: float, highs: list, lows: list,
                     closes: list, max_hold: int = MAX_HOLD_DAYS) -> dict:
    """Simulate a single trade from entry_idx forward.

    Returns: {outcome, exit_idx, exit_price, days_held, pnl_pct}
    outcome: WIN_TARGET / LOSS_STOP / TIMEOUT_GAIN / TIMEOUT_LOSS
    """
    target_price = entry_price * (1 + TARGET_GAIN_PCT)
    stop_price = entry_price * (1 + STOP_LOSS_PCT)
    n = len(closes)
    end_idx = min(n - 1, entry_idx + max_hold)

    for i in range(entry_idx + 1, end_idx + 1):
        # Check stop first (intraday low touched stop)
        if lows[i] <= stop_price:
            return {
                "outcome": "LOSS_STOP",
                "exit_idx": i,
                "exit_price": round(stop_price, 2),
                "days_held": i - entry_idx,
                "pnl_pct": round(STOP_LOSS_PCT * 100, 2),
            }
        # Check target (intraday high touched target)
        if highs[i] >= target_price:
            return {
                "outcome": "WIN_TARGET",
                "exit_idx": i,
                "exit_price": round(target_price, 2),
                "days_held": i - entry_idx,
                "pnl_pct": round(TARGET_GAIN_PCT * 100, 2),
            }

    # Timeout — close at last day's close
    exit_price = closes[end_idx]
    pnl = (exit_price / entry_price - 1) * 100
    outcome = "TIMEOUT_GAIN" if pnl >= 0 else "TIMEOUT_LOSS"
    return {
        "outcome": outcome,
        "exit_idx": end_idx,
        "exit_price": round(exit_price, 2),
        "days_held": end_idx - entry_idx,
        "pnl_pct": round(pnl, 2),
    }


def _find_entry_dates_aggressive(closes: list, lookback: int = 180) -> list[int]:
    """AGGRESSIVE: simulate 'enter now' decisions every 5 days during lookback."""
    n = len(closes)
    start = max(0, n - lookback)
    # Every 5 days within the lookback window
    return list(range(start, n - 10, 5))


def _find_entry_dates_primary(closes: list, highs: list, lows: list, volumes: list,
                                 lookback: int = 180) -> list[int]:
    """PRIMARY: CAN SLIM pivot breakouts detected during lookback."""
    from agents.entry_price import (
        detect_best_base_pattern, ONEIL_PIVOT_BUFFER, VOLUME_CONFIRM_RATIO,
    )
    n = len(closes)
    entry_dates: list[int] = []
    # Sliding window: detect a pattern using 120 days of history, see if today breaks the pivot
    for i in range(120, n - 10):
        sub_c = closes[max(0,i-120):i]
        sub_h = highs[max(0,i-120):i]
        sub_l = lows[max(0,i-120):i]
        sub_v = volumes[max(0,i-120):i]
        base = detect_best_base_pattern(sub_c, sub_h, sub_l, sub_v)
        if not base: continue
        pivot = base.get("pivot")
        if pivot is None: continue
        # Today's close > pivot + buffer
        if closes[i] > pivot:
            # Volume confirmation
            avg_v = sum(sub_v[-50:-5]) / max(1, 45) if len(sub_v) >= 50 else 0
            today_v = volumes[i]
            if avg_v > 0 and today_v / avg_v >= VOLUME_CONFIRM_RATIO:
                entry_dates.append(i)
    return entry_dates


def _find_entry_dates_conservative(closes: list, lookback: int = 180) -> list[int]:
    """CONSERVATIVE: SMA50 reclaim from below."""
    n = len(closes)
    entries: list[int] = []
    if n < 60: return entries
    # Compute SMA50 series
    sma50 = []
    for i in range(n):
        if i < 49:
            sma50.append(None)
        else:
            sma50.append(sum(closes[i-49:i+1]) / 50)
    # Find places where price crosses above SMA50 from below
    for i in range(max(60, n - lookback), n - 10):
        if sma50[i-1] is None or sma50[i] is None: continue
        if closes[i-1] < sma50[i-1] and closes[i] >= sma50[i]:
            entries.append(i)
    return entries


def backtest_entry_methods(tickers: list[str], lookback_days: int = 180,
                              max_hold_days: int = MAX_HOLD_DAYS,
                              verbose: bool = False) -> dict:
    """Run full backtest across methodologies.

    Returns:
        {
          AGGRESSIVE: {hit_rate, avg_gain, avg_loss, expectancy, n_trades, ...},
          PRIMARY: {...},
          CONSERVATIVE: {...},
        }
    """
    import yfinance as yf

    all_results = {
        "AGGRESSIVE": [],
        "PRIMARY": [],
        "CONSERVATIVE": [],
    }

    for ticker in tickers:
        if verbose: print(f"  Backtesting {ticker}...")
        try:
            df = yf.download(ticker, period="2y", interval="1d",
                              progress=False, auto_adjust=True)
            if df.empty or len(df) < 200: continue
            if hasattr(df.columns, "get_level_values"):
                try: df.columns = df.columns.get_level_values(0)
                except: pass
            df = df.dropna(subset=["High","Low","Close","Volume"])
            if len(df) < 200: continue
            highs = df["High"].values.tolist()
            lows  = df["Low"].values.tolist()
            closes = df["Close"].values.tolist()
            volumes = df["Volume"].values.tolist()

            # Method 1: AGGRESSIVE
            aggressive_dates = _find_entry_dates_aggressive(closes, lookback_days)
            for d in aggressive_dates:
                trade = _simulate_trade(d, closes[d], highs, lows, closes, max_hold_days)
                all_results["AGGRESSIVE"].append({**trade, "ticker": ticker})

            # Method 2: PRIMARY (CAN SLIM)
            primary_dates = _find_entry_dates_primary(closes, highs, lows, volumes, lookback_days)
            for d in primary_dates:
                trade = _simulate_trade(d, closes[d], highs, lows, closes, max_hold_days)
                all_results["PRIMARY"].append({**trade, "ticker": ticker})

            # Method 3: CONSERVATIVE (SMA50 reclaim)
            cons_dates = _find_entry_dates_conservative(closes, lookback_days)
            for d in cons_dates:
                trade = _simulate_trade(d, closes[d], highs, lows, closes, max_hold_days)
                all_results["CONSERVATIVE"].append({**trade, "ticker": ticker})

            time.sleep(0.3)  # yfinance rate-limit
        except Exception as e:
            if verbose: print(f"    ✗ {ticker}: {e}")
            continue

    # Compute stats per method
    stats = {}
    for method, trades in all_results.items():
        if not trades:
            stats[method] = {"n_trades": 0}
            continue
        n = len(trades)
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        hit_rate = len(wins) / n
        avg_gain = sum(t["pnl_pct"] for t in wins) / max(1, len(wins))
        avg_loss = sum(t["pnl_pct"] for t in losses) / max(1, len(losses))
        avg_days = sum(t["days_held"] for t in trades) / n
        expectancy = hit_rate * avg_gain + (1 - hit_rate) * avg_loss
        # Outcome breakdown
        from collections import Counter
        outcomes = Counter(t["outcome"] for t in trades)
        stats[method] = {
            "n_trades": n,
            "hit_rate": round(hit_rate * 100, 1),
            "avg_gain_pct": round(avg_gain, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "expectancy_pct": round(expectancy, 2),
            "avg_days_held": round(avg_days, 1),
            "outcomes": dict(outcomes),
            "best_trade": max(trades, key=lambda t: t["pnl_pct"]),
            "worst_trade": min(trades, key=lambda t: t["pnl_pct"]),
        }
    return stats


def cache_results(stats: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps({
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "stats": stats,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception: pass


def load_cached_results() -> Optional[dict]:
    if not CACHE_PATH.exists(): return None
    try: return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception: return None
