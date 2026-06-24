# -*- coding: utf-8 -*-
"""entry_price.py — 3-tier Entry Price calculator with CAN SLIM + Elliott + SMA50.

================================================================================
PURPOSE
================================================================================

For each buy_list pick, compute three entry levels:
  - AGGRESSIVE  : current price (system signal strong enough to enter NOW)
  - PRIMARY     : CAN SLIM pivot point (cup-with-handle / flat-base / double-bottom)
                  with O'Neil's "high of base + $0.10" rule
  - CONSERVATIVE: SMA50 pullback OR Elliott Wave 4 retracement (lower-risk entry)

================================================================================
METHODOLOGY
================================================================================

1. CAN SLIM (William O'Neil):
   - Base patterns: cup-with-handle (5-65 weeks), flat-base (5+ weeks),
                    double-bottom (W-shape)
   - Pivot point = base resistance + $0.10 (or +0.1% for low-priced)
   - Volume confirmation: ≥1.4x avg on breakout day
   - "M" condition: SPX above 50-day SMA (market direction)

2. Elliott Wave Fibonacci entries:
   - Wave 2 retracement: 38.2-61.8% of Wave 1 (entry zone for Wave 3)
   - Wave 4 retracement: 23.6-38.2% of Wave 3 (entry zone for Wave 5)

3. SMA50 reclaim entry:
   - Pullback to SMA50 + 1% (low-risk trend-continuation entry)
   - Combined with volume confirmation

4. Composite-based gating:
   - Composite < 55 → SKIP (no entry recommended)
   - Composite 55-65 → CONSERVATIVE only
   - Composite 65-75 → PRIMARY + CONSERVATIVE
   - Composite 75+ → All three tiers

================================================================================
USAGE
================================================================================

from agents.entry_price import compute_entry_for_ticker, annotate_buy_list_with_entries

# Single ticker
result = compute_entry_for_ticker("AAPL", "core", scan_row)
# returns: {aggressive, primary, conservative, base_pattern, volume_ok, ...}

# Whole buy_list
annotate_buy_list_with_entries(buy_list)
# augments each pick with entry_aggressive, entry_primary, entry_conservative
"""
from __future__ import annotations

import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

CACHE_PATH = Path(".entry_prices_cache.json")
CACHE_TTL_HOURS = 24

# CAN SLIM constants
ONEIL_PIVOT_BUFFER = 0.10          # Add $0.10 to base high
VOLUME_CONFIRM_RATIO = 1.4         # Volume must be ≥1.4x avg
CUP_WITH_HANDLE_MIN_WEEKS = 5      # Cup minimum 5 weeks
CUP_WITH_HANDLE_MAX_WEEKS = 65     # Cup maximum 65 weeks
CUP_MAX_DEPTH = -0.33              # Max cup depth -33%
CUP_MIN_DEPTH = -0.12              # Min cup depth -12%
HANDLE_MIN_WEEKS = 1
HANDLE_MAX_WEEKS = 4
HANDLE_MAX_DEPTH = -0.12           # Handle max depth -12%
HANDLE_TYPICAL_DEPTH = -0.08       # Handle typical -8%
FLAT_BASE_MIN_WEEKS = 5
FLAT_BASE_MAX_DEPTH = -0.15        # Flat base max depth -15%
ONEIL_CUT_LOSS_PCT = -0.07         # -7% absolute stop (O'Neil rule)


# ─────────────────────────────────────────────────────────────────
# Currency detection (shared with elliott_wave_stops.py)
# ─────────────────────────────────────────────────────────────────

def _detect_currency(ticker: str) -> tuple[str, str]:
    """Return (currency_code, currency_symbol) for a ticker."""
    suffix_to_currency = {
        ".KS": "KRW", ".KQ": "KRW", ".T": "JPY", ".L": "GBP",
        ".HK": "HKD", ".SS": "CNY", ".SZ": "CNY", ".SI": "SGD",
        ".AX": "AUD", ".TO": "CAD", ".PA": "EUR", ".DE": "EUR",
    }
    currency = "USD"
    for sfx, ccy in suffix_to_currency.items():
        if ticker.upper().endswith(sfx):
            currency = ccy; break
    sym = {"USD": "$", "KRW": "₩", "JPY": "¥", "EUR": "€",
            "GBP": "£", "HKD": "HK$", "CNY": "¥",
            "SGD": "S$", "AUD": "A$", "CAD": "C$"}.get(currency, currency + " ")
    return currency, sym


# ─────────────────────────────────────────────────────────────────
# CAN SLIM Base Pattern Detection
# ─────────────────────────────────────────────────────────────────

def _to_weekly(daily_closes: list, daily_highs: list, daily_lows: list,
                daily_volume: list) -> tuple:
    """Aggregate daily to weekly (Mon-Fri) for base pattern detection.

    O'Neil patterns are measured in WEEKS, not days. Simple grouping of 5 days.
    """
    n = len(daily_closes)
    weeks = n // 5
    if weeks == 0:
        return [], [], [], []
    w_close = []
    w_high  = []
    w_low   = []
    w_vol   = []
    for i in range(weeks):
        seg_c = daily_closes[i*5:(i+1)*5]
        seg_h = daily_highs[i*5:(i+1)*5]
        seg_l = daily_lows[i*5:(i+1)*5]
        seg_v = daily_volume[i*5:(i+1)*5]
        w_close.append(seg_c[-1])
        w_high.append(max(seg_h))
        w_low.append(min(seg_l))
        w_vol.append(sum(seg_v))
    return w_close, w_high, w_low, w_vol


def detect_cup_with_handle(weekly_closes: list, weekly_highs: list,
                             weekly_lows: list, weekly_volume: list) -> Optional[dict]:
    """Detect O'Neil's Cup-with-Handle pattern.

    Criteria (strict):
    1. Prior uptrend: ≥30% rise in last 30 weeks before cup starts
    2. Cup: 5-65 weeks, depth -12% to -33%, U-shape (not V)
    3. Right side recovers to ≥90% of left side high
    4. Handle: 1-4 weeks, depth -5% to -12%
    5. Handle high < cup left side high

    Returns: {type, cup_start_idx, cup_low_idx, cup_right_idx, handle_start_idx,
              handle_low_idx, pivot, cup_depth, handle_depth, quality (A/B/C),
              volume_dryup_in_handle, ...} or None if no pattern.
    """
    n = len(weekly_highs)
    if n < CUP_WITH_HANDLE_MIN_WEEKS + 5:   # need at least cup + some context
        return None

    # Look for cup in the last 80 weeks (or all available)
    search_start = max(0, n - 80)
    best: Optional[dict] = None

    # Iterate possible cup left-side highs (peak from which cup starts dropping)
    for left_idx in range(search_start, n - CUP_WITH_HANDLE_MIN_WEEKS):
        left_high = weekly_highs[left_idx]
        # Cup minimum size
        for cup_weeks in range(CUP_WITH_HANDLE_MIN_WEEKS,
                                 min(CUP_WITH_HANDLE_MAX_WEEKS, n - left_idx)):
            right_idx = left_idx + cup_weeks
            if right_idx >= n:
                break
            right_high = max(weekly_highs[right_idx:min(right_idx+3, n)])
            cup_segment_lows = weekly_lows[left_idx:right_idx+1]
            cup_low = min(cup_segment_lows)
            cup_low_idx = left_idx + cup_segment_lows.index(cup_low)
            cup_depth = (cup_low / left_high) - 1
            recovery_ratio = right_high / left_high

            # Validate cup
            if not (CUP_MAX_DEPTH <= cup_depth <= CUP_MIN_DEPTH):
                continue
            if recovery_ratio < 0.90:
                continue
            # U-shape: cup_low should be near the middle (not in first/last 20%)
            cup_low_position = (cup_low_idx - left_idx) / cup_weeks
            if not (0.25 <= cup_low_position <= 0.75):
                continue

            # Look for handle AFTER right_idx
            for h_weeks in range(HANDLE_MIN_WEEKS, HANDLE_MAX_WEEKS + 1):
                handle_end = right_idx + h_weeks
                if handle_end >= n:
                    break
                handle_segment_lows = weekly_lows[right_idx:handle_end+1]
                handle_low = min(handle_segment_lows)
                handle_depth = (handle_low / right_high) - 1

                if not (HANDLE_MAX_DEPTH <= handle_depth <= -0.03):
                    continue

                # Handle should drift SIDEWAYS or slightly down, not crater
                handle_high = max(weekly_highs[right_idx:handle_end+1])
                # Volume dry-up confirmation
                cup_avg_vol = sum(weekly_volume[left_idx:right_idx+1]) / max(1, right_idx - left_idx + 1)
                handle_avg_vol = sum(weekly_volume[right_idx:handle_end+1]) / max(1, h_weeks + 1)
                vol_dryup = handle_avg_vol < cup_avg_vol * 0.85

                # Pivot = handle_high + buffer
                pivot = handle_high + ONEIL_PIVOT_BUFFER

                # Quality grade
                quality = "A"
                if not vol_dryup:                   quality = "B"
                if recovery_ratio < 0.95:           quality = "B"
                if cup_depth < -0.25:               quality = "C"
                if h_weeks > 3:                     quality = "C"

                candidate = {
                    "type": "cup_with_handle",
                    "cup_start_idx": left_idx,
                    "cup_low_idx": cup_low_idx,
                    "cup_right_idx": right_idx,
                    "handle_start_idx": right_idx,
                    "handle_end_idx": handle_end,
                    "cup_weeks": cup_weeks,
                    "handle_weeks": h_weeks,
                    "cup_depth": round(cup_depth * 100, 1),       # %
                    "handle_depth": round(handle_depth * 100, 1),
                    "left_high": round(left_high, 2),
                    "cup_low": round(cup_low, 2),
                    "right_high": round(right_high, 2),
                    "handle_high": round(handle_high, 2),
                    "handle_low": round(handle_low, 2),
                    "pivot": round(pivot, 2),
                    "recovery_ratio": round(recovery_ratio, 3),
                    "volume_dryup_in_handle": vol_dryup,
                    "quality": quality,
                }
                # Prefer most recent and highest quality
                if (best is None or
                    handle_end > best["handle_end_idx"] or
                    (handle_end == best["handle_end_idx"] and quality < best["quality"])):
                    best = candidate
                break   # found handle for this cup, try next cup

    return best


def detect_flat_base(weekly_highs: list, weekly_lows: list,
                       weekly_volume: list) -> Optional[dict]:
    """Detect O'Neil Flat Base pattern.

    Criteria:
    1. ≥5 weeks of sideways action
    2. Depth ≤ -15% from peak
    3. Recent close near top of range
    4. Volume contracting toward end
    """
    n = len(weekly_highs)
    if n < FLAT_BASE_MIN_WEEKS:
        return None

    # Look at last 5-15 weeks for flat base
    best: Optional[dict] = None
    for window in range(FLAT_BASE_MIN_WEEKS, min(15, n)):
        start_idx = n - window
        base_high = max(weekly_highs[start_idx:n])
        base_low  = min(weekly_lows[start_idx:n])
        depth = (base_low / base_high) - 1

        if not (FLAT_BASE_MAX_DEPTH <= depth <= -0.05):
            continue

        # Top half check: current price near top of range
        current_close = weekly_highs[n-1]
        if current_close < (base_high + base_low) / 2:
            continue   # not near top

        # Volume contraction (recent < earlier)
        early_avg = sum(weekly_volume[start_idx:start_idx+window//2]) / max(1, window//2)
        late_avg = sum(weekly_volume[start_idx+window//2:n]) / max(1, window - window//2)
        vol_contraction = late_avg < early_avg * 0.9

        pivot = base_high + ONEIL_PIVOT_BUFFER
        quality = "A" if vol_contraction else "B"
        if depth < -0.12: quality = "C"

        candidate = {
            "type": "flat_base",
            "start_idx": start_idx,
            "weeks": window,
            "base_high": round(base_high, 2),
            "base_low": round(base_low, 2),
            "depth": round(depth * 100, 1),
            "pivot": round(pivot, 2),
            "vol_contraction": vol_contraction,
            "quality": quality,
        }
        if best is None or window > best["weeks"]:
            best = candidate
    return best


def detect_double_bottom(weekly_closes: list, weekly_highs: list,
                          weekly_lows: list) -> Optional[dict]:
    """Detect Double-Bottom (W) pattern.

    Criteria:
    1. Two lows with similar depth (within 5% of each other)
    2. Middle peak ("center" of W)
    3. 2nd low ≥ 1st low (slightly higher or equal)
    4. Pivot = high of middle peak
    """
    n = len(weekly_lows)
    if n < 7:
        return None

    # Search last 20 weeks
    search_start = max(0, n - 20)
    sub_lows = weekly_lows[search_start:]
    sub_highs = weekly_highs[search_start:]
    sub_n = len(sub_lows)

    best: Optional[dict] = None

    # Find lowest low in the search window
    for low1_offset in range(0, sub_n - 5):
        low1 = sub_lows[low1_offset]
        # Find peak in between
        for peak_offset in range(low1_offset + 2, sub_n - 2):
            peak = sub_highs[peak_offset]
            # Look for second low after the peak
            for low2_offset in range(peak_offset + 1, sub_n - 1):
                low2 = sub_lows[low2_offset]
                # Both lows within 5% of each other
                if abs(low2 - low1) / max(low1, 1e-9) > 0.05:
                    continue
                # Peak must be at least 5% above both lows
                if peak / max(low1, low2) < 1.05:
                    continue
                # 2nd low should not be much lower than 1st
                if low2 < low1 * 0.97:
                    continue

                pivot = peak + ONEIL_PIVOT_BUFFER
                depth = (min(low1, low2) / peak) - 1
                quality = "A" if low2 >= low1 else "B"
                if depth < -0.20: quality = "C"

                candidate = {
                    "type": "double_bottom",
                    "low1_idx": search_start + low1_offset,
                    "peak_idx": search_start + peak_offset,
                    "low2_idx": search_start + low2_offset,
                    "low1": round(low1, 2),
                    "low2": round(low2, 2),
                    "peak": round(peak, 2),
                    "depth": round(depth * 100, 1),
                    "pivot": round(pivot, 2),
                    "quality": quality,
                }
                # Prefer most recent
                if best is None or low2_offset > best["low2_idx"] - search_start:
                    best = candidate
    return best


def detect_best_base_pattern(daily_closes: list, daily_highs: list,
                                daily_lows: list, daily_volume: list) -> Optional[dict]:
    """Detect strongest CAN SLIM base pattern from OHLCV.

    Returns best pattern from cup-with-handle / flat-base / double-bottom,
    or None if none detected.
    """
    w_close, w_high, w_low, w_vol = _to_weekly(
        daily_closes, daily_highs, daily_lows, daily_volume)
    if not w_close:
        return None

    cup = detect_cup_with_handle(w_close, w_high, w_low, w_vol)
    flat = detect_flat_base(w_high, w_low, w_vol)
    db = detect_double_bottom(w_close, w_high, w_low)

    # Priority: cup-with-handle > double-bottom > flat-base (O'Neil hierarchy)
    candidates = [c for c in (cup, db, flat) if c]
    if not candidates:
        return None

    # Prefer A-quality, then most recent
    candidates.sort(key=lambda c: (c.get("quality") or "Z", -c.get("handle_end_idx", c.get("low2_idx", c.get("start_idx", 0)))))
    return candidates[0]


# ─────────────────────────────────────────────────────────────────
# Elliott Wave Fibonacci Entries (lightweight, reuse pivot detection)
# ─────────────────────────────────────────────────────────────────

def _fib_retracement(start: float, end: float, level: float) -> float:
    """Compute Fibonacci retracement level between start (low) and end (high)."""
    return end - (end - start) * level


def compute_elliott_entry(highs: list, lows: list, closes: list,
                           current_price: float) -> Optional[dict]:
    """Compute Elliott Wave-based entry zones using Fib retracements.

    Strategy:
      - Identify recent swing high + swing low (5-bar fractals)
      - If currently in pullback from recent high:
        • W2 entry: 38.2-61.8% retracement of recent swing
        • W4 entry: 23.6-38.2% retracement (if W3 detected)
    """
    if len(highs) < 30:
        return None

    # Find swing high + low in last 60 bars
    recent_highs = highs[-60:]
    recent_lows  = lows[-60:]
    swing_high = max(recent_highs)
    swing_low = min(recent_lows)
    swing_high_idx = len(recent_highs) - 1 - recent_highs[::-1].index(swing_high)
    swing_low_idx  = len(recent_lows)  - 1 - recent_lows[::-1].index(swing_low)

    if swing_high == swing_low:
        return None

    # If current price has pulled back from recent high
    if swing_high_idx > swing_low_idx and current_price < swing_high:
        # We're in pullback — compute Fib retracements of swing_low → swing_high
        fib_236 = _fib_retracement(swing_low, swing_high, 0.236)
        fib_382 = _fib_retracement(swing_low, swing_high, 0.382)
        fib_500 = _fib_retracement(swing_low, swing_high, 0.500)
        fib_618 = _fib_retracement(swing_low, swing_high, 0.618)
        return {
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "fib_236": round(fib_236, 2),     # shallowest pullback
            "fib_382": round(fib_382, 2),     # W4 typical
            "fib_500": round(fib_500, 2),
            "fib_618": round(fib_618, 2),     # W2 typical / deepest acceptable
            "entry_w4_zone": (round(fib_382, 2), round(fib_236, 2)),
            "entry_w2_zone": (round(fib_618, 2), round(fib_500, 2)),
        }
    return None


# ─────────────────────────────────────────────────────────────────
# SMA50 reclaim entry
# ─────────────────────────────────────────────────────────────────

def compute_sma_entry(closes: list, current_price: float) -> Optional[dict]:
    """Conservative entry on SMA50 reclaim or pullback."""
    if len(closes) < 60:
        return None
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50

    sma20_entry = sma20 * 1.005   # +0.5% above SMA20
    sma50_entry = sma50 * 1.01    # +1.0% above SMA50

    return {
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "sma20_entry": round(sma20_entry, 2),
        "sma50_entry": round(sma50_entry, 2),
        "current_vs_sma20": round((current_price / sma20 - 1) * 100, 2),
        "current_vs_sma50": round((current_price / sma50 - 1) * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────
# Volume confirmation
# ─────────────────────────────────────────────────────────────────

def check_volume_confirmation(daily_volume: list, lookback: int = 50) -> dict:
    """Check if recent volume supports breakout (CAN SLIM rule)."""
    if len(daily_volume) < lookback:
        return {"avg_vol": None, "recent_vol_ratio": None, "confirmed": False}
    avg_vol = sum(daily_volume[-lookback:-5]) / max(1, lookback - 5)
    recent_vol = sum(daily_volume[-5:]) / 5
    ratio = recent_vol / max(1, avg_vol)
    return {
        "avg_vol": round(avg_vol, 0),
        "recent_vol": round(recent_vol, 0),
        "recent_vol_ratio": round(ratio, 2),
        "confirmed": ratio >= VOLUME_CONFIRM_RATIO,
    }


# ─────────────────────────────────────────────────────────────────
# Main entry calculator (3-tier output)
# ─────────────────────────────────────────────────────────────────

def _composite_gate(composite: float, classification: str) -> str:
    """Return tier eligibility based on composite + classification."""
    cls = (classification or "").upper()
    strong_cls = any(s in cls for s in ("CONTINUATION","FORMATION","RECOVERY","LAGGING_CATCHUP"))
    if composite >= 75 and strong_cls:
        return "ALL"            # AGGRESSIVE + PRIMARY + CONSERVATIVE
    if composite >= 65:
        return "PRIMARY_AND_CONSERVATIVE"
    if composite >= 55:
        return "CONSERVATIVE_ONLY"
    return "SKIP"


def compute_entry_for_ticker(ticker: str, horizon: str,
                                scan_row: Optional[dict] = None,
                                cache_dict: Optional[dict] = None,
                                use_cache: bool = True) -> Optional[dict]:
    """Main entry-point computation for a single ticker.

    Returns:
        {
          ticker, horizon, currency, currency_symbol, current_price,
          composite_tier: ALL / PRIMARY_AND_CONSERVATIVE / CONSERVATIVE_ONLY / SKIP,
          entry_aggressive, entry_primary, entry_conservative,
          aggressive_rationale, primary_rationale, conservative_rationale,
          base_pattern: {type, quality, pivot, ...} or None,
          volume_confirmed: bool,
          oneil_cut_loss: stop_price (-7% from primary entry),
          computed_at,
        }
    """
    cache = cache_dict if cache_dict is not None else (_load_cache() if use_cache else {})
    cache_key = f"{ticker}::{horizon}"
    if use_cache and cache_key in cache and _is_cache_fresh(cache[cache_key]):
        return cache[cache_key]

    try:
        import yfinance as yf
        df = yf.download(ticker, period="1y", interval="1d",
                          progress=False, auto_adjust=True)
        if df.empty or len(df) < 60:
            return None
        if hasattr(df.columns, "get_level_values"):
            try: df.columns = df.columns.get_level_values(0)
            except Exception: pass
        df = df.dropna(subset=["High","Low","Close","Volume"])
        if len(df) < 60:
            return None

        highs   = df["High"].values.tolist()
        lows    = df["Low"].values.tolist()
        closes  = df["Close"].values.tolist()
        volumes = df["Volume"].values.tolist()
        current_price = float(closes[-1])
        if not math.isfinite(current_price) or current_price <= 0:
            return None

        currency, sym = _detect_currency(ticker)

        # Composite gate
        comp = float(scan_row.get("composite", 0) if scan_row else 0)
        cls  = (scan_row.get("classification", "") if scan_row else "") or ""
        tier_eligibility = _composite_gate(comp, cls)
        if tier_eligibility == "SKIP":
            result = {
                "ticker": ticker, "horizon": horizon,
                "current_price": round(current_price, 2),
                "currency": currency, "currency_symbol": sym,
                "composite_tier": "SKIP",
                "entry_aggressive": None, "entry_primary": None, "entry_conservative": None,
                "skip_reason": f"composite={comp:.1f} 또는 classification={cls} 부적합",
                "computed_at": datetime.now().isoformat(timespec="seconds"),
            }
            if use_cache:
                cache[cache_key] = result; _save_cache(cache)
            return result

        # --- Compute the 3 entry levels ---
        # 1) AGGRESSIVE = current price (only if comp ≥ 75)
        aggressive = None
        aggressive_rationale = None
        if tier_eligibility == "ALL":
            aggressive = round(current_price, 2)
            aggressive_rationale = f"Composite {comp:.0f} + 강한 분류 ({cls[:20]}) → 즉시 진입 (시스템 강력 신호)"

        # 2) PRIMARY = CAN SLIM pivot — with O'Neil "extended" check
        base = detect_best_base_pattern(closes, highs, lows, volumes)
        vol_info = check_volume_confirmation(volumes)
        primary = None
        primary_rationale = None
        primary_status = None   # "actionable" / "extended" / "stale" / "missing"

        if base and tier_eligibility in ("ALL", "PRIMARY_AND_CONSERVATIVE"):
            raw_pivot = base["pivot"]
            ptype = base["type"]
            quality = base.get("quality", "?")
            pname = {"cup_with_handle":"Cup-with-Handle","flat_base":"Flat Base","double_bottom":"Double Bottom"}.get(ptype, ptype)

            # O'Neil EXTENDED rule: don't buy more than 5% above pivot.
            # Detect stale/extended/actionable status to keep entries logically consistent.
            extended_pct = (current_price / raw_pivot - 1) * 100
            ONEIL_EXTENDED_THRESHOLD = 5.0   # %

            if extended_pct > ONEIL_EXTENDED_THRESHOLD:
                # Pivot already broken out + extended → no actionable PRIMARY entry.
                # Setting primary=None forces the table to show only CONSERVATIVE
                # (SMA50 pullback) as the safe re-entry path.
                primary = None
                primary_status = "extended"
                primary_rationale = (
                    f"⚠ EXTENDED — {pname} pivot ${raw_pivot:.2f} 이미 돌파 후 "
                    f"현재가 +{extended_pct:.1f}% (O'Neil 5% 룰 초과). "
                    f"추가 매수 금지 — 새 base 형성 또는 CONSERVATIVE pullback 대기."
                )
            elif extended_pct >= 0 and extended_pct <= ONEIL_EXTENDED_THRESHOLD:
                # Just broke pivot, still in buy zone (within 5%)
                primary = raw_pivot
                primary_status = "buy_zone"
                vol_str = f"거래량 {vol_info['recent_vol_ratio']:.1f}x" if vol_info["recent_vol_ratio"] else "거래량 미확인"
                primary_rationale = (
                    f"CAN SLIM {pname} (Quality {quality}) — Pivot ${primary:.2f} 직후 진입 가능 "
                    f"(현재 +{extended_pct:.1f}%, 5% buy zone 이내). "
                    f"O'Neil 룰: handle high + $0.10. {vol_str} "
                    f"{'(✓ 확인됨)' if vol_info['confirmed'] else '(미확인 — 1.4x+ 대기)'}"
                )
            else:
                # Pivot still ABOVE current price → await breakout (classic CAN SLIM entry)
                primary = raw_pivot
                primary_status = "await_breakout"
                vol_str = f"거래량 {vol_info['recent_vol_ratio']:.1f}x" if vol_info["recent_vol_ratio"] else "거래량 미확인"
                primary_rationale = (
                    f"CAN SLIM {pname} (Quality {quality}) — Pivot ${primary:.2f} 돌파 시 진입 "
                    f"(현재 {extended_pct:.1f}% 아래, 돌파 대기). "
                    f"O'Neil 룰: handle high + $0.10. {vol_str} "
                    f"{'(✓ 확인됨)' if vol_info['confirmed'] else '(미확인 — 1.4x+ 대기)'}"
                )
        elif tier_eligibility in ("ALL", "PRIMARY_AND_CONSERVATIVE"):
            # No base pattern → fallback to Elliott W2 zone
            elliott = compute_elliott_entry(highs, lows, closes, current_price)
            if elliott:
                primary = elliott["fib_500"]
                primary_status = "elliott_fallback"
                primary_rationale = (
                    f"CAN SLIM base 미감지 → Elliott Fibonacci 50% pullback 진입 "
                    f"(${elliott['fib_618']:.2f}-${elliott['fib_500']:.2f} 구간). "
                    f"W2/W4 retracement 매수 후보."
                )

        # 3) CONSERVATIVE = SMA50 + 1% OR Elliott W2 deep
        conservative = None
        conservative_rationale = None
        sma_info = compute_sma_entry(closes, current_price)
        if sma_info:
            conservative = sma_info["sma50_entry"]
            sma50_dist = sma_info["current_vs_sma50"]
            conservative_rationale = (
                f"SMA50 reclaim 진입 ${conservative:.2f} (SMA50 ${sma_info['sma50']:.2f} + 1%). "
                f"현재가 SMA50 대비 {sma50_dist:+.1f}%. "
                f"강세 trend continuation 진입 — 가장 보수적."
            )

        # O'Neil 7% cut-loss from primary entry
        oneil_cut_loss = None
        if primary:
            oneil_cut_loss = round(primary * (1 + ONEIL_CUT_LOSS_PCT), 2)

        # Final risk/reward analysis
        rr = None
        if primary and conservative and aggressive:
            # Assume target = +20% (O'Neil typical)
            target = primary * 1.20
            risk = primary - (oneil_cut_loss or primary * 0.93)
            reward = target - primary
            rr = round(reward / max(0.01, risk), 2) if risk > 0 else None

        result = {
            "ticker": ticker,
            "horizon": horizon,
            "currency": currency,
            "currency_symbol": sym,
            "current_price": round(current_price, 2),
            "composite_tier": tier_eligibility,
            "entry_aggressive": aggressive,
            "entry_primary": primary,
            "entry_conservative": conservative,
            "primary_status": primary_status,
            "aggressive_rationale": aggressive_rationale,
            "primary_rationale": primary_rationale,
            "conservative_rationale": conservative_rationale,
            "base_pattern": base,
            "volume_info": vol_info,
            "sma_info": sma_info,
            "oneil_cut_loss": oneil_cut_loss,
            "risk_reward_ratio": rr,
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        }
        if use_cache:
            cache[cache_key] = result; _save_cache(cache)
        return result

    except Exception as e:
        return {"ticker": ticker, "horizon": horizon, "_error": str(e)[:200]}


# ─────────────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not CACHE_PATH.exists(): return {}
    try: return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception: return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    except Exception: pass


def _is_cache_fresh(entry: dict) -> bool:
    ts = entry.get("computed_at", "")
    if not ts: return False
    try:
        dt = datetime.fromisoformat(ts)
        age_h = (datetime.now() - dt).total_seconds() / 3600.0
        return age_h < CACHE_TTL_HOURS
    except Exception: return False


# ─────────────────────────────────────────────────────────────────
# Pyramiding (Livermore-inspired)
# ─────────────────────────────────────────────────────────────────

def compute_pyramid_levels(entry_price: float, num_layers: int = 3) -> list:
    """Compute Livermore pyramid levels — add positions as price rises.

    Strategy: 1st position at entry, 2nd at +5%, 3rd at +10%.
    Stop on all positions moves up to entry price + small buffer (trailing).

    Returns: list of {layer, price, size_ratio, condition}.
    """
    levels = []
    pct_increments = [0.0, 0.05, 0.10][:num_layers]
    sizes = [0.5, 0.3, 0.2][:num_layers]   # decreasing size (pyramid shape)
    for i, (pct, sz) in enumerate(zip(pct_increments, sizes)):
        price = round(entry_price * (1 + pct), 2)
        levels.append({
            "layer": i + 1,
            "price": price,
            "trigger_pct": round(pct * 100, 1),
            "position_size_pct": int(sz * 100),
            "condition": f"Layer {i+1}: 진입가 {pct*100:+.0f}% 도달 시 {int(sz*100)}% 사이즈 추가" if i > 0
                          else f"Layer 1 (초기): 진입 시 {int(sz*100)}% 사이즈",
        })
    return levels


# ─────────────────────────────────────────────────────────────────
# buy_list annotation
# ─────────────────────────────────────────────────────────────────

def annotate_buy_list_with_entries(buy_list: list, scan_lookup: Optional[dict] = None,
                                       use_cache: bool = True,
                                       rate_limit_delay: float = 0.3) -> list:
    """Augment buy_list picks with entry_aggressive / entry_primary / entry_conservative."""
    cache = _load_cache() if use_cache else {}
    for pick in buy_list:
        t = pick.get("ticker")
        h = pick.get("horizon", "core")
        if not t:
            continue
        scan_row = (scan_lookup or {}).get(t) or {
            "composite": pick.get("composite", 0),
            "classification": pick.get("classification") or pick.get("cls", ""),
        }
        # Cache-hit detection — skip rate-limit sleep when served from cache
        cache_hit = use_cache and f"{t}::{h}" in cache and _is_cache_fresh(cache[f"{t}::{h}"])
        info = compute_entry_for_ticker(t, h, scan_row=scan_row,
                                          cache_dict=cache, use_cache=use_cache)
        if info and not info.get("_error"):
            pick["entry_aggressive"]      = info.get("entry_aggressive")
            pick["entry_primary"]         = info.get("entry_primary")
            pick["entry_conservative"]    = info.get("entry_conservative")
            pick["entry_primary_status"]  = info.get("primary_status")
            pick["entry_aggressive_rationale"]    = info.get("aggressive_rationale")
            pick["entry_primary_rationale"]       = info.get("primary_rationale")
            pick["entry_conservative_rationale"]  = info.get("conservative_rationale")
            pick["entry_base_pattern"]    = (info.get("base_pattern") or {}).get("type")
            pick["entry_base_quality"]    = (info.get("base_pattern") or {}).get("quality")
            pick["entry_volume_confirmed"] = (info.get("volume_info") or {}).get("confirmed")
            pick["entry_volume_ratio"]    = (info.get("volume_info") or {}).get("recent_vol_ratio")
            pick["entry_oneil_cut_loss"]  = info.get("oneil_cut_loss")
            pick["entry_rr_ratio"]        = info.get("risk_reward_ratio")
            pick["entry_composite_tier"]  = info.get("composite_tier")
            # Pyramid levels for primary entry
            if info.get("entry_primary"):
                pick["entry_pyramid_layers"] = compute_pyramid_levels(info["entry_primary"])
            # Inherit currency from elliott (in case present)
            if not pick.get("currency"):
                pick["currency"] = info.get("currency")
                pick["currency_symbol"] = info.get("currency_symbol")
        else:
            pick["entry_aggressive"] = None
            pick["entry_primary"] = None
            pick["entry_conservative"] = None
            pick["entry_skip_reason"] = info.get("_error", "no_data") if info else "no_data"
        if rate_limit_delay and not cache_hit:
            time.sleep(rate_limit_delay)   # only throttle on actual yfinance calls
    if use_cache:
        _save_cache(cache)
    return buy_list
