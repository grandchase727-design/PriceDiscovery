# -*- coding: utf-8 -*-
"""elliott_wave_stops.py — Elliott Wave-based stop-loss calculator.

================================================================================
PURPOSE
================================================================================

For each buy_list pick, compute a horizon-aware stop-loss using Elliott Wave
theory:
  - tactical (5d hold)  → TIGHT stop (Wave 4 low - 1%)
  - core     (21d hold) → PRIMARY stop (Wave 1 top - 1%, strict Elliott rule)
  - strategic (63d hold)→ PRIMARY + INVALID (Wave 2 low) fallback

================================================================================
ELLIOTT WAVE PRINCIPLES APPLIED
================================================================================

1. **Wave 4 cannot enter Wave 1 territory** (PRIMARY invalidation)
2. **Wave 5 progression invalidated by W4 low breach** (TIGHT trigger)
3. **Wave 2 low breach = entire impulse count invalid** (INVALID stop)

Swing pivot detection: 5-bar fractal method (center > 5 left + 5 right).

================================================================================
USAGE
================================================================================

from agents.elliott_wave_stops import compute_stops_for_picks

picks = [{"ticker":"CMI","horizon":"tactical"}, ...]
stops = compute_stops_for_picks(picks, cache=True)
# returns {"CMI": {"stop_price": 623.85, "stop_pct": -14.5, "stop_type":"W4_TIGHT", ...}}

Cached to .elliott_stops_cache.json (date-keyed). Refresh once per day.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime, date
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

CACHE_PATH = Path(".elliott_stops_cache.json")
CACHE_TTL_HOURS = 24


def _find_swing_pivots(highs, lows, n_bars: int = 5):
    """Find swing high/low pivots using n-bar fractal method.

    A pivot is a center bar that is the most extreme within (n left + 1 + n right).
    Returns sorted list of (index, price, type) where type='H' or 'L'.
    """
    pivots = []
    for i in range(n_bars, len(highs) - n_bars):
        left_h  = highs[i - n_bars:i]
        right_h = highs[i + 1:i + 1 + n_bars]
        if highs[i] > left_h.max() and highs[i] > right_h.max():
            pivots.append((i, float(highs[i]), 'H'))
        left_l  = lows[i - n_bars:i]
        right_l = lows[i + 1:i + 1 + n_bars]
        if lows[i] < left_l.min() and lows[i] < right_l.min():
            pivots.append((i, float(lows[i]), 'L'))
    pivots.sort(key=lambda p: p[0])
    # Filter consecutive same-direction pivots (keep most extreme)
    filtered = []
    for p in pivots:
        if filtered and filtered[-1][2] == p[2]:
            if (p[2] == 'H' and p[1] > filtered[-1][1]) or (p[2] == 'L' and p[1] < filtered[-1][1]):
                filtered[-1] = p
        else:
            filtered.append(p)
    return filtered


def _classify_wave_position(pivots: list) -> dict:
    """Heuristic wave position classification from last 5 pivots."""
    if len(pivots) < 4:
        return {"wave_guess": "INSUFFICIENT_DATA", "levels": {}}

    p = pivots[-5:] if len(pivots) >= 5 else pivots[-4:]

    if len(p) >= 5 and p[-5][2]=='L' and p[-4][2]=='H' and p[-3][2]=='L' and p[-2][2]=='H' and p[-1][2]=='L':
        return {
            "wave_guess": "WAVE_5_ACTIVE",
            "levels": {
                "W1_start": p[-5][1], "W1_top": p[-4][1],
                "W2_low":   p[-3][1], "W3_top": p[-2][1],
                "W4_low":   p[-1][1],
            },
        }
    elif len(p) >= 4 and p[-4][2]=='L' and p[-3][2]=='H' and p[-2][2]=='L' and p[-1][2]=='H':
        return {
            "wave_guess": "WAVE_3_TOP",
            "levels": {
                "W1_start": p[-4][1], "W1_top": p[-3][1],
                "W2_low":   p[-2][1], "W3_top": p[-1][1],
            },
        }
    elif len(p) >= 3 and p[-3][2]=='L' and p[-2][2]=='H' and p[-1][2]=='L':
        return {
            "wave_guess": "WAVE_3_EARLY",
            "levels": {
                "W1_start": p[-3][1], "W1_top": p[-2][1],
                "W2_low":   p[-1][1],
            },
        }
    elif len(p) >= 2 and p[-2][2]=='L' and p[-1][2]=='H':
        return {
            "wave_guess": "WAVE_1_NEW",
            "levels": {"W0_start": p[-2][1], "W1_top": p[-1][1]},
        }
    else:
        return {"wave_guess": "AMBIGUOUS", "levels": {}}


# ─────────────────────────────────────────────────────────────────
# O'Neil 7% absolute cut-loss rule (CAN SLIM integration)
# ─────────────────────────────────────────────────────────────────
ONEIL_CUT_LOSS_PCT = -0.07  # -7% from entry — O'Neil's strict sell-rule #1


def _apply_oneil_cap(stop_price: float, current_price: float) -> tuple[float, bool]:
    """Apply O'Neil's 7% absolute cap to a wave-derived stop.

    Returns: (final_stop_price, was_capped). If Elliott Wave stop is deeper than
    -7%, override with -7% from current price.
    """
    if current_price <= 0:
        return stop_price, False
    oneil_floor = current_price * (1 + ONEIL_CUT_LOSS_PCT)   # -7% level
    if stop_price < oneil_floor:
        return round(oneil_floor, 2), True
    return stop_price, False


def _pick_stop_for_horizon(wave_info: dict, current_price: float,
                            horizon: str) -> Optional[dict]:
    """Select horizon-appropriate stop level from wave structure.

    Returns: {stop_price, stop_type, stop_pct, rationale}
    """
    levels = wave_info.get("levels", {})
    wave = wave_info.get("wave_guess", "")
    h = (horizon or "core").lower()

    # ── WAVE 5 ACTIVE → W4 low / W1 top / W2 low ──
    if wave == "WAVE_5_ACTIVE":
        w4 = levels.get("W4_low"); w1_top = levels.get("W1_top"); w2 = levels.get("W2_low")
        if h == "tactical" and w4:
            stop = w4 * 0.99
            # CAN SLIM 7% absolute cap (don't allow stops deeper than -7%)
            stop, capped = _apply_oneil_cap(stop, current_price)
            if capped:
                return {"stop_price": stop, "stop_type": "W4_TIGHT_ONEIL_CAP",
                        "rationale": f"W4 저점 ${w4:.2f} 깊음 → O'Neil 7% 룰로 -7% cap 적용. "
                                      f"CAN SLIM 절대 손실 한도 우선 적용."}
            return {"stop_price": stop, "stop_type": "W4_TIGHT",
                    "rationale": f"W4 저점 ${w4:.2f} 직하. Wave 5 invalidation 직전 청산."}
        elif h == "core" and w1_top:
            stop = w1_top * 0.99
            return {"stop_price": stop, "stop_type": "W1_PRIMARY",
                    "rationale": f"W1 상단 ${w1_top:.2f} 직하. Elliott 엄격 룰: W4가 W1 침범 불가."}
        elif h == "strategic":
            # Use deeper of W1 top or W2 low — whichever is more conservative below current
            if w2 and w1_top:
                if w2 < w1_top:    # W2 is deeper
                    return {"stop_price": w2 * 0.98, "stop_type": "W2_INVALID",
                            "rationale": f"W2 저점 ${w2:.2f}. 침범 시 전체 파동 카운트 무효 — 장기 안전 손절."}
                else:
                    return {"stop_price": w1_top * 0.99, "stop_type": "W1_PRIMARY",
                            "rationale": f"W1 상단 ${w1_top:.2f}. Elliott 엄격 룰."}
            elif w1_top:
                return {"stop_price": w1_top * 0.99, "stop_type": "W1_PRIMARY",
                        "rationale": f"W1 상단 ${w1_top:.2f}. Elliott 엄격 룰."}

    # ── WAVE 3 EARLY / TOP → W2 low / W1 start ──
    if wave in ("WAVE_3_EARLY", "WAVE_3_TOP"):
        w2 = levels.get("W2_low"); w1_start = levels.get("W1_start")
        if h in ("tactical", "core") and w2:
            return {"stop_price": w2 * 0.99, "stop_type": "W2_TIGHT",
                    "rationale": f"W2 저점 ${w2:.2f} 직하. Wave 3는 W2 저점 침범 불가."}
        elif h == "strategic" and w1_start:
            return {"stop_price": w1_start * 0.99, "stop_type": "W1_START_INVALID",
                    "rationale": f"W1 시작점 ${w1_start:.2f}. 침범 시 전체 wave count 무효."}

    # ── WAVE 1 (early new trend) → recent swing low ──
    if wave == "WAVE_1_NEW":
        w0 = levels.get("W0_start")
        if w0:
            return {"stop_price": w0 * 0.99, "stop_type": "SWING_LOW",
                    "rationale": f"최근 swing 저점 ${w0:.2f}. 새 추세 가설의 1차 지지."}

    # ── AMBIGUOUS / INSUFFICIENT_DATA → mechanical % stop ──
    mech_pct = {"tactical": 0.03, "core": 0.05, "strategic": 0.08}.get(h, 0.05)
    return {"stop_price": current_price * (1 - mech_pct), "stop_type": "MECHANICAL",
            "rationale": f"명확한 Elliott 파동 패턴 없음. 기계적 -{int(mech_pct*100)}% 손절."}


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    except Exception:
        pass


def _is_cache_fresh(entry: dict) -> bool:
    ts = entry.get("computed_at", "")
    if not ts:
        return False
    try:
        dt = datetime.fromisoformat(ts)
        age_h = (datetime.now() - dt).total_seconds() / 3600.0
        return age_h < CACHE_TTL_HOURS
    except Exception:
        return False


def compute_stop_for_ticker(ticker: str, horizon: str,
                              cache_dict: Optional[dict] = None,
                              use_cache: bool = True) -> Optional[dict]:
    """Compute Elliott Wave stop for a single ticker.

    Returns: {ticker, current_price, stop_price, stop_pct, stop_type, rationale,
              wave_guess, computed_at} or None on error.
    """
    cache = cache_dict if cache_dict is not None else (_load_cache() if use_cache else {})
    cache_key = f"{ticker}::{horizon}"

    if use_cache and cache_key in cache and _is_cache_fresh(cache[cache_key]):
        return cache[cache_key]

    try:
        import math
        import yfinance as yf
        df = yf.download(ticker, period="6mo", interval="1d",
                          progress=False, auto_adjust=True)
        if df.empty or len(df) < 30:
            return None
        # Flatten potential MultiIndex columns
        if hasattr(df.columns, "get_level_values"):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass
        # Drop rows with NaN to avoid Inf/NaN downstream
        df = df.dropna(subset=["High","Low","Close"])
        if len(df) < 30:
            return None
        highs  = df["High"].values
        lows   = df["Low"].values
        closes = df["Close"].values
        current_price = float(closes[-1])
        if not math.isfinite(current_price) or current_price <= 0:
            return None

        # Detect currency from ticker suffix (.KS/.KQ=KRW, .T=JPY, .L=GBP, default=USD)
        suffix_to_currency = {
            ".KS": "KRW", ".KQ": "KRW", ".T": "JPY", ".L": "GBP",
            ".HK": "HKD", ".SS": "CNY", ".SZ": "CNY", ".SI": "SGD",
            ".AX": "AUD", ".TO": "CAD", ".PA": "EUR", ".DE": "EUR",
        }
        currency = "USD"
        for sfx, ccy in suffix_to_currency.items():
            if ticker.upper().endswith(sfx):
                currency = ccy; break
        # Currency symbol for display
        currency_symbol = {"USD": "$", "KRW": "₩", "JPY": "¥", "EUR": "€",
                            "GBP": "£", "HKD": "HK$", "CNY": "¥",
                            "SGD": "S$", "AUD": "A$", "CAD": "C$"}.get(currency, currency + " ")

        pivots = _find_swing_pivots(highs, lows, n_bars=5)
        wave_info = _classify_wave_position(pivots)
        stop_info = _pick_stop_for_horizon(wave_info, current_price, horizon)

        # JSON-safe: drop NaN/Inf values
        def _safe(v):
            if v is None: return None
            try:
                v = float(v)
                return v if math.isfinite(v) else None
            except (TypeError, ValueError):
                return None

        stop_p = _safe(stop_info.get("stop_price"))
        if stop_p is None:
            return None

        result = {
            "ticker": ticker,
            "horizon": horizon,
            "current_price": round(current_price, 2),
            "stop_price": round(stop_p, 2),
            "stop_pct": round((stop_p / current_price - 1) * 100, 2),
            "stop_type": stop_info["stop_type"],
            "rationale": stop_info["rationale"],
            "wave_guess": wave_info["wave_guess"],
            "wave_levels": {k: round(_safe(v) or 0, 2) for k, v in wave_info.get("levels", {}).items() if _safe(v) is not None},
            "n_pivots": len(pivots),
            "currency": currency,
            "currency_symbol": currency_symbol,
            "computed_at": datetime.now().isoformat(timespec="seconds"),
        }

        if use_cache:
            cache[cache_key] = result
            _save_cache(cache)

        return result

    except Exception as e:
        return {"ticker": ticker, "horizon": horizon, "_error": str(e)[:200]}


def compute_stops_for_picks(picks: list, use_cache: bool = True,
                              rate_limit_delay: float = 0.3,
                              progress_cb=None) -> dict:
    """Compute Elliott Wave stops for a list of picks.

    Args:
        picks: list of dicts with at least 'ticker' and 'horizon' keys
        use_cache: load/save from .elliott_stops_cache.json
        rate_limit_delay: seconds between yfinance calls
        progress_cb: optional callable(idx, total, ticker) for progress

    Returns:
        {ticker: stop_info_dict} keyed by ticker
    """
    cache = _load_cache() if use_cache else {}
    results = {}
    total = len(picks)
    for idx, pick in enumerate(picks):
        ticker = pick.get("ticker")
        horizon = pick.get("horizon", "core")
        if not ticker:
            continue
        if progress_cb:
            try: progress_cb(idx, total, ticker)
            except Exception: pass

        cache_key = f"{ticker}::{horizon}"
        if use_cache and cache_key in cache and _is_cache_fresh(cache[cache_key]):
            results[ticker] = cache[cache_key]
            continue

        info = compute_stop_for_ticker(ticker, horizon,
                                        cache_dict=cache, use_cache=use_cache)
        if info and not info.get("_error"):
            results[ticker] = info
        elif info:   # error case — still return so frontend shows the issue
            results[ticker] = info

        if rate_limit_delay:
            time.sleep(rate_limit_delay)

    if use_cache:
        _save_cache(cache)
    return results


def compute_trailing_action(entry_price: float, current_price: float,
                              sma50: float, volume_ratio_today: float = 1.0,
                              days_held: int = 0) -> dict:
    """O'Neil-style trailing stop decision based on P&L and technical state.

    Implements William O'Neil's CAN SLIM sell rules:
      1. -7~8% from buy point → CUT_LOSS (absolute rule, no exceptions)
      2. +20-25% gain → TRIM_50 (lock half profit)
      3. SMA50 violation + heavy volume → FULL_SELL (institutional distribution)
      4. Climax run (>+50%, parabolic) → SELL_INTO_STRENGTH
      5. Hold otherwise

    Args:
        entry_price: actual purchase price (or PRIMARY entry)
        current_price: today's price
        sma50: current 50-day SMA
        volume_ratio_today: today's volume / 50-day avg (1.0 = average)
        days_held: holding period in days (for parabolic detection)

    Returns: {action, reason, severity (1=info, 2=warn, 3=urgent)}
    """
    if entry_price <= 0 or current_price <= 0:
        return {"action": "HOLD", "reason": "invalid_price", "severity": 1}

    pnl_pct = (current_price / entry_price - 1) * 100

    # Rule 1: Absolute -7~8% cut-loss (highest priority)
    if pnl_pct <= -7.0:
        return {
            "action": "CUT_LOSS",
            "reason": f"O'Neil 7% 룰 위반: P&L {pnl_pct:+.1f}% — 즉시 청산 (절대 손실 한도)",
            "severity": 3,
        }

    # Rule 2: SMA50 violation with heavy volume (institutional distribution)
    if sma50 > 0 and current_price < sma50 * 0.99 and volume_ratio_today >= 1.5:
        return {
            "action": "FULL_SELL",
            "reason": f"SMA50 ${sma50:.2f} 위반 + 거래량 {volume_ratio_today:.1f}x — 기관 distribution 신호",
            "severity": 3,
        }

    # Rule 4: Climax run / parabolic blow-off (sell into strength)
    if pnl_pct >= 50.0 and days_held >= 21:   # parabolic over 3+ weeks
        return {
            "action": "SELL_INTO_STRENGTH",
            "reason": f"Climax run: P&L {pnl_pct:+.0f}% ({days_held}d) — 강한 상승에 매도 (FOMO 절정)",
            "severity": 2,
        }

    # Rule 3: +20-25% gain → trim half
    if 20.0 <= pnl_pct < 50.0:
        return {
            "action": "TRIM_50",
            "reason": f"P&L {pnl_pct:+.1f}% — O'Neil 20% 룰: 절반 익절 (lock profit, runner 유지)",
            "severity": 2,
        }

    # SMA50 close but volume normal — warning only
    if sma50 > 0 and current_price < sma50 * 1.01 and pnl_pct > -7:
        return {
            "action": "HOLD",
            "reason": f"SMA50 ${sma50:.2f} 근접 (현재 ${current_price:.2f}) — 거래량 정상, 주의 모니터링",
            "severity": 1,
        }

    # Normal hold
    return {
        "action": "HOLD",
        "reason": f"P&L {pnl_pct:+.1f}% — 정상 보유, 트레일링 stop 모니터링 계속",
        "severity": 1,
    }


def annotate_buy_list_with_stops(buy_list: list, use_cache: bool = True) -> list:
    """In-place annotation: adds stop_loss fields to each pick in buy_list.

    Adds to each pick:
      - stop_price: float (the stop-loss target)
      - stop_pct:   float (%change vs current price, e.g. -2.9)
      - stop_type:  str (W4_TIGHT / W1_PRIMARY / W2_INVALID / SWING_LOW / MECHANICAL)
      - stop_rationale: str (Korean explanation)
      - stop_wave_guess: str (WAVE_5_ACTIVE / WAVE_3_EARLY / etc.)
      - stop_computed_at: ISO timestamp
    """
    stops = compute_stops_for_picks(buy_list, use_cache=use_cache)
    for pick in buy_list:
        t = pick.get("ticker")
        info = stops.get(t)
        if info and not info.get("_error"):
            pick["stop_price"]       = info.get("stop_price")
            pick["stop_pct"]         = info.get("stop_pct")
            pick["stop_type"]        = info.get("stop_type")
            pick["stop_rationale"]   = info.get("rationale")
            pick["stop_wave_guess"]  = info.get("wave_guess")
            pick["stop_computed_at"] = info.get("computed_at")
            pick["currency"]         = info.get("currency", "USD")
            pick["currency_symbol"]  = info.get("currency_symbol", "$")
            pick["current_price"]    = info.get("current_price")
        else:
            pick["stop_price"]       = None
            pick["stop_pct"]         = None
            pick["stop_type"]        = "UNAVAILABLE"
            pick["stop_rationale"]   = info.get("_error", "데이터 없음") if info else "데이터 없음"
            pick["stop_wave_guess"]  = None
            pick["stop_computed_at"] = None
            pick["currency"]         = "USD"
            pick["currency_symbol"]  = "$"
    return buy_list
