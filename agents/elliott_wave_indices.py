# -*- coding: utf-8 -*-
"""elliott_wave_indices.py — YTD Elliott-Wave labeler for 4 headline indices.

================================================================================
PURPOSE
================================================================================
Label the YTD price action of ACWI / SPY / QQQ / IWM with an Elliott-Wave count
and emit the per-index contract consumed by the frontend /api/elliott-wave-indices
endpoint. Pure deterministic function of (dates, OHLC) — no randomness, no LLM.

Algorithm: "anchored-impulse" (Greedy Anchor-and-Walk with Hard-Rule Validation)
upgraded with two grafts:
  GRAFT 1: adaptive ZigZag amplitude filter over the raw fractal pivots (noise kill).
  GRAFT 2: bounded multi-anchor scorer (<=3 candidate anchors) — pick the labeling
           with the highest tiny score (clean-legs + light Fibonacci bonus).

Reuses VERBATIM from elliott_wave_stops.py:
  - _find_swing_pivots(highs, lows, n_bars)   (imported)
  - the ATR20 %-of-price block
  - the _safe() finite guard
  - the per-ticker try/except + date-keyed cache pattern

See the design spec in the task brief for the full failure-mode ladder.
"""
from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    from agents.elliott_wave_stops import _find_swing_pivots, compute_stop_for_ticker
    from agents.entry_price import compute_elliott_entry, compute_sma_entry
except ImportError:
    # Allow standalone execution (python3 agents/elliott_wave_indices.py) by
    # putting the repo root on sys.path before retrying the package import.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agents.elliott_wave_stops import _find_swing_pivots, compute_stop_for_ticker
    from agents.entry_price import compute_elliott_entry, compute_sma_entry

import threading

CACHE_PATH = Path(".elliott_indices_cache.json")
CACHE_TTL_HOURS = 24

# Serializes calls to compute_stop_for_ticker across our 8 worker threads. That fn
# internally uses yfinance's thread-unsafe yf.download() (multi-ticker aggregator);
# under parallelism it cross-contaminates bars between tickers. Holding this lock
# during the stop fetch keeps those downloads mutually exclusive. The expensive
# wave analysis (pure CPU) + our own thread-safe Ticker.history fetch stay parallel,
# so we keep most of the speedup while the stop I/O is serialized for correctness.
_STOP_LOCK = threading.Lock()

# ── Index metadata — 3 groups, 25 tickers. EXACT order preserved within each group. ──
GROUPS = [
    {
        "key": "broad", "label": "광역 지수", "emoji": "📊", "default_open": True,
        "members": [
            ("ACWI", "MSCI ACWI (전세계)"),
            ("SPY",  "S&P 500 (미국 대형주)"),
            ("QQQ",  "나스닥 100 (미국 기술주)"),
            ("IWM",  "러셀 2000 (미국 소형주)"),
        ],
    },
    {
        "key": "sector", "label": "GICS 11 섹터", "emoji": "🏛", "default_open": False,
        "members": [
            ("XLK",  "기술 (Technology)"),
            ("XLF",  "금융 (Financials)"),
            ("XLV",  "헬스케어 (Health Care)"),
            ("XLY",  "임의소비재 (Cons. Disc.)"),
            ("XLP",  "필수소비재 (Cons. Staples)"),
            ("XLI",  "산업재 (Industrials)"),
            ("XLE",  "에너지 (Energy)"),
            ("XLB",  "소재 (Materials)"),
            ("XLU",  "유틸리티 (Utilities)"),
            ("XLRE", "부동산 (Real Estate)"),
            ("XLC",  "커뮤니케이션 (Comm. Svcs.)"),
        ],
    },
    {
        "key": "leveraged", "label": "레버리지 ETF", "emoji": "⚡", "default_open": False,
        "members": [
            ("TQQQ", "나스닥100 3배"),
            ("QLD",  "나스닥100 2배"),
            ("UPRO", "S&P500 3배"),
            ("TNA",  "러셀2000 3배"),
            ("SOXL", "반도체 3배"),
            ("TECL", "기술 3배"),
            ("FAS",  "금융 3배"),
            ("LABU", "바이오 3배"),
            ("FNGU", "FANG+ 3배"),
            ("KORU", "한국 3배"),
        ],
    },
]

# Flat list of (ticker, name) in group order — any other caller / self-test still works.
INDICES = [(t, n) for g in GROUPS for (t, n) in g["members"]]

# ── Analysis period presets ──────────────────────────────────────────────
# "ytd": from Jan 1 of the current year (original/default behavior, unchanged).
# "1m":  trailing ~1 calendar month. calendar_days=42 (not 30) so the fetched window
#        reliably clears min_bars/pref_bars/the ATR(20) floor even across a month with
#        several holidays (~29-30 trading days typical vs the ~21 a strict 30-calendar-day
#        window would give). min_bars/pref_bars are lowered from the YTD floors (30/40)
#        since a 1-month window genuinely has fewer bars — the pivot fractal window
#        (build_pivots' n_bars=3 for n<90) already adapts fine at this scale.
PERIOD_CONFIG = {
    "ytd": {"label": "YTD",   "calendar_days": None, "min_bars": 30, "pref_bars": 40},
    "1m":  {"label": "1개월", "calendar_days": 42,   "min_bars": 15, "pref_bars": 21},
}
DEFAULT_PERIOD = "ytd"


def _period_cfg(period: str) -> dict:
    return PERIOD_CONFIG.get(period, PERIOD_CONFIG[DEFAULT_PERIOD])

# ── Static enum → color / Korean label lookups (pure dict, deterministic) ──
PHASE_COLOR = {
    "IMPULSE_W1": "cyan", "IMPULSE_W2": "green", "IMPULSE_W3": "green",
    "IMPULSE_W4": "blue", "IMPULSE_W5": "amber",
    "CORRECTIVE_A": "red", "CORRECTIVE_B": "amber", "CORRECTIVE_C": "red",
    "UNCLEAR": "gray",
}
PHASE_LABEL = {
    "IMPULSE_W1": "상승 1파 진행 (초기 추세 형성)",
    "IMPULSE_W2": "상승 2파 조정 (1파 되돌림, 재진입 관찰)",
    "IMPULSE_W3": "상승 3파 진행 (강세 확장 국면)",
    "IMPULSE_W4": "상승 4파 조정 (되돌림, 눌림목)",
    "IMPULSE_W5": "상승 5파 진행 (종반 국면, 과열 경계)",
    "CORRECTIVE_A": "조정 A파 하락 (상승 사이클 종료 후 첫 하락)",
    "CORRECTIVE_B": "조정 B파 반등 (약세 되돌림, 반등 함정 주의)",
    "CORRECTIVE_C": "조정 C파 하락 (조정 마무리 국면)",
    "UNCLEAR": "파동 카운트 불명확 (횡보/비정형 구조)",
}


def _safe(v):
    """Finite guard — NaN/Inf/None → None, else float. (verbatim pattern)"""
    if v is None:
        return None
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────
# ENTRY / STOP SUB-DICTS (deterministic, no LLM)
# ─────────────────────────────────────────────────────────────────
# Actionable = the 4 ascending impulse legs. Everything else (W5 / A / B / C /
# UNCLEAR) is 관망 — primary=null, 🎯 prefix reserved ONLY for TRUE entries.
_ACTIONABLE_PHASES = {"IMPULSE_W1", "IMPULSE_W2", "IMPULSE_W3", "IMPULSE_W4"}

# Korean primary_label + rationale per non-actionable phase (관망, NEVER 🎯 prefix).
_WATCH = {
    "IMPULSE_W5": ("관망 (5파 종반, 신규 진입 부적합)",
                   "5파 종반·과열 국면 — 상승 여력 제한적이고 A파 반전 리스크, 신규 롱 부적합. 손절은 리스크 참고용."),
    "CORRECTIVE_A": ("관망 (A파 하락, 조정 진행)",
                     "상승 사이클 종료 후 첫 하락(A파) — 조정 미완, 롱 진입 금지. B파 반등은 함정·C파 저점 확인 후 재평가."),
    "CORRECTIVE_B": ("관망 (B파 반등 함정)",
                     "ABC 조정의 B파 반등 — 역추세 반등 함정(countertrend trap)으로 지속성 없음, 롱 진입 금지·C파 하락 경계."),
    "CORRECTIVE_C": ("관망 (C파 하락, 조정 마무리)",
                     "ABC 조정의 C파 하락 — 조정 마무리 국면이나 저점 미확정, 아직 롱 부적합. 새 1파 형성 확인 후 진입 검토."),
    "UNCLEAR": ("관망 (카운트 불명확)",
                "파동 카운트 불명확 — 방향성 미확정으로 진입 근거 없음. 명확한 스윙 구조 형성 후 재평가, 손절은 리스크 참고용."),
}


def _r2(v):
    """_safe() then round to 2dp — guarantees no NaN/Inf reaches JSON."""
    s = _safe(v)
    return None if s is None else round(s, 2)


# ── Currency inference from ticker suffix (2026-07-14 fix) ──
# 이전엔 모든 return path가 USD/$를 하드코딩 → 한국(.KS 등) 종목이 "$180,100"으로
# 오표기. elliott_wave_stops.py와 동일 규칙으로 접미사에서 통화를 유추한다.
_CCY_BY_SUFFIX = {
    ".KS": "KRW", ".KQ": "KRW", ".T": "JPY", ".L": "GBP", ".HK": "HKD",
    ".SS": "CNY", ".SZ": "CNY", ".SI": "SGD", ".AX": "AUD", ".TO": "CAD",
    ".PA": "EUR", ".DE": "EUR",
}
_CCY_SYMBOL = {"USD": "$", "KRW": "₩", "JPY": "¥", "EUR": "€", "GBP": "£",
               "HKD": "HK$", "CNY": "¥", "SGD": "S$", "AUD": "A$", "CAD": "C$"}


def _currency_for(ticker):
    """(currency, symbol) from ticker suffix — .KS/.KQ→KRW/₩, .T→JPY/¥, else USD/$."""
    t = (ticker or "").upper()
    for sfx, ccy in _CCY_BY_SUFFIX.items():
        if t.endswith(sfx):
            return ccy, _CCY_SYMBOL.get(ccy, ccy + " ")
    return "USD", "$"


def _stop_null():
    """Single source of truth for the unusable-stop sub-dict."""
    return {"price": None, "pct": None, "type": "NONE", "rationale": "손절 계산 불가"}


def _build_stop_subdict(ticker, current_price):
    """Reuse compute_stop_for_ticker(ticker,'core') VERBATIM (its own yfinance 6mo
    download). NOT composite-gated. 'Unusable' is any of:
    (1) r is None, (2) r.get('_error') truthy, (3) _r2(stop_price) is None → _stop_null().

    THREAD SAFETY: we run 8 tickers concurrently, and compute_stop_for_ticker's default
    use_cache path read-modify-writes a single shared .elliott_stops_cache.json — under
    parallelism that races and cross-contaminates stops between tickers. We pass
    use_cache=False so each call is a pure, isolated compute (no shared-file touch); the
    Elliott module's own date-keyed disk cache already covers same-day fast restarts."""
    try:
        with _STOP_LOCK:  # serialize the thread-unsafe internal yf.download
            r = compute_stop_for_ticker(ticker, "core", use_cache=False)  # NOT composite-gated
    except Exception:
        r = None
    if not r or r.get("_error") or _r2(r.get("stop_price")) is None:
        return _stop_null()
    sp = _r2(r.get("stop_price"))
    pct = _safe(r.get("stop_pct"))
    if pct is None and current_price:  # derive if fn omitted it
        cp = _safe(current_price)
        pct = ((sp / cp - 1) * 100) if (cp and sp is not None) else None
    return {"price": sp,
            "pct": (round(pct, 2) if pct is not None else None),
            "type": r.get("stop_type") or "NONE",
            "rationale": r.get("rationale") or ""}


def compute_entry_stop(phase, ticker, highs, lows, closes, current_price, labels_ctx=None):
    """Return (entry_subdict, stop_subdict) for one index. Deterministic, no LLM.

    highs/lows/closes: already-fetched YTD numpy arrays or lists (coerced to list here).
    labels_ctx: optional {'from_high_pct':..,'confidence':..} — reserved, not used for level math.
    STOP is ALWAYS computed from compute_stop_for_ticker (works for any liquid ticker regardless
    of wave count) → surfaced as a downside-risk reference even on non-actionable phases.
    """
    cp = _safe(current_price)
    stop = _build_stop_subdict(ticker, cp)  # ALWAYS computed → risk reference

    # ---- NON-ACTIONABLE (W5 / A / B / C / UNCLEAR) or no current price ----
    if phase not in _ACTIONABLE_PHASES or cp is None:
        label, why = _WATCH.get(phase, ("관망 (카운트 불명확)",
                                        "진입 근거 없음 — 손절은 리스크 참고용."))
        entry = {"actionable": False, "primary": None, "primary_label": label,
                 "zone_low": None, "zone_high": None, "rationale": why}
        # reframe the (still-rendered) stop as a downside RISK reference, not a protective stop
        if stop["price"] is not None:
            stop = {**stop,
                    "rationale": (f"신규 진입 없음 — 하방 위험 참고선 {stop['price']} "
                                  f"({stop['pct']}%). " + (stop["rationale"] or ""))}
        return entry, stop

    # ---- ACTIONABLE (W1 / W2 / W3 / W4) : precompute the two reusable sources ----
    try:
        el = compute_elliott_entry(list(highs), list(lows), list(closes), cp)  # None if not in pullback
    except Exception:
        el = None
    try:
        sm = compute_sma_entry(list(closes), cp)  # None if <60 closes
        # Plausibility gate (mirrors the same guard in entry_price.py's CAN SLIM
        # conservative-entry path): a trailing SMA lags far behind after a large
        # recent move (crash OR rally), so "SMA+offset" stops being a meaningful
        # near-term entry. Observed live: KORU fell ~40%+36% in two crashes; even the
        # SMA20 fallback (used when the Elliott Fib zone is itself invalidated, see
        # compute_elliott_entry's own guard) still sat +37% above current price and
        # was presented as an actionable "SMA20 눌림목" entry. Null the whole `sm`
        # dict when EITHER average has drifted implausibly far — every phase branch
        # below already falls back to plain current-price ("즉시 진입") when sm is
        # None, which has zero gap by construction.
        if sm and (abs(sm["current_vs_sma20"]) > 20.0 or abs(sm["current_vs_sma50"]) > 20.0):
            sm = None
    except Exception:
        sm = None

    def _band(a, b):
        a, b = _r2(a), _r2(b)
        if a is None or b is None:
            return (None, None)
        return (min(a, b), max(a, b))

    if phase == "IMPULSE_W1":
        lvl = _r2(sm["sma20_entry"]) if sm else _r2(cp)
        entry = {"actionable": True, "primary": lvl,
                 "primary_label": "🎯 진입가 SMA20 되돌림" if sm else "🎯 진입가 즉시 (현재가)",
                 "zone_low": None, "zone_high": None,
                 "rationale": ("1파 초기 추세 형성 — SMA20 되돌림 재진입, 추세 확인 후 소량 대응 "
                               "(원저점 이탈 시 가설 폐기).")}

    elif phase == "IMPULSE_W2":
        if el:
            z_lo, z_hi = _band(el["fib_618"], el["fib_500"])
            entry = {"actionable": True, "primary": _r2(el["fib_500"]),
                     "primary_label": "🎯 진입가 W2 되돌림 (Fib 0.500–0.618)",
                     "zone_low": z_lo, "zone_high": z_hi,
                     "rationale": (f"1파 후 2파 되돌림 — Fib 0.5~0.618 구간({z_lo}~{z_hi})에서 "
                                   f"3파 대비 분할 매수 (1파 저점 미침범이 조건, 침범 시 카운트 무효).")}
        elif sm:
            entry = {"actionable": True, "primary": _r2(sm["sma50_entry"]),
                     "primary_label": "🎯 진입가 SMA50 되돌림",
                     "zone_low": None, "zone_high": None,
                     "rationale": "1파 후 2파 되돌림 — 되돌림 구간 미형성, SMA50 지지 재진입으로 3파 대비."}
        else:
            entry = {"actionable": True, "primary": _r2(cp),
                     "primary_label": "🎯 진입가 즉시 (현재가)",
                     "zone_low": None, "zone_high": None,
                     "rationale": "1파 후 2파 되돌림 — 레벨 산출 불가, 현재가 기준 소량 대응."}

    elif phase == "IMPULSE_W3":
        lvl = _r2(cp)
        if el and _safe(el.get("fib_236")) is not None:
            lvl = _r2(min(cp, el["fib_236"]))  # shallow-pullback fill if offered
        z_lo, z_hi = _band(sm["sma20_entry"], cp) if sm else (None, None)  # optional SMA20 add-band
        entry = {"actionable": True, "primary": lvl,
                 "primary_label": "🎯 진입가 즉시 (현재가, 추세추종)",
                 "zone_low": z_lo, "zone_high": z_hi,
                 "rationale": ("3파 강세 확장 — 되돌림 얕아 현재가 추격 가능하나 W2 저점 손절 필수, "
                               "신규 진입은 소량·타이트 스톱 권장.")}

    else:  # IMPULSE_W4
        if el:
            z_lo, z_hi = _band(el["fib_382"], el["fib_236"])
            entry = {"actionable": True, "primary": _r2(el["fib_382"]),
                     "primary_label": "🎯 진입가 W4 눌림목 (Fib 0.382)",
                     "zone_low": z_lo, "zone_high": z_hi,
                     "rationale": (f"3파 후 4파 조정 눌림목 — Fib 0.236~0.382 구간({z_lo}~{z_hi}) "
                                   f"5파 대비 진입, 단 W4는 W1 고점 영역 침범 불가(침범 시 진입 보류).")}
        elif sm:
            entry = {"actionable": True, "primary": _r2(sm["sma20_entry"]),
                     "primary_label": "🎯 진입가 SMA20 눌림목",
                     "zone_low": None, "zone_high": None,
                     "rationale": "3파 후 4파 조정 — 되돌림 미형성, SMA20 눌림목을 5파 대비 진입으로 대용."}
        else:
            entry = {"actionable": True, "primary": _r2(cp),
                     "primary_label": "🎯 진입가 즉시 (현재가)",
                     "zone_low": None, "zone_high": None,
                     "rationale": "3파 후 4파 조정 — 레벨 산출 불가, 현재가 기준."}

    return entry, stop


# ── Impulse / ABC expected type patterns per direction ──
def _patterns(direction):
    if direction == "UP":
        imp = {"1": "H", "2": "L", "3": "H", "4": "L", "5": "H"}
        abc = {"A": "L", "B": "H", "C": "L"}
    else:
        imp = {"1": "L", "2": "H", "3": "L", "4": "H", "5": "L"}
        abc = {"A": "H", "B": "L", "C": "H"}
    return imp, abc


# ─────────────────────────────────────────────────────────────────
# §1 PIVOT DETECTION
# ─────────────────────────────────────────────────────────────────
def build_pivots(highs, lows, close, dates, atr_pct):
    """Fractal pivots → ZigZag amplitude filter → provisional-tail augmentation.

    Returns (piv_list, reason). reason is None on success or a failure string.
    Each pivot dict: {idx, date, price(=close[idx]), type('H'/'L'), label:""}.
    """
    n = len(close)
    n_bars = 3 if n < 90 else (4 if n < 140 else 5)
    raw = _find_swing_pivots(highs, lows, n_bars)
    if len(raw) < 4 and n_bars > 3:
        raw = _find_swing_pivots(highs, lows, n_bars - 1)
    if len(raw) < 3:
        return [], "insufficient_pivots"

    # 1b. price reported = close[idx] (keep pivot on plotted close line); keep type.
    pivots = [{"idx": i, "date": dates[i], "price": float(close[i]),
               "type": t, "label": ""} for (i, _price, t) in raw]

    # 1c. adaptive ZigZag amplitude filter
    ap = atr_pct or 1.0
    min_swing_pct = max(1.0, 0.5 * ap)
    kept = []
    for p in pivots:
        if not kept:
            kept.append(p)
            continue
        last = kept[-1]
        if p["type"] == last["type"]:
            # same-type adjacency → keep the more extreme, drop the other
            if (p["type"] == "H" and p["price"] > last["price"]) or \
               (p["type"] == "L" and p["price"] < last["price"]):
                kept[-1] = p
            continue
        move = abs(p["price"] / last["price"] - 1) * 100 if last["price"] else 0.0
        if move < min_swing_pct:
            # too small a swing — replace last only if p sets a new extreme past it
            prev = kept[-2] if len(kept) >= 2 else None
            if prev is not None and prev["type"] == p["type"]:
                if (p["type"] == "H" and p["price"] > prev["price"]) or \
                   (p["type"] == "L" and p["price"] < prev["price"]):
                    kept[-1] = p  # re-collapse: p extends prev, drops the tiny middle
            # else: drop p entirely (noise)
            continue
        kept.append(p)

    # enforce strict alternation (collapse any surviving same-type adjacency)
    alt = []
    for p in kept:
        if alt and alt[-1]["type"] == p["type"]:
            if (p["type"] == "H" and p["price"] > alt[-1]["price"]) or \
               (p["type"] == "L" and p["price"] < alt[-1]["price"]):
                alt[-1] = p
        else:
            alt.append(p)
    kept = alt

    if not kept:
        return [], "insufficient_pivots"

    # 1d. provisional tail augmentation (fractal can't see last n_bars bars)
    last = kept[-1]
    dir_up = close[-1] > last["price"]
    tail_type = "H" if dir_up else "L"
    move_pct = abs(close[-1] / last["price"] - 1) * 100 if last["price"] else 0.0
    if move_pct > max(1.5, ap) and tail_type != last["type"]:
        kept.append({"idx": n - 1, "date": dates[-1], "price": float(close[-1]),
                     "type": tail_type, "label": "", "provisional": True})

    if len(kept) < 3:
        return kept, "insufficient_pivots"
    return kept, None


# ─────────────────────────────────────────────────────────────────
# §2 DIRECTION & CANDIDATE ANCHORS
# ─────────────────────────────────────────────────────────────────
def choose_anchors(piv, direction, ytd_low_idx, ytd_high_idx):
    """Bounded candidate anchors (<=3), dedup by idx, need >=2 pivots after."""
    cands = []
    if direction == "UP":
        lows_before_high = [p for p in piv if p["type"] == "L" and p["idx"] <= ytd_high_idx]
        # A. global-min-close 'L' at/left of ytd_high_idx
        if lows_before_high:
            a = min(lows_before_high, key=lambda p: p["price"])
            cands.append(a)
        # B. earliest 'L' preceding a >3% net advance to a later 'H'
        for p in piv:
            if p["type"] != "L":
                continue
            later_highs = [q for q in piv if q["type"] == "H" and q["idx"] > p["idx"]]
            if later_highs and any((q["price"] / p["price"] - 1) * 100 > 3.0 for q in later_highs):
                cands.append(p)
                break
        # C. synthetic idx-0 'L' handled in _maybe_add_synth_anchor (caller, §2 case C)
    else:
        highs_before_low = [p for p in piv if p["type"] == "H" and p["idx"] <= ytd_low_idx]
        if highs_before_low:
            a = max(highs_before_low, key=lambda p: p["price"])
            cands.append(a)
        for p in piv:
            if p["type"] != "H":
                continue
            later_lows = [q for q in piv if q["type"] == "L" and q["idx"] > p["idx"]]
            if later_lows and any((1 - q["price"] / p["price"]) * 100 > 3.0 for q in later_lows):
                cands.append(p)
                break

    # dedup by idx, keep only anchors with >=2 pivots after them
    seen, out = set(), []
    for p in cands:
        pos = _pos_of(piv, p["idx"])
        if pos is None or pos in seen:
            continue
        if len(piv) - pos - 1 < 2:
            continue
        seen.add(pos)
        out.append(pos)
    return out[:3]


def _pos_of(piv, idx):
    for k, p in enumerate(piv):
        if p["idx"] == idx:
            return k
    return None


# ─────────────────────────────────────────────────────────────────
# §3 HARD RULES + WALK & LABEL
# ─────────────────────────────────────────────────────────────────
def passes_hard_rules(origin, labels, cur, sign):
    def L(a, b):
        return sign * (b["price"] - a["price"])
    if cur == "2":
        if sign * (labels["2"]["price"] - origin["price"]) <= 0:
            return (False, "R1_W2_ge_100pct_W1")
    if cur == "3":
        if sign * (labels["3"]["price"] - labels["1"]["price"]) <= 0:
            return (False, "R2_W3_no_new_extreme")
    if cur == "5":
        len1 = L(origin, labels["1"])
        len3 = L(labels["2"], labels["3"])
        len5 = L(labels["4"], labels["5"])
        if len3 < len1 and len3 < len5:
            return (False, "R2_W3_shortest")
    if cur == "4":
        tol = 0.01 * abs(labels["1"]["price"])
        if sign * (labels["4"]["price"] - labels["1"]["price"]) < -tol:
            return (False, "R3_W4_overlap_W1")
    return (True, "")


def violates_B_limit(bpv, w5_pivot, sign):
    return sign * (bpv["price"] - w5_pivot["price"]) > 0


def label_from_anchor(piv, anchor_pos, direction):
    """Walk from anchor labeling 1-5 then A-B-C. Returns (status, labels/why, wp)."""
    sign = 1 if direction == "UP" else -1
    imp, abc = _patterns(direction)
    origin = piv[anchor_pos]
    walk = piv[anchor_pos + 1:]
    labels = {}
    wp = {"W0": origin}
    i = 0
    for lab in ["1", "2", "3", "4", "5"]:
        if i >= len(walk):
            break
        pv = walk[i]
        if pv["type"] != imp[lab]:
            break
        trial = {**labels, lab: pv}
        ok, why = passes_hard_rules(origin, trial, lab, sign)
        if not ok:
            if lab == "2":
                return ("REANCHOR", why, None)
            break
        labels[lab] = pv
        wp["W" + lab] = pv
        i += 1

    top_imp = max((int(k) for k in labels), default=0)
    if top_imp >= 3:
        rest = walk[i:]
        last_imp_pivot = labels[str(top_imp)]
        for lab, pv in zip(["A", "B", "C"], rest):
            if pv["type"] != abc[lab]:
                break
            if lab == "B" and violates_B_limit(pv, last_imp_pivot, sign):
                break
            labels[lab] = pv
    return ("OK", labels, wp)


# ─────────────────────────────────────────────────────────────────
# §4 BOUNDED MULTI-ANCHOR SCORE & SELECT
# ─────────────────────────────────────────────────────────────────
def _fib_bonus(origin, labels, sign):
    def L(a, b):
        return abs(b["price"] - a["price"])
    bonus = 0
    if "1" in labels:
        w1 = L(origin, labels["1"])
        if w1 > 0:
            if "2" in labels:
                r = L(labels["1"], labels["2"]) / w1
                bonus += 5 if 0.45 <= r <= 0.68 else (2 if r < 1.0 else 0)
            if "3" in labels and "2" in labels:
                r = L(labels["2"], labels["3"]) / w1
                bonus += 5 if 1.4 <= r <= 2.1 else (2 if r > 1.0 else 0)
            if "5" in labels and "4" in labels:
                r = L(labels["4"], labels["5"]) / w1
                bonus += 2 if 0.6 <= r <= 1.7 else 0
    if "3" in labels and "4" in labels and "2" in labels:
        w3 = L(labels["2"], labels["3"])
        if w3 > 0:
            r = L(labels["3"], labels["4"]) / w3
            bonus += 3 if 0.25 <= r <= 0.50 else 0
    return bonus


def _score_labeling(origin, labels, sign):
    n_imp = sum(1 for k in labels if k in ("1", "2", "3", "4", "5"))
    n_abc = sum(1 for k in labels if k in ("A", "B", "C"))
    return 10 * n_imp + 3 * n_abc + _fib_bonus(origin, labels, sign)


def _next_anchor_pos(piv, anchor_pos, direction):
    """For REANCHOR: next-lower 'L' (UP) / next-higher 'H' (DOWN) before anchor_pos+ region."""
    ttype = "L" if direction == "UP" else "H"
    cur = piv[anchor_pos]
    best = None
    for k, p in enumerate(piv):
        if p["type"] != ttype or k == anchor_pos:
            continue
        if direction == "UP" and p["price"] < cur["price"]:
            if best is None or p["price"] < piv[best]["price"]:
                best = k
        elif direction == "DOWN" and p["price"] > cur["price"]:
            if best is None or p["price"] > piv[best]["price"]:
                best = k
    return best


def select_labeling(piv, anchor_positions, direction, ytd_low_idx, ytd_high_idx):
    """Run label_from_anchor per candidate; pick highest score. Returns dict or None."""
    sign = 1 if direction == "UP" else -1
    ext_idx = ytd_low_idx if direction == "UP" else ytd_high_idx
    results = []
    for pos in anchor_positions:
        status, labels, wp = label_from_anchor(piv, pos, direction)
        if status == "REANCHOR":
            nxt = _next_anchor_pos(piv, pos, direction)
            if nxt is not None and len(piv) - nxt - 1 >= 2:
                status, labels, wp = label_from_anchor(piv, nxt, direction)
                if status == "OK" and labels:
                    pos = nxt
                else:
                    continue
            else:
                continue
        if status != "OK" or not labels:
            continue
        origin = piv[pos]
        score = _score_labeling(origin, labels, sign)
        results.append({
            "score": score, "labels": labels, "wp": wp, "anchor_pos": pos,
            "origin": origin,
            "_tie_extreme": abs(origin["idx"] - ext_idx),
        })
    if not results:
        return None
    # TIE-BREAK: score desc → nearer YTD extreme → earliest idx → deepest/highest price
    def keyf(r):
        px = -r["origin"]["price"] if direction == "UP" else r["origin"]["price"]
        return (-r["score"], r["_tie_extreme"], r["origin"]["idx"], px)
    results.sort(key=keyf)
    best = results[0]
    if best["score"] < 10:
        return None  # confidence floor: no defensible count
    return best


# ─────────────────────────────────────────────────────────────────
# §5 CURRENT_PHASE
# ─────────────────────────────────────────────────────────────────
def decide_current_phase(labels, origin, close_last, direction, from_high_pct):
    sign = 1 if direction == "UP" else -1
    imp_present = [k for k in ("1", "2", "3", "4", "5") if k in labels]
    last_lab = imp_present[-1] if imp_present else None
    has_A = "A" in labels
    has_B = "B" in labels
    has_C = "C" in labels

    # last assigned label's pivot (ABC take priority for "last" when present)
    if has_C:
        last_pivot = labels["C"]
    elif has_B:
        last_pivot = labels["B"]
    elif has_A:
        last_pivot = labels["A"]
    elif last_lab:
        last_pivot = labels[last_lab]
    else:
        last_pivot = origin
    leg_now = sign * (close_last - last_pivot["price"])

    # DECISION TABLE (first match wins) — uses the appropriate leg reference
    if has_C:
        phase = "UNCLEAR"
    elif has_B and (sign * (close_last - labels["A"]["price"]) < 0):
        phase = "CORRECTIVE_C"
    elif has_A and (sign * (close_last - labels["A"]["price"]) > 0):
        phase = "CORRECTIVE_B"
    elif last_lab == "5" and leg_now < 0:
        phase = "CORRECTIVE_A"
    elif last_lab == "5" and leg_now >= 0:
        phase = "IMPULSE_W5"
    elif last_lab == "4" and leg_now > 0:
        phase = "IMPULSE_W5"
    elif last_lab == "3" and leg_now < 0:
        phase = "IMPULSE_W4"
    elif last_lab == "3" and leg_now >= 0:
        phase = "IMPULSE_W3"
    elif last_lab == "2" and leg_now > 0:
        phase = "IMPULSE_W3"
    elif last_lab == "1" and leg_now < 0:
        phase = "IMPULSE_W2"
    elif last_lab == "1" and leg_now >= 0:
        phase = "IMPULSE_W1"
    else:
        phase = "UNCLEAR"

    # CORROBORATION (soft): demote clearly-contradicted table hits toward legibility
    if phase == "IMPULSE_W5" and from_high_pct is not None and from_high_pct < -8:
        phase = "CORRECTIVE_A"
    return phase


# ─────────────────────────────────────────────────────────────────
# §7 INTERPRETATION STRING RECIPE
# ─────────────────────────────────────────────────────────────────
def _interpretation(phase, ctx):
    A = ctx["anchor_date"]
    R = ctx["ytd_return_pct"]
    Lp = ctx["from_low_pct"]
    H = ctx["from_high_pct"]
    C = ctx["confidence"]
    w2w1 = ctx["w2w1"]
    w3w1 = ctx["w3w1"]
    w5w1 = ctx["w5w1"]
    n_pivots = ctx["n_pivots"]
    T = {
        "IMPULSE_W1": (
            f"{A} 저점에서 새 상승 추세가 태동 중 — YTD 저점 대비 +{Lp:.0f}%, 고점 대비 {H:.0f}%. "
            f"1파 초기 국면으로 추세 확인 후 대응 (신뢰도 {C:.0%})."),
        "IMPULSE_W2": (
            f"{A} 저점에서 시작한 임펄스의 1파 후 2파 되돌림 진행 — 저점 대비 +{Lp:.0f}%, 고점 대비 {H:.0f}%. "
            f"2파는 통상 1파의 0.5~0.618배 되돌리며(현 {w2w1}배), 3파 진입 전 재매수 관찰 구간 (신뢰도 {C:.0%})."),
        "IMPULSE_W3": (
            f"{A} 저점에서 시작한 상승 임펄스의 3파가 진행 중 — 저점 대비 +{Lp:.0f}%, 고점 대비 {H:.0f}%. "
            f"3파는 통상 1파의 1.618배로 확장(현 {w3w1}배)되며 가장 강한 구간, 추세 추종 유효 (신뢰도 {C:.0%})."),
        "IMPULSE_W4": (
            f"{A} 저점 기반 임펄스의 3파 후 4파 조정 — 저점 대비 +{Lp:.0f}%, 고점 대비 {H:.0f}%. "
            f"4파는 통상 3파의 0.382배 되돌리는 눌림목이며 1파 영역을 침범하지 않아야 함, 5파 대비 관찰 (신뢰도 {C:.0%})."),
        "IMPULSE_W5": (
            f"{A} 저점 기반 임펄스의 5파 종반 — 저점 대비 +{Lp:.0f}%, 고점 대비 {H:.0f}%. "
            f"5파는 통상 1파와 등장(현 {w5w1}배)하거나 3파의 0.618배, 신고가 갱신 중이나 상승 여력 제한적 — "
            f"이익 실현/과열 경계 구간 (신뢰도 {C:.0%})."),
        "CORRECTIVE_A": (
            f"{A} 기점 상승 사이클 종료 후 첫 하락(A파) — 저점 대비 +{Lp:.0f}%, 고점 대비 {H:.0f}%. "
            f"고점에서 이탈한 첫 조정 파동, 반등(B파) 후 추가 하락 가능성 관찰 (신뢰도 {C:.0%})."),
        "CORRECTIVE_B": (
            f"ABC 조정의 B파 반등 — 고점 대비 {H:.0f}%, 저점 대비 +{Lp:.0f}%. "
            f"약세 되돌림(반등 함정) 구간으로 B파 실패 후 C파 하락 경계 (신뢰도 {C:.0%})."),
        "CORRECTIVE_C": (
            f"ABC 조정의 C파 하락 — 고점 대비 {H:.0f}%, 저점 대비 +{Lp:.0f}%. "
            f"C파는 통상 A파와 유사 길이, 조정 마무리 후 새 임펄스 형성 관찰 (신뢰도 {C:.0%})."),
        "UNCLEAR": (
            f"YTD {R:+.0f}% 구간에서 임펄스/조정 어느 쪽으로도 깨끗한 카운트 미형성 — "
            f"스윙 피벗 {n_pivots}개, 횡보/비정형 구조로 확정 카운트 불가. 방향성 확인 후 재평가."),
    }
    return T.get(phase, T["UNCLEAR"])


# ─────────────────────────────────────────────────────────────────
# UNCLEAR result builder (§6 ladder terminus)
# ─────────────────────────────────────────────────────────────────
def _unclear_result(ticker, name, ctx, piv, reason):
    n_pivots = len(piv) if piv else ctx.get("n_pivots", 0)
    ictx = {**ctx, "confidence": 0.0, "anchor_date": ctx.get("ytd_start", ""),
            "w2w1": "-", "w3w1": "-", "w5w1": "-", "n_pivots": n_pivots}
    # ticker + real series exist → still compute the STOP (works regardless of
    # wave count) so the downside-risk reference renders; entry is 관망.
    cp = None
    cl = ctx.get("close") or []
    if cl:
        cp = _safe(cl[-1])
    entry, stop = compute_entry_stop("UNCLEAR", ticker, [], [], [], cp)  # highs/lows unused for UNCLEAR
    return {
        "ticker": ticker, "name": name, "currency": _currency_for(ticker)[0], "currency_symbol": _currency_for(ticker)[1],
        "ytd_start": ctx.get("ytd_start", ""), "as_of": ctx.get("as_of", ""),
        "dates": ctx.get("dates", []), "close": ctx.get("close", []),
        "pivots": [{"idx": p["idx"], "date": p["date"], "price": _safe(p["price"]),
                    "type": p["type"], "label": ""} for p in (piv or [])],
        "current_phase": "UNCLEAR", "phase_color": "gray",
        "phase_label": PHASE_LABEL["UNCLEAR"],
        "interpretation": _interpretation("UNCLEAR", ictx),
        "ytd_return_pct": ctx.get("ytd_return_pct"),
        "from_high_pct": ctx.get("from_high_pct"),
        "from_low_pct": ctx.get("from_low_pct"),
        "confidence": 0.0, "anchor_date": None, "n_pivots": n_pivots,
        "current_price": (_r2(cp) if cp is not None else None),
        "entry": entry,
        "stop": stop,
        "_reason": reason,
    }


def _stub_result(ticker, name):
    # NO data → NO yfinance call for entry/stop (the sub-download would also
    # fail/waste a request). Hard-null both, inline the _stop_null() literal.
    return {
        "ticker": ticker, "name": name, "currency": _currency_for(ticker)[0], "currency_symbol": _currency_for(ticker)[1],
        "ytd_start": "", "as_of": "", "dates": [], "close": [], "pivots": [],
        "current_phase": "UNCLEAR", "phase_color": "gray",
        "phase_label": "파동 카운트 불명확 (횡보/비정형 구조)",
        "interpretation": "가격 데이터 로드 실패 또는 데이터 부족 — 파동 카운트 불가.",
        "ytd_return_pct": None, "from_high_pct": None, "from_low_pct": None,
        "confidence": 0.0, "anchor_date": None, "n_pivots": 0,
        "current_price": None,
        "entry": {"actionable": False, "primary": None,
                  "primary_label": "관망 (데이터 없음)", "zone_low": None,
                  "zone_high": None,
                  "rationale": "가격 데이터 로드 실패 — 진입/손절 산출 불가."},
        "stop": {"price": None, "pct": None, "type": "NONE", "rationale": "손절 계산 불가"},
    }


# ─────────────────────────────────────────────────────────────────
# PER-TICKER PIPELINE
# ─────────────────────────────────────────────────────────────────
def _compute_one(ticker, name, year, disk_cache=None, period="ytd"):
    """Compute one index contract. Reads the date-keyed disk cache first: a fresh
    entry for f"{ticker}:{period}:{today}" short-circuits the yfinance fetches. Every
    computed result is written back into disk_cache (in-place) under the same key.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"{ticker}:{period}:{today}"
    if disk_cache is not None:
        hit = disk_cache.get(cache_key)
        if isinstance(hit, dict) and isinstance(hit.get("result"), dict):
            res = dict(hit["result"])
            res["name"] = name  # keep the label authoritative from the plan
            return res

    res = _compute_one_fresh(ticker, name, year, period)
    res["period"] = period
    res["period_label"] = _period_cfg(period)["label"]

    # 2026-07-30 fix: 빈 가격시계열(close=[])은 캐싱하지 않는다. yfinance db-lock으로
    # 페치가 실패하면 res.close가 비는데, 이걸 캐싱하면 전이적 락이 하루 종일 고착되어
    # 파동 차트/카운트가 계속 빈 채로 남는다(대시보드 '가격 데이터 없음'). 빈 결과는
    # 캐싱을 건너뛰어 다음 호출이 재페치(사실상 재시도)하도록 한다.
    if disk_cache is not None and (res.get("close") or []):
        disk_cache[cache_key] = {
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "result": res,
        }
    return res


def _compute_one_fresh(ticker, name, year, period="ytd"):
    # THREAD SAFETY: yfinance 1.2.0's multi-ticker yf.download() aggregator has a
    # shared results path that cross-contaminates data (returns the wrong ticker's
    # bars) under a ThreadPoolExecutor — empirically verified. yf.Ticker().history()
    # is the per-object path and is thread-safe here, so we use it for our fetch.
    import yfinance as yf
    cfg = _period_cfg(period)
    start_str = (f"{year}-01-01" if cfg["calendar_days"] is None
                 else (datetime.now() - timedelta(days=cfg["calendar_days"])).strftime("%Y-%m-%d"))
    try:
        df = yf.Ticker(ticker).history(start=start_str, interval="1d",
                                       auto_adjust=True)
    except Exception:
        return _stub_result(ticker, name)
    if df is None or df.empty:
        return _stub_result(ticker, name)
    if hasattr(df.columns, "get_level_values"):
        try:
            df.columns = df.columns.get_level_values(0)
        except Exception:
            pass
    df = df.dropna(subset=["High", "Low", "Close"])
    if len(df) < cfg["min_bars"]:
        return _stub_result(ticker, name)

    highs = df["High"].values
    lows = df["Low"].values
    close = df["Close"].values
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    if len(close) < cfg["pref_bars"]:
        # build minimal ctx for UNCLEAR stub with real series
        ctx0 = _series_ctx(dates, close)
        return _unclear_result(ticker, name, ctx0, [], "insufficient_bars")

    # ATR20 %-of-price (verbatim block)
    atr_pct = None
    try:
        if len(close) >= 21:
            trs = []
            for i in range(-20, 0):
                tr = max(highs[i] - lows[i],
                         abs(highs[i] - close[i - 1]),
                         abs(lows[i] - close[i - 1]))
                trs.append(tr)
            atr = sum(trs) / len(trs)
            cur = float(close[-1])
            if math.isfinite(atr) and atr > 0 and cur > 0:
                atr_pct = atr / cur * 100
    except Exception:
        atr_pct = None

    ctx = _series_ctx(dates, close)
    ytd_low_idx = ctx["ytd_low_idx"]
    ytd_high_idx = ctx["ytd_high_idx"]

    # §1 pivots
    piv, reason = build_pivots(highs, lows, close, dates, atr_pct)
    if reason == "insufficient_pivots":
        return _unclear_result(ticker, name, ctx, piv, reason)

    # §2 direction
    R = ctx["ytd_return_pct"]
    up_vote = ytd_high_idx > ytd_low_idx
    ret_up = R > 0
    if abs(R) < 2.0 and abs(ctx["from_high_pct"]) < 3 and ctx["from_low_pct"] < 3:
        return _unclear_result(ticker, name, ctx, piv, "no_dominant_trend")
    if up_vote and ret_up:
        direction = "UP"
    elif (not up_vote) and (not ret_up):
        direction = "DOWN"
    else:
        direction = "UP" if ret_up else "DOWN"

    # §2 anchors + synthetic idx-0 handling
    anchor_positions = choose_anchors(piv, direction, ytd_low_idx, ytd_high_idx)
    # synthetic idx-0 candidate C (spec §2 case C for UP / mirror for DOWN)
    _maybe_add_synth_anchor(piv, direction, close, ctx, anchor_positions)
    if not anchor_positions:
        return _unclear_result(ticker, name, ctx, piv, "no_anchor")

    # §3-4 select best labeling
    best = select_labeling(piv, anchor_positions, direction, ytd_low_idx, ytd_high_idx)
    if best is None:
        return _unclear_result(ticker, name, ctx, piv, "low_confidence")

    # §4 MOST-RECENT-CYCLE guard
    labels, origin, anchor_pos = best["labels"], best["origin"], best["anchor_pos"]
    if all(k in labels for k in ("1", "2", "3", "4", "5", "A", "B", "C")):
        c_pos = _pos_of(piv, labels["C"]["idx"])
        if c_pos is not None and len(piv) - c_pos - 1 >= 2:
            re_best = select_labeling(piv, [c_pos], direction, ytd_low_idx, ytd_high_idx)
            if re_best is not None:
                best = re_best
                labels, origin, anchor_pos = best["labels"], best["origin"], best["anchor_pos"]

    sign = 1 if direction == "UP" else -1
    confidence = min(1.0, best["score"] / 40.0)

    # §5 current phase
    phase = decide_current_phase(labels, origin, float(close[-1]), direction,
                                 ctx["from_high_pct"])

    # apply labels onto the pivot list
    lab_by_idx = {pv["idx"]: lab for lab, pv in labels.items()}
    out_pivots = []
    for p in piv:
        out_pivots.append({
            "idx": p["idx"], "date": p["date"], "price": _safe(p["price"]),
            "type": p["type"], "label": lab_by_idx.get(p["idx"], ""),
        })

    # fib ratio context for interpretation strings.
    # Each wave leg = |end.price - start.price|; ratio is leg / W1 leg.
    #   W1 = |W1 - origin|, W2 = |W2 - W1|, W3 = |W3 - W2|, W5 = |W5 - W4|.
    def _leg(a_lab, b_pivot):
        a = origin if a_lab == "W0" else labels.get(a_lab)
        if a is None or b_pivot is None:
            return None
        return abs(b_pivot["price"] - a["price"])

    w1_leg = _leg("W0", labels.get("1"))

    def _ratio(leg):
        if leg is None or w1_leg is None or w1_leg == 0:
            return "-"
        return f"{leg / w1_leg:.2f}"

    w2w1 = _ratio(_leg("1", labels.get("2")))
    w3w1 = _ratio(_leg("2", labels.get("3")))
    w5w1 = _ratio(_leg("4", labels.get("5")))

    ictx = {
        "anchor_date": origin["date"], "ytd_return_pct": R,
        "from_low_pct": ctx["from_low_pct"], "from_high_pct": ctx["from_high_pct"],
        "confidence": confidence, "w2w1": w2w1, "w3w1": w3w1, "w5w1": w5w1,
        "n_pivots": len(piv),
    }

    # ── ENTRY / STOP (deterministic; reuses YTD arrays already in hand) ──
    cur_price = float(close[-1])
    entry, stop = compute_entry_stop(
        phase, ticker, highs.tolist(), lows.tolist(), close.tolist(), cur_price,
        labels_ctx={"from_high_pct": ctx["from_high_pct"], "confidence": confidence})

    return {
        "ticker": ticker, "name": name, "currency": _currency_for(ticker)[0], "currency_symbol": _currency_for(ticker)[1],
        "ytd_start": ctx["ytd_start"], "as_of": ctx["as_of"],
        "dates": ctx["dates"], "close": ctx["close"],
        "pivots": out_pivots,
        "current_phase": phase, "phase_color": PHASE_COLOR.get(phase, "gray"),
        "phase_label": PHASE_LABEL.get(phase, PHASE_LABEL["UNCLEAR"]),
        "interpretation": _interpretation(phase, ictx),
        "ytd_return_pct": ctx["ytd_return_pct"],
        "from_high_pct": ctx["from_high_pct"], "from_low_pct": ctx["from_low_pct"],
        "confidence": round(confidence, 2), "anchor_date": origin["date"],
        "n_pivots": len(piv),
        "current_price": _r2(cur_price),
        "entry": entry,
        "stop": stop,
    }


def _maybe_add_synth_anchor(piv, direction, close, ctx, anchor_positions):
    """§2 case C: synthetic idx-0 anchor if close[0] near YTD extreme or no valid anchor left."""
    if len(anchor_positions) >= 3:
        return
    first = piv[0]
    want_type = "L" if direction == "UP" else "H"
    if first["type"] != want_type:
        return
    pos0 = 0
    if pos0 in anchor_positions:
        return
    if len(piv) - pos0 - 1 < 2:
        return
    if direction == "UP":
        near_extreme = abs(close[0] / ctx["ytd_low"] - 1) * 100 < 2.0 if ctx["ytd_low"] else False
        has_L_left = any(p["type"] == "L" and p["idx"] < ctx["ytd_high_idx"] for p in piv[1:])
        if near_extreme or not has_L_left:
            anchor_positions.append(pos0)
    else:
        near_extreme = abs(close[0] / ctx["ytd_high"] - 1) * 100 < 2.0 if ctx["ytd_high"] else False
        has_H_left = any(p["type"] == "H" and p["idx"] < ctx["ytd_low_idx"] for p in piv[1:])
        if near_extreme or not has_H_left:
            anchor_positions.append(pos0)


def _series_ctx(dates, close):
    close_list = [_safe(c) for c in close]
    c0 = float(close[0])
    cN = float(close[-1])
    ytd_high = float(max(close))
    ytd_low = float(min(close))
    ytd_high_idx = int(max(range(len(close)), key=lambda i: close[i]))
    ytd_low_idx = int(min(range(len(close)), key=lambda i: close[i]))
    ytd_return_pct = (cN / c0 - 1) * 100 if c0 else 0.0
    from_high_pct = (cN / ytd_high - 1) * 100 if ytd_high else 0.0
    from_low_pct = (cN / ytd_low - 1) * 100 if ytd_low else 0.0
    return {
        "dates": dates, "close": close_list,
        "ytd_start": dates[0], "as_of": dates[-1],
        "ytd_return_pct": round(ytd_return_pct, 2),
        "from_high_pct": round(from_high_pct, 2),
        "from_low_pct": round(from_low_pct, 2),
        "ytd_high": ytd_high, "ytd_low": ytd_low,
        "ytd_high_idx": ytd_high_idx, "ytd_low_idx": ytd_low_idx,
    }


# ─────────────────────────────────────────────────────────────────
# CACHE (date-keyed, TTL 24h — mirrors elliott_wave_stops pattern)
# ─────────────────────────────────────────────────────────────────
def _load_cache():
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache):
    try:
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    except Exception:
        pass


# Simple in-process module cache keyed by today's date
_MEM_CACHE = {}


def _compute_waves_core(members, disk, year, period="ytd"):
    """Shared parallel core. `members` is a list of (ticker, name) tuples.

    Runs _compute_one concurrently over a ThreadPoolExecutor(max_workers=8) — each cold
    call triggers 2 yfinance downloads (wave OHLC + stop). Output order is deterministic
    (futures mapped back to input order via enumerate — completion order never leaks). A
    per-ticker failure degrades to _stub_result and never propagates. `disk` is the
    date-keyed disk-cache dict, mutated in-place by _compute_one (caller persists it).

    Returns a list of result dicts in EXACTLY `members` order (1:1, duplicates preserved).
    Does NOT map by ticker — the same ticker may appear once here (index groups) or as a
    genuine duplicate elsewhere; positional mapping is the only correct contract.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _work(item):
        idx, (ticker, name) = item
        try:
            res = _compute_one(ticker, name, year, disk_cache=disk, period=period)
        except Exception as e:
            res = _stub_result(ticker, name)
            res["period"], res["period_label"] = period, _period_cfg(period)["label"]
            res["_error"] = str(e)[:200]
        return idx, res

    if not members:
        return []

    # Submit all, then read futures back and re-sort by the submitted index →
    # deterministic input-order output regardless of completion order.
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_work, item) for item in enumerate(members)]
        pairs = [f.result() for f in futures]
    pairs.sort(key=lambda p: p[0])
    return [res for _idx, res in pairs]


def compute_waves_for_tickers(members, refresh: bool = False, period: str = "ytd") -> dict:
    """Public entry — flat contract dict for an ARBITRARY (ticker, name) list.

    Reuses the exact machinery compute_index_waves uses: the shared _compute_waves_core
    parallel executor over _compute_one, the date-keyed disk cache (read/write, so
    tickers overlapping the fixed 25 are free cache hits), the _STOP_LOCK serialization
    inside _compute_one, and the per-ticker try/except → _stub_result. Output order ==
    `members` order.

    `period`: "ytd" (default) or "1m" — see PERIOD_CONFIG. Invalid values fall back to
    "ytd". The disk cache key already includes period, so YTD and 1M results for the
    same ticker never collide.

    No mem-cache here: the caller list is dynamic (buy-list ETFs change on each swarm /
    final-list refresh), so we rely on the per-ticker disk cache for same-day speed and
    skip the day-keyed _MEM_CACHE (which is keyed only by date, not by ticker set).

    Returns {as_of, ytd_start, period, period_label, generated_at, indices:[...]}.
    members=[] → indices:[].
    """
    period = period if period in PERIOD_CONFIG else DEFAULT_PERIOD
    year = datetime.now().year
    disk = {} if refresh else _load_cache()

    results = _compute_waves_core(list(members), disk, year, period)

    as_of = ""
    ytd_start = ""
    for res in results:
        if res.get("as_of"):
            as_of = res["as_of"]
        if res.get("ytd_start"):
            ytd_start = res["ytd_start"]

    out = {
        "as_of": as_of,
        "ytd_start": ytd_start,
        "period": period,
        "period_label": _period_cfg(period)["label"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "indices": results,
    }

    # Persist the disk cache (mutated in-place by _compute_one). Even on refresh=True
    # this writes fresh entries so a later index call / same-ticker call hits them.
    _save_cache(disk)
    return out


def compute_index_waves(refresh: bool = False, period: str = "ytd") -> dict:
    """Public entry — grouped + flat contract dict for all 25 tickers (broad → sector
    → leveraged, plan order preserved within each group).

    `period`: "ytd" (default) or "1m" — see PERIOD_CONFIG. Invalid values fall back to
    "ytd". The in-process _MEM_CACHE is keyed by (date, period) so switching the period
    toggle never serves the other period's cached payload.

    Per-ticker computation runs concurrently via the shared _compute_waves_core
    executor (ThreadPoolExecutor(max_workers=8)) since each cold _compute_one triggers
    2 yfinance downloads (wave OHLC + stop). Output order is deterministic (futures
    mapped back to plan order — completion order never leaks). A per-ticker failure
    degrades to _stub_result and never fails the endpoint.
    """
    period = period if period in PERIOD_CONFIG else DEFAULT_PERIOD
    today = datetime.now().strftime("%Y-%m-%d")
    mem_key = f"{today}:{period}"
    if not refresh and mem_key in _MEM_CACHE:
        return _MEM_CACHE[mem_key]

    year = datetime.now().year
    disk = {} if refresh else _load_cache()

    # Flat plan in group order: (ticker, name), with a parallel group-key list so we
    # can tag + regroup after the positional-ordered core returns.
    flat_members = [(t, n) for g in GROUPS for (t, n) in g["members"]]
    flat_gkeys = [g["key"] for g in GROUPS for (_t, _n) in g["members"]]

    results = _compute_waves_core(flat_members, disk, year, period)

    # Tag each result with its group key (positional 1:1 with flat_members / flat_gkeys).
    for res, gkey in zip(results, flat_gkeys):
        res["group"] = gkey

    # Map ticker → computed result (tickers are unique across the whole plan).
    by_ticker = {res["ticker"]: res for res in results}

    as_of = ""
    ytd_start = ""
    for res in results:
        if res.get("as_of"):
            as_of = res["as_of"]
        if res.get("ytd_start"):
            ytd_start = res["ytd_start"]

    # Grouped assembly (each group carries its index dicts in member order).
    groups_out = []
    indices_flat = []
    for g in GROUPS:
        g_indices = [by_ticker[t] for (t, _n) in g["members"]]
        groups_out.append({
            "key": g["key"], "label": g["label"], "emoji": g["emoji"],
            "default_open": g["default_open"], "indices": g_indices,
        })
        indices_flat.extend(g_indices)

    out = {
        "as_of": as_of,
        "ytd_start": ytd_start,
        "period": period,
        "period_label": _period_cfg(period)["label"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "groups": groups_out,
        "indices": indices_flat,  # backward-compat: all 25 flat, group order
    }

    # Persist disk cache (disk was mutated in-place by _compute_one under the
    # f"{ticker}:{period}:{today}" key; a single write covers all newly-computed tickers).
    _save_cache(disk)

    _MEM_CACHE[mem_key] = out
    return out


if __name__ == "__main__":
    d = compute_index_waves(refresh=True)
    print(f"n_indices = {len(d['indices'])}  as_of={d['as_of']}  ytd_start={d['ytd_start']}")
    print("groups = " + ", ".join(f"{g['key']}({len(g['indices'])})" for g in d["groups"]))
    for x in d["indices"]:
        nb = len(x.get("close") or [])
        print(f"  {x['ticker']:5s}  grp={x.get('group',''):10s}  n_bars={nb:4d}  "
              f"phase={x['current_phase']:12s}  n_pivots={x['n_pivots']:2d}  "
              f"ytd={x['ytd_return_pct']}  conf={x['confidence']}")
