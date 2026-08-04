# -*- coding: utf-8 -*-
"""position_state.py — Stateful Hysteresis Signal System.

Solves the daily flip-flop problem (BUY → NEUTRAL → BUY on consecutive days).
Tracks each (ticker, horizon) pair through a state machine with hysteresis:

  PROSPECTING (watching)
       ↓  signal=BUY_NOW × 2 consecutive days
   ENTERED (just bought — 1-day transient)
       ↓  next day (auto)
   HOLDING (carrying position — STAYS unless STRICT exit condition)
       ↓  signal=SKIP × 2 days OR WAIT × 5 days (signal degradation)
   EXIT_PENDING (exit flagged — 1-day transient)
       ↓  next day (auto)
   EXITED (closed) → dropped from monitoring

KEY INSIGHT: Once HOLDING, daily WAIT/SKIP flips are IGNORED unless they
persist or escalate. This eliminates the BUY→NEUTRAL→BUY flip-flop.

Persistence: .position_state.json (per-ticker × per-horizon state).
Alerts: generated only on state transitions (NEW_BUY, EXIT_TRIGGER, etc.)
"""
from __future__ import annotations

import json
import math as _math
import time
from pathlib import Path
from typing import Optional

STATE_PATH = Path(".position_state.json")

# ─── State definitions ──────────────────────────────────────────────
class State:
    PROSPECTING  = "PROSPECTING"    # Watching, not yet entered
    ENTERED      = "ENTERED"        # Just entered (transient, 1 day)
    HOLDING      = "HOLDING"        # Carrying position (sticky default)
    EXIT_PENDING = "EXIT_PENDING"   # Exit triggered (transient, 1 day)
    EXITED       = "EXITED"         # Closed (terminal)
    DROPPED      = "DROPPED"        # SKIP × 3 days while PROSPECTING

# ─── Hysteresis parameters ──────────────────────────────────────────
ENTRY_CONFIRMATION_DAYS = 2  # BUY_NOW for N days to enter
EXIT_CONFIRMATION_DAYS  = 2  # (deprecated) legacy SKIP-based exit — superseded by thesis-break
WAIT_TOLERANCE_DAYS     = 5  # (deprecated) legacy WAIT-based exit — superseded by thesis-break
DROP_DAYS               = 3  # SKIP for N days while PROSPECTING = drop

# ─── Thesis-break exit (2026-07) — replaces mechanical SKIP/WAIT exits ──
# A HOLDING position now exits ONLY when its BUY THESIS objectively breaks —
# measured against live scan metrics + stop — NOT because it wasn't re-picked
# (SKIP) or is indecisive (WAIT). This honors the 21d core hold intent: a still
# healthy position (bullish/neutral classification, composite above floor, above
# its stop) is HELD even when it drops out of today's top-N picks.
MIN_CORE_HOLD_DAYS   = 21   # soft breaks suppressed before this many days held
COMPOSITE_EXIT_FLOOR = 45   # composite below this = technical strength gone (soft break)
SOFT_CONFIRM_DAYS    = 2    # a soft break needs N DISTINCT days of confirmation

# ─── T5: give-back latch (이익반납 방어, 2026-07-28) ──────────────────
# The ONLY rule that watches "peak-to-current profit give-back". T1~T4 only see
# downtrend/stop/regime; a winner that round-trips back to breakeven (the zombie
# corridor) escapes all of them. One-way ARM: a position must first have been up
# ≥ arm-threshold (so a never-profitable name can NEVER fire → no early-exit of
# losers). SHADOW-FIRST: while _GIVEBACK_SHADOW is True the latch is COMPUTED and
# annotated (giveback_fired/reason) but does NOT trigger EXIT_PENDING — flip to
# False only after a 4~6주 shadow period with acceptable false-fire rate.
_GIVEBACK_SHADOW       = True
GIVEBACK_ARM_MFE       = 10.0   # arm once peak profit (MFE) ≥ 10% (mature: days_held≥21)
GIVEBACK_ARM_MFE_EARLY = 15.0   # higher bar before min-hold (avoid arming on entry noise)
GIVEBACK_ARM_ATR_MULT  = 1.25   # or 1.25·atr_pct·√21 (vol-scaled), whichever is larger
GIVEBACK_RETAIN_FLOOR  = 0.50   # a "give-back day" = retained profit (pnl/mfe) < 50%
GIVEBACK_CONFIRM_DAYS  = 3      # fire after ≥3 give-back days …
GIVEBACK_WINDOW_DAYS   = 5      # … within a rolling 5-day window
# Herding-Reversal give-back AMPLIFIER (paper 2607.27063) — when a HELD ticker is flagged
# REVERSAL_RISK (과밀 run-up+RSI + herding 약화=단기 균열; validated −7%p give-back at 42d),
# arm the latch EARLIER and fire on a SMALLER give-back. Amplifier only (never a new exit
# path); inherits _GIVEBACK_SHADOW. Rare conjunction → touches very few positions.
HRR_GIVEBACK_ARM_MULT     = 0.7   # arm at 70% of the normal (vol-scaled) bar — earlier
HRR_GIVEBACK_RETAIN_FLOOR = 0.65  # more sensitive give-back day (retain < 65% vs 50%)
# Bearish classification keywords (mirrors api.py) — HARD, immediate thesis break.
# Substring-matched so emoji-prefixed values ("⬇️ DOWNTREND") still match.
BEARISH_CLS = ("DOWNTREND", "WEAKENING", "FADING", "EXHAUSTING", "CYCLE_PEAK", "COUNTER_RALLY")

# ─── Alert types ────────────────────────────────────────────────────
class Alert:
    NEW_BUY          = "🆕 NEW BUY"
    EXIT_TRIGGER     = "⚠ EXIT TRIGGER"
    SIGNAL_DEGRADED  = "📊 SIGNAL DEGRADED"
    PROSPECT_DROPPED = "✗ PROSPECT DROPPED"
    EXIT_RESCINDED   = "↩️ EXIT RESCINDED"

# ─── Telemetry (L1) ─────────────────────────────────────────────────
DAILY_LOG_CAP        = 90   # per-position daily_log rows kept (≈ 4 review cycles)
NO_SCAN_ALERT_DAYS   = 3    # HOLDING with no scan data N consecutive days → alert
BLIND_PENDING_MAX    = 5    # EXIT_PENDING runs without scan data before alerting


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def load_state() -> dict:
    """Load .position_state.json. Returns empty structure if missing."""
    if not STATE_PATH.exists():
        return {"positions": {}, "last_update": None, "alerts_history": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"positions": {}, "last_update": None, "alerts_history": []}


def save_state(state: dict) -> None:
    state["last_update"] = _now()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                          encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# State transition logic
# ─────────────────────────────────────────────────────────────────────

def _empty_position(ticker: str, horizon: str) -> dict:
    """Initial state when a ticker first appears in PM picks."""
    return {
        "ticker": ticker, "horizon": horizon,
        "state": State.PROSPECTING,
        "first_seen": _today(),
        "entered_date":   None,
        "entered_signal": None,
        "consecutive_signal_days": {"BUY_NOW": 0, "WAIT": 0, "SKIP": 0},
        "days_in_state": 0,
        "state_history": [{"date": _today(), "to": State.PROSPECTING,
                            "reason": "initial appearance in PM picks"}],
        "last_signal": None,
        "last_alert":  None,
        "entered_price":    None,   # fill price at ENTERED (stop/drawdown reference)
        "last_eval_date":   None,   # date-dedup guard: day-counters advance once/day
        "thesis_break_days": 0,     # consecutive DAYS composite < floor (soft break)
        # ── L1 telemetry (2026-07) ──
        # Entry-day scan snapshot (captured at the ENTERED transition; None if scan
        # data was unavailable that day). Baseline for any later re-underwrite.
        "entered_composite": None,
        "entered_qvr":       None,
        "entered_rss_short": None,
        "entered_rss_long":  None,
        # trade_mgmt — per-position price telemetry, updated daily while HOLDING.
        # effective_stop is a MONOTONE ratchet: max(previous, today's fresh stop) —
        # a stop that re-bases lower after a price drop can no longer walk down
        # with the price (this is what makes the T3 breach check honest).
        "trade_mgmt": {
            "entry_price_eff": None,   # entered_price, else one-time backfill
            "entry_price_src": None,   # "live" | "backfill"
            "hwm": None, "hwm_date": None,   # high-water-mark close since entry
            "lwm": None, "lwm_date": None,   # low-water-mark close since entry
            "pnl_pct": None,           # (close/entry − 1) × 100, latest
            "mfe_pct": None,           # max favorable excursion % (hwm vs entry)
            "mae_pct": None,           # max adverse excursion % (lwm vs entry)
            "effective_stop": None,    # monotone stop ratchet (never decreases)
        },
        # daily_log — rolling per-day scan snapshot (capped DAILY_LOG_CAP rows).
        # The 8-day global scan history is shorter than the 21-day core hold, so
        # any multi-day judgment window needs this per-position record.
        "daily_log": [],
        "no_scan_days": 0,          # consecutive days a HOLDING had no scan data
        "blind_pending_runs": 0,    # EXIT_PENDING runs without scan data (no blind close)
        "exit_price": None,         # close recorded at the EXITED transition
        "exit_date":  None,
    }


def _transition(pos: dict, new_state: str, reason: str) -> None:
    """Update position state + append history."""
    if pos["state"] != new_state:
        pos["state_history"].append({
            "date": _today(), "from": pos["state"],
            "to": new_state, "reason": reason,
        })
    pos["state"] = new_state
    pos["days_in_state"] = 0


def evaluate_thesis(pos: dict, ctx: Optional[dict]) -> tuple[bool, str]:
    """Decide whether a HOLDING position's BUY THESIS has broken.

    ctx (built by apply_state_machine from live scan data) carries:
      classification, composite, price, stop_price, short_today, days_held, has_scan.

    Returns (broken?, reason). Thesis-break conditions:
      HARD (act immediately, ignore min-hold):
        T1  classification flipped bearish  (BEARISH_CLS)
        T3  price ≤ stop_price              (Elliott/mechanical stop breach)
        T4  regime flip                     (same ticker is a SHORT pick today)
      SOFT (needs min-hold AND multi-day confirmation):
        T2  composite < COMPOSITE_EXIT_FLOOR

    If scan data is unavailable (has_scan False) we CANNOT assess the thesis →
    return (False, "") so the position is HELD (never exit blind). SKIP / WAIT /
    dropped-from-picks are deliberately NOT thesis breaks.
    """
    if not ctx or not ctx.get("has_scan"):
        return (False, "")
    cls = (ctx.get("classification") or "").upper()
    # ── T1 — bearish classification flip (HARD) ──
    for kw in BEARISH_CLS:
        if kw in cls:
            return (True, f"thesis break: 분류 약세 전환 ({ctx.get('classification')})")
    # ── T3 — stop-loss breach (HARD) ──
    # Uses the persisted MONOTONE effective_stop when available (ratcheted in
    # _update_trade_mgmt): the raw cache stop re-bases to current price every 24h,
    # so a slow bleed could walk the stop down forever without ever breaching it.
    px = ctx.get("price")
    stp = (pos.get("trade_mgmt") or {}).get("effective_stop") or ctx.get("stop_price")
    if px and stp and px <= stp:
        return (True, f"thesis break: 스톱로스 이탈 ({px:.2f} ≤ {stp:.2f})")
    # ── T4 — regime flip: held name is a SHORT pick today (HARD) ──
    if ctx.get("short_today"):
        return (True, "thesis break: 레짐 플립 (오늘 SHORT 시그널)")
    # ── T5 — give-back latch (armed winner round-tripping; computed in _update_trade_mgmt) ──
    # SHADOW: while _GIVEBACK_SHADOW is True this NEVER triggers a real exit — the
    # annotation (trade_mgmt.giveback_fired/reason) is surfaced read-only for the
    # validation period. Flip _GIVEBACK_SHADOW=False to activate the exit.
    if not _GIVEBACK_SHADOW:
        tm = pos.get("trade_mgmt") or {}
        if tm.get("giveback_fired"):
            return (True, tm.get("giveback_reason") or "thesis break: 이익반납")
    # ── T2 — composite floor (SOFT: min-hold + multi-day confirm) ──
    if float(ctx.get("composite") or 0) < COMPOSITE_EXIT_FLOOR:
        if (ctx.get("days_held", 0) >= MIN_CORE_HOLD_DAYS
                and pos.get("thesis_break_days", 0) >= SOFT_CONFIRM_DAYS):
            return (True, f"thesis 약화: composite {float(ctx.get('composite') or 0):.0f} "
                          f"< {COMPOSITE_EXIT_FLOOR} ({pos.get('thesis_break_days', 0)}일 확인)")
    return (False, "")


def _update_trade_mgmt(pos: dict, ctx: Optional[dict], is_new_day: bool) -> None:
    """Refresh the per-position telemetry block (L1) from today's ctx.

    Idempotent within a day: hwm/lwm/pnl/effective_stop are max/min updates and safe
    to run on every swarm pass; the daily_log row appends only once per calendar day
    (is_new_day), keyed off the same last_eval_date dedup as the day counters.

    Never raises — telemetry must not be able to break the state machine.
    """
    try:
        tm = pos.setdefault("trade_mgmt", {})
        today = _today()

        # entry price reference: prefer live capture, keep backfill if already set
        if tm.get("entry_price_eff") is None and pos.get("entered_price"):
            tm["entry_price_eff"] = float(pos["entered_price"])
            tm["entry_price_src"] = "live"

        px = (ctx or {}).get("price")
        try:
            px = float(px) if px else None
            if px is not None and not _math.isfinite(px):
                px = None   # NaN/Inf must never seed the hwm/lwm ratchet
        except (TypeError, ValueError):
            px = None

        # ── corporate-action re-base guard (stock split) ──
        # A >±45% overnight gap vs the last recorded close while the SCAN still looks
        # normal (non-bearish classification) is a split/re-base, not a price move —
        # yfinance adjusts history on splits so indicators stay calm, whereas a real
        # crash flips the classification bearish. Without this, the monotone ratchet
        # traps a pre-split-scale stop above the post-split price → guaranteed false
        # T3 hard exit (the exact spurious-exit class L1 must not introduce).
        try:
            log = pos.get("daily_log") or []
            prev_close = float(log[-1].get("close")) if (log and log[-1].get("close")) else None
            if px and prev_close and prev_close > 0:
                ratio = px / prev_close
                cls_now = str((ctx or {}).get("classification") or "").upper()
                is_bearish = any(kw in cls_now for kw in BEARISH_CLS)
                if (ratio < 0.55 or ratio > 1.9) and not is_bearish:
                    # ── PLAUSIBILITY GUARD (2026-07-14 fix) ──
                    # A real split re-base must land the sticky entry on the CURRENT
                    # price scale (post-split entry ≈ today's px). If the rebased entry
                    # would sit implausibly far from px, the ±45% gap is a DATA GLITCH
                    # (contaminated daily-log close / RT spike), NOT a split — skip, or it
                    # permanently corrupts entered_price (the 보유이후 수익률 = current/entry
                    # blows up, e.g. CB 340→7.93 → +4285%). hwm/stop self-recompute each
                    # scan so they recover; entered_price is sticky and never would.
                    ep0 = pos.get("entered_price")
                    rebased_ep = (float(ep0) * ratio) if ep0 else None
                    plausible = (rebased_ep is None) or (0.2 * px <= rebased_ep <= 5.0 * px)
                    if plausible:
                        for k in ("effective_stop", "hwm", "lwm", "entry_price_eff"):
                            if tm.get(k):
                                tm[k] = round(float(tm[k]) * ratio, 6)
                        if ep0:
                            pos["entered_price"] = round(rebased_ep, 6)
                    # else: spurious gap → leave sticky fields untouched
        except Exception:
            pass

        # ── SELF-HEAL corrupted sticky basis (2026-07-14) ──
        # The old un-guarded re-base could permanently corrupt the STICKY basis fields
        # (entered_price / entry_price_eff / hwm / lwm) to a value implausibly far from
        # today's price. hwm self-recovers (px>hwm bumps it up) but lwm NEVER does (px<lwm
        # is false once lwm is a tiny corrupt value), so mae_pct/보유이후 수익률 stay broken.
        # Reset every implausible sticky field to the best reference (earliest daily-log
        # close ≈ entry scale); the pnl/mfe/mae recompute below then rebuilds them correctly.
        try:
            def _implausible(v):
                try:
                    return bool(v) and px and (float(v) < 0.15 * px or float(v) > 6.0 * px)
                except (TypeError, ValueError):
                    return False
            if px and (_implausible(pos.get("entered_price")) or _implausible(tm.get("entry_price_eff"))
                       or _implausible(tm.get("hwm")) or _implausible(tm.get("lwm"))):
                ref = None
                for e in (pos.get("daily_log") or []):
                    c = e.get("close")
                    if c:
                        ref = float(c); break        # earliest recorded close ≈ entry scale
                ref = ref or px
                if _implausible(pos.get("entered_price")):
                    pos["entered_price"] = round(ref, 4)
                if _implausible(tm.get("entry_price_eff")):
                    tm["entry_price_eff"] = round(ref, 4)
                if _implausible(tm.get("hwm")):
                    tm["hwm"] = round(max(ref, px), 4)
                if _implausible(tm.get("lwm")):
                    tm["lwm"] = round(min(ref, px), 4)
        except Exception:
            pass

        # ── monotone effective-stop ratchet (never decreases) ──
        fresh_stop = (ctx or {}).get("stop_price")
        if fresh_stop:
            try:
                fresh_stop = float(fresh_stop)
                prev = tm.get("effective_stop")
                tm["effective_stop"] = max(prev, fresh_stop) if prev else fresh_stop
            except (TypeError, ValueError):
                pass

        # ── hwm / lwm / pnl / mfe / mae ──
        if px:
            if tm.get("hwm") is None or px > tm["hwm"]:
                tm["hwm"], tm["hwm_date"] = px, today
            if tm.get("lwm") is None or px < tm["lwm"]:
                tm["lwm"], tm["lwm_date"] = px, today
            entry = tm.get("entry_price_eff")
            if entry:
                tm["pnl_pct"] = round((px / entry - 1) * 100, 2)
                tm["mfe_pct"] = round((tm["hwm"] / entry - 1) * 100, 2)
                tm["mae_pct"] = round((tm["lwm"] / entry - 1) * 100, 2)

        # ── T5 give-back latch (이익반납 방어; SHADOW until _GIVEBACK_SHADOW=False) ──
        # ARM one-way once peak profit clears the (vol-scaled) bar, then count days
        # where retained profit (pnl/mfe) < floor over a rolling window. Fires an
        # ANNOTATION only here; the actual EXIT is gated in evaluate_thesis.
        try:
            mfe = tm.get("mfe_pct"); pnl = tm.get("pnl_pct")
            if tm.get("entry_price_eff") and mfe is not None and pnl is not None:
                days_held = int((ctx or {}).get("days_held") or 0)
                atr_pct = (ctx or {}).get("atr_pct")
                # HRR give-back amplifier — only when THIS held ticker is REVERSAL_RISK.
                hrr_amp = ((ctx or {}).get("hrr_flag") == "REVERSAL_RISK")
                floor = GIVEBACK_ARM_MFE_EARLY if days_held < MIN_CORE_HOLD_DAYS else GIVEBACK_ARM_MFE
                vol_bar = 0.0
                try:
                    if atr_pct and float(atr_pct) > 0:
                        vol_bar = GIVEBACK_ARM_MFE_EARLY if days_held < MIN_CORE_HOLD_DAYS else \
                                  GIVEBACK_ARM_ATR_MULT * float(atr_pct) * _math.sqrt(21)
                except (TypeError, ValueError):
                    vol_bar = 0.0
                arm_at = max(floor, vol_bar)
                if hrr_amp:
                    arm_at *= HRR_GIVEBACK_ARM_MULT      # arm earlier under reversal risk
                if not tm.get("giveback_armed") and mfe >= arm_at:
                    tm["giveback_armed"] = True          # one-way — never disarms
                    tm["giveback_arm_at"] = round(arm_at, 1)
                    tm["giveback_hrr_amped"] = bool(hrr_amp)
                # 리뷰 수정①: window 진전은 FRESH price(px)가 있는 새 날에만 — blind(스캔없음)
                # 날은 pnl/mfe가 stale이라 daily_log/thesis_break_days와 동일하게 게이트.
                if tm.get("giveback_armed") and is_new_day and px:
                    retained = (pnl / mfe) if mfe > 1e-9 else 1.0
                    tm["retained_pct"] = round(retained * 100, 1)
                    retain_floor = HRR_GIVEBACK_RETAIN_FLOOR if hrr_amp else GIVEBACK_RETAIN_FLOOR
                    hist = tm.setdefault("giveback_hist", [])
                    hist.append(1 if retained < retain_floor else 0)
                    if len(hist) > GIVEBACK_WINDOW_DAYS:
                        del hist[:len(hist) - GIVEBACK_WINDOW_DAYS]
                    tm["giveback_days"] = sum(hist)
                    if tm["giveback_days"] >= GIVEBACK_CONFIRM_DAYS and not tm.get("giveback_fired"):
                        tm["giveback_fired"] = True
                        tm["giveback_fired_date"] = today
                        tm["giveback_shadow"] = bool(_GIVEBACK_SHADOW)
                        tm["giveback_reason"] = (
                            f"thesis break: 이익반납 (고점 +{mfe:.0f}% → 현재 {pnl:+.0f}%, "
                            f"보존율 {retained*100:.0f}% < {int(retain_floor*100)}% "
                            f"× {tm['giveback_days']}/{GIVEBACK_WINDOW_DAYS}일"
                            f"{' · ⚠herding반전 증폭' if hrr_amp else ''})")
                    # 리뷰 수정②: 되돌림 회복 시 fired 해제 — 롤링 윈도우가 임계 아래로
                    # 내려오면 annotation이 현재 상태를 추종(회복한 포지션을 계속 flag 안 함).
                    elif tm.get("giveback_fired") and tm["giveback_days"] < GIVEBACK_CONFIRM_DAYS:
                        tm["giveback_fired"] = False
                        tm.pop("giveback_reason", None)
                        tm.pop("giveback_fired_date", None)
        except Exception:
            pass

        # ── daily_log append (once per calendar day) ──
        if is_new_day and ctx and ctx.get("has_scan"):
            log = pos.setdefault("daily_log", [])
            if not log or log[-1].get("date") != today:
                log.append({
                    "date": today,
                    "close": px,
                    "composite": ctx.get("composite"),
                    "classification": ctx.get("classification"),
                    "rss_short": ctx.get("rss_short"),
                    "rss_long": ctx.get("rss_long"),
                    "oer": ctx.get("oer"),
                })
                if len(log) > DAILY_LOG_CAP:
                    del log[:len(log) - DAILY_LOG_CAP]
    except Exception:
        pass   # telemetry is best-effort by design


def update_position(pos: dict, today_signal: str, ctx: Optional[dict] = None) -> Optional[str]:
    """Apply state machine rules to one position with today's signal.

    ctx (optional) carries live thesis data for HOLDING / EXIT_PENDING evaluation.
    When ctx is None or lacks scan data, HOLDING positions are never mechanically
    exited (safe default). Entry/prospecting logic still uses today_signal.

    Returns alert string if a meaningful transition happened, else None.
    """
    sig = (today_signal or "WAIT").upper()
    # ConvictionDebate signal vocabulary: BUY_NOW / WAIT / SKIP. Normalize.
    if sig not in ("BUY_NOW", "WAIT", "SKIP"):
        sig = "WAIT"

    # Date-dedup guard: day-counters advance at most ONCE per calendar day, so
    # multiple swarm runs in a day don't inflate entry/exit confirmation. (HARD
    # thesis breaks are evaluated live below and still act on any run.)
    today = _today()
    is_new_day = pos.get("last_eval_date") != today

    cnt = pos["consecutive_signal_days"]
    if is_new_day:
        for key in cnt:
            cnt[key] = (cnt[key] + 1) if key == sig else 0
        pos["days_in_state"] = pos.get("days_in_state", 0) + 1
    pos["last_signal"] = sig

    state = pos["state"]
    alert: Optional[str] = None

    # ── PROSPECTING ──
    if state == State.PROSPECTING:
        if cnt["BUY_NOW"] >= ENTRY_CONFIRMATION_DAYS:
            _transition(pos, State.ENTERED,
                        f"BUY_NOW × {cnt['BUY_NOW']} days")
            pos["entered_date"]   = today
            pos["entered_signal"] = "BUY_NOW"
            if ctx and ctx.get("price"):
                pos["entered_price"] = ctx.get("price")   # stop/drawdown reference
            # Entry-day scan snapshot — baseline for later re-underwrite (L1)
            if ctx and ctx.get("has_scan"):
                pos["entered_composite"] = ctx.get("composite")
                pos["entered_qvr"]       = ctx.get("qvr_score")
                pos["entered_rss_short"] = ctx.get("rss_short")
                pos["entered_rss_long"]  = ctx.get("rss_long")
            _update_trade_mgmt(pos, ctx, is_new_day)
            alert = Alert.NEW_BUY
        elif cnt["SKIP"] >= DROP_DAYS:
            _transition(pos, State.DROPPED,
                        f"SKIP × {cnt['SKIP']} days")
            alert = Alert.PROSPECT_DROPPED

    # ── ENTERED (transient — auto promote to HOLDING) ──
    elif state == State.ENTERED:
        _transition(pos, State.HOLDING, "auto-transition from ENTERED")
        # Telemetry must not skip the promotion day (this run consumes is_new_day —
        # without this call the daily_log permanently misses entry-day+1 and the
        # stop ratchet starts a day late).
        _update_trade_mgmt(pos, ctx, is_new_day)
        # No alert; this is mechanical

    # ── HOLDING — THESIS-BREAK exit only ──
    # SKIP / WAIT / dropping out of today's top-N picks NO LONGER sell. A 21d
    # core hold is carried until its thesis objectively breaks (see evaluate_thesis).
    elif state == State.HOLDING:
        # L1 telemetry: refresh hwm/pnl/effective-stop ratchet + daily_log BEFORE
        # the thesis check so T3 sees today's ratcheted stop.
        _update_trade_mgmt(pos, ctx, is_new_day)
        # Stale-scan watchdog: N consecutive days without scan data → alert once.
        # Reset happens on ANY scan-bearing run (not gated on is_new_day) so a blind
        # first-run-of-day can't trap a spurious SIGNAL_DEGRADED when a later run
        # the same day does have scan data; the increment stays once-per-day.
        has_scan = bool(ctx and ctx.get("has_scan"))
        if has_scan:
            pos["no_scan_days"] = 0
        elif is_new_day:
            pos["no_scan_days"] = pos.get("no_scan_days", 0) + 1
            if pos["no_scan_days"] == NO_SCAN_ALERT_DAYS:
                alert = Alert.SIGNAL_DEGRADED
        # Soft-break confirmation counter (composite below floor) — once per day.
        # The reset requires a SCAN-CONFIRMED recovery: a blind run must neither
        # count toward nor wipe the confirmation (absence of data ≠ recovery).
        below = bool(has_scan
                     and float(ctx.get("composite") or 0) < COMPOSITE_EXIT_FLOOR)
        if below and is_new_day:
            pos["thesis_break_days"] = pos.get("thesis_break_days", 0) + 1
        elif has_scan and not below:
            pos["thesis_break_days"] = 0
        broken, reason = evaluate_thesis(pos, ctx)
        if broken:
            _transition(pos, State.EXIT_PENDING, reason)
            alert = Alert.EXIT_TRIGGER
        # else: stay HOLDING. Not-re-picked / WAIT / SKIP IGNORED. ✓

    # ── EXIT_PENDING — thesis-aware close OR rescind (self-heal) ──
    # If a position landed here but its thesis is actually intact (e.g. a legacy
    # SKIP×2 mechanical exit, or the break condition cleared), restore it to
    # HOLDING instead of closing. Genuine breaks still close to EXITED.
    # NO BLIND CLOSE (L1): without scan data the thesis cannot be re-verified, so
    # the position STAYS pending (mirrors HOLDING's never-exit-blind rule) and
    # alerts after BLIND_PENDING_MAX runs instead of silently closing.
    elif state == State.EXIT_PENDING:
        _update_trade_mgmt(pos, ctx, is_new_day)
        broken, reason = evaluate_thesis(pos, ctx)
        if not (ctx and ctx.get("has_scan")):
            pos["blind_pending_runs"] = pos.get("blind_pending_runs", 0) + 1
            if pos["blind_pending_runs"] == BLIND_PENDING_MAX:
                alert = Alert.SIGNAL_DEGRADED
            # stay EXIT_PENDING — never close without re-verifying the thesis
        elif not broken:
            _transition(pos, State.HOLDING, "thesis intact — 청산 철회 (자동 복구)")
            pos["thesis_break_days"] = 0
            pos["blind_pending_runs"] = 0
            alert = Alert.EXIT_RESCINDED
        else:
            _transition(pos, State.EXITED, f"auto-close from EXIT_PENDING — {reason}")
            # exit accounting (L1): record the close used for the final breach check
            # so every trigger's outcome is retrospectively scoreable.
            if ctx and ctx.get("price"):
                pos["exit_price"] = ctx.get("price")
            pos["exit_date"] = today
            # retention hygiene: terminal positions keep only the tail of the log
            # (trade_mgmt summary + exit accounting stay intact for EQS scoring).
            if pos.get("daily_log"):
                pos["daily_log"] = pos["daily_log"][-10:]

    # ── EXITED / DROPPED (terminal — nothing to do) ──

    pos["last_eval_date"] = today
    pos["last_alert"] = alert
    return alert


def _days_held(pos: dict) -> int:
    """Calendar days since ENTERED (0 if never entered)."""
    ed = pos.get("entered_date")
    if not ed:
        return 0
    try:
        from datetime import datetime as _dtm
        return (_dtm.strptime(_today(), "%Y-%m-%d") - _dtm.strptime(ed, "%Y-%m-%d")).days
    except Exception:
        return 0


def _build_thesis_ctx(t: str, h: str, pos: dict, scan_lookup: Optional[dict],
                      stops: Optional[dict], prices: Optional[dict],
                      short_tickers: Optional[set]) -> dict:
    """Assemble the live thesis context for one (ticker, horizon) from scan data."""
    sr = (scan_lookup or {}).get(t) or {}
    stp = (stops or {}).get(f"{t}::{h}") or {}
    return {
        "has_scan":       bool(sr),
        "classification": sr.get("classification", ""),
        "composite":      float(sr.get("composite") or 0),
        "price":          (prices or {}).get(t),
        "stop_price":     stp.get("stop_price"),
        "atr_pct":        stp.get("atr_pct"),   # T5 give-back latch vol-scaled arm
        "short_today":    t in (short_tickers or set()),
        "days_held":      _days_held(pos),
        # L1 telemetry inputs (entry snapshot + daily_log) — None-safe when absent
        "qvr_score":      sr.get("qvr_score"),
        "rss_short":      sr.get("rss_short"),
        "rss_long":       sr.get("rss_long"),
        "oer":            sr.get("oer"),
        # Herding-Reversal (과열 후 반전, paper 2607.27063) — 과밀+herding약화 conjunction.
        # T5 give-back 증폭기 입력: REVERSAL_RISK 보유종목은 arm 문턱↓·retain floor↑ (SHADOW).
        "hrr_flag":       sr.get("hrr_flag"),
    }


def apply_state_machine(pm_horizons: dict, scan_lookup: Optional[dict] = None,
                        stops: Optional[dict] = None, prices: Optional[dict] = None,
                        short_tickers: Optional[set] = None) -> dict:
    """Update state for every (ticker, horizon) in PM picks. Returns updated state.

    Args:
        pm_horizons: dict like {"core": {long_stocks:[...], ...}}
        scan_lookup: {ticker: scan_row} — live classification/composite for thesis eval.
        stops:       {ticker::horizon: {stop_price,...}} — Elliott/mechanical stops (T3).
        prices:      {ticker: last_close} — current price for stop-breach check (T3).
        short_tickers: set of tickers that are SHORT picks today — regime flip (T4).

    When scan_lookup is absent, HOLDING positions are never mechanically exited
    (thesis cannot be assessed → hold). Generates alerts list for today's dashboard.
    """
    state = load_state()
    positions = state.setdefault("positions", {})
    today_alerts: list[dict] = []

    # Tracks (ticker, horizon) → first pick reference (for state-update + pick mutation)
    # AND prevents double-counting if same ticker appears in multiple buckets.
    processed_today: dict[tuple[str, str], dict] = {}

    # First pass — collect first occurrence per (ticker, horizon) and run state machine
    for h in ("core",):
        h_data = pm_horizons.get(h, {})
        for bucket in ("long_stocks", "long_etfs", "short_stocks", "short_etfs"):
            picks = h_data.get(bucket, [])
            for pick in picks:
                t = pick.get("ticker")
                if not t:
                    continue
                key_tup = (t, h)
                key = f"{t}::{h}"

                if key_tup in processed_today:
                    # Same ticker appeared earlier in this horizon — attach same state
                    pick["position_state"] = dict(processed_today[key_tup]["position_state"])
                    continue

                # Today's signal from Phase 5.5 Trading Layer
                sig = (pick.get("timing") or {}).get("entry_signal", "WAIT")

                # Get or create position
                pos = positions.get(key)
                if pos is None:
                    pos = _empty_position(t, h)
                    positions[key] = pos

                # Run state machine ONCE per (ticker, horizon), with live thesis ctx
                ctx = _build_thesis_ctx(t, h, pos, scan_lookup, stops, prices, short_tickers)
                alert = update_position(pos, sig, ctx)

                # Attach state to the pick for frontend display
                pick["position_state"] = {
                    "state": pos["state"],
                    "entered_date": pos.get("entered_date"),
                    "days_in_state": pos.get("days_in_state", 0),
                    "consecutive_signal_days": dict(pos["consecutive_signal_days"]),
                    "alert": alert,
                    "last_signal": sig,
                }
                processed_today[key_tup] = {
                    "alert": alert, "signal": sig, "bucket": bucket,
                    "position_state": pick["position_state"],
                }

                if alert:
                    today_alerts.append({
                        "ticker": t, "horizon": h, "bucket": bucket,
                        "alert": alert, "signal": sig,
                        "state": pos["state"],
                        "rationale": (pick.get("timing") or {}).get("rationale", "")[:200],
                    })

    # Build seen_today set from processed map
    seen_today: set[tuple[str, str]] = set(processed_today.keys())

    # Tickers that DROPPED OUT of today's picks (were in state but no longer picked).
    # A removed pick is treated as SKIP, BUT for HOLDING positions this no longer
    # forces an exit — we still evaluate the live thesis (classification/stop),
    # so a healthy 21d hold that simply fell out of today's top-N is KEPT.
    for key, pos in positions.items():
        t, h = key.split("::")
        if (t, h) in seen_today:
            continue
        if pos["state"] in (State.EXITED, State.DROPPED):
            continue
        prev_state = pos["state"]
        ctx = _build_thesis_ctx(t, h, pos, scan_lookup, stops, prices, short_tickers)
        alert = update_position(pos, "SKIP", ctx)
        # Report on the ALERT itself (mirrors Pass 1), not on a state transition —
        # the L1 watchdogs (SIGNAL_DEGRADED at no_scan_days==3, blind EXIT_PENDING
        # ==5) fire WITHOUT a transition, and no-scan tickers are by construction
        # only ever processed in this pass; the old transition gate swallowed them.
        if alert:
            today_alerts.append({
                "ticker": t, "horizon": h, "bucket": "(dropped)",
                "alert": alert, "signal": "SKIP (dropped from picks)",
                "state": pos["state"],
                "rationale": ("Removed from PM picks — thesis re-checked"
                              if pos["state"] != prev_state
                              else "스캔 데이터 부재 감시 (watchdog)"),
            })

    # Update alerts_history
    if today_alerts:
        state.setdefault("alerts_history", []).append({
            "date": _today(),
            "n_alerts": len(today_alerts),
            "alerts": today_alerts,
        })
        # Cap history to last 90 days
        state["alerts_history"] = state["alerts_history"][-90:]

    save_state(state)

    return {
        "today_alerts": today_alerts,
        "n_alerts": len(today_alerts),
        "n_positions": len(positions),
        "states_summary": _summarize_states(positions),
    }


def _summarize_states(positions: dict) -> dict:
    """Count of positions per state per horizon."""
    summary: dict = {}
    for key, pos in positions.items():
        _, h = key.split("::")
        s = pos["state"]
        summary.setdefault(h, {}).setdefault(s, 0)
        summary[h][s] += 1
    return summary


# ─────────────────────────────────────────────────────────────────────
# One-time backfill (L1) — legacy holdings entered before telemetry existed
# ─────────────────────────────────────────────────────────────────────

def backfill_trade_mgmt(price_cache_path: str = ".backtest_price_cache.pkl",
                        dry_run: bool = False) -> dict:
    """Backfill entry_price_eff + retroactive HWM/MFE/MAE for active positions.

    For every HOLDING / ENTERED / EXIT_PENDING position:
      - entry_price_eff: entered_price if captured live, else the last daily Close
        at/before entered_date from the backtest price cache (marked "backfill").
      - hwm/lwm + mfe/mae/pnl: replayed over the daily Close path since entered_date,
        so telemetry has no cold-start gap for positions entered before L1 shipped.
    Positions whose telemetry is already populated keep their live values (replay
    only fills fields that are None). Returns a per-ticker report dict.
    """
    import pickle as _pk

    state = load_state()
    positions = state.get("positions", {})
    try:
        pc = _pk.load(open(price_cache_path, "rb")).get("data", {})
    except Exception as e:
        return {"_error": f"price cache load failed: {e}"}

    report: dict = {}
    for key, pos in positions.items():
        if pos.get("state") not in (State.HOLDING, State.ENTERED, State.EXIT_PENDING):
            continue
        t = pos.get("ticker") or key.split("::")[0]
        ed = pos.get("entered_date")
        if not ed:
            report[key] = "skip: no entered_date"
            continue
        df = pc.get(t)
        if df is None or getattr(df, "empty", True):
            report[key] = "skip: no price data"
            continue
        try:
            closes = df["Close"].dropna()
            closes.index = closes.index.tz_localize(None) if getattr(closes.index, "tz", None) is not None else closes.index
            import pandas as _pd
            ed_ts = _pd.Timestamp(ed)
            upto = closes[closes.index <= ed_ts]
            entry_px = float(upto.iloc[-1]) if len(upto) else float(closes.iloc[0])

            tm = pos.setdefault("trade_mgmt", {})
            if tm.get("entry_price_eff") is None:
                if pos.get("entered_price"):
                    tm["entry_price_eff"] = float(pos["entered_price"])
                    tm["entry_price_src"] = "live"
                else:
                    tm["entry_price_eff"] = round(entry_px, 4)
                    tm["entry_price_src"] = "backfill"
                    pos["entered_price"] = round(entry_px, 4)   # existing consumers (보유이후 수익률)

            # retroactive HWM/LWM replay over closes AFTER entry date
            path = closes[closes.index >= ed_ts]
            if len(path):
                hi_i = path.idxmax(); lo_i = path.idxmin()
                entry = float(tm["entry_price_eff"])
                if tm.get("hwm") is None or float(path.max()) > tm["hwm"]:
                    tm["hwm"] = round(float(path.max()), 4)
                    tm["hwm_date"] = str(hi_i.date())
                if tm.get("lwm") is None or float(path.min()) < tm["lwm"]:
                    tm["lwm"] = round(float(path.min()), 4)
                    tm["lwm_date"] = str(lo_i.date())
                # idempotent re-run: never regress a live pnl_pct to the (possibly
                # days-stale) cache close; mfe/mae stay derived from the monotone
                # hwm/lwm so a replay that gap-fills a higher high keeps them in sync.
                last_px = float(path.iloc[-1])
                if tm.get("pnl_pct") is None:
                    tm["pnl_pct"] = round((last_px / entry - 1) * 100, 2)
                tm["mfe_pct"] = round((tm["hwm"] / entry - 1) * 100, 2)
                tm["mae_pct"] = round((tm["lwm"] / entry - 1) * 100, 2)
            report[key] = (f"ok src={tm.get('entry_price_src')} entry={tm.get('entry_price_eff')} "
                           f"pnl={tm.get('pnl_pct')}% mfe={tm.get('mfe_pct')}% mae={tm.get('mae_pct')}%")
        except Exception as e:
            report[key] = f"error: {str(e)[:100]}"

    if not dry_run:
        save_state(state)
    return report


# ─────────────────────────────────────────────────────────────────────
# Diagnostic / inspection helpers
# ─────────────────────────────────────────────────────────────────────

def get_ticker_state(ticker: str, horizon: str = "core") -> Optional[dict]:
    """Look up current state for one (ticker, horizon)."""
    state = load_state()
    return state.get("positions", {}).get(f"{ticker}::{horizon}")


def get_recent_alerts(days: int = 7) -> list:
    """Most recent alerts across all positions."""
    state = load_state()
    return state.get("alerts_history", [])[-days:]


if __name__ == "__main__":
    # Quick smoke test
    print(f"State file: {STATE_PATH}")
    s = load_state()
    print(f"Positions tracked: {len(s.get('positions', {}))}")
    summary = _summarize_states(s.get("positions", {}))
    for h, st in summary.items():
        print(f"  {h}: {st}")
