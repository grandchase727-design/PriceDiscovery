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
EXIT_CONFIRMATION_DAYS  = 2  # SKIP for N days to exit (while HOLDING)
WAIT_TOLERANCE_DAYS     = 5  # WAIT for N days while HOLDING = degraded
DROP_DAYS               = 3  # SKIP for N days while PROSPECTING = drop

# ─── Alert types ────────────────────────────────────────────────────
class Alert:
    NEW_BUY          = "🆕 NEW BUY"
    EXIT_TRIGGER     = "⚠ EXIT TRIGGER"
    SIGNAL_DEGRADED  = "📊 SIGNAL DEGRADED"
    PROSPECT_DROPPED = "✗ PROSPECT DROPPED"


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


def update_position(pos: dict, today_signal: str) -> Optional[str]:
    """Apply state machine rules to one position with today's signal.

    Returns alert string if a meaningful transition happened, else None.
    """
    sig = (today_signal or "WAIT").upper()
    # ConvictionDebate signal vocabulary: BUY_NOW / WAIT / SKIP. Normalize.
    if sig not in ("BUY_NOW", "WAIT", "SKIP"):
        sig = "WAIT"

    # Update consecutive-day counters
    cnt = pos["consecutive_signal_days"]
    for key in cnt:
        cnt[key] = (cnt[key] + 1) if key == sig else 0
    pos["last_signal"] = sig
    pos["days_in_state"] = pos.get("days_in_state", 0) + 1

    state = pos["state"]
    alert: Optional[str] = None

    # ── PROSPECTING ──
    if state == State.PROSPECTING:
        if cnt["BUY_NOW"] >= ENTRY_CONFIRMATION_DAYS:
            _transition(pos, State.ENTERED,
                        f"BUY_NOW × {cnt['BUY_NOW']} consecutive days")
            pos["entered_date"]   = _today()
            pos["entered_signal"] = "BUY_NOW"
            alert = Alert.NEW_BUY
        elif cnt["SKIP"] >= DROP_DAYS:
            _transition(pos, State.DROPPED,
                        f"SKIP × {cnt['SKIP']} consecutive days")
            alert = Alert.PROSPECT_DROPPED

    # ── ENTERED (transient — auto promote to HOLDING) ──
    elif state == State.ENTERED:
        _transition(pos, State.HOLDING, "auto-transition from ENTERED")
        # No alert; this is mechanical

    # ── HOLDING (sticky — eats flip-flops) ──
    elif state == State.HOLDING:
        # Strong sell: SKIP confirmation
        if cnt["SKIP"] >= EXIT_CONFIRMATION_DAYS:
            _transition(pos, State.EXIT_PENDING,
                        f"SKIP × {cnt['SKIP']} days while HOLDING")
            alert = Alert.EXIT_TRIGGER
        # Weak sell: prolonged WAIT (5+ days of indecision)
        elif cnt["WAIT"] >= WAIT_TOLERANCE_DAYS:
            _transition(pos, State.EXIT_PENDING,
                        f"WAIT × {cnt['WAIT']} days — signal degraded")
            alert = Alert.SIGNAL_DEGRADED
        # else: stay HOLDING. BUY→WAIT→BUY flips IGNORED. ✓

    # ── EXIT_PENDING (transient — auto close) ──
    elif state == State.EXIT_PENDING:
        _transition(pos, State.EXITED, "auto-transition from EXIT_PENDING")

    # ── EXITED / DROPPED (terminal — nothing to do) ──

    pos["last_alert"] = alert
    return alert


def apply_state_machine(pm_horizons: dict) -> dict:
    """Update state for every (ticker, horizon) in PM picks. Returns updated state.

    Args:
        pm_horizons: dict like {"tactical": {long_stocks:[...], ...}, ...}

    Generates alerts list for today's dashboard.
    """
    state = load_state()
    positions = state.setdefault("positions", {})
    today_alerts: list[dict] = []

    # Tracks (ticker, horizon) → first pick reference (for state-update + pick mutation)
    # AND prevents double-counting if same ticker appears in multiple buckets.
    processed_today: dict[tuple[str, str], dict] = {}

    # First pass — collect first occurrence per (ticker, horizon) and run state machine
    for h in ("tactical", "core", "strategic"):
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

                # Run state machine ONCE per (ticker, horizon)
                alert = update_position(pos, sig)

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

    # Tickers that DROPPED OUT of today's picks (were in state but no longer picked):
    # Move them toward EXIT (a removed pick is a signal in itself).
    for key, pos in positions.items():
        t, h = key.split("::")
        if (t, h) in seen_today:
            continue
        if pos["state"] in (State.EXITED, State.DROPPED):
            continue
        # Was tracking, now disappeared — treat as implicit SKIP
        prev_state = pos["state"]
        update_position(pos, "SKIP")
        if pos["state"] != prev_state and pos.get("last_alert"):
            today_alerts.append({
                "ticker": t, "horizon": h, "bucket": "(dropped)",
                "alert": pos["last_alert"], "signal": "SKIP (dropped from picks)",
                "state": pos["state"], "rationale": "Removed from PM picks",
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
