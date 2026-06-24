# -*- coding: utf-8 -*-
"""trade_log.py — Persistent trade history per ticker.

Stores .trade_log.json with append-only history:
  - first_seen          : date ticker first appeared in PM picks
  - entries             : list of entry events (date, price, horizon)
  - exits               : list of exit events (date, price, reason)
  - current_status      : current state (HOLDING / EXITED / PROSPECTING / ...)
  - persistence_days    : days since first_seen
  - days_held           : days since last entry (if currently HOLDING)
  - total_appearances   : count of days ticker has appeared in picks
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


LOG_PATH = Path(".trade_log.json")


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def load_log() -> dict:
    if not LOG_PATH.exists():
        return {"tickers": {}, "last_update": None, "created_at": _today()}
    try:
        return json.loads(LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"tickers": {}, "last_update": None, "created_at": _today()}


def save_log(log: dict) -> None:
    log["last_update"] = _today()
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False),
                        encoding="utf-8")


def _days_between(start_iso: str, end_iso: str) -> int:
    try:
        a = datetime.strptime(start_iso, "%Y-%m-%d").date()
        b = datetime.strptime(end_iso, "%Y-%m-%d").date()
        return (b - a).days
    except Exception:
        return 0


def update_log_from_positions(positions: dict) -> dict:
    """Update trade log based on current Phase 5.6 position state.

    Args:
        positions: dict from .position_state.json positions field
                   {key="TICKER::horizon" → state dict}

    Returns: updated log
    """
    log = load_log()
    tickers_log = log.setdefault("tickers", {})
    today = _today()

    for key, pos in positions.items():
        ticker, horizon = key.split("::", 1)
        rec = tickers_log.setdefault(ticker, {
            "ticker":          ticker,
            "first_seen":      pos.get("first_seen") or today,
            "horizons":        {},
            "entries":         [],
            "exits":           [],
            "current_status":  "PROSPECTING",
            "total_appearances": 0,
        })

        state = pos.get("state", "PROSPECTING")
        rec["horizons"][horizon] = {
            "state": state,
            "entered_date": pos.get("entered_date"),
            "days_in_state": pos.get("days_in_state", 0),
            "last_signal": pos.get("last_signal"),
            "last_alert": pos.get("last_alert"),
        }

        # Track entry event
        ent_date = pos.get("entered_date")
        if state == "ENTERED" and ent_date:
            # Check if this entry already logged
            already = any(e.get("date") == ent_date and e.get("horizon") == horizon
                          for e in rec["entries"])
            if not already:
                rec["entries"].append({
                    "date": ent_date,
                    "horizon": horizon,
                    "signal": pos.get("entered_signal", "BUY_NOW"),
                })

        # Track exit event
        if state == "EXITED":
            last_state_change = pos.get("state_history", [{}])[-1]
            exit_date = last_state_change.get("date", today)
            already = any(e.get("date") == exit_date and e.get("horizon") == horizon
                          for e in rec["exits"])
            if not already:
                rec["exits"].append({
                    "date": exit_date,
                    "horizon": horizon,
                    "reason": last_state_change.get("reason", "—"),
                })

        # Track persistence + appearance count
        rec["total_appearances"] = rec.get("total_appearances", 0) + 1
        rec["persistence_days"] = _days_between(rec["first_seen"], today)

        # Current status — priority: HOLDING > ENTERED > EXIT_PENDING > etc.
        STATUS_PRIORITY = {
            "HOLDING": 0, "ENTERED": 1, "EXIT_PENDING": 2,
            "PROSPECTING": 3, "TRIMMED": 4, "EXITED": 5, "DROPPED": 6,
        }
        # Choose highest-priority state across horizons
        states = [h["state"] for h in rec["horizons"].values()]
        if states:
            rec["current_status"] = min(states, key=lambda s: STATUS_PRIORITY.get(s, 9))

    save_log(log)
    return log


def get_persistence_days(ticker: str) -> int:
    """Get how many days a ticker has been tracked."""
    log = load_log()
    rec = log.get("tickers", {}).get(ticker, {})
    if not rec:
        return 0
    return rec.get("persistence_days", 0)


def get_active_holdings_summary() -> dict:
    """Summary of currently held positions across all tickers."""
    log = load_log()
    active_states = ("HOLDING", "ENTERED", "EXIT_PENDING")
    counts = {"HOLDING": 0, "ENTERED": 0, "EXIT_PENDING": 0}
    active_tickers = []
    for ticker, rec in log.get("tickers", {}).items():
        if rec.get("current_status") in active_states:
            counts[rec["current_status"]] = counts.get(rec["current_status"], 0) + 1
            active_tickers.append(ticker)
    return {
        "n_active": len(active_tickers),
        "by_state": counts,
        "tickers": active_tickers,
    }


if __name__ == "__main__":
    # Quick test: update from current position state
    state = json.load(open(".position_state.json"))
    log = update_log_from_positions(state.get("positions", {}))
    summary = get_active_holdings_summary()
    print(f"Trade log updated. Total tickers tracked: {len(log['tickers'])}")
    print(f"Active holdings: {summary}")
