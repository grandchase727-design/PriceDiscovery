# -*- coding: utf-8 -*-
"""trading_proxy.py — Deterministic Trading Agent signal generator + lifecycle simulator.

Mirrors what the LLM Trading Agent (Pattern T2) would output, but uses
RULE-BASED logic from historical technicals. No LLM cost.

For each PM proxy pick, generates:
  - entry_signal:  BUY_NOW | WAIT | SKIP
  - entry_trigger: standardized condition (parseable)
  - exit_triggers: list of {condition, action, type}
  - urgency:       URGENT | NORMAL | PATIENT

Then simulates each trade through a STATE MACHINE:
  PENDING → WAITING_FOR_ENTRY → ENTERED → TRIMMED → EXITED
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Technical indicators (from price data only)
# ─────────────────────────────────────────────────────────────────────

def compute_rsi(closes: pd.Series, period: int = 14) -> Optional[float]:
    """Standard 14-period RSI from close prices. Returns latest value."""
    if len(closes) < period + 1:
        return None
    deltas = closes.diff().dropna()
    gains = deltas.where(deltas > 0, 0)
    losses = (-deltas).where(deltas < 0, 0)
    avg_gain = gains.iloc[-period:].mean()
    avg_loss = losses.iloc[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def compute_extension(close: float, sma: float) -> float:
    """% extension from SMA (positive = above, negative = below)."""
    if sma == 0:
        return 0.0
    return (close - sma) / sma


def compute_position_quality(closes: pd.Series) -> dict:
    """Compute technical position metrics for signal generation."""
    if len(closes) < 50:
        return {}
    close = float(closes.iloc[-1])
    sma10 = float(closes.iloc[-10:].mean())
    sma20 = float(closes.iloc[-20:].mean())
    sma50 = float(closes.iloc[-50:].mean())
    rsi = compute_rsi(closes)
    high_20d = float(closes.iloc[-20:].max())
    low_20d  = float(closes.iloc[-20:].min())
    range_20d = high_20d - low_20d if high_20d > low_20d else 1
    position_in_range = (close - low_20d) / range_20d   # 0=at low, 1=at high

    return {
        "close": close, "sma10": sma10, "sma20": sma20, "sma50": sma50,
        "rsi": rsi,
        "ext_sma20": compute_extension(close, sma20),
        "ext_sma50": compute_extension(close, sma50),
        "high_20d": high_20d, "low_20d": low_20d,
        "position_in_range": position_in_range,
        "above_sma20": close > sma20,
        "above_sma50": close > sma50,
    }


# ─────────────────────────────────────────────────────────────────────
# Entry signal generator (rule-based mirror of LLM Trading Agent)
# ─────────────────────────────────────────────────────────────────────

# Exit triggers per horizon (% from entry, etc.)
EXIT_TRIGGERS_BY_HORIZON = {
    "tactical": {   # 5d hold
        "TAKE_PROFIT_pct": 5.0,
        "STOP_LOSS_pct":  -3.0,
        "STOP_BREAK_SMA": "sma10",
        "OVEREXT_rsi":    75,
    },
    "core": {       # 21d hold
        "TAKE_PROFIT_pct": 10.0,
        "STOP_LOSS_pct":  -6.0,
        "STOP_BREAK_SMA": "sma20",
        "OVEREXT_rsi":    78,
    },
    "strategic": {  # 63d hold
        "TAKE_PROFIT_pct": 20.0,
        "STOP_LOSS_pct":  -12.0,
        "STOP_BREAK_SMA": "sma50",
        "OVEREXT_rsi":    80,
    },
}


def trading_proxy_signal(price_history: pd.DataFrame, side: str,
                          horizon: str = "core",
                          classification: str = "LONG_OK") -> dict:
    """Generate timing signal for one ticker from its price history.

    Args:
        price_history: DataFrame indexed by date, with 'Close' column.
        side: "long" | "short"
        horizon: "tactical" | "core" | "strategic"
        classification: from PM proxy ("LONG_OK"/"SHORT_OK"/"NEUTRAL")
    """
    pq = compute_position_quality(price_history["Close"])
    if not pq:
        return _stub_signal("WAIT", "insufficient history", side, horizon)

    rsi = pq["rsi"] or 50
    ext20 = pq["ext_sma20"]
    above50 = pq["above_sma50"]
    pos_range = pq["position_in_range"]

    # ── ENTRY SIGNAL LOGIC ──
    if side == "long":
        if classification == "LONG_OK":
            # BUY_NOW: established trend, mid-RSI, near SMA20 (not extended)
            if above50 and 35 < rsi < 65 and abs(ext20) < 0.04:
                signal, trigger = "BUY_NOW", "Already triggered — trend + RSI neutral + near SMA20"
                urgency = "URGENT" if pos_range > 0.85 else "NORMAL"
            # WAIT: extended (above SMA20 > 5%) or overbought (RSI > 70)
            elif rsi > 70 or ext20 > 0.05:
                signal, trigger = "WAIT", "Pullback to SMA20 with RSI retreating below 65"
                urgency = "PATIENT"
            # WAIT: below SMA20 in established trend (wait for reversal)
            elif above50 and not pq["above_sma20"]:
                signal, trigger = "WAIT", "Reversal candle off SMA50 support"
                urgency = "PATIENT"
            else:
                signal, trigger = "WAIT", "Trend confirmation needed"
                urgency = "PATIENT"
        else:  # NEUTRAL or invalid
            signal, trigger = "SKIP", "No clear trend setup"
            urgency = "NORMAL"
    else:  # short
        if classification == "SHORT_OK":
            if not above50 and 35 < rsi < 65:
                signal, trigger = "BUY_NOW", "Trend down + RSI neutral — enter short"
                urgency = "URGENT" if pos_range < 0.15 else "NORMAL"
            elif rsi < 30 or ext20 < -0.05:
                signal, trigger = "WAIT", "Wait for bounce to SMA20 to short"
                urgency = "PATIENT"
            else:
                signal, trigger = "WAIT", "Short setup not yet confirmed"
                urgency = "PATIENT"
        else:
            signal, trigger = "SKIP", "No clear breakdown setup"
            urgency = "NORMAL"

    # ── EXIT TRIGGERS (standardized) ──
    h_rules = EXIT_TRIGGERS_BY_HORIZON[horizon]
    if side == "long":
        exits = [
            {"condition": f"Price hits +{h_rules['TAKE_PROFIT_pct']}% above entry",
             "action": "TRIM 50%", "type": "TAKE_PROFIT",
             "_pct_threshold": h_rules['TAKE_PROFIT_pct']},
            {"condition": f"Close below {h_rules['STOP_BREAK_SMA']}",
             "action": "EXIT 100%", "type": "STOP_LOSS",
             "_sma_break": h_rules['STOP_BREAK_SMA']},
            {"condition": f"Price hits {h_rules['STOP_LOSS_pct']}% below entry",
             "action": "EXIT 100%", "type": "STOP_LOSS",
             "_pct_threshold": h_rules['STOP_LOSS_pct']},
            {"condition": f"RSI spikes above {h_rules['OVEREXT_rsi']}",
             "action": "TRIM 33%", "type": "OVEREXT",
             "_rsi_threshold": h_rules['OVEREXT_rsi']},
        ]
    else:  # short
        exits = [
            {"condition": f"Price drops -{h_rules['TAKE_PROFIT_pct']}% below entry",
             "action": "COVER 50%", "type": "TAKE_PROFIT",
             "_pct_threshold": -h_rules['TAKE_PROFIT_pct']},
            {"condition": f"Close above {h_rules['STOP_BREAK_SMA']}",
             "action": "COVER 100%", "type": "STOP_LOSS",
             "_sma_recl": h_rules['STOP_BREAK_SMA']},
            {"condition": f"Price rises +{abs(h_rules['STOP_LOSS_pct'])}% above entry",
             "action": "COVER 100%", "type": "STOP_LOSS",
             "_pct_threshold": abs(h_rules['STOP_LOSS_pct'])},
        ]

    return {
        "entry_signal": signal,
        "entry_trigger": trigger,
        "exit_triggers": exits,
        "urgency": urgency,
        "rationale": f"{horizon} {side}: cls={classification} rsi={rsi:.0f} "
                     f"ext20={ext20*100:+.1f}% above_sma50={above50}",
        "_pq": pq,
    }


def _stub_signal(signal, reason, side, horizon):
    return {"entry_signal": signal, "entry_trigger": reason,
            "exit_triggers": [], "urgency": "NORMAL",
            "rationale": reason, "_pq": {}}


# ─────────────────────────────────────────────────────────────────────
# Trade Lifecycle Simulator (state machine)
# ─────────────────────────────────────────────────────────────────────

def _check_entry_trigger(trigger_text: str, day_data: dict,
                          first_day_data: dict) -> bool:
    """Heuristic: did today's price data satisfy the entry trigger?"""
    t = (trigger_text or "").lower()
    close = day_data.get("close", 0)
    sma20 = day_data.get("sma20", 0)
    rsi = day_data.get("rsi", 50)
    high_20d = day_data.get("high_20d", 0)
    above_sma20 = day_data.get("above_sma20", False)
    if "already triggered" in t:
        return True
    if "pullback to sma20" in t and "rsi" in t and "65" in t:
        return abs(close - sma20) / sma20 < 0.015 and rsi < 65
    if "reversal candle off sma50" in t:
        return abs(close - day_data.get("sma50", 0)) / day_data.get("sma50", 1) < 0.02
    if "break of 20" in t or "high" in t:
        return close > high_20d * 0.998
    if "bounce to sma20" in t:
        return abs(close - sma20) / sma20 < 0.02
    return False


def _check_exit_trigger(trig: dict, day_data: dict, entry_price: float,
                        side: str) -> Optional[str]:
    """Return exit_type string if exit trigger fires today, else None."""
    close = day_data.get("close", 0)
    pct_from_entry = ((close - entry_price) / entry_price) * 100
    if side == "short":
        pct_from_entry = -pct_from_entry   # short PnL is reverse

    typ = trig.get("type")
    if typ == "TAKE_PROFIT":
        if pct_from_entry >= trig.get("_pct_threshold", 999):
            return "TAKE_PROFIT"
    if typ == "STOP_LOSS":
        if "_pct_threshold" in trig and pct_from_entry <= -trig["_pct_threshold"]:
            return "STOP_LOSS"
        if "_sma_break" in trig:
            sma_key = trig["_sma_break"]   # "sma10"/"sma20"/"sma50"
            sma_val = day_data.get(sma_key, 0)
            if side == "long" and close < sma_val:
                return "STOP_LOSS"
            if side == "short" and "_sma_recl" in trig and close > sma_val:
                return "STOP_LOSS"
        if "_sma_recl" in trig:
            sma_val = day_data.get(trig["_sma_recl"], 0)
            if side == "short" and close > sma_val:
                return "STOP_LOSS"
    if typ == "OVEREXT":
        rsi = day_data.get("rsi", 50)
        if rsi and rsi >= trig.get("_rsi_threshold", 999):
            return "OVEREXT"
    return None


def simulate_trade_lifecycle(price_history: pd.DataFrame,
                              entry_date: pd.Timestamp,
                              signal: dict, side: str,
                              horizon_days: int = 21) -> dict:
    """Simulate from entry_date forward through state machine.

    Returns: {
      state: "EXITED" | "TIMED_OUT" | "NEVER_ENTERED" | "SKIPPED",
      entry_date, entry_price, exit_date, exit_price, exit_type,
      days_held, realized_return, days_to_trigger, trim_taken
    }
    """
    out = {
        "state": "PENDING", "entry_date": str(entry_date.date()),
        "entry_price": None, "exit_date": None, "exit_price": None,
        "exit_type": None, "days_held": 0, "realized_return": None,
        "days_to_trigger": None, "trim_taken": False,
        "side": side,
    }

    entry_signal = signal.get("entry_signal", "WAIT")
    if entry_signal == "SKIP":
        out["state"] = "SKIPPED"
        return out

    # Walk forward
    fwd = price_history[price_history.index > entry_date]
    if len(fwd) == 0:
        out["state"] = "NEVER_ENTERED"
        return out

    state = "PENDING" if entry_signal == "WAIT" else "ENTERED"
    if state == "ENTERED":
        out["entry_price"] = float(fwd["Close"].iloc[0])
        out["entry_date"] = str(fwd.index[0].date())
        out["state"] = "ENTERED"

    realized_legs = []   # list of (exit_price, weight, exit_type)
    days_iter = 0
    first_day_data = {}
    for i in range(min(len(fwd), horizon_days)):
        days_iter += 1
        d = fwd.index[i]
        # Build per-day technical snapshot
        hist_to_d = price_history[price_history.index <= d]
        if len(hist_to_d) < 50:
            continue
        pq = compute_position_quality(hist_to_d["Close"])
        if not first_day_data:
            first_day_data = pq

        if state == "PENDING":
            # check entry trigger
            if _check_entry_trigger(signal.get("entry_trigger", ""), pq, first_day_data):
                state = "ENTERED"
                out["entry_price"] = pq["close"]
                out["entry_date"] = str(d.date())
                out["days_to_trigger"] = days_iter
            continue

        if state in ("ENTERED", "TRIMMED"):
            # check exit triggers in priority order
            for trig in signal.get("exit_triggers", []):
                exit_type = _check_exit_trigger(trig, pq, out["entry_price"], side)
                if exit_type:
                    if exit_type == "TAKE_PROFIT" and state == "ENTERED" and not out["trim_taken"]:
                        # First TP: trim 50%, continue holding rest
                        realized_legs.append((pq["close"], 0.5, "TAKE_PROFIT"))
                        out["trim_taken"] = True
                        state = "TRIMMED"
                        break  # check next day for remaining
                    else:
                        # Final exit (STOP_LOSS / OVEREXT / second TP)
                        remaining_weight = 0.5 if out["trim_taken"] else 1.0
                        realized_legs.append((pq["close"], remaining_weight, exit_type))
                        state = "EXITED"
                        out["exit_date"] = str(d.date())
                        out["exit_price"] = pq["close"]
                        out["exit_type"] = exit_type
                        out["days_held"] = days_iter
                        break

        if state == "EXITED":
            break

    # If still in trade at horizon expiry — close at last price
    if state in ("ENTERED", "TRIMMED"):
        last_idx = min(horizon_days - 1, len(fwd) - 1)
        if last_idx >= 0:
            close = float(fwd["Close"].iloc[last_idx])
            remaining_weight = 0.5 if out["trim_taken"] else 1.0
            realized_legs.append((close, remaining_weight, "TIMED_OUT"))
            out["exit_date"] = str(fwd.index[last_idx].date())
            out["exit_price"] = close
            out["exit_type"] = "TIMED_OUT"
            out["days_held"] = last_idx + 1
        out["state"] = "EXITED_AT_HORIZON"
    elif state == "PENDING":
        out["state"] = "NEVER_ENTERED"

    # Compute weighted realized return
    if realized_legs and out["entry_price"]:
        weighted_return = 0.0
        for exit_p, weight, etype in realized_legs:
            r = (exit_p / out["entry_price"] - 1)
            if side == "short":
                r = -r
            weighted_return += r * weight
        out["realized_return"] = weighted_return
        out["state"] = out.get("state") or "EXITED"
        if not out["exit_type"]:
            # Mixed legs — use first non-TP type
            out["exit_type"] = next((etype for _,_,etype in realized_legs if etype != "TAKE_PROFIT"),
                                    realized_legs[0][2])

    return out
