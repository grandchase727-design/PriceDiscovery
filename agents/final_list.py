# -*- coding: utf-8 -*-
"""final_list.py — Buy/Sell Final List synthesizer.

Combines 4 signal sources through a confidence-tiering pipeline:
  1. PM Swarm picks (Phase 5)
  2. Trading Signal (Phase 5.5)
  3. Position State (Phase 5.6 — hysteresis filter)
  4. Backtest cross-checks (proxy 6/12 + historical alpha leaders)

Produces two final lists:
  - 🟢 BUY  : LONG (stocks + ETFs) candidates with positive technical alignment
  - 🔴 SELL : SHORT (stocks + ETFs) candidates + LONG positions to exit
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _safe_load(path: str, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


# Cached price data (loaded once per process for performance)
_PRICE_CACHE: dict = {}
_PRICE_CACHE_LOADED = False

def _load_price_cache() -> dict:
    """Load backtest price cache once; reused across calls."""
    global _PRICE_CACHE, _PRICE_CACHE_LOADED
    if _PRICE_CACHE_LOADED:
        return _PRICE_CACHE
    try:
        import pickle
        cache_path = Path(".backtest_price_cache.pkl")
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
                _PRICE_CACHE = cache.get("data", {})
    except Exception:
        _PRICE_CACHE = {}
    _PRICE_CACHE_LOADED = True
    return _PRICE_CACHE


# Trailing return horizons in trading days
TRAILING_HORIZONS = {
    "ret_5d":  5,    # 1 week
    "ret_1mo": 21,   # ~1 month
    "ret_3mo": 63,   # ~3 months
    "ret_6mo": 126,  # ~6 months
    "ret_1y":  252,  # ~1 year
}


def _infer_bucket(scan_row: dict, side: str = "long") -> str:
    """Infer bucket name (long_stocks / long_etfs / short_*) from scan record category.

    Used when a position's PM-pick bucket is unavailable (e.g. HOLDING position
    not in today's PM picks, or EXIT_PENDING entry built from position_state only).

    Convention from price_discovery.py universe loading:
      - ETF tickers: category starts with "EQ_" (EQ_Broad, EQ_Healthcare, EQ_Tech, etc.)
      - Stock tickers: category starts with "STK_" (STK_Financials, STK_Tech, etc.)

    Args:
        scan_row: row from .scan_cache.pkl results (must have 'category' key)
        side: "long" or "short"

    Returns: one of "long_stocks", "long_etfs", "short_stocks", "short_etfs".
             Falls back to "long_stocks" if classification ambiguous.
    """
    cat = (scan_row.get("category") or "").upper()
    is_etf = cat.startswith("EQ_") or cat.startswith("ETF_")
    if not cat:
        # Last-resort heuristic — ticker length / format
        t = (scan_row.get("ticker") or "").upper()
        if t.endswith(".KS") or t.endswith(".KQ"):
            # Korean: 6-digit numeric → ETF by convention here (universe-loaded)
            t_base = t.split(".")[0]
            is_etf = t_base.isdigit() and len(t_base) == 6
    prefix = "long" if side == "long" else "short"
    return f"{prefix}_{'etfs' if is_etf else 'stocks'}"


def _compute_trailing_returns(ticker: str, price_data: dict) -> dict:
    """Compute trailing total returns for a ticker over standard horizons.

    Returns dict like {"ret_5d": 0.034, "ret_1mo": 0.12, ...}. Any horizon with
    insufficient history → None.
    """
    out = {k: None for k in TRAILING_HORIZONS}
    df = price_data.get(ticker)
    if df is None or df.empty:
        return out
    try:
        closes = df["Close"]
        if len(closes) == 0:
            return out
        current = float(closes.iloc[-1])
        if current <= 0:
            return out
        for key, days in TRAILING_HORIZONS.items():
            if len(closes) < days + 1:
                continue
            past = float(closes.iloc[-days - 1])
            if past > 0:
                out[key] = current / past - 1
    except Exception:
        pass
    return out


def _load_sources() -> dict:
    """Pull all signal sources for 3-Agent Voting (PM + Trading + Risk).

    Sources:
      - PM Swarm picks      → PM vote (composite + classification 기반)
      - Trading entry_signal → Trading vote (BUY_NOW/WAIT/SKIP)
      - Scan cache         → Risk vote (deterministic risk score)
      - Backtest data      → cross-check info (legacy compatibility)
    """
    sw = _safe_load(".market_leaders_swarm_cache.json", {})
    bt = _safe_load("backtest/results.json", {})
    rk = _safe_load("backtest/ticker_rankings.json", {})
    ps = _safe_load(".position_state.json", {"positions": {}})

    # Proxy RECENT cohorts — union of last 4 weeks (was: only latest 1)
    tlc = bt.get("trading_lifecycles_compact", {}).get("core", {})
    proxy_long_stocks: set[str] = set()
    proxy_long_etfs:   set[str] = set()
    latest_date = bt.get("end_date") or ""
    if tlc:
        all_cohorts_stocks = sorted({r["d"] for r in tlc.get("long_stocks", [])})
        all_cohorts_etfs   = sorted({r["d"] for r in tlc.get("long_etfs",   [])})
        # Take last 4 cohorts (1 month of proxy activity)
        recent_stocks_cohorts = set(all_cohorts_stocks[-4:]) if all_cohorts_stocks else set()
        recent_etfs_cohorts   = set(all_cohorts_etfs[-4:])   if all_cohorts_etfs   else set()
        proxy_long_stocks = {r["t"] for r in tlc.get("long_stocks", []) if r["d"] in recent_stocks_cohorts}
        proxy_long_etfs   = {r["t"] for r in tlc.get("long_etfs",   []) if r["d"] in recent_etfs_cohorts}

    # Top alpha leaders — EXTENDED to top 40 (was: 20)
    top_stocks = {x["ticker"] for x in rk.get("long_stocks", {}).get("top", [])[:40]}
    top_etfs   = {x["ticker"] for x in rk.get("long_etfs",   {}).get("top", [])[:40]}
    worst_stocks = {x["ticker"] for x in rk.get("long_stocks", {}).get("worst", [])[:30]}
    worst_etfs   = {x["ticker"] for x in rk.get("long_etfs",   {}).get("worst", [])[:30]}

    # Load scan_cache for risk evaluation (Phase 5.7 Risk Manager)
    scan_lookup: dict[str, dict] = {}
    try:
        import pickle
        sc_path = Path(".scan_cache.pkl")
        if sc_path.exists():
            sc = pickle.load(open(sc_path, "rb"))
            for r in sc.get("results", []) or []:
                t = r.get("ticker")
                if t:
                    scan_lookup[t] = r
    except Exception:
        pass

    # Count sector concentration across PM picks (for Risk concentration vote)
    sector_counts: dict[str, int] = {}
    for h in ("tactical", "core", "strategic"):
        h_data = (sw.get("phase5_pm") or {}).get("horizons", {}).get(h, {}) or {}
        for bucket in ("long_stocks", "long_etfs"):
            for p in h_data.get(bucket, []) or []:
                sector = (p.get("sector") or "").strip()
                if sector:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1

    return {
        "pm_swarm": sw,
        "proxy_long_stocks": proxy_long_stocks,
        "proxy_long_etfs":   proxy_long_etfs,
        "proxy_cohort_date": latest_date,
        "top_stocks": top_stocks,
        "top_etfs":   top_etfs,
        "worst_stocks": worst_stocks,
        "worst_etfs":   worst_etfs,
        "positions": ps.get("positions", {}),
        "price_data": _load_price_cache(),
        "scan_lookup": scan_lookup,
        "sector_counts": sector_counts,
    }


# Composite threshold for "high composite" cross-check (lowered: more inclusive)
HIGH_COMPOSITE_THRESHOLD = 70
# Two-track tier logic — primary (backtest) vs bonus (current state):
#   ★★★ = both primary + ANY bonus
#   ★★  = ANY primary OR (no primary but 2 bonuses)
#   ★   = otherwise (PM only)


# ─────────────────────────────────────────────────────────────────────
# Per-pick scoring + filtering
# ─────────────────────────────────────────────────────────────────────

def _eval_buy_pick_with_debate(p: dict, horizon: str, bucket: str, sources: dict) -> Optional[dict]:
    """Score one PM Buy candidate using DEBATE SYNTHESIS (Option A).

    Uses debate_synthesis output if available (from Phase 5.6a Synthesizer).
    Falls back to deterministic voting if synthesis missing.
    """
    debate = p.get("debate_synthesis")
    timing = p.get("timing") or {}
    t = p.get("ticker", "")
    composite = float(p.get("composite") or 0)
    key = f"{t}::{horizon}"
    pos = sources["positions"].get(key, {})
    state = pos.get("state", "PROSPECTING")
    sig_days = (pos.get("consecutive_signal_days") or {}).get("BUY_NOW", 0)

    if not debate:
        # No debate synthesis → use legacy voting
        return _eval_buy_pick(p, horizon, bucket, sources)

    # Use synthesizer's output
    tier = debate.get("tier", "SOLO")
    stars = debate.get("stars", 1)
    final_decision = debate.get("final_decision", "WATCH")
    debate_transcript = debate.get("debate_transcript", "")
    key_factor = debate.get("key_factor", "")

    # EXCLUDED → filter out
    if final_decision == "EXCLUDE" or stars == 0:
        return None

    # Trailing returns
    trailing = _compute_trailing_returns(t, sources.get("price_data") or {})

    # Action urgency
    if state == "ENTERED" and pos.get("last_alert") and "NEW BUY" in (pos.get("last_alert") or ""):
        action = "EXECUTE_TODAY"
    elif state == "ENTERED":
        action = "ALREADY_HELD"
    elif state == "PROSPECTING" and sig_days >= 1:
        action = "WATCH_TOMORROW"
    elif state == "HOLDING":
        action = "ALREADY_HELD"
    else:
        action = "OBSERVE"

    # Legacy votes (for backward-compat display)
    from agents.risk_manager import compute_risk_score, risk_vote, risk_rationale, pm_vote, trading_vote, tally_votes
    scan_row = sources["scan_lookup"].get(t, {})
    pm_pick_enriched = {**p, "classification": scan_row.get("classification") or p.get("classification") or ""}
    pm_v = pm_vote(pm_pick_enriched)
    trading_v = trading_vote(timing)
    risk_breakdown = compute_risk_score(scan_row, sources["sector_counts"])
    risk_v_deterministic = risk_vote(risk_breakdown)
    # Use LLM risk verdict if available
    risk_llm = p.get("risk_verdict") or {}
    risk_v = risk_llm.get("vote", risk_v_deterministic)

    # ── EXPLICIT FAILURE DETECTION (Option C Tier 1) ──
    # Honest-failure mode: agents now mark _failed=True when LLM didn't produce valid output.
    # Previously default SOLO/WATCH/WAIT/CAUTION cascaded silently.
    composite = float(p.get("composite") or 0)
    cls_raw = scan_row.get("classification") or p.get("classification") or ""
    cls_str = cls_raw if isinstance(cls_raw, str) else ""
    is_strong = any(s in cls_str for s in ("CONTINUATION","FORMATION","RECOVERY","LAGGING_CATCHUP"))

    debate_failed  = bool((p.get("debate_synthesis") or {}).get("_failed"))
    trading_failed = bool((timing or {}).get("_failed"))
    risk_failed    = bool(risk_llm.get("_failed"))
    any_agent_failed = debate_failed or trading_failed or risk_failed

    # Detect heuristic degradation (legacy cache may not have _failed flag yet)
    legacy_degraded = (
        tier == "SOLO"
        and final_decision == "WATCH"
        and (timing or {}).get("entry_signal", "") == "WAIT"
        and risk_llm.get("vote") == "CAUTION"
    )

    if any_agent_failed or (legacy_degraded and composite >= 65 and is_strong):
        # Composite-based PM-conviction override
        if composite >= 75 and is_strong:
            pm_v = "APPROVE"; trading_v = "APPROVE"
            risk_v = "APPROVE" if risk_v_deterministic == "APPROVE" else risk_v
            p["_votes_overridden"] = "agent_failure" if any_agent_failed else "swarm_degraded"
        elif composite >= 65 and is_strong:
            pm_v = "APPROVE"; trading_v = "CAUTION"
            risk_v = "APPROVE" if risk_v_deterministic == "APPROVE" else risk_v
            p["_votes_overridden"] = "agent_failure" if any_agent_failed else "swarm_degraded"
        elif composite >= 55:
            # Mid composite — fall back to deterministic verdicts (don't fake APPROVE)
            pm_v = "CAUTION"; trading_v = "CAUTION"; risk_v = risk_v_deterministic
            p["_votes_fallback"] = "deterministic_only"
        else:
            # Low composite — accept REJECT signal
            pm_v = "REJECT" if composite < 40 else "CAUTION"
            trading_v = "CAUTION"; risk_v = risk_v_deterministic
            p["_votes_fallback"] = "deterministic_only"

        # Record which agents failed (for downstream visibility)
        p["_failed_agents"] = {
            "trading": trading_failed,
            "risk":    risk_failed,
            "debate":  debate_failed,
        }

    proxy_set = sources["proxy_long_stocks"] if "stocks" in bucket else sources["proxy_long_etfs"]
    top_set   = sources["top_stocks"]        if "stocks" in bucket else sources["top_etfs"]
    in_proxy_recent = t in proxy_set
    in_top_alpha    = t in top_set
    state_confirmed = state == "ENTERED"
    high_composite  = composite >= HIGH_COMPOSITE_THRESHOLD

    trailing = _compute_trailing_returns(t, sources.get("price_data") or {})

    return {
        "ticker": t,
        "name": (p.get("name") or "")[:40],
        "sector": p.get("sector") or "",
        "composite": composite,
        "horizon": horizon,
        "bucket": bucket,
        "stars": stars,
        # ── DEBATE SYNTHESIS (Option A — Phase 5.6a output) ──
        "tier": tier,
        "consensus": tier,   # alias for backward-compat
        "debate_transcript": debate_transcript,
        "key_factor": key_factor,
        "final_decision": final_decision,
        # 3-Agent Voting (for legacy display)
        "votes": {
            "pm":      pm_v,
            "trading": trading_v,
            "risk":    risk_v,
        },
        "n_approve":   sum(1 for v in [pm_v, trading_v, risk_v] if v == "APPROVE"),
        "n_reject":    sum(1 for v in [pm_v, trading_v, risk_v] if v == "REJECT"),
        "n_caution":   sum(1 for v in [pm_v, trading_v, risk_v] if v == "CAUTION"),
        # Risk details (LLM if available, else deterministic)
        "risk_score":  risk_breakdown["total"],
        "risk_breakdown": risk_breakdown,
        "risk_reason": (risk_llm.get("rationale") or risk_rationale(risk_breakdown))[:300],
        "risk_key":    risk_llm.get("key_risk", "—"),
        # Legacy cross-check
        "n_validations": sum([in_proxy_recent, in_top_alpha, state_confirmed, high_composite]),
        "in_proxy_recent":   in_proxy_recent,
        "in_proxy_latest":   in_proxy_recent,
        "in_top_alpha":      in_top_alpha,
        "state_confirmed":   state_confirmed,
        "high_composite":    high_composite,
        # Position state
        "state": state,
        "signal_days": sig_days,
        "urgency": timing.get("urgency", "NORMAL"),
        "action": action,
        # Reasoning
        "entry_trigger": (timing.get("entry_trigger") or "")[:200],
        "rationale": (p.get("rationale") or "")[:300],
        "trading_rationale": (timing.get("rationale") or "")[:300],
        "exit_triggers": (timing.get("exit_triggers") or [])[:2],
        # Trailing returns
        **trailing,
    }


def _eval_buy_pick(p: dict, horizon: str, bucket: str, sources: dict) -> Optional[dict]:
    """Score one PM Buy candidate using 3-Agent Voting (PM + Trading + Risk).

    Three agents vote APPROVE / CAUTION / REJECT:
      1. PM Vote      : conviction (composite + classification)
      2. Trading Vote : entry_signal (BUY_NOW → APPROVE)
      3. Risk Vote    : risk score (overheating + vol + liquidity + concentration + drawdown)

    Tier (Stars):
      ★★★ UNANIMOUS         : 3 APPROVE
      ★★★ MAJORITY_CLEAN    : 2 APPROVE + 1 CAUTION
      ★★  MAJORITY_DISSENT  : 2 APPROVE + 1 REJECT
      ★★  SOLO_CLEAN        : 1 APPROVE + 2 CAUTION
      ★   SOLO_DISSENT      : 1 APPROVE + 1 REJECT + 1 CAUTION
      ★   ALL_CAUTION       : 3 CAUTION (no clear signal)
      (excluded) REJECTED   : ≥2 REJECT
    """
    from agents.risk_manager import (
        compute_risk_score, risk_vote, risk_rationale,
        pm_vote, trading_vote, tally_votes,
    )

    timing = p.get("timing") or {}
    t = p.get("ticker", "")
    composite = float(p.get("composite") or 0)
    key = f"{t}::{horizon}"
    pos = sources["positions"].get(key, {})
    state = pos.get("state", "PROSPECTING")
    sig_days = (pos.get("consecutive_signal_days") or {}).get("BUY_NOW", 0)

    # ── 3 AGENT VOTES ──
    # 1. PM Agent vote (conviction)
    scan_row = sources["scan_lookup"].get(t, {})
    pm_pick_enriched = {
        **p,
        "classification": scan_row.get("classification") or p.get("classification") or "",
    }
    pm_v = pm_vote(pm_pick_enriched)

    # 2. Trading Agent vote (entry_signal)
    trading_v = trading_vote(timing)

    # 3. Risk Manager vote (deterministic score)
    risk_breakdown = compute_risk_score(scan_row, sources["sector_counts"])
    risk_v = risk_vote(risk_breakdown)
    risk_reason = risk_rationale(risk_breakdown)

    # Tally votes → stars
    tally = tally_votes(pm_v, trading_v, risk_v)
    stars = tally["stars"]

    # Filter: REJECTED tier (≥2 REJECT) excluded from Final List
    if stars == 0:
        return None

    # Legacy cross-check info (for display "additional info" but no longer gates filtering)
    proxy_set = sources["proxy_long_stocks"] if "stocks" in bucket else sources["proxy_long_etfs"]
    top_set   = sources["top_stocks"]        if "stocks" in bucket else sources["top_etfs"]
    in_proxy_recent = t in proxy_set
    in_top_alpha    = t in top_set
    state_confirmed = state == "ENTERED"
    high_composite  = composite >= HIGH_COMPOSITE_THRESHOLD
    n_validations = sum([in_proxy_recent, in_top_alpha, state_confirmed, high_composite])

    trailing = _compute_trailing_returns(t, sources.get("price_data") or {})

    # Action urgency (Phase 5.6 state machine integration)
    if state == "ENTERED" and pos.get("last_alert") and "NEW BUY" in (pos.get("last_alert") or ""):
        action = "EXECUTE_TODAY"
    elif state == "ENTERED":
        action = "ALREADY_HELD"
    elif state == "PROSPECTING" and sig_days >= 1:
        action = "WATCH_TOMORROW"
    elif state == "HOLDING":
        action = "ALREADY_HELD"
    else:
        action = "OBSERVE"

    return {
        "ticker": t,
        "name": (p.get("name") or "")[:40],
        "sector": p.get("sector") or "",
        "composite": composite,
        "horizon": horizon,
        "bucket": bucket,
        "stars": stars,
        # ── 3-Agent Voting (Option C) ──
        "votes": {
            "pm":      pm_v,
            "trading": trading_v,
            "risk":    risk_v,
        },
        "consensus":   tally["consensus"],
        "n_approve":   tally["n_approve"],
        "n_reject":    tally["n_reject"],
        "n_caution":   tally["n_caution"],
        "risk_score":  risk_breakdown["total"],
        "risk_breakdown": risk_breakdown,
        "risk_reason": risk_reason,
        # Legacy cross-check info (still shown, no longer gates)
        "n_validations": n_validations,
        "in_proxy_recent":   in_proxy_recent,
        "in_proxy_latest":   in_proxy_recent,
        "in_top_alpha":      in_top_alpha,
        "state_confirmed":   state_confirmed,
        "high_composite":    high_composite,
        # Position state info
        "state": state,
        "signal_days": sig_days,
        "urgency": timing.get("urgency", "NORMAL"),
        "action": action,
        # Reasoning
        "entry_trigger": (timing.get("entry_trigger") or "")[:200],
        "rationale": (p.get("rationale") or "")[:300],                # PM Agent's reason
        "trading_rationale": (timing.get("rationale") or "")[:300],   # Trading Agent's reason
        "exit_triggers": (timing.get("exit_triggers") or [])[:2],
        # Trailing total returns
        **trailing,
    }


def _pm_vote_short(pick: dict) -> str:
    """PM Agent vote for SHORT side (mirror of LONG)."""
    composite = float(pick.get("composite") or 0)
    cls = pick.get("classification", "") or ""
    weak = ("DOWNTREND", "WEAKENING", "FADING", "CYCLE_PEAK")
    strong_bear = any(s in cls for s in weak)
    if composite <= 30 and strong_bear:  return "APPROVE"
    if composite <= 25:                  return "APPROVE"
    if composite >= 50:                  return "REJECT"
    return "CAUTION"


def _risk_vote_short(scan_row: dict, sector_counts: dict) -> tuple[str, dict, str]:
    """Risk vote for SHORT side — mirrors the long-side risk logic.

    For shorts, the major risk dimensions are:
      - Squeeze risk (rising momentum, low short interest data, oversold bounce)
      - Liquidity risk (illiquid shorts dangerous)
      - Drawdown via short squeeze
    We approximate by inverting overheating: high RSI/OER on a "weak" ticker = bounce risk.
    """
    from agents.risk_manager import compute_risk_score, RISK_VOTE_APPROVE_MAX, RISK_VOTE_REJECT_MIN
    # Reuse long-side risk score but interpret as "execution risk of short"
    rb = compute_risk_score(scan_row, sector_counts)
    score = rb.get("total", 50)
    if score <= RISK_VOTE_APPROVE_MAX: vote = "APPROVE"
    elif score >= RISK_VOTE_REJECT_MIN: vote = "REJECT"
    else: vote = "CAUTION"
    reason = f"Short 실행 리스크 총점 {score:.0f}/100"
    return vote, rb, reason


def _eval_sell_pick(p: dict, horizon: str, bucket: str, sources: dict) -> Optional[dict]:
    """Score one PM Sell/Short candidate using 3-Agent Voting (mirror of buy)."""
    from agents.risk_manager import trading_vote, tally_votes

    timing = p.get("timing") or {}
    t = p.get("ticker", "")
    composite = float(p.get("composite") or 0)
    key = f"{t}::{horizon}"
    pos = sources["positions"].get(key, {})
    state = pos.get("state", "PROSPECTING")
    sig_days = (pos.get("consecutive_signal_days") or {}).get("BUY_NOW", 0)

    # 3 votes (mirror buy side, adapted for SHORT)
    scan_row = sources["scan_lookup"].get(t, {})
    pm_pick_enriched = {**p, "classification": scan_row.get("classification") or p.get("classification") or ""}
    pm_v = _pm_vote_short(pm_pick_enriched)
    trading_v = trading_vote(timing)
    risk_v, risk_breakdown, risk_reason = _risk_vote_short(scan_row, sources["sector_counts"])

    tally = tally_votes(pm_v, trading_v, risk_v)
    stars = tally["stars"]
    if stars == 0:
        return None

    # Legacy worst-alpha cross-check
    worst_set = sources["worst_stocks"] if "stocks" in bucket else sources["worst_etfs"]
    in_worst_alpha = t in worst_set
    state_confirmed = state == "ENTERED"
    low_composite   = composite <= 30
    n_validations = sum([in_worst_alpha, state_confirmed, low_composite])

    if state == "ENTERED" and pos.get("last_alert") and "NEW BUY" in (pos.get("last_alert") or ""):
        action = "EXECUTE_TODAY"
    elif state == "PROSPECTING" and sig_days >= 1:
        action = "WATCH_TOMORROW"
    else:
        action = "OBSERVE"

    trailing = _compute_trailing_returns(t, sources.get("price_data") or {})

    return {
        "ticker": t,
        "name": (p.get("name") or "")[:40],
        "sector": p.get("sector") or "",
        "composite": composite,
        "horizon": horizon,
        "bucket": bucket,
        "stars": stars,
        # ── 3-Agent Voting (mirror of buy) ──
        "votes": {"pm": pm_v, "trading": trading_v, "risk": risk_v},
        "consensus":   tally["consensus"],
        "n_approve":   tally["n_approve"],
        "n_reject":    tally["n_reject"],
        "n_caution":   tally["n_caution"],
        "risk_score":  risk_breakdown["total"],
        "risk_breakdown": risk_breakdown,
        "risk_reason": risk_reason,
        # Legacy cross-check
        "n_validations": n_validations,
        "in_worst_alpha":    in_worst_alpha,
        "state_confirmed":   state_confirmed,
        "low_composite":     low_composite,
        # Position state
        "state": state,
        "signal_days": sig_days,
        "urgency": timing.get("urgency", "NORMAL"),
        "action": action,
        # Reasoning
        "entry_trigger": (timing.get("entry_trigger") or "")[:200],
        "rationale": (p.get("rationale") or "")[:300],
        "trading_rationale": (timing.get("rationale") or "")[:300],
        # Trailing total returns
        **trailing,
    }


def _exit_pick(t: str, horizon: str, pos: dict) -> dict:
    """Build an 'exit existing position' entry from Phase 5.6 state."""
    return {
        "ticker": t, "name": "", "sector": "", "composite": 0,
        "horizon": horizon, "bucket": "(existing)",
        "stars": 3,
        "state": pos.get("state"),
        "signal_days": (pos.get("consecutive_signal_days") or {}).get("SKIP", 0),
        "urgency": "URGENT",
        "action": "CLOSE_TODAY",
        "entry_trigger": "",
        "rationale": (pos.get("state_history") or [{}])[-1].get("reason", "")[:140],
        "is_exit": True,
    }


# ─────────────────────────────────────────────────────────────────────
# Main entry — assemble both lists
# ─────────────────────────────────────────────────────────────────────

def _days_between(start_iso: str, end_iso: str) -> int:
    from datetime import datetime
    try:
        a = datetime.strptime(start_iso, "%Y-%m-%d").date()
        b = datetime.strptime(end_iso, "%Y-%m-%d").date()
        return (b - a).days
    except Exception:
        return 0


def _build_active_positions(sources: dict, buy_list: list, sell_list: list) -> list[dict]:
    """Build list of currently HELD positions (HOLDING / ENTERED state).

    These persist even if today's PM picks didn't include them.
    Cross-reference with today's buy/sell to mark which are in_today_picks.
    """
    import time
    today = time.strftime("%Y-%m-%d")
    positions = sources.get("positions") or {}
    scan_lookup = sources.get("scan_lookup") or {}
    price_data = sources.get("price_data") or {}
    sw = sources.get("pm_swarm") or {}
    horizons = (sw.get("phase5_pm") or {}).get("horizons", {})

    # Set of (ticker, horizon) in today's voted buy/sell lists for cross-ref
    today_buy_set = {(r["ticker"], r.get("horizon")) for r in buy_list}
    today_sell_set = {(r["ticker"], r.get("horizon")) for r in sell_list}

    active = []
    seen = set()
    for key, pos in positions.items():
        ticker, horizon = key.split("::", 1)
        state = pos.get("state", "")
        if state not in ("HOLDING", "ENTERED"):
            continue
        if (ticker, horizon) in seen:
            continue
        seen.add((ticker, horizon))

        # Find current PM pick (if today picked again)
        pm_pick = None
        bucket = None
        for b in ("long_stocks", "long_etfs", "short_stocks", "short_etfs"):
            for p in horizons.get(horizon, {}).get(b, []) or []:
                if p.get("ticker") == ticker:
                    pm_pick = p
                    bucket = b
                    break
            if pm_pick: break

        # Metadata
        scan_row = scan_lookup.get(ticker, {})
        # Agent debate fields — populated when the held position is ALSO in today's
        # PM swarm picks (mirror _eval_buy_pick_with_debate so HOLDING rows don't lose
        # the Debate Transcript / 3-Agent Votes columns).
        debate_fields: dict = {}
        if pm_pick:
            name = pm_pick.get("name") or ""
            sector = pm_pick.get("sector") or scan_row.get("sector") or scan_row.get("category", "")
            composite = float(pm_pick.get("composite") or scan_row.get("composite") or 0)
            rationale = (pm_pick.get("rationale") or "")[:200]
            timing = pm_pick.get("timing") or {}
            current_signal = timing.get("entry_signal", "—")
            trading_rationale = (timing.get("rationale") or "")[:200]
            # Extract debate synthesis (today's 3-agent verdict for this held position)
            debate = pm_pick.get("debate_synthesis") or {}
            risk_verdict = pm_pick.get("risk_verdict") or {}
            if debate and not debate.get("_failed"):
                pm_vote_d = "APPROVE"   # PM picked it → implicit approve
                trading_sig = (timing.get("entry_signal") or "").upper()
                trading_vote_d = ("APPROVE" if trading_sig == "BUY_NOW"
                                   else "REJECT" if trading_sig == "SKIP" else "CAUTION")
                risk_vote_d = (risk_verdict.get("vote") or "CAUTION").upper()
                debate_fields = {
                    "tier": debate.get("tier"),
                    "consensus": debate.get("tier"),
                    "stars": debate.get("stars"),
                    "debate_transcript": debate.get("debate_transcript", ""),
                    "key_factor": debate.get("key_factor", ""),
                    "final_decision": debate.get("final_decision"),
                    "votes": {"pm": pm_vote_d, "trading": trading_vote_d, "risk": risk_vote_d},
                    "n_approve": sum(1 for v in (pm_vote_d, trading_vote_d, risk_vote_d) if v == "APPROVE"),
                    "n_reject":  sum(1 for v in (pm_vote_d, trading_vote_d, risk_vote_d) if v == "REJECT"),
                    "n_caution": sum(1 for v in (pm_vote_d, trading_vote_d, risk_vote_d) if v == "CAUTION"),
                }
        else:
            name = scan_row.get("name") or ""
            sector = scan_row.get("sector") or scan_row.get("category", "")
            composite = float(scan_row.get("composite") or 0)
            rationale = "오늘 PM picks에 없음 — 보유 유지 중 (직전 진입 근거로 보유 지속)"
            current_signal = "—"
            trading_rationale = "오늘 Trading Agent 미평가 — HOLDING은 sticky (신규 신호 없어도 보유 유지)"
            # No today verdict — mark clearly so frontend shows "오늘 미평가" instead of blank
            debate_fields = {
                "tier": "HELD_NO_EVAL",
                "consensus": "HELD_NO_EVAL",
                "debate_transcript": "오늘 PM swarm이 이 종목을 재평가하지 않음 — 보유 포지션 유지 중. "
                                      "신규 매수/청산 신호 발생 시에만 재토론. (직전 진입 근거는 유효)",
                "key_factor": "보유 유지",
                "final_decision": "HOLD",
            }

        # Compute days
        entered_date = pos.get("entered_date")
        first_seen = pos.get("first_seen") or entered_date
        days_held = _days_between(entered_date, today) if entered_date else 0
        persistence_days = _days_between(first_seen, today) if first_seen else 0

        # Trailing returns
        trailing = _compute_trailing_returns(ticker, price_data)

        # Risk
        from agents.risk_manager import compute_risk_score, risk_vote, risk_rationale
        risk_breakdown = compute_risk_score(scan_row, sources.get("sector_counts") or {})
        risk_v = risk_vote(risk_breakdown)

        # Bucket fallback: if PM didn't include this position today, infer from scan category
        if not bucket:
            bucket = _infer_bucket(scan_row, side="long")
        active.append({
            "ticker": ticker,
            "name": name[:40],
            "sector": sector,
            "composite": composite,
            "horizon": horizon,
            "bucket": bucket,
            "state": state,
            "entered_date": entered_date,
            "first_seen": first_seen,
            "days_held": days_held,
            "persistence_days": persistence_days,
            "in_today_buy_picks":  (ticker, horizon) in today_buy_set,
            "in_today_sell_picks": (ticker, horizon) in today_sell_set,
            "current_signal": current_signal,
            "rationale": rationale,
            "trading_rationale": trading_rationale,
            "risk_score": risk_breakdown["total"],
            "risk_vote": risk_v,
            "risk_reason": risk_rationale(risk_breakdown),
            "last_alert": pos.get("last_alert"),
            **debate_fields,
            **trailing,
        })

    # Sort by days_held descending (longest held first)
    active.sort(key=lambda x: (-x.get("days_held", 0), -x.get("composite", 0)))
    return active


def _build_exit_pending(sources: dict) -> list[dict]:
    """Build list of positions with EXIT_PENDING state."""
    import time
    today = time.strftime("%Y-%m-%d")
    positions = sources.get("positions") or {}
    scan_lookup = sources.get("scan_lookup") or {}
    price_data = sources.get("price_data") or {}

    pending = []
    for key, pos in positions.items():
        if pos.get("state") != "EXIT_PENDING":
            continue
        ticker, horizon = key.split("::", 1)
        scan_row = scan_lookup.get(ticker, {})
        entered_date = pos.get("entered_date")
        first_seen = pos.get("first_seen") or entered_date
        days_held = _days_between(entered_date, today) if entered_date else 0
        persistence_days = _days_between(first_seen, today) if first_seen else 0

        trailing = _compute_trailing_returns(ticker, price_data)

        # Risk evaluation — match HOLDING records so frontend renders Risk Reason column
        from agents.risk_manager import compute_risk_score, risk_vote, risk_rationale
        risk_breakdown = compute_risk_score(scan_row, sources.get("sector_counts") or {})
        risk_v = risk_vote(risk_breakdown)

        exit_reason = (pos.get("state_history") or [{}])[-1].get("reason", "")[:200]

        # State-driven exit (NOT a swarm debate) — explain WHY there's no agent transcript.
        # The position_state machine triggers EXIT_PENDING mechanically when:
        #   • SKIP × 2 days while HOLDING (consistent exit signal), or
        #   • WAIT × 5 days while HOLDING (signal degraded), or
        #   • classification flipped to bearish.
        # No 3-agent swarm re-debate happens for these — it's a rule-based exit.
        held_no_debate_msg = (
            f"⚙️ 기계적 청산 신호 (swarm 토론 아님) — {exit_reason or 'state machine 청산 판정'}. "
            f"position_state 규칙: HOLDING 중 SKIP 2일 연속 / WAIT 5일 연속 / 약세 분류 전환 시 "
            f"자동 EXIT_PENDING 전환. 신규 SHORT 시그널이 아니므로 3-Agent 재토론 미발생. "
            f"보유 {days_held}일째 — 청산 실행 권장."
        )

        pending.append({
            "ticker": ticker,
            "name": (scan_row.get("name") or "")[:40],
            "sector": scan_row.get("sector") or scan_row.get("category", ""),
            "composite": float(scan_row.get("composite") or 0),
            "horizon": horizon,
            "bucket": _infer_bucket(scan_row, side="long"),   # ETF vs Stock for frontend grouping
            "state": "EXIT_PENDING",
            "tier": "STATE_DRIVEN_EXIT",   # signals frontend this is mechanical, not swarm
            "consensus": "STATE_DRIVEN_EXIT",
            "entered_date": entered_date,
            "first_seen": first_seen,
            "days_held": days_held,
            "persistence_days": persistence_days,
            "exit_reason": exit_reason,
            "rationale": exit_reason or "포지션 청산 후보 — state machine이 EXIT_PENDING 판정",
            "debate_transcript": held_no_debate_msg,   # explain blank-transcript reason
            "key_factor": "기계적 청산",
            "final_decision": "EXIT",
            "current_signal": "—",   # no current BUY_NOW signal for exits
            "trading_rationale": "오늘 Trading Agent 미평가 (청산 대상) — state machine 규칙 기반 청산이라 timing agent 재평가 없음.",
            "risk_score": risk_breakdown["total"],
            "risk_vote": risk_v,
            "risk_reason": risk_rationale(risk_breakdown),
            "in_today_buy_picks":  False,
            "in_today_sell_picks": False,
            "last_alert": pos.get("last_alert"),
            **trailing,
        })
    pending.sort(key=lambda x: -x.get("days_held", 0))
    return pending


def _promote_holding_short_overlap(active_positions: list[dict],
                                    exit_pending: list[dict],
                                    sell_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """Promote HOLDING/ENTERED positions that received SHORT signals to EXIT_PENDING.

    Returns (filtered_active_positions, augmented_exit_pending).
    A position currently held LONG that PM swarm flags as SHORT today is a
    "regime-flip exit" — surface it in EXIT_PENDING so the user closes the long
    instead of being confused by it appearing as both HOLDING and SHORT-buy.
    """
    # Map of SHORT-recommended tickers → first matching short pick (any horizon)
    short_picks: dict[str, dict] = {}
    for r in sell_list:
        # Skip records that are themselves position-state exits (not new SHORT picks)
        if r.get("action") == "CLOSE_TODAY" or r.get("state") in ("EXIT_PENDING", "EXITED"):
            continue
        # Only treat sell-side PM picks (bucket starts with short_) as fresh SHORT signals
        if not str(r.get("bucket", "")).startswith("short_"):
            continue
        short_picks.setdefault(r["ticker"], r)

    if not short_picks:
        return active_positions, exit_pending

    already_pending = {r["ticker"] for r in exit_pending}
    kept_active: list[dict] = []
    promoted: list[dict] = []

    for pos in active_positions:
        t = pos.get("ticker")
        if t in short_picks and t not in already_pending and pos.get("state") in ("HOLDING", "ENTERED"):
            sp = short_picks[t]
            # Pull the SHORT pick's swarm debate (it DID go through Phase 5 + per-ticker debate)
            sp_debate = (sp.get("debate_synthesis") or sp.get("_pt_debate") or {})
            sp_transcript = (sp.get("debate_transcript")
                              or sp_debate.get("debate_transcript")
                              or sp.get("rationale") or "")
            sp_tier = sp.get("tier") or sp_debate.get("tier") or "SOLO"
            # Build a regime-flip explanation that combines the SHORT-side debate
            flip_transcript = (
                f"🔻 Regime-Flip 청산 — 보유 LONG이 오늘 PM swarm에서 SHORT으로 재판정됨. "
                f"[SHORT 토론 ★{sp.get('stars',1)} / {sp_tier}]: {sp_transcript[:280]}"
            )
            sp_votes = sp.get("votes") or {}
            promoted.append({
                "ticker": t,
                "name": pos.get("name", ""),
                "sector": pos.get("sector", ""),
                "composite": pos.get("composite", 0),
                "horizon": pos.get("horizon", "core"),
                "bucket": pos.get("bucket") or "long_stocks",   # inherit from active position
                "state": "EXIT_PENDING",
                "tier": "REGIME_FLIP",
                "consensus": "REGIME_FLIP",
                "entered_date": pos.get("entered_date"),
                "first_seen": pos.get("first_seen") or pos.get("entered_date"),
                "days_held": pos.get("days_held", 0),
                "persistence_days": pos.get("persistence_days", 0),
                "exit_reason": f"보유 중 LONG + 오늘 PM SHORT 시그널 감지 (★{sp.get('stars', 1)} / "
                               f"{sp.get('action', '?')}) — regime flip 청산 권장",
                "rationale": (sp.get("rationale") or sp_transcript or "")[:300]
                              or "Regime flip: 보유 LONG + 오늘 PM SHORT 시그널",
                # Carry the SHORT-side swarm debate so the transcript column isn't blank
                "debate_transcript": flip_transcript,
                "key_factor": "Regime-Flip (LONG→SHORT)",
                "final_decision": "EXIT",
                "votes": sp_votes if sp_votes else None,
                "n_approve": sp.get("n_approve"),
                "n_reject": sp.get("n_reject"),
                "n_caution": sp.get("n_caution"),
                "current_signal": "SELL",   # PM flagged this as SHORT today
                "trading_rationale": (
                    f"오늘 PM swarm이 SHORT 시그널 발생 (★{sp.get('stars',1)}, {sp.get('horizon','?')} horizon) "
                    f"→ 보유 LONG 청산 권고. SHORT 측 Trading 판단: {(sp.get('trading_rationale') or '추세 하방 전환')[:150]}"
                ),
                "promoted_from_short": True,
                "short_signal_stars": sp.get("stars", 1),
                "short_signal_action": sp.get("action", "?"),
                "short_signal_horizon": sp.get("horizon", "?"),
                "short_signal_reason": (sp.get("rationale") or sp_transcript or "")[:300],
                "last_alert": pos.get("last_alert"),
                # Returns — inherit from active position (already populated via _compute_trailing_returns)
                "ret_5d":  pos.get("ret_5d"),
                "ret_1mo": pos.get("ret_1mo"),
                "ret_3mo": pos.get("ret_3mo"),
                "ret_6mo": pos.get("ret_6mo"),
                "ret_1y":  pos.get("ret_1y"),
                # Risk — inherit from active position
                "risk_score":  pos.get("risk_score"),
                "risk_vote":   pos.get("risk_vote"),
                "risk_reason": pos.get("risk_reason"),
                "in_today_buy_picks":  False,
                "in_today_sell_picks": True,
            })
        else:
            kept_active.append(pos)

    # Promoted items go to top of exit_pending (more urgent than passive state-driven exits)
    return kept_active, promoted + exit_pending


def build_final_lists() -> dict:
    """Top-level builder. Returns {buy_list, sell_list, metadata}."""
    sources = _load_sources()
    sw = sources["pm_swarm"]
    horizons = (sw.get("phase5_pm") or {}).get("horizons") or {}

    buy_list: list[dict] = []
    sell_list: list[dict] = []

    for h in ("tactical", "core", "strategic"):
        h_data = horizons.get(h) or {}
        # BUY side
        for bucket in ("long_stocks", "long_etfs"):
            for p in (h_data.get(bucket) or []):
                rec = _eval_buy_pick_with_debate(p, h, bucket, sources)
                if rec: buy_list.append(rec)
        # SELL side (SHORT picks = bet against)
        for bucket in ("short_stocks", "short_etfs"):
            for p in (h_data.get(bucket) or []):
                rec = _eval_sell_pick(p, h, bucket, sources)
                if rec: sell_list.append(rec)

    # ALSO add exit candidates from position state (existing LONG to close)
    for key, pos in (sources["positions"] or {}).items():
        if pos.get("state") in ("EXIT_PENDING", "EXITED"):
            t, h = key.split("::", 1)
            sell_list.append(_exit_pick(t, h, pos))

    # Sort: stars desc → urgency (URGENT first) → composite desc
    URG_ORDER = {"URGENT": 0, "NORMAL": 1, "PATIENT": 2}
    ACT_ORDER = {"EXECUTE_TODAY": 0, "CLOSE_TODAY": 0, "WATCH_TOMORROW": 1, "ALREADY_HELD": 2, "OBSERVE": 3}
    buy_list.sort(key=lambda x: (-x["stars"], ACT_ORDER.get(x["action"], 9),
                                  URG_ORDER.get(x["urgency"], 9), -x.get("composite", 0)))
    sell_list.sort(key=lambda x: (-x["stars"], ACT_ORDER.get(x["action"], 9),
                                   URG_ORDER.get(x["urgency"], 9), -x.get("composite", 0)))

    # Dedup by (ticker, bucket) — same ticker may appear in multiple horizons
    def _dedup(lst):
        seen = set(); out = []
        for r in lst:
            k = (r["ticker"], r["bucket"])
            if k in seen: continue
            seen.add(k); out.append(r)
        return out

    buy_dedup = _dedup(buy_list)
    sell_dedup = _dedup(sell_list)

    # Build active_positions section — currently HELD tickers (regardless of today's voting)
    active_positions = _build_active_positions(sources, buy_dedup, sell_dedup)
    exit_pending = _build_exit_pending(sources)

    # Promote HOLDING + today's SHORT signal → EXIT_PENDING (regime-flip exit)
    # Eliminates user confusion of seeing same ticker as HOLDING and Sell candidate.
    active_positions, exit_pending = _promote_holding_short_overlap(
        active_positions, exit_pending, sell_dedup
    )

    # Build category groups for per-category commentaries (mirror frontend merge logic)
    held_set = {p["ticker"] for p in active_positions if p.get("state") in ("HOLDING", "ENTERED")}
    new_picks = [r for r in buy_dedup
                  if r.get("state") != "ENTERED" and r.get("action") != "EXECUTE_TODAY"
                  and r.get("ticker") not in {p["ticker"] for p in active_positions if p.get("state") == "HOLDING"}]

    # ── Holdings-aware re-ranking (방식 A+B) ──
    # Re-score NEW candidates by considering the existing portfolio:
    #   A) sector concentration cap — penalize picks piling into crowded sectors
    #   B) correlation penalty — penalize picks highly correlated with held names
    # This surfaces diversifying picks (unheld sectors, low-correlation) ahead of
    # redundant ones (e.g. EEM when IEMG already held, ρ≈0.99).
    try:
        from agents.holdings_aware_selection import rerank_new_picks, summarize_reranking
        held_for_rerank = [p for p in active_positions if p.get("state") in ("HOLDING", "ENTERED")]
        if new_picks and held_for_rerank:
            rerank_new_picks(new_picks, held_for_rerank, sources.get("price_data") or {})
            sources["_ha_rerank_summary"] = summarize_reranking(new_picks)
    except Exception as _ha_err:
        sources["_ha_rerank_error"] = str(_ha_err)[:200]

    items_by_category_full = {
        "ENTERED":      [r for r in buy_dedup if r.get("state") == "ENTERED" and r.get("action") == "EXECUTE_TODAY"],
        "EXIT_PENDING": exit_pending,
        "HOLDING":      [r for r in active_positions if r.get("state") == "HOLDING"],
        "NEW":          new_picks,
    }

    # ── Turnover cap: GLOBAL 20 stocks + 20 ETFs across the WHOLE Buy Final List ──
    # Asset-class balance: top 20 stocks + top 20 ETFs by quality score (regardless of category).
    # Each ticker keeps its category assignment; tickers below the cap are filtered out
    # of all categories. Replaces the previous "5 per category" cap.
    STOCK_CAP = 20
    ETF_CAP = 20
    # EXIT_PENDING tickers represent URGENT sell signals — bypass the 20-cap.
    # Up to this many EXIT_PENDING entries always surface (per asset type).
    EXIT_PENDING_CAP = 15

    def _score_for_cap(r: dict) -> float:
        # NEW picks carry a holdings-aware adjusted score (방식 A+B) — use it so the
        # 20+20 cap keeps the DIVERSIFIED top set, not the crowded-sector one.
        ha = r.get("ha_adjusted_score")
        if ha is not None:
            return float(ha) + (50 if r.get("promoted_from_short") else 0)
        return (
            (r.get("stars") or 0) * 100
            + (r.get("composite") or 0)
            + (r.get("days_held") or 0) * 0.1
            + (50 if r.get("promoted_from_short") else 0)
        )

    def _is_etf(r: dict) -> bool:
        bk = (r.get("bucket") or "").lower()
        if "etf" in bk: return True
        if "stock" in bk: return False
        at = (r.get("asset_type") or r.get("type") or "").lower()
        return "etf" in at

    # 1) Flatten all categories with category label attached + dedupe by ticker (highest score wins)
    flat: dict[str, tuple[str, dict]] = {}
    cat_priority = {"EXIT_PENDING": 0, "ENTERED": 1, "HOLDING": 2, "NEW": 3}
    for cat, rows in items_by_category_full.items():
        for r in rows:
            t = r.get("ticker")
            if not t: continue
            score = _score_for_cap(r)
            if t in flat:
                old_cat, old_r = flat[t]
                # Higher-priority (more urgent) category wins; tiebreak on score
                p_new = cat_priority.get(cat, 9)
                p_old = cat_priority.get(old_cat, 9)
                if p_new < p_old or (p_new == p_old and score > _score_for_cap(old_r)):
                    flat[t] = (cat, r)
            else:
                flat[t] = (cat, r)

    # 2) Split EXIT_PENDING out — they get reserved slots, not subject to global cap.
    flat_items = [(cat, r) for cat, r in flat.values()]
    exit_pending_stocks = sorted(
        [(cat, r) for cat, r in flat_items if cat == "EXIT_PENDING" and not _is_etf(r)],
        key=lambda x: -_score_for_cap(x[1]))[:EXIT_PENDING_CAP]
    exit_pending_etfs = sorted(
        [(cat, r) for cat, r in flat_items if cat == "EXIT_PENDING" and _is_etf(r)],
        key=lambda x: -_score_for_cap(x[1]))[:EXIT_PENDING_CAP]

    # 3) Apply the 20+20 cap to non-EXIT_PENDING (entered/holding/new) only.
    other_items = [(cat, r) for cat, r in flat_items if cat != "EXIT_PENDING"]
    stocks = sorted([(cat, r) for cat, r in other_items if not _is_etf(r)],
                    key=lambda x: -_score_for_cap(x[1]))[:STOCK_CAP]
    etfs   = sorted([(cat, r) for cat, r in other_items if _is_etf(r)],
                    key=lambda x: -_score_for_cap(x[1]))[:ETF_CAP]
    kept = stocks + etfs + exit_pending_stocks + exit_pending_etfs

    # 3) Redistribute back into categories (whitelist)
    items_by_category: dict[str, list[dict]] = {"ENTERED": [], "EXIT_PENDING": [], "HOLDING": [], "NEW": []}
    for cat, r in kept:
        items_by_category.setdefault(cat, []).append(r)
    # Stable sort within each category by score
    for cat in items_by_category:
        items_by_category[cat].sort(key=_score_for_cap, reverse=True)

    # Update persistence_days on buy/sell items
    try:
        from agents.trade_log import update_log_from_positions, get_persistence_days
        update_log_from_positions(sources.get("positions") or {})
        for r in buy_dedup:
            r["persistence_days"] = get_persistence_days(r["ticker"])
        for r in sell_dedup:
            r["persistence_days"] = get_persistence_days(r["ticker"])
    except Exception:
        pass

    # Executive Commentary — UNIFIED 10,000-char.
    # OPTIMIZATION: only return CACHED commentary in the API path. If the cache is stale,
    # serve last-known commentary (or empty placeholder) and let a separate background job
    # regenerate it. This keeps /api/final-list response under 1 second even when commentary
    # needs to be rebuilt (which takes 3-5 minutes with the 10,000-char prompt + WebSearch).
    commentary = {}
    category_commentary = {}  # deprecated; kept empty for backward compat
    try:
        from agents.final_list_commentary import build_executive_commentary
        pm_commentary = (sw.get("phase5_pm") or {}).get("pm_commentary", "")[:600]
        commentary = build_executive_commentary(
            buy_dedup, sell_dedup,
            market_context=pm_commentary,
            items_by_category=items_by_category,
            swarm_generated_at=sw.get("generated_at", ""),
            cache_only=True,   # ← never trigger LLM call inside the API path
        )
    except Exception as e:
        commentary = {"unified_commentary": "", "buy_commentary": "", "sell_commentary": "",
                      "unified_cached": False, "buy_cached": False, "sell_cached": False,
                      "error": str(e)[:200]}

    # Compute capped ticker set per category (frontend uses to filter mergedItems → top-5 cap)
    capped_tickers_by_category = {
        cat: [r.get("ticker") for r in rows if r.get("ticker")]
        for cat, rows in items_by_category.items()
    }

    return {
        "buy_list":  buy_dedup,
        "sell_list": sell_dedup,
        "active_positions": active_positions,
        "exit_pending":     exit_pending,
        "commentary": commentary,
        "category_commentary": category_commentary,
        "items_by_category": items_by_category,        # NEW: capped top-5 per cat (canonical)
        "capped_tickers_by_category": capped_tickers_by_category,  # NEW: ticker whitelist for frontend
        "metadata": {
            "proxy_cohort_date": sources["proxy_cohort_date"],
            "n_proxy_long_stocks": len(sources["proxy_long_stocks"]),
            "n_proxy_long_etfs":   len(sources["proxy_long_etfs"]),
            "n_top_stocks": len(sources["top_stocks"]),
            "n_top_etfs":   len(sources["top_etfs"]),
            "n_positions_tracked": len(sources["positions"]),
            "swarm_generated_at": sw.get("generated_at", ""),
            # Holdings-aware NEW pick re-ranking summary (방식 A+B)
            "holdings_aware_rerank": sources.get("_ha_rerank_summary"),
        },
    }
