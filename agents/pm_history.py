# -*- coding: utf-8 -*-
"""pm_history.py — append-only PM swarm picks history for forward collection.

Snapshot triggered automatically after a successful fresh swarm run
(market_leaders_swarm.run_swarm). Each daily snapshot captures:
  - date / snapshot_at
  - regime_tag_deterministic
  - phase4_action picks (20 × 4 buckets, 80 names)
  - phase5_pm picks    (PM-revised 80 names, with change_type tags)

Stored in `.pm_picks_history.json` (gitignored).

Future use:
  1. After 4-8 weeks of accumulation → backfill forward returns
  2. Compare PM agent's actual picks vs deterministic proxy from same date
  3. Quantify PM agent's marginal alpha vs rule-based baseline
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

HISTORY_PATH = Path(".pm_picks_history.json")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {"snapshots": [], "created_at": _now()}
    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"snapshots": [], "created_at": _now()}


def _strip_pick(p: dict) -> dict:
    """Keep only fields needed for forward evaluation."""
    return {
        "ticker": p.get("ticker"),
        "name": (p.get("name") or "")[:40],
        "composite": p.get("composite"),
        "sector": p.get("sector"),
        "rationale": (p.get("rationale") or "")[:200],
        "change_type": p.get("change_type"),     # only present in phase5_pm picks
        "pool_source": p.get("pool_source"),      # momentum|pre_momentum (forming 생존률 추적, 2026-07-28)
        "pre_momentum_score": p.get("pre_momentum_score"),
    }


def _strip_bucket(picks: list) -> list:
    return [_strip_pick(p) for p in (picks or [])[:20]]


def append_trading_snapshot(swarm_result: dict, source: str = "fresh") -> dict:
    """Append Phase 5.5 Trading Agent signals to forward-collection history.

    Stored separately from PM picks history so we can compare:
      - PM-only performance (from .pm_picks_history.json)
      - PM + actual Trading Agent timing (from .trading_signals_history.json)
    """
    if not isinstance(swarm_result, dict):
        return {"appended": False, "reason": "invalid_payload"}
    pm = swarm_result.get("phase5_pm") or {}
    tt = pm.get("trading_timing") or {}
    if not tt or not tt.get("core"):
        return {"appended": False, "reason": "no_trading_timing"}

    from pathlib import Path as _P
    import json as _json, time as _time
    TRADING_HISTORY = _P(".trading_signals_history.json")

    def _strip_timing(timings: dict) -> dict:
        """Compact: keep only the essentials per ticker."""
        out = {}
        for t, sig in (timings or {}).items():
            if not isinstance(sig, dict): continue
            out[t] = {
                "entry_signal":  sig.get("entry_signal"),
                "entry_trigger": (sig.get("entry_trigger") or "")[:200],
                "exit_triggers": [{"type": e.get("type"), "condition": (e.get("condition") or "")[:120],
                                   "action": e.get("action")}
                                  for e in (sig.get("exit_triggers") or [])[:5]],
                "urgency":       sig.get("urgency"),
            }
        return out

    today = _today()
    snapshot = {
        "date": today,
        "snapshot_at": _now(),
        "source": source,
        "horizons": {
            h: _strip_timing((tt.get(h) or {}).get("timings", {}))
            for h in ("core",)
        },
    }

    history = _json.loads(TRADING_HISTORY.read_text(encoding="utf-8")) \
              if TRADING_HISTORY.exists() else {"snapshots": []}
    snaps = history.setdefault("snapshots", [])
    overwritten = False
    for i, s in enumerate(snaps):
        if s.get("date") == today:
            snaps[i] = snapshot; overwritten = True; break
    if not overwritten:
        snaps.append(snapshot)
    history["last_update"] = _now()
    history["n_snapshots"] = len(snaps)
    TRADING_HISTORY.write_text(_json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"appended": True, "overwritten": overwritten, "date": today,
            "n_snapshots": len(snaps)}


def append_snapshot(swarm_result: dict, source: str = "fresh") -> dict:
    """Append today's PM picks snapshot. Dedup by date (overwrite if same day)."""
    if not isinstance(swarm_result, dict):
        return {"appended": False, "reason": "invalid_payload"}

    p4 = swarm_result.get("phase4_action") or {}
    p5 = swarm_result.get("phase5_pm")     or {}
    # FIX (2026-07): phase5 picks are nested under phase5_pm.horizons.core, NOT
    # directly on phase5_pm. The old code read p5.get("long_stocks") (always None),
    # so phase5_picks was EMPTY on every snapshot for ~1 month. Read the core bucket.
    p5_core = (p5.get("horizons") or {}).get("core") or {}

    # Don't snapshot if both phases failed
    if not p5_core.get("long_stocks") and not p4.get("long_stocks"):
        return {"appended": False, "reason": "no_picks_in_payload"}

    today = _today()
    snapshot = {
        "date": today,
        "snapshot_at": _now(),
        "source": source,
        "regime_tag": (swarm_result.get("snapshot") or {}).get("regime_tag_deterministic"),
        "synthesis_neutral_regime": (swarm_result.get("synthesis_neutral") or {}).get("regime_tag"),
        "phase4_picks": {
            "long_stocks":  _strip_bucket(p4.get("long_stocks")),
            "long_etfs":    _strip_bucket(p4.get("long_etfs")),
            "short_stocks": _strip_bucket(p4.get("short_stocks")),
            "short_etfs":   _strip_bucket(p4.get("short_etfs")),
        },
        "phase5_picks": {
            "long_stocks":  _strip_bucket(p5_core.get("long_stocks")),
            "long_etfs":    _strip_bucket(p5_core.get("long_etfs")),
            "short_stocks": _strip_bucket(p5_core.get("short_stocks")),
            "short_etfs":   _strip_bucket(p5_core.get("short_etfs")),
        },
        "pm_commentary": (p5.get("pm_commentary") or "")[:1500],
    }

    # ── Faithful PIT record: capture the ACTUAL 매수 Final List + conviction weights ──
    # append_snapshot runs AFTER CACHE_PATH is written (market_leaders_swarm.py), so
    # build_final_lists() reads the fresh cache. This stores exactly what the Portfolio
    # panel holds (ticker + weight + weighting fields), per day — the point-in-time
    # foundation for the weekly-rebalanced backtest.
    try:
        from agents.final_list import build_final_lists
        from agents.portfolio_agent import build_portfolios
        _fl = build_final_lists()
        _buy = list(_fl.get("buy_list") or []) + list(_fl.get("active_positions") or [])
        _seen, _dedup = set(), []
        for _it in _buy:
            _tk = _it.get("ticker")
            if _tk and _tk not in _seen:
                _seen.add(_tk); _dedup.append(_it)
        # QVR lookup — api /api/portfolio 경로와 동일 소스 (STATE['df']).
        # 기존엔 미전달 → 전 종목 중립 50으로 PIT 가중치가 평탄화됐음.
        _qvr_lookup: dict = {}
        try:
            from api import STATE as _ST_Q
            _df_q = _ST_Q.get("df")
            if _df_q is not None and "qvr_score" in getattr(_df_q, "columns", []):
                _qvr_lookup = dict(zip(_df_q["ticker"], _df_q["qvr_score"]))
        except Exception:
            _qvr_lookup = {}
        _pf = build_portfolios(_dedup, _qvr_lookup)
        def _fb(rows):
            return [{"ticker": r.get("ticker"), "weight": r.get("weight"),
                     "composite": r.get("composite"), "tier": r.get("tier"),
                     "sector": r.get("sector"), "classification": r.get("classification"),
                     "risk_score": r.get("risk_score")} for r in (rows or [])]
        snapshot["final_portfolio"] = {"stocks": _fb(_pf.get("stocks")),
                                       "etfs":   _fb(_pf.get("etfs"))}
    except Exception as _fe:
        snapshot["final_portfolio_error"] = str(_fe)[:200]

    history = load_history()
    snaps = history.setdefault("snapshots", [])

    # Dedup: overwrite if same date already exists
    overwritten = False
    for i, s in enumerate(snaps):
        if s.get("date") == today:
            snaps[i] = snapshot
            overwritten = True
            break
    if not overwritten:
        snaps.append(snapshot)

    history["snapshots"] = snaps
    history["last_update"] = _now()
    history["n_snapshots"] = len(snaps)
    HISTORY_PATH.write_text(json.dumps(history, indent=2, ensure_ascii=False),
                            encoding="utf-8")

    return {
        "appended": True,
        "overwritten": overwritten,
        "date": today,
        "n_snapshots": len(snaps),
        "first_date": snaps[0]["date"] if snaps else None,
    }


def summarize_history() -> dict:
    """Aggregate stats — useful for dashboard banner."""
    h = load_history()
    snaps = h.get("snapshots", [])
    if not snaps:
        return {"n_snapshots": 0, "first_date": None, "last_date": None,
                "unique_tickers": 0}

    all_tickers: set[str] = set()
    for s in snaps:
        for bucket in ("phase5_picks", "phase4_picks"):
            for picks in (s.get(bucket) or {}).values():
                for p in picks:
                    if p.get("ticker"):
                        all_tickers.add(p["ticker"])

    # Ticker frequency in phase5_picks
    freq: dict[str, int] = {}
    for s in snaps:
        for picks in (s.get("phase5_picks") or {}).values():
            for p in picks:
                t = p.get("ticker")
                if t:
                    freq[t] = freq.get(t, 0) + 1
    top_freq = sorted(freq.items(), key=lambda x: -x[1])[:10]

    return {
        "n_snapshots": len(snaps),
        "first_date": snaps[0]["date"],
        "last_date":  snaps[-1]["date"],
        "unique_tickers": len(all_tickers),
        "top_persistent_tickers": [{"ticker": t, "count": n, "pct": round(n / len(snaps) * 100, 1)}
                                    for t, n in top_freq],
    }


if __name__ == "__main__":
    # Quick test — append current swarm cache
    cache_path = Path(".market_leaders_swarm_cache.json")
    if cache_path.exists():
        sw = json.loads(cache_path.read_text(encoding="utf-8"))
        out = append_snapshot(sw, source="manual_test")
        print(json.dumps(out, indent=2))
        print()
        print("=== Summary ===")
        print(json.dumps(summarize_history(), indent=2, ensure_ascii=False))
