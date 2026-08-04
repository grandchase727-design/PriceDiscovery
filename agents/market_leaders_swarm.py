# -*- coding: utf-8 -*-
"""market_leaders_swarm.py — 6-agent swarm for Market Leaders narrative.

Phase 1 (4 parallel domain analysts, each with strict lane + ≤2 WebSearch):
  - Macro Analyst        (regime, sector rotation, fiscal context)
  - Cross-Asset Analyst  (yields, credit spreads, DXY, VIX, oil)
  - Sector/Theme Analyst (leadership breadth + transition signals)
  - Flow & Momentum Analyst (strategy net direction, ETF flows)

Phase 2 (1 agent, conditional):
  - Coherence Debater    (cross-check 4 verdicts for contradictions)

Phase 3 (1 agent, dual mode neutral/averse):
  - Synthesis Arbitrator (final Market Leaders verdict)

All LLM via `claude -p` subprocess → user's Max plan. No API key.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Top-level agent module imports (hoisted from in-function imports during Option B refactor)
from agents.per_ticker_debate import (
    run_per_ticker_debate, summarize_debate_results,
)
from agents.portfolio_composer import compose_portfolio, summarize_composition
from agents.pareto_tracker import (
    ParetoFrontTracker, adaptive_convergence_threshold,
)
from agents.fact_collector import (
    run_fact_collector, filter_evidence_for_agent, format_evidence_for_prompt,
)
from agents.position_state import apply_state_machine
from agents.pm_history import append_snapshot, append_trading_snapshot

CACHE_PATH = Path(".market_leaders_swarm_cache.json")
CACHE_TTL_HOURS = 12

# ── Per-Ticker Debate 범위 — 전 horizon × top-15 픽 × R1 only ──
# 3 horizon × top-15 × batch-8 = 6~9 batches; semaphore(2) 병렬 시 ~15-20min.
# R1만으로도 _synthesize_from_r1이 완전한 LLM transcript를 생성하므로 quality 손실 없음.
# R2/R3을 되살리려면 PT_DEBATE_MAX_ROUNDS = 3 으로 변경 (추가 ~10-15min).
PT_DEBATE_HORIZONS = ("core",)
PT_DEBATE_MAX_PICKS = None  # 캡 없음 — PM pool 전체 debate (final-list 픽 전원 LLM transcript)
PT_DEBATE_MAX_ROUNDS = 2   # R1 (Trading+Risk+Critic) + R2 (비판/반론) — R3 off for speed

# 경로 A: Phase 1 analysts synthesize from the Phase 0 shared evidence pool WITHOUT
# re-running their own WebSearch (Phase 0's 20 authoritative queries already cover it).
# Removes redundant per-analyst search latency → phase1 ~7min → ~3min. No structure
# change (still 5 analysts → same phase1 dict), so downstream phases are unaffected.
PHASE1_RELY_ON_PHASE0 = False  # Phase 0 제거 — news_narrative가 직접 WebSearch 수행

# Fan-out concurrency for independent stages (phase1 analysts, pt_debate batches).
# Matches the subprocess semaphore cap (CLAUDE_MAX_CONCURRENCY, default 3) so the
# ThreadPoolExecutor submits enough work to saturate the allowed concurrency.
import os as _os
try:
    _SWARM_FANOUT_WORKERS = max(1, int(_os.environ.get("CLAUDE_MAX_CONCURRENCY", "1")))
except (TypeError, ValueError):
    _SWARM_FANOUT_WORKERS = 1


# ─────────────────────────────────────────────────────────────────────
# Snapshot extractor — mirror of frontend computeMarketLeaders()
# ─────────────────────────────────────────────────────────────────────

CYCLICAL_SECTORS = {"Technology","Communication Services","Consumer Discretionary",
                    "Industrials","Materials","Energy","Financials","Real Estate"}
DEFENSIVE_SECTORS = {"Consumer Staples","Utilities","Healthcare"}
GROWTH_SECTORS = {"Technology","Communication Services","Consumer Discretionary"}
VALUE_SECTORS = {"Financials","Energy","Materials","Utilities","Real Estate","Consumer Staples"}

_BULLISH = {"CONTINUATION","FORMATION","RECOVERY","OVEREXTENDED","LAGGING_CATCHUP","PULLBACK"}
_BEARISH = {"DOWNTREND","WEAKENING","FADING","EXHAUSTING","CYCLE_PEAK","COUNTER_RALLY"}


def _clean(cls: str) -> str:
    return re.sub(r"[^\x00-\x7F]+", "", cls or "").strip()


def _aggregate_group(sector_rows: list[dict], sectors: set[str]) -> dict:
    rows = [s for s in sector_rows if s.get("sec") in sectors]
    total_n = sum(s.get("total", 0) for s in rows)
    sum_comp = sum(s.get("avgComp", 0) * s.get("total", 0) for s in rows)
    n_comp   = total_n
    sum_1m   = sum(s.get("avg1m", 0) * s.get("n1m", 0) for s in rows)
    n_1m     = sum(s.get("n1m", 0) for s in rows)
    sum_mom  = sum(s.get("mom", 0) for s in rows)
    return {
        "n": total_n,
        "avg_comp": (sum_comp / n_comp) if n_comp > 0 else 0,
        "avg_1m":   (sum_1m / n_1m)     if n_1m > 0   else 0,
        "mom_pct":  (sum_mom / total_n * 100) if total_n > 0 else 0,
        "sectors":  [s.get("sec") for s in rows],
    }


def build_snapshot() -> dict:
    """Pull live data from STATE + quant_strategies and shape for agents."""
    from api import STATE
    results = STATE.get("results") or []
    regime  = STATE.get("regime") or {}
    if not results:
        return {"error": "scan_cache empty"}

    # Sector rows: aggregate per Level-1 sector (fallback to cleaned category)
    def _sec_of(r: dict) -> str:
        s = r.get("sector")
        if s:
            return s
        c = r.get("category") or "Other"
        if isinstance(c, str) and c.startswith(("STK_", "EQ_", "FI_", "MA_", "ETF_")):
            c = c.split("_", 1)[1]
        return c

    sec_acc: dict[str, dict] = {}
    for r in results:
        sec = _sec_of(r)
        d = sec_acc.setdefault(sec, {"sec": sec, "total": 0, "avgComp": 0, "_comp_sum": 0,
                                     "avg1m": 0, "n1m": 0, "_1m_sum": 0, "mom": 0, "bullish": 0, "bearish": 0})
        d["total"] += 1
        comp = float(r.get("composite") or 0)
        d["_comp_sum"] += comp
        ret1m = r.get("return_1m")
        if ret1m is not None:
            d["_1m_sum"] += float(ret1m); d["n1m"] += 1
        cls = _clean(r.get("classification") or "")
        if cls in _BULLISH:
            d["mom"] += 1; d["bullish"] += 1
        elif cls in _BEARISH:
            d["bearish"] += 1
    for d in sec_acc.values():
        d["avgComp"] = d["_comp_sum"] / d["total"] if d["total"] else 0
        d["avg1m"]   = d["_1m_sum"] / d["n1m"] if d["n1m"] else 0
        d["pct_bullish"] = d["bullish"] / d["total"] * 100 if d["total"] else 0
        d["pct_bearish"] = d["bearish"] / d["total"] * 100 if d["total"] else 0
    sector_rows = sorted(sec_acc.values(), key=lambda x: -x["pct_bullish"])

    # Aggregate cyclical/defensive/growth/value
    cyc = _aggregate_group(sector_rows, CYCLICAL_SECTORS)
    dfn = _aggregate_group(sector_rows, DEFENSIVE_SECTORS)
    gro = _aggregate_group(sector_rows, GROWTH_SECTORS)
    val = _aggregate_group(sector_rows, VALUE_SECTORS)
    cd_gap = cyc["avg_comp"] - dfn["avg_comp"]
    cd_1m  = cyc["avg_1m"] - dfn["avg_1m"]
    gv_gap = gro["avg_comp"] - val["avg_comp"]
    gv_1m  = gro["avg_1m"] - val["avg_1m"]

    # Regime tag (same logic as TS)
    if cd_gap > 5 and gv_gap > 5:        tag = "Risk-On / Pro-Growth"
    elif cd_gap > 5 and gv_gap < -5:     tag = "Reflation / Late-Cycle"
    elif cd_gap < -5 and gv_gap > 5:     tag = "Mixed / Defensive Growth"
    elif cd_gap < -5 and gv_gap < -5:    tag = "Risk-Off / Bear"
    elif abs(cd_gap) <= 5 and abs(gv_gap) <= 5: tag = "Neutral"
    else: tag = "Transitional"

    # OER avg
    oers = [float(r.get("oer") or 0) for r in results if r.get("oer") is not None]
    oer_avg = sum(oers) / len(oers) if oers else 0

    # Top CONTINUATION leaders
    cont_rows = [r for r in results if _clean(r.get("classification") or "") == "CONTINUATION"]
    cont_rows.sort(key=lambda r: -float(r.get("composite") or 0))
    top_cont = [{
        "ticker": r.get("ticker"), "name": r.get("name", ""),
        "composite": float(r.get("composite") or 0),
        "ret_1m": r.get("return_1m"), "sector": r.get("sector"),
    } for r in cont_rows[:5]]

    # ── Candidate pools for Phase 4 (action selector) ──
    def _is_etf(r: dict) -> bool:
        cat = r.get("category") or ""
        return not (isinstance(cat, str) and cat.startswith("STK_"))

    def _candidate(r: dict, source: str = "momentum") -> dict:
        return {
            "ticker": r.get("ticker"), "name": (r.get("name") or "")[:60],
            "composite": round(float(r.get("composite") or 0), 1),
            "classification": _clean(r.get("classification") or ""),
            "oer": round(float(r.get("oer") or 0), 1),
            "ret_1m": r.get("return_1m"),
            "sector": _sec_of(r), "industry": r.get("gics_industry_group") or r.get("industry") or "",
            # Entry Timing (진입 적시성) — 강도(composite)와 직교, LLM이 "지금 진입이 늦었나" 참고
            "entry_timing_score": r.get("entry_timing_score"),
            "entry_timing_status": r.get("entry_timing_status"),
            # Pre-Momentum 선행 신호 (③ forming 후보 식별용)
            "pre_momentum_score": round(float(r.get("pre_momentum_score") or 0), 1),
            "pm_agreement_ratio": r.get("pm_agreement_ratio"),
            "pool_source": source,   # "momentum" (확정) | "pre_momentum" (forming/조기)
        }

    LONG_CLS  = {"CONTINUATION", "FORMATION", "LAGGING_CATCHUP", "RECOVERY"}
    SHORT_CLS = {"DOWNTREND", "WEAKENING", "CYCLE_PEAK", "FADING", "EXHAUSTING"}

    # Layer-5 Eligibility Gate 정합: 신규 롱 후보는 composite ≥ 55 + ADV ≥ $5M.
    # 기존에는 bullish 분류 + composite 정렬만으로 풀을 만들어 게이트 미달
    # (MOAT 52.8 등)이 매수 후보로 유입됐음. RECOVERY는 조기 로테이션 포착용으로
    # 클래스는 유지하되 동일 플로어를 적용.
    _GATE_MIN_COMP, _GATE_MIN_ADV = 55.0, 5_000_000

    def _passes_long_gate(r: dict) -> bool:
        if float(r.get("composite") or 0) < _GATE_MIN_COMP:
            return False
        adv = r.get("adv_usd")
        if adv is not None:
            try:
                if float(adv) < _GATE_MIN_ADV:
                    return False
            except (TypeError, ValueError):
                pass
        return True

    long_rows  = [r for r in results
                  if _clean(r.get("classification") or "") in LONG_CLS
                  and _passes_long_gate(r)]
    short_rows = [r for r in results if _clean(r.get("classification") or "") in SHORT_CLS]
    long_rows.sort(key=lambda r: -float(r.get("composite") or 0))
    short_rows.sort(key=lambda r: float(r.get("composite") or 0))   # lowest composite first

    # ── ③ Pre-Momentum forming candidates — 조기 편입 (사전-스트레치 발굴) ──
    # 매수 후보 풀은 composite≥55(=이미 오른) 종목으로 구조적으로 쏠린다. Pre-Momentum
    # 시스템("모멘텀이 어디로 갈까")이 잡은 forming 종목(provisional_eligible 또는 강한
    # pre-mom+agreement)을 별도 풀로 주입 → 스트레치 이전 후보를 매수 슬레이트에 노출.
    # 이미 long_rows(확정 모멘텀)에 있는 종목은 제외, ADV 플로어는 동일 적용.
    _long_tks = {r.get("ticker") for r in long_rows}
    def _is_forming(r: dict) -> bool:
        if r.get("ticker") in _long_tks:
            return False
        adv = r.get("adv_usd")
        try:
            if adv is not None and float(adv) < _GATE_MIN_ADV:
                return False
        except (TypeError, ValueError):
            pass
        if r.get("provisional_eligible"):
            return True
        pm = float(r.get("pre_momentum_score") or 0)
        ar = float(r.get("pm_agreement_ratio") or 0)
        return pm >= 45.0 and ar >= 0.4    # MODERATE+ 선행 합의
    forming_rows = [r for r in results if _is_forming(r)]
    # 선행 신호가 강한 순 (pre_momentum_score desc, agreement 2차)
    forming_rows.sort(key=lambda r: (-float(r.get("pre_momentum_score") or 0),
                                     -float(r.get("pm_agreement_ratio") or 0)))

    # Expanded pool — 35 candidates per cell so the agent has headroom to
    # pick 20 with sufficient sector/regime diversity. 확정 모멘텀 우선(composite 순),
    # 그 뒤 forming 후보를 최대 10개까지 덧붙여(중복 제거) 조기 후보도 검토되게 함.
    def _merge_pool(confirmed: list, forming: list, cap: int = 35, forming_cap: int = 15) -> list:  # forming 10→15 (2026-07-28: 조기포착 가시권 확대)
        out = list(confirmed[:cap])
        seen = {c["ticker"] for c in out}
        for c in forming:
            if len(out) >= cap + forming_cap:
                break
            if c["ticker"] in seen:
                continue
            seen.add(c["ticker"]); out.append(c)
        return out

    long_stocks_pool = _merge_pool(
        [_candidate(r) for r in long_rows if not _is_etf(r)],
        [_candidate(r, "pre_momentum") for r in forming_rows if not _is_etf(r)])
    long_etfs_pool = _merge_pool(
        [_candidate(r) for r in long_rows if _is_etf(r)],
        [_candidate(r, "pre_momentum") for r in forming_rows if _is_etf(r)])
    short_stocks_pool = [_candidate(r) for r in short_rows if not _is_etf(r)][:35]
    short_etfs_pool   = [_candidate(r) for r in short_rows if     _is_etf(r)][:35]

    # ── GICS sectors pool (only the 11 standard sectors for scoring) ──
    GICS_11 = ["Technology","Communication Services","Consumer Discretionary",
               "Industrials","Materials","Energy","Financials","Real Estate",
               "Consumer Staples","Utilities","Healthcare"]
    gics_pool = []
    for sec in GICS_11:
        row = next((s for s in sector_rows if s["sec"] == sec), None)
        if row:
            gics_pool.append({
                "sector": sec, "n": row["total"],
                "pct_bullish": round(row["pct_bullish"], 1),
                "pct_bearish": round(row["pct_bearish"], 1),
                "avg_comp":    round(row["avgComp"], 1),
                "avg_1m":      round(row["avg1m"], 2),
            })

    # ── Themes pool (subthemes from results — use category + industry_group fallback) ──
    theme_acc: dict[str, dict] = {}
    for r in results:
        theme = r.get("subtheme") or r.get("theme") or r.get("category") or "Other"
        if not theme:
            continue
        t = theme_acc.setdefault(theme, {"theme": theme, "total": 0, "_csum": 0,
                                          "mom": 0, "_1m_sum": 0, "n1m": 0})
        t["total"] += 1
        t["_csum"] += float(r.get("composite") or 0)
        cls = _clean(r.get("classification") or "")
        if cls in _BULLISH:
            t["mom"] += 1
        ret1m = r.get("return_1m")
        if ret1m is not None:
            t["_1m_sum"] += float(ret1m); t["n1m"] += 1
    themes = []
    for t in theme_acc.values():
        if t["total"] < 2:
            continue   # require ≥2 tickers per theme
        themes.append({
            "theme": t["theme"], "n": t["total"],
            "mom_pct": round(t["mom"] / t["total"] * 100, 1),
            "avg_comp": round(t["_csum"] / t["total"], 1),
            "avg_1m":   round(t["_1m_sum"] / t["n1m"], 2) if t["n1m"] else 0,
        })
    themes_pool = sorted(themes, key=lambda x: -x["avg_comp"])

    # Classification breakdown
    cls_counts: dict[str, int] = {}
    for r in results:
        c = _clean(r.get("classification") or "")
        cls_counts[c] = cls_counts.get(c, 0) + 1

    # Run quant strategies
    try:
        from quant_strategies import run_quant_strategies
        qs = run_quant_strategies(results)
    except Exception:
        qs = {"strategies": {}, "net_direction": "MIXED"}

    return {
        "as_of": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tickers": len(results),
        "regime_tag": tag,
        "regime_state": regime,
        "cd_gap": round(cd_gap, 1), "cd_1m": round(cd_1m, 1),
        "gv_gap": round(gv_gap, 1), "gv_1m": round(gv_1m, 1),
        "cyclical": {**cyc, "avg_comp": round(cyc["avg_comp"], 1), "avg_1m": round(cyc["avg_1m"], 1)},
        "defensive": {**dfn, "avg_comp": round(dfn["avg_comp"], 1), "avg_1m": round(dfn["avg_1m"], 1)},
        "growth":    {**gro, "avg_comp": round(gro["avg_comp"], 1), "avg_1m": round(gro["avg_1m"], 1)},
        "value":     {**val, "avg_comp": round(val["avg_comp"], 1), "avg_1m": round(val["avg_1m"], 1)},
        "oer_avg": round(oer_avg, 1),
        "top_continuation": top_cont,
        "classification_counts": cls_counts,
        "sector_breadth": [{
            "sector": s["sec"], "n": s["total"],
            "pct_bullish": round(s["pct_bullish"], 1),
            "pct_bearish": round(s["pct_bearish"], 1),
            "avg_comp": round(s["avgComp"], 1),
            "avg_1m": round(s["avg1m"], 1),
        } for s in sector_rows],
        "quant_strategies": qs,
        # ── Phase 4 pools ──
        "long_stocks_pool":  long_stocks_pool,
        "long_etfs_pool":    long_etfs_pool,
        "short_stocks_pool": short_stocks_pool,
        "short_etfs_pool":   short_etfs_pool,
        "gics_sectors":      gics_pool,
        "themes":            themes_pool[:30],
    }


# ─────────────────────────────────────────────────────────────────────
# Prompt builders — strict lane, ≤2 WebSearch, JSON-fenced output
# ─────────────────────────────────────────────────────────────────────

def _fpct(v, default: str = "—") -> str:
    """Safe percentage formatter — returns '—' for None/non-finite."""
    if v is None:
        return default
    try:
        f = float(v)
        if not math.isfinite(f):
            return default
        return f"{f:+.1f}%"
    except Exception:
        return default


def _fnum(v, fmt: str = "{:.1f}", default: str = "—") -> str:
    if v is None:
        return default
    try:
        f = float(v)
        if not math.isfinite(f):
            return default
        return fmt.format(f)
    except Exception:
        return default


_OUTPUT_RULES = """
Return STRICTLY a fenced ```json block, nothing else. No prose before or after.

★★ WEBSEARCH 강제 (★★ MANDATORY ★★):
You MUST execute 2-3 WebSearch queries to ground your analysis in CURRENT data.
Do NOT rely on training data for: rates, central bank stances, geopolitical events,
recent earnings, current ETF flows, or any time-sensitive macro signal.
Your output MUST include a "websearch_results" field with the actual queries you ran
AND the URLs/snippets you found. Empty array = your response will be rejected and
re-run.
Confidence is 0.0-1.0.

★★ LANGUAGE — ALL human-facing commentary text MUST be in KOREAN (한국어) ★★
Applies to all narrative/rationale/commentary/thesis/reason fields. Examples:
  - "narrative" / "commentary" / "rationale" / "portfolio_thesis"
  - "pm_commentary" / "biggest_risk" / "biggest_opportunity" / "key_signals"
  - "entry_trigger" / "change_reason" / per-pick "rationale" strings
DO KEEP IN ENGLISH (case-sensitive identifiers / fixed tokens):
  - JSON keys, enum values like "BUY_NOW", "WAIT", "SKIP", "BULLISH", "BEARISH"
  - ticker symbols (AAPL, MSFT, 005930.KS), classification tags (CONTINUATION, etc.)
  - rating values (BUY/HOLD/SELL/STRONG_BUY etc.) — keep as defined
Free-text human commentary inside string values → 한국어로 작성."""

_PHASE1_SCHEMA = """
{"agent":"<your_agent_id>","rating":"<one of allowed ratings>","confidence":0.0-1.0,
"confidence_factors":{
  "data_freshness":0.0-1.0,    // 1.0 = data from today/this week; 0.5 = month-old; 0.0 = stale
  "signal_clarity":0.0-1.0,    // 1.0 = unanimous indicators; 0.5 = mixed; 0.0 = contradictory
  "cross_source_agreement":0.0-1.0 // 1.0 = ≥3 corroborating sources; 0.5 = 2 sources; 0.0 = single
},
"narrative":"2-3 sentence summary of your domain's leadership signal (한국어, must reflect websearch_results)",
"key_signals":["3-5 bullet observations — each citing websearch source where applicable"],
"biggest_risk":"single biggest risk to your read",
"biggest_opportunity":"single biggest opportunity",
"websearch_queries":["queries you actually ran"],
"websearch_results":[
  {"query":"q1 you ran","url":"https://...","snippet":"≤200자 인용 (검색 결과 핵심)","retrieved_at":"YYYY-MM-DD"},
  {"query":"q2 you ran","url":"https://...","snippet":"..."}
]}

CONFIDENCE CALIBRATION RULES (H3 fix — was uniformly 0.72 across agents):
- confidence MUST be derived from confidence_factors:
  confidence = round(0.4 * data_freshness + 0.3 * signal_clarity + 0.3 * cross_source_agreement, 2)
- Different agents see different data; DO NOT all default to 0.72.
- Range guidance:
  • 0.85-0.95: today's web data + ≥3 unanimous sources + clear trend
  • 0.70-0.85: web data + 2 sources + mostly clear
  • 0.50-0.70: mixed signals OR single source OR stale data
  • 0.30-0.50: training-data only OR contradictory signals
  • 0.00-0.30: WEBSEARCH_UNAVAILABLE / agent failure"""


def _macro_prompt(snap: dict) -> str:
    return f"""You are the MACRO ANALYST in a market leadership swarm — GLOBAL macroeconomics.

YOUR STRICT LANE:
- Economic cycle phase, sector rotation, and the GLOBAL macro backdrop:
  • Central-bank POLICY RATES & stance (decisions, not market yields): Fed, ECB,
    Bank of England (BOE), Bank of Japan (BOJ), Bank of Korea (BOK), Brazil (Selic/BCB),
    Reserve Bank of Australia (RBA), Reserve Bank of India (RBI).
  • Activity — PMI: US ISM + S&P Global (global) + China (Caixin/NBS) manufacturing & services.
  • Growth / prices / labor (US AND non-US): GDP, inflation (CPI), employment/unemployment
    across the US, Eurozone, Japan, UK, Korea, China, India.
  • Fiscal & external balances: government budget / fiscal balance, trade balance, exports/imports.
    Also: US Fed Beige Book (regional activity / pricing / labor anecdotes) and Korea Customs
    Service (관세청) monthly + 10-day export/import trends (early Korea/Asia trade read).
- You may NOT discuss MARKET-PRICED cross-asset signals: yield curves (10Y/2Y), credit
  spreads (OAS), DXY/FX levels, VIX, commodity prices, ETF flows — those are the
  Cross-Asset / Flow analysts' lanes. (Central-bank POLICY rate is yours; market YIELDS are theirs.)

LIVE DATA SNAPSHOT (as of {snap['as_of']}, {snap['total_tickers']} tickers):
- System regime tag (deterministic): {snap['regime_tag']}
- Cyclical avg Comp {snap['cyclical']['avg_comp']} (n={snap['cyclical']['n']}, 1M {_fpct(snap['cyclical']['avg_1m'])}) vs Defensive avg Comp {snap['defensive']['avg_comp']} (n={snap['defensive']['n']}, 1M {_fpct(snap['defensive']['avg_1m'])})
  → CD gap: Comp {_fnum(snap['cd_gap'], '{:+.1f}')}, 1M {_fnum(snap['cd_1m'], '{:+.1f}pp')}
- Growth avg Comp {snap['growth']['avg_comp']} (n={snap['growth']['n']}, 1M {_fpct(snap['growth']['avg_1m'])}) vs Value avg Comp {snap['value']['avg_comp']} (n={snap['value']['n']}, 1M {_fpct(snap['value']['avg_1m'])})
  → GV gap: Comp {_fnum(snap['gv_gap'], '{:+.1f}')}, 1M {_fnum(snap['gv_1m'], '{:+.1f}pp')}
- OER average across universe: {_fnum(snap['oer_avg'])} (high=overheated, low=cool)
- Sector regime: cyclical_dom={snap['regime_state'].get('cyclical_dom')}, defensive_dom={snap['regime_state'].get('defensive_dom')}, growth_dom={snap['regime_state'].get('growth_dom')}, value_dom={snap['regime_state'].get('value_dom')}

PRIMARY VERIFICATION SOURCES (anchor EVERY data point — anti-hallucination):
- Trading Economics (tradingeconomics.com): canonical for all-country policy rates, PMI,
  GDP, CPI, unemployment, budget/fiscal balance, trade balance.
- FRED (fred.stlouisfed.org): official US series; US Fed Beige Book (federalreserve.gov). OECD
  (oecd.org): cross-country GDP/CPI/jobs. Korea Customs Service (customs.go.kr) for KR export/import.
- S&P Global (pmi.spglobal.com): global PMI. Official central banks for stance/decisions:
  ecb.europa.eu, boj.or.jp, bankofengland.co.uk, bok.or.kr, bcb.gov.br. CNBC for narrative.
- Phase 0 evidence pool (if injected above) already pulled these — PREFER those numbers;
  only re-search for gaps.

⚠ ANTI-HALLUCINATION: never state a central-bank rate/stance or a macro print (PMI, CPI,
  GDP, unemployment, fiscal/trade balance) WITHOUT a Phase-0 number or a WebSearch in this
  run. If unverifiable, omit it rather than rely on training-data heuristics.

YOUR TASK:
1. Use ≤3 WebSearch queries (only to fill gaps not already in the Phase 0 pool), anchored
   to the sources above. Examples:
   - "Fed ECB BOE BOJ BOK RBA RBI Brazil Selic policy rate decision {snap['as_of']} site:tradingeconomics.com"
   - "US S&P Global China Caixin PMI, US Eurozone Japan Korea GDP inflation unemployment {snap['as_of']} site:tradingeconomics.com OR site:oecd.org"
   - "US Eurozone China fiscal balance trade balance, US Fed Beige Book, Korea customs export import trend {snap['as_of']} site:tradingeconomics.com OR site:federalreserve.gov"
2. Interpret system signals through this GLOBAL macro lens (synchronized easing/tightening?
   global growth re-accel or slowdown? fiscal/trade stress?).
3. Output verdict in JSON. COVERAGE: do NOT over-index on US ISM manufacturing PMI — span ≥4
   indicator families (rates / manufacturing+services PMI / GDP+CPI+jobs / fiscal+trade) and
   the US AND ≥3 non-US regions; use 6-9 key_signals with ≥3 non-US datapoints, each citing a
   specific verified number (e.g. "ECB deposit 2.00% held 2026-06-05", "China Caixin mfg PMI
   50.8", "Eurozone CPI 2.1% YoY", "Korea 관세청 수출 +4.2% YoY").

Allowed ratings: RISK_ON | PRO_GROWTH | REFLATION | LATE_CYCLE | DEFENSIVE | RISK_OFF | MIXED | TRANSITIONAL

OUTPUT SCHEMA:
```json
{_PHASE1_SCHEMA.replace('<your_agent_id>', 'macro_analyst')}
```
{_OUTPUT_RULES}"""


def _cross_asset_prompt(snap: dict) -> str:
    return f"""You are the CROSS-ASSET ANALYST in a 4-agent market leadership swarm.

YOUR STRICT LANE — MARKET-PRICED cross-asset signals, GLOBAL coverage (NOT US-only):
- US: yield curve (UST 10Y/2Y/3M + 10Y-2Y spread, inverted?), credit spreads (IG/HY OAS), DXY, VIX
- Japan: JGB 10Y yield + USD/JPY (BOJ stance only as it drives the yield/FX)
- Europe: Bund 10Y yield + EUR/USD (ECB stance as yield/FX driver)
- UK: Gilt 10Y yield + GBP/USD (BOE stance as yield/FX driver)
- Korea: KTB yield + USD/KRW;  Brazil: USD/BRL (high-carry EM FX signal)
- Australia: RBA policy stance + AUD/USD (China-demand-proxy commodity-currency signal —
  AUD often leads/confirms industrial-metal and China-growth narratives)
- FX PAIRS TO ALWAYS COVER (four core pairs the swarm depends on): USD/JPY, EUR/USD, USD/KRW,
  AUD/USD — report level + direction (strengthening/weakening) for each when verifiable.
- Commodities: crude oil (WTI/Brent), copper, gold cross-signals
- DELIVERABLES the swarm depends on: (a) yield-curve shape, (b) credit-spread level/direction,
  (c) DXY level/direction, (d) oil level/direction, (e) VIX, (f) USD/JPY, EUR/USD, USD/KRW,
  AUD/USD levels/direction. Always report these 6 when verified.
- LANE SPLIT: central-bank POLICY-RATE decisions/levels belong to the MACRO analyst; you cover
  the MARKET YIELDS/FX/commodities those decisions move. Reference stance only to explain yields.
- You may NOT discuss: equity sector rotation, fund flows, quant strategy direction.
- Your job: tell the swarm if GLOBAL cross-asset signals AGREE with the equity regime tag.

⚠ ANTI-HALLUCINATION RULE (강제):
- NEVER state foreign central bank stance ("BOJ dovish", "ECB hawkish", "BOK 인하", etc.)
  WITHOUT verifying via a WebSearch query in this run.
- If you cannot verify a central bank's CURRENT stance, omit it from your output rather
  than fall back on training-data heuristics (e.g. "BOJ has historically been dovish").
- Downstream agents (PM, Trading, Risk) will use your output AS THE ONLY SOURCE of foreign
  rate/FX information. Stale or fabricated central bank narratives propagate as portfolio
  errors (e.g. recommending DXJ "because BOJ dovish" when BOJ has actually pivoted).

REFERENCE — current equity regime (for cross-check only, NOT your conclusion):
- System regime tag: {snap['regime_tag']}
- OER avg: {_fnum(snap['oer_avg'])}

PRIMARY VERIFICATION SOURCES (use these to ANCHOR every rate/yield number — anti-hallucination):
- Trading Economics (tradingeconomics.com): canonical for global policy rates, JGB/Bund/Treasury
  yields, FX, commodities. Append "site:tradingeconomics.com" to queries when verifying levels.
- FRED (fred.stlouisfed.org): official Fed data — Treasury yields, credit spreads, DXY, money supply.
- Phase 0 evidence pool (above, if injected) already pulled these — prefer those numbers; only
  re-search for gaps.

YOUR TASK:
1. Use 2-3 WebSearch queries to cover GLOBAL central bank + rate + FX landscape, anchored to
   the primary sources above:
   - "10-year Treasury yield VIX credit spreads DXY today site:tradingeconomics.com OR site:fred.stlouisfed.org"
   - "USD/JPY EUR/USD USD/KRW AUD/USD exchange rate today site:tradingeconomics.com OR site:xe.com"
   - "BOJ policy rate JGB yield Japan yen latest decision site:tradingeconomics.com"
   - "ECB deposit rate Bund yield euro latest decision site:tradingeconomics.com"  (or BOK Korea rate KRW / RBA Australia rate AUD if Asia-Pacific focus)
2. Synthesize: are global rates moving SAME direction (synchronized hike/cut) or DIVERGING?
   - This synchronization signal is CRITICAL for the PM Agent's regional ETF decisions.
3. Interpret as Risk-On/Risk-Off + flag central bank divergence.
4. Score how WELL cross-asset matches the equity regime tag ({snap['regime_tag']}).

key_signals MUST include (when verified):
- US rates + credit + DXY + VIX (as before)
- USD/JPY, EUR/USD, USD/KRW, AUD/USD — level + direction for EACH of these 4 pairs
  (e.g. "USD/JPY 148.2, yen weakening on BOJ-Fed policy gap")
- ≥1 explicit Japan/Europe/Korea/Australia signal (e.g. "BOJ 0.50% maintained, no hawkish
  shift as of YYYY-MM-DD")
- Global central bank synchronization assessment (synchronized vs divergent)
- CNN Fear & Greed score + direction (if available from Phase 0 evidence pool):
  • current_score (0-100) + label
  • Direction vs 1-week-ago (improving/deteriorating)
  • Note divergences with VIX (e.g. F&G "greed" while VIX rising = warning signal)

Allowed ratings: CONFIRMS_RISK_ON | CONFIRMS_RISK_OFF | DIVERGES_FROM_EQUITY | MIXED | TRANSITIONAL

OUTPUT SCHEMA:
```json
{_PHASE1_SCHEMA.replace('<your_agent_id>', 'cross_asset_analyst')}
```
{_OUTPUT_RULES}"""


def _sector_theme_prompt(snap: dict) -> str:
    top_bull = snap['sector_breadth'][:5]
    top_bear = sorted(snap['sector_breadth'], key=lambda s: -s['pct_bearish'])[:3]
    top_cont = snap['top_continuation']
    cls = snap['classification_counts']
    return f"""You are the SECTOR/THEME ANALYST in a 4-agent market leadership swarm.

YOUR STRICT LANE:
- Sector leadership breadth, theme breadth, classification distribution
  (CONTINUATION/FORMATION/LAGGING_CATCHUP/RECOVERY vs WEAKENING/DOWNTREND/FADING).
- You may NOT discuss: macro regime, yields, ETF AUM flows.
- Focus: who is leading, who is rolling, how concentrated is leadership.

LIVE DATA:
- Top 5 sectors by bullish %: {", ".join(f"{s['sector']} ({s['pct_bullish']}%, Comp {s['avg_comp']})" for s in top_bull)}
- Top 3 sectors by bearish %: {", ".join(f"{s['sector']} ({s['pct_bearish']}%)" for s in top_bear)}
- Top CONTINUATION leaders: {", ".join(f"{r['ticker']} (Comp {r['composite']:.0f}, 1M {_fpct(r.get('ret_1m'))})" for r in top_cont[:5])}
- Classification breakdown: CONTINUATION={cls.get('CONTINUATION',0)}, FORMATION={cls.get('FORMATION',0)}, LAGGING_CATCHUP={cls.get('LAGGING_CATCHUP',0)}, RECOVERY={cls.get('RECOVERY',0)}, OVEREXTENDED={cls.get('OVEREXTENDED',0)} | WEAKENING={cls.get('WEAKENING',0)}, DOWNTREND={cls.get('DOWNTREND',0)}, FADING={cls.get('FADING',0)}

YOUR TASK:
1. Use ≤2 WebSearch queries for THEME context (e.g., "AI semiconductor leadership 2026", "energy sector rotation").
2. Judge if leadership is BROAD (many sectors participating) or NARROW (few mega-caps).
3. Identify any LEADERSHIP TRANSITIONS (sector moving from CONTINUATION to FADING, or vice versa).

Allowed ratings: BROAD_LEADERSHIP | NARROW_LEADERSHIP | ROTATION_IN_PROGRESS | LEADERSHIP_DECAY | EMERGING_LEADERSHIP | MIXED

OUTPUT SCHEMA:
```json
{_PHASE1_SCHEMA.replace('<your_agent_id>', 'sector_theme_analyst')}
```
{_OUTPUT_RULES}"""


def _flow_momentum_prompt(snap: dict) -> str:
    qs = snap['quant_strategies'].get('strategies', {})
    qs_summary = [f"{k}: {v.get('summary','—')[:100]}" for k, v in list(qs.items())[:6]]
    net = snap['quant_strategies'].get('net_direction', 'MIXED')
    return f"""You are the FLOW & MOMENTUM ANALYST in a 4-agent market leadership swarm.

YOUR STRICT LANE:
- Quant strategy net direction, momentum acceleration/deceleration, ETF flow proxies,
  factor leadership (momentum vs value vs quality).
- You may NOT discuss: macro regime tags, individual sector narratives, yield curves.
- Focus: where is CAPITAL moving, and is momentum accelerating or stalling.

LIVE DATA:
- Quant strategies net direction (6 strategies aggregated): {net}
- OER avg: {_fnum(snap['oer_avg'])} (>50 = leadership getting stretched, <35 = cool/early)
- Cyclical 1M: {_fpct(snap['cyclical']['avg_1m'])} (n={snap['cyclical']['n']}) vs Defensive 1M: {_fpct(snap['defensive']['avg_1m'])} (n={snap['defensive']['n']})
- Growth 1M: {_fpct(snap['growth']['avg_1m'])} (n={snap['growth']['n']}) vs Value 1M: {_fpct(snap['value']['avg_1m'])} (n={snap['value']['n']})
- Top quant strategies output:
{chr(10).join('  • ' + s for s in qs_summary)}

YOUR TASK:
1. Use ≤2 WebSearch queries for ETF FLOW data (e.g., "SPY QQQ inflow May 2026", "factor ETF rotation").
2. Compare WHAT flows say vs WHAT the system regime tag says.
3. Score momentum strength (accelerating/stalling/reversing).
4. If CNN Fear & Greed available from Phase 0 evidence pool, use its subcomponents:
   - momentum subcomponent → confirms or contradicts your momentum read
   - put/call ratio → option flow signal (defensive vs offensive positioning)
   - junk bond demand → risk appetite proxy
   - safe haven demand → flight-to-quality indicator
   Cite F&G score + relevant subcomponent in key_signals when used.

Allowed ratings: ACCELERATING_LEADERSHIP | STALLING_LEADERSHIP | ROTATING_FLOWS | DECAYING_FLOWS | RISK_OFF_FLOWS | MIXED

OUTPUT SCHEMA:
```json
{_PHASE1_SCHEMA.replace('<your_agent_id>', 'flow_momentum_analyst')}
```
{_OUTPUT_RULES}"""


def _news_narrative_prompt(snap: dict) -> str:
    return f"""You are the NEWS NARRATIVE ANALYST in a 5-agent Phase 1.

YOUR STRICT LANE:
- DOMINANT MARKET NARRATIVES from financial news flow last 24-48h.
- Emerging vs fading themes. Sentiment polarity shifts (greed → fear, complacency → panic).
- You may NOT discuss: macro data (PMI/jobs/CPI), cross-asset prices (VIX/yields/credit),
  sector breadth %, ETF flows. Those are other agents' lanes.
- Focus: WHAT IS THE MARKET TALKING ABOUT TODAY, and how is sentiment shifting?

REQUIRED SOURCES (use WebFetch first, fall back to WebSearch if dynamic):
1. Yahoo Finance home: https://finance.yahoo.com/
2. Finviz News feed: https://finviz.com/news
   (US-focused real-time headlines + analyst chatter; complements Yahoo by
    surfacing wire-service stories, pre-market movers, and earnings flow)
3. CNN Fear & Greed Index: https://edition.cnn.com/markets/fear-and-greed
   (quantitative sentiment 0-100 across 7 indicators — VIX, momentum, breadth,
    put/call, junk bond demand, safe haven demand, stock price strength.
    Score interpretation: 0-25=Extreme Fear, 25-45=Fear, 45-55=Neutral,
    55-75=Greed, 75-100=Extreme Greed. Also compare to 1-week/1-month ago
    to detect SENTIMENT REGIME SHIFT.)
4. CNBC Markets: https://www.cnbc.com/markets/
   (US 시장 헤드라인 허브 + live blog — wire-service breaking news, index moves,
    pre-market/after-hours movers. RSS fallback: https://www.cnbc.com/id/100003114/device/rss/rss.html)
5. Google News finance topic (Korea): https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR%3Ako
6. KED Global (Korea Economic Daily, 영문): https://www.kedglobal.com/
   (한국 시장 기관급 분석 — KOSPI/KOSDAQ, 삼성·SK하이닉스 반도체, ETF 규제,
    MSCI 분류. Google News보다 심층적인 한국 시장 narrative)
7. WebSearch fallback: "biggest market story today" + current date, or theme-specific queries.
   직접 fetch 어려운 wire(Reuters/AP/MarketWatch/Bloomberg)는 "site:reuters.com 시장 헤드라인 {snap['as_of']}" 형태 검색으로 보강.

YOUR PROCESS:
Step 1: Pull top headlines + Fear & Greed score from sources
        (≤6 fetch/search ops total — fetch each of the required sources first,
         then 1 optional WebSearch for gap-fill).
Step 2: Cross-check headlines across sources to identify the DOMINANT NARRATIVE
        (a story that appears in ≥2 sources is more credible than single-source).
        Output 1-line narrative + 3-5 supporting headlines (cite source per headline).
Step 3: Note any EMERGING vs FADING narratives (theme picking up coverage vs
        receding from top headlines).
Step 4: Assess sentiment polarity shift (last 24-48h) — combine qualitative
        (headlines) + QUANTITATIVE (CNN F&G score now vs 1wk/1mo ago).
        Examples: "F&G 32→58 in 1mo + headlines shifting from recession to AI →
        clear greed regime emerging" or "F&G 65→42 in 2wk + headlines spike
        in 'sell-off' mentions → fear taking hold".
Step 5: Issue your rating, including current F&G score in key_signals.

REFERENCE — current equity regime tag (for cross-check, NOT your conclusion):
- System regime tag: {snap['regime_tag']}
- Total tickers scanned: {snap['total_tickers']}

Allowed ratings:
- NARRATIVE_RISK_ON      — broad bullish narratives, low-fear headlines, growth story dominant
- NARRATIVE_RISK_OFF     — fear/recession/credit-stress narratives dominant
- NARRATIVE_ROTATION     — leadership rotation narrative (out of X into Y), no clear risk regime
- NARRATIVE_AMBIGUOUS    — mixed/conflicting narratives, no dominant story
- NARRATIVE_BLOWOFF      — euphoria/FOMO narratives (AI bubble talk, melt-up)
- NARRATIVE_CAPITULATION — panic/forced-selling narratives

OUTPUT SCHEMA:
```json
{_PHASE1_SCHEMA.replace('<your_agent_id>', 'news_narrative_analyst')}
```
Key_signals should be ACTUAL headlines or narrative summaries you observed (3-5 bullet items).

ADDITIONAL FIELD (append to the JSON object above, top-level key "emerging_tickers"):
헤드라인에서 관찰된, 이벤트로 급부상 중인 개별 종목 티커 0-5개를 추출하라(없으면 빈 배열 []).
특히 '유니버스에 아직 없을 법한 신흥/테마 수혜주'(예: 네오클라우드·신규상장·급등주)를 우선.
각 항목: {{"ticker":"NBIS","company":"Nebius","theme":"neocloud","catalyst":"1줄 촉매"}}.
반드시 헤드라인에 실제 등장한 티커만. 추측/환각 금지. 이 필드는 유니버스 확장 후보 발굴용.
{_OUTPUT_RULES}"""


# ── Unified Phase 1 prompt — all 5 analysts in ONE LLM call ──────────
def _unified_analyst_prompt(snap: dict, facts: dict, asof: str) -> str:
    """Combine all 5 Phase 1 domain analysts into a single structured LLM call.

    The model returns ONE JSON object with keys:
      "macro", "cross_asset", "sector_theme", "flow_momentum", "news_narrative"
    Each value matches _PHASE1_SCHEMA with the appropriate agent id.

    Reduces Phase 1 from 5 parallel subprocess calls (~4 min) to 1 call (~1-2 min).
    The Phase 0 evidence pool (if available) is injected once and used by all sections.
    Each section specifies 1-2 targeted WebSearch queries for its domain.
    """
    # Build the Phase 0 evidence block once (shared across all sections)
    ev_block = ""
    if facts.get("evidence_pool"):
        from agents.fact_collector import format_evidence_for_prompt
        ev_block = format_evidence_for_prompt(facts["evidence_pool"])

    # search_mode: 2026-07-05 — SPEED MODE(WebSearch 전면 금지) 폐지. 4개 섹션 전부
    # targeted WebSearch로 실시간 데이터를 검증한다 (각 섹션 본문에 구체 쿼리 예시 명시).
    search_mode = (
        "🌐 WEBSEARCH ENABLED (전 섹션): 아래 4개 섹션 모두 각자 lane에 맞는 targeted "
        "WebSearch 2-3회로 실시간 수치/뉴스를 검증하고, 실제 실행한 쿼리·결과를 "
        "websearch_queries / websearch_results 에 기록할 것 (websearch_results를 빈 배열로 "
        "남겨두지 말 것 — 각 섹션 최소 1회 이상 실행). 아래 LIVE DATA 수치는 시스템 내부 신호이니 "
        "웹서치로 검증한 외부 데이터와 교차 확인해 narrative에 반영한다. "
        "news_narrative 섹션은 이 call에서 생략 — 별도 call에서 처리됨."
    )

    # Snapshot snippets reused by multiple sections
    top_bull = snap['sector_breadth'][:5]
    top_bear = sorted(snap['sector_breadth'], key=lambda s: -s['pct_bearish'])[:3]
    top_cont = snap['top_continuation']
    cls = snap['classification_counts']
    qs = snap['quant_strategies'].get('strategies', {})
    qs_summary = [f"{k}: {v.get('summary','—')[:100]}" for k, v in list(qs.items())[:6]]
    net = snap['quant_strategies'].get('net_direction', 'MIXED')

    ev_section = f"\n\n{ev_block}\n\n" if ev_block else ""

    return f"""You are a UNIFIED MARKET ANALYST running 4 domain analyses in a single pass.
Produce ONE JSON object with exactly 4 keys: "macro", "cross_asset", "sector_theme",
"flow_momentum". Each value must match the per-section schema below.
(news_narrative는 별도 call에서 처리 — 이 call에서 생략)

{search_mode}
{ev_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## MACRO ANALYST SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANE: GLOBAL macroeconomics — central-bank POLICY RATES (Fed, ECB, BOE, BOJ, BOK, RBA,
RBI, Brazil/BCB), activity (manufacturing AND services PMI: US ISM + S&P Global global +
China Caixin/NBS), growth/prices/labor (US AND non-US GDP, CPI, employment: Eurozone, Japan,
UK, Korea, China, India), fiscal/external balances (budget, trade balance, exports/imports;
incl. US Fed Beige Book + Korea Customs 관세청 export/import trend). NOT market-priced
yields/FX/VIX (those belong to cross_asset).

LIVE DATA (as of {snap['as_of']}, {snap['total_tickers']} tickers):
- System regime tag: {snap['regime_tag']}
- Cyclical Comp {snap['cyclical']['avg_comp']} vs Defensive Comp {snap['defensive']['avg_comp']} → CD gap: {_fnum(snap['cd_gap'], '{:+.1f}')}
- Growth Comp {snap['growth']['avg_comp']} vs Value Comp {snap['value']['avg_comp']} → GV gap: {_fnum(snap['gv_gap'], '{:+.1f}')}
- OER avg: {_fnum(snap['oer_avg'])}
- Regime state: cyclical_dom={snap['regime_state'].get('cyclical_dom')}, defensive_dom={snap['regime_state'].get('defensive_dom')}

★ COVERAGE (avoid over-indexing on any single US print — in particular do NOT let US ISM
manufacturing PMI dominate the read): give BALANCED depth to ≥4 indicator families —
(1) policy rates/stance, (2) PMI manufacturing AND services, (3) growth+inflation+labor
(GDP/CPI/jobs), (4) fiscal/trade (incl. Fed Beige Book, Korea customs 관세청).
REGIONAL BREADTH: cover the US AND ≥3 non-US regions (choose among Eurozone, Japan, UK, Korea,
China, India, Australia, Brazil), EACH with a SPECIFIC verified number — key_signals MUST
contain ≥3 non-US datapoints. Interpret synchronized vs divergent global easing/tightening and
global growth re-accel vs slowdown. For THIS section use 6-9 key_signals (more than the 3-5
default) to fit the required breadth; narrative should name ≥2 non-US regions explicitly.

WebSearch REQUIRED (this section): run 3-4 targeted queries SPANNING indicators AND regions —
do NOT run US-only PMI/CPI queries. Examples:
- "Fed ECB BOJ BOE BOK RBA RBI Brazil policy rate decision latest site:tradingeconomics.com"
- "S&P Global manufacturing + services PMI US Eurozone Japan China Korea latest site:pmi.spglobal.com OR site:tradingeconomics.com"
- "Eurozone Japan Korea China India GDP CPI unemployment latest site:tradingeconomics.com OR site:oecd.org"
- "US Fed Beige Book, Korea customs 관세청 export import trend, China trade balance latest site:tradingeconomics.com"
The LIVE DATA above is an internal equity-derived proxy only (NOT real CPI/PMI/rate data) —
ground your rating in the actual current macro releases you find, not just the proxy.

Allowed ratings: RISK_ON | PRO_GROWTH | REFLATION | LATE_CYCLE | DEFENSIVE | RISK_OFF | MIXED | TRANSITIONAL
agent id = "macro_analyst"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## CROSS-ASSET SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANE: Market-priced cross-asset signals — UST/JGB/Bund yields, credit spreads (IG/HY OAS),
DXY, FX pairs (USD/JPY, EUR/USD, USD/KRW, AUD/USD), VIX, crude oil (WTI/Brent), gold, copper.
Global coverage. NOT policy-rate decisions (macro section's lane). Check whether cross-asset
agrees with equity regime.

Reference: System regime tag = {snap['regime_tag']}, OER avg = {_fnum(snap['oer_avg'])}
WebSearch REQUIRED (this section): run 2-3 targeted queries to verify CURRENT levels
(e.g., "10 year treasury yield today", "high yield OAS credit spread current",
"USD/JPY EUR/USD USD/KRW AUD/USD exchange rate today", "DXY WTI VIX today").
Ground (a)-(f) below in those live numbers — do NOT guess from training data.
Key deliverables: (a) yield-curve shape, (b) credit-spread direction, (c) DXY level,
(d) oil level, (e) VIX, (f) USD/JPY · EUR/USD · USD/KRW · AUD/USD levels + direction.
Include ≥1 Japan/Europe/Korea/Australia signal when verifiable (AUD/USD doubles as a
China-demand-proxy commodity-currency read).

Allowed ratings: CONFIRMS_RISK_ON | CONFIRMS_RISK_OFF | DIVERGES_FROM_EQUITY | MIXED | TRANSITIONAL
agent id = "cross_asset_analyst"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## SECTOR/THEME SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANE: Sector leadership breadth, theme breadth, classification distribution.
NOT macro regime, NOT yields, NOT ETF AUM flows.

LIVE DATA:
- Top 5 sectors by bullish %: {", ".join(f"{s['sector']} ({s['pct_bullish']}%, Comp {s['avg_comp']})" for s in top_bull)}
- Top 3 sectors by bearish %: {", ".join(f"{s['sector']} ({s['pct_bearish']}%)" for s in top_bear)}
- Top CONTINUATION leaders: {", ".join(f"{r['ticker']} (Comp {r['composite']:.0f}, 1M {_fpct(r.get('ret_1m'))})" for r in top_cont[:5])}
- Classification: CONTINUATION={cls.get('CONTINUATION',0)}, FORMATION={cls.get('FORMATION',0)}, LAGGING_CATCHUP={cls.get('LAGGING_CATCHUP',0)}, RECOVERY={cls.get('RECOVERY',0)}, OVEREXTENDED={cls.get('OVEREXTENDED',0)} | WEAKENING={cls.get('WEAKENING',0)}, DOWNTREND={cls.get('DOWNTREND',0)}, FADING={cls.get('FADING',0)}

WebSearch REQUIRED (this section): run 1-2 targeted queries to check whether real-world news
corroborates or contradicts the internal breadth reading above (e.g., "stock market sector
leadership rotation today", "which sectors are outperforming this week"). Cite what you find
in key_signals alongside the internal breadth numbers.

Allowed ratings: BROAD_LEADERSHIP | NARROW_LEADERSHIP | ROTATION_IN_PROGRESS | LEADERSHIP_DECAY | EMERGING_LEADERSHIP | MIXED
agent id = "sector_theme_analyst"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## FLOW & MOMENTUM SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANE: Quant strategy net direction, momentum acceleration/deceleration, ETF flow proxies,
factor leadership. NOT macro regime tags, NOT individual sector narratives, NOT yield curves.

LIVE DATA:
- Quant strategies net direction: {net}
- OER avg: {_fnum(snap['oer_avg'])} (>50=stretched, <35=cool/early)
- Cyclical 1M: {_fpct(snap['cyclical']['avg_1m'])} vs Defensive 1M: {_fpct(snap['defensive']['avg_1m'])}
- Growth 1M: {_fpct(snap['growth']['avg_1m'])} vs Value 1M: {_fpct(snap['value']['avg_1m'])}
- Top quant strategies:
{chr(10).join('  • ' + s for s in qs_summary)}

WebSearch REQUIRED (this section): run 1-2 targeted queries on real fund-flow / factor data
(e.g., "ETF fund flows this week", "value vs growth factor performance today") to check whether
external flow data agrees with the internal quant-strategy direction above. Cite findings in
key_signals.

Allowed ratings: ACCELERATING_LEADERSHIP | STALLING_LEADERSHIP | ROTATING_FLOWS | DECAYING_FLOWS | RISK_OFF_FLOWS | MIXED
agent id = "flow_momentum_analyst"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## NEWS NARRATIVE SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LANE: Dominant market narratives from financial news (last 24-48h). Emerging vs fading
themes. Sentiment polarity shifts. NOT macro data (PMI/CPI), NOT cross-asset prices,
NOT sector breadth %, NOT ETF flows.

REQUIRED SOURCES (fetch or search as allowed):
1. Yahoo Finance: https://finance.yahoo.com/
2. Finviz News: https://finviz.com/news
3. CNN Fear & Greed: https://edition.cnn.com/markets/fear-and-greed
4. CNBC Markets: https://www.cnbc.com/markets/
5. KED Global (Korea): https://www.kedglobal.com/
Include CNN F&G score + direction (now vs 1wk/1mo ago) in key_signals.

Reference: System regime tag = {snap['regime_tag']}, Total tickers = {snap['total_tickers']}

Allowed ratings: NARRATIVE_RISK_ON | NARRATIVE_RISK_OFF | NARRATIVE_ROTATION | NARRATIVE_AMBIGUOUS | NARRATIVE_BLOWOFF | NARRATIVE_CAPITULATION
agent id = "news_narrative_analyst"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PER-SECTION SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Each of the 5 values must be a JSON object matching:
{_PHASE1_SCHEMA}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return STRICTLY a fenced ```json block, nothing else. No prose before or after.

```json
{{
  "macro":          {{ ...macro_analyst schema... }},
  "cross_asset":    {{ ...cross_asset_analyst schema... }},
  "sector_theme":   {{ ...sector_theme_analyst schema... }},
  "flow_momentum":  {{ ...flow_momentum_analyst schema... }}
}}
```

★★ LANGUAGE — ALL human-facing commentary text MUST be in KOREAN (한국어) ★★
Applies to narrative / commentary / rationale / key_signals / biggest_risk /
biggest_opportunity fields. Keep JSON keys, enum values, ticker symbols in English.

CONFIDENCE CALIBRATION: Each section's confidence MUST be derived independently:
  confidence = round(0.4 * data_freshness + 0.3 * signal_clarity + 0.3 * cross_source_agreement, 2)
Do NOT use the same confidence value across all 5 sections."""


# ── H1: Lossless Phase 1 propagation helper ─────────────────────────
def _fmt_phase1_full(phase1_dict: dict, key: str, label: str = "") -> str:
    """Lossless formatter for Phase 1 verdict — preserves narrative + key_signals +
    biggest_risk + biggest_opportunity. Replaces the lossy 120-200 char truncation.

    Used by all downstream prompts (Phase 2 coherence, Phase 3 synthesis, Phase 4
    action selector, Phase 5 PM) to ensure specific facts (Fed dot plot, PMI,
    sector breadth) don't get lost in narrative truncation.
    """
    v = phase1_dict.get(key) or {}
    lbl = label or key.upper().replace("_ANALYST", "")
    out = [f"\n┌─ {lbl} [{v.get('rating','—')} conf {v.get('confidence',0)}]"]
    # Bound the prose narrative (was unbounded → bloated every downstream prompt
    # 5×, esp. after the 20-query macro expansion). Discrete facts live in
    # key_signals below, so 700 chars of prose is ample without losing data.
    narr = (v.get('narrative') or '').strip()
    if narr:
        if len(narr) > 700:
            narr = narr[:700] + "…"
        out.append(f"│ {narr}")
    sigs = v.get('key_signals') or []
    if sigs:
        out.append(f"│ KEY SIGNALS:")
        for s in sigs[:5]:
            out.append(f"│   • {str(s)[:180]}")
    risk = (v.get('biggest_risk') or '').strip()
    if risk:
        out.append(f"│ ⚠ RISK: {risk}")
    opp = (v.get('biggest_opportunity') or '').strip()
    if opp:
        out.append(f"│ ✓ OPPORTUNITY: {opp}")
    out.append(f"└─")
    return "\n".join(out)


def _coherence_prompt(phase1: dict) -> str:
    p = phase1
    def _fmt(k):
        # H1 FIX: lossless propagation (was: narrative truncated to 200 chars)
        return _fmt_phase1_full(p, k)
    return f"""You are the COHERENCE DEBATER in a market leadership swarm.

You have just received Phase 1 verdicts from 5 domain analysts:

1. MACRO ANALYST:        {_fmt('macro_analyst')}
2. CROSS-ASSET ANALYST:  {_fmt('cross_asset_analyst')}
3. SECTOR/THEME ANALYST: {_fmt('sector_theme_analyst')}
4. FLOW & MOMENTUM:      {_fmt('flow_momentum_analyst')}
5. NEWS NARRATIVE:       {_fmt('news_narrative_analyst')}

YOUR TASK — cross-coherence check:
1. Do the 5 verdicts AGREE on the dominant market regime? (e.g., all signaling Risk-On vs one diverging)
2. If they DIVERGE: which is the most credible signal (use confidence-weighted reasoning)?
3. List specific CONTESTED AREAS (e.g., "Macro says PRO_GROWTH but Flow says STALLING_LEADERSHIP — momentum may be late-cycle").
4. **Produce a numerical coherence_score (0.0-1.0)** — same scale as Phase 3 cross_panel_coherence_score (M6 fix).
5. Do NOT produce a final verdict — that is the Synthesis Arbitrator's job. Just diagnose coherence.

COHERENCE SCORE CALIBRATION (M6 fix — consistent with Phase 3):
- 1.0: all 5 agents agree on regime tag + direction
- 0.7-0.9: 4 of 5 agents agree, 1 dissenter (still strong consensus)
- 0.4-0.7: 3 of 5 agree OR specific contested area with 2 vs 3 split
- 0.2-0.4: 2 of 5 agree, broad divergence
- 0.0-0.2: no consensus at all

OUTPUT SCHEMA:
```json
{{"coherent": true/false,
"coherence_score": 0.0-1.0,
"dominant_signal": "one-line description of consensus signal (or 'No consensus')",
"contested_areas": ["specific 2-3 sentence descriptions of disagreement"],
"confidence_weighted_winner": "if divergent, which agent's view should weight more and why",
"reasoning": "3-5 sentence cross-agent diagnosis"}}
```

CRITICAL — coherent=true MUST imply coherence_score ≥ 0.65.
                  coherent=false MUST imply coherence_score < 0.65.
                  (Previous bug: coherent=false but conflicts=[]=meta_narrative="" — meaningless.)

{_OUTPUT_RULES}"""


def _synthesis_prompt(phase1: dict, phase2: dict, snap: dict, mode: str) -> str:
    p = phase1
    def _short(k):
        # H1 FIX: lossless propagation (was: 150-char narrative truncation)
        return _fmt_phase1_full(p, k)
    mode_desc = ("RISK-NEUTRAL: capture upside, accept proportional downside" if mode == "neutral"
                 else "RISK-AVERSE: focus on protecting capital; weight Phase 2 contested areas heavier")
    return f"""You are the SYNTHESIS ARBITRATOR ({mode.upper()} mode) in a market leadership swarm.

You combine Phase 1 (5 domain verdicts) + Phase 2 (coherence check) into a single Market Leaders verdict.

MODE: {mode_desc}

PHASE 1 VERDICTS:
- Macro:        {_short('macro_analyst')}
- Cross-Asset:  {_short('cross_asset_analyst')}
- Sector/Theme: {_short('sector_theme_analyst')}
- Flow:         {_short('flow_momentum_analyst')}
- News:         {_short('news_narrative_analyst')}

PHASE 2 COHERENCE (M6: numerical score):
- Coherent: {phase2.get('coherent')} (binary)
- Coherence Score: {phase2.get('coherence_score', '—')} (M6 numerical, same scale as cross_panel_coherence_score below)
- Dominant: {phase2.get('dominant_signal', '—')}
- Contested: {phase2.get('contested_areas', [])}
- Confidence weighted winner: {phase2.get('confidence_weighted_winner', '—')}

CURRENT SYSTEM DETERMINISTIC TAG: {snap['regime_tag']}
(Use this as a baseline. Your synthesis should refine, confirm, or refute it with evidence from Phase 1+2.)

YOUR TASK:
1. Synthesize a final regime tag (can match or refine the system tag).
2. Produce a 4-6 sentence narrative weaving all 4 domains + coherence findings.
3. Provide ONE historical analog (specific period / similar regime).
4. List 3-5 WATCH triggers (specific quant thresholds or macro events that would flip the regime).
5. Score cross-panel coherence (0.0 = strong disagreement, 1.0 = unanimous).

OUTPUT SCHEMA:
```json
{{"regime_tag":"<short tag like 'Risk-On / Pro-Growth' or 'Late-Cycle Stalling'>",
"confidence": 0.0-1.0,
"narrative":"4-6 sentence narrative grounded in Phase 1+2 evidence",
"historical_analog":"one specific past period + 1 sentence rationale",
"watch_triggers":["3-5 specific quant or macro triggers that would flip the regime"],
"cross_panel_coherence_score": 0.0-1.0,
"key_risks":["2-3 risk vectors specific to current regime"]}}
```
{_OUTPUT_RULES}"""


def _strategist_prompt(phase1: dict, snap: dict) -> str:
    """MERGED Phase 2+3 — one call produces coherence + neutral + averse synthesis.
    Replaces _coherence_prompt + 2× _synthesis_prompt (3 LLM calls → 1). Output is
    parsed into the same phase2 / syn_neutral / syn_averse dicts downstream expects,
    so Phase 4/5 are unchanged."""
    p = phase1
    def _f(k):
        return _fmt_phase1_full(p, k)
    return f"""You are the MARKET STRATEGIST in a market leadership swarm. In ONE pass you
merge the 5 domain analysts into: (A) a coherence diagnosis, (B) a RISK-NEUTRAL regime
synthesis, and (C) a RISK-AVERSE risk-overlay synthesis.

PHASE 1 VERDICTS (5 domain analysts):
- Macro:        {_f('macro_analyst')}
- Cross-Asset:  {_f('cross_asset_analyst')}
- Sector/Theme: {_f('sector_theme_analyst')}
- Flow:         {_f('flow_momentum_analyst')}
- News:         {_f('news_narrative_analyst')}

SYSTEM DETERMINISTIC TAG (baseline — refine/confirm/refute with evidence): {snap['regime_tag']}

YOUR TASK (produce all three):
A) COHERENCE — Do the 5 agree on the dominant regime? Which is most credible if divergent
   (confidence-weighted)? List specific contested areas. Score coherence 0.0-1.0
   (1.0=all agree, 0.7-0.9=4/5, 0.4-0.7=3/5, <0.4=broad divergence).
B) NEUTRAL synthesis (capture upside, accept proportional downside) — refined regime tag,
   4-6 sentence narrative weaving all domains, 1 historical analog, 3-5 watch triggers, key risks.
   When the analysts cite market-chart context (valuation / forward P/E / earnings revisions /
   fund-flows — e.g. Yardeni-style charts in the evidence pool), weave that into the
   market-implication narrative (is the regime cheap/expensive vs history?).
C) AVERSE synthesis (protect capital; weight contested areas heavier) — the risk-off scenario:
   what would break the thesis, hedging-relevant risks + triggers.

OUTPUT SCHEMA — strict JSON, one object:
```json
{{
  "coherence": {{"coherent": true/false, "coherence_score": 0.0-1.0,
    "dominant_signal": "one-line consensus (or 'No consensus')",
    "contested_areas": ["2-3 sentence disagreement descriptions"],
    "confidence_weighted_winner": "which agent weights more + why (if divergent)",
    "reasoning": "3-5 sentence cross-agent diagnosis"}},
  "neutral": {{"regime_tag": "<short tag>", "confidence": 0.0-1.0,
    "narrative": "4-6 sentence narrative grounded in Phase 1 evidence",
    "historical_analog": "one past period + 1 sentence rationale",
    "watch_triggers": ["3-5 quant/macro triggers that flip the regime"],
    "cross_panel_coherence_score": 0.0-1.0, "key_risks": ["2-3 regime-specific risks"]}},
  "averse": {{"regime_tag": "<risk-off tag>", "confidence": 0.0-1.0,
    "narrative": "4-6 sentence risk-off scenario", "watch_triggers": ["3-5 risk triggers"],
    "key_risks": ["2-3 capital-protection risk vectors for hedging"]}}
}}
```
CALIBRATION: coherent=true ⟺ coherence_score ≥ 0.65.
{_OUTPUT_RULES}"""


def _action_selector_prompt(phase1: dict, phase2: dict, syn_neutral: dict, snap: dict,
                             syn_averse: dict = None) -> str:
    """H1: lossless Phase 1 propagation. H2: AVERSE synthesis integration for hedge_pairs."""
    def _short(k):
        # H1 FIX: lossless propagation (was: just rating + confidence)
        return _fmt_phase1_full(phase1, k)

    def _fmt_cand(c: dict) -> str:
        ret1m = _fpct(c.get('ret_1m'), '—')
        # Entry Timing: FRESH(적시)/EXTENDED(스트레치) + forming(조기) 태그 — LLM이
        # "이미 늦은 추격"과 "사전 진입"을 구분하도록 노출.
        ets = c.get('entry_timing_score'); ests = c.get('entry_timing_status') or ''
        et = f", ET {ets}·{ests[:4]}" if ets is not None else ""
        pm = f", ⚑PRE-MOM {c.get('pre_momentum_score')}" if c.get('pool_source') == 'pre_momentum' else ""
        return (f"{c['ticker']} (Comp {c['composite']}, OER {c['oer']}, 1M {ret1m}, "
                f"{c.get('classification','')[:14]}, {c.get('sector','')[:14]}{et}{pm})")

    # Pool rendering (2026-07 ③): show top confirmed-momentum names AND the appended
    # forming(pre_momentum) candidates so the early/pre-stretch names aren't truncated
    # by the top-N slice. Confirmed first (composite order), then the ⚑PRE-MOM tail.
    def _fmt_long_pool(pool: list, n_conf: int = 16, n_form: int = 15) -> str:  # n_form 8→15 (forming_cap 동기화)
        conf = [c for c in pool if c.get('pool_source') != 'pre_momentum'][:n_conf]
        form = [c for c in pool if c.get('pool_source') == 'pre_momentum'][:n_form]
        lines = [f"  - {_fmt_cand(c)}" for c in conf]
        if form:
            lines.append("  · ⚑ 조기(Pre-Momentum) 후보 — 아직 스트레치 이전, 사전 진입 검토:")
            lines += [f"  - {_fmt_cand(c)}" for c in form]
        return "\n".join(lines)

    long_stk  = _fmt_long_pool(snap['long_stocks_pool'])
    long_etf  = _fmt_long_pool(snap['long_etfs_pool'])
    short_stk = "\n".join(f"  - {_fmt_cand(c)}" for c in snap['short_stocks_pool'][:20])
    short_etf = "\n".join(f"  - {_fmt_cand(c)}" for c in snap['short_etfs_pool'][:20])
    sec_lines = "\n".join(
        f"  - {s['sector']}: n={s['n']}, bullish {s['pct_bullish']}%, bearish {s['pct_bearish']}%, "
        f"avgComp {s['avg_comp']}, 1M {_fpct(s['avg_1m'])}"
        for s in snap['gics_sectors']
    )
    theme_lines = "\n".join(
        f"  - {t['theme']}: n={t['n']}, mom% {t['mom_pct']}, avgComp {t['avg_comp']}, 1M {_fpct(t['avg_1m'])}"
        for t in snap['themes'][:12]
    )

    # H2: AVERSE synthesis as risk overlay for hedge_pairs + position sizing
    syn_averse = syn_averse or {}
    averse_block = ""
    if syn_averse.get('regime_tag'):
        averse_triggers = syn_averse.get('watch_triggers') or []
        averse_risks = syn_averse.get('key_risks') or []
        averse_block = f"""
INPUT 1b — Phase 3 Synthesis (AVERSE mode, your RISK OVERLAY for hedge_pairs):
- Averse Regime: {syn_averse.get('regime_tag','—')} (conf {syn_averse.get('confidence',0)})
- Averse narrative: {(syn_averse.get('narrative','') or '')[:400]}
- Risk-Off Watch Triggers ({len(averse_triggers)}): {averse_triggers[:5]}
- Averse Key Risks ({len(averse_risks)}): {averse_risks[:3]}

→ Use AVERSE for: (a) sizing down ambiguous picks, (b) selecting hedge_pairs
  that protect against the AVERSE scenario (e.g. if AVERSE warns of Energy
  crash, ensure SHORT side picks include Energy exposure).
"""

    return f"""You are the ACTION SELECTOR — the final synthesis layer that converts swarm regime
analysis into ACTIONABLE picks.

INPUT 1 — Phase 3 Synthesis (NEUTRAL mode, your primary guide):
- Regime: {syn_neutral.get('regime_tag', '—')} (conf {syn_neutral.get('confidence', 0)})
- Narrative: {syn_neutral.get('narrative', '')[:400]}
- Cross-panel coherence: {syn_neutral.get('cross_panel_coherence_score', 0)}
- Key risks: {syn_neutral.get('key_risks', [])}
{averse_block}
INPUT 2 — Phase 1 ratings (H1: full key_signals + risk + opportunity preserved):
- Macro: {_short('macro_analyst')}
- Cross-Asset: {_short('cross_asset_analyst')}
- Sector/Theme: {_short('sector_theme_analyst')}
- Flow: {_short('flow_momentum_analyst')}
- News: {_short('news_narrative_analyst')}

INPUT 3 — Phase 2 Coherence:
- Coherent: {phase2.get('coherent')} · Dominant: {phase2.get('dominant_signal', '—')[:200]}
- Contested areas: {phase2.get('contested_areas', [])[:3]}

═══════════════════════════════════════════════════════════
CANDIDATE POOLS (system-scored top names by classification + composite)
═══════════════════════════════════════════════════════════

LONG STOCK candidates (CONTINUATION/FORMATION/LAGGING_CATCHUP/RECOVERY):
{long_stk}

LONG ETF candidates:
{long_etf}

SHORT STOCK candidates (DOWNTREND/WEAKENING/CYCLE_PEAK/FADING):
{short_stk}

SHORT ETF candidates:
{short_etf}

GICS 11 sectors (current breadth):
{sec_lines}

THEME pool (top 20 by avg composite):
{theme_lines}

═══════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════

1. **LONG picks (20 stocks + 20 ETFs)** — Choose names that BEST FIT the regime tag.
   Examples: Risk-On + Pro-Growth → cyclical/growth leaders; Defensive → quality/staples;
   Late-cycle distribution → rotate to defensive leaders.
   Rank by quality of regime fit + quant signal (highest conviction → #1, lower → #20).
   ★ ENTRY TIMING (ET) — 각 후보의 ET 점수/상태를 반드시 반영: ET FRESH(적시, 아직
     스트레치 이전)를 선호하고, ET EXTENDED(이미 과확장·추격) 종목은 강한 규제적합이
     아니면 후순위로. ⚑PRE-MOM 태그(조기/사전 진입 후보)는 스트레치 이전에 포착할 기회
     이므로 규제에 맞으면 적극 편입 검토. "이미 많이 오른 강한 종목"보다 "곧 오를 적시
     종목"을 우선하는 것이 목표.
   Provide a 1-sentence rationale per pick that cites the regime + a specific quant signal.

2. **SHORT picks (20 stocks + 20 ETFs)** — Choose names with deteriorating signals that
   fit the regime's downside scenario. Rank by short conviction (strongest → #1).
   Cite cover-risk if relevant.

3. **GICS 11 sector scoring (0-100 each)** — Score each of the 11 GICS sectors based on:
   - Current breadth (bullish %)
   - Composite average
   - 1M momentum
   - Regime fit (e.g., Tech high in Risk-On, Utilities high in Risk-Off)
   Provide 1-line rationale per sector.

4. **Top 5 BEST themes / Top 5 WORST themes** — Rank themes by quality (mom_pct + avg_comp
   + 1M consistency + regime fit). Brief rationale per theme.

OUTPUT SCHEMA — strict JSON in a ```json fence:
```json
{{
  "long_stocks": [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent"}}, ...top 8 by conviction],
  "long_etfs":   [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent"}}, ...top 8 by conviction],
  "short_stocks":[{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent"}}, ...top 8 by conviction],
  "short_etfs":  [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent"}}, ...top 8 by conviction],
  "sector_scores":[{{"sector":"Technology","score":0-100,"rationale":"1 sent"}}, ...all 11 GICS sectors],
  "top_themes":   [{{"theme":"...","score":0-100,"rationale":"1 sent"}}, ...exactly 5],
  "bottom_themes":[{{"theme":"...","score":0-100,"rationale":"1 sent"}}, ...exactly 5]
}}
```
Pick tickers FROM the candidate pools above only. Stay grounded — every pick must cite
either a quant signal (composite, OER, classification) or a regime fit (e.g., "fits Pro-Growth tilt").
{_OUTPUT_RULES}"""


# ═══════════════════════════════════════════════════════════════════
# Phase 5 PM Agent — SPLIT into 3 parallel horizon calls
# ═══════════════════════════════════════════════════════════════════
#
# Rationale: A single PM call producing 240 picks (3 horizons × 80) takes
# 15-20 min of LLM generation time, hitting/exceeding our 1200s timeout.
# Splitting into 3 parallel per-horizon calls (each 80 picks) reduces
# wall-clock time to ~5-7 min while keeping total cost the same.
#
# Each per-horizon call:
#   - tactical (5d)   : 80 picks, horizon-specific rationale
#   - core (21d)      : 80 picks WITH change_type diff vs Phase 4 +
#                       global commentary/thesis/drops/hedge_pairs/risk_budget
#   - strategic (63d) : 80 picks, horizon-specific rationale

_HORIZON_GUIDANCE = {
    "core": {
        "days": "21 trading days (~1 month)",
        "label": "CORE",
        "signal_source": "BALANCED across all 5 Phase 1 agents + Phase 3 synthesis",
        "long_thesis": '"I expect +5-15% within 1 month if regime tag holds"',
        "short_thesis": '"I expect -5-15% within 1 month"',
        "sector_cap": 5,
        "avoid": "names lacking 1-month thesis",
        "phase4_use": "USE Phase 4 draft as STARTING POINT. Apply change_type diff (NEW/PROMOTED/DEMOTED/SAME) tags.",
    },
}


def _pm_horizon_prompt(phase1: dict, phase2: dict, syn_n: dict, syn_a: dict,
                       phase4: dict, snap: dict, horizon: str) -> str:
    """Generate a PM agent prompt for ONE specific horizon (tactical/core/strategic).

    Core horizon also requests global outputs (pm_commentary, portfolio_thesis,
    phase4_drops, hedge_pairs, risk_budget) since it's the primary horizon.
    Tactical/Strategic produce ONLY their 80 picks.
    """
    h = _HORIZON_GUIDANCE[horizon]
    is_core = (horizon == "core")

    def _short(k):
        # H1 FIX: lossless propagation (was: 120-char narrative truncation)
        return _fmt_phase1_full(phase1, k)

    def _fmt_p4(picks: list, label: str) -> str:
        if not picks:
            return f"  {label}: (empty)"
        lines = [f"  {label} ({len(picks)} picks ranked by Phase 4 conviction):"]
        for i, p in enumerate(picks[:20], 1):
            lines.append(f"    {i:2}. {p.get('ticker','?'):8} {p.get('name','')[:24]:24} Comp {p.get('composite',0):>5} · {p.get('sector','')[:18]:18} · {p.get('rationale','')[:100]}")
        return "\n".join(lines)

    # H5 FIX: include Phase 4 sector_scores + themes (previously wasted)
    def _fmt_p4_sectors(scores: list) -> str:
        if not scores: return ""
        lines = ["\nPHASE 4 SECTOR SCORES (use as sector tilt bias):"]
        for s in scores[:15]:
            tag = "🟢" if s.get('score', 0) >= 70 else "🔴" if s.get('score', 0) <= 30 else "⚪"
            lines.append(f"  {tag} {s.get('sector','?'):24} score={s.get('score','?'):>3} · {(s.get('rationale','') or '')[:100]}")
        return "\n".join(lines)

    def _fmt_p4_themes(top: list, bottom: list) -> str:
        if not top and not bottom: return ""
        lines = ["\nPHASE 4 THEME SCORES (use as theme tilt bias):"]
        if top:
            lines.append("  ▲ TOP themes (favor):")
            for t in top[:5]:
                lines.append(f"    + {t.get('theme','?'):24} score={t.get('score','?'):>3} · {(t.get('rationale','') or '')[:100]}")
        if bottom:
            lines.append("  ▼ BOTTOM themes (avoid):")
            for t in bottom[:5]:
                lines.append(f"    - {t.get('theme','?'):24} score={t.get('score','?'):>3} · {(t.get('rationale','') or '')[:100]}")
        return "\n".join(lines)

    p4_section = ""
    if is_core:
        p4_long_stk  = _fmt_p4(phase4.get('long_stocks',  []), "LONG STOCKS")
        p4_long_etf  = _fmt_p4(phase4.get('long_etfs',    []), "LONG ETFs")
        p4_short_stk = _fmt_p4(phase4.get('short_stocks', []), "SHORT STOCKS")
        p4_short_etf = _fmt_p4(phase4.get('short_etfs',   []), "SHORT ETFs")
        # H5: include sector_scores + themes (previously discarded)
        p4_sector_block = _fmt_p4_sectors(phase4.get('sector_scores', []))
        p4_theme_block  = _fmt_p4_themes(phase4.get('top_themes', []),
                                          phase4.get('bottom_themes', []))
        p4_section = f"""
PHASE 4 — Action Selector DRAFT (your Core horizon STARTING POINT):
{p4_long_stk}

{p4_long_etf}

{p4_short_stk}

{p4_short_etf}
{p4_sector_block}
{p4_theme_block}
"""

    # Format candidate pools
    def _fmt_cand(pool):
        # 2026-07-28: pool_source(⚑PRE-MOM)·entry_timing 노출 — forming/조기 후보가
        # 'composite 낮은 숫자'로만 심판받아 SKIP되던 병목 해소(Phase4 프롬프트와 정합).
        def _one(c):
            ests = (c.get('entry_timing_status') or '')[:4]
            et = f" · ET {ests}" if ests else ""
            pm = f" · ⚑PRE-MOM {c.get('pre_momentum_score')}" if c.get('pool_source') == 'pre_momentum' else ""
            return (f"  - {c['ticker']:8} {c['name'][:25]:25} Comp {c['composite']:>5} "
                    f"OER {c['oer']:>4} · {c.get('classification',''):14} · {c.get('sector','')[:18]}{et}{pm}")
        return "\n".join(_one(c) for c in pool)

    pools_section = f"""
LONG STOCK candidates ({len(snap['long_stocks_pool'])} names):
{_fmt_cand(snap['long_stocks_pool'])}

LONG ETF candidates ({len(snap['long_etfs_pool'])} names):
{_fmt_cand(snap['long_etfs_pool'])}

SHORT STOCK candidates ({len(snap['short_stocks_pool'])} names):
{_fmt_cand(snap['short_stocks_pool'])}

SHORT ETF candidates ({len(snap['short_etfs_pool'])} names):
{_fmt_cand(snap['short_etfs_pool'])}
"""

    # Output schema — different for core vs tactical/strategic
    # core keeps the change_type diff tag (core-specific value) but the heavy GLOBAL
    # outputs (commentary/thesis/hedge_pairs/risk_budget) are MOVED to a separate
    # lightweight PM Synthesis call — core was timing out at 600s generating all of it.
    if is_core:
        core_change_field = '"change_type":"SAME|PROMOTED|DEMOTED|NEW","change_reason":"1 sent if not SAME"'
        global_fields = ""
    else:
        core_change_field = '"change_type":"NEW"'
        global_fields = ""

    return f"""You are the PORTFOLIO MANAGER (PM) AGENT — Phase 5, **{h['label']} horizon** ({h['days']}).

You receive the upstream research dossier and produce the **{h['label']} horizon picks**:
exactly 10 picks per bucket (long_stocks, long_etfs, short_stocks, short_etfs) = 40 picks total per horizon.
Apply portfolio construction principles WITHIN this horizon only.

═══════════════════════════════════════════════════════════
HORIZON CHARACTER — {h['label']} ({h['days']})
═══════════════════════════════════════════════════════════
Primary signal source: {h['signal_source']}
LONG thesis: {h['long_thesis']}
SHORT thesis: {h['short_thesis']}
Sector concentration cap: max {h['sector_cap']} per GICS sector per bucket
Avoid: {h['avoid']}
Phase 4 baseline: {h['phase4_use']}

═══════════════════════════════════════════════════════════
RESEARCH DOSSIER
═══════════════════════════════════════════════════════════

PHASE 1 — 5 Domain Analysts:
- Macro:       {_short('macro_analyst')}
- Cross-Asset: {_short('cross_asset_analyst')}
- Sector/Theme:{_short('sector_theme_analyst')}
- Flow:        {_short('flow_momentum_analyst')}
- News:        {_short('news_narrative_analyst')}

PHASE 2 — Coherence:
- Coherent: {phase2.get('coherent')}
- Dominant: {phase2.get('dominant_signal','—')[:200]}
- Contested:
{chr(10).join('  • ' + str(c)[:280] for c in (phase2.get('contested_areas') or [])[:5])}

PHASE 3 — Synthesis (Neutral + Averse):
- Neutral: {syn_n.get('regime_tag','—')} (conf {syn_n.get('confidence',0)}) — {syn_n.get('narrative','')[:250]}
- Averse:  {syn_a.get('regime_tag','—')} (conf {syn_a.get('confidence',0)})
- Key risks: {syn_n.get('key_risks', [])[:3]}
{p4_section}
═══════════════════════════════════════════════════════════
CANDIDATE POOLS
═══════════════════════════════════════════════════════════
{pools_section}
═══════════════════════════════════════════════════════════
PORTFOLIO CONSTRUCTION RULES (apply within this horizon)
═══════════════════════════════════════════════════════════
1. Sector concentration: max {h['sector_cap']} per GICS sector per bucket
2. Correlation awareness: don't pick 3 nearly-identical mega-cap value ETFs
3. Phase 2 contested area hedging
4. Phase 1 dissenting agent reflection in SHORT picks
5. Horizon discipline: all 80 picks MUST match the {h['label']} thesis

═══════════════════════════════════════════════════════════
OUTPUT SCHEMA — strict JSON in a ```json fence
═══════════════════════════════════════════════════════════
```json
{{
  "horizon": "{horizon}",
  "long_stocks":  [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent {h['label']}-specific","{core_change_field if is_core else 'change_type'}":"...{('","change_reason":"..."' if is_core else '')}"}}, ...exactly 10, ranked by {h['days']} conviction],
  "long_etfs":    [...exactly 10],
  "short_stocks": [...exactly 10],
  "short_etfs":   [...exactly 10]{(',' + global_fields) if (is_core and global_fields) else ''}
}}
```

CRITICAL: Pick tickers FROM the candidate pools above. Rationale MUST cite horizon-specific
reasoning ({h['days']} time-frame). Same ticker MAY appear in other horizon calls — that's expected.

⚠ ANTI-HALLUCINATION RULE for rationale text (강제):
- Macro claims about foreign central banks (BOJ/ECB/BOK 정책 방향, YCC, 인상/인하 등) MUST be
  sourced from the Cross-Asset Analyst's key_signals or narrative ABOVE in this dossier.
- If Cross-Asset Analyst did NOT verify the central bank stance for this run, you MAY NOT state
  it in rationale (no "BOJ 완화기조 지속", "ECB 매파 전환" 등 unsourced phrases).
- Yen/Euro/Won FX direction claims likewise: cite Cross-Asset Analyst or omit.
- Training-data heuristics about historical central bank behavior are NOT acceptable substitutes
  for current verified signals — these propagate into trade decisions (e.g. DXJ recommendations
  based on stale "BOJ dovish" narrative when BOJ has actually pivoted).
- Domestic/sector reasoning (FORMATION/RECOVERY/OER/Composite/sector rotation) is unaffected.

{_OUTPUT_RULES}"""


def _pm_synthesis_prompt(phase1: dict, phase2: dict, syn_n: dict, syn_a: dict,
                          snap: dict, pm_horizons: dict) -> str:
    """LIGHTWEIGHT PM global-synthesis call — produces pm_commentary / portfolio_thesis /
    hedge_pairs / risk_budget by READING the already-produced 3-horizon picks (no per-pick
    generation). Split out of the core PM call, which was timing out (600s) trying to
    generate 40 picks + change diffs + all of these at once."""
    def _pick_line(p):
        return f"{p.get('ticker','?')}({p.get('sector','')[:12]})"
    def _horizon_summary(h):
        hd = pm_horizons.get(h, {}) or {}
        parts = []
        for bk in ("long_stocks", "long_etfs", "short_stocks", "short_etfs"):
            picks = hd.get(bk, []) or []
            if picks:
                parts.append(f"  {bk}: " + ", ".join(_pick_line(p) for p in picks[:10]))
        return f"[{h.upper()}]\n" + ("\n".join(parts) if parts else "  (none)")
    horizons_block = "\n".join(_horizon_summary(h) for h in ("core",))
    def _f(k):
        return _fmt_phase1_full(phase1, k)
    return f"""You are the PORTFOLIO MANAGER — GLOBAL SYNTHESIS pass. The 3 horizon pick lists
are ALREADY produced (below). Your job is the cross-horizon narrative + hedges + risk
budget ONLY. Do NOT re-list picks.

REGIME: neutral={syn_n.get('regime_tag','—')} (conf {syn_n.get('confidence',0)}) · averse={syn_a.get('regime_tag','—')}
COHERENCE: {phase2.get('dominant_signal','—')[:160]}
KEY RISKS: {syn_n.get('key_risks', [])[:3]} · AVERSE TRIGGERS: {(syn_a.get('watch_triggers') or [])[:3]}
MACRO: {_f('macro_analyst')[:400]}

PRODUCED PICKS (3 horizons):
{horizons_block}

OUTPUT SCHEMA — strict JSON in a ```json fence:
```json
{{
  "pm_commentary": "APPROXIMATELY 1000 CHARACTERS, 2-4 dense paragraphs. Cover: (1) regime synthesis + which Phase 1 agents dominated, (2) contested-area resolution, (3) overall portfolio posture, (4) how the 3 horizons differ in their picks, (5) sector tilt rationale, (6) the key unhedged risk, (7) watch triggers.",
  "portfolio_thesis": "4-6 sentence summary of overall posture across all 3 horizons",
  "hedge_pairs": [{{"long":"X","short":"Y","sector":"...","horizon":"core","rationale":"why this pair"}}, ...3-5 pairs],
  "risk_budget": [{{"sector":"...","allocation_pct":N,"rationale":"1 sent"}}, ...top 5-8 sectors]
}}
```
hedge_pairs MUST use tickers that appear in the produced picks above.
{_OUTPUT_RULES}"""


def _pm_agent_prompt(phase1: dict, phase2: dict, syn_n: dict, syn_a: dict,
                     phase4: dict, snap: dict) -> str:
    """[LEGACY] Single-call PM prompt producing all 3 horizons + global fields.
    Retained for backward compat / fallback. Prefer _pm_horizon_prompt + parallel calls.
    Synthesizes all upstream layers (P1-P4) and produces FINAL portfolio-constructed picks.
    Applies diversification, correlation awareness, contested-area hedging, and long-short
    pair structure. Explicitly tags each pick with change_type vs Phase 4 draft.
    """
    def _short(k):
        # H1 FIX: lossless propagation (was: 120-char narrative truncation)
        return _fmt_phase1_full(phase1, k)

    def _fmt_p4(picks: list, label: str) -> str:
        if not picks:
            return f"  {label}: (empty)"
        lines = [f"  {label} ({len(picks)} picks ranked by Phase 4 conviction):"]
        for i, p in enumerate(picks[:20], 1):
            lines.append(f"    {i:2}. {p.get('ticker','?'):8} {p.get('name','')[:24]:24} Comp {p.get('composite',0):>5} · {p.get('sector','')[:18]:18} · {p.get('rationale','')[:100]}")
        return "\n".join(lines)

    p4_long_stk  = _fmt_p4(phase4.get('long_stocks',  []), "LONG STOCKS")
    p4_long_etf  = _fmt_p4(phase4.get('long_etfs',    []), "LONG ETFs")
    p4_short_stk = _fmt_p4(phase4.get('short_stocks', []), "SHORT STOCKS")
    p4_short_etf = _fmt_p4(phase4.get('short_etfs',   []), "SHORT ETFs")

    return f"""You are the PORTFOLIO MANAGER (PM) AGENT — Phase 5, the FINAL decision layer.

You receive the complete research dossier from 4 upstream phases and produce the
**actually-deployable** portfolio picks. Phase 4 ranks by individual conviction;
your job is portfolio CONSTRUCTION.

═══════════════════════════════════════════════════════════
RESEARCH DOSSIER (upstream synthesis)
═══════════════════════════════════════════════════════════

PHASE 1 — 5 Domain Analysts:
- Macro:       {_short('macro_analyst')}
- Cross-Asset: {_short('cross_asset_analyst')}
- Sector/Theme:{_short('sector_theme_analyst')}
- Flow:        {_short('flow_momentum_analyst')}
- News:        {_short('news_narrative_analyst')}

PHASE 2 — Coherence Debate:
- Coherent: {phase2.get('coherent')}
- Dominant: {phase2.get('dominant_signal','—')[:200]}
- Contested areas (KEY for hedging):
{chr(10).join('  • ' + str(c)[:280] for c in (phase2.get('contested_areas') or [])[:5])}

PHASE 3 — Dual Synthesis:
- Neutral regime: {syn_n.get('regime_tag','—')} (conf {syn_n.get('confidence',0)})
  Narrative: {syn_n.get('narrative','')[:300]}
  Watch triggers: {syn_n.get('watch_triggers', [])[:3]}
  Key risks:     {syn_n.get('key_risks', [])[:3]}
- Averse regime: {syn_a.get('regime_tag','—')} (conf {syn_a.get('confidence',0)})
  Key risks: {syn_a.get('key_risks', [])[:3]}

PHASE 4 — Action Selector DRAFT (your starting point — REVISE):
{p4_long_stk}

{p4_long_etf}

{p4_short_stk}

{p4_short_etf}

═══════════════════════════════════════════════════════════
YOUR PM MANDATE
═══════════════════════════════════════════════════════════

Phase 4 picks are RAW conviction rankings. Apply PM-level portfolio construction:

1. **Sector concentration limit** — max 5 of 20 per single GICS sector. If Phase 4
   over-concentrated (e.g., 8 financials in LONG stocks), demote weaker ones and
   promote next-best from candidate pool below.

2. **Correlation awareness** — Don't pick MGV+VTV+DIA (3 nearly-identical mega-cap
   value ETFs). Keep the strongest, replace others with non-correlated names.

3. **Phase 2 contested area hedging** — If Phase 2 flagged a divergence (e.g., Flow
   says distribution risk while Macro is bullish), explicitly add defensive hedges
   to LONG side AND tighten SHORT side. Cite the contested area in change_reason.

4. **Phase 1 dissenting agent reflection** — If one Phase 1 agent strongly diverges
   (e.g., News RISK_OFF while others RISK_ON), use SHORT picks to express that
   dissent (don't let LONG conviction overrun a clear warning).

5. **Long-Short pair structure** — Where possible, identify pairs (LONG X, SHORT Y
   in same sector/theme) for risk-neutral expression. List them explicitly.

6. **Tail hedge** — Always include 1-2 names in SHORT ETFs that hedge a tail risk
   not yet in consensus (e.g., crypto, EM single-country, sector ETF).

═══════════════════════════════════════════════════════════
HORIZON STRATIFICATION — Produce picks for 3 DISTINCT horizons
═══════════════════════════════════════════════════════════

You produce THREE separate sets of 10 picks per bucket, each optimized for a
different investment horizon. Same ticker CAN appear across horizons IF rationale
aligns with each character. Apply the 6 portfolio principles WITHIN each horizon.

1. **TACTICAL — 5 trading days (~1 week)**
   - Primary signal source: News (Phase 1 #5), Cross-Asset (Phase 1 #2),
     Flow (Phase 1 #4) — short-term catalysts in motion.
   - LONG: "I expect +3-7% move within 1 week" (earnings, headline-driven,
     technical breakout, oversold bounce)
   - SHORT: "I expect -3-7% move within 1 week" (post-earnings momentum unwind,
     headline-driven selloff, technical breakdown)
   - Sector concentration cap: 4 per sector (tighter — tactical bets need diversity)
   - Avoid: slow-moving compounders, multi-quarter thesis names
   - Phase 4 mostly NOT applicable as starting point — these are independent

2. **CORE — 21 trading days (~1 month)** ⭐ Primary horizon
   - Primary signal source: BALANCED across all 5 Phase 1 agents +
     Phase 3 dual synthesis (neutral/averse).
   - LONG: "I expect +5-15% within 1 month if regime tag holds"
   - SHORT: "I expect -5-15% within 1 month"
   - Use Phase 4 draft as STARTING POINT for this horizon. Apply diff
     (NEW/PROMOTED/DEMOTED/SAME tags) ONLY for the Core horizon.
   - Sector concentration cap: 5 per sector
   - Watch triggers from Phase 3 inform this horizon

3. **STRATEGIC — 63 trading days (~3 months)**
   - Primary signal source: Macro (Phase 1 #1), Sector/Theme (Phase 1 #3) dominant.
     De-emphasize News short-term noise.
   - LONG: "I expect +15-30% over 3 months as macro/secular thesis plays out"
     (capital cycle beneficiary, structural growth, regulatory tailwind)
   - SHORT: "I expect -15-30% over 3 months as theme breakdown plays out"
     (secular decline, regulatory headwind, capital cycle peak)
   - Sector concentration cap: 5 per sector
   - Avoid: short-term technical entries, headline-driven names
   - Phase 4 NOT applicable as baseline — these are 3-month strategic picks

KEY RULE: A name in MULTIPLE horizons must have HORIZON-SPECIFIC rationale.
E.g., "AAPL tactical: 1-week earnings beat momentum" vs "AAPL strategic: 3-month
AI integration cycle". Don't just copy the same rationale across horizons.

═══════════════════════════════════════════════════════════
EXPANDED CANDIDATE POOLS (use these to ADD names not in Phase 4)
═══════════════════════════════════════════════════════════

LONG STOCK candidate pool ({len(snap['long_stocks_pool'])} names):
{chr(10).join(f"  - {c['ticker']:8} {c['name'][:25]:25} Comp {c['composite']:>5} OER {c['oer']:>4} · {c.get('classification',''):14} · {c.get('sector','')[:18]}" for c in snap['long_stocks_pool'])}

LONG ETF candidate pool ({len(snap['long_etfs_pool'])} names):
{chr(10).join(f"  - {c['ticker']:8} {c['name'][:25]:25} Comp {c['composite']:>5} OER {c['oer']:>4} · {c.get('classification',''):14} · {c.get('sector','')[:18]}" for c in snap['long_etfs_pool'])}

SHORT STOCK candidate pool ({len(snap['short_stocks_pool'])} names):
{chr(10).join(f"  - {c['ticker']:8} {c['name'][:25]:25} Comp {c['composite']:>5} OER {c['oer']:>4} · {c.get('classification',''):14} · {c.get('sector','')[:18]}" for c in snap['short_stocks_pool'])}

SHORT ETF candidate pool ({len(snap['short_etfs_pool'])} names):
{chr(10).join(f"  - {c['ticker']:8} {c['name'][:25]:25} Comp {c['composite']:>5} OER {c['oer']:>4} · {c.get('classification',''):14} · {c.get('sector','')[:18]}" for c in snap['short_etfs_pool'])}

═══════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════

For EACH pick, tag it with change_type vs the Phase 4 DRAFT above:
  - "SAME"     : ticker was in Phase 4 at similar rank (±3 positions)
  - "PROMOTED" : ticker was in Phase 4 but you moved it up >3 positions
  - "DEMOTED"  : ticker was in Phase 4 but you moved it down >3 positions
  - "NEW"      : ticker was NOT in Phase 4 top-20 (you added from candidate pool)

For NEW/PROMOTED/DEMOTED picks, include `change_reason` (1 sentence explaining the
PM judgment — e.g., "added as defensive hedge per Phase 2 contested Flow risk").

Also list `phase4_drops` — Phase 4 picks you DROPPED from top-20 (with reason).

OUTPUT SCHEMA — strict JSON in a ```json fence:
```json
{{
  "pm_commentary": "Comprehensive PM commentary, APPROXIMATELY 1000 CHARACTERS (700-1100 char range, Korean or English). Cover: (1) regime synthesis + Phase 1 agent dominance, (2) Phase 2 contested area resolution, (3) overall portfolio posture for Core horizon (21d), (4) 2-3 most significant Phase 4 overrides for Core horizon, (5) sector tilt rationale referencing risk_budget, (6) the key risk you are explicitly NOT hedging, (7) watch triggers that would flip Core posture. 2-4 dense paragraphs.",
  "portfolio_thesis": "4-6 sentence summary of your overall portfolio posture, citing which Phase 1+2+3 signals dominated each horizon's construction",
  "horizons": {{
    "core": {{
      "long_stocks":  [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent — CORE 21d rationale","change_type":"SAME|PROMOTED|DEMOTED|NEW","change_reason":"1 sent if not SAME"}}, ...exactly 10, ranked by 1-month conviction],
      "long_etfs":    [...exactly 10, with change_type vs Phase 4],
      "short_stocks": [...exactly 10, with change_type vs Phase 4],
      "short_etfs":   [...exactly 10, with change_type vs Phase 4]
    }}
  }},
  "phase4_drops": [{{"bucket":"long_stocks|long_etfs|short_stocks|short_etfs","ticker":"X","reason":"why dropped from CORE horizon"}}, ...as many as needed],
  "hedge_pairs": [{{"long":"X","short":"Y","sector":"...","horizon":"core","rationale":"why this pair at this horizon"}}, ...3-5 pairs total],
  "risk_budget": [{{"sector":"...","allocation_pct":N,"rationale":"1 sent (refers to CORE horizon allocation)"}}, ...top 5-8 by allocation]
}}
```

CRITICAL FORMATTING: "horizons" key MUST contain "core" —
with all 4 bucket arrays of exactly 10 picks. Total picks = 1 × 4 × 10 = 40.
Pick tickers FROM the candidate pools above. Same ticker can appear across horizons
ONLY with horizon-specific rationale. Every PM Core override must cite either a
portfolio construction principle OR an upstream signal (Phase 2 contested area,
Phase 1 dissent).

⚠ ANTI-HALLUCINATION RULE for rationale text (강제, mirror of horizon prompt):
- Foreign central bank claims (BOJ/ECB/BOK 방향, 완화/긴축, YCC 등) MUST be sourced from
  Cross-Asset Analyst's key_signals/narrative in the dossier above. If not present there,
  OMIT from rationale — do not fall back on training-data heuristics like "BOJ has
  historically been dovish".
- Yen/Euro/Won FX direction claims must also cite Cross-Asset Analyst or be omitted.
- Domestic technical reasoning (FORMATION/RECOVERY/OER/Composite/sector rotation) unaffected.

{_OUTPUT_RULES}"""


# ─────────────────────────────────────────────────────────────────────
# `claude -p` invocation moved to agents.swarm.subprocess_runner during Option B refactor.
# Re-exported here for backward-compat with existing callers.
# ─────────────────────────────────────────────────────────────────────
from agents.swarm.subprocess_runner import (
    run_claude as _run_claude,
    reap_zombie_claude_processes as _reap_zombie_claude_processes,
    find_claude as _find_claude,
    extract_json as _extract_json,
)


# ─────────────────────────────────────────────────────────────────────
# Phase helper functions (extracted from run_swarm for readability)
# ─────────────────────────────────────────────────────────────────────

def _run_phase0_facts(snap: dict, asof: str, run_claude_fn, emit_fn) -> dict:
    """Phase 0 — 제거됨. news_narrative_analyst가 Phase 1에서 직접 WebSearch 수행."""
    emit_fn("phase0_fact", "fact_collector", "skipped — merged into news_narrative_analyst")
    return {"evidence_pool": []}


def _run_phase1_analysts(snap: dict, facts: dict, run_claude_fn, emit_fn) -> tuple:
    """Phase 1 — unified analyst call (fast path) with 5-parallel fallback.
    Returns (phase1_results, phase1_errors) dicts.
    """
    emit_fn("phase1", "all", "started")

    # Build Phase 1 prompts with Phase 0 evidence pool injected (if available).
    # These are kept for the fallback path — individual analyst calls.
    def _inject_evidence_for(agent_name: str, base_prompt: str) -> str:
        if not facts.get("evidence_pool"):
            return base_prompt
        filtered = filter_evidence_for_agent(facts["evidence_pool"], agent_name)
        if not filtered:
            return base_prompt
        ev_block = format_evidence_for_prompt(filtered)
        if PHASE1_RELY_ON_PHASE0:
            return (
                base_prompt
                + f"\n\n{ev_block}\n\n"
                + "⚡ SPEED MODE: Do NOT run any WebSearch. The shared evidence pool "
                + "above (Phase 0, authoritative sources) is your ONLY data source — "
                + "synthesize your verdict from it. Set websearch_results to [] and cite "
                + "pool item ids in key_signals."
            )
        return (
            base_prompt
            + f"\n\n{ev_block}\n\n"
            + "NOTE: The shared evidence pool above already covers high-value facts. "
            + "You MAY add 1-2 additional WebSearch calls for domain-specific gaps, "
            + "but do NOT duplicate searches the pool already covered."
        )

    phase1_prompts = {
        "macro_analyst":         _inject_evidence_for("macro_analyst",         _macro_prompt(snap)),
        "cross_asset_analyst":   _inject_evidence_for("cross_asset_analyst",   _cross_asset_prompt(snap)),
        "sector_theme_analyst":  _inject_evidence_for("sector_theme_analyst",  _sector_theme_prompt(snap)),
        "flow_momentum_analyst": _inject_evidence_for("flow_momentum_analyst", _flow_momentum_prompt(snap)),
        "news_narrative_analyst":_inject_evidence_for("news_narrative_analyst",_news_narrative_prompt(snap)),
    }

    phase1: dict = {}
    phase1_errors: dict = {}
    # 2026-07-05: SPEED MODE 폐지로 4개 섹션(news_narrative 제외) 전부 WebSearch 필수 —
    # sector_theme/flow_momentum도 strict 집합에 포함 (기존엔 macro/cross_asset/news만 강제).
    strict_agents = {"macro_analyst", "cross_asset_analyst",
                      "sector_theme_analyst", "flow_momentum_analyst", "news_narrative_analyst"}
    MAX_WEBSEARCH_RETRIES = 1   # M7 FIX: 1 retry with stronger enforcement

    # ── Unified Phase 1 — try ONE call for all 5 analysts ──────────────
    _UNIFIED_KEY_MAP = {
        "macro":          "macro_analyst",
        "cross_asset":    "cross_asset_analyst",
        "sector_theme":   "sector_theme_analyst",
        "flow_momentum":  "flow_momentum_analyst",
        "news_narrative": "news_narrative_analyst",
    }

    def _parse_unified_result(raw: dict) -> tuple:
        """Parse unified call output into phase1 dict + list of missing keys."""
        parsed = {}
        missing = []
        for unified_key, phase1_key in _UNIFIED_KEY_MAP.items():
            section = raw.get(unified_key)
            if not section or not isinstance(section, dict):
                missing.append(unified_key)
                continue
            if not section.get("rating") or not section.get("narrative"):
                missing.append(unified_key)
                continue
            section.setdefault("agent", phase1_key)
            section.setdefault("confidence", 0.5)
            section.setdefault("confidence_factors", {})
            section.setdefault("key_signals", [])
            section.setdefault("biggest_risk", "")
            section.setdefault("biggest_opportunity", "")
            section.setdefault("websearch_queries", [])
            section.setdefault("websearch_results", [])
            parsed[phase1_key] = section
        return parsed, missing

    # ── news_narrative 병렬 선실행 (WebSearch 포함, 별도 call) ──────────
    import concurrent.futures as _cf
    _news_result: dict = {}
    def _run_news_narrative():
        emit_fn("phase1", "news_narrative_analyst", "started")
        # news_narrative WebFetches 5 news sites + WebSearch fallback → routinely 8-11 min.
        # (240s was far too tight once SPEED MODE was removed — call succeeded at ~655s but
        #  the old 240s timeout would SIGKILL it mid-fetch.)
        res = run_claude_fn(phase1_prompts["news_narrative_analyst"], 900, 0)
        if res and res.get("rating"):
            res.setdefault("agent", "news_narrative_analyst")
            res.setdefault("confidence", 0.5)
            res.setdefault("key_signals", [])
            res.setdefault("biggest_risk", "")
            res.setdefault("biggest_opportunity", "")
            res.setdefault("websearch_queries", [])
            res.setdefault("websearch_results", [])
            # emerging_tickers (2026-07-28) — 유니버스 확장 후보 발굴용. graceful: 필드
            # 없거나 형식 이상이면 빈 배열. 티커 형식 검증만(온보딩은 수동 판단).
            _emg = res.get("emerging_tickers")
            if not isinstance(_emg, list):
                _emg = []
            _clean = []
            for _e in _emg[:8]:
                if isinstance(_e, dict) and _e.get("ticker"):
                    _tk = str(_e.get("ticker")).strip().upper()[:12]
                    if _tk and all(c.isalnum() or c in ".-" for c in _tk):
                        _clean.append({"ticker": _tk, "company": str(_e.get("company") or "")[:40],
                                       "theme": str(_e.get("theme") or "")[:30],
                                       "catalyst": str(_e.get("catalyst") or "")[:120]})
            res["emerging_tickers"] = _clean
            emit_fn("phase1", "news_narrative_analyst", "ok")
            return res
        emit_fn("phase1", "news_narrative_analyst", "fail")
        return None

    _news_ex = _cf.ThreadPoolExecutor(max_workers=1)
    _news_future = _news_ex.submit(_run_news_narrative)

    # ── unified call: 4개 섹션 전부 WebSearch 1-3회씩 수행 (2026-07-05, SPEED MODE 폐지) ──
    # 4섹션 × 2-3회 = 최대 ~12회 WebSearch를 한 subprocess가 순차 수행 → 실측 12분+.
    # 600s는 너무 짧아 정상 진행 중 타임아웃되던 값. 20분으로 상향.
    _unified_timeout = 1200
    _unified_success = False
    try:
        emit_fn("phase1", "unified_analyst", "started")
        unified_prompt = _unified_analyst_prompt(snap, facts, snap.get("as_of", ""))
        unified_raw = run_claude_fn(unified_prompt, _unified_timeout, 0)
        parsed_phase1, missing_keys = _parse_unified_result(unified_raw)
        # news_narrative는 별도 call에서 처리 — unified 결과에서 제외
        parsed_phase1.pop("news_narrative_analyst", None)
        missing_keys = [k for k in missing_keys if k != "news_narrative"]

        if missing_keys:
            emit_fn("phase1", "unified_analyst",
                    f"partial: missing sections {missing_keys} — falling back for those")
        else:
            emit_fn("phase1", "unified_analyst", "ok — all 4 sections parsed")

        if len(parsed_phase1) >= 3:
            phase1.update(parsed_phase1)
            _unified_success = True
            for unified_key in missing_keys:
                phase1_key = _UNIFIED_KEY_MAP[unified_key]
                phase1_errors[phase1_key] = f"unified call: section '{unified_key}' missing/malformed"
                phase1[phase1_key] = {
                    "agent": phase1_key, "rating": "MIXED", "confidence": 0.0,
                    "narrative": f"[unified section missing: {unified_key}]",
                    "key_signals": [], "biggest_risk": "", "biggest_opportunity": "",
                    "websearch_queries": [], "websearch_results": [],
                }
                emit_fn("phase1", phase1_key, "stub_from_unified_miss")
            for name in phase1:
                if not phase1[name].get("_failed"):
                    emit_fn("phase1", name, "ok")
        else:
            raise ValueError(
                f"unified call returned only {len(parsed_phase1)}/4 valid sections "
                f"(missing: {missing_keys})"
            )
    except Exception as e:
        emit_fn("phase1", "unified_analyst", f"fail ({str(e)[:120]}) — falling back to 5 individual calls")

    # ── H-WS: unified 성공 후 섹션별 WebSearch 강제 검증 (2026-07-05) ──────
    # 통합 call은 구조적으로 성공(4섹션 파싱)해도, 모델이 특정 섹션(주로
    # macro/sector_theme/flow_momentum)에서 WebSearch를 건너뛰는 경우가 실측됨
    # (cross_asset은 lane 자체에 내부 데이터가 없어 안정적으로 검색하지만 나머지
    # 3개는 LIVE DATA만으로도 응답 가능해 모델이 검색을 생략하는 경향). 통합 call
    # 전체를 재실행하는 대신, 비어있는 섹션만 개별 프롬프트(_macro_prompt 등,
    # 이미 lane별 WebSearch 지시문 포함)로 1회 단건 재시도한다.
    UNIFIED_WS_SECTIONS = {"macro_analyst", "cross_asset_analyst",
                           "sector_theme_analyst", "flow_momentum_analyst"}

    def _has_websearch(name: str) -> bool:
        ws = phase1.get(name, {}).get("websearch_results")
        return isinstance(ws, list) and len(ws) >= 1

    if _unified_success:
        retry_names = [n for n in UNIFIED_WS_SECTIONS if not _has_websearch(n)]
        if retry_names:
            emit_fn("phase1", "websearch_enforcement",
                    f"empty websearch_results in {retry_names} — retrying individually")

            def _retry_section(name: str) -> dict:
                base_prompt = phase1_prompts.get(name, "")
                retry_prompt = base_prompt + (
                    "\n\n⚠⚠⚠ CRITICAL RETRY: A previous pass for this exact section returned NO "
                    "websearch_results. You MUST execute at least 1 WebSearch call for THIS "
                    "section's lane BEFORE drafting your response — do not skip it even if the "
                    "LIVE DATA above seems sufficient. If WebSearch genuinely fails, set "
                    'rating to "WEBSEARCH_UNAVAILABLE", confidence 0.0, and say so plainly in '
                    "narrative — do NOT silently answer from training data."
                )
                # single-section websearch retry — 480s (300s was tight for a full
                # analyst prompt that must run ≥1 WebSearch + generate a Korean section)
                return run_claude_fn(retry_prompt, 480, 1)

            with ThreadPoolExecutor(max_workers=max(1, len(retry_names))) as ex:
                fut_map = {ex.submit(_retry_section, name): name for name in retry_names}
                for fut in as_completed(fut_map):
                    name = fut_map[fut]
                    try:
                        retried = fut.result()
                        ws = retried.get("websearch_results")
                        if isinstance(ws, list) and len(ws) >= 1:
                            retried.setdefault("agent", name)
                            phase1[name] = retried
                            emit_fn("phase1", name, "ok_after_websearch_retry")
                        else:
                            phase1[name]["_websearch_warning"] = (
                                f"⚠⚠ {name}: websearch_results EMPTY after unified call + 1 retry — "
                                f"output flagged as POSSIBLE_HALLUCINATION"
                            )
                            phase1[name]["_failed"] = True
                            phase1[name]["confidence"] = min(phase1[name].get("confidence", 0.5), 0.3)
                            emit_fn("phase1", name, "websearch_retry_still_empty")
                    except Exception as _re:
                        emit_fn("phase1", name, f"websearch_retry_failed: {str(_re)[:80]}")

    # ── news_narrative 병렬 결과 수집 ───────────────────────────────────
    # 960s: news 호출 자체가 최대 900s이므로 그보다 길게 대기해야 결과를 버리지 않음
    # (기존 260s는 news가 655s 걸리면 결과를 폐기하고 '[뉴스 검색 실패]' 스텁으로 대체하던 값).
    try:
        _news_res = _news_future.result(timeout=960)
        if _news_res:
            phase1["news_narrative_analyst"] = _news_res
        else:
            phase1["news_narrative_analyst"] = {
                "agent": "news_narrative_analyst", "rating": "NARRATIVE_AMBIGUOUS",
                "confidence": 0.0, "narrative": "[뉴스 검색 실패]",
                "key_signals": [], "biggest_risk": "", "biggest_opportunity": "",
                "websearch_queries": [], "websearch_results": [],
            }
    except Exception as _ne:
        emit_fn("phase1", "news_narrative_analyst", f"timeout/fail: {str(_ne)[:80]}")
    finally:
        # Shut the single-worker executor so its (non-daemon) thread EXITS after the news
        # call returns instead of lingering idle — otherwise each run leaks one idle thread
        # in the long-running API process. On timeout the thread keeps running the (single,
        # retries=0) call; run_swarm's finally SIGKILLs its subprocess so it ends promptly.
        try: _news_ex.shutdown(wait=False)
        except Exception: pass

    # ── Fallback: 5 individual parallel calls (original behaviour) ─────
    if not _unified_success:
        def _phase1_call_with_websearch_enforcement(name: str, prompt: str) -> dict:
            """Strict-mode WebSearch enforcement — re-prompt if WebSearch not used."""
            if PHASE1_RELY_ON_PHASE0 and facts.get("evidence_pool"):
                return run_claude_fn(prompt, 240, 2)   # no WebSearch → shorter timeout
            for ws_attempt in range(MAX_WEBSEARCH_RETRIES + 1):
                # 600s: full analyst prompt WITH WebSearch (was 420s — tight now that
                # every analyst lane runs its own WebSearch verification).
                result = run_claude_fn(prompt, 600, 2)
                ws_results = result.get("websearch_results", [])
                if name not in strict_agents:
                    return result
                if isinstance(ws_results, list) and len(ws_results) >= 1:
                    return result
                if ws_attempt < MAX_WEBSEARCH_RETRIES:
                    emit_fn("phase1", name, f"retry_ws_attempt_{ws_attempt+1}")
                    prompt = prompt + (
                        "\n\n⚠⚠⚠ CRITICAL RETRY: Your previous response did NOT include "
                        "websearch_results. You MUST execute at least 1 WebSearch call BEFORE "
                        "drafting your response. If you cannot search, return:\n"
                        '```json\n{\"agent\":\"' + name + '\", \"rating\":\"WEBSEARCH_UNAVAILABLE\", '
                        '\"confidence\":0.0, \"narrative\":\"WebSearch tool failed — no fresh data\", '
                        '\"websearch_results\":[]}\n```\n'
                        "Do NOT answer from training data when WebSearch is required."
                    )
                else:
                    result["_websearch_warning"] = (
                        f"⚠⚠ {name}: websearch_results EMPTY after {MAX_WEBSEARCH_RETRIES+1} attempts — "
                        f"output flagged as POSSIBLE_HALLUCINATION"
                    )
                    result["_failed"] = True
                    result["confidence"] = min(result.get("confidence", 0.5), 0.3)
            return result

        pending_prompts = {name: p for name, p in phase1_prompts.items() if name not in phase1}
        with ThreadPoolExecutor(max_workers=_SWARM_FANOUT_WORKERS) as ex:
            fut_map = {ex.submit(_phase1_call_with_websearch_enforcement, name, p): name
                       for name, p in pending_prompts.items()}
            for fut in as_completed(fut_map):
                name = fut_map[fut]
                try:
                    result = fut.result()
                    phase1[name] = result
                    if result.get("_failed"):
                        emit_fn("phase1", name, "ok_websearch_warned")
                    else:
                        emit_fn("phase1", name, "ok")
                except Exception as e:
                    phase1_errors[name] = str(e)[:300]
                    phase1[name] = {"agent": name, "rating": "MIXED", "confidence": 0.0,
                                    "narrative": f"[agent failed: {str(e)[:120]}]",
                                    "key_signals": [], "biggest_risk": "", "biggest_opportunity": "",
                                    "websearch_queries": [], "websearch_results": []}
                    emit_fn("phase1", name, "fail")

    return phase1, phase1_errors


def _run_phase23_strategist(snap: dict, phase1_results: dict, run_claude_fn, emit_fn) -> tuple:
    """Phase 2+3 MERGED — Strategist (coherence + neutral + averse in ONE call).
    Replaces the old coherence(1) + dual-synthesis(2) = 3 serial calls → 1 call
    (~6min saved). Parsed into the same phase2 / syn_n / syn_a shapes so Phase 4/5
    are untouched. Honest-failure: any missing sub-block falls back to a safe default.
    Returns (phase2, synthesis_neutral, synthesis_averse).
    """
    emit_fn("phase2", "strategist", "started")
    _fallback_syn = lambda tag: {"regime_tag": tag, "confidence": 0.0, "narrative": "",
                                 "historical_analog": "", "watch_triggers": [],
                                 "cross_panel_coherence_score": 0.0, "key_risks": []}
    try:
        strat = run_claude_fn(_strategist_prompt(phase1_results, snap), 420, 2)
        phase2 = strat.get("coherence") or {}
        syn_n = strat.get("neutral") or {}
        syn_a = strat.get("averse") or {}
        # backfill required keys if the model omitted any
        if not phase2.get("dominant_signal"):
            phase2.setdefault("coherent", True); phase2.setdefault("contested_areas", [])
        if not syn_n.get("regime_tag"): syn_n = {**_fallback_syn(snap['regime_tag']), **syn_n}
        if not syn_a.get("regime_tag"): syn_a = {**_fallback_syn(snap['regime_tag']), **syn_a}
        emit_fn("phase2", "strategist", "ok")
        emit_fn("phase3", "synthesis", f"ok (merged) neutral={syn_n.get('regime_tag','—')}")
    except Exception as e:
        _fb = _fallback_syn(snap['regime_tag'])
        phase2 = {"coherent": True, "dominant_signal": "(strategist failed)",
                  "contested_areas": [], "reasoning": str(e)[:300]}
        syn_n = {**_fb, "narrative": f"[strategist failed: {str(e)[:120]}]"}
        syn_a = _fallback_syn(snap['regime_tag'])
        emit_fn("phase2", "strategist", f"fail: {str(e)[:100]}")
    return phase2, syn_n, syn_a


def _run_phase4_action(snap: dict, phase1_results: dict, phase2: dict,
                       syn_n: dict, syn_a: dict, run_claude_fn, emit_fn) -> dict:
    """Phase 4 — Action Selector (picks + GICS sector scores + themes).
    8-min timeout — output is 80 picks + 11 GICS scores + 10 theme rankings.
    """
    emit_fn("phase4", "action_selector", "started")
    try:
        # H2: pass AVERSE synthesis so action_selector can pick hedges
        action = run_claude_fn(
            _action_selector_prompt(phase1_results, phase2, syn_n, snap, syn_a),
            540, 2)  # 35×4 pools — needs ~5-8min
        emit_fn("phase4", "action_selector", "ok")
    except Exception as e:
        action = {
            "long_stocks": [], "long_etfs": [], "short_stocks": [], "short_etfs": [],
            "sector_scores": [], "top_themes": [], "bottom_themes": [],
            "_error": str(e)[:300],
        }
        emit_fn("phase4", "action_selector", "fail")
    return action


def _p5_aggregate_objection_patterns(obj: dict, scan_data: list = None) -> str:
    """Fix 2-lite: Aggregate per-pick objections into sector/factor patterns."""
    from collections import Counter
    patterns = []
    for h, issues in obj.items():
        if not issues: continue
        trading_wait = sum(1 for i in issues for ms in i.get("issues",[]) if "Trading WAIT" in ms)
        trading_skip = sum(1 for i in issues for ms in i.get("issues",[]) if "Trading SKIP" in ms)
        risk_caution = sum(1 for i in issues for ms in i.get("issues",[]) if "Risk CAUTION" in ms)
        risk_reject  = sum(1 for i in issues for ms in i.get("issues",[]) if "Risk REJECT" in ms)
        if trading_wait + trading_skip + risk_caution + risk_reject == 0:
            continue
        patterns.append(
            f"  [{h}] {len(issues)}개 우려 픽 — Trading: WAIT {trading_wait}/SKIP {trading_skip} · "
            f"Risk: CAUTION {risk_caution}/REJECT {risk_reject}"
        )
    return "\n".join(patterns) if patterns else ""


def _p5_audit_pinned(pinned_picks: dict, rng, pin_max_age: int,
                     pin_re_audit_prob: float) -> tuple:
    """Fix 4 survival-bias mitigation: age pinned picks, force-unpin old ones.
    Returns (kept_pins, released_pins). Mutates pinned_picks in place.
    """
    if not pinned_picks:
        return [], []
    released = []
    for ticker in list(pinned_picks.keys()):
        entry = pinned_picks[ticker]
        entry["age"] += 1
        if entry["age"] >= pin_max_age:
            released.append({"ticker": ticker, "reason": "max_age", **entry})
            del pinned_picks[ticker]
        elif entry["age"] >= 2 and rng.random() < pin_re_audit_prob:
            released.append({"ticker": ticker, "reason": "random_audit", **entry})
            del pinned_picks[ticker]
    return list(pinned_picks.keys()), released


def _p5_update_pinned_after_round(pm_horizons: dict, objections: dict,
                                   pinned_picks: dict) -> int:
    """Fix 4: Pin picks that received NO objections this round.
    Returns count of newly pinned. Mutates pinned_picks in place.
    """
    objection_tickers = set()
    for h_issues in objections.values():
        for it in h_issues:
            objection_tickers.add(it.get("ticker"))
    newly_pinned = 0
    for h, hd in (pm_horizons or {}).items():
        for bk, picks in (hd or {}).items():
            for p in picks or []:
                t = p.get("ticker")
                if not t or t in objection_tickers or t in pinned_picks:
                    continue
                pinned_picks[t] = {"age": 0, "horizon": h, "bucket": bk}
                newly_pinned += 1
    return newly_pinned


def _p5_build_wildcards(snap_data: dict, current_pool_tickers: set,
                         pinned_picks: dict, rejected_pool: set,
                         rng, count: int = 2) -> list:
    """Fix 5: Inject wildcard candidates from outside current consideration pool."""
    wildcards = []
    for pool_key in ("long_stocks_pool", "long_etfs_pool", "short_stocks_pool", "short_etfs_pool"):
        pool = snap_data.get(pool_key, []) or []
        outside = [
            c for c in pool
            if c.get("ticker")
            and c["ticker"] not in current_pool_tickers
            and c["ticker"] not in pinned_picks
            and c["ticker"] not in rejected_pool
        ]
        if outside:
            sampled = rng.sample(outside, min(count, len(outside)))
            for c in sampled:
                wildcards.append({"pool": pool_key, **c})
    return wildcards


def _p5_build_iteration_context(round_n: int, current_picks: set, snap_data: dict,
                                 pinned_picks: dict, rejected_pool: set,
                                 iteration_history: list, rng,
                                 pin_max_age: int, pin_re_audit_prob: float,
                                 wildcard_per_round: int) -> str:
    """Builds enriched context: pinned + memory + wildcards + Pareto framing."""
    ctx_lines = []
    kept_pins, released_pins = _p5_audit_pinned(pinned_picks, rng, pin_max_age, pin_re_audit_prob)
    if kept_pins:
        ctx_lines.append(f"\n═══ PINNED PICKS (no objections in prev rounds, 우선 유지 권고) ═══")
        ctx_lines.append("아래 픽은 직전 라운드들에서 우려를 받지 않은 종목입니다.")
        ctx_lines.append("특별한 이유 없는 한 유지하세요. 그러나 새 후보가 명백히 더 우수하면 교체 가능.")
        for t in list(pinned_picks.keys())[:15]:
            e = pinned_picks[t]
            ctx_lines.append(f"  ★ {t} ({e.get('horizon')}/{e.get('bucket')}, pin_age={e.get('age')})")
    if released_pins:
        ctx_lines.append(f"\n═══ RELEASED PICKS (재평가 필요) ═══")
        ctx_lines.append("Survival bias 방지: 아래 픽은 pin이 해제되어 다시 평가 대상이 됩니다.")
        for r in released_pins[:8]:
            reason_kr = {"max_age": "max age 도달", "random_audit": "random sampling"}.get(r.get("reason"), "?")
            ctx_lines.append(f"  ⚠ {r.get('ticker')} ({reason_kr}, pin_age={r.get('age')})")
    if iteration_history:
        last_round = iteration_history[-1]
        ctx_lines.append(f"\n═══ ROUND {last_round.get('round')} 내 결정 기록 ═══")
        kept = last_round.get("kept_tickers") or []
        added = last_round.get("added_tickers") or []
        removed = last_round.get("removed_tickers") or []
        if kept[:10]:
            ctx_lines.append(f"  유지: {', '.join(kept[:10])}")
        if added[:10]:
            ctx_lines.append(f"  추가: {', '.join(added[:10])} (왜 추가했는지 일관성 유지)")
        if removed[:10]:
            ctx_lines.append(f"  제거: {', '.join(removed[:10])} (재고려 가능 — 패턴 학습 차원)")
    if rejected_pool:
        sample = list(rejected_pool)[:15]
        ctx_lines.append(f"\n═══ 이전 거절 후보 — 재고 가능 (단순 누락 방지) ═══")
        ctx_lines.append(f"  {', '.join(sample)}")
        ctx_lines.append("  ↑ 거절 사유가 더 이상 유효하지 않다면 다시 검토 가능")
    wildcards = _p5_build_wildcards(snap_data, current_picks, pinned_picks,
                                     rejected_pool, rng, wildcard_per_round)
    if wildcards:
        ctx_lines.append(f"\n═══ ⚡ WILDCARD 후보 (overfitting 방지 — 외부 풀에서 random injection) ═══")
        ctx_lines.append("아래는 통상 후보 풀 밖에서 random sampling된 종목입니다.")
        ctx_lines.append("필수 채택은 아니지만, 검토 후 가치 있다고 판단되면 추가 가능.")
        for w in wildcards[:6]:
            ctx_lines.append(
                f"  ⚡ {w.get('ticker')} (comp {w.get('composite','?')}, "
                f"{w.get('classification','?')}, {(w.get('sector') or '')[:14]})"
            )
    ctx_lines.append(f"\n═══ TRADE-OFF FRAMING (Pareto-aware) ═══")
    ctx_lines.append("Trading WAIT/SKIP + Risk CAUTION/REJECT은 trade-off를 명시:")
    ctx_lines.append("  (a) 유지 + size 축소 (timing risk accept)")
    ctx_lines.append("  (b) 교체 → 동일 sector 내 (concentration risk 유지)")
    ctx_lines.append("  (c) 교체 → 다른 sector (diversification 효과)")
    ctx_lines.append("각 픽마다 (a/b/c) 명시적 선택 + 근거 — 단순 swap만 반복하지 말 것.")
    return "\n".join(ctx_lines)


def _p5_extract_tickers(pm_horizons: dict) -> set:
    out = set()
    for h, hd in (pm_horizons or {}).items():
        for bk, picks in (hd or {}).items():
            for p in picks or []:
                if p.get("ticker"):
                    out.add((h, bk, p["ticker"]))
    return out


def _p5_compute_delta(prev_set: set, new_set: set) -> float:
    if not prev_set and not new_set:
        return 0.0
    if not prev_set:
        return 1.0
    sym = prev_set ^ new_set
    union = prev_set | new_set
    return len(sym) / len(union) if union else 0.0


def _p5_aggregate_objections(pm_horizons: dict) -> dict:
    """Trading + Risk verdicts → structured objections per pick for next round."""
    obj = {"core": []}
    for h, hd in (pm_horizons or {}).items():
        if h not in obj: continue
        for bk in ("long_stocks", "long_etfs", "short_stocks", "short_etfs"):
            for p in hd.get(bk, []) or []:
                issues = []
                tm = p.get("timing") or {}
                sig = tm.get("entry_signal", "")
                if sig in ("WAIT", "SKIP"):
                    issues.append(f"Trading {sig}: {(tm.get('rationale') or '')[:120]}")
                rv = p.get("risk_verdict") or {}
                rvote = rv.get("vote", "")
                if rvote in ("CAUTION", "REJECT"):
                    kr = rv.get("key_risk", "—")
                    issues.append(f"Risk {rvote} ({kr}): {(rv.get('rationale') or '')[:120]}")
                if issues:
                    obj[h].append({
                        "ticker": p.get("ticker", "?"), "bucket": bk,
                        "composite": p.get("composite", 0),
                        "issues": issues,
                    })
    return obj


def _p5_fmt_objections_for_prompt(obj: dict) -> str:
    if not obj or not any(obj.values()):
        return ""
    lines = ["\n═══ PREVIOUS ROUND OBJECTIONS (반드시 반영) ═══"]
    for h in ("core",):
        issues = obj.get(h, [])
        if not issues: continue
        lines.append(f"\n[{h}] — {len(issues)}개 pick에 우려 사항:")
        for it in issues[:15]:
            lines.append(f"  • {it['ticker']} ({it['bucket']}, comp {it['composite']}):")
            for ms in it.get("issues", []):
                lines.append(f"      - {ms}")
    lines.append("\n→ 위 우려를 반영하여 픽을 재구성하세요:")
    lines.append("  - Trading WAIT/SKIP인 종목 중 timing 개선 어려운 것 → 교체")
    lines.append("  - Risk CAUTION/REJECT 중 sector concentration 문제 → 다양화")
    lines.append("  - 동일 ticker 유지 시 sizing 조정/rationale 강화 필수")
    return "\n".join(lines)


def _stamp_pool_source(snap: dict, pm_output: dict, emit_fn=None) -> None:
    """pool_source provenance 관통 (2026-07-28) — 결정론 후처리. LLM 픽 dict엔
    pool_source가 없어 forming 후보의 단계별 생존률 추적이 불가능했다. snapshot 풀의
    ticker→pool_source/pre_mom/ET를 join해 픽에 스탬핑(setdefault, idempotent).
    ★반드시 per_ticker_debate 이전에 호출 — debate 프롬프트가 forming 맥락을 읽어야 함."""
    try:
        import math as _m

        def _finite(v):   # _candidate()의 `nan or 0`→nan 이슈로 pre_momentum_score가 NaN일 수 있음
            return v if (isinstance(v, (int, float)) and _m.isfinite(v)) else None
        prov = {}
        for pk in ("long_stocks_pool", "long_etfs_pool", "short_stocks_pool", "short_etfs_pool"):
            for c in (snap.get(pk) or []):
                tk = c.get("ticker")
                if tk and tk not in prov:
                    prov[tk] = {"pool_source": c.get("pool_source") or "momentum",
                                "pre_momentum_score": _finite(c.get("pre_momentum_score")),
                                "entry_timing_status": c.get("entry_timing_status")}
        for h in ("core",):
            for bk in ("long_stocks", "long_etfs", "short_stocks", "short_etfs"):
                for p in (pm_output.get("horizons", {}).get(h, {}).get(bk, []) or []):
                    m = prov.get(p.get("ticker"))
                    if m:
                        p.setdefault("pool_source", m["pool_source"])
                        p.setdefault("pre_momentum_score", m["pre_momentum_score"])
                        p.setdefault("entry_timing_status", m["entry_timing_status"])
                    else:
                        p.setdefault("pool_source", "momentum")
        pm_output["forming_tickers"] = sorted(
            tk for tk, m in prov.items() if m.get("pool_source") == "pre_momentum")
    except Exception as e:
        if emit_fn:
            emit_fn("phase5", "pool_source_stamp", f"skipped: {e}")


def _run_phase5_pm_debate(snap: dict, phase1_results: dict, phase2: dict,
                           syn_n: dict, syn_a: dict, action_result: dict,
                           run_claude_fn, emit_fn) -> dict:
    """Phase 5 — PM swarm (single-pass).

    2026-07 FLATTEN: the outer PM/objection iterative-convergence loop was removed.
    It ran a single round (max rounds was 1) so its multi-round machinery
    (delta/convergence, pins, wildcards, Pareto tracker, best-round fallback) was
    inert, and enabling it risked overfitting to debate objections. The real
    per-name vetting lives in run_per_ticker_debate (Trading/Risk/Critic x2 rounds,
    uncapped over the whole core pool). This function now runs one linear pass:
        PM Agent -> global synthesis -> HOLDING injection -> Per-Ticker Debate.
    Prior multi-round implementation recoverable from git history if ever needed.
    Returns the final pm_output dict.
    """
    round_n = 1                 # single pass (retained for emit_fn phase labels)
    pm_output: dict = {}
    prev_pm_horizons: dict = {}

    # ─── Phase 5: PM Agent (with previous round's objections + bias-aware context) ──
    try:
        empty_bucket = {"long_stocks": [], "long_etfs": [], "short_stocks": [], "short_etfs": []}
        horizon_results: dict = {}
        horizon_errors:  dict = {}

        def _pm_prompt(h: str) -> str:
            return _pm_horizon_prompt(phase1_results, phase2, syn_n, syn_a, action_result, snap, h)

        # core horizon runs (only core is active now)
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut_map = {
                ex.submit(run_claude_fn, _pm_prompt(h), 600, 1): h
                for h in ("core",)
            }
            for fut in as_completed(fut_map):
                h = fut_map[fut]
                emit_fn("phase5", f"pm_{h}_r{round_n}", "started")
                try:
                    horizon_results[h] = fut.result()
                    emit_fn("phase5", f"pm_{h}_r{round_n}", "ok")
                except Exception as e:
                    horizon_errors[h] = str(e)[:300]
                    horizon_results[h] = dict(empty_bucket)
                    emit_fn("phase5", f"pm_{h}_r{round_n}", "fail")

        core = horizon_results.get("core", {})
        pm_output = {
            "pm_commentary":    core.get("pm_commentary", "") or pm_output.get("pm_commentary", ""),
            "portfolio_thesis": core.get("portfolio_thesis", "") or pm_output.get("portfolio_thesis", ""),
            "horizons": {
                "core": {b: core.get(b, []) for b in empty_bucket},
            },
            "phase4_drops": core.get("phase4_drops", []) or [],
            "hedge_pairs":  core.get("hedge_pairs", [])  or [],
            "risk_budget":  core.get("risk_budget", [])  or [],
        }
        if horizon_errors:
            pm_output["_horizon_errors"] = horizon_errors
        # PER-HORIZON backfill: if core hung/failed, fill from action_selector
        for _h in ("core",):
            _hd = pm_output["horizons"][_h]
            if not (_hd.get("long_stocks") or _hd.get("long_etfs")):
                _af = {b: list((action_result or {}).get(b, []) or []) for b in empty_bucket}
                if any(_af.values()):
                    pm_output["horizons"][_h] = _af
                    pm_output.setdefault("_horizon_backfilled", []).append(_h)
        # If core horizon failed → use previous round's picks
        if not any(pm_output["horizons"][h]["long_stocks"] for h in ("core",)):
            if prev_pm_horizons:
                pm_output["horizons"] = {h: {b: list(prev_pm_horizons.get(h, {}).get(b, []))
                                              for b in empty_bucket}
                                          for h in ("core",)}
                pm_output["_round_failed_recovered"] = True
                emit_fn("phase5", f"pm_agent_r{round_n}", "fail_recovered_from_prev")
            else:
                # ROBUSTNESS: round-1 PM failed → fall back to action_selector's picks
                _af = {b: list((action_result or {}).get(b, []) or []) for b in empty_bucket}
                if any(_af[b] for b in empty_bucket):
                    pm_output["horizons"] = {"core": {b: list(_af[b]) for b in empty_bucket}}
                    pm_output["_pm_fallback_action"] = True
                    emit_fn("phase5", f"pm_agent_r{round_n}", "fail_recovered_from_action")
                else:
                    raise RuntimeError(f"Core horizon call failed + no action picks: {horizon_errors}")
        else:
            prev_pm_horizons = {h: {b: list(pm_output["horizons"][h].get(b, []))
                                      for b in empty_bucket}
                                  for h in ("core",)}
            emit_fn("phase5", f"pm_agent_r{round_n}", "ok")
    except Exception as e:
        if prev_pm_horizons:
            empty_bucket = {"long_stocks": [], "long_etfs": [], "short_stocks": [], "short_etfs": []}
            if "horizons" not in pm_output:
                pm_output["horizons"] = {h: {b: list(prev_pm_horizons.get(h, {}).get(b, []))
                                              for b in empty_bucket}
                                          for h in ("core",)}
            pm_output["_round_failed_recovered"] = True
            emit_fn("phase5", f"pm_agent_r{round_n}", "fail_recovered_from_prev")
        else:
            empty_bucket = {"long_stocks": [], "long_etfs": [], "short_stocks": [], "short_etfs": []}
            _af = {b: list((action_result or {}).get(b, []) or []) for b in empty_bucket}
            if not pm_output.get("horizons") and any(_af[b] for b in empty_bucket):
                pm_output = {
                    "portfolio_thesis": f"[PM failed → action_selector fallback: {str(e)[:120]}]",
                    "pm_commentary": "",
                    "horizons": {"core": {b: list(_af[b]) for b in empty_bucket}},
                    "phase4_drops": [], "hedge_pairs": [], "risk_budget": [],
                    "_pm_fallback_action": True, "_error": str(e)[:200],
                }
                emit_fn("phase5", f"pm_agent_r{round_n}", "fail_recovered_from_action")
            elif not pm_output.get("horizons"):
                pm_output = {
                    "portfolio_thesis": f"[PM agent failed: {str(e)[:200]}]",
                    "pm_commentary": "",
                    "horizons": {"core": dict(empty_bucket)},
                    "phase4_drops": [], "hedge_pairs": [], "risk_budget": [],
                    "_error": str(e)[:300],
                }
                emit_fn("phase5", f"pm_agent_r{round_n}", "fail")

    # ─── Phase 5 GLOBAL SYNTHESIS — pm_commentary/thesis/hedge_pairs/risk_budget ──
    _syn_has_picks = any(
        (pm_output.get("horizons", {}).get(h, {}).get("long_stocks") or
         pm_output.get("horizons", {}).get(h, {}).get("long_etfs"))
        for h in ("core",))
    if _syn_has_picks and not pm_output.get("pm_commentary"):
        emit_fn("phase5", "pm_synthesis", "started")
        try:
            _syn = run_claude_fn(
                _pm_synthesis_prompt(phase1_results, phase2, syn_n, syn_a, snap,
                                     pm_output.get("horizons", {})), 300, 1)
            pm_output["pm_commentary"]    = _syn.get("pm_commentary", "")    or pm_output.get("pm_commentary", "")
            pm_output["portfolio_thesis"] = _syn.get("portfolio_thesis", "") or pm_output.get("portfolio_thesis", "")
            pm_output["hedge_pairs"]      = _syn.get("hedge_pairs", [])       or pm_output.get("hedge_pairs", [])
            pm_output["risk_budget"]      = _syn.get("risk_budget", [])       or pm_output.get("risk_budget", [])
            emit_fn("phase5", "pm_synthesis", f"ok ({len(str(pm_output['pm_commentary']))} chars)")
        except Exception as e:
            emit_fn("phase5", "pm_synthesis", f"fail: {str(e)[:80]}")

    # ─── Phase 5.5: Trading Timing (per-round) ───
    _has_picks_round = any(
        (pm_output.get("horizons", {}).get(h, {}).get("long_stocks") or [])
        for h in ("core",)
    )
    if _has_picks_round:
        # ── HOLDING 포지션을 debate pool에 강제 주입 ──────────────────
        # action_selector top-N에 들지 못한 HOLDING 종목도 재평가
        try:
            _ps_holding = {}
            _ps_path_h = Path(".position_state.json")
            if _ps_path_h.exists():
                _ps_raw = json.loads(_ps_path_h.read_text(encoding="utf-8"))
                # 구조: {"positions": {"TICKER::core": {...}}, ...} — positions 하위만 순회
                _ps_positions = _ps_raw.get("positions", {}) if isinstance(_ps_raw, dict) else {}
                from api import STATE as _STATE_H
                _scan_lu_h = {r.get("ticker",""): r for r in (_STATE_H.get("results") or []) if r.get("ticker")}
                # 통합 sector는 STATE['df']에만 존재 (scan results 행에는 sector 키 없음)
                # → 주입 보유분이 sector=''로 composer 캡/디베이트에 들어가던 문제 보정
                _sector_lu_h: dict = {}
                try:
                    _df_h = _STATE_H.get("df")
                    if _df_h is not None and "sector" in _df_h.columns:
                        _sector_lu_h = dict(zip(_df_h["ticker"], _df_h["sector"]))
                except Exception:
                    _sector_lu_h = {}
                for _pkey, _pos in _ps_positions.items():
                    if not isinstance(_pos, dict):
                        continue
                    if _pos.get("state") in ("HOLDING", "ENTERED"):
                        # 키는 "TICKER::horizon" 형식 → ticker만 추출
                        _tk = _pkey.split("::")[0]
                        _ps_holding[_tk] = _pos
            _horizons_mut = pm_output.get("horizons", {})
            for h in PT_DEBATE_HORIZONS:
                _existing_tickers = {
                    p.get("ticker") for bk in ("long_stocks","long_etfs")
                    for p in (_horizons_mut.get(h, {}).get(bk) or [])
                }
                for _tk, _pos in _ps_holding.items():
                    if _tk in _existing_tickers:
                        continue
                    _sr = _scan_lu_h.get(_tk, {})
                    # ETF 판별 — 정식 로직과 동일: category가 "STK_"로 시작하지 않으면 ETF
                    # (asset_type 필드는 STATE['results']에 없어 None → 잘못된 stock 분류 유발했음)
                    _cat = _sr.get("category") or ""
                    _is_etf_tk = not (isinstance(_cat, str) and _cat.startswith("STK_"))
                    _bk = "long_etfs" if _is_etf_tk else "long_stocks"
                    _entered_h = _pos.get("entered_date") or "?"
                    _horizons_mut.setdefault(h, {}).setdefault(_bk, []).append({
                        "ticker": _tk,
                        "composite": _sr.get("composite", 0),
                        "classification": _sr.get("classification", ""),
                        # sector 폴백: df 통합 sector → category (scan 행에 sector 키 없음)
                        "sector": _sector_lu_h.get(_tk) or _sr.get("sector") or _cat or "",
                        "name": _sr.get("name", _tk),
                        # 디베이트가 "논지 없는 픽"으로 오판하지 않도록 재평가 프레임 명시
                        "rationale": (f"보유 재평가 — {_pos.get('state','HOLDING')} "
                                      f"(진입 {_entered_h}, 현재 comp {_sr.get('composite', 0)}). "
                                      f"진입 논지 유효성 재검토 대상"),
                        "change_type": "HELD",
                        "_holding_inject": True,
                    })
                    emit_fn("phase5_pt_debate", "holding_inject", f"{_tk} 보유 종목 debate 강제 추가")
            pm_output["horizons"] = _horizons_mut
        except Exception as _he:
            emit_fn("phase5_pt_debate", "holding_inject_err", str(_he)[:100])

        # Option C: Per-Ticker Debate Engine (replaces legacy 5.5 + 5.55 + 5.6a)
        emit_fn("phase5_pt_debate", f"pt_debate_r{round_n}", "started")
        # forming/ET provenance를 debate 이전에 스탬핑 — 프롬프트가 조기후보 맥락을 읽도록
        _stamp_pool_source(snap, pm_output, emit_fn)
        try:
            macro_ctx = (snap.get("macro_summary") or
                          json.dumps(snap.get("phase1") or {})[:1000])
            pt_horizons = run_per_ticker_debate(
                pm_horizons=pm_output.get("horizons", {}),
                regime_tag=snap.get("regime_tag", "—"),
                macro_context=macro_ctx,
                run_claude_fn=run_claude_fn,
                _emit_fn=emit_fn,
                horizons=PT_DEBATE_HORIZONS,
                max_picks_per_horizon=PT_DEBATE_MAX_PICKS,
                max_rounds=PT_DEBATE_MAX_ROUNDS,
            )
            pm_output["horizons"] = pt_horizons
            pm_output["per_ticker_debate_summary"] = summarize_debate_results(pt_horizons)
            emit_fn("phase5_pt_debate", f"pt_debate_r{round_n}",
                    f"ok summary={pm_output['per_ticker_debate_summary']}")
        except Exception as e:
            pm_output["per_ticker_debate_error"] = str(e)[:300]
            emit_fn("phase5_pt_debate", f"pt_debate_r{round_n}", f"fail: {str(e)[:100]}")

    return pm_output


def _run_phase6_finalize(snap: dict, pm_output: dict, phase0_facts: dict,
                          phase1_results: dict, phase1_errors: dict,
                          phase2: dict, syn_n: dict, syn_a: dict,
                          action: dict, emit_fn, run_claude_fn=None) -> dict:
    """Phase 6 — Post-convergence polish + final payload assembly.
    Runs debate degradation check, portfolio composer, position state machine,
    EXIT_PENDING exit debate, then assembles, writes the cache, and returns the payload.
    """
    # pool_source provenance 안전망 재스탬핑 (idempotent; 주 스탬핑은 debate 이전 _run_phase5_pm_debate)
    _stamp_pool_source(snap, pm_output, emit_fn)
    # ─── Phase 5.6a — Debate Synthesizer (runs ONCE on final converged picks) ───
    _has_picks = any(
        (pm_output.get("horizons", {}).get(h, {}).get("long_stocks") or [])
        for h in ("core",)
    )
    if _has_picks:
        # Per-Ticker Debate already populated debate_synthesis on each pick.
        emit_fn("phase5_6a", "debate", "skipped_per_ticker_mode")

        # ─── Detect degraded Debate Synthesizer output (uniform SOLO/WATCH) ───
        all_picks_check = []
        for h in ("core",):
            for bk in ("long_stocks", "long_etfs"):
                for p in pm_output.get("horizons", {}).get(h, {}).get(bk, []) or []:
                    ds = p.get("debate_synthesis") or {}
                    all_picks_check.append((p, ds.get("tier"), ds.get("final_decision")))

        if all_picks_check:
            n_solo_watch = sum(1 for _, t, fd in all_picks_check if t == "SOLO" and fd == "WATCH")
            solo_watch_ratio = n_solo_watch / len(all_picks_check)
            if solo_watch_ratio > 0.8:
                emit_fn("phase5_6a", "debate_degraded_override",
                        f"{n_solo_watch}/{len(all_picks_check)} picks SOLO/WATCH — overriding with PM conviction")
                pm_output["_debate_degraded"] = True
                for p, _, _ in all_picks_check:
                    comp = float(p.get("composite") or 0)
                    cls = (p.get("classification") or "")
                    cls_str = cls if isinstance(cls, str) else ""
                    if comp >= 75:
                        new_tier, new_dec = "UNANIMOUS", "INCLUDE"
                        new_stars = 3
                    elif comp >= 65 and any(s in cls_str for s in ("CONTINUATION","FORMATION","RECOVERY","LAGGING_CATCHUP")):
                        new_tier, new_dec = "MAJORITY_CLEAN", "INCLUDE"
                        new_stars = 2
                    elif comp >= 55:
                        new_tier, new_dec = "SOLO", "WATCH"
                        new_stars = 1
                    else:
                        new_tier, new_dec = "SOLO", "EXCLUDE"
                        new_stars = 0
                    p["debate_synthesis"] = {
                        "tier": new_tier,
                        "final_decision": new_dec,
                        "stars": new_stars,
                        "_override_reason": "debate_synthesizer_degraded",
                        "debate_transcript": f"⚠ Debate Synthesizer 출력 degradation 감지 ({n_solo_watch}/{len(all_picks_check)} SOLO/WATCH). PM Agent의 composite + classification 기준으로 자동 결정: comp={comp:.1f} → {new_tier}/{new_dec}.",
                        "key_factor": f"comp {comp:.1f} override",
                    }

        # ── Phase 5b — Portfolio Composer ──
        emit_fn("phase5b_compose", "portfolio_composer", "started")
        try:
            composition = compose_portfolio(
                pm_horizons=pm_output.get("horizons", {}),
                regime_tag=snap.get("regime_tag", ""),
                _emit_fn=emit_fn,
            )
            pm_output["horizons"] = composition["horizons"]
            pm_output["portfolio_composition"] = composition["metadata"]
            pm_output["portfolio_composition_summary"] = summarize_composition(composition)
            emit_fn("phase5b_compose", "portfolio_composer",
                    f"ok {pm_output['portfolio_composition_summary']}")
        except Exception as e:
            pm_output["portfolio_composition"] = {"_error": str(e)[:300]}
            emit_fn("phase5b_compose", "portfolio_composer", f"fail: {str(e)[:100]}")

        # ── Phase 5.6 — Position State Machine (hysteresis + thesis-break exits) ──
        emit_fn("phase5_6", "position_state", "started")
        try:
            # Build live thesis context so HOLDING positions exit on THESIS BREAK
            # (bearish classification / stop breach / regime flip / composite floor),
            # NOT on merely dropping out of today's top-N picks.
            try:
                from api import STATE as _ST
                _scan_lu = {r.get("ticker", ""): r for r in (_ST.get("results") or [])
                            if r.get("ticker")}
            except Exception:
                _scan_lu = {}
            try:
                from agents.elliott_wave_stops import _load_cache as _load_stops
                _stops = _load_stops()
            except Exception:
                _stops = {}
            _prices: dict = {}
            try:
                import pickle as _pk
                _pc = _pk.load(open(".backtest_price_cache.pkl", "rb")).get("data", {})
                for _t, _df in _pc.items():
                    try:
                        _prices[_t] = float(_df["Close"].iloc[-1])
                    except Exception:
                        pass
            except Exception:
                pass
            # T3 스톱 체크 가격 신선화: 백테스트 캐시 종가는 수일 묵을 수 있음
            # (스톱은 당일가로 계산됐는데 브레치 판정은 옛 가격 → 판정 왜곡).
            # 스톱 캐시의 current_price(24h TTL 내 재계산)를 우선 사용하고,
            # TTL 지난 스톱 엔트리는 상태머신에 전달하지 않음 (옛 레벨로 오판 방지).
            try:
                from agents.elliott_wave_stops import _is_cache_fresh as _stops_fresh
                _fresh_stops: dict = {}
                for _sk, _se in (_stops or {}).items():
                    if not isinstance(_se, dict):
                        continue
                    if _stops_fresh(_se):
                        _fresh_stops[_sk] = _se
                        _scp = _se.get("current_price")
                        if _scp:
                            _prices[_sk.split("::")[0]] = float(_scp)
                _stops = _fresh_stops
            except Exception:
                pass
            _core = pm_output.get("horizons", {}).get("core", {})
            _short_tk = {p.get("ticker") for b in ("short_stocks", "short_etfs")
                         for p in (_core.get(b) or []) if p.get("ticker")}
            state_summary = apply_state_machine(
                pm_output.get("horizons", {}),
                scan_lookup=_scan_lu, stops=_stops, prices=_prices, short_tickers=_short_tk,
            )
            pm_output["position_state_summary"] = state_summary
            emit_fn("phase5_6", "position_state", "ok")
        except Exception as e:
            pm_output["position_state_summary"] = {"_error": str(e)[:300]}
            emit_fn("phase5_6", "position_state", "fail")

        # ── Phase 5.6b — Exit Debate: EXIT_PENDING 청산 후보 LLM 토론 ──
        emit_fn("phase5_exit_debate", "start", "loading positions")
        try:
            _ps_path = Path(".position_state.json")
            _ps_data = json.loads(_ps_path.read_text(encoding="utf-8")) if _ps_path.exists() else {}
            from api import STATE as _STATE
            _scan_lu = {r.get("ticker", ""): r for r in (_STATE.get("results") or []) if r.get("ticker")}
            from agents.position_state import _days_held as _ps_days_held
            _exit_picks = []
            for _key, _pos in (_ps_data.get("positions") or {}).items():
                if _pos.get("state") != "EXIT_PENDING":
                    continue
                _t, _h = (_key.split("::", 1) + ["core"])[:2]
                _sr = _scan_lu.get(_t, {})
                # L1: real holding duration (the 'days_held' key is never persisted on
                # position dicts — the old _pos.get() always yielded 0, so the debate
                # judged every candidate as 보유 0일) + P&L/MFE evidence from trade_mgmt.
                _tm = _pos.get("trade_mgmt") or {}
                _mfe = _tm.get("mfe_pct"); _pnl = _tm.get("pnl_pct")
                _retained = (round(_pnl / _mfe * 100, 1)
                             if (_mfe or 0) > 0 and _pnl is not None else None)
                _exit_picks.append({
                    "ticker": _t, "horizon": _h,
                    "composite": float(_sr.get("composite") or 0),
                    "classification": _sr.get("classification", ""),
                    "sector": _sr.get("sector") or _sr.get("category", ""),
                    "days_held": _ps_days_held(_pos),
                    "entered_date": _pos.get("entered_date"),
                    "pnl_pct": _pnl, "mfe_pct": _mfe, "mae_pct": _tm.get("mae_pct"),
                    "retained_pct": _retained,
                    "effective_stop": _tm.get("effective_stop"),
                    "exit_reason": ((_pos.get("state_history") or [{}])[-1].get("reason", ""))[:200],
                })
            if _exit_picks:
                from agents.per_ticker_debate import run_exit_debate
                _macro_ctx = snap.get("macro_summary") or json.dumps(snap.get("phase1") or {})[:800]
                _exit_verdicts = run_exit_debate(
                    exit_picks=_exit_picks,
                    regime_tag=snap.get("regime_tag", ""),
                    macro_context=_macro_ctx,
                    run_claude_fn=run_claude_fn,
                    _emit_fn=emit_fn,
                )
                pm_output["exit_debate_results"] = _exit_verdicts
                emit_fn("phase5_exit_debate", "done",
                        f"ok {len(_exit_verdicts)}/{len(_exit_picks)} verdicts")
            else:
                emit_fn("phase5_exit_debate", "done", "no EXIT_PENDING positions")
        except Exception as e:
            emit_fn("phase5_exit_debate", "error", str(e)[:100])

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "snapshot": {
            "as_of": snap["as_of"], "total_tickers": snap["total_tickers"],
            "regime_tag_deterministic": snap["regime_tag"],
            "cd_gap": snap["cd_gap"], "gv_gap": snap["gv_gap"],
            "oer_avg": snap["oer_avg"],
        },
        "phase0_facts":  phase0_facts,    # H4: shared evidence pool
        "phase1": phase1_results,
        "phase1_errors": phase1_errors,
        "phase2": phase2,
        "synthesis_neutral": syn_n,
        "synthesis_averse":  syn_a,
        "phase4_action": action,
        "phase5_pm":     pm_output,
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Auto-snapshot to PM history (forward collection) ──
    try:
        append_snapshot(payload, source="swarm_fresh")
        # Phase 5.5 trading signals — separate history for proxy-vs-actual comparison
        append_trading_snapshot(payload, source="swarm_fresh")
    except Exception:
        pass   # history snapshot is best-effort, never block swarm

    return payload


# ─────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────

def run_swarm(progress_cb=None) -> dict:
    """Execute the 6-agent swarm. Returns the final structured output.

    progress_cb(phase, agent, status) — optional callback for status updates.
    Delegates each phase to a private helper function for readability.
    """
    def _emit(phase, agent, status):
        if progress_cb:
            try: progress_cb(phase, agent, status)
            except Exception: pass

    # 완료-후 좀비 누수 근본수정 (2026-07-18): reset the no-spawn flag on entry, and on ANY
    # exit path (return OR exception) kill the claude subprocesses this run owns — a
    # background thread that outlived run_swarm (e.g. a news_narrative future whose
    # .result(timeout=…) already gave up) otherwise keeps spawning claude and blocks process
    # exit. Reproduced in BOTH the launchd daily_pipeline and the long-running API server.
    from agents.swarm.subprocess_runner import reset_shutdown, shutdown_and_kill_all
    reset_shutdown()
    try:
        snap = build_snapshot()
        if snap.get("error"):
            raise RuntimeError(snap["error"])

        # ─── Phase 0 ─────────────────────────────────────────────────────
        phase0_facts = _run_phase0_facts(snap, snap.get("as_of", ""), _run_claude, _emit)

        # ─── Phase 1 ─────────────────────────────────────────────────────
        phase1, phase1_errors = _run_phase1_analysts(snap, phase0_facts, _run_claude, _emit)

        # ─── Phase 2+3 ───────────────────────────────────────────────────
        phase2, syn_n, syn_a = _run_phase23_strategist(snap, phase1, _run_claude, _emit)

        # ─── Phase 4 ─────────────────────────────────────────────────────
        action = _run_phase4_action(snap, phase1, phase2, syn_n, syn_a, _run_claude, _emit)

        # ─── Phase 5 ─────────────────────────────────────────────────────
        pm_output = _run_phase5_pm_debate(
            snap=snap, phase1_results=phase1, phase2=phase2,
            syn_n=syn_n, syn_a=syn_a, action_result=action,
            run_claude_fn=_run_claude, emit_fn=_emit,
        )

        # ─── Phase 6 ─────────────────────────────────────────────────────
        return _run_phase6_finalize(
            snap=snap, pm_output=pm_output, phase0_facts=phase0_facts,
            phase1_results=phase1, phase1_errors=phase1_errors,
            phase2=phase2, syn_n=syn_n, syn_a=syn_a, action=action,
            emit_fn=_emit, run_claude_fn=_run_claude,
        )
    finally:
        try:
            shutdown_and_kill_all()
        except Exception:
            pass


def load_cached() -> Optional[dict]:
    if not CACHE_PATH.exists():
        return None
    try:
        d = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        return d
    except Exception:
        return None


def cache_fresh() -> bool:
    """True if cache exists and is within TTL."""
    d = load_cached()
    if not d:
        return False
    try:
        gen = time.mktime(time.strptime(d["generated_at"], "%Y-%m-%dT%H:%M:%S"))
        age_h = (time.time() - gen) / 3600
        return age_h < CACHE_TTL_HOURS
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    if "--snapshot" in sys.argv:
        snap = build_snapshot()
        print(json.dumps(snap, indent=2, ensure_ascii=False, default=str)[:3000])
    elif "--prompts" in sys.argv:
        snap = build_snapshot()
        for name, fn in [("MACRO", _macro_prompt), ("CROSS-ASSET", _cross_asset_prompt),
                         ("SECTOR/THEME", _sector_theme_prompt), ("FLOW", _flow_momentum_prompt)]:
            print(f"\n{'━'*20} {name} {'━'*20}")
            print(fn(snap))
    elif "--run" in sys.argv:
        print(json.dumps(run_swarm(lambda p, a, s: print(f"  [{p}] {a}: {s}")),
                         indent=2, ensure_ascii=False))
    else:
        print("Usage: --snapshot | --prompts | --run")
