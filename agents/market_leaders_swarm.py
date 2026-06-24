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

    def _candidate(r: dict) -> dict:
        return {
            "ticker": r.get("ticker"), "name": (r.get("name") or "")[:60],
            "composite": round(float(r.get("composite") or 0), 1),
            "classification": _clean(r.get("classification") or ""),
            "oer": round(float(r.get("oer") or 0), 1),
            "ret_1m": r.get("return_1m"),
            "sector": _sec_of(r), "industry": r.get("gics_industry_group") or r.get("industry") or "",
        }

    LONG_CLS  = {"CONTINUATION", "FORMATION", "LAGGING_CATCHUP", "RECOVERY"}
    SHORT_CLS = {"DOWNTREND", "WEAKENING", "CYCLE_PEAK", "FADING", "EXHAUSTING"}
    long_rows  = [r for r in results if _clean(r.get("classification") or "") in LONG_CLS]
    short_rows = [r for r in results if _clean(r.get("classification") or "") in SHORT_CLS]
    long_rows.sort(key=lambda r: -float(r.get("composite") or 0))
    short_rows.sort(key=lambda r: float(r.get("composite") or 0))   # lowest composite first

    # Expanded pool — 35 candidates per cell so the agent has headroom to
    # pick 20 with sufficient sector/regime diversity.
    long_stocks_pool  = [_candidate(r) for r in long_rows  if not _is_etf(r)][:35]
    long_etfs_pool    = [_candidate(r) for r in long_rows  if     _is_etf(r)][:35]
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
    return f"""You are the MACRO ANALYST in a 4-agent market leadership swarm.

YOUR STRICT LANE:
- Economic cycle phase, sector rotation, Fed/central-bank stance, fiscal context.
- Inputs: cyclical-vs-defensive composite gap, growth-vs-value gap, OER average.
- You may NOT discuss: yield curves, credit spreads, DXY, VIX, ETF flows
  (those are Cross-Asset / Flow analysts' lanes). Stay in macro regime.

LIVE DATA SNAPSHOT (as of {snap['as_of']}, {snap['total_tickers']} tickers):
- System regime tag (deterministic): {snap['regime_tag']}
- Cyclical avg Comp {snap['cyclical']['avg_comp']} (n={snap['cyclical']['n']}, 1M {_fpct(snap['cyclical']['avg_1m'])}) vs Defensive avg Comp {snap['defensive']['avg_comp']} (n={snap['defensive']['n']}, 1M {_fpct(snap['defensive']['avg_1m'])})
  → CD gap: Comp {_fnum(snap['cd_gap'], '{:+.1f}')}, 1M {_fnum(snap['cd_1m'], '{:+.1f}pp')}
- Growth avg Comp {snap['growth']['avg_comp']} (n={snap['growth']['n']}, 1M {_fpct(snap['growth']['avg_1m'])}) vs Value avg Comp {snap['value']['avg_comp']} (n={snap['value']['n']}, 1M {_fpct(snap['value']['avg_1m'])})
  → GV gap: Comp {_fnum(snap['gv_gap'], '{:+.1f}')}, 1M {_fnum(snap['gv_1m'], '{:+.1f}pp')}
- OER average across universe: {_fnum(snap['oer_avg'])} (high=overheated, low=cool)
- Sector regime: cyclical_dom={snap['regime_state'].get('cyclical_dom')}, defensive_dom={snap['regime_state'].get('defensive_dom')}, growth_dom={snap['regime_state'].get('growth_dom')}, value_dom={snap['regime_state'].get('value_dom')}

YOUR TASK:
1. Use ≤2 WebSearch queries for CURRENT macro context (Fed dot-plot, PMI, jobs, fiscal).
2. Interpret system signals through that macro lens.
3. Output verdict in JSON.

Allowed ratings: RISK_ON | PRO_GROWTH | REFLATION | LATE_CYCLE | DEFENSIVE | RISK_OFF | MIXED | TRANSITIONAL

OUTPUT SCHEMA:
```json
{_PHASE1_SCHEMA.replace('<your_agent_id>', 'macro_analyst')}
```
{_OUTPUT_RULES}"""


def _cross_asset_prompt(snap: dict) -> str:
    return f"""You are the CROSS-ASSET ANALYST in a 4-agent market leadership swarm.

YOUR STRICT LANE — GLOBAL coverage (NOT US-only):
- US: yield curve (10Y/2Y/3M), credit spreads (IG/HY OAS), DXY, VIX
- Japan: JGB 10Y yield + BOJ policy stance (rate level, YCC, recent statements), JPY/USD
- Europe: Bund 10Y yield + ECB policy stance (rate level, recent decisions), EUR/USD
- Korea: BOK 7-day repo rate stance, KRW/USD
- Commodities: oil, copper, gold cross-signals
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

YOUR TASK:
1. Use 2-3 WebSearch queries to cover GLOBAL central bank + rate landscape:
   Recommended pattern (pick what's most current; 2-3 queries max):
   - "10-year Treasury yield VIX credit spreads today"
   - "BOJ policy rate JGB yield Japan yen latest decision"
   - "ECB deposit rate Bund yield euro latest decision"  (or "BOK Korea rate KRW" if Asia focus)
2. Synthesize: are global rates moving SAME direction (synchronized hike/cut) or DIVERGING?
   - This synchronization signal is CRITICAL for the PM Agent's regional ETF decisions.
3. Interpret as Risk-On/Risk-Off + flag central bank divergence.
4. Score how WELL cross-asset matches the equity regime tag ({snap['regime_tag']}).

key_signals MUST include (when verified):
- US rates + credit + DXY + VIX (as before)
- ≥1 explicit Japan/Europe/Korea signal (e.g. "BOJ 0.50% maintained, no hawkish shift as of YYYY-MM-DD")
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
4. Google News finance topic (Korea): https://news.google.com/topics/CAAqJggKIiBDQkFTRWdvSUwyMHZNRGx1YlY4U0FtdHZHZ0pMVWlnQVAB?hl=ko&gl=KR&ceid=KR%3Ako
5. WebSearch fallback: "biggest market story today" + current date, or theme-specific queries

YOUR PROCESS:
Step 1: Pull top headlines + Fear & Greed score from sources
        (≤5 fetch/search ops total — fetch each of the 4 required sources first,
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
{_OUTPUT_RULES}"""


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
    narr = (v.get('narrative') or '').strip()
    if narr:
        out.append(f"│ {narr}")
    sigs = v.get('key_signals') or []
    if sigs:
        out.append(f"│ KEY SIGNALS:")
        for s in sigs[:5]:
            out.append(f"│   • {s}")
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


def _action_selector_prompt(phase1: dict, phase2: dict, syn_neutral: dict, snap: dict,
                             syn_averse: dict = None) -> str:
    """H1: lossless Phase 1 propagation. H2: AVERSE synthesis integration for hedge_pairs."""
    def _short(k):
        # H1 FIX: lossless propagation (was: just rating + confidence)
        return _fmt_phase1_full(phase1, k)

    def _fmt_cand(c: dict) -> str:
        ret1m = _fpct(c.get('ret_1m'), '—')
        return f"{c['ticker']} (Comp {c['composite']}, OER {c['oer']}, 1M {ret1m}, {c.get('classification','')[:14]}, {c.get('sector','')[:14]})"

    long_stk  = "\n".join(f"  - {_fmt_cand(c)}" for c in snap['long_stocks_pool'][:35])
    long_etf  = "\n".join(f"  - {_fmt_cand(c)}" for c in snap['long_etfs_pool'][:35])
    short_stk = "\n".join(f"  - {_fmt_cand(c)}" for c in snap['short_stocks_pool'][:35])
    short_etf = "\n".join(f"  - {_fmt_cand(c)}" for c in snap['short_etfs_pool'][:35])
    sec_lines = "\n".join(
        f"  - {s['sector']}: n={s['n']}, bullish {s['pct_bullish']}%, bearish {s['pct_bearish']}%, "
        f"avgComp {s['avg_comp']}, 1M {_fpct(s['avg_1m'])}"
        for s in snap['gics_sectors']
    )
    theme_lines = "\n".join(
        f"  - {t['theme']}: n={t['n']}, mom% {t['mom_pct']}, avgComp {t['avg_comp']}, 1M {_fpct(t['avg_1m'])}"
        for t in snap['themes'][:20]
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
  "long_stocks": [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent"}}, ...exactly 20, ranked by conviction],
  "long_etfs":   [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent"}}, ...exactly 20, ranked by conviction],
  "short_stocks":[{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent"}}, ...exactly 20, ranked by conviction],
  "short_etfs":  [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent"}}, ...exactly 20, ranked by conviction],
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
    "tactical": {
        "days": "5 trading days (~1 week)",
        "label": "TACTICAL",
        "signal_source": "News (Phase 1 #5), Cross-Asset (#2), Flow (#4) — short-term catalysts",
        "long_thesis": '"I expect +3-7% within 1 week" (earnings, headline, breakout, oversold bounce)',
        "short_thesis": '"I expect -3-7% within 1 week" (post-earnings unwind, headline selloff, breakdown)',
        "sector_cap": 4,
        "avoid": "slow-moving compounders, multi-quarter thesis names",
        "phase4_use": "Phase 4 NOT applicable as baseline — tactical picks are independent",
    },
    "core": {
        "days": "21 trading days (~1 month)",
        "label": "CORE",
        "signal_source": "BALANCED across all 5 Phase 1 agents + Phase 3 synthesis",
        "long_thesis": '"I expect +5-15% within 1 month if regime tag holds"',
        "short_thesis": '"I expect -5-15% within 1 month"',
        "sector_cap": 5,
        "avoid": "names lacking 1-month thesis (too tactical or too long-term)",
        "phase4_use": "USE Phase 4 draft as STARTING POINT. Apply change_type diff (NEW/PROMOTED/DEMOTED/SAME) tags.",
    },
    "strategic": {
        "days": "63 trading days (~3 months)",
        "label": "STRATEGIC",
        "signal_source": "Macro (Phase 1 #1), Sector/Theme (#3) dominant; de-emphasize News short-term noise",
        "long_thesis": '"I expect +15-30% over 3 months as macro/secular thesis plays out" (capital cycle, structural growth, regulatory tailwind)',
        "short_thesis": '"I expect -15-30% over 3 months as theme breakdown plays out" (secular decline, regulatory headwind, capital cycle peak)',
        "sector_cap": 5,
        "avoid": "short-term technical entries, headline-driven names",
        "phase4_use": "Phase 4 NOT applicable as baseline — strategic picks are independent",
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
        return "\n".join(
            f"  - {c['ticker']:8} {c['name'][:25]:25} Comp {c['composite']:>5} OER {c['oer']:>4} · {c.get('classification',''):14} · {c.get('sector','')[:18]}"
            for c in pool
        )

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
    if is_core:
        core_change_field = '"change_type":"SAME|PROMOTED|DEMOTED|NEW","change_reason":"1 sent if not SAME"'
        global_fields = '''
  "pm_commentary": "Comprehensive PM commentary, APPROXIMATELY 1000 CHARACTERS. Cover: (1) regime synthesis + Phase 1 agent dominance, (2) Phase 2 contested area resolution, (3) overall portfolio posture across horizons, (4) 2-3 most significant Phase 4 overrides for Core, (5) how the 3 horizons differ in their picks (note: tactical/strategic produced in parallel calls), (6) sector tilt rationale, (7) key unhedged risk, (8) watch triggers. 2-4 dense paragraphs.",
  "portfolio_thesis": "4-6 sentence summary of overall portfolio posture across all 3 horizons",
  "phase4_drops": [{{"bucket":"long_stocks|long_etfs|short_stocks|short_etfs","ticker":"X","reason":"why dropped from CORE"}}],
  "hedge_pairs": [{{"long":"X","short":"Y","sector":"...","horizon":"tactical|core|strategic","rationale":"why this pair"}}, ...3-5 pairs],
  "risk_budget": [{{"sector":"...","allocation_pct":N,"rationale":"1 sent (CORE allocation)"}}, ...top 5-8],'''
    else:
        core_change_field = '"change_type":"NEW"'
        global_fields = ""

    return f"""You are the PORTFOLIO MANAGER (PM) AGENT — Phase 5, **{h['label']} horizon** ({h['days']}).

You receive the upstream research dossier and produce the **{h['label']} horizon picks**:
exactly 20 picks per bucket (long_stocks, long_etfs, short_stocks, short_etfs) = 80 picks total.
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
  "long_stocks":  [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent {h['label']}-specific","{core_change_field if is_core else 'change_type'}":"...{('","change_reason":"..."' if is_core else '')}"}}, ...exactly 20, ranked by {h['days']} conviction],
  "long_etfs":    [...exactly 20],
  "short_stocks": [...exactly 20],
  "short_etfs":   [...exactly 20]{(',' + global_fields) if is_core else ''}
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

You produce THREE separate sets of 20 picks per bucket, each optimized for a
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
  "pm_commentary": "Comprehensive PM commentary, APPROXIMATELY 1000 CHARACTERS (700-1100 char range, Korean or English). Cover: (1) regime synthesis + Phase 1 agent dominance, (2) Phase 2 contested area resolution, (3) overall portfolio posture per horizon (tactical bias vs core posture vs strategic thesis), (4) 2-3 most significant Phase 4 overrides for Core horizon, (5) how you differentiated picks across the 3 horizons (which signals drove each), (6) sector tilt rationale referencing risk_budget, (7) the key risk you are explicitly NOT hedging, (8) watch triggers that would flip Core posture. 2-4 dense paragraphs.",
  "portfolio_thesis": "4-6 sentence summary of your overall portfolio posture, citing which Phase 1+2+3 signals dominated each horizon's construction",
  "horizons": {{
    "tactical": {{
      "long_stocks":  [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent — TACTICAL 5d-specific rationale","change_type":"NEW"}}, ...exactly 20, ranked by 1-week conviction],
      "long_etfs":    [...exactly 20],
      "short_stocks": [...exactly 20],
      "short_etfs":   [...exactly 20]
    }},
    "core": {{
      "long_stocks":  [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent — CORE 21d rationale","change_type":"SAME|PROMOTED|DEMOTED|NEW","change_reason":"1 sent if not SAME"}}, ...exactly 20, ranked by 1-month conviction],
      "long_etfs":    [...exactly 20, with change_type vs Phase 4],
      "short_stocks": [...exactly 20, with change_type vs Phase 4],
      "short_etfs":   [...exactly 20, with change_type vs Phase 4]
    }},
    "strategic": {{
      "long_stocks":  [{{"ticker":"X","name":"...","composite":N,"sector":"...","rationale":"1 sent — STRATEGIC 63d rationale","change_type":"NEW"}}, ...exactly 20, ranked by 3-month conviction],
      "long_etfs":    [...exactly 20],
      "short_stocks": [...exactly 20],
      "short_etfs":   [...exactly 20]
    }}
  }},
  "phase4_drops": [{{"bucket":"long_stocks|long_etfs|short_stocks|short_etfs","ticker":"X","reason":"why dropped from CORE horizon"}}, ...as many as needed],
  "hedge_pairs": [{{"long":"X","short":"Y","sector":"...","horizon":"tactical|core|strategic","rationale":"why this pair at this horizon"}}, ...3-5 pairs total],
  "risk_budget": [{{"sector":"...","allocation_pct":N,"rationale":"1 sent (refers to CORE horizon allocation)"}}, ...top 5-8 by allocation]
}}
```

CRITICAL FORMATTING: "horizons" key MUST contain "tactical", "core", "strategic" —
each with all 4 bucket arrays of exactly 20 picks. Total picks = 3 × 4 × 20 = 240.
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
# Orchestrator
# ─────────────────────────────────────────────────────────────────────

def run_swarm(progress_cb=None) -> dict:
    """Execute the 6-agent swarm. Returns the final structured output.

    progress_cb(phase, agent, status) — optional callback for status updates.
    """
    def _emit(phase, agent, status):
        if progress_cb:
            try: progress_cb(phase, agent, status)
            except Exception: pass

    snap = build_snapshot()
    if snap.get("error"):
        raise RuntimeError(snap["error"])

    # ─── Phase 0 — Fact Collector (shared evidence pool) ─────────────
    # Single LLM call with 10 WebSearch queries → shared facts for all Phase 1 agents
    # Reduces total WebSearch from 14 → 10 + improves cross-agent consistency.
    phase0_facts = {}
    _emit("phase0_fact", "fact_collector", "started")
    try:
        phase0_facts = run_fact_collector(
            asof=snap.get("as_of", ""),
            run_claude_fn=_run_claude,
            _emit_fn=_emit,
            timeout=600,
        )
    except Exception as e:
        phase0_facts = {"_failed": True, "_failure_reason": str(e)[:200], "evidence_pool": []}
        _emit("phase0_fact", "fact_collector", f"fail: {str(e)[:100]}")

    # Phase 1 — 5 parallel
    _emit("phase1", "all", "started")

    # Build Phase 1 prompts with Phase 0 evidence pool injected (if available)
    def _inject_evidence_for(agent_name: str, base_prompt: str) -> str:
        if not phase0_facts.get("evidence_pool"):
            return base_prompt
        filtered = filter_evidence_for_agent(phase0_facts["evidence_pool"], agent_name)
        if not filtered:
            return base_prompt
        ev_block = format_evidence_for_prompt(filtered)
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
    # Reduced from 5 → 2 to avoid Max plan concurrent session lock ("Not logged in")
    # 5 analysts run in 2-batch pattern: takes ~2.5x longer but avoids session conflicts
    strict_agents = {"macro_analyst", "cross_asset_analyst", "news_narrative_analyst"}
    MAX_WEBSEARCH_RETRIES = 1   # M7 FIX: 1 retry with stronger enforcement

    def _phase1_call_with_websearch_enforcement(name: str, prompt: str) -> dict:
        """Strict-mode WebSearch enforcement — re-prompt if WebSearch not used."""
        for ws_attempt in range(MAX_WEBSEARCH_RETRIES + 1):
            result = _run_claude(prompt, 420, 2)
            ws_results = result.get("websearch_results", [])
            if name not in strict_agents:
                return result
            if isinstance(ws_results, list) and len(ws_results) >= 1:
                return result   # has real WebSearch results
            if ws_attempt < MAX_WEBSEARCH_RETRIES:
                # Re-prompt with hardened WebSearch enforcement
                _emit("phase1", name, f"retry_ws_attempt_{ws_attempt+1}")
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
                # Final attempt failed — mark as failed (do NOT silently accept)
                result["_websearch_warning"] = (
                    f"⚠⚠ {name}: websearch_results EMPTY after {MAX_WEBSEARCH_RETRIES+1} attempts — "
                    f"output flagged as POSSIBLE_HALLUCINATION"
                )
                result["_failed"] = True
                result["confidence"] = min(result.get("confidence", 0.5), 0.3)   # cap confidence
        return result

    with ThreadPoolExecutor(max_workers=1) as ex:
        # Phase 1: WebSearch + WebFetch enabled → longer timeout (was 180s, now 420s = 7 min)
        # Each agent runs 2-3 WebSearch calls (≈30-60s each) + LLM synthesis
        fut_map = {ex.submit(_phase1_call_with_websearch_enforcement, name, p): name
                    for name, p in phase1_prompts.items()}
        for fut in as_completed(fut_map):
            name = fut_map[fut]
            try:
                result = fut.result()
                phase1[name] = result
                if result.get("_failed"):
                    _emit("phase1", name, "ok_websearch_warned")
                else:
                    _emit("phase1", name, "ok")
            except Exception as e:
                phase1_errors[name] = str(e)[:300]
                # Provide stub so phase 2/3 still run
                phase1[name] = {"agent": name, "rating": "MIXED", "confidence": 0.0,
                                "narrative": f"[agent failed: {str(e)[:120]}]",
                                "key_signals": [], "biggest_risk": "", "biggest_opportunity": "",
                                "websearch_queries": [], "websearch_results": []}
                _emit("phase1", name, "fail")

    # Phase 2 — coherence check (always runs)
    _emit("phase2", "coherence_debater", "started")
    try:
        phase2 = _run_claude(_coherence_prompt(phase1), 120, 2)
        _emit("phase2", "coherence_debater", "ok")
    except Exception as e:
        phase2 = {"coherent": True, "dominant_signal": "(phase2 failed)",
                  "contested_areas": [], "reasoning": str(e)[:300]}
        _emit("phase2", "coherence_debater", "fail")

    # Phase 3 — dual synthesis (parallel)
    _emit("phase3", "synthesis", "started")
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut_n = ex.submit(_run_claude, _synthesis_prompt(phase1, phase2, snap, "neutral"), 180, 2)
        fut_a = ex.submit(_run_claude, _synthesis_prompt(phase1, phase2, snap, "averse"),  180, 2)
        try:
            syn_n = fut_n.result()
            _emit("phase3", "synthesis_neutral", "ok")
        except Exception as e:
            syn_n = {"regime_tag": snap['regime_tag'], "confidence": 0.0,
                     "narrative": f"[neutral synthesis failed: {str(e)[:120]}]",
                     "historical_analog": "", "watch_triggers": [],
                     "cross_panel_coherence_score": 0.0, "key_risks": []}
            _emit("phase3", "synthesis_neutral", "fail")
        try:
            syn_a = fut_a.result()
            _emit("phase3", "synthesis_averse", "ok")
        except Exception as e:
            syn_a = {"regime_tag": snap['regime_tag'], "confidence": 0.0,
                     "narrative": f"[averse synthesis failed: {str(e)[:120]}]",
                     "historical_analog": "", "watch_triggers": [],
                     "cross_panel_coherence_score": 0.0, "key_risks": []}
            _emit("phase3", "synthesis_averse", "fail")

    # Phase 4 — Action Selector (picks + GICS sector scores + themes)
    # 8-min timeout — output is 80 picks + 11 GICS scores + 10 theme rankings
    # ≈ 15-20k generated tokens. 240s was tight under API congestion.
    _emit("phase4", "action_selector", "started")
    try:
        # H2: pass AVERSE synthesis so action_selector can pick hedges
        action = _run_claude(_action_selector_prompt(phase1, phase2, syn_n, snap, syn_a), 480, 2)
        _emit("phase4", "action_selector", "ok")
    except Exception as e:
        action = {
            "long_stocks": [], "long_etfs": [], "short_stocks": [], "short_etfs": [],
            "sector_scores": [], "top_themes": [], "bottom_themes": [],
            "_error": str(e)[:300],
        }
        _emit("phase4", "action_selector", "fail")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 5 — ITERATIVE SWARM (5-Round Convergent)
    # ═══════════════════════════════════════════════════════════════════
    # PM ↔ Trading + Risk loop, max 5 rounds, terminates when:
    #   Δpicks < 10% between rounds (convergence threshold)
    # ═══════════════════════════════════════════════════════════════════
    # ── Iteration constants (overfitting + survival-bias aware) ──
    # Per-ticker debate is the sole path; legacy 5-round batch removed (2026-06-23).
    ITERATION_MAX_ROUNDS = 1
    # Adaptive convergence threshold by regime (replaces fixed 0.20)
    _regime_for_adaptive = (snap.get("regime_tag") or "") if isinstance(snap, dict) else ""
    CONVERGENCE_DELTA_THRESHOLD = adaptive_convergence_threshold(_regime_for_adaptive, base=0.20)
    PIN_MAX_AGE = 3                      # Fix 4: max 3 rounds before forced re-evaluation (survival bias prevention)
    PIN_RE_AUDIT_PROB = 0.20             # Fix 4: 20% chance to unpin age≥2 picks (random sampling)
    WILDCARD_PER_ROUND = 2               # Fix 5: 2 wildcard candidates injected per round (local optima escape)
    ITERATION_TIME_BUDGET_SEC = 60 * 60  # T1 FIX: 1 hour total iteration budget (prevent 4hr runaway)
    _iter_start_time = time.time()

    iteration_history: list[dict] = []
    pinned_picks: dict = {}              # ticker → {"age": n, "horizon": h, "bucket": b}
    rejected_pool: set = set()           # tickers PM considered+rejected (kept for memory, can be re-added)
    # Phase 5: Pareto front tracker for multi-objective best-pick selection across rounds
    pareto_tracker = ParetoFrontTracker()

    import random as _rand
    _rng = _rand.Random(42)              # deterministic for reproducibility

    def _aggregate_objection_patterns(obj: dict, scan_data: list = None) -> str:
        """Fix 2-lite: Aggregate per-pick objections into sector/factor patterns."""
        from collections import Counter
        patterns = []
        for h, issues in obj.items():
            if not issues: continue
            # Count by issue type
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

    def _audit_pinned(round_n: int) -> tuple[list, list]:
        """Fix 4 survival-bias mitigation:
        - Age all pinned picks
        - Force unpin: age >= PIN_MAX_AGE
        - Random re-audit: age >= 2 with PIN_RE_AUDIT_PROB probability
        Returns: (kept_pins, released_pins) for memory/audit
        """
        if not pinned_picks:
            return [], []
        released = []
        for ticker in list(pinned_picks.keys()):
            entry = pinned_picks[ticker]
            entry["age"] += 1
            if entry["age"] >= PIN_MAX_AGE:
                # Force unpin — back to subject of full evaluation
                released.append({"ticker": ticker, "reason": "max_age", **entry})
                del pinned_picks[ticker]
            elif entry["age"] >= 2 and _rng.random() < PIN_RE_AUDIT_PROB:
                # Random sampling: 20% of age≥2 picks lose pin
                released.append({"ticker": ticker, "reason": "random_audit", **entry})
                del pinned_picks[ticker]
        return list(pinned_picks.keys()), released

    def _update_pinned_after_round(pm_horizons: dict, objections: dict) -> int:
        """Fix 4: Pin picks that received NO objections this round.
        Returns count of newly pinned.
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
                    # No objections + not already pinned → pin
                    pinned_picks[t] = {"age": 0, "horizon": h, "bucket": bk}
                    newly_pinned += 1
        return newly_pinned

    def _build_wildcards(snap_data: dict, current_pool_tickers: set, count: int = 2) -> list:
        """Fix 5: Inject wildcard candidates from outside current consideration pool.
        Prevents PM from being locked in local optima (overfitting prevention).
        Selects from full candidate pool, randomly, excluding tickers in current/pinned/rejected.
        """
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
                # Randomly sample (deterministic seed for reproducibility)
                sampled = _rng.sample(outside, min(count, len(outside)))
                for c in sampled:
                    wildcards.append({"pool": pool_key, **c})
        return wildcards

    def _build_iteration_context(round_n: int, current_picks: set, snap_data: dict) -> str:
        """Builds enriched context: pinned + memory + wildcards + Pareto framing."""
        ctx_lines = []

        # ── Audit pinned (Fix 4) ──
        kept_pins, released_pins = _audit_pinned(round_n)
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

        # ── Sequential Memory (Fix 3) ──
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

        # ── Rejected pool reminder (Fix 3 bias prevention) ──
        if rejected_pool:
            sample = list(rejected_pool)[:15]
            ctx_lines.append(f"\n═══ 이전 거절 후보 — 재고 가능 (단순 누락 방지) ═══")
            ctx_lines.append(f"  {', '.join(sample)}")
            ctx_lines.append("  ↑ 거절 사유가 더 이상 유효하지 않다면 다시 검토 가능")

        # ── Wildcards (Fix 5 overfitting prevention) ──
        wildcards = _build_wildcards(snap_data, current_picks, WILDCARD_PER_ROUND)
        if wildcards:
            ctx_lines.append(f"\n═══ ⚡ WILDCARD 후보 (overfitting 방지 — 외부 풀에서 random injection) ═══")
            ctx_lines.append("아래는 통상 후보 풀 밖에서 random sampling된 종목입니다.")
            ctx_lines.append("필수 채택은 아니지만, 검토 후 가치 있다고 판단되면 추가 가능.")
            for w in wildcards[:6]:
                ctx_lines.append(
                    f"  ⚡ {w.get('ticker')} (comp {w.get('composite','?')}, "
                    f"{w.get('classification','?')}, {(w.get('sector') or '')[:14]})"
                )

        # ── Pareto framing (Fix 5) ──
        ctx_lines.append(f"\n═══ TRADE-OFF FRAMING (Pareto-aware) ═══")
        ctx_lines.append("Trading WAIT/SKIP + Risk CAUTION/REJECT은 trade-off를 명시:")
        ctx_lines.append("  (a) 유지 + size 축소 (timing risk accept)")
        ctx_lines.append("  (b) 교체 → 동일 sector 내 (concentration risk 유지)")
        ctx_lines.append("  (c) 교체 → 다른 sector (diversification 효과)")
        ctx_lines.append("각 픽마다 (a/b/c) 명시적 선택 + 근거 — 단순 swap만 반복하지 말 것.")

        return "\n".join(ctx_lines)


    def _extract_tickers(pm_horizons: dict) -> set:
        out = set()
        for h, hd in (pm_horizons or {}).items():
            for bk, picks in (hd or {}).items():
                for p in picks or []:
                    if p.get("ticker"):
                        out.add((h, bk, p["ticker"]))
        return out

    def _compute_delta(prev_set: set, new_set: set) -> float:
        if not prev_set and not new_set:
            return 0.0
        if not prev_set:
            return 1.0
        sym = prev_set ^ new_set
        union = prev_set | new_set
        return len(sym) / len(union) if union else 0.0

    def _aggregate_objections(pm_horizons: dict) -> dict:
        """Trading + Risk verdicts → structured objections per pick for next round."""
        obj = {"tactical": [], "core": [], "strategic": []}
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

    def _fmt_objections_for_prompt(obj: dict) -> str:
        if not obj or not any(obj.values()):
            return ""
        lines = ["\n═══ PREVIOUS ROUND OBJECTIONS (반드시 반영) ═══"]
        for h in ("tactical", "core", "strategic"):
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

    prev_tickers: set = set()
    pm_output: dict = {}
    converged = False
    converged_at_round = None
    prev_pm_horizons: dict = {}   # ← snapshot of last successful PM output (for fail-recovery)

    # NEW: Track best round snapshot (lowest objections + non-empty picks)
    # If iteration degrades, we use BEST round's picks instead of last round's defensive WATCH state
    best_round_snapshot: dict = {}
    best_round_score: float = float("inf")   # lower = better (fewer objections, more picks)

    for round_n in range(1, ITERATION_MAX_ROUNDS + 1):
        # T1 FIX: Time budget check — abort iteration if cumulative time exceeded
        elapsed = time.time() - _iter_start_time
        if elapsed > ITERATION_TIME_BUDGET_SEC:
            _emit("phase5_iter", f"time_budget_exhausted_at_r{round_n}",
                  f"elapsed {int(elapsed)}s > {ITERATION_TIME_BUDGET_SEC}s — aborting iteration")
            break

        _emit("phase5_iter", f"round_{round_n}_start",
              f"elapsed={int(elapsed)}s/{ITERATION_TIME_BUDGET_SEC}s")
        prev_round_objections = (
            iteration_history[-1].get("objections") if iteration_history else None
        )
        objections_block = _fmt_objections_for_prompt(prev_round_objections or {})
        # Aggregate pattern (sector/factor-level objection insight) — Fix 2-lite
        if prev_round_objections:
            patterns_block = _aggregate_objection_patterns(prev_round_objections)
            if patterns_block:
                objections_block += "\n\n═══ OBJECTION PATTERNS (sector/factor 집계) ═══\n" + patterns_block

        # Build bias-aware iteration context (Fix 3+4+5)
        current_pool_tickers = prev_tickers  # set of (h, bucket, ticker) tuples
        current_ticker_only = {t[2] for t in current_pool_tickers}
        ctx_block = _build_iteration_context(round_n, current_ticker_only, snap) if round_n > 1 else ""

        # ─── Phase 5: PM Agent (with previous round's objections + bias-aware context) ──
        try:
            empty_bucket = {"long_stocks": [], "long_etfs": [], "short_stocks": [], "short_etfs": []}
            horizon_results: dict = {}
            horizon_errors:  dict = {}

            def _pm_prompt_with_obj(h: str) -> str:
                base = _pm_horizon_prompt(phase1, phase2, syn_n, syn_a, action, snap, h)
                if round_n > 1 and (objections_block or ctx_block):
                    return base + "\n\n" + objections_block + "\n" + ctx_block
                return base

            # Reduced from 3 → 2 to avoid Max plan concurrent session lock ("Not logged in")
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut_map = {
                    ex.submit(_run_claude, _pm_prompt_with_obj(h), 600, 2): h
                    for h in ("tactical", "core", "strategic")
                }
                for fut in as_completed(fut_map):
                    h = fut_map[fut]
                    _emit("phase5", f"pm_{h}_r{round_n}", "started")
                    try:
                        horizon_results[h] = fut.result()
                        _emit("phase5", f"pm_{h}_r{round_n}", "ok")
                    except Exception as e:
                        horizon_errors[h] = str(e)[:300]
                        horizon_results[h] = dict(empty_bucket)
                        _emit("phase5", f"pm_{h}_r{round_n}", "fail")

            core = horizon_results.get("core", {})
            tactical = horizon_results.get("tactical", {})
            strategic = horizon_results.get("strategic", {})
            pm_output = {
                "pm_commentary":    core.get("pm_commentary", "") or pm_output.get("pm_commentary", ""),
                "portfolio_thesis": core.get("portfolio_thesis", "") or pm_output.get("portfolio_thesis", ""),
                "horizons": {
                    "tactical": {b: tactical.get(b, []) for b in empty_bucket},
                    "core":     {b: core.get(b, [])     for b in empty_bucket},
                    "strategic":{b: strategic.get(b, []) for b in empty_bucket},
                },
                "phase4_drops": core.get("phase4_drops", []) or [],
                "hedge_pairs":  core.get("hedge_pairs", [])  or [],
                "risk_budget":  core.get("risk_budget", [])  or [],
            }
            if horizon_errors:
                pm_output["_horizon_errors"] = horizon_errors
            # If ALL 3 horizons failed → use previous round's picks (no progress this round)
            if not any(pm_output["horizons"][h]["long_stocks"] for h in ("tactical","core","strategic")):
                if prev_pm_horizons:
                    # Restore prev round's picks — this round is a "no-op"
                    pm_output["horizons"] = {h: {b: list(prev_pm_horizons.get(h, {}).get(b, []))
                                                  for b in empty_bucket}
                                              for h in ("tactical","core","strategic")}
                    pm_output["_round_failed_recovered"] = True
                    _emit("phase5", f"pm_agent_r{round_n}", "fail_recovered_from_prev")
                else:
                    raise RuntimeError(f"All 3 horizon calls failed: {horizon_errors}")
            else:
                # Snapshot for future recovery
                prev_pm_horizons = {h: {b: list(pm_output["horizons"][h].get(b, []))
                                          for b in empty_bucket}
                                      for h in ("tactical","core","strategic")}
                _emit("phase5", f"pm_agent_r{round_n}", "ok")
        except Exception as e:
            if prev_pm_horizons:
                # All-horizons failed but we have prev round → reuse
                empty_bucket = {"long_stocks": [], "long_etfs": [], "short_stocks": [], "short_etfs": []}
                if "horizons" not in pm_output:
                    pm_output["horizons"] = {h: {b: list(prev_pm_horizons.get(h, {}).get(b, []))
                                                  for b in empty_bucket}
                                              for h in ("tactical","core","strategic")}
                pm_output["_round_failed_recovered"] = True
                _emit("phase5", f"pm_agent_r{round_n}", "fail_recovered_from_prev")
            else:
                empty_bucket = {"long_stocks": [], "long_etfs": [], "short_stocks": [], "short_etfs": []}
                if not pm_output.get("horizons"):
                    pm_output = {
                        "portfolio_thesis": f"[PM agent failed: {str(e)[:200]}]",
                        "pm_commentary": "",
                        "horizons": {"tactical": dict(empty_bucket),
                                     "core":     dict(empty_bucket),
                                     "strategic":dict(empty_bucket)},
                        "phase4_drops": [], "hedge_pairs": [], "risk_budget": [],
                        "_error": str(e)[:300],
                    }
                _emit("phase5", f"pm_agent_r{round_n}", "fail")

        # ─── Phase 5.5: Trading Timing (per-round) ───
        _has_picks_round = any(
            (pm_output.get("horizons", {}).get(h, {}).get("long_stocks") or [])
            for h in ("tactical", "core", "strategic")
        )
        if _has_picks_round:
            # Option C: Per-Ticker Debate Engine (replaces legacy 5.5 + 5.55 + 5.6a)
            # Each ticker gets its own 3-round mini-debate (Trading + Risk + Critic
            # in R1, Revise in R2, Arbiter in R3). Small focused LLM contexts
            # eliminate universal-CAUTION cascade from legacy batched 240-ticker calls.
            _emit("phase5_pt_debate", f"pt_debate_r{round_n}", "started")
            try:
                macro_ctx = (snap.get("macro_summary") or
                              json.dumps(snap.get("phase1") or {})[:1000])
                pt_horizons = run_per_ticker_debate(
                    pm_horizons=pm_output.get("horizons", {}),
                    regime_tag=snap.get("regime_tag", "—"),
                    macro_context=macro_ctx,
                    run_claude_fn=_run_claude,
                    _emit_fn=_emit,
                )
                pm_output["horizons"] = pt_horizons
                pm_output["per_ticker_debate_summary"] = summarize_debate_results(pt_horizons)
                _emit("phase5_pt_debate", f"pt_debate_r{round_n}",
                      f"ok summary={pm_output['per_ticker_debate_summary']}")
            except Exception as e:
                pm_output["per_ticker_debate_error"] = str(e)[:300]
                _emit("phase5_pt_debate", f"pt_debate_r{round_n}", f"fail: {str(e)[:100]}")

        # ─── Convergence check (end of round) ───
        cur_tickers = _extract_tickers(pm_output.get("horizons", {}))
        delta = _compute_delta(prev_tickers, cur_tickers) if round_n > 1 else 1.0
        objections = _aggregate_objections(pm_output.get("horizons", {}))
        n_obj = sum(len(v) for v in objections.values())

        # ── Sequential memory tracking (Fix 3) ──
        prev_ticker_set = {t[2] for t in prev_tickers}
        cur_ticker_set  = {t[2] for t in cur_tickers}
        kept    = sorted(prev_ticker_set & cur_ticker_set)
        added   = sorted(cur_ticker_set - prev_ticker_set)
        removed = sorted(prev_ticker_set - cur_ticker_set)
        # Add removed tickers to rejected_pool (so they can be reconsidered later)
        rejected_pool.update(removed)

        # ── Pin no-objection picks (Fix 4) ──
        n_newly_pinned = _update_pinned_after_round(pm_output.get("horizons", {}), objections)

        iteration_history.append({
            "round": round_n,
            "n_tickers": len(cur_tickers),
            "delta": round(delta, 4),
            "n_objections": n_obj,
            "objections": objections,
            "kept_tickers": kept,
            "added_tickers": added,
            "removed_tickers": removed,
            "n_pinned": len(pinned_picks),
            "n_newly_pinned": n_newly_pinned,
            "n_rejected_pool": len(rejected_pool),
        })

        # ── Phase 5: Pareto front tracking — record this round's picks ──
        # Each ticker's "best version" (across rounds) is kept on the Pareto front.
        # Used at iteration end to surface Pareto-optimal portfolio if best_round
        # snapshot falls short.
        try:
            pareto_tracker.add_round(round_n, pm_output.get("horizons", {}))
        except Exception:
            pass

        # ── Track BEST round (lowest objections + has picks) ──
        # Score = n_objections - bonus for having more picks
        # → prefers round with FEW objections AND MANY non-degraded picks
        if len(cur_tickers) > 0:
            round_score = n_obj - len(cur_tickers) * 0.1   # lower = better
            if round_score < best_round_score:
                best_round_score = round_score
                # Deep copy of current horizons + iteration metadata
                import copy as _copy
                best_round_snapshot = {
                    "round": round_n,
                    "n_objections": n_obj,
                    "n_tickers": len(cur_tickers),
                    "horizons": _copy.deepcopy(pm_output.get("horizons", {})),
                    "pm_commentary": pm_output.get("pm_commentary", ""),
                    "portfolio_thesis": pm_output.get("portfolio_thesis", ""),
                    "phase4_drops": pm_output.get("phase4_drops", []),
                    "hedge_pairs": pm_output.get("hedge_pairs", []),
                    "risk_budget": pm_output.get("risk_budget", []),
                    "trading_timing": pm_output.get("trading_timing", {}),
                    "risk_llm": pm_output.get("risk_llm", {}),
                }
                _emit("phase5_iter", f"best_snapshot_r{round_n}",
                      f"obj={n_obj} tickers={len(cur_tickers)} score={round_score:.1f}")
        _emit("phase5_iter", f"round_{round_n}_end",
              f"Δ={delta:.1%} obj={n_obj} pins={len(pinned_picks)} rejected={len(rejected_pool)}")

        # ── Convergence check + Cross-Validation (Fix 1 with quality gate) ──
        # PRE-GATE: If this round was a failure-recovery (Δ=0 because prev picks restored),
        # do NOT treat it as convergence. Empirical bug observed: Round 4 PM all-fails →
        # prev_pm_horizons restored → Δ=0% → falsely converged at R4. Block this.
        if pm_output.get("_round_failed_recovered"):
            iteration_history[-1]["round_failed_recovered"] = True

        if round_n > 1 and delta < CONVERGENCE_DELTA_THRESHOLD:
            # T1 FIX: False convergence block — Δ=0% caused by failure-recovery is not convergence
            this_round_was_failure = pm_output.get("_round_failed_recovered", False)
            if this_round_was_failure and delta < 0.05:
                _emit("phase5_iter", f"r{round_n}_false_convergence_blocked",
                      f"Δ={delta:.1%} but PM all-failed this round — not real convergence")
                # Fall through to "consecutive failures" check below
            else:
                # Quality gate: objection trend should be improving, not worsening
                prev_n_obj = iteration_history[-2].get("n_objections", 0)
                obj_worsened = n_obj > prev_n_obj * 1.3   # >30% more objections = warning sign
                # T1 FIX EXTENDED: If absolute obj count is suspiciously high (>= n_tickers),
                # every pick has at least one objection — universal pessimism, not convergence
                universal_pessimism = n_obj >= len(cur_tickers) * 0.9

                if not obj_worsened and not universal_pessimism:
                    converged = True
                    converged_at_round = round_n
                    _emit("phase5_iter", f"converged_r{round_n}",
                          f"Δ={delta:.1%}<{CONVERGENCE_DELTA_THRESHOLD:.0%} (quality OK)")
                    break
                elif universal_pessimism:
                    _emit("phase5_iter", f"r{round_n}_universal_pessimism",
                          f"Δ={delta:.1%} but {n_obj}/{len(cur_tickers)} picks objected — system stress, not convergence")
                else:
                    _emit("phase5_iter", f"r{round_n}_not_converged",
                          f"Δ={delta:.1%} but obj worsened {prev_n_obj}→{n_obj} (overfitting risk)")

        # If round-failure recovered (no progress vs prev) AND there have been ≥2 consecutive
        # recoveries — treat as converged-by-stability and stop wasting API calls.
        if pm_output.get("_round_failed_recovered"):
            consecutive_failed = sum(
                1 for h in iteration_history[-2:] if h.get("round_failed_recovered")
            )
            iteration_history[-1]["round_failed_recovered"] = True
            if consecutive_failed >= 1:   # this is 2nd in a row
                converged = True
                converged_at_round = round_n
                _emit("phase5_iter", f"converged_r{round_n}",
                      "early-stop after consecutive PM failures (prev picks preserved)")
                break

        prev_tickers = cur_tickers

    # ── Use BEST round's picks if final round is degraded ──
    # Degraded = final round has 30%+ more objections than best round
    last_round_obj = iteration_history[-1].get("n_objections", 0) if iteration_history else 0
    best_n_obj = best_round_snapshot.get("n_objections", last_round_obj)
    degradation_threshold = 1.5   # last round has 50%+ more obj than best → use best
    used_best_round = False
    if best_round_snapshot and last_round_obj > best_n_obj * degradation_threshold:
        used_best_round = True
        # Restore best round's PM picks + agent outputs
        for k in ("horizons", "trading_timing", "risk_llm"):
            if k in best_round_snapshot:
                pm_output[k] = best_round_snapshot[k]
        _emit("phase5_iter", "used_best_round",
              f"R{best_round_snapshot.get('round')} obj={best_n_obj} (vs last R{round_n} obj={last_round_obj})")

    # Persist iteration metadata in pm_output
    pm_output["iteration"] = {
        "history": iteration_history,
        "converged": converged,
        "converged_at_round": converged_at_round if converged else round_n,
        "max_rounds": ITERATION_MAX_ROUNDS,
        "convergence_threshold": CONVERGENCE_DELTA_THRESHOLD,
        "best_round": best_round_snapshot.get("round") if best_round_snapshot else None,
        "used_best_round": used_best_round,
        "last_round_n_objections": last_round_obj,
        "best_round_n_objections": best_n_obj,
    }

    # Phase 5: persist Pareto front summary
    try:
        pm_output["pareto_summary"] = pareto_tracker.summary()
    except Exception as e:
        pm_output["pareto_summary"] = {"_error": str(e)[:200]}

    # ─── Phase 5.6a — Debate Synthesizer (runs ONCE on final converged picks) ───
    _has_picks = any(
        (pm_output.get("horizons", {}).get(h, {}).get("long_stocks") or [])
        for h in ("tactical", "core", "strategic")
    )
    if _has_picks:
        # Per-Ticker Debate already populated debate_synthesis on each pick during
        # the iteration loop above. Legacy batch debate_synthesizer removed.
        _emit("phase5_6a", "debate", "skipped_per_ticker_mode")

        # ─── Detect degraded Debate Synthesizer output (uniform SOLO/WATCH) ───
        # If 80%+ of picks have tier=SOLO + final_decision=WATCH → debate failed silently
        # Override with PM Agent's direct conviction (composite-based)
        all_picks_check = []
        for h in ("tactical", "core", "strategic"):
            for bk in ("long_stocks", "long_etfs"):
                for p in pm_output.get("horizons", {}).get(h, {}).get(bk, []) or []:
                    ds = p.get("debate_synthesis") or {}
                    all_picks_check.append((p, ds.get("tier"), ds.get("final_decision")))

        if all_picks_check:
            n_solo_watch = sum(1 for _, t, fd in all_picks_check if t == "SOLO" and fd == "WATCH")
            solo_watch_ratio = n_solo_watch / len(all_picks_check)
            if solo_watch_ratio > 0.8:
                # Debate Synthesizer collapsed — override with PM Agent's composite-based conviction
                _emit("phase5_6a", "debate_degraded_override",
                      f"{n_solo_watch}/{len(all_picks_check)} picks SOLO/WATCH — overriding with PM conviction")
                pm_output["_debate_degraded"] = True
                for p, _, _ in all_picks_check:
                    comp = float(p.get("composite") or 0)
                    cls = (p.get("classification") or "")
                    cls_str = cls if isinstance(cls, str) else ""
                    # Conviction tier based on PM-side signals
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
        # Takes per-ticker debate verdicts and composes final portfolio:
        # sector cap, adaptive regime-aware budget, conviction sort, EXCLUDE drop.
        _emit("phase5b_compose", "portfolio_composer", "started")
        try:
            composition = compose_portfolio(
                pm_horizons=pm_output.get("horizons", {}),
                regime_tag=snap.get("regime_tag", ""),
                _emit_fn=_emit,
            )
            pm_output["horizons"] = composition["horizons"]
            pm_output["portfolio_composition"] = composition["metadata"]
            pm_output["portfolio_composition_summary"] = summarize_composition(composition)
            _emit("phase5b_compose", "portfolio_composer",
                  f"ok {pm_output['portfolio_composition_summary']}")
        except Exception as e:
            pm_output["portfolio_composition"] = {"_error": str(e)[:300]}
            _emit("phase5b_compose", "portfolio_composer", f"fail: {str(e)[:100]}")

        # ── Phase 5.6 — Position State Machine (hysteresis + alerts) ──
        # Stateful tracking eliminates daily BUY→NEUTRAL→BUY flip-flops.
        # Each (ticker, horizon) progresses through:
        #   PROSPECTING → ENTERED → HOLDING → EXIT_PENDING → EXITED
        # Alerts surface only on meaningful state transitions.
        _emit("phase5_6", "position_state", "started")
        try:
            state_summary = apply_state_machine(pm_output.get("horizons", {}))
            pm_output["position_state_summary"] = state_summary
            _emit("phase5_6", "position_state", "ok")
        except Exception as e:
            pm_output["position_state_summary"] = {"_error": str(e)[:300]}
            _emit("phase5_6", "position_state", "fail")

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "snapshot": {
            "as_of": snap["as_of"], "total_tickers": snap["total_tickers"],
            "regime_tag_deterministic": snap["regime_tag"],
            "cd_gap": snap["cd_gap"], "gv_gap": snap["gv_gap"],
            "oer_avg": snap["oer_avg"],
        },
        "phase0_facts":  phase0_facts,    # H4: shared evidence pool
        "phase1": phase1,
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
