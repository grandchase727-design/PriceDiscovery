# -*- coding: utf-8 -*-
"""fact_collector.py — H4 Improvement: Phase 0 Shared Fact Collection.

================================================================================
PURPOSE
================================================================================

Replaces the legacy "each Phase 1 agent does its own WebSearch" pattern with
a single shared fact-collection pass. Each Phase 1 analyst then reads from
the shared fact pool, eliminating redundant queries.

LEGACY (14 WebSearch calls per swarm):
  macro_analyst       → 2 queries (Fed, PMI)
  cross_asset_analyst → 5 queries (BOJ, ECB, BOK, VIX, etc.)
  news_narrative      → 3 queries (Eng/Kor news)
  flow_momentum       → 2 queries
  sector_theme        → 2 queries

  → 4 agents query "Fed June 2026" with slightly different phrasings
  → Same URLs hit multiple times
  → 14 queries total, ~5-10 unique URLs

NEW (8-10 unified queries):
  Phase 0 fact_collector → executes 8-10 high-coverage queries
                          → produces shared evidence_pool
  Phase 1 agents → no WebSearch tool; consume evidence_pool
                  → faster, deterministic, shared facts

================================================================================
WHEN TO USE
================================================================================

Set USE_PHASE0_FACT_COLLECTOR=True at top of market_leaders_swarm.py to enable.
When enabled:
  - Phase 1 prompts inject evidence_pool block
  - Phase 1 LLM calls run WITHOUT --allowedTools WebSearch
  - Latency: 1 Phase 0 call (~3 min) + 5 fast Phase 1 calls (~1 min each)
            vs legacy 5 slow Phase 1 calls (~3 min each)

================================================================================
"""
from __future__ import annotations

import json
from typing import Optional, Callable


# Curated query plan — 10 queries covering all 5 agent domains
DEFAULT_QUERY_PLAN = [
    # Macro
    {"id": "fed_decision",      "query": "Federal Reserve latest interest rate decision dot plot {asof}"},
    {"id": "us_pmi_jobs",       "query": "US ISM manufacturing services PMI jobs report unemployment {asof}"},
    # Cross-asset
    {"id": "global_rates",      "query": "10-year Treasury yield VIX credit spreads DXY level {asof}"},
    {"id": "boj_ecb_bok",       "query": "BOJ ECB BOK central bank rate decisions {asof}"},
    {"id": "commodities",       "query": "oil gold copper commodities price levels {asof}"},
    # Flow + sector
    {"id": "etf_flows",         "query": "SPY QQQ IWM sector ETF inflows outflows weekly {asof}"},
    {"id": "sector_leadership", "query": "S&P 500 sector leadership rotation YTD {asof}"},
    # News (source-targeted to encourage diverse cross-source coverage)
    {"id": "market_today",      "query": "stock market biggest story today {asof} site:finviz.com OR site:finance.yahoo.com OR site:reuters.com"},
    {"id": "finviz_news",       "query": "finviz news headlines today {asof} pre-market movers analyst chatter"},
    {"id": "korea_market",      "query": "한국 주식시장 코스피 코스닥 주요 뉴스 {asof}"},
    # Sentiment (quantitative — feeds news_narrative + cross_asset agents)
    {"id": "fear_greed_index",  "query": "CNN Fear and Greed Index current value {asof} previous week previous month historical comparison"},
    # Geopolitical
    {"id": "geopolitical",      "query": "geopolitical risk events tariffs sanctions {asof}"},
]


def _build_fact_collector_prompt(asof: str, query_plan: list = None) -> str:
    """Build the single Phase 0 fact-collection prompt."""
    if query_plan is None:
        query_plan = DEFAULT_QUERY_PLAN

    formatted_queries = []
    for q in query_plan:
        qstr = q["query"].format(asof=asof)
        formatted_queries.append(f"  • [{q['id']}]: {qstr}")

    return f"""You are the **FACT COLLECTOR** — Phase 0 of a market intelligence swarm.

Your single job: execute {len(query_plan)} WebSearch queries and return a structured
evidence pool that Phase 1 analysts will consume (saving them from duplicate searches).

═══════════════════════════════════════════════════════════
QUERY PLAN — execute ALL {len(query_plan)} via WebSearch tool
═══════════════════════════════════════════════════════════

{chr(10).join(formatted_queries)}

═══════════════════════════════════════════════════════════
OUTPUT — Aggregated evidence pool
═══════════════════════════════════════════════════════════

For each query, return:
- query: the query string you ran
- topic: one of [macro/cross_asset/sector/flow/news/sentiment/geopolitical]
- findings: 2-4 bullet facts extracted (each with source URL + date if available)
- key_data_points: structured data extracted (numbers, dates, named entities)
- relevant_agents: which Phase 1 agents should use this
  (macro/cross_asset/sector/flow/news — multi-agent tagging encouraged for
   cross-cutting facts like fear_greed which feeds news + cross_asset + flow)

SPECIAL: for fear_greed_index query — fetch from https://edition.cnn.com/markets/fear-and-greed
(or WebSearch fallback). Extract:
  - current_score (0-100)
  - current_label (Extreme Fear / Fear / Neutral / Greed / Extreme Greed)
  - week_ago_score, month_ago_score, year_ago_score (if available)
  - subcomponents (VIX, momentum, breadth, put/call, junk bonds, safe haven, strength)
relevant_agents: ["news_narrative_analyst", "cross_asset_analyst", "flow_momentum_analyst"]

```json
{{
  "as_of": "{asof}",
  "n_queries_executed": <count>,
  "evidence_pool": [
    {{
      "id": "fed_decision",
      "query": "...",
      "topic": "macro",
      "findings": [
        "Fed held rates at 3.50-3.75% on 6/17 (source: CNBC, retrieved 6/22)",
        "Dot plot median raised to 3.8% (vs 3.4% in March)",
        "..."
      ],
      "key_data_points": {{
        "fed_funds_target": "3.50-3.75%",
        "dot_plot_median_2026": 3.8,
        "decision_date": "2026-06-17"
      }},
      "relevant_agents": ["macro_analyst", "cross_asset_analyst"]
    }},
    {{ ... 9 more ... }}
  ],
  "summary": "2-3 sentence overall market snapshot covering all collected facts (한국어)"
}}
```

EXECUTION RULES:
1. Execute each WebSearch query SEPARATELY (don't combine — model loses query context)
2. Prefer authoritative sources (Federal Reserve, BLS, BOJ, ECB releases / WSJ / FT / CNBC / Reuters)
3. If a query returns no results, mark `"findings": []` (do not hallucinate)
4. Tag each finding with relevant_agents so Phase 1 can filter efficiently
5. Total output ≤ 8K tokens — be concise on findings, dense on key_data_points

Return STRICTLY a fenced ```json block."""


def run_fact_collector(
    asof: str,
    run_claude_fn: Callable,
    _emit_fn: Optional[Callable] = None,
    query_plan: list = None,
    timeout: int = 600,
) -> dict:
    """Execute Phase 0 fact collection.

    Args:
        asof: as-of date for queries (YYYY-MM-DD)
        run_claude_fn: claude -p subprocess wrapper
        _emit_fn: progress emission
        query_plan: optional override of DEFAULT_QUERY_PLAN
        timeout: subprocess timeout in seconds (600 = 10 min)

    Returns:
        {as_of, n_queries_executed, evidence_pool, summary} or
        {_failed, _failure_reason} on failure.
    """
    def _emit(phase: str, status: str):
        if _emit_fn:
            try: _emit_fn(phase, "fact_collector", status)
            except Exception: pass

    _emit("phase0_fact", "started")
    prompt = _build_fact_collector_prompt(asof, query_plan)

    try:
        result = run_claude_fn(prompt, timeout, 2)
        if not isinstance(result, dict) or not result.get("evidence_pool"):
            _emit("phase0_fact", "failed_invalid_output")
            return {"_failed": True, "_failure_reason": "no_evidence_pool", "evidence_pool": []}
        _emit("phase0_fact", f"ok n={len(result.get('evidence_pool',[]))}")
        return result
    except Exception as e:
        _emit("phase0_fact", f"failed: {str(e)[:100]}")
        return {"_failed": True, "_failure_reason": str(e)[:200], "evidence_pool": []}


def filter_evidence_for_agent(evidence_pool: list, agent_name: str) -> list:
    """Filter shared evidence pool to items relevant for a specific Phase 1 agent.

    Maps Phase 1 agent name → relevant_agents tag in evidence items.
    """
    if not evidence_pool:
        return []
    # Normalize agent name (e.g. "macro_analyst" → "macro")
    short_name = agent_name.replace("_analyst", "").replace("_momentum","").replace("_narrative","")
    aliases = {
        "macro": ["macro", "macro_analyst"],
        "cross_asset": ["cross_asset", "cross_asset_analyst"],
        "sector_theme": ["sector", "sector_theme", "sector_theme_analyst"],
        "flow": ["flow", "flow_momentum_analyst"],
        "news": ["news", "news_narrative_analyst"],
    }
    relevant_tags = aliases.get(short_name, [short_name])

    out = []
    for item in evidence_pool:
        rel = item.get("relevant_agents") or []
        if any(t in rel for t in relevant_tags):
            out.append(item)
    return out


def format_evidence_for_prompt(filtered: list, max_items: int = 6) -> str:
    """Format filtered evidence into a compact block injected into Phase 1 prompts."""
    if not filtered:
        return "(no relevant evidence in shared pool for this agent)"
    lines = ["═══ SHARED EVIDENCE POOL (Phase 0 collected — DO NOT re-search) ═══"]
    for ev in filtered[:max_items]:
        lines.append(f"\n[{ev.get('id','?')} / {ev.get('topic','?')}]:")
        for f in (ev.get('findings') or [])[:4]:
            lines.append(f"  • {f}")
        kdp = ev.get('key_data_points') or {}
        if kdp:
            kdp_str = " · ".join(f"{k}={v}" for k, v in list(kdp.items())[:8])
            lines.append(f"  DATA: {kdp_str}")
    lines.append("\n═══ END EVIDENCE POOL ═══")
    return "\n".join(lines)
