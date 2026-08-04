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

NEW (≈20 unified, authoritative-source queries):
  Phase 0 fact_collector → executes the DEFAULT_QUERY_PLAN below
                          → produces shared evidence_pool

  Macro coverage (all anchored to Trading Economics / FRED / OECD / S&P Global /
  official central banks ECB·BOJ·BOE·BCB·BOK):
    • Central-bank policy rates: Fed, ECB, BOE, BOJ, BOK, Brazil (Selic/BCB)
    • PMI: US ISM + S&P Global + China (Caixin/NBS) manufacturing & services
    • Growth/prices/labor (global): GDP, inflation (CPI), employment — US + Eurozone/
      Japan/UK/Korea/China/India
    • Fiscal & external: budget/fiscal balance, trade balance, exports/imports
  Cross-asset coverage:
    • Yield curve (UST 10Y/2Y), credit spreads (IG/HY OAS), DXY + FX, global yields
      (JGB/Bund/Gilt), commodities (oil/gold/copper), VIX
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
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable


# 뉴스 2쿼리만 유지 — 매크로/크로스에셋/센티먼트 제거 (scan cache가 이미 커버)
DEFAULT_QUERY_PLAN = [
    # ── Market news: 시장 주요 뉴스 ──
    {"id": "market_today",    "query": "stock market biggest story today {asof} site:cnbc.com OR site:reuters.com OR site:finance.yahoo.com"},
    # ── Korea: 코스피/코스닥 뉴스 ──
    {"id": "korea_market",    "query": "코스피 코스닥 삼성전자 SK하이닉스 주요 뉴스 {asof} site:kedglobal.com OR site:tradingeconomics.com"},
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


def _collect_shard(asof: str, shard_plan: list, run_claude_fn: Callable, timeout: int) -> dict:
    """Run one fact-collection shard (a subset of the query plan) as a single call."""
    prompt = _build_fact_collector_prompt(asof, shard_plan)
    result = run_claude_fn(prompt, timeout, 1)
    if isinstance(result, dict) and result.get("evidence_pool"):
        return result
    return {"_failed": True, "evidence_pool": []}


def run_fact_collector(
    asof: str,
    run_claude_fn: Callable,
    _emit_fn: Optional[Callable] = None,
    query_plan: list = None,
    timeout: int = 600,
    max_workers: int = None,
) -> dict:
    """Execute Phase 0 fact collection — parallel-sharded across collectors.

    The query plan is split into `max_workers` shards that run CONCURRENTLY
    (bounded by the subprocess semaphore), then their evidence pools are merged.
    Identical queries/output to the single-call path — only wall-clock drops
    (e.g. 20 queries serial ~8.5 min → 3 shards ~3 min). max_workers<=1 or a tiny
    plan falls back to a single call.

    Args:
        asof: as-of date for queries (YYYY-MM-DD)
        run_claude_fn: claude -p subprocess wrapper
        _emit_fn: progress emission
        query_plan: optional override of DEFAULT_QUERY_PLAN
        timeout: per-call subprocess timeout (capped per-shard)
        max_workers: shard count (default: CLAUDE_MAX_CONCURRENCY env, else 3)

    Returns:
        {as_of, n_queries_executed, evidence_pool, summary} or
        {_failed, _failure_reason} on failure.
    """
    def _emit(phase: str, status: str):
        if _emit_fn:
            try: _emit_fn(phase, "fact_collector", status)
            except Exception: pass

    plan = query_plan if query_plan is not None else DEFAULT_QUERY_PLAN
    if max_workers is None:
        try: max_workers = max(1, int(os.environ.get("CLAUDE_MAX_CONCURRENCY", "2")))
        except (TypeError, ValueError): max_workers = 2

    _emit("phase0_fact", "started")

    # ── Single-call path (very small plan only) ──
    # max_workers<=1 조건 제거: 단일 콜에 22쿼리가 몰리면 900s timeout에 걸리기 쉬움.
    # 항상 sharding → 각 shard 프롬프트가 작아져 timeout 위험 감소.
    # (max_workers=1이면 shard가 직렬 실행되지만 각 call이 더 짧아 fault isolation 효과)
    if len(plan) <= 6:
        try:
            result = run_claude_fn(_build_fact_collector_prompt(asof, plan), timeout, 2)
            if not isinstance(result, dict) or not result.get("evidence_pool"):
                _emit("phase0_fact", "failed_invalid_output")
                return {"_failed": True, "_failure_reason": "no_evidence_pool", "evidence_pool": []}
            _emit("phase0_fact", f"ok n={len(result.get('evidence_pool',[]))}")
            return result
        except Exception as e:
            _emit("phase0_fact", f"failed: {str(e)[:100]}")
            return {"_failed": True, "_failure_reason": str(e)[:200], "evidence_pool": []}

    # ── Parallel-sharded path ──
    # 2쿼리: 각 shard가 1쿼리씩 → 단일 웹검색 1~2분, timeout 120s로 충분.
    n_shards = min(max_workers, len(plan))
    shards = [plan[i::n_shards] for i in range(n_shards)]
    shard_timeout = min(timeout, 120)   # 쿼리 1개 → 2분이면 충분
    _emit("phase0_fact", f"sharded n_shards={n_shards} ({len(plan)} queries)")

    results: list = [None] * n_shards
    with ThreadPoolExecutor(max_workers=n_shards) as ex:
        fut_map = {ex.submit(_collect_shard, asof, shards[i], run_claude_fn, shard_timeout): i
                   for i in range(n_shards)}
        for fut in as_completed(fut_map):
            i = fut_map[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"_failed": True, "evidence_pool": [], "_err": str(e)[:120]}
            ok = bool(results[i] and results[i].get("evidence_pool"))
            _emit("phase0_fact", f"shard{i+1}/{n_shards} {'ok' if ok else 'empty'}")

    # ── Merge evidence pools ──
    merged_pool: list = []
    n_exec = 0
    summaries: list = []
    n_ok = 0
    for r in results:
        if r and r.get("evidence_pool"):
            merged_pool.extend(r["evidence_pool"])
            n_exec += r.get("n_queries_executed", len(r["evidence_pool"]))
            if r.get("summary"): summaries.append(r["summary"])
            n_ok += 1

    if not merged_pool:
        _emit("phase0_fact", "failed_all_shards")
        return {"_failed": True, "_failure_reason": "all shards empty", "evidence_pool": []}

    _emit("phase0_fact", f"ok n={len(merged_pool)} ({n_ok}/{n_shards} shards)")
    return {"as_of": asof, "n_queries_executed": n_exec, "evidence_pool": merged_pool,
            "summary": " | ".join(summaries), "_shards": n_shards, "_shards_ok": n_ok}


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
