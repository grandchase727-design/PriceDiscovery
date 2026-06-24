# agents/ — LLM Agent Layer (Claude Max plan only)

LLM-driven enrichment layer that plugs into the Pre-Momentum scoring stack
**without** any out-of-plan API spend. Every LLM/sub-agent/MCP call happens
inside an active Claude Code session under the user's Claude Max plan.

## Hard rule — execution boundary

> **No code in this directory ever calls the Anthropic API directly.**
> No standalone cron scripts. No uvicorn-side LLM calls. No HTTP shim.
> If a piece of code needs an LLM result, it gets it from a JSON cache
> that was written by a previous in-session run.

This is the only way to keep total spend inside the Claude Max plan.

## Files

| File | Pure-Python? | Role |
|---|---|---|
| `claude_finance_bridge.py` | ✓ | Schema/cache/normalization layer over Aiera/FactSet/Daloopa MCP calls. Holds *callables* injected by the session. |
| `llm_catalyst_agent.py` | ✓ | Mutates `cache["results"]` with `catalyst_score_v2`, `qvr_r_v2`, `llm_catalyst_score`, `llm_revision_score`. |
| `conviction_debate.py` | ✓ | Target selection + prompt template + verdict cache I/O for the ConvictionDebate UI panel. |
| `__init__.py` | ✓ | empty marker |

All four files are pure Python — they do no network I/O and call no LLM
themselves. The Claude Code session does all the I/O and passes results in.

## How execution actually happens

### Mode A — Manual (you ask Claude)

You: *"daily refresh + 오늘 Top-10 conviction debate 돌려줘"*

Claude (this Claude Code session):
1. Authenticates Aiera/FactSet/Daloopa MCPs if not already (one-time).
2. Loads `.scan_cache.pkl`.
3. For each top-N ticker, calls the MCP tools and wires the results into
   the bridge via `bridge.inject_callables(...)`, then runs
   `enrich_with_llm_agents(cache)`.
4. Selects ConvictionDebate targets via `select_debate_targets(cache, 10)`.
5. For each target, calls the **`market-researcher` sub-agent** with a
   prompt built by `build_debate_prompt(...)`.
6. Parses each sub-agent's reply via `parse_verdict_text(...)` and
   persists with `save_all_verdicts(...)`.
7. Writes the enriched scan cache back to disk.

All of step 5 (sub-agent calls) is billed against your Claude Max plan.
No external API spend.

### Mode B — Scheduled (Claude Code's `CronCreate`)

You ask Claude *once*: *"매일 장 마감 후 1시간 뒤에 위 daily refresh를
자동으로 돌리도록 cron 설정해줘."*

Claude calls `CronCreate(...)` (built into Claude Code, billed under your
plan). At each fire time, a fresh Claude Code session starts under your
account, executes the same Mode A flow, and exits. No external spend.

## Authentication (one-time, in-session)

Run once per environment from within a session:

```
mcp__plugin_financial-analysis_aiera__authenticate()
mcp__plugin_financial-analysis_factset__authenticate()
mcp__plugin_financial-analysis_daloopa__authenticate()
```

Each returns a browser URL — complete the OAuth flow once and the tools
become callable from any future session under the same account.

## What the dashboard sees (the server side)

The uvicorn FastAPI server (`api.py`) **does not call any LLM**. It only
reads the JSON caches that the in-session runs produce, and serves them
through endpoints:

- `/api/table` — the existing per-ticker JSON, now optionally enriched
  with `_v2` fields from `enrich_with_llm_agents` (if a session ran it).
- `/api/conviction-debate` — new — serves `.conviction_debate_cache.json`
  via `conviction_debate.summary_for_dashboard()`. Add the endpoint with:

```python
# api.py — pure cache-serve, no LLM call
from agents.conviction_debate import summary_for_dashboard

@app.get("/api/conviction-debate")
def get_conviction_debate():
    return summary_for_dashboard()
```

That endpoint is safe to add to api.py — it does no I/O beyond reading a
local JSON file. The dashboard renders a "stale_minutes: 740" badge to
warn when the cache is older than today.

## Cost & coverage controls

| Knob | Default | Effect |
|---|---|---|
| `enrich_with_llm_agents(top_n=50)` | 50 | Max MCP calls per scan |
| `select_debate_targets(top_n=10)` | 10 | Max sub-agent invocations per session |
| `min_composite=60` | 60 | Filter weak signals out of debate |
| `Aiera cache TTL` | 7 days | Avoid re-running on stale catalysts |
| `FactSet cache TTL` | 1 day | Daily refresh is enough for revisions |
| `Daloopa cache TTL` | 30 days | Monthly enough for standardized fundamentals |

A single Top-10 debate run uses ~10 sub-agent calls (≈ 17k tokens each,
≈ 36s each based on the NVDA verdict you saw). All within Claude Max.

## A/B comparison workflow

When LLM signals flow, compare against legacy before promoting:

1. Run `enrich_with_llm_agents(...)` once in-session.
2. Run `signal_win_ratio.py` against both `catalyst_score` and
   `catalyst_score_v2`. Promote only if v2 win-rate is ≥ legacy + 2pp
   on out-of-sample SVE data.
3. Use `report_llm_movers(cache)` to spot-check positive/negative names.
4. Segment `SignalValidityEngine` output by `llm_catalyst_event_type` —
   earnings-driven names should show higher Pre-Mom hit rate than
   macro-driven names if the LLM signal is real.

## Roadmap (still no out-of-plan spend)

Phase 2 (1 month):
- `agents/filings_rag.py` — Daloopa-backed 10-K RAG for moat scoring
  (QVR Q boost). Same in-session pattern.

Phase 3 (3 months):
- `agents/macro_context_llm.py` — Moody's + S&P Global macro brief
  parsing for regime tilt.
- `agents/signal_validity_reflector.py` — LLM reads SVE hit-rate output
  and proposes weight adjustments (one-shot per week, in-session).

All of the above continue the same rule: nothing outside the session
ever calls an LLM.

## See also

- [CLAUDE.md](../CLAUDE.md) — full system docs
- [reports/sector_overview_2026-05-21.md](../reports/sector_overview_2026-05-21.md) — pitch-agent:sector-overview example output
- [reports/Nvidia_Q1_FY27_Earnings_Update.docx](../reports/Nvidia_Q1_FY27_Earnings_Update.docx) — equity-research:earnings-analysis example output
