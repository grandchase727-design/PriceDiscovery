# -*- coding: utf-8 -*-
"""per_ticker_debate.py — Option C Phase 1: Per-Ticker Debate Pipeline.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

Replaces the legacy "batched 240-pick debate" with per-ticker independent
debates. Each ticker is evaluated through a focused 3-round LLM mini-debate:

  Round 1: PM rationale → Trading verdict + Risk verdict (parallel? no — sequential
           on Max plan due to concurrent session lock) → Critic synthesis
  Round 2: PM revises based on Critic feedback → re-evaluate → Critic
  Round 3: If still divergent, final Arbiter call → forced consensus

================================================================================
WHY PER-TICKER (vs batch 240)
================================================================================

PROBLEM with legacy batch approach (root cause of universal-CAUTION cascade):
  - Trading LLM call: 240 tickers in one prompt → 8K+ token output → truncation
  - Risk LLM call: same — model loses focus on individual tickers
  - Debate Synthesizer: receives 240 picks each with their own (PM, Trading, Risk)
    verdicts → can't simulate genuine debate, falls back to defensive defaults
  - When ANY phase fails (concurrent session lock, rate limit, malformed output),
    ALL 240 picks are degraded — empirically observed: 240/240 SOLO/WATCH/WAIT/CAUTION

PER-TICKER SOLUTION:
  - Each ticker gets its OWN debate (small focused prompt)
  - Concurrent session lock affects ONE ticker, not 240
  - Smaller context → LLM stays focused → fewer hallucinations
  - Natural parallelism (different tickers are independent)
  - Quality degradation is per-ticker, not cascading

================================================================================
COST ANALYSIS
================================================================================

Legacy: 3 horizons × (1 PM + 1 Trading + 1 Risk + 1 Debate) × 5 iter rounds
      = 60 LLM calls (with large 240-ticker contexts)

Per-Ticker: N tickers (~60 surviving) × 3 rounds × (1 PM-rev + 1 Trading + 1 Risk + 1 Critic)
          = 720 LLM calls (with small 1-ticker contexts)

Cost: 12x more calls but each call is ~6x smaller → net ~2x output tokens.
Time: Sequential = 720 × 30s = 6 hours (TOO SLOW)
      → Need batching strategy: groups of 5 tickers per call

ACTUAL DESIGN (this module):
  Groups of 5 tickers/call, 3 rounds → 60/5 × 3 × 4 = 144 calls
  At 30s each (sequential) → 72 min. With parallelism (max_workers=2) → ~36 min.

================================================================================
INTEGRATION POINT
================================================================================

Called from market_leaders_swarm.py AFTER initial PM picks (Phase 5).
Replaces Phase 5.5 (Trading) + 5.55 (Risk) + 5.6a (Debate Synthesizer).

Input:  pm_horizons (3 horizons × 4 buckets × 20 picks = 240 picks)
Output: same shape but each pick has merged debate result:
        - debate_synthesis: tier, final_decision, stars, debate_transcript
        - timing.entry_signal: BUY_NOW / WAIT / SKIP
        - risk_verdict.vote: APPROVE / CAUTION / REJECT
        - _failed: True if all rounds failed for this ticker

================================================================================
ROUND STRUCTURE
================================================================================

Round 1 — Initial verdict batch (group of 5 tickers):
  Input: PM picks + rationale + (Phase 1 macro context)
  Output: per-ticker {pm_revise_needed, trading_verdict, risk_verdict, critic_notes}

Round 2 — Revision (only for divergent tickers):
  Input: Round 1 result + critic notes
  Output: PM revised rationale + updated Trading/Risk

Round 3 — Final arbiter (only if Round 2 still divergent):
  Single LLM call decides UNANIMOUS / MAJORITY / SOLO / EXCLUDED

================================================================================
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable

# ── Constants ─────────────────────────────────────────────────────
TICKERS_PER_BATCH = 5         # Small batch for focused LLM debate
MAX_ROUNDS = 3                # Round 1 (initial) + R2 (revise) + R3 (arbiter)
PER_BATCH_TIMEOUT = 240       # 4 min per 5-ticker batch
MAX_WORKERS = 1               # Strict sequential (Max plan session lock)


def _short(s: str, n: int = 100) -> str:
    return (s or "")[:n]


def _fmt_pick_block(p: dict, h: str, bk: str) -> str:
    """Compact format of a single pick for prompt."""
    return (
        f"  ticker: {p.get('ticker','?')}\n"
        f"  name:   {(p.get('name') or '')[:30]}\n"
        f"  sector: {p.get('sector','?')}\n"
        f"  bucket: {bk} ({h})\n"
        f"  composite: {p.get('composite','?')}\n"
        f"  classification: {p.get('classification','?')}\n"
        f"  pm_rationale: {_short(p.get('rationale',''), 200)}"
    )


def _build_round1_prompt(picks: list[dict], h: str, regime_tag: str = "—",
                          macro_context: str = "") -> str:
    """Per-batch Round 1 prompt — initial verdict on 5 tickers.

    Each ticker is evaluated independently. The model produces, per ticker:
      - trading_verdict (BUY_NOW / WAIT / SKIP + rationale)
      - risk_verdict (APPROVE / CAUTION / REJECT + rationale + key_risk)
      - critic_assessment (does PM rationale stand up to Trading/Risk?)
    """
    pick_blocks = []
    for p in picks:
        bk = p.get("_bucket", "?")
        pick_blocks.append(_fmt_pick_block(p, h, bk))

    macro_block = f"\nMACRO CONTEXT:\n{macro_context[:500]}\n" if macro_context else ""

    return f"""You are the **PER-TICKER DEBATE PANEL** for the **{h.upper()}** horizon.

You will evaluate {len(picks)} tickers INDEPENDENTLY. For each, produce verdicts
as if you were THREE separate agents debating (Trading / Risk / Critic).

Current market regime: {regime_tag}
{macro_block}

═══════════════════════════════════════════════════════════
TICKERS TO EVALUATE
═══════════════════════════════════════════════════════════

{chr(10).join(pick_blocks)}

═══════════════════════════════════════════════════════════
EVALUATION FRAMEWORK
═══════════════════════════════════════════════════════════

For EACH ticker, produce three INDEPENDENT verdicts:

1. **TRADING VERDICT** (timing-only lens):
   - entry_signal: "BUY_NOW" / "WAIT" / "SKIP"
   - entry_trigger: 1-sentence specific condition
   - urgency: "URGENT" / "NORMAL" / "PATIENT"
   - rationale (1-2 sentences, English): why this timing

2. **RISK VERDICT** (risk-only lens, regime-aware):
   - vote: "APPROVE" / "CAUTION" / "REJECT"
   - key_risk: 1-3 word identifier (e.g. "concentration", "OER>70", "macro")
   - rationale (1-2 sentences, 한국어): regime-aware risk assessment

3. **CRITIC ASSESSMENT** (does PM rationale survive scrutiny?):
   - assessment: "STRONG" / "OK" / "WEAK" / "FAIL"
   - challenge: 1 sentence challenging PM thesis (or "none — thesis holds")
   - revise_needed: true/false (does PM rationale need revision in Round 2?)

═══════════════════════════════════════════════════════════
ANTI-DEFAULT GUARD (T1 FIX)
═══════════════════════════════════════════════════════════

DO NOT default to WAIT/CAUTION for every ticker. Calibrate by regime:
  - If regime says RISK-ON: 40-60% should be BUY_NOW, not WAIT
  - If regime says RISK-OFF: 30-50% can be WAIT, but still 20-30% BUY_NOW for true conviction picks
  - If regime says ROTATION: differentiate by sector (some BUY_NOW, others WAIT)

NEVER give all tickers identical verdicts — that signals model failure, not real analysis.

═══════════════════════════════════════════════════════════
OUTPUT — STRICT JSON
═══════════════════════════════════════════════════════════

```json
{{
  "tickers": {{
    "AAPL": {{
      "trading": {{
        "entry_signal": "BUY_NOW",
        "entry_trigger": "Close above 20-day high with volume",
        "urgency": "NORMAL",
        "rationale": "Trend confirmation, low overhead supply"
      }},
      "risk": {{
        "vote": "APPROVE",
        "key_risk": "concentration",
        "rationale": "Mega cap concentration risk이나 OER 35로 양호. 정상 사이즈 진입 가능."
      }},
      "critic": {{
        "assessment": "STRONG",
        "challenge": "none — thesis holds",
        "revise_needed": false
      }}
    }},
    "MSFT": {{ ... same shape ... }}
  }}
}}
```

CRITICAL — output ALL {len(picks)} tickers in the JSON object. Missing tickers will be flagged as failed.

Return STRICTLY a fenced ```json block, nothing else."""


def _build_round2_prompt(picks_to_revise: list[dict], h: str,
                          round1_results: dict, regime_tag: str = "—") -> str:
    """Round 2 — revise PM rationale based on Critic feedback.

    Only tickers where critic.revise_needed=True go through this round.
    """
    pick_blocks = []
    for p in picks_to_revise:
        t = p.get("ticker")
        r1 = round1_results.get(t, {}) or {}
        critic = r1.get("critic", {}) or {}
        trading = r1.get("trading", {}) or {}
        risk = r1.get("risk", {}) or {}
        pick_blocks.append(
            f"  {t}:\n"
            f"    PM original rationale: {_short(p.get('rationale',''), 150)}\n"
            f"    Round 1 Critic: {critic.get('assessment','?')} — {_short(critic.get('challenge',''), 150)}\n"
            f"    Round 1 Trading: {trading.get('entry_signal','?')} — {_short(trading.get('rationale',''), 100)}\n"
            f"    Round 1 Risk: {risk.get('vote','?')} ({risk.get('key_risk','?')}) — {_short(risk.get('rationale',''), 100)}"
        )

    return f"""You are the **PM REVISION AGENT** for the **{h.upper()}** horizon.

{len(picks_to_revise)} tickers had Round 1 critic flags requiring rationale revision.

Current market regime: {regime_tag}

═══════════════════════════════════════════════════════════
TICKERS NEEDING REVISION
═══════════════════════════════════════════════════════════

{chr(10).join(pick_blocks)}

═══════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════

For each ticker, revise the PM rationale to address Critic's challenge.
Then re-issue Trading + Risk verdicts (with revised PM context).

If the Critic challenge CANNOT be answered → mark "drop_pick: true"
(this ticker fails Round 2 and will be EXCLUDED from final portfolio).

═══════════════════════════════════════════════════════════
OUTPUT — STRICT JSON
═══════════════════════════════════════════════════════════

```json
{{
  "tickers": {{
    "AAPL": {{
      "revised_rationale": "...",
      "trading": {{ "entry_signal":"BUY_NOW", "urgency":"NORMAL", "rationale":"..." }},
      "risk":    {{ "vote":"APPROVE", "key_risk":"...", "rationale":"..." }},
      "drop_pick": false
    }},
    ...
  }}
}}
```

Return STRICTLY a fenced ```json block."""


def _build_round3_arbiter_prompt(picks: list[dict], h: str,
                                  round_history: dict) -> str:
    """Round 3 — Final arbiter: decide tier + final_decision per ticker.

    Sees full debate history (R1 + R2) and produces final UNANIMOUS/MAJORITY/SOLO/EXCLUDED.
    """
    pick_blocks = []
    for p in picks:
        t = p.get("ticker")
        history = round_history.get(t, {}) or {}
        r1 = history.get("round1", {}) or {}
        r2 = history.get("round2", {}) or {}
        pick_blocks.append(
            f"  {t} (composite {p.get('composite','?')}):\n"
            f"    R1 Trading: {(r1.get('trading') or {}).get('entry_signal','?')} — {_short((r1.get('trading') or {}).get('rationale',''), 80)}\n"
            f"    R1 Risk:    {(r1.get('risk') or {}).get('vote','?')} — {_short((r1.get('risk') or {}).get('rationale',''), 80)}\n"
            f"    R1 Critic:  {(r1.get('critic') or {}).get('assessment','?')} — {_short((r1.get('critic') or {}).get('challenge',''), 80)}\n"
            f"    R2 (if any): revised={_short(r2.get('revised_rationale',''), 100)} drop={r2.get('drop_pick','—')}"
        )

    return f"""You are the **FINAL ARBITER** for {h.upper()} horizon.

Synthesize the debate history into a FINAL verdict per ticker.

═══════════════════════════════════════════════════════════
DEBATE HISTORY
═══════════════════════════════════════════════════════════

{chr(10).join(pick_blocks)}

═══════════════════════════════════════════════════════════
TIER DECISION RULES
═══════════════════════════════════════════════════════════

- UNANIMOUS (★★★)      : R1 Trading=BUY_NOW + Risk=APPROVE + Critic=STRONG, no R2 needed
- MAJORITY_CLEAN (★★★) : 2 of 3 strong agree, 1 minor caution (e.g. Risk=CAUTION but mild)
- MAJORITY_DISSENT (★★): 2 agree, 1 disagree — R2 revision may have helped
- SOLO (★)             : 1 strong + 2 mixed/weak
- EXCLUDED (0)         : 2+ REJECT/SKIP, or drop_pick=true

final_decision:
- INCLUDE              : execute the pick
- INCLUDE_REDUCED_SIZE : execute at half size (concentration/timing concern)
- WATCH                : do not enter yet, wait for trigger
- EXCLUDE              : drop from portfolio

═══════════════════════════════════════════════════════════
ANTI-DEFAULT GUARD
═══════════════════════════════════════════════════════════

Do NOT mark every ticker as SOLO/WATCH. That is a degraded output pattern.
Differentiate based on the actual debate evidence.

═══════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════

```json
{{
  "tickers": {{
    "AAPL": {{
      "tier": "UNANIMOUS",
      "stars": 3,
      "final_decision": "INCLUDE",
      "debate_transcript": "PM: ... R1 Trading BUY_NOW + Risk APPROVE + Critic STRONG → 만장일치 매수.",
      "key_factor": "만장일치"
    }},
    ...
  }}
}}
```

Return STRICTLY a fenced ```json block."""


def _extract_json_block(text: str) -> Optional[dict]:
    """Extract first ```json...``` block."""
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        # Fallback: try parsing entire text as JSON
        try: return json.loads(text)
        except Exception: return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _batch_picks(all_picks: list[dict], batch_size: int = TICKERS_PER_BATCH) -> list[list[dict]]:
    """Split picks into small batches of N tickers each."""
    return [all_picks[i:i + batch_size] for i in range(0, len(all_picks), batch_size)]


def run_per_ticker_debate(pm_horizons: dict, regime_tag: str = "—",
                           macro_context: str = "",
                           run_claude_fn: Callable = None,
                           _emit_fn: Optional[Callable] = None) -> dict:
    """Main entry point.

    Args:
        pm_horizons: PM picks {horizon: {bucket: [picks]}}
        regime_tag: macro regime tag
        macro_context: Phase 1 summary
        run_claude_fn: claude -p subprocess wrapper
        _emit_fn: progress emission

    Returns:
        Same shape as pm_horizons, with each pick augmented with:
          - debate_synthesis: {tier, final_decision, stars, debate_transcript, key_factor}
          - timing: {entry_signal, urgency, rationale}
          - risk_verdict: {vote, key_risk, rationale}
          - _failed: True if all rounds failed for this ticker
    """
    def _emit(phase: str, agent: str, status: str):
        if _emit_fn:
            try: _emit_fn(phase, agent, status)
            except Exception: pass

    out_horizons = {}
    for h in ("tactical", "core", "strategic"):
        hd = pm_horizons.get(h, {}) or {}
        out_horizons[h] = {bk: list(hd.get(bk, []) or []) for bk in
                            ("long_stocks","long_etfs","short_stocks","short_etfs")}

        # Flatten picks for this horizon with bucket tag
        all_picks: list[dict] = []
        for bk in ("long_stocks","long_etfs","short_stocks","short_etfs"):
            for p in out_horizons[h][bk]:
                p["_bucket"] = bk
                all_picks.append(p)

        if not all_picks:
            continue

        _emit("phase5_pt_debate", f"{h}_start", f"n_picks={len(all_picks)}")

        # ── Round 1: batched initial debate ──
        batches = _batch_picks(all_picks)
        round1_results: dict[str, dict] = {}   # ticker → {trading, risk, critic}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut_map = {}
            for bi, batch in enumerate(batches):
                prompt = _build_round1_prompt(batch, h, regime_tag, macro_context)
                fut = ex.submit(run_claude_fn, prompt, PER_BATCH_TIMEOUT, 2)
                fut_map[fut] = (bi, batch)

            for fut in as_completed(fut_map):
                bi, batch = fut_map[fut]
                _emit("phase5_pt_debate", f"{h}_r1_b{bi}", "started")
                try:
                    result = fut.result()
                    if isinstance(result, dict):
                        ticker_results = result.get("tickers", {})
                    else:
                        ticker_results = {}
                    for p in batch:
                        t = p.get("ticker")
                        if t in ticker_results:
                            round1_results[t] = ticker_results[t]
                        else:
                            round1_results[t] = {"_failed": True, "_failure_reason": "missing_in_r1"}
                    _emit("phase5_pt_debate", f"{h}_r1_b{bi}", "ok")
                except Exception as e:
                    for p in batch:
                        round1_results[p.get("ticker")] = {"_failed": True, "_failure_reason": str(e)[:200]}
                    _emit("phase5_pt_debate", f"{h}_r1_b{bi}", f"fail: {str(e)[:100]}")

        # ── Round 2: revision (only for picks where critic flagged) ──
        picks_to_revise = [
            p for p in all_picks
            if (round1_results.get(p.get("ticker"), {}) or {}).get("critic", {}).get("revise_needed")
        ]
        round2_results: dict[str, dict] = {}
        if picks_to_revise:
            r2_batches = _batch_picks(picks_to_revise)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                fut_map = {}
                for bi, batch in enumerate(r2_batches):
                    prompt = _build_round2_prompt(batch, h, round1_results, regime_tag)
                    fut = ex.submit(run_claude_fn, prompt, PER_BATCH_TIMEOUT, 2)
                    fut_map[fut] = (bi, batch)

                for fut in as_completed(fut_map):
                    bi, batch = fut_map[fut]
                    _emit("phase5_pt_debate", f"{h}_r2_b{bi}", "started")
                    try:
                        result = fut.result()
                        if isinstance(result, dict):
                            ticker_results = result.get("tickers", {})
                        else:
                            ticker_results = {}
                        for p in batch:
                            t = p.get("ticker")
                            if t in ticker_results:
                                round2_results[t] = ticker_results[t]
                        _emit("phase5_pt_debate", f"{h}_r2_b{bi}", "ok")
                    except Exception as e:
                        _emit("phase5_pt_debate", f"{h}_r2_b{bi}", f"fail: {str(e)[:100]}")

        # ── Round 3: final arbiter (single call, all picks) ──
        round_history = {
            t: {"round1": round1_results.get(t, {}), "round2": round2_results.get(t, {})}
            for t in (p.get("ticker") for p in all_picks)
        }
        prompt3 = _build_round3_arbiter_prompt(all_picks, h, round_history)
        try:
            r3_result = run_claude_fn(prompt3, PER_BATCH_TIMEOUT, 2)
            r3_tickers = r3_result.get("tickers", {}) if isinstance(r3_result, dict) else {}
            _emit("phase5_pt_debate", f"{h}_r3", "ok")
        except Exception as e:
            r3_tickers = {}
            _emit("phase5_pt_debate", f"{h}_r3", f"fail: {str(e)[:100]}")

        # ── Merge results back into picks ──
        for p in all_picks:
            t = p.get("ticker")
            r1 = round1_results.get(t, {}) or {}
            r2 = round2_results.get(t, {}) or {}
            r3 = r3_tickers.get(t, {}) or {}

            # Trading verdict (prefer R2 revision if exists)
            trading = (r2.get("trading") if r2 else None) or r1.get("trading") or {}
            if not trading or not trading.get("entry_signal"):
                p["timing"] = {"_failed": True, "_failure_reason": "no_trading_verdict"}
            else:
                p["timing"] = {
                    "entry_signal": trading.get("entry_signal"),
                    "entry_trigger": trading.get("entry_trigger", ""),
                    "urgency": trading.get("urgency", "NORMAL"),
                    "rationale": trading.get("rationale", ""),
                    "exit_triggers": [],
                }

            # Risk verdict
            risk = (r2.get("risk") if r2 else None) or r1.get("risk") or {}
            if not risk or not risk.get("vote"):
                p["risk_verdict"] = {"_failed": True, "_failure_reason": "no_risk_verdict"}
            else:
                p["risk_verdict"] = {
                    "vote": risk.get("vote"),
                    "key_risk": risk.get("key_risk", "—"),
                    "rationale": risk.get("rationale", ""),
                }

            # Debate synthesis (from R3)
            if r3 and r3.get("tier") and r3.get("final_decision"):
                p["debate_synthesis"] = {
                    "tier": r3["tier"],
                    "stars": int(r3.get("stars", 1) or 1),
                    "final_decision": r3["final_decision"],
                    "debate_transcript": (r3.get("debate_transcript") or "")[:600],
                    "key_factor": (r3.get("key_factor") or "—")[:30],
                }
            else:
                p["debate_synthesis"] = {
                    "_failed": True,
                    "_failure_reason": "no_r3_verdict",
                    "tier": None, "stars": None, "final_decision": None,
                    "debate_transcript": "⚠ R3 Arbiter가 이 ticker에 대한 verdict 미생성 — composite-based fallback 필요",
                    "key_factor": "_failed",
                }

            # Per-ticker debug
            p["_pt_debate"] = {
                "round1": r1,
                "round2": r2,
                "round3": r3,
            }

        _emit("phase5_pt_debate", f"{h}_done",
              f"r1_picks={len(round1_results)} r2_picks={len(round2_results)} r3_picks={len(r3_tickers)}")

    return out_horizons


# ─────────────────────────────────────────────────────────────────
# Helpers for integration / debugging
# ─────────────────────────────────────────────────────────────────

def summarize_debate_results(pm_horizons: dict) -> dict:
    """Compact summary of debate outcomes for monitoring."""
    from collections import Counter
    summary = {"tier_dist": Counter(), "trading_dist": Counter(),
                "risk_dist": Counter(), "n_failed": 0, "n_total": 0}
    for h in ("tactical", "core", "strategic"):
        for bk in ("long_stocks","long_etfs","short_stocks","short_etfs"):
            for p in (pm_horizons.get(h, {}) or {}).get(bk, []) or []:
                summary["n_total"] += 1
                ds = p.get("debate_synthesis") or {}
                if ds.get("_failed"):
                    summary["n_failed"] += 1
                summary["tier_dist"][ds.get("tier") or "_failed"] += 1
                tm = p.get("timing") or {}
                summary["trading_dist"][tm.get("entry_signal") or "_failed"] += 1
                rv = p.get("risk_verdict") or {}
                summary["risk_dist"][rv.get("vote") or "_failed"] += 1
    summary["tier_dist"] = dict(summary["tier_dist"])
    summary["trading_dist"] = dict(summary["trading_dist"])
    summary["risk_dist"] = dict(summary["risk_dist"])
    return summary
