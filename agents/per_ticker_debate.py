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
from concurrent.futures import TimeoutError as _FutureTimeoutError
from typing import Optional, Callable

# ── Constants ─────────────────────────────────────────────────────
TICKERS_PER_BATCH = 15        # 8→15: horizon당 배치 수 절반으로 감소 (20픽 → 2배치 → 1~2배치)
MAX_ROUNDS = 3                # Round 1 (initial) + R2 (revise) + R3 (arbiter)
PER_BATCH_TIMEOUT = 300       # 15-ticker batch (360→300, retry 곱연산 완화)
DEBATE_BATCH_RETRIES = 1      # run_claude retries (2→1: 3 attempt → 2 attempt, 최악 wall 1/3 감소)
# 전역 wall-clock budget — debate 단계 전체가 이 시간을 넘으면 남은 배치/retry를 중단.
# backend 장애 시 multi-hour hang을 방지하는 실질적 차단선 (ITERATION_TIME_BUDGET와 달리
# in-flight 배치 대기를 실제로 끊음). core 단일 horizon · ~6배치 기준 20분이면 충분.
DEBATE_WALL_BUDGET_SEC = 1200
# Batch concurrency — bounded by the subprocess semaphore (CLAUDE_MAX_CONCURRENCY).
# R1/R2 batches are independent → run concurrently for lower wall-clock.
try:
    import os as _os
    MAX_WORKERS = max(1, int(_os.environ.get("CLAUDE_MAX_CONCURRENCY", "1")))
except (TypeError, ValueError):
    MAX_WORKERS = 1


def _short(s: str, n: int = 100) -> str:
    return (s or "")[:n]


def _fmt_pick_block(p: dict, h: str, bk: str) -> str:
    """Compact format of a single pick for prompt."""
    # 2026-07-28: forming/조기(pre_momentum) + entry_timing 맥락 노출 — 형성단계 픽이
    # 'composite 낮은 숫자'만으로 SKIP/REJECT 유도되던 구조 해소. pool_source가
    # pre_momentum이면 확정모멘텀 잣대(comp≥55) 대신 '사전 진입 후보'로 평가하도록.
    _extra = ""
    if p.get("pool_source") == "pre_momentum":
        _extra += (f"\n  pool_source: ⚑PRE-MOM (조기/형성단계 — 확정모멘텀 미달이 정상, "
                   f"pre_momentum={p.get('pre_momentum_score','?')}. 낮은 composite로 감점 말 것)")
    if p.get("entry_timing_status"):
        _extra += f"\n  entry_timing: {p.get('entry_timing_status')} (FRESH=적시/EXTENDED=추격위험)"
    return (
        f"  ticker: {p.get('ticker','?')}\n"
        f"  name:   {(p.get('name') or '')[:30]}\n"
        f"  sector: {p.get('sector','?')}\n"
        f"  bucket: {bk} ({h})\n"
        f"  composite: {p.get('composite','?')}\n"
        f"  classification: {p.get('classification','?')}{_extra}\n"
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
    # SEC 10-K/10-Q RAG context per ticker (semantic retrieval over filings).
    # Optional + graceful: returns "" if the index isn't built (sec_filings_pipeline.py)
    # or fastembed is missing, so the debate runs unchanged without it.
    try:
        from filing_rag import format_filing_block
    except Exception:
        format_filing_block = None

    pick_blocks = []
    for p in picks:
        bk = p.get("_bucket", "?")
        block = _fmt_pick_block(p, h, bk)
        if format_filing_block:
            try:
                tk = p.get("ticker", "")
                fb = format_filing_block(
                    tk, f"{tk} risk factors, recent quarterly results, guidance, outlook, "
                        f"competitive and regulatory risks", k=3)
                if fb:
                    block += "\n" + fb
            except Exception:
                pass
        pick_blocks.append(block)

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

For EACH ticker, produce three INDEPENDENT verdicts. When a ticker includes a
"📄 SEC 공시 발췌" block, GROUND your Risk and Critic verdicts in those actual
10-K/10-Q excerpts (cite the specific risk factor / MD&A point) — they are the
company's own disclosures, not market chatter.

1. **TRADING VERDICT** (timing-only lens):
   - entry_signal: "BUY_NOW" / "WAIT" / "SKIP"
   - entry_trigger: 1-sentence specific condition (한국어)
   - urgency: "URGENT" / "NORMAL" / "PATIENT"
   - rationale (1-2 sentences, 한국어): why this timing

2. **RISK VERDICT** (risk-only lens, regime-aware):
   - vote: "APPROVE" / "CAUTION" / "REJECT"
   - key_risk: 1-3 word identifier, 한국어 (e.g. "집중도", "과열(OER>70)", "매크로", "신용사이클")
   - rationale (1-2 sentences, 한국어): regime-aware risk assessment

3. **CRITIC ASSESSMENT** (does PM rationale survive scrutiny?):
   - assessment: "STRONG" / "OK" / "WEAK" / "FAIL"
   - challenge: 1 sentence challenging PM thesis, 한국어 (or "none — thesis holds")
   - revise_needed: true/false (does PM rationale need revision in Round 2?)

═══════════════════════════════════════════════════════════
★★ LANGUAGE — ALL human-facing commentary text MUST be in KOREAN (한국어) ★★
═══════════════════════════════════════════════════════════

Applies to every free-text string value you emit — trading `rationale` /
`entry_trigger`, risk `rationale` / `key_risk`, and critic `challenge`.
DO KEEP IN ENGLISH (case-sensitive identifiers / fixed tokens):
  - JSON keys, enum values ("BUY_NOW", "WAIT", "SKIP", "APPROVE", "CAUTION",
    "REJECT", "STRONG", "OK", "WEAK", "FAIL", "URGENT", "NORMAL", "PATIENT")
  - ticker symbols (AAPL, MSFT, 005930.KS), classification tags (CONTINUATION,
    OVEREXTENDED, NEUTRAL, …), indicator names (Composite, OER, RSI, SMA20)
Everything else — the actual sentences a human reads — must be 한국어.
Do NOT emit English prose like "OER 24 is among the lowest" or snake_case
identifiers like "credit_cycle" in any human-facing field; write "OER 24는
Industrials 그룹 최저 수준", "신용 사이클" instead. The critic `challenge`
sentinel "none — thesis holds" is the ONLY allowed English free-text value.

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
        "entry_trigger": "거래량을 동반한 20일 신고가 돌파",
        "urgency": "NORMAL",
        "rationale": "추세 확인 완료, 위쪽 매물벽이 얕아 즉시 진입 가능"
      }},
      "risk": {{
        "vote": "APPROVE",
        "key_risk": "집중도",
        "rationale": "메가캡 집중 리스크는 있으나 OER 35로 과열도가 양호해 정상 사이즈 진입 가능."
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
★★ LANGUAGE — ALL human-facing commentary text MUST be in KOREAN (한국어) ★★
═══════════════════════════════════════════════════════════

`revised_rationale`, trading `rationale`, and risk `rationale` / `key_risk`
are read by humans → write them in 한국어. Keep ONLY JSON keys, enum values
("BUY_NOW"/"WAIT"/"SKIP", "APPROVE"/"CAUTION"/"REJECT"), ticker symbols,
classification tags, and indicator names (Composite, OER, RSI) in English.
Do NOT emit English prose or snake_case identifiers in any human-facing field.

═══════════════════════════════════════════════════════════
OUTPUT — STRICT JSON
═══════════════════════════════════════════════════════════

```json
{{
  "tickers": {{
    "AAPL": {{
      "revised_rationale": "비판을 반영해 수정한 PM 논지 (한국어)",
      "trading": {{ "entry_signal":"BUY_NOW", "urgency":"NORMAL", "rationale":"진입 타이밍 근거 (한국어)" }},
      "risk":    {{ "vote":"APPROVE", "key_risk":"집중도", "rationale":"리스크 근거 (한국어)" }},
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
★★ LANGUAGE — `debate_transcript` and `key_factor` MUST be in KOREAN (한국어) ★★
═══════════════════════════════════════════════════════════

Keep tier/final_decision enums, ticker symbols, and indicator names (Composite,
OER) in English; all free-text you write must be 한국어.

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


def _synthesize_from_r1(r1: dict, r2: dict = None) -> dict:
    """Build debate_synthesis from R1 (+ optional R2) data when R3 doesn't run.

    Used when max_rounds < 3 or when R3 arbiter fails.  R1 always has per-ticker
    Trading + Risk + Critic verdicts; we map those to tier/stars/final_decision
    and generate a transcript so the frontend never shows a blank debate column.
    """
    trading = (r2.get("trading") if r2 else None) or r1.get("trading") or {}
    risk    = (r2.get("risk")    if r2 else None) or r1.get("risk")    or {}
    critic  = r1.get("critic") or {}

    entry  = trading.get("entry_signal", "")
    r_vote = risk.get("vote", "")
    c_ass  = critic.get("assessment", "")
    drop   = bool((r2 or {}).get("drop_pick"))

    # Tier / stars / final_decision derivation from R1 verdicts
    if drop or entry == "SKIP":
        tier, stars, fd = "EXCLUDED", 0, "EXCLUDE"
    elif entry == "BUY_NOW" and r_vote == "APPROVE" and c_ass == "STRONG":
        tier, stars, fd = "UNANIMOUS", 3, "INCLUDE"
    elif entry == "BUY_NOW" and r_vote == "APPROVE" and c_ass in ("STRONG", "OK"):
        tier, stars, fd = "MAJORITY_CLEAN", 3, "INCLUDE"
    elif entry == "BUY_NOW" and r_vote in ("APPROVE", "CAUTION"):
        tier, stars, fd = "MAJORITY_DISSENT", 2, "INCLUDE"
    elif entry == "WAIT" and r_vote == "APPROVE" and c_ass in ("STRONG", "OK"):
        tier, stars, fd = "SOLO_CLEAN", 2, "WATCH"
    elif entry == "BUY_NOW" and r_vote == "REJECT":
        tier, stars, fd = "SOLO_DISSENT", 1, "INCLUDE_REDUCED_SIZE"
    elif entry == "WAIT" and r_vote in ("APPROVE", "CAUTION"):
        tier, stars, fd = "SOLO", 1, "WATCH"
    elif entry == "WAIT" and r_vote == "REJECT":
        tier, stars, fd = "EXCLUDED", 0, "EXCLUDE"
    else:
        agree = sum([entry == "BUY_NOW", r_vote == "APPROVE", c_ass in ("STRONG", "OK")])
        if agree >= 2:
            tier, stars, fd = "MAJORITY_DISSENT", 2, "WATCH"
        else:
            tier, stars, fd = "SOLO", 1, "WATCH"

    OUTCOME = {
        "UNANIMOUS":       "→ 만장일치 매수 신호.",
        "MAJORITY_CLEAN":  "→ 강한 다수 합의 매수.",
        "MAJORITY_DISSENT":"→ 다수 매수 동의 (이견 있음).",
        "SOLO_CLEAN":      "→ 단독 승인, 타이밍 대기.",
        "SOLO_DISSENT":    "→ 단독 승인이나 리스크 우려. 소규모 진입 고려.",
        "SOLO":            "→ 약한 신호 — 추가 확인 필요.",
        "EXCLUDED":        "→ 제외 (Trading SKIP / Risk REJECT).",
    }
    KEY = {
        "UNANIMOUS": "만장일치", "MAJORITY_CLEAN": "강한 다수 합의",
        "MAJORITY_DISSENT": "다수 동의 (이견)", "SOLO_CLEAN": "단독 승인",
        "SOLO_DISSENT": "단독 승인 (리스크)", "SOLO": "단독 신호", "EXCLUDED": "제외",
    }

    transcript = (
        f"[R1] Trading={entry} ({trading.get('urgency','?')}) | "
        f"Risk={r_vote} [{risk.get('key_risk','—')}] | Critic={c_ass}. "
        f"Trading: {(trading.get('rationale') or '')[:150]}. "
        f"Risk: {(risk.get('rationale') or '')[:150]}. "
        f"Critic: {(critic.get('challenge') or 'none — thesis holds')[:100]}. "
        f"{OUTCOME.get(tier, '')}"
    )

    return {
        "tier": tier,
        "stars": stars,
        "final_decision": fd,
        "debate_transcript": transcript[:600],
        "key_factor": KEY.get(tier, tier),
        # 구조화 critic challenge — final_list._unified_commentary가 '비판적으로는 …'
        # 절을 만들 때 소비. sentinel("none — thesis holds")는 그쪽에서 자동 생략.
        "critic_challenge": (critic.get("challenge") or "none — thesis holds")[:200],
        "_from_r1_synthesis": True,
    }


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
                           _emit_fn: Optional[Callable] = None,
                           horizons: tuple = ("core",),
                           max_picks_per_horizon: Optional[int] = None,
                           max_rounds: int = MAX_ROUNDS) -> dict:
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

    # ── 전역 wall-clock budget ──────────────────────────────────────
    _debate_start = time.time()
    def _budget_remaining() -> float:
        return DEBATE_WALL_BUDGET_SEC - (time.time() - _debate_start)
    def _budget_exhausted() -> bool:
        return _budget_remaining() <= 0

    out_horizons = {}
    for h in ("core",):
        hd = pm_horizons.get(h, {}) or {}
        out_horizons[h] = {bk: list(hd.get(bk, []) or []) for bk in
                            ("long_stocks","long_etfs","short_stocks","short_etfs")}

        # "안전 우선" 30분 모드: only debate the configured horizons (e.g. core only).
        # Non-debated horizons keep their PM picks verbatim (no debate augmentation).
        if h not in horizons:
            continue

        # Flatten picks for this horizon with bucket tag
        all_picks: list[dict] = []
        for bk in ("long_stocks","long_etfs","short_stocks","short_etfs"):
            for p in out_horizons[h][bk]:
                p["_bucket"] = bk
                all_picks.append(p)

        if not all_picks:
            continue

        # Cap to the top-N highest-conviction picks (by composite) to bound LLM calls.
        if max_picks_per_horizon and len(all_picks) > max_picks_per_horizon:
            all_picks.sort(key=lambda p: float(p.get("composite") or 0), reverse=True)
            all_picks = all_picks[:max_picks_per_horizon]

        _emit("phase5_pt_debate", f"{h}_start", f"n_picks={len(all_picks)}")

        # ── Round 1: batched initial debate ──
        batches = _batch_picks(all_picks)
        round1_results: dict[str, dict] = {}   # ticker → {trading, risk, critic}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            fut_map = {}
            for bi, batch in enumerate(batches):
                prompt = _build_round1_prompt(batch, h, regime_tag, macro_context)
                fut = ex.submit(run_claude_fn, prompt, PER_BATCH_TIMEOUT, DEBATE_BATCH_RETRIES)
                fut_map[fut] = (bi, batch)

            # as_completed with budget-bounded timeout — 전체 R1이 budget을 넘으면
            # 미완료 배치는 _failed 처리하고 빠져나간다 (in-flight hang 차단).
            try:
                for fut in as_completed(fut_map, timeout=max(1.0, _budget_remaining())):
                    bi, batch = fut_map[fut]
                    _emit("phase5_pt_debate", f"{h}_r1_b{bi}", "started")
                    try:
                        result = fut.result()
                        ticker_results = result.get("tickers", {}) if isinstance(result, dict) else {}
                        for p in batch:
                            t = p.get("ticker")
                            round1_results[t] = ticker_results.get(t) or {"_failed": True, "_failure_reason": "missing_in_r1"}
                        _emit("phase5_pt_debate", f"{h}_r1_b{bi}", "ok")
                    except Exception as e:
                        for p in batch:
                            round1_results[p.get("ticker")] = {"_failed": True, "_failure_reason": str(e)[:200]}
                        _emit("phase5_pt_debate", f"{h}_r1_b{bi}", f"fail: {str(e)[:100]}")
            except (TimeoutError, _FutureTimeoutError):
                # budget 소진 — 아직 결과 없는 배치는 모두 _failed 처리.
                # ★ Python 3.9-3.10에서는 concurrent.futures.TimeoutError가 builtins.TimeoutError와
                # 다른 클래스라 as_completed()가 던지는 예외를 놓치면 run_per_ticker_debate() 전체가
                # 예외로 죽어 이미 완료된 배치까지 통째로 유실된다 — 두 클래스 모두 catch 필수.
                for fut, (bi, batch) in fut_map.items():
                    for p in batch:
                        round1_results.setdefault(p.get("ticker"), {"_failed": True, "_failure_reason": "budget_exhausted_r1"})
                _emit("phase5_pt_debate", f"{h}_r1_budget", "wall-clock budget 소진 — 미완료 배치 skip")

        # ── Retry: 배치 실패 종목을 1-ticker씩 단독 재시도 ──
        # budget 잔여 시간 내에서만 순차 재시도 — 소진 시 남은 종목은 _failed 유지.
        failed_picks = [p for p in all_picks
                        if (round1_results.get(p.get("ticker"), {}) or {}).get("_failed")]
        if failed_picks:
            _emit("phase5_pt_debate", f"{h}_r1_retry", f"retrying {len(failed_picks)} failed picks solo")
            for p in failed_picks:
                t = p.get("ticker")
                if _budget_remaining() < PER_BATCH_TIMEOUT:
                    _emit("phase5_pt_debate", f"{h}_retry_budget", "budget 부족 — 남은 solo retry skip")
                    break
                try:
                    prompt = _build_round1_prompt([p], h, regime_tag, macro_context)
                    result = run_claude_fn(prompt, PER_BATCH_TIMEOUT, DEBATE_BATCH_RETRIES)
                    if isinstance(result, dict):
                        tr = result.get("tickers", {})
                        if t in tr:
                            round1_results[t] = tr[t]
                            _emit("phase5_pt_debate", f"{h}_retry_{t}", "ok")
                            continue
                except Exception as e:
                    _emit("phase5_pt_debate", f"{h}_retry_{t}", f"fail: {str(e)[:80]}")
                # 단독 retry도 실패 시 _failed 유지

        # ── Round 2: revision (only for picks where critic flagged) ──
        picks_to_revise = [
            p for p in all_picks
            if max_rounds >= 2 and (round1_results.get(p.get("ticker"), {}) or {}).get("critic", {}).get("revise_needed")
        ]
        round2_results: dict[str, dict] = {}
        if picks_to_revise and not _budget_exhausted():
            r2_batches = _batch_picks(picks_to_revise)
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                fut_map = {}
                for bi, batch in enumerate(r2_batches):
                    prompt = _build_round2_prompt(batch, h, round1_results, regime_tag)
                    fut = ex.submit(run_claude_fn, prompt, PER_BATCH_TIMEOUT, DEBATE_BATCH_RETRIES)
                    fut_map[fut] = (bi, batch)

                try:
                    for fut in as_completed(fut_map, timeout=max(1.0, _budget_remaining())):
                        bi, batch = fut_map[fut]
                        _emit("phase5_pt_debate", f"{h}_r2_b{bi}", "started")
                        try:
                            result = fut.result()
                            ticker_results = result.get("tickers", {}) if isinstance(result, dict) else {}
                            for p in batch:
                                t = p.get("ticker")
                                if t in ticker_results:
                                    round2_results[t] = ticker_results[t]
                            _emit("phase5_pt_debate", f"{h}_r2_b{bi}", "ok")
                        except Exception as e:
                            _emit("phase5_pt_debate", f"{h}_r2_b{bi}", f"fail: {str(e)[:100]}")
                except (TimeoutError, _FutureTimeoutError):
                    _emit("phase5_pt_debate", f"{h}_r2_budget", "wall-clock budget 소진 — R2 미완료 배치 skip")
        elif picks_to_revise:
            _emit("phase5_pt_debate", f"{h}_r2_budget", "budget 소진 — R2 전체 skip")

        # ── Round 3: final arbiter (single call, all picks) — only if max_rounds>=3 ──
        round_history = {
            t: {"round1": round1_results.get(t, {}), "round2": round2_results.get(t, {})}
            for t in (p.get("ticker") for p in all_picks)
        }
        r3_tickers = {}
        if max_rounds >= 3 and not _budget_exhausted():
            prompt3 = _build_round3_arbiter_prompt(all_picks, h, round_history)
            try:
                r3_result = run_claude_fn(prompt3, PER_BATCH_TIMEOUT, DEBATE_BATCH_RETRIES)
                r3_tickers = r3_result.get("tickers", {}) if isinstance(r3_result, dict) else {}
                _emit("phase5_pt_debate", f"{h}_r3", "ok")
            except Exception as e:
                _emit("phase5_pt_debate", f"{h}_r3", f"fail: {str(e)[:100]}")
        elif max_rounds >= 3:
            _emit("phase5_pt_debate", f"{h}_r3_budget", "budget 소진 — R3 arbiter skip")

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

            # Debate synthesis (prefer R3 if available, else synthesize from R1/R2)
            if r3 and r3.get("tier") and r3.get("final_decision"):
                # critic challenge: R3 스키마엔 별도 필드가 없으므로 R1(→R2) critic에서
                # 도출해 구조화 저장 (final_list._unified_commentary의 '비판적으로는 …' 절).
                _r3_critic = ((r2 or {}).get("critic") if r2 else None) or r1.get("critic") or {}
                p["debate_synthesis"] = {
                    "tier": r3["tier"],
                    "stars": int(r3.get("stars", 1) or 1),
                    "final_decision": r3["final_decision"],
                    "debate_transcript": (r3.get("debate_transcript") or "")[:600],
                    "key_factor": (r3.get("key_factor") or "—")[:30],
                    "critic_challenge": (_r3_critic.get("challenge")
                                         or "none — thesis holds")[:200],
                }
            elif r1 and not r1.get("_failed") and (r1.get("trading") or r1.get("risk")):
                # R1 data available — synthesize tier/transcript without R3
                p["debate_synthesis"] = _synthesize_from_r1(r1, r2 if r2 else None)
            else:
                p["debate_synthesis"] = {
                    "_failed": True,
                    "_failure_reason": "no_r1_verdict",
                    "tier": None, "stars": None, "final_decision": None,
                    "debate_transcript": "⚠ R1 debate 미실행 — composite-based fallback 필요",
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
    for h in ("core",):
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


# ─────────────────────────────────────────────────────────────────
# Exit Debate — EXIT_PENDING 청산 후보 LLM 토론
# ─────────────────────────────────────────────────────────────────

EXIT_DEBATE_BATCH = 6    # 청산 후보는 일반적으로 수가 적으므로 한 배치에 6종목
EXIT_DEBATE_TIMEOUT = 300


def _build_exit_debate_prompt(picks: list[dict], regime_tag: str, macro_context: str) -> str:
    """청산 후보 종목들에 대한 EXIT vs HOLD 2-panel 토론 프롬프트."""
    n = len(picks)
    lines = []
    for p in picks:
        t = p.get("ticker", "?")
        h = p.get("horizon", "?")
        line = (
            f"  • {t} ({h}): composite={p.get('composite', 0):.1f}, "
            f"보유={p.get('days_held', 0)}일, "
            f"분류={p.get('classification', '?')}"
        )
        # L1 evidence: P&L trajectory since entry (present when telemetry exists)
        pnl, mfe = p.get("pnl_pct"), p.get("mfe_pct")
        if pnl is not None:
            line += f", 진입후손익={pnl:+.1f}%"
        if mfe is not None:
            line += f", 고점대비 MFE={mfe:+.1f}%"
            if p.get("retained_pct") is not None:
                line += f" (이익보존율 {p['retained_pct']:.0f}%)"
        if p.get("mae_pct") is not None:
            line += f", 최대낙폭 MAE={p['mae_pct']:+.1f}%"
        line += f", 청산사유={str(p.get('exit_reason', ''))[:120]}"
        lines.append(line)
    picks_block = "\n".join(lines)

    return f"""당신은 포트폴리오 청산 심의 패널입니다. 아래 {n}개 종목은 position_state machine이 EXIT_PENDING으로 표시한 보유 포지션입니다. 각 종목에 대해 **청산 주장 vs 보유 주장**을 심의하고 최종 판정을 내려주세요.

매크로 레짐: {regime_tag}
시장 컨텍스트: {macro_context[:600]}

═══════════════════════════════════
EXIT_PENDING 종목 목록:
═══════════════════════════════════
{picks_block}

═══════════════════════════════════
심의 기준:
═══════════════════════════════════
• EXIT_NOW    : 청산 사유 명확, 추가 손실 위험 높음, 즉시 매도 권장
• PARTIAL_EXIT: 일부 청산 (50%), 나머지는 1주 관망
• HOLD_1W     : 청산 신호 과도, 1주 더 관망 후 재평가

각 판정의 confidence: HIGH / MEDIUM / LOW

═══════════════════════════════════
JSON 출력 (반드시 아래 스키마):
═══════════════════════════════════
```json
{{
  "exit_verdicts": [
    {{
      "ticker": "...",
      "horizon": "...",
      "verdict": "EXIT_NOW" | "PARTIAL_EXIT" | "HOLD_1W",
      "confidence": "HIGH" | "MEDIUM" | "LOW",
      "exit_case": "청산 근거 1-2문장",
      "hold_case": "보유 근거 1문장 (있다면)",
      "key_factor": "핵심 판단 근거 한 줄",
      "debate_transcript": "패널 토론 요약 (한국어 150자 내)"
    }}
  ]
}}
```"""


def run_exit_debate(
    exit_picks: list[dict],
    regime_tag: str,
    macro_context: str,
    run_claude_fn,
    _emit_fn=None,
    batch_timeout: int = EXIT_DEBATE_TIMEOUT,
) -> dict:
    """EXIT_PENDING 포지션에 대해 LLM exit debate를 실행.

    Returns dict keyed by "ticker::horizon" → verdict dict.
    """
    def _emit(phase, agent, status):
        if _emit_fn:
            try: _emit_fn(phase, agent, status)
            except Exception: pass

    if not exit_picks:
        return {}

    results: dict[str, dict] = {}

    # 배치 분할
    batches = [exit_picks[i:i+EXIT_DEBATE_BATCH] for i in range(0, len(exit_picks), EXIT_DEBATE_BATCH)]
    _emit("phase5_exit_debate", "start", f"n={len(exit_picks)} batches={len(batches)}")

    for bi, batch in enumerate(batches):
        prompt = _build_exit_debate_prompt(batch, regime_tag, macro_context)
        try:
            raw = run_claude_fn(prompt, batch_timeout, 1)
            verdicts = (raw or {}).get("exit_verdicts") if isinstance(raw, dict) else None
            if not verdicts:
                # JSON 파싱 재시도
                import re as _re
                text = raw if isinstance(raw, str) else json.dumps(raw or {})
                m = _re.search(r'"exit_verdicts"\s*:\s*(\[.*?\])', text, _re.DOTALL)
                if m:
                    verdicts = json.loads(m.group(1))
            for v in (verdicts or []):
                t = v.get("ticker", "")
                h = v.get("horizon", "")
                if t and h:
                    results[f"{t}::{h}"] = v
            _emit("phase5_exit_debate", f"batch_{bi}", f"ok {len(verdicts or [])} verdicts")
        except Exception as e:
            _emit("phase5_exit_debate", f"batch_{bi}", f"fail: {str(e)[:80]}")
            # Fail-safe direction is HOLD (L1): on LLM failure we cannot judge, so the
            # displayed default defers 1 week instead of endorsing an immediate sell.
            # (The state machine's own EXIT_PENDING re-verify path is unaffected.)
            for p in batch:
                key = f"{p.get('ticker', '')}::{p.get('horizon', '')}"
                results[key] = {
                    "ticker": p.get("ticker"), "horizon": p.get("horizon"),
                    "verdict": "HOLD_1W", "confidence": "LOW",
                    "exit_case": "",
                    "hold_case": "LLM 토론 실패 — 판단 불가 시 보유 유지(안전 기본값), 다음 런에서 재평가",
                    "key_factor": "_debate_failed",
                    "debate_transcript": f"[Exit debate 실패 — {str(e)[:80]}] 판단 불가 → 보유 유지 기본값.",
                }

    _emit("phase5_exit_debate", "done", f"verdicts={len(results)}")
    return results
