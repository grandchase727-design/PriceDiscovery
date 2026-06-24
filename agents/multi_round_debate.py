"""
multi_round_debate.py — Multi-Agent ConvictionDebate Orchestrator

Pure-Python helper (NO LLM calls inside this file). All sub-agent invocations
are performed by the calling Claude Code session — this module only:
  1. Builds prompts per persona / round
  2. Parses sub-agent responses into structured opinions
  3. Runs convergence detection
  4. Persists MultiAgentVerdict to .multi_agent_debate_cache.json

Why pure-Python: Claude Max plan rule — LLM/sub-agent work only inside an
active Claude Code session. This file is callable from cron, scripts, etc.,
without needing LLM access. The session injects the parsed verdicts.

Typical in-session flow:
    from agents.debate_selector import select_debate_targets, build_quant_block
    from agents.specialist_prompts import build_round1_prompt, build_round2_prompt
    from agents.multi_round_debate import (
        parse_specialist_opinion, check_convergence,
        save_multi_verdict, build_synthesis_payload,
    )

    targets = select_debate_targets(cache["results"], week_ending="2026-05-29")
    for ticker in targets.all_tier_a():
        row = lookup(ticker)
        quant = build_quant_block(row)
        # Round 1: parallel 3 sub-agent calls
        fund_text = Agent(market-researcher, build_round1_prompt("fundamental", ...))
        sent_text = Agent(market-researcher, build_round1_prompt("sentiment",   ...))
        val_text  = Agent(market-researcher, build_round1_prompt("valuation",   ...))
        r1 = {
            "fundamental": parse_specialist_opinion(fund_text),
            "sentiment":   parse_specialist_opinion(sent_text, with_reflection=True),
            "valuation":   parse_specialist_opinion(val_text),
        }
        if check_convergence(r1):
            rounds = [r1]
        else:
            # Round 2 cross-exam, parallel
            r2 = { ... }
            rounds = [r1, r2]
        # Synthesis (always — but neutral + averse both)
        final = rounds[-1]
        synth_neutral = Agent(claude, build_synthesis_prompt("neutral", ...))
        synth_averse  = Agent(claude, build_synthesis_prompt("averse",  ...))
        save_multi_verdict(ticker, MultiAgentVerdict(...))
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

MULTI_CACHE = Path(".multi_agent_debate_cache.json")


# ─────────────────────────────────────────────────────────────────────
# Data contracts
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SpecialistOpinion:
    persona: str                       # fundamental | sentiment | valuation
    rating: str                        # BUY | HOLD | SELL
    confidence: float                  # 0..1
    key_points: list[str] = field(default_factory=list)
    biggest_risk: str = ""
    biggest_opportunity: str = ""
    raw_text: str = ""
    # Sentiment-only (3-step):
    narrative_summary: str = ""
    critique: str = ""


@dataclass
class RoundSnapshot:
    round_num: int
    fundamental: Optional[SpecialistOpinion] = None
    sentiment:   Optional[SpecialistOpinion] = None
    valuation:   Optional[SpecialistOpinion] = None
    notes: str = ""

    def all_specialists(self) -> list[SpecialistOpinion]:
        return [x for x in [self.fundamental, self.sentiment, self.valuation] if x]


@dataclass
class SynthesisOutput:
    risk_mode: str                     # neutral | averse
    rating: str                        # STRONG_BUY..AVOID
    position_modifier: int             # -5..+5
    sizing_recommendation: str = ""
    reasoning: str = ""
    raw_text: str = ""


@dataclass
class MultiAgentVerdict:
    ticker: str
    tier: str = "A"                    # always "A" for multi-agent
    asset_type: str = "stock"          # stock | etf
    side: str = "long"                 # long | short (Top-N BUY vs Bottom-N SELL)
    group: str = "momentum"            # momentum | pre_momentum (hold-period bucket)
    rounds: list[RoundSnapshot] = field(default_factory=list)
    synthesis_neutral: Optional[SynthesisOutput] = None
    synthesis_averse:  Optional[SynthesisOutput] = None
    converged_round: int = 0
    disagreement: dict = field(default_factory=dict)
    composite_at_time: Optional[float] = None
    classification_at_time: Optional[str] = None
    generated_at: str = ""


# ─────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────

_RATING_RE = re.compile(
    r"\b(STRONG[\s_-]?BUY|BUY|HOLD|SELL|AVOID)\b", re.IGNORECASE
)
_CONF_RE = re.compile(r"\*\*?Confidence\*?\*?:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_MODIFIER_RE = re.compile(r"([+\-]?\d+)")


def _find_rating(text: str) -> str:
    """Find the FIRST rating mention near a 'Rating' label; fallback to first in doc."""
    for line in text.splitlines():
        if "rating" in line.lower():
            m = _RATING_RE.search(line)
            if m:
                return m.group(1).upper().replace(" ", "_").replace("-", "_")
    m = _RATING_RE.search(text)
    if m:
        return m.group(1).upper().replace(" ", "_").replace("-", "_")
    return "HOLD"


def _find_confidence(text: str) -> float:
    m = _CONF_RE.search(text)
    if not m: return 0.5
    try:
        c = float(m.group(1))
        return max(0.0, min(1.0, c))
    except ValueError:
        return 0.5


def _extract_section_after(text: str, keyword: str, lines_max: int = 6) -> str:
    """Get content after a line containing `keyword`, up to lines_max non-empty lines."""
    lines = text.splitlines()
    captured = []
    found = False
    for line in lines:
        if not found and keyword.lower() in line.lower():
            found = True
            continue
        if found:
            s = line.strip()
            if not s:
                if captured: break
                continue
            # Stop at next section header
            if any(c in s for c in ("**", "##", ":")) and len(s) < 80 and "biggest" not in s.lower() and keyword.lower() not in s.lower():
                if not s.startswith(("-", "*", "•")):
                    break
            captured.append(s.lstrip("-*• ").strip())
            if len(captured) >= lines_max:
                break
    return " ".join(captured).strip()


def _extract_bullets_after(text: str, keyword: str, max_bullets: int = 5) -> list[str]:
    """Pull bullet items after a line containing `keyword`."""
    lines = text.splitlines()
    bullets = []
    found = False
    for line in lines:
        if not found and keyword.lower() in line.lower():
            found = True
            continue
        if found:
            s = line.strip()
            if not s: continue
            if s.startswith(("-", "*", "•")):
                bullets.append(s.lstrip("-*• ").strip().strip("*").strip())
                if len(bullets) >= max_bullets:
                    break
            elif bullets and len(s) < 150 and not any(c in s[:10] for c in ("**", "##")):
                # Continuation of previous bullet
                bullets[-1] = bullets[-1] + " " + s
            else:
                # New section header — stop
                if any(c in s for c in ("**", "##")):
                    break
    return bullets


def parse_specialist_opinion(persona: str, raw_text: str,
                              with_reflection: bool = False) -> SpecialistOpinion:
    """Parse a single specialist's free-text response."""
    op = SpecialistOpinion(
        persona=persona,
        rating=_find_rating(raw_text),
        confidence=_find_confidence(raw_text),
        raw_text=raw_text,
        key_points=_extract_bullets_after(raw_text, "key", max_bullets=5)
                  or _extract_bullets_after(raw_text, "observation", max_bullets=5)
                  or _extract_bullets_after(raw_text, "step 1", max_bullets=5),
        biggest_risk=_extract_section_after(raw_text, "biggest", lines_max=2)
                    or _extract_section_after(raw_text, "risk", lines_max=2),
        biggest_opportunity=_extract_section_after(raw_text, "opportunity", lines_max=2),
    )
    if with_reflection:
        op.narrative_summary = _extract_section_after(raw_text, "narrative", lines_max=6)
        op.critique          = _extract_section_after(raw_text, "critique", lines_max=6) \
                            or _extract_section_after(raw_text, "step 2", lines_max=6)
    return op


def parse_synthesis(raw_text: str, risk_mode: str) -> SynthesisOutput:
    """Parse synthesis arbitrator output."""
    rating = _find_rating(raw_text)
    # Find modifier in 'modifier' line
    modifier = 0
    for line in raw_text.splitlines():
        low = line.lower()
        if "modifier" in low or "position-sizing" in low or "position sizing" in low:
            idx = low.find("modifier")
            if idx < 0:
                idx = max(low.find("position-sizing"), low.find("position sizing"))
            tail = line[idx:]
            m = _MODIFIER_RE.search(tail)
            if m:
                try:
                    modifier = max(-5, min(5, int(m.group(1))))
                    break
                except ValueError:
                    pass

    return SynthesisOutput(
        risk_mode=risk_mode,
        rating=rating,
        position_modifier=modifier,
        sizing_recommendation=_extract_section_after(raw_text, "sizing", lines_max=3),
        reasoning=_extract_section_after(raw_text, "reasoning", lines_max=4),
        raw_text=raw_text,
    )


# ─────────────────────────────────────────────────────────────────────
# Convergence + disagreement
# ─────────────────────────────────────────────────────────────────────

_RATING_RANK = {"SELL": 0, "HOLD": 1, "BUY": 2}


def check_convergence(snapshot: RoundSnapshot) -> bool:
    """Return True if 3 specialists have converged (all same rating, or adjacent two-tier only).

    Adjacent = {BUY, HOLD} or {HOLD, SELL}. Polar disagreement {BUY, SELL} = not converged.
    """
    ratings = [op.rating for op in snapshot.all_specialists()]
    if not ratings: return False
    rank = sorted(set(_RATING_RANK.get(r, 1) for r in ratings))
    if len(rank) == 1: return True
    if len(rank) == 2 and abs(rank[0] - rank[1]) == 1: return True
    return False  # 3 tiers spanned OR polar split


def compute_disagreement(snapshot: RoundSnapshot) -> dict:
    """Quantify how much the 3 specialists disagree."""
    ratings = [op.rating for op in snapshot.all_specialists()]
    confidences = [op.confidence for op in snapshot.all_specialists()]
    if not ratings:
        return {"rating_axis": 0, "specialist_dispersion": 0.0, "type": "EMPTY"}

    rank = [_RATING_RANK.get(r, 1) for r in ratings]
    rating_axis = max(rank) - min(rank)

    # Dispersion = stdev of confidence levels
    if len(confidences) >= 2:
        mu = sum(confidences) / len(confidences)
        var = sum((c - mu) ** 2 for c in confidences) / len(confidences)
        dispersion = var ** 0.5
    else:
        dispersion = 0.0

    # Disagreement type
    rset = set(ratings)
    if rating_axis == 0:
        type_ = "CONSENSUS_" + ratings[0]
    elif rset == {"BUY", "HOLD"}:
        type_ = "ENTRY_TIMING"   # bulls + cautious technician
    elif rset == {"SELL", "HOLD"}:
        type_ = "EXIT_TIMING"
    elif rset == {"BUY", "SELL"}:
        type_ = "POLAR_SPLIT"
    elif rating_axis == 2:
        type_ = "WIDE_SPAN"
    else:
        type_ = "MIXED"

    return {
        "rating_axis": int(rating_axis),
        "specialist_dispersion": round(dispersion, 3),
        "type": type_,
    }


def risk_mode_gap(neutral: SynthesisOutput, averse: SynthesisOutput) -> int:
    """How many rating tiers apart are neutral and averse synthesis?"""
    if neutral is None or averse is None: return 0
    tiers = ["STRONG_BUY", "BUY", "HOLD", "SELL", "AVOID"]
    try:
        return abs(tiers.index(neutral.rating) - tiers.index(averse.rating))
    except ValueError:
        return 0


# ─────────────────────────────────────────────────────────────────────
# Build the synthesis payload string (final opinions block)
# ─────────────────────────────────────────────────────────────────────

def build_synthesis_payload(final_round: RoundSnapshot) -> str:
    """Turn the final-round specialists into a single concatenated text block."""
    blocks = []
    for op in final_round.all_specialists():
        blocks.append(
            f"\n### {op.persona.upper()} ANALYST (final)\n"
            f"Rating: {op.rating}  |  Confidence: {op.confidence:.2f}\n"
            f"Key points: " + (" | ".join(op.key_points) if op.key_points else "(none parsed)") + "\n"
            f"Biggest risk: {op.biggest_risk or '(none)'}\n"
            f"Biggest opportunity: {op.biggest_opportunity or '(none)'}"
        )
    return "\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────
# Cache I/O
# ─────────────────────────────────────────────────────────────────────

def save_multi_verdict(ticker: str, verdict: MultiAgentVerdict,
                       cache_path: Path = MULTI_CACHE) -> None:
    cache = _load(cache_path)
    cache[ticker] = asdict(verdict)
    cache["_meta"] = {"last_update": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _save(cache, cache_path)


def save_all_multi_verdicts(verdicts: dict[str, MultiAgentVerdict],
                            cache_path: Path = MULTI_CACHE) -> None:
    payload = {t: asdict(v) for t, v in verdicts.items()}
    payload["_meta"] = {
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_verdicts": len(verdicts),
        "tier": "A",
    }
    _save(payload, cache_path)


def load_multi_verdicts(cache_path: Path = MULTI_CACHE) -> dict:
    return _load(cache_path)


def _load(path: Path) -> dict:
    if not path.exists(): return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save(payload: dict, path: Path) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, default=str))
    except Exception:
        pass


def freshness_minutes(cache_path: Path = MULTI_CACHE) -> Optional[int]:
    meta = _load(cache_path).get("_meta", {})
    ts = meta.get("last_update")
    if not ts: return None
    try:
        last = time.mktime(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))
        return int((time.time() - last) / 60)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Dashboard summary builder
# ─────────────────────────────────────────────────────────────────────

def summary_for_dashboard(cache_path: Path = MULTI_CACHE) -> dict:
    """Shape JSON for /api/conviction-debate/multi endpoint."""
    cache = _load(cache_path)
    meta = cache.pop("_meta", {}) if cache else {}
    verdicts = []
    for ticker, v in cache.items():
        if ticker.startswith("_"): continue
        # Round summary (per-round ratings only — full text kept in cache)
        rounds_compact = []
        for r in v.get("rounds", []):
            rounds_compact.append({
                "round": r.get("round_num"),
                "fundamental": {
                    "rating": (r.get("fundamental") or {}).get("rating", "—"),
                    "confidence": (r.get("fundamental") or {}).get("confidence", 0),
                },
                "sentiment": {
                    "rating": (r.get("sentiment") or {}).get("rating", "—"),
                    "confidence": (r.get("sentiment") or {}).get("confidence", 0),
                },
                "valuation": {
                    "rating": (r.get("valuation") or {}).get("rating", "—"),
                    "confidence": (r.get("valuation") or {}).get("confidence", 0),
                },
            })
        neutral = v.get("synthesis_neutral") or {}
        averse  = v.get("synthesis_averse") or {}
        verdicts.append({
            "ticker": ticker,
            "asset_type": v.get("asset_type", "stock"),
            "side": v.get("side", "long"),
            "tier": v.get("tier", "A"),
            "synthesis_neutral": {
                "rating": neutral.get("rating", "HOLD"),
                "position_modifier": neutral.get("position_modifier", 0),
                "sizing_recommendation": neutral.get("sizing_recommendation", ""),
                "reasoning": neutral.get("reasoning", ""),
            },
            "synthesis_averse": {
                "rating": averse.get("rating", "HOLD"),
                "position_modifier": averse.get("position_modifier", 0),
                "sizing_recommendation": averse.get("sizing_recommendation", ""),
                "reasoning": averse.get("reasoning", ""),
            },
            "rounds": rounds_compact,
            "converged_round": v.get("converged_round", 0),
            "disagreement": v.get("disagreement", {}),
            "composite_at_time": v.get("composite_at_time"),
            "classification_at_time": v.get("classification_at_time"),
            "generated_at": v.get("generated_at", ""),
        })

    return {
        "last_update": meta.get("last_update"),
        "stale_minutes": freshness_minutes(cache_path),
        "n_verdicts": len(verdicts),
        "verdicts": verdicts,
    }
