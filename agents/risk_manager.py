# -*- coding: utf-8 -*-
"""risk_manager.py — 3-Agent Voting의 Risk Manager (Phase 5.7).

각 pick에 대해 deterministic risk score를 계산하고 vote 산출:
  - APPROVE  (YES, low risk)
  - CAUTION  (ABSTAIN, mid risk)
  - REJECT   (NO, high risk)

Risk score 0-100 (높을수록 위험):
  - Overheating risk     (35%): OER + RSI + classification
  - Volatility risk      (25%): 연환산 변동성 + 단기 변동성
  - Liquidity risk       (15%): ADV (1d 거래대금)
  - Concentration risk   (15%): sector 집중도
  - Drawdown/momentum    (10%): TFS_short + 분류 약화 시그널

설계 원칙: LLM 호출 없이 scan_cache 데이터만으로 deterministic 평가.
"""
from __future__ import annotations

from typing import Optional


# ─── Risk score weights ────────────────────────────────────────────
WEIGHT_OVERHEATING   = 0.35
WEIGHT_VOLATILITY    = 0.25
WEIGHT_LIQUIDITY     = 0.15
WEIGHT_CONCENTRATION = 0.15
WEIGHT_DRAWDOWN      = 0.10


# ─── Risk vote thresholds ──────────────────────────────────────────
RISK_VOTE_APPROVE_MAX = 35   # score ≤ 35 → APPROVE (YES)
RISK_VOTE_REJECT_MIN  = 55   # score ≥ 55 → REJECT  (NO)
# Between 35 and 55 → CAUTION (ABSTAIN)


# ─── Per-pick risk computation ─────────────────────────────────────

def _overheating_score(scan_row: dict) -> float:
    """Overheating risk 0-100. OER, RSI, classification 기반."""
    oer = scan_row.get("oer") or 0
    cls = scan_row.get("classification", "")
    rsi = scan_row.get("rsi") or 50

    score = 0.0
    # OER component (40 points max)
    if oer >= 80:    score += 40
    elif oer >= 70:  score += 30
    elif oer >= 60:  score += 20
    elif oer >= 50:  score += 10
    elif oer >= 40:  score += 5

    # RSI extreme (30 points max)
    if rsi >= 80:    score += 30
    elif rsi >= 70:  score += 20
    elif rsi >= 65:  score += 10

    # Classification danger flags (30 points max)
    if "OVEREXTENDED" in cls:   score += 25
    if "CYCLE_PEAK" in cls:     score += 30
    if "EXHAUSTING" in cls:     score += 20

    return min(100, score)


def _volatility_score(scan_row: dict) -> float:
    """Volatility risk 0-100. 연환산 변동성 기준."""
    vol = scan_row.get("vol_3y_ann") or scan_row.get("vol_ann") or 25
    if vol >= 60:    return 100
    if vol >= 50:    return 80
    if vol >= 40:    return 60
    if vol >= 30:    return 40
    if vol >= 25:    return 25
    return 10


def _liquidity_score(scan_row: dict) -> float:
    """Liquidity risk 0-100. ADV (1일 거래대금) 기반."""
    adv_M = scan_row.get("adv_M") or scan_row.get("adv_usd_M") or 10
    if adv_M < 1:    return 100
    if adv_M < 5:    return 70
    if adv_M < 20:   return 40
    if adv_M < 50:   return 20
    return 5


def _concentration_score(sector: str, sector_counts: dict) -> float:
    """Concentration risk 0-100. 같은 sector picks 수 기반."""
    n = sector_counts.get(sector, 0)
    if n >= 8:   return 100
    if n >= 6:   return 70
    if n >= 5:   return 50
    if n >= 4:   return 30
    if n >= 3:   return 15
    return 5


def _drawdown_score(scan_row: dict) -> float:
    """Drawdown/momentum-weakening risk."""
    tfs_short = scan_row.get("tfs_short") or 50
    cls = scan_row.get("classification", "")

    score = 0
    # Weak short-term trend
    if tfs_short < 30:   score += 50
    elif tfs_short < 40: score += 30
    elif tfs_short < 50: score += 15

    # Weakening classification
    if any(w in cls for w in ("FADING", "WEAKENING", "PULLBACK", "DOWNTREND")):
        score += 40

    return min(100, score)


def compute_risk_score(scan_row: dict, sector_counts: Optional[dict] = None) -> dict:
    """Per-pick risk score breakdown + total.

    Args:
        scan_row: scan_cache row for this ticker (composite, oer, vol_3y_ann, etc.)
        sector_counts: dict like {"Technology": 5, "Healthcare": 3, ...} for concentration

    Returns:
        {
          "total": 0-100,
          "overheating": ...,
          "volatility": ...,
          "liquidity": ...,
          "concentration": ...,
          "drawdown": ...
        }
    """
    if sector_counts is None:
        sector_counts = {}
    sector = scan_row.get("sector", "") or scan_row.get("category", "")

    breakdown = {
        "overheating":   _overheating_score(scan_row),
        "volatility":    _volatility_score(scan_row),
        "liquidity":     _liquidity_score(scan_row),
        "concentration": _concentration_score(sector, sector_counts),
        "drawdown":      _drawdown_score(scan_row),
    }
    total = (
        WEIGHT_OVERHEATING   * breakdown["overheating"] +
        WEIGHT_VOLATILITY    * breakdown["volatility"] +
        WEIGHT_LIQUIDITY     * breakdown["liquidity"] +
        WEIGHT_CONCENTRATION * breakdown["concentration"] +
        WEIGHT_DRAWDOWN      * breakdown["drawdown"]
    )
    breakdown["total"] = round(total, 1)
    return breakdown


def risk_vote(risk_breakdown: dict) -> str:
    """Convert risk score to vote: APPROVE / CAUTION / REJECT."""
    score = risk_breakdown.get("total", 50)
    if score <= RISK_VOTE_APPROVE_MAX: return "APPROVE"
    if score >= RISK_VOTE_REJECT_MIN:  return "REJECT"
    return "CAUTION"


def risk_rationale(risk_breakdown: dict) -> str:
    """Per-ticker Korean Risk Manager reasoning (detailed).

    Reports total score + verdict + top 2 risk drivers + mitigation advice.
    """
    score = risk_breakdown.get("total", 0)

    # Verdict
    if score <= RISK_VOTE_APPROVE_MAX:
        verdict = "✓ 리스크 양호"
        advice = "정상 사이즈 진입 가능, 일반 stop-loss 설정"
    elif score >= RISK_VOTE_REJECT_MIN:
        verdict = "✗ 리스크 높음"
        advice = "포지션 축소 또는 회피 권장"
    else:
        verdict = "○ 리스크 중간"
        advice = "절반 사이즈로 진입 + 타이트한 stop-loss"

    # Rank risk components by score
    components = [
        ("과열 (Overheating)",     risk_breakdown.get("overheating", 0)),
        ("변동성 (Volatility)",    risk_breakdown.get("volatility", 0)),
        ("유동성 (Liquidity)",     risk_breakdown.get("liquidity", 0)),
        ("섹터 집중 (Concentration)", risk_breakdown.get("concentration", 0)),
        ("추세 약화 (Drawdown)",   risk_breakdown.get("drawdown", 0)),
    ]
    components.sort(key=lambda x: -x[1])

    # Build detailed breakdown
    top_risks = [f"{name} {val:.0f}" for name, val in components[:2] if val >= 30]
    if not top_risks:
        risk_detail = "5개 차원 모두 안전 구간"
    else:
        risk_detail = "주요 리스크: " + " · ".join(top_risks)

    return f"{verdict} (총점 {score:.0f}/100) — {risk_detail}. {advice}."


# ─── PM Agent vote (based on conviction) ────────────────────────────

def pm_vote(pick: dict) -> str:
    """PM Agent's vote — based on whether the actual PM Agent (LLM) picked this ticker.

    REWRITTEN (Option 1, 2026-06-21):
      Previous version was a deterministic heuristic (composite ≥ 70 → APPROVE) that
      contradicted the actual PM Agent's nuanced LLM judgment. E.g. SPHQ at composite
      69.4 was CONTINUATION + recommended by PM but the heuristic returned CAUTION.

    NEW vote rules — reflect the LLM PM Agent's actual decision:
      REJECT   : pick was DEMOTED or DROPPED by PM Core override (vs Phase 4 draft),
                 or in phase4_drops list, or marked for removal
      CAUTION  : PM included with reduced size (final_decision = INCLUDE_REDUCED_SIZE
                 or WATCH from Debate Synthesizer)
      APPROVE  : PM included this ticker (default — because PM Agent picked it)
    """
    change_type = pick.get("change_type", "") or ""
    final_decision = (pick.get("debate_synthesis") or {}).get("final_decision", "")

    # If PM Core override DEMOTED/DROPPED this pick → REJECT
    if change_type in ("DEMOTED", "DROPPED"):
        return "REJECT"
    if final_decision == "EXCLUDE":
        return "REJECT"

    # If Debate Synthesizer flagged for reduced size or watch-only → CAUTION
    if final_decision in ("INCLUDE_REDUCED_SIZE", "WATCH"):
        return "CAUTION"

    # Default: PM Agent picked this ticker → APPROVE
    # (the pick exists in PM swarm output, which is the PM's recommendation)
    return "APPROVE"


# ─── Trading Agent vote (existing entry_signal) ─────────────────────

def trading_vote(timing: dict) -> str:
    """Trading Agent's vote from entry_signal."""
    sig = (timing or {}).get("entry_signal", "WAIT")
    if sig == "BUY_NOW": return "APPROVE"
    if sig == "SKIP":    return "REJECT"
    return "CAUTION"   # WAIT or unknown


# ─── 3-Agent Voting Synthesis ───────────────────────────────────────

def tally_votes(pm_v: str, trading_v: str, risk_v: str) -> dict:
    """Count approve / reject votes → final tier.

    Returns:
        {
          "n_approve": int,
          "n_reject":  int,
          "n_caution": int,
          "stars":     1 / 2 / 3,
          "consensus": "UNANIMOUS" / "MAJORITY" / "SOLO" / "REJECTED",
        }
    """
    votes = [pm_v, trading_v, risk_v]
    n_approve = votes.count("APPROVE")
    n_reject  = votes.count("REJECT")
    n_caution = votes.count("CAUTION")

    # Stars + consensus
    if n_approve == 3:
        stars, consensus = 3, "UNANIMOUS"
    elif n_approve == 2 and n_reject == 0:
        stars, consensus = 3, "MAJORITY_CLEAN"   # 2 APPROVE + 1 CAUTION
    elif n_approve == 2:
        stars, consensus = 2, "MAJORITY_DISSENT" # 2 APPROVE + 1 REJECT
    elif n_approve == 1 and n_reject == 0:
        stars, consensus = 2, "SOLO_CLEAN"       # 1 APPROVE + 2 CAUTION
    elif n_approve == 1:
        stars, consensus = 1, "SOLO_DISSENT"
    elif n_reject >= 2:
        stars, consensus = 0, "REJECTED"
    else:
        stars, consensus = 1, "ALL_CAUTION"      # 0 APPROVE + 3 CAUTION

    return {
        "n_approve": n_approve,
        "n_reject":  n_reject,
        "n_caution": n_caution,
        "stars": stars,
        "consensus": consensus,
    }
