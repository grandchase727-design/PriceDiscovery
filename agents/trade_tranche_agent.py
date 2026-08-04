# -*- coding: utf-8 -*-
"""trade_tranche_agent.py — "목표매매횟수" (분할 매매 횟수) — DETERMINISTIC.

Tranche count is a risk-sizing RULE (volatility × conviction × urgency), not a
judgement that needs an LLM. The previous LLM implementation hung (claude
subprocess could stall holding a Max-plan session slot), so it is replaced with a
pure formula: instant, reliable, explainable, no subprocess, no cache, no hang.

  • BUY  (신규 진입)  — 몇 번에 나누어 *매수*       (1–4회)
  • ADD  (보유 종목)  — 추가 매수가 타당하면 몇 번에 (0–3회; 0 = 추가매수 비권장)
  • SELL (청산 대상)  — 몇 번에 나누어 *매도*       (1–4회)

Fields written onto each pick:
  tranche_count      : int   (목표 매매 횟수)
  tranche_reasoning  : str   (한국어 근거)
  tranche_mode       : str   ("BUY" / "ADD" / "SELL")
"""
from __future__ import annotations

from typing import Optional

# count clamp per mode → (min, max)
_CLAMP = {"BUY": (1, 4), "ADD": (0, 3), "SELL": (1, 4)}

_BEARISH = ("DOWNTREND", "WEAKENING", "FADING", "EXHAUSTING", "CYCLE_PEAK", "COUNTER_RALLY")


def _num(v, default=None):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default


def _clamp(n: int, mode: str) -> int:
    lo, hi = _CLAMP[mode]
    return max(lo, min(hi, int(round(n))))


def _ctx(row: dict) -> dict:
    """Decision-relevant signals (all from existing final-list fields)."""
    cls = (row.get("classification") or "")
    ret1m = _num(row.get("ret_1mo"))
    return {
        "composite": _num(row.get("composite"), 0) or 0,
        "cls": cls,
        "bearish": any(b in cls for b in _BEARISH),
        "stars": int(_num(row.get("stars"), 0) or 0),
        "tier": (row.get("tier") or ""),
        "oer": _num(row.get("oer"), 0) or 0,
        "stop_pct": abs(_num(row.get("stop_pct"), 0) or 0),
        "extended": (row.get("entry_primary_status") == "extended"),
        "ret1m_pct": (ret1m * 100) if ret1m is not None else None,
    }


def compute_tranche(row: dict, mode: str) -> tuple[int, str]:
    """Deterministic tranche count + Korean reasoning for one pick."""
    c = _ctx(row)
    comp, oer, stop, stars = c["composite"], c["oer"], c["stop_pct"], c["stars"]
    tier, ext = c["tier"], c["extended"]
    r1 = c["ret1m_pct"]
    vol_bits, conv_bits = [], []

    if mode == "BUY":
        n = 2
        if oer >= 60: n += 1; vol_bits.append(f"OER {oer:.0f} 과열")
        if stop >= 13: n += 1; vol_bits.append(f"손절거리 {stop:.0f}% 넓음")
        if r1 is not None and abs(r1) >= 15: n += 1; vol_bits.append(f"1M {r1:+.0f}% 급등락")
        if ext: n += 1; vol_bits.append("이미 돌파+연장(extended)")
        if comp >= 75 and stars >= 3: n -= 1; conv_bits.append(f"고신뢰(Comp {comp:.0f}·★{stars})")
        if tier == "UNANIMOUS": n -= 1; conv_bits.append("만장일치")
        n = _clamp(n, "BUY")
        if n == 1:
            why = f"{('·'.join(conv_bits) or '신뢰 양호')} + 변동성 낮아 1회 집중 매수."
        else:
            why = (f"{('·'.join(vol_bits) or '변동성 보통')}로 평단 리스크 분산 위해 {n}회 분할 매수."
                   + (f" ({'·'.join(conv_bits)}로 과분할 억제)" if conv_bits else ""))
        return n, why

    if mode == "ADD":
        # 추가매수 타당성 게이트
        block = []
        if ext: block.append("이미 extended")
        if c["bearish"]: block.append(f"분류 약세({c['cls'][:10]})")
        if comp < 55: block.append(f"Comp {comp:.0f} 낮음")
        if stars < 2: block.append(f"★{stars} 낮은 신뢰")
        if tier in ("REGIME_FLIP", "MAJORITY_DISSENT"): block.append(f"{tier}")
        if block:
            return 0, f"추가매수 비권장(0회): {'·'.join(block)} — 기존 보유 유지/관망."
        n = 1
        if comp >= 70 and ("CONTINUATION" in c["cls"]): n += 1; conv_bits.append(f"건강한 추세(Comp {comp:.0f}·CONTINUATION)")
        if oer < 40: n += 1; vol_bits.append(f"OER {oer:.0f} 미과열(여유)")
        n = _clamp(n, "ADD")
        why = (f"추가매수 타당 — {('·'.join(conv_bits + vol_bits) or '추세 유지')}으로 {n}회 분할 추가.")
        return n, why

    # SELL
    n = 2
    urgent = []
    if tier == "REGIME_FLIP": urgent.append("REGIME_FLIP")
    if comp < 30: urgent.append(f"Comp {comp:.0f} 추세붕괴")
    if r1 is not None and r1 <= -10: urgent.append(f"1M {r1:.0f}% 손실확대")
    if c["bearish"]: urgent.append(f"분류 {c['cls'][:10]}")
    if urgent:
        n = 1 if (comp < 25 or (r1 is not None and r1 <= -15)) else 2
        why = f"긴급 청산 {n}회: {'·'.join(urgent)} — 신속 처분."
    else:
        n = 3
        if comp >= 45: n = 4
        why = f"질서있는 분할 매도 {n}회: Comp {comp:.0f}로 일부 지지 잔존 — 단계적 이익/포지션 축소."
    return _clamp(n, "SELL"), why


def annotate_picks_with_tranches(picks: list, mode: str, **_ignored) -> dict:
    """Annotate picks with tranche_count / tranche_reasoning / tranche_mode (instant).

    Signature kept compatible with callers (extra kwargs ignored). No LLM, no cache.
    """
    if mode not in _CLAMP:
        raise ValueError(f"bad mode {mode}")
    n = 0
    for p in picks or []:
        if not p.get("ticker"):
            continue
        cnt, why = compute_tranche(p, mode)
        p["tranche_count"] = cnt
        p["tranche_reasoning"] = why
        p["tranche_mode"] = mode
        n += 1
    return {"mode": mode, "total": len(picks or []), "computed": n, "from_cache": 0, "failed": 0}


def refresh_all_tranches(buy_list: list, active_positions: list, exit_pending: list,
                         **_ignored) -> dict:
    """Annotate all three Final-List sources with mode-appropriate tranche counts.

      buy_list → BUY · active_positions → ADD · exit_pending → SELL.
    Deterministic + instant — safe to call inline on every /api/final-list request.
    """
    return {
        "buy":  annotate_picks_with_tranches(buy_list or [], "BUY"),
        "add":  annotate_picks_with_tranches(active_positions or [], "ADD"),
        "sell": annotate_picks_with_tranches(exit_pending or [], "SELL"),
    }
