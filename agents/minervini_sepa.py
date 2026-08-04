# -*- coding: utf-8 -*-
"""agents/minervini_sepa.py — Mark Minervini SEPA (Specific Entry Point Analysis)
per-ticker evaluator for the 매매전략 panel.

SEPA has three deterministic pillars we surface here:
  1. **Trend Template** — the 8-point Stage-2 uptrend filter (price vs 150/200-day
     MAs, MA stacking + slopes, 52-week range position, RS rating).
  2. **VCP** — Volatility Contraction Pattern (variance ratio + breakout).
  3. **Stage** — Weinstein/Minervini stage (1 base / 2 advance / 3 top / 4 decline).

The authoritative 0-100 SEPA *strength* score is already computed at scan time by
`hedge_strategies.score_minervini_long` and persisted on the row as `minervini_long`
(Stage-2 template 35 + 52w range 25 + RS 20 + VCP 20). This module reads that score
PLUS the individual persisted trend-template inputs to build a human-readable
breakdown + a single ENTER / WATCH / AVOID verdict.

Pure & deterministic — no I/O, no LLM. Every field access is None-guarded so it
degrades gracefully on an older cache that predates the persisted template fields
(the `minervini_long` scalar alone still yields a valid verdict).
"""
from __future__ import annotations

import math
from typing import Optional


def _fnum(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _tri(cond: Optional[bool]) -> Optional[bool]:
    """Normalize a possibly-unknown boolean (None stays None)."""
    return None if cond is None else bool(cond)


def evaluate_sepa(row: dict) -> dict:
    """Build the full SEPA analysis dict from a scan/buy-list row.

    Returns:
      {
        "sepa_score": float|None,          # 0-100 minervini_long
        "stage": str,                       # "Stage 2" / "Stage 1" / "Stage 3" / "Stage 4"
        "trend_template": {"criteria": [{"label","pass"}...], "n_pass": int, "n_known": int},
        "vcp": {"status","label","color"},
        "verdict": "ENTER"|"WATCH"|"AVOID",
        "verdict_label": str, "verdict_color": "green"|"amber"|"red"|"gray",
      }
    """
    row = row or {}

    score = _fnum(row.get("minervini_long"))
    short_score = _fnum(row.get("minervini_short"))

    above150 = row.get("above_sma150")
    above200 = row.get("above_sma200")
    above50  = row.get("above_sma50")
    s150_slope = _fnum(row.get("sma150_slope"))
    s200_slope = _fnum(row.get("sma200_slope"))
    s50_s200   = _fnum(row.get("sma50_sma200_spread"))
    range_pct  = _fnum(row.get("range_pct"))
    pct_from_high = _fnum(row.get("pct_from_high"))
    rss = _fnum(row.get("rss"))

    # ── 8-point Trend Template ──
    def _price_above_150_200():
        if above150 is not None and above200 is not None:
            return bool(above150) and bool(above200)
        if above200 is not None:
            return bool(above200)      # SMA150 미영속(구캐시) → SMA200만으로 부분 판정
        return None

    criteria = [
        {"label": "가격 > SMA150·SMA200", "pass": _tri(_price_above_150_200())},
        {"label": "SMA50 > SMA200 (정배열)", "pass": _tri(s50_s200 > 0) if s50_s200 is not None else None},
        {"label": "SMA200 상승추세",        "pass": _tri(s200_slope > 0) if s200_slope is not None else None},
        {"label": "SMA150 상승추세",        "pass": _tri(s150_slope > 0) if s150_slope is not None else None},
        {"label": "가격 > SMA50",           "pass": _tri(bool(above50)) if above50 is not None else None},
        {"label": "52주 저점 30%+ 상단",    "pass": _tri(range_pct >= 30) if range_pct is not None else None},
        {"label": "52주 고점 25% 이내",     "pass": _tri(pct_from_high >= -25) if pct_from_high is not None else None},
        {"label": "상대강도(RS) ≥ 70",      "pass": _tri(rss >= 70) if rss is not None else None},
    ]
    n_pass  = sum(1 for c in criteria if c["pass"] is True)
    n_known = sum(1 for c in criteria if c["pass"] is not None)
    pass_ratio = (n_pass / n_known) if n_known else 0.0

    # ── VCP (Volatility Contraction Pattern) ──
    vcr = _fnum(row.get("vcr"))
    has_bo = bool(row.get("breakout_20d")) or bool(row.get("breakout_10d"))
    if vcr is None:
        vcp = {"status": "N/A", "label": "VCP 데이터 없음", "color": "gray"}
    elif vcr < 0.7 and has_bo:
        vcp = {"status": "돌파", "label": "변동성 수축 후 돌파(피벗 이탈)", "color": "green"}
    elif vcr < 0.8 and has_bo:
        vcp = {"status": "돌파", "label": "변동성 수축 후 돌파", "color": "green"}
    elif vcr < 0.8:
        vcp = {"status": "수축", "label": "변동성 수축 진행(돌파 대기)", "color": "amber"}
    else:
        vcp = {"status": "없음", "label": "VCP 미형성(변동성 넓음)", "color": "gray"}

    # ── Stage 판정 (Weinstein/Minervini) ──
    below_200 = (above200 is not None and not bool(above200))
    stage4 = below_200 and ((s200_slope is not None and s200_slope < 0)
                            or (short_score is not None and short_score >= 45))
    if stage4:
        stage = "Stage 4"      # 하락 국면
    elif score is not None and pass_ratio >= 0.7 and score >= 60 and not below_200:
        stage = "Stage 2"      # 상승 추세 (actionable)
    elif below_200 or (score is not None and score < 40):
        stage = "Stage 1"      # 바닥/base 형성
    else:
        stage = "Stage 3"      # 고점권/추세 약화

    # ── 단일 verdict (ENTER / WATCH / AVOID) ──
    if stage == "Stage 4":
        verdict, vlabel, vcolor = "AVOID", "SEPA 부적합 (Stage 4 하락)", "red"
    elif stage == "Stage 2" and (score is not None and score >= 60):
        verdict, vlabel, vcolor = "ENTER", f"SEPA 진입 가능 (Stage 2·템플릿 {n_pass}/{n_known})", "green"
    elif score is not None and score >= 50 and pass_ratio >= 0.5:
        verdict, vlabel, vcolor = "WATCH", f"SEPA 관망 (템플릿 {n_pass}/{n_known} 부분충족)", "amber"
    elif score is None:
        verdict, vlabel, vcolor = "WATCH", "SEPA 데이터 없음", "gray"
    else:
        verdict, vlabel, vcolor = "AVOID", f"SEPA 부적합 (추세템플릿 {n_pass}/{n_known})", "red"

    return {
        "sepa_score": score,
        "stage": stage,
        "trend_template": {"criteria": criteria, "n_pass": n_pass, "n_known": n_known},
        "vcp": vcp,
        "verdict": verdict,
        "verdict_label": vlabel,
        "verdict_color": vcolor,
    }
