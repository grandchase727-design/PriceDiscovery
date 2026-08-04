# -*- coding: utf-8 -*-
"""entry_timing.py — Entry Timing Score (진입 적시성): "얼마나 늦지 않았는가".

================================================================================
PURPOSE
================================================================================
Composite/RSS는 "모멘텀이 얼마나 강한가"를 측정하지만, 강한 모멘텀 = 이미 많이
오른 종목이라 매수 리스트가 구조적으로 "이미 스트레치된" 종목으로 쏠린다
(예: XBI Composite 74.6인데 RSI 76.9·21일 +21%·이미 고점 근처).

Entry Timing Score(ETS, 0-100)는 이와 직교하는 "지금 진입이 적시인가"를 본다:
  ① CHASE (추격/신전개) — SMA20 이격 + RSI + 고점 근접도로 "이미 확장됐는가"
  ② ACCEL (가속)        — rss_short − rss_long: +면 단기가 장기를 추월(신선), −면 감속(후기)
  ⑤ FRESH (신선도)      — trend_age(추세 나이) + 최근 급등 크기: 갓 시작 vs 오래+과열

높을수록 "지금이 적시(초기)" · 낮을수록 "이미 스트레치(눌림목 대기)".
순수 결정론 함수 — scan-row 필드만 사용, 네트워크/캐시 없음. Composite를 건드리지
않으므로 OER 페널티(강도)와 이중 계상되지 않는다 (강도 vs 타이밍 = 직교 축).
"""
from __future__ import annotations

from typing import Optional


def _num(v, default=None):
    if v is None:
        return default
    try:
        f = float(v)
        return f if f == f else default   # NaN → default
    except (TypeError, ValueError):
        return default


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


# ── weights + thresholds (한 곳에 모아 튜닝 용이) ──
_W_CHASE, _W_ACCEL, _W_FRESH = 0.40, 0.35, 0.25
_RSI_HOT = 80.0            # 이 이상은 극단 과매수 → ETS 하드 캡
_ETS_HARD_CAP_ON_HOT = 35.0
FRESH_THRESHOLD = 60.0    # ETS ≥ → "즉시 매수 가능(적시)"
EXTENDED_THRESHOLD = 40.0 # ETS < → "눌림목 대기(스트레치)"


def compute_entry_timing(row: dict) -> dict:
    """scan-row(dict) → {ets, status, chase_sub, accel_sub, fresh_sub, ...}.

    필드 누락에 견고: sma20_dist 없으면 sma50_dist를 절반 스케일로 폴백, 그 외
    누락값은 중립(50)으로 처리. 어떤 자산군(주식/ETF)에도 동일 적용."""
    rsi   = _num(row.get("rsi"), 50.0)
    pfh   = _num(row.get("pct_from_high"), -5.0)          # 자연히 음수
    rss_s = _num(row.get("rss_short"), 50.0)
    rss_l = _num(row.get("rss_long"), 50.0)
    ret21 = _num(row.get("ret_21d"), 0.0)
    tage  = _num(row.get("trend_age"), 30.0)

    # 확장 게이지 — sma20 이격 우선, 없으면 sma50 이격 × 0.6(장기이격이 더 크므로 축소)
    sma20d = _num(row.get("sma20_dist"))
    if sma20d is None:
        s50 = _num(row.get("sma50_dist"))
        sma20d = (s50 * 0.6) if s50 is not None else 3.0

    # ── ① CHASE (0-100, 높을수록 덜 확장됨=적시) ──
    #   SMA20 이격: 0%↓ → 100, +8%↑ → 0
    chase_sma = _clip(100.0 - sma20d * 12.5)
    #   RSI: 55↓ → 100, 75↑ → 0
    chase_rsi = _clip((75.0 - rsi) / 20.0 * 100.0)
    #   고점 근접: 고점 붙어있으면(≈0%) 확장, -5~-15% 여유면 room. (-pfh-2)*8
    chase_high = _clip((-pfh - 2.0) * 8.0)
    chase_sub = 0.40 * chase_sma + 0.40 * chase_rsi + 0.20 * chase_high

    # ── ② ACCEL (0-100) — rss_short−rss_long, +20→100 / -20→0 ──
    accel_sub = _clip(50.0 + (rss_s - rss_l) * 2.5)

    # ── ⑤ FRESH (0-100) — 추세 나이 + 최근 급등 크기 ──
    age_sub = _clip((60.0 - tage) / 50.0 * 100.0)         # <10 → 100, >60 → 0
    run_sub = _clip((20.0 - max(0.0, ret21)) / 20.0 * 100.0)  # 21일 +20%↑ → 0(이미 과주행)
    fresh_sub = 0.60 * age_sub + 0.40 * run_sub

    ets = _W_CHASE * chase_sub + _W_ACCEL * accel_sub + _W_FRESH * fresh_sub

    # 극단 과매수 하드 캡 — RSI≥80이면 어떤 조합이든 "적시"일 수 없다
    if rsi >= _RSI_HOT:
        ets = min(ets, _ETS_HARD_CAP_ON_HOT)

    ets = round(_clip(ets), 1)
    status = ("FRESH" if ets >= FRESH_THRESHOLD
              else "EXTENDED" if ets < EXTENDED_THRESHOLD
              else "NEUTRAL")
    return {
        "entry_timing_score": ets,
        "entry_timing_status": status,        # FRESH / NEUTRAL / EXTENDED
        "et_chase": round(chase_sub, 1),      # ① 덜 확장됨
        "et_accel": round(accel_sub, 1),      # ② 가속
        "et_fresh": round(fresh_sub, 1),      # ⑤ 신선도
    }


# 한국어 상태 라벨 (프론트/커멘터리 공용)
STATUS_LABEL_KR = {
    "FRESH":    "적시 (즉시 매수 가능)",
    "NEUTRAL":  "보통",
    "EXTENDED": "스트레치 (눌림목 대기)",
}
