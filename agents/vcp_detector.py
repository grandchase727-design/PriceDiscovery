# -*- coding: utf-8 -*-
"""agents/vcp_detector.py — Mark Minervini VCP (Volatility Contraction Pattern) 정통 탐지.

기존 `vcr = 5일변동성/40일변동성`은 VCP의 '변동성 수축' 한 축만 거칠게 근사한 프록시였다.
이 모듈은 미너비니 정통 방법론의 구조를 결정론적으로 재현한다:

  1. 선행 상승추세(Stage 2 전제) — VCP는 continuation 패턴, 사전 advance 필수.
  2. 연속 수축(contractions) 시퀀스 — 스윙 고→저 되돌림 2~6개("T"/footprints).
  3. 점진적 얕아짐 — 각 수축이 이전보다 얕아짐 (25%→12%→6%→3% 식, 좌→우 타이트닝).
  4. 거래량 고갈(VDU) — 베이스 후반(마지막 수축)의 거래량이 베이스 평균 대비 마름.
  5. 피벗(pivot) — 마지막 수축 상단(저항선) = 매수 트리거; 돌파 시 거래량 급증 확인.
  6. 베이스 깊이/타이트니스 — 전체 베이스 깊이 과도하지 않고, 마지막 수축은 타이트.

순수 결정론 · OHLCV만 입력 · 외부 I/O 없음. Close-only도 동작하나 High/Low가 있으면
수축 깊이(peak-to-trough)가 정확해진다.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from agents.elliott_wave_stops import _find_swing_pivots
except Exception:
    _find_swing_pivots = None


def _sma(a, n):
    if len(a) < n:
        return None
    return float(np.mean(a[-n:]))


def detect_vcp(highs, lows, closes, volumes,
               n_bars: int = 4,
               long_ma: int = 200,             # 선행 상승추세 판정 장기 MA (백테스트는 짧게)
               base_lookback: int = 160,       # 베이스 탐색 창(거래일) — 약 8개월
               max_contractions: int = 6,
               min_contractions: int = 2,
               max_base_depth: float = 40.0,    # 전체 베이스 최대 깊이 %
               max_final_depth: float = 12.0,   # 마지막 수축 타이트 상한 %
               near_pivot_band: float = 8.0,    # 피벗 이하 이 % 이내면 '근접'
               vdu_ratio_max: float = 0.90,     # 거래량 고갈 판정(마지막/베이스)
               breakout_vol_mult: float = 1.4   # 돌파 거래량 급증 배수
               ) -> dict:
    """OHLCV로 VCP를 탐지해 구조화 결과를 반환.

    반환 keys: is_vcp(bool), stage(str), verdict(str), reason(str),
      contractions(list[%]), n_contractions(T), final_depth, base_depth, base_weeks,
      progressive(bool), vdu(bool), vdu_ratio, pivot, pivot_buy, dist_to_pivot_pct,
      prior_uptrend(bool), breakout(bool).
    """
    H = np.asarray(highs, dtype=float)
    L = np.asarray(lows, dtype=float)
    C = np.asarray(closes, dtype=float)
    V = np.asarray(volumes, dtype=float) if volumes is not None else None
    n = len(C)
    out = {
        "is_vcp": False, "stage": "NONE", "verdict": "VCP 미형성",
        "reason": "", "contractions": [], "n_contractions": 0,
        "final_depth": None, "base_depth": None, "base_weeks": None,
        "progressive": False, "vdu": False, "vdu_ratio": None,
        "pivot": None, "pivot_buy": None, "dist_to_pivot_pct": None,
        "prior_uptrend": False, "breakout": False,
    }
    if n < 60 or _find_swing_pivots is None:
        out["reason"] = "데이터 부족"
        return out

    cur = float(C[-1])

    # ── 1) 선행 상승추세 (Stage 2 전제) ──
    # 베이스 저점에선 price가 50일선 아래로 잠깐 빠질 수 있으므로 cur>sma50은 요구하지 않음.
    # 핵심: 장기추세 살아있음(price>200일선, 50>200 정배열, 200일선 상승) + 150일선 위.
    sma50, smaL = _sma(C, 50), _sma(C, long_ma)
    prior = False
    if sma50 and smaL:
        prevL = _sma(C[:-21], long_ma)
        rising = (prevL is not None and smaL > prevL)
        prior = (cur > smaL) and (sma50 > smaL) and rising
    out["prior_uptrend"] = bool(prior)

    # ── 2) 스윙 피벗 → 수축 시퀀스 (베이스 창 내) ──
    lo = max(0, n - base_lookback)
    piv = _find_swing_pivots(H[lo:], L[lo:], n_bars=n_bars)  # [(i, price, 'H'/'L')]
    piv = [(i + lo, p, k) for (i, p, k) in piv]
    # 수축 = 스윙 고(H) → 다음 스윙 저(L). 좌→우 순서.
    contractions = []  # (h_idx, h_price, l_idx, l_price, depth%)
    for j in range(len(piv) - 1):
        if piv[j][2] == 'H' and piv[j + 1][2] == 'L':
            hp, lp = piv[j][1], piv[j + 1][1]
            if hp > 0:
                depth = (hp - lp) / hp * 100.0
                contractions.append((piv[j][0], hp, piv[j + 1][0], lp, depth))
    if not contractions:
        out["reason"] = "수축 미탐지"
        return out

    # ── 베이스 앵커링: 가장 깊은 최근 수축(=베이스 시작)부터의 '타이트닝 레그'만 사용 ──
    # 전체 창의 상승 되돌림까지 뒤섞지 않도록, 최근 max_contractions개 중 '가장 깊은 수축'
    # 이후부터를 베이스로 본다(그 지점이 베이스의 좌측 립 = 최대 조정). 최소 2개 확보되게.
    window = contractions[-max_contractions:]
    deepest_pos = max(range(len(window)), key=lambda k: window[k][4])
    leg = window[deepest_pos:]
    if len(leg) < min_contractions:                # 가장 깊은 게 너무 오른쪽 → 최근 것들 사용
        leg = window[-max(min_contractions, 3):]
    contractions = leg
    depths = [round(c[4], 1) for c in contractions]
    out["contractions"] = depths
    out["n_contractions"] = len(contractions)

    if len(contractions) < min_contractions:
        out["reason"] = f"수축 {len(contractions)}개 (< {min_contractions}) — VCP 미형성/초기"
        out["stage"] = "FORMING" if len(contractions) == 1 else "NONE"
        _hs = [p for p in piv if p[2] == 'H']
        if _hs:
            out["pivot"] = round(_hs[-1][1], 2)
            out["dist_to_pivot_pct"] = round((cur / _hs[-1][1] - 1) * 100, 1)
        return out

    # ── 3) 점진적 얕아짐 (핵심) ── 타이트닝 레그가 우측으로 좁아지는지 검증
    final_depth = depths[-1]
    max_leg = max(depths)
    #  (a) 마지막 수축이 레그 최대 대비 충분히 타이트(≤0.6×), (b) 마지막이 직전보다 크게
    #  넓어지지 않음(≤1.25×), (c) 마지막이 절대적으로 타이트(≤max_final_depth).
    overall_tighten = final_depth <= max_leg * 0.60
    recent_ok = (len(depths) < 2) or (depths[-1] <= depths[-2] * 1.25)
    prog = bool(overall_tighten and recent_ok and final_depth <= max_final_depth)
    out["progressive"] = prog
    out["final_depth"] = final_depth

    # ── 4) 베이스 깊이/기간 ──
    base_i0 = contractions[0][0]
    base_high = float(np.max(H[base_i0:]))
    base_low = float(np.min(L[base_i0:]))
    base_depth = (base_high - base_low) / base_high * 100.0 if base_high > 0 else 0.0
    out["base_depth"] = round(base_depth, 1)
    out["base_weeks"] = round((n - 1 - base_i0) / 5.0, 1)

    # ── 5) 피벗 = 마지막 수축의 고(저항선), 매수가 = 피벗+0.1% ──
    pivot = float(contractions[-1][1])
    # 마지막 수축 이후 더 높은 스윙고가 있으면 그걸 피벗으로(재랠리)
    later_h = [p[1] for p in piv if p[2] == 'H' and p[0] > contractions[-1][0]]
    if later_h:
        pivot = max(pivot, float(max(later_h)))
    out["pivot"] = round(pivot, 2)
    out["pivot_buy"] = round(pivot * 1.001, 2)
    out["dist_to_pivot_pct"] = round((cur / pivot - 1) * 100, 1)

    # ── 6) 거래량 고갈(VDU): 마지막 수축 구간 vs 베이스 평균 ──
    vdu_ratio = None
    if V is not None and len(V) == n:
        f_i0 = contractions[-1][0]              # 마지막 수축 시작(고점)부터 현재까지
        base_vol = float(np.mean(V[base_i0:])) or 1.0
        final_vol = float(np.mean(V[f_i0:])) if n - f_i0 >= 2 else base_vol
        vdu_ratio = final_vol / base_vol
        out["vdu_ratio"] = round(vdu_ratio, 2)
        out["vdu"] = bool(vdu_ratio <= vdu_ratio_max)

    # ── 7) 돌파 여부 (현재가 > 피벗 + 최근 거래량 급증) ──
    breakout = False
    if cur > pivot:
        if V is not None and len(V) == n:
            recent_vol = float(np.mean(V[-3:]))
            avg_vol = float(np.mean(V[-50:])) or 1.0
            breakout = recent_vol >= breakout_vol_mult * avg_vol
        else:
            breakout = True
    out["breakout"] = bool(breakout)

    # ── 8) 종합 판정 ──
    dist = out["dist_to_pivot_pct"]
    if not prior:
        out["verdict"] = "VCP 미형성 (선행 상승추세 없음)"
        out["reason"] = f"수축 {len(depths)}개 {depths} 있으나 Stage 2 전제 미충족(가격이 상승 MA 위 아님)"
        return out
    if not prog:
        out["verdict"] = "VCP 미형성 (수축 점진적 아님)"
        out["reason"] = f"수축 {depths} — 좌→우 얕아짐 조건 불충족(넓어지거나 불규칙)"
        return out
    if base_depth > max_base_depth:
        out["verdict"] = "VCP 미형성 (베이스 과도)"
        out["reason"] = f"베이스 깊이 {base_depth:.0f}% > {max_base_depth:.0f}% (wide & loose)"
        return out

    # 유효 VCP
    out["is_vcp"] = True
    tight = final_depth <= max_final_depth
    vdu_txt = f"VDU {vdu_ratio:.2f}" if vdu_ratio is not None else "VDU N/A"
    if breakout:
        out["stage"] = "BREAKOUT"
        out["verdict"] = "VCP 돌파 (피벗 이탈 + 거래량)"
    elif -near_pivot_band <= dist <= 1.0 and tight:
        out["stage"] = "ACTIONABLE"
        out["verdict"] = "VCP 완성 (피벗 근접 · 돌파 대기)"
    elif tight:
        out["stage"] = "FORMED"
        out["verdict"] = "VCP 형성 (피벗 아래 · 셋업 대기)"
    else:
        out["stage"] = "FORMING"
        out["verdict"] = "VCP 형성 중 (마지막 수축 넓음)"
    out["reason"] = (f"T={len(depths)} 수축 {depths} (점진적 얕아짐✓) · 마지막 {final_depth:.1f}%"
                     f"{'(타이트)' if tight else '(넓음)'} · 베이스 {base_depth:.0f}%/{out['base_weeks']:.0f}주 · "
                     f"{vdu_txt}{'(고갈✓)' if out['vdu'] else ''} · 피벗 {out['pivot']} (현재 {dist:+.1f}%)")
    return out
