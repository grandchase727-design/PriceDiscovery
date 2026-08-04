# -*- coding: utf-8 -*-
"""holdings_aware_selection.py — Holdings-aware NEW entry re-ranking (A+B 방식).

================================================================================
PURPOSE
================================================================================

Re-rank NEW buy candidates by considering the EXISTING portfolio (holdings),
so the system doesn't pile new entries into already-crowded sectors/correlated
names.

방식 A — 섹터/테마 집중 cap (Sector concentration):
  신규 후보의 conviction score에 분산 기여 가중치를 곱한다.
  diversification_weight = 1.0 - min(1.0, held_in_sector / SECTOR_CAP)
  → 이미 보유 종목이 많은 섹터의 신규 후보는 감점.

방식 B — 상관관계 인지 (Correlation penalty):
  신규 후보와 보유 종목 간 90일 일간수익률 상관계수를 계산.
  보유 종목 중 최대 상관계수가 높으면 (ρ > 0.8) 중복 베팅으로 간주 → 감점.
  correlation_penalty = 1.0 if max_corr < 0.3
                        = linear taper 0.3~0.8
                        = 0.5 (floor) if max_corr >= 0.8

최종 점수:
  adjusted_score = base_conviction × diversification_weight × correlation_penalty

================================================================================
INTEGRATION
================================================================================

Called from final_list.py during items_by_category build, BEFORE the 20+20 cap.
NEW picks get re-sorted by adjusted_score so diversifying picks rise and
redundant picks fall — the cap then keeps the diversified top set.

Each NEW pick is augmented with:
  - ha_diversification_weight: float (0.3-1.0)
  - ha_correlation_penalty: float (0.5-1.0)
  - ha_max_corr_ticker: str (most-correlated held name)
  - ha_max_corr: float
  - ha_adjusted_score: float
  - ha_rationale: str (Korean explanation)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional


# Tunables
SECTOR_CAP = 4                  # 섹터당 (보유+신규) 목표 상한
CORR_LOOKBACK_DAYS = 90         # 상관계수 계산 윈도우 (거래일)
CORR_HIGH_THRESHOLD = 0.80      # 이 이상이면 중복 베팅 (full penalty)
CORR_LOW_THRESHOLD = 0.30       # 이 이하면 진정한 분산 (no penalty)
CORR_PENALTY_FLOOR = 0.50       # 최대 감점 (50%)
DIVERSIFICATION_FLOOR = 0.30    # 섹터 과집중 시 최저 가중치

# ── 노출(exposure) 버킷 오버라이드 ─────────────────────────────────────
# ETF 랩퍼는 스웜의 free-text sector 라벨이 파편화되어 집중 캡을 우회한다:
# VTV/VBR='Broad', DVY/NOBL='Factor', 360750.KS='Korea_Equity' — 전부 같은
# US 밸류/배당 베타인데 서로 다른 "섹터"로 계산돼 캡이 전혀 안 걸렸음 (2026-07-03
# 스냅샷에서 ETF 슬리브의 44.7%가 이 클러스터). 잘 알려진 랩퍼를 실제 노출
# 버킷으로 정규화한다 (측정 상관: VTV-VBR 0.87, NOBL-DVY 0.78, VTV-DVY 0.67).
ETF_EXPOSURE_BUCKET = {
    # US 밸류 + 배당 팩터 클러스터 (배당 스크린은 밸류 틸트와 사실상 동일 노출)
    "VTV": "US Value/Dividend", "MGV": "US Value/Dividend",
    "IWD": "US Value/Dividend", "VLUE": "US Value/Dividend",
    "VOE": "US Value/Dividend", "VBR": "US Value/Dividend",
    "IJS": "US Value/Dividend",
    "DVY": "US Value/Dividend", "NOBL": "US Value/Dividend",
    "SCHD": "US Value/Dividend", "VIG": "US Value/Dividend",
    "SDY": "US Value/Dividend", "HDV": "US Value/Dividend",
    # US 대형 브로드 베타 (해외 상장 트래커 포함 — 360750.KS/143850.KS는
    # Korea_Equity 라벨이지만 실체는 S&P500 트래커)
    "SPY": "US Broad Beta", "VOO": "US Broad Beta", "IVV": "US Broad Beta",
    "SPLG": "US Broad Beta", "RSP": "US Broad Beta", "DIA": "US Broad Beta",
    "360750.KS": "US Broad Beta", "143850.KS": "US Broad Beta",
    # US 나스닥/대형 성장
    "QQQ": "US Growth/Nasdaq", "QQQM": "US Growth/Nasdaq",
    "133690.KS": "US Growth/Nasdaq",
    # US 소형 블렌드
    "IWM": "US Small Blend", "IJR": "US Small Blend", "VB": "US Small Blend",
}


def _exposure_key(pick_or_pos: dict) -> str:
    """집중도 계산용 노출 키: ETF 오버라이드 → 없으면 sector 라벨."""
    t = (pick_or_pos.get("ticker") or "").upper()
    # .KS 티커는 대문자화해도 그대로 매칭됨
    return ETF_EXPOSURE_BUCKET.get(t) or ETF_EXPOSURE_BUCKET.get(pick_or_pos.get("ticker") or "") \
        or (pick_or_pos.get("sector") or "?")


def _daily_returns(price_df, lookback: int = CORR_LOOKBACK_DAYS):
    """Extract last-N-day daily returns series from a price DataFrame (Close col)."""
    try:
        closes = price_df["Close"].dropna()
        if len(closes) < 20:
            return None
        rets = closes.pct_change().dropna()
        if lookback and len(rets) > lookback:
            rets = rets.iloc[-lookback:]
        return rets
    except Exception:
        return None


def _pairwise_corr(ret_a, ret_b) -> Optional[float]:
    """Pearson correlation of two return series on their overlapping index."""
    if ret_a is None or ret_b is None:
        return None
    try:
        joined = ret_a.align(ret_b, join="inner")
        a, b = joined
        if len(a) < 20:
            return None
        c = a.corr(b)
        if c is None or (isinstance(c, float) and math.isnan(c)):
            return None
        return float(c)
    except Exception:
        return None


def _diversification_weight(sector: str, held_sector_count: Counter,
                              already_added_this_sector: int = 0) -> float:
    """방식 A: 섹터 집중 가중치.

    held_count + already_added vs SECTOR_CAP.
    가득 찬 섹터 → DIVERSIFICATION_FLOOR, 빈 섹터 → 1.0.
    """
    occupied = held_sector_count.get(sector, 0) + already_added_this_sector
    ratio = min(1.0, occupied / max(1, SECTOR_CAP))
    weight = 1.0 - ratio
    return max(DIVERSIFICATION_FLOOR, weight)


def _correlation_penalty(max_corr: Optional[float]) -> float:
    """방식 B: 상관관계 감점 multiplier.

    max_corr < 0.3 → 1.0 (no penalty)
    0.3 ~ 0.8 → linear taper 1.0 → CORR_PENALTY_FLOOR
    >= 0.8 → CORR_PENALTY_FLOOR
    """
    if max_corr is None:
        return 1.0   # no data → no penalty
    if max_corr <= CORR_LOW_THRESHOLD:
        return 1.0
    if max_corr >= CORR_HIGH_THRESHOLD:
        return CORR_PENALTY_FLOOR
    # Linear interpolation
    span = CORR_HIGH_THRESHOLD - CORR_LOW_THRESHOLD
    frac = (max_corr - CORR_LOW_THRESHOLD) / span
    return 1.0 - frac * (1.0 - CORR_PENALTY_FLOOR)


def _base_conviction(pick: dict) -> float:
    """Base conviction score (mirrors final_list _score_for_cap)."""
    return (
        (pick.get("stars") or 0) * 100
        + (pick.get("composite") or 0)
        + (pick.get("days_held") or 0) * 0.1
    )


def rerank_new_picks(new_picks: list[dict], held_positions: list[dict],
                       price_data: dict, sector_cap: int = SECTOR_CAP) -> list[dict]:
    """Re-rank NEW candidates by holdings-aware adjusted score (방식 A+B).

    Args:
        new_picks: list of NEW buy candidate dicts (have ticker, sector, composite, stars)
        held_positions: list of currently-held position dicts (ticker, sector)
        price_data: {ticker: DataFrame[Close, Volume]} from .backtest_price_cache.pkl
        sector_cap: target max positions per sector (held + new)

    Returns:
        new_picks sorted by ha_adjusted_score desc, each augmented with ha_* fields.
    """
    # Held sector distribution — 노출 버킷 키 사용 (라벨 파편화 방지)
    held_sector_count: Counter = Counter(
        _exposure_key(p) for p in held_positions
    )
    held_tickers = [p.get("ticker") for p in held_positions if p.get("ticker")]

    # Pre-compute held returns series (once)
    held_returns: dict = {}
    for t in held_tickers:
        df = price_data.get(t)
        if df is not None:
            r = _daily_returns(df)
            if r is not None:
                held_returns[t] = r

    # Process in conviction order so the strongest pick per sector keeps full weight;
    # subsequent same-sector NEW picks get progressively penalized (intra-batch crowding).
    # 기존 버그: held_sector_count만 사용 → 같은 미보유 섹터 NEW 픽들이 모두 div_w=1.0
    # (Healthcare 3개 무페널티). already_added_this_sector를 threading하여 배치 내 분산 강화.
    new_picks.sort(key=lambda p: -_base_conviction(p))
    added_sector_count: Counter = Counter()
    # 배치 내 상관 비교 풀 — 수락 순서대로 편입. 기존 버그: 보유분만 비교해
    # 같은 날 NEW 픽 5개(VTV/DVY/NOBL/VBR/360750.KS, ρ 0.67-0.87)가 서로
    # 전혀 페널티를 받지 않았음.
    accepted_returns: dict = {}

    # Score each new pick
    for pick in new_picks:
        t = pick.get("ticker")
        sector = _exposure_key(pick)          # 노출 버킷 (라벨 파편화 방지)
        base = _base_conviction(pick)

        # 방식 A: diversification weight (held + 배치 내 이미 추가된 same-sector 수)
        div_w = _diversification_weight(sector, held_sector_count,
                                        already_added_this_sector=added_sector_count.get(sector, 0))

        # 방식 B: correlation penalty — 보유분 + 이미 수락된 배치 내 픽 모두 비교
        cand_df = price_data.get(t)
        cand_ret = _daily_returns(cand_df) if cand_df is not None else None
        max_corr = None
        max_corr_ticker = None
        max_corr_src = "보유"
        if cand_ret is not None:
            for ht, hr in held_returns.items():
                c = _pairwise_corr(cand_ret, hr)
                if c is not None and (max_corr is None or c > max_corr):
                    max_corr, max_corr_ticker, max_corr_src = c, ht, "보유"
            for ht, hr in accepted_returns.items():
                c = _pairwise_corr(cand_ret, hr)
                if c is not None and (max_corr is None or c > max_corr):
                    max_corr, max_corr_ticker, max_corr_src = c, ht, "동일배치"
        corr_pen = _correlation_penalty(max_corr)

        adjusted = base * div_w * corr_pen

        # Rationale (Korean) — held + 배치 내 이미 추가된 same-sector 반영
        parts = []
        held_n = held_sector_count.get(sector, 0)
        batch_n = added_sector_count.get(sector, 0)
        occupied = held_n + batch_n
        if occupied >= sector_cap:
            parts.append(f"⚠ {sector} 점유 {occupied}개 (보유 {held_n}+배치 {batch_n}, cap {sector_cap}) → 분산 감점 ×{div_w:.2f}")
        elif occupied >= 1:
            _src = f"보유 {held_n}" + (f"+배치 {batch_n}" if batch_n else "")
            parts.append(f"{sector} 점유 {occupied}개 ({_src}) → 분산 가중 ×{div_w:.2f}")
        else:
            parts.append(f"✓ {sector} 미점유 섹터 → 분산 기여 (가중 ×{div_w:.2f})")
        # 배치 내 누적 (다음 same-sector 픽에 crowding 페널티 적용)
        added_sector_count[sector] += 1
        if cand_ret is not None and t:
            accepted_returns[t] = cand_ret   # 이후 픽들은 이 픽과도 상관 비교
        if max_corr is not None:
            if max_corr >= CORR_HIGH_THRESHOLD:
                parts.append(f"⚠ {max_corr_src} {max_corr_ticker}와 ρ={max_corr:.2f} (중복 베팅) → ×{corr_pen:.2f}")
            elif max_corr <= CORR_LOW_THRESHOLD:
                parts.append(f"✓ 최대 상관 {max_corr_ticker} ρ={max_corr:.2f} (저상관, 진정 분산)")
            else:
                parts.append(f"{max_corr_src} {max_corr_ticker}와 ρ={max_corr:.2f} → ×{corr_pen:.2f}")

        pick["ha_exposure_bucket"] = sector
        pick["ha_diversification_weight"] = round(div_w, 3)
        pick["ha_correlation_penalty"] = round(corr_pen, 3)
        pick["ha_max_corr"] = round(max_corr, 3) if max_corr is not None else None
        pick["ha_max_corr_ticker"] = max_corr_ticker
        pick["ha_adjusted_score"] = round(adjusted, 2)
        pick["ha_base_score"] = round(base, 2)
        pick["ha_rationale"] = " · ".join(parts)

    # Sort by adjusted score desc
    new_picks.sort(key=lambda p: -(p.get("ha_adjusted_score") or 0))
    return new_picks


def summarize_reranking(new_picks: list[dict]) -> dict:
    """Compact summary of the re-ranking effect for monitoring."""
    n = len(new_picks)
    n_div_penalized = sum(1 for p in new_picks if (p.get("ha_diversification_weight") or 1) < 0.7)
    n_corr_penalized = sum(1 for p in new_picks if (p.get("ha_correlation_penalty") or 1) < 0.9)
    return {
        "n_new": n,
        "n_diversification_penalized": n_div_penalized,
        "n_correlation_penalized": n_corr_penalized,
        "top3_after_rerank": [
            {"ticker": p.get("ticker"), "sector": p.get("sector"),
             "adjusted": p.get("ha_adjusted_score"), "base": p.get("ha_base_score")}
            for p in new_picks[:3]
        ],
    }
