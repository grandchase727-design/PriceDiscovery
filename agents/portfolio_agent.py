"""
portfolio_agent.py — Portfolio Agent (종목별 비중 산출 + 누적성과 시계열)

매수 Final List를 입력받아 개별종목 포트폴리오 / ETF 포트폴리오 2개를 구성하고,
전반적 시스템 분석 결과치(Composite·QVR·debate tier·risk·classification)를 종합한
RISK-ADJUSTED CONVICTION 가중으로 종목별 비중을 선택한다.

방법론 (institutional risk-adjusted conviction weighting):
  conviction = 0.50·Composite + 0.22·QVR + 0.28·DebateTier   (각 0-100)
  risk_mult  = clip(1.25 − risk_score·0.0075, 0.5, 1.25)       (저위험 → 고비중)
  class_mult = CONTINUATION 1.0 / FORMATION 0.95 / RECOVERY 0.92 /
               LAGGING_CATCHUP 0.88 / OVEREXTENDED 0.72 / 기타 0.82
  raw_w = conviction × risk_mult × class_mult
  → 각 sleeve(주식/ETF) 내에서 정규화 후 [MIN 2%, MAX 15%] 캡 + 재정규화(water-filling)

성과: 각 포트폴리오의 일별 index(base=1) 시계열 — buy-and-hold 가중합
  value(t) = Σ wᵢ · closeᵢ(t)/closeᵢ(start)
프론트엔드 dual-handle 슬라이더가 [start,end] 구간을 re-base 하여 표시.
"""
from __future__ import annotations

from typing import Optional

# Weighting constants
MIN_W, MAX_W = 0.02, 0.15
TIER_SCORE = {
    "UNANIMOUS": 100.0, "MAJORITY_CLEAN": 82.0, "MAJORITY_DISSENT": 62.0,
    "SOLO_CLEAN": 58.0, "SOLO_DISSENT": 50.0, "SOLO": 45.0,
    "ALL_CAUTION": 30.0, "HELD_NO_EVAL": 55.0,
    # 디베이트 만장일치 거부 — 가중치 산정에서 최하위 (build_portfolios에서 sleeve 제외됨)
    "EXCLUDED": 0.0, "REJECTED": 0.0,
}
CLASS_MULT = {
    "CONTINUATION": 1.00, "FORMATION": 0.95, "RECOVERY": 0.92,
    "LAGGING_CATCHUP": 0.88, "OVEREXTENDED": 0.72,
}


_warned_tiers: set = set()


def _conviction(item: dict, qvr: Optional[float]) -> float:
    composite = float(item.get("composite") or 0.0)
    qvr_v = float(qvr) if qvr is not None else 50.0          # ETF/결측 → 중립 50
    tier = item.get("tier") or item.get("consensus") or ""
    if tier and tier not in TIER_SCORE and tier not in _warned_tiers:
        _warned_tiers.add(tier)
        print(f"⚠️ portfolio_agent: unknown debate tier '{tier}' — defaulting to 50")
    tier_score = TIER_SCORE.get(tier, 50.0)
    return 0.50 * composite + 0.22 * qvr_v + 0.28 * tier_score


def _risk_mult(item: dict) -> float:
    risk = float(item.get("risk_score") or 50.0)            # 낮을수록 안전
    return max(0.5, min(1.25, 1.25 - risk * 0.0075))        # risk0→1.25, 100→0.5


def _class_mult(cls: str) -> float:
    c = str(cls)
    for k, v in CLASS_MULT.items():
        if k in c:
            return v
    return 0.82


def _normalize_with_caps(raw: list[float]) -> list[float]:
    """정규화 + [MIN,MAX] water-filling 캡 (3-pass 수렴)."""
    n = len(raw)
    if n == 0:
        return []
    s = sum(raw) or 1.0
    w = [x / s for x in raw]
    for _ in range(4):
        w = [min(MAX_W, max(MIN_W, x)) for x in w]
        s = sum(w) or 1.0
        w = [x / s for x in w]
    return w


def _sleeve(items: list[dict], qvr_lookup: dict) -> list[dict]:
    """한 sleeve(주식 또는 ETF)의 비중 + 근거 산출."""
    if not items:
        return []
    # Portfolio Composer final_size 소비: 0.0(EXCLUDED_BY_CAP/BUDGET, WATCH)은
    # sleeve에서 제외 — MIN_W 플로어가 0-사이즈 픽에 2%를 보장하지 않도록
    # 정규화 전에 드롭. 필드 없는 행(보유분/레거시 캐시)은 1.0으로 간주.
    def _final_size(it: dict) -> float:
        fs = it.get("final_size")
        try:
            return 1.0 if fs is None else float(fs)
        except (TypeError, ValueError):
            return 1.0

    items = [it for it in items if _final_size(it) > 0]
    if not items:
        return []
    rows = []
    raw = []
    for it in items:
        tk = it.get("ticker", "")
        qvr = qvr_lookup.get(tk)
        conv = _conviction(it, qvr)
        rm = _risk_mult(it)
        cm = _class_mult(it.get("classification", ""))
        fs = _final_size(it)
        # Holdings-aware 배수 (분산 가중 × 상관 페널티) — 기존엔 표시 순서에만
        # 쓰이고 가중치엔 미반영이라 중복 베팅 픽도 풀 비중을 받았음
        try:
            ha_mult = (float(it.get("ha_diversification_weight") or 1.0)
                       * float(it.get("ha_correlation_penalty") or 1.0))
        except (TypeError, ValueError):
            ha_mult = 1.0
        raw_w = conv * rm * cm * fs * ha_mult
        raw.append(raw_w)
        rows.append({
            "ticker": tk,
            "name": it.get("name", "") or tk,
            "sector": it.get("sector", ""),
            "composite": round(float(it.get("composite") or 0), 1),
            "qvr_score": round(float(qvr), 1) if qvr is not None else None,
            "tier": it.get("tier") or it.get("consensus") or "",
            "risk_score": round(float(it.get("risk_score") or 50), 1),
            "classification": it.get("classification", ""),
            "_conviction": round(conv, 1),
            "_risk_mult": round(rm, 3),
            "_class_mult": round(cm, 3),
            "_ha_mult": round(ha_mult, 3),
            "_final_size": round(fs, 2),
        })
    weights = _normalize_with_caps(raw)
    for r, w in zip(rows, weights):
        r["weight"] = round(w * 100.0, 2)                   # percent
        _extra = ""
        if r["_ha_mult"] < 1.0:
            _extra += f" × 분산/상관 ×{r['_ha_mult']:.2f}"
        if r["_final_size"] < 1.0:
            _extra += f" × composer ×{r['_final_size']:.1f}"
        r["rationale"] = (
            f"conviction {r['_conviction']:.0f} "
            f"(Comp {r['composite']:.0f}·QVR {r['qvr_score'] if r['qvr_score'] is not None else '—'}·{r['tier'][:10] or 'tier?'}) "
            f"× risk ×{r['_risk_mult']:.2f} × class ×{r['_class_mult']:.2f}{_extra}"
        )
    rows.sort(key=lambda x: -x["weight"])
    return rows


def build_portfolios(buy_items: list[dict], qvr_lookup: Optional[dict] = None) -> dict:
    """매수 Final List → 개별종목/ETF 포트폴리오 2개 (비중 포함)."""
    qvr_lookup = qvr_lookup or {}

    def _is_etf(it: dict) -> bool:
        bk = str(it.get("bucket") or "").lower()
        if "etf" in bk:
            return True
        if "stock" in bk:
            return False
        return str(it.get("asset_type") or "").lower() == "etf"

    # ENTERED/NEW/HOLDING 만 (EXIT_PENDING/청산 제외) — 실제 보유 의도 종목
    # 당일 디베이트 EXCLUDE 평결(보유분 강제주입 포함)은 sleeve에서 제외:
    # buy_list는 final_list에서 이미 걸러지지만 active_positions는 tier를 그대로
    # 들고 오므로 여기서 단일 지점으로 차단 (보유 유지 여부는 thesis-break가 결정,
    # 여기서는 "추천 비중"에서만 제외).
    def _debate_excluded(it: dict) -> bool:
        if str(it.get("tier") or "").upper() in ("EXCLUDED", "REJECTED"):
            return True
        return str(it.get("final_decision") or "").upper() == "EXCLUDE"

    investable = [it for it in buy_items
                  if str(it.get("action") or "").upper() not in ("CLOSE_TODAY",)
                  and it.get("ticker")
                  and not _debate_excluded(it)]
    stocks = [it for it in investable if not _is_etf(it)]
    etfs = [it for it in investable if _is_etf(it)]

    return {
        "stocks": _sleeve(stocks, qvr_lookup),
        "etfs": _sleeve(etfs, qvr_lookup),
        "methodology": (
            "Risk-adjusted conviction weighting: "
            "conviction(0.50·Composite + 0.22·QVR + 0.28·DebateTier) × "
            "risk_mult(저위험↑) × class_mult(CONTINUATION 1.0…OVEREXTENDED 0.72) × "
            "composer final_size(0.5/0, EXCLUDE·캡·WATCH 반영) × "
            "holdings-aware 배수(분산 가중·상관 페널티), "
            "디베이트 EXCLUDED 및 final_size=0 픽은 sleeve 제외, "
            "각 sleeve 내 정규화 + [2%,15%] 캡."
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Cumulative performance time-series (buy-and-hold weighted index, base=1)
# ──────────────────────────────────────────────────────────────────────

def compute_cumulative(holdings: list[dict], price_data: dict,
                       start_date: str) -> tuple[list[str], list[float]]:
    """가중 buy-and-hold 누적 index 시계열.

    Args:
        holdings: [{ticker, weight(%)}]  (weight는 percent)
        price_data: {ticker: DataFrame[Close]} (Date index)
        start_date: 'YYYY-MM-DD' — index base 시점 (이 날 이후 첫 거래일부터)
    Returns:
        (dates[list of 'YYYY-MM-DD'], index_values[base=1.0])
        커버리지 부족 종목은 비중 재정규화로 제외.
    """
    import pandas as pd

    # 가용 종목만 + 비중 재정규화
    avail = []
    for h in holdings:
        tk = h.get("ticker")
        df = price_data.get(tk)
        if df is not None and "Close" in getattr(df, "columns", []) and len(df) > 1:
            avail.append((tk, float(h.get("weight") or 0), df))
    if not avail:
        return [], []
    wsum = sum(w for _, w, _ in avail) or 1.0
    avail = [(tk, w / wsum, df) for tk, w, df in avail]

    # 공통 날짜축 (start_date 이후)
    start_ts = pd.Timestamp(start_date)
    # 각 종목의 close 시리즈를 start_date 이후로 자르고 정규화
    norm_series = {}
    common_idx = None
    for tk, w, df in avail:
        s = df["Close"]
        s = s[s.index >= start_ts]
        if len(s) < 2:
            continue
        base = float(s.iloc[0])
        if base <= 0:
            continue
        norm = s / base                                     # base=1 at start
        norm_series[tk] = (w, norm)
        common_idx = norm.index if common_idx is None else common_idx.union(norm.index)
    if not norm_series or common_idx is None:
        return [], []

    common_idx = common_idx.sort_values()
    # 가중합 (forward-fill로 결측 보간)
    total = pd.Series(0.0, index=common_idx)
    wtot = 0.0
    for tk, (w, norm) in norm_series.items():
        aligned = norm.reindex(common_idx).ffill().bfill()
        total = total + aligned * w
        wtot += w
    if wtot > 0:
        total = total / wtot                                # 재정규화 (커버리지 보정)

    dates = [str(d)[:10] for d in common_idx]
    values = [round(float(v), 5) for v in total.tolist()]
    return dates, values
