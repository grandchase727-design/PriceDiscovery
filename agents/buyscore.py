"""
buyscore.py — Python mirror of the frontend ConvictionPicks BuyScore formula.

Mirrors MarketCommentaryTab.tsx:computeConvictionPicks() so that the
debate_selector and the Conviction Picks panel surface the SAME top-N
tickers — eliminating the "two lists, different names" confusion.

Score components (matches frontend ±10% — see notes for known small deltas):
  + composite
  + bullish_class_bonus       (CONTINUATION/FORMATION/RECOVERY/etc.)
  − bearish_class_penalty     (DOWNTREND/WEAKENING/CYCLE_PEAK)
  − oer_penalty               max(0, OER−50) × 0.4
  + consensus_bonus           strategy_agreement × 4
  + ret_1m_bonus              ret_21d × 0.3, capped ±5
  + ytd_context               +2 / −2 / 0
  + sector_bonus              +6 if bull sector / −6 if bear / 0
  − weak_qvr_penalty          15 if Stock & QVR<40
  + etf_flag_bonus            +10/+6/+2/−4 by divergence_flag × coverage_weight
"""

from __future__ import annotations

import re
from typing import Optional

# ── EXACT mirror of frontend constants (MarketCommentaryTab.tsx:2652-2677) ──
# Frontend uses emoji-prefixed keys (raw classification string from scan_cache).
BULLISH_CLASS_BONUS = {
    "🟢 CONTINUATION":    20,
    "🔵 FORMATION":       18,
    "🟦 LAGGING_CATCHUP": 14,
    "🔵 RECOVERY":        12,
    "🟡 OVEREXTENDED":    -8,   # bullish-but-stretched → PENALTY in BuyScore
}

BEARISH_CLASS_PENALTY = {
    "🔴 CYCLE_PEAK":      35,
    "⬇️ DOWNTREND":       30,
    "⚠️ WEAKENING":       22,
    "🟤 EXHAUSTING":      20,
    "🟤 FADING":          18,
    "🟣 COUNTER_RALLY":   15,
}

WEAK_QVR_PENALTY = 15
WEAK_QVR_THRESHOLD = 40

# ── Classification-group split (2026-05 — Pre-Momentum vs Momentum stratification) ──
# Goal: produce two distinct holding-period buckets so Tier-A turnover isn't daily.
#   • MOMENTUM    — trend already established, target hold ≤ weekly
#   • PRE_MOMENTUM — trend forming OR re-forming, target hold weekly~bi-weekly
# Buckets are derived from raw (emoji-prefixed) classification strings.
MOMENTUM_LONG_CLASSES = {
    "🟢 CONTINUATION", "🟡 OVEREXTENDED", "🟦 LAGGING_CATCHUP",
}
PRE_MOMENTUM_LONG_CLASSES = {
    "🔵 FORMATION", "🔵 RECOVERY", "🔶 PULLBACK",
}
MOMENTUM_SHORT_CLASSES = {
    "⬇️ DOWNTREND", "⚠️ WEAKENING", "🔴 CYCLE_PEAK",
}
PRE_MOMENTUM_SHORT_CLASSES = {
    "🟤 FADING", "🟤 EXHAUSTING", "🟣 COUNTER_RALLY",
}


def classification_group(cls_raw: str, side: str = "long") -> Optional[str]:
    """Return 'momentum' | 'pre_momentum' | None for a given raw classification.

    side='long' uses bullish classifications; side='short' uses bearish.
    Returns None for classes that don't belong to either bucket (NEUTRAL,
    CONSOLIDATION, COUNTER_RALLY counted under pre_mom short, etc.)
    """
    if not cls_raw:
        return None
    if side == "long":
        if cls_raw in MOMENTUM_LONG_CLASSES:     return "momentum"
        if cls_raw in PRE_MOMENTUM_LONG_CLASSES: return "pre_momentum"
    else:
        if cls_raw in MOMENTUM_SHORT_CLASSES:     return "momentum"
        if cls_raw in PRE_MOMENTUM_SHORT_CLASSES: return "pre_momentum"
    return None

ETF_FLAG_BONUS = {
    "STEALTH_STRENGTH": 10,
    "HEALTHY_TREND":     6,
    "WRAPPER_DRAG":      2,
    "NARROW_RALLY":     -4,
    "NEUTRAL":           0,
}

# Hard filter rejection substrings (exact mirror of frontend HARD_FILTER_REJECTIONS)
_HARD_FILTER_REASONS = {
    "Downtrend", "Exhausting", "Fading", "CounterRally", "CyclePeak", "Weakening",
    "Neutral(PM)", "Consolidation(PM)", "Recovery(PM)", "Pullback(PM)",
    "LowScore", "Liq",
}


def _clean_cls(c: str) -> str:
    return re.sub(r"[^\x00-\x7F]+", "", c or "").strip()


def _parse_rejection(raw: Optional[str]) -> dict:
    """Mirror frontend parseRejection. Returns hard_filter / weak_qvr / score."""
    if not raw or str(raw).lower() in ("none", "null", "nan"):
        return {"hard_filter": False, "weak_qvr": False, "qvr_score": None}
    s = str(raw)
    m = re.search(r"WeakQVR\((\d+)\)", s)
    if m:
        return {"hard_filter": False, "weak_qvr": True, "qvr_score": int(m.group(1))}
    for reason in _HARD_FILTER_REASONS:
        if reason in s:
            return {"hard_filter": True, "weak_qvr": False, "qvr_score": None}
    # Unknown rejection → conservative hard filter
    return {"hard_filter": True, "weak_qvr": False, "qvr_score": None}


def compute_buyscore(
    row: dict,
    bull_sectors: Optional[set[str]] = None,
    bear_sectors: Optional[set[str]] = None,
    consensus_map: Optional[dict[str, int]] = None,
) -> dict:
    """Compute BuyScore for a single scan_cache result row.

    Returns dict with: buyScore, hard_filter, weak_qvr, qvr_score, components.
    """
    bull_sectors = bull_sectors or set()
    bear_sectors = bear_sectors or set()
    consensus_map = consensus_map or {}

    composite = float(row.get("composite") or 0)
    cls_raw   = row.get("classification", "") or ""    # raw form WITH emoji (frontend lookup style)
    oer       = float(row.get("oer") or 0)
    ret_21d   = float(row.get("ret_21d") or 0)
    ret_ytd   = float(row.get("ret_ytd") or 0)
    # Frontend uses row.category which (for /api/table response) is cleaned ("Energy", "Technology").
    # If feeding raw scan_cache (e.g., "STK_Energy"), strip the prefix.
    cat_raw   = row.get("category") or row.get("sector") or "Other"
    sector    = cat_raw.split("_", 1)[1] if isinstance(cat_raw, str) and cat_raw.startswith(("STK_", "EQ_", "FI_", "MA_", "ETF_")) else cat_raw
    asset     = (row.get("asset_type") or row.get("asset") or "").lower()
    flag      = row.get("divergence_flag") or "NEUTRAL"
    coverage  = float(row.get("constituent_coverage") or 0)
    ticker    = row.get("ticker", "")

    rj = _parse_rejection(row.get("rejection"))
    weak_qvr_penalty = WEAK_QVR_PENALTY if rj["weak_qvr"] else 0

    # Lookup using RAW emoji-prefixed classification (matches frontend)
    bullish_bonus   = BULLISH_CLASS_BONUS.get(cls_raw, 0)
    bearish_penalty = BEARISH_CLASS_PENALTY.get(cls_raw, 0)
    oer_penalty     = max(0.0, oer - 50.0) * 0.4
    consensus       = consensus_map.get(ticker, 0)
    consensus_bonus = consensus * 4
    ret_1m_bonus    = max(-5.0, min(5.0, ret_21d * 0.3))
    ytd_context     = 2 if ret_ytd > 10 else (-2 if ret_ytd < -10 else 0)
    sector_bonus    = 6 if sector in bull_sectors else (-6 if sector in bear_sectors else 0)

    # ETF flag bonus (coverage-weighted reliability)
    flag_weight = 1.0 if coverage >= 50 else (0.5 if coverage >= 30 else 0.0)
    etf_flag_bonus = 0.0
    if asset == "etf" and flag:
        etf_flag_bonus = ETF_FLAG_BONUS.get(flag, 0) * flag_weight

    score = (
        composite + bullish_bonus - bearish_penalty - oer_penalty
        + consensus_bonus + ret_1m_bonus + ytd_context + sector_bonus
        - weak_qvr_penalty + etf_flag_bonus
    )

    return {
        "ticker": ticker,
        "buyScore": score,
        "hard_filter": rj["hard_filter"],
        "weak_qvr": rj["weak_qvr"],
        "qvr_score": rj["qvr_score"],
        "components": {
            "composite": composite,
            "bullish_bonus": bullish_bonus,
            "bearish_penalty": bearish_penalty,
            "oer_penalty": oer_penalty,
            "consensus_bonus": consensus_bonus,
            "ret_1m_bonus": ret_1m_bonus,
            "ytd_context": ytd_context,
            "sector_bonus": sector_bonus,
            "weak_qvr_penalty": weak_qvr_penalty,
            "etf_flag_bonus": etf_flag_bonus,
        },
    }


def top_buy_picks(
    all_results: list[dict],
    top_n_stock: int = 5,
    top_n_etf: int = 5,
    bull_sectors: Optional[set[str]] = None,
    bear_sectors: Optional[set[str]] = None,
    consensus_map: Optional[dict[str, int]] = None,
    min_adv_M: float = 1.0,
) -> dict:
    """Return Buy Top-N stocks + ETFs by BuyScore (excluding hard_filter)."""
    scored = []
    for row in all_results:
        if not row.get("ticker"):
            continue
        adv_M = float(row.get("adv_M") or 0)
        if adv_M and adv_M < min_adv_M:
            continue
        s = compute_buyscore(row, bull_sectors, bear_sectors, consensus_map)
        if s["hard_filter"]:
            continue
        s["asset"] = (row.get("asset_type") or row.get("asset") or "").lower()
        scored.append(s)

    stocks = [s for s in scored if s["asset"] == "stock"]
    etfs   = [s for s in scored if s["asset"] == "etf"]

    stocks.sort(key=lambda x: -x["buyScore"])
    etfs.sort(key=lambda x: -x["buyScore"])

    return {
        "stocks": stocks[:top_n_stock],
        "etfs":   etfs[:top_n_etf],
        "pool_size_stock": len(stocks),
        "pool_size_etf":   len(etfs),
    }


def top_sell_picks(
    all_results: list[dict],
    top_n_stock: int = 5,
    top_n_etf: int = 5,
    bull_sectors: Optional[set[str]] = None,
    bear_sectors: Optional[set[str]] = None,
    consensus_map: Optional[dict[str, int]] = None,
    min_adv_M: float = 1.0,
) -> dict:
    """Return Sell Bottom-N stocks + ETFs by BuyScore (ASC).

    Mirror of ConvictionPicks sortedSellStocks/ETFs:
      - NO hard_filter exclusion (bearish/CycleP​eak/Downtrend names INCLUDED)
      - Sorted ASC by BuyScore — most negative scores first
    """
    scored = []
    for row in all_results:
        if not row.get("ticker"):
            continue
        adv_M = float(row.get("adv_M") or 0)
        if adv_M and adv_M < min_adv_M:
            continue
        s = compute_buyscore(row, bull_sectors, bear_sectors, consensus_map)
        s["asset"] = (row.get("asset_type") or row.get("asset") or "").lower()
        scored.append(s)

    stocks = [s for s in scored if s["asset"] == "stock"]
    etfs   = [s for s in scored if s["asset"] == "etf"]

    stocks.sort(key=lambda x: x["buyScore"])    # ASC — worst first
    etfs.sort(key=lambda x: x["buyScore"])

    return {
        "stocks": stocks[:top_n_stock],
        "etfs":   etfs[:top_n_etf],
        "pool_size_stock": len(stocks),
        "pool_size_etf":   len(etfs),
    }


# ─────────────────────────────────────────────────────────────────────
# Group-split selection (Momentum vs Pre-Momentum)
# ─────────────────────────────────────────────────────────────────────
#
# Same BuyScore ranking, but candidates are first bucketed by
# classification_group(). For the Pre-Momentum LONG bucket we blend
# 0.6 × pre_momentum_score + 0.4 × BuyScore so forward-looking signal
# (Micro/Macro/Graph/Catalyst/QVR) carries more weight — the BuyScore
# alone overweights *current* composite, which by definition is lower
# for formation/recovery names.
PRE_MOM_BLEND_W = 0.6   # weight on pre_momentum_score (Pre-Mom LONG only)


def _score_for_group(scored: dict, row: dict, group: str, side: str) -> float:
    """Effective ranking score given the group/side."""
    if group == "pre_momentum" and side == "long":
        pm = float(row.get("pre_momentum_score") or 0)
        return PRE_MOM_BLEND_W * pm + (1.0 - PRE_MOM_BLEND_W) * scored["buyScore"]
    return float(scored["buyScore"])


_SOFT_FILTER_REASONS = {"Liq"}  # only exclude illiquid names from Pre-Mom LONG


def _passes_filter(scored: dict, raw_rejection: Optional[str], group: str, side: str) -> bool:
    """Per-group filter logic.

    Momentum LONG  → full hard_filter (mirror of ConvictionPicks).
    Pre-Mom LONG   → soft filter (Liq only). Recovery(PM)/Pullback(PM)/LowScore
                     are the WHOLE POINT of this bucket — keep them in.
    SHORT (both)   → no filter (bearish names are EXPECTED).
    """
    if side == "short":
        return True
    if group == "momentum":
        return not scored["hard_filter"]
    # Pre-Mom LONG: soft filter
    if not raw_rejection:
        return True
    s = str(raw_rejection)
    return not any(reason in s for reason in _SOFT_FILTER_REASONS)


def _split_picks(
    all_results: list[dict],
    side: str,
    top_n_stock: int,
    top_n_etf: int,
    bull_sectors: Optional[set[str]],
    bear_sectors: Optional[set[str]],
    consensus_map: Optional[dict[str, int]],
    min_adv_M: float,
    exclude_hard_filter: bool,
) -> dict:
    """Return per-group selection: {momentum: {stocks,etfs}, pre_momentum: {stocks,etfs}}."""
    buckets = {
        "momentum":     {"stocks": [], "etfs": []},
        "pre_momentum": {"stocks": [], "etfs": []},
    }
    _ = exclude_hard_filter  # kept for backward compat; per-group logic below supersedes

    for row in all_results:
        if not row.get("ticker"):
            continue
        adv_M = float(row.get("adv_M") or 0)
        if adv_M and adv_M < min_adv_M:
            continue
        cls_raw = row.get("classification", "") or ""
        group = classification_group(cls_raw, side=side)
        if group is None:
            continue
        s = compute_buyscore(row, bull_sectors, bear_sectors, consensus_map)
        if not _passes_filter(s, row.get("rejection"), group, side):
            continue
        s["asset"] = (row.get("asset_type") or row.get("asset") or "").lower()
        s["group"] = group
        s["rank_score"] = _score_for_group(s, row, group, side)
        s["pre_momentum_score"] = float(row.get("pre_momentum_score") or 0)
        if s["asset"] == "stock":
            buckets[group]["stocks"].append(s)
        elif s["asset"] == "etf":
            buckets[group]["etfs"].append(s)

    reverse = (side == "long")   # LONG: DESC (best first); SHORT: ASC (worst first)
    for g in buckets:
        buckets[g]["stocks"].sort(key=lambda x: x["rank_score"], reverse=reverse)
        buckets[g]["etfs"].sort(key=lambda x: x["rank_score"],   reverse=reverse)
        buckets[g]["pool_size_stock"] = len(buckets[g]["stocks"])
        buckets[g]["pool_size_etf"]   = len(buckets[g]["etfs"])
        buckets[g]["stocks"] = buckets[g]["stocks"][:top_n_stock]
        buckets[g]["etfs"]   = buckets[g]["etfs"][:top_n_etf]
    return buckets


def top_buy_picks_split(
    all_results: list[dict],
    top_n_stock: int = 5,
    top_n_etf: int = 5,
    bull_sectors: Optional[set[str]] = None,
    bear_sectors: Optional[set[str]] = None,
    consensus_map: Optional[dict[str, int]] = None,
    min_adv_M: float = 1.0,
) -> dict:
    """BUY (LONG) Top-N split by classification group.

    Returns: {momentum: {stocks,etfs}, pre_momentum: {stocks,etfs}}
    Excludes hard_filter (bearish names).
    """
    return _split_picks(
        all_results, "long", top_n_stock, top_n_etf,
        bull_sectors, bear_sectors, consensus_map, min_adv_M,
        exclude_hard_filter=True,
    )


def top_sell_picks_split(
    all_results: list[dict],
    top_n_stock: int = 5,
    top_n_etf: int = 5,
    bull_sectors: Optional[set[str]] = None,
    bear_sectors: Optional[set[str]] = None,
    consensus_map: Optional[dict[str, int]] = None,
    min_adv_M: float = 1.0,
) -> dict:
    """SELL (SHORT) Bottom-N split by classification group.

    Returns: {momentum: {stocks,etfs}, pre_momentum: {stocks,etfs}}
    Does NOT exclude hard_filter (bearish names are EXPECTED here).
    """
    return _split_picks(
        all_results, "short", top_n_stock, top_n_etf,
        bull_sectors, bear_sectors, consensus_map, min_adv_M,
        exclude_hard_filter=False,
    )
