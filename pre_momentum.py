"""
Pre-Momentum Detection Multi-Agent Framework.
==============================================
Identifies tickers showing structural conditions for momentum formation
BEFORE the actual breakout occurs.

Architecture:
  5 Specialist Agents + 1 Orchestrator

  Agent 1 — MicrostructureAgent (Pure Quant)
      Volatility compression, accumulation patterns, structural divergence,
      volume regime change, range contraction.

  Agent 2 — MacroRegimeAgent (Pure Quant)
      Sector rotation signals, cross-asset alignment, category breadth,
      relative improvement trajectories.

  Agent 3 — GraphRelationalAgent (Hybrid)
      Peer lead signals, theme breadth, leader-lagger gap,
      community momentum from graph data.

  Agent 4 — CatalystAgent (Quant Proxy)
      Momentum acceleration, strategy agreement, score trajectory,
      reversal risk checks.

  Agent 5 — QVRAgent (Fundamentals)  ★ Option B addition
      Quality (margin/ROE) + Value (PE/PEG/PB) + Revision (EPS revision
      momentum, analyst sentiment, target upside). Most leading dimension —
      analyst forecast changes precede price moves.

  Orchestrator — PreMomentumOrchestrator
      Filters universe, runs all agents, computes conviction & agreement,
      produces final ranked output.

Usage:
    from pre_momentum import run_pre_momentum
    import pickle

    with open(".scan_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    output = run_pre_momentum(cache)
"""

import os
import json
import math
from datetime import date
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price_discovery import (
    STOCK_THEMES, STOCK_THEMES_CONSOLIDATED, STOCK_UNIVERSE, GLOBAL_ETF_UNIVERSE,
    CATEGORY_BENCHMARK, STOCK_BENCHMARK,
)

# QVR Agent (5th — fundamentals)
try:
    from qvr_agent import QVRAgent
    _HAS_QVR = True
except Exception:
    _HAS_QVR = False
    QVRAgent = None  # type: ignore


###############################################################################
# UTILITIES
###############################################################################

def _sf(value, default: float = 0.0) -> float:
    """Safe float: convert to float, return default on failure or non-finite."""
    try:
        r = float(value)
        return r if math.isfinite(r) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a numeric value into [lo, hi]."""
    return max(lo, min(hi, _sf(value)))


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division guarded against zero / non-finite denominators."""
    n = _sf(numerator)
    d = _sf(denominator)
    if abs(d) < 1e-12:
        return default
    result = n / d
    return result if math.isfinite(result) else default


def _percentile_rank(value: float, values: List[float]) -> float:
    """Percentile rank of *value* within *values* (0-100)."""
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return 50.0
    below = sum(1 for v in clean if v < value)
    return (below / (len(clean) - 1)) * 100.0


def _get(d: dict, key: str, default: float = 0.0) -> float:
    """Safely retrieve a float from a result dict."""
    return _sf(d.get(key, default), default)


###############################################################################
# CLASSIFICATION SETS
###############################################################################

# Classifications indicating momentum is ALREADY established — exclude these.
_EXCLUDE_CLASSIFICATIONS = {
    "🟢 CONTINUATION",
    "🔵 FORMATION",
    "🟡 OVEREXTENDED",
    "🔴 CYCLE_PEAK",
    "⬇️ DOWNTREND",
    "🟣 COUNTER_RALLY",
    "🟤 EXHAUSTING",
}

# Classifications where pre-momentum detection is relevant.
_INCLUDE_CLASSIFICATIONS = {
    "🟠 NEUTRAL",
    "🟡 CONSOLIDATION",
    "🔵 RECOVERY",
    "🔶 PULLBACK",
    "⚠️ WEAKENING",
    "🟤 FADING",
}

# ── FICC-specific filters ──
# FICC categories
_FICC_CATEGORIES = {
    "FI_Short", "FI_Intermediate", "FI_Long", "FI_Credit",
    "FI_Inflation", "FI_International",
    "Commodities", "Real_Assets", "Currency_Vol", "Multi_Asset",
}

# Minimum ADV for FICC (lower bar than equity due to inherently lower volumes)
_FICC_MIN_ADV = 10_000_000  # $10M

# Duplicate underlying groups: keep only the highest-ADV ETF per group
_FICC_DUPLICATE_GROUPS = {
    "Gold":       ["GLD", "GLDM", "IAU", "SGOL", "AAAU", "BAR"],
    "Oil_WTI":    ["USO", "BNO"],
    "Commodity_Broad": ["DBC", "GSG", "PDBC", "COMT", "FTGC"],
}

# Macro factor mapping: category -> relevant ETF tickers to check alignment
_MACRO_FACTOR_MAP = {
    "STK_Technology": ["TLT", "IEF", "QQQ", "SMH"],
    "EQ_Technology": ["TLT", "IEF", "QQQ", "SMH"],
    "STK_CommServices": ["QQQ", "XLC"],
    "EQ_CommServices": ["QQQ", "XLC"],
    "STK_Healthcare": ["XLV", "IBB", "XBI"],
    "EQ_Healthcare": ["XLV", "IBB", "XBI"],
    "STK_Financials": ["XLF", "KRE", "TLT"],
    "EQ_Financials": ["XLF", "KRE", "TLT"],
    "STK_ConsDisc": ["XLY", "XRT", "SPY"],
    "EQ_ConsDisc": ["XLY", "XRT", "SPY"],
    "STK_ConsStaples": ["XLP", "SPY"],
    "EQ_ConsStaples": ["XLP", "SPY"],
    "STK_Industrials": ["XLI", "SPY"],
    "EQ_Industrials": ["XLI", "SPY"],
    "STK_Energy": ["XLE", "USO", "DBC"],
    "EQ_Energy": ["XLE", "USO", "DBC"],
    "STK_Materials": ["XLB", "DBC", "GLD"],
    "EQ_Materials": ["XLB", "DBC", "GLD"],
    "STK_Utilities": ["XLU", "TLT"],
    "EQ_Utilities": ["XLU", "TLT"],
    "STK_RealEstate": ["XLRE", "VNQ", "TLT"],
    "EQ_RealEstate": ["XLRE", "VNQ", "TLT"],
    "STK_Korea": ["069500.KS", "EWY"],
    "Korea_Equity": ["069500.KS", "EWY"],
    "Commodities": ["DBC", "GLD", "USO"],
    "Real_Assets": ["VNQ", "DBC"],
}

# Agent weight configuration (5 agents, sum = 1.0)
# QVR weighted highest among leading agents — analyst revisions precede price.
_AGENT_WEIGHTS = {
    "microstructure": 0.20,
    "macro_regime": 0.15,
    "graph_relational": 0.20,
    "catalyst": 0.20,
    "qvr": 0.25,
}

_AGENT_THRESHOLD = 50.0  # Score above which an agent "signals"
_AGENT_COUNT = len(_AGENT_WEIGHTS)


###############################################################################
# HELPER: Build lookup indices from results
###############################################################################

def _build_indices(results: List[dict]) -> dict:
    """
    Pre-compute lookup structures used by multiple agents.

    Returns a dict with:
        by_ticker     : {ticker: result_dict}
        by_category   : {category: [result_dicts]}
        by_theme      : {theme: [result_dicts]}
        by_consol_theme: {consolidated_theme: [result_dicts]}
        all_realized_vols: [float] (for percentile ranking)
        all_composites:    [float]
        all_oers:          [float]
        etf_by_ticker:     {ticker: result_dict} for ETF-only tickers
    """
    by_ticker: Dict[str, dict] = {}
    by_category: Dict[str, List[dict]] = defaultdict(list)
    by_theme: Dict[str, List[dict]] = defaultdict(list)
    by_consol_theme: Dict[str, List[dict]] = defaultdict(list)
    all_realized_vols: List[float] = []
    all_composites: List[float] = []
    all_oers: List[float] = []
    etf_by_ticker: Dict[str, dict] = {}

    # Collect all ETF tickers for cross-asset checks
    etf_tickers_set = set()
    for cat_data in GLOBAL_ETF_UNIVERSE.values():
        etf_tickers_set.update(cat_data.get("tickers", {}).keys())

    for r in results:
        tk = r.get("ticker", "")
        if not tk:
            continue
        by_ticker[tk] = r
        cat = r.get("category", "")
        if cat:
            by_category[cat].append(r)

        # Theme mapping
        theme = STOCK_THEMES.get(tk, "")
        if theme:
            by_theme[theme].append(r)
        consol = STOCK_THEMES_CONSOLIDATED.get(tk, "")
        if consol:
            by_consol_theme[consol].append(r)

        rv = _get(r, "realized_vol")
        if rv > 0:
            all_realized_vols.append(rv)
        all_composites.append(_get(r, "composite"))
        all_oers.append(_get(r, "oer"))

        if tk in etf_tickers_set:
            etf_by_ticker[tk] = r

    return {
        "by_ticker": by_ticker,
        "by_category": by_category,
        "by_theme": by_theme,
        "by_consol_theme": by_consol_theme,
        "all_realized_vols": all_realized_vols,
        "all_composites": all_composites,
        "all_oers": all_oers,
        "etf_by_ticker": etf_by_ticker,
    }


###############################################################################
# AGENT 1: MicrostructureAgent
###############################################################################

class MicrostructureAgent:
    """
    Detects structural changes in price/volume before momentum appears.

    Sub-signals (each 0-100):
        volatility_compression  — low vol + tight SMA distance + low OER
        accumulation_pattern    — TFS building while TCS is still low
        structural_divergence   — high structural_q vs. low composite
        volume_regime           — flow_long rising while classification neutral
        range_contraction       — RSI near 50 + price compressed
    """

    def __init__(self, indices: dict):
        self.indices = indices

    def score(self, r: dict) -> Tuple[float, Dict[str, float], str]:
        """Return (agent_score, signals_dict, summary_string)."""
        signals = {
            "volatility_compression": self._volatility_compression(r),
            "accumulation_pattern": self._accumulation_pattern(r),
            "structural_divergence": self._structural_divergence(r),
            "volume_regime": self._volume_regime(r),
            "range_contraction": self._range_contraction(r),
        }
        # Agent score = weighted average of sub-signals
        weights = {
            "volatility_compression": 0.25,
            "accumulation_pattern": 0.25,
            "structural_divergence": 0.20,
            "volume_regime": 0.15,
            "range_contraction": 0.15,
        }
        agent_score = sum(signals[k] * weights[k] for k in signals)
        agent_score = _clamp(agent_score)

        summary = self._summarize(signals)
        return agent_score, signals, summary

    # ── Sub-signals ──────────────────────────────────────────────────────

    def _volatility_compression(self, r: dict) -> float:
        """
        Low realized_vol + tight sma50_dist + low OER → coiled spring.
        Score = inverse of (abs(sma50_dist) + vol_percentile + oer).
        """
        sma50_dist = abs(_get(r, "sma50_dist"))
        realized_vol = _get(r, "realized_vol")
        oer = _get(r, "oer")

        vol_pctile = _percentile_rank(realized_vol, self.indices["all_realized_vols"])

        # Raw penalty: higher = worse (more extended / volatile / overextended)
        raw_penalty = (sma50_dist * 5.0) + (vol_pctile / 100.0 * 40.0) + (oer * 0.4)
        # Invert: 100 minus penalty, clamped
        return _clamp(100.0 - raw_penalty)

    def _accumulation_pattern(self, r: dict) -> float:
        """
        TCS is low (no trend yet) but TFS_short ticking up + wyckoff_long rising.
        Score = tfs_short * (1 - tcs/100) * (wyckoff_long / 100).
        """
        tfs_short = _get(r, "tfs_short")
        tcs = _get(r, "tcs")
        wyckoff_long = _get(r, "wyckoff_long", 50.0)

        # Guard: if tcs is near 100, the inverse factor → 0 (already in trend)
        tcs_inverse = 1.0 - (tcs / 100.0)
        wyckoff_factor = wyckoff_long / 100.0

        raw = tfs_short * tcs_inverse * wyckoff_factor
        # Scale: tfs_short max 100 × 1 × 1 = 100
        return _clamp(raw)

    def _structural_divergence(self, r: dict) -> float:
        """
        High structural_q but low composite = quality building before breakout.
        Score = structural_q - composite (positive = building).
        """
        structural_q = _get(r, "structural_q", 50.0)
        composite = _get(r, "composite")

        divergence = structural_q - composite
        # Map: divergence of +50 → 100, 0 → 50, -50 → 0
        return _clamp(divergence + 50.0)

    def _volume_regime(self, r: dict) -> float:
        """
        flow_long rising while price is flat (NEUTRAL/CONSOLIDATION).
        """
        classification = r.get("classification", "")
        flow_long = _get(r, "flow_long", 50.0)

        # Only relevant for neutral-ish classifications
        neutral_classes = {
            "🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔶 PULLBACK", "🔵 RECOVERY",
        }
        if classification not in neutral_classes:
            # Still give partial credit based on flow_long alone
            return _clamp(flow_long * 0.5)

        # flow_long is 0-100; higher = more institutional buying
        return _clamp(flow_long)

    def _range_contraction(self, r: dict) -> float:
        """
        RSI near 50 (neither overbought nor oversold) + price compressed.
        Score = 100 - abs(rsi - 50) * 2 when sma50_dist is tight.
        """
        rsi = _get(r, "rsi", 50.0)
        sma50_dist = abs(_get(r, "sma50_dist"))

        rsi_score = 100.0 - abs(rsi - 50.0) * 2.0

        # Tightness bonus: if sma50_dist < 3%, full credit; fade above 10%
        if sma50_dist <= 3.0:
            tightness = 1.0
        elif sma50_dist >= 10.0:
            tightness = 0.3
        else:
            tightness = 1.0 - 0.7 * ((sma50_dist - 3.0) / 7.0)

        return _clamp(rsi_score * tightness)

    # ── Summary ──────────────────────────────────────────────────────────

    @staticmethod
    def _summarize(signals: Dict[str, float]) -> str:
        """Build a human-readable summary of the microstructure signals."""
        parts = []
        if signals["volatility_compression"] >= 60:
            parts.append("strong volatility compression")
        elif signals["volatility_compression"] >= 40:
            parts.append("moderate volatility compression")

        if signals["accumulation_pattern"] >= 55:
            parts.append("emerging accumulation")
        if signals["structural_divergence"] >= 60:
            parts.append("quality divergence (structural_q > composite)")
        if signals["volume_regime"] >= 55:
            parts.append("institutional flow building")
        if signals["range_contraction"] >= 60:
            parts.append("tight range contraction")

        if not parts:
            return "No notable microstructure signals"
        return "; ".join(parts).capitalize()


###############################################################################
# AGENT 2: MacroRegimeAgent
###############################################################################

class MacroRegimeAgent:
    """
    Analyzes whether the macro environment supports this ticker's sector.

    Sub-signals (each 0-100):
        sector_rotation   — % of category members improving (score_1w > score_1m)
        cross_asset        — macro factor ETFs stable / improving
        category_breadth   — % of category with positive ret_5d
        relative_improvement — category avg (score_1w - score_1m)
    """

    def __init__(self, indices: dict):
        self.indices = indices
        # Phase 2C — Cross-sectional rotation indicators (cyclical/style/region 우위).
        # Macro tags(cyclical_tag/style_tilt/region)가 ticker dict에 주입된 경우에만 작동.
        self._rotation_ctx = self._compute_rotation_context()

    def _compute_rotation_context(self) -> dict:
        """전체 universe에서 cyclical vs defensive / growth vs value / region leadership 진단."""
        ctx = {
            "cyclical_dom": False, "defensive_dom": False,
            "growth_dom": False, "value_dom": False,
            "top_region": None, "bot_region": None,
        }
        try:
            all_results = self.indices.get("by_category", {})
            # 모든 ticker dict 평면화
            flat = []
            for members in all_results.values():
                flat.extend(members)
            if not flat:
                return ctx
            def _avg(group_key, group_val):
                vals = [m.get("composite", 0) for m in flat
                        if m.get(group_key) == group_val and m.get("composite") is not None]
                return sum(vals) / len(vals) if vals else 50.0
            cyc = _avg("cyclical_tag", "cyclical")
            dfn = _avg("cyclical_tag", "defensive")
            ctx["cyclical_dom"] = (cyc - dfn) > 3.0
            ctx["defensive_dom"] = (dfn - cyc) > 3.0
            grw = _avg("style_tilt", "growth")
            val = _avg("style_tilt", "value")
            ctx["growth_dom"] = (grw - val) > 3.0
            ctx["value_dom"] = (val - grw) > 3.0
            # Region leadership
            region_avg = {}
            for m in flat:
                rg = m.get("region")
                if rg:
                    region_avg.setdefault(rg, []).append(m.get("composite", 0) or 0)
            region_means = {k: sum(v) / len(v) for k, v in region_avg.items() if v}
            if region_means:
                ctx["top_region"] = max(region_means.items(), key=lambda kv: kv[1])[0]
                ctx["bot_region"] = min(region_means.items(), key=lambda kv: kv[1])[0]
        except Exception:
            pass
        return ctx

    def score(self, r: dict) -> Tuple[float, Dict[str, float], str]:
        signals = {
            "sector_rotation": self._sector_rotation(r),
            "cross_asset": self._cross_asset(r),
            "category_breadth": self._category_breadth(r),
            "relative_improvement": self._relative_improvement(r),
            # Phase 2C — macro regime alignment (Risk/Style/Region)
            "rotation_alignment": self._rotation_alignment(r),
            # Hybrid Phase D — parent ETF divergence signal (bottom-up)
            "etf_parent_signal": self._etf_parent_signal(r),
        }
        weights = {
            "sector_rotation": 0.20,
            "cross_asset": 0.10,
            "category_breadth": 0.20,
            "relative_improvement": 0.15,
            "rotation_alignment": 0.20,
            "etf_parent_signal": 0.15,
        }
        agent_score = sum(signals[k] * weights[k] for k in signals)
        agent_score = _clamp(agent_score)
        summary = self._summarize(signals)
        return agent_score, signals, summary

    # ── Hybrid Phase D: parent ETF signal ──────────────────────────────
    def _etf_parent_signal(self, r: dict) -> float:
        """Stock: ETF가 STEALTH_STRENGTH/HEALTHY_TREND이면 + boost (api.py 가 계산해 주입).
        ETF: own constituent_breadth_mom 사용.
        시그널 미주입 시 50 (neutral)."""
        v = r.get("parent_etf_signal")
        if v is None:
            return 50.0
        try:
            return float(v)
        except Exception:
            return 50.0

    # ── Phase 2C: rotation_alignment sub-signal ────────────────────────
    def _rotation_alignment(self, r: dict) -> float:
        """현 cross-sectional regime과 ticker의 macro tags 정렬도 (0-100).

        정렬 차원:
          1) Risk axis: cyclical_dom + ticker cyclical → 정렬 / def_dom + defensive → 정렬
          2) Style axis: growth_dom + ticker growth / value_dom + value
          3) Region: top_region 종목이면 +30, bot_region 종목이면 -10
        """
        ctx = self._rotation_ctx
        # Macro tag 미주입 시 중립
        if r.get("cyclical_tag") is None:
            return 50.0
        score = 50.0  # neutral baseline
        cyc_dom = ctx.get("cyclical_dom", False)
        def_dom = ctx.get("defensive_dom", False)
        grw_dom = ctx.get("growth_dom", False)
        val_dom = ctx.get("value_dom", False)
        # Risk alignment
        tag = r.get("cyclical_tag")
        if cyc_dom and tag == "cyclical":
            score += 15
        elif def_dom and tag == "defensive":
            score += 15
        elif (cyc_dom and tag == "defensive") or (def_dom and tag == "cyclical"):
            score -= 10
        # Style alignment
        st = r.get("style_tilt")
        if grw_dom and st == "growth":
            score += 15
        elif val_dom and st == "value":
            score += 15
        elif (grw_dom and st == "value") or (val_dom and st == "growth"):
            score -= 10
        # Region alignment
        rg = r.get("region")
        if rg and rg == ctx.get("top_region"):
            score += 15
        elif rg and rg == ctx.get("bot_region"):
            score -= 5
        return _clamp(score)

    # ── Sub-signals ──────────────────────────────────────────────────────

    def _sector_rotation(self, r: dict) -> float:
        """
        % of category members with score_1w > score_1m (momentum improving).
        Also counts % above sma50.
        """
        cat = r.get("category", "")
        members = self.indices["by_category"].get(cat, [])
        if len(members) < 2:
            return 50.0  # neutral when insufficient data

        improving = 0
        above_sma50 = 0
        for m in members:
            s1w = _get(m, "score_1w")
            s1m = _get(m, "score_1m")
            if s1w > s1m:
                improving += 1
            if m.get("above_sma50", False):
                above_sma50 += 1

        improving_pct = (improving / len(members)) * 100.0
        breadth_pct = (above_sma50 / len(members)) * 100.0

        # Blend: 60% improving momentum, 40% SMA50 breadth
        return _clamp(improving_pct * 0.6 + breadth_pct * 0.4)

    def _cross_asset(self, r: dict) -> float:
        """
        Check if macro factor ETFs for this category are stable/improving.
        """
        cat = r.get("category", "")
        factor_tickers = _MACRO_FACTOR_MAP.get(cat, [])
        if not factor_tickers:
            return 50.0  # neutral fallback

        etf_data = self.indices["etf_by_ticker"]
        scores = []
        for ftk in factor_tickers:
            etf = etf_data.get(ftk)
            if etf is None:
                continue
            # Positive ret_5d or score_1w > score_1m → supporting
            ret5 = _get(etf, "ret_5d")
            s1w = _get(etf, "score_1w")
            s1m = _get(etf, "score_1m")

            # Simple alignment score: ret_5d contribution + trajectory contribution
            ret_component = _clamp(50.0 + ret5 * 10.0)  # ±5% → 0-100 range
            traj_component = _clamp(50.0 + (s1w - s1m) * 2.0)
            scores.append(ret_component * 0.5 + traj_component * 0.5)

        if not scores:
            return 50.0
        return _clamp(sum(scores) / len(scores))

    def _category_breadth(self, r: dict) -> float:
        """
        Count of category members with positive ret_5d / total.
        """
        cat = r.get("category", "")
        members = self.indices["by_category"].get(cat, [])
        if len(members) < 2:
            return 50.0

        positive = sum(1 for m in members if _get(m, "ret_5d") > 0)
        pct = (positive / len(members)) * 100.0
        return _clamp(pct)

    def _relative_improvement(self, r: dict) -> float:
        """
        Category average (score_1w - score_1m). Positive = improving.
        """
        cat = r.get("category", "")
        members = self.indices["by_category"].get(cat, [])
        if len(members) < 2:
            return 50.0

        diffs = [_get(m, "score_1w") - _get(m, "score_1m") for m in members]
        avg_diff = sum(diffs) / len(diffs)

        # Map: +20 → 100, 0 → 50, -20 → 0
        return _clamp(50.0 + avg_diff * 2.5)

    # ── Summary ──────────────────────────────────────────────────────────

    @staticmethod
    def _summarize(signals: Dict[str, float]) -> str:
        parts = []
        if signals["sector_rotation"] >= 60:
            parts.append("sector rotation favorable")
        elif signals["sector_rotation"] <= 35:
            parts.append("sector rotation headwind")

        if signals["cross_asset"] >= 60:
            parts.append("macro factors aligned")
        elif signals["cross_asset"] <= 35:
            parts.append("macro factors adverse")

        if signals["category_breadth"] >= 60:
            parts.append("broad category participation")
        elif signals["category_breadth"] <= 35:
            parts.append("narrow category breadth")

        if signals["relative_improvement"] >= 60:
            parts.append("category trajectory improving")
        elif signals["relative_improvement"] <= 35:
            parts.append("category trajectory deteriorating")

        if not parts:
            return "Mixed macro signals"
        return "; ".join(parts).capitalize()


###############################################################################
# AGENT 3: GraphRelationalAgent
###############################################################################

class GraphRelationalAgent:
    """
    Uses graph data + results to detect momentum propagation effects.

    Sub-signals (each 0-100):
        peer_lead       — count of peers in same theme with CONTINUATION/FORMATION
        theme_breadth   — % of theme members with composite > 55
        leader_lagger_gap — catch-up potential (max peer composite - this composite)
        community_momentum — graph community bullishness
    """

    def __init__(self, indices: dict, graph_data: dict):
        self.indices = indices
        self.graph = graph_data or {}

    def score(self, r: dict) -> Tuple[float, Dict[str, float], str]:
        signals = {
            "peer_lead": self._peer_lead(r),
            "theme_breadth": self._theme_breadth(r),
            "leader_lagger_gap": self._leader_lagger_gap(r),
            "community_momentum": self._community_momentum(r),
        }
        weights = {
            "peer_lead": 0.30,
            "theme_breadth": 0.25,
            "leader_lagger_gap": 0.25,
            "community_momentum": 0.20,
        }
        agent_score = sum(signals[k] * weights[k] for k in signals)
        agent_score = _clamp(agent_score)
        summary = self._summarize(signals, r)
        return agent_score, signals, summary

    # ── Sub-signals ──────────────────────────────────────────────────────

    def _peer_lead(self, r: dict) -> float:
        """
        Count peers in same theme with CONTINUATION or FORMATION classification.
        If peers are already moving, this ticker may follow.
        """
        tk = r.get("ticker", "")
        theme = STOCK_THEMES.get(tk, "")
        consol_theme = STOCK_THEMES_CONSOLIDATED.get(tk, "")

        # Use consolidated theme for broader peer group
        peers = self.indices["by_consol_theme"].get(consol_theme, [])
        if not peers:
            peers = self.indices["by_theme"].get(theme, [])
        if len(peers) <= 1:
            return 30.0  # low confidence when no peers

        momentum_classes = {"🟢 CONTINUATION", "🔵 FORMATION"}
        leading_count = 0
        for p in peers:
            if p.get("ticker") == tk:
                continue
            if p.get("classification", "") in momentum_classes:
                leading_count += 1

        peer_count = len(peers) - 1  # exclude self
        if peer_count <= 0:
            return 30.0

        # Ratio of leading peers: if 50%+ are leading, very strong signal
        ratio = leading_count / peer_count
        return _clamp(ratio * 120.0)  # 0.83 ratio → 100

    def _theme_breadth(self, r: dict) -> float:
        """
        % of theme members with composite > 55.
        """
        tk = r.get("ticker", "")
        consol_theme = STOCK_THEMES_CONSOLIDATED.get(tk, "")
        peers = self.indices["by_consol_theme"].get(consol_theme, [])
        if not peers:
            theme = STOCK_THEMES.get(tk, "")
            peers = self.indices["by_theme"].get(theme, [])
        if len(peers) < 2:
            return 40.0

        above = sum(1 for p in peers if _get(p, "composite") > 55)
        pct = (above / len(peers)) * 100.0
        return _clamp(pct)

    def _leader_lagger_gap(self, r: dict) -> float:
        """
        If theme leaders have high composite but this ticker doesn't → catch-up potential.
        Score = max(peer composites) - this ticker's composite, capped at 60.
        """
        tk = r.get("ticker", "")
        consol_theme = STOCK_THEMES_CONSOLIDATED.get(tk, "")
        peers = self.indices["by_consol_theme"].get(consol_theme, [])
        if not peers:
            theme = STOCK_THEMES.get(tk, "")
            peers = self.indices["by_theme"].get(theme, [])

        peer_composites = [
            _get(p, "composite") for p in peers if p.get("ticker") != tk
        ]
        if not peer_composites:
            return 30.0

        max_peer = max(peer_composites)
        my_composite = _get(r, "composite")
        gap = max_peer - my_composite

        if gap <= 0:
            # This ticker is already the leader — no catch-up needed
            return 20.0

        # Gap of 30+ points → 100, gap of 5 → ~30
        # Scale: min(gap, 60) / 60 * 100
        return _clamp(min(gap, 60.0) / 60.0 * 100.0)

    def _community_momentum(self, r: dict) -> float:
        """
        From graph community_stats, assess community bullishness.
        """
        tk = r.get("ticker", "")
        community_stats = self.graph.get("community_stats", {})
        communities = self.graph.get("communities", {})

        # Find this ticker's community
        comm_id = communities.get(tk)
        if comm_id is None:
            return 45.0  # neutral fallback

        stats = community_stats.get(comm_id)
        if stats is None:
            return 45.0

        # Use avg_composite and eligible_pct as proxy for bullishness
        avg_composite = _sf(stats.get("avg_composite", 50.0))
        eligible_pct = _sf(stats.get("eligible_pct", 0.0))

        # Also check the classification distribution for bullish classes
        cls_dist = stats.get("classification_dist", {})
        total_members = max(stats.get("n", 1), 1)
        bullish_count = 0
        for cls_name, count in cls_dist.items():
            if cls_name in {"🟢 CONTINUATION", "🔵 FORMATION", "🔵 RECOVERY"}:
                bullish_count += count

        bullish_pct = (bullish_count / total_members) * 100.0

        # Blend: avg_composite weight, eligible_pct, bullish_pct
        score = avg_composite * 0.4 + eligible_pct * 0.2 + bullish_pct * 0.4
        return _clamp(score)

    # ── Summary ──────────────────────────────────────────────────────────

    def _summarize(self, signals: Dict[str, float], r: dict) -> str:
        parts = []
        tk = r.get("ticker", "")

        if signals["peer_lead"] >= 55:
            consol_theme = STOCK_THEMES_CONSOLIDATED.get(tk, "")
            peers = self.indices["by_consol_theme"].get(consol_theme, [])
            leaders = [
                p.get("ticker")
                for p in peers
                if p.get("ticker") != tk
                and p.get("classification") in {"🟢 CONTINUATION", "🔵 FORMATION"}
            ]
            if leaders:
                parts.append(f"peers leading: {', '.join(leaders[:3])}")

        if signals["theme_breadth"] >= 55:
            parts.append("theme breadth expanding")
        elif signals["theme_breadth"] <= 30:
            parts.append("theme breadth narrow")

        if signals["leader_lagger_gap"] >= 60:
            parts.append("catch-up potential vs. theme leaders")

        if signals["community_momentum"] >= 55:
            parts.append("community bullish")

        if not parts:
            return "Limited graph-based signals"
        return "; ".join(parts).capitalize()


###############################################################################
# AGENT 4: CatalystAgent
###############################################################################

class CatalystAgent:
    """
    Uses quantitative proxy signals in lieu of news/options data.

    Sub-signals (each 0-100):
        momentum_acceleration — rss_short > rss_long (building faster short-term)
        strategy_agreement    — long_count / total strategies
        score_trajectory      — accelerating improvement across 1w/1m/3m
        reversal_risk_check   — reversal_pctile safety check
    """

    def __init__(self, indices: dict):
        self.indices = indices

    def score(self, r: dict) -> Tuple[float, Dict[str, float], str]:
        signals = {
            "momentum_acceleration": self._momentum_acceleration(r),
            "strategy_agreement": self._strategy_agreement(r),
            "score_trajectory": self._score_trajectory(r),
            "reversal_risk_check": self._reversal_risk_check(r),
        }
        weights = {
            "momentum_acceleration": 0.30,
            "strategy_agreement": 0.25,
            "score_trajectory": 0.30,
            "reversal_risk_check": 0.15,
        }
        agent_score = sum(signals[k] * weights[k] for k in signals)
        agent_score = _clamp(agent_score)
        summary = self._summarize(signals)
        return agent_score, signals, summary

    # ── Sub-signals ──────────────────────────────────────────────────────

    def _momentum_acceleration(self, r: dict) -> float:
        """
        rss_short > rss_long → short-term momentum building faster.
        Score = max(0, rss_short - rss_long), scaled.
        """
        rss_short = _get(r, "rss_short")
        rss_long = _get(r, "rss_long")

        diff = rss_short - rss_long
        if diff <= 0:
            # Even if negative, partial credit if rss_short is decent
            return _clamp(rss_short * 0.3)

        # diff range: 0-80ish realistically; scale so diff=40 → ~80
        return _clamp(diff * 2.0 + 10.0)

    def _strategy_agreement(self, r: dict) -> float:
        """
        How many hedge strategies are turning long?
        long_count / total_strategies, normalized to 0-100.
        """
        long_count = _get(r, "long_count")
        short_count = _get(r, "short_count")
        total = long_count + short_count
        if total < 1:
            # Fallback: use net_signal or conviction
            net_sig = _get(r, "net_signal")
            return _clamp(50.0 + net_sig * 5.0)

        ratio = long_count / total
        return _clamp(ratio * 100.0)

    def _score_trajectory(self, r: dict) -> float:
        """
        score_1w > score_1m > score_3m → accelerating improvement.
        Points for each positive step.
        """
        s1w = _get(r, "score_1w")
        s1m = _get(r, "score_1m")
        s3m = _get(r, "score_3m")

        points = 0.0

        # Step 1: 3m → 1m improvement
        if s1m > s3m:
            points += 30.0
            # Bonus for magnitude
            points += min((s1m - s3m) * 0.5, 10.0)

        # Step 2: 1m → 1w improvement
        if s1w > s1m:
            points += 35.0
            points += min((s1w - s1m) * 0.5, 10.0)

        # Bonus: both steps positive (full acceleration)
        if s1w > s1m > s3m:
            points += 15.0

        return _clamp(points)

    def _reversal_risk_check(self, r: dict) -> float:
        """
        reversal_pctile < 70 → safe (not at risk of mean reversion).
        This is a SAFETY signal: higher score = safer.
        """
        rev_pctile = _get(r, "reversal_pctile", 50.0)

        if rev_pctile <= 30:
            return 90.0  # very safe — deeply undervalued / low reversal risk
        elif rev_pctile <= 50:
            return 75.0
        elif rev_pctile <= 70:
            return 55.0
        elif rev_pctile <= 85:
            return 30.0  # caution
        else:
            return 10.0  # high reversal risk

    # ── Summary ──────────────────────────────────────────────────────────

    @staticmethod
    def _summarize(signals: Dict[str, float]) -> str:
        parts = []
        if signals["momentum_acceleration"] >= 55:
            parts.append("short-term momentum accelerating")
        if signals["strategy_agreement"] >= 55:
            parts.append("strategy consensus turning long")
        elif signals["strategy_agreement"] <= 30:
            parts.append("strategies still bearish/mixed")
        if signals["score_trajectory"] >= 55:
            parts.append("score trajectory improving 3m→1m→1w")
        if signals["reversal_risk_check"] >= 60:
            parts.append("low reversal risk")
        elif signals["reversal_risk_check"] <= 30:
            parts.append("elevated reversal risk")

        if not parts:
            return "No strong catalyst proxy signals"
        return "; ".join(parts).capitalize()


###############################################################################
# ORCHESTRATOR
###############################################################################

class PreMomentumOrchestrator:
    """
    Combines all 4 agents to produce the final pre-momentum analysis.

    Steps:
        1. Filter target universe (exclude already-in-momentum tickers)
        2. Run all 4 agents on each candidate
        3. Compute agreement ratio, conviction level, final score
        4. Sort & return results with per-agent breakdowns
    """

    def __init__(self, cache: dict):
        self.results: List[dict] = cache.get("results", [])
        self.graph_data: dict = cache.get("graph", {})
        self.history: dict = cache.get("history", {})
        self.fundamentals: dict = cache.get("fundamentals", {})  # injected by api.py
        self.indices: dict = _build_indices(self.results)

    def run(self) -> dict:
        """Execute the full pre-momentum detection pipeline."""

        # Step 1: Filter candidates
        candidates, excluded = self._filter_universe()

        # Step 2: Initialize agents
        micro_agent = MicrostructureAgent(self.indices)
        macro_agent = MacroRegimeAgent(self.indices)
        graph_agent = GraphRelationalAgent(self.indices, self.graph_data)
        catalyst_agent = CatalystAgent(self.indices)

        # 5th agent — QVR (fundamentals). If module/cache missing, agent is None
        # and we fall back gracefully.
        qvr_agent = None
        if _HAS_QVR and QVRAgent is not None:
            try:
                qvr_agent = QVRAgent(self.indices,
                                     fundamentals_cache=self.fundamentals or None)
            except Exception:
                qvr_agent = None

        # Step 3: Score each candidate
        scored: List[dict] = []
        for r in candidates:
            try:
                entry = self._score_candidate(
                    r, micro_agent, macro_agent, graph_agent, catalyst_agent,
                    qvr_agent,
                )
                scored.append(entry)
            except Exception:
                # Silently skip candidates that cause errors; production code
                # could log this for debugging.
                continue

        # Step 4: Sort by pre_momentum_score descending
        scored.sort(key=lambda x: x["pre_momentum_score"], reverse=True)

        # Step 5: Build summary
        summary = self._build_summary(scored, len(self.results), len(candidates))

        # Step 6: Build methodology
        methodology = self._build_methodology()

        return {
            "candidates": scored,
            "summary": summary,
            "methodology": methodology,
        }

    # ── Filtering ────────────────────────────────────────────────────────

    def _filter_universe(self) -> Tuple[List[dict], List[dict]]:
        """
        Separate tickers into candidates (pre-momentum targets) and excluded
        (already in momentum or unsuitable).

        Pre-momentum focuses on tickers BEFORE momentum confirmation:
          - Classification must be in _INCLUDE_CLASSIFICATIONS
          - Must NOT already be eligible (eligible = momentum confirmed)
          - FICC assets: additional filters for duplicates, ADV, suitability
        """
        candidates = []
        excluded = []

        # Pre-filter: determine which FICC tickers to keep (dedup by group)
        ficc_keep = self._ficc_dedup_filter()

        for r in self.results:
            cls = r.get("classification", "")
            eligible = bool(r.get("eligible", False))
            tk = r.get("ticker", "")
            cat = r.get("category", "")

            if cls not in _INCLUDE_CLASSIFICATIONS or eligible:
                excluded.append(r)
                continue

            # FICC-specific filters
            if cat in _FICC_CATEGORIES:
                adv = _sf(r.get("adv_usd", 0))
                if adv < _FICC_MIN_ADV:
                    excluded.append(r)
                    continue
                if tk not in ficc_keep:
                    excluded.append(r)
                    continue

            candidates.append(r)

        return candidates, excluded

    def _ficc_dedup_filter(self) -> set:
        """
        For FICC duplicate groups (e.g. 6 gold ETFs), keep only the
        highest-ADV ticker per group. All non-grouped FICC tickers pass.
        """
        # Build set of all grouped tickers
        grouped_tickers: Dict[str, str] = {}  # ticker → group_name
        for group, tickers in _FICC_DUPLICATE_GROUPS.items():
            for tk in tickers:
                grouped_tickers[tk] = group

        # For each group, find the highest-ADV ticker from results
        group_best: Dict[str, Tuple[str, float]] = {}  # group → (best_ticker, adv)
        for r in self.results:
            tk = r.get("ticker", "")
            if tk not in grouped_tickers:
                continue
            group = grouped_tickers[tk]
            adv = _sf(r.get("adv_usd", 0))
            if group not in group_best or adv > group_best[group][1]:
                group_best[group] = (tk, adv)

        # Build keep set: best per group + all non-grouped FICC tickers
        keep = set()
        for r in self.results:
            tk = r.get("ticker", "")
            cat = r.get("category", "")
            if cat not in _FICC_CATEGORIES:
                continue
            if tk in grouped_tickers:
                best_tk = group_best.get(grouped_tickers[tk], ("", 0))[0]
                if tk == best_tk:
                    keep.add(tk)
            else:
                keep.add(tk)

        return keep

    # ── Scoring ──────────────────────────────────────────────────────────

    def _score_candidate(
        self,
        r: dict,
        micro: MicrostructureAgent,
        macro: MacroRegimeAgent,
        graph: GraphRelationalAgent,
        catalyst: CatalystAgent,
        qvr: Optional["QVRAgent"] = None,
    ) -> dict:
        """Run all agents on a single candidate, compute final output."""

        # Run each agent
        micro_score, micro_signals, micro_summary = micro.score(r)
        macro_score, macro_signals, macro_summary = macro.score(r)
        graph_score, graph_signals, graph_summary = graph.score(r)
        catalyst_score, catalyst_signals, catalyst_summary = catalyst.score(r)

        # 5th agent — QVR (fundamentals). Returns 50 (neutral) for ETFs / missing.
        if qvr is not None:
            qvr_score, qvr_signals, qvr_summary = qvr.score(r)
        else:
            qvr_score, qvr_signals, qvr_summary = 50.0, {
                "quality": 50.0, "value": 50.0, "revision": 50.0,
                "net_30d": 0, "ratio_30d": 50, "n_analysts": 0,
            }, "QVR agent unavailable"

        # Final weighted score
        pre_momentum_score = (
            micro_score * _AGENT_WEIGHTS["microstructure"]
            + macro_score * _AGENT_WEIGHTS["macro_regime"]
            + graph_score * _AGENT_WEIGHTS["graph_relational"]
            + catalyst_score * _AGENT_WEIGHTS["catalyst"]
            + qvr_score * _AGENT_WEIGHTS["qvr"]
        )
        pre_momentum_score = round(_clamp(pre_momentum_score), 1)

        # Agreement ratio: count agents scoring above threshold (5 agents)
        agent_scores = {
            "microstructure": micro_score,
            "macro_regime": macro_score,
            "graph_relational": graph_score,
            "catalyst": catalyst_score,
            "qvr": qvr_score,
        }
        signaling_count = sum(
            1 for s in agent_scores.values() if s > _AGENT_THRESHOLD
        )
        agreement_ratio = round(signaling_count / float(_AGENT_COUNT), 2)

        # Expected timeline (uses pre_momentum_score + agreement_ratio thresholds;
        # conviction labels removed — agreement_ratio carries the same info numerically)
        timeline = self._estimate_timeline(pre_momentum_score, agreement_ratio)

        # Key catalysts & risk factors
        key_catalysts = self._extract_catalysts(
            r, micro_signals, macro_signals, graph_signals, catalyst_signals
        )
        risk_factors = self._extract_risks(
            r, micro_signals, macro_signals, graph_signals, catalyst_signals
        )

        # Theme
        tk = r.get("ticker", "")
        theme = STOCK_THEMES.get(tk, "")
        if not theme:
            # For ETFs, use the category as a proxy
            theme = r.get("category", "")

        return {
            "ticker": tk,
            "name": r.get("name", ""),
            "category": r.get("category", ""),
            "theme": theme,
            "current_classification": r.get("classification", ""),
            "current_composite": round(_get(r, "composite"), 1),
            "pre_momentum_score": pre_momentum_score,
            "agreement_ratio": agreement_ratio,
            "agents": {
                "microstructure": {
                    "score": round(micro_score, 1),
                    "signals": {k: round(v, 1) for k, v in micro_signals.items()},
                    "summary": micro_summary,
                },
                "macro_regime": {
                    "score": round(macro_score, 1),
                    "signals": {k: round(v, 1) for k, v in macro_signals.items()},
                    "summary": macro_summary,
                },
                "graph_relational": {
                    "score": round(graph_score, 1),
                    "signals": {k: round(v, 1) for k, v in graph_signals.items()},
                    "summary": graph_summary,
                },
                "catalyst": {
                    "score": round(catalyst_score, 1),
                    "signals": {k: round(v, 1) for k, v in catalyst_signals.items()},
                    "summary": catalyst_summary,
                },
                "qvr": {
                    "score": round(qvr_score, 1),
                    "signals": {k: (round(v, 1) if isinstance(v, float) else v)
                                for k, v in qvr_signals.items()},
                    "summary": qvr_summary,
                },
            },
            "expected_timeline": timeline,
            "key_catalysts": key_catalysts,
            "risk_factors": risk_factors,
            # Returns (multi-horizon)
            "ret_1d":   round(_get(r, "ret_1d"), 2),
            "ret_5d":   round(_get(r, "ret_5d"), 2),
            "ret_21d":  round(_get(r, "ret_21d"), 2),
            "ret_63d":  round(_get(r, "ret_63d"), 2),
            "ret_126d": round(_get(r, "ret_126d"), 2),
            "ret_252d": round(_get(r, "ret_252d"), 2),
            "ret_3y_ann": round(_get(r, "ret_3y_ann"), 2) if r.get("ret_3y_ann") is not None else None,
            "ret_5y_ann": round(_get(r, "ret_5y_ann"), 2) if r.get("ret_5y_ann") is not None else None,
            "vol_3y_ann": round(_get(r, "vol_3y_ann"), 2) if r.get("vol_3y_ann") is not None else None,
        }

    # ── Timeline estimation ──────────────────────────────────────────────

    @staticmethod
    def _estimate_timeline(score: float, agreement_ratio: float) -> str:
        # agreement_ratio is 0..1 (5 agents → 0.2 increments).
        # Buckets mirror the prior conviction tiers: ≥0.6 ≈ HIGH, ≥0.4 ≈ MEDIUM,
        # ≥0.2 ≈ LOW, 0 ≈ NONE.
        if agreement_ratio >= 0.6 and score >= 70:
            return "1-2 weeks"
        elif agreement_ratio >= 0.6 or (agreement_ratio >= 0.4 and score >= 65):
            return "2-3 weeks"
        elif agreement_ratio >= 0.4:
            return "3-4 weeks"
        elif agreement_ratio >= 0.2 and score >= 50:
            return "4-6 weeks"
        else:
            return "6+ weeks (speculative)"

    # ── Catalyst & risk extraction ───────────────────────────────────────

    def _extract_catalysts(
        self,
        r: dict,
        micro_sig: dict,
        macro_sig: dict,
        graph_sig: dict,
        cat_sig: dict,
    ) -> List[str]:
        """Extract the top catalysts (positive signals) as human-readable strings."""
        catalysts = []

        # Microstructure catalysts
        if micro_sig.get("volatility_compression", 0) >= 65:
            catalysts.append("Volatility compression — coiled spring setup")
        if micro_sig.get("accumulation_pattern", 0) >= 55:
            catalysts.append("Early accumulation pattern (TFS building, low TCS)")
        if micro_sig.get("structural_divergence", 0) >= 60:
            catalysts.append("Structural quality exceeds current price score")

        # Macro catalysts
        if macro_sig.get("sector_rotation", 0) >= 60:
            cat_name = r.get("category", "").replace("STK_", "").replace("EQ_", "")
            catalysts.append(f"{cat_name} sector rotation momentum improving")
        if macro_sig.get("category_breadth", 0) >= 65:
            catalysts.append("Category breadth expanding (majority positive ret_5d)")

        # Graph catalysts
        if graph_sig.get("peer_lead", 0) >= 55:
            tk = r.get("ticker", "")
            consol_theme = STOCK_THEMES_CONSOLIDATED.get(tk, "")
            peers = self.indices["by_consol_theme"].get(consol_theme, [])
            leaders = [
                p.get("ticker")
                for p in peers
                if p.get("ticker") != tk
                and p.get("classification") in {"🟢 CONTINUATION", "🔵 FORMATION"}
            ]
            if leaders:
                catalysts.append(
                    f"Peer(s) {', '.join(leaders[:3])} already in CONTINUATION/FORMATION"
                )
        if graph_sig.get("leader_lagger_gap", 0) >= 60:
            catalysts.append("Significant catch-up potential vs. theme leaders")

        # Catalyst agent
        if cat_sig.get("score_trajectory", 0) >= 55:
            catalysts.append("Score trajectory accelerating (3m→1m→1w)")
        if cat_sig.get("strategy_agreement", 0) >= 60:
            catalysts.append("Multiple hedge strategies turning long")

        return catalysts[:6]  # cap at 6 most relevant

    def _extract_risks(
        self,
        r: dict,
        micro_sig: dict,
        macro_sig: dict,
        graph_sig: dict,
        cat_sig: dict,
    ) -> List[str]:
        """Extract risk factors as human-readable strings."""
        risks = []

        rsi = _get(r, "rsi", 50.0)
        if rsi < 45:
            risks.append(f"RSI at {rsi:.0f} — below neutral")

        if not r.get("golden_cross", False):
            risks.append("No golden cross (SMA50 < SMA200)")

        if not r.get("above_sma50", False):
            risks.append("Price below SMA50")

        if cat_sig.get("reversal_risk_check", 100) <= 35:
            risks.append(
                f"Elevated reversal risk (pctile={_get(r, 'reversal_pctile', 0):.0f})"
            )

        if macro_sig.get("cross_asset", 100) <= 35:
            risks.append("Macro factor headwinds for this category")

        if macro_sig.get("category_breadth", 100) <= 35:
            risks.append("Category breadth narrow — few peers participating")

        if graph_sig.get("peer_lead", 100) <= 25:
            risks.append("No peers showing momentum — early / isolated signal")

        ret_5d = _get(r, "ret_5d")
        if ret_5d < -3.0:
            risks.append(f"Recent weakness: ret_5d = {ret_5d:.1f}%")

        return risks[:5]  # cap at 5

    # ── Summary statistics ───────────────────────────────────────────────

    @staticmethod
    def _build_summary(
        scored: List[dict], total_universe: int, candidates_analyzed: int
    ) -> dict:
        """Produce aggregate summary statistics."""
        # Agreement distribution: bucket agreement_ratio (0..1) into 5 bins
        # (since there are 5 agents → ratio steps of 0.2).
        agreement_dist = Counter(
            round(c["agreement_ratio"] * 5) for c in scored
        )

        # Strong / moderate / weak agreement buckets — replaces conviction tiers
        n_strong = sum(1 for c in scored if c["agreement_ratio"] >= 0.6)
        n_moderate = sum(1 for c in scored if 0.4 <= c["agreement_ratio"] < 0.6)
        n_weak = sum(1 for c in scored if 0 < c["agreement_ratio"] < 0.4)
        n_none = sum(1 for c in scored if c["agreement_ratio"] == 0)

        # Top sectors
        sector_scores: Dict[str, List[float]] = defaultdict(list)
        for c in scored:
            cat = c.get("category", "Other")
            sector_scores[cat].append(c["pre_momentum_score"])

        top_sectors = sorted(
            [
                {
                    "sector": cat.replace("STK_", "").replace("EQ_", ""),
                    "count": len(scores),
                    "avg_score": round(sum(scores) / len(scores), 1),
                }
                for cat, scores in sector_scores.items()
                if scores
            ],
            key=lambda x: -x["avg_score"],
        )[:10]

        return {
            "total_universe": total_universe,
            "candidates_analyzed": candidates_analyzed,
            "agreement_strong": n_strong,        # ratio ≥ 0.6 (≥3 of 5 agents)
            "agreement_moderate": n_moderate,    # 0.4 ≤ ratio < 0.6 (2 agents)
            "agreement_weak": n_weak,            # 0 < ratio < 0.4 (1 agent)
            "agreement_none": n_none,            # ratio == 0
            "top_sectors": top_sectors,
            "agent_agreement_distribution": dict(sorted(agreement_dist.items())),
        }

    # ── Methodology ──────────────────────────────────────────────────────

    @staticmethod
    def _build_methodology() -> dict:
        return {
            "description": (
                "Pre-Momentum Detection identifies tickers showing structural "
                "conditions for momentum formation before the actual breakout occurs. "
                "Four independent specialist agents analyze microstructure, macro regime, "
                "graph-relational patterns, and catalyst proxy signals. Conviction is "
                "determined by agent agreement."
            ),
            "agents": [
                {
                    "name": "Microstructure",
                    "weight": _AGENT_WEIGHTS["microstructure"],
                    "type": "Quant",
                    "description": (
                        "Volatility compression, accumulation patterns, "
                        "structural quality divergence, volume regime change, "
                        "range contraction"
                    ),
                },
                {
                    "name": "Macro Regime",
                    "weight": _AGENT_WEIGHTS["macro_regime"],
                    "type": "Quant",
                    "description": (
                        "Sector rotation signals, cross-asset macro factor alignment, "
                        "category breadth (% positive), relative improvement trajectory"
                    ),
                },
                {
                    "name": "Graph Relational",
                    "weight": _AGENT_WEIGHTS["graph_relational"],
                    "type": "Hybrid",
                    "description": (
                        "Peer momentum lead-lag, theme breadth propagation, "
                        "leader-lagger catch-up gap, community momentum from "
                        "graph community detection"
                    ),
                },
                {
                    "name": "Catalyst Proxy",
                    "weight": _AGENT_WEIGHTS["catalyst"],
                    "type": "Quant",
                    "description": (
                        "Momentum acceleration (RSS short vs. long), hedge strategy "
                        "agreement, score trajectory (3m→1m→1w), reversal risk check"
                    ),
                },
            ],
            "agreement_thresholds": {
                "strong":   "agreement_ratio ≥ 0.6 (≥3 of 5 agents above 50)",
                "moderate": "0.4 ≤ ratio < 0.6 (2 of 5 agents)",
                "weak":     "0 < ratio < 0.4 (1 of 5 agents)",
                "none":     "ratio == 0",
            },
        }


###############################################################################
# PUBLIC API
###############################################################################

def run_pre_momentum(cache: dict) -> dict:
    """
    Main entry point. Takes the loaded .scan_cache.pkl dict and returns
    the full pre-momentum analysis.

    Parameters
    ----------
    cache : dict
        The loaded cache dictionary containing at minimum:
            - "results": List[dict] — per-ticker scan results
        Optionally:
            - "graph": dict — graph engine data (community_stats, communities, etc.)
            - "history": dict — 7-day historical snapshots

    Returns
    -------
    dict
        {
            "candidates": [...],   # sorted by pre_momentum_score desc
            "summary": {...},      # aggregate statistics
            "methodology": {...},  # documentation of the approach
        }
    """
    orchestrator = PreMomentumOrchestrator(cache)
    output = orchestrator.run()

    candidates = output["candidates"]
    all_results = cache.get("results", [])
    ve_obs = cache.get("ve_observations", [])
    # ve_observations may also be nested in a 've_stats' or similar structure
    if not ve_obs:
        ve_obs = (cache.get("ve_stats", {}) or {}).get("observations", [])

    # ── pm_age: robust retroactive calculation using ve_observations ──
    _enrich_pm_age_robust(candidates, ve_obs)

    # ── Backtest conversion: use 1-month historical data ──
    output["conversion"] = _backtest_conversion(all_results)

    # Partition by asset type
    etf_cands = [c for c in candidates if not c.get("category", "").startswith("STK_")]
    stk_cands = [c for c in candidates if c.get("category", "").startswith("STK_")]
    output["candidates_etf"] = etf_cands
    output["candidates_stock"] = stk_cands

    return output


# ── Pre-Momentum Age Tracking ──────────────────────────────────────────────

_PM_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".pm_history.json"
)


def _enrich_pm_age_robust(candidates: List[dict], ve_obs: List[dict]) -> None:
    """
    Compute pm_age robustly using bi-weekly ve_observations.

    Logic:
      - Walk backwards through ve_observations for each candidate
      - Count consecutive observations in pre-momentum classifications
      - Allow MAX_GAPS non-PM observations (noise tolerance)
      - Stop at momentum-confirmed classification (breakout = age reset)
      - Require MIN_PERSISTENCE consecutive obs for non-zero age (filter single-point noise)
      - age = (today - oldest_confirmed_pm_date).days

    Parameters:
      MAX_GAPS = 1         — allow one non-PM obs in the chain (daily noise tolerance)
      MIN_PERSISTENCE = 2  — require at least 2 PM obs before granting non-zero age
      MAX_AGE_DAYS = 90    — cap age at 90 days (3 months) for practical display
    """
    MAX_GAPS = 1
    MIN_PERSISTENCE = 2
    MAX_AGE_DAYS = 90

    today = date.today()

    # Group observations by ticker
    obs_by_ticker: Dict[str, List[dict]] = defaultdict(list)
    for o in ve_obs:
        if not isinstance(o, dict):
            continue
        tk = o.get("ticker", "")
        if tk:
            obs_by_ticker[tk].append(o)

    # Sort each ticker's observations by eval_date desc (newest first)
    for tk in obs_by_ticker:
        obs_by_ticker[tk].sort(key=lambda x: x.get("eval_date", ""), reverse=True)

    for c in candidates:
        tk = c.get("ticker", "")
        obs = obs_by_ticker.get(tk, [])

        if not obs:
            c["pm_age"] = 0
            continue

        # Walk backwards through observations
        # Today's state already confirmed PM (included as 1 confirmed implicitly)
        confirmed_dates: List[str] = [today.isoformat()]
        gaps_used = 0

        for o in obs:
            cls = o.get("classification", "")
            eval_dt = o.get("eval_date", "")
            if not eval_dt:
                continue

            if cls in _INCLUDE_CLASSIFICATIONS:
                confirmed_dates.append(eval_dt)
            elif cls in _MOMENTUM_CONFIRMED:
                # Breakout occurred — stop (age counts only since breakout)
                break
            else:
                # Bearish or other — allow gap tolerance
                if gaps_used < MAX_GAPS:
                    gaps_used += 1
                    continue
                else:
                    break

        # Apply minimum persistence filter (today + at least 1 past obs)
        if len(confirmed_dates) < MIN_PERSISTENCE:
            c["pm_age"] = 0
            continue

        # Age = days from today to oldest confirmed PM observation (capped)
        try:
            oldest = min(confirmed_dates)
            age_days = (today - date.fromisoformat(oldest)).days
            c["pm_age"] = min(max(0, age_days), MAX_AGE_DAYS)
        except (ValueError, TypeError):
            c["pm_age"] = 0


# ── Momentum Age (for confirmed momentum tickers) ─────────────────────────

# Tier A: confirmed momentum — count as age
_MOMENTUM_CONFIRMED_AGE = {
    "🟢 CONTINUATION",
    "🔵 FORMATION",
    "🟡 OVEREXTENDED",
}

# Tier B: healthy corrections — tolerated as gaps (don't break chain)
_MOMENTUM_GAP_TOLERANT = {
    "🔵 RECOVERY",
    "🟡 CONSOLIDATION",
    "🔶 PULLBACK",
}

# Tier C: hard breaks — immediately stop age counting
_MOMENTUM_BREAK = {
    "⬇️ DOWNTREND",
    "🔴 CYCLE_PEAK",
    "🟣 COUNTER_RALLY",
    "🟤 EXHAUSTING",
    "🟤 FADING",
    "⚠️ WEAKENING",
}


def compute_momentum_ages(results: List[dict], ve_obs: List[dict]) -> Dict[str, int]:
    """
    Compute momentum age (days in confirmed uptrend) for each eligible ticker.

    3-Tier classification with differentiated gap tolerance:
      - Tier A (CONTINUATION/FORMATION/OVEREXTENDED): confirmed, counts as age
      - Tier B (RECOVERY/CONSOLIDATION/PULLBACK): healthy correction — allow up to
        MAX_B_GAPS (2) gaps
      - Tier C (DOWNTREND/CYCLE_PEAK/etc.): bearish — allow up to MAX_C_GAPS (1)
        gaps to absorb market-wide event-driven noise
      - Total gaps (B + C) cap at MAX_TOTAL_GAPS (3) to prevent over-generosity

    Additional logic:
      - 2+ consecutive Tier C observations → hard break (genuine trend reversal)
      - MIN_PERSISTENCE (2) confirmed obs required for non-zero age
      - Age capped at MAX_AGE_DAYS (180)

    Returns:
      {ticker: age_in_days} for all tickers with current momentum classification
    """
    MAX_B_GAPS = 2
    MAX_C_GAPS = 1
    MAX_TOTAL_GAPS = 3
    MIN_PERSISTENCE = 2
    MAX_AGE_DAYS = 180

    today = date.today()

    # Group observations by ticker, sort desc by date
    obs_by_ticker: Dict[str, List[dict]] = defaultdict(list)
    for o in ve_obs:
        if not isinstance(o, dict):
            continue
        tk = o.get("ticker", "")
        if tk:
            obs_by_ticker[tk].append(o)
    for tk in obs_by_ticker:
        obs_by_ticker[tk].sort(key=lambda x: x.get("eval_date", ""), reverse=True)

    ages: Dict[str, int] = {}

    for r in results:
        tk = r.get("ticker", "")
        current_cls = r.get("classification", "")

        if current_cls not in _MOMENTUM_CONFIRMED_AGE:
            ages[tk] = 0
            continue

        obs = obs_by_ticker.get(tk, [])
        if not obs:
            ages[tk] = 0
            continue

        # Walk backwards
        # Today's state already confirmed Tier A (included as 1 confirmed implicitly)
        confirmed_dates: List[str] = [today.isoformat()]
        b_gaps_used = 0
        c_gaps_used = 0
        prev_was_c = False

        for o in obs:
            cls = o.get("classification", "")
            eval_dt = o.get("eval_date", "")
            if not eval_dt:
                continue

            if cls in _MOMENTUM_CONFIRMED_AGE:
                confirmed_dates.append(eval_dt)
                prev_was_c = False
            elif cls in _MOMENTUM_GAP_TOLERANT:
                if b_gaps_used >= MAX_B_GAPS or (b_gaps_used + c_gaps_used) >= MAX_TOTAL_GAPS:
                    break
                b_gaps_used += 1
                prev_was_c = False
            elif cls in _MOMENTUM_BREAK:
                # 2+ consecutive Tier C → genuine reversal, hard break
                if prev_was_c:
                    break
                if c_gaps_used >= MAX_C_GAPS or (b_gaps_used + c_gaps_used) >= MAX_TOTAL_GAPS:
                    break
                c_gaps_used += 1
                prev_was_c = True
            else:
                # Unknown — treat as B gap
                if b_gaps_used >= MAX_B_GAPS or (b_gaps_used + c_gaps_used) >= MAX_TOTAL_GAPS:
                    break
                b_gaps_used += 1
                prev_was_c = False

        if len(confirmed_dates) < MIN_PERSISTENCE:
            ages[tk] = 0
            continue

        try:
            oldest = min(confirmed_dates)
            age_days = (today - date.fromisoformat(oldest)).days
            ages[tk] = min(max(0, age_days), MAX_AGE_DAYS)
        except (ValueError, TypeError):
            ages[tk] = 0

    return ages


# ── Backtest Conversion (1-month retrospective) ──────────────────────────

# Momentum-confirmed classifications
_MOMENTUM_CONFIRMED = {"🟢 CONTINUATION", "🔵 FORMATION"}
_MOMENTUM_FAILED = {"⬇️ DOWNTREND", "🔴 CYCLE_PEAK", "🟣 COUNTER_RALLY"}


def _backtest_conversion(all_results: List[dict]) -> dict:
    """
    Retrospective conversion analysis using 1-month historical data.

    For each ticker, compares its state 1 month ago (score_1m, eligible_1m)
    to its current state to determine if pre-momentum conditions led to
    actual momentum formation.

    A ticker was a "pre-momentum candidate 1M ago" if:
      - eligible_1m == False  (was NOT in confirmed momentum)
      - score_1m >= 25        (had some structural buildup)
      - score_1m < 55         (below momentum threshold)

    Current outcome:
      - Graduated: now eligible or CONTINUATION/FORMATION
      - Failed: now DOWNTREND/CYCLE_PEAK/COUNTER_RALLY or composite < 25
      - In Progress: still building, not yet confirmed
    """
    graduated = []
    failed = []
    in_progress = []

    for r in all_results:
        score_1m = _sf(r.get("score_1m", 0))
        eligible_1m = bool(r.get("eligible_1m", False))

        # Was this ticker a pre-momentum candidate 1M ago?
        if eligible_1m or score_1m < 25 or score_1m >= 55:
            continue

        tk = r.get("ticker", "")
        name = r.get("name", "")
        category = r.get("category", "")
        current_cls = r.get("classification", "")
        current_composite = _sf(r.get("composite", 0))
        current_eligible = bool(r.get("eligible", False))
        score_change = round(current_composite - score_1m, 1)

        entry = {
            "ticker": tk,
            "name": name,
            "category": category,
            "score_1m_ago": round(score_1m, 1),
            "current_composite": round(current_composite, 1),
            "score_change": score_change,
            "current_class": current_cls,
            "current_eligible": current_eligible,
        }

        if current_eligible or current_cls in _MOMENTUM_CONFIRMED:
            graduated.append(entry)
        elif current_cls in _MOMENTUM_FAILED or current_composite < 25:
            failed.append(entry)
        else:
            in_progress.append(entry)

    # Sort: graduated by score_change desc, failed by score_change asc
    graduated.sort(key=lambda x: -x["score_change"])
    failed.sort(key=lambda x: x["score_change"])
    in_progress.sort(key=lambda x: -x["current_composite"])

    total_decided = len(graduated) + len(failed)
    hit_rate = (len(graduated) / total_decided * 100) if total_decided > 0 else 0.0
    avg_improvement = 0.0
    if graduated:
        avg_improvement = sum(g["score_change"] for g in graduated) / len(graduated)

    return {
        "graduated": graduated,
        "failed": failed,
        "in_progress": in_progress,
        "stats": {
            "total_pm_candidates_1m": len(graduated) + len(failed) + len(in_progress),
            "total_graduated": len(graduated),
            "total_failed": len(failed),
            "total_in_progress": len(in_progress),
            "hit_rate": round(hit_rate, 1),
            "avg_score_improvement": round(avg_improvement, 1),
        },
    }


###############################################################################
# CLI convenience
###############################################################################

if __name__ == "__main__":
    import pickle

    cache_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".scan_cache.pkl"
    )
    if not os.path.exists(cache_path):
        print(f"Cache file not found: {cache_path}")
        print("Run price_discovery.py first to generate the scan cache.")
        sys.exit(1)

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    print("=" * 80)
    print("  Pre-Momentum Detection Engine")
    print("  Identifying structural conditions for momentum formation")
    print("=" * 80)

    output = run_pre_momentum(cache)

    summary = output["summary"]
    print(f"\n  Universe: {summary['total_universe']} tickers")
    print(f"  Candidates analyzed: {summary['candidates_analyzed']}")
    print(f"  Strong agreement   (ratio ≥ 0.6): {summary['agreement_strong']}")
    print(f"  Moderate agreement (0.4 ≤ < 0.6): {summary['agreement_moderate']}")
    print(f"  Weak agreement     (0  < < 0.4): {summary['agreement_weak']}")
    print(f"  No agreement       (ratio == 0): {summary['agreement_none']}")

    print(f"\n  Agent Agreement Distribution: {summary['agent_agreement_distribution']}")

    print("\n  Top Sectors:")
    for sec in summary["top_sectors"][:5]:
        print(f"    {sec['sector']:25s}  count={sec['count']:3d}  avg_score={sec['avg_score']:.1f}")

    print("\n" + "=" * 80)
    print("  TOP PRE-MOMENTUM CANDIDATES")
    print("=" * 80)

    for c in output["candidates"][:20]:
        print(
            f"\n  {c['ticker']:8s} | {c['name'][:25]:25s} | "
            f"{c['current_classification']:22s} | "
            f"Composite={c['current_composite']:5.1f} | "
            f"PreMom={c['pre_momentum_score']:5.1f} | "
            f"Agree={c['agreement_ratio']:.2f}"
        )
        agents = c["agents"]
        print(
            f"           Micro={agents['microstructure']['score']:5.1f}  "
            f"Macro={agents['macro_regime']['score']:5.1f}  "
            f"Graph={agents['graph_relational']['score']:5.1f}  "
            f"Catal={agents['catalyst']['score']:5.1f}"
        )
        if c["key_catalysts"]:
            for cat_str in c["key_catalysts"][:2]:
                print(f"           + {cat_str}")
        if c["risk_factors"]:
            for risk_str in c["risk_factors"][:2]:
                print(f"           - {risk_str}")
        print(f"           Timeline: {c['expected_timeline']}")

    print("\n" + "=" * 80)
    print("  Analysis complete.")
    print("=" * 80)
