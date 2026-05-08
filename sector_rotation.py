"""
sector_rotation.py — US Sector Rotation Strategy (Phase 1)

Filters the universe to the 11 SPDR GICS sector ETFs, ranks them
*within the 11* (separate from the universe-wide RSS), and produces:

  • per-sector tier (OVERWEIGHT / NEUTRAL+ / CATCH-UP / NEUTRAL- / UNDERWEIGHT)
  • decision tag (BUY / HOLD / TRIM / HEDGE / EXIT / WATCH / CATCH-UP)
  • sector breadth (% of constituent stocks that are eligible / bullish)
  • dispersion (max - min composite among 11 — higher = more rotation alpha)
  • leaders / laggards (top vs bottom by Composite)

Reads data from the loaded scan cache + per-ticker df (already enriched
with Sector / SubTheme by api.py:_load_cache).

Output is JSON-friendly (no pandas/numpy types — caller wraps with
api._clean_dict).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from datetime import datetime


# ── 11 SPDR sector ETFs and their canonical Sector label ──
# (Sector label matches SUBTHEME_TO_SECTOR taxonomy in api.py)
SPDR_SECTOR_ETFS: Dict[str, str] = {
    "XLK":  "Technology",
    "XLC":  "Communication Services",
    "XLV":  "Healthcare",
    "XLF":  "Financials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLI":  "Industrials",
    "XLE":  "Energy",
    "XLB":  "Materials",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
}


# ── Bullish classification set (must match Eligibility Gate convention) ──
_BULLISH_CLASSIFICATIONS = {
    "🟢 CONTINUATION", "🔵 FORMATION", "🔵 RECOVERY",
    "🟡 OVEREXTENDED", "🟦 LAGGING_CATCHUP",
}


# ──────────────────────────────────────────────────────────────────────
# Macro regime overlay (Phase 2)
# ──────────────────────────────────────────────────────────────────────

# Sector group classification — used for top-down regime detection.
SECTOR_GROUPS: Dict[str, List[str]] = {
    "cyclical":  ["XLY", "XLB", "XLI", "XLF"],   # Aggressive cyclicals
    "growth":    ["XLK", "XLC"],                  # Tech / Comm Services
    "defensive": ["XLP", "XLU", "XLV", "XLRE"],   # Staples / Utilities / Healthcare / RealEstate
    "commodity": ["XLE"],                         # Energy
}
TICKER_TO_GROUP: Dict[str, str] = {
    tk: grp for grp, lst in SECTOR_GROUPS.items() for tk in lst
}

# Regime → expected sector leadership (0-100 fit score).
# Based on classical business-cycle sector rotation theory:
#   • Early cycle (recovery)  : ConsDisc, Materials, Industrials, Financials lead
#   • Mid expansion (growth)  : Tech, Comm Services lead
#   • Late cycle (overheat)   : Energy, Materials lead; Tech weakens
#   • Recession / Risk-off    : Defensive (Staples, Utilities, Healthcare) lead
#   • Mixed / Transitional    : No clear leadership (all moderate)
REGIME_FIT_MATRIX: Dict[str, Dict[str, int]] = {
    "RISK_ON_EARLY_CYCLE": {
        "XLY": 90, "XLB": 88, "XLI": 88, "XLF": 80,
        "XLK": 60, "XLC": 55,
        "XLE": 45,
        "XLP": 30, "XLU": 25, "XLV": 35, "XLRE": 40,
    },
    "TECH_GROWTH_LED": {
        "XLK": 95, "XLC": 90,
        "XLY": 70, "XLF": 60, "XLI": 55, "XLB": 45,
        "XLE": 35,
        "XLP": 35, "XLU": 30, "XLV": 55, "XLRE": 45,
    },
    "LATE_CYCLE": {
        "XLE": 90, "XLB": 80,
        "XLI": 65, "XLF": 60,
        "XLK": 40, "XLC": 40, "XLY": 35,
        "XLP": 55, "XLU": 60, "XLV": 65, "XLRE": 40,
    },
    "DEFENSIVE_RISK_OFF": {
        "XLP": 90, "XLU": 90, "XLV": 85, "XLRE": 70,
        "XLF": 30, "XLY": 20, "XLI": 30, "XLB": 25,
        "XLK": 35, "XLC": 35, "XLE": 45,
    },
    "MIXED_TRANSITIONAL": {
        "XLK": 55, "XLC": 55, "XLF": 55, "XLV": 60,
        "XLY": 50, "XLI": 50, "XLB": 50, "XLE": 50,
        "XLP": 55, "XLU": 50, "XLRE": 50,
    },
}

REGIME_LABELS: Dict[str, str] = {
    "RISK_ON_EARLY_CYCLE":  "Risk-on / Early Cycle",
    "TECH_GROWTH_LED":      "Tech & Growth Led",
    "LATE_CYCLE":           "Late Cycle / Inflation",
    "DEFENSIVE_RISK_OFF":   "Defensive / Risk-off",
    "MIXED_TRANSITIONAL":   "Mixed / Transitional",
}

REGIME_DESCRIPTIONS: Dict[str, str] = {
    "RISK_ON_EARLY_CYCLE":
        "Cyclical sectors (ConsDisc / Materials / Industrials / Financials) leading. "
        "Typical of recovery / early expansion. Defensives lag.",
    "TECH_GROWTH_LED":
        "Tech + Comm Services dominate. Mid-expansion growth phase. "
        "Long-duration assets favored.",
    "LATE_CYCLE":
        "Energy + Materials leading; Tech weakening. Inflationary / late cycle. "
        "Watch for rotation INTO defensives.",
    "DEFENSIVE_RISK_OFF":
        "Staples / Utilities / Healthcare / Real Estate leading. Recession or risk-off. "
        "Cyclicals underperform.",
    "MIXED_TRANSITIONAL":
        "No clear group leadership. Regime change in progress or low-conviction market. "
        "Use within-sector signals + breadth.",
}


def _detect_regime(sector_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect macro regime from group-level composite leadership.

    Returns dict with: regime, label, confidence ('HIGH'/'MEDIUM'/'LOW'),
    confidence_pct, group_averages, leading_group, second_group, gap.
    """
    valid = [s for s in sector_rows if not s.get("missing")]
    if len(valid) < 8:
        return {
            "regime": "MIXED_TRANSITIONAL",
            "label": REGIME_LABELS["MIXED_TRANSITIONAL"],
            "description": REGIME_DESCRIPTIONS["MIXED_TRANSITIONAL"],
            "confidence": "LOW",
            "confidence_pct": 0,
            "group_averages": {},
            "leading_group": None,
            "second_group": None,
            "gap": 0.0,
        }

    # Compute average composite per group
    by_ticker = {s["ticker"]: s for s in valid}
    group_avgs: Dict[str, float] = {}
    for grp, tickers in SECTOR_GROUPS.items():
        comps = [by_ticker[tk]["composite"] for tk in tickers if tk in by_ticker]
        if comps:
            group_avgs[grp] = round(sum(comps) / len(comps), 1)

    # Sort groups
    sorted_grps = sorted(group_avgs.items(), key=lambda x: -x[1])
    if not sorted_grps:
        return {
            "regime": "MIXED_TRANSITIONAL",
            "label": REGIME_LABELS["MIXED_TRANSITIONAL"],
            "description": REGIME_DESCRIPTIONS["MIXED_TRANSITIONAL"],
            "confidence": "LOW",
            "confidence_pct": 0,
            "group_averages": {},
            "leading_group": None,
            "second_group": None,
            "gap": 0.0,
        }

    top_grp, top_avg = sorted_grps[0]
    second_grp, second_avg = sorted_grps[1] if len(sorted_grps) > 1 else (None, 0)
    gap = top_avg - second_avg
    defensive_avg = group_avgs.get("defensive", 0)
    cyclical_avg = group_avgs.get("cyclical", 0)
    growth_avg = group_avgs.get("growth", 0)
    commodity_avg = group_avgs.get("commodity", 0)

    # Regime classification rules
    if top_grp == "cyclical" and cyclical_avg > defensive_avg + 8:
        regime = "RISK_ON_EARLY_CYCLE"
    elif top_grp == "growth" and growth_avg > defensive_avg + 5:
        regime = "TECH_GROWTH_LED"
    elif top_grp == "commodity" and commodity_avg > 55 and cyclical_avg > defensive_avg:
        regime = "LATE_CYCLE"
    elif top_grp == "defensive" and defensive_avg > cyclical_avg + 5:
        regime = "DEFENSIVE_RISK_OFF"
    elif gap < 5:
        regime = "MIXED_TRANSITIONAL"
    else:
        # Fallback: use top group's natural mapping
        regime = {
            "cyclical": "RISK_ON_EARLY_CYCLE",
            "growth":   "TECH_GROWTH_LED",
            "commodity": "LATE_CYCLE",
            "defensive": "DEFENSIVE_RISK_OFF",
        }.get(top_grp, "MIXED_TRANSITIONAL")

    # Confidence: based on the gap between leading group and second + absolute level
    if gap >= 15 and top_avg >= 55:
        confidence = "HIGH"
        confidence_pct = min(100, int(50 + gap * 2))
    elif gap >= 8:
        confidence = "MEDIUM"
        confidence_pct = int(40 + gap * 2)
    else:
        confidence = "LOW"
        confidence_pct = max(20, int(gap * 4))

    return {
        "regime": regime,
        "label": REGIME_LABELS[regime],
        "description": REGIME_DESCRIPTIONS[regime],
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "group_averages": group_avgs,
        "leading_group": top_grp,
        "second_group": second_grp,
        "gap": round(gap, 1),
    }


def _regime_fit(ticker: str, regime: str) -> int:
    """Return per-sector fit score (0-100) for the current regime."""
    return REGIME_FIT_MATRIX.get(regime, {}).get(ticker, 50)


# ──────────────────────────────────────────────────────────────────────
# Tier + Decision logic
# ──────────────────────────────────────────────────────────────────────

def _classify_tier(comp: float, classification: str, oer: float, urs: float) -> str:
    """Assign a sector to a rotation tier based on Composite + classification + OER + URS.

    OVERWEIGHT  : strong leader (high comp + bullish, not overextended)
    NEUTRAL+    : healthy but not exceptional
    CATCH-UP    : LAGGING_CATCHUP override or high URS (catch-up potential)
    NEUTRAL-    : weak / borderline
    UNDERWEIGHT : downtrend or severely overextended
    """
    cls = classification or ""

    # Underweight first — bearish signals trump everything else
    if "DOWNTREND" in cls or "CYCLE_PEAK" in cls or "FADING" in cls or "WEAKENING" in cls:
        return "UNDERWEIGHT"
    if "OVEREXTENDED" in cls and oer >= 75:
        return "UNDERWEIGHT"

    # Catch-up — sector is technically lagging but has strong catch-up signal
    if "LAGGING_CATCHUP" in cls:
        return "CATCH-UP"
    if comp < 55 and urs >= 70:
        return "CATCH-UP"

    # Overweight — confirmed leadership
    if comp >= 70 and ("CONTINUATION" in cls or "FORMATION" in cls or "RECOVERY" in cls):
        return "OVEREXTENDED" if oer >= 70 else "OVERWEIGHT"
    if comp >= 70 and "OVEREXTENDED" in cls:
        return "OVERWEIGHT"   # still leading but watch OER

    # Neutral tiers
    if comp >= 55 and cls in _BULLISH_CLASSIFICATIONS:
        return "NEUTRAL+"
    if comp < 40:
        return "UNDERWEIGHT"
    return "NEUTRAL-"


def _decide_action(comp: float, classification: str, oer: float, tier: str
                    ) -> Dict[str, Any]:
    """Per-sector trade recommendation, mapping to rank for sorting.

    Decision driven by tier (Composite), classification, and OER only.
    Hedge-strategy net_signal NOT used — it doesn't drive sector rotation
    selection in any backtest mode, so we keep the decision logic strictly
    self-contained to the 4-axis scoring model.
    """
    cls = classification or ""

    # Exit / Hedge first
    if "DOWNTREND" in cls or "CYCLE_PEAK" in cls:
        return {"action": "EXIT",
                "rationale": f"{cls.split(' ')[-1]} — rotate out",
                "rank": 10}
    if "OVEREXTENDED" in cls and oer >= 75:
        return {"action": "HEDGE",
                "rationale": f"OVEREXTENDED + OER {oer:.0f} — hedge or trim",
                "rank": 9}
    if "OVEREXTENDED" in cls and oer >= 60:
        return {"action": "TRIM",
                "rationale": f"OVEREXTENDED — partial exit (OER {oer:.0f})",
                "rank": 8}

    # Overweight tier
    if tier == "OVERWEIGHT":
        return {"action": "BUY",
                "rationale": f"Top sector — overweight (Comp {comp:.0f})",
                "rank": 2}

    # Catch-up tier
    if tier == "CATCH-UP":
        return {"action": "CATCH-UP",
                "rationale": "Lagging sector with catch-up potential",
                "rank": 4}

    # Bearish / weakening
    if "FADING" in cls or "WEAKENING" in cls:
        return {"action": "AVOID",
                "rationale": f"{cls.split(' ')[-1]} — weakening trend",
                "rank": 8}

    # Neutral tiers
    if tier == "NEUTRAL+":
        return {"action": "HOLD",
                "rationale": f"Healthy sector (Comp {comp:.0f})",
                "rank": 5}
    if tier == "NEUTRAL-":
        return {"action": "WATCH",
                "rationale": f"Borderline sector (Comp {comp:.0f}) — monitor",
                "rank": 6}
    if tier == "UNDERWEIGHT":
        return {"action": "AVOID",
                "rationale": f"Weak sector (Comp {comp:.0f}) — underweight",
                "rank": 7}

    return {"action": "WATCH", "rationale": "—", "rank": 6}


# ──────────────────────────────────────────────────────────────────────
# Breadth (per-sector constituent stock analysis)
# ──────────────────────────────────────────────────────────────────────

def _compute_sector_breadth(sector_label: str, df) -> Dict[str, Any]:
    """Aggregate constituent stocks of a sector into breadth metrics.

    Uses the df that already has Sector + asset_type + classification + composite + eligible.
    """
    if df is None or df.empty:
        return {"n_constituents": 0}

    # Subset: stocks (not ETFs) in this sector
    sub = df[(df["sector"] == sector_label) & (df["asset_type"] == "Stock")]
    n = len(sub)
    if n == 0:
        return {"n_constituents": 0}

    n_eligible = int((sub["eligible"] == True).sum())
    avg_comp = float(sub["composite"].mean()) if "composite" in sub.columns else 0.0
    bullish_mask = sub["classification"].isin(_BULLISH_CLASSIFICATIONS)
    n_bullish = int(bullish_mask.sum())

    return {
        "n_constituents": int(n),
        "n_eligible": n_eligible,
        "pct_eligible": round(100.0 * n_eligible / n, 1) if n else 0.0,
        "avg_composite": round(avg_comp, 1),
        "n_bullish_class": n_bullish,
        "pct_bullish": round(100.0 * n_bullish / n, 1) if n else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# Within-11 percentile ranking (independent of universe-wide RSS)
# ──────────────────────────────────────────────────────────────────────

def _rank_within(values: List[float]) -> List[int]:
    """Return rank (1=highest) for each value in the list. Ties broken by order."""
    sorted_idx = sorted(range(len(values)), key=lambda i: -values[i])
    rank = [0] * len(values)
    for r, i in enumerate(sorted_idx, 1):
        rank[i] = r
    return rank


def _pctile_within(values: List[float]) -> List[float]:
    """Cross-sectional percentile within the list (0-100)."""
    if not values:
        return []
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    out = []
    for v in values:
        # Find position of v in sorted (count of items < v)
        pos = sum(1 for s in sorted_vals if s < v)
        out.append(round(100.0 * pos / max(n - 1, 1), 1))
    return out


# ──────────────────────────────────────────────────────────────────────
# Pairs trade suggestions
# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def compute_sector_rotation(results: List[dict],
                             df=None,
                             scan_time: Optional[str] = None) -> Dict[str, Any]:
    """Build the US sector rotation snapshot.

    Args:
        results: list of per-ticker scan result dicts (from .scan_cache.pkl)
        df: pandas DataFrame from api.STATE['df'] (with sector / eligible columns)
        scan_time: ISO timestamp string for the snapshot

    Returns:
        JSON-ready dict with keys: as_of, sectors, summary, methodology
    """
    # Index results by ticker for fast lookup
    by_tk = {r.get("ticker", ""): r for r in (results or [])}

    # 1. Extract the 11 sector ETFs
    sector_rows: List[Dict[str, Any]] = []
    for tk, sector_label in SPDR_SECTOR_ETFS.items():
        r = by_tk.get(tk)
        if r is None:
            # Sector ETF missing from universe — record placeholder
            sector_rows.append({
                "ticker": tk,
                "sector": sector_label,
                "missing": True,
            })
            continue

        composite = float(r.get("composite", 0))
        tcs = float(r.get("tcs", 0))
        tfs = float(r.get("tfs", 0))
        rss = float(r.get("rss", 0))
        urs = float(r.get("urs", 0))
        oer = float(r.get("oer", 0))
        classification = r.get("classification", "")

        tier = _classify_tier(composite, classification, oer, urs)
        decision = _decide_action(composite, classification, oer, tier)
        breadth = _compute_sector_breadth(sector_label, df)

        sector_rows.append({
            "ticker": tk,
            "sector": sector_label,
            "name": r.get("name", tk),
            "composite": round(composite, 1),
            "tcs": round(tcs, 0),
            "tfs": round(tfs, 0),
            "rss": round(rss, 0),
            "urs": round(urs, 0),
            "oer": round(oer, 0),
            "classification": classification,
            "eligible": bool(r.get("eligible", False)),
            "tier": tier,
            "decision": decision["action"],
            "decision_rationale": decision["rationale"],
            "decision_rank": decision["rank"],
            "breadth": breadth,
            # Multi-horizon returns for chart context
            "ret_1d":  round(float(r.get("ret_1d",  0)), 2),
            "ret_5d":  round(float(r.get("ret_5d",  0)), 2),
            "ret_21d": round(float(r.get("ret_21d", 0)), 2),
            "ret_63d": round(float(r.get("ret_63d", 0)), 2),
            "missing": False,
        })

    # 2. Compute within-11 ranking + percentile (only on non-missing entries)
    valid_rows = [s for s in sector_rows if not s.get("missing")]
    if valid_rows:
        comps = [s["composite"] for s in valid_rows]
        ranks = _rank_within(comps)
        pctiles = _pctile_within(comps)
        for s, r, p in zip(valid_rows, ranks, pctiles):
            s["within11_rank"] = r
            s["within11_pctile"] = p

    # 2b. Regime detection (Phase 2)
    regime_info = _detect_regime(sector_rows)
    current_regime = regime_info["regime"]

    # 2c. Per-sector regime_fit + group label + combined score
    for s in sector_rows:
        if s.get("missing"):
            continue
        tk = s["ticker"]
        s["regime_fit"] = _regime_fit(tk, current_regime)
        s["group"] = TICKER_TO_GROUP.get(tk, "other")
        # Combined: equal weight composite + regime_fit (both 0-100)
        s["composite_x_regime"] = round((s["composite"] + s["regime_fit"]) / 2, 1)
        # Regime alignment tag
        if s["regime_fit"] >= 70:
            s["regime_alignment"] = "ALIGNED"
        elif s["regime_fit"] >= 50:
            s["regime_alignment"] = "NEUTRAL"
        else:
            s["regime_alignment"] = "CONTRARY"

    # 3. Sort by composite descending (leaders first)
    sector_rows.sort(key=lambda s: (-s.get("composite", -1), s.get("ticker", "")))

    # 4. Summary metrics
    valid_comps = [s["composite"] for s in sector_rows if not s.get("missing")]
    median_comp = sorted(valid_comps)[len(valid_comps) // 2] if valid_comps else 0.0
    dispersion = (max(valid_comps) - min(valid_comps)) if valid_comps else 0.0
    n_overweight = sum(1 for s in sector_rows if s.get("tier") == "OVERWEIGHT")
    n_underweight = sum(1 for s in sector_rows if s.get("tier") == "UNDERWEIGHT")
    n_catchup = sum(1 for s in sector_rows if s.get("tier") == "CATCH-UP")
    leaders = [s["ticker"] for s in sector_rows[:3] if not s.get("missing")]
    laggards = [s["ticker"] for s in sector_rows[-3:] if not s.get("missing")]

    summary = {
        "n_overweight": n_overweight,
        "n_underweight": n_underweight,
        "n_catchup": n_catchup,
        "median_composite": round(median_comp, 1),
        "dispersion": round(dispersion, 1),
        "leaders": leaders,
        "laggards": laggards,
        "alpha_signal": "HIGH" if dispersion >= 30 else "MODERATE" if dispersion >= 15 else "LOW",
    }

    methodology = {
        "tier_thresholds": {
            "OVERWEIGHT":  "Composite ≥ 70 + bullish classification (CONTINUATION/FORMATION/RECOVERY)",
            "NEUTRAL+":    "Composite 55-70 + bullish classification",
            "CATCH-UP":    "LAGGING_CATCHUP override OR (Composite < 55 AND URS ≥ 70)",
            "NEUTRAL-":    "Composite 40-55",
            "UNDERWEIGHT": "Composite < 40 OR DOWNTREND/CYCLE_PEAK/FADING/WEAKENING OR (OVEREXTENDED + OER ≥ 75)",
        },
        "rebalance": "Snapshot only (Phase 1) — no automated rebalance backtest yet.",
        "universe": "11 SPDR sector ETFs (XLK/XLC/XLV/XLF/XLY/XLP/XLI/XLE/XLB/XLU/XLRE).",
        "breadth_note": (
            "Breadth uses constituent stocks (asset_type=Stock) mapped to the same Sector "
            "via SubTheme→Sector taxonomy. % eligible / % bullish classification can confirm "
            "or contradict the ETF-level signal."
        ),
        "dispersion_note": (
            "Dispersion = max(composite) - min(composite) across the 11 sectors. "
            "Higher dispersion = more rotation alpha available (sectors decoupling)."
        ),
    }

    return {
        "as_of": scan_time or datetime.utcnow().isoformat(),
        "regime": regime_info,
        "sectors": sector_rows,
        "summary": summary,
        "methodology": methodology,
    }


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys, pickle
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_cache.pkl")
    if not os.path.exists(cache_path):
        print("No scan cache found. Run price_discovery.py first.")
        sys.exit(1)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    # Build df via api._load_cache for breadth (or use raw without breadth)
    try:
        from api import _load_cache, STATE
        _load_cache()
        df = STATE.get("df")
    except Exception as e:
        print(f"Could not load df: {e}. Running without breadth.")
        df = None

    out = compute_sector_rotation(cache.get("results", []), df=df,
                                    scan_time=cache.get("scan_time"))
    print(f"As of: {out['as_of']}")
    print(f"\nSummary: {out['summary']}\n")
    print(f"{'Ticker':<6} {'Sector':<25} {'Comp':>5} {'OER':>4} {'Tier':<12} "
          f"{'Decision':<10} {'Class':<22} {'Br%':>5}")
    print("-" * 105)
    for s in out["sectors"]:
        if s.get("missing"):
            print(f"{s['ticker']:<6} {s['sector']:<25} (missing from universe)")
            continue
        br = s.get("breadth", {}).get("pct_eligible", 0)
        print(f"{s['ticker']:<6} {s['sector']:<25} {s['composite']:>5.1f} {s['oer']:>4.0f} "
              f"{s['tier']:<12} {s['decision']:<10} {s['classification']:<22} {br:>5.1f}")

