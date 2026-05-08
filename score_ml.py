"""
score_ml.py — apply ML-optimized Composite weights to the scan cache.

Loads:
  • .scan_cache.pkl           (full scan results — 756 tickers)
  • .ml_optimized_params.json (per-asset-class best weights from optimize_params.py)

Produces:
  • .scan_cache_ml.pkl        (parallel cache with ML-rescored results)

What changes per ticker:
  1. composite_ml = w_tcs·tcs + w_tfs·tfs + w_rss·rss + w_urs·urs
                    using the asset-class-specific weights
  2. classification_ml — recomputed via 3×3 matrix using new composite
                          (rough re-classification: same tcs_short/long, tfs_short/long,
                           but threshold buckets shift if composite shifts buckets)
  3. eligible_ml — Eligibility Gate re-applied with new composite
        (composite_ml ≥ 55, classification_ml ∈ bullish, ADV ≥ $5M, QVR ≥ 40 for stocks)
  4. rejection_ml — updated rejection reason

The 3-bucket lifecycle (pre-momentum / momentum / excluded) is then
recomputed using the new fields. This produces a parallel result list
suitable for the "Price Discovery (ML)" dashboard tab.

The original fields (composite, classification, eligible, etc.) are
preserved unchanged so the original Price Discovery tab is unaffected.

Usage:
  python3 score_ml.py
  python3 score_ml.py --cache .scan_cache.pkl --params .ml_optimized_params.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from copy import deepcopy
from typing import Any, Dict, List, Optional

from optimize_params import asset_class_of, DEFAULT_WEIGHTS

CACHE_PATH = ".scan_cache.pkl"
ML_CACHE_PATH = ".scan_cache_ml.pkl"
PARAMS_PATH = ".ml_optimized_params.json"


# Bullish classifications (mirror api.py:_BULLISH_SET / Eligibility Gate)
BULLISH_SET = {
    "🟢 CONTINUATION", "🔵 RECOVERY", "🔵 FORMATION",
    "🟡 OVEREXTENDED", "🟦 LAGGING_CATCHUP",
}

# Pre-momentum classifications (mirror PriceDiscoveryTab.tsx PM_CLASSIFICATIONS)
PM_SET = {
    "🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔵 RECOVERY",
    "🔶 PULLBACK", "⚠️ WEAKENING", "🟤 FADING",
}


# ─────────────────────────────────────────────────────────────────────
# Composite reweighting
# ─────────────────────────────────────────────────────────────────────

def _reweight_composite(r: Dict[str, Any], weights: Dict[str, float]) -> float:
    """ML-Composite using new weights.

    Uses tcs, tfs from the result. For RSS / URS, we have to reverse-engineer
    from score_composite (= the original Composite) since the per-result dict
    doesn't always carry rss/urs separately. Same approach as optimize_params.py
    attach_features().
    """
    tcs = float(r.get("tcs", 50.0) or 50.0)
    tfs = float(r.get("tfs", 50.0) or 50.0)
    rss = r.get("rss")
    urs = r.get("urs")
    if rss is None or urs is None:
        # Reverse-engineer combined RSS+URS from original Composite
        sc = float(r.get("composite", 50.0))
        combined = (sc - 0.30 * tcs - 0.25 * tfs) / 0.45
        combined = max(0.0, min(100.0, combined))
        rss = float(rss) if rss is not None else combined
        urs = float(urs) if urs is not None else combined
    rss = float(rss)
    urs = float(urs)
    return round(weights["w_tcs"] * tcs + weights["w_tfs"] * tfs
                 + weights["w_rss"] * rss + weights["w_urs"] * urs, 1)


def _classify_ml(r: Dict[str, Any], composite_ml: float) -> str:
    """Re-classify based on ML-Composite.

    Heuristic: when ML-Composite shifts up/down significantly vs original,
    bump classification across thresholds. Otherwise keep original classification.

    This is a coarse approximation — full re-classification would require
    re-running NaiveDiscoveryDetector.classify with raw indicators.
    """
    orig_class = r.get("classification", "🟠 NEUTRAL")
    orig_comp = float(r.get("composite", 50.0))
    delta = composite_ml - orig_comp

    # If shift is small (<5 points) keep original classification
    if abs(delta) < 5.0:
        return orig_class

    # Major upgrade: composite jumps ≥10 points → upgrade weak classes to RECOVERY
    if delta >= 10.0 and orig_class in {"🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔶 PULLBACK", "⚠️ WEAKENING"}:
        return "🔵 RECOVERY"
    # Major downgrade: composite drops ≥10 points → downgrade strong classes
    if delta <= -10.0 and orig_class in {"🟢 CONTINUATION", "🔵 RECOVERY", "🔵 FORMATION"}:
        return "🟡 CONSOLIDATION"
    return orig_class


def _eligible_ml(r: Dict[str, Any], composite_ml: float, classification_ml: str
                 ) -> Dict[str, Any]:
    """Re-apply Eligibility Gate using ML-Composite + ML-classification.

    4 conditions (mirror api.py Eligibility Gate):
        1. Composite ≥ 55
        2. classification ∈ bullish set
        3. ADV ≥ $5M
        4. asset = ETF OR QVR ≥ 40 (Stock-only fundamental sanity)
    """
    rejection_parts: List[str] = []

    if composite_ml < 55:
        rejection_parts.append(f"LowScore({composite_ml:.0f})")
    if classification_ml not in BULLISH_SET:
        rejection_parts.append(f"Class({classification_ml})")

    adv_usd = float(r.get("adv_usd") or 0.0)
    if adv_usd < 5_000_000:
        rejection_parts.append(f"Liq(${adv_usd / 1e6:.1f}M)")

    # QVR check (Stock only — ETF auto-pass)
    cat = r.get("category", "")
    if isinstance(cat, str) and cat.startswith("STK"):
        qvr = r.get("qvr_score") or r.get("qvr") or 50.0
        try:
            qvr_v = float(qvr)
        except Exception:
            qvr_v = 50.0
        if qvr_v < 40:
            rejection_parts.append(f"WeakQVR({qvr_v:.0f})")

    eligible = len(rejection_parts) == 0
    return {
        "eligible_ml": eligible,
        "rejection_ml": "" if eligible else "·".join(rejection_parts),
    }


def _stage_ml(r: Dict[str, Any]) -> str:
    """Map ML state → 3-tier lifecycle stage (parallels stageBadge in PriceDiscoveryTab)."""
    if r.get("eligible_ml"):
        return "momentum"
    cls = r.get("classification_ml", r.get("classification", ""))
    if cls in PM_SET:
        return "pre-momentum"
    return "excluded"


# ─────────────────────────────────────────────────────────────────────
# Apply to cache
# ─────────────────────────────────────────────────────────────────────

def apply_ml_weights(cache_path: str = CACHE_PATH,
                     params_path: str = PARAMS_PATH,
                     output_path: str = ML_CACHE_PATH) -> Dict[str, Any]:
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} not found.")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"{params_path} not found. Run optimize_params.py first.")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    with open(params_path) as f:
        params = json.load(f)

    classes = params.get("asset_classes", {})
    weights_per_class: Dict[str, Dict[str, float]] = {}
    for ac, info in classes.items():
        if "best_weights" in info:
            weights_per_class[ac] = info["best_weights"]
    # Fallback for asset classes without optimized weights
    fallback = DEFAULT_WEIGHTS

    results = cache.get("results", [])
    new_results: List[Dict[str, Any]] = []
    stage_counts = {"pre-momentum": 0, "momentum": 0, "excluded": 0}
    delta_distribution = {"upgraded_to_momentum": 0, "demoted_from_momentum": 0, "unchanged": 0}

    for r in results:
        nr = deepcopy(r)
        ac = asset_class_of(nr["ticker"], nr.get("category", ""))
        weights = weights_per_class.get(ac, fallback)

        composite_ml = _reweight_composite(nr, weights)
        classification_ml = _classify_ml(nr, composite_ml)
        gate = _eligible_ml(nr, composite_ml, classification_ml)

        nr["composite_ml"] = composite_ml
        nr["classification_ml"] = classification_ml
        nr["eligible_ml"] = gate["eligible_ml"]
        nr["rejection_ml"] = gate["rejection_ml"]
        nr["asset_class_ml"] = ac
        nr["weights_ml"] = weights

        # Original stage (for delta tracking)
        orig_eligible = bool(nr.get("eligible"))
        new_eligible = gate["eligible_ml"]
        if orig_eligible == new_eligible:
            delta_distribution["unchanged"] += 1
        elif new_eligible and not orig_eligible:
            delta_distribution["upgraded_to_momentum"] += 1
        else:
            delta_distribution["demoted_from_momentum"] += 1

        nr["stage_ml"] = _stage_ml(nr)
        stage_counts[nr["stage_ml"]] += 1
        new_results.append(nr)

    cache["results_ml"] = new_results
    cache["ml_meta"] = {
        "as_of": params.get("as_of"),
        "weights_per_class": weights_per_class,
        "stage_counts": stage_counts,
        "delta_distribution": delta_distribution,
        "n_total": len(new_results),
        "params_source": params_path,
    }

    with open(output_path, "wb") as f:
        pickle.dump(cache, f)

    return cache["ml_meta"]


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=CACHE_PATH)
    p.add_argument("--params", default=PARAMS_PATH)
    p.add_argument("--output", default=ML_CACHE_PATH)
    args = p.parse_args()

    meta = apply_ml_weights(args.cache, args.params, args.output)
    print("ML scoring applied.")
    print(f"  total tickers : {meta['n_total']}")
    print(f"  stage counts  : {meta['stage_counts']}")
    print(f"  delta vs orig : {meta['delta_distribution']}")
    print(f"  weights       :")
    for ac, w in meta["weights_per_class"].items():
        print(f"    {ac:10s}  TCS {w['w_tcs']:.3f} | TFS {w['w_tfs']:.3f} "
              f"| RSS {w['w_rss']:.3f} | URS {w['w_urs']:.3f}")
    print(f"\n✓ Wrote {args.output}")
