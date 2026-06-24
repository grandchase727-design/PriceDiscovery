# -*- coding: utf-8 -*-
"""portfolio_composer.py — Option C Phase 2: Portfolio Composition Layer.

================================================================================
PURPOSE
================================================================================

Takes per-ticker debate results (with debate_synthesis, timing, risk_verdict
per pick) and composes a FINAL PORTFOLIO with:

  1. EXCLUDED tickers dropped
  2. INCLUDE_REDUCED_SIZE tickers marked for half-size
  3. Sector concentration cap enforced
  4. Horizon balance preserved
  5. Final pick count budget respected
  6. Re-validation after drops (no orphan picks)

================================================================================
INPUT FORMAT (from per_ticker_debate.run_per_ticker_debate)
================================================================================

pm_horizons = {
  "tactical":  {"long_stocks":[...], "long_etfs":[...], "short_stocks":[...], "short_etfs":[...]},
  "core":      {...},
  "strategic": {...},
}

Each pick augmented with:
  p["debate_synthesis"] = {tier, final_decision, stars, debate_transcript, key_factor, _failed?}
  p["timing"]           = {entry_signal, urgency, ...}
  p["risk_verdict"]     = {vote, key_risk, rationale}

================================================================================
OUTPUT FORMAT
================================================================================

Same shape as input but:
  - Each pick has p["composition_decision"] = "INCLUDE" / "INCLUDE_HALF" / "EXCLUDED_BY_CAP" / "EXCLUDED_BY_DEBATE"
  - Excluded picks REMAIN in the structure (for transparency) but flagged
  - Each pick has p["final_size"] = 1.0 / 0.5 / 0.0
  - portfolio_metadata key added at top level

================================================================================
CAPS & BUDGETS (defaults — overridable)
================================================================================

  MAX_SECTOR_WEIGHT       = 0.30  # 30% per sector max
  MAX_PICKS_PER_HORIZON   = 20    # already constrained upstream
  MIN_PICKS_PER_HORIZON   = 3     # below this, fail-soft
  REGIME_PICK_MULTIPLIER  = {     # T5 fix: adaptive sizing by regime
      "RISK_ON": 1.0,
      "NEUTRAL": 0.8,
      "RISK_OFF": 0.6,
      "LATE_CYCLE": 0.7,
      "ROTATION_IN_PROGRESS": 0.85,
  }
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional


# ── Default constants (overridable) ────────────────────────────────
MAX_SECTOR_WEIGHT          = 0.30
MAX_PICKS_PER_HORIZON      = 20
MIN_PICKS_PER_HORIZON      = 3
REGIME_PICK_MULTIPLIER = {
    "RISK_ON":               1.0,
    "RISK_ON_BIAS":          0.95,
    "NEUTRAL":               0.8,
    "RISK_OFF":              0.6,
    "RISK_AVERSE":           0.65,
    "LATE_CYCLE":            0.7,
    "EARLY_CYCLE":           0.9,
    "ROTATION_IN_PROGRESS":  0.85,
    "TRANSITIONAL":          0.75,
    "MIXED":                 0.8,
}


def _classify_decision(p: dict) -> str:
    """Translate per-ticker debate output into compose-level decision.

    Returns: "INCLUDE" / "INCLUDE_HALF" / "EXCLUDE" / "WATCH"
    """
    ds = p.get("debate_synthesis") or {}
    if ds.get("_failed"):
        # Honest failure path: composite-based fallback (mirrors final_list.py logic)
        comp = float(p.get("composite") or 0)
        cls = p.get("classification") or ""
        cls_str = cls if isinstance(cls, str) else ""
        is_strong = any(s in cls_str for s in ("CONTINUATION","FORMATION","RECOVERY","LAGGING_CATCHUP"))
        if comp >= 75 and is_strong: return "INCLUDE"
        if comp >= 65 and is_strong: return "INCLUDE_HALF"
        if comp >= 55:               return "WATCH"
        return "EXCLUDE"
    fd = (ds.get("final_decision") or "WATCH").upper()
    if fd == "INCLUDE":              return "INCLUDE"
    if fd == "INCLUDE_REDUCED_SIZE": return "INCLUDE_HALF"
    if fd == "EXCLUDE":              return "EXCLUDE"
    return "WATCH"


def _adaptive_pick_budget(regime_tag: str, base: int = MAX_PICKS_PER_HORIZON) -> int:
    """Compute adaptive per-horizon pick budget based on regime.

    Hard market regimes (RISK_OFF, LATE_CYCLE) reduce pick count to avoid
    forcing low-conviction picks into portfolio.
    """
    if not regime_tag:
        return base
    rt = (regime_tag or "").upper()
    # Find the highest-matching multiplier (tags can have multiple keywords)
    best_mult = 1.0
    for key, mult in REGIME_PICK_MULTIPLIER.items():
        if key in rt:
            best_mult = min(best_mult, mult)
    return max(MIN_PICKS_PER_HORIZON, int(base * best_mult))


def _enforce_sector_cap(picks: list[dict], max_weight: float = MAX_SECTOR_WEIGHT) -> tuple[list[dict], list[dict]]:
    """Enforce sector concentration cap.

    Picks are accepted in order (assume sorted by conviction). Once a sector
    reaches the cap, additional picks from that sector are downgraded to WATCH.

    Returns: (accepted_picks, capped_picks)
    """
    if not picks:
        return [], []

    total_target = len(picks)
    cap_count = max(1, int(total_target * max_weight))

    sector_counter: Counter = Counter()
    accepted = []
    capped = []
    for p in picks:
        sector = p.get("sector") or "UNKNOWN"
        if sector_counter[sector] < cap_count:
            sector_counter[sector] += 1
            accepted.append(p)
        else:
            p["composition_decision"] = "EXCLUDED_BY_CAP"
            p["final_size"] = 0.0
            p["_excluded_reason"] = f"sector_cap:{sector}"
            capped.append(p)
    return accepted, capped


def _sort_by_conviction(picks: list[dict]) -> list[dict]:
    """Sort picks by composite × tier_weight × signal_strength."""
    def _score(p: dict) -> float:
        comp = float(p.get("composite") or 0)
        ds = p.get("debate_synthesis") or {}
        tier_weight = {"UNANIMOUS": 1.3, "MAJORITY_CLEAN": 1.2,
                        "MAJORITY_DISSENT": 1.0, "SOLO": 0.8, "EXCLUDED": 0.5}.get(
            ds.get("tier") or "SOLO", 0.8)
        sig = (p.get("timing") or {}).get("entry_signal") or ""
        sig_weight = {"BUY_NOW": 1.2, "WAIT": 1.0, "SKIP": 0.6}.get(sig, 1.0)
        risk_v = (p.get("risk_verdict") or {}).get("vote") or ""
        risk_weight = {"APPROVE": 1.1, "CAUTION": 1.0, "REJECT": 0.6}.get(risk_v, 1.0)
        return comp * tier_weight * sig_weight * risk_weight
    return sorted(picks, key=_score, reverse=True)


def compose_portfolio(pm_horizons: dict, regime_tag: str = "",
                       _emit_fn: Optional[callable] = None,
                       max_sector_weight: float = MAX_SECTOR_WEIGHT) -> dict:
    """Main entry point — compose portfolio from per-ticker debate results.

    Args:
        pm_horizons: per_ticker_debate output
        regime_tag:  for adaptive pick budget
        _emit_fn:    optional progress emission

    Returns:
        portfolio = {
          "horizons": same shape as input, augmented with composition_decision + final_size,
          "metadata": {
             "regime_tag": ...,
             "adaptive_budget_per_horizon": ...,
             "totals": {n_include, n_include_half, n_excluded_debate, n_excluded_cap, n_watch},
             "sector_distribution": {...},
             "horizon_distribution": {...},
             "warnings": [],
          }
        }
    """
    def _emit(phase: str, status: str):
        if _emit_fn:
            try: _emit_fn(phase, "portfolio_composer", status)
            except Exception: pass

    _emit("phase5b_compose", "started")

    adaptive_budget = _adaptive_pick_budget(regime_tag)

    out_horizons = {}
    totals = Counter()
    sector_dist: Counter = Counter()
    horizon_dist: Counter = Counter()
    warnings = []

    for h in ("tactical", "core", "strategic"):
        hd = pm_horizons.get(h, {}) or {}
        out_horizons[h] = {}

        for bk in ("long_stocks", "long_etfs", "short_stocks", "short_etfs"):
            picks = list(hd.get(bk, []) or [])

            # ── Step 1: Classify each pick by debate decision ──
            for p in picks:
                decision = _classify_decision(p)
                p["composition_decision"] = decision  # may be overwritten by cap below
                if decision == "INCLUDE":
                    p["final_size"] = 1.0
                elif decision == "INCLUDE_HALF":
                    p["final_size"] = 0.5
                else:
                    p["final_size"] = 0.0
                    p["_excluded_reason"] = "debate_" + decision.lower()

            # ── Step 2: Filter EXCLUDE/WATCH out for cap enforcement ──
            keepers = [p for p in picks
                        if p.get("composition_decision") in ("INCLUDE", "INCLUDE_HALF")]

            # ── Step 3: Sort by conviction (highest first) ──
            keepers_sorted = _sort_by_conviction(keepers)

            # ── Step 4: Apply adaptive budget ──
            if len(keepers_sorted) > adaptive_budget:
                budget_overflow = keepers_sorted[adaptive_budget:]
                for p in budget_overflow:
                    p["composition_decision"] = "EXCLUDED_BY_BUDGET"
                    p["final_size"] = 0.0
                    p["_excluded_reason"] = f"budget_overflow:{adaptive_budget}"
                keepers_sorted = keepers_sorted[:adaptive_budget]

            # ── Step 5: Enforce sector cap ──
            accepted, capped = _enforce_sector_cap(keepers_sorted, max_sector_weight)

            # ── Step 6: Re-attach all picks (accepted + excluded) to output ──
            # Preserve original ticker order for downstream compatibility
            ticker_to_pick = {p.get("ticker"): p for p in picks}
            out_horizons[h][bk] = list(ticker_to_pick.values())

            # ── Tally ──
            for p in picks:
                d = p.get("composition_decision") or "EXCLUDED"
                if d == "INCLUDE":
                    totals["n_include"] += 1
                    sector_dist[p.get("sector") or "UNKNOWN"] += 1
                    horizon_dist[h] += 1
                elif d == "INCLUDE_HALF":
                    totals["n_include_half"] += 1
                    sector_dist[p.get("sector") or "UNKNOWN"] += 0.5
                    horizon_dist[h] += 0.5
                elif d == "EXCLUDED_BY_CAP":
                    totals["n_excluded_cap"] += 1
                elif d == "EXCLUDED_BY_BUDGET":
                    totals["n_excluded_budget"] += 1
                elif d == "EXCLUDE":
                    totals["n_excluded_debate"] += 1
                else:
                    totals["n_watch"] += 1

        # ── Per-horizon health checks ──
        active_in_h = sum(
            1 for bk in ("long_stocks","long_etfs","short_stocks","short_etfs")
            for p in out_horizons[h][bk]
            if p.get("composition_decision") in ("INCLUDE", "INCLUDE_HALF")
        )
        if active_in_h < MIN_PICKS_PER_HORIZON:
            warnings.append(f"{h}: only {active_in_h} active picks (min {MIN_PICKS_PER_HORIZON})")

    # ── Cross-horizon health checks ──
    if totals["n_include"] + totals["n_include_half"] < 5:
        warnings.append("portfolio_too_thin: less than 5 INCLUDE picks across all horizons")

    metadata = {
        "regime_tag": regime_tag,
        "adaptive_budget_per_horizon": adaptive_budget,
        "max_sector_weight": max_sector_weight,
        "totals": dict(totals),
        "sector_distribution": dict(sector_dist),
        "horizon_distribution": dict(horizon_dist),
        "warnings": warnings,
    }

    _emit("phase5b_compose",
          f"included={totals.get('n_include',0)} half={totals.get('n_include_half',0)} "
          f"cap={totals.get('n_excluded_cap',0)} bud={totals.get('n_excluded_budget',0)} "
          f"exc={totals.get('n_excluded_debate',0)} watch={totals.get('n_watch',0)}")

    return {"horizons": out_horizons, "metadata": metadata}


# ─────────────────────────────────────────────────────────────────
# Helpers for monitoring + integration
# ─────────────────────────────────────────────────────────────────

def summarize_composition(portfolio: dict) -> dict:
    """Compact summary for monitoring."""
    meta = portfolio.get("metadata", {})
    return {
        "regime_tag": meta.get("regime_tag"),
        "adaptive_budget": meta.get("adaptive_budget_per_horizon"),
        "active_picks": meta.get("totals", {}).get("n_include", 0)
                        + meta.get("totals", {}).get("n_include_half", 0) * 0.5,
        "excluded_total": meta.get("totals", {}).get("n_excluded_debate", 0)
                          + meta.get("totals", {}).get("n_excluded_cap", 0)
                          + meta.get("totals", {}).get("n_excluded_budget", 0),
        "warnings": meta.get("warnings", []),
        "sector_top3": sorted(
            (meta.get("sector_distribution") or {}).items(),
            key=lambda x: -x[1]
        )[:3],
    }
