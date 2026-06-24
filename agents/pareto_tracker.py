# -*- coding: utf-8 -*-
"""pareto_tracker.py — Option C Phase 5: Pareto Front Tracking for Picks.

================================================================================
PURPOSE
================================================================================

Replaces the single best_round_snapshot heuristic with proper multi-objective
Pareto-optimal tracking across all picks seen during the iteration.

A pick is Pareto-dominated by another pick if the other:
  - has higher composite OR equal
  - has fewer objections OR equal
  - has higher debate conviction OR equal
  - has equal or better risk verdict
  AND is strictly better in at least one dimension.

A pick is on the Pareto front if NO other pick dominates it. The Pareto front
is the set of non-dominated picks — these are the "best" picks in a
multi-objective sense.

================================================================================
WHY THIS MATTERS
================================================================================

Single best_round_snapshot picks a "best round" by a scalar score. But a round
might be "best on average" while having a few duds and missing some great picks
from a worse round. Pareto front tracking solves this:
  - Take the BEST PICKS across all rounds, not the BEST ROUND
  - A pick from Round 2 that dominates a Round 4 pick survives
  - Final portfolio = union of Pareto-optimal picks across iteration

================================================================================
USAGE
================================================================================

tracker = ParetoFrontTracker()
for round_num in range(1, max_rounds + 1):
    picks = run_pm_round(round_num)
    tracker.add_round(round_num, picks)

# Get Pareto-optimal picks (across all rounds)
final_picks = tracker.get_pareto_optimal()

================================================================================
"""
from __future__ import annotations

from typing import Optional


def _conviction_score(pick: dict) -> float:
    """Composite × tier_weight × signal_strength."""
    comp = float(pick.get("composite") or 0)
    ds = pick.get("debate_synthesis") or {}
    tier_weight = {"UNANIMOUS": 1.3, "MAJORITY_CLEAN": 1.2,
                    "MAJORITY_DISSENT": 1.0, "SOLO": 0.8, "EXCLUDED": 0.5}.get(
        ds.get("tier") or "SOLO", 0.8)
    sig = (pick.get("timing") or {}).get("entry_signal") or ""
    sig_weight = {"BUY_NOW": 1.2, "WAIT": 1.0, "SKIP": 0.6}.get(sig, 1.0)
    risk_v = (pick.get("risk_verdict") or {}).get("vote") or ""
    risk_weight = {"APPROVE": 1.1, "CAUTION": 1.0, "REJECT": 0.6}.get(risk_v, 1.0)
    return comp * tier_weight * sig_weight * risk_weight


def _pick_dimensions(pick: dict) -> tuple:
    """Extract Pareto comparison dimensions: (composite, -obj_count, conviction, risk_score).

    Higher is better for all 4. Risk_score: APPROVE=2, CAUTION=1, REJECT=0.
    Obj count negated so higher = fewer objections = better.
    """
    comp = float(pick.get("composite") or 0)
    # Objections — heuristic: count failure markers
    debate_failed  = bool((pick.get("debate_synthesis") or {}).get("_failed"))
    trading_failed = bool((pick.get("timing") or {}).get("_failed"))
    risk_failed    = bool((pick.get("risk_verdict") or {}).get("_failed"))
    n_failed = sum([debate_failed, trading_failed, risk_failed])
    obj_neg = -n_failed
    # Conviction
    conv = _conviction_score(pick)
    # Risk score
    risk_v = (pick.get("risk_verdict") or {}).get("vote") or ""
    risk_score = {"APPROVE": 2, "CAUTION": 1, "REJECT": 0}.get(risk_v, 0)
    return (comp, obj_neg, conv, risk_score)


def _dominates(a: tuple, b: tuple) -> bool:
    """Return True if A dominates B (A is >= B on all dims, > on at least one)."""
    if any(av < bv for av, bv in zip(a, b)):
        return False
    return any(av > bv for av, bv in zip(a, b))


class ParetoFrontTracker:
    """Tracks Pareto-optimal picks across iteration rounds.

    For each ticker, keeps the "best version" seen across rounds, where "best"
    is defined by multi-objective Pareto dominance.
    """

    def __init__(self):
        # (h, bucket, ticker) → (round_seen, pick_dict, dims)
        self._best: dict[tuple, tuple] = {}
        self._all_seen: list[dict] = []   # full audit trail

    def add_round(self, round_num: int, pm_horizons: dict) -> None:
        """Process picks from one iteration round, updating Pareto front."""
        for h in ("tactical", "core", "strategic"):
            for bk in ("long_stocks", "long_etfs", "short_stocks", "short_etfs"):
                picks = (pm_horizons.get(h, {}) or {}).get(bk, []) or []
                for p in picks:
                    t = p.get("ticker")
                    if not t:
                        continue
                    key = (h, bk, t)
                    dims = _pick_dimensions(p)
                    self._all_seen.append({
                        "round": round_num, "key": key, "pick": p, "dims": dims
                    })
                    if key not in self._best:
                        self._best[key] = (round_num, p, dims)
                    else:
                        prev_round, prev_pick, prev_dims = self._best[key]
                        # If new dims dominate previous, replace; else keep previous.
                        if _dominates(dims, prev_dims):
                            self._best[key] = (round_num, p, dims)
                        elif _dominates(prev_dims, dims):
                            pass   # keep previous
                        else:
                            # Incomparable — prefer the more recent (latest evidence)
                            self._best[key] = (round_num, p, dims)

    def get_pareto_optimal(self) -> dict:
        """Return Pareto-optimal picks as pm_horizons-shaped dict."""
        result = {h: {bk: [] for bk in ("long_stocks","long_etfs","short_stocks","short_etfs")}
                  for h in ("tactical","core","strategic")}
        for (h, bk, t), (round_seen, pick, dims) in self._best.items():
            pick["_pareto_best_round"] = round_seen
            pick["_pareto_dims"] = dims
            result[h][bk].append(pick)
        return result

    def get_audit_trail(self) -> list[dict]:
        """Full audit of every (round, ticker) seen with dimensions — for debugging."""
        return [
            {
                "round": r["round"],
                "ticker": r["key"][2],
                "horizon": r["key"][0],
                "bucket": r["key"][1],
                "dims": r["dims"],
            }
            for r in self._all_seen
        ]

    def summary(self) -> dict:
        """Compact summary for monitoring."""
        rounds_per_ticker = {}
        for r in self._all_seen:
            t = r["key"][2]
            rounds_per_ticker.setdefault(t, []).append(r["round"])
        # Stability: tickers that survived ALL rounds
        if rounds_per_ticker:
            max_rounds_seen = max(len(rs) for rs in rounds_per_ticker.values())
            stable = sum(1 for rs in rounds_per_ticker.values() if len(rs) == max_rounds_seen)
        else:
            max_rounds_seen = 0
            stable = 0
        # Average Pareto dims
        if self._best:
            avg_dims = [0.0] * 4
            n = len(self._best)
            for _, _, dims in self._best.values():
                for i, v in enumerate(dims):
                    avg_dims[i] += v
            avg_dims = [v / n for v in avg_dims]
        else:
            avg_dims = [0.0] * 4
        return {
            "n_unique_tickers": len(self._best),
            "n_observations": len(self._all_seen),
            "stable_tickers": stable,
            "max_rounds_seen": max_rounds_seen,
            "avg_pareto_dims": {
                "composite": avg_dims[0],
                "neg_n_failed": avg_dims[1],
                "conviction": avg_dims[2],
                "risk_score": avg_dims[3],
            },
        }


# ─────────────────────────────────────────────────────────────────
# Adaptive Macro Convergence Threshold
# ─────────────────────────────────────────────────────────────────

def adaptive_convergence_threshold(regime_tag: str, base: float = 0.20) -> float:
    """Adjust convergence threshold based on regime difficulty.

    Hard regimes (RISK_OFF, LATE_CYCLE) → tighter threshold (harder to converge,
    accept more iteration). Easy regimes (RISK_ON) → looser threshold (faster
    convergence is fine).
    """
    if not regime_tag:
        return base
    rt = (regime_tag or "").upper()
    if "RISK_OFF" in rt or "LATE_CYCLE" in rt or "ROTATION" in rt:
        # Hard regime → tighter (require more agreement)
        return base * 0.75
    elif "RISK_ON" in rt or "EARLY_CYCLE" in rt:
        # Easy regime → looser
        return base * 1.25
    elif "TRANSITIONAL" in rt or "MIXED" in rt:
        return base * 0.9
    return base


def adaptive_pick_pool_size(regime_tag: str, base: int = 240) -> int:
    """Adjust initial PM pick pool by regime — feeds upstream PM prompt.

    The PM gets a smaller candidate pool to evaluate in hard regimes, focusing
    on highest-conviction picks rather than diluting across many low-conviction.
    """
    if not regime_tag:
        return base
    rt = (regime_tag or "").upper()
    if "RISK_OFF" in rt:
        return max(60, int(base * 0.4))   # 240 → 96
    if "LATE_CYCLE" in rt:
        return max(80, int(base * 0.55))   # 240 → 132
    if "ROTATION" in rt:
        return max(100, int(base * 0.7))   # 240 → 168
    if "RISK_ON" in rt:
        return base                        # 240 → 240
    return int(base * 0.85)                # transitional default
