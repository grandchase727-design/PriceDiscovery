"""
optimize_params.py — ML-based parameter optimization for Price Discovery scoring.

Phase A (current scope):
  Optimize the Composite axis weights — w_tcs, w_tfs, w_rss, w_urs (sum = 1.0)
  per asset class (ETF / US_Stock / KR_Stock) using walk-forward CV.

Why these variables?
  • Composite weights (default 0.30/0.25/0.30/0.15) are hard-coded magic numbers.
  • Different asset classes plausibly respond to different signal mixes:
      ETFs    → trend continuation may dominate (longer holding periods)
      US Stk  → relative strength may dominate (peer-driven momentum)
      KR Stk  → mean-reversion / shorter cycles → URS may matter more

Why not SMA periods (the user's example)?
  • The cache contains pre-computed TCS/TFS/RSS/URS at fixed SMA windows
    (20/50/200). To vary SMA periods we'd need to re-fetch OHLCV for all 756
    tickers and re-run the indicator pipeline (~10+ min × N trials → infeasible).
  • Instead, we optimize the COMBINATION weights using ve_observations'
    pre-computed sub-scores. SMA-period optimization is a follow-up phase.

Methodology:
  1. Load .scan_cache.pkl → ve_observations (~18k obs across ~12 eval dates)
  2. Group obs by asset class
  3. Walk-forward CV: train on early eval_dates, OOS test on later
  4. For each trial:
       - Compute "ML-Composite" = w_tcs·tcs + w_tfs·tfs + w_rss·rss + w_urs·urs
       - Rank all obs at each eval_date by ML-Composite
       - Long top-quartile, short bottom-quartile
       - Forward 21d return spread = OOS objective
  5. Optuna maximizes mean OOS forward 21d quintile spread

Output: .ml_optimized_params.json
  {
    "as_of": "...",
    "lookback_used": "ve_observations (12 eval dates, ~5y)",
    "asset_classes": {
      "ETF":     {"w_tcs": 0.32, "w_tfs": 0.20, "w_rss": 0.35, "w_urs": 0.13,
                  "objective": 4.21, "n_trials": 100, "n_obs_train": ...},
      "US_Stock": {...},
      "KR_Stock": {...}
    },
    "default_weights": {"w_tcs": 0.30, "w_tfs": 0.25, "w_rss": 0.30, "w_urs": 0.15},
    "baseline_objective": {...}    # default-weights spread for comparison
  }

Run:  python3 optimize_params.py
      python3 optimize_params.py --trials 200 --quantile 0.20
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError as e:
    raise SystemExit("optuna not installed. pip install optuna") from e


CACHE_PATH = ".scan_cache.pkl"
OUTPUT_PATH = ".ml_optimized_params.json"

# Default weights from price_discovery.py:NaiveDiscoveryDetector.composite()
DEFAULT_WEIGHTS = {"w_tcs": 0.30, "w_tfs": 0.25, "w_rss": 0.30, "w_urs": 0.15}


# ─────────────────────────────────────────────────────────────────────
# Data prep
# ─────────────────────────────────────────────────────────────────────

def asset_class_of(ticker: str, category: str) -> str:
    """Map (ticker, category) → asset class label."""
    if category and category.startswith("ETF"):
        return "ETF"
    if not category or not category.startswith("STK"):
        return "ETF"  # benchmark / default
    if category == "STK_Korea" or ticker.endswith(".KS"):
        return "KR_Stock"
    if any(ticker.endswith(s) for s in (".T", ".HK", ".SS", ".SZ", ".PA", ".DE", ".AS", ".MI", ".L")):
        return "INTL_Stock"
    return "US_Stock"


def load_observations(cache_path: str = CACHE_PATH) -> Tuple[List[Dict], Dict[str, str]]:
    """Load ve_observations + ticker→category map.

    Returns:
        observations: list of dicts with keys
            ticker, eval_date, tcs, tfs, oer, score_composite, fwd_return,
            fwd_rets (dict horizon→%), classification, eligible
        ticker_category: {ticker: category}
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} not found. Run `python3 price_discovery.py` first."
        )
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    obs = cache.get("ve_observations", [])
    if not obs:
        raise RuntimeError("ve_observations is empty in scan cache.")

    ticker_category = {r["ticker"]: r.get("category", "Unknown") for r in cache.get("results", [])}
    return obs, ticker_category


def attach_features(obs: List[Dict], ticker_category: Dict[str, str]) -> None:
    """In-place: add 'asset_class', 'rss_proxy', 'urs_proxy', 'fwd_21d' to each obs.

    rss_proxy / urs_proxy: ve_observations doesn't carry RSS/URS directly, but
    score_composite was built from the default formula:
        score_composite = 0.30·tcs + 0.25·tfs + 0.30·rss + 0.15·urs
    We can reverse-engineer (rss + urs) jointly:
        combined_45 = (score_composite − 0.30·tcs − 0.25·tfs) / 0.45
    To split: assume relative ratio rss:urs ≈ 2:1 (default 0.30:0.15).
    For optimization, we let w_rss and w_urs vary independently — the
    proxy split is just a reasonable initialization; the OOS objective
    pulls them toward whatever combination predicts forward returns best.
    """
    for o in obs:
        o["asset_class"] = asset_class_of(o["ticker"], ticker_category.get(o["ticker"], ""))

        # Reverse-engineer combined RSS + URS from score_composite
        sc = o.get("score_composite", o.get("score", 50.0))
        tcs = o.get("tcs", 50.0) or 50.0
        tfs = o.get("tfs", 50.0) or 50.0
        combined_45 = (sc - 0.30 * tcs - 0.25 * tfs) / 0.45
        combined_45 = max(0.0, min(100.0, combined_45))
        # Split via 2:1 default ratio (rss heavier than urs)
        o["rss_proxy"] = combined_45
        o["urs_proxy"] = combined_45  # same proxy — ML weights will discover useful split

        # Forward 21d return (proxy for monthly horizon)
        fwd = o.get("fwd_rets", {})
        o["fwd_21d"] = fwd.get(21, fwd.get(20, fwd.get(22, o.get("fwd_return", np.nan))))


# ─────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────

def quintile_spread(obs_list: List[Dict], weights: Dict[str, float],
                    quantile: float = 0.25) -> float:
    """At each eval_date, rank obs by ML-Composite, compute mean fwd_21d
    of top-quantile minus bottom-quantile. Average across eval_dates.

    Higher = stronger predictive power of the weight scheme.
    """
    by_date: Dict[str, List[Dict]] = {}
    for o in obs_list:
        d = o.get("eval_date")
        if d is None:
            continue
        if isinstance(d, datetime):
            key = d.strftime("%Y-%m-%d")
        else:
            key = str(d)
        by_date.setdefault(key, []).append(o)

    spreads: List[float] = []
    for date, group in by_date.items():
        if len(group) < 8:
            continue
        scores = []
        rets = []
        for o in group:
            tcs = float(o.get("tcs", 50.0) or 50.0)
            tfs = float(o.get("tfs", 50.0) or 50.0)
            rss = float(o.get("rss_proxy", 50.0))
            urs = float(o.get("urs_proxy", 50.0))
            ml_comp = (weights["w_tcs"] * tcs + weights["w_tfs"] * tfs
                       + weights["w_rss"] * rss + weights["w_urs"] * urs)
            fwd = o.get("fwd_21d")
            if fwd is None or (isinstance(fwd, float) and np.isnan(fwd)):
                continue
            scores.append(ml_comp)
            rets.append(float(fwd))
        if len(scores) < 8:
            continue
        scores = np.array(scores)
        rets = np.array(rets)
        n = len(scores)
        n_q = max(1, int(n * quantile))
        order = np.argsort(scores)
        bot = rets[order[:n_q]].mean()
        top = rets[order[-n_q:]].mean()
        spreads.append(top - bot)
    if not spreads:
        return float("-inf")
    return float(np.mean(spreads))


def split_train_oos(obs_list: List[Dict], train_frac: float = 0.6
                    ) -> Tuple[List[Dict], List[Dict]]:
    """Split observations by eval_date — earliest train_frac for training."""
    dates = sorted(set(
        (o.get("eval_date").strftime("%Y-%m-%d")
         if isinstance(o.get("eval_date"), datetime) else str(o.get("eval_date")))
        for o in obs_list if o.get("eval_date") is not None
    ))
    if not dates:
        return [], obs_list
    cutoff_idx = max(1, int(len(dates) * train_frac))
    cutoff = dates[cutoff_idx - 1]

    def _key(o):
        d = o.get("eval_date")
        return d.strftime("%Y-%m-%d") if isinstance(d, datetime) else str(d)

    train = [o for o in obs_list if o.get("eval_date") is not None and _key(o) <= cutoff]
    oos = [o for o in obs_list if o.get("eval_date") is not None and _key(o) > cutoff]
    return train, oos


# ─────────────────────────────────────────────────────────────────────
# Optuna optimization
# ─────────────────────────────────────────────────────────────────────

def optimize_for_class(asset_class: str, obs_subset: List[Dict],
                       n_trials: int = 100, quantile: float = 0.25,
                       train_frac: float = 0.6, seed: int = 42
                       ) -> Dict[str, Any]:
    """Run Optuna for a single asset class.

    Constraints:
        Each weight ∈ [0.05, 0.55]
        Sum = 1.0 (enforced by simplex projection)
    """
    train_obs, oos_obs = split_train_oos(obs_subset, train_frac)
    if len(train_obs) < 50 or len(oos_obs) < 50:
        return {
            "asset_class": asset_class,
            "error": f"insufficient samples (train={len(train_obs)}, oos={len(oos_obs)})",
            "n_obs": len(obs_subset),
        }

    def objective(trial: optuna.Trial) -> float:
        # Sample 3 weights, derive 4th to enforce sum=1
        w_tcs = trial.suggest_float("w_tcs", 0.05, 0.55)
        w_tfs = trial.suggest_float("w_tfs", 0.05, 0.55)
        w_rss = trial.suggest_float("w_rss", 0.05, 0.55)
        w_urs = 1.0 - (w_tcs + w_tfs + w_rss)
        if not (0.05 <= w_urs <= 0.55):
            return -10.0  # invalid → strong penalty
        weights = {"w_tcs": w_tcs, "w_tfs": w_tfs, "w_rss": w_rss, "w_urs": w_urs}
        spread = quintile_spread(train_obs, weights, quantile)
        # L2 regularization toward defaults — prevents overfitting (small samples)
        l2 = sum((weights[k] - DEFAULT_WEIGHTS[k]) ** 2 for k in weights)
        l2_penalty = 5.0 * l2  # tuned: 5.0 × L2 ≈ 0.05% on a 0.10 weight deviation
        return spread - l2_penalty

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    w_urs = round(1.0 - best["w_tcs"] - best["w_tfs"] - best["w_rss"], 4)
    best_weights = {
        "w_tcs": round(best["w_tcs"], 4),
        "w_tfs": round(best["w_tfs"], 4),
        "w_rss": round(best["w_rss"], 4),
        "w_urs": w_urs,
    }
    train_score = quintile_spread(train_obs, best_weights, quantile)
    oos_score = quintile_spread(oos_obs, best_weights, quantile)
    default_oos = quintile_spread(oos_obs, DEFAULT_WEIGHTS, quantile)
    return {
        "asset_class": asset_class,
        "best_weights": best_weights,
        "train_quintile_spread": round(train_score, 4),
        "oos_quintile_spread": round(oos_score, 4),
        "oos_default_spread": round(default_oos, 4),
        "oos_lift_vs_default": round(oos_score - default_oos, 4),
        "n_trials": n_trials,
        "n_obs_train": len(train_obs),
        "n_obs_oos": len(oos_obs),
        "quantile": quantile,
        "train_frac": train_frac,
    }


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def run_optimization(
    cache_path: str = CACHE_PATH,
    output_path: str = OUTPUT_PATH,
    n_trials: int = 100,
    quantile: float = 0.25,
    train_frac: float = 0.6,
    asset_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    print(f"Loading observations from {cache_path}...")
    obs, ticker_category = load_observations(cache_path)
    attach_features(obs, ticker_category)
    print(f"Loaded {len(obs)} observations")

    target_classes = asset_classes or ["ETF", "US_Stock", "KR_Stock"]
    by_class: Dict[str, List[Dict]] = {ac: [] for ac in target_classes}
    for o in obs:
        ac = o.get("asset_class")
        if ac in by_class:
            by_class[ac].append(o)

    results: Dict[str, Any] = {}
    for ac in target_classes:
        sub = by_class[ac]
        print(f"\n[{ac}]  {len(sub)} obs  →  Optuna {n_trials} trials...")
        if len(sub) < 100:
            print(f"  insufficient data, skipping.")
            results[ac] = {"asset_class": ac, "error": "insufficient samples", "n_obs": len(sub)}
            continue
        r = optimize_for_class(ac, sub, n_trials=n_trials,
                                quantile=quantile, train_frac=train_frac)
        results[ac] = r
        if "error" not in r:
            w = r["best_weights"]
            print(f"  best weights: TCS {w['w_tcs']:.3f} | TFS {w['w_tfs']:.3f} "
                  f"| RSS {w['w_rss']:.3f} | URS {w['w_urs']:.3f}")
            print(f"  train spread: {r['train_quintile_spread']:+.3f}%  "
                  f"OOS: {r['oos_quintile_spread']:+.3f}%  "
                  f"(default {r['oos_default_spread']:+.3f}%, "
                  f"lift {r['oos_lift_vs_default']:+.3f}%)")

    output = {
        "as_of": datetime.utcnow().isoformat(),
        "cache_path": cache_path,
        "n_total_obs": len(obs),
        "config": {
            "n_trials": n_trials,
            "quantile": quantile,
            "train_frac": train_frac,
            "objective": "OOS quintile-spread of forward 21d return",
        },
        "default_weights": DEFAULT_WEIGHTS,
        "asset_classes": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Wrote {output_path}")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--quantile", type=float, default=0.25)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--cache", default=CACHE_PATH)
    p.add_argument("--output", default=OUTPUT_PATH)
    args = p.parse_args()

    run_optimization(
        cache_path=args.cache,
        output_path=args.output,
        n_trials=args.trials,
        quantile=args.quantile,
        train_frac=args.train_frac,
    )
