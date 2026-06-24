# -*- coding: utf-8 -*-
"""metrics.py — individual ticker selection evaluation metrics.

5 metric families:
  1. Forward return distribution per pick     (direction)
  2. Sector-neutral alpha vs benchmark         (edge)
  3. Conviction rank quintile analysis         (PM ranking value)
  4. Win/Loss expected value (EV)              (asymmetry)
  5. Cross-cut attribution: sector / rank      (skill breakdown)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ─────────────────────────────────────────────────────────────────────
# Fix C — Sector ETF benchmark map (sector-neutral alpha)
# ─────────────────────────────────────────────────────────────────────

SECTOR_ETF_MAP: dict[str, str] = {
    "Financials":             "XLF",
    "Technology":             "XLK",
    "Healthcare":             "XLV",
    "Energy":                 "XLE",
    "Materials":              "XLB",
    "Industrials":            "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Utilities":              "XLU",
    "Real Estate":            "XLRE",
    "Communication Services": "XLC",
    # Cross-asset / Broad / regional → SPY as fallback
    "Broad":             "SPY",
    "Equity Broad":      "SPY",
    "Emerging Markets":  "EEM",
    "Intl Developed":    "EFA",
    "International":     "EFA",
    "China_ADR":         "FXI",
    "Korea":             "EWY",
    "Japan":             "EWJ",
    "Europe":            "VGK",
    "Currency_Vol":      "SPY",
    "Commodities":       "DBC",
    "Factor":            "SPY",
    "Thematic":          "SPY",
    "Real_Assets":       "SPY",
    "Other":             "SPY",
}


def sector_benchmark(sector: str) -> str:
    """Return the appropriate sector ETF for a ticker's sector. SPY fallback."""
    return SECTOR_ETF_MAP.get(sector or "Other", "SPY")


# ─────────────────────────────────────────────────────────────────────
# Fix D — Transaction cost adjustment
# ─────────────────────────────────────────────────────────────────────

TURNOVER_COST_BP = 10   # 10 basis points per round-trip
TURNOVER_COST = TURNOVER_COST_BP / 10000.0


# ─────────────────────────────────────────────────────────────────────
# Fix F — Bootstrap confidence interval
# ─────────────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], stat_fn=np.mean,
                  n_iter: int = 1000, ci: float = 95.0,
                  seed: int = 42) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for stat_fn on values."""
    if not values or len(values) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    samples = [stat_fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_iter)]
    lo = float(np.percentile(samples, (100 - ci) / 2))
    hi = float(np.percentile(samples, 100 - (100 - ci) / 2))
    return (lo, hi)


def fwd_return(price_df: pd.DataFrame, entry_date: pd.Timestamp, horizon: int) -> Optional[float]:
    """Forward return from entry_date over `horizon` trading days."""
    sub = price_df[price_df.index > entry_date]
    if len(sub) < horizon:
        return None
    entry_close = price_df[price_df.index <= entry_date]["Close"].iloc[-1]
    exit_close  = sub["Close"].iloc[horizon - 1]
    if pd.isna(entry_close) or pd.isna(exit_close) or entry_close <= 0:
        return None
    return float(exit_close / entry_close - 1)


def evaluate_pick(
    pick: dict,
    price_data: dict[str, pd.DataFrame],
    entry_date: pd.Timestamp,
    horizons: list[int] = (5, 21, 63),
    benchmark_ticker: str = "SPY",
    use_sector_etf: bool = True,
    apply_transaction_cost: bool = True,
) -> dict:
    """Compute forward returns + sector-neutral alpha for a single pick.

    Fix C: sector-ETF benchmark (XLF for Financials, XLK for Tech, etc.)
           — removes sector beta, isolates true selection alpha.
    Fix D: transaction cost (10bp per round trip) applied to net returns.
    """
    t = pick["ticker"]
    side = pick["side"]
    df = price_data.get(t)
    spy_df = price_data.get(benchmark_ticker)

    # Fix C: choose benchmark by sector when available
    sec = pick.get("sector") or "Other"
    sector_etf_ticker = sector_benchmark(sec) if use_sector_etf else benchmark_ticker
    sec_df = price_data.get(sector_etf_ticker)
    if sec_df is None or sec_df.empty:
        sec_df = spy_df

    out = {
        "ticker": t, "side": side, "entry_date": str(entry_date.date()),
        "rank": pick.get("rank"), "sector": pick.get("sector"),
        "asset_type": pick.get("asset_type"),
        "proxy_score": pick.get("proxy_score"),
        "classification": pick.get("classification"),
        "benchmark_etf": sector_etf_ticker,
    }

    if df is None or spy_df is None:
        for h in horizons:
            out[f"ret_{h}d"] = None
            out[f"spy_{h}d"] = None
            out[f"sec_{h}d"] = None
            out[f"alpha_{h}d"] = None
            out[f"alpha_spy_{h}d"] = None
            out[f"hit_{h}d"] = None
        return out

    cost = TURNOVER_COST if apply_transaction_cost else 0.0

    for h in horizons:
        ret = fwd_return(df, entry_date, h)
        spy_ret = fwd_return(spy_df, entry_date, h)
        sec_ret = fwd_return(sec_df, entry_date, h) if sec_df is not None else spy_ret

        out[f"ret_{h}d"] = ret
        out[f"spy_{h}d"] = spy_ret
        out[f"sec_{h}d"] = sec_ret
        # Legacy alias for backward compat (ticker_details.json uses "bench_*")
        out[f"bench_{h}d"] = sec_ret if use_sector_etf else spy_ret

        if ret is None or sec_ret is None:
            out[f"alpha_{h}d"] = None
            out[f"alpha_spy_{h}d"] = None
            out[f"hit_{h}d"] = None
        else:
            # Fix D: net of round-trip cost
            net_ret = ret - cost
            # Sector-neutral alpha (primary metric per Fix C)
            if side == "long":
                out[f"alpha_{h}d"]     = net_ret - sec_ret
                out[f"alpha_spy_{h}d"] = net_ret - (spy_ret or 0)
                out[f"hit_{h}d"]       = 1 if net_ret > sec_ret else 0
            else:  # short
                out[f"alpha_{h}d"]     = sec_ret - net_ret
                out[f"alpha_spy_{h}d"] = (spy_ret or 0) - net_ret
                out[f"hit_{h}d"]       = 1 if net_ret < sec_ret else 0
    return out


# ─────────────────────────────────────────────────────────────────────
# Metric 1: Forward return distribution
# ─────────────────────────────────────────────────────────────────────

def metric_forward_return(evals: list[dict], horizon: int = 21,
                          group_by: str = None) -> dict:
    """Compute distribution stats over a list of evaluation dicts.
    Fix F: Includes 95% bootstrap CI for hit_rate and mean."""
    def _stats(subset: list[dict]) -> dict:
        rets = [e[f"ret_{horizon}d"] for e in subset if e.get(f"ret_{horizon}d") is not None]
        hits = [e[f"hit_{horizon}d"] for e in subset if e.get(f"hit_{horizon}d") is not None]
        if not rets:
            return {"n": 0}
        arr = np.array(rets)
        mean_ci = bootstrap_ci(rets, np.mean)
        hit_ci  = bootstrap_ci(hits, np.mean) if hits else (float("nan"), float("nan"))
        return {
            "n": len(rets),
            "hit_rate": (sum(hits) / len(hits)) * 100 if hits else 0,
            "hit_ci_lo": hit_ci[0] * 100 if not np.isnan(hit_ci[0]) else None,
            "hit_ci_hi": hit_ci[1] * 100 if not np.isnan(hit_ci[1]) else None,
            "mean": float(arr.mean()) * 100,
            "mean_ci_lo": mean_ci[0] * 100 if not np.isnan(mean_ci[0]) else None,
            "mean_ci_hi": mean_ci[1] * 100 if not np.isnan(mean_ci[1]) else None,
            "median": float(np.median(arr)) * 100,
            "std": float(arr.std()) * 100,
            "p10": float(np.percentile(arr, 10)) * 100,
            "p25": float(np.percentile(arr, 25)) * 100,
            "p75": float(np.percentile(arr, 75)) * 100,
            "p90": float(np.percentile(arr, 90)) * 100,
            "min": float(arr.min()) * 100,
            "max": float(arr.max()) * 100,
        }

    if not group_by:
        return _stats(evals)
    out = {}
    groups = defaultdict(list)
    for e in evals:
        groups[e.get(group_by, "—")].append(e)
    for k, sub in groups.items():
        out[str(k)] = _stats(sub)
    return out


# ─────────────────────────────────────────────────────────────────────
# Metric 2: Sector-neutral alpha
# ─────────────────────────────────────────────────────────────────────

def metric_alpha(evals: list[dict], horizon: int = 21,
                 group_by: str = None) -> dict:
    """Sector-ETF-neutral alpha (Fix C). Includes bootstrap CI (Fix F)."""
    def _stats(subset: list[dict]) -> dict:
        alphas = [e[f"alpha_{horizon}d"] for e in subset if e.get(f"alpha_{horizon}d") is not None]
        if not alphas:
            return {"n": 0}
        arr = np.array(alphas)
        wins = arr[arr > 0]
        loss = arr[arr <= 0]
        mean_ci = bootstrap_ci(alphas, np.mean)
        return {
            "n": len(alphas),
            "mean_alpha": float(arr.mean()) * 100,
            "mean_alpha_ci_lo": mean_ci[0] * 100 if not np.isnan(mean_ci[0]) else None,
            "mean_alpha_ci_hi": mean_ci[1] * 100 if not np.isnan(mean_ci[1]) else None,
            "median_alpha": float(np.median(arr)) * 100,
            "win_rate": (len(wins) / len(arr)) * 100,
            "avg_win": float(wins.mean()) * 100 if len(wins) else 0,
            "avg_loss": float(loss.mean()) * 100 if len(loss) else 0,
            "win_loss_ratio": float(wins.mean() / abs(loss.mean())) if len(wins) and len(loss) and loss.mean() < 0 else 0,
            "profit_factor": float(wins.sum() / abs(loss.sum())) if len(wins) and len(loss) and loss.sum() < 0 else 0,
            "t_stat": float(arr.mean() / (arr.std() / np.sqrt(len(arr)))) if arr.std() > 0 else 0,
        }
    if not group_by:
        return _stats(evals)
    out = {}
    groups = defaultdict(list)
    for e in evals:
        groups[e.get(group_by, "—")].append(e)
    for k, sub in groups.items():
        out[str(k)] = _stats(sub)
    return out


# ─────────────────────────────────────────────────────────────────────
# Metric 3: Conviction rank quintile + IC
# ─────────────────────────────────────────────────────────────────────

def metric_rank_quintile(evals: list[dict], horizon: int = 21, n_quintile: int = 5) -> dict:
    """Group by rank quintile (rank 1-4=Q1, 5-8=Q2, ...) and report return/alpha stats."""
    quintiles = defaultdict(list)
    bucket_size = 20 // n_quintile
    for e in evals:
        rank = e.get("rank")
        if rank is None:
            continue
        q = min(n_quintile, (rank - 1) // bucket_size + 1)
        quintiles[f"Q{q}"].append(e)

    out = {}
    for q, picks in quintiles.items():
        rets = [e[f"ret_{horizon}d"] for e in picks if e.get(f"ret_{horizon}d") is not None]
        alphas = [e[f"alpha_{horizon}d"] for e in picks if e.get(f"alpha_{horizon}d") is not None]
        hits = [e[f"hit_{horizon}d"] for e in picks if e.get(f"hit_{horizon}d") is not None]
        if not rets:
            out[q] = {"n": 0}
            continue
        out[q] = {
            "n": len(rets),
            "rank_range": f"{(int(q[1:])-1)*bucket_size+1}-{int(q[1:])*bucket_size}",
            "hit_rate": (sum(hits) / len(hits)) * 100,
            "mean_ret": float(np.mean(rets)) * 100,
            "mean_alpha": float(np.mean(alphas)) * 100 if alphas else 0,
        }

    # Spearman IC: rank vs forward return
    pairs = [(e.get("rank"), e.get(f"ret_{horizon}d"), e.get("side"))
             for e in evals
             if e.get("rank") is not None and e.get(f"ret_{horizon}d") is not None]
    if pairs:
        ranks = np.array([p[0] for p in pairs])
        # For LONG, low rank → high return is expected (so we negate ret for correlation)
        targets = np.array([(-p[1] if p[2] == "long" else p[1]) for p in pairs])
        ic, _ = spearmanr(ranks, targets)
        out["_ic"] = float(ic) if not np.isnan(ic) else 0
    else:
        out["_ic"] = 0

    # Monotonicity check (Q1 > Q2 > Q3 > Q4 > Q5 for alpha)
    qkeys = sorted([k for k in out if k.startswith("Q")])
    alphas_seq = [out[k].get("mean_alpha", 0) for k in qkeys if out[k].get("n", 0) > 0]
    mono = 0.0
    if len(alphas_seq) >= 2:
        diffs = [1 if alphas_seq[i] > alphas_seq[i+1] else 0 for i in range(len(alphas_seq)-1)]
        mono = sum(diffs) / len(diffs)
    out["_monotonicity"] = mono

    return out


# ─────────────────────────────────────────────────────────────────────
# Trading Agent Layer Metrics (4 new families)
# ─────────────────────────────────────────────────────────────────────

def metric_entry_signal_edge(picks_with_signals: list[dict]) -> dict:
    """Compare forward-return performance by entry_signal type.

    Each pick has: signal dict, forward return @ pick date, and (for WAIT)
    forward return @ trigger date.

    Returns aggregate stats: BUY_NOW / WAIT (triggered) / WAIT (not triggered) / SKIP
    """
    buckets = {
        "BUY_NOW":         {"rets": [], "alphas": [], "n": 0},
        "WAIT_TRIGGERED":  {"rets": [], "alphas": [], "n": 0, "days_to_trigger": []},
        "WAIT_NEVER_FIRED":{"rets": [], "alphas": [], "n": 0},
        "SKIP":            {"rets": [], "alphas": [], "n": 0},
    }
    for p in picks_with_signals:
        sig = (p.get("signal") or {}).get("entry_signal", "WAIT")
        lc  = p.get("lifecycle") or {}
        ret = lc.get("realized_return")
        alpha = p.get("alpha_at_pick_21d")

        if sig == "BUY_NOW":
            buckets["BUY_NOW"]["n"] += 1
            if ret is not None: buckets["BUY_NOW"]["rets"].append(ret)
            if alpha is not None: buckets["BUY_NOW"]["alphas"].append(alpha)
        elif sig == "WAIT":
            if lc.get("days_to_trigger") is not None:
                b = buckets["WAIT_TRIGGERED"]
                b["n"] += 1
                b["days_to_trigger"].append(lc["days_to_trigger"])
                if ret is not None: b["rets"].append(ret)
                if alpha is not None: b["alphas"].append(alpha)
            else:
                b = buckets["WAIT_NEVER_FIRED"]
                b["n"] += 1
                # Counterfactual: what if we'd entered at pick date?
                if alpha is not None: b["alphas"].append(alpha)
        elif sig == "SKIP":
            buckets["SKIP"]["n"] += 1
            # Counterfactual return — would we have lost money?
            if alpha is not None: buckets["SKIP"]["alphas"].append(alpha)

    def _stats(b: dict) -> dict:
        out = {"n": b["n"]}
        if b.get("rets"):
            arr = np.array(b["rets"])
            out["mean_return"] = float(arr.mean()) * 100
            out["hit_rate"]    = (arr > 0).mean() * 100
            ci = bootstrap_ci(b["rets"], np.mean)
            out["mean_return_ci"] = [ci[0]*100, ci[1]*100] if not np.isnan(ci[0]) else None
        if b.get("alphas"):
            arr = np.array(b["alphas"])
            out["mean_alpha"] = float(arr.mean()) * 100
            ci = bootstrap_ci(b["alphas"], np.mean)
            out["mean_alpha_ci"] = [ci[0]*100, ci[1]*100] if not np.isnan(ci[0]) else None
        if b.get("days_to_trigger"):
            arr = np.array(b["days_to_trigger"])
            out["days_to_trigger_mean"] = float(arr.mean())
            out["days_to_trigger_median"] = float(np.median(arr))
        return out

    return {k: _stats(v) for k, v in buckets.items()}


def metric_exit_trigger_effectiveness(lifecycles: list[dict]) -> dict:
    """For each exit type, win-rate + realized return + delta vs hold."""
    by_type: dict[str, dict] = {}
    for lc in lifecycles:
        et = lc.get("exit_type")
        ret = lc.get("realized_return")
        bh = lc.get("buyhold_return")   # set by orchestrator
        if et is None or ret is None:
            continue
        b = by_type.setdefault(et, {"n": 0, "rets": [], "deltas": [], "days_held": []})
        b["n"] += 1
        b["rets"].append(ret)
        if bh is not None:
            b["deltas"].append(ret - bh)
        if lc.get("days_held") is not None:
            b["days_held"].append(lc["days_held"])

    out = {}
    total_lifecycles = sum(1 for lc in lifecycles if lc.get("exit_type"))
    for et, b in by_type.items():
        rets = np.array(b["rets"])
        deltas = np.array(b["deltas"]) if b["deltas"] else None
        days = np.array(b["days_held"]) if b["days_held"] else None
        out[et] = {
            "n": b["n"],
            "fire_pct": b["n"] / total_lifecycles * 100 if total_lifecycles else 0,
            "mean_return": float(rets.mean()) * 100,
            "win_rate": (rets > 0).mean() * 100,
            "delta_vs_hold_mean": float(deltas.mean()) * 100 if deltas is not None else None,
            "avg_days_held": float(days.mean()) if days is not None else None,
        }
    return out


def metric_trade_lifecycle_pnl(lifecycles: list[dict]) -> dict:
    """Aggregate trade-managed vs buy-and-hold P&L comparison."""
    managed = []
    buyhold = []
    for lc in lifecycles:
        r = lc.get("realized_return")
        bh = lc.get("buyhold_return")
        if r is not None and bh is not None:
            managed.append(r)
            buyhold.append(bh)
    if not managed:
        return {"n": 0}

    m = np.array(managed); b = np.array(buyhold)
    delta = m - b

    # Sharpe (no risk-free rate adjustment; per-trade)
    sharpe_m = float(m.mean() / m.std()) if m.std() > 0 else 0
    sharpe_b = float(b.mean() / b.std()) if b.std() > 0 else 0

    # Max DD on cumulative sum (proxy)
    def _max_dd(arr):
        cum = np.cumsum(arr)
        peak = np.maximum.accumulate(cum)
        return float((cum - peak).min())

    return {
        "n": len(managed),
        "managed_mean_return": float(m.mean()) * 100,
        "buyhold_mean_return": float(b.mean()) * 100,
        "delta_alpha":          float(delta.mean()) * 100,
        "delta_ci":             [bootstrap_ci(list(delta), np.mean)[0]*100,
                                 bootstrap_ci(list(delta), np.mean)[1]*100],
        "managed_win_rate":     (m > 0).mean() * 100,
        "buyhold_win_rate":     (b > 0).mean() * 100,
        "managed_sharpe":       sharpe_m,
        "buyhold_sharpe":       sharpe_b,
        "managed_max_dd":       _max_dd(m) * 100,
        "buyhold_max_dd":       _max_dd(b) * 100,
        "trading_helped_pct":   (delta > 0).mean() * 100,
    }


def metric_urgency_calibration(picks_with_signals: list[dict]) -> dict:
    """URGENT vs PATIENT label accuracy."""
    urg = {"URGENT": [], "NORMAL": [], "PATIENT": []}
    for p in picks_with_signals:
        u = (p.get("signal") or {}).get("urgency", "NORMAL")
        lc = p.get("lifecycle") or {}
        d2t = lc.get("days_to_trigger")     # for WAIT picks
        days_held = lc.get("days_held")
        early_move = p.get("return_first_3d")  # set by orchestrator
        if u in urg:
            urg[u].append({"d2t": d2t, "days_held": days_held, "early_move": early_move})

    out = {}
    for k, picks in urg.items():
        if not picks: out[k] = {"n": 0}; continue
        early_moves = [p["early_move"] for p in picks if p["early_move"] is not None]
        d2ts = [p["d2t"] for p in picks if p["d2t"] is not None]
        out[k] = {
            "n": len(picks),
            "pct_moved_3pct_in_3d": (np.abs(np.array(early_moves)) > 0.03).mean() * 100 if early_moves else None,
            "avg_days_to_trigger":  float(np.mean(d2ts)) if d2ts else None,
            "avg_early_move":       float(np.mean(early_moves)) * 100 if early_moves else None,
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Aggregate metrics report
# ─────────────────────────────────────────────────────────────────────

def full_report(evals_by_bucket: dict[str, list[dict]],
                horizon: int = 21) -> dict:
    """Run all 5 metrics for each bucket."""
    out = {}
    for bucket, evals in evals_by_bucket.items():
        out[bucket] = {
            "n_total": len(evals),
            "forward_return_dist": metric_forward_return(evals, horizon),
            "alpha_stats":         metric_alpha(evals, horizon),
            "rank_quintile":       metric_rank_quintile(evals, horizon),
            "by_sector":           metric_alpha(evals, horizon, group_by="sector"),
            "by_classification":   metric_alpha(evals, horizon, group_by="classification"),
        }
    return out
