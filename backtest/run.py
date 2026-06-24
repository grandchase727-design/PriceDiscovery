# -*- coding: utf-8 -*-
"""run.py — orchestrator: deterministic-proxy backtest end-to-end.

Workflow:
  1. Load universe + historical prices (cached)
  2. For each Friday from 2026-01-02 to 2026-06-05 (~22 weeks):
       - compute proxy scores for all tickers
       - select top-20 LONG/SHORT × stock/ETF (4 buckets)
       - record picks with entry_date metadata
  3. For each pick, compute forward returns at 5d/21d/63d + sector-neutral alpha
  4. Aggregate via metrics.py
  5. Save results to backtest/results.json + print summary
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data import load_universe, download_prices, get_trading_fridays
from backtest.proxy import compute_proxy_score_one_date, select_picks_one_date
from backtest.metrics import (
    evaluate_pick, full_report,
    metric_entry_signal_edge, metric_exit_trigger_effectiveness,
    metric_trade_lifecycle_pnl, metric_urgency_calibration,
)
from backtest.trading_proxy import trading_proxy_signal, simulate_trade_lifecycle


def run_backtest(
    year: int = 2026,
    end_date: str = "",
    top_n: int = 20,
    max_per_sector: int = 5,
    output_path: str = "backtest/results.json",
    min_history_days: int = 120,
    apply_bias_fixes: bool = True,
) -> dict:
    # Default end_date = yesterday (so we always have full last-day prices)
    if not end_date:
        from datetime import datetime, timedelta
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    """Run deterministic-proxy backtest with bias mitigations.

    Bias fixes applied:
      Fix A: IPO/history filter (min_history_days)
      Fix C: Sector-ETF benchmark (XLF, XLK, XLV, etc.) instead of SPY alone
      Fix D: Transaction cost (10bp per round trip) deducted from returns
      Fix F: Bootstrap 95% CI on key metrics
    """
    t0 = time.time()

    print("=" * 70)
    print(f"PM Agent Backtest — Deterministic Proxy ({year} YTD)")
    if apply_bias_fixes:
        print("  Bias mitigations: Fix A (IPO filter) + C (Sector ETF) + D (cost) + F (CI)")
    else:
        print("  ⚠ BIASED MODE: no fixes — for comparison purposes only")
    print("=" * 70)

    universe = load_universe()
    if "SPY" not in {u["ticker"] for u in universe}:
        universe.append({"ticker": "SPY", "name": "SPDR S&P 500 ETF",
                         "sector": "Broad", "asset_type": "ETF"})
    tickers = [u["ticker"] for u in universe]
    print(f"\nUniverse: {len(universe)} tickers")

    print("\nLoading prices (cached)…")
    data = download_prices(tickers, start="2025-08-01", end=end_date)
    print(f"Loaded data for {len(data)} / {len(universe)} tickers")
    universe = [u for u in universe if u["ticker"] in data]

    # ── 2. Selection per Friday ──
    fridays = get_trading_fridays(year, end_date)
    print(f"\nEvaluation Fridays: {len(fridays)} (from {fridays[0].date()} to {fridays[-1].date()})")

    # Effective history filter (Fix A): 120 days if fixes enabled, else looser
    hist_filter = min_history_days if apply_bias_fixes else 60

    all_picks: list[dict] = []
    weekly_summary = []
    for i, friday in enumerate(fridays):
        scored = compute_proxy_score_one_date(data, universe, friday,
                                              min_history_days=hist_filter)
        if scored is None:
            print(f"  W{i:2} {friday.date()}: insufficient data — skip")
            continue
        picks = select_picks_one_date(scored, top_n=top_n, max_per_sector=max_per_sector)
        for bucket, plist in picks.items():
            for p in plist:
                p["bucket"] = bucket
                p["entry_date"] = friday.isoformat()
                all_picks.append(p)
        sizes = {b: len(picks[b]) for b in picks}
        weekly_summary.append({"week": i, "date": friday.isoformat(), "picks": sizes})
        if i % 4 == 0 or i == len(fridays) - 1:
            print(f"  W{i:2} {friday.date()}  picks: " +
                  " | ".join(f"{b}={n}" for b, n in sizes.items()))

    print(f"\nTotal picks generated: {len(all_picks)}")

    # ── 3. Forward returns + alpha ──
    label = "sector-ETF alpha + 10bp cost" if apply_bias_fixes else "SPY alpha (no fixes)"
    print(f"\nComputing forward returns + {label}…")
    evaluations = []
    for p in all_picks:
        ed = pd.Timestamp(p["entry_date"])
        ev = evaluate_pick(p, data, ed,
                           use_sector_etf=apply_bias_fixes,
                           apply_transaction_cost=apply_bias_fixes)
        evaluations.append(ev)
    print(f"Evaluated: {len(evaluations)} picks")

    # ── 4. Trading Layer (Architecture C — Phase 1) ──
    # For each pick: generate deterministic trading proxy signal
    # + simulate trade lifecycle (entry/exit triggers)
    print("\nComputing Trading proxy signals + lifecycles…")
    HORIZON_DAYS = {"tactical": 5, "core": 21, "strategic": 63}
    picks_with_signals: list[dict] = []
    lifecycles_by_horizon: dict[str, list[dict]] = {"tactical": [], "core": [], "strategic": []}

    for p, e in zip(all_picks, evaluations):
        t = p["ticker"]
        df = data.get(t)
        if df is None: continue
        # Use Core horizon as default for proxy (deterministic doesn't have explicit horizon)
        # We'll simulate the same pick across all 3 horizons.
        side = p["side"]
        cls_proxy = p.get("classification") or "LONG_OK" if side == "long" else "SHORT_OK"

        for horizon_label in ("tactical", "core", "strategic"):
            entry_date = pd.Timestamp(p["entry_date"])
            history_up_to = df[df.index <= entry_date]
            if len(history_up_to) < 50: continue

            sig = trading_proxy_signal(history_up_to, side, horizon_label, cls_proxy)
            lc = simulate_trade_lifecycle(df, entry_date, sig, side,
                                          horizon_days=HORIZON_DAYS[horizon_label])
            # Attach buy-and-hold return for comparison
            lc["buyhold_return"] = e.get(f"ret_{HORIZON_DAYS[horizon_label]}d")

            # Partial BH return: (latest_available_price / entry_price) - 1
            # Always populated (even when full-horizon return is in-flight).
            # For in-flight cohorts: shows current MTM with days_elapsed indicator.
            fwd = df[df.index > entry_date]
            if len(fwd) > 0:
                entry_hist = df[df.index <= entry_date]
                if len(entry_hist) > 0:
                    entry_close = float(entry_hist["Close"].iloc[-1])
                    last_close = float(fwd["Close"].iloc[-1])
                    if entry_close > 0:
                        partial_r = (last_close / entry_close - 1)
                        if side == "short":
                            partial_r = -partial_r
                        lc["buyhold_partial_return"] = partial_r
                        lc["buyhold_days_elapsed"] = len(fwd)

            # Early move for urgency calibration (first 3 days)
            early_ret = e.get("ret_5d")  # closest available
            lc["return_first_3d"] = early_ret * 0.6 if early_ret else None  # rough estimate

            ps = {
                "ticker": t, "name": p.get("name", ""),
                "bucket": p["bucket"], "side": side,
                "horizon": horizon_label,
                "entry_date": p["entry_date"][:10],
                "signal": sig, "lifecycle": lc,
                "alpha_at_pick_21d": e.get("alpha_21d"),
                "return_first_3d": lc["return_first_3d"],
            }
            picks_with_signals.append(ps)
            lifecycles_by_horizon[horizon_label].append(lc)

    print(f"Trading lifecycles simulated: {len(picks_with_signals)} (across 3 horizons)")

    # ── 5. Metrics ──
    print("\nComputing metrics…")
    evals_by_bucket: dict[str, list[dict]] = {}
    for e, p in zip(evaluations, all_picks):
        evals_by_bucket.setdefault(p["bucket"], []).append(e)

    report = {
        "as_of_run": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "year": year,
        "end_date": end_date,
        "weekly_summary": weekly_summary,
        "n_picks": len(all_picks),
        "n_evaluations": len(evaluations),
        "horizon_metrics": {},
    }
    for horizon in (5, 21, 63):
        report["horizon_metrics"][f"h{horizon}d"] = full_report(evals_by_bucket, horizon)

    # Trading Layer metrics (per-horizon)
    report["trading_metrics"] = {}
    for h_label in ("tactical", "core", "strategic"):
        h_picks = [p for p in picks_with_signals if p["horizon"] == h_label]
        h_lcs   = lifecycles_by_horizon[h_label]
        report["trading_metrics"][h_label] = {
            "n_picks": len(h_picks),
            "entry_signal_edge":         metric_entry_signal_edge(h_picks),
            "exit_trigger_effectiveness": metric_exit_trigger_effectiveness(h_lcs),
            "trade_lifecycle":            metric_trade_lifecycle_pnl(h_lcs),
            "urgency_calibration":        metric_urgency_calibration(h_picks),
        }

    # ── 4b. Lifecycle Timeline data (for PositionTimelineHeatmap viz) ──
    # Compact per-pick records: enables timeline rendering of Managed vs
    # Buy-and-Hold portfolio composition per weekly cohort.
    report["trading_lifecycles_compact"] = {}
    for h_label in ("tactical", "core", "strategic"):
        h_picks = [p for p in picks_with_signals if p["horizon"] == h_label]
        by_bucket: dict[str, list] = {"long_stocks": [], "long_etfs": [],
                                      "short_stocks": [], "short_etfs": []}
        for p in h_picks:
            lc = p.get("lifecycle") or {}
            rec = {
                "t":  p.get("ticker"),
                "n":  (p.get("name") or "")[:40],           # ticker display name
                "d":  (p.get("entry_date") or "")[:10],     # weekly cohort date
                "sig":p.get("signal", {}).get("entry_signal", "?")
                       if isinstance(p.get("signal"), dict) else "?",
                # Managed outcome
                "mst":  lc.get("state"),                    # final state
                "mex":  lc.get("exit_type"),                # exit reason
                "mr":   lc.get("realized_return"),          # managed P&L
                "mdh":  lc.get("days_held"),                # days in trade
                "mdt":  lc.get("days_to_trigger"),          # WAIT→entry delay
                # Buy-and-hold benchmark
                "bh":   lc.get("buyhold_return"),           # full-horizon BH (null if in-flight)
                "bhp":  lc.get("buyhold_partial_return"),   # partial BH (MTM at latest price)
                "bhd":  lc.get("buyhold_days_elapsed"),     # trading days since entry
            }
            b = p.get("bucket")
            if b in by_bucket:
                by_bucket[b].append(rec)
        # Sort each bucket by date
        for b, lst in by_bucket.items():
            lst.sort(key=lambda r: (r["d"], r["t"]))
        report["trading_lifecycles_compact"][h_label] = by_bucket

    # ── 5. Save ──
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    print(f"\nResults saved → {output_path}")

    # ── 5b. Per-ticker drilldown (Task 4) ──
    ticker_details: dict[str, dict] = {}
    for ev, p in zip(evaluations, all_picks):
        t = p["ticker"]
        rec = ticker_details.setdefault(t, {
            "ticker": t, "name": p.get("name", ""), "sector": p.get("sector", ""),
            "asset_type": p.get("asset_type"),
            "appearances": [],
        })
        rec["appearances"].append({
            "entry_date": p["entry_date"][:10],
            "bucket": p["bucket"],
            "side": p["side"],
            "rank": p.get("rank"),
            "proxy_score": p.get("proxy_score"),
            "classification": p.get("classification"),
            "ret_5d":   ev.get("ret_5d"),
            "ret_21d":  ev.get("ret_21d"),
            "ret_63d":  ev.get("ret_63d"),
            "bench_5d":  ev.get("bench_5d"),
            "bench_21d": ev.get("bench_21d"),
            "bench_63d": ev.get("bench_63d"),
            "alpha_5d":  ev.get("alpha_5d"),
            "alpha_21d": ev.get("alpha_21d"),
            "alpha_63d": ev.get("alpha_63d"),
            "hit_5d":   ev.get("hit_5d"),
            "hit_21d":  ev.get("hit_21d"),
            "hit_63d":  ev.get("hit_63d"),
        })

    # Compute per-ticker aggregates
    import statistics as _st
    for t, rec in ticker_details.items():
        apps = rec["appearances"]
        for h in (5, 21, 63):
            alphas = [a[f"alpha_{h}d"] for a in apps if a.get(f"alpha_{h}d") is not None]
            hits   = [a[f"hit_{h}d"]   for a in apps if a.get(f"hit_{h}d")   is not None]
            rets   = [a[f"ret_{h}d"]   for a in apps if a.get(f"ret_{h}d")   is not None]
            rec[f"mean_alpha_{h}d"] = (sum(alphas) / len(alphas)) if alphas else None
            rec[f"win_rate_{h}d"]   = (sum(hits)   / len(hits)) * 100 if hits else None
            rec[f"mean_ret_{h}d"]   = (sum(rets)   / len(rets)) if rets else None
        ranks = [a["rank"] for a in apps if a.get("rank")]
        rec["n_appearances"] = len(apps)
        rec["avg_rank"]      = (sum(ranks) / len(ranks)) if ranks else None
        rec["buckets"]       = list(set(a["bucket"] for a in apps))
        rec["sides"]         = list(set(a["side"] for a in apps))

    Path("backtest/ticker_details.json").write_text(
        json.dumps({"as_of_run": report["as_of_run"], "tickers": ticker_details},
                   indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Per-ticker drilldown saved → backtest/ticker_details.json ({len(ticker_details)} unique tickers)")

    # ── 5c. Top/Worst tickers per bucket for dashboard summary ──
    bucket_rankings: dict[str, dict] = {}
    for bucket in ["long_stocks", "long_etfs", "short_stocks", "short_etfs"]:
        # All tickers that appeared in this bucket
        cand = []
        for t, rec in ticker_details.items():
            apps_in_bucket = [a for a in rec["appearances"] if a["bucket"] == bucket]
            if not apps_in_bucket:
                continue
            alphas = [a["alpha_21d"] for a in apps_in_bucket if a.get("alpha_21d") is not None]
            hits   = [a["hit_21d"]   for a in apps_in_bucket if a.get("hit_21d")   is not None]
            if not alphas:
                continue
            cand.append({
                "ticker": t, "name": rec.get("name", "")[:30], "sector": rec.get("sector", ""),
                "n": len(apps_in_bucket),
                "mean_alpha_21d": sum(alphas) / len(alphas),
                "win_rate_21d":   (sum(hits) / len(hits)) * 100 if hits else 0,
                "avg_rank":       sum(a["rank"] for a in apps_in_bucket) / len(apps_in_bucket),
            })
        # Min appearances filter to avoid noise from single-pick outliers
        cand = [c for c in cand if c["n"] >= 2]
        cand.sort(key=lambda x: -x["mean_alpha_21d"])
        bucket_rankings[bucket] = {
            "top": cand[:10],
            "worst": cand[-10:][::-1] if len(cand) >= 10 else cand[::-1],
        }
    Path("backtest/ticker_rankings.json").write_text(
        json.dumps(bucket_rankings, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Per-bucket top/worst tickers saved → backtest/ticker_rankings.json")

    # ── 6. Print headline summary ──
    print("\n" + "=" * 70)
    print("HEADLINE SUMMARY  (21d horizon, sector-neutral alpha)")
    print("=" * 70)
    for bucket in ["long_stocks", "long_etfs", "short_stocks", "short_etfs"]:
        m = report["horizon_metrics"]["h21d"][bucket]
        a = m["alpha_stats"]
        f = m["forward_return_dist"]
        rq = m["rank_quintile"]
        if a.get("n", 0) == 0:
            print(f"\n{bucket}: no data")
            continue
        print(f"\n{bucket}  (n={m['n_total']})")
        print(f"  Forward return : hit_rate={f.get('hit_rate',0):.1f}%  mean={f.get('mean',0):+.2f}%  median={f.get('median',0):+.2f}%")
        print(f"  Alpha vs SPY   : mean={a['mean_alpha']:+.2f}%  win%={a['win_rate']:.1f}%  W/L={a['win_loss_ratio']:.2f}  PF={a['profit_factor']:.2f}  t={a['t_stat']:+.2f}")
        print(f"  Rank quintile  : IC={rq.get('_ic',0):+.3f}  monotonicity={rq.get('_monotonicity',0)*100:.0f}%")
        for q in ["Q1","Q2","Q3","Q4","Q5"]:
            if q in rq and rq[q].get("n", 0):
                print(f"    {q} (rank {rq[q]['rank_range']:6}): n={rq[q]['n']:>4}  hit={rq[q]['hit_rate']:5.1f}%  ret={rq[q]['mean_ret']:+5.2f}%  alpha={rq[q]['mean_alpha']:+5.2f}%")

    # Trading metrics summary
    print("\n" + "=" * 70)
    print("TRADING LAYER METRICS (deterministic proxy)")
    print("=" * 70)
    for h_label in ("tactical", "core", "strategic"):
        tm = report["trading_metrics"][h_label]
        print(f"\n--- {h_label.upper()} horizon (n={tm['n_picks']}) ---")
        # Entry signal edge
        print("  Entry Signal Edge:")
        for sig_type, stats in tm["entry_signal_edge"].items():
            n = stats.get("n", 0)
            if n == 0: continue
            mr = stats.get("mean_return", 0) if "mean_return" in stats else None
            ma = stats.get("mean_alpha", 0) if "mean_alpha" in stats else None
            ddt = stats.get("days_to_trigger_mean")
            tail = f" · d2trigger {ddt:.1f}d" if ddt else ""
            ret_str = f"return {mr:+.2f}%" if mr is not None else ""
            print(f"    {sig_type:18} n={n:>4}  {ret_str}{tail}")
        # Exit trigger effectiveness
        print("  Exit Trigger Effectiveness:")
        for et, s in tm["exit_trigger_effectiveness"].items():
            d = s.get("delta_vs_hold_mean", 0) or 0
            print(f"    {et:15} fire {s['fire_pct']:5.1f}% (n={s['n']:>4})  ret {s['mean_return']:+5.2f}%  win {s['win_rate']:5.1f}%  Δvs hold {d:+5.2f}%")
        # Lifecycle PnL
        tl = tm["trade_lifecycle"]
        if tl.get("n"):
            print(f"  Lifecycle P&L (n={tl['n']}):")
            print(f"    Managed avg : {tl['managed_mean_return']:+.2f}%  vs B&H: {tl['buyhold_mean_return']:+.2f}%  Δα: {tl['delta_alpha']:+.2f}%")
            print(f"    Managed Sharpe: {tl['managed_sharpe']:.2f}  vs B&H: {tl['buyhold_sharpe']:.2f}")
            print(f"    Managed Max DD: {tl['managed_max_dd']:+.2f}%  vs B&H: {tl['buyhold_max_dd']:+.2f}%")
            print(f"    'Trading helped' picks: {tl['trading_helped_pct']:.1f}%")
        # Urgency calibration
        print("  Urgency Calibration:")
        for u, s in tm["urgency_calibration"].items():
            if s.get("n", 0) == 0: continue
            p3 = s.get("pct_moved_3pct_in_3d")
            d2t = s.get("avg_days_to_trigger")
            p3s = f"{p3:.0f}% moved ≥3% in 3d" if p3 is not None else ""
            d2ts = f" · avg trigger {d2t:.1f}d" if d2t else ""
            print(f"    {u:10} n={s['n']:>4}  {p3s}{d2ts}")

    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    return report


def run_comparison():
    """Run both biased + de-biased versions, print side-by-side delta."""
    print("\n\n" + "█" * 70)
    print("  BIASED BASELINE  (for comparison only — known to overestimate)")
    print("█" * 70)
    biased = run_backtest(apply_bias_fixes=False,
                          output_path="backtest/results_biased.json")

    print("\n\n" + "█" * 70)
    print("  DE-BIASED  (Fix A + C + D + F applied)")
    print("█" * 70)
    fixed = run_backtest(apply_bias_fixes=True,
                         output_path="backtest/results.json")

    # ── Side-by-side delta table ──
    print("\n\n" + "═" * 90)
    print("BIAS IMPACT — Side-by-side comparison (21d horizon)")
    print("═" * 90)
    print(f"{'Bucket':18} {'Metric':16} {'Biased':>10} {'De-biased':>12} {'Δ (impact)':>14}")
    print("-" * 90)
    for bucket in ["long_stocks", "long_etfs", "short_stocks", "short_etfs"]:
        b = biased.get("horizon_metrics", {}).get("h21d", {}).get(bucket, {})
        f = fixed.get("horizon_metrics", {}).get("h21d", {}).get(bucket, {})
        if not b or not f: continue
        bp_n = b.get("n_total", 0); fp_n = f.get("n_total", 0)
        print(f"\n{bucket}  (biased n={bp_n}, fixed n={fp_n})")
        for label, key, sub in [
            ("Hit Rate",         "hit_rate",   "forward_return_dist"),
            ("Mean Alpha",       "mean_alpha", "alpha_stats"),
            ("Win Rate",         "win_rate",   "alpha_stats"),
            ("Profit Factor",    "profit_factor","alpha_stats"),
            ("t-stat",           "t_stat",     "alpha_stats"),
        ]:
            bv = b.get(sub, {}).get(key)
            fv = f.get(sub, {}).get(key)
            if bv is None or fv is None: continue
            delta = fv - bv
            sig = "⬇ over-stated" if (bucket.startswith("long") and delta < -0.1) else \
                  "⬆ under-stated" if (bucket.startswith("short") and delta > 0.1) else \
                  ""
            print(f"  {label:14}             {bv:>9.2f}  {fv:>11.2f}  {delta:>+8.2f}    {sig}")
    return biased, fixed


if __name__ == "__main__":
    import sys
    if "--compare" in sys.argv:
        run_comparison()
    elif "--biased" in sys.argv:
        run_backtest(apply_bias_fixes=False, output_path="backtest/results_biased.json")
    else:
        run_backtest()   # default: de-biased
