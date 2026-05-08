###############################################################################
# Multi-Benchmark Validation Suite for P4 Signal
# ============================================================================
# Tests the user's core hypothesis:
#   "vs holding 100% equity, can adjusting bond/cash allocation produce
#    superior long-term cumulative performance (risk-adjusted)?"
#
# Framework follows AQR / Bridgewater / BlackRock institutional reporting:
#   compare strategy against MULTIPLE benchmarks (passive equity, matched-equity,
#   60/40 balanced, current default, T-bill) on RISK-ADJUSTED metrics.
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
from scipy import stats as sps


BENCHMARK_SUITE = [
    {"tag": "acwi100",  "name": "ACWI 100% (passive equity)",       "eq": 1.00, "bd": 0.00, "ch": 0.00, "role": "primary"},
    {"tag": "acwi75ch25","name": "ACWI 75 / Cash 25 (matched-equity)","eq": 0.75, "bd": 0.00, "ch": 0.25, "role": "matched"},
    {"tag": "acwi60agg40","name": "ACWI 60 / AGG 40 (institutional)","eq": 0.60, "bd": 0.40, "ch": 0.00, "role": "institutional"},
    {"tag": "acwi90ch10","name": "ACWI 90 / Cash 10 (current)",      "eq": 0.90, "bd": 0.00, "ch": 0.10, "role": "current"},
    {"tag": "tbill",    "name": "T-bill (risk-free)",                "eq": 0.00, "bd": 0.00, "ch": 1.00, "role": "riskfree"},
]

CASH_ANN_YIELD = 0.02


def _ann_stats(r: np.ndarray) -> dict:
    n = len(r)
    if n < 2:
        return {}
    cagr  = float((1 + r).prod() ** (12 / n) - 1)
    vol   = float(np.std(r, ddof=1) * np.sqrt(12))
    curve = np.cumprod(1 + r)
    peak  = np.maximum.accumulate(curve)
    dd    = curve / peak - 1
    max_dd = float(dd.min())

    sharpe = (cagr / vol) if vol > 1e-6 else float("nan")
    downside = r[r < 0]
    dn_std = float(np.std(downside, ddof=1) * np.sqrt(12)) if len(downside) > 1 else float("nan")
    sortino = (cagr / dn_std) if dn_std and dn_std > 1e-6 else float("nan")
    calmar = (cagr / abs(max_dd)) if max_dd < -1e-6 else float("nan")
    return {
        "ann_return": cagr, "ann_vol": vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd": max_dd,
    }


def _capture(strat: np.ndarray, market: np.ndarray) -> dict:
    """Geometric up/down capture vs the equity market (ACWI 100%)."""
    up = market > 0
    dn = market < 0
    def _g(s, b, mask):
        if not mask.any():
            return float("nan")
        n_m = mask.sum()
        s_a = float((1 + s[mask]).prod() ** (12 / n_m) - 1)
        b_a = float((1 + b[mask]).prod() ** (12 / n_m) - 1)
        if b_a == 0:
            return float("nan")
        return float(s_a / b_a)
    return {"up_capture": _g(strat, market, up), "dn_capture": _g(strat, market, dn)}


def _vs_benchmark(strat: np.ndarray, bench: np.ndarray) -> dict:
    excess = strat - bench
    alpha_ann = float(np.mean(excess) * 12)
    te = float(np.std(excess, ddof=1) * np.sqrt(12)) if len(excess) > 1 else float("nan")
    ir = (alpha_ann / te) if te and te > 1e-6 else float("nan")
    return {"alpha_ann": alpha_ann, "tracking_error": te, "information_ratio": ir}


def main():
    # P4 monthly returns (winner cache)
    ret_p4 = pd.read_csv("ai_pred_returns.csv", index_col=0, parse_dates=True)
    df = pd.read_csv("regime_dataset.csv", index_col=0, parse_dates=True)

    common_idx = ret_p4.index.intersection(df.index)
    eq = df.loc[common_idx, "fwd_ret"].values        # ACWI fwd 1M
    bd = df.loc[common_idx, "bond_fwd_ret"].values   # Global Agg fwd 1M
    ch = np.full_like(eq, CASH_ANN_YIELD / 12)       # cash drift
    p4 = ret_p4.loc[common_idx, "strategy_ret"].values

    # Market reference for capture ratios = ACWI 100%
    market = eq

    # ── Benchmark stats (each benchmark vs ACWI 100% as market reference) ──
    benchmarks = []
    for spec in BENCHMARK_SUITE:
        bench_ret = spec["eq"] * eq + spec["bd"] * bd + spec["ch"] * ch
        stats = _ann_stats(bench_ret)
        cap = _capture(bench_ret, market)
        benchmarks.append({
            "tag": spec["tag"], "name": spec["name"], "role": spec["role"],
            "weights": {"equity": spec["eq"], "bond": spec["bd"], "cash": spec["ch"]},
            **stats, **cap,
        })

    # ── P4 standalone stats ──
    p4_stats = _ann_stats(p4)
    p4_cap_vs_market = _capture(p4, market)
    p4_summary = {
        "name": "P4 Meta (winner)",
        "tag": "p4",
        **p4_stats,
        **p4_cap_vs_market,
    }

    # ── P4 vs each benchmark ──
    comparisons = []
    for spec, bench_obj in zip(BENCHMARK_SUITE, benchmarks):
        bench_ret = spec["eq"] * eq + spec["bd"] * bd + spec["ch"] * ch
        comp = _vs_benchmark(p4, bench_ret)
        cap_vs = _capture(p4, bench_ret)
        comparisons.append({
            "tag": spec["tag"], "name": spec["name"],
            **comp,
            "p4_sharpe_minus_bench":  float(p4_stats["sharpe"]  - bench_obj["sharpe"]),
            "p4_sortino_minus_bench": float(p4_stats["sortino"] - bench_obj["sortino"]) if not np.isnan(bench_obj["sortino"]) else None,
            "p4_calmar_minus_bench":  float(p4_stats["calmar"]  - bench_obj["calmar"])  if not np.isnan(bench_obj["calmar"])  else None,
            "p4_dd_minus_bench":      float(p4_stats["max_dd"]  - bench_obj["max_dd"]),
            "p4_up_capture_vs_bench": cap_vs["up_capture"],
            "p4_dn_capture_vs_bench": cap_vs["dn_capture"],
        })

    # ── Hypothesis verdict ──
    # User's hypothesis: "P4가 100% 보유 대비 risk-adjusted 우월"
    # Pass = Sharpe / Sortino / Calmar 모두 우월 OR 최소 2/3
    primary_bench = next(b for b in benchmarks if b["role"] == "primary")
    verdicts = {
        "vs_acwi100": {
            "sharpe_better":  bool(p4_stats["sharpe"] > primary_bench["sharpe"]),
            "sortino_better": bool(p4_stats["sortino"] > primary_bench["sortino"]) if primary_bench["sortino"] == primary_bench["sortino"] else None,
            "calmar_better":  bool(p4_stats["calmar"] > primary_bench["calmar"]) if primary_bench["calmar"] == primary_bench["calmar"] else None,
            "max_dd_better":  bool(p4_stats["max_dd"] > primary_bench["max_dd"]),  # less negative = better
        }
    }
    n_pass = sum(1 for v in verdicts["vs_acwi100"].values() if v is True)
    n_total = sum(1 for v in verdicts["vs_acwi100"].values() if v is not None)
    verdicts["vs_acwi100"]["overall_pass"] = n_pass >= 3
    verdicts["vs_acwi100"]["score"] = f"{n_pass}/{n_total}"

    out = {
        "n_oos_months": int(len(common_idx)),
        "period_start": str(common_idx.min().date()),
        "period_end":   str(common_idx.max().date()),
        "p4_strategy":  p4_summary,
        "benchmarks":   benchmarks,
        "comparisons":  comparisons,
        "hypothesis_verdict": verdicts,
        "cash_ann_yield": CASH_ANN_YIELD,
    }

    with open("ai_pred_benchmarks.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print("[save] ai_pred_benchmarks.json")

    # ── Monthly returns CSV (for cumulative chart) ──
    rows = {"date": [d.date().isoformat() for d in common_idx], "p4": p4.tolist()}
    for spec in BENCHMARK_SUITE:
        bench_ret = spec["eq"] * eq + spec["bd"] * bd + spec["ch"] * ch
        rows[spec["tag"]] = bench_ret.tolist()
    pd.DataFrame(rows).to_csv("ai_pred_benchmarks_monthly.csv", index=False)
    print(f"[save] ai_pred_benchmarks_monthly.csv ({len(common_idx)} rows)")

    # ── Console pretty print ──
    print("\n" + "=" * 110)
    print(" Multi-Benchmark Validation Suite — P4 vs 5 Reference Portfolios")
    print("=" * 110)
    print(f" Period: {out['period_start']} → {out['period_end']} ({out['n_oos_months']} months)")
    print()
    hdr = f"{'Strategy':<32}{'CAGR':>9}{'Vol':>9}{'Sharpe':>9}{'Sortino':>9}{'Calmar':>9}{'MaxDD':>9}{'UpCap':>8}{'DnCap':>8}"
    print(hdr); print("-" * len(hdr))

    p4_row = (f"{'P4 Meta (winner)':<32}"
              f"{p4_stats['ann_return']*100:>8.2f}%"
              f"{p4_stats['ann_vol']*100:>8.2f}%"
              f"{p4_stats['sharpe']:>9.2f}"
              f"{p4_stats['sortino']:>9.2f}"
              f"{p4_stats['calmar']:>9.2f}"
              f"{p4_stats['max_dd']*100:>+8.1f}%"
              f"{p4_cap_vs_market['up_capture']:>8.2f}"
              f"{p4_cap_vs_market['dn_capture']:>8.2f}")
    print(p4_row)
    for b in benchmarks:
        line = (f"{b['name']:<32}"
                f"{b['ann_return']*100:>8.2f}%"
                f"{b['ann_vol']*100:>8.2f}%"
                f"{b['sharpe']:>9.2f}"
                f"{b['sortino']:>9.2f}"
                f"{b['calmar']:>9.2f}"
                f"{b['max_dd']*100:>+8.1f}%"
                f"{b['up_capture']:>8.2f}"
                f"{b['dn_capture']:>8.2f}")
        print(line)

    print("\n" + "-" * 110)
    print(" P4 vs Each Benchmark (alpha / IR / Sharpe-diff / DD-diff)")
    print("-" * 110)
    hdr2 = f"{'Benchmark':<32}{'Alpha':>10}{'TE':>8}{'IR':>8}{'ΔSharpe':>10}{'ΔMaxDD':>10}{'P4 UpCap':>10}{'P4 DnCap':>10}"
    print(hdr2); print("-" * len(hdr2))
    for c in comparisons:
        print(f"{c['name']:<32}"
              f"{c['alpha_ann']*100:>+9.2f}%"
              f"{c['tracking_error']*100:>+7.2f}%"
              f"{c['information_ratio']:>+8.2f}"
              f"{c['p4_sharpe_minus_bench']:>+10.2f}"
              f"{c['p4_dd_minus_bench']*100:>+9.1f}%"
              f"{c['p4_up_capture_vs_bench']:>10.2f}"
              f"{c['p4_dn_capture_vs_bench']:>10.2f}")

    v = verdicts["vs_acwi100"]
    print("\n" + "=" * 110)
    print(f" HYPOTHESIS VERDICT vs ACWI 100% — {v['score']} risk-adjusted dimensions outperform")
    print("=" * 110)
    print(f"   Sharpe better:  {v['sharpe_better']}")
    print(f"   Sortino better: {v['sortino_better']}")
    print(f"   Calmar better:  {v['calmar_better']}")
    print(f"   MaxDD better:   {v['max_dd_better']}")
    print(f"   → Overall pass (≥3/4): {v['overall_pass']}")


if __name__ == "__main__":
    main()
