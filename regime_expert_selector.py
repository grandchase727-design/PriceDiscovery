###############################################################################
# Step 2 (Plan B): Regime-Conditional Expert Blending — Mixture-of-Experts
# ============================================================================
# Per diagnostic, each regime has a distinctly-best variant:
#   BULL → P4_meta           (cleanest in bull, override→BASE cuts bad bets)
#   BASE → P0_baseline       (variants tied; use simplest reference)
#   BEAR → VIXFREE_w25_t030  (best defensive alpha in bears)
#
# Mixture design:
#   (1) Train each expert independently → per-month allocation recommendation
#   (2) Train a GATE model (P0 primary) → per-month P(BULL|X), P(BASE|X), P(BEAR|X)
#   (3) Blend:
#         SOFT: w_final(t) = Σ_r gate_proba[r](t) × w_expert[r](t)
#         HARD: w_final(t) = w_expert[argmax(gate_proba)](t)
#   (4) Backtest vs ACWI 90 / Cash 10 benchmark AND vs P4_meta solo.
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from ml_signal_engine import (VariantConfig, run_variant, load_dataset,
                              blend_allocation, ALLOCATION_GRID,
                              BENCHMARK_WEIGHTS, BENCHMARK_LABEL, REGIMES)


# ─────────────────────────────────────────────────────────────────────────────
# Expert assignment (based on diagnostic bp/month alpha per regime)
# ─────────────────────────────────────────────────────────────────────────────
EXPERT_CONFIGS = {
    "BULL": VariantConfig(
        name="expert_BULL_P4meta",
        cv_mode="walkforward",
        class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 3.0},
        bull_threshold=0.25, use_meta=True,
    ),
    "BASE": VariantConfig(
        name="expert_BASE_P0",
        cv_mode="walkforward", class_weight="balanced",
    ),
    "BEAR": VariantConfig(
        name="expert_BEAR_VIXFREE",
        dataset_path="regime_dataset_novix.csv",
        cv_mode="walkforward",
        class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 2.5},
        bull_threshold=0.30,
    ),
}

# Gate model — provides regime probabilities that drive the routing
GATE_CONFIG = VariantConfig(
    name="gate_P0", cv_mode="walkforward", class_weight="balanced",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _expert_allocations(out, dataset_path: str) -> pd.DataFrame:
    """Per-month (w_eq, w_bd, w_cash) this expert would recommend."""
    proba = out["proba_df"]
    if "final_regime" in proba.columns:
        weights = [ALLOCATION_GRID[r] for r in proba["final_regime"].values]
    else:
        weights = [blend_allocation(proba.iloc[i]) for i in range(len(proba))]
    return pd.DataFrame({
        "w_equity": [w["equity"] for w in weights],
        "w_bond":   [w["bond"]   for w in weights],
        "w_cash":   [w["cash"]   for w in weights],
    }, index=proba.index)


def _ann_stats(r: np.ndarray) -> dict:
    ann = 12
    m = float(np.mean(r) * ann)
    s = float(np.std(r, ddof=1) * np.sqrt(ann))
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(curve)
    dd = float((curve / peak - 1).min())
    return {"ann_ret": m, "ann_vol": s,
            "sharpe": m / s if s > 0 else float("nan"),
            "max_dd": dd}


def _run_and_score(w_eq, w_bd, w_ch, eq, bd, ch, tc_bps=5.0):
    bm = BENCHMARK_WEIGHTS
    w_prev = np.vstack([np.array([bm["equity"], bm["bond"], bm["cash"]])[None, :],
                        np.stack([w_eq, w_bd, w_ch], axis=1)[:-1]])
    w_curr = np.stack([w_eq, w_bd, w_ch], axis=1)
    turn = np.abs(w_curr - w_prev).sum(axis=1)
    tc = turn * (tc_bps / 10000.0)
    strat = w_eq * eq + w_bd * bd + w_ch * ch - tc
    bench = bm["equity"] * eq + bm["bond"] * bd + bm["cash"] * ch
    s = _ann_stats(strat)
    b = _ann_stats(bench)
    alpha = float(np.mean(strat - bench) * 12)
    te    = float(np.std(strat - bench, ddof=1) * np.sqrt(12))
    return {
        "monthly_strat": strat, "monthly_bench": bench, "turnover": turn,
        "strategy": s, "benchmark": b, "alpha_ann": alpha, "tracking_error": te,
        "mean_turnover_pct": float(np.mean(turn) * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_moe():
    print("=" * 88)
    print(" MoE: Regime-Conditional Expert Blending (Plan B)")
    print("=" * 88)

    # 1. Train experts
    expert_allocs = {}
    for regime, cfg in EXPERT_CONFIGS.items():
        print(f"\n[expert for {regime}] {cfg.name} training…")
        out = run_variant(cfg)
        alloc = _expert_allocations(out, cfg.dataset_path or "regime_dataset.csv")
        expert_allocs[regime] = alloc
        print(f"  OOS months: {len(alloc)} "
              f"({alloc.index.min().date()} → {alloc.index.max().date()})")

    # 2. Gate model
    print(f"\n[gate] {GATE_CONFIG.name} training…")
    gate_out = run_variant(GATE_CONFIG)
    gate_proba = gate_out["proba_df"][REGIMES].copy()
    print(f"  Gate OOS months: {len(gate_proba)}")

    # 3. Align on intersection
    common_idx = gate_proba.index
    for alloc in expert_allocs.values():
        common_idx = common_idx.intersection(alloc.index)
    common_idx = common_idx.sort_values()
    print(f"\n[align] intersection: {len(common_idx)} months "
          f"({common_idx.min().date()} → {common_idx.max().date()})")

    # 4. Realize returns
    df_main, _ = load_dataset("regime_dataset.csv")
    df_sub = df_main.loc[common_idx]
    eq = df_sub["fwd_ret"].values
    bd = df_sub["bond_fwd_ret"].values
    ch = np.full_like(eq, 0.02 / 12.0)

    # Gate proba aligned
    gp = gate_proba.loc[common_idx]
    # Expert allocations aligned
    ea = {r: expert_allocs[r].loc[common_idx] for r in REGIMES}

    # ── Mode 1: SOFT blend ──
    w_eq_s = np.zeros(len(common_idx))
    w_bd_s = np.zeros(len(common_idx))
    w_ch_s = np.zeros(len(common_idx))
    for r in REGIMES:
        p = gp[r].values
        w_eq_s += p * ea[r]["w_equity"].values
        w_bd_s += p * ea[r]["w_bond"].values
        w_ch_s += p * ea[r]["w_cash"].values
    moe_soft = _run_and_score(w_eq_s, w_bd_s, w_ch_s, eq, bd, ch)

    # ── Mode 2: HARD select (argmax gate) ──
    argmax_regime = gp.idxmax(axis=1).values  # per-month winner
    w_eq_h = np.array([ea[r]["w_equity"].loc[d] for d, r in zip(common_idx, argmax_regime)])
    w_bd_h = np.array([ea[r]["w_bond"].loc[d]   for d, r in zip(common_idx, argmax_regime)])
    w_ch_h = np.array([ea[r]["w_cash"].loc[d]   for d, r in zip(common_idx, argmax_regime)])
    moe_hard = _run_and_score(w_eq_h, w_bd_h, w_ch_h, eq, bd, ch)

    # ── Baselines on same window ──
    # P4_meta solo on same intersection
    p4_alloc = ea["BULL"]  # P4_meta is the BULL expert
    w_eq_p4 = p4_alloc["w_equity"].values
    w_bd_p4 = p4_alloc["w_bond"].values
    w_ch_p4 = p4_alloc["w_cash"].values
    p4_solo = _run_and_score(w_eq_p4, w_bd_p4, w_ch_p4, eq, bd, ch)

    # P0 baseline solo on same intersection
    p0_alloc = ea["BASE"]
    p0_solo = _run_and_score(p0_alloc["w_equity"].values,
                             p0_alloc["w_bond"].values,
                             p0_alloc["w_cash"].values, eq, bd, ch)

    # ── Report ──
    print("\n" + "=" * 88)
    print(f" Results — common window {common_idx.min().date()} → {common_idx.max().date()} "
          f"({len(common_idx)} months)")
    print("=" * 88)
    rows = [
        ("MoE (soft blend)",  moe_soft),
        ("MoE (hard select)", moe_hard),
        ("P4_meta solo",      p4_solo),
        ("P0 baseline solo",  p0_solo),
    ]
    print(f"\n{'variant':<22}{'AnnRet':>10}{'Vol':>10}{'Sharpe':>10}"
          f"{'MaxDD':>10}{'Alpha':>10}{'TE':>10}{'Turn/mo':>10}")
    print("-" * 92)
    for name, r in rows:
        s = r["strategy"]
        print(f"{name:<22}"
              f"{s['ann_ret']*100:>9.2f}%"
              f"{s['ann_vol']*100:>9.2f}%"
              f"{s['sharpe']:>10.2f}"
              f"{s['max_dd']*100:>9.1f}%"
              f"{r['alpha_ann']*100:>+9.2f}%"
              f"{r['tracking_error']*100:>9.2f}%"
              f"{r['mean_turnover_pct']:>9.2f}%")
    # benchmark
    b = moe_soft["benchmark"]
    print(f"{'Benchmark '+BENCHMARK_LABEL:<22}"
          f"{b['ann_ret']*100:>9.2f}%"
          f"{b['ann_vol']*100:>9.2f}%"
          f"{b['sharpe']:>10.2f}"
          f"{b['max_dd']*100:>9.1f}%"
          f"{'—':>10}{'—':>10}{'—':>10}")

    # Per-regime alpha decomposition for soft/hard
    print("\n" + "=" * 88)
    print(" MoE Per-Regime Alpha Decomposition (true regime)")
    print("=" * 88)
    reg_true = df_sub["regime"].values
    for name, r in [("MoE soft", moe_soft), ("MoE hard", moe_hard),
                    ("P4 solo", p4_solo), ("P0 solo", p0_solo)]:
        strat = r["monthly_strat"]
        bench = r["monthly_bench"]
        print(f"\n{name}:")
        for regime in ["BULL", "BASE", "BEAR"]:
            mask = reg_true == regime
            n = int(mask.sum())
            if n == 0:
                continue
            alpha_bp = float((strat[mask] - bench[mask]).mean() * 10000)
            print(f"  {regime:<5} n={n:<3}  alpha = {alpha_bp:>+6.1f} bp/mo")

    # Gate diagnostics
    print("\n" + "=" * 88)
    print(" Gate Diagnostics — how often does argmax pick each expert?")
    print("=" * 88)
    from collections import Counter
    cnt = Counter(argmax_regime)
    for r in REGIMES:
        print(f"  argmax = {r:<5} {cnt.get(r, 0):>3} / {len(argmax_regime)} months "
              f"({cnt.get(r, 0)/len(argmax_regime)*100:.1f}%)")

    # Persist MoE monthly output
    moe_df = pd.DataFrame({
        "date": common_idx,
        "w_eq_soft": w_eq_s, "w_bd_soft": w_bd_s, "w_ch_soft": w_ch_s,
        "w_eq_hard": w_eq_h, "w_bd_hard": w_bd_h, "w_ch_hard": w_ch_h,
        "gate_argmax": argmax_regime,
        "gate_P_BEAR": gp["BEAR"].values,
        "gate_P_BASE": gp["BASE"].values,
        "gate_P_BULL": gp["BULL"].values,
        "soft_ret": moe_soft["monthly_strat"],
        "hard_ret": moe_hard["monthly_strat"],
        "bench_ret": moe_soft["monthly_bench"],
        "true_regime": reg_true,
    }).set_index("date")
    moe_df.to_csv("ai_pred_moe.csv")
    print(f"\n  Saved → ai_pred_moe.csv  ({len(moe_df)} rows)")

    # Summary JSON for dashboard
    import json
    summary = {
        "n_oos_months": int(len(common_idx)),
        "period_start": str(common_idx.min().date()),
        "period_end":   str(common_idx.max().date()),
        "moe_soft":     {"ann_return": moe_soft["strategy"]["ann_ret"],
                         "ann_vol":    moe_soft["strategy"]["ann_vol"],
                         "sharpe":     moe_soft["strategy"]["sharpe"],
                         "max_dd":     moe_soft["strategy"]["max_dd"],
                         "alpha_ann":  moe_soft["alpha_ann"],
                         "tracking_error": moe_soft["tracking_error"],
                         "turnover":   moe_soft["mean_turnover_pct"]},
        "moe_hard":     {"ann_return": moe_hard["strategy"]["ann_ret"],
                         "ann_vol":    moe_hard["strategy"]["ann_vol"],
                         "sharpe":     moe_hard["strategy"]["sharpe"],
                         "max_dd":     moe_hard["strategy"]["max_dd"],
                         "alpha_ann":  moe_hard["alpha_ann"],
                         "tracking_error": moe_hard["tracking_error"],
                         "turnover":   moe_hard["mean_turnover_pct"]},
        "p4_solo":      {"ann_return": p4_solo["strategy"]["ann_ret"],
                         "sharpe":     p4_solo["strategy"]["sharpe"],
                         "max_dd":     p4_solo["strategy"]["max_dd"],
                         "alpha_ann":  p4_solo["alpha_ann"],
                         "turnover":   p4_solo["mean_turnover_pct"]},
        "p0_solo":      {"ann_return": p0_solo["strategy"]["ann_ret"],
                         "sharpe":     p0_solo["strategy"]["sharpe"],
                         "max_dd":     p0_solo["strategy"]["max_dd"],
                         "alpha_ann":  p0_solo["alpha_ann"],
                         "turnover":   p0_solo["mean_turnover_pct"]},
        "benchmark":    {"ann_return": moe_soft["benchmark"]["ann_ret"],
                         "sharpe":     moe_soft["benchmark"]["sharpe"],
                         "max_dd":     moe_soft["benchmark"]["max_dd"],
                         "label":      BENCHMARK_LABEL},
        "expert_assignment": {
            "BULL": "P4_meta",
            "BASE": "P0_baseline",
            "BEAR": "VIXFREE_w25_t030",
        },
        "gate": "P0_baseline regime proba",
    }
    with open("ai_pred_moe.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved → ai_pred_moe.json")


if __name__ == "__main__":
    run_moe()
