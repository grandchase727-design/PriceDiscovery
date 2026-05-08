###############################################################################
# Regime-Conditional Diagnostic (Step 1)
# ============================================================================
# Question: do P0 / P2 / P3 / P1 / P4 variants behave DIFFERENTLY by true regime?
# If yes → worth building a regime-conditional selector.
# If all variants produce similar per-regime returns → don't bother.
#
# Output: per-regime (BULL/BASE/BEAR) table of
#   mean return, Sharpe, alpha vs ACWI90/10 benchmark, n_months
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from ml_signal_engine import (VariantConfig, run_variant, load_dataset,
                              blend_allocation, hard_allocation,
                              BENCHMARK_WEIGHTS)


# ─────────────────────────────────────────────────────────────────────────────
# Variants to diagnose
# ─────────────────────────────────────────────────────────────────────────────
VARIANTS = [
    VariantConfig(name="P0_baseline",  cv_mode="walkforward",
                  class_weight="balanced"),
    VariantConfig(name="P2_w3_t025",   cv_mode="walkforward",
                  class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 3.0},
                  bull_threshold=0.25),
    VariantConfig(name="P3_full",      cv_mode="walkforward",
                  class_weight="balanced"),  # P3 features are default in dataset
    VariantConfig(name="P4_meta",      cv_mode="walkforward",
                  class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 3.0},
                  bull_threshold=0.25, use_meta=True),
    VariantConfig(name="VIXFREE_w25_t030",
                  dataset_path="regime_dataset_novix.csv",
                  cv_mode="walkforward",
                  class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 2.5},
                  bull_threshold=0.30),
]


def _monthly_strategy_ret(out, df, tc_bps: float = 5.0) -> pd.DataFrame:
    """Produce per-month strategy and benchmark returns + true regime."""
    proba = out["proba_df"]
    eq = df.loc[proba.index, "fwd_ret"].values
    bd = df.loc[proba.index, "bond_fwd_ret"].values
    ch = np.full_like(eq, 0.02 / 12.0)

    if "final_regime" in proba.columns:
        weights = [hard_allocation(r) for r in proba["final_regime"].values]
    else:
        weights = [blend_allocation(proba.iloc[i]) for i in range(len(proba))]
    w_eq = np.array([w["equity"] for w in weights])
    w_bd = np.array([w["bond"] for w in weights])
    w_ch = np.array([w["cash"] for w in weights])

    bm = BENCHMARK_WEIGHTS
    w_prev = np.vstack([np.array([bm["equity"], bm["bond"], bm["cash"]])[None, :],
                        np.stack([w_eq, w_bd, w_ch], axis=1)[:-1]])
    w_curr = np.stack([w_eq, w_bd, w_ch], axis=1)
    turn = np.abs(w_curr - w_prev).sum(axis=1)
    tc = turn * (tc_bps / 10000.0)

    strat = w_eq * eq + w_bd * bd + w_ch * ch - tc
    bench = bm["equity"] * eq + bm["bond"] * bd + bm["cash"] * ch

    return pd.DataFrame({
        "strategy": strat, "benchmark": bench,
        "excess": strat - bench,
        "true_regime": proba["true"].values,
    }, index=proba.index)


def _per_regime_stats(monthly: pd.DataFrame) -> pd.DataFrame:
    """Group by true_regime, compute mean return, Sharpe, alpha."""
    out = []
    for regime in ["BULL", "BASE", "BEAR"]:
        sub = monthly[monthly["true_regime"] == regime]
        n = len(sub)
        if n == 0:
            continue
        out.append({
            "regime": regime,
            "n": n,
            "strat_mean_m": sub["strategy"].mean(),
            "bench_mean_m": sub["benchmark"].mean(),
            "alpha_m":      sub["excess"].mean(),
            "strat_sharpe": (sub["strategy"].mean() / sub["strategy"].std(ddof=1)
                             * np.sqrt(12)) if sub["strategy"].std(ddof=1) > 0 else np.nan,
            "bench_sharpe": (sub["benchmark"].mean() / sub["benchmark"].std(ddof=1)
                             * np.sqrt(12)) if sub["benchmark"].std(ddof=1) > 0 else np.nan,
            "hit_rate":     float((sub["excess"] > 0).mean()),
        })
    return pd.DataFrame(out)


def run_diagnostic():
    print("=" * 92)
    print(" Regime-Conditional Diagnostic (do variants differ by true regime?)")
    print("=" * 92)

    # Pre-load datasets (reused across variants)
    df_main, _ = load_dataset("regime_dataset.csv")
    df_novix, _ = load_dataset("regime_dataset_novix.csv")

    all_tables = {}
    for cfg in VARIANTS:
        print(f"\n[{cfg.name}] running…")
        out = run_variant(cfg)
        if cfg.dataset_path == "regime_dataset_novix.csv":
            df = df_novix.loc[out["proba_df"].index]
        else:
            df = df_main.loc[out["proba_df"].index]
        monthly = _monthly_strategy_ret(out, df)
        stats = _per_regime_stats(monthly)
        stats.insert(0, "variant", cfg.name)
        all_tables[cfg.name] = stats

    combined = pd.concat(all_tables.values(), ignore_index=True)

    # Print pivot: alpha by (variant, regime)
    print("\n\n" + "=" * 92)
    print(" Per-regime mean MONTHLY alpha (strategy − benchmark), bp/month")
    print("=" * 92)
    pivot_alpha = combined.pivot(index="variant", columns="regime",
                                  values="alpha_m") * 10000  # to bps
    print(pivot_alpha.round(1).to_string())

    print("\n" + "=" * 92)
    print(" Per-regime strategy SHARPE (regime-conditional, annualized)")
    print("=" * 92)
    pivot_sh = combined.pivot(index="variant", columns="regime",
                               values="strat_sharpe")
    print(pivot_sh.round(2).to_string())

    print("\n" + "=" * 92)
    print(" Hit rate (% months with positive excess return), per regime")
    print("=" * 92)
    pivot_hit = combined.pivot(index="variant", columns="regime",
                                values="hit_rate") * 100
    print(pivot_hit.round(0).to_string())

    print("\n" + "=" * 92)
    print(" Sample size (n months per regime in each variant's OOS window)")
    print("=" * 92)
    pivot_n = combined.pivot(index="variant", columns="regime", values="n")
    print(pivot_n.fillna(0).astype(int).to_string())

    # Best variant per regime
    print("\n" + "=" * 92)
    print(" Best variant PER REGIME (by monthly alpha)")
    print("=" * 92)
    for regime in ["BULL", "BASE", "BEAR"]:
        col = pivot_alpha[regime]
        best = col.idxmax()
        worst = col.idxmin()
        spread_bp = col.max() - col.min()
        print(f"  {regime:<5}  best: {best:<22} ({col[best]:>+6.1f} bp/m)"
              f"   |   worst: {worst:<22} ({col[worst]:>+6.1f} bp/m)"
              f"   |   spread: {spread_bp:>5.1f} bp")

    # Dispersion check: if spread across variants within each regime is > 20 bp/month
    # on at least 2 regimes → meaningful differentiation exists.
    spreads = [pivot_alpha[r].max() - pivot_alpha[r].min()
               for r in pivot_alpha.columns]
    meaningful = sum(1 for s in spreads if s > 20)

    print("\n" + "=" * 92)
    print(" VERDICT")
    print("=" * 92)
    print(f"  Within-regime spread across variants: "
          f"{', '.join(f'{r}={s:.0f}bp' for r, s in zip(pivot_alpha.columns, spreads))}")
    print(f"  Regimes with >20bp/month spread: {meaningful}/3")
    if meaningful >= 2:
        print("  → STRONG dispersion. Proceed to Step 2 (soft blending regime-conditional).")
    elif meaningful == 1:
        print("  → MODERATE dispersion. Step 2 may help marginally; weigh effort vs gain.")
    else:
        print("  → LOW dispersion. Variants behave similarly across regimes; "
              "stick with P4 meta (no selector needed).")

    # Save
    combined.to_csv("regime_conditional_diagnostic.csv", index=False)
    print(f"\n  Saved → regime_conditional_diagnostic.csv")


if __name__ == "__main__":
    run_diagnostic()
