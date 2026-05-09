###############################################################################
# AI Prediction Cache — produces artifacts consumed by the "AI Prediction" tab
# ============================================================================
# Runs the winning P4_meta configuration and persists:
#   - ai_pred_proba.parquet     OOS probabilities + meta output + true/pred
#   - ai_pred_returns.parquet   Monthly strategy vs benchmark returns & weights
#   - ai_pred_feature_imp.csv   Feature importance for winner
#   - ai_pred_metrics.json      KPI numbers for the tab headline
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd

from ml.ml_signal_engine import (VariantConfig, run_variant, load_dataset,
                              blend_allocation, hard_allocation,
                              ALLOCATION_GRID, REGIMES,
                              BENCHMARK_WEIGHTS, BENCHMARK_LABEL)


WINNER = VariantConfig(
    name="P4_meta_winner",
    class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 3.0},
    bull_threshold=0.25,
    cv_mode="walkforward",
    use_meta=True,
)

BASELINE = VariantConfig(
    name="P0_baseline",
    class_weight="balanced",
    cv_mode="walkforward",
)


def _monthly_returns(proba_df, df, mode, hard_col="final_regime", tc_bps=5.0):
    eq = df.loc[proba_df.index, "fwd_ret"].values
    bd = df.loc[proba_df.index, "bond_fwd_ret"].values
    ch = np.full_like(eq, 0.02 / 12.0)
    if mode == "hard":
        weights = [hard_allocation(r) for r in proba_df[hard_col].values]
    else:
        weights = [blend_allocation(proba_df.iloc[i]) for i in range(len(proba_df))]
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
        "strategy_ret": strat,
        "benchmark_ret": bench,
        "w_equity": w_eq, "w_bond": w_bd, "w_cash": w_ch,
        "turnover": turn,
    }, index=proba_df.index)


def generate():
    print("[ai_pred] Running WINNER (P4_meta)…")
    out_w = run_variant(WINNER)
    print("[ai_pred] Running BASELINE (P0_walkforward)…")
    out_b = run_variant(BASELINE)

    # Winner proba (with meta_confidence + final_regime)
    proba_w = out_w["proba_df"].copy()
    # Baseline proba (blend-mode, 3-class argmax)
    proba_b = out_b["proba_df"].copy()

    # Monthly returns
    df_full, feats = load_dataset("regime_dataset.csv")
    df = df_full.loc[proba_w.index]
    ret_w = _monthly_returns(proba_w, df, mode="hard", hard_col="final_regime")
    ret_b = _monthly_returns(proba_b, df_full.loc[proba_b.index], mode="blend")
    # Align both on intersection for fair comparison (use winner's index as master)
    common_b = ret_b.loc[ret_b.index.isin(ret_w.index)]
    ret_w = ret_w.copy()
    ret_w["benchmark_static_ret"] = ret_w["benchmark_ret"]  # 75/10/15 benchmark
    ret_w["p0_baseline_ret"] = common_b["strategy_ret"].reindex(ret_w.index)

    # Persist
    proba_w.to_parquet("ai_pred_proba.parquet") if False else proba_w.to_csv("ai_pred_proba.csv")
    ret_w.to_csv("ai_pred_returns.csv")

    fi = pd.Series(out_w["result"]["feature_importance"]).sort_values(ascending=False)
    fi.to_csv("ai_pred_feature_imp.csv", header=["importance"])

    summary_w = out_w["summary"]
    summary_b = out_b["summary"]
    metrics = {
        "winner_name":         summary_w["variant"],
        "winner_sharpe":       summary_w["sharpe"],
        "winner_alpha_ann":    summary_w["alpha_ann"],
        "winner_max_dd":       summary_w["max_dd"],
        "winner_ann_return":   summary_w["ann_return"],
        "winner_ann_vol":      summary_w["ann_vol"],
        "winner_bull_recall":  summary_w["bull_recall"],
        "winner_base_recall":  summary_w["base_recall"],
        "winner_bear_recall":  summary_w["bear_recall"],
        "winner_turnover":     summary_w["mean_turnover_pct"],
        "winner_n_features":   summary_w["n_features"],
        "winner_n_rows":       summary_w["n_rows"],
        "baseline_sharpe":     summary_b["sharpe"],
        "baseline_alpha_ann":  summary_b["alpha_ann"],
        "baseline_max_dd":     summary_b["max_dd"],
        "baseline_turnover":   summary_b["mean_turnover_pct"],
        "baseline_bull_recall":summary_b["bull_recall"],
        "benchmark_sharpe":    summary_w["benchmark_sharpe"],
        "benchmark_ann_return":summary_w["benchmark_ann_return"],
        "benchmark_max_dd":    summary_w["benchmark_max_dd"],
        "meta_hit_rate":       (out_w["meta_info"] or {}).get("meta_hit_rate"),
        "meta_agreement":      (out_w["meta_info"] or {}).get("meta_agreement"),
        "meta_override_rate":  (out_w["meta_info"] or {}).get("meta_override_rate"),
        "n_oos_months":        int(len(proba_w)),
        "period_start":        str(proba_w.index.min().date()),
        "period_end":          str(proba_w.index.max().date()),
        "benchmark_label":     BENCHMARK_LABEL,
        "benchmark_weights":   BENCHMARK_WEIGHTS,
    }
    with open("ai_pred_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Also write per-regime distribution (confusion matrix)
    cm = out_w["result"]["confusion"]
    cm.to_csv("ai_pred_confusion.csv")

    print("[ai_pred] Saved:")
    for f in ["ai_pred_proba.csv", "ai_pred_returns.csv", "ai_pred_feature_imp.csv",
              "ai_pred_metrics.json", "ai_pred_confusion.csv"]:
        print(f"  - {f}")
    print(f"\n[ai_pred] Winner Sharpe={metrics['winner_sharpe']:.3f}  "
          f"Alpha={metrics['winner_alpha_ann']*100:+.2f}%  "
          f"BullRec={metrics['winner_bull_recall']:.2f}")


if __name__ == "__main__":
    generate()
