###############################################################################
# Ablation Harness — runs named variants and logs metrics to CSV
# ============================================================================
# Each variant = VariantConfig object. run_all() trains, backtests, and
# appends one row per variant to ablation_results.csv for attribution.
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import pandas as pd
from typing import List, Optional

from ml_signal_engine import VariantConfig, run_variant, load_dataset


RESULT_COLUMNS = [
    "variant", "n_features", "n_rows", "cv_mode", "bull_threshold", "class_weight",
    "use_meta",
    "sharpe", "ann_return", "ann_vol", "max_dd", "alpha_ann", "tracking_error",
    "mean_turnover_pct", "accuracy", "balanced_acc",
    "bull_recall", "base_recall", "bear_recall", "mean_proba_entropy",
    "benchmark_sharpe", "benchmark_ann_return", "benchmark_max_dd",
]


def run_all(variants: List[VariantConfig], out_path: str = "ablation_results.csv",
            verbose: bool = True) -> pd.DataFrame:
    rows = []
    for i, cfg in enumerate(variants, 1):
        if verbose:
            print(f"\n[{i}/{len(variants)}] Running variant: {cfg.name} "
                  f"(cv={cfg.cv_mode}, cw={cfg.class_weight}, "
                  f"thr={cfg.bull_threshold}, meta={cfg.use_meta})")
        try:
            out = run_variant(cfg)
            summary = out["summary"]
            rows.append(summary)
            if verbose:
                print(f"  → Sharpe={summary['sharpe']:.3f}  Alpha={summary['alpha_ann']*100:+.2f}%  "
                      f"BullRec={summary['bull_recall']:.2f}  MaxDD={summary['max_dd']*100:.1f}%  "
                      f"Turn={summary['mean_turnover_pct']:.1f}%/mo")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            rows.append({"variant": cfg.name, "error": str(e)})

    df = pd.DataFrame(rows)
    # Reorder columns
    known = [c for c in RESULT_COLUMNS if c in df.columns]
    extra = [c for c in df.columns if c not in known]
    df = df[known + extra]
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"\nSaved → {out_path}  ({len(df)} variants)")
    return df


# ---------------------------------------------------------------------------
# Variant definitions (phase-by-phase)
# ---------------------------------------------------------------------------
def p0_baseline_variants() -> List[VariantConfig]:
    """Phase 0: correctness sanity checks."""
    return [
        VariantConfig(name="P0_kfold_baseline", cv_mode="kfold",
                      class_weight="balanced"),
        VariantConfig(name="P0_walkforward_baseline", cv_mode="walkforward",
                      class_weight="balanced"),
    ]


def p2_bull_detection_variants(feature_cols: Optional[List[str]] = None) -> List[VariantConfig]:
    """
    Phase 2: class weight sweep × threshold sweep.
    4 weights × 3 thresholds = 12 variants.
    """
    weights = [
        {"BEAR": 1.0, "BASE": 1.0, "BULL": 1.5},
        {"BEAR": 1.0, "BASE": 1.0, "BULL": 2.0},
        {"BEAR": 1.0, "BASE": 1.0, "BULL": 2.5},
        {"BEAR": 1.0, "BASE": 1.0, "BULL": 3.0},
    ]
    thresholds = [0.25, 0.30, 0.35]
    variants = []
    for w in weights:
        for t in thresholds:
            name = f"P2_w{w['BULL']:.1f}_t{t:.2f}"
            variants.append(VariantConfig(
                name=name, feature_cols=feature_cols,
                class_weight=w, bull_threshold=t, cv_mode="walkforward",
            ))
    return variants


def vix_free_ablation_variants(feature_cols: Optional[List[str]] = None) -> List[VariantConfig]:
    """
    Run on a separate 'regime_dataset_novix.csv' built with label_regime(bull_vix=inf, bear_vix=inf).
    Tests whether BULL recall ceiling is driven by label-feature circularity.
    """
    return [
        VariantConfig(
            name="VIXFREE_baseline",
            dataset_path="regime_dataset_novix.csv",
            feature_cols=feature_cols,
            class_weight="balanced",
            cv_mode="walkforward",
        ),
        VariantConfig(
            name="VIXFREE_w2.5_t0.30",
            dataset_path="regime_dataset_novix.csv",
            feature_cols=feature_cols,
            class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 2.5},
            bull_threshold=0.30,
            cv_mode="walkforward",
        ),
    ]


def p1_breadth_variants(best_p2_cfg: VariantConfig = None) -> List[VariantConfig]:
    """
    Phase 3 (P1): breadth features added to feature set.
    Assumes regime_dataset.csv already contains breadth cols (eq_*, bd_*).
    """
    breadth_cols = ["eq_pct_bullish", "eq_pct_downtrend", "eq_tcs_median",
                    "eq_rss_std", "bd_pct_bullish", "bd_tcs_median"]

    # Load dataset to get full feature list minus reserved
    from ml_signal_engine import load_dataset
    df, all_feats = load_dataset("regime_dataset.csv")
    macro_feats = [f for f in all_feats if f not in breadth_cols]
    all_feats_incl_breadth = macro_feats + breadth_cols

    variants = [
        VariantConfig(
            name="P1_breadth_default",
            feature_cols=all_feats_incl_breadth,
            class_weight="balanced",
            cv_mode="walkforward",
        ),
    ]
    # Apply P1 breadth on top of P2 best config
    if best_p2_cfg is not None:
        variants.append(VariantConfig(
            name="P1_breadth_plus_P2best",
            feature_cols=all_feats_incl_breadth,
            class_weight=best_p2_cfg.class_weight,
            bull_threshold=best_p2_cfg.bull_threshold,
            cv_mode="walkforward",
        ))
    return variants


def p4_meta_variants(winner_cfg: VariantConfig) -> List[VariantConfig]:
    """Phase 4: meta-labeling wrapper on best P1+P2+P3 config."""
    return [
        VariantConfig(
            name="P4_meta",
            feature_cols=winner_cfg.feature_cols,
            class_weight=winner_cfg.class_weight,
            bull_threshold=winner_cfg.bull_threshold,
            cv_mode="walkforward",
            use_meta=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def select_best(df: pd.DataFrame, metric_fn=None) -> pd.Series:
    """Pick best variant by combined score (Sharpe + 0.5 × BULL recall)."""
    if metric_fn is None:
        def metric_fn(row):
            return row["sharpe"] + 0.5 * row["bull_recall"]
    df = df.dropna(subset=["sharpe"]).copy()
    df["__score"] = df.apply(metric_fn, axis=1)
    return df.sort_values("__score", ascending=False).iloc[0]


def _best_p2_cfg_from_results(df_results: pd.DataFrame) -> VariantConfig:
    """Parse the best P2 variant from the sweep results."""
    df_p2 = df_results[df_results["variant"].str.startswith("P2_")].copy()
    if df_p2.empty:
        return None
    df_p2["score"] = df_p2["sharpe"] + 0.5 * df_p2["bull_recall"]
    best = df_p2.sort_values("score", ascending=False).iloc[0]
    # Parse weight & threshold from variant name: "P2_w2.5_t0.30"
    import re
    m = re.match(r"P2_w([\d.]+)_t([\d.]+)", best["variant"])
    if not m:
        return None
    w = float(m.group(1))
    t = float(m.group(2))
    return VariantConfig(
        name=f"best_p2_{best['variant']}",
        class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": w},
        bull_threshold=t,
        cv_mode="walkforward",
    )


if __name__ == "__main__":
    phases = sys.argv[1:] if len(sys.argv) > 1 else ["p0", "p2", "vixfree", "p1", "p4"]

    all_variants = []
    if "p0" in phases:
        all_variants += p0_baseline_variants()
    if "p2" in phases:
        all_variants += p2_bull_detection_variants()
    if "vixfree" in phases:
        all_variants += vix_free_ablation_variants()

    # Run P0+P2+VIXFREE first so we can find best_p2 for the later phases
    if all_variants:
        df_stage1 = run_all(all_variants, out_path="ablation_results.csv")
    else:
        df_stage1 = pd.DataFrame()

    best_p2 = _best_p2_cfg_from_results(df_stage1) if len(df_stage1) else None
    if best_p2:
        print(f"\n[best P2] {best_p2.name}  (w={best_p2.class_weight['BULL']}, "
              f"thr={best_p2.bull_threshold})")

    stage2 = []
    if "p1" in phases:
        stage2 += p1_breadth_variants(best_p2)
    if stage2:
        df_stage2 = run_all(stage2, out_path="ablation_results.csv",
                            verbose=True)
        df_all = pd.concat([df_stage1, df_stage2], ignore_index=True)
    else:
        df_all = df_stage1

    # Winner among P1/P2/P3 for P4 meta
    stage3 = []
    if "p4" in phases and len(df_all):
        ranked = df_all.dropna(subset=["sharpe"]).copy()
        ranked["score"] = ranked["sharpe"] + 0.5 * ranked["bull_recall"]
        winner_row = ranked.sort_values("score", ascending=False).iloc[0]
        breadth_cols = ["eq_pct_bullish", "eq_pct_downtrend", "eq_tcs_median",
                        "eq_rss_std", "bd_pct_bullish", "bd_tcs_median"]
        from ml_signal_engine import load_dataset
        df_x, all_feats = load_dataset("regime_dataset.csv")
        winner_cfg = VariantConfig(
            name="winner_for_meta",
            feature_cols=all_feats,
            class_weight=(best_p2.class_weight if best_p2 else "balanced"),
            bull_threshold=(best_p2.bull_threshold if best_p2 else None),
            cv_mode="walkforward",
        )
        stage3 += p4_meta_variants(winner_cfg)
        df_stage3 = run_all(stage3, out_path="ablation_results.csv",
                            verbose=True)
        df_all = pd.concat([df_all, df_stage3], ignore_index=True)

    # Final consolidated output
    df_all.to_csv("ablation_results.csv", index=False)
    print("\n" + "=" * 72)
    print(f" Final ablation results ({len(df_all)} variants) — top 15")
    print("=" * 72)
    if "bull_recall" in df_all.columns and df_all["sharpe"].notna().any():
        df_rank = df_all.dropna(subset=["sharpe"]).copy()
        df_rank["score"] = df_rank["sharpe"] + 0.5 * df_rank["bull_recall"]
        cols = ["variant", "sharpe", "alpha_ann", "bull_recall", "bear_recall",
                "max_dd", "mean_turnover_pct", "score"]
        print(df_rank[cols].sort_values("score", ascending=False).head(15).to_string(index=False))
