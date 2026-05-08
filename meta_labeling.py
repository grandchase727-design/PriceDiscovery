###############################################################################
# Meta-Labeling (Lopez de Prado, AFML Ch.3) — proper formulation
# ============================================================================
# Primary:  multiclass argmax from the main LightGBM regime classifier
# Meta:     binary LightGBM predicting "will primary's bet be correct?"
# Decision: if meta P(correct) > 0.5 → use primary's allocation grid
#           else → default to BASE (75/10/15)
###############################################################################

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import entropy as sp_entropy

from purged_cv import PurgedKFold, PurgedWalkForward


REGIMES = ["BEAR", "BASE", "BULL"]

# Daily-resolution path-dependent features are EXCLUDED from meta-input.
# Reason: meta has only ~36-month train sub-sample; adding 10 extra features
# overfits and degrades P4 production performance. Primary model still uses
# them. (Empirical: with daily-in-meta, P4 Sharpe 0.86 → 0.83.)
META_EXCLUDE = {
    "vix_max_in_month", "vix_max_minus_end",
    "acwi_max_dd_in_month", "acwi_pos_days_pct",
    "acwi_vol_of_vol_5d", "acwi_tail_days_count",
    "move_max_jump_1d", "breakdown_below_sma20_days",
    "vix_skew_in_month", "corr_acwi_agg_21d",
}


def apply_meta_filter(df: pd.DataFrame, feat_cols: List[str],
                      primary_result: Dict, cfg) -> Dict:
    """
    Train a binary meta-classifier on OOS predictions from the primary model,
    then produce final regime assignments.

    primary_result : output of train_and_evaluate() — must contain 'proba_df'.
    """
    proba_df = primary_result["proba_df"].copy()
    # Align order with df (same index)
    proba_df = proba_df.loc[proba_df.index.isin(df.index)]
    df = df.loc[proba_df.index]

    # Meta target: was the primary's argmax prediction correct?
    y_meta = (proba_df["pred"].values == proba_df["true"].values).astype(int)

    # Meta features = (base features minus daily) + primary proba + entropy
    meta_feat_cols = [c for c in feat_cols if c not in META_EXCLUDE]
    P = proba_df[REGIMES].values
    P = np.clip(P, 1e-12, 1.0)
    meta_X = np.hstack([
        df[meta_feat_cols].values,
        P,
        np.array([sp_entropy(row, base=2) for row in P]).reshape(-1, 1),
    ])

    if cfg.cv_mode == "kfold":
        cv = PurgedKFold(n_splits=cfg.n_splits, label_horizon=1, embargo=cfg.embargo)
    else:
        cv = PurgedWalkForward(n_splits=cfg.n_splits, label_horizon=1,
                               embargo=cfg.embargo, min_train=cfg.min_train)

    meta_proba = np.full(len(y_meta), np.nan)
    for train_idx, test_idx in cv.split(meta_X):
        clf = lgb.LGBMClassifier(
            objective="binary",
            learning_rate=0.03,
            n_estimators=300,
            num_leaves=15,
            max_depth=4,
            min_child_samples=15,
            reg_lambda=1.0,
            reg_alpha=0.5,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
        clf.fit(meta_X[train_idx], y_meta[train_idx])
        meta_proba[test_idx] = clf.predict_proba(meta_X[test_idx])[:, 1]

    # Final regime decision
    final_regime = []
    mask = ~np.isnan(meta_proba)
    for i, idx in enumerate(proba_df.index):
        p = meta_proba[i]
        if np.isnan(p):
            final_regime.append("BASE")
        elif p > 0.5:
            final_regime.append(proba_df["pred"].iloc[i])
        else:
            final_regime.append("BASE")

    out_df = proba_df.copy()
    out_df["meta_confidence"] = meta_proba
    out_df["final_regime"] = final_regime
    out_df = out_df.loc[mask]  # Drop rows without meta prediction

    hit_rate = float(np.mean(y_meta[mask])) if mask.any() else 0.0
    agreement = float(np.mean(out_df["final_regime"] == out_df["pred"]))
    override_rate = float(np.mean(out_df["final_regime"] != out_df["pred"]))

    return {
        "proba_df": out_df,
        "meta_hit_rate":  hit_rate,
        "meta_agreement": agreement,
        "meta_override_rate": override_rate,
    }
