###############################################################################
# ML Signal Engine — Forward 1M Regime Classifier (Bull / Base / Bear)
# ============================================================================
# LightGBM multiclass classifier with Purged K-Fold / Purged Walk-Forward CV.
# Exposes `run_variant(config)` for ablation-style experimentation.
###############################################################################

import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import entropy as sp_entropy
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, classification_report,
                             recall_score)

from purged_cv import PurgedKFold, PurgedWalkForward


REGIMES = ["BEAR", "BASE", "BULL"]  # ordered: low → high risk-on

ALLOCATION_GRID = {
    "BULL": {"equity": 0.90, "bond": 0.05, "cash": 0.05},
    "BASE": {"equity": 0.75, "bond": 0.10, "cash": 0.15},
    "BEAR": {"equity": 0.60, "bond": 0.15, "cash": 0.25},
}

# Passive benchmark used for Alpha / TE / Max DD comparison
# Set to "MSCI ACWI 90 / Cash 10" per user request
BENCHMARK_WEIGHTS = {"equity": 0.90, "bond": 0.00, "cash": 0.10}
BENCHMARK_LABEL = "ACWI 90 / Cash 10"

RESERVED_COLS = {"regime", "fwd_ret", "fwd_dd", "fwd_vol", "bond_fwd_ret"}


# ---------------------------------------------------------------------------
# Config object for a single variant
# ---------------------------------------------------------------------------
@dataclass
class VariantConfig:
    name: str
    dataset_path: str = "regime_dataset.csv"
    feature_cols: Optional[List[str]] = None           # None => use all non-reserved
    class_weight: Union[str, Dict, None] = "balanced"  # {'BEAR':1.0,'BASE':1.0,'BULL':2.5}
    bull_threshold: Optional[float] = None             # None=>argmax; else P(BULL)>thr → BULL
    cv_mode: str = "walkforward"                       # "kfold" or "walkforward"
    n_splits: int = 5
    embargo: int = 1
    min_train: int = 36
    use_meta: bool = False                             # P4 meta-labeling wrapper
    tc_bps: float = 5.0


# ---------------------------------------------------------------------------
# Data & model helpers
# ---------------------------------------------------------------------------
def load_dataset(path: str = "regime_dataset.csv") -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, index_col=0, parse_dates=True).dropna()
    feat_cols = [c for c in df.columns if c not in RESERVED_COLS]
    return df, feat_cols


def lgbm_params(n_classes: int = 3,
                class_weight: Union[str, Dict, None] = "balanced") -> Dict:
    """Strong regularization given ~225 rows × ~20 features."""
    return dict(
        objective="multiclass",
        num_class=n_classes,
        learning_rate=0.03,
        n_estimators=400,
        num_leaves=15,
        max_depth=4,
        min_child_samples=15,
        reg_alpha=0.5,
        reg_lambda=1.0,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=3,
        class_weight=class_weight,
        random_state=42,
        verbose=-1,
    )


def _build_cv(cfg: VariantConfig):
    if cfg.cv_mode == "kfold":
        return PurgedKFold(n_splits=cfg.n_splits, label_horizon=1, embargo=cfg.embargo)
    return PurgedWalkForward(
        n_splits=cfg.n_splits, label_horizon=1,
        embargo=cfg.embargo, min_train=cfg.min_train,
    )


def _classify_with_threshold(proba: np.ndarray, bull_threshold: Optional[float]) -> np.ndarray:
    """Argmax unless bull_threshold set — then BULL if P(BULL) > threshold."""
    if bull_threshold is None:
        return proba.argmax(axis=1)
    bull_idx = REGIMES.index("BULL")
    pred = proba.argmax(axis=1)
    bull_mask = proba[:, bull_idx] > bull_threshold
    pred[bull_mask] = bull_idx
    return pred


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------
def train_and_evaluate(df: pd.DataFrame, feat_cols: List[str],
                       cfg: VariantConfig) -> Dict:
    X = df[feat_cols].values
    y_str = df["regime"].values
    label_to_idx = {r: i for i, r in enumerate(REGIMES)}
    idx_to_label = {i: r for r, i in label_to_idx.items()}
    y = np.array([label_to_idx[r] for r in y_str])

    cv = _build_cv(cfg)
    oos_pred_idx = np.full(len(y), -1)
    oos_proba = np.full((len(y), len(REGIMES)), np.nan)
    fold_metrics = []
    feature_importance = np.zeros(len(feat_cols))
    n_folds_ran = 0

    # LightGBM sklearn wrapper expects class_weight keyed by integer labels,
    # not regime strings. Convert if user passed {'BULL': 2.5, ...}.
    cw = cfg.class_weight
    if isinstance(cw, dict):
        cw = {label_to_idx[k]: v for k, v in cw.items() if k in label_to_idx}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        clf = lgb.LGBMClassifier(**lgbm_params(len(REGIMES), cw))
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])
        pred = _classify_with_threshold(proba, cfg.bull_threshold)
        oos_pred_idx[test_idx] = pred
        oos_proba[test_idx] = proba
        acc = accuracy_score(y[test_idx], pred)
        bal = balanced_accuracy_score(y[test_idx], pred)
        fold_metrics.append({
            "fold": fold,
            "train_n": len(train_idx), "test_n": len(test_idx),
            "train_period": (df.index[train_idx].min().date(),
                             df.index[train_idx].max().date()),
            "test_period":  (df.index[test_idx].min().date(),
                             df.index[test_idx].max().date()),
            "accuracy": acc, "balanced_acc": bal,
        })
        feature_importance += clf.feature_importances_
        n_folds_ran += 1

    if n_folds_ran == 0:
        raise RuntimeError("No folds yielded train/test splits — check CV config.")
    feature_importance /= n_folds_ran

    mask = oos_pred_idx >= 0
    y_true_oos = y[mask]
    y_pred_oos = oos_pred_idx[mask]
    proba_oos = oos_proba[mask]
    idx_oos = df.index[mask]
    proba_oos = proba_oos / proba_oos.sum(axis=1, keepdims=True)

    proba_df = pd.DataFrame(proba_oos, index=idx_oos, columns=REGIMES)
    proba_df["true"] = [idx_to_label[i] for i in y_true_oos]
    proba_df["pred"] = [idx_to_label[i] for i in y_pred_oos]

    cm = confusion_matrix(y_true_oos, y_pred_oos, labels=list(range(len(REGIMES))))
    cm_df = pd.DataFrame(cm, index=[f"true_{r}" for r in REGIMES],
                         columns=[f"pred_{r}" for r in REGIMES])
    clf_report = classification_report(
        y_true_oos, y_pred_oos,
        labels=list(range(len(REGIMES))),
        target_names=REGIMES, digits=3, zero_division=0,
    )

    per_class_recall = recall_score(y_true_oos, y_pred_oos,
                                    labels=list(range(len(REGIMES))),
                                    average=None, zero_division=0)
    recall_by_regime = dict(zip(REGIMES, [float(r) for r in per_class_recall]))

    return {
        "proba_df": proba_df,
        "confusion": cm_df,
        "classification_report": clf_report,
        "fold_metrics": fold_metrics,
        "overall_accuracy": float(accuracy_score(y_true_oos, y_pred_oos)),
        "overall_balanced_acc": float(balanced_accuracy_score(y_true_oos, y_pred_oos)),
        "recall_by_regime": recall_by_regime,
        "feature_importance": dict(zip(feat_cols, feature_importance)),
    }


# ---------------------------------------------------------------------------
# Allocation & backtest
# ---------------------------------------------------------------------------
def blend_allocation(proba_row: pd.Series) -> Dict[str, float]:
    """Posterior-weighted blend: w = Σ_r P(r) × ALLOCATION_GRID[r]."""
    w = {"equity": 0.0, "bond": 0.0, "cash": 0.0}
    for r in REGIMES:
        p = float(proba_row[r])
        for a, v in ALLOCATION_GRID[r].items():
            w[a] += p * v
    return w


def hard_allocation(regime: str) -> Dict[str, float]:
    return dict(ALLOCATION_GRID[regime])


def backtest_allocation(proba_df: pd.DataFrame, df: pd.DataFrame,
                        cash_annual_yield: float = 0.02,
                        tc_bps: float = 5.0,
                        allocation_mode: str = "blend",
                        hard_regime_col: str = "pred") -> Dict:
    """
    Monthly rebalance backtest. allocation_mode:
      'blend' (default): posterior-weighted mixture of grids
      'hard' : use ALLOCATION_GRID[proba_df[hard_regime_col]] each month
    """
    eq_ret = df.loc[proba_df.index, "fwd_ret"].values
    bd_ret = df.loc[proba_df.index, "bond_fwd_ret"].values
    cash_ret = np.full_like(eq_ret, cash_annual_yield / 12.0)

    if allocation_mode == "hard":
        weights = [hard_allocation(r) for r in proba_df[hard_regime_col].values]
    else:
        weights = [blend_allocation(proba_df.iloc[i]) for i in range(len(proba_df))]

    w_eq = np.array([w["equity"] for w in weights])
    w_bd = np.array([w["bond"]   for w in weights])
    w_ch = np.array([w["cash"]   for w in weights])

    # Transaction cost (L1 weight change, first month starts at BENCHMARK_WEIGHTS)
    bm = BENCHMARK_WEIGHTS
    w_prev = np.vstack(
        [np.array([bm["equity"], bm["bond"], bm["cash"]])[None, :],
         np.stack([w_eq, w_bd, w_ch], axis=1)[:-1]]
    )
    w_curr = np.stack([w_eq, w_bd, w_ch], axis=1)
    turnover = np.abs(w_curr - w_prev).sum(axis=1)
    tc = turnover * (tc_bps / 10000.0)

    strat_ret = w_eq * eq_ret + w_bd * bd_ret + w_ch * cash_ret - tc
    base_ret  = bm["equity"] * eq_ret + bm["bond"] * bd_ret + bm["cash"] * cash_ret

    ann_factor = 12

    def _stats(r):
        m = float(np.mean(r) * ann_factor)
        v = float(np.std(r, ddof=1) * np.sqrt(ann_factor))
        curve = np.cumprod(1 + r)
        peak = np.maximum.accumulate(curve)
        dd = float((curve / peak - 1).min())
        return {"ann_return": m, "ann_vol": v,
                "sharpe": m / v if v > 0 else float("nan"),
                "max_dd": dd}

    return {
        "n_months": int(len(strat_ret)),
        "strategy": {**_stats(strat_ret),
                     "mean_turnover_pct": float(np.mean(turnover) * 100)},
        "benchmark": _stats(base_ret),
        "alpha_ann": float(np.mean(strat_ret - base_ret) * ann_factor),
        "tracking_error": float(np.std(strat_ret - base_ret, ddof=1) * np.sqrt(ann_factor)),
    }


# ---------------------------------------------------------------------------
# Unified variant runner (used by ablation_harness)
# ---------------------------------------------------------------------------
def run_variant(cfg: VariantConfig, df: Optional[pd.DataFrame] = None,
                feat_cols: Optional[List[str]] = None) -> Dict:
    """Single-variant end-to-end: load → train/CV → backtest → return summary dict."""
    if df is None:
        df, full_feats = load_dataset(cfg.dataset_path)
    else:
        full_feats = [c for c in df.columns if c not in RESERVED_COLS]
    feat_cols = cfg.feature_cols if cfg.feature_cols is not None else (
        feat_cols if feat_cols is not None else full_feats
    )
    # Keep only rows where all requested features are valid
    sub = df[feat_cols + ["regime", "fwd_ret", "bond_fwd_ret"]].dropna()
    result = train_and_evaluate(sub, feat_cols, cfg)

    proba_df = result["proba_df"]
    # Mean entropy (uncertainty) of OOS probability distribution
    p = proba_df[REGIMES].values
    p = np.clip(p, 1e-12, 1.0)
    mean_entropy = float(np.mean([sp_entropy(row, base=2) for row in p]))

    # Optional meta-labeling wrapper (P4)
    meta_info = None
    hard_regime_col = None
    allocation_mode = "blend"
    if cfg.use_meta:
        from meta_labeling import apply_meta_filter
        meta_info = apply_meta_filter(sub, feat_cols, result, cfg)
        proba_df = meta_info["proba_df"]
        hard_regime_col = "final_regime"
        allocation_mode = "hard"

    backtest = backtest_allocation(
        proba_df, sub, tc_bps=cfg.tc_bps,
        allocation_mode=allocation_mode,
        hard_regime_col=hard_regime_col or "pred",
    )

    summary = {
        "variant":        cfg.name,
        "n_features":     len(feat_cols),
        "n_rows":         int(len(sub)),
        "cv_mode":        cfg.cv_mode,
        "bull_threshold": cfg.bull_threshold if cfg.bull_threshold is not None else "argmax",
        "class_weight":   str(cfg.class_weight),
        "use_meta":       cfg.use_meta,
        "sharpe":         backtest["strategy"]["sharpe"],
        "ann_return":     backtest["strategy"]["ann_return"],
        "ann_vol":        backtest["strategy"]["ann_vol"],
        "max_dd":         backtest["strategy"]["max_dd"],
        "alpha_ann":      backtest["alpha_ann"],
        "tracking_error": backtest["tracking_error"],
        "mean_turnover_pct": backtest["strategy"]["mean_turnover_pct"],
        "accuracy":       result["overall_accuracy"],
        "balanced_acc":   result["overall_balanced_acc"],
        "bull_recall":    result["recall_by_regime"].get("BULL", 0.0),
        "base_recall":    result["recall_by_regime"].get("BASE", 0.0),
        "bear_recall":    result["recall_by_regime"].get("BEAR", 0.0),
        "mean_proba_entropy": mean_entropy,
        "benchmark_sharpe":   backtest["benchmark"]["sharpe"],
        "benchmark_ann_return": backtest["benchmark"]["ann_return"],
        "benchmark_max_dd":     backtest["benchmark"]["max_dd"],
    }
    return {"summary": summary, "backtest": backtest,
            "result": result, "proba_df": proba_df, "meta_info": meta_info}


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------
def print_report(df: pd.DataFrame, result: Dict, backtest: Dict, cfg: VariantConfig):
    print("=" * 72)
    print(f" Variant: {cfg.name}  (cv={cfg.cv_mode}, bull_threshold={cfg.bull_threshold})")
    print("=" * 72)
    print(f"Dataset: {len(df)} rows, {df.index.min().date()} → {df.index.max().date()}")
    for r in REGIMES:
        cnt = int((df["regime"] == r).sum())
        print(f"   {r:<6} {cnt:>4}  ({cnt/len(df)*100:>5.1f}%)")

    print("\n--- Fold Metrics ---")
    for m in result["fold_metrics"]:
        print(f"Fold {m['fold']}: train {m['train_n']:>3} ({m['train_period'][0]}→{m['train_period'][1]})"
              f"  |  test {m['test_n']:>3} ({m['test_period'][0]}→{m['test_period'][1]})"
              f"  |  acc={m['accuracy']:.3f}  bal_acc={m['balanced_acc']:.3f}")

    print(f"\nOverall OOS accuracy:          {result['overall_accuracy']:.3f}")
    print(f"Overall OOS balanced accuracy: {result['overall_balanced_acc']:.3f}")
    print(f"Recall — BEAR {result['recall_by_regime']['BEAR']:.3f}  "
          f"BASE {result['recall_by_regime']['BASE']:.3f}  "
          f"BULL {result['recall_by_regime']['BULL']:.3f}")

    print("\n--- Confusion Matrix (OOS) ---")
    print(result["confusion"].to_string())

    print("\n--- Feature Importance (mean gain across folds, top) ---")
    fi = sorted(result["feature_importance"].items(), key=lambda x: -x[1])
    for name, imp in fi[:20]:
        print(f"   {name:<26} {imp:>8.1f}")

    print("\n--- TAA Backtest ---")
    s = backtest["strategy"]; b = backtest["benchmark"]
    print(f"  Months: {backtest['n_months']}")
    print(f"  Strategy  ret={s['ann_return']*100:>6.2f}%  vol={s['ann_vol']*100:>6.2f}%  "
          f"Sharpe={s['sharpe']:>5.2f}  MaxDD={s['max_dd']*100:>6.2f}%  "
          f"Turn={s['mean_turnover_pct']:>5.2f}%/mo")
    print(f"  Bench     ret={b['ann_return']*100:>6.2f}%  vol={b['ann_vol']*100:>6.2f}%  "
          f"Sharpe={b['sharpe']:>5.2f}  MaxDD={b['max_dd']*100:>6.2f}%")
    print(f"  Alpha: {backtest['alpha_ann']*100:>+6.2f}%  TE: {backtest['tracking_error']*100:.2f}%")


if __name__ == "__main__":
    cfg = VariantConfig(name="baseline_walkforward", cv_mode="walkforward")
    out = run_variant(cfg)
    df, _ = load_dataset(cfg.dataset_path)
    print_report(df.loc[out["proba_df"].index], out["result"], out["backtest"], cfg)
    out["proba_df"].to_csv("regime_predictions.csv")
    print("\nSaved OOS regime probabilities → regime_predictions.csv")
