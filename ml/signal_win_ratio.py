###############################################################################
# Per-Signal Win Ratio for P0~P4
# ============================================================================
# For each variant × predicted regime, compute:
#   - n            : number of times signal fires
#   - precision    : pred regime == true regime (classification accuracy)
#   - directional  : "did the signal's directional bet pay off?"
#       BEAR pred → fwd_ret < 0    (defensive bet correct)
#       BASE pred → |fwd_ret| < 3% (no-big-move correct)
#       BULL pred → fwd_ret > 0    (offensive bet correct)
#   - fwd_ret_mean : avg ACWI 1M return when signal fires
#   - fwd_ret_pos  : pct positive forward returns when signal fires
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd

from ml.ml_signal_engine import run_variant, load_dataset
from ml.performance_analytics import VARIANTS

REGIMES = ["BEAR", "BASE", "BULL"]
BASE_RANGE = 0.03  # ±3% defines a "BASE" outcome


def compute_per_variant(tag: str, cfg, df: pd.DataFrame) -> dict:
    out = run_variant(cfg, df=df)
    proba = out["proba_df"]

    fwd = df.loc[proba.index, "fwd_ret"]
    pred = proba["final_regime"] if "final_regime" in proba.columns else proba["pred"]
    true = proba["true"]

    rows = []
    for regime in REGIMES:
        mask = pred.values == regime
        n = int(mask.sum())
        if n == 0:
            rows.append({"regime": regime, "n": 0,
                         "precision": None, "directional": None,
                         "fwd_ret_mean": None, "fwd_ret_pos_pct": None})
            continue
        true_arr = true.values[mask]
        fwd_arr = fwd.values[mask]
        precision = float((true_arr == regime).mean())
        if regime == "BEAR":
            directional = float((fwd_arr < 0).mean())
        elif regime == "BULL":
            directional = float((fwd_arr > 0).mean())
        else:  # BASE
            directional = float(((fwd_arr > -BASE_RANGE) & (fwd_arr < BASE_RANGE)).mean())
        rows.append({
            "regime": regime,
            "n": n,
            "precision": precision,
            "directional": directional,
            "fwd_ret_mean": float(np.mean(fwd_arr)),
            "fwd_ret_pos_pct": float((fwd_arr > 0).mean()),
            "fwd_ret_median": float(np.median(fwd_arr)),
        })

    n_total = int(len(proba))
    overall_acc = float((pred.values == true.values).mean())

    # Class distribution of predictions
    pred_dist = {r: int((pred.values == r).sum()) for r in REGIMES}
    true_dist = {r: int((true.values == r).sum()) for r in REGIMES}

    return {
        "variant": tag,
        "n_oos_months": n_total,
        "period_start": str(proba.index.min().date()),
        "period_end":   str(proba.index.max().date()),
        "overall_accuracy": overall_acc,
        "pred_distribution": pred_dist,
        "true_distribution": true_dist,
        "per_regime": rows,
    }


def main():
    df_main, _ = load_dataset("regime_dataset.csv")
    output = {"variants": {}, "regimes": REGIMES, "base_range": BASE_RANGE,
              "definitions": {
                  "precision": "% of times prediction matches true regime (classification accuracy)",
                  "directional_BEAR": "% of times fwd_ret < 0 when BEAR signal fires (defensive bet wins)",
                  "directional_BASE": f"% of times |fwd_ret| < {int(BASE_RANGE*100)}% when BASE signal fires (no-big-move correct)",
                  "directional_BULL": "% of times fwd_ret > 0 when BULL signal fires (offensive bet wins)",
              }}

    for tag, cfg in VARIANTS.items():
        print(f"[{tag}] {cfg.name}…")
        res = compute_per_variant(tag, cfg, df_main)
        output["variants"][tag] = res

    with open("ai_pred_winratio.json", "w") as f:
        json.dump(output, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print("\n[save] ai_pred_winratio.json")

    # Console pretty print
    print("\n" + "=" * 92)
    print(" Per-Signal Win Ratio (precision = classification, directional = bet correctness)")
    print("=" * 92)
    print(f"{'Variant':<6}{'Regime':<6}{'N':>5}{'Precision':>12}{'Directional':>14}{'AvgFwdRet':>12}{'PosPct':>10}")
    print("-" * 78)
    for tag in ["P0", "P1", "P2", "P3", "P4"]:
        v = output["variants"][tag]
        for row in v["per_regime"]:
            if row["n"] == 0:
                continue
            print(f"{tag:<6}{row['regime']:<6}{row['n']:>5}"
                  f"{row['precision']*100:>11.1f}%"
                  f"{row['directional']*100:>13.1f}%"
                  f"{row['fwd_ret_mean']*100:>+11.2f}%"
                  f"{row['fwd_ret_pos_pct']*100:>9.1f}%")
        print()


if __name__ == "__main__":
    main()
