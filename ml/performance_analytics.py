###############################################################################
# Performance Analytics — Institutional-style metrics for P0~P4 models
# ============================================================================
# Methodologies inspired by AQR, Bridgewater, BlackRock Aladdin, MSCI Barra,
# Man AHL reporting standards:
#
#   Absolute:       CAGR, Ann Vol, Max DD, Calmar
#   Risk-adjusted:  Sharpe, Sortino, Omega
#   vs Benchmark:   Alpha, Beta, TE, Information Ratio, Up/Down Capture
#   Rolling:        12 / 36 / 60-month CAGR, Sharpe, Vol, MaxDD
#   Distribution:   Skew, Excess Kurtosis, VaR/CVaR 5%, Monthly hit rate
#
# Variants:
#   P0 — baseline (15 macro features, default class weight, argmax)
#   P1 — P0 + 6 breadth features from price_discovery replay (21 features)
#   P2 — P0 + class weight {BULL:3} + P(BULL)>0.25 threshold override
#   P3 — P0 + 6 additional macro/cross-asset features (21 features)
#   P4 — P3 + P1 breadth + P2 weighting + meta-labeling (27 features, WINNER)
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import json
import numpy as np
import pandas as pd
from scipy import stats as sps

from ml.ml_signal_engine import (VariantConfig, run_variant, load_dataset,
                              blend_allocation, ALLOCATION_GRID,
                              BENCHMARK_WEIGHTS, BENCHMARK_LABEL, REGIMES)


# ─────────────────────────────────────────────────────────────────────────────
# Feature groups
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_FEATS = [
    "vix_level", "vix_z252", "vix_chg_1m",
    "tnx_level", "slope_5s10s", "tnx_chg_1m",
    "dxy_ret_3m", "copper_gold_ratio", "gold_ret_3m", "hyg_lqd_diff_3m",
    "acwi_ret_1m", "acwi_ret_12_1m", "acwi_vs_sma200", "acwi_rvol_21d",
    "agg_vs_sma200",
]
P3_MACRO_FEATS = [
    "vix_move_ratio", "vix_vxv_ratio", "wti_copper_ratio",
    "dxy_z252", "hy_ig_spread_level", "tip_ief_ratio",
]
# Daily-resolution path-dependent features — DEFINED but NOT USED (Plan C rollback)
# Kept for future selective-subset experiments (see feature_pipeline.py note).
DAILY_FEATS = [
    "vix_max_in_month", "vix_max_minus_end",
    "acwi_max_dd_in_month", "acwi_pos_days_pct",
    "acwi_vol_of_vol_5d", "acwi_tail_days_count",
    "move_max_jump_1d", "breakdown_below_sma20_days",
    "vix_skew_in_month", "corr_acwi_agg_21d",
]
BREADTH_FEATS = [
    "eq_pct_bullish", "eq_pct_downtrend", "eq_tcs_median", "eq_rss_std",
    "bd_pct_bullish", "bd_tcs_median",
]

# Plan C: rolled back to monthly-only features. P4 Sharpe restored to ~0.856.
VARIANTS = {
    "P0": VariantConfig(
        name="P0_baseline",
        feature_cols=BASELINE_FEATS,
        class_weight="balanced", bull_threshold=None,
        cv_mode="walkforward", use_meta=False,
    ),
    "P1": VariantConfig(
        name="P1_breadth",
        feature_cols=BASELINE_FEATS + BREADTH_FEATS,
        class_weight="balanced", bull_threshold=None,
        cv_mode="walkforward", use_meta=False,
    ),
    "P2": VariantConfig(
        name="P2_bull_detection",
        feature_cols=BASELINE_FEATS,
        class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 3.0}, bull_threshold=0.25,
        cv_mode="walkforward", use_meta=False,
    ),
    "P3": VariantConfig(
        name="P3_macro_extended",
        feature_cols=BASELINE_FEATS + P3_MACRO_FEATS,
        class_weight="balanced", bull_threshold=None,
        cv_mode="walkforward", use_meta=False,
    ),
    "P4": VariantConfig(
        name="P4_meta_full",
        feature_cols=BASELINE_FEATS + P3_MACRO_FEATS + BREADTH_FEATS,
        class_weight={"BEAR": 1.0, "BASE": 1.0, "BULL": 3.0}, bull_threshold=0.25,
        cv_mode="walkforward", use_meta=True,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Return extraction per variant
# ─────────────────────────────────────────────────────────────────────────────
def extract_monthly_returns(out, df, tc_bps: float = 5.0) -> pd.DataFrame:
    """Monthly strategy & benchmark returns for one variant."""
    proba = out["proba_df"]
    eq = df.loc[proba.index, "fwd_ret"].values
    bd = df.loc[proba.index, "bond_fwd_ret"].values
    ch = np.full_like(eq, 0.02 / 12.0)

    if "final_regime" in proba.columns:
        weights = [ALLOCATION_GRID[r] for r in proba["final_regime"].values]
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
        "strat": strat, "bench": bench, "turnover": turn,
        "w_equity": w_eq, "w_bond": w_bd, "w_cash": w_ch,
    }, index=proba.index)


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive metrics
# ─────────────────────────────────────────────────────────────────────────────
def comprehensive_metrics(strat: np.ndarray, bench: np.ndarray,
                          freq: int = 12) -> dict:
    n = len(strat)
    s = np.asarray(strat, dtype=float)
    b = np.asarray(bench, dtype=float)
    excess = s - b

    # Absolute
    cagr  = float((1 + s).prod() ** (freq / n) - 1) if n > 0 else np.nan
    cagr_b = float((1 + b).prod() ** (freq / n) - 1) if n > 0 else np.nan
    vol   = float(np.std(s, ddof=1) * np.sqrt(freq)) if n > 1 else np.nan
    vol_b = float(np.std(b, ddof=1) * np.sqrt(freq)) if n > 1 else np.nan

    # Drawdown
    curve = np.cumprod(1 + s)
    peak  = np.maximum.accumulate(curve)
    dd    = curve / peak - 1
    max_dd = float(dd.min())
    curve_b = np.cumprod(1 + b)
    peak_b  = np.maximum.accumulate(curve_b)
    max_dd_b = float((curve_b / peak_b - 1).min())

    # Drawdown duration (longest underwater streak in months)
    underwater = dd < 0
    streaks = []
    cur = 0
    for u in underwater:
        if u:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    max_dd_duration = int(max(streaks)) if streaks else 0

    # Risk-adjusted
    sharpe = float(np.mean(s) / np.std(s, ddof=1) * np.sqrt(freq)) if np.std(s, ddof=1) > 0 else np.nan
    sharpe_b = float(np.mean(b) / np.std(b, ddof=1) * np.sqrt(freq)) if np.std(b, ddof=1) > 0 else np.nan

    downside = s[s < 0]
    dn_std = float(np.std(downside, ddof=1) * np.sqrt(freq)) if len(downside) > 1 else np.nan
    sortino = float(np.mean(s) * freq / dn_std) if dn_std and dn_std > 0 else np.nan

    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else np.nan

    # Omega ratio (threshold 0)
    gains = s[s > 0].sum()
    losses = -s[s < 0].sum()
    omega = float(gains / losses) if losses > 0 else np.nan

    # vs Benchmark
    alpha_ann = float(np.mean(excess) * freq)
    te = float(np.std(excess, ddof=1) * np.sqrt(freq)) if len(excess) > 1 else np.nan
    ir = float(alpha_ann / te) if te and te > 0 else np.nan

    # Beta / correlation
    if n > 1 and np.std(b, ddof=1) > 0:
        cov = np.cov(s, b, ddof=1)
        beta = float(cov[0, 1] / cov[1, 1])
        corr = float(np.corrcoef(s, b)[0, 1])
    else:
        beta = corr = np.nan

    # Up/Down capture
    up_m = b > 0
    dn_m = b < 0
    # Geometric capture (standard formula)
    def _geom_capture(strat_r, bench_r, mask):
        if not mask.any():
            return np.nan
        n_m = mask.sum()
        strat_ann = (1 + strat_r[mask]).prod() ** (freq / n_m) - 1
        bench_ann = (1 + bench_r[mask]).prod() ** (freq / n_m) - 1
        if bench_ann == 0:
            return np.nan
        return float(strat_ann / bench_ann)
    up_capture = _geom_capture(s, b, up_m)
    dn_capture = _geom_capture(s, b, dn_m)

    # Win rates
    win_rate = float((s > 0).mean())
    excess_win = float((excess > 0).mean())

    # Distribution
    skew = float(sps.skew(s, bias=False)) if n > 2 else np.nan
    kurt_excess = float(sps.kurtosis(s, fisher=True, bias=False)) if n > 3 else np.nan

    # VaR / CVaR (historical 5%, monthly)
    var_5 = float(np.percentile(s, 5))
    cvar_5 = float(s[s <= var_5].mean()) if (s <= var_5).any() else np.nan

    # Jensen's alpha (monthly; arithmetic against beta-adjusted bench)
    if not np.isnan(beta):
        jensens_alpha_m = float(np.mean(s) - beta * np.mean(b))
        jensens_alpha_ann = jensens_alpha_m * freq
    else:
        jensens_alpha_ann = np.nan

    # Treynor
    treynor = float((np.mean(s) * freq) / beta) if beta and not np.isnan(beta) and beta != 0 else np.nan

    return {
        "n_months": int(n),
        # Absolute
        "cagr": cagr, "cagr_bench": cagr_b,
        "ann_vol": vol, "ann_vol_bench": vol_b,
        "max_dd": max_dd, "max_dd_bench": max_dd_b,
        "max_dd_duration_months": max_dd_duration,
        # Risk-adjusted
        "sharpe": sharpe, "sharpe_bench": sharpe_b,
        "sortino": sortino, "calmar": calmar, "omega": omega,
        # vs Benchmark
        "alpha_ann": alpha_ann, "tracking_error": te,
        "information_ratio": ir, "beta": beta, "correlation": corr,
        "jensens_alpha_ann": jensens_alpha_ann, "treynor": treynor,
        "up_capture": up_capture, "dn_capture": dn_capture,
        # Win rates
        "win_rate": win_rate, "excess_win_rate": excess_win,
        # Distribution
        "skew": skew, "kurt_excess": kurt_excess,
        "var_5_monthly": var_5, "cvar_5_monthly": cvar_5,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rolling metrics
# ─────────────────────────────────────────────────────────────────────────────
def rolling_metrics(strat: pd.Series, bench: pd.Series,
                    window: int, freq: int = 12) -> pd.DataFrame:
    """Rolling annualized return, Sharpe, vol, rolling max DD (window-based)."""
    idx = strat.index
    n = len(strat)
    cagr = pd.Series(index=idx, dtype=float)
    vol  = pd.Series(index=idx, dtype=float)
    shar = pd.Series(index=idx, dtype=float)
    maxdd = pd.Series(index=idx, dtype=float)
    alpha = pd.Series(index=idx, dtype=float)
    ir    = pd.Series(index=idx, dtype=float)

    for i in range(window - 1, n):
        seg = strat.iloc[i - window + 1 : i + 1].values
        seg_b = bench.iloc[i - window + 1 : i + 1].values
        if len(seg) < window:
            continue
        cagr_v = float((1 + seg).prod() ** (freq / window) - 1)
        vol_v  = float(np.std(seg, ddof=1) * np.sqrt(freq))
        cagr.iloc[i] = cagr_v
        vol.iloc[i] = vol_v
        shar.iloc[i] = cagr_v / vol_v if vol_v > 0 else np.nan
        # Within-window MaxDD
        curve = np.cumprod(1 + seg)
        peak = np.maximum.accumulate(curve)
        maxdd.iloc[i] = float((curve / peak - 1).min())
        # Rolling alpha vs benchmark (arithmetic)
        excess = seg - seg_b
        a_ann = float(np.mean(excess) * freq)
        te_v = float(np.std(excess, ddof=1) * np.sqrt(freq))
        alpha.iloc[i] = a_ann
        ir.iloc[i] = a_ann / te_v if te_v > 0 else np.nan

    return pd.DataFrame({
        f"cagr_{window}m": cagr,
        f"vol_{window}m":  vol,
        f"sharpe_{window}m": shar,
        f"maxdd_{window}m": maxdd,
        f"alpha_{window}m": alpha,
        f"ir_{window}m":    ir,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def generate():
    print("=" * 84)
    print(" Performance Analytics — running P0~P4 variants")
    print("=" * 84)

    df_main, _ = load_dataset("regime_dataset.csv")

    variant_returns = {}
    variant_metrics = {}
    for tag, cfg in VARIANTS.items():
        print(f"\n[{tag}] {cfg.name}…")
        out = run_variant(cfg)
        ret = extract_monthly_returns(out, df_main.loc[out["proba_df"].index])
        variant_returns[tag] = ret
        strat = ret["strat"].values
        bench = ret["bench"].values
        m = comprehensive_metrics(strat, bench)
        m["period_start"] = str(ret.index.min().date())
        m["period_end"]   = str(ret.index.max().date())
        m["mean_turnover_pct"] = float(ret["turnover"].mean() * 100)
        variant_metrics[tag] = m
        print(f"  n={m['n_months']}  Sharpe={m['sharpe']:.2f}  "
              f"CAGR={m['cagr']*100:.2f}%  Alpha={m['alpha_ann']*100:+.2f}%  "
              f"MaxDD={m['max_dd']*100:.1f}%")

    # ── Persist monthly returns (long format for dashboard) ──
    long_rows = []
    for tag, ret in variant_returns.items():
        for dt, row in ret.iterrows():
            long_rows.append({
                "date": dt.date().isoformat(),
                "variant": tag,
                "strat_ret": float(row["strat"]),
                "bench_ret": float(row["bench"]),
                "w_equity": float(row["w_equity"]),
                "w_bond": float(row["w_bond"]),
                "w_cash": float(row["w_cash"]),
            })
    pd.DataFrame(long_rows).to_csv("ai_perf_monthly.csv", index=False)
    print(f"\n[save] ai_perf_monthly.csv ({len(long_rows)} rows)")

    # ── Rolling metrics: 12, 36, 60 month windows ──
    rolling_rows = []
    for tag, ret in variant_returns.items():
        strat = ret["strat"]
        bench = ret["bench"]
        for window in [12, 36, 60]:
            if len(strat) < window:
                continue
            rm = rolling_metrics(strat, bench, window)
            rm = rm.dropna(how="all")
            for dt, row in rm.iterrows():
                rolling_rows.append({
                    "date": dt.date().isoformat(),
                    "variant": tag,
                    "window": window,
                    **{k: (float(v) if pd.notna(v) else None) for k, v in row.items()},
                })
    pd.DataFrame(rolling_rows).to_csv("ai_perf_rolling.csv", index=False)
    print(f"[save] ai_perf_rolling.csv ({len(rolling_rows)} rows)")

    # ── Summary JSON ──
    summary = {
        "benchmark_label": BENCHMARK_LABEL,
        "benchmark_weights": BENCHMARK_WEIGHTS,
        "variants": variant_metrics,
        "variant_order": ["P0", "P1", "P2", "P3", "P4"],
    }
    with open("ai_perf_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if pd.isna(x) else x)
    print(f"[save] ai_perf_summary.json")

    # ── Console pretty print ──
    print("\n" + "=" * 120)
    print(" Summary Metrics (full OOS per variant)")
    print("=" * 120)
    cols = [("CAGR", "cagr", "%+5.2f%%"),
            ("Vol",  "ann_vol", "%+5.2f%%"),
            ("Sharpe", "sharpe", "%.2f"),
            ("Sortino", "sortino", "%.2f"),
            ("Calmar",  "calmar", "%.2f"),
            ("MaxDD",   "max_dd", "%+5.2f%%"),
            ("Alpha",   "alpha_ann", "%+5.2f%%"),
            ("IR",      "information_ratio", "%.2f"),
            ("Beta",    "beta", "%.2f"),
            ("UpCap",   "up_capture", "%.2f"),
            ("DnCap",   "dn_capture", "%.2f"),
            ("Skew",    "skew", "%+5.2f"),
            ("Kurt",    "kurt_excess", "%+5.2f"),
            ("VaR5%",   "var_5_monthly", "%+5.2f%%"),
            ("CVaR5%",  "cvar_5_monthly", "%+5.2f%%"),
            ("n",       "n_months", "%d")]
    hdr = f"{'Variant':<10}" + "".join(f"{c[0]:>9}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for tag in ["P0", "P1", "P2", "P3", "P4"]:
        m = variant_metrics[tag]
        row = f"{tag:<10}"
        for _, key, fmt in cols:
            v = m.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row += f"{'-':>9}"
            else:
                if "%%" in fmt:
                    row += f"{fmt % (v * 100):>9}"
                else:
                    row += f"{fmt % v:>9}"
        print(row)


if __name__ == "__main__":
    generate()
