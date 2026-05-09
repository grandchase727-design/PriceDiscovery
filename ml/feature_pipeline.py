###############################################################################
# Feature Pipeline for Forward 1M Regime Signal
# ============================================================================
# Builds monthly feature matrix for Bull/Base/Bear regime classifier:
#   - Target: forward 21-day ACWI return + max drawdown + contemporaneous VIX
#   - Features (bottom-up):  Price Discovery breadth/dispersion aggregates
#   - Features (top-down):   macro / cross-asset / vol / credit
#   - Labels: BULL / BASE / BEAR via multi-criteria triple-barrier rule
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple, Optional


# Index proxies (yfinance-available ETFs/indices)
PROXY_TICKERS = {
    "acwi":  "ACWI",        # iShares MSCI ACWI ETF (2008-03-)
    "agg":   "AGG",         # iShares Core U.S. Aggregate Bond (2003-09-) — proxy for Global Agg
    "bndw":  "BNDW",        # Vanguard Total World Bond (2018-09-) — true Global Agg where available
    "spy":   "SPY",         # pre-2008 equity proxy
    "vix":   "^VIX",        # CBOE VIX (1990-)
    "vxv":   "^VIX3M",      # CBOE 3-Month Volatility (2007-12-) — VIX term structure
    "move":  "^MOVE",       # ICE BofA MOVE (bond vol, sparse history)
    "tnx":   "^TNX",        # CBOE 10Y Treasury yield
    "fvx":   "^FVX",        # CBOE 5Y Treasury yield
    "dxy":   "DX-Y.NYB",    # ICE US Dollar Index
    "hyg":   "HYG",         # iShares HY Credit (OAS proxy via spread-to-LQD)
    "lqd":   "LQD",         # iShares IG Credit
    "tip":   "TIP",         # iShares TIPS (inflation-protected)
    "ief":   "IEF",         # iShares 7-10Y Treasury (nominal; paired with TIP for breakeven)
    "gold":  "GC=F",        # Gold futures
    "copper":"HG=F",        # Copper futures
    "wti":   "CL=F",        # Crude futures
}


def download_macro_panel(start: str = "1990-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Download adjusted-close panel for all PROXY_TICKERS."""
    frames = {}
    for key, ticker in PROXY_TICKERS.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                print(f"  [skip] {ticker}: empty")
                continue
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            s = df[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            frames[key] = s
        except Exception as e:
            print(f"  [err]  {ticker}: {e}")
    panel = pd.DataFrame(frames).sort_index()
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    return panel


def build_acwi_series(panel: pd.DataFrame) -> pd.Series:
    """
    Unified equity index series.
    ACWI from its 2008 inception; SPY used as pre-2008 proxy (splice on first ACWI day).
    """
    acwi = panel["acwi"].dropna()
    spy  = panel["spy"].dropna()
    if acwi.empty:
        return spy
    splice_date = acwi.index[0]
    spy_pre = spy.loc[:splice_date].iloc[:-1]
    if spy_pre.empty:
        return acwi
    # Scale SPY to meet ACWI at splice
    scale = acwi.iloc[0] / spy.loc[splice_date] if splice_date in spy.index else None
    if scale is None or not np.isfinite(scale):
        return acwi
    spy_scaled = spy_pre * scale
    return pd.concat([spy_scaled, acwi]).sort_index()


def build_bond_series(panel: pd.DataFrame) -> pd.Series:
    """
    Unified bond index series.
    BNDW (true Global Agg) where available; AGG (US Agg) as pre-2018 proxy spliced backward.
    """
    bndw = panel["bndw"].dropna() if "bndw" in panel else pd.Series(dtype=float)
    agg  = panel["agg"].dropna()  if "agg"  in panel else pd.Series(dtype=float)
    if bndw.empty:
        return agg
    if agg.empty:
        return bndw
    splice_date = bndw.index[0]
    agg_pre = agg.loc[:splice_date].iloc[:-1]
    if agg_pre.empty:
        return bndw
    scale = bndw.iloc[0] / agg.loc[splice_date] if splice_date in agg.index else None
    if scale is None or not np.isfinite(scale):
        return bndw
    agg_scaled = agg_pre * scale
    return pd.concat([agg_scaled, bndw]).sort_index()


def compute_forward_metrics(price: pd.Series, horizon: int = 21) -> pd.DataFrame:
    """
    Forward-looking metrics over next `horizon` trading days:
      - fwd_ret:   (P_{t+h} / P_t) - 1
      - fwd_dd:    min running drawdown within (t, t+h]
      - fwd_vol:   realized vol of daily log returns within the window (annualized)
    """
    logp = np.log(price.astype(float))
    fwd_ret = np.exp(logp.shift(-horizon) - logp) - 1

    arr = price.values.astype(float)
    n = len(arr)
    fwd_dd = np.full(n, np.nan)
    fwd_vol = np.full(n, np.nan)
    logr = np.diff(np.log(arr))
    for i in range(n - 1):
        end = min(i + horizon + 1, n)
        window = arr[i + 1:end]
        if len(window) == 0:
            continue
        running_max = np.maximum.accumulate(window)
        dd = (window / running_max - 1.0).min()
        fwd_dd[i] = dd
        # realized vol of daily log returns within [i+1, end-1]
        if end - 1 - i >= 2:
            seg = logr[i:end - 1]
            if len(seg) >= 2:
                fwd_vol[i] = seg.std(ddof=1) * np.sqrt(252)

    out = pd.DataFrame(
        {"fwd_ret": fwd_ret.values, "fwd_dd": fwd_dd, "fwd_vol": fwd_vol},
        index=price.index,
    )
    return out


def label_regime(fwd_ret: float, fwd_dd: float, vix_level: float,
                 bull_ret: float = 0.03, bear_ret: float = -0.03,
                 bull_dd: float = -0.02, bear_dd: float = -0.05,
                 bull_vix: float = 25.0, bear_vix: float = 30.0) -> Optional[str]:
    """
    Triple-barrier multi-criteria labeling (Option B — VIX<25 for BULL).

    BULL: fwd_ret > +3% AND fwd_dd > -2% AND VIX < 25
    BEAR: fwd_ret < -3% OR  fwd_dd < -5% OR  VIX > 30
    BASE: otherwise
    """
    if any(pd.isna(x) for x in (fwd_ret, fwd_dd, vix_level)):
        return None
    if fwd_ret > bull_ret and fwd_dd > bull_dd and vix_level < bull_vix:
        return "BULL"
    if fwd_ret < bear_ret or fwd_dd < bear_dd or vix_level > bear_vix:
        return "BEAR"
    return "BASE"


def build_regime_labels(panel: pd.DataFrame, horizon: int = 21,
                        resample_freq: str = "ME",
                        use_vix_in_label: bool = True) -> pd.DataFrame:
    """
    Construct monthly label frame.

    Returns: DataFrame indexed by month-end with columns
      [acwi_px, vix, fwd_ret, fwd_dd, fwd_vol, bond_fwd_ret, regime]

    If use_vix_in_label=False, VIX threshold is disabled
    (label uses only fwd_ret + fwd_dd) — for P2 VIX-free ablation.
    """
    acwi = build_acwi_series(panel)
    bond = build_bond_series(panel)
    vix  = panel["vix"].reindex(acwi.index).ffill()

    fwd = compute_forward_metrics(acwi, horizon=horizon)
    bond_fwd = compute_forward_metrics(bond, horizon=horizon)[["fwd_ret"]].rename(
        columns={"fwd_ret": "bond_fwd_ret"}
    )
    daily = pd.concat(
        [acwi.rename("acwi_px"), vix.rename("vix"), fwd, bond_fwd], axis=1
    )

    # Month-end snapshot
    monthly = daily.resample(resample_freq).last().dropna(subset=["acwi_px", "vix"])
    if use_vix_in_label:
        regimes = [
            label_regime(r, d, v)
            for r, d, v in zip(monthly["fwd_ret"], monthly["fwd_dd"], monthly["vix"])
        ]
    else:
        regimes = [
            label_regime(r, d, v, bull_vix=1e9, bear_vix=1e9)
            for r, d, v in zip(monthly["fwd_ret"], monthly["fwd_dd"], monthly["vix"])
        ]
    monthly["regime"] = regimes
    return monthly


def build_feature_matrix(panel: pd.DataFrame,
                         resample_freq: str = "ME") -> pd.DataFrame:
    """
    Construct month-end feature matrix (all features observable at time t).

    Feature design (15 features, macro + cross-asset + own-trend):
      Vol / Risk        : vix_level, vix_z252, vix_chg_1m
      Rates / Curve     : tnx_level, slope_5s10s, tnx_chg_1m
      FX / Commodity    : dxy_ret_3m, copper_gold_ratio, gold_ret_3m
      Credit            : hyg_lqd_diff_3m
      Equity own-trend  : acwi_ret_1m, acwi_ret_12_1m, acwi_vs_sma200,
                          acwi_rvol_21d
      Bond own-trend    : agg_vs_sma200
    """
    acwi = build_acwi_series(panel)
    agg  = build_bond_series(panel)
    vix  = panel["vix"].reindex(acwi.index).ffill()
    vxv  = panel["vxv"].reindex(acwi.index).ffill() if "vxv" in panel else pd.Series(dtype=float, index=acwi.index)
    move = panel["move"].reindex(acwi.index).ffill() if "move" in panel else pd.Series(dtype=float, index=acwi.index)
    tnx  = panel["tnx"].reindex(acwi.index).ffill()
    fvx  = panel["fvx"].reindex(acwi.index).ffill()
    dxy  = panel["dxy"].reindex(acwi.index).ffill()
    gold = panel["gold"].reindex(acwi.index).ffill()
    cop  = panel["copper"].reindex(acwi.index).ffill()
    wti  = panel["wti"].reindex(acwi.index).ffill()
    hyg  = panel["hyg"].reindex(acwi.index).ffill()
    lqd  = panel["lqd"].reindex(acwi.index).ffill()
    tip  = panel["tip"].reindex(acwi.index).ffill() if "tip" in panel else pd.Series(dtype=float, index=acwi.index)
    ief  = panel["ief"].reindex(acwi.index).ffill() if "ief" in panel else pd.Series(dtype=float, index=acwi.index)

    def _ret(s, n):
        return s / s.shift(n) - 1

    feats = pd.DataFrame(index=acwi.index)
    # --- Baseline 15 features (v1) ---
    feats["vix_level"]      = vix
    feats["vix_z252"]       = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    feats["vix_chg_1m"]     = vix - vix.shift(21)
    feats["tnx_level"]      = tnx
    feats["slope_5s10s"]    = tnx - fvx
    feats["tnx_chg_1m"]     = tnx - tnx.shift(21)
    feats["dxy_ret_3m"]     = _ret(dxy, 63)
    feats["copper_gold_ratio"] = cop / gold
    feats["gold_ret_3m"]    = _ret(gold, 63)
    feats["hyg_lqd_diff_3m"] = _ret(hyg, 63) - _ret(lqd, 63)
    feats["acwi_ret_1m"]    = _ret(acwi, 21)
    feats["acwi_ret_12_1m"] = acwi.shift(21) / acwi.shift(252) - 1
    feats["acwi_vs_sma200"] = acwi / acwi.rolling(200).mean() - 1
    log_ret_acwi = np.log(acwi / acwi.shift(1))
    feats["acwi_rvol_21d"]  = log_ret_acwi.rolling(21).std() * np.sqrt(252)
    feats["agg_vs_sma200"]  = agg / agg.rolling(200).mean() - 1

    # --- P3 additions: +6 cross-asset / term-structure features ---
    # VIX/MOVE cross-asset vol ratio (equity vol vs bond vol)
    feats["vix_move_ratio"]    = vix / move
    # VIX/VXV term structure: >1 = backwardation (stress), <1 = normal
    feats["vix_vxv_ratio"]     = vix / vxv
    # WTI/Copper: inflation-sensitive vs growth-sensitive commodity
    feats["wti_copper_ratio"]  = np.log(wti / cop)
    # DXY z-score — mean-reversion / risk-off signal
    feats["dxy_z252"]          = (dxy - dxy.rolling(252).mean()) / dxy.rolling(252).std()
    # HY/IG credit spread level (price ratio — HYG lower vs LQD = wider spread)
    feats["hy_ig_spread_level"] = hyg / lqd
    # TIP/IEF inflation breakeven proxy (higher = inflation expectations rising)
    feats["tip_ief_ratio"]     = tip / ief

    monthly = feats.resample(resample_freq).last()

    # ── Daily-resolution / intra-month features (DISABLED, see PLAN C) ──
    # Empirically, adding 10 path-dependent features (vix_max_in_month,
    # acwi_max_dd_in_month, etc.) DEGRADED P4 Sharpe 0.856 → 0.83 due to
    # primary overfit on 225-month sample. Improved BEAR recall 35→42% but
    # net Alpha worsened by 50bp. Function `build_intramonth_features` retained
    # below for future selective-subset experiments (Plan B).
    # intra = build_intramonth_features(panel, resample_freq=resample_freq)
    # monthly = monthly.join(intra, how="left")
    return monthly


def build_intramonth_features(panel: pd.DataFrame,
                              resample_freq: str = "ME") -> pd.DataFrame:
    """
    Path-dependent features computed from DAILY data and resampled to month-end.

    Each value at time t uses only data ≤ t (no leakage).

    Features:
      vix_max_in_month, vix_max_minus_end   — single-day stress spikes
      acwi_max_dd_in_month                  — true intra-month drawdown
      acwi_pos_days_pct                      — trend consistency
      acwi_realized_vol_5d_of_5d            — vol of vol
      acwi_tail_days_count                   — fat-tail event count (|r| > 1.5σ)
      move_max_jump_1d                       — max 1-day MOVE jump
      breakdown_below_sma20_days             — days closing below SMA20
      vix_skew_in_month                      — daily VIX distribution skew
      corr_acwi_agg_21d                      — 21d ACWI-AGG correlation at month-end
    """
    acwi = build_acwi_series(panel)
    agg  = build_bond_series(panel)
    vix  = panel["vix"].reindex(acwi.index).ffill()
    move = panel["move"].reindex(acwi.index).ffill() if "move" in panel else pd.Series(dtype=float, index=acwi.index)

    feats = pd.DataFrame(index=acwi.index)
    grouper = pd.Grouper(freq=resample_freq)

    def _safe_max_dd(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) < 2:
            return np.nan
        running = s.cummax()
        return float((s / running - 1).min())

    def _safe_skew(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) < 3:
            return np.nan
        return float(s.skew())

    # 1) VIX path stats
    vix_max = vix.groupby(grouper).max()
    vix_end = vix.groupby(grouper).last()
    vix_max_minus_end = vix_max - vix_end

    # 2) ACWI intra-month max DD
    acwi_max_dd = acwi.groupby(grouper).apply(_safe_max_dd)

    # 3) ACWI pct positive days
    daily_ret = acwi.pct_change()
    pos_days_pct = daily_ret.gt(0).groupby(grouper).mean()

    # 4) Realized vol of vol (5d std of 5d std)
    vol5 = daily_ret.rolling(5, min_periods=3).std()
    vov = vol5.rolling(5, min_periods=3).std()
    vov_me = vov.groupby(grouper).last()

    # 5) Tail-day count: |r_t| > 1.5 × rolling 63d std
    roll_std = daily_ret.rolling(63, min_periods=21).std()
    tail_flag = (daily_ret.abs() > (1.5 * roll_std)).astype(float)
    tail_count = tail_flag.groupby(grouper).sum()

    # 6) MOVE max 1-day jump
    move_jump = move.diff().abs()
    move_max_jump = move_jump.groupby(grouper).max()

    # 7) Breakdown below SMA20 — count of days closing below SMA20 within the month
    sma20 = acwi.rolling(20, min_periods=15).mean()
    below_flag = (acwi < sma20).astype(float)
    below_days = below_flag.groupby(grouper).sum()

    # 8) VIX daily distribution skew
    vix_skew = vix.groupby(grouper).apply(_safe_skew)

    # 9) ACWI–AGG 21d rolling correlation
    agg_aligned = agg.reindex(acwi.index).ffill()
    agg_ret = agg_aligned.pct_change()
    corr_21 = daily_ret.rolling(21, min_periods=15).corr(agg_ret)
    corr_me = corr_21.groupby(grouper).last()

    out = pd.DataFrame({
        "vix_max_in_month":          vix_max,
        "vix_max_minus_end":         vix_max_minus_end,
        "acwi_max_dd_in_month":      acwi_max_dd,
        "acwi_pos_days_pct":         pos_days_pct,
        "acwi_vol_of_vol_5d":        vov_me,
        "acwi_tail_days_count":      tail_count,
        "move_max_jump_1d":          move_max_jump,
        "breakdown_below_sma20_days": below_days,
        "vix_skew_in_month":         vix_skew,
        "corr_acwi_agg_21d":         corr_me,
    })
    return out


def _load_breadth(path: str = "breadth_monthly.parquet") -> Optional[pd.DataFrame]:
    """Load pre-computed monthly breadth; return None if file missing."""
    import os
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        df = pd.read_csv(path.replace(".parquet", ".csv"), index_col=0, parse_dates=True)
    keep = [c for c in df.columns if not c.startswith("_")]
    return df[keep]


def build_dataset(panel: pd.DataFrame, horizon: int = 21,
                  resample_freq: str = "ME",
                  use_vix_in_label: bool = True,
                  include_breadth: bool = True,
                  breadth_path: str = "breadth_monthly.parquet") -> pd.DataFrame:
    """Join month-end features (at t) with regime labels (forward window [t+1, t+h])."""
    labels = build_regime_labels(panel, horizon=horizon,
                                 resample_freq=resample_freq,
                                 use_vix_in_label=use_vix_in_label)
    feats  = build_feature_matrix(panel, resample_freq=resample_freq)
    if include_breadth:
        breadth = _load_breadth(breadth_path)
        if breadth is not None:
            # Align month-ends (both ME-frequency) and join
            feats = feats.join(breadth, how="left")
    df = feats.join(
        labels[["regime", "fwd_ret", "fwd_dd", "fwd_vol", "bond_fwd_ret"]], how="inner"
    )
    df = df.dropna(subset=["regime"])  # drop rows where forward window is unknown
    return df


def distribution_summary(labels: pd.DataFrame) -> Dict:
    """Class distribution + simple diagnostics."""
    valid = labels.dropna(subset=["regime"])
    counts = valid["regime"].value_counts().to_dict()
    total = int(len(valid))
    shares = {k: counts.get(k, 0) / total for k in ["BULL", "BASE", "BEAR"]}

    # Forward return stats per regime
    per_regime = {}
    for r in ["BULL", "BASE", "BEAR"]:
        sub = valid[valid["regime"] == r]
        if len(sub) > 0:
            per_regime[r] = {
                "n": int(len(sub)),
                "fwd_ret_mean": float(sub["fwd_ret"].mean()),
                "fwd_ret_median": float(sub["fwd_ret"].median()),
                "fwd_dd_mean": float(sub["fwd_dd"].mean()),
                "vix_mean": float(sub["vix"].mean()),
            }
    return {
        "total_months": total,
        "date_range": (str(valid.index.min().date()), str(valid.index.max().date())),
        "counts": counts,
        "shares": shares,
        "per_regime_stats": per_regime,
    }


if __name__ == "__main__":
    print("=" * 70)
    print(" Forward 1M Regime Labeling — Historical Distribution Check")
    print("=" * 70)

    print("\n[1/3] Downloading macro panel (1990-01-01 → today)…")
    panel = download_macro_panel(start="1990-01-01")
    print(f"      Panel shape: {panel.shape}, columns: {list(panel.columns)}")
    print(f"      Date range:  {panel.index.min().date()} → {panel.index.max().date()}")

    print("\n[2/3] Building regime labels (21-day forward, month-end snapshot)…")
    labels = build_regime_labels(panel, horizon=21, resample_freq="ME")
    print(f"      Labeled months: {len(labels)}")

    print("\n[3/3] Class distribution & per-regime statistics:")
    summary = distribution_summary(labels)
    print(f"      Period: {summary['date_range'][0]} → {summary['date_range'][1]}")
    print(f"      Total months (with labels): {summary['total_months']}")
    print()
    print(f"      {'Regime':<8}{'Count':>8}{'Share':>10}"
          f"{'FwdRet mean':>14}{'FwdRet med':>14}{'FwdDD mean':>14}{'VIX mean':>12}")
    print(f"      {'-'*80}")
    for r in ["BULL", "BASE", "BEAR"]:
        s = summary["per_regime_stats"].get(r, {})
        if s:
            print(f"      {r:<8}{s['n']:>8}{summary['shares'][r]*100:>9.1f}%"
                  f"{s['fwd_ret_mean']*100:>13.2f}%"
                  f"{s['fwd_ret_median']*100:>13.2f}%"
                  f"{s['fwd_dd_mean']*100:>13.2f}%"
                  f"{s['vix_mean']:>12.1f}")

    out_path = "regime_labels.csv"
    labels.to_csv(out_path)
    print(f"\n      Saved labeled dataset → {out_path}")

    print("\n[4/4] Building feature matrix + joined dataset…")
    dataset = build_dataset(panel, horizon=21, resample_freq="ME")
    feat_cols = [c for c in dataset.columns if c not in ("regime", "fwd_ret", "fwd_dd", "fwd_vol")]
    print(f"      Dataset shape: {dataset.shape}  ({len(feat_cols)} features)")
    print(f"      Features: {feat_cols}")
    print(f"      First valid row: {dataset.dropna().index.min().date()}")
    print(f"      Last  valid row: {dataset.dropna().index.max().date()}")
    print(f"      Rows after dropna: {len(dataset.dropna())}")
    dataset.to_csv("regime_dataset.csv")
    print(f"      Saved → regime_dataset.csv")

    print("\n[5/5] Building VIX-free dataset (P2 ablation)…")
    dataset_novix = build_dataset(panel, horizon=21, resample_freq="ME",
                                   use_vix_in_label=False)
    print(f"      Dataset shape: {dataset_novix.shape}")
    from collections import Counter
    dist = Counter(dataset_novix["regime"].dropna())
    total = sum(dist.values())
    for r in ["BULL", "BASE", "BEAR"]:
        print(f"      {r:<6} {dist.get(r, 0):>4}  ({dist.get(r, 0)/total*100:>5.1f}%)")
    dataset_novix.to_csv("regime_dataset_novix.csv")
    print(f"      Saved → regime_dataset_novix.csv")
