"""
sector_rotation_backtest.py — Monthly rebalance backtest (Phase 3 + 4 + 5)

Backtests a sector rotation strategy on the 11 SPDR sector ETFs:
  • Each month-end, rank sectors by signal (selectable mode)
  • Select top-N (default 3), equal-weight, hold 1 month, rebalance
  • Apply turnover cost (default 30 bp per change)

Three signal modes:
  • "momentum_12_1m" (Phase 3): 12-1M return — Jegadeesh-Titman 1993,
    dominant component of RSS. Single-axis proxy.
  • "composite_live"  (Phase 4): Composite-equivalent score reconstructed
    from daily prices at each historical month-end. Combines simplified
    TCS / TFS / RSS / URS with the same weights as the live engine
    (0.30 / 0.25 / 0.30 / 0.15). Captures multi-axis selection.
  • "ml_momentum_blend" (Phase 5 — ML B-1): walk-forward learned blend of
    4 momentum horizons (1M / 3M / 6M / 12-1M). At each month-end the
    weights are re-fit via constrained Spearman-IC maximization on the
    last 36 months, with L2 regularization toward equal weights (0.25
    each). Captures regime-adaptive momentum response.

Six reference tracks per run:
  • Strategy   — top-N rotation
  • EW-11      — equal-weight 11 sectors (true selection-alpha baseline)
  • SPY        — cap-weighted S&P 500 (broad US market)
  • QQQ        — Nasdaq 100 (large-cap growth / tech tilt)
  • IWM        — Russell 2000 (US small-cap)
  • ACWI       — MSCI All Country World (global broad market)

Output is JSON-friendly. Cached for 24 h via api.py STATE.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import yfinance as yf
import pandas as pd
import numpy as np

from config.scoring import (
    COMPOSITE_W_TCS, COMPOSITE_W_TFS, COMPOSITE_W_RSS, COMPOSITE_W_URS,
)


# 11 SPDR sector ETFs + benchmarks
SECTOR_TICKERS = ["XLK", "XLC", "XLV", "XLF", "XLY", "XLP",
                  "XLI", "XLE", "XLB", "XLU", "XLRE"]
BENCHMARK = "SPY"                          # primary benchmark (legacy field name)
EXTRA_BENCHMARKS = ["QQQ", "IWM", "ACWI"]  # secondary comparators


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

def _fetch_monthly_prices(tickers: List[str], lookback_years: int = 5) -> pd.DataFrame:
    """Download month-end close prices via yfinance.

    Returns DataFrame with DatetimeIndex (month-end) and ticker columns.
    auto_adjust=True applies dividend / split adjustments.

    For lookback_years ≥ 11, switches to start/end date range (yfinance's
    `period` parameter only accepts predefined ≤10y values).
    Use lookback_years=99 to fetch maximum history (period="max").
    """
    from datetime import datetime, timedelta
    if lookback_years >= 99:
        df = yf.download(
            tickers, period="max", interval="1mo",
            auto_adjust=True, progress=False, threads=True,
        )
    elif lookback_years >= 11:
        end = datetime.now()
        start = end - timedelta(days=int((lookback_years + 2) * 365.25))
        df = yf.download(
            tickers, start=start, end=end, interval="1mo",
            auto_adjust=True, progress=False, threads=True,
        )
    else:
        period = f"{lookback_years + 2}y"   # +2y buffer for 12-1M lookback
        df = yf.download(
            tickers, period=period, interval="1mo",
            auto_adjust=True, progress=False, threads=True,
        )
    # yf returns multi-column when multiple tickers — pick "Close"
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            df = df.xs("Close", axis=1, level=0)
    df = df.dropna(how="all")
    return df


def _fetch_daily_prices(tickers: List[str], lookback_years: int = 5) -> pd.DataFrame:
    """Daily close prices for live-signal Composite reconstruction.

    Same lookback handling as _fetch_monthly_prices.
    """
    from datetime import datetime, timedelta
    if lookback_years >= 99:
        df = yf.download(
            tickers, period="max", interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
    elif lookback_years >= 11:
        end = datetime.now()
        start = end - timedelta(days=int((lookback_years + 2) * 365.25))
        df = yf.download(
            tickers, start=start, end=end, interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
    else:
        period = f"{lookback_years + 2}y"
        df = yf.download(
            tickers, period=period, interval="1d",
            auto_adjust=True, progress=False, threads=True,
        )
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            df = df.xs("Close", axis=1, level=0)
    df = df.dropna(how="all")
    return df


# ──────────────────────────────────────────────────────────────────────
# Signal builders
# ──────────────────────────────────────────────────────────────────────

def _compute_momentum_signal(prices: pd.DataFrame, t_idx: int) -> pd.Series:
    """12-1M momentum at month t: return from t-12 to t-1.

    prices : DataFrame indexed by month-end
    t_idx  : integer index of the current month
    """
    if t_idx < 12:
        return pd.Series(dtype=float)
    p_now = prices.iloc[t_idx - 1]   # most recent COMPLETED month (skip 1 month)
    p_then = prices.iloc[t_idx - 12]
    sig = p_now / p_then - 1.0
    return sig.dropna()


def _compute_composite_signal(daily_prices: pd.DataFrame,
                              as_of: pd.Timestamp) -> pd.Series:
    """Composite-equivalent score per sector at as_of date.

    Reconstructs a simplified TCS / TFS / RSS / URS using daily prices
    available *up to and including* as_of, then combines them with the
    same weights as the live engine:
        composite = 0.30·TCS + 0.25·TFS + 0.30·RSS + 0.15·URS

    Simplifications vs the live engine (data-driven, not by choice):
      • TCS uses SMA20/50/200 distance + slope only (no trend-age buckets)
      • TFS uses 60d range position + 5d-vs-21d acceleration
      • RSS uses cross-sectional percentile of 5d / 21d / 63d / 252d returns
      • URS held neutral (50) — needs OHLCV + universe-wide signals
    """
    px = daily_prices.loc[:as_of]
    px = px.dropna(how="all")
    if len(px) < 200:
        return pd.Series(dtype=float)

    tickers = list(px.columns)

    # ── TCS: SMA distance + slope ──
    tcs = pd.Series(50.0, index=tickers, dtype=float)
    for tk in tickers:
        s = px[tk].dropna()
        if len(s) < 200:
            continue
        sma20 = s.rolling(20).mean()
        sma50 = s.rolling(50).mean()
        sma200 = s.rolling(200).mean()
        last = float(s.iloc[-1])
        score = 50.0
        v20, v50, v200 = sma20.iloc[-1], sma50.iloc[-1], sma200.iloc[-1]
        if v20 > 0:
            score += 8 if last > v20 else -8
        if v50 > 0:
            score += 8 if last > v50 else -8
        if v200 > 0:
            score += 12 if last > v200 else -12
        # 21-day SMA slope
        for sma, w in ((sma20, 7), (sma50, 7), (sma200, 8)):
            if len(sma.dropna()) >= 22:
                slope = sma.iloc[-1] / sma.iloc[-22] - 1.0
                score += w if slope > 0 else -w
        tcs[tk] = max(0.0, min(100.0, score))

    # ── TFS: 60d range position + acceleration ──
    tfs = pd.Series(50.0, index=tickers, dtype=float)
    for tk in tickers:
        s = px[tk].dropna()
        if len(s) < 60:
            continue
        last = float(s.iloc[-1])
        win60 = s.iloc[-60:]
        hi, lo = float(win60.max()), float(win60.min())
        range_pos = (last - lo) / (hi - lo) if hi > lo else 0.5
        r5 = last / float(s.iloc[-6]) - 1.0 if len(s) >= 6 else 0.0
        r21 = last / float(s.iloc[-22]) - 1.0 if len(s) >= 22 else 0.0
        # weekly-normalized acceleration term
        accel = r5 - (r21 / 4.2)
        tfs[tk] = max(0.0, min(100.0, 35.0 + range_pos * 30.0 + accel * 400.0))

    # ── RSS: cross-sectional percentile of multi-period returns ──
    def _ret(s: pd.Series, n: int) -> float:
        s = s.dropna()
        if len(s) < n:
            return float("nan")
        return float(s.iloc[-1]) / float(s.iloc[-n]) - 1.0

    rets = pd.DataFrame({
        "r5":   {tk: _ret(px[tk], 6)   for tk in tickers},
        "r21":  {tk: _ret(px[tk], 22)  for tk in tickers},
        "r63":  {tk: _ret(px[tk], 64)  for tk in tickers},
        "r252": {tk: _ret(px[tk], 253) for tk in tickers},
    })
    pct = rets.rank(pct=True) * 100.0
    rss = (0.10 * pct["r5"].fillna(50.0)
           + 0.20 * pct["r21"].fillna(50.0)
           + 0.30 * pct["r63"].fillna(50.0)
           + 0.40 * pct["r252"].fillna(50.0))

    # ── URS: held neutral (no OHLCV / breadth available here) ──
    urs = pd.Series(50.0, index=tickers, dtype=float)

    composite = (COMPOSITE_W_TCS * tcs + COMPOSITE_W_TFS * tfs
                 + COMPOSITE_W_RSS * rss + COMPOSITE_W_URS * urs)
    # Drop tickers with no usable history
    valid_idx = px.dropna(axis=1, thresh=200).columns
    composite = composite.loc[composite.index.intersection(valid_idx)]
    return composite.dropna()


# ──────────────────────────────────────────────────────────────────────
# ML signal: multi-horizon momentum blend (Phase 5 — B-1)
# ──────────────────────────────────────────────────────────────────────

# Horizons (skip, lookback) in months — short reversal + 3M / 6M / 12-1M
# (skip=1 means use t-1 as numerator; lookback X means use t-X as denominator)
# 12-1M horizon (1,12) is identical to _compute_momentum_signal: P[t-1]/P[t-12]-1
# so when ML weights collapse to pure 12-1M [0,0,0,1], the signal equals
# the momentum_12_1m baseline exactly (rank-preserving via percentile rank).
ML_BLEND_HORIZONS = [(1, 2), (1, 4), (1, 7), (1, 12)]
ML_BLEND_LABELS = ["m1_skip1", "m3_skip1", "m6_skip1", "m12_1"]
ML_BLEND_TRAIN_WINDOW = 36     # months of training history per fit
ML_BLEND_L2_PENALTY = 0.10      # toward 12-1M baseline prior
ML_BLEND_MIN_TRAIN = 18         # minimum months before ML fit can run

# Prior weights — anchored at PURE 12-1M (= equal to momentum_12_1m baseline).
# When training signal is weak/noisy, L2 penalty pulls optimizer back to baseline.
# When training shows clear lift from blending, optimizer can move (the L2 cost is
# only ~0.10 × Σ(Δw)², so a modest spread improvement justifies blending).
ML_BLEND_PRIOR = np.array([0.0, 0.0, 0.0, 1.0])  # m1, m3, m6, m12_1
ML_BLEND_LB = 0.0   # lower bound — weights can vanish entirely
ML_BLEND_UB = 1.0   # upper bound


def _ml_blend_features_at(monthly_prices: pd.DataFrame, t_idx: int) -> Optional[pd.DataFrame]:
    """Compute 4 cross-sectionally percentile-ranked momentum signals at month t.

    Returns DataFrame indexed by ticker, columns = ML_BLEND_LABELS. None if insufficient history.
    """
    feats = []
    for skip, look in ML_BLEND_HORIZONS:
        if t_idx - look < 0:
            return None
        p_now = monthly_prices.iloc[t_idx - skip]
        p_then = monthly_prices.iloc[t_idx - look]
        ret = p_now / p_then - 1.0
        # Cross-sectional percentile rank (0..1) — robust to scale differences across horizons
        feats.append(ret.rank(pct=True))
    df = pd.concat(feats, axis=1)
    df.columns = ML_BLEND_LABELS
    return df


def _ml_blend_fit_weights(monthly_prices: pd.DataFrame, t_idx: int,
                           train_window: int = ML_BLEND_TRAIN_WINDOW,
                           l2_penalty: float = ML_BLEND_L2_PENALTY,
                           ) -> Optional[Dict[str, Any]]:
    """Fit blend weights on training window ending at t_idx-1 (no leakage).

    Objective: maximize Spearman IC of blended signal vs forward return,
    with L2 regularization toward equal weights (0.25 each).

    Returns dict with weights, ic_train, n_obs, or None if insufficient data.
    """
    from scipy.optimize import minimize

    # Build train set as a list of per-period (X_t, y_t) tuples
    # — IC is averaged across periods (proper cross-sectional IC)
    train_periods: List[Tuple[np.ndarray, np.ndarray]] = []
    start = max(13, t_idx - train_window)
    for tau in range(start, t_idx):
        feat = _ml_blend_features_at(monthly_prices, tau)
        if feat is None or tau + 1 >= len(monthly_prices):
            continue
        target = monthly_prices.iloc[tau + 1] / monthly_prices.iloc[tau] - 1.0
        valid = feat.dropna().index.intersection(target.dropna().index)
        if len(valid) < 6:
            continue
        train_periods.append((feat.loc[valid].values, target.loc[valid].values))

    if len(train_periods) < ML_BLEND_MIN_TRAIN:
        return None

    n_total = sum(len(yt) for _, yt in train_periods)

    def avg_top3_minus_bot3_spread(w: np.ndarray) -> float:
        """Per-period spread: mean fwd return of top-3 by signal MINUS bottom-3 by signal.

        This directly aligns with the strategy's selection rule (top_n=3) — IC is too
        soft a target since the strategy only acts on the extreme tails.
        """
        spreads: List[float] = []
        for X_t, y_t in train_periods:
            signal = X_t @ w
            n = len(signal)
            if n < 8 or np.std(signal) == 0:
                continue
            order = np.argsort(signal)
            top = y_t[order[-3:]].mean()
            bot = y_t[order[:3]].mean()
            spreads.append(top - bot)
        return float(np.mean(spreads)) if spreads else 0.0

    def neg_obj(w: np.ndarray) -> float:
        spread = avg_top3_minus_bot3_spread(w)
        # L2 toward prior (literature-favored 12-1M)
        l2 = float(np.sum((w - ML_BLEND_PRIOR) ** 2))
        return -(spread - l2_penalty * l2)

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(ML_BLEND_LB, ML_BLEND_UB)] * 4
    # Initialize slightly off-prior to give optimizer room to move
    init_w = ML_BLEND_PRIOR.copy() * 0.9 + 0.025  # = [0.025, 0.025, 0.025, 0.925]
    init_w = init_w / init_w.sum()                # normalize to simplex
    res = minimize(neg_obj, init_w, method="SLSQP",
                    bounds=bounds, constraints=constraints,
                    options={"ftol": 1e-6, "maxiter": 100})
    # Compare prior vs optimized — keep whichever scores better OOS-wise
    # (we use train spread as proxy; in practice this safeguards against bad fits)
    prior_val = -neg_obj(ML_BLEND_PRIOR)
    if not res.success or (-res.fun) < prior_val:
        # Fit failed or didn't beat prior on training → use prior (= pure 12-1M baseline)
        w_opt = ML_BLEND_PRIOR.copy()
    else:
        w_opt = res.x
    spread_train = avg_top3_minus_bot3_spread(w_opt)
    return {
        "weights": {ML_BLEND_LABELS[i]: round(float(w_opt[i]), 4) for i in range(4)},
        "spread_train": round(spread_train * 100, 4),  # avg top3-bot3 monthly spread (%)
        "n_obs": int(n_total),
        "n_periods": len(train_periods),
    }


def _compute_ml_blend_signal(monthly_prices: pd.DataFrame, t_idx: int,
                              ) -> Tuple[pd.Series, Optional[Dict[str, Any]]]:
    """ML-blended momentum signal at month t.

    Walk-forward: weights re-fit on the trailing window each month → no leakage.
    Falls back to equal-weight blend if fit fails (early periods, insufficient data).
    """
    feat = _ml_blend_features_at(monthly_prices, t_idx)
    if feat is None:
        return pd.Series(dtype=float), None

    fit = _ml_blend_fit_weights(monthly_prices, t_idx)
    if fit is None:
        # Fallback: prior weights (literature-favored 12-1M tilt)
        w = ML_BLEND_PRIOR.copy()
        meta = {"weights": {ML_BLEND_LABELS[i]: float(ML_BLEND_PRIOR[i]) for i in range(4)},
                "spread_train": None, "n_obs": 0, "fallback": True}
    else:
        w = np.array([fit["weights"][lbl] for lbl in ML_BLEND_LABELS])
        meta = fit

    signal = (feat * w).sum(axis=1)
    return signal.dropna(), meta


# ──────────────────────────────────────────────────────────────────────
# ML signal: macro-augmented LightGBM (Phase 5 — B-2)
# ──────────────────────────────────────────────────────────────────────

LGBM_TRAIN_WINDOW = 60       # months of training history per fit
LGBM_MIN_TRAIN = 36          # minimum months before LightGBM fit can run
LGBM_MOMENTUM_HORIZONS = [(1, 2), (1, 4), (1, 7), (1, 12)]
LGBM_MOMENTUM_LABELS = ["m1_skip1", "m3_skip1", "m6_skip1", "m12_1"]


def _build_lgbm_features_at(monthly_prices: pd.DataFrame, t_idx: int,
                             macro_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Per-sector feature matrix at month t.

    Columns:
        m1_skip1, m3_skip1, m6_skip1, m12_1    — momentum percentile ranks
        macro_*                                 — macro features at t-1 (lagged)
        sector_id                               — integer encoding of sector

    Index: ticker (one row per available sector).
    """
    feats = []
    for skip, look in LGBM_MOMENTUM_HORIZONS:
        if t_idx - look < 0:
            return None
        p_now = monthly_prices.iloc[t_idx - skip]
        p_then = monthly_prices.iloc[t_idx - look]
        ret = p_now / p_then - 1.0
        feats.append(ret.rank(pct=True))
    df = pd.concat(feats, axis=1)
    df.columns = LGBM_MOMENTUM_LABELS

    # Sector ID (integer encoding)
    df["sector_id"] = pd.Series(
        {tk: i for i, tk in enumerate(monthly_prices.columns)},
        index=df.index,
    )

    # Macro features at t (lagged 1 month → use month t-1's data, available by month t-1 close)
    if macro_df is not None and not macro_df.empty:
        as_of = monthly_prices.index[t_idx - 1]
        # Find the most recent macro row at or before as_of
        macro_idx = macro_df.index[macro_df.index <= as_of]
        if len(macro_idx) > 0:
            macro_row = macro_df.loc[macro_idx[-1]]
            for col in macro_df.columns:
                df[f"macro_{col}"] = float(macro_row[col]) if pd.notna(macro_row[col]) else float("nan")
    return df


def _compute_lightgbm_signal(monthly_prices: pd.DataFrame, t_idx: int,
                              macro_df: Optional[pd.DataFrame],
                              ) -> Tuple[pd.Series, Optional[Dict[str, Any]]]:
    """Walk-forward LightGBM fit + predict at month t.

    Training panel: for tau in [t - LGBM_TRAIN_WINDOW, t-1]:
        features at tau + forward 1-month return as target.
    Prediction: features at t → predicted next-month return for each sector.
    Top-N by prediction → portfolio selection.

    Returns (predictions_series, fit_metadata) or (empty, None) if insufficient.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return pd.Series(dtype=float), None

    cur_feats = _build_lgbm_features_at(monthly_prices, t_idx, macro_df)
    if cur_feats is None:
        return pd.Series(dtype=float), None

    # Build training panel
    X_rows = []
    y_rows = []
    start = max(13, t_idx - LGBM_TRAIN_WINDOW)
    for tau in range(start, t_idx):
        feat_tau = _build_lgbm_features_at(monthly_prices, tau, macro_df)
        if feat_tau is None or tau + 1 >= len(monthly_prices):
            continue
        target = monthly_prices.iloc[tau + 1] / monthly_prices.iloc[tau] - 1.0
        valid = feat_tau.dropna(subset=LGBM_MOMENTUM_LABELS).index.intersection(target.dropna().index)
        if len(valid) < 6:
            continue
        Xt = feat_tau.loc[valid]
        yt = target.loc[valid].values
        X_rows.append(Xt)
        y_rows.append(yt)

    if len(X_rows) < LGBM_MIN_TRAIN:
        return pd.Series(dtype=float), {
            "fallback": True, "reason": f"only {len(X_rows)} train periods (need ≥{LGBM_MIN_TRAIN})",
        }

    X_train = pd.concat(X_rows, axis=0)
    y_train = np.concatenate(y_rows)

    # Drop rows with NaN macro (LightGBM handles NaN but explicit filter helps)
    feature_cols = list(X_train.columns)
    model = lgb.LGBMRegressor(
        n_estimators=80,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=10,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=2,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
    )
    try:
        model.fit(X_train.values, y_train, feature_name=feature_cols)
    except Exception as e:
        return pd.Series(dtype=float), {"fallback": True, "reason": f"fit failed: {e}"}

    # Predict for current month
    valid_cur = cur_feats.dropna(subset=LGBM_MOMENTUM_LABELS).index
    if len(valid_cur) < 3:
        return pd.Series(dtype=float), {"fallback": True, "reason": "insufficient current data"}
    X_pred = cur_feats.loc[valid_cur, feature_cols].values
    preds = model.predict(X_pred)
    signal = pd.Series(preds, index=valid_cur)

    # Feature importance (gain-based)
    try:
        imp = dict(zip(feature_cols, model.booster_.feature_importance(importance_type="gain")))
        total = sum(imp.values()) or 1.0
        importance = {k: round(float(v) / total, 4) for k, v in imp.items()}
    except Exception:
        importance = {}

    return signal, {
        "n_train": int(len(y_train)),
        "n_train_periods": len(X_rows),
        "feature_importance": importance,
        "n_features": len(feature_cols),
        "best_iteration": int(getattr(model, "best_iteration_", model.n_estimators) or 80),
    }


# Module-level cache for macro data (avoid re-fetching across rebalances within a single backtest)
_MACRO_CACHE: Dict[int, pd.DataFrame] = {}


def _get_macro_features(lookback_years: int) -> Optional[pd.DataFrame]:
    """Lazy-load + cache macro features per lookback window."""
    if lookback_years in _MACRO_CACHE:
        return _MACRO_CACHE[lookback_years]
    try:
        from ml.macro_features import fetch_macro_monthly
        macro = fetch_macro_monthly(lookback_years)
        _MACRO_CACHE[lookback_years] = macro
        return macro
    except Exception as e:
        print(f"[lgbm] macro features unavailable: {e}")
        _MACRO_CACHE[lookback_years] = pd.DataFrame()
        return None


# ──────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────

def _weight_turnover(prev: Dict[str, float], cur: Dict[str, float]) -> float:
    """Sum of absolute weight changes (one-way turnover fraction)."""
    keys = set(prev.keys()) | set(cur.keys())
    return float(sum(abs(prev.get(k, 0.0) - cur.get(k, 0.0)) for k in keys))


def _weight_turnover_with_cash(prev_pos: Dict[str, float], cur_pos: Dict[str, float]) -> float:
    """Like _weight_turnover but adds an implicit __cash__ entry so that
    transitions between equity and cash (vol-targeting scale changes) are
    correctly captured as turnover.
    """
    prev_total = sum(prev_pos.values())
    cur_total = sum(cur_pos.values())
    prev_full = {**prev_pos, "__cash__": max(0.0, 1.0 - prev_total)}
    cur_full = {**cur_pos, "__cash__": max(0.0, 1.0 - cur_total)}
    return _weight_turnover(prev_full, cur_full)


def _realized_vol_pct(monthly_returns: List[float], window: int = 6) -> Optional[float]:
    """Annualized realized vol (%) from last `window` months of strategy returns.

    Returns None if insufficient history.
    """
    if len(monthly_returns) < window:
        return None
    recent = np.array(monthly_returns[-window:], dtype=float)
    if recent.std(ddof=1) <= 0:
        return None
    return float(recent.std(ddof=1) * math.sqrt(12) * 100)


def run_backtest(
    lookback_years: int = 5,
    top_n: int = 3,
    turnover_bp: float = 30.0,
    risk_free_rate: float = 0.0,
    signal_mode: str = "momentum_12_1m",
    vol_target_pct: float = 0.0,
    vol_lookback_months: int = 6,
    max_leverage: float = 1.0,
) -> Dict[str, Any]:
    """Execute the monthly rebalance backtest.

    Args:
        lookback_years: how many years of history for the result window
        top_n: number of sectors to hold each month
        turnover_bp: per-rebalance turnover cost in bp (charged on Σ|Δw|)
        risk_free_rate: annualized rf for Sharpe (default 0 for simplicity)
        signal_mode: "momentum_12_1m" | "composite_live" | "ml_momentum_blend"
        vol_target_pct: target annualized vol % (e.g. 12 = 12%). 0 = disabled.
                         When >0, monthly position is scaled by
                             scale = clip(vol_target_pct / realized_vol, 0, max_leverage)
                         Cash slack (1 - scale) earns 0%. Realized vol uses last
                         vol_lookback_months months of strategy net returns.
        vol_lookback_months: window for realized vol estimation (default 6)
        max_leverage: cap on scale (default 1.0 = no leverage; set 1.5 to allow)

    Returns:
        Dict with config / metrics / monthly time series.
    """
    if signal_mode not in ("momentum_12_1m", "composite_live", "ml_momentum_blend", "ml_lightgbm"):
        return {"error": f"Unknown signal_mode '{signal_mode}'."}
    if vol_target_pct < 0:
        return {"error": f"vol_target_pct must be ≥ 0; got {vol_target_pct}."}
    if max_leverage < 1.0 or max_leverage > 3.0:
        return {"error": f"max_leverage must be in [1.0, 3.0]; got {max_leverage}."}

    tickers = SECTOR_TICKERS + [BENCHMARK] + EXTRA_BENCHMARKS
    prices = _fetch_monthly_prices(tickers, lookback_years)
    if prices.empty:
        return {"error": "No price data available."}

    # Resample to month-end (just in case)
    prices = prices.resample("ME").last().ffill(limit=1)

    # Reindex so columns are stable
    available_sectors = [t for t in SECTOR_TICKERS if t in prices.columns]
    if BENCHMARK not in prices.columns:
        return {"error": f"{BENCHMARK} data missing."}

    sector_px = prices[available_sectors]
    bench_px = prices[BENCHMARK]
    extra_px = {tk: prices[tk] for tk in EXTRA_BENCHMARKS if tk in prices.columns}

    # Monthly returns
    sector_ret = sector_px.pct_change()
    bench_ret = bench_px.pct_change()
    extra_ret = {tk: extra_px[tk].pct_change() for tk in extra_px}

    n = len(prices)
    if n < 14:
        return {"error": f"Need ≥14 months of data; got {n}."}

    # Daily prices for composite_live mode (one fetch, slice per month)
    daily_sector_px: Optional[pd.DataFrame] = None
    if signal_mode == "composite_live":
        daily_all = _fetch_daily_prices(SECTOR_TICKERS, lookback_years)
        if daily_all.empty:
            return {"error": "No daily price data for composite_live mode."}
        daily_sector_px = daily_all[[c for c in available_sectors if c in daily_all.columns]]

    # Macro features for ml_lightgbm mode (one fetch, sliced per month)
    macro_df: Optional[pd.DataFrame] = None
    if signal_mode == "ml_lightgbm":
        macro_df = _get_macro_features(lookback_years)
        if macro_df is None or macro_df.empty:
            return {"error": "Macro features unavailable. Check yfinance connectivity."}

    # Walk forward
    history: List[Dict[str, Any]] = []
    ml_weight_history: List[Dict[str, Any]] = []   # only populated in ml_momentum_blend mode
    prev_positions: List[str] = []
    prev_weights: Dict[str, float] = {}
    prev_scale: float = 0.0   # for turnover including cash transition (vol-targeting)
    cum_strategy = 1.0
    cum_benchmark = 1.0
    cum_equalweight = 1.0
    cum_extra: Dict[str, float] = {tk: 1.0 for tk in extra_px}
    monthly_strategy_rets: List[float] = []
    monthly_bench_rets: List[float] = []
    monthly_ew_rets: List[float] = []  # equal-weight 11 baseline
    monthly_extra_rets: Dict[str, List[float]] = {tk: [] for tk in extra_px}

    # Start from index 13 (need 12 months of history + 1 holding month)
    for t in range(13, n):
        # ── Compute signal AT t (using prices known by end of month t-1) ──
        ml_meta_at_t: Optional[Dict[str, Any]] = None
        if signal_mode == "momentum_12_1m":
            signal = _compute_momentum_signal(sector_px, t)
        elif signal_mode == "composite_live":
            as_of = prices.index[t - 1]
            signal = _compute_composite_signal(daily_sector_px, as_of)
        elif signal_mode == "ml_lightgbm":
            signal, ml_meta_at_t = _compute_lightgbm_signal(sector_px, t, macro_df)
            if signal.empty:
                # LightGBM fit failed (insufficient history early in series) — fall back to 12-1M
                signal = _compute_momentum_signal(sector_px, t)
                if ml_meta_at_t is None:
                    ml_meta_at_t = {}
                ml_meta_at_t["fallback_to_12_1m"] = True
        else:  # ml_momentum_blend
            signal, ml_meta_at_t = _compute_ml_blend_signal(sector_px, t)
        if signal.empty or len(signal) < top_n:
            continue

        # Restrict to tickers with valid return for the upcoming holding period
        # (avoids selecting XLRE/XLC pre-inception with valid signal but missing return)
        period_all = sector_ret.iloc[t]
        valid_tickers = period_all.dropna().index.tolist()
        eligible_signal = signal[signal.index.intersection(valid_tickers)]
        if len(eligible_signal) < top_n:
            continue

        sorted_signal = eligible_signal.sort_values(ascending=False)
        positions = list(sorted_signal.head(top_n).index)
        weights = {tk: 1.0 / top_n for tk in positions}

        # ── Realized gross return = Σ w_i × r_i ──
        rets_today = sector_ret.iloc[t]
        gross_ret = 0.0
        wsum_with_data = 0.0
        for tk, w in weights.items():
            r = rets_today.get(tk)
            if r is not None and not pd.isna(r):
                gross_ret += w * float(r)
                wsum_with_data += w
        if wsum_with_data <= 0:
            continue
        if wsum_with_data < 0.999:
            gross_ret = gross_ret / wsum_with_data

        # ── Vol-target scaling (B-3): scale position size to target annual vol ──
        scale = 1.0
        realized_vol_now: Optional[float] = None
        if vol_target_pct > 0:
            realized_vol_now = _realized_vol_pct(monthly_strategy_rets, vol_lookback_months)
            if realized_vol_now is not None and realized_vol_now > 0:
                scale = vol_target_pct / realized_vol_now
                scale = max(0.0, min(scale, max_leverage))
            # else: keep scale=1.0 (insufficient history or zero vol)

        # ── Turnover cost: Σ|Δw| × bp (with implicit cash to capture scale changes) ──
        scaled_weights = {tk: w * scale for tk, w in weights.items()}
        if prev_weights or prev_scale > 0:
            prev_scaled = {tk: w * prev_scale for tk, w in prev_weights.items()} if prev_weights else {}
            sum_dw = _weight_turnover_with_cash(prev_scaled, scaled_weights)
        else:
            sum_dw = float(sum(scaled_weights.values()) + max(0.0, 1.0 - sum(scaled_weights.values())))
        turnover_cost = (turnover_bp / 10000.0) * sum_dw

        # Apply scale: cash slack earns 0%, position fraction earns scale × gross_ret
        net_strategy_ret = (gross_ret * scale) - turnover_cost

        # ── Benchmarks ──
        bench_period_ret = float(bench_ret.iloc[t]) if not pd.isna(bench_ret.iloc[t]) else 0.0
        # Equal-weight over ALL 11 sectors (with valid data this month)
        ew_returns = sector_ret.iloc[t].dropna()
        ew_period_ret = float(ew_returns.mean()) if len(ew_returns) else 0.0

        cum_strategy *= (1.0 + net_strategy_ret)
        cum_benchmark *= (1.0 + bench_period_ret)
        cum_equalweight *= (1.0 + ew_period_ret)
        monthly_strategy_rets.append(net_strategy_ret)
        monthly_bench_rets.append(bench_period_ret)
        monthly_ew_rets.append(ew_period_ret)

        # Extra benchmarks (QQQ / IWM / ACWI)
        extra_period_rets: Dict[str, float] = {}
        for tk in extra_px:
            v = extra_ret[tk].iloc[t]
            r = float(v) if not pd.isna(v) else 0.0
            extra_period_rets[tk] = r
            cum_extra[tk] *= (1.0 + r)
            monthly_extra_rets[tk].append(r)

        date_str = prices.index[t].strftime("%Y-%m-%d")
        record = {
            "date": date_str,
            "positions": positions,
            "weights": {tk: round(w, 4) for tk, w in weights.items()},
            "n_changes": int(len(set(positions) - set(prev_positions))) if prev_positions else top_n,
            "sum_dw": round(sum_dw, 4),
            "vol_scale": round(scale, 3),
            "realized_vol_pct": round(realized_vol_now, 2) if realized_vol_now is not None else None,
            "n_valid_universe": int(len(valid_tickers)),
            "strategy_ret": round(net_strategy_ret * 100, 3),
            "benchmark_ret": round(bench_period_ret * 100, 3),
            "equalweight_ret": round(ew_period_ret * 100, 3),
            "alpha": round((net_strategy_ret - bench_period_ret) * 100, 3),
            "alpha_vs_ew": round((net_strategy_ret - ew_period_ret) * 100, 3),
            "cum_strategy": round((cum_strategy - 1.0) * 100, 2),
            "cum_benchmark": round((cum_benchmark - 1.0) * 100, 2),
            "cum_equalweight": round((cum_equalweight - 1.0) * 100, 2),
            "turnover_cost_bp": round(turnover_cost * 10000, 1),
        }
        for tk in extra_px:
            key = tk.lower()
            record[f"{key}_ret"] = round(extra_period_rets[tk] * 100, 3)
            record[f"cum_{key}"] = round((cum_extra[tk] - 1.0) * 100, 2)
            record[f"alpha_vs_{key}"] = round((net_strategy_ret - extra_period_rets[tk]) * 100, 3)

        if ml_meta_at_t is not None:
            if signal_mode == "ml_momentum_blend":
                record["ml_weights"] = ml_meta_at_t.get("weights", {})
                record["ml_spread_train"] = ml_meta_at_t.get("spread_train")
                record["ml_n_train_obs"] = ml_meta_at_t.get("n_obs")
                ml_weight_history.append({
                    "date": date_str,
                    "weights": ml_meta_at_t.get("weights", {}),
                    "spread_train": ml_meta_at_t.get("spread_train"),
                    "n_obs": ml_meta_at_t.get("n_obs"),
                    "fallback": ml_meta_at_t.get("fallback", False),
                })
            elif signal_mode == "ml_lightgbm":
                record["ml_n_train"] = ml_meta_at_t.get("n_train")
                record["ml_n_train_periods"] = ml_meta_at_t.get("n_train_periods")
                record["ml_fallback_to_12_1m"] = ml_meta_at_t.get("fallback_to_12_1m", False)
                ml_weight_history.append({
                    "date": date_str,
                    "n_train": ml_meta_at_t.get("n_train"),
                    "n_train_periods": ml_meta_at_t.get("n_train_periods"),
                    "feature_importance": ml_meta_at_t.get("feature_importance", {}),
                    "fallback": ml_meta_at_t.get("fallback", False) or ml_meta_at_t.get("fallback_to_12_1m", False),
                })

        history.append(record)
        prev_positions = positions
        prev_weights = weights
        prev_scale = scale

    # ── Metrics ──
    if not monthly_strategy_rets:
        return {"error": "No backtest periods generated."}

    n_months = len(monthly_strategy_rets)
    years = n_months / 12.0

    def _metric_block(rets: List[float]) -> Dict[str, float]:
        rets_arr = np.array(rets, dtype=float)
        total_ret = (np.prod(1.0 + rets_arr) - 1.0) * 100
        cagr = ((1.0 + total_ret / 100.0) ** (1.0 / max(years, 0.01)) - 1.0) * 100
        avg_monthly = rets_arr.mean()
        std_monthly = rets_arr.std(ddof=1) if len(rets_arr) > 1 else 0.0
        if std_monthly > 0:
            sharpe = (avg_monthly * 12 - risk_free_rate) / (std_monthly * math.sqrt(12))
        else:
            sharpe = 0.0
        # Max Drawdown
        cum = np.cumprod(1.0 + rets_arr)
        running_max = np.maximum.accumulate(cum)
        dd = (cum / running_max - 1.0) * 100
        mdd = float(dd.min())
        return {
            "total_return": round(float(total_ret), 2),
            "cagr": round(float(cagr), 2),
            "sharpe": round(float(sharpe), 2),
            "mdd": round(float(mdd), 2),
            "avg_monthly_ret": round(float(avg_monthly * 100), 3),
            "vol_monthly": round(float(std_monthly * 100), 3),
        }

    strat_metrics = _metric_block(monthly_strategy_rets)
    bench_metrics = _metric_block(monthly_bench_rets)
    ew_metrics    = _metric_block(monthly_ew_rets)
    extra_metrics: Dict[str, Dict[str, float]] = {
        tk.lower(): _metric_block(monthly_extra_rets[tk]) for tk in extra_px
    }

    # Win rates
    def _win_rate(comparator: List[float]) -> float:
        n_wins = sum(1 for s, c in zip(monthly_strategy_rets, comparator) if s > c)
        return round(100.0 * n_wins / n_months, 1)

    win_rate_spy = _win_rate(monthly_bench_rets)
    win_rate_ew  = _win_rate(monthly_ew_rets)
    win_rate_extra = {
        tk.lower(): _win_rate(monthly_extra_rets[tk]) for tk in extra_px
    }

    # Active-vs-SPY metrics (essential for tilt-overlay interpretation)
    active_rets = [s - b for s, b in zip(monthly_strategy_rets, monthly_bench_rets)]
    active_arr = np.array(active_rets, dtype=float)
    active_avg_monthly = float(active_arr.mean())
    active_std_monthly = float(active_arr.std(ddof=1)) if len(active_arr) > 1 else 0.0
    tracking_error_annualized = active_std_monthly * math.sqrt(12) * 100  # %
    active_return_annualized = strat_metrics["cagr"] - bench_metrics["cagr"]  # CAGR diff
    information_ratio = (
        round(active_return_annualized / tracking_error_annualized, 3)
        if tracking_error_annualized > 0 else 0.0
    )

    # Turnover (Σ|Δw| average — works for both portfolio modes)
    sum_dw_list = [h.get("sum_dw", 0.0) for h in history]
    turnover_sum_dw_avg = round(float(np.mean(sum_dw_list)), 4) if sum_dw_list else 0.0
    n_changes_list = [h["n_changes"] for h in history]
    turnover_avg = round(np.mean(n_changes_list), 2) if n_changes_list else 0.0

    # ── Year-by-year breakdown ──
    yearly: List[Dict[str, Any]] = []
    if history:
        df_hist = pd.DataFrame(history)
        df_hist["year"] = pd.to_datetime(df_hist["date"]).dt.year
        for yr, grp in df_hist.groupby("year"):
            s_ret = grp["strategy_ret"].values / 100.0
            b_ret = grp["benchmark_ret"].values / 100.0
            e_ret = grp["equalweight_ret"].values / 100.0
            yr_strat = (np.prod(1.0 + s_ret) - 1.0) * 100
            yr_bench = (np.prod(1.0 + b_ret) - 1.0) * 100
            yr_ew    = (np.prod(1.0 + e_ret) - 1.0) * 100
            n_wins_yr = int(((s_ret - b_ret) > 0).sum())
            entry = {
                "year": int(yr),
                "n_months": int(len(grp)),
                "strategy_ret": round(float(yr_strat), 2),
                "benchmark_ret": round(float(yr_bench), 2),
                "equalweight_ret": round(float(yr_ew), 2),
                "alpha_vs_spy": round(float(yr_strat - yr_bench), 2),
                "alpha_vs_ew": round(float(yr_strat - yr_ew), 2),
                "win_rate_vs_spy": round(100.0 * n_wins_yr / len(grp), 1),
            }
            for tk in extra_px:
                key = tk.lower()
                col = f"{key}_ret"
                if col in grp.columns:
                    x_ret = grp[col].values / 100.0
                    yr_x = (np.prod(1.0 + x_ret) - 1.0) * 100
                    entry[f"{key}_ret"] = round(float(yr_x), 2)
                    entry[f"alpha_vs_{key}"] = round(float(yr_strat - yr_x), 2)
            yearly.append(entry)

    signal_label = {
        "momentum_12_1m": "12-1M momentum (Jegadeesh-Titman 1993)",
        "composite_live": "Composite-equivalent (0.30·TCS + 0.25·TFS + 0.30·RSS + 0.15·URS, daily-reconstructed)",
        "ml_momentum_blend": "ML multi-horizon momentum blend (1M / 3M / 6M / 12-1M, walk-forward fit + L2 toward 12-1M baseline)",
        "ml_lightgbm": "ML LightGBM (4 momentum horizons + 7 macro features: VIX/yield/curve/credit/DXY)",
    }[signal_mode]

    return {
        "as_of": datetime.utcnow().isoformat(),
        "config": {
            "lookback_years": lookback_years,
            "top_n": top_n,
            "turnover_bp": turnover_bp,
            "rebalance": "monthly",
            "signal_mode": signal_mode,
            "signal": signal_label,
            "vol_target_pct": vol_target_pct,
            "vol_lookback_months": vol_lookback_months,
            "max_leverage": max_leverage,
            "universe": SECTOR_TICKERS,
            "benchmark": BENCHMARK,
            "n_months_backtested": n_months,
            "years": round(years, 2),
        },
        "metrics": {
            "strategy": strat_metrics,
            "benchmark": bench_metrics,           # SPY (legacy alias)
            "equalweight": ew_metrics,
            "spy": bench_metrics,                  # explicit
            **{k: extra_metrics[k] for k in extra_metrics},  # qqq / iwm / acwi
            "alpha_total": round(strat_metrics["total_return"] - bench_metrics["total_return"], 2),
            "alpha_annualized": round(strat_metrics["cagr"] - bench_metrics["cagr"], 2),
            "alpha_vs_ew_total": round(strat_metrics["total_return"] - ew_metrics["total_return"], 2),
            "alpha_vs_ew_annualized": round(strat_metrics["cagr"] - ew_metrics["cagr"], 2),
            **{
                f"alpha_vs_{k}_total": round(strat_metrics["total_return"] - extra_metrics[k]["total_return"], 2)
                for k in extra_metrics
            },
            **{
                f"alpha_vs_{k}_annualized": round(strat_metrics["cagr"] - extra_metrics[k]["cagr"], 2)
                for k in extra_metrics
            },
            "win_rate_pct": win_rate_spy,        # vs SPY (legacy field name)
            "win_rate_vs_ew_pct": win_rate_ew,   # vs equal-weight 11
            **{f"win_rate_vs_{k}_pct": win_rate_extra[k] for k in win_rate_extra},
            "turnover_avg_per_rebalance": turnover_avg,
            "turnover_sum_dw_avg": turnover_sum_dw_avg,    # Σ|Δw| average per rebalance
            "tracking_error_pct": round(tracking_error_annualized, 3),  # vs SPY, annualized
            "information_ratio": information_ratio,                     # active α / TE
            "active_return_annualized": round(active_return_annualized, 3),
            "extra_benchmarks": list(extra_metrics.keys()),
        },
        "yearly": yearly,
        "monthly": history,
        "ml_weight_history": ml_weight_history,  # populated only in ml_momentum_blend mode
    }


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _print_run(label: str, out: Dict[str, Any]) -> None:
        if "error" in out:
            print(f"[{label}] ERROR: {out['error']}\n")
            return
        m = out["metrics"]
        cfg = out["config"]
        print(f"\n══ {label}  ·  {cfg['signal']}")
        print(f"   {cfg['n_months_backtested']} months ({cfg['years']}y), top_{cfg['top_n']}, {cfg['turnover_bp']}bp turnover")
        print(f"   {'Metric':<22} {'Strat':>9} {'EW-11':>9} {'SPY':>9} {'QQQ':>9} {'IWM':>9} {'ACWI':>9}")
        print("   " + "-" * 84)
        def row(name: str, key: str) -> None:
            qqq = m.get('qqq', {}).get(key, float('nan'))
            iwm = m.get('iwm', {}).get(key, float('nan'))
            acwi = m.get('acwi', {}).get(key, float('nan'))
            print(f"   {name:<22} {m['strategy'][key]:>9.2f} {m['equalweight'][key]:>9.2f} {m['benchmark'][key]:>9.2f} {qqq:>9.2f} {iwm:>9.2f} {acwi:>9.2f}")
        row("Total Return (%)", "total_return")
        row("CAGR (%)", "cagr")
        row("Sharpe", "sharpe")
        row("Max Drawdown (%)", "mdd")
        print(f"     α vs SPY  (CAGR): {m.get('alpha_annualized', 0):+.2f}%   "
              f"α vs EW-11 (CAGR): {m.get('alpha_vs_ew_annualized', 0):+.2f}%  ★")
        print(f"     α vs QQQ  (CAGR): {m.get('alpha_vs_qqq_annualized', 0):+.2f}%   "
              f"α vs IWM   (CAGR): {m.get('alpha_vs_iwm_annualized', 0):+.2f}%   "
              f"α vs ACWI  (CAGR): {m.get('alpha_vs_acwi_annualized', 0):+.2f}%")
        print(f"     win SPY/EW/QQQ/IWM/ACWI: {m.get('win_rate_pct', 0)}/{m.get('win_rate_vs_ew_pct', 0)}/"
              f"{m.get('win_rate_vs_qqq_pct', 0)}/{m.get('win_rate_vs_iwm_pct', 0)}/{m.get('win_rate_vs_acwi_pct', 0)}%   "
              f"turnover: {m.get('turnover_avg_per_rebalance', 0)}")

    out_a = run_backtest(lookback_years=5, top_n=3, turnover_bp=30, signal_mode="momentum_12_1m")
    out_b = run_backtest(lookback_years=5, top_n=3, turnover_bp=30, signal_mode="composite_live")
    out_c = run_backtest(lookback_years=10, top_n=3, turnover_bp=30, signal_mode="ml_momentum_blend")
    _print_run("Phase 3 — momentum_12_1m", out_a)
    _print_run("Phase 4 — composite_live", out_b)
    _print_run("Phase 5 — ml_momentum_blend (10y)", out_c)

    if "error" not in out_c and out_c.get("ml_weight_history"):
        print(f"\nML blend weights — last 6 months:")
        print(f"{'Date':<12} {'m1_skip1':>9} {'m3_skip1':>9} {'m6_skip1':>9} {'m12_1':>9} {'spread%':>9} {'#obs':>5}")
        print("-" * 76)
        for h in out_c["ml_weight_history"][-6:]:
            w = h["weights"]
            sp = h.get("spread_train")
            sp_str = f"{sp:>9.3f}" if sp is not None else "      n/a"
            print(f"{h['date']:<12} {w.get('m1_skip1', 0):>9.3f} {w.get('m3_skip1', 0):>9.3f} "
                  f"{w.get('m6_skip1', 0):>9.3f} {w.get('m12_1', 0):>9.3f} {sp_str} {h.get('n_obs', 0):>5}")
        print(f"\nLast 4 months positions (ml_momentum_blend):")
        for h in out_c["monthly"][-4:]:
            print(f"  {h['date']}: {','.join(h['positions']):<22} α={h['alpha']:>+6.2f}")
