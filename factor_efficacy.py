"""
Factor Efficacy Engine — Reverse Factor Model
==============================================
전통적 팩터모델의 역방향 접근: 가격 데이터 → 팩터 우월성 분석

Instead of:  factors → expected returns → stock selection  (traditional)
This does:   price data → factor premium → factor superiority ranking (reverse)

5 Methodologies:
    1. Fama-MacBeth Cross-Sectional Regression  — 팩터 프리미엄 λ 추정
    2. Information Coefficient (IC) Analysis     — 팩터 예측력 순위상관
    3. Long-Short Factor Portfolio               — 팩터 모방 포트폴리오
    4. PCA Statistical Factor Model              — 잠재 팩터 추출 & 매핑
    5. Regime-Conditional Factor Premium          — 레짐별 팩터 유효성

Usage:
    # Integrated (during run_scan):
    from factor_efficacy import FactorEfficacyEngine
    fe = FactorEfficacyEngine(all_data, detector)
    output = fe.run()

    # Standalone:
    from factor_efficacy import run_factor_efficacy
    output = run_factor_efficacy()
"""

import os
import sys
import math
import warnings
from datetime import datetime, date
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from price_discovery import (
    NaiveDiscoveryDetector, DataEngine, sf, ss,
    GLOBAL_ETF_UNIVERSE, STOCK_UNIVERSE,
)


###############################################################################
# FACTOR DEFINITIONS
###############################################################################

# 분석 대상 팩터 후보 — compute_raw() 출력 필드명 기준
FACTOR_CANDIDATES = [
    # ── Momentum (Short-Term) ──
    'ret_5d', 'ret_10d', 'ret_21d',
    # ── Momentum (Long-Term) ──
    'ret_63d', 'ret_126d', 'ret_12_1m',
    # ── Risk-Adjusted Momentum ──
    'vol_adj_mom',
    # ── Trend ──
    'sma20_slope', 'sma50_slope', 'sma200_slope',
    'sma50_dist', 'sma20_dist',
    'above_sma50', 'above_sma200', 'golden_cross',
    # ── Volume ──
    'vol_ratio', 'vol_ratio_3d_10d', 'obv_slope',
    # ── Volatility ──
    'realized_vol', 'vcr',
    # ── Positioning ──
    'range_pct', 'pct_from_high', 'rsi',
    # ── Breakout ──
    'breakout_20d', 'breakout_10d',
    # ── Trend Age ──
    'trend_age', 'trend_age_short',
    # ── Ichimoku ──
    'ichimoku_above_cloud',
]

# 의미론적 그룹핑 (해석용)
FACTOR_GROUPS = {
    'Momentum_Short':    ['ret_5d', 'ret_10d', 'ret_21d'],
    'Momentum_Long':     ['ret_63d', 'ret_126d', 'ret_12_1m'],
    'Momentum_RiskAdj':  ['vol_adj_mom'],
    'Trend_Slope':       ['sma20_slope', 'sma50_slope', 'sma200_slope'],
    'Trend_Distance':    ['sma50_dist', 'sma20_dist'],
    'Trend_Position':    ['above_sma50', 'above_sma200', 'golden_cross'],
    'Volume':            ['vol_ratio', 'vol_ratio_3d_10d', 'obv_slope'],
    'Volatility':        ['realized_vol', 'vcr'],
    'Positioning':       ['range_pct', 'pct_from_high', 'rsi'],
    'Breakout':          ['breakout_20d', 'breakout_10d'],
    'Trend_Age':         ['trend_age', 'trend_age_short'],
    'Ichimoku':          ['ichimoku_above_cloud'],
}

# 팩터 → 그룹 역매핑
_FACTOR_TO_GROUP = {}
for grp, factors in FACTOR_GROUPS.items():
    for f in factors:
        _FACTOR_TO_GROUP[f] = grp

# Forward return 측정 기간 (거래일)
FORWARD_WINDOWS = {
    '1W': 5,
    '1M': 21,
    '3M': 63,
}

# 기본 분석 기간: 12개 월간 평가 시점 (21거래일 간격)
N_EVAL_POINTS = 12
EVAL_INTERVAL = 21  # trading days


###############################################################################
# UTILITIES (numpy only — no scipy dependency)
###############################################################################

def _safe_float(v, default=0.0):
    try:
        r = float(v)
        return r if math.isfinite(r) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (numpy only)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    mx, my = rx.mean(), ry.mean()
    cov = np.sum((rx - mx) * (ry - my))
    sx = np.sqrt(np.sum((rx - mx) ** 2))
    sy = np.sqrt(np.sum((ry - my) ** 2))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(cov / (sx * sy))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation (numpy only)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    mx, my = x.mean(), y.mean()
    cov = np.sum((x - mx) * (y - my))
    sx = np.sqrt(np.sum((x - mx) ** 2))
    sy = np.sqrt(np.sum((y - my) ** 2))
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(cov / (sx * sy))


def _ols_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS regression via numpy lstsq. Returns [intercept, β1, β2, ...]."""
    n = len(y)
    if n < X.shape[1] + 2:
        return np.full(X.shape[1] + 1, np.nan)
    X_aug = np.column_stack([np.ones(n), X])
    try:
        coefs, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        return coefs
    except np.linalg.LinAlgError:
        return np.full(X.shape[1] + 1, np.nan)


def _standardize(arr: np.ndarray) -> np.ndarray:
    """Z-score standardization, NaN-safe."""
    mask = np.isfinite(arr)
    if mask.sum() < 3:
        return np.zeros_like(arr)
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if s < 1e-12:
        return np.zeros_like(arr)
    out = (arr - m) / s
    out[~mask] = 0.0
    return out


def _pct_rank(value: float, arr: np.ndarray) -> float:
    """Percentile rank (0-100)."""
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        return 50.0
    return float(np.sum(finite < value)) / (len(finite) - 1) * 100


###############################################################################
# FACTOR EFFICACY ENGINE
###############################################################################

class FactorEfficacyEngine:
    """
    Reverse Factor Model: 가격 데이터 → 팩터 우월성 분석

    5가지 방법론을 통합하여 현재 어떤 팩터가 가장 유효한지 판단.
    """

    def __init__(self, all_data: Dict, detector: NaiveDiscoveryDetector,
                 n_eval_points: int = N_EVAL_POINTS,
                 eval_interval: int = EVAL_INTERVAL):
        """
        Parameters
        ----------
        all_data : dict
            {ticker: ETFData} from DataEngine.download_universe()
        detector : NaiveDiscoveryDetector
            Instance for compute_raw()
        n_eval_points : int
            Number of historical evaluation dates
        eval_interval : int
            Trading days between evaluation points
        """
        self.all_data = all_data
        self.detector = detector
        self.n_eval = n_eval_points
        self.eval_interval = eval_interval

        # Panel storage: list of cross-sectional snapshots
        # Each snapshot: {eval_date, factor_matrix, forward_returns, tickers}
        self._panel: List[dict] = []
        self._daily_returns: Optional[pd.DataFrame] = None
        self._spy_raw_series: Optional[dict] = None

    # ------------------------------------------------------------------
    # STEP 1: Build Historical Panel
    # ------------------------------------------------------------------

    def _build_panel(self):
        """
        12개 월간 평가 시점에서 모든 종목의 팩터 값과 forward return을 계산.
        Top-Long Backtest (price_discovery.py) 의 replay 패턴을 따름.
        """
        print(f"\n{'='*70}")
        print("  Factor Efficacy Engine: Building Historical Panel")
        print(f"  {self.n_eval} evaluation points, {self.eval_interval}d interval")
        print(f"{'='*70}")

        # Reference dates from first valid ticker
        ref_ticker = None
        for tk, etf in self.all_data.items():
            if etf.valid and len(etf.df) >= 300:
                ref_ticker = tk
                break
        if ref_ticker is None:
            print("  ⚠️ No valid reference ticker found")
            return

        all_dates = self.all_data[ref_ticker].df.index
        n_dates = len(all_dates)

        # Evaluation offsets (from latest date, going backward)
        max_fwd = max(FORWARD_WINDOWS.values())  # need room for forward return
        eval_offsets = [i * self.eval_interval for i in range(1, self.n_eval + 1)]
        eval_offsets = [o for o in eval_offsets if o + max_fwd < n_dates - 60]

        for idx, offset in enumerate(eval_offsets):
            eval_idx = n_dates - 1 - offset
            if eval_idx < 60:
                continue
            eval_date = all_dates[eval_idx]
            eval_str = str(eval_date.date()) if hasattr(eval_date, 'date') else str(eval_date)[:10]

            # Compute raw indicators at eval_date for all tickers
            snap_raw = {}
            snap_cats = {}
            for ticker, etf in self.all_data.items():
                if not etf.valid:
                    continue
                df_cut = etf.df[etf.df.index <= eval_date]
                if len(df_cut) < 60:
                    continue
                try:
                    raw = self.detector.compute_raw(df_cut, etf.category)
                    snap_raw[ticker] = raw
                    snap_cats[ticker] = etf.category
                except Exception:
                    continue

            if len(snap_raw) < 20:
                continue

            tickers = sorted(snap_raw.keys())
            n_tickers = len(tickers)

            # Build factor matrix: (n_tickers × n_factors)
            factor_values = {}
            for f in FACTOR_CANDIDATES:
                vals = np.array([_safe_float(snap_raw[t].get(f, 0)) for t in tickers])
                factor_values[f] = vals

            # Compute forward returns for each horizon
            fwd_returns = {}
            for period_name, fwd_days in FORWARD_WINDOWS.items():
                rets = np.full(n_tickers, np.nan)
                for i, tk in enumerate(tickers):
                    etf = self.all_data[tk]
                    tk_close = ss(etf.df['Close'])
                    tk_dates = etf.df.index
                    tk_pos = tk_dates.searchsorted(eval_date, side='right') - 1
                    if tk_pos < 0 or tk_pos >= len(tk_close):
                        continue
                    entry_price = sf(tk_close.iloc[tk_pos])
                    fwd_pos = tk_pos + fwd_days
                    if entry_price > 0 and fwd_pos < len(tk_close):
                        fwd_price = sf(tk_close.iloc[fwd_pos])
                        rets[i] = (fwd_price / entry_price - 1) * 100
                fwd_returns[period_name] = rets

            # SPY regime data at this eval point
            spy_raw = snap_raw.get('SPY', {})

            self._panel.append({
                'eval_date': eval_date,
                'eval_str': eval_str,
                'tickers': tickers,
                'categories': snap_cats,
                'factor_values': factor_values,
                'fwd_returns': fwd_returns,
                'n_tickers': n_tickers,
                'spy_raw': spy_raw,
            })

            print(f"  [{idx+1}/{len(eval_offsets)}] {eval_str}: "
                  f"{n_tickers} tickers scored")

        print(f"  ✅ Panel built: {len(self._panel)} evaluation points")

        # Build daily returns matrix for PCA
        self._build_daily_returns()

    def _build_daily_returns(self):
        """Build (T × N) daily returns matrix for PCA analysis."""
        # Use last 252 trading days
        valid_tickers = []
        for tk, etf in sorted(self.all_data.items()):
            if etf.valid and len(etf.df) >= 252:
                valid_tickers.append(tk)

        if len(valid_tickers) < 20:
            self._daily_returns = None
            return

        # Align on common dates
        ref = self.all_data[valid_tickers[0]].df.index[-252:]
        returns_dict = {}
        for tk in valid_tickers:
            close = ss(self.all_data[tk].df['Close'])
            daily_ret = close.pct_change().dropna()
            # Reindex to ref dates
            aligned = daily_ret.reindex(ref)
            if aligned.notna().sum() >= 200:
                returns_dict[tk] = aligned.values

        if len(returns_dict) < 20:
            self._daily_returns = None
            return

        tickers = sorted(returns_dict.keys())
        mat = np.column_stack([returns_dict[t] for t in tickers])
        # Fill NaN with 0
        mat = np.nan_to_num(mat, nan=0.0)
        self._daily_returns = pd.DataFrame(mat, columns=tickers, index=ref)

    # ------------------------------------------------------------------
    # METHOD 1: Fama-MacBeth Cross-Sectional Regression
    # ------------------------------------------------------------------

    def fama_macbeth(self, fwd_period: str = '1M') -> dict:
        """
        Fama-MacBeth (1973) 횡단면 회귀:
        매 시점 t: R_i,t+k = α_t + Σ λ_k,t · F_k,i,t + ε_i,t
        → λ̄_k = mean(λ_k,t), t-stat = λ̄ / SE(λ̄)

        Returns
        -------
        dict with factor_premiums list sorted by abs(t_stat)
        """
        if not self._panel:
            return {'factor_premiums': [], 'method': 'fama_macbeth'}

        # Collect factor premium time series: {factor: [λ_t1, λ_t2, ...]}
        lambda_series = {f: [] for f in FACTOR_CANDIDATES}

        for snap in self._panel:
            fwd_ret = snap['fwd_returns'].get(fwd_period)
            if fwd_ret is None:
                continue
            valid_mask = np.isfinite(fwd_ret)
            if valid_mask.sum() < 30:
                continue

            # Standardize factor exposures cross-sectionally
            X_cols = []
            valid_factors = []
            for f in FACTOR_CANDIDATES:
                vals = snap['factor_values'][f].copy()
                vals[~valid_mask] = np.nan
                std_vals = _standardize(vals[valid_mask])
                if np.all(np.abs(std_vals) < 1e-10):
                    continue
                X_cols.append(std_vals)
                valid_factors.append(f)

            if len(X_cols) < 3:
                continue

            X = np.column_stack(X_cols)
            y = fwd_ret[valid_mask]

            # OLS: y = α + Σ λ_k * X_k
            coefs = _ols_coefficients(X, y)
            if np.any(np.isnan(coefs)):
                continue

            # coefs[0] = intercept, coefs[1:] = factor premiums
            for i, f in enumerate(valid_factors):
                lambda_series[f].append(coefs[i + 1])

        # Compute summary statistics
        factor_premiums = []
        for f in FACTOR_CANDIDATES:
            lambdas = np.array(lambda_series[f])
            lambdas = lambdas[np.isfinite(lambdas)]
            if len(lambdas) < 3:
                continue
            mean_lambda = float(np.mean(lambdas))
            std_lambda = float(np.std(lambdas, ddof=1))
            t_stat = mean_lambda / (std_lambda / np.sqrt(len(lambdas))) if std_lambda > 1e-10 else 0.0
            # Significance: |t| > 1.96 → 95%, |t| > 2.58 → 99%
            if abs(t_stat) >= 2.58:
                sig = '***'
            elif abs(t_stat) >= 1.96:
                sig = '**'
            elif abs(t_stat) >= 1.65:
                sig = '*'
            else:
                sig = ''

            factor_premiums.append({
                'factor': f,
                'group': _FACTOR_TO_GROUP.get(f, 'Other'),
                'mean_premium': round(mean_lambda, 4),
                'std_premium': round(std_lambda, 4),
                't_stat': round(t_stat, 3),
                'significance': sig,
                'n_periods': len(lambdas),
                'pct_positive': round(float(np.sum(lambdas > 0)) / len(lambdas) * 100, 1),
            })

        factor_premiums.sort(key=lambda x: -abs(x['t_stat']))

        return {
            'method': 'fama_macbeth',
            'description': 'Fama-MacBeth 횡단면 회귀: 팩터 프리미엄 λ 추정',
            'forward_period': fwd_period,
            'n_periods': len(self._panel),
            'factor_premiums': factor_premiums,
        }

    # ------------------------------------------------------------------
    # METHOD 2: Information Coefficient (IC) Analysis
    # ------------------------------------------------------------------

    def ic_analysis(self, fwd_period: str = '1M') -> dict:
        """
        Information Coefficient: IC_k,t = Spearman(Factor_k at t, Return at t+k)

        IC Mean: 팩터의 평균 예측력
        IC IR (= IC_mean / IC_std): 안정성 포함한 팩터 품질
        IC > 0.05, IC IR > 0.5 → 실무적으로 유의미한 팩터

        Returns
        -------
        dict with factor_ic list sorted by IC IR
        """
        if not self._panel:
            return {'factor_ic': [], 'method': 'ic_analysis'}

        # IC time series: {factor: [IC_t1, IC_t2, ...]}
        ic_series = {f: [] for f in FACTOR_CANDIDATES}

        for snap in self._panel:
            fwd_ret = snap['fwd_returns'].get(fwd_period)
            if fwd_ret is None:
                continue

            for f in FACTOR_CANDIDATES:
                vals = snap['factor_values'][f]
                ic = _spearman_corr(vals, fwd_ret)
                if np.isfinite(ic):
                    ic_series[f].append(ic)

        # Summary statistics
        factor_ic = []
        for f in FACTOR_CANDIDATES:
            ics = np.array(ic_series[f])
            ics = ics[np.isfinite(ics)]
            if len(ics) < 3:
                continue

            ic_mean = float(np.mean(ics))
            ic_std = float(np.std(ics, ddof=1))
            ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
            ic_hit_rate = float(np.sum(ics > 0)) / len(ics) * 100

            # IC autocorrelation (stability measure)
            if len(ics) >= 4:
                ic_auto = _pearson_corr(ics[:-1], ics[1:])
            else:
                ic_auto = np.nan

            # Qualitative assessment
            if abs(ic_ir) >= 0.5 and abs(ic_mean) >= 0.05:
                quality = 'STRONG'
            elif abs(ic_ir) >= 0.3 and abs(ic_mean) >= 0.03:
                quality = 'MODERATE'
            elif abs(ic_mean) >= 0.02:
                quality = 'WEAK'
            else:
                quality = 'NOISE'

            factor_ic.append({
                'factor': f,
                'group': _FACTOR_TO_GROUP.get(f, 'Other'),
                'ic_mean': round(ic_mean, 4),
                'ic_std': round(ic_std, 4),
                'ic_ir': round(ic_ir, 3),
                'ic_hit_rate': round(ic_hit_rate, 1),
                'ic_autocorr': round(ic_auto, 3) if np.isfinite(ic_auto) else None,
                'quality': quality,
                'n_periods': len(ics),
                'ic_timeseries': [round(float(x), 4) for x in ics],
                'direction': 'POSITIVE' if ic_mean > 0.01 else ('NEGATIVE' if ic_mean < -0.01 else 'NEUTRAL'),
            })

        factor_ic.sort(key=lambda x: -abs(x['ic_ir']))

        return {
            'method': 'ic_analysis',
            'description': 'Information Coefficient: 팩터값과 미래수익률 간 순위상관',
            'forward_period': fwd_period,
            'n_periods': len(self._panel),
            'factor_ic': factor_ic,
            'interpretation': {
                'ic_mean': '양수 = 팩터값 높을수록 수익률 높음, 음수 = 역방향',
                'ic_ir': 'IC의 샤프비율. 0.5 이상이면 유의미',
                'quality': 'STRONG (IC_IR≥0.5, IC≥0.05) > MODERATE > WEAK > NOISE',
            },
        }

    # ------------------------------------------------------------------
    # METHOD 3: Long-Short Factor Portfolio
    # ------------------------------------------------------------------

    def long_short_portfolio(self, fwd_period: str = '1M') -> dict:
        """
        팩터 모방 포트폴리오 (Factor Mimicking Portfolio):
        매 시점 팩터값 기준 5분위 정렬 → Top quintile Long, Bottom quintile Short
        → L-S 수익률 시계열 → Sharpe Ratio로 팩터 우열 비교

        Returns
        -------
        dict with factor_returns list sorted by Sharpe ratio
        """
        if not self._panel:
            return {'factor_returns': [], 'method': 'long_short_portfolio'}

        # Per-factor quintile return series
        factor_ls_returns = {f: [] for f in FACTOR_CANDIDATES}
        factor_quintile_returns = {f: {q: [] for q in range(5)} for f in FACTOR_CANDIDATES}

        for snap in self._panel:
            fwd_ret = snap['fwd_returns'].get(fwd_period)
            if fwd_ret is None:
                continue
            valid_mask = np.isfinite(fwd_ret)
            if valid_mask.sum() < 25:  # need at least 5 per quintile
                continue

            for f in FACTOR_CANDIDATES:
                vals = snap['factor_values'][f]
                # Both factor and return must be valid
                mask = valid_mask & np.isfinite(vals)
                if mask.sum() < 25:
                    continue

                v = vals[mask]
                r = fwd_ret[mask]

                # Sort into quintiles
                quintile_cuts = np.percentile(v, [20, 40, 60, 80])
                for q_idx in range(5):
                    if q_idx == 0:
                        q_mask = v <= quintile_cuts[0]
                    elif q_idx == 4:
                        q_mask = v > quintile_cuts[3]
                    else:
                        q_mask = (v > quintile_cuts[q_idx - 1]) & (v <= quintile_cuts[q_idx])
                    if q_mask.sum() > 0:
                        factor_quintile_returns[f][q_idx].append(float(np.mean(r[q_mask])))

                # L-S return: Top quintile - Bottom quintile
                top_mask = v > quintile_cuts[3]
                bot_mask = v <= quintile_cuts[0]
                if top_mask.sum() > 0 and bot_mask.sum() > 0:
                    ls_ret = float(np.mean(r[top_mask]) - np.mean(r[bot_mask]))
                    factor_ls_returns[f].append(ls_ret)

        # Summary
        factor_returns = []
        for f in FACTOR_CANDIDATES:
            ls_rets = np.array(factor_ls_returns[f])
            if len(ls_rets) < 3:
                continue

            mean_ret = float(np.mean(ls_rets))
            std_ret = float(np.std(ls_rets, ddof=1))
            sharpe = mean_ret / std_ret if std_ret > 1e-10 else 0.0
            # Annualize (assuming monthly periods for '1M')
            periods_per_year = {'1W': 52, '1M': 12, '3M': 4}.get(fwd_period, 12)
            ann_ret = mean_ret * periods_per_year
            ann_sharpe = sharpe * np.sqrt(periods_per_year)

            # Quintile means for monotonicity check
            q_means = []
            for q in range(5):
                q_rets = factor_quintile_returns[f][q]
                q_means.append(round(float(np.mean(q_rets)), 3) if q_rets else 0.0)

            # Monotonicity: is Q5 > Q4 > Q3 > Q2 > Q1? (or reversed)
            diffs = [q_means[i+1] - q_means[i] for i in range(4)]
            positive_steps = sum(1 for d in diffs if d > 0)
            mono_score = max(positive_steps, 4 - positive_steps) / 4.0  # 1.0 = perfect
            mono_direction = 'ASCENDING' if positive_steps >= 3 else (
                'DESCENDING' if positive_steps <= 1 else 'MIXED')

            factor_returns.append({
                'factor': f,
                'group': _FACTOR_TO_GROUP.get(f, 'Other'),
                'ls_mean_return': round(mean_ret, 3),
                'ls_std': round(std_ret, 3),
                'ls_sharpe': round(sharpe, 3),
                'ann_return': round(ann_ret, 2),
                'ann_sharpe': round(ann_sharpe, 3),
                'quintile_returns': q_means,
                'monotonicity': round(mono_score, 2),
                'mono_direction': mono_direction,
                'win_rate': round(float(np.sum(ls_rets > 0)) / len(ls_rets) * 100, 1),
                'n_periods': len(ls_rets),
            })

        factor_returns.sort(key=lambda x: -abs(x['ann_sharpe']))

        return {
            'method': 'long_short_portfolio',
            'description': '팩터 모방 포트폴리오: Top-Bottom 분위 L-S 수익률',
            'forward_period': fwd_period,
            'n_periods': len(self._panel),
            'factor_returns': factor_returns,
            'interpretation': {
                'ls_mean_return': '양수 = 팩터값 높을수록 수익률 높음',
                'ann_sharpe': '연환산 샤프비율. 0.5 이상 유의미',
                'monotonicity': '1.0 = 완벽한 단조성 (Q1<Q2<Q3<Q4<Q5)',
            },
        }

    # ------------------------------------------------------------------
    # METHOD 4: PCA Statistical Factor Model
    # ------------------------------------------------------------------

    def pca_analysis(self, n_components: int = 8) -> dict:
        """
        주성분분석 (PCA):
        수익률 공분산행렬 → 잠재 팩터 추출 → 알려진 팩터와 상관 분석

        순수 데이터 드리븐: 가격 데이터만으로 시장의 잠재적 차원을 발견하고,
        이것이 어떤 경제적/기술적 팩터에 해당하는지 해석.

        Returns
        -------
        dict with PCA results
        """
        if self._daily_returns is None or len(self._daily_returns.columns) < 20:
            return {'method': 'pca_analysis', 'components': [],
                    'error': 'Insufficient data for PCA'}

        R = self._daily_returns.values.copy()
        tickers = list(self._daily_returns.columns)
        T, N = R.shape

        # Standardize returns
        R_std = R - R.mean(axis=0)
        stds = R_std.std(axis=0)
        stds[stds < 1e-10] = 1.0
        R_std = R_std / stds

        # Correlation matrix
        try:
            corr = np.corrcoef(R_std.T)
            corr = np.nan_to_num(corr, nan=0.0)
            eigenvalues, eigenvectors = np.linalg.eigh(corr)
        except np.linalg.LinAlgError:
            return {'method': 'pca_analysis', 'components': [],
                    'error': 'Eigendecomposition failed'}

        # Sort descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp negative eigenvalues to 0
        eigenvalues = np.maximum(eigenvalues, 0)
        total_var = eigenvalues.sum()
        if total_var < 1e-10:
            return {'method': 'pca_analysis', 'components': [],
                    'error': 'Zero variance'}

        var_explained = eigenvalues / total_var
        cum_var_explained = np.cumsum(var_explained)

        # Effective dimensionality (eigenvalues > 1 criterion, Kaiser rule)
        effective_dim = int(np.sum(eigenvalues > 1.0))

        # PC scores: (T × n_components)
        n_comp = min(n_components, N, len(eigenvalues))
        pc_scores = R_std @ eigenvectors[:, :n_comp]

        # Map each PC to known factors using the latest cross-section
        # Get current factor values for all tickers in the PCA universe
        pc_factor_mapping = []

        if self._panel:
            latest = self._panel[0]  # most recent evaluation
            common_tickers = [t for t in tickers if t in latest['tickers']]
            if len(common_tickers) >= 20:
                # Get factor values for common tickers
                tk_idx_pca = [tickers.index(t) for t in common_tickers]
                tk_idx_panel = [latest['tickers'].index(t) for t in common_tickers]

                for pc_i in range(n_comp):
                    # PC loadings for common tickers
                    loadings = eigenvectors[tk_idx_pca, pc_i]

                    # Correlate loadings with each factor's cross-sectional values
                    factor_corrs = {}
                    for f in FACTOR_CANDIDATES:
                        f_vals = latest['factor_values'][f][tk_idx_panel]
                        corr_val = _pearson_corr(loadings, f_vals)
                        if np.isfinite(corr_val):
                            factor_corrs[f] = round(corr_val, 3)

                    # Top correlated factors
                    sorted_corrs = sorted(factor_corrs.items(),
                                          key=lambda x: -abs(x[1]))
                    top_factors = sorted_corrs[:5]

                    # Determine dominant interpretation
                    if top_factors:
                        top_group = _FACTOR_TO_GROUP.get(top_factors[0][0], 'Unknown')
                        interpretation = top_group
                    else:
                        interpretation = 'Unidentified'

                    pc_factor_mapping.append({
                        'pc': pc_i + 1,
                        'eigenvalue': round(float(eigenvalues[pc_i]), 3),
                        'var_explained': round(float(var_explained[pc_i]) * 100, 2),
                        'cum_var_explained': round(float(cum_var_explained[pc_i]) * 100, 2),
                        'top_factor_correlations': [
                            {'factor': f, 'correlation': c, 'group': _FACTOR_TO_GROUP.get(f, 'Other')}
                            for f, c in top_factors
                        ],
                        'interpretation': interpretation,
                    })

        # Top/bottom loadings for each PC (which tickers load most)
        pc_top_tickers = []
        for pc_i in range(min(3, n_comp)):  # top 3 PCs only
            loadings = eigenvectors[:, pc_i]
            sorted_idx = np.argsort(np.abs(loadings))[::-1]
            top_load = [
                {'ticker': tickers[j], 'loading': round(float(loadings[j]), 4)}
                for j in sorted_idx[:10]
            ]
            pc_top_tickers.append({
                'pc': pc_i + 1,
                'top_tickers': top_load,
            })

        return {
            'method': 'pca_analysis',
            'description': 'PCA 잠재 팩터 모델: 수익률 공분산에서 잠재 차원 추출',
            'n_tickers': N,
            'n_days': T,
            'n_components': n_comp,
            'effective_dimensionality': effective_dim,
            'components': pc_factor_mapping,
            'pc_top_tickers': pc_top_tickers,
            'total_var_explained_top3': round(float(cum_var_explained[min(2, n_comp-1)]) * 100, 2) if n_comp >= 1 else 0,
            'interpretation': {
                'effective_dim': f'시장은 약 {effective_dim}개의 독립적 차원으로 구성',
                'pc1': '시장 전체 방향성 (Market Beta)',
                'rule': '각 PC가 어떤 명명된 팩터와 가장 높은 상관인지 매핑',
            },
        }

    # ------------------------------------------------------------------
    # METHOD 5: Regime-Conditional Factor Premium
    # ------------------------------------------------------------------

    def regime_conditional(self, fwd_period: str = '1M') -> dict:
        """
        시장 레짐별 팩터 유효성 분석:
        SPY 가격구조로 레짐 분류 → 레짐별 IC 및 팩터 프리미엄 비교

        Regimes:
            BULL:       SPY > SMA50, SMA50 slope > 0
            BEAR:       SPY < SMA50, SMA50 slope < 0
            TRANSITION: otherwise

        Returns
        -------
        dict with regime-specific factor rankings
        """
        if not self._panel:
            return {'method': 'regime_conditional', 'regimes': {}}

        # Classify each evaluation point's regime
        regime_snapshots = {'BULL': [], 'BEAR': [], 'TRANSITION': []}

        for snap in self._panel:
            spy = snap.get('spy_raw', {})
            if not spy:
                regime_snapshots['TRANSITION'].append(snap)
                continue

            above_sma50 = spy.get('above_sma50', 0)
            sma50_slope = _safe_float(spy.get('sma50_slope', 0))
            above_sma200 = spy.get('above_sma200', 0)

            if above_sma50 and sma50_slope > 0:
                regime = 'BULL'
            elif not above_sma50 and sma50_slope < 0:
                regime = 'BEAR'
            else:
                regime = 'TRANSITION'
            regime_snapshots[regime].append(snap)

        # For each regime, compute IC for all factors
        regime_factor_ic = {}
        for regime, snaps in regime_snapshots.items():
            if not snaps:
                continue

            ic_by_factor = {f: [] for f in FACTOR_CANDIDATES}
            for snap in snaps:
                fwd_ret = snap['fwd_returns'].get(fwd_period)
                if fwd_ret is None:
                    continue
                for f in FACTOR_CANDIDATES:
                    vals = snap['factor_values'][f]
                    ic = _spearman_corr(vals, fwd_ret)
                    if np.isfinite(ic):
                        ic_by_factor[f].append(ic)

            # Summarize
            regime_factors = []
            for f in FACTOR_CANDIDATES:
                ics = np.array(ic_by_factor[f])
                if len(ics) < 2:
                    continue
                ic_mean = float(np.mean(ics))
                ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
                ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0.0

                regime_factors.append({
                    'factor': f,
                    'group': _FACTOR_TO_GROUP.get(f, 'Other'),
                    'ic_mean': round(ic_mean, 4),
                    'ic_ir': round(ic_ir, 3),
                    'n_periods': len(ics),
                })
            regime_factors.sort(key=lambda x: -abs(x['ic_ir']))
            regime_factor_ic[regime] = regime_factors

        # Determine current regime
        current_regime = 'TRANSITION'
        if self._panel:
            latest_spy = self._panel[0].get('spy_raw', {})
            if latest_spy:
                if latest_spy.get('above_sma50', 0) and _safe_float(latest_spy.get('sma50_slope', 0)) > 0:
                    current_regime = 'BULL'
                elif not latest_spy.get('above_sma50', 0) and _safe_float(latest_spy.get('sma50_slope', 0)) < 0:
                    current_regime = 'BEAR'

        # Regime-switching recommendations
        recommendations = []
        current_factors = regime_factor_ic.get(current_regime, [])
        if current_factors:
            top3 = current_factors[:3]
            for ff in top3:
                recommendations.append(
                    f"현재 {current_regime} 레짐에서 {ff['factor']} (IC={ff['ic_mean']:.3f}) 이 유효"
                )

        # Cross-regime comparison: which factors are robust across all regimes?
        all_regime_ics = {}
        for regime, factors in regime_factor_ic.items():
            for ff in factors:
                f = ff['factor']
                if f not in all_regime_ics:
                    all_regime_ics[f] = {}
                all_regime_ics[f][regime] = ff['ic_mean']

        cross_regime_stability = []
        for f, regime_ics in all_regime_ics.items():
            if len(regime_ics) < 2:
                continue
            ics = list(regime_ics.values())
            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics))
            # All-weather = consistent sign across regimes
            same_sign = all(x > 0 for x in ics) or all(x < 0 for x in ics)
            cross_regime_stability.append({
                'factor': f,
                'group': _FACTOR_TO_GROUP.get(f, 'Other'),
                'regime_ics': {r: round(v, 4) for r, v in regime_ics.items()},
                'mean_ic': round(mean_ic, 4),
                'std_ic': round(std_ic, 4),
                'all_weather': same_sign,
                'stability': round(abs(mean_ic) / (std_ic + 1e-10), 3),
            })
        cross_regime_stability.sort(key=lambda x: -x['stability'])

        return {
            'method': 'regime_conditional',
            'description': '레짐별 팩터 유효성: 시장 환경에 따른 팩터 프리미엄 변화',
            'forward_period': fwd_period,
            'current_regime': current_regime,
            'regime_distribution': {r: len(s) for r, s in regime_snapshots.items()},
            'regime_factor_rankings': regime_factor_ic,
            'cross_regime_stability': cross_regime_stability[:15],
            'recommendations': recommendations,
            'interpretation': {
                'BULL': '상승장 — Momentum, Trend 팩터 우세 경향',
                'BEAR': '하락장 — Low-Vol, Quality, 역모멘텀 우세 경향',
                'TRANSITION': '전환기 — Value, Breakout 팩터 기회',
                'all_weather': '모든 레짐에서 같은 방향의 IC → 레짐 불변 팩터',
            },
        }

    # ------------------------------------------------------------------
    # UNIFIED RANKING: 5개 방법론 종합 팩터 순위
    # ------------------------------------------------------------------

    def unified_ranking(self, fm: dict, ic: dict, ls: dict,
                        pca: dict, rc: dict) -> dict:
        """
        5개 방법론의 결과를 종합하여 최종 팩터 우열 순위 산출.

        Scoring:
            FM: |t-stat| 기반 순위점수
            IC: |IC IR| 기반 순위점수
            LS: |ann_sharpe| 기반 순위점수
            PCA: top-3 PC와의 상관 기반 가산점
            RC: current regime IC 기반 가산점

        Returns unified factor ranking.
        """
        # Collect scores from each method
        factor_scores = defaultdict(lambda: {
            'fm_rank': 0, 'ic_rank': 0, 'ls_rank': 0,
            'pca_bonus': 0, 'regime_bonus': 0,
            'fm_tstat': 0, 'ic_ir': 0, 'ls_sharpe': 0,
        })

        # FM scores (by t-stat ranking)
        fm_list = fm.get('factor_premiums', [])
        for i, item in enumerate(fm_list):
            f = item['factor']
            rank_score = max(0, len(fm_list) - i) / max(len(fm_list), 1) * 100
            factor_scores[f]['fm_rank'] = round(rank_score, 1)
            factor_scores[f]['fm_tstat'] = item['t_stat']

        # IC scores (by IC IR ranking)
        ic_list = ic.get('factor_ic', [])
        for i, item in enumerate(ic_list):
            f = item['factor']
            rank_score = max(0, len(ic_list) - i) / max(len(ic_list), 1) * 100
            factor_scores[f]['ic_rank'] = round(rank_score, 1)
            factor_scores[f]['ic_ir'] = item['ic_ir']

        # LS scores (by Sharpe ranking)
        ls_list = ls.get('factor_returns', [])
        for i, item in enumerate(ls_list):
            f = item['factor']
            rank_score = max(0, len(ls_list) - i) / max(len(ls_list), 1) * 100
            factor_scores[f]['ls_rank'] = round(rank_score, 1)
            factor_scores[f]['ls_sharpe'] = item['ann_sharpe']

        # PCA bonus: factors that appear in top-3 PCs' top correlations
        pca_comps = pca.get('components', [])
        for comp in pca_comps[:3]:
            for fc in comp.get('top_factor_correlations', [])[:3]:
                f = fc['factor']
                bonus = abs(fc['correlation']) * 20  # max ~20 points
                factor_scores[f]['pca_bonus'] = max(
                    factor_scores[f]['pca_bonus'], round(bonus, 1))

        # Regime bonus: current regime top factors get a boost
        current_regime = rc.get('current_regime', 'TRANSITION')
        regime_factors = rc.get('regime_factor_rankings', {}).get(current_regime, [])
        for i, item in enumerate(regime_factors[:10]):
            f = item['factor']
            bonus = max(0, 10 - i) * 2  # top factor gets 20, 2nd gets 18, etc.
            factor_scores[f]['regime_bonus'] = round(bonus, 1)

        # Compute unified score
        unified = []
        for f, scores in factor_scores.items():
            # Weighted combination: FM 25%, IC 30%, LS 25%, PCA 10%, Regime 10%
            composite = (
                0.25 * scores['fm_rank'] +
                0.30 * scores['ic_rank'] +
                0.25 * scores['ls_rank'] +
                0.10 * scores['pca_bonus'] +
                0.10 * scores['regime_bonus']
            )
            unified.append({
                'rank': 0,  # filled below
                'factor': f,
                'group': _FACTOR_TO_GROUP.get(f, 'Other'),
                'unified_score': round(composite, 1),
                'fm_tstat': scores['fm_tstat'],
                'ic_ir': scores['ic_ir'],
                'ls_sharpe': scores['ls_sharpe'],
                'pca_bonus': scores['pca_bonus'],
                'regime_bonus': scores['regime_bonus'],
                'component_scores': {
                    'fama_macbeth': scores['fm_rank'],
                    'ic_analysis': scores['ic_rank'],
                    'long_short': scores['ls_rank'],
                    'pca': scores['pca_bonus'],
                    'regime': scores['regime_bonus'],
                },
            })

        unified.sort(key=lambda x: -x['unified_score'])
        for i, u in enumerate(unified):
            u['rank'] = i + 1

        # Group-level aggregation
        group_scores = defaultdict(list)
        for u in unified:
            group_scores[u['group']].append(u['unified_score'])

        group_ranking = []
        for grp, scores in group_scores.items():
            group_ranking.append({
                'group': grp,
                'avg_score': round(float(np.mean(scores)), 1),
                'max_score': round(float(np.max(scores)), 1),
                'n_factors': len(scores),
                'top_factor': next(
                    (u['factor'] for u in unified if u['group'] == grp), ''),
            })
        group_ranking.sort(key=lambda x: -x['avg_score'])

        return {
            'unified_ranking': unified,
            'group_ranking': group_ranking,
            'current_regime': current_regime,
            'top3_factors': [u['factor'] for u in unified[:3]],
            'top3_groups': [g['group'] for g in group_ranking[:3]],
        }

    # ------------------------------------------------------------------
    # TICKER SIGNALS: 팩터 유효성 기반 종목별 시그널
    # ------------------------------------------------------------------

    def score_universe(self, ic: dict, unified: dict) -> dict:
        """
        IC 분석 + 통합 팩터 순위를 활용하여 현재 유니버스 전 종목에
        STRONG_LONG / LONG / SHORT / STRONG_SHORT 시그널 부여.

        방법:
        1. IC 분석에서 quality가 NOISE가 아닌 팩터 중 상위 10개 선택
        2. 각 팩터의 IC_mean 부호 = 방향, |IC_IR| = 가중치
        3. 최신 패널 스냅샷에서 각 종목의 팩터값을 횡단면 z-score로 변환
        4. 가중합산 → factor_signal_score
        5. 백분위로 4분류: STRONG_LONG(≥80) / LONG(≥55) / SHORT(≥25) / STRONG_SHORT(<25)
        """
        if not self._panel:
            return {'ticker_signals': [], 'signal_factors_used': []}

        latest = self._panel[0]  # most recent evaluation snapshot
        tickers = latest['tickers']
        n = len(tickers)

        # Select top effective factors (non-NOISE, by |IC_IR|)
        ic_list = ic.get('factor_ic', [])
        effective = [f for f in ic_list if f.get('quality', 'NOISE') != 'NOISE']
        effective = effective[:10]  # top 10

        if not effective:
            return {'ticker_signals': [], 'signal_factors_used': []}

        # Build weights: |IC_IR| * sign(IC_mean)
        factor_weights = {}
        for f_info in effective:
            f = f_info['factor']
            ic_ir = abs(f_info.get('ic_ir', 0))
            direction = 1.0 if f_info.get('ic_mean', 0) >= 0 else -1.0
            factor_weights[f] = ic_ir * direction

        # Normalize weights so they sum to 1
        total_w = sum(abs(w) for w in factor_weights.values())
        if total_w < 1e-10:
            return {'ticker_signals': [], 'signal_factors_used': []}
        factor_weights = {f: w / total_w for f, w in factor_weights.items()}

        # Compute weighted score per ticker
        scores = np.zeros(n)
        factor_contributions = {f: np.zeros(n) for f in factor_weights}

        for f, w in factor_weights.items():
            if f not in latest['factor_values']:
                continue
            vals = latest['factor_values'][f].copy()
            z = _standardize(vals)
            contribution = w * z
            scores += contribution
            factor_contributions[f] = contribution

        # Percentile rank of scores
        score_pctiles = np.zeros(n)
        for i in range(n):
            score_pctiles[i] = _pct_rank(scores[i], scores)

        # Classify
        def _classify(pctile: float, score: float) -> str:
            if pctile >= 80:
                return 'STRONG_LONG'
            elif pctile >= 55:
                return 'LONG'
            elif pctile >= 25:
                return 'SHORT'
            else:
                return 'STRONG_SHORT'

        # Build per-ticker output
        # Get unified score lookup for context
        uni_lookup = {u['factor']: u.get('unified_score', 0)
                      for u in unified.get('unified_ranking', [])}

        ticker_signals = []
        for i, tk in enumerate(tickers):
            etf = self.all_data.get(tk)
            name = etf.name if etf else tk
            category = etf.category if etf else ''

            signal_cls = _classify(score_pctiles[i], scores[i])

            # Top 3 contributing factors for this ticker
            contribs = sorted(
                [(f, float(factor_contributions[f][i])) for f in factor_weights],
                key=lambda x: -abs(x[1])
            )
            top_contribs = [
                {'factor': f, 'contribution': round(c, 4), 'direction': 'bull' if c > 0 else 'bear'}
                for f, c in contribs[:3]
            ]

            # Raw factor values for display
            raw_vals = {}
            for f in list(factor_weights.keys())[:5]:
                if f in latest['factor_values']:
                    raw_vals[f] = round(float(latest['factor_values'][f][i]), 4)

            ticker_signals.append({
                'ticker': tk,
                'name': name,
                'category': category,
                'signal': signal_cls,
                'factor_score': round(float(scores[i]), 4),
                'score_pctile': round(float(score_pctiles[i]), 1),
                'top_contributions': top_contribs,
                'raw_factors': raw_vals,
            })

        # Sort by factor_score descending
        ticker_signals.sort(key=lambda x: -x['factor_score'])

        # Signal distribution
        from collections import Counter
        dist = Counter(s['signal'] for s in ticker_signals)

        # Factors used info
        signal_factors_used = [
            {
                'factor': f,
                'weight': round(w, 4),
                'direction': 'positive' if w > 0 else 'negative',
                'ic_ir': next((x['ic_ir'] for x in effective if x['factor'] == f), 0),
                'quality': next((x['quality'] for x in effective if x['factor'] == f), '?'),
            }
            for f, w in sorted(factor_weights.items(), key=lambda x: -abs(x[1]))
        ]

        return {
            'ticker_signals': ticker_signals,
            'signal_factors_used': signal_factors_used,
            'signal_distribution': dict(dist),
            'n_tickers': n,
            'eval_date': str(latest.get('eval_str', '')),
        }

    # ------------------------------------------------------------------
    # MAIN RUN
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        5가지 방법론을 모두 실행하고 통합 결과를 반환.

        Returns
        -------
        dict with all 5 method results + unified ranking
        """
        # Step 1: Build panel
        self._build_panel()

        if not self._panel:
            print("  ⚠️ Insufficient data for factor efficacy analysis")
            return {'error': 'Insufficient panel data'}

        # Step 2: Run all 5 methods
        print(f"\n  📊 Method 1/5: Fama-MacBeth Cross-Sectional Regression...")
        fm = self.fama_macbeth('1M')
        print(f"     → {len(fm.get('factor_premiums', []))} factors analyzed")

        print(f"  📊 Method 2/5: Information Coefficient Analysis...")
        ic = self.ic_analysis('1M')
        print(f"     → {len(ic.get('factor_ic', []))} factors analyzed")

        print(f"  📊 Method 3/5: Long-Short Factor Portfolio...")
        ls = self.long_short_portfolio('1M')
        print(f"     → {len(ls.get('factor_returns', []))} factors analyzed")

        print(f"  📊 Method 4/5: PCA Statistical Factor Model...")
        pca = self.pca_analysis()
        print(f"     → Effective dimensionality: {pca.get('effective_dimensionality', '?')}")

        print(f"  📊 Method 5/5: Regime-Conditional Factor Premium...")
        rc = self.regime_conditional('1M')
        print(f"     → Current regime: {rc.get('current_regime', '?')}")

        # Step 3: Unified ranking
        print(f"  📊 Computing unified factor ranking...")
        unified = self.unified_ranking(fm, ic, ls, pca, rc)
        print(f"     → Top 3 factors: {unified.get('top3_factors', [])}")
        print(f"     → Top 3 groups:  {unified.get('top3_groups', [])}")

        # Step 4: Score universe → per-ticker signals
        print(f"  📊 Scoring universe (IC-weighted factor signals)...")
        signals = self.score_universe(ic, unified)
        sig_dist = signals.get('signal_distribution', {})
        print(f"     → {signals.get('n_tickers', 0)} tickers scored: "
              f"SL={sig_dist.get('STRONG_LONG', 0)} L={sig_dist.get('LONG', 0)} "
              f"S={sig_dist.get('SHORT', 0)} SS={sig_dist.get('STRONG_SHORT', 0)}")

        # Multi-horizon analysis (supplementary)
        print(f"  📊 Multi-horizon IC analysis (1W, 1M, 3M)...")
        ic_multi = {}
        for period in FORWARD_WINDOWS:
            ic_p = self.ic_analysis(period)
            ic_multi[period] = {
                'top5': [
                    {'factor': x['factor'], 'ic_mean': x['ic_mean'], 'ic_ir': x['ic_ir']}
                    for x in ic_p.get('factor_ic', [])[:5]
                ]
            }

        output = {
            'fama_macbeth': fm,
            'ic_analysis': ic,
            'long_short': ls,
            'pca': pca,
            'regime_conditional': rc,
            'unified': unified,
            'ticker_signals': signals,
            'multi_horizon_ic': ic_multi,
            'scan_time': datetime.now().isoformat(),
            'panel_size': {
                'n_eval_points': len(self._panel),
                'avg_tickers': round(float(np.mean([s['n_tickers'] for s in self._panel])), 0),
            },
        }

        self._print_summary(output)
        return output

    # ------------------------------------------------------------------
    # Console Summary
    # ------------------------------------------------------------------

    def _print_summary(self, output: dict):
        W = 90
        print(f"\n{'='*W}")
        print("  FACTOR EFFICACY ANALYSIS — REVERSE FACTOR MODEL")
        print(f"{'='*W}")

        # Unified Ranking
        unified = output.get('unified', {})
        ranking = unified.get('unified_ranking', [])
        if ranking:
            print(f"\n  {'─'*W}")
            print(f"  UNIFIED FACTOR RANKING (5-Method Composite)")
            print(f"  {'Rk':>3} {'Factor':<22} {'Group':<18} {'Score':>6} | "
                  f"{'FM t':>6} {'IC IR':>6} {'LS SR':>6} {'PCA':>5} {'Reg':>5}")
            print(f"  {'─'*(W-2)}")
            for u in ranking[:15]:
                print(f"  {u['rank']:>3} {u['factor']:<22} {u['group']:<18} "
                      f"{u['unified_score']:>6.1f} | "
                      f"{u['fm_tstat']:>6.2f} {u['ic_ir']:>6.3f} "
                      f"{u['ls_sharpe']:>6.3f} {u['pca_bonus']:>5.1f} "
                      f"{u['regime_bonus']:>5.1f}")

        # Group Ranking
        grp_rank = unified.get('group_ranking', [])
        if grp_rank:
            print(f"\n  {'─'*W}")
            print(f"  FACTOR GROUP RANKING")
            print(f"  {'Rk':>3} {'Group':<20} {'AvgScore':>8} {'MaxScore':>8} {'N':>3} {'TopFactor':<22}")
            print(f"  {'─'*(W-2)}")
            for i, g in enumerate(grp_rank):
                print(f"  {i+1:>3} {g['group']:<20} {g['avg_score']:>8.1f} "
                      f"{g['max_score']:>8.1f} {g['n_factors']:>3} {g['top_factor']:<22}")

        # Regime Analysis
        rc = output.get('regime_conditional', {})
        print(f"\n  {'─'*W}")
        print(f"  REGIME ANALYSIS — Current: {rc.get('current_regime', '?')}")
        print(f"  Distribution: {rc.get('regime_distribution', {})}")
        for rec in rc.get('recommendations', [])[:3]:
            print(f"    → {rec}")

        # PCA Insights
        pca = output.get('pca', {})
        print(f"\n  {'─'*W}")
        print(f"  PCA LATENT FACTOR MODEL")
        print(f"  Effective dimensionality: {pca.get('effective_dimensionality', '?')}")
        print(f"  Top-3 PCs explain: {pca.get('total_var_explained_top3', '?')}% of variance")
        for comp in pca.get('components', [])[:3]:
            top_fcs = comp.get('top_factor_correlations', [])[:3]
            fc_str = ', '.join(f"{fc['factor']}({fc['correlation']:.2f})" for fc in top_fcs)
            print(f"    PC{comp['pc']}: {comp['var_explained']:.1f}% var — {comp['interpretation']} — [{fc_str}]")

        # Multi-horizon
        mh = output.get('multi_horizon_ic', {})
        if mh:
            print(f"\n  {'─'*W}")
            print(f"  MULTI-HORIZON TOP FACTORS (by IC IR)")
            for period, data in mh.items():
                top5 = data.get('top5', [])
                f_str = ', '.join(f"{x['factor']}({x['ic_ir']:.2f})" for x in top5[:3])
                print(f"    {period:>3}: {f_str}")

        print(f"\n{'='*W}")
        print(f"  Factor Efficacy Analysis Complete.")
        print(f"{'='*W}")


###############################################################################
# PUBLIC API
###############################################################################

def run_factor_efficacy(all_data: Dict = None,
                        detector: NaiveDiscoveryDetector = None,
                        lookback_days: int = 365 * 5,
                        include_stocks: bool = True) -> dict:
    """
    Main entry point for Factor Efficacy Analysis.

    Parameters
    ----------
    all_data : dict, optional
        Pre-downloaded {ticker: ETFData} dict. If None, downloads fresh.
    detector : NaiveDiscoveryDetector, optional
        Pre-initialized detector. If None, creates new one.
    lookback_days : int
        Data lookback period in calendar days.
    include_stocks : bool
        Whether to include stock universe.

    Returns
    -------
    dict
        Complete factor efficacy analysis output.
    """
    if all_data is None:
        print("  📥 Downloading universe data...")
        engine = DataEngine(lookback_days=lookback_days, use_realtime=False)
        all_data = engine.download_universe()
        if include_stocks:
            stock_data = engine.download_universe(universe=STOCK_UNIVERSE)
            all_data.update(stock_data)

    if detector is None:
        detector = NaiveDiscoveryDetector()

    fe = FactorEfficacyEngine(all_data, detector)
    return fe.run()


###############################################################################
# CLI
###############################################################################

if __name__ == "__main__":
    import pickle

    # Try loading from scan cache first (uses pre-downloaded data)
    cache_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".scan_cache.pkl"
    )

    # Standalone mode: download fresh data
    print("=" * 70)
    print("  Factor Efficacy Engine — Reverse Factor Model")
    print("  5 Methodologies × Full Universe Analysis")
    print("=" * 70)

    output = run_factor_efficacy(
        lookback_days=365 * 5,
        include_stocks=True,
    )

    # Save results
    result_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f".factor_efficacy_{date.today().isoformat()}.json",
    )
    try:
        import json

        # Convert numpy types for JSON serialization
        def _json_safe(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        class _Encoder(json.JSONEncoder):
            def default(self, o):
                return _json_safe(o)

        with open(result_path, 'w') as f:
            json.dump(output, f, indent=2, cls=_Encoder)
        print(f"\n💾 Results saved: {result_path}")
    except Exception as e:
        print(f"\n⚠️ JSON save failed: {e}")
