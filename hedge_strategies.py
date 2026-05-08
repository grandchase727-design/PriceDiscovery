"""
Hedge Fund Strategy Scoring Module
===================================
8 strategies for individual ticker Long/Short signal generation.
Each returns a score 0-100 for both Long and Short.

Strategies:
1. O'Neil CANSLIM — already in price_discovery.py (score_oneil_long/short)
2. Minervini SEPA — Stage 2 template + VCP
3. Wyckoff Accumulation/Distribution — Volume-price analysis
4. Ichimoku Kinko Hyo — 5-component Japanese system
5. Darvas Box Breakout — Box formation + breakout
6. Regime Adaptive — Meta-strategy with market regime
7. Institutional Flow — Smart money detection
8. Relative Value — Intra-sector divergence
"""


def _sf(val, default=0.0):
    try:
        r = float(val)
        return r if r == r else default
    except:
        return default


# =============================================================================
# Strategy 2: Minervini SEPA (Specific Entry Point Analysis)
# =============================================================================

def score_minervini_long(raw, ranks=None):
    """Minervini Stage 2 Template + VCP (0-100)"""
    pts = 0

    # (1) Stage 2 Template — price structure (35 pts)
    # Price > SMA150 > SMA200, all rising
    if raw.get('above_sma150'):   pts += 7
    if raw.get('above_sma200'):   pts += 7
    if raw.get('above_sma50'):    pts += 7
    if _sf(raw.get('sma150_slope')) > 0: pts += 7   # SMA150 rising
    if _sf(raw.get('sma200_slope')) > 0: pts += 7   # SMA200 rising 1M+

    # (2) Price position within 52-week range (25 pts)
    # Minervini: at least 25% above 52w low, within 25% of 52w high
    rp = _sf(raw.get('range_pct'))
    if rp >= 75:   pts += 25   # Near 52w high
    elif rp >= 50: pts += 18
    elif rp >= 25: pts += 8    # Minimum threshold

    # (3) Relative Strength (20 pts)
    rs = _sf(ranks.get('rss', 50)) if ranks else 50
    if rs >= 80:   pts += 20
    elif rs >= 70: pts += 15
    elif rs >= 60: pts += 8

    # (4) VCP — Volatility Contraction Pattern (20 pts)
    vcr = _sf(raw.get('vcr', 1.0))
    has_bo = raw.get('breakout_20d') or raw.get('breakout_10d')
    if vcr < 0.5 and has_bo:    pts += 20   # Extremely tight + breakout
    elif vcr < 0.7 and has_bo:  pts += 15   # Tight + breakout
    elif vcr < 0.8 and has_bo:  pts += 10   # Moderate + breakout
    elif vcr < 0.8:             pts += 5    # Contracting, awaiting breakout

    return min(100, pts)


def score_minervini_short(raw, ranks=None):
    """Minervini Stage 4 Decline Template (0-100)"""
    pts = 0

    # (1) Stage 4 — price below key MAs, all declining (35 pts)
    if not raw.get('above_sma200'): pts += 10
    if not raw.get('above_sma150'): pts += 8
    if not raw.get('above_sma50'):  pts += 7
    if _sf(raw.get('sma200_slope')) < 0: pts += 5
    if _sf(raw.get('sma150_slope')) < 0: pts += 5

    # (2) Price near 52w low (25 pts)
    rp = _sf(raw.get('range_pct'))
    if rp < 15:   pts += 25
    elif rp < 30: pts += 18
    elif rp < 50: pts += 8

    # (3) RS weakness (20 pts)
    rs = _sf(ranks.get('rss', 50)) if ranks else 50
    if rs <= 15:   pts += 20
    elif rs <= 25: pts += 15
    elif rs <= 40: pts += 8

    # (4) Volume on decline (20 pts)
    declining = _sf(raw.get('ret_5d')) < 0
    vr = max(_sf(raw.get('vol_ratio')), _sf(raw.get('vol_ratio_3d_10d')))
    if declining and vr > 2.0:   pts += 20
    elif declining and vr > 1.5: pts += 15
    elif declining and vr > 1.2: pts += 8

    return min(100, pts)


# =============================================================================
# Strategy 3: Wyckoff Accumulation/Distribution
# =============================================================================

def score_wyckoff_long(raw, ranks=None):
    """Wyckoff Accumulation Score (0-100): volume-price institutional buying"""
    pts = 0

    # (1) OBV uptrend — hidden accumulation (25 pts)
    obv = _sf(raw.get('obv_slope'))
    if obv > 2.0:   pts += 25   # Strong OBV uptrend
    elif obv > 1.0: pts += 18
    elif obv > 0.3: pts += 10
    elif obv > 0:   pts += 5

    # (2) Low distribution days — no institutional selling (20 pts)
    dd = int(_sf(raw.get('dist_days')))
    if dd <= 1:   pts += 20   # Almost no distribution
    elif dd <= 3: pts += 15
    elif dd <= 5: pts += 8

    # (3) Close position — closes near highs = demand (20 pts)
    cp = _sf(raw.get('avg_close_pos', 0.5))
    if cp >= 0.75:  pts += 20   # Consistent closes near highs
    elif cp >= 0.6: pts += 15
    elif cp >= 0.5: pts += 8

    # (4) Volume surge on advance (20 pts)
    advancing = _sf(raw.get('ret_5d')) > 0
    vr = max(_sf(raw.get('vol_ratio')), _sf(raw.get('vol_ratio_3d_10d')))
    if advancing and vr > 1.8:   pts += 20   # Sign of Strength
    elif advancing and vr > 1.3: pts += 14
    elif advancing and vr > 1.1: pts += 7

    # (5) Spring/markup context (15 pts)
    # Pullback to support then recovery = spring
    sma50_dist = _sf(raw.get('sma50_dist'))
    ret_5d = _sf(raw.get('ret_5d'))
    if -3 < sma50_dist < 3 and ret_5d > 1:
        pts += 15   # Near SMA50 + bouncing = spring
    elif sma50_dist > 0 and ret_5d > 0:
        pts += 8    # Above support + advancing

    return min(100, pts)


def score_wyckoff_short(raw, ranks=None):
    """Wyckoff Distribution Score (0-100): institutional selling"""
    pts = 0

    # (1) OBV downtrend — hidden distribution (25 pts)
    obv = _sf(raw.get('obv_slope'))
    if obv < -2.0:   pts += 25
    elif obv < -1.0: pts += 18
    elif obv < -0.3: pts += 10
    elif obv < 0:    pts += 5

    # (2) High distribution days — active selling (20 pts)
    dd = int(_sf(raw.get('dist_days')))
    if dd >= 8:   pts += 20
    elif dd >= 6: pts += 15
    elif dd >= 4: pts += 8

    # (3) Close position — closes near lows = supply (20 pts)
    cp = _sf(raw.get('avg_close_pos', 0.5))
    if cp <= 0.25:  pts += 20
    elif cp <= 0.35: pts += 15
    elif cp <= 0.45: pts += 8

    # (4) Heavy volume on decline (20 pts)
    declining = _sf(raw.get('ret_5d')) < 0
    vr = max(_sf(raw.get('vol_ratio')), _sf(raw.get('vol_ratio_3d_10d')))
    if declining and vr > 1.8:   pts += 20   # Sign of Weakness
    elif declining and vr > 1.3: pts += 14
    elif declining and vr > 1.1: pts += 7

    # (5) Upthrust/distribution context (15 pts)
    pfh = _sf(raw.get('pct_from_high'))
    ret_5d = _sf(raw.get('ret_5d'))
    if pfh > -5 and ret_5d < -2:
        pts += 15   # Near high + falling = upthrust after distribution
    elif pfh > -10 and ret_5d < -1:
        pts += 8

    return min(100, pts)


# =============================================================================
# Strategy 4: Ichimoku Kinko Hyo
# =============================================================================

def score_ichimoku_long(raw, ranks=None):
    """Ichimoku Long Score (0-100): 5-component bullish assessment"""
    pts = 0

    # (1) Price above cloud (25 pts)
    if raw.get('ichimoku_above_cloud'): pts += 25

    # (2) Tenkan-Kijun bull cross (20 pts)
    if raw.get('ichimoku_tk_bull'): pts += 20

    # (3) Cloud color green — Senkou A > Senkou B (20 pts)
    if raw.get('ichimoku_cloud_green'): pts += 20

    # (4) Chikou span bullish — current > 26 ago (20 pts)
    if raw.get('ichimoku_chikou_bull'): pts += 20

    # (5) Momentum confirmation — price trending up (15 pts)
    if _sf(raw.get('ret_21d')) > 2:   pts += 10
    if _sf(raw.get('sma50_slope')) > 0: pts += 5

    return min(100, pts)


def score_ichimoku_short(raw, ranks=None):
    """Ichimoku Short Score (0-100): 5-component bearish assessment"""
    pts = 0

    # (1) Price below cloud (25 pts)
    if raw.get('ichimoku_below_cloud'): pts += 25

    # (2) Tenkan-Kijun bear cross (20 pts)
    if not raw.get('ichimoku_tk_bull'): pts += 20

    # (3) Cloud color red — Senkou A < Senkou B (20 pts)
    if not raw.get('ichimoku_cloud_green'): pts += 20

    # (4) Chikou span bearish (20 pts)
    if not raw.get('ichimoku_chikou_bull'): pts += 20

    # (5) Momentum confirmation (15 pts)
    if _sf(raw.get('ret_21d')) < -2:    pts += 10
    if _sf(raw.get('sma50_slope')) < 0: pts += 5

    return min(100, pts)


# =============================================================================
# Strategy 5: Darvas Box Breakout
# =============================================================================

def score_darvas_long(raw, ranks=None):
    """Darvas Box Long Score (0-100): breakout from consolidation"""
    pts = 0

    # (1) Box formation — consolidation period (25 pts)
    box_days = int(_sf(raw.get('darvas_box_days')))
    if box_days >= 20:   pts += 25   # Extended consolidation
    elif box_days >= 10: pts += 18
    elif box_days >= 5:  pts += 10

    # (2) Box tightness — range compression (25 pts)
    box_range = _sf(raw.get('darvas_box_range', 100))
    if box_range < 5:    pts += 25   # Very tight box
    elif box_range < 8:  pts += 20
    elif box_range < 12: pts += 14
    elif box_range < 18: pts += 8

    # (3) Breakout confirmation (25 pts)
    if raw.get('darvas_breakout'):
        vr = max(_sf(raw.get('vol_ratio')), _sf(raw.get('vol_ratio_3d_10d')))
        if vr > 1.5:   pts += 25   # Breakout + volume
        elif vr > 1.2: pts += 18
        else:           pts += 10   # Breakout, low volume

    # (4) Trend context (25 pts)
    if raw.get('above_sma50'):    pts += 10
    if raw.get('above_sma200'):   pts += 8
    if _sf(raw.get('sma50_slope')) > 0: pts += 7

    return min(100, pts)


def score_darvas_short(raw, ranks=None):
    """Darvas Box Short Score (0-100): breakdown from support"""
    pts = 0

    # (1) Failed breakout or box breakdown (30 pts)
    if not raw.get('darvas_breakout') and int(_sf(raw.get('darvas_box_days'))) > 10:
        # In box and failing to break out — potential breakdown
        ret_5d = _sf(raw.get('ret_5d'))
        if ret_5d < -2: pts += 30   # Falling within box
        elif ret_5d < 0: pts += 15

    # (2) Below key MAs (25 pts)
    if not raw.get('above_sma50'):  pts += 12
    if not raw.get('above_sma200'): pts += 13

    # (3) Volume on decline (25 pts)
    declining = _sf(raw.get('ret_5d')) < 0
    vr = max(_sf(raw.get('vol_ratio')), _sf(raw.get('vol_ratio_3d_10d')))
    if declining and vr > 1.5:   pts += 25
    elif declining and vr > 1.2: pts += 15
    elif declining:              pts += 8

    # (4) Weak range position (20 pts)
    rp = _sf(raw.get('range_pct'))
    if rp < 20:   pts += 20
    elif rp < 35: pts += 12
    elif rp < 50: pts += 5

    return min(100, pts)


# =============================================================================
# Strategy 6: Regime Adaptive (Bridgewater/AQR style)
# =============================================================================

def compute_regime(spy_raw):
    """Compute market regime from SPY indicators.
    Returns: 'risk_on', 'risk_off', or 'transition'
    """
    if spy_raw is None:
        return 'transition'

    above_200 = spy_raw.get('above_sma200', 0)
    above_50 = spy_raw.get('above_sma50', 0)
    slope_50 = _sf(spy_raw.get('sma50_slope'))
    ret_21d = _sf(spy_raw.get('ret_21d'))
    rsi = _sf(spy_raw.get('rsi', 50))

    bull_count = sum([
        above_200 == 1,
        above_50 == 1,
        slope_50 > 0,
        ret_21d > 0,
        rsi > 50,
    ])

    if bull_count >= 4:
        return 'risk_on'
    elif bull_count <= 1:
        return 'risk_off'
    else:
        return 'transition'


def score_regime_long(raw, ranks=None, regime='transition'):
    """Regime Adaptive Long Score (0-100): regime-weighted momentum"""
    pts = 0

    # Base momentum score (40 pts)
    rs = _sf(ranks.get('rss', 50)) if ranks else 50
    pts += min(40, rs * 0.4)

    # Regime multiplier for bullish signals
    if regime == 'risk_on':
        # Favor momentum and trend following
        if raw.get('above_sma50'):     pts += 15
        if raw.get('golden_cross'):    pts += 10
        if _sf(raw.get('sma50_slope')) > 0: pts += 10
        # Breakout bonus in risk-on
        if raw.get('breakout_20d'):    pts += 10
        vr = max(_sf(raw.get('vol_ratio')), _sf(raw.get('vol_ratio_3d_10d')))
        if vr > 1.3: pts += 15

    elif regime == 'risk_off':
        # Favor defensive: low vol, mean reversion
        vol = _sf(raw.get('realized_vol', 20))
        if vol < 15: pts += 15   # Low vol in risk-off = safe haven
        elif vol < 20: pts += 10
        # Oversold bounce opportunity
        rsi = _sf(raw.get('rsi', 50))
        if rsi < 30: pts += 20  # Deep oversold in risk-off
        elif rsi < 40: pts += 10
        # SMA200 support
        if raw.get('above_sma200'): pts += 15

    else:  # transition
        # Balanced approach
        if raw.get('above_sma50'):  pts += 10
        if raw.get('above_sma200'): pts += 10
        vcr = _sf(raw.get('vcr', 1.0))
        if vcr < 0.8: pts += 10  # Consolidation = safe entry
        if _sf(raw.get('sma50_slope')) > 0: pts += 10
        comp = _sf(raw.get('composite_score', 0))
        if comp > 60: pts += 10

    return min(100, int(pts))


def score_regime_short(raw, ranks=None, regime='transition'):
    """Regime Adaptive Short Score (0-100): regime-weighted bearish signals"""
    pts = 0

    # Base weakness score (40 pts)
    rs = _sf(ranks.get('rss', 50)) if ranks else 50
    pts += min(40, (100 - rs) * 0.4)

    if regime == 'risk_off':
        # Aggressive shorting in risk-off
        if not raw.get('above_sma50'):  pts += 15
        if not raw.get('above_sma200'): pts += 10
        if _sf(raw.get('sma50_slope')) < 0: pts += 10
        # High vol in risk-off = vulnerability
        vol = _sf(raw.get('realized_vol', 20))
        if vol > 30: pts += 15
        elif vol > 25: pts += 8
        dd = int(_sf(raw.get('dist_days')))
        if dd >= 5: pts += 10

    elif regime == 'risk_on':
        # Only short the weakest in risk-on
        if not raw.get('above_sma200'): pts += 10
        if _sf(raw.get('range_pct')) < 20: pts += 15
        if rs <= 15: pts += 15
        # Overextended shorts
        rsi = _sf(raw.get('rsi', 50))
        if rsi > 80: pts += 10
        if _sf(raw.get('pct_from_high')) > -2 and _sf(raw.get('ret_5d')) < -1:
            pts += 10  # Reversal from top

    else:  # transition
        if not raw.get('above_sma50'):  pts += 12
        if not raw.get('above_sma200'): pts += 12
        if _sf(raw.get('ret_21d')) < -3: pts += 8
        if _sf(raw.get('sma50_slope')) < 0: pts += 8
        dd = int(_sf(raw.get('dist_days')))
        if dd >= 4: pts += 10

    return min(100, int(pts))


# =============================================================================
# Strategy 7: Institutional Flow / Smart Money
# =============================================================================

def score_flow_long(raw, ranks=None):
    """Institutional Flow Long Score (0-100): smart money accumulation"""
    pts = 0

    # (1) MFI bullish — money flowing in (25 pts)
    mfi = _sf(raw.get('mfi', 50))
    if mfi >= 70:   pts += 25   # Strong buying pressure
    elif mfi >= 60: pts += 18
    elif mfi >= 50: pts += 10

    # (2) Low distribution day count (20 pts)
    dd = int(_sf(raw.get('dist_days')))
    if dd <= 1:   pts += 20
    elif dd <= 3: pts += 14
    elif dd <= 5: pts += 7

    # (3) Quiet accumulation — sideways + rising OBV (20 pts)
    obv = _sf(raw.get('obv_slope'))
    ret = abs(_sf(raw.get('ret_21d')))
    if obv > 1.0 and ret < 5:
        pts += 20   # Price flat but OBV rising = accumulation
    elif obv > 0.5 and ret < 8:
        pts += 12
    elif obv > 0:
        pts += 5

    # (4) Close position — institutional buying closes near highs (20 pts)
    cp = _sf(raw.get('avg_close_pos', 0.5))
    if cp >= 0.7:  pts += 20
    elif cp >= 0.6: pts += 14
    elif cp >= 0.5: pts += 7

    # (5) Volume climax reversal (15 pts)
    rsi = _sf(raw.get('rsi', 50))
    vr = max(_sf(raw.get('vol_ratio')), _sf(raw.get('vol_ratio_3d_10d')))
    ret_5d = _sf(raw.get('ret_5d'))
    if rsi < 35 and vr > 2.0 and ret_5d > 0:
        pts += 15   # Extreme volume + oversold + bouncing = capitulation buy
    elif rsi < 40 and vr > 1.5 and ret_5d > 0:
        pts += 8

    return min(100, pts)


def score_flow_short(raw, ranks=None):
    """Institutional Flow Short Score (0-100): smart money distribution"""
    pts = 0

    # (1) MFI bearish — money flowing out (25 pts)
    mfi = _sf(raw.get('mfi', 50))
    if mfi <= 30:   pts += 25
    elif mfi <= 40: pts += 18
    elif mfi <= 50: pts += 10

    # (2) High distribution day count (20 pts)
    dd = int(_sf(raw.get('dist_days')))
    if dd >= 8:   pts += 20
    elif dd >= 6: pts += 14
    elif dd >= 4: pts += 7

    # (3) Churning — high volume + narrow range (20 pts)
    vr = max(_sf(raw.get('vol_ratio')), _sf(raw.get('vol_ratio_3d_10d')))
    vcr = _sf(raw.get('vcr', 1.0))
    if vr > 1.5 and vcr > 1.2:
        pts += 20   # High volume + expanding volatility on flat price = churning
    elif vr > 1.3 and vcr > 1.0:
        pts += 12

    # (4) Close position — closes near lows (20 pts)
    cp = _sf(raw.get('avg_close_pos', 0.5))
    if cp <= 0.3:  pts += 20
    elif cp <= 0.4: pts += 14
    elif cp <= 0.5: pts += 7

    # (5) Volume dry-up on rally (15 pts)
    ret_5d = _sf(raw.get('ret_5d'))
    if ret_5d > 0 and vr < 0.7:
        pts += 15   # Price up but no volume = no institutional participation
    elif ret_5d > 0 and vr < 0.9:
        pts += 8

    return min(100, pts)


# =============================================================================
# Strategy 8: Relative Value / Intra-Sector Pairs
# =============================================================================

def compute_category_stats(all_raw):
    """Compute per-category statistics for relative value scoring.
    Returns: {category: {mean_ret_21d, std_ret_21d, mean_ret_63d, std_ret_63d}}
    """
    from collections import defaultdict
    cat_rets = defaultdict(lambda: {'ret_21d': [], 'ret_63d': []})

    for ticker, raw in all_raw.items():
        cat = raw.get('_category', 'Unknown')
        cat_rets[cat]['ret_21d'].append(_sf(raw.get('ret_21d')))
        cat_rets[cat]['ret_63d'].append(_sf(raw.get('ret_63d')))

    import numpy as np
    stats = {}
    for cat, data in cat_rets.items():
        r21 = np.array(data['ret_21d'])
        r63 = np.array(data['ret_63d'])
        stats[cat] = {
            'mean_ret_21d': float(np.mean(r21)) if len(r21) > 0 else 0,
            'std_ret_21d': max(float(np.std(r21)), 1.0) if len(r21) > 1 else 10.0,
            'mean_ret_63d': float(np.mean(r63)) if len(r63) > 0 else 0,
            'std_ret_63d': max(float(np.std(r63)), 1.0) if len(r63) > 1 else 10.0,
            'n': len(r21),
        }
    return stats


def score_relval_long(raw, ranks=None, cat_stats=None):
    """Relative Value Long Score (0-100): sector-underperformer mean-reversion buy"""
    pts = 0
    cat = raw.get('_category', 'Unknown')
    cs = (cat_stats or {}).get(cat)
    if not cs or cs['n'] < 3:
        # Not enough peers — fallback to absolute metrics
        rsi = _sf(raw.get('rsi', 50))
        if rsi < 35:  pts += 40
        elif rsi < 45: pts += 20
        if _sf(raw.get('sma50_dist')) < -5: pts += 30
        if raw.get('above_sma200'): pts += 30
        return min(100, pts)

    # Z-score vs category peers (lower = undervalued relative to peers)
    ret_21d = _sf(raw.get('ret_21d'))
    ret_63d = _sf(raw.get('ret_63d'))
    z_21 = (ret_21d - cs['mean_ret_21d']) / cs['std_ret_21d']
    z_63 = (ret_63d - cs['mean_ret_63d']) / cs['std_ret_63d']

    # (1) Short-term relative underperformance (30 pts)
    # Negative z-score = underperforming peers = potential long
    if z_21 < -2.0:   pts += 30
    elif z_21 < -1.5: pts += 24
    elif z_21 < -1.0: pts += 16
    elif z_21 < -0.5: pts += 8

    # (2) Medium-term relative underperformance (25 pts)
    if z_63 < -2.0:   pts += 25
    elif z_63 < -1.5: pts += 20
    elif z_63 < -1.0: pts += 13
    elif z_63 < -0.5: pts += 6

    # (3) Structural support — not broken, just lagging (25 pts)
    if raw.get('above_sma200'):   pts += 12
    if raw.get('above_sma50'):    pts += 8
    if _sf(raw.get('sma200_slope')) > 0: pts += 5

    # (4) Reversal signal — RSI oversold + OBV divergence (20 pts)
    rsi = _sf(raw.get('rsi', 50))
    obv = _sf(raw.get('obv_slope'))
    if rsi < 35 and obv > 0:
        pts += 20   # Price down but OBV up = accumulation divergence
    elif rsi < 40:
        pts += 8
    elif rsi < 45 and obv > 0.5:
        pts += 10

    return min(100, pts)


def score_relval_short(raw, ranks=None, cat_stats=None):
    """Relative Value Short Score (0-100): sector-outperformer mean-reversion sell"""
    pts = 0
    cat = raw.get('_category', 'Unknown')
    cs = (cat_stats or {}).get(cat)
    if not cs or cs['n'] < 3:
        rsi = _sf(raw.get('rsi', 50))
        if rsi > 75:  pts += 40
        elif rsi > 65: pts += 20
        if _sf(raw.get('pct_from_high')) > -3: pts += 30
        if not raw.get('above_sma200'): pts += 30
        return min(100, pts)

    ret_21d = _sf(raw.get('ret_21d'))
    ret_63d = _sf(raw.get('ret_63d'))
    z_21 = (ret_21d - cs['mean_ret_21d']) / cs['std_ret_21d']
    z_63 = (ret_63d - cs['mean_ret_63d']) / cs['std_ret_63d']

    # (1) Short-term relative outperformance (30 pts)
    # Positive z-score = outperforming peers = potential short (mean reversion)
    if z_21 > 2.0:   pts += 30
    elif z_21 > 1.5: pts += 24
    elif z_21 > 1.0: pts += 16
    elif z_21 > 0.5: pts += 8

    # (2) Medium-term relative outperformance (25 pts)
    if z_63 > 2.0:   pts += 25
    elif z_63 > 1.5: pts += 20
    elif z_63 > 1.0: pts += 13
    elif z_63 > 0.5: pts += 6

    # (3) Overextension signals (25 pts)
    rsi = _sf(raw.get('rsi', 50))
    if rsi > 75:   pts += 15
    elif rsi > 65: pts += 8
    pfh = _sf(raw.get('pct_from_high'))
    if pfh > -2:   pts += 10

    # (4) Distribution signs (20 pts)
    obv = _sf(raw.get('obv_slope'))
    dd = int(_sf(raw.get('dist_days')))
    if obv < -0.5 and dd >= 3:
        pts += 20   # OBV declining + distribution days
    elif obv < 0 or dd >= 4:
        pts += 10
    elif dd >= 3:
        pts += 5

    return min(100, pts)


# =============================================================================
# Combined Signal Computation
# =============================================================================

STRATEGY_NAMES = [
    'oneil', 'minervini', 'wyckoff', 'ichimoku',
    'darvas', 'regime', 'flow', 'relval',
]

STRATEGY_WEIGHTS = {
    'oneil': 1.5,       # Proven, well-calibrated
    'minervini': 1.3,   # Strong track record
    'wyckoff': 1.2,     # Institutional flow is high-signal
    'ichimoku': 1.0,    # Independent system
    'darvas': 0.8,      # Overlaps with O'Neil/Minervini
    'regime': 1.2,      # Context-dependent, important
    'flow': 1.1,        # Smart money detection
    'relval': 0.9,      # Mean-reversion, contrarian
}


def compute_combined_signal(scores):
    """
    Combine 8 strategy scores into a single Long/Short signal.

    scores: dict with keys like 'oneil_long', 'oneil_short', etc.

    Returns: {
        'combined_long': float (0-100),
        'combined_short': float (0-100),
        'long_count': int,   # strategies with long >= 60
        'short_count': int,  # strategies with short >= 60
        'net_signal': str,   # STRONG_LONG / LONG / NEUTRAL / SHORT / STRONG_SHORT
        'conviction': float, # 0-100
    }
    """
    long_scores = []
    short_scores = []
    long_count = 0
    short_count = 0
    total_weight = 0

    weighted_long = 0
    weighted_short = 0

    for strat in STRATEGY_NAMES:
        w = STRATEGY_WEIGHTS.get(strat, 1.0)
        ls = _sf(scores.get(f'{strat}_long', 0))
        ss = _sf(scores.get(f'{strat}_short', 0))

        weighted_long += ls * w
        weighted_short += ss * w
        total_weight += w

        long_scores.append(ls)
        short_scores.append(ss)

        if ls >= 60:
            long_count += 1
        if ss >= 60:
            short_count += 1

    if total_weight > 0:
        combined_long = weighted_long / total_weight
        combined_short = weighted_short / total_weight
    else:
        combined_long = 0
        combined_short = 0

    # Net signal determination
    net = combined_long - combined_short
    conviction = abs(net)

    if long_count >= 6 and net > 20:
        signal = 'STRONG_LONG'
    elif long_count >= 4 and net > 10:
        signal = 'LONG'
    elif short_count >= 6 and net < -20:
        signal = 'STRONG_SHORT'
    elif short_count >= 4 and net < -10:
        signal = 'SHORT'
    else:
        signal = 'NEUTRAL'

    return {
        'combined_long': round(combined_long, 1),
        'combined_short': round(combined_short, 1),
        'long_count': long_count,
        'short_count': short_count,
        'net_signal': signal,
        'conviction': round(conviction, 1),
    }


def score_all_strategies(raw, ranks=None, regime='transition', cat_stats=None):
    """Score all 7 new strategies for a single ticker.
    O'Neil is scored separately in price_discovery.py.

    Returns dict with keys: minervini_long/short, wyckoff_long/short, etc.
    """
    return {
        'minervini_long': score_minervini_long(raw, ranks),
        'minervini_short': score_minervini_short(raw, ranks),
        'wyckoff_long': score_wyckoff_long(raw, ranks),
        'wyckoff_short': score_wyckoff_short(raw, ranks),
        'ichimoku_long': score_ichimoku_long(raw, ranks),
        'ichimoku_short': score_ichimoku_short(raw, ranks),
        'darvas_long': score_darvas_long(raw, ranks),
        'darvas_short': score_darvas_short(raw, ranks),
        'regime_long': score_regime_long(raw, ranks, regime),
        'regime_short': score_regime_short(raw, ranks, regime),
        'flow_long': score_flow_long(raw, ranks),
        'flow_short': score_flow_short(raw, ranks),
        'relval_long': score_relval_long(raw, ranks, cat_stats),
        'relval_short': score_relval_short(raw, ranks, cat_stats),
    }
