"""
Quant Strategies Module for Price Discovery Scanner v5.0
========================================================
Takes scan results and computes signals for 6 systematic strategies.

Each strategy returns:
  - name, description
  - signal_summary: {long, neutral, short}
  - picks: top conviction list with ticker, name, sector, signal, score, composite, ret_1m, reason
  - sector_allocation: % weights by sector/category
  - metrics: strategy-specific stats
"""

from collections import defaultdict


def _category_display(cat):
    """Shorten category for display."""
    return cat.replace("STK_", "").replace("EQ_", "")


def _safe(val, default=0.0):
    """Safe numeric extraction."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if f == f else default  # NaN check
    except (TypeError, ValueError):
        return default


def _percentile_rank(value, values):
    """Compute percentile rank of value within a list of values (0-100)."""
    vals = sorted([v for v in values if v is not None and v == v])
    if len(vals) < 2:
        return 50.0
    below = sum(1 for v in vals if v < value)
    return below / (len(vals) - 1) * 100


# =============================================================================
# Strategy 1: Dual Momentum (Antonacci)
# =============================================================================

def strategy_dual_momentum(results):
    """
    Absolute momentum: ret_12_1m > 0 -> invest, else cash
    Relative momentum: RSS percentile rank -> top 20%
    Signal: LONG if abs_mom > 0 AND rel_mom >= 80th percentile
            CASH if abs_mom <= 0
    """
    long_tickers = []
    neutral_tickers = []
    cash_tickers = []

    for r in results:
        ret_12_1m = _safe(r.get('ret_12_1m'))
        rss = _safe(r.get('rss'))
        abs_mom = ret_12_1m > 0
        rel_mom = rss >= 80

        if abs_mom and rel_mom:
            long_tickers.append(r)
        elif abs_mom:
            neutral_tickers.append(r)
        else:
            cash_tickers.append(r)

    # Top 10 by RSS among longs
    long_tickers.sort(key=lambda x: -_safe(x.get('rss')))
    picks = []
    for r in long_tickers[:10]:
        picks.append({
            'ticker': r['ticker'],
            'name': r.get('name', ''),
            'sector': _category_display(r.get('category', '')),
            'signal': 'LONG',
            'score': round(_safe(r.get('rss')), 1),
            'composite': round(_safe(r.get('composite')), 1),
            'ret_1m': round(_safe(r.get('ret_1m')), 2),
            'reason': f"Abs mom: {_safe(r.get('ret_12_1m')):.1f}% | RSS: {_safe(r.get('rss')):.0f}",
        })

    # Sector allocation from picks
    sector_alloc = defaultdict(float)
    if picks:
        weight = 100.0 / len(picks)
        for p in picks:
            sector_alloc[p['sector']] += weight

    return {
        'name': 'Dual Momentum',
        'description': (
            'Antonacci dual momentum: requires both absolute momentum (12-1M return > 0) '
            'and relative momentum (RSS >= 80th percentile). Top 10 by RSS. '
            'Monthly rebalance.'
        ),
        'signal_summary': {
            'long': len(long_tickers),
            'neutral': len(neutral_tickers),
            'short': len(cash_tickers),
        },
        'picks': picks,
        'sector_allocation': dict(sector_alloc),
        'metrics': {
            'n_long': len(long_tickers),
            'n_neutral': len(neutral_tickers),
            'n_cash': len(cash_tickers),
            'avg_rss_long': round(
                sum(_safe(r.get('rss')) for r in long_tickers) / max(len(long_tickers), 1), 1
            ),
            'avg_ret12m_long': round(
                sum(_safe(r.get('ret_12_1m')) for r in long_tickers) / max(len(long_tickers), 1), 1
            ),
        },
    }


# =============================================================================
# Strategy 2: Sector Rotation
# =============================================================================

def strategy_sector_rotation(results):
    """
    Rank sectors (categories) by average composite score.
    Go long top 3 sectors, avoid bottom 3.
    Within top sectors, select highest composite tickers.
    """
    # Group by category
    cat_scores = defaultdict(list)
    cat_results = defaultdict(list)
    for r in results:
        cat = r.get('category', 'Unknown')
        cat_scores[cat].append(_safe(r.get('composite')))
        cat_results[cat].append(r)

    # Rank categories
    cat_avg = {cat: sum(scores) / len(scores) for cat, scores in cat_scores.items() if scores}
    sorted_cats = sorted(cat_avg.items(), key=lambda x: -x[1])

    top_cats = [c for c, _ in sorted_cats[:3]]
    bottom_cats = [c for c, _ in sorted_cats[-3:]] if len(sorted_cats) >= 6 else []

    long_count = 0
    short_count = 0
    neutral_count = 0
    picks = []

    for r in results:
        cat = r.get('category', 'Unknown')
        if cat in top_cats:
            long_count += 1
        elif cat in bottom_cats:
            short_count += 1
        else:
            neutral_count += 1

    # Select top 5 per overweight sector
    for cat in top_cats:
        ranked = sorted(cat_results[cat], key=lambda x: -_safe(x.get('composite')))
        for r in ranked[:5]:
            picks.append({
                'ticker': r['ticker'],
                'name': r.get('name', ''),
                'sector': _category_display(cat),
                'signal': 'OVERWEIGHT',
                'score': round(_safe(r.get('composite')), 1),
                'composite': round(_safe(r.get('composite')), 1),
                'ret_1m': round(_safe(r.get('ret_1m')), 2),
                'reason': f"Top sector avg: {cat_avg[cat]:.1f} | Cat rank: {top_cats.index(cat)+1}/3",
            })

    picks.sort(key=lambda x: -x['score'])

    # Sector allocation
    sector_alloc = {}
    for cat, avg in sorted_cats:
        display = _category_display(cat)
        if cat in top_cats:
            sector_alloc[display] = round(100.0 / max(len(top_cats), 1), 1)
        elif cat in bottom_cats:
            sector_alloc[display] = 0.0

    return {
        'name': 'Sector Rotation',
        'description': (
            'Rank all categories by average composite score. '
            'Overweight top 3 sectors, underweight bottom 3. '
            'Select top 5 tickers per overweight sector.'
        ),
        'signal_summary': {
            'long': long_count,
            'neutral': neutral_count,
            'short': short_count,
        },
        'picks': picks,
        'sector_allocation': sector_alloc,
        'metrics': {
            'top_sectors': [
                {'sector': _category_display(c), 'avg_composite': round(cat_avg[c], 1)}
                for c in top_cats
            ],
            'bottom_sectors': [
                {'sector': _category_display(c), 'avg_composite': round(cat_avg[c], 1)}
                for c in bottom_cats
            ],
            'n_categories': len(cat_avg),
        },
    }


# =============================================================================
# Strategy 3: Trend Following (CTA-style)
# =============================================================================

def strategy_trend_following(results):
    """
    Multi-timeframe trend:
    - Entry: above_sma50 AND above_sma200 AND golden_cross AND sma50_slope > 0
    - Exit: below sma50 OR sma50_slope < 0
    - Position size: inverse volatility (1/realized_vol)
    Signal: classification in (CONTINUATION, FORMATION) -> LONG
            classification in (DOWNTREND, FADING) -> SHORT
    """
    LONG_CLS = {'CONTINUATION', 'FORMATION', 'RECOVERY'}
    SHORT_CLS = {'DOWNTREND', 'FADING', 'WEAKENING'}

    long_tickers = []
    short_tickers = []
    neutral_tickers = []

    for r in results:
        cls = r.get('classification', '')
        # Extract core classification (strip emoji prefix)
        cls_core = cls.split()[-1] if cls else ''

        above_sma50 = r.get('above_sma50', False)
        above_sma200 = r.get('above_sma200', False)
        golden_cross = r.get('golden_cross', False)
        sma50_slope = _safe(r.get('sma50_slope'))

        # Full entry: structural trend + MA confirmation
        full_entry = above_sma50 and above_sma200 and sma50_slope > 0
        # Exit signal
        exit_signal = not above_sma50 or sma50_slope < 0

        if full_entry and cls_core in LONG_CLS:
            long_tickers.append(r)
        elif exit_signal and cls_core in SHORT_CLS:
            short_tickers.append(r)
        else:
            neutral_tickers.append(r)

    # Rank longs by TCS * inverse_vol
    def _trend_score(r):
        tcs = _safe(r.get('tcs'))
        vol = max(_safe(r.get('realized_vol'), 20.0), 1.0)
        return tcs / vol

    long_tickers.sort(key=lambda x: -_trend_score(x))

    picks = []
    for r in long_tickers[:15]:
        vol = max(_safe(r.get('realized_vol'), 20.0), 1.0)
        picks.append({
            'ticker': r['ticker'],
            'name': r.get('name', ''),
            'sector': _category_display(r.get('category', '')),
            'signal': 'LONG',
            'score': round(_trend_score(r) * 100, 1),
            'composite': round(_safe(r.get('composite')), 1),
            'ret_1m': round(_safe(r.get('ret_1m')), 2),
            'reason': (
                f"TCS: {_safe(r.get('tcs')):.0f} | Vol: {vol:.1f}% | "
                f"SMA50+200: {'Y' if r.get('above_sma200') else 'N'} | "
                f"GC: {'Y' if r.get('golden_cross') else 'N'}"
            ),
        })

    # Inverse-vol weighted allocation
    sector_alloc = defaultdict(float)
    total_inv_vol = 0
    for r in long_tickers[:15]:
        vol = max(_safe(r.get('realized_vol'), 20.0), 1.0)
        total_inv_vol += 1.0 / vol
    if total_inv_vol > 0:
        for r in long_tickers[:15]:
            vol = max(_safe(r.get('realized_vol'), 20.0), 1.0)
            wt = (1.0 / vol) / total_inv_vol * 100
            sector_alloc[_category_display(r.get('category', ''))] += wt

    return {
        'name': 'Trend Following',
        'description': (
            'CTA-style trend following: enter when above SMA50+SMA200 with positive SMA50 slope '
            'and bullish classification. Position sized by inverse volatility. '
            'Exit on SMA50 break or slope reversal.'
        ),
        'signal_summary': {
            'long': len(long_tickers),
            'neutral': len(neutral_tickers),
            'short': len(short_tickers),
        },
        'picks': picks,
        'sector_allocation': {k: round(v, 1) for k, v in sector_alloc.items()},
        'metrics': {
            'n_long': len(long_tickers),
            'n_neutral': len(neutral_tickers),
            'n_short': len(short_tickers),
            'avg_tcs_long': round(
                sum(_safe(r.get('tcs')) for r in long_tickers) / max(len(long_tickers), 1), 1
            ),
            'avg_vol_long': round(
                sum(_safe(r.get('realized_vol'), 20.0) for r in long_tickers) / max(len(long_tickers), 1), 1
            ),
        },
    }


# =============================================================================
# Strategy 4: Multi-Factor
# =============================================================================

def strategy_multi_factor(results):
    """
    3-factor composite:
    - Momentum (40%): RSS percentile
    - Quality (30%): structural_q score
    - Low Vol (30%): inverse realized_vol percentile
    Combined score -> rank -> top 15
    """
    # Compute percentile arrays
    rss_values = [_safe(r.get('rss')) for r in results]
    sq_values = [_safe(r.get('structural_q')) for r in results]
    vol_values = [_safe(r.get('realized_vol'), 20.0) for r in results]
    # For low vol: invert so lower vol = higher score
    inv_vol_values = [100.0 / max(v, 1.0) for v in vol_values]

    scored = []
    for i, r in enumerate(results):
        rss_pctile = _percentile_rank(rss_values[i], rss_values)
        sq_pctile = _percentile_rank(sq_values[i], sq_values)
        inv_vol_pctile = _percentile_rank(inv_vol_values[i], inv_vol_values)

        mf_score = 0.4 * rss_pctile + 0.3 * sq_pctile + 0.3 * inv_vol_pctile
        scored.append((r, mf_score, rss_pctile, sq_pctile, inv_vol_pctile))

    scored.sort(key=lambda x: -x[1])

    # Thresholds
    n = len(scored)
    long_threshold = 70
    short_threshold = 30

    long_count = sum(1 for _, s, _, _, _ in scored if s >= long_threshold)
    short_count = sum(1 for _, s, _, _, _ in scored if s <= short_threshold)
    neutral_count = n - long_count - short_count

    picks = []
    for r, mf_score, rss_p, sq_p, vol_p in scored[:15]:
        picks.append({
            'ticker': r['ticker'],
            'name': r.get('name', ''),
            'sector': _category_display(r.get('category', '')),
            'signal': 'LONG' if mf_score >= long_threshold else 'NEUTRAL',
            'score': round(mf_score, 1),
            'composite': round(_safe(r.get('composite')), 1),
            'ret_1m': round(_safe(r.get('ret_1m')), 2),
            'reason': (
                f"Mom: {rss_p:.0f} | Qual: {sq_p:.0f} | LowVol: {vol_p:.0f} "
                f"(vol={_safe(r.get('realized_vol'), 20):.1f}%)"
            ),
        })

    # Sector allocation from top 15
    sector_alloc = defaultdict(float)
    if picks:
        wt = 100.0 / len(picks)
        for p in picks:
            sector_alloc[p['sector']] += wt

    return {
        'name': 'Multi-Factor',
        'description': (
            '3-factor composite: Momentum (40%, RSS percentile) + Quality (30%, structural_q) + '
            'Low Volatility (30%, inverse realized_vol). Top 15 by combined score.'
        ),
        'signal_summary': {
            'long': long_count,
            'neutral': neutral_count,
            'short': short_count,
        },
        'picks': picks,
        'sector_allocation': {k: round(v, 1) for k, v in sector_alloc.items()},
        'metrics': {
            'n_long': long_count,
            'n_neutral': neutral_count,
            'n_short': short_count,
            'avg_mf_score_top15': round(
                sum(s for _, s, _, _, _ in scored[:15]) / min(15, max(len(scored), 1)), 1
            ),
            'factor_avg': {
                'momentum': round(sum(rss_p for _, _, rss_p, _, _ in scored) / max(n, 1), 1),
                'quality': round(sum(sq_p for _, _, _, sq_p, _ in scored) / max(n, 1), 1),
                'low_vol': round(sum(vol_p for _, _, _, _, vol_p in scored) / max(n, 1), 1),
            },
        },
    }


# =============================================================================
# Strategy 5: Mean Reversion
# =============================================================================

def strategy_mean_reversion(results):
    """
    Oversold bounce:
    - RSI < 35 AND sma50_dist < -5% AND classification in (PULLBACK, WEAKENING)
    - Additional: trend_age > 0 (still above SMA50 at some point recently)
    Signal: BUY when oversold in uptrend context
    """
    OVERSOLD_CLS = {'PULLBACK', 'WEAKENING', 'CONSOLIDATION'}

    candidates = []
    for r in results:
        rsi = _safe(r.get('rsi'), 50)
        sma50_dist = _safe(r.get('sma50_dist'))
        cls = r.get('classification', '')
        cls_core = cls.split()[-1] if cls else ''
        trend_age = _safe(r.get('trend_age'))
        above_sma200 = r.get('above_sma200', False)

        # Oversold conditions
        is_oversold = rsi < 35 and sma50_dist < -5
        # Context: not in full downtrend (still has some structural support)
        has_context = cls_core in OVERSOLD_CLS or (above_sma200 and sma50_dist > -15)

        if is_oversold and has_context:
            candidates.append(r)

    # Rank by RSI (lower = more oversold = higher priority)
    candidates.sort(key=lambda x: _safe(x.get('rsi'), 50))

    n_buy = len(candidates)
    n_total = len(results)
    n_neutral = n_total - n_buy

    picks = []
    for r in candidates[:10]:
        picks.append({
            'ticker': r['ticker'],
            'name': r.get('name', ''),
            'sector': _category_display(r.get('category', '')),
            'signal': 'BUY',
            'score': round(100 - _safe(r.get('rsi'), 50), 1),  # inverse RSI as score
            'composite': round(_safe(r.get('composite')), 1),
            'ret_1m': round(_safe(r.get('ret_1m')), 2),
            'reason': (
                f"RSI: {_safe(r.get('rsi')):.1f} | SMA50 dist: {_safe(r.get('sma50_dist')):.1f}% | "
                f"Class: {r.get('classification', '')[:15]}"
            ),
        })

    sector_alloc = defaultdict(float)
    if picks:
        wt = 100.0 / len(picks)
        for p in picks:
            sector_alloc[p['sector']] += wt

    return {
        'name': 'Mean Reversion',
        'description': (
            'Oversold bounce strategy: RSI < 35 with SMA50 distance < -5%, '
            'in pullback/weakening classification with structural support. '
            'Ranked by RSI (lower = higher conviction). Max 10 picks.'
        ),
        'signal_summary': {
            'long': n_buy,
            'neutral': n_neutral,
            'short': 0,
        },
        'picks': picks,
        'sector_allocation': {k: round(v, 1) for k, v in sector_alloc.items()},
        'metrics': {
            'n_oversold': n_buy,
            'n_universe': n_total,
            'avg_rsi_picks': round(
                sum(_safe(r.get('rsi'), 50) for r in candidates[:10]) / max(len(candidates[:10]), 1), 1
            ),
            'avg_sma50_dist_picks': round(
                sum(_safe(r.get('sma50_dist')) for r in candidates[:10]) / max(len(candidates[:10]), 1), 1
            ),
        },
    }


# =============================================================================
# Strategy 6: Inverse Vol (Risk Parity Simplified)
# =============================================================================

def strategy_inverse_vol(results):
    """
    Simplified risk parity:
    - Weight = 1/realized_vol for each eligible ticker
    - Normalize weights to sum to 100%
    - Higher allocation to lower volatility assets
    """
    eligible = [r for r in results if r.get('eligible')]
    if not eligible:
        # Fallback to top 50 by composite
        eligible = sorted(results, key=lambda x: -_safe(x.get('composite')))[:50]

    # Compute inverse vol weights
    weighted = []
    total_inv_vol = 0
    for r in eligible:
        vol = max(_safe(r.get('realized_vol'), 20.0), 1.0)
        inv_vol = 1.0 / vol
        total_inv_vol += inv_vol
        weighted.append((r, inv_vol))

    if total_inv_vol == 0:
        total_inv_vol = 1.0

    # Normalize and sort by weight
    weighted_normed = [(r, inv_vol / total_inv_vol * 100) for r, inv_vol in weighted]
    weighted_normed.sort(key=lambda x: -x[1])

    picks = []
    for r, wt in weighted_normed[:15]:
        vol = max(_safe(r.get('realized_vol'), 20.0), 1.0)
        picks.append({
            'ticker': r['ticker'],
            'name': r.get('name', ''),
            'sector': _category_display(r.get('category', '')),
            'signal': 'LONG',
            'score': round(wt, 2),
            'composite': round(_safe(r.get('composite')), 1),
            'ret_1m': round(_safe(r.get('ret_1m')), 2),
            'reason': f"Weight: {wt:.2f}% | Vol: {vol:.1f}% | 1/vol rank",
        })

    # Sector allocation
    sector_alloc = defaultdict(float)
    for r, wt in weighted_normed:
        sector_alloc[_category_display(r.get('category', ''))] += wt

    # Long/neutral split: top 50% weight = long
    n_eligible = len(eligible)
    cumulative_wt = 0
    n_long = 0
    for _, wt in weighted_normed:
        cumulative_wt += wt
        n_long += 1
        if cumulative_wt >= 80:
            break

    return {
        'name': 'Inverse Volatility',
        'description': (
            'Simplified risk parity: weight each eligible ticker by 1/realized_vol, '
            'normalized to 100%. Lower volatility assets get higher allocation. '
            'Top 15 by weight shown.'
        ),
        'signal_summary': {
            'long': n_eligible,
            'neutral': len(results) - n_eligible,
            'short': 0,
        },
        'picks': picks,
        'sector_allocation': {k: round(v, 1) for k, v in sorted(sector_alloc.items(), key=lambda x: -x[1])},
        'metrics': {
            'n_eligible': n_eligible,
            'n_universe': len(results),
            'avg_vol_universe': round(
                sum(_safe(r.get('realized_vol'), 20.0) for r in results) / max(len(results), 1), 1
            ),
            'avg_vol_eligible': round(
                sum(_safe(r.get('realized_vol'), 20.0) for r in eligible) / max(len(eligible), 1), 1
            ),
            'top_weight': round(weighted_normed[0][1], 2) if weighted_normed else 0,
            'concentration_top10': round(
                sum(wt for _, wt in weighted_normed[:10]), 1
            ) if len(weighted_normed) >= 10 else 100.0,
        },
    }


# =============================================================================
# Cross-Strategy Consensus
# =============================================================================

def _compute_consensus(strategies):
    """
    Find tickers that appear in multiple strategies' pick lists.
    Returns list sorted by number of strategy appearances.
    """
    ticker_appearances = defaultdict(lambda: {
        'strategies': [], 'signals': [], 'scores': [],
        'name': '', 'sector': '', 'composite': 0, 'ret_1m': 0
    })

    for strat_key, strat in strategies.items():
        for pick in strat.get('picks', []):
            tk = pick['ticker']
            ticker_appearances[tk]['strategies'].append(strat['name'])
            ticker_appearances[tk]['signals'].append(pick.get('signal', 'LONG'))
            ticker_appearances[tk]['scores'].append(pick.get('score', 0))
            ticker_appearances[tk]['name'] = pick.get('name', '')
            ticker_appearances[tk]['sector'] = pick.get('sector', '')
            ticker_appearances[tk]['composite'] = pick.get('composite', 0)
            ticker_appearances[tk]['ret_1m'] = pick.get('ret_1m', 0)

    # Filter to tickers in 2+ strategies
    consensus = []
    for tk, data in ticker_appearances.items():
        n = len(data['strategies'])
        if n >= 2:
            consensus.append({
                'ticker': tk,
                'name': data['name'],
                'sector': data['sector'],
                'n_strategies': n,
                'strategies': data['strategies'],
                'signals': data['signals'],
                'avg_score': round(sum(data['scores']) / len(data['scores']), 1),
                'composite': data['composite'],
                'ret_1m': data['ret_1m'],
            })

    consensus.sort(key=lambda x: (-x['n_strategies'], -x['avg_score']))
    return consensus


# =============================================================================
# Main entry point
# =============================================================================

def compute_all_strategies(results):
    """Run all 6 strategies on scan results. Returns dict of strategy results + consensus."""
    strategies = {
        'dual_momentum': strategy_dual_momentum(results),
        'sector_rotation': strategy_sector_rotation(results),
        'trend_following': strategy_trend_following(results),
        'multi_factor': strategy_multi_factor(results),
        'mean_reversion': strategy_mean_reversion(results),
        'inverse_vol': strategy_inverse_vol(results),
    }

    consensus = _compute_consensus(strategies)

    return {
        **strategies,
        '_consensus': consensus,
    }
