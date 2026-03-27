# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Scanner

```bash
# Full scan with stocks included (default __main__ config)
python price_discovery.py

# Run with custom parameters in a script or REPL:
python -c "
from price_discovery import run_scan
df, results, all_data = run_scan(
    lookback_days=365*5,
    custom_date='2026-02-27',
    use_realtime=True,
    include_stocks=True,
)
"
```

Output: a dated PDF file named `Omega(PD_v5[_STK])_YYYYMMDD.pdf` in the working directory.

## Dependencies

```bash
pip install numpy pandas yfinance matplotlib
```

Minimum Python 3.8. No build step, no tests, no linter config.

## Architecture

Single-file script (`price_discovery.py`) organized in 7 sections:

### Signal Architecture
Three orthogonal axes scored 0–100, combined into a `composite` score:
- **TCS** (Trend Continuation Score): `above_sma50` + `golden_cross` + positive `sma50_slope` + `trend_age > 20` — each worth 25 pts
- **TFS** (Trend Formation Score): early trend age (1–15 days) + volume surge + 20-day breakout + slope reversal
- **OER** (Overextension Risk): distance above SMA50 + RSI overbought + proximity to 52-week high
- **RSS** (Relative Strength Score): cross-sectional percentile of `ret_21d`, `ret_63d`, `ret_126d`, `sma50_slope`

**Composite formula**: `0.35 × TCS + 0.30 × TFS + 0.35 × RSS`

**Classification priority** (evaluated top-to-bottom):
1. `⬇️ DOWNTREND` — below SMA50
2. `🟡 OVEREXTENDED` — OER ≥ 60
3. `🔵 FORMATION` — TFS ≥ 50 and trend_age ≤ 20
4. `🟤 EXHAUSTING` — trend_age > 60 and ret_21d < ret_63d/3
5. `🟢 CONTINUATION` — TCS ≥ 75
6. `🟠 NEUTRAL` — everything else

**Portfolio eligibility** requires: not DOWNTREND, not EXHAUSTING, composite ≥ 55, ADV ≥ $5M.

### Key Classes
- **`DataEngine`**: downloads via `yfinance`, applies adj-close factor, optionally injects real-time price via `fast_info`
- **`NaiveDiscoveryDetector`**: stateless scoring — `compute_raw()` → `score_tcs/tfs/oer()` → `classify()`. Cross-sectional `compute_percentile_ranks()` requires the full universe at once.
- **`SignalValidityEngine`**: backtests signal quality by replaying 12 evaluation points over the past 22 trading days; produces bucket/class/per-ticker hit rates and a class transition matrix
- **`VizEngine`**: all PDF pages. Dark-theme matplotlib. `_text_page()` renders monospace text pages; other methods render bar/line charts.

### Data Flow in `run_scan()`
1. Download all ETFs → optionally merge stock universe
2. Load benchmarks (per-category, fallback to SPY)
3. Compute `all_raw` (per-ticker indicator dict)
4. Compute cross-sectional `all_ranks` (percentiles over full universe)
5. Run `SignalValidityEngine.compute()` (backtests signals)
6. Score + classify each ticker, compute 1W/1M/3M/custom historical snapshots
7. Console output → 7-day history → PDF export

### Universes
- `GLOBAL_ETF_UNIVERSE`: ~180 ETFs across 12 categories
- `STOCK_UNIVERSE`: individual stocks (M7, semiconductors, etc.) — enabled via `include_stocks=True`
- `CATEGORY_BENCHMARK`: per-category benchmark ticker for excess return calculation
- `STOCK_BENCHMARK`: benchmark overrides for stock categories

### Adding Tickers
Add to the appropriate dict in `GLOBAL_ETF_UNIVERSE` or `STOCK_UNIVERSE`. Korean ETFs use `.KS` suffix (e.g., `069500.KS`). Minimum 60 trading days of data required for any ticker to be scored.
