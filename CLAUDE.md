# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

For deeper documentation, see [`docs/`](docs/) — architecture, scoring details, data pipeline, API reference, and the rendered dependency graph at [`reports/score_dependency_graph.pdf`](reports/score_dependency_graph.pdf).

---

## What this project does

Price Discovery is a **multi-asset momentum scanner + investment workflow dashboard**. It scores 770 tickers (~232 ETFs + ~538 stocks across US/Korea/Japan/Europe/China/India) every day on:

1. **Momentum Composite (4-axis technical)** — "where momentum IS now"
2. **Pre-Momentum Score (5-agent forward-looking)** — "where momentum WILL BE"
3. **QVR (Quality + Value + Revision)** — fundamentals dimension, also acts as eligibility filter

Output surfaces: PDF report (`reports/`), React dashboard (FastAPI backend + Vite frontend in `frontend/`), and the legacy Streamlit dashboard (`legacy/dashboard.py`).

---

## Common commands

### Daily refresh (full pipeline)

```bash
# 1. Full scan — produces .scan_cache.pkl (technical signals)
python3 price_discovery.py

# 2. Fetch fundamentals (yfinance, ~5 min for 770 tickers)
python3 fundamentals_pipeline.py
python3 fundamentals_pipeline.py --retry-failed     # if rate-limited

# 3. Enrich with Finnhub (US tickers only, ~28 min)
python3 finnhub_fundamentals.py

# 4. Restart API to pick up the new cache
lsof -ti:8000 | xargs kill 2>/dev/null
python3 -m uvicorn api:app --port 8000 &
```

### Frontend dev

```bash
cd frontend
npm install                    # first time only
npm run dev                    # dev server on http://localhost:5173
npm run build                  # production build → frontend/dist/
```

### One-off scripts

```bash
# Render the main score-system dependency graph PDF (6 pages, into reports/)
python3 reports/scripts/draw_dependency_graph.py

# Render the US Sector Rotation dependency graph PDF (4 pages, into reports/)
# ★ Re-run this after EVERY change to sector_rotation.py or sector_rotation_backtest.py
python3 reports/scripts/draw_sector_rotation_graph.py

# Quick QVR self-test (uses .fundamentals_cache.pkl)
python3 qvr_agent.py

# Sector rotation backtest CLI (offline test)
python3 sector_rotation_backtest.py
```

### ★ Dependency-graph maintenance rule

When you modify any of the following files, **regenerate the corresponding PDF** so the
visual documentation stays in sync. The graphs are visual contracts — drift between
diagram and code makes onboarding harder.

| Code change in… | Regenerate PDF |
|---|---|
| `price_discovery.py` (scoring axes) / `pre_momentum.py` / `qvr_agent.py` / `api.py` (Eligibility Gate) | `python3 reports/scripts/draw_dependency_graph.py` → `reports/score_dependency_graph.pdf` |
| `sector_rotation.py` / `sector_rotation_backtest.py` | `python3 reports/scripts/draw_sector_rotation_graph.py` → `reports/us_sector_rotation_graph.pdf` |

---

## Repository layout

```
price_discovery.py        — Main scanner entry: data download, indicators, axes, classification
api.py                    — FastAPI backend entry; loads cache, computes QVR + Eligibility Gate

config/                   — Centralized constants
  └── scoring.py            — Composite weights, ELIGIBLE_COMPOSITE, ADV_MIN_USD, QVR_GATE
core/                     — Cross-cutting score primitives
  └── eligibility.py        — evaluate_eligible() (Layer 5 gate)
pipelines/                — Data fetchers
  ├── fundamentals_pipeline.py  — yfinance fundamentals (writes .fundamentals_cache.pkl)
  ├── finnhub_client.py         — Finnhub REST wrapper
  └── finnhub_fundamentals.py   — Finnhub enricher (US tickers, in-place cache update)
agents/                   — Pre-Momentum agents + knowledge graph
  ├── pre_momentum.py       — 5-agent Pre-Momentum framework (Micro/Macro/Graph/Catalyst/QVR)
  ├── qvr_agent.py          — Quality-Value-Revision agent
  └── graph_engine.py       — GraphRAG knowledge graph + Louvain communities
strategies/               — Hedge / quant / rotation
  ├── hedge_strategies.py       — 8 quant strategies for long/short signals
  ├── quant_strategies.py       — Top-pick generators (per-strategy ranking)
  ├── sector_rotation.py        — US Sector Rotation Phase 1+2
  ├── sector_rotation_backtest.py — US Sector Rotation Phase 3 (monthly rebalance backtest)
  └── unified_classifier.py     — Classification validation
ml/                       — ML pipelines + diagnostics
  ├── ml_signal_engine.py / score_ml.py / factor_efficacy.py
  ├── feature_pipeline.py / meta_labeling.py / purged_cv.py
  ├── ablation_harness.py / optimize_params.py / ai_prediction_cache.py
  ├── regime_expert_selector.py / regime_conditional_diagnostic.py
  ├── signal_win_ratio.py / performance_analytics.py
  └── breadth_pipeline.py / macro_features.py / multi_benchmark_validation.py

frontend/                 — React/Vite dashboard (the primary UI)
reports/                  — Daily PDF outputs + dependency graph
  └── scripts/              — Standalone PDF renderers (draw_*_graph.py)
tests/                    — Regression harness
  ├── golden/                — Frozen API response baselines (13 endpoints)
  ├── golden_endpoints.py / capture_golden.py / diff_golden.py
  └── test_no_leakage.py     — Feature/breadth as-of consistency tests
legacy/                   — Streamlit dashboard + one-off simulators
docs/                     — System documentation (long-form reference)
```

Cache files (gitignored, all generated by the daily refresh):
- `.scan_cache.pkl` — full scan results (770 tickers × ~80 fields)
- `.fundamentals_cache.pkl` — yfinance + Finnhub merged fundamentals
- `.finnhub_config.json` — Finnhub API key (free tier)

---

## Score architecture (the core)

### Layer 4 — Final Scores

**Momentum Composite** (0-100):
```
Composite = 0.30·TCS + 0.25·TFS + 0.30·RSS + 0.15·URS
```
- **TCS** (Trend Continuation): SMA20/50/200 distance + slope + trend age (long-horizon weighted 60%)
- **TFS** (Trend Formation): SMA breakout strength + vol_ratio_3d_10d + breakout freshness
- **RSS** (Relative Strength): cross-sectional percentile of 5d/21d/63d/12-1M returns
- **URS** (Underreaction Score): LeadLag + AttnGap + post-event Drift + Dispersion (behavioral overlay)

**Pre-Momentum Score** (0-100):
```
Pre-Mom = 0.20·Microstructure + 0.15·MacroRegime + 0.20·GraphRelational
        + 0.20·Catalyst + 0.25·QVR
```
- **Microstructure**: vol compression, accumulation, structural divergence, range contraction
- **Macro Regime**: peer category breadth, cross-asset alignment (uses peer composites only)
- **Graph Relational**: theme breadth, leader-lagger gap, community momentum (graph-derived)
- **Catalyst**: momentum_acceleration (rss_short - rss_long), strategy_agreement, score_trajectory, reversal_risk_check
- **QVR** (★ only Pre-Mom agent fully orthogonal to Composite — correlation ≈ 0)

**agreement_ratio** (0..1, sidecar):
```
agreement_ratio = count(agent_score > 50) / 5
```
Tiers: ≥0.6 STRONG / 0.4-0.6 MODERATE / 0-0.4 WEAK / 0 NONE.

### Layer 5 — Eligibility Gate (4 conditions, ALL must pass)

```
1. Composite ≥ 55                                (technical strength)
2. classification ∈ bullish set                  (CONTINUATION/FORMATION/RECOVERY/OVEREXTENDED/LAGGING_CATCHUP)
3. ADV ≥ $5M                                     (liquidity floor)
4. asset = ETF  OR  QVR ≥ 40                     (fundamentals sanity check, Stock-only)
```

Pass → **Momentum tab** (eligible). Fail → **Excluded tab** with rejection tag (LowScore, Liq($X.XM), Downtrend, WeakQVR(N), etc.).

### 3 × 3 Classification Matrix (Composite-derived)

| Short \ Long | UP | FLAT | DOWN |
|---|---|---|---|
| **UP** | 🟢 CONTINUATION | 🔵 RECOVERY | 🟣 COUNTER_RALLY |
| **FLAT** | 🟡 CONSOLIDATION | 🟠 NEUTRAL | 🟤 FADING |
| **DOWN** | 🔶 PULLBACK | ⚠️ WEAKENING | ⬇️ DOWNTREND |

Overrides: 🟡 OVEREXTENDED (OER ≥ 60 on bullish), 🔵 FORMATION (rapid short breakout), 🟤 EXHAUSTING, 🔴 CYCLE_PEAK, 🟦 LAGGING_CATCHUP.

---

## Universe & Taxonomy (Option B — unified)

- **Total**: 770 tickers (~232 ETFs + ~538 stocks)
- **Sector** (Level 1, 17 buckets): GICS-aligned sectors + Fixed Income / International / Equity Broad / Macro / Multi-Asset / Alternatives
- **SubTheme** (Level 2, 105 buckets): granular thematic groups (e.g. "Semiconductor Design" — both NVDA stock and SMH ETF map to the same SubTheme)
- **Category** (legacy, ~43 buckets): retained internally for universe management + benchmark mapping; hidden from UI

ETFs and stocks now share the **same SubTheme namespace** — graph propagation, peer aggregation, and cross-sectional ranking work uniformly across asset types.

Universe definitions live in:
- `GLOBAL_ETF_UNIVERSE` (price_discovery.py:176) — ETF tickers grouped by Category
- `STOCK_UNIVERSE` (price_discovery.py:210) — Stock tickers grouped by Category
- `STOCK_THEMES` + `STOCK_THEMES_CONSOLIDATED` (price_discovery.py:622-1106) — Stock SubTheme map
- `ETF_SUBTHEMES` (price_discovery.py:1108+) — ETF SubTheme map (unified with stock themes)
- `SUBTHEME_TO_SECTOR` (api.py:27+) — SubTheme → Sector lookup

### Adding tickers

Add to the appropriate dict in `GLOBAL_ETF_UNIVERSE` or `STOCK_UNIVERSE`. For new stocks, also add to `STOCK_THEMES`. For new ETFs, add to `ETF_SUBTHEMES`. Korean tickers use `.KS` suffix (e.g. `069500.KS`). Minimum 60 trading days of data required for any ticker to be scored.

---

## Data sources

| Source | What | Free tier limit | Coverage |
|---|---|---|---|
| **yfinance** | Prices (OHLCV) + basic fundamentals | sustained ~2k req/hr | All 770 tickers globally |
| **Finnhub** | 70+ ratios, monthly recommendation history, EPS surprises, news | 60 req/min, US-listed only | ~620 of 770 tickers |
| **GraphRAG (internal)** | Knowledge graph of theme/community structure | unlimited | Computed from scan results |

Korean tickers (.KS) bypass Finnhub (free tier blocks them) and rely on yfinance only — accept lower analyst coverage as a known limitation.

---

## Data flow (in `run_scan()`)

1. Download all ETFs → optionally merge stock universe
2. Load benchmarks (per-category, fallback to SPY)
3. Compute `all_raw` (per-ticker indicator dict, ~38 indicators including dual-timeframe)
4. Compute cross-sectional `all_ranks` (percentiles over full universe, rss_short/rss_long)
5. Run `SignalValidityEngine.compute()` (backtests signals over 12 evaluation points × 63 days)
6. Score + classify each ticker, compute 1W/1M/3M/custom historical snapshots
7. Console output → 7-day history → PDF export
8. GraphRAG: build knowledge graph → Louvain community detection → insights → cache

The api.py loader then:
9. Loads `.scan_cache.pkl` + `.fundamentals_cache.pkl`
10. Computes QVR scores for every ticker (cross-sectional Q/V/R percentile)
11. Applies the Eligibility Gate (demotes Stock with QVR < 40 → adds `WeakQVR(<n>)` rejection tag)
12. Serves `/api/table`, `/api/pre-momentum`, etc.

---

## Key classes

- **`DataEngine`** (price_discovery.py): yfinance batch downloader; applies adj-close factor; optionally injects real-time price via `fast_info`.
- **`NaiveDiscoveryDetector`** (price_discovery.py): dual-timeframe scoring — `compute_raw()` → `score_tcs_short/long()`, `score_tfs_short/long()`, `score_oer()`, `score_urs()` → `classify()` (3×3 matrix). Cross-sectional `compute_percentile_ranks()` produces `rss_short/long` + URS sub-signals.
- **`SignalValidityEngine`** (price_discovery.py): backtests signal quality over 12 evaluation points × 63 trading days; produces bucket/class/per-ticker hit rates.
- **`VizEngine`** (price_discovery.py): all PDF report pages. Dark-theme matplotlib.
- **`PriceDiscoveryGraph`** (graph_engine.py): builds knowledge graph from results, runs Louvain community detection, generates theme propagation / ETF-stock divergence / leader-lagger insights. Multi-hop: `query_impact_radius()`, `query_theme_status()`, `query_formation_pipeline()`.
- **`PreMomentumOrchestrator`** (pre_momentum.py): runs 5 agents (Micro/Macro/Graph/Catalyst/QVR), filters universe to pre-momentum candidates, computes pre_momentum_score + agreement_ratio.
- **`QVRAgent`** (qvr_agent.py): cross-sectional percentile rank of Q (margin/ROE), V (inverse PE/PEG/PB), R (EPS revision + bullish_change_3m + EPS surprise). Fed by both yfinance and Finnhub fundamentals.

---

## US Sector Rotation strategy

Two implementations live side-by-side. They share the same scoring math but answer different questions.

### A. In-repo — daily cross-sectional pick list

`strategy_sector_rotation()` in [`quant_strategies.py`](quant_strategies.py:124) runs alongside the other 8 quant strategies on every scan:

- Groups all 770 scored tickers by `category`, computes per-category mean Composite
- Top 3 categories → OVERWEIGHT; bottom 3 (when ≥6 categories) → UNDERWEIGHT
- Picks top 5 tickers within each overweight category by Composite
- Surfaces in `/api/strategies` and the React dashboard's strategy cards

This is a **point-in-time signal** — refreshed each scan, no time series, no backtest.

### B. Downstream port — focused 11 GICS sector ETF rotation (monthly backtest)

Lives in the sibling `daimon` project: `../project/daimon/strategy/us_sector_rotation/`.

Applies the same Momentum Composite formula to **11 SPDR-equivalent sector ETFs only** (`SC_US_TECH/FIN/ENERGY/DISCR/STAPLES/HEALTH/INDUST/MATER/REIT/UTIL/COMM`), reading from Snowflake `MKT100_MARKET_DAILY` instead of yfinance. End-of-month rebalance backtest from 2011-04: score 11 sectors → take top-N by Composite (default 3) → equal-weight → next month-end close-to-close return − turnover × 30 bp. Benchmark is `EQ_SP500` buy & hold.

**Ported faithfully (CLOSE-only mode):**
- TCS / TFS / RSS / URS — formulas identical to `NaiveDiscoveryDetector.score_*` (weights `W_SHORT=0.4 / W_LONG=0.6` each)
- 3×3 Classification matrix + OVEREXTENDED / FORMATION / EXHAUSTING / CYCLE_PEAK / LAGGING_CATCHUP overrides — identical
- `composite = 0.30·TCS + 0.25·TFS + 0.30·RSS + 0.15·URS`
- URS 4 components (LeadLag + AttnGap + Drift + Dispersion) — identical (single-category dispersion is degenerate by design)

**Omitted (input data difference):**
- Volume-derived signals (`vol_ratio`, `vol_ratio_3d_10d`) → defaulted to 1.0; `load_wide_close()` returns CLOSE only. TFS loses its volume term, no Composite reweighting.
- OHLC-derived sidecars (Wyckoff OBV, Ichimoku, Darvas, MFI, distribution_days) — none feed Composite, all dropped
- RSS `range_pct_pctile` term → placeholder 50.0 (no rolling 52w high/low without OHLC). RSS one of five inputs only, mild impact.
- Hysteresis on classification → off (monthly rebalance has low oscillation risk)

**Outputs** (`strategy/us_sector_rotation/data/`):
- `latest_signal.json` — top-N picks + full 11-sector score table at the latest date
- `sector_scores.parquet` — month-end scoring history (long format: code × as_of)
- `monthly_backtest.parquet` — month-by-month portfolio/benchmark returns + cumulative
- `summary_metrics.json` — Sharpe / CAGR / MDD / win-rate, strategy vs benchmark

**Run:** `python scripts/run_pipeline.py -s us_sector_rotation` (in the daimon repo). Streamlit page at `views/page_us_sector_rotation.py`.

### When to update which

- Touching the **scoring math** (TCS/TFS/RSS/URS/Composite/Classification): update `price_discovery.py` here first — it's the source of truth. Then port the change to `daimon/strategy/us_sector_rotation/scoring.py`.
- Touching **how categories are aggregated for the daily pick list**: `quant_strategies.py:strategy_sector_rotation` only.
- Touching **rotation logic, rebalance cadence, backtest universe**: `daimon/strategy/us_sector_rotation/{backtest.py, experiment.py}` only — does not affect this repo.

### Surfacing in the React dashboard

To pipe daimon's monthly backtest results into the React dashboard here, the FastAPI backend (`api.py`) needs to load the daimon parquets and expose them as a new endpoint (e.g. `/api/sector-rotation`). Suggested shape:

```json
{
  "as_of": "YYYY-MM-DD",
  "picks": ["SC_US_TECH", "SC_US_HEALTH", ...],
  "scores": [{"code": "...", "label": "...", "composite": ..., "classification": "...", ...}],
  "backtest": {"cum_strategy": [...], "cum_benchmark": [...], "dates": [...]},
  "metrics": {"strategy": {...}, "benchmark": {...}}
}
```

The frontend can then add a new tab (or extend the existing strategy view) to render top-N picks, 11-sector score table, and the monthly cumulative-return chart vs S&P500.

---

## Frontend (React/Vite)

The primary UI lives in `frontend/`. Tabs (6 total):
1. **Price Discovery** — Pre-Momentum / Momentum / Excluded sub-tabs (lifecycle 3-tier view)
2. **Validation** — backtest hit rates
3. **Market Environment** — macro regime, breadth
4. **Analysis** — ablation, factor efficacy
5. **AI Prediction** — ML signal engine output
6. **Appendix** — universe, descriptions, references, signal efficacy

The sidebar has Sector-based filters (17 sectors) + classification filters + composite range slider. Asset-class toggle (Equity / FICC) maps to specific Sector subsets.

---

## Dependencies

```bash
pip install numpy pandas yfinance matplotlib networkx python-louvain fastapi uvicorn
cd frontend && npm install
```

Minimum Python 3.8. No build step for backend, no tests, no linter config.
