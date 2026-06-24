# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

For deeper documentation, see [`docs/`](docs/) — architecture, scoring details, data pipeline, API reference, and the rendered dependency graph at [`reports/score_dependency_graph.pdf`](reports/score_dependency_graph.pdf).

---

## What this project does

Price Discovery is a **multi-asset momentum scanner + investment workflow dashboard**. It scores 770 tickers (~232 ETFs + ~538 stocks across US/Korea/Japan/Europe/China/India) every day on:

1. **Momentum Composite (4-axis technical, OER-penalized)** — "where momentum IS now"
2. **Pre-Momentum Score (5-agent forward-looking, rotation-aware)** — "where momentum WILL BE"
3. **QVR (Quality + Value + Revision)** — fundamentals dimension, also acts as eligibility filter

Layered on top (api.py post-load):
- **Macro Context Tags** — cyclical/style/region/industry per-ticker (Phase 1-3)
- **Hybrid Bottom-up** — ETF constituent breadth + divergence flags (Phase A-E)
- **Anti-Lag Discovery** — Pre-Mom PROVISIONAL surfaced into Momentum tier
- **Sector Discovery** — sector-segmented top-N (forced diversification)

Output surfaces: PDF report (`reports/`), React dashboard (FastAPI backend + Vite frontend in `frontend/`), and the legacy Streamlit dashboard (`dashboard.py`).

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
python3 draw_dependency_graph.py

# Render the US Sector Rotation dependency graph PDF (10 pages, into reports/)
# ★ Re-run this after EVERY change to sector_rotation.py or sector_rotation_backtest.py
python3 draw_sector_rotation_graph.py

# Quick QVR self-test (uses .fundamentals_cache.pkl)
python3 qvr_agent.py

# Sector rotation backtest CLI (offline test)
python3 sector_rotation_backtest.py

# Fetch / refresh ETF top-holdings cache for the Hybrid Bottom-up layer
# (70 equity sector + thematic ETFs by default; --skip-existing for incremental)
python3 etf_holdings_pipeline.py
python3 etf_holdings_pipeline.py --skip-existing
```

### ★ Dependency-graph maintenance rule

When you modify any of the following files, **regenerate the corresponding PDF** so the
visual documentation stays in sync. The graphs are visual contracts — drift between
diagram and code makes onboarding harder.

| Code change in… | Regenerate PDF |
|---|---|
| `price_discovery.py` (scoring axes) / `pre_momentum.py` / `qvr_agent.py` / `api.py` (Eligibility Gate) | `python3 draw_dependency_graph.py` → `reports/score_dependency_graph.pdf` |
| `sector_rotation.py` / `sector_rotation_backtest.py` | `python3 draw_sector_rotation_graph.py` → `reports/us_sector_rotation_graph.pdf` |

---

## Repository layout

```
price_discovery.py        — Main scanner: data download, indicators, axes, classification
pre_momentum.py           — 5-agent Pre-Momentum framework (Micro/Macro/Graph/Catalyst/QVR)
qvr_agent.py              — Quality-Value-Revision agent (5th PM agent + Eligibility Gate input)
fundamentals_pipeline.py  — yfinance fundamentals fetcher (writes .fundamentals_cache.pkl)
finnhub_client.py         — Finnhub REST wrapper
finnhub_fundamentals.py   — Finnhub enricher (US tickers, in-place cache update)
graph_engine.py           — GraphRAG knowledge graph + Louvain communities
hedge_strategies.py       — 8 quant strategies for long/short signals
quant_strategies.py       — Top-pick generators (per-strategy ranking)
etf_holdings_pipeline.py  — ETF top-holdings fetcher via yfinance (writes .etf_holdings_cache.json)
unified_classifier.py     — GICS sector/industry/cap-tier classifier (Phase Y)

api.py                    — FastAPI backend; loads cache, computes QVR + Eligibility Gate +
                            macro context tags + ETF hybrid sidecar + Anti-Lag/Sector tiers
dashboard.py              — Legacy Streamlit dashboard (kept for compatibility)
draw_dependency_graph.py  — Renders the 6-page main score-system documentation PDF

sector_rotation.py            — US Sector Rotation Phase 1+2 (tier classification + macro regime overlay)
sector_rotation_backtest.py   — US Sector Rotation Phase 3 (monthly rebalance backtest, yfinance)
                                — Composite includes TFS_resid + OER penalty (Live-engine parity)
draw_sector_rotation_graph.py — Renders the 10-page US Sector Rotation documentation PDF

frontend/                 — React/Vite dashboard (the primary UI)
docs/                     — System documentation (this is the long-form reference)
reports/                  — Daily PDF outputs + dependency graph
```

Cache files (gitignored, all generated by the daily refresh / pipeline scripts):
- `.scan_cache.pkl` — full scan results (770 tickers × ~80 fields)
- `.fundamentals_cache.pkl` — yfinance + Finnhub merged fundamentals
- `.finnhub_config.json` — Finnhub API key (free tier)
- `.etf_holdings_cache.json` — ETF top holdings (Hybrid Bottom-up layer, refresh weekly)
- `.unified_classification.json` — per-ticker GICS sector / industry / cap-tier
- `.ytd_returns.json` — YTD return enrichment

---

## Score architecture (the core)

### Layer 4 — Final Scores

**Momentum Composite** (0-100) — current formula (2026-05 update):
```
base       = 0.30·TCS + 0.25·TFS_resid + 0.30·RSS_hybrid + 0.15·URS
Composite  = base − 0.10·max(0, OER − 40)         # OER penalty, max −6 pts
```
- **TCS** (Trend Continuation): SMA20/50/200 distance + slope + trend age (long-horizon weighted 60%)
- **TFS_resid** (Trend Formation, **residualized vs TCS**): cross-sectional OLS removes TCS-TFS info overlap. `tfs_short/long` (used by `classify()`) keep raw values; only Composite uses TFS_resid.
- **RSS_hybrid** (Relative Strength): `0.6·within_sector + 0.4·universe` percentile (sector beta correction; small categories n<8 fall back to universe-wide)
- **URS** (Underreaction Score): LeadLag + AttnGap + post-event Drift + Dispersion (behavioral overlay)
- **OER penalty**: 40 미만은 무패널티, 100 이면 −6점 (과열 종목과 건전 추세주의 비대칭 해소)

**Pre-Momentum Score** (0-100):
```
Pre-Mom = 0.20·Microstructure + 0.15·MacroRegime + 0.20·GraphRelational
        + 0.20·Catalyst + 0.25·QVR
```
- **Microstructure**: vol compression, accumulation, structural divergence, range contraction
- **Macro Regime** (6 sub-signals): peer category breadth (0.20), cross-asset (0.10), category breadth (0.20), relative improvement (0.15), **rotation_alignment** (0.20, Phase 2C — Risk/Style/Region alignment), **etf_parent_signal** (0.15, Hybrid Phase D — bottom-up ETF flag aggregation)
- **Graph Relational**: theme breadth, leader-lagger gap, community momentum (graph-derived)
- **Catalyst**: momentum_acceleration (rss_short − rss_long), strategy_agreement, score_trajectory, reversal_risk_check
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

### Layer 5b — Anti-Lag / Sector tiers (api.py post-load, **2026-05 addition**)

Two complementary tiers solve different problems:

- **`provisional_eligible`** (ProvisionalPM tier) — surfaces tickers with strong forward-looking Pre-Mom signal even though they didn't pass the main gate:
  ```
  pre_momentum_score ≥ 45  AND  agreement_ratio ≥ 0.6 (STRONG)  AND  bullish classification  AND  NOT eligible
  ```
  → "Anti Lag Discovery" sub-tab. Goal: 10-15-day lag reduction.

- **`sector_segmented_eligible`** (Sector top-N) — within each sector, top-5 by composite among bullish + composite ≥ 40:
  → "Sector Discovery" sub-tab. Goal: forced diversification + sector-best capture (does NOT reduce lag).

`eligibility_tier` (Anti-Lag): `EligibleMomentum | ProvisionalPM | PreMomentum | Excluded`.
`eligibility_tier_v2` (Sector): `BothEligible | UniverseOnly | SectorOnly | Neither`.

### 3 × 3 Classification Matrix (Composite-derived, with Sticky FLAT hysteresis)

| Short \ Long | UP | FLAT | DOWN |
|---|---|---|---|
| **UP** | 🟢 CONTINUATION | 🔵 RECOVERY | 🟣 COUNTER_RALLY |
| **FLAT** | 🟡 CONSOLIDATION | 🟠 NEUTRAL | 🟤 FADING |
| **DOWN** | 🔶 PULLBACK | ⚠️ WEAKENING | ⬇️ DOWNTREND |

Overrides: 🟡 OVEREXTENDED (OER ≥ 60 on bullish), 🔵 FORMATION (rapid short breakout), 🟤 EXHAUSTING, 🔴 CYCLE_PEAK, 🟦 LAGGING_CATCHUP.

**Sticky FLAT hysteresis** (2026-05 addition): when `prev_classification` indicates FLAT direction, entry threshold is multiplied by 1.3× → suppresses bi-weekly NEUTRAL ↔ CONSOLIDATION ↔ RECOVERY ↔ FADING flip noise.

**Phase 3B Regime-aware CYCLE_PEAK override** (api.py post-load): in Risk-Off regime, cyclical ticker + OVEREXTENDED + OER ≥ 70 → upgraded to CYCLE_PEAK (and symmetric for Risk-On + defensive). Original classification preserved in `classification_raw`.

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
4. Compute cross-sectional `all_ranks` (percentiles over full universe, rss_short/rss_long, plus **within-sector RSS hybrid**)
5. **TFS residualization**: cross-sectional OLS `TFS ~ a + b·TCS`, store residualized TFS for Composite (raw TFS retained for classify)
6. Run `SignalValidityEngine.compute()` (24 bi-weekly **Friday-anchored** eval points; `eval_indices` walk backward in 14-day steps from most recent Friday)
7. Score + classify each ticker (Composite uses TFS_resid + OER penalty; classify uses Sticky FLAT hysteresis), compute 1W/1M/3M/custom historical snapshots
8. Console output → 7-day history → PDF export
9. GraphRAG: build knowledge graph → Louvain community detection → insights → cache

The api.py loader then (per `_load_cache()`):
10. Loads `.scan_cache.pkl` + `.fundamentals_cache.pkl`
11. Computes QVR scores for every ticker (cross-sectional Q/V/R percentile)
12. Applies the Eligibility Gate (demotes Stock with QVR < 40 → adds `WeakQVR(<n>)` rejection tag)
13. **Macro Context Tags** (Phase 1.0/1.5): `cyclical_tag`, `style_tilt`, `region`, `industry_group` (+ Biotech→cyclical/growth, Telecom→defensive/value industry refinement)
14. **Phase 1G** inject tags into ve_observations (enables `/api/validation` segmented hit rates)
15. **Phase 2D + 3B**: regime detection → `rotation_long`/`rotation_short` scores → CYCLE_PEAK override
16. **Hybrid Phase A+B**: ETF sidecar (constituent_breadth_mom / weighted_comp / coverage / leader_gap / divergence_flag)
17. **Hybrid Phase D**: `parent_etf_signal` (Stock = weighted avg of parent ETF flags; ETF = own breadth) → injected into STATE['results']
18. **Anti-Lag Phase 1**: run `pre_momentum.run_pre_momentum(cache)` → derive `provisional_eligible` + `eligibility_tier`
19. **Sector-Segmented (New2)**: per-sector top-5 → `sector_segmented_eligible` + `eligibility_tier_v2`
20. Serves `/api/table`, `/api/pre-momentum`, `/api/new-pd/validation`, `/api/new-pd-v2/validation`, etc.

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

## Macro Context Tags & Hybrid Bottom-up (api.py post-load, 2026-05 layer)

These layers attach macro context to each ticker AFTER the core scan, without re-running price_discovery.py. All are computed at API load time and exposed via `/api/table`.

### Per-ticker tag columns

| Column | Values | Source |
|---|---|---|
| `cyclical_tag` | cyclical / defensive / broad | sector → set (+ industry refinement: Biotech → cyclical, Telecom → defensive) |
| `style_tilt` | growth / value / balanced | sector → set (+ industry refinement: Biotech/Uranium/Solar → growth, Telecom → value) |
| `region` | US / Korea / Japan / China / Europe / Other Asia / LatAm / EMEA / Canada / Global Broad | ticker suffix + theme prefix |
| `industry_group` | GICS industry group name | `gics_industry_group` from `.unified_classification.json` |
| `rotation_long` / `rotation_short` | 0-100 | regime alignment vs cross-sectional Risk/Style/Region dominance |
| `pre_momentum_score`, `pm_agreement_ratio`, `pm_conviction`, `pm_timeline` | from `run_pre_momentum()` | injected for Anti-Lag tier |
| `provisional_eligible`, `eligibility_tier` | bool / 4-tier | Anti-Lag Phase 1 |
| `sector_segmented_eligible`, `sector_rank`, `sector_pct_rank`, `eligibility_tier_v2` | bool / int / 4-tier | Sector Discovery |
| `constituent_breadth_mom`, `constituent_weighted_comp`, `constituent_coverage`, `constituent_concentration`, `constituent_leader_gap`, `divergence_flag` | ETF only | Hybrid Bottom-up Phase A+B |
| `parent_etf_signal` | 0-100 | Stock: weighted avg of parent ETF flags; ETF: own breadth |

### Cross-sectional regime detection

`_detect_market_regime()` produces `STATE["regime"]`:
- `cyclical_dom` / `defensive_dom`: Cyclical sector avg Composite vs Defensive (gap > 3)
- `growth_dom` / `value_dom`: same for style
- `top_region` / `bot_region`: highest / lowest avg Composite region
- Used by Phase 2D rotation scoring + Phase 3B CYCLE_PEAK override

### Hybrid Bottom-up data dependency

ETF holdings cache (`.etf_holdings_cache.json`) is built by `etf_holdings_pipeline.py`:
- 70 equity-sector / thematic ETFs by default (SPDR Select 11 + reabsorbed Tech sub-themes + Energy/Materials/Defense/Factor/Broad + International)
- Top 10 holdings per ETF via `yfinance.Ticker.get_funds_data().top_holdings`
- Refresh weekly: `python3 etf_holdings_pipeline.py --skip-existing`

Divergence flags (per ETF):
- **HEALTHY_TREND** — ETF Composite ≥ 60 + constituent breadth ≥ 70% (광범위 leadership ✓)
- **NARROW_RALLY** — ETF Composite ≥ 60 + breadth < 40% (소수 mega-cap 견인, concentration risk)
- **STEALTH_STRENGTH** — ETF Composite < 55 + breadth ≥ 60% (ETF lagging, breakout 예고)
- **WRAPPER_DRAG** — ETF Composite < 50 + breadth ≥ 50% (FX/leverage drag, constituents OK)
- **NEUTRAL** — none of the above

Coverage warning: ETFs with `constituent_coverage < 50%` (구성종목 중 우리 universe 포함 비율) are flagged unreliable in the UI.

### API endpoints added 2026-05

| Endpoint | Purpose |
|---|---|
| `/api/new-pd/validation` | Anti-Lag PROVISIONAL tier forward-return validation (SVE-style, proxy) |
| `/api/new-pd-v2/validation` | Sector-Segmented tier forward-return validation (BothEligible vs SectorOnly vs UniverseOnly) |

Both use ve_observations as historical proxy since real PM scores aren't stored per snapshot.

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

**Ported faithfully (CLOSE-only mode, 2026-05 Live-engine parity update):**
- TCS / TFS / RSS / URS — formulas identical to `NaiveDiscoveryDetector.score_*` (weights `W_SHORT=0.4 / W_LONG=0.6` each)
- 3×3 Classification matrix + OVEREXTENDED / FORMATION / EXHAUSTING / CYCLE_PEAK / LAGGING_CATCHUP overrides — identical
- **`composite = base − 0.10·max(0, OER−40)`** where `base = 0.30·TCS + 0.25·TFS_resid + 0.30·RSS + 0.15·URS`
- **TFS residualization** via cross-sectional OLS (sector_rotation_backtest.py:`_compute_composite_signal`)
- **OER computed from close-only** via `_compute_oer_signal()` — sma20_dist, sma50_dist, RSI(14), pct_from_high (reversal_pctile omitted)
- URS 4 components (LeadLag + AttnGap + Drift + Dispersion) — identical (single-category dispersion is degenerate by design)

**Omitted (input data difference):**
- Volume-derived signals (`vol_ratio`, `vol_ratio_3d_10d`) → defaulted to 1.0; `load_wide_close()` returns CLOSE only. TFS loses its volume term, no Composite reweighting.
- OHLC-derived sidecars (Wyckoff OBV, Ichimoku, Darvas, MFI, distribution_days) — none feed Composite, all dropped
- RSS `range_pct_pctile` term → placeholder 50.0 (no rolling 52w high/low without OHLC). RSS one of five inputs only, mild impact.
- Within-sector RSS hybrid → degenerate (11-sector universe IS the sector)
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

The primary UI lives in `frontend/`. Top-level tabs (8 total — order in `App.tsx`):
1. **Market Commentary** — comprehensive ~15k-char auto-generated report (Executive Summary + Conviction Picks + Market Leaders + ETF Hybrid Health + 23 sections)
2. **Price Discovery** — sub-tabs:
   - **Pre-Momentum** — wider watchlist (forming candidates)
   - **Momentum** — confirmed eligible (Composite ≥ 55)
   - **Anti Lag Discovery** — PROVISIONAL tier (PM strong, anti-lag) ★
   - **Sector Discovery** — Sector-Segmented top-5 per sector ★
   - **Excluded** — bearish/ineligible
3. **Price Discovery (ML)** — ML-rescored variants
4. **Validation** — backtest hit rates + segmented (cyclical/style/region)
5. **Market Environment** — macro regime, breadth
6. **Analysis** — ablation, factor efficacy
7. **AI Prediction** — ML signal engine output
8. **Appendix** — universe, descriptions, references, signal efficacy

The sidebar has Sector-based filters (17 sectors) + classification filters + composite range slider. Asset-class toggle (Equity / FICC) maps to specific Sector subsets.

### Key composite UI panels

- **Conviction Picks** (Market Commentary) — Buy/Sell top-5 stocks + ETFs with BuyScore = Composite + Classification ± OER penalty + Consensus + 1M/Sector regime − WeakQVR penalty + ETF flag bonus. WeakQVR Buy candidates marked with ⚠️ badge + amber row highlight.
- **Market Leaders** (Market Commentary) — 주도주 / 주도섹터 / 시장회전(Cyclical-Defensive + Growth-Value) / 주도테마+전략 / 주도지역 (10 region buckets).
- **ETF Hybrid Health** card — per-ETF constituent metrics + 4 divergence flags (HEALTHY_TREND / NARROW_RALLY / STEALTH_STRENGTH / WRAPPER_DRAG).
- **Rolling Return chart** (Sector Rotation tab) — 6M / 12M / 2Y / 3Y window toggle.
- **Position History by Year** table — Strategy vs SPY / QQQ / IWM with monthly α columns.

---

## Dependencies

```bash
pip install numpy pandas yfinance matplotlib networkx python-louvain fastapi uvicorn
cd frontend && npm install
```

Minimum Python 3.8. No build step for backend, no tests, no linter config.
