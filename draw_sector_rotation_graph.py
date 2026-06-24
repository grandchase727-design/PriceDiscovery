"""
draw_sector_rotation_graph.py — US Sector Rotation Dependency Graph (multi-page PDF)

Output: reports/us_sector_rotation_graph.pdf

Pages:
  1. Overall flow diagram — data → ranking → tier → decision → backtest (4 signal modes + vol overlay)
  2. Phase 1 — Within-11 ranking + tier classification + breadth
  3. Phase 2 — Macro regime overlay (5 regimes × 11 sectors fit matrix)
  4. Phase 3 — Monthly rebalance backtest (12-1M momentum signal)
  5. Phase 4 — Composite-live signal mode (TCS/TFS/RSS/URS daily-reconstructed)
  6. Phase 5 — ML B-1: multi-horizon momentum blend (walk-forward, baseline-anchored)
  7. Phase 5 — ML B-3: volatility targeting overlay (position size scaling)
  8. Phase 5 — ML B-2: macro-augmented LightGBM (8 features × walk-forward fit)
  9. Dashboard Reference 1 — Strategy diagnostic view (regime banner, tier legend, sector table)
 10. Dashboard Reference 2 — Backtest view (controls, KPIs, charts, tables)

★ Maintenance:
  Run this script after every change to sector_rotation.py or
  sector_rotation_backtest.py to keep the graph in sync.
      python3 draw_sector_rotation_graph.py
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

# ── Palette (consistent with score_dependency_graph.pdf) ──
BG = "#ffffff"
FG = "#1f2937"
MUTED = "#6b7280"
DIVIDER = "#e5e7eb"
LINE = "#9ca3af"

DATA_FILL,    DATA_EDGE    = "#f3f4f6", "#6b7280"
RANK_FILL,    RANK_EDGE    = "#eef2ff", "#6366f1"   # indigo — ranking / tier
REGIME_FILL,  REGIME_EDGE  = "#f5f3ff", "#7c3aed"   # purple — regime
DECISION_FILL,DECISION_EDGE= "#ecfeff", "#0891b2"   # cyan — decision / output
BACKTEST_FILL,BACKTEST_EDGE= "#fef2f2", "#dc2626"   # red — backtest
BREADTH_FILL, BREADTH_EDGE = "#f0fdf4", "#16a34a"   # green — breadth


# ─────────────────────────────────────────────────────────────────
# Drawing primitives
# ─────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, text, fill, edge, fontsize=9, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=FG, zorder=3,
            linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=0.8):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=8,
        color=color, linewidth=lw, alpha=0.7, zorder=1,
    ))


def setup_page(figsize=(14, 18)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def page_header(ax, page_num, title, subtitle="", accent=RANK_EDGE):
    ax.add_patch(FancyBboxPatch((0, 95.2), 100, 4,
                                 boxstyle="square,pad=0",
                                 facecolor=accent, edgecolor="none", zorder=1))
    ax.text(2, 97.2, f"PAGE {page_num}", fontsize=9, fontweight="bold",
            color="white", va="center", ha="left", zorder=2)
    ax.text(50, 97.2, title, fontsize=15, fontweight="bold",
            color="white", va="center", ha="center", zorder=2)
    if subtitle:
        ax.text(50, 93, subtitle, fontsize=10, color=MUTED,
                va="center", ha="center", style="italic")


def section_h2(ax, x, y, text, color=FG, fontsize=12):
    ax.text(x, y, text, fontsize=fontsize, fontweight="bold",
            color=color, va="center", ha="left")
    ax.plot([x, x + 25], [y - 1.2, y - 1.2], color=color, linewidth=0.8, alpha=0.4)


def info_box(ax, x, y, w, h, title, content, fill, edge,
             title_size=10, content_size=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor=fill, edgecolor=edge,
                                 linewidth=1.0, zorder=1))
    ax.text(x + 1.5, y + h - 1.5, title, fontsize=title_size,
            fontweight="bold", color=edge, va="top", ha="left")
    ax.text(x + 1.5, y + h - 4.5, content, fontsize=content_size, color=FG,
            va="top", ha="left", linespacing=1.5, wrap=True)


def text_block(ax, x, y, lines, fontsize=8.5, color=FG, linespacing=1.5,
               weight="normal"):
    text = "\n".join(lines) if isinstance(lines, list) else lines
    ax.text(x, y, text, fontsize=fontsize, color=color,
            va="top", ha="left", linespacing=linespacing, fontweight=weight)


# ═════════════════════════════════════════════════════════════════
# PAGE 1 — Overall flow diagram
# ═════════════════════════════════════════════════════════════════

def draw_page1_flow(fig, ax):
    ax.text(50, 97.5, "US Sector Rotation — Dependency Graph",
            ha="center", va="center",
            fontsize=18, fontweight="bold", color=FG)
    ax.text(50, 95, "11 SPDR ETFs  ·  Tier classification  ·  Macro regime overlay  ·  Monthly backtest",
            ha="center", va="center", fontsize=10, color=MUTED)

    # Layer labels
    layer_y = [88, 76, 60, 44, 26, 10]
    layer_lbl = ["INPUT", "BUILDING BLOCKS", "RANKING + TIER", "REGIME OVERLAY", "DECISIONS", "BACKTEST"]
    for y, lbl in zip(layer_y, layer_lbl):
        ax.text(2.5, y, lbl, fontsize=8, fontweight="bold",
                color=MUTED, va="center")

    for y in [84, 70, 53, 36, 18]:
        ax.plot([6, 96], [y, y], color=DIVIDER, linewidth=0.6, zorder=0)

    # ── INPUT ──
    box(ax, 10, 86, 80, 5,
        ".scan_cache.pkl  →  results dict (770 tickers + per-ticker fields)",
        DATA_FILL, DATA_EDGE, fontsize=10)

    # ── BUILDING BLOCKS ──
    box(ax, 8, 72, 26, 8,
        "11 SPDR ETFs\nXLK · XLC · XLV · XLF · XLY · XLP\nXLI · XLE · XLB · XLU · XLRE",
        RANK_FILL, RANK_EDGE, fontsize=8.5)
    box(ax, 37, 72, 26, 8,
        "Composite Score\n(0.30·TCS + 0.25·TFS\n+ 0.30·RSS + 0.15·URS)\n+ Classification + OER",
        RANK_FILL, RANK_EDGE, fontsize=8.5)
    box(ax, 66, 72, 26, 8,
        "Constituent Stocks\n(via SubTheme → Sector taxonomy)\nbreadth %eligible · %bullish",
        BREADTH_FILL, BREADTH_EDGE, fontsize=8.5)

    arrow(ax, 50, 86, 21, 80)
    arrow(ax, 50, 86, 50, 80)
    arrow(ax, 50, 86, 79, 80)

    # ── RANKING + TIER ──
    box(ax, 12, 56, 35, 10,
        "Within-11 RSS Ranking\n+ Tier Classification\n\n(OVERWEIGHT / NEUTRAL+ /\nCATCH-UP / NEUTRAL- /\nUNDERWEIGHT)",
        RANK_FILL, RANK_EDGE, fontsize=8.5)
    box(ax, 53, 56, 35, 10,
        "Sector Breadth\n\n(% of constituent stocks\neligible / bullish)",
        BREADTH_FILL, BREADTH_EDGE, fontsize=8.5)

    arrow(ax, 22, 72, 30, 66, color=RANK_EDGE)
    arrow(ax, 50, 72, 30, 66, color=RANK_EDGE)
    arrow(ax, 79, 72, 70, 66, color=BREADTH_EDGE)

    # ── REGIME OVERLAY (Phase 2) ──
    box(ax, 8, 39, 84, 11,
        "Macro Regime Overlay  (Phase 2)\n\n"
        "Group leadership of 4 sector groups → regime classification (5 types)\n"
        "  cyclical  ·  growth  ·  defensive  ·  commodity\n"
        "Each sector → regime_fit (0-100) for current regime + alignment tag (ALIGNED / NEUTRAL / CONTRARY)",
        REGIME_FILL, REGIME_EDGE, fontsize=9)
    arrow(ax, 30, 56, 30, 50, color=RANK_EDGE)
    arrow(ax, 70, 56, 70, 50, color=BREADTH_EDGE)

    # ── DECISIONS ──
    box(ax, 8, 21, 40, 13,
        "Decision Tags  (per sector)\n\n"
        "BUY · HOLD · CATCH-UP · WATCH\n"
        "TRIM · HEDGE · EXIT · AVOID\n\n"
        "Logic: tier × OER × classification × signal\n"
        "Sort: rank 1 (most actionable BUY) → rank 10 (EXIT)",
        DECISION_FILL, DECISION_EDGE, fontsize=8.5)
    box(ax, 52, 21, 40, 13,
        "Summary Metrics\n\n"
        "• n_overweight / n_underweight / n_catchup\n"
        "• dispersion = max(comp) - min(comp)\n"
        "• alpha_signal = HIGH / MODERATE / LOW\n"
        "• leaders / laggards (top 3 / bottom 3)\n"
        "• regime + confidence (HIGH/MEDIUM/LOW)",
        DECISION_FILL, DECISION_EDGE, fontsize=8.5)
    arrow(ax, 22, 39, 22, 34, color=DECISION_EDGE)
    arrow(ax, 79, 39, 72, 34, color=DECISION_EDGE)

    # ── BACKTEST (Phase 3 + 4 + 5) ──
    box(ax, 8, 4, 84, 13,
        "Monthly Rebalance Backtest  (Phase 3-5)\n\n"
        "Each month-end → rank 11 sectors by SELECTABLE signal → top-N (default 3) eq-weight → hold 1 month\n"
        "Signal modes:  Phase 3 momentum_12_1m  ·  Phase 4 composite_live  ·  B-1 ml_momentum_blend  ·  B-2 ml_lightgbm\n"
        "Vol overlay (B-3): optional position scaling to target annual vol (8/10/12/15/20% × max-lev 1.0-2.0x)\n"
        "Benchmarks: EW-11 ★, SPY (legacy α), QQQ, IWM, ACWI  ·  Cached 24h per full param tuple",
        BACKTEST_FILL, BACKTEST_EDGE, fontsize=9)
    arrow(ax, 28, 21, 28, 17, color=BACKTEST_EDGE, lw=1.2)
    arrow(ax, 72, 21, 72, 17, color=BACKTEST_EDGE, lw=1.2)

    # Footer
    ax.text(50, 1,
            "Files: sector_rotation.py · sector_rotation_backtest.py · macro_features.py  ·  "
            "API: /api/sector-rotation + /api/sector-rotation/backtest",
            ha="center", va="center", fontsize=7, color=MUTED, style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — Phase 1 detail (ranking + tier + breadth)
# ═════════════════════════════════════════════════════════════════

def draw_page2_phase1(fig, ax):
    page_header(ax, 2, "Phase 1 — Tier Classification & Breadth",
                "Within-11 ranking + per-sector tier + constituent breadth",
                accent=RANK_EDGE)

    section_h2(ax, 4, 90, "Tier Thresholds (5-tier classification)", color=RANK_EDGE)
    info_box(ax, 4, 65, 92, 23,
        "Tier Classification Logic",
        "OVERWEIGHT     :  Composite ≥ 70  AND  classification ∈ {CONTINUATION, FORMATION, RECOVERY}\n"
        "                  (OVEREXTENDED also included if Composite ≥ 70 — but watch OER)\n"
        "NEUTRAL+       :  Composite 55-70  AND  bullish classification\n"
        "CATCH-UP       :  classification = LAGGING_CATCHUP  OR  (Composite < 55  AND  URS ≥ 70)\n"
        "                  → lagging sector with catch-up potential (high URS = market under-reacting)\n"
        "NEUTRAL-       :  Composite 40-55  (borderline / watch)\n"
        "UNDERWEIGHT    :  Composite < 40  OR  DOWNTREND/CYCLE_PEAK/FADING/WEAKENING\n"
        "                  OR  (OVEREXTENDED + OER ≥ 75 — severely overextended)",
        RANK_FILL, RANK_EDGE,
        title_size=11, content_size=9)

    section_h2(ax, 4, 60, "Decision Mapping", color=DECISION_EDGE)
    info_box(ax, 4, 30, 45, 28,
        "Per-Sector Action (long-only — no shorting)",
        "Tier OVERWEIGHT     →  BUY (rank 2)\n"
        "Tier NEUTRAL+       →  HOLD (rank 5)\n"
        "Tier CATCH-UP       →  CATCH-UP (rank 4)\n"
        "Tier NEUTRAL-       →  WATCH (rank 6)\n"
        "Tier UNDERWEIGHT    →  AVOID (rank 7)\n\n"
        "Overrides (precedence):\n"
        "  classification ∈ {DOWNTREND, CYCLE_PEAK}\n"
        "                                         →  EXIT (rank 10)\n"
        "  OVEREXTENDED + OER ≥ 75                →  HEDGE (rank 9)\n"
        "  OVEREXTENDED + OER ≥ 60                →  TRIM  (rank 8)\n"
        "  FADING / WEAKENING                     →  AVOID (rank 8)\n\n"
        "Decision logic is self-contained to the\n"
        "4-axis scoring model (Composite + class +\n"
        "OER) — does NOT use hedge-strategy signals.",
        DECISION_FILL, DECISION_EDGE,
        title_size=11, content_size=8)

    info_box(ax, 51, 30, 45, 28,
        "Sector Breadth (constituent stocks)",
        "For each of the 11 sectors, aggregate\n"
        "constituent stocks:\n"
        "  pct_eligible   :  % stocks passing Eligibility Gate\n"
        "  pct_bullish    :  % in bullish classification\n"
        "  avg_composite  :  mean composite of constituents\n"
        "  n_constituents :  total stocks mapped to sector\n\n"
        "Mapping path:\n"
        "  Stock  →  STOCK_THEMES_CONSOLIDATED  →  SubTheme\n"
        "  SubTheme  →  SUBTHEME_TO_SECTOR (api.py)  →  Sector\n\n"
        "Use case: ETF-level signal vs stock-level\n"
        "breadth divergence — if ETF strong but stock\n"
        "breadth weak (XLY=55 / 9% breadth), rotation\n"
        "conviction is lower (only mega-caps driving).",
        BREADTH_FILL, BREADTH_EDGE,
        title_size=11, content_size=8)


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — Phase 2 detail (regime overlay)
# ═════════════════════════════════════════════════════════════════

def draw_page3_phase2(fig, ax):
    page_header(ax, 3, "Phase 2 — Macro Regime Overlay",
                "5 regimes × 11 sectors fit matrix · top-down rotation context",
                accent=REGIME_EDGE)

    section_h2(ax, 4, 90, "Sector Group Classification", color=REGIME_EDGE)
    info_box(ax, 4, 75, 92, 13,
        "4 sector groups (used for regime detection)",
        "  cyclical    :  XLY · XLB · XLI · XLF       (aggressive cyclicals; early-cycle leaders)\n"
        "  growth      :  XLK · XLC                    (Tech / Comm Services; mid-expansion leaders)\n"
        "  defensive   :  XLP · XLU · XLV · XLRE       (Staples / Utilities / Healthcare / RealEstate; risk-off leaders)\n"
        "  commodity   :  XLE                          (Energy; late-cycle / inflation hedge)",
        REGIME_FILL, REGIME_EDGE,
        title_size=11, content_size=9)

    section_h2(ax, 4, 70, "Regime Detection Algorithm", color=REGIME_EDGE)
    info_box(ax, 4, 47, 92, 21,
        "Detection rules (priority order)",
        "1. Compute average Composite per group (e.g. cyclical_avg = mean(XLY, XLB, XLI, XLF composites))\n"
        "2. Sort groups by avg descending → leading_group, second_group\n"
        "3. gap = leading_avg − second_avg\n\n"
        "Classification rules:\n"
        "   if cyclical leads AND cyclical > defensive + 8           →  RISK_ON_EARLY_CYCLE\n"
        "   elif growth leads AND growth > defensive + 5              →  TECH_GROWTH_LED\n"
        "   elif commodity leads AND commodity > 55 AND cyclical > def → LATE_CYCLE\n"
        "   elif defensive leads AND defensive > cyclical + 5         →  DEFENSIVE_RISK_OFF\n"
        "   elif gap < 5                                              →  MIXED_TRANSITIONAL\n"
        "   else (fallback): map by leading group's natural regime\n\n"
        "Confidence:\n"
        "   gap ≥ 15 AND leading_avg ≥ 55  →  HIGH (50 + gap × 2 %)\n"
        "   gap ≥ 8                          →  MEDIUM (40 + gap × 2 %)\n"
        "   else                              →  LOW (gap × 4 %)",
        REGIME_FILL, REGIME_EDGE,
        title_size=11, content_size=8)

    section_h2(ax, 4, 42, "Regime → Per-Sector Fit Matrix (sample)", color=REGIME_EDGE)
    info_box(ax, 4, 4, 92, 36,
        "REGIME_FIT_MATRIX  (sector_rotation.py:60+)",
        "Each cell is 0-100 fit score. Higher = sector expected to outperform in that regime.\n\n"
        "                    XLK   XLC   XLY   XLB   XLI   XLF   XLE   XLP   XLU   XLV   XLRE\n"
        "RISK_ON_EARLY        60    55    90    88    88    80    45    30    25    35    40\n"
        "TECH_GROWTH_LED      95    90    70    45    55    60    35    35    30    55    45\n"
        "LATE_CYCLE           40    40    35    80    65    60    90    55    60    65    40\n"
        "DEFENSIVE_RISK_OFF   35    35    20    25    30    30    45    90    90    85    70\n"
        "MIXED_TRANSITIONAL   55    55    50    50    50    55    50    55    50    60    50\n\n"
        "Per-ticker output:\n"
        "  regime_fit            :  matrix lookup for current regime\n"
        "  composite_x_regime    :  (composite + regime_fit) / 2  — combined score\n"
        "  regime_alignment      :  ALIGNED (≥70)  ·  NEUTRAL (50-70)  ·  CONTRARY (<50)\n\n"
        "Practical use:\n"
        "  • ALIGNED + high Composite → strongest conviction (regime + price agree)\n"
        "  • CONTRARY + high Composite → potential rotation OUT (regime changing)\n"
        "  • ALIGNED + low Composite  → potential rotation IN (early signal)\n"
        "  • CONTRARY + low Composite → ignore (consistent with regime)",
        REGIME_FILL, REGIME_EDGE,
        title_size=11, content_size=8)


# ═════════════════════════════════════════════════════════════════
# PAGE 4 — Phase 3 detail (backtest methodology + metrics)
# ═════════════════════════════════════════════════════════════════

def draw_page4_phase3(fig, ax):
    page_header(ax, 4, "Phase 3 — Monthly Rebalance Backtest",
                "Walk-forward validation · selection alpha vs EW-11 + drawdown reduction vs SPY",
                accent=BACKTEST_EDGE)

    section_h2(ax, 4, 90, "Backtest Methodology", color=BACKTEST_EDGE)
    info_box(ax, 4, 65, 92, 23,
        "Strategy specification",
        "Universe       :  11 SPDR sector ETFs (XLK, XLC, XLV, XLF, XLY, XLP, XLI, XLE, XLB, XLU, XLRE)\n"
        "Benchmarks     :  EW-11 (★ true selection-α baseline) · SPY · QQQ · IWM · ACWI\n"
        "                  EW-11   = equal-weight 11 sectors (isolates SELECTION α)\n"
        "                  SPY     = cap-weighted S&P 500 (broad US, cap-weight bias)\n"
        "                  QQQ     = Nasdaq 100 (large-cap growth/tech tilt)\n"
        "                  IWM     = Russell 2000 (US small-cap)\n"
        "                  ACWI    = MSCI All Country World (global broad)\n"
        "Lookback       :  configurable 3 / 5 / 7 / 10 years\n"
        "Signal         :  selectable per request — Phase 3 12-1M momentum  /  Phase 4 Composite-live\n"
        "Rebalance      :  month-end · top-N sectors (default 3) · equal-weight · 1-month hold\n"
        "Turnover cost  :  configurable bp (default 30) · applied per position change\n"
        "Missing data   :  restrict universe to ETFs with valid forward return AND signal\n"
        "Data source    :  yfinance, monthly auto-adjusted Close prices",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=11, content_size=8)

    section_h2(ax, 4, 60, "Computed Metrics", color=BACKTEST_EDGE)
    info_box(ax, 4, 36, 45, 22,
        "Per-track metrics (× 6 tracks: strategy + 5 benchmarks)",
        "total_return       :  ∏(1 + r_t) − 1\n"
        "cagr               :  (1 + total_return)^(1/years) − 1\n"
        "sharpe             :  (avg × 12 − rf) / (std × √12)\n"
        "                      (rf = 0 default)\n"
        "mdd                :  min(cum / running_max − 1)\n"
        "avg_monthly_ret    :  mean of monthly returns\n"
        "vol_monthly        :  monthly std deviation",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)
    info_box(ax, 51, 36, 45, 22,
        "Comparison metrics (per benchmark)",
        "alpha_total / annualized              :  vs SPY (legacy)\n"
        "alpha_vs_ew_total / annualized      ★ :  vs EW-11 (selection α)\n"
        "alpha_vs_qqq_total / annualized       :  vs QQQ\n"
        "alpha_vs_iwm_total / annualized       :  vs IWM\n"
        "alpha_vs_acwi_total / annualized      :  vs ACWI\n"
        "win_rate_pct                          :  % months > SPY (legacy)\n"
        "win_rate_vs_ew/qqq/iwm/acwi_pct       :  per-benchmark win rate\n"
        "turnover_avg_per_rebalance            :  avg position changes\n"
        "yearly                              ★ :  per-year ret/α/win across all\n\n"
        "Cache: 24h TTL per (lookback, top_n, turnover_bp, signal_mode)",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 31, "Live Result (5y window · top-3 · 30 bp · 12-1M signal)", color=BACKTEST_EDGE)
    info_box(ax, 4, 11, 92, 19,
        "Six-track comparison",
        "                              STRAT      EW-11     SPY      QQQ      IWM      ACWI\n"
        "  CAGR                       ~17.1%    ~15.3%   ~17.0%   ~19.1%   ~13.3%   ~14.8%\n"
        "  Sharpe                      ~1.07     ~1.03    ~1.07    ~0.97    ~0.69    ~0.99\n"
        "  Max Drawdown               ~−12%     ~−18%    ~−24%    ~−33%    ~−27%    ~−25%   ★ best DD\n"
        "  α vs benchmark (CAGR)         —      +1.79%  +0.12%  −1.98%  +3.87%  +2.29%\n"
        "  win rate                      —       52%     41%     45%     55%     46%\n\n"
        "Reading: best vs IWM (+3.87%, small-cap weak), beats EW/ACWI/SPY,\n"
        "underperforms QQQ (−1.98%) — tech mega-cap concentration in QQQ was hard to match.\n"
        "Strategy excels in DEFENSIVE-ROTATION bear cycles (2022 +28.7% α vs EW); lags narrow tech bulls.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    # Footer note
    ax.text(50, 5,
            "Maintenance:  re-render this PDF after edits to sector_rotation.py / "
            "sector_rotation_backtest.py  ·  python3 draw_sector_rotation_graph.py",
            ha="center", va="center", fontsize=8, color=MUTED, style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 5 — Phase 4 detail (composite_live signal mode)
# ═════════════════════════════════════════════════════════════════

def draw_page5_phase4(fig, ax):
    page_header(ax, 5, "Phase 4 — Composite-Live Signal Mode",
                "Daily-reconstructed TCS/TFS/RSS/URS · same weights as live engine",
                accent=BACKTEST_EDGE)

    section_h2(ax, 4, 90, "Why this exists", color=BACKTEST_EDGE)
    info_box(ax, 4, 73, 92, 15,
        "Motivation",
        "Phase 3 used 12-1M momentum as a single-axis proxy because historical Composite is\n"
        "not persisted per-month. Phase 4 reconstructs a Composite-equivalent at each historical\n"
        "month-end directly from daily prices, applying the same logic as the live engine\n"
        "(UPDATED 2026-05: TFS residualization + OER penalty added — Live-engine parity):\n\n"
        "      base       =  0.30·TCS  +  0.25·TFS_resid  +  0.30·RSS  +  0.15·URS\n"
        "      composite  =  base  −  0.10·max(0, OER−40)\n\n"
        "Both modes are exposed via /api/sector-rotation/backtest?signal_mode=...",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=11, content_size=8)

    section_h2(ax, 4, 68, "Per-axis reconstruction (simplified vs live)", color=BACKTEST_EDGE)
    info_box(ax, 4, 35, 92, 31,
        "Components and their data-driven simplifications",
        "TCS  (0.30)  Trend Continuation\n"
        "             • SMA 20 / 50 / 200 distance (above = +, below = − ; weights 8 / 8 / 12)\n"
        "             • SMA 21-day slope (positive = + ; weights 7 / 7 / 8)\n\n"
        "TFS_resid  (0.25)  Trend Formation, residualized vs TCS  [NEW 2026-05]\n"
        "             • 60d range position (low → high mapped 0 → 30) + 5d-vs-21d acceleration\n"
        "             • Cross-sectional OLS: TFS_resid = clip(50 + (TFS − (a + b·TCS)), 0, 100)\n"
        "             • Removes TCS-TFS information overlap (Live-engine parity)\n\n"
        "RSS  (0.30)  Relative Strength\n"
        "             • Cross-sectional percentile of 5d / 21d / 63d / 252d returns (.10/.20/.30/.40)\n"
        "             • Within-sector RSS hybrid: degenerate (11-sector universe IS the sector)\n\n"
        "URS  (0.15)  Underreaction — held NEUTRAL (50)\n"
        "             • LeadLag/AttnGap/Drift/Dispersion need richer data than daily Close\n\n"
        "OER  (penalty)  Overextension Risk  [NEW 2026-05]\n"
        "             • sma20_dist (>8: +15, >5: +8), sma50_dist (>15: +35, >10: +25, >5: +12)\n"
        "             • RSI(14) (>80: +25, >70: +12), pct_from_high (>−2: +15)\n"
        "             • Composite penalty: −0.10·max(0, OER−40) (max −6 points)",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=11, content_size=7.5)

    section_h2(ax, 4, 30, "Empirical comparison (5y window, top-3, 30 bp)", color=BACKTEST_EDGE)
    info_box(ax, 4, 10, 92, 19,
        "Composite-live performance — post 2026-05 update (TFS_resid + OER penalty)",
        "                                   Strategy (composite_live)    EW-11 baseline    SPY benchmark\n"
        "  CAGR                              ~12.4%                       ~15.5%             ~17.6%\n"
        "  Sharpe                             ~0.81                        ~1.05              ~1.10\n"
        "  Max Drawdown                      ~−17.7%                      ~−18.3%            ~−24.0%\n"
        "  Total Return (5y)                 ~100%                        ~135%              ~160%\n\n"
        "Insight:  OER penalty introduces defensive bias — overheated sectors automatically demoted.\n"
        "Strategy MDD now LOWER than SPY (−17.7% vs −24.0%), reflecting risk-aware sector selection.\n"
        "→ Live-engine parity achieved; backtest now accurately mirrors current production signals.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    ax.text(50, 5,
            "Maintenance:  re-render this PDF after edits to sector_rotation.py / "
            "sector_rotation_backtest.py  ·  python3 draw_sector_rotation_graph.py",
            ha="center", va="center", fontsize=8, color=MUTED, style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 6 — Phase 5 (ML signal: multi-horizon momentum blend, B-1)
# ═════════════════════════════════════════════════════════════════

def draw_page6_phase5_ml(fig, ax):
    page_header(ax, 6, "Phase 5 (B-1) — ML Signal: Multi-Horizon Momentum Blend",
                "Walk-forward fit · top-3 vs bottom-3 spread maximization · L2 toward 12-1M-favored prior",
                accent=BACKTEST_EDGE)

    section_h2(ax, 4, 90, "Why this exists", color=BACKTEST_EDGE)
    info_box(ax, 4, 76, 92, 12,
        "Motivation",
        "Phases 3 & 4 use FIXED single-axis signals. Different momentum horizons capture different\n"
        "regime dynamics (1M = short reversal, 3M/6M = trend formation, 12-1M = trend continuation).\n"
        "ML signal blends all four and learns the optimal combination from history with walk-forward CV.\n"
        "Goal: regime-adaptive signal that adjusts when one horizon stops working.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=11, content_size=8)

    section_h2(ax, 4, 71, "Per-month construction", color=BACKTEST_EDGE)
    info_box(ax, 4, 47, 92, 23,
        "Walk-forward fit at each rebalance date t",
        "1. Build 4 cross-sectionally percentile-ranked momentum features at t (per sector, in [0,1]):\n"
        "      m1_skip1   = pct_rank(P[t-1] / P[t-2] − 1)         — 1M short reversal signal\n"
        "      m3_skip1   = pct_rank(P[t-1] / P[t-4] − 1)         — 3M momentum\n"
        "      m6_skip1   = pct_rank(P[t-1] / P[t-7] − 1)         — 6M momentum\n"
        "      m12_1      = pct_rank(P[t-1] / P[t-12] − 1)        — Jegadeesh-Titman 12-1M\n"
        "                                                                  (= momentum_12_1m baseline)\n"
        "2. Training window: trailing 36 months (min 18)\n"
        "3. Build features at tau + forward return at tau+1 for each historical tau\n"
        "4. Optimize w (Σw=1, w∈[0,1]) via SLSQP to MAXIMIZE:\n"
        "      obj(w) = AVG_t [ mean(top-3 fwd_ret) − mean(bottom-3 fwd_ret) ]\n"
        "               − 0.10 × Σ (w − prior)²                    (L2 toward baseline)\n"
        "5. ★ Safeguard: if optimizer's solution train-score < prior train-score → fall back to prior\n"
        "   (guarantees ML never UNDERPERFORMS baseline within training)\n"
        "6. Apply final w to current month features → ranked signal → top-3 selection\n\n"
        "Prior: [m1=0.0, m3=0.0, m6=0.0, m12_1=1.0]  (PURE 12-1M = momentum_12_1m baseline)",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=11, content_size=8)

    section_h2(ax, 4, 42, "Empirical comparison (top-3 · 30 bp · same data)", color=BACKTEST_EDGE)
    info_box(ax, 4, 20, 92, 20,
        "Phase 3 vs Phase 4 vs Phase 5 (B-1) — across lookback windows (C-1 expansion)",
        "                            momentum_12_1m   composite_live   ml_momentum_blend  ML active%\n"
        " 5y   (N=71):   CAGR           17.13%           10.42%           17.13% ✓        0/71\n"
        "                α vs EW-11     +1.79%           −4.92%           +1.79% ✓\n"
        "10y   (N=131):  CAGR           12.91%           10.42%           12.91% ✓        0/131\n"
        "                α vs EW-11     +0.99%           −4.92%           +0.99% ✓\n"
        "20y   (N=251):  CAGR           10.11%             —              10.11% ✓        0/251\n"
        "                α vs EW-11     −0.17%             —              −0.17% ✓        (incl. GFC)\n"
        "max   (N=329):  CAGR            8.50%             —               8.88% (slightly different selection)\n"
        "(1998-2026)     α vs EW-11     −0.24%             —              +0.03%          0/329\n\n"
        "→ Across ALL window lengths up to 28y (~330 monthly obs × 11 sectors), ML fits\n"
        "  fall back to prior 100% of the time. Sample size IS the constraint — but baseline\n"
        "  is also already well-tuned. Long-term realistic CAGR ≈ 8-10% with MDD ≈ −41% (incl. 2008).",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 17, "Design rationale & next iterations", color=BACKTEST_EDGE)
    info_box(ax, 4, 4, 92, 11,
        "Findings from C-1 sample expansion (1998-2026, ~28y)",
        "• Sample size 4-5x ↑ (130 → 330 obs), but ML still doesn't beat prior (12-1M)\n"
        "• Long-term backtest reveals true MDD (−41% in 2008 GFC) vs short-window (−18%)\n"
        "• Long-term α vs SPY +0.22% (max) vs −1.12% (10y) — DD compounding helps over time\n\n"
        "Implications for further ML:\n"
        "  → Pure momentum signal alone is unlikely to beat 12-1M even with more data\n"
        "  → Lift will come from NEW INFORMATION: macro features (B-2), regime conditioning (B-4),\n"
        "    or volatility targeting (B-3, doesn't change selection but improves Sharpe)",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    ax.text(50, 1.5,
            "Maintenance:  re-render this PDF after edits to sector_rotation.py / "
            "sector_rotation_backtest.py  ·  python3 draw_sector_rotation_graph.py",
            ha="center", va="center", fontsize=8, color=MUTED, style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 7 — Phase 5 (B-3 Vol-targeting overlay)
# ═════════════════════════════════════════════════════════════════

def draw_page7_b3_voltarget(fig, ax):
    page_header(ax, 7, "Phase 5 (B-3) — Volatility-Targeted Position Sizing",
                "Scale strategy exposure to maintain target annual vol · improves Sharpe in vol-bursty regimes",
                accent=BACKTEST_EDGE)

    section_h2(ax, 4, 90, "Mechanism", color=BACKTEST_EDGE)
    info_box(ax, 4, 70, 92, 18,
        "Each month-end, scale position size based on trailing realized vol",
        "1. Compute realized_vol from last 6 months of strategy net returns:\n"
        "      realized_vol = std(monthly_returns_last_6m) × √12 × 100   (annualized %)\n\n"
        "2. Compute scale = clip(target_vol / realized_vol, 0, max_leverage)\n"
        "      target_vol = user-set target (8/10/12/15/20% annualized)\n"
        "      max_leverage = user-set cap (1.0 = no leverage; up to 2.0)\n\n"
        "3. Apply scale to monthly portfolio:\n"
        "      Equity portion: scale × (top-N equal-weight positions)\n"
        "      Cash portion:   (1 − scale)  earns 0%\n"
        "      net_return = scale × gross_strategy_return − turnover_cost\n\n"
        "4. Turnover includes implicit cash transitions (Σ|Δw_position| + |Δcash|)\n"
        "   so vol-target adjustments incur cost like any rebalance.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=11, content_size=8)

    section_h2(ax, 4, 65, "Empirical comparison (10y · momentum_12_1m · top-3 · 30 bp)", color=BACKTEST_EDGE)
    info_box(ax, 4, 41, 92, 22,
        "Vol-target sweep on baseline 12-1M signal",
        "                            CAGR     Sharpe   MDD       α_EW       avg turnover\n"
        " no vol-target              12.91%    0.89    −18.8%    +0.99%      0.41\n"
        " target  8% / max_lev 1.0    8.13%    0.76    −14.4%    −3.79%      0.47\n"
        " target 10% / max_lev 1.0    8.95%    0.78    −15.7%    −2.97%      0.48\n"
        " target 12% / max_lev 1.0    9.77%    0.80    −17.0%    −2.15%      0.47\n"
        " target 15% / max_lev 1.0   10.66%    0.81    −18.8%    −1.26%      0.45\n\n"
        " target 12% / max_lev 1.5   10.71%    0.80    −18.6%    −1.21%\n"
        " target 15% / max_lev 1.5   11.65%    0.81    −19.8%    −0.27%\n\n"
        "Reading: in this 10y window strategy already had Sharpe 0.89, so reducing equity\n"
        "exposure (low target) hurts CAGR without large Sharpe lift. Vol targeting shines\n"
        "when (i) strategy has high MDD vs Sharpe, (ii) macro vol regimes shift sharply.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 36, "When to use vol-targeting", color=BACKTEST_EDGE)
    info_box(ax, 4, 17, 92, 17,
        "Decision matrix",
        "USE vol-target IF:\n"
        "  • baseline Sharpe < 0.7 AND vol regimes shift sharply (e.g., 2008 GFC inclusion)\n"
        "  • portfolio is core holding requiring smooth equity curve (behavioral persistence)\n"
        "  • drawdown reduction more important than absolute return\n\n"
        "AVOID vol-target IF:\n"
        "  • baseline Sharpe > 0.9 (improvement marginal, CAGR cost real)\n"
        "  • short investment horizon (vol-target cash drag compounds)\n"
        "  • already using vol-aware signals (composite_live URS captures dispersion)\n\n"
        "Combinable with any signal_mode: orthogonal mechanism (separate from selection).",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    ax.text(50, 1.5,
            "Maintenance:  re-render this PDF after edits to sector_rotation.py / "
            "sector_rotation_backtest.py  ·  python3 draw_sector_rotation_graph.py",
            ha="center", va="center", fontsize=8, color=MUTED, style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 8 — Phase 5 (B-2 Macro-augmented LightGBM)
# ═════════════════════════════════════════════════════════════════

def draw_page8_b2_lgbm(fig, ax):
    page_header(ax, 8, "Phase 5 (B-2) — Macro-Augmented LightGBM",
                "Walk-forward gradient boosting with momentum + macro features (VIX/yield/credit/DXY)",
                accent=BACKTEST_EDGE)

    section_h2(ax, 4, 90, "Why this works (when momentum-only doesn't)", color=BACKTEST_EDGE)
    info_box(ax, 4, 78, 92, 10,
        "Information addition vs information rearrangement",
        "B-1 reblends 4 momentum horizons — same information, different combination → no lift.\n"
        "B-2 adds 7 NEW macro features capturing market regime → lift comes from new information,\n"
        "not from better blending of existing signals. Macro features dominate LightGBM importance\n"
        "(60-80% combined gain), with momentum features adding marginal lift.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=11, content_size=8)

    section_h2(ax, 4, 73, "Features (per sector × month)", color=BACKTEST_EDGE)
    info_box(ax, 4, 55, 92, 17,
        "12 input features → 1 prediction (next-month sector return)",
        "  Momentum (4):     m1_skip1, m3_skip1, m6_skip1, m12_1   (cross-sectional pct rank)\n\n"
        "  Macro (7):        vix             — CBOE VIX level\n"
        "                    vix_chg21       — 1M change in VIX\n"
        "                    yield_10y       — 10-year Treasury yield\n"
        "                    yield_curve     — 10y − 13w T-bill spread\n"
        "                    credit_proxy    — HYG return − TLT return (1M)\n"
        "                    dxy             — US dollar index level\n"
        "                    dxy_chg21       — 1M change in DXY\n\n"
        "  Categorical (1):  sector_id       — integer encoding of XLK/XLC/.../XLRE\n\n"
        "All macro features lagged 1 month at training (use month t-1 row to predict t→t+1 return).",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=11, content_size=8)

    section_h2(ax, 4, 50, "Walk-forward fit", color=BACKTEST_EDGE)
    info_box(ax, 4, 36, 92, 13,
        "At each rebalance date t",
        "1. Build training panel: tau ∈ [t − 60, t − 1], features at tau + forward return at tau+1\n"
        "2. Min 36 train periods × ~9-11 sectors = 320-660 training observations\n"
        "3. Fit LightGBMRegressor(n_estimators=80, max_depth=4, num_leaves=15, lr=0.05,\n"
        "                          reg_α=0.1, reg_λ=0.1, min_child_samples=10,\n"
        "                          feature/bagging_fraction=0.8, bagging_freq=2)\n"
        "4. Predict next-month return for each sector at t → top-N selection\n"
        "5. Fall back to 12-1M momentum if training data insufficient (early periods)",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 31, "Empirical result (10y window · top-3 · 30 bp)", color=BACKTEST_EDGE)
    info_box(ax, 4, 12, 92, 17,
        "ml_lightgbm vs other modes",
        "                              CAGR     Sharpe   MDD       α_EW       win_EW\n"
        " momentum_12_1m              12.91%    0.89    −18.8%    +0.99%       51%\n"
        " composite_live              10.42%    0.69    −17.7%    −4.92%       42%\n"
        " ml_momentum_blend (B-1)     12.91%    0.89    −18.8%    +0.99%       51% (= baseline)\n"
        " ml_lightgbm (B-2)        ★  16.04%    0.96    −24.6%    +4.12%       62%\n\n"
        "Avg feature importance (gain-based, % of total):\n"
        "  vix_chg21 16.1% · vix 14.7% · yield_curve 13.5% · credit_proxy 12.9% · dxy_chg21 12.1%\n"
        "  yield_10y 11.3% · dxy 10.4% · m12_1 2.9% · m6/m1/m3 ~1.5% each · sector_id 1.6%\n\n"
        "→ +3.13% CAGR, +0.07 Sharpe, +3.13% α vs EW-11 over 10y. Trade-off: MDD worse (−24.6%)\n"
        "  due to occasional concentrated regime bets. Pair with B-3 vol-target for smoother equity.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    ax.text(50, 1.5,
            "Maintenance:  re-render this PDF after edits to sector_rotation.py / "
            "sector_rotation_backtest.py / macro_features.py  ·  python3 draw_sector_rotation_graph.py",
            ha="center", va="center", fontsize=7, color=MUTED, style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 9 — Dashboard Reference Part 1 (Strategy diagnostic view)
# ═════════════════════════════════════════════════════════════════

def draw_page9_dashboard_strategy(fig, ax):
    page_header(ax, 9, "Dashboard Reference 1 — Strategy Diagnostic View",
                "Top of Sector Rotation tab: regime banner · master sector table · pairs trade",
                accent=DECISION_EDGE)

    section_h2(ax, 4, 90, "Layout (top → bottom in tab)", color=DECISION_EDGE)
    info_box(ax, 4, 81, 92, 7,
        "Section ordering (long-only — pairs/short-leg components removed)",
        "  ① Regime Banner   ② Tier Legend   ③ Master Sector Table   ④ Backtest section (Page 10)",
        DECISION_FILL, DECISION_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 78, "① Regime Banner", color=REGIME_EDGE)
    info_box(ax, 4, 70, 92, 6,
        "Top-of-tab macro regime indicator",
        "Shows current regime label + confidence (HIGH/MEDIUM/LOW) + leading group + group_avg gap.\n"
        "Color-coded: cyclical/growth = green/cyan, defensive = orange, commodity = yellow, mixed = gray.",
        REGIME_FILL, REGIME_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 67, "② Tier Legend", color=RANK_EDGE)
    info_box(ax, 4, 56, 92, 9,
        "5-tier classification color codes (per sector_rotation.py)",
        "  OVERWEIGHT  (green)   :  Composite ≥ 70 + bullish class — STRONG conviction\n"
        "  NEUTRAL+    (cyan)    :  Composite 55-70 + bullish — moderate hold\n"
        "  CATCH-UP    (purple)  :  LAGGING_CATCHUP class OR (Composite < 55 + URS ≥ 70)\n"
        "  NEUTRAL-    (gray)    :  Composite 40-55 — borderline / watch\n"
        "  UNDERWEIGHT (red)     :  Composite < 40 OR DOWNTREND/CYCLE_PEAK/etc.",
        RANK_FILL, RANK_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 53, "③ Master Sector Table — 16 columns × 11 rows", color=DECISION_EDGE)
    info_box(ax, 4, 28, 92, 23,
        "Column reference (sortable; SectorRotationTab.tsx)",
        "  #              row index 1-11\n"
        "  Rank           overall rank 1-10 (composite-derived)\n"
        "  Ticker         XLK / XLC / XLV / XLF / XLY / XLP / XLI / XLE / XLB / XLU / XLRE\n"
        "  Sector         human-readable sector name (Tech, Financials, Healthcare, ...)\n"
        "  Group          cyclical / growth / defensive / commodity (Phase 2 grouping)\n"
        "  Comp           Composite score 0-100  (0.30·TCS + 0.25·TFS + 0.30·RSS + 0.15·URS)\n"
        "  Fit            regime_fit 0-100 — sector's expected fit for CURRENT regime (Phase 2 matrix)\n"
        "  Comp×Fit       composite_x_regime = (composite + regime_fit) / 2 — combined score\n"
        "  Class          3×3 classification (CONTINUATION / RECOVERY / OVEREXTENDED / etc.)\n"
        "  OER            Overheating / Exhaustion Risk 0-100 (≥60 = warning, ≥75 = severe)\n"
        "  Tier           5-tier label (OVERWEIGHT / NEUTRAL+ / CATCH-UP / NEUTRAL- / UNDERWEIGHT)\n"
        "  Decision       BUY / HOLD / CATCH-UP / WATCH / TRIM / HEDGE / EXIT / AVOID\n"
        "                  (overrides: DOWNTREND→EXIT, OVEREXTENDED+OER≥75→HEDGE, ≥60→TRIM)\n"
        "  Br% Elig       % of constituent stocks passing Eligibility Gate\n"
        "  Br% Bull       % of constituent stocks in bullish classification\n"
        "  1D / 1W / 1M / 3M  forward returns at horizons (1, 5, 21, 63 trading days)\n\n"
        "Note: 8-strategy hedge engine (Wyckoff/Ichimoku/Darvas/...) is COMPUTED for sector ETFs\n"
        "but NOT used by the rotation strategy — see Decision logic in sector_rotation.py.",
        DECISION_FILL, DECISION_EDGE,
        title_size=10, content_size=8)

    ax.text(50, 1.5,
            "Maintenance:  re-render this PDF after edits to SectorRotationTab.tsx structure  ·  "
            "python3 draw_sector_rotation_graph.py",
            ha="center", va="center", fontsize=7, color=MUTED, style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 10 — Dashboard Reference Part 2 (Backtest view)
# ═════════════════════════════════════════════════════════════════

def draw_page10_dashboard_backtest(fig, ax):
    page_header(ax, 10, "Dashboard Reference 2 — Backtest Section",
                "Controls · KPI cards · cumulative chart · ML/vol viz · detailed metrics · yearly · positions",
                accent=BACKTEST_EDGE)

    section_h2(ax, 4, 90, "Layout (top → bottom in Backtest section)", color=BACKTEST_EDGE)
    info_box(ax, 4, 84, 92, 4,
        "Section ordering",
        "  ① Controls bar  →  ② KPI cards (5)  →  ③ Mode-conditional viz (vol-scale / ML weights / LGBM importance)  "
        "→  ④ Cumulative return chart (6 lines)  →  ⑤ Detailed Performance table  "
        "→  ⑥ Yearly Breakdown table  →  ⑦ Position History by Year",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=7)

    section_h2(ax, 4, 81, "① Controls (5 dropdowns + always-visible vol-target)", color=BACKTEST_EDGE)
    info_box(ax, 4, 67, 92, 12,
        "Configurable backtest parameters",
        "  Top-N sectors:    2 / 3 / 4 / 5             (number of sectors held each month)\n"
        "  Lookback:         3 / 5 / 7 / 10 / 15 / 20 / 25 / Max(1998+)   (history window)\n"
        "  Signal:           momentum_12_1m / composite_live / ml_momentum_blend / ml_lightgbm\n"
        "  Vol target:       Off / 8 / 10 / 12 / 15 / 20 % annualized\n"
        "  Max leverage:     1.0x / 1.25x / 1.5x / 2.0x  (only visible when vol-target on)\n"
        "  (Turnover cost fixed at 30 bp · monthly rebalance, month-end day)",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 64, "② KPI Cards (5 alpha-vs-benchmark headlines)", color=BACKTEST_EDGE)
    info_box(ax, 4, 53, 92, 9,
        "All α values are CAGR difference (annualized)",
        "  α vs EW-11   ★  TRUE selection α — vs equal-weight 11-sector basket (no cap-weight bias)\n"
        "  α vs SPY        — vs cap-weighted S&P 500 (broad market, Tech-tilted)\n"
        "  α vs QQQ        — vs Nasdaq 100 (large-cap growth/tech)\n"
        "  α vs IWM        — vs Russell 2000 (US small-cap)\n"
        "  α vs ACWI       — vs MSCI All Country World (global broad)",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 50, "③ Mode-conditional viz (one of three appears)", color=BACKTEST_EDGE)
    info_box(ax, 4, 39, 92, 9,
        "Auto-renders based on signal/vol-target combination",
        "  Vol-target ON           → Realized-vol + position-scale time-series chart (orange line + cyan line)\n"
        "  signal=ml_momentum_blend → Stacked bar chart of weight history (12-1M cyan / 6M amber / 3M pink / 1M gray)\n"
        "                              + train spread % overlay\n"
        "  signal=ml_lightgbm       → Average feature importance bar chart (macro orange / momentum cyan)\n"
        "                              showing which of 12 features dominate gain across all walk-forward fits",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 36, "④ Cumulative Return Chart", color=BACKTEST_EDGE)
    info_box(ax, 4, 28, 92, 6,
        "Line chart: 6 series",
        "  Strategy (cyan, thick) · EW-11 (purple, dotted) · SPY (gray, dashed) ·\n"
        "  QQQ (amber, dashed) · IWM (pink, dashed) · ACWI (green, dashed)\n"
        "  Y-axis: cumulative %   ·   X-axis: monthly date",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 25, "⑤ Detailed Performance Table  &  ⑥ Yearly Breakdown", color=BACKTEST_EDGE)
    info_box(ax, 4, 12, 45, 11,
        "⑤ Detailed Performance",
        "Rows: Total Return / CAGR / Sharpe / MDD /\n"
        "      Avg Monthly / Vol Monthly\n"
        "Cols: Strategy + each visible benchmark\n"
        "      (each cell shows value + Δ vs strategy)\n\n"
        "Strategy column highlighted cyan;\n"
        "Δ values colored green/red.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)
    info_box(ax, 51, 12, 45, 11,
        "⑥ Yearly Breakdown",
        "One row per calendar year:\n"
        "  Year · #Mo · Strategy · EW-11 · SPY ·\n"
        "  QQQ · IWM · ACWI ·\n"
        "  α vs EW · α vs SPY · α vs QQQ · α vs IWM · α vs ACWI\n\n"
        "Compounded yearly return (∏(1+r)−1).",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)

    section_h2(ax, 4, 9, "⑦ Position History by Year (★ replaces old 'last 12 months')", color=BACKTEST_EDGE)
    info_box(ax, 4, 1, 92, 7,
        "Full backtest history grouped by year — scrollable max-h 600px",
        "Per year: bold cyan summary row (year · #months · total changes · YTD% strat / YTD% SPY / YTD α)\n"
        "Below: monthly rows  Date · Positions (e.g. 'XLK,XLE,XLU') · Δ# (n_changes) · Strat % · SPY % · Alpha\n"
        "Years sorted descending (latest first). Sticky header. Use to inspect specific historical regimes.",
        BACKTEST_FILL, BACKTEST_EDGE,
        title_size=10, content_size=8)


# ═════════════════════════════════════════════════════════════════
# Main entry point
# ═════════════════════════════════════════════════════════════════

def draw_graph(out_path: str):
    pages = [
        draw_page1_flow,
        draw_page2_phase1,
        draw_page3_phase2,
        draw_page4_phase3,
        draw_page5_phase4,
        draw_page6_phase5_ml,
        draw_page7_b3_voltarget,
        draw_page8_b2_lgbm,
        draw_page9_dashboard_strategy,
        draw_page10_dashboard_backtest,
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with PdfPages(out_path) as pdf:
        for fn in pages:
            fig, ax = setup_page()
            fn(fig, ax)
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f"✓ Wrote {out_path}  ({len(pages)} pages)")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "reports", "us_sector_rotation_graph.pdf")
    draw_graph(out)
