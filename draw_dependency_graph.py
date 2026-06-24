"""
draw_dependency_graph.py — Score-System Dependency Graph (multi-page PDF)

Output: reports/score_dependency_graph.pdf

Pages:
  1. Dependency Graph diagram (Layers 1-5)
  2. Layer 1 + 2 — Data Sources & Derived Indicators
  3. Layer 3a — The 4 Composite Axes (TCS, TFS, RSS, URS)
  4. Layer 3b — The 5 Pre-Momentum Agents
  5. Layer 4 — Final Scores + 3x3 Classification + agreement_ratio
  6. Layer 5 — Eligibility Gate, Rejection Tags, Decision Logic, Tab Routing
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

# ── Palette ──
BG = "#ffffff"
FG = "#1f2937"
MUTED = "#6b7280"
DIVIDER = "#e5e7eb"
LINE = "#9ca3af"

DATA_FILL,  DATA_EDGE  = "#f3f4f6", "#6b7280"
COMP_FILL,  COMP_EDGE  = "#eef2ff", "#6366f1"
PM_FILL,    PM_EDGE    = "#fff7ed", "#f97316"
QVR_FILL,   QVR_EDGE   = "#f5f3ff", "#7c3aed"
SCORE_FILL, SCORE_EDGE = "#ecfeff", "#0891b2"
GATE_FILL,  GATE_EDGE  = "#fef2f2", "#dc2626"


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
    """Create a blank page with consistent layout."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def page_header(ax, page_num, title, subtitle="", accent=COMP_EDGE):
    """Top header bar with page number, title, subtitle."""
    # Accent bar
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
    """Section header (e.g. '## TCS — Trend Continuation Score')"""
    ax.text(x, y, text, fontsize=fontsize, fontweight="bold",
            color=color, va="center", ha="left")
    # Underline
    ax.plot([x, x + 25], [y - 1.2, y - 1.2], color=color, linewidth=0.8, alpha=0.4)


def text_block(ax, x, y, lines, fontsize=8.5, color=FG, linespacing=1.5,
               width=None, weight="normal"):
    """Multi-line text block. lines = list of strings or single string."""
    if isinstance(lines, str):
        text = lines
    else:
        text = "\n".join(lines)
    ax.text(x, y, text, fontsize=fontsize, color=color,
            va="top", ha="left", linespacing=linespacing, fontweight=weight,
            wrap=True)


def info_box(ax, x, y, w, h, title, content, fill, edge,
             title_size=10, content_size=8):
    """Boxed callout with bold title at top + content below."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor=fill, edgecolor=edge,
                                 linewidth=1.0, zorder=1))
    ax.text(x + 1.5, y + h - 1.5, title, fontsize=title_size,
            fontweight="bold", color=edge, va="top", ha="left")
    ax.text(x + 1.5, y + h - 4.5, content, fontsize=content_size, color=FG,
            va="top", ha="left", linespacing=1.5, wrap=True)


def divider_h(ax, x1, x2, y, color=DIVIDER, linewidth=0.6):
    ax.plot([x1, x2], [y, y], color=color, linewidth=linewidth, zorder=0)


# ═════════════════════════════════════════════════════════════════
# PAGE 1 — Dependency Graph (the diagram)
# ═════════════════════════════════════════════════════════════════

def draw_page1_diagram(fig, ax):
    # ── Title (top) ──
    ax.text(50, 97.5, "Score System — Dependency Graph",
            ha="center", va="center",
            fontsize=18, fontweight="bold", color=FG)
    ax.text(50, 95, "Momentum Composite  ·  Pre-Momentum Score  ·  QVR Eligibility Gate",
            ha="center", va="center",
            fontsize=10, color=MUTED)

    # Layer labels
    layers = [
        (88, "DATA"),
        (76, "INDICATORS"),
        (56, "AXES / AGENTS"),
        (39, "FINAL SCORES"),
        (19, "DOWNSTREAM"),
    ]
    for y, lbl in layers:
        ax.text(2.5, y, lbl, fontsize=8, fontweight="bold",
                color=MUTED, va="center")

    for y in [84, 65, 49, 30]:
        ax.plot([6, 96], [y, y], color=DIVIDER, linewidth=0.6, zorder=0)

    # LAYER 1
    box(ax, 10, 86, 26, 5, "yfinance — Prices",         DATA_FILL, DATA_EDGE, fontsize=10)
    box(ax, 38, 86, 26, 5, "yfinance — Fundamentals",   DATA_FILL, DATA_EDGE, fontsize=10)
    box(ax, 66, 86, 26, 5, "Finnhub — Metrics + News",  DATA_FILL, DATA_EDGE, fontsize=10)

    # LAYER 2
    box(ax, 10, 72, 26, 8,
        "Technical signals\nSMA · vol_ratio · RSI · returns · score_*m",
        COMP_FILL, COMP_EDGE, fontsize=9)
    box(ax, 38, 72, 26, 8,
        "Fundamentals (Q + V)\nmargin · ROE · PE · PEG · PB",
        QVR_FILL, QVR_EDGE, fontsize=9)
    box(ax, 66, 72, 26, 8,
        "Analyst data (R)\nrec trend · EPS surprise · revisions",
        QVR_FILL, QVR_EDGE, fontsize=9)
    for x in [23, 51, 79]:
        arrow(ax, x, 86, x, 80)

    # LAYER 3 — Composite axes (4)
    axes = [("TCS", "0.30"), ("TFS", "0.25"), ("RSS", "0.30"), ("URS", "0.15")]
    for i, (name, w) in enumerate(axes):
        x = 7 + i * 9
        box(ax, x, 53, 8, 10, f"{name}\n(w={w})",
            COMP_FILL, COMP_EDGE, fontsize=11, weight="bold")

    # LAYER 3 — PM agents (5)
    agents = [
        ("Micro",    "0.20", PM_FILL,  PM_EDGE),
        ("Macro",    "0.15", PM_FILL,  PM_EDGE),
        ("Graph",    "0.20", PM_FILL,  PM_EDGE),
        ("Catalyst", "0.20", PM_FILL,  PM_EDGE),
        ("QVR",      "0.25", QVR_FILL, QVR_EDGE),
    ]
    for i, (name, w, fill, edge) in enumerate(agents):
        x = 47 + i * 9
        box(ax, x, 53, 8, 10, f"{name}\n(w={w})",
            fill, edge, fontsize=10, weight="bold")

    for cx in [11, 20, 29, 38]:
        arrow(ax, 23, 72, cx, 63, color=COMP_EDGE)
    arrow(ax, 23, 72, 51, 63, color=LINE)
    arrow(ax, 23, 72, 78, 63, color=LINE)
    arrow(ax, 51, 72, 87, 63, color=QVR_EDGE)
    arrow(ax, 79, 72, 87, 63, color=QVR_EDGE)

    # LAYER 4
    box(ax, 6,  35, 39, 12,
        "A. MOMENTUM COMPOSITE\n\n"
        "0.30·TCS + 0.25·TFS_resid\n+ 0.30·RSS + 0.15·URS\n"
        "− 0.10·max(0, OER−40)",
        SCORE_FILL, SCORE_EDGE, fontsize=9, weight="bold")
    box(ax, 50, 35, 30, 12,
        "B. PRE-MOMENTUM SCORE\n\n"
        "0.20·Micro + 0.15·Macro\n+ 0.20·Graph + 0.20·Catalyst\n+ 0.25·QVR",
        SCORE_FILL, SCORE_EDGE, fontsize=9, weight="bold")
    box(ax, 82, 35, 14, 12,
        "agreement_ratio\n"
        "= count(agent>50)/5\n"
        "(0..1, breadth)\n\n"
        "≥0.6  STRONG\n"
        "0.4   MODERATE\n"
        "0-0.4 WEAK / NONE",
        PM_FILL, PM_EDGE, fontsize=7, weight="bold")

    for cx in [11, 20, 29, 38]:
        arrow(ax, cx, 53, 25, 47, color=COMP_EDGE)
    for i in range(5):
        cx = 51 + i * 9
        col = QVR_EDGE if i == 4 else PM_EDGE
        arrow(ax, cx, 53, 65, 47, color=col)
        arrow(ax, cx, 53, 89, 47, color=col, lw=0.5)

    # LAYER 5
    box(ax, 6, 14, 50, 15,
        "ELIGIBILITY GATE  (Momentum tab)\n\n"
        "Comp ≥ 55\n"
        "AND   classification ∈ bullish set\n"
        "AND   ADV ≥ $5M\n"
        "AND   (asset = ETF  OR  QVR ≥ 40)",
        GATE_FILL, GATE_EDGE, fontsize=10, weight="bold")

    ax.add_patch(FancyBboxPatch((58, 14), 38, 15,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#fffbeb", edgecolor=GATE_EDGE,
                                 linewidth=0.8, linestyle="--", zorder=2))
    ax.text(77, 27, "WHAT THE GATE DOES",
            fontsize=8, fontweight="bold", color=GATE_EDGE,
            ha="center", va="center")
    ax.text(60, 25,
            "•  Pass → ticker appears in Momentum tab (eligible)\n"
            "•  Fail → ticker moves to Excluded tab with rejection tag\n"
            "•  ETF: only 4-axis technical checked (no fundamentals → exempt)\n"
            "•  Stock: must pass BOTH technical (Comp) AND fundamentals (QVR)\n"
            "•  Filters 'junk momentum' — strong technicals but weak fundamentals\n"
            "   (e.g. SBUX QVR=31) auto-demoted with WeakQVR(<n>) rejection tag\n"
            "•  Composite definition unchanged (pure technical)",
            fontsize=6.5, color=FG, ha="left", va="top", linespacing=1.5)

    arrow(ax, 25, 35, 25, 29, color=SCORE_EDGE, lw=1.2)
    arrow(ax, 65, 35, 40, 29, color=PM_EDGE, lw=0.8)
    arrow(ax, 75, 56, 50, 29, color=QVR_EDGE, lw=1.6)

    # Footer
    ax.text(50, 9,
            "QVR plays dual role:  feeds Pre-Momentum Score (B)  AND  acts as Eligibility Gate filter  "
            "·  QVR is the only Pre-Mom agent fully orthogonal to Composite (correlation ≈ 0)",
            ha="center", va="center", fontsize=8, color=MUTED, style="italic")

    legend_items = [
        ("Composite path",  COMP_FILL,  COMP_EDGE),
        ("Pre-Mom path",    PM_FILL,    PM_EDGE),
        ("QVR (dual role)", QVR_FILL,   QVR_EDGE),
        ("Final scores",    SCORE_FILL, SCORE_EDGE),
        ("Downstream",      GATE_FILL,  GATE_EDGE),
    ]
    item_width = 18
    total_w = item_width * len(legend_items)
    start_x = (100 - total_w) / 2
    for i, (label, fill, edge) in enumerate(legend_items):
        lx = start_x + i * item_width
        ax.add_patch(FancyBboxPatch((lx, 2), 2.5, 2.5,
                                     boxstyle="round,pad=0.05,rounding_size=0.1",
                                     facecolor=fill, edgecolor=edge, linewidth=1.0))
        ax.text(lx + 3.2, 3.25, label, fontsize=8, color=FG, va="center")


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — Layer 1 (Data Sources) + Layer 2 (Derived Indicators)
# ═════════════════════════════════════════════════════════════════

def draw_page2_data(fig, ax):
    page_header(ax, 2, "Layer 1 + 2 — Data Sources & Indicators",
                "Where every input comes from and how raw data is processed",
                accent=DATA_EDGE)

    # ── Layer 1 (taller boxes h=20 to fit 11-line content cleanly) ──
    section_h2(ax, 4, 90, "Layer 1 — Raw Data Sources", color=DATA_EDGE)
    text_block(ax, 4, 87,
        "All system inputs come from 3 external services, refreshed daily by separate batch scripts.",
        fontsize=9, color=MUTED)

    info_box(ax, 4, 65, 29, 19,
        "yfinance — Prices",
        "Daily OHLCV for 770 tickers,\n"
        "5 years history.\n\n"
        "Computed from this:\n"
        "  • SMA20 / SMA50 / SMA200 + slopes\n"
        "  • RSI (14-day)\n"
        "  • vol_ratio_3d_10d\n"
        "  • Returns: 5d / 21d / 63d / 12-1M\n"
        "  • Trend age, sma distance",
        DATA_FILL, DATA_EDGE,
        title_size=10, content_size=8)

    info_box(ax, 35.5, 65, 29, 19,
        "yfinance — Fundamentals",
        "Per-ticker accounting + analyst data.\n\n"
        "Fields used:\n"
        "  • trailing PE, forward PE, PEG\n"
        "  • gross / operating margins\n"
        "  • ROE, debt/equity\n"
        "  • EPS estimates (4 quarters)\n"
        "  • Recommendations summary\n"
        "  • Price targets\n\n"
        "Limitation: thin coverage for non-US.",
        DATA_FILL, DATA_EDGE,
        title_size=10, content_size=8)

    info_box(ax, 67, 65, 29, 19,
        "Finnhub — Metrics + News",
        "US-listed only (free tier).\n\n"
        "Fields used:\n"
        "  • 70+ ratios in single API call\n"
        "  • Monthly recommendation history\n"
        "    (4 months → bullish_change_3m)\n"
        "  • Quarterly EPS surprises\n"
        "    (→ beat_rate, avg_surprise)\n"
        "  • Company news + timestamps\n\n"
        "Used to enrich US tickers only.",
        DATA_FILL, DATA_EDGE,
        title_size=10, content_size=8)

    divider_h(ax, 4, 96, 61)

    # ── Layer 2 (taller boxes h=22 to fit 14-line content) ──
    section_h2(ax, 4, 58, "Layer 2 — Derived Indicators", color=COMP_EDGE)
    text_block(ax, 4, 55,
        "Raw data is processed into 3 indicator groups before scoring. Each group flows to specific axes/agents.",
        fontsize=9, color=MUTED)

    info_box(ax, 4, 28, 29, 24,
        "Technical signals",
        "Fed by: yfinance Prices.\n\n"
        "Specific computations:\n"
        "  • SMA distances: (price-SMA)/SMA\n"
        "  • SMA slopes (1-month linear fit)\n"
        "  • Trend age (days above SMA50)\n"
        "  • vol_ratio_3d_10d\n"
        "  • RSI14 (Wilder smoothing)\n"
        "  • Returns at 5/21/63/126/252d\n"
        "  • 12-1M: prior 11-month return,\n"
        "    skipping most recent month\n\n"
        "Feeds: TCS, TFS, RSS, URS axes;\n"
        "        Microstructure / Catalyst.",
        COMP_FILL, COMP_EDGE,
        title_size=10, content_size=8)

    info_box(ax, 35.5, 28, 29, 24,
        "Fundamentals (Q + V)",
        "Fed by: yfinance + Finnhub.\n\n"
        "QUALITY block:\n"
        "  • gross_margin\n"
        "  • operating_margin\n"
        "  • ROE / ROA\n"
        "  • debt/equity, current ratio\n\n"
        "VALUE block:\n"
        "  • forward PE\n"
        "  • PEG ratio\n"
        "  • price/book (PB)\n\n"
        "Source priority:\n"
        "  • US: Finnhub > yfinance fallback\n"
        "  • KR / intl: yfinance only\n\n"
        "Feeds: QVR Agent (Q + V).",
        QVR_FILL, QVR_EDGE,
        title_size=10, content_size=8)

    info_box(ax, 67, 28, 29, 24,
        "Analyst data (R)",
        "Fed by: Finnhub recs + EPS.\n\n"
        "Specific signals:\n"
        "  • bullish_change_3m =\n"
        "    bull_ratio(now) - (3m ago)\n"
        "  • eps_beat_rate\n"
        "    = #beats / #quarters\n"
        "  • eps_surprise_avg (% avg)\n"
        "  • EPS revision counts\n"
        "    (up/down 30d/7d)\n"
        "  • Net price-target upside\n\n"
        "Why this is most leading:\n"
        "Analysts adjust forecasts BEFORE\n"
        "the stock price reacts.\n\n"
        "Feeds: QVR Agent (R component).",
        QVR_FILL, QVR_EDGE,
        title_size=10, content_size=8)

    # Bottom note (in dedicated box)
    ax.add_patch(FancyBboxPatch((4, 4), 92, 20,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#f8fafc", edgecolor=MUTED,
                                 linewidth=0.8, zorder=1))
    ax.text(50, 22, "DAILY UPDATE FLOW",
            fontsize=10, fontweight="bold", color=DATA_EDGE,
            ha="center", va="center")
    ax.text(7, 18,
        "  1.  price_discovery.py            — runs full scan, produces .scan_cache.pkl (technical signals)\n"
        "  2.  fundamentals_pipeline.py      — fetches yfinance fundamentals for all 770 tickers\n"
        "  3.  finnhub_fundamentals.py       — enriches cache with Finnhub data for US tickers (~620 of 770)\n"
        "  4.  api.py                        — loads cache, computes QVR scores, applies Eligibility Gate",
        fontsize=8.5, color=FG, va="top", ha="left", linespacing=1.7,
        family="monospace")


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — Composite Axes (TCS, TFS, RSS, URS)
# ═════════════════════════════════════════════════════════════════

def axis_card(ax, x, y, w, h, name, weight, color_fill, color_edge,
              question, inputs, score_meaning, example):
    """Standardized axis explanation card.
    Sized for h=33 with 4 sub-blocks of ~7 units each (5 header + 4*7 = 33).
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor=color_fill, edgecolor=color_edge,
                                 linewidth=1.2, zorder=1))
    # Header bar (top 5 units)
    ax.add_patch(FancyBboxPatch((x, y + h - 5), w, 5,
                                 boxstyle="square,pad=0",
                                 facecolor=color_edge, edgecolor="none", zorder=2))
    ax.text(x + 1.5, y + h - 2.5, name, fontsize=12, fontweight="bold",
            color="white", va="center", ha="left", zorder=3)
    ax.text(x + w - 1.5, y + h - 2.5, f"weight {weight}", fontsize=8.5,
            color="white", va="center", ha="right", zorder=3, style="italic")

    # Body — 4 sub-blocks, each ~7 units of vertical space
    block_step = 7.0
    body_y = y + h - 6   # first header below the header bar
    for label, content in [("QUESTION", question),
                            ("INPUTS", inputs),
                            ("WHAT THE SCORE MEANS", score_meaning),
                            ("EXAMPLE", example)]:
        ax.text(x + 1.5, body_y, label, fontsize=6.5, fontweight="bold",
                color=color_edge, va="top")
        ax.text(x + 1.5, body_y - 1.4, content,
                fontsize=7.5, color=FG, va="top", linespacing=1.35,
                style=("italic" if label == "EXAMPLE" else "normal"))
        body_y -= block_step


def draw_page3_axes(fig, ax):
    page_header(ax, 3, "Layer 3a — The 4 Composite Axes",
                "TCS · TFS · RSS · URS — each scored 0-100, combined into Momentum Composite",
                accent=COMP_EDGE)

    # 2x2 grid of axis cards (h=33 each)
    # Row 1: y=56-89  ·  Row 2: y=20-53  ·  Footer: y=2-18
    axis_card(ax, 4, 56, 44, 33,
        "TCS — Trend Continuation Score", "0.30",
        COMP_FILL, COMP_EDGE,
        "Is the existing uptrend established and persistent?\n"
        "(Are we already IN a confirmed trend?)",
        "• SMA20 / SMA50 / SMA200 distance from price\n"
        "• Slope of each SMA over the last month\n"
        "• Trend age (consecutive days above SMA50)\n"
        "• Long-horizon weight 60% > short-horizon 40%",
        "0-30  : trend not established / sideways\n"
        "30-55 : forming or weak trend\n"
        "55-75 : confirmed trend\n"
        "75-100: established, persistent trend",
        "NVDA above SMA50 for 60+ days, all SMAs sloping\n"
        "up, price 30% above SMA200 → TCS ~ 80-100.")

    axis_card(ax, 52, 56, 44, 33,
        "TFS — Trend Formation Score", "0.25",
        COMP_FILL, COMP_EDGE,
        "Is a NEW trend forming RIGHT NOW?\n"
        "(Are we at the start of something?)",
        "• SMA20 / SMA50 breakout strength + freshness\n"
        "• vol_ratio_3d_10d (3-day vs 10-day avg volume)\n"
        "• 20-day high proximity\n"
        "• Slope reversal (was negative, now positive)",
        "0-30  : no breakout / range-bound\n"
        "30-55 : early signs / weak breakout\n"
        "55-75 : confirmed fresh breakout\n"
        "75-100: very recent strong breakout w/ volume",
        "Ticker just crossed above SMA50 with 1.8x volume,\n"
        "price near 20-day high → TFS ~ 75-90.")

    axis_card(ax, 4, 20, 44, 33,
        "RSS — Relative Strength Score", "0.30",
        COMP_FILL, COMP_EDGE,
        "How does this ticker rank against the universe?\n"
        "(Is it outperforming peers?)",
        "• 5d / 21d / 63d / 12-1M return percentile\n"
        "  (vs all 770 tickers, cross-sectional rank)\n"
        "• sma20_slope, vol_ratio percentile\n"
        "• Long-horizon 65% > short 35%\n"
        "• 12-1M (Jegadeesh-Titman 1993): prior year\n"
        "  minus most recent month",
        "Pure cross-sectional ranking.\n"
        "0-30  : bottom quartile of universe\n"
        "55-75 : top half\n"
        "75-100: top decile",
        "Even in a bear market, the best-performing\n"
        "stocks still get RSS 90+ (purely relative).")

    axis_card(ax, 52, 20, 44, 33,
        "URS — Underreaction Score", "0.15",
        COMP_FILL, COMP_EDGE,
        "Is the market slow to digest new information?\n"
        "(Behavioral overlay on top of pure technicals.)",
        "• LeadLag: peer category avg ret - own ret_63d\n"
        "• AttnGap: vol_ratio pctile - ret_5d pctile\n"
        "• Drift: post-event price drift (PEAD proxy)\n"
        "• Dispersion: cross-sectional std of returns\n"
        "  (AQR, Hong-Stein 1999)",
        "0-30  : market has fully reacted / efficient\n"
        "55-75 : moderate underreaction\n"
        "75-100: strong attention/price gap\n"
        "        (high catch-up potential)",
        "Sector peers all up 20%, this ticker only up 5%\n"
        "with rising volume → URS ~ 70-85.")

    # Footnote — Composite formula
    ax.add_patch(FancyBboxPatch((4, 2), 92, 14,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#f8fafc", edgecolor=COMP_EDGE,
                                 linewidth=1.0, linestyle="-", zorder=1))
    ax.text(50, 13.5, "How the 4 axes combine into MOMENTUM COMPOSITE",
            fontsize=11, fontweight="bold", color=COMP_EDGE,
            ha="center", va="center")
    ax.text(50, 9.5,
            "Composite  =  0.30 · TCS  +  0.25 · TFS  +  0.30 · RSS  +  0.15 · URS",
            fontsize=11, fontweight="bold", color=FG,
            ha="center", va="center", family="monospace")
    ax.text(50, 5,
            "TCS + RSS get the highest weight (established + relative strength).  URS is a behavioral correction — never dominates.\n"
            "Composite ≥ 55 is the eligibility threshold (combined with classification + ADV + QVR gate on Stocks).",
            fontsize=7.5, color=MUTED, ha="center", va="center", linespacing=1.5)


# ═════════════════════════════════════════════════════════════════
# PAGE 4 — Pre-Momentum Agents (5)
# ═════════════════════════════════════════════════════════════════

def agent_card(ax, x, y, w, h, name, weight, color_fill, color_edge,
               detects, sub_signals, leading_note):
    """Standardized agent explanation card.
    Sized for h=29 with 3 sub-blocks (4 header + 3*8 + small padding ≈ 28-29).
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor=color_fill, edgecolor=color_edge,
                                 linewidth=1.2, zorder=1))
    # Header bar (top 4 units)
    ax.add_patch(FancyBboxPatch((x, y + h - 4), w, 4,
                                 boxstyle="square,pad=0",
                                 facecolor=color_edge, edgecolor="none", zorder=2))
    ax.text(x + 1.5, y + h - 2, name, fontsize=10, fontweight="bold",
            color="white", va="center", ha="left", zorder=3)
    ax.text(x + w - 1.5, y + h - 2, f"weight {weight}", fontsize=8,
            color="white", va="center", ha="right", zorder=3, style="italic")

    block_step = 8.0
    body_y = y + h - 5
    for label, content, italic in [("DETECTS", detects, False),
                                    ("SUB-SIGNALS", sub_signals, False),
                                    ("WHY IT'S LEADING", leading_note, True)]:
        ax.text(x + 1.5, body_y, label, fontsize=6.5, fontweight="bold",
                color=color_edge, va="top")
        ax.text(x + 1.5, body_y - 1.4, content,
                fontsize=7.5, color=FG, va="top", linespacing=1.35,
                style=("italic" if italic else "normal"))
        body_y -= block_step


def draw_page4_agents(fig, ax):
    page_header(ax, 4, "Layer 3b — The 5 Pre-Momentum Agents",
                "Each scored 0-100, combined into Pre-Momentum Score (forward-looking signal)",
                accent=PM_EDGE)

    # 5 agents — 3+2 grid with h=29 each
    # Row 1: y=60-89  ·  Row 2: y=29-58  ·  Footer: y=2-26
    agent_card(ax, 4, 60, 30, 29,
        "1. Microstructure", "0.20", PM_FILL, PM_EDGE,
        "Pre-breakout structural patterns —\n"
        "low volatility setup, accumulation,\n"
        "RSI compression.",
        "• vol_compression\n"
        "• accumulation_pattern\n"
        "  (TFS building while TCS still low)\n"
        "• structural_divergence\n"
        "  (structural_q > composite)\n"
        "• volume_regime\n"
        "• range_contraction (RSI ≈ 50)",
        "Low vol + tight range often precedes\n"
        "breakouts. Inputs use TCS/TFS in\n"
        "INVERSE pattern (low TCS + building\n"
        "TFS = pre-momentum).")

    agent_card(ax, 36, 60, 30, 29,
        "2. Macro Regime", "0.15", PM_FILL, PM_EDGE,
        "Top-down regime + rotation +\n"
        "Risk/Style/Region + Bottom-up ETF.",
        "• sector_rotation (0.20)\n"
        "• cross_asset (0.10)\n"
        "• category_breadth (0.20)\n"
        "• relative_improvement (0.15)\n"
        "• rotation_alignment (0.20)\n"
        "• etf_parent_signal (0.15) NEW\n"
        "   = bottom-up ETF flag aggregation",
        "etf_parent_signal (Hybrid Phase D):\n"
        "Stock = weighted avg of parent ETF\n"
        "divergence flags (STEALTH/HEALTHY/\n"
        "WRAPPER/NARROW). ETF = own breadth.\n"
        "Forward-looking bottom-up signal.")

    agent_card(ax, 68, 60, 28, 29,
        "3. Graph Relational", "0.20", PM_FILL, PM_EDGE,
        "Theme leadership signals from the\n"
        "GraphRAG knowledge graph.",
        "• peer_lead\n"
        "  (% of theme peers above 55)\n"
        "• theme_breadth\n"
        "• leader_lagger_gap\n"
        "  (max peer composite - own)\n"
        "• community_momentum",
        "Uses graph engine's community\n"
        "structure. If theme leaders are running\n"
        "but this ticker is lagging, it's a\n"
        "catch-up candidate.")

    agent_card(ax, 4, 29, 44, 29,
        "4. Catalyst", "0.20", PM_FILL, PM_EDGE,
        "Quantitative proxies for catalyst events\n"
        "(no direct news / options data on free tier).",
        "• momentum_acceleration:\n"
        "  rss_short - rss_long  (short-term RS picking up)\n"
        "• strategy_agreement:\n"
        "  long_count / total  (8 hedge strategies' consensus)\n"
        "• score_trajectory: score_1w > score_1m > score_3m\n"
        "• reversal_risk_check: reversal_pctile (safety)",
        "Mostly DELTA/CHANGE measures rather than levels — captures\n"
        "acceleration. Some overlap with Composite (uses score_*m and\n"
        "RSS) is by design: it's the 'building momentum' signal.")

    agent_card(ax, 52, 29, 44, 29,
        "5. QVR  (Quality + Value + Revision)", "0.25", QVR_FILL, QVR_EDGE,
        "Fundamentals dimension — only Pre-Mom agent\n"
        "fully orthogonal to Composite (corr ≈ 0).",
        "• Q (30%): margin + ROE percentile\n"
        "  (Q_WEIGHTS 0.40 / 0.30 / 0.30)\n"
        "• V (20%): inverse PE/PEG/PB pctile (cheap = high)\n"
        "• R (50%): EPS revisions + bullish_change_3m +\n"
        "  eps_beat_rate + eps_surprise_avg + target upside\n"
        "  → R weighted highest (Chan-Jegadeesh 1996)",
        "Independent of price/composite — captures fundamental\n"
        "quality + analyst sentiment momentum BEFORE technicals\n"
        "confirm. Also acts as Eligibility Gate filter (QVR ≥ 40).")

    # Bottom — Pre-Mom Score formula + key insight
    ax.add_patch(FancyBboxPatch((4, 4), 92, 22,
                                 boxstyle="round,pad=0.05,rounding_size=0.1",
                                 facecolor="#f8fafc", edgecolor=PM_EDGE,
                                 linewidth=1.0, zorder=1))
    ax.text(50, 22.5, "How the 5 agents combine into PRE-MOMENTUM SCORE",
            fontsize=11, fontweight="bold", color=PM_EDGE,
            ha="center", va="center")
    ax.text(50, 17.5,
            "Pre-Mom  =  0.20·Micro + 0.15·Macro + 0.20·Graph + 0.20·Catalyst + 0.25·QVR",
            fontsize=10.5, fontweight="bold", color=FG,
            ha="center", va="center", family="monospace")
    ax.text(50, 12,
            "QVR has the highest single weight (0.25) because it's the only fully orthogonal source of forward information.\n"
            "agreement_ratio = count(agent_score > 50) / 5  (sidecar — measures BREADTH instead of magnitude).",
            fontsize=8, color=MUTED, ha="center", va="center", linespacing=1.6)
    ax.text(50, 6.5,
            "Goal: detect tickers BEFORE Composite confirms momentum — solve the 'after-the-fact' detection problem.",
            fontsize=8.5, color=PM_EDGE, ha="center", va="center", style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 5 — Final Scores + Classification + agreement_ratio
# ═════════════════════════════════════════════════════════════════

def draw_page5_scores(fig, ax):
    page_header(ax, 5, "Layer 4 — Final Scores & Classification",
                "How axes/agents become actionable scores + the 3x3 classification matrix",
                accent=SCORE_EDGE)

    # ── Left: Composite (A) ──
    info_box(ax, 4, 75, 45, 16,
        "A. MOMENTUM COMPOSITE  (0-100)",
        "base = 0.30·TCS + 0.25·TFS_resid\n"
        "     + 0.30·RSS_hybrid + 0.15·URS\n"
        "Composite = base - 0.10*max(0, OER-40)\n\n"
        "TFS_resid = TFS residualized vs TCS\n"
        "  (removes TCS-TFS info overlap; same in SVE)\n"
        "RSS_hybrid = 0.6*within_sector + 0.4*universe\n"
        "  (sector beta correction; sector n>=8)\n"
        "OER penalty: linear from 40+ (max -6 pts)\n\n"
        "Eligibility threshold: >= 55",
        SCORE_FILL, SCORE_EDGE,
        title_size=10, content_size=7.5)

    # ── Right: Pre-Mom Score (B) ──
    info_box(ax, 51, 75, 45, 16,
        "B. PRE-MOMENTUM SCORE  (0-100)",
        "Pre-Mom = 0.20*Micro + 0.15*Macro*\n"
        "        + 0.20*Graph + 0.20*Catalyst\n"
        "        + 0.25*QVR\n\n"
        "* Macro Regime Agent — 6 sub-signals:\n"
        "  sector_rotation (0.20), cross_asset (0.10),\n"
        "  category_breadth (0.20), rel_improvement (0.15),\n"
        "  rotation_alignment (0.20) [Phase 2C],\n"
        "  etf_parent_signal (0.15) [Hybrid Phase D, NEW]\n\n"
        "Purpose: 'Where momentum WILL BE.'\n"
        "Forward-looking bottom-up signal integrated.",
        SCORE_FILL, SCORE_EDGE,
        title_size=10, content_size=7.5)

    # ── agreement_ratio explanation (condensed to fit) ──
    info_box(ax, 4, 56, 92, 17,
        "agreement_ratio  (sidecar to Pre-Momentum Score, 0..1)",
        "Formula:  agreement_ratio = count(agent_score > 50) / 5\n"
        "Counts HOW MANY of the 5 agents fire simultaneously. "
        "Captures BREADTH; PM Score captures MAGNITUDE.\n\n"
        "Tier mapping:\n"
        "  ≥ 0.6   STRONG     — 3+ agents firing      → PREPARE / STRONG PREPARE\n"
        "  0.4-0.6 MODERATE   — 2 agents firing       → WATCH\n"
        "  0-0.4   WEAK       — 1 agent firing        → TRACK\n"
        "  0       NONE       — no agents above 50    → IGNORE\n\n"
        "Why both?  Case A: QVR=95, others=20 → PM=39 / agree=0.2 (narrow)\n"
        "          Case B: all 5 agents = 55 → PM=55 / agree=1.0 (broad — more reliable)",
        QVR_FILL, QVR_EDGE,
        title_size=11, content_size=8)

    # ── 3×3 Classification matrix ──
    section_h2(ax, 4, 51, "3 × 3 Classification Matrix (also derived from Composite axes)",
               color=SCORE_EDGE, fontsize=11)
    text_block(ax, 4, 48,
        "Each ticker is also classified into 1 of 9 cells (short × long term direction). "
        "Categorical descriptor used for filtering and overrides.",
        fontsize=8, color=MUTED)

    # Draw the 3x3 matrix
    matrix = [
        # row = short, col = long
        # rows top→bottom: UP, FLAT, DOWN  ; cols L→R: UP, FLAT, DOWN
        [("CONTINUATION", "#22c55e"), ("RECOVERY",      "#3b82f6"), ("COUNTER_RALLY", "#a855f7")],
        [("CONSOLIDATION","#eab308"), ("NEUTRAL",       "#fb923c"), ("FADING",        "#a16207")],
        [("PULLBACK",     "#fb923c"), ("WEAKENING",     "#dc2626"), ("DOWNTREND",     "#7f1d1d")],
    ]
    matrix_x, matrix_y = 4, 22
    cell_w, cell_h = 22, 6
    # Column headers (long direction)
    col_labels = ["Long: UP", "Long: FLAT", "Long: DOWN"]
    row_labels = ["Short: UP", "Short: FLAT", "Short: DOWN"]
    for ci, label in enumerate(col_labels):
        ax.text(matrix_x + 14 + ci * cell_w + cell_w / 2 - 14, matrix_y + 3 * cell_h + 1.2,
                label, fontsize=8, fontweight="bold", color=MUTED, ha="center")
    for ri, label in enumerate(row_labels):
        ax.text(matrix_x + 12, matrix_y + (2 - ri) * cell_h + cell_h / 2,
                label, fontsize=8, fontweight="bold", color=MUTED, ha="right", va="center")
    for ri in range(3):
        for ci in range(3):
            name, color = matrix[ri][ci]
            x = matrix_x + 14 + ci * cell_w
            y = matrix_y + (2 - ri) * cell_h
            ax.add_patch(FancyBboxPatch((x, y), cell_w - 1, cell_h - 0.5,
                                         boxstyle="round,pad=0.05,rounding_size=0.1",
                                         facecolor=color + "33", edgecolor=color,
                                         linewidth=1.0))
            ax.text(x + (cell_w - 1) / 2, y + (cell_h - 0.5) / 2, name,
                    fontsize=8, fontweight="bold", color=color, ha="center", va="center")

    # Overrides
    ax.text(4, 16, "Overrides (applied AFTER the matrix):",
            fontsize=9, fontweight="bold", color=SCORE_EDGE)
    text_block(ax, 4, 13.7,
        "  • OVEREXTENDED   = OER >= 60 on a bullish cell (warns of mean-reversion risk)\n"
        "  • FORMATION      = rapid short-term breakout from a low base (overrides RECOVERY)\n"
        "  • EXHAUSTING     = old established trend losing steam (overrides CONTINUATION when age + slope decay)\n"
        "  • CYCLE_PEAK     = ret_36_12m percentile >= 85 + 12M momentum declining (long bull cycle peak)\n"
        "  • LAGGING_CATCHUP = sector-lagging name with broad sector strength (catch-up candidate)\n"
        "  * Phase 3B (api.py): Regime-aware CYCLE_PEAK early promotion\n"
        "      Risk-Off + cyclical ticker + OVEREXTENDED + OER>=70 -> CYCLE_PEAK\n"
        "      Risk-On + defensive ticker + OVEREXTENDED + OER>=70 -> CYCLE_PEAK (anomaly)\n"
        "  * Sticky FLAT hysteresis: prev=FLAT -> entry threshold * 1.3 (stricter)\n"
        "      Removes bi-weekly NEUTRAL <-> CONSOLIDATION <-> RECOVERY <-> FADING flip noise",
        fontsize=8, color=FG)

    # Bullish set used by Eligibility Gate
    ax.text(50, 2.5,
            "Bullish set (used by Eligibility Gate):  CONTINUATION · FORMATION · RECOVERY · OVEREXTENDED · LAGGING_CATCHUP",
            fontsize=8, color=MUTED, ha="center", style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 6 — Eligibility Gate + Decision Logic + Tab Routing
# ═════════════════════════════════════════════════════════════════

def draw_page6_gate(fig, ax):
    page_header(ax, 6, "Layer 5 — Eligibility Gate, Decisions & Tab Routing",
                "How scores + classification become actionable filtering and recommendations",
                accent=GATE_EDGE)

    # ── Eligibility Gate detail ──
    section_h2(ax, 4, 90, "Eligibility Gate — 4 conditions (ALL must pass)",
               color=GATE_EDGE, fontsize=12)

    # Box with conditions
    info_box(ax, 4, 70, 92, 17,
        "Conditions",
        "1.  Composite ≥ 55                                               (technical strength threshold)\n"
        "2.  Classification ∈ bullish set                                  (CONTINUATION/FORMATION/RECOVERY/OVEREXTENDED/LAGGING_CATCHUP)\n"
        "3.  ADV ≥ $5M                                                     (liquidity floor — avg daily volume in USD)\n"
        "4.  Asset-class branch:                                           ETF: skip (no fundamentals available — exempt)\n"
        "                                                                  Stock: QVR ≥ 40 (fundamentals sanity check)\n\n"
        "Pass → Momentum tab (eligible)\n"
        "Fail → Excluded tab (with rejection tag explaining why)",
        GATE_FILL, GATE_EDGE,
        title_size=10, content_size=8.5)

    # ── Rejection tag catalog ──
    section_h2(ax, 4, 65, "Rejection Tag Catalog (shown in Excluded tab)",
               color=GATE_EDGE, fontsize=11)

    info_box(ax, 4, 47, 92, 16,
        "Tag types (multiple can combine, e.g. 'CONTINUATION/WeakQVR(31)')",
        "  LowScore           — Composite < 55 (technical strength insufficient)\n"
        "  Liq($X.XM)         — ADV below $5M floor; actual value shown in parens\n"
        "  Downtrend          — classification = DOWNTREND (long+short both DOWN)\n"
        "  CyclePeak          — long bull cycle showing exhaustion signs\n"
        "  Fading             — classification = FADING (short FLAT, long DOWN — losing momentum)\n"
        "  Weakening          — classification = WEAKENING (short DOWN, long FLAT — bearish forming)\n"
        "  Exhausting         — long-established trend losing slope (override of CONTINUATION)\n"
        "  WeakQVR(N)         — Stock with QVR below 40; technical may be strong but fundamentals are weak\n"
        "                       (the 'junk momentum' filter — example: SBUX classified CONTINUATION but QVR=31)",
        GATE_FILL, GATE_EDGE,
        title_size=10, content_size=8)

    # ── Decision logic (Momentum vs Pre-Mom) ──
    section_h2(ax, 4, 42, "Decision Logic (Action recommendation per ticker)",
               color=GATE_EDGE, fontsize=11)

    info_box(ax, 4, 22, 45, 18,
        "Momentum tab — decideAction()",
        "Inputs: composite, classification, OER, age, signal\n\n"
        "Output tags:\n"
        "  BUY        composite ≥ 65 + STRONG_LONG signal\n"
        "  HOLD       composite ≥ 55, signal LONG/NEUTRAL\n"
        "  TRIM       OER ≥ 60 (overextended)\n"
        "  HEDGE      OER ≥ 70 + classification OVEREXTENDED\n"
        "  EXIT       classification DOWNTREND or CYCLE_PEAK\n"
        "  WATCH      composite 55-65, signal NEUTRAL\n\n"
        "Sorting: rank 1 (most actionable BUY)\n"
        "        rank 10 (EXIT / HEDGE)",
        SCORE_FILL, SCORE_EDGE,
        title_size=9.5, content_size=7.8)

    info_box(ax, 51, 22, 45, 18,
        "Pre-Momentum tab — decidePMAction()",
        "Inputs: pre_momentum_score, agreement_ratio, age, classification\n\n"
        "Output tags:\n"
        "  STRONG PREPARE  agreement ≥ 0.6 + PM ≥ 75 + age ≥ 14d\n"
        "  PREPARE         agreement ≥ 0.6 + PM ≥ 65\n"
        "  WATCH (Active)  agreement ≥ 0.4 + PM ≥ 60 + 3+ catalysts\n"
        "  WATCH           agreement ≥ 0.4 + PM ≥ 55\n"
        "  TRACK           weak agreement but PM ≥ 50\n"
        "  STAGNANT        age ≥ 80d but agreement not strong\n"
        "  IGNORE          weak agreement + low score\n\n"
        "Risk override: FADING/WEAKENING classification\n"
        "  → IGNORE (or 'WATCH (caution)' if PM still strong)",
        PM_FILL, PM_EDGE,
        title_size=9.5, content_size=7.8)

    # ── Tab routing summary ──
    section_h2(ax, 4, 17, "Tab Routing + Phase 1-3 Macro Context Tags (api.py post-load)",
               color=GATE_EDGE, fontsize=11)

    text_block(ax, 4, 14.5,
        "  Momentum tab     : passes ALL 4 gate conditions  ->  current actionable holdings\n"
        "  Excluded tab     : bearish classification OR demoted by WeakQVR  ->  monitor for exit / avoidance\n"
        "  Pre-Momentum tab : not yet eligible BUT pre_momentum_score signals forming  ->  watchlist / preparation\n\n"
        "  -- Macro Context Tags (api.py post-load) --\n"
        "  * Phase 1     : cyclical_tag . style_tilt . region . industry_group (+ Biotech/Telecom industry refinement)\n"
        "  * Phase 1G    : ve_obs tag injection -> /api/validation segmented hit rates (cyc/style/region)\n"
        "  * Phase 2C    : MacroRegimeAgent rotation_alignment sub-signal (weight 0.20)\n"
        "  * Phase 2D    : per-ticker rotation_long / rotation_short scores\n"
        "  * Phase 3B    : Regime-aware CYCLE_PEAK early promotion (Risk-Off + cyclical + OVEREXTENDED + OER>=70)\n\n"
        "  -- Hybrid Bottom-up (ETF constituent integration) --\n"
        "  * Phase A     : etf_holdings_pipeline.py fetches top-10 holdings via yfinance (70 ETFs cached)\n"
        "  * Phase A+B   : ETF sidecar metrics (constituent_breadth_mom, weighted_comp, coverage, leader_gap)\n"
        "                  + divergence_flag = {HEALTHY_TREND, NARROW_RALLY, STEALTH_STRENGTH, WRAPPER_DRAG, NEUTRAL}\n"
        "  * Phase C     : MarketCommentaryTab ETF Hybrid Health Card (per-ETF 8-col table + flag badges)\n"
        "  * Phase D     : parent_etf_signal -> MacroRegimeAgent etf_parent_signal sub-signal (weight 0.15)\n"
        "                  Stock = weighted avg of parent ETF flags; ETF = own constituent breadth\n"
        "  * Phase E     : Conviction Picks BuyScore += ETF flag bonus (STEALTH +10, HEALTHY +6, NARROW -4)",
        fontsize=7.2, color=FG)

    # End-of-document footer
    ax.text(50, 4.5,
            "End of document.  ·  Source: pre_momentum.py · price_discovery.py · qvr_agent.py · api.py · etf_holdings_pipeline.py",
            fontsize=8, color=MUTED, ha="center", style="italic")


# ═════════════════════════════════════════════════════════════════
# Main entry point — multi-page PDF
# ═════════════════════════════════════════════════════════════════

def draw_graph(out_path: str):
    pages = [
        draw_page1_diagram,
        draw_page2_data,
        draw_page3_axes,
        draw_page4_agents,
        draw_page5_scores,
        draw_page6_gate,
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with PdfPages(out_path) as pdf:
        for fn in pages:
            fig, ax = setup_page()
            fn(fig, ax)
            # Save with explicit figure size (no bbox trimming) — guarantees
            # all pages share identical dimensions in the final PDF.
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
    print(f"✓ Wrote {out_path}  ({len(pages)} pages)")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "reports", "score_dependency_graph.pdf")
    draw_graph(out)
