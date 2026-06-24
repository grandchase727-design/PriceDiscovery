"""
draw_agent_architecture.py  —  Agent Architecture (multi-page PDF)

Output: reports/agent_architecture.pdf

Pages:
  1. Overview — 6-tier Agent Map (the whole picture)
  2. L1 + L2 — Data Layer & per-ticker Signal Agents
  3. L3 — Composite Scorer + Classification + PreMomentum 5-agent
  4. L4 + L5 — Context/Regime overlay + Decision/Filtering tiers
  5. L6 + Validators — Output synthesis + SVE + MLRescorer
  6. LLM Agent Layer (Claude Max plan only) — bridge / catalyst / debate

This complements draw_dependency_graph.py (score-system view) by giving
the *agent-centric* view that's now necessary because the system has
grown LLM/sub-agent components.
"""

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os

# ── Palette (consistent with draw_dependency_graph.py) ──
BG    = "#ffffff"
FG    = "#1f2937"
MUTED = "#6b7280"
DIVIDER = "#e5e7eb"
LINE  = "#9ca3af"

DATA_FILL,  DATA_EDGE  = "#f3f4f6", "#6b7280"
SIG_FILL,   SIG_EDGE   = "#eef2ff", "#6366f1"
COMP_FILL,  COMP_EDGE  = "#ecfeff", "#0891b2"
PM_FILL,    PM_EDGE    = "#fff7ed", "#f97316"
QVR_FILL,   QVR_EDGE   = "#f5f3ff", "#7c3aed"
CTX_FILL,   CTX_EDGE   = "#fefce8", "#ca8a04"
GATE_FILL,  GATE_EDGE  = "#fef2f2", "#dc2626"
OUT_FILL,   OUT_EDGE   = "#f0fdf4", "#16a34a"
LLM_FILL,   LLM_EDGE   = "#fdf2f8", "#db2777"
VAL_FILL,   VAL_EDGE   = "#f1f5f9", "#475569"


# ─────────────────────────────────────────────────────────────────
# Drawing primitives
# ─────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, text, fill, edge, fontsize=9, weight="normal"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=2,
    ))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=FG, zorder=3,
            linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=0.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=8,
        color=color, linewidth=lw, alpha=0.75, zorder=1,
    ))


def setup_page(figsize=(14, 18)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect("auto"); ax.axis("off")
    return fig, ax


def page_header(ax, page_num, title, subtitle="", accent=COMP_EDGE):
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
    ax.plot([x, x + 28], [y - 1.2, y - 1.2], color=color, linewidth=0.8, alpha=0.4)


def text_block(ax, x, y, lines, fontsize=8.5, color=FG, linespacing=1.5, weight="normal"):
    text = "\n".join(lines) if isinstance(lines, list) else lines
    ax.text(x, y, text, fontsize=fontsize, color=color,
            va="top", ha="left", linespacing=linespacing, fontweight=weight, wrap=True)


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


# ═════════════════════════════════════════════════════════════════
# PAGE 1 — Overview (6-tier Agent Map)
# ═════════════════════════════════════════════════════════════════
def draw_page1_overview(fig, ax):
    ax.text(50, 97.5, "Price Discovery — Agent Architecture",
            ha="center", va="center", fontsize=18, fontweight="bold", color=FG)
    ax.text(50, 95, "23 in-house agents + LLM agent layer (Claude Max plan)",
            ha="center", va="center", fontsize=10, color=MUTED)

    # Tier labels (left margin)
    tiers = [
        (87, "L6  OUTPUT",      OUT_EDGE),
        (75, "L5  DECISION",    GATE_EDGE),
        (62, "L4  CONTEXT",     CTX_EDGE),
        (47, "L3  COMPOSITE",   COMP_EDGE),
        (32, "L2  SIGNAL",      SIG_EDGE),
        (17, "L1  DATA",        DATA_EDGE),
    ]
    for y, lbl, color in tiers:
        ax.text(2.5, y, lbl, fontsize=9, fontweight="bold",
                color=color, va="center")

    # Divider lines
    for y in [82, 70, 56, 41, 26]:
        ax.plot([6, 96], [y, y], color=DIVIDER, linewidth=0.6, zorder=0)

    # L6 — Output (3 boxes)
    box(ax, 10, 84, 23, 6, "ConvictionPicksAgent",  OUT_FILL, OUT_EDGE, fontsize=9)
    box(ax, 37, 84, 23, 6, "MarketCommentaryAgent", OUT_FILL, OUT_EDGE, fontsize=9)
    box(ax, 64, 84, 28, 6, "ReportRenderAgent\n(PDF/DOCX/Dashboard JSON)", OUT_FILL, OUT_EDGE, fontsize=8.5)

    # L5 — Decision (4)
    for i, (name, fs) in enumerate([
        ("EligibilityGateAgent", 8),
        ("AntiLagDiscoveryAgent", 8),
        ("SectorDiscoveryAgent", 8),
        ("HedgeStrategyAgent\n(8 strategies)", 8),
    ]):
        x = 8 + i*22
        box(ax, x, 72, 20, 5.5, name, GATE_FILL, GATE_EDGE, fontsize=fs)

    # L4 — Context (4)
    for i, name in enumerate([
        "MacroContextTagger",
        "RegimeDetectionAgent",
        "ETFHybridAgent",
        "GraphCommunityAgent",
    ]):
        x = 8 + i*22
        box(ax, x, 58.5, 20, 5.5, name, CTX_FILL, CTX_EDGE, fontsize=8.5)

    # L3 — Composite (3)
    box(ax, 10, 43, 25, 6, "CompositeScorer\n(0.30·TCS + 0.25·TFS_res + 0.30·RSS + 0.15·URS\n − 0.10·max(0,OER−40))", COMP_FILL, COMP_EDGE, fontsize=7)
    box(ax, 37, 43, 25, 6, "ClassificationAgent\n(3×3 + 6 overrides\n + Sticky-FLAT hysteresis)", COMP_FILL, COMP_EDGE, fontsize=8)
    box(ax, 64, 43, 28, 6, "PreMomentumOrchestrator\n(5 sub-agents: Micro / Macro / Graph / Catalyst / QVR)", PM_FILL, PM_EDGE, fontsize=8)

    # L2 — Signal agents (6)
    for i, (name, w) in enumerate([
        ("TCS", "0.30"), ("TFS", "0.25"), ("RSS", "0.30"),
        ("URS", "0.15"), ("OER", "penalty"), ("QVR", "→ L3"),
    ]):
        x = 8 + i*14.5
        box(ax, x, 28, 12.5, 5, f"{name}\nAgent\n({w})", SIG_FILL, SIG_EDGE, fontsize=8)

    # L1 — Data (4)
    for i, name in enumerate([
        "MarketDataAgent\n(yfinance OHLCV)",
        "FundamentalDataAgent\n(yfinance + Finnhub)",
        "ETFHoldingsAgent\n(yfinance)",
        "UniverseClassifierAgent\n(GICS)",
    ]):
        x = 8 + i*22
        box(ax, x, 13, 20, 5.5, name, DATA_FILL, DATA_EDGE, fontsize=7.5)

    # Validators (right margin column)
    ax.add_patch(FancyBboxPatch((85, 8), 13, 75, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor="#fafafa", edgecolor=VAL_EDGE, linewidth=1.0,
                                linestyle="--", alpha=0.7, zorder=0))
    ax.text(91.5, 86, "VALIDATORS\n(orthogonal)", fontsize=8, fontweight="bold",
            color=VAL_EDGE, ha="center", va="top", linespacing=1.3)
    box(ax, 86, 70, 11, 7, "SignalValidity\nEngine", VAL_FILL, VAL_EDGE, fontsize=7.5)
    box(ax, 86, 58, 11, 7, "ML\nRescorer", VAL_FILL, VAL_EDGE, fontsize=8)

    # Flow arrows (vertical, simplified)
    for x in [20, 50, 78]:
        for y_pair in [(13+5.5, 28), (28+5, 43), (43+6, 58.5), (58.5+5.5, 72), (72+5.5, 84)]:
            arrow(ax, x, y_pair[0], x, y_pair[1])

    # Footer
    ax.text(50, 5, "See PAGES 2-6 for per-tier detail.  "
            "LLM layer (PAGE 6) is fed exclusively from within Claude Code sessions — "
            "no out-of-plan API spend.",
            fontsize=8, color=MUTED, ha="center", va="center", style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — L1 Data + L2 Signal Agents
# ═════════════════════════════════════════════════════════════════
def draw_page2_data_signals(fig, ax):
    page_header(ax, 2, "L1 Data + L2 Signal Agents",
                "Per-ticker, vectorized compute → cross-sectional standardization",
                accent=SIG_EDGE)

    # L1 section
    section_h2(ax, 6, 88, "L1 — Data Layer (4 agents)", color=DATA_EDGE)
    box(ax, 6, 78, 22, 7, "MarketDataAgent\n• yfinance OHLCV\n• adj-close, fast_info\n• 770 tickers × T days",
        DATA_FILL, DATA_EDGE, fontsize=8)
    box(ax, 30, 78, 22, 7, "FundamentalDataAgent\n• yfinance + Finnhub\n• PE / PEG / margin / ROE\n• revisions, surprises",
        DATA_FILL, DATA_EDGE, fontsize=8)
    box(ax, 54, 78, 22, 7, "ETFHoldingsAgent\n• yfinance top_holdings\n• 70 sector / thematic ETFs\n• weekly refresh",
        DATA_FILL, DATA_EDGE, fontsize=8)
    box(ax, 78, 78, 16, 7, "UniverseClassifier\n• GICS sector / industry\n• cap-tier / region",
        DATA_FILL, DATA_EDGE, fontsize=8)

    # L2 section
    section_h2(ax, 6, 71, "L2 — Signal Agents (6, per-ticker)", color=SIG_EDGE)
    text_block(ax, 6, 68,
        "Each agent follows compute_raw() → score() → percentile_rank(). All outputs are 0..100 and "
        "are cross-sectionally standardized before flowing into L3.",
        fontsize=8.5, color=MUTED, linespacing=1.4)

    signals = [
        ("TCSAgent",  "Trend Continuation",
            ["SMA20/50/200 distance + slope",
             "trend age (long-horizon weighted 60%)",
             "score_tcs_short + score_tcs_long"],
            (6, 50)),
        ("TFSAgent",  "Trend Formation (residualized)",
            ["volatility compression, BB squeeze",
             "volume surge, structural breakout",
             "cross-sectional OLS removes TCS overlap"],
            (52, 50)),
        ("RSSAgent",  "Relative Strength (hybrid)",
            ["0.6 within-sector + 0.4 universe",
             "rss_short / rss_long percentiles",
             "small categories n<8 → universe fallback"],
            (6, 35)),
        ("URSAgent",  "Underreaction",
            ["LeadLag + AttnGap + Drift + Dispersion",
             "behavioral overlay (post-event)",
             "4 sub-signal blend"],
            (52, 35)),
        ("OERAgent",  "Overextension (penalty)",
            ["52w-high distance, RSI(14), dist_days",
             "feeds Composite as −0.10·max(0, OER−40)",
             "asymmetric: caps at −6pts"],
            (6, 20)),
        ("QVRAgent",  "Quality + Value + Revision",
            ["cross-sectional percentile of Q/V/R",
             "doubles as Eligibility Gate input",
             "Stock-only; ETFs bypass QVR ≥ 40 check"],
            (52, 20)),
    ]
    for name, sub, bullets, (x, y) in signals:
        info_box(ax, x, y, 42, 12, name, sub + "\n  • " + "\n  • ".join(bullets),
                 SIG_FILL, SIG_EDGE, title_size=9.5, content_size=7.5)

    # Footer note
    text_block(ax, 6, 8,
        ["Cross-sectional standardization happens ONCE per scan (single pass): TFS residualized vs TCS, "
         "RSS hybrid blend, universe percentiles. Output is consumed by L3 CompositeScorer."],
        fontsize=8, color=MUTED, linespacing=1.4)


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — L3 Composite + Classification + PreMomentum 5-agent
# ═════════════════════════════════════════════════════════════════
def draw_page3_composite(fig, ax):
    page_header(ax, 3, "L3 — Composite + Classification + PreMomentum",
                "Where signals become decisions: 4-axis composite + 3×3 matrix + 5-agent PM",
                accent=COMP_EDGE)

    # CompositeScorer
    section_h2(ax, 6, 88, "CompositeScorer", color=COMP_EDGE)
    info_box(ax, 6, 72, 88, 14, "Formula (2026-05 update)",
        "base = 0.30·TCS + 0.25·TFS_residualized + 0.30·RSS_hybrid + 0.15·URS\n"
        "Composite = base − 0.10·max(0, OER − 40)        # OER penalty, capped −6 pts\n\n"
        "• TFS_residualized via cross-sectional OLS (removes TCS↔TFS info overlap)\n"
        "• RSS_hybrid = 0.6 · within-sector percentile + 0.4 · universe percentile\n"
        "• OER < 40 → no penalty; OER = 100 → −6 pts (penalizes overheated names, not healthy trends)",
        COMP_FILL, COMP_EDGE, title_size=10, content_size=8.5)

    # ClassificationAgent
    section_h2(ax, 6, 68, "ClassificationAgent — 3×3 Matrix", color=COMP_EDGE)

    # 3x3 grid (colors carry the meaning; emoji omitted for font portability)
    cls = [
        [("CONT.",  "#dcfce7"), ("RECOV.", "#dbeafe"), ("COUNTER", "#f3e8ff")],
        [("CONSOL", "#fef3c7"), ("NEUT.",  "#ffedd5"), ("FADING",  "#fef3c7")],
        [("PULLBK", "#ffedd5"), ("WEAKEN", "#fef2f2"), ("DOWN",    "#fee2e2")],
    ]
    for ri, row in enumerate(cls):
        for ci, (lbl, fill) in enumerate(row):
            x = 12 + ci*10; y = 56 - ri*5
            box(ax, x, y, 9, 4.5, lbl, fill, "#9ca3af", fontsize=7.5)
    # axis labels
    for ci, lbl in enumerate(["UP", "FLAT", "DOWN"]):
        ax.text(16.5 + ci*10, 62.5, lbl, fontsize=8, fontweight="bold", color=COMP_EDGE, ha="center")
    for ri, lbl in enumerate(["UP", "FLAT", "DOWN"]):
        ax.text(10, 58.3 - ri*5, lbl, fontsize=8, fontweight="bold", color=COMP_EDGE, ha="center", va="center")
    ax.text(16.5, 65, "← long_dir →", fontsize=7, color=MUTED, ha="center", style="italic")
    ax.text(10, 47.5, "↑\nshort_dir\n↓", fontsize=7, color=MUTED, ha="center", style="italic", linespacing=1.2)

    # Overrides box
    info_box(ax, 47, 39, 47, 25, "Overrides (6) + Hysteresis",
        "  OVEREXTENDED    — OER ≥ 60 on bullish base\n"
        "  FORMATION       — rapid short breakout\n"
        "  EXHAUSTING      — peak signs after long uptrend\n"
        "  CYCLE_PEAK      — top of full cycle (regime-aware)\n"
        "  LAGGING_CATCHUP — catching up to peers\n"
        "→ Phase 3B: regime-aware override in Risk-Off ×\n"
        "    cyclical × OVEREXTENDED × OER≥70 → CYCLE_PEAK\n\n"
        "STICKY-FLAT HYSTERESIS\n"
        "  if prev_classification = FLAT → entry threshold × 1.3\n"
        "  → suppresses bi-weekly NEUTRAL↔CONSOL↔RECOV flips",
        COMP_FILL, COMP_EDGE, title_size=9, content_size=7.5)

    # PreMomentum 5-agent
    section_h2(ax, 6, 35, "PreMomentumOrchestrator — 5 sub-agents", color=PM_EDGE)

    pm_agents = [
        ("Microstructure\n0.20", "vol compression\naccumulation\nBB squeeze"),
        ("MacroRegime\n0.15",    "6 sub-signals\n→ rotation_alignment\n→ etf_parent_signal"),
        ("GraphRelational\n0.20", "theme breadth\nleader-lagger gap\ncommunity momentum"),
        ("Catalyst\n0.20",       "rss_short − rss_long\nstrategy_agreement\nreversal_risk"),
        ("QVR\n0.25",            "★ only PM agent\northogonal to Composite\n(corr ≈ 0)"),
    ]
    for i, (name, body) in enumerate(pm_agents):
        x = 7 + i*18
        # header box
        box(ax, x, 27, 16, 5, name, PM_FILL, PM_EDGE, fontsize=8, weight="bold")
        # body box
        info_box(ax, x, 11, 16, 14, "", body, "#ffffff", PM_EDGE,
                 title_size=1, content_size=7)

    # Final formula
    ax.text(50, 6,
            "pre_momentum_score = 0.20·Micro + 0.15·Macro + 0.20·Graph + 0.20·Catalyst + 0.25·QVR     "
            "|     agreement_ratio = count(agent>50) / 5",
            fontsize=8.5, color=PM_EDGE, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PM_FILL, edgecolor=PM_EDGE, linewidth=0.8))


# ═════════════════════════════════════════════════════════════════
# PAGE 4 — L4 Context/Regime + L5 Decision tiers
# ═════════════════════════════════════════════════════════════════
def draw_page4_context_decision(fig, ax):
    page_header(ax, 4, "L4 Context/Regime + L5 Decision Filtering",
                "Cross-sectional macro overlay + multi-tier eligibility logic",
                accent=CTX_EDGE)

    # L4
    section_h2(ax, 6, 88, "L4 — Context & Regime (4 agents)", color=CTX_EDGE)

    info_box(ax, 6, 70, 42, 15, "MacroContextTagger",
        "Per-ticker tags (Phase 1.0/1.5):\n"
        "  • cyclical_tag  : cyclical / defensive / broad\n"
        "  • style_tilt    : growth / value / balanced\n"
        "  • region        : US / Korea / Japan / China / ...\n"
        "  • industry_group: GICS industry group name\n\n"
        "Industry refinement:\n"
        "  Biotech → cyclical/growth, Telecom → defensive/value",
        CTX_FILL, CTX_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 52, 70, 42, 15, "RegimeDetectionAgent",
        "STATE['regime']:\n"
        "  • cyclical_dom / defensive_dom (gap > 3)\n"
        "  • growth_dom / value_dom\n"
        "  • top_region / bot_region\n\n"
        "Per-ticker:\n"
        "  • rotation_long  / rotation_short (0..100)\n"
        "  → Phase 2C alignment, Phase 3B CYCLE_PEAK",
        CTX_FILL, CTX_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 6, 53, 42, 15, "ETFHybridAgent",
        "7-field sidecar per ETF:\n"
        "  • constituent_breadth_mom / weighted_comp\n"
        "  • coverage / concentration / leader_gap\n"
        "  • parent_etf_signal (for stocks)\n\n"
        "4 divergence flags:\n"
        "  HEALTHY_TREND / NARROW_RALLY / STEALTH_STRENGTH / WRAPPER_DRAG",
        CTX_FILL, CTX_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 52, 53, 42, 15, "GraphCommunityAgent",
        "GraphRAG knowledge graph (graph_engine.py):\n"
        "  • Louvain communities (theme clusters)\n"
        "  • theme propagation insights\n"
        "  • leader-lagger detection\n"
        "  • ETF–stock divergence flags\n"
        "  • multi-hop: impact_radius / theme_status",
        CTX_FILL, CTX_EDGE, title_size=9.5, content_size=7.5)

    # L5 Decision
    section_h2(ax, 6, 49, "L5 — Decision & Filtering (4 agents → tiers)", color=GATE_EDGE)

    info_box(ax, 6, 32, 42, 15, "EligibilityGateAgent",
        "4 conditions, ALL required:\n"
        "  1. Composite ≥ 55\n"
        "  2. classification ∈ bullish set\n"
        "  3. ADV ≥ $5M (liquidity floor)\n"
        "  4. asset = ETF  OR  QVR ≥ 40\n\n"
        "Pass → Momentum tab  |  Fail → Excluded tab (with rejection tag)",
        GATE_FILL, GATE_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 52, 32, 42, 15, "AntiLagDiscoveryAgent (Phase 1)",
        "ProvisionalPM tier:\n"
        "  PM ≥ 45 AND agreement_ratio ≥ 0.6 AND\n"
        "  bullish classification AND NOT eligible\n\n"
        "→ surfaces strong forward-looking signals BEFORE\n"
        "  Composite breakout. Goal: 10-15-day lag reduction.\n"
        "  → Anti Lag Discovery sub-tab",
        GATE_FILL, GATE_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 6, 15, 42, 15, "SectorDiscoveryAgent",
        "Per-sector top-5 (forced diversification):\n"
        "  within each sector, top-5 by Composite\n"
        "  AND composite ≥ 40 AND bullish\n\n"
        "→ Sector Discovery sub-tab\n"
        "Goal: forced diversification + sector-best capture\n"
        "      (does NOT reduce lag)",
        GATE_FILL, GATE_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 52, 15, 42, 15, "HedgeStrategyAgent",
        "8 quant strategies (hedge_strategies.py):\n"
        "  Combined / Flow / Wyckoff / Ichimoku /\n"
        "  Darvas / Minervini / sector_rotation / ...\n\n"
        "Per-strategy long/short top picks → surfaced via\n"
        "/api/quant-strategies, Strategy cards in dashboard",
        GATE_FILL, GATE_EDGE, title_size=9.5, content_size=7.5)

    # Tier hierarchy callout (bottom)
    ax.text(50, 9,
            "TIERS:   EligibleMomentum   →   ProvisionalPM (Anti-Lag)   →   PreMomentum   →   Excluded     "
            "|     v2:   BothEligible / UniverseOnly / SectorOnly / Neither",
            fontsize=8, color=GATE_EDGE, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=GATE_FILL, edgecolor=GATE_EDGE, linewidth=0.8))


# ═════════════════════════════════════════════════════════════════
# PAGE 5 — L6 Output + Validators (SVE, MLRescorer)
# ═════════════════════════════════════════════════════════════════
def draw_page5_output_validators(fig, ax):
    page_header(ax, 5, "L6 Output Synthesis + Validators",
                "Top of stack: synthesis + orthogonal hit-rate validation",
                accent=OUT_EDGE)

    # L6
    section_h2(ax, 6, 88, "L6 — Output & Synthesis (3 agents)", color=OUT_EDGE)

    info_box(ax, 6, 70, 42, 15, "ConvictionPicksAgent",
        "Buy/Sell top-5 Stocks + ETFs:\n\n"
        "BuyScore = Composite + Class + Consensus + 1M/Sector regime\n"
        "  − OER penalty\n"
        "  − WeakQVR penalty (Stock-only)\n"
        "  + ETF flag bonus (HEALTHY_TREND etc.)\n\n"
        "(!) amber row highlight for WeakQVR Buy candidates",
        OUT_FILL, OUT_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 52, 70, 42, 15, "MarketCommentaryAgent",
        "~15k-char auto-report:\n"
        "  Executive Summary + Conviction Picks +\n"
        "  Market Leaders (5-axis leadership) +\n"
        "  ETF Hybrid Health + 23 narrative sections\n\n"
        "Includes the new ConvictionDebateCard panel\n"
        "(L6 ↔ LLM layer integration — see PAGE 6)",
        OUT_FILL, OUT_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 6, 53, 88, 15, "ReportRenderAgent",
        "Multi-format output:\n"
        "  • PDF — daily Omega(PD_v5)_YYYYMMDD.pdf in reports/  (matplotlib dark-theme)\n"
        "  • DOCX — earnings-analysis skill outputs (e.g. Nvidia_Q1_FY27_Earnings_Update.docx)\n"
        "  • FastAPI JSON — /api/table, /api/pre-momentum, /api/conviction-debate, /api/sector-rotation, ...\n"
        "  • React dashboard (frontend/) — 8 top-level tabs, 5 PD sub-tabs, ConvictionDebateCard\n"
        "  • Sector overview reports (markdown) — pitch-agent:sector-overview outputs",
        OUT_FILL, OUT_EDGE, title_size=9.5, content_size=7.5)

    # Validators
    section_h2(ax, 6, 49, "Orthogonal Validators (SVE + ML)", color=VAL_EDGE)

    info_box(ax, 6, 28, 42, 19, "SignalValidityEngine",
        "24 bi-weekly Friday-anchored eval points × 63d:\n"
        "  • bucket hit rates (Composite quintiles)\n"
        "  • per-class hit rates\n"
        "  • per-ticker hit rates\n"
        "  • segmented by:\n"
        "      - cyclical_tag (cyclical/defensive/broad)\n"
        "      - style_tilt (growth/value/balanced)\n"
        "      - region\n"
        "      - llm_catalyst_event_type (new)\n\n"
        "Feeds /api/validation + Validation tab",
        VAL_FILL, VAL_EDGE, title_size=9.5, content_size=7.5)

    info_box(ax, 52, 28, 42, 19, "MLRescorerAgent",
        "ML overlay on top of L3 outputs:\n"
        "  • feature_pipeline.py — engineered features\n"
        "  • score_ml.py — LightGBM rescorer\n"
        "  • purged_cv.py — leakage-safe CV\n"
        "  • meta_labeling.py — tri-class meta labels\n\n"
        "Output: /api/ml/* mirror endpoints,\n"
        "       Price Discovery (ML) tab + AI Prediction tab\n\n"
        "Used for the composite_live + ml variants of\n"
        "sector_rotation backtest (signal_mode params)",
        VAL_FILL, VAL_EDGE, title_size=9.5, content_size=7.5)

    # Footer
    text_block(ax, 6, 18,
        ["DAILY EXECUTION FLOW:",
         "  1. price_discovery.py (full scan, ~15min)  →  .scan_cache.pkl",
         "  2. fundamentals_pipeline.py  →  .fundamentals_cache.pkl",
         "  3. finnhub_fundamentals.py (US enrich, ~28min)",
         "  4. uvicorn restart  →  api.py post-load applies L4 + L5 + Pre-Mom",
         "  5. SVE runs inside run_scan() (no separate pass)",
         "  6. MLRescorer runs in /api/ml/* endpoints (lazy)",
         "",
         "→ New layer (PAGE 6): in-session LLM enrichment via Claude Code (Manual or CronCreate)"],
        fontsize=8, color=FG, linespacing=1.5)


# ═════════════════════════════════════════════════════════════════
# PAGE 6 — LLM Agent Layer (Claude Max plan only)
# ═════════════════════════════════════════════════════════════════
def draw_page6_llm_layer(fig, ax):
    page_header(ax, 6, "LLM Agent Layer — Claude Max plan only",
                "External MCP data + Multi-Agent ConvictionDebate, no out-of-plan API spend",
                accent=LLM_EDGE)

    # Hard rule banner
    ax.add_patch(FancyBboxPatch((6, 88), 88, 5,
                                boxstyle="round,pad=0.1,rounding_size=0.1",
                                facecolor="#fce7f3", edgecolor=LLM_EDGE,
                                linewidth=1.5, zorder=1))
    ax.text(50, 90.5, "HARD RULE  —  No code in agents/ ever calls the Anthropic API directly.   "
            "All LLM / sub-agent / MCP work happens inside an active Claude Code session.",
            fontsize=8.5, color=LLM_EDGE, ha="center", va="center", fontweight="bold")

    # The 3 agents
    section_h2(ax, 6, 84, "Three pure-Python agents (no network I/O)", color=LLM_EDGE)

    info_box(ax, 6, 66, 28, 14, "claude_finance_bridge",
        "Aiera / FactSet / Daloopa\nMCP adapter (in-session)\n\n"
        "• inject_callables(aiera=fn, ...)\n"
        "• CatalystSignal / RevisionSignal\n  / FundamentalSignal\n"
        "• cache TTL: 7d / 1d / 30d\n"
        "• MCP unavail → neutral 50",
        LLM_FILL, LLM_EDGE, title_size=9, content_size=7)

    info_box(ax, 36, 66, 28, 14, "llm_catalyst_agent",
        "Plug-in to CatalystAgent +\nQVR R sub-axis\n\n"
        "• enrich_with_llm_agents(cache)\n"
        "• Top-50 cost control (filter)\n"
        "• adds catalyst_score_v2,\n  qvr_r_v2, llm_*\n"
        "• report_llm_movers() helper",
        LLM_FILL, LLM_EDGE, title_size=9, content_size=7)

    info_box(ax, 66, 66, 28, 14, "conviction_debate",
        "Multi-Agent Debate runner\n(in-session sub-agent helper)\n\n"
        "• select_debate_targets()\n"
        "• build_debate_prompt()\n"
        "• parse_verdict_text()\n"
        "• save_all_verdicts() →\n   .conviction_debate_cache.json",
        LLM_FILL, LLM_EDGE, title_size=9, content_size=7)

    # Execution flow diagram
    section_h2(ax, 6, 60, "Execution Flow — Claude Code session boundary", color=LLM_EDGE)

    # Session box (outer)
    ax.add_patch(FancyBboxPatch((6, 30), 88, 27,
                                boxstyle="round,pad=0.2,rounding_size=0.3",
                                facecolor="#fdf2f8", edgecolor=LLM_EDGE,
                                linewidth=1.5, linestyle="--", zorder=0))
    ax.text(50, 55, "CLAUDE CODE SESSION  (plan-billed)",
            fontsize=10, color=LLM_EDGE, ha="center", va="center", fontweight="bold")

    # Inside-session steps
    box(ax, 10, 45, 18, 6, "1. MCP authenticate\n(one-time)",         "#fff", LLM_EDGE, fontsize=8)
    box(ax, 30, 45, 18, 6, "2. bridge.inject_\ncallables(...)",       "#fff", LLM_EDGE, fontsize=8)
    box(ax, 50, 45, 18, 6, "3. enrich_with_llm\n_agents(cache)",      "#fff", LLM_EDGE, fontsize=8)
    box(ax, 70, 45, 18, 6, "4. select_debate_\ntargets(cache,10)",    "#fff", LLM_EDGE, fontsize=8)
    box(ax, 10, 35, 36, 6, "5. Agent(market-researcher) × Top-N\n   (parallel, plan-billed)",
        "#fff", LLM_EDGE, fontsize=8)
    box(ax, 50, 35, 38, 6, "6. parse_verdict_text() + save_all_verdicts()\n   → .conviction_debate_cache.json",
        "#fff", LLM_EDGE, fontsize=8)
    for x_pair in [(28, 30), (48, 50), (68, 70)]:
        arrow(ax, x_pair[0], 48, x_pair[1], 48, color=LLM_EDGE)
    arrow(ax, 28, 38, 50, 38, color=LLM_EDGE)
    arrow(ax, 78, 45, 78, 41, color=LLM_EDGE)  # 4 → 6

    # Outside session: server
    box(ax, 6, 19, 42, 6,
        "uvicorn api.py  (always-on, NO LLM call)\n→ /api/conviction-debate just reads JSON",
        "#ffffff", OUT_EDGE, fontsize=8.5, weight="bold")
    box(ax, 52, 19, 42, 6,
        "React dashboard — ConvictionDebateCard\n(Market Commentary tab, expandable rows)",
        "#ffffff", OUT_EDGE, fontsize=8.5, weight="bold")
    arrow(ax, 48, 22, 52, 22, color=OUT_EDGE)
    # Cache → server arrow
    arrow(ax, 70, 32, 27, 26, color=LLM_EDGE, lw=0.7)
    ax.text(50, 28.5, "cache JSON",
            fontsize=7, color=MUTED, ha="center", style="italic")

    # Two run modes
    section_h2(ax, 6, 16, "Two run modes (both inside Claude Max plan)", color=LLM_EDGE)
    info_box(ax, 6, 4, 42, 11, "Mode A — Manual",
        "User asks Claude:\n"
        "  \"Run today's Top-N conviction debate\"\n\n"
        "Claude opens session, runs steps 1-6, writes cache,\n"
        "exits.  Server picks up on next /api/conviction-debate.",
        LLM_FILL, LLM_EDGE, title_size=9, content_size=7.5)

    info_box(ax, 52, 4, 42, 11, "Mode B — Scheduled",
        "User asks Claude once:\n"
        "  \"Set cron to run 1h after market close daily\"\n\n"
        "Claude calls CronCreate(...).  At each fire time a\n"
        "fresh Claude Code session runs steps 1-6.",
        LLM_FILL, LLM_EDGE, title_size=9, content_size=7.5)


# ═════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════
def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "agent_architecture.pdf")

    with PdfPages(out_pdf) as pdf:
        for page_fn in [
            draw_page1_overview,
            draw_page2_data_signals,
            draw_page3_composite,
            draw_page4_context_decision,
            draw_page5_output_validators,
            draw_page6_llm_layer,
        ]:
            fig, ax = setup_page()
            page_fn(fig, ax)
            pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
            plt.close(fig)

    print(f"Wrote: {out_pdf}")
    print(f"Pages: 6")
    sz_kb = os.path.getsize(out_pdf) / 1024
    print(f"Size: {sz_kb:.1f} KB")


if __name__ == "__main__":
    main()
