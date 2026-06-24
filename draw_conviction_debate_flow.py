"""
draw_conviction_debate_flow.py — Multi-Agent ConvictionDebate Decision Flow (5-page PDF)

Output: reports/conviction_debate_flow.pdf

Pages:
  1. Overview — 9-step decision pipeline (vertical timeline)
  2. Step-by-step detail (9 cards)
  3. Case A: CNQ — CONSENSUS_BUY (R1 stop, simple path)
  4. Case B: QCLN — POLAR_SPLIT (R2 cross-examination triggered)
  5. Cost profile + In-session vs Server boundary
"""

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os

# ── Palette (consistent with draw_agent_architecture.py) ──
BG    = "#ffffff"
FG    = "#1f2937"
MUTED = "#6b7280"
DIVIDER = "#e5e7eb"
LINE  = "#9ca3af"

DATA_FILL,  DATA_EDGE  = "#f3f4f6", "#6b7280"   # gray — data
SEL_FILL,   SEL_EDGE   = "#ecfeff", "#0891b2"   # cyan — selection
PM_FILL,    PM_EDGE    = "#fff7ed", "#f97316"   # orange — prompts
LLM_FILL,   LLM_EDGE   = "#fdf2f8", "#db2777"   # purple — LLM (in-session)
CTX_FILL,   CTX_EDGE   = "#fefce8", "#ca8a04"   # yellow — convergence
GATE_FILL,  GATE_EDGE  = "#fef2f2", "#dc2626"   # red — Round 2 (rare)
OUT_FILL,   OUT_EDGE   = "#f0fdf4", "#16a34a"   # green — synthesis
PERSIST_FILL, PERSIST_EDGE = "#f1f5f9", "#475569"  # slate — cache + UI

# Rating colors (for case studies)
BUY_COLOR   = "#16a34a"
HOLD_COLOR  = "#6b7280"
SELL_COLOR  = "#f97316"
AVOID_COLOR = "#dc2626"


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


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=10,
        color=color, linewidth=lw, alpha=0.85, zorder=1,
    ))


def setup_page(figsize=(14, 18)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_aspect("auto"); ax.axis("off")
    return fig, ax


def page_header(ax, page_num, title, subtitle="", accent=LLM_EDGE):
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
# PAGE 1 — Overview: 9-step decision pipeline (vertical timeline)
# ═════════════════════════════════════════════════════════════════
def draw_page1_overview(fig, ax):
    page_header(ax, 1, "Multi-Agent ConvictionDebate — 9-Step Decision Pipeline",
                "Selection (server, auto) → Debate (in-session, LLM) → Persist + Surface",
                accent=LLM_EDGE)

    # Left rail label
    ax.text(4, 88, "DATA\nLAYER", fontsize=9, fontweight="bold", color=DATA_EDGE,
            ha="center", va="center", linespacing=1.2)
    ax.text(4, 75, "SELECTION\n(server-side\nauto)", fontsize=9, fontweight="bold", color=SEL_EDGE,
            ha="center", va="center", linespacing=1.2)
    ax.text(4, 53, "MULTI-AGENT\nDEBATE\n(in-session\nLLM only)", fontsize=9, fontweight="bold", color=LLM_EDGE,
            ha="center", va="center", linespacing=1.2)
    ax.text(4, 24, "SYNTHESIS\n+ PERSIST", fontsize=9, fontweight="bold", color=OUT_EDGE,
            ha="center", va="center", linespacing=1.2)

    # 9 steps as vertical chain
    steps = [
        # (y, label, body, fill, edge)
        (87, "STEP 1: DATA REFRESH",
         "price_discovery.py -> .scan_cache.pkl (770 ticker x 80+ fields)\n"
         "uvicorn /api/reload -> api.py STATE",
         DATA_FILL, DATA_EDGE),
        (78, "STEP 2: SELECTION (auto, no LLM)",
         "BuyScore = Composite + Class +/- OER + Consensus + 1M/Sector regime - WeakQVR + ETF flag\n"
         "Top 5 stock + Top 5 ETF DESC (LONG)  /  Bottom 5 stock + Bottom 5 ETF ASC (SHORT)\n"
         "Rotation cooldown (2-week)  ->  20 ticker targets",
         SEL_FILL, SEL_EDGE),
        (68, "STEP 3: SPECIALIST PROMPT BUILD",
         "Per ticker x 3 personas (Fundamental / Sentiment / Valuation)\n"
         "Inject quant readings (Composite, OER, TCS, TFS, RSS, URS, QVR)\n"
         "Lane enforcement: STAY IN YOUR LANE",
         PM_FILL, PM_EDGE),
        (58, "STEP 4: ROUND 1 - 3 PARALLEL SUB-AGENT CALLS",
         "Agent(market-researcher, fund_prompt)  +  sent_prompt  +  val_prompt\n"
         "3 specialists work independently, no cross-knowledge of others\n"
         "~30-40s per call (parallel)",
         LLM_FILL, LLM_EDGE),
        (48, "STEP 5: PARSE",
         "parse_specialist_opinion(): free-text -> SpecialistOpinion\n"
         "Extract rating (BUY/HOLD/SELL/AVOID), confidence (0-1), key_points",
         LLM_FILL, LLM_EDGE),
        (40, "STEP 6: CONVERGENCE CHECK",
         "CONSENSUS_BUY/HOLD/SELL  (1 tier)            -> STOP\n"
         "ENTRY_TIMING (BUY/HOLD)  /  EXIT_TIMING (HOLD/SELL)  (adjacent) -> STOP\n"
         "POLAR_SPLIT (BUY/SELL)  /  WIDE_SPAN (3 tiers)        -> R2 trigger",
         CTX_FILL, CTX_EDGE),
        (30, "STEP 7 (conditional): ROUND 2 CROSS-EXAM",
         "Only if POLAR_SPLIT or WIDE_SPAN.  Each specialist sees other 2's R1 opinions.\n"
         "Re-issue rating + confidence delta. Usually narrows but rating may persist.",
         GATE_FILL, GATE_EDGE),
        (21, "STEP 8: DUAL SYNTHESIS (rule-based, no LLM)",
         "NEUTRAL mode: confidence-weighted ensemble  ->  rating + modifier [-5..+5]\n"
         "AVERSE  mode: bear/risk language x1.2 weight  ->  typically 1 tier more conservative\n"
         "risk_mode_gap = |neutral_tier - averse_tier|",
         OUT_FILL, OUT_EDGE),
        (12, "STEP 9: PERSIST + SURFACE",
         ".multi_agent_debate_cache.json (immutable, append-only)\n"
         "/api/conviction-debate/multi -> selection (fresh) + cache (lookup) merge\n"
         "Dashboard 4 sections: Stock LONG/SHORT, ETF LONG/SHORT",
         PERSIST_FILL, PERSIST_EDGE),
    ]
    for y, label, body, fill, edge in steps:
        # Step box on right of left rail
        info_box(ax, 10, y - 4, 84, 5.5, label, body, fill, edge,
                 title_size=9.5, content_size=7.5)
        # Vertical chain arrow between consecutive steps
        if y > 12:
            arrow(ax, 50, y - 4, 50, y - 5.5, color=LINE, lw=1.2)

    # Footer
    ax.text(50, 5.5, "Pages 2-5 expand each step + walk through 2 real cases (CNQ consensus / QCLN polar-split).",
            fontsize=8.5, color=MUTED, ha="center", style="italic")


# ═════════════════════════════════════════════════════════════════
# PAGE 2 — Step-by-step detail (9 cards in 3x3 grid)
# ═════════════════════════════════════════════════════════════════
def draw_page2_step_detail(fig, ax):
    page_header(ax, 2, "Step Detail — Input / Process / Output",
                "Each step: what goes in, what happens, what comes out", accent=LLM_EDGE)

    cards = [
        # (col, row, title, body, fill, edge)
        (0, 0, "1. DATA REFRESH",
         "IN:  user trigger or cron\n"
         "DO:  scan 770 ticker, compute 80+ fields\n"
         "OUT: .scan_cache.pkl + api.py STATE",
         DATA_FILL, DATA_EDGE),
        (1, 0, "2. SELECTION",
         "IN:  scan_cache + sector regime + consensus\n"
         "DO:  BuyScore -> rank DESC/ASC by asset\n"
         "OUT: Top 5+5 long + Top 5+5 short = 20",
         SEL_FILL, SEL_EDGE),
        (2, 0, "3. PROMPT BUILD",
         "IN:  20 targets + quant readings per ticker\n"
         "DO:  formula 3 persona prompts per ticker\n"
         "OUT: 60 prompts (20x3)  total",
         PM_FILL, PM_EDGE),
        (0, 1, "4. ROUND 1",
         "IN:  60 prompts\n"
         "DO:  3 parallel sub-agent calls per ticker\n"
         "OUT: 60 free-text specialist opinions",
         LLM_FILL, LLM_EDGE),
        (1, 1, "5. PARSE",
         "IN:  60 free-text\n"
         "DO:  regex + bullet extract\n"
         "OUT: 60 SpecialistOpinion (rating, conf, ...)",
         LLM_FILL, LLM_EDGE),
        (2, 1, "6. CONVERGENCE",
         "IN:  per ticker {Fund, Sent, Val} ratings\n"
         "DO:  classify 6 patterns\n"
         "OUT: converged?  type label",
         CTX_FILL, CTX_EDGE),
        (0, 2, "7. ROUND 2 (rare)",
         "IN:  POLAR_SPLIT/WIDE_SPAN only (~10-30%)\n"
         "DO:  cross-exam: inject other 2 R1\n"
         "OUT: updated opinions (rating + conf delta)",
         GATE_FILL, GATE_EDGE),
        (1, 2, "8. DUAL SYNTHESIS",
         "IN:  final round per ticker\n"
         "DO:  rule-based blend (Neutral / Averse)\n"
         "OUT: 2 verdict per ticker (mod + reasoning)",
         OUT_FILL, OUT_EDGE),
        (2, 2, "9. PERSIST + SURFACE",
         "IN:  MultiAgentVerdict per ticker\n"
         "DO:  append cache + serve via API\n"
         "OUT: dashboard renders 4 sections",
         PERSIST_FILL, PERSIST_EDGE),
    ]
    # 3-col x 3-row grid
    card_w, card_h = 29, 22
    x0, y0 = 5, 64
    gap_x, gap_y = 1.5, 3

    for col, row, title, body, fill, edge in cards:
        x = x0 + col * (card_w + gap_x)
        y = y0 - row * (card_h + gap_y)
        info_box(ax, x, y, card_w, card_h, title, body, fill, edge,
                 title_size=10, content_size=7.5)

    # Bottom: arrows showing serial flow (left-to-right, top-to-bottom)
    # (just a small reminder)
    text_block(ax, 5, 8,
        "Flow: 1 -> 2 -> 3 -> 4 -> 5 -> 6 (-> 7 if needed) -> 8 -> 9",
        fontsize=10, color=FG, weight="bold")
    text_block(ax, 5, 4.5,
        "Steps 4 + 7 are the ONLY LLM calls (in-session, Claude Max plan). All other steps are pure-Python or server-side cache I/O.",
        fontsize=8.5, color=MUTED)


# ═════════════════════════════════════════════════════════════════
# PAGE 3 — Case A: CNQ — CONSENSUS_BUY (R1 stop, fastest path)
# ═════════════════════════════════════════════════════════════════
def draw_page3_case_cnq(fig, ax):
    page_header(ax, 3, "Case A: CNQ — CONSENSUS_BUY (R1 stop)",
                "Fastest path: 3 specialists agree, R2 skipped, full target weight in both modes",
                accent=OUT_EDGE)

    # Top: quant readings
    info_box(ax, 5, 80, 90, 9, "Step 1-2: Quant readings + BuyScore selection",
        "Composite 74.9  |  Class CONTINUATION  |  OER 0 (PRISTINE)  |  TCS 100/85  |  RSS 74/74  |  Pre-Mom 0.8 STRONG\n"
        "BuyScore = 95.86  ->  Stock LONG Top 1 / 5\n"
        "Comment: OER 0 at Composite >70 occurs in <3% of universe -> rare 'trend with max room' setup.",
        DATA_FILL, DATA_EDGE, title_size=10, content_size=8.5)

    # Round 1 — 3 specialist columns
    section_h2(ax, 5, 75, "Step 3-5: Round 1 Parallel (3 sub-agent calls, ~30s)", color=LLM_EDGE)

    # 3 columns
    cols = [
        ("FUNDAMENTAL", "BUY  conf 0.80", "#16a34a",
         "60+ yr reserve life moat\n"
         "TMX structural -$4 differential lift\n"
         "100% FCF return policy\n"
         "$30-35/bbl breakeven among lowest globally\n"
         "DCF +15-20% upside vs current"),
        ("SENTIMENT",   "BUY  conf 0.70", "#16a34a",
         "TMX 12m structural lift recognized\n"
         "Q1 in-line + 7% dividend hike\n"
         "Berkshire 6% stake unchanged (sticky)\n"
         "Carney govt energy-supportive\n"
         "TMX2 rumor as optionality"),
        ("VALUATION",   "BUY  conf 0.85", "#16a34a",
         "Composite 74.9 top decile\n"
         "OER 0 PRISTINE (very rare)\n"
         "TCS 100/85 max-strength trend\n"
         "RSS 74/74 sector leadership\n"
         "Distribution days = 0"),
    ]
    for i, (name, verdict, color, bullets) in enumerate(cols):
        x = 5 + i * 30
        # Specialist header
        box(ax, x, 67, 27, 5, f"{name}\n{verdict}", "#ffffff", color, fontsize=10, weight="bold")
        # Bullets box
        info_box(ax, x, 50, 27, 16, "", bullets, "#f0fdf4", color,
                 title_size=1, content_size=7)

    # Convergence
    section_h2(ax, 5, 45, "Step 6: Convergence Check", color=CTX_EDGE)
    info_box(ax, 5, 36, 90, 8, "Result: CONSENSUS_BUY (1 tier, dispersion 0.06)",
        "All 3 specialists rate BUY with confidence 0.70-0.85.  rating_axis = 0.  Convergence detector returns TRUE.\n"
        "Round 2 skipped — synthesis proceeds directly. converged_round = 1.",
        CTX_FILL, CTX_EDGE, title_size=10, content_size=8.5)

    # Synthesis
    section_h2(ax, 5, 32, "Step 8: Dual Synthesis (rule-based, no LLM)", color=OUT_EDGE)

    # Two synthesis boxes side-by-side
    info_box(ax, 5, 17, 43, 14, "NEUTRAL mode: BUY +3",
        "Confidence-weighted ensemble: 0.80+0.70+0.85 = 2.35 all-BUY.\n\n"
        "Reasoning: Triple BUY consensus with the rarest possible OER 0 + Composite 75.\n"
        "TMX structural lift + Berkshire anchor + dividend coverage.\n\n"
        "SIZING: Full target weight. Add on any 5-8% pullback to SMA50.",
        "#ffffff", BUY_COLOR, title_size=10, content_size=7.5)
    info_box(ax, 52, 17, 43, 14, "AVERSE mode: BUY +3  (same!)",
        "Risk-language weighting does NOT change verdict because:\n"
        "- OER 0 = zero tactical drawdown risk\n"
        "- 5.2% dividend yield = downside protection\n"
        "- Berkshire anchor reduces forced-seller risk\n\n"
        "SIZING: Full target weight.  Averse-default (gap = 0).",
        "#ffffff", BUY_COLOR, title_size=10, content_size=7.5)

    # Footer
    info_box(ax, 5, 7, 90, 8, "Step 9: Persist + Surface",
        "Cache entry: { ticker: CNQ, side: long, synthesis_neutral: BUY+3, synthesis_averse: BUY+3, "
        "converged_round: 1, disagreement: {type: CONSENSUS_BUY, rating_axis: 0, dispersion: 0.06} }\n"
        "Dashboard: Stock LONG section, row 1, both Neutral and Averse cells green BUY +3, Gap badge green (0), R = 1.",
        PERSIST_FILL, PERSIST_EDGE, title_size=10, content_size=8)

    # Cost summary
    ax.text(50, 3, "Total: 3 sub-agent calls  |  ~30 seconds  |  ~45k tokens  |  R2 skipped",
            fontsize=9, color=MUTED, ha="center", style="italic", fontweight="bold")


# ═════════════════════════════════════════════════════════════════
# PAGE 4 — Case B: QCLN — POLAR_SPLIT (R2 cross-examination)
# ═════════════════════════════════════════════════════════════════
def draw_page4_case_qcln(fig, ax):
    page_header(ax, 4, "Case B: QCLN — POLAR_SPLIT (R2 cross-examination)",
                "Fund SELL vs Sent/Val BUY -> R2 triggered -> narrows confidences but rating spread persists -> risk-mode-dependent verdict",
                accent=GATE_EDGE)

    # Quant
    info_box(ax, 5, 86, 90, 7, "Step 1-2: Quant readings",
        "Composite 68.7  |  Class OVEREXTENDED  |  OER 58  |  HEALTHY_TREND breadth 72%  |  BuyScore 78  ->  ETF LONG candidate\n"
        "Comment: Strong technical + flow signals, but fundamental specialist must evaluate basket quality independently.",
        DATA_FILL, DATA_EDGE, title_size=10, content_size=8)

    # Round 1
    section_h2(ax, 5, 81, "Step 3-5: ROUND 1 Parallel — divergent ratings", color=LLM_EDGE)

    r1_cols = [
        ("FUNDAMENTAL", "SELL conf 0.72", SELL_COLOR,
         "ROIC 6.5% < WACC 8%\n"
         "(value-destroying basket)\n"
         "GM 32% vs SPY 41%\n"
         "22% holdings -EPS\n"
         "IRA dependence 73%"),
        ("SENTIMENT",   "BUY  conf 0.62", BUY_COLOR,
         "IRA continuity post-election\n"
         "AI-PPA demand floor\n"
         "Inflow $80M reverses 6m outflow\n"
         "Goldman OW thematic\n"
         "Reddit shift contrarian-long"),
        ("VALUATION",   "BUY  conf 0.78", BUY_COLOR,
         "Composite top 5%\n"
         "HEALTHY_TREND flag\n"
         "TFS>TCS fresh formation\n"
         "Pre-Mom 0.8 STRONG\n"
         "RSS 78/83 leadership"),
    ]
    for i, (name, verdict, color, bullets) in enumerate(r1_cols):
        x = 5 + i * 30
        box(ax, x, 73, 27, 5, f"{name}\n{verdict}", "#ffffff", color, fontsize=10, weight="bold")
        info_box(ax, x, 57, 27, 15, "", bullets, "#ffffff", color,
                 title_size=1, content_size=7)

    # Convergence check R1
    info_box(ax, 5, 49, 90, 6.5, "Step 6: Convergence -> POLAR_SPLIT (SELL ^ BUY both present)",
        "rating_axis = 2 (max).  rset = {SELL, BUY}.  Convergence detector returns FALSE.\n"
        "Trigger Round 2 cross-examination. converged_round = 0 (will remain if R2 also fails).",
        GATE_FILL, GATE_EDGE, title_size=10, content_size=8)

    # Round 2
    section_h2(ax, 5, 45, "Step 7: ROUND 2 — cross-examination (3 more sub-agent calls)", color=GATE_EDGE)

    r2_cols = [
        ("FUND  -> SELL 0.62", "#fef2f2", SELL_COLOR,
         "Acknowledges HEALTHY_TREND flag\n"
         "+ flow inflection (conf -0.10)\n"
         "Rating UNCHANGED: basket quality\n"
         "concern overrides tape signals"),
        ("SENT  -> BUY  0.57", "#f0fdf4", BUY_COLOR,
         "Acknowledges Fund's IRA-dependence\n"
         "+ ROIC<WACC tail risks (conf -0.05)\n"
         "Rating UNCHANGED but conviction reduced\n"
         "from 0.62 to 0.57"),
        ("VAL   -> BUY  0.75", "#f0fdf4", BUY_COLOR,
         "Acknowledges Fund's basket-quality lane\n"
         "Defers to that lane (conf -0.03)\n"
         "Pure technical view unchanged:\n"
         "HEALTHY_TREND + breadth still valid"),
    ]
    for i, (header, fill, color, body) in enumerate(r2_cols):
        x = 5 + i * 30
        box(ax, x, 39, 27, 4, header, fill, color, fontsize=9.5, weight="bold")
        info_box(ax, x, 27, 27, 11, "", body, fill, color,
                 title_size=1, content_size=7)

    # Convergence R2
    text_block(ax, 5, 24,
        "R2 result: still POLAR_SPLIT.  Conf deltas narrow gap but rating spread persists.  "
        "STOP at R2.  converged_round = 0 (signals 'no full consensus').",
        fontsize=8.5, color=GATE_EDGE, weight="bold")

    # Dual synthesis
    section_h2(ax, 5, 20, "Step 8: Dual Synthesis", color=OUT_EDGE)

    info_box(ax, 5, 7, 43, 12, "NEUTRAL: BUY +1  (mild)",
        "Conf-weighted ensemble: 2 BUY (0.57+0.75=1.32) > 1 SELL (0.62).\n"
        "Tech HEALTHY_TREND + sentiment flow reversal outweigh fundamental basket concern.\n"
        "+1 (not +3) due to dissent.\n\n"
        "SIZING: 50% target entry, add on 50DMA pullback or breakout.",
        "#ffffff", BUY_COLOR, title_size=10, content_size=7)
    info_box(ax, 52, 7, 43, 12, "AVERSE: SELL -2  (gap = 3!)",
        "Averse mode upweights bear language: Fund's ROIC<WACC + IRA dependence\n"
        "becomes primary driver. Subsidy concentration in 73% of basket = unacceptable\n"
        "risk under averse mandate.\n\n"
        "SIZING: Avoid wrapper. Wait for ROIC inflection OR 1Y sustained inflows.",
        "#ffffff", SELL_COLOR, title_size=10, content_size=7)

    ax.text(50, 3, "Total: 6 sub-agent calls  |  ~90 seconds  |  ~95k tokens  |  risk_mode_gap = 3 (PM risk profile decides)",
            fontsize=9, color=MUTED, ha="center", style="italic", fontweight="bold")


# ═════════════════════════════════════════════════════════════════
# PAGE 5 — Cost profile + In-session vs Server boundary
# ═════════════════════════════════════════════════════════════════
def draw_page5_cost_boundary(fig, ax):
    page_header(ax, 5, "Cost Profile + In-Session / Server Boundary",
                "All LLM work inside Claude Code session (Plan-billed) — server never calls LLM",
                accent=PERSIST_EDGE)

    # Cost table
    section_h2(ax, 5, 88, "Per-ticker cost", color=OUT_EDGE)

    # Table header
    box(ax, 5, 80, 90, 5, "", PERSIST_FILL, PERSIST_EDGE, fontsize=9)
    headers = ["Pattern", "Sub-agent calls", "Token usage", "Wall time", "Synthesis"]
    xs = [12, 30, 50, 68, 84]
    for x, h in zip(xs, headers):
        ax.text(x, 82.5, h, fontsize=10, fontweight="bold", color=PERSIST_EDGE,
                ha="center", va="center")
    # Rows
    rows = [
        ("CONSENSUS / ENTRY_TIMING (R1 stop)", "3", "~50k", "~30-40s", "rule-based (0)"),
        ("POLAR_SPLIT / WIDE_SPAN (R2)",       "6", "~95k", "1-1.5 min", "rule-based (0)"),
    ]
    for i, row in enumerate(rows):
        y = 76 - i * 5
        bg = "#f9fafb" if i % 2 == 0 else "#ffffff"
        box(ax, 5, y - 1, 90, 4, "", bg, "#e5e7eb", fontsize=9)
        for x, cell in zip(xs, row):
            ax.text(x, y + 0.8, cell, fontsize=9, color=FG, ha="center", va="center")

    # Full cycle cost
    info_box(ax, 5, 56, 90, 11, "20-ticker Tier-A full cycle (assume 70% R1 only / 30% R2)",
        "14 tickers x 3 calls (R1)   = 42 calls\n"
        " 6 tickers x 6 calls (R1+R2) = 36 calls\n"
        "------------------------------------\n"
        "TOTAL: ~78 sub-agent calls  |  ~5-7 min wall time  |  ~1.2M tokens\n"
        "All billed to Claude Max plan — no external API spend.",
        OUT_FILL, OUT_EDGE, title_size=10, content_size=9)

    # In-session vs Server boundary
    section_h2(ax, 5, 52, "Responsibility Boundary", color=LLM_EDGE)

    # Two columns
    info_box(ax, 5, 28, 43, 22,
        "IN-SESSION  (Claude Code, Plan-billed)",
        "Step 3: build prompts (specialist_prompts.py)\n"
        "Step 4: ROUND 1 — 3 parallel sub-agent calls\n"
        "        (Agent tool, market-researcher)\n"
        "Step 5: parse free-text -> SpecialistOpinion\n"
        "Step 7: ROUND 2 cross-exam (if needed)\n"
        "Step 8: rule-based synthesis (Neutral/Averse)\n"
        "Step 9a: cache write (.json)\n\n"
        "Trigger:\n"
        "  - Manual:    'run today multi-agent debate'\n"
        "  - Scheduled: CronCreate (fresh session)",
        LLM_FILL, LLM_EDGE, title_size=10, content_size=8)

    info_box(ax, 52, 28, 43, 22,
        "SERVER  (uvicorn, LLM call ZERO)",
        "Step 1: data refresh (price_discovery.py)\n"
        "Step 2: SELECTION — BuyScore auto-compute\n"
        "        (top_buy_picks / top_sell_picks)\n"
        "Step 9b: /api/conviction-debate/multi\n"
        "         - selection live (fresh per request)\n"
        "         - cache lookup\n"
        "         - merge -> response\n"
        "UI: React dashboard renders 4 sections\n\n"
        "Trigger:\n"
        "  - Continuous (always-on uvicorn)\n"
        "  - Auto-refresh on /api/reload",
        PERSIST_FILL, PERSIST_EDGE, title_size=10, content_size=8)

    # Bottom: Key invariants
    section_h2(ax, 5, 23, "Key Design Invariants", color=FG)

    invariants = [
        "1. LLM calls ONLY in-session. Server (uvicorn) NEVER calls an LLM. -> guarantees Plan-only spending.",
        "2. Selection re-computed every endpoint hit -> live scan immediately reflected in dashboard.",
        "3. Verdict cache immutable. New verdict overwrites only on next in-session run. -> reproducibility.",
        "4. Convergence-driven rounds. R2 only when needed -> cost discipline (~70% of tickers exit at R1).",
        "5. Dual synthesis (Neutral + Averse) -> single source serves all PM risk profiles.",
        "6. Lane enforcement -> prevents specialist domain creep; preserves intentional disagreement signal.",
    ]
    for i, line in enumerate(invariants):
        ax.text(5, 18 - i * 2.5, line, fontsize=9, color=FG, va="top", linespacing=1.3, wrap=True)


# ═════════════════════════════════════════════════════════════════
# Driver
# ═════════════════════════════════════════════════════════════════
def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, "conviction_debate_flow.pdf")

    with PdfPages(out_pdf) as pdf:
        for page_fn in [
            draw_page1_overview,
            draw_page2_step_detail,
            draw_page3_case_cnq,
            draw_page4_case_qcln,
            draw_page5_cost_boundary,
        ]:
            fig, ax = setup_page()
            page_fn(fig, ax)
            pdf.savefig(fig, bbox_inches="tight", facecolor=BG)
            plt.close(fig)

    print(f"Wrote: {out_pdf}")
    print(f"Pages: 5")
    sz_kb = os.path.getsize(out_pdf) / 1024
    print(f"Size: {sz_kb:.1f} KB")


if __name__ == "__main__":
    main()
