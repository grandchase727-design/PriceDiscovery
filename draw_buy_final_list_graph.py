"""
draw_buy_final_list_graph.py — 매수 Final List 선정 Dependency Graph (3-page PDF)
Output: reports/buy_final_list_graph.pdf

매수 final list가 어떻게 선정되는지의 전체 의존 흐름을 시각화:
  Page 1 — End-to-end pipeline (Scan → API Gate → Swarm → Final List)
  Page 2 — Swarm Phase 0~6 (PM picks 생성 경로)
  Page 3 — build_final_lists() 내부 (tier/category/cap/commentary 조립)

2026-07 현재 코드 기준 (institutional QVR + Financials override + MR tier +
HOLDING 주입 + 통합 commentary + FICC bucket 정규화 반영).
"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ── Korean font (macOS AppleSDGothicNeo) ──
for _fp in ("/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/NanumGothic.ttf",
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf"):
    if os.path.exists(_fp):
        try:
            fm.fontManager.addfont(_fp)
            plt.rcParams["font.family"] = fm.FontProperties(fname=_fp).get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False

# ── Palette ──
BG, FG, MUTED, LINE = "#ffffff", "#1f2937", "#6b7280", "#9ca3af"
SCAN_F, SCAN_E   = "#eff6ff", "#2563eb"   # blue — scan/scoring
GATE_F, GATE_E   = "#fef2f2", "#dc2626"   # red — eligibility gate
API_F,  API_E    = "#ecfeff", "#0891b2"   # cyan — api post-load
LLM_F,  LLM_E    = "#fdf2f8", "#db2777"   # pink — swarm LLM
COMP_F, COMP_E   = "#fff7ed", "#f97316"   # orange — composer / deterministic
OUT_F,  OUT_E    = "#f0fdf4", "#16a34a"   # green — final output
TIER_F, TIER_E   = "#fefce8", "#ca8a04"   # yellow — tiers/categories
SLATE_F, SLATE_E = "#f1f5f9", "#475569"   # slate — cache/persist


def box(ax, x, y, w, h, text, fill, edge, fs=8.5, weight="normal", tcolor=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.4",
                 facecolor=fill, edgecolor=edge, linewidth=1.1, zorder=2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color=tcolor or FG, zorder=3, linespacing=1.2)


def arrow(ax, x1, y1, x2, y2, color=LINE, lw=1.1, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=11, color=color, linewidth=lw, alpha=0.85, zorder=1))


def setup(figsize=(13, 18)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    return fig, ax


def header(ax, page, title, subtitle, accent=GATE_E):
    ax.add_patch(FancyBboxPatch((0, 95.4), 100, 4, boxstyle="square,pad=0",
                 facecolor=accent, edgecolor="none", zorder=1))
    ax.text(2, 97.4, f"PAGE {page}", fontsize=9, fontweight="bold", color="white", va="center")
    ax.text(50, 97.4, title, fontsize=15, fontweight="bold", color="white", va="center", ha="center")
    if subtitle:
        ax.text(50, 93.2, subtitle, fontsize=9.5, color=MUTED, va="center", ha="center", style="italic")


def caption(ax, x, y, text, fs=7.0, color=MUTED, ha="left"):
    ax.text(x, y, text, fontsize=fs, color=color, va="center", ha=ha, linespacing=1.3)


def legend(ax, items, x=2, y=2.5):
    cx = x
    for label, fill, edge in items:
        ax.add_patch(FancyBboxPatch((cx, y), 2.4, 2.0, boxstyle="round,pad=0.02,rounding_size=0.3",
                     facecolor=fill, edgecolor=edge, linewidth=1.0))
        ax.text(cx + 3.0, y + 1.0, label, fontsize=7.0, color=FG, va="center", ha="left")
        cx += 3.0 + len(label) * 0.95 + 2.5


# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — End-to-end pipeline
# ═══════════════════════════════════════════════════════════════════
def page1(pdf):
    fig, ax = setup()
    header(ax, 1, "매수 Final List 선정 — End-to-End Pipeline",
           "Scan → API Eligibility → Swarm (Phase 0~6) → build_final_lists() → 매수 Final List", GATE_E)

    # Stage 1 — SCAN
    box(ax, 30, 87, 40, 4.2, "① SCAN — price_discovery.py  (830 tickers)", SCAN_F, SCAN_E, 10, "bold")
    box(ax, 6, 80.5, 26, 4.5,
        "Composite (0-100)\n0.30·TCS + 0.25·TFS_resid\n+ 0.30·RSS_hybrid + 0.15·URS\n- 0.10·max(0, OER-40)", SCAN_F, SCAN_E, 7.0)
    box(ax, 37, 80.5, 26, 4.5,
        "3×3 Classification\nCONTINUATION / FORMATION /\nPULLBACK / NEUTRAL / DOWNTREND …\n(Sticky FLAT hysteresis)", SCAN_F, SCAN_E, 7.0)
    box(ax, 68, 80.5, 26, 4.5,
        "Pass 4 — MR sub-signals\n(OU·Idio·Stab·LT·Stretch)\n→ mr_score (별도 tier)\n* 이번 세션 추가", SCAN_F, SCAN_E, 7.0)
    arrow(ax, 50, 87, 19, 85); arrow(ax, 50, 87, 50, 85); arrow(ax, 50, 87, 81, 85)

    # Stage 2 — API post-load
    box(ax, 28, 72, 44, 4.2, "② API POST-LOAD — api.py  _load_cache()", API_F, API_E, 10, "bold")
    arrow(ax, 50, 80.5, 50, 76.2, SCAN_E)
    box(ax, 5, 64.5, 28, 5.5,
        "QVR (institutional)\nQ 0.45 / V 0.25 / R 0.30\n* Financials 전용 팩터\n(ROE·ROA·장부가 / P-B·P-E·배당)", API_F, API_E, 7.0)
    box(ax, 36, 64.5, 28, 5.5,
        "ELIGIBILITY GATE\n① Comp≥55  ② bullish 분류\n③ ADV≥$5M  ④ ETF or QVR≥40\n→ eligible = True/False", GATE_F, GATE_E, 7.2, "bold")
    box(ax, 67, 64.5, 28, 5.5,
        "Sidecar tiers (병렬)\n• Anti-Lag PROVISIONAL\n• Sector-Segmented top-5\n* Mean Reversion (OER 거울상)", TIER_F, TIER_E, 7.0)
    for tx in (19, 50, 81):
        arrow(ax, 50, 72, tx, 70, API_E)
    caption(ax, 50, 62.2, "eligible=True 종목만 매수 후보 pool로 → Swarm action_selector 입력", 7.5, GATE_E, "center")

    # Stage 3 — SWARM
    box(ax, 26, 56, 48, 4.2, "③ SWARM — market_leaders_swarm.py  (Phase 0~6, 상세 Page 2)", LLM_F, LLM_E, 10, "bold")
    arrow(ax, 50, 64.5, 50, 60.2, GATE_E)
    box(ax, 8, 49.5, 38, 5.0,
        "Phase 4 action_selector → PM 후보 pool\nPhase 5 PM Agent + per-ticker debate (R1+R2)\n* HOLDING 종목 강제 주입", LLM_F, LLM_E, 7.2)
    box(ax, 54, 49.5, 38, 5.0,
        "Phase 5b portfolio_composer (sizing)\nPhase 5.6 position_state machine\nPhase 5.6b exit_debate (청산 LLM)", COMP_F, COMP_E, 7.2)
    arrow(ax, 40, 56, 27, 54.5, LLM_E); arrow(ax, 60, 56, 73, 54.5, COMP_E)
    box(ax, 30, 43.5, 40, 4.0, "swarm cache → phase5_pm.horizons.core (debate_synthesis)", SLATE_F, SLATE_E, 8.0)
    arrow(ax, 27, 49.5, 45, 47.5, LLM_E); arrow(ax, 73, 49.5, 55, 47.5, COMP_E)

    # Stage 4 — build_final_lists
    box(ax, 28, 36.5, 44, 4.2, "④ FINAL LIST — final_list.py  build_final_lists()  (상세 Page 3)", OUT_F, OUT_E, 10, "bold")
    arrow(ax, 50, 43.5, 50, 40.7, SLATE_E)
    box(ax, 6, 29, 27, 5.5,
        "각 PM pick →\n_eval_buy_pick_with_debate\n(debate 있으면 synthesis,\n없으면 결정론 3-agent votes)", OUT_F, OUT_E, 7.0)
    box(ax, 36.5, 29, 27, 5.5,
        "buy_dedup → 카테고리\nENTERED / NEW /\nHOLDING / EXIT_PENDING\n+ holdings-aware re-rank", TIER_F, TIER_E, 7.0)
    box(ax, 67, 29, 27, 5.5,
        "Turnover Cap\nStock 20 + ETF 20\n(quality = stars×100\n+ composite + days_held)", COMP_F, COMP_E, 7.0)
    for tx in (19, 50, 80.5):
        arrow(ax, 50, 36.5, tx, 34.5, OUT_E)

    # Final output
    box(ax, 22, 20, 56, 5.2,
        "* 매수 FINAL LIST  (buy_list + active_positions + exit_pending)\n+ 통합 Debate Commentary (PM·Trading·Risk) + Executive Commentary",
        OUT_F, OUT_E, 9.5, "bold", OUT_E)
    arrow(ax, 50, 29, 50, 25.2, OUT_E, 1.5)
    box(ax, 30, 13.5, 40, 4.0, "React Dashboard — Final List (통합) 테이블 + 종목별 commentary 서브행", SLATE_F, SLATE_E, 8.0)
    arrow(ax, 50, 20, 50, 17.5, OUT_E)

    legend(ax, [("Scan/Score", SCAN_F, SCAN_E), ("Gate", GATE_F, GATE_E),
                ("API tiers", API_F, API_E), ("Swarm LLM", LLM_F, LLM_E),
                ("Compose", COMP_F, COMP_E), ("Final", OUT_F, OUT_E)])
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — Swarm Phase 0~6
# ═══════════════════════════════════════════════════════════════════
def page2(pdf):
    fig, ax = setup()
    header(ax, 2, "Swarm Phase 0~6 — PM Picks 생성 경로",
           "각 Phase가 어떻게 매수 후보 pool을 좁혀가는가 (market_leaders_swarm.py)", LLM_E)

    steps = [
        (88, "Phase 0 — fact_collector", "* 제거됨 (news_narrative_analyst로 통합) — Phase 0 hang 해소", SLATE_F, SLATE_E),
        (80.5, "Phase 1 — 5 Analysts", "unified_analyst(4섹션, scan데이터) ∥ news_narrative(WebSearch) 병렬\nMacro·Cross-Asset·Sector·Flow·News → 각 rating + confidence", LLM_F, LLM_E),
        (71.5, "Phase 2-3 — Strategist + Synthesis", "5 analyst 통합 → regime tag (예: DEFENSIVE_TRANSITIONAL)\nneutral + averse 시나리오 합성", LLM_F, LLM_E),
        (62.5, "Phase 4 — action_selector", "eligible 후보 pool → LONG stocks/ETFs 후보 35개씩 → top picks 선별\n(_is_etf: category가 STK_ 아니면 ETF)", COMP_F, COMP_E),
        (52.5, "Phase 5 — PM Agent + per-ticker Debate", "PM Agent core horizon picks → per-ticker debate R1+R2\nTrading·Risk·Critic 3-agent 토론 → tier (UNANIMOUS/MAJORITY/SOLO)\n* HOLDING 종목 강제 주입 (action_selector 누락분 재평가)\n* DEBATE_WALL_BUDGET 1200s (hang 방어)", LLM_F, LLM_E),
        (41, "Phase 5b — portfolio_composer", "regime-adaptive budget → active picks sizing (full/half/cap)\nsector top-3 분산", COMP_F, COMP_E),
        (33, "Phase 5.6 — position_state machine", "PROSPECTING→ENTERED→HOLDING→EXIT_PENDING→EXITED\n(2일 confirmation hysteresis)", COMP_F, COMP_E),
        (25, "Phase 5.6b — exit_debate", "EXIT_PENDING 종목 → 청산 LLM 토론\n(EXIT_NOW / PARTIAL_EXIT / HOLD_1W)", GATE_F, GATE_E),
    ]
    for i, (y, title, desc, fill, edge) in enumerate(steps):
        h = 5.5 if desc.count("\n") >= 2 else (4.6 if "\n" in desc else 3.6)
        box(ax, 14, y, 72, h, "", fill, edge, 8)
        ax.text(16, y + h - 1.3, title, fontsize=9.5, fontweight="bold", color=edge, va="top", ha="left")
        ax.text(16, y + h - 3.4, desc, fontsize=7.2, color=FG, va="top", ha="left", linespacing=1.35)
        if i < len(steps) - 1:
            arrow(ax, 50, y, 50, steps[i+1][0] + (5.5 if steps[i+1][2].count("\n") >= 2 else 4.6 if "\n" in steps[i+1][2] else 3.6), edge)

    box(ax, 22, 17, 56, 4.4, "→ swarm cache: phase5_pm.horizons.core.{long_stocks, long_etfs}\n각 pick에 debate_synthesis {tier, stars, final_decision, transcript}",
        OUT_F, OUT_E, 8.0, "bold")
    arrow(ax, 50, 25, 50, 21.4, GATE_E, 1.4)
    caption(ax, 50, 13.5, "이 cache가 build_final_lists()의 핵심 입력 (Page 3)", 8.0, OUT_E, "center")
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — build_final_lists() 내부
# ═══════════════════════════════════════════════════════════════════
def page3(pdf):
    fig, ax = setup()
    header(ax, 3, "build_final_lists() — 매수 리스트 조립",
           "swarm cache + scan + position_state → 카테고리 분류 + cap + 통합 commentary", OUT_E)

    # Sources
    box(ax, 8, 88, 84, 3.8, "_load_sources() — scan_cache + swarm_cache + .position_state.json + price_data + QVR", SLATE_F, SLATE_E, 8.5, "bold")

    # Per-pick eval
    box(ax, 6, 79, 40, 6.5,
        "for each PM pick (core horizon):\n_eval_buy_pick_with_debate()\n• debate_synthesis 있으면 → tier/stars/final_decision\n• 없으면 → _eval_buy_pick (결정론 3-agent votes)\n• bucket 정규화 (FICC ETF → long_etfs 교정 *)",
        OUT_F, OUT_E, 7.0)
    box(ax, 54, 79, 40, 6.5,
        "3-Agent Votes:\n• PM vote (composite + classification)\n• Trading vote (entry_signal)\n• Risk vote (5-차원 risk score)\n→ consensus tier + stars (0~3)",
        TIER_F, TIER_E, 7.0)
    arrow(ax, 50, 88, 26, 85.5, SLATE_E); arrow(ax, 50, 88, 74, 85.5, SLATE_E)
    arrow(ax, 46, 82.2, 54, 82.2, OUT_E)

    # buy_dedup + categories
    box(ax, 30, 71, 40, 3.8, "buy_dedup — 중복 제거 (ticker별 best)", OUT_F, OUT_E, 8.5, "bold")
    arrow(ax, 26, 79, 45, 74.8, OUT_E); arrow(ax, 74, 79, 55, 74.8, TIER_E)

    cats = [
        (6,  "[ENTERED]", "state=ENTERED\n+ EXECUTE_TODAY\n오늘 진입", "#16a34a"),
        (30, "[NEW]", "오늘 voting 통과\n미보유 신규후보", "#0A7D3F"),
        (54, "[HOLDING]", "_build_active_positions\n보유 중 (sticky)\n* HELD 종목도 debate", "#0891b2"),
        (78, "[EXIT_PENDING]", "_build_exit_pending\nSKIP×2/WAIT×5 또는\nRegime-Flip → 청산", "#dc2626"),
    ]
    for x, title, desc, col in cats:
        box(ax, x, 61, 21, 7.5, "", "#ffffff", col, 8)
        ax.text(x + 10.5, 67.3, title, fontsize=8.5, fontweight="bold", color=col, va="center", ha="center")
        ax.text(x + 10.5, 63.5, desc, fontsize=6.5, color=FG, va="center", ha="center", linespacing=1.3)
        arrow(ax, 50, 71, x + 10.5, 68.5, OUT_E)

    # Holdings-aware re-rank + cap
    box(ax, 8, 52, 40, 5.5,
        "Holdings-aware re-rank (NEW)\n• sector 집중도 페널티\n• 보유종목 상관관계 페널티\n→ 분산 기여 종목 우선",
        TIER_F, TIER_E, 7.2)
    box(ax, 52, 52, 40, 5.5,
        "Turnover Cap (asset-class 균형)\nStock 20 + ETF 20\nquality = stars×100 + composite\n+ days_held×0.1  (EXIT_PENDING bypass)",
        COMP_F, COMP_E, 7.2)
    arrow(ax, 16, 61, 20, 57.5, OUT_E); arrow(ax, 84, 61, 80, 57.5, OUT_E)

    # Commentary
    box(ax, 14, 43, 72, 5.5,
        "* 통합 Debate Commentary (_unified_commentary)\ndebate synthesis + PM + Trading + Risk reason → 1개 코멘터리\n+ Executive Commentary (12,000자, 백그라운드 LLM, 캐시)",
        LLM_F, LLM_E, 7.5, "bold")
    arrow(ax, 34, 52, 42, 48.5, TIER_E); arrow(ax, 70, 52, 60, 48.5, COMP_E)

    # Final
    box(ax, 24, 33, 52, 5.5,
        "* 매수 FINAL LIST\nbuy_list + active_positions + exit_pending\nitems_by_category (ENTERED/NEW/HOLDING/EXIT_PENDING)",
        OUT_F, OUT_E, 9.0, "bold", OUT_E)
    arrow(ax, 50, 43, 50, 38.5, LLM_E, 1.5)

    box(ax, 28, 25, 44, 4.0, "/api/final-list → React Final List (통합) 테이블", SLATE_F, SLATE_E, 8.5)
    arrow(ax, 50, 33, 50, 29, OUT_E)

    caption(ax, 50, 20,
            "핵심 게이트: ① Eligibility(Comp≥55+QVR) → ② Swarm 3-agent 합의 → ③ 카테고리 → ④ 20+20 cap → ⑤ 통합 commentary",
            7.5, MUTED, "center")
    pdf.savefig(fig, facecolor=BG); plt.close(fig)


def main():
    os.makedirs("reports", exist_ok=True)
    out = "reports/buy_final_list_graph.pdf"
    with PdfPages(out) as pdf:
        page1(pdf); page2(pdf); page3(pdf)
    print(f"✓ Saved {out}  (3 pages)")


if __name__ == "__main__":
    main()
