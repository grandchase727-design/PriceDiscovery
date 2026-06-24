"""Streamlit Dashboard — Market Commentary tab port.

Mirrors frontend/src/components/tabs/MarketCommentaryTab.tsx 1:1 with:
- Theme toggle (dark/light)
- 23 analytical sections
- Market Leaders, Executive Summary, Conviction Picks widgets
- Multi-Agent Swarm analysis panel
- Trading Layer Backtest panel
- Multi-Agent Conviction Debate panel

All data fetched from the existing FastAPI backend (localhost:8000).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add lib/ to path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from lib.api import (
    clear_caches, fetch_classification_history, fetch_market_regime, fetch_meta,
    fetch_quant_strategies, fetch_table, fetch_validation,
)
from lib.backtest import render_backtest_panel
from lib.compute_stats import compute_quant_stats, compute_regime_stats, compute_stats, compute_validation_stats
from lib.conviction_debate import render_conviction_debate
from lib.sections import (
    section_1, section_2, section_3, section_4, section_5, section_6,
    section_7, section_8, section_9, section_10, section_11,
    section_12, section_13, section_14, section_15, section_16,
    section_17, section_18, section_19, section_20, section_21, section_22,
)
from lib.swarm import render_swarm_panel
from lib.theme import get_theme, inject_css, theme_toggle_widget
from lib.widgets import (
    render_conviction_picks, render_executive_summary, render_market_leaders,
)


# ─── Page config ────────────────────────────────────
st.set_page_config(
    page_title="Price Discovery — Market Commentary",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Sidebar ────────────────────────────────────────
def sidebar() -> dict:
    st.sidebar.markdown("# 📊 Price Discovery")
    st.sidebar.caption("Streamlit Edition — Market Commentary")

    # Theme toggle
    theme_toggle_widget()

    st.sidebar.markdown("---")

    # Meta info
    try:
        meta = fetch_meta()
        scan_time = meta.get("scan_time", "—")
        total_tickers = meta.get("total_tickers", 0)
        st.sidebar.markdown(f"**Universe:** {total_tickers} tickers")
        st.sidebar.markdown(f"**Scan:** `{scan_time[:16] if scan_time else '—'}`")
    except Exception as e:
        st.sidebar.error(f"Meta API error: {e}")
        meta = {}

    st.sidebar.markdown("---")

    # Filters
    st.sidebar.markdown("### 🔎 Filters")

    sectors_list = sorted(meta.get("sectors", []))
    selected_sectors = st.sidebar.multiselect(
        "Sectors", sectors_list, default=sectors_list,
    )
    eligible_only = st.sidebar.checkbox("Eligible only (Comp ≥55)", value=False)
    comp_range = st.sidebar.slider("Composite Range", 0, 100, (0, 100))

    st.sidebar.markdown("---")

    # Cache controls
    if st.sidebar.button("🔄 Refresh All Data"):
        clear_caches()
        st.rerun()

    st.sidebar.caption("Backend: localhost:8000")

    return {
        "sectors": tuple(selected_sectors) if selected_sectors else None,
        "eligible_only": eligible_only,
        "comp_min": comp_range[0],
        "comp_max": comp_range[1],
    }


# ─── Main ───────────────────────────────────────────
def main() -> None:
    inject_css()
    filters = sidebar()
    C = get_theme()

    st.markdown(
        f'<h1 style="color:{C["cyan"]};margin-bottom:4px">📊 Market Commentary</h1>'
        f'<p style="color:{C["gray"]};font-size:12px;margin-top:0">'
        f'다층적 시장 분석 · 23개 분석 섹션 + 컨빅션 픽 + 멀티에이전트 디베이트'
        f'</p>',
        unsafe_allow_html=True,
    )

    # Fetch data
    with st.spinner("Loading data…"):
        table_data = fetch_table(
            sectors=filters["sectors"],
            eligible_only=filters["eligible_only"],
            comp_min=filters["comp_min"],
            comp_max=filters["comp_max"],
        )
        all_results = table_data.get("data", []) if isinstance(table_data, dict) else []

        class_history = fetch_classification_history()
        regime_data = fetch_market_regime(sectors=filters["sectors"])
        validation_data = fetch_validation()
        quant_data = fetch_quant_strategies()

    if not all_results:
        st.warning("⚠ 데이터가 없습니다. 백엔드 (`localhost:8000`)가 실행 중이고 스캔이 완료됐는지 확인하세요.")
        return

    # Compute stats
    stats = compute_stats(class_history, all_results)
    regime_stats = compute_regime_stats(regime_data)
    val_stats = compute_validation_stats(validation_data)
    quant_stats = compute_quant_stats(quant_data)

    # Render widgets
    render_executive_summary(stats, regime_stats, val_stats, quant_stats)
    render_market_leaders(stats, regime_stats, quant_stats, all_results)
    render_conviction_picks(all_results, quant_stats)

    # ─── Tabs for major report groups ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Commentary Report (§1-§22)",
        "🤖 Multi-Agent Swarm",
        "🎭 Conviction Debate",
        "📊 Backtest",
        "📦 Raw Data",
    ])

    with tab1:
        # §1-§11: Commentary classification analysis
        st.markdown(f'<h2 style="color:{C["cyan"]}">📝 Classification & Breadth Analysis</h2>',
                    unsafe_allow_html=True)
        section_1(stats)
        section_2(stats)
        section_3(stats)
        section_4(stats)
        section_5(stats)
        section_6(stats)
        section_7(stats)
        section_8(stats)
        section_9(stats)
        section_10(stats)
        section_11(stats)

        # §12-§16: Regime
        st.markdown(f'<h2 style="color:{C["cyan"]};margin-top:24px">🌡 Market Regime Context</h2>',
                    unsafe_allow_html=True)
        section_12(regime_stats)
        section_13(regime_stats)
        section_14(regime_stats)
        section_15(regime_stats)
        section_16(regime_stats)

        # §17-§19: Validation
        st.markdown(f'<h2 style="color:{C["cyan"]};margin-top:24px">✅ Validation</h2>',
                    unsafe_allow_html=True)
        section_17(val_stats)
        section_18(val_stats)
        section_19(val_stats)

        # §20-§22: Quant
        st.markdown(f'<h2 style="color:{C["cyan"]};margin-top:24px">🔬 Quant Strategy</h2>',
                    unsafe_allow_html=True)
        section_20(quant_stats)
        section_21(quant_stats)
        section_22(quant_stats)

    with tab2:
        render_swarm_panel()

    with tab3:
        render_conviction_debate()

    with tab4:
        render_backtest_panel()

    with tab5:
        st.markdown(f'<h2 style="color:{C["cyan"]}">📦 Raw Universe Data</h2>',
                    unsafe_allow_html=True)
        import pandas as pd
        df = pd.DataFrame(all_results)
        st.caption(f"Total: {len(df)} tickers (filtered)")
        st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
