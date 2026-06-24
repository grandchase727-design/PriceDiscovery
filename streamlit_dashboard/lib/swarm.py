"""SwarmAnalysis — 6-Agent Market Leaders swarm port."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from .api import fetch_swarm_result, fetch_swarm_status, fetch_pm_history_summary
from .theme import get_theme
from .utils import badge, colored_pct, regime_color


def render_swarm_panel() -> None:
    C = get_theme()
    st.markdown(f'<h2 style="color:{C["purple"]}">🤖 Multi-Agent Swarm Analysis</h2>',
                unsafe_allow_html=True)

    # Fetch swarm status + result
    status = fetch_swarm_status()
    result = fetch_swarm_result()

    if status.get("running"):
        phase = status.get("phase", "?")
        cur = (status.get("current") or "")[:60]
        st.info(f"🤖 Swarm 실행 중 — Phase: **{phase}** · {cur}")
        st.warning("⚠ Swarm 진행 상태는 사이드바 Run Live Scan으로 트리거됩니다 (이 페이지는 결과 표시만).")
        if not result:
            return

    if not result:
        st.warning("📭 Swarm 결과 캐시 없음 — 사이드바 Run Live Scan으로 실행 후 다시 확인하세요.")
        return

    # ─── Header ────────────────────────────────────────
    gen_at = result.get("generated_at", "")
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(f"**Generated:** `{gen_at}`")
    with cols[1]:
        if st.button("🔄 Refresh", key="swarm_refresh"):
            fetch_swarm_result.clear()
            fetch_swarm_status.clear()
            st.rerun()

    # ─── PM History Banner ────────────────────────────
    try:
        pm_hist = fetch_pm_history_summary()
        if pm_hist and pm_hist.get("snapshots"):
            with st.expander("📜 PM History Summary", expanded=False):
                st.write(f"**Total snapshots:** {pm_hist.get('total_snapshots', 0)}")
                st.write(f"**Date range:** {pm_hist.get('date_range', '?')}")
                most_picked = pm_hist.get("most_picked", [])[:10]
                if most_picked:
                    st.markdown("**Most Picked Tickers:**")
                    df = pd.DataFrame(most_picked)
                    st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        pass

    # ─── Phase 1: 5 domain analysts ───────────────────
    st.markdown(f'<h3 style="color:{C["purple"]};margin-top:14px">🔬 Phase 1 — Domain Analysts</h3>',
                unsafe_allow_html=True)
    phase1 = result.get("phase1", {})
    for name, label in [
        ("macro_analyst", "📊 Macro"),
        ("cross_asset_analyst", "📈 Cross-Asset"),
        ("sector_theme_analyst", "🎯 Sector/Theme"),
        ("flow_momentum_analyst", "💧 Flow"),
        ("news_narrative_analyst", "📰 News"),
    ]:
        agent = phase1.get(name, {})
        if not agent:
            continue
        with st.expander(f"{label} — {agent.get('rating', '?')} (conf {agent.get('confidence', 0)})", expanded=False):
            st.write(f"**Narrative:** {agent.get('narrative', '')}")
            sigs = agent.get("key_signals", [])
            if sigs:
                st.markdown("**Key Signals:**")
                for s in sigs:
                    st.markdown(f"- {s}")
            risk = agent.get("biggest_risk", "")
            opp = agent.get("biggest_opportunity", "")
            if risk:
                st.markdown(f"🛑 **Biggest Risk:** {risk}")
            if opp:
                st.markdown(f"⭐ **Biggest Opportunity:** {opp}")
            ws_q = agent.get("websearch_queries", [])
            ws_r = agent.get("websearch_results", [])
            if ws_q:
                st.markdown(f"**WebSearch queries:** {len(ws_q)}")
            if ws_r:
                with st.expander(f"WebSearch results ({len(ws_r)})", expanded=False):
                    for r in ws_r:
                        if isinstance(r, dict):
                            st.markdown(f"- [{r.get('query','?')}]({r.get('url','#')})")
                            st.caption(r.get("snippet", "")[:300])
                        else:
                            st.markdown(f"- {r}")

    # ─── Phase 2: Coherence ───────────────────────────
    st.markdown(f'<h3 style="color:{C["purple"]};margin-top:14px">⚖️ Phase 2 — Coherence Debate</h3>',
                unsafe_allow_html=True)
    p2 = result.get("phase2", {})
    coherent = p2.get("coherent")
    col2 = C["green"] if coherent else C["amber"]
    st.markdown(f'<div class="pd-card" style="border:2px solid {col2}80">'
                f'<b>Coherent:</b> {coherent} · '
                f'<b>Dominant:</b> {p2.get("dominant_signal", "—")[:200]}'
                f'</div>', unsafe_allow_html=True)
    contested = p2.get("contested_areas", [])
    if contested:
        with st.expander(f"Contested Areas ({len(contested)})", expanded=False):
            for c in contested:
                st.markdown(f"- {c}")

    # ─── Phase 3: Synthesis (neutral + averse) ────────
    st.markdown(f'<h3 style="color:{C["purple"]};margin-top:14px">🎭 Phase 3 — Dual Synthesis</h3>',
                unsafe_allow_html=True)
    mode = st.radio("Synthesis mode", ["neutral", "averse"], horizontal=True, key="swarm_synth_mode")
    syn = result.get(f"synthesis_{mode}", {})
    if syn:
        regime = syn.get("regime_tag", "?")
        c = regime_color(regime)
        st.markdown(f'<div class="pd-card" style="border:2px solid {c}80">'
                    f'<b>Regime:</b> {badge(regime, c)} · '
                    f'<b>Confidence:</b> {syn.get("confidence", 0)}<br><br>'
                    f'{syn.get("narrative", "")[:600]}</div>',
                    unsafe_allow_html=True)
        risks = syn.get("key_risks", [])
        if risks:
            st.markdown("**Key Risks:**")
            for r in risks[:5]: st.markdown(f"- 🔴 {r}")
        triggers = syn.get("watch_triggers", [])
        if triggers:
            st.markdown("**Watch Triggers:**")
            for t in triggers[:5]: st.markdown(f"- 🟡 {t}")

    # ─── Phase 4: Action Selector ─────────────────────
    st.markdown(f'<h3 style="color:{C["purple"]};margin-top:14px">🎯 Phase 4 — Action Selector</h3>',
                unsafe_allow_html=True)
    p4 = result.get("phase4_action", {})

    def _picks_df(picks: list[dict]) -> pd.DataFrame:
        return pd.DataFrame([{
            "Ticker": p.get("ticker", ""),
            "Name": (p.get("name") or "")[:25],
            "Sector": (p.get("sector") or "")[:15],
            "Comp": f"{p.get('composite', 0):.1f}",
            "Rationale": (p.get("rationale") or "")[:80],
        } for p in picks])

    for bucket, emoji in [
        ("long_stocks", "🟢📈"), ("long_etfs", "🟢📦"),
        ("short_stocks", "🔴📈"), ("short_etfs", "🔴📦"),
    ]:
        picks = p4.get(bucket, [])
        if not picks: continue
        with st.expander(f"{emoji} {bucket} ({len(picks)})", expanded=False):
            st.dataframe(_picks_df(picks[:20]), use_container_width=True, hide_index=True)

    sector_scores = p4.get("sector_scores", [])
    if sector_scores:
        with st.expander("📊 GICS Sector Scores", expanded=False):
            df = pd.DataFrame(sector_scores)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ─── Phase 5: PM Agent ────────────────────────────
    st.markdown(f'<h3 style="color:{C["purple"]};margin-top:14px">📋 Phase 5 — PM Agent</h3>',
                unsafe_allow_html=True)
    p5 = result.get("phase5_pm", {})
    if p5:
        pm_comm = p5.get("pm_commentary", "")
        thesis = p5.get("portfolio_thesis", "")
        if thesis:
            st.markdown(f'<div class="pd-card" style="border:2px solid {C["purple"]}80">'
                        f'<b>Portfolio Thesis:</b><br>{thesis}</div>',
                        unsafe_allow_html=True)
        if pm_comm:
            with st.expander("PM Commentary (full)", expanded=False):
                st.markdown(pm_comm)

        horizon = st.radio("Horizon", ["tactical", "core", "strategic"],
                           horizontal=True, key="swarm_horizon")
        h_data = p5.get("horizons", {}).get(horizon, {})
        for bucket, emoji in [
            ("long_stocks", "🟢📈"), ("long_etfs", "🟢📦"),
            ("short_stocks", "🔴📈"), ("short_etfs", "🔴📦"),
        ]:
            picks = h_data.get(bucket, [])
            if not picks: continue
            with st.expander(f"{emoji} {bucket} ({len(picks)})", expanded=False):
                st.dataframe(_picks_df(picks[:20]), use_container_width=True, hide_index=True)

        # Hedge pairs
        hedge_pairs = p5.get("hedge_pairs", [])
        if hedge_pairs:
            with st.expander(f"⚖️ Hedge Pairs ({len(hedge_pairs)})", expanded=False):
                for hp in hedge_pairs:
                    st.markdown(
                        f"- **{hp.get('long', '?')}** ⇄ **{hp.get('short', '?')}** "
                        f"· {hp.get('sector', '?')} · {hp.get('horizon', '?')}"
                    )
                    st.caption(hp.get("rationale", ""))

        # Risk budget
        risk_budget = p5.get("risk_budget", [])
        if risk_budget:
            with st.expander(f"💰 Risk Budget ({len(risk_budget)})", expanded=False):
                df = pd.DataFrame(risk_budget)
                st.dataframe(df, use_container_width=True, hide_index=True)
