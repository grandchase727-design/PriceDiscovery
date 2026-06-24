"""MultiAgentConvictionDebateCard port."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .api import fetch_multi_agent_debate
from .theme import get_theme
from .utils import badge, comp_color


def render_conviction_debate() -> None:
    C = get_theme()
    st.markdown(f'<h2 style="color:{C["purple"]}">🎭 Multi-Agent Conviction Debate</h2>',
                unsafe_allow_html=True)

    data = fetch_multi_agent_debate()
    if not data:
        st.warning("📭 Multi-Agent Debate 캐시 없음 — Run Live Scan으로 실행하세요.")
        return

    # Staleness indicator
    last_run = data.get("last_run")
    if last_run:
        st.caption(f"Last run: `{last_run}`")

    group = st.radio("Group", ["momentum", "pre_momentum"], horizontal=True, key="cd_group")
    mode = st.radio("Synthesis Mode", ["neutral", "averse"], horizontal=True, key="cd_mode")

    verdicts = data.get("verdicts", {}) if isinstance(data.get("verdicts"), dict) else {}
    bucket = verdicts.get(group, {}) if verdicts else {}

    for section_key, section_label, color in [
        ("long_stocks", "🟢📈 LONG Stocks", C["green"]),
        ("long_etfs", "🟢📦 LONG ETFs", C["green"]),
        ("short_stocks", "🔴📈 SHORT Stocks", C["red"]),
        ("short_etfs", "🔴📦 SHORT ETFs", C["red"]),
    ]:
        rows = bucket.get(section_key, [])
        if not rows: continue

        st.markdown(f'<h3 style="color:{color};margin-top:14px">{section_label} ({len(rows)})</h3>',
                    unsafe_allow_html=True)

        df_rows = []
        for r in rows[:20]:
            specialists = r.get("specialists", {})
            syn = r.get(f"synthesis_{mode}", r.get("synthesis", {}))
            df_rows.append({
                "Ticker": r.get("ticker", ""),
                "Name": (r.get("name") or "")[:25],
                "Comp": f"{r.get('composite', 0):.1f}",
                "Fund": specialists.get("fundamental", {}).get("rating", "?"),
                "Sent": specialists.get("sentiment", {}).get("rating", "?"),
                "Val":  specialists.get("valuation", {}).get("rating", "?"),
                "Synthesis": syn.get("rating", "?"),
                "Confidence": f"{syn.get('confidence', 0):.0f}%",
                "Risk Gap": r.get("risk_gap", 0),
            })

        st.dataframe(pd.DataFrame(df_rows), use_container_width=True, hide_index=True)

        # Detail expander for first 3
        for r in rows[:3]:
            with st.expander(f"📋 {r.get('ticker', '?')} — Details", expanded=False):
                specialists = r.get("specialists", {})
                for sp_key, sp_label in [("fundamental", "🏛 Fundamental"),
                                          ("sentiment", "💬 Sentiment"),
                                          ("valuation", "💰 Valuation")]:
                    sp = specialists.get(sp_key, {})
                    if sp:
                        st.markdown(f"**{sp_label}** — {sp.get('rating', '?')} (conf {sp.get('confidence', 0)}%)")
                        for k in sp.get("key_points", [])[:3]:
                            st.markdown(f"- {k}")
                syn = r.get(f"synthesis_{mode}", r.get("synthesis", {}))
                if syn.get("reasoning"):
                    st.markdown(f"**Synthesis ({mode}):** {syn['reasoning']}")
