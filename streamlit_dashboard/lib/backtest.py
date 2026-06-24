"""BacktestPanel — Trading Layer Backtest port."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .api import (
    fetch_backtest_rankings, fetch_backtest_results, fetch_backtest_status,
    fetch_final_list, fetch_validated_extra_timeline,
)
from .theme import get_theme, plotly_theme
from .utils import badge, colored_pct, fmt_pct


def render_backtest_panel() -> None:
    C = get_theme()
    st.markdown(f'<h2 style="color:{C["amber"]}">📊 Backtest — Trading Layer Analysis</h2>',
                unsafe_allow_html=True)

    status = fetch_backtest_status()
    if status.get("running"):
        st.info("📊 Backtest 실행 중 — 사이드바 Run Live Scan으로 트리거됩니다.")

    results = fetch_backtest_results()
    rankings = fetch_backtest_rankings()

    if not results:
        st.warning("📭 Backtest 결과 캐시 없음 — 사이드바 Run Live Scan으로 실행 후 다시 확인하세요.")
        return

    # ─── Header ────────────────────────────────────────
    cols = st.columns([3, 1])
    with cols[0]:
        n_picks = results.get("n_picks", 0)
        date_range = results.get("date_range", "?")
        st.markdown(f"**Picks:** {n_picks} · **Date Range:** `{date_range}`")
    with cols[1]:
        if st.button("🔄 Refresh", key="bt_refresh"):
            fetch_backtest_results.clear()
            fetch_backtest_rankings.clear()
            st.rerun()

    horizon = st.radio("Horizon", ["5d", "21d", "63d"], horizontal=True, index=1, key="bt_horizon")

    # ─── Per-bucket metrics ──────────────────────────
    buckets = results.get("buckets", {})
    for bucket_key, label, color in [
        ("long_stocks", "📈 Long Stocks", C["green"]),
        ("long_etfs", "📦 Long ETFs", C["green"]),
        ("short_stocks", "📈 Short Stocks", C["red"]),
        ("short_etfs", "📦 Short ETFs", C["red"]),
    ]:
        b = buckets.get(bucket_key, {})
        if not b: continue
        h = b.get(horizon, {})
        if not h: continue

        st.markdown(f'<h3 style="color:{color};margin-top:14px">{label} — {horizon}</h3>',
                    unsafe_allow_html=True)

        # Layer 1: Direction stats
        direction = h.get("direction", {})
        if direction:
            cols = st.columns(6)
            stats_d = [
                ("Hit Rate", direction.get("hit_rate", 0), "%", "higher"),
                ("Mean Return", direction.get("mean_return", 0) * 100, "%", "higher"),
                ("Median", direction.get("median", 0) * 100, "%", "higher"),
                ("P25", direction.get("p25", 0) * 100, "%", None),
                ("P75", direction.get("p75", 0) * 100, "%", None),
                ("Std", direction.get("std", 0) * 100, "%", None),
            ]
            for i, (label_s, val, unit, _) in enumerate(stats_d):
                cols[i].metric(label_s, f"{val:.1f}{unit}")

        # Layer 2: Alpha (edge)
        edge = h.get("edge", {})
        if edge:
            st.markdown("**Layer 2 — Alpha vs Sector ETF:**")
            cols = st.columns(6)
            stats_e = [
                ("Mean α", edge.get("mean_alpha", 0) * 100, "%"),
                ("Win Rate", edge.get("win_rate", 0), "%"),
                ("Avg Win", edge.get("avg_win", 0) * 100, "%"),
                ("Avg Loss", edge.get("avg_loss", 0) * 100, "%"),
                ("W/L", f"{edge.get('win_loss_ratio', 0):.2f}", ""),
                ("t-stat", f"{edge.get('t_stat', 0):.2f}", ""),
            ]
            for i, (l, v, u) in enumerate(stats_e):
                cols[i].metric(l, f"{v}{u}" if isinstance(v, str) else f"{v:.2f}{u}")

        # Layer 3: Rank quintile
        quintile = h.get("rank_quintile", [])
        if quintile:
            with st.expander("Layer 3 — Rank Quintile", expanded=False):
                df = pd.DataFrame(quintile)
                st.dataframe(df, use_container_width=True, hide_index=True)

    # ─── Top/Worst tickers ───────────────────────────
    if rankings:
        st.markdown(f'<h3 style="color:{C["amber"]};margin-top:14px">🏆 Top/Worst Tickers</h3>',
                    unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**🟢 Top 10**")
            top = rankings.get("top", [])[:10]
            if top:
                df = pd.DataFrame([{
                    "Ticker": t.get("ticker", ""),
                    "Name": (t.get("name") or "")[:25],
                    "Sector": (t.get("sector") or "")[:15],
                    "Avg Rank": t.get("avg_rank", 0),
                    "Win%": f"{t.get('win_pct', 0):.1f}",
                    "α 21d": f"{t.get('alpha_21d', 0) * 100:.2f}%",
                } for t in top])
                st.dataframe(df, use_container_width=True, hide_index=True)

        with cols[1]:
            st.markdown("**🔴 Worst 10**")
            worst = rankings.get("worst", [])[:10]
            if worst:
                df = pd.DataFrame([{
                    "Ticker": t.get("ticker", ""),
                    "Name": (t.get("name") or "")[:25],
                    "Sector": (t.get("sector") or "")[:15],
                    "Avg Rank": t.get("avg_rank", 0),
                    "Win%": f"{t.get('win_pct', 0):.1f}",
                    "α 21d": f"{t.get('alpha_21d', 0) * 100:.2f}%",
                } for t in worst])
                st.dataframe(df, use_container_width=True, hide_index=True)

    # ─── Validated buy list timeline ─────────────────
    try:
        final_list = fetch_final_list()
        buy_list = final_list.get("buy_list", [])
        high_tier = [r for r in buy_list if r.get("stars", 0) >= 2]
        stocks = [r for r in high_tier if "stocks" in (r.get("bucket") or "").lower()]
        etfs = [r for r in high_tier if "etfs" in (r.get("bucket") or "").lower()]
        if stocks or etfs:
            st.markdown(
                f'<h3 style="color:{C["green"]};margin-top:14px">'
                f'🏆 검증된 매수 종목 (★★ / ★★★) — 과거 성과 추이</h3>',
                unsafe_allow_html=True
            )
            st.caption(f"Stocks: {len(stocks)} · ETFs: {len(etfs)}")
            if stocks:
                with st.expander(f"📈 Validated Long Stocks ({len(stocks)})", expanded=False):
                    df = pd.DataFrame([{
                        "Ticker": r["ticker"], "Stars": r.get("stars", 0),
                        "Composite": r.get("composite", 0),
                        "Consensus": r.get("consensus", "?"),
                    } for r in stocks])
                    st.dataframe(df, use_container_width=True, hide_index=True)
            if etfs:
                with st.expander(f"📦 Validated Long ETFs ({len(etfs)})", expanded=False):
                    df = pd.DataFrame([{
                        "Ticker": r["ticker"], "Stars": r.get("stars", 0),
                        "Composite": r.get("composite", 0),
                        "Consensus": r.get("consensus", "?"),
                    } for r in etfs])
                    st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"Final List 로드 실패: {e}")
