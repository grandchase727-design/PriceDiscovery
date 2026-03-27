"""
Price Discovery Scanner v5.0 — Streamlit Dashboard
====================================================
Usage:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

# ── make sure price_discovery module is importable ──
sys.path.insert(0, os.path.dirname(__file__))
from price_discovery import (
    GLOBAL_ETF_UNIVERSE, STOCK_UNIVERSE, CATEGORY_BENCHMARK, STOCK_BENCHMARK,
    DataEngine, NaiveDiscoveryDetector, SignalValidityEngine,
    evaluate_eligible, compute_7day_history, fmt_data_as_of, sf, ss,
)
from collections import defaultdict
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Price Discovery Scanner v5.0",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette (matches VizEngine dark theme) ──
C = {
    "bg": "#0a0e17", "panel": "#111827", "text": "#e5e7eb",
    "cyan": "#06b6d4", "green": "#22c55e", "red": "#ef4444",
    "yellow": "#f59e0b", "orange": "#f97316", "blue": "#3b82f6",
    "gray": "#6b7280", "purple": "#8b5cf6", "brown": "#a87c5a",
}

CLASS_COLORS = {
    "🟢 CONTINUATION": C["green"],
    "🔵 FORMATION": C["blue"],
    "🟡 OVEREXTENDED": C["yellow"],
    "🟤 EXHAUSTING": C["brown"],
    "🟠 NEUTRAL": C["orange"],
    "⬇️ DOWNTREND": C["red"],
}

CLASS_SHORT = {
    "⬇️ DOWNTREND": "DOWN", "🟠 NEUTRAL": "NEUTRAL",
    "🔵 FORMATION": "FORMATION", "🟢 CONTINUATION": "CONT",
    "🟡 OVEREXTENDED": "OVEREXT", "🟤 EXHAUSTING": "EXHAUST",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Downloading & scoring universe …", ttl=3600)
def load_scan(lookback_days, use_realtime, include_stocks):
    """Run the full scan pipeline and return structured results."""
    engine = DataEngine(lookback_days=lookback_days, use_realtime=use_realtime)
    all_data = engine.download_universe()
    if not all_data:
        return pd.DataFrame(), [], {}, None, {}

    if include_stocks:
        stock_data = engine.download_universe(universe=STOCK_UNIVERSE)
        all_data.update(stock_data)

    detector = NaiveDiscoveryDetector()
    detector.load_benchmarks(all_data, extra_benchmarks=STOCK_BENCHMARK if include_stocks else None)

    # Phase 1: raw indicators
    all_raw = {}
    for ticker, etf in sorted(all_data.items()):
        if etf.valid and len(etf.df) >= 60:
            try:
                all_raw[ticker] = detector.compute_raw(etf.df, etf.category)
            except Exception:
                pass

    # Phase 2: cross-sectional percentile ranking
    all_ranks = NaiveDiscoveryDetector.compute_percentile_ranks(all_raw)

    # Phase 3: signal validity
    ve = SignalValidityEngine()
    ve.compute(all_data, detector)

    # Phase 4: scoring & classification
    results = []
    for ticker in sorted(all_raw.keys()):
        etf = all_data[ticker]
        raw = all_raw[ticker]
        ranks = all_ranks[ticker]
        try:
            tcs = NaiveDiscoveryDetector.score_tcs(raw)
            tfs = NaiveDiscoveryDetector.score_tfs(raw)
            oer = NaiveDiscoveryDetector.score_oer(raw)
            rss = round(ranks["rss"], 1)
            comp = NaiveDiscoveryDetector.composite(tcs, tfs, rss, oer)
            cls = NaiveDiscoveryDetector.classify(raw, tcs, tfs, oer)
            eligible, rejection = evaluate_eligible(
                {"classification": cls, "composite": comp}, raw["adv_usd"]
            )
            validity = ve.get_validity(ticker, comp, cls)
            current_close = raw["last_close"]
            data_as_of = fmt_data_as_of(etf.df)

            def _hist(df_hist):
                if df_hist is None or df_hist.empty or len(df_hist) < 60:
                    return 0.0, False, 0.0
                a = detector.analyze_single(df_hist, etf.category)
                el, _ = evaluate_eligible(a, a["adv_usd"])
                hc = sf(df_hist["Close"].iloc[-1])
                ret = ((current_close / hc) - 1) * 100 if hc > 0 else 0.0
                return a["composite"], el, ret

            df_1w = etf.df[etf.df.index <= (etf.df.index[-1] - pd.Timedelta(days=7))]
            sc_1w, el_1w, ret_1w = _hist(df_1w)
            df_1m = etf.df[etf.df.index <= (etf.df.index[-1] - pd.Timedelta(days=30))]
            sc_1m, el_1m, ret_1m = _hist(df_1m)
            df_3m = etf.df[etf.df.index <= (etf.df.index[-1] - pd.Timedelta(days=90))]
            sc_3m, el_3m, ret_3m = _hist(df_3m)

            results.append({
                "ticker": ticker, "name": etf.name, "category": etf.category,
                "data_as_of": data_as_of,
                "composite": comp, "tcs": tcs, "tfs": tfs, "oer": oer, "rss": rss,
                "classification": cls, "eligible": eligible, "rejection": rejection,
                "rsi": round(raw["rsi"], 1), "trend_age": raw["trend_age"],
                "sma50_dist": round(raw["sma50_dist"], 2),
                "adv_usd": raw["adv_usd"],
                **validity,
                "score_1w": sc_1w, "ret_1w": ret_1w,
                "score_1m": sc_1m, "ret_1m": ret_1m,
                "score_3m": sc_3m, "ret_3m": ret_3m,
            })
        except Exception:
            pass

    results.sort(key=lambda x: (-x["composite"], x["ticker"]))

    # 7-day history
    history = compute_7day_history(all_data, detector)

    # validity engine stats
    ve_stats = {
        "bucket": ve.bucket_stats,
        "class": ve.class_stats,
        "transitions": dict(ve.transition_counts),
        "transition_totals": dict(ve.transition_totals),
    }

    df = pd.DataFrame(results)
    return df, results, history, ve, ve_stats


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📡 Scanner Controls")
    lookback = st.selectbox("Lookback", [1, 2, 3, 5], index=3, format_func=lambda y: f"{y}Y")
    use_rt = st.toggle("Real-time price", value=True)
    inc_stk = st.toggle("Include stocks", value=True)

    if st.button("🔄 Run Scan", type="primary", use_container_width=True):
        st.cache_data.clear()

    st.divider()
    st.caption("Filters")

# ── run / load ──
df, results, history, ve, ve_stats = load_scan(lookback * 365, use_rt, inc_stk)

if df.empty:
    st.error("No data returned. Check network / yfinance.")
    st.stop()

# ── sidebar filters (need df loaded) ──
with st.sidebar:
    all_cats = sorted(df["category"].unique())
    sel_cats = st.multiselect("Categories", all_cats, default=all_cats)

    all_classes = sorted(df["classification"].unique())
    sel_classes = st.multiselect("Classification", all_classes, default=all_classes)

    eligible_only = st.toggle("Eligible only", value=False)

    comp_range = st.slider("Composite range", 0.0, 100.0, (0.0, 100.0), 0.5)

# apply filters
mask = (
    df["category"].isin(sel_cats)
    & df["classification"].isin(sel_classes)
    & df["composite"].between(*comp_range)
)
if eligible_only:
    mask &= df["eligible"]
fdf = df[mask].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📊 Price Discovery Scanner v5.0")
st.caption(f"Universe: {len(df)} tickers | Filtered: {len(fdf)} | Scan date: {df['data_as_of'].iloc[0] if not df.empty else 'N/A'}")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_table, tab_signals, tab_category, tab_validity, tab_history = st.tabs(
    ["Overview", "Master Table", "Signal Decomposition", "Category Analysis", "Signal Validity", "7-Day History"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    # KPIs
    n_eligible = int(fdf["eligible"].sum())
    avg_comp = fdf["composite"].mean()
    avg_rsi = fdf["rsi"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tickers", len(fdf))
    c2.metric("Eligible", n_eligible)
    c3.metric("Avg Composite", f"{avg_comp:.1f}")
    c4.metric("Avg RSI", f"{avg_rsi:.1f}")

    col_left, col_right = st.columns(2)

    # Classification distribution
    with col_left:
        cls_counts = fdf["classification"].value_counts().reset_index()
        cls_counts.columns = ["classification", "count"]
        cls_counts["color"] = cls_counts["classification"].map(CLASS_COLORS)
        fig_pie = px.pie(
            cls_counts, values="count", names="classification",
            color="classification",
            color_discrete_map=CLASS_COLORS,
            title="Classification Distribution",
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
            font_color=C["text"], legend=dict(font_size=11),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Composite distribution
    with col_right:
        fig_hist = px.histogram(
            fdf, x="composite", nbins=25, color="classification",
            color_discrete_map=CLASS_COLORS,
            title="Composite Score Distribution",
        )
        fig_hist.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
            font_color=C["text"], bargap=0.05,
            xaxis_title="Composite", yaxis_title="Count",
        )
        fig_hist.add_vline(x=55, line_dash="dash", line_color=C["orange"],
                           annotation_text="Eligible=55")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Top 10 eligible
    st.subheader("Top 15 Eligible Tickers")
    top_el = fdf[fdf["eligible"]].head(15)
    if not top_el.empty:
        fig_top = go.Figure(go.Bar(
            y=top_el["ticker"] + " " + top_el["name"].str[:15],
            x=top_el["composite"],
            orientation="h",
            marker_color=top_el["classification"].map(CLASS_COLORS),
            text=top_el.apply(
                lambda r: f"TCS:{r['tcs']} TFS:{r['tfs']} OER:{r['oer']} RSS:{r['rss']:.0f}",
                axis=1,
            ),
            textposition="outside",
            textfont_size=10,
        ))
        fig_top.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
            yaxis=dict(autorange="reversed"), xaxis_title="Composite",
            height=max(350, len(top_el) * 30),
            margin=dict(l=0, r=120),
        )
        fig_top.add_vline(x=55, line_dash="dot", line_color=C["orange"])
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No eligible tickers in current filter.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: MASTER TABLE
# ─────────────────────────────────────────────────────────────────────────────
with tab_table:
    st.subheader("Master Summary")

    display_cols = [
        "ticker", "name", "category", "composite", "tcs", "tfs", "oer", "rss",
        "classification", "eligible", "rejection",
        "rsi", "trend_age", "sma50_dist",
        "val_prob", "val_persist", "val_conf",
        "score_1w", "ret_1w", "score_1m", "ret_1m", "score_3m", "ret_3m",
    ]
    show_df = fdf[display_cols].copy()
    show_df["adv_M"] = (fdf["adv_usd"] / 1e6).round(1)

    # Format with color via column_config
    st.dataframe(
        show_df,
        use_container_width=True,
        height=700,
        column_config={
            "composite": st.column_config.NumberColumn("Comp", format="%.1f"),
            "tcs": st.column_config.ProgressColumn("TCS", min_value=0, max_value=100, format="%d"),
            "tfs": st.column_config.ProgressColumn("TFS", min_value=0, max_value=100, format="%d"),
            "oer": st.column_config.ProgressColumn("OER", min_value=0, max_value=100, format="%d"),
            "rss": st.column_config.NumberColumn("RSS", format="%.1f"),
            "val_prob": st.column_config.NumberColumn("Val%", format="%.1f"),
            "val_persist": st.column_config.NumberColumn("Persist", format="%.0f"),
            "ret_1w": st.column_config.NumberColumn("1W Ret%", format="%.2f%%"),
            "ret_1m": st.column_config.NumberColumn("1M Ret%", format="%.2f%%"),
            "ret_3m": st.column_config.NumberColumn("3M Ret%", format="%.2f%%"),
            "sma50_dist": st.column_config.NumberColumn("SMA50 Dist%", format="%.2f"),
            "adv_M": st.column_config.NumberColumn("ADV ($M)", format="%.1f"),
        },
        hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: SIGNAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
with tab_signals:
    st.subheader("3-Axis Signal Decomposition")

    # scatter: TCS vs TFS, sized by composite, colored by classification
    fig_scatter = px.scatter(
        fdf, x="tcs", y="tfs", size="composite",
        color="classification", color_discrete_map=CLASS_COLORS,
        hover_data=["ticker", "name", "oer", "rss", "composite"],
        title="TCS vs TFS (size = Composite)",
        size_max=20,
    )
    fig_scatter.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
        xaxis_title="TCS (Trend Continuation)", yaxis_title="TFS (Trend Formation)",
        height=550,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3-axis radar for top eligible
    st.subheader("Signal Radar — Top Eligible")
    top_radar = fdf[fdf["eligible"]].head(10)
    if not top_radar.empty:
        fig_radar = go.Figure()
        cats_radar = ["TCS", "TFS", "OER", "RSS", "TCS"]
        for _, r in top_radar.iterrows():
            vals = [r["tcs"], r["tfs"], r["oer"], r["rss"], r["tcs"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats_radar, name=f"{r['ticker']}",
                fill="toself", opacity=0.3,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor=C["panel"],
                radialaxis=dict(range=[0, 100], color=C["gray"]),
                angularaxis=dict(color=C["text"]),
            ),
            paper_bgcolor=C["bg"], font_color=C["text"],
            height=500, showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # OER heatmap
    st.subheader("Overextension Risk (OER) vs RSI")
    fig_oer = px.scatter(
        fdf, x="rsi", y="sma50_dist", color="oer",
        color_continuous_scale="YlOrRd",
        hover_data=["ticker", "name", "classification"],
        title="RSI vs SMA50 Distance (color = OER)",
    )
    fig_oer.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
        xaxis_title="RSI-14", yaxis_title="SMA50 Distance (%)",
        height=450,
    )
    fig_oer.add_hline(y=0, line_dash="dash", line_color=C["gray"])
    fig_oer.add_vline(x=70, line_dash="dash", line_color=C["yellow"],
                      annotation_text="RSI 70")
    st.plotly_chart(fig_oer, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: CATEGORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_category:
    st.subheader("Category Breakdown")

    # summary stats per category
    cat_agg = fdf.groupby("category").agg(
        count=("ticker", "size"),
        eligible=("eligible", "sum"),
        avg_comp=("composite", "mean"),
        avg_tcs=("tcs", "mean"),
        avg_tfs=("tfs", "mean"),
        avg_oer=("oer", "mean"),
        avg_ret_1w=("ret_1w", "mean"),
        avg_ret_1m=("ret_1m", "mean"),
        avg_ret_3m=("ret_3m", "mean"),
    ).round(1).reset_index()
    cat_agg["eligible"] = cat_agg["eligible"].astype(int)

    st.dataframe(cat_agg, use_container_width=True, hide_index=True,
                 column_config={
                     "avg_ret_1w": st.column_config.NumberColumn("1W Ret%", format="%.2f%%"),
                     "avg_ret_1m": st.column_config.NumberColumn("1M Ret%", format="%.2f%%"),
                     "avg_ret_3m": st.column_config.NumberColumn("3M Ret%", format="%.2f%%"),
                 })

    # category avg composite bar
    fig_cat = px.bar(
        cat_agg.sort_values("avg_comp", ascending=True),
        y="category", x="avg_comp", orientation="h",
        color="avg_comp", color_continuous_scale="Viridis",
        text="eligible", title="Avg Composite by Category (label = eligible count)",
    )
    fig_cat.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
        height=max(350, len(cat_agg) * 28),
    )
    fig_cat.add_vline(x=55, line_dash="dot", line_color=C["orange"])
    st.plotly_chart(fig_cat, use_container_width=True)

    # category returns comparison
    st.subheader("Returns by Category")
    ret_melt = cat_agg.melt(
        id_vars="category",
        value_vars=["avg_ret_1w", "avg_ret_1m", "avg_ret_3m"],
        var_name="period", value_name="return_%",
    )
    ret_melt["period"] = ret_melt["period"].map(
        {"avg_ret_1w": "1 Week", "avg_ret_1m": "1 Month", "avg_ret_3m": "3 Month"}
    )
    fig_ret = px.bar(
        ret_melt, x="category", y="return_%", color="period",
        barmode="group", title="Average Returns by Category & Period",
        color_discrete_sequence=[C["cyan"], C["blue"], C["purple"]],
    )
    fig_ret.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
        xaxis_tickangle=-45, height=450,
    )
    fig_ret.add_hline(y=0, line_dash="dash", line_color=C["gray"])
    st.plotly_chart(fig_ret, use_container_width=True)

    # per-category detail expander
    st.subheader("Per-Category Detail")
    for cat in sorted(fdf["category"].unique()):
        cat_df = fdf[fdf["category"] == cat].sort_values("composite", ascending=False)
        n_el = int(cat_df["eligible"].sum())
        with st.expander(f"{cat}  ({len(cat_df)} tickers, {n_el} eligible)"):
            st.dataframe(
                cat_df[["ticker", "name", "composite", "tcs", "tfs", "oer", "rss",
                         "classification", "eligible", "ret_1w", "ret_1m", "ret_3m"]],
                use_container_width=True, hide_index=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: SIGNAL VALIDITY
# ─────────────────────────────────────────────────────────────────────────────
with tab_validity:
    st.subheader("Signal Validity Verification (Past 1-Month Backtest)")

    if ve_stats:
        col_b, col_c = st.columns(2)

        # Bucket stats
        with col_b:
            st.markdown("**Score Bucket Analysis**")
            bucket_order = ["0-30", "30-50", "50-70", "70-100"]
            bdata = []
            for b in bucket_order:
                s = ve_stats["bucket"].get(b, {"n": 0, "hit_rate": 0, "exc_hit": 0, "avg_ret": 0, "avg_exc": 0})
                bdata.append({"Bucket": b, **s})
            bdf = pd.DataFrame(bdata)
            st.dataframe(bdf, use_container_width=True, hide_index=True)

            fig_bkt = px.bar(
                bdf, x="Bucket", y=["hit_rate", "exc_hit"],
                barmode="group", title="Hit Rate by Score Bucket",
                color_discrete_sequence=[C["green"], C["cyan"]],
                labels={"value": "%", "variable": "Metric"},
            )
            fig_bkt.update_layout(
                paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
            )
            st.plotly_chart(fig_bkt, use_container_width=True)

        # Class stats
        with col_c:
            st.markdown("**Classification Analysis**")
            class_order = ["⬇️ DOWNTREND", "🟠 NEUTRAL", "🔵 FORMATION",
                           "🟢 CONTINUATION", "🟡 OVEREXTENDED", "🟤 EXHAUSTING"]
            cdata = []
            for c in class_order:
                s = ve_stats["class"].get(c, {"n": 0, "hit_rate": 0, "exc_hit": 0, "avg_ret": 0, "avg_exc": 0})
                cdata.append({"Class": CLASS_SHORT.get(c, c), **s})
            cdf = pd.DataFrame(cdata)
            st.dataframe(cdf, use_container_width=True, hide_index=True)

            fig_cls = px.bar(
                cdf, x="Class", y=["avg_ret", "avg_exc"],
                barmode="group", title="Avg Return by Classification",
                color_discrete_sequence=[C["blue"], C["purple"]],
                labels={"value": "%", "variable": "Metric"},
            )
            fig_cls.update_layout(
                paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
            )
            st.plotly_chart(fig_cls, use_container_width=True)

        # Transition matrix
        st.subheader("Class Transition Matrix")
        classes = ["⬇️ DOWNTREND", "🟠 NEUTRAL", "🔵 FORMATION",
                   "🟢 CONTINUATION", "🟡 OVEREXTENDED", "🟤 EXHAUSTING"]
        shorts = [CLASS_SHORT[c] for c in classes]
        trans_matrix = []
        for cf in classes:
            tot = ve_stats["transition_totals"].get(cf, 0)
            row = {}
            for ct in classes:
                key = str((cf, ct))
                cnt = ve_stats["transitions"].get((cf, ct), 0)
                row[CLASS_SHORT[ct]] = round(cnt / tot * 100, 1) if tot > 0 else 0.0
            trans_matrix.append(row)

        tdf = pd.DataFrame(trans_matrix, index=shorts)
        fig_heat = px.imshow(
            tdf.values, x=shorts, y=shorts,
            color_continuous_scale="Blues",
            text_auto=".1f",
            title="Transition Probability (%, row → col)",
        )
        fig_heat.update_layout(
            paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
            xaxis_title="To", yaxis_title="From", height=400,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Validity engine data not available.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: 7-DAY HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("7-Day Composite Trend")

    eligible_tickers = fdf[fdf["eligible"]]["ticker"].tolist()[:30]
    if not eligible_tickers:
        eligible_tickers = fdf["ticker"].tolist()[:20]

    sel_tickers = st.multiselect(
        "Select tickers to chart",
        fdf["ticker"].tolist(),
        default=eligible_tickers[:15],
    )

    if sel_tickers and history:
        traces = []
        for t in sel_tickers:
            h = history.get(t, [])
            if len(h) < 2:
                continue
            dates = [x["date"] for x in h]
            comps = [x["composite"] for x in h]
            traces.append(go.Scatter(
                x=dates, y=comps, mode="lines+markers",
                name=t, marker=dict(size=5),
            ))

        if traces:
            fig_trend = go.Figure(data=traces)
            fig_trend.add_hline(y=55, line_dash="dash", line_color=C["orange"],
                                annotation_text="Eligible threshold (55)")
            fig_trend.update_layout(
                paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
                xaxis_title="Date", yaxis_title="Composite",
                height=500, legend=dict(font_size=10),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

    # CONTINUATION class trend
    cont_tickers = fdf[fdf["classification"].str.contains("CONTINUATION")]["ticker"].tolist()
    if cont_tickers and history:
        st.subheader("CONTINUATION Class — 1-Week Trend Tracking")
        trend_data = []
        for t in cont_tickers:
            h = history.get(t, [])
            cls_trend = " → ".join(CLASS_SHORT.get(x["class"], "?") for x in h)
            name = fdf.loc[fdf["ticker"] == t, "name"].iloc[0] if len(fdf[fdf["ticker"] == t]) else ""
            trend_data.append({"Ticker": t, "Name": name, "Class Trend (Oldest → Newest)": cls_trend})
        st.dataframe(pd.DataFrame(trend_data), use_container_width=True, hide_index=True)

    # Score change analysis
    st.subheader("Score Change (1W / 1M / 3M)")
    change_df = fdf[["ticker", "name", "composite", "score_1w", "score_1m", "score_3m"]].copy()
    change_df["Δ1W"] = change_df["composite"] - change_df["score_1w"]
    change_df["Δ1M"] = change_df["composite"] - change_df["score_1m"]
    change_df["Δ3M"] = change_df["composite"] - change_df["score_3m"]

    sort_col = st.selectbox("Sort by", ["Δ1W", "Δ1M", "Δ3M"], index=0)
    ascending = st.toggle("Ascending (worst first)", value=False)
    change_sorted = change_df.sort_values(sort_col, ascending=ascending).head(30)

    fig_change = go.Figure(go.Bar(
        y=change_sorted["ticker"],
        x=change_sorted[sort_col],
        orientation="h",
        marker_color=[C["green"] if v >= 0 else C["red"] for v in change_sorted[sort_col]],
        text=change_sorted[sort_col].round(1),
        textposition="outside",
    ))
    fig_change.update_layout(
        paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
        xaxis_title=f"Composite Change ({sort_col})",
        yaxis=dict(autorange="reversed"),
        height=max(350, len(change_sorted) * 22),
        margin=dict(l=0, r=80),
        title=f"Top {len(change_sorted)} — Composite Score Change ({sort_col})",
    )
    fig_change.add_vline(x=0, line_color=C["gray"])
    st.plotly_chart(fig_change, use_container_width=True)
