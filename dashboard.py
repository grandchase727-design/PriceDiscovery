"""
Price Discovery Scanner v5.0 — Streamlit Dashboard
====================================================
Usage:  python3 -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, pickle
from datetime import datetime

# ── make sure price_discovery module is importable ──
sys.path.insert(0, os.path.dirname(__file__))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, ".scan_cache.pkl")

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

DARK_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — cache-first, live-scan only when requested
# ═══════════════════════════════════════════════════════════════════════════════
def load_from_cache():
    """Load pre-computed results from .scan_cache.pkl (written by run_scan)."""
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        results = cache["results"]
        df = pd.DataFrame(results)
        history = cache.get("history", {})
        ve_stats = {
            "bucket": cache.get("ve_bucket", {}),
            "class": cache.get("ve_class", {}),
            "transitions": cache.get("ve_transitions", {}),
            "transition_totals": cache.get("ve_transition_totals", {}),
        }
        scan_time = cache.get("scan_time", "unknown")
        return df, results, history, ve_stats, scan_time
    except Exception as e:
        st.warning(f"Cache load failed: {e}")
        return None

@st.cache_data(show_spinner="Running live scan (this takes a few minutes) …", ttl=3600)
def run_live_scan(lookback_days, use_realtime, include_stocks):
    """Full live scan — only called when user clicks Run Scan."""
    from price_discovery import run_scan as _run_scan
    _df, results, _all_data = _run_scan(
        lookback_days=lookback_days,
        use_realtime=use_realtime,
        include_stocks=include_stocks,
    )
    loaded = load_from_cache()
    if loaded:
        return loaded
    df = pd.DataFrame(results)
    return df, results, {}, {}, datetime.today().isoformat()

@st.cache_data(show_spinner="Loading price data …", ttl=3600)
def load_ticker_price(ticker, period="1y"):
    """Download price data for single ticker (for Deep Dive tab)."""
    import yfinance as yf
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df is not None and not df.empty:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📡 Scanner Controls")

    cache_exists = os.path.exists(CACHE_PATH)
    if cache_exists:
        mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
        age_min = (datetime.now() - mtime).total_seconds() / 60
        if age_min < 60:
            age_str = f"{age_min:.0f}min ago"
        else:
            age_str = mtime.strftime("%Y-%m-%d %H:%M")
        st.success(f"Cache: {age_str}")
    else:
        st.warning("No cache — run `python3 price_discovery.py` first")

    st.divider()
    st.caption("Live Scan (optional)")
    lookback = st.selectbox("Lookback", [1, 2, 3, 5], index=3, format_func=lambda y: f"{y}Y")
    use_rt = st.toggle("Real-time price", value=True)
    inc_stk = st.toggle("Include stocks", value=True)
    do_scan = st.button("Run Live Scan", type="primary", use_container_width=True)

    st.divider()
    st.caption("Filters")

# ── data load strategy ──
if do_scan:
    st.cache_data.clear()
    loaded = run_live_scan(lookback * 365, use_rt, inc_stk)
    df, results, history, ve_stats, scan_time = loaded
else:
    cached = load_from_cache()
    if cached:
        df, results, history, ve_stats, scan_time = cached
    else:
        st.info("No cached results. Click **Run Live Scan** or run `python3 price_discovery.py` first.")
        st.stop()

if df.empty:
    st.error("No data returned. Check network / yfinance.")
    st.stop()

# ── sidebar filters ──
with st.sidebar:
    all_cats = sorted(df["category"].unique())
    sel_cats = st.multiselect("Categories", all_cats, default=all_cats)
    all_classes = sorted(df["classification"].unique())
    sel_classes = st.multiselect("Classification", all_classes, default=all_classes)
    eligible_only = st.toggle("Eligible only", value=False)
    comp_range = st.slider("Composite range", 0.0, 100.0, (0.0, 100.0), 0.5)

mask = (
    df["category"].isin(sel_cats)
    & df["classification"].isin(sel_classes)
    & df["composite"].between(*comp_range)
)
if eligible_only:
    mask &= df["eligible"]
fdf = df[mask].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER & TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Price Discovery Scanner v5.0")
st.caption(f"Universe: {len(df)} tickers | Filtered: {len(fdf)} | "
           f"Scan: {df['data_as_of'].iloc[0] if not df.empty else 'N/A'}")

(tab_overview, tab_table, tab_deep, tab_signals, tab_category,
 tab_breadth, tab_portfolio, tab_validity, tab_history) = st.tabs([
    "Overview", "Master Table", "Ticker Deep Dive",
    "Signal Decomposition", "Category Analysis",
    "Market Breadth", "Portfolio View", "Signal Validity", "7-Day History",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW  (+Conviction Bubble)
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    n_eligible = int(fdf["eligible"].sum())
    avg_comp = fdf["composite"].mean() if len(fdf) else 0
    avg_rsi = fdf["rsi"].mean() if len(fdf) else 0
    pct_above = (fdf["sma50_dist"] > 0).mean() * 100 if len(fdf) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total tickers", len(fdf))
    c2.metric("Eligible", n_eligible)
    c3.metric("Avg Composite", f"{avg_comp:.1f}")
    c4.metric("Above SMA50", f"{pct_above:.0f}%")

    col_left, col_right = st.columns(2)

    with col_left:
        cls_counts = fdf["classification"].value_counts().reset_index()
        cls_counts.columns = ["classification", "count"]
        fig_pie = px.pie(
            cls_counts, values="count", names="classification",
            color="classification", color_discrete_map=CLASS_COLORS,
            title="Classification Distribution", hole=0.4,
        )
        fig_pie.update_layout(**DARK_LAYOUT, legend=dict(font_size=11))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        fig_hist = px.histogram(
            fdf, x="composite", nbins=25, color="classification",
            color_discrete_map=CLASS_COLORS,
            title="Composite Score Distribution",
        )
        fig_hist.update_layout(**DARK_LAYOUT, bargap=0.05,
                               xaxis_title="Composite", yaxis_title="Count")
        fig_hist.add_vline(x=55, line_dash="dash", line_color=C["orange"],
                           annotation_text="Eligible=55")
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Conviction Bubble Chart ──
    st.subheader("Conviction Map")
    bubble_df = fdf.copy()
    bubble_df["adv_M"] = (bubble_df["adv_usd"] / 1e6).clip(lower=1)
    fig_bubble = px.scatter(
        bubble_df, x="composite", y="val_prob",
        size="adv_M", color="classification",
        color_discrete_map=CLASS_COLORS,
        hover_data=["ticker", "name", "category", "tcs", "tfs", "oer"],
        title="Composite vs Validity% (size = ADV $M)",
        size_max=35,
    )
    fig_bubble.update_layout(**DARK_LAYOUT, height=550,
                              xaxis_title="Composite Score",
                              yaxis_title="Validity Probability %")
    fig_bubble.add_vline(x=55, line_dash="dot", line_color=C["orange"])
    fig_bubble.add_hline(y=50, line_dash="dot", line_color=C["gray"])
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Top 15 eligible
    st.subheader("Top 15 Eligible Tickers")
    top_el = fdf[fdf["eligible"]].head(15)
    if not top_el.empty:
        fig_top = go.Figure(go.Bar(
            y=top_el["ticker"] + " " + top_el["name"].str[:15],
            x=top_el["composite"], orientation="h",
            marker_color=top_el["classification"].map(CLASS_COLORS),
            text=top_el.apply(
                lambda r: f"TCS:{r['tcs']} TFS:{r['tfs']} OER:{r['oer']} RSS:{r['rss']:.0f}",
                axis=1),
            textposition="outside", textfont_size=10,
        ))
        fig_top.update_layout(**DARK_LAYOUT,
                              yaxis=dict(autorange="reversed"), xaxis_title="Composite",
                              height=max(350, len(top_el) * 30), margin=dict(l=0, r=120))
        fig_top.add_vline(x=55, line_dash="dot", line_color=C["orange"])
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.info("No eligible tickers in current filter.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: MASTER TABLE (+CSV Export)
# ─────────────────────────────────────────────────────────────────────────────
with tab_table:
    st.subheader("Master Summary")

    # CSV export
    csv_data = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_data, "scan_results.csv", "text/csv",
                       use_container_width=False)

    display_cols = [
        "ticker", "name", "category", "composite", "tcs", "tfs", "oer", "rss",
        "classification", "eligible", "rejection",
        "rsi", "trend_age", "sma50_dist",
        "val_prob", "val_persist", "val_conf",
        "score_1w", "ret_1w", "score_1m", "ret_1m", "score_3m", "ret_3m",
    ]
    show_df = fdf[display_cols].copy()
    show_df["adv_M"] = (fdf["adv_usd"] / 1e6).round(1)

    st.dataframe(
        show_df, use_container_width=True, height=700,
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
# TAB 3: TICKER DEEP DIVE (NEW)
# ─────────────────────────────────────────────────────────────────────────────
with tab_deep:
    st.subheader("Ticker Deep Dive")

    ticker_list = fdf["ticker"].tolist()
    sel_ticker = st.selectbox("Select ticker", ticker_list,
                              index=0 if ticker_list else None)

    if sel_ticker:
        row = fdf[fdf["ticker"] == sel_ticker].iloc[0]

        # ── Signal score gauges ──
        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("Composite", f"{row['composite']:.1f}")
        g2.metric("TCS", int(row["tcs"]))
        g3.metric("TFS", int(row["tfs"]))
        g4.metric("OER", int(row["oer"]))
        g5.metric("RSS", f"{row['rss']:.0f}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Classification", row["classification"])
        m2.metric("RSI-14", f"{row['rsi']:.1f}")
        m3.metric("Trend Age", f"{int(row['trend_age'])} days")
        m4.metric("SMA50 Dist", f"{row['sma50_dist']:.2f}%")

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Eligible", "Yes" if row["eligible"] else f"No ({row['rejection']})")
        v2.metric("Validity %", f"{row['val_prob']:.1f}")
        v3.metric("Persistence", f"{row['val_persist']:.0f}")
        v4.metric("ADV", f"${row['adv_usd']/1e6:.1f}M")

        # ── Return comparison ──
        st.markdown("**Period Returns & Score Change**")
        ret_cols = st.columns(3)
        ret_cols[0].metric("1W Return", f"{row['ret_1w']:.2f}%",
                           delta=f"Score: {row['composite'] - row['score_1w']:+.1f}")
        ret_cols[1].metric("1M Return", f"{row['ret_1m']:.2f}%",
                           delta=f"Score: {row['composite'] - row['score_1m']:+.1f}")
        ret_cols[2].metric("3M Return", f"{row['ret_3m']:.2f}%",
                           delta=f"Score: {row['composite'] - row['score_3m']:+.1f}")

        # ── Radar chart for this ticker ──
        fig_single_radar = go.Figure(go.Scatterpolar(
            r=[row["tcs"], row["tfs"], row["oer"], row["rss"], row["tcs"]],
            theta=["TCS", "TFS", "OER", "RSS", "TCS"],
            fill="toself", fillcolor=CLASS_COLORS.get(row["classification"], C["cyan"]),
            opacity=0.4, line_color=CLASS_COLORS.get(row["classification"], C["cyan"]),
            name=sel_ticker,
        ))
        fig_single_radar.update_layout(
            polar=dict(bgcolor=C["panel"],
                       radialaxis=dict(range=[0, 100], color=C["gray"]),
                       angularaxis=dict(color=C["text"])),
            **DARK_LAYOUT, height=350, showlegend=False,
            title=f"{sel_ticker} Signal Profile",
        )

        # ── Price chart with SMA50/200 ──
        price_df = load_ticker_price(sel_ticker, period="1y")

        col_radar, col_price = st.columns([1, 2])
        with col_radar:
            st.plotly_chart(fig_single_radar, use_container_width=True)

        with col_price:
            if price_df is not None and not price_df.empty:
                close = price_df["Close"].squeeze() if isinstance(price_df["Close"], pd.DataFrame) else price_df["Close"]
                sma50 = close.rolling(50, min_periods=30).mean()
                sma200 = close.rolling(200, min_periods=100).mean()

                fig_price = go.Figure()
                fig_price.add_trace(go.Candlestick(
                    x=price_df.index,
                    open=price_df["Open"].squeeze() if isinstance(price_df["Open"], pd.DataFrame) else price_df["Open"],
                    high=price_df["High"].squeeze() if isinstance(price_df["High"], pd.DataFrame) else price_df["High"],
                    low=price_df["Low"].squeeze() if isinstance(price_df["Low"], pd.DataFrame) else price_df["Low"],
                    close=close,
                    name="Price",
                ))
                fig_price.add_trace(go.Scatter(
                    x=price_df.index, y=sma50, name="SMA50",
                    line=dict(color=C["cyan"], width=1.5),
                ))
                fig_price.add_trace(go.Scatter(
                    x=price_df.index, y=sma200, name="SMA200",
                    line=dict(color=C["orange"], width=1.5, dash="dot"),
                ))
                fig_price.update_layout(
                    **DARK_LAYOUT, height=350,
                    title=f"{sel_ticker} — 1Y Price + SMA50/200",
                    xaxis_rangeslider_visible=False, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.warning("Price data not available.")

        # ── 7-day class history ──
        if history:
            h = history.get(sel_ticker, [])
            if h:
                st.markdown("**7-Day History**")
                hist_df = pd.DataFrame(h)
                if "date" in hist_df.columns:
                    hist_df["date"] = pd.to_datetime(hist_df["date"]).dt.strftime("%m-%d")
                    hist_df["class_short"] = hist_df["class"].map(
                        lambda x: CLASS_SHORT.get(x, x[:6]))
                    st.dataframe(hist_df[["date", "composite", "tcs", "tfs", "oer",
                                          "class_short", "eligible"]],
                                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SIGNAL DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────
with tab_signals:
    st.subheader("3-Axis Signal Decomposition")

    fig_scatter = px.scatter(
        fdf, x="tcs", y="tfs", size="composite",
        color="classification", color_discrete_map=CLASS_COLORS,
        hover_data=["ticker", "name", "oer", "rss", "composite"],
        title="TCS vs TFS (size = Composite)", size_max=20,
    )
    fig_scatter.update_layout(**DARK_LAYOUT, height=550,
                               xaxis_title="TCS (Trend Continuation)",
                               yaxis_title="TFS (Trend Formation)")
    st.plotly_chart(fig_scatter, use_container_width=True)

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
            polar=dict(bgcolor=C["panel"],
                       radialaxis=dict(range=[0, 100], color=C["gray"]),
                       angularaxis=dict(color=C["text"])),
            **DARK_LAYOUT, height=500, showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.subheader("Overextension Risk (OER) vs RSI")
    fig_oer = px.scatter(
        fdf, x="rsi", y="sma50_dist", color="oer",
        color_continuous_scale="YlOrRd",
        hover_data=["ticker", "name", "classification"],
        title="RSI vs SMA50 Distance (color = OER)",
    )
    fig_oer.update_layout(**DARK_LAYOUT, height=450,
                           xaxis_title="RSI-14", yaxis_title="SMA50 Distance (%)")
    fig_oer.add_hline(y=0, line_dash="dash", line_color=C["gray"])
    fig_oer.add_vline(x=70, line_dash="dash", line_color=C["yellow"],
                      annotation_text="RSI 70")
    st.plotly_chart(fig_oer, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: CATEGORY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_category:
    st.subheader("Category Breakdown")

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

    fig_cat = px.bar(
        cat_agg.sort_values("avg_comp", ascending=True),
        y="category", x="avg_comp", orientation="h",
        color="avg_comp", color_continuous_scale="Viridis",
        text="eligible", title="Avg Composite by Category (label = eligible count)",
    )
    fig_cat.update_layout(**DARK_LAYOUT, height=max(350, len(cat_agg) * 28))
    fig_cat.add_vline(x=55, line_dash="dot", line_color=C["orange"])
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Returns by Category")
    ret_melt = cat_agg.melt(
        id_vars="category",
        value_vars=["avg_ret_1w", "avg_ret_1m", "avg_ret_3m"],
        var_name="period", value_name="return_%",
    )
    ret_melt["period"] = ret_melt["period"].map(
        {"avg_ret_1w": "1 Week", "avg_ret_1m": "1 Month", "avg_ret_3m": "3 Month"})
    fig_ret = px.bar(
        ret_melt, x="category", y="return_%", color="period",
        barmode="group", title="Average Returns by Category & Period",
        color_discrete_sequence=[C["cyan"], C["blue"], C["purple"]],
    )
    fig_ret.update_layout(**DARK_LAYOUT, xaxis_tickangle=-45, height=450)
    fig_ret.add_hline(y=0, line_dash="dash", line_color=C["gray"])
    st.plotly_chart(fig_ret, use_container_width=True)

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
# TAB 6: MARKET BREADTH (NEW) + Classification Change Tracker
# ─────────────────────────────────────────────────────────────────────────────
with tab_breadth:
    st.subheader("Market Breadth")

    # ── Breadth KPIs ──
    total = len(fdf) or 1
    n_up = int((fdf["sma50_dist"] > 0).sum())
    n_down = int((fdf["classification"] == "⬇️ DOWNTREND").sum())
    n_overext = int((fdf["classification"] == "🟡 OVEREXTENDED").sum())
    n_form = int((fdf["classification"] == "🔵 FORMATION").sum())

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Above SMA50", f"{n_up}/{total}", delta=f"{n_up/total*100:.0f}%")
    b2.metric("DOWNTREND", n_down, delta=f"{n_down/total*100:.0f}%", delta_color="inverse")
    b3.metric("OVEREXTENDED", n_overext, delta=f"{n_overext/total*100:.0f}%", delta_color="inverse")
    b4.metric("FORMATION", n_form, delta=f"{n_form/total*100:.0f}%")

    # ── SMA50 breadth by category ──
    breadth_cat = fdf.groupby("category").agg(
        total=("ticker", "size"),
        above_sma50=("sma50_dist", lambda x: (x > 0).sum()),
    ).reset_index()
    breadth_cat["pct_above"] = (breadth_cat["above_sma50"] / breadth_cat["total"] * 100).round(1)

    fig_breadth = px.bar(
        breadth_cat.sort_values("pct_above", ascending=True),
        y="category", x="pct_above", orientation="h",
        color="pct_above", color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        title="% Above SMA50 by Category",
        text="pct_above",
    )
    fig_breadth.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig_breadth.update_layout(**DARK_LAYOUT, height=max(350, len(breadth_cat) * 28),
                               xaxis=dict(range=[0, 110]))
    fig_breadth.add_vline(x=50, line_dash="dot", line_color=C["gray"])
    st.plotly_chart(fig_breadth, use_container_width=True)

    # ── Classification mix by category (stacked bar) ──
    st.subheader("Classification Mix by Category")
    cls_by_cat = fdf.groupby(["category", "classification"]).size().reset_index(name="count")
    fig_stack = px.bar(
        cls_by_cat, x="category", y="count", color="classification",
        color_discrete_map=CLASS_COLORS, barmode="stack",
        title="Classification Composition per Category",
    )
    fig_stack.update_layout(**DARK_LAYOUT, xaxis_tickangle=-45, height=450)
    st.plotly_chart(fig_stack, use_container_width=True)

    # ── Classification Change Tracker ──
    st.subheader("Classification Change Tracker (7-Day)")
    if history:
        upgrades, downgrades = [], []
        class_rank = {"⬇️ DOWNTREND": 0, "🟠 NEUTRAL": 1, "🟤 EXHAUSTING": 1,
                      "🔵 FORMATION": 2, "🟡 OVEREXTENDED": 2, "🟢 CONTINUATION": 3}
        for t in fdf["ticker"]:
            h = history.get(t, [])
            if len(h) < 2:
                continue
            first_cls = h[0].get("class", "")
            last_cls = h[-1].get("class", "")
            if first_cls == last_cls:
                continue
            name = fdf.loc[fdf["ticker"] == t, "name"].iloc[0]
            r_old = class_rank.get(first_cls, 1)
            r_new = class_rank.get(last_cls, 1)
            entry = {
                "Ticker": t, "Name": name,
                "From": CLASS_SHORT.get(first_cls, first_cls[:8]),
                "To": CLASS_SHORT.get(last_cls, last_cls[:8]),
            }
            if r_new > r_old:
                upgrades.append(entry)
            elif r_new < r_old:
                downgrades.append(entry)

        col_up, col_dn = st.columns(2)
        with col_up:
            st.markdown(f"**Upgrades ({len(upgrades)})**")
            if upgrades:
                st.dataframe(pd.DataFrame(upgrades), use_container_width=True, hide_index=True)
            else:
                st.caption("None in past 7 days")
        with col_dn:
            st.markdown(f"**Downgrades ({len(downgrades)})**")
            if downgrades:
                st.dataframe(pd.DataFrame(downgrades), use_container_width=True, hide_index=True)
            else:
                st.caption("None in past 7 days")
    else:
        st.info("7-day history not available in cache.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: PORTFOLIO VIEW (NEW)
# ─────────────────────────────────────────────────────────────────────────────
with tab_portfolio:
    st.subheader("Portfolio Construction View")

    port = fdf[fdf["eligible"]].copy()
    if port.empty:
        st.info("No eligible tickers in current filter.")
    else:
        # ── Summary KPIs ──
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Eligible count", len(port))
        p2.metric("Avg Composite", f"{port['composite'].mean():.1f}")
        p3.metric("Avg OER (risk)", f"{port['oer'].mean():.0f}")
        p4.metric("Avg RSI", f"{port['rsi'].mean():.1f}")

        col_pie, col_bar = st.columns(2)

        # ── Sector allocation ──
        with col_pie:
            alloc = port["category"].value_counts().reset_index()
            alloc.columns = ["category", "count"]
            fig_alloc = px.pie(
                alloc, values="count", names="category",
                title="Portfolio Sector Allocation", hole=0.35,
            )
            fig_alloc.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_alloc, use_container_width=True)

        # ── Classification mix of portfolio ──
        with col_bar:
            pcls = port["classification"].value_counts().reset_index()
            pcls.columns = ["classification", "count"]
            fig_pcls = px.bar(
                pcls, x="classification", y="count",
                color="classification", color_discrete_map=CLASS_COLORS,
                title="Portfolio Classification Mix",
            )
            fig_pcls.update_layout(**DARK_LAYOUT, showlegend=False)
            st.plotly_chart(fig_pcls, use_container_width=True)

        # ── Risk scatter: OER vs RSI for portfolio ──
        st.subheader("Portfolio Risk Map")
        fig_risk = px.scatter(
            port, x="rsi", y="oer", color="classification",
            color_discrete_map=CLASS_COLORS,
            size="composite", size_max=20,
            hover_data=["ticker", "name", "composite", "trend_age"],
            title="RSI vs OER (Eligible Only)",
        )
        fig_risk.update_layout(**DARK_LAYOUT, height=450,
                                xaxis_title="RSI-14", yaxis_title="OER Score")
        fig_risk.add_hline(y=60, line_dash="dash", line_color=C["yellow"],
                           annotation_text="OER=60 (Overextended)")
        fig_risk.add_vline(x=70, line_dash="dash", line_color=C["yellow"],
                           annotation_text="RSI 70")
        st.plotly_chart(fig_risk, use_container_width=True)

        # ── Eligible list with rankings ──
        st.subheader("Full Eligible List")
        port_display = port[["ticker", "name", "category", "composite", "tcs",
                             "tfs", "oer", "rss", "classification",
                             "val_prob", "val_persist",
                             "rsi", "trend_age", "ret_1w", "ret_1m", "ret_3m"]].copy()
        port_display["adv_M"] = (port["adv_usd"] / 1e6).round(1)
        st.dataframe(port_display, use_container_width=True, hide_index=True, height=500)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 8: SIGNAL VALIDITY
# ─────────────────────────────────────────────────────────────────────────────
with tab_validity:
    st.subheader("Signal Validity Verification (Past 1-Month Backtest)")

    if ve_stats:
        col_b, col_c = st.columns(2)

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
            fig_bkt.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_bkt, use_container_width=True)

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
            fig_cls.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_cls, use_container_width=True)

        st.subheader("Class Transition Matrix")
        classes = ["⬇️ DOWNTREND", "🟠 NEUTRAL", "🔵 FORMATION",
                   "🟢 CONTINUATION", "🟡 OVEREXTENDED", "🟤 EXHAUSTING"]
        shorts = [CLASS_SHORT[c] for c in classes]
        trans_matrix = []
        for cf in classes:
            tot = ve_stats["transition_totals"].get(cf, 0)
            row = {}
            for ct in classes:
                cnt = ve_stats["transitions"].get((cf, ct), 0)
                row[CLASS_SHORT[ct]] = round(cnt / tot * 100, 1) if tot > 0 else 0.0
            trans_matrix.append(row)

        tdf = pd.DataFrame(trans_matrix, index=shorts)
        fig_heat = px.imshow(
            tdf.values, x=shorts, y=shorts,
            color_continuous_scale="Blues", text_auto=".1f",
            title="Transition Probability (%, row -> col)",
        )
        fig_heat.update_layout(**DARK_LAYOUT, xaxis_title="To", yaxis_title="From", height=400)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Validity engine data not available.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 9: 7-DAY HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("7-Day Composite Trend")

    eligible_tickers = fdf[fdf["eligible"]]["ticker"].tolist()[:30]
    if not eligible_tickers:
        eligible_tickers = fdf["ticker"].tolist()[:20]

    sel_tickers = st.multiselect(
        "Select tickers to chart", fdf["ticker"].tolist(),
        default=eligible_tickers[:15],
    )

    if sel_tickers and history:
        traces = []
        for t in sel_tickers:
            h = history.get(t, [])
            if len(h) < 2:
                continue
            traces.append(go.Scatter(
                x=[x["date"] for x in h], y=[x["composite"] for x in h],
                mode="lines+markers", name=t, marker=dict(size=5),
            ))
        if traces:
            fig_trend = go.Figure(data=traces)
            fig_trend.add_hline(y=55, line_dash="dash", line_color=C["orange"],
                                annotation_text="Eligible threshold (55)")
            fig_trend.update_layout(**DARK_LAYOUT, height=500,
                                     xaxis_title="Date", yaxis_title="Composite",
                                     legend=dict(font_size=10))
            st.plotly_chart(fig_trend, use_container_width=True)

    # CONTINUATION class trend
    cont_tickers = fdf[fdf["classification"].str.contains("CONTINUATION")]["ticker"].tolist()
    if cont_tickers and history:
        st.subheader("CONTINUATION Class — 1-Week Trend Tracking")
        trend_data = []
        for t in cont_tickers:
            h = history.get(t, [])
            cls_trend = " -> ".join(CLASS_SHORT.get(x["class"], "?") for x in h)
            name = fdf.loc[fdf["ticker"] == t, "name"].iloc[0] if len(fdf[fdf["ticker"] == t]) else ""
            trend_data.append({"Ticker": t, "Name": name, "Class Trend": cls_trend})
        st.dataframe(pd.DataFrame(trend_data), use_container_width=True, hide_index=True)

    # Score change analysis
    st.subheader("Score Change (1W / 1M / 3M)")
    change_df = fdf[["ticker", "name", "composite", "score_1w", "score_1m", "score_3m"]].copy()
    change_df["d1W"] = change_df["composite"] - change_df["score_1w"]
    change_df["d1M"] = change_df["composite"] - change_df["score_1m"]
    change_df["d3M"] = change_df["composite"] - change_df["score_3m"]

    sort_col = st.selectbox("Sort by", ["d1W", "d1M", "d3M"], index=0,
                            format_func=lambda x: x.replace("d", "Delta "))
    ascending = st.toggle("Ascending (worst first)", value=False)
    change_sorted = change_df.sort_values(sort_col, ascending=ascending).head(30)

    fig_change = go.Figure(go.Bar(
        y=change_sorted["ticker"], x=change_sorted[sort_col],
        orientation="h",
        marker_color=[C["green"] if v >= 0 else C["red"] for v in change_sorted[sort_col]],
        text=change_sorted[sort_col].round(1), textposition="outside",
    ))
    fig_change.update_layout(
        **DARK_LAYOUT, xaxis_title=f"Composite Change ({sort_col})",
        yaxis=dict(autorange="reversed"),
        height=max(350, len(change_sorted) * 22),
        margin=dict(l=0, r=80),
        title=f"Top {len(change_sorted)} — Composite Score Change",
    )
    fig_change.add_vline(x=0, line_color=C["gray"])
    st.plotly_chart(fig_change, use_container_width=True)
