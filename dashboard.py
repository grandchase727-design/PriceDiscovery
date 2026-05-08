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
    # 3x3 Matrix base classes
    "🟢 CONTINUATION": C["green"],
    "🔵 RECOVERY": C["blue"],
    "🟣 COUNTER_RALLY": C["purple"],
    "🟡 CONSOLIDATION": C["yellow"],
    "🟠 NEUTRAL": C["orange"],
    "🟤 FADING": C["brown"],
    "🔶 PULLBACK": "#f97316",
    "⚠️ WEAKENING": "#f59e0b",
    "⬇️ DOWNTREND": C["red"],
    # Override classes
    "🟡 OVEREXTENDED": C["yellow"],
    "🔵 FORMATION": C["blue"],
    "🟤 EXHAUSTING": C["brown"],
    "🔴 CYCLE_PEAK": C["red"],
    "🟦 LAGGING_CATCHUP": "#3b82f6",
}

CLASS_SHORT = {
    # 3x3 Matrix base classes
    "⬇️ DOWNTREND": "DOWN", "🟤 FADING": "FADING", "🟣 COUNTER_RALLY": "CNTR",
    "⚠️ WEAKENING": "WEAK", "🟠 NEUTRAL": "NEUTRAL", "🟡 CONSOLIDATION": "CONSOL",
    "🔶 PULLBACK": "PULL", "🔵 RECOVERY": "RECV",
    "🟢 CONTINUATION": "CONT",
    # Override classes
    "🔵 FORMATION": "FORM", "🟡 OVEREXTENDED": "OVEXT", "🟤 EXHAUSTING": "EXHAUST",
    "🔴 CYCLE_PEAK": "CPEAK", "🟦 LAGGING_CATCHUP": "LAGCU",
}

DARK_LAYOUT = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["panel"], font_color=C["text"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — cache-first, live-scan only when requested
# ═══════════════════════════════════════════════════════════════════════════════
EXPECTED_CACHE_VERSION = 3

def load_from_cache():
    """Load pre-computed results from .scan_cache.pkl (written by run_scan)."""
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        # Cache version check — reject stale v1 caches
        cv = cache.get("cache_version", 1)
        if cv < EXPECTED_CACHE_VERSION:
            st.warning(f"Cache version {cv} is outdated (expected {EXPECTED_CACHE_VERSION}). "
                       "Please re-run `python3 price_discovery.py` to regenerate.")
            return None
        results = cache["results"]
        df = pd.DataFrame(results)
        # Backfill new dual-timeframe fields for backward compat
        for col in ("tcs_short", "tcs_long", "tfs_short", "tfs_long",
                     "rss_short", "rss_long"):
            if col not in df.columns:
                df[col] = 0.0
        history = cache.get("history", {})
        ve_stats = {
            "bucket": cache.get("ve_bucket", {}),
            "class": cache.get("ve_class", {}),
            "transitions": cache.get("ve_transitions", {}),
            "transition_totals": cache.get("ve_transition_totals", {}),
            "observations": cache.get("ve_observations", []),
            "fwd_bucket": cache.get("ve_fwd_bucket", {}),
            "fwd_class": cache.get("ve_fwd_class", {}),
            "fwd_eligible": cache.get("ve_fwd_eligible", {}),
            "transition_hit": cache.get("ve_transition_hit", {}),
            "score_weighted": cache.get("ve_score_weighted", {}),
        }
        graph_data = cache.get("graph", {})
        factor_efficacy = cache.get("factor_efficacy", {})
        scan_time = cache.get("scan_time", "unknown")
        return df, results, history, ve_stats, scan_time, graph_data, factor_efficacy
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
    return df, results, {}, {}, datetime.today().isoformat(), {}, {}

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
    df, results, history, ve_stats, scan_time, graph_data, factor_efficacy = loaded
else:
    cached = load_from_cache()
    if cached:
        df, results, history, ve_stats, scan_time, graph_data, factor_efficacy = cached
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

# ── attach theme (consolidated macro themes) ──
try:
    from price_discovery import STOCK_THEMES_CONSOLIDATED, STOCK_THEMES
    fdf["theme"] = fdf["ticker"].map(STOCK_THEMES_CONSOLIDATED).fillna("-")
    df["theme"] = df["ticker"].map(STOCK_THEMES_CONSOLIDATED).fillna("-")
    fdf["theme_detail"] = fdf["ticker"].map(STOCK_THEMES).fillna("-")
    df["theme_detail"] = df["ticker"].map(STOCK_THEMES).fillna("-")
except ImportError:
    fdf["theme"] = "-"; df["theme"] = "-"
    fdf["theme_detail"] = "-"; df["theme_detail"] = "-"


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER & TABS
# ═══════════════════════════════════════════════════════════════════════════════
st.title("Price Discovery Scanner v5.0")
st.caption(f"Universe: {len(df)} tickers | Filtered: {len(fdf)} | "
           f"Scan: {df['data_as_of'].iloc[0] if not df.empty else 'N/A'}")

(tab_overview, tab_table, tab_deep, tab_signals, tab_category,
 tab_theme, tab_breadth, tab_portfolio, tab_effectiveness, tab_validity,
 tab_history, tab_graph, tab_factor, tab_ai_pred, tab_report) = st.tabs([
    "Overview", "Master Table", "Ticker Deep Dive",
    "Signal Decomposition", "Category Analysis", "Theme Analysis",
    "Market Breadth", "Portfolio View",
    "Signal Effectiveness", "Signal Validity", "7-Day History",
    "Graph Analysis", "Factor Efficacy", "AI Prediction", "Report",
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
                lambda r: f"TCS:{r['tcs']}({r.get('tcs_short','-')}/{r.get('tcs_long','-')}) "
                           f"TFS:{r['tfs']} OER:{r['oer']} RSS:{r['rss']:.0f}",
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

    # Build display columns — include sub-scores if available
    base_cols = [
        "ticker", "name", "category", "composite", "tcs", "tfs", "oer", "rss",
        "classification", "eligible", "rejection",
        "rsi", "trend_age", "sma50_dist",
        "val_prob", "val_persist", "val_conf",
        "score_1w", "ret_1w", "score_1m", "ret_1m", "score_3m", "ret_3m",
    ]
    sub_score_cols = ["tcs_short", "tcs_long", "tfs_short", "tfs_long",
                      "rss_short", "rss_long"]
    display_cols = base_cols + [c for c in sub_score_cols if c in fdf.columns]
    show_df = fdf[display_cols].copy()
    show_df["adv_M"] = (fdf["adv_usd"] / 1e6).round(1)

    _col_cfg = {
        "composite": st.column_config.NumberColumn("Comp", format="%.1f"),
        "tcs": st.column_config.ProgressColumn("TCS", min_value=0, max_value=100, format="%d"),
        "tfs": st.column_config.ProgressColumn("TFS", min_value=0, max_value=100, format="%d"),
        "oer": st.column_config.ProgressColumn("OER", min_value=0, max_value=100, format="%d"),
        "rss": st.column_config.NumberColumn("RSS", format="%.1f"),
        "tcs_short": st.column_config.NumberColumn("TCS_S", format="%.0f"),
        "tcs_long": st.column_config.NumberColumn("TCS_L", format="%.0f"),
        "tfs_short": st.column_config.NumberColumn("TFS_S", format="%.0f"),
        "tfs_long": st.column_config.NumberColumn("TFS_L", format="%.0f"),
        "rss_short": st.column_config.NumberColumn("RSS_S", format="%.1f"),
        "rss_long": st.column_config.NumberColumn("RSS_L", format="%.1f"),
        "val_prob": st.column_config.NumberColumn("Val%", format="%.1f"),
        "val_persist": st.column_config.NumberColumn("Persist", format="%.0f"),
        "ret_1w": st.column_config.NumberColumn("1W Ret%", format="%.2f%%"),
        "ret_1m": st.column_config.NumberColumn("1M Ret%", format="%.2f%%"),
        "ret_3m": st.column_config.NumberColumn("3M Ret%", format="%.2f%%"),
        "sma50_dist": st.column_config.NumberColumn("SMA50 Dist%", format="%.2f"),
        "adv_M": st.column_config.NumberColumn("ADV ($M)", format="%.1f"),
    }

    st.dataframe(show_df, use_container_width=True, height=700,
                 column_config=_col_cfg, hide_index=True)

    # ── ETF / Stock split tables ──
    is_stock = show_df["category"].str.startswith("STK_")
    etf_df = show_df[~is_stock]
    stk_df = show_df[is_stock]

    st.divider()
    st.subheader(f"ETF ({len(etf_df)})")
    st.dataframe(etf_df, use_container_width=True, height=700,
                 column_config=_col_cfg, hide_index=True)

    st.divider()
    st.subheader(f"Stocks ({len(stk_df)})")
    st.dataframe(stk_df, use_container_width=True, height=700,
                 column_config=_col_cfg, hide_index=True)


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
        g2.metric("TCS", int(row["tcs"]),
                  help=f"Short: {row.get('tcs_short', '-')}  Long: {row.get('tcs_long', '-')}")
        g3.metric("TFS", int(row["tfs"]),
                  help=f"Short: {row.get('tfs_short', '-')}  Long: {row.get('tfs_long', '-')}")
        g4.metric("OER", int(row["oer"]))
        g5.metric("RSS", f"{row['rss']:.0f}",
                  help=f"Short: {row.get('rss_short', '-')}  Long: {row.get('rss_long', '-')}")

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

        # ── Radar chart for this ticker (6-axis dual-timeframe) ──
        _tcs_s = row.get("tcs_short", row["tcs"])
        _tcs_l = row.get("tcs_long", row["tcs"])
        _tfs_s = row.get("tfs_short", row["tfs"])
        _tfs_l = row.get("tfs_long", row["tfs"])
        _rss_s = row.get("rss_short", row.get("rss", 50))
        _rss_l = row.get("rss_long", row.get("rss", 50))
        radar_r = [_tcs_s, _tcs_l, _tfs_s, _tfs_l, row["oer"], _rss_s, _tcs_s]
        radar_theta = ["TCS_S", "TCS_L", "TFS_S", "TFS_L", "OER", "RSS_S", "TCS_S"]
        fig_single_radar = go.Figure(go.Scatterpolar(
            r=radar_r, theta=radar_theta,
            fill="toself", fillcolor=CLASS_COLORS.get(row["classification"], C["cyan"]),
            opacity=0.4, line_color=CLASS_COLORS.get(row["classification"], C["cyan"]),
            name=sel_ticker,
        ))
        fig_single_radar.update_layout(
            polar=dict(bgcolor=C["panel"],
                       radialaxis=dict(range=[0, 100], color=C["gray"]),
                       angularaxis=dict(color=C["text"])),
            **DARK_LAYOUT, height=350, showlegend=False,
            title=f"{sel_ticker} Signal Profile (Dual-Timeframe)",
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
    st.subheader("Dual-Timeframe Signal Decomposition")

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

    st.subheader("Signal Radar — Top Eligible (Dual-Timeframe)")
    top_radar = fdf[fdf["eligible"]].head(10)
    if not top_radar.empty:
        fig_radar = go.Figure()
        cats_radar = ["TCS_S", "TCS_L", "TFS_S", "TFS_L", "OER", "RSS_S", "TCS_S"]
        for _, r in top_radar.iterrows():
            vals = [r.get("tcs_short", r["tcs"]), r.get("tcs_long", r["tcs"]),
                    r.get("tfs_short", r["tfs"]), r.get("tfs_long", r["tfs"]),
                    r["oer"],
                    r.get("rss_short", r.get("rss", 50)),
                    r.get("tcs_short", r["tcs"])]
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
# TAB 6: THEME ANALYSIS (NEW)
# ─────────────────────────────────────────────────────────────────────────────
with tab_theme:
    st.subheader("Theme Analysis")
    st.caption("Stock 전체를 테마 기준으로 분류 — 카테고리에 관계없이 동일 테마 종목을 한눈에 비교")

    themed = fdf[fdf["theme"] != "-"].copy()
    if themed.empty:
        st.info("Theme 데이터가 없습니다 (ETF에는 theme이 할당되지 않음). Stocks 포함 필터를 확인하세요.")
    else:
        # ── Min ticker filter ──
        min_n = st.slider("Minimum tickers per theme", 1, 10, 2)

        # ── Theme summary table (all stocks, no category split) ──
        theme_agg = themed.groupby("theme").agg(
            n=("ticker", "size"),
            eligible=("eligible", "sum"),
            avg_comp=("composite", "mean"),
            avg_tcs=("tcs", "mean"),
            avg_tfs=("tfs", "mean"),
            avg_oer=("oer", "mean"),
            avg_rsi=("rsi", "mean"),
            avg_ret_1w=("ret_1w", "mean"),
            avg_ret_1m=("ret_1m", "mean"),
            avg_ret_3m=("ret_3m", "mean"),
            categories=("category", lambda x: ", ".join(sorted(set(
                c.replace("STK_", "") for c in x)))),
        ).round(1).reset_index()
        theme_agg["eligible"] = theme_agg["eligible"].astype(int)
        theme_agg = theme_agg[theme_agg["n"] >= min_n].sort_values("avg_comp", ascending=False)

        st.dataframe(theme_agg, use_container_width=True, hide_index=True,
                     column_config={
                         "avg_ret_1w": st.column_config.NumberColumn("1W%", format="%.2f%%"),
                         "avg_ret_1m": st.column_config.NumberColumn("1M%", format="%.2f%%"),
                         "avg_ret_3m": st.column_config.NumberColumn("3M%", format="%.2f%%"),
                         "categories": st.column_config.TextColumn("Source Categories"),
                     })

        col_tbar, col_tret = st.columns(2)

        # ── Theme avg composite bar ──
        with col_tbar:
            fig_tbar = px.bar(
                theme_agg.sort_values("avg_comp", ascending=True),
                y="theme", x="avg_comp", orientation="h",
                color="avg_comp", color_continuous_scale="Viridis",
                text="n",
                title="Avg Composite by Theme (label = ticker count)",
            )
            fig_tbar.update_layout(**DARK_LAYOUT,
                                    height=max(350, len(theme_agg) * 25))
            fig_tbar.add_vline(x=55, line_dash="dot", line_color=C["orange"])
            st.plotly_chart(fig_tbar, use_container_width=True)

        # ── Theme returns comparison ──
        with col_tret:
            tret_melt = theme_agg.melt(
                id_vars="theme",
                value_vars=["avg_ret_1w", "avg_ret_1m", "avg_ret_3m"],
                var_name="period", value_name="return_%",
            )
            tret_melt["period"] = tret_melt["period"].map(
                {"avg_ret_1w": "1W", "avg_ret_1m": "1M", "avg_ret_3m": "3M"})
            fig_tret = px.bar(
                tret_melt, x="theme", y="return_%", color="period",
                barmode="group", title="Returns by Theme & Period",
                color_discrete_sequence=[C["cyan"], C["blue"], C["purple"]],
            )
            fig_tret.update_layout(**DARK_LAYOUT, xaxis_tickangle=-45,
                                    height=max(350, len(theme_agg) * 25))
            fig_tret.add_hline(y=0, line_dash="dash", line_color=C["gray"])
            st.plotly_chart(fig_tret, use_container_width=True)

        # ── Classification mix by theme (stacked bar) ──
        st.subheader("Classification by Theme")
        themed_filtered = themed[themed["theme"].isin(theme_agg["theme"])]
        cls_theme = themed_filtered.groupby(["theme", "classification"]).size().reset_index(name="count")
        fig_cls_theme = px.bar(
            cls_theme, x="theme", y="count", color="classification",
            color_discrete_map=CLASS_COLORS, barmode="stack",
        )
        fig_cls_theme.update_layout(**DARK_LAYOUT, xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig_cls_theme, use_container_width=True)

        # ── Theme scatter: Composite vs 1M Return ──
        st.subheader("Theme Momentum Map")
        fig_tmom = px.scatter(
            themed_filtered, x="composite", y="ret_1m",
            color="theme", size="adv_usd", size_max=25,
            hover_data=["ticker", "name", "category", "classification", "tcs", "tfs"],
            title="Composite vs 1M Return (size = ADV, color = Theme)",
        )
        fig_tmom.update_layout(**DARK_LAYOUT, height=550,
                                xaxis_title="Composite Score",
                                yaxis_title="1M Return %")
        fig_tmom.add_vline(x=55, line_dash="dot", line_color=C["orange"])
        fig_tmom.add_hline(y=0, line_dash="dash", line_color=C["gray"])
        st.plotly_chart(fig_tmom, use_container_width=True)

        # ── Per-theme ticker detail ──
        st.subheader("Ticker Detail by Theme")
        detail_cols = ["ticker", "name", "category", "theme_detail", "composite",
                       "tcs", "tfs", "oer", "rss",
                       "classification", "eligible", "rsi", "trend_age",
                       "ret_1w", "ret_1m", "ret_3m"]
        # Only include columns that exist
        detail_cols = [c for c in detail_cols if c in themed_filtered.columns]
        for theme in theme_agg["theme"]:
            tdf = themed_filtered[themed_filtered["theme"] == theme].sort_values("composite", ascending=False)
            n_el = int(tdf["eligible"].sum())
            cats = ", ".join(sorted(set(c.replace("STK_", "") for c in tdf["category"])))
            with st.expander(f"{theme}  ({len(tdf)} tickers, {n_el} eligible) — [{cats}]"):
                st.dataframe(tdf[detail_cols], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 7: MARKET BREADTH + Classification Change Tracker
# ─────────────────────────────────────────────────────────────────────────────
with tab_breadth:
    st.subheader("Market Breadth")

    # ── Breadth KPIs ──
    total = len(fdf) or 1
    n_up = int((fdf["sma50_dist"] > 0).sum())
    n_down = int((fdf["classification"] == "⬇️ DOWNTREND").sum())
    n_cont = int((fdf["classification"] == "🟢 CONTINUATION").sum())
    n_overext = int((fdf["classification"] == "🟡 OVEREXTENDED").sum())
    n_weak = int(fdf["classification"].isin(
        ["🟤 FADING", "⚠️ WEAKENING", "🟣 COUNTER_RALLY"]).sum())

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Above SMA50", f"{n_up}/{total}", delta=f"{n_up/total*100:.0f}%")
    b2.metric("CONTINUATION", n_cont, delta=f"{n_cont/total*100:.0f}%")
    b3.metric("DOWNTREND", n_down, delta=f"{n_down/total*100:.0f}%", delta_color="inverse")
    b4.metric("OVEREXTENDED", n_overext, delta=f"{n_overext/total*100:.0f}%", delta_color="inverse")
    b5.metric("Bearish (FADING/WEAK/CNTR)", n_weak,
              delta=f"{n_weak/total*100:.0f}%", delta_color="inverse")

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
        class_rank = {
            "⬇️ DOWNTREND": 0, "🟤 FADING": 0, "🟣 COUNTER_RALLY": 0, "🔴 CYCLE_PEAK": 0,
            "⚠️ WEAKENING": 1, "🟠 NEUTRAL": 1, "🟤 EXHAUSTING": 1, "🟡 OVEREXTENDED": 1,
            "🟡 CONSOLIDATION": 2, "🔶 PULLBACK": 2, "🔵 RECOVERY": 2,
            "🔵 FORMATION": 2, "🟦 LAGGING_CATCHUP": 2,
            "🟢 CONTINUATION": 3,
        }
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
# TAB 8: SIGNAL EFFECTIVENESS (NEW)
# ─────────────────────────────────────────────────────────────────────────────
with tab_effectiveness:
    st.subheader("Signal Effectiveness Analysis")
    st.caption("Composite score가 실제 forward return을 얼마나 잘 예측하는지 정량 평가")

    obs_list = ve_stats.get("observations", [])
    if not obs_list:
        st.warning("Observations 데이터가 캐시에 없습니다. `python3 price_discovery.py`를 다시 실행해주세요.")
    else:
        obs_df = pd.DataFrame(obs_list)

        # ══════════════════════════════════════════════════════════════════
        # 1. SUMMARY KPIs
        # ══════════════════════════════════════════════════════════════════
        from scipy import stats as sp_stats

        ic_spearman, ic_pval = sp_stats.spearmanr(obs_df["score"], obs_df["excess_return"])
        overall_hit = (obs_df["excess_return"] > 0).mean() * 100
        avg_exc = obs_df["excess_return"].mean()
        n_obs = len(obs_df)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Information Coefficient (IC)", f"{ic_spearman:.3f}",
                   help="Spearman rank correlation: score vs excess return. >0.05 = useful signal")
        k2.metric("Excess Hit Rate", f"{overall_hit:.1f}%",
                   help="% of observations that beat benchmark")
        k3.metric("Avg Excess Return", f"{avg_exc:.2f}%")
        k4.metric("Observations", f"{n_obs:,}")

        # IC interpretation
        if ic_spearman > 0.10:
            st.success(f"IC = {ic_spearman:.3f} — Strong predictive signal (p={ic_pval:.4f})")
        elif ic_spearman > 0.05:
            st.info(f"IC = {ic_spearman:.3f} — Moderate predictive signal (p={ic_pval:.4f})")
        elif ic_spearman > 0.0:
            st.warning(f"IC = {ic_spearman:.3f} — Weak positive signal (p={ic_pval:.4f})")
        else:
            st.error(f"IC = {ic_spearman:.3f} — No predictive power (p={ic_pval:.4f})")

        # ══════════════════════════════════════════════════════════════════
        # 2. IC BY EVAL DATE (Signal Consistency)
        # ══════════════════════════════════════════════════════════════════
        if "eval_date" in obs_df.columns:
            st.subheader("IC Time Series (Signal Consistency)")
            ic_by_date = []
            for d, grp in obs_df.groupby("eval_date"):
                if len(grp) >= 10:
                    ic_val, _ = sp_stats.spearmanr(grp["score"], grp["excess_return"])
                    ic_by_date.append({"date": d, "IC": ic_val, "n": len(grp)})
            if ic_by_date:
                ic_df = pd.DataFrame(ic_by_date)
                fig_ic = go.Figure()
                fig_ic.add_trace(go.Bar(
                    x=ic_df["date"], y=ic_df["IC"],
                    marker_color=[C["green"] if v > 0 else C["red"] for v in ic_df["IC"]],
                    name="IC",
                ))
                fig_ic.add_hline(y=0, line_color=C["gray"])
                fig_ic.add_hline(y=ic_spearman, line_dash="dash", line_color=C["cyan"],
                                 annotation_text=f"Avg IC={ic_spearman:.3f}")
                fig_ic.update_layout(**DARK_LAYOUT, height=350,
                                      xaxis_title="Eval Date", yaxis_title="IC (Spearman)",
                                      title="IC per Evaluation Point — 양수=시그널 유효, 음수=역행")
                st.plotly_chart(fig_ic, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════
        # 3. QUINTILE ANALYSIS
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Quintile Analysis")
        st.caption("Composite score 5분위별 평균 수익률 — 단조 증가하면 시그널 유효")

        obs_df["quintile"] = pd.qcut(obs_df["score"], 5, labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"],
                                      duplicates="drop")
        q_stats = obs_df.groupby("quintile", observed=True).agg(
            n=("score", "size"),
            avg_score=("score", "mean"),
            avg_fwd=("fwd_return", "mean"),
            avg_exc=("excess_return", "mean"),
            hit_rate=("excess_return", lambda x: (x > 0).mean() * 100),
        ).round(2).reset_index()

        col_qt, col_qc = st.columns(2)

        with col_qt:
            st.dataframe(q_stats, use_container_width=True, hide_index=True)

        with col_qc:
            fig_q = go.Figure()
            fig_q.add_trace(go.Bar(
                x=q_stats["quintile"], y=q_stats["avg_fwd"],
                name="Avg Forward Return", marker_color=C["blue"],
            ))
            fig_q.add_trace(go.Bar(
                x=q_stats["quintile"], y=q_stats["avg_exc"],
                name="Avg Excess Return", marker_color=C["cyan"],
            ))
            fig_q.update_layout(**DARK_LAYOUT, barmode="group", height=350,
                                 title="Quintile Returns",
                                 yaxis_title="Return %")
            fig_q.add_hline(y=0, line_color=C["gray"])
            st.plotly_chart(fig_q, use_container_width=True)

        # Long-Short spread
        if len(q_stats) >= 2:
            ls_spread = q_stats["avg_exc"].iloc[-1] - q_stats["avg_exc"].iloc[0]
            mono_check = all(q_stats["avg_exc"].iloc[i] <= q_stats["avg_exc"].iloc[i+1]
                             for i in range(len(q_stats)-1))
            s1, s2 = st.columns(2)
            s1.metric("Q5-Q1 Spread (Long-Short)", f"{ls_spread:.2f}%",
                      help="Top quintile excess return minus bottom quintile")
            s2.metric("Monotonic", "Yes" if mono_check else "No",
                      help="수익률이 quintile 순서대로 단조 증가하는지 여부")

        # ══════════════════════════════════════════════════════════════════
        # 4. HIT RATE CURVE
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Hit Rate Curve")
        st.caption("Composite threshold별 해당 점수 이상 종목의 벤치마크 초과 확률")

        thresholds = list(range(10, 95, 5))
        hr_data = []
        for th in thresholds:
            sub = obs_df[obs_df["score"] >= th]
            if len(sub) >= 5:
                hr_data.append({
                    "threshold": th,
                    "hit_rate": (sub["excess_return"] > 0).mean() * 100,
                    "avg_exc": sub["excess_return"].mean(),
                    "n": len(sub),
                })
        if hr_data:
            hr_df = pd.DataFrame(hr_data)
            fig_hr = make_subplots(specs=[[{"secondary_y": True}]])
            fig_hr.add_trace(go.Scatter(
                x=hr_df["threshold"], y=hr_df["hit_rate"],
                mode="lines+markers", name="Excess Hit Rate %",
                line=dict(color=C["green"], width=2), marker=dict(size=6),
            ), secondary_y=False)
            fig_hr.add_trace(go.Bar(
                x=hr_df["threshold"], y=hr_df["avg_exc"],
                name="Avg Excess Ret %", marker_color=C["cyan"], opacity=0.5,
            ), secondary_y=True)
            fig_hr.add_hline(y=50, line_dash="dot", line_color=C["gray"],
                             annotation_text="50% (random)")
            fig_hr.update_layout(**DARK_LAYOUT, height=400,
                                  title="Score Threshold vs Hit Rate & Excess Return",
                                  xaxis_title="Minimum Composite Score")
            fig_hr.update_yaxes(title_text="Hit Rate %", secondary_y=False)
            fig_hr.update_yaxes(title_text="Avg Excess %", secondary_y=True)
            st.plotly_chart(fig_hr, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════
        # 5. CLASSIFICATION EFFECTIVENESS (Box Plot)
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Classification Effectiveness")
        st.caption("각 분류가 의도대로 작동하는지 — CONT/FORM은 양의 초과수익, DOWN은 음의 수익 예상")

        obs_df["cls_short"] = obs_df["classification"].map(CLASS_SHORT).fillna(
            obs_df["classification"].str.split(" ", n=1).str[-1].str[:6])
        cls_order = ["DOWN", "FADING", "CNTR", "WEAK", "NEUTRAL", "EXHAUST",
                     "CONSOL", "PULL", "RECV", "FORM", "OVEXT", "CONT"]
        # Keep only categories that exist in the data
        cls_order = [c for c in cls_order if c in obs_df["cls_short"].values]
        obs_df["cls_short"] = pd.Categorical(obs_df["cls_short"], categories=cls_order, ordered=True)

        fig_box = px.box(
            obs_df.sort_values("cls_short"), x="cls_short", y="excess_return",
            color="cls_short",
            color_discrete_map={v: CLASS_COLORS.get(k, C["gray"]) for k, v in CLASS_SHORT.items()},
            title="Excess Return Distribution by Classification",
        )
        fig_box.update_layout(**DARK_LAYOUT, height=450, showlegend=False,
                               xaxis_title="Classification", yaxis_title="Excess Return %")
        fig_box.add_hline(y=0, line_dash="dash", line_color=C["gray"])
        st.plotly_chart(fig_box, use_container_width=True)

        # Classification summary table
        cls_eff = obs_df.groupby("cls_short", observed=True).agg(
            n=("score", "size"),
            avg_score=("score", "mean"),
            avg_fwd=("fwd_return", "mean"),
            avg_exc=("excess_return", "mean"),
            hit_rate=("excess_return", lambda x: (x > 0).mean() * 100),
            median_exc=("excess_return", "median"),
            std_exc=("excess_return", "std"),
        ).round(2).reset_index()
        cls_eff.columns = ["Class", "N", "Avg Score", "Avg Fwd%", "Avg Exc%",
                           "Hit Rate%", "Median Exc%", "Std Exc%"]
        st.dataframe(cls_eff, use_container_width=True, hide_index=True)

        # ══════════════════════════════════════════════════════════════════
        # 6. SCORE vs FORWARD RETURN (Scatter + Regression)
        # ══════════════════════════════════════════════════════════════════
        st.subheader("Score vs Forward Return")
        fig_scatter_eff = px.scatter(
            obs_df, x="score", y="excess_return",
            color="cls_short",
            color_discrete_map={v: CLASS_COLORS.get(k, C["gray"]) for k, v in CLASS_SHORT.items()},
            hover_data=["ticker", "eval_date"] if "eval_date" in obs_df.columns else ["ticker"],
            opacity=0.5, title="Composite Score vs Excess Return (all observations)",
        )
        # OLS trend line
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(obs_df["score"], obs_df["excess_return"])
        x_line = np.array([obs_df["score"].min(), obs_df["score"].max()])
        fig_scatter_eff.add_trace(go.Scatter(
            x=x_line, y=intercept + slope * x_line,
            mode="lines", name=f"OLS (R²={r_val**2:.3f})",
            line=dict(color=C["yellow"], width=2, dash="dash"),
        ))
        fig_scatter_eff.update_layout(**DARK_LAYOUT, height=500,
                                       xaxis_title="Composite Score",
                                       yaxis_title="Excess Return %")
        fig_scatter_eff.add_hline(y=0, line_color=C["gray"])
        fig_scatter_eff.add_vline(x=55, line_dash="dot", line_color=C["orange"],
                                   annotation_text="Eligible=55")
        st.plotly_chart(fig_scatter_eff, use_container_width=True)

        # R² and slope summary
        r1, r2, r3 = st.columns(3)
        r1.metric("R²", f"{r_val**2:.4f}", help="결정계수: 0에 가까우면 노이즈, 1에 가까우면 완벽한 예측")
        r2.metric("Slope", f"{slope:.4f}", help="Score 1pt 증가 시 Excess Return 변화폭")
        r3.metric("p-value", f"{p_val:.4f}", help="<0.05면 통계적으로 유의미한 관계")

        # ══════════════════════════════════════════════════════════════════
        # 7. SUB-SIGNAL EFFECTIVENESS (TCS / TFS / OER)
        # ══════════════════════════════════════════════════════════════════
        if "tcs" in obs_df.columns:
            st.subheader("Sub-Signal IC (Blended & Dual-Timeframe)")
            st.caption("개별 축 시그널의 예측력 비교 — 블렌딩 점수 및 단기/장기 분해")

            sub_ics = {}
            sub_signals = ["tcs", "tfs", "oer",
                           "tcs_short", "tcs_long", "tfs_short", "tfs_long",
                           "rss_short", "rss_long"]
            for sig in sub_signals:
                if sig in obs_df.columns:
                    valid = obs_df[[sig, "excess_return"]].dropna()
                    if len(valid) >= 10:
                        ic_val, p_val = sp_stats.spearmanr(valid[sig], valid["excess_return"])
                        sub_ics[sig.upper()] = {"IC": round(ic_val, 4), "p-value": round(p_val, 4)}

            if sub_ics:
                sub_ic_df = pd.DataFrame(sub_ics).T.reset_index()
                sub_ic_df.columns = ["Signal", "IC", "p-value"]

                col_ic_tbl, col_ic_bar = st.columns(2)
                with col_ic_tbl:
                    st.dataframe(sub_ic_df, use_container_width=True, hide_index=True)
                with col_ic_bar:
                    fig_sub = go.Figure(go.Bar(
                        x=sub_ic_df["Signal"], y=sub_ic_df["IC"],
                        marker_color=[C["green"] if v > 0 else C["red"] for v in sub_ic_df["IC"]],
                        text=sub_ic_df["IC"].apply(lambda v: f"{v:.4f}"),
                        textposition="outside",
                    ))
                    fig_sub.add_hline(y=0, line_color=C["gray"])
                    fig_sub.update_layout(**DARK_LAYOUT, height=350,
                                           title="Sub-Signal IC Comparison (Blended + Short/Long)",
                                           yaxis_title="IC (Spearman)",
                                           xaxis_tickangle=-45)
                    st.plotly_chart(fig_sub, use_container_width=True)

                st.caption("TCS IC > 0 = 추세 지속 시그널 유효 | TFS IC > 0 = 형성 시그널 유효 | "
                           "OER IC < 0 = 과열 시그널 유효 (OER 높을수록 수익률 낮아야 정상) | "
                           "_S = 단기, _L = 장기 분해")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 9: SIGNAL VALIDITY
# ─────────────────────────────────────────────────────────────────────────────
with tab_validity:
    st.subheader("Signal Validity v2 (Fixed-Forward Backtest)")

    if ve_stats:
        fwd_bucket = ve_stats.get("fwd_bucket", {})
        fwd_class = ve_stats.get("fwd_class", {})
        fwd_eligible = ve_stats.get("fwd_eligible", {})
        score_weighted = ve_stats.get("score_weighted", {})
        transition_hit = ve_stats.get("transition_hit", {})

        # ── 1. Fixed-Forward Hit Rate by Score Bucket ──
        st.subheader("Fixed-Forward Hit Rate by Score Bucket")
        st.caption("동일한 forward 기간(5일/10일/21일)으로 측정하여 look-ahead bias 제거")
        bucket_order = ["0-30", "30-50", "50-70", "70-100"]

        if fwd_bucket:
            _fd_label = {5: "1W (5d)", 21: "1M (21d)", 63: "3M (63d)", 10: "10d"}
            fwd_periods = sorted(fwd_bucket.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else x)
            sel_fd = st.radio("Forward Period", fwd_periods,
                              format_func=lambda x: _fd_label.get(int(x) if str(x).isdigit() else x, f"{x}d"),
                              horizontal=True)
            fd_data = fwd_bucket.get(sel_fd, {})
            sw_data = score_weighted.get(sel_fd, {})
            el_data = fwd_eligible.get(sel_fd, {})

            bdata = []
            for b in bucket_order:
                s = fd_data.get(b, {"n": 0, "hit_rate": 0, "exc_hit": 0, "avg_ret": 0, "avg_exc": 0, "sharpe": 0, "risk_adj_hit": 0})
                sw = sw_data.get(b, {"w_hit": 0, "w_exc": 0})
                bdata.append({"Bucket": b, **s, "w_hit": sw.get("w_hit", 0), "w_exc": sw.get("w_exc", 0)})
            # Add eligible-only row
            el_all = el_data.get("eligible", {"n": 0, "hit_rate": 0, "exc_hit": 0, "avg_ret": 0, "avg_exc": 0, "sharpe": 0, "risk_adj_hit": 0})
            if el_all.get("n", 0) > 0:
                bdata.append({"Bucket": "ELIGIBLE", **el_all, "w_hit": 0, "w_exc": 0})

            bdf = pd.DataFrame(bdata)
            st.dataframe(bdf, use_container_width=True, hide_index=True)

            col_hit, col_risk = st.columns(2)
            with col_hit:
                fig_bkt = px.bar(
                    bdf, x="Bucket", y=["hit_rate", "exc_hit", "w_hit"],
                    barmode="group", title=f"{sel_fd}d Forward: Abs Hit / Excess Hit / Score-Weighted Hit",
                    color_discrete_sequence=[C["green"], C["cyan"], C["purple"]],
                    labels={"value": "%", "variable": "Metric"},
                )
                fig_bkt.update_layout(**DARK_LAYOUT, height=400)
                fig_bkt.add_hline(y=50, line_dash="dash", line_color=C["gray"])
                st.plotly_chart(fig_bkt, use_container_width=True)

            with col_risk:
                fig_risk = px.bar(
                    bdf, x="Bucket", y=["risk_adj_hit", "sharpe"],
                    barmode="group", title=f"{sel_fd}d Forward: Risk-Adjusted Hit Rate & Sharpe",
                    color_discrete_sequence=[C["yellow"], C["blue"]],
                    labels={"value": "", "variable": "Metric"},
                )
                fig_risk.update_layout(**DARK_LAYOUT, height=400)
                fig_risk.add_hline(y=50, line_dash="dash", line_color=C["gray"])
                st.plotly_chart(fig_risk, use_container_width=True)

            # ── Multi-period comparison ──
            st.subheader("Multi-Period Hit Rate Comparison")
            mp_rows = []
            for fd_p in fwd_periods:
                for b in bucket_order:
                    s = fwd_bucket.get(fd_p, {}).get(b, {})
                    if s.get("n", 0) > 0:
                        mp_rows.append({"Period": f"{fd_p}d", "Bucket": b,
                                        "Excess Hit %": s.get("exc_hit", 0),
                                        "Avg Excess %": s.get("avg_exc", 0)})
            if mp_rows:
                mp_df = pd.DataFrame(mp_rows)
                fig_mp = px.bar(mp_df, x="Bucket", y="Excess Hit %", color="Period",
                                barmode="group", title="Excess Hit Rate: 1W vs 1M vs 3M",
                                color_discrete_sequence=[C["cyan"], C["blue"], C["purple"]])
                fig_mp.update_layout(**DARK_LAYOUT, height=400)
                fig_mp.add_hline(y=50, line_dash="dash", line_color=C["gray"])
                st.plotly_chart(fig_mp, use_container_width=True)
        else:
            # Fallback to legacy stats
            bdata = []
            for b in bucket_order:
                s = ve_stats["bucket"].get(b, {"n": 0, "hit_rate": 0, "exc_hit": 0, "avg_ret": 0, "avg_exc": 0})
                bdata.append({"Bucket": b, **s})
            bdf = pd.DataFrame(bdata)
            st.dataframe(bdf, use_container_width=True, hide_index=True)

        st.divider()

        # ── 2. Classification Hit Rate ──
        st.subheader("Classification Hit Rate (10d Forward)")
        class_order = list(CLASS_SHORT.keys())
        fd10_class = fwd_class.get(10, fwd_class.get("10", ve_stats.get("class", {})))
        cdata = []
        for c in class_order:
            s = fd10_class.get(c, {"n": 0})
            if s.get("n", 0) > 0:
                cdata.append({"Class": CLASS_SHORT.get(c, c), "Full": c, **s})
        if cdata:
            cdf = pd.DataFrame(cdata)
            col_cls1, col_cls2 = st.columns(2)
            with col_cls1:
                st.dataframe(cdf.drop(columns=["Full"], errors="ignore"),
                             use_container_width=True, hide_index=True)
            with col_cls2:
                fig_cls = px.bar(
                    cdf, x="Class", y=["hit_rate", "exc_hit", "risk_adj_hit"],
                    barmode="group", title="10d Forward Hit Rate by Classification",
                    color_discrete_sequence=[C["green"], C["cyan"], C["yellow"]],
                )
                fig_cls.update_layout(**DARK_LAYOUT, height=400)
                fig_cls.add_hline(y=50, line_dash="dash", line_color=C["gray"])
                st.plotly_chart(fig_cls, use_container_width=True)

        st.divider()

        # ── 3. Transition Hit Rate ──
        st.subheader("Transition Hit Rate (from→to classification change)")
        st.caption("분류 전환 시 10일 forward excess hit rate — 어떤 전환이 가장 수익성 높은지")
        if transition_hit:
            tr_rows = []
            for key, s in transition_hit.items():
                if isinstance(key, str) and "|||" in key:
                    fc, tc = key.split("|||", 1)
                else:
                    fc, tc = key
                fc_s = CLASS_SHORT.get(fc, str(fc)[:6])
                tc_s = CLASS_SHORT.get(tc, str(tc)[:6])
                tr_rows.append({
                    "From": fc_s, "To": tc_s, "Transition": f"{fc_s}→{tc_s}",
                    **s
                })
            if tr_rows:
                tr_df = pd.DataFrame(tr_rows).sort_values("exc_hit", ascending=False)
                col_tr1, col_tr2 = st.columns(2)
                with col_tr1:
                    st.dataframe(tr_df[["Transition", "n", "hit_rate", "exc_hit", "avg_ret", "avg_exc"]].head(20),
                                 use_container_width=True, hide_index=True)
                with col_tr2:
                    top_tr = tr_df.head(15)
                    fig_tr = px.bar(top_tr, x="Transition", y="exc_hit",
                                   color="avg_exc",
                                   color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
                                   title="Top 15 Transitions: Excess Hit Rate",
                                   text="n")
                    fig_tr.update_layout(**DARK_LAYOUT, height=400, xaxis_tickangle=-45)
                    fig_tr.add_hline(y=50, line_dash="dash", line_color=C["gray"])
                    st.plotly_chart(fig_tr, use_container_width=True)

        st.divider()

        # ── 4. Transition Matrix ──
        st.subheader("Class Transition Matrix")
        all_classes_full = list(CLASS_SHORT.keys())
        classes = [c for c in all_classes_full
                   if ve_stats["transition_totals"].get(c, 0) > 0]
        if not classes:
            classes = all_classes_full
        shorts = [CLASS_SHORT.get(c, c[:6]) for c in classes]
        trans_matrix = []
        for cf in classes:
            tot = ve_stats["transition_totals"].get(cf, 0)
            row = {}
            for ct in classes:
                cnt = ve_stats["transitions"].get((cf, ct), 0)
                row[CLASS_SHORT.get(ct, ct[:6])] = round(cnt / tot * 100, 1) if tot > 0 else 0.0
            trans_matrix.append(row)

        tdf = pd.DataFrame(trans_matrix, index=shorts)
        fig_heat = px.imshow(
            tdf.values, x=shorts, y=shorts,
            color_continuous_scale="Blues", text_auto=".1f",
            title="Transition Probability (%, row -> col)",
        )
        fig_heat.update_layout(**DARK_LAYOUT, xaxis_title="To", yaxis_title="From",
                               height=max(400, len(classes) * 35))
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

    # CONTINUATION & RECOVERY class trend
    cont_tickers = fdf[fdf["classification"].str.contains("CONTINUATION|RECOVERY")]["ticker"].tolist()
    if cont_tickers and history:
        st.subheader("CONTINUATION / RECOVERY Class — 1-Week Trend Tracking")
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


# ─────────────────────────────────────────────────────────────────────────────
# TAB 12: REPORT (NEW)
# ─────────────────────────────────────────────────────────────────────────────
def _build_report(fdf, df, history, ve_stats):
    """Generate a full analytical report (~A4 20 pages) from scan data."""
    scan_date = df["data_as_of"].iloc[0] if not df.empty else "N/A"
    total = len(df)
    filtered = len(fdf)
    n_eligible = int(fdf["eligible"].sum())
    avg_comp = fdf["composite"].mean() if len(fdf) else 0
    avg_rsi = fdf["rsi"].mean() if len(fdf) else 0
    n_above_sma = int((fdf["sma50_dist"] > 0).sum())
    pct_above = n_above_sma / max(filtered, 1) * 100

    is_stk = fdf["category"].str.startswith("STK_")
    etf_df = fdf[~is_stk]
    stk_df = fdf[is_stk]

    cls_dist = fdf["classification"].value_counts()
    cats = sorted(fdf["category"].unique())

    lines = []
    L = lines.append

    # ══════════════════════════════════════════════════════════════════
    # 1. COVER & EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════
    L("# Global Price Discovery Scanner v5.0 — Analytical Report")
    L(f"**Report Date:** {scan_date}  ")
    L(f"**Universe:** {total} tickers ({len(etf_df)} ETFs + {len(stk_df)} Stocks)  ")
    L(f"**Eligible Candidates:** {n_eligible}  ")
    L("")
    L("---")
    L("## 1. Executive Summary")
    L("")
    # Key findings
    top3_cls = cls_dist.head(3)
    top_eligible = fdf[fdf["eligible"]].head(5)
    L(f"전체 유니버스 {total}개 종목 중 **{n_eligible}개({n_eligible/max(total,1)*100:.1f}%)**가 포트폴리오 편입 기준을 충족합니다. "
      f"유니버스 평균 Composite Score는 **{avg_comp:.1f}**, 평균 RSI는 **{avg_rsi:.1f}**이며, "
      f"SMA50 상회 비율은 **{pct_above:.1f}%**입니다.")
    L("")
    L("**주요 발견:**")
    L("")
    # Classification summary
    for cls_name, cnt in top3_cls.items():
        pct = cnt / filtered * 100
        L(f"- **{cls_name}**: {cnt}개 ({pct:.1f}%) — ", )
    L("")
    if not top_eligible.empty:
        tickers_str = ", ".join(f"**{r['ticker']}**({r['composite']:.1f})" for _, r in top_eligible.iterrows())
        L(f"- Top Eligible: {tickers_str}")
    L("")

    # Market regime
    if pct_above >= 70:
        regime = "강세 (Bullish)"
        regime_desc = "대부분의 종목이 SMA50 위에 위치하며 광범위한 상승 추세가 확인됩니다."
    elif pct_above >= 50:
        regime = "중립-강세 (Neutral-Bullish)"
        regime_desc = "과반의 종목이 상승 추세이나 일부 섹터에서 약세 신호가 관찰됩니다."
    elif pct_above >= 30:
        regime = "중립-약세 (Neutral-Bearish)"
        regime_desc = "상승 추세 종목이 소수이며 방어적 포지셔닝이 필요한 구간입니다."
    else:
        regime = "약세 (Bearish)"
        regime_desc = "대부분의 종목이 SMA50 하회 — 현금 비중 확대 또는 숏 포지션 고려 구간입니다."
    L(f"**Market Regime: {regime}** — {regime_desc}")
    L("")

    # ══════════════════════════════════════════════════════════════════
    # 2. MARKET BREADTH & HEALTH
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 2. Market Breadth & Health Analysis")
    L("")
    L("### 2.1 Classification Distribution")
    L("")
    L("| Classification | Count | % | Interpretation |")
    L("|---|---|---|---|")
    interp = {
        "🟢 CONTINUATION": "단기+장기 모두 상승 — 확립된 모멘텀 지속",
        "🔵 RECOVERY": "단기 상승, 장기 횡보 — 회복 초기 단계",
        "🟣 COUNTER_RALLY": "단기 상승, 장기 하락 — 기술적 반등 (주의)",
        "🟡 CONSOLIDATION": "단기 횡보, 장기 상승 — 조정 후 재진입 구간",
        "🟠 NEUTRAL": "단기+장기 횡보 — 방향성 없음",
        "🟤 FADING": "단기 횡보, 장기 하락 — 하락 추세 이행 중",
        "🔶 PULLBACK": "단기 하락, 장기 상승 — 건전한 되돌림",
        "⚠️ WEAKENING": "단기 하락, 장기 횡보 — 약세 전환 경고",
        "⬇️ DOWNTREND": "단기+장기 모두 하락 — 진입 부적합",
        "🟡 OVEREXTENDED": "과열 구간 — 단기 조정 리스크 (Override)",
        "🔵 FORMATION": "새로운 단기 브레이크아웃 (Override)",
        "🟤 EXHAUSTING": "장기 추세 소진 — 모멘텀 약화 (Override)",
    }
    all_report_classes = [
        "🟢 CONTINUATION", "🔵 RECOVERY", "🟣 COUNTER_RALLY",
        "🟡 CONSOLIDATION", "🟠 NEUTRAL", "🟤 FADING",
        "🔶 PULLBACK", "⚠️ WEAKENING", "⬇️ DOWNTREND",
        "🟡 OVEREXTENDED", "🔵 FORMATION", "🟤 EXHAUSTING",
    ]
    for cls_name in all_report_classes:
        cnt = cls_dist.get(cls_name, 0)
        pct = cnt / max(filtered, 1) * 100
        L(f"| {cls_name} | {cnt} | {pct:.1f}% | {interp.get(cls_name, '')} |")
    L("")

    L("### 2.2 SMA50 Breadth by Category")
    L("")
    L("| Category | Total | Above SMA50 | % | Avg Composite |")
    L("|---|---|---|---|---|")
    for cat in cats:
        cdf = fdf[fdf["category"] == cat]
        n = len(cdf)
        na = int((cdf["sma50_dist"] > 0).sum())
        ac = cdf["composite"].mean()
        L(f"| {cat} | {n} | {na} | {na/max(n,1)*100:.0f}% | {ac:.1f} |")
    L("")

    L("### 2.3 RSI & Overextension")
    L("")
    n_rsi_over70 = int((fdf["rsi"] > 70).sum())
    n_rsi_under30 = int((fdf["rsi"] < 30).sum())
    n_oer_high = int((fdf["oer"] >= 60).sum())
    L(f"- RSI > 70 (과매수): **{n_rsi_over70}**개 ({n_rsi_over70/max(filtered,1)*100:.1f}%)")
    L(f"- RSI < 30 (과매도): **{n_rsi_under30}**개 ({n_rsi_under30/max(filtered,1)*100:.1f}%)")
    L(f"- OER ≥ 60 (과열): **{n_oer_high}**개 ({n_oer_high/max(filtered,1)*100:.1f}%)")
    L("")

    # ══════════════════════════════════════════════════════════════════
    # 3. SECTOR ROTATION
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 3. Sector / Category Rotation Analysis")
    L("")
    cat_stats = fdf.groupby("category").agg(
        n=("ticker", "size"), eligible=("eligible", "sum"),
        avg_comp=("composite", "mean"), avg_tcs=("tcs", "mean"),
        avg_tfs=("tfs", "mean"), avg_oer=("oer", "mean"),
        avg_rsi=("rsi", "mean"),
        r1w=("ret_1w", "mean"), r1m=("ret_1m", "mean"), r3m=("ret_3m", "mean"),
    ).round(1)
    cat_stats["eligible"] = cat_stats["eligible"].astype(int)
    cat_stats = cat_stats.sort_values("avg_comp", ascending=False)

    L("| Category | N | Elig | Comp | TCS | TFS | OER | RSI | 1W% | 1M% | 3M% |")
    L("|---|---|---|---|---|---|---|---|---|---|---|")
    for cat, r in cat_stats.iterrows():
        L(f"| {cat} | {r['n']:.0f} | {r['eligible']} | {r['avg_comp']:.1f} | "
          f"{r['avg_tcs']:.0f} | {r['avg_tfs']:.0f} | {r['avg_oer']:.0f} | {r['avg_rsi']:.0f} | "
          f"{r['r1w']:.2f} | {r['r1m']:.2f} | {r['r3m']:.2f} |")
    L("")

    # Top/bottom sectors
    best_cat = cat_stats.index[0]
    worst_cat = cat_stats.index[-1]
    L(f"**가장 강한 섹터:** {best_cat} (Avg Composite {cat_stats.loc[best_cat, 'avg_comp']:.1f}, "
      f"1M Return {cat_stats.loc[best_cat, 'r1m']:.2f}%)")
    L("")
    L(f"**가장 약한 섹터:** {worst_cat} (Avg Composite {cat_stats.loc[worst_cat, 'avg_comp']:.1f}, "
      f"1M Return {cat_stats.loc[worst_cat, 'r1m']:.2f}%)")
    L("")

    # Momentum shift
    L("### 3.1 Momentum Shift (Score Change)")
    L("")
    cat_delta = fdf.groupby("category").agg(
        d1w=("ret_1w", "mean"),
        score_now=("composite", "mean"),
    )
    # Compare avg score_1w if available
    if "score_1w" in fdf.columns:
        cat_delta["score_1w_avg"] = fdf.groupby("category")["score_1w"].mean()
        cat_delta["delta_1w"] = cat_delta["score_now"] - cat_delta["score_1w_avg"]
        improving = cat_delta[cat_delta["delta_1w"] > 2].sort_values("delta_1w", ascending=False)
        declining = cat_delta[cat_delta["delta_1w"] < -2].sort_values("delta_1w")
        if not improving.empty:
            L("**개선 중인 섹터 (1W score +2 이상):**")
            for cat, r in improving.iterrows():
                L(f"- {cat}: +{r['delta_1w']:.1f}pt")
            L("")
        if not declining.empty:
            L("**악화 중인 섹터 (1W score -2 이상 하락):**")
            for cat, r in declining.iterrows():
                L(f"- {cat}: {r['delta_1w']:.1f}pt")
            L("")

    # ══════════════════════════════════════════════════════════════════
    # 4. THEME ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 4. Theme Rotation Analysis")
    L("")
    themed = fdf[fdf["theme"] != "-"]
    if not themed.empty:
        theme_stats = themed.groupby("theme").agg(
            n=("ticker", "size"), eligible=("eligible", "sum"),
            avg_comp=("composite", "mean"),
            r1m=("ret_1m", "mean"), r3m=("ret_3m", "mean"),
        ).round(1)
        theme_stats["eligible"] = theme_stats["eligible"].astype(int)

        L("### 4.1 Top 15 Themes by Composite")
        L("")
        top_themes = theme_stats.sort_values("avg_comp", ascending=False).head(15)
        L("| Theme | N | Elig | Avg Comp | 1M% | 3M% |")
        L("|---|---|---|---|---|---|")
        for th, r in top_themes.iterrows():
            L(f"| {th} | {r['n']:.0f} | {r['eligible']} | {r['avg_comp']:.1f} | {r['r1m']:.2f} | {r['r3m']:.2f} |")
        L("")

        L("### 4.2 Bottom 15 Themes by Composite")
        L("")
        bot_themes = theme_stats.sort_values("avg_comp", ascending=True).head(15)
        L("| Theme | N | Elig | Avg Comp | 1M% | 3M% |")
        L("|---|---|---|---|---|---|")
        for th, r in bot_themes.iterrows():
            L(f"| {th} | {r['n']:.0f} | {r['eligible']} | {r['avg_comp']:.1f} | {r['r1m']:.2f} | {r['r3m']:.2f} |")
        L("")

    # ══════════════════════════════════════════════════════════════════
    # 5. TOP CONVICTION IDEAS
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 5. Top Conviction Ideas")
    L("")
    top_elig = fdf[fdf["eligible"]].head(25)
    if not top_elig.empty:
        L("### 5.1 Top 25 Eligible by Composite Score")
        L("")
        L("| Rk | Ticker | Name | Cat | Comp | TCS | TFS | OER | RSS | Class | Val% | 1M% | 3M% |")
        L("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for i, (_, r) in enumerate(top_elig.iterrows()):
            L(f"| {i+1} | {r['ticker']} | {r['name'][:16]} | {r['category'][:12]} | "
              f"{r['composite']:.1f} | {r['tcs']:.0f} | {r['tfs']:.0f} | {r['oer']:.0f} | {r['rss']:.1f} | "
              f"{CLASS_SHORT.get(r['classification'], '?')} | {r['val_prob']:.1f} | "
              f"{r['ret_1m']:.2f} | {r['ret_3m']:.2f} |")
        L("")

        # Newly eligible (eligible now, score_1w was below threshold)
        if "score_1w" in fdf.columns:
            new_elig = fdf[(fdf["eligible"]) & (fdf["score_1w"] < 55)].head(10)
            if not new_elig.empty:
                L("### 5.2 Newly Eligible (1주 내 편입 기준 충족)")
                L("")
                for _, r in new_elig.iterrows():
                    L(f"- **{r['ticker']}** ({r['name'][:16]}): Composite {r['composite']:.1f} "
                      f"(1W ago: {r['score_1w']:.1f}), {r['classification']}")
                L("")

    # By validity
    top_val = fdf[fdf["eligible"]].sort_values("val_prob", ascending=False).head(15)
    if not top_val.empty:
        L("### 5.3 Top 15 by Validity Probability")
        L("")
        L("| Ticker | Name | Comp | Val% | Persist | Class |")
        L("|---|---|---|---|---|---|")
        for _, r in top_val.iterrows():
            L(f"| {r['ticker']} | {r['name'][:16]} | {r['composite']:.1f} | "
              f"{r['val_prob']:.1f} | {r['val_persist']:.0f} | {CLASS_SHORT.get(r['classification'], '?')} |")
        L("")

    # ══════════════════════════════════════════════════════════════════
    # 6. SIGNAL EFFECTIVENESS
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 6. Signal Quality & Effectiveness")
    L("")
    obs_list = ve_stats.get("observations", [])
    if obs_list:
        obs_df = pd.DataFrame(obs_list)
        try:
            from scipy import stats as sp_stats
            ic, ic_p = sp_stats.spearmanr(obs_df["score"], obs_df["excess_return"])
            overall_hr = (obs_df["excess_return"] > 0).mean() * 100
            avg_exc = obs_df["excess_return"].mean()

            L(f"- **Information Coefficient (IC):** {ic:.4f} (p={ic_p:.4f})")
            L(f"- **Excess Hit Rate:** {overall_hr:.1f}%")
            L(f"- **Avg Excess Return:** {avg_exc:.2f}%")
            L(f"- **Observations:** {len(obs_df):,}")
            L("")

            if ic > 0.05:
                L("IC가 양수이며 통계적으로 유의미합니다. Composite Score가 높을수록 벤치마크 대비 "
                  "초과수익을 달성할 확률이 높다는 것을 의미합니다.")
            else:
                L("IC가 약하여 시그널의 예측력이 제한적입니다. 시장 국면 변화에 따른 재검토가 필요합니다.")
            L("")

            # Quintile
            obs_df["quintile"] = pd.qcut(obs_df["score"], 5,
                                          labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"],
                                          duplicates="drop")
            q_agg = obs_df.groupby("quintile", observed=True).agg(
                n=("score", "size"), avg_exc=("excess_return", "mean"),
                hr=("excess_return", lambda x: (x > 0).mean() * 100),
            ).round(2)
            L("### 6.1 Quintile Analysis")
            L("")
            L("| Quintile | N | Avg Excess% | Hit Rate% |")
            L("|---|---|---|---|")
            for q, r in q_agg.iterrows():
                L(f"| {q} | {r['n']:.0f} | {r['avg_exc']:.2f} | {r['hr']:.1f} |")
            L("")
            if len(q_agg) >= 2:
                spread = q_agg["avg_exc"].iloc[-1] - q_agg["avg_exc"].iloc[0]
                L(f"**Q5-Q1 Long-Short Spread:** {spread:.2f}%")
                L("")
        except Exception:
            L("Signal effectiveness computation not available.")
            L("")
    else:
        L("Observations 데이터가 캐시에 없습니다.")
        L("")

    # ══════════════════════════════════════════════════════════════════
    # 7. RISK ALERTS
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 7. Risk Alerts & Watchlist")
    L("")

    L("### 7.1 OVEREXTENDED — 과열 종목 (단기 조정 주의)")
    L("")
    overext = fdf[fdf["classification"] == "🟡 OVEREXTENDED"].sort_values("oer", ascending=False).head(15)
    if not overext.empty:
        L("| Ticker | Name | Comp | OER | RSI | SMA50 Dist% | 1M% |")
        L("|---|---|---|---|---|---|---|")
        for _, r in overext.iterrows():
            L(f"| {r['ticker']} | {r['name'][:16]} | {r['composite']:.1f} | "
              f"{r['oer']:.0f} | {r['rsi']:.1f} | {r['sma50_dist']:.1f}% | {r['ret_1m']:.2f} |")
        L("")
    else:
        L("해당 종목 없음.")
        L("")

    L("### 7.2 EXHAUSTING / FADING — 추세 소진 & 하락 이행 종목")
    L("")
    exhaust_fade = fdf[fdf["classification"].isin(
        ["🟤 EXHAUSTING", "🟤 FADING"])].sort_values("composite", ascending=False).head(15)
    if not exhaust_fade.empty:
        L("| Ticker | Name | Class | Comp | Trend Age | 1M% | 3M% |")
        L("|---|---|---|---|---|---|---|")
        for _, r in exhaust_fade.iterrows():
            L(f"| {r['ticker']} | {r['name'][:16]} | {CLASS_SHORT.get(r['classification'], '?')} | "
              f"{r['composite']:.1f} | {r['trend_age']:.0f}d | {r['ret_1m']:.2f} | {r['ret_3m']:.2f} |")
        L("")
    else:
        L("해당 종목 없음.")
        L("")

    L("### 7.3 DOWNTREND / WEAKENING — 하락 추세 & 약세 전환 주요 종목")
    L("")
    down_weak = fdf[fdf["classification"].isin(
        ["⬇️ DOWNTREND", "⚠️ WEAKENING"])].sort_values("composite", ascending=False).head(15)
    if not down_weak.empty:
        L("| Ticker | Name | Cat | Class | Comp | RSI | 1M% | 3M% |")
        L("|---|---|---|---|---|---|---|---|")
        for _, r in down_weak.iterrows():
            L(f"| {r['ticker']} | {r['name'][:16]} | {r['category'][:12]} | "
              f"{CLASS_SHORT.get(r['classification'], '?')} | "
              f"{r['composite']:.1f} | {r['rsi']:.1f} | {r['ret_1m']:.2f} | {r['ret_3m']:.2f} |")
        L("")
    else:
        L("해당 종목 없음.")
        L("")

    L("### 7.4 COUNTER_RALLY — 기술적 반등 (위험 신호)")
    L("")
    cntr = fdf[fdf["classification"] == "🟣 COUNTER_RALLY"].sort_values("composite", ascending=False).head(10)
    if not cntr.empty:
        L("| Ticker | Name | Cat | Comp | RSI | 1M% | 3M% |")
        L("|---|---|---|---|---|---|---|")
        for _, r in cntr.iterrows():
            L(f"| {r['ticker']} | {r['name'][:16]} | {r['category'][:12]} | "
              f"{r['composite']:.1f} | {r['rsi']:.1f} | {r['ret_1m']:.2f} | {r['ret_3m']:.2f} |")
        L("")
    else:
        L("해당 종목 없음.")
        L("")

    # ══════════════════════════════════════════════════════════════════
    # 8. ETF vs STOCK COMPARISON
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 8. ETF vs Individual Stock Comparison")
    L("")
    if len(etf_df) and len(stk_df):
        L("| Metric | ETF | Stock |")
        L("|---|---|---|")
        L(f"| Count | {len(etf_df)} | {len(stk_df)} |")
        L(f"| Eligible | {int(etf_df['eligible'].sum())} | {int(stk_df['eligible'].sum())} |")
        L(f"| Avg Composite | {etf_df['composite'].mean():.1f} | {stk_df['composite'].mean():.1f} |")
        L(f"| Avg RSI | {etf_df['rsi'].mean():.1f} | {stk_df['rsi'].mean():.1f} |")
        L(f"| Above SMA50% | {(etf_df['sma50_dist']>0).mean()*100:.0f}% | {(stk_df['sma50_dist']>0).mean()*100:.0f}% |")
        L(f"| Avg 1M Return | {etf_df['ret_1m'].mean():.2f}% | {stk_df['ret_1m'].mean():.2f}% |")
        L(f"| Avg 3M Return | {etf_df['ret_3m'].mean():.2f}% | {stk_df['ret_3m'].mean():.2f}% |")
        L("")

    # ══════════════════════════════════════════════════════════════════
    # 9. CLASSIFICATION CHANGES
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 9. 7-Day Classification Changes")
    L("")
    if history:
        class_rank = {
            "⬇️ DOWNTREND": 0, "🟤 FADING": 0, "🟣 COUNTER_RALLY": 0, "🔴 CYCLE_PEAK": 0,
            "⚠️ WEAKENING": 1, "🟠 NEUTRAL": 1, "🟤 EXHAUSTING": 1, "🟡 OVEREXTENDED": 1,
            "🟡 CONSOLIDATION": 2, "🔶 PULLBACK": 2, "🔵 RECOVERY": 2,
            "🔵 FORMATION": 2, "🟦 LAGGING_CATCHUP": 2,
            "🟢 CONTINUATION": 3,
        }
        upgrades, downgrades = [], []
        for t in fdf["ticker"]:
            h = history.get(t, [])
            if len(h) < 2:
                continue
            fc, lc = h[0].get("class", ""), h[-1].get("class", "")
            if fc == lc:
                continue
            ro, rn = class_rank.get(fc, 1), class_rank.get(lc, 1)
            name = fdf.loc[fdf["ticker"] == t, "name"].iloc[0] if len(fdf[fdf["ticker"] == t]) else ""
            entry = f"**{t}** ({name[:14]}): {CLASS_SHORT.get(fc, '?')} → {CLASS_SHORT.get(lc, '?')}"
            if rn > ro:
                upgrades.append(entry)
            elif rn < ro:
                downgrades.append(entry)

        L(f"### Upgrades ({len(upgrades)})")
        L("")
        for e in upgrades[:20]:
            L(f"- {e}")
        L("")
        L(f"### Downgrades ({len(downgrades)})")
        L("")
        for e in downgrades[:20]:
            L(f"- {e}")
        L("")

    # ══════════════════════════════════════════════════════════════════
    # 10. APPENDIX
    # ══════════════════════════════════════════════════════════════════
    L("---")
    L("## 10. Appendix — Full Top 40 Eligible & Bottom 20")
    L("")
    L("### A. Top 40 Eligible")
    L("")
    L("| Rk | Ticker | Name | Category | Comp | TCS | TFS | OER | RSS | Class | RSI | Age | Val% |")
    L("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (_, r) in enumerate(fdf[fdf["eligible"]].head(40).iterrows()):
        L(f"| {i+1} | {r['ticker']} | {r['name'][:14]} | {r['category'][:12]} | "
          f"{r['composite']:.1f} | {r['tcs']:.0f} | {r['tfs']:.0f} | {r['oer']:.0f} | {r['rss']:.1f} | "
          f"{CLASS_SHORT.get(r['classification'], '?')} | {r['rsi']:.0f} | {r['trend_age']:.0f} | {r['val_prob']:.1f} |")
    L("")

    L("### B. Bottom 20 (Lowest Composite)")
    L("")
    L("| Rk | Ticker | Name | Category | Comp | Class | RSI | 1M% | 3M% |")
    L("|---|---|---|---|---|---|---|---|---|")
    bottom = fdf.sort_values("composite").head(20)
    for i, (_, r) in enumerate(bottom.iterrows()):
        L(f"| {i+1} | {r['ticker']} | {r['name'][:14]} | {r['category'][:12]} | "
          f"{r['composite']:.1f} | {CLASS_SHORT.get(r['classification'], '?')} | "
          f"{r['rsi']:.0f} | {r['ret_1m']:.2f} | {r['ret_3m']:.2f} |")
    L("")

    L("---")
    L("*This report was auto-generated by Price Discovery Scanner v5.0.*")

    return "\n".join(lines)


def _build_llm_prompt(report_md, fdf, ve_stats):
    """Build a prompt for LLM to generate narrative commentary."""
    scan_date = fdf["data_as_of"].iloc[0] if not fdf.empty else "N/A"
    prompt = f"""당신은 글로벌 자산운용사의 시니어 포트폴리오 매니저이자 투자 전략 리서치 헤드입니다.
아래 제공된 Price Discovery Scanner v5.0 (Dual-Timeframe Architecture) 결과 데이터를 기반으로, A4 20페이지 분량의 **전문 투자 코멘터리 리포트**를 한국어로 작성해주세요.

## 시그널 아키텍처 참고

스캐너 v5.0은 **듀얼 타임프레임** 구조를 사용합니다:
- **TCS_S / TCS_L**: 단기/장기 추세 지속 점수 → 블렌딩하여 TCS
- **TFS_S / TFS_L**: 단기/장기 추세 형성 점수 → 블렌딩하여 TFS
- **RSS_S / RSS_L**: 단기/장기 상대 강도 → 블렌딩하여 RSS
- **OER**: Overextension Risk (단일 타임프레임)
- **Classification**: 단기(short_dir)와 장기(long_dir) 방향의 3x3 매트릭스로 결정
  - 🟢 CONTINUATION (UP/UP), 🔵 RECOVERY (UP/FLAT), 🟣 COUNTER_RALLY (UP/DOWN)
  - 🟡 CONSOLIDATION (FLAT/UP), 🟠 NEUTRAL (FLAT/FLAT), 🟤 FADING (FLAT/DOWN)
  - 🔶 PULLBACK (DOWN/UP), ⚠️ WEAKENING (DOWN/FLAT), ⬇️ DOWNTREND (DOWN/DOWN)
  - Override: 🟡 OVEREXTENDED, 🔵 FORMATION, 🟤 EXHAUSTING

## 작성 지침

1. **어조**: 기관투자자 대상 리서치 리포트 수준의 전문적이고 분석적인 문체
2. **분량**: A4 20페이지 (약 10,000~12,000 단어)
3. **구조**: 아래 10개 섹션으로 구성
4. **데이터 기반**: 아래 제공된 스캐너 결과를 인용하며 구체적 수치를 포함
5. **해석 중심**: 단순 데이터 나열이 아닌, "왜 이런 결과가 나왔는가", "이것이 투자에 어떤 의미인가"에 초점
6. **듀얼 타임프레임 인사이트**: 단기/장기 방향성 불일치 시 어떤 전략이 적합한지 구체적으로 제시
7. **실행 가능한 인사이트**: 각 섹션마다 구체적인 투자 행동 제안 포함

## 리포트 구조 (각 섹션 2페이지)

### 1. Executive Summary (2p)
- 현재 시장 국면 진단 (듀얼 타임프레임 기반 모멘텀/브레드스)
- 핵심 발견 5가지
- 포트폴리오 액션 플랜 요약

### 2. Market Regime & Breadth (2p)
- SMA50 브레드스 분석: 전체/섹터별
- 3x3 Classification 매트릭스 분포가 의미하는 시장 사이클 위치
- 단기/장기 방향 불일치 비율 분석 (COUNTER_RALLY, PULLBACK, RECOVERY 등)
- RSI/OER 분포로 본 과열/과매도 진단

### 3. Sector Rotation (2p)
- 카테고리별 Composite/Return 분석
- 리더 섹터 vs 래거드 섹터 대비
- 1W/1M/3M 모멘텀 시프트 해석
- 섹터 로테이션 전략 제안

### 4. Theme Deep Dive (2p)
- 상위/하위 테마 분석
- 테마 내 종목 분산도
- AI인프라/반도체/에너지 등 핵심 테마 심층 분석
- 테마 기반 포지셔닝 제안

### 5. Top Conviction Ideas (2p)
- 상위 10개 Eligible 종목 심층 분석 (TCS_S/TCS_L 분해)
- 신규 편입 종목 분석
- Validity/Persistence 기반 확신도 평가
- 포지션 사이징 고려사항

### 6. Signal Quality Review (2p)
- IC/Quintile 분석 해석
- 현재 시그널의 신뢰도 평가
- 분류별 실효성 (CONTINUATION이 실제로 수익을 내는가? RECOVERY는?)
- 시그널 한계점 및 보완 방안

### 7. Risk Monitor (2p)
- OVEREXTENDED 종목의 조정 리스크 평가
- EXHAUSTING/FADING 종목의 추세 소진 시나리오
- COUNTER_RALLY 종목의 기술적 반등 위험 분석
- DOWNTREND/WEAKENING 종목 중 반등 후보 스크리닝
- 포트폴리오 리스크 관리 체크리스트

### 8. ETF vs Stock (1p)
- ETF와 개별주식의 모멘텀 차이
- 각 자산군의 현 시점 활용 전략

### 9. 7-Day Changes & Outlook (1p)
- 주간 분류 변동 해석
- 향후 1~2주 시장 전망

### 10. Conclusion & Action Items (2p)
- 종합 결론
- 구체적 투자 행동 체크리스트 (매수/매도/관망)
- 모니터링 포인트

## 스캐너 결과 데이터

```
Scan Date: {scan_date}
```

아래는 자동 생성된 데이터 리포트입니다. 이 데이터를 기반으로 위 구조에 따라 전문 코멘터리를 작성하세요.

---

{report_md}
"""
    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# TAB: GRAPH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_graph:
    st.subheader("GraphRAG Knowledge Graph Analysis")

    if not graph_data:
        st.info("No graph data available. Re-run `python3 price_discovery.py` to generate GraphRAG analysis.")
    else:
        # ── Summary metrics ──
        summary = graph_data.get('summary', {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes", summary.get('n_nodes', 0))
        c2.metric("Edges", summary.get('n_edges', 0))
        c3.metric("Communities", summary.get('n_communities', 0))
        c4.metric("Insights", summary.get('n_insights', 0))

        # ── Node / Edge type breakdown ──
        with st.expander("Graph Structure", expanded=False):
            nc1, nc2 = st.columns(2)
            with nc1:
                st.caption("Node Types")
                node_types = summary.get('node_types', {})
                for nt, cnt in sorted(node_types.items(), key=lambda x: -x[1]):
                    st.write(f"**{nt}**: {cnt}")
            with nc2:
                st.caption("Edge Types")
                edge_types = summary.get('edge_types', {})
                for et, cnt in sorted(edge_types.items(), key=lambda x: -x[1]):
                    st.write(f"**{et}**: {cnt}")

        st.divider()

        # ── Community Analysis ──
        st.subheader("Community Structure")
        comm_stats = graph_data.get('community_stats', {})
        if comm_stats:
            # Sort by size
            sorted_comms = sorted(comm_stats.items(), key=lambda x: -x[1]['n'])

            # Overview chart
            import plotly.express as px
            comm_df_data = []
            for cid, stats in sorted_comms:
                if stats['n'] < 3:
                    continue
                comm_df_data.append({
                    'Community': f"C{cid}",
                    'Size': stats['n'],
                    'Avg Composite': stats['avg_composite'],
                    'Eligible %': stats['eligible_pct'],
                    '1M Return': stats['avg_ret_1m'],
                    '3M Return': stats['avg_ret_3m'],
                    'Dominant': stats['dominant_class'],
                    'Top Category': stats['top_categories'][0][0] if stats['top_categories'] else 'N/A',
                })
            if comm_df_data:
                comm_df = pd.DataFrame(comm_df_data)
                fig = px.scatter(comm_df, x='Avg Composite', y='1M Return',
                                 size='Size', color='Dominant',
                                 hover_data=['Community', 'Top Category', 'Eligible %', '3M Return'],
                                 color_discrete_map={k: v for k, v in CLASS_COLORS.items()},
                                 title="Community Map: Composite vs 1M Return (size = # tickers)")
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)

            # Community details
            for cid, stats in sorted_comms:
                if stats['n'] < 3:
                    continue
                with st.expander(
                    f"Community {cid} — {stats['n']} tickers | "
                    f"Comp={stats['avg_composite']} | {stats['dominant_class']} | "
                    f"Eligible={stats['eligible_pct']}%",
                    expanded=False
                ):
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Avg Composite", stats['avg_composite'])
                    mc2.metric("1M Return", f"{stats['avg_ret_1m']:.1f}%")
                    mc3.metric("3M Return", f"{stats['avg_ret_3m']:.1f}%")

                    st.caption("Classification Distribution")
                    cls_data = stats['classification_dist']
                    if cls_data:
                        cls_df = pd.DataFrame([
                            {'Class': k, 'Count': v} for k, v in cls_data.items()
                        ]).sort_values('Count', ascending=False)
                        st.bar_chart(cls_df.set_index('Class'))

                    st.caption("Top Categories")
                    for cat, cnt in stats['top_categories']:
                        st.write(f"  {cat}: {cnt}")

                    if stats['top_themes']:
                        st.caption("Top Themes")
                        for thm, cnt in stats['top_themes']:
                            st.write(f"  {thm}: {cnt}")

                    st.caption(f"Tickers ({stats['n']})")
                    st.write(", ".join(stats['tickers']))

        st.divider()
        viz_data = graph_data.get('viz_data', {})

        # ══════════════════════════════════════════════════════════════
        # 1. CATEGORY ENTROPY — 카테고리별 의견 분산도
        # ══════════════════════════════════════════════════════════════
        st.subheader("Category Consensus vs Divergence")
        st.caption("엔트로피가 높을수록 카테고리 내 분류가 분산 (의견 분열) → 종목 선별 필요 | 낮을수록 컨센서스 강함 → 카테고리 레벨 베팅 가능")
        entropy_viz = viz_data.get('category_entropy', [])
        if entropy_viz:
            ent_df = pd.DataFrame(entropy_viz)
            fig = px.bar(ent_df, x='category', y='entropy',
                         color='label',
                         color_discrete_map={
                             'HIGH — Opinion Split': '#ef4444',
                             'MODERATE': '#f59e0b',
                             'LOW — Strong Consensus': '#22c55e',
                         },
                         hover_data=['n', 'dominant', 'dominant_pct'],
                         text='entropy',
                         title="Classification Entropy by Category (0=consensus, 1=split)")
            fig.update_layout(template="plotly_dark", height=450,
                              xaxis_tickangle=-45, showlegend=True)
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

            # Stacked classification distribution per category
            rows = []
            for e in entropy_viz:
                for cls, cnt in e['cls_dist'].items():
                    rows.append({'Category': e['category'], 'Classification': cls, 'Count': cnt})
            if rows:
                stack_df = pd.DataFrame(rows)
                fig2 = px.bar(stack_df, x='Category', y='Count', color='Classification',
                              color_discrete_map={k: v for k, v in CLASS_COLORS.items()},
                              title="Classification Distribution by Category",
                              barmode='stack')
                fig2.update_layout(template="plotly_dark", height=450, xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════
        # 2. THEME PROPAGATION — 테마 내 모멘텀 전파 맵
        # ══════════════════════════════════════════════════════════════
        st.subheader("Theme Momentum Propagation")
        st.caption("같은 테마 내 종목의 분류 분포: Leader(CONT) → Forming(FORM/RECV) → Neutral → Bearish 순서로 모멘텀 전파")
        prop_viz = viz_data.get('theme_propagation', [])
        if prop_viz:
            prop_df = pd.DataFrame(prop_viz)
            # Only show themes with 3+ members
            theme_sizes = prop_df['theme'].value_counts()
            big_themes = theme_sizes[theme_sizes >= 3].index.tolist()
            prop_df = prop_df[prop_df['theme'].isin(big_themes)]

            if not prop_df.empty:
                stage_colors = {
                    'Leader': '#22c55e', 'Forming': '#3b82f6',
                    'Neutral': '#f59e0b', 'Pullback': '#f97316', 'Bearish': '#ef4444',
                }

                # Theme propagation heatmap: theme × stage count
                stage_order = ['Leader', 'Forming', 'Neutral', 'Pullback', 'Bearish']
                pivot = prop_df.groupby(['theme', 'stage']).size().reset_index(name='count')

                # Filter to themes with propagation activity (Leader + Forming > 0)
                active_themes = prop_df[prop_df['stage'].isin(['Leader', 'Forming'])]['theme'].unique()
                if len(active_themes) > 0:
                    active_df = prop_df[prop_df['theme'].isin(active_themes)]
                    fig3 = px.strip(active_df, x='composite', y='theme', color='stage',
                                    color_discrete_map=stage_colors,
                                    hover_data=['ticker', 'classification', 'ret_1m'],
                                    title="Active Propagation: Ticker Scores by Theme & Stage",
                                    stripmode='overlay')
                    fig3.update_layout(template="plotly_dark",
                                       height=max(350, len(active_themes) * 40 + 100),
                                       yaxis={'categoryorder': 'total ascending'})
                    fig3.update_traces(marker_size=10)
                    st.plotly_chart(fig3, use_container_width=True)

                # Stage distribution bar
                stage_counts = prop_df.groupby('stage').size().reset_index(name='count')
                stage_counts['stage'] = pd.Categorical(stage_counts['stage'], categories=stage_order, ordered=True)
                stage_counts = stage_counts.sort_values('stage')
                fig4 = px.bar(stage_counts, x='stage', y='count', color='stage',
                              color_discrete_map=stage_colors,
                              title="Overall Theme Stage Distribution (all themes with 3+ members)",
                              text='count')
                fig4.update_layout(template="plotly_dark", height=350, showlegend=False)
                st.plotly_chart(fig4, use_container_width=True)

        st.divider()

        # ══════════════════════════════════════════════════════════════
        # 3. ETF-STOCK DIVERGENCE — ETF vs 구성종목 괴리
        # ══════════════════════════════════════════════════════════════
        st.subheader("ETF vs Stock Classification Divergence")
        st.caption("ETF의 분류와 해당 섹터 개별 종목들의 Bull/Bear 비율 비교 — 괴리가 크면 ETF 시그널이 소수 대형주에 의존")
        div_viz = viz_data.get('etf_stock_divergence', [])
        if div_viz:
            div_df = pd.DataFrame(div_viz)
            fig5 = px.bar(div_df, x='etf', y=['stock_bull_pct', 'stock_bear_pct'],
                          barmode='group',
                          color_discrete_sequence=['#22c55e', '#ef4444'],
                          title="Stock Bull% vs Bear% for each Sector ETF",
                          labels={'value': '%', 'variable': 'Direction'},
                          hover_data=['etf_class', 'etf_composite', 'stock_avg_composite', 'stock_n'])
            fig5.update_layout(template="plotly_dark", height=400)
            # Add ETF classification as annotation
            for i, row in div_df.iterrows():
                marker = "⚠️" if row['divergent'] else ""
                fig5.add_annotation(x=row['etf'], y=105,
                                    text=f"{marker}{row['etf_class'][:6]}",
                                    showarrow=False, font=dict(size=10))
            st.plotly_chart(fig5, use_container_width=True)

            # Divergence table
            divergent = div_df[div_df['divergent']]
            if not divergent.empty:
                st.warning(f"⚠️ {len(divergent)} divergent ETF-Stock pair(s) detected:")
                for _, row in divergent.iterrows():
                    st.write(f"  **{row['etf']}** ({row['etf_class']}) vs "
                             f"**{row['stock_category']}** "
                             f"(Bull={row['stock_bull_pct']}% / Bear={row['stock_bear_pct']}%)")
            else:
                st.success("No significant ETF-Stock divergence detected.")

        st.divider()

        # ══════════════════════════════════════════════════════════════
        # 4. FORMATION PIPELINE — 브레이크아웃 파이프라인
        # ══════════════════════════════════════════════════════════════
        st.subheader("Formation Pipeline (Themes with Active Breakouts)")
        st.caption("테마별 모멘텀 breadth: Forming+Continuing 비율이 높을수록 테마 전체가 강세")
        pipeline = graph_data.get('formation_pipeline', {})
        if pipeline:
            pipe_data = []
            for theme, data in pipeline.items():
                pipe_data.append({
                    'Theme': theme,
                    'Breadth': data['momentum_breadth'],
                    'Forming': len(data['forming']),
                    'Continuing': len(data['continuing']),
                    'Neutral Queue': len(data['neutral_queue']),
                    'Pullback': len(data['pullback_candidates']),
                    'Total': data['total'],
                })
            pipe_df = pd.DataFrame(pipe_data).sort_values('Breadth', ascending=True)

            fig6 = px.bar(pipe_df.tail(20), x='Breadth', y='Theme',
                          orientation='h',
                          color='Breadth',
                          color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
                          hover_data=['Forming', 'Continuing', 'Neutral Queue', 'Total'],
                          title="Top 20 Formation Pipeline: Momentum Breadth by Theme")
            fig6.update_layout(template="plotly_dark",
                               height=max(400, len(pipe_df.tail(20)) * 30 + 100))
            st.plotly_chart(fig6, use_container_width=True)

            # Expandable details for top themes
            for theme, data in list(pipeline.items())[:10]:
                breadth = data['momentum_breadth']
                icon = "🟢" if breadth > 60 else "🔵" if breadth > 30 else "🟠"
                with st.expander(
                    f"{icon} {theme} — Breadth={breadth}% | "
                    f"Forming={len(data['forming'])} | Continuing={len(data['continuing'])}",
                    expanded=False
                ):
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.caption("Forming (breakout)")
                        st.write(", ".join(data['forming']) if data['forming'] else "—")
                        st.caption("Continuing (leaders)")
                        st.write(", ".join(data['continuing']) if data['continuing'] else "—")
                    with pc2:
                        st.caption("Neutral Queue (potential)")
                        st.write(", ".join(data['neutral_queue'][:10]) if data['neutral_queue'] else "—")
                        st.caption("Pullback (dip candidates)")
                        st.write(", ".join(data['pullback_candidates']) if data['pullback_candidates'] else "—")
        else:
            st.write("No active formation pipeline.")

        st.divider()

        # ══════════════════════════════════════════════════════════════
        # 5. LEADER-LAGGER + RAW INSIGHTS
        # ══════════════════════════════════════════════════════════════
        st.subheader("Other Insights")
        insights = graph_data.get('insights', [])
        other_insights = [i for i in insights if i['type'] in ('leader_lagger', 'cross_category_flow')]
        if other_insights:
            for ins in other_insights:
                icon = {'leader_lagger': '🏃', 'cross_category_flow': '🌊'}.get(ins['type'], '💡')
                st.info(f"{icon} **[{ins['type'].upper()}]** {ins['detail']}")
        else:
            st.write("No additional insights.")

        st.divider()

        # ── LLM Export ──
        st.subheader("LLM Export (Copy for AI Analysis)")
        llm_text = graph_data.get('llm_export', '')
        if llm_text:
            st.code(llm_text, language="text")
        else:
            st.write("No LLM export available.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB: FACTOR EFFICACY — Reverse Factor Model (5 Methodologies)
# ─────────────────────────────────────────────────────────────────────────────
with tab_factor:
    st.subheader("Factor Efficacy — Reverse Factor Model")
    st.caption("가격 데이터 → 팩터 우월성 역산: 5가지 방법론 통합 분석")

    if not factor_efficacy or factor_efficacy.get("error"):
        st.warning("Factor Efficacy 데이터가 없습니다. `python3 price_discovery.py`를 다시 실행하세요.")
    else:
        fe = factor_efficacy  # shorthand
        unified = fe.get("unified", {})
        panel_info = fe.get("panel_size", {})

        # ══════════════════════════════════════════════════════════════════
        # 0. HEADER KPIs
        # ══════════════════════════════════════════════════════════════════
        rc = fe.get("regime_conditional", {})
        pca_data = fe.get("pca", {})
        top3 = unified.get("top3_factors", [])
        top3_grp = unified.get("top3_groups", [])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Current Regime", rc.get("current_regime", "?"))
        k2.metric("Effective Dimensions (PCA)", pca_data.get("effective_dimensionality", "?"))
        k3.metric("Eval Points", panel_info.get("n_eval_points", "?"))
        k4.metric("Avg Universe", f'{panel_info.get("avg_tickers", 0):.0f}')

        if top3:
            st.info(f"**Top 3 Factors**: {', '.join(top3)}  |  **Top 3 Groups**: {', '.join(top3_grp)}")

        # Sub-tabs for each methodology
        (ft_unified, ft_fm, ft_ic, ft_ls, ft_pca, ft_regime) = st.tabs([
            "Unified Ranking", "1. Fama-MacBeth", "2. IC Analysis",
            "3. Long-Short", "4. PCA Model", "5. Regime Analysis",
        ])

        # ══════════════════════════════════════════════════════════════════
        # SUB-TAB: UNIFIED RANKING
        # ══════════════════════════════════════════════════════════════════
        with ft_unified:
            st.markdown("##### 5-Method Composite Factor Ranking")
            st.caption("FM 25% + IC 30% + LS 25% + PCA 10% + Regime 10%")

            ranking = unified.get("unified_ranking", [])
            if ranking:
                udf = pd.DataFrame(ranking)
                display_cols = ["rank", "factor", "group", "unified_score",
                                "fm_tstat", "ic_ir", "ls_sharpe", "pca_bonus", "regime_bonus"]
                display_cols = [c for c in display_cols if c in udf.columns]
                st.dataframe(
                    udf[display_cols].style.background_gradient(
                        subset=["unified_score"], cmap="YlOrRd"
                    ).format({
                        "unified_score": "{:.1f}", "fm_tstat": "{:.2f}",
                        "ic_ir": "{:.3f}", "ls_sharpe": "{:.3f}",
                        "pca_bonus": "{:.1f}", "regime_bonus": "{:.1f}",
                    }),
                    use_container_width=True, hide_index=True, height=600,
                )

                # Horizontal bar chart
                top_n = udf.head(15)
                fig_rank = go.Figure()
                fig_rank.add_trace(go.Bar(
                    y=top_n["factor"][::-1], x=top_n["unified_score"][::-1],
                    orientation="h",
                    marker_color=[C["cyan"] if s >= 50 else C["blue"] if s >= 30
                                  else C["gray"] for s in top_n["unified_score"][::-1]],
                    text=[f'{s:.1f}' for s in top_n["unified_score"][::-1]],
                    textposition="outside",
                ))
                fig_rank.update_layout(
                    **DARK_LAYOUT, height=500,
                    title="Unified Factor Ranking (Top 15)",
                    xaxis_title="Composite Score", yaxis_title="",
                    margin=dict(l=150),
                )
                st.plotly_chart(fig_rank, use_container_width=True)

            # Group ranking
            grp_rank = unified.get("group_ranking", [])
            if grp_rank:
                st.markdown("##### Factor Group Ranking")
                gdf = pd.DataFrame(grp_rank)
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.dataframe(gdf, use_container_width=True, hide_index=True)
                with col_g2:
                    fig_grp = go.Figure()
                    fig_grp.add_trace(go.Bar(
                        x=gdf["group"], y=gdf["avg_score"],
                        marker_color=C["cyan"], name="Avg Score",
                    ))
                    fig_grp.add_trace(go.Bar(
                        x=gdf["group"], y=gdf["max_score"],
                        marker_color=C["blue"], name="Max Score", opacity=0.5,
                    ))
                    fig_grp.update_layout(
                        **DARK_LAYOUT, barmode="overlay", height=350,
                        title="Factor Group Scores",
                        xaxis_tickangle=-45,
                    )
                    st.plotly_chart(fig_grp, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════
        # SUB-TAB 1: FAMA-MACBETH
        # ══════════════════════════════════════════════════════════════════
        with ft_fm:
            fm = fe.get("fama_macbeth", {})
            st.markdown("##### Fama-MacBeth Cross-Sectional Regression")
            st.caption("매 시점 횡단면 회귀: R_i = α + Σ λ_k · F_k,i → 팩터 프리미엄 λ 추정")

            fp = fm.get("factor_premiums", [])
            if fp:
                fpdf = pd.DataFrame(fp)

                # KPIs
                sig_factors = [f for f in fp if f["significance"] in ("**", "***")]
                k1, k2, k3 = st.columns(3)
                k1.metric("Significant Factors (p<0.05)", len(sig_factors))
                k2.metric("Total Factors Analyzed", len(fp))
                k3.metric("Evaluation Periods", fm.get("n_periods", "?"))

                # Table
                display_cols = ["factor", "group", "mean_premium", "t_stat",
                                "significance", "pct_positive", "n_periods"]
                display_cols = [c for c in display_cols if c in fpdf.columns]
                st.dataframe(
                    fpdf[display_cols].style.format({
                        "mean_premium": "{:.4f}", "t_stat": "{:.3f}",
                        "pct_positive": "{:.1f}%",
                    }),
                    use_container_width=True, hide_index=True, height=500,
                )

                # T-stat chart
                top20 = fpdf.head(20)
                fig_t = go.Figure()
                fig_t.add_trace(go.Bar(
                    x=top20["factor"], y=top20["t_stat"],
                    marker_color=[C["green"] if t > 1.96 else C["red"] if t < -1.96
                                  else C["gray"] for t in top20["t_stat"]],
                ))
                fig_t.add_hline(y=1.96, line_dash="dash", line_color=C["yellow"],
                                annotation_text="t=1.96 (95%)")
                fig_t.add_hline(y=-1.96, line_dash="dash", line_color=C["yellow"])
                fig_t.add_hline(y=0, line_color=C["gray"])
                fig_t.update_layout(
                    **DARK_LAYOUT, height=400,
                    title="Factor Premium t-Statistics (|t|>1.96 = significant)",
                    xaxis_tickangle=-45, yaxis_title="t-stat",
                )
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.warning("Fama-MacBeth 결과가 없습니다.")

        # ══════════════════════════════════════════════════════════════════
        # SUB-TAB 2: IC ANALYSIS
        # ══════════════════════════════════════════════════════════════════
        with ft_ic:
            ic = fe.get("ic_analysis", {})
            st.markdown("##### Information Coefficient (IC) Analysis")
            st.caption("IC = Spearman(팩터값, 미래수익률) — IC_IR ≥ 0.5이면 유의미한 예측력")

            fi = ic.get("factor_ic", [])
            if fi:
                icdf = pd.DataFrame(fi)

                # KPIs
                strong = [f for f in fi if f["quality"] == "STRONG"]
                moderate = [f for f in fi if f["quality"] == "MODERATE"]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("STRONG Factors", len(strong))
                k2.metric("MODERATE Factors", len(moderate))
                k3.metric("Avg IC (all)", f'{icdf["ic_mean"].mean():.4f}')
                k4.metric("Avg IC IR (all)", f'{icdf["ic_ir"].mean():.3f}')

                # Table
                display_cols = ["factor", "group", "ic_mean", "ic_std", "ic_ir",
                                "ic_hit_rate", "quality", "direction"]
                display_cols = [c for c in display_cols if c in icdf.columns]
                st.dataframe(
                    icdf[display_cols].style.format({
                        "ic_mean": "{:.4f}", "ic_std": "{:.4f}", "ic_ir": "{:.3f}",
                        "ic_hit_rate": "{:.1f}%",
                    }),
                    use_container_width=True, hide_index=True, height=500,
                )

                col_ic1, col_ic2 = st.columns(2)

                with col_ic1:
                    # IC IR bar chart
                    top15 = icdf.head(15)
                    fig_ir = go.Figure()
                    fig_ir.add_trace(go.Bar(
                        x=top15["factor"], y=top15["ic_ir"].abs(),
                        marker_color=[C["green"] if q == "STRONG" else C["blue"]
                                      if q == "MODERATE" else C["yellow"]
                                      if q == "WEAK" else C["gray"]
                                      for q in top15["quality"]],
                    ))
                    fig_ir.add_hline(y=0.5, line_dash="dash", line_color=C["yellow"],
                                     annotation_text="IR=0.5 threshold")
                    fig_ir.update_layout(
                        **DARK_LAYOUT, height=400,
                        title="IC Information Ratio (|IC_IR|)",
                        xaxis_tickangle=-45, yaxis_title="|IC IR|",
                    )
                    st.plotly_chart(fig_ir, use_container_width=True)

                with col_ic2:
                    # IC mean scatter
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=icdf["ic_mean"], y=icdf["ic_ir"],
                        mode="markers+text",
                        text=icdf["factor"],
                        textposition="top center",
                        textfont=dict(size=9),
                        marker=dict(
                            size=10,
                            color=[{"STRONG": C["green"], "MODERATE": C["blue"],
                                    "WEAK": C["yellow"], "NOISE": C["gray"]}.get(q, C["gray"])
                                   for q in icdf["quality"]],
                        ),
                    ))
                    fig_scatter.add_hline(y=0.5, line_dash="dash", line_color=C["yellow"])
                    fig_scatter.add_hline(y=-0.5, line_dash="dash", line_color=C["yellow"])
                    fig_scatter.add_vline(x=0.05, line_dash="dash", line_color=C["cyan"])
                    fig_scatter.add_vline(x=-0.05, line_dash="dash", line_color=C["cyan"])
                    fig_scatter.update_layout(
                        **DARK_LAYOUT, height=400,
                        title="IC Mean vs IC IR (Quality Map)",
                        xaxis_title="IC Mean", yaxis_title="IC IR",
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                # IC time series for top factor
                st.markdown("##### IC Time Series — Top Factor")
                if fi[0].get("ic_timeseries"):
                    top_factor = fi[0]
                    ts = top_factor["ic_timeseries"]
                    fig_ts = go.Figure()
                    fig_ts.add_trace(go.Bar(
                        x=list(range(1, len(ts)+1)), y=ts,
                        marker_color=[C["green"] if v > 0 else C["red"] for v in ts],
                        name=top_factor["factor"],
                    ))
                    fig_ts.add_hline(y=0, line_color=C["gray"])
                    avg_ic = sum(ts) / len(ts) if ts else 0
                    fig_ts.add_hline(y=avg_ic, line_dash="dash", line_color=C["cyan"],
                                     annotation_text=f"Avg={avg_ic:.3f}")
                    fig_ts.update_layout(
                        **DARK_LAYOUT, height=300,
                        title=f'IC Time Series: {top_factor["factor"]} (IR={top_factor["ic_ir"]:.3f})',
                        xaxis_title="Evaluation Period", yaxis_title="IC",
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)

                # Multi-horizon comparison
                mh = fe.get("multi_horizon_ic", {})
                if mh:
                    st.markdown("##### Multi-Horizon IC Comparison")
                    mh_rows = []
                    for period, data in mh.items():
                        for item in data.get("top5", []):
                            mh_rows.append({"Horizon": period, **item})
                    if mh_rows:
                        mhdf = pd.DataFrame(mh_rows)
                        fig_mh = px.bar(
                            mhdf, x="factor", y="ic_ir", color="Horizon",
                            barmode="group", text="ic_ir",
                            color_discrete_map={"1W": C["yellow"], "1M": C["cyan"], "3M": C["blue"]},
                        )
                        fig_mh.update_layout(
                            **DARK_LAYOUT, height=400,
                            title="Top Factors by Horizon (IC IR)",
                            xaxis_tickangle=-45,
                        )
                        fig_mh.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        st.plotly_chart(fig_mh, use_container_width=True)
            else:
                st.warning("IC Analysis 결과가 없습니다.")

        # ══════════════════════════════════════════════════════════════════
        # SUB-TAB 3: LONG-SHORT PORTFOLIO
        # ══════════════════════════════════════════════════════════════════
        with ft_ls:
            ls = fe.get("long_short", {})
            st.markdown("##### Long-Short Factor Portfolio")
            st.caption("팩터값 Top/Bottom 분위 포트폴리오 수익률 비교 → Sharpe Ratio로 팩터 우열 판단")

            fr = ls.get("factor_returns", [])
            if fr:
                lsdf = pd.DataFrame(fr)

                # KPIs
                profitable = [f for f in fr if f["ann_return"] > 0]
                k1, k2, k3 = st.columns(3)
                k1.metric("Profitable Factors", f'{len(profitable)}/{len(fr)}')
                best = fr[0] if fr else {}
                k2.metric("Best Factor", f'{best.get("factor", "?")} (SR={best.get("ann_sharpe", 0):.2f})')
                k3.metric("Best Ann. Return", f'{best.get("ann_return", 0):.1f}%')

                # Table
                display_cols = ["factor", "group", "ann_return", "ann_sharpe",
                                "monotonicity", "mono_direction", "win_rate"]
                display_cols = [c for c in display_cols if c in lsdf.columns]
                st.dataframe(
                    lsdf[display_cols].style.format({
                        "ann_return": "{:.2f}%", "ann_sharpe": "{:.3f}",
                        "monotonicity": "{:.2f}", "win_rate": "{:.1f}%",
                    }),
                    use_container_width=True, hide_index=True, height=500,
                )

                col_ls1, col_ls2 = st.columns(2)

                with col_ls1:
                    # Sharpe ratio chart
                    top15 = lsdf.head(15)
                    fig_sr = go.Figure()
                    fig_sr.add_trace(go.Bar(
                        x=top15["factor"], y=top15["ann_sharpe"],
                        marker_color=[C["green"] if s > 0 else C["red"] for s in top15["ann_sharpe"]],
                    ))
                    fig_sr.add_hline(y=0.5, line_dash="dash", line_color=C["yellow"],
                                     annotation_text="SR=0.5")
                    fig_sr.add_hline(y=0, line_color=C["gray"])
                    fig_sr.update_layout(
                        **DARK_LAYOUT, height=400,
                        title="Ann. Sharpe Ratio (Long-Short)",
                        xaxis_tickangle=-45, yaxis_title="Sharpe Ratio",
                    )
                    st.plotly_chart(fig_sr, use_container_width=True)

                with col_ls2:
                    # Monotonicity vs Sharpe scatter
                    fig_ms = go.Figure()
                    fig_ms.add_trace(go.Scatter(
                        x=lsdf["monotonicity"], y=lsdf["ann_sharpe"],
                        mode="markers+text",
                        text=lsdf["factor"],
                        textposition="top center",
                        textfont=dict(size=9),
                        marker=dict(size=10, color=C["cyan"]),
                    ))
                    fig_ms.add_hline(y=0.5, line_dash="dash", line_color=C["yellow"])
                    fig_ms.add_vline(x=0.75, line_dash="dash", line_color=C["yellow"])
                    fig_ms.update_layout(
                        **DARK_LAYOUT, height=400,
                        title="Monotonicity vs Sharpe (↗ = ideal factor)",
                        xaxis_title="Monotonicity (1.0=perfect)", yaxis_title="Ann. Sharpe",
                    )
                    st.plotly_chart(fig_ms, use_container_width=True)

                # Quintile return detail for selected factor
                st.markdown("##### Quintile Return Profile")
                sel_factor = st.selectbox("Select factor", [f["factor"] for f in fr],
                                          key="ls_factor_sel")
                sel_data = next((f for f in fr if f["factor"] == sel_factor), None)
                if sel_data and sel_data.get("quintile_returns"):
                    qr = sel_data["quintile_returns"]
                    q_labels = ["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"]
                    fig_qr = go.Figure()
                    fig_qr.add_trace(go.Bar(
                        x=q_labels[:len(qr)], y=qr,
                        marker_color=[C["red"], C["orange"], C["yellow"],
                                      C["blue"], C["green"]][:len(qr)],
                        text=[f'{v:.2f}%' for v in qr],
                        textposition="outside",
                    ))
                    fig_qr.add_hline(y=0, line_color=C["gray"])
                    fig_qr.update_layout(
                        **DARK_LAYOUT, height=350,
                        title=f'{sel_factor}: Quintile Returns (Mono={sel_data["monotonicity"]:.2f}, '
                              f'Direction={sel_data["mono_direction"]})',
                        yaxis_title="Avg Return %",
                    )
                    st.plotly_chart(fig_qr, use_container_width=True)
            else:
                st.warning("Long-Short 결과가 없습니다.")

        # ══════════════════════════════════════════════════════════════════
        # SUB-TAB 4: PCA MODEL
        # ══════════════════════════════════════════════════════════════════
        with ft_pca:
            pca = fe.get("pca", {})
            st.markdown("##### PCA Statistical Factor Model")
            st.caption("수익률 공분산행렬에서 잠재 팩터 추출 → 알려진 팩터와 상관 분석")

            comps = pca.get("components", [])
            if comps:
                # KPIs
                k1, k2, k3 = st.columns(3)
                k1.metric("Effective Dimensions", pca.get("effective_dimensionality", "?"),
                          help="Kaiser criterion: eigenvalue > 1")
                k2.metric("Top-3 Var Explained", f'{pca.get("total_var_explained_top3", 0):.1f}%')
                k3.metric("Universe Size", f'{pca.get("n_tickers", 0)} tickers × {pca.get("n_days", 0)}d')

                # Scree plot
                eigenvals = [c["eigenvalue"] for c in comps]
                var_expl = [c["var_explained"] for c in comps]
                cum_var = [c["cum_var_explained"] for c in comps]

                fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
                fig_scree.add_trace(go.Bar(
                    x=[f'PC{c["pc"]}' for c in comps],
                    y=var_expl,
                    name="Var Explained %",
                    marker_color=C["cyan"],
                ), secondary_y=False)
                fig_scree.add_trace(go.Scatter(
                    x=[f'PC{c["pc"]}' for c in comps],
                    y=cum_var,
                    name="Cumulative %",
                    mode="lines+markers",
                    line=dict(color=C["yellow"], width=2),
                ), secondary_y=True)
                fig_scree.add_hline(y=1.0, line_dash="dash", line_color=C["red"],
                                    annotation_text="Kaiser (eigenval=1)", secondary_y=False)
                fig_scree.update_layout(
                    **DARK_LAYOUT, height=400,
                    title="Scree Plot — Variance Explained by Principal Component",
                )
                fig_scree.update_yaxes(title_text="Var Explained %", secondary_y=False)
                fig_scree.update_yaxes(title_text="Cumulative %", secondary_y=True)
                st.plotly_chart(fig_scree, use_container_width=True)

                # PC → Factor mapping table
                st.markdown("##### PC → Factor Mapping")
                pc_rows = []
                for c in comps:
                    top_fcs = c.get("top_factor_correlations", [])
                    mapping_str = ", ".join(
                        f'{fc["factor"]}({fc["correlation"]:.2f})' for fc in top_fcs[:3]
                    )
                    pc_rows.append({
                        "PC": f'PC{c["pc"]}',
                        "Eigenvalue": c["eigenvalue"],
                        "Var %": c["var_explained"],
                        "Cum Var %": c["cum_var_explained"],
                        "Interpretation": c["interpretation"],
                        "Top Correlations": mapping_str,
                    })
                st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

                # PC → Factor heatmap (top 3 PCs × all factors)
                st.markdown("##### PC-Factor Correlation Heatmap")
                heat_data = []
                for c in comps[:5]:
                    for fc in c.get("top_factor_correlations", []):
                        heat_data.append({
                            "PC": f'PC{c["pc"]}',
                            "Factor": fc["factor"],
                            "Correlation": fc["correlation"],
                        })
                if heat_data:
                    hdf = pd.DataFrame(heat_data)
                    pivot = hdf.pivot_table(index="Factor", columns="PC",
                                            values="Correlation", aggfunc="first").fillna(0)
                    fig_heat = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale="RdBu_r", zmid=0,
                        text=np.round(pivot.values, 2), texttemplate="%{text}",
                    ))
                    fig_heat.update_layout(
                        **DARK_LAYOUT, height=max(300, len(pivot) * 25),
                        title="PC ↔ Factor Correlation",
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                # Top tickers per PC
                pc_tickers = pca.get("pc_top_tickers", [])
                if pc_tickers:
                    st.markdown("##### Top Loadings per PC")
                    for pct in pc_tickers[:3]:
                        with st.expander(f'PC{pct["pc"]} — Top 10 Tickers'):
                            tdf = pd.DataFrame(pct["top_tickers"])
                            st.dataframe(tdf, use_container_width=True, hide_index=True)
            else:
                st.warning("PCA 결과가 없습니다.")

        # ══════════════════════════════════════════════════════════════════
        # SUB-TAB 5: REGIME ANALYSIS
        # ══════════════════════════════════════════════════════════════════
        with ft_regime:
            rc = fe.get("regime_conditional", {})
            st.markdown("##### Regime-Conditional Factor Premium")
            st.caption("시장 레짐(BULL/BEAR/TRANSITION)별 팩터 유효성 비교")

            regime_dist = rc.get("regime_distribution", {})
            current = rc.get("current_regime", "?")

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Current Regime", current)
            k2.metric("BULL periods", regime_dist.get("BULL", 0))
            k3.metric("BEAR periods", regime_dist.get("BEAR", 0))
            k4.metric("TRANSITION periods", regime_dist.get("TRANSITION", 0))

            # Recommendations
            recs = rc.get("recommendations", [])
            if recs:
                for rec in recs:
                    st.success(rec)

            # Regime-specific factor rankings (side by side)
            regime_rankings = rc.get("regime_factor_rankings", {})
            if regime_rankings:
                st.markdown("##### Factor Rankings by Regime")
                regime_cols = st.columns(len(regime_rankings))
                for col, (regime, factors) in zip(regime_cols, regime_rankings.items()):
                    with col:
                        color = {"BULL": C["green"], "BEAR": C["red"],
                                 "TRANSITION": C["yellow"]}.get(regime, C["gray"])
                        st.markdown(f'<span style="color:{color};font-weight:bold;">'
                                    f'{regime}</span>', unsafe_allow_html=True)
                        if factors:
                            rdf = pd.DataFrame(factors[:10])
                            display_cols = ["factor", "ic_mean", "ic_ir"]
                            display_cols = [c for c in display_cols if c in rdf.columns]
                            st.dataframe(
                                rdf[display_cols].style.format({
                                    "ic_mean": "{:.4f}", "ic_ir": "{:.3f}",
                                }),
                                use_container_width=True, hide_index=True,
                            )

                # Regime comparison heatmap
                st.markdown("##### Cross-Regime Factor Heatmap")
                heat_rows = []
                for regime, factors in regime_rankings.items():
                    for f in factors[:15]:
                        heat_rows.append({
                            "Regime": regime,
                            "Factor": f["factor"],
                            "IC Mean": f["ic_mean"],
                        })
                if heat_rows:
                    hrdf = pd.DataFrame(heat_rows)
                    pivot = hrdf.pivot_table(index="Factor", columns="Regime",
                                             values="IC Mean", aggfunc="first").fillna(0)
                    fig_rh = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale="RdBu_r", zmid=0,
                        text=np.round(pivot.values, 3), texttemplate="%{text}",
                    ))
                    fig_rh.update_layout(
                        **DARK_LAYOUT, height=max(350, len(pivot) * 22),
                        title="Factor IC by Regime (blue=positive, red=negative)",
                    )
                    st.plotly_chart(fig_rh, use_container_width=True)

            # Cross-regime stability (all-weather factors)
            stability = rc.get("cross_regime_stability", [])
            if stability:
                st.markdown("##### All-Weather Factor Stability")
                st.caption("모든 레짐에서 같은 방향으로 작동하는 팩터 = 레짐 불변 (All-Weather)")
                sdf = pd.DataFrame(stability)
                display_cols = ["factor", "group", "mean_ic", "std_ic",
                                "stability", "all_weather"]
                display_cols = [c for c in display_cols if c in sdf.columns]
                st.dataframe(
                    sdf[display_cols].style.format({
                        "mean_ic": "{:.4f}", "std_ic": "{:.4f}", "stability": "{:.3f}",
                    }),
                    use_container_width=True, hide_index=True,
                )

                # Highlight all-weather factors
                aw = [f for f in stability if f.get("all_weather")]
                if aw:
                    aw_names = [f["factor"] for f in aw[:5]]
                    st.info(f'**All-Weather Factors**: {", ".join(aw_names)}')


# ─────────────────────────────────────────────────────────────────────────────
# TAB: AI PREDICTION  (Forward 1M regime classifier — P1~P4 pipeline results)
# ─────────────────────────────────────────────────────────────────────────────
with tab_ai_pred:
    import json as _json

    st.subheader("AI Regime Prediction (Forward 1M)")
    st.caption(
        "MSCI ACWI / Bloomberg Global Agg forward 1-month regime classifier. "
        "LightGBM multiclass + meta-labeling (Lopez de Prado). "
        "Allocation grids — BULL 90/5/5, BASE 75/10/15, BEAR 60/15/25 (equity/bond/cash). "
        "**Benchmark: ACWI 90 / Cash 10.**"
    )

    # ── Model Definitions (P0 → P4) ──────────────────────────────────────────
    st.markdown("##### Model Definitions (P0 → P4)")
    st.caption("Incremental ablation — each phase isolates one design decision.")
    _phase_defs = [
        ("P0", "Baseline", C["cyan"], False,
         "Reference model without enhancements.",
         [
             "LightGBM multiclass on 15 macro / cross-asset features",
             "Purged Walk-Forward CV with 21-day embargo (Lopez de Prado)",
             "Posterior-weighted blend across BULL / BASE / BEAR grids",
             "Reference point for ablation attribution",
         ]),
        ("P1", "Breadth Features", C["orange"], False,
         "Bottom-up signal from price_discovery replay.",
         [
             "Monthly historical replay over 170 ETFs (2007-07~)",
             "6 aggregate features: equity (4) × bond (2)",
             "eq_pct_bullish / eq_pct_downtrend / eq_tcs_median / eq_rss_std",
             "bd_pct_bullish / bd_tcs_median",
             "BlackRock SAE-style aggregation",
         ]),
        ("P2", "BULL Detection", C["yellow"], False,
         "Address minority-class imbalance (BULL = 14%).",
         [
             "Class weight sweep: BULL {1.5, 2.0, 2.5, 3.0}",
             "BULL threshold override: P(BULL) > {0.25, 0.30, 0.35}",
             "4 × 3 = 12 variants in the sweep",
             "VIX-free label ablation (test circularity hypothesis)",
         ]),
        ("P3", "Additional Macro Features", C["blue"], False,
         "+6 cross-asset / term-structure features.",
         [
             "vix_move_ratio (equity vol / bond vol)",
             "vix_vxv_ratio (VIX term structure)",
             "wti_copper_ratio (inflation vs growth)",
             "dxy_z252 (USD 252-day z-score)",
             "hy_ig_spread_level (credit ratio)",
             "tip_ief_ratio (inflation breakeven proxy)",
         ]),
        ("P4", "Meta-Labeling", C["green"], True,
         "Two-stage ensemble — Lopez de Prado AFML Ch.3. **Production winner.**",
         [
             "Primary = multiclass argmax from P3-featured LightGBM",
             "Meta = binary classifier: 'will primary be correct?'",
             "If meta confidence < 0.5 → fallback to BASE (75/10/15)",
             "Cuts turnover ~7% → ~3.5%/mo and MaxDD to −16.5%",
             "Highest Sharpe, lowest DD vs benchmark",
         ]),
    ]

    _def_cols = st.columns(5)
    for col, (tag, title, color, is_winner, short, bullets) in zip(_def_cols, _phase_defs):
        with col:
            win_badge = " <span style='font-size:9px;color:#22c55e;font-weight:600'>WINNER</span>" if is_winner else ""
            border_style = "1px solid rgba(34,197,94,0.55);box-shadow:0 0 0 1px rgba(34,197,94,0.25)" if is_winner else "1px solid #1f2937"
            bullets_html = "".join(f"<li>{b}</li>" for b in bullets)
            st.markdown(
                f"""
<div style="background:#0a0e17;border:{border_style};border-radius:8px;padding:10px;min-height:260px">
  <div style="display:flex;justify-content:space-between;align-items:baseline">
    <span style="color:{color};font-size:18px;font-weight:700">{tag}</span>{win_badge}
  </div>
  <div style="color:#e5e7eb;font-size:13px;font-weight:600;margin-top:2px">{title}</div>
  <div style="color:#9ca3af;font-size:10.5px;font-style:italic;margin:2px 0 6px">{short}</div>
  <ul style="color:#d1d5db;font-size:10px;line-height:1.35;padding-left:14px;margin:0">
    {bullets_html}
  </ul>
</div>
""", unsafe_allow_html=True)

    st.caption(
        "**Composition** — P4 sits on top of P3 features and uses the same CV as P0. "
        "P1 and P2 are independent variants (ablation showed marginal lift only). "
        "MoE (Plan B, below) attempts to combine experts conditional on regime."
    )
    st.divider()

    # ── Load cached artifacts ────────────────────────────────────────────────
    cache_ok = True
    try:
        with open(os.path.join(SCRIPT_DIR, "ai_pred_metrics.json")) as f:
            ai_metrics = _json.load(f)
        ai_proba = pd.read_csv(os.path.join(SCRIPT_DIR, "ai_pred_proba.csv"),
                               index_col=0, parse_dates=True)
        ai_ret   = pd.read_csv(os.path.join(SCRIPT_DIR, "ai_pred_returns.csv"),
                               index_col=0, parse_dates=True)
        ai_fi    = pd.read_csv(os.path.join(SCRIPT_DIR, "ai_pred_feature_imp.csv"),
                               index_col=0)
        ai_cm    = pd.read_csv(os.path.join(SCRIPT_DIR, "ai_pred_confusion.csv"),
                               index_col=0)
        ai_abl   = pd.read_csv(os.path.join(SCRIPT_DIR, "ablation_results.csv"))
    except FileNotFoundError as e:
        cache_ok = False
        st.warning(
            f"AI prediction cache missing ({e.filename}). "
            f"Run `python3 ai_prediction_cache.py` to generate it."
        )

    if cache_ok:
        # ── KPI row: winner vs benchmark vs baseline ─────────────────────────
        st.markdown("#### Winner: P4 meta-labeled ensemble")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Sharpe",
                  f"{ai_metrics['winner_sharpe']:.2f}",
                  f"{ai_metrics['winner_sharpe'] - ai_metrics['benchmark_sharpe']:+.2f} vs BM")
        k2.metric("Alpha (ann)",
                  f"{ai_metrics['winner_alpha_ann']*100:+.2f}%",
                  delta_color=("normal" if ai_metrics['winner_alpha_ann'] > 0 else "inverse"))
        k3.metric("Max DD",
                  f"{ai_metrics['winner_max_dd']*100:.1f}%",
                  f"{(ai_metrics['winner_max_dd'] - ai_metrics['benchmark_max_dd'])*100:+.1f}pp")
        k4.metric("Turnover",
                  f"{ai_metrics['winner_turnover']:.1f}%/mo",
                  f"{ai_metrics['winner_turnover'] - ai_metrics['baseline_turnover']:+.1f}pp")
        k5.metric("BULL recall",
                  f"{ai_metrics['winner_bull_recall']*100:.0f}%",
                  f"{(ai_metrics['winner_bull_recall'] - ai_metrics['baseline_bull_recall'])*100:+.0f}pp")
        k6.metric("OOS months",
                  f"{ai_metrics['n_oos_months']}",
                  f"{ai_metrics['period_start']} → {ai_metrics['period_end']}")

        st.caption(
            f"**Meta diagnostics** — "
            f"hit rate {ai_metrics['meta_hit_rate']*100:.1f}%, "
            f"agreement with primary {ai_metrics['meta_agreement']*100:.1f}%, "
            f"override→BASE rate {ai_metrics['meta_override_rate']*100:.1f}%. "
            f"Dataset: {ai_metrics['winner_n_features']} features × {ai_metrics['winner_n_rows']} months."
        )

        st.divider()

        # ── Charts row 1: equity curve + allocation weights ──────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Cumulative Return (OOS)")
            bm_label = f"Benchmark {ai_metrics.get('benchmark_label', 'ACWI 90 / Cash 10')}"
            eq_df = (1 + ai_ret[["strategy_ret", "benchmark_static_ret",
                                  "p0_baseline_ret"]].fillna(0)).cumprod()
            eq_df.columns = ["P4 meta (winner)", bm_label, "P0 baseline"]
            fig_eq = go.Figure()
            colors_eq = {"P4 meta (winner)": C["green"],
                         bm_label: C["gray"],
                         "P0 baseline": C["orange"]}
            for c in eq_df.columns:
                fig_eq.add_trace(go.Scatter(
                    x=eq_df.index, y=eq_df[c], mode="lines", name=c,
                    line=dict(color=colors_eq[c], width=2),
                ))
            fig_eq.update_layout(
                height=360, template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified", yaxis_title="Cumulative",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_eq, use_container_width=True)

        with col_b:
            st.markdown("##### Allocation Weights Over Time")
            w_df = ai_ret[["w_equity", "w_bond", "w_cash"]].copy()
            w_df.columns = ["Equity (ACWI)", "Bond (GlobalAgg)", "Cash"]
            fig_w = go.Figure()
            colors_w = {"Equity (ACWI)": C["green"],
                        "Bond (GlobalAgg)": C["blue"],
                        "Cash":  C["yellow"]}
            for c in w_df.columns:
                fig_w.add_trace(go.Scatter(
                    x=w_df.index, y=w_df[c], mode="lines",
                    name=c, stackgroup="one",
                    line=dict(width=0.5, color=colors_w[c]),
                    fillcolor=colors_w[c],
                ))
            fig_w.update_layout(
                height=360, template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        # ── Charts row 2: regime probability heatmap + confusion matrix ──────
        col_c, col_d = st.columns([2, 1])

        with col_c:
            st.markdown("##### OOS Regime Probabilities (Primary Model)")
            proba_long = ai_proba[["BEAR", "BASE", "BULL"]].copy()
            fig_pr = go.Figure()
            regime_colors = {"BEAR": C["red"], "BASE": C["gray"], "BULL": C["green"]}
            for r in ["BEAR", "BASE", "BULL"]:
                fig_pr.add_trace(go.Scatter(
                    x=proba_long.index, y=proba_long[r],
                    mode="lines", name=r, stackgroup="one",
                    line=dict(width=0.3, color=regime_colors[r]),
                    fillcolor=regime_colors[r],
                ))
            fig_pr.update_layout(
                height=360, template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                hovermode="x unified",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

        with col_d:
            st.markdown("##### Confusion Matrix (OOS)")
            cm = ai_cm.copy()
            cm.index = [i.replace("true_", "true: ") for i in cm.index]
            cm.columns = [c.replace("pred_", "pred: ") for c in cm.columns]
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm.values, x=list(cm.columns), y=list(cm.index),
                colorscale="Blues", showscale=False,
                text=cm.values, texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
            ))
            fig_cm.update_layout(
                height=360, template="plotly_dark",
                margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            total = cm.values.sum()
            correct = np.trace(cm.values)
            st.caption(f"Overall accuracy: **{correct/total*100:.1f}%**  ({correct}/{total})")

        st.divider()

        # ── Feature importance ───────────────────────────────────────────────
        st.markdown("##### Feature Importance (LightGBM mean gain across folds)")
        fi_top = ai_fi.head(20).reset_index()
        fi_top.columns = ["feature", "importance"]
        fi_top["is_breadth"] = fi_top["feature"].str.startswith(("eq_", "bd_"))
        fig_fi = px.bar(
            fi_top.sort_values("importance"),
            x="importance", y="feature", orientation="h",
            color="is_breadth",
            color_discrete_map={True: C["purple"], False: C["cyan"]},
            labels={"is_breadth": "Breadth (P1)"},
            height=500,
        )
        fig_fi.update_layout(template="plotly_dark",
                             margin=dict(l=10, r=10, t=10, b=10),
                             showlegend=True,
                             legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99))
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("Purple = breadth features from historical price_discovery replay (P1). "
                   "Cyan = macro / cross-asset features (baseline + P3).")

        st.divider()

        # ── Ablation results table ───────────────────────────────────────────
        st.markdown("##### P1-P4 Ablation Results (19 variants)")
        abl = ai_abl.copy()
        keep = ["variant", "sharpe", "alpha_ann", "bull_recall", "bear_recall",
                "max_dd", "mean_turnover_pct"]
        abl = abl[[c for c in keep if c in abl.columns]].dropna(subset=["sharpe"])
        abl = abl.sort_values("alpha_ann", ascending=False).reset_index(drop=True)

        def _row_style(row):
            if "P4_meta" in row["variant"]:
                return ["background-color: rgba(34,197,94,0.18)"] * len(row)
            if "VIXFREE" in row["variant"]:
                return ["background-color: rgba(139,92,246,0.12)"] * len(row)
            if "P1_breadth" in row["variant"]:
                return ["background-color: rgba(245,158,11,0.10)"] * len(row)
            return [""] * len(row)

        styled = (abl.style
                  .apply(_row_style, axis=1)
                  .format({"sharpe": "{:.3f}", "alpha_ann": "{:+.2%}",
                           "bull_recall": "{:.0%}", "bear_recall": "{:.0%}",
                           "max_dd": "{:.1%}",
                           "mean_turnover_pct": "{:.2f}%"}))
        st.dataframe(styled, use_container_width=True, height=560)

        st.caption("Green = P4 meta-labeling (winner). "
                   "Purple = VIX-free label ablation. "
                   "Orange = P1 breadth variants.")

        st.divider()

        # ── Ablation scatter: Sharpe vs Alpha ────────────────────────────────
        st.markdown("##### Sharpe vs Alpha (per variant)")
        abl2 = ai_abl.copy().dropna(subset=["sharpe"])
        abl2["tag"] = abl2["variant"].apply(
            lambda v: ("P4 meta" if "P4_meta" in v else
                       "VIX-free" if "VIXFREE" in v else
                       "P1 breadth" if "P1_breadth" in v else
                       "P0 kfold"   if "kfold" in v else
                       "P0/P2/P3")
        )
        fig_sc = px.scatter(
            abl2, x="alpha_ann", y="sharpe",
            color="tag", size="bull_recall", size_max=25,
            hover_data=["variant", "max_dd", "mean_turnover_pct"],
            color_discrete_map={"P4 meta": C["green"], "VIX-free": C["purple"],
                                "P1 breadth": C["orange"], "P0 kfold": C["red"],
                                "P0/P2/P3": C["cyan"]},
            labels={"alpha_ann": "Annual Alpha", "sharpe": "Sharpe Ratio"},
            height=430,
        )
        fig_sc.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_sc.update_layout(template="plotly_dark",
                             margin=dict(l=10, r=10, t=10, b=10),
                             xaxis_tickformat=".2%")
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(f"Point size = BULL recall. Dashed line = 0% alpha vs "
                   f"{ai_metrics.get('benchmark_label', 'ACWI 90 / Cash 10')}.")

        st.divider()

        # ── Plan B: MoE (regime-conditional expert blending) comparison ─────
        moe_summary_path = os.path.join(SCRIPT_DIR, "ai_pred_moe.json")
        moe_monthly_path = os.path.join(SCRIPT_DIR, "ai_pred_moe.csv")
        if os.path.exists(moe_summary_path) and os.path.exists(moe_monthly_path):
            with open(moe_summary_path) as _f:
                moe_s = _json.load(_f)
            moe_m = pd.read_csv(moe_monthly_path, index_col=0, parse_dates=True)

            st.markdown("##### Plan B · Regime-Conditional Expert Blending (MoE) — Comparison")
            st.caption(
                f"Common window {moe_s['period_start']} → {moe_s['period_end']} "
                f"({moe_s['n_oos_months']} months). "
                f"Experts: **BULL→{moe_s['expert_assignment']['BULL']}**, "
                f"**BASE→{moe_s['expert_assignment']['BASE']}**, "
                f"**BEAR→{moe_s['expert_assignment']['BEAR']}**. Gate: {moe_s['gate']}."
            )

            moe_rows = [
                ("P4_meta solo (current winner)", moe_s["p4_solo"], "winner"),
                ("MoE hard (argmax gate)",        moe_s["moe_hard"], "moe"),
                ("MoE soft (proba-weighted)",     moe_s["moe_soft"], "moe"),
                ("P0 baseline solo",              moe_s["p0_solo"],  "base"),
            ]
            tbl_df = pd.DataFrame([
                {
                    "Strategy": name,
                    "AnnRet": f"{s['ann_return']*100:.2f}%",
                    "Sharpe": f"{s['sharpe']:.2f}",
                    "Max DD": f"{s['max_dd']*100:.1f}%",
                    "Alpha (ann)": f"{'+' if s['alpha_ann']>=0 else ''}{s['alpha_ann']*100:.2f}%",
                    "Turnover": f"{s['turnover']:.2f}%/mo",
                    "__flag": flag,
                }
                for name, s, flag in moe_rows
            ])
            bm_s = moe_s["benchmark"]
            tbl_df.loc[len(tbl_df)] = {
                "Strategy": f"Benchmark {bm_s['label']}",
                "AnnRet": f"{bm_s['ann_return']*100:.2f}%",
                "Sharpe": f"{bm_s['sharpe']:.2f}",
                "Max DD": f"{bm_s['max_dd']*100:.1f}%",
                "Alpha (ann)": "—",
                "Turnover": "—",
                "__flag": "bench",
            }

            def _moe_row_style(row):
                if row["__flag"] == "winner":
                    return ["background-color: rgba(34,197,94,0.18)"] * len(row)
                if row["__flag"] == "moe":
                    return ["background-color: rgba(139,92,246,0.12)"] * len(row)
                if row["__flag"] == "bench":
                    return ["color: rgba(156,163,175,0.9); font-style: italic"] * len(row)
                return [""] * len(row)

            st.dataframe(
                tbl_df.style.apply(_moe_row_style, axis=1).hide(axis="index"),
                use_container_width=True,
                column_config={"__flag": None},
            )

            col_moe1, col_moe2 = st.columns(2)

            with col_moe1:
                # Cumulative chart
                import numpy as _np
                dates = moe_m.index
                soft_curve = _np.cumprod(1 + moe_m["soft_ret"].values)
                hard_curve = _np.cumprod(1 + moe_m["hard_ret"].values)
                bench_curve = _np.cumprod(1 + moe_m["bench_ret"].values)
                fig_moe_eq = go.Figure()
                fig_moe_eq.add_trace(go.Scatter(x=dates, y=soft_curve, mode="lines",
                                                name="MoE soft", line=dict(color=C["purple"], width=2)))
                fig_moe_eq.add_trace(go.Scatter(x=dates, y=hard_curve, mode="lines",
                                                name="MoE hard", line=dict(color=C["orange"], width=2)))
                fig_moe_eq.add_trace(go.Scatter(x=dates, y=bench_curve, mode="lines",
                                                name="Benchmark", line=dict(color=C["gray"], width=2, dash="dot")))
                fig_moe_eq.update_layout(height=300, template="plotly_dark",
                                         margin=dict(l=10, r=10, t=10, b=10),
                                         hovermode="x unified",
                                         yaxis_title="Cumulative",
                                         legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.markdown("**MoE vs Benchmark — Cumulative (common window)**")
                st.plotly_chart(fig_moe_eq, use_container_width=True)

            with col_moe2:
                # Per-regime alpha bar
                reg_true = moe_m["true_regime"].values
                reg_labels = ["BULL", "BASE", "BEAR"]
                soft_alpha_bp = []; hard_alpha_bp = []; ns = []
                for r in reg_labels:
                    mask = reg_true == r
                    ns.append(int(mask.sum()))
                    if mask.any():
                        soft_alpha_bp.append(float(
                            (moe_m.loc[mask, "soft_ret"] - moe_m.loc[mask, "bench_ret"]).mean() * 10000
                        ))
                        hard_alpha_bp.append(float(
                            (moe_m.loc[mask, "hard_ret"] - moe_m.loc[mask, "bench_ret"]).mean() * 10000
                        ))
                    else:
                        soft_alpha_bp.append(0.0); hard_alpha_bp.append(0.0)
                fig_moe_reg = go.Figure()
                fig_moe_reg.add_trace(go.Bar(x=reg_labels, y=soft_alpha_bp, name="MoE soft",
                                             marker_color=C["purple"],
                                             text=[f"{v:.0f}" for v in soft_alpha_bp],
                                             textposition="auto"))
                fig_moe_reg.add_trace(go.Bar(x=reg_labels, y=hard_alpha_bp, name="MoE hard",
                                             marker_color=C["orange"],
                                             text=[f"{v:.0f}" for v in hard_alpha_bp],
                                             textposition="auto"))
                fig_moe_reg.update_layout(height=300, template="plotly_dark",
                                          margin=dict(l=10, r=10, t=10, b=10),
                                          barmode="group",
                                          yaxis_title="bp / month",
                                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
                st.markdown("**Per-Regime Alpha (bp / month, true regime)**")
                st.plotly_chart(fig_moe_reg, use_container_width=True)
                st.caption(f"n: BULL={ns[0]} · BASE={ns[1]} · BEAR={ns[2]}")

            # Gate argmax distribution
            from collections import Counter
            gate_cnt = Counter(moe_m["gate_argmax"].values)
            total_g = len(moe_m)
            g1, g2, g3 = st.columns(3)
            for _col, _r in zip([g1, g2, g3], ["BEAR", "BASE", "BULL"]):
                n = gate_cnt.get(_r, 0)
                _col.metric(f"Gate = {_r}", f"{n} / {total_g}",
                            f"{n/total_g*100:.1f}% of months")

            st.info(
                "**Verdict** — MoE does NOT beat P4_meta solo. "
                "P4's meta-labeling (override→BASE when low confidence) is already a "
                "stronger form of regime-conditional routing than external gating; plus "
                "the gate (P0 primary) has BULL recall ~16%, so argmax under-routes to "
                "the BULL expert. Oracle upper bound ≈ −0.4%/yr alpha — still negative "
                "vs the aggressive 90/10 benchmark. **Production choice remains P4_meta.**"
            )

            st.divider()

        # ── Methodology / findings ──────────────────────────────────────────
        st.markdown("##### Methodology & Key Findings")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
**Pipeline (P0 → P4)**

- **P0**: Purged Walk-Forward CV with 21-day embargo (Lopez de Prado)
- **P3**: 6 macro cross-asset features (VIX/MOVE, VIX/VXV, WTI/Copper, DXY z-score,
  HY/IG spread, TIP/IEF)
- **P2**: Class weight sweep × BULL-threshold sweep (4×3 = 12 variants)
- **VIX-free ablation**: Test whether VIX-in-label circularity caps BULL recall
- **P1**: Monthly historical replay of price_discovery over 170 ETFs →
  6 breadth features (eq/bd × pct_bullish, pct_downtrend, tcs_median, rss_std)
- **P4 (winner)**: Meta-labeling — primary = multiclass argmax,
  meta = binary "is primary correct?" → BASE fallback if meta confidence < 0.5
""")
        with col_m2:
            bm_lbl = ai_metrics.get("benchmark_label", "ACWI 90 / Cash 10")
            st.markdown(f"""
**What moved the needle**

- **Meta-labeling (P4)** — highest Sharpe **{ai_metrics['winner_sharpe']:.2f}** vs benchmark
  **{ai_metrics['benchmark_sharpe']:.2f}**, lowest Max DD **{ai_metrics['winner_max_dd']*100:.1f}%** vs
  **{ai_metrics['benchmark_max_dd']*100:.1f}%** — turnover cut to
  **{ai_metrics['winner_turnover']:.1f}%/mo**
- **Alpha vs {bm_lbl}**: **{ai_metrics['winner_alpha_ann']*100:+.2f}% ann** —
  absolute underperformance in bull tape, but superior risk-adjusted profile
- **P3 macro features** doubled BULL recall 8% → {ai_metrics['winner_bull_recall']*100:.0f}%
- **VIX-free ablation** lowered BULL recall — label circularity is NOT the bottleneck
- **BULL recall ceiling ~16%** with {ai_metrics['winner_n_rows']} monthly samples —
  primarily a data-volume constraint
- All 3 leakage tests **PASS** (feature as-of, breadth as-of, label correlation < 0.2)
""")

        st.info(
            "**Production config** — "
            "`use_meta=True`, `class_weight={BULL: 3.0}`, `bull_threshold=0.25`, "
            "`cv_mode='walkforward'`, 22 macro + 6 breadth features. "
            "Regenerate cache via `python3 ai_prediction_cache.py`."
        )


with tab_report:
    st.subheader("Analytical Report")

    report_md = _build_report(fdf, df, history, ve_stats)

    mode = st.radio("Mode", ["Auto Report (Data-Driven)", "LLM Prompt (Copy for AI)"],
                    horizontal=True)

    if mode == "Auto Report (Data-Driven)":
        st.caption("스캐너 결과 기반 자동 생성 리포트 — Markdown 형식")

        # Download button
        st.download_button(
            "Download Report (.md)", report_md.encode("utf-8"),
            f"PD_Report_{datetime.now().strftime('%Y%m%d')}.md",
            "text/markdown", use_container_width=False,
        )

        st.markdown(report_md)

    else:
        st.caption("아래 프롬프트를 복사하여 Claude / ChatGPT에 붙여넣으면 A4 20페이지 분량의 전문 코멘터리가 생성됩니다.")

        llm_prompt = _build_llm_prompt(report_md, fdf, ve_stats)

        st.download_button(
            "Download Prompt (.txt)", llm_prompt.encode("utf-8"),
            f"PD_LLM_Prompt_{datetime.now().strftime('%Y%m%d')}.txt",
            "text/plain", use_container_width=False,
        )

        st.text_area("LLM Prompt", llm_prompt, height=600)
        st.info(f"Prompt length: {len(llm_prompt):,} chars (~{len(llm_prompt)//4:,} tokens)")
