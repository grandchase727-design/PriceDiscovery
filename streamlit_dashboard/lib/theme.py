"""Theme system — dark/light mode toggle with React-parity color tokens.

Mirrors frontend/src/styles/theme.ts so visual identity is preserved.
"""
from __future__ import annotations

import streamlit as st


# Dark theme (default — matches existing React dashboard)
DARK = {
    "bg":      "#0f1419",
    "bgAlt":   "#1a1f2e",
    "border":  "#2a3142",
    "text":    "#e2e6ee",
    "gray":    "#7884a0",
    "green":   "#16c784",
    "amber":   "#f1b03e",
    "yellow":  "#facc15",
    "red":     "#ea3943",
    "cyan":    "#22b8cf",
    "purple":  "#9961ff",
    "orange":  "#fb923c",
    "blue":    "#3b82f6",
}

# Light theme (high-contrast variant for daylight viewing)
LIGHT = {
    "bg":      "#fafbfc",
    "bgAlt":   "#ffffff",
    "border":  "#d6dae3",
    "text":    "#0e1320",
    "gray":    "#5b6678",
    "green":   "#06994c",
    "amber":   "#b87800",
    "yellow":  "#a36b00",
    "red":     "#c1192e",
    "cyan":    "#0589a0",
    "purple":  "#6d3bc8",
    "orange":  "#c44d0f",
    "blue":    "#2057c4",
}


def get_theme() -> dict:
    """Return active theme dict based on session state toggle."""
    mode = st.session_state.get("theme_mode", "dark")
    return DARK if mode == "dark" else LIGHT


def set_theme_mode(mode: str) -> None:
    st.session_state["theme_mode"] = mode


def inject_css() -> None:
    """Inject CSS to override Streamlit defaults with active theme."""
    C = get_theme()
    mode = st.session_state.get("theme_mode", "dark")
    css = f"""
    <style>
      /* Whole app background */
      [data-testid="stAppViewContainer"],
      [data-testid="stHeader"] {{
        background-color: {C['bg']} !important;
      }}
      [data-testid="stSidebar"] {{
        background-color: {C['bgAlt']} !important;
      }}
      /* Main text */
      .stMarkdown, .stText, p, span, label, li,
      [data-testid="stMarkdownContainer"] {{
        color: {C['text']} !important;
      }}
      /* Section cards */
      .pd-card {{
        background-color: {C['bgAlt']};
        border: 1px solid {C['border']};
        border-radius: 6px;
        padding: 12px 14px;
        margin-bottom: 12px;
      }}
      .pd-section-wrap {{
        border-left: 3px solid var(--accent, {C['cyan']});
        padding: 6px 14px;
        margin-bottom: 16px;
        background-color: {C['bgAlt']};
        border-radius: 0 4px 4px 0;
      }}
      .pd-section-title {{
        font-size: 14px;
        font-weight: bold;
        margin-bottom: 8px;
        color: var(--accent, {C['cyan']});
      }}
      .pd-section-body {{
        font-size: 12.5px;
        line-height: 1.65;
        color: {C['text']};
      }}
      .pd-pill {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: bold;
        margin-right: 4px;
        white-space: nowrap;
      }}
      .pd-metric {{
        display: inline-block;
        margin-right: 12px;
      }}
      .pd-metric-label {{
        color: {C['gray']};
        font-size: 10px;
      }}
      .pd-metric-value {{
        color: {C['text']};
        font-weight: bold;
        font-size: 14px;
      }}
      .pd-mono {{
        font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      }}
      /* Tables (st.dataframe) — color cells */
      .stDataFrame {{
        border: 1px solid {C['border']};
        border-radius: 4px;
      }}
      /* Buttons */
      .stButton button {{
        background-color: {C['bgAlt']};
        color: {C['text']};
        border: 1px solid {C['border']};
        border-radius: 4px;
      }}
      .stButton button:hover {{
        border-color: {C['cyan']};
        color: {C['cyan']};
      }}
      /* Sidebar widgets */
      [data-testid="stSidebar"] .stMarkdown {{
        color: {C['text']} !important;
      }}
      /* Tabs */
      .stTabs [data-baseweb="tab"] {{
        color: {C['gray']};
      }}
      .stTabs [aria-selected="true"] {{
        color: {C['cyan']} !important;
        font-weight: bold;
      }}
      /* Code/pre blocks */
      code, pre {{
        background-color: {C['bg']} !important;
        color: {C['cyan']} !important;
      }}
      /* Plotly background */
      .js-plotly-plot .plotly {{
        background-color: transparent !important;
      }}
      /* Expander */
      .streamlit-expanderHeader {{
        background-color: {C['bgAlt']};
        color: {C['text']};
        border: 1px solid {C['border']};
      }}
      /* Custom badge classes */
      .pd-badge-green  {{ background-color: {C['green']}25;  color: {C['green']};  border: 1px solid {C['green']}80; }}
      .pd-badge-red    {{ background-color: {C['red']}25;    color: {C['red']};    border: 1px solid {C['red']}80; }}
      .pd-badge-cyan   {{ background-color: {C['cyan']}25;   color: {C['cyan']};   border: 1px solid {C['cyan']}80; }}
      .pd-badge-amber  {{ background-color: {C['amber']}25;  color: {C['amber']};  border: 1px solid {C['amber']}80; }}
      .pd-badge-purple {{ background-color: {C['purple']}25; color: {C['purple']}; border: 1px solid {C['purple']}80; }}
      .pd-badge-gray   {{ background-color: {C['gray']}25;   color: {C['gray']};   border: 1px solid {C['gray']}80; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def theme_toggle_widget() -> None:
    """Sidebar widget for theme toggle (radio buttons)."""
    cur = st.session_state.get("theme_mode", "dark")
    label_map = {"dark": "🌑 Dark", "light": "☀️ Light"}
    mode = st.sidebar.radio(
        "Theme",
        options=["dark", "light"],
        format_func=lambda x: label_map[x],
        index=0 if cur == "dark" else 1,
        horizontal=True,
        key="theme_mode",
    )
    return mode


def plotly_theme(C: dict) -> dict:
    """Return Plotly layout dict matching current theme."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": C["text"], "family": "ui-sans-serif, system-ui"},
        "xaxis": {"gridcolor": C["border"], "zerolinecolor": C["border"], "color": C["text"]},
        "yaxis": {"gridcolor": C["border"], "zerolinecolor": C["border"], "color": C["text"]},
        "margin": {"l": 40, "r": 20, "t": 40, "b": 40},
    }
