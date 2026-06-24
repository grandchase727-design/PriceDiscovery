"""FastAPI client — calls existing backend (localhost:8000).

Streamlit dashboard delegates ALL data + scoring to the FastAPI server,
matching the React dashboard's API consumption pattern.
"""
from __future__ import annotations

import requests
import streamlit as st

BASE_URL = "http://localhost:8000/api"


def _get(path: str, params: dict | None = None, timeout: int = 30) -> dict | list:
    try:
        r = requests.get(f"{BASE_URL}{path}", params=params or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API error {path}: {e}")
        return {}


# Cache layer — Streamlit reruns on every interaction
@st.cache_data(ttl=60)
def fetch_table(sectors: tuple | None = None, classifications: tuple | None = None,
                eligible_only: bool = False, comp_min: int = 0, comp_max: int = 100) -> dict:
    params = {"comp_min": comp_min, "comp_max": comp_max, "eligible_only": eligible_only}
    if sectors:         params["sectors"] = ",".join(sectors)
    if classifications: params["classifications"] = ",".join(classifications)
    return _get("/table", params)


@st.cache_data(ttl=60)
def fetch_classification_history() -> dict:
    return _get("/classification-history")


@st.cache_data(ttl=60)
def fetch_market_regime(sectors: tuple | None = None) -> dict:
    params = {}
    if sectors: params["sectors"] = ",".join(sectors)
    return _get("/market-regime", params)


@st.cache_data(ttl=60)
def fetch_validation() -> dict:
    return _get("/validation")


@st.cache_data(ttl=60)
def fetch_quant_strategies() -> dict:
    return _get("/quant-strategies")


@st.cache_data(ttl=300)
def fetch_meta() -> dict:
    return _get("/meta")


@st.cache_data(ttl=60)
def fetch_final_list() -> dict:
    return _get("/final-list", timeout=20)


@st.cache_data(ttl=60)
def fetch_multi_agent_debate() -> dict:
    return _get("/multi-agent-debate")


@st.cache_data(ttl=30)
def fetch_swarm_result() -> dict:
    return _get("/market-leaders/swarm/result")


@st.cache_data(ttl=10)
def fetch_swarm_status() -> dict:
    return _get("/market-leaders/swarm/status")


@st.cache_data(ttl=30)
def fetch_backtest_results() -> dict:
    return _get("/backtest/results")


@st.cache_data(ttl=30)
def fetch_backtest_rankings() -> dict:
    return _get("/backtest/rankings")


@st.cache_data(ttl=10)
def fetch_backtest_status() -> dict:
    return _get("/backtest/status")


@st.cache_data(ttl=300)
def fetch_validated_extra_timeline() -> dict:
    return _get("/validated-extra-timeline")


@st.cache_data(ttl=60)
def fetch_pm_history_summary() -> dict:
    return _get("/pm-history-summary")


def clear_caches() -> None:
    """Clear all cached API responses."""
    fetch_table.clear()
    fetch_classification_history.clear()
    fetch_market_regime.clear()
    fetch_validation.clear()
    fetch_quant_strategies.clear()
    fetch_meta.clear()
    fetch_final_list.clear()
    fetch_multi_agent_debate.clear()
    fetch_swarm_result.clear()
    fetch_swarm_status.clear()
    fetch_backtest_results.clear()
    fetch_backtest_rankings.clear()
    fetch_backtest_status.clear()
    fetch_pm_history_summary.clear()
