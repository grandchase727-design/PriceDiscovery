"""Top-level widgets: Market Leaders + Executive Summary + Conviction Picks."""
from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
import streamlit as st

from .theme import get_theme
from .utils import (
    badge, colored_pct, colored_span, comp_color, format_name, fmt_pct,
    MOMENTUM_SET, PM_SET, EXCLUDED_SET,
)


# ─────────────────────────────────────────────────────────────────
# Executive Summary
# ─────────────────────────────────────────────────────────────────
def render_executive_summary(stats: dict, regime_stats: dict, val_stats: dict, quant_stats: dict) -> None:
    if not stats:
        return
    C = get_theme()
    regime = stats.get("regime", "?")
    mom_pct = stats.get("momentum_pct", 0)
    pm_pct = stats.get("pm_pct", 0)
    exc_pct = stats.get("excluded_pct", 0)
    avg_ytd = stats.get("avg_ytd", 0)
    eligible_pct = stats.get("eligible_pct", 0)
    breadth_pct = stats.get("breadth_pct", 0)
    top_sec = stats.get("comp_sec", [])
    top_leader = top_sec[0]["sec"] if top_sec else "—"
    nd = quant_stats.get("net_direction", "MIXED") if quant_stats else "—"
    val_avg = ((val_stats.get("overall_mom", 0) + val_stats.get("overall_pm", 0)) / 2) if val_stats else 0

    # Tone
    if regime == "건전한 상승 추세":
        tone = "다층 강세"; tc = C["green"]
    elif regime == "약세·방어 국면":
        tone = "다층 약세"; tc = C["red"]
    elif regime == "과열 단계":
        tone = "부분 강세 — 과열 경고"; tc = C["amber"]
    elif regime == "회복·전환 국면":
        tone = "부분 강세"; tc = C["cyan"]
    else:
        tone = "방향성 미확정"; tc = C["gray"]

    html = (
        f'<div class="pd-card" style="border:2px solid {tc}80">'
        f'<h3 style="color:{tc};margin:0 0 8px 0">📋 Executive Summary</h3>'
        f'<p style="font-size:13px;line-height:1.7;color:{C["text"]};margin:0">'
        f'현재 시장은 <b>{badge(regime, tc)}</b> 진단 — Mom {mom_pct:.1f}% / PM {pm_pct:.1f}% / Excl {exc_pct:.1f}%. '
        f'Composite eligible(≥55) <b>{eligible_pct:.1f}%</b>, YTD breadth <b>{breadth_pct:.1f}%</b>, '
        f'평균 YTD {colored_pct(avg_ytd)}. '
        f'주도 섹터 <b>{top_leader}</b>. '
        f'Validation 평균 {val_avg:.1f}% — 시스템 신뢰도 {"양호" if val_avg > 65 else "보정 필요"}. '
        f'Quant strategy 합산 방향 {badge(nd, C["green"] if nd=="LONG" else (C["red"] if nd=="SHORT" else C["gray"]))}. '
        f'종합 톤: {colored_span(tone, tc, True)}.'
        f'</p></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# Market Leaders Panel (4 cards)
# ─────────────────────────────────────────────────────────────────
def render_market_leaders(stats: dict, regime_stats: dict, quant_stats: dict, all_results: list[dict]) -> None:
    if not stats:
        return
    C = get_theme()

    st.markdown("### 🏆 Market Leaders")

    cols = st.columns([1, 1, 1, 1])

    # Card 1: 주도주
    with cols[0]:
        html = f'<div class="pd-card"><h4 style="color:{C["green"]};margin:0 0 8px 0">📈 주도주</h4>'
        top_cont = stats.get("top_cont", [])[:3]
        if top_cont:
            html += '<p style="font-size:11px;color:' + C["gray"] + '">CONTINUATION Top 3</p>'
            html += '<ul style="margin:2px 0 6px 14px;font-size:11px">'
            for r in top_cont:
                html += (f"<li>{format_name(r)} · {r.get('sector', '')[:18]}<br>"
                         f"<span style='font-size:10px'>Comp {r.get('composite', 0):.1f} · "
                         f"1M {fmt_pct(r.get('ret_1m'))}</span></li>")
            html += '</ul>'
        ultra = quant_stats.get("ultra_consensus", [])[:2] if quant_stats else []
        if ultra:
            html += '<p style="font-size:11px;color:' + C["purple"] + '">🌟 Ultra-Consensus</p>'
            html += '<ul style="margin:2px 0 6px 14px;font-size:11px">'
            for u in ultra:
                html += f"<li><b>{u['ticker']}</b> · {u['n_strategies']}전략 합의</li>"
            html += '</ul>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # Card 2: 주도섹터
    with cols[1]:
        html = f'<div class="pd-card"><h4 style="color:{C["cyan"]};margin:0 0 8px 0">🏢 주도 섹터</h4>'
        bs = stats.get("bullish_sec", [])[:3]
        if bs:
            html += '<p style="font-size:11px;color:' + C["gray"] + '">모멘텀 강세 Top 3</p>'
            html += '<ul style="margin:2px 0 6px 14px;font-size:11px">'
            for s in bs:
                html += (f"<li>{s['sec']} — Mom {s['mom_pct']:.1f}%<br>"
                         f"<span style='font-size:10px'>1M {fmt_pct(s['avg_1m'])} · n={s['total']}</span></li>")
            html += '</ul>'
        ty = stats.get("top_ytd_sec", [])[:2]
        if ty and ty[0]["sec"] not in {s["sec"] for s in bs}:
            html += '<p style="font-size:11px;color:' + C["green"] + '">YTD 리더</p>'
            html += '<ul style="margin:2px 0 6px 14px;font-size:11px">'
            for s in ty:
                html += f"<li>{s['sec']} · YTD {fmt_pct(s['avg_ytd'])}</li>"
            html += '</ul>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # Card 3: 시장 회전 (Cyclical/Defensive + Growth/Value)
    with cols[2]:
        html = f'<div class="pd-card"><h4 style="color:{C["orange"]};margin:0 0 8px 0">🔄 시장 회전</h4>'
        # Cyclical vs Defensive
        cyc_sectors = ["Technology", "Financials", "Industrials", "Materials", "Consumer Discretionary",
                       "Energy", "Real_Assets"]
        def_sectors = ["Consumer Staples", "Utilities", "Healthcare", "Communication"]
        cyc_comps = [r.get("composite", 0) for r in all_results
                     if any(c in (r.get("sector", "") or "") for c in cyc_sectors)
                     and r.get("composite") is not None]
        def_comps = [r.get("composite", 0) for r in all_results
                     if any(d in (r.get("sector", "") or "") for d in def_sectors)
                     and r.get("composite") is not None]
        if cyc_comps and def_comps:
            cyc_avg = sum(cyc_comps) / len(cyc_comps)
            def_avg = sum(def_comps) / len(def_comps)
            cyc_c = C["orange"] if cyc_avg > def_avg else C["gray"]
            def_c = C["cyan"] if def_avg > cyc_avg else C["gray"]
            html += '<p style="font-size:11px;color:' + C["gray"] + '">Cyclical vs Defensive</p>'
            html += (f'<div style="font-size:11px">'
                     f'<div>Cyc: {colored_span(f"{cyc_avg:.1f}", cyc_c, True)} (n={len(cyc_comps)})</div>'
                     f'<div>Def: {colored_span(f"{def_avg:.1f}", def_c, True)} (n={len(def_comps)})</div>'
                     f'</div>')
        # Growth vs Value
        cap_stats = stats.get("cap_stats", {})
        if cap_stats:
            html += '<p style="font-size:11px;color:' + C["gray"] + ';margin-top:6px">Cap-Tier Mom%</p>'
            html += '<div style="font-size:11px">'
            for tier in ["MEGA", "LARGE", "MID", "SMALL"]:
                if tier in cap_stats:
                    d = cap_stats[tier]
                    html += f"<div>{tier}: {d['mom_pct']:.1f}% (n={d['total']})</div>"
            html += '</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # Card 4: 주도 테마 & 전략
    with cols[3]:
        html = f'<div class="pd-card"><h4 style="color:{C["purple"]};margin:0 0 8px 0">🎯 테마 & 전략</h4>'
        bt = stats.get("bullish_themes", [])[:3]
        if bt:
            html += '<p style="font-size:11px;color:' + C["gray"] + '">강세 테마</p>'
            html += '<ul style="margin:2px 0 6px 14px;font-size:11px">'
            for t in bt:
                html += f"<li>{t['name']} — {t['mom_pct']:.1f}% (n={t['total']})</li>"
            html += '</ul>'
        if regime_stats and regime_stats.get("strategy_groups"):
            groups = regime_stats["strategy_groups"]
            lead = max(groups, key=lambda g: g.get("net_breadth", 0))
            html += '<p style="font-size:11px;color:' + C["gray"] + '">주도 전략 그룹</p>'
            html += (f'<div style="font-size:11px">'
                     f'<b>{lead.get("group", "?")}</b> · '
                     f'Net {colored_pct(lead.get("net_breadth", 0))}'
                     f'</div>')
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # Card 5 (full width): Regional Leadership table
    geo_stats = stats.get("geo_stats", {})
    if geo_stats:
        st.markdown(f'<h4 style="color:{C["cyan"]};margin:14px 0 6px 0">🌍 주도 지역 (Regional Leadership)</h4>',
                    unsafe_allow_html=True)
        regions = sorted(geo_stats.values(), key=lambda x: -x["mom_pct"])
        df = pd.DataFrame([{
            "Region": g["geo"],
            "Mom%": f"{g['mom_pct']:.1f}",
            "YTD%": f"{g['avg_ytd']:+.1f}",
            "Count": g["total"],
        } for g in regions])
        st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────
# Conviction Picks
# ─────────────────────────────────────────────────────────────────
def _parse_rejection(rejection: str | None) -> dict:
    if not rejection:
        return {"is_hard": False, "is_weak_qvr": False, "weak_qvr_score": None, "raw": ""}
    raw = rejection or ""
    is_hard = any(k in raw for k in [
        "Downtrend", "Exhausting", "Fading", "CounterRally", "CyclePeak", "Weakening",
        "LowScore", "Liq",
    ])
    is_weak_qvr = "WeakQVR" in raw
    return {"is_hard": is_hard, "is_weak_qvr": is_weak_qvr, "raw": raw}


def _is_etf(row: dict) -> bool:
    at = (row.get("asset_type") or "").lower()
    if "etf" in at: return True
    if "stock" in at: return False
    name = (row.get("name") or "").upper()
    return any(t in name for t in ["ETF", "FUND", "TRUST", "ISHARES", "SPDR", "VANGUARD"])


def _buy_score(row: dict, side: str = "buy") -> float:
    """BuyScore = Composite + bonuses - penalties."""
    score = row.get("composite") or 0
    cls = row.get("classification", "")
    if cls in MOMENTUM_SET: score += 10
    elif cls in EXCLUDED_SET: score -= 15
    elif cls in PM_SET: score += 5
    oer = row.get("oer") or 0
    if oer > 70: score -= (oer - 70) * 0.3
    ret_1m = row.get("ret_1m") or 0
    score += ret_1m * 0.5
    return score if side == "buy" else -score


def render_conviction_picks(all_results: list[dict], quant_stats: dict) -> None:
    if not all_results:
        return
    C = get_theme()
    st.markdown(f'<h3 style="color:{C["purple"]}">🎯 Conviction Picks</h3>', unsafe_allow_html=True)

    # Ultra-consensus ticker set (for bonus)
    ultra_tickers = {c["ticker"] for c in quant_stats.get("ultra_consensus", [])} if quant_stats else set()
    consensus_n_map = {c["ticker"]: c["n_strategies"] for c in quant_stats.get("consensus", [])} if quant_stats else {}

    # Filter pool
    buy_pool = []
    sell_pool = []
    for r in all_results:
        rej = _parse_rejection(r.get("rejection_reason"))
        cls = r.get("classification", "")
        if cls in EXCLUDED_SET or rej["is_hard"]:
            if cls in EXCLUDED_SET:
                sell_pool.append({**r, "_buy_score": _buy_score(r, "sell")})
        elif cls in MOMENTUM_SET or cls in PM_SET:
            r_copy = {**r}
            if r["ticker"] in ultra_tickers:
                r_copy["_consensus_bonus"] = 15
            elif r["ticker"] in consensus_n_map:
                r_copy["_consensus_bonus"] = 7
            else:
                r_copy["_consensus_bonus"] = 0
            r_copy["_buy_score"] = _buy_score(r) + r_copy["_consensus_bonus"]
            buy_pool.append(r_copy)

    buy_pool.sort(key=lambda x: -x["_buy_score"])
    sell_pool.sort(key=lambda x: -x["_buy_score"])

    buy_stocks = [r for r in buy_pool if not _is_etf(r)][:5]
    buy_etfs   = [r for r in buy_pool if _is_etf(r)][:5]
    sell_stocks = [r for r in sell_pool if not _is_etf(r)][:5]
    sell_etfs   = [r for r in sell_pool if _is_etf(r)][:5]

    def _table(picks: list[dict], title: str, color: str, side: str = "buy") -> None:
        if not picks:
            return
        st.markdown(f'<h4 style="color:{color};margin:12px 0 4px 0">{title}</h4>',
                    unsafe_allow_html=True)
        df = pd.DataFrame([{
            "Ticker": r.get("ticker", ""),
            "Name": (r.get("name") or "")[:30],
            "Sector": (r.get("sector") or "")[:18],
            "Class": r.get("classification", ""),
            "Comp": f"{r.get('composite', 0):.1f}",
            "1M%": fmt_pct(r.get("ret_1m"), 1),
            "YTD%": fmt_pct(r.get("ytd_return"), 1),
            "OER": f"{r.get('oer', 0):.0f}",
            "Score": f"{r.get('_buy_score', 0):.1f}",
            "Consensus": r.get("_consensus_bonus", 0) > 0,
        } for r in picks])
        st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        _table(buy_stocks, "🟢 Buy Stocks Top 5", C["green"])
        _table(buy_etfs, "🟢 Buy ETFs Top 5", C["green"])
    with col2:
        _table(sell_stocks, "🔴 Sell Stocks Top 5", C["red"], "sell")
        _table(sell_etfs, "🔴 Sell ETFs Top 5", C["red"], "sell")
