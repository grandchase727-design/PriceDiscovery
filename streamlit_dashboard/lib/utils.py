"""Shared utility functions matching MarketCommentaryTab.tsx helpers."""
from __future__ import annotations

from typing import Any

from .theme import get_theme


# Classification sets (mirrors React MOMENTUM_SET / PM_SET)
MOMENTUM_SET = {"🟢 CONTINUATION", "🟡 OVEREXTENDED", "🔵 FORMATION", "🟦 LAGGING_CATCHUP"}
PM_SET = {"🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔵 RECOVERY", "🔶 PULLBACK", "⚠️ WEAKENING", "🟤 FADING"}
EXCLUDED_SET = {"🔴 CYCLE_PEAK", "🟤 EXHAUSTING", "⬇️ DOWNTREND", "🟣 COUNTER_RALLY"}


def format_name(row: dict) -> str:
    """Match React formatName(): 'Name (TICKER)' truncated to 32 chars."""
    name = (row.get("name") or "").strip()
    ticker = row.get("ticker", "")
    if not name:
        return ticker
    full = f"{name} ({ticker})"
    if len(full) > 32:
        return f"{name[:28]}… ({ticker})"
    return full


def fmt_list(rows: list[dict], with_return: bool = False, limit: int = 5) -> str:
    """Match React fmtList(): semicolon-joined with optional metrics."""
    out = []
    for r in rows[:limit]:
        s = format_name(r)
        if with_return:
            ytd = r.get("ytd_return")
            mom = r.get("ret_1m")
            extras = []
            if ytd is not None: extras.append(f"YTD {ytd:+.1f}%")
            if mom is not None: extras.append(f"1M {mom:+.1f}%")
            if extras: s += f" ({', '.join(extras)})"
        out.append(s)
    return "; ".join(out)


def verdict_color(score: float | None) -> str:
    """Hit-rate color: ≥75 green, ≥50 yellow, <50 red."""
    C = get_theme()
    if score is None:
        return C["gray"]
    if score >= 75:
        return C["green"]
    if score >= 50:
        return C["yellow"]
    return C["red"]


def comp_color(composite: float | None) -> str:
    """Composite gradient (matches MultiAgentConvictionDebateCard.compColor)."""
    C = get_theme()
    if composite is None:
        return C["gray"]
    if composite >= 75:
        return C["green"]
    if composite >= 60:
        return C["cyan"]
    if composite >= 40:
        return C["gray"]
    return C["red"]


def regime_color(regime: str | None) -> str:
    """Color for regime tag."""
    C = get_theme()
    if not regime:
        return C["gray"]
    r = regime.upper()
    if "RISK-ON" in r or "PRO-GROWTH" in r or "건전" in r:
        return C["green"]
    if "RISK-OFF" in r or "약세" in r:
        return C["red"]
    if "ROTATION" in r:
        return C["amber"]
    if "COMPRESSION" in r or "혼조" in r:
        return C["yellow"]
    if "TRANSITION" in r or "전환" in r:
        return C["blue"]
    if "과열" in r:
        return C["amber"]
    if "회복" in r:
        return C["cyan"]
    return C["gray"]


def fmt_pct(v: float | None, decimals: int = 1, signed: bool = True) -> str:
    if v is None:
        return "—"
    sign = "+" if (v >= 0 and signed) else ""
    return f"{sign}{v:.{decimals}f}%"


def fmt_int(v: float | None) -> str:
    return "—" if v is None else f"{int(v)}"


def colored_span(text: str, color: str, bold: bool = False, size: int = 12) -> str:
    """Inline HTML span with color."""
    w = "bold" if bold else "normal"
    return f'<span style="color:{color};font-weight:{w};font-size:{size}px">{text}</span>'


def badge(text: str, color: str, bg_alpha: str = "25", border_alpha: str = "80") -> str:
    """Pill-style badge."""
    return (
        f'<span class="pd-pill" style="'
        f'color:{color};background-color:{color}{bg_alpha};'
        f'border:1px solid {color}{border_alpha}">{text}</span>'
    )


def pct_color(v: float | None) -> str:
    """Returns color: green for positive, red for negative, gray for zero/None."""
    C = get_theme()
    if v is None or v == 0:
        return C["gray"]
    return C["green"] if v > 0 else C["red"]


def colored_pct(v: float | None, decimals: int = 1) -> str:
    """Format pct with color span."""
    if v is None:
        return colored_span("—", get_theme()["gray"])
    return colored_span(fmt_pct(v, decimals), pct_color(v), bold=True)


def safe_get(d: dict, *keys, default=None):
    """Nested get."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def section_wrap_html(num: int, title: str, accent: str, body_html: str) -> str:
    """Match React SectionWrap component."""
    return (
        f'<div class="pd-section-wrap" style="--accent:{accent};border-left-color:{accent}">'
        f'<div class="pd-section-title" style="color:{accent}">§{num}. {title}</div>'
        f'<div class="pd-section-body">{body_html}</div>'
        f'</div>'
    )
