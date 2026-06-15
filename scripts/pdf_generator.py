# -*- coding: utf-8 -*-
"""pdf_generator.py — Daily Report PDF builder (한국어 지원).

Generates a 2-3 page PDF with:
  - Title + date
  - Executive Commentary (PM commentary + 3-Agent voting summary)
  - Buy Final List Top 20 (table with votes, risk, returns)
  - Sell Final List Top 20
  - Disclaimer
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether,
)


# ─── Korean font registration (macOS system font) ───────────────────
KOREAN_FONT_PATH = "/System/Library/Fonts/AppleSDGothicNeo.ttc"

def _register_korean_font() -> str:
    """Register Korean font for reportlab. Returns font name."""
    try:
        pdfmetrics.registerFont(TTFont("Korean", KOREAN_FONT_PATH))
        return "Korean"
    except Exception:
        # Fallback to CID font (works without TTF file)
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
        return "HYSMyeongJo-Medium"


KOREAN_FONT = _register_korean_font()


# ─── Styles ─────────────────────────────────────────────────────────
def _build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleK", parent=base["Title"],
            fontName=KOREAN_FONT, fontSize=18, leading=22,
            textColor=colors.HexColor("#16c784"),
            alignment=TA_CENTER, spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubK", parent=base["Normal"],
            fontName=KOREAN_FONT, fontSize=10, leading=14,
            textColor=colors.HexColor("#7884a0"),
            alignment=TA_CENTER, spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "SectionK", parent=base["Heading2"],
            fontName=KOREAN_FONT, fontSize=12, leading=16,
            textColor=colors.HexColor("#1a1a1a"),
            spaceAfter=6, spaceBefore=10,
        ),
        "body": ParagraphStyle(
            "BodyK", parent=base["Normal"],
            fontName=KOREAN_FONT, fontSize=9, leading=13,
            textColor=colors.HexColor("#333333"),
            alignment=TA_JUSTIFY, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "SmallK", parent=base["Normal"],
            fontName=KOREAN_FONT, fontSize=7, leading=10,
            textColor=colors.HexColor("#666666"),
        ),
        "disclaimer": ParagraphStyle(
            "Disc", parent=base["Normal"],
            fontName=KOREAN_FONT, fontSize=7, leading=10,
            textColor=colors.HexColor("#999999"),
            alignment=TA_JUSTIFY, spaceAfter=2,
        ),
    }
    return styles


# ─── Table builders ──────────────────────────────────────────────────
def _vote_emoji(vote: str) -> str:
    if vote == "APPROVE": return "✓"
    if vote == "REJECT":  return "✗"
    return "○"


def _votes_text(item: dict) -> str:
    v = item.get("votes") or {}
    return (
        f"PM:{_vote_emoji(v.get('pm','?'))} "
        f"Trd:{_vote_emoji(v.get('trading','?'))} "
        f"Risk:{_vote_emoji(v.get('risk','?'))}"
    )


def _fmt_ret(v) -> str:
    if v is None: return "-"
    return f"{v*100:+.1f}%"


def _build_picks_table(items: list, mode: str = "buy", top_n: int = 20) -> Table:
    """Build a Buy or Sell picks table for the PDF."""
    items = items[:top_n]
    if not items:
        # Empty placeholder
        return Table([["조건 만족 종목 없음"]], colWidths=[16*cm])

    headers = [
        "★",      # Stars
        "Ticker",
        "Name",
        "Type",
        "Hrzn",
        "Comp",
        "Votes (PM/Trd/Risk)",
        "State",
        "Action",
        "Risk Score",
        "5d",
        "1M",
        "3M",
        "6M",
    ]
    rows = [headers]

    HORIZON_LBL = {"tactical": "T-5d", "core": "C-21d", "strategic": "S-63d"}
    ACTION_LBL = {
        "EXECUTE_TODAY": "오늘실행",
        "WATCH_TOMORROW": "내일확인",
        "ALREADY_HELD": "보유중",
        "OBSERVE": "관찰",
        "CLOSE_TODAY": "오늘청산",
    }

    for r in items:
        bucket = r.get("bucket", "")
        asset_type = "Stock" if "stocks" in bucket else "ETF" if "etfs" in bucket else "?"
        rows.append([
            "★" * (r.get("stars", 0) or 0),
            r.get("ticker", ""),
            (r.get("name", "") or "")[:22],
            asset_type,
            HORIZON_LBL.get(r.get("horizon", ""), r.get("horizon", "")[:5]),
            f"{r.get('composite', 0):.0f}",
            _votes_text(r),
            r.get("state", "")[:8],
            ACTION_LBL.get(r.get("action", ""), r.get("action", "")[:8]),
            f"{r.get('risk_score', 0):.0f}",
            _fmt_ret(r.get("ret_5d")),
            _fmt_ret(r.get("ret_1mo")),
            _fmt_ret(r.get("ret_3mo")),
            _fmt_ret(r.get("ret_6mo")),
        ])

    col_widths = [
        0.7*cm,  # Stars
        1.6*cm,  # Ticker
        3.6*cm,  # Name
        0.9*cm,  # Type
        0.9*cm,  # Hrzn
        0.8*cm,  # Comp
        2.6*cm,  # Votes
        1.4*cm,  # State
        1.4*cm,  # Action
        1.0*cm,  # Risk
        1.0*cm, 1.0*cm, 1.0*cm, 1.0*cm,  # Returns
    ]

    header_color = colors.HexColor("#16c784") if mode == "buy" else colors.HexColor("#ea3943")
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), KOREAN_FONT),
        ("FONTSIZE", (0, 0), (-1, 0), 7),
        ("FONTSIZE", (0, 1), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (5, 1), (5, -1), "RIGHT"),     # Composite
        ("ALIGN", (9, 1), (-1, -1), "RIGHT"),    # Risk score + returns
        ("ALIGN", (3, 1), (4, -1), "CENTER"),    # Type, Horizon
        ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    # Color positive/negative returns
    for col_idx in (10, 11, 12, 13):  # ret_5d, ret_1mo, ret_3mo, ret_6mo
        for row_idx in range(1, len(rows)):
            cell_text = rows[row_idx][col_idx]
            if isinstance(cell_text, str):
                if cell_text.startswith("+"):
                    style.append(("TEXTCOLOR", (col_idx, row_idx), (col_idx, row_idx),
                                  colors.HexColor("#16c784")))
                elif cell_text.startswith("-") and cell_text != "-":
                    style.append(("TEXTCOLOR", (col_idx, row_idx), (col_idx, row_idx),
                                  colors.HexColor("#ea3943")))

    table.setStyle(TableStyle(style))
    return table


def _build_summary_stats(final_list_data: dict, mode: str) -> str:
    """Build summary statistics text for buy or sell list."""
    from collections import Counter
    items = final_list_data.get("buy_list" if mode == "buy" else "sell_list", [])
    if not items:
        return "후보 없음"
    n = len(items)
    star_dist = Counter(r.get("stars", 0) for r in items)
    sector_dist = Counter(r.get("sector", "?") for r in items)
    cons_dist = Counter(r.get("consensus", "?") for r in items)
    avg_risk = sum(r.get("risk_score", 0) for r in items) / max(1, n)
    top_sectors = ", ".join(f"{s}({c})" for s, c in sector_dist.most_common(3))
    return (
        f"총 {n}종목 · "
        f"★★★ {star_dist.get(3,0)}개 · ★★ {star_dist.get(2,0)}개 · ★ {star_dist.get(1,0)}개 · "
        f"UNANIMOUS {cons_dist.get('UNANIMOUS',0)}개 · "
        f"평균 Risk Score {avg_risk:.1f}/100 · "
        f"섹터 Top 3: {top_sectors}"
    )


def build_daily_report_pdf(final_list_data: dict, swarm_cache: dict | None = None,
                           output_path: str = "/tmp/price_discovery_daily.pdf") -> str:
    """Build the daily PDF report.

    Args:
        final_list_data: result from build_final_lists()
        swarm_cache: optional swarm cache for PM commentary
        output_path: where to save PDF

    Returns: output_path
    """
    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=landscape(A4),
        leftMargin=1.0*cm, rightMargin=1.0*cm,
        topMargin=1.0*cm, bottomMargin=1.0*cm,
        title="Price Discovery Daily Report",
        author="Price Discovery System",
    )
    story = []

    # ── Title ──
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M KST")
    story.append(Paragraph(f"🟢 Price Discovery Daily Report", styles["title"]))
    story.append(Paragraph(f"Generated: {now_str}", styles["subtitle"]))

    # ── Buy Summary ──
    story.append(Paragraph("📈 매수 Final List Summary", styles["section"]))
    story.append(Paragraph(_build_summary_stats(final_list_data, "buy"), styles["body"]))
    story.append(Spacer(1, 4))

    # ── Buy Commentary (truncated) ──
    commentary = final_list_data.get("commentary", {}) or {}
    buy_comm = (commentary.get("buy_commentary") or "")[:1100]
    if buy_comm:
        story.append(Paragraph("📋 매수 Executive Commentary", styles["section"]))
        story.append(Paragraph(buy_comm.replace("\n", "<br/>"), styles["body"]))
        story.append(Spacer(1, 6))

    # ── Buy Table ──
    story.append(Paragraph("📊 매수 Final List — Top 20", styles["section"]))
    buy_table = _build_picks_table(final_list_data.get("buy_list", []), mode="buy", top_n=20)
    story.append(buy_table)

    story.append(PageBreak())

    # ── Sell Summary ──
    story.append(Paragraph("📉 매도 Final List Summary", styles["section"]))
    story.append(Paragraph(_build_summary_stats(final_list_data, "sell"), styles["body"]))
    story.append(Spacer(1, 4))

    sell_comm = (commentary.get("sell_commentary") or "")[:1100]
    if sell_comm:
        story.append(Paragraph("📋 매도 Executive Commentary", styles["section"]))
        story.append(Paragraph(sell_comm.replace("\n", "<br/>"), styles["body"]))
        story.append(Spacer(1, 6))

    story.append(Paragraph("📊 매도 Final List — Top 20", styles["section"]))
    sell_table = _build_picks_table(final_list_data.get("sell_list", []), mode="sell", top_n=20)
    story.append(sell_table)

    # ── PM Commentary (if available) ──
    if swarm_cache:
        pm = (swarm_cache.get("phase5_pm") or {}).get("pm_commentary", "")
        if pm:
            story.append(PageBreak())
            story.append(Paragraph("🤖 PM Agent Commentary (Phase 5)", styles["section"]))
            story.append(Paragraph(pm.replace("\n", "<br/>"), styles["body"]))

    # ── Disclaimer ──
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "⚠ 본 리포트는 Price Discovery 자동 분석 시스템의 출력물이며, 투자 권유 또는 자문이 아닙니다. "
        "모든 투자 결정은 본인의 책임 하에 이루어지며, 본 리포트의 정보를 근거로 한 거래로 인한 손익에 대해 "
        "시스템 제공자는 책임을 지지 않습니다. 매수/매도 추천은 알고리즘 분석 결과로 시장 상황 변화에 따라 "
        "달라질 수 있습니다.",
        styles["disclaimer"],
    ))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    # Quick test: build PDF from current /api/final-list
    import urllib.request
    print("Fetching final list data...")
    d = json.loads(urllib.request.urlopen("http://localhost:8000/api/final-list", timeout=120).read())
    sw = {}
    try:
        sw = json.load(open(".market_leaders_swarm_cache.json"))
    except Exception:
        pass
    out = build_daily_report_pdf(d, swarm_cache=sw, output_path="/tmp/price_discovery_daily_test.pdf")
    print(f"✓ PDF generated: {out}")
    import subprocess
    subprocess.run(["open", out])
