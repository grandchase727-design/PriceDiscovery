"""
Generate Bibliography PDF directly using reportlab.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _make_bibliography import REFERENCES, UNDOWNLOADED_REFERENCES

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
)
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Try to register Korean-capable font
KOREAN_FONT = "Helvetica"
for candidate in [
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/AppleGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]:
    if os.path.exists(candidate):
        try:
            pdfmetrics.registerFont(TTFont("Korean", candidate))
            KOREAN_FONT = "Korean"
            break
        except Exception:
            pass


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PDF = os.path.join(SCRIPT_DIR, "00_Bibliography_PriceDiscovery_References.pdf")


def build():
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=LETTER,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Title1", parent=styles["Title"],
        fontName=KOREAN_FONT, fontSize=20, textColor=HexColor("#1F4E79"),
        spaceAfter=10))
    styles.add(ParagraphStyle("H1Custom", parent=styles["Heading1"],
        fontName=KOREAN_FONT, fontSize=15, textColor=HexColor("#1F4E79"),
        spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle("H2Custom", parent=styles["Heading2"],
        fontName=KOREAN_FONT, fontSize=12, textColor=HexColor("#2E75B6"),
        spaceBefore=10, spaceAfter=4))
    styles.add(ParagraphStyle("Normal_K", parent=styles["Normal"],
        fontName=KOREAN_FONT, fontSize=9.5, leading=13, alignment=TA_LEFT))
    styles.add(ParagraphStyle("Bullet_K", parent=styles["Normal"],
        fontName=KOREAN_FONT, fontSize=9, leading=12, leftIndent=18, alignment=TA_LEFT))
    styles.add(ParagraphStyle("RefTitle", parent=styles["Normal"],
        fontName=KOREAN_FONT, fontSize=10.5, leading=14,
        textColor=HexColor("#1F4E79"), spaceBefore=6, spaceAfter=2))
    styles.add(ParagraphStyle("RefMeta", parent=styles["Normal"],
        fontName=KOREAN_FONT, fontSize=9, leading=11.5,
        textColor=HexColor("#444444"), leftIndent=12))
    styles.add(ParagraphStyle("RefFile", parent=styles["Normal"],
        fontName="Courier", fontSize=8.5, leading=10.5,
        textColor=HexColor("#06B6D4"), leftIndent=12))
    styles.add(ParagraphStyle("Note_K", parent=styles["Normal"],
        fontName=KOREAN_FONT, fontSize=8.5, leading=11,
        textColor=HexColor("#996B00"), leftIndent=12))

    story = []

    # Title
    story.append(Paragraph("Price Discovery System — Academic References", styles["Title1"]))
    story.append(Paragraph(
        "<b>작성일</b>: 2026-04-30 &nbsp;&nbsp;|&nbsp;&nbsp; <b>수록 논문</b>: 20개 (다운로드) + 6개 (citation only)",
        styles["Normal_K"]))
    story.append(Spacer(1, 12))

    # Overview
    story.append(Paragraph("Overview", styles["H1Custom"]))
    story.append(Paragraph(
        "Price Discovery 시스템에서 사용된 모든 정량 로직은 학술적으로 검증된 연구에 기반합니다. "
        "이 문서는 시스템 컴포넌트별로 그 학술적 근거를 매핑한 reference 모음입니다.",
        styles["Normal_K"]))
    story.append(Spacer(1, 8))

    # Component mapping table
    story.append(Paragraph("Component → Reference Mapping", styles["H1Custom"]))
    mapping = [
        ("Composite Score (TCS/TFS/RSS)", "Jegadeesh-Titman 1993, Moskowitz 2012, Brock 1992"),
        ("OER (Overextension Risk)", "Daniel-Moskowitz 2016, George-Hwang 2004, DHS 1998"),
        ("Pre-Momentum 4-Agent", "Hong-Stein 1999, BSV 1998, DHS 1998, AFP 2014"),
        ("Hedge Strategies (8개)", "Lo-Mamaysky-Wang 2000, Brock-Lakonishok 1992"),
        ("Classification (3×3 + overrides)", "Carhart 1997, Daniel-Moskowitz 2016, Asness 2013"),
        ("Phase 1: Vol-Adjusted Buffer", "Andersen-Bollerslev 1998, Frazzini-Pedersen 2014"),
        ("Validation (Forward / IC / Persistence)", "Grinold 1989, Sharpe 1994, Fama-MacBeth 1973"),
        ("LAGGING_CATCHUP / Underreaction", "Hong-Stein 1999, BSV 1998"),
        ("Long-term Reversal (CYCLE_PEAK)", "De Bondt-Thaler 1985, Jegadeesh-Titman 2001"),
        ("Multi-horizon Returns", "Asness 2013, Moskowitz 2012, Carhart 1997"),
    ]
    data = [["System Component", "Academic References"]] + mapping
    t = Table(data, colWidths=[2.5 * inch, 4.0 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, -1), KOREAN_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), KOREAN_FONT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#F5F8FB"), white]),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#CCCCCC")),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Detailed by category
    story.append(Paragraph("References by Category", styles["H1Custom"]))

    category_order = [
        "Momentum (Foundational)", "Momentum (Multi-Asset)", "Momentum (Cross-Asset)",
        "Momentum (Robustness)", "Momentum (52-Week High)", "Momentum (Risk Management)",
        "Behavioral (Underreaction)", "Behavioral (Information Diffusion)",
        "Behavioral (Overconfidence)", "Behavioral (Overreaction)",
        "Factor Model (Foundational)", "Factor Model (Methodology)",
        "Factor Model (Momentum Factor)", "Factor Model (Quality)", "Factor Model (BAB)",
        "Technical Analysis (Foundational)", "Technical Analysis (MA Rules)",
        "Volatility (Foundational)", "Risk-Adjusted Performance", "Option Theory (Reference)",
    ]
    by_cat = {}
    for r in REFERENCES:
        by_cat.setdefault(r["category"], []).append(r)

    for cat in category_order:
        if cat not in by_cat:
            continue
        story.append(Paragraph(cat, styles["H2Custom"]))
        for r in by_cat[cat]:
            # Title
            story.append(Paragraph(f"<b>[{r['id']}]</b> {r['title']}", styles["RefTitle"]))
            # Authors / venue
            story.append(Paragraph(
                f"<i>{r['authors']} ({r['year']}).</i> {r['venue']}",
                styles["RefMeta"]))
            # Filename
            story.append(Paragraph(f"📄 {r['filename']}", styles["RefFile"]))
            # Applies to
            story.append(Paragraph("<b>Applies to:</b>", styles["RefMeta"]))
            for item in r["applies_to"]:
                story.append(Paragraph(f"• {item}", styles["Bullet_K"]))
            story.append(Spacer(1, 4))

    # Page break before citation-only
    story.append(PageBreak())

    # Citation-only
    story.append(Paragraph("Citation-Only References (다운로드 불가)", styles["H1Custom"]))
    story.append(Paragraph(
        "다음 문헌들은 paywall, 사이트 차단, 또는 서적이라 직접 다운로드되지 않았습니다. "
        "URL이 있으면 방문하여 접근 가능합니다.",
        styles["Normal_K"]))
    story.append(Spacer(1, 6))

    for ref in UNDOWNLOADED_REFERENCES:
        story.append(Paragraph(f"<b>• {ref['title']}</b>", styles["RefTitle"]))
        story.append(Paragraph(
            f"<i>{ref['authors']} ({ref['year']}).</i> {ref['venue']}.",
            styles["RefMeta"]))
        if ref.get("url") and ref["url"] != "—":
            story.append(Paragraph(
                f'URL: <font color="#06B6D4"><u>{ref["url"]}</u></font>',
                styles["RefMeta"]))
        story.append(Paragraph(f"<b>Applies:</b> {ref['applies_to']}", styles["RefMeta"]))
        if ref.get("note"):
            story.append(Paragraph(f"<i>Note:</i> {ref['note']}", styles["Note_K"]))
        story.append(Spacer(1, 6))

    # Folder structure
    story.append(Spacer(1, 14))
    story.append(Paragraph("Folder Structure", styles["H1Custom"]))
    story.append(Paragraph(
        "모든 PDF는 <font face='Courier'>docs/references/</font> 폴더에 저장되어 있습니다. "
        "파일명은 <font face='Courier'>[ID]_[Authors]_[Year]_[ShortTitle].pdf</font> 형식입니다.",
        styles["Normal_K"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"총 다운로드: <b>20개 PDF</b> (약 38MB)", styles["Normal_K"]))

    doc.build(story)
    print(f"Saved: {OUTPUT_PDF}")


if __name__ == "__main__":
    build()
