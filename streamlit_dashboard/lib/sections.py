"""All 23 sections of the Market Commentary tab — Streamlit ports."""
from __future__ import annotations

import streamlit as st

from .theme import get_theme
from .utils import (
    badge, colored_pct, colored_span, fmt_list, fmt_pct, format_name,
    pct_color, regime_color, section_wrap_html, verdict_color,
)


def _section_html(num: int, title: str, accent_key: str, body_html: str) -> None:
    C = get_theme()
    accent = C.get(accent_key, C["cyan"])
    st.markdown(section_wrap_html(num, title, accent, body_html), unsafe_allow_html=True)


def _p(text: str) -> str:
    return f"<p style='margin:4px 0'>{text}</p>"


# ─────────────────────────────────────────────────────────────────
# § 1. 거시 시장 분포 & 광범위성 (Breadth)
# ─────────────────────────────────────────────────────────────────
def section_1(stats: dict) -> None:
    if not stats:
        return
    C = get_theme()
    total = stats["total"]
    dist = stats["distribution"]
    top_dist = sorted(dist.items(), key=lambda x: -x[1])[:4]
    top_text = " · ".join(
        f"{cls} {c}개 ({c/total*100:.1f}%)" for cls, c in top_dist
    )

    body = []
    body.append(_p(
        f"전체 <b>{total}종목</b> 중 분류 분포: {top_text}."
    ))
    mom_text = colored_span(f"{stats['momentum_n']}종목 ({stats['momentum_pct']:.1f}%)", C["green"], True)
    pm_text = colored_span(f"{stats['pm_n']}종목 ({stats['pm_pct']:.1f}%)", C["cyan"], True)
    exc_text = colored_span(f"{stats['excluded_n']}종목 ({stats['excluded_pct']:.1f}%)", C["red"], True)
    body.append(_p(
        f"Momentum 단계: {mom_text}"
        f" · Pre-Momentum 단계: {pm_text}"
        f" · 제외군: {exc_text}"
    ))
    body.append(_p(
        f"YTD 광범위성(긍정 비율): <b>{stats['breadth_pct']:.1f}%</b> · "
        f"평균 YTD {colored_pct(stats['avg_ytd'])} · "
        f"중앙값 {colored_pct(stats['ytd_median'])} · "
        f"표준편차 {stats['dispersion']:.1f}%p"
    ))
    body.append(_p(
        f"Eligibility (Composite ≥55): <b>{stats['eligible_n']}종목 ({stats['eligible_pct']:.1f}%)</b>"
    ))

    gainers = stats.get("gainers", [])[:3]
    losers  = stats.get("losers", [])[:3]
    if gainers or losers:
        g_text = " · ".join(f"{g['cls']} (+{g['delta']})" for g in gainers)
        l_text = " · ".join(f"{l['cls']} ({l['delta']})" for l in losers)
        body.append(_p(
            f"최근 변화 — 증가: <span style='color:{C['green']}'>{g_text or '—'}</span> · "
            f"감소: <span style='color:{C['red']}'>{l_text or '—'}</span>"
        ))

    _section_html(1, "거시 시장 분포 & 광범위성(Breadth) 분석", "cyan", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 2. Regime 진단 & 모멘텀 방향성
# ─────────────────────────────────────────────────────────────────
def section_2(stats: dict) -> None:
    if not stats:
        return
    C = get_theme()
    regime = stats["regime"]
    regime_c = regime_color(regime)
    bull = stats["bull_idx"]
    bear = stats["bear_idx"]

    body = []
    body.append(_p(
        f"현재 진단: {badge(regime, regime_c)} — "
        f"Bull 지표(CONT+FORM+RECV+LAG)={bull}종목 vs Bear 지표(DOWN+WEAK+FADE+PEAK+EXHAUST)={bear}종목 "
        f"(차이 <b>{bull - bear:+d}</b>)"
    ))
    body.append(_p(
        f"강세군 세부: "
        f"🟢 CONTINUATION {stats['cont']} · 🔵 FORMATION {stats['form']} · "
        f"🔵 RECOVERY {stats['recv']} · 🟦 LAGGING_CATCHUP {stats['lag']}"
    ))
    body.append(_p(
        f"약세군 세부: "
        f"⬇️ DOWNTREND {stats['down']} · ⚠️ WEAKENING {stats['weak']} · "
        f"🟤 FADING {stats['fade']} · 🔴 CYCLE_PEAK {stats['peak']} · "
        f"🟤 EXHAUSTING {stats['exhaust']}"
    ))
    pm_zone = stats['cons'] + stats['neut'] + stats['pull']
    body.append(_p(
        f"Pre-Momentum 압력 영역(CONS+NEUT+PULL): <b>{pm_zone}종목</b> — "
        f"방향성 미확정 비중이 이만큼 누적"
    ))
    body.append(_p(
        f"과열 경고: 🟡 OVEREXTENDED {stats['oext']} · 🔴 CYCLE_PEAK {stats['peak']} · 🟤 EXHAUSTING {stats['exhaust']} — "
        f"합계 <b>{stats['oext'] + stats['peak'] + stats['exhaust']}종목</b>"
    ))

    _section_html(2, "Regime 진단 & 모멘텀 방향성 해석", "amber" if "과열" in regime else "green", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 3. Cap-Tier & Geographic Leadership
# ─────────────────────────────────────────────────────────────────
def section_3(stats: dict) -> None:
    if not stats:
        return
    C = get_theme()
    body = []
    cap_stats = stats.get("cap_stats", {})
    if cap_stats:
        tiers = ["MEGA", "LARGE", "MID", "SMALL"]
        rows_html = []
        for t in tiers:
            if t not in cap_stats: continue
            d = cap_stats[t]
            rows_html.append(
                f"<li><b>{t}</b>: Mom {d['mom_pct']:.1f}% · "
                f"YTD {colored_pct(d['avg_ytd'])} · n={d['total']}</li>"
            )
        body.append(f"<p>Cap-Tier 분포:</p><ul style='margin:2px 0 6px 16px'>{''.join(rows_html)}</ul>")

        # Interpretation
        mega = cap_stats.get("MEGA", {}).get("mom_pct", 0)
        small = cap_stats.get("SMALL", {}).get("mom_pct", 0)
        if mega - small > 10:
            interp = "Mega/Large 집중 — 시장 폭이 좁은 양극화 양상"
            ic = C["amber"]
        elif small - mega > 10:
            interp = "Small/Mid 광범위 참여 — 건전한 광폭 강세"
            ic = C["green"]
        else:
            interp = "Cap-Tier 균형 — 광폭 + 메가캡 모두 동조"
            ic = C["cyan"]
        body.append(_p(colored_span(f"→ {interp}", ic, True)))

    geo_stats = stats.get("geo_stats", {})
    if geo_stats:
        body.append("<p style='margin-top:8px'>지역 분포:</p>")
        rows_html = []
        for geo, d in sorted(geo_stats.items(), key=lambda x: -x[1]["mom_pct"]):
            rows_html.append(
                f"<li><b>{geo}</b>: Mom {d['mom_pct']:.1f}% · PM {d['pm_pct']:.1f}% · "
                f"YTD {colored_pct(d['avg_ytd'])} · n={d['total']}</li>"
            )
        body.append(f"<ul style='margin:2px 0 6px 16px'>{''.join(rows_html)}</ul>")

    _section_html(3, "Cap-Tier & 지역(Geographic) Leadership 분해", "cyan", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 4. Sector Leadership — 강세군
# ─────────────────────────────────────────────────────────────────
def section_4(stats: dict) -> None:
    bullish = stats.get("bullish_sec", [])
    top_ytd = stats.get("top_ytd_sec", [])
    top_3m = stats.get("top_3m_sec", [])
    comp_sec = stats.get("comp_sec", [])
    if not bullish:
        return
    C = get_theme()
    body = []
    body.append("<p><b>Momentum 비율 강세 섹터 Top 5</b>:</p>")
    body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
        f"<li>{s['sec']} — {s['mom_pct']:.1f}% (n={s['total']}) · "
        f"1M {colored_pct(s['avg_1m'])} · 3M {colored_pct(s['avg_3m'])} · "
        f"YTD {colored_pct(s['avg_ytd'])} · Comp {s['avg_comp']:.1f}</li>"
        for s in bullish
    ) + "</ul>")
    if top_ytd:
        body.append("<p><b>YTD 누적 수익률 Top 5</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{s['sec']} — YTD {colored_pct(s['avg_ytd'])} · n={s['total']}</li>"
            for s in top_ytd
        ) + "</ul>")
    if top_3m:
        body.append("<p><b>3개월 모멘텀 Top 5</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{s['sec']} — 3M {colored_pct(s['avg_3m'])} · n={s['total']}</li>"
            for s in top_3m
        ) + "</ul>")
    if comp_sec:
        body.append("<p><b>Composite 평균 Top 5</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{s['sec']} — Comp <b>{s['avg_comp']:.1f}</b> · n={s['total']}</li>"
            for s in comp_sec
        ) + "</ul>")
    _section_html(4, "Sector Leadership — 강세군 deep dive", "green", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 5. Sector 약세군 & Rotation 신호
# ─────────────────────────────────────────────────────────────────
def section_5(stats: dict) -> None:
    bearish = stats.get("bearish_sec", [])
    worst = stats.get("worst_ytd_sec", [])
    C = get_theme()
    body = []
    if bearish:
        body.append("<p><b>제외 비율 30%+ 약세 섹터</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{s['sec']} — Excluded {s['exc_pct']:.1f}% · YTD {colored_pct(s['avg_ytd'])} · n={s['total']}</li>"
            for s in bearish[:5]
        ) + "</ul>")
    else:
        body.append("<p>현재 명확한 약세 섹터(제외 비율 ≥30%) 없음.</p>")

    if worst:
        body.append("<p><b>YTD 마이너스 누적 섹터</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{s['sec']} — YTD {colored_pct(s['avg_ytd'])} · n={s['total']}</li>"
            for s in worst[:5]
        ) + "</ul>")

    # Rotation signal: top 1M vs YTD
    sector_rows = stats.get("sector_rows", [])
    if sector_rows:
        top_1m = sorted(sector_rows, key=lambda x: -x["avg_1m"])[:3]
        top_ytd3 = sorted(sector_rows, key=lambda x: -x["avg_ytd"])[:3]
        set_1m = {s["sec"] for s in top_1m}
        set_ytd = {s["sec"] for s in top_ytd3}
        newcomers = set_1m - set_ytd
        fallouts = set_ytd - set_1m
        if newcomers or fallouts:
            body.append(_p(
                f"<b>Rotation 신호</b>: 1M 새 진입 섹터 → "
                f"<span style='color:{C['green']}'>{', '.join(newcomers) or '—'}</span> · "
                f"YTD 이탈 섹터 → "
                f"<span style='color:{C['red']}'>{', '.join(fallouts) or '—'}</span>"
            ))

    _section_html(5, "Sector 약세군 & Rotation 신호", "red", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 6. Industry & Theme Granular
# ─────────────────────────────────────────────────────────────────
def section_6(stats: dict) -> None:
    body = []
    bi = stats.get("bullish_ind", [])
    if bi:
        body.append("<p><b>Industry 모멘텀 강세 Top 6</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{i['name']} — Mom {i['mom_pct']:.1f}% · "
            f"YTD {colored_pct(i['avg_ytd'])} · n={i['total']}</li>"
            for i in bi
        ) + "</ul>")
    bi2 = stats.get("bearish_ind", [])
    if bi2:
        body.append("<p><b>Industry 모멘텀 약세 Bottom 4</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{i['name']} — Mom {i['mom_pct']:.1f}% · "
            f"YTD {colored_pct(i['avg_ytd'])} · n={i['total']}</li>"
            for i in bi2
        ) + "</ul>")
    bt = stats.get("bullish_themes", [])
    if bt:
        body.append("<p><b>Theme 강세 (Mom ≥50%, n≥3)</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{t['name']} — Mom {t['mom_pct']:.1f}% (n={t['total']}) · "
            f"YTD {colored_pct(t['avg_ytd'])}</li>"
            for t in bt
        ) + "</ul>")
    yt = stats.get("ytd_themes", [])
    if yt:
        body.append("<p><b>YTD 누적 Theme Top 5</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{t['name']} — YTD {colored_pct(t['avg_ytd'])} · 1M {colored_pct(t['avg_1m'])}</li>"
            for t in yt
        ) + "</ul>")
    wt = stats.get("weak_themes", [])
    if wt:
        body.append("<p><b>약세 Theme (Mom &lt;20% + YTD 마이너스)</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{t['name']} — Mom {t['mom_pct']:.1f}% · YTD {colored_pct(t['avg_ytd'])}</li>"
            for t in wt
        ) + "</ul>")
    _section_html(6, "Industry & Theme 모멘텀 — Granular 분석", "cyan", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 7. CONTINUATION & 신규 형성 — Quality Leadership
# ─────────────────────────────────────────────────────────────────
def section_7(stats: dict) -> None:
    C = get_theme()
    body = []

    def _list_block(label: str, items: list[dict], with_ret: bool = True) -> str:
        if not items:
            return _p(f"<b>{label}</b>: (없음)")
        rows = "".join(
            f"<li>{format_name(r)} · Comp <b>{r.get('composite', 0):.1f}</b>"
            + (f" · 1M {colored_pct(r.get('ret_1m'))}" if with_ret else "")
            + (f" · YTD {colored_pct(r.get('ytd_return'))}" if with_ret else "")
            + "</li>"
            for r in items
        )
        return f"<p><b>{label}</b>:</p><ul style='margin:2px 0 8px 16px'>{rows}</ul>"

    body.append(_list_block("🟢 CONTINUATION Top 5 (확립된 강세)", stats.get("top_cont", [])))
    body.append(_list_block("🔵 FORMATION Top 4 (초기 브레이크아웃)", stats.get("top_form", [])))
    body.append(_list_block("🔵 RECOVERY Top 4 (회복/반등)", stats.get("top_recv", [])))
    body.append(_list_block("🟦 LAGGING_CATCHUP Top 4 (광범위 추격)", stats.get("top_lag", [])))

    # Quality assessment
    cont = stats.get("cont", 0); form = stats.get("form", 0)
    if cont + form > 0:
        new_ratio = form / max(1, cont + form)
        if new_ratio > 0.4:
            quality = "신규 형성 비중 높음 → 사이클 초기"
            qc = C["green"]
        elif new_ratio > 0.2:
            quality = "신규+확립 혼재 → 사이클 중반"
            qc = C["cyan"]
        else:
            quality = "확립된 추세 위주 → 사이클 후반"
            qc = C["amber"]
        body.append(_p(colored_span(f"→ {quality}", qc, True)))

    _section_html(7, "CONTINUATION & 신규 형성 — Quality Leadership 분석", "green", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 8. OVEREXTENDED & CYCLE_PEAK 경고
# ─────────────────────────────────────────────────────────────────
def section_8(stats: dict) -> None:
    C = get_theme()
    body = []
    oext = stats.get("oext", 0)
    peak = stats.get("peak", 0)
    body.append(_p(
        f"과열 신호 총합: <b>{oext + peak}종목</b> "
        f"(🟡 OVEREXTENDED {oext} + 🔴 CYCLE_PEAK {peak})"
    ))
    top_oext = stats.get("top_oext", [])
    if top_oext:
        body.append("<p><b>🟡 OVEREXTENDED Top 5</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{format_name(r)} · Comp {r.get('composite', 0):.1f}</li>"
            for r in top_oext
        ) + "</ul>")
    top_peak = stats.get("top_peak", [])
    if top_peak:
        body.append(f"<p style='color:{C['red']}'><b>🔴 CYCLE_PEAK Top 4 (즉시 검토 권고)</b>:</p>")
        body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
            f"<li>{format_name(r)} · Comp {r.get('composite', 0):.1f}</li>"
            for r in top_peak
        ) + "</ul>")
    total_heat = oext + peak
    if total_heat > 20:
        msg = "광범위 과열 — 익절 + 부분 현금화 검토"
        mc = C["red"]
    elif total_heat > 10:
        msg = "선별적 과열 — 종목별 OER 검토 필수"
        mc = C["amber"]
    else:
        msg = "과열 위험 낮음 — 기존 포지션 유지 가능"
        mc = C["green"]
    body.append(_p(colored_span(f"→ {msg}", mc, True)))
    _section_html(8, "OVEREXTENDED & CYCLE_PEAK — 과열·정점 경고", "yellow", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 9. DOWNTREND·WEAKENING·FADING — 회피군
# ─────────────────────────────────────────────────────────────────
def section_9(stats: dict) -> None:
    C = get_theme()
    body = []
    down = stats.get("down", 0); weak = stats.get("weak", 0); fade = stats.get("fade", 0)
    body.append(_p(
        f"회피·약세 신호 총합: <b>{down + weak + fade}종목</b> "
        f"(⬇️ DOWNTREND {down} + ⚠️ WEAKENING {weak} + 🟤 FADING {fade})"
    ))
    for cls_label, items_key, emoji in [
        ("⬇️ DOWNTREND Top 4", "top_down", "⬇️"),
        ("⚠️ WEAKENING Top 4", "top_weak", "⚠️"),
        ("🟤 FADING Top 4", "top_fade", "🟤"),
    ]:
        items = stats.get(items_key, [])
        if items:
            body.append(f"<p><b>{cls_label}</b>:</p>")
            body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
                f"<li>{format_name(r)} · Comp {r.get('composite', 0):.1f}</li>"
                for r in items
            ) + "</ul>")

    total_def = down + weak + fade
    if total_def > 50:
        msg = "광범위 약세 스트레스 — 헷지 + 현금 비중 확대 필요"
        mc = C["red"]
    elif total_def > 25:
        msg = "선택적 방어 우선 — 약세 종목 청산 + 헷지 부분 도입"
        mc = C["amber"]
    else:
        msg = "약세 압력 제한적 — 정상 포지션 유지 가능"
        mc = C["green"]
    body.append(_p(colored_span(f"→ {msg}", mc, True)))
    _section_html(9, "DOWNTREND·WEAKENING·FADING — 회피군 & 헷지 우선순위", "red", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 10. 신규 기회 — Pre-Momentum Watchlist
# ─────────────────────────────────────────────────────────────────
def section_10(stats: dict) -> None:
    C = get_theme()
    body = []
    cons = stats.get("cons", 0); pull = stats.get("pull", 0)
    recv = stats.get("recv", 0); form = stats.get("form", 0); lag = stats.get("lag", 0)
    total_watch = cons + pull + recv + form + lag
    body.append(_p(
        f"Pre-Momentum/Watchlist 총합: <b>{total_watch}종목</b> "
        f"(CONS {cons} + PULL {pull} + RECV {recv} + FORM {form} + LAG {lag})"
    ))
    for label, items_key, strategy in [
        ("🔵 FORMATION 후보 (3-5일 단기)", "top_form", "tight stop, 빠른 진입"),
        ("🔵 RECOVERY 후보 (장기 평균분할)", "top_recv", "분할 진입, 평균단가 축적"),
        ("🟦 LAGGING_CATCHUP 후보 (광폭 바스켓)", "top_lag", "바스켓 매수, 시장 따라가기"),
    ]:
        items = stats.get(items_key, [])
        if items:
            body.append(f"<p><b>{label}</b> — 전략: <i>{strategy}</i></p>")
            body.append("<ul style='margin:2px 0 8px 16px'>" + "".join(
                f"<li>{format_name(r)} · Comp {r.get('composite', 0):.1f} · "
                f"YTD {colored_pct(r.get('ytd_return'))}</li>"
                for r in items
            ) + "</ul>")
    _section_html(10, "신규 기회 — Pre-Momentum 압력 종목 & Watchlist", "blue", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 11. Forward Outlook & Action Plan
# ─────────────────────────────────────────────────────────────────
def section_11(stats: dict) -> None:
    C = get_theme()
    body = []
    regime = stats.get("regime", "—")
    breadth = stats.get("breadth_pct", 0)
    disp = stats.get("dispersion", 0)
    oext = stats.get("oext", 0); peak = stats.get("peak", 0); exhaust = stats.get("exhaust", 0)
    cont = stats.get("cont", 0); form = stats.get("form", 0)
    down = stats.get("down", 0); weak = stats.get("weak", 0)

    # Base case scenario
    if "과열" in regime:
        base = "Base Case (1-3주): 부분 익절 + RECOVERY 신규 진입 mix"
    elif "건전" in regime:
        base = "Base Case (1-3주): CONTINUATION 보유 + FORMATION 추가 매수"
    elif "약세" in regime:
        base = "Base Case (1-3주): 방어 + 헷지 + 회피군 청산"
    elif "회복" in regime:
        base = "Base Case (1-3주): RECOVERY/FORMATION 단계적 진입"
    else:
        base = "Base Case (1-3주): 선택적 진입 + 보수적 포지션 관리"
    body.append(_p(colored_span(f"📊 {base}", C["purple"], True)))

    # Upside / Downside
    upside = []
    if breadth > 55: upside.append("YTD breadth >55% → 광폭 강세 잠재")
    if cont + form > down + weak: upside.append("강세 신호 우위")
    if disp > 25: upside.append(f"dispersion {disp:.1f}%p → 알파 기회 확대")
    body.append(_p(
        f"📈 <b>상방 시나리오</b>: " + ("; ".join(upside) if upside else "추가 상승 트리거 제한적")
    ))

    downside = []
    if oext + peak + exhaust > 20: downside.append(f"과열 신호 {oext+peak+exhaust}종목")
    if down + weak > 30: downside.append(f"하방 신호 {down+weak}종목")
    if breadth < 45: downside.append(f"YTD breadth {breadth:.1f}% → 광폭 약세")
    body.append(_p(
        f"📉 <b>하방 시나리오</b>: " + ("; ".join(downside) if downside else "시스템적 하방 위험 제한적")
    ))

    body.append("<p style='margin-top:10px'><b>즉시 액션 (오늘):</b></p>")
    body.append("<ol style='margin:2px 0 8px 18px'>"
        f"<li>🔴 CYCLE_PEAK {stats.get('peak', 0)}종목 → 즉시 익절 검토</li>"
        f"<li>🟡 OVEREXTENDED 상위 종목 OER >70 → 부분 익절</li>"
        f"<li>🟢 CONTINUATION 상위 5종목 → 보유 유지</li>"
        f"<li>🔵 FORMATION/RECOVERY 상위 → 신규 진입 검토</li>"
        f"<li>⬇️ DOWNTREND {stats.get('down', 0)}종목 → 청산 우선순위</li>"
        f"</ol>")

    body.append("<p><b>1-2주 액션 (단계적):</b></p>")
    body.append("<ol style='margin:2px 0 8px 18px'>"
        f"<li>🔵 RECOVERY 평균분할 진입 시작</li>"
        f"<li>섹터 로테이션 강세군 비중 확대</li>"
        f"<li>베타 헷지 비중 점검 (VIX/SQQQ/SH)</li>"
        f"<li>매크로 트리거 모니터링 (Fed, 금리, geo)</li>"
        f"<li>월간 리밸런싱 시그널 사전 점검</li>"
        f"</ol>")

    _section_html(11, "Forward Outlook & Action Plan — 시나리오·우선순위", "purple", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 12-16: Regime sections
# ─────────────────────────────────────────────────────────────────
def section_12(regime_stats: dict) -> None:
    if not regime_stats:
        return
    C = get_theme()
    regime = regime_stats.get("regime", "UNKNOWN")
    breadth = regime_stats.get("breadth", {})
    body = []
    body.append(_p(
        f"현재 시장 Regime: {badge(regime, regime_color(regime))}"
    ))
    if breadth:
        bull = breadth.get("pct_bullish") or breadth.get("bull_pct", 0)
        bear = breadth.get("pct_bearish") or breadth.get("bear_pct", 0)
        neut = breadth.get("pct_neutral") or breadth.get("neut_pct", 0)
        body.append(_p(
            f"Breadth 분포: 강세 {colored_span(f'{bull:.1f}%', C['green'], True)} · "
            f"약세 {colored_span(f'{bear:.1f}%', C['red'], True)} · "
            f"중립 {colored_span(f'{neut:.1f}%', C['gray'], True)}"
        ))
        avg_comp = breadth.get("avg_composite", 0)
        eligible_pct = breadth.get("pct_eligible") or breadth.get("eligible_pct", 0)
        body.append(_p(
            f"Composite 평균: <b>{avg_comp:.1f}</b> · Eligible(≥55) <b>{eligible_pct:.1f}%</b>"
        ))
        for axis in ("avg_tcs", "avg_tfs", "avg_rss", "avg_oer", "avg_rsi"):
            v = breadth.get(axis)
            if v is not None:
                body.append(_p(f"{axis.replace('avg_', '').upper()} 평균: {v:.1f}"))
    _section_html(12, "Market Regime 진단 — Cross-Sectional Signal Aggregation", "cyan", "".join(body))


def section_13(regime_stats: dict) -> None:
    groups = regime_stats.get("strategy_groups", []) if regime_stats else []
    if not groups:
        return
    C = get_theme()
    body = []
    body.append("<p><b>5 전략 그룹 Net Breadth</b>:</p>")
    rows = []
    for g in groups:
        name = g.get("group") or g.get("name", "?")
        nb = g.get("net_breadth", 0)
        desc = g.get("desc", "")
        color = C["green"] if nb >= 5 else (C["red"] if nb <= -5 else C["gray"])
        rows.append(
            f"<li><b>{name}</b>: Net {colored_span(f'{nb:+.1f}%', color, True)} · {desc}</li>"
        )
    body.append(f"<ul style='margin:2px 0 6px 16px'>{''.join(rows)}</ul>")
    _section_html(13, "Strategy Group Breadth — 5 그룹별 시장 성격", "cyan", "".join(body))


def section_14(regime_stats: dict) -> None:
    sb = regime_stats.get("strategy_breadth", []) if regime_stats else []
    if not sb:
        return
    C = get_theme()
    body = []
    long_leaders = sorted(sb, key=lambda x: -x.get("long_breadth", 0))[:3]
    short_leaders = sorted(sb, key=lambda x: -x.get("short_breadth", 0))[:3]
    body.append("<p><b>Long Bias Top 3</b>:</p>")
    body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(
        f"<li>{s.get('label', '?')}: L {s.get('long_breadth', 0):.1f}% · "
        f"S {s.get('short_breadth', 0):.1f}% · Net {s.get('net_breadth', 0):+.1f}%</li>"
        for s in long_leaders
    ) + "</ul>")
    body.append("<p><b>Short Bias Top 3</b>:</p>")
    body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(
        f"<li>{s.get('label', '?')}: L {s.get('long_breadth', 0):.1f}% · "
        f"S {s.get('short_breadth', 0):.1f}% · Net {s.get('net_breadth', 0):+.1f}%</li>"
        for s in short_leaders
    ) + "</ul>")
    _section_html(14, "8-Strategy Breadth (Individual)", "cyan", "".join(body))


def section_15(regime_stats: dict) -> None:
    sr = regime_stats.get("sector_regime", []) if regime_stats else []
    if not sr:
        return
    C = get_theme()
    body = []
    top_sec = sorted(sr, key=lambda x: -x.get("avg_composite", 0))[:5]
    bot_sec = sorted(sr, key=lambda x: x.get("avg_composite", 0))[:3]
    body.append("<p><b>Composite 상위 섹터 Top 5</b>:</p>")
    body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(
        f"<li>{s.get('sector', '?')}: Comp <b>{s.get('avg_composite', 0):.1f}</b> · "
        f"Bull {s.get('pct_bullish', 0):.1f}% · n={s.get('count', 0)}</li>"
        for s in top_sec
    ) + "</ul>")
    body.append("<p><b>Composite 하위 섹터 Bottom 3</b>:</p>")
    body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(
        f"<li>{s.get('sector', '?')}: Comp <b>{s.get('avg_composite', 0):.1f}</b> · "
        f"Bear {s.get('pct_bearish', 0):.1f}% · n={s.get('count', 0)}</li>"
        for s in bot_sec
    ) + "</ul>")
    if top_sec and bot_sec:
        gap = top_sec[0].get("avg_composite", 0) - bot_sec[0].get("avg_composite", 0)
        if gap > 25:
            msg = "강력한 섹터 알파 — 로테이션 적극 활용"
            mc = C["green"]
        elif gap > 15:
            msg = "정상 섹터 격차 — 균형 잡힌 알파"
            mc = C["cyan"]
        else:
            msg = "섹터 격차 제한 — 광범위 시장 의존"
            mc = C["gray"]
        body.append(_p(f"Top-Bot 격차 <b>{gap:.1f}p</b>: " + colored_span(msg, mc, True)))
    _section_html(15, "Sector Regime", "cyan", "".join(body))


def section_16(regime_stats: dict) -> None:
    hist = regime_stats.get("regime_history", []) if regime_stats else []
    if not hist or len(hist) < 2:
        return
    C = get_theme()
    body = []
    first, last = hist[0], hist[-1]
    body.append(_p(
        f"관측 기간: {first.get('date', '?')} → {last.get('date', '?')} · "
        f"<b>{len(hist)}개 시점</b>"
    ))
    comp_delta = (last.get("avg_composite", 0) - first.get("avg_composite", 0))
    body.append(_p(
        f"Composite 변화: {first.get('avg_composite', 0):.1f} → "
        f"{last.get('avg_composite', 0):.1f} ({colored_pct(comp_delta)})"
    ))
    bull_delta = last.get("pct_bullish", 0) - first.get("pct_bullish", 0)
    body.append(_p(
        f"Bullish % 변화: {first.get('pct_bullish', 0):.1f}% → "
        f"{last.get('pct_bullish', 0):.1f}% ({colored_pct(bull_delta)})"
    ))
    if comp_delta > 3 and bull_delta > 5:
        trend = "개선 추세"; tc = C["green"]
    elif comp_delta < -3 or bull_delta < -5:
        trend = "악화 추세"; tc = C["red"]
    else:
        trend = "횡보"; tc = C["gray"]
    body.append(_p(colored_span(f"→ {trend}", tc, True)))
    _section_html(16, "Regime History (시계열)", "cyan", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 17-19: Validation
# ─────────────────────────────────────────────────────────────────
def section_17(val_stats: dict) -> None:
    if not val_stats:
        return
    C = get_theme()
    body = []
    overall = val_stats.get("overall_mom", 0)
    color = verdict_color(overall)
    body.append(_p(
        f"Momentum 분류 전체 Pass Score: "
        f"{colored_span(f'{overall:.1f}%', color, True, size=14)}"
    ))
    def _val_li(m: dict) -> str:
        cls = m.get("classification", "?")
        score = m.get("pass_score", 0)
        score_html = colored_span(f"{score:.1f}%", verdict_color(score), True)
        return f"<li>{cls}: {score_html}</li>"

    mom = val_stats.get("momentum", [])
    if mom:
        hi = sorted(mom, key=lambda x: -x.get("pass_score", 0))[:3]
        lo = sorted(mom, key=lambda x: x.get("pass_score", 0))[:3]
        body.append("<p><b>고신뢰 Top 3</b>:</p>")
        body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(_val_li(m) for m in hi) + "</ul>")
        body.append("<p><b>저신뢰 Bottom 3</b>:</p>")
        body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(_val_li(m) for m in lo) + "</ul>")
    fails = val_stats.get("top_fails", [])[:5]
    if fails:
        body.append("<p><b>가장 빈번한 실패 메트릭</b>:</p>")
        body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(
            f"<li>{f.get('label', '?') if isinstance(f, dict) else str(f)}: "
            f"{f.get('count', 0) if isinstance(f, dict) else '?'}회</li>"
            for f in fails
        ) + "</ul>")
    _section_html(17, "Momentum Classification Validation", "cyan", "".join(body))


def section_18(val_stats: dict) -> None:
    if not val_stats:
        return
    C = get_theme()
    body = []
    overall = val_stats.get("overall_pm", 0)
    color = verdict_color(overall)
    body.append(_p(
        f"Pre-Momentum 분류 Pass Score: "
        f"{colored_span(f'{overall:.1f}%', color, True, size=14)}"
    ))
    def _pm_li(m: dict) -> str:
        cls = m.get("classification", "?")
        score = m.get("pass_score", 0)
        score_html = colored_span(f"{score:.1f}%", verdict_color(score), True)
        return f"<li>{cls}: {score_html}</li>"

    pm = val_stats.get("pre_momentum", [])
    if pm:
        hi = sorted(pm, key=lambda x: -x.get("pass_score", 0))[:3]
        lo = sorted(pm, key=lambda x: x.get("pass_score", 0))[:3]
        body.append("<p><b>고신뢰 Top 3</b>:</p>")
        body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(_pm_li(m) for m in hi) + "</ul>")
        body.append("<p><b>저신뢰 Bottom 3</b>:</p>")
        body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(_pm_li(m) for m in lo) + "</ul>")
    fails = val_stats.get("top_fails_pm", [])[:4]
    if fails:
        body.append("<p><b>PM 실패 메트릭</b>:</p>")
        body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(
            f"<li>{f.get('label', '?') if isinstance(f, dict) else str(f)}: "
            f"{f.get('count', 0) if isinstance(f, dict) else '?'}회</li>"
            for f in fails
        ) + "</ul>")
    mom_overall = val_stats.get("overall_mom", 0)
    gap = mom_overall - overall
    if gap > 10:
        body.append(_p(f"Mom-PM 격차 <b>{gap:+.1f}%p</b> — PM 분류 보정 필요"))
    _section_html(18, "Pre-Momentum Validation", "cyan", "".join(body))


def section_19(val_stats: dict) -> None:
    if not val_stats:
        return
    C = get_theme()
    body = []
    all_cls = (val_stats.get("momentum", []) or []) + (val_stats.get("pre_momentum", []) or [])
    hi = sum(1 for c in all_cls if c.get("pass_score", 0) >= 75)
    mid = sum(1 for c in all_cls if 50 <= c.get("pass_score", 0) < 75)
    lo = sum(1 for c in all_cls if c.get("pass_score", 0) < 50)
    body.append(_p(
        f"Confidence Tier: "
        f"{colored_span(f'High (≥75%) {hi}개', C['green'], True)} · "
        f"{colored_span(f'Partial (50-75%) {mid}개', C['amber'], True)} · "
        f"{colored_span(f'Low (<50%) {lo}개', C['red'], True)}"
    ))
    body.append("<p><b>실전 가이드</b>:</p>")
    body.append("<ul style='margin:2px 0 6px 16px'>"
        "<li>High: 풀 사이즈 진입 OK</li>"
        "<li>Partial: 절반 사이즈 + 타이트 스탑</li>"
        "<li>Low: 추가 확인 시그널 필요</li>"
        "</ul>")
    avg = (val_stats.get("overall_mom", 0) + val_stats.get("overall_pm", 0)) / 2
    if avg > 65: msg = "시스템 신뢰도 양호"; mc = C["green"]
    elif avg > 50: msg = "시스템 부분 신뢰 — 보조 검증 필요"; mc = C["amber"]
    else: msg = "시스템 threshold 재보정 필요"; mc = C["red"]
    body.append(_p(colored_span(f"→ {msg} (avg {avg:.1f}%)", mc, True)))
    _section_html(19, "System-level Reliability", "cyan", "".join(body))


# ─────────────────────────────────────────────────────────────────
# § 20-22: Quant
# ─────────────────────────────────────────────────────────────────
def section_20(quant_stats: dict) -> None:
    if not quant_stats:
        return
    C = get_theme()
    body = []
    nd = quant_stats.get("net_direction", "MIXED")
    nd_color = C["green"] if nd == "LONG" else (C["red"] if nd == "SHORT" else C["gray"])
    body.append(_p(
        f"Net Direction: {badge(nd, nd_color)} · "
        f"평균 Long {quant_stats.get('avg_long', 0):.1f} · "
        f"Neutral {quant_stats.get('avg_neutral', 0):.1f} · "
        f"Short {quant_stats.get('avg_short', 0):.1f}"
    ))
    strats = quant_stats.get("strategies", [])
    if strats:
        body.append("<p><b>전략별 시그널 분포</b>:</p>")
        rows = []
        for s in strats[:8]:
            name = s.get("name", s.get("key", "?"))
            n_long = s.get("n_long", 0)
            n_short = s.get("n_short", 0)
            n_neu = s.get("n_neutral", 0)
            net = n_long - n_short
            net_c = C["green"] if net > 0 else (C["red"] if net < 0 else C["gray"])
            rows.append(
                f"<li><b>{name}</b>: L {n_long} / N {n_neu} / S {n_short} · "
                f"Net {colored_span(f'{net:+d}', net_c, True)}</li>"
            )
        body.append(f"<ul style='margin:2px 0 6px 16px'>{''.join(rows)}</ul>")
    _section_html(20, "Quant Strategy 시그널 요약", "cyan", "".join(body))


def section_21(quant_stats: dict) -> None:
    if not quant_stats:
        return
    body = []
    cons = quant_stats.get("consensus", [])
    ultra = quant_stats.get("ultra_consensus", [])
    body.append(_p(
        f"Multi-Strategy Consensus (2+ strategy 일치): <b>{len(cons)}종목</b>"
    ))
    if ultra:
        body.append("<p><b>🌟 Ultra-Consensus (3+ strategy)</b>:</p>")
        body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(
            f"<li><b>{c['ticker']}</b> — {c['n_strategies']}개 전략 동시 추천 "
            f"({', '.join(c.get('strategies', [])[:3])})</li>"
            for c in ultra
        ) + "</ul>")
    if cons:
        body.append("<p><b>Top Consensus (2+)</b>:</p>")
        body.append("<ul style='margin:2px 0 6px 16px'>" + "".join(
            f"<li>{c['ticker']} — {c['n_strategies']}개 전략</li>"
            for c in cons[:8]
        ) + "</ul>")
    _section_html(21, "Multi-Strategy Consensus", "cyan", "".join(body))


def section_22(quant_stats: dict) -> None:
    if not quant_stats:
        return
    body = []
    body.append("<p><b>전략 적용 가이드</b>:</p>")
    body.append("<ul style='margin:2px 0 6px 16px'>"
        "<li>Dual Momentum: 월별 리밸런싱</li>"
        "<li>Sector Rotation: 월/분기 리밸런싱</li>"
        "<li>Trend Following: 단기 추세 추격</li>"
        "<li>Multi-Factor: 분기 리밸런싱</li>"
        "<li>Mean Reversion: 단기 반등 진입</li>"
        "<li>Inverse Volatility: Risk Parity 분배</li>"
        "</ul>")
    avg_l = quant_stats.get("avg_long", 0)
    avg_s = quant_stats.get("avg_short", 0)
    if avg_s > 0:
        ratio = avg_l / avg_s
        body.append(_p(f"Long/Short Ratio: <b>{ratio:.2f}</b>"))
    _section_html(22, "Strategy Allocation & Guide", "cyan", "".join(body))
