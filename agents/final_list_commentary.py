# -*- coding: utf-8 -*-
"""final_list_commentary.py — Executive Commentary for Buy/Sell Final List.

Generates 1000-character Korean commentary explaining:
  - 3-Agent voting 결과 기반 선정 근거
  - Sector/theme 분포 + 합의도
  - Risk 평가
  - 향후 전망

Caches by (date + top-20 tickers content hash) to avoid redundant LLM calls.
"""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Optional


CACHE_PATH = Path(".final_list_commentary_cache.json")
CACHE_TTL_HOURS = 12   # regenerate if cached commentary is older than this


def _now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _content_hash(items: list[dict]) -> str:
    """Stable hash of top items by ticker + consensus."""
    sig_parts = []
    for r in items[:20]:
        sig_parts.append(f"{r.get('ticker','')}-{r.get('consensus','')}-{r.get('stars','')}")
    return hashlib.md5("|".join(sig_parts).encode()).hexdigest()[:16]


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def _cache_valid(entry: dict) -> bool:
    if not entry: return False
    try:
        cached_at = entry.get("cached_at")
        if not cached_at: return False
        from datetime import datetime, timedelta
        ts = datetime.strptime(cached_at, "%Y-%m-%dT%H:%M:%S")
        age = datetime.now() - ts
        return age < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False


def _summarize_for_prompt(items: list[dict], side: str) -> str:
    """Build a compact summary of buy/sell list for the LLM prompt."""
    from collections import Counter
    n = len(items)
    if n == 0:
        return f"({side} 리스트 비어있음 — 0종목)"

    # Consensus distribution
    consensus_dist = Counter(r.get("consensus", "?") for r in items)
    # Sector distribution
    sector_dist = Counter(r.get("sector", "?") for r in items)
    top_sectors = sector_dist.most_common(5)

    # Top picks (★★★ UNANIMOUS first, then MAJORITY_CLEAN)
    top_unanimous = [r for r in items if r.get("consensus") == "UNANIMOUS"][:6]
    top_majority  = [r for r in items if r.get("consensus") == "MAJORITY_CLEAN"][:4]

    # Risk score distribution
    risks = [r.get("risk_score") or 0 for r in items]
    avg_risk = sum(risks) / len(risks) if risks else 0

    lines = [
        f"총 {n}개 {side} 후보",
        f"Consensus 분포: " + " · ".join(f"{c}={n2}" for c, n2 in consensus_dist.most_common(5)),
        f"섹터 분포 Top 5: " + " · ".join(f"{s}({n2})" for s, n2 in top_sectors),
        f"평균 Risk score: {avg_risk:.1f}/100 (낮을수록 안전)",
        "",
        "★★★ UNANIMOUS 상위 6:",
    ]
    for r in top_unanimous:
        lines.append(f"  - {r.get('ticker','?')} ({r.get('name','')[:25]}) "
                      f"sector={r.get('sector','?')} comp={r.get('composite','?')} risk={r.get('risk_score','?')}")
    if top_majority:
        lines.append("MAJORITY_CLEAN 상위 4:")
        for r in top_majority:
            lines.append(f"  - {r.get('ticker','?')} ({r.get('name','')[:25]}) sector={r.get('sector','?')}")
    return "\n".join(lines)


def _buy_prompt(buy_list: list[dict], market_context: str = "") -> str:
    summary = _summarize_for_prompt(buy_list, "매수")
    return f"""당신은 포트폴리오 전략가입니다. 아래 매수 Final List를 분석하여 **정확히 1000자 한국어** Executive Commentary를 작성하세요.

## 매수 Final List 요약
{summary}

{f"## 시장 컨텍스트{chr(10)}{market_context}" if market_context else ""}

## 작성 가이드
1. **선정 근거** (300자 내외): 3-Agent Voting (PM + Trading + Risk) 합의 결과 어떻게 매수 후보가 도출됐는지 설명
2. **섹터/테마 분석** (300자 내외): Top 섹터 집중도, 핵심 테마, sector 회전 시사점
3. **위험 평가** (200자 내외): Risk Manager 점수 의미, 잠재 리스크
4. **향후 전망 + 액션** (200자 내외): 다음 1-4주 시나리오, 실행 우선순위

## 출력 요구사항
- 정확히 한국어 1000자 (±50자 허용)
- 자연스러운 단락 (3-4 단락)
- 구체적 종목 ticker 언급 (★★★ UNANIMOUS 위주)
- 객관적 톤 (sales talk 아닌 분석)

JSON으로 반환:
```json
{{"commentary": "여기에 1000자 commentary 작성"}}
```

JSON 외 다른 텍스트 출력 금지."""


def _sell_prompt(sell_list: list[dict], market_context: str = "") -> str:
    summary = _summarize_for_prompt(sell_list, "매도/공매도")
    return f"""당신은 포트폴리오 전략가입니다. 아래 매도(공매도) Final List를 분석하여 **정확히 1000자 한국어** Executive Commentary를 작성하세요.

## 매도/공매도 Final List 요약
{summary}

{f"## 시장 컨텍스트{chr(10)}{market_context}" if market_context else ""}

## 작성 가이드
1. **선정 근거** (300자): 3-Agent Voting이 어떤 약세 신호를 검증했는지 (PM의 약세 분류, Trading의 BUY_NOW=open short, Risk 평가)
2. **섹터/테마 분석** (300자): 약세 sector, theme rotation, 매크로 시사점
3. **위험 평가** (200자): 숏 squeeze risk, liquidity, hedge 비중
4. **향후 전망 + 액션** (200자): 다음 1-4주 시나리오, 청산 trigger

## 출력 요구사항
- 정확히 한국어 1000자 (±50자 허용)
- 자연스러운 단락 (3-4 단락)
- 구체적 종목 ticker 언급
- 객관적 톤

JSON으로 반환:
```json
{{"commentary": "여기에 1000자 commentary 작성"}}
```

JSON 외 다른 텍스트 출력 금지."""


def _generate_commentary(prompt: str, timeout: int = 180) -> Optional[str]:
    """Run Claude to generate commentary. Returns text or None on failure."""
    try:
        from agents.market_leaders_swarm import _run_claude
        result = _run_claude(prompt, timeout=timeout, retries=1)
        return (result.get("commentary") or "").strip() if isinstance(result, dict) else None
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────
# Category-specific commentaries (per ENTERED / EXIT_PENDING / HOLDING / NEW)
# ─────────────────────────────────────────────────────────────────────

def _summarize_category(items: list[dict], category: str) -> str:
    """Compact summary of items in one category for LLM prompt."""
    from collections import Counter
    n = len(items)
    if n == 0:
        return f"({category} 카테고리 비어있음 — 0종목)"
    sector_dist = Counter((r.get("sector") or "?") for r in items)
    top_sectors = ", ".join(f"{s}({n2})" for s, n2 in sector_dist.most_common(5))
    risks = [(r.get("risk_score") or 0) for r in items if r.get("risk_score") is not None]
    avg_risk = sum(risks) / len(risks) if risks else 0
    horizon_dist = Counter((r.get("horizon") or "?") for r in items)
    days_held_list = [r.get("days_held", 0) for r in items if r.get("days_held")]
    avg_days = sum(days_held_list) / len(days_held_list) if days_held_list else 0

    lines = [
        f"카테고리: {category} — 총 {n}종목",
        f"섹터 Top 5: {top_sectors}",
        f"평균 Risk Score: {avg_risk:.1f}/100",
        f"Horizon 분포: " + " · ".join(f"{h}={c}" for h, c in horizon_dist.most_common()),
    ]
    if avg_days > 0:
        lines.append(f"평균 보유 일수: {avg_days:.1f}일")

    # Highlight EXIT_PENDING sub-types: state-driven vs regime-flip (SHORT signal)
    if category == "EXIT_PENDING":
        promoted = [r for r in items if r.get("promoted_from_short")]
        if promoted:
            lines.append(f"\n[Regime-Flip 청산] 보유 중 LONG + 오늘 PM SHORT 시그널: {len(promoted)}종목")
            for r in promoted[:10]:
                lines.append(
                    f"  - {r.get('ticker','?'):10} ({r.get('sector','?')}) "
                    f"★{r.get('short_signal_stars','?')} {r.get('short_signal_action','?')} "
                    f"reason={(r.get('short_signal_reason') or '')[:60]}"
                )
        state_driven = [r for r in items if not r.get("promoted_from_short")]
        if state_driven:
            lines.append(f"\n[State-Driven 청산] Position state EXIT_PENDING: {len(state_driven)}종목")
            for r in state_driven[:10]:
                days = f" Day {r.get('days_held',0)}" if r.get('days_held') else ""
                lines.append(
                    f"  - {r.get('ticker','?'):10} sector={r.get('sector','?')}{days} "
                    f"reason={(r.get('exit_reason') or '')[:60]}"
                )
        return "\n".join(lines)

    lines.append("\n주요 종목 (Top 10):")
    for r in items[:10]:
        days = f" Day {r.get('days_held',0)}" if r.get('days_held') else ""
        lines.append(
            f"  - {r.get('ticker','?'):10} ({r.get('name','')[:25]}) "
            f"sector={r.get('sector','?')} comp={r.get('composite','?')}"
            f"{days}"
        )
    return "\n".join(lines)


def _category_prompt(items: list[dict], category: str,
                      market_context: str = "") -> str:
    """Generate prompt for one category's commentary."""
    summary = _summarize_category(items, category)

    # Tailored focus per category
    focus_map = {
        "ENTERED": (
            "## 작성 가이드 (오늘 진입 — 즉시 매수 대상)\n"
            "1. **진입 근거** (300자): Phase 5.6 hysteresis로 2일 BUY_NOW 확정된 의미, "
            "이 종목들의 공통적 강점, 3-Agent voting 통과 패턴 분석\n"
            "2. **섹터/테마 분석** (300자): 진입 종목들의 sector concentration, "
            "거시 환경 적합성, theme rotation 시사점\n"
            "3. **즉시 실행 우선순위** (200자): 어떤 종목부터 매수하면 좋은지, "
            "position sizing 권고, urgency 평가\n"
            "4. **위험 관리** (200자): 진입 시 stop-loss 설정, "
            "예상 보유 기간, 이익 실현 전략"
        ),
        "EXIT_PENDING": (
            "## 작성 가이드 (청산 후보 — 즉시 매도 검토)\n"
            "1. **청산 신호 분석** (300자): 두 가지 유형 구분 — "
            "(a) State-Driven (SKIP 2일/WAIT 5일 누적, hysteresis 종료), "
            "(b) Regime-Flip (보유 중 LONG에 오늘 PM이 SHORT 시그널 부여 → 즉시성 ↑). "
            "각 유형의 우선순위와 의미 분석\n"
            "2. **섹터/테마 분석** (300자): 청산 종목들의 sector trend, "
            "rotation 방향, 매크로 시그널과의 정합성, regime-flip이 sector-wide인지 ticker-specific인지\n"
            "3. **실행 방법** (200자): Regime-Flip은 즉시 청산, State-Driven은 부분 청산 vs 관망 선택, "
            "stop-loss trigger 활용, slippage 최소화\n"
            "4. **다음 단계** (200자): 청산 자금 재배분 방향, "
            "주의해야 할 connected positions, 시장 신호 모니터링"
        ),
        "HOLDING": (
            "## 작성 가이드 (보유 중 — 현재 포지션 평가)\n"
            "1. **포지션 안정성 평가** (300자): 평균 보유 일수, "
            "오늘 picks에 포함된 종목 vs 미포함 종목 비교, persistence 분석\n"
            "2. **섹터/테마 분석** (300자): 보유 portfolio의 섹터 distribution, "
            "분산 효과, 매크로 risk exposure\n"
            "3. **Hold vs Trim 판단** (200자): 각 종목별 trim 검토 기준, "
            "rebalancing 필요성, profit-taking 전략\n"
            "4. **모니터링 포인트** (200자): 보유 중 주의해야 할 신호, "
            "exit trigger conditions, 시장 변화에 따른 대응"
        ),
        "NEW": (
            "## 작성 가이드 (신규 후보 — 오늘 voting 통과)\n"
            "1. **신규 발굴 의미** (300자): 3-Agent voting 통과한 새 종목들의 "
            "공통 특징, 어제 대비 변화, 신규 시장 신호 분석\n"
            "2. **섹터/테마 분석** (300자): 신규 후보 sector concentration, "
            "emerging trends, theme rotation 시사점\n"
            "3. **진입 timing 판단** (200자): 즉시 watch vs 추가 confirmation 대기, "
            "Phase 5.6 hysteresis 진행 방향 예상\n"
            "4. **포지션 구축 전략** (200자): existing portfolio와의 조화, "
            "position sizing, 다음 1주일 모니터링 계획"
        ),
    }
    focus = focus_map.get(category, focus_map["NEW"])

    return f"""당신은 포트폴리오 전략가입니다. 아래 '{category}' 카테고리에 속한 종목들을 분석하여
**정확히 1000자 한국어** Executive Commentary를 작성하세요.

## 종목 요약
{summary}

{f"## 시장 컨텍스트{chr(10)}{market_context[:400]}" if market_context else ""}

{focus}

## 출력 요구사항
- 정확히 한국어 1000자 (±50자 허용)
- 자연스러운 단락 (3-4 단락)
- 구체적 종목 ticker 언급 (상위 종목 중심)
- 객관적 톤 (sales talk 아닌 분석)

JSON으로 반환:
```json
{{"commentary": "여기에 1000자 commentary 작성"}}
```

JSON 외 다른 텍스트 출력 금지."""


def build_category_commentaries(items_by_category: dict,
                                  market_context: str = "") -> dict:
    """Build commentary for each category in parallel.

    Args:
        items_by_category: {"ENTERED": [...], "EXIT_PENDING": [...], "HOLDING": [...], "NEW": [...]}
        market_context: optional PM commentary for context

    Returns:
        {
          "entered": {commentary, cached},
          "exit_pending": ...,
          "holding": ...,
          "new": ...,
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache = _load_cache()
    out = {}

    # Map category key → cache key
    CAT_TO_KEY = {
        "ENTERED":      "entered",
        "EXIT_PENDING": "exit_pending",
        "HOLDING":      "holding",
        "NEW":          "new",
    }

    def _process_one(cat: str) -> tuple[str, dict]:
        cache_key = CAT_TO_KEY[cat]
        items = items_by_category.get(cat) or []
        if not items:
            return cache_key, {"commentary": f"({cat} 카테고리 비어있음)", "cached": False, "n_items": 0}

        ch = _content_hash(items)
        entry = cache.get(f"cat_{cache_key}", {})
        if entry.get("content_hash") == ch and _cache_valid(entry):
            return cache_key, {"commentary": entry.get("commentary", ""), "cached": True, "n_items": len(items)}

        # Generate via LLM
        prompt = _category_prompt(items, cat, market_context)
        text = _generate_commentary(prompt, timeout=180)
        if not text:
            # Fallback to deterministic summary
            text = _fallback_category_commentary(items, cat)

        cache[f"cat_{cache_key}"] = {
            "content_hash": ch,
            "commentary":   text,
            "cached_at":    _now_ts(),
            "n_items":      len(items),
        }
        return cache_key, {"commentary": text, "cached": False, "n_items": len(items)}

    # Run 4 categories in parallel
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_process_one, cat): cat for cat in CAT_TO_KEY}
        for fut in as_completed(futs):
            try:
                k, v = fut.result()
                out[k] = v
            except Exception as e:
                cat = futs[fut]
                out[CAT_TO_KEY[cat]] = {"commentary": f"(생성 실패: {str(e)[:100]})",
                                          "cached": False, "n_items": 0}

    _save_cache(cache)
    return out


def _fallback_category_commentary(items: list[dict], category: str) -> str:
    """Template-based fallback when LLM fails."""
    from collections import Counter
    n = len(items)
    if n == 0:
        return f"{category} 카테고리에 종목 없음."
    sectors = Counter((r.get("sector") or "?") for r in items).most_common(3)
    top_tickers = ", ".join(r.get("ticker","?") for r in items[:5])
    avg_risk = sum(r.get("risk_score", 0) for r in items) / max(1, n)
    sector_str = ", ".join(f"{s}({c})" for s, c in sectors)

    cat_label = {
        "ENTERED": "오늘 진입 확정", "EXIT_PENDING": "청산 후보",
        "HOLDING": "보유 중", "NEW": "신규 후보"
    }.get(category, category)

    return (
        f"{cat_label} 카테고리에 총 {n}종목이 분포합니다. 대표 종목은 {top_tickers}이며, "
        f"섹터 집중도는 {sector_str} 순으로 나타납니다. 평균 Risk Score는 {avg_risk:.1f}/100로 "
        f"전반적 위험 수준이 측정되었습니다. "
        f"3-Agent Voting (PM + Trading + Risk) 합의 결과를 통해 도출된 후보군이며, "
        f"각 종목의 horizon (Tactical/Core/Strategic)에 따라 차별화된 전략이 권장됩니다. "
        f"향후 1-2주간 시장 모니터링을 통해 신호 변화를 추적하고, Phase 5.6 state machine의 "
        f"hysteresis 규칙에 따라 자동으로 보유/청산 결정이 갱신됩니다. "
        f"투자 결정 시 본인의 risk tolerance와 portfolio context를 함께 고려하시기 바랍니다."
    )


def _summarize_categories_for_unified(items_by_category: dict) -> str:
    """Detailed multi-category summary for unified 5000-char commentary prompt."""
    from collections import Counter
    sections = []
    cat_labels = {
        "ENTERED": "✓ 오늘 진입 (ENTERED — Phase 5.6 hysteresis 2일 BUY_NOW 확정)",
        "EXIT_PENDING": "⚠ 청산 후보 (EXIT_PENDING — State-Driven + Regime-Flip)",
        "HOLDING": "🔵 보유 중 (HOLDING — 이미 진입한 포지션)",
        "NEW": "🟢 신규 후보 (NEW — 오늘 처음 voting 통과)",
    }

    for cat, label in cat_labels.items():
        items = items_by_category.get(cat) or []
        n = len(items)
        if n == 0:
            sections.append(f"\n### {label}\n(카테고리 비어있음 — 0종목)")
            continue
        sec_dist = Counter((r.get("sector") or "?") for r in items).most_common(5)
        hor_dist = Counter((r.get("horizon") or "?") for r in items).most_common()
        risks = [r.get("risk_score") or 0 for r in items if r.get("risk_score") is not None]
        avg_risk = sum(risks) / len(risks) if risks else 0

        lines = [
            f"\n### {label} — 총 {n}종목",
            f"  섹터 Top 5: " + ", ".join(f"{s}({c})" for s, c in sec_dist),
            f"  Horizon 분포: " + " · ".join(f"{h}={c}" for h, c in hor_dist),
            f"  평균 Risk Score: {avg_risk:.1f}/100",
        ]

        # Special handling for EXIT_PENDING — split state-driven vs regime-flip
        if cat == "EXIT_PENDING":
            promoted = [r for r in items if r.get("promoted_from_short")]
            state_driven = [r for r in items if not r.get("promoted_from_short")]
            if promoted:
                lines.append(f"\n  [Regime-Flip 청산] 보유 LONG + 오늘 PM SHORT 시그널: {len(promoted)}종목")
                for r in promoted[:8]:
                    lines.append(
                        f"    · {r.get('ticker','?'):10} ({r.get('sector','?')[:18]:18}) "
                        f"★{r.get('short_signal_stars','?')} {r.get('short_signal_action','?')} "
                        f"h={r.get('short_signal_horizon','?')}"
                    )
                    rs = (r.get("short_signal_reason") or "")[:120]
                    if rs:
                        lines.append(f"        SHORT 시그널 근거: {rs}")
            if state_driven:
                lines.append(f"\n  [State-Driven 청산] Position state EXIT_PENDING: {len(state_driven)}종목")
                for r in state_driven[:6]:
                    er = (r.get("exit_reason") or "")[:80]
                    lines.append(f"    · {r.get('ticker','?'):10} days={r.get('days_held',0)} reason={er}")
        else:
            lines.append(f"\n  주요 종목 Top 12:")
            for r in items[:12]:
                days = f" Day {r.get('days_held',0)}" if r.get('days_held') else ""
                cons = r.get("consensus", "?")
                rat = (r.get("rationale") or "")[:80]
                lines.append(
                    f"    · {r.get('ticker','?'):10} ({r.get('name','')[:25]:25}) "
                    f"sec={r.get('sector','?')[:15]:15} comp={r.get('composite','?')} "
                    f"cons={cons}{days}"
                )
                if rat:
                    lines.append(f"        PM 근거: {rat}")
        sections.append("\n".join(lines))

    return "\n".join(sections)


def _build_phase1_briefing() -> str:
    """Extract Phase 1 agent verbatim findings to ground commentary in fresh data.

    The commentary LLM tends to fabricate macro scenarios (e.g. "FOMC dovish")
    even when upstream agents reported the opposite. Injecting the actual agent
    output VERBATIM forces the commentary to reflect real findings.
    """
    import json as _json
    try:
        sw = _json.load(open(".market_leaders_swarm_cache.json"))
    except Exception:
        return ""
    p1 = sw.get("phase1", {})
    lines = []
    for agent_name, label in [
        ("macro_analyst",          "📊 Macro Analyst"),
        ("cross_asset_analyst",    "📈 Cross-Asset Analyst"),
        ("news_narrative_analyst", "📰 News/Narrative Analyst"),
        ("sector_theme_analyst",   "🎯 Sector/Theme Analyst"),
        ("flow_momentum_analyst",  "💧 Flow/Momentum Analyst"),
    ]:
        a = p1.get(agent_name) or {}
        if not a:
            continue
        rating = a.get("rating", "—")
        conf = a.get("confidence", 0)
        narr = (a.get("narrative") or "")[:400]
        sigs = a.get("key_signals") or []
        risk = (a.get("biggest_risk") or "")[:200]
        opp  = (a.get("biggest_opportunity") or "")[:200]

        lines.append(f"\n### {label} — {rating} (conf {conf})")
        lines.append(f"  Narrative: {narr}")
        lines.append(f"  Key signals:")
        for s in sigs[:5]:
            lines.append(f"    • {s[:200]}")
        if risk:
            lines.append(f"  Biggest risk: {risk}")
        if opp:
            lines.append(f"  Biggest opportunity: {opp}")
    return "\n".join(lines) if lines else ""


def _unified_prompt(items_by_category: dict, market_context: str = "") -> str:
    """Build unified 5000-char Korean commentary prompt covering all categories + top-3 watchlist."""
    category_summary = _summarize_categories_for_unified(items_by_category)
    phase1_briefing = _build_phase1_briefing()

    # Build candidate pool for top-3 selection (across ENTERED + NEW + HOLDING + EXIT_PENDING)
    all_items = []
    for cat in ("ENTERED", "NEW", "HOLDING", "EXIT_PENDING"):
        for r in items_by_category.get(cat) or []:
            all_items.append({**r, "_category": cat})
    # Compact ticker pool for the LLM to pick from
    pool_lines = []
    for r in all_items[:60]:
        pool_lines.append(
            f"  - {r.get('ticker','?'):10} cat={r.get('_category','?')} "
            f"sec={r.get('sector','?')[:15]:15} comp={r.get('composite','?')} "
            f"cons={r.get('consensus','?')} h={r.get('horizon','?')}"
        )
    pool_str = "\n".join(pool_lines)

    # Build per-horizon ticker pools for horizon-specific commentary sections
    by_horizon: dict[str, list[dict]] = {"tactical": [], "core": [], "strategic": []}
    for r in all_items:
        h = r.get("horizon")
        if h in by_horizon:
            by_horizon[h].append(r)

    def _fmt_horizon_pool(h: str) -> str:
        items = by_horizon.get(h) or []
        if not items:
            return f"({h} horizon에 종목 없음)"
        from collections import Counter
        sec_dist = Counter((r.get("sector") or "?") for r in items).most_common(4)
        cat_dist = Counter((r.get("_category") or "?") for r in items).most_common()
        lines = [
            f"총 {len(items)}종목 · 섹터: " + ", ".join(f"{s}({c})" for s, c in sec_dist),
            f"카테고리 분포: " + " · ".join(f"{c}={n}" for c, n in cat_dist),
            "주요 종목:",
        ]
        for r in items[:10]:
            lines.append(
                f"  · {r.get('ticker','?'):10} cat={r.get('_category','?'):13} "
                f"sec={(r.get('sector','?') or '')[:14]:14} comp={r.get('composite','?')} "
                f"cons={r.get('consensus','?')}"
            )
        return "\n".join(lines)

    tactical_pool   = _fmt_horizon_pool("tactical")
    core_pool       = _fmt_horizon_pool("core")
    strategic_pool  = _fmt_horizon_pool("strategic")

    return f"""당신은 시니어 포트폴리오 전략가입니다. 아래 매수 Final List 전체 (오늘 진입 / 보유 중 / 청산 후보 / 신규 후보)를 통합 분석하여 **정확히 한국어 12,000자 (±400자 허용)** 의 매우 자세한 Executive Commentary를 작성하세요.

⚠ 중요 — 턴오버 관리: 전체 매수 final list는 quality 점수 상위 stock 20개 + ETF 20개 = 최대 40종목으로 제한됨. 각 카테고리는 이 40개 풀에서 분배되며, asset-class 균형(주식 vs ETF)이 유지됩니다.

═══════════════════════════════════════════════════════════════════
## 매수 Final List 카테고리별 상세 요약 (전체 stock 20 + ETF 20 cap)
═══════════════════════════════════════════════════════════════════
{category_summary}

═══════════════════════════════════════════════════════════════════
## 호라이즌별 풀 (tactical 5d / core 21d / strategic 63d)
═══════════════════════════════════════════════════════════════════
[TACTICAL 5d]
{tactical_pool}

[CORE 21d]
{core_pool}

[STRATEGIC 63d]
{strategic_pool}

═══════════════════════════════════════════════════════════════════
## 전체 종목 풀 (Top-3 유심히 보아야 할 종목 선정용)
═══════════════════════════════════════════════════════════════════
{pool_str}

{f"## 시장 컨텍스트 (PM Commentary){chr(10)}{market_context[:600]}" if market_context else ""}

═══════════════════════════════════════════════════════════════════
## ⚠ FACT BASE — Phase 1 Agent 원본 분석 결과 (반드시 인용)
═══════════════════════════════════════════════════════════════════
아래는 오늘 실행된 swarm의 phase1 agent들이 직접 분석한 결과입니다.
Commentary 작성 시 macro 시나리오, FOMC/Fed 입장, 글로벌 정책, 섹터 로테이션 등은
반드시 아래 내용을 충실히 반영해야 합니다. 이와 모순되는 내용 절대 금지.

{phase1_briefing if phase1_briefing else "(Phase 1 briefing not available)"}

═══════════════════════════════════════════════════════════════════
## 작성 가이드 — 정확히 다음 11개 섹션으로 작성 (총 12,000자)
═══════════════════════════════════════════════════════════════════

[섹션 1: 거시 시장 진단 — 약 1,000자]
오늘의 매수 Final List 전체 분포에서 도출되는 거시 시그널을 분석.
- 카테고리 간 비율 (ENTERED N개 / NEW N개 / HOLDING N개 / EXIT_PENDING N개)이 주는 의미
- 섹터 집중도/분산 패턴이 시사하는 시장 regime (위험 선호 vs 회피, 성장 vs 가치, 사이클 vs 방어)
- 매수 후보 평균 risk score, horizon 분포가 의미하는 trader appetite
- Regime-Flip 청산 종목이 어떤 매크로 전환을 암시하는지
- 글로벌 매크로 동조화 여부 (US/Japan/Europe 중앙은행)

[섹션 2: ✓ ENTERED (오늘 진입) 카테고리 심층 분석 — 약 800자]
ENTERED 카테고리 내 종목별 진입 근거 + 공통 강점 + Phase 5.6 hysteresis 2일 BUY_NOW 시그널 의미
실행 우선순위, position sizing, entry zone, stop-loss 권고, 예상 보유 기간

[섹션 3: 🟢 NEW (신규 후보) 카테고리 심층 분석 — 약 800자]
NEW 카테고리 내 종목별 emerging theme + 오늘 처음 voting 통과한 시장 신호
즉시 watch vs 추가 confirmation 대기 결정 기준, 1주일 모니터링 포인트
기존 portfolio와의 조화 (overlap/diversification)

[섹션 4: 🔵 HOLDING (보유 중) 카테고리 심층 분석 — 약 700자]
HOLDING 카테고리 내 종목별 평균 보유 일수, '✓오늘picks' 비율의 의미
sector distribution + 분산 효과, Hold vs Trim 판단 기준
Profit-taking 전략과 rebalancing 필요성, exit trigger

[섹션 5: ⚠ EXIT_PENDING (청산 후보) 카테고리 심층 분석 — 약 800자]
EXIT_PENDING 카테고리 내 종목별 State-Driven vs Regime-Flip 구분 분석
Regime-Flip이 sector-wide인지 ticker-specific인지 (매크로 vs 개별)
즉시 청산 우선순위 (Regime-Flip이 더 긴급), 자금 재배분 방향
시장 신호 모니터링 포인트

═══════════════════════════════════════════════════════════════════
## 호라이즌별 분석 (NEW — 시간축별 깊이 있는 분석)
═══════════════════════════════════════════════════════════════════

[섹션 6: ⚡ TACTICAL 5d 호라이즌 종합 분석 + 전망 — 약 1,200자]
Tactical (5-day) 호라이즌 종목 풀 전체에 대한 종합 분석.
- 5일 단기 전략의 핵심 thesis (단기 모멘텀 + 기술적 브레이크아웃 + 즉시 catalyst)
- Tactical 종목들의 공통적 시그널 (예: FORMATION 분류 비율, OER 평균, RSS short)
- 1주일 내 예상 시나리오 (가격 행동, 실현 가능한 수익률 분포)
- Tactical에서만 등장하는 종목 vs Core/Strategic과 중복되는 종목 구분
- 단기 catalyst (실적, 매크로 이벤트, 기술적 이정표) 임박도
- Tactical 호라이즌만의 위험 요인 (whipsaw, false breakout, 유동성 갑작스러운 변화)
- 5일 후 어떤 종목이 ENTERED → HOLDING으로 안착할지 예측

[섹션 7: 📊 CORE 21d 호라이즌 종합 분석 + 전망 — 약 1,200자]
Core (21-day, 1개월) 호라이즌 종목 풀 전체에 대한 종합 분석.
- 1개월 중기 전략의 핵심 thesis (추세 확립 + 펀더멘털 + 섹터 로테이션 정합)
- Core 종목들의 공통적 시그널 (예: CONTINUATION 분류, RSS long, QVR 분포)
- 1개월 내 예상 시나리오 (sector rotation, theme rotation 진행 방향)
- Core 호라이즌의 portfolio 안정성 기여 (핵심 보유 axis 역할)
- 1개월 중기 catalyst (분기 실적, FOMC, ECB 결정) 영향
- Core 호라이즌만의 위험 요인 (regime change, theme exhaustion)
- 1개월 후 어떤 종목이 strategic으로 승격되거나 청산될지 예측

[섹션 8: 🎯 STRATEGIC 63d 호라이즌 종합 분석 + 전망 — 약 1,200자]
Strategic (63-day, 3개월) 호라이즌 종목 풀 전체에 대한 종합 분석.
- 3개월 장기 전략의 핵심 thesis (구조적 변화 + 매크로 cycle + secular trend)
- Strategic 종목들의 공통적 시그널 (장기 추세 안정성, 펀더멘털 우위, 매크로 정합)
- 3개월 내 예상 시나리오 (cyclical/secular 분리, 글로벌 자본 흐름)
- Strategic 호라이즌의 portfolio anchor 역할 (장기 베타 + 알파)
- 3개월 장기 catalyst (대선, 정책 변화, sector cycle 진행) 영향
- Strategic 호라이즌만의 위험 요인 (long-term theme reversal, 매크로 regime change)
- 3개월 후 portfolio의 sector tilt가 어떻게 변할지 예측

═══════════════════════════════════════════════════════════════════
## 역사적 유사 구간 분석 (NEW)
═══════════════════════════════════════════════════════════════════

[섹션 9: 🕰 과거 유사 구간 (Historical Analog) — 약 2,000자]
현재 금융시장 상황과 유사했던 과거 구간이 있었는지 분석하고, 있다면 함의점을 도출.

작성 시 반드시 다음 7가지 차원을 점검 후 유사 정도가 가장 높은 과거 사례 2~3개를 선정:
1. **거시 환경**: Fed 정책 사이클 (인상/인하/동결 + dot-plot 스탠스), 인플레이션 vs 성장 trade-off, 글로벌 통화정책 동조화 여부
2. **신용시장 상태**: HY OAS 절대 수준 vs 역사적 분위, 신용-에쿼티 괴리 (Phase 1 cross_asset_analyst의 OAS 수치 인용)
3. **밸류에이션 상태**: Composite avg, OER 분포, 시장 breadth, 과열 vs 회복 단계
4. **섹터 로테이션 패턴**: 어떤 섹터가 leading, 어떤 섹터가 lagging — 후기 사이클 / 초기 사이클 / 중기 사이클 특성
5. **지정학 / 매크로 이벤트**: 이번 케이스의 핵심 이슈가 과거 유사 사건과 어떻게 매칭되는지 (news_narrative_analyst 인용)
6. **VIX / DXY / 금리커브**: 변동성·달러·yield curve 형상 (cross_asset_analyst 인용)
7. **시장 내러티브**: AI/Tech 집중도, 로테이션 vs 광폭, defensive vs cyclical 우위

각 과거 사례별 약 600~700자로:

**🏛 사례 1 (가장 강한 유사성)**:
- 시점: 정확한 연·월·구간 (예: "2018년 9월~12월", "2007년 10월~2008년 1월", "1998년 7월~10월")
- 당시 상황 요약: Fed 정책, 매크로 환경, 시장 분위기
- 본 사례와의 유사점 7가지 차원 중 매칭되는 항목 (최소 4개 매칭)
- 당시 이후 시장 궤적: 1개월 / 3개월 / 6개월 후 어떻게 전개됐는지 (정확한 수치 포함)
- 당시 best performer 섹터/종목 (가능한 한 구체)
- 당시 worst performer 섹터/종목
- 본 케이스에 적용 가능한 학습 (lesson)

**🏛 사례 2 (보조 사례)**:
- 동일 구조로 400~500자

**🏛 사례 3 (대조 사례 — 차이점 강조)**:
- 본 사례와의 결정적 차이점 (왜 단순 적용 불가한지) — 300자

**⚖ 종합 함의 (약 400자)**:
- 위 사례들을 종합하여 본 매수 Final List에 적용 가능한 전략적 함의 3가지
- 과거 패턴이 반복될 확률 평가 (높음 / 중간 / 낮음 — 근거 명시)
- "이번엔 다르다"는 차이 요인 (this time is different)
- Top-3 watchlist 종목 선정 시 과거 학습 어떻게 반영할지

⚠ 작성 시 주의:
- **Phase 1 macro_analyst + cross_asset_analyst + news_narrative_analyst 의 FACT BASE 데이터에 명시된 매크로 상태와 일치하는 과거 사례만 선정** (예: macro_analyst가 "Fed 매파 동결"이라고 보고 시 → 매파 인상 사이클 구간 우선 매칭, "비둘기 인하" 시 → 인하 사이클 매칭)
- 학습 데이터로 알려진 시장사건 (1987 Black Monday, 1998 LTCM, 2000 Dot-com, 2008 GFC, 2011 Eurozone, 2015-16 Manufacturing recession, 2018 hawkish hike cycle, 2020 COVID, 2022 hawkish tightening, 2023 banking crisis 등) 중 7가지 차원 매칭 가장 강한 사례 선정
- 과도하게 비관적 / 낙관적 사례만 선정 금지 — 객관적 매칭 우선
- 단순 "유사하다" 서술 금지 — 수치·시기·정책 액션을 구체적으로 인용

═══════════════════════════════════════════════════════════════════
## 종합 시사점
═══════════════════════════════════════════════════════════════════

[섹션 10: 🌟 유심히 보아야 할 Top-3 종목 — 약 1,300자]
전체 매수 Final List (ENTERED/NEW/HOLDING/EXIT_PENDING 모두 포함, 모든 호라이즌)에서 다음 측면을 종합 고려하여 가장 주목할 종목 3개를 선정:
- Conviction (3-Agent UNANIMOUS or 강한 합의)
- Asymmetry (잠재 상승 vs 하방 risk 비대칭)
- Catalyst 임박성 (실적/규제/매크로 이벤트)
- Sector rotation alignment (현재 시장 leadership과 정합)
- Multi-horizon 일치 (tactical+core+strategic 모두 등장 → 강한 확신)
- Risk-adjusted attractiveness
- **섹션 9의 과거 유사 구간 학습 반영** (해당 패턴에서 outperform한 섹터/스타일과 정합되는 종목 우선)

각 종목당 약 430자로:
- 🥇 1순위 종목 (티커, 카테고리, sector, 호라이즌): 왜 유심히 보아야 하는지 — 진입/청산/관망 권고와 timing, key risk, watch-out signal, 예상 수익률 시나리오, **섹션 9 사례 1/2/3 중 어디서 가장 강한 지지를 받는지**
- 🥈 2순위 종목: 동일 구조
- 🥉 3순위 종목: 동일 구조

[섹션 11: 💼 금융시장 시사점 + 종합 액션 플랜 — 약 800자]
- 이번 매수 Final List가 말하는 다음 1-12주 시장 시나리오
- Base case / Bull case / Bear case 시나리오 별 portfolio 대응
- **각 시나리오의 확률은 섹션 9 과거 유사 사례의 분포에서 도출 (예: 사례 1·2 모두 상승 마감 → Base case 확률 ↑)**
- 호라이즌별 비중 배분 권고 (tactical/core/strategic 가중치)
- 종합적 실행 우선순위 (오늘 / 1주 / 1개월 / 3개월)
- 모니터링해야 할 거시 지표/event (Fed dot plot, ECB rate, BOJ stance, HY OAS, VIX, DXY)
- Cross-asset 정합성 확인 포인트
- **섹션 9의 "이번엔 다르다" 요인이 base case를 무력화시킬 수 있는 trigger 명시**

═══════════════════════════════════════════════════════════════════
## 출력 요구사항
═══════════════════════════════════════════════════════════════════
- 정확히 한국어 12,000자 (±400자 허용)
- 모든 11개 섹션 포함 (1: 거시, 2-5: 카테고리, 6-8: 호라이즌, 9: 과거 유사 구간 (NEW), 10: Top-3, 11: 액션 플랜)
- 각 섹션 시작에 ## 헤더 사용 (예: "## 섹션 6: ⚡ TACTICAL 5d 호라이즌 종합 분석", "## 섹션 9: 🕰 과거 유사 구간 분석")
- 구체적 종목 ticker 풍부히 언급 (★★★ UNANIMOUS 위주)
- 객관적 분석 톤 (sales talk 금지)
- Top-3는 1순위/2순위/3순위 명확히 구분 (🥇🥈🥉)

⚠ ANTI-HALLUCINATION RULES (강제 준수):
1. **FOMC/Fed 입장**은 위 "FACT BASE" 섹션의 macro_analyst rating + narrative + key_signals
   에 명시된 내용만 사용. 만약 macro_analyst가 "매파적" 또는 "hawkish" 라고 보고했다면
   commentary 에서 "비둘기파" 또는 "dovish" 시나리오를 만들면 안 됨 (반대도 동일).
   예: macro_analyst가 "Fed 닷플롯 매파 전환" → Bull case에 "FOMC 비둘기 확인" 절대 금지.
2. **외국 중앙은행 (BOJ/ECB/BOK)** 입장은 cross_asset_analyst의 key_signals/narrative에
   명시된 내용만 인용. 명시 안 됐으면 commentary에서 언급 자체 금지.
3. **Bull/Base/Bear case 시나리오**도 macro_analyst의 narrative + biggest_risk/opportunity
   범위 내에서만 구성. Phase 1 데이터에 없는 가정 (예: "Fed 7월 인하 확인") 금지.
4. **글로벌 지정학**(이란, 호르무즈, 우크라이나 등) 도 news_narrative_analyst 인용 시에만 사용.
5. 학습 시점 데이터 기반 휴리스틱 (예: "BOJ는 전통적으로 dovish") 사용 절대 금지.
   → 모든 macro 판단은 위 FACT BASE에 명시된 오늘 swarm의 분석을 그대로 반영.

JSON으로 반환:
```json
{{"commentary": "전체 한국어 12,000자 commentary..."}}
```

JSON 외 다른 텍스트 출력 금지."""


# ─────────────────────────────────────────────────────────────────────
# Option B: Split commentary into 3 (공통 거시 + Stock + ETF)
# Each runs in parallel as its own LLM call.
# ─────────────────────────────────────────────────────────────────────

def _split_categories_by_asset(items_by_category: dict) -> tuple[dict, dict]:
    """Split items_by_category into stocks-only and ETFs-only dicts."""
    def _is_etf(r: dict) -> bool:
        bk = (r.get("bucket") or "").lower()
        if "etf" in bk: return True
        if "stock" in bk: return False
        at = (r.get("asset_type") or r.get("type") or "").lower()
        return "etf" in at

    stocks_by_cat: dict[str, list[dict]] = {}
    etfs_by_cat: dict[str, list[dict]] = {}
    for cat, rows in items_by_category.items():
        stocks_by_cat[cat] = [r for r in (rows or []) if not _is_etf(r)]
        etfs_by_cat[cat]   = [r for r in (rows or []) if _is_etf(r)]
    return stocks_by_cat, etfs_by_cat


def _common_macro_prompt(items_by_category: dict, market_context: str = "") -> str:
    """Common macro lens — 1,500자.

    Shared across Stock and ETF analyses. Establishes the macro regime + historical
    analog facts so the per-asset commentaries don't duplicate/contradict.
    """
    phase1_briefing = _build_phase1_briefing()
    stocks_by_cat, etfs_by_cat = _split_categories_by_asset(items_by_category)

    s_counts = {c: len(stocks_by_cat.get(c, [])) for c in ("ENTERED","NEW","HOLDING","EXIT_PENDING")}
    e_counts = {c: len(etfs_by_cat.get(c, []))   for c in ("ENTERED","NEW","HOLDING","EXIT_PENDING")}

    return f"""당신은 시니어 포트폴리오 전략가입니다. 매수 Final List 통합 분석의 **공통 거시 헤드라인**을 **정확히 한국어 1,500자 (±100자)** 으로 작성하세요.

이 commentary는 Stock 분석 + ETF 분석의 공통 기반(base)이 됩니다. 매크로/거시/과거 유사 구간만 다루며, 개별 종목·섹터 ETF 비중 등은 분석 금지.

═══════════════════════════════════════════════════════════════════
## 매수 Final List 분포 (asset type 별 카테고리)
═══════════════════════════════════════════════════════════════════
📈 Stocks (총 {sum(s_counts.values())}종목): ENTERED {s_counts['ENTERED']} · NEW {s_counts['NEW']} · HOLDING {s_counts['HOLDING']} · EXIT_PENDING {s_counts['EXIT_PENDING']}
📦 ETFs (총 {sum(e_counts.values())}종목): ENTERED {e_counts['ENTERED']} · NEW {e_counts['NEW']} · HOLDING {e_counts['HOLDING']} · EXIT_PENDING {e_counts['EXIT_PENDING']}

{f"## 시장 컨텍스트 (PM Commentary){chr(10)}{market_context[:400]}" if market_context else ""}

═══════════════════════════════════════════════════════════════════
## ⚠ FACT BASE — Phase 1 Agent 원본 분석 (반드시 인용)
═══════════════════════════════════════════════════════════════════
{phase1_briefing if phase1_briefing else "(Phase 1 briefing not available)"}

═══════════════════════════════════════════════════════════════════
## 작성 가이드 — 정확히 다음 3개 섹션으로 (총 1,500자)
═══════════════════════════════════════════════════════════════════

[섹션 A: 🌍 거시 환경 종합 진단 — 약 500자]
- Phase 1 agent들의 핵심 시그널 (Fed/BOJ/ECB 정책, HY OAS, VIX, DXY, 지정학)
- 글로벌 매크로 동조화 vs 분기 진단
- 현재 시장의 핵심 trade-off (인플레이션 vs 성장 vs 신용)
- Stock vs ETF 자산군별 카테고리 분포가 시사하는 trader appetite
- 매수 Final List 전체의 portfolio risk posture (offensive / defensive / neutral)

[섹션 B: 🕰 과거 유사 구간 분석 — 약 700자]
현재 매크로 상황과 유사했던 과거 사례 2개 선정 (Stock+ETF 양쪽 분석의 공통 기반).

**🏛 사례 1 (가장 강한 유사성)** — 약 400자:
- 시점 (정확한 연·월)
- 당시 상황 + 본 사례와 매칭되는 차원 4개 이상
- 이후 궤적 (1개월 / 3개월 / 6개월 후 수치)
- 당시 best/worst performer (섹터 + 스타일)
- 학습

**🏛 사례 2 (보조 사례)** — 약 250자:
- 시점, 매칭 차원, 궤적, 학습

**⚖ 종합 (50자)**: 과거 패턴 반복 확률 (높음/중간/낮음 + 근거 1개)

[섹션 C: 🎯 매크로 시나리오 + 트리거 — 약 300자]
Stock/ETF 분석이 공유할 공통 시나리오 프레임:
- Base / Bull / Bear case 확률 (섹션 B 사례 분포 기반)
- 각 시나리오 발동 trigger (Fed dot plot, HY OAS, VIX, DXY 임계치)
- "이번엔 다르다" 요인 1가지 (시나리오 무력화 가능)

═══════════════════════════════════════════════════════════════════
## 출력 요구사항
═══════════════════════════════════════════════════════════════════
- 정확히 한국어 1,500자 (±100자)
- 모든 3개 섹션 포함 (A: 거시, B: 과거 유사 구간, C: 시나리오)
- 각 섹션 시작에 ## 헤더 사용
- ⚠ ANTI-HALLUCINATION: FACT BASE에 없는 macro 주장 금지 (BOJ/Fed/ECB 입장 등)

JSON으로 반환:
```json
{{"commentary": "전체 한국어 1,500자 공통 거시 commentary..."}}
```

JSON 외 다른 텍스트 출력 금지."""


def _stock_prompt(stocks_by_cat: dict, market_context: str = "") -> str:
    """Stock deep-dive — 5,500자. 개별 펀더멘털/실적/catalyst 중심."""
    phase1_briefing = _build_phase1_briefing()
    category_summary = _summarize_categories_for_unified(stocks_by_cat)

    # Build per-horizon pools
    by_horizon: dict[str, list[dict]] = {"tactical": [], "core": [], "strategic": []}
    all_stocks: list[dict] = []
    for cat in ("ENTERED", "NEW", "HOLDING", "EXIT_PENDING"):
        for r in stocks_by_cat.get(cat) or []:
            r2 = {**r, "_category": cat}
            all_stocks.append(r2)
            h = r.get("horizon")
            if h in by_horizon:
                by_horizon[h].append(r2)

    def _fmt_pool(h: str) -> str:
        items = by_horizon.get(h, [])
        if not items: return f"({h}: 비어있음)"
        lines = [f"{h} 총 {len(items)}종목:"]
        for r in items[:8]:
            lines.append(
                f"  · {r.get('ticker','?'):8} cat={r.get('_category','?'):13} "
                f"sec={(r.get('sector','') or '')[:15]:15} "
                f"comp={r.get('composite','?')} cons={r.get('consensus','?')}"
            )
        return "\n".join(lines)

    pool_lines = []
    for r in all_stocks[:40]:
        pool_lines.append(
            f"  - {r.get('ticker','?'):8} cat={r.get('_category','?'):13} "
            f"sec={(r.get('sector','') or '')[:15]:15} comp={r.get('composite','?')} "
            f"cons={r.get('consensus','?')} h={r.get('horizon','?')}"
        )
    pool_str = "\n".join(pool_lines)

    return f"""당신은 시니어 포트폴리오 전략가입니다. 매수 Final List의 **📈 개별 Stock** 종목들에 대한 deep-dive를 **정확히 한국어 5,500자 (±200자)** 으로 작성하세요.

ETF는 별도 commentary로 분석되므로 이 commentary에서는 **개별 종목만** 다룹니다 — 섹터 ETF, 지역 ETF, 팩터 ETF 등은 절대 언급 금지.

이 commentary는 별도 생성되는 "공통 거시 commentary"와 함께 표시됩니다. 따라서 macro 분석/과거 유사 구간/시나리오는 공통 commentary 참조 가능하며, 여기서는 **stock-specific 영역**에 집중하세요:
- 실적 시즌 + 분기 catalyst
- 펀더멘털 (margin / ROIC / FCF / revision)
- 밸류에이션 (P/E, P/S, EV/EBITDA)
- 개별 stock의 sector positioning
- entry/stop level (가능한 한 구체)

═══════════════════════════════════════════════════════════════════
## 📈 Stock-only 카테고리 분포
═══════════════════════════════════════════════════════════════════
{category_summary}

═══════════════════════════════════════════════════════════════════
## 호라이즌별 Stock 풀
═══════════════════════════════════════════════════════════════════
[TACTICAL 5d]
{_fmt_pool('tactical')}

[CORE 21d]
{_fmt_pool('core')}

[STRATEGIC 63d]
{_fmt_pool('strategic')}

═══════════════════════════════════════════════════════════════════
## 전체 Stock 풀 (Top-3 선정용)
═══════════════════════════════════════════════════════════════════
{pool_str}

{f"## 시장 컨텍스트{chr(10)}{market_context[:400]}" if market_context else ""}

═══════════════════════════════════════════════════════════════════
## ⚠ FACT BASE — Phase 1 Agent (Stock 분석 시 인용)
═══════════════════════════════════════════════════════════════════
{phase1_briefing if phase1_briefing else "(N/A)"}

═══════════════════════════════════════════════════════════════════
## 작성 가이드 — 정확히 다음 7개 섹션 (총 5,500자)
═══════════════════════════════════════════════════════════════════

[섹션 S1: 📈 Stock 분포 진단 — 약 600자]
- Stock의 카테고리별 분포 의미 (ENTERED/NEW/HOLDING/EXIT_PENDING)
- Sector tilt (Tech/Financials/Industrials/etc.) 분석
- Stock 평균 risk score, horizon 균형, conviction (UNANIMOUS/MAJORITY) 분포
- Pure stock portfolio의 risk posture

[섹션 S2: ✓ ENTERED Stock 심층 분석 — 약 700자]
ENTERED 카테고리 Stock 종목별 진입 근거 + 실적 캘린더 + 펀더멘털 강점
실적 시즌 임박도 + 가이던스 예상 + position sizing

[섹션 S3: 🟢 NEW Stock 심층 분석 — 약 700자]
NEW 카테고리 Stock의 emerging theme + 펀더멘털 변화 시그널
밸류에이션 + earnings revision trend + 진입 timing

[섹션 S4: 🔵 HOLDING + ⚠ EXIT_PENDING Stock — 약 600자]
HOLDING Stock의 평균 보유 일수, profit-taking 검토
EXIT_PENDING Stock의 청산 우선순위 (실적 미스, 가이던스 하향, 펀더멘털 악화)

[섹션 S5: ⚡ Stock 호라이즌별 (Tactical / Core / Strategic) — 약 900자]
- Tactical Stock (5d): 단기 catalyst + 기술적 entry/stop
- Core Stock (21d): 1개월 실적 cycle + 펀더멘털 thesis
- Strategic Stock (63d): 분기 실적 + 산업 cycle 위치

[섹션 S6: 🌟 Stock Top-3 유심히 보아야 할 종목 — 약 1,500자]
Stock 전용 Top-3 (ETF는 제외) — 약 500자씩:
- 🥇 1순위: 티커 (카테고리, sector, 호라이즌) — 펀더멘털 강점, 실적 catalyst, 밸류에이션, key risk, watch-out, 예상 수익률 시나리오, **공통 commentary §B 과거 사례 1/2 중 어디서 강한 지지**
- 🥈 2순위: 동일 구조
- 🥉 3순위: 동일 구조

[섹션 S7: 💼 Stock 액션 플랜 — 약 500자]
- 실적 시즌 캘린더 우선순위 (1주 / 1개월 / 3개월)
- 펀더멘털 모니터링 포인트 (margin/ROIC/revision)
- Stock-specific stop-loss + 진입가 권고
- 공통 시나리오의 Stock 적용 방식

═══════════════════════════════════════════════════════════════════
## 출력 요구사항
═══════════════════════════════════════════════════════════════════
- 정확히 한국어 5,500자 (±200자)
- 모든 7개 섹션 포함 (S1-S7)
- 각 섹션 시작에 ## 헤더
- **개별 stock만 다룸 — ETF 절대 금지** (XLE, IWD, DXJ 등 ETF ticker 언급 금지)
- ⚠ ANTI-HALLUCINATION: FACT BASE에 없는 macro 주장 금지

JSON으로 반환:
```json
{{"commentary": "전체 한국어 5,500자 Stock commentary..."}}
```

JSON 외 다른 텍스트 출력 금지."""


def _etf_prompt(etfs_by_cat: dict, market_context: str = "") -> str:
    """ETF deep-dive — 5,500자. 섹터/region/style 중심."""
    phase1_briefing = _build_phase1_briefing()
    category_summary = _summarize_categories_for_unified(etfs_by_cat)

    by_horizon: dict[str, list[dict]] = {"tactical": [], "core": [], "strategic": []}
    all_etfs: list[dict] = []
    for cat in ("ENTERED", "NEW", "HOLDING", "EXIT_PENDING"):
        for r in etfs_by_cat.get(cat) or []:
            r2 = {**r, "_category": cat}
            all_etfs.append(r2)
            h = r.get("horizon")
            if h in by_horizon:
                by_horizon[h].append(r2)

    def _fmt_pool(h: str) -> str:
        items = by_horizon.get(h, [])
        if not items: return f"({h}: 비어있음)"
        lines = [f"{h} 총 {len(items)}종목:"]
        for r in items[:8]:
            lines.append(
                f"  · {r.get('ticker','?'):8} cat={r.get('_category','?'):13} "
                f"sec={(r.get('sector','') or '')[:15]:15} "
                f"comp={r.get('composite','?')} cons={r.get('consensus','?')}"
            )
        return "\n".join(lines)

    pool_lines = []
    for r in all_etfs[:40]:
        pool_lines.append(
            f"  - {r.get('ticker','?'):8} cat={r.get('_category','?'):13} "
            f"sec={(r.get('sector','') or '')[:15]:15} comp={r.get('composite','?')} "
            f"cons={r.get('consensus','?')} h={r.get('horizon','?')}"
        )
    pool_str = "\n".join(pool_lines)

    return f"""당신은 시니어 포트폴리오 전략가입니다. 매수 Final List의 **📦 ETF** 종목들에 대한 deep-dive를 **정확히 한국어 5,500자 (±200자)** 으로 작성하세요.

개별 stock은 별도 commentary로 분석되므로 이 commentary에서는 **ETF만** 다룹니다 — 개별 stock ticker 절대 언급 금지.

이 commentary는 별도 생성되는 "공통 거시 commentary"와 함께 표시됩니다. 따라서 macro 분석/과거 유사 구간/시나리오는 공통 commentary 참조 가능하며, 여기서는 **ETF-specific 영역**에 집중하세요:
- 섹터 로테이션 (어떤 섹터 ETF로 표현)
- Region tilt (US/Japan/Europe/EM)
- Factor/Style tilt (Value/Growth/Quality/Low-Vol/Momentum)
- ETF 구성 (top holdings, concentration, expense ratio 추정)
- 분산 효과 (ETF가 보유 stock에 비해 가지는 advantage)

═══════════════════════════════════════════════════════════════════
## 📦 ETF-only 카테고리 분포
═══════════════════════════════════════════════════════════════════
{category_summary}

═══════════════════════════════════════════════════════════════════
## 호라이즌별 ETF 풀
═══════════════════════════════════════════════════════════════════
[TACTICAL 5d]
{_fmt_pool('tactical')}

[CORE 21d]
{_fmt_pool('core')}

[STRATEGIC 63d]
{_fmt_pool('strategic')}

═══════════════════════════════════════════════════════════════════
## 전체 ETF 풀 (Top-3 선정용)
═══════════════════════════════════════════════════════════════════
{pool_str}

{f"## 시장 컨텍스트{chr(10)}{market_context[:400]}" if market_context else ""}

═══════════════════════════════════════════════════════════════════
## ⚠ FACT BASE — Phase 1 Agent (ETF 분석 시 인용)
═══════════════════════════════════════════════════════════════════
{phase1_briefing if phase1_briefing else "(N/A)"}

═══════════════════════════════════════════════════════════════════
## 작성 가이드 — 정확히 다음 7개 섹션 (총 5,500자)
═══════════════════════════════════════════════════════════════════

[섹션 E1: 📦 ETF 분포 진단 — 약 600자]
- ETF의 카테고리별 분포 의미
- Sector ETF / Region ETF / Factor ETF 비중
- ETF 평균 risk score, horizon 균형
- ETF portfolio가 표현하는 macro view (cyclical / defensive / barbell)

[섹션 E2: ✓ ENTERED ETF 심층 분석 — 약 700자]
ENTERED 카테고리 ETF 종목별 진입 근거 + 섹터/region/style positioning
공통 commentary §A 거시 환경과의 정합성

[섹션 E3: 🟢 NEW ETF 심층 분석 — 약 700자]
NEW 카테고리 ETF의 emerging theme + sector rotation 단서
ETF 구성 분석 (top holdings concentration, region tilt, factor exposure)

[섹션 E4: 🔵 HOLDING + ⚠ EXIT_PENDING ETF — 약 600자]
HOLDING ETF의 평균 보유 일수, 섹터 비중 조절
EXIT_PENDING ETF의 청산 우선순위 (섹터 약세, region 약세, factor reversal)

[섹션 E5: 🎯 ETF 호라이즌별 (Tactical / Core / Strategic) — 약 900자]
- Tactical ETF (5d): 단기 sector rotation + factor 회전
- Core ETF (21d): 1개월 sector cycle + region/style 비중
- Strategic ETF (63d): 분기 매크로 cycle + secular trend

[섹션 E6: 🌟 ETF Top-3 유심히 보아야 할 종목 — 약 1,500자]
ETF 전용 Top-3 (개별 stock 제외) — 약 500자씩:
- 🥇 1순위: 티커 (카테고리, sector/region/factor, 호라이즌) — 구성 분석, top holdings 강점, sector/factor catalyst, key risk (concentration/region/factor reversal), watch-out, **공통 commentary §B 과거 사례 1/2 중 어디서 강한 지지**
- 🥈 2순위: 동일 구조
- 🥉 3순위: 동일 구조

[섹션 E7: 💼 ETF 액션 플랜 — 약 500자]
- 섹터 ETF 비중 가이드
- Region tilt (US/Japan/Europe/EM) 권고
- Factor 비중 (Value/Growth/Quality 등)
- 공통 시나리오의 ETF 적용 방식

═══════════════════════════════════════════════════════════════════
## 출력 요구사항
═══════════════════════════════════════════════════════════════════
- 정확히 한국어 5,500자 (±200자)
- 모든 7개 섹션 포함 (E1-E7)
- 각 섹션 시작에 ## 헤더
- **ETF만 다룸 — 개별 stock ticker 절대 금지** (KEYS, MS, ABNB 등 stock ticker 언급 금지)
- ⚠ ANTI-HALLUCINATION: FACT BASE에 없는 macro 주장 금지

JSON으로 반환:
```json
{{"commentary": "전체 한국어 5,500자 ETF commentary..."}}
```

JSON 외 다른 텍스트 출력 금지."""


def build_executive_commentary(buy_list: list[dict], sell_list: list[dict],
                                market_context: str = "",
                                items_by_category: Optional[dict] = None,
                                swarm_generated_at: str = "",
                                cache_only: bool = False) -> dict:
    """Generate UNIFIED 5000-char executive commentary covering all categories.

    NEW (2026-06): single 5000-char unified commentary (covers ENTERED/NEW/HOLDING/EXIT_PENDING
    + Top-3 watchlist) instead of separate per-category 1000-char commentaries.

    Args:
        buy_list: full buy list (kept for backward compat / hashing)
        sell_list: full sell list (deprecated, ignored)
        market_context: PM commentary string
        items_by_category: dict of {"ENTERED": [...], "NEW": [...], "HOLDING": [...], "EXIT_PENDING": [...]}
                            REQUIRED for unified prompt; if None, falls back to legacy buy/sell prompts.

    Returns:
        {
          "unified_commentary": str (~5000 chars 한국어),
          "unified_cached": bool,
          "buy_commentary":  str (legacy, kept for backward compat = same as unified),
          "sell_commentary": "",
          "buy_cached":  bool (= unified_cached),
          "sell_cached": False,
          "generated_at": ISO timestamp,
        }
    """
    cache = _load_cache()
    out = {
        "unified_commentary": "",
        "unified_cached": False,
        "buy_commentary": "",
        "sell_commentary": "",
        "buy_cached": False,
        "sell_cached": False,
        "generated_at": _now_ts(),
    }

    if items_by_category is None:
        # Legacy path: just use buy_list / sell_list separately
        out["buy_commentary"] = _fallback_buy_commentary(buy_list) if buy_list else ""
        out["unified_commentary"] = out["buy_commentary"]
        _save_cache(cache)
        return out

    # Build a stable content hash spanning ALL categories.
    # Include swarm_generated_at so every fresh swarm forces commentary regen
    # (even if ticker list happens to coincide — narrative may have shifted).
    flat_items = []
    for cat in ("ENTERED", "NEW", "HOLDING", "EXIT_PENDING"):
        flat_items.extend(items_by_category.get(cat) or [])
    base_hash = _content_hash(flat_items)
    unified_hash = hashlib.md5(f"{base_hash}|{swarm_generated_at}".encode()).hexdigest()[:16]

    unified_entry = cache.get("unified", {})
    cache_fresh = (unified_entry.get("content_hash") == unified_hash and _cache_valid(unified_entry))
    if cache_fresh:
        out["unified_commentary"] = unified_entry.get("commentary", "")
        out["unified_cached"] = True
    elif cache_only:
        # API-path: return stale cache (if any) immediately + flag for background regen.
        # Avoids blocking /api/final-list for 6-12 min on the 12,000-char LLM call.
        stale = unified_entry.get("commentary", "")
        if stale:
            out["unified_commentary"] = stale
            out["unified_cached"] = True
            out["stale"] = True   # ← frontend can show "regenerating…" badge
        else:
            out["unified_commentary"] = (
                "📝 (Commentary is being generated in the background — this typically takes 3-5 minutes. "
                "Refresh the page in a few minutes to see the full 12,000자 analysis.)"
            )
            out["unified_cached"] = False
            out["pending"] = True
        # Trigger background regeneration so the next /api/final-list call hits cache
        import threading as _t
        def _bg_regen():
            try:
                prompt = _unified_prompt(items_by_category, market_context)
                text = _generate_commentary(prompt, timeout=600)
                if text:
                    cur = _load_cache()
                    cur["unified"] = {
                        "content_hash": unified_hash,
                        "commentary":   text,
                        "cached_at":    _now_ts(),
                        "n_categories": sum(1 for k in items_by_category if items_by_category[k]),
                        "n_total":      len(flat_items),
                    }
                    _save_cache(cur)
            except Exception:
                pass
        _t.Thread(target=_bg_regen, daemon=True).start()
    else:
        prompt = _unified_prompt(items_by_category, market_context)
        # 12,000-char output (11 sections incl. Historical Analog) → 900s = 15 min
        text = _generate_commentary(prompt, timeout=900)
        if text:
            out["unified_commentary"] = text
            cache["unified"] = {
                "content_hash": unified_hash,
                "commentary":   text,
                "cached_at":    _now_ts(),
                "n_categories": sum(1 for k in items_by_category if items_by_category[k]),
                "n_total":      len(flat_items),
            }
        else:
            out["unified_commentary"] = _fallback_buy_commentary(buy_list)

    # Mirror to buy_commentary for backward compat (frontend may still read either field)
    out["buy_commentary"] = out["unified_commentary"]
    out["buy_cached"]     = out["unified_cached"]

    # ── Option B: Split commentaries (Common / Stock / ETF) ──
    # Run in parallel as 3 separate LLM calls. Each cached independently.
    stocks_by_cat, etfs_by_cat = _split_categories_by_asset(items_by_category)

    s_flat = []
    e_flat = []
    for cat in ("ENTERED", "NEW", "HOLDING", "EXIT_PENDING"):
        s_flat.extend(stocks_by_cat.get(cat) or [])
        e_flat.extend(etfs_by_cat.get(cat) or [])

    common_hash = hashlib.md5(f"common|{base_hash}|{swarm_generated_at}".encode()).hexdigest()[:16]
    stock_hash  = hashlib.md5(f"stock|{_content_hash(s_flat)}|{swarm_generated_at}".encode()).hexdigest()[:16]
    etf_hash    = hashlib.md5(f"etf|{_content_hash(e_flat)}|{swarm_generated_at}".encode()).hexdigest()[:16]

    def _load_or_gen(key: str, hsh: str, prompt_builder, prompt_args: tuple,
                     timeout: int, cache_only_local: bool) -> tuple[str, bool, bool, bool]:
        """Returns (text, cached, stale, pending)."""
        entry = cache.get(key, {})
        fresh = (entry.get("content_hash") == hsh and _cache_valid(entry))
        if fresh:
            return entry.get("commentary", ""), True, False, False
        if cache_only_local:
            stale_txt = entry.get("commentary", "")
            if stale_txt:
                return stale_txt, True, True, False  # stale=True
            return ("📝 (Background generation in progress.)", False, False, True)  # pending=True
        # Live generation
        prompt = prompt_builder(*prompt_args)
        text = _generate_commentary(prompt, timeout=timeout)
        if text:
            cache[key] = {
                "content_hash": hsh,
                "commentary": text,
                "cached_at": _now_ts(),
            }
            return text, False, False, False
        return ("", False, False, False)

    if cache_only:
        # Trigger background regen for all 3 split commentaries (in parallel)
        from concurrent.futures import ThreadPoolExecutor

        def _bg_split_regen():
            try:
                with ThreadPoolExecutor(max_workers=3) as ex:
                    futs = {}
                    if cache.get("common_macro", {}).get("content_hash") != common_hash:
                        futs[ex.submit(_generate_commentary,
                                        _common_macro_prompt(items_by_category, market_context),
                                        300)] = "common_macro"
                    if cache.get("stock_split", {}).get("content_hash") != stock_hash:
                        futs[ex.submit(_generate_commentary,
                                        _stock_prompt(stocks_by_cat, market_context),
                                        600)] = "stock_split"
                    if cache.get("etf_split", {}).get("content_hash") != etf_hash:
                        futs[ex.submit(_generate_commentary,
                                        _etf_prompt(etfs_by_cat, market_context),
                                        600)] = "etf_split"
                    for fut, key in [(f, k) for f, k in futs.items()]:
                        try:
                            text = fut.result()
                            if text:
                                cur = _load_cache()
                                cur[key] = {
                                    "content_hash": (common_hash if key == "common_macro"
                                                     else stock_hash if key == "stock_split"
                                                     else etf_hash),
                                    "commentary": text,
                                    "cached_at": _now_ts(),
                                }
                                _save_cache(cur)
                        except Exception:
                            pass
            except Exception:
                pass
        import threading as _t2
        _t2.Thread(target=_bg_split_regen, daemon=True).start()

        # Return cached/stale/pending for each
        for key, hsh in [("common_macro", common_hash), ("stock_split", stock_hash), ("etf_split", etf_hash)]:
            entry = cache.get(key, {})
            fresh = (entry.get("content_hash") == hsh and _cache_valid(entry))
            txt = entry.get("commentary", "")
            if fresh:
                out[key] = txt
                out[f"{key}_cached"] = True
            elif txt:
                out[key] = txt
                out[f"{key}_cached"] = True
                out[f"{key}_stale"] = True
            else:
                out[key] = "📝 (생성 중 — 3-8분 후 새로고침)"
                out[f"{key}_cached"] = False
                out[f"{key}_pending"] = True
    else:
        # Synchronous generation: run all 3 in parallel
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as ex:
            f_common = ex.submit(_load_or_gen, "common_macro", common_hash,
                                  _common_macro_prompt, (items_by_category, market_context),
                                  300, False)
            f_stock  = ex.submit(_load_or_gen, "stock_split", stock_hash,
                                  _stock_prompt, (stocks_by_cat, market_context),
                                  600, False)
            f_etf    = ex.submit(_load_or_gen, "etf_split", etf_hash,
                                  _etf_prompt, (etfs_by_cat, market_context),
                                  600, False)
            c_txt, c_cached, _, _ = f_common.result()
            s_txt, s_cached, _, _ = f_stock.result()
            e_txt, e_cached, _, _ = f_etf.result()
        out["common_macro"] = c_txt
        out["common_macro_cached"] = c_cached
        out["stock_split"] = s_txt
        out["stock_split_cached"] = s_cached
        out["etf_split"] = e_txt
        out["etf_split_cached"] = e_cached

    _save_cache(cache)
    return out


# ─── Fallback deterministic commentary (when LLM fails) ──────────────

def _fallback_buy_commentary(buy_list: list[dict]) -> str:
    """Template-based fallback if LLM unavailable."""
    from collections import Counter
    n = len(buy_list)
    if n == 0:
        return "현재 3-Agent 합의된 매수 후보가 없습니다. 시장이 entry 대기 모드인 것으로 보입니다."
    cons = Counter(r.get("consensus") for r in buy_list)
    sectors = Counter(r.get("sector") for r in buy_list)
    top_sec = sectors.most_common(3)
    unanimous = [r for r in buy_list if r.get("consensus") == "UNANIMOUS"][:5]
    avg_risk = sum(r.get("risk_score", 0) for r in buy_list) / max(1, n)
    tickers_str = ", ".join(r["ticker"] for r in unanimous)
    return (
        f"이번 주기에서 3-Agent Voting(PM Agent + Trading Agent + Risk Manager) 합의를 통과한 매수 후보는 총 {n}개입니다. "
        f"이 중 UNANIMOUS({cons.get('UNANIMOUS',0)}개) 등급은 세 agent가 모두 APPROVE를 부여한 최고 신뢰도 종목으로, "
        f"{tickers_str} 가 대표 종목입니다. 섹터 분포는 {top_sec[0][0] if top_sec else '?'}({top_sec[0][1] if top_sec else 0}), "
        f"{top_sec[1][0] if len(top_sec)>1 else ''}({top_sec[1][1] if len(top_sec)>1 else 0}), "
        f"{top_sec[2][0] if len(top_sec)>2 else ''}({top_sec[2][1] if len(top_sec)>2 else 0})로 집중되어 있어, "
        f"현재 시장 리더십이 명확함을 시사합니다. 평균 Risk Score는 {avg_risk:.1f}/100으로 비교적 안전한 구간이며, "
        f"Risk Manager가 분석한 과열, 변동성, 유동성, 섹터 집중도 측면에서 양호한 후보군입니다. "
        f"향후 1-4주 동안 UNANIMOUS 등급부터 우선 진입하고, MAJORITY_CLEAN 등급은 timing 검증 후 추가 매수를 권장합니다. "
        f"실행 시 entry zone 활용 + Phase 5.6 state machine의 2일 confirmation을 거친 후 ENTERED 상태로 전환된 종목 우선."
    )


def _fallback_sell_commentary(sell_list: list[dict]) -> str:
    """Template-based fallback for sell list."""
    from collections import Counter
    n = len(sell_list)
    if n == 0:
        return "현재 3-Agent 합의된 매도 후보가 없습니다."
    cons = Counter(r.get("consensus") for r in sell_list)
    sectors = Counter(r.get("sector") for r in sell_list)
    top_sec = sectors.most_common(3)
    unanimous = [r for r in sell_list if r.get("consensus") == "UNANIMOUS"][:5]
    avg_risk = sum(r.get("risk_score", 0) for r in sell_list) / max(1, n)
    tickers_str = ", ".join(r["ticker"] for r in unanimous)
    return (
        f"3-Agent Voting을 통과한 매도/공매도 후보는 총 {n}개이며, UNANIMOUS({cons.get('UNANIMOUS',0)})는 약세 신호가 가장 명확한 종목입니다. "
        f"대표 종목: {tickers_str}. PM Agent는 composite 하락 및 약세 classification(DOWNTREND, WEAKENING 등)을 근거로, "
        f"Trading Agent는 SHORT BUY_NOW 신호를 통해 즉시 open short 가능 시점을 검증했습니다. "
        f"섹터 분포는 {top_sec[0][0] if top_sec else '?'}({top_sec[0][1] if top_sec else 0}) 등에 집중되어 있으며, "
        f"이는 해당 섹터의 매크로 약세 또는 theme rotation을 시사합니다. 평균 Risk Score는 {avg_risk:.1f}/100. "
        f"숏 포지션의 주요 리스크는 squeeze risk와 liquidity로, 진입 시 stop-loss(예: SMA20 상회 시 cover) 설정 필수입니다. "
        f"포트폴리오 hedge 비중은 long 대비 30-50% 이내로 제한 권장하며, "
        f"향후 1-4주 동안 약세 가속화 시 추가 short 확대, 반등 시 부분 cover로 lock-in 전략을 권장합니다."
    )
