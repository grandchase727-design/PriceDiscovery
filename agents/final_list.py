# -*- coding: utf-8 -*-
"""final_list.py — Buy/Sell Final List synthesizer.

Combines 4 signal sources through a confidence-tiering pipeline:
  1. PM Swarm picks (Phase 5)
  2. Trading Signal (Phase 5.5)
  3. Position State (Phase 5.6 — hysteresis filter)
  4. Backtest cross-checks (proxy 6/12 + historical alpha leaders)

Produces two final lists:
  - 🟢 BUY  : LONG (stocks + ETFs) candidates with positive technical alignment
  - 🔴 SELL : SHORT (stocks + ETFs) candidates + LONG positions to exit
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _safe_load(path: str, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


# Cached price data (loaded once per process for performance)
_PRICE_CACHE: dict = {}
_PRICE_CACHE_LOADED = False

def _load_price_cache() -> dict:
    """Load backtest price cache once; reused across calls."""
    global _PRICE_CACHE, _PRICE_CACHE_LOADED
    if _PRICE_CACHE_LOADED:
        return _PRICE_CACHE
    try:
        import pickle
        cache_path = Path(".backtest_price_cache.pkl")
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
                _PRICE_CACHE = cache.get("data", {})
    except Exception:
        _PRICE_CACHE = {}
    _PRICE_CACHE_LOADED = True
    return _PRICE_CACHE


# Trailing return horizons in trading days
TRAILING_HORIZONS = {
    "ret_5d":  5,    # 1 week
    "ret_1mo": 21,   # ~1 month
    "ret_3mo": 63,   # ~3 months
    "ret_6mo": 126,  # ~6 months
    "ret_1y":  252,  # ~1 year
}


def _infer_bucket(scan_row: dict, side: str = "long") -> str:
    """Infer bucket name (long_stocks / long_etfs / short_*) from scan record category.

    Used when a position's PM-pick bucket is unavailable (e.g. HOLDING position
    not in today's PM picks, or EXIT_PENDING entry built from position_state only).

    Convention from price_discovery.py universe loading:
      - Stock tickers: category starts with "STK_" (STK_Financials, STK_Tech, etc.)
      - ETF tickers: EVERYTHING ELSE — EQ_, FI_ (FICC 채권), Commodities_, Currency_,
        Korea_, Emerging_, Intl_, Real_, Multi_ 등. (FICC/국제 ETF가 stock으로
        오분류되던 버그 수정: EQ_/ETF_ 만 보던 것 → "STK_ 아니면 ETF"로 통일)

    Args:
        scan_row: row from .scan_cache.pkl results (must have 'category' key)
        side: "long" or "short"

    Returns: one of "long_stocks", "long_etfs", "short_stocks", "short_etfs".
             Falls back to "long_stocks" if classification ambiguous.
    """
    cat = (scan_row.get("category") or "").upper()
    is_etf = bool(cat) and not cat.startswith("STK_")
    if not cat:
        # Last-resort heuristic — ticker length / format
        t = (scan_row.get("ticker") or "").upper()
        if t.endswith(".KS") or t.endswith(".KQ"):
            # Korean: 6-digit numeric → ETF by convention here (universe-loaded)
            t_base = t.split(".")[0]
            is_etf = t_base.isdigit() and len(t_base) == 6
    prefix = "long" if side == "long" else "short"
    return f"{prefix}_{'etfs' if is_etf else 'stocks'}"


import re as _re

# ══════════════════════════════════════════════════════════════════════════
#  _unified_commentary() — plain-explainer 서술 문단 조립 (결정론적, LLM 불필요)
#
#  기존엔 debate_transcript(기계 헤더 "[R1] Trading=X|Risk=Y|Critic=Z" + 세 근거
#  요약) + "📊 PM:" + "🎯 Trading:" + "🛡 Risk:" 를 단순 이어붙여 같은 rationale이
#  [R1]줄과 라벨줄에 두 번씩 노출되는 중복을 낳았다. 이제는 사전 조립된
#  debate_transcript를 입력에서 완전히 배제하고, 원본 4 rationale(PM/Trading/Risk/
#  Critic)을 각 1회씩만 써서 하나의 매끄러운 한국어 문단을 만든다.
#
#  슬롯별 고정 전환구가 조각 간 '접착제':
#    PM(무리드 도입) → Trading(뒤 ENTRY_LEAD) → Risk(앞 RISK_LEAD)
#    → Critic(앞 '비판적으로는') → 결론(tier 딕셔너리)
# ══════════════════════════════════════════════════════════════════════════

# trading_signal → 진입 타이밍 전환구 (Trading 본문 '뒤'에 붙는 결론 연결어)
_ENTRY_LEAD = {
    "BUY_NOW": "지금이 곧바로 담을 만한 자리라, 진입 타이밍은 '즉시 매수'로 본다.",
    "WAIT":    "그래서 곧바로 담기보다 확인이 설 때까지 '관망 대기'가 맞다.",
    "SKIP":    "그래서 진입 타이밍은 곧바로 '제외'다.",
    "":        "",
}
# HOLDING 보유 재평가 + WAIT/SKIP 일 때는 (a)에서 이미 재점검 프레임을 깔았으므로
# 진입 리드를 '보류/제외 신규진입' 톤으로 약화
_ENTRY_LEAD_HOLD = {
    "WAIT": "따라서 추가 매수는 보류하고 기존 사이즈만 유지한다.",
    "SKIP": "따라서 신규 진입은 제외한다.",
}
# risk 조각 앞 전환구 (signal 톤 연동)
_RISK_LEAD = {
    "BUY_NOW": "리스크 쪽에서는 ", "WAIT": "리스크 측면에서 ",
    "SKIP": "리스크 측면에서도 ", "": "리스크를 보면 ",
}
_CRITIC_LEAD = "비판적으로는 "
# tier → 마지막 합의 결론 문장 (per_ticker_debate.OUTCOME/KEY 자연화)
_TIER_CONCLUSION = {
    "UNANIMOUS":        "세 에이전트가 모두 같은 결론에 도달한 만장일치 매수 신호다.",
    "MAJORITY_CLEAN":   "세 에이전트 중 다수가 뜻을 모은 강한 합의 매수 신호다.",
    "MAJORITY_DISSENT": "일부 이견은 있으나 다수가 매수에 동의한 신호다.",
    "SOLO_CLEAN":       "단독 승인 신호로, 진입 타이밍만 기다리는 단계다.",
    "SOLO_DISSENT":     "단독 승인이나 리스크 우려가 남아 소규모 진입을 고려할 단계다.",
    "SOLO":             "단독 신호에 그쳐 지금은 지켜보는 단계다.",
    "ALL_CAUTION":      "세 에이전트가 모두 신중론을 편 만큼 지금은 지켜보는 단계다.",
    "EXCLUDED":         "종합적으로 매수 후보에서 제외한다.",
    "REJECTED":         "종합적으로 매수 후보에서 제외한다.",
}
_UNANIMOUS_NODISSENT = "이견 없이 세 에이전트가 뜻을 모은 만장일치 매수 신호다."


def _is_none_critic(c: str) -> bool:
    """critic sentinel 판별 (변형·부분일치까지: 빈값/none/thesis holds/이견 없/—/없음)."""
    c = (c or "").strip().lower()
    return (c == "" or c.startswith("none") or "thesis holds" in c
            or "이견 없" in c or c in ("—", "-", "없음"))


def _critic_from_transcript(transcript: str) -> str:
    """레거시 캐시 폴백 — debate_synthesis에 구조화 critic_challenge가 없을 때
    사전 조립 transcript의 'Critic: ... .' / 'Critic: ... →' 구간에서 도전 문구만
    추출한다. 구조화 필드가 우선이며, 이 helper는 임시 안전망일 뿐이다.
    """
    if not transcript:
        return ""
    m = _re.search(r"Critic:\s*(.*)$", transcript, flags=_re.S)
    if not m:
        return ""
    tail = m.group(1).strip()
    # 결론(OUTCOME)은 항상 마지막 '→ ...'로 도입된다. critic 본문 자체가 내부에
    # '→'를 포함할 수 있으므로(예: KRE의 'AI 이탈 자금 → 지역은행 유입') 첫 화살표가
    # 아니라 **마지막 화살표**에서만 잘라 결론 구절을 제거한다.
    tail = _re.sub(r"\s*→[^→]*$", "", tail).strip()
    # 'challenge. → outcome' 형태에서 남는 후행 마침표 정리
    return tail.rstrip(". ").strip()


# ── 2차 안전망: cross-rationale fact-token dedup ─────────────────────────────
# 구조적 단일사용으로 [R1]↔🎯 중복은 이미 소거됨. 여기서는 '서로 다른 rationale이
# 같은 사실(예: DE OER 24)을 각각 언급'하는 나머지 케이스만 절 단위로 제거한다.
_FACT_PATTERNS = [
    r"OER\s*\d+", r"Composite\s*[\d.]+", r"OVEREXTENDED", r"NEUTRAL",
    r"Fitch\s*A\+?", r"Moody'?s\s*A1", r"S&P\s*A\b", r"신용등급",
    r"스트레스\s*테스트|stress\s*test", r"HYG", r"\bCRE\b", r"무보험\s*예금",
]


def _fact_sigs(s: str) -> set:
    out = set()
    for p in _FACT_PATTERNS:
        for m in _re.findall(p, s, flags=_re.I):
            out.add(m.strip().lower())
    return out


def _dedup_clauses(chunks: list) -> list:
    """앞 chunk에 이미 나온 fact-sig가 뒤 chunk의 '한 절'에서 재등장하고 그 절이
    저정보(부분집합)면 그 절만 삭제. 절 경계 = 쉼표/마침표/연결어미(며/고)."""
    seen: set = set()
    out: list = []
    LOW_INFO = 45   # 이보다 짧은 절만 순수 반복으로 간주(과잉 삭제 방지)
    for ch in chunks:
        clauses = _re.split(r"(?<=[,.。])\s+|(?<=며)\s+|(?<=고)\s+", ch)
        kept = []
        for cl in clauses:
            sig = _fact_sigs(cl)
            if sig and sig <= seen and len(cl) < LOW_INFO:
                continue                      # 순수 반복 절 → drop
            seen |= sig
            kept.append(cl)
        out.append(" ".join(x for x in kept if x).strip())
    return [c for c in out if c]


def _unified_commentary(tier: str = "", key_factor: str = "", final_decision: str = "",
                        trading_signal: str = "", pm_reason: str = "",
                        trading_reason: str = "", risk_reason: str = "",
                        critic_challenge: str = "") -> str:
    """PM 선정근거·진입 타이밍·리스크·비판을 하나의 매끄러운 한국어 서술 문단으로
    엮는다. 기계 헤더("[R1] Trading=…")·이모지 라벨 블록을 절대 만들지 않고,
    각 원본 rationale을 1회씩만 사용해 중복을 구조적으로 차단한다.

    흐름: 선정근거(PM) → 진입타이밍(Trading+ENTRY_LEAD) → 리스크(RISK_LEAD+Risk)
          → 비판(critic; sentinel이면 생략) → 합의결론(tier 딕셔너리).
    """
    def norm(s): return (s or "").strip()

    def cap(s):                                   # 문장 종결 보정
        s = norm(s)
        if s and s[-1] not in ".。!?…":
            s += "."
        return s

    pm   = norm(pm_reason)
    trd  = norm(trading_reason)
    rsk  = norm(risk_reason)
    crit = norm(critic_challenge)
    sig  = norm(trading_signal).upper()
    tier = norm(tier).upper() or "SOLO"
    # final_decision이 EXCLUDE면 tier가 무엇이든 제외 계열로 강제 (JCI 안전망)
    if norm(final_decision).upper().startswith("EXCLUDE"):
        tier = "EXCLUDED"
        sig = "SKIP"

    # HOLDING 보유 재평가 감지
    is_hold = pm.startswith("보유 재평가") or "HOLDING" in pm

    chunks: list = []

    # (a) 선정근거 = PM. HOLDING이면 "이미 보유 중인 종목…" 자연문으로 치환.
    if pm:
        if is_hold:
            pm = (pm.replace("보유 재평가 — HOLDING", "이미 보유 중인 종목")
                    .replace("보유 재평가 —", "이미 보유 중인 종목")
                    .replace("진입 논지 유효성 재검토 대상",
                             "진입 논지가 여전히 유효한지 재점검하는 대상이다")
                    .replace("진입 논지 재검토 대상",
                             "진입 논지가 여전히 유효한지 재점검하는 대상이다"))
        chunks.append(cap(pm))
    elif key_factor:
        chunks.append(cap(f"{norm(key_factor)} 기반 후보다"))

    # (b) 진입 타이밍 = Trading 본문 + ENTRY_LEAD 전환구.
    if trd:
        chunks.append(cap(trd))                    # '왜'
        if is_hold and sig in ("WAIT", "SKIP"):
            lead = _ENTRY_LEAD_HOLD.get(sig, _ENTRY_LEAD.get(sig, ""))
        else:
            lead = _ENTRY_LEAD.get(sig, "")
        if lead:
            chunks.append(cap(lead))               # '그래서 즉시/대기/제외'
    elif sig and _ENTRY_LEAD.get(sig):
        if is_hold and sig in ("WAIT", "SKIP"):
            chunks.append(cap(_ENTRY_LEAD_HOLD.get(sig, _ENTRY_LEAD[sig])))
        else:
            chunks.append(cap(_ENTRY_LEAD[sig]))

    # (c) 리스크 = RISK_LEAD + risk_rationale (없으면 risk_key 폴백은 호출부에서 전달)
    if rsk:
        chunks.append(cap(_RISK_LEAD.get(sig, _RISK_LEAD[""]) + rsk))

    # (d) 비판 = CRITIC_LEAD + critic. sentinel이면 통째 생략.
    has_dissent = not _is_none_critic(crit)
    if has_dissent:
        chunks.append(cap(_CRITIC_LEAD + crit))

    # (e) 합의 결론 = tier 딕셔너리. UNANIMOUS+무이견은 '이견 없이~'로 자연화.
    if tier == "UNANIMOUS" and not has_dissent:
        chunks.append(_UNANIMOUS_NODISSENT)
    else:
        chunks.append(_TIER_CONCLUSION.get(tier, _TIER_CONCLUSION["SOLO"]))

    # (f) 2차 안전망 dedup(cross-rationale fact 반복 절 제거) → 단일 문단
    chunks = _dedup_clauses(chunks)
    return " ".join(c for c in chunks if c).strip() or "—"


# ══════════════════════════════════════════════════════════════════════════
#  기술적 진입·손절 슬롯 — CAN SLIM(William O'Neil) + Elliott Wave
#
#  Debate Commentary(=debate_transcript)에 "기술적 진입·손절 관점"을 한 절(slot)로
#  덧붙인다. 원 PM/Trading/Risk/Critic 서술은 그대로 두고, 그 뒤에 CAN SLIM 베이스·
#  피벗·O'Neil 손절 + Elliott 파동국면·손절선을 자연문으로 이어붙인다.
#
#  ★ 완전 결정론(LLM 미사용) — annotate_buy_list_with_entries/_with_stops가 이미
#    붙여둔 구조화 필드(entry_*/stop_*)만 읽어서 문장을 만든다. LLM이 "피벗 돌파
#    시=ENTER_NOW" 같은 없는 인과관계를 지어내던 위험(사전 샘플 검증에서 확인)을
#    구조적으로 차단한다: 여기서 서술하는 verdict/파동라벨은 모두 필드값의 직역이다.
#  ★ 데이터가 없으면(ETF 채권형처럼 베이스 미탐지, 파동 불명확) 지어내지 않고
#    "미탐지/불명확"으로 솔직히 처리한다.
# ══════════════════════════════════════════════════════════════════════════

_BASE_PATTERN_KR2 = {
    "flat_base": "Flat Base",
    "cup_with_handle": "Cup-with-Handle",
    "double_bottom": "Double Bottom",
}

# Elliott wave_guess(elliott_wave_stops._guess_wave_position enum) → 한국어 국면
_WAVE_GUESS_KR = {
    "WAVE_5_ACTIVE": "상승 5파 진행",
    "WAVE_3_TOP":    "상승 3파 고점권",
    "WAVE_3_EARLY":  "상승 3파 초입",
    "WAVE_1_NEW":    "새 상승 1파",
    "AMBIGUOUS":     "파동 불명확",
    "INSUFFICIENT_DATA": "파동 데이터 부족",
}

# entry_primary_status → verdict 직역 (인과관계 창작 금지 — status 필드의 그대로의 뜻)
_PRIMARY_STATUS_KR = {
    "actionable":       "피벗 근접 즉시 진입",
    "buy_zone":         "피벗 근접 즉시 진입",
    "await_breakout":   "피벗 돌파 대기",
    "extended":         "과열 확장 — 신규 진입 보류",
    "stale":            "베이스 저점 이탈 — 재평가",
    "elliott_fallback": "엘리엇 되돌림 진입 참고",
}


def _tech_fnum(v):
    """None-safe finite float (final_list-local; entry_price._fnum과 동일 규칙)."""
    if v is None:
        return None
    try:
        f = float(v)
        import math as _m
        return f if _m.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _tech_price(v, sym: str = "$", currency: str = "USD") -> str:
    """통화별 가격 표기 — KRW/JPY는 정수·천단위 콤마, 그 외는 소수 2자리."""
    if v is None:
        return ""
    if (currency or "").upper() in ("KRW", "JPY"):
        return f"{sym}{v:,.0f}"
    return f"{sym}{v:,.2f}"


def _technical_entry_stop_clause(row: dict) -> str:
    """annotate_*가 붙인 CAN SLIM(entry_*) + Elliott(stop_*) 구조화 필드로부터
    '기술적으로는 …' 한 절을 결정론적으로 조립한다. 데이터가 없으면 "" 반환.

    출력 예:
      "기술적으로는 CAN SLIM Cup-with-Handle 베이스(B등급)가 확인되어 즉시 진입 가능
       (O'Neil 7% 손절 $158.32, 손익비 2.86:1), 엘리엇 파동상 파동 불명확 국면으로
       $152.09(-7.0%) 손절선으로 관리한다."
    """
    row = row or {}
    sym = row.get("currency_symbol") or "$"
    cur = row.get("currency") or "USD"
    segs: list = []

    # ── CAN SLIM (William O'Neil) 서브절 ──
    pattern = row.get("entry_base_pattern")
    tier = row.get("entry_composite_tier")
    if pattern and tier not in ("SKIP", "NO_DATA"):
        pname = _BASE_PATTERN_KR2.get(pattern, pattern)
        quality = row.get("entry_base_quality")
        qtxt = f"({quality}등급)" if quality else ""
        agg  = _tech_fnum(row.get("entry_aggressive"))
        prim = _tech_fnum(row.get("entry_primary"))
        cons = _tech_fnum(row.get("entry_conservative"))
        status = row.get("entry_primary_status")
        if agg is not None:
            verdict = "즉시 진입 가능"
        elif prim is not None:
            verdict = _PRIMARY_STATUS_KR.get(status, "관망")
        elif cons is not None:
            verdict = "SMA50 보수적 진입만 유효"
        else:
            verdict = "관망"
        seg = f"CAN SLIM {pname} 베이스{qtxt}가 확인되어 {verdict}"
        oneil = _tech_fnum(row.get("entry_oneil_cut_loss"))
        rr    = _tech_fnum(row.get("entry_rr_ratio"))
        rbits = []
        if oneil is not None:
            rbits.append(f"O'Neil 7% 손절 {_tech_price(oneil, sym, cur)}")
        if rr is not None:
            rbits.append(f"손익비 {rr:.2f}:1")
        if rbits:
            seg += "(" + ", ".join(rbits) + ")"
        segs.append(seg)
    else:
        # 베이스 미탐지 — 지어내지 않고 솔직히 (ETF/채권형 등)
        cons = _tech_fnum(row.get("entry_conservative"))
        if cons is not None:
            segs.append(f"CAN SLIM 베이스는 미탐지되어 SMA50 되돌림 {_tech_price(cons, sym, cur)} 부근만 보수적 참고")
        else:
            segs.append("CAN SLIM 베이스는 미탐지되어 SMA·모멘텀 시그널만 참고")

    # ── Elliott Wave 손절 서브절 ──
    stop_price = _tech_fnum(row.get("stop_price"))
    stop_pct   = _tech_fnum(row.get("stop_pct"))
    stop_type  = (row.get("stop_type") or "").upper()
    if stop_price is not None and stop_type not in ("UNAVAILABLE", ""):
        wave_kr = _WAVE_GUESS_KR.get(row.get("stop_wave_guess"), "")
        lead = f"엘리엇 파동상 {wave_kr} 국면으로 " if wave_kr else "엘리엇 파동 기준 "
        pct_str = f"({stop_pct:+.1f}%)" if stop_pct is not None else ""
        segs.append(f"{lead}{_tech_price(stop_price, sym, cur)}{pct_str} 손절선으로 관리")

    if not segs:
        return ""
    return "기술적으로는 " + ", ".join(segs) + "한다."


# 이 절이 이미 붙었는지 판별하는 마커(중복 append 방지)
_TECH_CLAUSE_MARKER = "기술적으로는 "


def _trend_freshness(trend_age) -> "str | None":
    """추세 나이(50일선 위 연속일) → 신선도 라벨.

    walk-forward(2026-07-14, 825종목·18리밸런스): 기술 top-100 풀 내에서
    FRESH(1~15d)가 MATURE(>45d)보다 forward 우위 — 21d Δ+5.7%p(t 2.4),
    42d Δ+13.5%p(t 3.2); 중첩제거 시 방향 6/6 유지되나 n 작음(단일 레짐 ~1년).
    → 근거 수준: '방향 일관·검증력 부족' — 표시 배지 + 동순위 타이브레이커 전용,
    하드 게이트/사이징 반영 금지.
    """
    try:
        a = float(trend_age)
    except (TypeError, ValueError):
        return None
    if a < 1:
        return None          # 50일선 아래 — 추세 없음
    if a <= 15:
        return "FRESH"       # 🌱 신선 추세 (전환 초기 — 남은 여력 상대적 우위)
    if a > 45:
        return "MATURE"      # 🧓 성숙 추세 (노후 — 감쇠/되돌림 위험 상대적 열위)
    return None              # 16~45d 중간 구간은 무라벨


def enrich_debate_with_technicals(rows: list) -> list:
    """buy_list/active_positions/exit_pending 행들의 debate_transcript 뒤에
    CAN SLIM + Elliott 기술 절을 덧붙인다. annotate_buy_list_with_entries /
    annotate_buy_list_with_stops가 먼저 실행되어 entry_*/stop_* 필드가 붙어 있어야
    한다(그 다음 호출). 이미 붙은 행은 건너뛴다(idempotent)."""
    for r in rows or []:
        try:
            existing = r.get("debate_transcript") or ""
            if _TECH_CLAUSE_MARKER in existing:
                continue  # 이미 기술 절 포함 — 중복 방지
            clause = _technical_entry_stop_clause(r)
            if not clause:
                continue
            r["debate_transcript"] = (existing + " " + clause).strip() if existing else clause
        except Exception:
            # 커멘터리 보강은 non-fatal — 실패해도 원 transcript 유지
            continue
    return rows


def _scan_row_returns(scan_row: dict) -> dict:
    """Derive trailing returns (as FRACTIONS) from a scan_cache row's precomputed
    percent returns. scan stores ret_5d/ret_1m/ret_3m/ret_126d/ret_252d in PERCENT.
    final_list convention is fraction (e.g. 0.034 = +3.4%), so divide by 100.
    """
    out = {k: None for k in TRAILING_HORIZONS}
    if not scan_row:
        return out
    # final_list horizon → scan_cache field
    MAP = {"ret_5d": "ret_5d", "ret_1mo": "ret_1m", "ret_3mo": "ret_3m",
           "ret_6mo": "ret_126d", "ret_1y": "ret_252d"}
    for fl_key, scan_key in MAP.items():
        v = scan_row.get(scan_key)
        if v is not None:
            try:
                out[fl_key] = float(v) / 100.0
            except (TypeError, ValueError):
                pass
    return out


def _compute_trailing_returns(ticker: str, price_data: dict, scan_row: dict = None) -> dict:
    """Compute trailing total returns for a ticker over standard horizons.

    Primary source: price_data (.backtest_price_cache.pkl) close series.
    Fallback: scan_cache row's precomputed percent returns (covers full 800-ticker
    universe even when the backtest price cache is stale/partial).

    Returns dict like {"ret_5d": 0.034, "ret_1mo": 0.12, ...} (fractions).
    """
    out = {k: None for k in TRAILING_HORIZONS}
    df = price_data.get(ticker)
    if df is not None and not df.empty:
        try:
            closes = df["Close"]
            if len(closes) > 0:
                current = float(closes.iloc[-1])
                if current > 0:
                    for key, days in TRAILING_HORIZONS.items():
                        if len(closes) < days + 1:
                            continue
                        past = float(closes.iloc[-days - 1])
                        if past > 0:
                            out[key] = current / past - 1
        except Exception:
            pass
    # Fallback for any missing horizon → scan_cache precomputed returns
    if scan_row and any(out[k] is None for k in out):
        scan_ret = _scan_row_returns(scan_row)
        for k in out:
            if out[k] is None and scan_ret.get(k) is not None:
                out[k] = scan_ret[k]
    return out


def _load_sources() -> dict:
    """Pull all signal sources for 3-Agent Voting (PM + Trading + Risk).

    Sources:
      - PM Swarm picks      → PM vote (composite + classification 기반)
      - Trading entry_signal → Trading vote (BUY_NOW/WAIT/SKIP)
      - Scan cache         → Risk vote (deterministic risk score)
      - Backtest data      → cross-check info (legacy compatibility)
    """
    sw = _safe_load(".market_leaders_swarm_cache.json", {})
    bt = _safe_load("backtest/results.json", {})
    rk = _safe_load("backtest/ticker_rankings.json", {})
    ps = _safe_load(".position_state.json", {"positions": {}})

    # Proxy RECENT cohorts — union of last 4 weeks (was: only latest 1)
    tlc = bt.get("trading_lifecycles_compact", {}).get("core", {})
    proxy_long_stocks: set[str] = set()
    proxy_long_etfs:   set[str] = set()
    latest_date = bt.get("end_date") or ""
    if tlc:
        all_cohorts_stocks = sorted({r["d"] for r in tlc.get("long_stocks", [])})
        all_cohorts_etfs   = sorted({r["d"] for r in tlc.get("long_etfs",   [])})
        # Take last 4 cohorts (1 month of proxy activity)
        recent_stocks_cohorts = set(all_cohorts_stocks[-4:]) if all_cohorts_stocks else set()
        recent_etfs_cohorts   = set(all_cohorts_etfs[-4:])   if all_cohorts_etfs   else set()
        proxy_long_stocks = {r["t"] for r in tlc.get("long_stocks", []) if r["d"] in recent_stocks_cohorts}
        proxy_long_etfs   = {r["t"] for r in tlc.get("long_etfs",   []) if r["d"] in recent_etfs_cohorts}

    # Top alpha leaders — EXTENDED to top 40 (was: 20)
    top_stocks = {x["ticker"] for x in rk.get("long_stocks", {}).get("top", [])[:40]}
    top_etfs   = {x["ticker"] for x in rk.get("long_etfs",   {}).get("top", [])[:40]}
    worst_stocks = {x["ticker"] for x in rk.get("long_stocks", {}).get("worst", [])[:30]}
    worst_etfs   = {x["ticker"] for x in rk.get("long_etfs",   {}).get("worst", [])[:30]}

    # Load scan_cache for risk evaluation (Phase 5.7 Risk Manager)
    scan_lookup: dict[str, dict] = {}
    try:
        import pickle
        sc_path = Path(".scan_cache.pkl")
        if sc_path.exists():
            sc = pickle.load(open(sc_path, "rb"))
            for r in sc.get("results", []) or []:
                t = r.get("ticker")
                if t:
                    scan_lookup[t] = r
    except Exception:
        pass

    # Entry Timing (①②⑤) — raw scan row에는 필요한 입력 필드(rsi/rss_short·long/
    # trend_age/pct_from_high 등)가 모두 있으므로 여기서 결정론적으로 계산해 주입
    # (api load의 df 계산과 동일 함수). 매수 pick 라벨/재정렬에 사용.
    try:
        from agents.entry_timing import compute_entry_timing
        for _t, _r in scan_lookup.items():
            _r.update(compute_entry_timing(_r))
    except Exception:
        pass

    # Herding-Reversal (과열 후 반전, paper 2607.27063) — scan_cache엔 없는 post-load 신호.
    # api._compute_herding_reversal_layer가 .herding_reversal_cache.json에 써둔 것을 읽어
    # scan_lookup에 병합 → 매수 tie-breaker(_hrr_rank) + 라벨. 없으면 무시(비치명적).
    try:
        _hrp = Path(".herding_reversal_cache.json")
        if _hrp.exists():
            _hr = json.loads(_hrp.read_text())
            for _t, _d in _hr.items():
                if _t in scan_lookup and isinstance(_d, dict):
                    scan_lookup[_t].update(_d)
    except Exception:
        pass

    # Count sector concentration across PM picks (for Risk concentration vote)
    sector_counts: dict[str, int] = {}
    for h in ("core",):
        h_data = (sw.get("phase5_pm") or {}).get("horizons", {}).get(h, {}) or {}
        for bucket in ("long_stocks", "long_etfs"):
            for p in h_data.get(bucket, []) or []:
                sector = (p.get("sector") or "").strip()
                if sector:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1

    return {
        "pm_swarm": sw,
        "proxy_long_stocks": proxy_long_stocks,
        "proxy_long_etfs":   proxy_long_etfs,
        "proxy_cohort_date": latest_date,
        "top_stocks": top_stocks,
        "top_etfs":   top_etfs,
        "worst_stocks": worst_stocks,
        "worst_etfs":   worst_etfs,
        "positions": ps.get("positions", {}),
        "price_data": _load_price_cache(),
        "scan_lookup": scan_lookup,
        "sector_counts": sector_counts,
    }


# Composite threshold for "high composite" cross-check (lowered: more inclusive)
HIGH_COMPOSITE_THRESHOLD = 70
# Two-track tier logic — primary (backtest) vs bonus (current state):
#   ★★★ = both primary + ANY bonus
#   ★★  = ANY primary OR (no primary but 2 bonuses)
#   ★   = otherwise (PM only)


# ─────────────────────────────────────────────────────────────────────
# Per-pick scoring + filtering
# ─────────────────────────────────────────────────────────────────────

def _eval_buy_pick_with_debate(p: dict, horizon: str, bucket: str, sources: dict) -> Optional[dict]:
    """Score one PM Buy candidate using DEBATE SYNTHESIS (Option A).

    Uses debate_synthesis output if available (from Phase 5.6a Synthesizer).
    Falls back to deterministic voting if synthesis missing.
    """
    debate = p.get("debate_synthesis")
    timing = p.get("timing") or {}
    t = p.get("ticker", "")
    composite = float(p.get("composite") or 0)
    key = f"{t}::{horizon}"
    pos = sources["positions"].get(key, {})
    state = pos.get("state", "PROSPECTING")
    sig_days = (pos.get("consecutive_signal_days") or {}).get("BUY_NOW", 0)

    if not debate:
        # No debate synthesis → use legacy voting
        return _eval_buy_pick(p, horizon, bucket, sources)

    # Use synthesizer's output. NOTE: .get(k, default) only applies the default
    # when the key is ABSENT — a present-but-None value (a failed/degenerate
    # debate synthesis can emit stars/tier/final_decision = None) slips through
    # and later breaks `stars >= 2` etc. Coerce None explicitly. stars==0 is a
    # valid EXCLUDE signal, so guard it with an is-None check (not `or`).
    tier = debate.get("tier") or "SOLO"
    stars = debate.get("stars")
    stars = 1 if stars is None else stars
    final_decision = debate.get("final_decision") or "WATCH"
    debate_transcript = debate.get("debate_transcript", "")
    key_factor = debate.get("key_factor", "")

    # EXCLUDED → filter out
    if final_decision == "EXCLUDE" or stars == 0:
        return None

    # Trailing returns
    trailing = _compute_trailing_returns(t, sources.get("price_data") or {}, sources["scan_lookup"].get(t))

    # Action urgency
    if state == "ENTERED" and pos.get("last_alert") and "NEW BUY" in (pos.get("last_alert") or ""):
        action = "EXECUTE_TODAY"
    elif state == "ENTERED":
        action = "ALREADY_HELD"
    elif state == "PROSPECTING" and sig_days >= 1:
        action = "WATCH_TOMORROW"
    elif state == "HOLDING":
        action = "ALREADY_HELD"
    else:
        action = "OBSERVE"

    # Legacy votes (for backward-compat display)
    from agents.risk_manager import compute_risk_score, risk_vote, risk_rationale, pm_vote, trading_vote, tally_votes
    scan_row = sources["scan_lookup"].get(t, {})
    # bucket 정규화 — swarm/PM이 FICC·국제 ETF를 long_stocks로 잘못 넣은 경우 교정.
    # scan_row category가 "STK_"가 아니면 ETF (single source of truth).
    if scan_row.get("category"):
        _side = "short" if bucket.startswith("short") else "long"
        bucket = _infer_bucket(scan_row, _side)
    # pm_v: debate 경로에서는 debate tier/consensus에서 도출 — deterministic pm_vote는
    # debate-driven stars와 35% 모순 (표시 votes ↔ tier 정합 보장). trading_v는 timing
    # (=per-ticker debate trading verdict), risk_v는 LLM verdict이라 이미 debate 정합.
    if final_decision == "EXCLUDE" or tier == "REJECTED":
        pm_v = "REJECT"
    elif tier in ("UNANIMOUS", "MAJORITY_CLEAN") or (final_decision == "INCLUDE" and stars >= 3):
        pm_v = "APPROVE"
    elif tier in ("MAJORITY_DISSENT", "SOLO_CLEAN", "ALL_CAUTION") or stars >= 2:
        pm_v = "CAUTION"
    else:
        pm_v = "REJECT"
    trading_v = trading_vote(timing)
    risk_breakdown = compute_risk_score(scan_row, sources["sector_counts"])
    risk_v_deterministic = risk_vote(risk_breakdown)
    # Use LLM risk verdict if available
    risk_llm = p.get("risk_verdict") or {}
    risk_v = risk_llm.get("vote", risk_v_deterministic)

    # ── EXPLICIT FAILURE DETECTION (Option C Tier 1) ──
    # Honest-failure mode: agents now mark _failed=True when LLM didn't produce valid output.
    # Previously default SOLO/WATCH/WAIT/CAUTION cascaded silently.
    composite = float(p.get("composite") or 0)
    cls_raw = scan_row.get("classification") or p.get("classification") or ""
    cls_str = cls_raw if isinstance(cls_raw, str) else ""
    is_strong = any(s in cls_str for s in ("CONTINUATION","FORMATION","RECOVERY","LAGGING_CATCHUP"))

    debate_failed  = bool((p.get("debate_synthesis") or {}).get("_failed"))
    trading_failed = bool((timing or {}).get("_failed"))
    risk_failed    = bool(risk_llm.get("_failed"))
    any_agent_failed = debate_failed or trading_failed or risk_failed

    # Detect heuristic degradation (legacy cache may not have _failed flag yet)
    legacy_degraded = (
        tier == "SOLO"
        and final_decision == "WATCH"
        and (timing or {}).get("entry_signal", "") == "WAIT"
        and risk_llm.get("vote") == "CAUTION"
    )

    if any_agent_failed or (legacy_degraded and composite >= 65 and is_strong):
        # Composite-based PM-conviction override
        if composite >= 75 and is_strong:
            pm_v = "APPROVE"; trading_v = "APPROVE"
            risk_v = "APPROVE" if risk_v_deterministic == "APPROVE" else risk_v
            p["_votes_overridden"] = "agent_failure" if any_agent_failed else "swarm_degraded"
        elif composite >= 65 and is_strong:
            pm_v = "APPROVE"; trading_v = "CAUTION"
            risk_v = "APPROVE" if risk_v_deterministic == "APPROVE" else risk_v
            p["_votes_overridden"] = "agent_failure" if any_agent_failed else "swarm_degraded"
        elif composite >= 55:
            # Mid composite — fall back to deterministic verdicts (don't fake APPROVE)
            pm_v = "CAUTION"; trading_v = "CAUTION"; risk_v = risk_v_deterministic
            p["_votes_fallback"] = "deterministic_only"
        else:
            # Low composite — accept REJECT signal
            pm_v = "REJECT" if composite < 40 else "CAUTION"
            trading_v = "CAUTION"; risk_v = risk_v_deterministic
            p["_votes_fallback"] = "deterministic_only"

        # Record which agents failed (for downstream visibility)
        p["_failed_agents"] = {
            "trading": trading_failed,
            "risk":    risk_failed,
            "debate":  debate_failed,
        }

    proxy_set = sources["proxy_long_stocks"] if "stocks" in bucket else sources["proxy_long_etfs"]
    top_set   = sources["top_stocks"]        if "stocks" in bucket else sources["top_etfs"]
    in_proxy_recent = t in proxy_set
    in_top_alpha    = t in top_set
    state_confirmed = state == "ENTERED"
    high_composite  = composite >= HIGH_COMPOSITE_THRESHOLD

    trailing = _compute_trailing_returns(t, sources.get("price_data") or {}, sources["scan_lookup"].get(t))

    return {
        "ticker": t,
        "name": (p.get("name") or "")[:40],
        "sector": p.get("sector") or "",
        "composite": composite,
        "classification": scan_row.get("classification") or p.get("classification") or "",
        "horizon": horizon,
        "bucket": bucket,
        "stars": stars,
        # ── DEBATE SYNTHESIS (Option A — Phase 5.6a output) ──
        "tier": tier,
        "consensus": tier,   # alias for backward-compat
        # Unified commentary = one flowing Korean paragraph (PM 선정근거 → 진입타이밍
        # → 리스크 → 비판 → 합의결론). 기계 헤더·이모지 라벨 없음, 중복 제거.
        "debate_transcript": _unified_commentary(
            tier=tier,
            key_factor=key_factor,
            final_decision=final_decision,
            trading_signal=timing.get("entry_signal") or "",
            pm_reason=p.get("rationale") or "",
            trading_reason=(timing.get("rationale") or ""),
            risk_reason=(risk_llm.get("rationale") or risk_rationale(risk_breakdown)),
            critic_challenge=(debate.get("critic_challenge")
                              or _critic_from_transcript(debate_transcript)),
        ),
        "key_factor": key_factor,
        "final_decision": final_decision,
        # Entry Timing (진입 적시성, ①②⑤) — scan에서 계산, 매수 리스트 재정렬/라벨용
        "entry_timing_score": scan_row.get("entry_timing_score"),
        "entry_timing_status": scan_row.get("entry_timing_status"),
        # pool_source provenance (2026-07-28) — "momentum"(확정) | "pre_momentum"(forming/조기).
        # forming 후보의 단계별 생존률 추적 + UI에서 조기포착 픽 식별용.
        "pool_source": p.get("pool_source") or "momentum",
        "pm_pre_score": p.get("pre_momentum_score"),
        # Herding-Reversal (과열 후 반전, paper 2607.27063) — 과밀+herding약화 conjunction.
        # 검증상 희귀·고확신 → 매수 tie-breaker 전용(하드제외 X). 표시/사유 라벨용.
        "hrr_flag": scan_row.get("hrr_flag"),
        "hrr_score": scan_row.get("hrr_score"),
        "hrr_reasons": scan_row.get("hrr_reasons") or [],
        "hrr_cross_confirmed": bool(scan_row.get("hrr_cross_confirmed")),
        # Trend Freshness (추세 나이) — walk-forward 검증(2026-07-14): 풀 내 신선(≤15d)이
        # 성숙(>45d)보다 forward 우위(방향 6/6 일관, 42d Δ+13.5%p). 표시/타이브레이커 전용.
        "trend_age": scan_row.get("trend_age"),
        "trend_freshness": _trend_freshness(scan_row.get("trend_age")),
        # Portfolio Composer (Phase 5b) 산출 — portfolio_agent가 가중치에 소비
        # (기존엔 계산만 되고 아무도 읽지 않던 죽은 레이어였음)
        "final_size": p.get("final_size"),
        "composition_decision": p.get("composition_decision"),
        # 3-Agent Voting (for legacy display)
        "votes": {
            "pm":      pm_v,
            "trading": trading_v,
            "risk":    risk_v,
        },
        "n_approve":   sum(1 for v in [pm_v, trading_v, risk_v] if v == "APPROVE"),
        "n_reject":    sum(1 for v in [pm_v, trading_v, risk_v] if v == "REJECT"),
        "n_caution":   sum(1 for v in [pm_v, trading_v, risk_v] if v == "CAUTION"),
        # Risk details (LLM if available, else deterministic)
        "risk_score":  risk_breakdown["total"],
        "risk_breakdown": risk_breakdown,
        "risk_reason": (risk_llm.get("rationale") or risk_rationale(risk_breakdown))[:300],
        "risk_key":    risk_llm.get("key_risk", "—"),
        # Legacy cross-check
        "n_validations": sum([in_proxy_recent, in_top_alpha, state_confirmed, high_composite]),
        "in_proxy_recent":   in_proxy_recent,
        "in_proxy_latest":   in_proxy_recent,
        "in_top_alpha":      in_top_alpha,
        "state_confirmed":   state_confirmed,
        "high_composite":    high_composite,
        # Position state
        "state": state,
        "signal_days": sig_days,
        "urgency": timing.get("urgency", "NORMAL"),
        "action": action,
        # Reasoning
        "entry_trigger": (timing.get("entry_trigger") or "")[:200],
        "rationale": (p.get("rationale") or "")[:300],
        "trading_rationale": (timing.get("rationale") or "")[:300],
        "exit_triggers": (timing.get("exit_triggers") or [])[:2],
        # Trailing returns
        **trailing,
    }


def _eval_buy_pick(p: dict, horizon: str, bucket: str, sources: dict) -> Optional[dict]:
    """Score one PM Buy candidate using 3-Agent Voting (PM + Trading + Risk).

    Three agents vote APPROVE / CAUTION / REJECT:
      1. PM Vote      : conviction (composite + classification)
      2. Trading Vote : entry_signal (BUY_NOW → APPROVE)
      3. Risk Vote    : risk score (overheating + vol + liquidity + concentration + drawdown)

    Tier (Stars):
      ★★★ UNANIMOUS         : 3 APPROVE
      ★★★ MAJORITY_CLEAN    : 2 APPROVE + 1 CAUTION
      ★★  MAJORITY_DISSENT  : 2 APPROVE + 1 REJECT
      ★★  SOLO_CLEAN        : 1 APPROVE + 2 CAUTION
      ★   SOLO_DISSENT      : 1 APPROVE + 1 REJECT + 1 CAUTION
      ★   ALL_CAUTION       : 3 CAUTION (no clear signal)
      (excluded) REJECTED   : ≥2 REJECT
    """
    from agents.risk_manager import (
        compute_risk_score, risk_vote, risk_rationale,
        pm_vote, trading_vote, tally_votes,
    )

    timing = p.get("timing") or {}
    t = p.get("ticker", "")
    composite = float(p.get("composite") or 0)
    key = f"{t}::{horizon}"
    pos = sources["positions"].get(key, {})
    state = pos.get("state", "PROSPECTING")
    sig_days = (pos.get("consecutive_signal_days") or {}).get("BUY_NOW", 0)

    # ── 3 AGENT VOTES ──
    # 1. PM Agent vote (conviction)
    scan_row = sources["scan_lookup"].get(t, {})
    # bucket 정규화 — FICC·국제 ETF의 long_stocks 오분류 교정 (category="STK_" 아니면 ETF)
    if scan_row.get("category"):
        _side = "short" if bucket.startswith("short") else "long"
        bucket = _infer_bucket(scan_row, _side)
    pm_pick_enriched = {
        **p,
        "classification": scan_row.get("classification") or p.get("classification") or "",
    }
    pm_v = pm_vote(pm_pick_enriched)

    # 2. Trading Agent vote (entry_signal)
    trading_v = trading_vote(timing)

    # 3. Risk Manager vote (deterministic score)
    risk_breakdown = compute_risk_score(scan_row, sources["sector_counts"])
    risk_v = risk_vote(risk_breakdown)
    risk_reason = risk_rationale(risk_breakdown)

    # Tally votes → stars
    tally = tally_votes(pm_v, trading_v, risk_v)
    stars = tally["stars"]

    # Filter: REJECTED tier (≥2 REJECT) excluded from Final List
    if stars == 0:
        return None

    # Legacy cross-check info (for display "additional info" but no longer gates filtering)
    proxy_set = sources["proxy_long_stocks"] if "stocks" in bucket else sources["proxy_long_etfs"]
    top_set   = sources["top_stocks"]        if "stocks" in bucket else sources["top_etfs"]
    in_proxy_recent = t in proxy_set
    in_top_alpha    = t in top_set
    state_confirmed = state == "ENTERED"
    high_composite  = composite >= HIGH_COMPOSITE_THRESHOLD
    n_validations = sum([in_proxy_recent, in_top_alpha, state_confirmed, high_composite])

    trailing = _compute_trailing_returns(t, sources.get("price_data") or {}, sources["scan_lookup"].get(t))

    # Action urgency (Phase 5.6 state machine integration)
    if state == "ENTERED" and pos.get("last_alert") and "NEW BUY" in (pos.get("last_alert") or ""):
        action = "EXECUTE_TODAY"
    elif state == "ENTERED":
        action = "ALREADY_HELD"
    elif state == "PROSPECTING" and sig_days >= 1:
        action = "WATCH_TOMORROW"
    elif state == "HOLDING":
        action = "ALREADY_HELD"
    else:
        action = "OBSERVE"

    # Build the commentary from the deterministic 3-agent votes → one flowing
    # paragraph (these picks weren't LLM-debated — beyond PT_DEBATE_MAX_PICKS cap).
    # No critic in this path → sentinel "" so the 비판 절이 자연 생략된다.
    _consensus = tally["consensus"]
    _tier_label = {
        "UNANIMOUS": "만장일치", "MAJORITY_CLEAN": "강한 다수 합의",
        "MAJORITY_DISSENT": "다수 동의 (이견)", "SOLO_CLEAN": "단독 승인",
        "SOLO_DISSENT": "단독 승인 (리스크)", "ALL_CAUTION": "전원 주의",
        "REJECTED": "거부",
    }.get(_consensus, _consensus)
    _det_transcript = _unified_commentary(
        tier=_consensus,
        key_factor=_tier_label,
        final_decision="INCLUDE" if stars >= 2 else "WATCH",
        trading_signal=timing.get("entry_signal") or "",
        pm_reason=p.get("rationale") or "",
        trading_reason=(timing.get("rationale") or ""),
        risk_reason=risk_reason,
        critic_challenge="",
    )

    return {
        "ticker": t,
        "name": (p.get("name") or "")[:40],
        "sector": p.get("sector") or "",
        "composite": composite,
        "classification": scan_row.get("classification") or p.get("classification") or "",
        "entry_timing_score": scan_row.get("entry_timing_score"),
        "entry_timing_status": scan_row.get("entry_timing_status"),
        "trend_age": scan_row.get("trend_age"),
        "trend_freshness": _trend_freshness(scan_row.get("trend_age")),
        "horizon": horizon,
        "bucket": bucket,
        "stars": stars,
        # ── 3-Agent Voting (Option C) ──
        "votes": {
            "pm":      pm_v,
            "trading": trading_v,
            "risk":    risk_v,
        },
        "consensus":   tally["consensus"],
        "tier":        tally["consensus"],
        "n_approve":   tally["n_approve"],
        "n_reject":    tally["n_reject"],
        "n_caution":   tally["n_caution"],
        "risk_score":  risk_breakdown["total"],
        "risk_breakdown": risk_breakdown,
        "risk_reason": risk_reason,
        "risk_key":    "—",
        # Debate fields (deterministic fallback — no LLM debate ran)
        "debate_transcript": _det_transcript[:900],
        "key_factor":        _tier_label,
        "final_decision":    "INCLUDE" if stars >= 2 else "WATCH",
        # Legacy cross-check info (still shown, no longer gates)
        "n_validations": n_validations,
        "in_proxy_recent":   in_proxy_recent,
        "in_proxy_latest":   in_proxy_recent,
        "in_top_alpha":      in_top_alpha,
        "state_confirmed":   state_confirmed,
        "high_composite":    high_composite,
        # Position state info
        "state": state,
        "signal_days": sig_days,
        "urgency": timing.get("urgency", "NORMAL"),
        "action": action,
        # Reasoning
        "entry_trigger": (timing.get("entry_trigger") or "")[:200],
        "rationale": (p.get("rationale") or "")[:300],                # PM Agent's reason
        "trading_rationale": (timing.get("rationale") or "")[:300],   # Trading Agent's reason
        "exit_triggers": (timing.get("exit_triggers") or [])[:2],
        # Trailing total returns
        **trailing,
    }


def _pm_vote_short(pick: dict) -> str:
    """PM Agent vote for SHORT side (mirror of LONG)."""
    composite = float(pick.get("composite") or 0)
    cls = pick.get("classification", "") or ""
    weak = ("DOWNTREND", "WEAKENING", "FADING", "CYCLE_PEAK")
    strong_bear = any(s in cls for s in weak)
    if composite <= 30 and strong_bear:  return "APPROVE"
    if composite <= 25:                  return "APPROVE"
    if composite >= 50:                  return "REJECT"
    return "CAUTION"


def _risk_vote_short(scan_row: dict, sector_counts: dict) -> tuple[str, dict, str]:
    """Risk vote for SHORT side — mirrors the long-side risk logic.

    For shorts, the major risk dimensions are:
      - Squeeze risk (rising momentum, low short interest data, oversold bounce)
      - Liquidity risk (illiquid shorts dangerous)
      - Drawdown via short squeeze
    We approximate by inverting overheating: high RSI/OER on a "weak" ticker = bounce risk.
    """
    from agents.risk_manager import compute_risk_score, RISK_VOTE_APPROVE_MAX, RISK_VOTE_REJECT_MIN
    # Reuse long-side risk score but interpret as "execution risk of short"
    rb = compute_risk_score(scan_row, sector_counts)
    score = rb.get("total", 50)
    if score <= RISK_VOTE_APPROVE_MAX: vote = "APPROVE"
    elif score >= RISK_VOTE_REJECT_MIN: vote = "REJECT"
    else: vote = "CAUTION"
    reason = f"Short 실행 리스크 총점 {score:.0f}/100"
    return vote, rb, reason


def _eval_sell_pick(p: dict, horizon: str, bucket: str, sources: dict) -> Optional[dict]:
    """Score one PM Sell/Short candidate using 3-Agent Voting (mirror of buy)."""
    from agents.risk_manager import trading_vote, tally_votes

    timing = p.get("timing") or {}
    t = p.get("ticker", "")
    composite = float(p.get("composite") or 0)
    key = f"{t}::{horizon}"
    pos = sources["positions"].get(key, {})
    state = pos.get("state", "PROSPECTING")
    sig_days = (pos.get("consecutive_signal_days") or {}).get("BUY_NOW", 0)

    # 3 votes (mirror buy side, adapted for SHORT)
    scan_row = sources["scan_lookup"].get(t, {})
    # bucket 정규화 — ETF/stock 오분류 교정 (category="STK_" 아니면 ETF)
    if scan_row.get("category"):
        _side = "short" if bucket.startswith("short") else "long"
        bucket = _infer_bucket(scan_row, _side)
    pm_pick_enriched = {**p, "classification": scan_row.get("classification") or p.get("classification") or ""}
    pm_v = _pm_vote_short(pm_pick_enriched)
    trading_v = trading_vote(timing)
    risk_v, risk_breakdown, risk_reason = _risk_vote_short(scan_row, sources["sector_counts"])

    tally = tally_votes(pm_v, trading_v, risk_v)
    stars = tally["stars"]
    if stars == 0:
        return None

    # Legacy worst-alpha cross-check
    worst_set = sources["worst_stocks"] if "stocks" in bucket else sources["worst_etfs"]
    in_worst_alpha = t in worst_set
    state_confirmed = state == "ENTERED"
    low_composite   = composite <= 30
    n_validations = sum([in_worst_alpha, state_confirmed, low_composite])

    if state == "ENTERED" and pos.get("last_alert") and "NEW BUY" in (pos.get("last_alert") or ""):
        action = "EXECUTE_TODAY"
    elif state == "PROSPECTING" and sig_days >= 1:
        action = "WATCH_TOMORROW"
    else:
        action = "OBSERVE"

    trailing = _compute_trailing_returns(t, sources.get("price_data") or {}, sources["scan_lookup"].get(t))

    return {
        "ticker": t,
        "name": (p.get("name") or "")[:40],
        "sector": p.get("sector") or "",
        "composite": composite,
        "classification": scan_row.get("classification") or p.get("classification") or "",
        "horizon": horizon,
        "bucket": bucket,
        "stars": stars,
        # ── 3-Agent Voting (mirror of buy) ──
        "votes": {"pm": pm_v, "trading": trading_v, "risk": risk_v},
        "consensus":   tally["consensus"],
        "n_approve":   tally["n_approve"],
        "n_reject":    tally["n_reject"],
        "n_caution":   tally["n_caution"],
        "risk_score":  risk_breakdown["total"],
        "risk_breakdown": risk_breakdown,
        "risk_reason": risk_reason,
        # Legacy cross-check
        "n_validations": n_validations,
        "in_worst_alpha":    in_worst_alpha,
        "state_confirmed":   state_confirmed,
        "low_composite":     low_composite,
        # Position state
        "state": state,
        "signal_days": sig_days,
        "urgency": timing.get("urgency", "NORMAL"),
        "action": action,
        # Reasoning
        "entry_trigger": (timing.get("entry_trigger") or "")[:200],
        "rationale": (p.get("rationale") or "")[:300],
        "trading_rationale": (timing.get("rationale") or "")[:300],
        # Trailing total returns
        **trailing,
    }


def _exit_pick(t: str, horizon: str, pos: dict) -> dict:
    """Build an 'exit existing position' entry from Phase 5.6 state."""
    return {
        "ticker": t, "name": "", "sector": "", "composite": 0,
        "horizon": horizon, "bucket": "(existing)",
        "stars": 3,
        "state": pos.get("state"),
        "signal_days": (pos.get("consecutive_signal_days") or {}).get("SKIP", 0),
        "urgency": "URGENT",
        "action": "CLOSE_TODAY",
        "entry_trigger": "",
        "rationale": (pos.get("state_history") or [{}])[-1].get("reason", "")[:140],
        "is_exit": True,
    }


# ─────────────────────────────────────────────────────────────────────
# Main entry — assemble both lists
# ─────────────────────────────────────────────────────────────────────

def _days_between(start_iso: str, end_iso: str) -> int:
    from datetime import datetime
    try:
        a = datetime.strptime(start_iso, "%Y-%m-%d").date()
        b = datetime.strptime(end_iso, "%Y-%m-%d").date()
        return (b - a).days
    except Exception:
        return 0


def _build_active_positions(sources: dict, buy_list: list, sell_list: list) -> list[dict]:
    """Build list of currently HELD positions (HOLDING / ENTERED state).

    These persist even if today's PM picks didn't include them.
    Cross-reference with today's buy/sell to mark which are in_today_picks.
    """
    import time
    from agents.risk_manager import compute_risk_score, risk_vote, risk_rationale
    today = time.strftime("%Y-%m-%d")
    positions = sources.get("positions") or {}
    scan_lookup = sources.get("scan_lookup") or {}
    price_data = sources.get("price_data") or {}
    sw = sources.get("pm_swarm") or {}
    horizons = (sw.get("phase5_pm") or {}).get("horizons", {})

    # Set of (ticker, horizon) in today's voted buy/sell lists for cross-ref
    today_buy_set = {(r["ticker"], r.get("horizon")) for r in buy_list}
    today_sell_set = {(r["ticker"], r.get("horizon")) for r in sell_list}

    active = []
    seen = set()
    for key, pos in positions.items():
        ticker, horizon = key.split("::", 1)
        state = pos.get("state", "")
        if state not in ("HOLDING", "ENTERED"):
            continue
        if (ticker, horizon) in seen:
            continue
        seen.add((ticker, horizon))

        # Find current PM pick (if today picked again)
        pm_pick = None
        bucket = None
        for b in ("long_stocks", "long_etfs", "short_stocks", "short_etfs"):
            for p in horizons.get(horizon, {}).get(b, []) or []:
                if p.get("ticker") == ticker:
                    pm_pick = p
                    bucket = b
                    break
            if pm_pick: break

        # Metadata
        scan_row = scan_lookup.get(ticker, {})
        # bucket 정규화 — swarm cache가 FICC·국제 ETF를 long_stocks로 잘못 넣었거나
        # PM이 오늘 안 뽑아 bucket=None인 경우, scan category로 재판별 (STK_ 아니면 ETF)
        if scan_row.get("category"):
            _side = "short" if (bucket or "long").startswith("short") else "long"
            bucket = _infer_bucket(scan_row, _side)
        # Agent debate fields — populated when the held position is ALSO in today's
        # PM swarm picks (mirror _eval_buy_pick_with_debate so HOLDING rows don't lose
        # the Debate Transcript / 3-Agent Votes columns).
        debate_fields: dict = {}
        if pm_pick:
            name = pm_pick.get("name") or ""
            sector = pm_pick.get("sector") or scan_row.get("sector") or scan_row.get("category", "")
            composite = float(pm_pick.get("composite") or scan_row.get("composite") or 0)
            rationale = (pm_pick.get("rationale") or "")[:200]
            timing = pm_pick.get("timing") or {}
            current_signal = timing.get("entry_signal", "—")
            trading_rationale = (timing.get("rationale") or "")[:200]
            # Extract debate synthesis (today's 3-agent verdict for this held position)
            debate = pm_pick.get("debate_synthesis") or {}
            risk_verdict = pm_pick.get("risk_verdict") or {}
            if debate and not debate.get("_failed"):
                # pm_vote: 디베이트 평결에서 도출 (buy-path _eval_buy_pick_with_debate와
                # 동일 매핑) — 기존 "PM picked it → implicit approve" 하드코딩은
                # 보유분 EXCLUDE 평결을 APPROVE로 표시하는 버그였음.
                _fd_d = debate.get("final_decision") or ""
                _tier_d = debate.get("tier") or ""
                _stars_d = int(debate.get("stars") or 0)
                if _fd_d == "EXCLUDE" or _tier_d in ("REJECTED", "EXCLUDED"):
                    pm_vote_d = "REJECT"
                elif _tier_d in ("UNANIMOUS", "MAJORITY_CLEAN") or (_fd_d == "INCLUDE" and _stars_d >= 3):
                    pm_vote_d = "APPROVE"
                elif _tier_d in ("MAJORITY_DISSENT", "SOLO_CLEAN", "ALL_CAUTION") or _stars_d >= 2:
                    pm_vote_d = "CAUTION"
                else:
                    pm_vote_d = "REJECT"
                trading_sig = (timing.get("entry_signal") or "").upper()
                trading_vote_d = ("APPROVE" if trading_sig == "BUY_NOW"
                                   else "REJECT" if trading_sig == "SKIP" else "CAUTION")
                risk_vote_d = (risk_verdict.get("vote") or "CAUTION").upper()
                # 보유 종목도 매수 픽과 동일한 흐름형 코멘터리로 렌더 (기계 헤더 없이).
                _held_commentary = _unified_commentary(
                    tier=debate.get("tier") or "",
                    key_factor=debate.get("key_factor") or "",
                    final_decision=debate.get("final_decision") or "",
                    trading_signal=(timing.get("entry_signal") or ""),
                    pm_reason=pm_pick.get("rationale") or "",
                    trading_reason=(timing.get("rationale") or ""),
                    risk_reason=(risk_verdict.get("rationale")
                                 or risk_rationale(
                                     compute_risk_score(scan_row, sources.get("sector_counts") or {}))),
                    critic_challenge=(debate.get("critic_challenge")
                                      or _critic_from_transcript(debate.get("debate_transcript", ""))),
                )
                debate_fields = {
                    "tier": debate.get("tier"),
                    "consensus": debate.get("tier"),
                    "stars": debate.get("stars"),
                    "debate_transcript": _held_commentary,
                    "key_factor": debate.get("key_factor", ""),
                    "final_decision": debate.get("final_decision"),
                    "votes": {"pm": pm_vote_d, "trading": trading_vote_d, "risk": risk_vote_d},
                    "n_approve": sum(1 for v in (pm_vote_d, trading_vote_d, risk_vote_d) if v == "APPROVE"),
                    "n_reject":  sum(1 for v in (pm_vote_d, trading_vote_d, risk_vote_d) if v == "REJECT"),
                    "n_caution": sum(1 for v in (pm_vote_d, trading_vote_d, risk_vote_d) if v == "CAUTION"),
                }
        else:
            name = scan_row.get("name") or ""
            sector = scan_row.get("sector") or scan_row.get("category", "")
            composite = float(scan_row.get("composite") or 0)
            rationale = "오늘 PM picks에 없음 — 보유 유지 중 (직전 진입 근거로 보유 지속)"
            current_signal = "—"
            trading_rationale = "오늘 Trading Agent 미평가 — HOLDING은 sticky (신규 신호 없어도 보유 유지)"
            # No today verdict — mark clearly so frontend shows "오늘 미평가" instead of blank
            debate_fields = {
                "tier": "HELD_NO_EVAL",
                "consensus": "HELD_NO_EVAL",
                "debate_transcript": "오늘 PM swarm이 이 종목을 재평가하지 않음 — 보유 포지션 유지 중. "
                                      "신규 매수/청산 신호 발생 시에만 재토론. (직전 진입 근거는 유효)",
                "key_factor": "보유 유지",
                "final_decision": "HOLD",
            }

        # Compute days
        entered_date = pos.get("entered_date")
        first_seen = pos.get("first_seen") or entered_date
        days_held = _days_between(entered_date, today) if entered_date else 0
        persistence_days = _days_between(first_seen, today) if first_seen else 0

        # Trailing returns
        trailing = _compute_trailing_returns(ticker, price_data, scan_lookup.get(ticker))

        # Risk (compute_risk_score/risk_vote/risk_rationale imported at fn top)
        risk_breakdown = compute_risk_score(scan_row, sources.get("sector_counts") or {})
        risk_v = risk_vote(risk_breakdown)

        # Bucket fallback: if PM didn't include this position today, infer from scan category
        if not bucket:
            bucket = _infer_bucket(scan_row, side="long")
        active.append({
            "ticker": ticker,
            "name": name[:40],
            "sector": sector,
            "composite": composite,
            # classification 누락 시 portfolio_agent._class_mult가 전 보유분에
            # 0.82 폴백을 일괄 적용하던 버그 수정 — scan의 실제 추세 상태 전달
            "classification": scan_row.get("classification")
                              or (pm_pick.get("classification") if pm_pick else "")
                              or "",
            "horizon": horizon,
            "bucket": bucket,
            "state": state,
            "entered_date": entered_date,
            "entered_price": pos.get("entered_price"),
            # L1 telemetry — MFE/MAE/이익보존율 (trade_mgmt, backfilled for legacy holds)
            "mfe_pct": (pos.get("trade_mgmt") or {}).get("mfe_pct"),
            "mae_pct": (pos.get("trade_mgmt") or {}).get("mae_pct"),
            "tm_pnl_pct": (pos.get("trade_mgmt") or {}).get("pnl_pct"),
            "effective_stop": (pos.get("trade_mgmt") or {}).get("effective_stop"),
            # T5 give-back latch (SHADOW) — read-only during 검증 기간
            "giveback_armed":  (pos.get("trade_mgmt") or {}).get("giveback_armed"),
            "giveback_fired":  (pos.get("trade_mgmt") or {}).get("giveback_fired"),
            "giveback_shadow": (pos.get("trade_mgmt") or {}).get("giveback_shadow"),
            "retained_pct":    (pos.get("trade_mgmt") or {}).get("retained_pct"),
            "giveback_reason": (pos.get("trade_mgmt") or {}).get("giveback_reason"),
            "first_seen": first_seen,
            "days_held": days_held,
            "persistence_days": persistence_days,
            "in_today_buy_picks":  (ticker, horizon) in today_buy_set,
            "in_today_sell_picks": (ticker, horizon) in today_sell_set,
            "current_signal": current_signal,
            "rationale": rationale,
            "trading_rationale": trading_rationale,
            "risk_score": risk_breakdown["total"],
            "risk_vote": risk_v,
            "risk_reason": risk_rationale(risk_breakdown),
            "last_alert": pos.get("last_alert"),
            **debate_fields,
            **trailing,
        })

    # Sort by days_held descending (longest held first)
    active.sort(key=lambda x: (-x.get("days_held", 0), -x.get("composite", 0)))
    return active


def _build_exit_pending(sources: dict) -> list[dict]:
    """Build list of positions with EXIT_PENDING state.
    Uses LLM exit_debate_results from swarm cache if available (Phase 5.6b).
    """
    import time
    today = time.strftime("%Y-%m-%d")
    positions = sources.get("positions") or {}
    scan_lookup = sources.get("scan_lookup") or {}
    price_data = sources.get("price_data") or {}
    # Exit debate verdicts from swarm Phase 5.6b (keyed by "ticker::horizon")
    exit_debate = (
        (sources.get("pm_swarm") or {})
        .get("phase5_pm", {})
        .get("exit_debate_results", {})
    )

    pending = []
    for key, pos in positions.items():
        if pos.get("state") != "EXIT_PENDING":
            continue
        ticker, horizon = key.split("::", 1)
        scan_row = scan_lookup.get(ticker, {})
        entered_date = pos.get("entered_date")
        first_seen = pos.get("first_seen") or entered_date
        days_held = _days_between(entered_date, today) if entered_date else 0
        persistence_days = _days_between(first_seen, today) if first_seen else 0

        trailing = _compute_trailing_returns(ticker, price_data, scan_lookup.get(ticker))

        # Risk evaluation — match HOLDING records so frontend renders Risk Reason column
        from agents.risk_manager import compute_risk_score, risk_vote, risk_rationale
        risk_breakdown = compute_risk_score(scan_row, sources.get("sector_counts") or {})
        risk_v = risk_vote(risk_breakdown)

        exit_reason = (pos.get("state_history") or [{}])[-1].get("reason", "")[:200]

        # LLM exit debate 결과 (swarm Phase 5.6b)
        ed = exit_debate.get(f"{ticker}::{horizon}") or {}
        has_llm_verdict = bool(ed and ed.get("verdict") and not ed.get("_debate_failed"))

        if has_llm_verdict:
            _verdict = ed.get("verdict", "EXIT_NOW")        # EXIT_NOW / PARTIAL_EXIT / HOLD_1W
            _confidence = ed.get("confidence", "MEDIUM")
            _key_factor = ed.get("key_factor", "")
            _debate_transcript = (
                f"🤖 Exit Debate 결과 ({_verdict} / {_confidence}): "
                f"{ed.get('debate_transcript', '')} "
                f"[청산근거: {ed.get('exit_case', '')}]"
                f"{' [보유근거: ' + ed.get('hold_case', '') + ']' if ed.get('hold_case') else ''}"
            )
            _tier = {
                "EXIT_NOW": "UNANIMOUS_EXIT",
                "PARTIAL_EXIT": "MAJORITY_EXIT",
                "HOLD_1W": "HOLD_PENDING",
            }.get(_verdict, "STATE_DRIVEN_EXIT")
            _final_decision = "EXIT" if _verdict in ("EXIT_NOW", "PARTIAL_EXIT") else "HOLD"
        else:
            _verdict = "EXIT_NOW"
            _confidence = "LOW"
            _key_factor = "기계적 청산"
            _debate_transcript = (
                f"⚙️ 기계적 청산 신호 (exit debate 미실행 또는 실패) — "
                f"{exit_reason or 'state machine 청산 판정'}. "
                f"position_state 규칙(thesis-break): HOLDING은 논리가 깨질 때만 청산 — "
                f"분류 약세 전환 / 스톱로스 이탈 / 레짐 플립(SHORT 전환) / composite<45(21일+확인). "
                f"단순 미선정(SKIP)·WAIT은 청산 아님. 보유 {days_held}일째."
            )
            _tier = "STATE_DRIVEN_EXIT"
            _final_decision = "EXIT"

        pending.append({
            "ticker": ticker,
            "name": (scan_row.get("name") or "")[:40],
            "sector": scan_row.get("sector") or scan_row.get("category", ""),
            "composite": float(scan_row.get("composite") or 0),
            "horizon": horizon,
            "bucket": _infer_bucket(scan_row, side="long"),
            "state": "EXIT_PENDING",
            "tier": _tier,
            "consensus": _tier,
            "exit_verdict": _verdict,            # EXIT_NOW / PARTIAL_EXIT / HOLD_1W
            "exit_confidence": _confidence,      # HIGH / MEDIUM / LOW
            "entered_date": entered_date,
            "first_seen": first_seen,
            "days_held": days_held,
            "persistence_days": persistence_days,
            "exit_reason": exit_reason,
            "rationale": ed.get("exit_case") or exit_reason or "포지션 청산 후보",
            "debate_transcript": _debate_transcript,
            "key_factor": _key_factor,
            "final_decision": _final_decision,
            "current_signal": "—",
            "trading_rationale": ed.get("exit_case") or "state machine EXIT_PENDING 판정.",
            "risk_score": risk_breakdown["total"],
            "risk_vote": risk_v,
            "risk_reason": risk_rationale(risk_breakdown),
            "in_today_buy_picks":  False,
            "in_today_sell_picks": False,
            "last_alert": pos.get("last_alert"),
            **trailing,
        })
    pending.sort(key=lambda x: -x.get("days_held", 0))
    return pending


def _promote_holding_short_overlap(active_positions: list[dict],
                                    exit_pending: list[dict],
                                    sell_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """Promote HOLDING/ENTERED positions that received SHORT signals to EXIT_PENDING.

    Returns (filtered_active_positions, augmented_exit_pending).
    A position currently held LONG that PM swarm flags as SHORT today is a
    "regime-flip exit" — surface it in EXIT_PENDING so the user closes the long
    instead of being confused by it appearing as both HOLDING and SHORT-buy.
    """
    # Map of SHORT-recommended tickers → first matching short pick (any horizon)
    short_picks: dict[str, dict] = {}
    for r in sell_list:
        # Skip records that are themselves position-state exits (not new SHORT picks)
        if r.get("action") == "CLOSE_TODAY" or r.get("state") in ("EXIT_PENDING", "EXITED"):
            continue
        # Only treat sell-side PM picks (bucket starts with short_) as fresh SHORT signals
        if not str(r.get("bucket", "")).startswith("short_"):
            continue
        short_picks.setdefault(r["ticker"], r)

    if not short_picks:
        return active_positions, exit_pending

    already_pending = {r["ticker"] for r in exit_pending}
    kept_active: list[dict] = []
    promoted: list[dict] = []

    for pos in active_positions:
        t = pos.get("ticker")
        if t in short_picks and t not in already_pending and pos.get("state") in ("HOLDING", "ENTERED"):
            sp = short_picks[t]
            # Pull the SHORT pick's swarm debate (it DID go through Phase 5 + per-ticker debate)
            sp_debate = (sp.get("debate_synthesis") or sp.get("_pt_debate") or {})
            sp_transcript = (sp.get("debate_transcript")
                              or sp_debate.get("debate_transcript")
                              or sp.get("rationale") or "")
            sp_tier = sp.get("tier") or sp_debate.get("tier") or "SOLO"
            # Build a regime-flip explanation that combines the SHORT-side debate
            flip_transcript = (
                f"🔻 Regime-Flip 청산 — 보유 LONG이 오늘 PM swarm에서 SHORT으로 재판정됨. "
                f"[SHORT 토론 ★{sp.get('stars',1)} / {sp_tier}]: {sp_transcript[:280]}"
            )
            sp_votes = sp.get("votes") or {}
            promoted.append({
                "ticker": t,
                "name": pos.get("name", ""),
                "sector": pos.get("sector", ""),
                "composite": pos.get("composite", 0),
                "horizon": pos.get("horizon", "core"),
                "bucket": pos.get("bucket") or "long_stocks",   # inherit from active position
                "state": "EXIT_PENDING",
                "tier": "REGIME_FLIP",
                "consensus": "REGIME_FLIP",
                "entered_date": pos.get("entered_date"),
                "first_seen": pos.get("first_seen") or pos.get("entered_date"),
                "days_held": pos.get("days_held", 0),
                "persistence_days": pos.get("persistence_days", 0),
                "exit_reason": f"보유 중 LONG + 오늘 PM SHORT 시그널 감지 (★{sp.get('stars', 1)} / "
                               f"{sp.get('action', '?')}) — regime flip 청산 권장",
                "rationale": (sp.get("rationale") or sp_transcript or "")[:300]
                              or "Regime flip: 보유 LONG + 오늘 PM SHORT 시그널",
                # Carry the SHORT-side swarm debate so the transcript column isn't blank
                "debate_transcript": flip_transcript,
                "key_factor": "Regime-Flip (LONG→SHORT)",
                "final_decision": "EXIT",
                "votes": sp_votes if sp_votes else None,
                "n_approve": sp.get("n_approve"),
                "n_reject": sp.get("n_reject"),
                "n_caution": sp.get("n_caution"),
                "current_signal": "SELL",   # PM flagged this as SHORT today
                "trading_rationale": (
                    f"오늘 PM swarm이 SHORT 시그널 발생 (★{sp.get('stars',1)}, {sp.get('horizon','?')} horizon) "
                    f"→ 보유 LONG 청산 권고. SHORT 측 Trading 판단: {(sp.get('trading_rationale') or '추세 하방 전환')[:150]}"
                ),
                "promoted_from_short": True,
                "short_signal_stars": sp.get("stars", 1),
                "short_signal_action": sp.get("action", "?"),
                "short_signal_horizon": sp.get("horizon", "?"),
                "short_signal_reason": (sp.get("rationale") or sp_transcript or "")[:300],
                "last_alert": pos.get("last_alert"),
                # Returns — inherit from active position (already populated via _compute_trailing_returns)
                "ret_5d":  pos.get("ret_5d"),
                "ret_1mo": pos.get("ret_1mo"),
                "ret_3mo": pos.get("ret_3mo"),
                "ret_6mo": pos.get("ret_6mo"),
                "ret_1y":  pos.get("ret_1y"),
                # Risk — inherit from active position
                "risk_score":  pos.get("risk_score"),
                "risk_vote":   pos.get("risk_vote"),
                "risk_reason": pos.get("risk_reason"),
                "in_today_buy_picks":  False,
                "in_today_sell_picks": True,
            })
        else:
            kept_active.append(pos)

    # Promoted items go to top of exit_pending (more urgent than passive state-driven exits)
    return kept_active, promoted + exit_pending


def _debate_health(sw: dict) -> dict:
    """per-ticker debate 건강도 — 전량/대량 실패 시 3-agent 검증층이 무력화되고
    매수 리스트가 사실상 composite 랭킹으로 퇴화하므로 표면화한다(2026-07-27 무통보
    50/50 실패 사례). n_failed/n_total(phase5_pm.per_ticker_debate_summary) 기반."""
    p5 = sw.get("phase5_pm") or {}
    s = p5.get("per_ticker_debate_summary") or {}
    n_total = int(s.get("n_total") or 0)
    n_failed = int(s.get("n_failed") or 0)
    rate = (n_failed / n_total) if n_total else 0.0
    # 엔진 전체 크래시 감지 (리뷰 수정): run_per_ticker_debate가 raise하면 summary가 아예
    # 없고(n_total=0) per_ticker_debate_error만 set된다 → 개별실패(rate)와 달리 여기서
    # 잡지 않으면 'unknown'으로 빠져 배너가 침묵(바로 우리가 막으려던 사일런트 실패).
    _err = p5.get("per_ticker_debate_error")
    _hz = p5.get("horizons") or {}
    _has_picks = any((_hz.get(h, {}) or {}).get(bk) for h in ("core",)
                     for bk in ("long_stocks", "long_etfs"))
    if n_total == 0 and (_err or _has_picks):
        return {"status": "degraded", "n_failed": 0, "n_total": 0, "failure_rate": 1.0,
                "degraded": True,
                "note": (f"⚠️ per-ticker debate 엔진 전체 실패"
                         + (f" ({str(_err)[:80]})" if _err else "")
                         + " — 3-agent 검증층 무력화, 매수 리스트가 composite 랭킹으로 퇴화. 스웜 재실행 권장.")}
    if n_total == 0:
        status, degraded, note = "unknown", False, "debate 결과 없음 (스웜 미실행/캐시 없음)"
    elif rate >= 0.6:
        status, degraded = "degraded", True
        note = (f"⚠️ per-ticker debate {n_failed}/{n_total} 실패 — 3-agent 검증층이 무력화되어 "
                f"매수 리스트가 사실상 composite 랭킹으로 퇴화. 스웜 재실행 권장.")
    elif rate >= 0.25:
        status, degraded = "partial", False
        note = f"debate {n_failed}/{n_total} 실패 — 일부 픽이 composite 오버라이드로 결정됨."
    else:
        status, degraded = "ok", False
        note = f"debate 정상 ({n_total - n_failed}/{n_total} 성공)."
    return {"status": status, "n_failed": n_failed, "n_total": n_total,
            "failure_rate": round(rate, 3), "degraded": degraded, "note": note}


def build_final_lists() -> dict:
    """Top-level builder. Returns {buy_list, sell_list, metadata}."""
    sources = _load_sources()
    sw = sources["pm_swarm"]
    horizons = (sw.get("phase5_pm") or {}).get("horizons") or {}

    buy_list: list[dict] = []
    sell_list: list[dict] = []

    for h in ("core",):
        h_data = horizons.get(h) or {}
        # BUY side
        for bucket in ("long_stocks", "long_etfs"):
            for p in (h_data.get(bucket) or []):
                rec = _eval_buy_pick_with_debate(p, h, bucket, sources)
                if rec: buy_list.append(rec)
        # SELL side (SHORT picks = bet against)
        for bucket in ("short_stocks", "short_etfs"):
            for p in (h_data.get(bucket) or []):
                rec = _eval_sell_pick(p, h, bucket, sources)
                if rec: sell_list.append(rec)

    # ALSO add exit candidates from position state (existing LONG to close)
    for key, pos in (sources["positions"] or {}).items():
        if pos.get("state") in ("EXIT_PENDING", "EXITED"):
            t, h = key.split("::", 1)
            sell_list.append(_exit_pick(t, h, pos))

    # Sort: stars desc → urgency (URGENT first) → **Entry Timing status** → composite desc
    URG_ORDER = {"URGENT": 0, "NORMAL": 1, "PATIENT": 2}
    ACT_ORDER = {"EXECUTE_TODAY": 0, "CLOSE_TODAY": 0, "WATCH_TOMORROW": 1, "ALREADY_HELD": 2, "OBSERVE": 3}
    # ①②⑤: 동일 확신(stars/action/urgency) 안에서 FRESH(적시)를 EXTENDED(스트레치)보다
    # 위로. 확신 서열은 그대로 두고(신호를 뒤엎지 않음), 같은 급이면 "지금 사기 좋은" 종목
    # 먼저 노출. 라벨은 아래에서 그대로 프론트에 전달돼 "즉시 매수 vs 눌림목 대기" 구분.
    _ET_ORDER = {"FRESH": 0, "NEUTRAL": 1, "EXTENDED": 2}
    def _et_rank(x):
        return _ET_ORDER.get(x.get("entry_timing_status") or "NEUTRAL", 1)
    # Trend Freshness 타이브레이커 — 동일 (stars/action/urgency/ET) 내에서 신선 추세를
    # 성숙 추세보다 위로 (walk-forward 방향 일관, _trend_freshness docstring 참조).
    _TF_ORDER = {"FRESH": 0, None: 1, "MATURE": 2}
    def _tf_rank(x):
        return _TF_ORDER.get(x.get("trend_freshness"), 1)
    # Herding-Reversal 타이브레이커 (paper 2607.27063) — 동일 순위 내에서 과열 후 반전위험
    # (과밀+herding약화 conjunction) 종목을 아래로. 검증상 희귀·고확신 → TIE-BREAKER 전용
    # (하드제외 안 함; 발동 시에도 리스트에는 남되 동점 시 후순위). WATCH < REVERSAL_RISK.
    _HRR_ORDER = {"REVERSAL_RISK": 2, "WATCH": 1}
    def _hrr_rank(x):
        return _HRR_ORDER.get(x.get("hrr_flag"), 0)
    # NULL-safe: picks can have stars/composite = None → guard against -None crash
    # that previously failed the whole build → empty buy_list.
    buy_list.sort(key=lambda x: (-(x.get("stars") or 0), ACT_ORDER.get(x.get("action"), 9),
                                  URG_ORDER.get(x.get("urgency"), 9), _et_rank(x), _hrr_rank(x),
                                  _tf_rank(x), -(x.get("composite") or 0)))
    sell_list.sort(key=lambda x: (-(x.get("stars") or 0), ACT_ORDER.get(x.get("action"), 9),
                                   URG_ORDER.get(x.get("urgency"), 9), -(x.get("composite") or 0)))

    # Dedup by (ticker, bucket) — same ticker may appear in multiple horizons
    def _dedup(lst):
        seen = set(); out = []
        for r in lst:
            k = (r["ticker"], r["bucket"])
            if k in seen: continue
            seen.add(k); out.append(r)
        return out

    buy_dedup = _dedup(buy_list)
    sell_dedup = _dedup(sell_list)

    # Build active_positions section — currently HELD tickers (regardless of today's voting)
    active_positions = _build_active_positions(sources, buy_dedup, sell_dedup)
    exit_pending = _build_exit_pending(sources)

    # Promote HOLDING + today's SHORT signal → EXIT_PENDING (regime-flip exit)
    # Eliminates user confusion of seeing same ticker as HOLDING and Sell candidate.
    active_positions, exit_pending = _promote_holding_short_overlap(
        active_positions, exit_pending, sell_dedup
    )

    # Build category groups for per-category commentaries (mirror frontend merge logic)
    held_set = {p["ticker"] for p in active_positions if p.get("state") in ("HOLDING", "ENTERED")}
    new_picks = [r for r in buy_dedup
                  if r.get("state") != "ENTERED" and r.get("action") != "EXECUTE_TODAY"
                  and r.get("ticker") not in {p["ticker"] for p in active_positions if p.get("state") == "HOLDING"}]

    # ── Holdings-aware re-ranking (방식 A+B) ──
    # Re-score NEW candidates by considering the existing portfolio:
    #   A) sector concentration cap — penalize picks piling into crowded sectors
    #   B) correlation penalty — penalize picks highly correlated with held names
    # This surfaces diversifying picks (unheld sectors, low-correlation) ahead of
    # redundant ones (e.g. EEM when IEMG already held, ρ≈0.99).
    try:
        from agents.holdings_aware_selection import rerank_new_picks, summarize_reranking
        held_for_rerank = [p for p in active_positions if p.get("state") in ("HOLDING", "ENTERED")]
        if new_picks and held_for_rerank:
            rerank_new_picks(new_picks, held_for_rerank, sources.get("price_data") or {})
            sources["_ha_rerank_summary"] = summarize_reranking(new_picks)
    except Exception as _ha_err:
        sources["_ha_rerank_error"] = str(_ha_err)[:200]

    items_by_category_full = {
        "ENTERED":      [r for r in buy_dedup if r.get("state") == "ENTERED" and r.get("action") == "EXECUTE_TODAY"],
        "EXIT_PENDING": exit_pending,
        "HOLDING":      [r for r in active_positions if r.get("state") == "HOLDING"],
        "NEW":          new_picks,
    }

    # ── Turnover cap: GLOBAL 20 stocks + 20 ETFs across the WHOLE Buy Final List ──
    # Asset-class balance: top 20 stocks + top 20 ETFs by quality score (regardless of category).
    # Each ticker keeps its category assignment; tickers below the cap are filtered out
    # of all categories. Replaces the previous "5 per category" cap.
    STOCK_CAP = 20
    ETF_CAP = 20
    # EXIT_PENDING tickers represent URGENT sell signals — bypass the 20-cap.
    # Up to this many EXIT_PENDING entries always surface (per asset type).
    EXIT_PENDING_CAP = 15

    def _score_for_cap(r: dict) -> float:
        # 기존 stars*100 + bare composite 문제: stars가 ~90% 종목에서 3에 군집 → 이진
        # 게이트로 작동, cap이 결국 bare composite로만 재정렬 → debate tier/agent 합의/risk
        # 등 풍부한 conviction 맥락 폐기. 개선: stars는 약한 정렬가중(×35)으로 낮추고,
        # debate tier + agent 합의 + risk를 함께 반영해 conviction-richness로 랭킹.
        promo = 50 if r.get("promoted_from_short") else 0
        days = (r.get("days_held") or 0) * 0.1
        # NEW picks는 holdings-aware adjusted score 보유 (sector 분산 반영) — 강도 base로 사용
        ha = r.get("ha_adjusted_score")
        base = float(ha) if ha is not None else float(r.get("composite") or 0)
        tier_bonus = {"UNANIMOUS": 12, "MAJORITY_CLEAN": 8, "MAJORITY_DISSENT": 4,
                      "SOLO_CLEAN": 3, "SOLO": 0, "ALL_CAUTION": -4}.get(
                          r.get("tier") or r.get("consensus") or "", 0)
        votes = ((r.get("n_approve") or 0) - (r.get("n_reject") or 0)) * 4
        risk_pen = max(0.0, float(r.get("risk_score") or 50) - 55.0) * 0.3
        return (r.get("stars") or 0) * 35 + base + tier_bonus + votes - risk_pen + days + promo

    def _is_etf(r: dict) -> bool:
        bk = (r.get("bucket") or "").lower()
        if "etf" in bk: return True
        if "stock" in bk: return False
        at = (r.get("asset_type") or r.get("type") or "").lower()
        return "etf" in at

    # 1) Flatten all categories with category label attached + dedupe by ticker (highest score wins)
    flat: dict[str, tuple[str, dict]] = {}
    cat_priority = {"EXIT_PENDING": 0, "ENTERED": 1, "HOLDING": 2, "NEW": 3}
    for cat, rows in items_by_category_full.items():
        for r in rows:
            t = r.get("ticker")
            if not t: continue
            score = _score_for_cap(r)
            if t in flat:
                old_cat, old_r = flat[t]
                # Higher-priority (more urgent) category wins; tiebreak on score
                p_new = cat_priority.get(cat, 9)
                p_old = cat_priority.get(old_cat, 9)
                if p_new < p_old or (p_new == p_old and score > _score_for_cap(old_r)):
                    flat[t] = (cat, r)
            else:
                flat[t] = (cat, r)

    # 2) Split EXIT_PENDING out — they get reserved slots, not subject to global cap.
    flat_items = [(cat, r) for cat, r in flat.values()]
    exit_pending_stocks = sorted(
        [(cat, r) for cat, r in flat_items if cat == "EXIT_PENDING" and not _is_etf(r)],
        key=lambda x: -_score_for_cap(x[1]))[:EXIT_PENDING_CAP]
    exit_pending_etfs = sorted(
        [(cat, r) for cat, r in flat_items if cat == "EXIT_PENDING" and _is_etf(r)],
        key=lambda x: -_score_for_cap(x[1]))[:EXIT_PENDING_CAP]

    # 3) Apply the 20+20 cap to non-EXIT_PENDING (entered/holding/new) only.
    other_items = [(cat, r) for cat, r in flat_items if cat != "EXIT_PENDING"]
    stocks = sorted([(cat, r) for cat, r in other_items if not _is_etf(r)],
                    key=lambda x: -_score_for_cap(x[1]))[:STOCK_CAP]
    etfs   = sorted([(cat, r) for cat, r in other_items if _is_etf(r)],
                    key=lambda x: -_score_for_cap(x[1]))[:ETF_CAP]
    kept = stocks + etfs + exit_pending_stocks + exit_pending_etfs

    # 3) Redistribute back into categories (whitelist)
    items_by_category: dict[str, list[dict]] = {"ENTERED": [], "EXIT_PENDING": [], "HOLDING": [], "NEW": []}
    for cat, r in kept:
        items_by_category.setdefault(cat, []).append(r)
    # Stable sort within each category by score
    for cat in items_by_category:
        items_by_category[cat].sort(key=_score_for_cap, reverse=True)

    # Update persistence_days on buy/sell items
    try:
        from agents.trade_log import update_log_from_positions, get_persistence_days
        update_log_from_positions(sources.get("positions") or {})
        for r in buy_dedup:
            r["persistence_days"] = get_persistence_days(r["ticker"])
        for r in sell_dedup:
            r["persistence_days"] = get_persistence_days(r["ticker"])
    except Exception:
        pass

    # Executive Commentary — UNIFIED 10,000-char.
    # OPTIMIZATION: only return CACHED commentary in the API path. If the cache is stale,
    # serve last-known commentary (or empty placeholder) and let a separate background job
    # regenerate it. This keeps /api/final-list response under 1 second even when commentary
    # needs to be rebuilt (which takes 3-5 minutes with the 10,000-char prompt + WebSearch).
    commentary = {}
    category_commentary = {}  # deprecated; kept empty for backward compat
    try:
        from agents.final_list_commentary import build_executive_commentary
        pm_commentary = (sw.get("phase5_pm") or {}).get("pm_commentary", "")[:600]
        commentary = build_executive_commentary(
            buy_dedup, sell_dedup,
            market_context=pm_commentary,
            items_by_category=items_by_category,
            swarm_generated_at=sw.get("generated_at", ""),
            cache_only=True,   # ← never trigger LLM call inside the API path
        )
    except Exception as e:
        commentary = {"unified_commentary": "", "buy_commentary": "", "sell_commentary": "",
                      "unified_cached": False, "buy_cached": False, "sell_cached": False,
                      "error": str(e)[:200]}

    # Compute capped ticker set per category (frontend uses to filter mergedItems → top-5 cap)
    capped_tickers_by_category = {
        cat: [r.get("ticker") for r in rows if r.get("ticker")]
        for cat, rows in items_by_category.items()
    }

    return {
        "buy_list":  buy_dedup,
        "sell_list": sell_dedup,
        "active_positions": active_positions,
        "exit_pending":     exit_pending,
        "commentary": commentary,
        "category_commentary": category_commentary,
        "items_by_category": items_by_category,        # NEW: capped top-5 per cat (canonical)
        "capped_tickers_by_category": capped_tickers_by_category,  # NEW: ticker whitelist for frontend
        "metadata": {
            "proxy_cohort_date": sources["proxy_cohort_date"],
            "n_proxy_long_stocks": len(sources["proxy_long_stocks"]),
            "n_proxy_long_etfs":   len(sources["proxy_long_etfs"]),
            "n_top_stocks": len(sources["top_stocks"]),
            "n_top_etfs":   len(sources["top_etfs"]),
            "n_positions_tracked": len(sources["positions"]),
            "swarm_generated_at": sw.get("generated_at", ""),
            # Holdings-aware NEW pick re-ranking summary (방식 A+B)
            "holdings_aware_rerank": sources.get("_ha_rerank_summary"),
            # per-ticker debate 건강도 — degraded면 리스트가 composite 랭킹으로 퇴화 (표면화)
            "debate_health": _debate_health(sw),
        },
    }
