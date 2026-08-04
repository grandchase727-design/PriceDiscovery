"""market_internals.py — Market-internals lens for the dashboard.

Computes, from headline index / ratio trends (yfinance, 1h TTL cache):
  • Lens 1  rotation ratio matrix   (QQQ/DIA, IWF/IWD, IWM/SPY, RSP/SPY, SPY/ACWI) × 1M/3M/6M
  • Lens 2  index momentum + breadth (QQQ/SPY/DIA/IWM/ACWI/RSP/SMH: 1M/3M/6M + vs-200d)
  • Lens 3  cross-asset confirmation (SPY/TLT, HYG/LQD, CPER/GLD, XLK/XLP, VIX)
  • Lens 6  composite regime label  (risk-on/off × breadth × growth × value-rotation)

Exposed via /api/market-internals. Deterministic; no LLM.
"""
from __future__ import annotations
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

_TTL_SEC = 3600
_CACHE: dict = {"ts": 0.0, "data": None}

INDICES = ["QQQ", "SPY", "DIA", "IWM", "ACWI", "RSP", "SMH"]
# (label, numerator, denominator, axis-meaning)
RATIOS = [
    ("QQQ/DIA", "QQQ", "DIA", "테크 vs 가치·산업"),
    ("IWF/IWD", "IWF", "IWD", "성장 vs 가치"),
    ("IWM/SPY", "IWM", "SPY", "소형 vs 대형"),
    ("RSP/SPY", "RSP", "SPY", "breadth (등가 vs 시총)"),
    ("SPY/ACWI", "SPY", "ACWI", "미국 vs 세계"),
]
CROSS = [
    ("SPY/TLT", "SPY", "TLT", "주식 vs 채권 (위험선호)"),
    ("HYG/LQD", "HYG", "LQD", "하이일드 vs IG (크레딧)"),
    ("CPER/GLD", "CPER", "GLD", "구리 vs 금 (성장기대)"),
    ("XLK/XLP", "XLK", "XLP", "공격 vs 방어"),
]


def _r1(v) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 1)


def compute_market_internals(force: bool = False) -> dict:
    now = time.time()
    if not force and _CACHE["data"] is not None and (now - _CACHE["ts"]) < _TTL_SEC:
        return _CACHE["data"]

    tickers = sorted(set(INDICES
                         + [t for _, a, b, _ in RATIOS + CROSS for t in (a, b)]
                         + ["^VIX"]))
    # yfinance sqlite 캐시는 장수 서버에서 'database is locked'로 간헐 실패한다(→ 빈 프레임).
    # 전이적이므로 짧은 백오프로 재시도 — 이게 없으면 한 번의 락이 패널을 통째로 비운다.
    px, last_err = None, ""
    for _attempt in range(3):
        try:
            raw = yf.download(tickers, period="1y", interval="1d",
                              auto_adjust=True, progress=False, threads=True)
            _px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            _px = _px.dropna(how="all").ffill()
            if not _px.empty and "SPY" in _px.columns:
                px = _px
                break
            last_err = "empty/no-SPY frame"
        except Exception as e:
            last_err = str(e)[:120]
        time.sleep(1.5)
    if px is None:
        return {"error": f"no index data (retries exhausted: {last_err})", "as_of": None}

    as_of = str(px.index[-1].date())

    def mom(t: str, d: int) -> float | None:
        if t not in px.columns:
            return None
        s = px[t].dropna()
        return _r1((s.iloc[-1] / s.iloc[-d] - 1) * 100) if len(s) > d else None

    def rat_mom(a: str, b: str, d: int) -> float | None:
        if a not in px.columns or b not in px.columns:
            return None
        r = (px[a] / px[b]).dropna()
        return _r1((r.iloc[-1] / r.iloc[-d] - 1) * 100) if len(r) > d else None

    # Lens 2 — index momentum + vs-200d
    idx = []
    for t in INDICES:
        if t not in px.columns:
            continue
        s = px[t].dropna()
        sma200 = s.rolling(200).mean().iloc[-1] if len(s) >= 200 else np.nan
        idx.append({"t": t, "m1": mom(t, 21), "m3": mom(t, 63), "m6": mom(t, 126),
                    "vs200": _r1((s.iloc[-1] / sma200 - 1) * 100) if np.isfinite(sma200) else None})

    # Lens 1 — rotation ratio matrix (multi-horizon)
    rotation = [{"pair": lbl, "axis": ax,
                 "m1": rat_mom(a, b, 21), "m3": rat_mom(a, b, 63), "m6": rat_mom(a, b, 126)}
                for lbl, a, b, ax in RATIOS]

    # Lens 3 — cross-asset confirmation (3M)
    cross = [{"pair": lbl, "axis": ax, "m3": rat_mom(a, b, 63)} for lbl, a, b, ax in CROSS]
    vix = _r1(float(px["^VIX"].dropna().iloc[-1])) if "^VIX" in px.columns else None

    # Lens 6 — composite regime label
    risk_on   = (rat_mom("SPY", "TLT", 63) or 0) > 0        # equities beating bonds
    credit_ok = (rat_mom("HYG", "LQD", 63) or 0) > 0        # credit spreads stable/tightening
    growth    = (rat_mom("CPER", "GLD", 63) or 0) > 0       # copper beating gold  ← 백테스트 유효 렌즈
    breadth   = (rat_mom("RSP", "SPY", 63) or 0) >= -1.0    # equal-weight ~keeping up ← 백테스트 최유효
    small_cap = (rat_mom("IWM", "SPY", 63) or 0) > 0        # small-cap leadership ← 백테스트 유효
    value_rot = (rat_mom("IWF", "IWD", 63) or 0) < 0        # growth underperforming = value rotation
    vix_calm  = (vix or 99) < 22
    stressed  = (not risk_on) or (not credit_ok and (vix or 0) > 25)

    # ── Internals alignment score (0-5) — 백테스트: 정렬↑ = forward SPY 수익↑ · downside↓ ──
    # (breadth / small-cap / copper-growth 이 개별 최유효; VIX는 역상관이라 스코어에서 제외)
    align_score = sum([risk_on, credit_ok, growth, breadth, small_cap])

    if stressed:
        label, color = "Risk-Off / 방어 도피", "red"
        rat = "주식이 채권 대비 약세 또는 크레딧 악화+VIX 상승 — 리스크 축소 국면."
    elif risk_on and breadth and growth:
        if value_rot:
            label, color = "Risk-On 순환 (테크→가치·건강)", "green"
            rat = "위험선호+성장(구리)+breadth 유지 속 성장주 상대약세 = 상승장 내 건강한 순환."
        else:
            label, color = "Risk-On 광범위 강세", "green"
            rat = "위험선호+성장확인+breadth 양호 — 광범위 강세."
    elif risk_on and not breadth:
        label, color = "Risk-On Narrow (소수 견인·취약)", "amber"
        rat = "위험선호이나 breadth 축소 — 소수 대형주 견인, 집중 리스크."
    else:
        label, color = "혼조 / 전환 국면", "gray"
        rat = "신호 혼재 — 방향성 불명확, 확인 대기."

    regime = {
        "label": label, "color": color, "rationale": rat,
        "signals": {"risk_on": risk_on, "credit_ok": credit_ok, "growth": growth,
                    "breadth": breadth, "small_cap": small_cap,
                    "value_rotation": value_rot, "vix_calm": vix_calm},
        # 백테스트 근거: 정렬 스코어가 개별 렌즈보다 예측력↑ (ON 5/5 → fwd1M +2.2%·하방 −2.1% vs 0-2 → 하방 −10%)
        "alignment": {"score": align_score, "max": 5,
                      "note": "breadth·소형·구리·위험선호·크레딧 정렬 개수 (↑ = forward 수익↑·downside↓). VIX는 역상관이라 제외."},
        "vix_contrarian": ("낮음=과열주의(forward 수익↓)" if vix_calm else "높음=공포(forward 수익↑ 경향)"),
    }

    data = {"as_of": as_of, "indices": idx, "rotation": rotation,
            "cross_asset": cross, "vix": vix, "regime": regime}
    _CACHE.update(ts=now, data=data)
    return data
