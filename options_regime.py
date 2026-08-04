"""Options-market regime layer (Tier 1) — market-level positioning/crowding gauge.

Complements the price-based `market_internals` and the economic `macro_regime`
with the ONE dimension neither sees: **options positioning** (fear/hedging
extremes + dealer vol structure). The thesis (user, 2026-07): extreme crowding
+ its unwind is where sharp early momentum is born; the volatility term structure
and skew are the market-level tells.

Signals (all free via yfinance, EOD/near-real-time):
  • VIX term structure — VIX9D / VIX(30d) / VIX3M(90d). Backwardation
    (short-dated > long-dated) = stress/capitulation → mean-reversion bounce setup.
  • SKEW (CBOE ^SKEW) — price of tail hedging; high percentile = crowded downside
    protection → skew-unwind relief-rally potential.
  • VVIX — vol-of-vol; elevated = unstable vol regime.
  • Put/Call OI ratio — from a near-term (non-0DTE) SPY chain; high = fear/hedging.

CAVEATS: yfinance option OI is EOD/T+1 (not intraday); a 1h-cached daily overlay,
never an intraday trigger. Index-level only here (Tier 1) — per-ticker dealer
gamma (GEX) is Tier 2/3.
"""
from __future__ import annotations

import json
import os
import threading
import time
import warnings
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

_CACHE_PATH = ".options_regime_cache.json"
_TTL_SEC = 3600            # 1h — VIX moves intraday but a slow positioning overlay is fine
_lock = threading.Lock()
_mem: dict = {}

_VOL_TICKERS = {"^VIX": "vix", "^VIX3M": "vix3m", "^VIX9D": "vix9d",
                "^VVIX": "vvix", "^SKEW": "skew"}


def _pctile(series, value) -> Optional[float]:
    """Percentile rank (0-100) of `value` within a trailing series."""
    try:
        import numpy as np
        s = series.dropna().values
        if len(s) < 30:
            return None
        return round(float((s < value).mean()) * 100, 0)
    except Exception:
        return None


def _fetch_vol(force_start_years=3):
    """Batch-fetch the vol complex → {code: (latest, pandas Series history)}.
    Retries (yfinance sqlite db-lock is transient → 빈결과 방지, market_internals 교훈)."""
    import yfinance as yf
    import pandas as pd
    out = {}
    for _attempt in range(3):
        try:
            raw = yf.download(list(_VOL_TICKERS), period=f"{force_start_years}y",
                              interval="1d", progress=False, auto_adjust=False, threads=True)
            close = raw["Close"] if hasattr(raw, "columns") and "Close" in getattr(raw, "columns", []) else raw
            if isinstance(close, pd.DataFrame):
                for t in _VOL_TICKERS:
                    if t in close.columns:
                        s = close[t].dropna()
                        if len(s):
                            out[t] = (float(s.iloc[-1]), s)
            # core complex(^VIX/^VIX3M/^SKEW) 확보되면 성공 — 아니면 재시도
            if {"^VIX", "^VIX3M", "^SKEW"} <= set(out):
                break
        except Exception:
            pass
        time.sleep(1.5)
    return out


def _put_call_oi(min_days=7, max_days=45):
    """Put/Call open-interest ratio from a near-term (non-0DTE) SPY chain.
    Picks the expiry closest to ~30d in [min_days, max_days] to avoid 0DTE distortion."""
    try:
        import yfinance as yf
        from datetime import date
        spy = yf.Ticker("SPY")
        exps = spy.options or []
        today = date.today()
        cand = []
        for e in exps:
            try:
                d = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
                if min_days <= d <= max_days:
                    cand.append((abs(d - 30), e))
            except Exception:
                continue
        if not cand:
            return None, None
        # 단일 만기는 일회성 대형 헤지블록에 흔들림 → [min,max]일 내 만기 전체 OI 집계.
        exps_in = [e for _, e in sorted(cand)][:4]   # 최대 4개 만기 (API콜 절제)
        put_oi = call_oi = 0.0
        for e in exps_in:
            try:
                oc = spy.option_chain(e)
                put_oi += float(oc.puts["openInterest"].fillna(0).sum())
                call_oi += float(oc.calls["openInterest"].fillna(0).sum())
            except Exception:
                continue
        if call_oi <= 0:
            return None, None
        sd = sorted(exps_in)   # 표시는 날짜순
        span = f"{sd[0]}~{sd[-1]}" if len(sd) > 1 else sd[0]
        return round(put_oi / call_oi, 2), span
    except Exception:
        return None, None


def _build(vol, pc, pc_exp):
    import math
    g = lambda code: vol.get(code, (None, None))[0]
    hist = lambda code: vol.get(code, (None, None))[1]
    vix, vix3m, vix9d = g("^VIX"), g("^VIX3M"), g("^VIX9D")
    vvix, skew = g("^VVIX"), g("^SKEW")

    signals = []
    # ── VIX term structure ── 표준 커브 역전 = 30d(VIX) > 90d(VIX3M). ratio와 state를
    # 동일 tenor(30d/90d)로 계산해 불일치 방지 + 3% 데드밴드로 노이즈 억제(단일 프린트로
    # 최상위 'backwardation' 라벨이 뒤집히지 않게). VIX9D는 급성 이벤트 스트레스 보조플래그만.
    ts_ratio, ts_state = None, "unknown"
    if vix and vix3m and vix3m > 0:
        ts_ratio = round(vix / vix3m, 3)
        if ts_ratio >= 1.03:
            ts_state = "backwardation"      # 스트레스/캐피출레이션
        elif ts_ratio <= 0.90:
            ts_state = "steep_contango"     # 매우 정상/복지부동
        else:
            ts_state = "contango"
        acute_9d = bool(vix9d and vix and vix9d > vix * 1.05)   # 9d 급등 = 급성 이벤트 스트레스
        signals.append({"code": "VIX_TS", "label": "VIX 기간구조(30d/3M)",
                        "value": ts_ratio, "state": ts_state, "acute_9d": acute_9d,
                        "detail": f"9D {vix9d} · 30D {vix} · 3M {vix3m}"})
    # ── SKEW (tail hedging crowding) ──
    skew_pct = _pctile(hist("^SKEW"), skew) if skew else None
    if skew is not None:
        sk_state = "crowded" if (skew_pct or 0) >= 80 else "elevated" if skew >= 140 else "normal"
        signals.append({"code": "SKEW", "label": "꼬리위험 헤지(SKEW)",
                        "value": round(skew, 1), "pctile": skew_pct, "state": sk_state})
    # ── VVIX (vol of vol) ──
    vvix_pct = _pctile(hist("^VVIX"), vvix) if vvix else None
    if vvix is not None:
        # 백분위 전용 — VVIX는 평시에도 100-105라 절대임계는 오발(regime 엔진도 pctile 사용).
        vv_state = "high" if (vvix_pct or 0) >= 75 else "normal"
        signals.append({"code": "VVIX", "label": "변동성의 변동성(VVIX)",
                        "value": round(vvix, 1), "pctile": vvix_pct, "state": vv_state})
    # ── Put/Call OI (fear/hedging) — index P/C runs structurally high (헤지수요)
    #    so use INDEX-appropriate bands (fear ≥1.9, complacent ≤1.1). 표시·확인용,
    #    regime의 단독 트리거는 아님(SKEW 백분위가 crowded-hedge의 주신호). ──
    if pc is not None:
        pc_state = "fear" if pc >= 1.9 else "complacent" if pc <= 1.1 else "neutral"
        signals.append({"code": "PC_OI", "label": f"풋콜 OI비(SPY {pc_exp})",
                        "value": pc, "state": pc_state})

    # ── Composite regime + unwind-setup flag (user's core interest) ──
    # crowded_hedge = SKEW 백분위 주도(≥80); 풋콜은 극단(≥1.9) 확인일 때만 보강.
    # ★falsy-zero 방지: pctile==0.0(3년 최저=최극단 안일)이 `or 100`에 삼켜지지 않게 명시 None처리.
    _sk = 100.0 if skew_pct is None else skew_pct
    _vv = 100.0 if vvix_pct is None else vvix_pct
    backwardation = ts_state == "backwardation"
    crowded_hedge = (_sk >= 80) or (_sk >= 65 and pc is not None and pc >= 1.9)
    complacent = (ts_state == "steep_contango") and (_sk < 40) and (_vv < 50)
    vvix_high = (vvix_pct or 0) >= 75

    if backwardation:
        label, color = "스트레스/캐피출레이션", "green"
        rationale = ("VIX 기간구조 백워데이션(단기 IV > 장기) — 공포·캐피출레이션 국면. "
                     "포지셔닝 극단에서 되돌림(바운스) 시 초기 모멘텀 강할 여지 ★unwind 셋업.")
        unwind = True
    elif crowded_hedge:
        label, color = "과밀 하방헤지", "amber"
        rationale = ("SKEW/풋콜 극단 — 하방 헤지가 과밀. 우려가 해소되면 스큐 언와인드 "
                     "+ 숏 헤지 청산으로 릴리프 랠리 여지 ★unwind 셋업(방향 확인 필요).")
        unwind = True
    elif complacent:
        label, color = "컴플레이선시(복지부동)", "yellow"
        rationale = ("깊은 콘탱고 + 낮은 SKEW/VVIX — 헤지 부재·안일. 충격에 취약(비대칭 하방 리스크).")
        unwind = False
    else:
        label, color = "중립", "gray"
        rationale = "옵션 포지셔닝 특별한 극단 없음."
        unwind = False

    asof = None
    for code in _VOL_TICKERS:
        h = hist(code)
        if h is not None and len(h):
            asof = h.index[-1].date().isoformat(); break

    _rnd = lambda v: round(v, 2) if isinstance(v, (int, float)) else v
    return {
        "as_of": asof or datetime.today().date().isoformat(),
        "vix": _rnd(vix), "vix3m": _rnd(vix3m), "vix9d": _rnd(vix9d),
        "vvix": _rnd(vvix), "skew": _rnd(skew),
        "term_structure": {"ratio": ts_ratio, "state": ts_state},
        "signals": signals,
        "regime": {"label": label, "color": color, "rationale": rationale,
                   "unwind_setup": unwind, "complacent": complacent, "vvix_high": vvix_high},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def compute_options_regime(force: bool = False) -> dict:
    """Main entry. Options-market positioning regime. Cached 1h (mem+disk), graceful
    degradation to last-good cache on yfinance failure. NEVER caches an empty/failed
    result (a transient yfinance db-lock must not stick — same lesson as market_internals)."""
    now = time.time()
    if not force and _mem.get("data") and (now - _mem.get("ts", 0) < _TTL_SEC):
        return _mem["data"]
    with _lock:
        if not force and _mem.get("data") and (now - _mem.get("ts", 0) < _TTL_SEC):
            return _mem["data"]
        if not force and os.path.exists(_CACHE_PATH):
            try:
                with open(_CACHE_PATH) as f:
                    disk = json.load(f)
                if now - disk.get("_ts", 0) < _TTL_SEC:
                    _mem.update({"data": disk, "ts": disk.get("_ts", now)})
                    return disk
            except Exception:
                pass
        try:
            vol = _fetch_vol()
            # core complex 전체가 있어야 캐싱 — 부분페치(^SKEW/^VVIX 누락) stub을 1h 캐싱하지 않게.
            if not {"^VIX", "^VIX3M", "^SKEW"} <= set(vol):
                raise RuntimeError("vol complex incomplete (yfinance)")
            pc, pc_exp = _put_call_oi()
            payload = _build(vol, pc, pc_exp)
            payload["_ts"] = now
            _mem.update({"data": payload, "ts": now})
            try:
                tmp = _CACHE_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
                os.replace(tmp, _CACHE_PATH)
            except Exception:
                pass
            return payload
        except Exception as e:
            if _mem.get("data"):
                return _mem["data"]
            if os.path.exists(_CACHE_PATH):
                try:
                    with open(_CACHE_PATH) as f:
                        return json.load(f)
                except Exception:
                    pass
            return {"error": f"options regime unavailable: {e}", "signals": [], "regime": None}


if __name__ == "__main__":
    import pprint
    pprint.pprint(compute_options_regime(force=True))
