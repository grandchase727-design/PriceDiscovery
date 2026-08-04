"""Macro-economic regime layer (Tier 1 + Tier 2).

Complements the *market-derived* regime (`api._detect_market_regime`, purely
cross-sectional over the universe's own Composite) and the price-based
`market_internals` with an **economic-cycle** regime from FRED data. Economic
data is orthogonal to price momentum → it can confirm/refute the market regime,
lead at turning points, and gate on recession risk.

  • Tier 1 — Growth × Inflation 4-quadrant from a z-score composite of FRED
    series (Reflation / Goldilocks / Stagflation / Slowdown).
  • Tier 2 — Markov-switching recession probability (Hamilton 1989) on a
    Conference-Board-style monthly coincident growth index.

Data: pandas_datareader → FRED public CSV (NO API key). We use SEASONALLY
ADJUSTED FRED series → X-13ARIMA-SEATS is unnecessary (its main job, seasonal
adjustment, is already done by the source; the x13as binary is also not
installed here). X-13 would only matter to self-adjust a raw NSA series.

CADENCE / CAVEATS (read before trusting this in a backtest):
  • Macro series are monthly (some daily/weekly), released with a 1–6 week lag
    and **heavily revised**. This layer is a SLOW OVERLAY (a tilt), never a
    daily trade trigger — the fast layer is `market_internals`.
  • The coincident index ends at its slowest component (real mfg&trade sales,
    ~3-month lag). `coincident_asof` reports the true as-of.
  • Values here are the CURRENT vintage. A faithful backtest must use real-time
    vintages (ALFRED) — final-revised data overstates signal quality.
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

_CACHE_PATH = ".macro_regime_cache.json"
_lock = threading.Lock()      # singleflight around recompute + _cache_mem access
_TTL_SEC = 6 * 3600          # 6h — macro data is monthly, but allow same-day re-reads cheaply
_FRED_START = datetime(1990, 1, 1)

# ── Tier 1 — Growth × Inflation axes (FRED code → spec) ──────────────────────
# dir=+1: higher value ⇒ axis-positive; dir=-1: higher value ⇒ axis-negative.
# transform: 'level' (z of level), 'yoy' (z of 12m % change), 'mom' (z of 1m % change).
# zwin: trailing observations used for the z-score baseline.
_TIER1 = [
    # Growth axis
    {"code": "T10Y3M",       "label": "수익률곡선(10y-3m)",  "axis": "growth", "dir": +1, "transform": "level", "zwin": 756},
    {"code": "ICSA",         "label": "신규실업수당(주)",     "axis": "growth", "dir": -1, "transform": "level", "zwin": 156},
    {"code": "INDPRO",       "label": "산업생산 YoY",         "axis": "growth", "dir": +1, "transform": "yoy",   "zwin": 60},
    {"code": "UMCSENT",      "label": "소비자심리",           "axis": "growth", "dir": +1, "transform": "level", "zwin": 60},
    {"code": "PAYEMS",       "label": "비농업고용 MoM",       "axis": "growth", "dir": +1, "transform": "mom",   "zwin": 60},
    # Inflation axis
    {"code": "T10YIE",       "label": "10y 기대인플레",       "axis": "inflation", "dir": +1, "transform": "level", "zwin": 756},
    {"code": "CPILFESL",     "label": "코어CPI YoY",          "axis": "inflation", "dir": +1, "transform": "yoy",   "zwin": 60},
    {"code": "CES0500000003","label": "평균시급 임금성장 YoY", "axis": "inflation", "dir": +1, "transform": "yoy",   "zwin": 60},
]
# Credit/risk overlay (not an axis — a risk-on/off indicator)
_CREDIT = [
    {"code": "BAMLH0A0HYM2", "label": "HY 신용스프레드",      "dir": -1, "transform": "level", "zwin": 504},
    {"code": "NFCI",         "label": "금융환경(시카고연)",    "dir": -1, "transform": "level", "zwin": 156},
]
# Liquidity overlay (orthogonal to growth×inflation — central-bank/banking plumbing).
# Net liquidity is a composite (Fed assets − ON RRP − TGA) built specially in
# _build_liquidity; the rest are single series. All measured as an IMPULSE (change),
# not level, since it's the flow of liquidity that drives risk appetite.
_LIQUIDITY_MEMBERS = ["WALCL", "RRPONTSYD", "WTREGEN", "WRESBAL", "TOTBKCR"]
# ── Tier 2 — Conference-Board coincident components (all monthly SA) ──────────
_COINCIDENT = ["PAYEMS", "INDPRO", "W875RX1", "CMRMTSPL"]
# Supplementary income-side growth (quarterly) — GDP+GDI average is a better
# recession signal than GDP alone (Nalewaik). Reported in Tier2, NOT fed into the
# monthly Markov (mixing quarterly forward-fill would distort switching variance).
_GDI_CODES = ["GDPC1", "A261RX1Q020SBEA"]

_QUADRANT = {
    (True,  True):  ("Reflation",  "리플레이션",   "성장+·인플+ → 밸류·에너지·커모디티·금융 순풍 (시클리컬 리플레이션)"),
    (True,  False): ("Goldilocks", "골디락스",     "성장+·인플− → 그로스·퀄리티 우위, 딥밸류/커모디티는 상대 역풍"),
    (False, True):  ("Stagflation","스태그플레이션","성장−·인플+ → 실물자산·에너지 방어, 듀레이션·그로스 취약"),
    (False, False): ("Slowdown",   "슬로다운",     "성장−·인플− → 방어주·듀레이션(장기채)·퀄리티, 시클리컬 회피"),
}

_cache_mem: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
def _fred(codes, start=_FRED_START):
    """Batch-pull FRED series → {code: pandas Series}. No API key (public CSV).
    Parallelized (bounded pool) so first-call latency ≈ slowest single series, not the sum."""
    import pandas_datareader.data as web
    from concurrent.futures import ThreadPoolExecutor, as_completed
    end = datetime.today()

    def _one(c):
        s = web.DataReader(c, "fred", start, end).iloc[:, 0].dropna()
        return c, (s if len(s) else None)

    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_one, c) for c in codes]
        for f in as_completed(futs):
            try:
                c, s = f.result()
                if s is not None:
                    out[c] = s
            except Exception:
                continue
    return out


def _transform(s, how):
    import numpy as np
    if how == "yoy":
        # monthly series → 12-period % change
        return (s.pct_change(12) * 100).dropna()
    if how == "mom":
        return (s.pct_change(1) * 100).dropna()
    return s  # level


def _zscore(s, win):
    """z of the latest value vs a trailing window of the (transformed) series.
    Winsorized to ±3 so one outlier observation cannot dominate the axis mean."""
    tail = s.tail(win)
    mu, sd = float(tail.mean()), float(tail.std())
    if sd < 1e-9:
        return 0.0
    z = (s.iloc[-1] - mu) / sd
    return float(max(-3.0, min(3.0, z)))


def _build_tier1(raw):
    import numpy as np
    comps, gz, iz = [], [], []
    for spec in _TIER1:
        s = raw.get(spec["code"])
        if s is None or len(s) < 5:
            continue
        t = _transform(s, spec["transform"])
        if len(t) < 5:
            continue
        z = _zscore(t, spec["zwin"]) * spec["dir"]
        comps.append({
            "code": spec["code"], "label": spec["label"], "axis": spec["axis"],
            "value": round(float(s.iloc[-1]), 2), "z": round(z, 2),
            "asof": s.index[-1].date().isoformat(),
        })
        (gz if spec["axis"] == "growth" else iz).append(z)
    # Credit overlay
    credit, cz = [], []
    for spec in _CREDIT:
        s = raw.get(spec["code"])
        if s is None or len(s) < 5:
            continue
        z = _zscore(s, spec["zwin"]) * spec["dir"]
        cz.append(z)
        credit.append({"code": spec["code"], "label": spec["label"],
                       "value": round(float(s.iloc[-1]), 2), "z": round(z, 2),
                       "asof": s.index[-1].date().isoformat()})
    n_growth, n_infl = len(gz), len(iz)
    growth_z = round(float(np.mean(gz)), 2) if gz else 0.0
    infl_z = round(float(np.mean(iz)), 2) if iz else 0.0
    credit_z = round(float(np.mean(cz)), 2) if cz else 0.0
    # Coverage gate: growth axis has 5 specs, inflation 2. If too few series
    # survived the FRED pull, the axis mean is unreliable → mark insufficient
    # (and force boundary) so the UI never renders a hard regime call on thin data.
    insufficient = (n_growth < 3) or (n_infl < 1)
    label, label_ko, impl = _QUADRANT[(growth_z >= 0, infl_z >= 0)]
    # Boundary zone: an axis within ±0.25 (or thin coverage) makes the quadrant
    # label unstable — flag it so the UI shows "near-boundary" not a hard call.
    boundary = abs(growth_z) < 0.25 or abs(infl_z) < 0.25 or insufficient
    # asof = latest of all component asofs
    asof = max([c["asof"] for c in comps + credit], default=None)
    return {
        "growth_z": growth_z, "inflation_z": infl_z, "credit_z": credit_z,
        "quadrant": label, "quadrant_ko": label_ko, "implication": impl,
        "boundary": boundary,
        "coverage": {"growth": n_growth, "inflation": n_infl, "insufficient": insufficient},
        "risk": ("risk_off" if credit_z < -0.5 else "risk_on" if credit_z > 0.5 else "neutral"),
        "components": comps, "credit": credit, "asof": asof,
    }


def _build_liquidity(raw):
    """Liquidity overlay — orthogonal to growth×inflation. Net liquidity (Fed
    assets − ON RRP − TGA) impulse + bank-reserves impulse + bank-credit YoY.
    Positive = ample/expanding liquidity = risk-asset tailwind. Measured as a
    ~3-month impulse (flow), z-scored so it's comparable to the other overlays."""
    import numpy as np
    items, lz = [], []
    walcl, rrp, tga = raw.get("WALCL"), raw.get("RRPONTSYD"), raw.get("WTREGEN")
    if walcl is not None and rrp is not None and tga is not None:
        idx = walcl.index.union(rrp.index).union(tga.index)
        w = walcl.reindex(idx).ffill()
        r = rrp.reindex(idx).ffill() * 1000.0    # RRP is $B; WALCL/TGA are $M → align to $M
        t = tga.reindex(idx).ffill()
        net = (w - r - t).dropna()               # $M
        chg = net.diff(13).dropna()              # weekly data → ~13wk = 3-month impulse
        if len(chg) >= 20:
            z = _zscore(chg, 156)
            lz.append(z)
            items.append({"code": "NET_LIQ", "label": "순유동성(연준−RRP−TGA) 3M임펄스",
                          "value": round(float(net.iloc[-1]) / 1e6, 2), "z": round(z, 2),
                          "asof": net.index[-1].date().isoformat()})
    res_s = raw.get("WRESBAL")
    if res_s is not None and len(res_s) > 20:
        chg = res_s.diff(13).dropna()
        if len(chg) >= 20:
            z = _zscore(chg, 156)
            lz.append(z)
            items.append({"code": "WRESBAL", "label": "은행 지급준비금 3M임펄스",
                          "value": round(float(res_s.iloc[-1]) / 1e6, 2), "z": round(z, 2),
                          "asof": res_s.index[-1].date().isoformat()})
    cr = raw.get("TOTBKCR")
    if cr is not None and len(cr) > 60:
        yoy = (cr.pct_change(52) * 100).dropna()   # weekly → 52wk YoY
        if len(yoy) >= 20:
            z = _zscore(yoy, 156)
            lz.append(z)
            items.append({"code": "TOTBKCR", "label": "은행 총신용 YoY",
                          "value": round(float(yoy.iloc[-1]), 1), "z": round(z, 2),
                          "asof": cr.index[-1].date().isoformat()})
    liq_z = round(float(np.mean(lz)), 2) if lz else 0.0
    state = "확장" if liq_z > 0.5 else "수축" if liq_z < -0.5 else "중립"
    return {"liquidity_z": liq_z, "liquidity_state": state, "liquidity": items}


def _gdi_reading(raw):
    """Supplementary income-side growth (quarterly): GDP+GDI average YoY. A better
    recession signal than GDP alone (Nalewaik). Reported alongside Tier2, not fed
    into the monthly Markov (quarterly forward-fill would distort switching var)."""
    import pandas as pd
    gdp, gdi = raw.get("GDPC1"), raw.get("A261RX1Q020SBEA")

    def _yoy(s):
        return (s.pct_change(4) * 100).dropna() if (s is not None and len(s) > 5) else None

    gy, iy = _yoy(gdp), _yoy(gdi)
    parts = [x for x in (gy, iy) if x is not None]
    if not parts:
        return None
    avg = pd.concat(parts, axis=1).mean(axis=1).dropna()
    if avg.empty:
        return None
    latest = float(avg.iloc[-1])
    return {
        "gdp_yoy": round(float(gy.iloc[-1]), 2) if gy is not None else None,
        "gdi_yoy": round(float(iy.iloc[-1]), 2) if iy is not None else None,
        "avg_yoy": round(latest, 2),
        "signal": "확장" if latest > 1.0 else "수축" if latest < 0 else "둔화",
        "asof": avg.index[-1].date().isoformat(),
        "note": "GDP+GDI 평균 YoY — 소득측 성장 교차확인(침체신호 우수)",
    }


def _build_tier2(raw):
    """Markov-switching recession probability on a CB-style coincident index.

    Fully self-guarded: every statsmodels touch (import, fit, param/prob extraction,
    which are version-fragile — e.g. the `const[i]` key) is inside the try so a
    missing/quirky statsmodels degrades to an error dict and never poisons Tier-1.
    """
    import numpy as np, pandas as pd
    data = {c: raw[c] for c in _COINCIDENT if c in raw}
    if len(data) < 3:
        return {"error": "coincident components unavailable", "recession_prob": None}
    df = pd.DataFrame(data).resample("MS").last().dropna()
    if len(df) < 120:
        return {"error": "insufficient coincident history", "recession_prob": None}
    g = np.log(df).diff() * 100.0            # monthly log-growth %
    gz = (g - g.mean()) / g.std()
    cei = gz.mean(axis=1).dropna()
    try:
        import statsmodels.api as sm
        mod = sm.tsa.MarkovRegression(cei, k_regimes=2, trend="c", switching_variance=True)
        res = mod.fit(disp=False)
        means = [float(res.params[f"const[{i}]"]) for i in range(2)]
        # Recession = low-growth state. Guard: the two regimes must be mean-separated,
        # else the label is arbitrary (degenerate fit) → mark unreliable.
        sep = abs(means[0] - means[1])
        rec_state = int(np.argmin(means))
        prob = res.smoothed_marginal_probabilities[rec_state]
        hist = [{"date": d.date().isoformat(), "prob": round(float(p) * 100, 1)}
                for d, p in prob.tail(36).items()]
        return {
            "recession_prob": round(float(prob.iloc[-1]) * 100, 1),
            "recession_prob_3m": [round(float(x) * 100, 1) for x in prob.tail(3)],
            "coincident_asof": cei.index[-1].date().isoformat(),
            "state_means": [round(m, 3) for m in means],
            "state_separation": round(sep, 3),
            "reliable": bool(res.mle_retvals.get("converged", False) and sep >= 0.2),
            "converged": bool(res.mle_retvals.get("converged", False)),
            "history": hist,
            "n_obs": int(len(cei)),
            "gdi": _gdi_reading(raw),   # supplementary income-side growth cross-check
        }
    except Exception as e:
        return {"error": f"markov unavailable: {type(e).__name__}: {str(e)[:80]}", "recession_prob": None,
                "gdi": _gdi_reading(raw)}


def _divergence(tier1, market_regime):
    """Macro (economic) vs market (price cross-sectional) regime agreement flag."""
    if not market_regime:
        return {"flag": "NO_MARKET_REGIME", "note": "시장 레짐 미제공"}
    g_up = tier1["growth_z"] >= 0
    i_up = tier1["inflation_z"] >= 0
    # Market regime dominance flags (two independent axes: cyclical↔defensive, growth↔value).
    mkt_cyc = bool(market_regime.get("cyclical_dom"))
    mkt_def = bool(market_regime.get("defensive_dom"))
    mkt_val = bool(market_regime.get("value_dom"))
    mkt_grw = bool(market_regime.get("growth_dom"))
    notes = []
    # Growth axis: cyclical/defensive market leadership vs macro growth momentum.
    if mkt_cyc and not g_up:
        notes.append("시장은 시클리컬 우위인데 매크로 성장 모멘텀은 약함 → 시클리컬 강세가 펀더 확증 부족")
    if mkt_def and g_up:
        notes.append("시장은 방어주 우위인데 매크로 성장은 견조 → 방어 쏠림이 경기와 어긋남(리스크온 여지)")
    # Inflation axis: value/growth market leadership vs macro inflation trend.
    if mkt_val and not i_up:
        notes.append("시장은 밸류/리플레이션 우위인데 매크로 인플레는 완화(디스인플레) → 밸류·커모디티 쏠림은 경기순환 확증 없는 모멘텀 현상일 수 있음")
    if mkt_grw and i_up:
        notes.append("시장은 그로스 우위인데 매크로 인플레는 상승 → 리플레이션은 밸류에 유리, 그로스 리더십과 어긋남")
    flag = "DIVERGENCE" if notes else "CONFIRM"
    if not notes:
        notes.append(f"매크로({'성장+' if g_up else '성장−'}·{'인플+' if i_up else '인플−'})와 시장 레짐 대체로 정합")
    return {"flag": flag, "note": " / ".join(notes),
            "macro": {"growth_up": g_up, "inflation_up": i_up},
            "market": {"cyclical_dom": mkt_cyc, "defensive_dom": mkt_def,
                       "value_dom": mkt_val, "growth_dom": mkt_grw}}


# ─────────────────────────────────────────────────────────────────────────────
def compute_macro_regime(market_regime: Optional[dict] = None, force: bool = False) -> dict:
    """Main entry. Returns the full macro-regime payload (Tier1 + Tier2 + divergence).

    Cached (mem + disk) with a 6h TTL. `market_regime` = api.STATE['regime'] for
    the divergence flag (optional). On FRED failure, returns last good cache if any.
    """
    now = time.time()

    def _fresh_mem():
        return _cache_mem.get("data") and (time.time() - _cache_mem.get("ts", 0) < _TTL_SEC)

    # fast path: fresh mem cache (no lock needed for a stale-tolerant read)
    if not force and _fresh_mem():
        return _refresh_divergence(_cache_mem["data"], market_regime)

    # Singleflight: serialize recompute so N concurrent requests trigger ONE FRED
    # pull + Markov fit, not N. Double-check the cache inside the lock.
    with _lock:
        if not force and _fresh_mem():
            return _refresh_divergence(_cache_mem["data"], market_regime)
        # disk cache (also under lock — avoids a torn read racing a writer)
        if not force and os.path.exists(_CACHE_PATH):
            try:
                with open(_CACHE_PATH) as f:
                    disk = json.load(f)
                if now - disk.get("_ts", 0) < _TTL_SEC:
                    _cache_mem.update({"data": disk, "ts": disk.get("_ts", now)})
                    return _refresh_divergence(disk, market_regime)
            except Exception:
                pass
        # recompute
        try:
            codes = sorted({s["code"] for s in _TIER1} | {s["code"] for s in _CREDIT}
                           | set(_COINCIDENT) | set(_LIQUIDITY_MEMBERS) | set(_GDI_CODES))
            raw = _fred(codes)
            if not raw:
                raise RuntimeError("FRED returned no data")
            tier1 = _build_tier1(raw)
            tier1.update(_build_liquidity(raw))   # merge liquidity overlay (liquidity_z/_state/liquidity)
            # Tier-2 is isolated: a statsmodels failure must NOT discard a valid Tier-1.
            try:
                tier2 = _build_tier2(raw)
            except Exception as e:
                tier2 = {"error": f"tier2 unavailable: {type(e).__name__}: {str(e)[:80]}",
                         "recession_prob": None}
            payload = {
                "as_of": tier1.get("asof") or datetime.today().date().isoformat(),
                "tier1": tier1,
                "tier2": tier2,
                "divergence": _divergence(tier1, market_regime),
                "_ts": now,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            _cache_mem.update({"data": payload, "ts": now})
            # atomic write: tmp + os.replace so a concurrent reader never sees a
            # half-written file (two writers documented: API proc + pipeline).
            try:
                tmp = _CACHE_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=1)
                os.replace(tmp, _CACHE_PATH)
            except Exception:
                pass
            return payload
        except Exception as e:
            # graceful degradation → last good cache
            if _cache_mem.get("data"):
                return _refresh_divergence(_cache_mem["data"], market_regime)
            if os.path.exists(_CACHE_PATH):
                try:
                    with open(_CACHE_PATH) as f:
                        return _refresh_divergence(json.load(f), market_regime)
                except Exception:
                    pass
            return {"error": f"macro regime unavailable: {e}", "tier1": None, "tier2": None}


def _refresh_divergence(payload: dict, market_regime: Optional[dict]) -> dict:
    """Recompute only the (cheap) divergence flag against the live market regime."""
    if market_regime and payload.get("tier1"):
        p = dict(payload)
        p["divergence"] = _divergence(payload["tier1"], market_regime)
        return p
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Orbit history — per-indicator z-score TIME SERIES for the 3D "solar system"
# visualization (each indicator = a planet; angle=time, radius=z over the window).
# ─────────────────────────────────────────────────────────────────────────────
_ORBIT_TTL = 12 * 3600
_orbit_mem: dict = {}


def _monthly_rolling_z(monthly, win_m=60, dirn=1):
    """Rolling (trailing-window) z of a monthly series, dir-adjusted + winsorized ±3.
    Same convention as the point-in-time _zscore so the orbit matches the panel."""
    mu = monthly.rolling(win_m, min_periods=24).mean()
    sd = monthly.rolling(win_m, min_periods=24).std()
    rz = ((monthly - mu) / sd) * dirn
    return rz.clip(-3.0, 3.0)


def _liquidity_monthly(raw):
    """Monthly (pre-z) liquidity signals → [(code, label, monthly_series)]. Same
    transforms as _build_liquidity (net-liq/reserves impulse, bank-credit YoY)."""
    out = []
    walcl, rrp, tga = raw.get("WALCL"), raw.get("RRPONTSYD"), raw.get("WTREGEN")
    if walcl is not None and rrp is not None and tga is not None:
        idx = walcl.index.union(rrp.index).union(tga.index)
        net = (walcl.reindex(idx).ffill() - rrp.reindex(idx).ffill() * 1000.0
               - tga.reindex(idx).ffill()).dropna()
        m = net.resample("MS").last().dropna()
        out.append(("NET_LIQ", "순유동성 3M임펄스", m.diff(3).dropna()))
    res_s = raw.get("WRESBAL")
    if res_s is not None:
        m = res_s.resample("MS").last().dropna()
        out.append(("WRESBAL", "은행 지준 3M임펄스", m.diff(3).dropna()))
    cr = raw.get("TOTBKCR")
    if cr is not None:
        m = cr.resample("MS").last().dropna()
        out.append(("TOTBKCR", "은행 총신용 YoY", (m.pct_change(12) * 100).dropna()))
    return out


def compute_orbit_history(months: int = 54, force: bool = False) -> dict:
    """Per-indicator monthly rolling-z history over the last `months` months, on a
    common monthly grid. Feeds the 3D orbit viz. Cached 12h (separate from the
    point-in-time regime cache). Reuses the SAME FRED series/transforms as Tier1."""
    import pandas as pd
    months = max(24, min(120, int(months)))
    now = time.time()
    key = f"orbit_{months}"
    if not force and _orbit_mem.get(key) and (now - _orbit_mem[key]["ts"] < _ORBIT_TTL):
        return _orbit_mem[key]["data"]
    with _lock:
        if not force and _orbit_mem.get(key) and (now - _orbit_mem[key]["ts"] < _ORBIT_TTL):
            return _orbit_mem[key]["data"]
        try:
            codes = sorted({s["code"] for s in _TIER1} | {s["code"] for s in _CREDIT}
                           | set(_LIQUIDITY_MEMBERS))
            raw = _fred(codes)
            if not raw:
                raise RuntimeError("FRED returned no data")
            series = []   # (code, label, axis, monthly_z_series)
            for spec in _TIER1 + _CREDIT:
                s = raw.get(spec["code"])
                if s is None or len(s) < 30:
                    continue
                t = _transform(s, spec["transform"])
                m = t.resample("MS").last().dropna()
                rz = _monthly_rolling_z(m, 60, spec["dir"]).dropna()
                if len(rz) >= 12:
                    series.append((spec["code"], spec["label"], spec.get("axis", "credit"), rz))
            for code, label, m in _liquidity_monthly(raw):
                rz = _monthly_rolling_z(m, 60, 1).dropna()
                if len(rz) >= 12:
                    series.append((code, label, "liquidity", rz))
            if not series:
                raise RuntimeError("no orbit series")
            end = max(s[3].index[-1] for s in series)
            grid = pd.date_range(end=end, periods=months, freq="MS")
            planets = []
            for code, label, axis, rz in series:
                z = rz.reindex(grid)
                planets.append({
                    "code": code, "label": label, "axis": axis,
                    "z": [None if pd.isna(v) else round(float(v), 2) for v in z.values],
                })
            payload = {
                "months": months,
                "dates": [d.strftime("%Y-%m") for d in grid],
                "planets": planets,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            _orbit_mem[key] = {"data": payload, "ts": now}
            return payload
        except Exception as e:
            if _orbit_mem.get(key):
                return _orbit_mem[key]["data"]
            return {"error": f"orbit unavailable: {e}", "planets": [], "dates": []}


if __name__ == "__main__":
    import pprint
    pprint.pprint(compute_macro_regime(force=True))
