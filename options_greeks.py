"""Options dealer-gamma layer — Tier 2 (per-ticker GEX/skew/positioning) + Tier 3
(unwind→momentum detector). The mechanism behind the user's thesis (2026-07):

  Crowded downside hedging pushes dealers SHORT gamma (negative GEX). In that regime
  dealer hedging AMPLIFIES moves — so when the crowded positioning unwinds (price
  reclaims the zero-gamma level / a key put strike), the dealer must buy into strength
  → sharp EARLY momentum. This is a SETUP signal visible BEFORE the price move, which
  the price-based momentum Composite cannot see. It's a natural feed for the forming /
  pre-momentum pool (early-entry work, 2026-07).

Tier 2 — per ticker (curated subset: index/sector ETFs + liquid US buy-list names):
  • GEX (net dealer gamma, SqueezeMetrics convention: +call_gamma·OI −put_gamma·OI)
    → gamma_regime (short/neutral/long by sign). SIGN is the signal, not the noisy bn.
  • zero_gamma — spot level where net GEX flips sign (regime boundary); dist from spot.
  • put/call OI, ATM IV, skew (OTM-put IV − OTM-call IV).
Tier 3 — unwind_score: short-gamma × crowding × trigger.

DATA/MODEL CAVEATS: yfinance option OI is EOD/T+1 (not intraday). GEX assumes the
standard dealer-sign convention (dealers long calls / short puts) — an unobservable
approximation, directionally useful but not exact. 0DTE distorts the front; we use
7-45d expiries. US-listed liquid names only (no Korean/intl options). NaN-safe (yf IV
is frequently NaN per contract → must filter, else GEX=NaN).
"""
from __future__ import annotations

import json
import os
import threading
import time
import warnings
from datetime import date, datetime
from typing import Optional

warnings.filterwarnings("ignore")

_CACHE_PATH = ".options_greeks_cache.json"
_TTL_SEC = 6 * 3600          # OI is EOD; 6h lets same-day refresh
_lock = threading.Lock()
_mem: dict = {}
_RATE = 0.045

# Fixed core: broad indices + SPDR sectors + key thematic. Buy-list US names appended.
_CORE = ["SPY", "QQQ", "IWM", "DIA", "SMH",
         "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE", "XLC"]
_MAX_TICKERS = 45


def _bs_gamma(S, K, iv, T):
    """Black-Scholes gamma (vectorized numpy-safe)."""
    import numpy as np
    with np.errstate(all="ignore"):
        d1 = (np.log(S / K) + (_RATE + iv * iv / 2) * T) / (iv * np.sqrt(T))
        g = np.exp(-d1 * d1 / 2) / np.sqrt(2 * np.pi) / (S * iv * np.sqrt(T))
    return np.where(np.isfinite(g), g, 0.0)


def _ticker_greeks(tk: str) -> Optional[dict]:
    """Per-ticker options metrics. NaN-safe. Returns None if no usable option data."""
    import numpy as np
    import yfinance as yf
    try:
        t = yf.Ticker(tk)
        exps = t.options or []
        sel = []
        for e in exps:
            try:
                dd = (np.datetime64(e) - np.datetime64("today")).astype(int)
                if 7 <= dd <= 45:
                    sel.append((abs(dd - 30), e, int(dd)))
            except Exception:
                continue
        sel = sorted(sel)[:3]
        if not sel:
            return None
        hist = t.history(period="1d")
        if hist.empty:
            return None
        spot = float(hist["Close"].iloc[-1])
        if not np.isfinite(spot) or spot <= 0:
            return None

        Ks, IVs, Ts, OIs, SGN, isput = [], [], [], [], [], []
        # skew/atm_iv from the NEAREST-30d expiry ONLY (maturity-consistent; sel[0] is nearest 30d)
        near_exp = sel[0][1]
        atm_iv_s, otm_put_iv, otm_call_iv = [], [], []
        n_exp_used = 0
        for _, e, dd in sel:
            T = max(1, dd) / 365.0
            try:
                oc = t.option_chain(e)
            except Exception:
                continue
            n_exp_used += 1
            for df, sign, put in ((oc.calls, +1, False), (oc.puts, -1, True)):
                k = df["strike"].to_numpy(dtype=float)
                iv = df["impliedVolatility"].to_numpy(dtype=float)
                oi = df["openInterest"].fillna(0).to_numpy(dtype=float)
                m = np.isfinite(k) & np.isfinite(iv) & (iv > 0.01) & np.isfinite(oi) & (oi > 0)
                Ks.append(k[m]); IVs.append(iv[m]); OIs.append(oi[m])
                Ts.append(np.full(m.sum(), T)); SGN.append(np.full(m.sum(), sign))
                isput.append(np.full(m.sum(), put))
                if e == near_exp:   # skew/atm only from the single nearest-30d expiry
                    atm_iv_s.extend(list(iv[m & (np.abs(k - spot) <= spot * 0.03)]))
                    if put:
                        otm_put_iv.extend(list(iv[m & (k <= spot * 0.97) & (k >= spot * 0.85)]))
                    else:
                        otm_call_iv.extend(list(iv[m & (k >= spot * 1.03) & (k <= spot * 1.15)]))
        if not Ks:
            return None
        K = np.concatenate(Ks); IV = np.concatenate(IVs); OI = np.concatenate(OIs)
        T = np.concatenate(Ts); SG = np.concatenate(SGN); PUT = np.concatenate(isput)
        tot_oi = float(OI.sum())

        # ── 유동성 게이트 (리뷰 수정) — 얇은 국제/단일 ETF의 희소 strike 노이즈 배제 ──
        # 총 계약수·총 OI·만기수·윙별 strike 수 미달이면 thin=True (unwind 후보서 제외).
        thin = (len(K) < 20) or (tot_oi < 2000) or (n_exp_used < 2) \
            or (len(otm_put_iv) < 2) or (len(otm_call_iv) < 2)

        # net GEX at current spot (sign: dealers long calls / short puts) + GROSS gamma
        g0 = _bs_gamma(spot, K, IV, T)
        dollar_g = g0 * OI * 100 * spot * spot * 0.01
        gex = float(np.sum(SG * dollar_g))
        gross = float(np.sum(np.abs(dollar_g)))
        # ★정규화 net/gross 비율 [-1,1] — 절대 $bn 대신 이걸로 regime 판정(단일종목도 탐지).
        gex_norm = round(gex / gross, 3) if gross > 0 else 0.0
        gamma_regime = "short" if gex_norm < -0.08 else "long" if gex_norm > 0.08 else "neutral"

        # zero-gamma: 감마가 유의미(non-neutral)할 때만 — near-zero 프로파일의 flip은 노이즈.
        zero_gamma = None
        if gamma_regime != "neutral":
            grid = spot * np.linspace(0.85, 1.15, 41)
            netg = np.array([np.sum(SG * _bs_gamma(S, K, IV, T) * OI * 100 * S * S * 0.01) for S in grid])
            sflip = np.where(np.sign(netg[:-1]) != np.sign(netg[1:]))[0]
            if len(sflip):
                # 각 flip의 보간 교차점 계산 후 spot에 가장 가까운 것 선택(좌측엣지 아님).
                cross = []
                for i in sflip:
                    ga, gb = netg[i], netg[i + 1]
                    x = grid[i] + (grid[i + 1] - grid[i]) * (0 - ga) / (gb - ga) if gb != ga else grid[i]
                    cross.append(x)
                zero_gamma = float(cross[int(np.argmin(np.abs(np.array(cross) - spot)))])

        put_oi = float(OI[PUT].sum()); call_oi = float(OI[~PUT].sum())
        pc_oi = round(put_oi / call_oi, 2) if call_oi > 0 else None
        atm_iv = float(np.median(atm_iv_s)) if atm_iv_s else None
        # tail skew: OTM(3-15%) put IV − OTM call IV (rich puts = crowded downside hedge)
        skew = None
        if len(otm_put_iv) >= 2 and len(otm_call_iv) >= 2:
            skew = round((float(np.median(otm_put_iv)) - float(np.median(otm_call_iv))) * 100, 1)
            skew = max(-15.0, min(15.0, skew))   # winsorize
        zg_dist = round((zero_gamma / spot - 1) * 100, 2) if zero_gamma else None
        return {
            "ticker": tk, "spot": round(spot, 2),
            "gex_bn": round(gex / 1e9, 3), "gex_norm": gex_norm, "gamma_regime": gamma_regime,
            "zero_gamma": round(zero_gamma, 2) if zero_gamma else None, "zg_dist_pct": zg_dist,
            "put_call_oi": pc_oi, "atm_iv": round(atm_iv * 100, 1) if atm_iv else None,
            "skew": skew, "n_exp": n_exp_used, "tot_oi": int(tot_oi), "thin": thin,
        }
    except Exception:
        return None


def _unwind_score(g: dict, scan_row: Optional[dict]) -> dict:
    """Tier 3 — unwind→momentum setup score (0-100). 재설계(리뷰): 딜러 숏감마는 증폭
    메커니즘 = 가설의 **필요조건** → GATE. 숏감마 아니거나 thin이면 셋업 불가('—').
    숏감마 위에서 과밀헤지(결합·캡) + 방향성 트리거(zero-gamma 상방 플립·모멘텀 전환)."""
    reasons = []
    if g.get("thin"):
        return {"unwind_score": 0, "unwind_flag": "—", "unwind_reasons": ["얇은 유동성(신뢰 불가)"]}
    if g.get("gamma_regime") != "short":
        return {"unwind_score": 0, "unwind_flag": "—", "unwind_reasons": ["딜러 숏감마 아님(증폭 메커니즘 부재)"]}
    # 숏감마 = 증폭 존재 → WATCH 하한(40)
    score = 40.0; reasons.append("딜러 숏감마(증폭)")
    # 과밀 하방헤지 — 풋OI·스큐는 같은 현상의 두 프록시 → 결합·상한 25.
    pc = g.get("put_call_oi"); sk = g.get("skew")
    crowd = (15 if (pc is not None and pc >= 1.8) else 0) + (15 if (sk is not None and sk >= 3.0) else 0)
    if crowd:
        score += min(25, crowd)
        det = []
        if pc is not None and pc >= 1.8: det.append(f"풋OI {pc}")
        if sk is not None and sk >= 3.0: det.append(f"스큐 +{sk}")
        reasons.append("과밀 하방헤지(" + "·".join(det) + ")")
    # 방향성 트리거 — zero-gamma가 spot '위'(0~+4%)로 근접 = 상방 탈환 시 숏감마 unwind up.
    zg = g.get("zg_dist_pct")
    if zg is not None and 0.0 <= zg <= 4.0:
        score += 20; reasons.append("zero-gamma 상방 근접(플립업 임박)")
    # 모멘텀 전환 확인
    if scan_row:
        rs_s = scan_row.get("rss_short"); rs_l = scan_row.get("rss_long")
        try:
            if rs_s is not None and rs_l is not None and float(rs_s) > float(rs_l):
                score += 15; reasons.append("단기 RS 상향(모멘텀 전환)")
        except (TypeError, ValueError):
            pass
    flag = "UNWIND_SETUP" if score >= 70 else "WATCH"   # 숏감마면 최소 WATCH
    return {"unwind_score": round(score, 0), "unwind_flag": flag, "unwind_reasons": reasons}


def compute_options_flow(force: bool = False) -> dict:
    """Tier2+3. Per-ticker dealer gamma for a curated US-liquid subset + unwind ranking.
    Cached 6h. NEVER caches an empty result (transient yfinance failure must not stick)."""
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
            # universe: core + liquid US names from STATE (eligible momentum, high ADV).
            # ★build_final_lists() 재호출 회피(스웜캐시/스캔 이중 재로드로 무거움) — STATE만 사용.
            scan = {}
            us_names = []
            try:
                import api
                rows = api.STATE.get("results") or []
                scan = {r.get("ticker"): r for r in rows}
                elig = [r for r in rows if r.get("eligible") and "." not in (r.get("ticker") or "")
                        and float(r.get("adv_usd") or 0) >= 50e6]   # 유동성 있는 미국 종목(ADV≥$50M)
                elig.sort(key=lambda r: -float(r.get("composite") or 0))
                us_names = [r.get("ticker") for r in elig]
            except Exception:
                us_names = []
            universe = list(dict.fromkeys(_CORE + us_names))[:_MAX_TICKERS]

            from concurrent.futures import ThreadPoolExecutor, as_completed
            recs = []
            with ThreadPoolExecutor(max_workers=6) as ex:
                futs = {ex.submit(_ticker_greeks, tk): tk for tk in universe}
                for fu in as_completed(futs):
                    try:
                        g = fu.result()
                    except Exception:
                        g = None
                    if g:
                        g.update(_unwind_score(g, scan.get(g["ticker"])))
                        recs.append(g)
            if not recs:
                raise RuntimeError("no option data (yfinance)")
            recs.sort(key=lambda r: -r.get("unwind_score", 0))
            # market-wide dealer gamma tilt — broad indices(SPY/QQQ/IWM/DIA) 합의; SPY 우선,
            # 없으면 존재하는 지수의 gex_norm 부호 합의로 폴백.
            idx = {r["ticker"]: r for r in recs}
            broad = [idx[t] for t in ("SPY", "QQQ", "IWM", "DIA") if t in idx]
            if idx.get("SPY"):
                spy_regime = idx["SPY"].get("gamma_regime")
            elif broad:
                avg = sum(r.get("gex_norm") or 0 for r in broad) / len(broad)
                spy_regime = "short" if avg < -0.08 else "long" if avg > 0.08 else "neutral"
            else:
                spy_regime = None
            payload = {
                "as_of": date.today().isoformat(),
                "n": len(recs),
                "market_gamma": {"spy": (idx.get("SPY") or {}).get("gex_bn"),
                                 "qqq": (idx.get("QQQ") or {}).get("gex_bn"),
                                 "regime": spy_regime},
                "tickers": recs,
                "unwind_candidates": [r for r in recs if r.get("unwind_flag") in ("UNWIND_SETUP", "WATCH")],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "_ts": now,
            }
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
            return {"error": f"options flow unavailable: {e}", "tickers": [], "unwind_candidates": []}


if __name__ == "__main__":
    import pprint
    pprint.pprint(compute_options_flow(force=True))
