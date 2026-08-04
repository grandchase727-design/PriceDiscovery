"""
herding_reversal.py — Momentum-overheating → Reversal detection (paper [3] operationalized).

Grounded in arXiv 2607.27063 "Herding, Momentum, and Reversal in China's A-Share Market":
    momentum = slow info-diffusion + herding amplification → price overshoots fundamental.
    REVERSAL is signalled NOT by strong herding (that continues the trend) but by
    herding WEAKENING while the name is already crowded/overextended.

Operationalized as a 3-gate conjunction (validated walk-forward on .backtest_price_cache.pkl,
826 tickers × 1yr; crowded+cracking underperforms crowded+holding by −5..−7%p at 21-42d,
but is RARE ~14 obs/yr → a low-frequency, high-conviction AVOID/de-prioritize signal,
NOT a broad OER-style penalty):

  Gate A (crowded/overextended): run-up(63d) x-sec pctile ≥ 80  AND  RSI(14) ≥ 68
  Gate B (herding present):      sector CSAD compression + strong directional move
                                 (and/or options put-crowding cross-confirm)
  Gate C (herding WEAKENING) ★:  short-horizon(10d) momentum x-sec pctile < 40
                                 (was a leader, now cracking short-term)

  REVERSAL_RISK  = A ∧ C          (the validated conjunction)
  WATCH          = A ∧ (40 ≤ 10d pctile < 55)   (crowded + slowing, not yet cracking)
  CROWDED_HOLDING= A ∧ (10d pctile ≥ 55)         (crowded but still leading → trend continues)

Plus a market-level CCK herding test (CSAD = a + b1|Rm| + b2·Rm²; b2<0 & significant via
Newey-West HAC = herding) and a per-sector CSAD herding state (gated on dispersion
COMPRESSION vs the sector's own history, so mechanical low-dispersion factor moves —
e.g. a coordinated rate move in long-duration FI — don't false-positive).

Display-first + validation-gated. Fed as a give-back amplifier (position_state) and a
buy-list TIE-BREAKER (final_list) — never a hard exclusion.
"""
from __future__ import annotations
import os, pickle, math
import numpy as np
import pandas as pd

# ── tunables (validated) ──────────────────────────────────────────
RUNUP_LB   = 63     # run-up / overextension lookback (trading days)
SHORT_LB   = 10     # short-horizon rollover lookback
RSI_N      = 14
CROWD_RUNUP_PCT = 80.0
CROWD_RSI       = 68.0
ROLLOVER_PCT    = 40.0   # < this = herding weakening (cracking)
WATCH_PCT       = 55.0   # 40..55 = slowing; ≥55 = still leading
SECTOR_MIN_N    = 8
SECTOR_MIN_MOVE = 2.0    # min |10d cumulative %| for a sector to count as "moving" (filters low-vol FI)

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".backtest_price_cache.pkl")


# ── helpers ───────────────────────────────────────────────────────
def _load_close_matrix(cache_path: str = _CACHE_PATH, min_len: int = 150) -> pd.DataFrame | None:
    if not os.path.exists(cache_path):
        return None
    try:
        blob = pickle.load(open(cache_path, "rb"))
    except Exception:
        return None
    data = blob.get("data") if isinstance(blob, dict) else None
    if not isinstance(data, dict):
        return None
    closes = {}
    for t, df in data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
        if col is None:
            continue
        s = df[col].dropna()
        if len(s) >= min_len:
            closes[t] = s
    if not closes:
        return None
    px = pd.DataFrame(closes).sort_index()
    px.index = pd.to_datetime(px.index)
    return px


def _rsi(px: pd.DataFrame, n: int = RSI_N) -> pd.DataFrame:
    dlt = px.diff()
    up = dlt.clip(lower=0).rolling(n).mean()
    dn = (-dlt.clip(upper=0)).rolling(n).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # zero-loss window with gains → RSI=100 (standard). Without this, a maximally
    # overbought name (no down-close in 14d) yields NaN and is silently dropped from
    # the crowded-gate universe. Fully-flat windows (up==0 & dn==0) stay NaN.
    return out.mask((dn == 0) & (up > 0), 100.0)


def _fin(x):
    try:
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    except Exception:
        return None


# ── per-ticker HRR at the latest date ─────────────────────────────
def _per_ticker(px: pd.DataFrame) -> dict:
    i = len(px) - 1
    if i < RUNUP_LB + 2:
        return {}
    p  = px.iloc[i]
    pL = px.iloc[i - RUNUP_LB]
    pR = px.iloc[i - SHORT_LB]
    valid = p.notna() & pL.notna() & pR.notna()
    tk = p.index[valid]
    if len(tk) < 20:
        return {}
    runup = (p[tk] / pL[tk] - 1) * 100
    ret10 = (p[tk] / pR[tk] - 1) * 100
    ru_pct = runup.rank(pct=True) * 100
    r10_pct = ret10.rank(pct=True) * 100
    rsi_now = _rsi(px[tk]).iloc[i]

    out = {}
    for t in tk:
        rc = _fin(ru_pct.get(t)); rr = _fin(rsi_now.get(t)); r10p = _fin(r10_pct.get(t))
        if rc is None or rr is None or r10p is None:
            continue
        crowded = (rc >= CROWD_RUNUP_PCT) and (rr >= CROWD_RSI)
        reasons = []
        flag = None
        score = 0.0
        if crowded:
            score = 50.0
            reasons.append(f"과밀(run-up 상위{100-rc:.0f}%·RSI{rr:.0f})")
            if r10p < ROLLOVER_PCT:
                flag = "REVERSAL_RISK"
                score += 25.0 + (ROLLOVER_PCT - r10p) / ROLLOVER_PCT * 15.0
                reasons.append(f"herding 약화(최근10d 하위{r10p:.0f}%)")
            elif r10p < WATCH_PCT:
                flag = "WATCH"
                score += 8.0
                reasons.append(f"단기 둔화(10d {r10p:.0f}%)")
            else:
                flag = "CROWDED_HOLDING"
                reasons.append(f"강세지속(10d 상위{100-r10p:.0f}%)")
            score += (rc - CROWD_RUNUP_PCT) / 20.0 * 8.0
            score += min(10.0, max(0.0, (rr - CROWD_RSI) / 12.0 * 10.0))
        out[t] = dict(
            hrr_flag=flag,
            hrr_score=round(min(100.0, score), 1),
            hrr_runup=round(_fin(runup.get(t)) or 0.0, 1),
            hrr_runup_pct=round(rc),
            hrr_rsi=round(rr),
            hrr_ret10=round(_fin(ret10.get(t)) or 0.0, 1),
            hrr_ret10_pct=round(r10p),
            hrr_reasons=reasons,
        )
    return out


# ── market-level CCK herding test ─────────────────────────────────
def _cck_market(px: pd.DataFrame) -> dict:
    rets = px.pct_change() * 100
    Rm = rets.mean(axis=1)
    CSAD = rets.sub(Rm, axis=0).abs().mean(axis=1)
    reg = pd.DataFrame({"CSAD": CSAD, "aRm": Rm.abs(), "Rm2": Rm ** 2}).dropna()
    res = dict(cck_b2=None, cck_t=None, market_herding=False,
               csad_recent=None, csad_hist=None, csad_state="정상")
    if len(reg) >= 40:
        X = np.column_stack([np.ones(len(reg)), reg.aRm.values, reg.Rm2.values])
        y = reg.CSAD.values
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            # HAC (Newey-West) SE for b2 — daily CSAD residuals are autocorrelated /
            # heteroskedastic (vol clustering), so homoskedastic OLS SE understates and
            # inflates |t|, over-declaring herding. Bartlett kernel, L≈4(n/100)^(2/9).
            XtX_inv = np.linalg.inv(X.T @ X)
            Xu = X * resid[:, None]                       # (n, k)
            n_obs = len(y)
            L = max(1, int(np.floor(4 * (n_obs / 100.0) ** (2.0 / 9.0))))
            S = Xu.T @ Xu                                 # Gamma_0
            for lag in range(1, L + 1):
                w = 1.0 - lag / (L + 1.0)
                G = Xu[lag:].T @ Xu[:-lag]
                S += w * (G + G.T)
            cov = XtX_inv @ S @ XtX_inv
            var = cov[2, 2]
            se = math.sqrt(var) if var > 0 else float("nan")
            t = beta[2] / se if se and not math.isnan(se) else 0.0
            res["cck_b2"] = round(float(beta[2]), 4)
            res["cck_t"] = round(float(t), 2)
            res["market_herding"] = bool(beta[2] < 0 and abs(t) > 2)
        except Exception:
            pass
    if len(CSAD.dropna()) >= 120:
        recent = float(CSAD.iloc[-10:].mean())
        hist = float(CSAD.iloc[-120:-10].mean())
        res["csad_recent"] = round(recent, 2)
        res["csad_hist"] = round(hist, 2)
        if recent < hist * 0.9:
            res["csad_state"] = "압축(쏠림↑ 반전경계)"
        elif recent > hist * 1.1:
            res["csad_state"] = "확대(분산·건강)"
        else:
            res["csad_state"] = "정상"
    return res


# ── per-sector CSAD herding (vol-normalized) ──────────────────────
def _sector_herding(px: pd.DataFrame, results: list) -> list:
    sec = {}
    for r in results:
        c = r.get("category") or r.get("sector")
        t = r.get("ticker")
        if c and t in px.columns:
            sec.setdefault(c, []).append(t)
    i = len(px) - 1
    if i < SHORT_LB + 62:
        return []
    rets = px.pct_change() * 100
    out = []
    for c, ts in sec.items():
        ts = [t for t in ts if t in px.columns]
        if len(ts) < SECTOR_MIN_N:
            continue
        sub = px[ts]
        cum10 = (sub.iloc[i] / sub.iloc[i - SHORT_LB] - 1) * 100  # member 10d cumulative %
        cum10 = cum10.dropna()
        if len(cum10) < SECTOR_MIN_N:
            continue
        m = float(cum10.mean())
        csad = float((cum10 - m).abs().mean()) or 1e-9
        # dispersion compression vs own trailing (daily x-sec dispersion)
        daily = rets[ts]
        recent_disp = float(daily.iloc[i - SHORT_LB + 1:i + 1].sub(daily.iloc[i - SHORT_LB + 1:i + 1].mean(axis=1), axis=0).abs().mean(axis=1).mean())
        trail_disp = float(daily.iloc[i - 62:i - SHORT_LB + 1].sub(daily.iloc[i - 62:i - SHORT_LB + 1].mean(axis=1), axis=0).abs().mean(axis=1).mean())
        compression = (recent_disp / trail_disp) if trail_disp else 1.0
        herding_index = abs(m) / csad
        # gate: genuine herding = UNUSUAL cross-sectional clustering (dispersion compressed
        # vs the sector's OWN recent history, <0.85) + a meaningful directional move. The
        # compression gate is what excludes mechanical low-dispersion factor moves (e.g. a
        # coordinated rate move in long-duration FI has NORMAL dispersion, not compressed).
        # <0.90 = recent dispersion ≥10% below the sector's own trailing norm (unusual).
        flag = bool(abs(m) >= SECTOR_MIN_MOVE and herding_index >= 0.5 and compression < 0.90)
        out.append(dict(
            sector=c, n=len(cum10),
            direction=round(m, 2), dispersion=round(csad, 2),
            herding_index=round(herding_index, 2),
            compression=round(compression, 2),
            herding_flag=flag,
        ))
    out.sort(key=lambda d: -d["herding_index"])
    return out


# ── main entry ────────────────────────────────────────────────────
def compute_herding_reversal(results: list, cache_path: str = _CACHE_PATH,
                             options_flow: dict | None = None) -> dict:
    """Return {per_ticker: {t: {...}}, market: {...}, sectors: [...], meta: {...}}.
    options_flow: optional /api/options-flow-style dict for put-crowding cross-confirm.
    """
    px = _load_close_matrix(cache_path)
    if px is None or px.shape[1] < 20:
        return dict(per_ticker={}, market={}, sectors=[],
                    meta=dict(ok=False, reason="price cache unavailable"))
    per = _per_ticker(px)
    market = _cck_market(px)
    sectors = _sector_herding(px, results)

    # options cross-confirm: put-crowded (put_call_oi≥1.8 or skew≥3) boosts confidence
    xconf = {}
    if options_flow and isinstance(options_flow, dict):
        for row in (options_flow.get("tickers") or []):
            t = row.get("ticker")
            pc = _fin(row.get("put_call_oi")); sk = _fin(row.get("skew"))
            if t and ((pc is not None and pc >= 1.8) or (sk is not None and sk >= 3)):
                xconf[t] = dict(put_call_oi=pc, skew=sk)
    for t, d in per.items():
        if t in xconf and d.get("hrr_flag") in ("REVERSAL_RISK", "WATCH"):
            d["hrr_cross_confirmed"] = True
            d["hrr_score"] = round(min(100.0, d["hrr_score"] + 10.0), 1)
            d["hrr_reasons"].append("옵션 풋과밀 교차확인")
        else:
            d.setdefault("hrr_cross_confirmed", False)

    n_rev = sum(1 for d in per.values() if d.get("hrr_flag") == "REVERSAL_RISK")
    n_watch = sum(1 for d in per.values() if d.get("hrr_flag") == "WATCH")
    n_hold = sum(1 for d in per.values() if d.get("hrr_flag") == "CROWDED_HOLDING")
    return dict(
        per_ticker=per,
        market=market,
        sectors=sectors,
        meta=dict(ok=True, as_of=str(px.index[-1].date()), n_tickers=px.shape[1],
                  n_reversal=n_rev, n_watch=n_watch, n_crowded_holding=n_hold),
    )


if __name__ == "__main__":
    # self-test against the live scan universe
    import api
    api._load_cache()
    res = api.STATE["results"]
    out = compute_herding_reversal(res)
    m = out["meta"]; mk = out["market"]
    print(f"ok={m.get('ok')} as_of={m.get('as_of')} n={m.get('n_tickers')} "
          f"| REVERSAL={m.get('n_reversal')} WATCH={m.get('n_watch')} HOLDING={m.get('n_crowded_holding')}")
    print(f"CCK: b2={mk.get('cck_b2')} t={mk.get('cck_t')} herding={mk.get('market_herding')} "
          f"| CSAD {mk.get('csad_recent')} vs {mk.get('csad_hist')} → {mk.get('csad_state')}")
    print("섹터 herding(⚠flag):")
    for s in out["sectors"][:8]:
        print(f"  {s['sector'][:20]:20} n={s['n']:3} 방향{s['direction']:+.1f}% 분산{s['dispersion']:.1f} "
              f"idx{s['herding_index']:.2f} 압축{s['compression']:.2f} {'⚠쏠림' if s['herding_flag'] else ''}")
    revs = [(t, d) for t, d in out["per_ticker"].items() if d.get("hrr_flag") == "REVERSAL_RISK"]
    print(f"\n반전위험 {len(revs)}종목:", [t for t, _ in revs][:20] or "(없음)")
