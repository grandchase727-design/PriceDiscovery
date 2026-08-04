# -*- coding: utf-8 -*-
"""portfolio_pit.py — Point-in-time WEEKLY-rebalanced backtest of the swarm Buy Final List.

Reads live-captured daily snapshots from .pm_picks_history.json (the `final_portfolio`
field, populated 2026-07-03 onward after the phase5-capture fix in pm_history.py). Each
snapshot holds the ACTUAL 매수 Final List with conviction weights AS-OF that date, so the
curve is genuinely point-in-time — no look-ahead, no proxy reconstruction.

Mechanics:
  - Rebalance WEEKLY: one portfolio per ISO week (the last snapshot of each week).
  - Hold weights between rebalances; mark-to-market DAILY from the price cache.
  - Chain into a base=1 equity curve per sleeve (stocks / ETFs) vs SPY.
  - Turnover cost applied at each rebalance (default 30 bps, half-turnover).

Because real capture began 2026-07-03, the curve is SHORT and grows ~one week at a time.
Until ≥1 full weekly holding period with forward prices exists, run_pit_backtest returns
status="accumulating" (nothing to plot yet) rather than a fabricated curve.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

HISTORY_PATH = Path(".pm_picks_history.json")
PRICE_CACHE = Path(".backtest_price_cache.pkl")
REAL_SINCE = "2026-07-03"          # first date with a real final_portfolio capture
BENCHMARK = "SPY"
TURNOVER_COST_BPS = 30             # per-side half-turnover cost (matches sector-rotation)


def _load_snapshots() -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        d = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    snaps = [s for s in d.get("snapshots", []) if isinstance(s, dict)]
    snaps.sort(key=lambda s: s.get("date", ""))
    return snaps


def _load_prices():
    """{ticker: pd.Series[Close] indexed by Timestamp}."""
    if not PRICE_CACHE.exists():
        return {}, None
    try:
        import pandas as pd
        data = pickle.load(open(PRICE_CACHE, "rb")).get("data", {})
    except Exception:
        return {}, None
    out = {}
    for tk, df in data.items():
        try:
            if df is not None and "Close" in getattr(df, "columns", []):
                out[tk] = df["Close"].dropna()
        except Exception:
            pass
    last = None
    for s in out.values():
        if len(s):
            last = s.index[-1] if last is None else max(last, s.index[-1])
    return out, last


def _weekly_rebalances(snaps: list[dict]) -> list[dict]:
    """One rebalance per ISO week — the LAST snapshot of each week that has a
    non-empty final_portfolio. Returns [{date, stocks:{tk:w_frac}, etfs:{tk:w_frac}}]."""
    import datetime as _dt
    by_week: dict = {}   # (iso_year, iso_week) -> snapshot (keep latest date)
    for s in snaps:
        fp = s.get("final_portfolio") or {}
        if not (fp.get("stocks") or fp.get("etfs")):
            continue
        try:
            d = _dt.date.fromisoformat(s["date"])
        except Exception:
            continue
        key = d.isocalendar()[:2]
        if key not in by_week or s["date"] > by_week[key]["date"]:
            by_week[key] = s

    def _w(rows):
        w = {r.get("ticker"): float(r.get("weight") or 0) / 100.0
             for r in (rows or []) if r.get("ticker")}
        tot = sum(w.values())
        return {t: x / tot for t, x in w.items()} if tot > 0 else {}

    out = []
    for key in sorted(by_week):
        s = by_week[key]
        fp = s["final_portfolio"]
        out.append({"date": s["date"], "stocks": _w(fp.get("stocks")),
                    "etfs": _w(fp.get("etfs"))})
    return out


def _sleeve_curve(rebals: list[dict], sleeve: str, prices: dict, last_price):
    """Daily base=1 equity for one sleeve, weekly rebalance + daily MTM + turnover cost.
    Returns (dates[list 'YYYY-MM-DD'], values[list float]) — empty if no forward prices."""
    import pandas as pd
    weighted = [(r["date"], r[sleeve]) for r in rebals if r.get(sleeve)]
    if not weighted:
        return [], []

    equity = 1.0
    prev_w: dict = {}
    dates: list[str] = []
    vals: list[float] = []

    for i, (t0s, w) in enumerate(weighted):
        t0 = pd.Timestamp(t0s)
        t_end = pd.Timestamp(weighted[i + 1][0]) if i + 1 < len(weighted) else last_price
        if t_end is None or t_end < t0:
            continue
        # available tickers with a base price on/before t0
        avail = {}
        for tk, wt in w.items():
            s = prices.get(tk)
            if s is None:
                continue
            s0 = s[s.index <= t0]
            if len(s0) == 0:
                continue
            avail[tk] = (wt, float(s0.iloc[-1]), s)
        tot = sum(a[0] for a in avail.values())
        if tot <= 0:
            continue
        avail = {tk: (wt / tot, base, s) for tk, (wt, base, s) in avail.items()}

        # turnover cost at rebalance (½·Σ|w_new − w_prev|·bps)
        cur_w = {tk: v[0] for tk, v in avail.items()}
        keys = set(cur_w) | set(prev_w)
        turnover = 0.5 * sum(abs(cur_w.get(k, 0) - prev_w.get(k, 0)) for k in keys)
        equity *= (1.0 - turnover * TURNOVER_COST_BPS / 1e4)
        prev_w = cur_w

        # daily MTM over [t0, t_end)  (last period includes t_end)
        axis = None
        for tk, (wt, base, s) in avail.items():
            seg = s[(s.index >= t0) & (s.index <= t_end)]
            axis = seg.index if axis is None else axis.union(seg.index)
        if axis is None or len(axis) == 0:
            continue
        axis = axis.sort_values()
        is_last = (i + 1 == len(weighted))
        for d in axis:
            if d == t0 and i > 0:
                continue  # avoid double-counting the boundary day
            if (not is_last) and d == t_end:
                continue  # boundary handled by next period's t0
            v = 0.0
            for tk, (wt, base, s) in avail.items():
                px = s[s.index <= d]
                if len(px) == 0:
                    continue
                v += wt * float(px.iloc[-1]) / base
            dates.append(str(d)[:10])
            vals.append(round(equity * v, 6))
        # advance equity to end-of-period value for next chaining
        v_end = 0.0
        for tk, (wt, base, s) in avail.items():
            px = s[s.index <= t_end]
            if len(px):
                v_end += wt * float(px.iloc[-1]) / base
        if v_end > 0:
            equity = equity * v_end
    return dates, vals


def _benchmark_curve(start: str, prices: dict, last_price):
    import pandas as pd
    s = prices.get(BENCHMARK)
    if s is None:
        return [], []
    t0 = pd.Timestamp(start)
    seg = s[(s.index >= t0) & (s.index <= (last_price or s.index[-1]))]
    if len(seg) == 0:
        return [], []
    base = float(seg.iloc[0])
    if base <= 0:
        return [], []
    return [str(d)[:10] for d in seg.index], [round(float(v) / base, 6) for v in seg.tolist()]


def run_pit_backtest() -> dict:
    """Point-in-time weekly-rebalanced backtest of the real Buy Final List.

    Returns a dict shaped for /api/portfolio's `performance` field:
      {dates, stock_index, etf_index, benchmark_index, rebalance_dates,
       real_since, n_rebalances, status, methodology}
    status: 'ok' | 'accumulating' | 'no_data'
    """
    snaps = _load_snapshots()
    rebals = _weekly_rebalances(snaps)
    real_rebals = [r["date"] for r in rebals]
    methodology = (
        "Point-in-time · 주간 리밸런스 · buy-and-hold within week · conviction-weighted "
        "(as-of capture) · turnover 30bps · vs SPY (ex-dividend). No look-ahead, no proxy — "
        f"membership+weights captured live daily since {REAL_SINCE}."
    )
    if not rebals:
        return {"dates": [], "stock_index": [], "etf_index": [], "benchmark_index": [],
                "rebalance_dates": [], "real_since": REAL_SINCE, "n_rebalances": 0,
                "status": "no_data", "methodology": methodology,
                "note": "아직 실제 final_portfolio 스냅샷이 없습니다 (phase5 캡처 수정 이후 첫 스웜부터 축적)."}

    prices, last_price = _load_prices()
    s_dates, s_vals = _sleeve_curve(rebals, "stocks", prices, last_price)
    e_dates, e_vals = _sleeve_curve(rebals, "etfs", prices, last_price)

    if not s_dates and not e_dates:
        # rebalances exist but no forward prices yet (price cache hasn't advanced past
        # the first rebalance) → honest "accumulating" state, nothing to plot.
        return {"dates": [], "stock_index": [], "etf_index": [], "benchmark_index": [],
                "rebalance_dates": real_rebals, "real_since": REAL_SINCE,
                "n_rebalances": len(rebals), "status": "accumulating",
                "methodology": methodology,
                "note": (f"실제 트랙 시작 {real_rebals[0]} · 리밸 {len(rebals)}주 · "
                         "곡선은 이후 가격이 쌓이면 표시됩니다 (몇 주 소요).")}

    # common date axis (union) for stocks/etfs; benchmark from first rebalance
    start = (s_dates or e_dates)[0]
    b_dates, b_vals = _benchmark_curve(start, prices, last_price)
    all_dates = sorted(set(s_dates) | set(e_dates) | set(b_dates))
    sidx = dict(zip(s_dates, s_vals)); eidx = dict(zip(e_dates, e_vals)); bidx = dict(zip(b_dates, b_vals))
    return {
        "dates": all_dates,
        "stock_index": [sidx.get(d) for d in all_dates],
        "etf_index":   [eidx.get(d) for d in all_dates],
        "benchmark_index": [bidx.get(d) for d in all_dates],
        "rebalance_dates": real_rebals,
        "real_since": REAL_SINCE,
        "n_rebalances": len(rebals),
        "status": "ok",
        "methodology": methodology,
    }


if __name__ == "__main__":
    import pprint
    r = run_pit_backtest()
    r2 = {k: (f"[{len(v)} pts]" if isinstance(v, list) and len(v) > 6 else v) for k, v in r.items()}
    pprint.pprint(r2)
