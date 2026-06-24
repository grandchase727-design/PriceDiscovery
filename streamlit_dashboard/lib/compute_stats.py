"""Port of MarketCommentaryTab.tsx computeStats / computeRegimeStats / etc."""
from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from .utils import MOMENTUM_SET, PM_SET, EXCLUDED_SET


def _avg(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0


def _median(xs: list[float]) -> float:
    xs = sorted([x for x in xs if x is not None])
    if not xs:
        return 0
    return statistics.median(xs)


def _std(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0
    return statistics.stdev(xs)


def _top_in_class(rows: list[dict], cls: str, key: str, k: int = 5,
                  reverse: bool = True) -> list[dict]:
    filtered = [r for r in rows if r.get("classification") == cls]
    return sorted(filtered, key=lambda r: r.get(key) or 0, reverse=reverse)[:k]


def compute_stats(history: dict | None, all_results: list[dict]) -> dict:
    """Mirror computeStats() in MarketCommentaryTab.tsx."""
    if not all_results:
        return {}

    total = len(all_results)
    counter = Counter(r.get("classification", "") for r in all_results)
    distribution = dict(counter)

    momentum_n = sum(c for cls, c in counter.items() if cls in MOMENTUM_SET)
    pm_n = sum(c for cls, c in counter.items() if cls in PM_SET)
    excluded_n = sum(c for cls, c in counter.items() if cls in EXCLUDED_SET)

    ytds = [r.get("ytd_return") for r in all_results if r.get("ytd_return") is not None]
    breadth_pct = sum(1 for y in ytds if y > 0) / len(ytds) * 100 if ytds else 0
    avg_ytd = _avg(ytds)
    ytd_median = _median(ytds)
    dispersion = _std(ytds)

    eligible_n = sum(1 for r in all_results if (r.get("composite") or 0) >= 55)
    eligible_pct = eligible_n / total * 100 if total else 0

    # Individual class counts
    cont = counter.get("🟢 CONTINUATION", 0)
    form = counter.get("🔵 FORMATION", 0)
    recv = counter.get("🔵 RECOVERY", 0)
    lag  = counter.get("🟦 LAGGING_CATCHUP", 0)
    oext = counter.get("🟡 OVEREXTENDED", 0)
    peak = counter.get("🔴 CYCLE_PEAK", 0)
    exhaust = counter.get("🟤 EXHAUSTING", 0)
    cons = counter.get("🟡 CONSOLIDATION", 0)
    neut = counter.get("🟠 NEUTRAL", 0)
    pull = counter.get("🔶 PULLBACK", 0)
    weak = counter.get("⚠️ WEAKENING", 0)
    fade = counter.get("🟤 FADING", 0)
    down = counter.get("⬇️ DOWNTREND", 0)

    bull_idx = cont + form + recv + lag
    bear_idx = down + weak + fade + peak + exhaust

    # Regime diagnosis
    if peak + oext + exhaust >= total * 0.25:
        regime = "과열 단계"
    elif bull_idx >= total * 0.4:
        regime = "건전한 상승 추세"
    elif bear_idx >= total * 0.4:
        regime = "약세·방어 국면"
    elif form + recv + lag >= total * 0.25:
        regime = "회복·전환 국면"
    else:
        regime = "혼조 국면"

    # History delta (2-period change)
    gainers: list[dict] = []
    losers: list[dict] = []
    if history and history.get("dates") and history.get("matrix"):
        dates = history["dates"]
        classes = history.get("classifications", [])
        matrix = history["matrix"]
        if len(matrix) >= 2:
            latest_row = matrix[-1]
            prev_row = matrix[-2]
            for i, cls in enumerate(classes):
                delta = latest_row[i] - prev_row[i]
                if delta != 0:
                    rec = {"cls": cls, "delta": delta}
                    if delta > 0: gainers.append(rec)
                    else: losers.append(rec)
            gainers.sort(key=lambda x: -x["delta"])
            losers.sort(key=lambda x: x["delta"])

    # Sector aggregation
    sector_agg: dict[str, dict] = defaultdict(lambda: {
        "mom": 0, "pm": 0, "exc": 0, "total": 0,
        "sum_ytd": 0, "n_ytd": 0,
        "sum_1m": 0, "n_1m": 0,
        "sum_3m": 0, "n_3m": 0,
        "sum_comp": 0, "n_comp": 0,
    })
    for r in all_results:
        sec = r.get("sector", "")
        if not sec: continue
        s = sector_agg[sec]
        s["total"] += 1
        cls = r.get("classification", "")
        if cls in MOMENTUM_SET: s["mom"] += 1
        elif cls in PM_SET:     s["pm"] += 1
        elif cls in EXCLUDED_SET: s["exc"] += 1
        if r.get("ytd_return") is not None:
            s["sum_ytd"] += r["ytd_return"]; s["n_ytd"] += 1
        if r.get("ret_1m") is not None:
            s["sum_1m"] += r["ret_1m"]; s["n_1m"] += 1
        if r.get("ret_3m") is not None:
            s["sum_3m"] += r["ret_3m"]; s["n_3m"] += 1
        if r.get("composite") is not None:
            s["sum_comp"] += r["composite"]; s["n_comp"] += 1

    sector_rows = []
    for sec, s in sector_agg.items():
        if s["total"] < 5: continue
        sector_rows.append({
            "sec": sec,
            "total": s["total"],
            "mom": s["mom"],
            "mom_pct": s["mom"] / s["total"] * 100,
            "exc_pct": s["exc"] / s["total"] * 100,
            "avg_ytd": s["sum_ytd"] / s["n_ytd"] if s["n_ytd"] else 0,
            "avg_1m":  s["sum_1m"] / s["n_1m"] if s["n_1m"] else 0,
            "avg_3m":  s["sum_3m"] / s["n_3m"] if s["n_3m"] else 0,
            "avg_comp": s["sum_comp"] / s["n_comp"] if s["n_comp"] else 0,
        })

    bullish_sec = sorted(sector_rows, key=lambda x: -x["mom_pct"])[:5]
    bearish_sec = [s for s in sector_rows if s["exc_pct"] > 30]
    bearish_sec.sort(key=lambda x: -x["exc_pct"])
    top_ytd_sec = sorted(sector_rows, key=lambda x: -x["avg_ytd"])[:5]
    worst_ytd_sec = [s for s in sector_rows if s["avg_ytd"] < -3]
    worst_ytd_sec.sort(key=lambda x: x["avg_ytd"])
    top_3m_sec = sorted(sector_rows, key=lambda x: -x["avg_3m"])[:5]
    comp_sec = sorted(sector_rows, key=lambda x: -x["avg_comp"])[:5]

    # Industry aggregation
    industry_agg: dict[str, dict] = defaultdict(lambda: {
        "mom": 0, "total": 0, "sum_ytd": 0, "n_ytd": 0,
    })
    for r in all_results:
        ind = r.get("industry", "") or r.get("industry_group", "")
        if not ind: continue
        i = industry_agg[ind]
        i["total"] += 1
        if r.get("classification") in MOMENTUM_SET: i["mom"] += 1
        if r.get("ytd_return") is not None:
            i["sum_ytd"] += r["ytd_return"]; i["n_ytd"] += 1
    industry_rows = []
    for ind, i in industry_agg.items():
        if i["total"] < 4: continue
        industry_rows.append({
            "name": ind,
            "total": i["total"],
            "mom_pct": i["mom"] / i["total"] * 100,
            "avg_ytd": i["sum_ytd"] / i["n_ytd"] if i["n_ytd"] else 0,
        })
    bullish_ind = sorted(industry_rows, key=lambda x: -x["mom_pct"])[:6]
    bearish_ind = sorted(industry_rows, key=lambda x: x["mom_pct"])[:4]

    # Theme aggregation
    theme_agg: dict[str, dict] = defaultdict(lambda: {
        "mom": 0, "total": 0, "sum_ytd": 0, "n_ytd": 0,
        "sum_1m": 0, "n_1m": 0,
    })
    for r in all_results:
        theme = r.get("sub_theme") or r.get("theme")
        if not theme: continue
        t = theme_agg[theme]
        t["total"] += 1
        if r.get("classification") in MOMENTUM_SET: t["mom"] += 1
        if r.get("ytd_return") is not None:
            t["sum_ytd"] += r["ytd_return"]; t["n_ytd"] += 1
        if r.get("ret_1m") is not None:
            t["sum_1m"] += r["ret_1m"]; t["n_1m"] += 1
    theme_rows = []
    for nm, t in theme_agg.items():
        if t["total"] < 3: continue
        theme_rows.append({
            "name": nm,
            "total": t["total"],
            "mom_pct": t["mom"] / t["total"] * 100,
            "avg_ytd": t["sum_ytd"] / t["n_ytd"] if t["n_ytd"] else 0,
            "avg_1m":  t["sum_1m"] / t["n_1m"] if t["n_1m"] else 0,
        })
    bullish_themes = [t for t in theme_rows if t["mom_pct"] >= 50]
    bullish_themes.sort(key=lambda x: -x["mom_pct"])
    bullish_themes = bullish_themes[:6]
    ytd_themes = sorted(theme_rows, key=lambda x: -x["avg_ytd"])[:5]
    weak_themes = [t for t in theme_rows if t["mom_pct"] < 20 and t["avg_ytd"] < 0]
    weak_themes.sort(key=lambda x: x["avg_ytd"])
    weak_themes = weak_themes[:5]

    # Cap-tier aggregation
    cap_agg: dict[str, dict] = defaultdict(lambda: {
        "mom": 0, "pm": 0, "exc": 0, "total": 0, "sum_ytd": 0, "n_ytd": 0,
    })
    for r in all_results:
        tier = (r.get("cap_tier") or "").upper()
        if tier in ("MEGA", "LARGE", "MID", "SMALL"):
            s = cap_agg[tier]
            s["total"] += 1
            cls = r.get("classification", "")
            if cls in MOMENTUM_SET: s["mom"] += 1
            elif cls in PM_SET:     s["pm"] += 1
            elif cls in EXCLUDED_SET: s["exc"] += 1
            if r.get("ytd_return") is not None:
                s["sum_ytd"] += r["ytd_return"]; s["n_ytd"] += 1

    cap_stats = {}
    for tier, s in cap_agg.items():
        if s["total"] == 0: continue
        cap_stats[tier] = {
            "tier": tier,
            "mom_pct": s["mom"] / s["total"] * 100,
            "pm_pct":  s["pm"] / s["total"] * 100,
            "exc_pct": s["exc"] / s["total"] * 100,
            "total": s["total"],
            "avg_ytd": s["sum_ytd"] / s["n_ytd"] if s["n_ytd"] else 0,
        }

    # Geographic aggregation by region heuristic
    def detect_region(r: dict) -> str:
        region = (r.get("region") or "").upper()
        if region: return region
        t = r.get("ticker", "")
        if t.endswith(".KS"): return "KOREA"
        if t.endswith(".T"):  return "JAPAN"
        sec = r.get("sector", "")
        if "Korea" in sec: return "KOREA"
        if "Japan" in sec: return "JAPAN"
        if "Intl" in sec or "International" in sec: return "INTL"
        return "US"

    geo_agg: dict[str, dict] = defaultdict(lambda: {
        "mom": 0, "pm": 0, "exc": 0, "total": 0, "sum_ytd": 0, "n_ytd": 0,
    })
    for r in all_results:
        geo = detect_region(r)
        s = geo_agg[geo]
        s["total"] += 1
        cls = r.get("classification", "")
        if cls in MOMENTUM_SET: s["mom"] += 1
        elif cls in PM_SET:     s["pm"] += 1
        elif cls in EXCLUDED_SET: s["exc"] += 1
        if r.get("ytd_return") is not None:
            s["sum_ytd"] += r["ytd_return"]; s["n_ytd"] += 1

    geo_stats = {}
    for geo, s in geo_agg.items():
        if s["total"] == 0: continue
        geo_stats[geo] = {
            "geo": geo,
            "mom_pct": s["mom"] / s["total"] * 100,
            "pm_pct":  s["pm"] / s["total"] * 100,
            "exc_pct": s["exc"] / s["total"] * 100,
            "total": s["total"],
            "avg_ytd": s["sum_ytd"] / s["n_ytd"] if s["n_ytd"] else 0,
        }

    # Top picks per class
    top_cont = _top_in_class(all_results, "🟢 CONTINUATION", "composite", 5)
    top_form = _top_in_class(all_results, "🔵 FORMATION", "composite", 4)
    top_recv = _top_in_class(all_results, "🔵 RECOVERY", "composite", 4)
    top_lag  = _top_in_class(all_results, "🟦 LAGGING_CATCHUP", "composite", 4)
    top_oext = _top_in_class(all_results, "🟡 OVEREXTENDED", "composite", 5)
    top_peak = _top_in_class(all_results, "🔴 CYCLE_PEAK", "composite", 4)
    top_down = _top_in_class(all_results, "⬇️ DOWNTREND", "composite", 4, reverse=False)
    top_weak = _top_in_class(all_results, "⚠️ WEAKENING", "composite", 4)
    top_fade = _top_in_class(all_results, "🟤 FADING", "composite", 4)

    return {
        "total": total,
        "distribution": distribution,
        "momentum_n": momentum_n,
        "pm_n": pm_n,
        "excluded_n": excluded_n,
        "momentum_pct": momentum_n / total * 100 if total else 0,
        "pm_pct": pm_n / total * 100 if total else 0,
        "excluded_pct": excluded_n / total * 100 if total else 0,
        "breadth_pct": breadth_pct,
        "avg_ytd": avg_ytd,
        "ytd_median": ytd_median,
        "dispersion": dispersion,
        "eligible_n": eligible_n,
        "eligible_pct": eligible_pct,
        "regime": regime,
        "bull_idx": bull_idx,
        "bear_idx": bear_idx,
        # individual class counts
        "cont": cont, "form": form, "recv": recv, "lag": lag,
        "oext": oext, "peak": peak, "exhaust": exhaust,
        "cons": cons, "neut": neut, "pull": pull,
        "weak": weak, "fade": fade, "down": down,
        "gainers": gainers, "losers": losers,
        # Sectors & themes
        "sector_rows": sector_rows,
        "bullish_sec": bullish_sec, "bearish_sec": bearish_sec,
        "top_ytd_sec": top_ytd_sec, "worst_ytd_sec": worst_ytd_sec,
        "top_3m_sec": top_3m_sec, "comp_sec": comp_sec,
        "bullish_ind": bullish_ind, "bearish_ind": bearish_ind,
        "bullish_themes": bullish_themes, "ytd_themes": ytd_themes, "weak_themes": weak_themes,
        # Cap & Geo
        "cap_stats": cap_stats,
        "geo_stats": geo_stats,
        # Top picks
        "top_cont": top_cont, "top_form": top_form, "top_recv": top_recv, "top_lag": top_lag,
        "top_oext": top_oext, "top_peak": top_peak,
        "top_down": top_down, "top_weak": top_weak, "top_fade": top_fade,
    }


def compute_regime_stats(regime_data: dict | None) -> dict:
    """Mirror computeRegimeStats() in MarketCommentaryTab.tsx."""
    if not regime_data:
        return {}
    return {
        "regime": regime_data.get("regime", "UNKNOWN"),
        "breadth": regime_data.get("breadth", {}),
        "agreement": regime_data.get("agreement", 0),
        "strategy_groups": regime_data.get("strategy_groups", []),
        "strategy_breadth": regime_data.get("strategy_breadth", []),
        "sector_regime": regime_data.get("sector_regime", []),
        "regime_history": regime_data.get("regime_history", []),
        "signal_distribution": regime_data.get("signal_distribution", []),
    }


def compute_validation_stats(validation: dict | None) -> dict:
    """Mirror computeValidationStats()."""
    if not validation:
        return {}
    momentum = validation.get("momentum", [])
    pre_momentum = validation.get("pre_momentum", [])

    def overall(arr: list[dict]) -> float:
        scores = [a.get("pass_score", 0) for a in arr]
        return _avg(scores)

    return {
        "summary": validation.get("summary", {}),
        "momentum": momentum,
        "pre_momentum": pre_momentum,
        "overall_mom": overall(momentum),
        "overall_pm": overall(pre_momentum),
        "top_fails": validation.get("top_fails", []),
        "top_fails_pm": validation.get("top_fails_pm", []),
    }


def compute_quant_stats(quant: dict | None) -> dict:
    """Mirror computeQuantStats()."""
    if not quant:
        return {}
    strategies = quant.get("strategies", []) or quant.get("strategy_list", [])
    total_long = sum(s.get("n_long", 0) for s in strategies)
    total_neutral = sum(s.get("n_neutral", 0) for s in strategies)
    total_short = sum(s.get("n_short", 0) for s in strategies)
    n = max(1, len(strategies))
    avg_long = total_long / n
    avg_short = total_short / n
    if avg_long > avg_short * 1.3:
        net_direction = "LONG"
    elif avg_short > avg_long * 1.3:
        net_direction = "SHORT"
    else:
        net_direction = "MIXED"

    # Consensus picks: tickers appearing in 2+ strategy pick lists
    ticker_strategies: dict[str, list[str]] = defaultdict(list)
    for s in strategies:
        for p in s.get("picks", []) or s.get("top_picks", []) or []:
            t = p.get("ticker") if isinstance(p, dict) else p
            if t:
                ticker_strategies[t].append(s.get("name") or s.get("key", "?"))
    consensus = [
        {"ticker": t, "n_strategies": len(set(strats)), "strategies": list(set(strats))}
        for t, strats in ticker_strategies.items()
        if len(set(strats)) >= 2
    ]
    consensus.sort(key=lambda x: -x["n_strategies"])
    ultra_consensus = [c for c in consensus if c["n_strategies"] >= 3]

    return {
        "strategies": strategies,
        "total_long": total_long,
        "total_neutral": total_neutral,
        "total_short": total_short,
        "avg_long": avg_long,
        "avg_neutral": total_neutral / n,
        "avg_short": avg_short,
        "net_direction": net_direction,
        "consensus": consensus[:10],
        "ultra_consensus": ultra_consensus[:6],
    }
