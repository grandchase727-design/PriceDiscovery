"""
qvr_agent.py — Quality-Value-Revision Agent (5th Pre-Momentum agent)

Adds a fundamentals-based dimension to the existing 4-agent Pre-Momentum
framework (Microstructure / Macro Regime / Graph Relational / Catalyst).

Sub-signals (each 0-100, cross-sectional percentile rank within stock universe):
  Q (Quality)   : gross_margin + operating_margin + ROE   → robustness/profitability
  V (Value)     : 100 − pctile(forward_PE + PEG + P/B)   → cheapness (inverted)
  R (Revision)  : net_30d EPS revision + ratio_30d        → leading earnings momentum
                  + earnings_growth + bullish_ratio + price-target upside

QVR combined  = 0.30·Q + 0.20·V + 0.50·R
                (R weighted highest — revision is most leading per
                 Chan-Jegadeesh-Lakonishok 1996)

ETFs and stocks without fundamentals → returns neutral 50 (does not penalize).

Source data: `.fundamentals_cache.pkl` (built by fundamentals_pipeline.py)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

# Lazy import — fundamentals_pipeline only needed if cache exists
try:
    from pipelines.fundamentals_pipeline import load_fundamentals_cache
except Exception:
    load_fundamentals_cache = None  # type: ignore

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fundamentals_cache.pkl")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _percentile_rank(value: Optional[float], sorted_values: List[float]) -> float:
    """Cross-sectional percentile rank (0-100) of `value` in `sorted_values`.

    Returns 50 (neutral) if value is None or distribution is empty.
    """
    if value is None or sorted_values is None or len(sorted_values) == 0:
        return 50.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 50.0
    n = len(sorted_values)
    # Binary search position
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return (lo / n) * 100.0


def _avg(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


# ──────────────────────────────────────────────────────────────────────
# QVR Agent
# ──────────────────────────────────────────────────────────────────────

class QVRAgent:
    """
    Quality-Value-Revision Agent.

    Usage:
        cache = load_fundamentals_cache()
        agent = QVRAgent(indices, fundamentals_cache=cache)
        score, signals, summary = agent.score(ticker_result)
    """

    NAME = "qvr"
    LABEL = "QVR"

    # Q/V/R sub-component weights (within each sub-signal)
    Q_WEIGHTS = {"gross_margin": 0.40, "operating_margin": 0.30, "roe": 0.30}
    V_WEIGHTS = {"forward_pe": 0.45, "peg": 0.30, "price_to_book": 0.25}
    # R weights — Finnhub-derived signals (bullish_change_3m, eps_beat_rate,
    # eps_surprise_avg) are added on top of yfinance signals when available.
    # When Finnhub is missing, _weighted_avg falls back to whichever yfinance
    # pieces exist.
    R_WEIGHTS = {
        # yfinance pieces
        "net_30d": 0.20, "ratio_30d": 0.15, "earnings_growth": 0.10,
        "bullish_ratio": 0.10, "upside_pct": 0.05,
        # Finnhub-derived pieces (most leading signals)
        "bullish_change_3m": 0.20, "eps_beat_rate": 0.10, "eps_surprise_avg": 0.10,
    }

    # Final QVR aggregation weights
    QVR_WEIGHTS = {"Q": 0.30, "V": 0.20, "R": 0.50}

    def __init__(self, indices: dict, fundamentals_cache: Optional[dict] = None):
        self.indices = indices

        # Auto-load if not passed
        if fundamentals_cache is None and load_fundamentals_cache is not None:
            fundamentals_cache = load_fundamentals_cache(CACHE_PATH)

        self.fund: dict = fundamentals_cache or {"tickers": {}, "stats": {}}
        self.tickers: Dict[str, dict] = self.fund.get("tickers", {})
        self.has_cache: bool = bool(self.tickers)

        # Pre-compute cross-sectional distributions (sorted ascending)
        self._dist = self._build_distributions()

    # ── Distribution builder ──

    def _build_distributions(self) -> Dict[str, List[float]]:
        """Sorted ascending distributions for each fundamental field over stocks.

        For Q (margin/ROE) we PREFER Finnhub when available since coverage and
        accuracy are typically better; fall back to yfinance.
        For Finnhub-only fields (bullish_change_3m, eps_beat_rate, eps_surprise_avg)
        we build distributions only from tickers that have Finnhub data.
        """
        # Quality
        gross, opm, roe = [], [], []
        # Value
        fpe, peg, pb = [], [], []
        # Revision (yfinance)
        net30, ratio30, egrowth, bullish, upside = [], [], [], [], []
        # Revision (Finnhub-derived)
        bull_change_3m: List[float] = []
        eps_beat_rate: List[float] = []
        eps_surprise_avg: List[float] = []

        for tk, t in self.tickers.items():
            if t.get("asset_type") != "Stock" or not t.get("fetch_ok"):
                continue
            info = t.get("info") or {}
            rev = t.get("revisions") or {}
            rec = t.get("recommendations") or {}
            pt = t.get("price_targets") or {}
            # Finnhub blocks
            fh_metrics = t.get("finnhub_metrics") or {}
            fh_derived = t.get("finnhub_derived") or {}

            def _maybe(lst, v, positive_only=False):
                if v is None:
                    return
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return
                if positive_only and fv <= 0:
                    return
                lst.append(fv)

            # ── Quality: prefer Finnhub (returned as percent → convert to decimal) ──
            fh_gross = fh_metrics.get("grossMarginTTM")
            fh_opm = fh_metrics.get("operatingMarginTTM")
            fh_roe = fh_metrics.get("roeTTM")
            _maybe(gross, fh_gross / 100.0 if fh_gross is not None else info.get("gross_margin"))
            _maybe(opm,   fh_opm   / 100.0 if fh_opm   is not None else info.get("operating_margin"))
            _maybe(roe,   fh_roe   / 100.0 if fh_roe   is not None else info.get("roe"))

            # ── Value: prefer Finnhub PE/PB (yfinance PEG fallback) ──
            fh_pe = fh_metrics.get("peNormalizedAnnual") or fh_metrics.get("peTTM")
            fh_pb = fh_metrics.get("pbAnnual") or fh_metrics.get("pbQuarterly")
            _maybe(fpe, fh_pe if fh_pe is not None else info.get("forward_pe"), positive_only=True)
            _maybe(peg, info.get("peg"), positive_only=True)
            _maybe(pb,  fh_pb if fh_pb is not None else info.get("price_to_book"), positive_only=True)

            # ── Revision (yfinance) ──
            _maybe(net30, rev.get("net_30d"))
            _maybe(ratio30, rev.get("ratio_30d"))
            _maybe(egrowth, info.get("earnings_growth"))
            _maybe(bullish, rec.get("bullish_ratio"))
            _maybe(upside, pt.get("upside_pct"))

            # ── Revision (Finnhub-derived) ──
            _maybe(bull_change_3m, fh_derived.get("bullish_change_3m"))
            _maybe(eps_beat_rate, fh_derived.get("eps_beat_rate"))
            _maybe(eps_surprise_avg, fh_derived.get("eps_surprise_avg"))

        return {
            "gross_margin": sorted(gross),
            "operating_margin": sorted(opm),
            "roe": sorted(roe),
            "forward_pe": sorted(fpe),
            "peg": sorted(peg),
            "price_to_book": sorted(pb),
            "net_30d": sorted(net30),
            "ratio_30d": sorted(ratio30),
            "earnings_growth": sorted(egrowth),
            "bullish_ratio": sorted(bullish),
            "upside_pct": sorted(upside),
            # Finnhub-derived
            "bullish_change_3m": sorted(bull_change_3m),
            "eps_beat_rate": sorted(eps_beat_rate),
            "eps_surprise_avg": sorted(eps_surprise_avg),
        }

    # ── Scoring ──

    def score(self, r: dict) -> Tuple[float, Dict[str, float], str]:
        """Compute (qvr_score, signals_dict, summary_string) for a ticker."""
        tk = r.get("ticker", "")
        t = self.tickers.get(tk)

        # No data → neutral signal (50). Won't penalize ETFs / missing fundamentals.
        if not t or not t.get("fetch_ok") or t.get("asset_type") != "Stock":
            return 50.0, {
                "quality": 50.0, "value": 50.0, "revision": 50.0,
                "net_30d": 0, "ratio_30d": 50, "n_analysts": 0,
            }, "No fundamentals (ETF or missing)"

        info = t.get("info") or {}
        rev = t.get("revisions") or {}
        est = (t.get("estimates") or {}).get("0q", {}) or {}
        rec = t.get("recommendations") or {}
        pt = t.get("price_targets") or {}
        # Finnhub blocks (may be empty)
        fh_metrics = t.get("finnhub_metrics") or {}
        fh_derived = t.get("finnhub_derived") or {}

        # Helper: pick Finnhub value (converted from percent if needed) over yfinance
        def _q_val(fh_field: str, yf_val, scale: float = 1.0):
            fh = fh_metrics.get(fh_field)
            if fh is not None:
                return fh * scale
            return yf_val

        def _v_val(fh_field: str, yf_val):
            fh = fh_metrics.get(fh_field)
            if fh is not None:
                return fh
            return yf_val

        # ── Q: Quality (Finnhub preferred — convert percent → decimal with /100) ──
        q_pieces: List[Tuple[float, float]] = []
        gm = _q_val("grossMarginTTM", info.get("gross_margin"), scale=0.01)
        if gm is not None:
            q_pieces.append((_percentile_rank(gm, self._dist["gross_margin"]),
                             self.Q_WEIGHTS["gross_margin"]))
        om = _q_val("operatingMarginTTM", info.get("operating_margin"), scale=0.01)
        if om is not None:
            q_pieces.append((_percentile_rank(om, self._dist["operating_margin"]),
                             self.Q_WEIGHTS["operating_margin"]))
        roe = _q_val("roeTTM", info.get("roe"), scale=0.01)
        if roe is not None:
            q_pieces.append((_percentile_rank(roe, self._dist["roe"]),
                             self.Q_WEIGHTS["roe"]))
        q_score = self._weighted_avg(q_pieces, default=50.0)

        # ── V: Value (Finnhub preferred for PE/PB; yfinance for PEG) ──
        v_pieces: List[Tuple[float, float]] = []
        pe = _v_val("peNormalizedAnnual", info.get("forward_pe")) or fh_metrics.get("peTTM")
        if pe is not None and pe > 0:
            v_pieces.append((100.0 - _percentile_rank(pe, self._dist["forward_pe"]),
                             self.V_WEIGHTS["forward_pe"]))
        peg_v = info.get("peg")
        if peg_v is not None and peg_v > 0:
            v_pieces.append((100.0 - _percentile_rank(peg_v, self._dist["peg"]),
                             self.V_WEIGHTS["peg"]))
        pb = _v_val("pbAnnual", info.get("price_to_book")) or fh_metrics.get("pbQuarterly")
        if pb is not None and pb > 0:
            v_pieces.append((100.0 - _percentile_rank(pb, self._dist["price_to_book"]),
                             self.V_WEIGHTS["price_to_book"]))
        v_score = self._weighted_avg(v_pieces, default=50.0)

        # ── R: Revision (most leading) ──
        r_pieces: List[Tuple[float, float]] = []
        # yfinance pieces
        if rev.get("net_30d") is not None:
            r_pieces.append((_percentile_rank(rev["net_30d"], self._dist["net_30d"]),
                             self.R_WEIGHTS["net_30d"]))
        if rev.get("ratio_30d") is not None:
            r_pieces.append((_percentile_rank(rev["ratio_30d"], self._dist["ratio_30d"]),
                             self.R_WEIGHTS["ratio_30d"]))
        if info.get("earnings_growth") is not None:
            r_pieces.append((_percentile_rank(info["earnings_growth"], self._dist["earnings_growth"]),
                             self.R_WEIGHTS["earnings_growth"]))
        if rec.get("bullish_ratio") is not None:
            r_pieces.append((_percentile_rank(rec["bullish_ratio"], self._dist["bullish_ratio"]),
                             self.R_WEIGHTS["bullish_ratio"]))
        if pt.get("upside_pct") is not None:
            r_pieces.append((_percentile_rank(pt["upside_pct"], self._dist["upside_pct"]),
                             self.R_WEIGHTS["upside_pct"]))
        # ── Finnhub-derived leading pieces ──
        if fh_derived.get("bullish_change_3m") is not None:
            r_pieces.append((_percentile_rank(fh_derived["bullish_change_3m"], self._dist["bullish_change_3m"]),
                             self.R_WEIGHTS["bullish_change_3m"]))
        if fh_derived.get("eps_beat_rate") is not None:
            r_pieces.append((_percentile_rank(fh_derived["eps_beat_rate"], self._dist["eps_beat_rate"]),
                             self.R_WEIGHTS["eps_beat_rate"]))
        if fh_derived.get("eps_surprise_avg") is not None:
            r_pieces.append((_percentile_rank(fh_derived["eps_surprise_avg"], self._dist["eps_surprise_avg"]),
                             self.R_WEIGHTS["eps_surprise_avg"]))
        r_score = self._weighted_avg(r_pieces, default=50.0)

        # ── Combined QVR ──
        qvr = (q_score * self.QVR_WEIGHTS["Q"]
               + v_score * self.QVR_WEIGHTS["V"]
               + r_score * self.QVR_WEIGHTS["R"])

        # Signals dict (raw + percentile decomposition)
        net30 = int(rev.get("net_30d") or 0)
        ratio30_pct = round((rev.get("ratio_30d") or 0.5) * 100, 0)
        # Prefer Finnhub analyst count when available (richer)
        n_analysts = int(fh_derived.get("rec_total_now") or est.get("n_analysts") or 0)
        # Effective forward PE (Finnhub or yfinance)
        eff_pe = pe if pe is not None else info.get("forward_pe")
        signals = {
            "quality": round(q_score, 1),
            "value": round(v_score, 1),
            "revision": round(r_score, 1),
            "net_30d": net30,
            "ratio_30d": ratio30_pct,
            "n_analysts": n_analysts,
            "fwd_pe": round(eff_pe, 1) if eff_pe else None,
            "earn_growth": round(info.get("earnings_growth") * 100, 1) if info.get("earnings_growth") else None,
            "upside_pct": round(pt.get("upside_pct"), 1) if pt.get("upside_pct") else None,
            # Finnhub-derived
            "bullish_change_3m": (round(fh_derived.get("bullish_change_3m") * 100, 1)
                                  if fh_derived.get("bullish_change_3m") is not None else None),
            "eps_beat_rate": (round(fh_derived.get("eps_beat_rate") * 100, 0)
                              if fh_derived.get("eps_beat_rate") is not None else None),
            "eps_surprise_avg": (round(fh_derived.get("eps_surprise_avg"), 2)
                                 if fh_derived.get("eps_surprise_avg") is not None else None),
        }

        summary = self._summary(qvr, q_score, v_score, r_score, net30, n_analysts,
                                fh_derived)
        return qvr, signals, summary

    # ── Helpers ──

    @staticmethod
    def _weighted_avg(pieces: List[Tuple[float, float]], default: float = 50.0) -> float:
        if not pieces:
            return default
        total_w = sum(w for _, w in pieces)
        if total_w == 0:
            return default
        return sum(v * w for v, w in pieces) / total_w

    @staticmethod
    def _summary(qvr: float, q: float, v: float, r: float,
                 net30: int, n_analysts: int,
                 fh_derived: Optional[dict] = None) -> str:
        if n_analysts == 0:
            return "Limited analyst coverage"
        parts = []
        if qvr >= 70:
            parts.append("Strong fundamentals (Q+V+R aligned)")
        elif qvr >= 60:
            parts.append("Bullish fundamentals")
        elif qvr <= 30:
            parts.append("Weak fundamentals")
        elif qvr <= 40:
            parts.append("Bearish fundamentals")
        else:
            parts.append("Neutral fundamentals")

        # Finnhub leading signal — analyst sentiment trend (3m change)
        if fh_derived:
            chg3m = fh_derived.get("bullish_change_3m")
            if chg3m is not None:
                if chg3m >= 0.05:
                    parts.append(f"analysts +{chg3m*100:.0f}% more bullish (3m)")
                elif chg3m <= -0.05:
                    parts.append(f"analysts {chg3m*100:.0f}% less bullish (3m)")
            beat = fh_derived.get("eps_beat_rate")
            if beat is not None and beat >= 0.75:
                parts.append("consistently beats EPS estimates")
            elif beat is not None and beat <= 0.25:
                parts.append("frequently misses EPS estimates")

        # yfinance revision counts (fallback / supplemental)
        if net30 >= 5:
            parts.append(f"+{net30} net EPS revisions (30d)")
        elif net30 <= -5:
            parts.append(f"{net30} net EPS revisions (30d, downward)")

        # Quality / Value flags
        if q >= 75 and v >= 60:
            parts.append("quality at reasonable price")
        elif q >= 75 and v <= 35:
            parts.append("high quality but expensive")
        elif v >= 75 and r >= 60:
            parts.append("cheap with positive revisions")

        return " · ".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity test
    cache = load_fundamentals_cache() if load_fundamentals_cache else None
    if cache is None:
        print("No fundamentals cache found — run fundamentals_pipeline.py first")
        exit(1)

    agent = QVRAgent(indices={}, fundamentals_cache=cache)
    print(f"Loaded {len(agent.tickers)} tickers from cache")
    print(f"Distribution sizes: " +
          ", ".join(f"{k}={len(v)}" for k, v in agent._dist.items()))

    # Test on a few sample tickers
    test_tickers = ["NVDA", "AAPL", "LLY", "META", "JPM", "XOM", "BA", "TSLA"]
    print(f"\n{'Ticker':<8} {'QVR':>6} {'Q':>6} {'V':>6} {'R':>6}  Summary")
    print("-" * 100)
    for tk in test_tickers:
        r = {"ticker": tk}
        qvr, sig, summary = agent.score(r)
        print(f"{tk:<8} {qvr:>6.1f} "
              f"{sig['quality']:>6.1f} {sig['value']:>6.1f} {sig['revision']:>6.1f}  {summary}")
