"""
qvr_agent.py — Quality-Value-Revision Agent (5th Pre-Momentum agent)

Institutional-grade fundamental score. Adds a fundamentals dimension to the
4-agent Pre-Momentum framework (Microstructure / Macro Regime / Graph / Catalyst),
designed the way a drawdown-averse PM + credit committee weigh a company —
a franchise-RISK / credit-quality overlay implemented with systematic
multi-factor (AQR / MSCI-Barra) hygiene.

────────────────────────────────────────────────────────────────────────
DESIGN (2026-06 institutional redesign — judge-panel synthesized)

  QVR = 0.45·Q + 0.25·V + 0.30·R       (was 0.30/0.20/0.50)

  Q — Quality (0.45, dominant: solvency + durable capital-efficient profit)
    roic_capital_efficiency  0.22  ↑  roiTTM(≈ROIC, Greenblatt) → roa
    gross_profitability       0.18  ↑  grossMarginTTM (Novy-Marx) → gross_margin
    net_interest_coverage     0.16  ↑  netInterestCoverageTTM  (winsor[-5,50], US-only)
    leverage_debt_to_equity   0.14  ↓  totalDebt/totalEquity → debt_to_equity/100
    margin_persistence_5y     0.16  ↑  netProfitMargin5Y (FF RMW) → profit_margin
    fcf_cash_conversion       0.14  ↑  DERIVED fcf/ocf (earnings quality)

  V — Value (0.25, EV-based to kill leverage-inflated value traps; ALL inverted ↓)
    ev_ebitda                 0.34  ↓  evEbitdaTTM (Greenblatt) → DERIVED ev/fcf
    ev_fcf                    0.26  ↓  currentEv/freeCashFlowTTM → DERIVED ev/fcf
    ev_sales                  0.22  ↓  evRevenueTTM → price_to_sales
    book_to_price             0.18  ↓  pbQuarterly → price_to_book

  R — Revision (0.30, downside-asymmetric: cuts weighted 2× per CJL-1996)
    revision_asymmetry        0.26  ↑  DERIVED (up_30d − 2·down_30d)/Σ → net_30d
    net_eps_revisions_30d     0.18  ↑  net_30d → ratio_30d
    analyst_rec_trend_3m      0.16  ↑  bullish_change_3m → bullish_ratio
    downgrade_pressure        0.14  ↓  bearish_ratio
    eps_beat_consistency      0.10  ↑  eps_beat_rate (US-only)
    eps_surprise_magnitude    0.08  ↑  eps_surprise_avg (US-only)
    target_upside             0.08  ↑  upside_pct (winsor[-150,150])

MECHANICS
  • Per-factor DIRECTION flag — lower_better → 100 − percentile (one code path).
  • SECTOR-NEUTRAL percentile for Q+V (margins/leverage/multiples are
    structurally sector-driven); n<8 sector → universe fallback. R stays universe-wide.
  • KOREA (.KS) SEPARATE POOL — 0% Finnhub; ranked within a Korea pool so missing
    Finnhub factors don't middle-bias against US (they DROP + renormalize).
  • WINSORIZATION before sort (interest coverage [-5,50]; fcf conv [-0.5,1.5];
    upside [-150,150]). Same clip applied at score time.
  • SCALE normalization — yf debt_to_equity /100 (percent→ratio); Finnhub margins ×0.01.
  • Missing sub-factor → drop + renormalize. Dimension <40% nominal weight present →
    blend 50/50 with neutral 50. ETF / no-fundamentals → 50 neutral (no penalty).
  • ORTHOGONALITY — every factor is balance-sheet / estimate / recommendation derived;
    NO price-return inputs → preserves ~0 correlation to the price-momentum Composite.
  • FINANCIALS SECTOR OVERRIDE — banks/insurers get a separate Q & V recipe
    (ROE + ROA + 5Y book-value growth · P/B + P/E + dividend yield) because the
    industrial factors (ROIC / FCF / interest-coverage / gross-margin / EV-multiples)
    don't apply to deposit-funded balance sheets. R is shared. See FACTORS_FINANCIALS.

Source data: `.fundamentals_cache.pkl` (fundamentals_pipeline.py + finnhub_fundamentals.py)
Sector map:  `.unified_classification.json` (gics_sector per ticker)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

# Lazy import — fundamentals_pipeline only needed if cache exists
try:
    from fundamentals_pipeline import load_fundamentals_cache
except Exception:
    load_fundamentals_cache = None  # type: ignore

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fundamentals_cache.pkl")
UNIFIED_CLASS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".unified_classification.json")


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
# Factor specification (declarative)
#   primary / fallback = (source, field, scale)
#     source ∈ {fh, fhd, yf, rev, rec, pt, derived}
#     scale  multiplies the raw value (e.g. 0.01 percent→decimal)
#   dir    ∈ {higher, lower}      lower → score = 100 − percentile
#   winsor = (lo, hi) clip applied at BOTH build & score time, or None
#   pos    = True → non-positive values dropped (meaningless multiples / neg-equity)
# ──────────────────────────────────────────────────────────────────────

FACTORS: Dict[str, List[dict]] = {
    "Q": [
        {"name": "roic_capital_efficiency", "w": 0.22, "dir": "higher",
         "primary": ("fh", "roiTTM", 0.01), "fallback": ("yf", "roa", 1.0)},
        {"name": "gross_profitability", "w": 0.18, "dir": "higher",
         "primary": ("fh", "grossMarginTTM", 0.01), "fallback": ("yf", "gross_margin", 1.0)},
        {"name": "net_interest_coverage", "w": 0.16, "dir": "higher",
         "primary": ("fh", "netInterestCoverageTTM", 1.0), "fallback": None,
         "winsor": (-5.0, 50.0)},
        {"name": "leverage_debt_to_equity", "w": 0.14, "dir": "lower", "pos": True,
         "primary": ("fh", "totalDebt/totalEquityQuarterly", 1.0),
         "fallback": ("yf", "debt_to_equity", 0.01)},
        {"name": "margin_persistence_5y", "w": 0.16, "dir": "higher",
         "primary": ("fh", "netProfitMargin5Y", 0.01), "fallback": ("yf", "profit_margin", 1.0)},
        {"name": "fcf_cash_conversion", "w": 0.14, "dir": "higher",
         "primary": ("derived", "fcf_conversion", 1.0), "fallback": None},
    ],
    "V": [
        {"name": "ev_ebitda", "w": 0.34, "dir": "lower", "pos": True,
         "primary": ("fh", "evEbitdaTTM", 1.0), "fallback": ("derived", "ev_fcf_yf", 1.0)},
        {"name": "ev_fcf", "w": 0.26, "dir": "lower", "pos": True,
         "primary": ("fh", "currentEv/freeCashFlowTTM", 1.0), "fallback": ("derived", "ev_fcf_yf", 1.0)},
        {"name": "ev_sales", "w": 0.22, "dir": "lower", "pos": True,
         "primary": ("fh", "evRevenueTTM", 1.0), "fallback": ("yf", "price_to_sales", 1.0)},
        {"name": "book_to_price", "w": 0.18, "dir": "lower", "pos": True,
         "primary": ("fh", "pbQuarterly", 1.0), "fallback": ("yf", "price_to_book", 1.0)},
    ],
    "R": [
        {"name": "revision_asymmetry", "w": 0.26, "dir": "higher",
         "primary": ("derived", "revision_asymmetry", 1.0), "fallback": ("rev", "net_30d", 1.0)},
        {"name": "net_eps_revisions_30d", "w": 0.18, "dir": "higher",
         "primary": ("rev", "net_30d", 1.0), "fallback": ("rev", "ratio_30d", 1.0)},
        {"name": "analyst_rec_trend_3m", "w": 0.16, "dir": "higher",
         "primary": ("fhd", "bullish_change_3m", 1.0), "fallback": ("rec", "bullish_ratio", 1.0)},
        {"name": "downgrade_pressure", "w": 0.14, "dir": "lower",
         "primary": ("rec", "bearish_ratio", 1.0), "fallback": None},
        {"name": "eps_beat_consistency", "w": 0.10, "dir": "higher",
         "primary": ("fhd", "eps_beat_rate", 1.0), "fallback": None},
        {"name": "eps_surprise_magnitude", "w": 0.08, "dir": "higher",
         "primary": ("fhd", "eps_surprise_avg", 1.0), "fallback": None},
        {"name": "target_upside", "w": 0.08, "dir": "higher",
         "primary": ("pt", "upside_pct", 1.0), "fallback": ("rec", "bullish_ratio", 1.0),
         "winsor": (-150.0, 150.0)},
    ],
}

# ── Financials sector override (Q & V only; R shared) ─────────────────
# Industrial factors (ROIC / FCF / interest-coverage / gross-margin / EV-multiples)
# misfire on banks/insurers: no clean COGS or FCF, leverage is the business model,
# and EV doesn't apply to deposit-funded balance sheets. Bank/insurance analysts
# score on ROE, ROA (DuPont leverage cross-check), book-value compounding, and
# P/B + P/E + dividend — the metrics that actually exist in the cache.
#
# Lean & non-redundant by design (per adversarial review):
#   • ROE only (NOT ROE+ROTE — rank-corr 0.90); P/B only (NOT P/B+P/TBV — corr 0.96
#     + goodwill would double-count ROTE↑ vs P/TBV↓ from the same BVPS/TBVPS ratio).
#   • margin_persistence dropped — it's a DuPont component of ROE (double-count).
#   • No DERIVED fields → no engine change; avoids the P/TBV winsor-wrong-term bug.
FACTORS_FINANCIALS: Dict[str, List[dict]] = {
    "Q": [
        {"name": "fin_roe", "w": 0.45, "dir": "higher",
         "primary": ("fh", "roeTTM", 0.01), "fallback": ("yf", "roe", 1.0)},
        {"name": "fin_roa", "w": 0.30, "dir": "higher",          # asset efficiency = leverage discipline
         "primary": ("fh", "roaTTM", 0.01), "fallback": ("yf", "roa", 1.0)},
        {"name": "fin_bv_growth_5y", "w": 0.25, "dir": "higher",  # franchise value compounding
         "primary": ("fh", "bookValueShareGrowth5Y", 1.0), "fallback": None},
    ],
    "V": [
        {"name": "fin_pb", "w": 0.45, "dir": "lower", "pos": True,   # P/B = PRIMARY bank multiple
         "primary": ("fh", "pbQuarterly", 1.0), "fallback": ("yf", "price_to_book", 1.0),
         "winsor": (0.2, 25.0)},                                     # clip corrupt near-zero/giant P/B
        {"name": "fin_pe", "w": 0.35, "dir": "lower", "pos": True,   # earnings multiple (orthogonal lens)
         "primary": ("fh", "peTTM", 1.0), "fallback": ("yf", "trailing_pe", 1.0),
         "winsor": (2.0, 80.0)},
        {"name": "fin_div_yield", "w": 0.20, "dir": "higher", "pos": True,  # capital return / yield
         "primary": ("fh", "currentDividendYieldTTM", 1.0), "fallback": ("yf", "dividend_yield", 1.0)},
    ],
}

FINANCIALS_SECTOR = "Financials"   # gics_sector value that triggers the override

QVR_WEIGHTS = {"Q": 0.45, "V": 0.25, "R": 0.30}
MIN_DIM_COVERAGE = 0.40   # <40% nominal weight present → blend 50/50 with neutral


def _factors_for(sector: str, dim: str) -> List[dict]:
    """Sector-conditional factor set. Financials get the bank/insurance Q & V
    recipe; all other sectors use the industrial default. R is shared (sector-agnostic)."""
    if dim in ("Q", "V") and sector == FINANCIALS_SECTOR:
        return FACTORS_FINANCIALS[dim]
    return FACTORS[dim]


def _get_field(t: dict, source: str, field: str):
    if source == "fh":  return (t.get("finnhub_metrics") or {}).get(field)
    if source == "fhd": return (t.get("finnhub_derived") or {}).get(field)
    if source == "yf":  return (t.get("info") or {}).get(field)
    if source == "rev": return (t.get("revisions") or {}).get(field)
    if source == "rec": return (t.get("recommendations") or {}).get(field)
    if source == "pt":  return (t.get("price_targets") or {}).get(field)
    return None


def _derive(name: str, t: dict):
    """Derived fundamental primitives (computed from cache fields)."""
    info = t.get("info") or {}
    rev = t.get("revisions") or {}
    if name == "fcf_conversion":
        fcf, ocf = info.get("free_cash_flow"), info.get("operating_cf")
        if fcf is None or ocf in (None, 0):
            return None
        try:
            v = float(fcf) / float(ocf)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return max(-0.5, min(1.5, v))               # clamp (earnings-quality bounds)
    if name == "ev_fcf_yf":
        ev, fcf = info.get("enterprise_value"), info.get("free_cash_flow")
        try:
            ev, fcf = float(ev), float(fcf)
        except (TypeError, ValueError):
            return None
        if fcf <= 0:
            return None                              # negative FCF → not a meaningful multiple
        return ev / fcf
    if name == "revision_asymmetry":
        up, dn = rev.get("up_30d"), rev.get("down_30d")
        if up is None and dn is None:
            return None
        up, dn = float(up or 0), float(dn or 0)
        return (up - 2.0 * dn) / max(1.0, up + dn)  # LAMBDA=2.0 on cuts
    return None


def _factor_value(fac: dict, t: dict) -> Optional[float]:
    """Resolve a factor's raw value (primary → fallback), with scale, winsor, pos-filter."""
    for slot in ("primary", "fallback"):
        spec = fac.get(slot)
        if not spec:
            continue
        source, field, scale = spec
        raw = _derive(field, t) if source == "derived" else _get_field(t, source, field)
        if raw is None:
            continue
        try:
            v = float(raw) * scale
        except (TypeError, ValueError):
            continue
        if fac.get("pos") and v <= 0:
            continue                                 # drop meaningless multiples / neg-equity
        w = fac.get("winsor")
        if w:
            v = max(w[0], min(w[1], v))
        return v
    return None


# ──────────────────────────────────────────────────────────────────────
# QVR Agent
# ──────────────────────────────────────────────────────────────────────

class QVRAgent:
    """Quality-Value-Revision Agent (institutional multi-factor).

    Usage:
        cache = load_fundamentals_cache()
        agent = QVRAgent(indices, fundamentals_cache=cache)
        score, signals, summary = agent.score(ticker_result)
    """

    NAME = "qvr"
    LABEL = "QVR"

    # Backward-compat: expose top-level weights (some callers read these)
    QVR_WEIGHTS = QVR_WEIGHTS

    def __init__(self, indices: dict, fundamentals_cache: Optional[dict] = None):
        self.indices = indices

        if fundamentals_cache is None and load_fundamentals_cache is not None:
            fundamentals_cache = load_fundamentals_cache(CACHE_PATH)

        self.fund: dict = fundamentals_cache or {"tickers": {}, "stats": {}}
        self.tickers: Dict[str, dict] = self.fund.get("tickers", {})
        self.has_cache: bool = bool(self.tickers)

        # Sector map (gics_sector per ticker) for sector-neutral ranking
        self._sector: Dict[str, str] = self._load_sectors()

        # Cross-sectional distributions: dist[factor_name][pool_key] = sorted ascending
        self._dist = self._build_distributions()

    # ── Sector / region helpers ──

    @staticmethod
    def _load_sectors() -> Dict[str, str]:
        try:
            uc = json.load(open(UNIFIED_CLASS_PATH, encoding="utf-8"))
            ut = uc.get("tickers", uc)
            return {tk: (v.get("gics_sector") or "")
                    for tk, v in ut.items() if isinstance(v, dict)}
        except Exception:
            return {}

    @staticmethod
    def _is_korea(ticker: str) -> bool:
        t = (ticker or "").upper()
        return t.endswith(".KS") or t.endswith(".KQ")

    def _pool_of(self, ticker: str, dim: str) -> str:
        """Percentile pool key. Korea → separate 'KS' pool (all dims).
        Q/V → sector-neutral; R → universe-wide. US universe = 'US'."""
        if self._is_korea(ticker):
            return "KS"
        if dim in ("Q", "V"):
            sec = self._sector.get(ticker, "")
            return f"S::{sec}" if sec else "US"
        return "US"

    # ── Distribution builder ──

    def _build_distributions(self) -> Dict[str, Dict[str, List[float]]]:
        """Per-(factor, pool) sorted distributions.

        For each US stock a Q/V factor value lands in BOTH its sector pool
        ('S::Sector') and the universe pool ('US') so n<8 sectors can fall back.
        R factors land in 'US' only. Korea stocks land in 'KS' only.
        Winsorization/scale/pos-filter applied by _factor_value before pooling.
        """
        raw: Dict[str, Dict[str, List[float]]] = {}

        def _push(fac_name: str, pool: str, val: float):
            raw.setdefault(fac_name, {}).setdefault(pool, []).append(val)

        for tk, t in self.tickers.items():
            if t.get("asset_type") != "Stock" or not t.get("fetch_ok"):
                continue
            is_kr = self._is_korea(tk)
            sec = self._sector.get(tk, "")
            for dim in ("Q", "V", "R"):
                for fac in _factors_for(sec, dim):                 # sector-conditional set
                    v = _factor_value(fac, t)
                    if v is None:
                        continue
                    name = fac["name"]
                    if is_kr:
                        _push(name, "KS", v)
                    else:
                        _push(name, "US", v)                       # universe pool
                        if dim in ("Q", "V") and sec:
                            _push(name, f"S::{sec}", v)            # sector pool

        # Sort every pool ascending
        return {fn: {pk: sorted(vals) for pk, vals in pools.items()}
                for fn, pools in raw.items()}

    # ── Scoring ──

    def _factor_pct(self, fac: dict, dim: str, ticker: str, value: float) -> float:
        """Percentile score (0-100) for a factor value, with sector→universe fallback
        and direction inversion."""
        pool = self._pool_of(ticker, dim)
        dist_for = self._dist.get(fac["name"], {})
        chosen = dist_for.get(pool)
        if pool.startswith("S::") and (chosen is None or len(chosen) < 8):
            chosen = dist_for.get("US")               # small-sector → universe fallback
        pct = _percentile_rank(value, chosen or [])
        if fac["dir"] == "lower":
            pct = 100.0 - pct
        return pct

    def _dim_score(self, dim: str, ticker: str, t: dict,
                   drill: Dict[str, float]) -> float:
        """Weighted, renormalized, coverage-guarded dimension score."""
        sector = self._sector.get(ticker, "")
        facs = _factors_for(sector, dim)
        present_w = 0.0
        acc = 0.0
        nominal_w = sum(f["w"] for f in facs)
        for fac in facs:
            v = _factor_value(fac, t)
            if v is None:
                continue
            pct = self._factor_pct(fac, dim, ticker, v)
            drill[fac["name"]] = round(pct, 1)
            acc += pct * fac["w"]
            present_w += fac["w"]
        if present_w <= 0:
            return 50.0                               # no data → neutral
        score = acc / present_w                       # renormalize over present factors
        if nominal_w > 0 and (present_w / nominal_w) < MIN_DIM_COVERAGE:
            score = 0.5 * score + 0.5 * 50.0          # thin coverage → blend toward neutral
        return score

    def score(self, r: dict) -> Tuple[float, Dict[str, float], str]:
        """Compute (qvr_score, signals_dict, summary_string) for a ticker."""
        tk = r.get("ticker", "")
        t = self.tickers.get(tk)

        # No data → neutral (50). Won't penalize ETFs / missing fundamentals.
        if not t or not t.get("fetch_ok") or t.get("asset_type") != "Stock":
            return 50.0, {
                "quality": 50.0, "value": 50.0, "revision": 50.0,
                "net_30d": 0, "ratio_30d": 50, "n_analysts": 0,
            }, "No fundamentals (ETF or missing)"

        drill: Dict[str, float] = {}
        q_score = self._dim_score("Q", tk, t, drill)
        v_score = self._dim_score("V", tk, t, drill)
        r_score = self._dim_score("R", tk, t, drill)

        qvr = (q_score * QVR_WEIGHTS["Q"]
               + v_score * QVR_WEIGHTS["V"]
               + r_score * QVR_WEIGHTS["R"])

        # Raw drill-down values for display
        info = t.get("info") or {}
        rev = t.get("revisions") or {}
        rec = t.get("recommendations") or {}
        pt = t.get("price_targets") or {}
        fh_derived = t.get("finnhub_derived") or {}
        est = (t.get("estimates") or {}).get("0q", {}) or {}

        net30 = int(rev.get("net_30d") or 0)
        ratio30_pct = round((rev.get("ratio_30d") or 0.5) * 100, 0)
        n_analysts = int(fh_derived.get("rec_total_now") or est.get("n_analysts") or 0)

        sector = self._sector.get(tk, "")
        is_fin = (sector == FINANCIALS_SECTOR)

        def _rawval(fac_list, idx):
            v = _factor_value(fac_list[idx], t)
            return v

        # Sector-aware raw drill-down (Financials show ROE/ROA/P/B; others ROIC/leverage/EV)
        if is_fin:
            _roe = _rawval(FACTORS_FINANCIALS["Q"], 0)
            _roa = _rawval(FACTORS_FINANCIALS["Q"], 1)
            _pb = _rawval(FACTORS_FINANCIALS["V"], 0)
            raw_extra = {
                "sector_model": "Financials",
                "roe": round(_roe, 3) if _roe is not None else None,
                "roa": round(_roa, 4) if _roa is not None else None,
                "pb": round(_pb, 2) if _pb is not None else None,
                "roic": None, "leverage_de": None, "ev_ebitda": None, "fcf_conversion": None,
            }
        else:
            _roic = _rawval(FACTORS["Q"], 0)
            _lev = _rawval(FACTORS["Q"], 3)
            _evb = _rawval(FACTORS["V"], 0)
            raw_extra = {
                "sector_model": "Industrial",
                "roic": round(_roic, 3) if _roic is not None else None,
                "leverage_de": round(_lev, 2) if _lev is not None else None,
                "ev_ebitda": round(_evb, 1) if _evb is not None else None,
                "fcf_conversion": round(_derive("fcf_conversion", t), 2) if _derive("fcf_conversion", t) is not None else None,
            }

        signals = {
            # backward-compat keys
            "quality": round(q_score, 1),
            "value": round(v_score, 1),
            "revision": round(r_score, 1),
            "net_30d": net30,
            "ratio_30d": ratio30_pct,
            "n_analysts": n_analysts,
            # raw fundamentals (sector-aware display)
            **raw_extra,
            "revision_asym": round(_derive("revision_asymmetry", t), 2) if _derive("revision_asymmetry", t) is not None else None,
            "bearish_ratio": round(rec.get("bearish_ratio"), 2) if rec.get("bearish_ratio") is not None else None,
            "bullish_change_3m": (round(fh_derived.get("bullish_change_3m") * 100, 1)
                                  if fh_derived.get("bullish_change_3m") is not None else None),
            "eps_beat_rate": (round(fh_derived.get("eps_beat_rate") * 100, 0)
                              if fh_derived.get("eps_beat_rate") is not None else None),
            "eps_surprise_avg": (round(fh_derived.get("eps_surprise_avg"), 2)
                                 if fh_derived.get("eps_surprise_avg") is not None else None),
            "upside_pct": round(pt.get("upside_pct"), 1) if pt.get("upside_pct") is not None else None,
            # per-factor percentile decomposition
            "factors": {k: v for k, v in drill.items()},
        }

        summary = self._summary(qvr, q_score, v_score, r_score, net30, n_analysts,
                                fh_derived, rec)
        return qvr, signals, summary

    # ── Helpers ──

    @staticmethod
    def _summary(qvr: float, q: float, v: float, r: float, net30: int,
                 n_analysts: int, fh_derived: Optional[dict] = None,
                 rec: Optional[dict] = None) -> str:
        if n_analysts == 0 and not (fh_derived or {}).get("bullish_change_3m"):
            base = ("Strong fundamentals" if qvr >= 65 else
                    "Weak fundamentals" if qvr <= 35 else "Neutral fundamentals")
            return base + " (limited analyst coverage)"
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

        # Quality / solvency flag
        if q >= 75:
            parts.append("durable, well-capitalized franchise")
        elif q <= 30:
            parts.append("fragile balance-sheet / thin margins")

        # Downside-asymmetric revision signal
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
        if rec:
            bear = rec.get("bearish_ratio")
            if bear is not None and bear >= 0.20:
                parts.append("notable downgrade pressure")
        if net30 <= -5:
            parts.append(f"{net30} net EPS revisions (30d, downward)")

        # Quality × Value interaction
        if q >= 75 and v >= 60:
            parts.append("quality at a reasonable EV multiple")
        elif q >= 75 and v <= 35:
            parts.append("high quality but richly valued")
        elif v >= 75 and r >= 60:
            parts.append("cheap (EV-based) with positive revisions")
        return " · ".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cache = load_fundamentals_cache() if load_fundamentals_cache else None
    if cache is None:
        print("No fundamentals cache found — run fundamentals_pipeline.py first")
        exit(1)

    agent = QVRAgent(indices={}, fundamentals_cache=cache)
    print(f"Loaded {len(agent.tickers)} tickers · sector map {len(agent._sector)} entries")
    print("Distribution pool sizes (US universe):")
    for dim, facs in FACTORS.items():
        for fac in facs:
            pools = agent._dist.get(fac["name"], {})
            us = len(pools.get("US", []))
            ks = len(pools.get("KS", []))
            print(f"  [{dim}] {fac['name']:26} US={us:3} KS={ks:2}")

    test_tickers = ["NVDA", "AAPL", "LLY", "META", "JPM", "XOM", "BA", "TSLA", "ROK", "005930.KS"]
    print(f"\n{'Ticker':<11} {'QVR':>6} {'Q':>6} {'V':>6} {'R':>6}  Summary")
    print("-" * 110)
    for tk in test_tickers:
        qvr, sig, summary = agent.score({"ticker": tk})
        print(f"{tk:<11} {qvr:>6.1f} {sig['quality']:>6.1f} {sig['value']:>6.1f} "
              f"{sig['revision']:>6.1f}  {summary}")
