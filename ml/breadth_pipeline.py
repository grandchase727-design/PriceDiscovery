###############################################################################
# Breadth Pipeline — Historical Monthly Replay of price_discovery Scoring
# ============================================================================
# For each month-end in [2007-07-31, ..., today], slice each ETF's full history
# to that date and recompute: compute_raw → score_tcs/tfs/oer → classify
# Cross-sectional percentile ranks computed per-date.
# Aggregate into 6 breadth features (4 equity + 2 bond).
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Import reusable pieces from the existing scanner. This triggers some
# module-level print statements in price_discovery.py but is acceptable once.
import sys
import contextlib
with contextlib.redirect_stdout(open(os.devnull, "w")):
    from price_discovery import (
        DataEngine, NaiveDiscoveryDetector, GLOBAL_ETF_UNIVERSE,
    )


# ---------------------------------------------------------------------------
# Universe partitioning: equity vs bond ETFs
# ---------------------------------------------------------------------------
EQUITY_CATS = [
    "EQ_Broad", "EQ_Technology", "EQ_Healthcare", "EQ_Financials",
    "EQ_ConsDisc", "EQ_ConsStaples", "EQ_Industrials", "EQ_Energy",
    "EQ_Materials", "EQ_Utilities", "EQ_RealEstate", "EQ_CommServices",
    "EQ_Factor", "EQ_Thematic", "Intl_Developed", "Emerging_Markets",
    "Korea_Equity",
]

BOND_CATS = [
    "FI_Short", "FI_Intermediate", "FI_Long", "FI_Credit",
    "FI_Inflation", "FI_International",
]


BULLISH_SUBSTRINGS = ("CONTINUATION", "RECOVERY", "FORMATION", "LAGGING_CATCHUP")
DOWNTREND_SUBSTRINGS = ("DOWNTREND", "CYCLE_PEAK", "FADING")


# ---------------------------------------------------------------------------
# Cached download
# ---------------------------------------------------------------------------
CACHE_PATH = ".breadth_universe_cache.pkl"


def download_universe_cached(start_date: str = "2007-01-01",
                             categories: List[str] = None,
                             force: bool = False) -> Dict:
    """
    Download ETFs once from `start_date` and cache to disk.
    Returns dict: ticker -> ETFData
    """
    if (not force) and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("start_date") == start_date:
            print(f"[breadth] Using cached universe ({len(cache['data'])} ETFs)")
            return cache["data"]

    categories = categories or (EQUITY_CATS + BOND_CATS)
    # DataEngine's start_date is set via lookback_days. We pass a very large
    # lookback so that start_date matches our target.
    today = pd.Timestamp.today()
    lookback_days = (today - pd.Timestamp(start_date)).days + 1
    eng = DataEngine(lookback_days=lookback_days, custom_date=None, use_realtime=False)
    print(f"[breadth] Downloading universe {start_date} → today "
          f"(categories={len(categories)})…")
    data = eng.download_universe(categories=categories)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"start_date": start_date, "data": data}, f)
    return data


def _category_of(ticker: str, universe: Dict = None) -> str:
    universe = universe or GLOBAL_ETF_UNIVERSE
    for cat, spec in universe.items():
        if ticker in spec["tickers"]:
            return cat
    return "Unknown"


# ---------------------------------------------------------------------------
# Single-date breadth computation
# ---------------------------------------------------------------------------
def compute_breadth_at(as_of: pd.Timestamp, universe_data: Dict,
                       min_days: int = 60) -> Dict[str, float]:
    """
    Slice each ETF's df to [:as_of], recompute raw/score/classify for any
    ticker with >=`min_days` of data, then aggregate breadth features.

    Returns dict of equity + bond breadth features (may contain NaN if a group
    has insufficient tickers).
    """
    det = NaiveDiscoveryDetector()
    all_raw: Dict[str, dict] = {}
    ticker_category: Dict[str, str] = {}

    for ticker, etf in universe_data.items():
        if etf is None or not etf.valid or etf.df is None:
            continue
        df_slice = etf.df[etf.df.index <= as_of]
        if len(df_slice) < min_days:
            continue
        try:
            raw = det.compute_raw(df_slice, category=etf.category)
        except Exception:
            continue
        raw["_category"] = etf.category
        all_raw[ticker] = raw
        ticker_category[ticker] = etf.category

    if len(all_raw) < 20:
        return {}

    # Cross-sectional ranks (populates rss, urs, etc. in each ranks[t])
    try:
        ranks = NaiveDiscoveryDetector.compute_percentile_ranks(all_raw)
    except Exception as e:
        print(f"[breadth] ranks failed at {as_of.date()}: {e}")
        return {}

    # Classify each ticker
    classifications: Dict[str, str] = {}
    tcs_values: Dict[str, float] = {}
    rss_values: Dict[str, float] = {}
    for t, raw in all_raw.items():
        tcs_s = NaiveDiscoveryDetector.score_tcs_short(raw)
        tcs_l = NaiveDiscoveryDetector.score_tcs_long(raw)
        tcs   = NaiveDiscoveryDetector.score_tcs(raw)
        tfs_s = NaiveDiscoveryDetector.score_tfs_short(raw)
        tfs_l = NaiveDiscoveryDetector.score_tfs_long(raw)
        oer   = NaiveDiscoveryDetector.score_oer(raw)
        urs   = NaiveDiscoveryDetector.score_urs(ranks.get(t))
        cls = NaiveDiscoveryDetector.classify(raw, tcs_s, tcs_l, tfs_s, tfs_l, oer,
                                              adaptive=None, urs=urs)
        classifications[t] = cls
        tcs_values[t] = tcs
        rss_values[t] = ranks.get(t, {}).get("rss", np.nan)

    def _aggregate(group_tickers: List[str], prefix: str) -> Dict[str, float]:
        group_tickers = [t for t in group_tickers if t in classifications]
        n = len(group_tickers)
        if n < 10:
            return {}
        cls_list = [classifications[t] for t in group_tickers]
        def _pct(substrings):
            return sum(1 for c in cls_list
                       if any(s in c for s in substrings)) / n
        tcs_arr = np.array([tcs_values[t] for t in group_tickers], dtype=float)
        rss_arr = np.array([rss_values[t] for t in group_tickers], dtype=float)
        rss_arr = rss_arr[~np.isnan(rss_arr)]
        return {
            f"{prefix}_pct_bullish":   _pct(BULLISH_SUBSTRINGS),
            f"{prefix}_pct_downtrend": _pct(DOWNTREND_SUBSTRINGS),
            f"{prefix}_tcs_median":    float(np.nanmedian(tcs_arr)),
            f"{prefix}_rss_std":       float(np.nanstd(rss_arr, ddof=1)) if len(rss_arr) > 1 else np.nan,
        }

    eq_tickers = [t for t, c in ticker_category.items() if c in EQUITY_CATS]
    bd_tickers = [t for t, c in ticker_category.items() if c in BOND_CATS]

    eq_feats = _aggregate(eq_tickers, "eq")
    bd_feats = _aggregate(bd_tickers, "bd")

    out = {}
    # Equity keeps all 4; bond keeps pct_bullish + tcs_median per plan
    for k in ["eq_pct_bullish", "eq_pct_downtrend", "eq_tcs_median", "eq_rss_std"]:
        out[k] = eq_feats.get(k, np.nan)
    for k in ["bd_pct_bullish", "bd_tcs_median"]:
        out[k] = bd_feats.get(k, np.nan)
    out["_n_eq_tickers"] = len(eq_tickers)
    out["_n_bd_tickers"] = len(bd_tickers)
    return out


# ---------------------------------------------------------------------------
# Historical replay
# ---------------------------------------------------------------------------
def build_monthly_breadth(start: str = "2007-07-31",
                          end: str = None,
                          out_path: str = "breadth_monthly.parquet") -> pd.DataFrame:
    """Loop over month-ends in [start, end], compute breadth, persist to parquet."""
    data = download_universe_cached(start_date="2007-01-01")
    if end is None:
        end = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    month_ends = pd.date_range(start=start, end=end, freq="ME")

    print(f"[breadth] Replaying {len(month_ends)} month-ends "
          f"({month_ends[0].date()} → {month_ends[-1].date()}) "
          f"over {len(data)} ETFs…")
    rows = []
    t0 = time.time()
    for i, as_of in enumerate(month_ends, 1):
        feats = compute_breadth_at(as_of, data)
        if feats:
            feats["Date"] = as_of
            rows.append(feats)
        if i % 25 == 0 or i == len(month_ends):
            print(f"  {i:>3}/{len(month_ends)}  ({as_of.date()})  "
                  f"elapsed={time.time()-t0:.1f}s")

    df = pd.DataFrame(rows).set_index("Date").sort_index()
    if out_path.endswith(".parquet"):
        try:
            df.to_parquet(out_path)
        except Exception as e:
            print(f"[breadth] parquet write failed ({e}); saving CSV instead")
            out_path = out_path.replace(".parquet", ".csv")
            df.to_csv(out_path)
    else:
        df.to_csv(out_path)
    print(f"[breadth] Saved → {out_path}  ({len(df)} rows, {df.shape[1]} cols)")
    return df


if __name__ == "__main__":
    df = build_monthly_breadth(start="2007-07-31")
    print("\nHead:")
    print(df.head(3))
    print("\nTail:")
    print(df.tail(3))
    print("\nSummary:")
    print(df.describe().T[["count", "mean", "std", "min", "max"]].round(3))
