"""
FastAPI backend for Price Discovery React Dashboard.
Loads .scan_cache.pkl and serves JSON endpoints.

Run: python3 -m uvicorn api:app --reload --port 8000
"""

import os, sys, pickle, math, json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

# ── Import from scanner ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from price_discovery import (
    STOCK_THEMES, STOCK_THEMES_CONSOLIDATED, ETF_SUBTHEMES,
    STOCK_UNIVERSE, GLOBAL_ETF_UNIVERSE,
    CATEGORY_BENCHMARK, STOCK_BENCHMARK, CATEGORY_BENCHMARK_ALT,
)

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan_cache.pkl")

# ── SubTheme → Sector (Option B unified taxonomy) ──
# 17 sectors × ~105 subthemes; ETFs and stocks share the same SubTheme namespace.
SUBTHEME_TO_SECTOR = {
    # ── Technology ──
    "Semiconductor Design": "Technology", "Semiconductor Foundry": "Technology",
    "Semiconductor Memory": "Technology", "Semiconductor Analog": "Technology",
    "Semiconductor Equipment & EDA": "Technology", "Enterprise Software": "Technology",
    "Cybersecurity": "Technology", "Cloud & Platform": "Technology",
    "Data Center & Networking": "Technology", "Industrial Technology": "Technology",
    "Consumer Hardware": "Technology", "Robotics & AI": "Technology",
    "Sector Broad - Tech": "Technology",
    # ── Communication Services ──
    "Digital Advertising & Media": "Communication Services",
    "Digital Media & Entertainment": "Communication Services",
    "Telecom & IT Services": "Communication Services",
    "Sector Broad - CommServices": "Communication Services",
    # ── Healthcare ──
    "Big Pharma": "Healthcare", "Biotech": "Healthcare",
    "Healthcare Services": "Healthcare",
    "Medical Devices & Diagnostics": "Healthcare", "Life Science Tools": "Healthcare",
    "Sector Broad - Healthcare": "Healthcare",
    # ── Financials ──
    "Banks": "Financials", "Insurance": "Financials",
    "Investment Banking & Asset Mgmt": "Financials",
    "Payments & Exchanges": "Financials", "Financial Data & Analytics": "Financials",
    "Fintech & Digital Finance": "Financials", "Consumer Finance": "Financials",
    "Conglomerate & Holding": "Financials",
    "Sector Broad - Financials": "Financials",
    # ── Consumer Discretionary ──
    "Auto & EV": "Consumer Discretionary", "Consumer Brands": "Consumer Discretionary",
    "E-commerce & Delivery": "Consumer Discretionary",
    "Restaurants & Leisure": "Consumer Discretionary", "Retail": "Consumer Discretionary",
    "Sector Broad - ConsDisc": "Consumer Discretionary",
    # ── Consumer Staples ──
    "Consumer Staples": "Consumer Staples",
    "Sector Broad - ConsStaples": "Consumer Staples",
    # ── Industrials ──
    "Aerospace & Defense": "Industrials", "Industrial Equipment": "Industrials",
    "Building & Construction": "Industrials", "Transport & Logistics": "Industrials",
    "Environmental & Water": "Industrials",
    "Sector Broad - Industrials": "Industrials",
    # ── Energy ──
    "Oil & Gas": "Energy", "Uranium & Nuclear Fuel": "Energy",
    "Energy Commodities": "Energy", "Sector Broad - Energy": "Energy",
    # ── Utilities ──
    "Utilities": "Utilities", "Power & Energy Infra": "Utilities",
    "Sector Broad - Utilities": "Utilities",
    # ── Materials ──
    "Chemicals": "Materials", "Steel & Metals": "Materials",
    "Base Metals & Mining": "Materials", "Precious Metals": "Materials",
    "Agriculture & Food": "Materials", "Battery & EV Materials": "Materials",
    "Natural Resources": "Materials", "Broad Commodity": "Materials",
    "Sector Broad - Materials": "Materials",
    # ── Real Estate ──
    "Real Estate": "Real Estate", "Sector Broad - RealEstate": "Real Estate",
    # ── Equity Broad (broad / factor / disruptive) ──
    "Broad Market": "Equity Broad", "Factor - Momentum": "Equity Broad",
    "Factor - Quality": "Equity Broad", "Factor - Min Vol": "Equity Broad",
    "Factor - Value": "Equity Broad", "Factor - Size": "Equity Broad",
    "Factor - Dividend": "Equity Broad", "Factor - Multi": "Equity Broad",
    "Disruptive Innovation": "Equity Broad",
    # ── International (regional) ──
    "Developed Markets": "International", "Europe": "International",
    "Japan": "International", "Asia Pacific": "International",
    "North America": "International", "Middle East": "International",
    "EM Broad": "International", "China": "International",
    "Korea (Index)": "International", "India": "International",
    "Latin America": "International", "Other EM": "International",
    # ── Fixed Income ──
    "Treasury - Short": "Fixed Income", "Treasury - Intermediate": "Fixed Income",
    "Treasury - Long": "Fixed Income",
    "IG Corporate - Short": "Fixed Income", "IG Corporate - Intermediate": "Fixed Income",
    "IG Corporate - Long": "Fixed Income", "Aggregate Bond": "Fixed Income",
    "MBS": "Fixed Income", "CLO": "Fixed Income", "Floating Rate": "Fixed Income",
    "High Yield": "Fixed Income", "Senior Loans": "Fixed Income",
    "Preferred": "Fixed Income", "Inflation-Linked": "Fixed Income",
    "International Bonds": "Fixed Income", "EM Bonds": "Fixed Income",
    # ── Macro / Multi-Asset / Alt ──
    "Currency": "Macro", "Volatility": "Macro",
    "Asset Allocation": "Multi-Asset",
    "Crypto": "Alternatives",
}

# Backwards-compat alias (older code may still reference THEME_TO_SECTOR)
THEME_TO_SECTOR = SUBTHEME_TO_SECTOR

app = FastAPI(title="Price Discovery API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Global state ──
STATE = {}


def _safe(v):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if math.isfinite(f) else 0.0
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return 0.0
    return v


def _clean_dict(d):
    """Recursively convert a dict to JSON-safe types."""
    if isinstance(d, dict):
        return {str(k): _clean_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_clean_dict(i) for i in d]
    if isinstance(d, tuple):
        return [_clean_dict(i) for i in d]
    return _safe(d)


def _load_cache():
    """Load pickle cache and build DataFrames."""
    if not os.path.exists(CACHE_PATH):
        return False

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)

    results = cache["results"]
    df = pd.DataFrame(results)

    # Backfill new fields
    for col in ("tcs_short", "tcs_long", "tfs_short", "tfs_long", "rss_short", "rss_long"):
        if col not in df.columns:
            df[col] = 0.0

    # Backfill hedge strategy fields
    hedge_cols = [
        "minervini_long", "minervini_short", "wyckoff_long", "wyckoff_short",
        "ichimoku_long", "ichimoku_short", "darvas_long", "darvas_short",
        "regime_long", "regime_short", "flow_long", "flow_short",
        "relval_long", "relval_short",
        "combined_long", "combined_short", "long_count", "short_count",
        "net_signal", "conviction",
    ]
    for col in hedge_cols:
        if col not in df.columns:
            df[col] = "" if col == "net_signal" else 0

    # ── Unified SubTheme (Option B) — works for both ETFs and stocks ──
    # Stock: STOCK_THEMES_CONSOLIDATED (~47 macro themes)
    # ETF:   ETF_SUBTHEMES (~55 ETF-specific or shared subthemes)
    def _subtheme(tk: str) -> str:
        if tk in STOCK_THEMES_CONSOLIDATED:
            return STOCK_THEMES_CONSOLIDATED[tk]
        if tk in ETF_SUBTHEMES:
            return ETF_SUBTHEMES[tk]
        return "-"
    df["theme"] = df["ticker"].apply(_subtheme)
    # Granular detail for stocks; ETFs reuse the subtheme as detail.
    df["theme_detail"] = df["ticker"].apply(
        lambda t: STOCK_THEMES.get(t, ETF_SUBTHEMES.get(t, "-"))
    )

    # Backfill universe return / vol fields
    for col in ("ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d"):
        if col not in df.columns:
            df[col] = 0.0
    for col in ("ret_3y_ann", "ret_5y_ann", "vol_3y_ann"):
        if col not in df.columns:
            df[col] = None

    # Backfill alpha_potential
    if "alpha_potential" not in df.columns:
        df["alpha_potential"] = 0

    # Backfill market cap
    if "market_cap" not in df.columns:
        df["market_cap"] = 0.0
    df["mktcap_B"] = df["market_cap"] / 1e9  # in billions

    # ADV in millions
    df["adv_M"] = df["adv_usd"] / 1e6

    # Determine ETF vs Stock + ETF full names
    stock_tickers = set()
    for cat, data in STOCK_UNIVERSE.items():
        stock_tickers.update(data["tickers"].keys())
    df["asset_type"] = df["ticker"].apply(lambda t: "Stock" if t in stock_tickers else "ETF")

    # Build ETF full name map: ticker → "Full Description (TICKER)"
    etf_name_map = {}
    for cat, data in GLOBAL_ETF_UNIVERSE.items():
        for tk, desc in data["tickers"].items():
            etf_name_map[tk] = desc
    # Override short names for ETFs with their universe descriptions
    def _full_name(row):
        if row["ticker"] in etf_name_map:
            return etf_name_map[row["ticker"]]
        return row["name"]
    df["name"] = df.apply(_full_name, axis=1)

    # ── Unified Sector — pure SubTheme→Sector lookup (Option B) ──
    df["sector"] = df["theme"].map(SUBTHEME_TO_SECTOR).fillna("Other")

    # ── QVR (Quality-Value-Revision) overlay + eligibility gate (Option A) ──
    # Loads .fundamentals_cache.pkl and scores every stock. ETFs get neutral 50.
    # Then re-applies eligibility: stocks with QVR < 40 are demoted from
    # eligible=True → eligible=False with rejection reason "WeakQVR(<n>)".
    try:
        from fundamentals_pipeline import load_fundamentals_cache
        from qvr_agent import QVRAgent
        fund_cache = load_fundamentals_cache()
        if fund_cache and fund_cache.get("tickers"):
            qvr_agent = QVRAgent(indices={}, fundamentals_cache=fund_cache)

            qvr_scores, qvr_q, qvr_v, qvr_r = [], [], [], []
            qvr_n_analysts, qvr_bull_chg_3m, qvr_eps_beat = [], [], []
            qvr_eps_surprise = []
            for _, row in df.iterrows():
                score, signals, _summary = qvr_agent.score({"ticker": row["ticker"]})
                qvr_scores.append(round(float(score), 1))
                qvr_q.append(round(float(signals.get("quality", 50.0)), 1))
                qvr_v.append(round(float(signals.get("value", 50.0)), 1))
                qvr_r.append(round(float(signals.get("revision", 50.0)), 1))
                qvr_n_analysts.append(int(signals.get("n_analysts") or 0))
                # Finnhub-derived (None when Finnhub data missing)
                qvr_bull_chg_3m.append(signals.get("bullish_change_3m"))
                qvr_eps_beat.append(signals.get("eps_beat_rate"))
                qvr_eps_surprise.append(signals.get("eps_surprise_avg"))
            df["qvr_score"] = qvr_scores
            df["qvr_q"] = qvr_q
            df["qvr_v"] = qvr_v
            df["qvr_r"] = qvr_r
            df["qvr_n_analysts"] = qvr_n_analysts
            df["qvr_bullish_chg_3m"] = qvr_bull_chg_3m
            df["qvr_eps_beat_rate"] = qvr_eps_beat
            df["qvr_eps_surprise_avg"] = qvr_eps_surprise

            # ── Eligibility Gate ──
            # Demote stocks with weak fundamentals (QVR < 40) even if technically
            # eligible. ETFs are exempt (no fundamentals available).
            QVR_GATE = 40.0
            gate_mask = (
                (df["asset_type"] == "Stock")
                & (df["eligible"] == True)
                & (df["qvr_score"] < QVR_GATE)
            )
            n_demoted = int(gate_mask.sum())
            if n_demoted > 0:
                df.loc[gate_mask, "eligible"] = False
                # Append rejection reason
                def _add_qvr_reject(row):
                    base = row.get("rejection") or ""
                    tag = f"WeakQVR({row['qvr_score']:.0f})"
                    if base in ("", "None"):
                        return tag
                    return f"{base}/{tag}"
                df.loc[gate_mask, "rejection"] = df.loc[gate_mask].apply(
                    _add_qvr_reject, axis=1
                )
            print(f"[qvr-gate] demoted {n_demoted} stocks (QVR < {QVR_GATE:.0f})")
        else:
            df["qvr_score"] = 50.0
            df["qvr_q"] = 50.0
            df["qvr_v"] = 50.0
            df["qvr_r"] = 50.0
            df["qvr_n_analysts"] = 0
            df["qvr_bullish_chg_3m"] = None
            df["qvr_eps_beat_rate"] = None
            df["qvr_eps_surprise_avg"] = None
            print("[qvr-gate] no fundamentals cache — skipping (run fundamentals_pipeline.py)")
    except Exception as _e:
        df["qvr_score"] = 50.0
        df["qvr_q"] = 50.0
        df["qvr_v"] = 50.0
        df["qvr_r"] = 50.0
        df["qvr_n_analysts"] = 0
        df["qvr_bullish_chg_3m"] = None
        df["qvr_eps_beat_rate"] = None
        df["qvr_eps_surprise_avg"] = None
        print(f"[qvr-gate] skipped due to error: {_e}")

    # VE stats
    ve_stats = {
        "bucket": cache.get("ve_bucket", {}),
        "class": cache.get("ve_class", {}),
        "transitions": cache.get("ve_transitions", {}),
        "transition_totals": cache.get("ve_transition_totals", {}),
        "observations": cache.get("ve_observations", []),
        "fwd_bucket": cache.get("ve_fwd_bucket", {}),
        "fwd_class": cache.get("ve_fwd_class", {}),
        "fwd_eligible": cache.get("ve_fwd_eligible", {}),
        "transition_hit": cache.get("ve_transition_hit", {}),
        "score_weighted": cache.get("ve_score_weighted", {}),
    }

    STATE["df"] = df
    STATE["results"] = results
    STATE["ve_stats"] = ve_stats
    STATE["graph"] = cache.get("graph", {})
    STATE["history"] = cache.get("history", {})
    STATE["top_long_bt"] = cache.get("top_long_bt", [])
    STATE["factor_efficacy"] = cache.get("factor_efficacy", {})
    STATE["scan_time"] = cache.get("scan_time", "unknown")

    # ── Unified classification (Phase Y — GICS sector / industry / cap tier) ──
    uc_path = ".unified_classification.json"
    if os.path.exists(uc_path):
        try:
            with open(uc_path) as f:
                uc = json.load(f)
            tickers_data = uc.get("tickers", {})
            STATE["unified_classification"] = uc
            # Build per-ticker DataFrame for merge
            uc_rows = []
            for tk, c in tickers_data.items():
                if not c.get("ok"):
                    continue
                uc_rows.append({
                    "ticker": tk,
                    "gics_sector": c.get("gics_sector"),
                    "gics_industry": c.get("gics_industry"),
                    "country": c.get("country"),
                    "cap_tier": c.get("cap_tier"),
                    "mktcap_usd_b": c.get("mktcap_usd_b"),
                })
            if uc_rows:
                uc_df = pd.DataFrame(uc_rows)
                STATE["df"] = STATE["df"].merge(uc_df, on="ticker", how="left")
            print(f"[unified] loaded classification · {uc.get('n_success', 0)}/{uc.get('n_total', 0)} tickers")
        except Exception as _uc_err:
            print(f"[unified] load failed: {_uc_err}")
            STATE["unified_classification"] = {}
    else:
        STATE["unified_classification"] = {}

    # ── ML cache (parallel results re-scored with optimized Composite weights) ──
    ml_path = ".scan_cache_ml.pkl"
    if os.path.exists(ml_path):
        try:
            with open(ml_path, "rb") as f:
                ml_cache = pickle.load(f)
            results_ml = ml_cache.get("results_ml", [])
            ml_meta = ml_cache.get("ml_meta", {})
            STATE["results_ml"] = results_ml
            STATE["ml_meta"] = ml_meta
            # Merge ML columns into df by ticker
            if results_ml:
                ml_df = pd.DataFrame(results_ml)
                ml_cols = ["ticker", "composite_ml", "classification_ml",
                            "eligible_ml", "rejection_ml", "stage_ml", "asset_class_ml"]
                ml_cols = [c for c in ml_cols if c in ml_df.columns]
                STATE["df"] = STATE["df"].merge(ml_df[ml_cols], on="ticker", how="left")
            print(f"[ml] loaded ML cache · {len(results_ml)} tickers · "
                  f"stage_counts={ml_meta.get('stage_counts')}")
        except Exception as _ml_err:
            print(f"[ml] load failed: {_ml_err}")
            STATE["results_ml"] = []
            STATE["ml_meta"] = {}
    else:
        STATE["results_ml"] = []
        STATE["ml_meta"] = {}
    return True


@app.on_event("startup")
def startup():
    if not _load_cache():
        print("⚠️ No cache found. Run price_discovery.py first.")


def _filter_df(
    categories: Optional[List[str]] = None,
    classifications: Optional[List[str]] = None,
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = None,
    subthemes: Optional[List[str]] = None,
):
    df = STATE.get("df")
    if df is None or df.empty:
        return pd.DataFrame()
    mask = df["composite"].between(comp_min, comp_max)
    if categories:
        mask &= df["category"].isin(categories)
    if sectors:
        mask &= df["sector"].isin(sectors)
    if subthemes:
        mask &= df["theme"].isin(subthemes)
    if classifications:
        mask &= df["classification"].isin(classifications)
    if eligible_only:
        mask &= df["eligible"]
    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/meta")
def meta():
    df = STATE.get("df")
    if df is None:
        return {"error": "No data loaded"}

    # Category → benchmark mapping with ticker counts
    all_bench = {**CATEGORY_BENCHMARK, **STOCK_BENCHMARK}
    cat_info = []
    for cat in sorted(df["category"].unique()):
        n = int((df["category"] == cat).sum())
        asset = "Stock" if cat.startswith("STK_") else "ETF"
        bench = all_bench.get(cat, "SPY")
        alt = CATEGORY_BENCHMARK_ALT.get(cat, [bench])
        cat_info.append({"category": cat, "n": n, "asset_type": asset,
                         "benchmark": bench, "alt_benchmarks": alt})

    # Theme summary
    themed = df[df["theme"] != "-"]
    theme_info = []
    if not themed.empty:
        for theme, grp in themed.groupby("theme"):
            theme_info.append({
                "theme": theme, "n": len(grp),
                "categories": sorted(set(c.replace("STK_", "") for c in grp["category"])),
            })
        theme_info.sort(key=lambda x: -x["n"])

    # ── Sector / SubTheme info (Option B unified taxonomy) ──
    sector_info = []
    for sec, grp in df.groupby("sector"):
        sector_info.append({
            "sector": sec,
            "n": len(grp),
            "n_etf": int((grp["asset_type"] == "ETF").sum()),
            "n_stock": int((grp["asset_type"] == "Stock").sum()),
            "subthemes": sorted(grp[grp["theme"] != "-"]["theme"].unique().tolist()),
        })
    sector_info.sort(key=lambda x: -x["n"])

    return {
        "scan_time": STATE.get("scan_time", ""),
        "total_tickers": len(df),
        "categories": sorted(df["category"].unique().tolist()),
        "sectors": sorted(df["sector"].unique().tolist()),
        "subthemes": sorted(df[df["theme"] != "-"]["theme"].unique().tolist()),
        "classifications": sorted(df["classification"].unique().tolist()),
        "themes": sorted(df[df["theme"] != "-"]["theme"].unique().tolist()),  # backwards-compat alias
        "category_info": cat_info,
        "sector_info": sector_info,
        "theme_info": theme_info,
    }


@app.post("/api/reload")
def reload():
    ok = _load_cache()
    return {"success": ok}


# ── Live Scan: run price_discovery.py in background, then reload cache ──
_SCAN_STATUS = {
    "running": False,
    "last_error": "",
    "started_at": "",
    "last_line": "",
    "phase": "",
}
_SCAN_TIMEOUT_SEC = 3600  # 60 min — 770 tickers × 5y can take 30-40 min
_SCAN_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".scan.log")


@app.post("/api/scan")
def run_scan_api(
    lookback_years: int = 5,
    use_realtime: bool = True,
    include_stocks: bool = True,
):
    """Trigger a full live scan. Streams stdout to .scan.log for progress tracking."""
    import threading, subprocess
    from datetime import datetime as _dt

    if _SCAN_STATUS["running"]:
        return {"status": "already_running"}

    def _run():
        _SCAN_STATUS["running"] = True
        _SCAN_STATUS["last_error"] = ""
        _SCAN_STATUS["started_at"] = _dt.now().isoformat()
        _SCAN_STATUS["last_line"] = "Starting..."
        _SCAN_STATUS["phase"] = "Init"
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cmd = [
                sys.executable, "-u", "-c",  # -u for unbuffered output
                f"import sys; sys.path.insert(0, {repr(script_dir)}); "
                f"from price_discovery import run_scan; "
                f"run_scan(lookback_days={365 * lookback_years}, "
                f"use_realtime={use_realtime}, include_stocks={include_stocks})"
            ]
            with open(_SCAN_LOG_PATH, "w") as logf:
                proc = subprocess.Popen(
                    cmd, cwd=script_dir, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                start = _dt.now()
                last_lines: list = []
                for line in proc.stdout:
                    line = line.rstrip()
                    logf.write(line + "\n")
                    logf.flush()
                    if line:
                        _SCAN_STATUS["last_line"] = line[:200]
                        last_lines.append(line)
                        if len(last_lines) > 50:
                            last_lines.pop(0)
                        # Heuristic phase tracking
                        for kw, ph in [
                            ("Phase 1", "Indicators"), ("Phase 2", "Ranking"),
                            ("Phase 3", "Validity"), ("Phase 4", "Scoring"),
                            ("Cache saved", "Done"), ("Downloading", "Downloading"),
                            ("MASTER SUMMARY", "Output"),
                        ]:
                            if kw in line:
                                _SCAN_STATUS["phase"] = ph
                    # Manual timeout check
                    if (_dt.now() - start).total_seconds() > _SCAN_TIMEOUT_SEC:
                        proc.kill()
                        _SCAN_STATUS["last_error"] = f"Scan timed out ({_SCAN_TIMEOUT_SEC // 60} min)"
                        return
                rc = proc.wait()
                if rc != 0:
                    tail = "\n".join(last_lines[-10:])
                    _SCAN_STATUS["last_error"] = f"Exit {rc}. Tail:\n{tail}"
                else:
                    _load_cache()
                    _SCAN_STATUS["phase"] = "Done"
        except Exception as e:
            _SCAN_STATUS["last_error"] = f"{type(e).__name__}: {e}"
        finally:
            _SCAN_STATUS["running"] = False

    threading.Thread(target=_run, daemon=True).start()
    return {"status": "started"}


@app.get("/api/scan/status")
def scan_status():
    return {
        "running": _SCAN_STATUS["running"],
        "last_error": _SCAN_STATUS["last_error"],
        "has_cache": os.path.exists(CACHE_PATH),
        "started_at": _SCAN_STATUS.get("started_at", ""),
        "last_line": _SCAN_STATUS.get("last_line", ""),
        "phase": _SCAN_STATUS.get("phase", ""),
    }


@app.get("/api/scan/log")
def scan_log():
    """Return the last 100 lines of the scan log."""
    if not os.path.exists(_SCAN_LOG_PATH):
        return {"lines": []}
    try:
        with open(_SCAN_LOG_PATH, "r") as f:
            lines = f.readlines()
        return {"lines": [l.rstrip() for l in lines[-100:]]}
    except IOError:
        return {"lines": []}


def _filter_top_long_bt(bt_data: list, fdf: pd.DataFrame) -> list:
    """Filter top_long_bt snapshots to only include tickers present in filtered df."""
    if not bt_data or fdf.empty:
        return bt_data
    valid_tickers = set(fdf["ticker"])
    filtered = []
    for snap in bt_data:
        tickers = snap.get("tickers", [])
        if not tickers:
            continue
        # Filter ticker details to only those in the current category set
        ft = [t for t in tickers if t.get("ticker") in valid_tickers]
        if not ft:
            continue
        # Recompute summary from filtered tickers
        summary = {}
        for period in ["1W", "1M", "3M", "CUM"]:
            key = f"ret_{period}"
            rets = [t[key] for t in ft if t.get(key) is not None]
            if rets:
                summary[period] = {
                    "avg_ret": round(sum(rets) / len(rets), 2),
                    "hit_rate": round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1),
                }
            else:
                summary[period] = {"avg_ret": 0, "hit_rate": 0}
        filtered.append({
            "eval_date": snap["eval_date"],
            "n_picks": len(ft),
            "tickers": ft,
            "summary": summary,
        })
    return filtered


@app.get("/api/overview")
def overview(
    categories: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = Query(None),
    subthemes: Optional[List[str]] = Query(None),
):
    fdf = _filter_df(categories, classifications, eligible_only, comp_min, comp_max,
                     sectors=sectors, subthemes=subthemes)
    if fdf.empty:
        return {"kpis": {}, "classification_dist": [], "composite_data": [],
                "conviction_bubble": [], "top_eligible": []}

    n_el = int(fdf["eligible"].sum())
    kpis = {
        "total": len(fdf),
        "eligible": n_el,
        "avg_composite": round(float(fdf["composite"].mean()), 1),
        "pct_continuation": round(
            len(fdf[fdf["classification"].str.contains("CONTINUATION")]) / len(fdf) * 100, 1
        ) if len(fdf) > 0 else 0,
    }

    cls_dist = fdf["classification"].value_counts().reset_index()
    cls_dist.columns = ["classification", "count"]

    # Composite data for histogram
    composite_data = fdf[["composite", "classification"]].to_dict(orient="records")

    # Conviction bubble (top by composite)
    bubble_cols = ["ticker", "name", "composite", "val_prob", "adv_M",
                   "classification", "category", "tcs", "tfs", "oer"]
    bubble = fdf.nlargest(50, "composite")[bubble_cols].to_dict(orient="records")

    # Top 15 eligible
    top_el = fdf[fdf["eligible"]].nlargest(15, "composite")
    top_cols = ["ticker", "name", "composite", "classification", "tcs", "tfs", "oer", "rss",
                "tcs_short", "tcs_long", "tfs_short", "tfs_long"]
    top_eligible = top_el[[c for c in top_cols if c in top_el.columns]].to_dict(orient="records")

    # Top 10 Strong Long: eligible + bullish classification + O'Neil Long signal
    # Risk management: sector cap (max 3 per sector) + event-driven discount (×0.7)
    SECTOR_CAP = 3
    top_long_cols = ["ticker", "name", "sector", "category", "mktcap_B", "composite",
                     "classification", "oneil_long", "oneil_short",
                     "minervini_long", "minervini_short",
                     "wyckoff_long", "wyckoff_short",
                     "ichimoku_long", "ichimoku_short",
                     "darvas_long", "darvas_short",
                     "regime_long", "regime_short",
                     "flow_long", "flow_short",
                     "relval_long", "relval_short",
                     "combined_long", "combined_short",
                     "long_count", "short_count", "net_signal", "conviction",
                     "event_flag", "event_reasons", "structural_q", "alpha_potential",
                     "tcs", "tfs", "oer", "rss", "rsi", "trend_age",
                     "val_prob", "ret_1w", "ret_1m", "ret_3m"]
    top_long_cols = [c for c in top_long_cols if c in fdf.columns]
    bullish_cls = {"🟢 CONTINUATION", "🔵 FORMATION", "🟡 CONSOLIDATION",
                   "🔶 PULLBACK", "🔵 RECOVERY"}
    oneil_col = "oneil_long" if "oneil_long" in fdf.columns else None
    has_aps = "alpha_potential" in fdf.columns
    if oneil_col:
        # Primary pool: eligible + bullish classification
        long_cands = fdf[fdf["eligible"] & fdf["classification"].isin(bullish_cls)].copy()
        # Hidden Gems pool: non-bullish but APS >= 70 + eligible
        if has_aps:
            hidden_gems = fdf[
                fdf["eligible"]
                & ~fdf["classification"].isin(bullish_cls)
                & (fdf["alpha_potential"] >= 70)
            ].copy()
            if not hidden_gems.empty:
                long_cands = pd.concat([long_cands, hidden_gems]).drop_duplicates(subset="ticker")
        # Event-driven discount: EVENT flagged tickers get 0.7x weight
        has_event = "event_flag" in long_cands.columns
        if has_event:
            discount = long_cands["event_flag"].apply(lambda x: 0.7 if x else 1.0)
        else:
            discount = 1.0
        comb_col = "combined_long" if "combined_long" in long_cands.columns else "oneil_long"
        # Ranking: combined_long(40%) + composite(35%) + APS(25%)
        aps_val = long_cands["alpha_potential"].fillna(0) if has_aps else 0
        long_cands["_long_rank"] = (long_cands[comb_col] * 0.40 + long_cands["composite"] * 0.35 + aps_val * 0.25) * discount
        long_cands = long_cands.sort_values("_long_rank", ascending=False)
        # Sector cap: max SECTOR_CAP tickers per sector
        selected = []
        sector_counts: dict = {}
        for _, row in long_cands.iterrows():
            sec = row.get("sector", "Other")
            if sector_counts.get(sec, 0) >= SECTOR_CAP:
                continue
            selected.append(row)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
            if len(selected) >= 10:
                break
        if selected:
            sel_df = pd.DataFrame(selected)
            top_long = sel_df[top_long_cols].round(2).to_dict(orient="records")
        else:
            top_long = []
        # Sector concentration warning
        if selected:
            sec_dist = {}
            for r in selected:
                s = r.get("sector", "Other")
                sec_dist[s] = sec_dist.get(s, 0) + 1
            top_long_warnings = []
            for s, cnt in sec_dist.items():
                if cnt >= SECTOR_CAP:
                    top_long_warnings.append(f"{s} ({cnt}/{SECTOR_CAP} cap reached)")
            n_event = sum(1 for r in selected if r.get("event_flag", False))
            if n_event > 0:
                top_long_warnings.append(f"{n_event} event-driven pick(s) — discount applied")
        else:
            top_long_warnings = []
    else:
        top_long = []
        top_long_warnings = []

    # Bottom 10 Strong Short: bearish classification + high short signal
    bearish_cls = {"⬇️ DOWNTREND", "⚠️ WEAKENING", "🟤 FADING",
                   "🟤 EXHAUSTING", "🟣 COUNTER_RALLY", "🔴 CYCLE_PEAK"}
    top_short_cols = list(top_long_cols)  # same columns
    short_col = "oneil_short" if "oneil_short" in fdf.columns else None
    if short_col:
        short_cands = fdf[fdf["classification"].isin(bearish_cls)].copy()
        if short_cands.empty:
            # fallback: lowest composite regardless of classification
            short_cands = fdf.nsmallest(30, "composite").copy()
        comb_s_col = "combined_short" if "combined_short" in short_cands.columns else "oneil_short"
        # Ranking: combined_short(40%) + inverse composite(35%) + inverse APS(25%) — low APS = weak structure
        inv_aps = (100 - short_cands["alpha_potential"].fillna(50)) if has_aps else 50
        short_cands["_short_rank"] = short_cands[comb_s_col] * 0.40 + (100 - short_cands["composite"]) * 0.35 + inv_aps * 0.25
        short_cands = short_cands.sort_values("_short_rank", ascending=False)
        selected_short = []
        sector_counts_s: dict = {}
        for _, row in short_cands.iterrows():
            sec = row.get("sector", "Other")
            if sector_counts_s.get(sec, 0) >= SECTOR_CAP:
                continue
            selected_short.append(row)
            sector_counts_s[sec] = sector_counts_s.get(sec, 0) + 1
            if len(selected_short) >= 10:
                break
        if selected_short:
            sel_s_df = pd.DataFrame(selected_short)
            top_short = sel_s_df[[c for c in top_short_cols if c in sel_s_df.columns]].round(2).to_dict(orient="records")
        else:
            top_short = []
    else:
        top_short = []

    # ── Top 10 Long / Short split by asset type (ETF vs Stock) ──
    def _select_top(pool: pd.DataFrame, direction: str, n: int = 10) -> list:
        """Reusable selection for long/short top-N with sector cap."""
        if pool.empty:
            return []
        pool = pool.copy()
        _has_aps = "alpha_potential" in pool.columns
        if direction == "long":
            _comb = "combined_long" if "combined_long" in pool.columns else "oneil_long"
            _aps = pool["alpha_potential"].fillna(0) if _has_aps else 0
            _has_ev = "event_flag" in pool.columns
            _disc = pool["event_flag"].apply(lambda x: 0.7 if x else 1.0) if _has_ev else 1.0
            pool["_rank"] = (pool[_comb] * 0.40 + pool["composite"] * 0.35 + _aps * 0.25) * _disc
        else:
            _comb = "combined_short" if "combined_short" in pool.columns else "oneil_short"
            _inv_aps = (100 - pool["alpha_potential"].fillna(50)) if _has_aps else 50
            pool["_rank"] = pool[_comb] * 0.40 + (100 - pool["composite"]) * 0.35 + _inv_aps * 0.25
        pool = pool.sort_values("_rank", ascending=False)
        sel = []
        sc: dict = {}
        for _, row in pool.iterrows():
            sec = row.get("sector", "Other")
            if sc.get(sec, 0) >= SECTOR_CAP:
                continue
            sel.append(row)
            sc[sec] = sc.get(sec, 0) + 1
            if len(sel) >= n:
                break
        if not sel:
            return []
        cols = [c for c in top_long_cols if c in pool.columns]
        return pd.DataFrame(sel)[cols].round(2).to_dict(orient="records")

    # ETF/Stock split — use FULL universe (unfiltered) so asset-type tables always show 10
    _full_df = STATE.get("df", pd.DataFrame())
    _full_has_aps = "alpha_potential" in _full_df.columns if not _full_df.empty else False
    _full_oneil = "oneil_long" in _full_df.columns if not _full_df.empty else False
    _full_short = "oneil_short" in _full_df.columns if not _full_df.empty else False
    if not _full_df.empty and _full_oneil:
        etf_df = _full_df[_full_df["asset_type"] == "ETF"]
        stock_df = _full_df[_full_df["asset_type"] == "Stock"]
        etf_long_pool = etf_df[etf_df["eligible"] & etf_df["classification"].isin(bullish_cls)]
        stock_long_pool = stock_df[stock_df["eligible"] & stock_df["classification"].isin(bullish_cls)]
        if _full_has_aps:
            etf_hg = etf_df[etf_df["eligible"] & ~etf_df["classification"].isin(bullish_cls) & (etf_df["alpha_potential"] >= 70)]
            stock_hg = stock_df[stock_df["eligible"] & ~stock_df["classification"].isin(bullish_cls) & (stock_df["alpha_potential"] >= 70)]
            if not etf_hg.empty:
                etf_long_pool = pd.concat([etf_long_pool, etf_hg]).drop_duplicates(subset="ticker")
            if not stock_hg.empty:
                stock_long_pool = pd.concat([stock_long_pool, stock_hg]).drop_duplicates(subset="ticker")
        top_long_etf = _select_top(etf_long_pool, "long")
        top_long_stock = _select_top(stock_long_pool, "long")

        etf_short_pool = etf_df[etf_df["classification"].isin(bearish_cls)]
        stock_short_pool = stock_df[stock_df["classification"].isin(bearish_cls)]
        if etf_short_pool.empty:
            etf_short_pool = etf_df.nsmallest(30, "composite")
        if stock_short_pool.empty:
            stock_short_pool = stock_df.nsmallest(30, "composite")
        top_short_etf = _select_top(etf_short_pool, "short") if _full_short else []
        top_short_stock = _select_top(stock_short_pool, "short") if _full_short else []
    else:
        top_long_etf = []
        top_long_stock = []
        top_short_etf = []
        top_short_stock = []

    # ── Additional overview data ──

    # Sector distribution
    sector_dist = fdf.groupby("sector").agg(
        n=("ticker", "size"), eligible=("eligible", "sum"),
        avg_comp=("composite", "mean"),
        avg_ret_1m=("ret_1m", "mean"), avg_ret_3m=("ret_3m", "mean"),
    ).round(1).reset_index()
    sector_dist["eligible"] = sector_dist["eligible"].astype(int)
    sector_dist = sector_dist.sort_values("avg_comp", ascending=False)

    # Asset type split
    asset_split = fdf.groupby("asset_type").agg(
        n=("ticker", "size"), eligible=("eligible", "sum"),
        avg_comp=("composite", "mean"),
    ).round(1).reset_index()
    asset_split["eligible"] = asset_split["eligible"].astype(int)

    # Classification by sector (for heatmap)
    cls_sector = fdf.groupby(["sector", "classification"]).size().reset_index(name="count")

    # Top movers: best/worst 1M return among eligible
    if n_el > 0:
        el_df = fdf[fdf["eligible"]]
        top_movers_up = el_df.nlargest(10, "ret_1m")[["ticker", "name", "sector", "ret_1m", "ret_3m", "composite", "classification"]].to_dict(orient="records")
        top_movers_dn = el_df.nsmallest(10, "ret_1m")[["ticker", "name", "sector", "ret_1m", "ret_3m", "composite", "classification"]].to_dict(orient="records")
    else:
        top_movers_up = fdf.nlargest(10, "ret_1m")[["ticker", "name", "sector", "ret_1m", "ret_3m", "composite", "classification"]].to_dict(orient="records")
        top_movers_dn = fdf.nsmallest(10, "ret_1m")[["ticker", "name", "sector", "ret_1m", "ret_3m", "composite", "classification"]].to_dict(orient="records")

    # Score distribution by axis (TCS/TFS/OER/RSS averages by sector)
    axis_by_sector = fdf.groupby("sector").agg(
        avg_tcs=("tcs", "mean"), avg_tfs=("tfs", "mean"),
        avg_oer=("oer", "mean"), avg_rss=("rss", "mean"),
    ).round(1).reset_index()

    # Trend age distribution
    age_bins = [0, 5, 10, 20, 40, 60, 100, 9999]
    age_labels = ["0-5", "6-10", "11-20", "21-40", "41-60", "61-100", "100+"]
    fdf_copy = fdf.copy()
    fdf_copy["age_bin"] = pd.cut(fdf_copy["trend_age"], bins=age_bins, labels=age_labels, right=True)
    age_dist = fdf_copy["age_bin"].value_counts().sort_index().reset_index()
    age_dist.columns = ["bin", "count"]

    # RSI distribution
    rsi_data = fdf[["rsi", "classification"]].to_dict(orient="records")

    # Additional KPIs
    kpis["avg_rsi"] = round(float(fdf["rsi"].mean()), 1)
    kpis["median_composite"] = round(float(fdf["composite"].median()), 1)
    kpis["pct_downtrend"] = round(
        len(fdf[fdf["classification"].str.contains("DOWNTREND")]) / len(fdf) * 100, 1
    ) if len(fdf) > 0 else 0
    kpis["avg_ret_1m"] = round(float(fdf["ret_1m"].mean()), 2)
    kpis["avg_ret_3m"] = round(float(fdf["ret_3m"].mean()), 2)
    kpis["n_etf"] = int((fdf["asset_type"] == "ETF").sum())
    kpis["n_stock"] = int((fdf["asset_type"] == "Stock").sum())

    return _clean_dict({
        "kpis": kpis,
        "classification_dist": cls_dist.to_dict(orient="records"),
        "composite_data": composite_data,
        "conviction_bubble": bubble,
        "top_eligible": top_eligible,
        "top_long": top_long,
        "top_long_warnings": top_long_warnings,
        "top_short": top_short,
        "top_long_etf": top_long_etf,
        "top_long_stock": top_long_stock,
        "top_short_etf": top_short_etf,
        "top_short_stock": top_short_stock,
        "top_long_bt": _filter_top_long_bt(STATE.get("top_long_bt", []), fdf),
        "sector_dist": sector_dist.to_dict(orient="records"),
        "asset_split": asset_split.to_dict(orient="records"),
        "cls_sector": cls_sector.to_dict(orient="records"),
        "top_movers_up": top_movers_up,
        "top_movers_dn": top_movers_dn,
        "axis_by_sector": axis_by_sector.to_dict(orient="records"),
        "age_dist": age_dist.to_dict(orient="records"),
        "rsi_data": rsi_data,
    })


@app.get("/api/table")
def table(
    categories: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = Query(None),
    subthemes: Optional[List[str]] = Query(None),
):
    fdf = _filter_df(categories, classifications, eligible_only, comp_min, comp_max,
                     sectors=sectors, subthemes=subthemes)
    cols = ["ticker", "name", "category", "sector", "asset_type", "theme", "theme_detail",
            "composite", "tcs", "tfs", "oer", "rss",
            "tcs_short", "tcs_long", "tfs_short", "tfs_long", "rss_short", "rss_long",
            "qvr_score", "qvr_q", "qvr_v", "qvr_r",
            "qvr_n_analysts", "qvr_bullish_chg_3m", "qvr_eps_beat_rate", "qvr_eps_surprise_avg",
            "classification", "eligible", "rejection",
            "rsi", "trend_age", "sma50_dist", "adv_M", "mktcap_B",
            "oneil_long", "oneil_short",
            "minervini_long", "minervini_short",
            "wyckoff_long", "wyckoff_short",
            "ichimoku_long", "ichimoku_short",
            "darvas_long", "darvas_short",
            "regime_long", "regime_short",
            "flow_long", "flow_short",
            "relval_long", "relval_short",
            "combined_long", "combined_short",
            "long_count", "short_count", "net_signal", "conviction",
            "event_flag", "event_reasons", "structural_q", "alpha_potential",
            "ret_36_12m", "reversal_pctile",
            "val_prob", "val_persist", "val_conf",
            "score_1w", "ret_1w", "score_1m", "ret_1m", "score_3m", "ret_3m",
            # Multi-horizon returns + 3Y volatility
            "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
            "ret_3y_ann", "ret_5y_ann", "vol_3y_ann"]
    cols = [c for c in cols if c in fdf.columns]
    records = fdf[cols].round(2).to_dict(orient="records")

    # Enrich with momentum age (bi-weekly retroactive, noise-tolerant)
    try:
        from pre_momentum import compute_momentum_ages
        ve_obs = STATE.get("ve_stats", {}).get("observations", [])
        ages = compute_momentum_ages(STATE.get("results", []), ve_obs)
        for r in records:
            r["mom_age"] = ages.get(r.get("ticker", ""), 0)
    except Exception:
        for r in records:
            r["mom_age"] = 0

    return _clean_dict({"data": records})


@app.get("/api/universe")
def universe(
    categories: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = Query(None),
    subthemes: Optional[List[str]] = Query(None),
):
    """All tickers with period returns and annualized metrics."""
    fdf = _filter_df(categories, classifications, eligible_only, comp_min, comp_max,
                     sectors=sectors, subthemes=subthemes)
    if fdf.empty:
        return {"rows": []}
    cols = ["ticker", "name", "category", "theme", "mktcap_B",
            "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
            "ret_3y_ann", "ret_5y_ann", "vol_3y_ann"]
    cols = [c for c in cols if c in fdf.columns]
    records = fdf[cols].round(2).to_dict(orient="records")
    return _clean_dict({"rows": records})


@app.get("/api/market-regime")
def market_regime(
    categories: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = Query(None),
    subthemes: Optional[List[str]] = Query(None),
):
    """Market regime analysis from cross-sectional signal aggregation."""
    fdf = _filter_df(categories, classifications, eligible_only, comp_min, comp_max,
                     sectors=sectors, subthemes=subthemes)
    if fdf.empty:
        return {"error": "No data"}
    n = len(fdf)

    # ── 1. Classification breadth ──
    bullish_cls = {"🟢 CONTINUATION", "🔵 FORMATION", "🟡 CONSOLIDATION",
                   "🔶 PULLBACK", "🔵 RECOVERY"}
    bearish_cls = {"⬇️ DOWNTREND", "⚠️ WEAKENING", "🟤 FADING",
                   "🟤 EXHAUSTING", "🟣 COUNTER_RALLY", "🔴 CYCLE_PEAK"}
    neutral_cls = {"🟠 NEUTRAL", "🟡 OVEREXTENDED"}

    n_bull = int(fdf["classification"].isin(bullish_cls).sum())
    n_bear = int(fdf["classification"].isin(bearish_cls).sum())
    n_neutral = n - n_bull - n_bear
    pct_bull = round(n_bull / n * 100, 1)
    pct_bear = round(n_bear / n * 100, 1)
    pct_neutral = round(n_neutral / n * 100, 1)

    # Classification distribution (sorted by count)
    cls_counts = fdf["classification"].value_counts()
    cls_dist = [{"classification": k, "count": int(v),
                 "pct": round(v / n * 100, 1),
                 "group": "bullish" if k in bullish_cls else "bearish" if k in bearish_cls else "neutral"}
                for k, v in cls_counts.items()]

    # ── 2. Core breadth metrics ──
    avg_comp = round(float(fdf["composite"].mean()), 1)
    med_comp = round(float(fdf["composite"].median()), 1)
    std_comp = round(float(fdf["composite"].std()), 1)
    avg_tcs = round(float(fdf["tcs"].mean()), 1)
    avg_tfs = round(float(fdf["tfs"].mean()), 1)
    avg_oer = round(float(fdf["oer"].mean()), 1)
    avg_rss = round(float(fdf["rss"].mean()), 1)
    pct_eligible = round(float(fdf["eligible"].sum()) / n * 100, 1)
    avg_rsi = round(float(fdf["rsi"].mean()), 1)

    lc = fdf["long_count"] if "long_count" in fdf.columns else pd.Series([0] * n)
    sc_col = fdf["short_count"] if "short_count" in fdf.columns else pd.Series([0] * n)
    avg_long_count = round(float(lc.mean()), 2)
    avg_short_count = round(float(sc_col.mean()), 2)

    # ── 3. Per-strategy breadth ──
    STRATEGIES = ["oneil", "minervini", "wyckoff", "ichimoku", "darvas", "regime", "flow", "relval"]
    STRATEGY_LABELS = {
        "oneil": "O'Neil", "minervini": "Minervini", "wyckoff": "Wyckoff",
        "ichimoku": "Ichimoku", "darvas": "Darvas", "regime": "Regime",
        "flow": "Flow", "relval": "RelVal",
    }
    strategy_breadth = []
    for s in STRATEGIES:
        lc_col = f"{s}_long"
        sc_col2 = f"{s}_short"
        if lc_col in fdf.columns and sc_col2 in fdf.columns:
            lv = pd.to_numeric(fdf[lc_col], errors="coerce").fillna(0)
            sv = pd.to_numeric(fdf[sc_col2], errors="coerce").fillna(0)
            l_pct = round(float((lv >= 50).sum()) / n * 100, 1)
            s_pct = round(float((sv >= 50).sum()) / n * 100, 1)
            l_avg = round(float(lv.mean()), 1)
            s_avg = round(float(sv.mean()), 1)
            strategy_breadth.append({
                "strategy": s, "label": STRATEGY_LABELS[s],
                "long_breadth": l_pct, "short_breadth": s_pct,
                "net_breadth": round(l_pct - s_pct, 1),
                "long_avg": l_avg, "short_avg": s_avg,
                "net_avg": round(l_avg - s_avg, 1),
            })

    # ── 4. Strategy group analysis ──
    def _group_net(names):
        vals = [sb["net_breadth"] for sb in strategy_breadth if sb["strategy"] in names]
        return round(sum(vals) / len(vals), 1) if vals else 0.0
    def _group_net_avg(names):
        vals = [sb["net_avg"] for sb in strategy_breadth if sb["strategy"] in names]
        return round(sum(vals) / len(vals), 1) if vals else 0.0

    strategy_groups = [
        {"group": "Trend-Following", "strategies": ["minervini", "regime", "darvas"],
         "net_breadth": _group_net(["minervini", "regime", "darvas"]),
         "net_avg": _group_net_avg(["minervini", "regime", "darvas"]),
         "desc": "방향성 추세 추종"},
        {"group": "Volume/Accumulation", "strategies": ["wyckoff", "flow"],
         "net_breadth": _group_net(["wyckoff", "flow"]),
         "net_avg": _group_net_avg(["wyckoff", "flow"]),
         "desc": "수급 기반 매집/분산"},
        {"group": "Breakout", "strategies": ["oneil"],
         "net_breadth": _group_net(["oneil"]),
         "net_avg": _group_net_avg(["oneil"]),
         "desc": "돌파 기반 신규 추세"},
        {"group": "Structure", "strategies": ["ichimoku"],
         "net_breadth": _group_net(["ichimoku"]),
         "net_avg": _group_net_avg(["ichimoku"]),
         "desc": "중기 구조적 추세 확인"},
        {"group": "Mean-Reversion", "strategies": ["relval"],
         "net_breadth": _group_net(["relval"]),
         "net_avg": _group_net_avg(["relval"]),
         "desc": "상대가치 평균회귀"},
    ]

    # ── 5. Regime classification ──
    trend_net = _group_net(["minervini", "regime", "darvas"])
    reversion_net = _group_net(["relval"])
    volume_net = _group_net(["wyckoff", "flow"])

    # Agreement score: how many strategies agree on direction
    pos_strats = sum(1 for sb in strategy_breadth if sb["net_breadth"] > 5)
    neg_strats = sum(1 for sb in strategy_breadth if sb["net_breadth"] < -5)
    agreement = round(abs(pos_strats - neg_strats) / max(len(strategy_breadth), 1) * 100, 1)

    if pct_bull >= 55 and avg_comp >= 55 and avg_long_count >= 3.5:
        regime = "RISK-ON"
        regime_desc = "광범위한 상승 추세. 대부분의 종목이 bullish classification, 높은 composite, 다수 전략 Long 합의."
    elif pct_bear >= 35 and avg_comp <= 45 and avg_short_count >= 2.5:
        regime = "RISK-OFF"
        regime_desc = "광범위한 하락 압력. Bearish classification 우세, 낮은 composite, Short 시그널 확산."
    elif std_comp >= 18 and abs(pct_bull - pct_bear) <= 20:
        regime = "ROTATION"
        regime_desc = "섹터/종목 간 극단적 양극화. Composite 분산 높고 Bull/Bear가 혼재."
    elif avg_tfs >= 45 and pct_bull < 55 and pct_bear < 35:
        regime = "TRANSITION"
        regime_desc = "Regime 전환 초기 신호. Formation/Breakout 시그널 증가, 아직 추세 미확립."
    elif std_comp <= 12 and agreement <= 30:
        regime = "COMPRESSION"
        regime_desc = "변동성 수축 및 시그널 응축. 전략 간 합의 부족, 큰 움직임 직전 가능성."
    elif pct_bull >= 40 and pct_bear <= 20:
        regime = "MILD-BULL"
        regime_desc = "완만한 상승 기조. Bullish 우세하나 강한 추세는 아님."
    elif pct_bear >= 25 and pct_bull <= 40:
        regime = "MILD-BEAR"
        regime_desc = "완만한 하락/조정 기조. Bearish 증가 중이나 아직 광범위하지 않음."
    else:
        regime = "NEUTRAL"
        regime_desc = "방향성 미약. 혼조세로 뚜렷한 regime 특성 없음."

    # ── 6. Sector-level regime ──
    sector_data = []
    for sec, grp in fdf.groupby("sector"):
        ns = len(grp)
        if ns < 2:
            continue
        sb = int(grp["classification"].isin(bullish_cls).sum())
        sbe = int(grp["classification"].isin(bearish_cls).sum())
        slc = float(grp["long_count"].mean()) if "long_count" in grp.columns else 0
        ssc = float(grp["short_count"].mean()) if "short_count" in grp.columns else 0
        sector_data.append({
            "sector": sec, "n": ns,
            "avg_composite": round(float(grp["composite"].mean()), 1),
            "pct_bullish": round(sb / ns * 100, 1),
            "pct_bearish": round(sbe / ns * 100, 1),
            "avg_long_count": round(slc, 1),
            "avg_short_count": round(ssc, 1),
            "avg_tcs": round(float(grp["tcs"].mean()), 1),
            "avg_tfs": round(float(grp["tfs"].mean()), 1),
            "avg_rss": round(float(grp["rss"].mean()), 1),
            "avg_oer": round(float(grp["oer"].mean()), 1),
        })
    sector_data.sort(key=lambda x: -x["avg_composite"])

    # ── 7. Composite distribution ──
    import numpy as _np
    comp_vals = fdf["composite"].dropna().values
    comp_hist_counts, comp_hist_edges = _np.histogram(comp_vals, bins=20, range=(0, 100))
    comp_distribution = {
        "mean": avg_comp, "median": med_comp, "std": std_comp,
        "q25": round(float(_np.percentile(comp_vals, 25)), 1),
        "q75": round(float(_np.percentile(comp_vals, 75)), 1),
        "skew": round(float(pd.Series(comp_vals).skew()), 2),
        "bins": [round(float(e), 1) for e in comp_hist_edges[:-1]],
        "counts": [int(c) for c in comp_hist_counts],
    }

    # ── 8. Signal distribution: net_signal counts ──
    if "net_signal" in fdf.columns:
        ns_counts = fdf["net_signal"].value_counts()
        signal_dist = [{"signal": str(k), "count": int(v)} for k, v in ns_counts.items()]
    else:
        signal_dist = []

    # ── 9. Historical regime from history_7d ──
    history = STATE.get("history", {})
    regime_history = []
    if history:
        # Collect all unique dates across tickers
        date_data: dict = {}
        for ticker, snaps in history.items():
            # Only include tickers in current filtered set
            if ticker not in fdf["ticker"].values:
                continue
            for snap in snaps:
                d = str(snap.get("date", ""))[:10]
                if not d:
                    continue
                if d not in date_data:
                    date_data[d] = {"composites": [], "bullish": 0, "bearish": 0, "total": 0,
                                    "tcs": [], "tfs": [], "oer": [], "eligible": 0}
                dd = date_data[d]
                dd["composites"].append(snap.get("composite", 0))
                dd["tcs"].append(snap.get("tcs", 0))
                dd["tfs"].append(snap.get("tfs", 0))
                dd["oer"].append(snap.get("oer", 0))
                dd["total"] += 1
                cls = snap.get("class", "")
                if cls in bullish_cls:
                    dd["bullish"] += 1
                elif cls in bearish_cls:
                    dd["bearish"] += 1
                if snap.get("eligible", False):
                    dd["eligible"] += 1

        for d in sorted(date_data.keys()):
            dd = date_data[d]
            t = dd["total"]
            if t < 5:
                continue
            regime_history.append({
                "date": d,
                "n": t,
                "avg_composite": round(sum(dd["composites"]) / t, 1),
                "pct_bullish": round(dd["bullish"] / t * 100, 1),
                "pct_bearish": round(dd["bearish"] / t * 100, 1),
                "pct_eligible": round(dd["eligible"] / t * 100, 1),
                "avg_tcs": round(sum(dd["tcs"]) / t, 1),
                "avg_tfs": round(sum(dd["tfs"]) / t, 1),
                "avg_oer": round(sum(dd["oer"]) / t, 1),
            })

    return _clean_dict({
        "regime": regime, "regime_desc": regime_desc,
        "breadth": {
            "n": n, "n_bull": n_bull, "n_bear": n_bear, "n_neutral": n_neutral,
            "pct_bull": pct_bull, "pct_bear": pct_bear, "pct_neutral": pct_neutral,
            "avg_composite": avg_comp, "median_composite": med_comp, "std_composite": std_comp,
            "avg_tcs": avg_tcs, "avg_tfs": avg_tfs, "avg_oer": avg_oer, "avg_rss": avg_rss,
            "avg_rsi": avg_rsi, "pct_eligible": pct_eligible,
            "avg_long_count": avg_long_count, "avg_short_count": avg_short_count,
        },
        "strategy_breadth": strategy_breadth,
        "strategy_groups": strategy_groups,
        "agreement_score": agreement,
        "classification_dist": cls_dist,
        "signal_dist": signal_dist,
        "composite_distribution": comp_distribution,
        "sector_regime": sector_data,
        "regime_history": regime_history,
    })


@app.get("/api/weekly-heatmap")
def weekly_heatmap():
    """Weekly backtest data formatted for 4 heatmaps: ticker, sector, category, theme."""
    bt_data = STATE.get("top_long_bt", [])
    if not bt_data:
        return {"error": "No backtest data"}

    snaps = sorted(bt_data, key=lambda s: s["eval_date"])
    dates = [s["eval_date"] for s in snaps]

    # 1. Ticker heatmap: top 30 by selection frequency, Z = 1M forward return
    tk_freq: dict = {}
    for snap in snaps:
        d = snap["eval_date"]
        for t in snap.get("tickers", []):
            tk = t["ticker"]
            if tk not in tk_freq:
                tk_freq[tk] = {"count": 0, "rets": {}, "cat": "", "theme": ""}
            tk_freq[tk]["count"] += 1
            tk_freq[tk]["rets"][d] = t.get("ret_1M")
            tk_freq[tk]["cat"] = t.get("category", "")
            tk_freq[tk]["theme"] = t.get("theme", "")

    top30 = sorted(tk_freq.items(), key=lambda x: -x[1]["count"])[:30]
    ticker_heatmap = {
        "tickers": [tk for tk, _ in top30],
        "dates": dates,
        "z": [[info["rets"].get(d) for d in dates] for _, info in top30],
        "categories": [info["cat"] for _, info in top30],
        "counts": [info["count"] for _, info in top30],
    }

    # 2-4. Group heatmaps
    group_heatmaps = {}
    for gtype in ("sector", "category", "theme"):
        all_keys: set = set()
        for snap in snaps:
            ga = snap.get("group_agg", {}).get(gtype, {})
            all_keys.update(ga.keys())
        keys = sorted(all_keys)
        z_comp = []
        z_bull = []
        for key in keys:
            row_c, row_b = [], []
            for snap in snaps:
                ga = snap.get("group_agg", {}).get(gtype, {}).get(key)
                row_c.append(ga["avg_comp"] if ga else None)
                row_b.append(ga["pct_bullish"] if ga else None)
            z_comp.append(row_c)
            z_bull.append(row_b)
        group_heatmaps[gtype] = {"keys": keys, "dates": dates, "z_composite": z_comp, "z_pct_bullish": z_bull}

    return _clean_dict({
        "dates": dates,
        "n_snapshots": len(snaps),
        "ticker_heatmap": ticker_heatmap,
        "sector_heatmap": group_heatmaps["sector"],
        "category_heatmap": group_heatmaps["category"],
        "theme_heatmap": group_heatmaps["theme"],
    })


@app.get("/api/category")
def category_analysis(
    categories: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = Query(None),
    subthemes: Optional[List[str]] = Query(None),
):
    fdf = _filter_df(categories, classifications, eligible_only, comp_min, comp_max,
                     sectors=sectors, subthemes=subthemes)
    if fdf.empty:
        return {"summary": [], "details": {}}

    agg = fdf.groupby("category").agg(
        count=("ticker", "size"),
        eligible=("eligible", "sum"),
        avg_comp=("composite", "mean"),
        avg_tcs=("tcs", "mean"),
        avg_tfs=("tfs", "mean"),
        avg_oer=("oer", "mean"),
        avg_ret_1w=("ret_1w", "mean"),
        avg_ret_1m=("ret_1m", "mean"),
        avg_ret_3m=("ret_3m", "mean"),
    ).round(1).reset_index()
    agg["eligible"] = agg["eligible"].astype(int)
    agg = agg.sort_values("avg_comp", ascending=False)

    details = {}
    det_cols = ["ticker", "name", "sector", "category", "theme",
                "composite", "tcs", "tfs", "oer", "rss",
                "classification", "eligible", "rejection",
                "rsi", "trend_age", "oneil_long", "oneil_short",
                "event_flag", "event_reasons", "structural_q",
                "val_prob", "val_persist",
                "ret_1w", "ret_1m", "ret_3m"]
    det_cols = [c for c in det_cols if c in fdf.columns]
    for cat in agg["category"]:
        cat_df = fdf[fdf["category"] == cat].sort_values("composite", ascending=False)
        details[cat] = cat_df[det_cols].round(2).to_dict(orient="records")

    return _clean_dict({
        "summary": agg.to_dict(orient="records"),
        "details": details,
    })


@app.get("/api/theme")
def theme_analysis(
    categories: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = Query(None),
    subthemes: Optional[List[str]] = Query(None),
    min_n: int = 2,
):
    fdf = _filter_df(categories, classifications, eligible_only, comp_min, comp_max,
                     sectors=sectors, subthemes=subthemes)
    themed = fdf[fdf["theme"] != "-"].copy()
    if themed.empty:
        return {"summary": [], "classification_by_theme": [], "momentum_map": [], "details": {}}

    agg = themed.groupby("theme").agg(
        n=("ticker", "size"),
        eligible=("eligible", "sum"),
        avg_comp=("composite", "mean"),
        avg_tcs=("tcs", "mean"),
        avg_tfs=("tfs", "mean"),
        avg_oer=("oer", "mean"),
        avg_rsi=("rsi", "mean"),
        avg_ret_1w=("ret_1w", "mean"),
        avg_ret_1m=("ret_1m", "mean"),
        avg_ret_3m=("ret_3m", "mean"),
        categories=("category", lambda x: ", ".join(sorted(set(c.replace("STK_", "") for c in x)))),
    ).round(1).reset_index()
    agg["eligible"] = agg["eligible"].astype(int)
    agg = agg[agg["n"] >= min_n].sort_values("avg_comp", ascending=False)

    valid_themes = set(agg["theme"])
    tf = themed[themed["theme"].isin(valid_themes)]

    cls_by_theme = tf.groupby(["theme", "classification"]).size().reset_index(name="count")

    mom_cols = ["ticker", "name", "composite", "ret_1m", "theme", "adv_M",
                "classification", "category", "tcs", "tfs"]
    mom_cols = [c for c in mom_cols if c in tf.columns]
    momentum_map = tf[mom_cols].round(2).to_dict(orient="records")

    details = {}
    det_cols = ["ticker", "name", "sector", "category", "theme",
                "composite", "tcs", "tfs", "oer", "rss",
                "classification", "eligible", "rejection",
                "rsi", "trend_age", "oneil_long", "oneil_short",
                "event_flag", "event_reasons", "structural_q",
                "val_prob", "val_persist",
                "ret_1w", "ret_1m", "ret_3m"]
    det_cols = [c for c in det_cols if c in tf.columns]
    for theme in agg["theme"]:
        tdf = tf[tf["theme"] == theme].sort_values("composite", ascending=False)
        details[theme] = tdf[det_cols].round(2).to_dict(orient="records")

    return _clean_dict({
        "summary": agg.to_dict(orient="records"),
        "classification_by_theme": cls_by_theme.to_dict(orient="records"),
        "momentum_map": momentum_map,
        "details": details,
    })


@app.get("/api/effectiveness")
def effectiveness():
    ve = STATE.get("ve_stats", {})
    obs = ve.get("observations", [])
    if not obs:
        return {"kpis": {}, "quintiles": {}, "classification_summary": {},
                "scatter": {}, "ic_timeseries": {}, "box_data": {}, "regression": {},
                "periods": []}

    obs_df = pd.DataFrame(obs)

    cls_short_map = {
        "⬇️ DOWNTREND": "DOWN", "🟤 FADING": "FADING", "🟣 COUNTER_RALLY": "CNTR",
        "⚠️ WEAKENING": "WEAK", "🟠 NEUTRAL": "NEUTRAL", "🟡 CONSOLIDATION": "CONSOL",
        "🔶 PULLBACK": "PULL", "🔵 RECOVERY": "RECV", "🔵 FORMATION": "FORM",
        "🟡 OVEREXTENDED": "OVEXT", "🟤 EXHAUSTING": "EXHAUST", "🟢 CONTINUATION": "CONT",
    }
    obs_df["cls_short"] = obs_df["classification"].map(cls_short_map).fillna("OTHER")

    # Extract fixed forward returns into flat columns
    FWD_PERIODS = {"1W": 5, "1M": 21, "3M": 63}
    for label, fd in FWD_PERIODS.items():
        obs_df[f"fwd_{label}"] = obs_df["fwd_rets"].apply(lambda x: x.get(fd) if isinstance(x, dict) else None)
        obs_df[f"bench_{label}"] = obs_df["fwd_bench"].apply(lambda x: x.get(fd, 0) if isinstance(x, dict) else 0)
        obs_df[f"exc_{label}"] = obs_df[f"fwd_{label}"] - obs_df[f"bench_{label}"]

    from scipy.stats import spearmanr, linregress

    # ── Per-period analysis ──
    per_period = {}
    for label, fd in FWD_PERIODS.items():
        pdf = obs_df.dropna(subset=[f"fwd_{label}"]).copy()
        if len(pdf) < 20:
            continue

        fwd_col = f"fwd_{label}"
        exc_col = f"exc_{label}"

        # IC time series
        ic_ts = []
        for dt, grp in pdf.groupby("eval_date"):
            if len(grp) < 10: continue
            try:
                ic, pv = spearmanr(grp["score"], grp[fwd_col])
                ic_ts.append({"date": str(dt), "IC": round(float(ic), 4), "n": len(grp)})
            except: pass

        # Overall IC
        try: o_ic, o_pv = spearmanr(pdf["score"], pdf[fwd_col])
        except: o_ic, o_pv = 0, 1

        # Quintiles
        try:
            pdf["quintile"] = pd.qcut(pdf["score"], 5,
                                       labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"],
                                       duplicates="drop")
        except:
            pdf["quintile"] = "ALL"
        qstats = pdf.groupby("quintile", observed=True).agg(
            n=("score", "size"), avg_score=("score", "mean"),
            avg_fwd=(fwd_col, "mean"), avg_exc=(exc_col, "mean"),
            hit_rate=(fwd_col, lambda x: (x > 0).mean() * 100),
        ).round(2).reset_index()
        qstats["quintile"] = qstats["quintile"].astype(str)

        # Classification summary
        cls_stats = pdf.groupby("cls_short").agg(
            n=("score", "size"), avg_score=("score", "mean"),
            avg_fwd=(fwd_col, "mean"), avg_exc=(exc_col, "mean"),
            hit_rate=(exc_col, lambda x: (x > 0).mean() * 100),
        ).round(2).reset_index()

        # Scatter
        scatter = pdf[["score", exc_col, "cls_short", "ticker", "eval_date"]].rename(
            columns={exc_col: "excess_return"})
        if len(scatter) > 2000:
            scatter = scatter.sample(2000, random_state=42)

        # Regression
        try:
            slope, intercept, r, p, se = linregress(pdf["score"], pdf[exc_col])
            reg = {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4),
                   "r_squared": round(float(r**2), 4), "p_value": round(float(p), 6)}
        except:
            reg = {"slope": 0, "intercept": 0, "r_squared": 0, "p_value": 1}

        # Box data
        box = pdf[["cls_short", exc_col]].rename(columns={exc_col: "excess_return"}).to_dict(orient="records")

        per_period[label] = {
            "kpis": {
                "ic_spearman": round(float(o_ic), 4), "ic_pval": round(float(o_pv), 6),
                "overall_hit_rate": round(float((pdf[exc_col] > 0).mean() * 100), 1),
                "avg_excess": round(float(pdf[exc_col].mean()), 2),
                "n_observations": len(pdf),
            },
            "ic_timeseries": ic_ts,
            "quintiles": qstats.to_dict(orient="records"),
            "classification_summary": cls_stats.to_dict(orient="records"),
            "scatter": scatter.round(2).to_dict(orient="records"),
            "box_data": box,
            "regression": reg,
        }

    return _clean_dict({
        "periods": list(FWD_PERIODS.keys()),
        "per_period": per_period,
    })


@app.get("/api/period-analysis")
def period_analysis(fwd_days: int = 21):
    """Detailed analysis for a user-selected forward period (1-126 trading days)."""
    ve = STATE.get("ve_stats", {})
    obs = ve.get("observations", [])
    if not obs:
        return {"error": "No observations"}

    fwd_days = max(1, min(252, fwd_days))
    obs_df = pd.DataFrame(obs)

    # Extract forward return for requested period
    obs_df["fwd_ret"] = obs_df["fwd_rets"].apply(lambda x: x.get(fwd_days) if isinstance(x, dict) else None)
    obs_df["fwd_bench"] = obs_df["fwd_bench"].apply(lambda x: x.get(fwd_days, 0) if isinstance(x, dict) else 0)
    pdf = obs_df.dropna(subset=["fwd_ret"]).copy()

    if len(pdf) < 20:
        return {"error": f"Not enough data for {fwd_days}d forward", "n": len(pdf)}

    pdf["exc_ret"] = pdf["fwd_ret"] - pdf["fwd_bench"]

    cls_short_map = {
        "⬇️ DOWNTREND": "DOWN", "🟤 FADING": "FADING", "🟣 COUNTER_RALLY": "CNTR",
        "⚠️ WEAKENING": "WEAK", "🟠 NEUTRAL": "NEUTRAL", "🟡 CONSOLIDATION": "CONSOL",
        "🔶 PULLBACK": "PULL", "🔵 RECOVERY": "RECV", "🔵 FORMATION": "FORM",
        "🟡 OVEREXTENDED": "OVEXT", "🟤 EXHAUSTING": "EXHAUST", "🟢 CONTINUATION": "CONT",
    }
    pdf["cls_short"] = pdf["classification"].map(cls_short_map).fillna("OTHER")

    from scipy.stats import spearmanr, linregress

    # IC time series
    ic_ts = []
    for dt, grp in pdf.groupby("eval_date"):
        if len(grp) < 10: continue
        try:
            ic, _ = spearmanr(grp["score"], grp["fwd_ret"])
            ic_ts.append({"date": str(dt), "IC": round(float(ic), 4), "n": len(grp)})
        except: pass

    # Overall IC
    try: o_ic, o_pv = spearmanr(pdf["score"], pdf["fwd_ret"])
    except: o_ic, o_pv = 0, 1

    # Quintiles
    try:
        pdf["quintile"] = pd.qcut(pdf["score"], 5,
                                   labels=["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"],
                                   duplicates="drop")
    except:
        pdf["quintile"] = "ALL"
    qstats = pdf.groupby("quintile", observed=True).agg(
        n=("score", "size"), avg_score=("score", "mean"),
        avg_fwd=("fwd_ret", "mean"), avg_exc=("exc_ret", "mean"),
        abs_hit=("fwd_ret", lambda x: round((x > 0).mean() * 100, 1)),
        exc_hit=("exc_ret", lambda x: round((x > 0).mean() * 100, 1)),
    ).round(2).reset_index()
    qstats["quintile"] = qstats["quintile"].astype(str)

    # Classification summary
    cls_stats = pdf.groupby("cls_short").agg(
        n=("score", "size"), avg_score=("score", "mean"),
        avg_fwd=("fwd_ret", "mean"), avg_exc=("exc_ret", "mean"),
        abs_hit=("fwd_ret", lambda x: round((x > 0).mean() * 100, 1)),
        exc_hit=("exc_ret", lambda x: round((x > 0).mean() * 100, 1)),
    ).round(2).reset_index()

    # Score bucket summary
    bucket_labels = ["0-30", "30-50", "50-70", "70-100"]
    pdf["bucket"] = pd.cut(pdf["score"], bins=[0, 30, 50, 70, 100.1],
                            labels=bucket_labels, right=False)
    bkt_stats = pdf.groupby("bucket", observed=True).agg(
        n=("score", "size"), avg_score=("score", "mean"),
        avg_fwd=("fwd_ret", "mean"), avg_exc=("exc_ret", "mean"),
        abs_hit=("fwd_ret", lambda x: round((x > 0).mean() * 100, 1)),
        exc_hit=("exc_ret", lambda x: round((x > 0).mean() * 100, 1)),
    ).round(2).reset_index()
    bkt_stats["bucket"] = bkt_stats["bucket"].astype(str)

    # Scatter (sample)
    scatter = pdf[["score", "exc_ret", "fwd_ret", "cls_short", "ticker", "eval_date"]].copy()
    scatter = scatter.rename(columns={"exc_ret": "excess_return", "fwd_ret": "forward_return"})
    if len(scatter) > 2000:
        scatter = scatter.sample(2000, random_state=42)

    # Regression
    try:
        slope, intercept, r, p, se = linregress(pdf["score"], pdf["exc_ret"])
        reg = {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4),
               "r_squared": round(float(r**2), 4), "p_value": round(float(p), 6)}
    except:
        reg = {"slope": 0, "intercept": 0, "r_squared": 0, "p_value": 1}

    # Box data
    box = pdf[["cls_short", "exc_ret", "fwd_ret"]].rename(
        columns={"exc_ret": "excess_return", "fwd_ret": "forward_return"}).to_dict(orient="records")

    # Detect if consensus score is available (new format) or legacy composite-only
    has_consensus = "combined_long" in pdf.columns
    score_label = "Multi-Strategy Consensus (combined_long×0.5 + composite×0.5)" if has_consensus else "Composite Score"

    kpis = {
        "fwd_days": fwd_days,
        "score_type": score_label,
        "ic_spearman": round(float(o_ic), 4), "ic_pval": round(float(o_pv), 6),
        "abs_hit_rate": round(float((pdf["fwd_ret"] > 0).mean() * 100), 1),
        "exc_hit_rate": round(float((pdf["exc_ret"] > 0).mean() * 100), 1),
        "avg_fwd": round(float(pdf["fwd_ret"].mean()), 2),
        "avg_excess": round(float(pdf["exc_ret"].mean()), 2),
        "n_observations": len(pdf),
    }

    return _clean_dict({
        "kpis": kpis,
        "ic_timeseries": ic_ts,
        "quintiles": qstats.to_dict(orient="records"),
        "buckets": bkt_stats.to_dict(orient="records"),
        "classification_summary": cls_stats.to_dict(orient="records"),
        "scatter": scatter.round(2).to_dict(orient="records"),
        "box_data": box,
        "regression": reg,
    })


@app.get("/api/period-curve")
def period_curve():
    """Pre-compute hit rates across all available forward periods for the slider curve."""
    ve = STATE.get("ve_stats", {})
    obs = ve.get("observations", [])
    if not obs:
        return {"curve": []}

    obs_df = pd.DataFrame(obs)
    # Check which forward days are available
    sample_fwd = obs_df["fwd_rets"].iloc[0] if len(obs_df) > 0 else {}
    available_days = sorted(int(k) for k in (sample_fwd.keys() if isinstance(sample_fwd, dict) else []))

    from scipy.stats import spearmanr

    curve = []
    for fd in available_days:
        obs_df[f"_fwd"] = obs_df["fwd_rets"].apply(lambda x: x.get(fd) if isinstance(x, dict) else None)
        obs_df[f"_bench"] = obs_df["fwd_bench"].apply(lambda x: x.get(fd, 0) if isinstance(x, dict) else 0)
        valid = obs_df.dropna(subset=["_fwd"])
        if len(valid) < 20: continue
        exc = valid["_fwd"] - valid["_bench"]
        try: ic, _ = spearmanr(valid["score"], valid["_fwd"])
        except: ic = 0
        curve.append({
            "fwd_days": fd,
            "label": f"{fd}d" if fd < 5 else f"{fd//5}W" if fd <= 10 else f"{fd//21}M" if fd % 21 == 0 else f"{fd}d",
            "n": len(valid),
            "abs_hit": round(float((valid["_fwd"] > 0).mean() * 100), 1),
            "exc_hit": round(float((exc > 0).mean() * 100), 1),
            "avg_fwd": round(float(valid["_fwd"].mean()), 2),
            "avg_exc": round(float(exc.mean()), 2),
            "ic": round(float(ic), 4),
        })

    return _clean_dict({"curve": curve})


@app.get("/api/graph")
def graph_analysis():
    g = STATE.get("graph", {})
    if not g:
        return {"summary": {}, "communities": [], "insights": [],
                "viz_data": {}, "formation_pipeline": {}, "llm_export": ""}

    # Convert community_stats to list
    comm_list = []
    for cid, stats in g.get("community_stats", {}).items():
        comm_list.append({"id": cid, **stats})

    return _clean_dict({
        "summary": g.get("summary", {}),
        "communities": comm_list,
        "insights": g.get("insights", []),
        "viz_data": g.get("viz_data", {}),
        "formation_pipeline": g.get("formation_pipeline", {}),
        "llm_export": g.get("llm_export", ""),
    })


@app.get("/api/report")
def report(
    categories: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = Query(None),
    subthemes: Optional[List[str]] = Query(None),
):
    """Generate markdown report from current data."""
    fdf = _filter_df(categories, classifications, eligible_only, comp_min, comp_max,
                     sectors=sectors, subthemes=subthemes)
    df = STATE.get("df", pd.DataFrame())
    if fdf.empty:
        return {"report_md": "No data available.", "llm_prompt": ""}

    # Build a simplified report
    n = len(fdf); n_el = int(fdf["eligible"].sum())
    cls_dist = fdf["classification"].value_counts().to_dict()
    cats = sorted(fdf["category"].unique())

    lines = []
    lines.append(f"# Global Price Discovery Scanner v5.0 — Analytical Report")
    lines.append(f"\n**Scan Time**: {STATE.get('scan_time', 'N/A')}")
    lines.append(f"**Universe**: {n} tickers ({n_el} eligible)")
    lines.append(f"\n## 1. Market Overview")
    lines.append(f"\n| Classification | Count | % |")
    lines.append(f"|---|---|---|")
    for cls, cnt in sorted(cls_dist.items(), key=lambda x: -x[1]):
        lines.append(f"| {cls} | {cnt} | {cnt/n*100:.1f}% |")

    lines.append(f"\n## 2. Category Summary")
    lines.append(f"\n| Category | N | Eligible | Avg Composite | 1M Return |")
    lines.append(f"|---|---|---|---|---|")
    cat_agg = fdf.groupby("category").agg(
        n=("ticker", "size"), eligible=("eligible", "sum"),
        avg_comp=("composite", "mean"), avg_ret_1m=("ret_1m", "mean"),
    ).round(1)
    for cat, row in cat_agg.iterrows():
        lines.append(f"| {cat} | {int(row['n'])} | {int(row['eligible'])} | {row['avg_comp']:.1f} | {row['avg_ret_1m']:.1f}% |")

    lines.append(f"\n## 3. Top 20 by Composite")
    lines.append(f"\n| Rank | Ticker | Name | Composite | Class | TCS | TFS | OER | RSS |")
    lines.append(f"|---|---|---|---|---|---|---|---|---|")
    top20 = fdf.nlargest(20, "composite")
    for i, (_, r) in enumerate(top20.iterrows()):
        lines.append(f"| {i+1} | {r['ticker']} | {r['name'][:20]} | {r['composite']:.1f} | "
                     f"{r['classification'][:15]} | {r['tcs']} | {r['tfs']} | {r['oer']} | {r['rss']} |")

    # Theme summary
    themed = fdf[fdf["theme"] != "-"]
    if not themed.empty:
        lines.append(f"\n## 4. Theme Summary (Top 15)")
        theme_agg = themed.groupby("theme").agg(
            n=("ticker", "size"), avg_comp=("composite", "mean"),
            avg_ret_1m=("ret_1m", "mean"),
        ).round(1).sort_values("avg_comp", ascending=False).head(15)
        lines.append(f"\n| Theme | N | Avg Composite | 1M Return |")
        lines.append(f"|---|---|---|---|")
        for theme, row in theme_agg.iterrows():
            lines.append(f"| {theme} | {int(row['n'])} | {row['avg_comp']:.1f} | {row['avg_ret_1m']:.1f}% |")

    report_md = "\n".join(lines)

    # LLM Prompt
    llm_prompt = f"""당신은 글로벌 자산운용사의 수석 포트폴리오 매니저 겸 리서치 헤드입니다.
아래는 Price Discovery Scanner v5.0의 분석 결과입니다.
이 데이터를 기반으로 A4 20페이지 분량의 포괄적 시장 분석 보고서를 한국어로 작성해주세요.

{report_md}
"""

    return {"report_md": report_md, "llm_prompt": llm_prompt}


# ═══════════════════════════════════════════════════════════════
# QUANT STRATEGIES
# ═══════════════════════════════════════════════════════════════

@app.get("/api/quant-strategies")
def quant_strategies():
    """Run 6 quant strategies on current scan results."""
    results = STATE.get("results", [])
    if not results:
        return {"strategies": {}}
    from quant_strategies import compute_all_strategies
    return _clean_dict({"strategies": compute_all_strategies(results)})


@app.get("/api/pre-momentum")
def pre_momentum():
    """Pre-Momentum Detection: multi-agent analysis of pre-breakout conditions."""
    if not STATE.get("results"):
        return {"candidates": [], "summary": {}, "methodology": {}}
    from pre_momentum import run_pre_momentum
    # Load fundamentals cache (for QVR Agent — 5th agent, fundamentals dimension)
    try:
        from fundamentals_pipeline import load_fundamentals_cache
        fund_cache = load_fundamentals_cache()
    except Exception:
        fund_cache = None

    cache = {
        "results": STATE["results"],
        "graph": STATE.get("graph"),
        "history": STATE.get("history"),
        "ve_observations": STATE.get("ve_stats", {}).get("observations", []),
        "fundamentals": fund_cache,
    }
    output = run_pre_momentum(cache)

    # Inject sector field into candidates (computed from df in _load_cache)
    df = STATE.get("df")
    if df is not None and "sector" in df.columns:
        sector_map = dict(zip(df["ticker"], df["sector"]))
        for c in output.get("candidates", []):
            c["sector"] = sector_map.get(c.get("ticker", ""), "Other")
        for c in output.get("candidates_etf", []):
            c["sector"] = sector_map.get(c.get("ticker", ""), "Other")
        for c in output.get("candidates_stock", []):
            c["sector"] = sector_map.get(c.get("ticker", ""), "Other")

    return _clean_dict(output)


@app.get("/api/factor-efficacy")
def factor_efficacy():
    """Factor Efficacy — Reverse Factor Model (5 Methodologies)."""
    fe = STATE.get("factor_efficacy", {})
    if not fe:
        return {"error": "No factor efficacy data. Re-run price_discovery.py."}
    return _clean_dict(fe)


@app.get("/api/classification")
def classification_meta():
    """Unified classification metadata + per-ticker GICS/cap-tier."""
    uc = STATE.get("unified_classification", {})
    if not uc:
        return {"available": False,
                "message": "Run `python3 unified_classifier.py` to generate."}
    tickers = uc.get("tickers", {})
    # Aggregate distribution stats
    by_gics: Dict[str, int] = {}
    by_cap: Dict[str, int] = {}
    by_country: Dict[str, int] = {}
    for c in tickers.values():
        if not c.get("ok"):
            continue
        sec = c.get("gics_sector") or "Unknown"
        cap = c.get("cap_tier") or "Unknown"
        country = c.get("country") or "?"
        by_gics[sec] = by_gics.get(sec, 0) + 1
        by_cap[cap] = by_cap.get(cap, 0) + 1
        by_country[country] = by_country.get(country, 0) + 1
    return _clean_dict({
        "available": True,
        "as_of": uc.get("as_of"),
        "n_total": uc.get("n_total"),
        "n_success": uc.get("n_success"),
        "n_failure": uc.get("n_failure"),
        "distribution": {
            "by_gics_sector": by_gics,
            "by_cap_tier": by_cap,
            "by_country": by_country,
        },
    })


@app.get("/api/classification/validation")
def classification_validation():
    """Curated vs auto-GICS agreement report (mismatch table)."""
    uc = STATE.get("unified_classification", {})
    if not uc:
        return {"error": "Unified classification not loaded."}
    try:
        from unified_classifier import validate
        report = validate(uc)
        return _clean_dict(report)
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# ML ENDPOINTS — Composite re-weighted via Optuna (per asset class)
# ═══════════════════════════════════════════════════════════════

def _filter_df_ml(
    categories: Optional[List[str]] = None,
    classifications: Optional[List[str]] = None,
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = None,
    subthemes: Optional[List[str]] = None,
):
    """Filter STATE['df'] using ML-rescored columns instead of original.

    ML columns: composite_ml, classification_ml, eligible_ml
    """
    df = STATE.get("df")
    if df is None or df.empty or "composite_ml" not in df.columns:
        return pd.DataFrame()
    mask = df["composite_ml"].between(comp_min, comp_max)
    if categories:
        mask &= df["category"].isin(categories)
    if sectors:
        mask &= df["sector"].isin(sectors)
    if subthemes:
        mask &= df["theme"].isin(subthemes)
    if classifications:
        mask &= df["classification_ml"].isin(classifications)
    if eligible_only:
        mask &= df["eligible_ml"].fillna(False)
    return df[mask].copy()


@app.get("/api/ml/meta")
def ml_meta():
    """ML scoring metadata — params, stage counts, weights per asset class."""
    return _clean_dict({
        "available": bool(STATE.get("results_ml")),
        "ml_meta": STATE.get("ml_meta", {}),
    })


@app.get("/api/ml/table")
def ml_table(
    categories: Optional[List[str]] = Query(None),
    classifications: Optional[List[str]] = Query(None),
    eligible_only: bool = False,
    comp_min: float = 0, comp_max: float = 100,
    sectors: Optional[List[str]] = Query(None),
    subthemes: Optional[List[str]] = Query(None),
):
    """ML-rescored master table — same shape as /api/table but Composite/Classification
    are the ML-rescored values (composite_ml / classification_ml / eligible_ml exposed
    under the standard column names so frontend reuse is trivial)."""
    fdf = _filter_df_ml(categories, classifications, eligible_only, comp_min, comp_max,
                          sectors=sectors, subthemes=subthemes)
    if fdf.empty:
        return _clean_dict({"data": []})

    # Project ML columns onto the standard names (mirror /api/table contract)
    fdf = fdf.copy()
    fdf["composite"] = fdf["composite_ml"]
    fdf["classification"] = fdf["classification_ml"]
    fdf["eligible"] = fdf["eligible_ml"].fillna(False)
    fdf["rejection"] = fdf["rejection_ml"].fillna("")

    cols = ["ticker", "name", "category", "sector", "asset_type", "theme", "theme_detail",
            "composite", "tcs", "tfs", "oer", "rss",
            "tcs_short", "tcs_long", "tfs_short", "tfs_long", "rss_short", "rss_long",
            "qvr_score", "qvr_q", "qvr_v", "qvr_r",
            "classification", "eligible", "rejection",
            "rsi", "trend_age", "sma50_dist", "adv_M", "mktcap_B",
            "combined_long", "combined_short",
            "long_count", "short_count", "net_signal", "conviction",
            "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
            "ret_3y_ann", "ret_5y_ann", "vol_3y_ann",
            # ML-specific extras
            "composite_ml", "classification_ml", "eligible_ml",
            "rejection_ml", "stage_ml", "asset_class_ml"]
    cols = [c for c in cols if c in fdf.columns]
    records = fdf[cols].round(2).to_dict(orient="records")

    # Momentum age (same as /api/table)
    try:
        from pre_momentum import compute_momentum_ages
        ve_obs = STATE.get("ve_stats", {}).get("observations", [])
        ages = compute_momentum_ages(STATE.get("results", []), ve_obs)
        for r in records:
            r["mom_age"] = ages.get(r.get("ticker", ""), 0)
    except Exception:
        for r in records:
            r["mom_age"] = 0

    return _clean_dict({"data": records})


@app.get("/api/ml/pre-momentum")
def ml_pre_momentum():
    """Pre-Momentum (ML mode) — same 5-agent output as /api/pre-momentum.

    Note: the 5 Pre-Momentum agents (Micro/Macro/Graph/Catalyst/QVR) are
    independent of the Composite formula, so their scoring is unchanged.
    What differs in ML mode is the lifecycle filter:
        a candidate counts as Pre-Momentum only if NOT in ML-momentum
        (i.e., its ML eligibility = False).
    """
    if not STATE.get("results"):
        return {"candidates": [], "summary": {}, "methodology": {}}
    if not STATE.get("results_ml"):
        return {"error": "ML cache not loaded. Run optimize_params.py + score_ml.py."}

    from pre_momentum import run_pre_momentum
    try:
        from fundamentals_pipeline import load_fundamentals_cache
        fund_cache = load_fundamentals_cache()
    except Exception:
        fund_cache = None
    cache_input = {
        "results": STATE["results"],
        "graph": STATE.get("graph"),
        "history": STATE.get("history"),
        "ve_observations": STATE.get("ve_stats", {}).get("observations", []),
        "fundamentals": fund_cache,
    }
    output = run_pre_momentum(cache_input)

    # Filter: drop candidates that are now ML-momentum (eligible_ml=True)
    df = STATE.get("df")
    eligible_ml_set: set = set()
    if df is not None and "eligible_ml" in df.columns:
        eligible_ml_set = set(df.loc[df["eligible_ml"].fillna(False), "ticker"].tolist())

    def _filter_pm(lst):
        if not lst:
            return []
        return [c for c in lst if c.get("ticker") not in eligible_ml_set]

    output["candidates"] = _filter_pm(output.get("candidates", []))
    output["candidates_etf"] = _filter_pm(output.get("candidates_etf", []))
    output["candidates_stock"] = _filter_pm(output.get("candidates_stock", []))

    # Sector enrichment (same as /api/pre-momentum)
    if df is not None and "sector" in df.columns:
        sector_map = dict(zip(df["ticker"], df["sector"]))
        for c in output.get("candidates", []):
            c["sector"] = sector_map.get(c.get("ticker", ""), "Other")
        for c in output.get("candidates_etf", []):
            c["sector"] = sector_map.get(c.get("ticker", ""), "Other")
        for c in output.get("candidates_stock", []):
            c["sector"] = sector_map.get(c.get("ticker", ""), "Other")

    # Update summary
    output.setdefault("summary", {})
    output["summary"]["n_candidates_after_ml_filter"] = len(output["candidates"])
    output["summary"]["mode"] = "ml"

    return _clean_dict(output)


@app.get("/api/ml/classification-history")
def ml_classification_history():
    """ML classification distribution — current snapshot only.

    Historical observations don't carry ML classifications (those require
    re-running the optimizer over historical OHLCV). For now we return
    only the current ML class distribution alongside the existing
    classification-history payload so the UI can compare side-by-side.
    """
    df = STATE.get("df")
    if df is None or "classification_ml" not in df.columns:
        return {"current_distribution": {}, "ml_meta": STATE.get("ml_meta", {})}
    dist = df["classification_ml"].value_counts().to_dict()
    return _clean_dict({
        "current_distribution": dist,
        "ml_meta": STATE.get("ml_meta", {}),
    })


@app.get("/api/sector-rotation")
def sector_rotation():
    """US Sector Rotation — within-11 SPDR ETF ranking + tier + breadth + pairs."""
    if not STATE.get("results"):
        return {"sectors": [], "summary": {}, "pairs": [], "methodology": {}}
    from sector_rotation import compute_sector_rotation
    return _clean_dict(compute_sector_rotation(
        results=STATE["results"],
        df=STATE.get("df"),
        scan_time=STATE.get("scan_time"),
    ))


@app.get("/api/sector-rotation/backtest")
def sector_rotation_backtest(
    lookback_years: int = 5,
    top_n: int = 3,
    turnover_bp: float = 30.0,
    signal_mode: str = "momentum_12_1m",
    vol_target_pct: float = 0.0,
    vol_lookback_months: int = 6,
    max_leverage: float = 1.0,
):
    """US Sector Rotation Backtest — monthly rebalance.

    signal_mode:
      • "momentum_12_1m" (Phase 3) — Jegadeesh-Titman 12-1M momentum
      • "composite_live" (Phase 4) — Composite-equivalent score (TCS/TFS/RSS/URS)
      • "ml_momentum_blend" (Phase 5 B-1) — walk-forward fit

    vol_target_pct (B-3): 0 = disabled. >0 sets annualized vol target (%).
                          Cash slack (1−scale) earns 0%; cap at max_leverage.

    Cached 24h per full param tuple.
    """
    import time
    if signal_mode not in ("momentum_12_1m", "composite_live", "ml_momentum_blend", "ml_lightgbm"):
        return {"error": f"Unknown signal_mode '{signal_mode}'."}
    cache_key = (
        f"sr_bt_{lookback_years}_{top_n}_{turnover_bp}_{signal_mode}"
        f"_vt{vol_target_pct}_vl{vol_lookback_months}_lev{max_leverage}"
    )
    cache_ttl = 24 * 3600  # 24 hours

    cached = STATE.get("_backtest_cache", {}).get(cache_key)
    if cached and (time.time() - cached["fetched_at"]) < cache_ttl:
        return _clean_dict(cached["data"])

    from sector_rotation_backtest import run_backtest
    out = run_backtest(
        lookback_years=lookback_years,
        top_n=top_n,
        turnover_bp=turnover_bp,
        signal_mode=signal_mode,
        vol_target_pct=vol_target_pct,
        vol_lookback_months=vol_lookback_months,
        max_leverage=max_leverage,
    )

    # Persist in memory cache
    STATE.setdefault("_backtest_cache", {})[cache_key] = {
        "fetched_at": time.time(),
        "data": out,
    }
    return _clean_dict(out)


@app.get("/api/classification-history")
def classification_history():
    """Classification distribution at ~2-week intervals over the past year (from ve_observations) + current scan."""
    ve_obs = STATE.get("ve_stats", {}).get("observations", [])
    results = STATE.get("results", [])
    if not ve_obs and not results:
        return {"dates": [], "classifications": [], "matrix": []}

    from collections import defaultdict as dd

    # 1. Aggregate ve_observations by eval_date
    date_class_counts: dict = dd(lambda: dd(int))
    for o in ve_obs:
        if not isinstance(o, dict):
            continue
        dt = o.get("eval_date", "")
        cls = o.get("classification", "")
        if dt and cls:
            date_class_counts[dt][cls] += 1

    # 2. Add current scan as latest data point
    if results:
        scan_time = STATE.get("scan_time", "")[:10] or "current"
        for r in results:
            cls = r.get("classification", "")
            if cls:
                date_class_counts[scan_time][cls] += 1

    # 3. Sort dates, keep last 6 (roughly 3 months of bi-weekly data)
    sorted_dates = sorted(date_class_counts.keys())[-6:]

    # 4. Collect all classifications, order by current count desc
    all_classes = set()
    for counts in date_class_counts.values():
        all_classes.update(counts.keys())
    # Sort by count in the latest date
    latest = sorted_dates[-1] if sorted_dates else ""
    all_classes = sorted(all_classes, key=lambda c: -date_class_counts[latest].get(c, 0))

    # 5. Build matrix: rows = classifications, cols = dates
    matrix = []
    for cls in all_classes:
        row = [date_class_counts[d].get(cls, 0) for d in sorted_dates]
        matrix.append(row)

    return _clean_dict({
        "dates": sorted_dates,
        "classifications": list(all_classes),
        "matrix": matrix,
    })


@app.get("/api/classification-history-by-sector")
def classification_history_by_sector():
    """Classification distribution per sector over time (bi-weekly from ve_observations)."""
    ve_obs = STATE.get("ve_stats", {}).get("observations", [])
    results = STATE.get("results", [])
    if not results:
        return {"sectors": []}

    from collections import defaultdict as dd

    # Build ticker → category mapping from current results
    ticker_cat = {r.get("ticker", ""): r.get("category", "Unknown") for r in results}

    # Aggregate: sector → date → classification → count
    sector_date_cls: dict = dd(lambda: dd(lambda: dd(int)))

    for o in ve_obs:
        if not isinstance(o, dict):
            continue
        tk = o.get("ticker", "")
        dt = o.get("eval_date", "")
        cls = o.get("classification", "")
        cat = ticker_cat.get(tk, "")
        if dt and cls and cat:
            sector_date_cls[cat][dt][cls] += 1

    # Add current scan
    scan_time = STATE.get("scan_time", "")[:10] or "current"
    for r in results:
        cat = r.get("category", "")
        cls = r.get("classification", "")
        if cat and cls:
            sector_date_cls[cat][scan_time][cls] += 1

    # Build per-sector output (last 6 dates)
    all_dates = set()
    for cat_data in sector_date_cls.values():
        all_dates.update(cat_data.keys())
    sorted_dates = sorted(all_dates)[-6:]

    # Standard classification order
    std_classes = [
        "🟢 CONTINUATION", "🔵 FORMATION", "🔵 RECOVERY", "🟡 CONSOLIDATION",
        "🟠 NEUTRAL", "🟤 FADING", "🔶 PULLBACK", "⚠️ WEAKENING",
        "⬇️ DOWNTREND", "🟡 OVEREXTENDED", "🟤 EXHAUSTING", "🟣 COUNTER_RALLY",
        "🔴 CYCLE_PEAK",
    ]

    sector_output = []
    for cat in sorted(sector_date_cls.keys()):
        cat_data = sector_date_cls[cat]
        # Only include classifications that appear in this sector
        present_classes = set()
        for dt_counts in cat_data.values():
            present_classes.update(dt_counts.keys())
        classes = [c for c in std_classes if c in present_classes]

        matrix = []
        for cls in classes:
            row = [cat_data.get(d, {}).get(cls, 0) for d in sorted_dates]
            matrix.append(row)

        total_latest = sum(cat_data.get(sorted_dates[-1], {}).values()) if sorted_dates else 0
        sector_output.append({
            "sector": cat,
            "total": total_latest,
            "dates": sorted_dates,
            "classifications": classes,
            "matrix": matrix,
        })

    # Sort by total desc
    sector_output.sort(key=lambda x: -x["total"])

    return _clean_dict({"sectors": sector_output})


# ─────────────────────────────────────────────────────────────────────────────
# Classification Efficacy Validation
# ─────────────────────────────────────────────────────────────────────────────

# Hypothesis specifications per classification
# Each entry: {metric_key: (lower_bound, upper_bound or None, "direction")}
# direction: "min" (must be ≥ lower), "max" (must be ≤ upper), "range" (must be in [lower, upper])
_MOMENTUM_HYPOTHESES = {
    "🟢 CONTINUATION": {
        "ret_1m_mean": {"min": 3.0, "ideal": 6.0, "label": "1M Mean Return ≥ 3%"},
        "ret_1m_hit": {"min": 60.0, "ideal": 70.0, "label": "1M Hit Rate ≥ 60%"},
        "persistence_1m": {"min": 65.0, "ideal": 80.0, "label": "1M Persistence ≥ 65%"},
        "max_dd_p95": {"max": -10.0, "ideal": -5.0, "label": "P95 Max DD ≥ -10%"},
    },
    "🔵 FORMATION": {
        "ret_1m_mean": {"min": 4.0, "ideal": 8.0, "label": "1M Mean Return ≥ 4%"},
        "ret_1m_hit": {"min": 55.0, "ideal": 65.0, "label": "1M Hit Rate ≥ 55%"},
        "persistence_1m": {"min": 50.0, "ideal": 65.0, "label": "1M Persistence ≥ 50%"},
        "max_dd_p95": {"max": -12.0, "ideal": -7.0, "label": "P95 Max DD ≥ -12%"},
    },
    "🟦 LAGGING_CATCHUP": {
        "ret_1m_mean": {"min": 5.0, "ideal": 10.0, "label": "1M Mean Return ≥ 5%"},
        "ret_1m_hit": {"min": 60.0, "ideal": 70.0, "label": "1M Hit Rate ≥ 60%"},
        "persistence_1m": {"min": 50.0, "ideal": 60.0, "label": "1M Persistence ≥ 50%"},
        "max_dd_p95": {"max": -10.0, "ideal": -5.0, "label": "P95 Max DD ≥ -10%"},
    },
    "🟡 OVEREXTENDED": {
        "ret_1m_mean": {"range": [0.0, 2.0], "ideal_range": [0.5, 1.5], "label": "1M Mean Return: 0~2% (caution)"},
        "ret_1m_hit": {"range": [45.0, 55.0], "ideal_range": [48.0, 52.0], "label": "1M Hit Rate: 45-55% (uncertain)"},
        "max_dd_p95": {"max": -15.0, "ideal": -10.0, "label": "P95 Max DD ≥ -15% (mean reversion risk)"},
    },
}

_PRE_MOMENTUM_HYPOTHESES = {
    "🔵 RECOVERY": {
        "conv_3m": {"min": 35.0, "ideal": 50.0, "label": "3M Conversion ≥ 35%"},
        "fail_rate": {"max": 20.0, "ideal": 10.0, "label": "Failure Rate ≤ 20%"},
        "median_time_days": {"max": 45.0, "ideal": 30.0, "label": "Median Time ≤ 45d"},
    },
    "🟡 CONSOLIDATION": {
        "conv_3m": {"min": 30.0, "ideal": 45.0, "label": "3M Conversion ≥ 30%"},
        "fail_rate": {"max": 20.0, "ideal": 10.0, "label": "Failure Rate ≤ 20%"},
        "median_time_days": {"max": 60.0, "ideal": 45.0, "label": "Median Time ≤ 60d"},
    },
    "🔶 PULLBACK": {
        "conv_3m": {"min": 30.0, "ideal": 45.0, "label": "3M Conversion ≥ 30%"},
        "fail_rate": {"max": 25.0, "ideal": 15.0, "label": "Failure Rate ≤ 25%"},
        "median_time_days": {"max": 45.0, "ideal": 30.0, "label": "Median Time ≤ 45d"},
    },
    "🟠 NEUTRAL": {
        "conv_3m": {"range": [20.0, 35.0], "ideal_range": [22.0, 30.0], "label": "3M Conversion: 20-35%"},
        "fail_rate": {"max": 30.0, "ideal": 20.0, "label": "Failure Rate ≤ 30%"},
        "median_time_days": {"max": 90.0, "ideal": 60.0, "label": "Median Time ≤ 90d"},
    },
    "⚠️ WEAKENING": {
        "conv_3m": {"max": 15.0, "ideal_max": 10.0, "label": "3M Conversion ≤ 15% (weak)"},
        "fail_rate": {"min": 40.0, "ideal": 50.0, "label": "Failure Rate ≥ 40%"},
    },
    "🟤 FADING": {
        "conv_3m": {"max": 15.0, "ideal_max": 10.0, "label": "3M Conversion ≤ 15% (weak)"},
        "fail_rate": {"min": 40.0, "ideal": 50.0, "label": "Failure Rate ≥ 40%"},
    },
}


def _eval_hypothesis(actual: float, spec: dict) -> str:
    """Returns 'PASS_IDEAL', 'PASS', or 'FAIL'."""
    if actual is None:
        return "INSUFFICIENT_DATA"
    # Range-based
    if "range" in spec:
        lo, hi = spec["range"]
        ideal_lo, ideal_hi = spec.get("ideal_range", spec["range"])
        if ideal_lo <= actual <= ideal_hi:
            return "PASS_IDEAL"
        if lo <= actual <= hi:
            return "PASS"
        return "FAIL"
    # Min-based
    if "min" in spec:
        if "ideal" in spec and actual >= spec["ideal"]:
            return "PASS_IDEAL"
        if actual >= spec["min"]:
            return "PASS"
        return "FAIL"
    # Max-based
    if "max" in spec:
        ideal = spec.get("ideal", spec.get("ideal_max"))
        if ideal is not None and actual <= ideal:
            return "PASS_IDEAL"
        if actual <= spec["max"]:
            return "PASS"
        return "FAIL"
    return "FAIL"


def _compute_momentum_actuals(ve_obs: list, target_class: str, observations_by_ticker: dict) -> dict:
    """Compute actual metrics for a momentum class from ve_observations."""
    obs_in_class = [o for o in ve_obs if isinstance(o, dict) and o.get("classification") == target_class]
    n = len(obs_in_class)
    if n < 5:
        return {"n": n, "ret_1m_mean": None, "ret_1m_median": None, "ret_1m_hit": None,
                "ret_3m_mean": None, "max_dd_p95": None, "max_dd_p50": None,
                "persistence_1m": None, "transitions": {}}

    # Forward returns at 21 (1M) and 63 (3M)
    rets_1m = []
    rets_3m = []
    max_dds = []
    for o in obs_in_class:
        fwd = o.get("fwd_rets", {}) or {}
        r1m = fwd.get("21") or fwd.get(21)
        r3m = fwd.get("63") or fwd.get(63)
        if r1m is not None:
            rets_1m.append(float(r1m))
        if r3m is not None:
            rets_3m.append(float(r3m))
        # Max DD from fwd_daily 21d window
        fwd_daily = (o.get("fwd_daily", {}) or {}).get("21") or (o.get("fwd_daily", {}) or {}).get(21) or []
        if fwd_daily:
            cumulative = []
            running = 1.0
            for d in fwd_daily:
                running *= (1.0 + float(d))
                cumulative.append((running - 1.0) * 100)
            if cumulative:
                # Max DD from any peak to subsequent trough
                peak = cumulative[0]
                max_dd = 0.0
                for v in cumulative:
                    peak = max(peak, v)
                    dd = v - peak
                    if dd < max_dd:
                        max_dd = dd
                max_dds.append(max_dd)

    def _percentile(arr, p):
        if not arr:
            return None
        s = sorted(arr)
        idx = int(p / 100.0 * (len(s) - 1))
        return s[idx]

    def _mean(arr):
        return sum(arr) / len(arr) if arr else None

    def _hit_rate(arr, threshold=0.0):
        if not arr:
            return None
        return sum(1 for v in arr if v > threshold) / len(arr) * 100

    # Persistence: same ticker's class at next eval point
    persistence_count = 0
    transitions_count = 0
    transitions = {}
    obs_by_tkdate = {}
    for o in ve_obs:
        if not isinstance(o, dict):
            continue
        tk = o.get("ticker", "")
        dt = o.get("eval_date", "")
        if tk and dt:
            obs_by_tkdate[(tk, dt)] = o

    # Sort eval dates per ticker
    ticker_dates = {}
    for (tk, dt) in obs_by_tkdate.keys():
        ticker_dates.setdefault(tk, []).append(dt)
    for tk in ticker_dates:
        ticker_dates[tk].sort()

    for o in obs_in_class:
        tk = o.get("ticker", "")
        dt = o.get("eval_date", "")
        if tk in ticker_dates:
            dates = ticker_dates[tk]
            if dt in dates:
                idx = dates.index(dt)
                if idx + 1 < len(dates):
                    next_dt = dates[idx + 1]
                    next_o = obs_by_tkdate.get((tk, next_dt))
                    if next_o:
                        next_cls = next_o.get("classification", "")
                        transitions_count += 1
                        transitions[next_cls] = transitions.get(next_cls, 0) + 1
                        if next_cls == target_class:
                            persistence_count += 1

    persistence = (persistence_count / transitions_count * 100) if transitions_count > 0 else None
    transition_pcts = {k: round(v / transitions_count * 100, 1) for k, v in transitions.items()} if transitions_count else {}

    return {
        "n": n,
        "ret_1m_mean": round(_mean(rets_1m), 2) if rets_1m else None,
        "ret_1m_median": round(_percentile(rets_1m, 50), 2) if rets_1m else None,
        "ret_1m_hit": round(_hit_rate(rets_1m), 1) if rets_1m else None,
        "ret_3m_mean": round(_mean(rets_3m), 2) if rets_3m else None,
        "ret_3m_hit": round(_hit_rate(rets_3m), 1) if rets_3m else None,
        "max_dd_p50": round(_percentile(max_dds, 50), 2) if max_dds else None,
        "max_dd_p95": round(_percentile(max_dds, 5), 2) if max_dds else None,  # 5th percentile = worst-case
        "persistence_1m": round(persistence, 1) if persistence is not None else None,
        "transitions": dict(sorted(transition_pcts.items(), key=lambda x: -x[1])[:5]),
    }


def _compute_premomentum_actuals(ve_obs: list, target_class: str) -> dict:
    """Compute conversion metrics for pre-momentum class."""
    pm_classes = {"🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔵 RECOVERY", "🔶 PULLBACK", "⚠️ WEAKENING", "🟤 FADING"}
    momentum_classes = {"🟢 CONTINUATION", "🔵 FORMATION", "🟡 OVEREXTENDED"}
    fail_classes = {"⬇️ DOWNTREND", "🔴 CYCLE_PEAK", "🟣 COUNTER_RALLY"}

    # Cohort: observations in target class with eligible=False
    cohort = [o for o in ve_obs if isinstance(o, dict)
              and o.get("classification") == target_class
              and not o.get("eligible", False)]
    n = len(cohort)
    if n < 5:
        return {"n": n, "conv_3m": None, "fail_rate": None, "median_time_days": None}

    # For each cohort member, find next obs of same ticker → check transition
    obs_by_tkdate = {}
    ticker_obs = {}
    for o in ve_obs:
        if not isinstance(o, dict):
            continue
        tk = o.get("ticker", "")
        dt = o.get("eval_date", "")
        if tk and dt:
            obs_by_tkdate[(tk, dt)] = o
            ticker_obs.setdefault(tk, []).append((dt, o))

    for tk in ticker_obs:
        ticker_obs[tk].sort(key=lambda x: x[0])

    from datetime import date as _date

    converted_within_3m = 0
    failed = 0
    convert_times = []
    eligible_outcomes = 0  # for failure rate denominator

    for o in cohort:
        tk = o.get("ticker", "")
        dt = o.get("eval_date", "")
        if not tk or not dt:
            continue
        history = ticker_obs.get(tk, [])
        # Find this obs index
        try:
            idx = next(i for i, (d, _) in enumerate(history) if d == dt)
        except StopIteration:
            continue

        # Walk forward up to ~6 obs (3 months at bi-weekly)
        try:
            base_d = _date.fromisoformat(dt)
        except (ValueError, TypeError):
            continue

        outcome = None
        for j in range(idx + 1, min(idx + 7, len(history))):
            future_dt, future_o = history[j]
            future_cls = future_o.get("classification", "")
            future_eligible = future_o.get("eligible", False)

            try:
                fd = _date.fromisoformat(future_dt)
                days = (fd - base_d).days
            except (ValueError, TypeError):
                continue

            if days > 95:  # outside 3M window
                break

            if future_eligible and future_cls in momentum_classes:
                outcome = "graduated"
                convert_times.append(days)
                break
            if future_cls in fail_classes:
                outcome = "failed"
                break

        if outcome == "graduated":
            converted_within_3m += 1
            eligible_outcomes += 1
        elif outcome == "failed":
            failed += 1
            eligible_outcomes += 1

    conv_3m = (converted_within_3m / n * 100) if n > 0 else None
    fail_rate = (failed / n * 100) if n > 0 else None
    median_time = sorted(convert_times)[len(convert_times) // 2] if convert_times else None

    return {
        "n": n,
        "conv_3m": round(conv_3m, 1) if conv_3m is not None else None,
        "fail_rate": round(fail_rate, 1) if fail_rate is not None else None,
        "median_time_days": median_time,
        "graduated": converted_within_3m,
        "failed": failed,
    }


@app.get("/api/validation")
def classification_validation():
    """Validate each classification's actual metrics against expected hypotheses."""
    ve_obs = STATE.get("ve_stats", {}).get("observations", [])
    if not ve_obs:
        return {"error": "No ve_observations data available"}

    momentum_results = []
    for cls, hyps in _MOMENTUM_HYPOTHESES.items():
        actuals = _compute_momentum_actuals(ve_obs, cls, {})
        checks = []
        pass_count = 0
        ideal_count = 0
        total_checks = 0
        for metric_key, spec in hyps.items():
            actual_val = actuals.get(metric_key)
            verdict = _eval_hypothesis(actual_val, spec)
            checks.append({
                "metric": metric_key,
                "label": spec.get("label", metric_key),
                "expected": spec,
                "actual": actual_val,
                "verdict": verdict,
            })
            if verdict == "PASS_IDEAL":
                ideal_count += 1
                pass_count += 1
                total_checks += 1
            elif verdict == "PASS":
                pass_count += 1
                total_checks += 1
            elif verdict == "FAIL":
                total_checks += 1
        score = round(pass_count / total_checks * 100, 1) if total_checks > 0 else 0
        overall = "PASS" if score >= 75 else "PARTIAL" if score >= 50 else "FAIL"
        momentum_results.append({
            "classification": cls,
            "actuals": actuals,
            "checks": checks,
            "pass_score": score,
            "ideal_count": ideal_count,
            "pass_count": pass_count,
            "total_checks": total_checks,
            "overall": overall,
        })

    pre_momentum_results = []
    for cls, hyps in _PRE_MOMENTUM_HYPOTHESES.items():
        actuals = _compute_premomentum_actuals(ve_obs, cls)
        checks = []
        pass_count = 0
        ideal_count = 0
        total_checks = 0
        for metric_key, spec in hyps.items():
            actual_val = actuals.get(metric_key)
            verdict = _eval_hypothesis(actual_val, spec)
            checks.append({
                "metric": metric_key,
                "label": spec.get("label", metric_key),
                "expected": spec,
                "actual": actual_val,
                "verdict": verdict,
            })
            if verdict == "PASS_IDEAL":
                ideal_count += 1
                pass_count += 1
                total_checks += 1
            elif verdict == "PASS":
                pass_count += 1
                total_checks += 1
            elif verdict == "FAIL":
                total_checks += 1
        score = round(pass_count / total_checks * 100, 1) if total_checks > 0 else 0
        overall = "PASS" if score >= 75 else "PARTIAL" if score >= 50 else "FAIL"
        pre_momentum_results.append({
            "classification": cls,
            "actuals": actuals,
            "checks": checks,
            "pass_score": score,
            "ideal_count": ideal_count,
            "pass_count": pass_count,
            "total_checks": total_checks,
            "overall": overall,
        })

    n_total_obs = len(ve_obs)
    eval_dates = sorted(set(o.get("eval_date", "") for o in ve_obs if isinstance(o, dict)))
    n_eval_points = len(eval_dates)

    return _clean_dict({
        "summary": {
            "total_observations": n_total_obs,
            "eval_points": n_eval_points,
            "date_range": [eval_dates[0] if eval_dates else "", eval_dates[-1] if eval_dates else ""],
        },
        "momentum": momentum_results,
        "pre_momentum": pre_momentum_results,
    })


# ─────────────────────────────────────────────────────────────────────────────
# AI Prediction (P1–P4 regime classifier results)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/ai-prediction")
def ai_prediction():
    """Load cached artifacts produced by ai_prediction_cache.py."""
    import json as _json
    base = os.path.dirname(os.path.abspath(__file__))

    def _load_csv(fname):
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        # Index column is unnamed (previous save) — rename for clients
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "Date":
            df = df.rename(columns={df.columns[0]: "date"})
        return df

    metrics_path = os.path.join(base, "ai_pred_metrics.json")
    if not os.path.exists(metrics_path):
        return {
            "error": "AI prediction cache missing. Run `python3 ai_prediction_cache.py`.",
            "metrics": None,
        }
    with open(metrics_path) as f:
        metrics = _json.load(f)

    proba = _load_csv("ai_pred_proba.csv")
    returns = _load_csv("ai_pred_returns.csv")
    fi = _load_csv("ai_pred_feature_imp.csv")
    cm = _load_csv("ai_pred_confusion.csv")
    ablation = _load_csv("ablation_results.csv")

    # Normalize
    def _records(df):
        if df is None:
            return []
        return df.where(pd.notna(df), None).to_dict("records")

    # Feature importance: the CSV has columns like [feature_name, importance] -
    # first col is the feature name index
    fi_records = []
    if fi is not None:
        col_names = list(fi.columns)
        feat_col = col_names[0]
        imp_col = col_names[1] if len(col_names) > 1 else "importance"
        for _, row in fi.iterrows():
            fi_records.append({
                "feature": str(row[feat_col]),
                "importance": float(row[imp_col]) if pd.notna(row[imp_col]) else 0.0,
            })

    # Confusion matrix: rows=true, cols=pred
    cm_records = []
    if cm is not None:
        for _, row in cm.iterrows():
            label = str(row.iloc[0])  # e.g. "true_BEAR"
            cm_records.append({
                "label": label,
                "BEAR": int(row["pred_BEAR"]) if "pred_BEAR" in row else 0,
                "BASE": int(row["pred_BASE"]) if "pred_BASE" in row else 0,
                "BULL": int(row["pred_BULL"]) if "pred_BULL" in row else 0,
            })

    # ── Optional: MoE (Plan B) comparison ──
    moe_summary = None
    moe_monthly = None
    moe_json_path = os.path.join(base, "ai_pred_moe.json")
    if os.path.exists(moe_json_path):
        with open(moe_json_path) as f:
            moe_summary = _json.load(f)
        moe_df = _load_csv("ai_pred_moe.csv")
        moe_monthly = _records(moe_df)

    return _clean_dict({
        "metrics": metrics,
        "proba": _records(proba),
        "returns": _records(returns),
        "feature_importance": fi_records,
        "confusion_matrix": cm_records,
        "ablation": _records(ablation),
        "moe_summary": moe_summary,
        "moe_monthly": moe_monthly,
    })


@app.get("/api/ai-winratio")
def ai_winratio():
    """Per-signal win ratio for P0~P4 (precision + directional hit rate)."""
    import json as _json
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "ai_pred_winratio.json")
    if not os.path.exists(path):
        return {"error": "Win ratio cache missing. Run `python3 signal_win_ratio.py`."}
    with open(path) as f:
        return _clean_dict(_json.load(f))


@app.get("/api/ai-benchmarks")
def ai_benchmarks():
    """Multi-benchmark validation suite — P4 vs 5 reference portfolios."""
    import json as _json
    base = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(base, "ai_pred_benchmarks.json")
    monthly_path = os.path.join(base, "ai_pred_benchmarks_monthly.csv")
    if not os.path.exists(summary_path):
        return {
            "error": "Multi-benchmark cache missing. "
                     "Run `python3 multi_benchmark_validation.py`."
        }
    with open(summary_path) as f:
        summary = _json.load(f)
    monthly = pd.read_csv(monthly_path) if os.path.exists(monthly_path) else pd.DataFrame()
    return _clean_dict({
        "summary": summary,
        "monthly": monthly.where(pd.notna(monthly), None).to_dict("records") if not monthly.empty else [],
    })


@app.get("/api/ai-performance")
def ai_performance():
    """Comprehensive performance analytics for P0~P4 (institutional-style metrics)."""
    import json as _json
    base = os.path.dirname(os.path.abspath(__file__))

    summary_path = os.path.join(base, "ai_perf_summary.json")
    monthly_path = os.path.join(base, "ai_perf_monthly.csv")
    rolling_path = os.path.join(base, "ai_perf_rolling.csv")

    if not os.path.exists(summary_path):
        return {
            "error": "Performance analytics cache missing. "
                     "Run `python3 performance_analytics.py`.",
        }
    with open(summary_path) as f:
        summary = _json.load(f)

    monthly = pd.read_csv(monthly_path) if os.path.exists(monthly_path) else pd.DataFrame()
    rolling = pd.read_csv(rolling_path) if os.path.exists(rolling_path) else pd.DataFrame()

    def _records(df):
        if df is None or df.empty:
            return []
        return df.where(pd.notna(df), None).to_dict("records")

    return _clean_dict({
        "summary": summary,
        "monthly": _records(monthly),
        "rolling": _records(rolling),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Academic References — PDF serving
# ─────────────────────────────────────────────────────────────────────────────

_REFERENCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", "references")


@app.get("/api/references")
def list_references():
    """List all reference PDFs with metadata grouped by category."""
    # Try to import metadata from the bibliography helper
    try:
        sys.path.insert(0, _REFERENCES_DIR)
        # Import metadata
        from _make_bibliography import REFERENCES, UNDOWNLOADED_REFERENCES
        # Build response
        downloaded = []
        for r in REFERENCES:
            fname = r.get("filename", "")
            fpath = os.path.join(_REFERENCES_DIR, fname)
            exists = os.path.isfile(fpath)
            size_kb = round(os.path.getsize(fpath) / 1024) if exists else 0
            downloaded.append({
                "id": r.get("id", ""),
                "filename": fname,
                "title": r.get("title", ""),
                "authors": r.get("authors", ""),
                "year": r.get("year", ""),
                "venue": r.get("venue", ""),
                "applies_to": r.get("applies_to", []),
                "category": r.get("category", "Other"),
                "available": exists,
                "size_kb": size_kb,
            })
        bibliography = {
            "filename": "00_Bibliography_PriceDiscovery_References.pdf",
            "available": os.path.isfile(os.path.join(
                _REFERENCES_DIR, "00_Bibliography_PriceDiscovery_References.pdf"
            )),
        }
        return _clean_dict({
            "downloaded": downloaded,
            "citation_only": UNDOWNLOADED_REFERENCES,
            "bibliography": bibliography,
        })
    except Exception as e:
        return {"error": f"Could not load reference metadata: {e}"}


@app.get("/api/references/file/{filename}")
def serve_reference(filename: str):
    """Serve a reference PDF file."""
    # Security: only allow PDF/DOCX in references folder, no path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF/DOCX files allowed")

    fpath = os.path.join(_REFERENCES_DIR, filename)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    media_type = "application/pdf" if filename.lower().endswith(".pdf") \
                 else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return FileResponse(fpath, media_type=media_type, filename=filename)
