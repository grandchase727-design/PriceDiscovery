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
from typing import List, Optional, Dict

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
    # ── International — Broad multi-country (sector-diversified, 그대로 International) ──
    "Developed Markets": "International", "EM Broad": "International",
    "Europe Broad": "International",
    # ── International — Country-level → GICS dominant sector ──
    # 단일국가 ETF를 dominant sector로 매핑하여 Cyclical/Defensive 회전 매트릭스 참여 가능.
    # (theme 필드는 국가명 그대로 유지 — granularity 보존)
    #
    # Europe (Developed)
    "Europe - Germany": "Industrials",          # Siemens / BMW / VW / BASF
    "Europe - UK": "Energy",                    # Shell / BP dominant
    "Europe - France": "Consumer Discretionary",# LVMH / Hermes / L'Oreal (luxury)
    "Europe - Switzerland": "Healthcare",       # Roche / Novartis / Nestle
    "Europe - Spain": "Financials",             # Santander / BBVA / Telefonica
    "Europe - Italy": "Financials",             # Banks + Enel
    # Europe (EM)
    "Europe - Poland": "Financials",            # banks dominant
    "Europe - Greece": "Financials",            # Eurobank / NBG
    # Asia / Pacific
    "Japan": "Industrials",                     # Toyota / Sony / Honda heavy industrial mix
    "Korea (Index)": "Technology",              # Samsung / SK Hynix tech dominance
    "China": "Communication Services",          # Tencent / Alibaba / Meituan
    "India": "Financials",                      # HDFC / Reliance financials
    "Asia Pacific - Australia": "Materials",    # BHP / Rio Tinto
    "Asia Pacific - Taiwan": "Technology",      # TSMC dominance
    "Asia Pacific - Hong Kong": "Financials",   # HSBC / AIA
    "Asia Pacific - Singapore": "Financials",   # DBS / OCBC / UOB
    "Asia Pacific - Thailand": "Financials",    # bank-heavy
    "Asia Pacific - Vietnam": "Financials",
    "Asia Pacific - Indonesia": "Financials",
    # Latin America
    "Latin America - Brazil": "Materials",      # Vale + Petrobras
    "Latin America - Mexico": "Consumer Staples",  # FEMSA + Walmex
    "Latin America - Colombia": "Financials",
    "Latin America - Chile": "Materials",       # copper
    # EMEA
    "EMEA - Turkey": "Financials",
    "EMEA - South Africa": "Materials",         # mining heavy
    "EMEA - Egypt": "Financials",
    # Other
    "Middle East - Israel": "Technology",       # Check Point / NICE / CyberArk
    "North America - Canada": "Financials",     # RBC / TD / Scotia (banks dominant)
    # ── Backward-compat fallbacks (이전 theme명 호환, 사용 안 되지만 안전망) ──
    "Europe": "International", "Asia Pacific": "International",
    "North America": "International", "Middle East": "International",
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

    # ── Option C — strict Pre-Momentum / Momentum separation ──
    # Demote ambiguous classifications (NEUTRAL/CONSOLIDATION/RECOVERY/PULLBACK)
    # from Momentum-eligible. These belong to Pre-Momentum stage by definition.
    # Runs UNCONDITIONALLY (before QVR gate) so behavior is consistent whether
    # or not the fundamentals cache is available.
    PM_ONLY_CLASSES = {
        "🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔵 RECOVERY", "🔶 PULLBACK",
    }
    if "eligible" in df.columns and "classification" in df.columns:
        pm_mask = df["eligible"].fillna(False) & df["classification"].isin(PM_ONLY_CLASSES)
        n_pm_demoted = int(pm_mask.sum())
        if n_pm_demoted > 0:
            df.loc[pm_mask, "eligible"] = False

            def _add_pm_reject(row):
                base = row.get("rejection") or ""
                # E.g. "🟠 NEUTRAL" → "NEUTRAL(PM)"
                cls_text = (row.get("classification") or "").split(" ", 1)[-1] or "Ambig"
                tag = f"{cls_text}(PM)"
                if base in ("", "None"):
                    return tag
                return f"{base}/{tag}"
            df.loc[pm_mask, "rejection"] = df.loc[pm_mask].apply(_add_pm_reject, axis=1)
            print(f"[pm-gate] demoted {n_pm_demoted} ambiguous-classification tickers from Momentum (Option C)")

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

            # ── QVR Eligibility Gate ──
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
                    "gics_industry_group": c.get("gics_industry_group"),
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

    # ── Category normalization (post-Phase Y) ──
    # 1. Drop "STK_" prefix from individual stock categories (display parity with ETFs)
    # 2. Re-map international stocks (STK_Korea/Japan/China_ADR/Europe/India) to
    #    the SAME sector categories as US stocks via GICS lookup, so a Korean Tech
    #    stock (e.g., 005930.KS Samsung) sits in "Technology" alongside NVDA/AAPL.
    _normalize_categories(STATE["df"], STATE.get("unified_classification", {}))

    # ── Phase 1: Macro-context tags (cyclical/style/region) + Phase 1.5 industry refinement ──
    # 종목별 거시 맥락을 부착하여 downstream(SVE 분해, Pre-Mom, Hedge strategy 등)이 활용.
    _apply_macro_context_tags(STATE["df"])

    # ── Phase 1G: Inject macro tags into ve_observations (for regime-segmented hit rates) ──
    _inject_tags_into_observations(STATE["df"], STATE.get("ve_stats", {}).get("observations", []))

    # ── Phase 1.0 also: Inject tags into STATE["results"] (used by Pre-Mom agent) ──
    _inject_tags_into_results(STATE["df"], STATE.get("results", []))

    # ── Phase 2D + 3B: Cross-sectional regime detection + per-ticker rotation scores + ──
    # ──                  regime-aware CYCLE_PEAK upgrade for over-extended misaligned tickers ──
    REGIME = _detect_market_regime(STATE["df"])
    STATE["regime"] = REGIME
    _compute_rotation_scores(STATE["df"], REGIME)
    _phase3b_regime_classify_override(STATE["df"], REGIME)

    # ── Hybrid Phase A + B: ETF bottom-up sidecar metrics + divergence flags ──
    _compute_etf_hybrid_sidecar(STATE["df"])

    # ── Hybrid Phase D (Pre-Mom integration): per-ticker parent_etf_signal ──
    # 각 stock에 대해 "어떤 ETF의 top holding인가" 역색인 후
    # 부모 ETF들의 divergence_flag를 가중 평균하여 forward-looking signal 산출.
    _compute_parent_etf_signal(STATE["df"], STATE.get("results", []))

    # ── Anti-Lag Phase 1: Pre-Momentum direct entry (PROVISIONAL eligibility) ──
    # Pre-Mom Score ≥ 70 + agreement_ratio ≥ 0.6 + bullish PM classification 종목을
    # Momentum 탭에 PROVISIONAL 태그로 surface — Lag 10-15일 단축 효과.
    _compute_provisional_eligibility(STATE["df"], STATE.get("results", []),
                                       STATE.get("graph"), STATE.get("history"),
                                       STATE.get("ve_stats", {}).get("observations", []))

    # ── Sector-Segmented Price Discovery (New2) ──
    # 섹터별로 독립적으로 top-N 종목 선별. universe-wide ranking과 별개로
    # 각 섹터 내 best-in-sector를 강제로 surface. Lag보단 diversification 효과.
    _compute_sector_segmented_picks(STATE["df"], top_per_sector=5, min_composite=40.0)

    # ── YTD return enrichment ──
    # Cache may not have ret_ytd (legacy scans). Load .ytd_returns.json
    # (produced by compute_ytd.py) and merge into df, preferring values from
    # the scan cache when present.
    ytd_path = ".ytd_returns.json"
    df = STATE["df"]
    if "ret_ytd" not in df.columns:
        df["ret_ytd"] = None
    if os.path.exists(ytd_path):
        try:
            with open(ytd_path) as f:
                yj = json.load(f)
            ytd_map = yj.get("tickers", {})
            mask = df["ret_ytd"].isna()
            df.loc[mask, "ret_ytd"] = df.loc[mask, "ticker"].map(ytd_map)
            n_filled = int((~df["ret_ytd"].isna()).sum())
            print(f"[ytd] loaded YTD returns · {n_filled}/{len(df)} tickers populated")
        except Exception as _ytd_err:
            print(f"[ytd] load failed: {_ytd_err}")
    STATE["df"] = df

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


# ───────────────────────────────────────────────────────────────
# Phase 1: Macro-context tags (cyclical/style/region/industry refinement)
# ───────────────────────────────────────────────────────────────

CYCLICAL_SECTORS_SET = {
    "Technology", "Communication Services", "Consumer Discretionary",
    "Industrials", "Materials", "Energy", "Financials", "Real Estate",
}
DEFENSIVE_SECTORS_SET = {"Consumer Staples", "Utilities", "Healthcare"}
GROWTH_SECTORS_SET = {"Technology", "Communication Services", "Consumer Discretionary"}
VALUE_SECTORS_SET = {
    "Financials", "Energy", "Materials", "Utilities", "Real Estate", "Consumer Staples",
}


def _detect_region(ticker: str, theme: str) -> str:
    """Frontend `detectRegion`과 동일 매커니즘 — ticker suffix + theme prefix matching."""
    t = ticker or ""
    th = theme or ""
    if t.endswith(".KS"):
        return "Korea"
    if t.endswith(".T"):
        return "Japan"
    if th.startswith("Europe -") or th == "Europe Broad" or th == "Europe":
        return "Europe"
    if th == "Japan":
        return "Japan"
    if th == "Korea (Index)":
        return "Korea"
    if th == "China":
        return "China"
    if th == "India" or th.startswith("Asia Pacific -") or th == "Asia Pacific":
        return "Other Asia"
    if th.startswith("Latin America"):
        return "LatAm"
    if th.startswith("EMEA") or th == "Other EM" or th.startswith("Middle East"):
        return "EMEA"
    if th.startswith("North America - Canada"):
        return "Canada"
    if th in ("Developed Markets", "EM Broad"):
        return "Global Broad"
    return "US"


def _refine_style_by_industry(sector: str, industry, industry_group, base_style: str) -> str:
    """Phase 1.5 — industry-level 세분화로 style_tilt 보정.

    Healthcare/Biotech → growth (Healthcare 기본 balanced를 override)
    Energy/Uranium·Solar → growth (Energy 기본 value를 override)
    Comm Services/Telecom → value (Comm Services 기본 growth를 override)
    Real Estate REITs → value (already value)
    """
    # NaN-safe coercion (pandas merge로 None/NaN 들어올 수 있음)
    ind = str(industry) if industry is not None and (isinstance(industry, str) or industry == industry) else ""
    ig = str(industry_group) if industry_group is not None and (isinstance(industry_group, str) or industry_group == industry_group) else ""
    if not ind: ind = ""
    if not ig: ig = ""
    # Biotech, Uranium, Solar → growth
    if "Biotech" in ind or "Uranium" in ind or "Solar" in ind:
        return "growth"
    # Telecom Services → value
    if "Telecom Services" in ind or "Telecommunication Services" in ig:
        return "value"
    return base_style


def _refine_cyclical_by_industry(sector: str, industry, base_tag: str) -> str:
    """Phase 1.5 — industry 단위에서 cyclical/defensive 재분류.

    Healthcare 하위:
      - Biotech → cyclical (high-beta growth, not defensive)
      - Pharma / Medical Devices → defensive (유지)
    Communication Services 하위:
      - Telecom Services → defensive (Verizon, T 같은 dividend-heavy)
      - Internet/Media → cyclical (유지)
    """
    # NaN-safe
    ind = str(industry) if industry is not None and (isinstance(industry, str) or industry == industry) else ""
    if not ind: ind = ""
    if "Biotech" in ind:
        return "cyclical"
    if "Telecom Services" in ind:
        return "defensive"
    return base_tag


def _apply_macro_context_tags(df) -> None:
    """In-place: add `cyclical_tag`, `style_tilt`, `region`, plus industry-refined variants.

    Output columns:
      - cyclical_tag      : 'cyclical' | 'defensive' | 'broad'
      - style_tilt        : 'growth' | 'value' | 'balanced'
      - region            : 'US' | 'Korea' | 'Japan' | 'China' | 'Europe' | ...
      - industry_group    : alias of gics_industry_group (or 'Unknown')
    """
    if df is None or df.empty:
        return
    if "sector" not in df.columns:
        df["sector"] = "Other"
    if "theme" not in df.columns:
        df["theme"] = "-"

    # Base sector-level tags
    df["cyclical_tag"] = df["sector"].apply(
        lambda s: "cyclical" if s in CYCLICAL_SECTORS_SET
        else "defensive" if s in DEFENSIVE_SECTORS_SET
        else "broad"
    )
    df["style_tilt"] = df["sector"].apply(
        lambda s: "growth" if s in GROWTH_SECTORS_SET
        else "value" if s in VALUE_SECTORS_SET
        else "balanced"
    )

    # Industry-level refinement (Phase 1.5)
    if "gics_industry" in df.columns or "gics_industry_group" in df.columns:
        df["style_tilt"] = df.apply(
            lambda r: _refine_style_by_industry(
                r.get("sector", ""),
                r.get("gics_industry", "") or "",
                r.get("gics_industry_group", "") or "",
                r["style_tilt"],
            ), axis=1,
        )
        df["cyclical_tag"] = df.apply(
            lambda r: _refine_cyclical_by_industry(
                r.get("sector", ""),
                r.get("gics_industry", "") or "",
                r["cyclical_tag"],
            ), axis=1,
        )

    # Region
    df["region"] = df.apply(
        lambda r: _detect_region(r.get("ticker", "") or "", r.get("theme", "") or ""),
        axis=1,
    )

    # industry_group alias (Unknown fallback)
    if "gics_industry_group" in df.columns:
        df["industry_group"] = df["gics_industry_group"].fillna("Unknown")
    else:
        df["industry_group"] = "Unknown"

    # Console summary
    try:
        n_total = len(df)
        n_cyc = int((df["cyclical_tag"] == "cyclical").sum())
        n_def = int((df["cyclical_tag"] == "defensive").sum())
        n_growth = int((df["style_tilt"] == "growth").sum())
        n_value = int((df["style_tilt"] == "value").sum())
        n_regions = df["region"].nunique()
        print(f"[macro-tags] {n_total} tickers · cyc={n_cyc} def={n_def} · "
              f"growth={n_growth} value={n_value} · regions={n_regions}")
    except Exception:
        pass


def _inject_tags_into_results(df, results) -> None:
    """Phase 1: STATE['results'] (Pre-Mom 입력) 의 각 dict에 macro tags 부착.

    pre_momentum.py 의 MacroRegimeAgent 가 cyclical_tag/style_tilt/region 활용 가능.
    """
    if df is None or df.empty or not results:
        return
    try:
        tag_map: Dict[str, Dict[str, str]] = {}
        for _, row in df.iterrows():
            tk = row.get("ticker", "")
            if tk:
                tag_map[tk] = {
                    "cyclical_tag": row.get("cyclical_tag", "broad"),
                    "style_tilt": row.get("style_tilt", "balanced"),
                    "region": row.get("region", "US"),
                    "industry_group": row.get("industry_group", "Unknown"),
                }
        for r in results:
            if not isinstance(r, dict):
                continue
            tk = r.get("ticker", "")
            if tk in tag_map:
                r.update(tag_map[tk])
    except Exception as e:
        print(f"[macro-tags] inject_results failed: {e}")


def _detect_market_regime(df) -> dict:
    """Cross-sectional regime detection — used by Phase 2D rotation scoring + Phase 3B override.

    Returns dict with:
      - cyclical_dom / defensive_dom : bool
      - growth_dom / value_dom       : bool
      - top_region / bot_region      : str
      - region_comp_map              : dict[region → avg composite]
      - sector_avg                   : dict[sector → avg composite]
      - cd_gap / gv_gap              : float (cyclical - defensive, growth - value composite diff)
    """
    regime = {
        "cyclical_dom": False, "defensive_dom": False,
        "growth_dom": False, "value_dom": False,
        "top_region": "US", "bot_region": "US",
        "region_comp_map": {}, "sector_avg": {},
        "cd_gap": 0.0, "gv_gap": 0.0,
    }
    if df is None or df.empty:
        return regime
    try:
        if "cyclical_tag" not in df.columns or "composite" not in df.columns:
            return regime
        comp = pd.to_numeric(df["composite"], errors="coerce")
        cyc_mask = df["cyclical_tag"] == "cyclical"
        def_mask = df["cyclical_tag"] == "defensive"
        cyc_avg = float(comp[cyc_mask].mean()) if cyc_mask.any() else 50.0
        def_avg = float(comp[def_mask].mean()) if def_mask.any() else 50.0
        regime["cd_gap"] = round(cyc_avg - def_avg, 2)
        regime["cyclical_dom"] = (cyc_avg - def_avg) > 3.0
        regime["defensive_dom"] = (def_avg - cyc_avg) > 3.0

        grw_mask = df["style_tilt"] == "growth"
        val_mask = df["style_tilt"] == "value"
        grw_avg = float(comp[grw_mask].mean()) if grw_mask.any() else 50.0
        val_avg = float(comp[val_mask].mean()) if val_mask.any() else 50.0
        regime["gv_gap"] = round(grw_avg - val_avg, 2)
        regime["growth_dom"] = (grw_avg - val_avg) > 3.0
        regime["value_dom"] = (val_avg - grw_avg) > 3.0

        # Region leadership
        if "region" in df.columns:
            region_comp = df.groupby("region", observed=True)["composite"].mean().to_dict()
            regime["region_comp_map"] = {k: round(float(v), 1) for k, v in region_comp.items() if pd.notna(v)}
            if region_comp:
                sorted_r = sorted(region_comp.items(), key=lambda kv: -kv[1])
                regime["top_region"] = sorted_r[0][0]
                regime["bot_region"] = sorted_r[-1][0]

        # Sector averages
        if "sector" in df.columns:
            sec_comp = df.groupby("sector", observed=True)["composite"].mean().to_dict()
            regime["sector_avg"] = {k: round(float(v), 1) for k, v in sec_comp.items() if pd.notna(v)}

        print(f"[regime] cd_gap={regime['cd_gap']} (cyc_dom={regime['cyclical_dom']} def_dom={regime['defensive_dom']}) · "
              f"gv_gap={regime['gv_gap']} (grw_dom={regime['growth_dom']} val_dom={regime['value_dom']}) · "
              f"top_region={regime['top_region']} bot_region={regime['bot_region']}")
    except Exception as e:
        print(f"[regime] detect failed: {e}")
    return regime


def _compute_rotation_scores(df, regime: dict) -> None:
    """Phase 2D — per-ticker rotation_long / rotation_short scores (0-100).

    Long: 종목이 현 regime의 강세 그룹(cyclical/growth/top-region)에 정렬되면 점수↑
    Short: 종목이 약세 그룹(defensive/value 안티-regime, bot-region)에 정렬되면 점수↑
    """
    if df is None or df.empty:
        return
    if "cyclical_tag" not in df.columns or "composite" not in df.columns:
        df["rotation_long"] = 0.0
        df["rotation_short"] = 0.0
        return

    cyc_dom = regime.get("cyclical_dom", False)
    def_dom = regime.get("defensive_dom", False)
    grw_dom = regime.get("growth_dom", False)
    val_dom = regime.get("value_dom", False)
    top_region = regime.get("top_region", "US")
    bot_region = regime.get("bot_region", "US")

    def _long(row):
        s = 0.0
        # Risk alignment
        if cyc_dom and row.get("cyclical_tag") == "cyclical":
            s += 25
        elif def_dom and row.get("cyclical_tag") == "defensive":
            s += 25
        elif row.get("cyclical_tag") == "broad":
            s += 10
        # Style alignment
        if grw_dom and row.get("style_tilt") == "growth":
            s += 25
        elif val_dom and row.get("style_tilt") == "value":
            s += 25
        elif row.get("style_tilt") == "balanced":
            s += 10
        # Region alignment
        rg = row.get("region", "US")
        if rg == top_region:
            s += 30
        elif rg == bot_region:
            s += 0
        else:
            s += 15
        # Composite tilt
        comp = float(row.get("composite") or 50)
        s += max(0.0, min(20.0, (comp - 50.0) * 0.4))
        return max(0.0, min(100.0, round(s, 1)))

    def _short(row):
        s = 0.0
        # Misalignment penalties (translate to short signal)
        if cyc_dom and row.get("cyclical_tag") == "defensive":
            s += 20
        elif def_dom and row.get("cyclical_tag") == "cyclical":
            s += 20
        if grw_dom and row.get("style_tilt") == "value":
            s += 20
        elif val_dom and row.get("style_tilt") == "growth":
            s += 20
        rg = row.get("region", "US")
        if rg == bot_region:
            s += 25
        # Composite penalty
        comp = float(row.get("composite") or 50)
        s += max(0.0, min(35.0, (50.0 - comp) * 0.7))
        return max(0.0, min(100.0, round(s, 1)))

    df["rotation_long"] = df.apply(_long, axis=1)
    df["rotation_short"] = df.apply(_short, axis=1)
    try:
        avg_l = float(df["rotation_long"].mean())
        avg_s = float(df["rotation_short"].mean())
        print(f"[rotation] avg long={avg_l:.1f}, short={avg_s:.1f}")
    except Exception:
        pass


def _phase3b_regime_classify_override(df, regime: dict) -> None:
    """Phase 3B — Regime-aware CYCLE_PEAK 조기 승격 (api.py post-load 단계).

    조건:
      A) (Risk-Off regime) AND (ticker cyclical) AND (OVEREXTENDED) AND (OER ≥ 70)
         → CYCLE_PEAK 으로 승격 (시장 환경이 cyclical 종목에 불리한 상황의 과열)
      B) (Risk-On regime) AND (ticker defensive) AND (OVEREXTENDED) AND (OER ≥ 70)
         → CYCLE_PEAK (anomaly — defensive leadership이 risk-on에 등장 → 곧 fade)

    Scan-time classification은 유지하고 served data만 업데이트.
    `classification_raw` 컬럼에 원본 보존.
    """
    if df is None or df.empty:
        return
    if "classification" not in df.columns or "cyclical_tag" not in df.columns or "oer" not in df.columns:
        return
    try:
        df["classification_raw"] = df["classification"]  # 원본 보존
        cyc_dom = regime.get("cyclical_dom", False)
        def_dom = regime.get("defensive_dom", False)

        def _maybe_override(row):
            cls = row.get("classification", "")
            if cls != "🟡 OVEREXTENDED":
                return cls
            oer = float(row.get("oer") or 0)
            tag = row.get("cyclical_tag", "broad")
            # Risk-Off + Cyclical 종목 + 과열
            if def_dom and tag == "cyclical" and oer >= 70:
                return "🔴 CYCLE_PEAK"
            # Risk-On + Defensive 종목 + 과열 (anomaly)
            if cyc_dom and tag == "defensive" and oer >= 70:
                return "🔴 CYCLE_PEAK"
            return cls

        new_cls = df.apply(_maybe_override, axis=1)
        n_changed = int((new_cls != df["classification"]).sum())
        if n_changed > 0:
            df["classification"] = new_cls
            print(f"[phase3b] Regime-aware CYCLE_PEAK override: {n_changed} tickers upgraded "
                  f"(def_dom={def_dom}, cyc_dom={cyc_dom})")
    except Exception as e:
        print(f"[phase3b] override failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Hybrid Bottom-up — ETF sidecar metrics + divergence flags (Phase A + B)
# ═════════════════════════════════════════════════════════════════════════════

_ETF_HOLDINGS_CACHE_PATH = ".etf_holdings_cache.json"

# Bullish/Excluded classification sets — used for breadth metrics.
_BULLISH_CLS = {
    "🟢 CONTINUATION", "🔵 FORMATION", "🟦 LAGGING_CATCHUP",
    "🔵 RECOVERY", "🟡 OVEREXTENDED",
}
_MOMENTUM_STAGE_CLS = {
    "🟢 CONTINUATION", "🔵 FORMATION", "🟦 LAGGING_CATCHUP", "🟡 OVEREXTENDED",
}


def _load_etf_holdings() -> dict:
    """Load ETF holdings cache (built by etf_holdings_pipeline.py).
    Returns {ticker: {holdings: [{ticker, name, weight}, ...], ...}}."""
    if not os.path.exists(_ETF_HOLDINGS_CACHE_PATH):
        return {}
    try:
        with open(_ETF_HOLDINGS_CACHE_PATH) as f:
            payload = json.load(f) or {}
        return payload.get("etfs", {}) or {}
    except Exception as e:
        print(f"[etf-hybrid] holdings load failed: {e}")
        return {}


def _compute_etf_hybrid_sidecar(df) -> None:
    """For each ETF with cached holdings, compute bottom-up sidecar metrics:

    - constituent_breadth_mom  : % of available constituents in momentum stage (0-100)
    - constituent_weighted_comp: cap-weighted avg constituent Composite (using top-10 weights)
    - constituent_coverage     : % of constituent weight that exists in our scan universe
    - constituent_concentration: HHI-like sum of squared weights (top-N) — concentration index
    - constituent_leader_gap   : max constituent Composite minus ETF own Composite
    - divergence_flag          : HEALTHY_TREND | NARROW_RALLY | STEALTH_STRENGTH | WRAPPER_DRAG | NEUTRAL
    """
    if df is None or df.empty:
        return
    holdings_map = _load_etf_holdings()
    if not holdings_map:
        # Initialize empty columns (downstream code-safe)
        for col in ("constituent_breadth_mom", "constituent_weighted_comp",
                    "constituent_coverage", "constituent_concentration",
                    "constituent_leader_gap", "divergence_flag"):
            df[col] = None
        print(f"[etf-hybrid] cache empty — skipping sidecar")
        return

    # Build ticker → row lookup for fast constituent access
    ticker_to_row = {}
    for _, row in df.iterrows():
        tk = row.get("ticker", "")
        if tk:
            ticker_to_row[tk] = {
                "composite": row.get("composite"),
                "classification": row.get("classification", ""),
            }

    # Per-ETF metrics
    metrics: Dict[str, Dict] = {}
    n_etfs_processed = 0
    for etf_ticker, holdings_info in holdings_map.items():
        holdings = holdings_info.get("holdings", []) or []
        if not holdings:
            continue
        # Aggregate constituent stats (only those in our universe)
        sum_weight_in_universe = 0.0
        weighted_comp_num = 0.0
        n_mom = 0
        n_in_universe = 0
        max_comp = -1.0
        concentration = 0.0
        for h in holdings:
            tk = h.get("ticker", "")
            w = float(h.get("weight", 0) or 0)
            concentration += w * w   # HHI-like
            if tk in ticker_to_row:
                row = ticker_to_row[tk]
                comp = row["composite"]
                if comp is None or (isinstance(comp, float) and comp != comp):
                    continue
                comp = float(comp)
                sum_weight_in_universe += w
                weighted_comp_num += w * comp
                n_in_universe += 1
                if row["classification"] in _MOMENTUM_STAGE_CLS:
                    n_mom += 1
                if comp > max_comp:
                    max_comp = comp
        if n_in_universe == 0:
            continue
        weighted_comp = weighted_comp_num / sum_weight_in_universe if sum_weight_in_universe > 0 else 0.0
        breadth_mom = (n_mom / n_in_universe) * 100 if n_in_universe > 0 else 0.0
        total_weight = sum(h.get("weight", 0) or 0 for h in holdings)
        coverage = (sum_weight_in_universe / total_weight) * 100 if total_weight > 0 else 0.0

        # ETF's own composite for divergence comparison
        etf_row = ticker_to_row.get(etf_ticker)
        etf_comp = float(etf_row["composite"]) if etf_row and etf_row["composite"] is not None else 50.0

        leader_gap = max_comp - etf_comp if max_comp >= 0 else 0.0

        metrics[etf_ticker] = {
            "constituent_breadth_mom": round(breadth_mom, 1),
            "constituent_weighted_comp": round(weighted_comp, 1),
            "constituent_coverage": round(coverage, 1),
            "constituent_concentration": round(concentration, 4),
            "constituent_leader_gap": round(leader_gap, 1),
            "_etf_comp": etf_comp,
            "_n_in_universe": n_in_universe,
        }
        n_etfs_processed += 1

    # Phase B — Divergence flags (4-quadrant)
    # Healthy Trend  : ETF strong (Comp >=60) + breadth_mom >=70%  → 광범위 leadership
    # Narrow Rally   : ETF strong + breadth_mom <40%               → 소수 mega-cap 견인, concentration risk
    # Stealth Strength: ETF flat (Comp <55)  + breadth_mom >60%    → 구성종목 강세, ETF lagging → 곧 breakout 가능
    # Wrapper Drag   : ETF weak (Comp <50)   + breadth_mom >50%    → 구성종목은 OK, ETF만 FX/leverage drag
    for tk, m in metrics.items():
        ec = m["_etf_comp"]
        bm = m["constituent_breadth_mom"]
        if ec >= 60 and bm >= 70:
            flag = "HEALTHY_TREND"
        elif ec >= 60 and bm < 40:
            flag = "NARROW_RALLY"
        elif ec < 55 and bm >= 60:
            flag = "STEALTH_STRENGTH"
        elif ec < 50 and bm >= 50:
            flag = "WRAPPER_DRAG"
        else:
            flag = "NEUTRAL"
        m["divergence_flag"] = flag

    # Inject into df
    def _get(t, key):
        m = metrics.get(t)
        return m.get(key) if m else None

    for col in ("constituent_breadth_mom", "constituent_weighted_comp",
                "constituent_coverage", "constituent_concentration",
                "constituent_leader_gap", "divergence_flag"):
        df[col] = df["ticker"].apply(lambda t: _get(t, col))

    # Console summary
    try:
        flag_counts: Dict[str, int] = {}
        for tk, m in metrics.items():
            f = m.get("divergence_flag", "NEUTRAL")
            flag_counts[f] = flag_counts.get(f, 0) + 1
        flag_str = ", ".join(f"{k}={v}" for k, v in sorted(flag_counts.items(), key=lambda x: -x[1]))
        print(f"[etf-hybrid] {n_etfs_processed} ETFs processed · flags: {flag_str}")
    except Exception:
        pass


def _compute_provisional_eligibility(df, results, graph, history, ve_obs) -> None:
    """Anti-Lag Phase 1 — Pre-Momentum 종목의 Momentum 탭 조기 surface.

    Pre-Momentum 시스템이 forward-looking 신호를 제공하나 현재는 별도 탭에 격리됨.
    아래 조건 모두 충족 시 Momentum 탭에 PROVISIONAL 태그로 노출:
      - pre_momentum_score ≥ 45  (실제 distribution: max ~52, top 10-15% range)
      - agreement_ratio ≥ 0.6 (STRONG = 3+ agents firing)
      - PM-stage classification (NEUTRAL/CONSOLIDATION/RECOVERY/PULLBACK) OR
        bullish-but-low-Composite (FORMATION/CONTINUATION but Composite < 55)

    이 작업으로 lag 단축 10-15일 기대.

    출력 컬럼:
      - pre_momentum_score   : 0-100
      - pm_agreement_ratio    : 0-1
      - pm_conviction         : STRONG / MODERATE / WEAK / NONE
      - pm_timeline           : "1-2 weeks" / "2-4 weeks" / ...
      - provisional_eligible  : bool — PM 조건 충족 + bullish setup
      - eligibility_tier      : 'EligibleMomentum' / 'ProvisionalPM' / 'PreMomentum' / 'Excluded'
    """
    if df is None or df.empty or not results:
        return
    try:
        from pre_momentum import run_pre_momentum
    except Exception as e:
        print(f"[phase1-antilag] pre_momentum import failed: {e}")
        return

    try:
        # Build pre_momentum input cache
        try:
            from fundamentals_pipeline import load_fundamentals_cache
            fund_cache = load_fundamentals_cache()
        except Exception:
            fund_cache = None
        pm_cache = {
            "results": results,
            "graph": graph,
            "history": history,
            "ve_observations": ve_obs,
            "fundamentals": fund_cache,
        }
        pm_output = run_pre_momentum(pm_cache)
        candidates = pm_output.get("candidates", []) or []

        # Build ticker → PM data lookup
        pm_map: Dict[str, Dict] = {}
        for c in candidates:
            if isinstance(c, dict):
                tk = c.get("ticker", "")
                if tk:
                    pm_map[tk] = {
                        "pre_momentum_score": float(c.get("pre_momentum_score", 0) or 0),
                        "pm_agreement_ratio": float(c.get("agreement_ratio", 0) or 0),
                        "pm_conviction": c.get("conviction") or _agreement_to_conviction(c.get("agreement_ratio", 0)),
                        "pm_timeline": c.get("expected_timeline") or "",
                    }

        # Inject columns
        df["pre_momentum_score"] = df["ticker"].apply(lambda t: pm_map.get(t, {}).get("pre_momentum_score"))
        df["pm_agreement_ratio"] = df["ticker"].apply(lambda t: pm_map.get(t, {}).get("pm_agreement_ratio"))
        df["pm_conviction"] = df["ticker"].apply(lambda t: pm_map.get(t, {}).get("pm_conviction"))
        df["pm_timeline"] = df["ticker"].apply(lambda t: pm_map.get(t, {}).get("pm_timeline"))

        # Bullish PM classes — those qualifying for provisional entry
        # (already filtered upstream by run_pre_momentum, but double-guard here)
        _BULL_PM_CLS = {
            "🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔵 RECOVERY", "🔶 PULLBACK",
            "🔵 FORMATION", "🟢 CONTINUATION", "🟦 LAGGING_CATCHUP",
        }
        # Per-row provisional check (NaN-safe)
        def _is_provisional(row):
            cls = row.get("classification", "")
            if cls not in _BULL_PM_CLS:
                return False
            pm = row.get("pre_momentum_score")
            ar = row.get("pm_agreement_ratio")
            if pm is None or ar is None:
                return False
            try:
                pm_v = float(pm)
                ar_v = float(ar)
                # NaN check (NaN != NaN)
                if pm_v != pm_v or ar_v != ar_v:
                    return False
                if pm_v < 45.0 or ar_v < 0.6:
                    return False
            except Exception:
                return False
            # 이미 eligible=True 인 종목은 별도 표시 불필요 (Momentum 탭에서 표시됨)
            if bool(row.get("eligible")):
                return False
            return True

        df["provisional_eligible"] = df.apply(_is_provisional, axis=1)

        # Eligibility tier (mutually exclusive 4-tier)
        def _tier(row):
            if bool(row.get("eligible")):
                return "EligibleMomentum"
            if bool(row.get("provisional_eligible")):
                return "ProvisionalPM"
            cls = row.get("classification", "")
            # PM stage classes — watchlist only
            if cls in ("🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔵 RECOVERY", "🔶 PULLBACK"):
                return "PreMomentum"
            return "Excluded"
        df["eligibility_tier"] = df.apply(_tier, axis=1)

        # Also inject into STATE["results"] for downstream consumers (Pre-Mom agent etc.)
        score_map: Dict[str, Dict] = {}
        for _, row in df.iterrows():
            tk = row.get("ticker", "")
            if tk:
                score_map[tk] = {
                    "pre_momentum_score": row.get("pre_momentum_score"),
                    "pm_agreement_ratio": row.get("pm_agreement_ratio"),
                    "pm_conviction": row.get("pm_conviction"),
                    "provisional_eligible": bool(row.get("provisional_eligible")),
                    "eligibility_tier": row.get("eligibility_tier"),
                }
        for r in results:
            if isinstance(r, dict):
                tk = r.get("ticker", "")
                if tk in score_map:
                    r.update({k: v for k, v in score_map[tk].items() if v is not None})

        n_prov = int(df["provisional_eligible"].fillna(False).sum())
        n_elig = int(df["eligible"].fillna(False).sum())
        n_pre = int((df["eligibility_tier"] == "PreMomentum").sum())
        n_exc = int((df["eligibility_tier"] == "Excluded").sum())
        print(f"[phase1-antilag] tiers · EligibleMomentum={n_elig} · ProvisionalPM={n_prov} · "
              f"PreMomentum={n_pre} · Excluded={n_exc}")
    except Exception as e:
        print(f"[phase1-antilag] failed: {e}")
        df["provisional_eligible"] = False
        df["eligibility_tier"] = df.get("eligible").apply(
            lambda x: "EligibleMomentum" if x else "Excluded"
        ) if "eligible" in df.columns else "Excluded"


def _compute_sector_segmented_picks(df, top_per_sector: int = 3, min_composite: float = 40.0) -> None:
    """Sector-Segmented Price Discovery — 각 섹터별로 top-N picks 선별.

    Universe-wide ranking과는 독립적으로 작동:
      - 각 섹터 내에서 bullish classification + composite ≥ min_composite
      - composite 기준 내림차순 정렬 → top-N 종목 sector_segmented_eligible=True
      - Sector rank/percentile, eligibility_tier_v2 컬럼 부착

    효과:
      - Lag 단축은 미미 (composite 기반 ranking은 동일)
      - 그러나 diversification 강화 (모든 sector에서 최소 N개 보장)
      - Universe top-N에서는 누락된 sector-best 종목 포착 (e.g., Defensive sector의 top picks)

    출력 컬럼:
      - sector_segmented_eligible : bool
      - sector_rank               : int (1 = best in sector)
      - sector_pct_rank           : float (0-100)
      - sector_top_n              : int (해당 sector 내 candidates 수)
      - eligibility_tier_v2       : 'BothEligible' / 'UniverseOnly' / 'SectorOnly' / 'Neither'
    """
    if df is None or df.empty:
        return
    if "sector" not in df.columns or "composite" not in df.columns:
        return

    BULLISH_CLS = {
        "🟢 CONTINUATION", "🔵 FORMATION", "🟦 LAGGING_CATCHUP",
        "🔵 RECOVERY", "🟡 OVEREXTENDED",
    }
    # Skip non-equity sectors where momentum-style scoring is less meaningful
    SKIP_SECTORS = {"Fixed Income", "Macro", "Multi-Asset", "Alternatives"}

    # Initialize columns
    df["sector_segmented_eligible"] = False
    df["sector_rank"] = None
    df["sector_pct_rank"] = None
    df["sector_top_n"] = None

    n_sectors_processed = 0
    n_picks_total = 0

    for sec, group in df.groupby("sector", observed=True):
        if not isinstance(sec, str) or not sec or sec in SKIP_SECTORS:
            continue
        # Candidates: bullish classification + min composite + valid data
        comp_num = pd.to_numeric(group["composite"], errors="coerce")
        mask = (
            group["classification"].isin(BULLISH_CLS)
            & comp_num.notna()
            & (comp_num >= min_composite)
        )
        candidates = group[mask].copy()
        candidates["__comp_num"] = pd.to_numeric(candidates["composite"], errors="coerce")
        if candidates.empty:
            continue

        # Sort by composite descending within sector
        candidates = candidates.sort_values("__comp_num", ascending=False)
        n_candidates = len(candidates)

        for rank, (idx, _) in enumerate(candidates.iterrows(), 1):
            df.at[idx, "sector_rank"] = int(rank)
            df.at[idx, "sector_pct_rank"] = round((1 - (rank - 1) / max(1, n_candidates)) * 100, 1)
            df.at[idx, "sector_top_n"] = int(n_candidates)
            if rank <= top_per_sector:
                df.at[idx, "sector_segmented_eligible"] = True
                n_picks_total += 1

        n_sectors_processed += 1

    # Eligibility tier v2 (mutually exclusive)
    def _tier_v2(row):
        u_elig = bool(row.get("eligible"))
        s_elig = bool(row.get("sector_segmented_eligible"))
        if u_elig and s_elig: return "BothEligible"
        if u_elig:            return "UniverseOnly"
        if s_elig:            return "SectorOnly"
        return "Neither"
    df["eligibility_tier_v2"] = df.apply(_tier_v2, axis=1)

    try:
        from collections import Counter
        tier_counts = Counter(df["eligibility_tier_v2"].fillna("Neither"))
        print(f"[sector-segmented] {n_sectors_processed} sectors · top-{top_per_sector} per sector · "
              f"total picks={n_picks_total} · tiers v2: "
              f"BothEligible={tier_counts.get('BothEligible', 0)}, "
              f"UniverseOnly={tier_counts.get('UniverseOnly', 0)}, "
              f"SectorOnly={tier_counts.get('SectorOnly', 0)}, "
              f"Neither={tier_counts.get('Neither', 0)}")
    except Exception:
        pass


def _agreement_to_conviction(ratio) -> str:
    try:
        v = float(ratio or 0)
    except Exception:
        return "NONE"
    if v >= 0.6: return "STRONG"
    if v >= 0.4: return "MODERATE"
    if v > 0:    return "WEAK"
    return "NONE"


def _compute_parent_etf_signal(df, results) -> None:
    """Phase D — 각 ticker에 parent_etf_signal 부착 (Pre-Mom MacroRegimeAgent 활용).

    Stock 입장: 자신을 top-holding으로 보유한 ETF들의 divergence_flag 가중 평균.
    ETF 입장: 자신의 constituent_breadth_mom (이미 계산됨)을 직접 사용.

    Signal contribution by flag (forward-looking 가치 순):
      STEALTH_STRENGTH   +30 : 구성종목 강세, ETF 아직 횡보 → breakout 예고 (가장 강함)
      HEALTHY_TREND      +20 : 광범위 leadership 형성
      WRAPPER_DRAG       +15 : 구성종목 OK, ETF만 drag
      NARROW_RALLY        +5 : 이미 강세이나 concentration risk
      NEUTRAL             0
    """
    if df is None or df.empty:
        return
    holdings_map = _load_etf_holdings()
    if not holdings_map:
        df["parent_etf_signal"] = None
        return

    FLAG_CONTRIB = {
        "STEALTH_STRENGTH": 30.0,
        "HEALTHY_TREND": 20.0,
        "WRAPPER_DRAG": 15.0,
        "NARROW_RALLY": 5.0,
        "NEUTRAL": 0.0,
    }

    # Build df-level flag/coverage lookup for ETFs
    etf_flag: Dict[str, str] = {}
    etf_cov: Dict[str, float] = {}
    for _, row in df.iterrows():
        if row.get("asset_type") == "ETF":
            tk = row.get("ticker", "")
            f = row.get("divergence_flag")
            c = row.get("constituent_coverage")
            if tk and f is not None:
                etf_flag[tk] = str(f)
                etf_cov[tk] = float(c) if c is not None and (isinstance(c,(int,float)) and c == c) else 0.0

    # Build reverse index: stock_ticker -> [(parent_etf, weight, flag, coverage)]
    reverse_idx: Dict[str, list] = {}
    for etf_tk, info in holdings_map.items():
        flag = etf_flag.get(etf_tk, "NEUTRAL")
        cov = etf_cov.get(etf_tk, 0.0)
        if cov < 30:    # 신뢰도 낮은 ETF는 reverse propagation에서 제외
            continue
        for h in info.get("holdings", []) or []:
            stk = h.get("ticker", "")
            w = float(h.get("weight", 0) or 0)
            if stk and w > 0:
                reverse_idx.setdefault(stk, []).append((etf_tk, w, flag, cov))

    # Compute per-ticker signal (50 = neutral baseline)
    def _compute(row):
        tk = row.get("ticker", "")
        asset = row.get("asset_type", "")
        # ETF: use own breadth as signal (already 0-100)
        if asset == "ETF":
            bm = row.get("constituent_breadth_mom")
            if bm is None or (isinstance(bm,float) and bm != bm):
                return None
            return round(float(bm), 1)
        # Stock: aggregate parent ETF signals
        parents = reverse_idx.get(tk, [])
        if not parents:
            return None
        # Weighted aggregation: weight by ETF holding weight × coverage_pct/100
        sig_num = 0.0
        wt_sum = 0.0
        for _, w, flag, cov in parents:
            ww = w * (cov / 100.0)
            sig_num += (50.0 + FLAG_CONTRIB.get(flag, 0.0)) * ww
            wt_sum += ww
        if wt_sum <= 0:
            return None
        return round(min(100.0, max(0.0, sig_num / wt_sum)), 1)

    df["parent_etf_signal"] = df.apply(_compute, axis=1)

    # Inject into STATE["results"] for Pre-Mom agent access
    if results:
        sig_map: Dict[str, float] = {}
        for _, row in df.iterrows():
            tk = row.get("ticker", "")
            sig = row.get("parent_etf_signal")
            if tk and sig is not None:
                sig_map[tk] = float(sig)
        n_inj = 0
        for r in results:
            if isinstance(r, dict):
                tk = r.get("ticker", "")
                if tk in sig_map:
                    r["parent_etf_signal"] = sig_map[tk]
                    n_inj += 1
        try:
            stocks_with_signal = sum(1 for _, row in df.iterrows()
                                       if row.get("asset_type") == "Stock"
                                       and row.get("parent_etf_signal") is not None)
            etfs_with_signal = sum(1 for _, row in df.iterrows()
                                     if row.get("asset_type") == "ETF"
                                     and row.get("parent_etf_signal") is not None)
            print(f"[etf-hybrid] parent_etf_signal · stocks={stocks_with_signal}, ETFs={etfs_with_signal}, injected={n_inj}")
        except Exception:
            pass


def _inject_tags_into_observations(df, observations) -> None:
    """Phase 1G — ve_observations 각 entry에 ticker의 macro tags 첨부.

    SVE의 hit rate를 cyclical_tag / style_tilt / region 으로 segment 분석할 수 있도록.
    """
    if df is None or df.empty or not observations:
        return
    try:
        # Build ticker -> tags lookup
        tag_map: Dict[str, Dict[str, str]] = {}
        for _, row in df.iterrows():
            tk = row.get("ticker", "")
            if tk:
                tag_map[tk] = {
                    "cyclical_tag": row.get("cyclical_tag", "broad"),
                    "style_tilt": row.get("style_tilt", "balanced"),
                    "region": row.get("region", "US"),
                    "industry_group": row.get("industry_group", "Unknown"),
                }
        n_patched = 0
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            tk = obs.get("ticker", "")
            if tk in tag_map:
                obs.update(tag_map[tk])
                n_patched += 1
        print(f"[macro-tags] injected into {n_patched}/{len(observations)} ve_observations")
    except Exception as e:
        print(f"[macro-tags] inject failed: {e}")


# ───────────────────────────────────────────────────────────────
# Category normalization helpers
# ───────────────────────────────────────────────────────────────

# GICS sector → STK-style suffix (matches existing STOCK_UNIVERSE category names
# minus "STK_" prefix), so KR/JP/CN/EU/IN stocks slot into the same buckets as US.
GICS_SECTOR_TO_CAT_SUFFIX: Dict[str, str] = {
    "Information Technology":   "Technology",
    "Health Care":              "Healthcare",
    "Financials":               "Financials",
    "Consumer Discretionary":   "ConsDisc",
    "Consumer Staples":         "ConsStaples",
    "Industrials":              "Industrials",
    "Communication Services":   "CommServices",
    "Energy":                   "Energy",
    "Materials":                "Materials",
    "Utilities":                "Utilities",
    "Real Estate":              "RealEstate",
}

# International stock category prefixes that should be remapped via GICS lookup.
INTL_STOCK_CATEGORIES = {"STK_Korea", "STK_Japan", "STK_China_ADR",
                          "STK_Europe", "STK_India"}

# Korea_Equity ETF → unified sector mapping (manual curation).
# Removes the "Korea Equity" silo by absorbing each Korean ETF into the
# appropriate sector bucket alongside US/global stocks and ETFs.
KOREA_ETF_TO_SECTOR: Dict[str, str] = {
    # Broad-market Korean index ETFs
    "292150.KS": "Broad",       # TIGER 코리아TOP10
    "102110.KS": "Broad",       # TIGER 200 (KOSPI 200)
    "069500.KS": "Broad",       # KODEX 200 (KOSPI 200)
    "229200.KS": "Broad",       # KODEX 코스닥150
    # Technology / Semiconductors (KR + US-tracking)
    "395160.KS": "Technology",  # KODEX AI반도체
    "091160.KS": "Technology",  # KODEX 반도체
    "396500.KS": "Technology",  # TIGER 반도체TOP10
    "381180.KS": "Technology",  # TIGER 미국필라델피아반도체나스닥
    "381170.KS": "Technology",  # TIGER 미국테크TOP10
    # Factor (dividend / low-vol / value style)
    "161510.KS": "Factor",      # PLUS 고배당주
    # Industrials (battery / shipbuilding / electrical equipment)
    "487240.KS": "Industrials", # AI핵심전력설비
    "305720.KS": "Industrials", # KODEX 2차전지
    "466920.KS": "Industrials", # SOL 조선TOP3플러스
}


def _normalize_categories(df: "pd.DataFrame", uc: dict) -> None:
    """In-place: rewrite df['category'] so that
       (a) all STK_ prefixes are dropped, and
       (b) international stock categories (STK_Korea etc.) are remapped
           to the GICS-derived sector category (Technology / Healthcare / ...).
    Original raw category preserved as df['category_raw'] for reference.
    """
    if df is None or df.empty or "category" not in df.columns:
        return

    tickers_data = uc.get("tickers", {}) if isinstance(uc, dict) else {}

    df["category_raw"] = df["category"]

    def _normalize(row):
        cat = row.get("category", "") or ""
        tk = row.get("ticker", "")
        # 0. Korea_Equity ETF → unified sector via manual mapping (removes the
        #    "Korea Equity" silo by absorbing each Korean ETF into the
        #    appropriate sector bucket).
        if cat == "Korea_Equity":
            mapped = KOREA_ETF_TO_SECTOR.get(tk)
            if mapped:
                return mapped
            return "Broad"  # fallback for unmapped Korean equity ETFs
        # 1. ETF EQ_X → X — sector ETFs (EQ_Technology / EQ_Healthcare / etc.)
        #    merge into the SAME sector buckets as their stock counterparts.
        #    Non-sector EQ_ entries (EQ_Broad / EQ_Factor / EQ_Thematic) lose prefix.
        if cat.startswith("EQ_"):
            cat = cat[3:]
        # 2. ETF FI_X → "Bond X" — uniform "Bond ..." naming across fixed income
        elif cat.startswith("FI_"):
            cat = "Bond " + cat[3:]
        # 3. STK_X → X (with intl remap for Korea/Japan/China_ADR/Europe/India)
        elif cat.startswith("STK_"):
            if cat in INTL_STOCK_CATEGORIES:
                tk = row.get("ticker", "")
                gics = (tickers_data.get(tk, {}) or {}).get("gics_sector")
                suffix = GICS_SECTOR_TO_CAT_SUFFIX.get(gics)
                if suffix:
                    return suffix
                cat = cat[4:]
            else:
                cat = cat[4:]
        # 4. Final cleanup: replace ANY remaining underscores with spaces so all
        #    categories share a uniform display style (no mixed snake_case).
        return cat.replace("_", " ")

    df["category"] = df.apply(_normalize, axis=1)


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

    # Category → benchmark mapping with ticker counts.
    # df["category"] is now the NORMALIZED label (e.g., "Technology"); benchmark
    # dicts in price_discovery.py are still keyed by raw STK_ prefix, so we
    # consult category_raw for the lookup.
    all_bench = {**CATEGORY_BENCHMARK, **STOCK_BENCHMARK}
    cat_info = []
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        n = int(len(sub))
        asset = sub["asset_type"].iloc[0] if "asset_type" in sub.columns else (
            "Stock" if not cat.startswith(("EQ_", "FI", "Comm", "Curr", "Multi",
                                          "Real", "Korea_", "Emerging", "Intl_"))
            else "ETF"
        )
        # Benchmark lookup via raw category (ETF cats are unchanged; STK_X stays usable)
        raw_cats = sub["category_raw"].unique().tolist() if "category_raw" in sub.columns else [cat]
        bench = next((all_bench[rc] for rc in raw_cats if rc in all_bench), "SPY")
        alt_list: List[str] = []
        for rc in raw_cats:
            if rc in CATEGORY_BENCHMARK_ALT:
                alt_list.extend(CATEGORY_BENCHMARK_ALT[rc])
        alt = list(dict.fromkeys(alt_list)) if alt_list else [bench]
        cat_info.append({"category": cat, "n": n, "asset_type": asset,
                         "benchmark": bench, "alt_benchmarks": alt})

    # Theme summary — categories already normalized; no STK_ stripping needed
    themed = df[df["theme"] != "-"]
    theme_info = []
    if not themed.empty:
        for theme, grp in themed.groupby("theme"):
            theme_info.append({
                "theme": theme, "n": len(grp),
                "categories": sorted(set(grp["category"])),
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
                        # Heuristic phase tracking — order matters (later matches override
                        # earlier ones within same log line). Scan order:
                        #   Phase 1 → 2 → 3 → 4 → MASTER SUMMARY → Phase 7 → Phase 6 →
                        #   KEY INSIGHTS → Phase 8 → Cache saved
                        for kw, ph in [
                            ("Downloading", "Downloading"),
                            ("Phase 1", "Indicators"),
                            ("Phase 2", "Ranking"),
                            ("Phase 3", "Validity"),
                            ("Phase 4", "Scoring"),
                            ("MASTER SUMMARY", "Summary"),     # intermediate, NOT final
                            ("Phase 7", "Backtest"),           # ~50 weekly snapshots
                            ("Phase 6", "GraphRAG"),
                            ("KEY INSIGHTS", "Insights"),
                            ("Phase 8", "FactorEfficacy"),    # 12 eval points
                            ("Generating PDF", "Output"),     # PDF rendering
                            ("Cache saved", "Done"),
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
            "cyclical_tag", "style_tilt", "region", "industry_group",
            "rotation_long", "rotation_short",
            # Hybrid Bottom-up (Phase A + B) — ETF constituent sidecar
            "constituent_breadth_mom", "constituent_weighted_comp",
            "constituent_coverage", "constituent_concentration",
            "constituent_leader_gap", "divergence_flag",
            # Anti-Lag Phase 1 — Pre-Momentum direct entry
            "pre_momentum_score", "pm_agreement_ratio", "pm_conviction",
            "pm_timeline", "provisional_eligible", "eligibility_tier",
            # Sector-Segmented Price Discovery (New2)
            "sector_segmented_eligible", "sector_rank", "sector_pct_rank",
            "sector_top_n", "eligibility_tier_v2",
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
            # Multi-horizon returns + YTD + 3Y volatility
            "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_ytd", "ret_252d",
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
    cols = ["ticker", "name", "asset_type", "category", "theme", "mktcap_B",
            "ret_1d", "ret_5d", "ret_21d", "ret_63d", "ret_126d", "ret_ytd", "ret_252d",
            "ret_3y_ann", "ret_5y_ann", "vol_3y_ann"]
    cols = [c for c in cols if c in fdf.columns]
    # Round only numeric columns (asset_type is a string)
    numeric_cols = [c for c in cols if c not in ("ticker", "name", "asset_type", "category", "theme")]
    out = fdf[cols].copy()
    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].round(2)
    records = out.to_dict(orient="records")
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
        categories=("category", lambda x: ", ".join(sorted(set(x)))),
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
    """Unified classification metadata + per-ticker GICS/cap-tier + hierarchy table."""
    uc = STATE.get("unified_classification", {})
    if not uc:
        return {"available": False,
                "message": "Run `python3 unified_classifier.py` to generate."}
    tickers = uc.get("tickers", {})
    # Aggregate distribution stats
    by_gics: Dict[str, int] = {}
    by_industry_group: Dict[str, int] = {}
    by_cap: Dict[str, int] = {}
    by_country: Dict[str, int] = {}
    # Hierarchy rollup: Sector → Industry Group → Industry → tickers
    hierarchy: Dict[str, Dict[str, Dict[str, List[Dict]]]] = {}

    for tk, c in tickers.items():
        if not c.get("ok"):
            continue
        sec = c.get("gics_sector") or "Unknown"
        grp = c.get("gics_industry_group") or "Unknown"
        ind = c.get("gics_industry") or "Unknown"
        cap = c.get("cap_tier") or "Unknown"
        country = c.get("country") or "?"
        by_gics[sec] = by_gics.get(sec, 0) + 1
        by_industry_group[grp] = by_industry_group.get(grp, 0) + 1
        by_cap[cap] = by_cap.get(cap, 0) + 1
        by_country[country] = by_country.get(country, 0) + 1

        # Build hierarchy (only for stocks — ones with gics_sector)
        if sec == "Unknown":
            continue
        sector_dict = hierarchy.setdefault(sec, {})
        group_dict = sector_dict.setdefault(grp, {})
        ind_list = group_dict.setdefault(ind, [])
        ind_list.append({
            "ticker": tk,
            "name": c.get("name") or tk,
            "country": country,
            "cap_tier": cap,
            "mktcap_usd_b": c.get("mktcap_usd_b"),
        })

    # Flatten hierarchy into table rows for easy frontend rendering.
    # Each row: sector / industry_group / industry / n_stocks / sample_tickers / total_mktcap_b
    table_rows: List[Dict] = []
    for sec, groups in hierarchy.items():
        for grp, industries in groups.items():
            for ind, stocks in industries.items():
                stocks_sorted = sorted(stocks, key=lambda s: -(s.get("mktcap_usd_b") or 0))
                total_cap = sum(s.get("mktcap_usd_b") or 0 for s in stocks_sorted)
                sample = [s["ticker"] for s in stocks_sorted[:5]]
                table_rows.append({
                    "sector": sec,
                    "industry_group": grp,
                    "industry": ind,
                    "n_stocks": len(stocks),
                    "total_mktcap_b": round(total_cap, 1),
                    "sample_tickers": sample,
                    "tickers": stocks_sorted,
                })
    # Sort: by sector → industry_group → n_stocks desc
    table_rows.sort(key=lambda r: (r["sector"], r["industry_group"], -r["n_stocks"]))

    # Universe Category distribution — derived from the NORMALIZED df["category"]
    # (matches what the Universe sub-tab shows). Lets the Classification view stay
    # consistent with the rest of the dashboard.
    by_universe_cat: Dict[str, int] = {}
    df = STATE.get("df")
    if df is not None and not df.empty and "category" in df.columns:
        for cat, n in df["category"].value_counts().items():
            by_universe_cat[str(cat)] = int(n)

    # Theme distribution (top-30 SubThemes — what Universe shows in 'theme' column)
    by_theme: Dict[str, int] = {}
    if df is not None and not df.empty and "theme" in df.columns:
        for theme, n in df["theme"].value_counts().items():
            if theme == "-" or not theme:
                continue
            by_theme[str(theme)] = int(n)

    return _clean_dict({
        "available": True,
        "as_of": uc.get("as_of"),
        "n_total": uc.get("n_total"),
        "n_success": uc.get("n_success"),
        "n_failure": uc.get("n_failure"),
        "distribution": {
            "by_universe_category": by_universe_cat,    # ← matches Universe tab
            "by_universe_theme": by_theme,              # ← matches Universe tab
            "by_gics_sector": by_gics,
            "by_gics_industry_group": by_industry_group,
            "by_cap_tier": by_cap,
            "by_country": by_country,
        },
        "hierarchy_table": table_rows,
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
            "cyclical_tag", "style_tilt", "region", "industry_group",
            "rotation_long", "rotation_short",
            # Hybrid Bottom-up (Phase A + B) — ETF constituent sidecar
            "constituent_breadth_mom", "constituent_weighted_comp",
            "constituent_coverage", "constituent_concentration",
            "constituent_leader_gap", "divergence_flag",
            # Anti-Lag Phase 1 — Pre-Momentum direct entry
            "pre_momentum_score", "pm_agreement_ratio", "pm_conviction",
            "pm_timeline", "provisional_eligible", "eligibility_tier",
            # Sector-Segmented Price Discovery (New2)
            "sector_segmented_eligible", "sector_rank", "sector_pct_rank",
            "sector_top_n", "eligibility_tier_v2",
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

    # 2. Add current scan as the last "today" column.
    #    SVE는 매 2주 금요일을 anchor 로 ~24개 bi-weekly snapshots 생성.
    #    scan_time(today) 추가 → "Friday + today" 표시.
    #    단, last ve_obs와 scan_time이 매우 가까우면 (< MIN_VISUAL_GAP days)
    #    heatmap cell width가 좁아져 annotation overlap 발생 → 그 경우 REPLACE.
    MIN_VISUAL_GAP = 3
    if results:
        from datetime import datetime as _dt
        scan_time_raw = STATE.get("scan_time", "") or ""
        scan_time = scan_time_raw[:10] or _dt.utcnow().strftime("%Y-%m-%d")

        existing_dates = sorted(date_class_counts.keys())
        if existing_dates:
            try:
                last = _dt.strptime(existing_dates[-1][:10], "%Y-%m-%d")
                cur = _dt.strptime(scan_time[:10], "%Y-%m-%d")
                if 0 < (cur - last).days < MIN_VISUAL_GAP and existing_dates[-1] != scan_time:
                    # 너무 가까움 — last ve_obs slot 비우고 scan_time으로 대체
                    date_class_counts.pop(existing_dates[-1], None)
            except Exception:
                pass

        for r in results:
            cls = r.get("classification", "")
            if cls:
                date_class_counts[scan_time][cls] += 1

    # 3. Sort dates, keep last 24 bi-weekly snapshots (roughly 1 year)
    sorted_dates = sorted(date_class_counts.keys())[-24:]

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

    # Add current scan as the "today" column (matching /api/classification-history).
    # Per-sector — too close to last ve_obs (< 3 days) replaces, else appends.
    from datetime import datetime as _dt
    scan_time_raw = STATE.get("scan_time", "") or ""
    scan_time = scan_time_raw[:10] or _dt.utcnow().strftime("%Y-%m-%d")
    MIN_VISUAL_GAP = 3
    for cat in list(sector_date_cls.keys()):
        existing = sorted(sector_date_cls[cat].keys())
        if existing:
            try:
                last = _dt.strptime(existing[-1][:10], "%Y-%m-%d")
                cur = _dt.strptime(scan_time[:10], "%Y-%m-%d")
                if 0 < (cur - last).days < MIN_VISUAL_GAP and existing[-1] != scan_time:
                    sector_date_cls[cat].pop(existing[-1], None)
            except Exception:
                pass

    for r in results:
        cat = r.get("category", "")
        cls = r.get("classification", "")
        if cat and cls:
            sector_date_cls[cat][scan_time][cls] += 1

    # Build per-sector output (last 6 dates)
    all_dates = set()
    for cat_data in sector_date_cls.values():
        all_dates.update(cat_data.keys())
    sorted_dates = sorted(all_dates)[-24:]

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

    # ── Phase 1G: Hit rate segmented by cyclical_tag / style_tilt / region ──
    # 각 obs에 inject된 macro tags 활용. 동일 hypothesis 평가를 segment별로 분리 실행.
    segments = {"cyclical_tag": {}, "style_tilt": {}, "region": {}}
    tag_dims = [("cyclical_tag", ["cyclical", "defensive", "broad"]),
                ("style_tilt", ["growth", "value", "balanced"]),
                ("region", ["US", "Korea", "Japan", "China", "Europe", "Other Asia",
                            "LatAm", "EMEA", "Canada", "Global Broad"])]
    for dim_key, dim_values in tag_dims:
        for dim_val in dim_values:
            sub_obs = [o for o in ve_obs if isinstance(o, dict) and o.get(dim_key) == dim_val]
            if len(sub_obs) < 30:
                continue
            # Aggregate momentum hit rate (PASS_IDEAL + PASS) across all classifications
            n_pass = 0
            n_total = 0
            for cls, hyps in _MOMENTUM_HYPOTHESES.items():
                actuals = _compute_momentum_actuals(sub_obs, cls, {})
                for metric_key, spec in hyps.items():
                    verdict = _eval_hypothesis(actuals.get(metric_key), spec)
                    if verdict in ("PASS", "PASS_IDEAL"):
                        n_pass += 1
                        n_total += 1
                    elif verdict == "FAIL":
                        n_total += 1
            hit_rate = round(n_pass / n_total * 100, 1) if n_total > 0 else 0.0
            segments[dim_key][dim_val] = {
                "n_obs": len(sub_obs),
                "hit_rate": hit_rate,
                "n_pass": n_pass,
                "n_total": n_total,
            }

    return _clean_dict({
        "summary": {
            "total_observations": n_total_obs,
            "eval_points": n_eval_points,
            "date_range": [eval_dates[0] if eval_dates else "", eval_dates[-1] if eval_dates else ""],
        },
        "momentum": momentum_results,
        "pre_momentum": pre_momentum_results,
        # Phase 1G — segmented hit rates by macro context (cyclical/style/region)
        "segments": segments,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Anti-Lag Phase 1 Validation — PROVISIONAL tier forward performance tracking
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/new-pd/validation")
def new_pd_validation():
    """SVE-style validation for the New Price Discovery (PROVISIONAL) tier.

    PROVISIONAL은 PM Score + agreement_ratio 임계값을 사용하나, ve_obs에는 historical PM
    score가 없음. 따라서 proxy 정의를 사용:
      - Classification ∈ {NEUTRAL, CONSOLIDATION, RECOVERY, PULLBACK}  (bullish PM-stage)
      - score_composite ∈ [35, 55)                                     (almost eligible)

    이 proxy는 "Pre-Momentum 강한 신호이나 아직 Composite 미충족" 그룹을 capture.
    Forward 5d/21d/63d/126d/252d 수익률 + 양수 비율(success rate) + 차익 vs EligibleMomentum 산출.
    """
    ve_obs = STATE.get("ve_stats", {}).get("observations", []) or []
    if not ve_obs:
        return {"error": "No ve_observations available"}

    PM_STAGE_BULL = {"🟠 NEUTRAL", "🟡 CONSOLIDATION", "🔵 RECOVERY", "🔶 PULLBACK"}
    MOMENTUM_CLS = {"🟢 CONTINUATION", "🟡 OVEREXTENDED", "🔵 FORMATION", "🟦 LAGGING_CATCHUP"}
    BEARISH_CLS = {"⬇️ DOWNTREND", "🟤 EXHAUSTING", "🟤 FADING", "🟣 COUNTER_RALLY",
                   "🔴 CYCLE_PEAK", "⚠️ WEAKENING"}

    def _tier(o):
        cls = o.get("classification", "")
        comp = o.get("score_composite", 0) or 0
        elig = o.get("eligible", False)
        if elig and cls in MOMENTUM_CLS:
            return "EligibleMomentum"
        if cls in PM_STAGE_BULL and 35 <= comp < 55:
            return "ProvisionalPM_proxy"
        if cls in PM_STAGE_BULL:
            return "PreMomentum"
        if cls in BEARISH_CLS:
            return "Excluded"
        return "Other"

    from collections import defaultdict
    import statistics as _stat
    tier_buckets: dict = defaultdict(lambda: {
        "n": 0,
        "fwd_returns": {5: [], 21: [], 63: [], 126: [], 252: []},
        "bench_returns": {5: [], 21: [], 63: [], 126: [], 252: []},
        "fwd_pos_count": {5: 0, 21: 0, 63: 0, 126: 0, 252: 0},
        "fwd_total_count": {5: 0, 21: 0, 63: 0, 126: 0, 252: 0},
    })

    for o in ve_obs:
        if not isinstance(o, dict):
            continue
        tier = _tier(o)
        tier_buckets[tier]["n"] += 1
        fr = o.get("fwd_rets") or {}
        fb = o.get("fwd_bench") or {}
        for w in (5, 21, 63, 126, 252):
            v = fr.get(w) if isinstance(fr, dict) else None
            if v is not None and isinstance(v, (int, float)) and v == v:
                tier_buckets[tier]["fwd_returns"][w].append(float(v))
                tier_buckets[tier]["fwd_total_count"][w] += 1
                if v > 0:
                    tier_buckets[tier]["fwd_pos_count"][w] += 1
            vb = fb.get(w) if isinstance(fb, dict) else None
            if vb is not None and isinstance(vb, (int, float)) and vb == vb:
                tier_buckets[tier]["bench_returns"][w].append(float(vb))

    # Build response per tier
    tier_summary = []
    for tier in ["EligibleMomentum", "ProvisionalPM_proxy", "PreMomentum", "Excluded", "Other"]:
        d = tier_buckets[tier]
        row = {"tier": tier, "n_observations": d["n"]}
        for w in (5, 21, 63, 126, 252):
            rets = d["fwd_returns"][w]
            bench = d["bench_returns"][w]
            n = d["fwd_total_count"][w]
            pos = d["fwd_pos_count"][w]
            avg = _stat.mean(rets) if rets else None
            avg_bench = _stat.mean(bench) if bench else None
            excess = (avg - avg_bench) if avg is not None and avg_bench is not None else None
            stdev = _stat.stdev(rets) if len(rets) > 1 else None
            row[f"fwd_{w}d_avg"] = round(avg, 3) if avg is not None else None
            row[f"fwd_{w}d_bench"] = round(avg_bench, 3) if avg_bench is not None else None
            row[f"fwd_{w}d_excess"] = round(excess, 3) if excess is not None else None
            row[f"fwd_{w}d_stdev"] = round(stdev, 3) if stdev is not None else None
            row[f"fwd_{w}d_n"] = n
            row[f"fwd_{w}d_pos_pct"] = round((pos / n) * 100, 1) if n > 0 else None
            row[f"fwd_{w}d_sharpe_like"] = (
                round(avg / stdev * (252 ** 0.5) / 16, 3)
                if (avg is not None and stdev is not None and stdev > 0) else None
            )
        tier_summary.append(row)

    # PROVISIONAL → EligibleMomentum 'promotion rate' (sequential snapshots)
    by_ticker: dict = defaultdict(list)
    for o in ve_obs:
        if isinstance(o, dict):
            by_ticker[o.get("ticker", "")].append(o)
    for tk in by_ticker:
        by_ticker[tk].sort(key=lambda x: x.get("eval_date", ""))

    promo_yes = 0
    promo_no = 0
    promo_to_bearish = 0
    for tk, lst in by_ticker.items():
        for i, o in enumerate(lst[:-1]):
            if _tier(o) == "ProvisionalPM_proxy":
                nxt = lst[i + 1]
                nxt_tier = _tier(nxt)
                if nxt_tier == "EligibleMomentum":
                    promo_yes += 1
                elif nxt_tier == "Excluded":
                    promo_to_bearish += 1
                else:
                    promo_no += 1
    total_transitions = promo_yes + promo_no + promo_to_bearish
    promotion_stats = {
        "total_provisional_transitions": total_transitions,
        "promoted_to_eligible": promo_yes,
        "promoted_to_eligible_pct": round(promo_yes / max(1, total_transitions) * 100, 1),
        "stayed_or_demoted_neutral": promo_no,
        "stayed_or_demoted_neutral_pct": round(promo_no / max(1, total_transitions) * 100, 1),
        "demoted_to_excluded": promo_to_bearish,
        "demoted_to_excluded_pct": round(promo_to_bearish / max(1, total_transitions) * 100, 1),
    }

    # Methodology disclosure
    methodology = {
        "validation_approach": "ve_observations 기반 proxy validation (historical PM score 부재로 인한 한계)",
        "provisional_proxy_definition": (
            "Classification ∈ {NEUTRAL, CONSOLIDATION, RECOVERY, PULLBACK} (bullish PM-stage) "
            "AND score_composite ∈ [35, 55) (almost eligible)"
        ),
        "comparison_tiers": [
            "EligibleMomentum: eligible=True + bullish momentum classification (정상 Momentum 탭)",
            "ProvisionalPM_proxy: PM-stage classification + composite 35-55 (조기 entry 후보)",
            "PreMomentum: PM-stage classification (composite 무관)",
            "Excluded: bearish classification",
        ],
        "windows_days": [5, 21, 63, 126, 252],
        "metrics_per_window": [
            "fwd_avg: average forward return %",
            "fwd_bench: benchmark (SPY) return for same window",
            "fwd_excess: tier average minus benchmark",
            "fwd_pos_pct: % of observations with positive forward return",
            "fwd_stdev: std-dev of returns",
            "fwd_sharpe_like: annualized risk-adjusted (avg / stdev × √(252/window_days))",
        ],
        "note": (
            "Real PROVISIONAL (PM score ≥ 45 + agreement ≥ 0.6) tracking requires forward-tracking "
            "from today. This proxy approximates the early-entry tier using available data."
        ),
    }

    return _clean_dict({
        "tier_summary": tier_summary,
        "promotion_stats": promotion_stats,
        "methodology": methodology,
        "summary": {
            "total_observations": len(ve_obs),
            "tiers_with_data": sum(1 for r in tier_summary if r["n_observations"] > 0),
        },
    })


# ─────────────────────────────────────────────────────────────────────────────
# Sector-Segmented Price Discovery (New2) — Validation endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/new-pd-v2/validation")
def new_pd_v2_validation():
    """SVE-style validation for sector-segmented Price Discovery (New2).

    Proxy tier definitions (using ve_obs which has classification + composite + sector):
      - SectorTopN_proxy: ticker가 해당 eval_date의 동일 sector 내 top-N (by composite)
                            + bullish classification + composite ≥ 40
      - UniverseEligible: 기존 eligible=True flag
      - BothEligible: 두 조건 모두 충족
      - SectorOnly_proxy: sector top-N이나 universe eligible 아님
      - UniverseOnly: eligible=True이나 sector top-N 아님

    Sector top-N의 forward return + win-rate를 universe eligible과 비교.
    """
    ve_obs = STATE.get("ve_stats", {}).get("observations", []) or []
    if not ve_obs:
        return {"error": "No ve_observations available"}

    # Need sector column on observations. ve_obs is enriched with cyclical_tag/region/style
    # but sector itself is on STATE["df"]. Build a ticker -> sector map.
    sector_map = {}
    df = STATE.get("df")
    if df is not None and not df.empty and "sector" in df.columns:
        for _, row in df.iterrows():
            tk = row.get("ticker", "")
            sec = row.get("sector", "")
            if tk and sec:
                sector_map[tk] = sec

    BULLISH_CLS = {
        "🟢 CONTINUATION", "🔵 FORMATION", "🟦 LAGGING_CATCHUP",
        "🔵 RECOVERY", "🟡 OVEREXTENDED",
    }
    SKIP_SECTORS = {"Fixed Income", "Macro", "Multi-Asset", "Alternatives"}
    TOP_N = 5
    MIN_COMP = 40.0

    # Group ve_obs by eval_date × sector and compute top-N within each cell
    from collections import defaultdict
    import statistics as _stat

    # eval_date -> sector -> list of obs (with composite filtered for bullish + min)
    by_date_sec: dict = defaultdict(lambda: defaultdict(list))
    for o in ve_obs:
        if not isinstance(o, dict): continue
        tk = o.get("ticker", "")
        sec = sector_map.get(tk)
        if not sec or sec in SKIP_SECTORS:
            continue
        cls = o.get("classification", "")
        comp = o.get("score_composite", 0) or 0
        if cls in BULLISH_CLS and comp >= MIN_COMP:
            by_date_sec[o.get("eval_date", "")][sec].append(o)

    # For each (date, sector), mark top-N
    sector_top_obs_keys = set()  # (eval_date, ticker) for membership lookup
    for eval_date, sec_dict in by_date_sec.items():
        for sec, obs_list in sec_dict.items():
            sorted_obs = sorted(obs_list, key=lambda x: -(x.get("score_composite", 0) or 0))
            for o in sorted_obs[:TOP_N]:
                sector_top_obs_keys.add((eval_date, o.get("ticker", "")))

    # Now tier-classify each observation
    def _tier_v2(o):
        cls = o.get("classification", "")
        comp = o.get("score_composite", 0) or 0
        elig = o.get("eligible", False)
        key = (o.get("eval_date", ""), o.get("ticker", ""))
        tk = o.get("ticker", "")
        sec = sector_map.get(tk)
        # Sector top-N membership
        is_sector_top = key in sector_top_obs_keys
        is_univ_elig = bool(elig and cls in BULLISH_CLS)
        if is_univ_elig and is_sector_top:
            return "BothEligible"
        if is_univ_elig:
            return "UniverseOnly"
        if is_sector_top:
            return "SectorOnly_proxy"
        # Other categories
        if cls in BULLISH_CLS and comp >= MIN_COMP:
            return "OtherBullish"
        return "OtherExcluded"

    # Aggregate forward returns per tier
    tier_buckets: dict = defaultdict(lambda: {
        "n": 0,
        "fwd_returns": {5: [], 21: [], 63: [], 126: [], 252: []},
        "bench_returns": {5: [], 21: [], 63: [], 126: [], 252: []},
        "fwd_pos_count": {5: 0, 21: 0, 63: 0, 126: 0, 252: 0},
        "fwd_total_count": {5: 0, 21: 0, 63: 0, 126: 0, 252: 0},
    })
    for o in ve_obs:
        if not isinstance(o, dict): continue
        tk = o.get("ticker", "")
        if sector_map.get(tk, "") in SKIP_SECTORS:
            continue
        tier = _tier_v2(o)
        tier_buckets[tier]["n"] += 1
        fr = o.get("fwd_rets") or {}
        fb = o.get("fwd_bench") or {}
        for w in (5, 21, 63, 126, 252):
            v = fr.get(w) if isinstance(fr, dict) else None
            if v is not None and isinstance(v, (int, float)) and v == v:
                tier_buckets[tier]["fwd_returns"][w].append(float(v))
                tier_buckets[tier]["fwd_total_count"][w] += 1
                if v > 0:
                    tier_buckets[tier]["fwd_pos_count"][w] += 1
            vb = fb.get(w) if isinstance(fb, dict) else None
            if vb is not None and isinstance(vb, (int, float)) and vb == vb:
                tier_buckets[tier]["bench_returns"][w].append(float(vb))

    tier_summary = []
    for tier in ["BothEligible", "UniverseOnly", "SectorOnly_proxy", "OtherBullish", "OtherExcluded"]:
        d = tier_buckets[tier]
        row = {"tier": tier, "n_observations": d["n"]}
        for w in (5, 21, 63, 126, 252):
            rets = d["fwd_returns"][w]
            bench = d["bench_returns"][w]
            n = d["fwd_total_count"][w]
            pos = d["fwd_pos_count"][w]
            avg = _stat.mean(rets) if rets else None
            avg_bench = _stat.mean(bench) if bench else None
            excess = (avg - avg_bench) if avg is not None and avg_bench is not None else None
            stdev = _stat.stdev(rets) if len(rets) > 1 else None
            row[f"fwd_{w}d_avg"] = round(avg, 3) if avg is not None else None
            row[f"fwd_{w}d_bench"] = round(avg_bench, 3) if avg_bench is not None else None
            row[f"fwd_{w}d_excess"] = round(excess, 3) if excess is not None else None
            row[f"fwd_{w}d_stdev"] = round(stdev, 3) if stdev is not None else None
            row[f"fwd_{w}d_n"] = n
            row[f"fwd_{w}d_pos_pct"] = round((pos / n) * 100, 1) if n > 0 else None
            row[f"fwd_{w}d_sharpe_like"] = (
                round(avg / stdev * ((252 / w) ** 0.5), 3)
                if (avg is not None and stdev is not None and stdev > 0) else None
            )
        tier_summary.append(row)

    # Sector coverage stats — current snapshot
    current_picks_by_sector: dict = defaultdict(list)
    if df is not None and "sector_segmented_eligible" in df.columns:
        eligible_now = df[df["sector_segmented_eligible"].fillna(False)]
        for _, row in eligible_now.iterrows():
            sec = row.get("sector", "Other")
            current_picks_by_sector[sec].append({
                "ticker": row.get("ticker"),
                "name": row.get("name"),
                "composite": float(row.get("composite") or 0),
                "classification": row.get("classification"),
                "sector_rank": row.get("sector_rank"),
            })
    sector_coverage = sorted([
        {"sector": sec, "n_picks": len(items), "picks": items}
        for sec, items in current_picks_by_sector.items()
    ], key=lambda x: -x["n_picks"])

    methodology = {
        "validation_approach": "ve_observations 기반 sector-segmented proxy validation",
        "sector_top_proxy_definition": (
            f"각 eval_date의 (sector, bullish classification, composite ≥ {MIN_COMP}) "
            f"종목들 중 composite 기준 top-{TOP_N}"
        ),
        "tiers": [
            "BothEligible: universe-eligible + sector top-N (가장 강한 합의)",
            "UniverseOnly: universe-eligible이나 sector top-N 미달 (sector 내 mediocre)",
            "SectorOnly_proxy: sector top-N이나 universe-eligible 아님 (sector best지만 전체 평균 이하)",
            "OtherBullish: bullish but neither (남은 bullish 종목)",
            "OtherExcluded: bearish or weak",
        ],
        "top_per_sector": TOP_N,
        "min_composite": MIN_COMP,
        "skip_sectors": list(SKIP_SECTORS),
        "windows_days": [5, 21, 63, 126, 252],
        "note": (
            "SectorOnly가 universe-only보다 forward return 우수하면 → sector-segmented가 alpha 추가. "
            "비슷하거나 낮으면 → 단순 diversification 효과만, lag 단축 안 됨."
        ),
    }

    return _clean_dict({
        "tier_summary": tier_summary,
        "sector_coverage": sector_coverage,
        "methodology": methodology,
        "summary": {
            "total_observations": len(ve_obs),
            "tiers_with_data": sum(1 for r in tier_summary if r["n_observations"] > 0),
            "current_total_picks": sum(s["n_picks"] for s in sector_coverage),
            "sectors_covered": len(sector_coverage),
        },
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


# ─────────────────────────────────────────────────────────────────────────
# Multi-Agent ConvictionDebate (cache-serve only — no LLM call here)
# ─────────────────────────────────────────────────────────────────────────
# The cache is written by an in-session Claude Code run (see agents/
# conviction_debate.py + agents/README.md). This server endpoint does no
# LLM I/O; it just reads the JSON and returns it. All LLM spend happens
# inside the Claude Max plan, never via this server.

@app.get("/api/conviction-debate/multi")
def get_conviction_debate_multi():
    """Return Multi-Agent ConvictionDebate — selection auto-refreshed on every call.

    SELECTION (always fresh from current scan_cache):
      Top 5 stocks LONG + Top 5 ETFs LONG  (highest BuyScore — buy candidates)
      Top 5 stocks SHORT + Top 5 ETFs SHORT (lowest BuyScore — sell candidates)
      = 20 tickers total per refresh

    VERDICTS (from cache, in-session generated):
      Each selected ticker looked up in .multi_agent_debate_cache.json.
      Present → full multi-agent verdict (3-specialist + dual synthesis).
      Missing → placeholder verdict ("pending sub-agent debate").

    => Live scan refresh updates the ticker list immediately; verdicts
       require a subsequent in-session Claude Code run to populate.
       Server makes NO LLM calls.
    """
    try:
        from agents.multi_round_debate import (
            load_multi_verdicts, freshness_minutes,
        )
        from agents.buyscore import top_buy_picks_split, top_sell_picks_split
        import re

        # ---- 1. Build live selection from current scan_cache (always fresh) ----
        results = STATE.get("results") or []
        if not results:
            return {"last_update": None, "stale_minutes": None, "n_verdicts": 0,
                    "verdicts": [], "selection_source": "scan_cache empty"}

        # Mirror frontend's regime + consensus inject (so BuyScore matches
        # ConvictionPicks exactly)
        regime = STATE.get("regime") or {}
        sec_regime = regime.get("sector_regime") or []
        bull = set(s.get("sector") for s in sorted(
            sec_regime, key=lambda s: -(s.get("pct_bullish", 0)))[:4])
        bear = set(s.get("sector") for s in [
            x for x in sorted(sec_regime, key=lambda s: -(s.get("pct_bearish", 0)))[:3]
            if x.get("pct_bearish", 0) > 30])
        # consensus from quant_strategies (computed by run_quant_strategies)
        try:
            from quant_strategies import run_quant_strategies
            q = run_quant_strategies(results)
        except Exception:
            q = {"strategies": {}}
        cmap = {}
        for k, s in (q.get("strategies") or {}).items():
            if isinstance(s, dict):
                for p in (s.get("picks") or []):
                    if isinstance(p, dict) and p.get("ticker"):
                        cmap[p["ticker"]] = cmap.get(p["ticker"], 0) + 1

        # Normalize results — api.py already adds asset_type / sector
        live_results = []
        for r in results:
            row = dict(r)
            # asset_type from category (STK_ vs others)
            if not row.get("asset_type"):
                cat = row.get("category", "") or ""
                row["asset_type"] = "Stock" if cat.startswith("STK_") else "ETF"
            # clean STK_ prefix from category if present (BuyScore needs it)
            cat = row.get("category", "")
            if isinstance(cat, str) and cat.startswith(("STK_", "EQ_", "FI_", "MA_", "ETF_")):
                row["category"] = cat.split("_", 1)[1]
            live_results.append(row)

        # ---- 2. Load cached verdicts (in-session generated) ----
        cache = load_multi_verdicts()
        cache_meta = cache.pop("_meta", {}) if cache else {}
        cached_tickers = {k for k in cache.keys() if not k.startswith("_")}

        # ── 2026-05 split: Momentum vs Pre-Momentum (hold-period buckets) ──
        # Take literal Top-5 per cell — selection_skipped fires ONLY when one of
        # the Top-5 BuyScore picks lacks a cached verdict, so auto-fill on the
        # Live Scan button reliably converges to skipped=0 in a single cycle.
        # Trade-off: cells with < 5 cached Top-5 picks display fewer rows until
        # auto-fill catches up.
        WIDE_N = 5
        long_split  = top_buy_picks_split(live_results, top_n_stock=WIDE_N, top_n_etf=WIDE_N,
                                          bull_sectors=bull, bear_sectors=bear, consensus_map=cmap)
        short_split = top_sell_picks_split(live_results, top_n_stock=WIDE_N, top_n_etf=WIDE_N,
                                           bull_sectors=bull, bear_sectors=bear, consensus_map=cmap)

        # Filter each bucket to cached-only, keep top 5, track skipped live picks
        FINAL_N = 5
        selection_skipped: list[dict] = []

        def _filter_to_cached(split: dict, side: str) -> None:
            for group in ("momentum", "pre_momentum"):
                bucket = split.get(group) or {}
                for asset_key, asset_label in (("stocks", "stock"), ("etfs", "etf")):
                    items = bucket.get(asset_key) or []
                    cached_items, skipped = [], []
                    for s in items:
                        (cached_items if s["ticker"] in cached_tickers else skipped).append(s)
                    # Top-FINAL_N cached (rank order preserved from selector)
                    bucket[asset_key] = cached_items[:FINAL_N]
                    # Record uncached live Top-FINAL_N as "would-have-been" picks
                    import math as _math
                    for s in skipped[:FINAL_N]:
                        rs = s.get("rank_score", 0) or 0
                        if not _math.isfinite(rs):
                            rs = 0.0
                        selection_skipped.append({
                            "ticker": s["ticker"], "side": side, "group": group,
                            "asset_type": asset_label, "rank_score": round(rs, 2),
                        })

        _filter_to_cached(long_split,  "long")
        _filter_to_cached(short_split, "short")

        # ---- 3. Pull each selected ticker's cached verdict (cache hit guaranteed) ----
        def _from_cache(ticker: str, asset: str, side: str, group: str) -> dict:
            cached = cache.get(ticker) or {}
            # Force live side/group (cache may be stale if ticker moved buckets)
            cached_asset = cached.get("asset_type") or asset
            rounds_out = []
            for r in cached.get("rounds", []):
                def _pass(d: dict) -> dict:
                    return {
                        "rating": d.get("rating", "—"),
                        "confidence": d.get("confidence", 0),
                        "key_points": d.get("key_points", []) or [],
                        "biggest_risk": d.get("biggest_risk", "") or "",
                        "biggest_opportunity": d.get("biggest_opportunity", "") or "",
                        "raw_text": d.get("raw_text", "") or "",
                    }
                rounds_out.append({
                    "round": r.get("round_num", 1),
                    "fundamental": _pass(r.get("fundamental") or {}),
                    "sentiment":   _pass(r.get("sentiment")   or {}),
                    "valuation":   _pass(r.get("valuation")   or {}),
                })
            n = cached.get("synthesis_neutral") or {}
            a = cached.get("synthesis_averse") or {}
            return {
                "ticker": ticker, "tier": cached.get("tier", "A"),
                "asset_type": cached_asset, "side": side, "group": group,
                "rounds": rounds_out,
                "converged_round": cached.get("converged_round", 0),
                "disagreement": cached.get("disagreement", {}),
                "synthesis_neutral": {
                    "rating": n.get("rating", "HOLD"),
                    "position_modifier": n.get("position_modifier", 0),
                    "sizing_recommendation": n.get("sizing_recommendation", ""),
                    "reasoning": n.get("reasoning", ""),
                },
                "synthesis_averse": {
                    "rating": a.get("rating", "HOLD"),
                    "position_modifier": a.get("position_modifier", 0),
                    "sizing_recommendation": a.get("sizing_recommendation", ""),
                    "reasoning": a.get("reasoning", ""),
                },
                "composite_at_time": cached.get("composite_at_time"),
                "classification_at_time": cached.get("classification_at_time"),
                "generated_at": cached.get("generated_at", ""),
            }

        verdicts = []
        group_counts = {"momentum": 0, "pre_momentum": 0}

        def _walk(split: dict, side: str):
            for group in ("momentum", "pre_momentum"):
                bucket = split.get(group) or {}
                for s in bucket.get("stocks", []):
                    verdicts.append(_from_cache(s["ticker"], "stock", side, group))
                    group_counts[group] += 1
                for e in bucket.get("etfs", []):
                    verdicts.append(_from_cache(e["ticker"], "etf", side, group))
                    group_counts[group] += 1

        _walk(long_split,  "long")
        _walk(short_split, "short")

        return {
            "last_update": cache_meta.get("last_update"),
            "stale_minutes": freshness_minutes(),
            "n_verdicts": len(verdicts),
            "n_cached": len(verdicts),
            "n_pending": 0,
            "n_momentum": group_counts["momentum"],
            "n_pre_momentum": group_counts["pre_momentum"],
            "cache_universe": len(cached_tickers),
            "selection_skipped": selection_skipped[:20],  # uncached live picks worth debating next
            "verdicts": verdicts,
            "selection_source": (
                "live BuyScore × classification-group split × cache-filtered "
                "(Top-5 cached per Momentum/Pre-Momentum × Stock/ETF × Long/Short cell, "
                "uncached live picks listed in selection_skipped)"
            ),
        }
    except ImportError as e:
        return {"last_update": None, "stale_minutes": None,
                "n_verdicts": 0, "verdicts": [], "error": f"import: {e}"}
    except Exception as e:
        import traceback
        return {"last_update": None, "stale_minutes": None,
                "n_verdicts": 0, "verdicts": [],
                "error": str(e), "trace": traceback.format_exc()[:500]}


# ─────────────────────────────────────────────────────────────────────────
# Live Scan integration — refresh queue
# ─────────────────────────────────────────────────────────────────────────
# Pattern: the dashboard's Live Scan button triggers /api/scan (Layer 1 quant).
# After scan completes, the frontend calls /api/conviction-debate/refresh-queue
# which detects uncached live picks and writes them to .debate_refresh_queue.json.
# The user then runs `python3 agents/refresh_from_queue.py` from a Claude Code
# session — this dispatches sub-agents (within Max plan), persists verdicts,
# and clears the queue. Server itself NEVER calls LLM.

@app.post("/api/conviction-debate/refresh-queue")
def post_refresh_debate_queue():
    """Detect uncached live picks and write them to .debate_refresh_queue.json.
    Returns queue size + processing instructions. Server makes NO LLM call.
    """
    try:
        from pathlib import Path as _P
        import time as _time
        # Re-use the same selection logic by calling the GET endpoint internally
        d = get_conviction_debate_multi()
        skipped = d.get("selection_skipped") or []
        queue_path = _P(".debate_refresh_queue.json")
        # Enrich each skipped item with quant snapshot
        results = STATE.get("results") or []
        rmap = {r.get("ticker"): r for r in results}
        enriched = []
        import math as _math
        def _clean(v):
            if v is None: return None
            try:
                f = float(v)
                return f if _math.isfinite(f) else None
            except (TypeError, ValueError):
                return v
        for s in skipped:
            r = rmap.get(s["ticker"], {})
            enriched.append({
                **s,
                "name": r.get("name", ""),
                "category": r.get("category", ""),
                "classification": r.get("classification", ""),
                "composite": _clean(r.get("composite")),
                "oer": _clean(r.get("oer")),
                "tcs_short": _clean(r.get("tcs_short")), "tcs_long": _clean(r.get("tcs_long")),
                "tfs_short": _clean(r.get("tfs_short")), "tfs_long": _clean(r.get("tfs_long")),
                "rss_short": _clean(r.get("rss_short")), "rss_long": _clean(r.get("rss_long")),
                "urs": _clean(r.get("urs")),
                "pre_momentum_score": _clean(r.get("pre_momentum_score")),
            })
        payload = {
            "queued_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_queued": len(enriched),
            "tickers": enriched,
            "instructions": (
                "Run `python3 agents/refresh_from_queue.py` from a Claude Code "
                "session to dispatch sub-agents and populate cache. "
                "Server side will not call LLM (Claude Max plan constraint)."
            ),
        }
        queue_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        return {
            "queue_file": ".debate_refresh_queue.json",
            "n_queued": len(enriched),
            "instructions": payload["instructions"],
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[:500]}


@app.get("/api/conviction-debate/refresh-queue")
def get_refresh_debate_queue():
    """Inspect current queue without modifying."""
    from pathlib import Path as _P
    queue_path = _P(".debate_refresh_queue.json")
    if not queue_path.exists():
        return {"queued_at": None, "n_queued": 0, "tickers": []}
    try:
        return json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# Auto-fill — server spawns `claude -p` subprocesses (within Max plan)
# ─────────────────────────────────────────────────────────────────────
# Each `claude -p` invocation runs in the user's authenticated Claude Code
# session and is billed against the Max plan — exactly what sub-agent
# dispatch costs would be. Server itself holds no API keys.

_AUTOFILL_STATUS: dict = {
    "running": False, "started_at": "", "finished_at": "",
    "n_total": 0, "n_completed": 0, "n_failed": 0, "n_persisted": 0,
    "last_error": "", "current": "", "errors": [],
}

_CLAUDE_BIN = None  # resolved lazily


def _find_claude() -> Optional[str]:
    """Return path to the `claude` CLI binary or None if not on PATH."""
    global _CLAUDE_BIN
    if _CLAUDE_BIN is not None:
        return _CLAUDE_BIN or None
    import shutil
    p = shutil.which("claude")
    _CLAUDE_BIN = p or ""
    return p


def _extract_json_fence(text: str) -> Optional[dict]:
    """Pull a {...} JSON object out of a ```json fence or first {...} block."""
    import re as _re
    # Try fenced block first
    m = _re.search(r"```json\s*(\{.*?\})\s*```", text, _re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    # Fallback: greedy first {...} block (matched braces)
    m = _re.search(r"\{.*\}", text, _re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return None


def _normalize_verdict(raw: dict, ticker: str, asset: str, side: str, group: str,
                       composite: float, classification: str) -> dict:
    """Coerce a free-form claude-p verdict into the cache schema."""
    import time as _t
    # Best-effort field extraction
    rounds = raw.get("rounds") or []
    r1 = rounds[0] if rounds else {}
    f = (r1.get("fundamental") or {})
    s = (r1.get("sentiment") or {})
    v = (r1.get("valuation") or {})
    n = raw.get("synthesis_neutral") or {}
    a = raw.get("synthesis_averse")  or {}
    dis = raw.get("disagreement") or {}

    def _rt(d: dict, default: str) -> str:
        r = d.get("rating") or d.get("verdict") or d.get("stance") or default
        return str(r).upper().replace(" ", "_").replace("-", "_")

    def _mod(d: dict) -> int:
        for k in ("position_modifier", "modifier", "final_modifier", "modifier_applied"):
            if k in d and d[k] is not None:
                try: return int(round(float(d[k])))
                except Exception: pass
        return 0

    def _reason(d: dict) -> str:
        return str(d.get("reasoning") or d.get("thesis") or d.get("rationale")
                   or d.get("summary") or "")[:4000]

    def _sizing(d: dict) -> str:
        return str(d.get("sizing_recommendation") or d.get("sizing_guidance")
                   or d.get("entry_plan") or "")[:1000]

    def _kp(d: dict) -> list:
        for k in ("key_points","key_drivers","points","drivers"):
            val = d.get(k)
            if isinstance(val, list) and val:
                return [str(x)[:300] for x in val[:8]]
        return []

    def _risk(d: dict) -> str:
        for k in ("biggest_risk","risk","key_risks","risks"):
            val = d.get(k)
            if isinstance(val, str) and val: return val[:500]
            if isinstance(val, list) and val: return "; ".join(str(x) for x in val[:3])[:500]
        return ""

    def _opp(d: dict) -> str:
        for k in ("biggest_opportunity","opportunity","key_opportunities","catalysts"):
            val = d.get(k)
            if isinstance(val, str) and val: return val[:500]
            if isinstance(val, list) and val: return "; ".join(str(x) for x in val[:3])[:500]
        return ""

    def _raw(d: dict) -> str:
        for k in ("raw_text","thesis","narrative","summary"):
            val = d.get(k)
            if isinstance(val, str) and val: return val[:1500]
        return ""

    return {
        "ticker": ticker, "tier": "A",
        "asset_type": asset, "side": side, "group": group,
        "rounds": [{
            "round_num": 1,
            "fundamental": {"persona":"fundamental","rating":_rt(f,"HOLD"),"confidence":float(f.get("confidence",0.6) or 0.6),"key_points":_kp(f),"biggest_risk":_risk(f),"biggest_opportunity":_opp(f),"raw_text":_raw(f) or "[archived from claude -p]","narrative_summary":"","critique":""},
            "sentiment":   {"persona":"sentiment","rating":_rt(s,"HOLD"),"confidence":float(s.get("confidence",0.6) or 0.6),"key_points":_kp(s),"biggest_risk":_risk(s),"biggest_opportunity":_opp(s),"raw_text":_raw(s) or "[archived from claude -p]","narrative_summary":"","critique":""},
            "valuation":   {"persona":"valuation","rating":_rt(v,"HOLD"),"confidence":float(v.get("confidence",0.6) or 0.6),"key_points":_kp(v),"biggest_risk":_risk(v),"biggest_opportunity":_opp(v),"raw_text":_raw(v) or "[archived from claude -p]","narrative_summary":"","critique":""},
            "notes": ""
        }],
        "synthesis_neutral": {
            "risk_mode":"neutral","rating":_rt(n,"HOLD"),"position_modifier":_mod(n),
            "sizing_recommendation":_sizing(n),"reasoning":_reason(n),
            "raw_text":"[rule-based neutral synthesis]"
        },
        "synthesis_averse": {
            "risk_mode":"averse","rating":_rt(a,"HOLD"),"position_modifier":_mod(a),
            "sizing_recommendation":_sizing(a),"reasoning":_reason(a),
            "raw_text":"[rule-based averse synthesis]"
        },
        "converged_round": int(raw.get("converged_round", 1) or 1),
        "disagreement": {
            "rating_axis": int(dis.get("rating_axis", 1) or 1),
            "specialist_dispersion": float(dis.get("specialist_dispersion", 0.15) or 0.15),
            "type": str(dis.get("type", "ENTRY_TIMING") or "ENTRY_TIMING"),
        },
        "composite_at_time": composite,
        "classification_at_time": classification,
        "generated_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _build_dispatch_prompt(item: dict) -> str:
    """Build the same prompt as agents/refresh_from_queue.py for consistency."""
    try:
        from agents.refresh_from_queue import build_prompt as _bp
        return _bp(item)
    except Exception:
        # Inline fallback
        return f"""Multi-Agent ConvictionDebate Round 1 + dual synthesis for **{item.get('ticker','?')}** in **{item.get('group','momentum')} {item.get('side','long').upper()}** bucket.
Composite: {item.get('composite',0)} | Classification: {item.get('classification','?')}
OER: {item.get('oer',0)} | TCS_s/l: {item.get('tcs_short',0)}/{item.get('tcs_long',0)} | TFS_s/l: {item.get('tfs_short',0)}/{item.get('tfs_long',0)} | RSS_s/l: {item.get('rss_short',0)}/{item.get('rss_long',0)} | URS: {item.get('urs',0)}
Return strict JSON in ```json fence with: rounds[0].fundamental/sentiment/valuation each with rating/confidence; synthesis_neutral and synthesis_averse each with rating/position_modifier/sizing_recommendation/reasoning; disagreement with type."""


def _run_one_claude(item: dict, timeout: int = 180, retries: int = 3) -> dict:
    """Invoke `claude -p` for a single ticker; retry on transient 529/rate-limit.
    Exponential backoff: 4s, 12s, 36s.
    """
    import subprocess as _sp, time as _t
    bin_path = _find_claude()
    if not bin_path:
        raise RuntimeError("claude CLI not on PATH")
    prompt = _build_dispatch_prompt(item)

    last_err = ""
    for attempt in range(retries + 1):
        proc = _sp.run(
            [bin_path, "-p"],
            input=prompt, capture_output=True, text=True,
            timeout=timeout,
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        transient = (
            "overloaded_error" in stdout
            or "rate_limit" in stdout.lower()
            or "529" in stdout[:200]
            or "503" in stdout[:200]
        )
        if "API Error:" in stdout or transient:
            err_line = next((ln for ln in stdout.splitlines() if "Error" in ln or "error" in ln),
                            stdout[:200])
            last_err = f"anthropic api error: {err_line[:300]}"
            if transient and attempt < retries:
                _t.sleep(4 * (3 ** attempt))   # 4s, 12s, 36s
                continue
            raise RuntimeError(last_err)
        if proc.returncode != 0:
            raise RuntimeError(f"claude exit {proc.returncode}: stderr={stderr[:200]} stdout={stdout[:200]}")
        raw = _extract_json_fence(stdout)
        if not raw:
            raise RuntimeError(f"no JSON fence in output: {stdout[:300]}")
        break  # success
    else:
        raise RuntimeError(last_err or "max retries exceeded")
    return _normalize_verdict(
        raw, item["ticker"], item.get("asset_type","stock"),
        item.get("side","long"), item.get("group","momentum"),
        float(item.get("composite") or 0),
        str(item.get("classification") or ""),
    )


@app.post("/api/conviction-debate/auto-fill")
def post_autofill_debate_cache(max_workers: int = 4, timeout_sec: int = 180):
    """Server-side dispatch: spawn `claude -p` subprocesses for every queued ticker.

    Each subprocess invokes the user's authenticated Claude Code CLI, so cost is
    billed against the Max plan (no API key, no out-of-plan charge). Runs in a
    background thread; poll /api/conviction-debate/auto-fill/status for progress.
    """
    import threading, concurrent.futures as _futures, time as _t
    from pathlib import Path as _P

    if _AUTOFILL_STATUS["running"]:
        return {"status": "already_running", **_AUTOFILL_STATUS}
    if not _find_claude():
        return {"status": "no_claude_cli",
                "hint": "Install Claude Code CLI and authenticate to enable auto-fill."}

    # Rebuild queue from current selection first
    try:
        post_refresh_debate_queue()
    except Exception:
        pass
    qpath = _P(".debate_refresh_queue.json")
    if not qpath.exists():
        return {"status": "no_queue"}
    queue = json.loads(qpath.read_text(encoding="utf-8"))
    items = queue.get("tickers") or []
    if not items:
        return {"status": "empty_queue", "n_total": 0}

    # Reset status
    _AUTOFILL_STATUS.update({
        "running": True, "started_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": "", "n_total": len(items),
        "n_completed": 0, "n_failed": 0, "n_persisted": 0,
        "last_error": "", "current": "", "errors": [],
    })

    def _worker():
        from agents.multi_round_debate import load_multi_verdicts, save_multi_verdict, MultiAgentVerdict
        cache_path = _P(".multi_agent_debate_cache.json")
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        try:
            with _futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {ex.submit(_run_one_claude, it, timeout_sec): it for it in items}
                for fut in _futures.as_completed(fut_map):
                    item = fut_map[fut]
                    ticker = item.get("ticker", "?")
                    _AUTOFILL_STATUS["current"] = ticker
                    try:
                        verdict = fut.result()
                        cache[ticker] = verdict
                        _AUTOFILL_STATUS["n_completed"] += 1
                        _AUTOFILL_STATUS["n_persisted"] += 1
                    except Exception as e:
                        _AUTOFILL_STATUS["n_failed"] += 1
                        _AUTOFILL_STATUS["errors"].append(f"{ticker}: {type(e).__name__}: {str(e)[:200]}")
                        if len(_AUTOFILL_STATUS["errors"]) > 25:
                            _AUTOFILL_STATUS["errors"] = _AUTOFILL_STATUS["errors"][-25:]
            cache["_meta"] = {
                "last_update": _t.strftime("%Y-%m-%dT%H:%M:%S"),
                "n_verdicts": len([k for k in cache if not k.startswith("_")]),
                "tier": "A",
            }
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
            # Clear queue
            if qpath.exists(): qpath.unlink()
        except Exception as e:
            _AUTOFILL_STATUS["last_error"] = f"{type(e).__name__}: {e}"
        finally:
            _AUTOFILL_STATUS["running"] = False
            _AUTOFILL_STATUS["finished_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S")

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "n_total": len(items)}


@app.get("/api/conviction-debate/auto-fill/status")
def get_autofill_status():
    return dict(_AUTOFILL_STATUS)


# ─────────────────────────────────────────────────────────────────────
# Market Leaders 6-agent swarm — Phase 1 (4 parallel) + Phase 2 + Phase 3 dual
# ─────────────────────────────────────────────────────────────────────

_SWARM_STATUS: dict = {
    "running": False, "started_at": "", "finished_at": "",
    "phase": "", "current": "", "events": [], "last_error": "",
}


@app.post("/api/market-leaders/swarm")
def post_market_leaders_swarm(force: bool = False):
    """Trigger 6-agent Market Leaders swarm. Caches result with 12h TTL.

    Args:
        force: if True, ignore cache and re-run swarm.
    """
    import threading
    import time as _t

    if _SWARM_STATUS["running"]:
        return {"status": "already_running", **_SWARM_STATUS}

    try:
        from agents.market_leaders_swarm import (
            cache_fresh, load_cached, run_swarm,
        )
    except ImportError as e:
        return {"status": "import_error", "error": str(e)}

    if not force and cache_fresh():
        cached = load_cached() or {}
        return {"status": "cached", "generated_at": cached.get("generated_at"),
                "ttl_hours": 12}

    if not _find_claude():
        return {"status": "no_claude_cli"}

    _SWARM_STATUS.update({
        "running": True, "started_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": "", "phase": "phase1", "current": "starting",
        "events": [], "last_error": "",
    })

    def _cb(phase: str, agent: str, status: str):
        _SWARM_STATUS["phase"] = phase
        _SWARM_STATUS["current"] = f"{agent}:{status}"
        _SWARM_STATUS["events"].append({
            "t": _t.strftime("%H:%M:%S"), "phase": phase, "agent": agent, "status": status,
        })
        if len(_SWARM_STATUS["events"]) > 50:
            _SWARM_STATUS["events"] = _SWARM_STATUS["events"][-50:]

    def _worker():
        try:
            run_swarm(progress_cb=_cb)
        except Exception as e:
            _SWARM_STATUS["last_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            _SWARM_STATUS["running"] = False
            _SWARM_STATUS["finished_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S")

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


@app.get("/api/market-leaders/swarm/status")
def get_swarm_status():
    return dict(_SWARM_STATUS)


@app.get("/api/backtest/results")
def get_backtest_results():
    """Return cached PM proxy backtest results."""
    from pathlib import Path as _P
    p = _P("backtest/results.json")
    if not p.exists():
        return {"available": False}
    try:
        return {"available": True, "result": json.loads(p.read_text(encoding="utf-8"))}
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.post("/api/backtest/run")
def trigger_backtest_run():
    """Re-run the PM proxy backtest in a background thread."""
    import threading, subprocess, sys, time as _t
    _BACKTEST_STATUS = globals().setdefault("_BACKTEST_STATUS", {
        "running": False, "started_at": "", "finished_at": "", "last_error": ""})
    if _BACKTEST_STATUS["running"]:
        return {"status": "already_running", **_BACKTEST_STATUS}

    def _worker():
        _BACKTEST_STATUS["running"] = True
        _BACKTEST_STATUS["started_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S")
        _BACKTEST_STATUS["last_error"] = ""
        try:
            proc = subprocess.run([sys.executable, "backtest/run.py"],
                                  capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                _BACKTEST_STATUS["last_error"] = (proc.stderr or "")[:500]
        except Exception as e:
            _BACKTEST_STATUS["last_error"] = str(e)
        finally:
            _BACKTEST_STATUS["running"] = False
            _BACKTEST_STATUS["finished_at"] = _t.strftime("%Y-%m-%dT%H:%M:%S")

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started"}


@app.get("/api/backtest/status")
def get_backtest_status():
    return dict(globals().get("_BACKTEST_STATUS", {"running": False}))


# ─────────────────────────────────────────────────────────────────────
# PM Picks History (forward collection)
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/pm-history/summary")
def get_pm_history_summary():
    """High-level stats — n snapshots, date range, top persistent tickers."""
    try:
        from agents.pm_history import summarize_history
        return summarize_history()
    except Exception as e:
        return {"error": str(e), "n_snapshots": 0}


@app.get("/api/final-list")
def get_final_list():
    """Buy/Sell Final List — synthesizes PM Swarm + Phase 5.5 + Phase 5.6 + Backtest.

    Returns:
      buy_list:  LONG candidates ranked by confidence (★★★ down to ★)
      sell_list: SHORT candidates + existing positions to close

    Each buy_list pick is augmented with Elliott Wave-based stop loss:
      stop_price, stop_pct, stop_type, stop_rationale, stop_wave_guess
    (cached daily in .elliott_stops_cache.json — first call after refresh
     takes ~30s for 50 picks via yfinance; subsequent calls use cache.)
    """
    try:
        from agents.final_list import build_final_lists
        result = build_final_lists()
        # All row sources that surface in the unified frontend table need entry/stop
        # annotations — not just buy_list. HOLDING/ENTERED rows come from
        # active_positions, and EXIT_PENDING rows from exit_pending. Annotating all
        # three ensures no row shows empty 진입가/손절가 columns.
        # Dedupe by (ticker, horizon): the annotate functions key their cache the same
        # way, so a ticker held AND in buy_list is computed once and the shared cache
        # serves the duplicate — but we still pass all dict refs so each gets mutated.
        all_annotatable = []
        for key in ("buy_list", "active_positions", "exit_pending"):
            all_annotatable.extend(result.get(key) or [])

        # ─── Annotate with Elliott Wave stop-loss prices ───
        try:
            from agents.elliott_wave_stops import annotate_buy_list_with_stops
            if all_annotatable:
                annotate_buy_list_with_stops(all_annotatable, use_cache=True)
        except Exception as ew_err:
            for r in all_annotatable:
                r.setdefault("stop_price", None)
                r.setdefault("stop_pct", None)
                r.setdefault("stop_type", "UNAVAILABLE")
                r.setdefault("stop_rationale", f"elliott_wave error: {str(ew_err)[:120]}")
        # ─── Annotate with 3-tier Entry prices (CAN SLIM + Elliott + SMA50) ───
        try:
            from agents.entry_price import annotate_buy_list_with_entries
            if all_annotatable:
                annotate_buy_list_with_entries(all_annotatable, use_cache=True)
        except Exception as ep_err:
            for r in all_annotatable:
                r.setdefault("entry_aggressive", None)
                r.setdefault("entry_primary", None)
                r.setdefault("entry_conservative", None)
                r.setdefault("entry_skip_reason", f"entry_price error: {str(ep_err)[:120]}")
        # JSON-safe sanitization (drop NaN/Inf in stop_price/stop_pct across all rows)
        import math as _math
        for r in all_annotatable:
            for k in ("stop_price", "stop_pct"):
                v = r.get(k)
                if v is not None:
                    try:
                        v = float(v)
                        if not _math.isfinite(v): r[k] = None
                    except (TypeError, ValueError):
                        r[k] = None
        return result
    except Exception as e:
        return {"error": str(e), "buy_list": [], "sell_list": [], "metadata": {}}


@app.post("/api/final-list/refresh-entries")
def refresh_entry_prices():
    """Force refresh CAN SLIM + Elliott entry prices cache for current buy_list.

    Bypasses 24h TTL cache and fetches fresh OHLCV from yfinance.
    Returns count + timing.
    """
    import time as _t
    try:
        from agents.final_list import build_final_lists
        from agents.entry_price import annotate_buy_list_with_entries
        t0 = _t.time()
        fl = build_final_lists()
        buy = fl.get("buy_list") or []
        annotate_buy_list_with_entries(buy, use_cache=False)
        return {
            "ok": True,
            "n_buy_list": len(buy),
            "n_with_primary":   sum(1 for r in buy if r.get("entry_primary")),
            "n_with_aggressive": sum(1 for r in buy if r.get("entry_aggressive")),
            "n_with_conservative": sum(1 for r in buy if r.get("entry_conservative")),
            "elapsed_sec": round(_t.time() - t0, 1),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


@app.post("/api/final-list/refresh-stops")
def refresh_elliott_stops():
    """Force refresh Elliott Wave stop-loss cache for current buy_list.

    Use when you want to recompute stops (e.g., after big price move).
    Bypasses cache TTL and fetches fresh OHLCV from yfinance.
    Returns count + timing.
    """
    import time as _t
    try:
        from agents.final_list import build_final_lists
        from agents.elliott_wave_stops import annotate_buy_list_with_stops
        t0 = _t.time()
        fl = build_final_lists()
        buy = fl.get("buy_list") or []
        annotate_buy_list_with_stops(buy, use_cache=False)   # force fresh
        return {
            "ok": True,
            "n_buy_list": len(buy),
            "n_with_stops": sum(1 for r in buy if r.get("stop_price")),
            "elapsed_sec": round(_t.time() - t0, 1),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


@app.get("/api/validated-extra-timeline")
def get_validated_extra_timeline():
    """Synthesize backtest-style records for ★★+ validated tickers MISSING from backtest data.

    Used by frontend to extend the '검증된 매수 종목' timeline visualization
    so PM-validated stocks (which proxy backtest didn't pick) also get a timeline.

    Returns same schema as trading_lifecycles_compact (per horizon + bucket).
    Each record contains backtest-style fields:
      t, n, d, sig, mst, mr (always null for synthetic), bh, bhp, bhd
    """
    try:
        import pickle, json
        from pathlib import Path
        import pandas as pd
        from agents.final_list import build_final_lists

        # Load backtest results + price cache
        bt_path = Path("backtest/results.json")
        if not bt_path.exists():
            return {"tactical": {"long_stocks": [], "long_etfs": []},
                    "core":     {"long_stocks": [], "long_etfs": []},
                    "strategic":{"long_stocks": [], "long_etfs": []}}
        bt = json.loads(bt_path.read_text(encoding="utf-8"))
        tlc = bt.get("trading_lifecycles_compact", {})
        core = tlc.get("core", {})

        # Existing tickers in backtest (per bucket)
        existing_stocks = {r["t"] for r in core.get("long_stocks", [])}
        existing_etfs   = {r["t"] for r in core.get("long_etfs",   [])}

        # All cohort dates from existing data
        all_dates = sorted({r["d"] for r in core.get("long_stocks", [])} |
                           {r["d"] for r in core.get("long_etfs",   [])})

        # Find ★★+ tickers NOT in existing data
        fl = build_final_lists()
        missing_stocks: list[tuple[str,str]] = []
        missing_etfs:   list[tuple[str,str]] = []
        for r in fl.get("buy_list", []):
            if r.get("stars", 0) < 2: continue
            t = r["ticker"]
            name = r.get("name", "") or ""
            if "stocks" in r.get("bucket", "") and t not in existing_stocks:
                missing_stocks.append((t, name))
            elif "etfs" in r.get("bucket", "") and t not in existing_etfs:
                missing_etfs.append((t, name))

        if not missing_stocks and not missing_etfs:
            return {"tactical": {"long_stocks": [], "long_etfs": []},
                    "core":     {"long_stocks": [], "long_etfs": []},
                    "strategic":{"long_stocks": [], "long_etfs": []}}

        # Load price cache
        cache_path = Path(".backtest_price_cache.pkl")
        if not cache_path.exists():
            return {"error": "no price cache", "core": {"long_stocks": [], "long_etfs": []}}
        cache = pickle.load(open(cache_path, "rb"))
        data = cache.get("data", {})

        HORIZON_DAYS = {"tactical": 5, "core": 21, "strategic": 63}

        def _compute(tickers_with_names: list[tuple[str,str]], horizon_days: int) -> list[dict]:
            recs = []
            for t, name in tickers_with_names:
                df = data.get(t)
                if df is None or df.empty: continue
                for d_str in all_dates:
                    entry_date = pd.Timestamp(d_str)
                    hist = df[df.index <= entry_date]
                    if len(hist) == 0: continue
                    entry_close = float(hist["Close"].iloc[-1])
                    if entry_close <= 0: continue

                    fwd = df[df.index > entry_date]
                    # Full horizon return
                    bh_ret = None
                    if len(fwd) >= horizon_days:
                        exit_close = float(fwd["Close"].iloc[horizon_days - 1])
                        if exit_close > 0:
                            bh_ret = exit_close / entry_close - 1
                    # Partial return (MTM at latest)
                    bhp = None
                    bhd = 0
                    if len(fwd) > 0:
                        last_close = float(fwd["Close"].iloc[-1])
                        if last_close > 0:
                            bhp = last_close / entry_close - 1
                            bhd = len(fwd)
                    recs.append({
                        "t":  t, "n": (name or "")[:40],
                        "d":  d_str,
                        "sig": "—",
                        "mst": "SYNTHETIC",
                        "mex": None,
                        "mr":  None,  # No proxy/managed for synthetic
                        "mdh": None, "mdt": None,
                        "bh":  bh_ret,
                        "bhp": bhp,
                        "bhd": bhd,
                    })
            return recs

        return {
            h: {
                "long_stocks": _compute(missing_stocks, HORIZON_DAYS[h]),
                "long_etfs":   _compute(missing_etfs,   HORIZON_DAYS[h]),
            }
            for h in ("tactical", "core", "strategic")
        }
    except Exception as e:
        return {"error": str(e),
                "tactical": {"long_stocks": [], "long_etfs": []},
                "core":     {"long_stocks": [], "long_etfs": []},
                "strategic":{"long_stocks": [], "long_etfs": []}}


@app.get("/api/pm-history/list")
def get_pm_history_list(limit: int = 200):
    """Return all snapshots (date + bucket sizes for each)."""
    try:
        from agents.pm_history import load_history
        h = load_history()
        snaps = h.get("snapshots", [])[-limit:]
        # Compact view — only metadata + counts (full picks fetched separately)
        out = []
        for s in snaps:
            p5 = s.get("phase5_picks") or {}
            out.append({
                "date": s.get("date"),
                "snapshot_at": s.get("snapshot_at"),
                "regime_tag": s.get("regime_tag"),
                "synthesis_neutral_regime": s.get("synthesis_neutral_regime"),
                "n_picks": {b: len(p5.get(b) or []) for b in
                            ("long_stocks","long_etfs","short_stocks","short_etfs")},
            })
        return {"snapshots": out, "n_total": len(h.get("snapshots", []))}
    except Exception as e:
        return {"error": str(e), "snapshots": []}


@app.get("/api/pm-history/date/{date}")
def get_pm_history_date(date: str):
    """Return the full snapshot (phase4 + phase5 picks + commentary) for one date."""
    try:
        from agents.pm_history import load_history
        h = load_history()
        for s in h.get("snapshots", []):
            if s.get("date") == date:
                return s
        return {"available": False, "date": date}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# Backtest per-ticker drilldown (Task 4)
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/backtest/ticker/{ticker}")
def get_backtest_ticker(ticker: str):
    """Per-ticker drilldown: all cohort appearances + forward returns + aggregates."""
    from pathlib import Path as _P
    p = _P("backtest/ticker_details.json")
    if not p.exists():
        return {"available": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        rec = (data.get("tickers") or {}).get(ticker)
        if not rec:
            return {"available": False, "ticker": ticker}
        return {"available": True, "ticker": ticker, "data": rec,
                "as_of_run": data.get("as_of_run")}
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/api/backtest/rankings")
def get_backtest_rankings():
    """Top + Worst tickers per bucket by 21d mean alpha."""
    from pathlib import Path as _P
    p = _P("backtest/ticker_rankings.json")
    if not p.exists():
        return {"available": False}
    try:
        return {"available": True, "rankings": json.loads(p.read_text(encoding="utf-8"))}
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/api/market-leaders/swarm/result")
def get_swarm_result():
    """Return cached swarm result (Phase 1 verdicts + Phase 2 + dual synthesis)."""
    try:
        from agents.market_leaders_swarm import load_cached, cache_fresh
        cached = load_cached() or {}
        return {
            "available": bool(cached),
            "fresh": cache_fresh(),
            "result": cached,
        }
    except Exception as e:
        return {"available": False, "fresh": False, "error": str(e)}
