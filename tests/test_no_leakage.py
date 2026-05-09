###############################################################################
# Leakage Sanity Tests
# ============================================================================
# 1. Feature leakage: features at time t must be computable from data ≤ t only.
#    Assertion: build_feature_matrix(panel.loc[:t]).iloc[-1] == full_matrix.loc[t]
# 2. Breadth leakage: compute_breadth_at('2015-06-30') with panel truncated to
#    that date must equal the value from the full-history computation.
# 3. Label-feature overlap audit: warn if |corr(feature, fwd_ret)| > 0.5.
###############################################################################

import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Test 1: feature leakage — as-of slicing must produce identical feature row
# ---------------------------------------------------------------------------
def test_feature_as_of_consistency():
    print("\n[test 1] Feature-level as-of consistency")
    from ml.feature_pipeline import download_macro_panel, build_feature_matrix

    panel = download_macro_panel(start="2005-01-01")
    full = build_feature_matrix(panel, resample_freq="ME")
    # Pick an intermediate date with enough prior history
    test_date = pd.Timestamp("2020-06-30")
    if test_date not in full.index:
        print(f"  SKIP — {test_date.date()} not in index")
        return

    truncated = panel.loc[:test_date]
    sliced = build_feature_matrix(truncated, resample_freq="ME")
    row_truncated = sliced.loc[test_date]
    row_full      = full.loc[test_date]

    diffs = {}
    for col in row_full.index:
        a = row_full[col]
        b = row_truncated[col] if col in row_truncated.index else np.nan
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b) or abs(a - b) > 1e-9:
            diffs[col] = (a, b)

    if diffs:
        print(f"  ✗ FAIL — {len(diffs)} columns differ:")
        for k, (a, b) in list(diffs.items())[:5]:
            print(f"      {k}: full={a}  truncated={b}")
        return False
    print(f"  ✓ PASS — all {len(row_full)} features match at {test_date.date()}")
    return True


# ---------------------------------------------------------------------------
# Test 2: breadth as-of consistency
# ---------------------------------------------------------------------------
def test_breadth_as_of_consistency():
    print("\n[test 2] Breadth as-of consistency")
    from ml.breadth_pipeline import download_universe_cached, compute_breadth_at

    data = download_universe_cached()
    as_of = pd.Timestamp("2015-06-30")
    feats = compute_breadth_at(as_of, data)

    # Manually truncate and recompute — since compute_breadth_at itself slices
    # to <= as_of internally, truncating the data externally shouldn't change
    # the result. But we verify by pre-slicing each ETF and seeing no change.
    class _FakeETF:
        def __init__(self, etf, cutoff):
            self.ticker = etf.ticker
            self.category = etf.category
            self.valid = etf.valid
            self.df = etf.df[etf.df.index <= cutoff]

    truncated = {t: _FakeETF(e, as_of) for t, e in data.items()
                 if e.valid and e.df is not None}
    feats_truncated = compute_breadth_at(as_of, truncated)

    keys = [k for k in feats.keys() if not k.startswith("_")]
    diffs = {}
    for k in keys:
        a = feats.get(k); b = feats_truncated.get(k)
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b) or abs(a - b) > 1e-9:
            diffs[k] = (a, b)
    if diffs:
        print(f"  ✗ FAIL — {len(diffs)} breadth features differ:")
        for k, (a, b) in diffs.items():
            print(f"      {k}: full={a}  truncated={b}")
        return False
    print(f"  ✓ PASS — all {len(keys)} breadth features identical")
    return True


# ---------------------------------------------------------------------------
# Test 3: label-feature correlation audit
# ---------------------------------------------------------------------------
def test_feature_label_correlation():
    print("\n[test 3] Label-feature correlation audit (|corr| > 0.5 warning)")
    df = pd.read_csv("regime_dataset.csv", index_col=0, parse_dates=True).dropna()
    reserved = {"regime", "fwd_ret", "fwd_dd", "fwd_vol", "bond_fwd_ret"}
    feat_cols = [c for c in df.columns if c not in reserved]
    corr = df[feat_cols].corrwith(df["fwd_ret"]).abs().sort_values(ascending=False)
    print(f"  Top 10 |corr(feature, fwd_ret)|:")
    for k, v in corr.head(10).items():
        flag = "  ⚠️" if v > 0.5 else ""
        print(f"    {k:<26} {v:.3f}{flag}")
    if (corr > 0.5).any():
        print(f"  ⚠️ {int((corr > 0.5).sum())} features exceed 0.5 threshold — investigate")
        return False
    print("  ✓ PASS — no feature exceeds 0.5 correlation with label")
    return True


if __name__ == "__main__":
    results = {
        "feature_as_of":    test_feature_as_of_consistency(),
        "breadth_as_of":    test_breadth_as_of_consistency(),
        "label_correlation": test_feature_label_correlation(),
    }
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    for k, v in results.items():
        status = "PASS" if v else "FAIL/WARN"
        print(f"  {k:<25} {status}")
    all_pass = all(results.values())
    sys.exit(0 if all_pass else 1)
