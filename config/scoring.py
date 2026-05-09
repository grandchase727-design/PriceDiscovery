"""Centralized scoring thresholds and weights.

Single source of truth — modules should import from here rather than inlining
literals. Refer to CLAUDE.md "Score architecture" for the meaning of each value.
"""

# ── Eligibility Gate (Layer 5) ─────────────────────────────────────────────
ELIGIBLE_COMPOSITE = 55          # Composite ≥ this passes the technical-strength gate
ADV_MIN_USD = 5_000_000          # Liquidity floor: 5-day avg dollar volume
QVR_GATE = 40.0                  # Stocks must have QVR ≥ this; ETFs are exempt

# ── Momentum Composite weights (must sum to 1.0) ──────────────────────────
# Composite = W_TCS·TCS + W_TFS·TFS + W_RSS·RSS + W_URS·URS
COMPOSITE_W_TCS = 0.30
COMPOSITE_W_TFS = 0.25
COMPOSITE_W_RSS = 0.30
COMPOSITE_W_URS = 0.15
