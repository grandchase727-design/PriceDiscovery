"""Endpoints whose responses we freeze as the regression baseline.

Picked for: deterministic from cache, high-signal for refactoring blast radius.
Excluded: POST actions, scan/log streams, file downloads, ML-cache-dependent paths
that may be empty depending on optional pipelines.
"""

ENDPOINTS = [
    "/api/table",
    "/api/pre-momentum",
    "/api/overview",
    "/api/universe",
    "/api/market-regime",
    "/api/category",
    "/api/theme",
    "/api/quant-strategies",
    "/api/factor-efficacy",
    "/api/classification",
    "/api/sector-rotation",
    "/api/effectiveness",
    "/api/meta",
]
