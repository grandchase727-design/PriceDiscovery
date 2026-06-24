# -*- coding: utf-8 -*-
"""cache_paths.py — Centralized cache file path resolver (Option B refactor).

Goal: consolidate 14 scattered root-level cache files into .cache/ directory
while preserving backward compatibility — if a root-level legacy file exists,
read from it; new writes go to .cache/.

USAGE:
    from agents.cache_paths import cache_path
    p = cache_path("scan_cache.pkl")    # Path object
    p = cache_path(".scan_cache.pkl")   # also accepted (legacy prefix stripped)
"""
from __future__ import annotations

from pathlib import Path

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)


def cache_path(name: str) -> Path:
    """Return Path for a cache file.

    Resolution priority:
      1. If .cache/<name> exists → use it
      2. If legacy ./<.name> exists → use it (read-compatible)
      3. Otherwise → return .cache/<name> for new write

    Args:
        name: cache filename. Both "scan_cache.pkl" and ".scan_cache.pkl" work.
    """
    name = name.lstrip(".")
    new_p = CACHE_DIR / name
    legacy_p = Path(f".{name}")
    if new_p.exists():
        return new_p
    if legacy_p.exists():
        return legacy_p
    return new_p   # default for new writes


def migrate_legacy_caches() -> dict:
    """Move all legacy root-level cache files into .cache/.

    Returns:
        {moved: int, skipped: int, errors: list[str]}
    """
    import shutil
    LEGACY_CACHES = [
        ".scan_cache.pkl", ".fundamentals_cache.pkl",
        ".backtest_price_cache.pkl", ".breadth_universe_cache.pkl",
        ".etf_holdings_cache.json", ".market_leaders_swarm_cache.json",
        ".final_list_commentary_cache.json", ".elliott_stops_cache.json",
        ".trade_log.json", ".multi_agent_debate_cache.json",
        ".debate_refresh_queue.json", ".ytd_returns.json",
        ".unified_classification.json",
    ]
    moved = skipped = 0
    errors = []
    for legacy_name in LEGACY_CACHES:
        src = Path(legacy_name)
        if not src.exists():
            skipped += 1
            continue
        name = legacy_name.lstrip(".")
        dst = CACHE_DIR / name
        if dst.exists():
            skipped += 1
            continue
        try:
            shutil.move(str(src), str(dst))
            moved += 1
        except Exception as e:
            errors.append(f"{legacy_name}: {e}")
    return {"moved": moved, "skipped": skipped, "errors": errors}
