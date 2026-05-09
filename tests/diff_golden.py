"""Compare current API responses against tests/golden/ baseline.

Exit code 0 = identical, 1 = drift, 2 = error.
Time-volatile fields (timestamps, run-id) are normalized before diff.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

from golden_endpoints import ENDPOINTS

BASE = os.environ.get("PD_API_BASE", "http://127.0.0.1:8000")
GOLDEN_DIR = Path(__file__).parent / "golden"

# Strip these top-level keys before comparing — they reflect process state, not data.
VOLATILE_KEYS = {"generated_at", "timestamp", "as_of_ts", "now", "server_time"}


def slug(path: str) -> str:
    return path.strip("/").replace("/", "_") or "root"


def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, float):
        # tiny float noise can drift across pandas/numpy versions; round to 6 decimals
        return round(obj, 6)
    return obj


def fetch(path: str):
    req = urllib.request.Request(BASE + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def diff_paths(a, b, prefix="") -> list[str]:
    out = []
    if type(a) is not type(b):
        out.append(f"{prefix}: type {type(a).__name__} → {type(b).__name__}")
        return out
    if isinstance(a, dict):
        ak, bk = set(a), set(b)
        for k in sorted(ak - bk):
            out.append(f"{prefix}.{k}: missing in current")
        for k in sorted(bk - ak):
            out.append(f"{prefix}.{k}: new in current")
        for k in sorted(ak & bk):
            out.extend(diff_paths(a[k], b[k], f"{prefix}.{k}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{prefix}: len {len(a)} → {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_paths(x, y, f"{prefix}[{i}]"))
    elif a != b:
        out.append(f"{prefix}: {a!r} → {b!r}")
    return out


def main() -> int:
    drifted = 0
    missing_baseline = 0
    for path in ENDPOINTS:
        baseline_file = GOLDEN_DIR / f"{slug(path)}.json"
        if not baseline_file.exists():
            print(f"  ? {path:<40} no baseline (skip)")
            missing_baseline += 1
            continue
        baseline = json.loads(baseline_file.read_text())
        try:
            current = fetch(path)
        except Exception as e:
            print(f"  ✗ {path:<40} fetch error: {e}")
            drifted += 1
            continue
        nb, nc = normalize(baseline), normalize(current)
        diffs = diff_paths(nb, nc)
        if not diffs:
            print(f"  ✓ {path:<40} identical")
        else:
            drifted += 1
            print(f"  ✗ {path:<40} {len(diffs)} differences")
            for d in diffs[:8]:
                print(f"      {d}")
            if len(diffs) > 8:
                print(f"      ... ({len(diffs) - 8} more)")
    print(f"\n{len(ENDPOINTS) - drifted - missing_baseline}/{len(ENDPOINTS)} match, "
          f"{drifted} drifted, {missing_baseline} no-baseline")
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
