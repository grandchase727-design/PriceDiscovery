"""Capture baseline API responses to tests/golden/<name>.json.

Run while the API is up on http://127.0.0.1:8000 (or override BASE).
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from golden_endpoints import ENDPOINTS

BASE = os.environ.get("PD_API_BASE", "http://127.0.0.1:8000")
OUT_DIR = Path(__file__).parent / "golden"


def fetch(path: str) -> tuple[int, bytes, float]:
    t0 = time.time()
    req = urllib.request.Request(BASE + path, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read()
            return resp.status, body, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t0


def slug(path: str) -> str:
    return path.strip("/").replace("/", "_") or "root"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    failed = 0
    for path in ENDPOINTS:
        status, body, elapsed = fetch(path)
        out = OUT_DIR / f"{slug(path)}.json"
        if status == 200:
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                print(f"  ✗ {path} HTTP 200 but not JSON ({len(body)}B)")
                failed += 1
                continue
            out.write_text(json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False))
            summary.append((path, status, len(body), elapsed))
            print(f"  ✓ {path:<40} {len(body):>10}B  {elapsed:.2f}s → {out.name}")
        else:
            print(f"  ✗ {path} HTTP {status}")
            failed += 1
    print(f"\n{len(summary)} captured, {failed} failed → {OUT_DIR}")
    (OUT_DIR / "_index.json").write_text(
        json.dumps(
            {p: {"status": s, "bytes": b, "seconds": round(t, 3)} for p, s, b, t in summary},
            indent=2,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
