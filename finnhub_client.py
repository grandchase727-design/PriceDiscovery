"""
finnhub_client.py — Thin REST wrapper for Finnhub free-tier endpoints.

Free tier (60 calls/min, US-listed only):
  - /stock/profile2          — basic profile
  - /stock/metric            — 70+ ratios (Q + V components)
  - /stock/recommendation    — monthly history (4 months) of buy/hold/sell counts
  - /stock/earnings          — quarterly EPS actual/estimate/surprise (4 quarters)
  - /company-news            — news + sentiment

Korean tickers (.KS / .KO) are blocked on free tier — handle via separate fetcher.
"""

from __future__ import annotations

import json, os, time, urllib.request, urllib.error
from typing import Optional

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".finnhub_config.json")
BASE_URL = "https://finnhub.io/api/v1"


def _load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Create it with {{\"api_key\": \"<key>\"}}"
        )
    with open(CONFIG_PATH) as f:
        return json.load(f)


class FinnhubClient:
    """Rate-limited Finnhub client. Sleeps when X-Ratelimit-Remaining hits 0."""

    def __init__(self, api_key: Optional[str] = None,
                 rate_limit_per_min: int = 60,
                 user_agent: str = "price-discovery/1.0"):
        if api_key is None:
            cfg = _load_config()
            api_key = cfg["api_key"]
            rate_limit_per_min = cfg.get("rate_limit_per_min", 60)
        self.api_key = api_key
        self.rate_limit = rate_limit_per_min
        self.user_agent = user_agent
        # Tracking
        self._reset_at: float = 0.0  # epoch when rate limit window resets
        self._remaining: int = rate_limit_per_min

    # ── Internal request ──

    def _request(self, path: str, params: Optional[dict] = None) -> tuple[int, dict, str]:
        url = f"{BASE_URL}{path}"
        if params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode(params)}"
        req = urllib.request.Request(url, headers={
            "X-Finnhub-Token": self.api_key,
            "User-Agent": self.user_agent,
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                self._update_rate_state(dict(resp.headers))
                return resp.status, dict(resp.headers), body
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            self._update_rate_state(dict(e.headers or {}))
            return e.code, dict(e.headers or {}), body

    def _update_rate_state(self, headers: dict) -> None:
        try:
            self._remaining = int(headers.get("X-Ratelimit-Remaining", self._remaining))
            self._reset_at = float(headers.get("X-Ratelimit-Reset", self._reset_at))
        except (TypeError, ValueError):
            pass

    def _wait_if_needed(self) -> None:
        """If remaining is low, sleep until reset window passes."""
        if self._remaining <= 1 and self._reset_at > 0:
            now = time.time()
            wait = max(0.0, self._reset_at - now + 0.5)
            if wait > 0:
                time.sleep(wait)

    # ── Endpoint wrappers ──

    def get_json(self, path: str, params: Optional[dict] = None,
                 retries: int = 2, backoff_sec: float = 5.0) -> Optional[dict | list]:
        """Make a request, return parsed JSON or None on failure."""
        for attempt in range(retries + 1):
            self._wait_if_needed()
            status, headers, body = self._request(path, params)
            if status == 200:
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return None
            if status == 429:
                # Rate limited — back off and retry
                time.sleep(backoff_sec * (attempt + 1))
                continue
            if status in (403, 404):
                # Unsupported ticker / endpoint — give up immediately
                return None
            # Other errors — retry once with backoff
            if attempt < retries:
                time.sleep(backoff_sec)
                continue
        return None

    def profile(self, symbol: str) -> Optional[dict]:
        """Basic company profile (name, country, industry, market cap)."""
        return self.get_json("/stock/profile2", {"symbol": symbol})

    def metrics(self, symbol: str, metric: str = "all") -> Optional[dict]:
        """70+ financial ratios. Returns full dict; key is 'metric'."""
        return self.get_json("/stock/metric", {"symbol": symbol, "metric": metric})

    def recommendation_trends(self, symbol: str) -> Optional[list]:
        """Monthly recommendation history (~4 months)."""
        return self.get_json("/stock/recommendation", {"symbol": symbol})

    def earnings_surprises(self, symbol: str) -> Optional[list]:
        """Quarterly EPS actual/estimate/surprise (~4 quarters)."""
        return self.get_json("/stock/earnings", {"symbol": symbol})

    def company_news(self, symbol: str, from_date: str, to_date: str) -> Optional[list]:
        """Company news (with sentiment when available)."""
        return self.get_json("/company-news",
                             {"symbol": symbol, "from": from_date, "to": to_date})


# ──────────────────────────────────────────────────────────────────────
# Standalone smoke test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    c = FinnhubClient()
    print(f"Initialized FinnhubClient (rate_limit={c.rate_limit}/min)")

    print("\n=== AAPL profile ===")
    p = c.profile("AAPL")
    print(f"  {p.get('name')} | {p.get('finnhubIndustry')} | mktCap={p.get('marketCapitalization')}")

    print("\n=== AAPL metrics (subset) ===")
    m = c.metrics("AAPL")
    if m and "metric" in m:
        for k in ("peNormalizedAnnual", "pbAnnual", "grossMarginTTM",
                  "operatingMarginTTM", "roeTTM", "epsTTM"):
            print(f"  {k} = {m['metric'].get(k)}")

    print("\n=== AAPL recommendation trends (4 months) ===")
    rec = c.recommendation_trends("AAPL")
    if rec:
        for r in rec:
            total = r.get("strongBuy", 0) + r.get("buy", 0) + r.get("hold", 0) + r.get("sell", 0) + r.get("strongSell", 0)
            bull_ratio = (r.get("strongBuy", 0) + r.get("buy", 0)) / total if total > 0 else 0
            print(f"  {r.get('period')}: SB={r.get('strongBuy')}/B={r.get('buy')}/H={r.get('hold')}/"
                  f"S={r.get('sell')}/SS={r.get('strongSell')} → bullish_ratio={bull_ratio:.2f}")

    print("\n=== AAPL earnings surprises (4 quarters) ===")
    es = c.earnings_surprises("AAPL")
    if es:
        for e in es:
            print(f"  {e.get('period')}: actual={e.get('actual')} est={e.get('estimate')} "
                  f"surprise%={e.get('surprisePercent')}")
