# Data Pipeline — 데이터 흐름 + 일일 갱신

이 문서는 외부 데이터 소스, batch refresh 파이프라인, cache 구조를 정리합니다.

## 1. 데이터 소스 요약

| 소스 | 종류 | 한도 (free tier) | 커버리지 | Cache 파일 |
|---|---|---|---|---|
| **yfinance** | 가격 (OHLCV) + 기본 fundamentals | sustained ~2k req/hr | 770 ticker (글로벌) | `.scan_cache.pkl`, `.fundamentals_cache.pkl` |
| **Finnhub** | 70+ ratios, recommendation 월별, EPS surprise, news | 60 req/min, US-listed only | ~620 of 770 | `.fundamentals_cache.pkl` (in-place enrich) |
| **GraphRAG** (internal) | 지식 그래프 (theme, community) | unlimited | 전체 universe | `.scan_cache.pkl["graph"]` |

### 한국 종목 (.KS) 한계

Finnhub 무료 tier는 한국 .KS ticker 차단 → yfinance만 사용. 결과:
- 분석가 수 적음 (현대차 ~3 vs Finnhub 미국 ~30+)
- bullish_change_3m 신호 없음
- EPS surprise 없음
- → QVR R component 약함

대안: KIS API (한국투자증권) 통합 가능하나 별도 작업 필요 (`docs/` 미작성).

## 2. 일일 갱신 파이프라인

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1 — 메인 스캔 (price_discovery.py, ~8분)           │
│  ────────────────────────────────────────────────       │
│  yfinance batch 다운로드 → all_raw indicator 계산        │
│  → cross-sectional ranks (RSS, URS) → 4축 score         │
│  → classification → PDF 출력 → .scan_cache.pkl 저장     │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2 — yfinance fundamentals                          │
│           (pipelines/fundamentals_pipeline.py, ~5분)     │
│  ────────────────────────────────────────────────       │
│  770 ticker × 4 endpoint (info, estimates, revisions,   │
│  recommendations) 병렬 fetch                             │
│  → .fundamentals_cache.pkl 저장                         │
│  → rate-limited 시 --retry-failed 옵션 제공             │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 3 — Finnhub enrichment                             │
│           (pipelines/finnhub_fundamentals.py, ~28분)     │
│  ────────────────────────────────────────────────       │
│  US ticker (~620) × 4 endpoint (profile, metric,        │
│  recommendation, earnings, news) fetch                   │
│  → .fundamentals_cache.pkl in-place 보강                │
│  → 한국 .KS 자동 skip                                    │
└─────────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 4 — API restart (uvicorn)                          │
│  ────────────────────────────────────────────────       │
│  api.py:_load_cache() 자동 호출                          │
│  → cache 로드 + Sector/SubTheme 매핑                     │
│  → QVR score 산출 + Eligibility Gate 적용                │
│  → 메모리 STATE에 저장                                    │
└─────────────────────────────────────────────────────────┘
                       ↓
            브라우저 → /api/* JSON 요청
```

### 명령어 시퀀스

```bash
# 1. 메인 스캔 (technical signals)
python3 price_discovery.py
# 결과: reports/Omega(PD_v5_STK)_YYYYMMDD.pdf, .scan_cache.pkl

# 2. yfinance fundamentals
python3 pipelines/fundamentals_pipeline.py
# rate-limited 시:
python3 pipelines/fundamentals_pipeline.py --retry-failed --retry-cooldown 300

# 3. Finnhub 보강 (US 종목만)
python3 pipelines/finnhub_fundamentals.py

# 4. API 재시작
lsof -ti:8000 | xargs kill 2>/dev/null
nohup python3 -m uvicorn api:app --port 8000 > .api.log 2>&1 &
```

> `pipelines/*` 스크립트는 파일 상단에서 `sys.path.insert(0, _PROJECT_ROOT)` 를 호출하므로 bare path 실행이 가능. `cwd`는 프로젝트 루트여야 함 (cache 파일이 루트에 저장됨).

### Cron 설정 예시 (매일 새벽 5시)

```cron
# US 마감 후 (KST 기준 새벽)
0 5 * * 1-5  cd /path/to/PriceDiscovery && python3 price_discovery.py >> .scan.log 2>&1
30 5 * * 1-5 cd /path/to/PriceDiscovery && python3 pipelines/fundamentals_pipeline.py --max-age-h 12 >> .fund.log 2>&1
40 5 * * 1-5 cd /path/to/PriceDiscovery && python3 pipelines/finnhub_fundamentals.py >> .finnhub.log 2>&1
20 6 * * 1-5 cd /path/to/PriceDiscovery && systemctl restart price-discovery-api
```

## 3. Cache 구조

### `.scan_cache.pkl` (price_discovery.py 생성)

```python
{
    "results": [                       # 770 ticker × ~80 fields
        {
            "ticker": "NVDA",
            "name": "NVIDIA",
            "category": "STK_Technology",
            "composite": 73.2,
            "tcs": 100, "tfs": 52, "rss": 97.1, "urs": 42.5,
            "tcs_short": 100, "tcs_long": 100,
            "tfs_short": 45, "tfs_long": 58,
            "rss_short": 98.5, "rss_long": 96.4,
            "oer": 98,
            "classification": "🟡 OVEREXTENDED",
            "eligible": True,
            "rejection": "None",
            "rsi": 86.4, "trend_age": 22, "sma50_dist": 58.08,
            "vol_ratio_3d_10d": 1.7,    # (계산되나 results에 미저장 — bug 가능)
            "ret_1d": 7.91, "ret_5d": 24.0, "ret_21d": 71.8, ...,
            "ret_12_1m": 370.8, "ret_36_12m": 72.77,
            # 8개 hedge strategy
            "oneil_long": 86, "oneil_short": 0,
            "minervini_long": 80, ...
            # 통합 hedge signal
            "combined_long": 75, "combined_short": 12,
            "long_count": 6, "short_count": 1,
            "net_signal": "STRONG_LONG",
            "conviction": 63.0,           # hedge conviction (numeric, NOT PM conviction)
            # validation engine
            "score_1w": 69.5, "score_1m": 69.1, "score_3m": 73.5,
            "ret_1w": 23.99, "ret_1m": 71.80, "ret_3m": -12.28,
            "val_prob": 50.7, "val_persist": 99.6, "val_conf": "H",
            # alpha potential
            "alpha_potential": 65, "structural_q": 73,
            "reversal_pctile": 82.3,
            # event signals
            "event_flag": False, "event_reasons": "None",
            "gap_drift_30d": 13.0, "gap_event_age": 3,
            # market cap & ADV
            "market_cap": 4850e9, "adv_usd": 25e9,
        },
        ...
    ],
    "graph": {                         # GraphRAG 결과 (graph_engine.py)
        "communities": {...},
        "community_stats": {...},
        "insights": [...],
    },
    "history": {                       # 7-day historical snapshots
        "NVDA": [
            {"date": "2026-04-30", "composite": 70.5, ...},
            ...
        ],
    },
    "ve_observations": [...],           # SignalValidityEngine
    "ve_bucket": {...},                 # bucket-level hit rates
    "ve_class": {...},                  # class-level hit rates
    "ve_transitions": {...},            # class transition matrix
    "factor_efficacy": {...},           # Reverse Factor Model
    "scan_time": "2026-05-04T05:30:00",
}
```

### `.fundamentals_cache.pkl` (`pipelines/fundamentals_pipeline.py` + `pipelines/finnhub_fundamentals.py` 생성)

```python
{
    "fetched_at": "2026-05-04T05:35:00",
    "tickers": {
        "NVDA": {
            "asset_type": "Stock",
            # yfinance 부분
            "info": {                  # yfinance .info attr
                "trailing_pe": 40.6, "forward_pe": 17.8,
                "price_to_book": 30.8, "peg": 0.76,
                "gross_margin": 0.711, "operating_margin": 0.65, "profit_margin": 0.556,
                "roe": 1.015, "roa": 0.512,
                "debt_to_equity": 7.26, "current_ratio": 3.91,
                "market_cap": 4850e9, "beta": 2.34,
                "earnings_growth": 0.956, "revenue_growth": 0.732,
                ...
            },
            "estimates": {              # yfinance earnings_estimate
                "0q": {"eps_avg": 1.77, "eps_low": 1.69, "eps_high": 1.99,
                       "year_ago_eps": 0.81, "n_analysts": 38, "growth": 1.19,
                       "rev_avg": 78.8e9, "rev_growth": 0.79},
                "+1q": {...}, "0y": {...}, "+1y": {...},
            },
            "revisions": {              # yfinance eps_revisions
                "up_7d": 1, "up_30d": 2,
                "down_7d": 0, "down_30d": 1,
                "net_30d": 1, "ratio_30d": 0.667,
                "by_period": {"0q": {...}, "+1q": {...}, "0y": {...}},
            },
            "recommendations": {        # yfinance recommendations[0]
                "strong_buy": 23, "buy": 14, "hold": 5, "sell": 0, "strong_sell": 0,
                "total": 42, "bullish_ratio": 0.881, "bearish_ratio": 0.0,
            },
            "price_targets": {          # yfinance target_*
                "mean": 269.2, "low": 200, "high": 350,
                "n_analysts": 57, "upside_pct": 34.9,
            },

            # Finnhub 부분 (US 종목만)
            "finnhub_metrics": {        # 70+ ratios
                "peNormalizedAnnual": 40.16, "pbAnnual": 28.81,
                "grossMarginTTM": 71.31, "operatingMarginTTM": 65.02,
                "roeTTM": 104.37,
                ...
            },
            "rec_history": [            # Finnhub recommendation 4개월
                {"period": "2026-04-01", "strongBuy": 14, "buy": 23,
                 "hold": 15, "sell": 2, "strongSell": 0, "symbol": "NVDA"},
                {"period": "2026-03-01", ...},
                ...
            ],
            "eps_surprises": [          # Finnhub earnings 4분기
                {"actual": 1.95, "estimate": 1.85,
                 "surprise": 0.10, "surprisePercent": 5.4,
                 "period": "2026-03-31", "quarter": 1, "year": 2026},
                ...
            ],
            "news_items_meta": [        # Finnhub news (datetime만)
                {"datetime": 1714809600},
                ...
            ],
            "finnhub_derived": {        # 계산된 leading signals
                "rec_total_now": 71,
                "bullish_ratio_now": 0.93,
                "bullish_change_3m": 0.041,
                "eps_beat_rate": 1.0,
                "eps_surprise_avg": 2.14,
                "eps_n_quarters": 4,
                "news_count_7d": 246,
                "news_count_3d": 246,
                "news_recency": 1.0,
            },
            "finnhub_ok": True,
            "finnhub_error": None,
            "fetch_ok": True,
            "error": None,
            "elapsed_sec": 1.42,
        },
        "005930.KS": {                  # 한국 종목 — Finnhub 없음
            "asset_type": "Stock",
            "info": {...},
            "estimates": {...},
            "revisions": {...},
            "recommendations": {...},
            "price_targets": {...},
            "finnhub_ok": False,        # Finnhub skipped
            ...
        },
    },
    "stats": {
        "total_attempted": 770,
        "stock_ok": 530, "etf_ok": 231,
        "has_estimates": 515, "has_revisions": 515,
        "failed_count": 9,
        "failed_tickers": ["NIGS", "ANSS", ...],
        "duration_sec": 48,
        "finnhub_enriched": 709,
        "finnhub_failed": 3,
        "finnhub_skipped_korean": 58,
        "finnhub_duration_sec": 1672,
    },
    "finnhub_enriched_at": "2026-05-03T12:08:00",
}
```

## 4. yfinance fetcher 상세 (`pipelines/fundamentals_pipeline.py`)

### 동작

- 770 ticker × 4-5 endpoint 병렬 fetch (default workers=8)
- 각 ticker per-fetch 함수 `fetch_one(ticker, asset_type)`
- ETF는 estimates / revisions / recommendations skip
- 진행률 25 ticker마다 출력

### Rate limit 대응

yfinance 자체는 rate limit 명시 없으나, sustained high-volume 시 IP-based throttle 발생. 패턴:
1. 첫 batch에서 ~100-200 ticker 성공
2. 갑자기 "Too Many Requests" 에러 시작
3. Retry without cooldown은 무의미

해결: `--retry-failed --retry-cooldown 300` 옵션:
- 기존 cache 로드
- rate-limited 실패 ticker만 추출
- 5분 sleep 후 workers=2 + per-request 0.3s delay로 재시도
- 최대 3 attempt

실측 (770 ticker):
- 첫 batch (workers=10): 50초, ~80% 성공
- Retry 1 (workers=2, cooldown 300s): 추가 ~150 복구
- Retry 2: 마지막 ~10개 복구
- 최종 성공률: ~99%

### 실패 카테고리

`no_info` 에러는 진짜 잘못된 ticker (delisted/티커변경):
- ANSS (Synopsys 합병)
- TEF (NYSE 상폐)
- WNS, NIGS 등 ~9개

이들은 universe에서 제거하거나 무시.

## 5. Finnhub fetcher 상세 (`pipelines/finnhub_fundamentals.py`)

### 전제

- API key는 `.finnhub_config.json`에 저장 (gitignored):
  ```json
  {"api_key": "your_key_here", "tier": "free", "rate_limit_per_min": 60}
  ```
- 무료 가입: https://finnhub.io/register

### Endpoints 사용

| Endpoint | 용도 | 응답 |
|---|---|---|
| `/stock/profile2` | 회사 메타데이터 | name, country, industry, marketCap |
| `/stock/metric?metric=all` | 70+ ratios | peNormalizedAnnual, grossMarginTTM, roeTTM, ... |
| `/stock/recommendation` | 월별 4개월 분석가 추천 | strongBuy/buy/hold/sell counts per month |
| `/stock/earnings` | 분기별 4분기 EPS surprise | actual, estimate, surprisePercent per quarter |
| `/company-news?symbol=&from=&to=` | 7일 news 목록 | datetime, headline, source 등 |

### 동작

- 기존 `.fundamentals_cache.pkl` 로드
- US-supported ticker (`is_supported_symbol()`) 만 fetch
- `.KS`, `.T`, `.HK`, `.SS` 등은 skip
- 각 ticker의 enrichment를 기존 cache entry에 in-place merge

### Rate limit 자동 처리

`pipelines/finnhub_client.py:FinnhubClient` 가 X-Ratelimit-Remaining 헤더 모니터링 후 자동 sleep. 60 calls/min 한도 안에서 동작.

770 × 4 endpoint = 약 3000 req → 실측 ~28분 (workers=4).

## 6. GraphRAG (`agents/graph_engine.py`)

### 입력

`run_scan()` 종료 시점의 results dict + STOCK_THEMES + GLOBAL_ETF_UNIVERSE.

### 처리

1. NetworkX 그래프 구축 (node = ticker, edge = peer/theme/category 관계)
2. Louvain community detection → community 별 평균 momentum, breadth 산출
3. Insights 생성:
   - Theme propagation (어느 theme이 spreading)
   - ETF-stock divergence (ETF 강세 vs 구성종목 약세 etc.)
   - Leader-lagger pairs
   - Category entropy
   - Cross-category flow
4. Multi-hop queries:
   - `query_impact_radius(ticker)` — ticker 변화 시 영향받는 peer
   - `query_theme_status(theme)` — theme 전반 상태
   - `query_formation_pipeline()` — pre-momentum candidates 추출

### 출력

`.scan_cache.pkl["graph"]` 에 저장:
```python
{
    "communities": {community_id: [tickers]},
    "community_stats": {id: {avg_composite, breadth, top_tickers}},
    "insights": [{"type": ..., "text": ..., "tickers": [...]}],
    "edges": [...],
}
```

이 graph 데이터는 `agents/pre_momentum.py:GraphRelationalAgent`에서 사용됨.

## 7. Cache 무결성 점검

### 데이터가 stale한지 확인

```bash
# 가장 최근 update 시각
ls -la .scan_cache.pkl .fundamentals_cache.pkl
```

또는 API 통해:
```bash
curl -s http://localhost:8000/api/meta | jq .scan_time
```

### Cache 정합성 검증 (수동)

```python
import pickle
with open('.scan_cache.pkl','rb') as f:
    sc = pickle.load(f)
with open('.fundamentals_cache.pkl','rb') as f:
    fc = pickle.load(f)

# 동일 universe 인지
sc_tickers = {r['ticker'] for r in sc['results']}
fc_tickers = set(fc['tickers'].keys())
print(f"only in scan: {sc_tickers - fc_tickers}")
print(f"only in fund: {fc_tickers - sc_tickers}")
```

### Recovery 시나리오

| 문제 | 조치 |
|---|---|
| API 시작 안 됨 — `_load_cache` failure | `.scan_cache.pkl` 손상 확인. 재생성: `python3 price_discovery.py` |
| QVR 모두 50 (neutral) | `.fundamentals_cache.pkl` 누락. 재생성: `python3 pipelines/fundamentals_pipeline.py` |
| Finnhub 신호 (bullish_change_3m) None | Finnhub 없음. `python3 pipelines/finnhub_fundamentals.py` 실행 |
| Eligibility Gate 강등 안 됨 | API 재시작 (`POST /api/reload` 또는 uvicorn 재기동) — cache 변경 후 reload 필요 |
| `ModuleNotFoundError: config` (sector_rotation_backtest) | bare path가 아닌 module 형식으로: `python3 -m strategies.sector_rotation_backtest` |
