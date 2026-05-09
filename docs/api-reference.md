# API Reference

FastAPI 백엔드 (`api.py`) 엔드포인트 목록 + 응답 schema + Frontend 탭 매핑.

기본 URL: `http://localhost:8000`

## Endpoint 목록 (역할별 그룹)

### 1. 시스템 메타 + 운영

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/meta` | 카테고리/섹터/SubTheme/classification 목록 + scan_time |
| POST | `/api/reload` | `.scan_cache.pkl` 강제 재로드 (cache 갱신 후) |
| POST | `/api/scan` | live scan trigger (price_discovery.py 백그라운드 실행) |
| GET | `/api/scan/status` | scan 진행 상태 + 마지막 phase |
| GET | `/api/scan/log` | scan 로그 tail |

### 2. 메인 데이터 조회

| Method | Path | 용도 | 주요 Query Params |
|---|---|---|---|
| GET | `/api/overview` | KPI cards, classification dist, top picks, conviction bubble | `categories`, `sectors`, `subthemes`, `classifications`, `eligible_only`, `comp_min/max` |
| GET | `/api/table` | 770 ticker × 70+ fields (메인 ticker 테이블) | 동일 filter |
| GET | `/api/universe` | universe-only basics (returns + ann metrics) | 동일 filter |
| GET | `/api/category` | 카테고리별 aggregate (avg score, breadth) | 동일 filter |
| GET | `/api/theme` | SubTheme별 aggregate + classification by theme | `min_n` (최소 ticker 수) + filter |

### 3. 분석 & 지표

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/market-regime` | 시장 환경 진단 (regime probability) |
| GET | `/api/weekly-heatmap` | 7일 ticker × class transition matrix |
| GET | `/api/effectiveness` | SignalValidityEngine 결과 (bucket/class hit rates) |
| GET | `/api/period-analysis?fwd_days=N` | forward N-day 분석 |
| GET | `/api/period-curve` | period-by-period composite trajectory |
| GET | `/api/graph` | GraphRAG community + insights |
| GET | `/api/factor-efficacy` | Reverse Factor Model 5-methodology 결과 |
| GET | `/api/quant-strategies` | top picks per strategy (8개) |
| GET | `/api/validation` | backtest 검증 |
| GET | `/api/classification` | classification 분포 + 통계 |
| GET | `/api/classification/validation` | classification rule validation |
| GET | `/api/classification-history` | 1년 ~2주 간격 classification 분포 변화 |
| GET | `/api/classification-history-by-sector` | sector별 동일 |

### 4. Pre-Momentum

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/pre-momentum` | Pre-Momentum 5-agent 분석 결과 (candidates + summary + methodology + conversion tracking) |

### 5. AI Prediction (ML 신호 엔진 — 사용자 대상 결과)

`ml/` 패키지의 ML 출력을 frontend AI Prediction 탭에 노출하는 endpoint:

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/ai-prediction` | ML 모델 prediction (probabilities, MoE 등) |
| GET | `/api/ai-winratio` | AI 신호 hit rate |
| GET | `/api/ai-benchmarks` | 벤치마크 비교 |
| GET | `/api/ai-performance` | rolling/monthly 성과 |

### 6. ML Internal (Legacy ML REST layer)

`ml/` 산출물 중 운영/디버깅용 raw view (`/api/ml/*`). 사용자 frontend 탭에는 노출 안 됨:

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/ml/meta` | ML 모델 버전 + feature 메타 |
| GET | `/api/ml/table` | 770 ticker × ML feature/score raw 테이블 |
| GET | `/api/ml/pre-momentum` | ML 기반 pre-momentum 후보 (실험용 — production은 `/api/pre-momentum`) |
| GET | `/api/ml/classification-history` | ML-classified 분포 추이 |

### 7. Sector Rotation (daimon sync)

[`daimon/strategy/us_sector_rotation/`](../CLAUDE.md#us-sector-rotation-strategy) 파이프라인의 parquet/JSON 산출물을 읽어 노출:

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/sector-rotation` | 11 GICS 섹터 ETF 월말 score table + 최신 top-N picks (`latest_signal.json`) |
| GET | `/api/sector-rotation/backtest` | 월별 cumulative return 시계열 + Sharpe/CAGR/MDD vs S&P500 (`monthly_backtest.parquet`) |

### 8. 보고서 & 참조 자료

| Method | Path | 용도 |
|---|---|---|
| GET | `/api/report` | 자동 생성 분석 리포트 |
| GET | `/api/references` | 참고 문헌 목록 |
| GET | `/api/references/file/{filename}` | 개별 파일 다운로드 |

## 공통 Filter Query Params

대부분의 데이터 endpoint는 다음 query params 지원:

```
categories     : List[str]   — 레거시 category 필터 (e.g. STK_Technology)
sectors        : List[str]   — 17개 sector 필터 (e.g. Technology, Fixed Income)
subthemes      : List[str]   — 105개 SubTheme 필터 (e.g. Semiconductor Design)
classifications: List[str]   — classification 필터 (e.g. CONTINUATION)
eligible_only  : bool        — Eligibility Gate 통과한 ticker만
comp_min       : float (0)   — composite 하한
comp_max       : float (100) — composite 상한
```

`_filter_df()` 헬퍼 (api.py)가 이 모든 조합 처리.

## Schema 예시

### `/api/meta` 응답

```json
{
    "scan_time": "2026-05-04T05:30:00",
    "total_tickers": 770,
    "categories": ["EQ_Broad", "EQ_Technology", ...],
    "sectors": ["Technology", "Fixed Income", "International", ...],
    "subthemes": ["Semiconductor Design", "Big Pharma", ...],
    "classifications": ["🟢 CONTINUATION", "🔵 RECOVERY", ...],
    "category_info": [
        {"category": "STK_Technology", "n": 87,
         "asset_type": "Stock", "benchmark": "XLK",
         "alt_benchmarks": ["SMH", "QQQ"]},
        ...
    ],
    "sector_info": [
        {"sector": "Technology", "n": 144,
         "n_etf": 18, "n_stock": 126,
         "subthemes": ["Semiconductor Design", "Cloud & Platform", ...]},
        ...
    ],
    "theme_info": [...]
}
```

### `/api/table` 응답

```json
{
    "data": [
        {
            "ticker": "NVDA",
            "name": "NVIDIA",
            "category": "STK_Technology",
            "sector": "Technology",
            "asset_type": "Stock",
            "theme": "Semiconductor Design",
            "theme_detail": "AI Chip",
            "composite": 73.2,
            "tcs": 100, "tfs": 52, "rss": 97.1,
            "tcs_short": 100, "tcs_long": 100,
            "tfs_short": 45, "tfs_long": 58,
            "rss_short": 98.5, "rss_long": 96.4,
            "oer": 98,
            "qvr_score": 73.4, "qvr_q": 90.9, "qvr_v": 48.3, "qvr_r": 73.0,
            "qvr_n_analysts": 71,
            "qvr_bullish_chg_3m": 4.07,
            "qvr_eps_beat_rate": 100,
            "qvr_eps_surprise_avg": 2.14,
            "classification": "🟡 OVEREXTENDED",
            "eligible": true,
            "rejection": "None",
            "rsi": 86.4, "trend_age": 22, "sma50_dist": 58.08,
            "adv_M": 25000.0, "mktcap_B": 4850.55,
            "oneil_long": 86, "oneil_short": 0,
            "minervini_long": 80, ...
            "combined_long": 75, "combined_short": 12,
            "long_count": 6, "short_count": 1,
            "net_signal": "STRONG_LONG", "conviction": 63.0,
            "score_1w": 69.5, "score_1m": 69.1, "score_3m": 73.5,
            "ret_1w": 23.99, "ret_1m": 71.80, "ret_3m": -12.28,
            "ret_1d": 7.91, "ret_5d": 24.0, "ret_21d": 71.8,
            "ret_63d": 78.62, "ret_126d": 174.87, "ret_252d": 708.85,
            "ret_3y_ann": 140.87, "ret_5y_ann": 57.55, "vol_3y_ann": 46.85,
            "mom_age": 22,
        },
        ...
    ]
}
```

### `/api/pre-momentum` 응답

```json
{
    "candidates": [
        {
            "ticker": "LVS",
            "name": "Las Vegas Sands",
            "category": "STK_ConsDisc",
            "sector": "Consumer Discretionary",
            "theme": "Restaurants & Leisure",
            "current_classification": "🔵 RECOVERY",
            "current_composite": 55.7,
            "pre_momentum_score": 55.7,
            "agreement_ratio": 0.6,
            "agents": {
                "microstructure": {
                    "score": 55.9,
                    "signals": {
                        "volatility_compression": 60.0,
                        "accumulation_pattern": 45.0,
                        ...
                    },
                    "summary": "Volume picking up; range contraction"
                },
                "macro_regime": {"score": 60.4, ...},
                "graph_relational": {"score": 32.1, ...},
                "catalyst": {"score": 52.4, ...},
                "qvr": {
                    "score": 74.4,
                    "signals": {
                        "quality": 85.1,
                        "value": 52.2,
                        "revision": 76.8,
                        "net_30d": 7,
                        "ratio_30d": 82.0,
                        "n_analysts": 11,
                        "fwd_pe": 14.8,
                        "earn_growth": 73.0,
                        "upside_pct": 27.4,
                        "bullish_change_3m": 6.2,
                        "eps_beat_rate": 100,
                        "eps_surprise_avg": 5.4
                    },
                    "summary": "Strong fundamentals (Q+V+R aligned) · +7 net EPS revisions (30d)"
                }
            },
            "expected_timeline": "2-3 weeks",
            "key_catalysts": ["Strong fundamentals", "Volume picking up", ...],
            "risk_factors": ["Sector concentration"],
            "ret_1d": 1.2, "ret_5d": 4.5, "ret_21d": 8.7,
            ...
            "pm_age": 12
        },
        ...
    ],
    "candidates_etf": [...],     // ETF only
    "candidates_stock": [...],   // Stock only
    "summary": {
        "total_universe": 770,
        "candidates_analyzed": 308,
        "agreement_strong": 35,
        "agreement_moderate": 120,
        "agreement_weak": 118,
        "agreement_none": 35,
        "top_sectors": [
            {"sector": "Technology", "count": 65, "avg_score": 47.2},
            ...
        ],
        "agent_agreement_distribution": {0: 35, 1: 50, 2: 70, 3: 80, 4: 35, 5: 38}
    },
    "methodology": {
        "description": "Pre-Momentum Detection identifies tickers showing structural conditions...",
        "agents": [
            {"name": "Microstructure", "weight": 0.20, "type": "Quant",
             "description": "Volatility compression, accumulation patterns, ..."},
            ...
        ],
        "agreement_thresholds": {
            "strong":   "agreement_ratio ≥ 0.6 (≥3 of 5 agents above 50)",
            "moderate": "0.4 ≤ ratio < 0.6 (2 of 5 agents)",
            "weak":     "0 < ratio < 0.4 (1 of 5 agents)",
            "none":     "ratio == 0"
        }
    },
    "conversion": {  // optional
        "graduated": [...],
        "failed": [...],
        "in_progress": [...],
        "stats": {
            "total_pm_candidates_1m": 200,
            "total_graduated": 120,
            "total_failed": 60,
            "total_in_progress": 20,
            "hit_rate": 60.0,
            "avg_score_improvement": 8.5
        }
    }
}
```

## Frontend 탭 매핑

| 탭 | 사용 endpoints |
|---|---|
| **Price Discovery → Pre-Momentum** | `/api/pre-momentum` |
| **Price Discovery → Momentum** | `/api/table?eligible_only=true` |
| **Price Discovery → Excluded** | `/api/table` (filter on rejection ≠ None) |
| **Validation** | `/api/validation`, `/api/effectiveness`, `/api/period-curve` |
| **Market Environment** | `/api/market-regime`, `/api/overview`, `/api/weekly-heatmap`, `/api/classification-history`, `/api/classification-history-by-sector` |
| **Analysis** | `/api/factor-efficacy`, `/api/quant-strategies`, `/api/period-analysis` |
| **Sector Rotation** (Strategy 카드) | `/api/sector-rotation`, `/api/sector-rotation/backtest` |
| **AI Prediction** | `/api/ai-prediction`, `/api/ai-winratio`, `/api/ai-benchmarks`, `/api/ai-performance` |
| **Appendix → Universe** | `/api/universe`, `/api/meta` |
| **Appendix → Description** | static (DescriptionTab.tsx) |
| **Appendix → Efficacy** | `/api/effectiveness` |
| **Appendix → Reference** | `/api/references` |

> `/api/ml/*` endpoints는 frontend 탭에 직접 노출되지 않음 — ML 운영/디버깅 용. Production frontend는 `/api/pre-momentum`, `/api/table` 같은 cache-backed endpoint 사용.

## Live Scan 트리거 (Frontend → Backend)

```
사용자가 "Run Live Scan" 클릭
   ↓
POST /api/scan?lookback_years=5&use_realtime=true&include_stocks=true
   ↓
api.py:run_scan_api() — subprocess로 price_discovery.py 백그라운드 실행
   ↓
3초마다 GET /api/scan/status 폴링
   ↓
status.running == false → 자동으로 POST /api/reload 호출
   ↓
Frontend setDataVersion(v=>v+1) → 모든 탭 자동 re-fetch
```

## CORS

```python
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
```

개발 환경 가정. Production 시 origin 제한 권장.

## 직접 호출 예시 (cURL)

```bash
# Eligible 종목 조회
curl -s "http://localhost:8000/api/table?eligible_only=true" | jq '.data[] | .ticker' | head -10

# Technology sector만
curl -s "http://localhost:8000/api/table?sectors=Technology&eligible_only=true" | jq '.data | length'

# Pre-Momentum top 5
curl -s "http://localhost:8000/api/pre-momentum" | jq '.candidates[0:5] | .[] | {ticker, pms: .pre_momentum_score, agree: .agreement_ratio}'

# Live scan 시작
curl -X POST "http://localhost:8000/api/scan?lookback_years=5&use_realtime=true"
curl -s "http://localhost:8000/api/scan/status"
```

## Error Handling

- `204 No Content`: 데이터 없음 (cache 미생성)
- `404 Not Found`: 잘못된 path
- `500 Internal Server Error`: cache 손상 / endpoint 함수 예외 — `.api.log` 확인
- `429 Too Many Requests`: 외부 API rate limit (Finnhub) — Frontend는 받지 않음, 백엔드에서 자동 처리
