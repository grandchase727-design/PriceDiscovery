# Architecture

Price Discovery 시스템의 전체 아키텍처를 5-layer dependency 모델로 정리합니다. 시각적 레퍼런스는 [`reports/score_dependency_graph.pdf`](../reports/score_dependency_graph.pdf) Page 1.

## 5-Layer 모델

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1 — DATA SOURCES                                              │
│   yfinance Prices      yfinance Fundamentals      Finnhub          │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 2 — DERIVED INDICATORS                                        │
│   Technical signals    Fundamentals (Q+V)    Analyst data (R)      │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 3 — AXES / AGENTS                                             │
│   Composite axes:  TCS · TFS · RSS · URS                            │
│   Pre-Mom agents:  Micro · Macro · Graph · Catalyst · QVR           │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 4 — FINAL SCORES                                              │
│   A. Momentum Composite     B. Pre-Momentum Score    agreement_ratio│
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 5 — DOWNSTREAM                                                │
│   Eligibility Gate → Momentum tab vs Excluded tab                   │
└─────────────────────────────────────────────────────────────────────┘
```

## 컴포넌트 매핑

### 백엔드 (Python)

2026-05 reorg 이후 패키지 구조 — 루트에는 `price_discovery.py` + `api.py` 두 entry만 남고 나머지는 책임별 패키지로 이동했습니다.

| 파일 | 역할 | 주요 엔트리 |
|---|---|---|
| `price_discovery.py` | 메인 스캐너 — 데이터 다운로드 → 4축 산출 → classification → cache 저장 | `run_scan()`, `class NaiveDiscoveryDetector` |
| `api.py` | FastAPI backend — cache loader, QVR 산출, Eligibility Gate, 36+ endpoints | `app`, `_load_cache()` |
| `config/scoring.py` | 중앙 상수 (Composite weight, ELIGIBLE_COMPOSITE, ADV_MIN_USD, QVR_GATE) | 모듈 상수 |
| `core/eligibility.py` | Layer 5 Eligibility Gate primitive | `evaluate_eligible()` |
| `agents/pre_momentum.py` | Pre-Momentum 5-agent orchestrator | `class PreMomentumOrchestrator`, `class CatalystAgent` 등 |
| `agents/qvr_agent.py` | QVR (Quality + Value + Revision) agent — Pre-Mom 5번째 + Eligibility Gate input | `class QVRAgent` |
| `agents/graph_engine.py` | GraphRAG knowledge graph + Louvain community + insights | `class PriceDiscoveryGraph` |
| `pipelines/fundamentals_pipeline.py` | yfinance fundamentals fetcher (병렬, retry-on-rate-limit) | `run_pipeline()`, `retry_failed()` |
| `pipelines/finnhub_client.py` | Finnhub REST wrapper (rate-limit 자동 throttle) | `class FinnhubClient` |
| `pipelines/finnhub_fundamentals.py` | Finnhub로 기존 cache enrich (US ticker만) | `enrich_cache()` |
| `strategies/hedge_strategies.py` | 8개 헤지 전략 (O'Neil, Minervini, Wyckoff, Ichimoku, Darvas, Regime, Flow, RelVal) | `score_all_strategies()`, `compute_combined_signal()` |
| `strategies/quant_strategies.py` | Top-pick generators (per-strategy ranking) | `compute_all_strategies()`, `strategy_sector_rotation()` |
| `strategies/sector_rotation.py` | US Sector Rotation Phase 1+2 (cross-sectional) | `compute_sector_signal()` |
| `strategies/sector_rotation_backtest.py` | Phase 3 — 월별 rebalance 백테스트 | `run_backtest()` (CLI: `python3 -m strategies.sector_rotation_backtest`) |
| `strategies/unified_classifier.py` | Classification validation | `validate_classification()` |
| `ml/factor_efficacy.py` | Reverse Factor Model — factor backtest (5 methodologies) | `compute_factor_efficacy()` |
| `ml/ml_signal_engine.py` | ML 신호 엔진 entry — feature → meta-label → MoE | `MLSignalEngine` |
| `ml/score_ml.py` | ML 스코어링 (production cache writer) | `score_universe()` |
| `ml/regime_expert_selector.py` | Mixture-of-Experts — regime별 expert 가중 | `RegimeExpertSelector` |
| `ml/purged_cv.py` / `meta_labeling.py` / `ablation_harness.py` / `breadth_pipeline.py` 등 | 진단·CV·feature 파이프라인 (López de Prado 방법론) | `PurgedKFold`, `MetaLabeler` 등 |
| `legacy/dashboard.py` | 레거시 Streamlit dashboard (호환용, 신규 개발 X) | `main()` |

### 프론트엔드 (React/Vite)

```
frontend/
├── src/
│   ├── App.tsx                    # 6 tabs + sidebar (Sector + Classification 필터)
│   ├── api/client.ts              # API client + FilterParams 타입
│   ├── components/
│   │   ├── shared/                # MetricCard 등 공용
│   │   ├── tabs/                  # 6 탭 컴포넌트
│   │   │   ├── PriceDiscoveryTab.tsx   # Pre-Momentum / Momentum / Excluded sub-tabs
│   │   │   ├── PreMomentumTab.tsx
│   │   │   ├── MomentumTab.tsx
│   │   │   ├── ExcludedTab.tsx
│   │   │   ├── ValidationTab.tsx
│   │   │   ├── MarketEnvironmentTab.tsx
│   │   │   ├── AnalysisTab.tsx
│   │   │   ├── AIPredictionTab.tsx
│   │   │   ├── AppendixTab.tsx
│   │   │   └── (DescriptionTab, EfficacyTab, UniverseTab, ReferenceTab)
│   │   └── ...
│   └── styles/theme.ts            # 색상 팔레트
└── dist/                          # production build (npm run build)
```

### Cache 파일 (gitignored)

| 파일 | 생성 주체 | 내용 |
|---|---|---|
| `.scan_cache.pkl` | `price_discovery.py` | 770 ticker × ~80 fields (composite, classification, hedge strategies, history, ve_observations 등) |
| `.fundamentals_cache.pkl` | `pipelines/fundamentals_pipeline.py` + `pipelines/finnhub_fundamentals.py` | yfinance fundamentals + Finnhub enrichment (info, estimates, revisions, recommendations, finnhub_metrics, finnhub_derived 등) |
| `.finnhub_config.json` | 사용자 직접 작성 | Finnhub API key |
| `.pm_history.json` | `agents/pre_momentum.py` | Pre-Mom 7-day history (per ticker stage tracking) |
| `.api.log` | `uvicorn` runtime | API server 로그 |

## 데이터 흐름 (런타임)

```
[일 1회 batch refresh]
    price_discovery.py ────────────────────► .scan_cache.pkl
    pipelines/fundamentals_pipeline.py ────► .fundamentals_cache.pkl
    pipelines/finnhub_fundamentals.py ─────► .fundamentals_cache.pkl (in-place enrich)

[API 시작 시]
    api.py:_load_cache() ─────────► STATE["df"] (memory)
        │
        ├── load .scan_cache.pkl → df 구성
        ├── add Sector + SubTheme columns (Option B taxonomy)
        ├── load .fundamentals_cache.pkl → QVRAgent 초기화
        ├── compute QVR score for every ticker → df["qvr_score"], qvr_q/v/r
        └── apply Eligibility Gate → demote Stocks with QVR<40 → rejection: WeakQVR(<n>)

[브라우저 요청]
    Frontend → /api/table → df 필터링 후 JSON 반환
    Frontend → /api/pre-momentum → run_pre_momentum() 즉석 실행 → JSON 반환
    Frontend → /api/factor-efficacy, /api/overview, /api/graph 등 → cached data
```

## Composite와 Pre-Momentum의 관계

두 score는 **상호 보완**적이며, 다른 질문에 답합니다:

| 측면 | Momentum Composite (A) | Pre-Momentum Score (B) |
|---|---|---|
| **질문** | "지금 momentum이 있는가?" | "곧 momentum이 형성될 것인가?" |
| **시점** | Lagging (사후 확인) | Leading (선행 감지) |
| **데이터** | 가격·거래량 (관측 가능) | 가격 + fundamentals + 분석가 (예측 신호) |
| **Eligibility 영향** | Composite ≥ 55 임계값 | 직접 영향 없음 (참조용) |
| **Tab 위치** | Momentum tab | Pre-Momentum tab |
| **Cross-correlation** | — | A vs B = +0.40 (자연 상관) |

A와 B의 *cross-sectional correlation*이 +0.40인 것은 자연스러움 — 둘 다 momentum 신호이기 때문. 진정한 leading 가치는 *temporal precedence* (오늘 B가 높은 ticker가 1-3개월 후 A가 높아지는가)로 측정해야 함. → Conversion Tracking 섹션 (Pre-Mom 탭 하단)에서 hit rate 모니터링.

## QVR의 dual role (이중 역할)

QVR Agent는 시스템 내 유일하게 **두 곳에서 사용**됩니다:

1. **Pre-Mom 5번째 agent** (가중치 0.25) — Pre-Momentum Score 산출에 기여
2. **Eligibility Gate filter** (Stock-only, threshold 40) — Momentum 자격 검증

이는 의도된 설계입니다:
- Pre-Mom 탭에서는 fundamentals이 *forming setup*의 한 차원으로 작동
- Momentum 탭에서는 fundamentals이 *junk momentum 필터*로 작동 (기술적 강세 + 약한 fundamentals = 강등)

## 사용자 시나리오별 entry point

| 시나리오 | 보는 탭 | 핵심 score |
|---|---|---|
| "오늘 어떤 종목을 살까" (현재 strong momentum) | Momentum tab | Composite + Decision |
| "곧 매수할 candidate 찾기" (forming) | Pre-Momentum tab | PM Score + agreement_ratio |
| "왜 이 종목이 빠졌나" (excluded check) | Excluded tab | rejection tag |
| "포트폴리오 위험 점검" (overextended/exit signals) | Momentum tab → OER, Decision="TRIM/HEDGE/EXIT" |
| "macro regime 변화 추적" | Market Environment tab | regime + breadth |
| "특정 종목 deep-dive" | 어느 탭이든 종목 클릭 → detail panel | 모든 axes + agents 분해 |
