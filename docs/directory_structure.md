# 디렉토리 구조

## 현재 구조 (패키지 기반 — 2026-05 reorg 이후)

```
PriceDiscovery/
├── .claude/                        # Claude Code 설정
├── .git/, .gitignore, .gitattributes
├── .env                            # 환경 변수 (gitignored)
├── .scan_cache.pkl                 # 스캔 결과 캐시 (~88MB, gitignored)
├── .fundamentals_cache.pkl         # yfinance + Finnhub 통합 fundamentals (gitignored)
├── .finnhub_config.json            # Finnhub API key (gitignored)
├── .pm_history.json                # pre_momentum 7-day 이력 (gitignored)
├── .api.log                        # uvicorn 런타임 로그 (gitignored)
├── CLAUDE.md                       # Claude Code 작업 가이드
├── requirements.txt                # 핀된 Python 의존성
│
├── price_discovery.py              # ★ 메인 스캐너 entry (루트)
├── api.py                          # ★ FastAPI backend entry (루트)
│
├── config/                         # 중앙 상수
│   └── scoring.py                    # Composite weight, ELIGIBLE_COMPOSITE, ADV_MIN_USD, QVR_GATE
│
├── core/                           # 횡단 점수 primitive
│   └── eligibility.py                # evaluate_eligible() — Layer 5 gate
│
├── pipelines/                      # 데이터 fetch 계층
│   ├── fundamentals_pipeline.py      # yfinance fundamentals → .fundamentals_cache.pkl
│   ├── finnhub_client.py             # Finnhub REST wrapper (rate-limit throttle)
│   └── finnhub_fundamentals.py       # Finnhub enricher (US 종목, in-place update)
│
├── agents/                         # Pre-Momentum 에이전트 + 지식 그래프
│   ├── pre_momentum.py               # 5-agent orchestrator (Micro/Macro/Graph/Catalyst/QVR)
│   ├── qvr_agent.py                  # Quality·Value·Revision 점수
│   └── graph_engine.py               # GraphRAG knowledge graph + Louvain communities
│
├── strategies/                     # 헤지 / 퀀트 / 로테이션
│   ├── hedge_strategies.py           # 8 hedge 전략 (long/short signal)
│   ├── quant_strategies.py           # Top-pick generator (per-strategy ranking)
│   ├── sector_rotation.py            # US Sector Rotation Phase 1+2
│   ├── sector_rotation_backtest.py   # Phase 3 — 월별 rebalance 백테스트
│   └── unified_classifier.py         # Classification validation
│
├── ml/                             # ML 파이프라인 + 진단
│   ├── ml_signal_engine.py           # ML 신호 엔진 entry
│   ├── score_ml.py                   # ML 스코어링
│   ├── factor_efficacy.py            # Reverse Factor Model (5 methodology)
│   ├── feature_pipeline.py
│   ├── meta_labeling.py
│   ├── purged_cv.py                  # Purged + Embargoed CV (López de Prado)
│   ├── ablation_harness.py           # Feature ablation
│   ├── optimize_params.py
│   ├── ai_prediction_cache.py
│   ├── regime_expert_selector.py     # MoE — regime 별 expert
│   ├── regime_conditional_diagnostic.py
│   ├── signal_win_ratio.py
│   ├── performance_analytics.py
│   ├── breadth_pipeline.py
│   ├── macro_features.py
│   └── multi_benchmark_validation.py
│
├── frontend/                       # React/Vite 대시보드 (Primary UI)
│   ├── src/
│   │   ├── App.tsx                   # 6 tab + sidebar
│   │   ├── api/client.ts             # API client + FilterParams
│   │   ├── components/tabs/          # 탭 컴포넌트
│   │   └── styles/theme.ts
│   ├── dist/                         # production build
│   ├── package.json, vite.config.ts
│   └── node_modules/
│
├── reports/                        # PDF 출력물 + 의존도 그래프
│   ├── Omega(PD_v5_STK)_YYYYMMDD.pdf  # 일별 스캔 리포트
│   ├── score_dependency_graph.pdf     # 5-layer 점수 의존도 그래프 (6p)
│   ├── us_sector_rotation_graph.pdf   # 섹터 로테이션 의존도 그래프 (4p)
│   └── scripts/                       # 그래프 렌더러 (수동 실행)
│       ├── draw_dependency_graph.py
│       └── draw_sector_rotation_graph.py
│
├── tests/                          # 회귀 테스트
│   ├── golden/                       # 골든 baseline JSON (13개 endpoint)
│   ├── golden_endpoints.py           # baseline 캡처 대상 endpoint 목록
│   ├── capture_golden.py             # 새 baseline 작성 (의도된 변경 후)
│   ├── diff_golden.py                # 현재 응답 vs baseline 비교 (CI 게이트)
│   └── test_no_leakage.py            # Feature/breadth as-of 일관성 테스트
│
├── legacy/                         # 레거시 / 폐지 예정
│   └── dashboard.py                  # Streamlit 대시보드 (호환용)
│
└── docs/                           # 시스템 문서 (이 폴더)
    ├── README.md                     # 문서 인덱스 + 빠른 시작
    ├── architecture.md               # 5-layer dependency, 컴포넌트 매핑
    ├── scoring.md                    # 점수 산출 로직 상세
    ├── data-pipeline.md              # 데이터 소스 + cache 구조 + 일일 갱신
    ├── api-reference.md              # FastAPI endpoint + 응답 schema
    ├── directory_structure.md        # (이 파일)
    ├── price_discovery_logic.md      # 초기 로직 설계 메모 (일부 outdated)
    └── references/                   # 참고 문헌 / 외부 자료
```

## 패키지 구조의 원칙

| 패키지 | 책임 | "이게 들어가면 안 됨" |
|---|---|---|
| `config/` | 상수, 임계값, 가중치 | 함수 로직, 데이터 fetch |
| `core/` | 도메인 primitive (eligibility 등) — 다른 패키지 의존 X | 외부 API 호출 |
| `pipelines/` | 외부 데이터 소스 fetch + cache 작성 | 점수 계산, 분류 |
| `agents/` | 점수/추론 에이전트 (Pre-Mom 5종 + GraphRAG) | 데이터 fetch (pipelines 결과를 입력으로 받음) |
| `strategies/` | 시그널 → 포지션 로직 (헤지/퀀트/로테이션) | UI 렌더링, fetch |
| `ml/` | ML 파이프라인 + 진단 + 검증 | 실시간 핫패스 의존성 (frontend → /api는 cache hit) |
| `frontend/` | React/Vite UI | 백엔드 비즈니스 로직 |
| `reports/` | PDF 산출물 + 그래프 렌더러 | 코드 의존성 (단방향: 코드 → reports) |
| `tests/` | 회귀 baseline + 테스트 | 프로덕션 코드 |
| `legacy/` | 폐지 예정/대체된 코드 | 신규 기능 추가 |
| `docs/` | 사람을 위한 문서 | 실행 코드 |

## 실행 가능한 entry point (루트 + 패키지 내)

```bash
# 루트 — bare 실행
python3 price_discovery.py
python3 -m uvicorn api:app --port 8000

# pipelines/ — sys.path shim 있어 bare 실행 OK
python3 pipelines/fundamentals_pipeline.py
python3 pipelines/finnhub_fundamentals.py

# agents/ — stdlib만 import, bare 실행 OK
python3 agents/qvr_agent.py     # self-test (cache 필요)

# strategies/ — config 패키지 import 때문에 module form 필요
python3 -m strategies.sector_rotation_backtest

# reports/scripts/ — bare 실행 OK
python3 reports/scripts/draw_dependency_graph.py
python3 reports/scripts/draw_sector_rotation_graph.py

# tests/ — bare 실행 OK (API 8000 실행 중이어야 함)
python3 tests/diff_golden.py
python3 tests/capture_golden.py    # baseline 갱신 (의도된 변경 후)
```

## 규칙

- **PDF 출력**: `price_discovery.py:run_scan()` 은 `reports/` 디렉토리에 저장. `os.makedirs(reports_dir, exist_ok=True)`.
- **Cache**: 모든 `.*.pkl`, `.*.json` cache 파일은 `.gitignore`. 재생성은 daily refresh 파이프라인에서.
- **루트 클린 유지**: 루트 `.py`는 `price_discovery.py` + `api.py` 두 개만. 신규 코드는 책임에 맞는 패키지에 배치.
- **신규 패키지 추가 시**: `__init__.py`를 두고, 절대 import 사용 (`from agents.pre_momentum import ...`). 상대 import는 패키지 내부 한정.

## 정리 이력

### 2026-05 — 패키지 기반 reorg
루트의 평탄한 `.py` 14개를 책임별로 분리 (`agents/`, `pipelines/`, `strategies/`, `ml/`, `core/`, `config/`).
의존도 그래프 분리 (`reports/score_dependency_graph.pdf`, `reports/us_sector_rotation_graph.pdf`).
테스트 회귀 baseline (`tests/golden/`) 도입 — 13개 endpoint snapshot.
Streamlit dashboard → `legacy/`로 격하.

### 2026-04-21 — 루트 정리 및 경로 수정
- `reports/` 생성, 루트에 쌓여 있던 PDF 22개 (`Omega(PD_v5_STK)_20260327.pdf` ~ `..._20260420.pdf`) 이동.
- `archive/` 생성, `아카이브.zip` 이동.
- `price_discovery.py`의 PDF 저장 블록을 `reports/` 아래 절대경로로 변경.
