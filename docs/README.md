# Price Discovery — Documentation Index

이 폴더는 Price Discovery 시스템의 종합 문서입니다. 빠른 시각적 개요는 [`reports/score_dependency_graph.pdf`](../reports/score_dependency_graph.pdf) (6 페이지) 참고.

## 문서 구성

| 문서 | 내용 | 대상 독자 |
|---|---|---|
| [architecture.md](architecture.md) | 시스템 전체 아키텍처 (5-layer dependency, 데이터 흐름, 컴포넌트 관계) | 시스템 전반을 이해하려는 모든 독자 |
| [scoring.md](scoring.md) | Composite 4축 + Pre-Mom 5 agent + QVR + Eligibility Gate 산출 로직 상세 | 점수 의미·임계값을 정확히 이해하고 싶은 분석가 |
| [data-pipeline.md](data-pipeline.md) | yfinance / Finnhub / GraphRAG 데이터 소스 + daily refresh 파이프라인 | 데이터 흐름, cache 구조, refresh 운영자 |
| [api-reference.md](api-reference.md) | FastAPI endpoint 목록 + 응답 schema + frontend 탭 매핑 | API 통합·프론트엔드 개발자 |
| [directory_structure.md](directory_structure.md) | 패키지 기반 파일 트리 + 패키지별 책임 + 실행 가능한 entry 정리 | 코드베이스 신규 진입자 |
| [price_discovery_logic.md](price_discovery_logic.md) | 초기 로직 설계 메모 (기존, 일부 outdated) | 역사적 컨텍스트 |

## 빠른 시작 (Quick Start)

```bash
# 1. 데이터 갱신 (총 ~35분)
python3 price_discovery.py                       # 메인 스캔, ~8분
python3 pipelines/fundamentals_pipeline.py       # yfinance fundamentals, ~5분
python3 pipelines/finnhub_fundamentals.py        # Finnhub 보강, ~28분

# 2. API 시작
python3 -m uvicorn api:app --port 8000 &

# 3. 프론트엔드 실행
cd frontend && npm run dev   # http://localhost:5173

# 4. (선택) 골든 13 회귀 테스트 — 변경 후 무결성 점검
python3 tests/diff_golden.py
```

## 핵심 개념 — 30초 요약

```
입력 (770 ticker)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  4축 Composite Score (TCS, TFS, RSS, URS)                        │
│    "지금 momentum이 있는가?"  →  pure technical                  │
│                                                                  │
│  5 Agent Pre-Momentum Score (Micro, Macro, Graph, Catalyst, QVR)│
│    "곧 momentum이 형성될 것인가?"  →  forward-looking            │
│                                                                  │
│  QVR (Quality + Value + Revision)                                │
│    fundamentals dimension, Composite와 완전 독립 (corr ≈ 0)     │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Eligibility Gate                                                │
│    Composite ≥ 55  AND  bullish classification                  │
│    AND  ADV ≥ $5M  AND  (ETF or QVR ≥ 40)                       │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
   Momentum 탭 (eligible)  vs  Excluded 탭 (rejection tag)
   별도 탭: Pre-Momentum (forming candidates)
```

## 기여 가이드

새 ticker 추가, 새 sub-signal 도입, 또는 가중치 조정 등 변경 시:
1. `price_discovery.py` (또는 `agents/`, `strategies/`, `ml/` 의 해당 파일) 수정
2. `python3 price_discovery.py` 재실행 → cache 갱신
3. 영향받는 `docs/` 페이지 업데이트
4. dependency graph 재생성: `python3 reports/scripts/draw_dependency_graph.py` (또는 sector rotation 변경 시 `draw_sector_rotation_graph.py`)
5. 골든 회귀 테스트로 무결성 확인: `python3 tests/diff_golden.py`. 응답 schema가 의도적으로 바뀐 경우는 `python3 tests/capture_golden.py` 로 baseline 갱신
