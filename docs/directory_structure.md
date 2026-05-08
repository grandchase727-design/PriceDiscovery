# 디렉토리 구조 및 정리 이력

## 현재 구조

```
price discovery/
├── .claude/                        # Claude Code 설정
├── .git/, .gitignore, .gitattributes
├── .pipeline_history.json          # investment_pipeline 실행 이력
├── .pm_history.json                # pre_momentum 실행 이력
├── .scan_cache.pkl                 # 스캔 결과 캐시 (~88MB, .gitignore됨)
├── CLAUDE.md                       # 프로젝트 가이드
├── docs/                           # 프로젝트 문서
│   └── directory_structure.md      # (이 파일)
├── reports/                        # PDF 출력물 (run_scan 결과)
│   └── Omega(PD_v5_STK)_YYYYMMDD.pdf
├── archive/                        # 과거 백업 zip 등
├── frontend/                       # (프론트엔드 자산)
├── question/                       # 하위 스크립트/리서치
│
├── price_discovery.py              # 메인 스캐너
├── dashboard.py                    # Streamlit 대시보드
├── api.py                          # API 레이어
├── graph_engine.py                 # GraphRAG
├── investment_pipeline.py
├── pre_momentum.py
├── factor_efficacy.py
├── hedge_strategies.py
└── quant_strategies.py
```

## 규칙

- **PDF 출력**: `price_discovery.py`의 `run_scan()`은 `reports/` 디렉토리에 저장한다.
  경로 생성 로직은 `price_discovery.py` 내 PDF 생성 블록 참고 (`os.makedirs(reports_dir, exist_ok=True)`).
- **`.gitignore`**: `*.pdf`, `.scan_cache.pkl`, `.DS_Store`, `__pycache__/`, `*.pyc` 이미 제외됨.
- **루트 클린 유지**: 코어 `.py` 파일, `CLAUDE.md`, 설정 폴더 외에 파일을 루트에 두지 않는다.
  - 생성된 산출물 → `reports/`
  - 백업/과거 스냅샷 → `archive/`
  - 문서 → `docs/`

## 정리 이력

### 2026-04-21 — 루트 정리 및 경로 수정
- `reports/` 생성, 루트에 쌓여 있던 PDF 22개 (`Omega(PD_v5_STK)_20260327.pdf` ~ `..._20260420.pdf`) 이동.
- `archive/` 생성, `아카이브.zip` (4/6 백업, 구버전 `api.py`/`graph_engine.py`/`price_discovery.py`/`CLAUDE.md`/PDF 1개 포함) 이동.
- `.DS_Store` 삭제 (`.gitignore`에 이미 있어 재생성돼도 추적 안 됨).
- 코드 반영:
  - `price_discovery.py`의 PDF 저장 블록: `pdf_fn`을 `reports/` 아래 절대경로로 변경, `os.makedirs(reports_dir, exist_ok=True)` 추가.
  - `CLAUDE.md`의 "Output:" 줄을 `reports/` 디렉토리 명시로 수정.
- 영향 범위 확인: 다른 모듈에서 PDF 경로나 `아카이브.zip`을 참조하는 곳 없음 (`question/momentum/make_pdf.py`는 자체 폴더에 저장하는 독립 스크립트).
