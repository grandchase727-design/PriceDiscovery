# Streamlit Dashboard — Market Commentary

`MarketCommentaryTab.tsx` 의 Streamlit 포트입니다. 기존 FastAPI 백엔드(`localhost:8000`)를 그대로 호출합니다.

## 구성

```
streamlit_dashboard/
├── app.py                    # 메인 진입점 (multi-tab)
├── requirements.txt
├── launch.sh                 # 백엔드 확인 + 실행 스크립트
├── .streamlit/config.toml    # 기본 다크 테마
└── lib/
    ├── api.py                # FastAPI 클라이언트 + 캐시
    ├── theme.py              # 다크/라이트 토글 + CSS injection
    ├── utils.py              # 헬퍼 (badge, color, format)
    ├── compute_stats.py      # React computeStats* 포트
    ├── sections.py           # 23개 분석 섹션 (§1-§22)
    ├── widgets.py            # Executive Summary + Market Leaders + Conviction Picks
    ├── swarm.py              # 6-Agent Swarm 분석 패널
    ├── backtest.py           # Trading Layer Backtest 패널
    └── conviction_debate.py  # Multi-Agent Debate 패널
```

## 실행

```bash
# 1. 백엔드 시작 (별도 터미널)
cd ..
python3 -m uvicorn api:app --port 8000 &

# 2. Streamlit 실행
cd streamlit_dashboard
./launch.sh
```

또는 직접:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 접속

- Streamlit: http://localhost:8501
- Backend API: http://localhost:8000

## 테마 토글

사이드바 상단 "Theme" 라디오 버튼으로 **🌑 Dark / ☀️ Light** 전환.

## 탭 구성

1. **📝 Commentary Report** — §1-§22 전체 분석
2. **🤖 Multi-Agent Swarm** — Phase 1-5 6-agent 결과
3. **🎭 Conviction Debate** — Specialist debate verdicts
4. **📊 Backtest** — Trading Layer 백테스트
5. **📦 Raw Data** — 770종목 universe 전체

## 데이터 흐름

```
사용자 → Streamlit → FastAPI (:8000) → 캐시된 데이터 + LLM 결과
```

Streamlit은 데이터 처리/계산만 수행 — LLM 호출/스캔/swarm 실행은 모두 FastAPI 백엔드에서.
