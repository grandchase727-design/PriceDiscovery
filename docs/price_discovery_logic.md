# Price Discovery System — Logic Documentation

> 마지막 업데이트: 2026-04-29
> 대상: Price Discovery 탭(Pre-Momentum / Momentum / Excluded 서브탭)에 적용된 모든 로직

---

## 1. 시스템 개요

Price Discovery는 글로벌 ETF + 개별주식 유니버스(약 770종목)에 대한 정량 모멘텀 스캐너입니다. 매 스캔마다 종목별로 다음 산출물을 생성합니다:

- **Classification** (14개 분류): 추세 상태 식별
- **Composite Score** (0-100): 종합 모멘텀 점수
- **Eligibility**: 포트폴리오 편입 가능 여부
- **8개 Hedge Strategy 점수**: 독립적 검증 레이어
- **Multi-horizon 수익률 + 변동성**: 9개 시간축

이 산출물들이 **Pre-Momentum / Momentum / Excluded** 3개 서브탭으로 재구성되어 사용자에게 노출됩니다.

```
유니버스 (770) → Classification 분류 → Eligibility 평가
                                              │
                ┌─────────────────────────────┼─────────────────────────────┐
                ▼                             ▼                             ▼
       Pre-Momentum (197)            Momentum (376)              Excluded (183)
       not eligible AND              eligible=True             bearish OR rejection
       in 6 PM classes
```

---

## 2. Classification 로직

### 2.1 기본 3×3 매트릭스

매 종목은 단기(SMA20)와 장기(SMA50/200) 추세 방향의 조합으로 분류됩니다.

| Short \ Long | UP | FLAT | DOWN |
|---|---|---|---|
| **UP** | 🟢 CONTINUATION | 🔵 RECOVERY | 🟣 COUNTER_RALLY |
| **FLAT** | 🟡 CONSOLIDATION | 🟠 NEUTRAL | 🟤 FADING |
| **DOWN** | 🔶 PULLBACK | ⚠️ WEAKENING | ⬇️ DOWNTREND |

방향성 판정 기준:
- **단기 UP**: `sma20_dist > 0.5%` AND `sma20_slope > 0`
- **단기 DOWN**: `sma20_dist < -0.5%` AND `sma20_slope < 0`
- **단기 FLAT**: 그 외
- **장기 UP**: `sma50_dist > 1%` AND (`sma50_sma200_spread > 0` OR `sma50_slope > 0`)
- **장기 DOWN**: `sma50_dist < -1%`
- **장기 FLAT**: 그 외

### 2.2 Override 규칙 (매트릭스 결정 후 적용)

| Override | 조건 | 의미 |
|---------|------|------|
| **🔵 FORMATION** | TFS_short ≥ 50 + trend_age_short ≤ 5 + long_dir = UP | 매우 초기 돌파 |
| **🟡 OVEREXTENDED** | OER ≥ 60 on bullish base class | 추세는 건강하나 단기 과열 |
| **🔴 CYCLE_PEAK** | reversal_pctile ≥ 85 + ret_36_12m > 30% + ret_12_1m < 5% + 단기 DOWN | 장기 사이클 정점 |
| **🟤 EXHAUSTING** | trend_age > 60 + ret_63d > 5% + ret_21d < 0 + 장기 UP | 오랜 추세 에너지 소진 |

Override 우선순위: OVEREXTENDED → FORMATION → CYCLE_PEAK → EXHAUSTING

### 2.3 추가 분류: LAGGING_CATCHUP

AQR underreaction 가설 기반의 카테고리 후행 종목 식별.
- **조건**: URS ≥ 75 + 단기 미상승 + 장기 미하락 + base ∈ {CONS/NEUT/PULL}
- **의미**: 카테고리 leader가 먼저 움직였으나 본 종목은 미반응 → 따라잡기 후보
- **Stage**: Momentum (eligible)

### 2.4 14개 Classification 그룹화

서브탭 매핑은 `eligibility` AND `classification`의 조합으로 결정됩니다.

#### Bullish — Momentum Group (eligible=True)
| Class | Tier | 행동 |
|-------|------|------|
| 🟢 CONTINUATION | A — Confirmed | HOLD / SCALE |
| 🔵 FORMATION | A — Confirmed | ENTER (초기 진입) |
| 🟦 LAGGING_CATCHUP | A — Confirmed | ENTER (카테고리 catch-up) |
| 🟡 OVEREXTENDED | A — Caution | HEDGE / TRIM (과열 주의) |

#### Building — Pre-Momentum Group (eligible=False AND in PM classes)
| Class | 의미 | 행동 |
|-------|------|------|
| 🔵 RECOVERY | 단기 UP × 장기 FLAT | WATCH (회복 초기) |
| 🟡 CONSOLIDATION | 단기 FLAT × 장기 UP | WATCH (돌파 임박) |
| 🟠 NEUTRAL | 양 시간축 횡보 (변동성 압축) | WATCH |
| 🔶 PULLBACK | 단기 DOWN × 장기 UP (건강한 조정) | PREPARE |
| ⚠️ WEAKENING | 단기 DOWN × 장기 FLAT | WATCH (약화 진행) |
| 🟤 FADING | 단기 FLAT × 장기 DOWN | WATCH (반등 시그널 주시) |

#### Bearish — Excluded Group
| Class | 의미 | 행동 |
|-------|------|------|
| ⬇️ DOWNTREND | 양 시간축 모두 하락 | AVOID / SHORT |
| 🔴 CYCLE_PEAK | 장기 정점 | EXIT |
| 🟣 COUNTER_RALLY | 장기 하락 중 단기 반등 (dead-cat) | AVOID |
| 🟤 EXHAUSTING | 추세 에너지 소진 | TRIM |

추가로 자격 미달 케이스(LowScore, Liq)도 Excluded에 포함됩니다.

---

## 3. Composite Score & 3-Axis 시그널

### 3.1 공식

```
Composite = 0.35 × TCS + 0.30 × TFS + 0.35 × RSS
```

OER은 분류 전용으로 Composite에는 포함되지 않습니다.

### 3.2 TCS (Trend Continuation Score) — 추세 지속성

- **Short (40%)**: SMA20 거리 연속점수(±2% 버퍼), 기울기, trend_age 단계적(2/5일)
- **Long (60%)**: SMA50 거리(±3%), SMA50-200 spread(±2%), 기울기, trend_age 단계적(5/10/20일), SMA200 거리(±5%)
- 핵심: Binary → 연속 점수 전환으로 노이즈 감소

### 3.3 TFS (Trend Formation Score) — 추세 형성 초기

- **Short (50%)**: SMA20 돌파 강도×신선도, 거래량 4단계(1.1x~2x+), 고점 근접도, slope reversal
- **Long (50%)**: SMA50 돌파 강도, 거래량 단계적(1.05x~1.8x+), 20일 브레이크아웃, slope reversal

### 3.4 OER (Overextension Risk) — 과열 위험

단일 점수로 양 시간축 통합:
- SMA20/50 거리
- RSI overbought
- 52주 고점 근접도
- 36-12M 반전 리스크

### 3.5 RSS (Relative Strength Score) — 다중 horizon 모멘텀

- **Short (35%)**: ret_5d, ret_10d, ret_21d, sma20_slope, vol_ratio_3d_10d 퍼센타일
- **Long (65%)**: ret_12-1m, ret_63d, vol_adj_mom, sma50_slope, range_pct 퍼센타일

---

## 4. Eligibility 평가

Momentum 자격은 3가지 필수 조건의 동시 충족을 요구합니다.

| 조건 | 임계값 | Rejection 이유 |
|------|--------|---------------|
| **Composite Score** | ≥ 55 | LowScore |
| **Classification** | bullish (CONTINUATION/FORMATION/RECOVERY/OVEREXTENDED 등) | Downtrend / CyclePeak / Exhausting / Fading / CounterRally |
| **ADV** | ≥ $5M (일평균 거래대금) | Liq($XM) |

이 조건을 모두 충족하면 `eligible=True` → Momentum 탭에 표시.

---

## 5. Pre-Momentum Detection (Multi-Agent Framework)

### 5.1 대상 필터링

Pre-Momentum 분석 대상 = `classification ∈ 6개 PM classes` AND `eligible=False`

```python
PM_CLASSIFICATIONS = {NEUTRAL, CONSOLIDATION, RECOVERY, PULLBACK, WEAKENING, FADING}
```

이 필터링은 Momentum 그룹과의 중복을 방지합니다.

### 5.2 4-Agent 시스템

| Agent | Weight | 역할 | 주요 신호 |
|-------|--------|------|----------|
| **Microstructure** | 30% | 가격/거래량 미시구조 | 변동성 압축, 축적 패턴, 구조적 괴리, 거래량 레짐, 레인지 수축 |
| **Macro Regime** | 20% | 매크로 환경 | 섹터 로테이션, 크로스에셋 정렬, 카테고리 breadth, 상대 개선도 |
| **Graph Relational** | 25% | 동료 관계 분석 | peer lead, 테마 breadth, leader-lagger gap, community momentum |
| **Catalyst Proxy** | 25% | 트리거 신호 | 모멘텀 가속, 전략 합의도, 스코어 궤적, 반전 리스크 |

### 5.3 Pre-Momentum Score 산출

```
PM Score = 0.30×Micro + 0.20×Macro + 0.25×Graph + 0.25×Catalyst
```

각 agent는 0-100 점수 + 50 임계값에서 "signaling" 여부 판정.

### 5.4 Conviction Level (4-Agent 합의도)

| 신호 Agent 수 | Conviction | 의미 |
|--------------|-----------|------|
| 3-4 | HIGH | 강한 합의 |
| 2 | MEDIUM | 부분 합의 |
| 1 | LOW | 초기/약한 신호 |
| 0 | NONE | 임계 미달 |

### 5.5 Expected Timeline

| 조건 | Timeline |
|------|----------|
| HIGH + score ≥ 70 | 1-2주 |
| HIGH 또는 MEDIUM+score≥65 | 2-3주 |
| MEDIUM | 3-4주 |
| LOW + score ≥ 50 | 4-6주 |
| 그 외 | 6주+ (speculative) |

### 5.6 FICC 전용 필터

FICC 자산(FI_*, Commodities, Real_Assets, Currency_Vol, Multi_Asset)에는 추가 필터:

| 필터 | 로직 |
|------|------|
| **ADV ≥ $10M** | 저유동성 ETF 제거 (CYB, FXA 등) |
| **중복 자산 그룹화** | 같은 기초자산 ETF 중 최고 ADV만 유지 |
| | Gold: GLDM/IAU/SGOL/AAAU/BAR 제거 → GLD만 |
| | Oil: BNO 제거 → USO만 |
| | Commodity Broad: GSG/PDBC/COMT/FTGC 제거 → DBC만 |

---

## 6. Pre-Momentum Age 추적

### 6.1 산출 방식: Bi-weekly Backtest

`ve_observations` (2주 간격, 24개 시점, 약 1년) 데이터를 사용해 **데일리 노이즈 무시**.

### 6.2 알고리즘

```
1. confirmed_dates = [today]  (현재 PM 상태는 confirmed로 간주)
2. ve_observations를 시간 역순으로 순회
3. 각 obs마다:
   - Tier A (CONTINUATION/FORMATION) → BREAK (돌파 발생)
   - PM 분류 (NEUTRAL/CONSOL/RECOVERY 등) → confirmed_dates 추가
   - 그 외 (bearish) → gap 1회 허용, 2회째는 BREAK
4. confirmed_dates ≥ MIN_PERSISTENCE(2)면 age 인정
5. age = (today - oldest confirmed) days, 90일 cap
```

### 6.3 파라미터

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| MAX_GAPS | 1 | 비PM 1회 허용 |
| MIN_PERSISTENCE | 2 | today + 1+ 과거 obs |
| MAX_AGE_DAYS | 90 | 3개월 cap |

### 6.4 Color Mapping

- 0d: 회색 (신규)
- 1-13d: 노랑 (단기)
- 14-29d: 시안 (중기)
- 30-89d: 초록 (장기 압축)
- 90d cap: 진초록 (최장기)

---

## 7. Momentum Age 추적

### 7.1 3-Tier Classification System

Pre-Momentum과 다른 핵심: **Pullback은 건강한 조정**으로 처리.

| Tier | Classifications | 처리 |
|------|----------------|------|
| **A — Confirmed** | 🟢 CONTINUATION, 🔵 FORMATION, 🟡 OVEREXTENDED | Age 카운트 |
| **B — Gap-Tolerant** | 🔵 RECOVERY, 🟡 CONSOLIDATION, 🔶 PULLBACK | 최대 2회 gap 허용 |
| **C — Hard Break** | ⬇️ DOWNTREND, 🔴 CYCLE_PEAK, 🟣 COUNTER_RALLY, 🟤 EXHAUSTING/FADING, ⚠️ WEAKENING | 최대 1회 gap (시장 이벤트), 2회 연속이면 BREAK |

### 7.2 파라미터

| 파라미터 | 값 |
|---------|-----|
| MAX_B_GAPS | 2 |
| MAX_C_GAPS | 1 |
| MAX_TOTAL_GAPS | 3 |
| MIN_PERSISTENCE | 2 (today 포함) |
| MAX_AGE_DAYS | 180 |

### 7.3 알고리즘 핵심

```python
# today 자동 confirmed (Tier A 상태이므로)
confirmed_dates = [today]
b_gaps = c_gaps = 0
prev_was_c = False

for obs in past_observations:
    if obs.cls in TIER_A:
        confirmed_dates.append(obs.date)
        prev_was_c = False
    elif obs.cls in TIER_B:
        if b_gaps < MAX_B_GAPS and (b_gaps + c_gaps) < MAX_TOTAL_GAPS:
            b_gaps += 1; prev_was_c = False
        else: break
    elif obs.cls in TIER_C:
        if prev_was_c: break  # 2 consecutive C → 진정한 반전
        if c_gaps >= MAX_C_GAPS or (b_gaps + c_gaps) >= MAX_TOTAL_GAPS: break
        c_gaps += 1; prev_was_c = True

if len(confirmed_dates) >= MIN_PERSISTENCE:
    age = min((today - min(confirmed_dates)).days, MAX_AGE_DAYS)
```

### 7.4 시장 이벤트 노이즈 처리 예시

```
000660.KS (SK하이닉스):
  4/20 OVEREXTENDED → A (today, confirmed)
  4/03 DOWNTREND   → C, gap 1회 (시장 급락 흡수)
  3/19 CONTINUATION → A (confirmed)
  3/04 CONSOLIDATION → B, gap 1회
  2/11 CONTINUATION → A (confirmed)
  1/27 OVEREXTENDED → A (confirmed)
  → age = today - 1/27 ≈ 84d
```

### 7.5 Color Mapping

- 0d: 회색 (신규)
- 1-29d: 초록 (Fresh)
- 30-59d: 시안 (Mid)
- 60-89d: 주황 (Mature)
- ≥90d: 빨강 (Aged/Cap)

---

## 8. 8-Hedge Strategy Layer

### 8.1 전략 구성

각 종목에 대해 8개 독립 전략이 Long/Short 점수(0-100)를 산출.

| Strategy | Weight | Long 핵심 | Short 핵심 |
|----------|--------|----------|-----------|
| **O'Neil CANSLIM** | 1.5x | Pivot + Volume + RS + MA + Base breakout | Support breakdown + Distribution + RS weakness + MA deterioration |
| **Minervini SEPA** | 1.3x | Stage 2 template (price > SMA50 > SMA150 > SMA200) | Stage 4 template |
| **Wyckoff** | 1.2x | Accumulation phase (OBV↑) | Distribution phase |
| **Ichimoku** | 1.0x | 구름 위 + 치코우 bullish | 구름 아래 + 치코우 bearish |
| **Darvas Box** | 0.8x | 박스 돌파 + 거래량 급증 | 박스 하단 이탈 |
| **Regime** | 1.2x | 시장 레짐 적응형 | Bear 레짐에서 short 보너스 |
| **Flow** | 1.1x | OBV + MFI + 기관 매집 | OBV↓ + Distribution day 빈도 |
| **Relative Value** | 0.9x | 카테고리 내 상위 + 벤치마크 초과 | 카테고리 최하위 |

### 8.2 Aggregation

```
combined_long  = weighted_avg(long scores by weight)
combined_short = weighted_avg(short scores by weight)
long_count  = count(strategy where long ≥ 60)   # 0-8
short_count = count(strategy where short ≥ 60)  # 0-8
```

### 8.3 Net Signal 분류

| Signal | 조건 |
|--------|------|
| **STRONG_LONG** | long_count ≥ 6 AND (combined_long − combined_short) > 20 |
| **LONG** | long_count ≥ 4 AND net > 10 |
| **NEUTRAL** | 양 방향 모두 임계 미달 |
| **SHORT** | short_count ≥ 4 AND net < -10 |
| **STRONG_SHORT** | short_count ≥ 6 AND net < -20 |

---

## 9. Sector Mapping

43개 Category → 17개 Global Sector로 매핑.

### 9.1 Stock 매핑

`Ticker → STOCK_THEMES_CONSOLIDATED → THEME_TO_SECTOR`

49개 theme → 11개 GICS sector + 보조 sectors (Real Estate, Utilities).

주요 매핑 수정사항:
- `Uranium & Nuclear Fuel` → **Energy** (이전: Materials)
- `Power & Energy Infra` → **Utilities** (이전: Industrials)

### 9.2 ETF 매핑

Category 직접 매핑.

| Category Pattern | Sector |
|-----------------|--------|
| EQ_Technology / EQ_Healthcare / EQ_Financials / EQ_ConsDisc / EQ_ConsStaples / EQ_Industrials / EQ_Energy / EQ_Materials / EQ_Utilities / EQ_RealEstate / EQ_CommServices | 각각 GICS 섹터 매칭 |
| EQ_Broad / EQ_Factor / EQ_Thematic | Equity Broad |
| Intl_Developed / Emerging_Markets / Korea_Equity | International |
| FI_* | Fixed Income |
| Commodities | Commodities |
| Real_Assets | Real Assets |
| Currency_Vol | Macro |
| Multi_Asset | Multi Asset |

---

## 10. Multi-Horizon Returns & Volatility

각 종목 9개 시간축 + 1개 변동성 측정값:

| 칼럼 | 의미 | 데이터 필드 |
|------|------|-----------|
| 1D | 1거래일 수익률 | ret_1d |
| 1W | 5거래일 수익률 | ret_5d |
| 1M | 21거래일 수익률 | ret_21d |
| 3M | 63거래일 수익률 | ret_63d |
| 6M | 126거래일 수익률 | ret_126d |
| 1Y | 252거래일 수익률 | ret_252d |
| 3Y/A | 3년 연환산 수익률 | ret_3y_ann |
| 5Y/A | 5년 연환산 수익률 | ret_5y_ann |
| Vol3Y | 3년 연환산 변동성 (σ) | vol_3y_ann |

활용:
- 1D/1W: 진입 타이밍
- 1M-6M: 추세 강도 확인
- 1Y/3Y/5Y: 장기 트랙 레코드
- Vol3Y: 포지션 사이징 (Sharpe-적응형 비중)

---

## 11. Pre-Momentum Conversion Tracking (1M 백테스트)

### 11.1 산출 방식

매 스캔마다 즉시 산출 (별도 daily 누적 불필요).

```
For each ticker in current results:
  if eligible_1m == False AND 25 ≤ score_1m < 55:
    → 1개월 전 PM 후보였음
    if current eligible == True OR current_class in {CONTINUATION, FORMATION}:
      → graduated (성공 전환)
    elif current_class in {DOWNTREND, CYCLE_PEAK, COUNTER_RALLY} OR current_composite < 25:
      → failed (하락 전환)
    else:
      → in_progress (아직 빌딩 중)
```

### 11.2 출력

- **Graduated**: 1개월 전 PM 상태였다가 현재 모멘텀 확인된 종목
- **Failed**: 1개월 전 PM 상태였다가 현재 bearish로 전환된 종목
- **In Progress**: 아직 PM 또는 빌딩 중인 종목
- **Hit Rate**: graduated / (graduated + failed) %

이 백테스트는 Pre-Momentum 시스템의 성능을 retrospective하게 검증합니다.

---

## 12. 주요 시각화

### 12.1 Classification Distribution Heatmap

`ve_observations` 6개 시점 × 13 classification 매트릭스. 각 시점에 각 분류에 속한 종목 수를 heatmap으로 표시. 우측에 최신 변화량(Δ) 칼럼.

### 12.2 Sector Classification Grid

43개 sector별 mini-table:
- 행: classification 약어
- 열: 6개 시점
- 셀: 종목 수
- Δ 칼럼: 최신 기간 대비 변화량

### 12.3 Ticker Search Panel

상단 globally available search:
- ticker 또는 name으로 검색
- 최대 10개 결과 + stage 뱃지 (Pre-Momentum / Momentum / Excluded)
- 선택 시 상세 정보 (4축 점수, 9축 수익률, PM-only 추가 정보)
- "Go to {tab} →" 자동 이동

---

## 13. 데이터 흐름

```
price_discovery.py (스캐너)
    ├─ DataEngine: yfinance 다운로드 (5년치, 770종목)
    ├─ NaiveDiscoveryDetector: TCS/TFS/OER/RSS 점수 산출
    ├─ Classification: 3×3 매트릭스 + overrides
    ├─ Eligibility: composite/class/ADV 평가
    ├─ Hedge Strategies: 8개 전략 점수
    ├─ SignalValidityEngine: 24개 시점 backtest (ve_observations)
    └─ Output → .scan_cache.pkl
            │
            ▼
      api.py (FastAPI)
            ├─ /api/table → Master Table
            ├─ /api/pre-momentum → 4-Agent + age + conversion
            ├─ /api/classification-history → bi-weekly heatmap
            ├─ /api/classification-history-by-sector → sector grid
            └─ /api/scan → 백그라운드 재스캔 (60분 timeout)
            │
            ▼
   frontend (React)
            ├─ Price Discovery 탭
            │   ├─ Pre-Momentum 서브탭
            │   ├─ Momentum 서브탭
            │   ├─ Excluded 서브탭
            │   ├─ Ticker Search Panel
            │   ├─ Classification History Heatmap
            │   ├─ Sector Classification Grid
            │   └─ Classification Definitions (14개)
            └─ 기타 탭 (Market Environment, Analysis, Pipeline, Appendix)
```

---

## 14. 파라미터 상수 요약

### Eligibility
- `composite_threshold`: 55 (adaptive 가능)
- `min_adv`: $5,000,000

### Pre-Momentum
- Agent weights: micro=0.30, macro=0.20, graph=0.25, catalyst=0.25
- Agent threshold: 50 (signal 임계)
- Conviction levels: HIGH(3-4), MEDIUM(2), LOW(1), NONE(0)

### Pre-Momentum Age
- MAX_GAPS = 1
- MIN_PERSISTENCE = 2
- MAX_AGE_DAYS = 90

### Momentum Age
- MAX_B_GAPS = 2
- MAX_C_GAPS = 1
- MAX_TOTAL_GAPS = 3
- MIN_PERSISTENCE = 2
- MAX_AGE_DAYS = 180

### FICC Filter
- MIN_ADV (FICC) = $10,000,000
- Duplicate group dedup: Gold, Oil_WTI, Commodity_Broad

### Hedge Strategy
- Long signal threshold: 60
- Short signal threshold: 60
- Strategy weights: O'Neil 1.5x, Minervini 1.3x, Wyckoff/Regime 1.2x, Flow 1.1x, Ichimoku 1.0x, RelVal 0.9x, Darvas 0.8x

---

## 15. 핵심 설계 철학

### 15.1 노이즈 vs 시그널 분리
- 데일리 데이터 노이즈 → 2주 간격 ve_observations 사용
- 단일 분류 변동 → MIN_PERSISTENCE + gap tolerance
- 시장 이벤트성 급락 → Tier C 1회 gap 허용
- Binary 점수 → 연속 점수 (TCS/TFS)

### 15.2 다층 검증
- 단일 신호 의존 회피 → 4-Agent + 8 hedge strategies
- 매트릭스 + Overrides + 추가 분류 (LAGGING_CATCHUP)
- 시간축 다중화 (9개 horizon)

### 15.3 상호배타적 그룹
- Pre-Momentum / Momentum / Excluded는 OVERLAP=0
- eligibility + classification 조합으로 결정
- 유니버스 합산 = 100%

### 15.4 액션 가능성
- 모든 분류에 행동 권고 (HOLD/SCALE/HEDGE/ENTER/WATCH/AVOID)
- Pipeline 탭에서 Lifecycle Stage로 통합
- 포지션 사이징 힌트 제공

---

## 16. 파일별 책임

| 파일 | 역할 |
|------|------|
| `price_discovery.py` | Core scanner — TCS/TFS/OER/RSS, classification, eligibility |
| `pre_momentum.py` | 4-Agent framework, pm_age, mom_age, conversion backtest |
| `hedge_strategies.py` | 8개 hedge strategy 점수 산출 |
| `investment_pipeline.py` | Lifecycle stage 통합 (DORMANT→WATCHLIST→READY→ACTIVE→CAUTION→EXIT) |
| `api.py` | FastAPI 백엔드, 30+ endpoint |
| `frontend/src/components/tabs/PriceDiscoveryTab.tsx` | 메인 라우터 + Search + Heatmap + Grid |
| `frontend/src/components/tabs/PreMomentumTab.tsx` | 4-Agent UI + Conversion Tracking |
| `frontend/src/components/tabs/MomentumTab.tsx` | Eligible 종목 + 8 strategy matrix + age |
| `frontend/src/components/tabs/ExcludedTab.tsx` | bearish + 자격 미달 종목 |
| `frontend/src/hooks/useSort.ts` | 칼럼 정렬 공용 훅 |

---

## 17. 검증 결과 (2026-04-29 기준)

| 그룹 | 종목 수 | 비율 |
|------|--------|------|
| Total Universe | 770 | 100% |
| Pre-Momentum | 197 | 25.6% |
| Momentum | 376 | 48.8% |
| Excluded | 197 | 25.6% |
| **OVERLAP (PM ∩ Mom)** | **0** | 0% (상호배타 검증됨) |

OVEREXTENDED 96종목 중 91개(95%)가 eligible=True → Momentum 탭에 정확히 표시.

1개월 전 PM 백테스트: graduated 257 / failed 45 / in_progress 76 → **Hit Rate 85.1%**.
