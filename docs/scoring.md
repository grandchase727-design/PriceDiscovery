# Scoring Logic — 산출 로직 상세

이 문서는 모든 score의 입력·공식·임계값·해석을 정리합니다. 시각적 레퍼런스는 [`reports/score_dependency_graph.pdf`](../reports/score_dependency_graph.pdf) Page 3-5.

## 목차

1. [Momentum Composite (4축)](#1-momentum-composite-4축)
2. [Pre-Momentum Score (5 agent)](#2-pre-momentum-score-5-agent)
3. [QVR Agent (Quality + Value + Revision)](#3-qvr-agent-quality--value--revision)
4. [agreement_ratio (sidecar)](#4-agreement_ratio-sidecar)
5. [3×3 Classification Matrix](#5-3-3-classification-matrix)
6. [Eligibility Gate](#6-eligibility-gate)
7. [Decision Tags (UI 의사결정)](#7-decision-tags-ui-의사결정)

---

## 1. Momentum Composite (4축)

### 산출 공식

```
Composite = 0.30·TCS + 0.25·TFS + 0.30·RSS + 0.15·URS
```

각 축은 0-100. Composite도 0-100. 임계값 **≥ 55** = momentum 확정 (Eligibility Gate 첫 번째 조건).

### 1.1 TCS — Trend Continuation Score (weight 0.30)

**질문**: 이미 추세가 확립되어 지속 중인가?

**입력** (`price_discovery.py: score_tcs_short/long`):
- SMA20 / SMA50 / SMA200 거리 (가격이 SMA 대비 얼마나 위/아래)
- 각 SMA의 1개월 slope
- Trend age (SMA50 위 연속 일수)
- **장기 가중 60% > 단기 40%** (지속성 우선)

**점수 해석**:
- 0-30: 추세 미확립 / 횡보
- 30-55: 약한 추세 형성 중
- 55-75: 추세 확립
- 75-100: 강하게 확립된 지속 추세

**예시**: NVDA가 SMA50 위 60일+ 유지, 모든 SMA 양의 slope, 가격이 SMA200 대비 +30% → TCS ~ 80-100

### 1.2 TFS — Trend Formation Score (weight 0.25)

**질문**: 지금 *새로* 추세가 형성되고 있는가?

**입력** (`price_discovery.py: score_tfs_short/long`):
- SMA20 / SMA50 돌파 강도 + 신선도 (얼마나 최근 cross)
- vol_ratio_3d_10d (3일 평균 거래량 / 10일 평균)
- 20-day high 근접도
- Slope reversal (음→양 전환)

**점수 해석**:
- 0-30: 돌파 없음 / 횡보
- 30-55: 약한 초기 신호
- 55-75: 확정된 신선한 돌파
- 75-100: 매우 최근의 강한 돌파 + 거래량 급증

**예시**: 방금 SMA50 돌파 + 거래량 1.8배 + 20-day high 근접 → TFS ~ 75-90

### 1.3 RSS — Relative Strength Score (weight 0.30)

**질문**: 우주(770 ticker) 대비 얼마나 outperform하는가?

**입력** (`price_discovery.py: compute_percentile_ranks`):
- 5d / 21d / 63d / 12-1M 수익률 백분위 (cross-sectional)
- sma20_slope, vol_ratio percentile
- **장기 가중 65% > 단기 35%** (장기 momentum 우선)
- 12-1M (Jegadeesh-Titman 1993): 직전 11개월 수익률, 가장 최근 1개월 제외

**점수 해석**:
- Pure cross-sectional ranking: 0-100 = bottom-to-top of universe
- 0-30: bottom quartile
- 55-75: top half
- 75-100: top decile

**중요**: bear market에서도 best-performing stock은 RSS 90+ — *순전히 상대적*

### 1.4 URS — Underreaction Score (weight 0.15)

**질문**: 시장이 새 정보를 충분히 반영했는가? (행동학적 보정)

**입력** (`price_discovery.py: compute_urs`):
- **LeadLag**: peer category 평균 ret_63d − 본인 ret_63d (catch-up potential)
- **AttnGap**: vol_ratio percentile − ret_5d percentile (관심은 있는데 가격 미반응)
- **Drift**: gap_drift_30d (PEAD — Post-Earnings Announcement Drift proxy)
- **Dispersion**: cross-sectional rss_std (정보 비대칭)
- 학술 기반: AQR, Hong-Stein 1999

**점수 해석**:
- 0-30: 시장이 완전 반영 (효율적)
- 55-75: 보통 underreaction
- 75-100: 강한 attention/price gap (catch-up potential 큼)

**예시**: 섹터 peer가 모두 +20%인데 본인은 +5%, 거래량 상승 중 → 높은 LeadLag → URS 70-85

---

## 2. Pre-Momentum Score (5 agent)

### 산출 공식

```
Pre-Mom = 0.20·Microstructure + 0.15·MacroRegime + 0.20·GraphRelational
        + 0.20·Catalyst + 0.25·QVR
```

각 agent는 0-100. Pre-Mom도 0-100. 임계값 없음 — agreement_ratio와 함께 해석.

### 2.1 Microstructure Agent (weight 0.20)

**감지 대상**: 돌파 *이전*의 구조적 패턴 — 저변동성 setup, 매집, RSI 압축.

**Sub-signals** (`agents/pre_momentum.py: class MicrostructureAgent`):
- `volatility_compression`: 저 vol + tight SMA distance + 낮은 OER
- `accumulation_pattern`: TFS_short × (1 - TCS/100) × Wyckoff (낮은 TCS + 형성 중인 TFS = pre-breakout)
- `structural_divergence`: structural_q − composite (양수 = quality building before breakout)
- `volume_regime`: flow_long 상승 + neutral classification
- `range_contraction`: RSI ≈ 50 + sma_dist < 3%

**Why leading**: 저변동성 + tight range가 종종 돌파에 선행. TCS/TFS를 *역패턴*으로 사용 (낮은 TCS + 형성 중인 TFS).

### 2.2 Macro Regime Agent (weight 0.15)

**감지 대상**: top-down regime fit + sector rotation alignment.

**Sub-signals** (`agents/pre_momentum.py: class MacroRegimeAgent`):
- 카테고리 breadth (% of peers eligible)
- Cross-asset alignment (예: Tech 강세 + Defensive 약세 = risk-on)
- `avg_composite` of category peers (peer average)
- Bullish % of peers
- Sector relative strength

**Why leading**: 자기 composite 미사용 (peer aggregate만). Sector 순환 상승 시 개별 종목이 따라가는 것을 감지.

### 2.3 Graph Relational Agent (weight 0.20)

**감지 대상**: GraphRAG 지식 그래프의 theme leadership 신호.

**Sub-signals** (`agents/pre_momentum.py: class GraphRelationalAgent`):
- `peer_lead`: theme 내 peer 중 composite > 55 비율
- `theme_breadth`: theme 전체 momentum 확산 정도
- `leader_lagger_gap`: max(peer composite) − own composite (catch-up 거리)
- `community_momentum`: Louvain community 평균 momentum

**Why leading**: graph engine의 community structure 사용. Theme leader가 달리는데 본인이 lagging이면 catch-up 후보 (leading 가치 가장 높음).

### 2.4 Catalyst Agent (weight 0.20)

**감지 대상**: catalyst 이벤트의 정량 proxy (free tier에서 직접 news/options 데이터 미사용).

**Sub-signals** (`agents/pre_momentum.py: class CatalystAgent`):
- `momentum_acceleration`: rss_short − rss_long (단기 RS 가속)
- `strategy_agreement`: long_count / total (8 헤지 전략 합의)
- `score_trajectory`: score_1w > score_1m > score_3m (composite 개선 추세)
- `reversal_risk_check`: reversal_pctile (낮을수록 안전 — safety check)

**참고**: 주로 **DELTA/CHANGE** 측정 (level 아닌 변화율). Composite 일부 입력 (score_*m, RSS) 사용 → composite와 +0.48 상관 (의도된 — "building momentum" 신호).

### 2.5 QVR Agent (weight 0.25) ★

**감지 대상**: fundamentals dimension — Composite와 fully orthogonal.

상세는 [Section 3](#3-qvr-agent-quality--value--revision) 참고.

**Why leading**: 가격·composite와 무관. Analyst sentiment + fundamental quality가 *technical 확정 이전*에 변화.

---

## 3. QVR Agent (Quality + Value + Revision)

구현: `agents/qvr_agent.py:QVRAgent`. 입력 cache: `.fundamentals_cache.pkl`.

### 산출 공식

```
QVR = 0.30·Q + 0.20·V + 0.50·R
```

R(Revision)이 가장 leading → 50% 가중.

### 3.1 Q (Quality) — weight 0.30

각 sub-signal cross-sectional 백분위 → 가중평균:

| Field | Weight (Q 내부) | 의미 |
|---|---|---|
| `gross_margin` | 0.40 | 매출총이익률 (산업 우월성) |
| `operating_margin` | 0.30 | 영업이익률 (경영 효율) |
| `roe` | 0.30 | 자기자본수익률 (자본효율) |

**Source priority**: Finnhub > yfinance fallback. Finnhub 값은 percent (47.86) → /100 변환.

### 3.2 V (Value) — weight 0.20

가격 대비 cheap = 높은 점수 (역백분위).

| Field | Weight (V 내부) | 의미 |
|---|---|---|
| `forward_pe` | 0.45 | 12개월 예상 PER |
| `peg` | 0.30 | PE / earnings growth (성장 대비 가격) |
| `price_to_book` | 0.25 | 자산 대비 가격 |

각 백분위에 대해 `100 - pctile` 적용 (낮은 PE = 비싼 평가? 아닌, 싼 평가) → 평균.

### 3.3 R (Revision) — weight 0.50 ★ 가장 leading

| Field | Weight (R 내부) | 출처 | 의미 |
|---|---|---|---|
| `net_30d` | 0.30 | yfinance eps_revisions | up_30d − down_30d (분석가 추정 변화 net) |
| `ratio_30d` | 0.25 | yfinance | up_30d / (up+down) (상향 비율) |
| `bullish_change_3m` | 0.20 | Finnhub recommendation history | 3개월간 bullish_ratio 변화 (delta!) |
| `eps_beat_rate` | 0.15 | Finnhub earnings | 최근 4분기 estimate beat 비율 |
| `eps_surprise_avg` | 0.10 | Finnhub earnings | 4분기 평균 surprise % |

학술 기반: Chan-Jegadeesh-Lakonishok 1996 ("Earnings Momentum") — 분석가 추정 변화가 가격 변화에 *시간적으로 선행*.

### 3.4 ETF / 한국 종목 처리

- **ETF**: fundamentals 없음 → QVR = 50 (neutral, 페널티 없음). Eligibility Gate에서도 면제.
- **Korean stocks** (.KS): Finnhub free tier 차단 → yfinance만 사용. R component 신호 약함 (분석가 수 적음 — 현대차 ~3 vs Finnhub US ~30+).

### 3.5 QVR 점수 분포 예시 (308 candidates 기준)

| Ticker | Composite | QVR | Q | V | R | 해석 |
|---|---|---|---|---|---|---|
| NVDA | 80+ | 73 | 90.9 | 48.3 | 73.0 | Strong fundamentals (Q+V+R 동반) |
| LLY | 70 | 56.8 | 93.6 | 17.1 | 50.6 | High quality but expensive |
| **SBUX** | 66 | **31** | 23.8 | 24.8 | 39.4 | **Junk momentum** — 기술적 강하지만 fundamentals 약함 → WeakQVR(32) 강등 |
| TSLA | 30 | 19.6 | 17.6 | 5.7 | 26.3 | Weak across the board |
| TLT (ETF) | varies | 50 | - | - | - | Neutral (ETF, gate 면제) |

---

## 4. agreement_ratio (sidecar)

### 산출 공식

```
agreement_ratio = count(agent_score > 50) / 5
```

5개 agent 중 50점 초과한 개수를 5로 나눈 값. **0.0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0** 6 단계.

### Tier mapping (decidePMAction 분기 기준)

| ratio | tier | 의미 (몇 개 agent firing) | UI 색상 |
|---|---|---|---|
| **≥ 0.6** | **STRONG** | 3개 이상 합의 | 🟢 초록 |
| **0.4-0.6** | **MODERATE** | 2개 합의 | 🔵 시안 |
| **0-0.4** | **WEAK** | 1개만 | 🟡 노랑 |
| **0** | **NONE** | 어떤 agent도 50 초과 안 함 | ⚫ 회색 |

### PM Score와의 차이 (왜 둘 다 필요)

같은 5 agent에서 도출되지만 *서로 다른 정보*:

| 케이스 | Agent 점수 분포 | PM Score (가중합) | agreement_ratio | 신뢰도 |
|---|---|---|---|---|
| **A** | QVR=95, 나머지 4개=20 | ≈ 39 | 0.2 (1/5) | 한 agent만 강함 (narrow) |
| **B** | 5개 모두 = 55 | ≈ 55 | 1.0 (5/5) | 모든 agent 합의 (broad) ★ |
| **C** | 3개=70, 2개=30 | ≈ 54 | 0.6 (3/5) | 다수 동의 |

→ B가 가장 신뢰도 높음. PM Score만 봐서는 A와 B 구분 불가.

---

## 5. 3 × 3 Classification Matrix

### Composite-derived (numeric score와 독립)

short-term direction × long-term direction:

| Short \ Long | UP | FLAT | DOWN |
|---|---|---|---|
| **UP** | 🟢 CONTINUATION | 🔵 RECOVERY | 🟣 COUNTER_RALLY |
| **FLAT** | 🟡 CONSOLIDATION | 🟠 NEUTRAL | 🟤 FADING |
| **DOWN** | 🔶 PULLBACK | ⚠️ WEAKENING | ⬇️ DOWNTREND |

### Override (matrix 적용 후 추가 검사)

| Override | 조건 | 의미 |
|---|---|---|
| 🟡 **OVEREXTENDED** | OER ≥ 60 on bullish cell | 과열 — mean-reversion 위험 |
| 🔵 **FORMATION** | rapid short breakout from low base | RECOVERY 대신 적용 |
| 🟤 **EXHAUSTING** | 오래된 추세 + slope decay | CONTINUATION 대신 적용 |
| 🔴 **CYCLE_PEAK** | ret_36_12m pctile ≥ 85 + 12M momentum 하락 + short ≠ UP | 장기 bull cycle peak signs |
| 🟦 **LAGGING_CATCHUP** | sector 강세 + 본인 lagging | catch-up candidate |

### Bullish set (Eligibility Gate에서 사용)

```
{CONTINUATION, FORMATION, RECOVERY, OVEREXTENDED, LAGGING_CATCHUP}
```

이 5가지 중 하나여야 Eligibility Gate 두 번째 조건 통과.

---

## 6. Eligibility Gate

### 4 조건 (ALL must pass)

```
1. Composite ≥ 55                                        (technical strength)
2. classification ∈ bullish_set                          (5가지 중 하나)
3. ADV ≥ $5M                                             (liquidity floor — avg daily volume USD)
4. asset_type == "ETF"   OR   QVR ≥ 40                   (fundamentals sanity, Stock-only)
```

### 적용 위치

- 게이트 primitive: `core/eligibility.py:evaluate_eligible()` (임계값/조건 정의)
- 임계값 상수: `config/scoring.py` (ELIGIBLE_COMPOSITE=55, ADV_MIN_USD=5e6, QVR_GATE=40)
- 호출 지점: `api.py:_load_cache()` 내 (cache 로드 직후, df 변환 후 적용)
- 통과 → `eligible=True` 유지
- 실패 → `eligible=False`로 강등 + `rejection` 필드에 태그 추가

### Rejection 태그 종류

| 태그 | 의미 |
|---|---|
| `LowScore` | Composite < 55 |
| `Liq($X.XM)` | ADV < $5M (실제 값 표시) |
| `Downtrend` | classification = DOWNTREND |
| `Fading` | classification = FADING |
| `Weakening` | classification = WEAKENING |
| `Exhausting` | EXHAUSTING override 적용 |
| `CyclePeak` | CYCLE_PEAK override 적용 |
| `WeakQVR(<n>)` | Stock + QVR < 40 (junk momentum 필터) |

태그 조합 가능: `Downtrend/LowScore`, `CONTINUATION/WeakQVR(31)` 등.

### ETF vs Stock 비대칭

- **ETF**: fundamentals 없으므로 4번째 조건 면제. 처음 3개만 검증.
- **Stock**: 4개 조건 모두 검증. QVR < 40 ⇒ 기술적 momentum 강하더라도 강등 ("junk momentum" 자동 차단).

이 비대칭은 **사후 포착 보완**의 핵심 — 기술적 momentum이 발생해도 fundamentals 약하면 추세 지속 가능성 낮음.

### 실제 효과 (308 candidates 기준)

- Before QVR gate: 309 eligible
- After QVR gate: **263 eligible** (46개 stock 강등 → WeakQVR)

---

## 7. Decision Tags (UI 의사결정)

### Momentum tab — `decideAction()` (frontend MomentumTab.tsx)

inputs: composite, classification, OER, age, signal

| Action | 조건 | rank (정렬) |
|---|---|---|
| **BUY** | composite ≥ 65 + STRONG_LONG signal | 1 |
| **HOLD** | composite ≥ 55 + LONG/NEUTRAL signal | 4 |
| **TRIM** | OER ≥ 60 (overextended) | 5 |
| **HEDGE** | OER ≥ 70 + classification = OVEREXTENDED | 7 |
| **EXIT** | classification ∈ {DOWNTREND, CYCLE_PEAK} | 9 |
| **WATCH** | composite 55-65, signal NEUTRAL | 6 |

### Pre-Momentum tab — `decidePMAction()` (frontend PreMomentumTab.tsx)

inputs: pre_momentum_score, agreement_ratio, age, classification

| Action | 조건 | rank |
|---|---|---|
| **STRONG PREPARE** | agreement ≥ 0.6 + PM ≥ 75 + age ≥ 14d | 1 |
| **PREPARE** | agreement ≥ 0.6 + PM ≥ 65 | 2 |
| **WATCH (Active)** | agreement ≥ 0.4 + PM ≥ 60 + 3+ catalysts | 3 |
| **WATCH** | agreement ≥ 0.4 + PM ≥ 55 | 4 |
| **TRACK** | weak agreement but PM ≥ 50 | 5 |
| **STAGNANT** | age ≥ 80d but agreement not strong | 6 |
| **IGNORE** | weak agreement + low score | 9 |

**Risk override**: classification = FADING / WEAKENING → IGNORE (또는 PM 강하면 "WATCH (caution)").

### 정렬 — Sort by rank (낮을수록 적극)

UI에서 Decision 칼럼 클릭 시 rank 낮은 것 (가장 적극적 행동) → 높은 것 (회피) 순서로 정렬.
