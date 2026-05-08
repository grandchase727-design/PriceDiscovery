import { useState } from "react";
import { C } from "../../styles/theme";

// ───── Reusable components ─────

function Section({ title, children, defaultOpen = false, accent }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean; accent?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button
        className="w-full px-5 py-3 text-left text-sm font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span style={{ color: accent }}>{title}</span>
        <span className="text-gray-500 text-xs">{open ? "▼" : "▶"}</span>
      </button>
      {open && <div className="px-5 py-4 bg-[#0d1117] text-sm text-gray-300 leading-relaxed space-y-4">{children}</div>}
    </div>
  );
}

function T({ children, c }: { children: React.ReactNode; c?: string }) {
  return <span className="font-mono text-[11px] px-1 py-0.5 rounded bg-gray-800/60" style={{ color: c || C.cyan }}>{children}</span>;
}

function Tbl({ headers, rows }: { headers: string[]; rows: (string | React.ReactNode)[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-gray-700">
            {headers.map((h, i) => <th key={i} className="py-1.5 px-2 text-left text-gray-500 font-semibold">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className="border-b border-gray-800/50 align-top">
              {row.map((cell, ci) => <td key={ci} className="py-1.5 px-2 text-gray-400">{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <div className="shrink-0 w-7 h-7 rounded-full bg-cyan-900/40 text-cyan-400 flex items-center justify-center text-xs font-bold">
        {n}
      </div>
      <div className="flex-1 pt-0.5">
        <div className="text-xs font-semibold text-gray-300 mb-1">{title}</div>
        <div className="text-[11px] text-gray-500 leading-relaxed">{children}</div>
      </div>
    </div>
  );
}

// ───── Main Efficacy Tab ─────

export function EfficacyTab() {
  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-gray-200">Classification Efficacy — 유효성 검증 절차</h2>
        <p className="text-xs text-gray-500 mt-1">
          Pre-Momentum과 Momentum 각 classification이 실제로 의도된 결과(돌파, 추세 지속, 수익)를
          가져오는지 정량적으로 검증하는 분석 프레임워크.
        </p>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          Overview & Data Source
         ════════════════════════════════════════════════════════════════════ */}

      <Section title="1. Data Source — ve_observations" defaultOpen accent={C.cyan}>
        <p className="text-xs text-gray-500">
          유효성 검증은 <T>SignalValidityEngine</T>이 산출하는 <T>ve_observations</T>를 사용합니다.
          이 데이터는 과거 1년간 24개 평가 시점(약 2주 간격)에서 모든 종목의 분류 + 5/21/63/126/252일
          forward returns + 카테고리 벤치마크 forward returns를 포함합니다.
        </p>

        <div className="bg-[#111827] border border-gray-800 rounded-lg p-3">
          <div className="text-xs font-semibold text-cyan-400 mb-2">ve_observation 단일 레코드 구조</div>
          <Tbl
            headers={["필드", "타입", "설명"]}
            rows={[
              [<T>ticker</T>, "str", "종목 코드"],
              [<T>eval_date</T>, "date", "평가 시점"],
              [<T>classification</T>, "str", "해당 시점의 14개 분류 중 하나"],
              [<T>composite</T>, "float", "Composite score (0-100)"],
              [<T>eligible</T>, "bool", "Momentum 자격 여부"],
              [<T>tcs / tfs / oer</T>, "int", "축별 점수"],
              [<T>net_signal / long_count / short_count</T>, "—", "8 hedge strategy 합산 신호"],
              [<T>fwd_rets</T>, "dict[N → %]", "1~252일 누적 수익률 시리즈"],
              [<T>fwd_bench</T>, "dict[N → %]", "카테고리 벤치마크 누적 수익률"],
              [<T>excess_return</T>, "float", "fwd_return − bench_return (252일 기준)"],
              [<T>fwd_daily</T>, "dict[H → list]", "horizon별 일별 수익률 (drawdown 계산용)"],
            ]}
          />
        </div>

        <p className="text-[11px] text-gray-500">
          이 데이터셋의 크기는 약 <strong className="text-gray-300">24 시점 × 770 종목 ≈ 18,000 관측치</strong>로
          분류별 통계적 유의성 확보에 충분합니다.
        </p>
      </Section>

      {/* ════════════════════════════════════════════════════════════════════
          Momentum Efficacy
         ════════════════════════════════════════════════════════════════════ */}

      <div className="border-l-2 border-cyan-500/40 pl-4 mt-8 mb-2">
        <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wide">Part A — Momentum Classification Efficacy</h3>
        <p className="text-xs text-gray-500 mt-1">대상: 🟢 CONTINUATION · 🔵 FORMATION · 🟡 OVEREXTENDED · 🟦 LAGGING_CATCHUP</p>
      </div>

      <Section title="A.1 분석 목적 — 무엇을 검증하는가?" defaultOpen accent={C.cyan}>
        <p className="text-xs text-gray-500">
          Momentum 분류는 <strong className="text-gray-300">"진입 권고"</strong>를 함의합니다.
          따라서 다음 4가지를 검증합니다.
        </p>

        <Tbl
          headers={["검증 질문", "측정 metric", "성공 기준 (예시)"]}
          rows={[
            ["진입 시 미래 수익이 양(+)인가?", "Mean / Median fwd return @ 1W·1M·3M", "Median ≥ 0, Hit rate ≥ 55%"],
            ["벤치마크를 초과하는가?", "Excess return = fwd_ret − bench_ret", "Mean excess > 0"],
            ["변동성 대비 수익이 합리적인가?", "Sharpe = mean / std, Sortino", "Sharpe > 0.3 (annualized)"],
            ["진입 후 큰 손실이 없는가?", "Max drawdown during forward window", "p95 max DD ≤ -15%"],
            ["분류가 지속되는가?", "Persistence: P(same class @ N days later)", "1M persistence ≥ 60%"],
          ]}
        />
      </Section>

      <Section title="A.2 절차 — 5단계 분석" accent={C.cyan}>
        <Step n={1} title="데이터 추출 (Filtering)">
          ve_observations에서 검증할 분류(예: CONTINUATION) 레코드만 필터링.
          카테고리(ETF / Stock / Korea / FICC)별로 추가 분리하여 segment-level 분석 가능.
          <div className="mt-1"><T>obs_filtered = [o for o in ve_obs if o.classification == &quot;🟢 CONTINUATION&quot;]</T></div>
        </Step>

        <Step n={2} title="Forward Return 분포 산출">
          각 horizon(5d/21d/63d)에서 fwd_return의 통계를 계산.
          <div className="mt-1 grid grid-cols-2 gap-1">
            <div><T>mean, median, p25, p75, std</T></div>
            <div><T>hit_rate = sum(ret &gt; 0) / N</T></div>
            <div><T>excess_mean = mean(ret − bench)</T></div>
            <div><T>sharpe = mean / std × √(252/H)</T></div>
          </div>
        </Step>

        <Step n={3} title="Drawdown 분석">
          fwd_daily 시리즈로부터 forward window 내 최대 손실 계산.
          <div className="mt-1"><T>max_dd = min(cumulative_return) for each obs</T></div>
          분류별 p50, p95 max drawdown 비교 → 진입 시 risk profile 파악.
        </Step>

        <Step n={4} title="Persistence & Transition 분석">
          같은 ticker의 다음 평가 시점 분류를 추적하여 전이 행렬 구성.
          <div className="mt-2 bg-[#111827] border border-gray-800 rounded-lg p-3 text-[11px]">
            <div className="text-gray-400 font-semibold mb-1">Transition Matrix 예시</div>
            <Tbl
              headers={["From \\ To", "🟢 CONT", "🟡 OEXT", "🔵 RECV", "⬇️ DOWN"]}
              rows={[
                ["🟢 CONTINUATION", "65%", "18%", "8%", "9%"],
                ["🔵 FORMATION", "55%", "12%", "20%", "13%"],
                ["🟡 OVEREXTENDED", "30%", "40%", "5%", "25%"],
              ]}
            />
            <div className="text-[10px] text-gray-600 mt-2">
              CONTINUATION의 65% 자기지속 → 강한 추세 안정성 입증.
              OVEREXTENDED의 25% downtrend 전이 → 과열 경고의 타당성 검증.
            </div>
          </div>
        </Step>

        <Step n={5} title="비교 & 통계적 유의성">
          각 분류의 forward return을 <strong className="text-gray-300">universe baseline</strong>(전체 평균)과 비교.
          t-test 또는 bootstrap으로 statistical significance 확인.
          <div className="mt-1"><T>H₀: mean(class_X) = mean(universe)</T> vs <T>H₁: mean(class_X) &gt; mean(universe)</T></div>
          p-value &lt; 0.05이면 분류가 통계적으로 유의미한 신호를 제공.
        </Step>
      </Section>

      <Section title="A.3 분류별 기대 결과 (가설)" accent={C.cyan}>
        <Tbl
          headers={["Class", "기대 1M ret", "기대 Hit rate", "기대 Persistence (1M)", "기대 max DD"]}
          rows={[
            ["🟢 CONTINUATION", "+3% ~ +6%", "≥ 60%", "≥ 65% (강한 자기지속)", "≤ -10%"],
            ["🔵 FORMATION", "+4% ~ +8% (높은 분산)", "≥ 55%", "≥ 50% (전환기)", "≤ -12%"],
            ["🟦 LAGGING_CATCHUP", "+5% ~ +10% (catch-up)", "≥ 60%", "→ CONTINUATION 전이 50%+", "≤ -10%"],
            ["🟡 OVEREXTENDED", "0% ~ +2% (변동성 큼)", "45-55% (불확실)", "→ DOWNTREND 전이 20%+", "≤ -15% (mean reversion 위험)"],
          ]}
        />
        <p className="text-[11px] text-gray-500 mt-2">
          OVEREXTENDED는 <strong className="text-gray-300">caution</strong> 분류 — 평균 수익은 낮고 drawdown은 큼.
          이 가설이 검증되어야 OVEREXTENDED를 별도 처리하는 시스템 설계가 정당화됩니다.
        </p>
      </Section>

      {/* ════════════════════════════════════════════════════════════════════
          Pre-Momentum Efficacy
         ════════════════════════════════════════════════════════════════════ */}

      <div className="border-l-2 border-purple-500/40 pl-4 mt-8 mb-2">
        <h3 className="text-sm font-bold text-purple-400 uppercase tracking-wide">Part B — Pre-Momentum Classification Efficacy</h3>
        <p className="text-xs text-gray-500 mt-1">대상: 🟠 NEUTRAL · 🟡 CONSOLIDATION · 🔵 RECOVERY · 🔶 PULLBACK · ⚠️ WEAKENING · 🟤 FADING</p>
      </div>

      <Section title="B.1 분석 목적 — 다른 metric 필요" defaultOpen accent={C.purple}>
        <p className="text-xs text-gray-500">
          Pre-Momentum 분류는 <strong className="text-gray-300">"관찰 권고"</strong>로, 진입은 아직 합니다.
          따라서 forward return보다는 <strong className="text-gray-300">"얼마나 빨리 모멘텀으로 전환되는가"</strong>가 핵심.
        </p>

        <Tbl
          headers={["검증 질문", "측정 metric", "성공 기준"]}
          rows={[
            ["곧 모멘텀으로 전환되는가?", "Conversion rate to eligible @ 1M·2M·3M", "≥ 30% @ 3M"],
            ["전환까지 시간이 합리적인가?", "Median time-to-conversion (days)", "Median ≤ 60 days"],
            ["하락으로 빠지지 않는가?", "Failure rate → DOWNTREND/etc.", "≤ 25%"],
            ["전환 시 수익률은?", "Forward return after conversion", "Hit rate ≥ 60% 후속 모멘텀"],
            ["PM Score가 예측력을 가지는가?", "Conversion rate by PM Score quintile", "Top quintile ≥ 2× Bottom"],
            ["Conviction이 예측력을 가지는가?", "HIGH vs LOW conversion rate ratio", "HIGH ≥ 2× LOW"],
          ]}
        />
      </Section>

      <Section title="B.2 절차 — 6단계 분석" accent={C.purple}>
        <Step n={1} title="Pre-Momentum 코호트 추출">
          ve_observations에서 PM 분류이면서 <T>eligible=False</T>인 모든 관측치 추출.
          이들을 t=0 코호트로 정의.
          <div className="mt-1">
            <T>cohort = [o for o in ve_obs if o.classification in PM_SET and not o.eligible]</T>
          </div>
        </Step>

        <Step n={2} title="Forward Trajectory 추적">
          각 코호트 멤버에 대해 다음 N개 평가 시점에서 분류 변화를 추적.
          <T>trajectory[ticker, t=0] → [(t+14d, class), (t+28d, class), ...]</T>
        </Step>

        <Step n={3} title="Conversion Event 식별">
          다음 4가지 outcome으로 분류:
          <div className="mt-2 grid grid-cols-2 gap-2">
            <div className="bg-[#111827] border border-green-900/40 rounded p-2">
              <span className="text-green-400 font-semibold text-[11px]">✅ Graduated</span>
              <div className="text-[10px] text-gray-500">eligible=True AND class ∈ &#123;CONT, FORM&#125;</div>
            </div>
            <div className="bg-[#111827] border border-red-900/40 rounded p-2">
              <span className="text-red-400 font-semibold text-[11px]">❌ Failed</span>
              <div className="text-[10px] text-gray-500">class ∈ &#123;DOWN, CYCLE_PEAK, COUNTER_RALLY&#125; or composite &lt; 25</div>
            </div>
            <div className="bg-[#111827] border border-cyan-900/40 rounded p-2">
              <span className="text-cyan-400 font-semibold text-[11px]">🔄 In Progress</span>
              <div className="text-[10px] text-gray-500">아직 PM 또는 인접 building 상태</div>
            </div>
            <div className="bg-[#111827] border border-gray-800 rounded p-2">
              <span className="text-gray-400 font-semibold text-[11px]">— Neutral Exit</span>
              <div className="text-[10px] text-gray-500">PM 빠져나갔지만 그래도 graduate/fail 아님</div>
            </div>
          </div>
        </Step>

        <Step n={4} title="Time-to-Conversion 분포">
          Graduated 코호트의 conversion 소요 시간 분포 산출.
          <div className="mt-1">
            <T>days_to_convert = (graduation_date − cohort_date).days</T>
          </div>
          분포 통계: median, p25, p75 → "이 분류가 보통 얼마나 빨리 모멘텀으로 가는지" 정량화.
        </Step>

        <Step n={5} title="PM Score / Conviction 분위 분석">
          PM Score를 quintile(또는 conviction levels)로 나누어 conversion rate 비교.
          <div className="mt-2 bg-[#111827] border border-gray-800 rounded-lg p-3">
            <div className="text-[11px] text-gray-400 font-semibold mb-1">예상 결과 (가설)</div>
            <Tbl
              headers={["Group", "Conversion @ 1M", "@ 3M", "Median time"]}
              rows={[
                ["HIGH conviction (Q5)", "30%", "55%", "30d"],
                ["MEDIUM (Q4)", "20%", "40%", "45d"],
                ["LOW (Q3)", "12%", "25%", "60d"],
                ["NONE (Q1-Q2)", "5%", "10%", "—"],
              ]}
            />
          </div>
          monotonicity (HIGH &gt; MEDIUM &gt; LOW)가 보이면 PM Score가 valid한 ranking 신호임을 입증.
        </Step>

        <Step n={6} title="False Positive 분석">
          HIGH conviction이지만 fail로 빠진 케이스 case study.
          <ul className="list-disc list-inside mt-1 space-y-0.5 text-[11px] text-gray-500">
            <li>어떤 agent의 신호가 잘못되었는가?</li>
            <li>특정 섹터/카테고리에서 false positive가 집중되는가?</li>
            <li>매크로 환경(시장 급락 등)에 의한 통제 불가 요인인가?</li>
          </ul>
          이 분석이 다음 버전 agent weight 조정의 기반이 됩니다.
        </Step>
      </Section>

      <Section title="B.3 분류별 기대 결과 (가설)" accent={C.purple}>
        <Tbl
          headers={["Class", "Conversion @ 3M", "Median time", "Failure rate", "비고"]}
          rows={[
            ["🔵 RECOVERY", "≥ 35% (강함)", "30-45d", "≤ 20%", "단기 UP × 장기 FLAT — 가장 확실한 빌딩"],
            ["🟡 CONSOLIDATION", "≥ 30%", "45-60d", "≤ 20%", "장기 UP 유지 → 다음 파동 대기"],
            ["🔶 PULLBACK", "≥ 30%", "30-45d", "≤ 25%", "장기 UP 추세 내 단기 조정 → 빠른 복귀 기대"],
            ["🟠 NEUTRAL", "20-25%", "60-90d", "20-30%", "변동성 압축 — coiled spring, 시간 필요"],
            ["⚠️ WEAKENING", "10-15%", "—", "≥ 40%", "추세 약화 진행 — fail 위험 높음"],
            ["🟤 FADING", "10-15%", "—", "≥ 40%", "장기 DOWN — bottom fishing 케이스"],
          ]}
        />

        <div className="mt-3 p-3 bg-[#111827] border border-gray-800 rounded-lg">
          <div className="text-xs font-semibold text-purple-400 mb-1">설계 검증 포인트</div>
          <ul className="list-disc list-inside text-[11px] text-gray-500 space-y-0.5">
            <li>RECOVERY/PULLBACK이 NEUTRAL/FADING보다 conversion rate가 명백히 높아야 함 (≥ 2배)</li>
            <li>WEAKENING/FADING의 failure rate가 다른 PM 분류보다 높아야 함</li>
            <li>실측 결과가 가설을 따르지 않으면 → classification 임계값 또는 PM Score 가중치 재조정</li>
          </ul>
        </div>
      </Section>

      {/* ════════════════════════════════════════════════════════════════════
          Implementation Notes
         ════════════════════════════════════════════════════════════════════ */}

      <div className="border-l-2 border-amber-500/40 pl-4 mt-8 mb-2">
        <h3 className="text-sm font-bold text-amber-400 uppercase tracking-wide">Part C — 구현 가이드</h3>
        <p className="text-xs text-gray-500 mt-1">실제 분석 실행을 위한 기술적 세부사항</p>
      </div>

      <Section title="C.1 분석 코드 모듈 위치" accent={C.orange}>
        <Tbl
          headers={["기능", "파일", "함수/클래스"]}
          rows={[
            ["ve_observations 산출", "price_discovery.py", <T>SignalValidityEngine</T>],
            ["Pre-Momentum 코호트 + conversion (1M)", "pre_momentum.py", <T>_backtest_conversion()</T>],
            ["Momentum age (Tier A/B/C 추적)", "pre_momentum.py", <T>compute_momentum_ages()</T>],
            ["Pre-Momentum age (bi-weekly)", "pre_momentum.py", <T>_enrich_pm_age_robust()</T>],
            ["Effectiveness 통계 (현재 구현)", "api.py", <span><T>/api/effectiveness</T> endpoint</span>],
            ["Effectiveness UI", "EffectivenessTab.tsx", "Analysis 탭 내"],
          ]}
        />
        <p className="text-[11px] text-gray-500 mt-2">
          상세 분류별 efficacy 분석은 향후 <T>classification_efficacy.py</T> 모듈로 분리 예정.
          이 모듈은 위에서 정의한 A.2 / B.2 절차를 자동화합니다.
        </p>
      </Section>

      <Section title="C.2 통계적 유의성 검증" accent={C.orange}>
        <p className="text-xs text-gray-500">
          관측치가 18,000개 수준이지만 분류별로는 100~1000 범위. 다음 통계 기법을 적용합니다.
        </p>
        <Tbl
          headers={["Metric", "테스트", "샘플 크기 가이드"]}
          rows={[
            ["Mean fwd return", "One-sample t-test (vs 0) / Welch's t-test (vs universe)", "n ≥ 30"],
            ["Median fwd return", "Mann-Whitney U test", "n ≥ 30"],
            ["Hit rate (fwd > 0)", "Binomial test (vs 50%)", "n ≥ 50"],
            ["Conversion rate", "Bootstrap 95% CI (1000 resamples)", "n ≥ 100 권장"],
            ["Persistence", "Markov chain stationarity test", "n ≥ 200 transitions"],
          ]}
        />
      </Section>

      <Section title="C.3 출력 형식 — Efficacy Report" accent={C.orange}>
        <p className="text-xs text-gray-500">
          분석 결과는 다음 표준 형식으로 산출하여 향후 자동 비교 가능하게 합니다.
        </p>
        <div className="bg-[#111827] border border-gray-800 rounded-lg p-3 font-mono text-[10px] text-gray-400">
{`{
  "as_of_date": "2026-04-29",
  "lookback_days": 252,
  "classifications": {
    "🟢 CONTINUATION": {
      "n_observations": 1247,
      "fwd_returns": {
        "1W":  {"mean": 1.2, "median": 0.9, "hit_rate": 58.3, "sharpe_ann": 0.45},
        "1M":  {"mean": 4.1, "median": 3.2, "hit_rate": 62.1, "sharpe_ann": 0.62},
        "3M":  {"mean": 9.8, "median": 7.4, "hit_rate": 65.5, "sharpe_ann": 0.78}
      },
      "excess_return_252d": {"mean": 5.2, "info_ratio": 0.51},
      "max_drawdown": {"p50": -4.5, "p95": -12.8},
      "persistence_1m": 0.66,
      "transitions_to": {"OVEREXTENDED": 0.18, "RECOVERY": 0.08, "DOWNTREND": 0.09},
      "p_value_vs_universe": 0.0004
    },
    ...
  }
}`}
        </div>
        <p className="text-[11px] text-gray-500 mt-2">
          이 JSON을 매월 1회 산출하여 <T>.efficacy_history/</T> 폴더에 저장 → 분류 효과의
          시계열 추이도 추적 가능합니다.
        </p>
      </Section>

      <Section title="C.4 한계 & 주의사항" accent={C.orange}>
        <ul className="list-disc list-inside text-[11px] text-gray-500 space-y-1">
          <li><strong className="text-gray-300">Survivorship bias</strong>: 상장폐지 종목은 universe에 없어 결과가 낙관 편향 가능. 향후 historical universe snapshot 구축 필요.</li>
          <li><strong className="text-gray-300">Look-ahead bias</strong>: classification은 t 시점 정보로 산출되며 fwd_return은 t+H 시점 데이터 사용. 코드 검증 시 인덱스 정합성 주의.</li>
          <li><strong className="text-gray-300">Regime dependence</strong>: Bull/Bear 시장 구간을 구분 분석하지 않으면 평균이 왜곡될 수 있음. 가능하면 SPY 추세별 segmentation.</li>
          <li><strong className="text-gray-300">Sample size 불균형</strong>: 분류별 n이 다름 (CONTINUATION 多, OVEREXTENDED 中, FORMATION 少). 통계적 power 가이드 준수.</li>
          <li><strong className="text-gray-300">Multiple comparison</strong>: 14개 분류 × 5 metrics 동시 테스트 시 Bonferroni correction (α=0.05/70 ≈ 0.0007) 적용.</li>
          <li><strong className="text-gray-300">Backtest ≠ live performance</strong>: 슬리피지, 수수료, 유동성 제약 미반영. live 운용 시 마진 감안.</li>
        </ul>
      </Section>

      {/* Summary */}
      <Section title="요약 — Why This Matters" defaultOpen accent={C.green}>
        <p className="text-xs text-gray-400">
          분류는 가설입니다. 검증되지 않은 분류는 단순 라벨링에 불과합니다.
          이 efficacy 분석 절차는 14개 classification 각각이 다음 질문에 정량 답변을 제공합니다.
        </p>
        <ul className="list-disc list-inside text-[11px] text-gray-500 space-y-0.5 mt-2">
          <li><strong className="text-gray-300">Momentum 분류</strong>: "이 분류 진입 시 평균 +X%, drawdown -Y%, persistence Z%로 검증됨"</li>
          <li><strong className="text-gray-300">Pre-Momentum 분류</strong>: "이 분류는 평균 N일 내 W% 확률로 모멘텀 전환됨"</li>
          <li><strong className="text-gray-300">PM Score / Conviction</strong>: "HIGH는 LOW 대비 conversion이 X배 높아 ranking 신호로 valid"</li>
        </ul>
        <p className="text-[11px] text-gray-500 mt-2">
          이러한 정량적 근거가 누적되면 시스템의 신뢰도가 높아지며, 분류 기준이나 가중치 변경 시
          before/after 비교가 가능해집니다.
        </p>
      </Section>
    </div>
  );
}
