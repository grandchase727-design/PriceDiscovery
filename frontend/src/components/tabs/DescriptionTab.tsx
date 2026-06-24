import { useState } from "react";
import { C } from "../../styles/theme";

function Section({ title, children, defaultOpen = false, accent }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean; accent?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
      <button className="w-full px-5 py-3 text-left text-[16px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}>
        <span style={{ color: accent }}>{title}</span>
        <span className="text-[#857F7A] text-[14px]">{open ? "▼" : "▶"}</span>
      </button>
      {open && <div className="px-5 py-4 bg-[#FBEEE3] text-[16px] text-[#33302E] leading-relaxed space-y-4">{children}</div>}
    </div>
  );
}

function T({ children, c }: { children: React.ReactNode; c?: string }) {
  return <span className="font-mono text-[13px] px-1 py-0.5 rounded bg-[#F2E5D7]/60" style={{ color: c || C.cyan }}>{children}</span>;
}

function SigTable({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[14px] border-collapse">
        <thead>
          <tr className="border-b border-[#E6D9CE]">
            {headers.map((h, i) => <th key={i} className="py-1.5 px-2 text-left text-[#857F7A] font-semibold">{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className="border-b border-[#E6D9CE]/50">
              {row.map((cell, ci) => <td key={ci} className="py-1.5 px-2 text-[#66605C]">{cell}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DescriptionTab() {
  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-[20px] font-bold text-[#33302E]">System Description</h2>
        <p className="text-[14px] text-[#857F7A] mt-1">Price Discovery Scanner v5.0 + Pre-Momentum Detection Multi-Agent Framework</p>
      </div>

      {/* ════════════════════════════════════════════════════════════════════
          PART 1: PRICE DISCOVERY
         ════════════════════════════════════════════════════════════════════ */}

      <div className="border-l-2 border-[#0F5499]/40 pl-4 mb-2">
        <h3 className="text-[16px] font-bold text-[#0F5499] uppercase tracking-wide">Part 1 — Momentum Detection</h3>
        <p className="text-[14px] text-[#857F7A] mt-1">확립된 모멘텀(uptrend) 종목 식별 및 추적 — Price Discovery Scanner v5.0 기반</p>
      </div>

      {/* Signal Architecture */}
      <Section title="Signal Architecture — Dual-Timeframe 3-Axis Scoring" defaultOpen accent={C.cyan}>
        <p className="text-[#66605C] text-[14px]">
          3개의 독립 축(TCS, TFS, OER)과 1개의 크로스섹셔널 랭킹(RSS)을 듀얼 타임프레임(단기/장기)으로 측정합니다.
          최종 Composite Score는 포트폴리오 편입, 분류, 백테스트의 기반이 됩니다.
        </p>

        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#0F5499] mb-2">Composite = 0.35 × TCS + 0.30 × TFS + 0.35 × RSS</div>
          <SigTable
            headers={["Axis", "Short (단기)", "Long (장기)", "Weight"]}
            rows={[
              ["TCS (Trend Continuation)", "SMA20 거리 연속점수(±2%), 기울기, trend_age 단계적(2/5)", "SMA50 거리(±3%), SMA50-200 spread(±2%), 기울기, trend_age 단계적(5/10/20), SMA200 거리(±5%)", "40/60%"],
              ["TFS (Trend Formation)", "SMA20 돌파 강도×신선도, 거래량 4단계(1.1x~2x+), 고점 근접도, SMA20 slope reversal", "SMA50 돌파 강도, 거래량 단계적(1.05x~1.8x+), 20일 브레이크아웃, SMA50 slope reversal", "50/50%"],
              ["OER (Overextension Risk)", "SMA20/50 거리 + RSI overbought + 52주 고점 근접도 + 36-12M 반전 리스크 (통합)", "— (분류 전용, Composite에 미포함)", "분류전용"],
              ["RSS (Relative Strength)", "ret_5d, ret_10d, ret_21d, SMA20 기울기, vol_ratio_3d_10d 퍼센타일", "ret_12-1m, ret_63d, vol_adj_mom, SMA50 기울기, range_pct 퍼센타일", "35/65%"],
            ]}
          />
        </div>

        <div className="mt-4">
          <div className="text-[14px] font-semibold text-[#66605C] mb-2">TCS (Trend Continuation Score) — 기존 추세의 지속성</div>
          <p className="text-[14px] text-[#857F7A]">
            이미 형성된 추세가 얼마나 건강하게 유지되고 있는가를 측정합니다.
            SMA20/50/200과의 거리, 기울기, Golden Cross 여부, trend_age(추세 지속 일수)를 기반으로 합니다.
            <strong className="text-[#33302E]"> Binary→연속 점수 전환</strong>으로 노이즈를 감소시켰습니다. 예: SMA50 위에 있으면 1이 아니라,
            SMA50과의 거리에 따라 ±3% 버퍼 내에서 연속적으로 점수가 변합니다.
          </p>
        </div>

        <div className="mt-3">
          <div className="text-[14px] font-semibold text-[#66605C] mb-2">TFS (Trend Formation Score) — 새로운 추세의 형성</div>
          <p className="text-[14px] text-[#857F7A]">
            돌파(breakout)의 강도와 신선도를 측정합니다. SMA20/50을 돌파한 직후 가장 높은 점수를 받으며,
            시간이 지날수록 신선도(freshness)가 감소합니다. 거래량은 4단계로 점수화됩니다:
            <T>1.1x→25</T> <T>1.3x→50</T> <T>1.5x→75</T> <T>2.0x+→100</T>.
            Slope reversal 보너스: 이전 5일 기울기가 음수였다가 양수로 전환되면 추가 점수.
          </p>
        </div>

        <div className="mt-3">
          <div className="text-[14px] font-semibold text-[#66605C] mb-2">OER (Overextension Risk) — 과매수 위험</div>
          <p className="text-[14px] text-[#857F7A]">
            Composite에는 포함되지 않지만 Classification에서 핵심 역할을 합니다.
            SMA20/50 거리가 과도하거나, RSI 70/80 초과, 52주 고점 근접, 36-12개월 반전 리스크(percentile ≥ 85)가 높으면
            OER이 올라갑니다. OER ≥ 60이면 bullish classification을 OVEREXTENDED로 오버라이드합니다.
          </p>
        </div>

        <div className="mt-3">
          <div className="text-[14px] font-semibold text-[#66605C] mb-2">RSS (Relative Strength Score) — 크로스섹셔널 모멘텀 랭킹</div>
          <p className="text-[14px] text-[#857F7A]">
            전체 유니버스 내에서의 상대적 위치를 퍼센타일 랭킹으로 측정합니다. 벤치마크 대비가 아닌 유니버스 내 순위.
            Short(35%): 최근 5/10/21일 수익률, SMA20 기울기, 단기 거래량.
            Long(65%): 12-1개월 모멘텀(skip 최근 1개월), 63일 수익률, 변동성 조정 모멘텀, SMA50 기울기.
          </p>
        </div>
      </Section>

      {/* Classification */}
      <Section title="Classification — Dual-Timeframe 3×3 Matrix + Overrides" accent={C.cyan}>
        <p className="text-[14px] text-[#857F7A] mb-3">
          단기(Short) 방향과 장기(Long) 방향을 각각 UP/FLAT/DOWN으로 판단한 뒤 3×3 매트릭스에서 기본 분류를 결정합니다.
          이후 4가지 오버라이드 규칙이 적용됩니다.
        </p>

        <div className="p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#66605C] mb-2">3×3 Classification Matrix</div>
          <div className="overflow-x-auto">
            <table className="w-full text-[14px] border-collapse text-center">
              <thead><tr className="border-b border-[#E6D9CE]">
                <th className="py-1.5 text-[#857F7A] text-left">Short＼Long</th>
                <th className="py-1.5 text-[#857F7A]">UP</th>
                <th className="py-1.5 text-[#857F7A]">FLAT</th>
                <th className="py-1.5 text-[#857F7A]">DOWN</th>
              </tr></thead>
              <tbody>
                <tr className="border-b border-[#E6D9CE]">
                  <td className="py-1.5 text-[#857F7A] font-medium text-left">UP</td>
                  <td className="py-1.5" style={{color: C.green}}>CONTINUATION</td>
                  <td className="py-1.5" style={{color: C.blue}}>RECOVERY</td>
                  <td className="py-1.5" style={{color: C.purple}}>COUNTER_RALLY</td>
                </tr>
                <tr className="border-b border-[#E6D9CE]">
                  <td className="py-1.5 text-[#857F7A] font-medium text-left">FLAT</td>
                  <td className="py-1.5" style={{color: C.yellow}}>CONSOLIDATION</td>
                  <td className="py-1.5" style={{color: C.orange}}>NEUTRAL</td>
                  <td className="py-1.5" style={{color: C.brown}}>FADING</td>
                </tr>
                <tr>
                  <td className="py-1.5 text-[#857F7A] font-medium text-left">DOWN</td>
                  <td className="py-1.5" style={{color: "#C2701C"}}>PULLBACK</td>
                  <td className="py-1.5" style={{color: "#990F3D"}}>WEAKENING</td>
                  <td className="py-1.5" style={{color: C.red}}>DOWNTREND</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#66605C] mb-2">Overrides (매트릭스 결정 후 적용)</div>
          <SigTable
            headers={["Override", "조건", "의미"]}
            rows={[
              ["OVEREXTENDED", "OER ≥ 60 on bullish base class", "추세는 건강하지만 단기 과열. 진입 타이밍 주의"],
              ["FORMATION", "TFS_short ≥ 50 + trend_age_short ≤ 5 + long_dir = UP", "매우 초기 돌파. 장기 방향은 상승이지만 아직 5일 이내"],
              ["CYCLE_PEAK", "ret_36_12m pctile ≥ 85 + 12M momentum declining + short ≠ UP", "장기 사이클 고점. 36-12개월 수익이 극도로 높고 최근 모멘텀 둔화"],
              ["EXHAUSTING", "trend_age > 60 + recent momentum < long momentum + long = UP", "오래된 추세가 에너지를 잃어가는 중"],
            ]}
          />
        </div>

        {/* ═══ Pre-Momentum vs Momentum vs Excluded 그룹화 ═══ */}
        <div className="mt-4 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#33302E] mb-3">Classification Grouping — Pre-Momentum / Momentum / Excluded</div>
          <p className="text-[13px] text-[#857F7A] mb-3">
            13개 분류는 Price Discovery 서브탭의 3가지 그룹으로 재정렬됩니다. 그룹은 상호배타적이며,
            대시보드의 Pre-Momentum / Momentum / Excluded 서브탭과 1:1 매핑됩니다.
          </p>

          {/* Pre-Momentum Group */}
          <div className="mb-4 border-l-2 border-[#C9B8DC]/40 pl-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[14px] font-bold text-[#7D5BA6] uppercase tracking-wide">Pre-Momentum Group</span>
              <span className="text-[12px] text-[#857F7A]">— 모멘텀 형성 이전 단계 (eligible=False AND in 6 PM classes)</span>
            </div>
            <SigTable
              headers={["Class", "Short / Long", "의미", "투자 행동"]}
              rows={[
                ["🔵 RECOVERY", "UP / FLAT", "단기 반등 시작, 장기는 횡보. 회복 초기 신호", "WATCH — 후속 상승 확인 대기"],
                ["🟡 CONSOLIDATION", "FLAT / UP", "단기 횡보 + 장기 상승. 매집 후 다음 파동 대기", "WATCH — 돌파 임박 가능"],
                ["🟠 NEUTRAL", "FLAT / FLAT", "양 시간축 모두 횡보. 변동성 압축 (coiled spring)", "WATCH — 방향성 결정 전"],
                ["🟤 FADING", "FLAT / DOWN", "장기 하락 + 단기 횡보. 바닥권 안정화 가능성", "WATCH — 반등 시그널 주시"],
                ["🔶 PULLBACK", "DOWN / UP", "장기 상승 추세 내 단기 조정. 건강한 휴식", "PREPARE — 장기 추세 유지 시 재진입 기회"],
                ["⚠️ WEAKENING", "DOWN / FLAT", "단기 하락 + 장기 횡보. 약화 진행", "WATCH — 추세 회복 또는 더 약화 결정"],
              ]}
            />
            <p className="text-[12px] text-[#857F7A] mt-2">
              4-Agent Pre-Momentum scoring 적용 → HIGH/MEDIUM/LOW conviction 부여 → 돌파 타임라인 추정 (1-6주+)
            </p>
          </div>

          {/* Momentum Group */}
          <div className="mb-4 border-l-2 border-[#0F5499]/40 pl-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[14px] font-bold text-[#0F5499] uppercase tracking-wide">Momentum Group</span>
              <span className="text-[12px] text-[#857F7A]">— 모멘텀 확인 단계 (eligible=True, composite ≥ 55, bullish class)</span>
            </div>
            <SigTable
              headers={["Class", "Tier", "의미", "투자 행동"]}
              rows={[
                ["🟢 CONTINUATION", "A — Confirmed", "단기 + 장기 모두 상승. 가장 견고한 추세", "HOLD / SCALE — 핵심 보유 종목"],
                ["🔵 FORMATION", "A — Confirmed", "초기 돌파 (trend_age ≤ 5일). 신선한 매수 기회", "ENTER — 초기 진입 적기, 손절 타이트"],
                ["🟡 OVEREXTENDED", "A — Caution", "상승 추세 + OER ≥ 60. 단기 과열 위험", "HEDGE / TRIM — 부분 청산 또는 풋옵션 헷지"],
              ]}
            />
            <p className="text-[12px] text-[#857F7A] mt-2">
              Momentum Age 추적: bi-weekly observations + 3-Tier gap tolerance (B 2회, C 1회).
              age 0-180일 cap. 8개 hedge strategies로 독립 검증.
            </p>
          </div>

          {/* Excluded Group */}
          <div className="border-l-2 border-[#E0AAAA]/40 pl-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[14px] font-bold text-[#CC0000] uppercase tracking-wide">Excluded Group</span>
              <span className="text-[12px] text-[#857F7A]">— 투자 부적격 (bearish or 자격미달)</span>
            </div>
            <SigTable
              headers={["Class", "Short / Long", "의미", "투자 행동"]}
              rows={[
                ["⬇️ DOWNTREND", "DOWN / DOWN", "양 시간축 모두 하락. 가장 명확한 하락 추세", "AVOID / SHORT 후보"],
                ["🔴 CYCLE_PEAK", "Override", "장기 사이클 고점. 36-12M 수익 극도로 높고 최근 둔화", "EXIT — 장기 정점 신호"],
                ["🟣 COUNTER_RALLY", "UP / DOWN", "장기 하락 중 단기 반등. 데드캣바운스 위험", "AVOID — 가짜 반등 가능성"],
                ["🟤 EXHAUSTING", "Override", "오래된 추세 에너지 소진. 변곡 임박", "TRIM — 차익실현 또는 보호 매도"],
              ]}
            />

            <div className="mt-3 text-[12px] text-[#857F7A]">
              <span className="text-[#66605C] font-semibold">+ Eligibility 미달 케이스</span> (분류는 Bullish이지만 자격 미달):
              <span className="ml-2"><T>LowScore</T> (composite &lt; 55), <T>Liq</T> (ADV &lt; $5M)</span>
            </div>
          </div>
        </div>
      </Section>

      {/* O'Neil CANSLIM + Hedge Strategies */}
      <Section title="Multi-Strategy Signal System — O'Neil + 7 Hedge Strategies" accent={C.cyan}>
        <p className="text-[14px] text-[#857F7A] mb-3">
          Composite Score와 별도로, 8개의 독립적인 매수/매도 전략이 각 종목을 0-100으로 평가합니다.
          이들의 가중 합산이 <T>combined_long</T> / <T>combined_short</T>이며, Top 10 Long/Short 선정에 사용됩니다.
        </p>
        <SigTable
          headers={["Strategy", "Weight", "Long 핵심 로직", "Short 핵심 로직"]}
          rows={[
            ["O'Neil (CANSLIM)", "1.5x", "Pivot 근접(25) + Volume surge(20) + RS rating(20) + MA structure(20) + Base breakout(15)", "Support breakdown(25) + Distribution vol(20) + RS weakness(20) + MA deterioration(20)"],
            ["Minervini", "1.3x", "Stage 2 template: price > SMA50 > SMA150 > SMA200, 모두 상승, 52주 고점 근접", "Stage 4 template: 모든 이동평균 하향 + 가격이 아래"],
            ["Wyckoff", "1.2x", "Accumulation phase 감지: OBV 상승 + 거래량 증가 + 가격 바닥 다지기", "Distribution phase: OBV 하락 + 고점에서 거래량 감소"],
            ["Ichimoku", "1.0x", "구름 위 + 구름 녹색 + 치코우 bullish", "구름 아래 + 구름 적색 + 치코우 bearish"],
            ["Darvas Box", "0.8x", "박스 돌파 + 거래량 급증 + 좁은 박스 범위(5-20일)", "박스 하단 이탈 + 넓은 박스 + 거래량 미확인"],
            ["Regime", "1.2x", "시장 레짐(bull/bear/transition)에 따른 조건부 가중치 조정", "Bear 레짐에서 추가 short 보너스"],
            ["Flow", "1.1x", "OBV 기울기 + MFI > 50 + 기관 매집 패턴", "OBV 하락 + MFI < 50 + Distribution day 빈도"],
            ["Relative Value", "0.9x", "카테고리 내 상대 순위 상위 + 벤치마크 대비 초과수익", "카테고리 최하위 + 벤치마크 대비 열위"],
          ]}
        />
        <p className="text-[12px] text-[#857F7A] mt-2">
          가중치(Weight)는 combined_long/short 계산 시 해당 전략의 영향력을 결정합니다. O'Neil이 가장 높은 1.5x.
        </p>
      </Section>

      {/* Portfolio Eligibility & Top-10 */}
      <Section title="Portfolio Eligibility & Top-10 Selection" accent={C.cyan}>
        <div className="space-y-3">
          <div>
            <div className="text-[14px] font-semibold text-[#66605C] mb-1">Eligibility 조건 (모두 충족)</div>
            <ul className="text-[14px] text-[#857F7A] space-y-0.5 list-disc list-inside">
              <li>Classification이 DOWNTREND, EXHAUSTING, FADING, COUNTER_RALLY, CYCLE_PEAK이 <strong className="text-[#33302E]">아닌</strong> 것</li>
              <li><T>composite ≥ 55</T></li>
              <li><T>ADV ≥ $5M</T> (Average Daily Volume)</li>
            </ul>
          </div>
          <div>
            <div className="text-[14px] font-semibold text-[#0A7D3F] mb-1">Top 10 Strong Long 선정</div>
            <ol className="text-[14px] text-[#857F7A] space-y-0.5 list-decimal list-inside">
              <li>Eligible + bullish classification (또는 APS ≥ 70 Hidden Gems)</li>
              <li><T c={C.cyan}>_long_rank = combined_long × 0.40 + composite × 0.35 + APS × 0.25</T> × EVENT discount(0.7x)</li>
              <li>동일 섹터 최대 3종목 (Sector Cap)</li>
              <li>상위 10개 선택</li>
            </ol>
          </div>
          <div>
            <div className="text-[14px] font-semibold text-[#CC0000] mb-1">Bottom 10 Strong Short 선정</div>
            <ol className="text-[14px] text-[#857F7A] space-y-0.5 list-decimal list-inside">
              <li>Bearish classification (DOWNTREND, WEAKENING, FADING, EXHAUSTING, COUNTER_RALLY, CYCLE_PEAK)</li>
              <li><T c={C.red}>_short_rank = combined_short × 0.40 + (100 - composite) × 0.35 + (100 - APS) × 0.25</T></li>
              <li>동일 섹터 최대 3종목</li>
              <li>상위 10개 선택</li>
            </ol>
          </div>
        </div>
      </Section>

      {/* Key Classes */}
      <Section title="Key Classes & Data Flow" accent={C.cyan}>
        <SigTable
          headers={["Class", "역할", "핵심 출력"]}
          rows={[
            ["DataEngine", "yfinance로 OHLCV 다운로드, adj-close 보정, 실시간 가격 주입", "all_data: {ticker: ETFData(df, valid, market_cap)}"],
            ["NaiveDiscoveryDetector", "듀얼 타임프레임 스코어링 + 분류", "all_raw(38개 지표), all_ranks(퍼센타일), results"],
            ["SignalValidityEngine", "과거 63거래일 12개 평가시점 백테스트", "bucket/class/ticker별 적중률, 분류 전환 매트릭스"],
            ["PriceDiscoveryGraph", "지식그래프 구축 + Louvain 커뮤니티 디텍션", "커뮤니티, 인사이트, 테마 전파, ETF-Stock 괴리"],
            ["VizEngine", "Dark-theme PDF 생성 (모든 페이지)", "Omega(PD_v5_STK)_YYYYMMDD.pdf"],
          ]}
        />
        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg text-[14px] text-[#857F7A]">
          <div className="text-[#66605C] font-semibold mb-1">run_scan() 데이터 플로우</div>
          <div className="font-mono text-[12px] text-[#857F7A] space-y-0.5">
            <div>1. Download ETFs → merge Stock universe</div>
            <div>2. Load benchmarks (per-category, fallback SPY)</div>
            <div>3. compute_raw() → 38개 지표 per ticker</div>
            <div>4. compute_percentile_ranks() → rss_short, rss_long</div>
            <div>5. SignalValidityEngine.compute() → 백테스트</div>
            <div>6. Score + classify → 1W/1M/3M historical snapshots</div>
            <div>7. GraphRAG: build → community detection → insights</div>
            <div>8. Cache → .scan_cache.pkl → API → Dashboard</div>
          </div>
        </div>
      </Section>

      {/* Universe */}
      <Section title="Universe — ETFs & Stocks" accent={C.cyan}>
        <div className="space-y-3">
          <div>
            <div className="text-[14px] font-semibold text-[#66605C] mb-1">GLOBAL_ETF_UNIVERSE (~231 ETFs, 17 categories)</div>
            <p className="text-[14px] text-[#857F7A]">
              <strong className="text-[#33302E]">Equity (7):</strong> US_Equity_Core, US_Sectors, US_Factors, Intl_Developed, Emerging_Markets, Thematic, Korea_Equity<br/>
              <strong className="text-[#33302E]">FICC (10):</strong> FI_Short(9), FI_Intermediate(8), FI_Long(8), FI_Credit(7), FI_Inflation(5), FI_International(8), Commodities, Real_Assets, Currency_Vol, Multi_Asset
            </p>
          </div>
          <div>
            <div className="text-[14px] font-semibold text-[#66605C] mb-1">STOCK_UNIVERSE (~85+ stocks, 10 GICS-aligned categories)</div>
            <p className="text-[14px] text-[#857F7A]">
              STK_Mag7, STK_Semicon, STK_Software, STK_AI_Infra, STK_Healthcare, STK_Financials,
              STK_Consumer, STK_Industrials, STK_Energy_Materials, STK_Korea
            </p>
          </div>
          <p className="text-[12px] text-[#857F7A]">
            Korean ETFs: .KS suffix. 최소 60 거래일 데이터 필요. 신규 종목은 GLOBAL_ETF_UNIVERSE 또는 STOCK_UNIVERSE에 추가.
          </p>
        </div>
      </Section>

      {/* ── Momentum Detection — Core Premise & Eligibility ── */}
      <Section title="Core Premise — What is 'Momentum'?" defaultOpen accent={C.cyan}>
        <p className="text-[14px] text-[#857F7A]">
          Momentum Detection은 <strong className="text-[#33302E]">현재 상승 추세가 확인된 종목</strong>을 식별하고
          그 추세의 건강도와 지속 기간을 추적합니다.
          Pre-Momentum이 "곧 움직일" 종목을 찾는다면, Momentum은 "이미 움직이고 있는" 종목을 다룹니다.
        </p>

        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg font-mono text-[12px] text-[#857F7A]">
          <div className="text-[#66605C] text-[14px] font-semibold mb-2">Lifecycle</div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="px-2 py-1 rounded bg-[#F2E5D7] text-[#857F7A]">DORMANT</span>
            <span className="text-[#857F7A]">→</span>
            <span className="px-2 py-1 rounded bg-[#EFE9F5]/30 text-[#7D5BA6]">PRE-MOMENTUM (Building)</span>
            <span className="text-[#857F7A]">→ 1-3주 →</span>
            <span className="px-2 py-1 rounded bg-[#E3EEF5]/30 text-[#0F5499]">MOMENTUM (Confirmed)</span>
            <span className="text-[#857F7A]">→</span>
            <span className="px-2 py-1 rounded bg-[#F7EDE0]/30 text-[#C2701C]">CAUTION (Overextended)</span>
            <span className="text-[#857F7A]">→</span>
            <span className="px-2 py-1 rounded bg-[#F7E3E3]/30 text-[#CC0000]">EXIT</span>
          </div>
        </div>

        <p className="text-[14px] text-[#857F7A] mt-3">
          <strong className="text-[#33302E]">핵심 설계 철학:</strong> 단일 신호로는 진짜 추세와 거짓 돌파(false breakout)를
          구분하기 어렵습니다. Momentum Detection은 <strong className="text-[#33302E]">3축 직교 점수(TCS/TFS/RSS) +
          OER 위험도 + 8개 독립 hedge strategy 합의 + 시간적 지속성(age)</strong>의 다층 검증을 통해
          진짜 모멘텀만 식별합니다.
        </p>
      </Section>

      {/* Eligibility */}
      <Section title="Eligibility — Who Counts as Momentum?" accent={C.cyan}>
        <p className="text-[14px] text-[#857F7A]">
          Momentum 자격 부여는 <strong className="text-[#33302E]">3가지 필수 조건</strong>의 동시 충족을 요구합니다.
        </p>

        <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="bg-[#FFFFFF] border border-[#9CC3D5]/40 rounded-lg p-3">
            <div className="text-[14px] font-semibold text-[#0F5499] mb-2">1. Composite ≥ 55</div>
            <p className="text-[13px] text-[#857F7A]">
              0.35×TCS + 0.30×TFS + 0.35×RSS 합계 점수가 임계값 이상.
              유니버스 대비 상위권의 추세 강도를 보유.
            </p>
          </div>
          <div className="bg-[#FFFFFF] border border-[#9CC3D5]/40 rounded-lg p-3">
            <div className="text-[14px] font-semibold text-[#0F5499] mb-2">2. Bullish Classification</div>
            <p className="text-[13px] text-[#857F7A]">
              <T c={C.green}>🟢 CONTINUATION</T>, <T c={C.blue}>🔵 FORMATION</T>,
              <T c={C.blue}>🔵 RECOVERY</T> 중 하나.
              하락/소진/사이클피크 분류는 자동 배제.
            </p>
          </div>
          <div className="bg-[#FFFFFF] border border-[#9CC3D5]/40 rounded-lg p-3">
            <div className="text-[14px] font-semibold text-[#0F5499] mb-2">3. ADV ≥ $5M</div>
            <p className="text-[13px] text-[#857F7A]">
              일평균 거래대금 임계값. 유동성이 부족한 종목은 진입/청산 시 슬리피지 위험.
              <T c={C.gray}>Liq</T> rejection으로 분류.
            </p>
          </div>
        </div>

        <div className="mt-4 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#C2701C] mb-2">Override: Overextended → Caution</div>
          <p className="text-[13px] text-[#857F7A]">
            Composite 60+ + Bullish + 정상 ADV 조건을 모두 충족하더라도, OER ≥ 60이면
            <T c={C.yellow}>🟡 OVEREXTENDED</T>로 분류되어 <strong className="text-[#C2701C]">CAUTION 단계</strong>로 격리됩니다.
            Pipeline 탭에서 HEDGE 액션 권장.
          </p>
        </div>
      </Section>

      {/* Momentum Age */}
      <Section title="Momentum Age — How Long Has the Trend Persisted?" accent={C.cyan}>
        <p className="text-[14px] text-[#857F7A]">
          Momentum Age는 <strong className="text-[#33302E]">현재 종목이 확립된 상승 추세에 머물러온 일수</strong>를
          측정합니다. Bi-weekly 관측 데이터(<T>ve_observations</T>, ~14일 간격)를 사용하여
          데일리 노이즈에 영향받지 않고 진정한 구조적 지속성을 산출합니다.
        </p>

        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#0F5499] mb-3">3-Tier Classification System</div>
          <SigTable
            headers={["Tier", "Classification", "처리", "근거"]}
            rows={[
              ["A — Confirmed", "🟢 CONTINUATION, 🔵 FORMATION, 🟡 OVEREXTENDED", "Age 카운트", "확립된 상승 추세"],
              ["B — Gap-Tolerant", "🔵 RECOVERY, 🟡 CONSOLIDATION, 🔶 PULLBACK", "최대 2회 gap 허용", "건강한 조정 — 추세 내 일시적 휴식"],
              ["C — Hard Break", "⬇️ DOWNTREND, 🔴 CYCLE_PEAK, 🟣 COUNTER_RALLY, 🟤 EXHAUSTING/FADING, ⚠️ WEAKENING", "최대 1회 gap 허용 (시장 이벤트 흡수). 2회 연속이면 즉시 중단", "진정한 추세 반전"],
            ]}
          />
        </div>

        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#0F5499] mb-2">Parameters</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[13px]">
            <div><T>MAX_B_GAPS = 2</T></div>
            <div><T>MAX_C_GAPS = 1</T></div>
            <div><T>MAX_TOTAL_GAPS = 3</T></div>
            <div><T>MAX_AGE_DAYS = 180</T></div>
          </div>
          <div className="mt-2 text-[13px] text-[#857F7A]">
            <T>MIN_PERSISTENCE = 2</T> (today + 1+ 과거 obs 필요).
            현재 Tier A 상태는 <strong className="text-[#33302E]">today를 1 confirmed로 자동 포함</strong>하여
            최근 재진입 종목도 합리적 age 부여.
          </div>
        </div>

        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#0F5499] mb-2">Example Walk</div>
          <div className="font-mono text-[12px] text-[#857F7A] space-y-0.5">
            <div>000660.KS (SK하이닉스) ve_observations:</div>
            <div className="ml-2">4/20 OVEREXTENDED → A (today, confirmed)</div>
            <div className="ml-2">4/03 DOWNTREND → C, gap 1회 (시장 급락 흡수)</div>
            <div className="ml-2">3/19 CONTINUATION → A (confirmed, +1)</div>
            <div className="ml-2">3/04 CONSOLIDATION → B, gap 1회</div>
            <div className="ml-2">2/11 CONTINUATION → A (confirmed, +1)</div>
            <div className="ml-2">1/27 OVEREXTENDED → A (confirmed, +1)</div>
            <div className="text-[#0F5499] mt-1">→ age = today − 1/27 = <strong>~84일</strong></div>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-2">
          <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-center">
            <div className="text-[11px] text-[#857F7A]">Color</div>
            <div className="text-[12px] mt-1" style={{ color: C.gray }}>0d (신규)</div>
          </div>
          <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-center">
            <div className="text-[11px] text-[#857F7A]">Color</div>
            <div className="text-[12px] mt-1" style={{ color: C.green }}>1-29d (Fresh)</div>
          </div>
          <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-center">
            <div className="text-[11px] text-[#857F7A]">Color</div>
            <div className="text-[12px] mt-1" style={{ color: C.cyan }}>30-59d (Mid)</div>
          </div>
          <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-center">
            <div className="text-[11px] text-[#857F7A]">Color</div>
            <div className="text-[12px] mt-1" style={{ color: C.orange }}>60-89d (Mature)</div>
          </div>
          <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-center">
            <div className="text-[11px] text-[#857F7A]">Color</div>
            <div className="text-[12px] mt-1" style={{ color: C.red }}>≥90d (Aged/Cap)</div>
          </div>
        </div>
      </Section>

      {/* 8 Hedge Strategies */}
      <Section title="8 Hedge Strategies — Independent Validation Layer" accent={C.cyan}>
        <p className="text-[14px] text-[#857F7A]">
          단일 시그널 의존을 피하기 위해, 학술/업계에서 검증된 <strong className="text-[#33302E]">8개 독립 전략</strong>이
          각 종목에 대해 Long/Short 점수를 0-100으로 산출합니다. 60 이상이면 해당 방향 시그널로 카운트.
        </p>

        <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2">
          {[
            { name: "O'Neil CANSLIM", desc: "피벗 근접 + 거래량 확인 + RS강도 + MA구조 + 베이스 돌파" },
            { name: "Minervini SEPA", desc: "Stage 2 분석 (가격 trend template), VCP 압축 패턴" },
            { name: "Wyckoff", desc: "축적/분배 단계 분석 (price-volume composite operator behavior)" },
            { name: "Ichimoku", desc: "Cloud, Tenkan/Kijun cross, Chikou span 분석" },
            { name: "Darvas Box", desc: "박스 돌파 + 거래량 surge 검증" },
            { name: "Regime", desc: "시장 레짐(bull/transition/bear) 적응형 가중" },
            { name: "Flow", desc: "OBV, A/D Line 등 자금 흐름 분석" },
            { name: "RelVal", desc: "카테고리 벤치마크 대비 상대강도" },
          ].map((s) => (
            <div key={s.name} className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-[13px]">
              <span className="text-[#0F5499] font-semibold">{s.name}</span>
              <span className="text-[#857F7A]"> — {s.desc}</span>
            </div>
          ))}
        </div>

        <div className="mt-4 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#0F5499] mb-2">Aggregation</div>
          <div className="font-mono text-[13px] text-[#66605C] space-y-1">
            <div><T c={C.green}>combined_long</T> = 가중 평균 (Minervini 30% + Wyckoff 20% + Ichimoku 15% + 나머지)</div>
            <div><T c={C.red}>combined_short</T> = 동일 가중으로 short 점수 평균</div>
            <div><T>long_count</T> = 8개 중 long ≥ 60인 전략 개수</div>
            <div><T>short_count</T> = 8개 중 short ≥ 60인 전략 개수</div>
          </div>
        </div>

        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="text-[14px] font-semibold text-[#0F5499] mb-2">Net Signal Classification</div>
          <SigTable
            headers={["Signal", "조건", "의미"]}
            rows={[
              ["STRONG_LONG", "long_count ≥ 6 AND combined_long − combined_short > 20", "강한 매수 합의"],
              ["LONG", "long_count ≥ 4 AND net > 10", "매수 우세"],
              ["NEUTRAL", "양 방향 모두 임계 미달", "방향성 불명확"],
              ["SHORT", "short_count ≥ 4 AND net < -10", "매도 우세"],
              ["STRONG_SHORT", "short_count ≥ 6 AND net < -20", "강한 매도 합의"],
            ]}
          />
        </div>
      </Section>

      {/* Multi-horizon returns */}
      <Section title="Multi-Horizon Returns & Volatility" accent={C.cyan}>
        <p className="text-[14px] text-[#857F7A]">
          단일 horizon이 아닌 <strong className="text-[#33302E]">9개 시간축 수익률 + 3년 변동성</strong>을 동시 표시하여
          단기 과열, 중기 추세, 장기 펀더멘털을 한눈에 평가합니다.
        </p>

        <div className="mt-3 grid grid-cols-3 md:grid-cols-5 gap-2">
          {[
            ["1D", "1거래일"], ["1W", "5거래일"], ["1M", "21거래일"],
            ["3M", "63거래일"], ["6M", "126거래일"], ["1Y", "252거래일"],
            ["3Y/A", "3년 연환산"], ["5Y/A", "5년 연환산"], ["Vol3Y", "3년 연환산 변동성 (σ)"],
          ].map(([k, v]) => (
            <div key={k} className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-center">
              <div className="text-[13px] font-semibold text-[#0F5499]">{k}</div>
              <div className="text-[12px] text-[#857F7A] mt-0.5">{v}</div>
            </div>
          ))}
        </div>

        <p className="text-[13px] text-[#857F7A] mt-3">
          <strong className="text-[#33302E]">활용 예시:</strong> 1D/1W는 진입 타이밍, 1M-6M는 추세 강도 확인,
          1Y/3Y/5Y는 장기 트랙 레코드, Vol3Y는 포지션 사이징(Sharpe-적응형 비중 산출).
        </p>
      </Section>

      {/* Output flow */}
      <Section title="Output — How Momentum Tickers Surface in the Dashboard" accent={C.cyan}>
        <p className="text-[14px] text-[#857F7A]">
          Momentum Detection의 결과는 다음 경로로 사용자에게 노출됩니다.
        </p>

        <div className="mt-3 space-y-2">
          {[
            { label: "Price Discovery → Momentum 서브탭", desc: "Eligible 종목 전체 + ETF/Stock 분리 + 8 hedge strategies 매트릭스" },
            { label: "Pipeline 탭", desc: "ACTIVE / CAUTION / EXIT 단계로 분류, 행동 권고와 포지션 사이징 힌트 제공" },
            { label: "Analysis 탭", desc: "Category/Theme별 momentum 분포, 섹터 로테이션 추적" },
            { label: "Ticker Search", desc: "어떤 종목이 어느 단계(Pre-Momentum/Momentum/Excluded)에 속하는지 즉시 조회" },
          ].map((s) => (
            <div key={s.label} className="flex gap-3 text-[13px] p-2 bg-[#FFFFFF] border border-[#E6D9CE] rounded">
              <div className="text-[#0F5499] font-semibold w-48 shrink-0">{s.label}</div>
              <div className="text-[#857F7A]">{s.desc}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* ════════════════════════════════════════════════════════════════════
          PART 2: PRE-MOMENTUM DETECTION
         ════════════════════════════════════════════════════════════════════ */}

      <div className="border-l-2 border-[#C9B8DC]/40 pl-4 mt-8 mb-2">
        <h3 className="text-[16px] font-bold text-[#7D5BA6] uppercase tracking-wide">Part 2 — Pre-Momentum Detection</h3>
        <p className="text-[14px] text-[#857F7A] mt-1">Multi-Agent Framework: 모멘텀 형성 이전 구조적 전조를 포착</p>
      </div>

      {/* Core Premise */}
      <Section title="Core Premise — Why Pre-Momentum?" defaultOpen accent={C.purple}>
        <p className="text-[14px] text-[#857F7A]">
          Price Discovery(PD)는 <strong className="text-[#33302E]">모멘텀이 이미 형성된 후</strong> 이를 측정합니다
          (TCS/TFS/RSS → Composite → CONTINUATION/FORMATION).
          Pre-Momentum Detection은 그 <strong className="text-[#33302E]">이전 단계</strong> —
          아직 모멘텀으로 분류되지 않았지만 구조적 전조가 보이는 종목을 포착합니다.
        </p>

        <div className="mt-3 p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg font-mono text-[12px] text-[#857F7A]">
          <div className="text-[#66605C] text-[14px] font-semibold mb-2">Timeline</div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="px-2 py-1 rounded bg-[#EFE9F5]/30 text-[#7D5BA6]">Pre-Momentum Detection</span>
            <span className="text-[#857F7A]">→ 1-3주 →</span>
            <span className="px-2 py-1 rounded bg-[#E3EEF5]/30 text-[#0F5499]">PD: FORMATION/RECOVERY</span>
            <span className="text-[#857F7A]">→</span>
            <span className="px-2 py-1 rounded bg-[#E3F0E8]/30 text-[#0A7D3F]">PD: CONTINUATION</span>
            <span className="text-[#857F7A]">→</span>
            <span className="px-2 py-1 rounded bg-[#E3F0E8]/30 text-[#0A7D3F]">Portfolio 편입</span>
          </div>
        </div>

        <p className="text-[14px] text-[#857F7A] mt-3">
          <strong className="text-[#33302E]">핵심 설계 철학:</strong> 단일 시그널로는 노이즈와 구분이 안 됩니다.
          서로 다른 정보원(가격 구조, 매크로, 네트워크, 전략 컨센서스)에서 독립적으로
          "뭔가 변하고 있다"는 약한 신호들이 <strong className="text-[#33302E]">동시에 수렴</strong>할 때 —
          그것이 모멘텀 형성의 전조입니다.
        </p>
      </Section>

      {/* Step 1: Universe Filtering */}
      <Section title="Step 1 — Universe Filtering" accent={C.purple}>
        <p className="text-[14px] text-[#857F7A] mb-3">
          전체 유니버스에서 이미 모멘텀이 확인된 종목을 제외하고, 모멘텀 형성 전 단계에 있는 종목만 분석 대상으로 합니다.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
            <div className="text-[14px] font-semibold text-[#CC0000] mb-1">제외 (이미 모멘텀 or 극단)</div>
            <ul className="text-[13px] text-[#857F7A] space-y-0.5">
              <li>CONTINUATION — 추세 확인됨</li>
              <li>FORMATION — 돌파 진행 중</li>
              <li>OVEREXTENDED — 과열</li>
              <li>CYCLE_PEAK — 사이클 고점</li>
              <li>DOWNTREND — 하락 추세 확정</li>
              <li>COUNTER_RALLY — 하락 중 반등</li>
              <li>EXHAUSTING — 추세 소진</li>
            </ul>
          </div>
          <div className="p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
            <div className="text-[14px] font-semibold text-[#0A7D3F] mb-1">포함 (Pre-Momentum 대상)</div>
            <ul className="text-[13px] text-[#857F7A] space-y-0.5">
              <li><strong className="text-[#33302E]">RECOVERY</strong> — 장기 하락 후 단기 반등 시작</li>
              <li><strong className="text-[#33302E]">NEUTRAL</strong> — 방향성 미정, 어디로든</li>
              <li><strong className="text-[#33302E]">FADING</strong> — 추세 약화지만 반전 가능</li>
              <li><strong className="text-[#33302E]">CONSOLIDATION</strong> — 횡보, 에너지 축적</li>
              <li><strong className="text-[#33302E]">WEAKENING</strong> — 약세지만 바닥 근접 가능</li>
              <li><strong className="text-[#33302E]">PULLBACK</strong> — 건전한 조정</li>
            </ul>
          </div>
        </div>
      </Section>

      {/* Step 2: 4 Agents */}
      <Section title="Step 2 — 4 Independent Specialist Agents" defaultOpen accent={C.purple}>
        <p className="text-[14px] text-[#857F7A] mb-4">
          각 에이전트는 서로 다른 정보 소스에서 독립적으로 0-100 점수를 산출합니다.
        </p>

        {/* Agent 1: Microstructure */}
        <div className="p-4 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-[16px] font-bold" style={{color: C.cyan}}>Agent 1: Microstructure</span>
            <span className="text-[12px] px-1.5 py-0.5 rounded font-semibold bg-[#E3EEF5]/30 text-[#0F5499]">Quant</span>
            <span className="text-[12px] text-[#857F7A]">Weight: 30%</span>
          </div>
          <p className="text-[14px] text-[#857F7A]">"가격은 안 움직였지만, <strong className="text-[#33302E]">구조</strong>가 변하고 있는가?"</p>
          <SigTable
            headers={["Sub-Signal", "산출 로직", "의미"]}
            rows={[
              ["volatility_compression", "100 - (│sma50_dist│×5 + vol_percentile×0.4 + oer×0.4)", "변동성 압축 = 코일 스프링. ATR 축소 + SMA 밀착 + 낮은 OER → 곧 큰 움직임"],
              ["accumulation_pattern", "tfs_short × (1 - tcs/100) × (wyckoff_long/100)", "TCS 낮아서 추세 없지만, TFS_short 상승 시작 = 초기 매집. Wyckoff로 검증"],
              ["structural_divergence", "structural_q - composite + 50", "구조적 품질(SQ)은 높은데 가격(Composite)은 낮음 = 아직 반영 안 된 품질"],
              ["volume_regime", "flow_long (NEUTRAL/CONSOLIDATION일 때 full, 기타 50%)", "가격 flat인데 기관 flow가 올라옴"],
              ["range_contraction", "(100 - │rsi-50│×2) × tightness_factor", "RSI 50 근처 + SMA50 거리 3% 이내 = 극도의 레인지 수축"],
            ]}
          />
          <p className="text-[12px] text-[#857F7A]">가중치: vol_compression 25%, accumulation 25%, structural_div 20%, volume 15%, range 15%</p>
        </div>

        {/* Agent 2: Macro Regime */}
        <div className="p-4 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg space-y-3 mt-4">
          <div className="flex items-center gap-2">
            <span className="text-[16px] font-bold" style={{color: C.blue}}>Agent 2: Macro Regime</span>
            <span className="text-[12px] px-1.5 py-0.5 rounded font-semibold bg-[#E3EEF5]/30 text-[#0F5499]">Quant</span>
            <span className="text-[12px] text-[#857F7A]">Weight: 20%</span>
          </div>
          <p className="text-[14px] text-[#857F7A]">"이 종목의 <strong className="text-[#33302E]">섹터/매크로 환경</strong>이 우호적으로 전환되고 있는가?"</p>
          <SigTable
            headers={["Sub-Signal", "산출 로직", "의미"]}
            rows={[
              ["sector_rotation", "카테고리 내 score_1w > score_1m 비율(60%) + above_sma50 비율(40%)", "같은 섹터 종목들이 전반적으로 개선 중. 섹터 로테이션 방향"],
              ["cross_asset", "관련 매크로 ETF의 ret_5d + 스코어 궤적 평균", "카테고리별 매크로 팩터 매핑: Tech→TLT/IEF, Energy→XLE/USO, Fin→XLF/KRE 등"],
              ["category_breadth", "카테고리 내 ret_5d > 0 비율 → 0-100", "섹터 참여 폭. 폭이 넓어지면 개별종목도 따라갈 가능성"],
              ["relative_improvement", "카테고리 평균 (score_1w - score_1m), 0→50, +20→100 매핑", "카테고리 전체의 1주간 개선 궤적"],
            ]}
          />
          <p className="text-[12px] text-[#857F7A]">가중치: sector_rotation 30%, cross_asset 20%, breadth 25%, improvement 25%</p>
        </div>

        {/* Agent 3: Graph Relational */}
        <div className="p-4 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg space-y-3 mt-4">
          <div className="flex items-center gap-2">
            <span className="text-[16px] font-bold" style={{color: C.purple}}>Agent 3: Graph Relational</span>
            <span className="text-[12px] px-1.5 py-0.5 rounded font-semibold bg-[#EFE9F5]/30 text-[#7D5BA6]">Hybrid</span>
            <span className="text-[12px] text-[#857F7A]">Weight: 25%</span>
          </div>
          <p className="text-[14px] text-[#857F7A]">"<strong className="text-[#33302E]">연관 종목/테마</strong>에서 이미 모멘텀이 시작되었는가? 전이 가능성은?"</p>
          <SigTable
            headers={["Sub-Signal", "산출 로직", "의미"]}
            rows={[
              ["peer_lead", "동일 consolidated theme 내 CONTINUATION/FORMATION 비율 × 120", "테마 피어가 이미 breakout → 후행 가능성. 예: NVDA → AVGO/MRVL"],
              ["theme_breadth", "테마 내 composite > 55 비율 → 0-100", "테마 전반의 모멘텀 건강도. breadth 확장 초기 = 기회"],
              ["leader_lagger_gap", "min(max_peer_composite - my_composite, 60) / 60 × 100", "리더와의 gap 클수록 catch-up potential. Gap 30pt → 50점"],
              ["community_momentum", "GraphRAG 커뮤니티의 avg_composite(40%) + eligible_pct(20%) + bullish%(40%)", "Louvain 커뮤니티가 전반적으로 bullish인가"],
            ]}
          />
          <p className="text-[12px] text-[#857F7A]">가중치: peer_lead 30%, theme_breadth 25%, leader_lagger 25%, community 20%. ETF는 테마 매핑이 없어 peer signal 약함</p>
        </div>

        {/* Agent 4: Catalyst */}
        <div className="p-4 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg space-y-3 mt-4">
          <div className="flex items-center gap-2">
            <span className="text-[16px] font-bold" style={{color: C.orange}}>Agent 4: Catalyst Proxy</span>
            <span className="text-[12px] px-1.5 py-0.5 rounded font-semibold bg-[#E3EEF5]/30 text-[#0F5499]">Quant</span>
            <span className="text-[12px] text-[#857F7A]">Weight: 25%</span>
          </div>
          <p className="text-[14px] text-[#857F7A]">"모멘텀 <strong className="text-[#33302E]">가속</strong>의 초기 신호가 있는가?"</p>
          <SigTable
            headers={["Sub-Signal", "산출 로직", "의미"]}
            rows={[
              ["momentum_acceleration", "max(0, rss_short - rss_long) × 2 + 10 (음수면 rss_short × 0.3)", "단기 RS > 장기 RS = 최근 모멘텀 가속. 장기 추세 약하지만 단기 먼저 반응"],
              ["strategy_agreement", "long_count / (long_count + short_count) × 100", "8개 헤지 전략 중 long 컨센서스 비율. 전환 초기 감지"],
              ["score_trajectory", "3m→1m 개선 30pt + 1m→1w 개선 35pt + 양방향 가속 15pt + 크기 보너스", "score_1w > score_1m > score_3m = 점수가 가속적으로 개선"],
              ["reversal_risk_check", "pctile ≤30→90, ≤50→75, ≤70→55, ≤85→30, >85→10", "안전 점검: 36-12개월 반전 리스크 낮을수록 안전 (safety signal)"],
            ]}
          />
          <p className="text-[12px] text-[#857F7A]">가중치: momentum_accel 30%, strategy_agreement 25%, trajectory 30%, reversal_risk 15%</p>
        </div>
      </Section>

      {/* Step 3: Orchestrator */}
      <Section title="Step 3 — Orchestrator: Agreement & Conviction" accent={C.purple}>
        <div className="p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg mb-3">
          <div className="text-[14px] font-semibold text-[#7D5BA6] mb-2">Final Score</div>
          <div className="font-mono text-[16px] text-[#33302E]">
            Pre-Momentum Score = Micro × <span className="text-[#0F5499]">0.30</span> + Macro × <span className="text-[#0F5499]">0.20</span> + Graph × <span className="text-[#7D5BA6]">0.25</span> + Catalyst × <span className="text-[#C2701C]">0.25</span>
          </div>
        </div>

        <p className="text-[14px] text-[#857F7A] mb-3">
          각 에이전트가 <strong className="text-[#33302E]">독립적으로 50점을 넘기는지</strong> 카운트하여 Conviction을 결정합니다.
          높은 Conviction = 서로 다른 소스의 시그널이 수렴하고 있다는 의미입니다.
        </p>

        <SigTable
          headers={["동의 에이전트 수", "Conviction", "의미"]}
          rows={[
            ["3-4개", "HIGH", "강한 수렴 — 다수의 독립 소스가 동시에 전조를 감지. 1-3주 내 모멘텀 형성 가능성 높음"],
            ["2개", "MEDIUM", "부분 수렴 — 일부 소스에서 전조. 추가 확인 필요"],
            ["1개", "LOW", "초기/약한 신호 — 단일 소스만 감지. 노이즈일 가능성"],
            ["0개", "NONE", "임계값 미달 — 의미있는 전조 없음"],
          ]}
        />
      </Section>

      {/* Step 4: Catalysts & Risks */}
      <Section title="Step 4 — Catalyst & Risk Extraction" accent={C.purple}>
        <p className="text-[14px] text-[#857F7A] mb-3">
          각 서브시그널의 절대값 임계치에서 자동으로 카탈리스트와 리스크 팩터를 추출합니다.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
            <div className="text-[14px] font-semibold text-[#0A7D3F] mb-2">Catalyst 추출 조건</div>
            <ul className="text-[13px] text-[#857F7A] space-y-1">
              <li><span className="text-[#66605C]">Micro:</span> vol_compression ≥ 65 → "Coiled spring setup"</li>
              <li><span className="text-[#66605C]">Micro:</span> accumulation ≥ 55 → "Early accumulation"</li>
              <li><span className="text-[#66605C]">Macro:</span> sector_rotation ≥ 60 → "Sector rotation favorable"</li>
              <li><span className="text-[#66605C]">Macro:</span> breadth ≥ 65 → "Category breadth expanding"</li>
              <li><span className="text-[#66605C]">Graph:</span> peer_lead ≥ 55 → "Peers in CONTINUATION" (종목명 포함)</li>
              <li><span className="text-[#66605C]">Graph:</span> leader_gap ≥ 60 → "Catch-up potential"</li>
              <li><span className="text-[#66605C]">Catalyst:</span> trajectory ≥ 55 → "Score trajectory accelerating"</li>
              <li><span className="text-[#66605C]">Catalyst:</span> strategy ≥ 60 → "Multiple strategies turning long"</li>
            </ul>
          </div>
          <div className="p-3 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
            <div className="text-[14px] font-semibold text-[#CC0000] mb-2">Risk 추출 조건</div>
            <ul className="text-[13px] text-[#857F7A] space-y-1">
              <li>RSI &lt; 45 → "RSI below neutral"</li>
              <li>golden_cross = false → "No golden cross"</li>
              <li>above_sma50 = false → "Price below SMA50"</li>
              <li>reversal_risk ≤ 35 → "Elevated reversal risk"</li>
              <li>cross_asset ≤ 35 → "Macro factor headwinds"</li>
              <li>category_breadth ≤ 35 → "Category breadth narrow"</li>
              <li>peer_lead ≤ 25 → "No peers showing momentum"</li>
              <li>ret_5d &lt; -3% → "Recent weakness"</li>
            </ul>
          </div>
        </div>
      </Section>

      {/* Pipeline */}
      <Section title="PD → Pre-Momentum Pipeline" accent={C.purple}>
        <div className="p-4 bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg">
          <div className="space-y-3 text-[14px]">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#EFE9F5]/40 flex items-center justify-center text-[#7D5BA6] font-bold text-[16px] shrink-0">1</div>
              <div>
                <div className="text-[#33302E] font-semibold">Pre-Momentum HIGH conviction 감지</div>
                <div className="text-[#857F7A]">아직 PD 점수는 낮지만 구조적 전조가 수렴 (NEUTRAL/CONSOLIDATION/RECOVERY)</div>
              </div>
            </div>
            <div className="border-l border-[#E6D9CE] ml-4 h-4"></div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#E3EEF5]/40 flex items-center justify-center text-[#0F5499] font-bold text-[16px] shrink-0">2</div>
              <div>
                <div className="text-[#33302E] font-semibold">PD에서 FORMATION / RECOVERY 감지 (1-3주 후)</div>
                <div className="text-[#857F7A]">TFS_short 상승, 단기 돌파 시작. Pre-Momentum 신호가 현실화</div>
              </div>
            </div>
            <div className="border-l border-[#E6D9CE] ml-4 h-4"></div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#E3EEF5]/40 flex items-center justify-center text-[#0F5499] font-bold text-[16px] shrink-0">3</div>
              <div>
                <div className="text-[#33302E] font-semibold">PD에서 CONTINUATION 확인</div>
                <div className="text-[#857F7A]">TCS 상승, 장기 추세 확인. Composite ≥ 55, eligible = True</div>
              </div>
            </div>
            <div className="border-l border-[#E6D9CE] ml-4 h-4"></div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#E3F0E8]/40 flex items-center justify-center text-[#0A7D3F] font-bold text-[16px] shrink-0">4</div>
              <div>
                <div className="text-[#33302E] font-semibold">Portfolio 편입</div>
                <div className="text-[#857F7A]">Top 10 Strong Long 선정, 8개 전략 컨센서스 확인</div>
              </div>
            </div>
          </div>
        </div>
        <p className="text-[12px] text-[#857F7A] mt-2">
          Pre-Momentum은 PD의 선행 지표 역할 — "아직 PD 점수는 낮지만, 곧 올라갈 구조적 조건이 갖추어진 종목"을 미리 식별합니다.
        </p>
      </Section>
    </div>
  );
}
