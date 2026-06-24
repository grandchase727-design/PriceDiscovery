import { useState, useEffect, useMemo, useCallback } from "react";
import Plot from "react-plotly.js";
import { fetchTable, fetchPreMomentum, fetchClassificationHistory, fetchClassificationHistoryBySector, type FilterParams } from "../../api/client";
import { PreMomentumTab } from "./PreMomentumTab";
import { MomentumTab } from "./MomentumTab";
import { NewPriceDiscoveryTab } from "./NewPriceDiscoveryTab";
import { New2PriceDiscoveryTab } from "./New2PriceDiscoveryTab";
import { ExcludedTab } from "./ExcludedTab";
import { DARK_LAYOUT } from "../../styles/theme";
import { CLASS_COLORS, C } from "../../styles/theme";

// ---------------------------------------------------------------------------
// Sub-tab definitions
// ---------------------------------------------------------------------------

const SUBS = [
  { label: "Pre-Momentum", desc: "Momentum formation candidates — before breakout" },
  { label: "Momentum", desc: "Confirmed momentum — eligible tickers with active trend" },
  { label: "Anti Lag Discovery", desc: "PROVISIONAL: Pre-Mom strong signals (Anti-Lag, 조기 surface)" },
  { label: "Sector Discovery", desc: "Sector-Segmented — 각 섹터별 독립 top-5 선별 (diversification)" },
  { label: "Excluded", desc: "Bearish, overextended, or ineligible — not actionable" },
] as const;

// Pre-momentum classification set (must match backend)
const PM_CLASSIFICATIONS = new Set([
  "\ud83d\udfe0 NEUTRAL", "\ud83d\udfe1 CONSOLIDATION", "\ud83d\udd35 RECOVERY",
  "\ud83d\udd36 PULLBACK", "\u26a0\ufe0f WEAKENING", "\ud83d\udfe4 FADING",
]);

// ---------------------------------------------------------------------------
// Stage badge
// ---------------------------------------------------------------------------

function stageBadge(stage: string): { label: string; color: string; bg: string } {
  switch (stage) {
    case "momentum":
      return { label: "Momentum", color: C.green, bg: C.green + "22" };
    case "pre-momentum":
      return { label: "Pre-Momentum", color: C.cyan, bg: C.cyan + "22" };
    case "excluded":
      return { label: "Excluded", color: C.red, bg: C.red + "22" };
    default:
      return { label: "Unknown", color: C.gray, bg: C.gray + "22" };
  }
}

function compColor(v: number): string {
  if (v >= 70) return C.green;
  if (v >= 55) return C.cyan;
  if (v >= 40) return C.yellow;
  return C.gray;
}

function retColor(v: number): string {
  return v > 0 ? C.green : v < 0 ? C.red : C.gray;
}

function oerColor(v: number): string {
  if (v >= 60) return C.red;
  if (v >= 40) return C.orange;
  return C.green;
}

// ---------------------------------------------------------------------------
// Ticker Lookup Panel
// ---------------------------------------------------------------------------

interface LookupResult {
  ticker: string;
  name: string;
  category: string;
  classification: string;
  composite: number;
  tcs: number;
  tfs: number;
  oer: number;
  rss: number;
  eligible: boolean;
  rejection: string;
  rsi: number;
  trend_age: number;
  sma50_dist: number;
  net_signal: string;
  long_count: number;
  short_count: number;
  ret_1w: number;
  ret_1m: number;
  ret_3m: number;
  adv_M: number;
  stage: string;
  // PM fields (only if pre-momentum)
  pm_score?: number;
  pm_agreement_ratio?: number;
  pm_timeline?: string;
  pm_catalysts?: string[];
  pm_agents?: Record<string, { score: number }>;
}

// ---------------------------------------------------------------------------
// Classification Definitions Panel — explains each classification used in
// Pre-Momentum / Momentum / Excluded stages
// ---------------------------------------------------------------------------

interface ClsDef {
  emoji: string;
  name: string;
  ko: string;
  rule: string;
  implication: string;
  stage: "momentum" | "pre-momentum" | "excluded";
  color: string;
}

const CLASSIFICATION_DEFS: { groupTitle: string; emoji: string; defs: ClsDef[] }[] = [
  {
    groupTitle: "Bullish — Momentum 단계 (eligible=True)",
    emoji: "💚",
    defs: [
      {
        emoji: "🟢", name: "CONTINUATION", ko: "강한 상승 추세 지속",
        rule: "단기(SMA20) UP × 장기(SMA50/200) UP",
        implication: "추세 강화 구간. 매수 후 보유 유효.",
        stage: "momentum", color: "#0A7D3F",
      },
      {
        emoji: "🔵", name: "FORMATION", ko: "신규 추세 형성 초기",
        rule: "trend_age_short ≤ 10일 + 거래량 증가(1.3×) + 20일 돌파 + long_dir UP/FLAT",
        implication: "새 추세 시작 시그널. early-entry 후보.",
        stage: "momentum", color: "#3A7CA5",
      },
      {
        emoji: "🟦", name: "LAGGING_CATCHUP", ko: "AQR underreaction (catch-up)",
        rule: "URS ≥ 75 + 단기 미상승 + 장기 미하락 + base ∈ {CONS/NEUT/PULL}",
        implication: "카테고리 leader가 먼저 움직였고 본인은 아직 안 움직임. 따라잡기 후보.",
        stage: "momentum", color: "#0F5499",
      },
      {
        emoji: "🟡", name: "OVEREXTENDED", ko: "과열 (역추세 위험) — 추세는 유지",
        rule: "OER ≥ 60 + 강세 base ∈ {CONT/RECV/CONS/PULL/CNTR}",
        implication: "추세 자체는 건강하나 단기 과매수. eligible=True로 Momentum 탭에 표시되지만 진입 시 OER 주의 (HEDGE/TRIM 권장).",
        stage: "momentum", color: "#eab308",
      },
    ],
  },
  {
    groupTitle: "Recovering / Consolidating — 주로 Pre-Momentum 단계",
    emoji: "🟡",
    defs: [
      {
        emoji: "🔵", name: "RECOVERY", ko: "단기 회복, 장기는 횡보",
        rule: "단기 UP × 장기 FLAT",
        implication: "바닥 다지기 후 반등 초기. 추세 재형성 가능성.",
        stage: "pre-momentum", color: "#0F5499",
      },
      {
        emoji: "🟡", name: "CONSOLIDATION", ko: "단기 횡보, 장기 상승 추세 유지",
        rule: "단기 FLAT × 장기 UP",
        implication: "조정 / 매수 누적 단계. 다시 상승 재개 가능.",
        stage: "pre-momentum", color: "#B85C00",
      },
      {
        emoji: "🟠", name: "NEUTRAL", ko: "단기·장기 모두 횡보",
        rule: "단기 FLAT × 장기 FLAT",
        implication: "방향성 부재. 추세 형성 대기.",
        stage: "pre-momentum", color: "#C2701C",
      },
      {
        emoji: "🔶", name: "PULLBACK", ko: "장기 상승 중 단기 조정",
        rule: "단기 DOWN × 장기 UP",
        implication: "건강한 조정. 매수 기회 가능 (RSI 과매도 등 확인 필요).",
        stage: "pre-momentum", color: "#C2701C",
      },
    ],
  },
  {
    groupTitle: "Weakening — Pre-Momentum 후반 / Watch",
    emoji: "⚠️",
    defs: [
      {
        emoji: "⚠️", name: "WEAKENING", ko: "단기 하락, 장기 횡보",
        rule: "단기 DOWN × 장기 FLAT",
        implication: "추세 약화 시그널. 추가 하락 가능성 모니터.",
        stage: "pre-momentum", color: "#990F3D",
      },
      {
        emoji: "🟤", name: "FADING", ko: "단기 횡보, 장기 하락",
        rule: "단기 FLAT × 장기 DOWN",
        implication: "장기 추세 무너지는 중. 비중 축소 후보.",
        stage: "pre-momentum", color: "#8A6D3B",
      },
    ],
  },
  {
    groupTitle: "Bearish / Exhausted — Excluded 단계",
    emoji: "🔴",
    defs: [
      {
        emoji: "🟤", name: "EXHAUSTING", ko: "오랜 상승 후 둔화",
        rule: "trend_age > 60일 + ret_63d > 5% + ret_21d < 0 + 장기 UP",
        implication: "추세 피로 누적. 기존 보유는 익절 검토.",
        stage: "excluded", color: "#92400e",
      },
      {
        emoji: "🟣", name: "COUNTER_RALLY", ko: "장기 하락 중 단기 반등",
        rule: "단기 UP × 장기 DOWN",
        implication: "하락추세 내 단기 반등 (dead-cat bounce 가능). 신뢰도 낮음.",
        stage: "excluded", color: "#7D5BA6",
      },
      {
        emoji: "🔴", name: "CYCLE_PEAK", ko: "장기 사이클 정점",
        rule: "reversal_pctile ≥ 85 + ret_36_12m > 30% + ret_12_1m < 5% + 단기 DOWN",
        implication: "다년간 상승 후 소진 신호. 방어 전환 필요.",
        stage: "excluded", color: "#990F3D",
      },
      {
        emoji: "⬇️", name: "DOWNTREND", ko: "단기·장기 모두 하락",
        rule: "단기 DOWN × 장기 DOWN",
        implication: "본격 약세. 매수 회피 / 숏 후보.",
        stage: "excluded", color: "#CC0000",
      },
    ],
  },
];

function ClassificationCard({ def }: { def: ClsDef }) {
  const stageBadge =
    def.stage === "momentum" ? { label: "Momentum", cls: "bg-[#E3F0E8]/30 text-[#0A7D3F] border-[#A8CDB6]/40" }
    : def.stage === "pre-momentum" ? { label: "Pre-Momentum", cls: "bg-[#E3EEF5]/30 text-[#0D7680] border-[#9CC3D5]/40" }
    : { label: "Excluded", cls: "bg-[#F7E3E3]/30 text-[#CC0000] border-[#E0AAAA]/40" };
  return (
    <div className="bg-[#FBEEE3] border border-[#E6D9CE] rounded-lg p-2.5 hover:border-[#E6D9CE] transition-colors">
      <div className="flex items-baseline justify-between gap-2 mb-1">
        <div className="flex items-baseline gap-1.5 min-w-0">
          <span className="text-[18px] shrink-0">{def.emoji}</span>
          <span className="font-bold text-[14px] truncate" style={{ color: def.color }}>{def.name}</span>
        </div>
        <span className={`text-[8.5px] px-1.5 py-0.5 rounded border font-semibold shrink-0 ${stageBadge.cls}`}>
          {stageBadge.label}
        </span>
      </div>
      <div className="text-[14px] text-[#33302E] font-medium mb-1">{def.ko}</div>
      <div className="text-[12px] text-[#857F7A] mb-1.5">
        <span className="text-[#66605C] font-semibold">기준:</span> <span className="font-mono">{def.rule}</span>
      </div>
      <div className="text-[10.5px] text-[#66605C] leading-snug">
        <span className="text-[#33302E]">의미:</span> {def.implication}
      </div>
    </div>
  );
}

function ClassificationDefinitionsPanel() {
  const [open, setOpen] = useState(true);
  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-2.5 text-left text-[14px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[#66605C]">
          Classification Definitions — Pre-Momentum / Momentum 분류 정의
          <span className="ml-2 text-[12px] text-[#857F7A]">14 classifications · 4 그룹</span>
        </span>
        <span className="text-[#857F7A] text-[12px]">{open ? "\u25bc" : "\u25b6"}</span>
      </button>
      {open && (
        <div className="bg-[#FBEEE3] p-4 space-y-4">
          <div className="text-[13px] text-[#857F7A] leading-relaxed">
            각 ticker는 매일 단기(SMA20) × 장기(SMA50/200) 트렌드 매트릭스 + override 규칙으로
            <span className="text-[#33302E]"> 14개 classification</span> 중 하나로 분류됩니다.
            여기에 composite ≥ 55 + ADV ≥ $5M 조건을 적용해
            <span className="text-[#0D7680]"> Momentum</span> /
            <span className="text-[#0D7680]"> Pre-Momentum</span> /
            <span className="text-[#CC0000]"> Excluded</span> 단계가 결정됩니다.
          </div>
          {CLASSIFICATION_DEFS.map((g) => (
            <div key={g.groupTitle}>
              <div className="text-[14px] font-semibold text-[#33302E] mb-2 flex items-baseline gap-2">
                <span className="text-[18px]">{g.emoji}</span>
                <span>{g.groupTitle}</span>
                <span className="text-[12px] text-[#857F7A]">({g.defs.length}개)</span>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-2">
                {g.defs.map((d) => <ClassificationCard key={d.name} def={d} />)}
              </div>
            </div>
          ))}
          <div className="bg-[#FFFFFF] border border-[#9CC3D5]/40 rounded p-2.5 text-[10.5px] text-[#0D7680]/80 leading-relaxed">
            <b className="text-[#0D7680]">📌 참고</b> —&nbsp;
            <span className="text-[#33302E]">
              "기준" 열은 분류 규칙(decision rule)이고, "단계" 뱃지는 <b>주로</b> 어느 stage에 속하는지를 나타냅니다.
              실제 stage는 classification + composite/ADV/eligibility 종합 판단으로 결정되므로,
              예컨대 NEUTRAL이라도 composite ≥ 55면 Momentum이 될 수 있습니다.
              자세한 규칙은 <code>price_discovery.py</code> 참조.
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Classification Age Panel — explains how age is computed for each
// classification (pm_age, momentum age, trend_age, days_in_stage)
// ---------------------------------------------------------------------------

function ClassificationAgePanel() {
  const [open, setOpen] = useState(true);
  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-2.5 text-left text-[14px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[#66605C]">
          Classification Age — 정의 및 산출 방식
          <span className="ml-2 text-[12px] text-[#857F7A]">pm_age · momentum_age · trend_age · days_in_stage</span>
        </span>
        <span className="text-[#857F7A] text-[12px]">{open ? "\u25bc" : "\u25b6"}</span>
      </button>
      {open && (
        <div className="bg-[#FBEEE3] p-4 space-y-4">
          <div className="text-[13px] text-[#857F7A] leading-relaxed">
            "Age"는 ticker가 현재 상태(classification 또는 stage)에 얼마나 오래 머물렀는지를 측정합니다.
            상태 종류에 따라 <span className="text-[#33302E]">4가지 age 개념</span>이 사용됩니다.
            가장 핵심은 <span className="text-[#0D7680]">pm_age</span> (Pre-Momentum 지속일)와
            <span className="text-[#0A7D3F]"> momentum_age</span> (Momentum 지속일)입니다.
          </div>

          {/* Two-column main: pm_age + momentum_age */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">

            {/* ── Pre-Momentum Age ── */}
            <div className="bg-[#FFFFFF] border-2 border-[#9CC3D5]/40 rounded-lg p-3">
              <div className="flex items-baseline justify-between mb-2">
                <div className="flex items-baseline gap-2">
                  <span className="text-[#0D7680] text-[18px] font-bold">🔵 pm_age</span>
                  <span className="text-[13px] text-[#66605C]">— Pre-Momentum 지속일</span>
                </div>
                <span className="text-[9.5px] px-1.5 py-0.5 rounded bg-[#E3EEF5]/30 text-[#0D7680] border border-[#9CC3D5]/40 font-semibold">
                  Pre-Momentum stage
                </span>
              </div>

              <div className="text-[13px] text-[#33302E] mb-2 leading-relaxed">
                <b className="text-[#33302E]">언제 적용:</b> Pre-Momentum 분류군에 속한 ticker (NEUTRAL · CONSOLIDATION · RECOVERY · PULLBACK · WEAKENING · FADING).
              </div>

              <div className="bg-[#FBEEE3] rounded p-2 mb-2 text-[10.5px] text-[#66605C]">
                <div className="text-[#0D7680] text-[12px] font-semibold mb-1">📐 산출 알고리즘</div>
                <ol className="list-decimal pl-5 space-y-0.5">
                  <li>2주 간격(bi-weekly) 과거 observations를 새것부터 거꾸로 walk</li>
                  <li>분류가 PM 분류군이면 → 카운트 추가</li>
                  <li>분류가 momentum-confirmed (CONTINUATION/FORMATION 등)면 → <span className="text-[#B85C00]">중단 (breakout = age reset)</span></li>
                  <li>그 외 (bearish 등)는 <b>gap</b>으로 처리 — 최대 1개까지 허용 (노이즈)</li>
                  <li>최소 2회 연속 PM 관찰돼야 age 부여 (단발 노이즈 필터)</li>
                  <li>age = (today − 가장 오래된 PM 관찰일).days</li>
                  <li>최대 90일로 cap</li>
                </ol>
              </div>

              <div className="bg-[#FBEEE3] rounded p-2 mb-2 text-[10.5px] text-[#66605C]">
                <div className="text-[#0D7680] text-[12px] font-semibold mb-1">🎯 의미</div>
                <ul className="list-disc pl-5 space-y-0.5">
                  <li><span className="text-[#0A7D3F]">짧음 (≤ 14일)</span>: 신선한 setup — 진입 적기 가능</li>
                  <li><span className="text-[#B85C00]">중간 (15~45일)</span>: 형성 중 — 모니터링</li>
                  <li><span className="text-[#CC0000]">길음 (&gt; 45일)</span>: 오래 정체 — pre-momentum 신뢰도 ↓</li>
                </ul>
              </div>

              <div className="text-[9.5px] text-[#857F7A] italic">
                구현: <code className="text-[#33302E]">_enrich_pm_age_robust()</code> in pre_momentum.py · 데이터: <code className="text-[#33302E]">.pm_history.json</code>
              </div>
            </div>

            {/* ── Momentum Age ── */}
            <div className="bg-[#FFFFFF] border-2 border-[#A8CDB6]/50 rounded-lg p-3">
              <div className="flex items-baseline justify-between mb-2">
                <div className="flex items-baseline gap-2">
                  <span className="text-[#0A7D3F] text-[18px] font-bold">🟢 momentum_age</span>
                  <span className="text-[13px] text-[#66605C]">— Momentum 지속일 (3-tier)</span>
                </div>
                <span className="text-[9.5px] px-1.5 py-0.5 rounded bg-[#E3F0E8]/30 text-[#0A7D3F] border border-[#A8CDB6]/40 font-semibold">
                  Momentum stage
                </span>
              </div>

              <div className="text-[13px] text-[#33302E] mb-2 leading-relaxed">
                <b className="text-[#33302E]">언제 적용:</b> Momentum 적격 ticker (composite ≥ 55 + ADV ≥ $5M + non-bearish).
              </div>

              <div className="bg-[#FBEEE3] rounded p-2 mb-2">
                <div className="text-[#0A7D3F] text-[12px] font-semibold mb-1">📊 3-Tier Classification</div>
                <table className="w-full text-[10.5px]">
                  <tbody>
                    <tr className="border-b border-[#E6D9CE]/50">
                      <td className="py-1 px-1 text-[#0A7D3F] font-semibold">Tier A (확정)</td>
                      <td className="py-1 px-1 text-[#33302E] font-mono text-[12px]">CONTINUATION · FORMATION · OVEREXTENDED</td>
                      <td className="py-1 px-1 text-right text-[#0A7D3F] text-[12px]">→ 카운트</td>
                    </tr>
                    <tr className="border-b border-[#E6D9CE]/50">
                      <td className="py-1 px-1 text-[#B85C00] font-semibold">Tier B (조정)</td>
                      <td className="py-1 px-1 text-[#33302E] font-mono text-[12px]">RECOVERY · CONSOLIDATION · PULLBACK</td>
                      <td className="py-1 px-1 text-right text-[#B85C00] text-[12px]">→ gap (최대 2)</td>
                    </tr>
                    <tr>
                      <td className="py-1 px-1 text-[#CC0000] font-semibold">Tier C (위험)</td>
                      <td className="py-1 px-1 text-[#33302E] font-mono text-[12px]">DOWNTREND · CYCLE_PEAK · COUNTER_RALLY · EXHAUSTING · FADING · WEAKENING</td>
                      <td className="py-1 px-1 text-right text-[#CC0000] text-[12px]">→ gap (최대 1)</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="bg-[#FBEEE3] rounded p-2 mb-2 text-[10.5px] text-[#66605C]">
                <div className="text-[#0A7D3F] text-[12px] font-semibold mb-1">📐 산출 알고리즘</div>
                <ol className="list-decimal pl-5 space-y-0.5">
                  <li>새것부터 거꾸로 ve_observations walk</li>
                  <li>Tier A → 카운트 / Tier B → gap (최대 2 누적) / Tier C → gap (최대 1)</li>
                  <li>총 gap 누적 3 초과 시 즉시 중단</li>
                  <li><span className="text-[#CC0000]">Tier C 2연속</span> → hard break (실제 추세 전환)</li>
                  <li>최소 2회 확정 관찰돼야 age 부여</li>
                  <li>age = (today − 가장 오래된 확정 관찰일).days</li>
                  <li>최대 180일로 cap</li>
                </ol>
              </div>

              <div className="bg-[#FBEEE3] rounded p-2 mb-2 text-[10.5px] text-[#66605C]">
                <div className="text-[#0A7D3F] text-[12px] font-semibold mb-1">🎯 의미</div>
                <ul className="list-disc pl-5 space-y-0.5">
                  <li><span className="text-[#0A7D3F]">짧음 (≤ 21일)</span>: 신규 진입 — 추세 신선</li>
                  <li><span className="text-[#0D7680]">중간 (22~90일)</span>: 활성 추세 — 보유 유지</li>
                  <li><span className="text-[#B85C00]">길음 (&gt; 90일)</span>: 오래된 추세 — exhaust 위험 모니터링</li>
                </ul>
              </div>

              <div className="text-[9.5px] text-[#857F7A] italic">
                구현: <code className="text-[#33302E]">compute_momentum_ages()</code> in pre_momentum.py
              </div>
            </div>
          </div>

          {/* Visual: gap tolerance example */}
          <div className="bg-[#FFFFFF] rounded-lg p-3">
            <div className="text-[14px] font-semibold text-[#33302E] mb-2">📈 Gap Tolerance — 예시 (momentum_age)</div>
            <div className="text-[10.5px] text-[#66605C] mb-2">
              매일 분류가 같지 않더라도 단기 노이즈는 흡수합니다. 예시:
            </div>
            <div className="overflow-x-auto">
              <table className="text-[10.5px] font-mono">
                <thead>
                  <tr className="border-b border-[#E6D9CE]">
                    <th className="px-2 py-1 text-left text-[#857F7A]">Day</th>
                    {Array.from({length: 14}, (_, i) => (
                      <th key={i} className="px-1.5 py-1 text-center text-[#857F7A] w-[44px]">D-{13-i}</th>
                    ))}
                    <th className="px-2 py-1 text-left text-[#33302E]">결과</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-[#E6D9CE]/50">
                    <td className="px-2 py-1 text-[#66605C]">분류</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#B85C00]">B</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#B85C00]">B</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#CC0000]">C</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-2 py-1 text-[#0A7D3F]">age = 14일 ✓</td>
                  </tr>
                  <tr>
                    <td className="px-2 py-1 text-[#66605C]">분류</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#CC0000]">C</td>
                    <td className="px-1.5 py-1 text-center text-[#CC0000]">C</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-1.5 py-1 text-center text-[#0A7D3F]">A</td>
                    <td className="px-2 py-1 text-[#CC0000]">break! Tier C 2연속 → age = 8일</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="text-[12px] text-[#857F7A] mt-2 leading-relaxed">
              위: A 분류 사이에 B(조정) 1회와 C(위험) 1회 섞여도 gap 한도 내라 age = 14일.&nbsp;
              아래: Tier C가 2연속 발생하면 즉시 hard break → 카운트는 가장 최근 break 이후로 리셋됨.
            </div>
          </div>

          {/* Supporting age concepts */}
          <div className="bg-[#FFFFFF] rounded-lg p-3">
            <div className="text-[14px] font-semibold text-[#33302E] mb-2">📎 보조 Age 개념 (참고)</div>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-2.5">
              <div className="bg-[#FBEEE3] border border-[#E6D9CE] rounded p-2.5">
                <div className="font-mono text-[14px] font-bold text-[#C2701C] mb-1">trend_age</div>
                <div className="text-[10.5px] text-[#33302E] mb-1">close &gt; SMA50인 연속 일수</div>
                <div className="text-[12px] text-[#857F7A] mb-1">
                  산출: 가장 최근일부터 거꾸로 카운트, close가 SMA50 하회하는 순간 reset (0).
                </div>
                <div className="text-[12px] text-[#66605C]">
                  → classification 규칙에 직접 사용 (예: <code>EXHAUSTING은 trend_age &gt; 60</code>).
                </div>
              </div>

              <div className="bg-[#FBEEE3] border border-[#E6D9CE] rounded p-2.5">
                <div className="font-mono text-[14px] font-bold text-[#B85C00] mb-1">trend_age_short</div>
                <div className="text-[10.5px] text-[#33302E] mb-1">close &gt; SMA20인 연속 일수</div>
                <div className="text-[12px] text-[#857F7A] mb-1">
                  산출: trend_age와 동일하나 SMA20 기준. 단기 트렌드 신선도 측정.
                </div>
                <div className="text-[12px] text-[#66605C]">
                  → 예: <code>FORMATION은 trend_age_short ≤ 10</code> (신규 추세 한정).
                </div>
              </div>

              <div className="bg-[#FBEEE3] border border-[#E6D9CE] rounded p-2.5">
                <div className="font-mono text-[14px] font-bold text-[#7D5BA6] mb-1">days_in_stage</div>
                <div className="text-[10.5px] text-[#33302E] mb-1">동일 stage 유지 일수 (lifecycle)</div>
                <div className="text-[12px] text-[#857F7A] mb-1">
                  산출: <code>.pipeline_history.json</code> 기록을 새것부터 거꾸로 walk, 동일 stage(Momentum/Pre-Momentum/Excluded)면 카운트.
                </div>
                <div className="text-[12px] text-[#66605C]">
                  → stage 안정성 / 전환 임박도 판단에 사용 (Pipeline 탭).
                </div>
              </div>
            </div>
          </div>

          {/* Summary */}
          <div className="bg-[#FFF1E5] border border-[#9CC3D5]/40 rounded-lg p-3 text-[13px] text-[#0D7680]/90 leading-relaxed">
            <b className="text-[#0D7680]">📌 정리</b> —&nbsp;
            <span className="text-[#33302E]">
              Pre-Momentum과 Momentum 분류는 각각 <b>pm_age</b>, <b>momentum_age</b>로 지속일 추적.
              둘 다 <b>2주 간격 ve_observations</b> 기반으로 계산하며, 단순 streak이 아닌
              <b> Tier 분류 + gap 허용</b> 방식이라 단기 노이즈에 강건.
              세부 분류 규칙(예: FORMATION trend_age_short ≤ 10) 자체에는 daily-resolution
              <b> trend_age</b> / <b>trend_age_short</b>가 직접 사용되며, 이는 close vs SMA의 일별 streak.
              상위 lifecycle stage에는 <b>days_in_stage</b>가 별도로 추적됨 (Pipeline 탭).
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Classification History Heatmap
// ---------------------------------------------------------------------------

interface ClassHistoryData {
  dates: string[];
  classifications: string[];
  matrix: number[][];
}

function ClassificationHistoryChart({ data }: { data: ClassHistoryData | null }) {
  const [open, setOpen] = useState(true);

  if (!data || !data.dates.length) return null;

  // Heatmap: rows = classifications (reversed for top-to-bottom), cols = dates
  const clsLabels = [...data.classifications].reverse();
  const zData = [...data.matrix].reverse();

  // Annotations: show ALL non-zero counts. Dense histories use a smaller font
  // so 2-3 digit numbers fit even in narrow cells.
  const dense = data.dates.length >= 12;
  const veryDense = data.dates.length >= 18;
  const annFontSize = veryDense ? 8 : dense ? 9 : 11;
  const annotations: any[] = [];
  for (let i = 0; i < clsLabels.length; i++) {
    for (let j = 0; j < data.dates.length; j++) {
      const val = zData[i][j];
      if (val <= 0) continue;   // only skip true zeros (empty cells)
      annotations.push({
        x: data.dates[j],
        y: clsLabels[i],
        text: String(val),
        showarrow: false,
        font: { color: val > 80 ? "#000" : "#33302E", size: annFontSize, family: "monospace" },
      });
    }
  }

  // Compute delta (last vs second-to-last) for the side column
  const deltas = data.classifications.map((_, i) => {
    const row = data.matrix[i];
    if (row.length < 2) return 0;
    return row[row.length - 1] - row[row.length - 2];
  }).reverse();

  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-2.5 text-left text-[14px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[#66605C]">Classification Distribution — Bi-weekly Change ({data.dates.length} periods)</span>
        <span className="text-[#857F7A] text-[12px]">{open ? "\u25bc" : "\u25b6"}</span>
      </button>
      {open && (
        <div className="bg-[#FBEEE3] p-4 space-y-4">

          {(() => {
            // ── Latest distribution panel — prominent, separated from history ──
            // ORDER matches heatmap top→bottom (data.classifications original order:
            // sorted by latest count desc on backend, displayed top→bottom in heatmap).
            const latestDate = data.dates[data.dates.length - 1] ?? "";
            const prevDate = data.dates[data.dates.length - 2] ?? "";
            const latestOrdered = data.classifications.map((cls, i) => {
              const row = data.matrix[i];
              const count = row[row.length - 1] ?? 0;
              const prev = row.length >= 2 ? row[row.length - 2] : 0;
              return { cls, count, delta: count - prev };
            });
            const latestTotal = latestOrdered.reduce((a, e) => a + e.count, 0);
            const latestMax = Math.max(1, ...latestOrdered.map((e) => e.count));
            return (
              <div className="bg-[#FFFFFF] rounded-lg border border-[#9CC3D5]/40 p-4">
                <div className="flex items-baseline justify-between mb-3">
                  <div>
                    <h3 className="text-[16px] font-semibold text-[#0F5499]">Latest Distribution</h3>
                    <div className="text-[12px] text-[#857F7A] mt-0.5">
                      {latestDate}
                      {prevDate && <span className="ml-2 text-[#857F7A]">· delta vs {prevDate}</span>}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[12px] text-[#857F7A] uppercase tracking-wide">Total</div>
                    <div className="text-[20px] font-mono font-bold text-[#0D7680] tabular-nums">{latestTotal}</div>
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-1.5">
                  {latestOrdered.map((e) => {
                    const color = (CLASS_COLORS as Record<string, string>)[e.cls] || C.gray;
                    const widthPct = (e.count / latestMax) * 100;
                    const sharePct = latestTotal > 0 ? (e.count / latestTotal) * 100 : 0;
                    const dColor = e.delta > 5 ? C.green
                                  : e.delta > 0 ? "#0A7D3F"
                                  : e.delta < -5 ? C.red
                                  : e.delta < 0 ? "#CC0000"
                                  : C.gray;
                    return (
                      <div key={e.cls} className="flex items-center gap-2 text-[13px]">
                        <span className="truncate min-w-[140px] max-w-[140px] text-[#33302E]" title={e.cls}>
                          {e.cls}
                        </span>
                        <div className="flex-1 relative h-4 bg-[#FBEEE3] rounded overflow-hidden">
                          <div
                            className="absolute left-0 top-0 h-full rounded"
                            style={{ width: `${widthPct}%`, backgroundColor: color, opacity: 0.5 }}
                          />
                          <div className="absolute inset-0 flex items-center justify-end pr-1.5 text-[12px] font-mono font-bold tabular-nums text-[#33302E]">
                            {e.count}
                          </div>
                        </div>
                        <span className="text-[12px] text-[#857F7A] font-mono w-10 text-right tabular-nums">
                          {sharePct.toFixed(1)}%
                        </span>
                        <span className="text-[12px] font-mono font-bold w-10 text-right tabular-nums" style={{ color: dColor }}>
                          {e.delta > 0 ? "+" : ""}{e.delta}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className="text-[12px] text-[#857F7A] mt-3 leading-relaxed">
                  Bar width = count vs max in current snapshot. Right columns: % share of total · delta vs prior bi-weekly snapshot ({prevDate || "n/a"}).
                </div>
              </div>
            );
          })()}

          {/* ── History (heatmap + delta column) ── */}
          <div>
            <div className="text-[13px] text-[#857F7A] uppercase tracking-wide mb-2">
              History — bi-weekly trend
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-[1fr_140px] gap-4">
            {/* Heatmap */}
            <Plot
              data={[
                {
                  type: "heatmap",
                  z: zData,
                  x: data.dates,
                  y: clsLabels,
                  colorscale: [
                    [0, "#FBEEE3"],
                    [0.05, "#1e3a5f"],
                    [0.15, "#1d4ed8"],
                    [0.3, "#0D7680"],
                    [0.5, "#0A7D3F"],
                    [0.75, "#B85C00"],
                    [1, "#CC0000"],
                  ],
                  showscale: true,
                  colorbar: { tickfont: { color: "#66605C", size: 9 }, len: 0.8, thickness: 12 },
                  hovertemplate: "%{y}<br>%{x}: %{z} tickers<extra></extra>",
                },
              ]}
              layout={{
                ...DARK_LAYOUT,
                height: Math.max(300, clsLabels.length * 32 + 80),
                margin: { t: 10, b: 80, l: 160, r: 60 },
                xaxis: {
                  type: "category" as const,   // 균일한 cell width — 날짜 간격 무관
                  tickangle: -45,
                  tickfont: { size: dense ? 8 : 10, color: "#66605C" },
                  side: "bottom",
                  // Categorical 모드에서 dtick=N → N번째마다 라벨 표시 (밀집도 완화)
                  ...(veryDense ? { dtick: 2 } : dense ? { dtick: 1 } : {}),
                },
                yaxis: { tickfont: { size: 10, color: "#66605C" }, automargin: true },
                annotations,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />

            {/* Delta column — change in ticker count for each classification, latest vs prior period.
                ORDER aligned with heatmap top→bottom (data.classifications original order, NOT reversed) */}
            <div className="space-y-0 pt-2">
              <div className="text-[11px] text-[#857F7A] uppercase tracking-wide mb-0.5 text-center"
                   title="Change in ticker count: latest snapshot minus prior bi-weekly snapshot">
                Latest Change
              </div>
              <div className="text-[10px] text-[#857F7A] text-center mb-1.5 leading-tight px-1">
                Δ tickers vs prior<br/>bi-weekly snapshot
              </div>
              {data.classifications.map((cls, i) => {
                const row = data.matrix[i];
                const d = row.length >= 2 ? row[row.length - 1] - row[row.length - 2] : 0;
                const color = d > 10 ? C.green : d > 0 ? "#0A7D3F" : d < -10 ? C.red : d < 0 ? "#CC0000" : C.gray;
                return (
                  <div key={cls} className="flex items-center justify-between px-2 py-[3px] text-[12px]"
                       style={{ height: "32px" }}
                       title={`${cls}: ${d > 0 ? "+" : ""}${d} tickers vs prior period`}>
                    <span className="text-[#857F7A] truncate max-w-[80px]">{cls.replace(/^.\s/, "")}</span>
                    <span className="font-mono font-bold" style={{ color }}>
                      {d > 0 ? "+" : ""}{d}
                    </span>
                  </div>
                );
              })}
              <div className="text-[10px] text-[#857F7A] mt-2 px-1 leading-relaxed border-t border-[#E6D9CE] pt-2">
                <span style={{ color: C.green }}>+10↑</span> · <span style={{ color: "#0A7D3F" }}>+1~10</span> ·
                <span className="text-[#857F7A]"> 0 </span>·
                <span style={{ color: "#CC0000" }}>−1~−10</span> · <span style={{ color: C.red }}>−10↓</span>
              </div>
            </div>
          </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sector Classification Grid
// ---------------------------------------------------------------------------

interface SectorEntry {
  sector: string;
  total: number;
  dates: string[];
  classifications: string[];
  matrix: number[][];
}

const CLS_SHORT: Record<string, string> = {
  "\ud83d\udfe2 CONTINUATION": "CONT",
  "\ud83d\udd35 FORMATION": "FORM",
  "\ud83d\udd35 RECOVERY": "RECV",
  "\ud83d\udfe1 CONSOLIDATION": "CONS",
  "\ud83d\udfe0 NEUTRAL": "NEUT",
  "\ud83d\udfe4 FADING": "FADE",
  "\ud83d\udd36 PULLBACK": "PULL",
  "\u26a0\ufe0f WEAKENING": "WEAK",
  "\u2b07\ufe0f DOWNTREND": "DOWN",
  "\ud83d\udfe1 OVEREXTENDED": "OEXT",
  "\ud83d\udfe4 EXHAUSTING": "EXHT",
  "\ud83d\udfe3 COUNTER_RALLY": "CNTR",
  "\ud83d\udd34 CYCLE_PEAK": "PEAK",
};

const CLS_COLOR: Record<string, string> = {
  "\ud83d\udfe2 CONTINUATION": "#0A7D3F",
  "\ud83d\udd35 FORMATION": "#3A7CA5",
  "\ud83d\udd35 RECOVERY": "#0F5499",
  "\ud83d\udfe1 CONSOLIDATION": "#B85C00",
  "\ud83d\udfe0 NEUTRAL": "#C2701C",
  "\ud83d\udfe4 FADING": "#8A6D3B",
  "\ud83d\udd36 PULLBACK": "#C2701C",
  "\u26a0\ufe0f WEAKENING": "#990F3D",
  "\u2b07\ufe0f DOWNTREND": "#CC0000",
  "\ud83d\udfe1 OVEREXTENDED": "#eab308",
  "\ud83d\udfe4 EXHAUSTING": "#92400e",
  "\ud83d\udfe3 COUNTER_RALLY": "#7D5BA6",
  "\ud83d\udd34 CYCLE_PEAK": "#990F3D",
};

function SectorMiniTable({ entry }: { entry: SectorEntry }) {
  const dates = entry.dates;
  const lastIdx = dates.length - 1;
  const prevIdx = dates.length - 2;

  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[14px] font-semibold text-[#33302E]">{entry.sector.replace("STK_", "").replace("EQ_", "")}</span>
        <span className="text-[12px] text-[#857F7A]">{entry.total} tickers</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr className="border-b border-[#E6D9CE]">
              <th className="py-1 px-1 text-left text-[#857F7A] w-12">Class</th>
              {dates.map((d) => (
                <th key={d} className="py-1 px-1 text-center text-[#857F7A] w-8">{d.slice(5)}</th>
              ))}
              <th className="py-1 px-1 text-center text-[#857F7A] w-8">\u0394</th>
            </tr>
          </thead>
          <tbody>
            {entry.classifications.map((cls, ci) => {
              const row = entry.matrix[ci];
              const curr = lastIdx >= 0 ? row[lastIdx] : 0;
              const prev = prevIdx >= 0 ? row[prevIdx] : 0;
              const delta = curr - prev;
              const clsColor = CLS_COLOR[cls] || C.gray;
              return (
                <tr key={cls} className="border-b border-[#E6D9CE]/30">
                  <td className="py-0.5 px-1 font-semibold" style={{ color: clsColor }}>
                    {CLS_SHORT[cls] || cls.replace(/^.\s/, "").slice(0, 4)}
                  </td>
                  {row.map((v, di) => (
                    <td key={di} className="py-0.5 px-1 text-center font-mono"
                        style={{ color: v > 0 ? "#33302E" : "#CCC1B7" }}>
                      {v > 0 ? v : "\u00b7"}
                    </td>
                  ))}
                  <td className="py-0.5 px-1 text-center font-mono font-bold"
                      style={{ color: delta > 0 ? "#0A7D3F" : delta < 0 ? "#CC0000" : "#857F7A" }}>
                    {delta !== 0 ? (delta > 0 ? "+" + delta : delta) : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// \u2500\u2500\u2500 Return by Sector \u2014 grouped bar chart (1M/3M/YTD per sector) \u2500\u2500\u2500
function ReturnBySectorPanel({ allResults }: { allResults: any[] }) {
  const data = useMemo(() => {
    const agg: Record<string, { n: number; s1m: number; s3m: number; sytd: number; nytd: number }> = {};
    for (const r of allResults || []) {
      const sec = r.category;
      if (!sec) continue;
      const m1 = Number(r.ret_21d ?? NaN);
      const m3 = Number(r.ret_63d ?? NaN);
      const ytd = Number(r.ret_ytd ?? NaN);
      const cd = (agg[sec] = agg[sec] || { n: 0, s1m: 0, s3m: 0, sytd: 0, nytd: 0 });
      cd.n += 1;
      if (!isNaN(m1)) cd.s1m += m1;
      if (!isNaN(m3)) cd.s3m += m3;
      if (!isNaN(ytd)) { cd.sytd += ytd; cd.nytd += 1; }
    }
    const rows = Object.entries(agg).map(([sec, cd]) => ({
      sec, n: cd.n,
      m1: cd.n > 0 ? cd.s1m / cd.n : 0,
      m3: cd.n > 0 ? cd.s3m / cd.n : 0,
      ytd: cd.nytd > 0 ? cd.sytd / cd.nytd : 0,
    }));
    rows.sort((a, b) => b.ytd - a.ytd);
    return rows;
  }, [allResults]);

  if (!data.length) return null;
  const sectors = data.map((d) => d.sec);
  const labels = data.map((d) => `${d.sec} (n=${d.n})`);

  return (
    <div className="bg-[#FFFFFF] rounded-lg border border-[#9CC3D5]/40 p-4 mb-4">
      <div className="mb-3">
        <h3 className="text-[16px] font-semibold text-[#0F5499]">Return by Sector (latest snapshot)</h3>
        <div className="text-[12px] text-[#857F7A] mt-0.5">
          Average ticker return % per sector \u00b7 3 horizons (1M / 3M / YTD) \u00b7 sorted by YTD descending
        </div>
      </div>
      <Plot
        data={[
          { x: sectors, y: data.map((d) => d.m1),  type: "bar", name: "1M",  marker: { color: "#3A7CA5" }, customdata: labels, hovertemplate: "%{customdata}<br>1M: %{y:.2f}%<extra></extra>" },
          { x: sectors, y: data.map((d) => d.m3),  type: "bar", name: "3M",  marker: { color: "#0A7D3F" }, customdata: labels, hovertemplate: "%{customdata}<br>3M: %{y:.2f}%<extra></extra>" },
          { x: sectors, y: data.map((d) => d.ytd), type: "bar", name: "YTD", marker: { color: "#B85C00" }, customdata: labels, hovertemplate: "%{customdata}<br>YTD: %{y:.2f}%<extra></extra>" },
        ]}
        layout={{
          ...DARK_LAYOUT,
          height: 360,
          margin: { t: 30, b: 110, l: 50, r: 30 },
          barmode: "group",
          xaxis: { tickangle: -35, tickfont: { size: 10, color: "#66605C" } },
          yaxis: { gridcolor: "#F2E5D7", color: "#66605C", title: { text: "Return %", font: { size: 10, color: "#66605C" } }, zeroline: true, zerolinecolor: "#CCC1B7" },
          legend: { orientation: "h", y: 1.12, font: { size: 10, color: "#33302E" } },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}


// \u2500\u2500\u2500 Return by Classification \u2014 grouped bar chart (1M/3M/YTD per classification) \u2500\u2500\u2500
function ReturnByClassificationPanel({ allResults }: { allResults: any[] }) {
  const STD_ORDER = [
    "\ud83d\udfe2 CONTINUATION", "\ud83d\udfe1 OVEREXTENDED", "\ud83d\udd35 RECOVERY", "\ud83d\udd35 FORMATION",
    "\ud83d\udfe6 LAGGING_CATCHUP", "\ud83d\udfe1 CONSOLIDATION", "\ud83d\udfe0 NEUTRAL",
    "\ud83d\udd36 PULLBACK", "\ud83d\udfe4 FADING", "\u26a0\ufe0f WEAKENING", "\u2b07\ufe0f DOWNTREND",
    "\ud83d\udfe4 EXHAUSTING", "\ud83d\udfe3 COUNTER_RALLY", "\ud83d\udd34 CYCLE_PEAK",
  ];

  const data = useMemo(() => {
    const agg: Record<string, { n: number; s1m: number; s3m: number; sytd: number; nytd: number }> = {};
    for (const r of allResults || []) {
      const cls = r.classification;
      if (!cls) continue;
      const m1 = Number(r.ret_21d ?? NaN);
      const m3 = Number(r.ret_63d ?? NaN);
      const ytd = Number(r.ret_ytd ?? NaN);
      const cd = (agg[cls] = agg[cls] || { n: 0, s1m: 0, s3m: 0, sytd: 0, nytd: 0 });
      cd.n += 1;
      if (!isNaN(m1)) cd.s1m += m1;
      if (!isNaN(m3)) cd.s3m += m3;
      if (!isNaN(ytd)) { cd.sytd += ytd; cd.nytd += 1; }
    }
    return STD_ORDER.filter((cls) => agg[cls]).map((cls) => {
      const cd = agg[cls];
      return {
        cls, n: cd.n,
        m1: cd.n > 0 ? cd.s1m / cd.n : 0,
        m3: cd.n > 0 ? cd.s3m / cd.n : 0,
        ytd: cd.nytd > 0 ? cd.sytd / cd.nytd : 0,
      };
    });
  }, [allResults]);

  if (!data.length) return null;
  const classes = data.map((d) => d.cls);
  const labels = data.map((d) => `${d.cls} (n=${d.n})`);

  return (
    <div className="bg-[#FFFFFF] rounded-lg border border-[#9CC3D5]/40 p-4 mb-4">
      <div className="mb-3">
        <h3 className="text-[16px] font-semibold text-[#0F5499]">Return by Classification (latest snapshot)</h3>
        <div className="text-[12px] text-[#857F7A] mt-0.5">
          Average ticker return % per classification \u00b7 3 horizons (1M / 3M / YTD) \u00b7 ordered bullish \u2192 bearish
        </div>
      </div>
      <Plot
        data={[
          { x: classes, y: data.map((d) => d.m1),  type: "bar", name: "1M",  marker: { color: "#3A7CA5" }, customdata: labels, hovertemplate: "%{customdata}<br>1M: %{y:.2f}%<extra></extra>" },
          { x: classes, y: data.map((d) => d.m3),  type: "bar", name: "3M",  marker: { color: "#0A7D3F" }, customdata: labels, hovertemplate: "%{customdata}<br>3M: %{y:.2f}%<extra></extra>" },
          { x: classes, y: data.map((d) => d.ytd), type: "bar", name: "YTD", marker: { color: "#B85C00" }, customdata: labels, hovertemplate: "%{customdata}<br>YTD: %{y:.2f}%<extra></extra>" },
        ]}
        layout={{
          ...DARK_LAYOUT,
          height: 360,
          margin: { t: 30, b: 130, l: 50, r: 30 },
          barmode: "group",
          xaxis: { tickangle: -35, tickfont: { size: 10, color: "#66605C" } },
          yaxis: { gridcolor: "#F2E5D7", color: "#66605C", title: { text: "Return %", font: { size: 10, color: "#66605C" } }, zeroline: true, zerolinecolor: "#CCC1B7" },
          legend: { orientation: "h", y: 1.12, font: { size: 10, color: "#33302E" } },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}


// (DEPRECATED \u2014 replaced by ReturnBySectorPanel + ReturnByClassificationPanel)
function ClassificationSectorReturnsPanel({ allResults }: { allResults: any[] }) {
  // Group: { sector: { classification: { count, sum_1m, sum_3m, sum_ytd } } }
  const grouped = useMemo(() => {
    const out: Record<string, Record<string, { n: number; s1m: number; s3m: number; sytd: number; nytd: number }>> = {};
    for (const r of allResults || []) {
      const sec = r.category;
      const cls = r.classification;
      if (!sec || !cls) continue;
      const m1 = Number(r.ret_21d ?? NaN);
      const m3 = Number(r.ret_63d ?? NaN);
      const ytd = Number(r.ret_ytd ?? NaN);
      const sd = (out[sec] = out[sec] || {});
      const cd = (sd[cls] = sd[cls] || { n: 0, s1m: 0, s3m: 0, sytd: 0, nytd: 0 });
      cd.n += 1;
      if (!isNaN(m1)) cd.s1m += m1;
      if (!isNaN(m3)) cd.s3m += m3;
      if (!isNaN(ytd)) { cd.sytd += ytd; cd.nytd += 1; }
    }
    return out;
  }, [allResults]);

  // Sectors sorted by total ticker count desc
  const sectors = useMemo(() =>
    Object.keys(grouped).sort((a, b) => {
      const ta = Object.values(grouped[a]).reduce((s, v) => s + v.n, 0);
      const tb = Object.values(grouped[b]).reduce((s, v) => s + v.n, 0);
      return tb - ta;
    }), [grouped]);

  // Standard classification order (matches heatmap ordering)
  const STD_ORDER = [
    "\ud83d\udfe2 CONTINUATION", "\ud83d\udfe1 OVEREXTENDED", "\ud83d\udd35 RECOVERY", "\ud83d\udd35 FORMATION",
    "\ud83d\udfe6 LAGGING_CATCHUP", "\ud83d\udfe1 CONSOLIDATION", "\ud83d\udfe0 NEUTRAL",
    "\ud83d\udd36 PULLBACK", "\ud83d\udfe4 FADING", "\u26a0\ufe0f WEAKENING", "\u2b07\ufe0f DOWNTREND",
    "\ud83d\udfe4 EXHAUSTING", "\ud83d\udfe3 COUNTER_RALLY", "\ud83d\udd34 CYCLE_PEAK",
  ];
  const presentClasses = new Set<string>();
  for (const sec of sectors) for (const cls of Object.keys(grouped[sec])) presentClasses.add(cls);
  const classes = STD_ORDER.filter((c) => presentClasses.has(c));

  // Build z matrices for each metric
  const buildZ = (metric: "1m" | "3m" | "ytd") => {
    return classes.map((cls) =>
      sectors.map((sec) => {
        const cd = grouped[sec]?.[cls];
        if (!cd) return null;
        if (metric === "1m") return cd.n > 0 ? cd.s1m / cd.n : null;
        if (metric === "3m") return cd.n > 0 ? cd.s3m / cd.n : null;
        return cd.nytd > 0 ? cd.sytd / cd.nytd : null;
      }) as (number | null)[],
    );
  };

  const z1m = buildZ("1m");
  const z3m = buildZ("3m");
  const zytd = buildZ("ytd");

  const heatmapShared = {
    type: "heatmap" as const,
    x: sectors,
    y: classes,
    colorscale: [
      [0,    "#7f1d1d"],   // dark red
      [0.25, "#CC0000"],   // red
      [0.45, "#F2E5D7"],   // dark (near zero)
      [0.55, "#F2E5D7"],
      [0.75, "#0A7D3F"],   // green
      [1,    "#14532d"],   // dark green
    ] as any,
    zmid: 0,
    zmin: -25,
    zmax: 25,
    showscale: true,
    hovertemplate: "%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
    colorbar: { tickfont: { color: "#66605C", size: 9 }, len: 0.85, thickness: 8 },
  };

  // Annotations on each cell \u2014 value only, count moved to tooltip
  const buildAnn = (z: (number | null)[][], metric: "1m" | "3m" | "ytd") => {
    const out: any[] = [];
    for (let i = 0; i < classes.length; i++) {
      for (let j = 0; j < sectors.length; j++) {
        const v = z[i][j];
        if (v == null) continue;
        const cd = grouped[sectors[j]][classes[i]];
        const n = metric === "ytd" ? cd.nytd : cd.n;
        if (n === 0) continue;
        out.push({
          x: sectors[j], y: classes[i],
          text: `${v >= 0 ? "+" : ""}${v.toFixed(1)}`,
          showarrow: false,
          font: { color: "#33302E", size: 10, family: "monospace" },
        });
      }
    }
    return out;
  };

  // Detailed hover with n
  const buildHover = (z: (number | null)[][], metric: "1m" | "3m" | "ytd") => {
    const text: string[][] = [];
    for (let i = 0; i < classes.length; i++) {
      const row: string[] = [];
      for (let j = 0; j < sectors.length; j++) {
        const v = z[i][j];
        const cd = grouped[sectors[j]]?.[classes[i]];
        const n = cd ? (metric === "ytd" ? cd.nytd : cd.n) : 0;
        if (v == null || n === 0) {
          row.push("");
        } else {
          row.push(`${classes[i]}<br>${sectors[j]}<br>${v >= 0 ? "+" : ""}${v.toFixed(2)}% (n=${n})`);
        }
      }
      text.push(row);
    }
    return text;
  };

  const heatHeight = Math.max(380, classes.length * 32 + 100);

  if (!sectors.length || !classes.length) return null;

  return (
    <div className="bg-[#FFFFFF] rounded-lg border border-[#9CC3D5]/40 p-4 mb-4">
      <div className="mb-3">
        <h3 className="text-[16px] font-semibold text-[#0F5499]">
          Returns by Classification × Sector (latest snapshot)
        </h3>
        <div className="text-[12px] text-[#857F7A] mt-0.5">
          Avg ticker return % per (classification × sector) cell · 3 horizons (1M / 3M / YTD)
          · color: red (−) ↔ green (+) capped at ±25% · hover for ticker count
        </div>
      </div>
      <div className="space-y-3">
        {[
          { key: "1m" as const,  label: "1-Month Return (21d)",  z: z1m  },
          { key: "3m" as const,  label: "3-Month Return (63d)",  z: z3m  },
          { key: "ytd" as const, label: "YTD Return",             z: zytd },
        ].map(({ key, label, z }) => (
          <div key={key} className="bg-[#FFF1E5] rounded p-2 border border-[#E6D9CE]">
            <div className="text-[13px] font-semibold text-[#33302E] mb-1.5 text-center">{label}</div>
            <Plot
              data={[{
                ...heatmapShared,
                z,
                text: buildHover(z, key) as any,
                hovertemplate: "%{text}<extra></extra>",
              }]}
              layout={{
                ...DARK_LAYOUT,
                height: heatHeight,
                margin: { t: 10, b: 90, l: 160, r: 60 },
                xaxis: { tickangle: -35, tickfont: { size: 10, color: "#66605C" }, side: "bottom" },
                yaxis: { tickfont: { size: 10, color: "#66605C" }, automargin: true },
                annotations: buildAnn(z, key),
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          </div>
        ))}
      </div>
      <div className="text-[12px] text-[#857F7A] mt-2 leading-relaxed">
        Cell value = mean return % across tickers in that (classification × sector) bucket.
        Hover for ticker count. Empty cells = no tickers in that combination.
      </div>
    </div>
  );
}


function SectorClassificationGrid({ data, allResults }: { data: SectorEntry[] | null; allResults?: any[] }) {
  const [open, setOpen] = useState(false);

  if (!data || !data.length) return null;

  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-2.5 text-left text-[14px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[#66605C]">Classification by Sector ({data.length} sectors)</span>
        <span className="text-[#857F7A] text-[12px]">{open ? "\u25bc" : "\u25b6"}</span>
      </button>
      {open && (
        <div className="bg-[#FBEEE3] p-4">
          {/* \u2500\u2500 Top: aggregate return panels by sector + classification \u2500\u2500 */}
          {allResults && allResults.length > 0 && (
            <>
              <ReturnBySectorPanel allResults={allResults} />
              <ReturnByClassificationPanel allResults={allResults} />
            </>
          )}

          {/* \u2500\u2500 Per-sector mini-tables (history) \u2500\u2500 */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {data.map((entry) => (
              <SectorMiniTable key={entry.sector} entry={entry} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Ticker Lookup Panel
// ---------------------------------------------------------------------------

function TickerLookup({
  allResults,
  pmMap,
  onNavigate,
}: {
  allResults: any[];
  pmMap: Map<string, any>;
  onNavigate: (stage: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<LookupResult | null>(null);

  // Build lookup results
  const results: LookupResult[] = useMemo(() => {
    if (!query.trim()) return [];
    const q = query.trim().toUpperCase();
    return allResults
      .filter(
        (r: any) =>
          r.ticker?.toUpperCase().includes(q) ||
          r.name?.toUpperCase().includes(q)
      )
      .slice(0, 10)
      .map((r: any) => {
        const cls = r.classification || "";
        const eligible = !!r.eligible;
        let stage: string;
        if (eligible) {
          stage = "momentum";
        } else if (PM_CLASSIFICATIONS.has(cls)) {
          stage = "pre-momentum";
        } else {
          stage = "excluded";
        }

        const pm = pmMap.get(r.ticker);
        return {
          ...r,
          stage,
          pm_score: pm?.pre_momentum_score,
          pm_agreement_ratio: pm?.agreement_ratio,
          pm_timeline: pm?.expected_timeline,
          pm_catalysts: pm?.key_catalysts,
          pm_agents: pm?.agents,
        };
      });
  }, [query, allResults, pmMap]);

  const handleSelect = useCallback((r: LookupResult) => {
    setSelected(r);
    setQuery(r.ticker);
  }, []);

  const badge = selected ? stageBadge(selected.stage) : null;

  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4 space-y-3">
      {/* Search input */}
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setSelected(null);
          }}
          placeholder="Search ticker or name..."
          className="w-full px-3 py-2 bg-[#FBEEE3] border border-[#E6D9CE] rounded text-[16px] text-[#33302E] placeholder-gray-600 focus:border-[#9CC3D5] focus:outline-none"
        />
        {query && !selected && results.length > 0 && (
          <div className="absolute z-30 top-full left-0 right-0 mt-1 bg-[#F2E5D7] border border-[#E6D9CE] rounded shadow-lg max-h-60 overflow-y-auto">
            {results.map((r) => {
              const b = stageBadge(r.stage);
              return (
                <button
                  key={r.ticker}
                  onClick={() => handleSelect(r)}
                  className="w-full px-3 py-2 text-left hover:bg-[#CCC1B7] flex items-center gap-3 text-[14px] border-b border-[#E6D9CE]/50 last:border-0"
                >
                  <span className="font-mono font-bold text-[#0F5499] w-16">{r.ticker}</span>
                  <span className="text-[#66605C] flex-1 truncate">{r.name}</span>
                  <span
                    className="text-[11px] px-1.5 py-0.5 rounded font-semibold shrink-0"
                    style={{ backgroundColor: b.bg, color: b.color }}
                  >
                    {b.label}
                  </span>
                  <span className="font-mono text-[#857F7A] w-12 text-right">{r.composite?.toFixed(1)}</span>
                </button>
              );
            })}
          </div>
        )}
        {query && !selected && results.length === 0 && (
          <div className="absolute z-30 top-full left-0 right-0 mt-1 bg-[#F2E5D7] border border-[#E6D9CE] rounded px-3 py-2 text-[14px] text-[#857F7A]">
            No results
          </div>
        )}
      </div>

      {/* Selected ticker detail */}
      {selected && badge && (
        <div className="space-y-3">
          {/* Header */}
          <div className="flex items-center gap-3 flex-wrap">
            <span className="text-[20px] font-bold text-[#0F5499] font-mono">{selected.ticker}</span>
            <span className="text-[16px] text-[#33302E]">{selected.name}</span>
            <span
              className="text-[12px] px-2 py-0.5 rounded font-bold"
              style={{ backgroundColor: badge.bg, color: badge.color }}
            >
              {badge.label}
            </span>
            <span
              className="text-[12px] px-2 py-0.5 rounded"
              style={{
                backgroundColor: (CLASS_COLORS[selected.classification] || C.gray) + "22",
                color: CLASS_COLORS[selected.classification] || C.gray,
              }}
            >
              {selected.classification}
            </span>
            {selected.rejection && (
              <span className="text-[12px] px-2 py-0.5 rounded bg-[#F7EDE0]/30 text-[#C2701C]">
                {selected.rejection}
              </span>
            )}
            <button
              onClick={() => onNavigate(selected.stage)}
              className="text-[12px] text-[#0F5499] hover:text-[#0D7680] underline ml-auto"
            >
              Go to {badge.label} tab →
            </button>
          </div>

          {/* Score axes */}
          <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
            {[
              { label: "Composite", value: selected.composite, color: compColor(selected.composite) },
              { label: "TCS", value: selected.tcs, color: compColor(selected.tcs) },
              { label: "TFS", value: selected.tfs, color: compColor(selected.tfs) },
              { label: "RSS", value: selected.rss, color: compColor(selected.rss) },
              { label: "OER", value: selected.oer, color: oerColor(selected.oer) },
              { label: "RSI", value: selected.rsi, color: C.gray },
              { label: "Trend", value: selected.trend_age, color: C.gray, suffix: "d" },
              { label: "SMA50", value: selected.sma50_dist, color: C.gray, suffix: "%" },
            ].map((a) => (
              <div key={a.label} className="bg-[#FBEEE3] rounded p-2 text-center">
                <div className="text-[11px] text-[#857F7A]">{a.label}</div>
                <div className="text-[16px] font-bold font-mono" style={{ color: a.color }}>
                  {typeof a.value === "number" ? (Number.isInteger(a.value) ? a.value : a.value.toFixed(1)) : "-"}
                  {(a as any).suffix || ""}
                </div>
              </div>
            ))}
          </div>

          {/* Strategy + Returns row */}
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            <div className="bg-[#FBEEE3] rounded p-2">
              <div className="text-[11px] text-[#857F7A]">Signal</div>
              <div className="text-[14px] font-semibold" style={{
                color: (selected.net_signal || "").includes("LONG") ? C.green
                  : (selected.net_signal || "").includes("SHORT") ? C.red : C.gray
              }}>
                {selected.net_signal || "-"}
              </div>
            </div>
            <div className="bg-[#FBEEE3] rounded p-2">
              <div className="text-[11px] text-[#857F7A]">L / S</div>
              <div className="text-[14px] font-mono">
                <span className="text-[#0A7D3F]">{selected.long_count ?? 0}</span>
                <span className="text-[#857F7A]"> / </span>
                <span className="text-[#CC0000]">{selected.short_count ?? 0}</span>
              </div>
            </div>
            <div className="bg-[#FBEEE3] rounded p-2">
              <div className="text-[11px] text-[#857F7A]">ADV</div>
              <div className="text-[14px] font-mono">${selected.adv_M?.toFixed(1)}M</div>
            </div>
            <div className="bg-[#FBEEE3] rounded p-2">
              <div className="text-[11px] text-[#857F7A]">1W Return</div>
              <div className="text-[14px] font-mono" style={{ color: retColor(selected.ret_1w) }}>
                {selected.ret_1w?.toFixed(1)}%
              </div>
            </div>
            <div className="bg-[#FBEEE3] rounded p-2">
              <div className="text-[11px] text-[#857F7A]">1M Return</div>
              <div className="text-[14px] font-mono" style={{ color: retColor(selected.ret_1m) }}>
                {selected.ret_1m?.toFixed(1)}%
              </div>
            </div>
            <div className="bg-[#FBEEE3] rounded p-2">
              <div className="text-[11px] text-[#857F7A]">3M Return</div>
              <div className="text-[14px] font-mono" style={{ color: retColor(selected.ret_3m) }}>
                {selected.ret_3m?.toFixed(1)}%
              </div>
            </div>
          </div>

          {/* PM-specific info (only for pre-momentum tickers) */}
          {selected.stage === "pre-momentum" && selected.pm_score != null && (
            <div className="border border-[#9CC3D5]/40 rounded-lg p-3 space-y-2">
              <div className="text-[12px] text-[#0F5499] font-semibold uppercase tracking-wide">Pre-Momentum Signals</div>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                <div className="bg-[#FBEEE3] rounded p-2 text-center">
                  <div className="text-[11px] text-[#857F7A]">PM Score</div>
                  <div className="text-[16px] font-bold font-mono text-[#0F5499]">{selected.pm_score.toFixed(1)}</div>
                </div>
                <div className="bg-[#FBEEE3] rounded p-2 text-center">
                  <div className="text-[11px] text-[#857F7A]">Agreement</div>
                  <div className="text-[14px] font-bold font-mono" style={{
                    color: (selected.pm_agreement_ratio ?? 0) >= 0.6 ? C.green
                      : (selected.pm_agreement_ratio ?? 0) >= 0.4 ? C.cyan
                      : (selected.pm_agreement_ratio ?? 0) > 0 ? C.yellow : C.gray
                  }}>
                    {selected.pm_agreement_ratio != null ? `${(selected.pm_agreement_ratio * 100).toFixed(0)}%` : "-"}
                  </div>
                </div>
                <div className="bg-[#FBEEE3] rounded p-2 text-center">
                  <div className="text-[11px] text-[#857F7A]">Timeline</div>
                  <div className="text-[14px] text-[#33302E]">{selected.pm_timeline}</div>
                </div>
                {selected.pm_agents && (
                  <>
                    <div className="bg-[#FBEEE3] rounded p-2 text-center">
                      <div className="text-[11px] text-[#857F7A]">Micro / Macro</div>
                      <div className="text-[14px] font-mono text-[#33302E]">
                        {selected.pm_agents.microstructure?.score?.toFixed(0)} / {selected.pm_agents.macro_regime?.score?.toFixed(0)}
                      </div>
                    </div>
                    <div className="bg-[#FBEEE3] rounded p-2 text-center">
                      <div className="text-[11px] text-[#857F7A]">Graph / Catalyst</div>
                      <div className="text-[14px] font-mono text-[#33302E]">
                        {selected.pm_agents.graph_relational?.score?.toFixed(0)} / {selected.pm_agents.catalyst?.score?.toFixed(0)}
                      </div>
                    </div>
                  </>
                )}
              </div>
              {selected.pm_catalysts && selected.pm_catalysts.length > 0 && (
                <div className="text-[13px] text-[#66605C]">
                  <span className="text-[#857F7A]">Catalysts: </span>
                  {selected.pm_catalysts.slice(0, 3).join(" · ")}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Export
// ---------------------------------------------------------------------------

export function PriceDiscoveryTab({ filters }: { filters: FilterParams }) {
  const [sub, setSub] = useState(0);
  const [allResults, setAllResults] = useState<any[]>([]);
  const [pmMap, setPmMap] = useState<Map<string, any>>(new Map());
  const [classHistory, setClassHistory] = useState<ClassHistoryData | null>(null);
  const [sectorHistory, setSectorHistory] = useState<SectorEntry[] | null>(null);

  // Load data for the search panel + classification history
  useEffect(() => {
    fetchTable(filters).then((res) => setAllResults(res.data || []));
    fetchClassificationHistory().then(setClassHistory).catch(() => {});
    fetchClassificationHistoryBySector().then((res) => setSectorHistory(res.sectors || [])).catch(() => {});
    fetchPreMomentum().then((res) => {
      const map = new Map<string, any>();
      for (const c of res.candidates || []) {
        map.set(c.ticker, c);
      }
      setPmMap(map);
    });
  }, [filters]);

  const handleNavigate = useCallback((stage: string) => {
    if (stage === "pre-momentum") setSub(0);
    else if (stage === "momentum") setSub(1);
    else setSub(2);
  }, []);

  return (
    <div className="space-y-4">
      {/* Ticker Search */}
      <TickerLookup allResults={allResults} pmMap={pmMap} onNavigate={handleNavigate} />

      {/* Classification Definitions (above the distribution heatmap) */}
      <ClassificationDefinitionsPanel />

      {/* Classification Age — definition & calculation */}
      <ClassificationAgePanel />

      {/* Classification History */}
      <ClassificationHistoryChart data={classHistory} />
      <SectorClassificationGrid data={sectorHistory} allResults={allResults} />

      {/* Sub-tab bar */}
      <div className="flex items-center gap-1 border-b border-[#E6D9CE]">
        {SUBS.map(({ label }, i) => (
          <button
            key={label}
            onClick={() => setSub(i)}
            className={`px-4 py-2 text-[16px] font-medium border-b-2 transition-colors ${
              sub === i
                ? "border-[#0F5499] text-[#0F5499]"
                : "border-transparent text-[#857F7A] hover:text-[#33302E]"
            }`}
          >
            {label}
          </button>
        ))}
        <span className="ml-3 text-[13px] text-[#857F7A]">{SUBS[sub].desc}</span>
      </div>

      {/* Sub-tab content */}
      {sub === 0 && <PreMomentumTab totalUniverse={allResults.length} filterSectors={filters.sectors} />}
      {sub === 1 && <MomentumTab filters={filters} totalUniverse={allResults.length} />}
      {sub === 2 && <NewPriceDiscoveryTab filters={filters} totalUniverse={allResults.length} />}
      {sub === 3 && <New2PriceDiscoveryTab filters={filters} totalUniverse={allResults.length} />}
      {sub === 4 && <ExcludedTab filters={filters} totalUniverse={allResults.length} />}
    </div>
  );
}
