import React, { useEffect, useState, useMemo } from "react";
import { fetchTable, fetchTableML, fetchOverview, type FilterParams } from "../../api/client";
import { MetricCard } from "../shared/MetricCard";
import { CLASS_COLORS, C } from "../../styles/theme";
import { useSort } from "../../hooks/useSort";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Ticker {
  ticker: string;
  name: string;
  category: string;
  sector?: string;
  theme?: string;
  asset_type: string;
  composite: number;
  tcs: number;
  tfs: number;
  oer: number;
  rss: number;
  qvr_score?: number;
  qvr_q?: number;
  qvr_v?: number;
  qvr_r?: number;
  qvr_n_analysts?: number;
  qvr_bullish_chg_3m?: number | null;
  qvr_eps_beat_rate?: number | null;
  qvr_eps_surprise_avg?: number | null;
  classification: string;
  eligible: boolean;
  rsi: number;
  trend_age: number;
  sma50_dist: number;
  oneil_long: number;
  oneil_short: number;
  minervini_long: number;
  minervini_short: number;
  wyckoff_long: number;
  wyckoff_short: number;
  ichimoku_long: number;
  ichimoku_short: number;
  darvas_long: number;
  darvas_short: number;
  regime_long: number;
  regime_short: number;
  flow_long: number;
  flow_short: number;
  relval_long: number;
  relval_short: number;
  combined_long: number;
  combined_short: number;
  net_signal: string;
  long_count: number;
  short_count: number;
  ret_1w: number;
  ret_1m: number;
  ret_3m: number;
  score_1w: number;
  score_1m: number;
  val_prob: number;
  mom_age: number;
  // Multi-horizon returns
  ret_1d: number;
  ret_5d: number;
  ret_21d: number;
  ret_63d: number;
  ret_126d: number;
  ret_ytd?: number | null;
  ret_252d: number;
  ret_3y_ann: number | null;
  ret_5y_ann: number | null;
  vol_3y_ann: number | null;
}

// ---------------------------------------------------------------------------
// Expandable Section (same pattern as PreMomentumTab)
// ---------------------------------------------------------------------------

export function Section({
  title,
  children,
  defaultOpen = false,
  badge,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-3 text-left text-[16px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="flex items-center gap-2">
          {title}
          {badge && (
            <span className="text-[12px] px-1.5 py-0.5 rounded bg-[#E3EEF5]/50 text-[#0F5499]">
              {badge}
            </span>
          )}
        </span>
        <span className="text-[#857F7A] text-[14px]">{open ? "\u25BC" : "\u25B6"}</span>
      </button>
      {open && <div className="p-4 bg-[#FBEEE3] space-y-4">{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

function compColor(v: number): string {
  if (v >= 70) return C.green;
  if (v >= 55) return C.cyan;
  if (v >= 40) return C.yellow;
  return C.gray;
}

function retColor(v: number): string {
  if (v > 3) return C.green;
  if (v > 0) return "#0A7D3F";
  if (v > -3) return "#CC0000";
  return C.red;
}

function oerColor(v: number): string {
  if (v >= 60) return C.red;
  if (v >= 40) return C.orange;
  return C.green;
}

function qvrColor(v: number): string {
  if (v >= 65) return C.green;
  if (v >= 50) return C.cyan;
  if (v >= 40) return C.yellow;
  return C.red;
}

function signalColor(sig: string): string {
  if (sig.includes("STRONG_LONG")) return C.green;
  if (sig.includes("LONG")) return "#0A7D3F";
  if (sig.includes("STRONG_SHORT")) return C.red;
  if (sig.includes("SHORT")) return "#CC0000";
  return C.gray;
}

// ---------------------------------------------------------------------------
// Decision Logic — classification × OER × age × signal 종합
// ---------------------------------------------------------------------------

interface Decision {
  action: string;
  rationale: string;
  color: string;
  rank: number;  // for sorting (lower = more aggressive buy, higher = sell/hedge)
}

function decideAction(r: Ticker): Decision {
  const cls = r.classification || "";
  const oer = r.oer ?? 0;
  const age = r.mom_age ?? 0;
  const signal = r.net_signal || "";
  const composite = r.composite ?? 0;

  // ── OVEREXTENDED — 과열 단계 ──
  if (cls === "🟡 OVEREXTENDED") {
    if (oer >= 80) {
      return {
        action: "HEDGE",
        rationale: `OER ${oer.toFixed(0)} 매우 높음 → 적극 청산 또는 풋옵션 헷지`,
        color: C.red,
        rank: 9,
      };
    }
    if (oer >= 70) {
      return {
        action: "TRIM",
        rationale: `과열 심화 (OER ${oer.toFixed(0)}) → 30-50% 부분 청산`,
        color: C.orange,
        rank: 8,
      };
    }
    return {
      action: "TRIM 검토",
      rationale: `과열 진입 (OER ${oer.toFixed(0)}) → 신규 매수 보류, 보유는 일부 익절`,
      color: "#C2701C",
      rank: 7,
    };
  }

  // ── CYCLE_PEAK / DOWNTREND — 청산 ──
  if (cls.includes("CYCLE_PEAK") || cls.includes("DOWNTREND")) {
    return {
      action: "EXIT",
      rationale: "추세 종료 — 보유 청산",
      color: C.red,
      rank: 10,
    };
  }

  // ── EXHAUSTING / FADING / WEAKENING — 약화 ──
  if (cls.includes("EXHAUSTING") || cls.includes("FADING") || cls.includes("WEAKENING")) {
    return {
      action: "REDUCE",
      rationale: "추세 약화 — 비중 축소",
      color: C.orange,
      rank: 8,
    };
  }

  // ── COUNTER_RALLY — 가짜 반등 ──
  if (cls.includes("COUNTER_RALLY")) {
    return {
      action: "AVOID",
      rationale: "장기 하락 중 단기 반등 (dead-cat) — 진입 회피",
      color: C.red,
      rank: 9,
    };
  }

  // ── FORMATION — 초기 돌파 ──
  if (cls === "🔵 FORMATION") {
    if (signal.includes("STRONG_LONG")) {
      return {
        action: "BUY",
        rationale: "초기 돌파 + 강한 전략 합의 → 적극 진입 (타이트 stop)",
        color: C.green,
        rank: 1,
      };
    }
    if (signal.includes("LONG")) {
      return {
        action: "BUY",
        rationale: "초기 돌파 확인 → 진입 적기 (분할 매수 권장)",
        color: C.green,
        rank: 2,
      };
    }
    return {
      action: "SCALE-IN",
      rationale: "초기 돌파이나 합의 부족 → 1/3씩 분할 매수",
      color: "#0A7D3F",
      rank: 3,
    };
  }

  // ── LAGGING_CATCHUP — 섹터 후행 종목 ──
  if (cls === "🟦 LAGGING_CATCHUP") {
    return {
      action: "BUY (Catch-up)",
      rationale: "섹터 후행 종목 → 따라잡기 진입 후보",
      color: C.green,
      rank: 2,
    };
  }

  // ── CONTINUATION — 메인 모멘텀 ──
  if (cls === "🟢 CONTINUATION") {
    // 장기 추세 (90일+) — 부분 익절 검토
    if (age >= 90 && oer >= 50) {
      return {
        action: "TRIM 일부",
        rationale: `장기 추세(${age}d) + OER ${oer.toFixed(0)} → 일부 익절 / stop 상향`,
        color: "#C2701C",
        rank: 7,
      };
    }
    if (age >= 90) {
      return {
        action: "HOLD",
        rationale: `장기 추세 유지(${age}d) → 보유, stop 상향`,
        color: C.cyan,
        rank: 5,
      };
    }
    // OER 주의 단계
    if (oer >= 50) {
      return {
        action: "HOLD + 주시",
        rationale: `상승 추세 + OER ${oer.toFixed(0)} → 60 도달 시 hedge 준비`,
        color: "#0A7D3F",
        rank: 4,
      };
    }
    // 강한 신호
    if (signal.includes("STRONG_LONG")) {
      return {
        action: "BUY",
        rationale: `상승 추세 + 강한 전략 합의 (composite ${composite.toFixed(0)}) → 신규 진입 적기`,
        color: C.green,
        rank: 1,
      };
    }
    if (signal.includes("LONG")) {
      return {
        action: "ACCUMULATE",
        rationale: `상승 추세 진행 중 → 점진 매집`,
        color: "#0A7D3F",
        rank: 3,
      };
    }
    return {
      action: "HOLD",
      rationale: "정상 추세 — 보유 유지",
      color: C.green,
      rank: 4,
    };
  }

  // ── RECOVERY — 회복 초기 ──
  if (cls === "🔵 RECOVERY") {
    if (composite >= 60 && signal.includes("LONG")) {
      return {
        action: "ACCUMULATE",
        rationale: `회복 초기 + 합의 (composite ${composite.toFixed(0)}) → 점진 매집 시작`,
        color: "#0A7D3F",
        rank: 3,
      };
    }
    if (composite >= 60) {
      return {
        action: "SCALE-IN",
        rationale: "회복 초기 — 분할 진입 (확정 후 추가)",
        color: "#0A7D3F",
        rank: 4,
      };
    }
    return {
      action: "WATCH",
      rationale: "회복 초기 — 추세 확인 후 진입 권장",
      color: C.yellow,
      rank: 6,
    };
  }

  // ── 기본값 ──
  return {
    action: "HOLD",
    rationale: "—",
    color: C.gray,
    rank: 5,
  };
}

// ---------------------------------------------------------------------------
// Column Definitions Toggle
// ---------------------------------------------------------------------------

const COLUMN_DEFS = [
  { col: "Ticker", desc: "Ticker symbol (click for detail)" },
  { col: "Name", desc: "Security name" },
  { col: "Sector", desc: "통합 17개 섹터 (GICS 11 + Fixed Income / International / Equity Broad / Macro / Multi-Asset / Alternatives). ETF/주식 동일 체계." },
  { col: "Class", desc: "Dual-timeframe classification (3\u00d73 matrix + overrides)" },
  { col: "Comp", desc: "Composite score (0-100): 0.35\u00d7TCS + 0.30\u00d7TFS + 0.35\u00d7RSS" },
  { col: "TCS", desc: "Trend Continuation Score \u2014 established momentum persistence (SMA position, slope, trend age)" },
  { col: "TFS", desc: "Trend Formation Score \u2014 early/new momentum (SMA breakout, volume surge, breakout freshness)" },
  { col: "RSS", desc: "Relative Strength Score \u2014 multi-horizon return percentile vs. universe (5d/21d/63d/12-1M)" },
  { col: "OER", desc: "Overextension Risk \u2014 mean-reversion exposure (SMA distance, RSI overbought, 52W high proximity). Higher = riskier" },
  { col: "QVR", desc: "Quality-Value-Revision (0-100). 0.30\u00b7Q (margin/ROE) + 0.20\u00b7V (inverse PE/PEG/PB) + 0.50\u00b7R (EPS revision momentum + analyst sentiment). Eligibility gate: stocks with QVR<40 are demoted from Momentum (rejection: WeakQVR). ETFs receive neutral 50 and bypass the gate. Hover for Q/V/R sub-scores." },
  { col: "Age", desc: "Momentum age in days — how long ticker has been in confirmed uptrend (CONTINUATION/FORMATION/OVEREXTENDED) using bi-weekly noise-tolerant tracking. Capped at 180d. Green=fresh, Red=aged/extended." },
  { col: "Decision", desc: "분석결과 종합 의사결정 (BUY/HOLD/ACCUMULATE/SCALE-IN/TRIM/HEDGE/EXIT 등). classification × OER × age × signal 조합 기반. Sort 시 적극 매수(rank 1) → 청산(rank 10) 순서." },
  { col: "Signal", desc: "Net hedge strategy signal: STRONG_LONG / LONG / NEUTRAL / SHORT / STRONG_SHORT (from 8 strategies)" },
  { col: "L/S", desc: "Long / Short strategy count \u2014 number of strategies (out of 8) scoring \u2265 60 for each direction" },
  { col: "RSI", desc: "14-day Relative Strength Index (>70 overbought, <30 oversold)" },
  { col: "Trend", desc: "Trend age \u2014 consecutive days price has been above SMA50" },
  { col: "1D", desc: "1-day return (%)" },
  { col: "1W", desc: "1-week return (%) \u2014 5 trading days" },
  { col: "1M", desc: "1-month return (%) \u2014 21 trading days" },
  { col: "3M", desc: "3-month return (%) \u2014 63 trading days" },
  { col: "6M", desc: "6-month return (%) \u2014 126 trading days" },
  { col: "YTD", desc: "Year-to-date return (%) \u2014 prior year-end close to today" },
  { col: "1Y", desc: "1-year return (%) \u2014 252 trading days" },
  { col: "3Y/A", desc: "3-year annualized return (%)" },
  { col: "5Y/A", desc: "5-year annualized return (%)" },
  { col: "Vol3Y", desc: "3-year annualized volatility (%) \u2014 standard deviation of daily returns" },
];

// ---------------------------------------------------------------------------
// Composite Score Formula Guide
// ---------------------------------------------------------------------------

interface EWeight {
  axis: string;
  component: string;
  outer: string;
  inner: string;
  effective: number;
  color: string;
  isMax?: boolean;
}

export function CompositeFormulaGuide() {
  const [open, setOpen] = useState(false);

  const effectiveWeights: EWeight[] = [
    { axis: "TCS", component: "TCS_short (단기 추세 지속)", outer: "0.30", inner: "0.40", effective: 0.120, color: "#0A7D3F" },
    { axis: "TCS", component: "TCS_long (장기 추세 지속)",  outer: "0.30", inner: "0.60", effective: 0.180, color: "#0A7D3F" },
    { axis: "TFS", component: "TFS_short (단기 추세 형성)", outer: "0.25", inner: "0.50", effective: 0.125, color: "#0F5499" },
    { axis: "TFS", component: "TFS_long (장기 추세 형성)",  outer: "0.25", inner: "0.50", effective: 0.125, color: "#0F5499" },
    { axis: "RSS", component: "RSS_short (단기 상대강도)",  outer: "0.30", inner: "0.35", effective: 0.105, color: "#B85C00" },
    { axis: "RSS", component: "RSS_long (장기 상대강도, 12-1M)", outer: "0.30", inner: "0.65", effective: 0.195, color: "#B85C00", isMax: true },
    { axis: "URS", component: "LeadLag (카테고리 평균 - 본인 ret_63d)", outer: "0.15", inner: "0.40", effective: 0.060, color: "#7D5BA6" },
    { axis: "URS", component: "AttnGap (vol_ratio_3d_10d - ret_5d pctile)", outer: "0.15", inner: "0.30", effective: 0.045, color: "#7D5BA6" },
    { axis: "URS", component: "Drift (PEAD post-event drift)", outer: "0.15", inner: "0.20", effective: 0.030, color: "#7D5BA6" },
    { axis: "URS", component: "Dispersion (cross-sectional rss_std)", outer: "0.15", inner: "0.10", effective: 0.015, color: "#7D5BA6" },
  ];

  const arrow = open ? "▼" : "▶";

  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden mb-2">
      <button
        className="w-full px-4 py-2 text-left text-[14px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[#66605C]">
          Composite Score 산출 공식
          <span className="ml-2 text-[12px] text-[#857F7A]">— 4축 통합 가중평균</span>
        </span>
        <span className="text-[#857F7A] text-[12px]">{arrow}</span>
      </button>
      {open && (
        <div className="bg-[#FBEEE3] p-4 space-y-4">
          <div className="bg-[#FFFFFF] border border-[#9CC3D5]/40 rounded-lg p-3 text-center">
            <div className="text-[12px] text-[#857F7A] uppercase tracking-wide mb-2">Main Formula</div>
            <div className="font-mono text-[16px] text-[#0D7680]">
              <span className="text-[#0F5499] font-bold">Composite</span>
              {" = "}
              <span className="text-[#0A7D3F]">0.30 × TCS</span>
              {" + "}
              <span className="text-[#0F5499]">0.25 × TFS</span>
              {" + "}
              <span className="text-[#C2701C]">0.30 × RSS</span>
              {" + "}
              <span className="text-[#7D5BA6]">0.15 × URS</span>
            </div>
            <div className="text-[12px] text-[#857F7A] mt-2">
              각 축은 0-100. <strong className="text-[#66605C]">OER은 분류 전용</strong>으로 Composite에 미포함.
            </div>
          </div>

          <div>
            <div className="text-[14px] font-semibold text-[#33302E] mb-2">각 축의 단기/장기 분해</div>
            <table className="w-full text-[14px] border-collapse">
              <thead>
                <tr className="border-b border-[#E6D9CE] bg-[#FFFFFF]">
                  <th className="py-1.5 px-2 text-left text-[#857F7A]">Axis</th>
                  <th className="py-1.5 px-2 text-center text-[#857F7A]">단기 weight</th>
                  <th className="py-1.5 px-2 text-center text-[#857F7A]">장기 weight</th>
                  <th className="py-1.5 px-2 text-left text-[#857F7A]">의미</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-[#E6D9CE]/50">
                  <td className="py-1.5 px-2 font-bold" style={{ color: "#0A7D3F" }}>TCS</td>
                  <td className="py-1.5 px-2 text-center font-mono text-[#33302E]">0.40</td>
                  <td className="py-1.5 px-2 text-center font-mono text-[#33302E]">0.60</td>
                  <td className="py-1.5 px-2 text-[#66605C]">추세 지속성 (장기 가중)</td>
                </tr>
                <tr className="border-b border-[#E6D9CE]/50">
                  <td className="py-1.5 px-2 font-bold" style={{ color: "#0F5499" }}>TFS</td>
                  <td className="py-1.5 px-2 text-center font-mono text-[#33302E]">0.50</td>
                  <td className="py-1.5 px-2 text-center font-mono text-[#33302E]">0.50</td>
                  <td className="py-1.5 px-2 text-[#66605C]">추세 형성 신선도 (대칭)</td>
                </tr>
                <tr className="border-b border-[#E6D9CE]/50">
                  <td className="py-1.5 px-2 font-bold" style={{ color: "#B85C00" }}>RSS</td>
                  <td className="py-1.5 px-2 text-center font-mono text-[#33302E]">0.35</td>
                  <td className="py-1.5 px-2 text-center font-mono text-[#33302E]">0.65</td>
                  <td className="py-1.5 px-2 text-[#66605C]">상대 강도 (장기 12-1M, Jegadeesh-Titman 1993)</td>
                </tr>
                <tr>
                  <td className="py-1.5 px-2 font-bold" style={{ color: "#7D5BA6" }}>URS</td>
                  <td className="py-1.5 px-2 text-center font-mono text-[#857F7A]" colSpan={2}>4 sub-signals</td>
                  <td className="py-1.5 px-2 text-[#66605C]">Underreaction (AQR, Hong-Stein 1999)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div>
            <div className="text-[14px] font-semibold text-[#33302E] mb-2">URS (Underreaction Score) 내부 구성</div>
            <table className="w-full text-[14px] border-collapse">
              <thead>
                <tr className="border-b border-[#E6D9CE] bg-[#FFFFFF]">
                  <th className="py-1.5 px-2 text-left text-[#857F7A]">Sub</th>
                  <th className="py-1.5 px-2 text-center text-[#857F7A]">Weight</th>
                  <th className="py-1.5 px-2 text-left text-[#857F7A]">의미</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-[#E6D9CE]/50">
                  <td className="py-1.5 px-2 font-bold text-[#7D5BA6]">LeadLag</td>
                  <td className="py-1.5 px-2 text-center font-mono">0.40</td>
                  <td className="py-1.5 px-2 text-[#66605C]">카테고리 평균 ret_63d − 본인 ret_63d (laggard 정도)</td>
                </tr>
                <tr className="border-b border-[#E6D9CE]/50">
                  <td className="py-1.5 px-2 font-bold text-[#7D5BA6]">AttnGap</td>
                  <td className="py-1.5 px-2 text-center font-mono">0.30</td>
                  <td className="py-1.5 px-2 text-[#66605C]">vol_ratio_3d_10d pctile − ret_5d pctile (관심은 있는데 가격 미반응)</td>
                </tr>
                <tr className="border-b border-[#E6D9CE]/50">
                  <td className="py-1.5 px-2 font-bold text-[#7D5BA6]">Drift</td>
                  <td className="py-1.5 px-2 text-center font-mono">0.20</td>
                  <td className="py-1.5 px-2 text-[#66605C]">gap_drift_30d (Post-Event Announcement Drift, PEAD)</td>
                </tr>
                <tr>
                  <td className="py-1.5 px-2 font-bold text-[#7D5BA6]">Dispersion</td>
                  <td className="py-1.5 px-2 text-center font-mono">0.10</td>
                  <td className="py-1.5 px-2 text-[#66605C]">cross-sectional rss_std (정보 비대칭 정도)</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div>
            <div className="text-[14px] font-semibold text-[#33302E] mb-2">
              Effective Weights — 최종 영향력 (outer × inner)
            </div>
            <table className="w-full text-[14px] border-collapse">
              <thead>
                <tr className="border-b border-[#E6D9CE] bg-[#FFFFFF]">
                  <th className="py-1.5 px-2 text-left text-[#857F7A]">Component</th>
                  <th className="py-1.5 px-2 text-center text-[#857F7A]">Outer</th>
                  <th className="py-1.5 px-2 text-center text-[#857F7A]">Inner</th>
                  <th className="py-1.5 px-2 text-right text-[#857F7A]">Effective</th>
                  <th className="py-1.5 px-2 text-left text-[#857F7A]">Bar</th>
                </tr>
              </thead>
              <tbody>
                {effectiveWeights.map((w, i) => (
                  <tr key={i} className="border-b border-[#E6D9CE]/50">
                    <td className="py-1 px-2">
                      <span className="font-mono text-[12px] mr-2 px-1 rounded" style={{ backgroundColor: w.color + "22", color: w.color }}>
                        {w.axis}
                      </span>
                      <span className="text-[#33302E]">{w.component}</span>
                      {w.isMax && <span className="ml-2 text-[12px] text-[#B85C00]">MAX</span>}
                    </td>
                    <td className="py-1 px-2 text-center font-mono text-[#857F7A]">{w.outer}</td>
                    <td className="py-1 px-2 text-center font-mono text-[#857F7A]">{w.inner}</td>
                    <td className="py-1 px-2 text-right font-mono font-bold" style={{ color: w.color }}>
                      {w.effective.toFixed(3)}
                    </td>
                    <td className="py-1 px-2">
                      <div
                        className="rounded h-2"
                        style={{
                          width: `${(w.effective / 0.195) * 100}%`,
                          backgroundColor: w.color + "AA",
                          minWidth: "20px",
                        }}
                      />
                    </td>
                  </tr>
                ))}
                <tr className="bg-[#FFFFFF]">
                  <td className="py-1.5 px-2 font-bold text-[#0F5499]">합계</td>
                  <td className="py-1.5 px-2"></td>
                  <td className="py-1.5 px-2"></td>
                  <td className="py-1.5 px-2 text-right font-mono font-bold text-[#0F5499]">1.000</td>
                  <td className="py-1.5 px-2"></td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-[13px]">
              <div className="text-[#33302E] font-semibold mb-1">장기 vs 단기 비중</div>
              <div className="text-[#857F7A]">
                장기 <span className="text-[#0A7D3F] font-mono">0.500</span> vs 단기 <span className="text-[#C2701C] font-mono">0.350</span>
                <br />→ 단기 노이즈보다 지속 추세 우선
              </div>
            </div>
            <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-[13px]">
              <div className="text-[#33302E] font-semibold mb-1">최대 가중치</div>
              <div className="text-[#857F7A]">
                <span className="text-[#C2701C]">RSS_long (19.5%)</span>
                <br />→ 12-1M 모멘텀 (Jegadeesh-Titman 1993)이 가장 강력한 신호
              </div>
            </div>
            <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-[13px]">
              <div className="text-[#33302E] font-semibold mb-1">퍼센타일 기반</div>
              <div className="text-[#857F7A]">
                <span className="text-[#C2701C]">RSS</span>, <span className="text-[#7D5BA6]">URS</span> 모두 cross-sectional 퍼센타일
                <br />→ 시장 환경에 robust (절대값이 아닌 상대 순위)
              </div>
            </div>
            <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded p-2 text-[13px]">
              <div className="text-[#33302E] font-semibold mb-1">가격 + 행동학 통합</div>
              <div className="text-[#857F7A]">
                가격 신호 (TCS+TFS+RSS): <span className="font-mono">85%</span>
                <br />행동학 보정 (URS): <span className="font-mono">15%</span>
              </div>
            </div>
          </div>

          <div className="bg-[#FFFFFF] border border-[#9CC3D5]/40 rounded-lg p-3">
            <div className="text-[14px] font-semibold text-[#0D7680] mb-1">Momentum 자격 조건</div>
            <div className="text-[13px] text-[#66605C] leading-relaxed">
              <code className="text-[#0F5499]">Composite ≥ 55</code> AND <code className="text-[#0F5499]">Bullish classification</code> (CONTINUATION/FORMATION/RECOVERY/OVEREXTENDED) AND <code className="text-[#0F5499]">ADV ≥ $5M</code>
              <br /><span className="text-[#857F7A]">→ 세 조건 모두 충족 시 Momentum 탭에 표시 (eligible=True)</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function ColumnDefinitions() {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden mb-2">
      <button
        className="w-full px-4 py-2 text-left text-[14px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[#66605C]">Column Definitions</span>
        <span className="text-[#857F7A] text-[12px]">{open ? "\u25BC" : "\u25B6"}</span>
      </button>
      {open && (
        <div className="bg-[#FBEEE3] px-4 py-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-6 gap-y-1.5">
          {COLUMN_DEFS.map((d) => (
            <div key={d.col} className="flex gap-2 text-[13px]">
              <span className="text-[#0F5499] font-semibold shrink-0 w-auto min-w-16">{d.col}</span>
              <span className="text-[#857F7A]">{d.desc}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Momentum Table
// ---------------------------------------------------------------------------

export function MomentumTable({
  rows,
  onSelect,
}: {
  rows: Ticker[];
  onSelect: (ticker: string) => void;
}) {
  const accessors = useMemo(() => ({
    ticker: (r: Ticker) => r.ticker,
    name: (r: Ticker) => r.name,
    sector: (r: Ticker) => r.sector || r.category,
    classification: (r: Ticker) => r.classification,
    composite: (r: Ticker) => r.composite,
    tcs: (r: Ticker) => r.tcs,
    tfs: (r: Ticker) => r.tfs,
    rss: (r: Ticker) => r.rss,
    oer: (r: Ticker) => r.oer,
    qvr: (r: Ticker) => r.qvr_score ?? 50,
    signal: (r: Ticker) => r.net_signal || "",
    long_count: (r: Ticker) => r.long_count ?? 0,
    age: (r: Ticker) => r.mom_age ?? 0,
    decision: (r: Ticker) => decideAction(r).rank,
    rsi: (r: Ticker) => r.rsi,
    trend: (r: Ticker) => r.trend_age,
    ret_1d: (r: Ticker) => r.ret_1d ?? 0,
    ret_5d: (r: Ticker) => r.ret_5d ?? 0,
    ret_21d: (r: Ticker) => r.ret_21d ?? 0,
    ret_63d: (r: Ticker) => r.ret_63d ?? 0,
    ret_126d: (r: Ticker) => r.ret_126d ?? 0,
    ret_ytd: (r: Ticker) => r.ret_ytd ?? 0,
    ret_252d: (r: Ticker) => r.ret_252d ?? 0,
    ret_3y_ann: (r: Ticker) => r.ret_3y_ann ?? -999,
    ret_5y_ann: (r: Ticker) => r.ret_5y_ann ?? -999,
    vol_3y_ann: (r: Ticker) => r.vol_3y_ann ?? -999,
  }), []);
  const { sorted, onSort, indicator } = useSort(rows, accessors);

  if (!rows.length) return <p className="text-[12px] text-[#857F7A]">No data</p>;
  const headerCls = "py-1.5 px-2 text-[#857F7A] cursor-pointer select-none hover:text-[#33302E] whitespace-nowrap";
  return (
    <div className="overflow-auto border border-[#E6D9CE] rounded" style={{ maxHeight: "600px" }}>
      <table className="w-full text-[14px] border-collapse">
        <thead className="sticky top-0 z-10 bg-[#F2E5D7]">
          <tr className="border-b border-[#E6D9CE]">
            <th className="py-1.5 px-2 text-left text-[#857F7A]">#</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("ticker")}>Ticker{indicator("ticker")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("name")}>Name{indicator("name")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("sector")}>Sector{indicator("sector")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("classification")}>Class{indicator("classification")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("composite")}>Comp{indicator("composite")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("decision")}>Decision{indicator("decision")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("tcs")}>TCS{indicator("tcs")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("tfs")}>TFS{indicator("tfs")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("rss")}>RSS{indicator("rss")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("oer")}>OER{indicator("oer")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("qvr")}>QVR{indicator("qvr")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("age")}>Age{indicator("age")}</th>
            <th className={`${headerCls} text-center`} onClick={() => onSort("signal")}>Signal{indicator("signal")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("long_count")}>L/S{indicator("long_count")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("rsi")}>RSI{indicator("rsi")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("trend")}>Trend{indicator("trend")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_1d")}>1D{indicator("ret_1d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_5d")}>1W{indicator("ret_5d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_21d")}>1M{indicator("ret_21d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_63d")}>3M{indicator("ret_63d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_126d")}>6M{indicator("ret_126d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_ytd")}>YTD{indicator("ret_ytd")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_252d")}>1Y{indicator("ret_252d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_3y_ann")}>3Y/A{indicator("ret_3y_ann")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_5y_ann")}>5Y/A{indicator("ret_5y_ann")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("vol_3y_ann")}>Vol3Y{indicator("vol_3y_ann")}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={r.ticker} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/30">
              <td className="py-1.5 px-2 text-[#857F7A]">{i + 1}</td>
              <td className="py-1.5 px-2">
                <button
                  onClick={() => onSelect(r.ticker)}
                  className="font-mono text-[14px] text-[#0F5499] hover:underline font-bold"
                >
                  {r.ticker}
                </button>
              </td>
              <td className="py-1.5 px-2 text-[#66605C] truncate max-w-[120px]">{r.name}</td>
              <td className="py-1.5 px-2 text-[#857F7A] text-[12px]" title={`SubTheme: ${r.theme || "-"}`}>
                {r.sector || r.category}
              </td>
              <td className="py-1.5 px-2">
                <span className="text-[12px]" style={{ color: CLASS_COLORS[r.classification] || C.gray }}>
                  {r.classification}
                </span>
              </td>
              <td className="py-1.5 px-2 text-right font-mono font-bold" style={{ color: compColor(r.composite) }}>
                {r.composite.toFixed(1)}
              </td>
              {(() => {
                const d = decideAction(r);
                return (
                  <td className="py-1.5 px-2 align-top" style={{ minWidth: "320px" }}>
                    <div className="flex flex-col gap-0.5">
                      <span className="text-[13px] font-bold whitespace-nowrap" style={{ color: d.color }}>
                        {d.action}
                      </span>
                      <span className="text-[12px] text-[#857F7A] leading-snug whitespace-normal break-words">
                        {d.rationale}
                      </span>
                    </div>
                  </td>
                );
              })()}
              <td className="py-1.5 px-2 text-right font-mono text-[#33302E]">{r.tcs}</td>
              <td className="py-1.5 px-2 text-right font-mono text-[#33302E]">{r.tfs}</td>
              <td className="py-1.5 px-2 text-right font-mono text-[#33302E]">{r.rss}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: oerColor(r.oer) }}>
                {r.oer}
              </td>
              {(() => {
                const b3m = r.qvr_bullish_chg_3m;
                const beat = r.qvr_eps_beat_rate;
                const surp = r.qvr_eps_surprise_avg;
                const fhParts: string[] = [];
                if (b3m != null) fhParts.push(`Bull3m ${b3m >= 0 ? "+" : ""}${b3m.toFixed(1)}%`);
                if (beat != null) fhParts.push(`Beat ${beat.toFixed(0)}%`);
                if (surp != null) fhParts.push(`Surp ${surp >= 0 ? "+" : ""}${surp.toFixed(1)}%`);
                const tip = r.asset_type === "ETF"
                  ? "ETF (no fundamentals)"
                  : `Q ${r.qvr_q ?? "-"} | V ${r.qvr_v ?? "-"} | R ${r.qvr_r ?? "-"}`
                    + (r.qvr_n_analysts ? ` · ${r.qvr_n_analysts} analysts` : "")
                    + (fhParts.length ? ` · ${fhParts.join(" / ")}` : "");
                return (
                  <td className="py-1.5 px-2 text-right font-mono" style={{ color: qvrColor(r.qvr_score ?? 50) }}
                      title={tip}>
                    {(r.qvr_score ?? 50).toFixed(0)}
                  </td>
                );
              })()}
              <td className="py-1.5 px-2 text-right font-mono font-semibold" style={{
                color: (r.mom_age ?? 0) >= 90 ? C.red : (r.mom_age ?? 0) >= 60 ? C.orange : (r.mom_age ?? 0) >= 30 ? C.cyan : (r.mom_age ?? 0) > 0 ? C.green : C.gray,
              }}>
                {r.mom_age ?? 0}d
              </td>
              <td className="py-1.5 px-2 text-center">
                <span className="text-[12px] font-semibold" style={{ color: signalColor(r.net_signal || "") }}>
                  {r.net_signal || "-"}
                </span>
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-[#66605C] text-[12px]">
                <span className="text-[#0A7D3F]">{r.long_count ?? 0}</span>
                <span className="text-[#857F7A]">/</span>
                <span className="text-[#CC0000]">{r.short_count ?? 0}</span>
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{r.rsi?.toFixed(0)}</td>
              <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{r.trend_age}d</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_1d ?? 0) }}>{(r.ret_1d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_5d ?? 0) }}>{(r.ret_5d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_21d ?? 0) }}>{(r.ret_21d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_63d ?? 0) }}>{(r.ret_63d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_126d ?? 0) }}>{(r.ret_126d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_ytd ?? 0) }}>{r.ret_ytd != null ? `${r.ret_ytd.toFixed(1)}%` : "—"}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_252d ?? 0) }}>{(r.ret_252d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: r.ret_3y_ann == null ? C.gray : retColor(r.ret_3y_ann) }}>{r.ret_3y_ann == null ? "-" : `${r.ret_3y_ann.toFixed(1)}%`}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: r.ret_5y_ann == null ? C.gray : retColor(r.ret_5y_ann) }}>{r.ret_5y_ann == null ? "-" : `${r.ret_5y_ann.toFixed(1)}%`}</td>
              <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{r.vol_3y_ann == null ? "-" : `${r.vol_3y_ann.toFixed(1)}%`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Ticker Detail
// ---------------------------------------------------------------------------

export function TickerDetail({ ticker, onClose }: { ticker: Ticker; onClose: () => void }) {
  const axes = [
    { label: "TCS", value: ticker.tcs, desc: "Trend Continuation" },
    { label: "TFS", value: ticker.tfs, desc: "Trend Formation" },
    { label: "RSS", value: ticker.rss, desc: "Relative Strength" },
    { label: "OER", value: ticker.oer, desc: "Overextension Risk" },
    { label: "QVR", value: ticker.qvr_score ?? 50, desc: "Quality-Value-Revision" },
  ];
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-[20px] font-bold text-[#0F5499] font-mono">{ticker.ticker}</span>
          <span className="text-[16px] text-[#33302E]">{ticker.name}</span>
          <span className="text-[12px] px-2 py-0.5 rounded" style={{
            backgroundColor: (CLASS_COLORS[ticker.classification] || C.gray) + "22",
            color: CLASS_COLORS[ticker.classification] || C.gray,
          }}>
            {ticker.classification}
          </span>
        </div>
        <button onClick={onClose} className="text-[12px] text-[#857F7A] hover:text-[#33302E] px-2 py-1 rounded border border-[#E6D9CE]">
          Close
        </button>
      </div>

      {/* Classification tags */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-[#857F7A]">
        <span>Sector: <span className="text-[#33302E] font-semibold">{ticker.sector || ticker.category}</span></span>
        {ticker.theme && ticker.theme !== "-" && (
          <span>SubTheme: <span className="text-[#33302E]">{ticker.theme}</span></span>
        )}
      </div>

      {/* Score Axes */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {axes.map((a) => (
          <div key={a.label} className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-3"
               title={a.label === "QVR" ? `Q ${ticker.qvr_q ?? "-"} | V ${ticker.qvr_v ?? "-"} | R ${ticker.qvr_r ?? "-"}` : undefined}>
            <div className="text-[12px] text-[#857F7A]">{a.desc}</div>
            <div className="text-[22px] font-bold font-mono mt-1" style={{
              color: a.label === "OER" ? oerColor(a.value)
                   : a.label === "QVR" ? qvrColor(a.value)
                   : compColor(a.value),
            }}>
              {typeof a.value === "number" ? a.value.toFixed(0) : a.value}
            </div>
          </div>
        ))}
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2 text-[14px]">
        <div className="bg-[#FFFFFF] rounded p-2">
          <div className="text-[12px] text-[#857F7A]">Composite</div>
          <div className="font-mono font-bold" style={{ color: compColor(ticker.composite) }}>{ticker.composite.toFixed(1)}</div>
        </div>
        <div className="bg-[#FFFFFF] rounded p-2">
          <div className="text-[12px] text-[#857F7A]">Signal</div>
          <div className="font-semibold" style={{ color: signalColor(ticker.net_signal || "") }}>{ticker.net_signal || "-"}</div>
        </div>
        <div className="bg-[#FFFFFF] rounded p-2">
          <div className="text-[12px] text-[#857F7A]">RSI</div>
          <div className="font-mono">{ticker.rsi?.toFixed(0)}</div>
        </div>
        <div className="bg-[#FFFFFF] rounded p-2">
          <div className="text-[12px] text-[#857F7A]">Trend Age</div>
          <div className="font-mono">{ticker.trend_age}d</div>
        </div>
        <div className="bg-[#FFFFFF] rounded p-2">
          <div className="text-[12px] text-[#857F7A]">SMA50 Dist</div>
          <div className="font-mono">{ticker.sma50_dist?.toFixed(1)}%</div>
        </div>
        <div className="bg-[#FFFFFF] rounded p-2">
          <div className="text-[12px] text-[#857F7A]">Ret 1M</div>
          <div className="font-mono" style={{ color: retColor(ticker.ret_1m) }}>{ticker.ret_1m?.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Hedge Strategy Score Table
// ---------------------------------------------------------------------------

const STRATEGIES = [
  { key: "oneil",     label: "O'Neil" },
  { key: "minervini", label: "Minervini" },
  { key: "wyckoff",   label: "Wyckoff" },
  { key: "ichimoku",  label: "Ichimoku" },
  { key: "darvas",    label: "Darvas" },
  { key: "regime",    label: "Regime" },
  { key: "flow",      label: "Flow" },
  { key: "relval",    label: "RelVal" },
] as const;

function stratColor(v: number): string {
  if (v >= 70) return C.green;
  if (v >= 60) return "#0A7D3F";
  if (v >= 40) return C.gray;
  if (v >= 30) return "#CC0000";
  return C.red;
}

export function StrategyTable({ rows }: { rows: Ticker[] }) {
  const accessors = useMemo(() => {
    const acc: Record<string, (r: Ticker) => any> = {
      ticker: (r) => r.ticker,
      name: (r) => r.name,
      combined_long: (r) => r.combined_long,
      combined_short: (r) => r.combined_short,
      signal: (r) => r.net_signal || "",
    };
    for (const s of STRATEGIES) {
      acc[`${s.key}_long`] = (r) => (r as any)[`${s.key}_long`] ?? 0;
      acc[`${s.key}_short`] = (r) => (r as any)[`${s.key}_short`] ?? 0;
    }
    return acc;
  }, []);
  const { sorted, onSort, indicator } = useSort(rows, accessors);

  if (!rows.length) return <p className="text-[12px] text-[#857F7A]">No data</p>;
  const headerCls = "cursor-pointer select-none hover:text-[#33302E]";
  return (
    <div className="overflow-auto border border-[#E6D9CE] rounded" style={{ maxHeight: "600px" }}>
      <table className="w-full text-[14px] border-collapse">
        <thead className="sticky top-0 z-10 bg-[#F2E5D7]">
          <tr className="border-b border-[#E6D9CE]">
            <th className="py-1.5 px-2 text-left text-[#857F7A]">#</th>
            <th className={`py-1.5 px-2 text-left text-[#857F7A] ${headerCls}`} onClick={() => onSort("ticker")}>Ticker{indicator("ticker")}</th>
            <th className={`py-1.5 px-2 text-left text-[#857F7A] ${headerCls}`} onClick={() => onSort("name")}>Name{indicator("name")}</th>
            {STRATEGIES.map((s) => (
              <th key={s.key} colSpan={2} className="py-1.5 px-1 text-center text-[#857F7A] border-l border-[#E6D9CE]/50">
                {s.label}
              </th>
            ))}
            <th className="py-1.5 px-1 text-center text-[#857F7A] border-l border-[#E6D9CE]/50" colSpan={2}>Combined</th>
            <th className={`py-1.5 px-2 text-center text-[#857F7A] border-l border-[#E6D9CE]/50 ${headerCls}`} onClick={() => onSort("signal")}>
              Signal{indicator("signal")}
            </th>
          </tr>
          <tr className="border-b border-[#E6D9CE]">
            <th colSpan={3} />
            {STRATEGIES.map((s) => (
              <React.Fragment key={s.key}>
                <th className={`py-1 px-1 text-[11px] text-[#0A7D3F] border-l border-[#E6D9CE]/50 ${headerCls}`} onClick={() => onSort(`${s.key}_long`)}>L{indicator(`${s.key}_long`)}</th>
                <th className={`py-1 px-1 text-[11px] text-[#CC0000] ${headerCls}`} onClick={() => onSort(`${s.key}_short`)}>S{indicator(`${s.key}_short`)}</th>
              </React.Fragment>
            ))}
            <th className={`py-1 px-1 text-[11px] text-[#0A7D3F] border-l border-[#E6D9CE]/50 ${headerCls}`} onClick={() => onSort("combined_long")}>L{indicator("combined_long")}</th>
            <th className={`py-1 px-1 text-[11px] text-[#CC0000] ${headerCls}`} onClick={() => onSort("combined_short")}>S{indicator("combined_short")}</th>
            <th className="border-l border-[#E6D9CE]/50" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={r.ticker} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/30">
              <td className="py-1 px-2 text-[#857F7A]">{i + 1}</td>
              <td className="py-1 px-2 font-mono text-[#0F5499] font-bold text-[12px]">{r.ticker}</td>
              <td className="py-1 px-2 text-[#66605C] truncate max-w-[100px] text-[12px]">{r.name}</td>
              {STRATEGIES.map((s) => {
                const lv = (r as any)[`${s.key}_long`] ?? 0;
                const sv = (r as any)[`${s.key}_short`] ?? 0;
                return (
                  <React.Fragment key={s.key}>
                    <td className="py-1 px-1 text-center font-mono text-[12px] border-l border-[#E6D9CE]/50"
                        style={{ color: stratColor(lv) }}>{Math.round(lv)}</td>
                    <td className="py-1 px-1 text-center font-mono text-[12px]"
                        style={{ color: stratColor(100 - sv) }}>{Math.round(sv)}</td>
                  </React.Fragment>
                );
              })}
              <td className="py-1 px-1 text-center font-mono font-bold text-[12px] border-l border-[#E6D9CE]/50"
                  style={{ color: stratColor(r.combined_long) }}>{Math.round(r.combined_long)}</td>
              <td className="py-1 px-1 text-center font-mono font-bold text-[12px]"
                  style={{ color: stratColor(100 - r.combined_short) }}>{Math.round(r.combined_short)}</td>
              <td className="py-1 px-2 text-center font-semibold text-[12px] border-l border-[#E6D9CE]/50"
                  style={{ color: signalColor(r.net_signal || "") }}>{r.net_signal || "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Export
// ---------------------------------------------------------------------------

export function MomentumTab({ filters, totalUniverse, mlMode = false }:
  { filters: FilterParams; totalUniverse?: number; mlMode?: boolean }) {
  const [allData, setAllData] = useState<Ticker[]>([]);
  const [kpis, setKpis] = useState<any>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const tableFetch = mlMode ? fetchTableML(filters) : fetchTable(filters);
    Promise.all([
      tableFetch,
      fetchOverview(filters),
    ]).then(([tableRes, overviewRes]) => {
      setAllData(tableRes.data || []);
      setKpis(overviewRes.kpis || {});
    }).finally(() => setLoading(false));
  }, [filters, mlMode]);

  // Only eligible (momentum confirmed)
  const eligible = useMemo(
    () => allData.filter((t) => t.eligible).sort((a, b) => b.composite - a.composite),
    [allData]
  );
  const etf = useMemo(() => eligible.filter((t) => t.asset_type === "ETF"), [eligible]);
  const stock = useMemo(() => eligible.filter((t) => t.asset_type === "Stock"), [eligible]);

  // Classification breakdown for eligible
  const classDist = useMemo(() => {
    const map: Record<string, number> = {};
    for (const t of eligible) {
      map[t.classification] = (map[t.classification] || 0) + 1;
    }
    return Object.entries(map).sort((a, b) => b[1] - a[1]);
  }, [eligible]);

  const selectedTicker = useMemo(
    () => (selected ? allData.find((t) => t.ticker === selected) ?? null : null),
    [selected, allData]
  );

  if (loading) return <div className="text-[#857F7A] p-8">Loading momentum data...</div>;

  const avgComp = eligible.length
    ? eligible.reduce((s, t) => s + t.composite, 0) / eligible.length
    : 0;

  return (
    <div className="space-y-5">
      {/* ── KPIs ── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Eligible" value={eligible.length} sub={`/ ${totalUniverse || allData.length} universe`} />
        <MetricCard label="ETF" value={etf.length} sub="momentum confirmed" />
        <MetricCard label="Stock" value={stock.length} sub="momentum confirmed" />
        <MetricCard label="Avg Composite" value={avgComp.toFixed(1)} sub="eligible avg" />
        <MetricCard
          label="Top Class"
          value={classDist[0]?.[0]?.replace(/^.\s/, "") || "-"}
          sub={`${classDist[0]?.[1] || 0} tickers`}
        />
      </div>

      {/* ── Classification Breakdown (inline) ── */}
      <div className="flex gap-2 flex-wrap">
        {classDist.map(([cls, count]) => (
          <span
            key={cls}
            className="text-[12px] px-2 py-1 rounded border border-[#E6D9CE]"
            style={{ color: CLASS_COLORS[cls] || C.gray, borderColor: (CLASS_COLORS[cls] || C.gray) + "44" }}
          >
            {cls} <span className="font-mono font-bold ml-1">{count}</span>
          </span>
        ))}
      </div>

      {/* ── All Eligible Table ── */}
      <CompositeFormulaGuide />
      <ColumnDefinitions />
      <MomentumTable rows={eligible} onSelect={setSelected} />

      {/* ── ETF + Stock side by side ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div>
          <h3 className="text-[14px] font-semibold text-[#0F5499] uppercase tracking-wide mb-2">
            ETF ({etf.length})
          </h3>
          <MomentumTable rows={etf} onSelect={setSelected} />
        </div>
        <div>
          <h3 className="text-[14px] font-semibold text-[#0A7D3F] uppercase tracking-wide mb-2">
            Stock ({stock.length})
          </h3>
          <MomentumTable rows={stock} onSelect={setSelected} />
        </div>
      </div>

      {/* ── Selected Ticker Detail ── */}
      {selectedTicker && (
        <Section title={`${selectedTicker.ticker} — ${selectedTicker.name}`} defaultOpen badge={selectedTicker.classification}>
          <TickerDetail ticker={selectedTicker} onClose={() => setSelected(null)} />
        </Section>
      )}

      {/* ── Hedge Strategy Scores ── */}
      <Section title="Hedge Strategy Scores" badge={`${eligible.length} tickers × 8 strategies`}>
        <p className="text-[13px] text-[#857F7A] mb-3">
          8개 hedge strategy별 Long/Short 점수 (0-100). 60 이상이면 해당 방향 시그널로 카운트.
        </p>
        <StrategyTable rows={eligible} />
      </Section>
    </div>
  );
}
