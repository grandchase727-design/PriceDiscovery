import { useEffect, useState, useMemo } from "react";
import { fetchTable, type FilterParams } from "../../api/client";
import { DataTable } from "../shared/DataTable";
import { CLASS_COLORS, C } from "../../styles/theme";
import type { ColumnDef } from "@tanstack/react-table";

/* ─── Column helpers ─── */
function numCol(key: string, header: string, fmt = 1): ColumnDef<any, any> {
  return {
    accessorKey: key, header,
    cell: (info) => {
      const v = info.getValue();
      return v != null ? Number(v).toFixed(fmt) : "-";
    },
  };
}

function pctCol(key: string, header: string): ColumnDef<any, any> {
  return {
    accessorKey: key, header,
    cell: (info) => {
      const v = Number(info.getValue());
      const color = v > 0 ? "text-green-400" : v < 0 ? "text-red-400" : "";
      return <span className={color}>{v.toFixed(2)}%</span>;
    },
  };
}

function signalCol(key: string, header: string, isLong: boolean): ColumnDef<any, any> {
  return {
    accessorKey: key, header,
    cell: (info) => {
      const v = Number(info.getValue());
      if (v == null || isNaN(v)) return <span className="text-gray-600">-</span>;
      const color = isLong
        ? (v >= 70 ? "text-green-400 font-bold" : v >= 50 ? "text-green-400/70" : v >= 30 ? "text-gray-400" : "text-gray-600")
        : (v >= 70 ? "text-red-400 font-bold" : v >= 50 ? "text-red-400/70" : v >= 30 ? "text-gray-400" : "text-gray-600");
      return <span className={`${color} text-[10px]`}>{v.toFixed(0)}</span>;
    },
  };
}

/* ─── Master Table columns ─── */
/* Helper: net_signal cell */
function netSignalCell(info: any) {
  const sig = String(info.getValue() || "");
  const colorMap: Record<string, string> = {
    STRONG_LONG: C.green, LONG: "#86efac",
    STRONG_SHORT: C.red, SHORT: "#fca5a5",
    NEUTRAL: C.gray,
  };
  return <span style={{ color: colorMap[sig] || C.gray }} className="text-[10px] font-bold">{sig || "-"}</span>;
}

/* Helper: classification cell */
function classCell(info: any) {
  const cls = String(info.getValue());
  const color = CLASS_COLORS[cls] || "#6b7280";
  return <span style={{ color }} className="text-[10px] font-medium">{cls.slice(0, 16)}</span>;
}

/* Helper: event risk cell */
function riskCell(info: any) {
  const v = info.getValue();
  const reasons = String(info.row.original?.event_reasons || "");
  if (v) return <span className="text-yellow-400 font-semibold text-[9px]" title={reasons}>⚡ EVENT</span>;
  return <span className="text-gray-700 text-[9px]">—</span>;
}

const masterColumns: ColumnDef<any, any>[] = [
  // Identity
  { accessorKey: "ticker", header: "Ticker", cell: (i) => <span className="font-mono font-bold">{i.getValue()}</span> },
  { accessorKey: "name", header: "Name", cell: (i) => <span className="text-gray-400 truncate max-w-[160px] block">{String(i.getValue()).slice(0, 25)}</span> },
  { accessorKey: "sector", header: "Sector", cell: (i) => <span className="text-cyan-400/70 text-[10px]">{String(i.getValue())}</span> },
  { accessorKey: "category", header: "Category", cell: (i) => <span className="text-gray-500 text-[10px]">{String(i.getValue()).replace("STK_", "")}</span> },
  { accessorKey: "mktcap_B", header: "MCap$B", cell: (i) => {
    const v = i.getValue(); if (v == null || Number(v) === 0) return <span className="text-gray-600">-</span>;
    const n = Number(v); return <span className="text-gray-400 text-[10px]">{n >= 1 ? n.toFixed(1) : n.toFixed(2)}</span>;
  }},
  { accessorKey: "classification", header: "Class", cell: classCell },
  // Combined signal
  { accessorKey: "net_signal", header: "Signal", cell: netSignalCell },
  numCol("long_count", "L#", 0), numCol("short_count", "S#", 0),
  numCol("conviction", "Conv", 1),
  // 8 strategy Long scores
  signalCol("oneil_long", "O'Neil", true),
  signalCol("minervini_long", "Minerv", true),
  signalCol("wyckoff_long", "Wyck", true),
  signalCol("ichimoku_long", "Ichi", true),
  signalCol("darvas_long", "Darvas", true),
  signalCol("regime_long", "Regime", true),
  signalCol("flow_long", "Flow", true),
  signalCol("relval_long", "RV", true),
  // Core scores
  numCol("composite", "Comp"),
  { accessorKey: "event_flag", header: "Risk", cell: riskCell },
  numCol("structural_q", "SQ", 0),
  { accessorKey: "alpha_potential", header: "APS", cell: (i) => {
    const v = Number(i.getValue() ?? 0);
    const c = v >= 70 ? "text-emerald-400 font-bold" : v >= 50 ? "text-emerald-400/70" : v >= 30 ? "text-gray-400" : "text-gray-600";
    return <span className={`${c} text-[10px]`}>{v.toFixed(0)}</span>;
  }},
  numCol("tcs", "TCS", 0), numCol("tfs", "TFS", 0), numCol("rss", "RSS", 0), numCol("oer", "OER", 0),
  numCol("rsi", "RSI"), numCol("trend_age", "Age", 0),
  // Eligibility
  { accessorKey: "eligible", header: "Elg", cell: (i) => i.getValue() ? "✅" : "" },
  numCol("val_prob", "Val%"),
  // Returns
  pctCol("ret_1w", "1W"), pctCol("ret_1m", "1M"), pctCol("ret_3m", "3M"),
];

/* ─── Strategy table config ─── */
interface StrategyDef {
  key: string;
  name: string;
  nameKr: string;
  desc: string;
  color: string;
  icon: string;
  longKey: string;
  shortKey: string;
}

const STRATEGIES: StrategyDef[] = [
  {
    key: "oneil", name: "O'Neil CANSLIM", nameKr: "윌리엄 오닐 CANSLIM",
    desc: "Pivot proximity(25) + Volume surge(20) + RS rating(20) + MA structure(20) + Base breakout quality(15)",
    color: C.cyan, icon: "C", longKey: "oneil_long", shortKey: "oneil_short",
  },
  {
    key: "minervini", name: "Minervini SEPA", nameKr: "미너비니 SEPA",
    desc: "Stage 2 template(35) + 52W range position(25) + RS rating(20) + VCP tightness(20)",
    color: C.green, icon: "M", longKey: "minervini_long", shortKey: "minervini_short",
  },
  {
    key: "wyckoff", name: "Wyckoff Accum/Dist", nameKr: "와이코프 수급분석",
    desc: "OBV trend(25) + Distribution days(20) + Close position(20) + Volume on advance(20) + Spring/markup(15)",
    color: C.blue, icon: "W", longKey: "wyckoff_long", shortKey: "wyckoff_short",
  },
  {
    key: "ichimoku", name: "Ichimoku Kinko Hyo", nameKr: "일목균형표",
    desc: "Cloud position(25) + TK cross(20) + Cloud color(20) + Chikou span(20) + Momentum(15)",
    color: C.purple, icon: "I", longKey: "ichimoku_long", shortKey: "ichimoku_short",
  },
  {
    key: "darvas", name: "Darvas Box", nameKr: "다바스 박스 돌파",
    desc: "Box formation(25) + Box tightness(25) + Breakout confirmation(25) + Trend context(25)",
    color: C.orange, icon: "D", longKey: "darvas_long", shortKey: "darvas_short",
  },
  {
    key: "regime", name: "Regime Adaptive", nameKr: "레짐 적응형",
    desc: "Market regime(SPY) determines weighting: Risk-On → momentum/trend, Risk-Off → defensive/mean-reversion",
    color: C.yellow, icon: "R", longKey: "regime_long", shortKey: "regime_short",
  },
  {
    key: "flow", name: "Institutional Flow", nameKr: "기관 수급 분석",
    desc: "MFI(25) + Distribution days(20) + Quiet accumulation/OBV(20) + Close position(20) + Volume climax(15)",
    color: "#22d3ee", icon: "F", longKey: "flow_long", shortKey: "flow_short",
  },
  {
    key: "relval", name: "Relative Value", nameKr: "섹터 상대가치",
    desc: "Intra-sector z-score(55) + Structural support(25) + Reversal signals(20). Mean-reversion within peers",
    color: "#a78bfa", icon: "V", longKey: "relval_long", shortKey: "relval_short",
  },
];

function makeStrategyColumns(strat: StrategyDef): ColumnDef<any, any>[] {
  return [
    { accessorKey: "ticker", header: "Ticker", cell: (i) => <span className="font-mono font-bold text-cyan-400">{i.getValue()}</span> },
    { accessorKey: "name", header: "Name", cell: (i) => <span className="text-gray-400 truncate max-w-[180px] block text-[11px]">{String(i.getValue()).slice(0, 30)}</span> },
    { accessorKey: "sector", header: "Sector", cell: (i) => <span className="text-gray-500 text-[10px]">{String(i.getValue())}</span> },
    { accessorKey: "category", header: "Category", cell: (i) => <span className="text-gray-600 text-[10px]">{String(i.getValue()).replace("STK_", "")}</span> },
    { accessorKey: "mktcap_B", header: "MCap$B", cell: (i) => {
      const v = i.getValue(); if (v == null || Number(v) === 0) return <span className="text-gray-600">-</span>;
      const n = Number(v); return <span className="text-gray-400 text-[10px]">{n >= 1 ? n.toFixed(1) : n.toFixed(2)}</span>;
    }},
    numCol("composite", "Comp"),
    signalCol(strat.longKey, "Long", true),
    signalCol(strat.shortKey, "Short", false),
    {
      accessorKey: "_signal", header: "Signal",
      cell: (info) => {
        const row = info.row.original;
        const ls = Number(row[strat.longKey] || 0);
        const ss = Number(row[strat.shortKey] || 0);
        if (ls >= 70) return <span className="text-green-400 font-bold text-[10px]">STRONG LONG</span>;
        if (ls >= 50 && ls > ss + 15) return <span className="text-green-400/80 text-[10px]">LONG</span>;
        if (ss >= 70) return <span className="text-red-400 font-bold text-[10px]">STRONG SHORT</span>;
        if (ss >= 50 && ss > ls + 15) return <span className="text-red-400/80 text-[10px]">SHORT</span>;
        return <span className="text-gray-600 text-[10px]">NEUTRAL</span>;
      },
    },
    {
      accessorKey: "classification", header: "Class",
      cell: (info) => {
        const cls = String(info.getValue());
        const color = CLASS_COLORS[cls] || "#6b7280";
        return <span style={{ color }} className="text-[10px]">{cls.slice(0, 16)}</span>;
      },
    },
    numCol("rss", "RSS", 0),
    numCol("rsi", "RSI"),
    pctCol("ret_1m", "1M"),
  ];
}

/* Combined signal table columns */
const combinedColumns: ColumnDef<any, any>[] = [
  { accessorKey: "ticker", header: "Ticker", cell: (i) => <span className="font-mono font-bold text-cyan-400">{i.getValue()}</span> },
  { accessorKey: "name", header: "Name", cell: (i) => <span className="text-gray-400 truncate max-w-[180px] block text-[11px]">{String(i.getValue()).slice(0, 30)}</span> },
  { accessorKey: "sector", header: "Sector", cell: (i) => <span className="text-gray-500 text-[10px]">{String(i.getValue())}</span> },
  numCol("composite", "Comp"),
  {
    accessorKey: "net_signal", header: "Signal",
    cell: (info) => {
      const sig = String(info.getValue() || "");
      const colorMap: Record<string, string> = {
        STRONG_LONG: C.green, LONG: "#86efac",
        STRONG_SHORT: C.red, SHORT: "#fca5a5",
        NEUTRAL: C.gray,
      };
      return <span style={{ color: colorMap[sig] || C.gray }} className="font-bold text-[11px]">{sig || "-"}</span>;
    },
  },
  numCol("combined_long", "L Score", 1),
  numCol("combined_short", "S Score", 1),
  numCol("long_count", "L#", 0),
  numCol("short_count", "S#", 0),
  numCol("conviction", "Conv", 1),
  signalCol("oneil_long", "O'Neil", true),
  signalCol("minervini_long", "Minerv", true),
  signalCol("wyckoff_long", "Wyckoff", true),
  signalCol("ichimoku_long", "Ichim", true),
  signalCol("darvas_long", "Darvas", true),
  signalCol("regime_long", "Regime", true),
  signalCol("flow_long", "Flow", true),
  signalCol("relval_long", "RelVal", true),
  {
    accessorKey: "classification", header: "Class",
    cell: (info) => {
      const cls = String(info.getValue());
      const color = CLASS_COLORS[cls] || "#6b7280";
      return <span style={{ color }} className="text-[10px]">{cls.slice(0, 16)}</span>;
    },
  },
  pctCol("ret_1m", "1M"),
];

/* ─── Unified sort: _long_rank (combined_long×0.40 + composite×0.35 + APS×0.25) ─── */
const SIGNAL_TIER: Record<string, number> = {
  STRONG_LONG: 5, LONG: 4, NEUTRAL: 3, SHORT: 2, STRONG_SHORT: 1,
};

function computeLongRank(d: any): number {
  const cl = Number(d.combined_long ?? 0);
  const comp = Number(d.composite ?? 0);
  const aps = Number(d.alpha_potential ?? 0);
  const discount = d.event_flag ? 0.7 : 1.0;
  return (cl * 0.40 + comp * 0.35 + aps * 0.25) * discount;
}

function signalTieredSort(a: any, b: any): number {
  return computeLongRank(b) - computeLongRank(a);             // _long_rank desc (same as Top 10)
}

/* ─── Collapsible Section ─── */
function Section({ title, children, defaultOpen = false, badge, color }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean; badge?: string; color?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-3 text-left text-sm font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between items-center"
        onClick={() => setOpen(!open)}>
        <span className="flex items-center gap-2">
          {color && <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />}
          {title}
          {badge && <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">{badge}</span>}
        </span>
        <span className="text-gray-500 text-xs">{open ? "▼" : "▶"}</span>
      </button>
      {open && <div className="p-4 bg-[#0d1117] space-y-4">{children}</div>}
    </div>
  );
}

/* ─── Column definitions reference panel ─── */
const COLUMN_DEFS: { col: string; desc: string }[] = [
  { col: "Ticker", desc: "종목 코드 (ETF 또는 개별주식)" },
  { col: "Name", desc: "종목 풀네임" },
  { col: "Sector", desc: "글로벌 GICS 기반 섹터" },
  { col: "Comp", desc: "Composite Score (0-100): 0.35×TCS + 0.30×TFS + 0.35×RSS" },
  { col: "TCS/TFS/OER/RSS", desc: "3축 시그널 + 상대강도 점수" },
  { col: "Class", desc: "Dual-timeframe 3×3 분류 + Override" },
  { col: "Signal", desc: "8개 전략 종합 Long/Short 판단: STRONG_LONG / LONG / NEUTRAL / SHORT / STRONG_SHORT" },
  { col: "L# / S#", desc: "Long/Short 시그널 발생 전략 수 (8개 중)" },
  { col: "Conv", desc: "Conviction: |combined_long - combined_short|. 높을수록 방향성 확실" },
  { col: "O'Neil L/S", desc: "CANSLIM 매수/매도 시그널 (0-100)" },
  { col: "Minervini", desc: "SEPA Stage 2 template + VCP (0-100)" },
  { col: "Wyckoff", desc: "수급(OBV/분배일/종가위치) 기반 기관 매집/분배 탐지 (0-100)" },
  { col: "Ichimoku", desc: "일목균형표 5요소: 구름/전환선/기준선/치코스팬/색상 (0-100)" },
  { col: "Darvas", desc: "박스 형성기간+타이트니스+돌파+추세 (0-100)" },
  { col: "Regime", desc: "시장 레짐(Risk-On/Off/Transition) 연동 적응형 점수 (0-100)" },
  { col: "Flow", desc: "MFI+분배일+OBV 괴리+종가위치 기반 스마트머니 탐지 (0-100)" },
  { col: "RelVal", desc: "동일 섹터 내 z-score 기반 상대가치. 저평가(Long)/고평가(Short) (0-100)" },
  { col: "APS", desc: "Alpha Potential Score (0-100): 매집 품질(25) + 변동성 셋업(20) + 모멘텀 구조(25) + 상대강도(15) + 돌파 근접(15). 분류 무관 상승 잠재력" },
];

const CLASS_DEFS: { cls: string; color: string; direction: string; condition: string; desc: string; rank: number; eligible: boolean }[] = [
  { cls: "CONTINUATION", color: "#22c55e", direction: "Short UP + Long UP", condition: "SMA20 괴리 > +0.5% & SMA50 괴리 > +1%", desc: "단기·장기 모두 상승. 가장 강한 추세 상태", rank: 3, eligible: true },
  { cls: "RECOVERY", color: "#3b82f6", direction: "Short UP + Long FLAT", condition: "SMA20 괴리 > +0.5%, SMA50 괴리 ±1%", desc: "장기 미확립, 단기 반등 시작", rank: 2, eligible: true },
  { cls: "COUNTER_RALLY", color: "#8b5cf6", direction: "Short UP + Long DOWN", condition: "SMA20 괴리 > +0.5%, SMA50 괴리 < -1%", desc: "하락 추세 중 기술적 반등", rank: 0, eligible: false },
  { cls: "CONSOLIDATION", color: "#f59e0b", direction: "Short FLAT + Long UP", condition: "SMA20 괴리 ±0.5%, SMA50 괴리 > +1%", desc: "장기 상승 내 단기 횡보", rank: 2, eligible: true },
  { cls: "NEUTRAL", color: "#f97316", direction: "Short FLAT + Long FLAT", condition: "SMA20·SMA50 모두 버퍼 이내", desc: "방향 없음. 박스권", rank: 1, eligible: true },
  { cls: "FADING", color: "#a87c5a", direction: "Short FLAT + Long DOWN", condition: "SMA20 ±0.5%, SMA50 < -1%", desc: "장기 하락 + 반등 없음", rank: 0, eligible: false },
  { cls: "PULLBACK", color: "#ff8c00", direction: "Short DOWN + Long UP", condition: "SMA20 < -0.5%, SMA50 > +1%", desc: "건강한 추세 내 단기 조정", rank: 2, eligible: true },
  { cls: "WEAKENING", color: "#dc2626", direction: "Short DOWN + Long FLAT", condition: "SMA20 < -0.5%, SMA50 ±1%", desc: "지지력 약화 + 단기 하락", rank: 1, eligible: true },
  { cls: "DOWNTREND", color: "#ef4444", direction: "Short DOWN + Long DOWN", condition: "SMA20 < -0.5% & SMA50 < -1%", desc: "양 타임프레임 하락", rank: 0, eligible: false },
];

const OVERRIDE_DEFS: { cls: string; color: string; priority: string; condition: string; desc: string; eligible: boolean }[] = [
  { cls: "OVEREXTENDED", color: "#f59e0b", priority: "1st", condition: "OER ≥ 60 & bullish base", desc: "과열 상태. 조정 대기 권장", eligible: true },
  { cls: "FORMATION", color: "#60a5fa", priority: "2nd", condition: "TFS_short ≥ 50 & trend_age_short ≤ 5", desc: "SMA20 막 돌파. 최적 매수 타이밍", eligible: true },
  { cls: "CYCLE_PEAK", color: "#dc143c", priority: "3rd", condition: "reversal_pctile ≥ 85", desc: "구조적 사이클 고점", eligible: false },
  { cls: "EXHAUSTING", color: "#a87c5a", priority: "4th", condition: "trend_age > 60 & 모멘텀 급감", desc: "추세 피로. 이익 실현 검토", eligible: false },
];

function ColumnDefPanel() {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden mb-4">
      <button className="w-full px-4 py-2 text-left text-xs font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between"
        onClick={() => setOpen(!open)}>
        <span>Column & Strategy Definitions</span>
        <span className="text-gray-500">{open ? "▼" : "▶"}</span>
      </button>
      {open && (
        <div className="p-3 bg-[#0d1117] space-y-4">
          <table className="w-full text-xs border-collapse">
            <thead><tr className="border-b border-gray-700">
              <th className="text-left py-1 px-2 text-gray-500 w-28">Column</th>
              <th className="text-left py-1 px-2 text-gray-500">Description</th>
            </tr></thead>
            <tbody>
              {COLUMN_DEFS.map((d) => (
                <tr key={d.col} className="border-b border-gray-800/50">
                  <td className="py-1 px-2 font-mono text-cyan-400 whitespace-nowrap">{d.col}</td>
                  <td className="py-1 px-2 text-gray-400">{d.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ClassDefPanel() {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden mb-4">
      <button className="w-full px-4 py-2 text-left text-xs font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between"
        onClick={() => setOpen(!open)}>
        <span>Classification Definitions</span>
        <span className="text-gray-500">{open ? "▼" : "▶"}</span>
      </button>
      {open && (
        <div className="p-3 bg-[#0d1117] space-y-4">
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b border-gray-700">
              <th className="text-left py-1 px-2 text-gray-500 w-28">Class</th>
              <th className="text-left py-1 px-2 text-gray-500 w-36">Direction</th>
              <th className="text-left py-1 px-2 text-gray-500">Condition</th>
              <th className="text-left py-1 px-2 text-gray-500">Description</th>
              <th className="text-center py-1 px-2 text-gray-500 w-10">Elg</th>
            </tr></thead>
            <tbody>
              {CLASS_DEFS.map((d) => (
                <tr key={d.cls} className="border-b border-gray-800/50">
                  <td className="py-1 px-2 font-medium" style={{ color: d.color }}>{d.cls}</td>
                  <td className="py-1 px-2 text-gray-500">{d.direction}</td>
                  <td className="py-1 px-2 text-gray-400">{d.condition}</td>
                  <td className="py-1 px-2 text-gray-400">{d.desc}</td>
                  <td className="py-1 px-2 text-center">{d.eligible ? <span className="text-green-400">✅</span> : <span className="text-red-400">✗</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h4 className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide mb-2">Override Classifications</h4>
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b border-gray-700">
              <th className="text-left py-1 px-2 text-gray-500 w-28">Class</th>
              <th className="text-left py-1 px-2 text-gray-500 w-10">순위</th>
              <th className="text-left py-1 px-2 text-gray-500">Condition</th>
              <th className="text-left py-1 px-2 text-gray-500">Description</th>
              <th className="text-center py-1 px-2 text-gray-500 w-10">Elg</th>
            </tr></thead>
            <tbody>
              {OVERRIDE_DEFS.map((d) => (
                <tr key={d.cls} className="border-b border-gray-800/50">
                  <td className="py-1 px-2 font-medium" style={{ color: d.color }}>{d.cls}</td>
                  <td className="py-1 px-2 text-gray-500">{d.priority}</td>
                  <td className="py-1 px-2 text-gray-400">{d.condition}</td>
                  <td className="py-1 px-2 text-gray-400">{d.desc}</td>
                  <td className="py-1 px-2 text-center">{d.eligible ? <span className="text-green-400">✅</span> : <span className="text-red-400">✗</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

/* ─── Strategy Scoring Logic Panel ─── */
const STRATEGY_LOGIC: { name: string; nameKr: string; color: string; components: { name: string; long: string; short: string; pts: string }[] }[] = [
  {
    name: "O'Neil CANSLIM", nameKr: "윌리엄 오닐 CANSLIM", color: C.cyan,
    components: [
      { name: "Pivot / Support", pts: "25", long: "52주 고점 거리: 2%이내(25) / 5%(18) / 10%(8)", short: "SMA50+200 이탈(25) / SMA50만(15) / SMA20(5)" },
      { name: "Volume", pts: "20", long: "vol_ratio 2.0x+(20) / 1.5x+(15) / 1.2x+(8)", short: "하락+vol 2.0x+(20) / 1.5x+(15) / 반등에 vol 미미(5)" },
      { name: "RS Rating", pts: "20", long: "RSS ≥85(20) / ≥75(15) / ≥60(8)", short: "RSS ≤15(20) / ≤25(15) / ≤40(8)" },
      { name: "MA Structure", pts: "20", long: "SMA50↑(6) + SMA200↑(5) + Golden Cross(5) + slope(4)", short: "Death Cross(6) + SMA50↓(6) + SMA200↓(4) + SMA20↓(4)" },
      { name: "Base / Trend", pts: "15", long: "VCR<0.6+돌파(15) / VCR<0.8+돌파(12) / 돌파만(6)", short: "range_pct<15(10) / <30(5) + 지속하락(5)" },
    ],
  },
  {
    name: "Minervini SEPA", nameKr: "미너비니 SEPA", color: C.green,
    components: [
      { name: "Stage 2/4 Template", pts: "35", long: "SMA50/150/200 위(각7) + SMA150↑(7) + SMA200↑(7)", short: "SMA200↓(10) + SMA150↓(8) + SMA50↓(7) + slopes↓(10)" },
      { name: "52W Range", pts: "25", long: "range_pct ≥75(25) / ≥50(18) / ≥25(8)", short: "range_pct <15(25) / <30(18) / <50(8)" },
      { name: "RS Rating", pts: "20", long: "RSS ≥80(20) / ≥70(15) / ≥60(8)", short: "RSS ≤15(20) / ≤25(15) / ≤40(8)" },
      { name: "VCP / Vol Decline", pts: "20", long: "VCR<0.5+돌파(20) / <0.7+돌파(15) / <0.8(5)", short: "하락+vol 2.0x+(20) / 1.5x+(15) / 1.2x+(8)" },
    ],
  },
  {
    name: "Wyckoff Accum/Dist", nameKr: "와이코프 수급분석", color: C.blue,
    components: [
      { name: "OBV Trend", pts: "25", long: "OBV 20일 slope >2.0(25) / >1.0(18) / >0.3(10)", short: "OBV slope <-2.0(25) / <-1.0(18) / <-0.3(10)" },
      { name: "Distribution Days", pts: "20", long: "25일 중 하락+고vol일 ≤1(20) / ≤3(15) / ≤5(8)", short: "≥8일(20) / ≥6(15) / ≥4(8)" },
      { name: "Close Position", pts: "20", long: "캔들 내 종가위치 ≥0.75(20) / ≥0.6(15)", short: "종가위치 ≤0.25(20) / ≤0.35(15)" },
      { name: "Vol on Move", pts: "20", long: "상승+vol 1.8x+(20)=SOS", short: "하락+vol 1.8x+(20)=SOW" },
      { name: "Spring / Upthrust", pts: "15", long: "SMA50 근처 반등=Spring(15)", short: "고점 근처 이탈=Upthrust(15)" },
    ],
  },
  {
    name: "Ichimoku Kinko Hyo", nameKr: "일목균형표", color: C.purple,
    components: [
      { name: "Cloud Position", pts: "25", long: "가격 > 구름 상단", short: "가격 < 구름 하단" },
      { name: "Tenkan-Kijun", pts: "20", long: "전환선(9) > 기준선(26) = Bull cross", short: "전환선 < 기준선 = Bear cross" },
      { name: "Cloud Color", pts: "20", long: "Senkou A > Senkou B = 녹색 구름", short: "Senkou A < Senkou B = 적색 구름" },
      { name: "Chikou Span", pts: "20", long: "현재가 > 26일 전 가격", short: "현재가 < 26일 전 가격" },
      { name: "Momentum", pts: "15", long: "ret_21d>2%(10) + sma50_slope↑(5)", short: "ret_21d<-2%(10) + sma50_slope↓(5)" },
    ],
  },
  {
    name: "Darvas Box", nameKr: "다바스 박스 돌파", color: C.orange,
    components: [
      { name: "Box Formation", pts: "25", long: "40일 내 고점 후 횡보 ≥20일(25) / ≥10(18)", short: "박스 내 하락: ret_5d<-2(30) / <0(15)" },
      { name: "Box Tightness", pts: "25", long: "box_range <5%(25) / <8%(20) / <12%(14)", short: "— (Short는 MA 하방 중심)" },
      { name: "Breakout + Vol", pts: "25", long: "박스 상단 돌파 + vol 1.5x+(25)", short: "하락+vol 1.5x+(25) / 1.2x+(15)" },
      { name: "Trend Context", pts: "25", long: "SMA50↑(10) + SMA200↑(8) + slope(7)", short: "SMA50↓(12) + SMA200↓(13) + range(20)" },
    ],
  },
  {
    name: "Regime Adaptive", nameKr: "레짐 적응형", color: C.yellow,
    components: [
      { name: "Base (RSS)", pts: "40", long: "RSS 백분위 × 0.4", short: "(100 - RSS) × 0.4" },
      { name: "Risk-On 추가", pts: "60", long: "SMA50↑(15) + GC(10) + slope(10) + breakout(10) + vol(15)", short: "최약체만: SMA200↓(10) + range<20(15) + RS<15(15)" },
      { name: "Risk-Off 추가", pts: "60", long: "저vol(15) + RSI<30 과매도(20) + SMA200 지지(15)", short: "SMA50/200↓(25) + 고vol(15) + 분배일(10)" },
      { name: "Transition 추가", pts: "60", long: "SMA50/200(20) + VCR<0.8(10) + slope(10) + comp(10)", short: "SMA50/200↓(24) + ret<-3%(8) + slope(8) + 분배일(10)" },
    ],
  },
  {
    name: "Institutional Flow", nameKr: "기관 수급 분석", color: "#22d3ee",
    components: [
      { name: "MFI (Money Flow)", pts: "25", long: "MFI ≥70(25) / ≥60(18) / ≥50(10)", short: "MFI ≤30(25) / ≤40(18) / ≤50(10)" },
      { name: "Distribution Days", pts: "20", long: "≤1일(20) / ≤3(14) / ≤5(7)", short: "≥8일(20) / ≥6(14) / ≥4(7)" },
      { name: "Quiet Accum / Churn", pts: "20", long: "OBV↑ + 가격횡보 = 숨은 매집(20)", short: "고vol + VCR확대 = churning(20)" },
      { name: "Close Position", pts: "20", long: "종가 고점 ≥0.7(20)", short: "종가 저점 ≤0.3(20)" },
      { name: "Climax / Dry-up", pts: "15", long: "RSI<35 + vol 2x+ + 반등 = 매도고갈(15)", short: "반등에 vol<0.7 = 기관 미참여(15)" },
    ],
  },
  {
    name: "Relative Value", nameKr: "섹터 상대가치", color: "#a78bfa",
    components: [
      { name: "21일 섹터 z-score", pts: "30", long: "z < -2.0(30) / <-1.5(24) / <-1.0(16) 과매도", short: "z > 2.0(30) / >1.5(24) / >1.0(16) 과매수" },
      { name: "63일 섹터 z-score", pts: "25", long: "z < -2.0(25) / <-1.5(20) / <-1.0(13)", short: "z > 2.0(25) / >1.5(20) / >1.0(13)" },
      { name: "Structural / Overext", pts: "25", long: "SMA200↑(12) + SMA50↑(8) + slope(5)", short: "RSI>75(15) + 52w 고점 근접(10)" },
      { name: "Reversal / Distrib", pts: "20", long: "RSI<35 + OBV↑ 괴리 = 매집(20)", short: "OBV↓ + dist_days ≥3 = 분배(20)" },
    ],
  },
];

function StrategyDefPanel() {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden mb-4">
      <button className="w-full px-4 py-2 text-left text-xs font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between"
        onClick={() => setOpen(!open)}>
        <span>Hedge Fund Strategy Scoring Logic (8 Strategies)</span>
        <span className="text-gray-500">{open ? "▼" : "▶"}</span>
      </button>
      {open && (
        <div className="p-3 bg-[#0d1117] space-y-5">
          <p className="text-[10px] text-gray-500">
            각 전략의 Long/Short 점수(0-100) 산출 로직. Combined Signal = 가중 평균(O'Neil 1.5x &gt; Minervini 1.3x &gt; Wyckoff/Regime 1.2x &gt; Flow 1.1x &gt; Ichimoku 1.0x &gt; RelVal 0.9x &gt; Darvas 0.8x).
            STRONG_LONG: 6개+ Long(≥60) &amp; net&gt;20 | LONG: 4개+ &amp; net&gt;10 | STRONG_SHORT/SHORT: 반대.
          </p>
          {STRATEGY_LOGIC.map((strat) => (
            <div key={strat.name}>
              <div className="flex items-center gap-2 mb-1.5">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: strat.color }} />
                <span className="text-[11px] font-semibold" style={{ color: strat.color }}>{strat.name}</span>
                <span className="text-[10px] text-gray-500">{strat.nameKr}</span>
              </div>
              <table className="w-full text-[10px] border-collapse ml-4">
                <thead><tr className="border-b border-gray-800">
                  <th className="text-left py-0.5 px-1.5 text-gray-600 w-32">Component</th>
                  <th className="text-center py-0.5 px-1.5 text-gray-600 w-10">Pts</th>
                  <th className="text-left py-0.5 px-1.5 text-green-500/50">Long Logic</th>
                  <th className="text-left py-0.5 px-1.5 text-red-500/50">Short Logic</th>
                </tr></thead>
                <tbody>
                  {strat.components.map((c) => (
                    <tr key={c.name} className="border-b border-gray-800/30">
                      <td className="py-0.5 px-1.5 text-gray-400">{c.name}</td>
                      <td className="py-0.5 px-1.5 text-center text-gray-500">{c.pts}</td>
                      <td className="py-0.5 px-1.5 text-gray-400">{c.long}</td>
                      <td className="py-0.5 px-1.5 text-gray-400">{c.short}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Strategy Section: shows ETF + Stock tables for one strategy ─── */
function StrategySection({ strat, data }: { strat: StrategyDef; data: any[] }) {
  const cols = useMemo(() => makeStrategyColumns(strat), [strat]);
  const [showLogic, setShowLogic] = useState(false);

  const sorted = useMemo(() => [...data].sort(signalTieredSort), [data]);

  const etf = useMemo(() => sorted.filter((d) => d.asset_type === "ETF"), [sorted]);
  const stock = useMemo(() => sorted.filter((d) => d.asset_type === "Stock"), [sorted]);

  // Count signals
  const etfLong = etf.filter((d) => Number(d[strat.longKey] || 0) >= 60).length;
  const etfShort = etf.filter((d) => Number(d[strat.shortKey] || 0) >= 60).length;
  const stkLong = stock.filter((d) => Number(d[strat.longKey] || 0) >= 60).length;
  const stkShort = stock.filter((d) => Number(d[strat.shortKey] || 0) >= 60).length;

  // Find matching strategy logic
  const logic = STRATEGY_LOGIC.find((s) => s.name === strat.name);

  return (
    <Section
      title={`${strat.name} — ${strat.nameKr}`}
      color={strat.color}
      badge={`L:${etfLong + stkLong} S:${etfShort + stkShort}`}
    >
      <p className="text-[11px] text-gray-500 mb-2">{strat.desc}</p>

      {/* Inline scoring logic toggle */}
      {logic && (
        <div className="mb-3 border border-gray-800 rounded-lg overflow-hidden">
          <button
            className="w-full px-3 py-1.5 text-left text-[10px] font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between items-center"
            onClick={() => setShowLogic(!showLogic)}
          >
            <span style={{ color: strat.color }}>Scoring Logic — Long / Short 산출 기준</span>
            <span className="text-gray-600">{showLogic ? "▲" : "▼"}</span>
          </button>
          {showLogic && (
            <div className="px-3 py-2 bg-[#0d1117]">
              <table className="w-full text-[10px] border-collapse">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left py-0.5 px-1.5 text-gray-600 w-32">Component</th>
                    <th className="text-center py-0.5 px-1.5 text-gray-600 w-10">Pts</th>
                    <th className="text-left py-0.5 px-1.5 text-green-500/60">Long Logic</th>
                    <th className="text-left py-0.5 px-1.5 text-red-500/60">Short Logic</th>
                  </tr>
                </thead>
                <tbody>
                  {logic.components.map((c) => (
                    <tr key={c.name} className="border-b border-gray-800/30">
                      <td className="py-0.5 px-1.5 text-gray-400 font-semibold">{c.name}</td>
                      <td className="py-0.5 px-1.5 text-center text-gray-500">{c.pts}</td>
                      <td className="py-0.5 px-1.5 text-gray-400">{c.long}</td>
                      <td className="py-0.5 px-1.5 text-gray-400">{c.short}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <div className="flex gap-4 mb-3 text-[10px]">
        <span className="text-gray-500">ETF: <span className="text-green-400">{etfLong} Long</span> / <span className="text-red-400">{etfShort} Short</span></span>
        <span className="text-gray-500">Stock: <span className="text-green-400">{stkLong} Long</span> / <span className="text-red-400">{stkShort} Short</span></span>
      </div>

      {etf.length > 0 && (
        <>
          <h4 className="text-xs font-semibold text-gray-400 mb-1">ETF ({etf.length})</h4>
          <DataTable data={etf} columns={cols} maxHeight="350px" />
        </>
      )}
      {stock.length > 0 && (
        <>
          <h4 className="text-xs font-semibold text-gray-400 mb-1 mt-3">Stocks ({stock.length})</h4>
          <DataTable data={stock} columns={cols} maxHeight="350px" />
        </>
      )}
    </Section>
  );
}

/* ─── Combined Signal Section ─── */
function CombinedSignalSection({ data }: { data: any[] }) {
  const etf = useMemo(() =>
    data.filter((d) => d.asset_type === "ETF").sort(signalTieredSort),
    [data]
  );
  const stock = useMemo(() =>
    data.filter((d) => d.asset_type === "Stock").sort(signalTieredSort),
    [data]
  );

  const strongLong = data.filter((d) => d.net_signal === "STRONG_LONG").length;
  const long = data.filter((d) => d.net_signal === "LONG").length;
  const neutral = data.filter((d) => d.net_signal === "NEUTRAL" || !d.net_signal).length;
  const short = data.filter((d) => d.net_signal === "SHORT").length;
  const strongShort = data.filter((d) => d.net_signal === "STRONG_SHORT").length;

  return (
    <Section title="Combined Multi-Strategy Signal" defaultOpen badge={`SL:${strongLong} L:${long} N:${neutral} S:${short} SS:${strongShort}`} color={C.cyan}>
      <p className="text-[11px] text-gray-500 mb-3">
        8개 전략의 가중 평균 Long/Short 점수 + 방향성 합의. 가중치: O'Neil(1.5x) &gt; Minervini(1.3x) &gt; Wyckoff/Regime(1.2x) &gt; Flow(1.1x) &gt; Ichimoku(1.0x) &gt; RelVal(0.9x) &gt; Darvas(0.8x)
      </p>
      <div className="flex gap-6 mb-4 text-[11px]">
        <span style={{ color: C.green }} className="font-bold">STRONG LONG: {strongLong}</span>
        <span className="text-green-400/70">LONG: {long}</span>
        <span className="text-gray-500">NEUTRAL: {neutral}</span>
        <span className="text-red-400/70">SHORT: {short}</span>
        <span style={{ color: C.red }} className="font-bold">STRONG SHORT: {strongShort}</span>
      </div>

      {etf.length > 0 && (
        <>
          <h4 className="text-xs font-semibold text-gray-400 mb-1">ETF ({etf.length})</h4>
          <DataTable data={etf} columns={combinedColumns} maxHeight="400px" />
        </>
      )}
      {stock.length > 0 && (
        <>
          <h4 className="text-xs font-semibold text-gray-400 mb-1 mt-3">Stocks ({stock.length})</h4>
          <DataTable data={stock} columns={combinedColumns} maxHeight="400px" />
        </>
      )}
    </Section>
  );
}

/* ─── Main Component ─── */
export function MasterTableTab({ filters }: { filters: FilterParams }) {
  const [rawData, setRawData] = useState<any[]>([]);
  useEffect(() => { fetchTable(filters).then((d) => setRawData(d.data || [])); }, [filters]);

  const data = useMemo(() => [...rawData].sort(signalTieredSort), [rawData]);
  const etf = useMemo(() => data.filter((d) => d.asset_type === "ETF"), [data]);
  const stock = useMemo(() => data.filter((d) => d.asset_type === "Stock"), [data]);

  const hasStrategies = data.length > 0 && data[0]?.minervini_long !== undefined;

  if (!data.length) return <div className="text-gray-500 p-8">Loading...</div>;

  return (
    <div className="space-y-6">
      <ColumnDefPanel />
      <ClassDefPanel />
      <StrategyDefPanel />

      {/* Master Summary */}
      <div className="flex items-center gap-4">
        <h3 className="text-lg font-semibold">Master Summary ({data.length} tickers)</h3>
        <button
          className="px-3 py-1 text-xs bg-[#1f2937] border border-gray-700 rounded hover:bg-[#374151]"
          onClick={() => {
            const csv = [Object.keys(data[0]).join(","), ...data.map((r) => Object.values(r).join(","))].join("\n");
            const blob = new Blob([csv], { type: "text/csv" });
            const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
            a.download = "price_discovery.csv"; a.click();
          }}>
          Download CSV
        </button>
      </div>
      <DataTable data={data} columns={masterColumns} />

      {etf.length > 0 && (
        <>
          <h3 className="text-lg font-semibold mt-6">ETF ({etf.length})</h3>
          <DataTable data={etf} columns={masterColumns} maxHeight="400px" />
        </>
      )}
      {stock.length > 0 && (
        <>
          <h3 className="text-lg font-semibold mt-6">Stocks ({stock.length})</h3>
          <DataTable data={stock} columns={masterColumns} maxHeight="400px" />
        </>
      )}

      {/* Strategy Sections */}
      {hasStrategies && (
        <>
          <div className="mt-8">
            <h2 className="text-lg font-bold text-gray-200 mb-1">Hedge Fund Strategy Signals</h2>
            <p className="text-xs text-gray-500 mb-4">
              8개 주요 헤지펀드 전략별 Long/Short 시그널. 각 전략 독립 산출 (0-100). ETF/Stock 분리 표시.
            </p>
          </div>

          {/* Combined signal first */}
          <CombinedSignalSection data={data} />

          {/* Individual strategy sections */}
          {STRATEGIES.map((strat) => (
            <StrategySection key={strat.key} strat={strat} data={data} />
          ))}
        </>
      )}
    </div>
  );
}
