import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { fetchOverview, fetchMeta, type FilterParams } from "../../api/client";
import { MetricCard } from "../shared/MetricCard";
import { C, CLASS_COLORS, DARK_LAYOUT } from "../../styles/theme";

function Section({ title, children, defaultOpen = false }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button className="w-full px-4 py-3 text-left text-sm font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between items-center"
        onClick={() => setOpen(!open)}>
        <span>{title}</span>
        <span className="text-gray-500 text-xs">{open ? "▼" : "▶"}</span>
      </button>
      {open && <div className="p-4 bg-[#0d1117] text-sm text-gray-300 space-y-3">{children}</div>}
    </div>
  );
}

/** Reusable Long/Short signal table for ETF/Stock split views */
function SignalTable({ rows, direction }: { rows: any[]; direction: "long" | "short" }) {
  if (!rows || rows.length === 0) return null;
  const isLong = direction === "long";
  const countKey = isLong ? "long_count" : "short_count";
  const strategies = isLong
    ? ["oneil_long","minervini_long","wyckoff_long","ichimoku_long","darvas_long","regime_long","flow_long","relval_long"]
    : ["oneil_short","minervini_short","wyckoff_short","ichimoku_short","darvas_short","regime_short","flow_short","relval_short"];
  const stratHeaders = ["O'Neil","Minerv","Wyck","Ichi","Darv","Regime","Flow","RV"];
  const sc = (v: any, isL = isLong) => {
    const n = Number(v ?? 0);
    const c = isL
      ? (n >= 70 ? "text-green-400 font-bold" : n >= 50 ? "text-green-400/70" : n >= 30 ? "text-gray-400" : "text-gray-600")
      : (n >= 70 ? "text-red-400 font-bold" : n >= 50 ? "text-red-400/70" : n >= 30 ? "text-gray-400" : "text-gray-600");
    return <span className={`${c} text-[10px]`}>{n.toFixed(0)}</span>;
  };
  const sigColor: Record<string, string> = { STRONG_LONG: "#22c55e", LONG: "#86efac", STRONG_SHORT: "#ef4444", SHORT: "#fca5a5", NEUTRAL: "#6b7280" };
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-gray-700 bg-[#111827]">
            <th className="py-2 px-2 text-left text-gray-500 w-8">#</th>
            <th className="py-2 px-2 text-left text-gray-500">Ticker</th>
            <th className="py-2 px-2 text-left text-gray-500">Name</th>
            <th className="py-2 px-2 text-left text-gray-500">Sector</th>
            <th className="py-2 px-2 text-right text-gray-500">MCap$B</th>
            <th className="py-2 px-2 text-left text-gray-500">Class</th>
            <th className="py-2 px-2 text-center text-gray-500">Signal</th>
            <th className="py-2 px-2 text-right text-gray-500">{isLong ? "L#" : "S#"}</th>
            <th className="py-2 px-2 text-right text-gray-500">Conv</th>
            {stratHeaders.map(h => <th key={h} className="py-2 px-2 text-right text-gray-500">{h}</th>)}
            <th className="py-2 px-2 text-right text-gray-500">Comp</th>
            {isLong && <th className="py-2 px-2 text-center text-gray-500">Risk</th>}
            <th className="py-2 px-2 text-right text-gray-500">SQ</th>
            <th className="py-2 px-2 text-right text-gray-500">APS</th>
            <th className="py-2 px-2 text-right text-gray-500">TCS</th>
            <th className="py-2 px-2 text-right text-gray-500">TFS</th>
            <th className="py-2 px-2 text-right text-gray-500">RSS</th>
            <th className="py-2 px-2 text-right text-gray-500">OER</th>
            <th className="py-2 px-2 text-right text-gray-500">RSI</th>
            <th className="py-2 px-2 text-right text-gray-500">Age</th>
            <th className="py-2 px-2 text-right text-gray-500">1W</th>
            <th className="py-2 px-2 text-right text-gray-500">1M</th>
            <th className="py-2 px-2 text-right text-gray-500">3M</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((d: any, i: number) => {
            const clsColor = CLASS_COLORS[d.classification] || C.gray;
            const isEvent = d.event_flag;
            const sq = d.structural_q;
            const sqColor = sq >= 70 ? "text-green-400" : sq >= 40 ? "text-gray-300" : "text-yellow-400";
            return (
              <tr key={d.ticker} className={`border-b border-gray-800/50 hover:bg-[#1f2937]/30 ${isEvent && isLong ? "bg-yellow-900/10" : ""}`}>
                <td className="py-1.5 px-2 text-gray-600">{i + 1}</td>
                <td className="py-1.5 px-2 font-mono font-bold text-white">
                  {d.ticker}
                  {isEvent && isLong && <span className="ml-1 text-yellow-400 text-[9px]" title={d.event_reasons}>⚡</span>}
                </td>
                <td className="py-1.5 px-2 text-gray-400 truncate max-w-[140px]">{String(d.name).slice(0, 22)}</td>
                <td className="py-1.5 px-2 text-cyan-400/70 text-[10px]">{d.sector}</td>
                <td className="py-1.5 px-2 text-right text-gray-400 text-[10px]">{d.mktcap_B != null && Number(d.mktcap_B) > 0 ? (Number(d.mktcap_B) >= 1 ? Number(d.mktcap_B).toFixed(1) : Number(d.mktcap_B).toFixed(2)) : "-"}</td>
                <td className="py-1.5 px-2 text-[10px] font-medium" style={{ color: clsColor }}>{String(d.classification).replace(/^.\s/, "")}</td>
                <td className="py-1.5 px-2 text-center"><span style={{ color: sigColor[d.net_signal] || "#6b7280" }} className="text-[10px] font-bold">{d.net_signal || "-"}</span></td>
                <td className="py-1.5 px-2 text-right text-gray-300 text-[10px]">{d[countKey] ?? "-"}</td>
                <td className="py-1.5 px-2 text-right text-gray-300 text-[10px]">{d.conviction != null ? Number(d.conviction).toFixed(1) : "-"}</td>
                {strategies.map(k => <td key={k} className="py-1.5 px-2 text-right">{sc(d[k])}</td>)}
                <td className="py-1.5 px-2 text-right">{Number(d.composite).toFixed(1)}</td>
                {isLong && (
                  <td className="py-1.5 px-2 text-center text-[9px]">
                    {isEvent ? <span className="text-yellow-400 font-semibold" title={d.event_reasons}>EVENT</span> : <span className="text-gray-600">—</span>}
                  </td>
                )}
                <td className={`py-1.5 px-2 text-right ${sqColor}`}>{sq != null ? sq : "-"}</td>
                <td className={`py-1.5 px-2 text-right text-[10px] ${Number(d.alpha_potential ?? 0) >= 70 ? "text-emerald-400 font-bold" : Number(d.alpha_potential ?? 0) >= 50 ? "text-emerald-400/70" : "text-gray-400"}`}>{d.alpha_potential != null ? Number(d.alpha_potential).toFixed(0) : "-"}</td>
                <td className="py-1.5 px-2 text-right text-gray-400">{d.tcs}</td>
                <td className="py-1.5 px-2 text-right text-gray-400">{d.tfs}</td>
                <td className="py-1.5 px-2 text-right text-gray-400">{d.rss}</td>
                <td className="py-1.5 px-2 text-right text-gray-400">{d.oer}</td>
                <td className="py-1.5 px-2 text-right text-gray-400">{Number(d.rsi).toFixed(1)}</td>
                <td className="py-1.5 px-2 text-right text-gray-400">{d.trend_age}</td>
                <td className={`py-1.5 px-2 text-right ${Number(d.ret_1w) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_1w).toFixed(2)}%</td>
                <td className={`py-1.5 px-2 text-right ${Number(d.ret_1m) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_1m).toFixed(2)}%</td>
                <td className={`py-1.5 px-2 text-right ${Number(d.ret_3m) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_3m).toFixed(2)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function OverviewTab({ filters }: { filters: FilterParams }) {
  const [data, setData] = useState<any>(null);
  const [meta, setMeta] = useState<any>(null);
  useEffect(() => { fetchOverview(filters).then(setData); }, [filters]);
  useEffect(() => { fetchMeta().then(setMeta); }, []);

  if (!data) return <div className="text-gray-500 p-8">Loading...</div>;
  const { kpis, classification_dist, composite_data, conviction_bubble, top_eligible, top_long, top_long_warnings, top_short, top_long_bt, top_long_etf, top_long_stock, top_short_etf, top_short_stock } = data;

  return (
    <div className="space-y-6">

      {/* ═══ METHODOLOGY SECTION ═══ */}
      <Section title="Signal Architecture — Dual-Timeframe 3-Axis Scoring" defaultOpen={true}>
        <p className="text-gray-400 text-xs mb-3">
          Price Discovery Scanner v5.0은 글로벌 ETF + 개별주식 유니버스를 대상으로 모멘텀 기반 시그널을 생성하는 정량 스캐너입니다.
          3개의 독립 축(TCS, TFS, OER)과 1개의 크로스섹셔널 랭킹(RSS)을 듀얼 타임프레임(단기/장기)으로 측정합니다.
        </p>

        <div className="grid grid-cols-2 gap-4">
          {/* Left: Scoring Axes */}
          <div className="space-y-2">
            <h4 className="text-cyan-400 font-semibold text-xs uppercase tracking-wide">Composite = 0.35×TCS + 0.30×TFS + 0.35×RSS</h4>
            <table className="w-full text-xs border-collapse">
              <thead><tr className="border-b border-gray-700">
                <th className="text-left py-1 text-gray-500">Axis</th>
                <th className="text-left py-1 text-gray-500">Short (단기)</th>
                <th className="text-left py-1 text-gray-500">Long (장기)</th>
                <th className="text-left py-1 text-gray-500">Weight</th>
              </tr></thead>
              <tbody className="text-gray-400">
                <tr className="border-b border-gray-800">
                  <td className="py-1.5 font-medium text-green-400">TCS</td>
                  <td className="py-1.5">SMA20 거리 연속점수(±2%), 기울기, trend_age 단계적</td>
                  <td className="py-1.5">SMA50 거리(±3%), SMA50-200 spread(±2%), 기울기, trend_age 단계적</td>
                  <td className="py-1.5">40/60%</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-1.5 font-medium text-blue-400">TFS</td>
                  <td className="py-1.5">SMA20 돌파 강도×신선도, 거래량 4단계(1.1x~2x+), 고점 근접도</td>
                  <td className="py-1.5">SMA50 돌파 강도, 거래량 단계적, 20일 브레이크아웃</td>
                  <td className="py-1.5">50/50%</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-1.5 font-medium text-yellow-400">OER</td>
                  <td className="py-1.5" colSpan={2}>SMA20/50 거리 + RSI + 52주 고점 근접도 + 36-12M 반전 리스크 (통합)</td>
                  <td className="py-1.5">분류전용</td>
                </tr>
                <tr>
                  <td className="py-1.5 font-medium text-purple-400">RSS</td>
                  <td className="py-1.5">ret_5d, ret_10d, ret_21d, SMA20 기울기</td>
                  <td className="py-1.5">ret_12-1m, ret_63d, vol_adj_mom, SMA50 기울기</td>
                  <td className="py-1.5">35/65%</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Right: Classification Matrix */}
          <div className="space-y-2">
            <h4 className="text-cyan-400 font-semibold text-xs uppercase tracking-wide">Classification — 3×3 Matrix + Overrides</h4>
            <table className="w-full text-xs border-collapse text-center">
              <thead><tr className="border-b border-gray-700">
                <th className="py-1 text-gray-500">Short＼Long</th>
                <th className="py-1 text-gray-500">UP ↑</th>
                <th className="py-1 text-gray-500">FLAT →</th>
                <th className="py-1 text-gray-500">DOWN ↓</th>
              </tr></thead>
              <tbody>
                <tr className="border-b border-gray-800">
                  <td className="py-1.5 text-gray-500 font-medium text-left">UP ↑</td>
                  <td className="py-1.5" style={{color: C.green}}>CONTINUATION</td>
                  <td className="py-1.5" style={{color: C.blue}}>RECOVERY</td>
                  <td className="py-1.5" style={{color: C.purple}}>COUNTER_RALLY</td>
                </tr>
                <tr className="border-b border-gray-800">
                  <td className="py-1.5 text-gray-500 font-medium text-left">FLAT →</td>
                  <td className="py-1.5" style={{color: C.yellow}}>CONSOLIDATION</td>
                  <td className="py-1.5" style={{color: C.orange}}>NEUTRAL</td>
                  <td className="py-1.5" style={{color: C.brown}}>FADING</td>
                </tr>
                <tr>
                  <td className="py-1.5 text-gray-500 font-medium text-left">DOWN ↓</td>
                  <td className="py-1.5 text-orange-400">PULLBACK</td>
                  <td className="py-1.5 text-red-400">WEAKENING</td>
                  <td className="py-1.5" style={{color: C.red}}>DOWNTREND</td>
                </tr>
              </tbody>
            </table>
            <p className="text-[10px] text-gray-500">
              Overrides: <span className="text-yellow-400">OVEREXTENDED</span> (OER≥adaptive) |{" "}
              <span className="text-blue-400">FORMATION</span> (rapid breakout) |{" "}
              <span className="text-amber-700">EXHAUSTING</span> (old trend losing steam) |{" "}
              <span style={{color: "#dc143c"}}>CYCLE_PEAK</span> (36-12M reversal + declining 12M)
            </p>
          </div>
        </div>

        {/* Classification Detail Reference */}
        <div className="mt-3 p-3 bg-[#111827] rounded border border-gray-800">
          <h4 className="text-cyan-400 font-semibold text-xs uppercase tracking-wide mb-2">Classification Reference — 조건, 특징, 해석</h4>

          {/* 3x3 Matrix Classifications */}
          <h5 className="text-gray-400 font-semibold text-[10px] uppercase tracking-wide mb-1.5 mt-2">3×3 Matrix (Base Classifications)</h5>
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b border-gray-700">
              <th className="text-left py-1 text-gray-500 w-28">Class</th>
              <th className="text-left py-1 text-gray-500 w-28">Direction</th>
              <th className="text-left py-1 text-gray-500">조건</th>
              <th className="text-left py-1 text-gray-500">특징 및 해석</th>
              <th className="text-center py-1 text-gray-500 w-12">Rank</th>
              <th className="text-center py-1 text-gray-500 w-10">Elg</th>
            </tr></thead>
            <tbody className="text-gray-400">
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5" style={{color: C.green}}>CONTINUATION</td>
                <td className="py-1.5 text-gray-500">Short UP + Long UP</td>
                <td className="py-1.5">SMA20 괴리 &gt; +0.5% &amp; SMA50 괴리 &gt; +1% &amp; (SMA50&gt;SMA200 or 기울기↑)</td>
                <td className="py-1.5">단기·장기 모두 상승. 버퍼(±0.5/1%) 적용으로 MA 근처 whipsaw 방지. 기존 포지션 유지 최적</td>
                <td className="py-1.5 text-center text-green-400 font-bold">3</td>
                <td className="py-1.5 text-center">✅</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5" style={{color: C.blue}}>RECOVERY</td>
                <td className="py-1.5 text-gray-500">Short UP + Long FLAT</td>
                <td className="py-1.5">SMA20 괴리 &gt; +0.5%, SMA50 괴리 ±1% 이내 (버퍼 구간)</td>
                <td className="py-1.5">장기 추세가 아직 확립되지 않았으나 단기 반등이 시작. 장기 방향 전환 초기 신호일 수 있음. 확인 대기 후 진입</td>
                <td className="py-1.5 text-center">2</td>
                <td className="py-1.5 text-center">✅</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5" style={{color: C.purple}}>COUNTER_RALLY</td>
                <td className="py-1.5 text-gray-500">Short UP + Long DOWN</td>
                <td className="py-1.5">SMA20 괴리 &gt; +0.5%, SMA50 괴리 &lt; -1% (장기 하락 내 단기 반등)</td>
                <td className="py-1.5">하락 추세 중 기술적 반등(Dead Cat Bounce). 장기 구조가 무너진 상태에서의 단기 상승은 지속 가능성 낮음. 매도 기회로 활용</td>
                <td className="py-1.5 text-center text-red-400">0</td>
                <td className="py-1.5 text-center text-red-400">✗</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5" style={{color: C.yellow}}>CONSOLIDATION</td>
                <td className="py-1.5 text-gray-500">Short FLAT + Long UP</td>
                <td className="py-1.5">SMA20 괴리 ±0.5% 이내, SMA50 괴리 &gt; +1%</td>
                <td className="py-1.5">장기 상승 추세 내 단기 횡보 구간. 에너지 축적 후 재상승(CONTINUATION) 또는 하락 전환(PULLBACK) 분기점. 베이스 형성 관찰 필요</td>
                <td className="py-1.5 text-center">2</td>
                <td className="py-1.5 text-center">✅</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5" style={{color: C.orange}}>NEUTRAL</td>
                <td className="py-1.5 text-gray-500">Short FLAT + Long FLAT</td>
                <td className="py-1.5">SMA20 괴리 ±0.5% 이내 &amp; SMA50 괴리 ±1% 이내</td>
                <td className="py-1.5">양 타임프레임 모두 뚜렷한 방향 없음. 박스권 또는 전환기. 시그널 신뢰도 낮아 진입 비권장, 방향 확인 후 대응</td>
                <td className="py-1.5 text-center">1</td>
                <td className="py-1.5 text-center">✅</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5" style={{color: C.brown}}>FADING</td>
                <td className="py-1.5 text-gray-500">Short FLAT + Long DOWN</td>
                <td className="py-1.5">SMA20 괴리 ±0.5% 이내, SMA50 괴리 &lt; -1%</td>
                <td className="py-1.5">장기 하락 추세에서 단기 반등 시도도 없는 상태. 매수세 완전 소멸. 추가 하락 가능성 높음</td>
                <td className="py-1.5 text-center text-red-400">0</td>
                <td className="py-1.5 text-center text-red-400">✗</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5 text-orange-400">PULLBACK</td>
                <td className="py-1.5 text-gray-500">Short DOWN + Long UP</td>
                <td className="py-1.5">SMA20 괴리 &lt; -0.5%, SMA50 괴리 &gt; +1%</td>
                <td className="py-1.5">건강한 장기 추세 내 단기 조정. 이동평균 지지선 접근 시 매수 기회(Buy the Dip). O'Neil Long과 동시 상승 시 최적 진입점</td>
                <td className="py-1.5 text-center">2</td>
                <td className="py-1.5 text-center">✅</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5 text-red-400">WEAKENING</td>
                <td className="py-1.5 text-gray-500">Short DOWN + Long FLAT</td>
                <td className="py-1.5">SMA20 괴리 &lt; -0.5%, SMA50 괴리 ±1% 이내</td>
                <td className="py-1.5">장기 지지력 약화 + 단기 하락. FADING 또는 DOWNTREND로 전환 가능성. 기존 포지션 리스크 관리 필요</td>
                <td className="py-1.5 text-center">1</td>
                <td className="py-1.5 text-center">✅</td>
              </tr>
              <tr>
                <td className="py-1.5" style={{color: C.red}}>DOWNTREND</td>
                <td className="py-1.5 text-gray-500">Short DOWN + Long DOWN</td>
                <td className="py-1.5">SMA20 괴리 &lt; -0.5% &amp; SMA50 괴리 &lt; -1%</td>
                <td className="py-1.5">양 타임프레임 모두 하락. 기관 매도세 지속. 절대 매수 금지 영역. Short 시그널 참조</td>
                <td className="py-1.5 text-center text-red-400">0</td>
                <td className="py-1.5 text-center text-red-400">✗</td>
              </tr>
            </tbody>
          </table>

          {/* Override Classifications */}
          <h5 className="text-gray-400 font-semibold text-[10px] uppercase tracking-wide mb-1.5 mt-4">Override Classifications (우선순위 순)</h5>
          <table className="w-full text-[11px] border-collapse">
            <thead><tr className="border-b border-gray-700">
              <th className="text-left py-1 text-gray-500 w-28">Class</th>
              <th className="text-left py-1 text-gray-500 w-10">순위</th>
              <th className="text-left py-1 text-gray-500">발동 조건</th>
              <th className="text-left py-1 text-gray-500">특징 및 해석</th>
              <th className="text-center py-1 text-gray-500 w-12">Rank</th>
              <th className="text-center py-1 text-gray-500 w-10">Elg</th>
            </tr></thead>
            <tbody className="text-gray-400">
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5 text-yellow-400">OVEREXTENDED</td>
                <td className="py-1.5 text-gray-500">1st</td>
                <td className="py-1.5">OER ≥ adaptive threshold(기본 60) AND base가 CONTINUATION/RECOVERY/CONSOLIDATION</td>
                <td className="py-1.5">강한 상승 추세이나 SMA20/50 괴리, RSI 과매수, 52주 고점 근접 등 단기 과열. 신규 진입 시 높은 평균단가 리스크. 조정 대기 권장</td>
                <td className="py-1.5 text-center">2</td>
                <td className="py-1.5 text-center">✅</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5 text-blue-400">FORMATION</td>
                <td className="py-1.5 text-gray-500">2nd</td>
                <td className="py-1.5">TFS_short ≥ adaptive(기본 50) AND trend_age_short ≤ 5일 AND long_dir = UP</td>
                <td className="py-1.5">장기 상승 추세 내에서 단기 조정 후 SMA20을 막 돌파한 초기 단계(5일 이내). 거래량 급증 + 브레이크아웃 동반. O'Neil Long과 동시 고점 시 최고 확신 매수 타이밍</td>
                <td className="py-1.5 text-center">2</td>
                <td className="py-1.5 text-center">✅</td>
              </tr>
              <tr className="border-b border-gray-800/50">
                <td className="py-1.5" style={{color: "#dc143c"}}>CYCLE_PEAK</td>
                <td className="py-1.5 text-gray-500">3rd</td>
                <td className="py-1.5">reversal_pctile ≥ 85 (36-12M 수익률 상위 15%) AND ret_12_1m {"<"} ret_36_12m × 0.3 AND short_dir ≠ UP</td>
                <td className="py-1.5">3년간 극단적 상승(상위 15%) 후 최근 12개월 모멘텀이 크게 둔화되고 단기마저 약세 전환. De Bondt &amp; Thaler(1985) 장기 반전 영역. 구조적 사이클 고점 경고</td>
                <td className="py-1.5 text-center text-red-400">0</td>
                <td className="py-1.5 text-center text-red-400">✗</td>
              </tr>
              <tr>
                <td className="py-1.5" style={{color: C.brown}}>EXHAUSTING</td>
                <td className="py-1.5 text-gray-500">4th</td>
                <td className="py-1.5">trend_age {">"} 60일 AND ret_21d {"<"} ret_63d / 3 AND long_dir = UP</td>
                <td className="py-1.5">장기 상승이 60일+ 지속되었으나 최근 1개월 수익률이 3개월 대비 급격히 둔화. 추세 피로 누적, 조정 임박 신호. 신규 진입 지양, 기존 포지션 이익 실현 검토</td>
                <td className="py-1.5 text-center">1</td>
                <td className="py-1.5 text-center text-red-400">✗</td>
              </tr>
            </tbody>
          </table>

          <p className="text-[10px] text-gray-500 mt-2">
            Override 우선순위: OVEREXTENDED → FORMATION → CYCLE_PEAK → EXHAUSTING. 복수 조건 충족 시 상위 우선순위가 적용됨.
            Rank: 0(부적격) → 1(약세) → 2(중립/초기) → 3(강세). Eligibility(Elg): ✅=포트폴리오 편입 가능, ✗=자동 제외.
          </p>
        </div>

        {/* Eligibility */}
        <div className="mt-3 p-3 bg-[#111827] rounded border border-gray-800">
          <h4 className="text-cyan-400 font-semibold text-xs uppercase tracking-wide mb-1">Portfolio Eligibility</h4>
          <p className="text-xs text-gray-400">
            Eligible = NOT (DOWNTREND | EXHAUSTING | FADING | COUNTER_RALLY | CYCLE_PEAK) AND Composite ≥ adaptive threshold AND ADV ≥ $5M
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Adaptive thresholds: train set 분포 기반 동적 산출 (Walk-Forward Validation, Purged CV embargo 5일)
          </p>
        </div>

        {/* O'Neil CANSLIM Signal */}
        <div className="mt-3 p-3 bg-[#111827] rounded border border-gray-800">
          <h4 className="text-cyan-400 font-semibold text-xs uppercase tracking-wide mb-2">O'Neil (CANSLIM) Long / Short Signal — William O'Neil Methodology</h4>
          <p className="text-xs text-gray-400 mb-3">
            Composite 스코어와 독립적으로 작동하는 William O'Neil 방법론 기반 매수/매도 시그널입니다.
            O'Neil의 핵심 원칙 — 피벗 돌파 매수, 거래량 확인, 상대강도 필터, 이동평균 구조, 베이스 패턴 — 을 정량 스코어(0-100)로 변환합니다.
          </p>

          <div className="grid grid-cols-2 gap-4">
            {/* Long Signal */}
            <div>
              <h5 className="text-green-400 font-semibold text-[11px] mb-1.5">Long Signal (매수)</h5>
              <table className="w-full text-[11px] border-collapse">
                <thead><tr className="border-b border-gray-700">
                  <th className="text-left py-1 text-gray-500">Component</th>
                  <th className="text-right py-1 text-gray-500 w-10">Pts</th>
                  <th className="text-left py-1 text-gray-500 pl-2">O'Neil Principle</th>
                </tr></thead>
                <tbody className="text-gray-400">
                  <tr className="border-b border-gray-800/50">
                    <td className="py-1 text-green-400/80">Pivot Proximity</td>
                    <td className="py-1 text-right">25</td>
                    <td className="py-1 pl-2">52주 고점 2% 이내(25) / 5%(18) / 10%(8) — 피벗 포인트 돌파 매수</td>
                  </tr>
                  <tr className="border-b border-gray-800/50">
                    <td className="py-1 text-green-400/80">Volume Surge</td>
                    <td className="py-1 text-right">20</td>
                    <td className="py-1 pl-2">vol_ratio 2x+(20) / 1.5x+(15) / 1.2x+(8) — 돌파 시 거래량 50%↑ 필수</td>
                  </tr>
                  <tr className="border-b border-gray-800/50">
                    <td className="py-1 text-green-400/80">RS Rating</td>
                    <td className="py-1 text-right">20</td>
                    <td className="py-1 pl-2">RSS ≥85(20) / ≥75(15) / ≥60(8) — RS 80 이상만 매수 대상</td>
                  </tr>
                  <tr className="border-b border-gray-800/50">
                    <td className="py-1 text-green-400/80">MA Structure</td>
                    <td className="py-1 text-right">20</td>
                    <td className="py-1 pl-2">SMA50↑(6) + SMA200↑(5) + Golden Cross(5) + Slope↑(4)</td>
                  </tr>
                  <tr>
                    <td className="py-1 text-green-400/80">Base Breakout</td>
                    <td className="py-1 text-right">15</td>
                    <td className="py-1 pl-2">VCR{"<"}0.6+돌파(15) / VCR{"<"}0.8+돌파(12) — 타이트 수렴 후 돌파</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Short Signal */}
            <div>
              <h5 className="text-red-400 font-semibold text-[11px] mb-1.5">Short Signal (매도)</h5>
              <table className="w-full text-[11px] border-collapse">
                <thead><tr className="border-b border-gray-700">
                  <th className="text-left py-1 text-gray-500">Component</th>
                  <th className="text-right py-1 text-gray-500 w-10">Pts</th>
                  <th className="text-left py-1 text-gray-500 pl-2">O'Neil Principle</th>
                </tr></thead>
                <tbody className="text-gray-400">
                  <tr className="border-b border-gray-800/50">
                    <td className="py-1 text-red-400/80">Support Break</td>
                    <td className="py-1 text-right">25</td>
                    <td className="py-1 pl-2">SMA50+200 이탈(25) / SMA50만(15) — 기관 매도 신호</td>
                  </tr>
                  <tr className="border-b border-gray-800/50">
                    <td className="py-1 text-red-400/80">Distribution Vol</td>
                    <td className="py-1 text-right">20</td>
                    <td className="py-1 pl-2">하락+vol 2x+(20) / 1.5x+(15) — 기관 분배일(Distribution Day)</td>
                  </tr>
                  <tr className="border-b border-gray-800/50">
                    <td className="py-1 text-red-400/80">RS Weakness</td>
                    <td className="py-1 text-right">20</td>
                    <td className="py-1 pl-2">RSS ≤15(20) / ≤25(15) / ≤40(8) — 하위 종목 가속 하락</td>
                  </tr>
                  <tr className="border-b border-gray-800/50">
                    <td className="py-1 text-red-400/80">MA Deterioration</td>
                    <td className="py-1 text-right">20</td>
                    <td className="py-1 pl-2">Death Cross(6) + SMA50↓(6) + SMA200↓(4) + SMA20↓(4)</td>
                  </tr>
                  <tr>
                    <td className="py-1 text-red-400/80">Trend Deterioration</td>
                    <td className="py-1 text-right">15</td>
                    <td className="py-1 pl-2">52주 저점 근접(10) + 지속 하락 모멘텀(5)</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-3 grid grid-cols-3 gap-3 text-[10px]">
            <div className="bg-[#0d1117] rounded p-2 border border-gray-800">
              <span className="text-gray-500">Score Interpretation:</span>
              <div className="mt-1 space-y-0.5">
                <div><span className="text-green-400 font-bold">70-100</span> <span className="text-gray-500">= Strong Signal (적극 매수/매도)</span></div>
                <div><span className="text-green-400/70">50-69</span> <span className="text-gray-500">= Moderate (조건부 진입)</span></div>
                <div><span className="text-gray-400">30-49</span> <span className="text-gray-500">= Weak (관망)</span></div>
                <div><span className="text-gray-600">0-29</span> <span className="text-gray-500">= No Signal</span></div>
              </div>
            </div>
            <div className="bg-[#0d1117] rounded p-2 border border-gray-800">
              <span className="text-gray-500">Key Indicator — VCR (Volatility Contraction Ratio):</span>
              <p className="mt-1 text-gray-400">
                최근 5일 변동성 / 과거 40일 변동성. 1 미만이면 가격 수렴(Base 형성) 진행 중.
                O'Neil: 타이트한 수렴 후 거래량 동반 돌파가 가장 높은 성공률.
              </p>
            </div>
            <div className="bg-[#0d1117] rounded p-2 border border-gray-800">
              <span className="text-gray-500">Composite와의 관계:</span>
              <p className="mt-1 text-gray-400">
                Long/Short은 Composite Score와 독립 산출. Composite는 추세 강도/형성/상대강도를 측정하고,
                O'Neil 시그널은 특정 매수/매도 <em>타이밍</em>을 포착. 두 지표 동시 고점 = 최고 확신 구간.
              </p>
            </div>
          </div>
        </div>
      </Section>

      {/* ═══ CATEGORY & BENCHMARK SECTION ═══ */}
      {meta?.category_info && (
        <Section title={`Category & Benchmark Definition (${meta.category_info.length} categories)`}>
          <div className="grid grid-cols-2 gap-4">
            {/* ETF Categories */}
            <div>
              <h4 className="text-cyan-400 font-semibold text-xs uppercase tracking-wide mb-2">ETF Categories</h4>
              <table className="w-full text-xs border-collapse">
                <thead><tr className="border-b border-gray-700">
                  <th className="text-left py-1 text-gray-500">Category</th>
                  <th className="text-right py-1 text-gray-500">N</th>
                  <th className="text-left py-1 text-gray-500 pl-3">Benchmark</th>
                  <th className="text-left py-1 text-gray-500">Alternatives</th>
                </tr></thead>
                <tbody className="text-gray-400">
                  {meta.category_info.filter((c: any) => c.asset_type === "ETF").map((c: any) => (
                    <tr key={c.category} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
                      <td className="py-1">{c.category}</td>
                      <td className="py-1 text-right text-gray-500">{c.n}</td>
                      <td className="py-1 pl-3 font-mono text-cyan-400">{c.benchmark}</td>
                      <td className="py-1 text-gray-600 text-[10px]">{c.alt_benchmarks.join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Stock Categories */}
            <div>
              <h4 className="text-cyan-400 font-semibold text-xs uppercase tracking-wide mb-2">Stock Categories</h4>
              <table className="w-full text-xs border-collapse">
                <thead><tr className="border-b border-gray-700">
                  <th className="text-left py-1 text-gray-500">Category</th>
                  <th className="text-right py-1 text-gray-500">N</th>
                  <th className="text-left py-1 text-gray-500 pl-3">Benchmark</th>
                  <th className="text-left py-1 text-gray-500">Alternatives</th>
                </tr></thead>
                <tbody className="text-gray-400">
                  {meta.category_info.filter((c: any) => c.asset_type === "Stock").map((c: any) => (
                    <tr key={c.category} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
                      <td className="py-1">{c.category.replace("STK_", "")}</td>
                      <td className="py-1 text-right text-gray-500">{c.n}</td>
                      <td className="py-1 pl-3 font-mono text-cyan-400">{c.benchmark}</td>
                      <td className="py-1 text-gray-600 text-[10px]">{c.alt_benchmarks.join(", ")}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="text-[10px] text-gray-500 mt-2">
            Excess Return = median(ticker return vs each alternative benchmark). 복수 벤치마크 중앙값으로 사후 선택 편향 완화.
          </p>
        </Section>
      )}

      {/* ═══ THEME SECTION ═══ */}
      {meta?.theme_info && meta.theme_info.length > 0 && (
        <Section title={`Consolidated Themes (${meta.theme_info.length} macro themes from 336 granular)`}>
          <p className="text-xs text-gray-400 mb-2">
            개별 종목의 세부 테마(336개)를 밸류체인·GICS·투자테마 단위로 통합하여 47개 매크로 테마로 재분류.
            최소 2종목 이상, 카테고리 무관 크로스섹셔널 비교 가능.
          </p>
          <div className="grid grid-cols-3 gap-x-4 gap-y-0.5 text-xs">
            {meta.theme_info.map((t: any) => (
              <div key={t.theme} className="flex justify-between py-0.5 border-b border-gray-800/30">
                <span className="text-gray-300 truncate mr-2">{t.theme}</span>
                <span className="text-gray-500 shrink-0">{t.n} <span className="text-gray-600 text-[9px]">({t.categories.join(", ")})</span></span>
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* ═══ KPIs ═══ */}
      <div className="grid grid-cols-6 gap-3">
        <MetricCard label="Total" value={kpis.total} sub={`ETF ${kpis.n_etf} / Stock ${kpis.n_stock}`} />
        <MetricCard label="Eligible" value={kpis.eligible} sub={`${(kpis.eligible / kpis.total * 100).toFixed(0)}%`} />
        <MetricCard label="Avg Composite" value={kpis.avg_composite} sub={`med ${kpis.median_composite}`} />
        <MetricCard label="Continuation" value={`${kpis.pct_continuation}%`} />
        <MetricCard label="Downtrend" value={`${kpis.pct_downtrend}%`} />
        <MetricCard label="1M / 3M Ret" value={`${kpis.avg_ret_1m}%`} sub={`3M: ${kpis.avg_ret_3m}%`} />
      </div>

      {/* ═══ Top 10 Strong Long ═══ */}
      {top_long && top_long.length > 0 && (
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
            Top 10 Strong Long — Multi-Strategy Consensus
          </h3>
          <div className="mb-3 p-3 bg-[#111827] border border-gray-800 rounded-lg text-[10px] text-gray-400 space-y-2">
            <div className="text-[11px] font-semibold text-cyan-400 mb-1">Selection Logic</div>
            <div className="grid grid-cols-[80px_1fr] gap-y-1">
              <span className="text-gray-500 font-semibold">Step 1</span>
              <span><span className="text-gray-300">후보 필터링</span> — eligible=True AND (bullish classification OR <span className="text-emerald-400">APS ≥ 70 Hidden Gems</span>)</span>
              <span className="text-gray-500 font-semibold">Step 2</span>
              <span><span className="text-gray-300">랭킹 점수</span> — <span className="text-cyan-400 font-mono">_long_rank = (combined_long × 0.40 + composite × 0.35 + APS × 0.25) × discount</span></span>
              <span className="text-gray-500 font-semibold"></span>
              <span>
                <span className="text-green-400/80">combined_long (40%)</span>: 8개 전략 가중 평균 Long 점수 (O&apos;Neil 1.5x &gt; Minervini 1.3x &gt; Wyckoff/Regime 1.2x &gt; Flow 1.1x &gt; Ichimoku 1.0x &gt; RelVal 0.9x &gt; Darvas 0.8x)
              </span>
              <span className="text-gray-500 font-semibold"></span>
              <span>
                <span className="text-blue-400/80">composite (35%)</span>: Price Discovery 종합 모멘텀 (0.35×TCS + 0.30×TFS + 0.35×RSS)
              </span>
              <span className="text-gray-500 font-semibold"></span>
              <span>
                <span className="text-emerald-400/80">APS (25%)</span>: Alpha Potential Score — 매집 품질(25) + 변동성 셋업(20) + 모멘텀 구조(25) + 상대강도(15) + 돌파 근접(15)
              </span>
              <span className="text-gray-500 font-semibold"></span>
              <span>
                <span className="text-yellow-400/80">discount</span>: EVENT 감지 시 ×0.7 (단기스파이크 + 거래량급증 + 초기추세 + 변동성확대 중 2개+ 충족)
              </span>
              <span className="text-gray-500 font-semibold">Step 3</span>
              <span><span className="text-gray-300">섹터 집중 제한</span> — 동일 섹터 최대 3종목 (Sector Cap). 초과 시 차순위 종목으로 대체</span>
              <span className="text-gray-500 font-semibold">Step 4</span>
              <span><span className="text-gray-300">상위 10개 선택</span> — _long_rank 내림차순 상위 10</span>
            </div>
          </div>

          {/* Risk Warnings */}
          {top_long_warnings && top_long_warnings.length > 0 && (
            <div className="mb-3 p-2.5 bg-yellow-900/20 border border-yellow-700/40 rounded-lg">
              <span className="text-yellow-400 text-[11px] font-semibold mr-2">Risk Warnings</span>
              {top_long_warnings.map((w: string, i: number) => (
                <span key={i} className="text-yellow-300/80 text-[10px] mr-3">{w}</span>
              ))}
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-700 bg-[#111827]">
                  <th className="py-2 px-2 text-left text-gray-500 w-8">#</th>
                  <th className="py-2 px-2 text-left text-gray-500">Ticker</th>
                  <th className="py-2 px-2 text-left text-gray-500">Name</th>
                  <th className="py-2 px-2 text-left text-gray-500">Sector</th>
                  <th className="py-2 px-2 text-right text-gray-500">MCap$B</th>
                  <th className="py-2 px-2 text-left text-gray-500">Class</th>
                  <th className="py-2 px-2 text-center text-gray-500">Signal</th>
                  <th className="py-2 px-2 text-right text-gray-500">L#</th>
                  <th className="py-2 px-2 text-right text-gray-500">Conv</th>
                  <th className="py-2 px-2 text-right text-gray-500">O'Neil</th>
                  <th className="py-2 px-2 text-right text-gray-500">Minerv</th>
                  <th className="py-2 px-2 text-right text-gray-500">Wyck</th>
                  <th className="py-2 px-2 text-right text-gray-500">Ichi</th>
                  <th className="py-2 px-2 text-right text-gray-500">Darv</th>
                  <th className="py-2 px-2 text-right text-gray-500">Regime</th>
                  <th className="py-2 px-2 text-right text-gray-500">Flow</th>
                  <th className="py-2 px-2 text-right text-gray-500">RV</th>
                  <th className="py-2 px-2 text-right text-gray-500">Comp</th>
                  <th className="py-2 px-2 text-center text-gray-500">Risk</th>
                  <th className="py-2 px-2 text-right text-gray-500">SQ</th>
                  <th className="py-2 px-2 text-right text-gray-500">APS</th>
                  <th className="py-2 px-2 text-right text-gray-500">TCS</th>
                  <th className="py-2 px-2 text-right text-gray-500">TFS</th>
                  <th className="py-2 px-2 text-right text-gray-500">RSS</th>
                  <th className="py-2 px-2 text-right text-gray-500">OER</th>
                  <th className="py-2 px-2 text-right text-gray-500">RSI</th>
                  <th className="py-2 px-2 text-right text-gray-500">Age</th>
                  <th className="py-2 px-2 text-right text-gray-500">Val%</th>
                  <th className="py-2 px-2 text-right text-gray-500">1W</th>
                  <th className="py-2 px-2 text-right text-gray-500">1M</th>
                  <th className="py-2 px-2 text-right text-gray-500">3M</th>
                </tr>
              </thead>
              <tbody>
                {top_long.map((d: any, i: number) => {
                  const clsColor = CLASS_COLORS[d.classification] || C.gray;
                  const isEvent = d.event_flag;
                  const sq = d.structural_q;
                  const sqColor = sq >= 70 ? "text-green-400" : sq >= 40 ? "text-gray-300" : "text-yellow-400";
                  const sigColor: Record<string, string> = { STRONG_LONG: "#22c55e", LONG: "#86efac", STRONG_SHORT: "#ef4444", SHORT: "#fca5a5", NEUTRAL: "#6b7280" };
                  const sc = (v: any, isL = true) => {
                    const n = Number(v ?? 0);
                    const c = isL
                      ? (n >= 70 ? "text-green-400 font-bold" : n >= 50 ? "text-green-400/70" : n >= 30 ? "text-gray-400" : "text-gray-600")
                      : (n >= 70 ? "text-red-400 font-bold" : n >= 50 ? "text-red-400/70" : n >= 30 ? "text-gray-400" : "text-gray-600");
                    return <span className={`${c} text-[10px]`}>{n.toFixed(0)}</span>;
                  };
                  return (
                    <tr key={d.ticker} className={`border-b border-gray-800/50 hover:bg-[#1f2937]/30 ${isEvent ? "bg-yellow-900/10" : ""}`}>
                      <td className="py-1.5 px-2 text-gray-600">{i + 1}</td>
                      <td className="py-1.5 px-2 font-mono font-bold text-white">
                        {d.ticker}
                        {isEvent && <span className="ml-1 text-yellow-400 text-[9px]" title={d.event_reasons}>⚡</span>}
                      </td>
                      <td className="py-1.5 px-2 text-gray-400 truncate max-w-[140px]">{String(d.name).slice(0, 22)}</td>
                      <td className="py-1.5 px-2 text-cyan-400/70 text-[10px]">{d.sector}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400 text-[10px]">{d.mktcap_B != null && Number(d.mktcap_B) > 0 ? (Number(d.mktcap_B) >= 1 ? Number(d.mktcap_B).toFixed(1) : Number(d.mktcap_B).toFixed(2)) : "-"}</td>
                      <td className="py-1.5 px-2 text-[10px] font-medium" style={{ color: clsColor }}>{String(d.classification).replace(/^.\s/, "")}</td>
                      <td className="py-1.5 px-2 text-center"><span style={{ color: sigColor[d.net_signal] || "#6b7280" }} className="text-[10px] font-bold">{d.net_signal || "-"}</span></td>
                      <td className="py-1.5 px-2 text-right text-gray-300 text-[10px]">{d.long_count ?? "-"}</td>
                      <td className="py-1.5 px-2 text-right text-gray-300 text-[10px]">{d.conviction != null ? Number(d.conviction).toFixed(1) : "-"}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.oneil_long)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.minervini_long)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.wyckoff_long)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.ichimoku_long)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.darvas_long)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.regime_long)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.flow_long)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.relval_long)}</td>
                      <td className="py-1.5 px-2 text-right">{Number(d.composite).toFixed(1)}</td>
                      <td className="py-1.5 px-2 text-center text-[9px]">
                        {isEvent
                          ? <span className="text-yellow-400 font-semibold" title={d.event_reasons}>EVENT</span>
                          : <span className="text-gray-600">—</span>}
                      </td>
                      <td className={`py-1.5 px-2 text-right ${sqColor}`}>{sq != null ? sq : "-"}</td>
                      <td className={`py-1.5 px-2 text-right text-[10px] ${Number(d.alpha_potential ?? 0) >= 70 ? "text-emerald-400 font-bold" : Number(d.alpha_potential ?? 0) >= 50 ? "text-emerald-400/70" : "text-gray-400"}`}>{d.alpha_potential != null ? Number(d.alpha_potential).toFixed(0) : "-"}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.tcs}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.tfs}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.rss}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.oer}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{Number(d.rsi).toFixed(1)}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.trend_age}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.val_prob != null ? Number(d.val_prob).toFixed(1) : "-"}</td>
                      <td className={`py-1.5 px-2 text-right ${Number(d.ret_1w) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_1w).toFixed(2)}%</td>
                      <td className={`py-1.5 px-2 text-right ${Number(d.ret_1m) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_1m).toFixed(2)}%</td>
                      <td className={`py-1.5 px-2 text-right ${Number(d.ret_3m) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_3m).toFixed(2)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-gray-500 mt-2">
            ⚡ EVENT = 이벤트 드리븐 스파이크 감지 (단기 수익 &gt; 중기 + 거래량 급증 + 초기 추세 + 변동성 확대 중 2개+ 충족). 랭킹 가중치 ×0.7 할인 적용.
            SQ = Structural Quality (구조적 모멘텀 품질 0-100): vol_adj_mom(30) + trend_age(25) + VCR 안정성(20) + 12M 모멘텀(25). 높을수록 지속 가능한 추세.
          </p>
        </div>
      )}

      {/* ═══ Bottom 10 Strong Short ═══ */}
      {top_short && top_short.length > 0 && (
        <div className="mt-8">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
            Bottom 10 Strong Short — Multi-Strategy Consensus
          </h3>
          <div className="mb-3 p-3 bg-[#111827] border border-gray-800 rounded-lg text-[10px] text-gray-400 space-y-2">
            <div className="text-[11px] font-semibold text-red-400 mb-1">Selection Logic</div>
            <div className="grid grid-cols-[80px_1fr] gap-y-1">
              <span className="text-gray-500 font-semibold">Step 1</span>
              <span><span className="text-gray-300">후보 필터링</span> — classification ∈ &#123;DOWNTREND, WEAKENING, FADING, EXHAUSTING, COUNTER_RALLY, CYCLE_PEAK&#125;</span>
              <span className="text-gray-500 font-semibold">Step 2</span>
              <span><span className="text-gray-300">랭킹 점수</span> — <span className="text-red-400 font-mono">_short_rank = combined_short × 0.40 + (100 - composite) × 0.35 + (100 - APS) × 0.25</span></span>
              <span className="text-gray-500 font-semibold">Step 3</span>
              <span><span className="text-gray-300">섹터 집중 제한</span> — 동일 섹터 최대 3종목</span>
              <span className="text-gray-500 font-semibold">Step 4</span>
              <span><span className="text-gray-300">상위 10개 선택</span> — _short_rank 내림차순 상위 10</span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-700 bg-[#111827]">
                  <th className="py-2 px-2 text-left text-gray-500 w-8">#</th>
                  <th className="py-2 px-2 text-left text-gray-500">Ticker</th>
                  <th className="py-2 px-2 text-left text-gray-500">Name</th>
                  <th className="py-2 px-2 text-left text-gray-500">Sector</th>
                  <th className="py-2 px-2 text-right text-gray-500">MCap$B</th>
                  <th className="py-2 px-2 text-left text-gray-500">Class</th>
                  <th className="py-2 px-2 text-center text-gray-500">Signal</th>
                  <th className="py-2 px-2 text-right text-gray-500">S#</th>
                  <th className="py-2 px-2 text-right text-gray-500">Conv</th>
                  <th className="py-2 px-2 text-right text-gray-500">O'Neil</th>
                  <th className="py-2 px-2 text-right text-gray-500">Minerv</th>
                  <th className="py-2 px-2 text-right text-gray-500">Wyck</th>
                  <th className="py-2 px-2 text-right text-gray-500">Ichi</th>
                  <th className="py-2 px-2 text-right text-gray-500">Darv</th>
                  <th className="py-2 px-2 text-right text-gray-500">Regime</th>
                  <th className="py-2 px-2 text-right text-gray-500">Flow</th>
                  <th className="py-2 px-2 text-right text-gray-500">RV</th>
                  <th className="py-2 px-2 text-right text-gray-500">Comp</th>
                  <th className="py-2 px-2 text-right text-gray-500">SQ</th>
                  <th className="py-2 px-2 text-right text-gray-500">APS</th>
                  <th className="py-2 px-2 text-right text-gray-500">TCS</th>
                  <th className="py-2 px-2 text-right text-gray-500">TFS</th>
                  <th className="py-2 px-2 text-right text-gray-500">RSS</th>
                  <th className="py-2 px-2 text-right text-gray-500">OER</th>
                  <th className="py-2 px-2 text-right text-gray-500">RSI</th>
                  <th className="py-2 px-2 text-right text-gray-500">Age</th>
                  <th className="py-2 px-2 text-right text-gray-500">1W</th>
                  <th className="py-2 px-2 text-right text-gray-500">1M</th>
                  <th className="py-2 px-2 text-right text-gray-500">3M</th>
                </tr>
              </thead>
              <tbody>
                {top_short.map((d: any, i: number) => {
                  const clsColor = CLASS_COLORS[d.classification] || C.gray;
                  const sq = d.structural_q;
                  const sqColor = sq >= 70 ? "text-green-400" : sq >= 40 ? "text-gray-300" : "text-yellow-400";
                  const sigColor: Record<string, string> = { STRONG_LONG: "#22c55e", LONG: "#86efac", STRONG_SHORT: "#ef4444", SHORT: "#fca5a5", NEUTRAL: "#6b7280" };
                  const sc = (v: any, isL = false) => {
                    const n = Number(v ?? 0);
                    const c = isL
                      ? (n >= 70 ? "text-green-400 font-bold" : n >= 50 ? "text-green-400/70" : n >= 30 ? "text-gray-400" : "text-gray-600")
                      : (n >= 70 ? "text-red-400 font-bold" : n >= 50 ? "text-red-400/70" : n >= 30 ? "text-gray-400" : "text-gray-600");
                    return <span className={`${c} text-[10px]`}>{n.toFixed(0)}</span>;
                  };
                  return (
                    <tr key={d.ticker} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
                      <td className="py-1.5 px-2 text-gray-600">{i + 1}</td>
                      <td className="py-1.5 px-2 font-mono font-bold text-white">{d.ticker}</td>
                      <td className="py-1.5 px-2 text-gray-400 truncate max-w-[140px]">{String(d.name).slice(0, 22)}</td>
                      <td className="py-1.5 px-2 text-cyan-400/70 text-[10px]">{d.sector}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400 text-[10px]">{d.mktcap_B != null && Number(d.mktcap_B) > 0 ? (Number(d.mktcap_B) >= 1 ? Number(d.mktcap_B).toFixed(1) : Number(d.mktcap_B).toFixed(2)) : "-"}</td>
                      <td className="py-1.5 px-2 text-[10px] font-medium" style={{ color: clsColor }}>{String(d.classification).replace(/^.\s/, "")}</td>
                      <td className="py-1.5 px-2 text-center"><span style={{ color: sigColor[d.net_signal] || "#6b7280" }} className="text-[10px] font-bold">{d.net_signal || "-"}</span></td>
                      <td className="py-1.5 px-2 text-right text-gray-300 text-[10px]">{d.short_count ?? "-"}</td>
                      <td className="py-1.5 px-2 text-right text-gray-300 text-[10px]">{d.conviction != null ? Number(d.conviction).toFixed(1) : "-"}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.oneil_short)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.minervini_short)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.wyckoff_short)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.ichimoku_short)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.darvas_short)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.regime_short)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.flow_short)}</td>
                      <td className="py-1.5 px-2 text-right">{sc(d.relval_short)}</td>
                      <td className="py-1.5 px-2 text-right">{Number(d.composite).toFixed(1)}</td>
                      <td className={`py-1.5 px-2 text-right ${sqColor}`}>{sq != null ? sq : "-"}</td>
                      <td className={`py-1.5 px-2 text-right text-[10px] ${Number(d.alpha_potential ?? 0) >= 70 ? "text-emerald-400 font-bold" : Number(d.alpha_potential ?? 0) >= 50 ? "text-emerald-400/70" : "text-gray-400"}`}>{d.alpha_potential != null ? Number(d.alpha_potential).toFixed(0) : "-"}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.tcs}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.tfs}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.rss}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.oer}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{Number(d.rsi).toFixed(1)}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{d.trend_age}</td>
                      <td className={`py-1.5 px-2 text-right ${Number(d.ret_1w) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_1w).toFixed(2)}%</td>
                      <td className={`py-1.5 px-2 text-right ${Number(d.ret_1m) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_1m).toFixed(2)}%</td>
                      <td className={`py-1.5 px-2 text-right ${Number(d.ret_3m) > 0 ? "text-green-400" : "text-red-400"}`}>{Number(d.ret_3m).toFixed(2)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══ ETF / Stock Split: Top 10 Long & Short ═══ */}
      {((top_long_etf && top_long_etf.length > 0) || (top_long_stock && top_long_stock.length > 0) ||
        (top_short_etf && top_short_etf.length > 0) || (top_short_stock && top_short_stock.length > 0)) && (
        <div className="mt-8 space-y-6">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
            Asset Type Breakdown — ETF vs Stock
          </h3>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {/* ETF Long */}
            {top_long_etf && top_long_etf.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wide">ETF Top 10 Long</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-900/30 text-cyan-400">{top_long_etf.length}</span>
                </div>
                <SignalTable rows={top_long_etf} direction="long" />
              </div>
            )}

            {/* Stock Long */}
            {top_long_stock && top_long_stock.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-green-400 uppercase tracking-wide">Stock Top 10 Long</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-900/30 text-green-400">{top_long_stock.length}</span>
                </div>
                <SignalTable rows={top_long_stock} direction="long" />
              </div>
            )}

            {/* ETF Short */}
            {top_short_etf && top_short_etf.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-red-400 uppercase tracking-wide">ETF Top 10 Short</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-900/30 text-red-400">{top_short_etf.length}</span>
                </div>
                <SignalTable rows={top_short_etf} direction="short" />
              </div>
            )}

            {/* Stock Short */}
            {top_short_stock && top_short_stock.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-orange-400 uppercase tracking-wide">Stock Top 10 Short</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-orange-900/30 text-orange-400">{top_short_stock.length}</span>
                </div>
                <SignalTable rows={top_short_stock} direction="short" />
              </div>
            )}
          </div>
        </div>
      )}

      {/* ═══ Top-Long Backtest ═══ */}
      {top_long_bt && top_long_bt.length > 0 && (() => {
        const snapshots: any[] = [...top_long_bt].sort((a: any, b: any) => b.eval_date.localeCompare(a.eval_date));
        const periods = ["1W", "1M", "3M"] as const;
        // Grand summary across all snapshots (including CUM)
        const allPeriods = ["1W", "1M", "3M", "CUM"];
        const grandRet: Record<string, number[]> = {"1W": [], "1M": [], "3M": [], "CUM": []};
        snapshots.forEach((snap: any) => {
          allPeriods.forEach(p => {
            const s = snap.summary?.[p];
            if (s && s.avg_ret !== 0) grandRet[p].push(s.avg_ret);
          });
        });
        const grandHit: Record<string, number[]> = {"1W": [], "1M": [], "3M": [], "CUM": []};
        snapshots.forEach((snap: any) => {
          allPeriods.forEach(p => {
            const s = snap.summary?.[p];
            if (s && s.hit_rate !== 0) grandHit[p].push(s.hit_rate);
          });
        });
        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
        const retColor = (v: number | null) => v == null ? "text-gray-600" : v > 0 ? "text-green-400" : v < 0 ? "text-red-400" : "text-gray-400";
        const hitColor = (v: number) => v >= 60 ? "text-green-400" : v >= 50 ? "text-gray-300" : "text-red-400";
        const fmtRet = (v: number | null) => v != null ? `${v.toFixed(2)}%` : "—";
        return (
          <div className="mt-6">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-2">
              Top-Long Backtest — Monthly Replay (Past 12 Months)
            </h3>
            <p className="text-[10px] text-gray-500 mb-3">
              매월 시점별로 Top 10 Strong Long을 선정하고 개별 종목의 1W/1M/3M forward return을 측정.
              Look-ahead bias 없음: 각 시점에서 해당 시점까지의 데이터만으로 시그널 산출. 행 클릭으로 개별 종목 상세 확인.
            </p>

            {snapshots.map((snap: any) => {
              const s = snap.summary || {};
              const sw = s["1W"] || {}; const sm = s["1M"] || {}; const sq = s["3M"] || {}; const sc = s["CUM"] || {};
              return (
                <details key={snap.eval_date} className="mb-1 border border-gray-800 rounded">
                  {/* Summary row as <summary> */}
                  <summary className="cursor-pointer px-3 py-2 hover:bg-[#1f2937]/50 flex items-center gap-2 text-xs">
                    <span className="font-mono text-gray-300 w-24">{snap.eval_date}</span>
                    <span className="text-gray-500 w-8 text-center">{snap.n_picks}</span>
                    <span className="text-gray-600 text-[10px] w-6 border-l border-gray-700 pl-2">1W</span>
                    <span className={`w-16 text-right ${retColor(sw.avg_ret)}`}>{fmtRet(sw.avg_ret)}</span>
                    <span className={`w-10 text-right text-[10px] ${hitColor(sw.hit_rate || 0)}`}>{sw.hit_rate != null ? `${sw.hit_rate.toFixed(0)}%` : "—"}</span>
                    <span className="text-gray-600 text-[10px] w-6 border-l border-gray-700 pl-2">1M</span>
                    <span className={`w-16 text-right ${retColor(sm.avg_ret)}`}>{fmtRet(sm.avg_ret)}</span>
                    <span className={`w-10 text-right text-[10px] ${hitColor(sm.hit_rate || 0)}`}>{sm.hit_rate != null ? `${sm.hit_rate.toFixed(0)}%` : "—"}</span>
                    <span className="text-gray-600 text-[10px] w-6 border-l border-gray-700 pl-2">3M</span>
                    <span className={`w-16 text-right ${retColor(sq.avg_ret)}`}>{fmtRet(sq.avg_ret)}</span>
                    <span className={`w-10 text-right text-[10px] ${hitColor(sq.hit_rate || 0)}`}>{sq.hit_rate != null ? `${sq.hit_rate.toFixed(0)}%` : "—"}</span>
                    <span className="text-amber-400/70 text-[10px] w-8 border-l border-gray-700 pl-2">CUM</span>
                    <span className={`w-16 text-right font-semibold ${retColor(sc.avg_ret)}`}>{fmtRet(sc.avg_ret)}</span>
                    <span className={`w-10 text-right text-[10px] ${hitColor(sc.hit_rate || 0)}`}>{sc.hit_rate != null ? `${sc.hit_rate.toFixed(0)}%` : "—"}</span>
                  </summary>
                  {/* Per-ticker detail — matches Top 10 Long/Short column layout */}
                  <div className="px-3 pb-2 overflow-x-auto">
                    <table className="w-full text-[11px] border-collapse">
                      <thead>
                        <tr className="border-b border-gray-700">
                          <th className="py-1 px-1.5 text-left text-gray-500">#</th>
                          <th className="py-1 px-1.5 text-left text-gray-500">Ticker</th>
                          <th className="py-1 px-1.5 text-left text-gray-500">Name</th>
                          <th className="py-1 px-1.5 text-left text-gray-500">Sector</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">MCap$B</th>
                          <th className="py-1 px-1.5 text-left text-gray-500">Class</th>
                          <th className="py-1 px-1.5 text-center text-gray-500">Signal</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">L#</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Conv</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">O'Neil</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Minerv</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Wyck</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Ichi</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Darv</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Regime</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Flow</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">RV</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Comp</th>
                          <th className="py-1 px-1.5 text-center text-gray-500">Risk</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">SQ</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">APS</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">TCS</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">TFS</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">RSS</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">OER</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">RSI</th>
                          <th className="py-1 px-1.5 text-right text-gray-500">Age</th>
                          <th className="py-1 px-1.5 text-right text-cyan-400/60 border-l border-gray-700">1W Fwd</th>
                          <th className="py-1 px-1.5 text-right text-blue-400/60 border-l border-gray-700">1M Fwd</th>
                          <th className="py-1 px-1.5 text-right text-purple-400/60 border-l border-gray-700">3M Fwd</th>
                          <th className="py-1 px-1.5 text-right text-amber-400/60 border-l border-gray-700">CUM</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(snap.tickers || []).map((t: any, idx: number) => {
                          const n = (v: any) => v != null && !isNaN(Number(v)) ? Number(v) : null;
                          const s = (v: any) => v != null ? String(v) : "";
                          const sqv = n(t.structural_q);
                          const sqc = sqv != null ? (sqv >= 70 ? "text-green-400" : sqv >= 40 ? "text-gray-300" : "text-yellow-400") : "text-gray-600";
                          const isEv = !!t.event_flag;
                          const fmt0 = (v: any) => { const x = n(v); return x != null ? String(Math.round(x)) : "—"; };
                          const fmt1 = (v: any) => { const x = n(v); return x != null ? x.toFixed(1) : "—"; };
                          const sigColorMap: Record<string, string> = { STRONG_LONG: "#22c55e", LONG: "#86efac", STRONG_SHORT: "#ef4444", SHORT: "#fca5a5", NEUTRAL: "#6b7280" };
                          const sigStr = s(t.net_signal) || "NEUTRAL";
                          const scL = (v: any) => {
                            const x = Number(v ?? 0);
                            const c = x >= 70 ? "text-green-400 font-bold" : x >= 50 ? "text-green-400/70" : x >= 30 ? "text-gray-400" : "text-gray-600";
                            return <span className={`${c} text-[10px]`}>{x.toFixed(0)}</span>;
                          };
                          return (
                          <tr key={t.ticker} className={`border-b border-gray-800/30 ${isEv ? "bg-yellow-900/10" : ""}`}>
                            <td className="py-1 px-1.5 text-gray-600">{idx + 1}</td>
                            <td className="py-1 px-1.5 font-mono font-bold text-white">
                              {t.ticker}{isEv && <span className="ml-0.5 text-yellow-400 text-[9px]" title={s(t.event_reasons)}>⚡</span>}
                            </td>
                            <td className="py-1 px-1.5 text-gray-400 truncate max-w-[130px]">{s(t.name).slice(0, 22) || "—"}</td>
                            <td className="py-1 px-1.5 text-cyan-400/70 text-[10px]">{s(t.sector) || s(t.category) || "—"}</td>
                            <td className="py-1 px-1.5 text-right text-gray-400 text-[10px]">{t.mktcap_B > 0 ? (t.mktcap_B >= 1 ? Number(t.mktcap_B).toFixed(1) : Number(t.mktcap_B).toFixed(2)) : "—"}</td>
                            <td className="py-1 px-1.5 text-gray-400 text-[10px]">{s(t.cls) || "—"}</td>
                            <td className="py-1 px-1.5 text-center"><span style={{ color: sigColorMap[sigStr] || "#6b7280" }} className="text-[10px] font-bold">{sigStr}</span></td>
                            <td className="py-1 px-1.5 text-right text-gray-300 text-[10px]">{t.long_count ?? "—"}</td>
                            <td className="py-1 px-1.5 text-right text-gray-300 text-[10px]">{t.conviction != null ? Number(t.conviction).toFixed(1) : "—"}</td>
                            <td className="py-1 px-1.5 text-right">{scL(t.oneil_long)}</td>
                            <td className="py-1 px-1.5 text-right">{scL(t.minervini_long)}</td>
                            <td className="py-1 px-1.5 text-right">{scL(t.wyckoff_long)}</td>
                            <td className="py-1 px-1.5 text-right">{scL(t.ichimoku_long)}</td>
                            <td className="py-1 px-1.5 text-right">{scL(t.darvas_long)}</td>
                            <td className="py-1 px-1.5 text-right">{scL(t.regime_long)}</td>
                            <td className="py-1 px-1.5 text-right">{scL(t.flow_long)}</td>
                            <td className="py-1 px-1.5 text-right">{scL(t.relval_long)}</td>
                            <td className="py-1 px-1.5 text-right text-gray-300">{fmt1(t.comp)}</td>
                            <td className="py-1 px-1.5 text-center text-[9px]">
                              {isEv ? <span className="text-yellow-400 font-semibold">EVENT</span> : <span className="text-gray-700">—</span>}
                            </td>
                            <td className={`py-1 px-1.5 text-right ${sqc}`}>{fmt0(sqv)}</td>
                            <td className={`py-1 px-1.5 text-right text-[10px] ${Number(t.alpha_potential ?? 0) >= 70 ? "text-emerald-400 font-bold" : Number(t.alpha_potential ?? 0) >= 50 ? "text-emerald-400/70" : "text-gray-400"}`}>{fmt0(t.alpha_potential)}</td>
                            <td className="py-1 px-1.5 text-right text-gray-400">{fmt0(t.tcs)}</td>
                            <td className="py-1 px-1.5 text-right text-gray-400">{fmt0(t.tfs)}</td>
                            <td className="py-1 px-1.5 text-right text-gray-400">{fmt0(t.rss)}</td>
                            <td className="py-1 px-1.5 text-right text-gray-400">{fmt0(t.oer)}</td>
                            <td className="py-1 px-1.5 text-right text-gray-400">{fmt1(t.rsi)}</td>
                            <td className="py-1 px-1.5 text-right text-gray-400">{fmt0(t.trend_age)}</td>
                            <td className={`py-1 px-1.5 text-right border-l border-gray-800 ${retColor(t.ret_1W)}`}>{fmtRet(t.ret_1W)}</td>
                            <td className={`py-1 px-1.5 text-right border-l border-gray-800 ${retColor(t.ret_1M)}`}>{fmtRet(t.ret_1M)}</td>
                            <td className={`py-1 px-1.5 text-right border-l border-gray-800 ${retColor(t.ret_3M)}`}>{fmtRet(t.ret_3M)}</td>
                            <td className={`py-1 px-1.5 text-right border-l border-gray-800 font-semibold ${retColor(t.ret_CUM)}`}>{fmtRet(t.ret_CUM)}</td>
                          </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </details>
              );
            })}

            {/* Grand Summary */}
            <div className="mt-2 px-3 py-2 bg-[#111827] border border-gray-700 rounded flex items-center gap-3 text-xs font-semibold">
              <span className="text-cyan-400 w-24">Average</span>
              <span className="text-gray-500 w-8 text-center">—</span>
              <span className="text-gray-600 text-[10px] w-6 border-l border-gray-700 pl-2">1W</span>
              <span className={`w-16 text-right ${retColor(avg(grandRet["1W"]))}`}>{avg(grandRet["1W"]).toFixed(2)}%</span>
              <span className={`w-10 text-right text-[10px] ${hitColor(avg(grandHit["1W"]))}`}>{avg(grandHit["1W"]).toFixed(1)}%</span>
              <span className="text-gray-600 text-[10px] w-6 border-l border-gray-700 pl-2">1M</span>
              <span className={`w-16 text-right ${retColor(avg(grandRet["1M"]))}`}>{avg(grandRet["1M"]).toFixed(2)}%</span>
              <span className={`w-10 text-right text-[10px] ${hitColor(avg(grandHit["1M"]))}`}>{avg(grandHit["1M"]).toFixed(1)}%</span>
              <span className="text-gray-600 text-[10px] w-6 border-l border-gray-700 pl-2">3M</span>
              <span className={`w-16 text-right ${retColor(avg(grandRet["3M"]))}`}>{avg(grandRet["3M"]).toFixed(2)}%</span>
              <span className={`w-10 text-right text-[10px] ${hitColor(avg(grandHit["3M"]))}`}>{avg(grandHit["3M"]).toFixed(1)}%</span>
              <span className="text-amber-400/70 text-[10px] w-8 border-l border-gray-700 pl-2">CUM</span>
              <span className={`w-16 text-right font-bold ${retColor(avg(grandRet["CUM"]))}`}>{avg(grandRet["CUM"]).toFixed(2)}%</span>
              <span className={`w-10 text-right text-[10px] ${hitColor(avg(grandHit["CUM"]))}`}>{avg(grandHit["CUM"]).toFixed(1)}%</span>
            </div>

            <p className="text-[10px] text-gray-500 mt-2">
              각 행 클릭 시 개별 종목의 Ticker, Class, Long Score, Composite, 1W/1M/3M/CUM forward return 확인 가능. CUM = eval_date 시점 진입가 → 최신 데이터 날짜까지 누적 수익률.
              Hit% = 양수 수익 비율. Average = 전체 기간 평균.
            </p>
          </div>
        );
      })()}

      {/* ═══ Top-Long Pattern Analysis ═══ */}
      {top_long_bt && top_long_bt.length >= 2 && top_long && top_long.length > 0 && (() => {
        const snaps: any[] = [...top_long_bt].sort((a: any, b: any) => a.eval_date.localeCompare(b.eval_date));
        const dates = snaps.map((s: any) => s.eval_date);

        // 1. Ticker frequency across all snapshots
        const tkFreq: Record<string, { count: number; rets1M: number[]; cats: Set<string> }> = {};
        snaps.forEach((snap: any) => {
          (snap.tickers || []).forEach((t: any) => {
            const tk = t.ticker;
            if (!tkFreq[tk]) tkFreq[tk] = { count: 0, rets1M: [], cats: new Set() };
            tkFreq[tk].count++;
            if (t.ret_1M != null) tkFreq[tk].rets1M.push(t.ret_1M);
            if (t.category) tkFreq[tk].cats.add(t.category);
          });
        });

        // Current top_long tickers
        const currentTickers = new Set(top_long.map((d: any) => d.ticker));

        // Sort by frequency descending, take top 25
        const freqList = Object.entries(tkFreq)
          .map(([tk, v]) => ({ ticker: tk, count: v.count, avgRet: v.rets1M.length > 0 ? v.rets1M.reduce((a, b) => a + b, 0) / v.rets1M.length : 0, cat: [...v.cats].join("/"), inCurrent: currentTickers.has(tk) }))
          .sort((a, b) => b.count - a.count)
          .slice(0, 25);

        // 2. Heatmap data: tickers (y) x dates (x), value = 1M return or null
        const heatTickers = freqList.map(f => f.ticker);
        const heatZ: (number | null)[][] = [];
        const heatText: string[][] = [];
        heatTickers.forEach(tk => {
          const row: (number | null)[] = [];
          const textRow: string[] = [];
          dates.forEach(d => {
            const snap = snaps.find((s: any) => s.eval_date === d);
            const t = (snap?.tickers || []).find((t: any) => t.ticker === tk);
            if (t) {
              row.push(t.ret_1M != null ? t.ret_1M : 0);
              textRow.push(`${tk}<br>${d}<br>1M: ${t.ret_1M != null ? t.ret_1M.toFixed(1) + "%" : "N/A"}<br>Long: ${t.oneil_long || "?"}`);
            } else {
              row.push(null as any);
              textRow.push("");
            }
          });
          heatZ.push(row);
          heatText.push(textRow);
        });

        // 3. Category rotation: count by category per date
        const allCats = new Set<string>();
        snaps.forEach((s: any) => (s.tickers || []).forEach((t: any) => { if (t.category) allCats.add(t.category); }));
        const catList = [...allCats].sort();
        const catTraces = catList.map(cat => ({
          type: "bar" as const,
          name: cat.replace("STK_", ""),
          x: dates,
          y: dates.map(d => {
            const snap = snaps.find((s: any) => s.eval_date === d);
            return (snap?.tickers || []).filter((t: any) => t.category === cat).length;
          }),
        }));

        // 4. Recurrence vs Performance scatter
        const recurData = freqList.filter(f => f.count >= 1);

        return (
          <div className="mt-8 space-y-6">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide">
              Top-Long Pattern Analysis — Backtest vs Current
            </h3>
            <p className="text-[10px] text-gray-500">
              백테스트 12개월 선정 이력과 현재 Top 10 간 반복 출현(persistence), 카테고리 로테이션, 반복 선정과 수익률 상관관계를 시각화합니다.
            </p>

            {/* Heatmap: Ticker Persistence */}
            <div>
              <h4 className="text-xs font-semibold text-gray-400 mb-1">Ticker Persistence Heatmap</h4>
              <p className="text-[10px] text-gray-500 mb-2">
                X=평가월, Y=종목(출현 빈도순). 셀 색상=1M Forward Return. 빈 셀=해당 월 미선정.
                ★ = 현재 Top 10에 포함된 종목.
              </p>
              <Plot
                data={[{
                  type: "heatmap",
                  x: dates,
                  y: heatTickers.map(tk => currentTickers.has(tk) ? `★ ${tk}` : tk),
                  z: heatZ,
                  text: heatText,
                  hovertemplate: "%{text}<extra></extra>",
                  colorscale: [[0, "#ef4444"], [0.4, "#1f2937"], [0.5, "#374151"], [0.6, "#1f2937"], [1, "#22c55e"]],
                  zmid: 0,
                  colorbar: { title: "1M Ret%", titleside: "right", len: 0.5 },
                  xgap: 2, ygap: 2,
                } as any]}
                layout={{
                  ...DARK_LAYOUT,
                  height: Math.max(300, heatTickers.length * 22 + 80),
                  margin: { t: 20, b: 60, l: 90, r: 80 },
                  xaxis: { color: C.gray, tickangle: -45 },
                  yaxis: { color: C.gray, autorange: "reversed" as any, tickfont: { size: 10 } },
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
            </div>

            <div className="grid grid-cols-2 gap-6">
              {/* Category Rotation Stacked Bar */}
              <div>
                <h4 className="text-xs font-semibold text-gray-400 mb-1">Category Rotation</h4>
                <p className="text-[10px] text-gray-500 mb-2">월별 Top 10 내 카테고리 구성 변화. 특정 카테고리 집중/분산 추이 확인.</p>
                <Plot
                  data={catTraces}
                  layout={{
                    ...DARK_LAYOUT, barmode: "stack", height: 350,
                    margin: { t: 20, b: 60, l: 40, r: 10 },
                    xaxis: { color: C.gray, tickangle: -45 },
                    yaxis: { title: "# Picks", color: C.gray },
                    legend: { font: { size: 8 }, x: 1.02, y: 1 },
                  }}
                  config={{ responsive: true }} style={{ width: "100%" }}
                />
              </div>

              {/* Recurrence vs Performance Scatter */}
              <div>
                <h4 className="text-xs font-semibold text-gray-400 mb-1">Recurrence vs 1M Performance</h4>
                <p className="text-[10px] text-gray-500 mb-2">X=선정 횟수(12개월 중), Y=평균 1M Return. 반복 선정 종목이 더 높은 수익을 내는지 확인.</p>
                <Plot
                  data={[
                    {
                      type: "scatter", mode: "markers+text",
                      x: recurData.map(d => d.count),
                      y: recurData.map(d => d.avgRet),
                      text: recurData.map(d => d.ticker),
                      textposition: "top center",
                      textfont: { size: 8, color: recurData.map(d => d.inCurrent ? "#06b6d4" : "#6b7280") },
                      marker: {
                        size: recurData.map(d => d.inCurrent ? 12 : 7),
                        color: recurData.map(d => d.avgRet),
                        colorscale: [[0, "#ef4444"], [0.5, "#374151"], [1, "#22c55e"]],
                        cmid: 0,
                        symbol: recurData.map(d => d.inCurrent ? "diamond" : "circle"),
                        line: { color: recurData.map(d => d.inCurrent ? "#06b6d4" : "transparent"), width: 2 },
                      },
                      hovertemplate: "%{text}<br>선정 %{x}회<br>Avg 1M: %{y:.1f}%<extra></extra>",
                    },
                  ]}
                  layout={{
                    ...DARK_LAYOUT, height: 350,
                    margin: { t: 20, b: 40, l: 50, r: 20 },
                    xaxis: { title: "# Times Selected (12mo)", color: C.gray, dtick: 1 },
                    yaxis: { title: "Avg 1M Return %", color: C.gray },
                    shapes: [
                      { type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray, dash: "dash", width: 1 } },
                    ],
                  }}
                  config={{ responsive: true }} style={{ width: "100%" }}
                />
              </div>
            </div>

            {/* Frequency Table */}
            <div>
              <h4 className="text-xs font-semibold text-gray-400 mb-1">Ticker Frequency Summary</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-[11px] border-collapse">
                  <thead>
                    <tr className="border-b border-gray-700 bg-[#111827]">
                      <th className="py-1.5 px-2 text-left text-gray-500">Ticker</th>
                      <th className="py-1.5 px-2 text-left text-gray-500">Category</th>
                      <th className="py-1.5 px-2 text-right text-gray-500">Freq</th>
                      <th className="py-1.5 px-2 text-right text-gray-500">Avg 1M Ret</th>
                      <th className="py-1.5 px-2 text-center text-gray-500">Current</th>
                      <th className="py-1.5 px-2 text-left text-gray-500">Pattern</th>
                    </tr>
                  </thead>
                  <tbody>
                    {freqList.map((f) => (
                      <tr key={f.ticker} className={`border-b border-gray-800/30 ${f.inCurrent ? "bg-cyan-900/10" : ""}`}>
                        <td className="py-1 px-2 font-mono font-bold text-gray-200">{f.ticker}</td>
                        <td className="py-1 px-2 text-gray-500 text-[10px]">{f.cat}</td>
                        <td className="py-1 px-2 text-right text-gray-300">{f.count}/{snaps.length}</td>
                        <td className={`py-1 px-2 text-right ${f.avgRet > 0 ? "text-green-400" : "text-red-400"}`}>{f.avgRet.toFixed(2)}%</td>
                        <td className="py-1 px-2 text-center">{f.inCurrent ? <span className="text-cyan-400">★</span> : ""}</td>
                        <td className="py-1 px-2 text-[10px] text-gray-500">
                          {f.count >= snaps.length * 0.7 ? "🔁 Persistent" :
                           f.count >= snaps.length * 0.4 ? "📈 Recurring" :
                           f.count === 1 && f.inCurrent ? "🆕 New Entry" :
                           f.count === 1 ? "⚡ One-shot" : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] text-gray-500 mt-2">
                🔁 Persistent = 70%+ 월 선정 | 📈 Recurring = 40%+ | 🆕 New Entry = 이번 처음 + 현재 포함 | ⚡ One-shot = 1회만 출현.
                ★ = 현재 Top 10 Strong Long에 포함. 시안 다이아몬드 = 현재 포함 종목.
              </p>
            </div>
          </div>
        );
      })()}

      {/* ═══ Market Snapshot ═══ */}
      <div className="mt-14">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Market Snapshot</h3>
        <div className="grid grid-cols-2 gap-8">
          <div>
            <Plot
              data={[{
                type: "pie", hole: 0.45,
                labels: classification_dist.map((d: any) => d.classification),
                values: classification_dist.map((d: any) => d.count),
                marker: { colors: classification_dist.map((d: any) => CLASS_COLORS[d.classification] || C.gray) },
                textinfo: "label+value", textfont: { size: 9 },
              }]}
              layout={{ ...DARK_LAYOUT, height: 340, showlegend: false, margin: { t: 20, b: 20, l: 20, r: 20 } }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
            <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
              전체 유니버스의 분류(Classification) 분포. 3×3 듀얼 타임프레임 매트릭스 + 오버라이드로 산출.
              CONTINUATION(초록)이 많을수록 시장 모멘텀이 강하고, DOWNTREND(빨강)/FADING이 많을수록 약세.
            </p>
          </div>
          <div>
            <Plot
              data={[{
                type: "histogram",
                x: composite_data.map((d: any) => d.composite),
                marker: { color: C.cyan, opacity: 0.7 },
                nbinsx: 25,
              }]}
              layout={{
                ...DARK_LAYOUT, height: 340,
                xaxis: { title: "Composite Score", color: C.gray }, yaxis: { title: "Count", color: C.gray },
                margin: { t: 20, b: 50, l: 50, r: 20 },
                shapes: [{ type: "line", x0: 55, x1: 55, y0: 0, y1: 1, yref: "paper", line: { color: C.orange, dash: "dot", width: 2 } }],
              }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
            <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
              Composite Score(0-100)의 전체 분포. 주황 점선(55)은 포트폴리오 적격 최소 기준.
              분포가 오른쪽에 몰리면 전반적 강세, 왼쪽이면 약세 시장.
            </p>
          </div>
        </div>
      </div>

      {/* ═══ Sector Overview ═══ */}
      <div className="mt-14">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Sector Overview</h3>
        <div className="grid grid-cols-2 gap-8">
          {data.sector_dist?.length > 0 && (() => {
            const sorted = [...data.sector_dist].sort((a: any, b: any) => a.avg_comp - b.avg_comp);
            return (
              <div>
                <Plot
                  data={[{
                    type: "bar", orientation: "h",
                    y: sorted.map((d: any) => d.sector),
                    x: sorted.map((d: any) => d.avg_comp),
                    marker: { color: sorted.map((d: any) => d.avg_comp), colorscale: "Viridis" },
                    text: sorted.map((d: any) => `${d.n} tickers`),
                    textposition: "outside", textfont: { size: 9 },
                  }]}
                  layout={{
                    ...DARK_LAYOUT, height: 420, margin: { t: 10, b: 40, l: 160, r: 60 },
                    xaxis: { title: "Avg Composite", color: C.gray },
                    shapes: [{ type: "line", x0: 55, x1: 55, y0: 0, y1: 1, yref: "paper", line: { color: C.orange, dash: "dot" } }],
                  }}
                  config={{ responsive: true }} style={{ width: "100%" }}
                />
                <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
                  GICS 기반 글로벌 섹터별 평균 Composite Score. 테마에서 자동 매핑되어 지역 무관 비교 가능.
                  주황 점선(55) 이상 섹터는 전반적 모멘텀 양호. 레이블은 해당 섹터 종목 수.
                </p>
              </div>
            );
          })()}
          {data.sector_dist?.length > 0 && (
            <div>
              <Plot
                data={[
                  { type: "bar", name: "1M", x: data.sector_dist.map((d: any) => d.sector), y: data.sector_dist.map((d: any) => d.avg_ret_1m), marker: { color: C.cyan, opacity: 0.8 } },
                  { type: "bar", name: "3M", x: data.sector_dist.map((d: any) => d.sector), y: data.sector_dist.map((d: any) => d.avg_ret_3m), marker: { color: C.purple, opacity: 0.8 } },
                ]}
                layout={{
                  ...DARK_LAYOUT, barmode: "group", height: 420,
                  margin: { t: 10, b: 80, l: 50, r: 20 },
                  xaxis: { tickangle: -40, color: C.gray }, yaxis: { title: "Return %", color: C.gray },
                  shapes: [{ type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray, dash: "dash" } }],
                  legend: { x: 0, y: 1, font: { size: 10 } },
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
              <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
                섹터별 평균 1개월(시안) / 3개월(보라) 수익률. 0선 위는 양의 수익, 아래는 손실.
                1M과 3M의 방향이 다르면 최근 모멘텀 전환이 진행 중일 수 있음.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ═══ Signal Profile ═══ */}
      <div className="mt-14">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Signal Profile</h3>
        <div className="grid grid-cols-3 gap-8">
          {data.rsi_data?.length > 0 && (
            <div>
              <Plot
                data={[{
                  type: "histogram",
                  x: data.rsi_data.map((d: any) => d.rsi),
                  marker: { color: C.blue, opacity: 0.7 },
                  nbinsx: 20,
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 280, margin: { t: 10, b: 40, l: 40, r: 10 },
                  xaxis: { color: C.gray }, yaxis: { color: C.gray },
                  shapes: [
                    { type: "line", x0: 70, x1: 70, y0: 0, y1: 1, yref: "paper", line: { color: C.red, dash: "dot" } },
                    { type: "line", x0: 30, x1: 30, y0: 0, y1: 1, yref: "paper", line: { color: C.green, dash: "dot" } },
                  ],
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
              <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
                RSI-14 분포. 빨간 점선(70) 이상 = 과매수 구간, 초록 점선(30) 이하 = 과매도 구간.
                분포가 70 근처에 집중되면 시장 과열, 30 근처면 공포 구간.
              </p>
            </div>
          )}
          {data.age_dist?.length > 0 && (
            <div>
              <Plot
                data={[{
                  type: "bar",
                  x: data.age_dist.map((d: any) => d.bin),
                  y: data.age_dist.map((d: any) => d.count),
                  marker: { color: data.age_dist.map((d: any) => {
                    const b = String(d.bin);
                    if (b === "0-5") return C.red;
                    if (b === "6-10" || b === "11-20") return C.orange;
                    if (b === "21-40") return C.yellow;
                    return C.green;
                  }) },
                  text: data.age_dist.map((d: any) => d.count), textposition: "outside", textfont: { size: 9 },
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 280, margin: { t: 10, b: 40, l: 40, r: 10 },
                  xaxis: { title: "days above SMA50", color: C.gray }, yaxis: { color: C.gray },
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
              <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
                SMA50 위에서 연속 유지된 거래일 수 분포. 빨강(0-5일) = 최근 하락 전환,
                초록(40일+) = 장기 상승 추세 유지. 빨강 비중이 높으면 시장 전반 추세 약화.
              </p>
            </div>
          )}
          {data.axis_by_sector?.length > 0 && (
            <div>
              <Plot
                data={data.axis_by_sector.slice(0, 6).map((s: any) => ({
                  type: "scatterpolar" as const,
                  r: [s.avg_tcs, s.avg_tfs, s.avg_oer, s.avg_rss, s.avg_tcs],
                  theta: ["TCS", "TFS", "OER", "RSS", "TCS"],
                  fill: "toself", name: s.sector, opacity: 0.25,
                }))}
                layout={{
                  ...DARK_LAYOUT, height: 280, margin: { t: 10, b: 10, l: 40, r: 40 },
                  polar: { bgcolor: C.panel, radialaxis: { visible: true, range: [0, 70], color: C.gray, tickfont: { size: 8 } }, angularaxis: { color: C.text, tickfont: { size: 9 } } },
                  showlegend: true, legend: { font: { size: 8 }, x: 1, y: 1 },
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
              <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
                상위 6개 섹터의 4축 시그널 프로필. TCS(추세 지속), TFS(추세 형성), OER(과열 리스크),
                RSS(상대 강도). 면적이 넓고 OER이 낮은 섹터가 이상적.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ═══ Top Picks & Movers ═══ */}
      <div className="mt-14">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Top Picks & Movers</h3>
        <div className="grid grid-cols-3 gap-8">
          {top_eligible.length > 0 && (
            <div>
              <Plot
                data={[{
                  type: "bar", orientation: "h",
                  y: top_eligible.slice(0, 10).map((d: any) => d.ticker).reverse(),
                  x: top_eligible.slice(0, 10).map((d: any) => d.composite).reverse(),
                  marker: { color: top_eligible.slice(0, 10).map((d: any) => CLASS_COLORS[d.classification] || C.cyan).reverse() },
                  text: top_eligible.slice(0, 10).map((d: any) => d.composite.toFixed(0)).reverse(),
                  textposition: "outside", textfont: { size: 9 },
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 320, margin: { t: 10, b: 30, l: 70, r: 40 },
                  xaxis: { color: C.gray },
                  shapes: [{ type: "line", x0: 55, x1: 55, y0: 0, y1: 1, yref: "paper", line: { color: C.orange, dash: "dot" } }],
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
              <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
                포트폴리오 적격 종목 중 Composite Score 상위 10개. 색상은 분류(Classification)를 나타냄.
                점선(55)은 적격 기준선.
              </p>
            </div>
          )}
          {data.top_movers_up?.length > 0 && (
            <div>
              <Plot
                data={[{
                  type: "bar", orientation: "h",
                  y: data.top_movers_up.slice(0, 8).map((d: any) => d.ticker).reverse(),
                  x: data.top_movers_up.slice(0, 8).map((d: any) => d.ret_1m).reverse(),
                  marker: { color: C.green, opacity: 0.8 },
                  text: data.top_movers_up.slice(0, 8).map((d: any) => `${d.ret_1m.toFixed(1)}%`).reverse(),
                  textposition: "outside", textfont: { size: 9 },
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 320, margin: { t: 10, b: 30, l: 70, r: 50 },
                  xaxis: { title: "1M Return %", color: C.gray },
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
              <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
                적격 종목 중 최근 1개월 수익률 상위 8개. 강한 모멘텀을 보이고 있는 종목들.
              </p>
            </div>
          )}
          {data.top_movers_dn?.length > 0 && (
            <div>
              <Plot
                data={[{
                  type: "bar", orientation: "h",
                  y: data.top_movers_dn.slice(0, 8).map((d: any) => d.ticker).reverse(),
                  x: data.top_movers_dn.slice(0, 8).map((d: any) => d.ret_1m).reverse(),
                  marker: { color: C.red, opacity: 0.8 },
                  text: data.top_movers_dn.slice(0, 8).map((d: any) => `${d.ret_1m.toFixed(1)}%`).reverse(),
                  textposition: "outside", textfont: { size: 9 },
                }]}
                layout={{
                  ...DARK_LAYOUT, height: 320, margin: { t: 10, b: 30, l: 70, r: 50 },
                  xaxis: { title: "1M Return %", color: C.gray },
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
              <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
                적격 종목 중 최근 1개월 수익률 하위 8개. 모멘텀 약화 또는 조정 중인 종목들.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ═══ Conviction Map ═══ */}
      <div className="mt-14">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Conviction Map</h3>
        <Plot
          data={[{
            type: "scatter", mode: "markers",
            x: conviction_bubble.map((d: any) => d.composite),
            y: conviction_bubble.map((d: any) => d.val_prob),
            text: conviction_bubble.map((d: any) => `${d.ticker}<br>${d.name}<br>ADV: ${d.adv_M >= 1000 ? (d.adv_M/1000).toFixed(0) + "B" : d.adv_M.toFixed(0) + "M"}`),
            marker: {
              size: conviction_bubble.map((d: any) => {
                const v = Math.max(1, d.adv_M || 1);
                return Math.min(40, Math.max(6, Math.log10(v) * 8));
              }),
              color: conviction_bubble.map((d: any) => CLASS_COLORS[d.classification] || C.gray),
              opacity: 0.65,
              line: { color: "rgba(255,255,255,0.15)", width: 1 },
            },
            hovertemplate: "%{text}<br>Comp=%{x:.1f}<br>Val=%{y:.1f}%<extra></extra>",
          }]}
          layout={{
            ...DARK_LAYOUT, height: 420,
            margin: { t: 10, b: 50, l: 50, r: 20 },
            xaxis: { title: "Composite Score", color: C.gray },
            yaxis: { title: "Validity Prob %", color: C.gray },
            shapes: [
              { type: "line", x0: 55, x1: 55, y0: 0, y1: 1, yref: "paper", line: { color: C.orange, dash: "dot", width: 1.5 } },
              { type: "line", y0: 50, y1: 50, x0: 0, x1: 1, xref: "paper", line: { color: C.gray, dash: "dash" } },
            ],
          }}
          config={{ responsive: true }} style={{ width: "100%" }}
        />
        <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
          X축 = Composite Score, Y축 = OOS 백테스트 기반 Validity Probability.
          버블 크기 = ADV(일평균 거래대금), 색상 = Classification. 우상단(고점수 + 고확률)이 최적 포지션.
          주황 점선 = 적격 기준(55), 회색 점선 = 확률 중립(50%).
        </p>
      </div>

      {/* ═══ Heatmap ═══ */}
      {data.cls_sector?.length > 0 && (() => {
        const sectors: string[] = [...new Set(data.cls_sector.map((d: any) => d.sector))] as string[];
        const classes: string[] = [...new Set(data.cls_sector.map((d: any) => d.classification))] as string[];
        const z = classes.map((cls: string) =>
          sectors.map((sec: string) => {
            const match = data.cls_sector.find((d: any) => d.sector === sec && d.classification === cls);
            return match ? match.count : 0;
          })
        );
        return (
          <div className="mt-14">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-4">Classification × Sector</h3>
            <Plot
              data={[{
                type: "heatmap", z, x: sectors, y: classes.map((c: string) => c.slice(0, 16)),
                colorscale: [[0, C.panel], [0.5, C.blue], [1, C.cyan]],
                text: z.map((row: number[]) => row.map(String)), texttemplate: "%{text}",
              }]}
              layout={{
                ...DARK_LAYOUT, height: Math.max(320, classes.length * 28),
                margin: { t: 10, b: 60, l: 150, r: 20 },
                xaxis: { tickangle: -40, color: C.gray },
              }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
            <p className="text-[11px] text-gray-500 mt-3 leading-relaxed px-2">
              분류(행) × 섹터(열) 교차표. 셀의 숫자는 해당 분류-섹터에 속한 종목 수.
              특정 섹터에 DOWNTREND가 집중되면 해당 섹터 전반이 약세. CONTINUATION이 집중되면 강세 섹터.
            </p>
          </div>
        );
      })()}
    </div>
  );
}
