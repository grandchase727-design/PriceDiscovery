import { useEffect, useState, useMemo, useRef, useCallback } from "react";
import { fetchPortfolio } from "../../api/client";
import { C } from "../../styles/theme";

// ---------------------------------------------------------------------------
// PortfolioPanel — 매수 Final List 기반 개별종목/ETF 포트폴리오 2개
//   • Portfolio Agent가 시스템 분석 결과(Composite·QVR·DebateTier·Risk·Class)를
//     종합한 risk-adjusted conviction 가중으로 종목별 비중 산출
//   • 누적성과 = POINT-IN-TIME 주간 리밸런스 백테스트 (실제 캡처된 매수 Final List,
//     look-ahead·프록시 없음; 2026-07-03~ 성장) — 주식/ETF sleeve vs SPY
//   • dual-handle 기간 슬라이더로 시작일/종료일 조정 (re-base)
// ---------------------------------------------------------------------------

interface Holding {
  ticker: string; name: string; sector: string; weight: number;
  composite: number; qvr_score: number | null; tier: string;
  risk_score: number; classification: string; rationale: string;
}
interface PortfolioData {
  stocks: Holding[]; etfs: Holding[]; methodology: string;
  performance: {
    dates: string[]; stock_index: (number | null)[]; etf_index: (number | null)[];
    benchmark_index?: (number | null)[]; rebalance_dates?: string[];
    status?: string; real_since?: string; n_rebalances?: number;
    note?: string; methodology?: string; ytd_start?: string;
  };
  n_stocks: number; n_etfs: number; as_of: string; error?: string;
}

const BENCH_COLOR = C.gray;

const STOCK_COLOR = C.purple;
const ETF_COLOR = C.cyan;

function fpct(v: number) { return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`; }

// ── Weight table ──
function WeightTable({ title, color, holdings }: { title: string; color: string; holdings: Holding[] }) {
  const total = holdings.reduce((s, h) => s + h.weight, 0);
  return (
    <div className="rounded border" style={{ borderColor: C.border }}>
      <div className="px-3 py-2 text-[13px] font-bold flex items-center justify-between"
           style={{ color, borderBottom: `1px solid ${C.border}`, background: color + "10" }}>
        <span>{title} ({holdings.length}종목)</span>
        <span className="text-[11px]" style={{ color: C.gray }}>합계 {total.toFixed(1)}%</span>
      </div>
      <table className="w-full text-[12px]">
        <thead>
          <tr style={{ color: C.gray, borderBottom: `1px solid ${C.border}` }}>
            <th className="text-left px-2 py-1">Ticker</th>
            <th className="text-right px-2 py-1" style={{ minWidth: 90 }}>비중</th>
            <th className="text-center px-2 py-1">Comp</th>
            <th className="text-center px-2 py-1">QVR</th>
            <th className="text-center px-2 py-1">Tier</th>
            <th className="text-left px-2 py-1">근거</th>
          </tr>
        </thead>
        <tbody>
          {holdings.map((h) => (
            <tr key={h.ticker} style={{ borderBottom: `1px solid ${C.border}40` }}>
              <td className="px-2 py-1">
                <span className="font-mono font-bold" style={{ color: C.text }}>{h.ticker}</span>
                <span className="text-[10px] ml-1" style={{ color: C.gray }}>{h.sector}</span>
              </td>
              <td className="px-2 py-1">
                <div className="flex items-center gap-1.5 justify-end">
                  <div className="h-2 rounded" style={{ width: `${Math.min(100, h.weight * 5)}px`, background: color, minWidth: 4 }} />
                  <span className="font-mono font-bold" style={{ color: C.text, minWidth: 38, textAlign: "right" }}>{h.weight.toFixed(1)}%</span>
                </div>
              </td>
              <td className="text-center px-2 py-1 font-mono" style={{ color: C.text }}>{h.composite?.toFixed(0)}</td>
              <td className="text-center px-2 py-1 font-mono" style={{ color: C.gray }}>{h.qvr_score != null ? h.qvr_score.toFixed(0) : "—"}</td>
              <td className="text-center px-2 py-1 text-[10px]" style={{ color: C.gray }}>{(h.tier || "").slice(0, 10)}</td>
              <td className="px-2 py-1 text-[10px]" style={{ color: C.gray }}>{h.rationale}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function PortfolioPanel({ dataVersion = 0, scanning = false }:
  { dataVersion?: number; scanning?: boolean }) {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState<[number, number] | null>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const dragging = useRef<"start" | "end" | null>(null);

  useEffect(() => {
    setLoading(true);
    fetchPortfolio()
      .then((d) => {
        setData(d);
        const n = d?.performance?.dates?.length || 0;
        if (n > 1) setRange([0, n - 1]);
      })
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [dataVersion]);

  // 폴링: scan 중이면 30s마다 갱신
  useEffect(() => {
    if (!scanning) return;
    const id = setInterval(() => {
      fetchPortfolio().then((d) => { if (d?.stocks) setData(d); }).catch(() => {});
    }, 30000);
    return () => clearInterval(id);
  }, [scanning]);

  const perf = data?.performance;
  const N = perf?.dates?.length || 0;

  // ── Drag handlers for dual-handle slider ──
  const handleMove = useCallback((clientX: number) => {
    if (!dragging.current || !trackRef.current || N < 2 || !range) return;
    const rect = trackRef.current.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const idx = Math.round(frac * (N - 1));
    setRange((prev) => {
      if (!prev) return prev;
      const [s, e] = prev;
      if (dragging.current === "start") return [Math.min(idx, e - 1), e];
      return [s, Math.max(idx, s + 1)];
    });
  }, [N, range]);

  useEffect(() => {
    const mm = (ev: MouseEvent) => handleMove(ev.clientX);
    const mu = () => { dragging.current = null; };
    window.addEventListener("mousemove", mm);
    window.addEventListener("mouseup", mu);
    return () => { window.removeEventListener("mousemove", mm); window.removeEventListener("mouseup", mu); };
  }, [handleMove]);

  // ── Re-based chart series for [start, end] ──
  const chart = useMemo(() => {
    if (!perf || !range || N < 2) return null;
    const [s, e] = range;
    const dates = perf.dates.slice(s, e + 1);
    const reBase = (idx?: (number | null)[]) => {
      if (!idx || idx.length === 0) return [];
      const b = Number(idx[s]) || 1;
      return idx.slice(s, e + 1).map((v) => (v == null ? NaN : (Number(v) / b - 1) * 100));
    };
    const stock = reBase(perf.stock_index);
    const etf = reBase(perf.etf_index);
    const bench = reBase(perf.benchmark_index);
    const all = [...stock, ...etf, ...bench].filter((v) => Number.isFinite(v));
    const ymin = Math.min(0, ...all), ymax = Math.max(0, ...all);
    return { dates, stock, etf, bench, ymin, ymax };
  }, [perf, range, N]);

  if (loading && !data) return null;
  if (!data || data.error) {
    return (
      <div className="mt-4 rounded border p-3 text-[12px]" style={{ borderColor: C.border, color: C.gray }}>
        🧺 포트폴리오 — {data?.error ? `오류: ${data.error}` : "데이터 없음 (Run Live Scan 후 생성)"}
      </div>
    );
  }
  // Point-in-time 주간 리밸 트랙은 실제 캡처 데이터가 쌓여야 그려짐 (2026-07-03~ 성장).
  // 곡선이 아직 없을 땐 fabricate 하지 않고 정직하게 "축적 중"을 표시 + 오늘 비중 테이블은 노출.
  if (!perf || N < 2 || !range || perf.status === "accumulating" || perf.status === "no_data") {
    return (
      <div className="mt-4 rounded-lg border" style={{ borderColor: C.border }}>
        <div className="px-3 py-2 flex items-center justify-between"
             style={{ borderBottom: `1px solid ${C.border}`, background: C.bgAlt }}>
          <div className="text-[14px] font-bold" style={{ color: C.text }}>
            🧺 포트폴리오 — 매수 Final List 기반 (개별종목 / ETF)
          </div>
          <div className="text-[11px]" style={{ color: C.gray }}>
            Portfolio Agent · risk-adjusted conviction 가중 · {data.as_of?.slice(0, 10)}
          </div>
        </div>
        <div className="p-3 space-y-3">
          <div className="rounded border p-3 text-[12px]"
               style={{ borderColor: C.cyan + "55", background: C.cyan + "10", color: C.text }}>
            <b>📈 시점정합 · 주간 리밸런스 누적성과 (vs SPY)</b> — 실제 캡처 데이터로 성장 중 (look-ahead·프록시 없음).<br />
            <span style={{ color: C.gray }}>
              {perf?.note || "실제 매수 Final List 트랙이 축적되면 곡선이 표시됩니다."}
            </span>
            {perf?.methodology && (
              <div className="mt-1 text-[11px]" style={{ color: C.gray, fontStyle: "italic" }}>{perf.methodology}</div>
            )}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <WeightTable title="📈 개별종목 포트폴리오" color={STOCK_COLOR} holdings={data.stocks} />
            <WeightTable title="📦 ETF 포트폴리오" color={ETF_COLOR} holdings={data.etfs} />
          </div>
        </div>
      </div>
    );
  }
  const P = perf;        // narrowed non-null
  const [rs, re] = range;

  // SVG geometry
  const W = 900, H = 280, padL = 48, padR = 60, padT = 16, padB = 28;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xOf = (i: number, n: number) => padL + (n <= 1 ? 0 : (i / (n - 1)) * plotW);
  const yOf = (v: number) => {
    if (!chart) return padT;
    const span = (chart.ymax - chart.ymin) || 1;
    return padT + plotH - ((v - chart.ymin) / span) * plotH;
  };
  const poly = (series: number[]) =>
    series.map((v, i) => (Number.isFinite(v) ? `${xOf(i, series.length).toFixed(1)},${yOf(v).toFixed(1)}` : ""))
          .filter(Boolean).join(" ");

  const lastFinite = (a: number[]) => { for (let i = a.length - 1; i >= 0; i--) if (Number.isFinite(a[i])) return a[i]; return 0; };
  const stockFinal = chart ? lastFinite(chart.stock) : 0;
  const etfFinal = chart ? lastFinite(chart.etf) : 0;
  const benchFinal = chart ? lastFinite(chart.bench) : 0;
  const startDate = P.dates[rs], endDate = P.dates[re];

  // Y gridlines
  const yTicks = chart ? (() => {
    const ticks: number[] = []; const { ymin, ymax } = chart;
    const step = Math.max(1, Math.ceil((ymax - ymin) / 5));
    for (let v = Math.ceil(ymin / step) * step; v <= ymax; v += step) ticks.push(v);
    return ticks;
  })() : [];

  return (
    <div className="mt-4 rounded-lg border" style={{ borderColor: C.border }}>
      {/* Header */}
      <div className="px-3 py-2 flex items-center justify-between"
           style={{ borderBottom: `1px solid ${C.border}`, background: C.bgAlt }}>
        <div className="text-[14px] font-bold" style={{ color: C.text }}>
          🧺 포트폴리오 — 매수 Final List 기반 (개별종목 / ETF)
        </div>
        <div className="text-[11px]" style={{ color: C.gray }}>
          Portfolio Agent · risk-adjusted conviction 가중 · {data.as_of?.slice(0, 10)}
        </div>
      </div>

      <div className="p-3 space-y-4">
        {/* Cumulative performance chart */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <div className="text-[13px] font-bold" style={{ color: C.text }}>
              📈 누적성과 (시점정합 · 주간 리밸 vs SPY)
            </div>
            <div className="flex items-center gap-3 text-[12px]">
              <span style={{ color: STOCK_COLOR }}>● 개별종목 <b>{fpct(stockFinal)}</b></span>
              <span style={{ color: ETF_COLOR }}>● ETF <b>{fpct(etfFinal)}</b></span>
              <span style={{ color: BENCH_COLOR }}>● SPY <b>{fpct(benchFinal)}</b></span>
            </div>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 300 }}>
            {/* Y gridlines + labels */}
            {yTicks.map((v) => (
              <g key={v}>
                <line x1={padL} y1={yOf(v)} x2={W - padR} y2={yOf(v)}
                      stroke={C.border} strokeWidth={0.5} strokeDasharray={v === 0 ? "" : "3 3"} opacity={v === 0 ? 0.8 : 0.4} />
                <text x={padL - 6} y={yOf(v) + 3} textAnchor="end" fontSize={10} fill={C.gray}>{fpct(v)}</text>
              </g>
            ))}
            {/* X labels (start / mid / end) */}
            {chart && [0, Math.floor(chart.dates.length / 2), chart.dates.length - 1].map((i) => (
              <text key={i} x={xOf(i, chart.dates.length)} y={H - 8} textAnchor="middle" fontSize={10} fill={C.gray}>
                {chart.dates[i]?.slice(5)}
              </text>
            ))}
            {/* Lines */}
            {chart && chart.bench.length > 0 &&
              <polyline points={poly(chart.bench)} fill="none" stroke={BENCH_COLOR} strokeWidth={1.4} strokeDasharray="4 3" opacity={0.85} />}
            {chart && chart.etf.length > 0 &&
              <polyline points={poly(chart.etf)} fill="none" stroke={ETF_COLOR} strokeWidth={1.8} />}
            {chart && chart.stock.length > 0 &&
              <polyline points={poly(chart.stock)} fill="none" stroke={STOCK_COLOR} strokeWidth={1.8} />}
            {/* Final value markers */}
            {chart && chart.stock.length > 0 &&
              <circle cx={xOf(chart.stock.length - 1, chart.stock.length)} cy={yOf(stockFinal)} r={3} fill={STOCK_COLOR} />}
            {chart && chart.etf.length > 0 &&
              <circle cx={xOf(chart.etf.length - 1, chart.etf.length)} cy={yOf(etfFinal)} r={3} fill={ETF_COLOR} />}
          </svg>

          {/* Dual-handle range slider */}
          <div className="mt-1 px-2">
            <div ref={trackRef} className="relative h-6 select-none" style={{ cursor: "pointer" }}>
              {/* track */}
              <div className="absolute top-1/2 left-0 right-0 h-1.5 rounded -translate-y-1/2" style={{ background: C.border }} />
              {/* selected range */}
              <div className="absolute top-1/2 h-1.5 rounded -translate-y-1/2"
                   style={{
                     background: C.purple,
                     left: `${(rs / (N - 1)) * 100}%`,
                     width: `${((re - rs) / (N - 1)) * 100}%`,
                   }} />
              {/* start handle */}
              <div onMouseDown={() => { dragging.current = "start"; }}
                   className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full border-2 shadow"
                   style={{ left: `${(rs / (N - 1)) * 100}%`, width: 16, height: 16,
                            background: "#fff", borderColor: STOCK_COLOR, cursor: "grab" }}
                   title={`시작 ${startDate}`} />
              {/* end handle */}
              <div onMouseDown={() => { dragging.current = "end"; }}
                   className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 rounded-full border-2 shadow"
                   style={{ left: `${(re / (N - 1)) * 100}%`, width: 16, height: 16,
                            background: "#fff", borderColor: ETF_COLOR, cursor: "grab" }}
                   title={`종료 ${endDate}`} />
            </div>
            <div className="flex items-center justify-between text-[11px] mt-0.5" style={{ color: C.gray }}>
              <span>시작: <b style={{ color: C.text }}>{startDate}</b></span>
              <span>{chart?.dates.length}거래일</span>
              <span>종료: <b style={{ color: C.text }}>{endDate}</b></span>
            </div>
          </div>
        </div>

        {/* Weight tables */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <WeightTable title="📈 개별종목 포트폴리오" color={STOCK_COLOR} holdings={data.stocks} />
          <WeightTable title="📦 ETF 포트폴리오" color={ETF_COLOR} holdings={data.etfs} />
        </div>

        {/* Methodology */}
        <div className="text-[11px] rounded border p-2 leading-relaxed" style={{ borderColor: C.border, color: C.gray }}>
          <b style={{ color: C.text }}>비중 산출 (Portfolio Agent):</b> {data.methodology}
        </div>
      </div>
    </div>
  );
}
