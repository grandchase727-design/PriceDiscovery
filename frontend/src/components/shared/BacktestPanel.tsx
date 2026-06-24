/**
 * BacktestPanel — Individual ticker selection skill evaluation.
 *
 * Renders 3-layer evaluation framework:
 *   Layer 1: Forward return distribution (direction)
 *   Layer 2: Sector-neutral alpha + EV (edge)
 *   Layer 3: Rank quintile + IC + sector breakdown (attribution)
 */
import { useEffect, useMemo, useState } from "react";
import {
  fetchBacktestResults,
  startBacktestRun,
  fetchBacktestStatus,
  fetchBacktestRankings,
  fetchTickerDrilldown,
  fetchFinalList,
  fetchValidatedExtraTimeline,
  type BacktestResult,
  type BacktestBucketMetric,
  type BacktestRankings,
  type BucketRanking,
  type TickerDetail,
  type TradingMetricsBucket,
  type TradingLifecyclesCompact,
  type LifecycleRecord,
} from "../../api/client";

// FT (Financial Times) palette — light pink paper, near-black text
const C = {
  bg: "#FFF1E5", bgAlt: "#FFFFFF", border: "#E6D9CE",
  text: "#33302E", gray: "#66605C",
  cyan: "#0D7680", purple: "#7D5BA6", green: "#0A7D3F",
  red: "#CC0000", amber: "#B85C00",
};

const BUCKETS = ["long_stocks", "long_etfs", "short_stocks", "short_etfs"] as const;
const BUCKET_META: Record<string, { label: string; emoji: string; color: string; isShort: boolean }> = {
  long_stocks:  { label: "LONG Stocks",  emoji: "📈", color: C.green, isShort: false },
  long_etfs:    { label: "LONG ETFs",    emoji: "📈", color: C.green, isShort: false },
  short_stocks: { label: "SHORT Stocks", emoji: "📉", color: C.red,   isShort: true  },
  short_etfs:   { label: "SHORT ETFs",   emoji: "📉", color: C.red,   isShort: true  },
};

function statColor(value: number | undefined, threshold: number, isHigherBetter = true): string {
  if (value == null) return C.gray;
  const isGood = isHigherBetter ? value >= threshold : value <= threshold;
  return isGood ? C.green : value === 0 ? C.gray : C.red;
}

function StatCard({ label, value, format, color, hint, ciLo, ciHi }:
  { label: string; value: number | undefined; format: string;
    color?: string; hint?: string;
    ciLo?: number | null; ciHi?: number | null }) {
  const v = value == null ? "—" :
    format === "pct" ? `${value >= 0 ? "+" : ""}${value.toFixed(1)}%` :
    format === "ratio" ? value.toFixed(2) :
    format === "t" ? `${value >= 0 ? "+" : ""}${value.toFixed(2)}` :
    format === "ic" ? `${value >= 0 ? "+" : ""}${value.toFixed(3)}` :
    value.toString();
  const ciStr = (ciLo != null && ciHi != null) ?
    (format === "pct"
      ? `${ciLo >= 0 ? "+" : ""}${ciLo.toFixed(1)}% ~ ${ciHi >= 0 ? "+" : ""}${ciHi.toFixed(1)}%`
      : `${ciLo.toFixed(2)} ~ ${ciHi.toFixed(2)}`)
    : null;
  return (
    <div className="rounded px-2.5 py-1.5" style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}>
      <div className="text-[11px] uppercase font-bold" style={{ color: C.gray }}>{label}</div>
      <div className="text-[16px] font-bold font-mono" style={{ color: color || C.text }}>{v}</div>
      {ciStr && <div className="text-[10px] mt-0.5 font-mono" style={{ color: C.cyan + "cc" }}>CI 95%: {ciStr}</div>}
      {hint && !ciStr && <div className="text-[10px] mt-0.5" style={{ color: C.gray }}>{hint}</div>}
    </div>
  );
}

function TopWorstTable({ title, color, rows, isShort, onClick }:
  { title: string; color: string; rows: BucketRanking[]; isShort: boolean; onClick: (t: string) => void }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div>
      <div className="text-[12px] uppercase font-bold mb-1" style={{ color }}>{title}</div>
      <table className="w-full text-[12px] border-collapse">
        <thead>
          <tr style={{ borderBottom: `1px solid ${C.border}` }}>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>#</th>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Ticker</th>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Name</th>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Sector</th>
            <th className="text-right py-1 px-1.5" style={{ color: C.gray }}>n</th>
            <th className="text-right py-1 px-1.5" style={{ color: C.gray }}>Avg Rank</th>
            <th className="text-right py-1 px-1.5" style={{ color: C.gray }}>Win %</th>
            <th className="text-right py-1 px-1.5" style={{ color: C.gray }}>Alpha 21d</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 10).map((r, i) => {
            const aPct = (r.mean_alpha_21d || 0) * 100;
            const aColor = aPct > 0 ? C.green : C.red;
            return (
              <tr key={r.ticker} onClick={() => onClick(r.ticker)}
                  className="cursor-pointer hover:opacity-80"
                  style={{ borderBottom: `1px solid ${C.border}40` }}>
                <td className="py-1 px-1.5" style={{ color: C.gray }}>{i + 1}</td>
                <td className="py-1 px-1.5 font-mono font-bold" style={{ color: C.cyan, textDecoration: "underline" }}>
                  {r.ticker}
                </td>
                <td className="py-1 px-1.5" style={{ color: C.text }}>{(r.name || "").slice(0, 22)}</td>
                <td className="py-1 px-1.5 whitespace-nowrap" style={{ color: C.cyan }}>{r.sector}</td>
                <td className="py-1 px-1.5 text-right font-mono">{r.n}</td>
                <td className="py-1 px-1.5 text-right font-mono">{(r.avg_rank || 0).toFixed(1)}</td>
                <td className="py-1 px-1.5 text-right font-mono"
                    style={{ color: statColor(r.win_rate_21d, isShort ? 50 : 55) }}>
                  {(r.win_rate_21d || 0).toFixed(0)}%
                </td>
                <td className="py-1 px-1.5 text-right font-mono" style={{ color: aColor }}>
                  {aPct >= 0 ? "+" : ""}{aPct.toFixed(2)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function BucketReport({ name, m, horizon, rankings, onTickerClick }:
  { name: string; m: BacktestBucketMetric; horizon: number;
    rankings?: { top: BucketRanking[]; worst: BucketRanking[] };
    onTickerClick: (t: string) => void }) {
  const meta = BUCKET_META[name];
  const fwd = m.forward_return_dist;
  const a = m.alpha_stats;
  const rq = m.rank_quintile;

  const sectorEntries = Object.entries(m.by_sector || {})
    .filter(([_, v]) => (v.n || 0) >= 5)
    .sort((a, b) => (b[1].mean_alpha || 0) - (a[1].mean_alpha || 0));

  return (
    <div className="rounded p-3 mb-3" style={{ backgroundColor: C.bg, border: `1px solid ${meta.color}40` }}>
      <div className="flex items-center justify-between mb-2 pb-1.5 border-b" style={{ borderColor: C.border }}>
        <div className="text-[14px] font-bold flex items-center gap-2" style={{ color: meta.color }}>
          {meta.emoji} {meta.label}
        </div>
        <div className="text-[12px]" style={{ color: C.gray }}>
          n={m.n_total} picks · horizon {horizon}d
        </div>
      </div>

      {/* Layer 1 — Direction */}
      <div className="mb-2">
        <div className="text-[11px] uppercase font-bold mb-1" style={{ color: C.cyan }}>
          Layer 1 — Direction
        </div>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-1.5">
          <StatCard label="Hit Rate" value={fwd.hit_rate} format="pct"
                    color={statColor(fwd.hit_rate, 55)}
                    ciLo={(fwd as any).hit_ci_lo} ciHi={(fwd as any).hit_ci_hi}
                    hint=">55% = edge" />
          <StatCard label="Mean Ret" value={fwd.mean} format="pct"
                    color={(fwd.mean || 0) > 0 ? C.green : C.red}
                    ciLo={(fwd as any).mean_ci_lo} ciHi={(fwd as any).mean_ci_hi} />
          <StatCard label="Median" value={fwd.median} format="pct" />
          <StatCard label="P25" value={fwd.p25} format="pct" />
          <StatCard label="P75" value={fwd.p75} format="pct" />
          <StatCard label="Std Dev" value={fwd.std} format="pct" />
        </div>
      </div>

      {/* Layer 2 — Edge */}
      <div className="mb-2">
        <div className="text-[11px] uppercase font-bold mb-1" style={{ color: C.purple }}>
          Layer 2 — Edge (Alpha vs Sector ETF, net of 10bp txn cost)
        </div>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-1.5">
          <StatCard label="Mean Alpha" value={a.mean_alpha} format="pct"
                    color={(a.mean_alpha || 0) > 0 ? C.green : C.red}
                    ciLo={(a as any).mean_alpha_ci_lo} ciHi={(a as any).mean_alpha_ci_hi}
                    hint=">0 = beats sector ETF" />
          <StatCard label="Win Rate" value={a.win_rate} format="pct"
                    color={statColor(a.win_rate, 55)} />
          <StatCard label="Avg Win" value={a.avg_win} format="pct" color={C.green} />
          <StatCard label="Avg Loss" value={a.avg_loss} format="pct" color={C.red} />
          <StatCard label="W/L Ratio" value={a.win_loss_ratio} format="ratio"
                    color={statColor(a.win_loss_ratio, 1.5)} hint=">1.5 = good asymm" />
          <StatCard label="t-stat" value={a.t_stat} format="t"
                    color={statColor(Math.abs(a.t_stat || 0), 2)} hint="|t|>2 = significant" />
        </div>
      </div>

      {/* Layer 3 — Rank Quintile */}
      <div className="mb-2">
        <div className="flex items-baseline justify-between mb-1">
          <div className="text-[11px] uppercase font-bold" style={{ color: C.amber }}>
            Layer 3 — Conviction Rank Quintile
          </div>
          <div className="text-[11px]" style={{ color: C.gray }}>
            IC={(rq._ic || 0).toFixed(3)} · monotonicity={Math.round((rq._monotonicity || 0) * 100)}%
          </div>
        </div>
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
              <th className="text-left py-1 px-2" style={{ color: C.gray }}>Quintile</th>
              <th className="text-left py-1 px-2" style={{ color: C.gray }}>Rank</th>
              <th className="text-right py-1 px-2" style={{ color: C.gray }}>n</th>
              <th className="text-right py-1 px-2" style={{ color: C.gray }}>Hit %</th>
              <th className="text-right py-1 px-2" style={{ color: C.gray }}>Mean Ret</th>
              <th className="text-right py-1 px-2" style={{ color: C.gray }}>Alpha</th>
              <th className="text-left py-1 px-2" style={{ color: C.gray }}>Bar</th>
            </tr>
          </thead>
          <tbody>
            {["Q1", "Q2", "Q3", "Q4", "Q5"].map((q) => {
              const qd = (rq as any)[q];
              if (!qd || qd.n === 0) return null;
              const aPct = qd.mean_alpha || 0;
              const bar = Math.min(100, Math.abs(aPct) * 5);
              const barColor = aPct > 0 ? C.green : C.red;
              return (
                <tr key={q} style={{ borderBottom: `1px solid ${C.border}40` }}>
                  <td className="py-1 px-2 font-bold" style={{ color: C.text }}>{q}</td>
                  <td className="py-1 px-2" style={{ color: C.gray }}>{qd.rank_range}</td>
                  <td className="py-1 px-2 text-right font-mono">{qd.n}</td>
                  <td className="py-1 px-2 text-right font-mono"
                      style={{ color: statColor(qd.hit_rate, 55) }}>
                    {qd.hit_rate.toFixed(1)}%
                  </td>
                  <td className="py-1 px-2 text-right font-mono"
                      style={{ color: (qd.mean_ret || 0) > 0 ? C.green : C.red }}>
                    {(qd.mean_ret || 0) >= 0 ? "+" : ""}{(qd.mean_ret || 0).toFixed(2)}%
                  </td>
                  <td className="py-1 px-2 text-right font-mono"
                      style={{ color: aPct > 0 ? C.green : C.red }}>
                    {aPct >= 0 ? "+" : ""}{aPct.toFixed(2)}%
                  </td>
                  <td className="py-1 px-2">
                    <div className="rounded" style={{ width: `${bar}%`, height: 8, backgroundColor: barColor + "60" }} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Layer 3 — Top/Worst tickers (click to drill down) */}
      {rankings && (rankings.top.length > 0 || rankings.worst.length > 0) && (
        <div className="mb-2">
          <div className="text-[11px] uppercase font-bold mb-1" style={{ color: C.amber }}>
            Layer 3 — Top/Worst Tickers (click ticker for drilldown)
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            <TopWorstTable title="🌟 TOP — Highest 21d Alpha"  color={C.green}
                           rows={rankings.top}   isShort={meta.isShort} onClick={onTickerClick} />
            <TopWorstTable title="⚠ WORST — Lowest 21d Alpha"  color={C.red}
                           rows={rankings.worst} isShort={meta.isShort} onClick={onTickerClick} />
          </div>
        </div>
      )}

      {/* Layer 3 — Sector breakdown (top 6) */}
      {sectorEntries.length > 0 && (
        <div>
          <div className="text-[11px] uppercase font-bold mb-1" style={{ color: C.amber }}>
            Layer 3 — Top Sectors by Alpha
          </div>
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                <th className="text-left py-1 px-2" style={{ color: C.gray }}>Sector</th>
                <th className="text-right py-1 px-2" style={{ color: C.gray }}>n</th>
                <th className="text-right py-1 px-2" style={{ color: C.gray }}>Win %</th>
                <th className="text-right py-1 px-2" style={{ color: C.gray }}>Alpha</th>
                <th className="text-right py-1 px-2" style={{ color: C.gray }}>t-stat</th>
              </tr>
            </thead>
            <tbody>
              {sectorEntries.slice(0, 8).map(([sec, v]) => {
                const aPct = v.mean_alpha || 0;
                return (
                  <tr key={sec} style={{ borderBottom: `1px solid ${C.border}40` }}>
                    <td className="py-1 px-2" style={{ color: C.text }}>{sec}</td>
                    <td className="py-1 px-2 text-right font-mono">{v.n}</td>
                    <td className="py-1 px-2 text-right font-mono">{(v.win_rate || 0).toFixed(1)}%</td>
                    <td className="py-1 px-2 text-right font-mono"
                        style={{ color: aPct > 0 ? C.green : C.red }}>
                      {aPct >= 0 ? "+" : ""}{aPct.toFixed(2)}%
                    </td>
                    <td className="py-1 px-2 text-right font-mono"
                        style={{ color: statColor(Math.abs(v.t_stat || 0), 2) }}>
                      {(v.t_stat || 0) >= 0 ? "+" : ""}{(v.t_stat || 0).toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function BacktestPanel() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [rankings, setRankings] = useState<BacktestRankings | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [horizon, setHorizon] = useState<5 | 21 | 63>(21);
  const [drilldownTicker, setDrilldownTicker] = useState<string | null>(null);
  const [drilldownData, setDrilldownData] = useState<TickerDetail | null>(null);
  const [drilldownLoading, setDrilldownLoading] = useState(false);

  useEffect(() => {
    Promise.all([
      fetchBacktestResults(),
      fetchBacktestRankings(),
    ]).then(([res, rks]) => {
      if (res.available && res.result) setResult(res.result);
      if (rks.available && rks.rankings) setRankings(rks.rankings);
    }).finally(() => setLoading(false));
  }, []);

  const handleTickerClick = (ticker: string) => {
    setDrilldownTicker(ticker);
    setDrilldownData(null);
    setDrilldownLoading(true);
    fetchTickerDrilldown(ticker).then((r) => {
      if (r.available && r.data) setDrilldownData(r.data);
    }).finally(() => setDrilldownLoading(false));
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      fetchBacktestStatus().then((s) => {
        if (!s.running) {
          setRunning(false);
          fetchBacktestResults().then((r) => {
            if (r.available && r.result) setResult(r.result);
          });
        }
      });
    }, 3000);
    return () => clearInterval(id);
  }, [running]);

  if (loading) return <div className="text-[14px]" style={{ color: C.gray }}>Loading backtest…</div>;

  if (!result) {
    return (
      <div className="border-l-4 rounded p-3"
           style={{ borderLeftColor: C.amber, border: `1px solid ${C.border}`, borderLeftWidth: 4 }}>
        <div className="flex items-center justify-between">
          <div>
            <div className="font-bold text-[16px]" style={{ color: C.amber }}>
              📊 PM Agent Backtest — Stock-Picker Skill Evaluation
            </div>
            <div className="text-[14px] mt-0.5" style={{ color: C.gray }}>
              Individual ticker selection evaluation (deterministic proxy, YTD 2026). Not yet run.
            </div>
          </div>
          <button onClick={() => { setRunning(true); startBacktestRun(); }}
                  className="px-3 py-1.5 rounded text-[14px] font-bold"
                  style={{ backgroundColor: C.amber + "30", color: C.amber, border: `1px solid ${C.amber}80` }}>
            {running ? "Running…" : "Run Backtest"}
          </button>
        </div>
      </div>
    );
  }

  const metrics = (result.horizon_metrics as any)[`h${horizon}d`];

  return (
    <div className="border-l-4 rounded p-3"
         style={{ borderLeftColor: C.amber, border: `1px solid ${C.border}`, borderLeftWidth: 4 }}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="font-bold text-[16px] flex items-center gap-2" style={{ color: C.amber }}>
            📊 PM Agent Backtest — Stock-Picker Skill Evaluation
            <span className="text-[11px] px-1.5 py-0.5 rounded font-bold"
                  style={{ backgroundColor: C.green + "25", color: C.green, border: `1px solid ${C.green}80` }}>
              ✓ BIAS-CORRECTED
            </span>
          </div>
          <div className="text-[12px] mt-0.5" style={{ color: C.gray }}>
            Deterministic proxy · {result.n_picks} picks across {result.weekly_summary.length} weeks
            · Jan 1 – {result.end_date} · last run {result.as_of_run}
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: C.gray }}>
            Fixes: <span style={{ color: C.cyan }}>Fix A</span> 120d history filter ·
            <span style={{ color: C.cyan }}> Fix C</span> sector-ETF benchmark (XLF/XLK/XLV/etc.) ·
            <span style={{ color: C.cyan }}> Fix D</span> 10bp txn cost ·
            <span style={{ color: C.cyan }}> Fix F</span> 95% bootstrap CI
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[12px]" style={{ color: C.gray }}>Horizon:</span>
          {([5, 21, 63] as const).map((h) => (
            <button key={h} onClick={() => setHorizon(h)}
                    className="px-2 py-1 rounded text-[12px] transition-colors"
                    style={{
                      backgroundColor: horizon === h ? C.amber + "30" : "transparent",
                      color: horizon === h ? C.amber : C.gray,
                      border: `1px solid ${horizon === h ? C.amber + "80" : C.border}`,
                      fontWeight: horizon === h ? "bold" : "normal",
                    }}>
              {h}d
            </button>
          ))}
          <button onClick={() => { setRunning(true); startBacktestRun(); }}
                  disabled={running}
                  className="ml-2 px-2 py-1 rounded text-[12px]"
                  style={{ backgroundColor: C.amber + "20", color: C.amber, border: `1px solid ${C.amber}50` }}>
            {running ? "Running…" : "Re-run"}
          </button>
        </div>
      </div>

      {BUCKETS.map((b) => metrics[b] && (
        <BucketReport key={b} name={b} m={metrics[b]} horizon={horizon}
                      rankings={rankings ? (rankings as any)[b] : undefined}
                      onTickerClick={handleTickerClick} />
      ))}

      {/* Trading Layer Analysis (Architecture C — Phase 1) */}
      {result.trading_metrics && (
        <TradingLayerReport
          tm={result.trading_metrics}
          lifecycles={result.trading_lifecycles_compact}
          endDate={result.end_date}
        />
      )}

      {drilldownTicker && (
        <TickerDrilldown ticker={drilldownTicker} data={drilldownData}
                         loading={drilldownLoading}
                         onClose={() => { setDrilldownTicker(null); setDrilldownData(null); }} />
      )}
    </div>
  );
}

function TickerDrilldown({ ticker, data, loading, onClose }:
  { ticker: string; data: TickerDetail | null; loading: boolean; onClose: () => void }) {
  const apps = (data?.appearances || []).slice().sort((a, b) => a.entry_date.localeCompare(b.entry_date));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
         style={{ backgroundColor: "rgba(0,0,0,0.75)" }} onClick={onClose}>
      <div className="rounded-lg p-4 max-h-[90vh] overflow-auto w-full max-w-4xl"
           style={{ backgroundColor: C.bg, border: `2px solid ${C.amber}80` }}
           onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3 pb-2 border-b" style={{ borderColor: C.border }}>
          <div>
            <div className="text-[18px] font-bold flex items-center gap-2" style={{ color: C.amber }}>
              📊 {ticker}
              <span className="text-[12px] font-normal" style={{ color: C.gray }}>
                {data?.name} · {data?.sector} · {data?.asset_type}
              </span>
            </div>
            <div className="text-[12px] mt-1" style={{ color: C.gray }}>
              {data ? `${data.n_appearances} cohort appearances · avg rank ${data.avg_rank?.toFixed(1)} · buckets: ${data.buckets?.join(", ")}` : ""}
            </div>
          </div>
          <button onClick={onClose} className="text-[14px] px-2 py-1 rounded"
                  style={{ backgroundColor: C.border, color: C.text }}>✕</button>
        </div>

        {loading && <div className="text-[14px]" style={{ color: C.gray }}>Loading drilldown…</div>}

        {data && (
          <>
            {/* Aggregate stats */}
            <div className="grid grid-cols-3 gap-2 mb-3">
              {[5, 21, 63].map((h) => {
                const a = (data as any)[`mean_alpha_${h}d`];
                const w = (data as any)[`win_rate_${h}d`];
                const r = (data as any)[`mean_ret_${h}d`];
                return (
                  <div key={h} className="rounded p-2"
                       style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}>
                    <div className="text-[11px] uppercase font-bold mb-1" style={{ color: C.cyan }}>
                      {h}d horizon
                    </div>
                    <div className="text-[12px]" style={{ color: C.gray }}>
                      Mean Ret: <span style={{ color: (r || 0) > 0 ? C.green : C.red }}>
                        {(r || 0) >= 0 ? "+" : ""}{((r || 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="text-[12px]" style={{ color: C.gray }}>
                      Mean Alpha: <span style={{ color: (a || 0) > 0 ? C.green : C.red }}>
                        {(a || 0) >= 0 ? "+" : ""}{((a || 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="text-[12px]" style={{ color: C.gray }}>
                      Win Rate: <span style={{ color: C.text }}>{(w || 0).toFixed(0)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Per-cohort appearances table */}
            <div className="text-[12px] uppercase font-bold mb-1" style={{ color: C.amber }}>
              All Cohort Appearances ({apps.length})
            </div>
            <div className="overflow-auto" style={{ maxHeight: 400 }}>
              <table className="w-full text-[12px] border-collapse">
                <thead className="sticky top-0" style={{ backgroundColor: "#FFF1E5", zIndex: 1 }}>
                  <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                    <th className="text-left py-1 px-2" style={{ color: C.gray }}>Entry Date</th>
                    <th className="text-left py-1 px-2" style={{ color: C.gray }}>Bucket</th>
                    <th className="text-right py-1 px-2" style={{ color: C.gray }}>Rank</th>
                    <th className="text-right py-1 px-2" style={{ color: C.gray }}>Score</th>
                    <th className="text-right py-1 px-2" style={{ color: C.gray }}>5d Ret</th>
                    <th className="text-right py-1 px-2" style={{ color: C.gray }}>21d Ret</th>
                    <th className="text-right py-1 px-2" style={{ color: C.gray }}>63d Ret</th>
                    <th className="text-right py-1 px-2" style={{ color: C.gray }}>5d α</th>
                    <th className="text-right py-1 px-2" style={{ color: C.gray }}>21d α</th>
                    <th className="text-right py-1 px-2" style={{ color: C.gray }}>63d α</th>
                  </tr>
                </thead>
                <tbody>
                  {apps.map((a, i) => {
                    const fmt = (v: number | null) => v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
                    const col = (v: number | null) => v == null ? C.gray : v > 0 ? C.green : C.red;
                    return (
                      <tr key={i} style={{ borderBottom: `1px solid ${C.border}40` }}>
                        <td className="py-1 px-2 font-mono">{a.entry_date}</td>
                        <td className="py-1 px-2" style={{ color: a.side === "long" ? C.green : C.red }}>{a.bucket}</td>
                        <td className="py-1 px-2 text-right font-mono">#{a.rank}</td>
                        <td className="py-1 px-2 text-right font-mono" style={{ color: C.gray }}>
                          {a.proxy_score?.toFixed(1) ?? "—"}
                        </td>
                        <td className="py-1 px-2 text-right font-mono" style={{ color: col(a.ret_5d) }}>{fmt(a.ret_5d)}</td>
                        <td className="py-1 px-2 text-right font-mono" style={{ color: col(a.ret_21d) }}>{fmt(a.ret_21d)}</td>
                        <td className="py-1 px-2 text-right font-mono" style={{ color: col(a.ret_63d) }}>{fmt(a.ret_63d)}</td>
                        <td className="py-1 px-2 text-right font-mono" style={{ color: col(a.alpha_5d) }}>{fmt(a.alpha_5d)}</td>
                        <td className="py-1 px-2 text-right font-mono" style={{ color: col(a.alpha_21d) }}>{fmt(a.alpha_21d)}</td>
                        <td className="py-1 px-2 text-right font-mono" style={{ color: col(a.alpha_63d) }}>{fmt(a.alpha_63d)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Trading Layer Analysis (Architecture C — 4 new metric families)
// ─────────────────────────────────────────────────────────────────────

type TradingHorizon = "tactical" | "core" | "strategic";
const TH_META: Record<TradingHorizon, { label: string; emoji: string; days: string; color: string }> = {
  tactical:  { label: "Tactical",  emoji: "🚀", days: "5d",  color: C.amber },
  core:      { label: "Core",      emoji: "⚓", days: "21d", color: C.purple },
  strategic: { label: "Strategic", emoji: "🌐", days: "63d", color: C.cyan },
};

function _pct(v: number | undefined | null, digits = 2): string {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

function _pctColor(v: number | undefined | null, positiveColor = C.green, negativeColor = C.red): string {
  if (v == null) return C.gray;
  return v > 0 ? positiveColor : v < 0 ? negativeColor : C.gray;
}

function TradingLayerReport({ tm, lifecycles, endDate }: {
  tm: { tactical: TradingMetricsBucket; core: TradingMetricsBucket; strategic: TradingMetricsBucket };
  lifecycles?: TradingLifecyclesCompact;
  endDate?: string;
}) {
  const [horizon, setHorizon] = useState<TradingHorizon>("core");
  const b = tm[horizon];
  const meta = TH_META[horizon];
  const lc = b.trade_lifecycle;
  const isNetPositive = (lc?.delta_alpha ?? -1) > 0;

  // Fetch validated buy-list tickers (★★ / ★★★ tier) and SPLIT into:
  //   • backtest-validated (in_proxy_recent OR in_top_alpha): timeline displayable
  //   • current-state-validated (state/composite only): NO backtest history → info list
  const [validated, setValidated] = useState<{
    stocks_with_bt: Set<string>;
    etfs_with_bt:   Set<string>;
    stocks_no_bt:   any[];
    etfs_no_bt:     any[];
    total: number; loading: boolean;
  }>({
    stocks_with_bt: new Set(), etfs_with_bt: new Set(),
    stocks_no_bt: [], etfs_no_bt: [], total: 0, loading: true,
  });

  // Extra lifecycles for stocks not in backtest data (synthesized from price cache)
  const [extraLifecycles, setExtraLifecycles] = useState<TradingLifecyclesCompact | null>(null);

  useEffect(() => {
    fetchFinalList().then((d) => {
      const high = (d.buy_list || []).filter((r) => r.stars >= 2);

      // ALL ★★+ tickers go into the timeline display set
      // (extra lifecycles will be merged so all of them have data)
      const stocks = new Set<string>();
      const etfs   = new Set<string>();
      for (const r of high) {
        if (r.bucket?.includes("stocks")) stocks.add(r.ticker);
        else if (r.bucket?.includes("etfs")) etfs.add(r.ticker);
      }
      setValidated({
        stocks_with_bt: stocks, etfs_with_bt: etfs,
        stocks_no_bt: [], etfs_no_bt: [],   // no longer split; all show in timeline
        total: high.length, loading: false,
      });
    }).catch(() => setValidated((s) => ({ ...s, loading: false })));

    // Fetch extra timeline (synthesized for missing tickers)
    fetchValidatedExtraTimeline().then(setExtraLifecycles).catch(() => {});
  }, []);

  // Merge extra lifecycles with main lifecycles (used by validated section heatmap)
  const mergedLifecycles = useMemo<TradingLifecyclesCompact | undefined>(() => {
    if (!lifecycles) return undefined;
    if (!extraLifecycles) return lifecycles;
    const out: any = { tactical: {}, core: {}, strategic: {} };
    for (const h of ["tactical", "core", "strategic"] as const) {
      const base = (lifecycles as any)[h] || {};
      const extra = (extraLifecycles as any)[h] || {};
      out[h] = {
        long_stocks:  [...(base.long_stocks  || []), ...(extra.long_stocks  || [])],
        long_etfs:    [...(base.long_etfs    || []), ...(extra.long_etfs    || [])],
        short_stocks: [...(base.short_stocks || []), ...(extra.short_stocks || [])],
        short_etfs:   [...(base.short_etfs   || []), ...(extra.short_etfs   || [])],
      };
    }
    return out as TradingLifecyclesCompact;
  }, [lifecycles, extraLifecycles]);

  return (
    <div className="rounded p-3 mt-4"
         style={{ backgroundColor: C.bg, border: `2px solid ${C.purple}80` }}>
      <div className="flex items-center justify-between mb-3 pb-2 border-b" style={{ borderColor: C.border }}>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-[15px] font-bold" style={{ color: C.purple }}>
            🎯 Trading Layer Analysis (Architecture C — Deterministic Proxy)
          </span>
          <span className="text-[11px] px-1.5 py-0.5 rounded font-bold"
                style={{ backgroundColor: isNetPositive ? C.green + "25" : C.red + "25",
                         color: isNetPositive ? C.green : C.red,
                         border: `1px solid ${isNetPositive ? C.green : C.red}80` }}>
            {isNetPositive ? "✓ NET POSITIVE" : "⚠ NET NEGATIVE"} (Δα {_pct(lc?.delta_alpha)})
          </span>
        </div>
        <div className="flex items-center gap-2">
          {(["tactical","core","strategic"] as TradingHorizon[]).map((k) => {
            const m = TH_META[k];
            const active = horizon === k;
            return (
              <button key={k} onClick={() => setHorizon(k)}
                      className="px-3 py-1 rounded text-[12px] transition-colors"
                      style={{ backgroundColor: active ? m.color + "30" : "transparent",
                               color: active ? m.color : C.gray,
                               border: `1px solid ${active ? m.color + "80" : C.border}`,
                               fontWeight: active ? "bold" : "normal" }}>
                {m.emoji} {m.label} ({m.days})
              </button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 mb-3">
        {/* Layer 4: Entry Signal Edge */}
        <div className="rounded p-2" style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}>
          <div className="text-[11px] uppercase font-bold mb-1.5" style={{ color: meta.color }}>
            Layer 4 — Entry Signal Edge (BUY_NOW vs WAIT vs SKIP)
          </div>
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                <th className="text-left py-0.5 px-1" style={{ color: C.gray }}>Signal</th>
                <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>n</th>
                <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Mean Ret</th>
                <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Days→Trig</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(b.entry_signal_edge).map(([sig, stats]) => {
                if ((stats as any).n === 0) return null;
                const s = stats as any;
                return (
                  <tr key={sig} style={{ borderBottom: `1px solid ${C.border}40` }}>
                    <td className="py-0.5 px-1 font-bold" style={{
                      color: sig.includes("BUY_NOW") ? C.green :
                             sig.includes("SKIP") ? C.red : C.amber }}>
                      {sig}
                    </td>
                    <td className="py-0.5 px-1 text-right font-mono">{s.n}</td>
                    <td className="py-0.5 px-1 text-right font-mono"
                        style={{ color: _pctColor(s.mean_return) }}>
                      {_pct(s.mean_return)}
                    </td>
                    <td className="py-0.5 px-1 text-right font-mono" style={{ color: C.gray }}>
                      {s.days_to_trigger_mean?.toFixed(1) ?? "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Layer 5: Exit Trigger Effectiveness */}
        <div className="rounded p-2" style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}>
          <div className="text-[11px] uppercase font-bold mb-1.5" style={{ color: meta.color }}>
            Layer 5 — Exit Trigger Effectiveness
          </div>
          <table className="w-full text-[12px] border-collapse">
            <thead>
              <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                <th className="text-left py-0.5 px-1" style={{ color: C.gray }}>Exit Type</th>
                <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Fire %</th>
                <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Ret</th>
                <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Win %</th>
                <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Δ vs Hold</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(b.exit_trigger_effectiveness).map(([et, stats]) => {
                const s = stats as any;
                return (
                  <tr key={et} style={{ borderBottom: `1px solid ${C.border}40` }}>
                    <td className="py-0.5 px-1 font-bold" style={{
                      color: et === "TAKE_PROFIT" ? C.green :
                             et === "STOP_LOSS" ? C.red :
                             et === "OVEREXT" ? C.amber : C.gray }}>
                      {et}
                    </td>
                    <td className="py-0.5 px-1 text-right font-mono">{s.fire_pct?.toFixed(1)}%</td>
                    <td className="py-0.5 px-1 text-right font-mono"
                        style={{ color: _pctColor(s.mean_return) }}>
                      {_pct(s.mean_return)}
                    </td>
                    <td className="py-0.5 px-1 text-right font-mono">{s.win_rate?.toFixed(0)}%</td>
                    <td className="py-0.5 px-1 text-right font-mono"
                        style={{ color: _pctColor(s.delta_vs_hold_mean) }}>
                      {_pct(s.delta_vs_hold_mean)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Layer 6: Trade Lifecycle PnL */}
      <div className="rounded p-2 mb-3" style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}>
        <div className="text-[11px] uppercase font-bold mb-1.5" style={{ color: meta.color }}>
          Layer 6 — Trade Lifecycle: Managed vs Buy-and-Hold (n={lc.n})
        </div>
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
              <th className="text-left py-0.5 px-1" style={{ color: C.gray }}>Metric</th>
              <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Managed (PM+Trading)</th>
              <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Buy-and-Hold (PM only)</th>
              <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Δ Trading α</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ borderBottom: `1px solid ${C.border}40` }}>
              <td className="py-0.5 px-1 font-semibold">Mean Return</td>
              <td className="py-0.5 px-1 text-right font-mono" style={{ color: _pctColor(lc.managed_mean_return) }}>
                {_pct(lc.managed_mean_return)}
              </td>
              <td className="py-0.5 px-1 text-right font-mono" style={{ color: _pctColor(lc.buyhold_mean_return) }}>
                {_pct(lc.buyhold_mean_return)}
              </td>
              <td className="py-0.5 px-1 text-right font-mono font-bold" style={{ color: _pctColor(lc.delta_alpha) }}>
                {_pct(lc.delta_alpha)}
              </td>
            </tr>
            <tr style={{ borderBottom: `1px solid ${C.border}40` }}>
              <td className="py-0.5 px-1 font-semibold">Win Rate</td>
              <td className="py-0.5 px-1 text-right font-mono">{lc.managed_win_rate?.toFixed(1)}%</td>
              <td className="py-0.5 px-1 text-right font-mono">{lc.buyhold_win_rate?.toFixed(1)}%</td>
              <td className="py-0.5 px-1 text-right font-mono"
                  style={{ color: _pctColor((lc.managed_win_rate || 0) - (lc.buyhold_win_rate || 0)) }}>
                {((lc.managed_win_rate || 0) - (lc.buyhold_win_rate || 0)).toFixed(1)}pp
              </td>
            </tr>
            <tr style={{ borderBottom: `1px solid ${C.border}40` }}>
              <td className="py-0.5 px-1 font-semibold">Sharpe (per trade)</td>
              <td className="py-0.5 px-1 text-right font-mono">{lc.managed_sharpe?.toFixed(2)}</td>
              <td className="py-0.5 px-1 text-right font-mono">{lc.buyhold_sharpe?.toFixed(2)}</td>
              <td className="py-0.5 px-1 text-right font-mono"
                  style={{ color: _pctColor((lc.managed_sharpe || 0) - (lc.buyhold_sharpe || 0)) }}>
                {((lc.managed_sharpe || 0) - (lc.buyhold_sharpe || 0)).toFixed(2)}
              </td>
            </tr>
          </tbody>
        </table>
        <div className="text-[11px] mt-1.5" style={{ color: C.gray }}>
          📊 'Trading helped' picks (managed beat hold): <span style={{ color: C.text, fontWeight: "bold" }}>
            {lc.trading_helped_pct?.toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Layer 7: Urgency Calibration */}
      <div className="rounded p-2 mb-2" style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}>
        <div className="text-[11px] uppercase font-bold mb-1.5" style={{ color: meta.color }}>
          Layer 7 — Urgency Calibration
        </div>
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
              <th className="text-left py-0.5 px-1" style={{ color: C.gray }}>Urgency</th>
              <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>n</th>
              <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>% Moved ≥3% in 3d</th>
              <th className="text-right py-0.5 px-1" style={{ color: C.gray }}>Avg Days→Trigger</th>
              <th className="text-left py-0.5 px-1" style={{ color: C.gray }}>Calibration</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(b.urgency_calibration).map(([urg, stats]) => {
              const s = stats as any;
              if (s.n === 0) return null;
              const pctMoved = s.pct_moved_3pct_in_3d || 0;
              const expectMoveFast = urg === "URGENT";
              const isAccurate = expectMoveFast ? pctMoved > 50 : urg === "PATIENT" ? pctMoved < 40 : true;
              return (
                <tr key={urg} style={{ borderBottom: `1px solid ${C.border}40` }}>
                  <td className="py-0.5 px-1 font-bold" style={{
                    color: urg === "URGENT" ? C.red : urg === "PATIENT" ? C.cyan : C.gray }}>
                    {urg === "URGENT" ? "🚨" : urg === "PATIENT" ? "⏸" : "▶"} {urg}
                  </td>
                  <td className="py-0.5 px-1 text-right font-mono">{s.n}</td>
                  <td className="py-0.5 px-1 text-right font-mono"
                      style={{ color: pctMoved > 50 ? C.green : pctMoved < 30 ? C.red : C.amber }}>
                    {pctMoved.toFixed(0)}%
                  </td>
                  <td className="py-0.5 px-1 text-right font-mono" style={{ color: C.gray }}>
                    {s.avg_days_to_trigger?.toFixed(1) ?? "—"}d
                  </td>
                  <td className="py-0.5 px-1 text-[11px]"
                      style={{ color: isAccurate ? C.green : C.amber }}>
                    {isAccurate ? "✓ accurate" : "⚠ miscal."}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="text-[11px] px-2 py-1.5 rounded mb-3" style={{ backgroundColor: "#FFF1E5", color: C.gray }}>
        <span style={{ color: C.amber, fontWeight: "bold" }}>⚠ Architecture C Phase 1 (Deterministic Proxy):</span>
        {" "}Trading signals are rule-based historical simulation. Real Trading Agent (LLM) signals will be
        captured via forward collection — Phase 2 enables Proxy vs Actual comparison after 1-3 months.
      </div>

      {/* ── Position Timeline Heatmaps — Long Stocks → Long ETFs stacked ── */}
      {lifecycles && (
        <>
          <PositionTimelineHeatmap
            title="📈 Long Stocks — Position Timeline"
            data={lifecycles}
            bucket="long_stocks"
            horizon={horizon}
            color={meta.color}
            endDate={endDate}
          />
          <PositionTimelineHeatmap
            title="📦 Long ETFs — Position Timeline"
            data={lifecycles}
            bucket="long_etfs"
            horizon={horizon}
            color={meta.color}
            endDate={endDate}
          />
        </>
      )}

      {/* ── Validated Buy List — unified timeline (Stocks + ETFs) ── */}
      {mergedLifecycles && !validated.loading && validated.total > 0 && (
        <div className="mt-5 pt-3 rounded p-3"
             style={{
               border: `2px dashed ${C.green}80`,
               backgroundColor: `${C.green}06`,
             }}>
          <div className="mb-3 pb-2 border-b" style={{ borderColor: `${C.green}30` }}>
            <div className="text-[14px] font-bold flex items-center gap-2" style={{ color: C.green }}>
              🏆 검증된 매수 종목 (★★ / ★★★ tier) — 과거 성과 추이
            </div>
            <div className="text-[11px] mt-1" style={{ color: C.gray }}>
              "Final Buy List" 에서 ★★ 이상으로 분류된 {validated.total}개 종목의 시계열.
              backtest에 없는 종목은 동일한 price cache로 synthesized된 과거 성과를 표시.
              {" "}<span style={{ color: C.text }}>
                Stocks: {validated.stocks_with_bt.size}개 · ETFs: {validated.etfs_with_bt.size}개
              </span>
            </div>
          </div>

          {validated.stocks_with_bt.size > 0 ? (
            <PositionTimelineHeatmap
              title="📈 검증된 Long Stocks (★★/★★★) — 과거 성과 추이"
              subtitle={`${validated.stocks_with_bt.size}개 stock의 ${horizon} horizon 결과 — backtest에 없는 종목은 동일 price cache 기반 synthesized 결과 (mr=N/A, BH/BHP만 표시)`}
              data={mergedLifecycles}
              bucket="long_stocks"
              horizon={horizon}
              color={C.green}
              endDate={endDate}
              tickerFilter={validated.stocks_with_bt}
            />
          ) : (
            <div className="text-[12px] px-2 py-3 mb-3 rounded text-center"
                 style={{ backgroundColor: C.bgAlt, color: C.gray, border: `1px solid ${C.border}` }}>
              검증된 매수 Stock 없음
            </div>
          )}

          {validated.etfs_with_bt.size > 0 ? (
            <PositionTimelineHeatmap
              title="📦 검증된 Long ETFs (★★/★★★) — 과거 성과 추이"
              subtitle={`${validated.etfs_with_bt.size}개 ETF의 ${horizon} horizon 결과 — backtest에 없는 종목은 동일 price cache 기반 synthesized 결과`}
              data={mergedLifecycles}
              bucket="long_etfs"
              horizon={horizon}
              color={C.green}
              endDate={endDate}
              tickerFilter={validated.etfs_with_bt}
            />
          ) : (
            <div className="text-[12px] px-2 py-3 rounded text-center"
                 style={{ backgroundColor: C.bgAlt, color: C.gray, border: `1px solid ${C.border}` }}>
              검증된 매수 ETF 없음
            </div>
          )}
        </div>
      )}
      {lifecycles && !validated.loading && validated.total === 0 && (
        <div className="mt-4 text-[12px] px-3 py-2 rounded text-center"
             style={{ backgroundColor: C.bgAlt, color: C.gray, border: `1px solid ${C.border}` }}>
          🏆 검증된 매수 종목 (★★ / ★★★) 없음 — 현재 Final Buy List는 모두 ★ tier (PM only)
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// PositionTimelineHeatmap — visualizes Managed vs Buy-and-Hold portfolio
// composition over time (23 weekly cohorts × top tickers).
// ─────────────────────────────────────────────────────────────────────

type TimelineView = "managed" | "buyhold" | "delta";

// Trading days per horizon. Converted to calendar days for in-flight detection.
const HORIZON_TRADING_DAYS: Record<TradingHorizon, number> = {
  tactical: 5, core: 21, strategic: 63,
};
// ~5/7 trading days per calendar week → calendar = trading × 1.45 + 1 buffer
function tradingToCalendarDays(td: number) { return Math.ceil(td * 1.45) + 1; }

function PositionTimelineHeatmap({ title, data, bucket, horizon, color, endDate,
                                   tickerFilter, subtitle, maxTickers }: {
  title: string;
  data: TradingLifecyclesCompact;
  bucket: "long_stocks" | "long_etfs" | "short_stocks" | "short_etfs";
  horizon: TradingHorizon;
  color: string;
  endDate?: string;
  tickerFilter?: Set<string>;        // when set, restricts rows to these tickers (no top-N cap)
  subtitle?: string;                  // optional descriptive subtitle
  maxTickers?: number;                // override default 25
}) {
  const [view, setView] = useState<TimelineView>("buyhold");

  const allRecords: LifecycleRecord[] = (data[horizon]?.[bucket] || []) as LifecycleRecord[];
  const records: LifecycleRecord[] = tickerFilter
    ? allRecords.filter((r) => tickerFilter.has(r.t))
    : allRecords;

  // In-flight detection: cohort entered too recently to complete the horizon
  // (need enough trading days AFTER the cohort date before end_date).
  const inflightFromDate = useMemo<string | null>(() => {
    if (!endDate) return null;
    const end = new Date(endDate + "T00:00:00");
    const needCalendar = tradingToCalendarDays(HORIZON_TRADING_DAYS[horizon]);
    const cutoff = new Date(end);
    cutoff.setDate(cutoff.getDate() - needCalendar);
    return cutoff.toISOString().slice(0, 10);
  }, [endDate, horizon]);
  const isInflightDate = (d: string) =>
    inflightFromDate != null && d > inflightFromDate;

  // Index by (ticker, date) for grid construction
  const { tickers, dates, grid, tickerNames } = useMemo(() => {
    const dateSet = new Set<string>();
    const tickerCount: Record<string, number> = {};
    const g: Record<string, Record<string, LifecycleRecord>> = {};
    const names: Record<string, string> = {};
    for (const r of records) {
      if (!r.t || !r.d) continue;
      dateSet.add(r.d);
      tickerCount[r.t] = (tickerCount[r.t] || 0) + 1;
      g[r.t] = g[r.t] || {};
      g[r.t][r.d] = r;
      if (r.n && !names[r.t]) names[r.t] = r.n;
    }
    const dates = Array.from(dateSet).sort();
    // When tickerFilter is provided, show ALL filtered tickers; otherwise cap at maxTickers (default 25)
    const cap = tickerFilter ? Infinity : (maxTickers ?? 25);
    const tickers = Object.entries(tickerCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, cap)
      .map(([t]) => t);
    return { tickers, dates, grid: g, tickerNames: names };
  }, [records, tickerFilter, maxTickers]);

  // Color scale for returns: red (negative) → white (zero) → green (positive)
  function returnColor(r: number | null | undefined, magScale = 0.15): string {
    if (r == null) return "transparent";
    const v = Math.max(-1, Math.min(1, r / magScale));
    if (v > 0) {
      const a = Math.min(0.85, 0.20 + v * 0.65);
      return `rgba(70, 200, 120, ${a.toFixed(2)})`;
    } else {
      const a = Math.min(0.85, 0.20 + Math.abs(v) * 0.65);
      return `rgba(230, 80, 80, ${a.toFixed(2)})`;
    }
  }
  function deltaColor(d: number | null | undefined, magScale = 0.10): string {
    if (d == null) return "transparent";
    const v = Math.max(-1, Math.min(1, d / magScale));
    if (v > 0) {
      // Trading helped (blue)
      const a = Math.min(0.85, 0.20 + v * 0.65);
      return `rgba(80, 150, 220, ${a.toFixed(2)})`;
    } else {
      // Trading hurt (orange)
      const a = Math.min(0.85, 0.20 + Math.abs(v) * 0.65);
      return `rgba(230, 150, 60, ${a.toFixed(2)})`;
    }
  }

  // In-flight cell style — diagonal stripe pattern with cyan tint
  const INFLIGHT_BG =
    "repeating-linear-gradient(45deg, rgba(100,180,220,0.18) 0 4px, rgba(100,180,220,0.32) 4px 8px)";

  // Border style for in-flight (cohort hasn't completed horizon)
  const INFLIGHT_BORDER = "1.5px dashed rgba(100,180,220,0.85)";

  function cellInfo(rec: LifecycleRecord | undefined, cohortDate: string): {
    color: string; text: string; tooltip: string;
    isInflight: boolean; border?: string;
  } {
    if (!rec) return { color: "transparent", text: "", tooltip: "", isInflight: false };
    const mr = rec.mr;
    const bh = rec.bh;             // full-horizon BH (null if in-flight)
    const bhp = rec.bhp;           // partial BH (MTM at latest price)
    const bhd = rec.bhd;           // trading days elapsed

    // Effective BH for display: full if available, else partial
    const bhEff = bh != null ? bh : bhp;
    // Delta uses partial BH if needed (so in-flight cells still show comparison)
    const delta = (mr != null && bhEff != null) ? (mr - bhEff) : null;

    const inflight = isInflightDate(cohortDate) && bh == null;
    const horizonDays = HORIZON_TRADING_DAYS[horizon];
    const progressPct = inflight && bhd != null
      ? Math.min(100, Math.round(bhd / horizonDays * 100))
      : null;

    let cellColor = "transparent", text = "";
    if (view === "managed") {
      cellColor = returnColor(mr);
      text = mr != null ? `${(mr*100).toFixed(0)}` : "—";
    } else if (view === "buyhold") {
      cellColor = returnColor(bhEff);
      text = bhEff != null ? `${(bhEff*100).toFixed(0)}` : "—";
    } else {
      cellColor = deltaColor(delta);
      text = delta != null ? `${(delta*100).toFixed(0)}` : "—";
    }
    // In-flight indicator: prepend ~ if showing partial value
    if (inflight && text && text !== "—") text = "~" + text;

    const tooltip = `${rec.t}  @  ${rec.d}\n` +
      `Signal: ${rec.sig} · State: ${rec.mst || "?"} (${rec.mex || "—"})\n` +
      (inflight
        ? `🕒 IN PROGRESS — ${bhd ?? "?"}/${horizonDays} trading days elapsed (${progressPct ?? "?"}%)\n` +
          `(showing partial return as MTM at latest price)\n`
        : "") +
      `Managed P&L : ${mr != null ? (mr*100).toFixed(2)+"%" : "N/A"}` +
      (rec.mdh != null ? `  · ${rec.mdh}d held` : "") +
      `\nBuy-and-Hold (full): ${bh != null ? (bh*100).toFixed(2)+"%" : "(in-flight)"}` +
      (bhp != null && bh == null
        ? `\nBuy-and-Hold (partial @${bhd}d): ${(bhp*100).toFixed(2)}%`
        : "") +
      (delta != null ? `\nΔ vs ${bh != null ? "BH" : "partial BH"}: ${(delta*100).toFixed(2)}%` : "");

    return { color: cellColor, text, tooltip, isInflight: inflight,
             border: inflight ? INFLIGHT_BORDER : undefined };
  }

  // Aggregate per-week summary stats for footer
  // For in-flight cohorts, falls back to partial BH so users can still see direction.
  const weekSummary = useMemo(() => {
    return dates.map((d) => {
      const recs = records.filter((r) => r.d === d);
      const inflight = isInflightDate(d);
      const mret = recs.filter((r) => r.mr != null).map((r) => r.mr!);
      const bretFull    = recs.filter((r) => r.bh  != null).map((r) => r.bh!);
      const bretPartial = recs.filter((r) => r.bhp != null).map((r) => r.bhp!);
      // Use partial when full not available (in-flight cohorts)
      const bret = bretFull.length ? bretFull : bretPartial;
      const mAvg = mret.length ? mret.reduce((a, b) => a + b, 0) / mret.length : null;
      const bAvg = bret.length ? bret.reduce((a, b) => a + b, 0) / bret.length : null;
      return {
        d, n: recs.length, mAvg, bAvg,
        delta: (mAvg != null && bAvg != null) ? mAvg - bAvg : null,
        inflight,
        isPartial: inflight && bretFull.length === 0 && bretPartial.length > 0,
      };
    });
  }, [records, dates, inflightFromDate]);

  // Early return AFTER all hooks (Rules of Hooks compliance)
  if (records.length === 0) {
    return (
      <div className="rounded p-2 mt-2" style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}>
        <div className="text-[12px]" style={{ color: C.gray }}>{title} — no data</div>
      </div>
    );
  }

  const colW = Math.max(28, Math.floor(900 / Math.max(dates.length, 1)));

  return (
    <div className="rounded p-2 mt-3" style={{ backgroundColor: C.bgAlt, border: `1px solid ${color}40` }}>
      <div className="flex items-center justify-between mb-2 pb-1 border-b" style={{ borderColor: C.border }}>
        <div>
          <div className="text-[12px] uppercase font-bold" style={{ color }}>
            {title} <span style={{ color: C.gray, fontWeight: "normal" }}>
              ({tickers.length} tickers · {dates.length} weekly cohorts · {records.length} picks)
            </span>
          </div>
          {subtitle && (
            <div className="text-[11px] mt-0.5" style={{ color: C.gray, fontStyle: "italic" }}>
              {subtitle}
            </div>
          )}
        </div>
        <div className="flex items-center gap-1">
          {([
            { k: "buyhold", label: "Buy & Hold",  emoji: "📦" },
            { k: "managed", label: "Managed",     emoji: "🎯" },
            { k: "delta",   label: "Δ Trading α", emoji: "📊" },
          ] as { k: TimelineView; label: string; emoji: string }[]).map((opt) => {
            const active = view === opt.k;
            return (
              <button key={opt.k} onClick={() => setView(opt.k)}
                      className="px-2 py-0.5 rounded text-[11px] transition-colors"
                      style={{
                        backgroundColor: active ? color + "30" : "transparent",
                        color: active ? color : C.gray,
                        border: `1px solid ${active ? color + "80" : C.border}`,
                        fontWeight: active ? "bold" : "normal",
                      }}>
                {opt.emoji} {opt.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="overflow-auto" style={{ maxHeight: 420 }}>
        <table className="border-collapse" style={{ fontSize: 11, minWidth: "100%" }}>
          <thead className="sticky top-0" style={{ backgroundColor: C.bgAlt, zIndex: 1 }}>
            {/* Row 1 — "진입시점" group label spanning the date columns */}
            <tr>
              <th className="px-1 py-0.5 sticky left-0"
                  style={{ backgroundColor: C.bgAlt, minWidth: 200 }}></th>
              <th colSpan={dates.length}
                  className="text-center px-1 py-0.5"
                  style={{
                    color: color, fontSize: 11, fontWeight: "bold",
                    letterSpacing: 1,
                    borderBottom: `1px solid ${color}40`,
                  }}>
                ▸ 진입시점 (Entry Date — Weekly Cohort) ◂
              </th>
              <th className="sticky right-0" style={{ backgroundColor: C.bgAlt, minWidth: 40 }}></th>
            </tr>
            {/* Row 2 — individual date column headers */}
            <tr>
              <th className="text-left px-1 py-0.5 sticky left-0"
                  style={{ color: C.gray, backgroundColor: C.bgAlt, minWidth: 200 }}>
                Ticker — Name
              </th>
              {dates.map((d) => (
                <th key={d} className="text-center px-0.5 py-0.5"
                    style={{ color: C.gray, minWidth: colW, fontSize: 10 }}>
                  {d.slice(5)}
                </th>
              ))}
              <th className="text-right px-1 py-0.5 sticky right-0"
                  style={{ color: C.gray, backgroundColor: C.bgAlt, minWidth: 40 }}>
                Freq
              </th>
            </tr>
          </thead>
          <tbody>
            {tickers.map((t) => {
              const freq = dates.reduce((n: number, d: string) => n + (grid[t]?.[d] ? 1 : 0), 0);
              return (
                <tr key={t} style={{ borderBottom: `1px solid ${C.border}30` }}>
                  <td className="px-1 py-0.5 sticky left-0"
                      style={{ color: C.text, backgroundColor: C.bgAlt, minWidth: 200, maxWidth: 220 }}
                      title={tickerNames[t] || t}>
                    <span className="font-mono font-bold" style={{ color: C.text }}>{t}</span>
                    {tickerNames[t] && (
                      <span className="ml-1.5" style={{ color: C.gray, fontWeight: "normal", fontSize: 11 }}>
                        {tickerNames[t].length > 22 ? tickerNames[t].slice(0, 22) + "…" : tickerNames[t]}
                      </span>
                    )}
                  </td>
                  {dates.map((d: string) => {
                    const rec = grid[t]?.[d];
                    const ci = cellInfo(rec, d);
                    // In-flight cells get a dashed outline; otherwise fully colored
                    return (
                      <td key={d} title={ci.tooltip}
                          className="text-center cursor-help"
                          style={{
                            background: ci.color,
                            color: ci.text ? "rgba(255,255,255,0.92)" : "transparent",
                            fontSize: 10,
                            fontWeight: "bold",
                            padding: "2px 1px",
                            minWidth: colW,
                            outline: ci.border,
                            outlineOffset: ci.border ? "-2px" : undefined,
                          }}>
                        {ci.text}
                      </td>
                    );
                  })}
                  <td className="text-right px-1 py-0.5 font-mono sticky right-0"
                      style={{ color: C.cyan, backgroundColor: C.bgAlt }}>
                    {freq}
                  </td>
                </tr>
              );
            })}
            {/* Footer: weekly aggregate stats */}
            <tr style={{ borderTop: `2px solid ${C.border}` }}>
              <td className="px-1 py-0.5 font-bold sticky left-0"
                  style={{ color: C.amber, backgroundColor: C.bgAlt, minWidth: 200 }}>
                Wk avg (all picks)
              </td>
              {weekSummary.map((w: { d: string; n: number; mAvg: number | null; bAvg: number | null; delta: number | null; inflight: boolean; isPartial: boolean }) => {
                const v = view === "managed" ? w.mAvg : view === "buyhold" ? w.bAvg : w.delta;
                const cellBg = view === "delta" ? deltaColor(v) : returnColor(v);
                const textVal = v != null ? (w.isPartial ? "~" : "") + `${(v*100).toFixed(0)}` : "—";
                return (
                  <td key={w.d} title={(w.inflight ? `🕒 IN PROGRESS — partial values${w.isPartial ? " (showing MTM)" : ""}\n` : "") +
                                       `Week ${w.d}\nManaged avg: ${w.mAvg != null ? (w.mAvg*100).toFixed(2)+"%" : "—"}\nBuy & Hold avg: ${w.bAvg != null ? (w.bAvg*100).toFixed(2)+"%" + (w.isPartial ? " (partial)" : "") : "—"}`}
                      className="text-center cursor-help font-bold"
                      style={{
                        background: cellBg,
                        color: "white", fontSize: 10, padding: "2px 1px",
                        outline: w.inflight ? INFLIGHT_BORDER : undefined,
                        outlineOffset: w.inflight ? "-2px" : undefined,
                      }}>
                    {textVal}
                  </td>
                );
              })}
              <td className="px-1 py-0.5 sticky right-0" style={{ backgroundColor: C.bgAlt }}></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-3 mt-2 text-[11px]" style={{ color: C.gray }}>
        <span>
          <span style={{ color: C.gray }}>Legend ({view}):</span>{" "}
          {view === "managed" && "Managed return at exit. "}
          {view === "buyhold" && "Buy-and-Hold horizon return. "}
          {view === "delta" && (
            <>
              <span style={{ color: "rgb(80,150,220)" }}>blue</span> = Trading α positive ·{" "}
              <span style={{ color: "rgb(230,150,60)" }}>orange</span> = Trading α negative
            </>
          )}
          {inflightFromDate && (
            <>
              {" · "}
              <span style={{
                display: "inline-block", width: 18, height: 12, verticalAlign: "middle",
                outline: INFLIGHT_BORDER, outlineOffset: -2,
                background: "rgba(100,180,220,0.15)",
                borderRadius: 2, marginRight: 3,
              }}></span>
              🕒 in-flight (cohort after {inflightFromDate}) — values shown as <span style={{color: C.text, fontWeight:"bold"}}>~partial MTM</span>
            </>
          )}
        </span>
        <span style={{ marginLeft: "auto" }}>
          Cell value: <span style={{ color: C.text, fontWeight: "bold" }}>return % at horizon</span> · hover for details
        </span>
      </div>
    </div>
  );
}
