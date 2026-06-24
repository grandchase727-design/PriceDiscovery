import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { fetchFactorEfficacy } from "../../api/client";
import { MetricCard } from "../shared/MetricCard";
import { DataTable } from "../shared/DataTable";
import { C, DARK_LAYOUT } from "../../styles/theme";

/* ------------------------------------------------------------------ */
/*  Sub-tab state                                                      */
/* ------------------------------------------------------------------ */
const SUB_TABS = [
  "Ticker Signals",
  "Unified Ranking",
  "1. Fama-MacBeth",
  "2. IC Analysis",
  "3. Long-Short",
  "4. PCA Model",
  "5. Regime",
] as const;

export function FactorEfficacyTab() {
  const [raw, setRaw] = useState<any>(null);
  const [sub, setSub] = useState(0);

  useEffect(() => { fetchFactorEfficacy().then(setRaw); }, []);

  if (!raw) return <div className="text-[#857F7A] p-8">Loading...</div>;
  if (raw.error) return <div className="text-[#B85C00] p-8">{raw.error}</div>;

  const unified = raw.unified ?? {};
  const fm = raw.fama_macbeth ?? {};
  const ic = raw.ic_analysis ?? {};
  const ls = raw.long_short ?? {};
  const pca = raw.pca ?? {};
  const rc = raw.regime_conditional ?? {};
  const panel = raw.panel_size ?? {};
  const mh = raw.multi_horizon_ic ?? {};
  const ts = raw.ticker_signals ?? {};

  return (
    <div className="space-y-6">

      {/* ── Header KPIs ── */}
      <div className="grid grid-cols-5 gap-3">
        <MetricCard label="Current Regime" value={rc.current_regime ?? "?"} />
        <MetricCard label="PCA Eff. Dimensions" value={pca.effective_dimensionality ?? "?"} />
        <MetricCard label="Eval Points" value={panel.n_eval_points ?? "?"} />
        <MetricCard label="Avg Universe" value={`${(panel.avg_tickers ?? 0).toFixed(0)}`} />
        <MetricCard label="Top Factor" value={unified.top3_factors?.[0] ?? "?"} />
      </div>

      {unified.top3_factors && (
        <div className="bg-[#FFFFFF] rounded-lg p-3 border border-[#E6D9CE] text-[16px]">
          <span className="text-[#66605C]">Top 3 Factors: </span>
          <span className="text-[#0F5499] font-semibold">{unified.top3_factors.join(", ")}</span>
          <span className="text-[#857F7A] mx-3">|</span>
          <span className="text-[#66605C]">Top 3 Groups: </span>
          <span className="text-[#0F5499] font-semibold">{unified.top3_groups?.join(", ")}</span>
        </div>
      )}

      {/* ── Sub-tab bar ── */}
      <div className="flex items-center gap-1 border-b border-[#E6D9CE] pb-0">
        {SUB_TABS.map((t, i) => (
          <button key={t}
            className={`px-4 py-2 text-[16px] border-b-2 transition-colors ${
              sub === i
                ? "border-[#0F5499] text-[#0F5499]"
                : "border-transparent text-[#857F7A] hover:text-[#33302E]"
            }`}
            onClick={() => setSub(i)}>
            {t}
          </button>
        ))}
      </div>

      {/* ── Sub-tab content ── */}
      {sub === 0 && <TickerSignalsSection ts={ts} />}
      {sub === 1 && <UnifiedSection unified={unified} />}
      {sub === 2 && <FamaMacBethSection fm={fm} />}
      {sub === 3 && <ICAnalysisSection ic={ic} mh={mh} />}
      {sub === 4 && <LongShortSection ls={ls} />}
      {sub === 5 && <PCASection pca={pca} />}
      {sub === 6 && <RegimeSection rc={rc} />}
    </div>
  );
}


/* ================================================================== */
/*  Section: Ticker Signals (Factor-Implied Directional View)          */
/* ================================================================== */
const SIGNAL_COLORS: Record<string, string> = {
  STRONG_LONG: C.green, LONG: C.cyan, SHORT: C.orange, STRONG_SHORT: C.red,
};
const SIGNAL_BG: Record<string, string> = {
  STRONG_LONG: "bg-emerald-900/40 border-emerald-600",
  LONG: "bg-[#E3EEF5]/40 border-[#9CC3D5]",
  SHORT: "bg-[#F7EDE0]/40 border-[#E0C3A0]",
  STRONG_SHORT: "bg-[#F7E3E3]/40 border-[#E0AAAA]",
};

function TickerSignalsSection({ ts }: { ts: any }) {
  const signals: any[] = ts.ticker_signals ?? [];
  const factorsUsed: any[] = ts.signal_factors_used ?? [];
  const dist = ts.signal_distribution ?? {};
  const [filter, setFilter] = useState<string>("ALL");

  if (!signals.length) return <div className="text-[#857F7A] p-8">No ticker signal data available.</div>;

  const filtered = filter === "ALL" ? signals : signals.filter((s: any) => s.signal === filter);

  // Group by signal for the 4-panel view
  const grouped: Record<string, any[]> = { STRONG_LONG: [], LONG: [], SHORT: [], STRONG_SHORT: [] };
  signals.forEach((s: any) => { if (grouped[s.signal]) grouped[s.signal].push(s); });

  return (
    <div className="space-y-8">
      <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide">
        Factor-Implied Ticker Signals
      </h3>
      <p className="text-[13px] text-[#857F7A]">
        IC 분석에서 유효한 팩터 Top 10의 IC_IR 가중합산 → 종목별 팩터 시그널 점수 → 백분위 기반 4분류.
        Eval Date: {ts.eval_date ?? "?"}
      </p>

      {/* Signal distribution KPIs */}
      <div className="grid grid-cols-5 gap-3">
        <MetricCard label="Total Tickers" value={ts.n_tickers ?? 0} />
        <MetricCard label="STRONG LONG" value={dist.STRONG_LONG ?? 0} sub={`${((dist.STRONG_LONG ?? 0) / (ts.n_tickers || 1) * 100).toFixed(0)}%`} />
        <MetricCard label="LONG" value={dist.LONG ?? 0} sub={`${((dist.LONG ?? 0) / (ts.n_tickers || 1) * 100).toFixed(0)}%`} />
        <MetricCard label="SHORT" value={dist.SHORT ?? 0} sub={`${((dist.SHORT ?? 0) / (ts.n_tickers || 1) * 100).toFixed(0)}%`} />
        <MetricCard label="STRONG SHORT" value={dist.STRONG_SHORT ?? 0} sub={`${((dist.STRONG_SHORT ?? 0) / (ts.n_tickers || 1) * 100).toFixed(0)}%`} />
      </div>

      {/* Factors used */}
      {factorsUsed.length > 0 && (
        <details className="bg-[#FFFFFF] rounded border border-[#E6D9CE] p-3">
          <summary className="text-[16px] text-[#66605C] cursor-pointer">Signal Factors Used ({factorsUsed.length} factors)</summary>
          <div className="mt-2 grid grid-cols-5 gap-2">
            {factorsUsed.map((f: any) => (
              <div key={f.factor} className="bg-[#F2E5D7] rounded p-2 text-[14px]">
                <div className="font-semibold text-[#33302E]">{f.factor}</div>
                <div className="text-[#857F7A]">
                  wt={f.weight?.toFixed(3)} | IR={f.ic_ir?.toFixed(2)} | {f.direction} | {f.quality}
                </div>
              </div>
            ))}
          </div>
        </details>
      )}

      {/* 4-panel signal groups */}
      <div className="grid grid-cols-4 gap-4">
        {(["STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"] as const).map((sig) => (
          <div key={sig} className={`rounded-lg border p-3 ${SIGNAL_BG[sig]}`}>
            <h4 className="text-[16px] font-bold mb-2" style={{ color: SIGNAL_COLORS[sig] }}>
              {sig.replace("_", " ")} ({grouped[sig]?.length ?? 0})
            </h4>
            <div className="max-h-[360px] overflow-y-auto space-y-1">
              {(grouped[sig] ?? []).slice(0, 30).map((s: any) => (
                <div key={s.ticker} className="flex items-center justify-between text-[14px] bg-black/30 rounded px-2 py-1">
                  <div>
                    <span className="font-mono font-semibold text-[#33302E]">{s.ticker}</span>
                    <span className="text-[#857F7A] ml-1.5">{s.category?.replace("STK_", "").replace("EQ_", "")}</span>
                  </div>
                  <span className="font-mono" style={{ color: SIGNAL_COLORS[sig] }}>
                    {s.score_pctile?.toFixed(0)}p
                  </span>
                </div>
              ))}
              {(grouped[sig]?.length ?? 0) > 30 && (
                <div className="text-[12px] text-[#857F7A] text-center">+{(grouped[sig]?.length ?? 0) - 30} more</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Filter + Full table */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <span className="text-[16px] text-[#66605C]">Filter:</span>
          {["ALL", "STRONG_LONG", "LONG", "SHORT", "STRONG_SHORT"].map((f) => (
            <button key={f}
              className={`px-3 py-1 rounded text-[14px] font-medium transition-colors ${
                filter === f ? "text-[#33302E]" : "text-[#857F7A] hover:text-[#33302E]"
              }`}
              style={filter === f ? { backgroundColor: SIGNAL_COLORS[f] ?? C.cyan } : { backgroundColor: "#F2E5D7" }}
              onClick={() => setFilter(f)}>
              {f.replace("_", " ")}
            </button>
          ))}
          <span className="text-[14px] text-[#857F7A] ml-auto">{filtered.length} tickers</span>
        </div>

        <DataTable data={filtered} columns={[
          { accessorKey: "ticker", header: "Ticker", size: 70,
            cell: (p: any) => <span className="font-mono font-semibold">{p.getValue()}</span> },
          { accessorKey: "name", header: "Name",
            cell: (p: any) => <span className="text-[14px] text-[#66605C] truncate block max-w-[180px]">{p.getValue()}</span> },
          { accessorKey: "category", header: "Category", size: 100,
            cell: (p: any) => <span className="text-[14px]">{(p.getValue() as string)?.replace("STK_", "").replace("EQ_", "")}</span> },
          { accessorKey: "signal", header: "Signal", size: 110,
            cell: (p: any) => {
              const v = p.getValue() as string;
              return <span className="font-semibold text-[14px] px-2 py-0.5 rounded"
                style={{ color: SIGNAL_COLORS[v], backgroundColor: `${SIGNAL_COLORS[v]}20` }}>
                {v?.replace("_", " ")}
              </span>;
            }},
          { accessorKey: "factor_score", header: "F-Score", size: 80,
            cell: (p: any) => <span className="font-mono">{p.getValue()?.toFixed(3)}</span> },
          { accessorKey: "score_pctile", header: "Pctile", size: 60,
            cell: (p: any) => <span className="font-mono">{p.getValue()?.toFixed(0)}</span> },
          { accessorKey: "top_contributions", header: "Top Contributing Factors",
            cell: (p: any) => {
              const contribs: any[] = p.getValue() ?? [];
              return <span className="text-[12px] text-[#66605C]">
                {contribs.map((c: any) => `${c.factor}(${c.contribution > 0 ? "+" : ""}${c.contribution.toFixed(3)})`).join(", ")}
              </span>;
            }},
        ]} maxHeight="600px" />
      </div>
    </div>
  );
}


/* ================================================================== */
/*  Section 0: Unified Ranking                                         */
/* ================================================================== */
function UnifiedSection({ unified }: { unified: any }) {
  const ranking: any[] = unified.unified_ranking ?? [];
  const grpRank: any[] = unified.group_ranking ?? [];

  return (
    <div className="space-y-8">
      <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide">
        5-Method Composite Factor Ranking
      </h3>
      <p className="text-[13px] text-[#857F7A]">FM 25% + IC 30% + LS 25% + PCA 10% + Regime 10%</p>

      {/* Horizontal bar */}
      {ranking.length > 0 && (
        <Plot
          data={[{
            type: "bar", orientation: "h",
            y: ranking.slice(0, 15).map((r: any) => r.factor).reverse(),
            x: ranking.slice(0, 15).map((r: any) => r.unified_score).reverse(),
            marker: { color: ranking.slice(0, 15).map((r: any) => r.unified_score >= 50 ? C.cyan : r.unified_score >= 30 ? C.blue : C.gray).reverse() },
            text: ranking.slice(0, 15).map((r: any) => r.unified_score.toFixed(1)).reverse(),
            textposition: "outside",
          }]}
          layout={{ ...DARK_LAYOUT, height: 480, margin: { t: 10, b: 40, l: 160, r: 60 }, xaxis: { title: "Composite Score" } }}
          config={{ responsive: true }} style={{ width: "100%" }}
        />
      )}

      {/* Table */}
      <DataTable data={ranking} columns={[
        { accessorKey: "rank", header: "Rk", size: 40 },
        { accessorKey: "factor", header: "Factor" },
        { accessorKey: "group", header: "Group" },
        { accessorKey: "unified_score", header: "Score", cell: (p: any) => p.getValue()?.toFixed(1) },
        { accessorKey: "fm_tstat", header: "FM t", cell: (p: any) => p.getValue()?.toFixed(2) },
        { accessorKey: "ic_ir", header: "IC IR", cell: (p: any) => p.getValue()?.toFixed(3) },
        { accessorKey: "ls_sharpe", header: "LS SR", cell: (p: any) => p.getValue()?.toFixed(3) },
        { accessorKey: "pca_bonus", header: "PCA", cell: (p: any) => p.getValue()?.toFixed(1) },
        { accessorKey: "regime_bonus", header: "Regime", cell: (p: any) => p.getValue()?.toFixed(1) },
      ]} maxHeight="400px" />

      {/* Group ranking */}
      {grpRank.length > 0 && (
        <>
          <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide mt-8">
            Factor Group Ranking
          </h3>
          <div className="grid grid-cols-2 gap-6">
            <DataTable data={grpRank} columns={[
              { accessorKey: "group", header: "Group" },
              { accessorKey: "avg_score", header: "Avg Score", cell: (p: any) => p.getValue()?.toFixed(1) },
              { accessorKey: "max_score", header: "Max Score", cell: (p: any) => p.getValue()?.toFixed(1) },
              { accessorKey: "n_factors", header: "N" },
              { accessorKey: "top_factor", header: "Top Factor" },
            ]} maxHeight="400px" />
            <Plot
              data={[
                { type: "bar", x: grpRank.map((g: any) => g.group), y: grpRank.map((g: any) => g.avg_score), name: "Avg", marker: { color: C.cyan } },
                { type: "bar", x: grpRank.map((g: any) => g.group), y: grpRank.map((g: any) => g.max_score), name: "Max", marker: { color: C.blue }, opacity: 0.5 },
              ]}
              layout={{ ...DARK_LAYOUT, barmode: "overlay", height: 350, margin: { t: 10, b: 80, l: 50, r: 20 }, xaxis: { tickangle: -45 } }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
          </div>
        </>
      )}
    </div>
  );
}


/* ================================================================== */
/*  Section 1: Fama-MacBeth                                            */
/* ================================================================== */
function FamaMacBethSection({ fm }: { fm: any }) {
  const fp: any[] = fm.factor_premiums ?? [];
  const sig = fp.filter((f: any) => f.significance === "**" || f.significance === "***");

  return (
    <div className="space-y-8">
      <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide">
        Fama-MacBeth Cross-Sectional Regression
      </h3>
      <p className="text-[13px] text-[#857F7A]">
        R_i = alpha + sum(lambda_k * F_k,i) — 팩터 프리미엄 lambda의 시계열 평균과 t-stat으로 유의성 판정
      </p>

      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Significant (p<0.05)" value={sig.length} sub={`of ${fp.length} factors`} />
        <MetricCard label="Eval Periods" value={fm.n_periods ?? "?"} />
        <MetricCard label="Top Factor" value={fp[0]?.factor ?? "?"} sub={`t=${fp[0]?.t_stat?.toFixed(2)} ${fp[0]?.significance}`} />
      </div>

      {/* T-stat bar */}
      {fp.length > 0 && (
        <Plot
          data={[{
            type: "bar",
            x: fp.slice(0, 20).map((f: any) => f.factor),
            y: fp.slice(0, 20).map((f: any) => f.t_stat),
            marker: { color: fp.slice(0, 20).map((f: any) => f.t_stat > 1.96 ? C.green : f.t_stat < -1.96 ? C.red : C.gray) },
          }]}
          layout={{
            ...DARK_LAYOUT, height: 380, margin: { t: 10, b: 80, l: 50, r: 20 },
            xaxis: { tickangle: -45 }, yaxis: { title: "t-stat" },
            shapes: [
              { type: "line", y0: 1.96, y1: 1.96, x0: 0, x1: 1, xref: "paper", line: { color: C.yellow, dash: "dash" } },
              { type: "line", y0: -1.96, y1: -1.96, x0: 0, x1: 1, xref: "paper", line: { color: C.yellow, dash: "dash" } },
              { type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray, dash: "dot" } },
            ],
          }}
          config={{ responsive: true }} style={{ width: "100%" }}
        />
      )}

      <DataTable data={fp} columns={[
        { accessorKey: "factor", header: "Factor" },
        { accessorKey: "group", header: "Group" },
        { accessorKey: "mean_premium", header: "lambda", cell: (p: any) => p.getValue()?.toFixed(4) },
        { accessorKey: "t_stat", header: "t-stat", cell: (p: any) => p.getValue()?.toFixed(3) },
        { accessorKey: "significance", header: "Sig" },
        { accessorKey: "pct_positive", header: "+%", cell: (p: any) => `${p.getValue()?.toFixed(1)}%` },
        { accessorKey: "n_periods", header: "N" },
      ]} maxHeight="400px" />
    </div>
  );
}


/* ================================================================== */
/*  Section 2: IC Analysis                                             */
/* ================================================================== */
function ICAnalysisSection({ ic, mh }: { ic: any; mh: any }) {
  const fi: any[] = ic.factor_ic ?? [];
  const [selFactor, setSelFactor] = useState(0);
  const strong = fi.filter((f: any) => f.quality === "STRONG");
  const moderate = fi.filter((f: any) => f.quality === "MODERATE");

  return (
    <div className="space-y-8">
      <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide">
        Information Coefficient (IC) Analysis
      </h3>
      <p className="text-[13px] text-[#857F7A]">
        IC = Spearman(Factor, Forward Return). IC_IR = IC_mean / IC_std. IR &ge; 0.5 = 유의미한 예측력.
      </p>

      <div className="grid grid-cols-4 gap-3">
        <MetricCard label="STRONG Factors" value={strong.length} />
        <MetricCard label="MODERATE Factors" value={moderate.length} />
        <MetricCard label="Avg IC" value={fi.length ? (fi.reduce((a: number, f: any) => a + f.ic_mean, 0) / fi.length).toFixed(4) : "?"} />
        <MetricCard label="Avg IC IR" value={fi.length ? (fi.reduce((a: number, f: any) => a + f.ic_ir, 0) / fi.length).toFixed(3) : "?"} />
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* IC IR bar */}
        {fi.length > 0 && (
          <Plot
            data={[{
              type: "bar",
              x: fi.slice(0, 15).map((f: any) => f.factor),
              y: fi.slice(0, 15).map((f: any) => Math.abs(f.ic_ir)),
              marker: { color: fi.slice(0, 15).map((f: any) =>
                f.quality === "STRONG" ? C.green : f.quality === "MODERATE" ? C.blue : f.quality === "WEAK" ? C.yellow : C.gray
              ) },
            }]}
            layout={{
              ...DARK_LAYOUT, height: 350, margin: { t: 10, b: 80, l: 50, r: 20 },
              xaxis: { tickangle: -45 }, yaxis: { title: "|IC IR|" },
              shapes: [{ type: "line", y0: 0.5, y1: 0.5, x0: 0, x1: 1, xref: "paper", line: { color: C.yellow, dash: "dash" } }],
            }}
            config={{ responsive: true }} style={{ width: "100%" }}
          />
        )}

        {/* IC mean vs IR scatter */}
        {fi.length > 0 && (
          <Plot
            data={[{
              type: "scatter", mode: "markers+text",
              x: fi.map((f: any) => f.ic_mean),
              y: fi.map((f: any) => f.ic_ir),
              text: fi.map((f: any) => f.factor),
              textposition: "top center",
              textfont: { size: 8, color: C.text },
              marker: {
                size: 10,
                color: fi.map((f: any) => ({ STRONG: C.green, MODERATE: C.blue, WEAK: C.yellow, NOISE: C.gray }[f.quality as string] ?? C.gray)),
              },
            }]}
            layout={{
              ...DARK_LAYOUT, height: 350, margin: { t: 10, b: 40, l: 50, r: 20 },
              xaxis: { title: "IC Mean" }, yaxis: { title: "IC IR" },
              shapes: [
                { type: "line", y0: 0.5, y1: 0.5, x0: 0, x1: 1, xref: "paper", line: { color: C.yellow, dash: "dash" } },
                { type: "line", y0: -0.5, y1: -0.5, x0: 0, x1: 1, xref: "paper", line: { color: C.yellow, dash: "dash" } },
                { type: "line", x0: 0.05, x1: 0.05, y0: 0, y1: 1, yref: "paper", line: { color: C.cyan, dash: "dash" } },
                { type: "line", x0: -0.05, x1: -0.05, y0: 0, y1: 1, yref: "paper", line: { color: C.cyan, dash: "dash" } },
              ],
            }}
            config={{ responsive: true }} style={{ width: "100%" }}
          />
        )}
      </div>

      {/* IC time series for selected factor */}
      {fi.length > 0 && (
        <div>
          <h4 className="text-[16px] font-semibold text-[#66605C] mb-2">IC Time Series</h4>
          <select className="bg-[#F2E5D7] text-[#33302E] text-[16px] rounded px-3 py-1.5 border border-[#E6D9CE] mb-3"
            value={selFactor} onChange={(e) => setSelFactor(Number(e.target.value))}>
            {fi.map((f: any, i: number) => (
              <option key={f.factor} value={i}>{f.factor} (IR={f.ic_ir?.toFixed(3)})</option>
            ))}
          </select>
          {fi[selFactor]?.ic_timeseries && (
            <Plot
              data={[{
                type: "bar",
                x: fi[selFactor].ic_timeseries.map((_: any, i: number) => `T${i + 1}`),
                y: fi[selFactor].ic_timeseries,
                marker: { color: fi[selFactor].ic_timeseries.map((v: number) => v > 0 ? C.green : C.red) },
              }]}
              layout={{
                ...DARK_LAYOUT, height: 250, margin: { t: 10, b: 40, l: 50, r: 20 },
                yaxis: { title: "IC" },
                shapes: [{ type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray } }],
              }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
          )}
        </div>
      )}

      {/* Multi-horizon */}
      {Object.keys(mh).length > 0 && (
        <div>
          <h4 className="text-[16px] font-semibold text-[#66605C] mb-2">Multi-Horizon IC Comparison</h4>
          {(() => {
            const horizonColors: Record<string, string> = { "1W": C.yellow, "1M": C.cyan, "3M": C.blue };
            const traces = Object.entries(mh).map(([period, data]: [string, any]) => ({
              type: "bar" as const,
              name: period,
              x: (data.top5 ?? []).map((d: any) => d.factor),
              y: (data.top5 ?? []).map((d: any) => d.ic_ir),
              marker: { color: horizonColors[period] ?? C.gray },
              text: (data.top5 ?? []).map((d: any) => d.ic_ir?.toFixed(2)),
              textposition: "outside" as const,
            }));
            return (
              <Plot data={traces}
                layout={{ ...DARK_LAYOUT, barmode: "group", height: 350, margin: { t: 10, b: 80, l: 50, r: 20 }, xaxis: { tickangle: -45 } }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
            );
          })()}
        </div>
      )}

      <DataTable data={fi} columns={[
        { accessorKey: "factor", header: "Factor" },
        { accessorKey: "group", header: "Group" },
        { accessorKey: "ic_mean", header: "IC Mean", cell: (p: any) => p.getValue()?.toFixed(4) },
        { accessorKey: "ic_ir", header: "IC IR", cell: (p: any) => p.getValue()?.toFixed(3) },
        { accessorKey: "ic_hit_rate", header: "Hit%", cell: (p: any) => `${p.getValue()?.toFixed(1)}%` },
        { accessorKey: "quality", header: "Quality" },
        { accessorKey: "direction", header: "Direction" },
      ]} maxHeight="400px" />
    </div>
  );
}


/* ================================================================== */
/*  Section 3: Long-Short Portfolio                                    */
/* ================================================================== */
function LongShortSection({ ls }: { ls: any }) {
  const fr: any[] = ls.factor_returns ?? [];
  const [selIdx, setSelIdx] = useState(0);
  const profitable = fr.filter((f: any) => f.ann_return > 0);
  const best = fr[0];

  return (
    <div className="space-y-8">
      <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide">
        Long-Short Factor Portfolio
      </h3>
      <p className="text-[13px] text-[#857F7A]">
        팩터값 Top/Bottom 분위 → L-S 수익률. Sharpe &ge; 0.5 + 단조적이면 강력한 팩터.
      </p>

      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Profitable Factors" value={`${profitable.length}/${fr.length}`} />
        <MetricCard label="Best Factor" value={best?.factor ?? "?"} sub={`SR=${best?.ann_sharpe?.toFixed(2)}`} />
        <MetricCard label="Best Ann. Return" value={`${best?.ann_return?.toFixed(1)}%`} />
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Sharpe bar */}
        {fr.length > 0 && (
          <Plot
            data={[{
              type: "bar",
              x: fr.slice(0, 15).map((f: any) => f.factor),
              y: fr.slice(0, 15).map((f: any) => f.ann_sharpe),
              marker: { color: fr.slice(0, 15).map((f: any) => f.ann_sharpe > 0 ? C.green : C.red) },
            }]}
            layout={{
              ...DARK_LAYOUT, height: 350, margin: { t: 10, b: 80, l: 50, r: 20 },
              xaxis: { tickangle: -45 }, yaxis: { title: "Ann. Sharpe Ratio" },
              shapes: [
                { type: "line", y0: 0.5, y1: 0.5, x0: 0, x1: 1, xref: "paper", line: { color: C.yellow, dash: "dash" } },
                { type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray } },
              ],
            }}
            config={{ responsive: true }} style={{ width: "100%" }}
          />
        )}

        {/* Monotonicity vs Sharpe */}
        {fr.length > 0 && (
          <Plot
            data={[{
              type: "scatter", mode: "markers+text",
              x: fr.map((f: any) => f.monotonicity),
              y: fr.map((f: any) => f.ann_sharpe),
              text: fr.map((f: any) => f.factor),
              textposition: "top center",
              textfont: { size: 8, color: C.text },
              marker: { size: 10, color: C.cyan },
            }]}
            layout={{
              ...DARK_LAYOUT, height: 350, margin: { t: 10, b: 40, l: 50, r: 20 },
              xaxis: { title: "Monotonicity (1.0=perfect)" }, yaxis: { title: "Ann. Sharpe" },
              shapes: [
                { type: "line", y0: 0.5, y1: 0.5, x0: 0, x1: 1, xref: "paper", line: { color: C.yellow, dash: "dash" } },
                { type: "line", x0: 0.75, x1: 0.75, y0: 0, y1: 1, yref: "paper", line: { color: C.yellow, dash: "dash" } },
              ],
            }}
            config={{ responsive: true }} style={{ width: "100%" }}
          />
        )}
      </div>

      {/* Quintile returns for selected factor */}
      {fr.length > 0 && (
        <div>
          <h4 className="text-[16px] font-semibold text-[#66605C] mb-2">Quintile Return Profile</h4>
          <select className="bg-[#F2E5D7] text-[#33302E] text-[16px] rounded px-3 py-1.5 border border-[#E6D9CE] mb-3"
            value={selIdx} onChange={(e) => setSelIdx(Number(e.target.value))}>
            {fr.map((f: any, i: number) => (
              <option key={f.factor} value={i}>{f.factor} (SR={f.ann_sharpe?.toFixed(2)}, Mono={f.monotonicity?.toFixed(2)})</option>
            ))}
          </select>
          {fr[selIdx]?.quintile_returns && (
            <Plot
              data={[{
                type: "bar",
                x: ["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"],
                y: fr[selIdx].quintile_returns,
                marker: { color: [C.red, C.orange, C.yellow, C.blue, C.green] },
                text: fr[selIdx].quintile_returns.map((v: number) => `${v.toFixed(2)}%`),
                textposition: "outside",
              }]}
              layout={{
                ...DARK_LAYOUT, height: 300, margin: { t: 10, b: 50, l: 50, r: 20 },
                yaxis: { title: "Avg Return %" },
                shapes: [{ type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray } }],
              }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
          )}
        </div>
      )}

      <DataTable data={fr} columns={[
        { accessorKey: "factor", header: "Factor" },
        { accessorKey: "group", header: "Group" },
        { accessorKey: "ann_return", header: "Ann Ret%", cell: (p: any) => `${p.getValue()?.toFixed(2)}%` },
        { accessorKey: "ann_sharpe", header: "Ann SR", cell: (p: any) => p.getValue()?.toFixed(3) },
        { accessorKey: "monotonicity", header: "Mono", cell: (p: any) => p.getValue()?.toFixed(2) },
        { accessorKey: "mono_direction", header: "Direction" },
        { accessorKey: "win_rate", header: "Win%", cell: (p: any) => `${p.getValue()?.toFixed(1)}%` },
      ]} maxHeight="400px" />
    </div>
  );
}


/* ================================================================== */
/*  Section 4: PCA Model                                               */
/* ================================================================== */
function PCASection({ pca }: { pca: any }) {
  const comps: any[] = pca.components ?? [];
  const pcTickers: any[] = pca.pc_top_tickers ?? [];

  return (
    <div className="space-y-8">
      <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide">
        PCA Statistical Factor Model
      </h3>
      <p className="text-[13px] text-[#857F7A]">
        수익률 공분산행렬 eigendecomposition → 잠재 팩터 추출 → 명명된 팩터와 상관 매핑.
      </p>

      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Effective Dimensions" value={pca.effective_dimensionality ?? "?"} sub="Kaiser criterion (eigenval > 1)" />
        <MetricCard label="Top-3 Var Explained" value={`${pca.total_var_explained_top3 ?? 0}%`} />
        <MetricCard label="Universe" value={`${pca.n_tickers ?? 0} tickers x ${pca.n_days ?? 0}d`} />
      </div>

      {/* Scree plot */}
      {comps.length > 0 && (
        <Plot
          data={[
            {
              type: "bar", name: "Var %",
              x: comps.map((c: any) => `PC${c.pc}`),
              y: comps.map((c: any) => c.var_explained),
              marker: { color: C.cyan }, yaxis: "y",
            },
            {
              type: "scatter", mode: "lines+markers", name: "Cumulative %",
              x: comps.map((c: any) => `PC${c.pc}`),
              y: comps.map((c: any) => c.cum_var_explained),
              line: { color: C.yellow, width: 2 }, yaxis: "y2",
            },
          ]}
          layout={{
            ...DARK_LAYOUT, height: 380, margin: { t: 10, b: 50, l: 50, r: 60 },
            yaxis: { title: "Var Explained %", side: "left" },
            yaxis2: { title: "Cumulative %", side: "right", overlaying: "y" },
            legend: { x: 0.7, y: 0.95, bgcolor: "rgba(0,0,0,0)" },
          }}
          config={{ responsive: true }} style={{ width: "100%" }}
        />
      )}

      {/* PC → Factor mapping table */}
      {comps.length > 0 && (
        <>
          <h4 className="text-[16px] font-semibold text-[#66605C]">PC → Factor Mapping</h4>
          <DataTable data={comps.map((c: any) => ({
            pc: `PC${c.pc}`,
            eigenvalue: c.eigenvalue?.toFixed(3),
            var_pct: `${c.var_explained}%`,
            cum_pct: `${c.cum_var_explained}%`,
            interpretation: c.interpretation,
            correlations: (c.top_factor_correlations ?? []).slice(0, 3)
              .map((fc: any) => `${fc.factor}(${fc.correlation?.toFixed(2)})`).join(", "),
          }))} columns={[
            { accessorKey: "pc", header: "PC", size: 60 },
            { accessorKey: "eigenvalue", header: "Eigenval" },
            { accessorKey: "var_pct", header: "Var %" },
            { accessorKey: "cum_pct", header: "Cum %" },
            { accessorKey: "interpretation", header: "Interpretation" },
            { accessorKey: "correlations", header: "Top Factor Correlations" },
          ]} maxHeight="350px" />
        </>
      )}

      {/* Heatmap */}
      {comps.length > 0 && (() => {
        const factors = new Set<string>();
        const pcLabels: string[] = [];
        comps.slice(0, 5).forEach((c: any) => {
          pcLabels.push(`PC${c.pc}`);
          (c.top_factor_correlations ?? []).forEach((fc: any) => factors.add(fc.factor));
        });
        const fList = Array.from(factors);
        const zData = fList.map((f) =>
          comps.slice(0, 5).map((c: any) => {
            const match = (c.top_factor_correlations ?? []).find((fc: any) => fc.factor === f);
            return match ? match.correlation : 0;
          })
        );
        return fList.length > 0 ? (
          <Plot
            data={[{
              type: "heatmap", z: zData, x: pcLabels, y: fList,
              colorscale: "RdBu", zmid: 0,
              text: zData.map((row) => row.map((v: number) => v.toFixed(2))),
              texttemplate: "%{text}", hoverinfo: "z",
            }]}
            layout={{ ...DARK_LAYOUT, height: Math.max(300, fList.length * 24), margin: { t: 10, b: 40, l: 150, r: 20 } }}
            config={{ responsive: true }} style={{ width: "100%" }}
          />
        ) : null;
      })()}

      {/* Top tickers per PC */}
      {pcTickers.slice(0, 3).map((pct: any) => (
        <details key={pct.pc} className="bg-[#FFFFFF] rounded border border-[#E6D9CE] p-3">
          <summary className="text-[16px] text-[#66605C] cursor-pointer">PC{pct.pc} — Top 10 Tickers</summary>
          <div className="mt-2">
            <DataTable data={pct.top_tickers ?? []} columns={[
              { accessorKey: "ticker", header: "Ticker" },
              { accessorKey: "loading", header: "Loading", cell: (p: any) => p.getValue()?.toFixed(4) },
            ]} maxHeight="200px" />
          </div>
        </details>
      ))}
    </div>
  );
}


/* ================================================================== */
/*  Section 5: Regime-Conditional                                      */
/* ================================================================== */
function RegimeSection({ rc }: { rc: any }) {
  const dist = rc.regime_distribution ?? {};
  const current = rc.current_regime ?? "?";
  const rankings = rc.regime_factor_rankings ?? {};
  const stability: any[] = rc.cross_regime_stability ?? [];
  const recs: string[] = rc.recommendations ?? [];
  const regimeColors: Record<string, string> = { BULL: C.green, BEAR: C.red, TRANSITION: C.yellow };

  return (
    <div className="space-y-8">
      <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide">
        Regime-Conditional Factor Premium
      </h3>
      <p className="text-[13px] text-[#857F7A]">
        SPY 가격구조로 BULL/BEAR/TRANSITION 레짐 분류 → 레짐별 팩터 IC 비교 → 현재 환경 최적 팩터 판단.
      </p>

      <div className="grid grid-cols-4 gap-3">
        <MetricCard label="Current Regime" value={current} />
        <MetricCard label="BULL periods" value={dist.BULL ?? 0} />
        <MetricCard label="BEAR periods" value={dist.BEAR ?? 0} />
        <MetricCard label="TRANSITION periods" value={dist.TRANSITION ?? 0} />
      </div>

      {recs.length > 0 && (
        <div className="bg-emerald-900/30 border border-emerald-700 rounded p-3 space-y-1">
          {recs.map((r, i) => <p key={i} className="text-[16px] text-emerald-300">{r}</p>)}
        </div>
      )}

      {/* Regime-specific rankings side-by-side */}
      {Object.keys(rankings).length > 0 && (
        <div className={`grid gap-4`} style={{ gridTemplateColumns: `repeat(${Object.keys(rankings).length}, 1fr)` }}>
          {Object.entries(rankings).map(([regime, factors]: [string, any]) => (
            <div key={regime}>
              <h4 className="text-[16px] font-bold mb-2" style={{ color: regimeColors[regime] ?? C.gray }}>{regime}</h4>
              <DataTable data={(factors as any[]).slice(0, 10)} columns={[
                { accessorKey: "factor", header: "Factor" },
                { accessorKey: "ic_mean", header: "IC", cell: (p: any) => p.getValue()?.toFixed(4) },
                { accessorKey: "ic_ir", header: "IR", cell: (p: any) => p.getValue()?.toFixed(3) },
              ]} maxHeight="300px" />
            </div>
          ))}
        </div>
      )}

      {/* Cross-regime heatmap */}
      {Object.keys(rankings).length > 0 && (() => {
        const regimes = Object.keys(rankings);
        const allFactors = new Set<string>();
        regimes.forEach((r) => (rankings[r] as any[]).slice(0, 15).forEach((f: any) => allFactors.add(f.factor)));
        const fList = Array.from(allFactors);
        const zData = fList.map((f) =>
          regimes.map((r) => {
            const match = (rankings[r] as any[]).find((x: any) => x.factor === f);
            return match ? match.ic_mean : 0;
          })
        );
        return fList.length > 0 ? (
          <>
            <h4 className="text-[16px] font-semibold text-[#66605C]">Cross-Regime Factor Heatmap</h4>
            <Plot
              data={[{
                type: "heatmap", z: zData, x: regimes, y: fList,
                colorscale: "RdBu", zmid: 0,
                text: zData.map((row) => row.map((v: number) => v.toFixed(3))),
                texttemplate: "%{text}", hoverinfo: "z",
              }]}
              layout={{ ...DARK_LAYOUT, height: Math.max(350, fList.length * 22), margin: { t: 10, b: 40, l: 150, r: 20 } }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
          </>
        ) : null;
      })()}

      {/* All-weather stability */}
      {stability.length > 0 && (
        <>
          <h4 className="text-[16px] font-semibold text-[#66605C]">All-Weather Factor Stability</h4>
          <p className="text-[13px] text-[#857F7A] mb-2">모든 레짐에서 같은 방향 IC → 레짐 불변 팩터</p>
          <DataTable data={stability} columns={[
            { accessorKey: "factor", header: "Factor" },
            { accessorKey: "group", header: "Group" },
            { accessorKey: "mean_ic", header: "Mean IC", cell: (p: any) => p.getValue()?.toFixed(4) },
            { accessorKey: "stability", header: "Stability", cell: (p: any) => p.getValue()?.toFixed(3) },
            { accessorKey: "all_weather", header: "All-Weather", cell: (p: any) => p.getValue() ? "Yes" : "No" },
          ]} maxHeight="300px" />
          {(() => {
            const aw = stability.filter((f: any) => f.all_weather).map((f: any) => f.factor);
            return aw.length > 0 ? (
              <div className="bg-[#E3EEF5]/30 border border-[#9CC3D5] rounded p-3">
                <span className="text-[16px] text-[#0F5499]">All-Weather Factors: <strong>{aw.join(", ")}</strong></span>
              </div>
            ) : null;
          })()}
        </>
      )}
    </div>
  );
}
