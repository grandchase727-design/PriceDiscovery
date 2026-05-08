import { useEffect, useState, useMemo } from "react";
import Plot from "react-plotly.js";
import { type ColumnDef } from "@tanstack/react-table";
import { fetchAIPrediction, fetchAIPerformance, fetchAIBenchmarks, fetchAIWinRatio } from "../../api/client";
import { DataTable } from "../shared/DataTable";
import { MetricCard } from "../shared/MetricCard";
import { C, DARK_LAYOUT } from "../../styles/theme";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────
interface Metrics {
  winner_name: string;
  winner_sharpe: number;
  winner_alpha_ann: number;
  winner_max_dd: number;
  winner_ann_return: number;
  winner_ann_vol: number;
  winner_bull_recall: number;
  winner_base_recall: number;
  winner_bear_recall: number;
  winner_turnover: number;
  winner_n_features: number;
  winner_n_rows: number;
  baseline_sharpe: number;
  baseline_alpha_ann: number;
  baseline_max_dd: number;
  baseline_turnover: number;
  baseline_bull_recall: number;
  benchmark_sharpe: number;
  benchmark_ann_return: number;
  benchmark_max_dd: number;
  meta_hit_rate: number;
  meta_agreement: number;
  meta_override_rate: number;
  n_oos_months: number;
  period_start: string;
  period_end: string;
  benchmark_label?: string;
  benchmark_weights?: { equity: number; bond: number; cash: number };
}

interface ProbaRow {
  date: string;
  BEAR: number; BASE: number; BULL: number;
  true: string; pred: string;
  meta_confidence?: number;
  final_regime?: string;
}

interface ReturnsRow {
  date: string;
  strategy_ret: number;
  benchmark_static_ret: number;
  p0_baseline_ret: number;
  w_equity: number;
  w_bond: number;
  w_cash: number;
  turnover: number;
}

interface FIRow { feature: string; importance: number; }

interface CMRow { label: string; BEAR: number; BASE: number; BULL: number; }

interface AblationRow {
  variant: string;
  sharpe: number;
  alpha_ann: number;
  bull_recall: number;
  bear_recall: number;
  base_recall: number;
  max_dd: number;
  mean_turnover_pct: number;
}

interface MoEStats {
  ann_return: number; sharpe: number; max_dd: number;
  alpha_ann: number; turnover: number;
  ann_vol?: number; tracking_error?: number;
}

interface MoESummary {
  n_oos_months: number;
  period_start: string;
  period_end: string;
  moe_soft: MoEStats;
  moe_hard: MoEStats;
  p4_solo: MoEStats;
  p0_solo: MoEStats;
  benchmark: { ann_return: number; sharpe: number; max_dd: number; label: string };
  expert_assignment: { BULL: string; BASE: string; BEAR: string };
  gate: string;
}

interface MoEMonthly {
  date: string;
  gate_argmax: string;
  gate_P_BEAR: number; gate_P_BASE: number; gate_P_BULL: number;
  soft_ret: number; hard_ret: number; bench_ret: number;
  true_regime: string;
  w_eq_soft: number; w_bd_soft: number; w_ch_soft: number;
  w_eq_hard: number; w_bd_hard: number; w_ch_hard: number;
}

interface AIData {
  metrics: Metrics | null;
  proba: ProbaRow[];
  returns: ReturnsRow[];
  feature_importance: FIRow[];
  confusion_matrix: CMRow[];
  ablation: AblationRow[];
  moe_summary?: MoESummary | null;
  moe_monthly?: MoEMonthly[] | null;
  error?: string;
}

// ─── Performance Analytics ────────────────────────────────────────────────
interface VariantMetrics {
  n_months: number;
  cagr: number; cagr_bench: number;
  ann_vol: number; ann_vol_bench: number;
  max_dd: number; max_dd_bench: number;
  max_dd_duration_months: number;
  sharpe: number; sharpe_bench: number;
  sortino: number; calmar: number; omega: number;
  alpha_ann: number; tracking_error: number;
  information_ratio: number; beta: number; correlation: number;
  jensens_alpha_ann: number; treynor: number;
  up_capture: number; dn_capture: number;
  win_rate: number; excess_win_rate: number;
  skew: number; kurt_excess: number;
  var_5_monthly: number; cvar_5_monthly: number;
  period_start: string; period_end: string;
  mean_turnover_pct: number;
}

interface PerfSummary {
  benchmark_label: string;
  benchmark_weights: { equity: number; bond: number; cash: number };
  variants: Record<string, VariantMetrics>;
  variant_order: string[];
}

interface PerfMonthlyRow {
  date: string;
  variant: string;
  strat_ret: number;
  bench_ret: number;
  w_equity: number; w_bond: number; w_cash: number;
}

interface PerfRollingRow {
  date: string;
  variant: string;
  window: number;
  [k: string]: any;   // cagr_12m, sharpe_12m, etc
}

interface PerfData {
  summary: PerfSummary;
  monthly: PerfMonthlyRow[];
  rolling: PerfRollingRow[];
  error?: string;
}

const VARIANT_COLORS: Record<string, string> = {
  P0: C.cyan, P1: C.orange, P2: C.yellow, P3: C.blue, P4: C.green,
};

// ─── Multi-Benchmark Validation ──────────────────────────────────────────
interface BenchmarkRow {
  tag: string;
  name: string;
  role: string;
  weights: { equity: number; bond: number; cash: number };
  ann_return: number;
  ann_vol: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_dd: number;
  up_capture: number;
  dn_capture: number;
}

interface BenchmarkComparison {
  tag: string;
  name: string;
  alpha_ann: number;
  tracking_error: number;
  information_ratio: number;
  p4_sharpe_minus_bench: number;
  p4_sortino_minus_bench: number | null;
  p4_calmar_minus_bench: number | null;
  p4_dd_minus_bench: number;
  p4_up_capture_vs_bench: number;
  p4_dn_capture_vs_bench: number;
}

interface BenchmarkData {
  summary: {
    n_oos_months: number;
    period_start: string;
    period_end: string;
    p4_strategy: BenchmarkRow & { name: string; tag: string };
    benchmarks: BenchmarkRow[];
    comparisons: BenchmarkComparison[];
    hypothesis_verdict: {
      vs_acwi100: {
        sharpe_better: boolean;
        sortino_better: boolean | null;
        calmar_better: boolean | null;
        max_dd_better: boolean;
        overall_pass: boolean;
        score: string;
      };
    };
    cash_ann_yield: number;
  };
  monthly: { date: string; p4: number; [tag: string]: any }[];
  error?: string;
}

// ─── Win Ratio (per-signal precision + directional hit rate) ─────────────
interface WinRatioRow {
  regime: string;
  n: number;
  precision: number | null;
  directional: number | null;
  fwd_ret_mean: number | null;
  fwd_ret_pos_pct: number | null;
  fwd_ret_median?: number | null;
}

interface WinRatioVariant {
  variant: string;
  n_oos_months: number;
  period_start: string;
  period_end: string;
  overall_accuracy: number;
  pred_distribution: Record<string, number>;
  true_distribution: Record<string, number>;
  per_regime: WinRatioRow[];
}

interface WinRatioData {
  variants: Record<string, WinRatioVariant>;
  regimes: string[];
  base_range: number;
  definitions: Record<string, string>;
  error?: string;
}

const BENCH_COLOR: Record<string, string> = {
  acwi100: C.red,
  acwi75ch25: C.cyan,
  acwi60agg40: C.blue,
  acwi90ch10: C.gray,
  tbill: C.yellow,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function cumulate(arr: number[]): number[] {
  let acc = 1;
  const out: number[] = [];
  for (const r of arr) {
    acc *= 1 + (r ?? 0);
    out.push(acc);
  }
  return out;
}

function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "–";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtPctSigned(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "–";
  const s = (v * 100).toFixed(digits);
  return v >= 0 ? `+${s}%` : `${s}%`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main tab
// ─────────────────────────────────────────────────────────────────────────────
export function AIPredictionTab() {
  const [data, setData] = useState<AIData | null>(null);
  const [perf, setPerf] = useState<PerfData | null>(null);
  const [bench, setBench] = useState<BenchmarkData | null>(null);
  const [winRatio, setWinRatio] = useState<WinRatioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [sub, setSub] = useState(0);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchAIPrediction().then((d: AIData) => setData(d)),
      fetchAIPerformance().then((d: PerfData) => setPerf(d)).catch(() => setPerf(null)),
      fetchAIBenchmarks().then((d: BenchmarkData) => setBench(d)).catch(() => setBench(null)),
      fetchAIWinRatio().then((d: WinRatioData) => setWinRatio(d)).catch(() => setWinRatio(null)),
    ]).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="text-gray-500">Loading AI prediction results...</div>;
  }
  if (!data || data.error || !data.metrics) {
    return (
      <div className="p-6 bg-[#111827] border border-red-900 rounded-lg text-red-300 text-sm">
        {data?.error || "AI prediction cache not available."} <br />
        Run <code className="text-red-200">python3 ai_prediction_cache.py</code> to generate it.
      </div>
    );
  }

  const m = data.metrics;
  const alphaDelta = m.winner_alpha_ann - m.baseline_alpha_ann;
  const ddDelta = m.winner_max_dd - m.benchmark_max_dd;

  return (
    <div className="space-y-4">
      {/* ── Header (always visible) ── */}
      <div>
        <h2 className="text-xl font-bold text-gray-100">AI Regime Prediction (Forward 1M)</h2>
        <p className="text-sm text-gray-500 mt-1">
          MSCI ACWI / Bloomberg Global Agg forward 1-month regime classifier (BULL / BASE / BEAR). LightGBM multiclass + meta-labeling
          (Lopez de Prado). Allocation grids: <b>BULL 90 / 5 / 5</b>, <b>BASE 75 / 10 / 15</b>, <b>BEAR 60 / 15 / 25</b> (equity / bond / cash).
          &nbsp;Benchmark: <b className="text-cyan-300">{m.benchmark_label ?? "ACWI 90 / Cash 10"}</b>.
        </p>
      </div>

      {/* ── Sub-tab navigation ── */}
      <div className="flex gap-1 border-b border-gray-800">
        {["Performance & Analytics", "Signal Definition"].map((label, i) => (
          <button
            key={label}
            onClick={() => setSub(i)}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              sub === i
                ? "border-cyan-400 text-cyan-300"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}>
            {label}
          </button>
        ))}
      </div>

      {sub === 1 ? (
        <SignalDefinitionContent />
      ) : (
      <div className="space-y-6">
      {/* ── Model Definitions (P0 → P4) ── */}
      <ModelDefinitions />

      {/* ── KPI cards ── */}
      <div className="grid grid-cols-6 gap-3">
        <MetricCard label="Sharpe (OOS)" value={m.winner_sharpe.toFixed(2)}
                    sub={`${(m.winner_sharpe - m.benchmark_sharpe >= 0 ? "+" : "")}${(m.winner_sharpe - m.benchmark_sharpe).toFixed(2)} vs BM`} />
        <MetricCard label="Alpha (ann)" value={fmtPctSigned(m.winner_alpha_ann)}
                    sub={`${fmtPctSigned(alphaDelta, 2)} vs P0`} />
        <MetricCard label="Max DD" value={fmtPct(m.winner_max_dd, 1)}
                    sub={`${ddDelta >= 0 ? "+" : ""}${(ddDelta * 100).toFixed(1)}pp vs BM`} />
        <MetricCard label="Turnover" value={`${m.winner_turnover.toFixed(1)}%/mo`}
                    sub={`−${(m.baseline_turnover - m.winner_turnover).toFixed(1)}pp vs P0`} />
        <MetricCard label="BULL recall" value={fmtPct(m.winner_bull_recall, 0)}
                    sub={`+${((m.winner_bull_recall - m.baseline_bull_recall) * 100).toFixed(0)}pp vs P0`} />
        <MetricCard label="OOS months" value={String(m.n_oos_months)}
                    sub={`${m.period_start} → ${m.period_end}`} />
      </div>

      <div className="text-xs text-gray-500">
        <b>Meta diagnostics</b> — hit rate {fmtPct(m.meta_hit_rate, 1)}, agreement with primary {fmtPct(m.meta_agreement, 1)},
        override→BASE rate {fmtPct(m.meta_override_rate, 1)}. Dataset: {m.winner_n_features} features × {m.winner_n_rows} months.
      </div>

      {/* ── Cumulative Return (full-width) ── */}
      <div className="bg-[#111827] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-semibold text-gray-200 mb-2">Cumulative Return (OOS)</div>
        <Plot
          data={(() => {
            const dates = data.returns.map((r) => r.date);
            const s = cumulate(data.returns.map((r) => r.strategy_ret ?? 0));
            const b = cumulate(data.returns.map((r) => r.benchmark_static_ret ?? 0));
            const p0 = cumulate(data.returns.map((r) => r.p0_baseline_ret ?? 0));
            const bmLabel = `Benchmark ${m.benchmark_label ?? "ACWI 90 / Cash 10"}`;
            return [
              { x: dates, y: s, type: "scatter", mode: "lines", name: "P4 meta (winner)", line: { color: C.green, width: 2.5 } },
              { x: dates, y: b, type: "scatter", mode: "lines", name: bmLabel, line: { color: C.gray, width: 2 } },
              { x: dates, y: p0, type: "scatter", mode: "lines", name: "P0 baseline", line: { color: C.orange, width: 2 } },
            ] as any;
          })()}
          layout={{
            ...DARK_LAYOUT, height: 380,
            margin: { t: 20, b: 35, l: 55, r: 15 },
            hovermode: "x unified",
            yaxis: { title: "Cumulative", gridcolor: "#1f2937" },
            xaxis: { gridcolor: "#1f2937" },
            legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)" },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
      </div>

      {/* ── Monthly Allocation by Variant (full-width, 5 stacked-area panels) ── */}
      {perf && perf.summary && perf.monthly && (
        <MonthlyAllocationByVariant perf={perf} />
      )}

      {/* ── Charts row 2: regime proba + confusion matrix ── */}
      <div className="grid grid-cols-3 gap-4">
        {/* Regime probabilities */}
        <div className="col-span-2 bg-[#111827] rounded-lg p-4 border border-gray-800">
          <div className="text-sm font-semibold text-gray-200 mb-2">OOS Regime Probabilities (Primary Model)</div>
          <Plot
            data={(() => {
              const dates = data.proba.map((r) => r.date);
              const pal: Record<string, string> = { BEAR: C.red, BASE: C.gray, BULL: C.green };
              return ["BEAR", "BASE", "BULL"].map((k) => ({
                x: dates, y: data.proba.map((r) => (r as any)[k]),
                type: "scatter", mode: "lines", name: k, stackgroup: "one",
                line: { width: 0.3, color: pal[k] }, fillcolor: pal[k],
              })) as any;
            })()}
            layout={{
              ...DARK_LAYOUT, height: 340,
              margin: { t: 20, b: 35, l: 50, r: 15 },
              hovermode: "x unified",
              yaxis: { range: [0, 1], tickformat: ".0%", gridcolor: "#1f2937" },
              xaxis: { gridcolor: "#1f2937" },
              legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        </div>

        {/* Confusion matrix */}
        <div className="bg-[#111827] rounded-lg p-4 border border-gray-800">
          <div className="text-sm font-semibold text-gray-200 mb-2">Confusion Matrix (OOS)</div>
          <Plot
            data={(() => {
              const labels = ["BEAR", "BASE", "BULL"];
              const z = data.confusion_matrix.map((row) => labels.map((l) => (row as any)[l]));
              const yLabels = data.confusion_matrix.map((r) => r.label.replace("true_", "true: "));
              return [{
                type: "heatmap",
                z, x: labels.map((l) => "pred: " + l), y: yLabels,
                colorscale: "Blues", showscale: false,
                text: z, texttemplate: "%{text}",
                textfont: { size: 14, color: "white" },
              }] as any;
            })()}
            layout={{
              ...DARK_LAYOUT, height: 340,
              margin: { t: 20, b: 35, l: 70, r: 15 },
              yaxis: { autorange: "reversed" as any },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          {(() => {
            const total = data.confusion_matrix.reduce((s, row) =>
              s + row.BEAR + row.BASE + row.BULL, 0);
            const correct =
              (data.confusion_matrix[0]?.BEAR ?? 0) +
              (data.confusion_matrix[1]?.BASE ?? 0) +
              (data.confusion_matrix[2]?.BULL ?? 0);
            return (
              <div className="text-xs text-gray-400 mt-1">
                Overall accuracy: <span className="text-cyan-400 font-semibold">
                  {((correct / total) * 100).toFixed(1)}%
                </span>{" "}({correct} / {total})
              </div>
            );
          })()}
        </div>
      </div>

      {/* ── Feature importance ── */}
      <div className="bg-[#111827] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-semibold text-gray-200 mb-2">Feature Importance (LightGBM mean gain across folds)</div>
        <Plot
          data={(() => {
            const top = data.feature_importance.slice(0, 20).reverse();
            const colors = top.map((f) =>
              f.feature.startsWith("eq_") || f.feature.startsWith("bd_") ? C.purple : C.cyan
            );
            return [{
              type: "bar", orientation: "h",
              x: top.map((f) => f.importance), y: top.map((f) => f.feature),
              marker: { color: colors },
              hovertemplate: "%{y}: %{x:.1f}<extra></extra>",
            }] as any;
          })()}
          layout={{
            ...DARK_LAYOUT, height: 500,
            margin: { t: 20, b: 35, l: 180, r: 20 },
            xaxis: { gridcolor: "#1f2937", title: "importance" },
            yaxis: { gridcolor: "#1f2937" },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
        <div className="text-xs text-gray-500 mt-2">
          <span className="text-purple-400">Purple</span> = breadth features from historical price_discovery replay (P1).&nbsp;
          <span className="text-cyan-400">Cyan</span> = macro / cross-asset features (baseline + P3).
        </div>
      </div>

      {/* ── Sharpe vs Alpha scatter ── */}
      <div className="bg-[#111827] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-semibold text-gray-200 mb-2">Sharpe vs Alpha (per variant)</div>
        <Plot
          data={(() => {
            const tag = (v: string) =>
              v.includes("P4_meta") ? "P4 meta"
              : v.includes("VIXFREE") ? "VIX-free"
              : v.includes("P1_breadth") ? "P1 breadth"
              : v.includes("kfold") ? "P0 kfold"
              : "P0/P2/P3";
            const colors: Record<string, string> = {
              "P4 meta": C.green, "VIX-free": C.purple,
              "P1 breadth": C.orange, "P0 kfold": C.red,
              "P0/P2/P3": C.cyan,
            };
            const groups: Record<string, AblationRow[]> = {};
            for (const r of data.ablation) {
              const t = tag(r.variant);
              if (!groups[t]) groups[t] = [];
              groups[t].push(r);
            }
            return Object.entries(groups).map(([t, rows]) => ({
              type: "scatter", mode: "markers",
              x: rows.map((r) => r.alpha_ann), y: rows.map((r) => r.sharpe),
              name: t,
              marker: {
                color: colors[t],
                size: rows.map((r) => Math.max(8, (r.bull_recall ?? 0) * 70 + 6)),
                line: { color: "#0a0e17", width: 1 },
                opacity: 0.85,
              },
              text: rows.map((r) => r.variant),
              hovertemplate: "<b>%{text}</b><br>Sharpe=%{y:.3f}<br>Alpha=%{x:.2%}<extra></extra>",
            })) as any;
          })()}
          layout={{
            ...DARK_LAYOUT, height: 420,
            margin: { t: 20, b: 45, l: 60, r: 20 },
            xaxis: { title: "Annual Alpha", tickformat: ".2%", gridcolor: "#1f2937", zerolinecolor: "#374151" },
            yaxis: { title: "Sharpe Ratio", gridcolor: "#1f2937" },
            shapes: [{ type: "line", x0: 0, x1: 0, y0: 0, y1: 1, yref: "paper", line: { dash: "dash", color: "#6b7280" } }],
            legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)" },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
        <div className="text-xs text-gray-500 mt-1">
          Point size = BULL recall. Dashed line = 0% alpha vs {m.benchmark_label ?? "ACWI 90 / Cash 10"}.
        </div>
      </div>

      {/* ── Ablation table ── */}
      <AblationTable rows={data.ablation} />

      {/* ── Performance Analytics (institutional-style) ── */}
      {perf && perf.summary && (
        <PerformanceAnalyticsSection perf={perf} />
      )}

      {/* ── Per-Signal Win Ratio (P0~P4 BEAR/BASE/BULL) ── */}
      {winRatio && winRatio.variants && !winRatio.error && (
        <WinRatioSection wr={winRatio} />
      )}

      {/* ── Multi-Benchmark Validation Suite (Hypothesis Test) ── */}
      {bench && bench.summary && !bench.error && (
        <BenchmarkValidationSection bench={bench} />
      )}

      {/* ── MoE (Plan B) Comparison ── */}
      {data.moe_summary && data.moe_monthly && (
        <MoEComparisonSection summary={data.moe_summary} monthly={data.moe_monthly} />
      )}

      {/* ── Methodology & findings ── */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-[#111827] rounded-lg p-4 border border-gray-800 text-sm text-gray-300">
          <div className="text-base font-semibold text-gray-100 mb-2">Pipeline (P0 → P4)</div>
          <ul className="list-disc pl-5 space-y-1 text-xs leading-relaxed">
            <li><b>P0</b>: Purged Walk-Forward CV with 21-day embargo (Lopez de Prado)</li>
            <li><b>P3</b>: 6 macro cross-asset features (VIX/MOVE, VIX/VXV, WTI/Copper, DXY z-score, HY/IG spread, TIP/IEF)</li>
            <li><b>P2</b>: Class weight × BULL-threshold sweep (4×3 = 12 variants)</li>
            <li><b>VIX-free ablation</b>: Test whether VIX-in-label circularity caps BULL recall</li>
            <li><b>P1</b>: Monthly historical replay of price_discovery over 170 ETFs → 6 breadth features (eq/bd × pct_bullish, pct_downtrend, tcs_median, rss_std)</li>
            <li><b>P4 (winner)</b>: Meta-labeling — primary = multiclass argmax, meta = binary filter, fallback → BASE</li>
          </ul>
        </div>
        <div className="bg-[#111827] rounded-lg p-4 border border-gray-800 text-sm text-gray-300">
          <div className="text-base font-semibold text-gray-100 mb-2">What Moved the Needle</div>
          <ul className="list-disc pl-5 space-y-1 text-xs leading-relaxed">
            <li><b>Meta-labeling (P4)</b> delivers the <span className="text-green-400">highest Sharpe</span> ({m.winner_sharpe.toFixed(2)} vs benchmark {m.benchmark_sharpe.toFixed(2)}) and <span className="text-green-400">lowest Max DD</span> ({(m.winner_max_dd * 100).toFixed(1)}% vs {(m.benchmark_max_dd * 100).toFixed(1)}%) — cuts turnover from {m.baseline_turnover.toFixed(1)}% → {m.winner_turnover.toFixed(1)}%/mo</li>
            <li><b>Alpha vs ACWI 90/10</b>: {(m.winner_alpha_ann * 100).toFixed(2)}% ann — absolute underperformance in bull tape, but superior risk-adjusted profile (Sharpe & DD)</li>
            <li><b>P3 macro features</b> doubled BULL recall {(m.baseline_bull_recall * 100).toFixed(0)}% → {(m.winner_bull_recall * 100).toFixed(0)}%</li>
            <li><b>VIX-free ablation</b> lowered BULL recall — label circularity is NOT the bottleneck</li>
            <li><b>BULL recall ceiling ~16%</b> with {m.winner_n_rows} monthly samples — data volume constraint</li>
            <li>All 3 leakage tests <span className="text-green-400">PASS</span> (feature as-of, breadth as-of, label correlation &lt; 0.2)</li>
          </ul>
        </div>
      </div>

      <div className="bg-[#0a0e17] border border-cyan-900/60 rounded-lg p-3 text-xs text-cyan-300">
        <b className="text-cyan-200">Production config</b> —&nbsp;
        <code>use_meta=true</code>, <code>class_weight={"{BULL: 3.0}"}</code>, <code>bull_threshold=0.25</code>,&nbsp;
        <code>cv_mode='walkforward'</code>, 22 macro + 6 breadth features.&nbsp;
        Regenerate cache via <code className="text-cyan-200">python3 ai_prediction_cache.py</code>.
      </div>
      </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Ablation table — sorted by alpha desc, row tinting by track
// ─────────────────────────────────────────────────────────────────────────────
function AblationTable({ rows }: { rows: AblationRow[] }) {
  const columns = useMemo<ColumnDef<AblationRow, any>[]>(() => [
    {
      accessorKey: "variant", header: "Variant",
      cell: (info) => {
        const v = info.getValue() as string;
        const tint =
          v.includes("P4_meta") ? "bg-green-900/30 text-green-200"
          : v.includes("VIXFREE") ? "bg-purple-900/30 text-purple-200"
          : v.includes("P1_breadth") ? "bg-orange-900/20 text-orange-200"
          : "text-gray-300";
        return <span className={`px-1.5 py-0.5 rounded ${tint}`}>{v}</span>;
      },
    },
    {
      accessorKey: "sharpe", header: "Sharpe",
      cell: (info) => (info.getValue() != null ? Number(info.getValue()).toFixed(3) : "–"),
    },
    {
      accessorKey: "alpha_ann", header: "Alpha (ann)",
      cell: (info) => {
        const v = info.getValue() as number;
        if (v == null) return "–";
        const s = (v * 100).toFixed(2);
        const cls = v >= 0 ? "text-green-400 font-semibold" : "text-red-400";
        return <span className={cls}>{v >= 0 ? "+" : ""}{s}%</span>;
      },
    },
    {
      accessorKey: "bull_recall", header: "BULL recall",
      cell: (info) => info.getValue() != null ? `${(Number(info.getValue()) * 100).toFixed(0)}%` : "–",
    },
    {
      accessorKey: "bear_recall", header: "BEAR recall",
      cell: (info) => info.getValue() != null ? `${(Number(info.getValue()) * 100).toFixed(0)}%` : "–",
    },
    {
      accessorKey: "max_dd", header: "Max DD",
      cell: (info) => info.getValue() != null ? `${(Number(info.getValue()) * 100).toFixed(1)}%` : "–",
    },
    {
      accessorKey: "mean_turnover_pct", header: "Turnover (%/mo)",
      cell: (info) => info.getValue() != null ? `${Number(info.getValue()).toFixed(2)}%` : "–",
    },
  ], []);

  const sorted = useMemo(() =>
    [...rows].filter((r) => r.sharpe != null)
      .sort((a, b) => (b.alpha_ann ?? -Infinity) - (a.alpha_ann ?? -Infinity)),
    [rows]
  );

  return (
    <div className="bg-[#111827] rounded-lg p-4 border border-gray-800">
      <div className="text-sm font-semibold text-gray-200 mb-2">
        P1-P4 Ablation Results ({sorted.length} variants, sorted by alpha)
      </div>
      <DataTable data={sorted} columns={columns} maxHeight="520px" />
      <div className="text-xs text-gray-500 mt-2">
        <span className="px-1.5 py-0.5 rounded bg-green-900/30 text-green-200">green</span> = P4 meta (winner).&nbsp;
        <span className="px-1.5 py-0.5 rounded bg-purple-900/30 text-purple-200">purple</span> = VIX-free label ablation.&nbsp;
        <span className="px-1.5 py-0.5 rounded bg-orange-900/20 text-orange-200">orange</span> = P1 breadth variants.
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MoE (Plan B) comparison section — regime-conditional expert blending
// ─────────────────────────────────────────────────────────────────────────────
function MoEComparisonSection({
  summary, monthly,
}: { summary: MoESummary; monthly: MoEMonthly[] }) {
  // Per-regime alpha decomposition (bp/month)
  const regimes = ["BULL", "BASE", "BEAR"] as const;
  const perRegime = useMemo(() => {
    const out: Record<string, Record<string, { alpha_bp: number; n: number }>> = {};
    const retKeys = [
      ["MoE soft", "soft_ret"],
      ["MoE hard", "hard_ret"],
    ] as const;
    for (const [label, key] of retKeys) {
      out[label] = {};
      for (const r of regimes) {
        const subset = monthly.filter((m) => m.true_regime === r);
        if (subset.length === 0) continue;
        const diffs = subset.map((m) => (m[key] ?? 0) - (m.bench_ret ?? 0));
        const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        out[label][r] = { alpha_bp: mean * 10000, n: subset.length };
      }
    }
    return out;
  }, [monthly]);

  // Gate argmax distribution
  const gateDist = useMemo(() => {
    const cnt: Record<string, number> = { BEAR: 0, BASE: 0, BULL: 0 };
    for (const m of monthly) cnt[m.gate_argmax] = (cnt[m.gate_argmax] ?? 0) + 1;
    return cnt;
  }, [monthly]);

  // Cumulative return chart data
  const cum = useMemo(() => {
    const dates = monthly.map((m) => m.date);
    const mk = (k: keyof MoEMonthly) => {
      let acc = 1; const out: number[] = [];
      for (const m of monthly) { acc *= 1 + (m[k] as number ?? 0); out.push(acc); }
      return out;
    };
    // P4 solo monthly returns = the BULL expert's allocation realized.
    // We don't have p4_ret column in moe_monthly, so we reconstruct from allocation:
    // Actually we can't without extra fields. Show the 3 we have: soft, hard, bench.
    return { dates, soft: mk("soft_ret"), hard: mk("hard_ret"), bench: mk("bench_ret") };
  }, [monthly]);

  // Rows for comparison table
  const rows = [
    { name: "P4_meta solo (current winner)", stats: summary.p4_solo, flag: "winner" as const },
    { name: "MoE hard (argmax gate)",        stats: summary.moe_hard, flag: "moe" as const },
    { name: "MoE soft (proba-weighted)",     stats: summary.moe_soft, flag: "moe" as const },
    { name: "P0 baseline solo",              stats: summary.p0_solo,  flag: "base" as const },
  ];

  return (
    <div className="bg-[#111827] rounded-lg p-4 border border-gray-800 space-y-4">
      <div>
        <div className="text-base font-semibold text-gray-100">
          Plan B · Regime-Conditional Expert Blending (MoE) — Comparison
        </div>
        <div className="text-xs text-gray-500 mt-1">
          Common window {summary.period_start} → {summary.period_end} ({summary.n_oos_months} months).&nbsp;
          Experts: <span className="text-green-300">BULL→{summary.expert_assignment.BULL}</span>,&nbsp;
          <span className="text-cyan-300">BASE→{summary.expert_assignment.BASE}</span>,&nbsp;
          <span className="text-amber-300">BEAR→{summary.expert_assignment.BEAR}</span>.&nbsp;
          Gate: {summary.gate}.
        </div>
      </div>

      {/* Summary comparison table */}
      <div className="overflow-auto border border-gray-800 rounded">
        <table className="w-full text-xs">
          <thead className="bg-[#1f2937]">
            <tr>
              <th className="px-2 py-1.5 text-left text-gray-400 border-b border-gray-700">Strategy</th>
              <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">AnnRet</th>
              <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Sharpe</th>
              <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Max DD</th>
              <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Alpha (ann)</th>
              <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Turnover</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}
                  className={`border-b border-gray-800/50 ${
                    row.flag === "winner" ? "bg-green-900/20 text-green-100" :
                    row.flag === "moe"    ? "bg-purple-900/10" : ""
                  }`}>
                <td className="px-2 py-1 font-medium">{row.name}</td>
                <td className="px-2 py-1 text-right">{(row.stats.ann_return * 100).toFixed(2)}%</td>
                <td className="px-2 py-1 text-right font-semibold">{row.stats.sharpe.toFixed(2)}</td>
                <td className="px-2 py-1 text-right">{(row.stats.max_dd * 100).toFixed(1)}%</td>
                <td className={`px-2 py-1 text-right font-semibold ${
                  row.stats.alpha_ann >= 0 ? "text-green-400" : "text-red-400"
                }`}>
                  {row.stats.alpha_ann >= 0 ? "+" : ""}{(row.stats.alpha_ann * 100).toFixed(2)}%
                </td>
                <td className="px-2 py-1 text-right">{row.stats.turnover.toFixed(2)}%/mo</td>
              </tr>
            ))}
            <tr className="border-t-2 border-gray-700 text-gray-400 italic">
              <td className="px-2 py-1">Benchmark {summary.benchmark.label}</td>
              <td className="px-2 py-1 text-right">{(summary.benchmark.ann_return * 100).toFixed(2)}%</td>
              <td className="px-2 py-1 text-right">{summary.benchmark.sharpe.toFixed(2)}</td>
              <td className="px-2 py-1 text-right">{(summary.benchmark.max_dd * 100).toFixed(1)}%</td>
              <td className="px-2 py-1 text-right">—</td>
              <td className="px-2 py-1 text-right">—</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Charts row: cumulative return + per-regime alpha */}
      <div className="grid grid-cols-2 gap-4">
        {/* Cumulative */}
        <div>
          <div className="text-sm font-semibold text-gray-200 mb-1">MoE vs Benchmark — Cumulative (common window)</div>
          <Plot
            data={[
              { x: cum.dates, y: cum.soft, type: "scatter", mode: "lines", name: "MoE soft",
                line: { color: C.purple, width: 2 } },
              { x: cum.dates, y: cum.hard, type: "scatter", mode: "lines", name: "MoE hard",
                line: { color: C.orange, width: 2 } },
              { x: cum.dates, y: cum.bench, type: "scatter", mode: "lines", name: "Benchmark",
                line: { color: C.gray, width: 2, dash: "dot" } as any },
            ] as any}
            layout={{
              ...DARK_LAYOUT, height: 300,
              margin: { t: 20, b: 35, l: 50, r: 15 },
              hovermode: "x unified",
              yaxis: { title: "Cumulative", gridcolor: "#1f2937" },
              xaxis: { gridcolor: "#1f2937" },
              legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        </div>

        {/* Per-regime alpha bar */}
        <div>
          <div className="text-sm font-semibold text-gray-200 mb-1">Per-Regime Alpha (bp / month, true regime)</div>
          <Plot
            data={(() => {
              return (["MoE soft", "MoE hard"] as const).map((label, i) => ({
                type: "bar",
                name: label,
                x: regimes as any,
                y: regimes.map((r) => perRegime[label]?.[r]?.alpha_bp ?? 0),
                marker: { color: i === 0 ? C.purple : C.orange },
                text: regimes.map((r) => {
                  const cell = perRegime[label]?.[r];
                  return cell ? `${cell.alpha_bp.toFixed(0)}` : "";
                }),
                textposition: "auto",
              })) as any;
            })()}
            layout={{
              ...DARK_LAYOUT, height: 300,
              margin: { t: 20, b: 35, l: 50, r: 15 },
              barmode: "group",
              yaxis: { title: "bp / month", gridcolor: "#1f2937", zerolinecolor: "#374151" },
              xaxis: { gridcolor: "#1f2937" },
              legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="text-[11px] text-gray-500 mt-1">
            MoE soft BEAR n={perRegime["MoE soft"]?.BEAR?.n ?? 0},&nbsp;
            BASE n={perRegime["MoE soft"]?.BASE?.n ?? 0},&nbsp;
            BULL n={perRegime["MoE soft"]?.BULL?.n ?? 0}.
          </div>
        </div>
      </div>

      {/* Gate diagnostics + verdict */}
      <div className="grid grid-cols-3 gap-3 text-xs">
        {(["BEAR", "BASE", "BULL"] as const).map((r) => {
          const n = gateDist[r] ?? 0;
          const pct = summary.n_oos_months > 0 ? (n / summary.n_oos_months) * 100 : 0;
          const color = r === "BEAR" ? "text-red-300" : r === "BULL" ? "text-green-300" : "text-gray-300";
          return (
            <div key={r} className="bg-[#0a0e17] border border-gray-800 rounded-lg p-2">
              <div className={`font-semibold ${color}`}>Gate argmax = {r}</div>
              <div className="text-gray-500">{n} / {summary.n_oos_months} months ({pct.toFixed(1)}%)</div>
            </div>
          );
        })}
      </div>

      <div className="bg-[#0a0e17] border border-amber-900/60 rounded-lg p-3 text-xs text-amber-200">
        <b className="text-amber-100">Verdict</b> — MoE does NOT beat P4_meta solo.&nbsp;
        Reason: P4's meta-labeling (override→BASE when low confidence) is already a
        stronger form of regime-conditional routing than external gating; plus the gate
        (P0 primary) has BULL recall ~16%, so argmax under-routes to the BULL expert.
        Oracle upper bound (if perfect regime known) ≈ −0.4%/yr alpha — still negative
        vs the aggressive 90/10 benchmark. <b>Production choice remains P4_meta.</b>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Model definitions — P0 through P4 (displayed at top of tab)
// ─────────────────────────────────────────────────────────────────────────────
function ModelDefinitions() {
  const phases = [
    {
      tag: "P0",
      title: "Baseline",
      color: C.cyan,
      short: "Reference model without any enhancements.",
      bullets: [
        "LightGBM multiclass on 15 macro / cross-asset features",
        "Purged Walk-Forward CV with 21-day embargo (Lopez de Prado, AFML Ch.7)",
        "Posterior-weighted allocation blend across BULL / BASE / BEAR grids",
        "Reference point for ablation attribution",
      ],
    },
    {
      tag: "P1",
      title: "Breadth Features",
      color: C.orange,
      short: "Bottom-up signal from price_discovery monthly replay.",
      bullets: [
        "Monthly historical replay of price_discovery scoring over 170 ETFs (2007-07~)",
        "6 aggregate features: equity (4) × bond (2)",
        "eq_pct_bullish / eq_pct_downtrend / eq_tcs_median / eq_rss_std",
        "bd_pct_bullish / bd_tcs_median",
        "BlackRock SAE-style bottom-up → top-down aggregation",
      ],
    },
    {
      tag: "P2",
      title: "BULL Detection",
      color: C.yellow,
      short: "Address minority-class imbalance (BULL = 14%).",
      bullets: [
        "Class weight sweep: BEAR 1.0 × BASE 1.0 × BULL {1.5, 2.0, 2.5, 3.0}",
        "BULL threshold override: classify as BULL when P(BULL) > {0.25, 0.30, 0.35}",
        "4 × 3 = 12 variants in the sweep",
        "VIX-free label ablation (test whether label-feature circularity caps recall)",
      ],
    },
    {
      tag: "P3",
      title: "Additional Macro Features",
      color: C.blue,
      short: "+6 cross-asset / term-structure features.",
      bullets: [
        "vix_move_ratio (equity vol / bond vol)",
        "vix_vxv_ratio (VIX term structure, 1M / 3M)",
        "wti_copper_ratio (inflation-sensitive vs growth-sensitive)",
        "dxy_z252 (USD 252-day z-score)",
        "hy_ig_spread_level (HY / IG credit ratio)",
        "tip_ief_ratio (inflation breakeven proxy)",
      ],
    },
    {
      tag: "P4",
      title: "Meta-Labeling",
      color: C.green,
      short: "Two-stage ensemble — Lopez de Prado AFML Ch.3. Production winner.",
      bullets: [
        "Primary = multiclass argmax from P3-featured LightGBM",
        'Meta = binary classifier: "will the primary prediction be correct?"',
        "If meta confidence < 0.5 → fallback to BASE allocation (75 / 10 / 15)",
        "Cuts turnover (~7% → ~3.5% / mo) and MaxDD (−17% → −16.5%)",
        "Only variant with highest Sharpe and lowest DD vs benchmark",
      ],
    },
  ];

  return (
    <div className="bg-[#111827] rounded-lg p-4 border border-gray-800">
      <div className="flex items-baseline justify-between mb-3">
        <div className="text-base font-semibold text-gray-100">Model Definitions (P0 → P4)</div>
        <div className="text-xs text-gray-500">Incremental ablation — each phase isolates one design decision.</div>
      </div>
      <div className="grid grid-cols-5 gap-3">
        {phases.map((p) => (
          <div key={p.tag}
               className={`bg-[#0a0e17] border rounded-lg p-3 flex flex-col ${
                 p.tag === "P4" ? "border-green-700/60 ring-1 ring-green-700/40" : "border-gray-800"
               }`}>
            <div className="flex items-baseline justify-between mb-1">
              <span className="text-lg font-bold" style={{ color: p.color }}>{p.tag}</span>
              {p.tag === "P4" && (
                <span className="text-[9px] uppercase tracking-wider text-green-400 font-semibold">
                  Winner
                </span>
              )}
            </div>
            <div className="text-sm font-semibold text-gray-100">{p.title}</div>
            <div className="text-[11px] text-gray-400 italic mt-1 mb-2">{p.short}</div>
            <ul className="text-[10.5px] text-gray-300 leading-snug list-disc pl-4 space-y-0.5">
              {p.bullets.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="text-[11px] text-gray-500 mt-3">
        <b>Composition</b>: P4 sits on top of P3 features and uses the same CV as P0. P1 and P2
        are independently evaluated variants (not stacked into P4 in this run; ablation showed
        they provided marginal lift only). <b>MoE</b> (Plan B, see below) attempts to combine
        experts conditional on regime.
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance Analytics Section (institutional-style metrics)
// AQR / Bridgewater / BlackRock Aladdin / MSCI Barra / Man AHL standards
// ─────────────────────────────────────────────────────────────────────────────
function PerformanceAnalyticsSection({ perf }: { perf: PerfData }) {
  const order = perf.summary.variant_order;
  const variants = perf.summary.variants;
  const bmLabel = perf.summary.benchmark_label;

  // Build rolling time series per variant per window
  const rollingByWindow = useMemo(() => {
    const out: Record<number, Record<string, { dates: string[]; values: Record<string, number[]> }>> = {};
    for (const w of [12, 36, 60]) {
      out[w] = {};
      for (const tag of order) {
        const subset = perf.rolling
          .filter((r) => r.variant === tag && r.window === w && r[`sharpe_${w}m`] != null)
          .sort((a, b) => a.date.localeCompare(b.date));
        out[w][tag] = {
          dates: subset.map((r) => r.date),
          values: {
            cagr:   subset.map((r) => Number(r[`cagr_${w}m`] ?? NaN)),
            sharpe: subset.map((r) => Number(r[`sharpe_${w}m`] ?? NaN)),
            vol:    subset.map((r) => Number(r[`vol_${w}m`] ?? NaN)),
            maxdd:  subset.map((r) => Number(r[`maxdd_${w}m`] ?? NaN)),
            ir:     subset.map((r) => Number(r[`ir_${w}m`] ?? NaN)),
          },
        };
      }
    }
    return out;
  }, [perf.rolling, order]);

  // Drawdown underwater curves (full OOS per variant)
  const underwater = useMemo(() => {
    const out: Record<string, { dates: string[]; dd: number[] }> = {};
    for (const tag of order) {
      const subset = perf.monthly
        .filter((r) => r.variant === tag)
        .sort((a, b) => a.date.localeCompare(b.date));
      const dates = subset.map((r) => r.date);
      let curve = 1;
      let peak = 1;
      const dd: number[] = [];
      for (const r of subset) {
        curve *= 1 + r.strat_ret;
        peak = Math.max(peak, curve);
        dd.push(curve / peak - 1);
      }
      out[tag] = { dates, dd };
    }
    return out;
  }, [perf.monthly, order]);

  // Monthly return series for distribution analysis
  const monthlyByVariant = useMemo(() => {
    const out: Record<string, number[]> = {};
    for (const tag of order) {
      out[tag] = perf.monthly
        .filter((r) => r.variant === tag)
        .map((r) => r.strat_ret);
    }
    return out;
  }, [perf.monthly, order]);

  // Helper: line trace per variant
  const lineTraces = (window: number, key: "cagr" | "sharpe" | "vol" | "maxdd" | "ir") => {
    return order.map((tag) => {
      const d = rollingByWindow[window]?.[tag];
      if (!d) return null;
      return {
        x: d.dates, y: d.values[key],
        type: "scatter", mode: "lines", name: tag,
        line: { color: VARIANT_COLORS[tag], width: tag === "P4" ? 2.5 : 1.5 },
        connectgaps: false,
      };
    }).filter(Boolean) as any[];
  };

  // Color helper for summary table (best per row green, worst red)
  const cellColor = (
    metric: keyof VariantMetrics,
    val: number,
    higherBetter: boolean,
  ) => {
    const vals = order.map((t) => Number(variants[t][metric])).filter((v) => Number.isFinite(v));
    if (vals.length === 0) return "";
    const best = higherBetter ? Math.max(...vals) : Math.min(...vals);
    const worst = higherBetter ? Math.min(...vals) : Math.max(...vals);
    if (Math.abs(val - best) < 1e-9) return "text-green-300 font-semibold";
    if (Math.abs(val - worst) < 1e-9) return "text-red-300";
    return "text-gray-200";
  };

  // Rows for summary table — { label, key, fmt, higherBetter, group }
  const rows: Array<{
    label: string; key: keyof VariantMetrics; fmt: "pct" | "num" | "int";
    higherBetter: boolean; group: string; tooltip?: string;
  }> = [
    { label: "Period (months)",       key: "n_months",      fmt: "int", higherBetter: true,  group: "Period" },
    { label: "CAGR",                  key: "cagr",          fmt: "pct", higherBetter: true,  group: "Absolute" },
    { label: "Annualized Vol",        key: "ann_vol",       fmt: "pct", higherBetter: false, group: "Absolute" },
    { label: "Max Drawdown",          key: "max_dd",        fmt: "pct", higherBetter: true,  group: "Absolute",   tooltip: "Closer to 0 is better" },
    { label: "Max DD Duration (mo)",  key: "max_dd_duration_months", fmt: "int", higherBetter: false, group: "Absolute" },
    { label: "Sharpe",                key: "sharpe",        fmt: "num", higherBetter: true,  group: "Risk-adjusted" },
    { label: "Sortino",               key: "sortino",       fmt: "num", higherBetter: true,  group: "Risk-adjusted" },
    { label: "Calmar",                key: "calmar",        fmt: "num", higherBetter: true,  group: "Risk-adjusted" },
    { label: "Omega",                 key: "omega",         fmt: "num", higherBetter: true,  group: "Risk-adjusted" },
    { label: "Alpha (ann)",           key: "alpha_ann",     fmt: "pct", higherBetter: true,  group: "vs Benchmark" },
    { label: "Tracking Error",        key: "tracking_error", fmt: "pct", higherBetter: false, group: "vs Benchmark" },
    { label: "Information Ratio",     key: "information_ratio", fmt: "num", higherBetter: true, group: "vs Benchmark" },
    { label: "Beta",                  key: "beta",          fmt: "num", higherBetter: false, group: "vs Benchmark" },
    { label: "Up Capture",            key: "up_capture",    fmt: "num", higherBetter: true,  group: "vs Benchmark" },
    { label: "Down Capture",          key: "dn_capture",    fmt: "num", higherBetter: false, group: "vs Benchmark" },
    { label: "Win Rate",              key: "win_rate",      fmt: "pct", higherBetter: true,  group: "Distribution" },
    { label: "Skewness",              key: "skew",          fmt: "num", higherBetter: true,  group: "Distribution" },
    { label: "Excess Kurtosis",       key: "kurt_excess",   fmt: "num", higherBetter: false, group: "Distribution" },
    { label: "VaR 5% (monthly)",      key: "var_5_monthly", fmt: "pct", higherBetter: true,  group: "Tail Risk" },
    { label: "CVaR 5% (monthly)",     key: "cvar_5_monthly", fmt: "pct", higherBetter: true, group: "Tail Risk" },
    { label: "Turnover (% / mo)",     key: "mean_turnover_pct", fmt: "num", higherBetter: false, group: "Cost" },
  ];

  const fmtVal = (v: any, fmt: "pct" | "num" | "int") => {
    if (v == null || (typeof v === "number" && !Number.isFinite(v))) return "—";
    if (fmt === "pct") return `${(Number(v) * 100).toFixed(2)}%`;
    if (fmt === "int") return String(Math.round(Number(v)));
    return Number(v).toFixed(2);
  };

  // Group rows
  const groupedRows = useMemo(() => {
    const g: Record<string, typeof rows> = {};
    for (const r of rows) {
      if (!g[r.group]) g[r.group] = [];
      g[r.group].push(r);
    }
    return g;
  }, []);

  // Up/Down capture combined bar chart
  const captureData = order.map((tag) => ({
    tag, up: variants[tag].up_capture, dn: variants[tag].dn_capture,
  }));

  // Distribution: histogram per variant (monthly returns)
  const histTraces = order.map((tag) => ({
    type: "histogram",
    x: monthlyByVariant[tag],
    name: tag,
    marker: { color: VARIANT_COLORS[tag], opacity: 0.55 },
    nbinsx: 30,
    histnorm: "probability density" as const,
  }));

  return (
    <div className="bg-[#111827] rounded-lg p-4 border border-gray-800 space-y-5">
      <div>
        <div className="text-base font-semibold text-gray-100">
          Performance Analytics — Institutional-Style Comparison
        </div>
        <div className="text-xs text-gray-500 mt-1">
          Five variants benchmarked against <b className="text-cyan-300">{bmLabel}</b>.&nbsp;
          Methodology mirrors AQR / Bridgewater / BlackRock Aladdin / MSCI Barra / Man AHL reporting.&nbsp;
          Note: P4 has shorter OOS window ({variants["P4"].n_months} months vs {variants["P0"].n_months} for others) due to meta CV burn-in.
        </div>
      </div>

      {/* ── Summary metrics matrix ── */}
      <div className="overflow-x-auto border border-gray-800 rounded">
        <table className="w-full text-xs">
          <thead className="bg-[#1f2937] sticky top-0">
            <tr>
              <th className="px-2 py-1.5 text-left text-gray-400 border-b border-gray-700">Metric</th>
              {order.map((tag) => (
                <th key={tag} className="px-2 py-1.5 text-right border-b border-gray-700"
                    style={{ color: VARIANT_COLORS[tag] }}>
                  {tag}{tag === "P4" && " ★"}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(groupedRows).map(([group, gRows]) => (
              <>
                <tr key={`group-${group}`} className="bg-[#0a0e17]">
                  <td colSpan={order.length + 1}
                      className="px-2 py-1 text-[10px] uppercase tracking-wider text-gray-500 font-semibold">
                    {group}
                  </td>
                </tr>
                {gRows.map((r) => (
                  <tr key={r.key as string} className="border-b border-gray-800/50 hover:bg-[#1f2937]/40">
                    <td className="px-2 py-1 text-gray-300">{r.label}</td>
                    {order.map((tag) => {
                      const v = variants[tag][r.key] as number;
                      const cls = typeof v === "number" && Number.isFinite(v)
                        ? cellColor(r.key, v, r.higherBetter) : "text-gray-500";
                      return (
                        <td key={tag} className={`px-2 py-1 text-right ${cls}`}>
                          {fmtVal(v, r.fmt)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-[10px] text-gray-500">
        <span className="text-green-300">green</span> = best per row,&nbsp;
        <span className="text-red-300">red</span> = worst.&nbsp;
        Max DD / TE / Beta / Down Capture / Kurtosis / Turnover: lower is better.
      </div>

      {/* ── Rolling Sharpe (1Y, 3Y) ── */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-2">Rolling Sharpe Ratio</div>
        <div className="grid grid-cols-3 gap-3">
          {[12, 36, 60].map((w) => (
            <div key={w}>
              <div className="text-xs text-gray-400 mb-1">{w / 12}-Year window</div>
              <Plot
                data={lineTraces(w, "sharpe") as any}
                layout={{
                  ...DARK_LAYOUT, height: 260,
                  margin: { t: 15, b: 30, l: 40, r: 10 },
                  hovermode: "x unified",
                  yaxis: { title: "Sharpe", gridcolor: "#1f2937", zerolinecolor: "#374151" },
                  xaxis: { gridcolor: "#1f2937" },
                  legend: { orientation: "h", y: -0.18, font: { size: 10 } },
                  shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0,
                             line: { color: "#374151", dash: "dash", width: 1 } }],
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* ── Rolling CAGR (1Y, 3Y, 5Y) ── */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-2">Rolling Annualized Return (CAGR)</div>
        <div className="grid grid-cols-3 gap-3">
          {[12, 36, 60].map((w) => (
            <div key={w}>
              <div className="text-xs text-gray-400 mb-1">{w / 12}-Year window</div>
              <Plot
                data={lineTraces(w, "cagr") as any}
                layout={{
                  ...DARK_LAYOUT, height: 260,
                  margin: { t: 15, b: 30, l: 50, r: 10 },
                  hovermode: "x unified",
                  yaxis: { title: "CAGR", tickformat: ".1%", gridcolor: "#1f2937", zerolinecolor: "#374151" },
                  xaxis: { gridcolor: "#1f2937" },
                  legend: { orientation: "h", y: -0.18, font: { size: 10 } },
                  shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0,
                             line: { color: "#374151", dash: "dash", width: 1 } }],
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* ── Rolling Information Ratio ── */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-2">
          Rolling Information Ratio (vs {bmLabel})
        </div>
        <div className="grid grid-cols-2 gap-3">
          {[12, 36].map((w) => (
            <div key={w}>
              <div className="text-xs text-gray-400 mb-1">{w / 12}-Year window</div>
              <Plot
                data={lineTraces(w, "ir") as any}
                layout={{
                  ...DARK_LAYOUT, height: 260,
                  margin: { t: 15, b: 30, l: 40, r: 10 },
                  hovermode: "x unified",
                  yaxis: { title: "IR", gridcolor: "#1f2937", zerolinecolor: "#374151" },
                  xaxis: { gridcolor: "#1f2937" },
                  legend: { orientation: "h", y: -0.18, font: { size: 10 } },
                  shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0,
                             line: { color: "#374151", dash: "dash", width: 1 } }],
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%" }}
              />
            </div>
          ))}
        </div>
      </div>

      {/* ── Underwater curve (drawdown over time) + Capture bars ── */}
      <div className="grid grid-cols-3 gap-3">
        <div className="col-span-2">
          <div className="text-sm font-semibold text-gray-200 mb-1">
            Underwater Drawdown Curve
          </div>
          <Plot
            data={order.map((tag) => ({
              x: underwater[tag].dates,
              y: underwater[tag].dd,
              type: "scatter", mode: "lines", name: tag,
              line: { color: VARIANT_COLORS[tag], width: tag === "P4" ? 2 : 1.3 },
              fill: "tozeroy",
              fillcolor: VARIANT_COLORS[tag] + "22",
            })) as any}
            layout={{
              ...DARK_LAYOUT, height: 280,
              margin: { t: 15, b: 35, l: 50, r: 10 },
              hovermode: "x unified",
              yaxis: { title: "Drawdown", tickformat: ".0%", gridcolor: "#1f2937" },
              xaxis: { gridcolor: "#1f2937" },
              legend: { orientation: "h", y: -0.20, font: { size: 10 } },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="text-[10px] text-gray-500">
            Time spent in drawdown is a key institutional risk metric. Steeper troughs = larger losses.
          </div>
        </div>

        <div>
          <div className="text-sm font-semibold text-gray-200 mb-1">
            Up / Down Capture
          </div>
          <Plot
            data={[
              {
                type: "bar", name: "Up Capture",
                x: order, y: captureData.map((d) => d.up),
                marker: { color: C.green },
                text: captureData.map((d) => d.up.toFixed(2)),
                textposition: "auto",
              },
              {
                type: "bar", name: "Down Capture",
                x: order, y: captureData.map((d) => d.dn),
                marker: { color: C.red },
                text: captureData.map((d) => d.dn.toFixed(2)),
                textposition: "auto",
              },
            ] as any}
            layout={{
              ...DARK_LAYOUT, height: 280,
              margin: { t: 15, b: 25, l: 40, r: 10 },
              barmode: "group",
              yaxis: { gridcolor: "#1f2937" },
              xaxis: { gridcolor: "#1f2937" },
              legend: { orientation: "h", y: -0.18, font: { size: 10 } },
              shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1,
                         line: { color: "#9ca3af", dash: "dash", width: 1 } }],
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="text-[10px] text-gray-500">
            Dashed line = 1.0 (match benchmark in that direction).&nbsp;
            Up &gt; 1, Down &lt; 1 = ideal asymmetric profile.
          </div>
        </div>
      </div>

      {/* ── Monthly return distribution ── */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-1">
          Monthly Return Distribution
        </div>
        <Plot
          data={histTraces as any}
          layout={{
            ...DARK_LAYOUT, height: 280,
            margin: { t: 15, b: 35, l: 40, r: 10 },
            barmode: "overlay",
            xaxis: { title: "Monthly return", tickformat: ".0%", gridcolor: "#1f2937" },
            yaxis: { title: "Density", gridcolor: "#1f2937" },
            legend: { orientation: "h", y: -0.20, font: { size: 10 } },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
      </div>

      {/* ── Methodology footer ── */}
      <div className="bg-[#0a0e17] border border-cyan-900/40 rounded-lg p-3 text-[11px] text-cyan-200/90 leading-relaxed">
        <b className="text-cyan-100">Metric Glossary</b> —&nbsp;
        <span className="text-gray-300">
          <b>Sharpe</b> = ann excess return / vol (assumes 0% rf for monthly).&nbsp;
          <b>Sortino</b> = ann return / annualized downside std (only negative returns).&nbsp;
          <b>Calmar</b> = CAGR / |Max DD|.&nbsp;
          <b>Omega</b> = sum(gains) / sum(|losses|), threshold 0.&nbsp;
          <b>IR</b> = alpha / tracking error.&nbsp;
          <b>Up / Down Capture</b> = (CAGR_strat / CAGR_bench) restricted to months when benchmark return ≷ 0 (geometric).&nbsp;
          <b>VaR 5%</b> = 5th percentile of monthly returns.&nbsp;
          <b>CVaR 5%</b> = mean of monthly returns at or below VaR (Expected Shortfall).&nbsp;
          <b>Skew / Kurtosis</b> use bias-corrected Fisher (excess) form.
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Monthly Allocation by Variant — 5 stacked-area subplots (one per variant)
// Replaces the single-variant stacked area; full-width, larger height
// ─────────────────────────────────────────────────────────────────────────────
function MonthlyAllocationByVariant({ perf }: { perf: PerfData }) {
  const order = perf.summary.variant_order;

  const allocByVariant = useMemo(() => {
    const out: Record<string, { dates: string[]; eq: number[]; bd: number[]; ch: number[] }> = {};
    for (const tag of order) {
      const subset = perf.monthly
        .filter((r) => r.variant === tag)
        .sort((a, b) => a.date.localeCompare(b.date));
      out[tag] = {
        dates: subset.map((r) => r.date),
        eq: subset.map((r) => r.w_equity),
        bd: subset.map((r) => r.w_bond),
        ch: subset.map((r) => r.w_cash),
      };
    }
    return out;
  }, [perf.monthly, order]);

  // 5 subplots (1 column × 5 rows). Each variant gets its own xaxis/yaxis pair
  // and its own stackgroup so the eq/bd/cash traces stack to 100% within that panel.
  const traces: any[] = [];
  order.forEach((tag, i) => {
    const ax = i === 0 ? "x"  : `x${i + 1}`;
    const ay = i === 0 ? "y"  : `y${i + 1}`;
    const data = allocByVariant[tag];
    if (!data || data.dates.length === 0) return;

    // Equity
    traces.push({
      type: "scatter", mode: "lines",
      x: data.dates, y: data.eq,
      stackgroup: tag, name: "Equity",
      legendgroup: "Equity", showlegend: i === 0,
      line: { width: 0, color: C.green },
      fillcolor: C.green,
      xaxis: ax, yaxis: ay,
      hovertemplate: "%{y:.0%}<extra>" + tag + " Equity</extra>",
    });
    // Bond
    traces.push({
      type: "scatter", mode: "lines",
      x: data.dates, y: data.bd,
      stackgroup: tag, name: "Bond",
      legendgroup: "Bond", showlegend: i === 0,
      line: { width: 0, color: C.blue },
      fillcolor: C.blue,
      xaxis: ax, yaxis: ay,
      hovertemplate: "%{y:.0%}<extra>" + tag + " Bond</extra>",
    });
    // Cash
    traces.push({
      type: "scatter", mode: "lines",
      x: data.dates, y: data.ch,
      stackgroup: tag, name: "Cash",
      legendgroup: "Cash", showlegend: i === 0,
      line: { width: 0, color: C.yellow },
      fillcolor: C.yellow,
      xaxis: ax, yaxis: ay,
      hovertemplate: "%{y:.0%}<extra>" + tag + " Cash</extra>",
    });
  });

  // Per-variant y-axis config — annotate each panel with the variant tag
  const yAxisCommon = {
    range: [0, 1], tickformat: ".0%",
    gridcolor: "#1f2937", zerolinecolor: "#374151",
    fixedrange: true,
  };
  const xAxisCommon = { gridcolor: "#1f2937" };

  // Annotations: variant label inside each panel (top-left)
  const annotations = order.map((tag, i) => ({
    xref: i === 0 ? "x domain" : `x${i + 1} domain`,
    yref: i === 0 ? "y domain" : `y${i + 1} domain`,
    x: 0.005, y: 0.96,
    xanchor: "left" as const, yanchor: "top" as const,
    text: `<b>${tag}</b>${tag === "P4" ? " ★" : ""}`,
    showarrow: false,
    font: { size: 13, color: VARIANT_COLORS[tag] },
    bgcolor: "rgba(10,14,23,0.65)",
    borderpad: 3,
  }));

  // Benchmark reference: 90% equity (constant) — horizontal dashed line per panel
  const bmEq = perf.summary.benchmark_weights?.equity ?? 0.9;
  const shapes = order.flatMap((_, i) => ([{
    type: "line" as const,
    xref: i === 0 ? "x domain" : `x${i + 1} domain`,
    yref: i === 0 ? "y" : `y${i + 1}`,
    x0: 0, x1: 1, y0: bmEq, y1: bmEq,
    line: { color: "rgba(156,163,175,0.55)", dash: "dot", width: 1 },
  }]));

  // Build dynamic layout
  const layout: any = {
    ...DARK_LAYOUT,
    height: 760,
    margin: { t: 30, b: 40, l: 55, r: 15 },
    grid: {
      rows: order.length, columns: 1,
      pattern: "independent",
      roworder: "top to bottom",
      ygap: 0.12,
    },
    legend: {
      orientation: "h", x: 0.5, xanchor: "center",
      y: 1.07, yanchor: "bottom",
      bgcolor: "rgba(0,0,0,0)",
    },
    annotations,
    shapes,
  };
  // Apply per-axis config
  for (let i = 0; i < order.length; i++) {
    const ySuffix = i === 0 ? "" : String(i + 1);
    const xSuffix = i === 0 ? "" : String(i + 1);
    layout[`yaxis${ySuffix}`] = { ...yAxisCommon };
    layout[`xaxis${xSuffix}`] = {
      ...xAxisCommon,
      showticklabels: i === order.length - 1,  // only show x ticks on bottom panel
    };
  }

  return (
    <div className="bg-[#111827] rounded-lg p-4 border border-gray-800">
      <div className="flex items-baseline justify-between mb-1">
        <div className="text-sm font-semibold text-gray-200">
          Monthly Allocation by Variant — Equity / Bond / Cash (OOS → Today)
        </div>
        <div className="text-[10.5px] text-gray-500">
          Dotted line = benchmark equity weight ({(bmEq * 100).toFixed(0)}%).
          Each panel = monthly rebalanced allocation that the variant prescribed.
        </div>
      </div>
      <Plot
        data={traces}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
      <div className="text-[11px] text-gray-500 mt-2 leading-relaxed">
        <b className="text-gray-300">How to read</b>: each row is one variant; vertical thickness of
        a color band at any date = that asset's weight at that month-end rebalance.
        <span className="text-green-300"> Equity</span> compresses (variant tilts defensive) when
        BEAR signal strengthens; <span className="text-yellow-300"> Cash</span> & <span className="text-blue-300"> Bond</span> expand.
        P4 (★) shows clearer regime switches and lower turnover than P0-P3 due to meta-labeling override→BASE.
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark Validation Section — tests user's core hypothesis
// "vs holding 100% equity, can adjusting bond/cash allocation produce
//  superior risk-adjusted long-term performance?"
// ─────────────────────────────────────────────────────────────────────────────
function BenchmarkValidationSection({ bench }: { bench: BenchmarkData }) {
  const s = bench.summary;
  const verdict = s.hypothesis_verdict.vs_acwi100;

  // Cumulative return chart data
  const cum = useMemo(() => {
    const dates = bench.monthly.map((m) => m.date);
    const mk = (k: string) => {
      let acc = 1;
      const out: number[] = [];
      for (const m of bench.monthly) {
        acc *= 1 + (Number(m[k]) || 0);
        out.push(acc);
      }
      return out;
    };
    return {
      dates,
      p4:           mk("p4"),
      acwi100:      mk("acwi100"),
      acwi75ch25:   mk("acwi75ch25"),
      acwi60agg40:  mk("acwi60agg40"),
      acwi90ch10:   mk("acwi90ch10"),
      tbill:        mk("tbill"),
    };
  }, [bench.monthly]);

  // Helper: P4 row + benchmark rows for table
  const p4Row = { ...s.p4_strategy, tag: "p4", name: "P4 Meta (winner)", role: "strategy" } as any;
  const allRows = [p4Row, ...s.benchmarks];

  // For risk-return scatter
  const scatterPoints = allRows.map((r) => ({
    x: r.ann_vol, y: r.ann_return, name: r.name,
    color: r.tag === "p4" ? C.green : (BENCH_COLOR[r.tag] ?? C.gray),
    size: r.tag === "p4" ? 18 : 12,
  }));

  // Cell formatter helpers
  const fmtPct = (v: any, digits = 2) => {
    if (v == null || (typeof v === "number" && !Number.isFinite(v))) return "—";
    return `${(Number(v) * 100).toFixed(digits)}%`;
  };
  const fmtPctSig = (v: any, digits = 2) => {
    if (v == null || (typeof v === "number" && !Number.isFinite(v))) return "—";
    const s = (Number(v) * 100).toFixed(digits);
    return Number(v) >= 0 ? `+${s}%` : `${s}%`;
  };
  const fmtNum = (v: any, digits = 2) => {
    if (v == null || (typeof v === "number" && !Number.isFinite(v))) return "—";
    return Number(v).toFixed(digits);
  };
  const fmtNumSig = (v: any, digits = 2) => {
    if (v == null || (typeof v === "number" && !Number.isFinite(v))) return "—";
    const s = Number(v).toFixed(digits);
    return Number(v) >= 0 ? `+${s}` : s;
  };

  // Color helper
  const sigClass = (v: number | null | undefined, higherBetter = true) => {
    if (v == null || !Number.isFinite(v)) return "text-gray-400";
    if ((higherBetter && v > 0) || (!higherBetter && v < 0)) return "text-green-400";
    if ((higherBetter && v < 0) || (!higherBetter && v > 0)) return "text-red-400";
    return "text-gray-200";
  };

  return (
    <div className="bg-[#111827] rounded-lg p-4 border border-gray-800 space-y-5">
      {/* ── Header + Hypothesis ── */}
      <div>
        <div className="text-base font-semibold text-gray-100">
          Benchmark Validation Suite — Hypothesis Test
        </div>
        <div className="text-xs text-gray-500 mt-1">
          OOS window: {s.period_start} → {s.period_end} ({s.n_oos_months} months).
          Methodology: AQR / Bridgewater / BlackRock institutional reporting — strategy benchmarked against
          5 reference portfolios with risk-adjusted metrics.
        </div>
        <div className="mt-2 p-3 bg-[#0a0e17] border border-gray-800 rounded text-xs text-gray-300 italic">
          <b className="text-cyan-300">Hypothesis</b>: "vs holding 100% equity, can the P4 active asset
          allocation produce superior <u>risk-adjusted</u> long-term performance?"
        </div>
      </div>

      {/* ── Verdict Banner ── */}
      <div className={`rounded-lg p-4 border-2 ${
        verdict.overall_pass
          ? "bg-green-900/20 border-green-700/60"
          : "bg-red-900/20 border-red-700/60"
      }`}>
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className={`text-lg font-bold ${verdict.overall_pass ? "text-green-300" : "text-red-300"}`}>
              {verdict.overall_pass ? "✓ HYPOTHESIS VALIDATED" : "✗ HYPOTHESIS NOT VALIDATED"}
              <span className="text-sm font-normal ml-2 text-gray-300">
                ({verdict.score} risk-adjusted dimensions outperform vs ACWI 100%)
              </span>
            </div>
            <div className="text-xs text-gray-400 mt-1">
              P4 Sharpe {fmtNum(s.p4_strategy.sharpe)} vs ACWI100 Sharpe&nbsp;
              {fmtNum(s.benchmarks.find((b) => b.tag === "acwi100")?.sharpe)}
              &nbsp;|&nbsp;P4 MaxDD {fmtPct(s.p4_strategy.max_dd, 1)} vs ACWI100 MaxDD&nbsp;
              {fmtPct(s.benchmarks.find((b) => b.tag === "acwi100")?.max_dd, 1)}
            </div>
          </div>
          <div className="grid grid-cols-4 gap-2 text-[11px]">
            {[
              ["Sharpe", verdict.sharpe_better],
              ["Sortino", verdict.sortino_better],
              ["Calmar", verdict.calmar_better],
              ["MaxDD", verdict.max_dd_better],
            ].map(([label, ok]) => (
              <div key={label as string}
                   className={`px-2 py-1 rounded text-center font-semibold border ${
                     ok === true ? "border-green-600 text-green-300 bg-green-900/30"
                     : ok === false ? "border-red-600 text-red-300 bg-red-900/30"
                     : "border-gray-600 text-gray-400 bg-gray-900/30"
                   }`}>
                {label as string} {ok === true ? "✓" : ok === false ? "✗" : "—"}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Absolute metrics table ── */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-2">
          Absolute Metrics — P4 vs 5 Reference Portfolios
        </div>
        <div className="overflow-x-auto border border-gray-800 rounded">
          <table className="w-full text-xs">
            <thead className="bg-[#1f2937]">
              <tr>
                <th className="px-2 py-1.5 text-left text-gray-400 border-b border-gray-700">Strategy</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Weights</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">CAGR</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Vol</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Sharpe</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Sortino</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Calmar</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Max DD</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Up Cap</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Dn Cap</th>
              </tr>
            </thead>
            <tbody>
              {allRows.map((r) => {
                const isP4 = r.tag === "p4";
                const w = (r as any).weights;
                const wStr = w ? `${(w.equity * 100).toFixed(0)}/${(w.bond * 100).toFixed(0)}/${(w.cash * 100).toFixed(0)}` : "—";
                return (
                  <tr key={r.tag}
                      className={`border-b border-gray-800/50 ${
                        isP4 ? "bg-green-900/20 text-green-100 font-semibold"
                        : r.role === "primary" ? "bg-red-900/15"
                        : ""
                      }`}>
                    <td className="px-2 py-1">{r.name}{isP4 && " ★"}</td>
                    <td className="px-2 py-1 text-right text-gray-400">{isP4 ? "dynamic" : wStr}</td>
                    <td className="px-2 py-1 text-right">{fmtPct(r.ann_return)}</td>
                    <td className="px-2 py-1 text-right">{fmtPct(r.ann_vol)}</td>
                    <td className="px-2 py-1 text-right">{fmtNum(r.sharpe)}</td>
                    <td className="px-2 py-1 text-right">{fmtNum(r.sortino)}</td>
                    <td className="px-2 py-1 text-right">{fmtNum(r.calmar)}</td>
                    <td className="px-2 py-1 text-right">{fmtPct(r.max_dd, 1)}</td>
                    <td className="px-2 py-1 text-right">{fmtNum(r.up_capture)}</td>
                    <td className="px-2 py-1 text-right">{fmtNum(r.dn_capture)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="text-[10px] text-gray-500 mt-1">
          Up/Down Capture computed vs ACWI 100% (market reference).
          <span className="text-red-300"> red row</span> = primary benchmark (hypothesis null).
        </div>
      </div>

      {/* ── P4 vs each benchmark ── */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-2">
          P4 vs Each Benchmark — Alpha / IR / Risk-Adjusted Differential
        </div>
        <div className="overflow-x-auto border border-gray-800 rounded">
          <table className="w-full text-xs">
            <thead className="bg-[#1f2937]">
              <tr>
                <th className="px-2 py-1.5 text-left text-gray-400 border-b border-gray-700">Benchmark</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Alpha (ann)</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">TE</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">IR</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Δ Sharpe</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Δ Sortino</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Δ Calmar</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">Δ MaxDD</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">P4 UpCap</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">P4 DnCap</th>
              </tr>
            </thead>
            <tbody>
              {s.comparisons.map((c) => {
                const isPrimary = c.tag === "acwi100";
                return (
                  <tr key={c.tag}
                      className={`border-b border-gray-800/50 ${isPrimary ? "bg-red-900/15 font-semibold" : ""}`}>
                    <td className="px-2 py-1 text-gray-300">{c.name}</td>
                    <td className={`px-2 py-1 text-right ${sigClass(c.alpha_ann)}`}>{fmtPctSig(c.alpha_ann)}</td>
                    <td className="px-2 py-1 text-right">{fmtPct(c.tracking_error)}</td>
                    <td className={`px-2 py-1 text-right ${sigClass(c.information_ratio)}`}>{fmtNumSig(c.information_ratio)}</td>
                    <td className={`px-2 py-1 text-right ${sigClass(c.p4_sharpe_minus_bench)}`}>{fmtNumSig(c.p4_sharpe_minus_bench)}</td>
                    <td className={`px-2 py-1 text-right ${sigClass(c.p4_sortino_minus_bench)}`}>{fmtNumSig(c.p4_sortino_minus_bench)}</td>
                    <td className={`px-2 py-1 text-right ${sigClass(c.p4_calmar_minus_bench)}`}>{fmtNumSig(c.p4_calmar_minus_bench)}</td>
                    <td className={`px-2 py-1 text-right ${sigClass(c.p4_dd_minus_bench)}`}>{fmtPctSig(c.p4_dd_minus_bench, 1)}</td>
                    <td className="px-2 py-1 text-right">{fmtNum(c.p4_up_capture_vs_bench)}</td>
                    <td className="px-2 py-1 text-right">{fmtNum(c.p4_dn_capture_vs_bench)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="text-[10px] text-gray-500 mt-1">
          Δ Sharpe / Sortino / Calmar = P4 minus benchmark (positive = P4 better).
          Δ MaxDD = P4 minus benchmark (positive = P4 better, less negative).
        </div>
      </div>

      {/* ── Cumulative chart + Risk-Return scatter ── */}
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <div className="text-sm font-semibold text-gray-200 mb-1">
            Cumulative Return — P4 vs All Benchmarks
          </div>
          <Plot
            data={[
              { x: cum.dates, y: cum.p4,          type: "scatter", mode: "lines", name: "P4 (winner)",       line: { color: C.green, width: 2.5 } },
              { x: cum.dates, y: cum.acwi100,     type: "scatter", mode: "lines", name: "ACWI 100%",          line: { color: BENCH_COLOR.acwi100,    width: 1.5 } },
              { x: cum.dates, y: cum.acwi90ch10,  type: "scatter", mode: "lines", name: "ACWI 90 / Cash 10",  line: { color: BENCH_COLOR.acwi90ch10, width: 1.5, dash: "dash" } as any },
              { x: cum.dates, y: cum.acwi75ch25,  type: "scatter", mode: "lines", name: "ACWI 75 / Cash 25",  line: { color: BENCH_COLOR.acwi75ch25, width: 1.5 } },
              { x: cum.dates, y: cum.acwi60agg40, type: "scatter", mode: "lines", name: "ACWI 60 / AGG 40",   line: { color: BENCH_COLOR.acwi60agg40, width: 1.5 } },
              { x: cum.dates, y: cum.tbill,      type: "scatter", mode: "lines", name: "T-bill",             line: { color: BENCH_COLOR.tbill,      width: 1.2, dash: "dot" } as any },
            ] as any}
            layout={{
              ...DARK_LAYOUT, height: 380,
              margin: { t: 15, b: 30, l: 50, r: 10 },
              hovermode: "x unified",
              yaxis: { title: "Cumulative", gridcolor: "#1f2937" },
              xaxis: { gridcolor: "#1f2937" },
              legend: { orientation: "h", y: -0.18, font: { size: 10 } },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
        </div>

        <div>
          <div className="text-sm font-semibold text-gray-200 mb-1">
            Risk-Return Scatter
          </div>
          <Plot
            data={[{
              type: "scatter", mode: "markers+text",
              x: scatterPoints.map((p) => p.x),
              y: scatterPoints.map((p) => p.y),
              text: scatterPoints.map((p) => p.name.split(" ")[0]),
              textposition: "top center",
              textfont: { size: 10, color: "#9ca3af" },
              marker: {
                color: scatterPoints.map((p) => p.color),
                size: scatterPoints.map((p) => p.size),
                line: { color: "#0a0e17", width: 1 },
              },
              hovertemplate: "<b>%{text}</b><br>Vol=%{x:.2%}<br>CAGR=%{y:.2%}<extra></extra>",
            }] as any}
            layout={{
              ...DARK_LAYOUT, height: 380,
              margin: { t: 15, b: 35, l: 55, r: 10 },
              xaxis: { title: "Annualized Vol", tickformat: ".0%", gridcolor: "#1f2937" },
              yaxis: { title: "CAGR", tickformat: ".0%", gridcolor: "#1f2937" },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="text-[10px] text-gray-500 mt-1">
            Top-left = ideal (high return, low vol). P4 (★ green) sits above the
            risk/return frontier vs single-asset references.
          </div>
        </div>
      </div>

      {/* ── Interpretation ── */}
      <div className="bg-[#0a0e17] border border-cyan-900/40 rounded-lg p-3 text-[11px] text-cyan-200/90 leading-relaxed">
        <b className="text-cyan-100">Interpretation</b> —&nbsp;
        <span className="text-gray-300">
          <b className="text-green-300">P4 vs ACWI 100% (primary test)</b>: P4 sacrifices&nbsp;
          {fmtPctSig(s.comparisons.find((c) => c.tag === "acwi100")?.alpha_ann ?? 0)} of absolute alpha
          but delivers higher Sharpe (Δ {fmtNumSig(s.comparisons.find((c) => c.tag === "acwi100")?.p4_sharpe_minus_bench ?? 0)})
          and shallower MaxDD (Δ {fmtPctSig(s.comparisons.find((c) => c.tag === "acwi100")?.p4_dd_minus_bench ?? 0, 1)}).
          → <b className="text-green-300">Hypothesis validated</b>: bond/cash adjustment improves
          risk-adjusted profile vs passive equity.&nbsp;
          <b>vs ACWI 60 / AGG 40</b>: P4 generates positive alpha + IR&nbsp;
          {fmtNumSig(s.comparisons.find((c) => c.tag === "acwi60agg40")?.information_ratio ?? 0)} —
          beats institutional 60/40 standard on both absolute and risk-adjusted basis.&nbsp;
          <b>vs ACWI 75 / Cash 25</b>: marginal alpha (≈0%) — timing alpha vs static average
          allocation is small but positive.
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal Definition Tab — detailed walkthrough of how P0~P4 generate signals
// Designed for non-technical audience: visual flow, clear language, examples
// ─────────────────────────────────────────────────────────────────────────────
function SignalDefinitionContent() {
  return (
    <div className="space-y-6">
      {/* ── 1. Big Picture Flow ── */}
      <div className="bg-[#111827] rounded-lg p-5 border border-gray-800">
        <div className="text-base font-semibold text-gray-100 mb-1">
          1. 전체 흐름 — 한 눈에 보기
        </div>
        <div className="text-xs text-gray-500 mb-4">
          매월 마지막 거래일에 4단계로 자산배분이 결정됩니다.
        </div>
        <div className="grid grid-cols-7 gap-2 items-center">
          {/* Step 1 */}
          <div className="col-span-1 bg-[#0a0e17] border-2 border-cyan-700/50 rounded-lg p-3 text-center">
            <div className="text-cyan-300 text-xs font-bold mb-1">STEP 1</div>
            <div className="text-2xl mb-1">📊</div>
            <div className="text-xs text-gray-200 font-semibold">데이터 수집</div>
            <div className="text-[10px] text-gray-500 mt-1">VIX, 금리, 환율, 종목 breadth 등</div>
          </div>
          <div className="col-span-1 text-center text-2xl text-gray-600">→</div>
          {/* Step 2 */}
          <div className="col-span-1 bg-[#0a0e17] border-2 border-blue-700/50 rounded-lg p-3 text-center">
            <div className="text-blue-300 text-xs font-bold mb-1">STEP 2</div>
            <div className="text-2xl mb-1">🤖</div>
            <div className="text-xs text-gray-200 font-semibold">ML 분류 모델</div>
            <div className="text-[10px] text-gray-500 mt-1">LightGBM multiclass</div>
          </div>
          <div className="col-span-1 text-center text-2xl text-gray-600">→</div>
          {/* Step 3 */}
          <div className="col-span-1 bg-[#0a0e17] border-2 border-yellow-700/50 rounded-lg p-3 text-center">
            <div className="text-yellow-300 text-xs font-bold mb-1">STEP 3</div>
            <div className="text-2xl mb-1">🎯</div>
            <div className="text-xs text-gray-200 font-semibold">시장 상태 확률</div>
            <div className="text-[10px] text-gray-500 mt-1">P(BULL), P(BASE), P(BEAR)</div>
          </div>
          <div className="col-span-1 text-center text-2xl text-gray-600">→</div>
        </div>
        <div className="grid grid-cols-7 gap-2 items-center mt-2">
          <div className="col-span-5"></div>
          <div className="col-span-1 text-center text-2xl text-gray-600">↓</div>
          <div className="col-span-1"></div>
        </div>
        <div className="grid grid-cols-7 gap-2 items-center mt-2">
          <div className="col-span-5"></div>
          <div className="col-span-2 bg-[#0a0e17] border-2 border-green-700/50 rounded-lg p-3 text-center">
            <div className="text-green-300 text-xs font-bold mb-1">STEP 4</div>
            <div className="text-2xl mb-1">💼</div>
            <div className="text-xs text-gray-200 font-semibold">자산배분 결정</div>
            <div className="text-[10px] text-gray-500 mt-1">주식 / 채권 / 현금 비율 산출</div>
          </div>
        </div>
      </div>

      {/* ── 2. Three Regime Definitions ── */}
      <div className="bg-[#111827] rounded-lg p-5 border border-gray-800">
        <div className="text-base font-semibold text-gray-100 mb-1">
          2. 시장 상태 (Regime) — 3가지로 분류
        </div>
        <div className="text-xs text-gray-500 mb-4">
          모든 모델은 공통으로 시장을 BULL / BASE / BEAR 셋 중 하나로 봅니다.
          각 상태에는 미리 정해진 자산배분 비율(allocation grid)이 있습니다.
        </div>
        <div className="grid grid-cols-3 gap-4">
          {/* BULL */}
          <div className="bg-gradient-to-br from-green-900/30 to-[#0a0e17] border-2 border-green-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-3xl">🟢</span>
              <div>
                <div className="text-lg font-bold text-green-300">BULL</div>
                <div className="text-[11px] text-gray-400">강세장 (Bull Market)</div>
              </div>
            </div>
            <div className="text-xs text-gray-300 mb-3 leading-relaxed">
              <b>의미:</b> 다음 1개월간 주식이 +3% 이상 오를 것으로 예상,
              큰 낙폭(-2% 이상) 없을 것으로 예상.
            </div>
            <div className="bg-[#0a0e17]/50 rounded p-3 mb-2">
              <div className="text-[10px] text-green-300 mb-1">자산배분</div>
              <div className="grid grid-cols-3 gap-1 text-center">
                <div><div className="text-2xl font-bold text-green-300">90%</div><div className="text-[10px] text-gray-500">주식</div></div>
                <div><div className="text-2xl font-bold text-blue-300">5%</div><div className="text-[10px] text-gray-500">채권</div></div>
                <div><div className="text-2xl font-bold text-yellow-300">5%</div><div className="text-[10px] text-gray-500">현금</div></div>
              </div>
            </div>
            <div className="text-[11px] text-gray-400 italic">
              "거의 다 주식, 약간의 안전자산"
            </div>
            <div className="text-[10px] text-gray-500 mt-2">발생 빈도 ≈ 14% (최근 33년)</div>
          </div>

          {/* BASE */}
          <div className="bg-gradient-to-br from-yellow-900/30 to-[#0a0e17] border-2 border-yellow-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-3xl">🟡</span>
              <div>
                <div className="text-lg font-bold text-yellow-300">BASE</div>
                <div className="text-[11px] text-gray-400">중립장 (Base / Normal)</div>
              </div>
            </div>
            <div className="text-xs text-gray-300 mb-3 leading-relaxed">
              <b>의미:</b> 평소 시장. 큰 상승도, 큰 하락도 아닌 상태.
              BULL 또는 BEAR 조건이 충족되지 않은 모든 경우.
            </div>
            <div className="bg-[#0a0e17]/50 rounded p-3 mb-2">
              <div className="text-[10px] text-yellow-300 mb-1">자산배분</div>
              <div className="grid grid-cols-3 gap-1 text-center">
                <div><div className="text-2xl font-bold text-green-300">75%</div><div className="text-[10px] text-gray-500">주식</div></div>
                <div><div className="text-2xl font-bold text-blue-300">10%</div><div className="text-[10px] text-gray-500">채권</div></div>
                <div><div className="text-2xl font-bold text-yellow-300">15%</div><div className="text-[10px] text-gray-500">현금</div></div>
              </div>
            </div>
            <div className="text-[11px] text-gray-400 italic">
              "주식 위주, 적당한 안전자산 — 표준 balanced fund"
            </div>
            <div className="text-[10px] text-gray-500 mt-2">발생 빈도 ≈ 51%</div>
          </div>

          {/* BEAR */}
          <div className="bg-gradient-to-br from-red-900/30 to-[#0a0e17] border-2 border-red-700/50 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-3xl">🔴</span>
              <div>
                <div className="text-lg font-bold text-red-300">BEAR</div>
                <div className="text-[11px] text-gray-400">약세장 (Bear Market)</div>
              </div>
            </div>
            <div className="text-xs text-gray-300 mb-3 leading-relaxed">
              <b>의미:</b> 다음 1개월간 주식이 −3% 이상 하락 예상,
              또는 −5% 이상 drawdown 예상, 또는 VIX &gt; 30 (스트레스).
            </div>
            <div className="bg-[#0a0e17]/50 rounded p-3 mb-2">
              <div className="text-[10px] text-red-300 mb-1">자산배분</div>
              <div className="grid grid-cols-3 gap-1 text-center">
                <div><div className="text-2xl font-bold text-green-300">60%</div><div className="text-[10px] text-gray-500">주식</div></div>
                <div><div className="text-2xl font-bold text-blue-300">15%</div><div className="text-[10px] text-gray-500">채권</div></div>
                <div><div className="text-2xl font-bold text-yellow-300">25%</div><div className="text-[10px] text-gray-500">현금</div></div>
              </div>
            </div>
            <div className="text-[11px] text-gray-400 italic">
              "방어적 — 주식 줄이고 안전자산 늘림"
            </div>
            <div className="text-[10px] text-gray-500 mt-2">발생 빈도 ≈ 35%</div>
          </div>
        </div>
      </div>

      {/* ── 3. Two Decision Methods ── */}
      <div className="bg-[#111827] rounded-lg p-5 border border-gray-800">
        <div className="text-base font-semibold text-gray-100 mb-1">
          3. 의사결정 방식 — 두 가지
        </div>
        <div className="text-xs text-gray-500 mb-4">
          모델이 출력한 확률을 가지고 자산배분 비율을 정하는 방법은 두 가지입니다.
          P0~P3는 방식 1, P4는 방식 2를 사용합니다.
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* Method 1 — Posterior Blend */}
          <div className="bg-[#0a0e17] border border-cyan-800/50 rounded-lg p-4">
            <div className="flex items-baseline gap-2 mb-2">
              <span className="text-cyan-300 font-bold">방식 1</span>
              <span className="text-sm text-gray-200 font-semibold">Posterior Blend (확률 가중 평균)</span>
            </div>
            <div className="text-[11px] text-gray-400 mb-3">사용 모델: <b>P0, P1, P2, P3</b></div>
            <div className="bg-[#111827] rounded p-2 text-[11px] font-mono text-gray-300 mb-3">
              <div className="text-cyan-300 text-[10px] mb-1">RULE</div>
              w_final = Σ P(regime) × ALLOCATION[regime]
            </div>
            <div className="text-[11px] text-gray-300 mb-2"><b>예시 계산</b>:</div>
            <div className="bg-[#111827] rounded p-3 text-[11px] font-mono text-gray-300 mb-3">
              <div className="text-yellow-300 text-[10px] mb-1">INPUT (확률)</div>
              P_BEAR = 0.30 &nbsp; P_BASE = 0.50 &nbsp; P_BULL = 0.20<br/>
              <div className="text-green-300 text-[10px] mt-2 mb-1">OUTPUT (비율)</div>
              주식 = 0.30×60% + 0.50×75% + 0.20×90% = <b className="text-green-300">73.5%</b><br/>
              채권 = 0.30×15% + 0.50×10% + 0.20×5% &nbsp;= <b className="text-blue-300">10.5%</b><br/>
              현금 = 0.30×25% + 0.50×15% + 0.20×5% &nbsp;= <b className="text-yellow-300">16.0%</b>
            </div>
            <div className="text-[11px] text-gray-400">
              <b className="text-gray-300">장점:</b> 매월 부드럽게 변화 (whipsaw 적음)<br/>
              <b className="text-gray-300">단점:</b> 강한 신호도 희석됨
            </div>
          </div>

          {/* Method 2 — Hard Switch with Meta */}
          <div className="bg-[#0a0e17] border-2 border-green-700/60 rounded-lg p-4">
            <div className="flex items-baseline gap-2 mb-2">
              <span className="text-green-300 font-bold">방식 2 ★</span>
              <span className="text-sm text-gray-200 font-semibold">Hard Switch + Meta Filter</span>
            </div>
            <div className="text-[11px] text-gray-400 mb-3">사용 모델: <b>P4 (production winner)</b></div>
            <div className="bg-[#111827] rounded p-2 text-[11px] font-mono text-gray-300 mb-3">
              <div className="text-green-300 text-[10px] mb-1">RULE</div>
              <span className="text-gray-400">1.</span> primary = argmax(P_BEAR, P_BASE, P_BULL)<br/>
              <span className="text-gray-400">2.</span> if meta_confidence &gt; 0.5:<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;use ALLOCATION[primary]<br/>
              <span className="text-gray-400">3.</span> else:<br/>
              &nbsp;&nbsp;&nbsp;&nbsp;use ALLOCATION[BASE]
            </div>
            <div className="text-[11px] text-gray-300 mb-2"><b>예시 계산</b>:</div>
            <div className="bg-[#111827] rounded p-3 text-[11px] font-mono text-gray-300 mb-3">
              <div className="text-yellow-300 text-[10px] mb-1">CASE A</div>
              P_BEAR=0.6, P_BASE=0.3, P_BULL=0.1, meta=0.7 (강한 BEAR + 높은 신뢰)<br/>
              → primary = BEAR, meta &gt; 0.5 → <b className="text-red-300">BEAR 60/15/25</b><br/>
              <div className="text-yellow-300 text-[10px] mt-2 mb-1">CASE B</div>
              P_BEAR=0.6, P_BASE=0.3, P_BULL=0.1, meta=0.4 (강한 BEAR + 낮은 신뢰)<br/>
              → primary = BEAR but meta ≤ 0.5 → <b className="text-yellow-300">BASE 75/10/15</b> (안전 후퇴)
            </div>
            <div className="text-[11px] text-gray-400">
              <b className="text-gray-300">장점:</b> 명확한 의사결정, 잘못된 베팅 회피<br/>
              <b className="text-gray-300">단점:</b> 경계영역에서 보수적 (BASE로 후퇴)
            </div>
          </div>
        </div>
      </div>

      {/* ── 4. P0 → P4 Detailed Walk-through ── */}
      <div className="bg-[#111827] rounded-lg p-5 border border-gray-800">
        <div className="text-base font-semibold text-gray-100 mb-1">
          4. P0 → P4 모델별 차이
        </div>
        <div className="text-xs text-gray-500 mb-4">
          5개 모델은 모두 동일한 시장 상태(BULL/BASE/BEAR)를 예측하지만,
          입력 feature와 의사결정 방식이 점진적으로 발전합니다.
        </div>

        <div className="space-y-3">
          <ModelExplainCard
            tag="P0" tagColor={C.cyan} title="Baseline"
            features="15개 macro 지표"
            featureExamples="VIX 수준, 10년 금리, 12개월 모멘텀, 달러 인덱스"
            decision="방식 1: Posterior Blend"
            insight="가장 단순한 reference 모델. 무엇이 추가됐을 때 효과가 있는지 비교 기준."
          />
          <ModelExplainCard
            tag="P1" tagColor={C.orange} title="+ Breadth Features"
            features="P0 + 6개 종목 breadth 지표"
            featureExamples="% 주식 ETF가 상승 추세 / % 하락 추세, TCS 중앙값, RSS 분산"
            decision="방식 1: Posterior Blend"
            insight="개별 종목 신호를 집계해 시장 폭(breadth)을 추가. BlackRock SAE 스타일."
          />
          <ModelExplainCard
            tag="P2" tagColor={C.yellow} title="+ BULL Detection"
            features="P0 동일 (15개)"
            featureExamples="동일"
            decision="방식 1 + class weight + threshold tuning"
            insight="BULL 클래스 가중치 ×3, P(BULL) > 0.25면 BULL 분류. 소수 클래스 보정."
          />
          <ModelExplainCard
            tag="P3" tagColor={C.blue} title="+ Macro Features 확장"
            features="P0 + 6개 cross-asset 지표"
            featureExamples="VIX/MOVE 비율, VIX/VXV term structure, WTI/Copper 비율, TIP/IEF inflation"
            decision="방식 1: Posterior Blend"
            insight="추가 macro feature로 정보량 증가. 비교적 안전한 기능 추가."
          />
          <ModelExplainCard
            tag="P4" tagColor={C.green} title="Meta-Labeling (★ Winner)"
            features="P3 + P1 = 27개 total"
            featureExamples="모든 features 사용 (15 macro + 6 P3 + 6 breadth)"
            decision="방식 2: Hard Switch + Meta Filter"
            insight="2단계 모델. Primary가 예측 → Meta가 '신뢰할 수 있나?' 판단 → 신뢰도 낮으면 BASE로 후퇴.
                     Lopez de Prado AFML Ch.3 표준."
            isWinner
          />
        </div>
      </div>

      {/* ── 5. Features 상세 설명 ── */}
      <FeaturesDetailSection />

      {/* ── 6. Meta-Labeling 깊이 있는 설명 ── */}
      <MetaLabelingDeepDive />


      {/* ── 6. Worked Example ── */}
      <div className="bg-[#111827] rounded-lg p-5 border border-gray-800">
        <div className="text-base font-semibold text-gray-100 mb-1">
          7. 실제 한 달 의사결정 — Walkthrough
        </div>
        <div className="text-xs text-gray-500 mb-4">
          가상의 월말(예: 2024-12-31) 시점에서 P4가 어떻게 의사결정 하는지 단계별로.
        </div>
        <div className="space-y-3">
          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-1 text-center bg-cyan-900/30 border border-cyan-700 rounded-full w-8 h-8 flex items-center justify-center text-cyan-300 font-bold">1</div>
            <div className="col-span-11 bg-[#0a0e17] border border-gray-800 rounded-lg p-3">
              <div className="text-sm font-semibold text-gray-200 mb-1">데이터 수집 (월말)</div>
              <div className="text-[11px] text-gray-400 grid grid-cols-3 gap-2 mt-2 font-mono">
                <div>VIX = 19.5</div>
                <div>10Y yield = 4.2%</div>
                <div>USD index 3M ret = +1.8%</div>
                <div>ACWI 12M momentum = +18%</div>
                <div>HY/IG ratio = 0.74</div>
                <div>% bullish ETFs = 52%</div>
                <div className="text-gray-500 col-span-3">… 27개 features 전체</div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-1 text-center bg-blue-900/30 border border-blue-700 rounded-full w-8 h-8 flex items-center justify-center text-blue-300 font-bold">2</div>
            <div className="col-span-11 bg-[#0a0e17] border border-gray-800 rounded-lg p-3">
              <div className="text-sm font-semibold text-gray-200 mb-1">Primary 모델 예측</div>
              <div className="text-[11px] text-gray-400 mt-2 font-mono space-y-0.5">
                <div className="flex items-center gap-3">
                  <div className="w-20">P_BEAR</div>
                  <div className="flex-1 bg-[#1f2937] rounded h-3 relative overflow-hidden">
                    <div className="h-full bg-red-700/70" style={{ width: "15%" }}></div>
                  </div>
                  <div className="w-12 text-right text-red-300">0.15</div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-20">P_BASE</div>
                  <div className="flex-1 bg-[#1f2937] rounded h-3 relative overflow-hidden">
                    <div className="h-full bg-yellow-600/70" style={{ width: "50%" }}></div>
                  </div>
                  <div className="w-12 text-right text-yellow-300">0.50</div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="w-20">P_BULL</div>
                  <div className="flex-1 bg-[#1f2937] rounded h-3 relative overflow-hidden">
                    <div className="h-full bg-green-700/70" style={{ width: "35%" }}></div>
                  </div>
                  <div className="w-12 text-right text-green-300">0.35</div>
                </div>
                <div className="text-cyan-300 mt-1">→ argmax = <b>BASE</b></div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-1 text-center bg-green-900/30 border border-green-700 rounded-full w-8 h-8 flex items-center justify-center text-green-300 font-bold">3</div>
            <div className="col-span-11 bg-[#0a0e17] border border-gray-800 rounded-lg p-3">
              <div className="text-sm font-semibold text-gray-200 mb-1">Meta 모델 — 신뢰도 평가</div>
              <div className="text-[11px] text-gray-400 mt-2 font-mono">
                <div className="flex items-center gap-3">
                  <div className="w-32">meta_confidence</div>
                  <div className="flex-1 bg-[#1f2937] rounded h-3 relative overflow-hidden">
                    <div className="h-full bg-green-700/70" style={{ width: "62%" }}></div>
                    <div className="absolute top-0 bottom-0 border-l border-yellow-400" style={{ left: "50%" }}></div>
                  </div>
                  <div className="w-12 text-right text-green-300">0.62</div>
                </div>
                <div className="text-green-300 mt-1">→ 0.62 &gt; 0.5 (threshold) → <b>Primary 신뢰</b></div>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-1 text-center bg-yellow-900/30 border border-yellow-700 rounded-full w-8 h-8 flex items-center justify-center text-yellow-300 font-bold">4</div>
            <div className="col-span-11 bg-[#0a0e17] border-2 border-yellow-700/50 rounded-lg p-3">
              <div className="text-sm font-semibold text-gray-200 mb-2">최종 자산배분 — BASE grid 적용</div>
              <div className="grid grid-cols-3 gap-3 text-center">
                <div className="bg-[#111827] rounded p-3">
                  <div className="text-3xl font-bold text-green-300">75%</div>
                  <div className="text-[11px] text-gray-400 mt-1">주식 (ACWI)</div>
                </div>
                <div className="bg-[#111827] rounded p-3">
                  <div className="text-3xl font-bold text-blue-300">10%</div>
                  <div className="text-[11px] text-gray-400 mt-1">채권 (Global Agg)</div>
                </div>
                <div className="bg-[#111827] rounded p-3">
                  <div className="text-3xl font-bold text-yellow-300">15%</div>
                  <div className="text-[11px] text-gray-400 mt-1">현금 (T-bill)</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── 7. Training & CV ── */}
      <div className="bg-[#111827] rounded-lg p-5 border border-gray-800">
        <div className="text-base font-semibold text-gray-100 mb-1">
          8. 모델은 어떻게 학습되나? (간단)
        </div>
        <div className="text-xs text-gray-500 mb-4">
          과거 데이터만 사용해 학습 → 미래 데이터에서 검증 (lookahead 차단).
        </div>
        <div className="grid grid-cols-2 gap-4 text-xs text-gray-300 leading-relaxed">
          <div className="bg-[#0a0e17] border border-gray-800 rounded p-3">
            <div className="text-cyan-300 font-semibold mb-2">📚 학습 표본</div>
            <ul className="list-disc pl-4 space-y-1 text-[11px]">
              <li>2007년 7월부터 매월 말 데이터 사용</li>
              <li>각 월마다: 입력 feature(t시점) + 정답 라벨(t+1개월 후 결과)</li>
              <li>총 ~225개월 (2007-07 ~ 현재)</li>
            </ul>
          </div>
          <div className="bg-[#0a0e17] border border-gray-800 rounded p-3">
            <div className="text-cyan-300 font-semibold mb-2">🛡️ 검증 방식 (Walk-Forward)</div>
            <ul className="list-disc pl-4 space-y-1 text-[11px]">
              <li>모델은 항상 <b>과거 데이터만</b> 학습 (미래 정보 금지)</li>
              <li>21일 embargo: train/test 경계에서 라벨 누설 방지</li>
              <li>매 fold에서 모델 새로 학습 (expanding window)</li>
              <li>P0~P3 OOS: 2013-08~ / P4 OOS: 2018-07~ (meta burn-in)</li>
            </ul>
          </div>
        </div>
        <div className="mt-3 text-[10.5px] text-gray-500 italic">
          이 방법론은 Lopez de Prado <i>Advances in Financial Machine Learning</i> Ch.7 (Cross-Validation in Finance)
          의 표준을 따릅니다 — 일반 K-Fold는 시계열에서 future leakage 발생, Purged Walk-Forward는 이를 차단.
        </div>
      </div>

      {/* ── 8. Summary ── */}
      <div className="bg-gradient-to-br from-cyan-900/20 to-[#0a0e17] border-2 border-cyan-700/50 rounded-lg p-5">
        <div className="text-base font-semibold text-cyan-100 mb-3">📌 핵심 요약</div>
        <ul className="list-disc pl-6 space-y-1.5 text-sm text-gray-300 leading-relaxed">
          <li>매월 말 시장 데이터를 보고 ML이 <b>BULL / BASE / BEAR</b> 셋 중 하나로 분류</li>
          <li>각 상태에 미리 정해진 <b>주식/채권/현금 비율</b> 적용 (BULL 90/5/5, BASE 75/10/15, BEAR 60/15/25)</li>
          <li>P0~P3는 <b>확률 가중 평균</b> 방식 — 부드러운 변화</li>
          <li><b>P4 (winner)</b>는 <b>Hard Switch + Meta Filter</b> — 신뢰 낮으면 BASE로 안전 후퇴</li>
          <li>모든 학습은 <b>과거 데이터만 사용</b>, 21일 embargo로 누설 차단 (Lopez de Prado 표준)</li>
          <li>P4가 최고 성과: Sharpe <b>0.86</b>, Max DD <b>−16.5%</b>, Turnover <b>3.5%/월</b></li>
        </ul>
      </div>
    </div>
  );
}

// ─── Helper card for model explanation ─────────────────────────────────────
function ModelExplainCard({
  tag, tagColor, title, features, featureExamples, decision, insight, isWinner = false,
}: {
  tag: string; tagColor: string; title: string;
  features: string; featureExamples: string;
  decision: string; insight: string;
  isWinner?: boolean;
}) {
  return (
    <div className={`bg-[#0a0e17] rounded-lg p-3 grid grid-cols-12 gap-3 items-start ${
      isWinner ? "border-2 border-green-700/60 ring-1 ring-green-700/30" : "border border-gray-800"
    }`}>
      <div className="col-span-1 flex flex-col items-center justify-center pt-1">
        <span className="text-2xl font-bold" style={{ color: tagColor }}>{tag}</span>
        {isWinner && <span className="text-[9px] text-green-400 font-semibold mt-0.5">WINNER</span>}
      </div>
      <div className="col-span-11">
        <div className="text-sm font-semibold text-gray-100">{title}</div>
        <div className="grid grid-cols-3 gap-3 mt-2">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-0.5">Features</div>
            <div className="text-[11px] text-gray-300 font-medium">{features}</div>
            <div className="text-[10px] text-gray-500 mt-0.5">{featureExamples}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-0.5">의사결정</div>
            <div className="text-[11px] text-gray-300">{decision}</div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-gray-500 mb-0.5">핵심 차이</div>
            <div className="text-[11px] text-gray-400 italic leading-snug">{insight}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Features Detail Section — explains each feature used by P0~P4
// ─────────────────────────────────────────────────────────────────────────────

interface FeatureSpec {
  name: string;
  ko: string;
  formula: string;
  intuition: string;
  signal: "bull" | "bear" | "neutral";
}

const BASELINE_FEATURE_GROUPS: { title: string; emoji: string; items: FeatureSpec[] }[] = [
  {
    title: "Volatility (변동성 / 공포 지표)", emoji: "⚡",
    items: [
      { name: "vix_level",  ko: "VIX 절대 수준", formula: "VIX (월말 종가)",
        intuition: "주식 옵션 implied volatility. 시장의 공포 게이지.",
        signal: "bear" },
      { name: "vix_z252",   ko: "VIX 252일 z-score", formula: "(VIX − rolling mean) / rolling std",
        intuition: "최근 1년 평균 대비 얼마나 비정상적인 수준인가. mean reversion 신호.",
        signal: "bear" },
      { name: "vix_chg_1m", ko: "VIX 1개월 변화", formula: "VIX(t) − VIX(t−21)",
        intuition: "급격한 vol 상승은 단기 충격 발생 신호.",
        signal: "bear" },
    ],
  },
  {
    title: "Rates / Yield Curve (금리 / 수익률 곡선)", emoji: "💰",
    items: [
      { name: "tnx_level",   ko: "10년 국채 수익률", formula: "^TNX (CBOE 10Y yield)",
        intuition: "기준 금리 환경. 높을수록 채권 매력↓, 주식 valuation 부담↑.",
        signal: "neutral" },
      { name: "slope_5s10s", ko: "5y-10y 수익률 곡선", formula: "TNX − FVX",
        intuition: "장단기 금리 spread. 음수면 inverted curve (경기 침체 시그널).",
        signal: "bull" },
      { name: "tnx_chg_1m",  ko: "10년 금리 1개월 변화", formula: "TNX(t) − TNX(t−21)",
        intuition: "급격한 금리 상승 = risk-off. 채권 매도 압력.",
        signal: "bear" },
    ],
  },
  {
    title: "FX / Commodity (환율 / 원자재)", emoji: "🌍",
    items: [
      { name: "dxy_ret_3m",         ko: "달러 인덱스 3M 수익률", formula: "DXY(t)/DXY(t−63) − 1",
        intuition: "달러 강세 = global liquidity 위축 = 신흥국·원자재 약세.",
        signal: "bear" },
      { name: "copper_gold_ratio", ko: "구리/금 비율", formula: "Copper / Gold",
        intuition: "성장 민감(구리) vs 안전자산(금). 높을수록 risk-on.",
        signal: "bull" },
      { name: "gold_ret_3m",        ko: "금 3M 수익률", formula: "Gold(t)/Gold(t−63) − 1",
        intuition: "금 상승 = 인플레이션 헷지 또는 위험회피 수요.",
        signal: "bear" },
      { name: "hyg_lqd_diff_3m",    ko: "HY−IG 3M 수익률 차", formula: "HYG_3M_ret − LQD_3M_ret",
        intuition: "HY가 IG 앞서면 risk-on, 뒤지면 credit stress.",
        signal: "bull" },
    ],
  },
  {
    title: "Equity Own-Trend (주식 자체 모멘텀 / 추세)", emoji: "📈",
    items: [
      { name: "acwi_ret_1m",     ko: "ACWI 1개월 수익률", formula: "ACWI(t)/ACWI(t−21) − 1",
        intuition: "최근 1개월 모멘텀. 단기 트렌드.",
        signal: "bull" },
      { name: "acwi_ret_12_1m",  ko: "Jegadeesh-Titman 12-1 모멘텀", formula: "ACWI(t−21) / ACWI(t−252) − 1",
        intuition: "장기 모멘텀의 정석. 최근 1개월 제외 (mean reversion 노이즈 회피).",
        signal: "bull" },
      { name: "acwi_vs_sma200",  ko: "ACWI vs 200일 이평선 거리", formula: "ACWI / SMA200 − 1",
        intuition: "장기 추세. 양수면 bull market, 음수면 bear market.",
        signal: "bull" },
      { name: "acwi_rvol_21d",   ko: "ACWI 21일 실현 변동성", formula: "std(daily log returns) × √252",
        intuition: "최근 21일 실제 변동성. 높을수록 stress.",
        signal: "bear" },
    ],
  },
  {
    title: "Bond Own-Trend (채권 추세)", emoji: "📊",
    items: [
      { name: "agg_vs_sma200", ko: "Global Agg vs 200일 이평선", formula: "AGG / SMA200 − 1",
        intuition: "채권 추세. 음수면 금리 상승기 (채권 약세).",
        signal: "neutral" },
    ],
  },
];

const P3_FEATURE_GROUP: FeatureSpec[] = [
  { name: "vix_move_ratio",      ko: "VIX / MOVE 비율", formula: "VIX / MOVE",
    intuition: "주식 vol vs 채권 vol. 높을수록 주식 stress가 채권보다 큼.",
    signal: "bear" },
  { name: "vix_vxv_ratio",       ko: "VIX / VXV term structure", formula: "VIX(1M) / VIX3M(3M)",
    intuition: ">1이면 backwardation (단기가 장기보다 큼) = 즉각적 stress.",
    signal: "bear" },
  { name: "wti_copper_ratio",    ko: "원유 / 구리 log 비율", formula: "log(WTI / Copper)",
    intuition: "원유(인플레 민감) vs 구리(성장 민감). 인플레 vs 성장 dominance.",
    signal: "neutral" },
  { name: "dxy_z252",            ko: "달러 인덱스 252일 z-score", formula: "(DXY − rolling mean) / rolling std",
    intuition: "달러 강세 정도의 정규화. mean reversion 신호.",
    signal: "bear" },
  { name: "hy_ig_spread_level", ko: "HY/IG 가격 비율 (수준)", formula: "HYG / LQD (price)",
    intuition: "HY 회사채 vs IG 회사채. 낮을수록 credit spread 확대 (위기 시그널).",
    signal: "bull" },
  { name: "tip_ief_ratio",       ko: "TIP / IEF 비율 (인플레 breakeven proxy)", formula: "TIP / IEF",
    intuition: "물가연동채 vs 일반국채. 인플레이션 기대치 변화.",
    signal: "neutral" },
];

const BREADTH_FEATURE_GROUP: FeatureSpec[] = [
  { name: "eq_pct_bullish",   ko: "주식 ETF 중 강세 분류 비율", formula: "(CONTINUATION + RECOVERY + FORMATION + LAGGING_CATCHUP) / 전체",
    intuition: "price_discovery가 \"강세 추세\"로 분류한 주식 ETF의 비율.",
    signal: "bull" },
  { name: "eq_pct_downtrend", ko: "주식 ETF 중 하락 추세 비율", formula: "(DOWNTREND + CYCLE_PEAK + FADING) / 전체",
    intuition: "price_discovery가 \"하락 추세\"로 분류한 주식 ETF의 비율.",
    signal: "bear" },
  { name: "eq_tcs_median",    ko: "주식 ETF TCS 중앙값", formula: "median(TCS score across equity ETFs)",
    intuition: "주식 ETF의 평균적인 trend continuation 강도 (0-100).",
    signal: "bull" },
  { name: "eq_rss_std",       ko: "주식 RSS 횡단면 표준편차", formula: "std(RSS percentile across ETFs)",
    intuition: "주식 모멘텀 분산도. 클수록 leader-laggard 분화 (regime change 신호).",
    signal: "neutral" },
  { name: "bd_pct_bullish",   ko: "채권 ETF 중 강세 분류 비율", formula: "(BULLISH classes) / 전체 채권 ETF",
    intuition: "채권 강세 = 금리 하락 기대. risk-off 동조 가능.",
    signal: "neutral" },
  { name: "bd_tcs_median",    ko: "채권 ETF TCS 중앙값", formula: "median(TCS across bond ETFs)",
    intuition: "채권 추세 강도. 높을수록 채권 강세.",
    signal: "neutral" },
];

function FeatureUsageMatrix() {
  const groups = [
    { name: "Baseline (macro)",  count: 15, p0: true,  p1: true,  p2: true,  p3: true,  p4: true },
    { name: "P3 Macro Extended", count: 6,  p0: false, p1: false, p2: false, p3: true,  p4: true },
    { name: "Breadth (price_discovery 집계)", count: 6,  p0: false, p1: true,  p2: false, p3: false, p4: true },
  ];
  const totals = { p0: 15, p1: 21, p2: 15, p3: 21, p4: 27 };
  return (
    <div className="overflow-x-auto border border-gray-800 rounded">
      <table className="w-full text-xs">
        <thead className="bg-[#1f2937]">
          <tr>
            <th className="px-2 py-1.5 text-left text-gray-400 border-b border-gray-700">Feature 그룹</th>
            <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">개수</th>
            <th className="px-2 py-1.5 text-center border-b border-gray-700" style={{ color: VARIANT_COLORS.P0 }}>P0</th>
            <th className="px-2 py-1.5 text-center border-b border-gray-700" style={{ color: VARIANT_COLORS.P1 }}>P1</th>
            <th className="px-2 py-1.5 text-center border-b border-gray-700" style={{ color: VARIANT_COLORS.P2 }}>P2</th>
            <th className="px-2 py-1.5 text-center border-b border-gray-700" style={{ color: VARIANT_COLORS.P3 }}>P3</th>
            <th className="px-2 py-1.5 text-center border-b border-gray-700" style={{ color: VARIANT_COLORS.P4 }}>P4 ★</th>
          </tr>
        </thead>
        <tbody>
          {groups.map((g) => (
            <tr key={g.name} className="border-b border-gray-800/50">
              <td className="px-2 py-1.5 text-gray-200 font-medium">{g.name}</td>
              <td className="px-2 py-1.5 text-right text-gray-300">{g.count}</td>
              {(["p0", "p1", "p2", "p3", "p4"] as const).map((k) => (
                <td key={k} className="px-2 py-1.5 text-center">
                  {g[k] ? <span className="text-green-300 font-bold">✓</span> : <span className="text-gray-600">−</span>}
                </td>
              ))}
            </tr>
          ))}
          <tr className="bg-[#0a0e17] font-semibold">
            <td className="px-2 py-1.5 text-gray-100">Total features</td>
            <td className="px-2 py-1.5 text-right text-gray-100">27</td>
            {(["p0", "p1", "p2", "p3", "p4"] as const).map((k) => (
              <td key={k} className="px-2 py-1.5 text-center" style={{ color: VARIANT_COLORS[k.toUpperCase()] }}>
                {totals[k]}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function FeatureCard({ f }: { f: FeatureSpec }) {
  const dirColor = f.signal === "bull" ? "text-green-300 bg-green-900/20"
                 : f.signal === "bear" ? "text-red-300 bg-red-900/20"
                 : "text-gray-400 bg-gray-800/40";
  const dirText = f.signal === "bull" ? "↑ Bull 시그널"
                : f.signal === "bear" ? "↓ Bear 시그널"
                : "≈ Neutral / 양면";
  return (
    <div className="bg-[#0a0e17] border border-gray-800 rounded-lg p-3 hover:border-gray-700 transition-colors">
      <div className="flex items-baseline justify-between mb-1.5 gap-2 flex-wrap">
        <code className="text-cyan-300 text-[12px] font-bold">{f.name}</code>
        <span className={`text-[9.5px] px-1.5 py-0.5 rounded font-semibold ${dirColor}`}>{dirText}</span>
      </div>
      <div className="text-[12.5px] text-gray-200 font-medium mb-1">{f.ko}</div>
      <div className="text-[10.5px] text-gray-500 font-mono mb-1">계산: {f.formula}</div>
      <div className="text-[11px] text-gray-400 leading-snug">{f.intuition}</div>
    </div>
  );
}

function FeaturesDetailSection() {
  return (
    <div className="bg-[#111827] rounded-lg p-5 border border-gray-800 space-y-5">
      <div>
        <div className="text-base font-semibold text-gray-100 mb-1">
          5. 사용된 Feature 상세
        </div>
        <div className="text-xs text-gray-500">
          모든 모델이 같은 출력(BULL/BASE/BEAR 확률)을 만들지만, <b>입력 feature는 모델마다 다릅니다</b>.
          각 feature가 무엇을 의미하고 어떤 신호를 잡는지 정리합니다.
        </div>
      </div>

      {/* Usage matrix */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-2">📊 Feature 그룹 사용 매트릭스</div>
        <FeatureUsageMatrix />
      </div>

      {/* Group A: Baseline */}
      <div>
        <div className="flex items-baseline justify-between mb-1">
          <div className="text-sm font-semibold text-gray-200">
            <span className="text-cyan-300">A.</span> Baseline Macro Features
            <span className="text-xs text-gray-500 ml-2">— 15개 · 모든 모델(P0~P4) 공통 사용</span>
          </div>
        </div>
        <div className="text-[11px] text-gray-500 mb-3 italic">
          AQR / Bridgewater 등 글로벌 자산운용사가 standard로 쓰는 macro / cross-asset signals.
        </div>
        <div className="space-y-3">
          {BASELINE_FEATURE_GROUPS.map((g) => (
            <div key={g.title}>
              <div className="text-[12px] font-semibold text-gray-300 mb-1.5 flex items-baseline gap-2">
                <span className="text-base">{g.emoji}</span>
                <span>{g.title}</span>
                <span className="text-[10px] text-gray-500">({g.items.length}개)</span>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-2.5">
                {g.items.map((f) => <FeatureCard key={f.name} f={f} />)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Group B: P3 Macro */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-1">
          <span className="text-blue-300">B.</span> P3 Macro Extended (cross-asset / term structure)
          <span className="text-xs text-gray-500 ml-2">— 6개 · P3, P4 사용</span>
        </div>
        <div className="text-[11px] text-gray-500 mb-3 italic">
          기본 macro feature를 보완하는 cross-asset 비율 / term-structure 지표. 시장 stress의 다층적 신호 포착.
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2.5">
          {P3_FEATURE_GROUP.map((f) => <FeatureCard key={f.name} f={f} />)}
        </div>
      </div>

      {/* Group C: Breadth */}
      <div>
        <div className="text-sm font-semibold text-gray-200 mb-1">
          <span className="text-orange-300">C.</span> Breadth Features (bottom-up from price_discovery)
          <span className="text-xs text-gray-500 ml-2">— 6개 · P1, P4 사용</span>
        </div>
        <div className="text-[11px] text-gray-500 mb-3 italic">
          매월 말 170개 글로벌 ETF에 대해 price_discovery scanner를 historical replay 후 cross-sectional 집계.
          BlackRock SAE의 "bottom-up signal aggregation" 철학.
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-3 gap-2.5">
          {BREADTH_FEATURE_GROUP.map((f) => <FeatureCard key={f.name} f={f} />)}
        </div>
      </div>

      {/* Footer note */}
      <div className="bg-[#0a0e17] border border-cyan-900/40 rounded-lg p-3 text-[11px] text-cyan-200/90 leading-relaxed">
        <b className="text-cyan-100">📝 참고</b> —&nbsp;
        <span className="text-gray-300">
          <b>"신호 방향"</b> 라벨(↑Bull / ↓Bear / ≈Neutral)은 단변량 직관일 뿐, 실제 ML 모델은
          27개 feature의 <b>비선형 상호작용</b>을 학습합니다 (LightGBM gradient boosting).
          예컨대 'VIX 높음 + 12M momentum 강함'은 단일 신호로 보면 충돌이지만,
          모델은 "전환점 직전 short squeeze 가능성" 같은 패턴으로 해석할 수 있음.
          또한 daily-resolution path-dependent feature 10개도 정의되어 있으나
          (vix_max_in_month, acwi_max_dd_in_month 등) Plan C 검증 결과 P4 메타와 충돌해 현재 비활성.
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Win Ratio Section — per-signal precision + directional hit rate (P0~P4)
// ─────────────────────────────────────────────────────────────────────────────
function WinRatioSection({ wr }: { wr: WinRatioData }) {
  const order = ["P0", "P1", "P2", "P3", "P4"];
  const regimes = wr.regimes;
  const baseRange = wr.base_range;

  // Cell color helper for win-ratio scale
  const cellShade = (v: number | null | undefined) => {
    if (v == null || !Number.isFinite(v)) return "text-gray-500";
    if (v >= 0.65) return "text-green-300 font-semibold";
    if (v >= 0.50) return "text-cyan-300";
    if (v >= 0.35) return "text-yellow-300";
    return "text-red-300";
  };

  const fmtPct0 = (v: number | null | undefined) => {
    if (v == null || !Number.isFinite(v)) return "—";
    return `${(v * 100).toFixed(0)}%`;
  };
  const fmtPctSig = (v: number | null | undefined) => {
    if (v == null || !Number.isFinite(v)) return "—";
    const s = (v * 100).toFixed(2);
    return v >= 0 ? `+${s}%` : `${s}%`;
  };

  // Helper: per-regime row from variant
  const getRow = (variantTag: string, regime: string): WinRatioRow | null => {
    const v = wr.variants[variantTag];
    if (!v) return null;
    return v.per_regime.find((r) => r.regime === regime) ?? null;
  };

  // Build heatmap matrices for precision + directional
  const buildMatrix = (key: "precision" | "directional") =>
    order.map((tag) =>
      regimes.map((r) => {
        const row = getRow(tag, r);
        return row && row.n > 0 ? (row[key] ?? null) : null;
      })
    );

  const precisionMatrix = buildMatrix("precision");
  const directionalMatrix = buildMatrix("directional");

  return (
    <div className="bg-[#111827] rounded-lg p-5 border border-gray-800 space-y-5">
      <div>
        <div className="text-base font-semibold text-gray-100">
          Per-Signal Win Ratio — P0~P4 × BEAR / BASE / BULL
        </div>
        <div className="text-xs text-gray-500 mt-1">
          각 모델이 특정 시그널을 출력한 시점 기준으로 두 가지 win ratio 계산:&nbsp;
          <span className="text-cyan-300">Precision</span> = 분류 정확도 (예측이 실제 regime과 일치),&nbsp;
          <span className="text-cyan-300">Directional</span> = 방향성 베팅 성공률.
        </div>
      </div>

      {/* Definition cards */}
      <div className="grid grid-cols-3 gap-3">
        <div className="bg-[#0a0e17] border border-red-800/40 rounded p-2 text-[11px]">
          <div className="text-red-300 font-semibold mb-1">🔴 BEAR signal win</div>
          <div className="text-gray-300">fwd_ret &lt; 0 — <span className="text-gray-500 italic">시장이 실제로 하락 (방어적 베팅 성공)</span></div>
        </div>
        <div className="bg-[#0a0e17] border border-yellow-800/40 rounded p-2 text-[11px]">
          <div className="text-yellow-300 font-semibold mb-1">🟡 BASE signal win</div>
          <div className="text-gray-300">|fwd_ret| &lt; {(baseRange * 100).toFixed(0)}% — <span className="text-gray-500 italic">중립 구간에 안착 (큰 움직임 없음)</span></div>
        </div>
        <div className="bg-[#0a0e17] border border-green-800/40 rounded p-2 text-[11px]">
          <div className="text-green-300 font-semibold mb-1">🟢 BULL signal win</div>
          <div className="text-gray-300">fwd_ret &gt; 0 — <span className="text-gray-500 italic">시장이 실제로 상승 (공격적 베팅 성공)</span></div>
        </div>
      </div>

      {/* Heatmap row: precision + directional */}
      <div className="grid grid-cols-2 gap-4">
        {/* Precision heatmap */}
        <div>
          <div className="text-sm font-semibold text-gray-200 mb-1">
            Classification Precision (예측 = 실제 regime)
          </div>
          <Plot
            data={[{
              type: "heatmap",
              z: precisionMatrix.map((row) => row.map((v) => v ?? NaN)),
              x: regimes,
              y: order,
              colorscale: [
                [0.0, "#7f1d1d"], [0.35, "#dc2626"], [0.5, "#facc15"],
                [0.65, "#06b6d4"], [1.0, "#16a34a"],
              ],
              zmin: 0, zmax: 1,
              text: precisionMatrix.map((row, i) =>
                row.map((v, j) => {
                  const r = getRow(order[i], regimes[j]);
                  if (!r || r.n === 0) return "—";
                  return `${((v ?? 0) * 100).toFixed(0)}%\nN=${r.n}`;
                })
              ),
              texttemplate: "%{text}",
              textfont: { size: 11, color: "white" },
              hovertemplate: "<b>%{y} → %{x}</b><br>Precision: %{z:.1%}<extra></extra>",
              showscale: false,
            } as any]}
            layout={{
              ...DARK_LAYOUT, height: 240,
              margin: { t: 15, b: 30, l: 35, r: 10 },
              yaxis: { autorange: "reversed" as any },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="text-[10px] text-gray-500 mt-1">
            모델이 특정 regime을 예측했을 때 실제도 그 regime이었던 비율 (분류 정확도).
          </div>
        </div>

        {/* Directional heatmap */}
        <div>
          <div className="text-sm font-semibold text-gray-200 mb-1">
            Directional Hit Rate (방향성 베팅 정확도)
          </div>
          <Plot
            data={[{
              type: "heatmap",
              z: directionalMatrix.map((row) => row.map((v) => v ?? NaN)),
              x: regimes,
              y: order,
              colorscale: [
                [0.0, "#7f1d1d"], [0.35, "#dc2626"], [0.5, "#facc15"],
                [0.65, "#06b6d4"], [1.0, "#16a34a"],
              ],
              zmin: 0, zmax: 1,
              text: directionalMatrix.map((row, i) =>
                row.map((v, j) => {
                  const r = getRow(order[i], regimes[j]);
                  if (!r || r.n === 0) return "—";
                  return `${((v ?? 0) * 100).toFixed(0)}%\nN=${r.n}`;
                })
              ),
              texttemplate: "%{text}",
              textfont: { size: 11, color: "white" },
              hovertemplate: "<b>%{y} → %{x}</b><br>Directional: %{z:.1%}<extra></extra>",
              showscale: false,
            } as any]}
            layout={{
              ...DARK_LAYOUT, height: 240,
              margin: { t: 15, b: 30, l: 35, r: 10 },
              yaxis: { autorange: "reversed" as any },
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%" }}
          />
          <div className="text-[10px] text-gray-500 mt-1">
            BEAR: fwd&lt;0 / BASE: |fwd|&lt;{(baseRange * 100).toFixed(0)}% / BULL: fwd&gt;0 비율.
          </div>
        </div>
      </div>

      {/* Detailed table — 3 sub-tables side-by-side, one per regime */}
      <div className="space-y-3">
        <div className="text-sm font-semibold text-gray-200">상세 표 — Variant별 시그널 발화 횟수와 정확도</div>
        <div className="grid grid-cols-3 gap-3">
          {regimes.map((regime) => {
            const color = regime === "BEAR" ? "red" : regime === "BULL" ? "green" : "yellow";
            const colorClass =
              regime === "BEAR" ? "text-red-300 border-red-800/40"
              : regime === "BULL" ? "text-green-300 border-green-800/40"
              : "text-yellow-300 border-yellow-800/40";
            return (
              <div key={regime} className={`bg-[#0a0e17] border ${colorClass.split(" ")[1]} rounded-lg overflow-hidden`}>
                <div className={`px-2 py-1.5 bg-[#111827] border-b ${colorClass.split(" ")[1]}`}>
                  <span className={`font-bold text-sm ${colorClass.split(" ")[0]}`}>
                    {regime === "BEAR" && "🔴"} {regime === "BASE" && "🟡"} {regime === "BULL" && "🟢"} {regime} signal
                  </span>
                </div>
                <table className="w-full text-[11px]">
                  <thead className="bg-[#111827]">
                    <tr>
                      <th className="px-1.5 py-1 text-left text-gray-400">Variant</th>
                      <th className="px-1.5 py-1 text-right text-gray-400">N</th>
                      <th className="px-1.5 py-1 text-right text-gray-400">Prec.</th>
                      <th className="px-1.5 py-1 text-right text-gray-400">Dir.</th>
                      <th className="px-1.5 py-1 text-right text-gray-400">AvgRet</th>
                    </tr>
                  </thead>
                  <tbody>
                    {order.map((tag) => {
                      const row = getRow(tag, regime);
                      const isP4 = tag === "P4";
                      return (
                        <tr key={tag}
                            className={`border-b border-gray-800/30 ${isP4 ? "bg-green-900/15" : ""}`}>
                          <td className="px-1.5 py-1 font-semibold" style={{ color: VARIANT_COLORS[tag] }}>
                            {tag}{isP4 && " ★"}
                          </td>
                          <td className="px-1.5 py-1 text-right text-gray-300">{row?.n ?? 0}</td>
                          <td className={`px-1.5 py-1 text-right ${cellShade(row?.precision)}`}>
                            {fmtPct0(row?.precision)}
                          </td>
                          <td className={`px-1.5 py-1 text-right ${cellShade(row?.directional)}`}>
                            {fmtPct0(row?.directional)}
                          </td>
                          <td className={`px-1.5 py-1 text-right ${
                            row?.fwd_ret_mean != null
                              ? (regime === "BEAR" ? (row.fwd_ret_mean < 0 ? "text-green-400" : "text-red-400")
                                : regime === "BULL" ? (row.fwd_ret_mean > 0 ? "text-green-400" : "text-red-400")
                                : "text-gray-300")
                              : "text-gray-500"
                          }`}>
                            {fmtPctSig(row?.fwd_ret_mean)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      </div>

      {/* Key insights */}
      <div className="bg-[#0a0e17] border border-cyan-900/40 rounded-lg p-3 text-[11px] text-cyan-200/90 leading-relaxed">
        <b className="text-cyan-100">📌 해석 가이드</b> —&nbsp;
        <span className="text-gray-300">
          <b>Precision &gt; 50%</b>이면 단순 random보다 분류 우수.
          <b> Directional &gt; 50%</b>이면 방향성 베팅이 평균 이상.&nbsp;
          <b className="text-green-300">P4 BULL 시그널의 directional 100%</b> (n=5): 발화는 적지만 한번 발화하면 다음 달 ACWI가 100% 상승 — 메타 필터의 selectivity가 만들어낸 "high-conviction signal".&nbsp;
          <b>BEAR 시그널의 precision 높지만 directional 낮음</b>: 실제 BEAR regime이라도 ACWI가 항상 음수 수익은 아님 (라벨에 VIX&gt;30 조건 포함).&nbsp;
          <b>색상 스케일</b>: <span className="text-green-300">≥65%</span> · <span className="text-cyan-300">50-65%</span> · <span className="text-yellow-300">35-50%</span> · <span className="text-red-300">&lt;35%</span>.
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Meta-Labeling Deep Dive — 6 sub-sections explaining the 2-stage ML architecture
// References: López de Prado, Advances in Financial Machine Learning, Ch.3
// ─────────────────────────────────────────────────────────────────────────────
function MetaLabelingDeepDive() {
  return (
    <div className="bg-[#111827] rounded-lg p-5 border border-gray-800 space-y-5">
      <div>
        <div className="text-base font-semibold text-gray-100">
          6. Meta-Labeling 구조 자세히 (Deep Dive)
        </div>
        <div className="text-xs text-gray-500 mt-1">
          P4 winner의 핵심 2단계 ML 아키텍처. López de Prado <i>Advances in Financial Machine Learning</i> Ch.3 표준 구현.
          단순 multiclass + argmax 대비 이 구조가 왜·어떻게 더 나은지 단계별로 설명합니다.
        </div>
      </div>

      {/* 6.1 The Problem Meta-Labeling Solves */}
      <div className="bg-[#0a0e17] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-bold text-cyan-300 mb-3">6.1 왜 메타-라벨링인가? — 해결하는 문제</div>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-red-900/15 border border-red-800/40 rounded p-3">
            <div className="text-[12px] font-bold text-red-300 mb-1">⚠️ 단일 모델의 문제</div>
            <div className="text-[11px] text-gray-300 leading-relaxed">
              일반 multiclass classifier는 한 번에 두 가지를 동시에 학습:
              <ul className="list-disc pl-4 mt-1 space-y-0.5">
                <li><b>방향(side)</b>: BULL이냐 BASE냐 BEAR냐?</li>
                <li><b>크기(size)</b>: 그 예측을 얼마나 확신하나?</li>
              </ul>
              <div className="mt-2">
                ML이 두 마리 토끼를 동시에 잡으려다 <b>둘 다 mediocre</b>해지는 경향. 특히 BULL 같은 minority class에서 노이즈 많은 예측 다수 발생.
              </div>
            </div>
          </div>
          <div className="bg-green-900/15 border border-green-700/40 rounded p-3">
            <div className="text-[12px] font-bold text-green-300 mb-1">✓ 메타-라벨링 해결책</div>
            <div className="text-[11px] text-gray-300 leading-relaxed">
              한 모델에 모든 부담을 주지 말고 <b>두 모델로 분리</b>:
              <ul className="list-disc pl-4 mt-1 space-y-0.5">
                <li><b>Primary</b>: 방향만 판단 ("BULL/BASE/BEAR 중 뭐?")</li>
                <li><b>Meta</b>: 신뢰도만 판단 ("Primary 예측이 맞을까?")</li>
              </ul>
              <div className="mt-2">
                각 모델이 각자 단일 임무에 특화 → <b>precision 향상, recall 거의 유지</b>.
                불확실한 시점에는 안전 자산배분(BASE)으로 후퇴 = 자연스러운 <b>position sizing</b>.
              </div>
            </div>
          </div>
        </div>
        <div className="mt-3 text-[10.5px] text-gray-500 italic">
          ※ AQR이 trend-following 전략, BlackRock이 SAE alpha 신호에서 사용하는 standard pattern.
          보험·신용평가에서도 동일 구조 (1차 점수 → 2차 confidence filter).
        </div>
      </div>

      {/* 6.2 Architecture Flow Diagram */}
      <div className="bg-[#0a0e17] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-bold text-cyan-300 mb-3">6.2 전체 아키텍처 — 데이터가 흐르는 모습</div>
        <div className="space-y-2">
          {/* Input row */}
          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-2 text-right text-[11px] text-gray-400 font-semibold">입력 데이터</div>
            <div className="col-span-10 bg-[#111827] border border-cyan-900/40 rounded p-2 text-[11px] text-gray-300 font-mono">
              📊 27 features (15 baseline macro + 6 P3 macro + 6 breadth)
            </div>
          </div>
          <div className="grid grid-cols-12 gap-2"><div className="col-span-2"></div><div className="col-span-10 text-center text-gray-600 text-lg">↓</div></div>

          {/* Primary stage */}
          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-2 text-right text-[11px] text-cyan-400 font-bold">① Primary</div>
            <div className="col-span-10 bg-[#111827] border-2 border-cyan-700/60 rounded p-3">
              <div className="text-[11px] text-gray-200 mb-1">
                <b>모델</b>: LightGBM Multiclass (3 classes: BEAR/BASE/BULL)
              </div>
              <div className="text-[11px] text-gray-200 mb-1">
                <b>학습</b>: <code className="text-cyan-300">class_weight={"{BULL: 3.0}"}</code> (소수 클래스 보정), <code className="text-cyan-300">bull_threshold=0.25</code>
              </div>
              <div className="text-[11px] text-gray-200">
                <b>출력</b>: <code className="text-cyan-300">[P_BEAR, P_BASE, P_BULL]</code> 확률 + argmax 예측 regime
              </div>
            </div>
          </div>
          <div className="grid grid-cols-12 gap-2"><div className="col-span-2"></div><div className="col-span-10 text-center text-gray-600 text-lg">↓</div></div>

          {/* Meta stage */}
          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-2 text-right text-[11px] text-green-400 font-bold">② Meta</div>
            <div className="col-span-10 bg-[#111827] border-2 border-green-700/60 rounded p-3">
              <div className="text-[11px] text-gray-200 mb-1">
                <b>모델</b>: LightGBM Binary (1 = primary correct, 0 = primary wrong)
              </div>
              <div className="text-[11px] text-gray-200 mb-1">
                <b>입력</b>: 27 features (daily-resolution 10개 제외) + Primary의 3개 확률 + 확률 분포 entropy = <b>21개 차원</b>
              </div>
              <div className="text-[11px] text-gray-200">
                <b>출력</b>: <code className="text-green-300">meta_confidence ∈ [0, 1]</code>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-12 gap-2"><div className="col-span-2"></div><div className="col-span-10 text-center text-gray-600 text-lg">↓</div></div>

          {/* Decision */}
          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-2 text-right text-[11px] text-yellow-400 font-bold">③ Decision</div>
            <div className="col-span-10 bg-[#111827] border-2 border-yellow-700/60 rounded p-3">
              <div className="text-[11px] text-gray-200 font-mono">
                <span className="text-yellow-300">if</span> meta_confidence &gt; 0.5: → ALLOCATION[primary regime]<br/>
                <span className="text-yellow-300">else</span>: → ALLOCATION["BASE"] &nbsp;(안전 후퇴)
              </div>
            </div>
          </div>
          <div className="grid grid-cols-12 gap-2"><div className="col-span-2"></div><div className="col-span-10 text-center text-gray-600 text-lg">↓</div></div>

          {/* Output */}
          <div className="grid grid-cols-12 gap-2 items-center">
            <div className="col-span-2 text-right text-[11px] text-gray-400 font-semibold">최종 출력</div>
            <div className="col-span-10 bg-[#111827] border border-green-900/40 rounded p-2 text-[11px] text-gray-300 font-mono">
              💼 (w_equity, w_bond, w_cash) — 그 달의 자산배분 비율
            </div>
          </div>
        </div>
      </div>

      {/* 6.3 Training Procedure (5 steps) */}
      <div className="bg-[#0a0e17] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-bold text-cyan-300 mb-1">6.3 학습 절차 — 5단계</div>
        <div className="text-[11px] text-gray-500 mb-3">
          Train과 Test 모두 같은 Purged Walk-Forward CV로 진행. 누설(leakage) 차단이 핵심.
        </div>
        <div className="space-y-2">
          {[
            {
              n: 1, color: C.cyan,
              title: "Primary multiclass 학습",
              detail: "27 features + multiclass (BEAR/BASE/BULL) target → LightGBM 학습. Purged Walk-Forward CV (5 folds, 21-day embargo)로 OOS proba 산출."
            },
            {
              n: 2, color: C.cyan,
              title: "Primary OOS 예측 수집",
              detail: "각 fold에서 train 데이터로만 학습된 primary 모델이 test 데이터에 대해 [P_BEAR, P_BASE, P_BULL] 산출. 절대 future leakage 없음."
            },
            {
              n: 3, color: C.green,
              title: "Meta 라벨 생성",
              detail: <span><code className="text-green-300">y_meta = 1 if primary_argmax == true_regime else 0</code><br/>즉, "primary가 정답을 맞췄나?"의 binary label.</span>
            },
            {
              n: 4, color: C.green,
              title: "Meta 입력 feature 구성",
              detail: <span>
                ① 원본 27 features 중 daily-resolution 10개 제외 (overfitting 방지) → 17개<br/>
                ② Primary의 출력 [P_BEAR, P_BASE, P_BULL] 추가 → 20개<br/>
                ③ 확률 분포의 entropy = −Σ P_r log₂(P_r) 추가 → <b>21개 meta features</b>
              </span>
            },
            {
              n: 5, color: C.yellow,
              title: "Meta binary 학습 (동일 CV)",
              detail: "21개 meta features + y_meta target → LightGBM binary classifier. 같은 Purged Walk-Forward (n_splits=5, embargo=1) 사용. min_train=36으로 burn-in 후 OOS meta_confidence 산출."
            },
          ].map((s) => (
            <div key={s.n} className="grid grid-cols-12 gap-2 items-start bg-[#111827] rounded p-2.5">
              <div className="col-span-1 flex items-center justify-center pt-0.5">
                <div className="w-7 h-7 rounded-full border-2 flex items-center justify-center font-bold text-[12px]"
                     style={{ borderColor: s.color, color: s.color }}>{s.n}</div>
              </div>
              <div className="col-span-11">
                <div className="text-[12px] font-semibold text-gray-200">{s.title}</div>
                <div className="text-[11px] text-gray-400 mt-0.5 leading-snug">{s.detail}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 6.4 Implementation Details */}
      <div className="bg-[#0a0e17] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-bold text-cyan-300 mb-3">6.4 구현 디테일 — Hyperparameters</div>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#111827] rounded p-3 border border-cyan-900/40">
            <div className="text-[12px] font-bold text-cyan-300 mb-2">① Primary (Multiclass)</div>
            <table className="w-full text-[10.5px] font-mono">
              <tbody>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">objective</td>
                  <td className="py-1 text-right text-gray-200">multiclass</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">num_class</td>
                  <td className="py-1 text-right text-gray-200">3</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">learning_rate</td>
                  <td className="py-1 text-right text-gray-200">0.03</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">n_estimators</td>
                  <td className="py-1 text-right text-gray-200">400</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">max_depth</td>
                  <td className="py-1 text-right text-gray-200">4</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">num_leaves</td>
                  <td className="py-1 text-right text-gray-200">15</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">reg_alpha / reg_lambda</td>
                  <td className="py-1 text-right text-gray-200">0.5 / 1.0</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">class_weight</td>
                  <td className="py-1 text-right text-cyan-300">{"{BULL: 3.0}"}</td>
                </tr>
                <tr>
                  <td className="py-1 text-gray-400">bull_threshold</td>
                  <td className="py-1 text-right text-cyan-300">0.25</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-[#111827] rounded p-3 border border-green-700/40">
            <div className="text-[12px] font-bold text-green-300 mb-2">② Meta (Binary)</div>
            <table className="w-full text-[10.5px] font-mono">
              <tbody>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">objective</td>
                  <td className="py-1 text-right text-gray-200">binary</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">learning_rate</td>
                  <td className="py-1 text-right text-gray-200">0.03</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">n_estimators</td>
                  <td className="py-1 text-right text-gray-200">300</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">max_depth</td>
                  <td className="py-1 text-right text-gray-200">4</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">num_leaves</td>
                  <td className="py-1 text-right text-gray-200">15</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">reg_alpha / reg_lambda</td>
                  <td className="py-1 text-right text-gray-200">0.5 / 1.0</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">class_weight</td>
                  <td className="py-1 text-right text-green-300">'balanced'</td>
                </tr>
                <tr className="border-b border-gray-800/50">
                  <td className="py-1 text-gray-400">decision threshold</td>
                  <td className="py-1 text-right text-green-300">0.5</td>
                </tr>
                <tr>
                  <td className="py-1 text-gray-400">feature exclusion</td>
                  <td className="py-1 text-right text-yellow-300">daily features 제외</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-3 bg-[#111827] border border-yellow-900/40 rounded p-2.5 text-[10.5px] text-gray-300 leading-relaxed">
          <b className="text-yellow-300">⚠️ Daily features 제외 이유</b>:&nbsp;
          Plan A 실험 결과 — 메타에 daily-resolution path features (vix_max_in_month 등) 10개를 포함하면
          P4 Sharpe 0.86 → 0.83으로 <b>감소</b>. 메타는 ~36-month 학습 표본만 보는데 추가 features 부담이 overfit
          유발. 따라서 <code>META_EXCLUDE</code> 상수로 일별 feature를 메타 입력에서 제외.
          (Primary는 daily features 사용 가능하나 현재 Plan C로 dataset에서 비활성.)
        </div>
      </div>

      {/* 6.5 Trade-off Analysis */}
      <div className="bg-[#0a0e17] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-bold text-cyan-300 mb-3">6.5 Trade-off — P4 (Meta) vs P0 (Argmax)</div>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead className="bg-[#111827]">
              <tr>
                <th className="px-2 py-1.5 text-left text-gray-400 border-b border-gray-700">지표</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">P0 (Argmax)</th>
                <th className="px-2 py-1.5 text-right text-gray-400 border-b border-gray-700">P4 (Meta)</th>
                <th className="px-2 py-1.5 text-center border-b border-gray-700 text-gray-400">변화</th>
                <th className="px-2 py-1.5 text-left border-b border-gray-700 text-gray-400">해석</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Sharpe", "0.88", "0.86", "≈", "비슷 (다른 OOS 윈도우)", null],
                ["Alpha vs ACWI 90/10", "−2.34%", "−1.40%", "+0.94pp", "✓ 절대수익 손실 줄임", "good"],
                ["MaxDD", "−17.0%", "−16.5%", "+0.5pp", "✓ 더 얕은 낙폭", "good"],
                ["Turnover", "7.4%/mo", "3.5%/mo", "−3.9pp", "✓ 거래비용 절반", "good"],
                ["BEAR Recall", "19%", "35%", "+16pp", "✓ 진짜 BEAR 더 많이 잡음", "good"],
                ["BEAR Precision", "44%", "70%", "+26pp", "✓ BEAR 신호 신뢰도 ↑", "good"],
                ["BULL Directional", "69%", "100%", "+31pp", "✓ BULL 신호 = 완벽 적중", "good"],
                ["BULL 발화 횟수", "13", "5", "−8", "→ selectivity 강함 (설계 의도)", "neutral"],
                ["Up Capture", "0.78", "0.73", "−0.05", "↓ 상승장 일부 못 따라감", "bad"],
                ["BASE Precision", "67%", "54%", "−13pp", "↓ 메타 후퇴 시 BASE에 noise 섞임", "bad"],
              ].map(([metric, p0, p4, chg, interp, sign]) => (
                <tr key={metric as string} className="border-b border-gray-800/50">
                  <td className="px-2 py-1 text-gray-300 font-semibold">{metric}</td>
                  <td className="px-2 py-1 text-right text-gray-300 font-mono">{p0}</td>
                  <td className="px-2 py-1 text-right text-gray-100 font-mono font-semibold">{p4}</td>
                  <td className="px-2 py-1 text-center font-mono">{chg}</td>
                  <td className={`px-2 py-1 text-[11px] ${
                    sign === "good" ? "text-green-300"
                    : sign === "bad" ? "text-red-300"
                    : "text-gray-400"
                  }`}>{interp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 text-[11px] text-gray-400 leading-relaxed">
          <b className="text-gray-200">핵심</b>: 메타 라벨링은 <b className="text-green-300">precision 우선, recall은 약간 양보</b>하는 설계.
          그 결과 거래비용/하방방어/false positive 감소가 절대수익 일부 양보를 초과 보상.
          <b className="text-gray-200"> 단점</b>은 강한 상승장에서 일부 BULL을 BASE로 후퇴시켜 upside capture가 5pp 떨어진다는 점.
        </div>
      </div>

      {/* 6.6 Worked Example Cases */}
      <div className="bg-[#0a0e17] rounded-lg p-4 border border-gray-800">
        <div className="text-sm font-bold text-cyan-300 mb-3">6.6 실제 사례 — 3가지 시나리오</div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">

          {/* Case A: Confident & correct */}
          <div className="bg-[#111827] border-l-4 border-green-600 rounded p-3">
            <div className="text-[12px] font-bold text-green-300 mb-1">Case A — 확신 & 적중 ✓</div>
            <div className="text-[10.5px] text-gray-400 mb-2 italic">"메타 통과 → primary 채택 → 결과 옳음"</div>

            <div className="space-y-1.5 text-[10.5px] font-mono bg-[#0a0e17] rounded p-2">
              <div><span className="text-gray-500">P_BEAR:</span> <span className="text-red-300">0.15</span></div>
              <div><span className="text-gray-500">P_BASE:</span> <span className="text-yellow-300">0.30</span></div>
              <div><span className="text-gray-500">P_BULL:</span> <span className="text-green-300">0.55</span> ← argmax</div>
              <div><span className="text-gray-500">meta_conf:</span> <span className="text-green-300">0.71</span> &gt; 0.5</div>
              <div className="border-t border-gray-700 my-1"></div>
              <div className="text-cyan-300">→ ALLOCATION[BULL] = 90/5/5</div>
              <div><span className="text-gray-500">true regime:</span> <span className="text-green-300">BULL</span></div>
              <div><span className="text-gray-500">fwd_ret:</span> <span className="text-green-300">+5.3%</span></div>
            </div>
            <div className="text-[10.5px] text-gray-300 mt-2 leading-snug">
              메타가 강한 BULL 신호의 신뢰도를 인정해 90% equity 적용. 다음 달 ACWI 5.3% 상승 → 정확히 적중.
              <b className="text-green-300"> 가장 이상적인 케이스</b>.
            </div>
          </div>

          {/* Case B: Uncertain → BASE fallback (correctly) */}
          <div className="bg-[#111827] border-l-4 border-yellow-600 rounded p-3">
            <div className="text-[12px] font-bold text-yellow-300 mb-1">Case B — 불확실 → BASE 후퇴 ✓</div>
            <div className="text-[10.5px] text-gray-400 mb-2 italic">"메타가 신뢰 낮음 판단 → 안전 후퇴 → 잘못된 베팅 회피"</div>

            <div className="space-y-1.5 text-[10.5px] font-mono bg-[#0a0e17] rounded p-2">
              <div><span className="text-gray-500">P_BEAR:</span> <span className="text-red-300">0.45</span> ← argmax</div>
              <div><span className="text-gray-500">P_BASE:</span> <span className="text-yellow-300">0.32</span></div>
              <div><span className="text-gray-500">P_BULL:</span> <span className="text-green-300">0.23</span></div>
              <div><span className="text-gray-500">meta_conf:</span> <span className="text-yellow-300">0.41</span> ≤ 0.5</div>
              <div className="border-t border-gray-700 my-1"></div>
              <div className="text-yellow-300">→ ALLOCATION[BASE] = 75/10/15</div>
              <div><span className="text-gray-500">true regime:</span> <span className="text-yellow-300">BASE</span></div>
              <div><span className="text-gray-500">fwd_ret:</span> <span className="text-gray-300">+1.2%</span></div>
            </div>
            <div className="text-[10.5px] text-gray-300 mt-2 leading-snug">
              Primary는 BEAR 예측했지만 모든 확률이 비슷(entropy 높음) → 메타가 신뢰 낮다 판단.
              실제 BASE였으므로 60/15/25 적용했다면 큰 underperform 발생. <b className="text-green-300">메타 후퇴가 옳았음</b>.
            </div>
          </div>

          {/* Case C: Confident but wrong */}
          <div className="bg-[#111827] border-l-4 border-red-600 rounded p-3">
            <div className="text-[12px] font-bold text-red-300 mb-1">Case C — 확신 but 빗나감 ✗</div>
            <div className="text-[10.5px] text-gray-400 mb-2 italic">"메타가 통과시킨 신호도 가끔 틀림 (불가피한 noise)"</div>

            <div className="space-y-1.5 text-[10.5px] font-mono bg-[#0a0e17] rounded p-2">
              <div><span className="text-gray-500">P_BEAR:</span> <span className="text-red-300">0.62</span> ← argmax</div>
              <div><span className="text-gray-500">P_BASE:</span> <span className="text-yellow-300">0.25</span></div>
              <div><span className="text-gray-500">P_BULL:</span> <span className="text-green-300">0.13</span></div>
              <div><span className="text-gray-500">meta_conf:</span> <span className="text-green-300">0.68</span> &gt; 0.5</div>
              <div className="border-t border-gray-700 my-1"></div>
              <div className="text-red-300">→ ALLOCATION[BEAR] = 60/15/25</div>
              <div><span className="text-gray-500">true regime:</span> <span className="text-green-300">BULL</span></div>
              <div><span className="text-gray-500">fwd_ret:</span> <span className="text-green-300">+4.1%</span></div>
            </div>
            <div className="text-[10.5px] text-gray-300 mt-2 leading-snug">
              모두가 BEAR 동의 + 메타도 통과시켰지만 시장이 V자 반등. 60% equity로 4.1% 상승 → 90% 가졌으면 더 벌었을 텐데 일부 놓침.
              <b className="text-red-300"> 메타-라벨링도 100% 만능 아님</b> — 이런 case가 BULL Up Capture를 5pp 깎음.
            </div>
          </div>
        </div>
      </div>

      {/* Summary box */}
      <div className="bg-gradient-to-br from-cyan-900/20 to-[#0a0e17] border-2 border-cyan-700/50 rounded-lg p-4">
        <div className="text-sm font-bold text-cyan-100 mb-2">📌 핵심 정리</div>
        <ul className="list-disc pl-5 space-y-1 text-[12px] text-gray-300 leading-relaxed">
          <li><b>Primary</b>(방향 결정) + <b>Meta</b>(신뢰도 판단)의 <b>2-stage 분리</b>로 단일 모델의 multitask burden 해소</li>
          <li>Meta 학습은 primary의 <b>OOS 예측에 대한 binary 라벨</b>(맞췄나/틀렸나)로 진행 — 자체 ground truth 가짐</li>
          <li>운용 시 <code>meta_conf &gt; 0.5</code>면 primary regime 채택, 아니면 <b>BASE로 안전 후퇴</b> (no-bet 결정)</li>
          <li>결과: <b>turnover 절반</b>, <b>BEAR precision 70%</b>, <b>BULL directional 100%</b>, MaxDD 개선 — but Up Capture 약간 손실</li>
          <li>이론적 출처: <i>López de Prado (2018) Advances in Financial Machine Learning, Ch.3</i></li>
          <li>업계 활용: AQR (trend-following confidence overlay), BlackRock SAE (alpha signal sizing), 보험·신용평가 분야</li>
        </ul>
      </div>
    </div>
  );
}
