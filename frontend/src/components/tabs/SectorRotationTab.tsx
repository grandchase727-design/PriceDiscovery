import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { fetchSectorRotation, fetchSectorRotationBacktest } from "../../api/client";
import { MetricCard } from "../shared/MetricCard";
import { CLASS_COLORS, C, DARK_LAYOUT } from "../../styles/theme";
import { useSort } from "../../hooks/useSort";

// ─────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────

interface BreadthInfo {
  n_constituents: number;
  n_eligible: number;
  pct_eligible: number;
  avg_composite: number;
  n_bullish_class: number;
  pct_bullish: number;
}

interface SectorEntry {
  ticker: string;
  sector: string;
  name: string;
  composite: number;
  tcs: number;
  tfs: number;
  rss: number;
  urs: number;
  oer: number;
  classification: string;
  eligible: boolean;
  tier: string;
  decision: string;
  decision_rationale: string;
  decision_rank: number;
  breadth: BreadthInfo;
  within11_rank?: number;
  within11_pctile?: number;
  ret_1d: number;
  ret_5d: number;
  ret_21d: number;
  ret_63d: number;
  // Phase 2 — regime overlay
  regime_fit?: number;
  group?: string;
  composite_x_regime?: number;
  regime_alignment?: string;
  missing?: boolean;
}

interface BacktestMetrics {
  total_return: number;
  cagr: number;
  sharpe: number;
  mdd: number;
  avg_monthly_ret: number;
  vol_monthly: number;
}

interface BacktestData {
  as_of: string;
  config: {
    lookback_years: number;
    top_n: number;
    turnover_bp: number;
    rebalance: string;
    signal: string;
    signal_mode?: "momentum_12_1m" | "composite_live" | "ml_momentum_blend";
    vol_target_pct?: number;
    vol_lookback_months?: number;
    max_leverage?: number;
    universe: string[];
    benchmark: string;
    n_months_backtested: number;
    years: number;
  };
  metrics: {
    strategy: BacktestMetrics;
    benchmark: BacktestMetrics;
    equalweight: BacktestMetrics;
    spy?: BacktestMetrics;
    qqq?: BacktestMetrics;
    iwm?: BacktestMetrics;
    acwi?: BacktestMetrics;
    alpha_total: number;
    alpha_annualized: number;
    alpha_vs_ew_total: number;
    alpha_vs_ew_annualized: number;
    alpha_vs_qqq_total?: number;
    alpha_vs_qqq_annualized?: number;
    alpha_vs_iwm_total?: number;
    alpha_vs_iwm_annualized?: number;
    alpha_vs_acwi_total?: number;
    alpha_vs_acwi_annualized?: number;
    win_rate_pct: number;       // vs SPY
    win_rate_vs_ew_pct: number; // vs equal-weight 11
    win_rate_vs_qqq_pct?: number;
    win_rate_vs_iwm_pct?: number;
    win_rate_vs_acwi_pct?: number;
    turnover_avg_per_rebalance: number;
    turnover_sum_dw_avg?: number;
    tracking_error_pct?: number;
    information_ratio?: number;
    active_return_annualized?: number;
    extra_benchmarks?: string[];
  };
  yearly?: Array<{
    year: number;
    n_months: number;
    strategy_ret: number;
    benchmark_ret: number;
    equalweight_ret: number;
    alpha_vs_spy: number;
    alpha_vs_ew: number;
    win_rate_vs_spy: number;
    qqq_ret?: number;
    iwm_ret?: number;
    acwi_ret?: number;
    alpha_vs_qqq?: number;
    alpha_vs_iwm?: number;
    alpha_vs_acwi?: number;
  }>;
  ml_weight_history?: Array<{
    date: string;
    weights?: Record<string, number>;
    spread_train?: number | null;
    n_obs?: number;
    n_train?: number;
    n_train_periods?: number;
    feature_importance?: Record<string, number>;
    fallback?: boolean;
  }>;
  monthly: Array<{
    date: string;
    positions: string[];
    weights?: Record<string, number>;
    n_changes: number;
    sum_dw?: number;
    vol_scale?: number;
    realized_vol_pct?: number | null;
    n_valid_universe?: number;
    ml_weights?: Record<string, number>;
    ml_spread_train?: number | null;
    ml_n_train_obs?: number;
    strategy_ret: number;
    benchmark_ret: number;
    equalweight_ret?: number;
    qqq_ret?: number;
    iwm_ret?: number;
    acwi_ret?: number;
    alpha: number;
    alpha_vs_ew?: number;
    alpha_vs_qqq?: number;
    alpha_vs_iwm?: number;
    alpha_vs_acwi?: number;
    cum_strategy: number;
    cum_benchmark: number;
    cum_equalweight?: number;
    cum_qqq?: number;
    cum_iwm?: number;
    cum_acwi?: number;
    turnover_cost_bp: number;
  }>;
  error?: string;
}

interface RegimeInfo {
  regime: string;
  label: string;
  description: string;
  confidence: string;
  confidence_pct: number;
  group_averages: Record<string, number>;
  leading_group: string | null;
  second_group: string | null;
  gap: number;
}

interface SectorRotationData {
  as_of: string;
  regime?: RegimeInfo;
  sectors: SectorEntry[];
  summary: {
    n_overweight: number;
    n_underweight: number;
    n_catchup: number;
    median_composite: number;
    dispersion: number;
    leaders: string[];
    laggards: string[];
    alpha_signal: string;
  };
  methodology: {
    tier_thresholds: Record<string, string>;
    rebalance: string;
    universe: string;
    breadth_note: string;
    dispersion_note: string;
  };
}

// ─────────────────────────────────────────────────────────────────
// Color helpers
// ─────────────────────────────────────────────────────────────────

const TIER_COLORS: Record<string, string> = {
  "OVERWEIGHT":   "#22c55e",
  "NEUTRAL+":     "#86efac",
  "CATCH-UP":     "#3b82f6",
  "NEUTRAL-":     "#fbbf24",
  "UNDERWEIGHT":  "#ef4444",
};

const DECISION_COLORS: Record<string, string> = {
  "BUY":      C.green,
  "HOLD":     "#86efac",
  "CATCH-UP": C.cyan,
  "WATCH":    C.yellow,
  "AVOID":    C.gray,
  "TRIM":     C.orange,
  "HEDGE":    "#fb923c",
  "EXIT":     C.red,
};

function compColor(v: number): string {
  if (v >= 70) return C.green;
  if (v >= 55) return C.cyan;
  if (v >= 40) return C.yellow;
  return C.red;
}

function oerColor(v: number): string {
  if (v >= 60) return C.red;
  if (v >= 40) return C.orange;
  return C.green;
}

function retColor(v: number): string {
  if (v > 3) return C.green;
  if (v > 0) return "#86efac";
  if (v > -3) return "#fca5a5";
  return C.red;
}

const REGIME_COLORS: Record<string, string> = {
  "RISK_ON_EARLY_CYCLE":  "#22c55e",   // green
  "TECH_GROWTH_LED":      "#06b6d4",   // cyan
  "LATE_CYCLE":           "#f97316",   // orange
  "DEFENSIVE_RISK_OFF":   "#ef4444",   // red
  "MIXED_TRANSITIONAL":   "#9ca3af",   // gray
};

const ALIGNMENT_COLORS: Record<string, string> = {
  "ALIGNED":  "#22c55e",
  "NEUTRAL":  "#fbbf24",
  "CONTRARY": "#ef4444",
};

const GROUP_COLORS: Record<string, string> = {
  "growth":    "#06b6d4",
  "cyclical":  "#22c55e",
  "defensive": "#a855f7",
  "commodity": "#f97316",
  "other":     "#9ca3af",
};

function regimeFitColor(v: number): string {
  if (v >= 70) return "#22c55e";
  if (v >= 50) return "#fbbf24";
  return "#ef4444";
}

// ─────────────────────────────────────────────────────────────────
// Regime Banner (Phase 2)
// ─────────────────────────────────────────────────────────────────

function RegimeBanner({ regime }: { regime: RegimeInfo }) {
  const color = REGIME_COLORS[regime.regime] || C.gray;
  return (
    <div className="rounded-lg border-2 p-4" style={{ borderColor: color + "66", backgroundColor: color + "0F" }}>
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-[10px] uppercase tracking-wider text-gray-500">Macro Regime</span>
            <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
                  style={{ color, backgroundColor: color + "22" }}>
              {regime.confidence} confidence ({regime.confidence_pct}%)
            </span>
          </div>
          <div className="text-lg font-bold mb-1" style={{ color }}>
            {regime.label}
          </div>
          <div className="text-xs text-gray-400 leading-relaxed max-w-3xl">
            {regime.description}
          </div>
        </div>
        <div className="flex gap-2 flex-wrap">
          {Object.entries(regime.group_averages)
            .sort((a, b) => b[1] - a[1])
            .map(([grp, avg]) => {
              const isLead = grp === regime.leading_group;
              const grpColor = GROUP_COLORS[grp] || C.gray;
              return (
                <div key={grp} className={`text-center px-3 py-2 rounded ${isLead ? "border" : ""}`}
                     style={{ backgroundColor: grpColor + "11", borderColor: isLead ? grpColor : "transparent" }}>
                  <div className="text-[9px] uppercase tracking-wider" style={{ color: grpColor }}>
                    {grp}{isLead ? " ★" : ""}
                  </div>
                  <div className="text-lg font-mono font-bold" style={{ color: grpColor }}>
                    {avg.toFixed(0)}
                  </div>
                </div>
              );
            })}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Heatmap (11 sector tiles)
// ─────────────────────────────────────────────────────────────────

function SectorHeatmap({ sectors }: { sectors: SectorEntry[] }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
      {sectors.filter((s) => !s.missing).map((s) => {
        const tierColor = TIER_COLORS[s.tier] || C.gray;
        const groupColor = GROUP_COLORS[s.group || "other"];
        const fitColor = s.regime_fit != null ? regimeFitColor(s.regime_fit) : C.gray;
        return (
          <div
            key={s.ticker}
            className="bg-[#0d1117] rounded-lg border-2 p-3 hover:bg-[#1f2937]/30 transition-colors"
            style={{ borderColor: tierColor + "66" }}
          >
            <div className="flex justify-between items-start mb-1">
              <div>
                <span className="font-mono text-cyan-400 font-bold text-sm">{s.ticker}</span>
                {s.group && (
                  <span className="ml-1.5 text-[8px] uppercase font-semibold" style={{ color: groupColor }}>
                    {s.group}
                  </span>
                )}
              </div>
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded"
                    style={{ color: tierColor, backgroundColor: tierColor + "22" }}>
                #{s.within11_rank}
              </span>
            </div>
            <div className="text-[10px] text-gray-500 mb-2 truncate">{s.sector}</div>
            <div className="text-2xl font-bold font-mono mb-1" style={{ color: compColor(s.composite) }}>
              {s.composite.toFixed(0)}
            </div>
            <div className="text-[9px]" style={{ color: CLASS_COLORS[s.classification] || C.gray }}>
              {s.classification}
            </div>
            <div className="mt-2 pt-2 border-t border-gray-800 text-[10px] flex justify-between">
              <span className="text-gray-500">Tier:</span>
              <span style={{ color: tierColor }} className="font-semibold">{s.tier}</span>
            </div>
            <div className="text-[10px] flex justify-between mt-0.5">
              <span className="text-gray-500">Action:</span>
              <span style={{ color: DECISION_COLORS[s.decision] || C.gray }} className="font-semibold">
                {s.decision}
              </span>
            </div>
            {s.regime_fit != null && (
              <div className="text-[10px] flex justify-between mt-0.5"
                   title={`Sector fit for current macro regime`}>
                <span className="text-gray-500">Fit:</span>
                <span className="font-semibold" style={{ color: fitColor }}>
                  {s.regime_fit} · {s.regime_alignment}
                </span>
              </div>
            )}
            <div className="text-[10px] flex justify-between mt-0.5">
              <span className="text-gray-500">Breadth:</span>
              <span className="text-gray-400">
                {s.breadth.pct_eligible.toFixed(0)}% / {s.breadth.n_constituents}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Detail table
// ─────────────────────────────────────────────────────────────────

function SectorTable({ sectors }: { sectors: SectorEntry[] }) {
  const accessors = useMemo(() => ({
    ticker: (s: SectorEntry) => s.ticker,
    sector: (s: SectorEntry) => s.sector,
    composite: (s: SectorEntry) => s.composite,
    tier: (s: SectorEntry) => s.tier,
    decision: (s: SectorEntry) => s.decision_rank,
    classification: (s: SectorEntry) => s.classification,
    oer: (s: SectorEntry) => s.oer,
    rank: (s: SectorEntry) => s.within11_rank ?? 99,
    breadth: (s: SectorEntry) => s.breadth.pct_eligible,
    bullish: (s: SectorEntry) => s.breadth.pct_bullish,
    ret_1d: (s: SectorEntry) => s.ret_1d,
    ret_5d: (s: SectorEntry) => s.ret_5d,
    ret_21d: (s: SectorEntry) => s.ret_21d,
    ret_63d: (s: SectorEntry) => s.ret_63d,
    regime_fit: (s: SectorEntry) => s.regime_fit ?? 50,
    combined: (s: SectorEntry) => s.composite_x_regime ?? s.composite,
    group: (s: SectorEntry) => s.group ?? "other",
  }), []);
  const validRows = sectors.filter((s) => !s.missing);
  const { sorted, onSort, indicator } = useSort(validRows, accessors);

  const headerCls = "py-1.5 px-2 text-gray-500 cursor-pointer select-none hover:text-gray-200 whitespace-nowrap";
  return (
    <div className="overflow-auto border border-gray-800 rounded">
      <table className="w-full text-xs border-collapse">
        <thead className="sticky top-0 z-10 bg-[#1f2937]">
          <tr className="border-b border-gray-700">
            <th className="py-1.5 px-2 text-left text-gray-500">#</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("rank")}>Rank{indicator("rank")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("ticker")}>Ticker{indicator("ticker")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("sector")}>Sector{indicator("sector")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("group")}>Group{indicator("group")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("composite")}>Comp{indicator("composite")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("regime_fit")}>Fit{indicator("regime_fit")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("combined")}>Comp×Fit{indicator("combined")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("classification")}>Class{indicator("classification")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("oer")}>OER{indicator("oer")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("tier")}>Tier{indicator("tier")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("decision")}>Decision{indicator("decision")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("breadth")}>Br% Elig{indicator("breadth")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("bullish")}>Br% Bull{indicator("bullish")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_1d")}>1D{indicator("ret_1d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_5d")}>1W{indicator("ret_5d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_21d")}>1M{indicator("ret_21d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_63d")}>3M{indicator("ret_63d")}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((s, i) => (
            <tr key={s.ticker} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
              <td className="py-1.5 px-2 text-gray-600">{i + 1}</td>
              <td className="py-1.5 px-2 text-right font-mono font-bold text-gray-300">#{s.within11_rank}</td>
              <td className="py-1.5 px-2 font-mono text-cyan-400 font-bold">{s.ticker}</td>
              <td className="py-1.5 px-2 text-gray-400 text-[11px]">{s.sector}</td>
              <td className="py-1.5 px-2 text-[10px] uppercase font-semibold"
                  style={{ color: GROUP_COLORS[s.group || "other"] }}>
                {s.group}
              </td>
              <td className="py-1.5 px-2 text-right font-mono font-bold" style={{ color: compColor(s.composite) }}>
                {s.composite.toFixed(1)}
              </td>
              <td className="py-1.5 px-2 text-right font-mono font-semibold"
                  style={{ color: regimeFitColor(s.regime_fit ?? 50) }}
                  title={`Regime alignment: ${s.regime_alignment}`}>
                {s.regime_fit ?? "-"}
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-300">
                {s.composite_x_regime?.toFixed(1) ?? "-"}
              </td>
              <td className="py-1.5 px-2 text-[10px]" style={{ color: CLASS_COLORS[s.classification] || C.gray }}>
                {s.classification}
              </td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: oerColor(s.oer) }}>
                {s.oer.toFixed(0)}
              </td>
              <td className="py-1.5 px-2">
                <span className="text-[10px] px-1.5 py-0.5 rounded font-semibold"
                      style={{ color: TIER_COLORS[s.tier], backgroundColor: (TIER_COLORS[s.tier] || C.gray) + "22" }}>
                  {s.tier}
                </span>
              </td>
              <td className="py-1.5 px-2" title={s.decision_rationale}>
                <div className="flex flex-col gap-0.5">
                  <span className="text-[11px] font-bold" style={{ color: DECISION_COLORS[s.decision] || C.gray }}>
                    {s.decision}
                  </span>
                  <span className="text-[9px] text-gray-500 leading-snug max-w-[200px] truncate">
                    {s.decision_rationale}
                  </span>
                </div>
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-300">
                {s.breadth.pct_eligible.toFixed(0)}%
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-400">
                {s.breadth.pct_bullish.toFixed(0)}%
              </td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(s.ret_1d) }}>{s.ret_1d.toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(s.ret_5d) }}>{s.ret_5d.toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(s.ret_21d) }}>{s.ret_21d.toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(s.ret_63d) }}>{s.ret_63d.toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Backtest Section (Phase 3)
// ─────────────────────────────────────────────────────────────────

function BacktestSection() {
  const [bt, setBt] = useState<BacktestData | null>(null);
  const [loading, setLoading] = useState(true);
  const [topN, setTopN] = useState(3);
  const [lookback, setLookback] = useState(5);
  const [signalMode, setSignalMode] = useState<"momentum_12_1m" | "composite_live" | "ml_momentum_blend" | "ml_lightgbm">("momentum_12_1m");
  const [volTargetPct, setVolTargetPct] = useState<number>(0);  // 0 = disabled
  const [maxLeverage, setMaxLeverage] = useState<number>(1.0);

  useEffect(() => {
    setLoading(true);
    fetchSectorRotationBacktest(lookback, topN, 30, signalMode, volTargetPct, 6, maxLeverage)
      .then(setBt)
      .finally(() => setLoading(false));
  }, [topN, lookback, signalMode, volTargetPct, maxLeverage]);

  if (loading) {
    return <div className="text-gray-500 p-4 text-sm">
      Running backtest (yfinance fetch may take ~10-30s on first call;
      longer for 20y+ windows; cached 24h after first fetch)...
    </div>;
  }
  if (!bt || bt.error) {
    return <div className="text-gray-500 p-4 text-sm">No backtest data: {bt?.error || "unavailable"}</div>;
  }

  const dates = bt.monthly.map((m) => m.date);
  const strategyCum = bt.monthly.map((m) => m.cum_strategy);
  const benchCum = bt.monthly.map((m) => m.cum_benchmark);
  const ewCum = bt.monthly.map((m) => m.cum_equalweight ?? 0);
  const qqqCum = bt.monthly.map((m) => m.cum_qqq ?? null);
  const iwmCum = bt.monthly.map((m) => m.cum_iwm ?? null);
  const acwiCum = bt.monthly.map((m) => m.cum_acwi ?? null);
  const hasQQQ = bt.monthly.some((m) => m.cum_qqq != null);
  const hasIWM = bt.monthly.some((m) => m.cum_iwm != null);
  const hasACWI = bt.monthly.some((m) => m.cum_acwi != null);

  // Per-comparator metrics row builder. Strategy is highlighted; deltas vs each comparator shown.
  type Comp = { key: string; label: string; m?: BacktestMetrics };
  const comps: Comp[] = [
    { key: "ew",   label: "EW-11", m: bt.metrics.equalweight },
    { key: "spy",  label: "SPY",   m: bt.metrics.benchmark },
    { key: "qqq",  label: "QQQ",   m: bt.metrics.qqq },
    { key: "iwm",  label: "IWM",   m: bt.metrics.iwm },
    { key: "acwi", label: "ACWI",  m: bt.metrics.acwi },
  ];
  const visibleComps = comps.filter((c) => !!c.m);

  const metricRow = (label: string, key: keyof BacktestMetrics, fmt = (v: number) => v.toFixed(2)) => {
    const strat = bt.metrics.strategy[key] as number;
    const isMdd = label.toLowerCase().includes("drawdown");
    return (
      <tr className="border-b border-gray-800/50">
        <td className="py-1.5 px-2 text-gray-400">{label}</td>
        <td className="py-1.5 px-2 text-right font-mono font-bold"
            style={{ color: C.cyan }}>
          {fmt(strat)}
        </td>
        {visibleComps.map((c) => {
          const v = c.m![key] as number;
          const delta = strat - v;
          const stratBetter = isMdd ? delta >= 0 : delta >= 0;
          return (
            <td key={c.key} className="py-1.5 px-2 text-right font-mono">
              <div className="text-gray-400">{fmt(v)}</div>
              <div className="text-[10px]" style={{ color: stratBetter ? C.green : C.red }}>
                {delta >= 0 ? "+" : ""}{fmt(delta)}
              </div>
            </td>
          );
        })}
      </tr>
    );
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <label className="text-[11px] text-gray-500">Top-N sectors:</label>
          <select value={topN} onChange={(e) => setTopN(parseInt(e.target.value, 10))}
                  className="bg-[#1f2937] border border-gray-700 text-xs rounded px-2 py-1">
            <option value={2}>2</option>
            <option value={3}>3</option>
            <option value={4}>4</option>
            <option value={5}>5</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[11px] text-gray-500">Lookback:</label>
          <select value={lookback} onChange={(e) => setLookback(parseInt(e.target.value, 10))}
                  className="bg-[#1f2937] border border-gray-700 text-xs rounded px-2 py-1">
            <option value={3}>3 years</option>
            <option value={5}>5 years</option>
            <option value={7}>7 years</option>
            <option value={10}>10 years</option>
            <option value={15}>15 years</option>
            <option value={20}>20 years (incl. GFC)</option>
            <option value={25}>25 years</option>
            <option value={99}>Max (1998+, full SPDR history)</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[11px] text-gray-500">Signal:</label>
          <select value={signalMode}
                  onChange={(e) => setSignalMode(e.target.value as "momentum_12_1m" | "composite_live" | "ml_momentum_blend" | "ml_lightgbm")}
                  className="bg-[#1f2937] border border-gray-700 text-xs rounded px-2 py-1">
            <option value="momentum_12_1m">12-1M momentum (Phase 3)</option>
            <option value="composite_live">Composite-live TCS/TFS/RSS/URS (Phase 4)</option>
            <option value="ml_momentum_blend">ML multi-horizon blend (Phase 5 — B-1)</option>
            <option value="ml_lightgbm">ML LightGBM + macro (Phase 5 — B-2)</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[11px] text-gray-500">Vol target (B-3):</label>
          <select value={volTargetPct} onChange={(e) => setVolTargetPct(parseFloat(e.target.value))}
                  className="bg-[#1f2937] border border-gray-700 text-xs rounded px-2 py-1">
            <option value={0}>Off</option>
            <option value={8}>8% ann</option>
            <option value={10}>10% ann</option>
            <option value={12}>12% ann</option>
            <option value={15}>15% ann</option>
            <option value={20}>20% ann</option>
          </select>
        </div>
        {volTargetPct > 0 && (
          <div className="flex items-center gap-2">
            <label className="text-[11px] text-gray-500">Max lev:</label>
            <select value={maxLeverage} onChange={(e) => setMaxLeverage(parseFloat(e.target.value))}
                    className="bg-[#1f2937] border border-gray-700 text-xs rounded px-2 py-1">
              <option value={1.0}>1.0x (no leverage)</option>
              <option value={1.25}>1.25x</option>
              <option value={1.5}>1.5x</option>
              <option value={2.0}>2.0x</option>
            </select>
          </div>
        )}
        <div className="text-[10px] text-gray-600">
          {bt.config.signal}  ·  {bt.config.rebalance}  ·  {bt.config.turnover_bp} bp
          {(bt.config.vol_target_pct ?? 0) > 0 && (
            <>  ·  vol-target {bt.config.vol_target_pct}%·max-lev {bt.config.max_leverage}x</>
          )}
        </div>
      </div>

      {/* KPI cards — α vs each benchmark */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          label="α vs EW-11"
          value={`${bt.metrics.alpha_vs_ew_annualized >= 0 ? "+" : ""}${bt.metrics.alpha_vs_ew_annualized.toFixed(2)}%`}
          sub="★ true selection α"
        />
        <MetricCard
          label="α vs SPY"
          value={`${bt.metrics.alpha_annualized >= 0 ? "+" : ""}${bt.metrics.alpha_annualized.toFixed(2)}%`}
          sub="S&P 500"
        />
        {bt.metrics.alpha_vs_qqq_annualized != null && (
          <MetricCard
            label="α vs QQQ"
            value={`${bt.metrics.alpha_vs_qqq_annualized >= 0 ? "+" : ""}${bt.metrics.alpha_vs_qqq_annualized.toFixed(2)}%`}
            sub="Nasdaq 100"
          />
        )}
        {bt.metrics.alpha_vs_iwm_annualized != null && (
          <MetricCard
            label="α vs IWM"
            value={`${bt.metrics.alpha_vs_iwm_annualized >= 0 ? "+" : ""}${bt.metrics.alpha_vs_iwm_annualized.toFixed(2)}%`}
            sub="Russell 2000"
          />
        )}
        {bt.metrics.alpha_vs_acwi_annualized != null && (
          <MetricCard
            label="α vs ACWI"
            value={`${bt.metrics.alpha_vs_acwi_annualized >= 0 ? "+" : ""}${bt.metrics.alpha_vs_acwi_annualized.toFixed(2)}%`}
            sub="MSCI Global"
          />
        )}
      </div>
      {/* Risk + win-rate row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard
          label="Strategy CAGR"
          value={`${bt.metrics.strategy.cagr.toFixed(2)}%`}
          sub={`over ${bt.config.years}y`}
        />
        <MetricCard
          label="MDD Reduction"
          value={`${(bt.metrics.benchmark.mdd - bt.metrics.strategy.mdd).toFixed(1)}%`}
          sub={`${bt.metrics.strategy.mdd.toFixed(1)}% vs SPY ${bt.metrics.benchmark.mdd.toFixed(1)}%`}
        />
        <MetricCard
          label="Sharpe"
          value={bt.metrics.strategy.sharpe.toFixed(2)}
          sub={`EW ${bt.metrics.equalweight.sharpe.toFixed(2)} · SPY ${bt.metrics.benchmark.sharpe.toFixed(2)}${bt.metrics.qqq ? ` · QQQ ${bt.metrics.qqq.sharpe.toFixed(2)}` : ""}`}
        />
        <MetricCard
          label="Win Rate"
          value={`${bt.metrics.win_rate_vs_ew_pct}%`}
          sub={`EW · SPY ${bt.metrics.win_rate_pct}%${bt.metrics.win_rate_vs_qqq_pct != null ? ` · QQQ ${bt.metrics.win_rate_vs_qqq_pct}%` : ""}${bt.metrics.win_rate_vs_iwm_pct != null ? ` · IWM ${bt.metrics.win_rate_vs_iwm_pct}%` : ""}${bt.metrics.win_rate_vs_acwi_pct != null ? ` · ACWI ${bt.metrics.win_rate_vs_acwi_pct}%` : ""}`}
        />
      </div>

      {/* Cumulative return chart — Strategy vs all benchmarks */}
      <div className="bg-[#111827] rounded-lg border border-gray-800 p-4">
        <h4 className="text-xs font-semibold text-gray-400 mb-2">
          Cumulative Return — Strategy vs EW-11 / SPY / QQQ / IWM / ACWI
        </h4>
        <Plot
          data={[
            {
              x: dates, y: strategyCum,
              type: "scatter", mode: "lines",
              name: `Strategy (top-${bt.config.top_n})`,
              line: { color: C.cyan, width: 2.5 },
            },
            {
              x: dates, y: ewCum,
              type: "scatter", mode: "lines",
              name: "EW-11",
              line: { color: C.purple, width: 1.4, dash: "dot" },
            },
            {
              x: dates, y: benchCum,
              type: "scatter", mode: "lines",
              name: "SPY",
              line: { color: C.gray, width: 1.4, dash: "dash" },
            },
            ...(hasQQQ ? [{
              x: dates, y: qqqCum,
              type: "scatter" as const, mode: "lines" as const,
              name: "QQQ",
              line: { color: "#f59e0b", width: 1.4, dash: "dash" as const },
            }] : []),
            ...(hasIWM ? [{
              x: dates, y: iwmCum,
              type: "scatter" as const, mode: "lines" as const,
              name: "IWM",
              line: { color: "#ec4899", width: 1.4, dash: "dash" as const },
            }] : []),
            ...(hasACWI ? [{
              x: dates, y: acwiCum,
              type: "scatter" as const, mode: "lines" as const,
              name: "ACWI",
              line: { color: "#10b981", width: 1.4, dash: "dash" as const },
            }] : []),
          ]}
          layout={{
            ...DARK_LAYOUT,
            height: 380,
            margin: { t: 30, b: 50, l: 50, r: 30 },
            xaxis: { gridcolor: "#1f2937", color: "#9ca3af" },
            yaxis: {
              gridcolor: "#1f2937", color: "#9ca3af",
              title: { text: "Cumulative %", font: { size: 10, color: "#9ca3af" } },
            },
            legend: { orientation: "h", y: 1.12, font: { size: 10, color: "#e5e7eb" } },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
      </div>

      {/* Vol-target scale + realized-vol history — only when vol_target is on (B-3) */}
      {volTargetPct > 0 && (() => {
        const scales = bt.monthly.map((m) => m.vol_scale ?? 1);
        const rvols = bt.monthly.map((m) => m.realized_vol_pct ?? null);
        const dates_v = bt.monthly.map((m) => m.date);
        const cashPct = bt.monthly.map((m) => Math.max(0, 1 - (m.vol_scale ?? 1)) * 100);
        const avgScale = scales.reduce((a, b) => a + b, 0) / Math.max(scales.length, 1);
        const avgCash = cashPct.reduce((a, b) => a + b, 0) / Math.max(cashPct.length, 1);
        return (
          <div className="bg-[#111827] rounded-lg border border-gray-800 p-4 space-y-3">
            <div>
              <h4 className="text-xs font-semibold text-cyan-400">
                Vol-target scaling (B-3) — realized vol &amp; position scale
              </h4>
              <div className="text-[10px] text-gray-500 mt-0.5">
                Each month-end: scale = clip(target_vol / realized_vol_6m, 0, max_lev).
                Cash slack (1−scale) earns 0%. Scale captures position size; turnover cost
                charged on Σ|Δw| including cash transitions.
                Avg scale: <span className="text-cyan-400">{(avgScale * 100).toFixed(0)}%</span>
                {" · avg cash: "}<span className="text-gray-400">{avgCash.toFixed(1)}%</span>
              </div>
            </div>
            <Plot
              data={[
                {
                  x: dates_v, y: rvols as (number | null)[],
                  type: "scatter", mode: "lines",
                  name: "Realized vol (6m, ann.)",
                  line: { color: "#f59e0b", width: 1.5 },
                },
                {
                  x: dates_v, y: Array(dates_v.length).fill(volTargetPct),
                  type: "scatter", mode: "lines",
                  name: `Target ${volTargetPct}%`,
                  line: { color: C.gray, width: 1, dash: "dash" },
                },
                {
                  x: dates_v, y: scales.map(s => s * 100),
                  type: "scatter", mode: "lines",
                  name: "Position scale %",
                  line: { color: C.cyan, width: 2 },
                  yaxis: "y2",
                },
              ]}
              layout={{
                ...DARK_LAYOUT,
                height: 280,
                margin: { t: 30, b: 50, l: 50, r: 60 },
                xaxis: { gridcolor: "#1f2937", color: "#9ca3af" },
                yaxis: {
                  gridcolor: "#1f2937", color: "#9ca3af",
                  title: { text: "Realized vol %", font: { size: 10, color: "#9ca3af" } },
                },
                yaxis2: {
                  overlaying: "y", side: "right", gridcolor: "transparent", color: C.cyan,
                  range: [0, Math.max(maxLeverage, 1) * 105],
                  title: { text: "Scale %", font: { size: 10, color: C.cyan } },
                },
                legend: { orientation: "h", y: 1.15, font: { size: 10, color: "#e5e7eb" } },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
          </div>
        );
      })()}

      {/* ML LightGBM feature importance — ml_lightgbm mode (B-2) */}
      {signalMode === "ml_lightgbm" && bt.ml_weight_history && bt.ml_weight_history.length > 0 && (() => {
        const wh = bt.ml_weight_history!;
        // Aggregate average importance across non-fallback fits
        const totals: Record<string, number> = {};
        const counts: Record<string, number> = {};
        let validFits = 0;
        for (const h of wh) {
          if (h.fallback) continue;
          const imp = h.feature_importance ?? {};
          validFits++;
          for (const k of Object.keys(imp)) {
            totals[k] = (totals[k] ?? 0) + imp[k];
            counts[k] = (counts[k] ?? 0) + 1;
          }
        }
        const totalSum = Object.values(totals).reduce((a, b) => a + b, 0) || 1;
        const featureNames = Object.keys(totals).sort((a, b) => totals[b] - totals[a]);
        const importance_pct = featureNames.map((k) => (totals[k] / totalSum) * 100);
        const fallbackCount = wh.length - validFits;
        const lastTrain = wh[wh.length - 1]?.n_train ?? 0;
        const lastPeriods = wh[wh.length - 1]?.n_train_periods ?? 0;
        return (
          <div className="bg-[#111827] rounded-lg border border-gray-800 p-4 space-y-3">
            <div>
              <h4 className="text-xs font-semibold text-cyan-400">
                LightGBM feature importance (B-2) — averaged across {validFits} fits
              </h4>
              <div className="text-[10px] text-gray-500 mt-0.5">
                Walk-forward fit at each rebalance: trailing 60 months × 11 sectors panel
                with 4 momentum + 7 macro features. Tree-gain importance, normalized.
                Fallback to 12-1M when training history insufficient: {fallbackCount}/{wh.length} months.
                Last fit: {lastTrain} obs over {lastPeriods} periods.
              </div>
            </div>
            <Plot
              data={[
                {
                  x: featureNames.map((k) => k.replace("macro_", "M:")),
                  y: importance_pct,
                  type: "bar",
                  marker: {
                    color: featureNames.map((k) => k.startsWith("macro_") ? "#f59e0b"
                                                : k === "sector_id" ? C.gray
                                                : C.cyan),
                  },
                  text: importance_pct.map((v) => v.toFixed(1) + "%"),
                  textposition: "outside",
                  textfont: { size: 9, color: "#e5e7eb" },
                },
              ]}
              layout={{
                ...DARK_LAYOUT,
                height: 320,
                margin: { t: 30, b: 80, l: 50, r: 30 },
                xaxis: { gridcolor: "#1f2937", color: "#9ca3af", tickangle: -30 },
                yaxis: {
                  gridcolor: "#1f2937", color: "#9ca3af",
                  title: { text: "Avg importance %", font: { size: 10, color: "#9ca3af" } },
                },
                showlegend: false,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <div className="text-[10px] text-gray-500">
              <span style={{ color: "#f59e0b" }}>Macro features (M:)</span> — VIX/yield/credit/DXY ·
              <span className="ml-2" style={{ color: C.cyan }}>Momentum features</span> — 1M/3M/6M/12-1M percentile ranks ·
              <span className="ml-2 text-gray-400">sector_id</span> — categorical encoding
            </div>
          </div>
        );
      })()}

      {/* ML weight history — only in ml_momentum_blend mode */}
      {signalMode === "ml_momentum_blend" && bt.ml_weight_history && bt.ml_weight_history.length > 0 && (() => {
        const wh = bt.ml_weight_history!;
        const dates_w = wh.map((h) => h.date);
        const w_m1 = wh.map((h) => (h.weights?.["m1_skip1"] ?? 0) * 100);
        const w_m3 = wh.map((h) => (h.weights?.["m3_skip1"] ?? 0) * 100);
        const w_m6 = wh.map((h) => (h.weights?.["m6_skip1"] ?? 0) * 100);
        const w_m12 = wh.map((h) => (h.weights?.["m12_1"] ?? 0) * 100);
        const spreads = wh.map((h) => h.spread_train ?? null);
        const lastIdx = wh.length - 1;
        const lastW = wh[lastIdx].weights ?? {};
        const lastSpread = wh[lastIdx].spread_train ?? null;
        const fallbackCount = wh.filter((h) => h.fallback).length;
        return (
          <div className="bg-[#111827] rounded-lg border border-gray-800 p-4 space-y-3">
            <div>
              <h4 className="text-xs font-semibold text-cyan-400">
                ML weight history — walk-forward Spearman-IC fit (B-1)
              </h4>
              <div className="text-[10px] text-gray-500 mt-0.5">
                Each month-end: re-fit blend weights on trailing 36 months, maximize top-3 vs bottom-3
                forward-return spread with L2 toward prior <span className="text-cyan-400">[0 / 0 / 0 / 1.0]</span>
                (PURE 12-1M = momentum_12_1m baseline). Safeguard: if optimizer's train-score doesn't beat
                prior, fall back to prior — ML never underperforms baseline within training.
                Fallback this run: {fallbackCount} of {wh.length} months.
              </div>
            </div>
            <Plot
              data={[
                { x: dates_w, y: w_m12, type: "bar", name: "12-1M", marker: { color: C.cyan } },
                { x: dates_w, y: w_m6,  type: "bar", name: "6M",    marker: { color: "#f59e0b" } },
                { x: dates_w, y: w_m3,  type: "bar", name: "3M",    marker: { color: "#ec4899" } },
                { x: dates_w, y: w_m1,  type: "bar", name: "1M",    marker: { color: C.gray } },
                {
                  x: dates_w, y: spreads as (number | null)[],
                  type: "scatter", mode: "lines",
                  name: "Train spread %",
                  line: { color: C.green, width: 1.5, dash: "dot" },
                  yaxis: "y2",
                },
              ]}
              layout={{
                ...DARK_LAYOUT,
                height: 320,
                margin: { t: 30, b: 50, l: 50, r: 60 },
                barmode: "stack",
                xaxis: { gridcolor: "#1f2937", color: "#9ca3af" },
                yaxis: {
                  gridcolor: "#1f2937", color: "#9ca3af",
                  range: [0, 100],
                  title: { text: "Weight %", font: { size: 10, color: "#9ca3af" } },
                },
                yaxis2: {
                  overlaying: "y", side: "right", gridcolor: "transparent", color: C.green,
                  title: { text: "Train spread %", font: { size: 10, color: C.green } },
                },
                legend: { orientation: "h", y: 1.12, font: { size: 10, color: "#e5e7eb" } },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <div className="text-[11px] text-gray-500">
              Latest fit ({wh[lastIdx].date}):
              <span className="ml-2 text-cyan-400 font-mono">12-1M {((lastW["m12_1"] ?? 0) * 100).toFixed(1)}%</span>
              <span className="ml-2 font-mono" style={{ color: "#f59e0b" }}>6M {((lastW["m6_skip1"] ?? 0) * 100).toFixed(1)}%</span>
              <span className="ml-2 font-mono" style={{ color: "#ec4899" }}>3M {((lastW["m3_skip1"] ?? 0) * 100).toFixed(1)}%</span>
              <span className="ml-2 font-mono text-gray-400">1M {((lastW["m1_skip1"] ?? 0) * 100).toFixed(1)}%</span>
              {lastSpread != null && (
                <span className="ml-3" style={{ color: lastSpread >= 0 ? C.green : C.red }}>
                  · train spread {lastSpread >= 0 ? "+" : ""}{lastSpread.toFixed(3)}%
                </span>
              )}
              <span className="ml-3 text-gray-600">
                · {wh[lastIdx].n_obs} train obs
              </span>
            </div>
          </div>
        );
      })()}

      {/* Metrics comparison table — Strategy vs all benchmarks */}
      <div className="bg-[#111827] rounded-lg border border-gray-800 p-4">
        <h4 className="text-xs font-semibold text-gray-400 mb-2">
          Detailed Performance — Strategy vs {visibleComps.map((c) => c.label).join(" / ")}
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-gray-700 bg-[#1f2937]">
                <th className="py-1.5 px-2 text-left text-gray-500">Metric</th>
                <th className="py-1.5 px-2 text-right text-gray-500">Strategy</th>
                {visibleComps.map((c) => (
                  <th key={c.key} className="py-1.5 px-2 text-right text-gray-500">
                    {c.label}<div className="text-[9px] text-gray-600 font-normal">value · Δ</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metricRow("Total Return (%)", "total_return")}
              {metricRow("CAGR (%)", "cagr")}
              {metricRow("Sharpe Ratio", "sharpe")}
              {metricRow("Max Drawdown (%)", "mdd")}
              {metricRow("Avg Monthly Ret (%)", "avg_monthly_ret", (v) => v.toFixed(3))}
              {metricRow("Monthly Vol (%)", "vol_monthly", (v) => v.toFixed(3))}
            </tbody>
          </table>
        </div>
        <div className="text-[10px] text-gray-600 mt-2">
          ★ <span className="text-gray-400">Δ vs EW-11</span> = true selection α (naive equal-weight diversification baseline).
          SPY = cap-weighted S&P 500 · QQQ = Nasdaq 100 (growth/tech) · IWM = Russell 2000 (small-cap) · ACWI = MSCI All Country World.
          Avg position changes/rebalance: {bt.metrics.turnover_avg_per_rebalance} · Periods: {bt.config.n_months_backtested} months
        </div>
      </div>

      {/* Yearly breakdown */}
      {bt.yearly && bt.yearly.length > 0 && (
        <div className="bg-[#111827] rounded-lg border border-gray-800 p-4">
          <h4 className="text-xs font-semibold text-gray-400 mb-2">Year-by-Year Breakdown</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-700 bg-[#1f2937]">
                  <th className="py-1.5 px-2 text-left text-gray-500">Year</th>
                  <th className="py-1.5 px-2 text-right text-gray-500">#Mo</th>
                  <th className="py-1.5 px-2 text-right text-gray-500">Strategy</th>
                  <th className="py-1.5 px-2 text-right text-gray-500">EW-11</th>
                  <th className="py-1.5 px-2 text-right text-gray-500">SPY</th>
                  {hasQQQ  && <th className="py-1.5 px-2 text-right text-gray-500">QQQ</th>}
                  {hasIWM  && <th className="py-1.5 px-2 text-right text-gray-500">IWM</th>}
                  {hasACWI && <th className="py-1.5 px-2 text-right text-gray-500">ACWI</th>}
                  <th className="py-1.5 px-2 text-right text-gray-500">α vs EW</th>
                  <th className="py-1.5 px-2 text-right text-gray-500">α vs SPY</th>
                  {hasQQQ  && <th className="py-1.5 px-2 text-right text-gray-500">α vs QQQ</th>}
                  {hasIWM  && <th className="py-1.5 px-2 text-right text-gray-500">α vs IWM</th>}
                  {hasACWI && <th className="py-1.5 px-2 text-right text-gray-500">α vs ACWI</th>}
                </tr>
              </thead>
              <tbody>
                {bt.yearly.map((y) => {
                  const fmt = (v: number | undefined) =>
                    v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
                  const colorize = (v: number | undefined) =>
                    v == null ? C.gray : (v >= 0 ? C.green : C.red);
                  return (
                  <tr key={y.year} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
                    <td className="py-1.5 px-2 font-mono text-gray-300 font-bold">{y.year}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-500">{y.n_months}</td>
                    <td className="py-1.5 px-2 text-right font-mono"
                        style={{ color: colorize(y.strategy_ret) }}>{fmt(y.strategy_ret)}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-400">{fmt(y.equalweight_ret)}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-gray-400">{fmt(y.benchmark_ret)}</td>
                    {hasQQQ  && <td className="py-1.5 px-2 text-right font-mono text-gray-400">{fmt(y.qqq_ret)}</td>}
                    {hasIWM  && <td className="py-1.5 px-2 text-right font-mono text-gray-400">{fmt(y.iwm_ret)}</td>}
                    {hasACWI && <td className="py-1.5 px-2 text-right font-mono text-gray-400">{fmt(y.acwi_ret)}</td>}
                    <td className="py-1.5 px-2 text-right font-mono font-semibold"
                        style={{ color: colorize(y.alpha_vs_ew) }}>{fmt(y.alpha_vs_ew)}</td>
                    <td className="py-1.5 px-2 text-right font-mono font-semibold"
                        style={{ color: colorize(y.alpha_vs_spy) }}>{fmt(y.alpha_vs_spy)}</td>
                    {hasQQQ  && <td className="py-1.5 px-2 text-right font-mono font-semibold"
                        style={{ color: colorize(y.alpha_vs_qqq) }}>{fmt(y.alpha_vs_qqq)}</td>}
                    {hasIWM  && <td className="py-1.5 px-2 text-right font-mono font-semibold"
                        style={{ color: colorize(y.alpha_vs_iwm) }}>{fmt(y.alpha_vs_iwm)}</td>}
                    {hasACWI && <td className="py-1.5 px-2 text-right font-mono font-semibold"
                        style={{ color: colorize(y.alpha_vs_acwi) }}>{fmt(y.alpha_vs_acwi)}</td>}
                  </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Position history grouped by year — full backtest period */}
      {(() => {
        // Group monthly records by year (descending — most recent first)
        const byYear: Record<number, typeof bt.monthly> = {};
        for (const h of bt.monthly) {
          const y = parseInt(h.date.slice(0, 4), 10);
          if (!byYear[y]) byYear[y] = [] as any;
          byYear[y].push(h);
        }
        const years = Object.keys(byYear).map(Number).sort((a, b) => b - a);

        // Aggregate per-year stats
        const yearStats = (rows: typeof bt.monthly) => {
          const cumStrat = rows.reduce((acc, h) => acc * (1 + h.strategy_ret / 100), 1) - 1;
          const cumBench = rows.reduce((acc, h) => acc * (1 + h.benchmark_ret / 100), 1) - 1;
          const totalChanges = rows.reduce((acc, h) => acc + h.n_changes, 0);
          return {
            cum_strat_pct: cumStrat * 100,
            cum_bench_pct: cumBench * 100,
            alpha_pct: (cumStrat - cumBench) * 100,
            n_months: rows.length,
            total_changes: totalChanges,
          };
        };

        return (
          <div className="bg-[#111827] rounded-lg border border-gray-800 p-4">
            <h4 className="text-xs font-semibold text-gray-400 mb-2">
              Position History by Year — full {bt.config.n_months_backtested} months ({years.length} years)
            </h4>
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
              <table className="w-full text-xs border-collapse">
                <thead className="sticky top-0 z-10 bg-[#1f2937]">
                  <tr className="border-b border-gray-700">
                    <th className="py-1.5 px-2 text-left text-gray-500">Date</th>
                    <th className="py-1.5 px-2 text-left text-gray-500">Positions</th>
                    <th className="py-1.5 px-2 text-right text-gray-500">Δ#</th>
                    <th className="py-1.5 px-2 text-right text-gray-500">Strat %</th>
                    <th className="py-1.5 px-2 text-right text-gray-500">SPY %</th>
                    <th className="py-1.5 px-2 text-right text-gray-500">Alpha</th>
                  </tr>
                </thead>
                <tbody>
                  {years.flatMap((y) => {
                    const rows = byYear[y].slice().sort((a, b) => b.date.localeCompare(a.date));
                    const stats = yearStats(rows);
                    // Year header (summary) + per-month rows
                    return [
                        <tr key={`year-${y}`} className="bg-[#0b1220] border-y border-cyan-900/40">
                          <td colSpan={2} className="py-2 px-2 font-bold text-cyan-400">
                            {y}
                            <span className="ml-3 text-[10px] text-gray-500 font-normal">
                              {stats.n_months} months · {stats.total_changes} total position changes
                            </span>
                          </td>
                          <td className="py-2 px-2 text-right text-[10px] text-gray-500 font-mono">YTD:</td>
                          <td className="py-2 px-2 text-right font-mono font-bold"
                              style={{ color: stats.cum_strat_pct >= 0 ? C.green : C.red }}>
                            {stats.cum_strat_pct >= 0 ? "+" : ""}{stats.cum_strat_pct.toFixed(2)}
                          </td>
                          <td className="py-2 px-2 text-right font-mono text-gray-400">
                            {stats.cum_bench_pct >= 0 ? "+" : ""}{stats.cum_bench_pct.toFixed(2)}
                          </td>
                          <td className="py-2 px-2 text-right font-mono font-bold"
                              style={{ color: stats.alpha_pct >= 0 ? C.green : C.red }}>
                            {stats.alpha_pct >= 0 ? "+" : ""}{stats.alpha_pct.toFixed(2)}
                          </td>
                        </tr>,
                        ...rows.map((h) => (
                          <tr key={h.date} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
                            <td className="py-1 px-2 text-gray-400 font-mono">{h.date}</td>
                            <td className="py-1 px-2 font-mono text-cyan-400">{h.positions.join(", ")}</td>
                            <td className="py-1 px-2 text-right text-gray-500 font-mono">{h.n_changes}</td>
                            <td className="py-1 px-2 text-right font-mono"
                                style={{ color: h.strategy_ret >= 0 ? C.green : C.red }}>
                              {h.strategy_ret >= 0 ? "+" : ""}{h.strategy_ret.toFixed(2)}
                            </td>
                            <td className="py-1 px-2 text-right font-mono text-gray-400">
                              {h.benchmark_ret >= 0 ? "+" : ""}{h.benchmark_ret.toFixed(2)}
                            </td>
                            <td className="py-1 px-2 text-right font-mono"
                                style={{ color: h.alpha >= 0 ? C.green : C.red }}>
                              {h.alpha >= 0 ? "+" : ""}{h.alpha.toFixed(2)}
                            </td>
                          </tr>
                        )),
                    ];
                  })}
                </tbody>
              </table>
            </div>
            <div className="text-[10px] text-gray-600 mt-2">
              YTD shows compounded yearly return (∏(1+r)−1) for strategy and SPY benchmark.
              Scrollable when total months exceed ~30 rows.
            </div>
          </div>
        );
      })()}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Main export
// ─────────────────────────────────────────────────────────────────

export function SectorRotationTab() {
  const [data, setData] = useState<SectorRotationData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchSectorRotation().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="text-gray-500 p-8">Loading sector rotation...</div>;
  }
  if (!data || !data.sectors.length) {
    return <div className="text-gray-500 p-8">No sector rotation data. Run a scan first.</div>;
  }

  const { regime, summary, sectors, methodology } = data;
  const alphaColor = summary.alpha_signal === "HIGH" ? C.green
                    : summary.alpha_signal === "MODERATE" ? C.yellow : C.gray;

  return (
    <div className="space-y-6">
      {/* ── Regime Banner (Phase 2) ── */}
      {regime && <RegimeBanner regime={regime} />}

      {/* ── Summary KPIs ── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Overweight" value={summary.n_overweight} sub="leading sectors" />
        <MetricCard label="Catch-up" value={summary.n_catchup} sub="lagging w/ potential" />
        <MetricCard label="Underweight" value={summary.n_underweight} sub="rotate out" />
        <MetricCard label="Dispersion" value={summary.dispersion.toFixed(1)} sub="comp max - min" />
        <div className="bg-[#111827] rounded-lg border border-gray-800 p-3 flex flex-col justify-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Alpha Signal</div>
          <div className="text-2xl font-bold" style={{ color: alphaColor }}>
            {summary.alpha_signal}
          </div>
          <div className="text-[10px] text-gray-600 mt-1">
            {summary.alpha_signal === "HIGH" ? "Sectors decoupling — rotation alpha available"
              : summary.alpha_signal === "MODERATE" ? "Some divergence" : "Sectors in sync — limited alpha"}
          </div>
        </div>
      </div>

      {/* ── Leaders / Laggards ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="bg-[#111827] rounded-lg border border-green-900/40 p-3">
          <div className="text-[10px] uppercase tracking-wider text-green-500 mb-1">Top 3 Leaders</div>
          <div className="flex gap-2 flex-wrap">
            {summary.leaders.map((tk) => (
              <span key={tk} className="font-mono text-cyan-400 font-bold bg-green-900/20 px-2 py-1 rounded text-sm">
                {tk}
              </span>
            ))}
          </div>
        </div>
        <div className="bg-[#111827] rounded-lg border border-red-900/40 p-3">
          <div className="text-[10px] uppercase tracking-wider text-red-500 mb-1">Bottom 3 Laggards</div>
          <div className="flex gap-2 flex-wrap">
            {summary.laggards.map((tk) => (
              <span key={tk} className="font-mono text-cyan-400 font-bold bg-red-900/20 px-2 py-1 rounded text-sm">
                {tk}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── 11 Sector Heatmap ── */}
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-2">
          Sector Heatmap (sorted by Composite, ranked within 11)
        </h3>
        <SectorHeatmap sectors={sectors} />
      </div>


      {/* ── Detail Table ── */}
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-2">Detailed Ranking + Breadth</h3>
        <SectorTable sectors={sectors} />
      </div>

      {/* ── Backtest (Phase 3) ── */}
      <div className="bg-[#0d1117] rounded-lg border border-cyan-900/40 p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-cyan-400">
            Strategy Backtest — Monthly Rebalance (Phase 3)
          </h3>
          <span className="text-[10px] text-gray-600 italic">
            12-1M momentum signal · approximate proxy for live Composite-driven rotation
          </span>
        </div>
        <BacktestSection />
      </div>

      {/* ── Methodology ── */}
      <div className="bg-[#111827] rounded-lg border border-gray-800 p-4 text-[11px] text-gray-500">
        <div className="text-xs font-semibold text-gray-400 mb-2">Methodology</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-gray-600 mb-1">Tier Thresholds</div>
            {Object.entries(methodology.tier_thresholds).map(([tier, desc]) => (
              <div key={tier} className="mb-1.5">
                <span className="font-mono font-bold mr-2" style={{ color: TIER_COLORS[tier] || C.gray }}>{tier}</span>
                <span>{desc}</span>
              </div>
            ))}
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-gray-600 mb-1">Notes</div>
            <div className="mb-1.5"><span className="text-gray-400 font-semibold">Universe: </span>{methodology.universe}</div>
            <div className="mb-1.5"><span className="text-gray-400 font-semibold">Rebalance: </span>{methodology.rebalance}</div>
            <div className="mb-1.5"><span className="text-gray-400 font-semibold">Breadth: </span>{methodology.breadth_note}</div>
            <div><span className="text-gray-400 font-semibold">Dispersion: </span>{methodology.dispersion_note}</div>
          </div>
        </div>
      </div>

      {/* As-of timestamp */}
      <div className="text-[10px] text-gray-600 text-right">
        As of: {data.as_of}
      </div>
    </div>
  );
}
