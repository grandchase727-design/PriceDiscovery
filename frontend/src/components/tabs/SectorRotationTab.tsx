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
  "OVERWEIGHT":   "#0A7D3F",
  "NEUTRAL+":     "#0A7D3F",
  "CATCH-UP":     "#0F5499",
  "NEUTRAL-":     "#B85C00",
  "UNDERWEIGHT":  "#CC0000",
};

const DECISION_COLORS: Record<string, string> = {
  "BUY":      C.green,
  "HOLD":     "#0A7D3F",
  "CATCH-UP": C.cyan,
  "WATCH":    C.yellow,
  "AVOID":    C.gray,
  "TRIM":     C.orange,
  "HEDGE":    "#C2701C",
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
  if (v > 0) return "#0A7D3F";
  if (v > -3) return "#CC0000";
  return C.red;
}

const REGIME_COLORS: Record<string, string> = {
  "RISK_ON_EARLY_CYCLE":  "#0A7D3F",   // green
  "TECH_GROWTH_LED":      "#0D7680",   // cyan
  "LATE_CYCLE":           "#C2701C",   // orange
  "DEFENSIVE_RISK_OFF":   "#CC0000",   // red
  "MIXED_TRANSITIONAL":   "#66605C",   // gray
};

const ALIGNMENT_COLORS: Record<string, string> = {
  "ALIGNED":  "#0A7D3F",
  "NEUTRAL":  "#B85C00",
  "CONTRARY": "#CC0000",
};

const GROUP_COLORS: Record<string, string> = {
  "growth":    "#0D7680",
  "cyclical":  "#0A7D3F",
  "defensive": "#7D5BA6",
  "commodity": "#C2701C",
  "other":     "#66605C",
};

function regimeFitColor(v: number): string {
  if (v >= 70) return "#0A7D3F";
  if (v >= 50) return "#B85C00";
  return "#CC0000";
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
            <span className="text-[12px] uppercase tracking-wider text-[#857F7A]">Macro Regime</span>
            <span className="px-2 py-0.5 rounded text-[12px] font-bold uppercase"
                  style={{ color, backgroundColor: color + "22" }}>
              {regime.confidence} confidence ({regime.confidence_pct}%)
            </span>
          </div>
          <div className="text-[20px] font-bold mb-1" style={{ color }}>
            {regime.label}
          </div>
          <div className="text-[14px] text-[#66605C] leading-relaxed max-w-3xl">
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
                  <div className="text-[11px] uppercase tracking-wider" style={{ color: grpColor }}>
                    {grp}{isLead ? " ★" : ""}
                  </div>
                  <div className="text-[20px] font-mono font-bold" style={{ color: grpColor }}>
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
            className="bg-[#FBEEE3] rounded-lg border-2 p-3 hover:bg-[#F2E5D7]/30 transition-colors"
            style={{ borderColor: tierColor + "66" }}
          >
            <div className="flex justify-between items-start mb-1">
              <div>
                <span className="font-mono text-[#0F5499] font-bold text-[16px]">{s.ticker}</span>
                {s.group && (
                  <span className="ml-1.5 text-[10px] uppercase font-semibold" style={{ color: groupColor }}>
                    {s.group}
                  </span>
                )}
              </div>
              <span className="text-[12px] font-bold px-1.5 py-0.5 rounded"
                    style={{ color: tierColor, backgroundColor: tierColor + "22" }}>
                #{s.within11_rank}
              </span>
            </div>
            <div className="text-[12px] text-[#857F7A] mb-2 truncate">{s.sector}</div>
            <div className="text-[26px] font-bold font-mono mb-1" style={{ color: compColor(s.composite) }}>
              {s.composite.toFixed(0)}
            </div>
            <div className="text-[11px]" style={{ color: CLASS_COLORS[s.classification] || C.gray }}>
              {s.classification}
            </div>
            <div className="mt-2 pt-2 border-t border-[#E6D9CE] text-[12px] flex justify-between">
              <span className="text-[#857F7A]">Tier:</span>
              <span style={{ color: tierColor }} className="font-semibold">{s.tier}</span>
            </div>
            <div className="text-[12px] flex justify-between mt-0.5">
              <span className="text-[#857F7A]">Action:</span>
              <span style={{ color: DECISION_COLORS[s.decision] || C.gray }} className="font-semibold">
                {s.decision}
              </span>
            </div>
            {s.regime_fit != null && (
              <div className="text-[12px] flex justify-between mt-0.5"
                   title={`Sector fit for current macro regime`}>
                <span className="text-[#857F7A]">Fit:</span>
                <span className="font-semibold" style={{ color: fitColor }}>
                  {s.regime_fit} · {s.regime_alignment}
                </span>
              </div>
            )}
            <div className="text-[12px] flex justify-between mt-0.5">
              <span className="text-[#857F7A]">Breadth:</span>
              <span className="text-[#66605C]">
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

  const headerCls = "py-1.5 px-2 text-[#857F7A] cursor-pointer select-none hover:text-[#33302E] whitespace-nowrap";
  return (
    <div className="overflow-auto border border-[#E6D9CE] rounded">
      <table className="w-full text-[14px] border-collapse">
        <thead className="sticky top-0 z-10 bg-[#F2E5D7]">
          <tr className="border-b border-[#E6D9CE]">
            <th className="py-1.5 px-2 text-left text-[#857F7A]">#</th>
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
            <tr key={s.ticker} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/30">
              <td className="py-1.5 px-2 text-[#857F7A]">{i + 1}</td>
              <td className="py-1.5 px-2 text-right font-mono font-bold text-[#33302E]">#{s.within11_rank}</td>
              <td className="py-1.5 px-2 font-mono text-[#0F5499] font-bold">{s.ticker}</td>
              <td className="py-1.5 px-2 text-[#66605C] text-[13px]">{s.sector}</td>
              <td className="py-1.5 px-2 text-[12px] uppercase font-semibold"
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
              <td className="py-1.5 px-2 text-right font-mono text-[#33302E]">
                {s.composite_x_regime?.toFixed(1) ?? "-"}
              </td>
              <td className="py-1.5 px-2 text-[12px]" style={{ color: CLASS_COLORS[s.classification] || C.gray }}>
                {s.classification}
              </td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: oerColor(s.oer) }}>
                {s.oer.toFixed(0)}
              </td>
              <td className="py-1.5 px-2">
                <span className="text-[12px] px-1.5 py-0.5 rounded font-semibold"
                      style={{ color: TIER_COLORS[s.tier], backgroundColor: (TIER_COLORS[s.tier] || C.gray) + "22" }}>
                  {s.tier}
                </span>
              </td>
              <td className="py-1.5 px-2" title={s.decision_rationale}>
                <div className="flex flex-col gap-0.5">
                  <span className="text-[13px] font-bold" style={{ color: DECISION_COLORS[s.decision] || C.gray }}>
                    {s.decision}
                  </span>
                  <span className="text-[11px] text-[#857F7A] leading-snug max-w-[250px] truncate">
                    {s.decision_rationale}
                  </span>
                </div>
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-[#33302E]">
                {s.breadth.pct_eligible.toFixed(0)}%
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">
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
  // Rolling Return window selector (6M / 12M / 24M / 36M)
  const [rollingMonths, setRollingMonths] = useState<6 | 12 | 24 | 36>(12);

  useEffect(() => {
    setLoading(true);
    fetchSectorRotationBacktest(lookback, topN, 30, signalMode, volTargetPct, 6, maxLeverage)
      .then(setBt)
      .finally(() => setLoading(false));
  }, [topN, lookback, signalMode, volTargetPct, maxLeverage]);

  if (loading) {
    return <div className="text-[#857F7A] p-4 text-[16px]">
      Running backtest (yfinance fetch may take ~10-30s on first call;
      longer for 20y+ windows; cached 24h after first fetch)...
    </div>;
  }
  if (!bt || bt.error) {
    return <div className="text-[#857F7A] p-4 text-[16px]">No backtest data: {bt?.error || "unavailable"}</div>;
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
      <tr className="border-b border-[#E6D9CE]/50">
        <td className="py-1.5 px-2 text-[#66605C]">{label}</td>
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
              <div className="text-[#66605C]">{fmt(v)}</div>
              <div className="text-[12px]" style={{ color: stratBetter ? C.green : C.red }}>
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
          <label className="text-[13px] text-[#857F7A]">Top-N sectors:</label>
          <select value={topN} onChange={(e) => setTopN(parseInt(e.target.value, 10))}
                  className="bg-[#F2E5D7] border border-[#E6D9CE] text-[14px] rounded px-2 py-1">
            <option value={2}>2</option>
            <option value={3}>3</option>
            <option value={4}>4</option>
            <option value={5}>5</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[13px] text-[#857F7A]">Lookback:</label>
          <select value={lookback} onChange={(e) => setLookback(parseInt(e.target.value, 10))}
                  className="bg-[#F2E5D7] border border-[#E6D9CE] text-[14px] rounded px-2 py-1">
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
          <label className="text-[13px] text-[#857F7A]">Signal:</label>
          <select value={signalMode}
                  onChange={(e) => setSignalMode(e.target.value as "momentum_12_1m" | "composite_live" | "ml_momentum_blend" | "ml_lightgbm")}
                  className="bg-[#F2E5D7] border border-[#E6D9CE] text-[14px] rounded px-2 py-1">
            <option value="momentum_12_1m">12-1M momentum (Phase 3)</option>
            <option value="composite_live">Composite-live TCS/TFS/RSS/URS (Phase 4)</option>
            <option value="ml_momentum_blend">ML multi-horizon blend (Phase 5 — B-1)</option>
            <option value="ml_lightgbm">ML LightGBM + macro (Phase 5 — B-2)</option>
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-[13px] text-[#857F7A]">Vol target (B-3):</label>
          <select value={volTargetPct} onChange={(e) => setVolTargetPct(parseFloat(e.target.value))}
                  className="bg-[#F2E5D7] border border-[#E6D9CE] text-[14px] rounded px-2 py-1">
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
            <label className="text-[13px] text-[#857F7A]">Max lev:</label>
            <select value={maxLeverage} onChange={(e) => setMaxLeverage(parseFloat(e.target.value))}
                    className="bg-[#F2E5D7] border border-[#E6D9CE] text-[14px] rounded px-2 py-1">
              <option value={1.0}>1.0x (no leverage)</option>
              <option value={1.25}>1.25x</option>
              <option value={1.5}>1.5x</option>
              <option value={2.0}>2.0x</option>
            </select>
          </div>
        )}
        <div className="text-[12px] text-[#857F7A]">
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
      <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4">
        <h4 className="text-[14px] font-semibold text-[#66605C] mb-2">
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
              line: { color: "#B85C00", width: 1.4, dash: "dash" as const },
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
              line: { color: "#0A7D3F", width: 1.4, dash: "dash" as const },
            }] : []),
          ]}
          layout={{
            ...DARK_LAYOUT,
            height: 380,
            margin: { t: 30, b: 50, l: 50, r: 30 },
            xaxis: { gridcolor: "#F2E5D7", color: "#66605C" },
            yaxis: {
              gridcolor: "#F2E5D7", color: "#66605C",
              title: { text: "Cumulative %", font: { size: 10, color: "#66605C" } },
            },
            legend: { orientation: "h", y: 1.12, font: { size: 10, color: "#33302E" } },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%" }}
        />
      </div>

      {/* Rolling Return Annualized — window-selectable comparison */}
      {(() => {
        // N개월 rolling annualized return:
        //   total_return_N[t] = (1 + cum[t]/100) / (1 + cum[t-N]/100) - 1
        //   annualized[t]     = (1 + total_return_N)^(12/N) - 1
        // N=12 인 경우 annualized == total_return_N (1년 window).
        const N = rollingMonths;
        const exp = 12 / N;  // annualization exponent
        const rolling = (cum: number[]) => {
          const out: (number | null)[] = [];
          for (let i = 0; i < cum.length; i++) {
            if (i < N || cum[i] == null || cum[i - N] == null) {
              out.push(null);
            } else {
              const a = 1 + cum[i] / 100;
              const b = 1 + cum[i - N] / 100;
              if (b <= 0) {
                out.push(null);
              } else {
                const totalRet = a / b;
                // Annualize. For positive base, use power. For negative base (rare), fallback to non-annualized.
                const annRet = totalRet > 0 ? Math.pow(totalRet, exp) - 1 : (totalRet - 1);
                out.push(annRet * 100);
              }
            }
          }
          return out;
        };
        const rollStrategy = rolling(strategyCum);
        const rollEW = rolling(ewCum);
        const rollSpy = rolling(benchCum);
        const rollQqq = hasQQQ ? rolling(qqqCum as number[]) : null;
        const rollIwm = hasIWM ? rolling(iwmCum as number[]) : null;
        const rollAcwi = hasACWI ? rolling(acwiCum as number[]) : null;
        const lastVal = (arr: (number|null)[]) => {
          for (let i = arr.length - 1; i >= 0; i--) {
            if (arr[i] != null) return arr[i] as number;
          }
          return null;
        };
        const ls = lastVal(rollStrategy), lew = lastVal(rollEW), lspy = lastVal(rollSpy);
        const windowLabel = N === 6 ? "6-Month" : N === 12 ? "12-Month" : N === 24 ? "2-Year" : "3-Year";
        const windowOptions: Array<{ months: 6|12|24|36; label: string }> = [
          { months: 6,  label: "6M" },
          { months: 12, label: "12M" },
          { months: 24, label: "2Y" },
          { months: 36, label: "3Y" },
        ];
        return (
          <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4">
            <div className="flex items-baseline justify-between mb-2 flex-wrap gap-2">
              <h4 className="text-[14px] font-semibold text-[#66605C]">
                Rolling {windowLabel} Return (Annualized) — Strategy vs Benchmarks
              </h4>
              <div className="flex items-center gap-2">
                {/* Window toggle */}
                <div className="flex items-center gap-1">
                  <span className="text-[12px] text-[#857F7A] mr-1">Window:</span>
                  {windowOptions.map((opt) => (
                    <button
                      key={opt.months}
                      onClick={() => setRollingMonths(opt.months)}
                      className={`px-2 py-0.5 text-[12px] font-mono rounded transition-colors ${
                        rollingMonths === opt.months
                          ? "bg-[#E3EEF5]/60 text-[#0D7680] border border-[#9CC3D5]"
                          : "bg-[#F2E5D7] text-[#66605C] border border-[#E6D9CE] hover:bg-[#283344] hover:text-[#33302E]"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
                <span className="text-[12px] text-[#857F7A] ml-2">
                  Current:&nbsp;
                  <span className="font-mono" style={{ color: C.cyan }}>Strategy {ls != null ? `${ls >= 0 ? "+" : ""}${ls.toFixed(1)}%` : "—"}</span>
                  {" · "}
                  <span className="font-mono" style={{ color: C.purple }}>EW {lew != null ? `${lew >= 0 ? "+" : ""}${lew.toFixed(1)}%` : "—"}</span>
                  {" · "}
                  <span className="font-mono" style={{ color: C.gray }}>SPY {lspy != null ? `${lspy >= 0 ? "+" : ""}${lspy.toFixed(1)}%` : "—"}</span>
                </span>
              </div>
            </div>
            <Plot
              data={[
                {
                  x: dates, y: rollStrategy,
                  type: "scatter", mode: "lines",
                  name: `Strategy (top-${bt.config.top_n})`,
                  line: { color: C.cyan, width: 2.5 },
                  connectgaps: false,
                },
                {
                  x: dates, y: rollEW,
                  type: "scatter", mode: "lines",
                  name: "EW-11",
                  line: { color: C.purple, width: 1.4, dash: "dot" },
                  connectgaps: false,
                },
                {
                  x: dates, y: rollSpy,
                  type: "scatter", mode: "lines",
                  name: "SPY",
                  line: { color: C.gray, width: 1.4, dash: "dash" },
                  connectgaps: false,
                },
                ...(hasQQQ && rollQqq ? [{
                  x: dates, y: rollQqq,
                  type: "scatter" as const, mode: "lines" as const,
                  name: "QQQ",
                  line: { color: "#B85C00", width: 1.4, dash: "dash" as const },
                  connectgaps: false,
                }] : []),
                ...(hasIWM && rollIwm ? [{
                  x: dates, y: rollIwm,
                  type: "scatter" as const, mode: "lines" as const,
                  name: "IWM",
                  line: { color: "#ec4899", width: 1.4, dash: "dash" as const },
                  connectgaps: false,
                }] : []),
                ...(hasACWI && rollAcwi ? [{
                  x: dates, y: rollAcwi,
                  type: "scatter" as const, mode: "lines" as const,
                  name: "ACWI",
                  line: { color: "#0A7D3F", width: 1.4, dash: "dash" as const },
                  connectgaps: false,
                }] : []),
                // 0% 기준선
                {
                  x: dates, y: dates.map(() => 0),
                  type: "scatter", mode: "lines",
                  name: "0%",
                  line: { color: "#CCC1B7", width: 0.8, dash: "dot" },
                  showlegend: false,
                  hoverinfo: "skip",
                },
              ]}
              layout={{
                ...DARK_LAYOUT,
                height: 340,
                margin: { t: 30, b: 50, l: 50, r: 30 },
                xaxis: { gridcolor: "#F2E5D7", color: "#66605C" },
                yaxis: {
                  gridcolor: "#F2E5D7", color: "#66605C",
                  title: { text: "12M Return %", font: { size: 10, color: "#66605C" } },
                  zeroline: false,
                  ticksuffix: "%",
                },
                legend: { orientation: "h", y: 1.12, font: { size: 10, color: "#33302E" } },
                hovermode: "x unified" as any,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <div className="text-[12px] text-[#857F7A] mt-2 leading-relaxed">
              직전 {N}개월 수익률을 매월 ending point 기준으로 계산 후 연환산:&nbsp;
              <span className="font-mono">annualized = (1 + total_return)^(12/{N}) − 1</span>.&nbsp;
              {N === 12 && "1년 window 이므로 연환산과 동일."}
              {N === 6 && "단기 변동성 큼 — momentum spike와 drawdown이 더 자주 가시화."}
              {N === 24 && "중기 안정 추세 — 단기 noise 감소, regime 변화 식별에 유리."}
              {N === 36 && "장기 추세 평가 — 사이클 전반의 alpha 누적도 비교에 최적."}
              {" "}Strategy &gt; 0% 구간 = 상승장 유리 / &lt; 0% 구간 = 약세장에서의 protective drawdown.
              SPY/EW-11와의 ★격차★가 selection alpha — Strategy 라인이 다른 벤치마크 라인 위에 있으면 outperform 구간.
              초기 {N}개월은 데이터 부족으로 비어 있음.
            </div>
          </div>
        );
      })()}

      {/* Vol-target scale + realized-vol history — only when vol_target is on (B-3) */}
      {volTargetPct > 0 && (() => {
        const scales = bt.monthly.map((m) => m.vol_scale ?? 1);
        const rvols = bt.monthly.map((m) => m.realized_vol_pct ?? null);
        const dates_v = bt.monthly.map((m) => m.date);
        const cashPct = bt.monthly.map((m) => Math.max(0, 1 - (m.vol_scale ?? 1)) * 100);
        const avgScale = scales.reduce((a, b) => a + b, 0) / Math.max(scales.length, 1);
        const avgCash = cashPct.reduce((a, b) => a + b, 0) / Math.max(cashPct.length, 1);
        return (
          <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4 space-y-3">
            <div>
              <h4 className="text-[14px] font-semibold text-[#0F5499]">
                Vol-target scaling (B-3) — realized vol &amp; position scale
              </h4>
              <div className="text-[12px] text-[#857F7A] mt-0.5">
                Each month-end: scale = clip(target_vol / realized_vol_6m, 0, max_lev).
                Cash slack (1−scale) earns 0%. Scale captures position size; turnover cost
                charged on Σ|Δw| including cash transitions.
                Avg scale: <span className="text-[#0F5499]">{(avgScale * 100).toFixed(0)}%</span>
                {" · avg cash: "}<span className="text-[#66605C]">{avgCash.toFixed(1)}%</span>
              </div>
            </div>
            <Plot
              data={[
                {
                  x: dates_v, y: rvols as (number | null)[],
                  type: "scatter", mode: "lines",
                  name: "Realized vol (6m, ann.)",
                  line: { color: "#B85C00", width: 1.5 },
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
                xaxis: { gridcolor: "#F2E5D7", color: "#66605C" },
                yaxis: {
                  gridcolor: "#F2E5D7", color: "#66605C",
                  title: { text: "Realized vol %", font: { size: 10, color: "#66605C" } },
                },
                yaxis2: {
                  overlaying: "y", side: "right", gridcolor: "transparent", color: C.cyan,
                  range: [0, Math.max(maxLeverage, 1) * 105],
                  title: { text: "Scale %", font: { size: 10, color: C.cyan } },
                },
                legend: { orientation: "h", y: 1.15, font: { size: 10, color: "#33302E" } },
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
          <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4 space-y-3">
            <div>
              <h4 className="text-[14px] font-semibold text-[#0F5499]">
                LightGBM feature importance (B-2) — averaged across {validFits} fits
              </h4>
              <div className="text-[12px] text-[#857F7A] mt-0.5">
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
                    color: featureNames.map((k) => k.startsWith("macro_") ? "#B85C00"
                                                : k === "sector_id" ? C.gray
                                                : C.cyan),
                  },
                  text: importance_pct.map((v) => v.toFixed(1) + "%"),
                  textposition: "outside",
                  textfont: { size: 9, color: "#33302E" },
                },
              ]}
              layout={{
                ...DARK_LAYOUT,
                height: 320,
                margin: { t: 30, b: 80, l: 50, r: 30 },
                xaxis: { gridcolor: "#F2E5D7", color: "#66605C", tickangle: -30 },
                yaxis: {
                  gridcolor: "#F2E5D7", color: "#66605C",
                  title: { text: "Avg importance %", font: { size: 10, color: "#66605C" } },
                },
                showlegend: false,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <div className="text-[12px] text-[#857F7A]">
              <span style={{ color: "#B85C00" }}>Macro features (M:)</span> — VIX/yield/credit/DXY ·
              <span className="ml-2" style={{ color: C.cyan }}>Momentum features</span> — 1M/3M/6M/12-1M percentile ranks ·
              <span className="ml-2 text-[#66605C]">sector_id</span> — categorical encoding
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
          <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4 space-y-3">
            <div>
              <h4 className="text-[14px] font-semibold text-[#0F5499]">
                ML weight history — walk-forward Spearman-IC fit (B-1)
              </h4>
              <div className="text-[12px] text-[#857F7A] mt-0.5">
                Each month-end: re-fit blend weights on trailing 36 months, maximize top-3 vs bottom-3
                forward-return spread with L2 toward prior <span className="text-[#0F5499]">[0 / 0 / 0 / 1.0]</span>
                (PURE 12-1M = momentum_12_1m baseline). Safeguard: if optimizer's train-score doesn't beat
                prior, fall back to prior — ML never underperforms baseline within training.
                Fallback this run: {fallbackCount} of {wh.length} months.
              </div>
            </div>
            <Plot
              data={[
                { x: dates_w, y: w_m12, type: "bar", name: "12-1M", marker: { color: C.cyan } },
                { x: dates_w, y: w_m6,  type: "bar", name: "6M",    marker: { color: "#B85C00" } },
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
                xaxis: { gridcolor: "#F2E5D7", color: "#66605C" },
                yaxis: {
                  gridcolor: "#F2E5D7", color: "#66605C",
                  range: [0, 100],
                  title: { text: "Weight %", font: { size: 10, color: "#66605C" } },
                },
                yaxis2: {
                  overlaying: "y", side: "right", gridcolor: "transparent", color: C.green,
                  title: { text: "Train spread %", font: { size: 10, color: C.green } },
                },
                legend: { orientation: "h", y: 1.12, font: { size: 10, color: "#33302E" } },
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%" }}
            />
            <div className="text-[13px] text-[#857F7A]">
              Latest fit ({wh[lastIdx].date}):
              <span className="ml-2 text-[#0F5499] font-mono">12-1M {((lastW["m12_1"] ?? 0) * 100).toFixed(1)}%</span>
              <span className="ml-2 font-mono" style={{ color: "#B85C00" }}>6M {((lastW["m6_skip1"] ?? 0) * 100).toFixed(1)}%</span>
              <span className="ml-2 font-mono" style={{ color: "#ec4899" }}>3M {((lastW["m3_skip1"] ?? 0) * 100).toFixed(1)}%</span>
              <span className="ml-2 font-mono text-[#66605C]">1M {((lastW["m1_skip1"] ?? 0) * 100).toFixed(1)}%</span>
              {lastSpread != null && (
                <span className="ml-3" style={{ color: lastSpread >= 0 ? C.green : C.red }}>
                  · train spread {lastSpread >= 0 ? "+" : ""}{lastSpread.toFixed(3)}%
                </span>
              )}
              <span className="ml-3 text-[#857F7A]">
                · {wh[lastIdx].n_obs} train obs
              </span>
            </div>
          </div>
        );
      })()}

      {/* Metrics comparison table — Strategy vs all benchmarks */}
      <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4">
        <h4 className="text-[14px] font-semibold text-[#66605C] mb-2">
          Detailed Performance — Strategy vs {visibleComps.map((c) => c.label).join(" / ")}
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-[14px] border-collapse">
            <thead>
              <tr className="border-b border-[#E6D9CE] bg-[#F2E5D7]">
                <th className="py-1.5 px-2 text-left text-[#857F7A]">Metric</th>
                <th className="py-1.5 px-2 text-right text-[#857F7A]">Strategy</th>
                {visibleComps.map((c) => (
                  <th key={c.key} className="py-1.5 px-2 text-right text-[#857F7A]">
                    {c.label}<div className="text-[11px] text-[#857F7A] font-normal">value · Δ</div>
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
        <div className="text-[12px] text-[#857F7A] mt-2">
          ★ <span className="text-[#66605C]">Δ vs EW-11</span> = true selection α (naive equal-weight diversification baseline).
          SPY = cap-weighted S&P 500 · QQQ = Nasdaq 100 (growth/tech) · IWM = Russell 2000 (small-cap) · ACWI = MSCI All Country World.
          Avg position changes/rebalance: {bt.metrics.turnover_avg_per_rebalance} · Periods: {bt.config.n_months_backtested} months
        </div>
      </div>

      {/* Yearly breakdown */}
      {bt.yearly && bt.yearly.length > 0 && (
        <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4">
          <h4 className="text-[14px] font-semibold text-[#66605C] mb-2">Year-by-Year Breakdown</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-[14px] border-collapse">
              <thead>
                <tr className="border-b border-[#E6D9CE] bg-[#F2E5D7]">
                  <th className="py-1.5 px-2 text-left text-[#857F7A]">Year</th>
                  <th className="py-1.5 px-2 text-right text-[#857F7A]">#Mo</th>
                  <th className="py-1.5 px-2 text-right text-[#857F7A]">Strategy</th>
                  <th className="py-1.5 px-2 text-right text-[#857F7A]">EW-11</th>
                  <th className="py-1.5 px-2 text-right text-[#857F7A]">SPY</th>
                  {hasQQQ  && <th className="py-1.5 px-2 text-right text-[#857F7A]">QQQ</th>}
                  {hasIWM  && <th className="py-1.5 px-2 text-right text-[#857F7A]">IWM</th>}
                  {hasACWI && <th className="py-1.5 px-2 text-right text-[#857F7A]">ACWI</th>}
                  <th className="py-1.5 px-2 text-right text-[#857F7A]">α vs EW</th>
                  <th className="py-1.5 px-2 text-right text-[#857F7A]">α vs SPY</th>
                  {hasQQQ  && <th className="py-1.5 px-2 text-right text-[#857F7A]">α vs QQQ</th>}
                  {hasIWM  && <th className="py-1.5 px-2 text-right text-[#857F7A]">α vs IWM</th>}
                  {hasACWI && <th className="py-1.5 px-2 text-right text-[#857F7A]">α vs ACWI</th>}
                </tr>
              </thead>
              <tbody>
                {bt.yearly.map((y) => {
                  const fmt = (v: number | undefined) =>
                    v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
                  const colorize = (v: number | undefined) =>
                    v == null ? C.gray : (v >= 0 ? C.green : C.red);
                  return (
                  <tr key={y.year} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/30">
                    <td className="py-1.5 px-2 font-mono text-[#33302E] font-bold">{y.year}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-[#857F7A]">{y.n_months}</td>
                    <td className="py-1.5 px-2 text-right font-mono"
                        style={{ color: colorize(y.strategy_ret) }}>{fmt(y.strategy_ret)}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{fmt(y.equalweight_ret)}</td>
                    <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{fmt(y.benchmark_ret)}</td>
                    {hasQQQ  && <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{fmt(y.qqq_ret)}</td>}
                    {hasIWM  && <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{fmt(y.iwm_ret)}</td>}
                    {hasACWI && <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{fmt(y.acwi_ret)}</td>}
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

        // Aggregate per-year stats — strategy + SPY + QQQ + IWM
        const yearStats = (rows: typeof bt.monthly) => {
          const compound = (rs: number[]) => rs.reduce((acc, r) => acc * (1 + r / 100), 1) - 1;
          const cumStrat = compound(rows.map((h) => h.strategy_ret));
          const cumBench = compound(rows.map((h) => h.benchmark_ret));
          // QQQ/IWM may be missing — compound only over months with valid data
          const qqqRows = rows.filter((h) => h.qqq_ret != null);
          const iwmRows = rows.filter((h) => h.iwm_ret != null);
          const cumQqq = qqqRows.length === rows.length ? compound(rows.map((h) => h.qqq_ret as number)) : null;
          const cumIwm = iwmRows.length === rows.length ? compound(rows.map((h) => h.iwm_ret as number)) : null;
          const totalChanges = rows.reduce((acc, h) => acc + h.n_changes, 0);
          return {
            cum_strat_pct: cumStrat * 100,
            cum_bench_pct: cumBench * 100,
            cum_qqq_pct: cumQqq != null ? cumQqq * 100 : null,
            cum_iwm_pct: cumIwm != null ? cumIwm * 100 : null,
            alpha_spy_pct: (cumStrat - cumBench) * 100,
            alpha_qqq_pct: cumQqq != null ? (cumStrat - cumQqq) * 100 : null,
            alpha_iwm_pct: cumIwm != null ? (cumStrat - cumIwm) * 100 : null,
            n_months: rows.length,
            total_changes: totalChanges,
          };
        };

        return (
          <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4">
            <h4 className="text-[14px] font-semibold text-[#66605C] mb-2">
              Position History by Year — full {bt.config.n_months_backtested} months ({years.length} years)
            </h4>
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
              <table className="w-full text-[14px] border-collapse">
                <thead className="sticky top-0 z-10 bg-[#F2E5D7]">
                  <tr className="border-b border-[#E6D9CE]">
                    <th className="py-1.5 px-2 text-left text-[#857F7A]">Date</th>
                    <th className="py-1.5 px-2 text-left text-[#857F7A]">Positions</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">Δ#</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">Strat %</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">SPY %</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">α-SPY</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">QQQ %</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">α-QQQ</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">IWM %</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">α-IWM</th>
                  </tr>
                </thead>
                <tbody>
                  {years.flatMap((y) => {
                    const rows = byYear[y].slice().sort((a, b) => b.date.localeCompare(a.date));
                    const stats = yearStats(rows);
                    const fmt = (v: number | null | undefined, decimals = 2) =>
                      v == null ? "—" : `${v >= 0 ? "+" : ""}${v.toFixed(decimals)}`;
                    // Year header (summary) + per-month rows
                    return [
                        <tr key={`year-${y}`} className="bg-[#FFF1E5] border-y border-[#9CC3D5]/40">
                          <td colSpan={2} className="py-2 px-2 font-bold text-[#0F5499]">
                            {y}
                            <span className="ml-3 text-[12px] text-[#857F7A] font-normal">
                              {stats.n_months} months · {stats.total_changes} total position changes
                            </span>
                          </td>
                          <td className="py-2 px-2 text-right text-[12px] text-[#857F7A] font-mono">YTD:</td>
                          <td className="py-2 px-2 text-right font-mono font-bold"
                              style={{ color: stats.cum_strat_pct >= 0 ? C.green : C.red }}>
                            {fmt(stats.cum_strat_pct)}
                          </td>
                          <td className="py-2 px-2 text-right font-mono text-[#66605C]">
                            {fmt(stats.cum_bench_pct)}
                          </td>
                          <td className="py-2 px-2 text-right font-mono font-bold"
                              style={{ color: stats.alpha_spy_pct >= 0 ? C.green : C.red }}>
                            {fmt(stats.alpha_spy_pct)}
                          </td>
                          <td className="py-2 px-2 text-right font-mono text-[#66605C]">
                            {fmt(stats.cum_qqq_pct)}
                          </td>
                          <td className="py-2 px-2 text-right font-mono font-bold"
                              style={{ color: stats.alpha_qqq_pct == null ? C.gray : stats.alpha_qqq_pct >= 0 ? C.green : C.red }}>
                            {fmt(stats.alpha_qqq_pct)}
                          </td>
                          <td className="py-2 px-2 text-right font-mono text-[#66605C]">
                            {fmt(stats.cum_iwm_pct)}
                          </td>
                          <td className="py-2 px-2 text-right font-mono font-bold"
                              style={{ color: stats.alpha_iwm_pct == null ? C.gray : stats.alpha_iwm_pct >= 0 ? C.green : C.red }}>
                            {fmt(stats.alpha_iwm_pct)}
                          </td>
                        </tr>,
                        ...rows.map((h) => {
                          const aQqq = h.alpha_vs_qqq ?? (h.qqq_ret != null ? h.strategy_ret - (h.qqq_ret as number) : null);
                          const aIwm = h.alpha_vs_iwm ?? (h.iwm_ret != null ? h.strategy_ret - (h.iwm_ret as number) : null);
                          return (
                          <tr key={h.date} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/30">
                            <td className="py-1 px-2 text-[#66605C] font-mono">{h.date}</td>
                            <td className="py-1 px-2 font-mono text-[#0F5499]">{h.positions.join(", ")}</td>
                            <td className="py-1 px-2 text-right text-[#857F7A] font-mono">{h.n_changes}</td>
                            <td className="py-1 px-2 text-right font-mono"
                                style={{ color: h.strategy_ret >= 0 ? C.green : C.red }}>
                              {fmt(h.strategy_ret)}
                            </td>
                            <td className="py-1 px-2 text-right font-mono text-[#66605C]">
                              {fmt(h.benchmark_ret)}
                            </td>
                            <td className="py-1 px-2 text-right font-mono"
                                style={{ color: h.alpha >= 0 ? C.green : C.red }}>
                              {fmt(h.alpha)}
                            </td>
                            <td className="py-1 px-2 text-right font-mono text-[#66605C]">
                              {fmt(h.qqq_ret)}
                            </td>
                            <td className="py-1 px-2 text-right font-mono"
                                style={{ color: aQqq == null ? C.gray : aQqq >= 0 ? C.green : C.red }}>
                              {fmt(aQqq)}
                            </td>
                            <td className="py-1 px-2 text-right font-mono text-[#66605C]">
                              {fmt(h.iwm_ret)}
                            </td>
                            <td className="py-1 px-2 text-right font-mono"
                                style={{ color: aIwm == null ? C.gray : aIwm >= 0 ? C.green : C.red }}>
                              {fmt(aIwm)}
                            </td>
                          </tr>
                          );
                        }),
                    ];
                  })}
                </tbody>
              </table>
            </div>
            <div className="text-[12px] text-[#857F7A] mt-2 leading-relaxed">
              YTD shows compounded yearly return (∏(1+r)−1) for strategy and benchmarks (SPY / QQQ / IWM).
              α-{`{Benchmark}`} = Strategy 수익률 − 해당 벤치마크 수익률. 양수 = strategy 우위.<br/>
              <span className="text-[#B85C00]">📅 날짜 해석</span>: 표의 'Date'는 <span className="font-mono">월말(month-end)</span> 기준 —
              "2026-05-31" 행은 <span className="font-mono">2026-05-01 → 2026-05-31</span> 기간 동안 해당 포지션을 보유했다는 의미.
              매월 말 close 가격으로 rebalance, 다음 달 한 달간 동일 포지션 holding (월간 rebalance backtest).
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
    return <div className="text-[#857F7A] p-8">Loading sector rotation...</div>;
  }
  if (!data || !data.sectors.length) {
    return <div className="text-[#857F7A] p-8">No sector rotation data. Run a scan first.</div>;
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
        <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-3 flex flex-col justify-center">
          <div className="text-[12px] text-[#857F7A] uppercase tracking-wider mb-1">Alpha Signal</div>
          <div className="text-[26px] font-bold" style={{ color: alphaColor }}>
            {summary.alpha_signal}
          </div>
          <div className="text-[12px] text-[#857F7A] mt-1">
            {summary.alpha_signal === "HIGH" ? "Sectors decoupling — rotation alpha available"
              : summary.alpha_signal === "MODERATE" ? "Some divergence" : "Sectors in sync — limited alpha"}
          </div>
        </div>
      </div>

      {/* ── Leaders / Laggards ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="bg-[#FFFFFF] rounded-lg border border-[#A8CDB6]/40 p-3">
          <div className="text-[12px] uppercase tracking-wider text-[#0A7D3F] mb-1">Top 3 Leaders</div>
          <div className="flex gap-2 flex-wrap">
            {summary.leaders.map((tk) => (
              <span key={tk} className="font-mono text-[#0F5499] font-bold bg-[#E3F0E8]/20 px-2 py-1 rounded text-[16px]">
                {tk}
              </span>
            ))}
          </div>
        </div>
        <div className="bg-[#FFFFFF] rounded-lg border border-[#E0AAAA]/40 p-3">
          <div className="text-[12px] uppercase tracking-wider text-[#CC0000] mb-1">Bottom 3 Laggards</div>
          <div className="flex gap-2 flex-wrap">
            {summary.laggards.map((tk) => (
              <span key={tk} className="font-mono text-[#0F5499] font-bold bg-[#F7E3E3]/20 px-2 py-1 rounded text-[16px]">
                {tk}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── 11 Sector Heatmap ── */}
      <div>
        <h3 className="text-[16px] font-semibold text-[#33302E] mb-2">
          Sector Heatmap (sorted by Composite, ranked within 11)
        </h3>
        <SectorHeatmap sectors={sectors} />
      </div>


      {/* ── Detail Table ── */}
      <div>
        <h3 className="text-[16px] font-semibold text-[#33302E] mb-2">Detailed Ranking + Breadth</h3>
        <SectorTable sectors={sectors} />
      </div>

      {/* ── Backtest (Phase 3) ── */}
      <div className="bg-[#FBEEE3] rounded-lg border border-[#9CC3D5]/40 p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-[16px] font-semibold text-[#0F5499]">
            Strategy Backtest — Monthly Rebalance (Phase 3)
          </h3>
          <span className="text-[12px] text-[#857F7A] italic">
            12-1M momentum signal · approximate proxy for live Composite-driven rotation
          </span>
        </div>
        <BacktestSection />
      </div>

      {/* ── Methodology ── */}
      <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4 text-[13px] text-[#857F7A]">
        <div className="text-[14px] font-semibold text-[#66605C] mb-2">Methodology</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <div className="text-[12px] uppercase tracking-wider text-[#857F7A] mb-1">Tier Thresholds</div>
            {Object.entries(methodology.tier_thresholds).map(([tier, desc]) => (
              <div key={tier} className="mb-1.5">
                <span className="font-mono font-bold mr-2" style={{ color: TIER_COLORS[tier] || C.gray }}>{tier}</span>
                <span>{desc}</span>
              </div>
            ))}
          </div>
          <div>
            <div className="text-[12px] uppercase tracking-wider text-[#857F7A] mb-1">Notes</div>
            <div className="mb-1.5"><span className="text-[#66605C] font-semibold">Universe: </span>{methodology.universe}</div>
            <div className="mb-1.5"><span className="text-[#66605C] font-semibold">Rebalance: </span>{methodology.rebalance}</div>
            <div className="mb-1.5"><span className="text-[#66605C] font-semibold">Breadth: </span>{methodology.breadth_note}</div>
            <div><span className="text-[#66605C] font-semibold">Dispersion: </span>{methodology.dispersion_note}</div>
          </div>
        </div>
      </div>

      {/* As-of timestamp */}
      <div className="text-[12px] text-[#857F7A] text-right">
        As of: {data.as_of}
      </div>
    </div>
  );
}
