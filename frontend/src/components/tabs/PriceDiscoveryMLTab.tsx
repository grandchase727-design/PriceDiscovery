import { useState, useEffect } from "react";
import { fetchTableML, fetchPreMomentumML, fetchMlMeta, type FilterParams } from "../../api/client";
import { PreMomentumTab } from "./PreMomentumTab";
import { MomentumTab } from "./MomentumTab";
import { ExcludedTab } from "./ExcludedTab";
import { C } from "../../styles/theme";

const SUBS = [
  { label: "Pre-Momentum", desc: "Momentum formation candidates (5-agent analysis is signal-independent)" },
  { label: "Momentum", desc: "ML-rescored Composite ≥ 55 + bullish + ADV/QVR gate" },
  { label: "Excluded", desc: "Bearish or fundamentals-rejected by ML scoring" },
] as const;

interface MlWeights {
  w_tcs: number; w_tfs: number; w_rss: number; w_urs: number;
}

interface MlMetaResponse {
  available: boolean;
  ml_meta: {
    as_of?: string;
    weights_per_class?: Record<string, MlWeights>;
    stage_counts?: { "pre-momentum"?: number; momentum?: number; excluded?: number };
    delta_distribution?: { upgraded_to_momentum?: number; demoted_from_momentum?: number; unchanged?: number };
    n_total?: number;
    params_source?: string;
  };
}

const DEFAULT_WEIGHTS: MlWeights = { w_tcs: 0.30, w_tfs: 0.25, w_rss: 0.30, w_urs: 0.15 };

function MlMetaPanel({ meta }: { meta: MlMetaResponse | null }) {
  if (!meta || !meta.available) {
    return (
      <div className="bg-[#F2E5D7] border border-[#E6D9CE] rounded p-3 text-[14px] text-[#66605C]">
        ML cache not loaded. Run <code className="text-[#0F5499]">python3 optimize_params.py</code>
        {" → "}<code className="text-[#0F5499]">python3 score_ml.py</code>, then restart the API.
      </div>
    );
  }
  const m = meta.ml_meta;
  const counts = m.stage_counts || {};
  const delta = m.delta_distribution || {};
  const weights = m.weights_per_class || {};
  const ts = m.as_of ? new Date(m.as_of).toLocaleString() : "—";

  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[16px] font-semibold text-[#0F5499]">
            ML-Optimized Composite — Per Asset Class
          </div>
          <div className="text-[12px] text-[#857F7A] mt-0.5">
            Optuna walk-forward CV on ve_observations · L2 reg toward defaults · {ts}
          </div>
        </div>
        <div className="text-right text-[13px] text-[#66605C]">
          {m.n_total ?? 0} tickers · stages:
          <span className="ml-2 text-[#0F5499]">PM {counts["pre-momentum"] ?? 0}</span>
          <span className="ml-2" style={{ color: C.green }}>Mom {counts.momentum ?? 0}</span>
          <span className="ml-2" style={{ color: C.red }}>Excl {counts.excluded ?? 0}</span>
        </div>
      </div>

      {/* Weights table */}
      <div className="overflow-x-auto">
        <table className="w-full text-[14px] border-collapse">
          <thead>
            <tr className="border-b border-[#E6D9CE] bg-[#F2E5D7]">
              <th className="py-1.5 px-2 text-left text-[#857F7A]">Asset class</th>
              <th className="py-1.5 px-2 text-right text-[#857F7A]">w_TCS</th>
              <th className="py-1.5 px-2 text-right text-[#857F7A]">w_TFS</th>
              <th className="py-1.5 px-2 text-right text-[#857F7A]">w_RSS</th>
              <th className="py-1.5 px-2 text-right text-[#857F7A]">w_URS</th>
              <th className="py-1.5 px-2 text-right text-[#857F7A]">Δ vs default</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-[#E6D9CE]/50 bg-[#FFF1E5]">
              <td className="py-1.5 px-2 text-[#66605C]">Default (baseline)</td>
              <td className="py-1.5 px-2 text-right font-mono text-[#857F7A]">{DEFAULT_WEIGHTS.w_tcs.toFixed(3)}</td>
              <td className="py-1.5 px-2 text-right font-mono text-[#857F7A]">{DEFAULT_WEIGHTS.w_tfs.toFixed(3)}</td>
              <td className="py-1.5 px-2 text-right font-mono text-[#857F7A]">{DEFAULT_WEIGHTS.w_rss.toFixed(3)}</td>
              <td className="py-1.5 px-2 text-right font-mono text-[#857F7A]">{DEFAULT_WEIGHTS.w_urs.toFixed(3)}</td>
              <td className="py-1.5 px-2 text-right text-[#857F7A]">—</td>
            </tr>
            {Object.entries(weights).map(([ac, w]) => {
              const dist = Math.sqrt(
                Object.keys(DEFAULT_WEIGHTS).reduce(
                  (s, k) => s + (w[k as keyof MlWeights] - DEFAULT_WEIGHTS[k as keyof MlWeights]) ** 2,
                  0,
                ),
              );
              return (
                <tr key={ac} className="border-b border-[#E6D9CE]/50">
                  <td className="py-1.5 px-2 text-[#33302E] font-semibold">{ac}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-[#0F5499]">{w.w_tcs.toFixed(3)}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-[#0F5499]">{w.w_tfs.toFixed(3)}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-[#0F5499]">{w.w_rss.toFixed(3)}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-[#0F5499]">{w.w_urs.toFixed(3)}</td>
                  <td className="py-1.5 px-2 text-right font-mono text-[#66605C]">{dist.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Delta vs original Eligibility Gate */}
      <div className="text-[13px] text-[#857F7A] flex flex-wrap gap-x-4 gap-y-1">
        <span>vs original gate:</span>
        <span style={{ color: C.green }}>upgraded → momentum: {delta.upgraded_to_momentum ?? 0}</span>
        <span style={{ color: C.red }}>demoted from momentum: {delta.demoted_from_momentum ?? 0}</span>
        <span className="text-[#857F7A]">unchanged: {delta.unchanged ?? 0}</span>
      </div>

      <div className="text-[12px] text-[#857F7A] italic">
        ★ Note: optimization uses ve_observations forward-21d returns with L2 regularization toward defaults.
        OOS lift vs default is typically small or negative — defaults are already well-tuned, and ML weights
        primarily serve to <em className="text-[#857F7A]">re-bucket</em> tickers into pre-momentum/momentum/excluded
        based on the alternate weighting. The 5-agent Pre-Momentum framework is unchanged (signal-independent of
        Composite weights).
      </div>
    </div>
  );
}

export function PriceDiscoveryMLTab({ filters }: { filters: FilterParams }) {
  const [sub, setSub] = useState(0);
  const [meta, setMeta] = useState<MlMetaResponse | null>(null);
  const [universe, setUniverse] = useState(0);

  useEffect(() => {
    fetchMlMeta().then(setMeta).catch(() => setMeta({ available: false, ml_meta: {} }));
    fetchTableML(filters).then((r) => setUniverse((r.data || []).length)).catch(() => setUniverse(0));
    // Touch ML pre-momentum to warm cache (not strictly needed, sub-tab refetches)
    fetchPreMomentumML().catch(() => {});
  }, [filters]);

  return (
    <div className="space-y-4">
      <MlMetaPanel meta={meta} />

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

      {/* Sub-tab content — same components, mlMode=true to swap fetchers */}
      {sub === 0 && <PreMomentumTab totalUniverse={universe} filterSectors={filters.sectors} mlMode={true} />}
      {sub === 1 && <MomentumTab filters={filters} totalUniverse={universe} mlMode={true} />}
      {sub === 2 && <ExcludedTab filters={filters} totalUniverse={universe} mlMode={true} />}
    </div>
  );
}
