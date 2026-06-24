import { useEffect, useState, useMemo } from "react";
import { fetchTable, fetchOverview, fetchNewPDValidation, type FilterParams } from "../../api/client";
import { MetricCard } from "../shared/MetricCard";
import { CLASS_COLORS, C } from "../../styles/theme";
import {
  Section,
  CompositeFormulaGuide,
  ColumnDefinitions,
  MomentumTable,
  TickerDetail,
  StrategyTable,
  type Ticker,
} from "./MomentumTab";

// ---------------------------------------------------------------------------
// Anti Lag Discovery — Anti-Lag Phase 1
// ---------------------------------------------------------------------------
//
// Surfaces "PROVISIONAL" candidates that haven't passed full Eligibility Gate
// but show strong Pre-Momentum signals:
//   • pre_momentum_score >= 45  (top 10-15% range)
//   • agreement_ratio >= 0.6    (STRONG — 3+ of 5 agents firing)
//   • bullish PM classification
//
// Goal: reduce 10-15 days of confirmation latency by surfacing early signals.
// ---------------------------------------------------------------------------

export function NewPriceDiscoveryTab({ filters, totalUniverse }:
  { filters: FilterParams; totalUniverse?: number }) {
  const [allData, setAllData] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [validation, setValidation] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetchTable(filters),
      fetchOverview(filters),
      fetchNewPDValidation().catch(() => null),
    ]).then(([tableRes, , valRes]) => {
      setAllData(tableRes.data || []);
      setValidation(valRes);
    }).finally(() => setLoading(false));
  }, [filters]);

  // Only ProvisionalPM tier — Pre-Mom strong + not yet Composite>=55
  const provisional = useMemo(
    () => allData
      .filter((t) => t.eligibility_tier === "ProvisionalPM")
      .sort((a, b) => (b.pre_momentum_score || 0) - (a.pre_momentum_score || 0)) as Ticker[],
    [allData]
  );
  const etf = useMemo(() => provisional.filter((t) => t.asset_type === "ETF"), [provisional]);
  const stock = useMemo(() => provisional.filter((t) => t.asset_type === "Stock"), [provisional]);

  const classDist = useMemo(() => {
    const map: Record<string, number> = {};
    for (const t of provisional) {
      map[t.classification] = (map[t.classification] || 0) + 1;
    }
    return Object.entries(map).sort((a, b) => b[1] - a[1]);
  }, [provisional]);

  const selectedTicker = useMemo(
    () => (selected ? allData.find((t) => t.ticker === selected) ?? null : null),
    [selected, allData]
  );

  if (loading) return <div className="text-[#857F7A] p-8">Loading new price discovery data...</div>;

  const avgPM = provisional.length
    ? provisional.reduce((s, t: any) => s + (t.pre_momentum_score || 0), 0) / provisional.length
    : 0;
  const avgComp = provisional.length
    ? provisional.reduce((s, t) => s + t.composite, 0) / provisional.length
    : 0;

  return (
    <div className="space-y-5">
      {/* ── Intro Banner (PROVISIONAL 의미 설명) ── */}
      <div className="bg-gradient-to-r from-[#1a1530] to-[#FFF1E5] rounded-lg border border-[#C9B8DC]/40 p-3">
        <div className="flex items-baseline justify-between mb-1.5">
          <h3 className="text-[16px] font-bold text-[#7D5BA6]">🚀 Anti Lag Discovery — Phase 1</h3>
          <span className="text-[12px] text-[#857F7A]">Pre-Momentum 강한 신호 종목의 조기 surface (lag 10-15일 단축)</span>
        </div>
        <p className="text-[13px] text-[#66605C] leading-relaxed">
          Eligibility Gate (Composite ≥ 55)를 아직 통과하지 못한 종목이지만,
          <span className="text-[#7D5BA6] font-semibold"> Pre-Momentum Score ≥ 45 + agreement_ratio ≥ 0.6 (STRONG)</span> 충족.
          기존 Momentum 탭이 확인된 추세를 보여준다면, 이 탭은
          <span className="text-[#0D7680]"> 아직 추세 확립 전이지만 5-agent framework가 강하게 합의한 forward-looking 후보</span> 표시.
          진입 시 PROVISIONAL 태그로 인지 — false positive risk 존재, sizing 보수적 권고.
        </p>
      </div>

      {/* ── KPIs ── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Provisional" value={provisional.length} sub={`/ ${totalUniverse || allData.length} universe`} />
        <MetricCard label="ETF" value={etf.length} sub="early signals" />
        <MetricCard label="Stock" value={stock.length} sub="early signals" />
        <MetricCard label="Avg PM Score" value={avgPM.toFixed(1)} sub="provisional avg" />
        <MetricCard label="Avg Composite" value={avgComp.toFixed(1)} sub="below eligibility (55)" />
      </div>

      {/* ── Classification Breakdown (inline) ── */}
      {classDist.length > 0 && (
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
      )}

      {/* ── Empty state ── */}
      {provisional.length === 0 && (
        <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-6 text-center">
          <div className="text-[16px] text-[#66605C] mb-2">현재 PROVISIONAL 기준 충족 종목 없음</div>
          <div className="text-[13px] text-[#857F7A] max-w-2xl mx-auto leading-relaxed">
            조건: pre_momentum_score ≥ 45 AND agreement_ratio ≥ 0.6 (STRONG) AND 아직 eligible=False.
            시장 전체가 모멘텀 확립 단계이거나 PM 신호가 약한 시기일 수 있음.
            Pre-Momentum 탭에서 wider universe 확인 가능.
          </div>
        </div>
      )}

      {provisional.length > 0 && (
        <>
          {/* ── Composite formula + column defs ── */}
          <CompositeFormulaGuide />
          <ColumnDefinitions />

          {/* ── All Provisional Table ── */}
          <Section title={`PROVISIONAL — Pre-Mom Strong Candidates (sorted by PM score)`} defaultOpen badge={`${provisional.length} tickers`}>
            <MomentumTable rows={provisional} onSelect={setSelected} />
          </Section>

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
          <Section title="Hedge Strategy Scores — Provisional 종목" badge={`${provisional.length} tickers × 8 strategies`}>
            <p className="text-[13px] text-[#857F7A] mb-3">
              Pre-Momentum 강세 신호 받은 종목들의 8개 hedge strategy 점수. 보조 confirmation 용도.
            </p>
            <StrategyTable rows={provisional} />
          </Section>
        </>
      )}

      {/* ── Validation Section (SVE-style historical tier performance) ── */}
      {validation && !validation.error && (
        <Section
          title="📊 PROVISIONAL Tier Validation — Historical Forward Performance"
          defaultOpen={false}
          badge={`${validation.summary?.total_observations || 0} observations`}
        >
          <ValidationContent validation={validation} />
        </Section>
      )}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// Validation Content — tier forward performance + promotion stats
// ═════════════════════════════════════════════════════════════════════════════

const TIER_LABELS: Record<string, string> = {
  EligibleMomentum: "Eligible Momentum",
  ProvisionalPM_proxy: "PROVISIONAL (proxy)",
  PreMomentum: "Pre-Momentum",
  Excluded: "Excluded",
  Other: "Other",
};
const TIER_COLORS: Record<string, string> = {
  EligibleMomentum: C.green,
  ProvisionalPM_proxy: "#7D5BA6",   // purple — highlight
  PreMomentum: C.cyan,
  Excluded: C.red,
  Other: C.gray,
};

function ValidationContent({ validation }: { validation: any }) {
  const tiers: any[] = validation.tier_summary || [];
  const promo = validation.promotion_stats || {};
  const meth = validation.methodology || {};
  // Filter out empty tiers
  const visible = tiers.filter((t) => t.n_observations > 0);
  // Get ProvisionalPM proxy for emphasis
  const provTier = visible.find((t) => t.tier === "ProvisionalPM_proxy");

  return (
    <div className="space-y-4">
      {/* ── Methodology note ── */}
      <div className="bg-[#FBEEE3]/80 rounded p-3 border border-[#E6D9CE]">
        <div className="text-[13px] text-[#66605C] leading-relaxed">
          <div className="font-semibold text-[#7D5BA6] mb-1">검증 방식 (proxy validation)</div>
          {meth.note || ""}
          <div className="mt-1 text-[#857F7A]">
            <span className="font-mono text-[12px]">{meth.provisional_proxy_definition}</span>
          </div>
        </div>
      </div>

      {/* ── Promotion Stats — sequential snapshot transitions ── */}
      <div className="bg-[#FBEEE3]/80 rounded p-3 border border-[#E6D9CE]">
        <h5 className="text-[13px] font-bold text-[#7D5BA6] mb-2">
          ⤴ PROVISIONAL → next-snapshot tier transition
        </h5>
        <div className="grid grid-cols-3 gap-3 text-[13px]">
          <div className="bg-[#FFF1E5] rounded p-2 border border-[#A8CDB6]/40">
            <div className="text-[12px] text-[#857F7A] mb-1">Promoted to Eligible</div>
            <div className="text-[18px] font-bold" style={{ color: C.green }}>
              {promo.promoted_to_eligible_pct ?? "—"}%
            </div>
            <div className="text-[12px] text-[#857F7A] mt-0.5 font-mono">
              {promo.promoted_to_eligible ?? "—"} / {promo.total_provisional_transitions ?? "—"}
            </div>
          </div>
          <div className="bg-[#FFF1E5] rounded p-2 border border-[#E6D9CE]">
            <div className="text-[12px] text-[#857F7A] mb-1">Stayed / Neutral</div>
            <div className="text-[18px] font-bold" style={{ color: C.gray }}>
              {promo.stayed_or_demoted_neutral_pct ?? "—"}%
            </div>
            <div className="text-[12px] text-[#857F7A] mt-0.5 font-mono">
              {promo.stayed_or_demoted_neutral ?? "—"} / {promo.total_provisional_transitions ?? "—"}
            </div>
          </div>
          <div className="bg-[#FFF1E5] rounded p-2 border border-[#E0AAAA]/40">
            <div className="text-[12px] text-[#857F7A] mb-1">Demoted to Excluded</div>
            <div className="text-[18px] font-bold" style={{ color: C.red }}>
              {promo.demoted_to_excluded_pct ?? "—"}%
            </div>
            <div className="text-[12px] text-[#857F7A] mt-0.5 font-mono">
              {promo.demoted_to_excluded ?? "—"} / {promo.total_provisional_transitions ?? "—"}
            </div>
          </div>
        </div>
        <div className="text-[12px] text-[#857F7A] mt-2 leading-relaxed">
          PROVISIONAL이 다음 bi-weekly 시점에 EligibleMomentum으로 승격하는 비율은 Pre-Mom 신호의 forward-looking 가치를 직접 측정.
          높은 promotion rate = anti-lag 효과 강함.
        </div>
      </div>

      {/* ── Forward returns per tier — comparison table ── */}
      <div>
        <h5 className="text-[13px] font-bold text-[#7D5BA6] mb-2">
          ▸ Forward Return Comparison — by Tier
        </h5>
        <div className="overflow-x-auto rounded border border-[#E6D9CE]">
          <table className="w-full text-[13px]">
            <thead className="bg-[#FFF1E5] text-[#857F7A] uppercase tracking-wide text-[12px]">
              <tr>
                <th className="px-2 py-1.5 text-left">Tier</th>
                <th className="px-2 py-1.5 text-right">n</th>
                <th className="px-2 py-1.5 text-right">5d avg%</th>
                <th className="px-2 py-1.5 text-right">21d avg%</th>
                <th className="px-2 py-1.5 text-right">63d avg%</th>
                <th className="px-2 py-1.5 text-right">126d avg%</th>
                <th className="px-2 py-1.5 text-right">21d Pos%</th>
                <th className="px-2 py-1.5 text-right">21d Excess vs SPY</th>
                <th className="px-2 py-1.5 text-right">63d Sharpe</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((t) => {
                const isProv = t.tier === "ProvisionalPM_proxy";
                const color = TIER_COLORS[t.tier] || C.gray;
                return (
                  <tr key={t.tier} className={`border-t border-[#E6D9CE]/50 ${isProv ? "bg-[#EFE9F5]/10" : ""}`}>
                    <td className="px-2 py-1.5">
                      <span className="font-bold" style={{ color }}>
                        {TIER_LABELS[t.tier] || t.tier}
                        {isProv && <span className="ml-1 text-[11px]">★</span>}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-[#33302E]">
                      {t.n_observations.toLocaleString()}
                    </td>
                    {[5, 21, 63, 126].map((w) => {
                      const v = t[`fwd_${w}d_avg`];
                      return (
                        <td key={w} className="px-2 py-1.5 text-right font-mono"
                            style={{ color: v == null ? C.gray : v > 0 ? C.green : C.red }}>
                          {v != null ? (v >= 0 ? "+" : "") + v.toFixed(2) : "—"}
                        </td>
                      );
                    })}
                    <td className="px-2 py-1.5 text-right font-mono"
                        style={{ color: (t.fwd_21d_pos_pct || 0) >= 60 ? C.green : (t.fwd_21d_pos_pct || 0) >= 50 ? C.yellow : C.red }}>
                      {t.fwd_21d_pos_pct != null ? `${t.fwd_21d_pos_pct.toFixed(1)}%` : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono"
                        style={{ color: (t.fwd_21d_excess ?? 0) > 0 ? C.green : C.red }}>
                      {t.fwd_21d_excess != null ? (t.fwd_21d_excess >= 0 ? "+" : "") + t.fwd_21d_excess.toFixed(2) : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-[#33302E]">
                      {t.fwd_63d_sharpe_like != null ? t.fwd_63d_sharpe_like.toFixed(2) : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── ProvisionalPM key insights ── */}
      {provTier && (
        <div className="bg-[#EFE9F5]/10 rounded p-3 border border-[#C9B8DC]/40">
          <h5 className="text-[13px] font-bold text-[#7D5BA6] mb-2">
            🔬 PROVISIONAL Tier — Key Metrics
          </h5>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-[13px]">
            <div className="bg-[#FFF1E5] rounded p-2">
              <div className="text-[9.5px] text-[#857F7A]">n observations</div>
              <div className="font-mono font-bold text-[#33302E]">{provTier.n_observations}</div>
            </div>
            <div className="bg-[#FFF1E5] rounded p-2">
              <div className="text-[9.5px] text-[#857F7A]">21d Avg Return</div>
              <div className="font-mono font-bold" style={{ color: (provTier.fwd_21d_avg ?? 0) > 0 ? C.green : C.red }}>
                {provTier.fwd_21d_avg != null ? (provTier.fwd_21d_avg >= 0 ? "+" : "") + provTier.fwd_21d_avg.toFixed(2) + "%" : "—"}
              </div>
            </div>
            <div className="bg-[#FFF1E5] rounded p-2">
              <div className="text-[9.5px] text-[#857F7A]">21d Positive %</div>
              <div className="font-mono font-bold" style={{ color: (provTier.fwd_21d_pos_pct ?? 0) >= 60 ? C.green : C.yellow }}>
                {provTier.fwd_21d_pos_pct != null ? provTier.fwd_21d_pos_pct.toFixed(1) + "%" : "—"}
              </div>
            </div>
            <div className="bg-[#FFF1E5] rounded p-2">
              <div className="text-[9.5px] text-[#857F7A]">63d Avg Return</div>
              <div className="font-mono font-bold" style={{ color: (provTier.fwd_63d_avg ?? 0) > 0 ? C.green : C.red }}>
                {provTier.fwd_63d_avg != null ? (provTier.fwd_63d_avg >= 0 ? "+" : "") + provTier.fwd_63d_avg.toFixed(2) + "%" : "—"}
              </div>
            </div>
            <div className="bg-[#FFF1E5] rounded p-2">
              <div className="text-[9.5px] text-[#857F7A]">126d Avg Return</div>
              <div className="font-mono font-bold" style={{ color: (provTier.fwd_126d_avg ?? 0) > 0 ? C.green : C.red }}>
                {provTier.fwd_126d_avg != null ? (provTier.fwd_126d_avg >= 0 ? "+" : "") + provTier.fwd_126d_avg.toFixed(2) + "%" : "—"}
              </div>
            </div>
          </div>
          <div className="text-[10.5px] text-[#66605C] mt-2 leading-relaxed">
            {(() => {
              const elig = visible.find((t) => t.tier === "EligibleMomentum");
              if (!elig || !provTier) return null;
              const pp = provTier.fwd_21d_pos_pct ?? 0;
              const ep = elig.fwd_21d_pos_pct ?? 0;
              const pa = provTier.fwd_21d_avg ?? 0;
              const ea = elig.fwd_21d_avg ?? 0;
              const promoteRate = promo.promoted_to_eligible_pct ?? 0;
              let insight = "";
              if (pp >= ep) {
                insight = `★ PROVISIONAL의 21d 양수 비율 ${pp.toFixed(1)}%는 Eligible Momentum ${ep.toFixed(1)}%보다 ${(pp-ep).toFixed(1)}p 높음 — 'win-rate' 측면에서 우수. `;
              } else {
                insight = `PROVISIONAL의 21d 양수 비율 ${pp.toFixed(1)}% < Eligible ${ep.toFixed(1)}%. `;
              }
              insight += `21d 평균 수익률은 ${pa.toFixed(2)}% (Eligible: ${ea.toFixed(2)}%) — magnitude는 일반적으로 Eligible Momentum이 높으나 win-rate는 비슷. `;
              insight += `Promotion rate ${promoteRate.toFixed(1)}%는 PROVISIONAL이 단기 내 Eligible로 승격될 가능성을 시사 — anti-lag 효과 확인.`;
              return insight;
            })()}
          </div>
        </div>
      )}
    </div>
  );
}
