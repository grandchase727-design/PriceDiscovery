import { useEffect, useState, useMemo } from "react";
import { fetchTable, fetchOverview, fetchNewPDv2Validation, type FilterParams } from "../../api/client";
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
// Sector Discovery — Sector-Segmented Selection
// ---------------------------------------------------------------------------
//
// Per-sector top-N picks (composite ≥ 40 + bullish classification, ranked
// within sector). Forces diversification across all equity sectors regardless
// of universe-wide ranking. Captures sector-best picks that universe-wide
// approach might miss (e.g., Defensive top in a Risk-On environment).
//
// Goal: diversification + sector rotation alpha (note: lag reduction limited).
// ---------------------------------------------------------------------------

export function New2PriceDiscoveryTab({ filters, totalUniverse }:
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
      fetchNewPDv2Validation().catch(() => null),
    ]).then(([tableRes, , valRes]) => {
      setAllData(tableRes.data || []);
      setValidation(valRes);
    }).finally(() => setLoading(false));
  }, [filters]);

  // Sector-segmented eligible
  const sectorPicks = useMemo(
    () => allData
      .filter((t) => t.sector_segmented_eligible)
      .sort((a, b) => {
        // Group by sector, then by sector_rank
        const secCmp = (a.sector || "").localeCompare(b.sector || "");
        if (secCmp !== 0) return secCmp;
        return (a.sector_rank || 999) - (b.sector_rank || 999);
      }) as Ticker[],
    [allData]
  );

  // Sort by composite for the main table (top picks regardless of sector)
  const sortedByComp = useMemo(
    () => [...sectorPicks].sort((a, b) => b.composite - a.composite),
    [sectorPicks]
  );

  const etf = useMemo(() => sortedByComp.filter((t) => t.asset_type === "ETF"), [sortedByComp]);
  const stock = useMemo(() => sortedByComp.filter((t) => t.asset_type === "Stock"), [sortedByComp]);

  // Group picks by sector for sector-by-sector display
  const bySector = useMemo(() => {
    const groups: Record<string, Ticker[]> = {};
    for (const t of sectorPicks) {
      const sec = (t as any).sector || "Other";
      if (!groups[sec]) groups[sec] = [];
      groups[sec].push(t);
    }
    return Object.entries(groups).sort((a, b) => b[1].length - a[1].length);
  }, [sectorPicks]);

  // Classification breakdown
  const classDist = useMemo(() => {
    const map: Record<string, number> = {};
    for (const t of sectorPicks) map[t.classification] = (map[t.classification] || 0) + 1;
    return Object.entries(map).sort((a, b) => b[1] - a[1]);
  }, [sectorPicks]);

  // Eligibility tier breakdown
  const tierDist = useMemo(() => {
    const map: Record<string, number> = {};
    for (const t of allData) {
      const tier = (t as any).eligibility_tier_v2 || "Neither";
      map[tier] = (map[tier] || 0) + 1;
    }
    return map;
  }, [allData]);

  const selectedTicker = useMemo(
    () => (selected ? allData.find((t) => t.ticker === selected) ?? null : null),
    [selected, allData]
  );

  if (loading) return <div className="text-[#857F7A] p-8">Loading sector-segmented data...</div>;

  const avgComp = sectorPicks.length
    ? sectorPicks.reduce((s, t) => s + t.composite, 0) / sectorPicks.length
    : 0;

  return (
    <div className="space-y-5">
      {/* ── Intro Banner ── */}
      <div className="bg-gradient-to-r from-[#1530_30] to-[#FFF1E5] rounded-lg border border-emerald-700/40 p-3">
        <div className="flex items-baseline justify-between mb-1.5">
          <h3 className="text-[16px] font-bold text-emerald-300">
            🌐 Sector Discovery — Sector-Segmented Selection
          </h3>
          <span className="text-[12px] text-[#857F7A]">
            섹터별 독립 top-5 선별 · diversification + sector-best 강제 포착
          </span>
        </div>
        <p className="text-[13px] text-[#66605C] leading-relaxed">
          전체 universe에서 ranking하는 기존 방식과 달리,
          <span className="text-emerald-300 font-semibold"> 각 sector 내에서 독립적으로 top-5</span> 종목 선별
          (bullish classification + composite ≥ 40).
          Universe-wide에서 누락되는 <span className="text-[#0D7680]">sector-best 종목</span>까지 surface하여 diversification 강화.
          ⚠️ <span className="text-[#B85C00]">후행성 해결 효과는 제한적</span> (composite 기반 ranking은 동일).
          진짜 가치는 diversification + sector rotation alpha 포착.
        </p>
      </div>

      {/* ── KPIs ── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Total Picks" value={sectorPicks.length} sub={`/ ${totalUniverse || allData.length} universe`} />
        <MetricCard label="Sectors Covered" value={bySector.length} sub="distinct sectors" />
        <MetricCard label="ETF" value={etf.length} sub="sector picks" />
        <MetricCard label="Stock" value={stock.length} sub="sector picks" />
        <MetricCard label="Avg Composite" value={avgComp.toFixed(1)} sub="picks avg" />
      </div>

      {/* ── Eligibility tier v2 (Universe vs Sector overlap) ── */}
      <div className="bg-[#FBEEE3]/80 rounded p-3 border border-[#E6D9CE]">
        <h4 className="text-[13px] font-bold text-[#66605C] mb-2">
          ⌬ Eligibility Cross-Reference (Universe-wide vs Sector-segmented)
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[13px]">
          <div className="bg-emerald-900/20 rounded p-2 border border-emerald-700/40">
            <div className="text-[12px] text-[#857F7A] mb-1">Both Eligible</div>
            <div className="text-[18px] font-bold text-emerald-300">{tierDist["BothEligible"] || 0}</div>
            <div className="text-[12px] text-[#857F7A] mt-0.5">High-conviction</div>
          </div>
          <div className="bg-[#E3EEF5]/20 rounded p-2 border border-[#9CC3D5]/40">
            <div className="text-[12px] text-[#857F7A] mb-1">Universe Only</div>
            <div className="text-[18px] font-bold text-[#0D7680]">{tierDist["UniverseOnly"] || 0}</div>
            <div className="text-[12px] text-[#857F7A] mt-0.5">Sector mediocre</div>
          </div>
          <div className="bg-[#EFE9F5]/20 rounded p-2 border border-[#C9B8DC]/40">
            <div className="text-[12px] text-[#857F7A] mb-1">Sector Only ★</div>
            <div className="text-[18px] font-bold text-[#7D5BA6]">{tierDist["SectorOnly"] || 0}</div>
            <div className="text-[12px] text-[#857F7A] mt-0.5">Hidden sector-best</div>
          </div>
          <div className="bg-[#FBEEE3]/20 rounded p-2 border border-[#E6D9CE]">
            <div className="text-[12px] text-[#857F7A] mb-1">Neither</div>
            <div className="text-[18px] font-bold text-[#66605C]">{tierDist["Neither"] || 0}</div>
            <div className="text-[12px] text-[#857F7A] mt-0.5">Excluded</div>
          </div>
        </div>
        <div className="text-[12px] text-[#857F7A] mt-2 leading-relaxed">
          <span className="text-[#7D5BA6]">Sector Only</span>는 universe ranking에서는 빠지나 sector 내 top-3에 포함된 종목 —
          기존 Momentum 탭에서는 보이지 않는 새로운 후보군.
        </div>
      </div>

      {/* ── Classification Breakdown ── */}
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
      {sectorPicks.length === 0 && (
        <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-6 text-center">
          <div className="text-[16px] text-[#66605C] mb-2">현재 sector-segmented picks 없음</div>
          <div className="text-[13px] text-[#857F7A] max-w-2xl mx-auto">
            조건: 각 sector 내 bullish classification + composite ≥ 40 → top-5.
            데이터 로딩 중이거나 모든 sector가 약세인 시기일 수 있음.
          </div>
        </div>
      )}

      {sectorPicks.length > 0 && (
        <>
          <CompositeFormulaGuide />
          <ColumnDefinitions />

          {/* ── All Sector Picks (composite-sorted) ── */}
          <Section
            title={`All Sector Picks (composite-sorted) — ${sectorPicks.length} tickers across ${bySector.length} sectors`}
            defaultOpen
            badge={`${sectorPicks.length}`}
          >
            <MomentumTable rows={sortedByComp} onSelect={setSelected} />
          </Section>

          {/* ── Per-Sector Picks (sector-grouped display) ── */}
          <Section
            title="Per-Sector Top Picks (grouped)"
            defaultOpen={false}
            badge={`${bySector.length} sectors`}
          >
            <div className="space-y-3">
              {bySector.map(([sec, picks]) => (
                <div key={sec} className="bg-[#FBEEE3]/80 rounded border border-[#E6D9CE]">
                  <div className="px-3 py-2 border-b border-[#E6D9CE] flex items-baseline justify-between bg-[#FFF1E5]">
                    <h5 className="text-[14px] font-bold text-emerald-300">
                      🏛 {sec}
                    </h5>
                    <span className="text-[12px] text-[#857F7A]">{picks.length} picks</span>
                  </div>
                  <div className="p-2">
                    {picks.map((t) => (
                      <div
                        key={t.ticker}
                        onClick={() => setSelected(t.ticker)}
                        className="grid grid-cols-[32px_minmax(180px,2fr)_140px_60px_70px_70px_minmax(160px,1fr)] gap-2 items-center text-[11.5px] py-1 px-2 border-b border-[#E6D9CE]/40 hover:bg-[#0e1626] cursor-pointer"
                      >
                        <div className="font-bold tabular-nums text-emerald-300">#{(t as any).sector_rank}</div>
                        <div className="truncate">
                          <span className="font-semibold text-[#33302E]">{(t.name || "").slice(0, 32) || t.ticker}</span>
                          <span className="text-[#857F7A]"> ({t.ticker})</span>
                        </div>
                        <div className="font-mono text-[13px] text-[#33302E] truncate" title={t.classification}>{t.classification}</div>
                        <div className="font-mono text-right tabular-nums" style={{ color: t.composite >= 60 ? C.green : t.composite >= 40 ? C.gray : C.red }}>
                          {t.composite?.toFixed(0)}
                        </div>
                        <div className="font-mono text-right tabular-nums" style={{ color: (t.ret_1m ?? 0) > 0 ? C.green : (t.ret_1m ?? 0) < 0 ? C.red : C.gray }}>
                          {t.ret_1m != null ? `${t.ret_1m >= 0 ? "+" : ""}${t.ret_1m.toFixed(1)}%` : "—"}
                        </div>
                        <div className="font-mono text-right tabular-nums text-[#66605C]">
                          {(t as any).sector_pct_rank != null ? `${(t as any).sector_pct_rank.toFixed(0)}p` : "—"}
                        </div>
                        <div className="text-[12px] text-[#857F7A] truncate">
                          {(t as any).eligibility_tier_v2 === "BothEligible" && <span className="text-emerald-300">✓ Universe + Sector</span>}
                          {(t as any).eligibility_tier_v2 === "SectorOnly" && <span className="text-[#7D5BA6]">★ Sector-only (hidden)</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
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
          <Section title="Hedge Strategy Scores — Sector Picks" badge={`${sectorPicks.length} tickers × 8 strategies`}>
            <p className="text-[13px] text-[#857F7A] mb-3">
              Sector-segmented picks의 8개 hedge strategy 점수. 보조 confirmation 용도.
            </p>
            <StrategyTable rows={sortedByComp} />
          </Section>
        </>
      )}

      {/* ── Validation Section ── */}
      {validation && !validation.error && (
        <Section
          title="📊 Sector-Segmented Validation — Historical Forward Performance"
          defaultOpen={false}
          badge={`${validation.summary?.total_observations || 0} observations`}
        >
          <ValidationContentV2 validation={validation} />
        </Section>
      )}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════════
// Validation Content (V2) — Sector-segmented vs Universe-wide comparison
// ═════════════════════════════════════════════════════════════════════════════

const TIER_LABELS_V2: Record<string, string> = {
  BothEligible: "Both Eligible",
  UniverseOnly: "Universe Only",
  SectorOnly_proxy: "SECTOR Only (proxy)",
  OtherBullish: "Other Bullish",
  OtherExcluded: "Other / Excluded",
};
const TIER_COLORS_V2: Record<string, string> = {
  BothEligible: C.green,
  UniverseOnly: C.cyan,
  SectorOnly_proxy: "#7D5BA6",
  OtherBullish: C.yellow,
  OtherExcluded: C.gray,
};

function ValidationContentV2({ validation }: { validation: any }) {
  const tiers: any[] = validation.tier_summary || [];
  const meth = validation.methodology || {};
  const coverage: any[] = validation.sector_coverage || [];
  const visible = tiers.filter((t) => t.n_observations > 0);
  const sectorOnly = visible.find((t) => t.tier === "SectorOnly_proxy");
  const univOnly = visible.find((t) => t.tier === "UniverseOnly");

  return (
    <div className="space-y-4">
      {/* ── Methodology note ── */}
      <div className="bg-[#FBEEE3]/80 rounded p-3 border border-[#E6D9CE]">
        <div className="text-[13px] text-[#66605C] leading-relaxed">
          <div className="font-semibold text-emerald-300 mb-1">검증 방식</div>
          {meth.note || ""}
          <div className="mt-1 text-[#857F7A] text-[12px] font-mono">{meth.sector_top_proxy_definition}</div>
        </div>
      </div>

      {/* ── Forward Return Comparison Table ── */}
      <div>
        <h5 className="text-[13px] font-bold text-emerald-300 mb-2">
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
                const isFocus = t.tier === "SectorOnly_proxy" || t.tier === "BothEligible";
                const color = TIER_COLORS_V2[t.tier] || C.gray;
                return (
                  <tr key={t.tier} className={`border-t border-[#E6D9CE]/50 ${isFocus ? "bg-emerald-900/10" : ""}`}>
                    <td className="px-2 py-1.5">
                      <span className="font-bold" style={{ color }}>
                        {TIER_LABELS_V2[t.tier] || t.tier}
                        {t.tier === "SectorOnly_proxy" && <span className="ml-1 text-[11px]">★</span>}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-[#33302E]">{t.n_observations.toLocaleString()}</td>
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

      {/* ── Sector-Only vs Universe-Only insight ── */}
      {sectorOnly && univOnly && (
        <div className="bg-emerald-900/10 rounded p-3 border border-emerald-700/40">
          <h5 className="text-[13px] font-bold text-emerald-300 mb-2">
            🔬 Sector-Segmented Value Test
          </h5>
          <div className="text-[10.5px] text-[#66605C] leading-relaxed">
            {(() => {
              const so = sectorOnly.fwd_21d_avg ?? 0;
              const uo = univOnly.fwd_21d_avg ?? 0;
              const so_pos = sectorOnly.fwd_21d_pos_pct ?? 0;
              const uo_pos = univOnly.fwd_21d_pos_pct ?? 0;
              const so_63 = sectorOnly.fwd_63d_avg ?? 0;
              const uo_63 = univOnly.fwd_63d_avg ?? 0;
              let insight = `SectorOnly 21d 평균 ${so.toFixed(2)}% (win-rate ${so_pos.toFixed(1)}%) vs UniverseOnly ${uo.toFixed(2)}% (${uo_pos.toFixed(1)}%). `;
              if (so > uo && so_pos > uo_pos) {
                insight += `★ Sector-segmented가 양 지표 모두 우위 — 새로운 alpha 추출 효과 확인. `;
              } else if (so > uo) {
                insight += `Sector-segmented가 magnitude 우위 (win-rate는 비슷) — selective alpha. `;
              } else if (so_pos > uo_pos) {
                insight += `Sector-segmented가 win-rate 우위 (magnitude는 비슷) — safer entries. `;
              } else {
                insight += `Sector-segmented가 명확한 우위 없음 — 단순 diversification 효과만. `;
              }
              insight += `63d 시점: Sector ${so_63.toFixed(2)}% vs Universe ${uo_63.toFixed(2)}%. `;
              return insight;
            })()}
          </div>
        </div>
      )}

      {/* ── Current Sector Coverage ── */}
      {coverage.length > 0 && (
        <div>
          <h5 className="text-[13px] font-bold text-emerald-300 mb-2">
            ▸ Current Sector Coverage — Live Picks
          </h5>
          <div className="overflow-x-auto rounded border border-[#E6D9CE]">
            <table className="w-full text-[13px]">
              <thead className="bg-[#FFF1E5] text-[#857F7A] uppercase tracking-wide text-[12px]">
                <tr>
                  <th className="px-2 py-1.5 text-left">Sector</th>
                  <th className="px-2 py-1.5 text-right">n picks</th>
                  <th className="px-2 py-1.5 text-left">Tickers (composite)</th>
                </tr>
              </thead>
              <tbody>
                {coverage.map((s) => (
                  <tr key={s.sector} className="border-t border-[#E6D9CE]/50">
                    <td className="px-2 py-1.5 font-bold text-emerald-300">{s.sector}</td>
                    <td className="px-2 py-1.5 text-right font-mono text-[#33302E]">{s.n_picks}</td>
                    <td className="px-2 py-1.5 text-[12px] text-[#66605C]">
                      {s.picks.map((p: any, i: number) => (
                        <span key={p.ticker}>
                          <span className="text-[#33302E] font-mono">{p.ticker}</span>
                          <span className="text-[#857F7A]"> ({(p.composite || 0).toFixed(0)})</span>
                          {i < s.picks.length - 1 && <span className="text-[#33302E]"> · </span>}
                        </span>
                      ))}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
