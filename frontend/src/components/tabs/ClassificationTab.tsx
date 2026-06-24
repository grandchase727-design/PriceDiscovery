import { useEffect, useMemo, useState } from "react";
import { fetchClassificationMeta, fetchClassificationValidation } from "../../api/client";
import { C } from "../../styles/theme";

interface HierarchyRow {
  sector: string;
  industry_group: string;
  industry: string;
  n_stocks: number;
  total_mktcap_b: number;
  sample_tickers: string[];
  tickers: Array<{
    ticker: string;
    name: string;
    country: string;
    cap_tier: string;
    mktcap_usd_b: number | null;
  }>;
}

interface ClassificationMeta {
  available: boolean;
  message?: string;
  as_of?: string;
  n_total?: number;
  n_success?: number;
  n_failure?: number;
  distribution?: {
    by_universe_category?: Record<string, number>;  // ← matches Universe sub-tab "category" column
    by_universe_theme?: Record<string, number>;     // ← matches Universe sub-tab "theme" column
    by_gics_sector: Record<string, number>;
    by_gics_industry_group?: Record<string, number>;
    by_cap_tier: Record<string, number>;
    by_country: Record<string, number>;
  };
  hierarchy_table?: HierarchyRow[];
}

interface MismatchRow {
  ticker: string;
  name: string;
  country: string;
  curated: string;
  auto_gics: string;
  auto_industry?: string;
  category?: string;
  subtheme?: string;
  mktcap_usd_b?: number;
  cap_tier?: string;
}

interface ValidationReport {
  n_total?: number;
  n_compared?: number;
  n_agree?: number;
  agreement_pct?: number;
  by_country?: Record<string, { compared: number; agree: number; agree_pct: number }>;
  mismatches?: MismatchRow[];
  error?: string;
}

const CAP_ORDER = ["MEGA", "LARGE", "MID", "SMALL", "MICRO"];


// ─────────────────────────────────────────────────────────────────
// DistributionPanel — single distribution card with safe overflow
// ─────────────────────────────────────────────────────────────────

function DistributionPanel({
  title, entries, footer, maxVisible,
}: {
  title: string;
  entries: Array<[string, number]>;
  footer?: string;
  maxVisible?: number;
}) {
  const total = entries.reduce((acc, [, n]) => acc + n, 0);
  const visible = maxVisible ? entries.slice(0, maxVisible) : entries;
  const hidden = entries.length - visible.length;
  return (
    <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-3 min-w-0">
      <div className="flex items-baseline justify-between mb-2 gap-2">
        <h4 className="text-[14px] font-semibold text-[#66605C] truncate">{title}</h4>
        <span className="text-[12px] text-[#857F7A] font-mono shrink-0">total {total}</span>
      </div>
      <div className="space-y-0.5">
        {visible.map(([label, n]) => (
          <div key={label} className="flex items-baseline gap-2 text-[13px]" title={label}>
            <span className="text-[#33302E] truncate min-w-0 flex-1 leading-snug">{label}</span>
            <span className="font-mono text-[#0F5499] shrink-0 tabular-nums">{n}</span>
          </div>
        ))}
        {hidden > 0 && (
          <div className="text-[12px] text-[#857F7A] italic pt-1 border-t border-[#E6D9CE]/40">
            + {hidden} more
          </div>
        )}
      </div>
      {footer && (
        <div className="text-[11px] text-[#857F7A] mt-2 leading-relaxed">{footer}</div>
      )}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────
// HierarchyTable — Sector → Industry Group → Industry drill-down
// ─────────────────────────────────────────────────────────────────

function HierarchyTable({ rows }: { rows: HierarchyRow[] }) {
  const [filterSector, setFilterSector] = useState<string>("All");
  const [filterGroup, setFilterGroup] = useState<string>("All");
  const [expandedKey, setExpandedKey] = useState<string | null>(null);

  const sectors = useMemo(
    () => Array.from(new Set(rows.map((r) => r.sector))).sort(),
    [rows],
  );
  const groups = useMemo(() => {
    const filtered = filterSector === "All" ? rows : rows.filter((r) => r.sector === filterSector);
    return Array.from(new Set(filtered.map((r) => r.industry_group))).sort();
  }, [rows, filterSector]);

  const visibleRows = useMemo(() => {
    return rows.filter((r) => {
      if (filterSector !== "All" && r.sector !== filterSector) return false;
      if (filterGroup !== "All" && r.industry_group !== filterGroup) return false;
      return true;
    });
  }, [rows, filterSector, filterGroup]);

  // Stats
  const totalStocks = visibleRows.reduce((acc, r) => acc + r.n_stocks, 0);
  const totalCap = visibleRows.reduce((acc, r) => acc + r.total_mktcap_b, 0);
  const nIndustries = visibleRows.length;
  const nGroups = new Set(visibleRows.map((r) => r.industry_group)).size;
  const nSectors = new Set(visibleRows.map((r) => r.sector)).size;

  // Group rows by sector for visual grouping
  type Grouped = { sector: string; rows: HierarchyRow[] };
  const grouped: Grouped[] = useMemo(() => {
    const map: Record<string, HierarchyRow[]> = {};
    for (const r of visibleRows) {
      (map[r.sector] = map[r.sector] || []).push(r);
    }
    return Object.entries(map).map(([sector, rs]) => ({ sector, rows: rs }));
  }, [visibleRows]);

  return (
    <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4 space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h4 className="text-[16px] font-semibold text-[#0F5499]">
            Stock Hierarchy — Sector / Industry Group / Industry
          </h4>
          <div className="text-[12px] text-[#857F7A] mt-0.5">
            3-level GICS taxonomy · {nSectors} sectors · {nGroups} industry groups · {nIndustries} industries
            · {totalStocks} stocks · ${totalCap.toFixed(0)}B total cap
          </div>
        </div>
        <div className="flex items-center gap-2">
          <select value={filterSector} onChange={(e) => { setFilterSector(e.target.value); setFilterGroup("All"); }}
                  className="bg-[#F2E5D7] border border-[#E6D9CE] text-[14px] rounded px-2 py-1">
            <option value="All">All sectors ({sectors.length})</option>
            {sectors.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
          <select value={filterGroup} onChange={(e) => setFilterGroup(e.target.value)}
                  className="bg-[#F2E5D7] border border-[#E6D9CE] text-[14px] rounded px-2 py-1"
                  disabled={filterSector === "All" && groups.length > 30}>
            <option value="All">All groups ({groups.length})</option>
            {groups.map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </div>
      </div>

      <div className="overflow-x-auto max-h-[700px] overflow-y-auto">
        <table className="w-full text-[14px] border-collapse table-fixed">
          <colgroup>
            <col className="w-6" />
            <col className="w-[260px]" />
            <col className="w-[260px]" />
            <col className="w-16" />
            <col className="w-20" />
            <col />
          </colgroup>
          <thead className="sticky top-0 z-10 bg-[#F2E5D7]">
            <tr className="border-b border-[#E6D9CE]">
              <th className="py-1.5 px-2 text-left text-[#857F7A]"></th>
              <th className="py-1.5 px-2 text-left text-[#857F7A]">Industry Group</th>
              <th className="py-1.5 px-2 text-left text-[#857F7A]">Industry</th>
              <th className="py-1.5 px-2 text-right text-[#857F7A]">#Stocks</th>
              <th className="py-1.5 px-2 text-right text-[#857F7A]">Total $B</th>
              <th className="py-1.5 px-2 text-left text-[#857F7A]">Top tickers</th>
            </tr>
          </thead>
          <tbody>
            {grouped.flatMap((g) => {
              const sectorTotalStocks = g.rows.reduce((a, r) => a + r.n_stocks, 0);
              const sectorTotalCap = g.rows.reduce((a, r) => a + r.total_mktcap_b, 0);
              const out: any[] = [
                <tr key={`sec-${g.sector}`} className="bg-[#FFF1E5] border-y border-[#9CC3D5]/40">
                  <td colSpan={3} className="py-2 px-2 font-bold text-[#0F5499]">
                    <span className="truncate" title={g.sector}>{g.sector}</span>
                    <span className="ml-3 text-[12px] text-[#857F7A] font-normal whitespace-nowrap">
                      {g.rows.length} industries · {new Set(g.rows.map((r) => r.industry_group)).size} groups
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right font-mono font-bold text-[#0D7680] tabular-nums">
                    {sectorTotalStocks}
                  </td>
                  <td className="py-2 px-2 text-right font-mono font-bold text-[#0D7680] tabular-nums">
                    ${sectorTotalCap.toFixed(0)}
                  </td>
                  <td className="py-2 px-2"></td>
                </tr>,
              ];
              for (const r of g.rows) {
                const key = `${r.sector}-${r.industry_group}-${r.industry}`;
                const expanded = expandedKey === key;
                out.push(
                  <tr key={key}
                      onClick={() => setExpandedKey(expanded ? null : key)}
                      className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/50 cursor-pointer">
                    <td className="py-1 px-2 text-[#857F7A] text-[12px] w-6 align-top">↳</td>
                    <td className="py-1 px-2 text-[#66605C] max-w-[260px]">
                      <div className="truncate" title={r.industry_group}>{r.industry_group}</div>
                    </td>
                    <td className="py-1 px-2 text-[#33302E] max-w-[260px]">
                      <div className="truncate" title={r.industry}>{r.industry}</div>
                    </td>
                    <td className="py-1 px-2 text-right font-mono text-[#33302E] tabular-nums w-16">{r.n_stocks}</td>
                    <td className="py-1 px-2 text-right font-mono text-[#66605C] tabular-nums w-20">
                      ${r.total_mktcap_b.toFixed(0)}
                    </td>
                    <td className="py-1 px-2 font-mono text-[#0F5499] text-[12px] max-w-[280px]">
                      <div className="truncate" title={r.sample_tickers.join(", ")}>
                        {r.sample_tickers.join(", ")}
                        {r.n_stocks > 5 && (
                          <span className="ml-1 text-[#857F7A]">
                            +{r.n_stocks - 5} {expanded ? "▾" : "▸"}
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>,
                );
                if (expanded) {
                  out.push(
                    <tr key={`${key}-exp`} className="bg-[#0a1018]">
                      <td colSpan={6} className="py-2 px-4">
                        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-1 text-[12px]">
                          {r.tickers.map((t) => (
                            <div key={t.ticker} className="flex justify-between px-2 py-0.5 bg-[#FFFFFF] rounded gap-2 min-w-0">
                              <span className="font-mono font-bold text-[#0F5499] shrink-0">{t.ticker}</span>
                              <span className="text-[#857F7A] truncate tabular-nums" title={t.name}>
                                {t.mktcap_usd_b != null ? `$${t.mktcap_usd_b.toFixed(0)}B` : ""}
                              </span>
                            </div>
                          ))}
                        </div>
                      </td>
                    </tr>,
                  );
                }
              }
              return out;
            })}
          </tbody>
        </table>
      </div>
      <div className="text-[12px] text-[#857F7A]">
        Click any industry row to expand the full ticker list.
        Industry Group is the GICS Level 2 rollup (24 groups across 11 sectors).
      </div>
    </div>
  );
}


export function ClassificationTab() {
  const [meta, setMeta] = useState<ClassificationMeta | null>(null);
  const [val, setVal] = useState<ValidationReport | null>(null);

  useEffect(() => {
    fetchClassificationMeta().then(setMeta).catch(() => setMeta({ available: false }));
    fetchClassificationValidation().then(setVal).catch(() => setVal({ error: "fetch failed" }));
  }, []);

  if (!meta) {
    return <div className="text-[#857F7A] text-[16px] p-4">Loading classification metadata…</div>;
  }
  if (!meta.available) {
    return (
      <div className="bg-[#F2E5D7] border border-[#E6D9CE] rounded p-4 text-[16px] text-[#66605C]">
        Unified classification not yet generated.
        Run <code className="text-[#0F5499]">python3 unified_classifier.py</code> and restart the API.
      </div>
    );
  }

  const universeCategory = meta.distribution?.by_universe_category ?? {};
  const universeTheme = meta.distribution?.by_universe_theme ?? {};
  const gics = meta.distribution?.by_gics_sector ?? {};
  const industryGroup = meta.distribution?.by_gics_industry_group ?? {};
  const cap = meta.distribution?.by_cap_tier ?? {};
  const country = meta.distribution?.by_country ?? {};
  const ts = meta.as_of ? new Date(meta.as_of).toLocaleString() : "—";

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-[16px] font-semibold text-[#0F5499]">
              Unified Classification — Cross-Border (US + KR + ETF)
            </h3>
            <div className="text-[12px] text-[#857F7A] mt-0.5">
              4-level taxonomy: Country → GICS Sector → GICS Industry → SubTheme &nbsp;·&nbsp; fetched {ts}
            </div>
          </div>
          <div className="text-right text-[13px]">
            <span className="text-[#66605C]">{meta.n_total} tickers · </span>
            <span className="text-[#0A7D3F]">{meta.n_success} OK</span>
            {(meta.n_failure ?? 0) > 0 && (
              <span className="text-[#CC0000] ml-1">· {meta.n_failure} failed</span>
            )}
          </div>
        </div>
      </div>

      {/* Row 1 — Universe taxonomy (matches Appendix → Universe tab columns) */}
      <div>
        <div className="text-[13px] text-[#857F7A] uppercase tracking-wide mb-1.5">
          Universe taxonomy <span className="text-[#33302E]">— matches Universe sub-tab</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <DistributionPanel
            title="By Category (normalized — STK_ stripped, intl→sector)"
            entries={Object.entries(universeCategory).sort((a, b) => b[1] - a[1])}
            maxVisible={20}
            footer="Stocks: Technology / Healthcare / etc. (Korean stocks remapped via GICS). ETFs: EQ_*/FI_*/Commodities/etc."
          />
          <DistributionPanel
            title="By SubTheme (top 20)"
            entries={Object.entries(universeTheme).sort((a, b) => b[1] - a[1])}
            maxVisible={20}
            footer="105 unified subthemes (cross-border: ETFs and stocks share same theme namespace)"
          />
        </div>
      </div>

      {/* Row 2 — GICS taxonomy (cross-border standard) */}
      <div>
        <div className="text-[13px] text-[#857F7A] uppercase tracking-wide mb-1.5">
          GICS taxonomy <span className="text-[#33302E]">— global standard for stocks (ETFs not included)</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          <DistributionPanel
            title="By GICS Sector (11)"
            entries={Object.entries(gics).filter(([k]) => k !== "Unknown").sort((a, b) => b[1] - a[1])}
          />
          <DistributionPanel
            title="By Industry Group (24)"
            entries={Object.entries(industryGroup).filter(([k]) => k !== "Unknown").sort((a, b) => b[1] - a[1])}
            maxVisible={12}
          />
          <DistributionPanel
            title="By Cap Tier (USD)"
            entries={CAP_ORDER.filter((t) => t in cap).map((t) => [t, cap[t]] as [string, number])}
            footer="MEGA ≥ $200B · LARGE $10-200B · MID $2-10B · SMALL < $2B"
          />
          <DistributionPanel
            title="By Listing Country"
            entries={Object.entries(country).sort((a, b) => b[1] - a[1])}
          />
        </div>
      </div>

      {/* Hierarchy table — Sector → Industry Group → Industry drill-down */}
      <HierarchyTable rows={meta.hierarchy_table ?? []} />

      {/* Validation report */}
      {val && !val.error && (
        <div className="bg-[#FFFFFF] rounded-lg border border-[#E6D9CE] p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-[16px] font-semibold text-[#0F5499]">
              Curated vs Auto-GICS Validation
            </h4>
            <div className="text-[14px] text-[#66605C]">
              {val.n_agree}/{val.n_compared} agree (
              <span className="font-mono font-bold text-[#0F5499]">{val.agreement_pct}%</span>)
            </div>
          </div>

          {/* By-country agreement */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            {val.by_country && Object.entries(val.by_country).map(([c, d]) => (
              <div key={c} className="bg-[#F2E5D7] rounded p-2">
                <div className="text-[12px] text-[#857F7A] uppercase tracking-wide">{c}</div>
                <div className="text-[16px] font-bold mt-1"
                     style={{ color: d.agree_pct >= 90 ? C.green : d.agree_pct >= 75 ? C.yellow : C.red }}>
                  {d.agree_pct}%
                </div>
                <div className="text-[12px] text-[#857F7A] mt-0.5">
                  {d.agree}/{d.compared}
                </div>
              </div>
            ))}
          </div>

          {/* Mismatch table — top 50 by market cap */}
          {(val.mismatches?.length ?? 0) > 0 && (
            <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
              <h5 className="text-[14px] text-[#857F7A] mb-1">
                Top mismatches (curated ≠ auto, sorted by market cap) — {val.mismatches!.length} total
              </h5>
              <table className="w-full text-[14px] border-collapse table-fixed">
                <colgroup>
                  <col className="w-20" />
                  <col className="w-48" />
                  <col className="w-16" />
                  <col className="w-44" />
                  <col className="w-44" />
                  <col className="w-48" />
                  <col className="w-16" />
                  <col className="w-14" />
                </colgroup>
                <thead className="sticky top-0 z-10 bg-[#F2E5D7]">
                  <tr className="border-b border-[#E6D9CE]">
                    <th className="py-1.5 px-2 text-left text-[#857F7A]">Ticker</th>
                    <th className="py-1.5 px-2 text-left text-[#857F7A]">Name</th>
                    <th className="py-1.5 px-2 text-center text-[#857F7A]">Country</th>
                    <th className="py-1.5 px-2 text-left text-[#857F7A]">Curated</th>
                    <th className="py-1.5 px-2 text-left text-[#857F7A]">Auto-GICS</th>
                    <th className="py-1.5 px-2 text-left text-[#857F7A]">GICS Industry</th>
                    <th className="py-1.5 px-2 text-right text-[#857F7A]">$B</th>
                    <th className="py-1.5 px-2 text-center text-[#857F7A]">Tier</th>
                  </tr>
                </thead>
                <tbody>
                  {val.mismatches!.slice(0, 100).map((r) => (
                    <tr key={r.ticker} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/30">
                      <td className="py-1 px-2 font-mono text-[#0F5499] font-bold">{r.ticker}</td>
                      <td className="py-1 px-2 text-[#33302E] truncate" title={r.name}>{r.name}</td>
                      <td className="py-1 px-2 text-center text-[#857F7A]">{r.country}</td>
                      <td className="py-1 px-2 text-[#B85C00] truncate" title={r.curated}>{r.curated}</td>
                      <td className="py-1 px-2 text-[#C2701C] truncate" title={r.auto_gics}>{r.auto_gics}</td>
                      <td className="py-1 px-2 text-[#66605C] truncate" title={r.auto_industry ?? ""}>
                        {r.auto_industry ?? ""}
                      </td>
                      <td className="py-1 px-2 text-right font-mono text-[#66605C] tabular-nums">
                        {r.mktcap_usd_b != null ? `$${r.mktcap_usd_b.toFixed(0)}` : "—"}
                      </td>
                      <td className="py-1 px-2 text-center text-[12px] text-[#857F7A]">{r.cap_tier ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {val?.error && (
        <div className="bg-[#F7E3E3]/20 border border-[#E0AAAA] rounded p-3 text-[14px] text-[#CC0000]">
          Validation failed: {val.error}
        </div>
      )}
    </div>
  );
}
