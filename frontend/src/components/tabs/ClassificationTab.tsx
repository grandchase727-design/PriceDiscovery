import { useEffect, useState } from "react";
import { fetchClassificationMeta, fetchClassificationValidation } from "../../api/client";
import { C } from "../../styles/theme";

interface ClassificationMeta {
  available: boolean;
  message?: string;
  as_of?: string;
  n_total?: number;
  n_success?: number;
  n_failure?: number;
  distribution?: {
    by_gics_sector: Record<string, number>;
    by_cap_tier: Record<string, number>;
    by_country: Record<string, number>;
  };
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

export function ClassificationTab() {
  const [meta, setMeta] = useState<ClassificationMeta | null>(null);
  const [val, setVal] = useState<ValidationReport | null>(null);

  useEffect(() => {
    fetchClassificationMeta().then(setMeta).catch(() => setMeta({ available: false }));
    fetchClassificationValidation().then(setVal).catch(() => setVal({ error: "fetch failed" }));
  }, []);

  if (!meta) {
    return <div className="text-gray-500 text-sm p-4">Loading classification metadata…</div>;
  }
  if (!meta.available) {
    return (
      <div className="bg-[#1f2937] border border-gray-700 rounded p-4 text-sm text-gray-400">
        Unified classification not yet generated.
        Run <code className="text-cyan-400">python3 unified_classifier.py</code> and restart the API.
      </div>
    );
  }

  const gics = meta.distribution?.by_gics_sector ?? {};
  const cap = meta.distribution?.by_cap_tier ?? {};
  const country = meta.distribution?.by_country ?? {};
  const ts = meta.as_of ? new Date(meta.as_of).toLocaleString() : "—";

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-[#111827] rounded-lg border border-gray-800 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-cyan-400">
              Unified Classification — Cross-Border (US + KR + ETF)
            </h3>
            <div className="text-[10px] text-gray-500 mt-0.5">
              4-level taxonomy: Country → GICS Sector → GICS Industry → SubTheme &nbsp;·&nbsp; fetched {ts}
            </div>
          </div>
          <div className="text-right text-[11px]">
            <span className="text-gray-400">{meta.n_total} tickers · </span>
            <span className="text-green-400">{meta.n_success} OK</span>
            {(meta.n_failure ?? 0) > 0 && (
              <span className="text-red-400 ml-1">· {meta.n_failure} failed</span>
            )}
          </div>
        </div>
      </div>

      {/* Distribution panels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-[#111827] rounded-lg border border-gray-800 p-3">
          <h4 className="text-xs font-semibold text-gray-400 mb-2">By GICS Sector</h4>
          <div className="space-y-1">
            {Object.entries(gics)
              .sort((a, b) => b[1] - a[1])
              .map(([sec, n]) => (
                <div key={sec} className="flex justify-between text-xs">
                  <span className="text-gray-300">{sec}</span>
                  <span className="font-mono text-cyan-400">{n}</span>
                </div>
              ))}
          </div>
        </div>
        <div className="bg-[#111827] rounded-lg border border-gray-800 p-3">
          <h4 className="text-xs font-semibold text-gray-400 mb-2">By Cap Tier (USD)</h4>
          <div className="space-y-1">
            {CAP_ORDER.filter((t) => t in cap).map((t) => (
              <div key={t} className="flex justify-between text-xs">
                <span className="text-gray-300">{t}</span>
                <span className="font-mono text-cyan-400">{cap[t]}</span>
              </div>
            ))}
          </div>
          <div className="text-[10px] text-gray-600 mt-2">
            MEGA ≥ $200B · LARGE $10-200B · MID $2-10B · SMALL &lt; $2B
          </div>
        </div>
        <div className="bg-[#111827] rounded-lg border border-gray-800 p-3">
          <h4 className="text-xs font-semibold text-gray-400 mb-2">By Listing Country</h4>
          <div className="space-y-1">
            {Object.entries(country)
              .sort((a, b) => b[1] - a[1])
              .map(([c, n]) => (
                <div key={c} className="flex justify-between text-xs">
                  <span className="text-gray-300">{c}</span>
                  <span className="font-mono text-cyan-400">{n}</span>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Validation report */}
      {val && !val.error && (
        <div className="bg-[#111827] rounded-lg border border-gray-800 p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-semibold text-cyan-400">
              Curated vs Auto-GICS Validation
            </h4>
            <div className="text-xs text-gray-400">
              {val.n_agree}/{val.n_compared} agree (
              <span className="font-mono font-bold text-cyan-400">{val.agreement_pct}%</span>)
            </div>
          </div>

          {/* By-country agreement */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            {val.by_country && Object.entries(val.by_country).map(([c, d]) => (
              <div key={c} className="bg-[#1f2937] rounded p-2">
                <div className="text-[10px] text-gray-500 uppercase tracking-wide">{c}</div>
                <div className="text-sm font-bold mt-1"
                     style={{ color: d.agree_pct >= 90 ? C.green : d.agree_pct >= 75 ? C.yellow : C.red }}>
                  {d.agree_pct}%
                </div>
                <div className="text-[10px] text-gray-500 mt-0.5">
                  {d.agree}/{d.compared}
                </div>
              </div>
            ))}
          </div>

          {/* Mismatch table — top 50 by market cap */}
          {(val.mismatches?.length ?? 0) > 0 && (
            <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
              <h5 className="text-xs text-gray-500 mb-1">
                Top mismatches (curated ≠ auto, sorted by market cap) — {val.mismatches!.length} total
              </h5>
              <table className="w-full text-xs border-collapse">
                <thead className="sticky top-0 z-10 bg-[#1f2937]">
                  <tr className="border-b border-gray-700">
                    <th className="py-1.5 px-2 text-left text-gray-500">Ticker</th>
                    <th className="py-1.5 px-2 text-left text-gray-500">Name</th>
                    <th className="py-1.5 px-2 text-center text-gray-500">Country</th>
                    <th className="py-1.5 px-2 text-left text-gray-500">Curated</th>
                    <th className="py-1.5 px-2 text-left text-gray-500">Auto-GICS</th>
                    <th className="py-1.5 px-2 text-left text-gray-500">GICS Industry</th>
                    <th className="py-1.5 px-2 text-right text-gray-500">$B</th>
                    <th className="py-1.5 px-2 text-center text-gray-500">Tier</th>
                  </tr>
                </thead>
                <tbody>
                  {val.mismatches!.slice(0, 100).map((r) => (
                    <tr key={r.ticker} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
                      <td className="py-1 px-2 font-mono text-cyan-400 font-bold">{r.ticker}</td>
                      <td className="py-1 px-2 text-gray-300">{r.name?.slice(0, 30) ?? ""}</td>
                      <td className="py-1 px-2 text-center text-gray-500">{r.country}</td>
                      <td className="py-1 px-2 text-yellow-400">{r.curated}</td>
                      <td className="py-1 px-2 text-orange-400">{r.auto_gics}</td>
                      <td className="py-1 px-2 text-gray-400">{r.auto_industry?.slice(0, 30) ?? ""}</td>
                      <td className="py-1 px-2 text-right font-mono text-gray-400">
                        {r.mktcap_usd_b != null ? `$${r.mktcap_usd_b.toFixed(0)}` : "—"}
                      </td>
                      <td className="py-1 px-2 text-center text-[10px] text-gray-500">{r.cap_tier ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {val?.error && (
        <div className="bg-red-900/20 border border-red-700 rounded p-3 text-xs text-red-400">
          Validation failed: {val.error}
        </div>
      )}
    </div>
  );
}
