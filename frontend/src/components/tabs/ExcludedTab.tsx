import React, { useEffect, useState, useMemo } from "react";
import { fetchTable, fetchTableML, type FilterParams } from "../../api/client";
import { MetricCard } from "../shared/MetricCard";
import { CLASS_COLORS, C } from "../../styles/theme";
import { useSort } from "../../hooks/useSort";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Ticker {
  ticker: string;
  name: string;
  category: string;
  sector?: string;
  theme?: string;
  asset_type: string;
  composite: number;
  tcs: number;
  tfs: number;
  oer: number;
  rss: number;
  qvr_score?: number;
  qvr_q?: number;
  qvr_v?: number;
  qvr_r?: number;
  qvr_n_analysts?: number;
  qvr_bullish_chg_3m?: number | null;
  qvr_eps_beat_rate?: number | null;
  qvr_eps_surprise_avg?: number | null;
  classification: string;
  eligible: boolean;
  rejection: string;
  rsi: number;
  trend_age: number;
  sma50_dist: number;
  combined_long: number;
  combined_short: number;
  net_signal: string;
  long_count: number;
  short_count: number;
  ret_1w: number;
  ret_1m: number;
  adv_M: number;
  // Multi-horizon returns
  ret_1d: number;
  ret_5d: number;
  ret_21d: number;
  ret_63d: number;
  ret_126d: number;
  ret_252d: number;
  ret_3y_ann: number | null;
  ret_5y_ann: number | null;
  vol_3y_ann: number | null;
}

// Excluded group includes:
//   1) the 4 bearish classifications (DOWNTREND/CYCLE_PEAK/COUNTER_RALLY/EXHAUSTING)
//   2) Stocks demoted by the QVR fundamentals gate (rejection contains "WeakQVR")
//      \u2014 bullish technicals but weak fundamentals \u2192 "junk momentum" filter.
const BEARISH_CLASSIFICATIONS = new Set([
  "\u2b07\ufe0f DOWNTREND",
  "\ud83d\udd34 CYCLE_PEAK",
  "\ud83d\udfe3 COUNTER_RALLY",
  "\ud83d\udfe4 EXHAUSTING",
]);

function isWeakQVRReject(rejection: string | undefined): boolean {
  return !!(rejection && rejection.indexOf("WeakQVR") >= 0);
}

// ---------------------------------------------------------------------------
// Section toggle
// ---------------------------------------------------------------------------

function Section({
  title,
  children,
  defaultOpen = false,
  badge,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-3 text-left text-sm font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="flex items-center gap-2">
          {title}
          {badge && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-900/50 text-cyan-400">
              {badge}
            </span>
          )}
        </span>
        <span className="text-gray-500 text-xs">{open ? "\u25bc" : "\u25b6"}</span>
      </button>
      {open && <div className="p-4 bg-[#0d1117] space-y-4">{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

function compColor(v: number): string {
  if (v >= 70) return C.green;
  if (v >= 55) return C.cyan;
  if (v >= 40) return C.yellow;
  return C.gray;
}

function retColor(v: number): string {
  if (v > 3) return C.green;
  if (v > 0) return "#86efac";
  if (v > -3) return "#fca5a5";
  return C.red;
}

function oerColor(v: number): string {
  if (v >= 60) return C.red;
  if (v >= 40) return C.orange;
  return C.green;
}

function qvrColor(v: number): string {
  if (v >= 65) return C.green;
  if (v >= 50) return C.cyan;
  if (v >= 40) return C.yellow;
  return C.red;
}

// ---------------------------------------------------------------------------
// Column Definitions
// ---------------------------------------------------------------------------

const COLUMN_DEFS = [
  { col: "Ticker", desc: "Ticker symbol (click for detail)" },
  { col: "Name", desc: "Security name" },
  { col: "Sector", desc: "통합 17개 섹터 (GICS 11 + Fixed Income / International / Equity Broad / Macro / Multi-Asset / Alternatives). ETF/주식 동일 체계." },
  { col: "Class", desc: "Current classification" },
  { col: "Rejection", desc: "Reason for ineligibility (LowScore, Downtrend, Liq, etc.)" },
  { col: "Comp", desc: "Composite score (0-100)" },
  { col: "TCS", desc: "Trend Continuation Score" },
  { col: "TFS", desc: "Trend Formation Score" },
  { col: "RSS", desc: "Relative Strength Score" },
  { col: "OER", desc: "Overextension Risk \u2014 higher = riskier" },
  { col: "QVR", desc: "Quality-Value-Revision (0-100). Stocks with QVR < 40 are excluded by the fundamentals gate (rejection: WeakQVR). ETFs receive neutral 50 and bypass the gate. Hover for Q/V/R sub-scores." },
  { col: "RSI", desc: "14-day Relative Strength Index" },
  { col: "ADV", desc: "Average Daily Volume ($M)" },
  { col: "1D", desc: "1-day return (%)" },
  { col: "1W", desc: "1-week return (%) \u2014 5 trading days" },
  { col: "1M", desc: "1-month return (%) \u2014 21 trading days" },
  { col: "3M", desc: "3-month return (%) \u2014 63 trading days" },
  { col: "6M", desc: "6-month return (%) \u2014 126 trading days" },
  { col: "1Y", desc: "1-year return (%) \u2014 252 trading days" },
  { col: "3Y/A", desc: "3-year annualized return (%)" },
  { col: "5Y/A", desc: "5-year annualized return (%)" },
  { col: "Vol3Y", desc: "3-year annualized volatility (%)" },
];

function ColumnDefinitions() {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden mb-2">
      <button
        className="w-full px-4 py-2 text-left text-xs font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="text-gray-400">Column Definitions</span>
        <span className="text-gray-500 text-[10px]">{open ? "\u25bc" : "\u25b6"}</span>
      </button>
      {open && (
        <div className="bg-[#0d1117] px-4 py-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-x-6 gap-y-1.5">
          {COLUMN_DEFS.map((d) => (
            <div key={d.col} className="flex gap-2 text-[11px]">
              <span className="text-cyan-400 font-semibold shrink-0 w-16">{d.col}</span>
              <span className="text-gray-500">{d.desc}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Excluded Table
// ---------------------------------------------------------------------------

function ExcludedTable({
  rows,
  onSelect,
}: {
  rows: Ticker[];
  onSelect: (ticker: string) => void;
}) {
  const accessors = useMemo(() => ({
    ticker: (r: Ticker) => r.ticker,
    name: (r: Ticker) => r.name,
    sector: (r: Ticker) => r.sector || r.category,
    classification: (r: Ticker) => r.classification,
    rejection: (r: Ticker) => r.rejection || "",
    composite: (r: Ticker) => r.composite,
    tcs: (r: Ticker) => r.tcs,
    tfs: (r: Ticker) => r.tfs,
    rss: (r: Ticker) => r.rss,
    oer: (r: Ticker) => r.oer,
    qvr: (r: Ticker) => r.qvr_score ?? 50,
    rsi: (r: Ticker) => r.rsi,
    adv: (r: Ticker) => r.adv_M,
    ret_1d: (r: Ticker) => r.ret_1d ?? 0,
    ret_5d: (r: Ticker) => r.ret_5d ?? 0,
    ret_21d: (r: Ticker) => r.ret_21d ?? 0,
    ret_63d: (r: Ticker) => r.ret_63d ?? 0,
    ret_126d: (r: Ticker) => r.ret_126d ?? 0,
    ret_252d: (r: Ticker) => r.ret_252d ?? 0,
    ret_3y_ann: (r: Ticker) => r.ret_3y_ann ?? -999,
    ret_5y_ann: (r: Ticker) => r.ret_5y_ann ?? -999,
    vol_3y_ann: (r: Ticker) => r.vol_3y_ann ?? -999,
  }), []);
  const { sorted, onSort, indicator } = useSort(rows, accessors);

  if (!rows.length) return <p className="text-[10px] text-gray-600">No data</p>;
  const headerCls = "py-1.5 px-2 text-gray-500 cursor-pointer select-none hover:text-gray-200 whitespace-nowrap";
  return (
    <div className="overflow-auto border border-gray-800 rounded" style={{ maxHeight: "600px" }}>
      <table className="w-full text-xs border-collapse">
        <thead className="sticky top-0 z-10 bg-[#1f2937]">
          <tr className="border-b border-gray-700">
            <th className="py-1.5 px-2 text-left text-gray-500">#</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("ticker")}>Ticker{indicator("ticker")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("name")}>Name{indicator("name")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("sector")}>Sector{indicator("sector")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("classification")}>Class{indicator("classification")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("rejection")}>Rejection{indicator("rejection")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("composite")}>Comp{indicator("composite")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("tcs")}>TCS{indicator("tcs")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("tfs")}>TFS{indicator("tfs")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("rss")}>RSS{indicator("rss")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("oer")}>OER{indicator("oer")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("qvr")}>QVR{indicator("qvr")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("rsi")}>RSI{indicator("rsi")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("adv")}>ADV{indicator("adv")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_1d")}>1D{indicator("ret_1d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_5d")}>1W{indicator("ret_5d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_21d")}>1M{indicator("ret_21d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_63d")}>3M{indicator("ret_63d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_126d")}>6M{indicator("ret_126d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_252d")}>1Y{indicator("ret_252d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_3y_ann")}>3Y/A{indicator("ret_3y_ann")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_5y_ann")}>5Y/A{indicator("ret_5y_ann")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("vol_3y_ann")}>Vol3Y{indicator("vol_3y_ann")}</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr key={r.ticker} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
              <td className="py-1.5 px-2 text-gray-600">{i + 1}</td>
              <td className="py-1.5 px-2">
                <button onClick={() => onSelect(r.ticker)} className="font-mono text-xs text-cyan-400 hover:underline font-bold">
                  {r.ticker}
                </button>
              </td>
              <td className="py-1.5 px-2 text-gray-400 truncate max-w-[120px]">{r.name}</td>
              <td className="py-1.5 px-2 text-gray-500 text-[10px]" title={`SubTheme: ${r.theme || "-"}`}>
                {r.sector || r.category}
              </td>
              <td className="py-1.5 px-2">
                <span className="text-[10px]" style={{ color: CLASS_COLORS[r.classification] || C.gray }}>
                  {r.classification}
                </span>
              </td>
              <td className="py-1.5 px-2 text-[10px] text-orange-400">{r.rejection || "-"}</td>
              <td className="py-1.5 px-2 text-right font-mono font-bold" style={{ color: compColor(r.composite) }}>
                {r.composite?.toFixed(1)}
              </td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-300">{r.tcs}</td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-300">{r.tfs}</td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-300">{r.rss}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: oerColor(r.oer) }}>{r.oer}</td>
              {(() => {
                const b3m = r.qvr_bullish_chg_3m;
                const beat = r.qvr_eps_beat_rate;
                const surp = r.qvr_eps_surprise_avg;
                const fhParts: string[] = [];
                if (b3m != null) fhParts.push(`Bull3m ${b3m >= 0 ? "+" : ""}${b3m.toFixed(1)}%`);
                if (beat != null) fhParts.push(`Beat ${beat.toFixed(0)}%`);
                if (surp != null) fhParts.push(`Surp ${surp >= 0 ? "+" : ""}${surp.toFixed(1)}%`);
                const tip = r.asset_type === "ETF"
                  ? "ETF (no fundamentals)"
                  : `Q ${r.qvr_q ?? "-"} | V ${r.qvr_v ?? "-"} | R ${r.qvr_r ?? "-"}`
                    + (r.qvr_n_analysts ? ` · ${r.qvr_n_analysts} analysts` : "")
                    + (fhParts.length ? ` · ${fhParts.join(" / ")}` : "");
                return (
                  <td className="py-1.5 px-2 text-right font-mono" style={{ color: qvrColor(r.qvr_score ?? 50) }}
                      title={tip}>
                    {(r.qvr_score ?? 50).toFixed(0)}
                  </td>
                );
              })()}
              <td className="py-1.5 px-2 text-right font-mono text-gray-400">{r.rsi?.toFixed(0)}</td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-400">{r.adv_M?.toFixed(1)}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_1d ?? 0) }}>{(r.ret_1d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_5d ?? 0) }}>{(r.ret_5d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_21d ?? 0) }}>{(r.ret_21d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_63d ?? 0) }}>{(r.ret_63d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_126d ?? 0) }}>{(r.ret_126d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: retColor(r.ret_252d ?? 0) }}>{(r.ret_252d ?? 0).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: r.ret_3y_ann == null ? C.gray : retColor(r.ret_3y_ann) }}>{r.ret_3y_ann == null ? "-" : `${r.ret_3y_ann.toFixed(1)}%`}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: r.ret_5y_ann == null ? C.gray : retColor(r.ret_5y_ann) }}>{r.ret_5y_ann == null ? "-" : `${r.ret_5y_ann.toFixed(1)}%`}</td>
              <td className="py-1.5 px-2 text-right font-mono text-gray-400">{r.vol_3y_ann == null ? "-" : `${r.vol_3y_ann.toFixed(1)}%`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Ticker Detail
// ---------------------------------------------------------------------------

function TickerDetail({ ticker, onClose }: { ticker: Ticker; onClose: () => void }) {
  const axes = [
    { label: "TCS", value: ticker.tcs, desc: "Trend Continuation" },
    { label: "TFS", value: ticker.tfs, desc: "Trend Formation" },
    { label: "RSS", value: ticker.rss, desc: "Relative Strength" },
    { label: "OER", value: ticker.oer, desc: "Overextension Risk" },
    { label: "QVR", value: ticker.qvr_score ?? 50, desc: "Quality-Value-Revision" },
  ];
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold text-cyan-400 font-mono">{ticker.ticker}</span>
          <span className="text-sm text-gray-300">{ticker.name}</span>
          <span className="text-[10px] px-2 py-0.5 rounded" style={{
            backgroundColor: (CLASS_COLORS[ticker.classification] || C.gray) + "22",
            color: CLASS_COLORS[ticker.classification] || C.gray,
          }}>
            {ticker.classification}
          </span>
          {ticker.rejection && (
            <span className="text-[10px] px-2 py-0.5 rounded bg-orange-900/30 text-orange-400">
              {ticker.rejection}
            </span>
          )}
        </div>
        <button onClick={onClose} className="text-[10px] text-gray-500 hover:text-gray-300 px-2 py-1 rounded border border-gray-800">
          Close
        </button>
      </div>

      {/* Classification tags */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-gray-500">
        <span>Sector: <span className="text-gray-300 font-semibold">{ticker.sector || ticker.category}</span></span>
        {ticker.theme && ticker.theme !== "-" && (
          <span>SubTheme: <span className="text-gray-300">{ticker.theme}</span></span>
        )}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {axes.map((a) => (
          <div key={a.label} className="bg-[#111827] border border-gray-800 rounded-lg p-3"
               title={a.label === "QVR" ? `Q ${ticker.qvr_q ?? "-"} | V ${ticker.qvr_v ?? "-"} | R ${ticker.qvr_r ?? "-"}` : undefined}>
            <div className="text-[10px] text-gray-500">{a.desc}</div>
            <div className="text-xl font-bold font-mono mt-1" style={{
              color: a.label === "OER" ? oerColor(a.value)
                   : a.label === "QVR" ? qvrColor(a.value)
                   : compColor(a.value),
            }}>
              {typeof a.value === "number" ? a.value.toFixed(0) : a.value}
            </div>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-2 text-xs">
        <div className="bg-[#111827] rounded p-2">
          <div className="text-[10px] text-gray-500">Composite</div>
          <div className="font-mono font-bold" style={{ color: compColor(ticker.composite) }}>{ticker.composite?.toFixed(1)}</div>
        </div>
        <div className="bg-[#111827] rounded p-2">
          <div className="text-[10px] text-gray-500">RSI</div>
          <div className="font-mono">{ticker.rsi?.toFixed(0)}</div>
        </div>
        <div className="bg-[#111827] rounded p-2">
          <div className="text-[10px] text-gray-500">Trend Age</div>
          <div className="font-mono">{ticker.trend_age}d</div>
        </div>
        <div className="bg-[#111827] rounded p-2">
          <div className="text-[10px] text-gray-500">SMA50 Dist</div>
          <div className="font-mono">{ticker.sma50_dist?.toFixed(1)}%</div>
        </div>
        <div className="bg-[#111827] rounded p-2">
          <div className="text-[10px] text-gray-500">ADV</div>
          <div className="font-mono">${ticker.adv_M?.toFixed(1)}M</div>
        </div>
        <div className="bg-[#111827] rounded p-2">
          <div className="text-[10px] text-gray-500">Ret 1M</div>
          <div className="font-mono" style={{ color: retColor(ticker.ret_1m) }}>{ticker.ret_1m?.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Export
// ---------------------------------------------------------------------------

export function ExcludedTab({ filters, totalUniverse, mlMode = false }:
  { filters: FilterParams; totalUniverse?: number; mlMode?: boolean }) {
  const [allData, setAllData] = useState<Ticker[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const tableFetch = mlMode ? fetchTableML(filters) : fetchTable(filters);
    tableFetch.then((res) => {
      setAllData(res.data || []);
    }).finally(() => setLoading(false));
  }, [filters, mlMode]);

  // Excluded = bearish classifications OR demoted by QVR fundamentals gate.
  // (Other bullish-but-ineligible cases like LowScore/Liq still hidden.)
  const excluded = useMemo(() => {
    return allData
      .filter((t) => BEARISH_CLASSIFICATIONS.has(t.classification)
                   || isWeakQVRReject(t.rejection))
      .sort((a, b) => b.composite - a.composite);
  }, [allData]);

  const etf = useMemo(() => excluded.filter((t) => t.asset_type === "ETF"), [excluded]);
  const stock = useMemo(() => excluded.filter((t) => t.asset_type === "Stock"), [excluded]);

  // Classification breakdown
  const classDist = useMemo(() => {
    const map: Record<string, number> = {};
    for (const t of excluded) {
      map[t.classification] = (map[t.classification] || 0) + 1;
    }
    return Object.entries(map).sort((a, b) => b[1] - a[1]);
  }, [excluded]);

  // Rejection breakdown
  const rejectionDist = useMemo(() => {
    const map: Record<string, number> = {};
    for (const t of excluded) {
      const reasons = (t.rejection || "Unknown").split("/");
      for (const r of reasons) {
        const key = r.trim() || "Unknown";
        map[key] = (map[key] || 0) + 1;
      }
    }
    return Object.entries(map).sort((a, b) => b[1] - a[1]);
  }, [excluded]);

  const selectedTicker = useMemo(
    () => (selected ? allData.find((t) => t.ticker === selected) ?? null : null),
    [selected, allData]
  );

  if (loading) return <div className="text-gray-500 p-8">Loading...</div>;

  const avgComp = excluded.length
    ? excluded.reduce((s, t) => s + t.composite, 0) / excluded.length
    : 0;

  return (
    <div className="space-y-5">
      {/* ── KPIs ── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="Excluded" value={excluded.length} sub={`/ ${totalUniverse || allData.length} universe`} />
        <MetricCard label="ETF" value={etf.length} sub="excluded" />
        <MetricCard label="Stock" value={stock.length} sub="excluded" />
        <MetricCard label="Avg Composite" value={avgComp.toFixed(1)} sub="excluded avg" />
        <MetricCard
          label="Top Reason"
          value={rejectionDist[0]?.[0] || "-"}
          sub={`${rejectionDist[0]?.[1] || 0} tickers`}
        />
      </div>

      {/* ── Classification + Rejection Breakdown ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Classification</div>
          <div className="flex gap-1.5 flex-wrap">
            {classDist.map(([cls, count]) => (
              <span
                key={cls}
                className="text-[10px] px-2 py-1 rounded border border-gray-800"
                style={{ color: CLASS_COLORS[cls] || C.gray, borderColor: (CLASS_COLORS[cls] || C.gray) + "44" }}
              >
                {cls} <span className="font-mono font-bold ml-1">{count}</span>
              </span>
            ))}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1.5">Rejection Reason</div>
          <div className="flex gap-1.5 flex-wrap">
            {rejectionDist.map(([reason, count]) => (
              <span key={reason} className="text-[10px] px-2 py-1 rounded border border-gray-800 text-orange-400 border-orange-900/40">
                {reason} <span className="font-mono font-bold ml-1">{count}</span>
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── All Excluded Table ── */}
      <ColumnDefinitions />
      <ExcludedTable rows={excluded} onSelect={setSelected} />

      {/* ── ETF + Stock side by side ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div>
          <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-wide mb-2">
            ETF ({etf.length})
          </h3>
          <ExcludedTable rows={etf} onSelect={setSelected} />
        </div>
        <div>
          <h3 className="text-xs font-semibold text-green-400 uppercase tracking-wide mb-2">
            Stock ({stock.length})
          </h3>
          <ExcludedTable rows={stock} onSelect={setSelected} />
        </div>
      </div>

      {/* ── Selected Ticker Detail ── */}
      {selectedTicker && (
        <Section title={`${selectedTicker.ticker} — ${selectedTicker.name}`} defaultOpen badge={selectedTicker.classification}>
          <TickerDetail ticker={selectedTicker as Ticker} onClose={() => setSelected(null)} />
        </Section>
      )}
    </div>
  );
}
