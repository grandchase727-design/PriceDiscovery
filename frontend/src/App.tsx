import { useEffect, useState, useMemo, useCallback, useRef } from "react";
import { fetchMeta, startScan, fetchScanStatus, reloadCache, type FilterParams } from "./api/client";
import { MarketEnvironmentTab } from "./components/tabs/MarketEnvironmentTab";
import { PriceDiscoveryTab } from "./components/tabs/PriceDiscoveryTab";
import { PriceDiscoveryMLTab } from "./components/tabs/PriceDiscoveryMLTab";
import { ValidationTab } from "./components/tabs/ValidationTab";
import { AnalysisTab } from "./components/tabs/AnalysisTab";
import { AppendixTab } from "./components/tabs/AppendixTab";
import { AIPredictionTab } from "./components/tabs/AIPredictionTab";

const TABS = ["Price Discovery", "Price Discovery (ML)", "Validation", "Market Environment", "Analysis", "AI Prediction", "Appendix"];

// Asset class mode definitions (Option B unified taxonomy — sector-based)
type AssetMode = "equity" | "fixed_income";

const EQUITY_SECTORS = new Set([
  "Technology", "Communication Services", "Healthcare", "Financials",
  "Consumer Discretionary", "Consumer Staples", "Industrials", "Energy",
  "Utilities", "Materials", "Real Estate",
  "Equity Broad", "International",
]);
const FICC_SECTORS = new Set([
  "Fixed Income", "Macro", "Multi-Asset", "Alternatives",
]);

function sectorsForMode(mode: AssetMode, allSectors: string[]): string[] {
  const target = mode === "equity" ? EQUITY_SECTORS : FICC_SECTORS;
  return allSectors.filter((s) => target.has(s));
}

export default function App() {
  const [meta, setMeta] = useState<any>(null);
  const [tab, setTab] = useState(0);
  const [dataVersion, setDataVersion] = useState(0); // bump to trigger re-fetch

  // Asset class mode
  const [assetMode, setAssetMode] = useState<AssetMode>("equity");

  // Scan state
  const [scanning, setScanning] = useState(false);
  const [scanMsg, setScanMsg] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Filters (Option B: sector-based, category dropped from UI)
  const [sectors, setSectors] = useState<string[]>([]);
  const [classifications, setClassifications] = useState<string[]>([]);
  const [eligibleOnly, setEligibleOnly] = useState(false);
  const [compMin, setCompMin] = useState(0);
  const [compMax, setCompMax] = useState(100);

  // Scan options
  const [lookbackYears, setLookbackYears] = useState(5);
  const [useRealtime, setUseRealtime] = useState(true);
  const [includeStocks, setIncludeStocks] = useState(true);

  const loadMeta = useCallback(() => {
    fetchMeta().then((m) => {
      setMeta(m);
      setSectors(sectorsForMode(assetMode, m.sectors || []));
      setClassifications(m.classifications || []);
    });
  }, [assetMode]);

  useEffect(() => { loadMeta(); }, [loadMeta]);

  // When mode changes, update sector filter
  useEffect(() => {
    if (meta?.sectors) {
      setSectors(sectorsForMode(assetMode, meta.sectors));
    }
  }, [assetMode, meta?.sectors]);

  // Poll scan status when scanning
  const handleStartScan = useCallback(() => {
    setScanning(true);
    setScanMsg("Starting scan...");
    startScan({ lookback_years: lookbackYears, use_realtime: useRealtime, include_stocks: includeStocks })
      .then((r) => {
        if (r.status === "already_running") {
          setScanMsg("Scan already in progress...");
        } else {
          setScanMsg("Scan running... (this takes a few minutes)");
        }
        // Poll every 3 seconds
        pollRef.current = setInterval(() => {
          fetchScanStatus().then((s) => {
            if (!s.running) {
              clearInterval(pollRef.current!);
              pollRef.current = null;
              setScanning(false);
              if (s.last_error) {
                setScanMsg(`Error: ${s.last_error}`);
              } else {
                setScanMsg("Scan complete! Reloading...");
                reloadCache().then(() => {
                  loadMeta();
                  setDataVersion((v) => v + 1);
                  setScanMsg("Data refreshed.");
                  setTimeout(() => setScanMsg(""), 3000);
                });
              }
            } else {
              // Show live progress
              const phase = s.phase || "Init";
              const elapsed = s.started_at ? Math.round((Date.now() - new Date(s.started_at).getTime()) / 1000) : 0;
              const tail = (s.last_line || "").slice(0, 80);
              setScanMsg(`[${phase}] ${elapsed}s · ${tail}`);
            }
          });
        }, 3000);
      })
      .catch((e) => {
        setScanning(false);
        setScanMsg(`Failed: ${e.message}`);
      });
  }, [lookbackYears, useRealtime, includeStocks, loadMeta]);

  // Cleanup poll on unmount
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  // dataVersion in dependency forces all tabs to re-fetch after scan completes
  // Always send sectors to enforce mode filter (never undefined)
  const filters: FilterParams = useMemo(() => ({
    sectors: sectors.length > 0 ? sectors : sectorsForMode(assetMode, meta?.sectors || []),
    classifications: classifications.length === (meta?.classifications?.length || 0) ? undefined : classifications,
    eligible_only: eligibleOnly,
    comp_min: compMin,
    comp_max: compMax,
    _v: dataVersion, // cache-bust key (not sent to API, triggers re-fetch)
  } as any), [sectors, classifications, eligibleOnly, compMin, compMax, meta, dataVersion, assetMode]);

  if (!meta) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-500 text-lg">Connecting to API...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-[#111827] border-r border-gray-800 p-4 overflow-y-auto shrink-0">
        <h1 className="text-lg font-bold text-cyan-400 mb-1">Price Discovery</h1>
        <div className="text-xs text-gray-500 mb-2">Scanner v5.0 | {meta.total_tickers} tickers</div>

        {/* ── Asset Mode Toggle ── */}
        <div className="mb-3 flex rounded-lg overflow-hidden border border-gray-700">
          <button
            className={`flex-1 py-1.5 text-xs font-semibold transition-colors ${
              assetMode === "equity"
                ? "bg-cyan-600 text-white"
                : "bg-[#1f2937] text-gray-400 hover:text-gray-200"
            }`}
            onClick={() => setAssetMode("equity")}>
            주식형
          </button>
          <button
            className={`flex-1 py-1.5 text-xs font-semibold transition-colors ${
              assetMode === "fixed_income"
                ? "bg-amber-600 text-white"
                : "bg-[#1f2937] text-gray-400 hover:text-gray-200"
            }`}
            onClick={() => setAssetMode("fixed_income")}>
            FICC
          </button>
        </div>
        <div className="text-[10px] text-gray-600 mb-3">
          {assetMode === "equity"
            ? "Equity · Sectors · Factors · International · Thematic · Stocks"
            : "FICC — FI Short/Intermediate/Long/Credit/Inflation/Intl · Commodities · Real Assets · Currency · Multi-Asset"}
        </div>

        <div className="text-xs text-gray-600 mb-3">
          Scan: {meta.scan_time ? new Date(meta.scan_time).toLocaleString("ko-KR", {
            year: "numeric", month: "2-digit", day: "2-digit",
            hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
          }) : "N/A"}
        </div>

        {/* ── Run Live Scan ── */}
        <div className="mb-4 p-3 bg-[#0d1117] border border-gray-800 rounded-lg space-y-2">
          <div className="flex items-center gap-2">
            <label className="text-[10px] text-gray-500 w-16">Lookback</label>
            <select value={lookbackYears} onChange={(e) => setLookbackYears(+e.target.value)}
              className="flex-1 px-1.5 py-0.5 text-xs bg-[#1f2937] border border-gray-700 rounded" disabled={scanning}>
              <option value={1}>1 year</option>
              <option value={3}>3 years</option>
              <option value={5}>5 years</option>
            </select>
          </div>
          <label className="flex items-center gap-1.5 text-[11px] cursor-pointer">
            <input type="checkbox" checked={useRealtime} onChange={(e) => setUseRealtime(e.target.checked)}
              className="rounded bg-[#1f2937] border-gray-600 w-3 h-3" disabled={scanning} />
            Realtime prices
          </label>
          <label className="flex items-center gap-1.5 text-[11px] cursor-pointer">
            <input type="checkbox" checked={includeStocks} onChange={(e) => setIncludeStocks(e.target.checked)}
              className="rounded bg-[#1f2937] border-gray-600 w-3 h-3" disabled={scanning} />
            Include stocks
          </label>
          <button
            onClick={handleStartScan}
            disabled={scanning}
            className={`w-full py-1.5 rounded text-xs font-semibold transition-colors ${
              scanning
                ? "bg-gray-700 text-gray-400 cursor-wait"
                : "bg-cyan-600 hover:bg-cyan-500 text-white"
            }`}>
            {scanning ? "Scanning..." : "Run Live Scan"}
          </button>
          {scanMsg && (
            <div className={`text-[10px] ${scanMsg.includes("Error") || scanMsg.includes("Failed") ? "text-red-400" : "text-cyan-400"}`}>
              {scanMsg}
            </div>
          )}
        </div>

        {/* ── Filters ── */}
        {/* Eligible Only */}
        <label className="flex items-center gap-2 text-sm mb-3 cursor-pointer">
          <input type="checkbox" checked={eligibleOnly} onChange={(e) => setEligibleOnly(e.target.checked)}
            className="rounded bg-[#1f2937] border-gray-600" />
          Eligible only
        </label>

        {/* Composite Range */}
        <div className="mb-4">
          <div className="text-xs text-gray-500 mb-1">Composite: {compMin} — {compMax}</div>
          <div className="flex gap-2">
            <input type="number" value={compMin} onChange={(e) => setCompMin(+e.target.value)}
              className="w-16 px-1 py-0.5 text-xs bg-[#1f2937] border border-gray-700 rounded" min={0} max={100} />
            <input type="number" value={compMax} onChange={(e) => setCompMax(+e.target.value)}
              className="w-16 px-1 py-0.5 text-xs bg-[#1f2937] border border-gray-700 rounded" min={0} max={100} />
          </div>
        </div>

        {/* Sectors (Option B unified taxonomy) */}
        <div className="mb-4">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs text-gray-500">Sectors ({sectors.length}/{(meta.sectors || []).length})</span>
            <button className="text-[10px] text-cyan-500 hover:underline"
              onClick={() => setSectors(sectors.length === (meta.sectors || []).length ? [] : (meta.sectors || []))}>
              {sectors.length === (meta.sectors || []).length ? "Clear" : "All"}
            </button>
          </div>
          <div className="max-h-48 overflow-y-auto space-y-0.5">
            {(meta.sectors || []).map((s: string) => (
              <label key={s} className="flex items-center gap-1.5 text-[11px] cursor-pointer hover:text-gray-200">
                <input type="checkbox" checked={sectors.includes(s)}
                  onChange={(e) => setSectors(e.target.checked ? [...sectors, s] : sectors.filter((x) => x !== s))}
                  className="rounded bg-[#1f2937] border-gray-600 w-3 h-3" />
                {s}
              </label>
            ))}
          </div>
        </div>

        {/* Classifications */}
        <div className="mb-4">
          <div className="flex justify-between items-center mb-1">
            <span className="text-xs text-gray-500">Classifications ({classifications.length}/{meta.classifications.length})</span>
            <button className="text-[10px] text-cyan-500 hover:underline"
              onClick={() => setClassifications(classifications.length === meta.classifications.length ? [] : meta.classifications)}>
              {classifications.length === meta.classifications.length ? "Clear" : "All"}
            </button>
          </div>
          <div className="max-h-40 overflow-y-auto space-y-0.5">
            {meta.classifications.map((c: string) => (
              <label key={c} className="flex items-center gap-1.5 text-[11px] cursor-pointer hover:text-gray-200">
                <input type="checkbox" checked={classifications.includes(c)}
                  onChange={(e) => setClassifications(e.target.checked ? [...classifications, c] : classifications.filter((x) => x !== c))}
                  className="rounded bg-[#1f2937] border-gray-600 w-3 h-3" />
                {c.slice(0, 20)}
              </label>
            ))}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        {/* Tab Bar */}
        <div className="sticky top-0 z-20 bg-[#0a0e17] border-b border-gray-800 px-4">
          <div className="flex items-center gap-0">
            <span className={`px-3 py-2.5 text-[10px] font-bold uppercase tracking-wider ${
              assetMode === "equity" ? "text-cyan-400" : "text-amber-400"
            }`}>
              {assetMode === "equity" ? "주식형" : "FICC"}
            </span>
            <span className="text-gray-700 mr-1">|</span>
            {TABS.map((t, i) => (
              <button key={t}
                className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                  tab === i
                    ? assetMode === "equity" ? "border-cyan-400 text-cyan-400" : "border-amber-400 text-amber-400"
                    : "border-transparent text-gray-500 hover:text-gray-300"
                }`}
                onClick={() => setTab(i)}>
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {tab === 0 && <PriceDiscoveryTab filters={filters} />}
          {tab === 1 && <PriceDiscoveryMLTab filters={filters} />}
          {tab === 2 && <ValidationTab />}
          {tab === 3 && <MarketEnvironmentTab filters={filters} />}
          {tab === 4 && <AnalysisTab filters={filters} />}
          {tab === 5 && <AIPredictionTab />}
          {tab === 6 && <AppendixTab filters={filters} />}
        </div>
      </main>
    </div>
  );
}
