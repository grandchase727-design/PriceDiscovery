import { useEffect, useState, useMemo, useCallback, useRef } from "react";
import { fetchMeta, startScan, fetchScanStatus, reloadCache, startAutoFill, fetchAutoFillStatus, startMarketLeadersSwarm, startBacktestRun, fetchBacktestStatus, type FilterParams } from "./api/client";
import { MarketEnvironmentTab } from "./components/tabs/MarketEnvironmentTab";
import { MarketCommentaryTab } from "./components/tabs/MarketCommentaryTab";
import { PriceDiscoveryTab } from "./components/tabs/PriceDiscoveryTab";
import { PriceDiscoveryMLTab } from "./components/tabs/PriceDiscoveryMLTab";
import { ValidationTab } from "./components/tabs/ValidationTab";
import { AnalysisTab } from "./components/tabs/AnalysisTab";
import { AppendixTab } from "./components/tabs/AppendixTab";
import { AIPredictionTab } from "./components/tabs/AIPredictionTab";
import FinalListPanel from "./components/shared/FinalListPanel";

const TABS = ["Market Commentary", "Price Discovery", "Price Discovery (ML)", "Validation", "Market Environment", "Analysis", "AI Prediction", "Appendix"];

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

  // Scan state — single unified pipeline (scan → cache → swarm + backtest)
  const [scanning, setScanning] = useState(false);
  const [scanMsg, setScanMsg] = useState("");
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Progress tracking — 0-100 across the WHOLE Run Live Scan pipeline
  // Stage allocation:
  //   0-20%  : Live Scan (770 ticker price refresh)
  //   20-30% : Reload cache + loadMeta
  //   30-40% : Auto-Fill Debate Cache
  //   40-95% : Market Leaders Swarm (phase1 → 5.6) + Backtest in parallel
  //   95-100%: Final wrap-up
  const [progressPct, setProgressPct] = useState(0);
  const [progressStage, setProgressStage] = useState<string>("");

  const SWARM_PHASE_PROGRESS: Record<string, number> = {
    // Maps swarm.phase → progress % within the 40-95 swarm range (inside pipeline)
    phase1:           45,  // 5 analysts in parallel
    phase2:           52,  // coherence debater
    phase3:           57,  // synthesis (neutral + averse)
    phase4:           62,  // action selector
    phase4_action:    65,
    phase5:           70,  // PM × 3 horizons
    phase5_pm:        72,
    phase5_5:         80,  // trading
    phase5_55:        85,  // risk_llm
    phase5_6a:        90,  // debate synthesizer
    phase5_6:         93,  // position state
  };

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
    setProgressPct(2);
    setProgressStage("🔄 Live Scan 시작");
    setScanMsg("Starting scan...");
    startScan({ lookback_years: lookbackYears, use_realtime: useRealtime, include_stocks: includeStocks })
      .then((r) => {
        if (r.status === "already_running") {
          setScanMsg("Scan already in progress...");
        } else {
          setScanMsg("Scan running... (this takes a few minutes)");
        }
        setProgressPct(5);
        setProgressStage("🔄 Live Scan 실행 중");
        // Poll every 3 seconds
        pollRef.current = setInterval(() => {
          fetchScanStatus().then((s) => {
            if (!s.running) {
              clearInterval(pollRef.current!);
              pollRef.current = null;
              if (s.last_error) {
                setScanning(false);
                setProgressPct(0);
                setProgressStage("");
                setScanMsg(`Error: ${s.last_error}`);
              } else {
                // KEEP scanning=true through the rest of the pipeline (autofill + swarm + backtest)
                // so FinalListPanel keeps polling for swarm cache updates.
                setProgressPct(20);
                setProgressStage("✓ Scan 완료 · Cache 재로딩");
                setScanMsg("Scan complete! Reloading...");
                reloadCache().then(() => {
                  loadMeta();
                  setDataVersion((v) => v + 1);
                  setProgressPct(30);
                  setProgressStage("🤖 Debate Cache 준비");
                  // After scan + reload, server auto-dispatches `claude -p`
                  // for each uncached live pick (Max plan-billed). Poll status.
                  // Chain function: start swarm + backtest + final poll
                  const chainSwarmAndBacktest = () => {
                    setProgressPct(40);
                    setProgressStage("🤖 Swarm + Backtest 시작");
                    const swP = startMarketLeadersSwarm(true).catch((e) => ({ status: "error", error: String(e) }));
                    const btP = startBacktestRun().catch((e) => ({ status: "error", error: String(e) }));
                    Promise.all([swP, btP]).then(([sw, bt]) => {
                      const swSt = (sw as any).status || "?";
                      const btSt = (bt as any).status || "?";
                      setScanMsg(`🤖 Swarm: ${swSt} · 📊 Backtest: ${btSt} — 진행 중…`);
                      let swDone = swSt === "already_running" || swSt === "cached" || swSt === "error";
                      let btDone = btSt === "already_running" || btSt === "error";
                      const finalPoll = setInterval(() => {
                        Promise.all([
                          fetch("/api/market-leaders/swarm/status").then((r) => r.json()).catch(() => null),
                          fetchBacktestStatus().catch(() => null),
                        ]).then(([swR, btR]) => {
                          if (swR && !swR.running) swDone = true;
                          if (btR && !btR.running) btDone = true;
                          const swPhase = swR?.running ? `${swR.phase || "—"}` : (swR?.last_error ? "⚠" : "✓");
                          const btPhase = btR?.running ? "실행 중" : (btR?.last_error ? "⚠" : "✓");
                          // Map swarm.phase → progress %
                          const swPctMap: Record<string, number> = {
                            phase1: 45, phase2: 52, phase3: 57, phase4: 62, phase4_action: 65,
                            phase5: 70, phase5_pm: 72, phase5_5: 80, phase5_55: 85, phase5_6a: 90, phase5_6: 93,
                          };
                          const swPct = swR?.running ? (swPctMap[swR.phase] ?? 50) : 95;
                          setProgressPct(swPct);
                          setProgressStage(`🤖 Swarm: ${swPhase} | 📊 Backtest: ${btPhase}`);
                          setScanMsg(`🤖 Swarm: ${swPhase} | 📊 Backtest: ${btPhase}`);
                          if (swDone && btDone) {
                            clearInterval(finalPoll);
                            setProgressPct(100);
                            setProgressStage("✓ 전체 파이프라인 완료");
                            setScanning(false);
                            setScanMsg("✓ 전체 시스템 (Scan + Cache + Swarm + Backtest) 완료 — 대시보드 갱신됨");
                            setDataVersion((v) => v + 1);
                            setTimeout(() => { setScanMsg(""); setProgressPct(0); setProgressStage(""); }, 12000);
                          }
                        }).catch(() => {});
                      }, 8000);
                    });
                  };

                  startAutoFill().then((a) => {
                    if (a.status === "no_claude_cli") {
                      setScanning(false);
                      setProgressPct(0);
                      setProgressStage("");
                      setScanMsg("Data refreshed. ⚠ Claude CLI not on server PATH — cache auto-fill unavailable.");
                      setTimeout(() => setScanMsg(""), 8000);
                      return;
                    }
                    if (a.status === "empty_queue" || a.status === "no_queue" || (a.n_total ?? 0) === 0) {
                      setScanMsg("Data refreshed. ✓ Debate cache up to date.");
                      chainSwarmAndBacktest();
                      return;
                    }
                    setProgressPct(32);
                    setProgressStage(`🤖 Debate Cache: ${a.n_total} jobs`);
                    setScanMsg(`Data refreshed. Dispatching ${a.n_total} debate jobs...`);
                    const pollId = setInterval(() => {
                      fetchAutoFillStatus().then((st) => {
                        if (!st.running) {
                          clearInterval(pollId);
                          setDataVersion((v) => v + 1);
                          if (st.n_failed > 0) {
                            setScanMsg(`✓ Cache filled: ${st.n_persisted}/${st.n_total} (${st.n_failed} failed) · swarm starting…`);
                          } else {
                            setScanMsg(`✓ Cache filled: ${st.n_persisted}/${st.n_total} · swarm starting…`);
                          }
                          chainSwarmAndBacktest();
                        } else {
                          const done = st.n_completed + st.n_failed;
                          const debatePct = 32 + Math.round((done / Math.max(1, st.n_total)) * 8);  // 32 → 40
                          setProgressPct(debatePct);
                          setProgressStage(`🤖 Debate Cache: ${done}/${st.n_total}`);
                          setScanMsg(`Filling cache: ${done}/${st.n_total} · current: ${st.current || "…"}`);
                        }
                      }).catch(() => clearInterval(pollId));
                    }, 4000);
                  }).catch(() => {
                    setScanning(false);
                    setProgressPct(0);
                    setProgressStage("");
                    setScanMsg("Data refreshed (cache auto-fill failed).");
                    setTimeout(() => setScanMsg(""), 5000);
                  });
                });
              }
            } else {
              // Phase-aware Live Scan progress mapping (matches api.py phase tracking):
              //   Init        → 3%
              //   Downloading → 4-6%
              //   Indicators  → 6-10%   (Phase 1)
              //   Ranking     → 10-15%  (Phase 2, eval N/24 interpolation)
              //   Validity    → 15-17%  (Phase 3)
              //   Scoring     → 17-18%  (Phase 4)
              //   Output      → 18-19%  (MASTER SUMMARY)
              //   Done        → 20% (handled in scan-complete branch)
              const phase = s.phase || "Init";
              const elapsed = s.started_at ? Math.round((Date.now() - new Date(s.started_at).getTime()) / 1000) : 0;
              const tail = (s.last_line || "").slice(0, 80);
              setScanMsg(`[${phase}] ${elapsed}s · ${tail}`);

              let pct = 5;
              let evalLabel = "";
              // Phase mapping reflects ACTUAL scan order observed in logs:
              //   Init → Downloading → Indicators(1) → Ranking(2) → Validity(3) → Scoring(4)
              //   → Summary(MASTER SUMMARY) → Backtest(7) → GraphRAG(6) → Insights
              //   → FactorEfficacy(8, 12 eval points) → Output(PDF) → Done(Cache saved)
              switch (phase) {
                case "Init":
                  pct = 3;
                  break;
                case "Downloading":
                  pct = Math.min(5, 3 + Math.round(elapsed / 60));
                  break;
                case "Indicators":  // Phase 1
                  pct = Math.min(8, 5 + Math.round(elapsed / 120));
                  break;
                case "Ranking": {   // Phase 2 — eval N/24
                  const m = (s.last_line || "").match(/eval\s+(\d+)\/(\d+)/i);
                  if (m) {
                    const cur = parseInt(m[1], 10);
                    const tot = parseInt(m[2], 10) || 24;
                    pct = Math.min(11, 8 + Math.round((cur / tot) * 3));
                    evalLabel = ` (eval ${cur}/${tot})`;
                  } else {
                    pct = 9;
                  }
                  break;
                }
                case "Validity":   // Phase 3
                  pct = 12;
                  break;
                case "Scoring":    // Phase 4
                  pct = 13;
                  break;
                case "Summary":    // MASTER SUMMARY (intermediate)
                  pct = 14;
                  break;
                case "Backtest":   // Phase 7 — ~50 weekly snapshots
                  pct = 15;
                  break;
                case "GraphRAG":   // Phase 6 — community detection
                  pct = 16;
                  break;
                case "Insights":   // KEY INSIGHTS section
                  pct = 16;
                  break;
                case "FactorEfficacy": {   // Phase 8 — [N/12] eval points
                  const m = (s.last_line || "").match(/\[(\d+)\/(\d+)\]/);
                  if (m) {
                    const cur = parseInt(m[1], 10);
                    const tot = parseInt(m[2], 10) || 12;
                    pct = Math.min(18, 16 + Math.round((cur / tot) * 2));
                    evalLabel = ` [${cur}/${tot}]`;
                  } else {
                    pct = 17;
                  }
                  break;
                }
                case "Output":     // PDF generation
                  pct = 18;
                  break;
                case "Done":
                  pct = 19;
                  break;
                default:
                  // Unknown phase — interpolate by elapsed (max 18% so we never lock at 19)
                  pct = Math.min(18, 3 + Math.round(elapsed / 60));
              }
              // Monotonic progress: never go backwards (prevents confusing reset when phase
              // mapping yields lower value than previous estimate)
              setProgressPct((prev) => Math.max(prev, pct));
              setProgressStage(`🔄 Live Scan: ${phase}${evalLabel}`);
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
        <div className="text-[#857F7A] text-[20px]">Connecting to API...</div>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-[#FFFFFF] border-r border-[#E6D9CE] p-4 overflow-y-auto shrink-0">
        <h1 className="text-[20px] font-bold text-[#0F5499] mb-1">Price Discovery</h1>
        <div className="text-[14px] text-[#857F7A] mb-2">Scanner v5.0 | {meta.total_tickers} tickers</div>

        {/* ── Asset Mode Toggle ── */}
        <div className="mb-3 flex rounded-lg overflow-hidden border border-[#E6D9CE]">
          <button
            className={`flex-1 py-1.5 text-[14px] font-semibold transition-colors ${
              assetMode === "equity"
                ? "bg-[#0F5499] text-white"
                : "bg-[#F2E5D7] text-[#66605C] hover:text-[#33302E]"
            }`}
            onClick={() => setAssetMode("equity")}>
            주식형
          </button>
          <button
            className={`flex-1 py-1.5 text-[14px] font-semibold transition-colors ${
              assetMode === "fixed_income"
                ? "bg-[#B85C00] text-white"
                : "bg-[#F2E5D7] text-[#66605C] hover:text-[#33302E]"
            }`}
            onClick={() => setAssetMode("fixed_income")}>
            FICC
          </button>
        </div>
        <div className="text-[12px] text-[#857F7A] mb-3">
          {assetMode === "equity"
            ? "Equity · Sectors · Factors · International · Thematic · Stocks"
            : "FICC — FI Short/Intermediate/Long/Credit/Inflation/Intl · Commodities · Real Assets · Currency · Multi-Asset"}
        </div>

        <div className="text-[14px] text-[#857F7A] mb-3">
          Scan: {meta.scan_time ? new Date(meta.scan_time).toLocaleString("ko-KR", {
            year: "numeric", month: "2-digit", day: "2-digit",
            hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
          }) : "N/A"}
        </div>

        {/* ── Run Live Scan ── */}
        <div className="mb-4 p-3 bg-[#FBEEE3] border border-[#E6D9CE] rounded-lg space-y-2">
          <div className="flex items-center gap-2">
            <label className="text-[12px] text-[#857F7A] w-16">Lookback</label>
            <select value={lookbackYears} onChange={(e) => setLookbackYears(+e.target.value)}
              className="flex-1 px-1.5 py-0.5 text-[14px] bg-[#F2E5D7] border border-[#E6D9CE] rounded" disabled={scanning}>
              <option value={1}>1 year</option>
              <option value={3}>3 years</option>
              <option value={5}>5 years</option>
            </select>
          </div>
          <label className="flex items-center gap-1.5 text-[13px] cursor-pointer">
            <input type="checkbox" checked={useRealtime} onChange={(e) => setUseRealtime(e.target.checked)}
              className="rounded bg-[#F2E5D7] border-[#CCC1B7] w-3 h-3" disabled={scanning} />
            Realtime prices
          </label>
          <label className="flex items-center gap-1.5 text-[13px] cursor-pointer">
            <input type="checkbox" checked={includeStocks} onChange={(e) => setIncludeStocks(e.target.checked)}
              className="rounded bg-[#F2E5D7] border-[#CCC1B7] w-3 h-3" disabled={scanning} />
            Include stocks
          </label>
          <button
            onClick={handleStartScan}
            disabled={scanning}
            title="전체 시스템 실행 (단일 버튼): (1) Live Scan — 770 ticker 데이터 갱신 · (2) Auto-Fill Debate Cache · (3) Market Leaders Swarm — Phase 5/5.5/5.55/5.6a Option A Iterative Debate (PM × 3 horizons → Trading → Risk LLM → Synthesizer) · (4) PM Agent Backtest — Stock-Picker Skill Eval. 총 ~15-20분 소요."
            className={`w-full py-1.5 rounded text-[14px] font-semibold transition-colors ${
              scanning
                ? "bg-[#F2E5D7] text-[#66605C] cursor-wait"
                : "bg-[#0F5499] hover:bg-[#0D7680] text-white"
            }`}>
            {scanning ? `Running… ${progressPct}%` : "Run Live Scan (전체 시스템)"}
          </button>

          {/* ── Progress Bar (visible during the whole 15-20 min pipeline) ── */}
          {(scanning || progressPct > 0) && (
            <div className="mt-1">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] font-bold" style={{ color: "#0D7680" }}>
                  {progressStage || "대기 중…"}
                </span>
                <span className="text-[12px] font-mono font-bold"
                      style={{ color: progressPct >= 100 ? "#0A7D3F" : "#0D7680" }}>
                  {progressPct}%
                </span>
              </div>
              {/* Progress bar */}
              <div className="w-full h-2 rounded-full overflow-hidden"
                   style={{ backgroundColor: "#F2E5D7", border: "1px solid #E6D9CE" }}>
                <div className="h-full rounded-full transition-all duration-500"
                     style={{
                       width: `${progressPct}%`,
                       background: progressPct >= 100
                         ? "linear-gradient(90deg, #0A7D3F 0%, #0A7D3F 100%)"
                         : "linear-gradient(90deg, #0D7680 0%, #0D7680 50%, #0D7680 100%)",
                       boxShadow: progressPct > 0 ? "0 0 8px rgba(34, 211, 238, 0.5)" : "none",
                     }} />
              </div>
              {/* Stage markers (4 milestones at 20/40/65/100) */}
              <div className="flex justify-between mt-1 text-[10px]" style={{ color: "#857F7A" }}>
                <span style={{ color: progressPct >= 20 ? "#0A7D3F" : "#857F7A" }}>
                  {progressPct >= 20 ? "✓" : "○"} Scan
                </span>
                <span style={{ color: progressPct >= 40 ? "#0A7D3F" : "#857F7A" }}>
                  {progressPct >= 40 ? "✓" : "○"} Cache
                </span>
                <span style={{ color: progressPct >= 80 ? "#0A7D3F" : progressPct >= 40 ? "#0D7680" : "#857F7A" }}>
                  {progressPct >= 80 ? "✓" : progressPct >= 40 ? "▶" : "○"} Swarm
                </span>
                <span style={{ color: progressPct >= 100 ? "#0A7D3F" : "#857F7A" }}>
                  {progressPct >= 100 ? "✓" : "○"} Done
                </span>
              </div>
            </div>
          )}

          {scanMsg && (
            <div className={`text-[12px] ${scanMsg.includes("Error") || scanMsg.includes("Failed") || scanMsg.includes("⚠") ? "text-[#CC0000]" : "text-[#0F5499]"}`}>
              {scanMsg}
            </div>
          )}
        </div>

        {/* ── Filters ── */}
        {/* Eligible Only */}
        <label className="flex items-center gap-2 text-[16px] mb-3 cursor-pointer">
          <input type="checkbox" checked={eligibleOnly} onChange={(e) => setEligibleOnly(e.target.checked)}
            className="rounded bg-[#F2E5D7] border-[#CCC1B7]" />
          Eligible only
        </label>

        {/* Composite Range */}
        <div className="mb-4">
          <div className="text-[14px] text-[#857F7A] mb-1">Composite: {compMin} — {compMax}</div>
          <div className="flex gap-2">
            <input type="number" value={compMin} onChange={(e) => setCompMin(+e.target.value)}
              className="w-16 px-1 py-0.5 text-[14px] bg-[#F2E5D7] border border-[#E6D9CE] rounded" min={0} max={100} />
            <input type="number" value={compMax} onChange={(e) => setCompMax(+e.target.value)}
              className="w-16 px-1 py-0.5 text-[14px] bg-[#F2E5D7] border border-[#E6D9CE] rounded" min={0} max={100} />
          </div>
        </div>

        {/* Sectors (Option B unified taxonomy) */}
        <div className="mb-4">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[14px] text-[#857F7A]">Sectors ({sectors.length}/{(meta.sectors || []).length})</span>
            <button className="text-[12px] text-[#0F5499] hover:underline"
              onClick={() => setSectors(sectors.length === (meta.sectors || []).length ? [] : (meta.sectors || []))}>
              {sectors.length === (meta.sectors || []).length ? "Clear" : "All"}
            </button>
          </div>
          <div className="max-h-48 overflow-y-auto space-y-0.5">
            {(meta.sectors || []).map((s: string) => (
              <label key={s} className="flex items-center gap-1.5 text-[13px] cursor-pointer hover:text-[#33302E]">
                <input type="checkbox" checked={sectors.includes(s)}
                  onChange={(e) => setSectors(e.target.checked ? [...sectors, s] : sectors.filter((x) => x !== s))}
                  className="rounded bg-[#F2E5D7] border-[#CCC1B7] w-3 h-3" />
                {s}
              </label>
            ))}
          </div>
        </div>

        {/* Classifications */}
        <div className="mb-4">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[14px] text-[#857F7A]">Classifications ({classifications.length}/{meta.classifications.length})</span>
            <button className="text-[12px] text-[#0F5499] hover:underline"
              onClick={() => setClassifications(classifications.length === meta.classifications.length ? [] : meta.classifications)}>
              {classifications.length === meta.classifications.length ? "Clear" : "All"}
            </button>
          </div>
          <div className="max-h-40 overflow-y-auto space-y-0.5">
            {meta.classifications.map((c: string) => (
              <label key={c} className="flex items-center gap-1.5 text-[13px] cursor-pointer hover:text-[#33302E]">
                <input type="checkbox" checked={classifications.includes(c)}
                  onChange={(e) => setClassifications(e.target.checked ? [...classifications, c] : classifications.filter((x) => x !== c))}
                  className="rounded bg-[#F2E5D7] border-[#CCC1B7] w-3 h-3" />
                {c.slice(0, 20)}
              </label>
            ))}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        {/* Tab Bar */}
        <div className="sticky top-0 z-20 bg-[#FFF1E5] border-b border-[#E6D9CE] px-4">
          <div className="flex items-center gap-0">
            <span className={`px-3 py-2.5 text-[12px] font-bold uppercase tracking-wider ${
              assetMode === "equity" ? "text-[#0F5499]" : "text-[#B85C00]"
            }`}>
              {assetMode === "equity" ? "주식형" : "FICC"}
            </span>
            <span className="text-[#33302E] mr-1">|</span>
            {TABS.map((t, i) => (
              <button key={t}
                className={`px-4 py-2.5 text-[16px] font-medium border-b-2 transition-colors ${
                  tab === i
                    ? assetMode === "equity" ? "border-[#0F5499] text-[#0F5499]" : "border-[#B85C00] text-[#B85C00]"
                    : "border-transparent text-[#857F7A] hover:text-[#33302E]"
                }`}
                onClick={() => setTab(i)}>
                {t}
              </button>
            ))}
          </div>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {tab === 0 && <MarketCommentaryTab filters={filters} dataVersion={dataVersion} scanning={scanning} />}
          {tab === 1 && <PriceDiscoveryTab filters={filters} />}
          {tab === 2 && <PriceDiscoveryMLTab filters={filters} />}
          {tab === 3 && <ValidationTab />}
          {tab === 4 && <MarketEnvironmentTab filters={filters} />}
          {tab === 5 && <AnalysisTab filters={filters} />}
          {tab === 6 && <AIPredictionTab />}
          {tab === 7 && <AppendixTab filters={filters} />}

          {/* Always-visible Final Buy/Sell List at the bottom of every tab —
              EXCEPT Market Commentary (tab 0), which embeds it right after Swarm Analysis. */}
          {tab !== 0 && <FinalListPanel dataVersion={dataVersion} scanning={scanning} />}
        </div>
      </main>
    </div>
  );
}
