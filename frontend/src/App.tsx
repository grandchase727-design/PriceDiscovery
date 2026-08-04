import { useEffect, useState, useMemo, useCallback, useRef } from "react";
import { fetchMeta, startScan, fetchScanStatus, reloadCache, startAutoFill, fetchAutoFillStatus, startMarketLeadersSwarm, startBacktestRun, fetchBacktestStatus, startTrancheRefresh, fetchTrancheStatus, type FilterParams } from "./api/client";
import { MarketEnvironmentTab } from "./components/tabs/MarketEnvironmentTab";
import { MarketCommentaryTab } from "./components/tabs/MarketCommentaryTab";
import { PriceDiscoveryTab } from "./components/tabs/PriceDiscoveryTab";
import { PriceDiscoveryMLTab } from "./components/tabs/PriceDiscoveryMLTab";
import { ValidationTab } from "./components/tabs/ValidationTab";
import { AnalysisTab } from "./components/tabs/AnalysisTab";
import { AppendixTab } from "./components/tabs/AppendixTab";
import { AIPredictionTab } from "./components/tabs/AIPredictionTab";
import FinalListPanel from "./components/shared/FinalListPanel";
import ElliottWavePanel from "./components/shared/ElliottWavePanel";
import FinalListEtfWavePanel from "./components/shared/FinalListEtfWavePanel";
import MarketInternalsPanel from "./components/shared/MarketInternalsPanel";
import PortfolioPanel from "./components/shared/PortfolioPanel";

const TABS = ["Market Commentary", "Price Discovery", "Price Discovery (ML)", "Validation", "Market Environment", "Analysis", "AI Prediction", "Appendix"];

export default function App() {
  const [meta, setMeta] = useState<any>(null);
  const [tab, setTab] = useState(0);
  const [dataVersion, setDataVersion] = useState(0); // bump to trigger re-fetch

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
  const [scanStartMs, setScanStartMs] = useState<number | null>(null);  // for ETA computation
  const [nowMs, setNowMs] = useState<number>(Date.now());                // ticks every 5s during scan

  const SWARM_PHASE_PROGRESS: Record<string, number> = {
    // Maps swarm.phase → progress % within the 40-95 swarm range (inside pipeline)
    phase1:           45,  // 5 analysts in parallel
    phase2:           52,  // coherence debater
    phase3:           57,  // synthesis (neutral + averse)
    phase4:           62,  // action selector
    phase4_action:    65,
    phase5:           70,  // PM — core horizon
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
      setSectors(m.sectors || []);
      setClassifications(m.classifications || []);
    });
  }, []);

  useEffect(() => { loadMeta(); }, [loadMeta]);

  // Poll scan status when scanning
  const handleStartScan = useCallback(() => {
    setScanning(true);
    setProgressPct(2);
    setProgressStage("🔄 Live Scan 시작");
    setScanMsg("Starting scan...");
    setScanStartMs(Date.now());   // anchor for ETA computation
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
                // The backend now owns the whole pipeline (scan → swarm → backtest →
                // final list) and keeps `running` True for the ENTIRE chain, so
                // reaching here means everything finished server-side — it completes
                // even if this tab was closed. We no longer start the swarm/backtest
                // from the browser (that used to get killed on tab close & double-run).
                const pipe = s.pipeline || {};
                const fl = pipe.final_list || {};
                const sw = pipe.swarm || {};
                const swWarn = sw.ok === false ? " · ⚠ 스웜 실패(기존 캐시로 생성)" : "";
                setProgressPct(100);
                setProgressStage("✓ 전체 파이프라인 완료");
                setScanMsg(
                  fl.ok
                    ? `✓ 완료 — 매수 ${fl.n_buy}개 · 매도 ${fl.n_sell}개 산출${swWarn}`
                    : `✓ 스캔 완료${swWarn}`,
                );
                reloadCache().then(() => {
                  loadMeta();
                  setDataVersion((v) => v + 1);
                });
                setScanning(false);
                // Best-effort: warm the per-ticker conviction-debate cache in the
                // background (non-blocking, Max-plan billed). Not part of the buy list.
                startAutoFill().catch(() => {});
                setTimeout(() => { setScanMsg(""); setProgressPct(0); setProgressStage(""); }, 12000);
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

              // Post-scan server pipeline (Swarm → PM_Backtest → FinalList). The
              // backend reports these AFTER the scan subprocess ends while `running`
              // stays True. Map to the 40-99% band; the swarm sub-phase arrives as
              // "Swarm:<phaseN>". (Live swarm detail also polled by SwarmAnalysis.)
              if (phase.startsWith("Swarm") || phase === "PM_Backtest" || phase === "FinalList") {
                let ppct = 42;
                let pstage = "🤖 Market Leaders Swarm";
                if (phase.startsWith("Swarm")) {
                  const sub = phase.split(":")[1] || "";
                  ppct = SWARM_PHASE_PROGRESS[sub] ?? 42;
                  pstage = `🤖 Swarm${sub ? " · " + sub : ""}`;
                } else if (phase === "PM_Backtest") {
                  ppct = 95;
                  pstage = "📊 PM Backtest";
                } else if (phase === "FinalList") {
                  ppct = 98;
                  pstage = "🧾 Final List 생성";
                }
                setProgressPct((prev) => Math.max(prev, ppct));
                setProgressStage(pstage);
                return;
              }

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
                case "Downloading": {
                  // Parse yfinance "X of Y completed" for live download progress
                  const m = (s.last_line || "").match(/(\d+)\s+of\s+(\d+)/i);
                  if (m) {
                    const cur = parseInt(m[1], 10);
                    const tot = parseInt(m[2], 10) || 1;
                    pct = Math.min(7, 4 + Math.round((cur / tot) * 3));
                    evalLabel = ` (다운로드 ${cur}/${tot})`;
                  } else {
                    pct = Math.min(5, 4 + Math.round(elapsed / 120));
                  }
                  break;
                }
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
                case "Validity": {   // SignalValidityEngine — "eval N/24"
                  const m = (s.last_line || "").match(/eval\s+(\d+)\/(\d+)/i);
                  if (m) {
                    const cur = parseInt(m[1], 10);
                    const tot = parseInt(m[2], 10) || 24;
                    pct = Math.min(15, 11 + Math.round((cur / tot) * 4));
                    evalLabel = ` (validity ${cur}/${tot})`;
                  } else {
                    pct = 12;
                  }
                  break;
                }
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

  // Tick a clock every 5s while scanning so the ETA display stays live.
  useEffect(() => {
    if (!scanning) return;
    const id = setInterval(() => setNowMs(Date.now()), 5000);
    return () => clearInterval(id);
  }, [scanning]);

  // ETA: extrapolate remaining time from elapsed × (100 - pct) / pct.
  // Only meaningful once progress has moved a bit (pct ≥ 5) and we have a start anchor.
  const eta = (() => {
    if (!scanning || !scanStartMs || progressPct < 5 || progressPct >= 100) return null;
    const elapsedMs = nowMs - scanStartMs;
    if (elapsedMs <= 0) return null;
    const totalMs = (elapsedMs / progressPct) * 100;
    const remainMs = Math.max(0, totalMs - elapsedMs);
    const finishAt = new Date(nowMs + remainMs);
    const remainMin = Math.round(remainMs / 60000);
    const hh = String(finishAt.getHours()).padStart(2, "0");
    const mm = String(finishAt.getMinutes()).padStart(2, "0");
    return { finishLabel: `${hh}:${mm}`, remainMin, elapsedMin: Math.round(elapsedMs / 60000) };
  })();

  // dataVersion in dependency forces all tabs to re-fetch after scan completes
  const filters: FilterParams = useMemo(() => ({
    sectors: sectors.length > 0 ? sectors : (meta?.sectors || []),
    classifications: classifications.length === (meta?.classifications?.length || 0) ? undefined : classifications,
    eligible_only: eligibleOnly,
    comp_min: compMin,
    comp_max: compMax,
    _v: dataVersion, // cache-bust key (not sent to API, triggers re-fetch)
  } as any), [sectors, classifications, eligibleOnly, compMin, compMax, meta, dataVersion]);

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
            title="전체 시스템 실행 (단일 버튼): (1) Live Scan — 810 ticker 데이터 갱신 · (2) Market Leaders Swarm — Phase 0~6 (Fact→Analyst→Strategist→Action→PM+Debate, core horizon) · 총 ~25-30분 소요."
            className={`w-full py-1.5 rounded text-[14px] font-semibold transition-colors ${
              scanning
                ? "bg-[#F2E5D7] text-[#66605C] cursor-wait"
                : "bg-[#0F5499] hover:bg-[#0D7680] text-white"
            }`}>
            {scanning ? `Running… ${progressPct}%` : "Run Live Scan (전체 시스템)"}
          </button>

          {/* ── ETA: 예상 완료 시간 (버튼 바로 밑) ── */}
          {eta && (
            <div className="flex items-center justify-between px-2 py-1 rounded text-[12px]"
                 style={{ backgroundColor: "#E3EEF5", border: "1px solid #9CC3D5" }}
                 title={`경과 ${eta.elapsedMin}분 · 진행률 ${progressPct}% 기반 선형 추정 (실제와 다를 수 있음)`}>
              <span style={{ color: "#0F5499", fontWeight: 600 }}>⏱ 예상 완료</span>
              <span className="font-mono font-bold" style={{ color: "#0F5499" }}>
                ~{eta.finishLabel} <span style={{ color: "#66605C", fontWeight: 400 }}>({eta.remainMin}분 남음)</span>
              </span>
            </div>
          )}

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

        {/* Sidebar filters (Composite / Sectors / Classifications) removed per request.
            filters state stays at show-all defaults (sectors = asset-mode scope,
            all classifications, composite 0-100) so downstream tabs receive a no-op filter. */}
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto">
        {/* Tab Bar */}
        <div className="sticky top-0 z-20 bg-[#FFF1E5] border-b border-[#E6D9CE] px-4">
          <div className="flex items-center gap-0">
            {TABS.map((t, i) => (
              <button key={t}
                className={`px-4 py-2.5 text-[16px] font-medium border-b-2 transition-colors ${
                  tab === i
                    ? "border-[#0F5499] text-[#0F5499]"
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
          {tab !== 0 && <MarketInternalsPanel />}
          {tab !== 0 && <ElliottWavePanel />}
          {tab !== 0 && <FinalListPanel dataVersion={dataVersion} scanning={scanning} />}
          {tab !== 0 && <FinalListEtfWavePanel />}
          {tab !== 0 && <PortfolioPanel dataVersion={dataVersion} scanning={scanning} />}
        </div>
      </main>
    </div>
  );
}
