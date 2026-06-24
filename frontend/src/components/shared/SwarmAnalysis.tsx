/**
 * SwarmAnalysis.tsx — Market Leaders 6-agent swarm output viewer.
 *
 * Renders:
 *   Phase 1: 4 domain analyst verdicts (Macro / Cross-Asset / Sector-Theme / Flow)
 *   Phase 2: coherence debate result (consensus or contested)
 *   Phase 3: dual-mode synthesis (Neutral / Averse) with toggle
 */
import { useEffect, useMemo, useState } from "react";
import {
  fetchSwarmResult,
  startMarketLeadersSwarm,
  fetchSwarmStatus,
  fetchTable,
  fetchPMHistorySummary,
  type SwarmResult,
  type SwarmPhase1Verdict,
  type SwarmStatus,
  type PMHistorySummary,
} from "../../api/client";
import { useSort } from "../../hooks/useSort";

// FT (Financial Times) palette — light pink paper, near-black text
const C = {
  bg: "#FFF1E5", bgAlt: "#FFFFFF", border: "#E6D9CE",
  text: "#33302E", gray: "#66605C",
  cyan: "#0D7680", purple: "#7D5BA6", green: "#0A7D3F",
  red: "#CC0000", amber: "#B85C00", yellow: "#B85C00",
};

const RATING_COLOR: Record<string, string> = {
  RISK_ON: C.green, PRO_GROWTH: C.green, BROAD_LEADERSHIP: C.green,
  ACCELERATING_LEADERSHIP: C.green, EMERGING_LEADERSHIP: C.green,
  CONFIRMS_RISK_ON: C.green,
  REFLATION: C.amber, LATE_CYCLE: C.amber, TRANSITIONAL: C.amber,
  ROTATION_IN_PROGRESS: C.amber, ROTATING_FLOWS: C.amber, MIXED: C.amber,
  DEFENSIVE: C.cyan, NARROW_LEADERSHIP: C.cyan, STALLING_LEADERSHIP: C.cyan,
  DIVERGES_FROM_EQUITY: C.amber,
  RISK_OFF: C.red, LEADERSHIP_DECAY: C.red, DECAYING_FLOWS: C.red,
  RISK_OFF_FLOWS: C.red, CONFIRMS_RISK_OFF: C.red,
};

function ratingColor(r: string): string {
  return RATING_COLOR[r] || C.gray;
}

const AGENT_META: Record<string, { label: string; emoji: string }> = {
  macro_analyst:          { label: "Macro",         emoji: "🌐" },
  cross_asset_analyst:    { label: "Cross-Asset",   emoji: "📊" },
  sector_theme_analyst:   { label: "Sector/Theme",  emoji: "🏛" },
  flow_momentum_analyst:  { label: "Flow/Momentum", emoji: "💧" },
  news_narrative_analyst: { label: "News Narrative", emoji: "📰" },
};

function Phase1Card({ id, v }: { id: string; v: SwarmPhase1Verdict | undefined }) {
  const meta = AGENT_META[id] || { label: id, emoji: "🤖" };
  if (!v) {
    return (
      <div className="rounded p-3" style={{ backgroundColor: C.bg, border: `1px solid ${C.border}` }}>
        <div className="text-[13px]" style={{ color: C.gray }}>{meta.emoji} {meta.label} — no output</div>
      </div>
    );
  }
  const rColor = ratingColor(v.rating);
  return (
    <div className="rounded p-3 h-full" style={{ backgroundColor: C.bg, border: `1px solid ${C.border}` }}>
      <div className="flex items-center justify-between mb-2 pb-1.5 border-b" style={{ borderColor: C.border }}>
        <span className="text-[13px] font-bold" style={{ color: C.text }}>
          {meta.emoji} {meta.label}
        </span>
        <span className="text-[12px] flex items-center gap-1.5">
          <span className="px-2 py-0.5 rounded font-bold whitespace-nowrap"
                style={{ backgroundColor: rColor + "25", color: rColor, border: `1px solid ${rColor}70` }}>
            {v.rating}
          </span>
          <span style={{ color: C.gray }}>conf {Math.round(v.confidence * 100)}%</span>
        </span>
      </div>

      {v.narrative && (
        <div className="text-[12px] leading-relaxed mb-2 whitespace-pre-wrap" style={{ color: C.text, lineHeight: 1.55 }}>
          {v.narrative}
        </div>
      )}

      {v.key_signals && v.key_signals.length > 0 && (
        <div className="mb-2">
          <div className="text-[11px] uppercase font-bold mb-0.5" style={{ color: C.cyan }}>Key Signals</div>
          <ul className="list-disc list-inside text-[12px] space-y-0.5" style={{ color: C.text, lineHeight: 1.45 }}>
            {v.key_signals.slice(0, 5).map((s, i) => <li key={i}>{s}</li>)}
          </ul>
        </div>
      )}

      {v.biggest_risk && (
        <div className="mb-1">
          <span className="text-[11px] uppercase font-bold" style={{ color: C.red }}>Risk: </span>
          <span className="text-[12px]" style={{ color: C.text }}>{v.biggest_risk}</span>
        </div>
      )}
      {v.biggest_opportunity && (
        <div className="mb-1">
          <span className="text-[11px] uppercase font-bold" style={{ color: C.green }}>Opp: </span>
          <span className="text-[12px]" style={{ color: C.text }}>{v.biggest_opportunity}</span>
        </div>
      )}

      {v.websearch_queries && v.websearch_queries.length > 0 && v.websearch_queries[0] !== "none" && (
        <div className="mt-1 text-[11px]" style={{ color: C.gray }}>
          🔍 {v.websearch_queries.join(" · ")}
        </div>
      )}
    </div>
  );
}

function PMHistoryBanner() {
  const [s, setS] = useState<PMHistorySummary | null>(null);
  useEffect(() => { fetchPMHistorySummary().then(setS).catch(() => setS(null)); }, []);
  if (!s || s.n_snapshots === 0) return null;
  return (
    <div className="mb-2 px-2.5 py-1.5 rounded text-[12px] flex items-center justify-between flex-wrap gap-2"
         style={{ backgroundColor: C.cyan + "12", border: `1px solid ${C.cyan}40` }}>
      <span style={{ color: C.cyan }}>
        📦 PM Picks Collection: <span className="font-bold">{s.n_snapshots} snapshot{s.n_snapshots !== 1 ? "s" : ""}</span>
        <span style={{ color: C.gray }}> · {s.first_date} → {s.last_date} · {s.unique_tickers} unique tickers</span>
      </span>
      {s.top_persistent_tickers && s.top_persistent_tickers.length > 0 && (
        <span style={{ color: C.gray }}>
          most-picked:
          {s.top_persistent_tickers.slice(0, 5).map((t) => (
            <span key={t.ticker} style={{ color: C.text, marginLeft: 6 }}>
              {t.ticker}<span style={{ color: C.gray }}>({t.pct.toFixed(0)}%)</span>
            </span>
          ))}
        </span>
      )}
    </div>
  );
}

export function SwarmAnalysis() {
  const [result, setResult] = useState<SwarmResult | null>(null);
  const [available, setAvailable] = useState(false);
  const [fresh, setFresh] = useState(false);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<SwarmStatus | null>(null);
  const [mode, setMode] = useState<"neutral" | "averse">("neutral");
  const [expanded, setExpanded] = useState(true);
  const [loading, setLoading] = useState(true);

  // Initial load
  useEffect(() => {
    fetchSwarmResult().then((r) => {
      setAvailable(r.available);
      setFresh(r.fresh);
      if (r.result) setResult(r.result);
    }).finally(() => setLoading(false));
  }, []);

  // Poll status when running
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      fetchSwarmStatus().then((st) => {
        setStatus(st);
        if (!st.running) {
          setRunning(false);
          // Refresh result when complete
          fetchSwarmResult().then((r) => {
            setAvailable(r.available); setFresh(r.fresh);
            if (r.result) setResult(r.result);
          });
        }
      }).catch(() => {});
    }, 4000);
    return () => clearInterval(id);
  }, [running]);

  const handleRun = (force: boolean) => {
    setRunning(true);
    startMarketLeadersSwarm(force).then((r) => {
      if (r.status === "cached" || r.status === "no_claude_cli" || r.status === "import_error") {
        setRunning(false);
      }
    }).catch(() => setRunning(false));
  };

  if (loading) {
    return <div className="text-[14px]" style={{ color: C.gray }}>Loading swarm analysis…</div>;
  }

  // Empty state — no cached result yet
  if (!available || !result) {
    return (
      <div className="border-l-4 rounded p-3"
           style={{ borderLeftColor: C.purple, border: `1px solid ${C.border}`, borderLeftWidth: 4 }}>
        <div className="flex items-center justify-between">
          <div>
            <div className="font-bold text-[16px]" style={{ color: C.purple }}>
              🤖 Market Leaders — 6-Agent Swarm Analysis
            </div>
            <div className="text-[14px] mt-0.5" style={{ color: C.gray }}>
              4 domain analysts × WebSearch → coherence check → dual synthesis (neutral/averse).
              Not yet run.
            </div>
          </div>
          <button onClick={() => handleRun(true)} disabled={running}
                  className="px-3 py-1.5 rounded text-[14px] font-bold transition-colors"
                  style={{ backgroundColor: C.purple + "30", color: C.purple, border: `1px solid ${C.purple}80` }}>
            {running ? "Running…" : "Run Swarm"}
          </button>
        </div>
        {running && status && (
          <div className="mt-2 text-[14px]" style={{ color: C.amber }}>
            [{status.phase}] {status.current}
          </div>
        )}
      </div>
    );
  }

  const syn = mode === "neutral" ? result.synthesis_neutral : result.synthesis_averse;
  const synColor = ratingColor(syn.regime_tag.toUpperCase().replace(/[^A-Z_]/g, "_"));

  return (
    <div className="border-l-4 rounded p-3"
         style={{ borderLeftColor: C.purple, border: `1px solid ${C.border}`, borderLeftWidth: 4 }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div>
          <button onClick={() => setExpanded((e) => !e)} className="font-bold text-[16px] flex items-center gap-1.5"
                  style={{ color: C.purple }}>
            <span>{expanded ? "▼" : "▶"}</span>
            🤖 Market Leaders — 6-Agent Swarm Analysis
          </button>
          <div className="text-[12px] mt-0.5" style={{ color: C.gray }}>
            generated {result.generated_at} · {fresh ? "✓ fresh" : "⚠ stale (>12h)"} ·
            deterministic tag: <span style={{ color: C.text }}>{result.snapshot.regime_tag_deterministic}</span>
          </div>
        </div>
        <button onClick={() => handleRun(true)} disabled={running}
                className="px-2 py-1 rounded text-[12px] font-bold transition-colors"
                style={{ backgroundColor: C.purple + "20", color: C.purple, border: `1px solid ${C.purple}50` }}>
          {running ? "Running…" : "Re-run"}
        </button>
      </div>

      {running && status && (
        <div className="mb-2 px-2 py-1.5 rounded text-[12px]"
             style={{ backgroundColor: C.amber + "15", color: C.amber, border: `1px solid ${C.amber}30` }}>
          [{status.phase}] {status.current} · {status.events.length} events
        </div>
      )}

      {/* PM history collection banner — populated as snapshots accumulate */}
      <PMHistoryBanner />

      {expanded && (
        <>
          {/* ── Phase 1: 5 Domain Analyst Cards ── */}
          <div className="mb-3">
            <div className="text-[12px] uppercase font-bold mb-1.5" style={{ color: C.gray }}>
              Phase 1 — 5 Domain Analysts (parallel, WebSearch + WebFetch)
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-2">
              <Phase1Card id="macro_analyst"          v={result.phase1.macro_analyst} />
              <Phase1Card id="cross_asset_analyst"    v={result.phase1.cross_asset_analyst} />
              <Phase1Card id="sector_theme_analyst"   v={result.phase1.sector_theme_analyst} />
              <Phase1Card id="flow_momentum_analyst"  v={result.phase1.flow_momentum_analyst} />
              <Phase1Card id="news_narrative_analyst" v={result.phase1.news_narrative_analyst} />
            </div>
          </div>

          {/* ── Phase 2: Coherence Debate ── */}
          <div className="mb-3 rounded p-2.5"
               style={{ backgroundColor: result.phase2.coherent ? C.green + "10" : C.amber + "10",
                        border: `1px solid ${result.phase2.coherent ? C.green : C.amber}40` }}>
            <div className="text-[12px] uppercase font-bold mb-1"
                 style={{ color: result.phase2.coherent ? C.green : C.amber }}>
              {result.phase2.coherent ? "✓" : "⚠"} Phase 2 — Coherence Debate
            </div>
            <div className="text-[13px] mb-1.5" style={{ color: C.text }}>
              <span className="font-semibold">Dominant signal: </span>{result.phase2.dominant_signal}
            </div>
            {result.phase2.contested_areas && result.phase2.contested_areas.length > 0 && (
              <div className="mb-1.5">
                <span className="text-[11px] uppercase font-bold" style={{ color: C.amber }}>Contested:</span>
                <ul className="list-disc list-inside text-[12px] space-y-0.5 mt-0.5" style={{ color: C.text }}>
                  {result.phase2.contested_areas.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            )}
            <div className="text-[12px] leading-relaxed whitespace-pre-wrap" style={{ color: C.text, lineHeight: 1.5 }}>
              {result.phase2.reasoning}
            </div>
          </div>

          {/* ── Phase 3: Dual Synthesis ── */}
          <div className="rounded p-3" style={{ backgroundColor: C.bg, border: `1px solid ${synColor}40` }}>
            <div className="flex items-center justify-between mb-2 pb-1.5 border-b" style={{ borderColor: C.border }}>
              <div className="flex items-center gap-2">
                <span className="text-[12px] uppercase font-bold" style={{ color: C.purple }}>
                  Phase 3 — Synthesis Arbitrator
                </span>
                <div className="inline-flex border rounded text-[12px]" style={{ borderColor: C.border }}>
                  {(["neutral", "averse"] as const).map((m) => (
                    <button key={m} onClick={() => setMode(m)}
                            className="px-3.5 py-1"
                            style={{
                              backgroundColor: mode === m ? C.purple + "30" : "transparent",
                              color: mode === m ? C.purple : C.gray,
                              fontWeight: mode === m ? "bold" : "normal",
                            }}>
                      {m === "neutral" ? "중립" : "보수적"}
                    </button>
                  ))}
                </div>
              </div>
              <span className="text-[14px] px-2 py-0.5 rounded font-bold"
                    style={{ backgroundColor: synColor + "25", color: synColor, border: `1px solid ${synColor}80` }}>
                {syn.regime_tag} · conf {Math.round(syn.confidence * 100)}%
              </span>
            </div>

            <div className="text-[16px] leading-relaxed whitespace-pre-wrap mb-3"
                 style={{ color: C.text, lineHeight: 1.65 }}>
              {syn.narrative}
            </div>

            {syn.historical_analog && (
              <div className="mb-2 px-2 py-1.5 rounded text-[13px]"
                   style={{ backgroundColor: C.bgAlt, borderLeft: `3px solid ${C.cyan}` }}>
                <span className="font-bold uppercase text-[11px]" style={{ color: C.cyan }}>Historical analog: </span>
                <span style={{ color: C.text }}>{syn.historical_analog}</span>
              </div>
            )}

            {syn.watch_triggers && syn.watch_triggers.length > 0 && (
              <div className="mb-2">
                <div className="text-[11px] uppercase font-bold mb-0.5" style={{ color: C.amber }}>
                  Watch Triggers (regime-flip conditions)
                </div>
                <ul className="list-disc list-inside text-[13px] space-y-0.5" style={{ color: C.text }}>
                  {syn.watch_triggers.map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              </div>
            )}

            {syn.key_risks && syn.key_risks.length > 0 && (
              <div>
                <div className="text-[11px] uppercase font-bold mb-0.5" style={{ color: C.red }}>
                  Key Risks
                </div>
                <ul className="list-disc list-inside text-[13px] space-y-0.5" style={{ color: C.text }}>
                  {syn.key_risks.map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              </div>
            )}

            <div className="mt-2 text-[11px]" style={{ color: C.gray }}>
              Cross-panel coherence: {Math.round(syn.cross_panel_coherence_score * 100)}%
              {Object.keys(result.phase1_errors || {}).length > 0 && (
                <span style={{ color: C.red }}>
                  {" · "}⚠ {Object.keys(result.phase1_errors).length} Phase 1 errors
                </span>
              )}
            </div>
          </div>

          {/* ── Phase 4: Action Selector (draft picks + GICS + universe) ── */}
          {result.phase4_action && <Phase4Output a={result.phase4_action} />}

          {/* ── Phase 5: PM Agent (final portfolio-constructed picks with diff) ── */}
          {result.phase5_pm && <Phase5Output pm={result.phase5_pm} p4={result.phase4_action} />}
        </>
      )}
    </div>
  );
}

function PicksTable({ title, color, picks, isShort }:
  { title: string; color: string; picks: import("../../api/client").SwarmPick[]; isShort: boolean }) {
  if (!picks || picks.length === 0) {
    return (
      <div>
        <div className="text-[13px] font-bold mb-1" style={{ color }}>{title}</div>
        <div className="text-[12px]" style={{ color: C.gray }}>—</div>
      </div>
    );
  }
  return (
    <div>
      <div className="text-[13px] font-bold mb-1 flex items-center justify-between" style={{ color }}>
        <span>{title}</span>
        <span className="text-[12px]" style={{ color: C.gray }}>{picks.length} picks</span>
      </div>
      <div className="overflow-auto" style={{ maxHeight: 520 }}>
      <table className="w-full text-[12px] border-collapse">
        <thead className="sticky top-0" style={{ backgroundColor: "#FFF1E5", zIndex: 1 }}>
          <tr style={{ borderBottom: `1px solid ${C.border}` }}>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>#</th>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Ticker</th>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Name</th>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Sector</th>
            <th className="text-right py-1 px-1.5" style={{ color: C.gray }}>Comp</th>
            <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Rationale</th>
          </tr>
        </thead>
        <tbody>
          {picks.slice(0, 20).map((p, i) => {
            const cColor = isShort
              ? (p.composite < 25 ? C.red : C.amber)
              : (p.composite >= 70 ? C.green : C.cyan);
            return (
              <tr key={i} style={{ borderBottom: `1px solid ${C.border}40` }}>
                <td className="py-1 px-1.5" style={{ color: C.gray }}>{i + 1}</td>
                <td className="py-1 px-1.5 font-mono font-bold" style={{ color: C.text }}>{p.ticker}</td>
                <td className="py-1 px-1.5" style={{ color: C.text }}>{p.name}</td>
                <td className="py-1 px-1.5 whitespace-nowrap" style={{ color: C.cyan }}>{p.sector || "—"}</td>
                <td className="py-1 px-1.5 text-right font-mono" style={{ color: cColor }}>{p.composite}</td>
                <td className="py-1 px-1.5" style={{ color: C.text, lineHeight: 1.4 }}>{p.rationale}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      </div>
    </div>
  );
}

function SectorScores({ scores }: { scores: import("../../api/client").SwarmSectorScore[] }) {
  if (!scores || scores.length === 0) return null;
  const sorted = [...scores].sort((a, b) => b.score - a.score);
  return (
    <div>
      <div className="text-[13px] font-bold mb-1.5" style={{ color: C.purple }}>
        🏛 GICS 11 Sector Scoring
      </div>
      <div className="space-y-1">
        {sorted.map((s) => {
          const c = s.score >= 70 ? C.green : s.score >= 55 ? C.cyan : s.score >= 40 ? C.amber : C.red;
          return (
            <div key={s.sector} className="flex items-center gap-2 text-[12px]">
              <div className="w-32 font-semibold" style={{ color: C.text }}>{s.sector}</div>
              <div className="flex-1 rounded overflow-hidden" style={{ backgroundColor: C.bgAlt, height: 14 }}>
                <div style={{ width: `${s.score}%`, backgroundColor: c + "80", height: "100%" }} />
              </div>
              <div className="w-8 text-right font-mono font-bold" style={{ color: c }}>{s.score}</div>
              <div className="flex-1 text-[12px]" style={{ color: C.gray, lineHeight: 1.35 }}>{s.rationale}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Phase4Output({ a }: { a: import("../../api/client").SwarmActionOutput }) {
  const [universeView, setUniverseView] = useState<"stock" | "etf" | null>(null);
  const [universe, setUniverse] = useState<any[] | null>(null);
  const [loadingUniverse, setLoadingUniverse] = useState(false);

  useEffect(() => {
    if (universeView && !universe && !loadingUniverse) {
      setLoadingUniverse(true);
      fetchTable({}).then((res) => setUniverse(res.data || []))
        .catch(() => setUniverse([]))
        .finally(() => setLoadingUniverse(false));
    }
  }, [universeView, universe, loadingUniverse]);

  const { stockCount, etfCount } = useMemo(() => {
    if (!universe) return { stockCount: 0, etfCount: 0 };
    let s = 0, e = 0;
    for (const r of universe) {
      // /api/table cleans the STK_ prefix from category, so use asset_type explicitly
      const at = (r.asset_type || "").toLowerCase();
      if (at === "stock") s += 1;
      else if (at === "etf") e += 1;
    }
    return { stockCount: s, etfCount: e };
  }, [universe]);

  if (a._error) {
    return (
      <div className="mt-3 rounded p-2.5 text-[13px]"
           style={{ backgroundColor: C.red + "15", border: `1px solid ${C.red}40`, color: C.red }}>
        ⚠ Phase 4 Action Selector failed: {a._error}
      </div>
    );
  }
  return (
    <div className="mt-3 rounded p-3"
         style={{ backgroundColor: C.bg, border: `1px solid ${C.purple}40` }}>
      <div className="text-[12px] uppercase font-bold mb-2" style={{ color: C.purple }}>
        Phase 4 — Action Selector (regime-driven picks + sectors + themes)
      </div>

      {/* Picks — 4-quadrant grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-3">
        <PicksTable title="📈 Top 20 LONG — Stocks"  color={C.green} picks={a.long_stocks}  isShort={false} />
        <PicksTable title="📈 Top 20 LONG — ETFs"    color={C.green} picks={a.long_etfs}    isShort={false} />
        <PicksTable title="📉 Top 20 SHORT — Stocks" color={C.red}   picks={a.short_stocks} isShort={true} />
        <PicksTable title="📉 Top 20 SHORT — ETFs"   color={C.red}   picks={a.short_etfs}   isShort={true} />
      </div>

      {/* Universe expand toggles */}
      <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: C.border }}>
        <span className="text-[12px] uppercase font-bold mr-1" style={{ color: C.gray }}>
          Full Universe View:
        </span>
        {(["stock", "etf"] as const).map((kind) => {
          const isActive = universeView === kind;
          const baseColor = kind === "stock" ? C.green : C.cyan;
          const n = kind === "stock" ? stockCount : etfCount;
          return (
            <button key={kind}
                    onClick={() => setUniverseView((v) => (v === kind ? null : kind))}
                    className="px-3 py-1.5 rounded text-[14px] transition-colors"
                    style={{
                      backgroundColor: isActive ? baseColor + "25" : "transparent",
                      color: isActive ? baseColor : C.gray,
                      border: `1px solid ${isActive ? baseColor + "80" : C.border}`,
                      fontWeight: isActive ? "bold" : "normal",
                    }}>
              {kind === "stock" ? "📊 전체 Stocks" : "📦 전체 ETFs"}
              {n > 0 && <span className="ml-1.5" style={{ opacity: 0.6 }}>({n})</span>}
              {isActive ? " ▴" : " ▾"}
            </button>
          );
        })}
        {loadingUniverse && (
          <span className="text-[12px]" style={{ color: C.gray }}>Loading universe…</span>
        )}
      </div>

      {/* Universe table */}
      {universeView && universe && (
        <UniverseTable rows={universe} view={universeView} />
      )}

      {/* GICS sector scoring */}
      <div className="mt-3">
        <SectorScores scores={a.sector_scores} />
      </div>
    </div>
  );
}

function UniverseTable({ rows, view }: { rows: any[]; view: "stock" | "etf" }) {
  const filtered = useMemo(() => {
    return rows.filter((r) => {
      const at = (r.asset_type || "").toLowerCase();
      return view === "stock" ? at === "stock" : at === "etf";
    });
  }, [rows, view]);

  const accessors = useMemo(() => ({
    ticker:         (r: any) => (r.ticker || "").toString(),
    name:           (r: any) => (r.name || "").toString(),
    sector:         (r: any) => (r.sector || r.category || "").toString(),
    composite:      (r: any) => r.composite ?? 0,
    classification: (r: any) => (r.classification || "").toString(),
    oer:            (r: any) => r.oer ?? 0,
    ret_1m:         (r: any) => r.ret_1m ?? r.return_1m ?? 0,
    eligible:       (r: any) => (r.eligible ? 1 : 0),
  }), []);

  const { sorted, onSort, indicator } = useSort(filtered, accessors,
    { key: "composite", dir: "desc" });

  const headerColor = view === "stock" ? C.green : C.cyan;
  const headerLabel = view === "stock" ? "📊 전체 Stock Universe" : "📦 전체 ETF Universe";

  const thCls = "py-1 px-1.5 cursor-pointer select-none whitespace-nowrap";

  return (
    <div className="rounded p-3" style={{ backgroundColor: C.bg, border: `1px solid ${headerColor}40` }}>
      <div className="flex items-center justify-between mb-2">
        <div className="text-[13px] font-bold" style={{ color: headerColor }}>
          {headerLabel}
        </div>
        <div className="text-[12px]" style={{ color: C.gray }}>
          {filtered.length} tickers · click column to sort
        </div>
      </div>

      <div className="overflow-auto" style={{ maxHeight: 520 }}>
        <table className="w-full text-[12px] border-collapse">
          <thead className="sticky top-0" style={{ backgroundColor: "#FFF1E5", zIndex: 1 }}>
            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
              <th className="py-1 px-1.5 text-left" style={{ color: C.gray }}>#</th>
              <th className={`${thCls} text-left`}   onClick={() => onSort("ticker")}        style={{ color: C.gray }}>Ticker{indicator("ticker")}</th>
              <th className={`${thCls} text-left`}   onClick={() => onSort("name")}          style={{ color: C.gray }}>Name{indicator("name")}</th>
              <th className={`${thCls} text-left`}   onClick={() => onSort("sector")}        style={{ color: C.gray }}>Sector{indicator("sector")}</th>
              <th className={`${thCls} text-right`}  onClick={() => onSort("composite")}     style={{ color: C.gray }}>Comp{indicator("composite")}</th>
              <th className={`${thCls} text-left`}   onClick={() => onSort("classification")}style={{ color: C.gray }}>Class{indicator("classification")}</th>
              <th className={`${thCls} text-right`}  onClick={() => onSort("oer")}           style={{ color: C.gray }}>OER{indicator("oer")}</th>
              <th className={`${thCls} text-right`}  onClick={() => onSort("ret_1m")}        style={{ color: C.gray }}>1M{indicator("ret_1m")}</th>
              <th className={`${thCls} text-center`} onClick={() => onSort("eligible")}      style={{ color: C.gray }}>Elig{indicator("eligible")}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => {
              const comp = r.composite ?? 0;
              const cColor = comp >= 70 ? C.green : comp >= 55 ? C.cyan : comp >= 35 ? C.gray : C.red;
              const oer = r.oer ?? 0;
              const oerColor = oer >= 60 ? C.red : oer >= 40 ? C.amber : C.gray;
              const ret = r.ret_1m ?? r.return_1m ?? 0;
              const retColor = ret > 5 ? C.green : ret < -5 ? C.red : C.gray;
              const elig = !!r.eligible;
              return (
                <tr key={r.ticker} style={{ borderBottom: `1px solid ${C.border}40` }}>
                  <td className="py-1 px-1.5" style={{ color: C.gray }}>{i + 1}</td>
                  <td className="py-1 px-1.5 font-mono font-bold" style={{ color: C.text }}>{r.ticker}</td>
                  <td className="py-1 px-1.5" style={{ color: C.text }}>{(r.name || "").slice(0, 32)}</td>
                  <td className="py-1 px-1.5 whitespace-nowrap" style={{ color: C.cyan }}>{r.sector || r.category || "—"}</td>
                  <td className="py-1 px-1.5 text-right font-mono" style={{ color: cColor }}>{Number(comp).toFixed(1)}</td>
                  <td className="py-1 px-1.5 whitespace-nowrap" style={{ color: C.text }}>{r.classification || "—"}</td>
                  <td className="py-1 px-1.5 text-right font-mono" style={{ color: oerColor }}>{Number(oer).toFixed(0)}</td>
                  <td className="py-1 px-1.5 text-right font-mono" style={{ color: retColor }}>
                    {ret >= 0 ? "+" : ""}{Number(ret).toFixed(1)}%
                  </td>
                  <td className="py-1 px-1.5 text-center" style={{ color: elig ? C.green : C.gray }}>
                    {elig ? "✓" : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Phase 5 — PM Agent (portfolio-constructed picks with diff badges)
// ─────────────────────────────────────────────────────────────────────

const CHANGE_BADGE: Record<string, { label: string; color: string; bg: string }> = {
  SAME:     { label: "=",         color: C.gray,   bg: "#F2E5D7" },
  PROMOTED: { label: "▲ UP",      color: C.green,  bg: C.green + "20" },
  DEMOTED:  { label: "▼ DOWN",    color: C.amber,  bg: C.amber + "20" },
  NEW:      { label: "★ NEW",     color: C.cyan,   bg: C.cyan + "20" },
};

function ChangeBadge({ change_type, reason }: { change_type: string; reason?: string }) {
  const meta = CHANGE_BADGE[change_type] || CHANGE_BADGE.SAME;
  return (
    <span title={reason || ""}
          className="text-[11px] px-2 py-0.75 rounded font-bold whitespace-nowrap cursor-help"
          style={{ color: meta.color, backgroundColor: meta.bg,
                   border: `1px solid ${meta.color}50` }}>
      {meta.label}
    </span>
  );
}

const SIGNAL_META: Record<string, { emoji: string; color: string; label: string }> = {
  BUY_NOW: { emoji: "🟢", color: C.green, label: "BUY NOW" },
  WAIT:    { emoji: "🟡", color: C.amber, label: "WAIT" },
  SKIP:    { emoji: "🔴", color: C.red,   label: "SKIP" },
};
const URGENCY_META: Record<string, { emoji: string; color: string }> = {
  URGENT:  { emoji: "🚨", color: C.red },
  NORMAL:  { emoji: "▶",  color: C.gray },
  PATIENT: { emoji: "⏸",  color: C.cyan },
};

function TimingBadge({ timing }: { timing?: import("../../api/client").PickTiming }) {
  if (!timing) return <span className="text-[11px]" style={{ color: C.gray }}>—</span>;
  const sig = SIGNAL_META[timing.entry_signal] || { emoji: "?", color: C.gray, label: timing.entry_signal };
  const urg = URGENCY_META[timing.urgency] || { emoji: " ", color: C.gray };
  return (
    <span title={timing.rationale} className="text-[11px] inline-flex items-center gap-1 cursor-help">
      <span className="px-2 py-0.75 rounded font-bold whitespace-nowrap"
            style={{ color: sig.color, backgroundColor: sig.color + "20",
                     border: `1px solid ${sig.color}80` }}>
        {sig.emoji} {sig.label}
      </span>
      <span title={`Urgency: ${timing.urgency}`} style={{ color: urg.color, fontSize: 13 }}>
        {urg.emoji}
      </span>
    </span>
  );
}

// ── Phase 5.6 Position State badges ──
const STATE_META: Record<string, { emoji: string; color: string; label: string }> = {
  PROSPECTING:  { emoji: "⏳",  color: C.gray,   label: "WATCH" },
  ENTERED:      { emoji: "✓",   color: C.green,  label: "ENTERED" },
  HOLDING:      { emoji: "▶",   color: C.cyan,   label: "HOLDING" },
  EXIT_PENDING: { emoji: "⚠",   color: C.amber,  label: "EXIT_P" },
  EXITED:       { emoji: "●",   color: C.gray,   label: "EXITED" },
  DROPPED:      { emoji: "✗",   color: C.gray,   label: "DROPPED" },
};

function StateBadge({ ps }: { ps?: import("../../api/client").PickPositionState }) {
  if (!ps) return <span className="text-[11px]" style={{ color: C.gray }}>—</span>;
  const m = STATE_META[ps.state] || { emoji: "?", color: C.gray, label: ps.state };
  const days = ps.days_in_state || 0;
  const cnt = ps.consecutive_signal_days || { BUY_NOW: 0, WAIT: 0, SKIP: 0 };
  const tooltip = `State: ${ps.state} · ${days}d in state\nSignals — BUY_NOW:${cnt.BUY_NOW}d WAIT:${cnt.WAIT}d SKIP:${cnt.SKIP}d${ps.entered_date ? `\nEntered: ${ps.entered_date}` : ""}`;
  return (
    <span title={tooltip} className="text-[11px] inline-flex items-center gap-1 cursor-help whitespace-nowrap">
      <span className="px-2 py-0.5 rounded font-bold whitespace-nowrap"
            style={{ color: m.color, backgroundColor: m.color + "20",
                     border: `1px solid ${m.color}80` }}>
        {m.emoji} {m.label}
      </span>
      {days > 0 && <span style={{ color: C.gray }}>{days}d</span>}
    </span>
  );
}

function AlertBadge({ alert }: { alert?: string | null }) {
  if (!alert) return null;
  const color =
    alert.includes("NEW BUY") ? C.green :
    alert.includes("EXIT TRIGGER") ? C.red :
    alert.includes("DEGRADED") ? C.amber :
    alert.includes("DROPPED") ? C.gray : C.purple;
  return (
    <span className="text-[11px] px-2 py-0.75 rounded font-bold animate-pulse whitespace-nowrap ml-1"
          style={{ color, backgroundColor: color + "25",
                   border: `1.5px solid ${color}` }}>
      {alert}
    </span>
  );
}

function PMTable({ title, color, picks, isShort }:
  { title: string; color: string;
    picks: import("../../api/client").SwarmPMPick[]; isShort: boolean }) {
  if (!picks || picks.length === 0) {
    return (
      <div>
        <div className="text-[13px] font-bold mb-1" style={{ color }}>{title}</div>
        <div className="text-[12px]" style={{ color: C.gray }}>—</div>
      </div>
    );
  }
  const counts = picks.reduce((acc, p) => {
    const k = p.change_type || "SAME";
    acc[k] = (acc[k] || 0) + 1; return acc;
  }, {} as Record<string, number>);
  return (
    <div>
      <div className="text-[13px] font-bold mb-1 flex items-center justify-between" style={{ color }}>
        <span>{title}</span>
        <span className="text-[12px] flex items-center gap-1.5" style={{ color: C.gray }}>
          {counts.NEW ? <span style={{ color: C.cyan }}>★{counts.NEW}</span> : null}
          {counts.PROMOTED ? <span style={{ color: C.green }}>▲{counts.PROMOTED}</span> : null}
          {counts.DEMOTED ? <span style={{ color: C.amber }}>▼{counts.DEMOTED}</span> : null}
          {counts.SAME ? <span>={counts.SAME}</span> : null}
        </span>
      </div>
      <div className="overflow-auto" style={{ maxHeight: 520 }}>
        <table className="w-full text-[12px] border-collapse">
          <thead className="sticky top-0" style={{ backgroundColor: "#FFF1E5", zIndex: 1 }}>
            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>#</th>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Δ</th>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Ticker</th>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Name</th>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Sector</th>
              <th className="text-right py-1 px-1.5" style={{ color: C.gray }}>Comp</th>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>🎯 Signal</th>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>📊 State</th>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Entry Trigger</th>
              <th className="text-left py-1 px-1.5" style={{ color: C.gray }}>Rationale · PM reason</th>
            </tr>
          </thead>
          <tbody>
            {picks.slice(0, 20).map((p, i) => {
              const cColor = isShort
                ? (p.composite < 25 ? C.red : C.amber)
                : (p.composite >= 70 ? C.green : C.cyan);
              const ct = p.change_type || "SAME";
              const ps = p.position_state;
              // Row highlight priority: ALERT > change_type
              const hasAlert = !!ps?.alert;
              const rowBg = hasAlert
                ? (ps!.alert!.includes("NEW BUY") ? C.green + "1a"
                  : ps!.alert!.includes("EXIT TRIGGER") ? C.red + "1a"
                  : C.amber + "12")
                : ct === "NEW" ? C.cyan + "08"
                : ct === "PROMOTED" ? C.green + "08"
                : ct === "DEMOTED"  ? C.amber + "08"
                : "transparent";
              const timing = p.timing;
              return (
                <tr key={i} style={{ borderBottom: `1px solid ${C.border}40`, backgroundColor: rowBg }}>
                  <td className="py-1 px-1.5" style={{ color: C.gray }}>{i + 1}</td>
                  <td className="py-1 px-1.5"><ChangeBadge change_type={ct} reason={p.change_reason} /></td>
                  <td className="py-1 px-1.5 font-mono font-bold" style={{ color: C.text }}>
                    {p.ticker}
                    <AlertBadge alert={ps?.alert} />
                  </td>
                  <td className="py-1 px-1.5" style={{ color: C.text }}>{(p.name || "").slice(0, 30)}</td>
                  <td className="py-1 px-1.5 whitespace-nowrap" style={{ color: C.cyan }}>{p.sector || "—"}</td>
                  <td className="py-1 px-1.5 text-right font-mono" style={{ color: cColor }}>{p.composite}</td>
                  <td className="py-1 px-1.5 whitespace-nowrap"><TimingBadge timing={timing} /></td>
                  <td className="py-1 px-1.5 whitespace-nowrap"><StateBadge ps={ps} /></td>
                  <td className="py-1 px-1.5 text-[12px]" style={{ color: C.text, lineHeight: 1.35, maxWidth: 220 }}>
                    {timing?.entry_trigger || "—"}
                    {timing?.exit_triggers && timing.exit_triggers.length > 0 && (
                      <div className="text-[11px] mt-0.5" style={{ color: C.gray }}>
                        exit: {timing.exit_triggers[0].type} · {timing.exit_triggers[0].action}
                      </div>
                    )}
                  </td>
                  <td className="py-1 px-1.5" style={{ color: C.text, lineHeight: 1.4 }}>
                    {p.rationale}
                    {p.change_reason && ct !== "SAME" && (
                      <div className="text-[11px] mt-0.5" style={{ color: C.amber, fontStyle: "italic" }}>
                        PM: {p.change_reason}
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

type HorizonKey = "tactical" | "core" | "strategic";

const HORIZON_META: Record<HorizonKey, { label: string; emoji: string; days: string; sub: string; color: string }> = {
  tactical:  { label: "Tactical",  emoji: "🚀", days: "5d",  sub: "1-week catalysts · News + Cross-Asset + Flow",       color: C.amber },
  core:      { label: "Core",      emoji: "⚓", days: "21d", sub: "1-month regime · all Phase 1 + Phase 3 (Phase 4 diff)", color: C.purple },
  strategic: { label: "Strategic", emoji: "🌐", days: "63d", sub: "3-month thesis · Macro + Sector/Theme dominant",       color: C.cyan },
};

function Phase5Output({ pm, p4 }:
  { pm: import("../../api/client").SwarmPMOutput;
    p4?: import("../../api/client").SwarmActionOutput }) {
  void p4;
  const [horizon, setHorizon] = useState<HorizonKey>("core");

  // Pull picks from new schema (horizons.{key}.{bucket}) with legacy fallback
  const picks = (pm.horizons && pm.horizons[horizon]) || {
    long_stocks:  (horizon === "core" ? pm.long_stocks  : []) || [],
    long_etfs:    (horizon === "core" ? pm.long_etfs    : []) || [],
    short_stocks: (horizon === "core" ? pm.short_stocks : []) || [],
    short_etfs:   (horizon === "core" ? pm.short_etfs   : []) || [],
  };

  const horizonCounts = {
    tactical:  (pm.horizons?.tactical?.long_stocks?.length  || 0) +
               (pm.horizons?.tactical?.long_etfs?.length    || 0) +
               (pm.horizons?.tactical?.short_stocks?.length || 0) +
               (pm.horizons?.tactical?.short_etfs?.length   || 0),
    core:      (pm.horizons?.core?.long_stocks?.length  || pm.long_stocks?.length  || 0) +
               (pm.horizons?.core?.long_etfs?.length    || pm.long_etfs?.length    || 0) +
               (pm.horizons?.core?.short_stocks?.length || pm.short_stocks?.length || 0) +
               (pm.horizons?.core?.short_etfs?.length   || pm.short_etfs?.length   || 0),
    strategic: (pm.horizons?.strategic?.long_stocks?.length  || 0) +
               (pm.horizons?.strategic?.long_etfs?.length    || 0) +
               (pm.horizons?.strategic?.short_stocks?.length || 0) +
               (pm.horizons?.strategic?.short_etfs?.length   || 0),
  };

  if (pm._error) {
    return (
      <div className="mt-3 rounded p-2.5 text-[13px]"
           style={{ backgroundColor: C.red + "15", border: `1px solid ${C.red}40`, color: C.red }}>
        ⚠ Phase 5 PM Agent failed: {pm._error}
      </div>
    );
  }
  const dropsByBucket = (pm.phase4_drops || []).reduce((acc, d) => {
    (acc[d.bucket] = acc[d.bucket] || []).push(d);
    return acc;
  }, {} as Record<string, import("../../api/client").SwarmPhase4Drop[]>);

  return (
    <div className="mt-3 rounded p-3"
         style={{ backgroundColor: C.bg, border: `2px solid ${C.purple}80` }}>
      <div className="flex items-center justify-between mb-2 pb-2 border-b" style={{ borderColor: C.border }}>
        <div className="text-[13px] uppercase font-bold" style={{ color: C.purple }}>
          🎯 Phase 5 — PM Agent (3-horizon portfolio-constructed picks)
        </div>
        <div className="flex items-center gap-3 text-[11px]" style={{ color: C.gray }}>
          <span>Legend:</span>
          <ChangeBadge change_type="NEW" />
          <ChangeBadge change_type="PROMOTED" />
          <ChangeBadge change_type="DEMOTED" />
          <ChangeBadge change_type="SAME" />
        </div>
      </div>

      {/* Iterative Swarm — Convergence summary (Option 2: 5-round convergent) */}
      {pm.iteration && (
        <div className="mb-3 px-3 py-2 rounded"
             style={{ backgroundColor: C.cyan + "12",
                      border: `1.5px solid ${C.cyan}60`,
                      borderLeft: `5px solid ${C.cyan}` }}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[13px] uppercase font-bold tracking-wide" style={{ color: C.cyan }}>
              🔄 Iterative Swarm — Convergence
            </div>
            <div className="text-[11px]" style={{ color: C.gray }}>
              {pm.iteration.converged ? (
                <span style={{ color: C.green, fontWeight: "bold" }}>
                  ✓ Converged at Round {pm.iteration.converged_at_round} (Δ &lt; {(pm.iteration.convergence_threshold * 100).toFixed(0)}%)
                </span>
              ) : (
                <span style={{ color: C.amber, fontWeight: "bold" }}>
                  ⚠ Max rounds reached ({pm.iteration.max_rounds}) without convergence
                </span>
              )}
            </div>
          </div>
          {/* Per-round history table */}
          <table className="w-full text-[12px]" style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ backgroundColor: C.bgAlt }}>
                <th className="text-left px-2 py-1" style={{ color: C.gray }}>Round</th>
                <th className="text-right px-2 py-1" style={{ color: C.gray }}>Tickers</th>
                <th className="text-right px-2 py-1" style={{ color: C.gray }}>Δ</th>
                <th className="text-right px-2 py-1" style={{ color: C.gray }}>Obj</th>
                <th className="text-right px-2 py-1" style={{ color: C.gray }}
                    title="Pinned picks (no objections, survival-bias protected with age-aware re-audit)">
                  📌Pin
                </th>
                <th className="text-right px-2 py-1" style={{ color: C.gray }}
                    title="Picks added this round (PM's response to objections)">
                  ➕
                </th>
                <th className="text-right px-2 py-1" style={{ color: C.gray }}
                    title="Picks removed this round (rejected, retained in rejected_pool for future reconsideration)">
                  ➖
                </th>
                <th className="text-left px-2 py-1" style={{ color: C.gray }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {pm.iteration.history.map((r) => {
                const isConverged = pm.iteration!.converged && r.round === pm.iteration!.converged_at_round;
                const deltaC = r.delta < pm.iteration!.convergence_threshold ? C.green
                             : r.delta < 0.30 ? C.amber : C.red;
                return (
                  <tr key={r.round} style={{ borderTop: `1px solid ${C.border}40` }}>
                    <td className="px-2 py-1 font-bold" style={{ color: isConverged ? C.green : C.text }}>
                      R{r.round}
                    </td>
                    <td className="text-right px-2 py-1 font-mono" style={{ color: C.text }}>{r.n_tickers}</td>
                    <td className="text-right px-2 py-1 font-mono" style={{ color: deltaC, fontWeight: "bold" }}>
                      {r.round === 1 ? "—" : `${(r.delta * 100).toFixed(1)}%`}
                    </td>
                    <td className="text-right px-2 py-1 font-mono"
                        style={{ color: r.n_objections > 50 ? C.red : r.n_objections > 20 ? C.amber : C.gray }}>
                      {r.n_objections}
                    </td>
                    <td className="text-right px-2 py-1 font-mono" style={{ color: C.cyan }}>
                      {r.n_pinned ?? "—"}
                      {r.n_newly_pinned ? <span style={{ color: C.green, fontSize: "10px" }}> (+{r.n_newly_pinned})</span> : null}
                    </td>
                    <td className="text-right px-2 py-1 font-mono"
                        style={{ color: C.green, fontSize: "11px" }}
                        title={(r.added_tickers || []).slice(0, 10).join(", ")}>
                      {r.added_tickers?.length ?? "—"}
                    </td>
                    <td className="text-right px-2 py-1 font-mono"
                        style={{ color: C.red, fontSize: "11px" }}
                        title={(r.removed_tickers || []).slice(0, 10).join(", ")}>
                      {r.removed_tickers?.length ?? "—"}
                    </td>
                    <td className="px-2 py-1 text-[11px]" style={{ color: C.gray }}>
                      {isConverged ? "✓ Converged" : r.round === pm.iteration!.history.length ? "▶ Final" : "→ continued"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="mt-1.5 text-[11px]" style={{ color: C.gray, fontStyle: "italic", lineHeight: 1.5 }}>
            ⓘ <b>4-Fix Iterative Swarm:</b> Δ&lt;{(pm.iteration.convergence_threshold * 100).toFixed(0)}% + quality gate (Fix 1) ·
            Sequential memory + rejected_pool (Fix 3) ·
            📌Pin no-objection picks with age≤3 + 20% random re-audit (Fix 4 survival-bias prevention) ·
            Wildcard injection from outside pool + Pareto trade-off framing (Fix 5 overfitting prevention).
          </div>
        </div>
      )}

      {/* ───── Option C: Per-Ticker Debate Summary ───── */}
      {pm.per_ticker_debate_summary && (
        <div className="mb-3 px-3 py-2 rounded"
             style={{ backgroundColor: C.purple + "12",
                      border: `1.5px solid ${C.purple}60`,
                      borderLeft: `5px solid ${C.purple}` }}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[13px] uppercase font-bold tracking-wide" style={{ color: C.purple }}>
              🎯 Per-Ticker Debate (Option C — 3 rounds per ticker)
            </div>
            <div className="text-[11px]" style={{ color: C.gray }}>
              {pm.per_ticker_debate_summary.n_failed > 0 ? (
                <span style={{ color: C.amber, fontWeight: "bold" }}>
                  ⚠ {pm.per_ticker_debate_summary.n_failed}/{pm.per_ticker_debate_summary.n_total} failed
                </span>
              ) : (
                <span style={{ color: C.green, fontWeight: "bold" }}>
                  ✓ All {pm.per_ticker_debate_summary.n_total} tickers debated
                </span>
              )}
            </div>
          </div>
          <div className="grid grid-cols-3 gap-2 text-[12px]">
            <div className="rounded p-2" style={{ backgroundColor: C.bg, border: `1px solid ${C.border}` }}>
              <div className="text-[11px] uppercase mb-1" style={{ color: C.gray }}>Tier Distribution</div>
              {Object.entries(pm.per_ticker_debate_summary.tier_dist || {}).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span style={{ color: k === "UNANIMOUS" ? C.green : k === "MAJORITY_CLEAN" ? C.cyan
                                      : k === "SOLO" ? C.amber : k === "EXCLUDED" ? C.red : C.gray }}>{k}</span>
                  <span className="font-mono" style={{ color: C.text }}>{v}</span>
                </div>
              ))}
            </div>
            <div className="rounded p-2" style={{ backgroundColor: C.bg, border: `1px solid ${C.border}` }}>
              <div className="text-[11px] uppercase mb-1" style={{ color: C.gray }}>Trading Signal</div>
              {Object.entries(pm.per_ticker_debate_summary.trading_dist || {}).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span style={{ color: k === "BUY_NOW" ? C.green : k === "WAIT" ? C.amber : k === "SKIP" ? C.red : C.gray }}>{k}</span>
                  <span className="font-mono" style={{ color: C.text }}>{v}</span>
                </div>
              ))}
            </div>
            <div className="rounded p-2" style={{ backgroundColor: C.bg, border: `1px solid ${C.border}` }}>
              <div className="text-[11px] uppercase mb-1" style={{ color: C.gray }}>Risk Vote</div>
              {Object.entries(pm.per_ticker_debate_summary.risk_dist || {}).map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span style={{ color: k === "APPROVE" ? C.green : k === "CAUTION" ? C.amber : k === "REJECT" ? C.red : C.gray }}>{k}</span>
                  <span className="font-mono" style={{ color: C.text }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="mt-1.5 text-[11px]" style={{ color: C.gray, fontStyle: "italic", lineHeight: 1.5 }}>
            ⓘ <b>Per-Ticker Debate (Option C):</b> Each ticker gets focused 3-round mini-debate
            (R1 Trading+Risk+Critic verdict · R2 Revise if critic flagged · R3 Final Arbiter) ·
            Small 5-ticker batches eliminate batch-240 cascade failure mode that produced universal SOLO/WATCH.
          </div>
        </div>
      )}

      {/* ───── Option C: Portfolio Composition ───── */}
      {pm.portfolio_composition_summary && pm.portfolio_composition && (
        <div className="mb-3 px-3 py-2 rounded"
             style={{ backgroundColor: C.amber + "10",
                      border: `1.5px solid ${C.amber}60`,
                      borderLeft: `5px solid ${C.amber}` }}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[13px] uppercase font-bold tracking-wide" style={{ color: C.amber }}>
              📊 Portfolio Composition (Phase 5b)
            </div>
            <div className="text-[11px]" style={{ color: C.gray }}>
              Regime: <b style={{ color: C.text }}>{pm.portfolio_composition_summary.regime_tag || "—"}</b>
              {" · "}Adaptive budget: <b style={{ color: C.text }}>{pm.portfolio_composition_summary.adaptive_budget ?? "—"}</b>/horizon
            </div>
          </div>
          <div className="grid grid-cols-4 gap-2 text-[12px]">
            <div className="rounded p-2 text-center" style={{ backgroundColor: C.green + "10", border: `1px solid ${C.green}40` }}>
              <div className="text-[11px] uppercase" style={{ color: C.gray }}>Active</div>
              <div className="text-[16px] font-bold font-mono" style={{ color: C.green }}>
                {pm.portfolio_composition_summary.active_picks ?? 0}
              </div>
              <div className="text-[10px]" style={{ color: C.gray }}>incl. half-size</div>
            </div>
            <div className="rounded p-2 text-center" style={{ backgroundColor: C.red + "10", border: `1px solid ${C.red}40` }}>
              <div className="text-[11px] uppercase" style={{ color: C.gray }}>Excluded</div>
              <div className="text-[16px] font-bold font-mono" style={{ color: C.red }}>
                {pm.portfolio_composition_summary.excluded_total ?? 0}
              </div>
              <div className="text-[10px]" style={{ color: C.gray }}>cap+budget+debate</div>
            </div>
            <div className="rounded p-2" style={{ backgroundColor: C.bg, border: `1px solid ${C.border}` }}>
              <div className="text-[11px] uppercase mb-1" style={{ color: C.gray }}>Sector Top 3</div>
              {(pm.portfolio_composition_summary.sector_top3 || []).map(([s, n]) => (
                <div key={s} className="flex justify-between text-[11px]">
                  <span style={{ color: C.text }}>{s}</span>
                  <span className="font-mono" style={{ color: C.cyan }}>{n}</span>
                </div>
              ))}
            </div>
            <div className="rounded p-2" style={{ backgroundColor: C.bg, border: `1px solid ${C.border}` }}>
              <div className="text-[11px] uppercase mb-1" style={{ color: C.gray }}>Warnings</div>
              {(pm.portfolio_composition_summary.warnings || []).length > 0 ? (
                (pm.portfolio_composition_summary.warnings || []).slice(0, 3).map((w, i) => (
                  <div key={i} className="text-[10px]" style={{ color: C.amber }}>⚠ {w}</div>
                ))
              ) : (
                <div className="text-[11px]" style={{ color: C.green }}>✓ none</div>
              )}
            </div>
          </div>
          <div className="mt-1.5 text-[11px]" style={{ color: C.gray, fontStyle: "italic", lineHeight: 1.5 }}>
            ⓘ <b>Portfolio Composer:</b> Adaptive regime-aware budget
            (RISK_OFF×0.6 · LATE_CYCLE×0.7 · NEUTRAL×0.8 · RISK_ON×1.0)
            · Max sector weight {Math.round((pm.portfolio_composition.max_sector_weight || 0.3) * 100)}%
            · Conviction-sorted EXCLUDE drop · INCLUDE_REDUCED_SIZE → half size.
          </div>
        </div>
      )}

      {/* PM Commentary (executive summary, ~1000 chars) */}
      {pm.pm_commentary && (
        <div className="mb-3 px-4 py-3 rounded"
             style={{ backgroundColor: C.purple + "18",
                      border: `1.5px solid ${C.purple}80`,
                      borderLeft: `5px solid ${C.purple}` }}>
          <div className="flex items-center justify-between mb-2">
            <div className="text-[13px] uppercase font-bold tracking-wide" style={{ color: C.purple }}>
              💼 PM Agent Commentary — Executive Summary
            </div>
            <div className="text-[11px]" style={{ color: C.gray }}>
              {pm.pm_commentary.length} chars
            </div>
          </div>
          <div className="text-[14px] leading-relaxed whitespace-pre-wrap"
               style={{ color: C.text, lineHeight: 1.75 }}>
            {pm.pm_commentary}
          </div>
        </div>
      )}

      {/* Portfolio Thesis (concise tagline) */}
      {pm.portfolio_thesis && (
        <div className="mb-3 px-3 py-2 rounded"
             style={{ backgroundColor: C.purple + "10", borderLeft: `3px solid ${C.purple}` }}>
          <div className="text-[11px] uppercase font-bold mb-1" style={{ color: C.purple }}>
            Portfolio Thesis (Tagline)
          </div>
          <div className="text-[13px] leading-relaxed whitespace-pre-wrap"
               style={{ color: C.text, lineHeight: 1.6 }}>
            {pm.portfolio_thesis}
          </div>
        </div>
      )}

      {/* ── Horizon Toggle — Tactical 5d / Core 21d / Strategic 63d ── */}
      <div className="flex items-center gap-2 mb-3 pb-2 border-b" style={{ borderColor: C.border }}>
        <span className="text-[12px] uppercase font-bold" style={{ color: C.gray }}>
          Investment Horizon:
        </span>
        {(Object.keys(HORIZON_META) as HorizonKey[]).map((k) => {
          const meta = HORIZON_META[k];
          const isActive = horizon === k;
          const n = horizonCounts[k];
          return (
            <button key={k} onClick={() => setHorizon(k)}
                    className="px-3 py-1.5 rounded text-[14px] transition-colors"
                    style={{
                      backgroundColor: isActive ? meta.color + "25" : "transparent",
                      color: isActive ? meta.color : C.gray,
                      border: `1px solid ${isActive ? meta.color + "80" : C.border}`,
                      fontWeight: isActive ? "bold" : "normal",
                    }}>
              {meta.emoji} {meta.label} <span style={{ opacity: 0.7, marginLeft: 4 }}>({meta.days})</span>
              {n > 0 && <span className="ml-1.5 text-[12px]" style={{ opacity: 0.6 }}>· {n}</span>}
            </button>
          );
        })}
        <span className="text-[12px] ml-2" style={{ color: C.gray }}>{HORIZON_META[horizon].sub}</span>
      </div>

      {/* 4 PM picks tables for selected horizon */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        <PMTable title={`${HORIZON_META[horizon].emoji} ${HORIZON_META[horizon].label} LONG — Stocks (${HORIZON_META[horizon].days})`}
                 color={C.green} picks={picks.long_stocks}  isShort={false} />
        <PMTable title={`${HORIZON_META[horizon].emoji} ${HORIZON_META[horizon].label} LONG — ETFs (${HORIZON_META[horizon].days})`}
                 color={C.green} picks={picks.long_etfs}    isShort={false} />
        <PMTable title={`${HORIZON_META[horizon].emoji} ${HORIZON_META[horizon].label} SHORT — Stocks (${HORIZON_META[horizon].days})`}
                 color={C.red}   picks={picks.short_stocks} isShort={true} />
        <PMTable title={`${HORIZON_META[horizon].emoji} ${HORIZON_META[horizon].label} SHORT — ETFs (${HORIZON_META[horizon].days})`}
                 color={C.red}   picks={picks.short_etfs}   isShort={true} />
      </div>

      {/* Phase 4 drops */}
      {pm.phase4_drops && pm.phase4_drops.length > 0 && (
        <div className="mb-4 rounded p-2.5"
             style={{ backgroundColor: C.red + "08", border: `1px solid ${C.red}30` }}>
          <div className="text-[12px] uppercase font-bold mb-1.5" style={{ color: C.red }}>
            ✗ Removed from Phase 4 Draft ({pm.phase4_drops.length})
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {Object.entries(dropsByBucket).map(([bucket, drops]) => (
              <div key={bucket}>
                <div className="text-[11px] uppercase font-bold mb-0.5" style={{ color: C.gray }}>{bucket}</div>
                <ul className="space-y-0.5">
                  {drops.map((d, i) => (
                    <li key={i} className="text-[12px]" style={{ color: C.text }}>
                      <span className="font-mono font-bold line-through" style={{ color: C.red }}>{d.ticker}</span>
                      <span className="ml-1.5" style={{ color: C.gray }}>— {d.reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Hedge pairs */}
      {pm.hedge_pairs && pm.hedge_pairs.length > 0 && (
        <div className="mb-3 rounded p-2.5"
             style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}>
          <div className="text-[12px] uppercase font-bold mb-1.5" style={{ color: C.cyan }}>
            🔗 Hedge Pairs (long-short pair structure)
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
            {pm.hedge_pairs.map((h, i) => (
              <div key={i} className="text-[12px] flex items-start gap-2">
                <span className="font-mono font-bold whitespace-nowrap">
                  <span style={{ color: C.green }}>{h.long}</span>
                  <span style={{ color: C.gray }}> ⇄ </span>
                  <span style={{ color: C.red }}>{h.short}</span>
                  {h.sector && <span className="ml-1.5 text-[11px]" style={{ color: C.cyan }}>{h.sector}</span>}
                </span>
                <span style={{ color: C.text, lineHeight: 1.4 }}>{h.rationale}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Risk budget */}
      {pm.risk_budget && pm.risk_budget.length > 0 && (
        <div>
          <div className="text-[12px] uppercase font-bold mb-1.5" style={{ color: C.amber }}>
            💰 Risk Budget Allocation
          </div>
          <div className="space-y-1">
            {pm.risk_budget.map((r, i) => (
              <div key={i} className="flex items-center gap-2 text-[12px]">
                <div className="w-32 font-semibold" style={{ color: C.text }}>{r.sector}</div>
                <div className="flex-1 rounded overflow-hidden"
                     style={{ backgroundColor: C.bgAlt, height: 12 }}>
                  <div style={{ width: `${Math.min(100, r.allocation_pct)}%`,
                                backgroundColor: C.amber + "80", height: "100%" }} />
                </div>
                <div className="w-10 text-right font-mono font-bold" style={{ color: C.amber }}>
                  {r.allocation_pct}%
                </div>
                <div className="flex-1 text-[12px]" style={{ color: C.gray, lineHeight: 1.35 }}>
                  {r.rationale}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
