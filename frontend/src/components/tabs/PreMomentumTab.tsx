import { useEffect, useState, useMemo } from "react";
import Plot from "react-plotly.js";
import { type ColumnDef } from "@tanstack/react-table";
import { fetchPreMomentum, fetchPreMomentumML } from "../../api/client";
import { DataTable } from "../shared/DataTable";
import { MetricCard } from "../shared/MetricCard";
import { ColDefToggle } from "../shared/ColDefToggle";
import { C, CLASS_COLORS, DARK_LAYOUT } from "../../styles/theme";
import { useSort } from "../../hooks/useSort";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface AgentResult {
  score: number;
  signals: Record<string, number>;
  summary: string;
}

interface Candidate {
  ticker: string;
  name: string;
  category: string;
  theme: string;
  sector?: string;
  current_classification: string;
  current_composite: number;
  pre_momentum_score: number;
  agreement_ratio: number;
  agents: {
    microstructure: AgentResult;
    macro_regime: AgentResult;
    graph_relational: AgentResult;
    catalyst: AgentResult;
    qvr?: AgentResult;
  };
  expected_timeline: string;
  key_catalysts: string[];
  risk_factors: string[];
  pm_age?: number;
  // Multi-horizon returns
  ret_1d?: number;
  ret_5d?: number;
  ret_21d?: number;
  ret_63d?: number;
  ret_126d?: number;
  ret_ytd?: number | null;
  ret_252d?: number;
  ret_3y_ann?: number | null;
  ret_5y_ann?: number | null;
  vol_3y_ann?: number | null;
}

interface ConversionEntry {
  ticker: string;
  name: string;
  category: string;
  score_1m_ago: number;
  current_composite: number;
  score_change: number;
  current_class: string;
  current_eligible?: boolean;
  exit_class?: string;
}

interface ConversionData {
  graduated: ConversionEntry[];
  failed: ConversionEntry[];
  in_progress: ConversionEntry[];
  stats: {
    total_pm_candidates_1m: number;
    total_graduated: number;
    total_failed: number;
    total_in_progress: number;
    hit_rate: number;
    avg_score_improvement: number;
  };
}

interface PreMomentumData {
  candidates: Candidate[];
  candidates_etf?: Candidate[];
  candidates_stock?: Candidate[];
  summary: {
    total_universe: number;
    candidates_analyzed: number;
    agreement_strong: number;
    agreement_moderate: number;
    agreement_weak: number;
    agreement_none: number;
    top_sectors: { sector: string; count: number; avg_score: number }[];
    agent_agreement_distribution: Record<string, number>;
  };
  methodology: {
    description: string;
    agents: { name: string; weight: number; type: string; description: string }[];
    agreement_thresholds: Record<string, string>;
  };
  conversion?: ConversionData;
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

// Agreement-tier color (replaces convictionColor; keyed by agreement_ratio).
function agreementColor(ratio: number): string {
  if (ratio >= 0.6) return C.green;     // strong (≥3 of 5 agents)
  if (ratio >= 0.4) return C.cyan;      // moderate (2 agents)
  if (ratio > 0)    return C.yellow;    // weak (1 agent)
  return C.gray;                        // none
}

function scoreColor(score: number): string {
  if (score > 70) return C.green;
  if (score >= 50) return C.cyan;
  return C.gray;
}

function agentScoreColor(score: number): string {
  if (score > 60) return C.green;
  if (score >= 40) return C.yellow;
  return C.gray;
}

function retColor(v: number): string {
  if (v > 3) return C.green;
  if (v > 0) return "#0A7D3F";
  if (v > -3) return "#CC0000";
  return C.red;
}

// ---------------------------------------------------------------------------
// Pre-Momentum Decision Logic — agreement_ratio × PM score × age × catalysts
// (Conviction labels removed; agreement_ratio carries the same breadth signal numerically)
// ---------------------------------------------------------------------------

interface PMDecision {
  action: string;
  rationale: string;
  color: string;
  rank: number;  // lower = more actionable
}

// Agreement tier helper — replaces categorical conviction.
//   strong   : ratio ≥ 0.6 (≥3 of 5 agents fired)
//   moderate : 0.4 ≤ ratio < 0.6 (2 agents)
//   weak     : 0 < ratio < 0.4 (1 agent)
//   none     : ratio == 0
type AgreementTier = "strong" | "moderate" | "weak" | "none";
function agreementTier(ratio: number): AgreementTier {
  if (ratio >= 0.6) return "strong";
  if (ratio >= 0.4) return "moderate";
  if (ratio > 0) return "weak";
  return "none";
}

function decidePMAction(c: Candidate): PMDecision {
  const pmScore = c.pre_momentum_score ?? 0;
  const age = c.pm_age ?? 0;
  const composite = c.current_composite ?? 0;
  const cls = c.current_classification || "";
  const ratio = c.agreement_ratio ?? 0;
  const agreePct = ratio * 100;
  const tier = agreementTier(ratio);
  const catalystCount = (c.key_catalysts || []).length;

  // ── Risk filter — 약화 분류는 무조건 IGNORE ──
  if (cls.includes("FADING") || cls.includes("WEAKENING")) {
    if (pmScore < 50) {
      return {
        action: "IGNORE",
        rationale: `${cls.split(" ")[1]} + 낮은 PM (${pmScore.toFixed(0)}) → 추세 약화, 회피`,
        color: C.gray,
        rank: 9,
      };
    }
    return {
      action: "WATCH (caution)",
      rationale: `${cls.split(" ")[1]} 분류이나 PM ${pmScore.toFixed(0)} → 반등 가능성 주시`,
      color: C.yellow,
      rank: 7,
    };
  }

  // ── 장기 정체 (Stagnant) — 80일+ 지속이나 strong agreement 못 만듬 ──
  if (age >= 80 && tier !== "strong") {
    return {
      action: "STAGNANT",
      rationale: `장기 PM 상태(${age}d)이나 agreement ${agreePct.toFixed(0)}% → 돌파 동력 부족, 우선순위 낮춤`,
      color: "#8A6D3B",
      rank: 7,
    };
  }

  // ── strong agreement (3+ agents firing) ──
  if (tier === "strong") {
    if (pmScore >= 75 && age >= 14) {
      return {
        action: "STRONG PREPARE",
        rationale: `다중 agent 합의 (${agreePct.toFixed(0)}%) + 14일+ 지속 (${age}d) → 진입 가격/사이즈 사전 설정, ${c.expected_timeline} 내 돌파 임박`,
        color: C.green,
        rank: 1,
      };
    }
    if (pmScore >= 65) {
      return {
        action: "PREPARE",
        rationale: `Strong agreement (PM ${pmScore.toFixed(0)}, ${agreePct.toFixed(0)}% agents) → 워치리스트 등록, 진입 트리거 설정`,
        color: C.green,
        rank: 2,
      };
    }
    return {
      action: "PREPARE",
      rationale: `Strong agreement이나 PM ${pmScore.toFixed(0)} → 점진 매집 후보, 신호 강화 대기`,
      color: "#0A7D3F",
      rank: 3,
    };
  }

  // ── moderate agreement (2 agents) ──
  if (tier === "moderate") {
    if (pmScore >= 60 && catalystCount >= 3) {
      return {
        action: "WATCH (Active)",
        rationale: `Moderate agreement + PM ${pmScore.toFixed(0)} + ${catalystCount} catalysts → 매주 추적, agreement 상승 시 PREPARE 격상`,
        color: C.cyan,
        rank: 4,
      };
    }
    if (pmScore >= 55) {
      return {
        action: "WATCH",
        rationale: `Moderate agreement (${agreePct.toFixed(0)}%, PM ${pmScore.toFixed(0)}) → 정기 관찰, 추가 agent 합류 대기`,
        color: C.cyan,
        rank: 5,
      };
    }
    return {
      action: "TRACK",
      rationale: `Moderate agreement이나 PM ${pmScore.toFixed(0)} 미달 → 후순위 관찰`,
      color: "#0A7D3F",
      rank: 6,
    };
  }

  // ── weak agreement (1 agent) ──
  if (tier === "weak") {
    if (pmScore >= 50 && composite >= 45) {
      return {
        action: "TRACK",
        rationale: `Weak agreement (${agreePct.toFixed(0)}%, PM ${pmScore.toFixed(0)}, composite ${composite.toFixed(0)}) → 약한 신호, 주간 monitoring`,
        color: C.yellow,
        rank: 6,
      };
    }
    return {
      action: "IGNORE",
      rationale: `Weak agreement + 약한 신호 → 대상 제외`,
      color: C.gray,
      rank: 8,
    };
  }

  // ── no agreement ──
  return {
    action: "IGNORE",
    rationale: `Agreement 0% (어떤 agent도 신호 미발생) → 감시 대상 아님`,
    color: C.gray,
    rank: 9,
  };
}

// ---------------------------------------------------------------------------
// Expandable Section
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
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-3 text-left text-[16px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="flex items-center gap-2">
          {title}
          {badge && (
            <span className="text-[12px] px-2.5 py-0.5 rounded bg-[#E3EEF5]/50 text-[#0F5499]">
              {badge}
            </span>
          )}
        </span>
        <span className="text-[#857F7A] text-[14px]">{open ? "▼" : "▶"}</span>
      </button>
      {open && <div className="p-4 bg-[#FBEEE3] space-y-4">{children}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Agent Type Badge
// ---------------------------------------------------------------------------

function AgentTypeBadge({ type }: { type: string }) {
  const isQuant = type.toLowerCase() === "quant";
  return (
    <span
      className="text-[12px] px-2.5 py-0.5 rounded font-semibold"
      style={{
        backgroundColor: (isQuant ? C.blue : C.purple) + "22",
        color: isQuant ? C.blue : C.purple,
      }}
    >
      {type}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Section 1: Methodology Banner
// ---------------------------------------------------------------------------

function MethodologyBanner({
  methodology,
}: {
  methodology: PreMomentumData["methodology"];
}) {
  return (
    <Section title="Methodology: Pre-Momentum Detection" defaultOpen>
      <p className="text-[16px] text-[#66605C] leading-relaxed max-w-4xl">
        {methodology.description}
      </p>

      {/* Agent Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-4">
        {methodology.agents.map((agent) => (
          <div
            key={agent.name}
            className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[16px] font-semibold text-[#33302E]">
                {agent.name}
              </span>
              <AgentTypeBadge type={agent.type} />
            </div>
            <div className="text-[12px] text-[#857F7A] mb-2">
              Weight:{" "}
              <span className="text-[#0F5499] font-mono">
                {(agent.weight * 100).toFixed(0)}%
              </span>
            </div>
            <p className="text-[13px] text-[#857F7A] leading-relaxed">
              {agent.description}
            </p>
          </div>
        ))}
      </div>

      {/* Agreement Tier Thresholds */}
      <div className="mt-4">
        <h4 className="text-[14px] font-semibold text-[#857F7A] uppercase tracking-wide mb-2">
          Agreement Tier Thresholds
        </h4>
        <div className="flex flex-wrap gap-3">
          {Object.entries(methodology.agreement_thresholds || {}).map(
            ([level, desc]) => {
              const ratioByTier: Record<string, number> = {
                strong: 0.6, moderate: 0.4, weak: 0.2, none: 0,
              };
              const color = agreementColor(ratioByTier[level] ?? 0);
              return (
                <div
                  key={level}
                  className="flex items-center gap-2 px-3 py-1.5 bg-[#FFFFFF] border border-[#E6D9CE] rounded text-[14px]"
                >
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                  <span className="font-semibold uppercase" style={{ color }}>{level}</span>
                  <span className="text-[#857F7A]">{desc}</span>
                </div>
              );
            }
          )}
        </div>
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// Section 3: Agreement Distribution Chart
// ---------------------------------------------------------------------------

function AgreementDistributionChart({
  distribution,
}: {
  distribution: Record<string, number>;
}) {
  const entries = Object.entries(distribution)
    .map(([k, v]) => ({ agents: k, count: v }))
    .sort((a, b) => Number(b.agents) - Number(a.agents));

  const barColors = entries.map((e) => {
    const n = Number(e.agents);
    if (n >= 4) return C.green;
    if (n === 3) return C.cyan;
    if (n === 2) return C.yellow;
    return C.gray;
  });

  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-3">
        Agent Agreement Distribution
      </h3>
      <Plot
        data={[
          {
            type: "bar",
            y: entries.map((e) => `${e.agents} agents`),
            x: entries.map((e) => e.count),
            orientation: "h",
            marker: { color: barColors, opacity: 0.85 },
            text: entries.map((e) => String(e.count)),
            textposition: "outside" as const,
            textfont: { color: C.text, size: 11 },
            hovertemplate:
              "%{y}: %{x} tickers<extra></extra>",
          },
        ]}
        layout={{
          ...DARK_LAYOUT,
          height: 220,
          margin: { t: 10, b: 30, l: 90, r: 50 },
          xaxis: {
            title: "Number of Tickers",
            gridcolor: "#F2E5D7",
            color: "#66605C",
          },
          yaxis: { automargin: true, color: "#66605C" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
      <p className="text-[12px] text-[#857F7A] mt-2">
        How many tickers have 0, 1, 2, 3, 4, or 5 agents signaling pre-momentum.
        Higher agreement_ratio = broader cross-agent confirmation.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section 4: Top Sectors Chart
// ---------------------------------------------------------------------------

function TopSectorsChart({
  sectors,
}: {
  sectors: { sector: string; count: number; avg_score: number }[];
}) {
  if (!sectors.length) return null;

  const sorted = [...sectors].sort((a, b) => a.count - b.count);
  const maxScore = Math.max(...sorted.map((s) => s.avg_score), 1);
  const minScore = Math.min(...sorted.map((s) => s.avg_score), 0);

  const barColors = sorted.map((s) => {
    const t = maxScore === minScore ? 0.5 : (s.avg_score - minScore) / (maxScore - minScore);
    if (t > 0.66) return C.green;
    if (t > 0.33) return C.cyan;
    return C.yellow;
  });

  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-3">
        Top Sectors by Candidate Count
      </h3>
      <Plot
        data={[
          {
            type: "bar",
            y: sorted.map((s) => s.sector),
            x: sorted.map((s) => s.count),
            orientation: "h",
            marker: { color: barColors, opacity: 0.85 },
            text: sorted.map(
              (s) => `${s.count} (avg: ${s.avg_score.toFixed(1)})`
            ),
            textposition: "outside" as const,
            textfont: { color: C.text, size: 10 },
            hovertemplate:
              "%{y}<br>Count: %{x}<br>Avg Score: %{text}<extra></extra>",
          },
        ]}
        layout={{
          ...DARK_LAYOUT,
          height: Math.max(200, sorted.length * 35 + 60),
          margin: { t: 10, b: 30, l: 140, r: 80 },
          xaxis: {
            title: "Candidate Count",
            gridcolor: "#F2E5D7",
            color: "#66605C",
          },
          yaxis: { automargin: true, color: "#66605C" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
      <p className="text-[12px] text-[#857F7A] mt-2">
        Sectors with the most strong/moderate-agreement pre-momentum candidates.
        Color intensity reflects average pre-momentum score.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section 6: Radar Chart for top candidates
// ---------------------------------------------------------------------------

function AgentRadarChart({ candidates }: { candidates: Candidate[] }) {
  const top10 = candidates
    .slice()
    .sort((a, b) => b.pre_momentum_score - a.pre_momentum_score)
    .slice(0, 10);

  if (!top10.length) return null;

  const categories = ["Microstructure", "Macro Regime", "Graph Relational", "Catalyst", "QVR"];
  const agentKeys: (keyof Candidate["agents"])[] = [
    "microstructure",
    "macro_regime",
    "graph_relational",
    "catalyst",
    "qvr",
  ];

  const traceColors = [
    C.cyan,
    C.green,
    C.yellow,
    C.purple,
    C.orange,
    C.blue,
    C.red,
    "#0D7680",
    "#7D5BA6",
    "#C2701C",
  ];

  const traces = top10.map((c, idx) => ({
    type: "scatterpolar" as const,
    r: [...agentKeys.map((k) => c.agents[k]?.score ?? 0), c.agents[agentKeys[0]]?.score ?? 0],
    theta: [...categories, categories[0]],
    name: c.ticker,
    line: { color: traceColors[idx % traceColors.length], width: 2 },
    fill: "toself" as const,
    fillcolor: traceColors[idx % traceColors.length] + "10",
    opacity: 0.8,
    hovertemplate: `${c.ticker}<br>%{theta}: %{r:.1f}<extra></extra>`,
  }));

  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-3">
        Agent Score Radar — Top 10 Candidates
      </h3>
      <Plot
        data={traces}
        layout={{
          ...DARK_LAYOUT,
          height: 420,
          margin: { t: 30, b: 30, l: 60, r: 60 },
          polar: {
            bgcolor: C.panel,
            radialaxis: {
              visible: true,
              range: [0, 100],
              gridcolor: "#F2E5D7",
              color: "#66605C",
              tickfont: { size: 9 },
            },
            angularaxis: {
              gridcolor: "#F2E5D7",
              color: "#66605C",
              tickfont: { size: 10 },
            },
          },
          legend: {
            font: { size: 10, color: C.text },
            bgcolor: "rgba(0,0,0,0)",
            x: 1.05,
            y: 1,
          },
          showlegend: true,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section 6b: Agent Signal Heatmap
// ---------------------------------------------------------------------------

function AgentSignalHeatmap({ candidates }: { candidates: Candidate[] }) {
  const top = candidates
    .slice()
    .sort((a, b) => b.pre_momentum_score - a.pre_momentum_score)
    .slice(0, 15);

  if (!top.length) return null;

  const agentKeys: { key: keyof Candidate["agents"]; label: string }[] = [
    { key: "microstructure", label: "Micro" },
    { key: "macro_regime", label: "Macro" },
    { key: "graph_relational", label: "Graph" },
    { key: "catalyst", label: "Catalyst" },
    { key: "qvr", label: "QVR" },
  ];

  // Collect all unique sub-signal keys per agent
  const agentSignalKeys: Record<string, string[]> = {};
  for (const { key } of agentKeys) {
    const allKeys = new Set<string>();
    for (const c of top) {
      const signals = c.agents[key]?.signals;
      if (signals) Object.keys(signals).forEach((k) => allKeys.add(k));
    }
    agentSignalKeys[key] = [...allKeys].sort();
  }

  // Build combined signal columns: "agent:signal"
  const signalCols: { agent: string; agentLabel: string; signal: string }[] = [];
  for (const { key, label } of agentKeys) {
    for (const sig of agentSignalKeys[key] || []) {
      signalCols.push({ agent: key, agentLabel: label, signal: sig });
    }
  }

  if (!signalCols.length) return null;

  const tickers = top.map((c) => c.ticker);
  const z: number[][] = [];
  const hoverText: string[][] = [];

  for (const c of top) {
    const row: number[] = [];
    const hoverRow: string[] = [];
    for (const col of signalCols) {
      const val =
        c.agents[col.agent as keyof Candidate["agents"]]?.signals?.[
          col.signal
        ] ?? 0;
      row.push(val);
      hoverRow.push(
        `${c.ticker}<br>${col.agentLabel}: ${col.signal}<br>Value: ${val.toFixed(2)}`
      );
    }
    z.push(row);
    hoverText.push(hoverRow);
  }

  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-3">
        Agent Sub-Signal Heatmap — Top 15 Candidates
      </h3>
      <Plot
        data={[
          {
            type: "heatmap",
            z: z,
            x: signalCols.map((c) => `${c.agentLabel}:${c.signal}`),
            y: tickers,
            colorscale: [
              [0, "#FFFFFF"],
              [0.25, "#1e3a5f"],
              [0.5, C.cyan],
              [0.75, "#0D7680"],
              [1, C.green],
            ],
            hovertext: hoverText,
            hoverinfo: "text",
            showscale: true,
            colorbar: {
              tickfont: { color: C.text, size: 9 },
              len: 0.8,
            },
          },
        ]}
        layout={{
          ...DARK_LAYOUT,
          height: Math.max(300, top.length * 28 + 100),
          margin: { t: 10, b: 80, l: 70, r: 20 },
          xaxis: {
            tickangle: -45,
            tickfont: { size: 9, color: "#66605C" },
            gridcolor: "#F2E5D7",
          },
          yaxis: {
            automargin: true,
            tickfont: { size: 10, color: "#66605C" },
          },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
      <p className="text-[12px] text-[#857F7A] mt-2">
        Each cell shows the sub-signal score from the respective agent. Brighter
        = stronger signal. Columns grouped by agent.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section 7: Candidate Detail Panel
// ---------------------------------------------------------------------------

function CandidateDetail({ candidate }: { candidate: Candidate }) {
  const agentEntries: {
    key: keyof Candidate["agents"];
    label: string;
    color: string;
  }[] = [
    { key: "microstructure", label: "Microstructure", color: C.cyan },
    { key: "macro_regime", label: "Macro Regime", color: C.blue },
    { key: "graph_relational", label: "Graph Relational", color: C.purple },
    { key: "catalyst", label: "Catalyst", color: C.orange },
    { key: "qvr", label: "QVR", color: C.green },
  ];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-4 flex-wrap">
        <span className="text-[20px] font-bold text-[#0F5499] font-mono">
          {candidate.ticker}
        </span>
        <span className="text-[16px] text-[#33302E]">{candidate.name}</span>
        {(() => {
          const ratio = candidate.agreement_ratio || 0;
          const tier = ratio >= 0.6 ? "STRONG"
                      : ratio >= 0.4 ? "MODERATE"
                      : ratio > 0    ? "WEAK"
                      : "NONE";
          const color = agreementColor(ratio);
          return (
            <span
              className="text-[12px] px-3 py-0.5 rounded font-semibold"
              style={{ backgroundColor: color + "22", color }}
            >
              {tier} AGREEMENT ({(ratio * 100).toFixed(0)}%)
            </span>
          );
        })()}
        <span className="text-[14px] text-[#857F7A]">
          Pre-Mom Score:{" "}
          <span
            className="font-mono font-bold"
            style={{ color: scoreColor(candidate.pre_momentum_score) }}
          >
            {candidate.pre_momentum_score.toFixed(1)}
          </span>
        </span>
        <span className="text-[14px] text-[#857F7A]">
          Agreement:{" "}
          <span className="font-mono text-[#33302E]">
            {(candidate.agreement_ratio * 100).toFixed(0)}%
          </span>
        </span>
      </div>

      {/* Meta row */}
      <div className="flex flex-wrap gap-4 text-[14px] text-[#857F7A]">
        <span>
          Sector: <span className="text-[#33302E] font-semibold">{candidate.sector || candidate.category}</span>
        </span>
        <span>
          SubTheme: <span className="text-[#33302E]">{candidate.theme}</span>
        </span>
        <span>
          Classification:{" "}
          <span className="text-[#33302E]">
            {candidate.current_classification}
          </span>
        </span>
        <span>
          Composite:{" "}
          <span className="text-[#33302E] font-mono">
            {candidate.current_composite.toFixed(1)}
          </span>
        </span>
        <span>
          Timeline:{" "}
          <span className="text-[#33302E]">{candidate.expected_timeline}</span>
        </span>
      </div>

      {/* Agent Breakdown Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {agentEntries.map(({ key, label, color }) => {
          const agent = candidate.agents[key];
          if (!agent) return null;
          return (
            <div
              key={key}
              className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-[14px] font-semibold" style={{ color }}>
                  {label}
                </span>
                <span
                  className="text-[16px] font-bold font-mono"
                  style={{ color: agentScoreColor(agent.score) }}
                >
                  {agent.score.toFixed(1)}
                </span>
              </div>
              <p className="text-[12px] text-[#857F7A] mb-2 leading-relaxed line-clamp-3">
                {agent.summary}
              </p>
              {/* Sub-signals */}
              <div className="space-y-1">
                {Object.entries(agent.signals).map(([sig, val]) => {
                  const pct = Math.min(Math.max(val, 0), 100);
                  return (
                    <div key={sig} className="flex items-center gap-2">
                      <span className="text-[11px] text-[#857F7A] w-24 truncate">
                        {sig}
                      </span>
                      <div className="flex-1 h-1.5 rounded-full bg-[#F2E5D7] overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${pct}%`,
                            backgroundColor: agentScoreColor(val),
                          }}
                        />
                      </div>
                      <span className="text-[11px] text-[#857F7A] font-mono w-8 text-right">
                        {val.toFixed(0)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Key Catalysts & Risk Factors */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-3">
          <h4 className="text-[14px] font-semibold text-[#0A7D3F] uppercase tracking-wide mb-2">
            Key Catalysts
          </h4>
          {candidate.key_catalysts.length > 0 ? (
            <ul className="space-y-1">
              {candidate.key_catalysts.map((cat, i) => (
                <li
                  key={i}
                  className="text-[13px] text-[#66605C] flex items-start gap-2"
                >
                  <span className="text-[#0A7D3F] mt-0.5 shrink-0">+</span>
                  {cat}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[13px] text-[#857F7A]">No catalysts identified.</p>
          )}
        </div>
        <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-3">
          <h4 className="text-[14px] font-semibold text-[#CC0000] uppercase tracking-wide mb-2">
            Risk Factors
          </h4>
          {candidate.risk_factors.length > 0 ? (
            <ul className="space-y-1">
              {candidate.risk_factors.map((risk, i) => (
                <li
                  key={i}
                  className="text-[13px] text-[#66605C] flex items-start gap-2"
                >
                  <span className="text-[#CC0000] mt-0.5 shrink-0">!</span>
                  {risk}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[13px] text-[#857F7A]">No risk factors identified.</p>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Column definitions for the candidates table
// ---------------------------------------------------------------------------

const TABLE_COL_DEFS = [
  { col: "Ticker", desc: "Ticker symbol" },
  { col: "Name", desc: "Security name" },
  { col: "Sector", desc: "통합 17개 섹터 (GICS 11 + Fixed Income / International / Equity Broad / Macro / Multi-Asset / Alternatives). ETF/주식 동일 체계." },
  { col: "Class", desc: "Current dual-timeframe classification" },
  { col: "Composite", desc: "Current Price Discovery composite score (0-100)" },
  {
    col: "Pre-Mom",
    desc: "Pre-Momentum score (0-100). Higher = stronger early momentum signals",
  },
  {
    col: "Conviction",
    desc: "Conviction level: HIGH (green), MEDIUM (cyan), LOW (yellow), NONE (gray)",
  },
  {
    col: "Age",
    desc: "Days since first detected as pre-momentum candidate. Longer = more persistent structural buildup",
  },
  {
    col: "Decision",
    desc: "분석결과 종합 의사결정 (STRONG PREPARE / PREPARE / WATCH / TRACK / STAGNANT / IGNORE). agreement_ratio × PM Score × Age × catalysts 조합 기반. Sort 시 적극 행동(rank 1) → 회피(rank 9) 순서.",
  },
  {
    col: "Agreement",
    desc: "Agent agreement ratio (0-100%). Percentage of agents signaling pre-momentum",
  },
  { col: "Micro", desc: "Microstructure agent score (order flow, volume patterns)" },
  { col: "Macro", desc: "Macro Regime agent score (economic regime alignment)" },
  { col: "Graph", desc: "Graph Relational agent score (network/community signals)" },
  { col: "Catal", desc: "Catalyst agent score (event-driven triggers)" },
  { col: "QVR", desc: "Quality-Value-Revision agent score (fundamentals dimension). 0.30·Q (margin/ROE pctile) + 0.20·V (inverse PE/PEG/PB pctile) + 0.50·R (EPS revision momentum + analyst sentiment + target upside). Most leading — analyst forecast changes precede price. Hover the cell for Q/V/R breakdown. ETFs return neutral 50." },
  { col: "Timeline", desc: "Expected timeline for momentum materialization" },
  { col: "1D", desc: "1-day return (%)" },
  { col: "1W", desc: "1-week return (%) \u2014 5 trading days" },
  { col: "1M", desc: "1-month return (%) \u2014 21 trading days" },
  { col: "3M", desc: "3-month return (%) \u2014 63 trading days" },
  { col: "6M", desc: "6-month return (%) \u2014 126 trading days" },
  { col: "YTD", desc: "Year-to-date return (%) \u2014 prior year-end close to today" },
  { col: "1Y", desc: "1-year return (%) \u2014 252 trading days" },
  { col: "3Y/A", desc: "3-year annualized return (%)" },
  { col: "5Y/A", desc: "5-year annualized return (%)" },
  { col: "Vol3Y", desc: "3-year annualized volatility (%)" },
  { col: "Key Catalyst", desc: "Primary catalyst driving the pre-momentum signal" },
];

function buildColumns(
  onSelect: (ticker: string) => void,
  selectedTicker: string | null
): ColumnDef<Candidate, any>[] {
  return [
    {
      accessorKey: "ticker",
      header: "Ticker",
      cell: ({ row }) => (
        <button
          onClick={() => onSelect(row.original.ticker)}
          className={`font-mono text-[14px] hover:underline ${
            selectedTicker === row.original.ticker
              ? "text-[#0D7680] font-bold"
              : "text-[#0F5499]"
          }`}
        >
          {row.original.ticker}
        </button>
      ),
    },
    {
      accessorKey: "name",
      header: "Name",
      cell: ({ getValue }) => (
        <span className="text-[#33302E] text-[14px] max-w-[120px] truncate block">
          {getValue() as string}
        </span>
      ),
    },
    {
      accessorKey: "sector",
      header: "Sector",
      cell: ({ row }) => (
        <span
          className="text-[#33302E] text-[14px]"
          title={`SubTheme: ${row.original.theme || "-"}`}
        >
          {(row.original as any).sector || row.original.category}
        </span>
      ),
    },
    {
      accessorKey: "theme",
      header: "SubTheme",
      cell: ({ getValue }) => (
        <span className="text-[#857F7A] text-[14px] max-w-[120px] truncate block">
          {getValue() as string}
        </span>
      ),
    },
    {
      accessorKey: "current_classification",
      header: "Class",
      cell: ({ getValue }) => (
        <span className="text-[12px] text-[#66605C]">
          {getValue() as string}
        </span>
      ),
    },
    {
      accessorKey: "current_composite",
      header: "Composite",
      cell: ({ getValue }) => {
        const v = getValue() as number;
        return (
          <span className="text-[14px] font-mono text-[#33302E]">
            {v.toFixed(1)}
          </span>
        );
      },
    },
    {
      accessorKey: "pre_momentum_score",
      header: "Pre-Mom",
      cell: ({ getValue }) => {
        const v = getValue() as number;
        return (
          <span
            className="text-[14px] font-mono font-bold"
            style={{ color: scoreColor(v) }}
          >
            {v.toFixed(1)}
          </span>
        );
      },
    },
    {
      accessorKey: "agreement_ratio",
      header: "Agreement",
      cell: ({ getValue }) => {
        const ratio = (getValue() as number) ?? 0;
        const tier = ratio >= 0.6 ? "STRONG"
                    : ratio >= 0.4 ? "MODERATE"
                    : ratio > 0    ? "WEAK"
                    : "NONE";
        const color = agreementColor(ratio);
        return (
          <span
            className="text-[12px] px-2.5 py-0.5 rounded font-semibold"
            style={{ backgroundColor: color + "22", color }}
            title={`${tier} (${(ratio * 100).toFixed(0)}%)`}
          >
            {(ratio * 100).toFixed(0)}%
          </span>
        );
      },
    },
    {
      accessorKey: "pm_age",
      header: "Age",
      cell: ({ getValue }) => {
        const v = (getValue() as number) ?? 0;
        const color = v >= 14 ? C.green : v >= 7 ? C.cyan : v >= 3 ? C.yellow : C.gray;
        return (
          <span className="text-[14px] font-mono font-semibold" style={{ color }}>
            {v}d
          </span>
        );
      },
    },
    {
      accessorKey: "agreement_ratio",
      header: "Agreement",
      cell: ({ getValue }) => {
        const v = getValue() as number;
        const pct = (v * 100).toFixed(0);
        return (
          <span className="text-[14px] font-mono text-[#33302E]">{pct}%</span>
        );
      },
    },
    {
      id: "micro",
      header: "Micro",
      accessorFn: (row) => row.agents.microstructure?.score ?? 0,
      cell: ({ getValue }) => {
        const v = getValue() as number;
        return (
          <span
            className="text-[14px] font-mono"
            style={{ color: agentScoreColor(v) }}
          >
            {v.toFixed(0)}
          </span>
        );
      },
    },
    {
      id: "macro",
      header: "Macro",
      accessorFn: (row) => row.agents.macro_regime?.score ?? 0,
      cell: ({ getValue }) => {
        const v = getValue() as number;
        return (
          <span
            className="text-[14px] font-mono"
            style={{ color: agentScoreColor(v) }}
          >
            {v.toFixed(0)}
          </span>
        );
      },
    },
    {
      id: "graph",
      header: "Graph",
      accessorFn: (row) => row.agents.graph_relational?.score ?? 0,
      cell: ({ getValue }) => {
        const v = getValue() as number;
        return (
          <span
            className="text-[14px] font-mono"
            style={{ color: agentScoreColor(v) }}
          >
            {v.toFixed(0)}
          </span>
        );
      },
    },
    {
      id: "catalyst",
      header: "Catalyst",
      accessorFn: (row) => row.agents.catalyst?.score ?? 0,
      cell: ({ getValue }) => {
        const v = getValue() as number;
        return (
          <span
            className="text-[14px] font-mono"
            style={{ color: agentScoreColor(v) }}
          >
            {v.toFixed(0)}
          </span>
        );
      },
    },
    {
      accessorKey: "expected_timeline",
      header: "Timeline",
      cell: ({ getValue }) => (
        <span className="text-[12px] text-[#857F7A]">{getValue() as string}</span>
      ),
    },
    {
      id: "key_catalyst",
      header: "Key Catalyst",
      accessorFn: (row) =>
        row.key_catalysts?.length ? row.key_catalysts[0] : "-",
      cell: ({ getValue }) => (
        <span className="text-[12px] text-[#857F7A] max-w-[160px] truncate block">
          {getValue() as string}
        </span>
      ),
    },
  ];
}

// ---------------------------------------------------------------------------
// Conversion Tracking Section
// ---------------------------------------------------------------------------

function ConversionTable({ rows, variant }: { rows: ConversionEntry[]; variant: "graduated" | "failed" | "in_progress" }) {
  if (!rows.length) return <p className="text-[12px] text-[#857F7A]">No data</p>;
  const isGrad = variant === "graduated";
  const isFail = variant === "failed";
  return (
    <div className="overflow-x-auto border border-[#E6D9CE] rounded">
      <table className="w-full text-[14px] border-collapse">
        <thead>
          <tr className="border-b border-[#E6D9CE] bg-[#FFFFFF]">
            <th className="py-1.5 px-2 text-left text-[#857F7A]">#</th>
            <th className="py-1.5 px-2 text-left text-[#857F7A]">Ticker</th>
            <th className="py-1.5 px-2 text-left text-[#857F7A]">Name</th>
            <th className="py-1.5 px-2 text-left text-[#857F7A]">Category</th>
            <th className="py-1.5 px-2 text-right text-[#857F7A]">1M Ago</th>
            <th className="py-1.5 px-2 text-right text-[#857F7A]">Now</th>
            <th className="py-1.5 px-2 text-right text-[#857F7A]">Change</th>
            <th className="py-1.5 px-2 text-left text-[#857F7A]">Current Class</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const changeColor = r.score_change > 20 ? C.green : r.score_change > 0 ? C.cyan : r.score_change > -10 ? C.yellow : C.red;
            const tickerColor = isGrad ? C.green : isFail ? C.red : C.cyan;
            return (
              <tr key={r.ticker} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/30">
                <td className="py-1.5 px-2 text-[#857F7A]">{i + 1}</td>
                <td className="py-1.5 px-2 font-mono font-bold" style={{ color: tickerColor }}>{r.ticker}</td>
                <td className="py-1.5 px-2 text-[#66605C] truncate max-w-[120px]">{r.name}</td>
                <td className="py-1.5 px-2 text-[#857F7A] text-[12px]">{r.category}</td>
                <td className="py-1.5 px-2 text-right font-mono text-[#857F7A]">{r.score_1m_ago.toFixed(1)}</td>
                <td className="py-1.5 px-2 text-right font-mono text-[#33302E]">{r.current_composite.toFixed(1)}</td>
                <td className="py-1.5 px-2 text-right font-mono font-semibold" style={{ color: changeColor }}>
                  {r.score_change > 0 ? "+" : ""}{r.score_change.toFixed(1)}
                </td>
                <td className="py-1.5 px-2 text-[12px] text-[#66605C]">{r.current_class}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ConversionTracking({ conversion }: { conversion: ConversionData }) {
  const { stats, graduated, failed, in_progress } = conversion;

  return (
    <div className="space-y-4">
      <p className="text-[13px] text-[#857F7A]">
        1개월 전 pre-momentum 상태(not eligible, composite 25-54)였던 종목들의 현재 전환 결과.
      </p>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard label="PM Candidates (1M)" value={stats.total_pm_candidates_1m} sub="1개월 전 PM 상태 종목" />
        <MetricCard label="Graduated" value={stats.total_graduated} sub="→ Momentum confirmed" />
        <MetricCard label="Failed" value={stats.total_failed} sub="→ Bearish exit" />
        <MetricCard label="Hit Rate" value={`${stats.hit_rate}%`} sub={`${stats.total_graduated}/${stats.total_graduated + stats.total_failed}`} />
        <MetricCard label="Avg Improvement" value={`+${stats.avg_score_improvement}`} sub="Graduated avg score change" />
      </div>

      {/* Graduated */}
      <div>
        <h4 className="text-[14px] font-semibold text-[#0A7D3F] uppercase tracking-wide mb-2">
          Graduated — PM → Momentum ({graduated.length})
        </h4>
        <ConversionTable rows={graduated.slice(0, 30)} variant="graduated" />
      </div>

      {/* In Progress */}
      {in_progress.length > 0 && (
        <div>
          <h4 className="text-[14px] font-semibold text-[#0F5499] uppercase tracking-wide mb-2">
            In Progress — Still Building ({in_progress.length})
          </h4>
          <ConversionTable rows={in_progress.slice(0, 20)} variant="in_progress" />
        </div>
      )}

      {/* Failed */}
      {failed.length > 0 && (
        <div>
          <h4 className="text-[14px] font-semibold text-[#CC0000] uppercase tracking-wide mb-2">
            Failed — PM → Bearish ({failed.length})
          </h4>
          <ConversionTable rows={failed.slice(0, 20)} variant="failed" />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Agent Score Distribution (Box Plot)
// ---------------------------------------------------------------------------

function AgentScoreBoxPlot({ candidates }: { candidates: Candidate[] }) {
  if (!candidates.length) return null;
  const agents: { key: keyof Candidate["agents"]; label: string; color: string }[] = [
    { key: "microstructure", label: "Microstructure", color: C.cyan },
    { key: "macro_regime", label: "Macro Regime", color: C.blue },
    { key: "graph_relational", label: "Graph Relational", color: C.purple },
    { key: "catalyst", label: "Catalyst", color: C.orange },
    { key: "qvr", label: "QVR", color: C.green },
  ];
  const traces = agents.map(({ key, label, color }) => ({
    type: "box" as const,
    y: candidates.map((c) => c.agents[key]?.score ?? 0),
    name: label,
    marker: { color, opacity: 0.8 },
    boxmean: "sd" as const,
    line: { color },
    fillcolor: color + "22",
  }));
  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-1">Agent Score Distribution</h3>
      <p className="text-[12px] text-[#857F7A] mb-3">전체 후보 대상 에이전트별 점수 분포. 박스: Q1-Q3, 선: 중앙값, 다이아몬드: 평균±σ</p>
      <Plot
        data={traces}
        layout={{
          ...DARK_LAYOUT,
          height: 320,
          margin: { t: 10, b: 40, l: 50, r: 20 },
          yaxis: { title: "Score", range: [0, 105], gridcolor: "#F2E5D7", color: "#66605C" },
          showlegend: false,
          boxgap: 0.3,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-Signal Average per Agent (Grouped Bar)
// ---------------------------------------------------------------------------

function SubSignalAverageChart({ candidates }: { candidates: Candidate[] }) {
  if (!candidates.length) return null;
  const agents: { key: keyof Candidate["agents"]; label: string; color: string }[] = [
    { key: "microstructure", label: "Micro", color: C.cyan },
    { key: "macro_regime", label: "Macro", color: C.blue },
    { key: "graph_relational", label: "Graph", color: C.purple },
    { key: "catalyst", label: "Catalyst", color: C.orange },
    { key: "qvr", label: "QVR", color: C.green },
  ];
  // Collect all signal keys per agent and compute averages
  const traces: any[] = [];
  for (const { key, label, color } of agents) {
    const allKeys = new Set<string>();
    for (const c of candidates) {
      const s = c.agents[key]?.signals;
      if (s) Object.keys(s).forEach((k) => allKeys.add(k));
    }
    const sigKeys = [...allKeys].sort();
    const avgs = sigKeys.map((sk) => {
      const vals = candidates.map((c) => c.agents[key]?.signals?.[sk] ?? 0);
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    });
    traces.push({
      type: "bar" as const,
      x: sigKeys.map((s) => s.replace(/_/g, " ")),
      y: avgs,
      name: label,
      marker: { color, opacity: 0.85 },
      text: avgs.map((v) => v.toFixed(1)),
      textposition: "outside" as const,
      textfont: { color: C.text, size: 9 },
      hovertemplate: `${label}<br>%{x}: %{y:.1f}<extra></extra>`,
    });
  }
  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-1">Sub-Signal Average by Agent</h3>
      <p className="text-[12px] text-[#857F7A] mb-3">전체 후보의 서브시그널 평균. 어떤 시그널이 전반적으로 강한지/약한지를 파악</p>
      <Plot
        data={traces}
        layout={{
          ...DARK_LAYOUT,
          height: 360,
          margin: { t: 10, b: 100, l: 50, r: 20 },
          barmode: "group",
          xaxis: { tickangle: -35, tickfont: { size: 9, color: "#66605C" }, gridcolor: "#F2E5D7" },
          yaxis: { title: "Avg Score", range: [0, 105], gridcolor: "#F2E5D7", color: "#66605C" },
          legend: { font: { size: 10, color: C.text }, bgcolor: "rgba(0,0,0,0)", x: 0.7, y: 1.0 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Agent Score Scatter (pairwise)
// ---------------------------------------------------------------------------

function AgentScatterMatrix({ candidates }: { candidates: Candidate[] }) {
  if (candidates.length < 5) return null;
  const pairs: { xKey: keyof Candidate["agents"]; yKey: keyof Candidate["agents"]; xLabel: string; yLabel: string }[] = [
    { xKey: "microstructure", yKey: "catalyst", xLabel: "Microstructure", yLabel: "Catalyst" },
    { xKey: "macro_regime", yKey: "graph_relational", xLabel: "Macro Regime", yLabel: "Graph Relational" },
    { xKey: "microstructure", yKey: "macro_regime", xLabel: "Microstructure", yLabel: "Macro Regime" },
    { xKey: "graph_relational", yKey: "catalyst", xLabel: "Graph Relational", yLabel: "Catalyst" },
  ];
  // Group by agreement tier (replaces conviction)
  const tierLabel = (r: number) => r >= 0.6 ? "STRONG" : r >= 0.4 ? "MODERATE" : r > 0 ? "WEAK" : "NONE";
  const tierColors: Record<string, string> = { STRONG: C.green, MODERATE: C.cyan, WEAK: C.yellow, NONE: C.gray };
  const subplots = pairs.map(({ xKey, yKey, xLabel, yLabel }, idx) => {
    const groups: Record<string, Candidate[]> = {};
    for (const c of candidates) {
      const tier = tierLabel(c.agreement_ratio || 0);
      if (!groups[tier]) groups[tier] = [];
      groups[tier].push(c);
    }
    return Object.entries(groups).map(([tier, group]) => ({
      type: "scatter" as const,
      x: group.map((c) => c.agents[xKey]?.score ?? 0),
      y: group.map((c) => c.agents[yKey]?.score ?? 0),
      mode: "markers" as const,
      name: tier,
      legendgroup: tier,
      showlegend: idx === 0,
      marker: { color: tierColors[tier] || C.gray, size: 6, opacity: 0.7 },
      text: group.map((c) => c.ticker),
      hovertemplate: `%{text}<br>${xLabel}: %{x:.1f}<br>${yLabel}: %{y:.1f}<br>${tier}<extra></extra>`,
      xaxis: `x${idx === 0 ? "" : idx + 1}`,
      yaxis: `y${idx === 0 ? "" : idx + 1}`,
    }));
  }).flat();
  const axisBase = { gridcolor: "#F2E5D7", color: "#66605C", range: [0, 100], tickfont: { size: 9 } };
  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-1">Agent Score Scatter — Pairwise</h3>
      <p className="text-[12px] text-[#857F7A] mb-3">에이전트 간 점수 상관관계. 색상 = Agreement tier (STRONG/MODERATE/WEAK/NONE). 우상단 클러스터 = 양쪽 모두 강한 시그널</p>
      <Plot
        data={subplots}
        layout={{
          ...DARK_LAYOUT,
          height: 580,
          margin: { t: 30, b: 50, l: 50, r: 20 },
          grid: { rows: 2, columns: 2, pattern: "independent" as const },
          xaxis: { ...axisBase, title: { text: pairs[0].xLabel, font: { size: 10 } }, domain: [0, 0.47] },
          yaxis: { ...axisBase, title: { text: pairs[0].yLabel, font: { size: 10 } }, domain: [0.55, 1] },
          xaxis2: { ...axisBase, title: { text: pairs[1].xLabel, font: { size: 10 } }, domain: [0.53, 1] },
          yaxis2: { ...axisBase, title: { text: pairs[1].yLabel, font: { size: 10 } }, domain: [0.55, 1] },
          xaxis3: { ...axisBase, title: { text: pairs[2].xLabel, font: { size: 10 } }, domain: [0, 0.47] },
          yaxis3: { ...axisBase, title: { text: pairs[2].yLabel, font: { size: 10 } }, domain: [0, 0.45] },
          xaxis4: { ...axisBase, title: { text: pairs[3].xLabel, font: { size: 10 } }, domain: [0.53, 1] },
          yaxis4: { ...axisBase, title: { text: pairs[3].yLabel, font: { size: 10 } }, domain: [0, 0.45] },
          legend: { font: { size: 10, color: C.text }, bgcolor: "rgba(0,0,0,0)" },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Agreement Profile — avg agent scores by agreement tier
// ---------------------------------------------------------------------------

function ConvictionProfileChart({ candidates }: { candidates: Candidate[] }) {
  if (!candidates.length) return null;
  const levels = ["STRONG", "MODERATE", "WEAK", "NONE"];
  const tierLabel = (r: number) => r >= 0.6 ? "STRONG" : r >= 0.4 ? "MODERATE" : r > 0 ? "WEAK" : "NONE";
  const agents: { key: keyof Candidate["agents"]; label: string; color: string }[] = [
    { key: "microstructure", label: "Micro", color: C.cyan },
    { key: "macro_regime", label: "Macro", color: C.blue },
    { key: "graph_relational", label: "Graph", color: C.purple },
    { key: "catalyst", label: "Catalyst", color: C.orange },
    { key: "qvr", label: "QVR", color: C.green },
  ];
  const traces = agents.map(({ key, label, color }) => {
    const avgs = levels.map((lv) => {
      const group = candidates.filter((c) => tierLabel(c.agreement_ratio || 0) === lv);
      if (!group.length) return 0;
      return group.reduce((s, c) => s + (c.agents[key]?.score ?? 0), 0) / group.length;
    });
    return {
      type: "bar" as const,
      x: levels,
      y: avgs,
      name: label,
      marker: { color, opacity: 0.85 },
      text: avgs.map((v) => v.toFixed(1)),
      textposition: "outside" as const,
      textfont: { size: 9, color: C.text },
    };
  });
  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-1">Agreement Profile — Agent Contribution</h3>
      <p className="text-[12px] text-[#857F7A] mb-3">Agreement tier (STRONG/MODERATE/WEAK/NONE)별 에이전트 평균 점수. STRONG agreement에서 어떤 에이전트가 가장 기여하는지 비교</p>
      <Plot
        data={traces}
        layout={{
          ...DARK_LAYOUT,
          height: 320,
          margin: { t: 10, b: 40, l: 50, r: 20 },
          barmode: "group",
          xaxis: { color: "#66605C" },
          yaxis: { title: "Avg Score", range: [0, 105], gridcolor: "#F2E5D7", color: "#66605C" },
          legend: { font: { size: 10, color: C.text }, bgcolor: "rgba(0,0,0,0)", orientation: "h" as const, y: 1.12 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-Signal Contribution — Top 20 candidates stacked horizontal bar
// ---------------------------------------------------------------------------

function SubSignalContributionChart({ candidates }: { candidates: Candidate[] }) {
  const top = candidates
    .slice()
    .sort((a, b) => b.pre_momentum_score - a.pre_momentum_score)
    .slice(0, 20);
  if (!top.length) return null;
  const agents: { key: keyof Candidate["agents"]; label: string; color: string; weight: number }[] = [
    { key: "microstructure", label: "Micro (20%)", color: C.cyan, weight: 0.20 },
    { key: "macro_regime", label: "Macro (15%)", color: C.blue, weight: 0.15 },
    { key: "graph_relational", label: "Graph (20%)", color: C.purple, weight: 0.20 },
    { key: "catalyst", label: "Catalyst (20%)", color: C.orange, weight: 0.20 },
    { key: "qvr", label: "QVR (25%)", color: C.green, weight: 0.25 },
  ];
  const tickers = top.map((c) => c.ticker).reverse(); // reverse for horizontal bar bottom-to-top
  const traces = agents.map(({ key, label, color, weight }) => ({
    type: "bar" as const,
    y: tickers,
    x: [...top].reverse().map((c) => (c.agents[key]?.score ?? 0) * weight),
    name: label,
    orientation: "h" as const,
    marker: { color, opacity: 0.85 },
    hovertemplate: `%{y}<br>${label}: %{x:.1f}<extra></extra>`,
  }));
  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-1">Weighted Agent Contribution — Top 20</h3>
      <p className="text-[12px] text-[#857F7A] mb-3">최종 Pre-Mom Score에 대한 에이전트별 가중 기여도. 각 바의 총 길이 = Pre-Momentum Score</p>
      <Plot
        data={traces}
        layout={{
          ...DARK_LAYOUT,
          height: Math.max(400, top.length * 24 + 80),
          margin: { t: 10, b: 40, l: 70, r: 20 },
          barmode: "stack",
          xaxis: { title: "Weighted Score Contribution", gridcolor: "#F2E5D7", color: "#66605C" },
          yaxis: { automargin: true, tickfont: { size: 10, color: "#66605C" } },
          legend: { font: { size: 10, color: C.text }, bgcolor: "rgba(0,0,0,0)", orientation: "h" as const, y: 1.05 },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Agent Score Distribution Histogram
// ---------------------------------------------------------------------------

function AgentHistograms({ candidates }: { candidates: Candidate[] }) {
  if (!candidates.length) return null;
  const agents: { key: keyof Candidate["agents"]; label: string; color: string }[] = [
    { key: "microstructure", label: "Microstructure", color: C.cyan },
    { key: "macro_regime", label: "Macro Regime", color: C.blue },
    { key: "graph_relational", label: "Graph Relational", color: C.purple },
    { key: "catalyst", label: "Catalyst", color: C.orange },
    { key: "qvr", label: "QVR", color: C.green },
  ];
  const traces = agents.map(({ key, label, color }) => ({
    type: "histogram" as const,
    x: candidates.map((c) => c.agents[key]?.score ?? 0),
    name: label,
    marker: { color, opacity: 0.65 },
    xbins: { start: 0, end: 100, size: 5 },
    hovertemplate: `${label}<br>Score: %{x}<br>Count: %{y}<extra></extra>`,
  }));
  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4">
      <h3 className="text-[16px] font-semibold text-[#66605C] mb-1">Agent Score Histograms</h3>
      <p className="text-[12px] text-[#857F7A] mb-3">에이전트별 점수 분포. 50점 기준선(시그널 임계값) 대비 분포를 통해 각 에이전트의 전반적 활성도 파악</p>
      <Plot
        data={traces}
        layout={{
          ...DARK_LAYOUT,
          height: 300,
          margin: { t: 10, b: 40, l: 50, r: 20 },
          barmode: "overlay",
          xaxis: { title: "Score", range: [0, 100], gridcolor: "#F2E5D7", color: "#66605C" },
          yaxis: { title: "Count", gridcolor: "#F2E5D7", color: "#66605C" },
          legend: { font: { size: 10, color: C.text }, bgcolor: "rgba(0,0,0,0)", orientation: "h" as const, y: 1.12 },
          shapes: [{ type: "line", x0: 50, x1: 50, y0: 0, y1: 1, yref: "paper", line: { color: "#CC0000", width: 1, dash: "dot" } }],
          annotations: [{ x: 51, y: 1, yref: "paper", text: "Threshold", showarrow: false, font: { color: "#CC0000", size: 9 } }],
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Composite vs Pre-Mom Score Interpretation Guide
// ---------------------------------------------------------------------------

function CompPmScoreGuide() {
  const [open, setOpen] = useState(false);
  const cases = [
    {
      label: "A. 좋은 후보",
      comp: "35 ~ 50",
      pm: "60+",
      compColor: "text-[#B85C00]",
      pmColor: "text-[#0A7D3F]",
      caseColor: "text-[#0A7D3F]",
      action: "WATCH / PREPARE",
      actionColor: "text-[#0A7D3F]",
      desc: "가격은 아직 도달 안 했으나 구조적 조건 갖춰짐 — 1-2주 내 돌파 예상",
    },
    {
      label: "B. 약한 신호",
      comp: "35 ~ 50",
      pm: "30 ~ 50",
      compColor: "text-[#B85C00]",
      pmColor: "text-[#66605C]",
      caseColor: "text-[#66605C]",
      action: "IGNORE",
      actionColor: "text-[#857F7A]",
      desc: "양쪽 모두 약함 — 관심 대상 아님",
    },
    {
      label: "C. 함정 (Trap)",
      comp: "50 ~ 54",
      pm: "<40",
      compColor: "text-[#0F5499]",
      pmColor: "text-[#CC0000]",
      caseColor: "text-[#C2701C]",
      action: "AVOID",
      actionColor: "text-[#C2701C]",
      desc: "가격은 거의 eligible 임계 도달했으나 구조 약함 — 진입 보류 (false positive 위험)",
    },
    {
      label: "D. Prime Candidate",
      comp: "30 ~ 40",
      pm: "70+ (HIGH)",
      compColor: "text-[#B85C00]",
      pmColor: "text-[#0A7D3F]",
      caseColor: "text-[#0D7680]",
      action: "PREPARE — Top Priority",
      actionColor: "text-[#0F5499]",
      desc: "강한 압축 + 매크로 우호 — coiled spring, 즉시 관찰 시작",
    },
  ];

  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden">
      <button
        className="w-full px-4 py-2 text-left text-[14px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between items-center"
        onClick={() => setOpen(!open)}
      >
        <span className="text-[#66605C]">
          Composite vs Pre-Mom Score 해석 가이드
          <span className="ml-2 text-[12px] text-[#857F7A]">— 두 점수 조합으로 종목 단계 판정</span>
        </span>
        <span className="text-[#857F7A] text-[12px]">{open ? "▼" : "▶"}</span>
      </button>
      {open && (
        <div className="bg-[#FBEEE3] p-4 space-y-3">
          <p className="text-[13px] text-[#857F7A] leading-relaxed">
            <strong className="text-[#33302E]">Composite</strong>는 "지금" 가격 모멘텀을 측정하고 (PD 스캐너:
            0.35×TCS + 0.30×TFS + 0.35×RSS), <strong className="text-[#33302E]">Pre-Mom Score</strong>는 "곧"
            모멘텀이 형성될 구조적 조건을 측정합니다 (4-Agent: Micro/Macro/Graph/Catalyst).
            Pre-Momentum 탭은 <code className="text-[#0F5499]">eligible=False AND composite&lt;55</code> 종목만 표시되므로,
            두 점수의 조합으로 종목의 단계를 판정합니다.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-[14px] border-collapse">
              <thead>
                <tr className="border-b border-[#E6D9CE] bg-[#FFFFFF]">
                  <th className="py-2 px-2 text-left text-[#857F7A] font-semibold">케이스</th>
                  <th className="py-2 px-2 text-center text-[#857F7A] font-semibold">Composite</th>
                  <th className="py-2 px-2 text-center text-[#857F7A] font-semibold">PM Score</th>
                  <th className="py-2 px-2 text-left text-[#857F7A] font-semibold">Action</th>
                  <th className="py-2 px-2 text-left text-[#857F7A] font-semibold">해석</th>
                </tr>
              </thead>
              <tbody>
                {cases.map((c, i) => (
                  <tr key={i} className="border-b border-[#E6D9CE]/50 align-top hover:bg-[#F2E5D7]/30">
                    <td className={`py-2 px-2 font-bold ${c.caseColor}`}>{c.label}</td>
                    <td className={`py-2 px-2 text-center font-mono ${c.compColor}`}>{c.comp}</td>
                    <td className={`py-2 px-2 text-center font-mono font-bold ${c.pmColor}`}>{c.pm}</td>
                    <td className={`py-2 px-2 font-semibold text-[13px] ${c.actionColor}`}>{c.action}</td>
                    <td className="py-2 px-2 text-[#66605C] text-[13px]">{c.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="bg-[#FFFFFF] border border-[#9CC3D5]/40 rounded p-2 mt-2">
            <div className="text-[13px] text-[#0D7680]">
              <strong>핵심 가치:</strong>{" "}
              <span className="text-[#66605C]">
                케이스 A·D가 Pre-Momentum 시스템의 핵심 — Composite만으로는 놓칠 수 있는
                "곧 움직일 종목"을 PM Score가 잡아냅니다.
              </span>
            </div>
          </div>

          <div className="bg-[#FFFFFF] border border-[#E0C3A0]/40 rounded p-2">
            <div className="text-[13px] text-[#C2701C]">
              <strong>주의:</strong>{" "}
              <span className="text-[#66605C]">
                케이스 C는 Composite이 임계 근처라 진입하고 싶은 충동이 있으나,
                구조적 신호가 약하면 false breakout 가능성이 높음.
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Top-10 Compact Table
// ---------------------------------------------------------------------------

function CandidateTable({
  rows,
  onSelect,
}: {
  rows: Candidate[];
  onSelect: (ticker: string) => void;
}) {
  const accessors = useMemo(() => ({
    ticker: (c: Candidate) => c.ticker,
    name: (c: Candidate) => c.name,
    sector: (c: Candidate) => c.sector || c.category,
    classification: (c: Candidate) => c.current_classification,
    composite: (c: Candidate) => c.current_composite,
    pm_score: (c: Candidate) => c.pre_momentum_score,
    agreement: (c: Candidate) => c.agreement_ratio ?? 0,
    age: (c: Candidate) => c.pm_age ?? 0,
    decision: (c: Candidate) => decidePMAction(c).rank,
    agree: (c: Candidate) => c.agreement_ratio,
    micro: (c: Candidate) => c.agents?.microstructure?.score ?? 0,
    macro: (c: Candidate) => c.agents?.macro_regime?.score ?? 0,
    graph: (c: Candidate) => c.agents?.graph_relational?.score ?? 0,
    catalyst: (c: Candidate) => c.agents?.catalyst?.score ?? 0,
    qvr: (c: Candidate) => c.agents?.qvr?.score ?? 0,
    timeline: (c: Candidate) => c.expected_timeline,
    ret_1d: (c: Candidate) => c.ret_1d ?? 0,
    ret_5d: (c: Candidate) => c.ret_5d ?? 0,
    ret_21d: (c: Candidate) => c.ret_21d ?? 0,
    ret_63d: (c: Candidate) => c.ret_63d ?? 0,
    ret_126d: (c: Candidate) => c.ret_126d ?? 0,
    ret_ytd: (c: Candidate) => c.ret_ytd ?? 0,
    ret_252d: (c: Candidate) => c.ret_252d ?? 0,
    ret_3y_ann: (c: Candidate) => c.ret_3y_ann ?? -999,
    ret_5y_ann: (c: Candidate) => c.ret_5y_ann ?? -999,
    vol_3y_ann: (c: Candidate) => c.vol_3y_ann ?? -999,
  }), []);
  const { sorted, onSort, indicator } = useSort(rows, accessors);

  if (!rows || rows.length === 0) return <p className="text-[12px] text-[#857F7A]">No data</p>;
  const agentKeys: (keyof Candidate["agents"])[] = ["microstructure", "macro_regime", "graph_relational", "catalyst", "qvr"];
  const agentHeaders: { key: string; label: string }[] = [
    { key: "micro", label: "Micro" }, { key: "macro", label: "Macro" },
    { key: "graph", label: "Graph" }, { key: "catalyst", label: "Catal" },
    { key: "qvr", label: "QVR" },
  ];
  const headerCls = "py-1.5 px-2 text-[#857F7A] cursor-pointer select-none hover:text-[#33302E] whitespace-nowrap";
  return (
    <div className="overflow-auto border border-[#E6D9CE] rounded" style={{ maxHeight: "600px" }}>
      <table className="w-full text-[14px] border-collapse">
        <thead className="sticky top-0 z-10 bg-[#F2E5D7]">
          <tr className="border-b border-[#E6D9CE]">
            <th className="py-1.5 px-2 text-left text-[#857F7A] w-6">#</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("ticker")}>Ticker{indicator("ticker")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("name")}>Name{indicator("name")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("sector")}>Sector{indicator("sector")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("classification")}>Class{indicator("classification")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("composite")}>Comp{indicator("composite")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("pm_score")}>Pre-Mom{indicator("pm_score")}</th>
            <th className={`${headerCls} text-left`} onClick={() => onSort("decision")}>Decision{indicator("decision")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("age")}>Age{indicator("age")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("agree")}>Agree{indicator("agree")}</th>
            {agentHeaders.map(h => (
              <th key={h.key} className={`${headerCls} text-right`} onClick={() => onSort(h.key)}>
                {h.label}{indicator(h.key)}
              </th>
            ))}
            <th className={`${headerCls} text-left`} onClick={() => onSort("timeline")}>Timeline{indicator("timeline")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_1d")}>1D{indicator("ret_1d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_5d")}>1W{indicator("ret_5d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_21d")}>1M{indicator("ret_21d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_63d")}>3M{indicator("ret_63d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_126d")}>6M{indicator("ret_126d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_ytd")}>YTD{indicator("ret_ytd")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_252d")}>1Y{indicator("ret_252d")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_3y_ann")}>3Y/A{indicator("ret_3y_ann")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("ret_5y_ann")}>5Y/A{indicator("ret_5y_ann")}</th>
            <th className={`${headerCls} text-right`} onClick={() => onSort("vol_3y_ann")}>Vol3Y{indicator("vol_3y_ann")}</th>
            <th className="py-1.5 px-2 text-left text-[#857F7A]">Key Catalyst</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((c, i) => {
            const age = c.pm_age ?? 0;
            const ageColor = age >= 14 ? C.green : age >= 7 ? C.cyan : age >= 3 ? C.yellow : C.gray;
            return (
              <tr key={c.ticker} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/30">
                <td className="py-1.5 px-2 text-[#857F7A]">{i + 1}</td>
                <td className="py-1.5 px-2">
                  <button onClick={() => onSelect(c.ticker)} className="font-mono text-[14px] text-[#0F5499] hover:underline font-bold">
                    {c.ticker}
                  </button>
                </td>
                <td className="py-1.5 px-2 text-[#66605C] truncate max-w-[120px]">{c.name}</td>
                <td className="py-1.5 px-2 text-[#857F7A] text-[12px]" title={`SubTheme: ${c.theme || "-"}`}>
                  {c.sector || c.category}
                </td>
                <td className="py-1.5 px-2 text-[12px] text-[#66605C]">{c.current_classification}</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono text-[#33302E]">{c.current_composite.toFixed(1)}</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono font-bold" style={{ color: scoreColor(c.pre_momentum_score) }}>
                  {c.pre_momentum_score.toFixed(1)}
                </td>
                {(() => {
                  const d = decidePMAction(c);
                  return (
                    <td className="py-1.5 px-2 align-top" style={{ minWidth: "320px" }}>
                      <div className="flex flex-col gap-0.5">
                        <span className="text-[13px] font-bold whitespace-nowrap" style={{ color: d.color }}>
                          {d.action}
                        </span>
                        <span className="text-[12px] text-[#857F7A] leading-snug whitespace-normal break-words">
                          {d.rationale}
                        </span>
                      </div>
                    </td>
                  );
                })()}
                <td className="py-1.5 px-2 text-right text-[14px] font-mono font-semibold" style={{ color: ageColor }}>
                  {age}d
                </td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono font-semibold"
                    style={{ color: agreementColor(c.agreement_ratio || 0) }}
                    title={(c.agreement_ratio || 0) >= 0.6 ? "STRONG"
                          : (c.agreement_ratio || 0) >= 0.4 ? "MODERATE"
                          : (c.agreement_ratio || 0) > 0 ? "WEAK" : "NONE"}>
                  {(c.agreement_ratio * 100).toFixed(0)}%
                </td>
                {agentKeys.map(k => {
                  const ag = c.agents[k];
                  const score = ag?.score ?? 0;
                  // QVR: surface Q/V/R sub-scores + revision counts in tooltip
                  let title: string | undefined;
                  if (k === "qvr" && ag?.signals) {
                    const s = ag.signals as any;
                    title = `Q ${s.quality ?? "-"} | V ${s.value ?? "-"} | R ${s.revision ?? "-"}`
                          + ` · net30d ${s.net_30d ?? 0} (ratio ${s.ratio_30d ?? "-"}%)`
                          + ` · ${s.n_analysts ?? 0} analysts`
                          + (ag.summary ? ` — ${ag.summary}` : "");
                  }
                  return (
                    <td key={k} className="py-1.5 px-2 text-right text-[14px] font-mono"
                        style={{ color: agentScoreColor(score) }} title={title}>
                      {score.toFixed(0)}
                    </td>
                  );
                })}
                <td className="py-1.5 px-2 text-[12px] text-[#857F7A]">{c.expected_timeline}</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: retColor(c.ret_1d ?? 0) }}>{(c.ret_1d ?? 0).toFixed(1)}%</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: retColor(c.ret_5d ?? 0) }}>{(c.ret_5d ?? 0).toFixed(1)}%</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: retColor(c.ret_21d ?? 0) }}>{(c.ret_21d ?? 0).toFixed(1)}%</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: retColor(c.ret_63d ?? 0) }}>{(c.ret_63d ?? 0).toFixed(1)}%</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: retColor(c.ret_126d ?? 0) }}>{(c.ret_126d ?? 0).toFixed(1)}%</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: retColor(c.ret_ytd ?? 0) }}>{c.ret_ytd != null ? `${c.ret_ytd.toFixed(1)}%` : "—"}</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: retColor(c.ret_252d ?? 0) }}>{(c.ret_252d ?? 0).toFixed(1)}%</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: c.ret_3y_ann == null ? C.gray : retColor(c.ret_3y_ann) }}>{c.ret_3y_ann == null ? "-" : `${c.ret_3y_ann.toFixed(1)}%`}</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono" style={{ color: c.ret_5y_ann == null ? C.gray : retColor(c.ret_5y_ann) }}>{c.ret_5y_ann == null ? "-" : `${c.ret_5y_ann.toFixed(1)}%`}</td>
                <td className="py-1.5 px-2 text-right text-[14px] font-mono text-[#66605C]">{c.vol_3y_ann == null ? "-" : `${c.vol_3y_ann.toFixed(1)}%`}</td>
                <td className="py-1.5 px-2 text-[12px] text-[#857F7A] truncate max-w-[160px]">{c.key_catalysts?.[0] ?? "-"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Export
// ---------------------------------------------------------------------------

export function PreMomentumTab({ totalUniverse, filterSectors, mlMode = false }:
  { totalUniverse?: number; filterSectors?: string[]; mlMode?: boolean }) {
  const [data, setData] = useState<PreMomentumData | null>(null);
  const [selectedTicker, setSelectedTicker] = useState<string | null>(null);

  useEffect(() => {
    const fetcher = mlMode ? fetchPreMomentumML : fetchPreMomentum;
    fetcher().then(setData);
  }, [mlMode]);

  // Filter candidates by sidebar sector selection (Option B)
  const filteredCandidates = useMemo(() => {
    if (!data) return [];
    const secs = filterSectors && filterSectors.length > 0 ? new Set(filterSectors) : null;
    const all = data.candidates;
    return secs ? all.filter((c) => secs.has((c as any).sector || "")) : all;
  }, [data, filterSectors]);

  const filteredEtf = useMemo(() => {
    if (!data) return [];
    const secs = filterSectors && filterSectors.length > 0 ? new Set(filterSectors) : null;
    const all = data.candidates_etf ?? [];
    return secs ? all.filter((c) => secs.has((c as any).sector || "")) : all;
  }, [data, filterSectors]);

  const filteredStock = useMemo(() => {
    if (!data) return [];
    const secs = filterSectors && filterSectors.length > 0 ? new Set(filterSectors) : null;
    const all = data.candidates_stock ?? [];
    return secs ? all.filter((c) => secs.has((c as any).sector || "")) : all;
  }, [data, filterSectors]);

  // Sorted candidates (default: pre_momentum_score desc)
  const sortedCandidates = useMemo(() => {
    return [...filteredCandidates].sort(
      (a, b) => b.pre_momentum_score - a.pre_momentum_score
    );
  }, [filteredCandidates]);

  const selectedCandidate = useMemo(() => {
    if (!selectedTicker || !data) return null;
    return data.candidates.find((c) => c.ticker === selectedTicker) ?? null;
  }, [selectedTicker, data]);

  const columns = useMemo(
    () => buildColumns(setSelectedTicker, selectedTicker),
    [selectedTicker]
  );

  // Average pre-momentum score for strong + moderate agreement candidates
  const avgPreMomScore = useMemo(() => {
    const eligible = filteredCandidates.filter(
      (c) => (c.agreement_ratio || 0) >= 0.4
    );
    if (!eligible.length) return 0;
    return (
      eligible.reduce((s, c) => s + c.pre_momentum_score, 0) / eligible.length
    );
  }, [filteredCandidates]);

  // Classification distribution (of current candidates)
  const classDist = useMemo(() => {
    const map: Record<string, number> = {};
    for (const c of filteredCandidates) {
      const cls = c.current_classification || "";
      map[cls] = (map[cls] || 0) + 1;
    }
    return Object.entries(map).sort((a, b) => b[1] - a[1]);
  }, [filteredCandidates]);

  if (!data)
    return (
      <div className="text-[#857F7A] p-8">
        Loading pre-momentum analysis...
      </div>
    );

  const { summary } = data;

  return (
    <div className="space-y-5">
      {/* ── KPIs ── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          label="Candidates"
          value={sortedCandidates.length}
          sub={`/ ${totalUniverse || summary.total_universe} universe`}
        />
        <MetricCard label="STRONG" value={sortedCandidates.filter((c) => (c.agreement_ratio || 0) >= 0.6).length} sub="agreement ≥60%" />
        <MetricCard label="MODERATE" value={sortedCandidates.filter((c) => { const r = c.agreement_ratio || 0; return r >= 0.4 && r < 0.6; }).length} sub="agreement 40-60%" />
        <MetricCard
          label="Avg Score"
          value={avgPreMomScore.toFixed(1)}
          sub="strong+moderate avg"
        />
        {data.conversion && (
          <MetricCard
            label="1M Hit Rate"
            value={`${data.conversion.stats.hit_rate}%`}
            sub={`${data.conversion.stats.total_graduated}/${data.conversion.stats.total_graduated + data.conversion.stats.total_failed}`}
          />
        )}
      </div>

      {/* ── Classification Breakdown ── */}
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

      {/* ── Candidate Tables: Total → ETF → Stock ── */}
      <ColDefToggle title="Column Definitions" defs={TABLE_COL_DEFS} />

      {/* ── Composite vs Pre-Mom Score 해석 가이드 ── */}
      <CompPmScoreGuide />

      <CandidateTable rows={sortedCandidates} onSelect={setSelectedTicker} />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <div>
          <h3 className="text-[14px] font-semibold text-[#0F5499] uppercase tracking-wide mb-2">
            ETF Candidates ({filteredEtf.length})
          </h3>
          <CandidateTable
            rows={[...filteredEtf].sort((a, b) => b.pre_momentum_score - a.pre_momentum_score)}
            onSelect={setSelectedTicker}
          />
        </div>
        <div>
          <h3 className="text-[14px] font-semibold text-[#0A7D3F] uppercase tracking-wide mb-2">
            Stock Candidates ({filteredStock.length})
          </h3>
          <CandidateTable
            rows={[...filteredStock].sort((a, b) => b.pre_momentum_score - a.pre_momentum_score)}
            onSelect={setSelectedTicker}
          />
        </div>
      </div>

      {/* ── Conversion Tracking (1M backtest) ── */}
      {data.conversion && (
        <Section title="Conversion Tracking — 1M Backtest" defaultOpen badge={`Hit Rate: ${data.conversion.stats.hit_rate}%`}>
          <ConversionTracking conversion={data.conversion} />
        </Section>
      )}

      {/* ── Selected Candidate Detail ── */}
      {selectedCandidate && (
        <Section
          title={`${selectedCandidate.ticker} — ${selectedCandidate.name}`}
          defaultOpen
          badge={`Agreement ${((selectedCandidate.agreement_ratio || 0) * 100).toFixed(0)}%`}
        >
          <div className="flex justify-end mb-2">
            <button
              onClick={() => setSelectedTicker(null)}
              className="text-[12px] text-[#857F7A] hover:text-[#33302E] px-2 py-1 rounded border border-[#E6D9CE] hover:border-[#CCC1B7] transition-colors"
            >
              Close
            </button>
          </div>
          <CandidateDetail candidate={selectedCandidate} />
        </Section>
      )}

      {/* ── Agent Analysis (collapsed by default) ── */}
      <Section title="Agent Signal Breakdown" badge="Detail">
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <AgentRadarChart candidates={sortedCandidates} />
          <SubSignalContributionChart candidates={sortedCandidates} />
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          <AgreementDistributionChart distribution={summary.agent_agreement_distribution} />
          <TopSectorsChart sectors={summary.top_sectors} />
        </div>
      </Section>
    </div>
  );
}
