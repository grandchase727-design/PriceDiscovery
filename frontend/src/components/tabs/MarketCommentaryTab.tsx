import { useEffect, useState, type ReactNode } from "react";
import { fetchClassificationHistory, type FilterParams } from "../../api/client";
import { C } from "../../styles/theme";
import { SwarmAnalysis } from "../shared/SwarmAnalysis";
import { BacktestPanel } from "../shared/BacktestPanel";
import FinalListPanel from "../shared/FinalListPanel";

interface ClassHistoryData {
  dates: string[];
  classifications: string[];
  matrix: number[][];
}

// ─────────────────────────────────────────────────────────────────
// Collapsible section — hidden by default, reveals analysis on button click.
// Keeps Market Commentary clean: Swarm + Final List always visible,
// everything else collapsed behind a toggle button.
// ─────────────────────────────────────────────────────────────────
function CollapsibleSection({ title, subtitle, accent, defaultOpen = false, children }: {
  title: string; subtitle?: string; accent?: string; defaultOpen?: boolean; children: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const color = accent || C.blue;
  return (
    <div className="rounded-lg border" style={{ borderColor: C.border, backgroundColor: C.panel }}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 text-left transition-colors"
        style={{ cursor: "pointer" }}>
        <div className="flex items-center gap-2">
          <span className="text-[15px] font-bold" style={{ color }}>{title}</span>
          {subtitle && <span className="text-[12px]" style={{ color: C.gray }}>{subtitle}</span>}
        </div>
        <span className="flex items-center gap-2 text-[12px]" style={{ color }}>
          {open ? "숨기기" : "분석 보기"}
          <span style={{ fontSize: 12 }}>{open ? "▼" : "▶"}</span>
        </span>
      </button>
      {open && <div className="px-4 pb-4 pt-1 border-t" style={{ borderColor: C.border }}>{children}</div>}
    </div>
  );
}

export function MarketCommentaryTab({ dataVersion = 0, scanning = false }: {
  filters?: FilterParams; dataVersion?: number; scanning?: boolean;
}) {
  const [classHistory, setClassHistory] = useState<ClassHistoryData | null>(null);
  const [loading, setLoading] = useState(true);

  // Only fetch classification history for the "As of" header date.
  // SwarmAnalysis / FinalListPanel / BacktestPanel each fetch their own data.
  useEffect(() => {
    setLoading(true);
    fetchClassificationHistory()
      .then(setClassHistory)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="text-[#857F7A] text-[16px]">Loading market commentary…</div>;
  }

  const asOf = classHistory?.dates?.[classHistory.dates.length - 1] ?? "-";

  return (
    <div className="space-y-4">
      <div className="bg-[#FFF1E5] rounded-lg border border-[#9CC3D5]/40 p-5">
        <div className="flex items-baseline justify-between mb-3">
          <div>
            <h2 className="text-[18px] font-bold text-[#0F5499]">Market Commentary — Comprehensive Analytical Report</h2>
            <div className="text-[13px] text-[#857F7A] mt-0.5">
              6-Agent Swarm 분석 + 매수 Final List(통합) + PM Agent Backtest
            </div>
          </div>
          <div className="text-right">
            <div className="text-[12px] text-[#857F7A] uppercase tracking-wide">As of</div>
            <div className="text-[16px] font-mono font-bold text-[#0D7680] tabular-nums">{asOf}</div>
          </div>
        </div>

        {/* ── Market Leaders 6-Agent Swarm Analysis (always visible — first item) ── */}
        <SwarmAnalysis />
      </div>

      {/* ── 매수 Final List (통합) — placed right after Swarm Analysis (always visible) ── */}
      <FinalListPanel dataVersion={dataVersion} scanning={scanning} />

      {/* ── Rest: collapsed by default, reveal on button click ── */}
      <CollapsibleSection
        title="📊 PM Agent Backtest"
        subtitle="개별 종목 선정 skill 평가 (forward-return / alpha / IC)"
        accent={C.teal}>
        <BacktestPanel />
      </CollapsibleSection>
    </div>
  );
}
