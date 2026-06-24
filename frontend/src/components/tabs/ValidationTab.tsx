import { useEffect, useState } from "react";
import axios from "axios";
import { C, CLASS_COLORS } from "../../styles/theme";
import { MetricCard } from "../shared/MetricCard";

// ───── Types ─────

interface CheckResult {
  metric: string;
  label: string;
  expected: any;
  actual: number | null;
  verdict: "PASS_IDEAL" | "PASS" | "FAIL" | "INSUFFICIENT_DATA";
}

interface ClassResult {
  classification: string;
  actuals: Record<string, any>;
  checks: CheckResult[];
  pass_score: number;
  ideal_count: number;
  pass_count: number;
  total_checks: number;
  overall: "PASS" | "PARTIAL" | "FAIL";
}

interface ValidationData {
  summary: {
    total_observations: number;
    eval_points: number;
    date_range: [string, string];
  };
  momentum: ClassResult[];
  pre_momentum: ClassResult[];
}

const fetchValidation = () => axios.get("/api/validation").then((r) => r.data);

// ───── Helpers ─────

function verdictColor(v: string): string {
  if (v === "PASS_IDEAL") return C.green;
  if (v === "PASS") return "#0A7D3F";
  if (v === "FAIL") return C.red;
  return C.gray;
}

function verdictLabel(v: string): string {
  if (v === "PASS_IDEAL") return "✓ IDEAL";
  if (v === "PASS") return "✓ PASS";
  if (v === "FAIL") return "✗ FAIL";
  return "— N/A";
}

function overallColor(v: string): string {
  if (v === "PASS") return C.green;
  if (v === "PARTIAL") return C.yellow;
  return C.red;
}

// ───── Components ─────

function ClassificationCard({ result, kind }: { result: ClassResult; kind: "momentum" | "pre-momentum" }) {
  const cls = result.classification;
  const clsColor = CLASS_COLORS[cls] || C.gray;
  const a = result.actuals;

  return (
    <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-[#FBEEE3] border-b border-[#E6D9CE]">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="text-[18px]" style={{ color: clsColor }}>{cls}</span>
            <span className="text-[12px] text-[#857F7A]">n={a.n ?? 0} obs</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-[12px] text-[#857F7A]">
              {result.pass_count}/{result.total_checks} pass · {result.ideal_count} ideal
            </span>
            <span
              className="text-[13px] px-2 py-0.5 rounded font-bold"
              style={{
                backgroundColor: overallColor(result.overall) + "22",
                color: overallColor(result.overall),
              }}
            >
              {result.overall} {result.pass_score}%
            </span>
          </div>
        </div>
      </div>

      {/* Hypothesis Checks */}
      <div className="p-3 space-y-1.5">
        {result.checks.map((chk, i) => (
          <div key={i} className="flex items-center gap-3 text-[13px]">
            <span
              className="font-mono font-bold w-16 shrink-0 text-center px-1.5 py-0.5 rounded"
              style={{
                backgroundColor: verdictColor(chk.verdict) + "22",
                color: verdictColor(chk.verdict),
              }}
            >
              {verdictLabel(chk.verdict)}
            </span>
            <span className="text-[#66605C] flex-1 truncate">{chk.label}</span>
            <span className="font-mono text-[#33302E] w-16 text-right shrink-0">
              {chk.actual === null ? "—" : typeof chk.actual === "number" ? chk.actual.toFixed(2) : chk.actual}
            </span>
          </div>
        ))}
      </div>

      {/* Detail metrics */}
      <div className="px-4 py-2 bg-[#FBEEE3] border-t border-[#E6D9CE]">
        {kind === "momentum" ? (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[12px]">
            <div>
              <span className="text-[#857F7A]">1M ret:</span>
              <span className="text-[#33302E] ml-1 font-mono">{a.ret_1m_mean ?? "-"}%</span>
            </div>
            <div>
              <span className="text-[#857F7A]">1M hit:</span>
              <span className="text-[#33302E] ml-1 font-mono">{a.ret_1m_hit ?? "-"}%</span>
            </div>
            <div>
              <span className="text-[#857F7A]">3M ret:</span>
              <span className="text-[#33302E] ml-1 font-mono">{a.ret_3m_mean ?? "-"}%</span>
            </div>
            <div>
              <span className="text-[#857F7A]">Persistence:</span>
              <span className="text-[#33302E] ml-1 font-mono">{a.persistence_1m ?? "-"}%</span>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[12px]">
            <div>
              <span className="text-[#857F7A]">3M Conv:</span>
              <span className="text-[#0A7D3F] ml-1 font-mono">{a.conv_3m ?? "-"}%</span>
            </div>
            <div>
              <span className="text-[#857F7A]">Failed:</span>
              <span className="text-[#CC0000] ml-1 font-mono">{a.fail_rate ?? "-"}%</span>
            </div>
            <div>
              <span className="text-[#857F7A]">Median:</span>
              <span className="text-[#33302E] ml-1 font-mono">{a.median_time_days ?? "-"}d</span>
            </div>
            <div>
              <span className="text-[#857F7A]">Graduated:</span>
              <span className="text-[#33302E] ml-1 font-mono">{a.graduated ?? "-"}</span>
            </div>
          </div>
        )}

        {/* Transitions for momentum */}
        {kind === "momentum" && a.transitions && Object.keys(a.transitions).length > 0 && (
          <div className="mt-2 pt-2 border-t border-[#E6D9CE]">
            <span className="text-[12px] text-[#857F7A] mr-2">Top transitions:</span>
            {Object.entries(a.transitions).slice(0, 3).map(([nextCls, pct]: any) => (
              <span key={nextCls} className="text-[12px] mr-2" style={{ color: CLASS_COLORS[nextCls] || C.gray }}>
                {nextCls.replace(/^.\s/, "")} {pct}%
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ───── Main Tab ─────

export function ValidationTab() {
  const [data, setData] = useState<ValidationData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetchValidation().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-[#857F7A] p-8">Loading validation...</div>;
  if (!data) return <div className="text-[#857F7A] p-8">No data</div>;

  const { summary, momentum, pre_momentum } = data;

  // Aggregate stats
  const totalChecksMom = momentum.reduce((s, r) => s + r.total_checks, 0);
  const totalPassMom = momentum.reduce((s, r) => s + r.pass_count, 0);
  const totalIdealMom = momentum.reduce((s, r) => s + r.ideal_count, 0);

  const totalChecksPm = pre_momentum.reduce((s, r) => s + r.total_checks, 0);
  const totalPassPm = pre_momentum.reduce((s, r) => s + r.pass_count, 0);
  const totalIdealPm = pre_momentum.reduce((s, r) => s + r.ideal_count, 0);

  const passRateMom = totalChecksMom > 0 ? (totalPassMom / totalChecksMom * 100).toFixed(0) : "0";
  const passRatePm = totalChecksPm > 0 ? (totalPassPm / totalChecksPm * 100).toFixed(0) : "0";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-[20px] font-bold text-[#33302E]">Classification Validation — Hypothesis vs Actual</h2>
        <p className="text-[14px] text-[#857F7A] mt-1">
          Pre-Momentum 6개 + Momentum 4개 분류의 가설 vs 실제 데이터 검증 결과.
          24개 평가 시점 × 770종목 = {summary.total_observations.toLocaleString()} 관측치 기반.
        </p>
        <p className="text-[12px] text-[#857F7A] mt-1">
          기간: {summary.date_range[0]} ~ {summary.date_range[1]} ({summary.eval_points} eval points, ~2주 간격)
        </p>
      </div>

      {/* Classification Logic 개선 적용 안내 */}
      <div className="border border-[#9CC3D5]/50 bg-[#E3EEF5]/20 rounded-lg p-3 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-[12px] px-2 py-0.5 rounded bg-[#E3EEF5]/50 text-[#0D7680] font-bold">PHASE 1 + B APPLIED</span>
          <span className="text-[14px] font-semibold text-[#0D7680]">Vol-Adjusted Buffer + Universal Hysteresis</span>
        </div>

        <div className="text-[13px] text-[#66605C] leading-relaxed">
          <div className="text-[#33302E] font-semibold mb-1">Phase 1 — Volatility-Adjusted Buffer</div>
          기존 hard rule(short ±0.5%, long ±1%)을 자산별 변동성 비례 buffer로 변경.
          공식: <code className="text-[#0F5499] text-[12px]">short = max(0.3, 0.4 × σ_daily × √20)</code>,
          <code className="text-[#0F5499] text-[12px] ml-2">long = max(0.7, 0.6 × σ_daily × √60)</code>
        </div>

        <div className="text-[13px] text-[#66605C] leading-relaxed">
          <div className="text-[#33302E] font-semibold mb-1">Option B — Universal Hysteresis</div>
          모든 임계값에 enter(strict) / exit(loose) 분리하여 일별 임계점 진동 방지.
          이전 분류가 UP이면 exit 임계값(40%)으로 유지 — 추세 안정성 ↑.
          OVEREXTENDED override 동일: enter OER ≥ 60, exit OER ≥ 50.
          예상 효과: 일별 분류 진동 -37%, high-churn 종목 -79%.
        </div>

        <div className="text-[13px] text-[#B85C00]">
          ⚠ 새 로직은 <strong>다음 Run Live Scan부터</strong> 분류 + ve_observations에 반영됩니다.
          현재 표시된 검증 결과는 이전 로직 기준입니다.
        </div>
      </div>

      {/* Summary KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <MetricCard
          label="Momentum Pass Rate"
          value={`${passRateMom}%`}
          sub={`${totalPassMom}/${totalChecksMom} checks`}
        />
        <MetricCard
          label="Momentum Ideal"
          value={totalIdealMom}
          sub="이상치 충족"
        />
        <MetricCard
          label="Pre-Momentum Pass Rate"
          value={`${passRatePm}%`}
          sub={`${totalPassPm}/${totalChecksPm} checks`}
        />
        <MetricCard
          label="Pre-Momentum Ideal"
          value={totalIdealPm}
          sub="이상치 충족"
        />
        <MetricCard
          label="Total Observations"
          value={summary.total_observations.toLocaleString()}
          sub={`${summary.eval_points} eval points`}
        />
      </div>

      {/* Verdict Legend */}
      <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-3 text-[13px]">
        <div className="flex flex-wrap gap-3 items-center">
          <span className="text-[#857F7A]">Verdict:</span>
          <span><span style={{ color: C.green }}>✓ IDEAL</span> — 이상 기준 충족</span>
          <span><span style={{ color: "#0A7D3F" }}>✓ PASS</span> — 최소 기준 충족</span>
          <span><span style={{ color: C.red }}>✗ FAIL</span> — 기준 미달</span>
          <span><span style={{ color: C.gray }}>— N/A</span> — 데이터 부족 (n &lt; 5)</span>
          <span className="text-[#857F7A]">|</span>
          <span className="text-[#857F7A]">Overall:</span>
          <span><span style={{ color: C.green }}>PASS</span> ≥ 75% 통과</span>
          <span><span style={{ color: C.yellow }}>PARTIAL</span> 50-75%</span>
          <span><span style={{ color: C.red }}>FAIL</span> &lt; 50%</span>
        </div>
      </div>

      {/* Momentum Section */}
      <div className="space-y-3">
        <div className="border-l-2 border-[#0F5499]/40 pl-4">
          <h3 className="text-[16px] font-bold text-[#0F5499] uppercase tracking-wide">
            Part A — Momentum Classification ({momentum.length})
          </h3>
          <p className="text-[14px] text-[#857F7A] mt-1">
            가설: forward return 양(+), 벤치마크 초과, 큰 drawdown 없음, 분류 지속성.
          </p>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
          {momentum.map((r) => (
            <ClassificationCard key={r.classification} result={r} kind="momentum" />
          ))}
        </div>
      </div>

      {/* Pre-Momentum Section */}
      <div className="space-y-3">
        <div className="border-l-2 border-[#C9B8DC]/40 pl-4 mt-6">
          <h3 className="text-[16px] font-bold text-[#7D5BA6] uppercase tracking-wide">
            Part B — Pre-Momentum Classification ({pre_momentum.length})
          </h3>
          <p className="text-[14px] text-[#857F7A] mt-1">
            가설: 모멘텀 전환율, 전환 소요시간, 실패율 (DOWNTREND/CYCLE_PEAK 등으로 빠질 비율).
          </p>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
          {pre_momentum.map((r) => (
            <ClassificationCard key={r.classification} result={r} kind="pre-momentum" />
          ))}
        </div>
      </div>

      {/* Methodology footer */}
      <div className="bg-[#FFFFFF] border border-[#E6D9CE] rounded-lg p-4 mt-6">
        <h3 className="text-[16px] font-semibold text-[#33302E] mb-2">Methodology Reference</h3>
        <p className="text-[13px] text-[#857F7A] leading-relaxed">
          상세 검증 절차는 <strong className="text-[#33302E]">Appendix → Efficacy</strong> 서브탭 참고.
          각 분류의 가설은 <code className="text-[#0F5499] text-[12px]">_MOMENTUM_HYPOTHESES</code> /
          <code className="text-[#0F5499] text-[12px] ml-1">_PRE_MOMENTUM_HYPOTHESES</code>
          (api.py)에 정의되어 있으며, 향후 데이터 누적에 따라 임계값 재조정 가능합니다.
        </p>
      </div>
    </div>
  );
}
