import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { fetchEffectiveness } from "../../api/client";
import { MetricCard } from "../shared/MetricCard";
import { DataTable } from "../shared/DataTable";
import { ColDefToggle } from "../shared/ColDefToggle";
import { C, DARK_LAYOUT } from "../../styles/theme";

export function EffectivenessTab() {
  const [raw, setRaw] = useState<any>(null);
  const [period, setPeriod] = useState("1M");
  useEffect(() => { fetchEffectiveness().then(setRaw); }, []);

  if (!raw) return <div className="text-[#857F7A] p-8">Loading...</div>;

  const periods: string[] = raw.periods || [];
  const data = raw.per_period?.[period];

  if (!data || !periods.length) return <div className="text-[#857F7A] p-8">No effectiveness data available.</div>;

  const { kpis, ic_timeseries, quintiles, classification_summary, scatter, box_data, regression } = data;

  return (
    <div className="space-y-6">

      {/* Period Selector */}
      <div className="flex items-center gap-2">
        <span className="text-[16px] text-[#66605C]">Forward Period:</span>
        {periods.map((p: string) => (
          <button key={p}
            className={`px-4 py-1.5 rounded text-[16px] font-medium transition-colors ${
              period === p ? "bg-[#0F5499] text-white" : "bg-[#F2E5D7] text-[#66605C] hover:text-[#33302E]"
            }`}
            onClick={() => setPeriod(p)}>
            {p}
          </button>
        ))}
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-5 gap-3">
        <MetricCard label={`Spearman IC (${period})`} value={kpis.ic_spearman} sub={`p=${kpis.ic_pval}`} />
        <MetricCard label="Excess Hit Rate" value={`${kpis.overall_hit_rate}%`} />
        <MetricCard label="Avg Excess Return" value={`${kpis.avg_excess}%`} />
        <MetricCard label="Observations" value={kpis.n_observations} />
        <MetricCard label="R²" value={regression.r_squared} sub={`slope=${regression.slope}`} />
      </div>

      {/* IC Time Series */}
      <div className="mt-12">
        <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide mb-3">
          Information Coefficient Over Time ({period} Forward)
        </h3>
        {ic_timeseries.length > 0 && (
          <div>
            <Plot
              data={[{
                type: "bar",
                x: ic_timeseries.map((d: any) => d.date),
                y: ic_timeseries.map((d: any) => d.IC),
                marker: { color: ic_timeseries.map((d: any) => d.IC > 0 ? C.green : C.red) },
              }]}
              layout={{
                ...DARK_LAYOUT, height: 300, margin: { t: 10, b: 50, l: 50, r: 20 },
                xaxis: { title: "Eval Date", color: C.gray }, yaxis: { title: "IC", color: C.gray },
                shapes: [{ type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray, dash: "dash" } }],
              }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
            <p className="text-[13px] text-[#857F7A] mt-3 px-2 leading-relaxed">
              각 평��� 시점에서 Composite Score와 {period} forward return 간 Spearman 순위상관계수(IC).
              양수(초록) = 점수가 높을수록 수익률 높음. 음수(빨강) = 역상관. 일관되게 양수이면 시그널 예측력 양호.
            </p>
          </div>
        )}
      </div>

      {/* Quintile Analysis */}
      <div className="mt-12">
        <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide mb-3">
          Quintile Analysis ({period})
        </h3>
        <div className="grid grid-cols-2 gap-8">
          <div>
            <ColDefToggle defs={[
              { col: "Quintile", desc: "Composite Score를 5분위로 나눈 그룹. Q1=최하위, Q5=최상위" },
              { col: "N", desc: "해당 분위 관측치 수" },
              { col: "Avg Score", desc: "분위 평균 Composite Score" },
              { col: "Avg Fwd%", desc: "평균 forward return (절대 수익률)" },
              { col: "Avg Exc%", desc: "평균 excess return (벤치마크 대비 초과수익)" },
              { col: "Hit Rate%", desc: "excess return > 0 비율. 50% 초과 시 알파 존재" },
            ]} />
            <DataTable data={quintiles} columns={[
              { accessorKey: "quintile", header: "Quintile" },
              { accessorKey: "n", header: "N" },
              { accessorKey: "avg_score", header: "Avg Score" },
              { accessorKey: "avg_fwd", header: `Avg Fwd%` },
              { accessorKey: "avg_exc", header: `Avg Exc%` },
              { accessorKey: "hit_rate", header: "Hit Rate%" },
            ]} maxHeight="250px" />
            <p className="text-[13px] text-[#857F7A] mt-3 px-2 leading-relaxed">
              Composite Score를 5분위로 나누어 각 분위의 {period} forward return 성과 비교.
              Q5(고점수)의 수익률이 Q1(저점수)보다 높으면 스코어링 시스템이 유효.
            </p>
          </div>
          <div>
            <Plot
              data={[
                { type: "bar", name: `Avg Fwd%`, x: quintiles.map((d: any) => d.quintile), y: quintiles.map((d: any) => d.avg_fwd), marker: { color: C.blue } },
                { type: "bar", name: `Avg Exc%`, x: quintiles.map((d: any) => d.quintile), y: quintiles.map((d: any) => d.avg_exc), marker: { color: C.purple } },
              ]}
              layout={{
                ...DARK_LAYOUT, barmode: "group", height: 300,
                margin: { t: 10, b: 50, l: 50, r: 20 },
                shapes: [{ type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray, dash: "dash" } }],
              }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
            <p className="text-[13px] text-[#857F7A] mt-3 px-2 leading-relaxed">
              Q1→Q5�� 갈수록 수익률이 단조증가하면 시그널의 모노토닉(단조성) 충족.
              Q5-Q1 스프레드가 클수록 변별력 높음.
            </p>
          </div>
        </div>
      </div>

      {/* Classification Box Plot */}
      <div className="mt-12">
        <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide mb-3">
          Excess Return by Classification ({period})
        </h3>
        {box_data.length > 0 && (() => {
          const classes: string[] = [...new Set(box_data.map((d: any) => d.cls_short))] as string[];
          return (
            <div>
              <Plot
                data={classes.map((cls: string) => ({
                  type: "box" as const, name: cls,
                  y: box_data.filter((d: any) => d.cls_short === cls).map((d: any) => d.excess_return),
                  boxpoints: false,
                }))}
                layout={{
                  ...DARK_LAYOUT, height: 400, margin: { t: 10, b: 50, l: 50, r: 20 },
                  yaxis: { title: `Excess Return % (${period})`, color: C.gray }, showlegend: false,
                  shapes: [{ type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray, dash: "dash" } }],
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
              <p className="text-[13px] text-[#857F7A] mt-3 px-2 leading-relaxed">
                각 분류(Classification)의 {period} excess return 분포(박스플롯). 0선 위 = 벤치마크 초과수익.
                CONTINUATION의 중앙값이 양수이면 해당 시그널이 알파 생성에 유효. DOWNTREND가 음수면 리스크 필터 유효.
              </p>
            </div>
          );
        })()}
      </div>

      {/* Classification Summary Table */}
      <div className="mt-12">
        <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide mb-3">
          Classification Summary ({period})
        </h3>
        <ColDefToggle defs={[
          { col: "Class", desc: "Dual-timeframe 분류 (약칭). CONT=Continuation, FORM=Formation 등" },
          { col: "N", desc: "해당 분류 관측치 수" },
          { col: "Avg Score", desc: "분류 평균 Composite Score" },
          { col: "Fwd%", desc: "평균 forward return (절대)" },
          { col: "Exc%", desc: "평균 excess return (벤치마크 대비)" },
          { col: "Hit Rate%", desc: "excess return > 0 비율" },
        ]} />
        <DataTable data={classification_summary} columns={[
          { accessorKey: "cls_short", header: "Class" },
          { accessorKey: "n", header: "N" },
          { accessorKey: "avg_score", header: "Avg Score" },
          { accessorKey: "avg_fwd", header: `Fwd%` },
          { accessorKey: "avg_exc", header: `Exc%` },
          { accessorKey: "hit_rate", header: "Hit Rate%" },
        ]} maxHeight="350px" />
        <p className="text-[13px] text-[#857F7A] mt-3 px-2 leading-relaxed">
          분류별 {period} forward 성�� 요약. Hit Rate = excess return &gt; 0 비율. 50% 이상이면 벤치마크 대비 우위.
        </p>
      </div>

      {/* Score vs Excess Return Scatter */}
      <div className="mt-12">
        <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide mb-3">
          Composite Score vs Excess Return ({period})
        </h3>
        {scatter.length > 0 && (
          <div>
            <Plot
              data={[{
                type: "scatter", mode: "markers",
                x: scatter.map((d: any) => d.score),
                y: scatter.map((d: any) => d.excess_return),
                text: scatter.map((d: any) => `${d.ticker} ${d.eval_date}`),
                marker: { size: 3, opacity: 0.35, color: C.cyan },
                hovertemplate: "%{text}<br>Score=%{x:.1f}<br>ExcRet=%{y:.1f}%<extra></extra>",
              }, {
                type: "scatter", mode: "lines", name: "Regression",
                x: [0, 100], y: [regression.intercept, regression.intercept + regression.slope * 100],
                line: { color: C.orange, dash: "dash" },
              }]}
              layout={{
                ...DARK_LAYOUT, height: 420, margin: { t: 10, b: 50, l: 50, r: 20 },
                xaxis: { title: "Composite Score", color: C.gray },
                yaxis: { title: `Excess Return % (${period})`, color: C.gray },
                shapes: [
                  { type: "line", x0: 55, x1: 55, y0: 0, y1: 1, yref: "paper", line: { color: C.orange, dash: "dot" } },
                  { type: "line", y0: 0, y1: 0, x0: 0, x1: 1, xref: "paper", line: { color: C.gray, dash: "dash" } },
                ],
                showlegend: false,
              }}
              config={{ responsive: true }} style={{ width: "100%" }}
            />
            <p className="text-[13px] text-[#857F7A] mt-3 px-2 leading-relaxed">
              각 관측(시점×종���)의 Composite Score(X)와 {period} excess return(Y) 산점도.
              주황 회귀선의 기울기(slope={regression.slope})가 양수면 고점수 = 고수익 관계 성립.
              R²={regression.r_squared}는 설명력. 점선(55) = 적격 기준.
            </p>
          </div>
        )}
      </div>

      {/* Multi-Period Comparison (only show when period buttons exist) */}
      {periods.length > 1 && (() => {
        const compRows: any[] = [];
        for (const p of periods) {
          const pd = raw.per_period?.[p];
          if (!pd) continue;
          compRows.push({
            period: p,
            ic: pd.kpis.ic_spearman,
            hit_rate: pd.kpis.overall_hit_rate,
            avg_exc: pd.kpis.avg_excess,
            n: pd.kpis.n_observations,
            r_sq: pd.regression.r_squared,
          });
        }
        if (compRows.length < 2) return null;
        return (
          <div className="mt-12">
            <h3 className="text-[16px] font-semibold text-[#66605C] uppercase tracking-wide mb-3">
              Multi-Period Comparison (1W vs 1M vs 3M)
            </h3>
            <div className="grid grid-cols-2 gap-8">
              <div>
                <DataTable data={compRows} columns={[
                  { accessorKey: "period", header: "Period" },
                  { accessorKey: "n", header: "N" },
                  { accessorKey: "ic", header: "IC" },
                  { accessorKey: "hit_rate", header: "Hit Rate%" },
                  { accessorKey: "avg_exc", header: "Avg Exc%" },
                  { accessorKey: "r_sq", header: "R²" },
                ]} maxHeight="200px" />
                <p className="text-[13px] text-[#857F7A] mt-3 px-2 leading-relaxed">
                  1주/1개월/3개월 forward 기간별 시그널 유효성 비교.
                  기간이 길수록 IC와 R²가 높으면 중장��� 시그널로서 유효, 짧은 기간이 높으면 단기 트레이딩 시그널.
                </p>
              </div>
              <Plot
                data={[
                  { type: "bar", name: "IC", x: compRows.map(r => r.period), y: compRows.map(r => r.ic), marker: { color: C.cyan } },
                  { type: "bar", name: "Hit Rate%", x: compRows.map(r => r.period), y: compRows.map(r => r.hit_rate), marker: { color: C.green }, yaxis: "y2" },
                ]}
                layout={{
                  ...DARK_LAYOUT, barmode: "group", height: 280,
                  margin: { t: 10, b: 40, l: 50, r: 50 },
                  yaxis: { title: "IC", color: C.gray },
                  yaxis2: { title: "Hit Rate %", color: C.gray, overlaying: "y", side: "right" },
                  legend: { x: 0, y: 1, font: { size: 10 } },
                }}
                config={{ responsive: true }} style={{ width: "100%" }}
              />
            </div>
          </div>
        );
      })()}
    </div>
  );
}
