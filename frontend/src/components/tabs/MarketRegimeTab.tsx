import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { fetchMarketRegime, type FilterParams } from "../../api/client";
import { ColDefToggle } from "../shared/ColDefToggle";
import { C, DARK_LAYOUT } from "../../styles/theme";

/* ─── Color helpers ─── */
const regimeColor: Record<string, string> = {
  "RISK-ON": C.green, "RISK-OFF": C.red, "ROTATION": C.orange,
  "TRANSITION": C.blue, "COMPRESSION": C.purple,
  "MILD-BULL": "#86efac", "MILD-BEAR": "#fca5a5", "NEUTRAL": C.gray,
};

function pctBar(pct: number, color: string, maxW = 200) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-2 rounded" style={{ width: Math.max(2, pct / 100 * maxW), background: color }} />
      <span className="text-[10px] text-gray-400">{pct.toFixed(1)}%</span>
    </div>
  );
}

function MetricBox({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="bg-[#111827] border border-gray-800 rounded-lg p-3 min-w-[120px]">
      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-lg font-bold ${color || "text-gray-200"}`}>{value}</div>
      {sub && <div className="text-[10px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

const REGIME_DEFS: { regime: string; color: string; condition: string; desc: string }[] = [
  {
    regime: "RISK-ON",
    color: C.green,
    condition: "Bullish classification >= 55% AND Avg Composite >= 55 AND Avg Long# >= 3.5",
    desc: "광범위한 상승 추세. 대부분의 종목이 bullish classification을 보이고, composite 점수가 높으며, 다수의 전략이 동시에 Long 시그널을 발생. 적극적 매수 환경.",
  },
  {
    regime: "MILD-BULL",
    color: "#86efac",
    condition: "Bullish >= 40% AND Bearish <= 20% (RISK-ON 미충족)",
    desc: "완만한 상승 기조. Bullish 종목이 우세하지만 강한 추세 합의에는 미달. 선별적 매수 유효, 추세 강화 여부 모니터링 필요.",
  },
  {
    regime: "RISK-OFF",
    color: C.red,
    condition: "Bearish classification >= 35% AND Avg Composite <= 45 AND Avg Short# >= 2.5",
    desc: "광범위한 하락 압력. Bearish classification 우세, 낮은 composite, Short 시그널 확산. 방어적 포지셔닝 또는 현금 비중 확대 권장.",
  },
  {
    regime: "MILD-BEAR",
    color: "#fca5a5",
    condition: "Bearish >= 25% AND Bullish <= 40% (RISK-OFF 미충족)",
    desc: "완만한 하락 또는 조정 기조. Bearish 종목이 증가 중이나 아직 광범위하지 않음. 리스크 관리 강화, 약세 전환 가능성 경계.",
  },
  {
    regime: "ROTATION",
    color: C.orange,
    condition: "Composite Std >= 18 AND |Bull% - Bear%| <= 20",
    desc: "섹터/종목 간 극단적 양극화. 일부는 강한 상승, 일부는 급락. Bull/Bear가 혼재하며 composite 분산이 높음. 승자/패자 간 격차가 크므로 종목 선별이 핵심.",
  },
  {
    regime: "TRANSITION",
    color: C.blue,
    condition: "Avg TFS >= 45 AND Bull% < 55% AND Bear% < 35%",
    desc: "Regime 전환 초기 신호. Trend Formation Score(TFS)가 높아 새로운 추세가 형성 중이나 아직 확립되지 않음. 돌파/형성 시그널 주시, 방향 확인 후 포지션 진입.",
  },
  {
    regime: "COMPRESSION",
    color: C.purple,
    condition: "Composite Std <= 12 AND Strategy Agreement <= 30%",
    desc: "변동성 수축 및 시그널 응축. 전략 간 합의가 극히 낮고 점수가 밀집. 큰 방향성 움직임 직전의 에너지 축적 상태. 돌파 방향에 대비한 양방향 시나리오 준비.",
  },
  {
    regime: "NEUTRAL",
    color: C.gray,
    condition: "상기 조건 모두 미충족",
    desc: "뚜렷한 방향성 없는 혼조세. 특정 regime 특성이 관찰되지 않음. 관망 또는 소규모 포지션 유지, regime 확립 시까지 대기.",
  },
];

export function MarketRegimeTab({ filters }: { filters: FilterParams }) {
  const [data, setData] = useState<any>(null);
  const [showDefs, setShowDefs] = useState(false);
  useEffect(() => { fetchMarketRegime(filters).then(setData); }, [filters]);

  if (!data || data.error) return <div className="text-gray-500 p-8">Loading...</div>;

  const { regime, regime_desc, breadth: b, strategy_breadth, strategy_groups,
          agreement_score, classification_dist, signal_dist,
          composite_distribution: cd, sector_regime, regime_history } = data;

  const rc = regimeColor[regime] || C.gray;

  return (
    <div className="space-y-6">

      {/* ═══ Regime Definitions Toggle ═══ */}
      <div className="bg-[#111827] border border-gray-800 rounded-lg">
        <button
          className="w-full flex items-center justify-between px-4 py-3 text-left"
          onClick={() => setShowDefs(!showDefs)}
        >
          <span className="text-sm font-semibold text-gray-400 uppercase tracking-wide">Market Regime Definitions</span>
          <span className="text-gray-500 text-xs">{showDefs ? "▲ 접기" : "▼ 펼치기"}</span>
        </button>
        {showDefs && (
          <div className="px-4 pb-4 space-y-0">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="py-2 px-2 text-left text-gray-500 w-[120px]">Regime</th>
                  <th className="py-2 px-2 text-left text-gray-500 w-[380px]">판정 조건</th>
                  <th className="py-2 px-2 text-left text-gray-500">설명</th>
                </tr>
              </thead>
              <tbody>
                {REGIME_DEFS.map((rd) => (
                  <tr key={rd.regime} className={`border-b border-gray-800/50 ${regime === rd.regime ? "bg-[#1f2937]/60" : ""}`}>
                    <td className="py-2 px-2">
                      <div className="flex items-center gap-2">
                        <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: rd.color }} />
                        <span className="font-bold" style={{ color: rd.color }}>{rd.regime}</span>
                        {regime === rd.regime && <span className="text-[9px] text-cyan-400 font-semibold ml-1">CURRENT</span>}
                      </div>
                    </td>
                    <td className="py-2 px-2 text-gray-500 font-mono text-[10px]">{rd.condition}</td>
                    <td className="py-2 px-2 text-gray-400">{rd.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ═══ 1. Regime Indicator ═══ */}
      <div className="flex items-start gap-6">
        <div className="bg-[#111827] border-2 rounded-xl p-5 flex-shrink-0" style={{ borderColor: rc }}>
          <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-1">Current Market Regime</div>
          <div className="text-2xl font-black" style={{ color: rc }}>{regime}</div>
          <div className="text-xs text-gray-400 mt-2 max-w-[400px]">{regime_desc}</div>
          <div className="text-[10px] text-gray-600 mt-2">Strategy Agreement: <span className="text-gray-300 font-semibold">{agreement_score}%</span></div>
        </div>
        <div className="flex flex-wrap gap-3">
          <MetricBox label="Bull / Bear / Neutral" value={`${b.pct_bull}% / ${b.pct_bear}% / ${b.pct_neutral}%`}
            sub={`${b.n_bull} / ${b.n_bear} / ${b.n_neutral} of ${b.n}`} />
          <MetricBox label="Avg Composite" value={b.avg_composite.toFixed(1)} sub={`Median ${b.median_composite} | Std ${b.std_composite}`}
            color={b.avg_composite >= 55 ? "text-green-400" : b.avg_composite <= 45 ? "text-red-400" : "text-gray-200"} />
          <MetricBox label="Avg Long# / Short#" value={`${b.avg_long_count} / ${b.avg_short_count}`}
            sub={`Net ${(b.avg_long_count - b.avg_short_count).toFixed(2)}`}
            color={b.avg_long_count > b.avg_short_count ? "text-green-400" : "text-red-400"} />
          <MetricBox label="Eligible %" value={`${b.pct_eligible}%`} />
          <MetricBox label="Avg RSI" value={b.avg_rsi.toFixed(1)}
            color={b.avg_rsi >= 60 ? "text-yellow-400" : b.avg_rsi <= 40 ? "text-blue-400" : "text-gray-200"} />
          <MetricBox label="Score Axes" value={`T${b.avg_tcs} F${b.avg_tfs} R${b.avg_rss}`} sub={`OER ${b.avg_oer}`} />
        </div>
      </div>

      {/* ═══ 2. Strategy Breadth Matrix ═══ */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">Strategy Breadth — Long vs Short (% of universe with signal &ge; 50)</h3>
        <ColDefToggle defs={[
          { col: "Strategy", desc: "8개 hedge strategy 이름" },
          { col: "Long Breadth", desc: "유니버스 중 해당 전략 Long 점수 ≥ 50인 종목 비율 (%)" },
          { col: "Short Breadth", desc: "유니버스 중 해당 전략 Short 점수 ≥ 50인 종목 비율 (%)" },
          { col: "Net", desc: "Long Breadth − Short Breadth. 양수면 강세, 음수면 약세 우세" },
          { col: "Long Avg / Short Avg", desc: "전 종목 평균 Long/Short 점수 (0-100)" },
          { col: "Net Avg", desc: "Long Avg − Short Avg. 전략의 전체적 방향성 강도" },
        ]} />
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-gray-700 bg-[#111827]">
                <th className="py-2 px-3 text-left text-gray-500">Strategy</th>
                <th className="py-2 px-3 text-right text-gray-500">Long Breadth</th>
                <th className="py-2 px-3 text-left text-gray-500 w-[220px]"></th>
                <th className="py-2 px-3 text-right text-gray-500">Short Breadth</th>
                <th className="py-2 px-3 text-left text-gray-500 w-[220px]"></th>
                <th className="py-2 px-3 text-right text-gray-500">Net</th>
                <th className="py-2 px-3 text-right text-gray-500">Long Avg</th>
                <th className="py-2 px-3 text-right text-gray-500">Short Avg</th>
                <th className="py-2 px-3 text-right text-gray-500">Net Avg</th>
              </tr>
            </thead>
            <tbody>
              {strategy_breadth.map((sb: any) => (
                <tr key={sb.strategy} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
                  <td className="py-1.5 px-3 font-semibold text-gray-300">{sb.label}</td>
                  <td className="py-1.5 px-3 text-right text-green-400">{sb.long_breadth}%</td>
                  <td className="py-1.5 px-3">{pctBar(sb.long_breadth, C.green)}</td>
                  <td className="py-1.5 px-3 text-right text-red-400">{sb.short_breadth}%</td>
                  <td className="py-1.5 px-3">{pctBar(sb.short_breadth, C.red)}</td>
                  <td className={`py-1.5 px-3 text-right font-bold ${sb.net_breadth > 0 ? "text-green-400" : sb.net_breadth < 0 ? "text-red-400" : "text-gray-400"}`}>
                    {sb.net_breadth > 0 ? "+" : ""}{sb.net_breadth}%
                  </td>
                  <td className="py-1.5 px-3 text-right text-gray-400">{sb.long_avg}</td>
                  <td className="py-1.5 px-3 text-right text-gray-400">{sb.short_avg}</td>
                  <td className={`py-1.5 px-3 text-right ${sb.net_avg > 0 ? "text-green-400" : sb.net_avg < 0 ? "text-red-400" : "text-gray-400"}`}>
                    {sb.net_avg > 0 ? "+" : ""}{sb.net_avg}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══ 3. Strategy Group Analysis ═══ */}
      <div>
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">Strategy Group Divergence</h3>
        <div className="grid grid-cols-5 gap-3">
          {strategy_groups.map((g: any) => {
            const netColor = g.net_breadth > 10 ? C.green : g.net_breadth < -10 ? C.red : C.yellow;
            return (
              <div key={g.group} className="bg-[#111827] border border-gray-800 rounded-lg p-3">
                <div className="text-[11px] font-bold text-gray-300 mb-1">{g.group}</div>
                <div className="text-[10px] text-gray-600 mb-2">{g.desc}</div>
                <div className="text-lg font-black" style={{ color: netColor }}>
                  {g.net_breadth > 0 ? "+" : ""}{g.net_breadth}%
                </div>
                <div className="text-[10px] text-gray-500">Net Breadth</div>
                <div className="text-sm font-bold mt-1" style={{ color: netColor }}>
                  {g.net_avg > 0 ? "+" : ""}{g.net_avg}
                </div>
                <div className="text-[10px] text-gray-500">Net Avg Score</div>
                <div className="text-[9px] text-gray-600 mt-1">{g.strategies.join(", ")}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ═══ 4. Charts Row ═══ */}
      <div className="space-y-4">

        {/* Classification Distribution — full width for label legibility */}
        <div className="bg-[#111827] border border-gray-800 rounded-lg p-4">
          <Plot
            data={[{
              type: "bar",
              x: classification_dist.map((d: any) => d.classification.replace(/^.\s/, "")),
              y: classification_dist.map((d: any) => d.count),
              marker: {
                color: classification_dist.map((d: any) =>
                  d.group === "bullish" ? C.green : d.group === "bearish" ? C.red : C.yellow
                ),
              },
              text: classification_dist.map((d: any) => `${d.pct}%`),
              textposition: "outside" as const,
              textfont: { size: 11, color: "#9ca3af" },
            }]}
            layout={{ ...DARK_LAYOUT, title: { text: "Classification Distribution", font: { size: 13, color: "#9ca3af" } },
                      height: 420, xaxis: { tickangle: -45, tickfont: { size: 10 }, automargin: true }, yaxis: { title: "Count" },
                      margin: { t: 40, b: 160, l: 50, r: 20 }, bargap: 0.3 }}
            config={{ displayModeBar: false }}
            style={{ width: "100%" }}
          />
        </div>

        {/* Composite Histogram */}
        <div className="bg-[#111827] border border-gray-800 rounded-lg p-4">
          <Plot
            data={[{
              type: "bar",
              x: cd.bins,
              y: cd.counts,
              marker: { color: cd.bins.map((b: number) => b >= 55 ? C.green : b >= 40 ? C.yellow : C.red) },
              width: 4.5,
            }, {
              type: "scatter", mode: "lines",
              x: [cd.mean, cd.mean], y: [0, Math.max(...cd.counts) * 1.1],
              line: { color: C.cyan, width: 2, dash: "dash" as const },
              name: `Mean ${cd.mean}`,
            }, {
              type: "scatter", mode: "lines",
              x: [cd.median, cd.median], y: [0, Math.max(...cd.counts) * 1.1],
              line: { color: C.yellow, width: 2, dash: "dot" as const },
              name: `Median ${cd.median}`,
            }]}
            layout={{ ...DARK_LAYOUT, title: { text: `Composite Distribution (Skew: ${cd.skew})`, font: { size: 13, color: "#9ca3af" } },
                      height: 300, xaxis: { title: "Composite Score", range: [0, 100] }, yaxis: { title: "Count" },
                      showlegend: true, legend: { x: 0.7, y: 0.95, font: { size: 9 } },
                      margin: { t: 40, b: 50, l: 40, r: 10 } }}
            config={{ displayModeBar: false }}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      {/* ═══ 5. Net Signal Distribution ═══ */}
      {signal_dist.length > 0 && (
        <div className="bg-[#111827] border border-gray-800 rounded-lg p-4">
          <Plot
            data={[{
              type: "bar",
              x: signal_dist.map((d: any) => d.signal || "(none)"),
              y: signal_dist.map((d: any) => d.count),
              marker: { color: signal_dist.map((d: any) => {
                const s = d.signal;
                if (s === "STRONG_LONG") return C.green;
                if (s === "LONG") return "#86efac";
                if (s === "STRONG_SHORT") return C.red;
                if (s === "SHORT") return "#fca5a5";
                return C.gray;
              })},
              text: signal_dist.map((d: any) => d.count),
              textposition: "outside" as const,
            }]}
            layout={{ ...DARK_LAYOUT, title: { text: "Net Signal Distribution (Multi-Strategy Consensus)", font: { size: 13, color: "#9ca3af" } },
                      height: 250, margin: { t: 40, b: 40, l: 40, r: 10 } }}
            config={{ displayModeBar: false }}
            style={{ width: "100%" }}
          />
        </div>
      )}

      {/* ═══ 6. Sector Regime Heatmap ═══ */}
      {sector_regime.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">Sector Regime Analysis</h3>
          <ColDefToggle defs={[
            { col: "Sector", desc: "글로벌 GICS 기반 섹터" },
            { col: "N", desc: "섹터 내 종목 수" },
            { col: "Avg Comp", desc: "섹터 평균 Composite Score. ≥55 강세, ≤45 약세" },
            { col: "Bull% / Bear%", desc: "Bullish / Bearish classification 비율" },
            { col: "Bull/Bear Ratio", desc: "Bull% − Bear% 시각화. 양수(녹색)=강세, 음수(적색)=약세" },
            { col: "Avg L# / S#", desc: "섹터 평균 Long/Short signal 발생 전략 수" },
            { col: "TCS/TFS/RSS/OER", desc: "섹터 평균 4축 점수" },
          ]} />
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-700 bg-[#111827]">
                  <th className="py-2 px-2 text-left text-gray-500">Sector</th>
                  <th className="py-2 px-2 text-right text-gray-500">N</th>
                  <th className="py-2 px-2 text-right text-gray-500">Avg Comp</th>
                  <th className="py-2 px-2 text-right text-gray-500">Bull%</th>
                  <th className="py-2 px-2 text-right text-gray-500">Bear%</th>
                  <th className="py-2 px-2 text-left text-gray-500 w-[200px]">Bull/Bear Ratio</th>
                  <th className="py-2 px-2 text-right text-gray-500">Avg L#</th>
                  <th className="py-2 px-2 text-right text-gray-500">Avg S#</th>
                  <th className="py-2 px-2 text-right text-gray-500">TCS</th>
                  <th className="py-2 px-2 text-right text-gray-500">TFS</th>
                  <th className="py-2 px-2 text-right text-gray-500">RSS</th>
                  <th className="py-2 px-2 text-right text-gray-500">OER</th>
                </tr>
              </thead>
              <tbody>
                {sector_regime.map((s: any) => {
                  const ratio = s.pct_bullish - s.pct_bearish;
                  const barColor = ratio > 20 ? C.green : ratio < -20 ? C.red : C.yellow;
                  return (
                    <tr key={s.sector} className="border-b border-gray-800/50 hover:bg-[#1f2937]/30">
                      <td className="py-1.5 px-2 font-semibold text-gray-300">{s.sector}</td>
                      <td className="py-1.5 px-2 text-right text-gray-500">{s.n}</td>
                      <td className={`py-1.5 px-2 text-right font-semibold ${s.avg_composite >= 55 ? "text-green-400" : s.avg_composite <= 45 ? "text-red-400" : "text-gray-300"}`}>
                        {s.avg_composite}
                      </td>
                      <td className="py-1.5 px-2 text-right text-green-400">{s.pct_bullish}%</td>
                      <td className="py-1.5 px-2 text-right text-red-400">{s.pct_bearish}%</td>
                      <td className="py-1.5 px-2">
                        <div className="flex items-center gap-1">
                          <div className="relative h-2 w-[180px] bg-gray-800 rounded overflow-hidden">
                            {ratio >= 0 ? (
                              <div className="absolute left-1/2 h-full rounded-r" style={{ width: `${Math.min(ratio, 100) / 2}%`, background: barColor }} />
                            ) : (
                              <div className="absolute right-1/2 h-full rounded-l" style={{ width: `${Math.min(-ratio, 100) / 2}%`, background: barColor }} />
                            )}
                            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-600" />
                          </div>
                          <span className="text-[10px]" style={{ color: barColor }}>{ratio > 0 ? "+" : ""}{ratio.toFixed(0)}</span>
                        </div>
                      </td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{s.avg_long_count}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{s.avg_short_count}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{s.avg_tcs}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{s.avg_tfs}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{s.avg_rss}</td>
                      <td className="py-1.5 px-2 text-right text-gray-400">{s.avg_oer}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ═══ 7. Regime History Timeline ═══ */}
      {regime_history.length >= 2 && (
        <div className="bg-[#111827] border border-gray-800 rounded-lg p-4">
          <Plot
            data={[
              {
                type: "scatter", mode: "lines+markers", name: "Avg Composite",
                x: regime_history.map((h: any) => h.date),
                y: regime_history.map((h: any) => h.avg_composite),
                line: { color: C.cyan, width: 2 }, marker: { size: 5 },
                yaxis: "y",
              },
              {
                type: "scatter", mode: "lines+markers", name: "% Bullish",
                x: regime_history.map((h: any) => h.date),
                y: regime_history.map((h: any) => h.pct_bullish),
                line: { color: C.green, width: 2 }, marker: { size: 4 },
                yaxis: "y2",
              },
              {
                type: "scatter", mode: "lines+markers", name: "% Bearish",
                x: regime_history.map((h: any) => h.date),
                y: regime_history.map((h: any) => h.pct_bearish),
                line: { color: C.red, width: 2 }, marker: { size: 4 },
                yaxis: "y2",
              },
              {
                type: "scatter", mode: "lines", name: "Avg TFS",
                x: regime_history.map((h: any) => h.date),
                y: regime_history.map((h: any) => h.avg_tfs),
                line: { color: C.purple, width: 1.5, dash: "dot" as const }, marker: { size: 3 },
                yaxis: "y",
              },
            ]}
            layout={{
              ...DARK_LAYOUT,
              title: { text: "Regime History (7-Day Snapshots)", font: { size: 13, color: "#9ca3af" } },
              height: 320,
              xaxis: { title: "Date" },
              yaxis: { title: "Score (0-100)", side: "left" as const, range: [0, 100] },
              yaxis2: { title: "% of Universe", side: "right" as const, overlaying: "y" as const, range: [0, 100] },
              legend: { orientation: "h" as const, y: -0.2, font: { size: 10 } },
              margin: { t: 40, b: 60, l: 50, r: 50 },
            }}
            config={{ displayModeBar: false }}
            style={{ width: "100%" }}
          />
        </div>
      )}

    </div>
  );
}
