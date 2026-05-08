import { useEffect, useState } from "react";
import Plot from "react-plotly.js";
import { fetchWeeklyHeatmap } from "../../api/client";
import { C, DARK_LAYOUT } from "../../styles/theme";

/* ─── Reusable group heatmap ─── */
function GroupHeatmap({ title, data, desc }: { title: string; data: any; desc: string }) {
  const [metric, setMetric] = useState<"composite" | "bullish">("composite");
  if (!data || !data.keys || data.keys.length === 0) return null;

  const z = metric === "composite" ? data.z_composite : data.z_pct_bullish;
  const zmid = 50;
  const cbarTitle = metric === "composite" ? "Avg Composite" : "% Bullish";
  const colorscale = metric === "composite"
    ? [[0, "#dc2626"], [0.3, "#ef4444"], [0.5, "#374151"], [0.7, "#22c55e"], [1, "#16a34a"]]
    : [[0, "#dc2626"], [0.3, "#ef4444"], [0.5, "#374151"], [0.7, "#22c55e"], [1, "#16a34a"]];

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div>
          <h4 className="text-xs font-semibold text-gray-400">{title} Heatmap</h4>
          <p className="text-[10px] text-gray-500">{desc}</p>
        </div>
        <div className="flex gap-1">
          <button
            className={`px-2 py-0.5 text-[10px] rounded ${metric === "composite" ? "bg-cyan-600 text-white" : "bg-[#1f2937] text-gray-400"}`}
            onClick={() => setMetric("composite")}>Composite</button>
          <button
            className={`px-2 py-0.5 text-[10px] rounded ${metric === "bullish" ? "bg-cyan-600 text-white" : "bg-[#1f2937] text-gray-400"}`}
            onClick={() => setMetric("bullish")}>% Bullish</button>
        </div>
      </div>
      <Plot
        data={[{
          type: "heatmap",
          x: data.dates, y: data.keys, z: z,
          colorscale, zmid,
          colorbar: { title: cbarTitle, titleside: "right" as const, len: 0.6, tickfont: { size: 9 } },
          xgap: 1, ygap: 2,
          hovertemplate: "%{y}<br>%{x}<br>" + cbarTitle + ": %{z:.1f}<extra></extra>",
        } as any]}
        layout={{
          ...DARK_LAYOUT,
          height: Math.max(280, data.keys.length * 24 + 80),
          margin: { t: 10, b: 60, l: 160, r: 80 },
          xaxis: { color: C.gray, tickangle: -45, tickfont: { size: 9 } },
          yaxis: { color: C.gray, autorange: "reversed" as any, tickfont: { size: 10 } },
        }}
        config={{ responsive: true }} style={{ width: "100%" }}
      />
    </div>
  );
}

export function BacktestTab() {
  const [data, setData] = useState<any>(null);
  useEffect(() => { fetchWeeklyHeatmap().then(setData); }, []);

  if (!data || data.error) return <div className="text-gray-500 p-8">{data?.error || "Loading..."}</div>;

  const { dates, n_snapshots, ticker_heatmap: th, sector_heatmap, category_heatmap, theme_heatmap } = data;

  // Ticker heatmap: mark current-week tickers
  const tickerLabels = (th.tickers || []).map((tk: string, i: number) =>
    `${tk} (${th.counts?.[i] || 0})`
  );

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-bold text-gray-200 mb-1">Weekly Backtest Heatmaps</h2>
        <p className="text-xs text-gray-500">
          {n_snapshots}주간 주간 스냅샷 기반. Ticker = Top 10 선정 이력 (1M Fwd Return), Sector/Category/Theme = 전체 유니버스 집계.
        </p>
      </div>

      {/* ═══ 1. Ticker Persistence Heatmap ═══ */}
      {th.tickers && th.tickers.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-gray-400 mb-1">Ticker Persistence Heatmap</h4>
          <p className="text-[10px] text-gray-500 mb-2">
            X=주간 평가일, Y=종목(선정 횟수 기준 상위 30). 셀 색상=1M Forward Return. 빈 셀=해당 주 미선정. 괄호 안 숫자=총 선정 횟수.
          </p>
          <Plot
            data={[{
              type: "heatmap",
              x: dates,
              y: tickerLabels,
              z: th.z,
              colorscale: [[0, "#dc2626"], [0.35, "#ef4444"], [0.5, "#374151"], [0.65, "#22c55e"], [1, "#16a34a"]],
              zmid: 0,
              colorbar: { title: "1M Ret%", titleside: "right" as const, len: 0.5, tickfont: { size: 9 } },
              xgap: 1, ygap: 2,
              hovertemplate: "%{y}<br>%{x}<br>1M Return: %{z:.1f}%<extra></extra>",
            } as any]}
            layout={{
              ...DARK_LAYOUT,
              height: Math.max(350, th.tickers.length * 22 + 80),
              margin: { t: 10, b: 60, l: 120, r: 80 },
              xaxis: { color: C.gray, tickangle: -45, tickfont: { size: 9 } },
              yaxis: { color: C.gray, autorange: "reversed" as any, tickfont: { size: 10 } },
            }}
            config={{ responsive: true }} style={{ width: "100%" }}
          />
        </div>
      )}

      {/* ═══ 2. Sector Heatmap ═══ */}
      <GroupHeatmap title="Sector" data={sector_heatmap}
        desc="전체 유니버스의 섹터별 주간 Avg Composite / % Bullish 추이. 섹터 로테이션 및 모멘텀 전파 패턴 확인." />

      {/* ═══ 3. Category Heatmap ═══ */}
      <GroupHeatmap title="Category" data={category_heatmap}
        desc="ETF/주식 카테고리별 주간 추이. 세분화된 자산군 간 강약 시차 및 자금 이동 패턴." />

      {/* ═══ 4. Theme Heatmap ═══ */}
      <GroupHeatmap title="Theme" data={theme_heatmap}
        desc="개별주 테마별 주간 추이 (주식 유니버스만 해당). 테마 crowding → reversal, 모멘텀 cascade 패턴." />
    </div>
  );
}
