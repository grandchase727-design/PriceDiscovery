import { useEffect, useState, useMemo } from "react";
import { fetchPreBreakoutBase } from "../../api/client";
import { C } from "../../styles/theme";

// ---------------------------------------------------------------------------
// Pre-Breakout Base — 상승추세 속 VCP 베이스 → 돌파 대기/승격 (①②)
// ---------------------------------------------------------------------------
// classify()는 이런 종목을 CONSOLIDATION으로 *정확히* 라벨링하지만 Eligibility Gate가
// 통째로 탈락시켜 '상승추세 속 건강한 베이스(pre-breakout)'와 '무추세 횡보(노이즈)'를
// 구분하지 못한다. 이 티어는 그 정보(장기 MA 상승 + SEPA Stage2 + VCP 수축)를 회수한다.
//   BASE_WATCH        : 베이스 조건 충족, 돌파 대기 (관망)
//   BREAKOUT_CONFIRMED: 거래량 동반 피벗 돌파 → 매수 후보 승격 (②)
// 매수 확정 리스트와 별개의 '관찰 티어'. 어느 모듈도 override하지 않음.
// ---------------------------------------------------------------------------

interface PBBCandidate {
  ticker: string; name: string; sector: string; asset_type: string;
  classification: string; composite: number; minervini_long: number;
  vcr: number; rss: number; rsi: number; oer: number;
  pct_from_high: number; range_pct: number; sma200_slope: number;
  breakout_20d: number; vol_ratio: number; adv_M: number;
  entry_timing_status: string; qvr_score: number;
  cyclical_tag: string; style_tilt: string; region: string;
  ret_1m: number; ret_3m: number; ret_21d: number; ret_252d: number; trend_age: number;
  pre_breakout_state: string; pre_breakout_score: number;
}

function pctColor(v: number | null | undefined) {
  if (v == null) return C.gray;
  return v > 0 ? C.green : v < 0 ? C.red : C.gray;
}
function fpct(v: number | null | undefined) {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

function StateBadge({ state }: { state: string }) {
  const confirmed = state === "BREAKOUT_CONFIRMED";
  const color = confirmed ? C.green : C.yellow;
  return (
    <span
      style={{
        color: confirmed ? "#fff" : color,
        backgroundColor: confirmed ? color : color + "1a",
        border: `1px solid ${color}${confirmed ? "" : "66"}`,
        borderRadius: 4, padding: "2px 8px", fontSize: 11, fontWeight: "bold",
        whiteSpace: "nowrap",
      }}
    >
      {confirmed ? "🟢 매수 승격 (돌파 확인)" : "🟡 돌파 대기"}
    </span>
  );
}

export function PreBreakoutBaseTab() {
  const [data, setData] = useState<PBBCandidate[]>([]);
  const [meta, setMeta] = useState<any>({});
  const [loading, setLoading] = useState(true);
  const [typeFilter, setTypeFilter] = useState<"ALL" | "Stock" | "ETF">("ALL");

  useEffect(() => {
    setLoading(true);
    fetchPreBreakoutBase()
      .then((r) => { setData(r.candidates || []); setMeta(r); })
      .catch(() => setData([]))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(
    () => (typeFilter === "ALL" ? data : data.filter((d) => d.asset_type === typeFilter)),
    [data, typeFilter]
  );

  if (loading) return <div className="text-[#857F7A] p-8">Loading pre-breakout base data…</div>;

  return (
    <div className="space-y-5">
      {/* Intro */}
      <div className="rounded-lg border p-3" style={{ borderColor: C.blue + "55", background: C.blue + "0d" }}>
        <div className="flex items-baseline justify-between mb-1.5">
          <h3 className="text-[16px] font-bold" style={{ color: C.blue }}>
            🧊 Pre-Breakout Base — 돌파 대기 베이스 티어
          </h3>
          <span className="text-[12px]" style={{ color: C.gray }}>
            상승추세 속 VCP 수축 → 돌파 대기/승격 · 매수 확정과 별개 관찰 티어
          </span>
        </div>
        <p className="text-[13px] leading-relaxed" style={{ color: C.text }}>
          모멘텀 분류가 <strong>CONSOLIDATION/NEUTRAL/PULLBACK</strong>(단기 FLAT)이라 매수 게이트에서 <strong>탈락</strong>했지만,
          장기 추세는 살아있고(200일선 위·상승) <strong>Minervini SEPA Stage 2</strong>(강도≥60) + <strong>VCP 수축</strong>(vcr&lt;0.85)으로
          52주 고점 부근에서 <strong>코일링</strong>하는 '건강한 베이스'를 회수합니다. 이는 O'Neil/Minervini의 pre-breakout 셋업으로,
          <strong style={{ color: C.green }}> 거래량 동반 피벗 돌파</strong> 시 <strong>매수 후보로 승격</strong>됩니다
          (다음 스캔에서 classify가 CONTINUATION으로 자동 승격). 채권·통화·크립토 등 비주식 섹터는 제외.
        </p>
      </div>

      {/* Summary */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="rounded px-3 py-2 border" style={{ borderColor: C.border }}>
          <div className="text-[11px]" style={{ color: C.gray }}>총 베이스</div>
          <div className="text-[15px] font-bold" style={{ color: C.text }}>{meta.n_total ?? data.length}</div>
        </div>
        <div className="rounded px-3 py-2 border" style={{ borderColor: C.border }}>
          <div className="text-[11px]" style={{ color: C.gray }}>돌파 대기 (BASE_WATCH)</div>
          <div className="text-[15px] font-bold" style={{ color: C.yellow }}>{meta.n_watch ?? 0}</div>
        </div>
        <div className="rounded px-3 py-2 border" style={{ borderColor: meta.n_confirmed ? C.green : C.border, background: meta.n_confirmed ? C.green + "10" : "transparent" }}>
          <div className="text-[11px]" style={{ color: C.gray }}>매수 승격 (돌파 확인)</div>
          <div className="text-[15px] font-bold" style={{ color: C.green }}>{meta.n_confirmed ?? 0}</div>
        </div>
        <div className="ml-auto flex rounded overflow-hidden border" style={{ borderColor: C.border }}>
          {(["ALL", "Stock", "ETF"] as const).map((t) => (
            <button key={t} onClick={() => setTypeFilter(t)}
              className="px-3 py-1 text-[12px] font-semibold"
              style={{ background: typeFilter === t ? C.blue : "transparent", color: typeFilter === t ? "#fff" : C.gray }}>
              {t}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="text-[13px] p-6 text-center rounded border" style={{ color: C.gray, borderColor: C.border }}>
          현재 Pre-Breakout Base 후보가 없습니다.
        </div>
      ) : (
        <div className="overflow-x-auto rounded border" style={{ borderColor: C.border }}>
          <table className="w-full text-[12px]">
            <thead className="sticky top-0" style={{ backgroundColor: C.bgAlt }}>
              <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                <th className="text-left px-2 py-1.5" style={{ color: C.gray }}>Ticker</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }}>Type</th>
                <th className="text-left px-2 py-1.5" style={{ color: C.gray, minWidth: 150 }}>상태</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray, minWidth: 68 }} title="베이스 품질: 0.40 SEPA + 0.25 VCP타이트 + 0.20 고점근접 + 0.15 RS">Base Score</th>
                <th className="text-left px-2 py-1.5" style={{ color: C.gray }}>분류</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }} title="Minervini SEPA 강도 (minervini_long)">SEPA</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }} title="Volatility Contraction Ratio (낮을수록 타이트)">VCP(vcr)</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }} title="상대강도 percentile">RS</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }} title="52주 고점 대비">vs 52wHi</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }}>Composite</th>
                <th className="text-center px-2 py-1.5" style={{ color: C.gray }}>QVR</th>
                <th className="text-right px-2 py-1.5" style={{ color: C.gray }}>1mo</th>
                <th className="text-right px-2 py-1.5" style={{ color: C.gray }}>3mo</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d) => {
                const confirmed = d.pre_breakout_state === "BREAKOUT_CONFIRMED";
                return (
                  <tr key={d.ticker}
                      style={{ borderBottom: `1px solid ${C.border}40`,
                               background: confirmed ? C.green + "0d" : "transparent" }}>
                    <td className="px-2 py-1.5">
                      <div className="font-mono font-bold" style={{ color: C.text }}>{d.ticker}</div>
                      <div className="text-[10px]" style={{ color: C.gray }}>{d.sector}</div>
                    </td>
                    <td className="text-center px-2 py-1.5">{d.asset_type === "ETF" ? "📦" : "📈"}</td>
                    <td className="px-2 py-1.5"><StateBadge state={d.pre_breakout_state} /></td>
                    <td className="text-center px-2 py-1.5">
                      <span className="font-mono font-bold text-[14px]" style={{ color: C.blue }}>
                        {d.pre_breakout_score?.toFixed(0) ?? "—"}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-[11px]" style={{ color: C.text }}>{d.classification}</td>
                    <td className="text-center px-2 py-1.5 font-mono" style={{ color: d.minervini_long >= 70 ? C.green : C.text }}>{d.minervini_long?.toFixed(0) ?? "—"}</td>
                    <td className="text-center px-2 py-1.5 font-mono" style={{ color: d.vcr != null && d.vcr < 0.5 ? C.green : C.text }}>{d.vcr?.toFixed(2) ?? "—"}</td>
                    <td className="text-center px-2 py-1.5 font-mono" style={{ color: d.rss >= 70 ? C.green : C.text }}>{d.rss?.toFixed(0) ?? "—"}</td>
                    <td className="text-center px-2 py-1.5 font-mono" style={{ color: pctColor(d.pct_from_high) }}>{fpct(d.pct_from_high)}</td>
                    <td className="text-center px-2 py-1.5 font-mono" style={{ color: C.text }}>{d.composite?.toFixed(0) ?? "—"}</td>
                    <td className="text-center px-2 py-1.5 font-mono" style={{ color: C.text }}>{d.qvr_score?.toFixed(0) ?? "—"}</td>
                    <td className="text-right px-2 py-1.5 font-mono" style={{ color: pctColor(d.ret_1m) }}>{fpct(d.ret_1m)}</td>
                    <td className="text-right px-2 py-1.5 font-mono" style={{ color: pctColor(d.ret_3m) }}>{fpct(d.ret_3m)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Methodology note */}
      <div className="text-[11px] rounded border p-3 leading-relaxed" style={{ borderColor: C.border, color: C.gray }}>
        <strong style={{ color: C.text }}>게이트:</strong> {meta.methodology?.gate || "NOT eligible · CONSOLIDATION/NEUTRAL/PULLBACK · 200일선 위·상승 · SEPA≥60 · VCP 수축 · 고점 20% 이내 · ADV≥$5M"}
        <br />
        <strong style={{ color: C.green }}>승격(②):</strong> {meta.methodology?.promotion || "거래량 동반 피벗 돌파 → 매수 후보 승격"}
        <br />
        <strong style={{ color: C.text }}>합의 원리:</strong> 모멘텀 스캔(확인)과 SEPA(예측)는 <strong>같은 상승추세를 다른 시간축</strong>에서 볼 뿐,
        진짜 합의점은 <strong style={{ color: C.green }}>거래량 동반 돌파</strong>다. 그 순간 단기 방향이 FLAT→UP으로 바뀌며 두 신호가 자동 정렬된다.
      </div>
    </div>
  );
}
