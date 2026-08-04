/**
 * MarketInternalsPanel — 시장 회전 진단 (Market Internals).
 * Reads /api/market-internals (headline index/ratio trends, yfinance 1h TTL) and renders:
 *   • 종합 레짐 라벨 + 내부 정렬 스코어 (백테스트: 정렬↑ = forward SPY 수익↑·downside↓)
 *   • Lens 1  회전 비율 매트릭스 (QQQ/DIA 등 × 1M/3M/6M, 부호 색)
 *   • Lens 3  크로스에셋 확인 (SPY/TLT·HYG/LQD·구리/금·공격/방어) + VIX(역상관 표기)
 *   • Lens 2  인덱스 모멘텀 (1M/3M/6M + vs-200d)
 */
import { useEffect, useState, type CSSProperties } from "react";
import { fetchMarketInternals } from "../../api/client";
import { C } from "../../styles/theme";

const REG_C: Record<string, string> = { green: C.green, amber: C.yellow, red: C.red, gray: C.gray };
const tint = (hex: string, a = "1A") => `${hex}${a}`;

// signed value → color (pos green / neg red / ~0 gray)
function sc(v: number | null | undefined): string {
  if (v == null) return C.gray;
  if (v > 0.3) return C.green;
  if (v < -0.3) return C.red;
  return C.gray;
}
function Cell({ v, suffix = "%" }: { v: number | null | undefined; suffix?: string }) {
  const col = sc(v);
  return (
    <span
      className="mono"
      style={{
        color: col, fontWeight: 600, fontSize: 12,
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(1)}${suffix}`}
    </span>
  );
}

export default function MarketInternalsPanel() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    let alive = true;
    fetchMarketInternals()
      .then((r) => { if (alive) setData(r); })
      .catch((e: any) => { if (alive) setErr(e?.message || String(e)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const reg = data?.regime;
  const regColor = reg ? (REG_C[reg.color] || C.gray) : C.gray;

  const th: CSSProperties = {
    fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase",
    color: C.gray, fontWeight: 700, padding: "6px 8px", whiteSpace: "nowrap",
  };
  const td: CSSProperties = { padding: "5px 8px", borderBottom: `1px solid ${C.bgAlt}`, whiteSpace: "nowrap" };

  return (
    <div className="mt-6 mb-4 px-3 py-3 rounded" style={{ backgroundColor: C.bg, border: `2px solid ${C.claret}44` }}>
      <div className="flex items-center gap-2 mb-2">
        <div>
          <div className="text-[15px] font-bold" style={{ color: C.claret }}>📈 Market Internals — 시장 회전 진단</div>
          <div className="text-[12px]" style={{ color: C.gray }}>
            헤드라인 인덱스·비율 추세로 레짐/회전 판독
            {data?.as_of ? ` · as_of ${data.as_of}` : ""}
            {data?.vix != null ? ` · VIX ${data.vix}` : ""}
          </div>
        </div>
        <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
          className="ml-auto rounded px-2 py-1 text-[13px]"
          style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, color: C.gray }}>
          {open ? "▾" : "▸"}
        </button>
      </div>

      {loading && !data && <div className="px-2 py-2 text-[12px]" style={{ color: C.gray }}>시장 내부 지표 로딩 중… (최초 1회 ~5초)</div>}
      {err && !data && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>Error: {err}</div>}
      {data?.error && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>{data.error}</div>}

      {open && reg && (
        <div className="grid gap-3">
          {/* 종합 레짐 + 정렬 스코어 */}
          <div className="rounded p-3" style={{ background: tint(regColor), border: `1px solid ${regColor}66` }}>
            <div className="flex items-center gap-3 flex-wrap">
              <span style={{ fontSize: 15, fontWeight: 800, color: regColor }}>{reg.label}</span>
              {reg.alignment && (
                <span className="rounded-full mono" style={{ fontSize: 12, fontWeight: 700, padding: "2px 10px",
                  background: C.panel, border: `1px solid ${regColor}`, color: regColor }}>
                  내부 정렬 {reg.alignment.score}/{reg.alignment.max}
                </span>
              )}
              <span style={{ fontSize: 11.5, color: C.gray }}>VIX {data.vix} — {reg.vix_contrarian}</span>
            </div>
            <div style={{ fontSize: 12.5, color: C.text, marginTop: 6, lineHeight: 1.5 }}>{reg.rationale}</div>
            {reg.alignment && <div style={{ fontSize: 10.5, color: C.gray, marginTop: 4 }}>{reg.alignment.note}</div>}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {/* Lens 1 — 회전 비율 매트릭스 */}
            <div className="rounded" style={{ border: `1px solid ${C.border}`, background: C.panel, overflowX: "auto" }}>
              <div style={{ ...th, borderBottom: "none", paddingBottom: 2 }}>🔄 회전 비율 (상대강도 모멘텀)</div>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
                <thead><tr>
                  <th style={{ ...th, textAlign: "left" }}>Pair · 축</th>
                  <th style={{ ...th, textAlign: "right" }}>1M</th><th style={{ ...th, textAlign: "right" }}>3M</th><th style={{ ...th, textAlign: "right" }}>6M</th>
                </tr></thead>
                <tbody>
                  {(data.rotation || []).map((r: any) => (
                    <tr key={r.pair}>
                      <td style={td}><span className="mono" style={{ fontWeight: 700, fontSize: 11.5 }}>{r.pair}</span>
                        <span style={{ color: C.gray, fontSize: 10.5 }}> · {r.axis}</span></td>
                      <td style={{ ...td, textAlign: "right" }}><Cell v={r.m1} /></td>
                      <td style={{ ...td, textAlign: "right" }}><Cell v={r.m3} /></td>
                      <td style={{ ...td, textAlign: "right" }}><Cell v={r.m6} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Lens 3 — 크로스에셋 확인 */}
            <div className="rounded" style={{ border: `1px solid ${C.border}`, background: C.panel, overflowX: "auto" }}>
              <div style={{ ...th, borderBottom: "none", paddingBottom: 2 }}>🔗 크로스에셋 확인 (3M) · ⭐breadth·구리 유효</div>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
                <tbody>
                  {(data.cross_asset || []).map((c: any) => (
                    <tr key={c.pair}>
                      <td style={td}><span className="mono" style={{ fontWeight: 700, fontSize: 11.5 }}>{c.pair}</span>
                        <span style={{ color: C.gray, fontSize: 10.5 }}> · {c.axis}</span></td>
                      <td style={{ ...td, textAlign: "right" }}><Cell v={c.m3} /></td>
                    </tr>
                  ))}
                  <tr><td style={td}><span className="mono" style={{ fontWeight: 700, fontSize: 11.5 }}>VIX</span>
                    <span style={{ color: C.gray, fontSize: 10.5 }}> · 변동성(역상관: 낮음=주의)</span></td>
                    <td style={{ ...td, textAlign: "right" }}><span className="mono" style={{ color: (data.vix ?? 20) < 20 ? C.yellow : C.text, fontWeight: 600 }}>{data.vix}</span></td></tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Lens 2 — 인덱스 모멘텀 */}
          <div className="rounded" style={{ border: `1px solid ${C.border}`, background: C.panel, overflowX: "auto" }}>
            <div style={{ ...th, borderBottom: "none", paddingBottom: 2 }}>📊 인덱스 모멘텀 + 200일선 이격</div>
            <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 420, fontSize: 12 }}>
              <thead><tr>
                <th style={{ ...th, textAlign: "left" }}>Index</th>
                <th style={{ ...th, textAlign: "right" }}>1M</th><th style={{ ...th, textAlign: "right" }}>3M</th>
                <th style={{ ...th, textAlign: "right" }}>6M</th><th style={{ ...th, textAlign: "right" }}>vs 200d</th>
              </tr></thead>
              <tbody>
                {(data.indices || []).map((r: any) => (
                  <tr key={r.t}>
                    <td style={td}><span className="mono" style={{ fontWeight: 700, fontSize: 12.5 }}>{r.t}</span></td>
                    <td style={{ ...td, textAlign: "right" }}><Cell v={r.m1} /></td>
                    <td style={{ ...td, textAlign: "right" }}><Cell v={r.m3} /></td>
                    <td style={{ ...td, textAlign: "right" }}><Cell v={r.m6} /></td>
                    <td style={{ ...td, textAlign: "right" }}><Cell v={r.vs200} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
