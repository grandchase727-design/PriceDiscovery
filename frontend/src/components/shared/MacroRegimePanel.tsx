/**
 * MacroRegimePanel — 경기순환 레짐 (Macro Regime, FRED 기반).
 * Reads /api/macro-regime (economic-cycle regime, monthly overlay — orthogonal to
 * the price-derived market regime). Renders:
 *   • Tier 1  Growth×Inflation 4-사분면 (z-score 합성) — 현재 위치 플롯 + 함의
 *   • Tier 2  마르코프 침체확률 게이지 + 36개월 스파크라인
 *   • Divergence  매크로(경제) vs 시장(가격) 레짐 정합/불일치 플래그
 * SLOW OVERLAY: 월간·사후수정 데이터 → 느린 틸트용, 일별 트리거 아님.
 */
import { lazy, Suspense, useEffect, useState, type CSSProperties } from "react";
import { fetchMacroRegime } from "../../api/client";
import { C } from "../../styles/theme";

// three.js 번들은 무겁다 → Orbit 열 때만 로드 (code-split)
const MacroOrbitView = lazy(() => import("./MacroOrbitView"));

// quadrant → accent color
const QUAD_C: Record<string, string> = {
  Reflation: C.yellow, Goldilocks: C.green, Stagflation: C.red, Slowdown: C.blue,
};
const zColor = (z: number | null | undefined): string =>
  z == null ? C.gray : z > 0.25 ? C.green : z < -0.25 ? C.red : C.gray;

// recession-probability → color
const recColor = (p: number | null | undefined): string =>
  p == null ? C.gray : p >= 50 ? C.red : p >= 20 ? C.yellow : C.green;

function ZBar({ z }: { z: number }) {
  // −3..+3 mapped to a centered horizontal bar
  const clamped = Math.max(-3, Math.min(3, z));
  const pct = (Math.abs(clamped) / 3) * 50;
  const col = zColor(z);
  return (
    <span style={{ position: "relative", display: "inline-block", width: 90, height: 10, background: C.bgAlt, borderRadius: 2, verticalAlign: "middle" }}>
      <span style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: C.gray }} />
      <span style={{ position: "absolute", top: 1, bottom: 1, borderRadius: 2, background: col,
        left: z >= 0 ? "50%" : `${50 - pct}%`, width: `${pct}%` }} />
    </span>
  );
}

// inline sparkline for recession-prob history
function Spark({ data, w = 200, h = 34 }: { data: { date: string; prob: number }[]; w?: number; h?: number }) {
  if (!data?.length) return null;
  const xs = data.map((_, i) => (i / (data.length - 1)) * w);
  const ys = data.map((d) => h - (Math.max(0, Math.min(100, d.prob)) / 100) * h);
  const path = xs.map((x, i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${ys[i].toFixed(1)}`).join(" ");
  const area = `${path} L${w},${h} L0,${h} Z`;
  const last = data[data.length - 1].prob;
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <line x1="0" y1={h - (50 / 100) * h} x2={w} y2={h - (50 / 100) * h} stroke={C.red} strokeWidth="0.5" strokeDasharray="3 3" opacity="0.5" />
      <path d={area} fill={recColor(last)} opacity="0.14" />
      <path d={path} fill="none" stroke={recColor(last)} strokeWidth="1.5" />
      <circle cx={xs[xs.length - 1]} cy={ys[ys.length - 1]} r="2.5" fill={recColor(last)} />
    </svg>
  );
}

// 2×2 quadrant plot with current position dot
function QuadrantPlot({ g, i, quad }: { g: number; i: number; quad: string }) {
  const S = 150, pad = 16, mid = S / 2;
  const scale = (z: number) => mid + Math.max(-2.5, Math.min(2.5, z)) / 2.5 * (mid - pad);
  const cx = scale(g), cy = S - scale(i); // x=growth, y=inflation (up = higher inflation)
  const quadCol = QUAD_C[quad] || C.gray;
  const lbl = (x: number, y: number, ko: string, on: boolean) => (
    <text x={x} y={y} fontSize="9" fontWeight={on ? 700 : 400} textAnchor="middle"
      fill={on ? quadCol : C.gray} opacity={on ? 1 : 0.55}>{ko}</text>
  );
  return (
    <svg width={S} height={S} style={{ display: "block" }}>
      <rect x="0" y="0" width={S} height={S} fill={C.bgAlt} rx="4" />
      <line x1={mid} y1={pad / 2} x2={mid} y2={S - pad / 2} stroke={C.border} strokeWidth="1" />
      <line x1={pad / 2} y1={mid} x2={S - pad / 2} y2={mid} stroke={C.border} strokeWidth="1" />
      {/* quadrant labels: TL 스태그, TR 리플레이션, BL 슬로다운, BR 골디락스 */}
      {lbl(mid / 2, pad, "스태그", quad === "Stagflation")}
      {lbl(mid + mid / 2, pad, "리플레이션", quad === "Reflation")}
      {lbl(mid / 2, S - pad / 2, "슬로다운", quad === "Slowdown")}
      {lbl(mid + mid / 2, S - pad / 2, "골디락스", quad === "Goldilocks")}
      {/* axis hints */}
      <text x={S - 3} y={mid - 3} fontSize="7" textAnchor="end" fill={C.gray}>성장+</text>
      <text x={3} y={mid - 3} fontSize="7" fill={C.gray}>성장−</text>
      {/* position */}
      <circle cx={cx} cy={cy} r="5.5" fill={quadCol} stroke={C.panel} strokeWidth="1.5" />
      <circle cx={cx} cy={cy} r="9" fill="none" stroke={quadCol} strokeWidth="1" opacity="0.4" />
    </svg>
  );
}

export default function MacroRegimePanel() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  const [orbitOpen, setOrbitOpen] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchMacroRegime()
      .then((r) => { if (alive) setData(r); })
      .catch((e: any) => { if (alive) setErr(e?.message || String(e)); })
      .finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, []);

  const t1 = data?.tier1, t2 = data?.tier2, dv = data?.divergence;
  const quadCol = t1 ? (QUAD_C[t1.quadrant] || C.gray) : C.gray;
  const th: CSSProperties = { fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", color: C.gray, fontWeight: 700, padding: "5px 8px", whiteSpace: "nowrap" };
  const td: CSSProperties = { padding: "4px 8px", borderBottom: `1px solid ${C.bgAlt}`, whiteSpace: "nowrap" };

  return (
    <div className="mt-6 mb-4 px-3 py-3 rounded" style={{ backgroundColor: C.bg, border: `2px solid ${C.blue}44` }}>
      <div className="flex items-center gap-2 mb-2">
        <div>
          <div className="text-[15px] font-bold" style={{ color: C.blue }}>🌐 Macro Regime — 경기순환 레짐 (FRED)</div>
          <div className="text-[12px]" style={{ color: C.gray }}>
            경제 데이터 기반 · 시장(가격) 레짐과 직교 · 느린 월간 오버레이
            {data?.as_of ? ` · as_of ${data.as_of}` : ""}
          </div>
        </div>
        <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
          className="ml-auto rounded px-2 py-1 text-[13px]"
          style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, color: C.gray }}>
          {open ? "▾" : "▸"}
        </button>
      </div>

      {loading && !data && <div className="px-2 py-2 text-[12px]" style={{ color: C.gray }}>FRED 매크로 시계열 로딩 중… (최초 1회 ~10초)</div>}
      {err && !data && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>Error: {err}</div>}
      {data?.error && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>{data.error}</div>}

      {open && t1 && (
        <div className="grid gap-3">
          {/* Divergence banner */}
          {dv && (
            <div className="rounded p-2.5" style={{
              background: (dv.flag === "DIVERGENCE" ? C.red : C.green) + "14",
              border: `1px solid ${(dv.flag === "DIVERGENCE" ? C.red : C.green)}66` }}>
              <span style={{ fontSize: 12, fontWeight: 800, color: dv.flag === "DIVERGENCE" ? C.red : C.green }}>
                {dv.flag === "DIVERGENCE" ? "⚠ 매크로↔시장 불일치" : "✓ 매크로↔시장 정합"}
              </span>
              <span style={{ fontSize: 12, color: C.text, marginLeft: 8 }}>{dv.note}</span>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {/* Tier 1 — Quadrant */}
            <div className="rounded p-3" style={{ border: `1px solid ${C.border}`, background: C.panel }}>
              <div style={{ ...th, borderBottom: "none", padding: "0 0 6px 0" }}>Tier 1 · 성장×인플레 4-사분면</div>
              <div className="flex items-center gap-3 flex-wrap">
                <QuadrantPlot g={t1.growth_z} i={t1.inflation_z} quad={t1.quadrant} />
                <div style={{ flex: 1, minWidth: 150 }}>
                  <div style={{ fontSize: 16, fontWeight: 800, color: quadCol }}>
                    {t1.quadrant_ko}
                    {t1.boundary && <span style={{ fontSize: 10, fontWeight: 600, color: C.yellow, marginLeft: 6 }}>경계</span>}
                  </div>
                  <div className="mt-1" style={{ fontSize: 11.5, lineHeight: 1.5, color: C.text }}>{t1.implication}</div>
                  <div className="mt-2 flex gap-2 flex-wrap" style={{ fontSize: 11 }}>
                    <span className="mono" style={{ color: zColor(t1.growth_z), fontWeight: 700 }}>성장 z {t1.growth_z >= 0 ? "+" : ""}{t1.growth_z}</span>
                    <span className="mono" style={{ color: zColor(t1.inflation_z), fontWeight: 700 }}>인플 z {t1.inflation_z >= 0 ? "+" : ""}{t1.inflation_z}</span>
                    <span className="mono" style={{ color: t1.risk === "risk_off" ? C.red : t1.risk === "risk_on" ? C.green : C.gray, fontWeight: 700 }}>
                      크레딧 z {t1.credit_z >= 0 ? "+" : ""}{t1.credit_z} · {t1.risk}
                    </span>
                    {t1.liquidity_z != null && (
                      <span className="mono" style={{ color: zColor(t1.liquidity_z), fontWeight: 700 }}>
                        유동성 z {t1.liquidity_z >= 0 ? "+" : ""}{t1.liquidity_z} · {t1.liquidity_state}
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Tier 2 — Recession probability */}
            <div className="rounded p-3" style={{ border: `1px solid ${C.border}`, background: C.panel }}>
              <div style={{ ...th, borderBottom: "none", padding: "0 0 6px 0" }}>Tier 2 · 마르코프 침체확률</div>
              {t2?.recession_prob != null ? (
                <div>
                  <div className="flex items-baseline gap-2">
                    <span className="mono" style={{ fontSize: 30, fontWeight: 800, color: recColor(t2.recession_prob) }}>{t2.recession_prob}%</span>
                    <span style={{ fontSize: 11, color: C.gray }}>
                      최근3M {(t2.recession_prob_3m || []).join(" → ")}%
                    </span>
                  </div>
                  {/* gauge */}
                  <div style={{ position: "relative", height: 8, background: C.bgAlt, borderRadius: 4, margin: "6px 0" }}>
                    <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, borderRadius: 4,
                      width: `${Math.max(1, Math.min(100, t2.recession_prob))}%`, background: recColor(t2.recession_prob) }} />
                    <div style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: C.red, opacity: 0.5 }} />
                  </div>
                  <Spark data={t2.history || []} />
                  <div style={{ fontSize: 10, color: C.gray, marginTop: 2 }}>
                    코인시던트 {t2.coincident_asof} · 36M 추이 · 수렴 {t2.converged ? "✓" : "✗"} (점선=50%)
                  </div>
                  {t2.gdi && (
                    <div className="mt-2 pt-2" style={{ borderTop: `1px solid ${C.bgAlt}`, fontSize: 11 }}>
                      <span style={{ color: C.gray }}>소득측 성장(GDP+GDI): </span>
                      <span className="mono" style={{ fontWeight: 700, color: t2.gdi.avg_yoy > 1 ? C.green : t2.gdi.avg_yoy < 0 ? C.red : C.yellow }}>
                        {t2.gdi.avg_yoy >= 0 ? "+" : ""}{t2.gdi.avg_yoy}% YoY · {t2.gdi.signal}
                      </span>
                      <span style={{ color: C.gray }}> (GDP {t2.gdi.gdp_yoy}% / GDI {t2.gdi.gdi_yoy}% · {t2.gdi.asof})</span>
                    </div>
                  )}
                </div>
              ) : (
                <div style={{ fontSize: 12, color: C.gray }}>{t2?.error || "침체확률 계산 불가"}</div>
              )}
            </div>
          </div>

          {/* Tier 1 components */}
          <div className="rounded" style={{ border: `1px solid ${C.border}`, background: C.panel, overflowX: "auto" }}>
            <div className="flex items-center gap-2" style={{ ...th, borderBottom: "none", paddingBottom: 2 }}>
              <span>구성 지표 (FRED · z = 방향정규화)</span>
              <button type="button" onClick={() => setOrbitOpen((v) => !v)}
                className="ml-auto rounded"
                style={{ background: orbitOpen ? C.blue : C.bg, color: orbitOpen ? "#fff" : C.blue,
                  border: `1px solid ${C.blue}`, padding: "2px 9px", fontSize: 11, fontWeight: 700,
                  letterSpacing: 0, textTransform: "none", cursor: "pointer" }}>
                🪐 {orbitOpen ? "Orbit 닫기" : "Orbit 3D 보기"}
              </button>
            </div>
            {orbitOpen && (
              <div className="px-2 pb-2">
                <Suspense fallback={<div style={{ height: 480, display: "grid", placeItems: "center",
                  color: C.gray, fontSize: 13, background: "#070B16", borderRadius: 8 }}>3D 엔진 로딩 중…</div>}>
                  <MacroOrbitView onClose={() => setOrbitOpen(false)} />
                </Suspense>
              </div>
            )}
            <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 480, fontSize: 12 }}>
              <thead><tr>
                <th style={{ ...th, textAlign: "left" }}>축</th>
                <th style={{ ...th, textAlign: "left" }}>지표</th>
                <th style={{ ...th, textAlign: "right" }}>값</th>
                <th style={{ ...th, textAlign: "left" }}>z</th>
                <th style={{ ...th, textAlign: "right" }}>as_of</th>
              </tr></thead>
              <tbody>
                {[...(t1.components || []),
                  ...(t1.credit || []).map((c: any) => ({ ...c, axis: "credit" })),
                  ...(t1.liquidity || []).map((c: any) => ({ ...c, axis: "liquidity" }))].map((c: any) => (
                  <tr key={c.code}>
                    <td style={td}><span style={{ fontSize: 10.5, color: C.gray }}>{c.axis}</span></td>
                    <td style={td}><span className="mono" style={{ fontWeight: 700, fontSize: 11 }}>{c.code}</span>
                      <span style={{ color: C.gray, fontSize: 10.5 }}> · {c.label}</span></td>
                    <td style={{ ...td, textAlign: "right" }}><span className="mono" style={{ fontVariantNumeric: "tabular-nums" }}>{c.value}</span></td>
                    <td style={td}>
                      <ZBar z={c.z} />
                      <span className="mono" style={{ fontSize: 11, fontWeight: 700, color: zColor(c.z), marginLeft: 6 }}>{c.z >= 0 ? "+" : ""}{c.z}</span>
                    </td>
                    <td style={{ ...td, textAlign: "right", color: C.gray, fontSize: 10.5 }}>{c.asof || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div style={{ fontSize: 10.5, color: C.gray, lineHeight: 1.5 }}>
            ⓘ 월간·주간 경제데이터의 <b>느린 오버레이</b> — 일별 매매 트리거 아님(빠른층은 Market Internals). 값은 현재 빈티지이며 사후수정됨 → 백테스트는 실시간 빈티지(ALFRED) 필요. FRED SA 계열 사용 → X-13 불필요.
          </div>
        </div>
      )}
    </div>
  );
}
