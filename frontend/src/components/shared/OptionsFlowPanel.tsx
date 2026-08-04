/**
 * OptionsFlowPanel — 딜러 감마(GEX) + Unwind 감지 (Tier 2/3).
 * Reads /api/options-flow (per-ticker GEX/zero-gamma/skew/풋콜, yfinance EOD 6h). Shows:
 *   • 시장 딜러 감마 틸트 (SPY/QQQ GEX 부호 — 숏감마=증폭/롱감마=핀)
 *   • ★Unwind 후보 워치리스트 (숏감마×과밀헤지×트리거 = 되돌림 시 초기모멘텀 셋업)
 *   • 종목별 GEX 테이블
 * ★모델 근사(딜러포지셔닝 가정) — 방향성 참고. 표시전용(선정 미급전).
 */
import { useEffect, useState, type CSSProperties } from "react";
import { fetchOptionsFlow } from "../../api/client";
import { C } from "../../styles/theme";

const flagColor = (f?: string) => f === "UNWIND_SETUP" ? C.green : f === "WATCH" ? C.yellow : C.gray;
const gammaColor = (r?: string) => r === "short" ? C.red : r === "long" ? C.blue : C.gray;

export default function OptionsFlowPanel() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchOptionsFlow()
      .then((r) => { if (alive) setData(r); })
      .catch((e: any) => { if (alive) setErr(e?.message || String(e)); });
    return () => { alive = false; };
  }, []);

  const th: CSSProperties = { fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase", color: C.gray, fontWeight: 700, padding: "5px 7px", whiteSpace: "nowrap" };
  const td: CSSProperties = { padding: "4px 7px", borderBottom: `1px solid ${C.bgAlt}`, whiteSpace: "nowrap" };
  const mg = data?.market_gamma;
  const cands = data?.unwind_candidates ?? [];
  const rows = showAll ? (data?.tickers ?? []) : cands;

  return (
    <div className="mt-6 mb-4 px-3 py-3 rounded" style={{ backgroundColor: C.bg, border: `2px solid ${C.red}44` }}>
      <div className="flex items-center gap-2 mb-2">
        <div>
          <div className="text-[15px] font-bold" style={{ color: C.red }}>🧨 Options Flow — 딜러 감마 · Unwind 감지</div>
          <div className="text-[12px]" style={{ color: C.gray }}>
            종목별 GEX/스큐/풋콜 · 숏감마+과밀헤지 되돌림 시 초기모멘텀 셋업 (Tier2/3)
            {data?.as_of ? ` · ${data.as_of}` : ""}
          </div>
        </div>
        <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
          className="ml-auto rounded px-2 py-1 text-[13px]"
          style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, color: C.gray }}>
          {open ? "▾" : "▸"}
        </button>
      </div>

      {!data && !err && <div className="px-2 py-2 text-[12px]" style={{ color: C.gray }}>옵션 그릭 로딩 중… (yfinance ~10초)</div>}
      {err && !data && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>Error: {err}</div>}
      {data?.error && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>{data.error}</div>}

      {open && data && !data.error && (
        <div className="grid gap-3">
          {/* 시장 딜러 감마 틸트 */}
          {mg && (
            <div className="rounded p-2.5 flex items-center gap-4 flex-wrap" style={{
              background: gammaColor(mg.regime) + "14", border: `1px solid ${gammaColor(mg.regime)}66` }}>
              <span style={{ fontSize: 13, fontWeight: 800, color: gammaColor(mg.regime) }}>
                시장 딜러감마: {mg.regime === "short" ? "숏감마 (움직임 증폭)" : mg.regime === "long" ? "롱감마 (핀·평균회귀)" : "중립"}
              </span>
              <span className="mono" style={{ fontSize: 12, color: C.text }}>SPY GEX {mg.spy} · QQQ {mg.qqq}</span>
              <span style={{ fontSize: 11, color: C.gray }}>
                숏감마 국면일수록 과밀 포지션 unwind 시 되돌림 모멘텀이 강해집니다
              </span>
            </div>
          )}

          {/* Unwind 후보 / 전체 토글 */}
          <div className="rounded" style={{ border: `1px solid ${C.border}`, background: C.panel, overflowX: "auto" }}>
            <div className="flex items-center gap-2" style={{ ...th, borderBottom: "none", paddingBottom: 2 }}>
              <span>{showAll ? "전체 종목 GEX" : `★ Unwind 후보 (${cands.length})`}</span>
              <button type="button" onClick={() => setShowAll((v) => !v)}
                className="ml-auto rounded" style={{ background: C.bg, color: C.blue, border: `1px solid ${C.blue}`,
                  padding: "1px 8px", fontSize: 10.5, fontWeight: 700, letterSpacing: 0, textTransform: "none", cursor: "pointer" }}>
                {showAll ? "후보만" : "전체 보기"}
              </button>
            </div>
            <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 620, fontSize: 12 }}>
              <thead><tr>
                <th style={{ ...th, textAlign: "left" }}>Ticker</th>
                <th style={{ ...th, textAlign: "right" }}>GEX(bn)</th>
                <th style={{ ...th, textAlign: "center" }}>감마</th>
                <th style={{ ...th, textAlign: "right" }}>풋콜 OI</th>
                <th style={{ ...th, textAlign: "right" }}>스큐</th>
                <th style={{ ...th, textAlign: "right" }}>zero-γ</th>
                {!showAll && <th style={{ ...th, textAlign: "right" }}>Unwind</th>}
                {!showAll && <th style={{ ...th, textAlign: "left" }}>사유</th>}
              </tr></thead>
              <tbody>
                {rows.map((r: any) => (
                  <tr key={r.ticker}>
                    <td style={td}><span className="mono" style={{ fontWeight: 700 }}>{r.ticker}</span></td>
                    <td style={{ ...td, textAlign: "right" }}>
                      <span className="mono" style={{ color: r.gex_bn < 0 ? C.red : C.blue, fontWeight: 600 }}>
                        {r.gex_bn > 0 ? "+" : ""}{r.gex_bn}</span>
                    </td>
                    <td style={{ ...td, textAlign: "center" }}>
                      <span style={{ color: gammaColor(r.gamma_regime), fontWeight: 700, fontSize: 11 }}>{r.gamma_regime}</span>
                    </td>
                    <td style={{ ...td, textAlign: "right" }}>
                      <span className="mono" style={{ color: (r.put_call_oi ?? 0) >= 1.8 ? C.red : C.text }}>{r.put_call_oi ?? "—"}</span>
                    </td>
                    <td style={{ ...td, textAlign: "right" }}>
                      <span className="mono" style={{ color: (r.skew ?? 0) >= 3 ? C.yellow : C.gray }}>{r.skew != null ? (r.skew > 0 ? "+" : "") + r.skew : "—"}</span>
                    </td>
                    <td style={{ ...td, textAlign: "right", color: C.gray }} className="mono">
                      {r.zg_dist_pct != null ? `${r.zg_dist_pct > 0 ? "+" : ""}${r.zg_dist_pct}%` : "—"}
                    </td>
                    {!showAll && (
                      <td style={{ ...td, textAlign: "right" }}>
                        <span style={{ color: flagColor(r.unwind_flag), fontWeight: 700, fontSize: 11 }}>
                          {r.unwind_score}{r.unwind_flag === "UNWIND_SETUP" ? " ★" : ""}
                        </span>
                      </td>
                    )}
                    {!showAll && <td style={{ ...td, color: C.gray, fontSize: 10.5 }}>{(r.unwind_reasons || []).slice(0, 2).join(", ")}</td>}
                  </tr>
                ))}
                {rows.length === 0 && (
                  <tr><td colSpan={showAll ? 6 : 8} style={{ ...td, textAlign: "center", color: C.gray }}>Unwind 셋업 후보 없음 (현재 과밀+숏감마 종목 부재)</td></tr>
                )}
              </tbody>
            </table>
          </div>

          <div style={{ fontSize: 10.5, color: C.gray, lineHeight: 1.5 }}>
            ⓘ GEX = 딜러 순감마(딜러 롱콜·숏풋 관례 근사). <b>숏감마</b>(GEX&lt;0)=움직임 증폭 → 과밀 unwind 시 강한 되돌림. yfinance OI는 EOD·T+1. <b>모델 근사</b>라 방향성 참고(절대값 주의)·미국 유동종목만. 표시전용(선정 미급전).
          </div>
        </div>
      )}
    </div>
  );
}
