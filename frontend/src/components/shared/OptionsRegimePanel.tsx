/**
 * OptionsRegimePanel — 옵션 시장 포지셔닝 레짐 (Tier 1).
 * Reads /api/options-regime (VIX 기간구조 + SKEW + VVIX + 풋콜, yfinance 1h). Shows the
 * POSITIONING dimension market_internals(가격)·macro_regime(경제)는 못 보는 것:
 *   • 종합 레짐 라벨 + ★unwind 셋업 배지 (극단 쏠림 되돌림 시 초기 모멘텀 여지)
 *   • VIX 기간구조 미니바(9D/30D/3M — 백워데이션=스트레스/콘탱고=안일)
 *   • 신호 테이블: SKEW(꼬리위험 백분위)·VVIX·풋콜 OI비
 * SLOW OVERLAY: EOD·1h 캐시 — 일별 틸트용, 인트라데이 트리거 아님.
 */
import { useEffect, useState, type CSSProperties } from "react";
import { fetchOptionsRegime } from "../../api/client";
import { C } from "../../styles/theme";

const REG_C: Record<string, string> = { green: C.green, amber: C.yellow, yellow: C.yellow, red: C.red, gray: C.gray };
const stateColor = (s?: string): string => {
  if (!s) return C.gray;
  if (["backwardation", "crowded", "fear", "high"].includes(s)) return C.red;
  if (["steep_contango", "complacent"].includes(s)) return C.yellow;
  if (["elevated"].includes(s)) return C.yellow;
  return C.gray;
};

export default function OptionsRegimePanel() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(true);

  useEffect(() => {
    let alive = true;
    fetchOptionsRegime()
      .then((r) => { if (alive) setData(r); })
      .catch((e: any) => { if (alive) setErr(e?.message || String(e)); });
    return () => { alive = false; };
  }, []);

  const reg = data?.regime;
  const regColor = reg ? (REG_C[reg.color] || C.gray) : C.gray;
  const th: CSSProperties = { fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", color: C.gray, fontWeight: 700, padding: "5px 8px", whiteSpace: "nowrap" };
  const td: CSSProperties = { padding: "5px 8px", borderBottom: `1px solid ${C.bgAlt}`, whiteSpace: "nowrap" };

  // VIX term structure mini-bars (9D / 30D / 3M)
  const ts = [
    { k: "9D", v: data?.vix9d }, { k: "30D", v: data?.vix }, { k: "3M", v: data?.vix3m },
  ].filter((x) => x.v != null) as { k: string; v: number }[];
  // min-anchored scale — the three tenors are near-equal (e.g. 14.9/17.1/19.5) so a
  // 0-anchored bar hides the shape; anchor at 0.92×min so differences read visually.
  const tsVals = ts.map((x) => x.v);
  const tsMin = tsVals.length ? Math.min(...tsVals) * 0.92 : 0;
  const tsMax = tsVals.length ? Math.max(...tsVals) : 1;
  const tsState = data?.term_structure?.state as string | undefined;
  const backwardation = tsState === "backwardation";
  const tsCaption = backwardation ? "· ⚠ 백워데이션(스트레스)"
    : tsState === "steep_contango" ? "· 콘탱고(깊음·안일)"
    : tsState === "contango" ? "· 콘탱고(정상)" : "";

  return (
    <div className="mt-6 mb-4 px-3 py-3 rounded" style={{ backgroundColor: C.bg, border: `2px solid ${C.red}44` }}>
      <div className="flex items-center gap-2 mb-2">
        <div>
          <div className="text-[15px] font-bold" style={{ color: C.red }}>🎲 Options Regime — 옵션 포지셔닝</div>
          <div className="text-[12px]" style={{ color: C.gray }}>
            VIX 기간구조·SKEW·VVIX·풋콜 · 쏠림/unwind 셋업 (가격·경제 레짐과 직교)
            {data?.as_of ? ` · as_of ${data.as_of}` : ""}
          </div>
        </div>
        <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
          className="ml-auto rounded px-2 py-1 text-[13px]"
          style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, color: C.gray }}>
          {open ? "▾" : "▸"}
        </button>
      </div>

      {!data && !err && <div className="px-2 py-2 text-[12px]" style={{ color: C.gray }}>옵션 데이터 로딩 중… (yfinance ~5초)</div>}
      {err && !data && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>Error: {err}</div>}
      {data?.error && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>{data.error}</div>}

      {open && reg && (
        <div className="grid gap-3">
          {/* 종합 레짐 + unwind 셋업 */}
          <div className="rounded p-3" style={{ background: regColor + "14", border: `1px solid ${regColor}66` }}>
            <div className="flex items-center gap-3 flex-wrap">
              <span style={{ fontSize: 15, fontWeight: 800, color: regColor }}>{reg.label}</span>
              {reg.unwind_setup && (
                <span className="rounded-full" style={{ fontSize: 11, fontWeight: 700, padding: "2px 10px",
                  background: C.green + "20", border: `1px solid ${C.green}`, color: C.green }}>
                  ★ Unwind 셋업 (되돌림 시 초기 모멘텀 여지)
                </span>
              )}
              {reg.complacent && (
                <span className="rounded-full" style={{ fontSize: 11, fontWeight: 700, padding: "2px 10px",
                  background: C.yellow + "20", border: `1px solid ${C.yellow}`, color: C.yellow }}>
                  ⚠ 충격 취약
                </span>
              )}
            </div>
            <div style={{ fontSize: 12.5, color: C.text, marginTop: 6, lineHeight: 1.5 }}>{reg.rationale}</div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {/* VIX 기간구조 미니바 */}
            <div className="rounded p-3" style={{ border: `1px solid ${C.border}`, background: C.panel }}>
              <div style={{ ...th, padding: "0 0 8px 0" }}>VIX 기간구조 {tsCaption}</div>
              <div className="flex items-end gap-4" style={{ height: 104 }}>
                {ts.map((x) => {
                  const h = tsMax > tsMin ? 14 + ((x.v - tsMin) / (tsMax - tsMin)) * 48 : 40;
                  return (
                    <div key={x.k} className="flex flex-col items-center justify-end" style={{ flex: 1 }}>
                      <span className="mono" style={{ fontSize: 12, fontWeight: 700, color: C.text }}>{x.v.toFixed(1)}</span>
                      <div style={{ width: "70%", height: `${h}px`,
                        background: backwardation ? C.red : C.blue, borderRadius: "3px 3px 0 0", marginTop: 3 }} />
                      <span style={{ fontSize: 10.5, color: C.gray, marginTop: 3 }}>{x.k}</span>
                    </div>
                  );
                })}
              </div>
              <div style={{ fontSize: 10.5, color: C.gray, marginTop: 4 }}>
                단기 IV &gt; 장기 IV = 백워데이션(공포·캐피출레이션 → 바운스 셋업)
              </div>
            </div>

            {/* 신호 테이블 */}
            <div className="rounded" style={{ border: `1px solid ${C.border}`, background: C.panel, overflowX: "auto" }}>
              <div style={{ ...th, borderBottom: "none", paddingBottom: 2 }}>포지셔닝 신호</div>
              <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
                <tbody>
                  {(data.signals || []).map((s: any) => (
                    <tr key={s.code}>
                      <td style={td}><span style={{ fontWeight: 700, fontSize: 11.5 }}>{s.label}</span></td>
                      <td style={{ ...td, textAlign: "right" }}>
                        <span className="mono" style={{ fontWeight: 700, color: C.text }}>{s.value}</span>
                        {s.pctile != null && <span style={{ color: C.gray, fontSize: 10.5 }}> · {s.pctile}%ile</span>}
                      </td>
                      <td style={{ ...td, textAlign: "right" }}>
                        <span style={{ color: stateColor(s.state), fontWeight: 700, fontSize: 11 }}>{s.state}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{ fontSize: 10.5, color: C.gray, lineHeight: 1.5 }}>
            ⓘ 옵션 OI는 EOD·T+1 지연 → <b>느린 포지셔닝 오버레이</b>(인트라데이 트리거 아님). 인덱스레벨(Tier1) — 종목별 딜러감마(GEX)는 Tier2/3. yfinance 무료.
          </div>
        </div>
      )}
    </div>
  );
}
