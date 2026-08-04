/**
 * HerdingReversalPanel — 과열 후 반전(momentum-overheating → reversal) 감지.
 * Reads /api/herding-regime (paper arXiv 2607.27063 operationalized). Shows:
 *   • 시장 CCK herding 검정 (CSAD = a+b1|Rm|+b2·Rm²; b2<0 유의 = herding) + CSAD 압축상태
 *   • 섹터별 CSAD 쏠림 (분산 압축 게이트; +매수쏠림=반전-하방 / −매도쏠림=반등후보)
 *   • ★반전위험 종목 (과밀 run-up+RSI + herding 약화=단기 균열) — 검증된 희귀·고확신 conjunction
 * 검증(walk-forward, 826종목×1yr): 과밀+균열이 과밀+강세유지보다 −5..−7%p(21-42d), 단 희귀.
 * give-back 증폭 + 매수 tie-breaker로 사용(하드제외 X). 표시선행.
 */
import { useEffect, useState, type CSSProperties } from "react";
import { fetchHerdingRegime } from "../../api/client";
import { C } from "../../styles/theme";

const flagMeta: Record<string, { color: string; label: string }> = {
  REVERSAL_RISK:   { color: C.red,    label: "반전위험" },
  WATCH:           { color: C.yellow, label: "관찰" },
  CROWDED_HOLDING: { color: C.blue,   label: "강세지속" },
};

export default function HerdingReversalPanel() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(true);
  const [showHold, setShowHold] = useState(false);

  useEffect(() => {
    let alive = true;
    fetchHerdingRegime()
      .then((r) => { if (alive) setData(r); })
      .catch((e: any) => { if (alive) setErr(e?.message || String(e)); });
    return () => { alive = false; };
  }, []);

  const th: CSSProperties = { fontSize: 10, letterSpacing: "0.05em", textTransform: "uppercase", color: C.gray, fontWeight: 700, padding: "5px 7px", whiteSpace: "nowrap" };
  const td: CSSProperties = { padding: "4px 7px", borderBottom: `1px solid ${C.bgAlt}`, whiteSpace: "nowrap" };

  const mk = data?.market ?? {};
  const herding = !!mk.market_herding;
  const sectors = (data?.sectors ?? []).filter((s: any) => s.herding_flag);
  const tickers = data?.tickers ?? [];
  const rows = tickers.filter((r: any) => showHold ? true : r.flag !== "CROWDED_HOLDING");
  const nRev = tickers.filter((r: any) => r.flag === "REVERSAL_RISK").length;
  const nWatch = tickers.filter((r: any) => r.flag === "WATCH").length;
  const nHold = tickers.filter((r: any) => r.flag === "CROWDED_HOLDING").length;

  return (
    <div className="mt-6 mb-4 px-3 py-3 rounded" style={{ backgroundColor: C.bg, border: `2px solid ${C.claret}44` }}>
      <div className="flex items-center gap-2 mb-2">
        <div>
          <div className="text-[15px] font-bold" style={{ color: C.claret }}>🔥 Herding-Reversal — 과열 후 반전 감지</div>
          <div className="text-[12px]" style={{ color: C.gray }}>
            시장 CSAD herding + 섹터 쏠림 + 과밀·herding약화 반전위험 (검증된 희귀·고확신 conjunction)
            {data?.as_of ? ` · ${data.as_of}` : ""}
          </div>
        </div>
        <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
          className="ml-auto rounded px-2 py-1 text-[13px]"
          style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, color: C.gray }}>
          {open ? "▾" : "▸"}
        </button>
      </div>

      {!data && !err && <div className="px-2 py-2 text-[12px]" style={{ color: C.gray }}>herding 로딩 중…</div>}
      {err && !data && <div className="px-2 py-2 text-[12px]" style={{ color: C.red }}>Error: {err}</div>}

      {open && data && (
        <div className="grid gap-3">
          {/* 시장 herding 게이지 */}
          <div className="rounded p-2.5 flex items-center gap-4 flex-wrap" style={{
            background: (herding ? C.red : C.green) + "14", border: `1px solid ${(herding ? C.red : C.green)}66` }}>
            <span style={{ fontSize: 13, fontWeight: 800, color: herding ? C.red : C.green }}>
              시장 herding: {herding ? "감지 (쏠림 반전경계)" : "뚜렷치 않음 (정상 분산)"}
            </span>
            <span className="mono" style={{ fontSize: 12, color: C.text }}>
              CCK b2 {mk.cck_b2 ?? "—"} (t {mk.cck_t ?? "—"})
            </span>
            <span className="mono" style={{ fontSize: 12, color: C.gray }}>
              CSAD {mk.csad_recent ?? "—"} vs {mk.csad_hist ?? "—"} → {mk.csad_state ?? "—"}
            </span>
            <span style={{ fontSize: 11, color: C.gray }}>
              b2&lt;0·|t|&gt;2 이면 군집매매(횡단면 분산이 큰 이동에도 압축) — 과잉상승 반전위험↑
            </span>
          </div>

          {/* 섹터 쏠림 */}
          {sectors.length > 0 && (
            <div className="rounded" style={{ border: `1px solid ${C.border}`, background: C.panel, overflowX: "auto" }}>
              <div style={{ ...th, borderBottom: "none", paddingBottom: 2 }}>섹터 CSAD 쏠림 ({sectors.length})</div>
              <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 480, fontSize: 12 }}>
                <thead><tr>
                  <th style={{ ...th, textAlign: "left" }}>섹터</th>
                  <th style={{ ...th, textAlign: "right" }}>10d 방향</th>
                  <th style={{ ...th, textAlign: "right" }}>분산</th>
                  <th style={{ ...th, textAlign: "right" }}>herding idx</th>
                  <th style={{ ...th, textAlign: "left" }}>해석</th>
                </tr></thead>
                <tbody>
                  {sectors.map((s: any) => {
                    const up = (s.direction ?? 0) >= 0;
                    return (
                      <tr key={s.sector}>
                        <td style={td}><span className="mono" style={{ fontWeight: 700 }}>{s.sector}</span> <span style={{ color: C.gray, fontSize: 10 }}>n{s.n}</span></td>
                        <td style={{ ...td, textAlign: "right" }}><span className="mono" style={{ color: up ? C.claret : C.teal }}>{up ? "+" : ""}{s.direction}%</span></td>
                        <td style={{ ...td, textAlign: "right", color: C.gray }} className="mono">{s.dispersion}</td>
                        <td style={{ ...td, textAlign: "right" }} className="mono">{s.herding_index}</td>
                        <td style={{ ...td, fontSize: 10.5, color: up ? C.claret : C.teal }}>{up ? "매수쏠림 → 반전-하방 경계" : "매도쏠림 → exhaust 후 반등후보"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* 반전위험 종목 */}
          <div className="rounded" style={{ border: `1px solid ${C.border}`, background: C.panel, overflowX: "auto" }}>
            <div className="flex items-center gap-2" style={{ ...th, borderBottom: "none", paddingBottom: 2 }}>
              <span>★ 반전위험 {nRev} · 관찰 {nWatch} · 강세지속 {nHold}</span>
              <button type="button" onClick={() => setShowHold((v) => !v)}
                className="ml-auto rounded" style={{ background: C.bg, color: C.blue, border: `1px solid ${C.blue}`,
                  padding: "1px 8px", fontSize: 10.5, fontWeight: 700, letterSpacing: 0, textTransform: "none", cursor: "pointer" }}>
                {showHold ? "위험/관찰만" : "강세지속 포함"}
              </button>
            </div>
            <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 640, fontSize: 12 }}>
              <thead><tr>
                <th style={{ ...th, textAlign: "left" }}>Ticker</th>
                <th style={{ ...th, textAlign: "center" }}>판정</th>
                <th style={{ ...th, textAlign: "right" }}>run-up</th>
                <th style={{ ...th, textAlign: "right" }}>RSI</th>
                <th style={{ ...th, textAlign: "right" }}>최근10d</th>
                <th style={{ ...th, textAlign: "left" }}>사유</th>
              </tr></thead>
              <tbody>
                {rows.map((r: any) => {
                  const fm = flagMeta[r.flag] ?? { color: C.gray, label: r.flag };
                  return (
                    <tr key={r.ticker}>
                      <td style={td}>
                        <span className="mono" style={{ fontWeight: 700 }}>{r.ticker}</span>
                        {r.eligible && <span style={{ color: C.green, fontSize: 9, marginLeft: 4 }}>●매수후보</span>}
                        {r.cross_confirmed && <span title="옵션 풋과밀 교차확인" style={{ color: C.claret, fontSize: 9, marginLeft: 3 }}>⊕옵션</span>}
                      </td>
                      <td style={{ ...td, textAlign: "center" }}>
                        <span style={{ color: fm.color, fontWeight: 700, fontSize: 11 }}>{fm.label}{r.flag === "REVERSAL_RISK" ? " ⚠" : ""}</span>
                      </td>
                      <td style={{ ...td, textAlign: "right" }}><span className="mono">{r.runup > 0 ? "+" : ""}{r.runup}%</span> <span style={{ color: C.gray, fontSize: 9 }}>상{100 - (r.runup_pct ?? 0)}%</span></td>
                      <td style={{ ...td, textAlign: "right" }}><span className="mono" style={{ color: (r.rsi ?? 0) >= 68 ? C.claret : C.text }}>{r.rsi}</span></td>
                      <td style={{ ...td, textAlign: "right" }}><span className="mono" style={{ color: (r.ret10_pct ?? 50) < 40 ? C.red : C.text }}>{r.ret10 > 0 ? "+" : ""}{r.ret10}%</span> <span style={{ color: C.gray, fontSize: 9 }}>{r.ret10_pct}%</span></td>
                      <td style={{ ...td, color: C.gray, fontSize: 10.5, whiteSpace: "normal" }}>{(r.reasons || []).join(" · ")}</td>
                    </tr>
                  );
                })}
                {rows.length === 0 && (
                  <tr><td colSpan={6} style={{ ...td, textAlign: "center", color: C.gray }}>반전위험/관찰 종목 없음 (현재 과밀종목 대부분 강세지속 — herding 형성 중)</td></tr>
                )}
              </tbody>
            </table>
          </div>

          <div style={{ fontSize: 10.5, color: C.gray, lineHeight: 1.5 }}>
            ⓘ 논문(arXiv 2607.27063): 반전은 herding이 <b>강할 때</b>가 아니라 <b>약해질 때</b> — 과밀(run-up 상위+RSI과매수) + 단기 모멘텀 균열(herding 약화) conjunction. walk-forward 검증(826종목×1yr): 이 conjunction이 과밀·강세유지 대비 <b>−5~−7%p</b>(21-42d) 언더퍼폼, 단 <b>희귀(~14/년)</b>. 광범위 필터 아님 — give-back 증폭 + 매수 <b>tie-breaker</b> 전용(하드제외 X). 일 EOD 가격 기반.
          </div>
        </div>
      )}
    </div>
  );
}
