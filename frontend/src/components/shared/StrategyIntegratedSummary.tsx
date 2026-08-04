/**
 * StrategyIntegratedSummary — 3-strategy (Elliott · CAN SLIM · SEPA) integrated
 * overview shown ABOVE the per-ticker cards in the 매매전략 (FinalListEtfWavePanel).
 *
 * Combines two views on the SAME live `indices` data:
 *   1) Section synthesis — verdict distribution bar + per-strategy lens counts + a
 *      generated stance line, for STOCKS and ETFs.
 *   2) 합치도 히트맵 — per-ticker Elliott/CAN SLIM/SEPA color cells + an agreement
 *      read, sorted by consensus; divergent (⚠괴리) rows highlighted.
 *
 * Data-driven (no hardcoded numbers) so it tracks the daily swarm output. Styled
 * entirely through the FT `C` theme to match the dashboard.
 */
import { useMemo, useState } from "react";
import { C } from "../../styles/theme";

type Tone = "good" | "warn" | "bad" | "neut";
const TONE: Record<Tone, string> = { good: C.green, warn: C.yellow, bad: C.red, neut: C.gray };
const tint = (hex: string, a = "1A") => `${hex}${a}`; // 8-digit hex alpha

// ── Elliott phase → {kr, tone} (엘리엇 = 손절·구조) ──
const PHASE: Record<string, { kr: string; tone: Tone }> = {
  IMPULSE_W1: { kr: "상승1파", tone: "good" },
  IMPULSE_W2: { kr: "상승2파", tone: "good" },
  IMPULSE_W3: { kr: "상승3파", tone: "good" },
  IMPULSE_W4: { kr: "상승4파눌림", tone: "good" },
  IMPULSE_W5: { kr: "상승5파종반", tone: "warn" },
  CORRECTIVE_A: { kr: "조정A파", tone: "bad" },
  CORRECTIVE_B: { kr: "조정B파", tone: "bad" },
  CORRECTIVE_C: { kr: "조정C파", tone: "bad" },
  UNCLEAR: { kr: "불명확", tone: "neut" },
};
const phaseInfo = (p?: string) => PHASE[p || "UNCLEAR"] || { kr: p || "불명확", tone: "neut" as Tone };

// ── CAN SLIM verdict → {kr, tone} (CAN SLIM = 진입) ──
const CS: Record<string, { kr: string; tone: Tone }> = {
  ENTER_NOW: { kr: "즉시진입", tone: "good" },
  WAIT_BREAKOUT: { kr: "돌파대기", tone: "warn" },
  CONSERVATIVE_ONLY: { kr: "보수(SMA50)", tone: "warn" },
  WAIT_EXTENDED: { kr: "과열확장", tone: "warn" },
  HOLD_NO_SIGNAL: { kr: "미탐지", tone: "neut" },
};
const csInfo = (v?: string) => CS[v || "HOLD_NO_SIGNAL"] || { kr: "미탐지", tone: "neut" as Tone };

// ── SEPA stage → tone (SEPA = 과열·스테이지) ──
const sepaTone = (s?: string): Tone =>
  s === "Stage 2" ? "good" : s === "Stage 3" ? "warn" : s === "Stage 4" ? "bad" : "neut";

// ── 실행 배지 (action) → {kr, color, ord} ──
const VERDICT: Record<string, { kr: string; color: string; ord: number }> = {
  ENTER: { kr: "즉시 진입", color: C.green, ord: 0 },
  ARM: { kr: "조건부 진입", color: C.blue, ord: 1 },
  HOLD_OFF: { kr: "진입 보류", color: C.yellow, ord: 2 },
  WATCH: { kr: "관망", color: C.gray, ord: 3 },
  EXIT: { kr: "청산·리스크", color: C.red, ord: 4 },
};
const vInfo = (a?: string) => VERDICT[a || "WATCH"] || VERDICT.WATCH;

interface Row {
  ticker: string;
  action: string;
  ellKr: string; ellTone: Tone;
  csKr: string; csTone: Tone;
  sepa: string; sepaTone: Tone;
  stopTxt: string;
  agLabel: string; agColor: string; agScore: number; agDiv: boolean;
  tones: Tone[];
}

function toRow(idx: any): Row {
  const ph = phaseInfo(idx.current_phase);
  const cs = csInfo((idx.can_slim || {}).verdict);
  const stage = (idx.sepa || {}).stage || "";
  const st = idx.stop || {};
  const sp = st.price;
  const sym = st.currency_symbol || "$";
  const stopTxt =
    sp == null ? "—" : `${sym}${sp >= 1000 ? Number(sp).toLocaleString() : Number(sp).toFixed(2)}`;
  const tones: Tone[] = [ph.tone, cs.tone, sepaTone(stage)];
  const g = tones.filter((t) => t === "good").length;
  const b = tones.filter((t) => t === "bad").length;
  let agLabel: string, agColor: string;
  if (g === 3) { agLabel = "3전략 우호"; agColor = C.green; }
  else if (g === 2 && b === 0) { agLabel = "2우호 합치"; agColor = C.green; }
  else if (g >= 1 && b >= 1) { agLabel = "⚠ 괴리"; agColor = C.yellow; }
  else if (b >= 2) { agLabel = "이탈 우세"; agColor = C.red; }
  else if (b === 1 && g === 0) { agLabel = "약이탈"; agColor = C.red; }
  else { agLabel = "중립·혼조"; agColor = C.gray; }
  return {
    ticker: idx.ticker,
    action: (idx.action_verdict || {}).action || "WATCH",
    ellKr: ph.kr, ellTone: ph.tone,
    csKr: cs.kr, csTone: cs.tone,
    sepa: stage || "—", sepaTone: sepaTone(stage),
    stopTxt,
    agLabel, agColor, agScore: g - b, agDiv: g >= 1 && b >= 1,
    tones,
  };
}

function tally<T>(arr: T[], fn: (x: T) => string): [string, number][] {
  const m: Record<string, number> = {};
  arr.forEach((x) => { const k = fn(x); m[k] = (m[k] || 0) + 1; });
  return Object.entries(m).sort((a, b) => b[1] - a[1]);
}

// ── distribution bar for a group ──
function DistBar({ rows }: { rows: Row[] }) {
  const order = ["ENTER", "ARM", "HOLD_OFF", "WATCH", "EXIT"];
  const counts = order.map((k) => [k, rows.filter((r) => r.action === k).length] as const).filter(([, n]) => n);
  const n = rows.length || 1;
  return (
    <div>
      <div className="flex rounded overflow-hidden" style={{ height: 20, border: `1px solid ${C.border}` }}>
        {counts.map(([k, c]) => (
          <div key={k} title={`${vInfo(k).kr} ${c}`}
            style={{ flex: c, backgroundColor: vInfo(k).color, color: "#fff", fontSize: 10, fontWeight: 700,
              display: "flex", alignItems: "center", justifyContent: "center", minWidth: 0,
              fontVariantNumeric: "tabular-nums" }}>{c}</div>
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1.5" style={{ fontSize: 11, color: C.gray }}>
        {counts.map(([k, c]) => (
          <span key={k} className="inline-flex items-center gap-1">
            <i style={{ width: 8, height: 8, borderRadius: 2, background: vInfo(k).color, display: "inline-block" }} />
            {vInfo(k).kr} <b style={{ color: C.text, fontVariantNumeric: "tabular-nums" }}>{c}</b>
          </span>
        ))}
        <span style={{ color: C.gray }}>· 100% = <b style={{ color: C.text }}>{n}</b>종목</span>
      </div>
    </div>
  );
}

// ── one strategy lens row: count chips ──
function Lens({ icon, name, tallyArr, toneOf }: { icon: string; name: string; tallyArr: [string, number][]; toneOf: (k: string) => Tone }) {
  return (
    <div className="flex items-start gap-2" style={{ fontSize: 11.5 }}>
      <span style={{ color: C.text, fontWeight: 700, whiteSpace: "nowrap", minWidth: 92 }}>{icon} {name}</span>
      <div className="flex flex-wrap gap-1">
        {tallyArr.map(([k, v]) => (
          <span key={k} className="inline-flex items-center gap-1 rounded-full"
            style={{ fontSize: 10.5, padding: "1px 7px", border: `1px solid ${C.border}`, background: C.bgAlt, color: C.gray, whiteSpace: "nowrap" }}>
            <i style={{ width: 7, height: 7, borderRadius: 2, background: TONE[toneOf(k)], display: "inline-block" }} />
            {k} <b style={{ color: C.text }}>{v}</b>
          </span>
        ))}
      </div>
    </div>
  );
}

// ── section synthesis (distribution + lenses + stance) ──
function GroupSynth({ label, rows }: { label: string; rows: Row[] }) {
  const n = rows.length;
  const c = (a: string) => rows.filter((r) => r.action === a).length;
  const div = rows.filter((r) => r.agDiv).length;
  const stage2 = rows.filter((r) => r.sepa === "Stage 2").length;
  const corr = rows.filter((r) => r.ellTone === "bad").length;
  const stance =
    `${n}종목 · 즉시진입 ${c("ENTER")} · 조건부 ${c("ARM")}` +
    (c("HOLD_OFF") ? ` · 보류 ${c("HOLD_OFF")}` : "") +
    ` · 관망 ${c("WATCH")} · 청산 ${c("EXIT")}. ` +
    `SEPA Stage2 ${stage2}/${n}` +
    (corr ? ` · 엘리엇 조정 ${corr}` : "") +
    (div ? ` · ⚠3전략 괴리 ${div}종목(전략 충돌)` : " · 괴리 없음") + ".";
  return (
    <div className="rounded p-3" style={{ background: C.panel, border: `1px solid ${C.border}` }}>
      <div className="flex items-baseline gap-2 mb-2">
        <span style={{ fontWeight: 700, color: C.text, fontSize: 13 }}>{label}</span>
        <span style={{ fontSize: 11, color: C.gray }}>종목 {n}</span>
      </div>
      <DistBar rows={rows} />
      <div className="grid gap-1.5 mt-3">
        <Lens icon="🌊" name="엘리엇" tallyArr={tally(rows, (r) => r.ellKr)} toneOf={(k) => (rows.find((r) => r.ellKr === k)?.ellTone) || "neut"} />
        <Lens icon="📈" name="CAN SLIM" tallyArr={tally(rows, (r) => r.csKr)} toneOf={(k) => (rows.find((r) => r.csKr === k)?.csTone) || "neut"} />
        <Lens icon="🎯" name="SEPA" tallyArr={tally(rows, (r) => r.sepa)} toneOf={(k) => (rows.find((r) => r.sepa === k)?.sepaTone) || "neut"} />
      </div>
      <div className="mt-2.5 rounded" style={{ background: C.bgAlt, borderLeft: `3px solid ${C.claret}`, padding: "8px 11px", fontSize: 12, color: C.text, lineHeight: 1.5 }}>
        {stance}
      </div>
    </div>
  );
}

// ── 합치도 히트맵 (per-ticker) ──
function HeatCell({ label, tone }: { label: string; tone: Tone }) {
  return (
    <div style={{ borderRadius: 3, padding: "5px 4px", textAlign: "center", fontSize: 11, fontWeight: 600,
      color: TONE[tone], background: tint(TONE[tone]), border: `1px solid ${TONE[tone]}66`, lineHeight: 1.2 }}>{label}</div>
  );
}
function Heatmap({ rows }: { rows: Row[] }) {
  const sorted = [...rows].sort((a, b) => b.agScore - a.agScore || Number(a.agDiv) - Number(b.agDiv) || vInfo(a.action).ord - vInfo(b.action).ord);
  const th = { fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase" as const, color: C.gray, fontWeight: 700, padding: "8px 8px", whiteSpace: "nowrap" as const };
  return (
    <div style={{ overflowX: "auto", border: `1px solid ${C.border}`, borderRadius: 4, background: C.panel }}>
      <table style={{ borderCollapse: "collapse", width: "100%", minWidth: 560, fontSize: 12 }}>
        <thead>
          <tr>
            <th style={{ ...th, textAlign: "left" }}>종목</th>
            <th style={{ ...th, textAlign: "center" }}>🌊 엘리엇</th>
            <th style={{ ...th, textAlign: "center" }}>📈 CAN SLIM</th>
            <th style={{ ...th, textAlign: "center" }}>🎯 SEPA</th>
            <th style={{ ...th, textAlign: "center" }}>합치도</th>
            <th style={{ ...th, textAlign: "center" }}>배지</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.ticker} style={r.agDiv ? { background: tint(C.yellow, "12") } : undefined}>
              <td style={{ padding: "5px 8px 5px 10px", borderBottom: `1px solid ${C.bgAlt}` }}>
                <span style={{ fontFamily: "ui-monospace,Menlo,monospace", fontWeight: 700, fontSize: 12.5 }}>{r.ticker}</span>
              </td>
              <td style={{ padding: "4px 6px", borderBottom: `1px solid ${C.bgAlt}` }}><HeatCell label={r.ellKr} tone={r.ellTone} /></td>
              <td style={{ padding: "4px 6px", borderBottom: `1px solid ${C.bgAlt}` }}><HeatCell label={r.csKr} tone={r.csTone} /></td>
              <td style={{ padding: "4px 6px", borderBottom: `1px solid ${C.bgAlt}` }}><HeatCell label={r.sepa} tone={r.sepaTone} /></td>
              <td style={{ padding: "4px 6px", borderBottom: `1px solid ${C.bgAlt}`, textAlign: "center" }}>
                <span className="inline-flex items-center gap-1.5" style={{ fontSize: 11, fontWeight: 700, color: r.agColor, whiteSpace: "nowrap" }}>
                  <span className="inline-flex gap-0.5">
                    {r.tones.map((t, i) => <i key={i} style={{ width: 8, height: 13, borderRadius: 2, background: TONE[t] }} />)}
                  </span>
                  {r.agLabel}
                </span>
              </td>
              <td style={{ padding: "4px 6px", borderBottom: `1px solid ${C.bgAlt}`, textAlign: "center" }}>
                <span style={{ fontSize: 10.5, fontWeight: 700, color: vInfo(r.action).color, border: `1px solid ${vInfo(r.action).color}`, borderRadius: 20, padding: "2px 8px", whiteSpace: "nowrap" }}>{vInfo(r.action).kr}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function StrategyIntegratedSummary({ indices }: { indices: any[] }) {
  const [open, setOpen] = useState(true);
  const [heatAsset, setHeatAsset] = useState<"stock" | "etf">("stock");

  const { stocks, etfs } = useMemo(() => {
    const s: Row[] = [], e: Row[] = [];
    (indices || []).forEach((idx) => {
      const r = toRow(idx);
      (String(idx.asset_type || "").toLowerCase() === "etf" ? e : s).push(r);
    });
    return { stocks: s, etfs: e };
  }, [indices]);

  if (!stocks.length && !etfs.length) return null;
  const heatRows = heatAsset === "etf" ? etfs : stocks;

  return (
    <div className="rounded mb-3" style={{ background: C.bgAlt, border: `1px solid ${C.border}` }}>
      <div className="flex items-center gap-2 px-3 py-2" style={{ borderBottom: open ? `1px solid ${C.border}` : "none" }}>
        <span style={{ fontSize: 13, fontWeight: 700, color: C.text }}>📊 3전략 통합 요약 · 합치도</span>
        <span style={{ fontSize: 11, color: C.gray }}>엘리엇=손절 · CAN SLIM=진입 · SEPA=과열</span>
        <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
          className="ml-auto rounded px-2 py-0.5" style={{ fontSize: 12, background: C.panel, border: `1px solid ${C.border}`, color: C.gray }}>
          {open ? "▾" : "▸"}
        </button>
      </div>

      {open && (
        <div className="p-3 grid gap-3">
          {/* 섹션 종합 — STOCKS / ETFs */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {stocks.length > 0 && <GroupSynth label="STOCKS — 개별 종목" rows={stocks} />}
            {etfs.length > 0 && <GroupSynth label="ETFs — 상장지수펀드" rows={etfs} />}
          </div>

          {/* 합치도 히트맵 (자산 토글) */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span style={{ fontSize: 12, fontWeight: 700, color: C.text }}>🔥 합치도 히트맵</span>
              <span style={{ fontSize: 11, color: C.gray }}>3전략 합치·괴리 · 합치도순</span>
              <div className="ml-auto flex gap-1">
                {([["stock", `Stock ${stocks.length}`], ["etf", `ETF ${etfs.length}`]] as const).map(([k, lbl]) => {
                  const active = heatAsset === k;
                  return (
                    <button key={k} type="button" onClick={() => setHeatAsset(k)}
                      className="rounded px-2.5 py-0.5" style={{ fontSize: 11, fontWeight: 600,
                        background: active ? C.blue : C.panel, color: active ? "#fff" : C.gray,
                        border: `1px solid ${active ? C.blue : C.border}` }}>{lbl}</button>
                  );
                })}
              </div>
            </div>
            {heatRows.length > 0 ? <Heatmap rows={heatRows} />
              : <div style={{ fontSize: 12, color: C.gray, padding: "8px 4px" }}>해당 자산군 종목이 없습니다.</div>}
            <div className="mt-1.5" style={{ fontSize: 10.5, color: C.gray, lineHeight: 1.5 }}>
              <span style={{ color: C.green }}>■</span> 우호 · <span style={{ color: C.yellow }}>■</span> 경계 · <span style={{ color: C.red }}>■</span> 이탈 · <span style={{ color: C.gray }}>■</span> 중립 —
              3칸 모두 우호=고확신, 혼색(⚠괴리, 앰버 하이라이트)=전략 충돌.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
