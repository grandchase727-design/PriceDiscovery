/**
 * ElliottWaveCard — the reusable per-index/ETF Elliott-wave card.
 *
 * Extracted verbatim from ElliottWavePanel.tsx so multiple panels (the 4-core
 * indices panel and the buy-Final-List ETF panel) can render the SAME card
 * shape from the same per-index contract dict.
 *
 * The card renders: title row (ticker/name + YTD%/고점%/현재가), a colored
 * PhaseBadge, the 🎯 진입가 / 🌊 손절가 mini-grid, the close-line <Plot> with
 * labeled swing pivots, and the Korean interpretation.
 */
import Plot from "react-plotly.js";
import { C, DARK_LAYOUT } from "../../styles/theme";

// CAN SLIM base_pattern enum → Korean-friendly display label
const BASE_PATTERN_LABEL: Record<string, string> = {
  flat_base: "Flat Base",
  cup_with_handle: "Cup with Handle",
  double_bottom: "Double Bottom",
};

// phase_color enum → theme color
const PHASE_C: Record<string, string> = {
  green: C.green,
  cyan: C.cyan,
  blue: C.blue,
  amber: C.yellow,
  red: C.red,
  purple: C.purple,
  gray: C.gray,
};

// idx.category (매수 Final List 분류) → emoji/label/color — mirrors FinalListPanel's
// 📂 분류 column (catMeta) so the same ticker reads identically in both places.
const CATEGORY_META: Record<string, { emoji: string; label: string; color: string }> = {
  ENTERED:      { emoji: "✓",  label: "오늘 진입", color: C.green },
  HOLDING:      { emoji: "🔵", label: "보유 중",   color: C.cyan },
  EXIT_PENDING: { emoji: "⚠",  label: "청산 후보", color: C.yellow },
  NEW:          { emoji: "🟢", label: "신규 후보", color: "#0A7D3F" },
};

// can_slim.verdict_color enum → theme color
const VERDICT_C: Record<string, string> = {
  green: C.green,
  amber: C.yellow,
  red: C.red,
  gray: C.gray,
};

// action_verdict.color (실행도구 관점 판정) enum → theme color
const ACTION_C: Record<string, string> = {
  green: C.green,
  blue: C.blue,
  amber: C.yellow,
  red: C.red,
  gray: C.gray,
};

// action_verdict.action → prefix emoji (실행 판정 상태)
const ACTION_EMOJI: Record<string, string> = {
  ENTER: "🟢",    // 즉시 진입 실행
  ARM: "🔵",      // 조건부 진입 (트리거 대기)
  HOLD: "🟠",     // 보유 유지 (추가 보류, 과열)
  HOLD_OFF: "🟠", // 진입 보류 (과열 veto)
  WATCH: "⚪",    // 관망 (신호 대기)
  EXIT: "🔴",     // 청산·리스크 관리
};

function fmtPct(v: number | null | undefined, signed = true): string {
  if (v == null || !isFinite(v)) return "—";
  return `${signed && v >= 0 ? "+" : ""}${v.toFixed(1)}%`;
}

function fmtPrice(v: number | null | undefined, sym = "$"): string {
  if (v == null || !isFinite(v)) return "—";
  return `${sym}${v.toFixed(2)}`;
}

// stop pct → color (only for actionable rows): ≥-3% red, ≥-8% amber, else green
function stopPctColor(pct: number | null | undefined): string {
  if (pct == null || !isFinite(pct)) return C.gray;
  if (pct >= -3) return C.red;
  if (pct >= -8) return C.yellow;
  return C.green;
}

// The backend prefixes actionable primary_labels with "🎯 진입가 "; the cell already
// shows that emoji as its heading, so strip the redundant prefix for the value line.
function cleanEntryLabel(label: string | null | undefined): string {
  if (!label) return "—";
  return label.replace(/^🎯\s*진입가\s*/, "").trim() || label;
}

function PhaseBadge({ label, color }: { label: string; color: string }) {
  return (
    <span
      style={{
        color,
        backgroundColor: color + "18",
        border: `1px solid ${color}66`,
        borderRadius: 4,
        padding: "2px 8px",
        fontSize: 11,
        fontWeight: "bold",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}

export function IndexCard({ idx }: { idx: any }) {
  const phaseColor = PHASE_C[idx.phase_color] || C.gray;
  const dates: string[] = idx.dates || [];
  const close: number[] = idx.close || [];
  const pivots: any[] = idx.pivots || [];

  // Labeled pivots only (skip label === "") for the marker+text overlay
  const labeled = pivots.filter((p) => p && p.label);
  const pivColor = (label: string): string => {
    if (/^[1-5]$/.test(label)) return C.green; // numeric impulse
    if (/^[ABC]$/.test(label)) return C.red; // ABC corrective
    return C.gray;
  };

  const ytd = idx.ytd_return_pct as number | null | undefined;
  const fromHigh = idx.from_high_pct as number | null | undefined;
  // period_label: "YTD" or "1개월" (backend PERIOD_CONFIG) — falls back to "YTD"
  // for any older cached payload that predates the period toggle.
  const periodLabel: string = idx.period_label || "YTD";

  // Entry / Stop sub-dicts (contract-guaranteed shape; defensive fallbacks anyway)
  const entry = idx.entry || {
    actionable: false,
    primary: null,
    primary_label: "관망",
    zone_low: null,
    zone_high: null,
    rationale: "",
  };
  const stop = idx.stop || { price: null, pct: null, type: "NONE", rationale: "" };
  const entryActionable = entry.actionable === true;
  // Currency symbol: prefer the stop dict's, then the index dict's, default "$"
  const sym: string = stop.currency_symbol || idx.currency_symbol || "$";

  const hasSeries = dates.length > 0 && close.length > 0;

  return (
    <div
      className="rounded-lg p-3"
      style={{ backgroundColor: C.panel, border: `1px solid ${C.border}` }}
    >
      {/* Title row: ticker + name + phase badge + returns */}
      <div className="flex items-center gap-2 flex-wrap mb-1">
        <span className="font-mono font-bold text-[14px]" style={{ color: C.text }}>
          {idx.ticker}
        </span>
        <span className="text-[12px]" style={{ color: C.gray }}>
          {idx.name}
        </span>
        {idx.action_verdict && idx.action_verdict.label && (() => {
          const ac = ACTION_C[idx.action_verdict.color] || C.gray;
          return (
            <span
              style={{
                color: "#fff",
                backgroundColor: ac,
                border: `1px solid ${ac}`,
                borderRadius: 4,
                padding: "2px 10px",
                fontSize: 12,
                fontWeight: "bold",
                whiteSpace: "nowrap",
              }}
              title={idx.action_verdict.reason || ""}
            >
              {ACTION_EMOJI[idx.action_verdict.action] || ""} {idx.action_verdict.label}
            </span>
          );
        })()}
        {/* Trend Freshness — 추세 나이(50일선 위 연속일). 종목명 옆 배지. 표시 전용. */}
        {idx.trend_freshness === "FRESH" && (
          <span
            title={`신선 추세 — 강세 전환 후 ${idx.trend_age ?? "?"}일 (남은 여력 상대적 우위)`}
            style={{
              fontSize: 10, fontWeight: "bold", color: C.green,
              backgroundColor: C.green + "18", border: `1px solid ${C.green}66`,
              borderRadius: 3, padding: "1px 5px", whiteSpace: "nowrap",
            }}
          >
            🌱 {idx.trend_age}d
          </span>
        )}
        {idx.trend_freshness === "MATURE" && (
          <span
            title={`성숙 추세 — 50일선 위 ${idx.trend_age ?? "?"}일 연속 (추세 노후 — 감쇠/되돌림 유의)`}
            style={{
              fontSize: 10, fontWeight: "bold", color: C.gray,
              backgroundColor: C.gray + "18", border: `1px solid ${C.gray}66`,
              borderRadius: 3, padding: "1px 5px", whiteSpace: "nowrap",
            }}
          >
            🧓 {idx.trend_age}d
          </span>
        )}
        {idx.category && CATEGORY_META[idx.category] && (
          <span
            style={{
              display: "inline-block",
              fontSize: 10,
              fontWeight: "bold",
              color: CATEGORY_META[idx.category].color,
              backgroundColor: CATEGORY_META[idx.category].color + "18",
              border: `1px solid ${CATEGORY_META[idx.category].color}66`,
              borderRadius: 3,
              padding: "1px 5px",
            }}
            title="매수 Final List 내 분류"
          >
            {CATEGORY_META[idx.category].emoji} {CATEGORY_META[idx.category].label}
          </span>
        )}
        <span className="ml-auto flex items-center gap-2">
          <span
            className="font-mono font-bold text-[12px]"
            style={{ color: (ytd ?? 0) >= 0 ? C.green : C.red }}
            title={`${periodLabel} 수익률 (분석기간 시작 대비)`}
          >
            {periodLabel} {fmtPct(ytd)}
          </span>
          <span className="font-mono text-[11px]" style={{ color: C.gray }} title={`${periodLabel} 고점 대비`}>
            고점 {fmtPct(fromHigh)}
          </span>
          {idx.current_price != null && isFinite(idx.current_price) && (
            <span
              className="font-mono text-[11px]"
              style={{ color: C.gray }}
              title="현재가 (종가 기준)"
            >
              현재 {fmtPrice(idx.current_price, sym)}
            </span>
          )}
        </span>
      </div>
      <div className="mb-1">
        <PhaseBadge label={idx.phase_label || idx.current_phase || "—"} color={phaseColor} />
      </div>
      {idx.action_verdict && idx.action_verdict.reason && (
        <div className="mb-1 text-[11px]" style={{ color: C.gray, lineHeight: 1.4 }}>
          ⚡ {idx.action_verdict.reason}
        </div>
      )}

      {/* 전략별 시사점 & 배지 도출 근거 (엘리엇=손절·구조 / CAN SLIM=진입 / SEPA=과열 veto) */}
      {idx.action_verdict && idx.action_verdict.commentary && (
        <div
          className="mb-2 rounded px-2 py-1.5 text-[11px]"
          style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, lineHeight: 1.5 }}
        >
          <div style={{ color: C.gray, fontWeight: 600, marginBottom: 3, opacity: 0.85 }}>
            전략별 시사점 · 배지 근거
          </div>
          {idx.action_verdict.commentary.elliott && (
            <div style={{ color: C.gray }}>🌊 {idx.action_verdict.commentary.elliott}</div>
          )}
          {idx.action_verdict.commentary.can_slim && (
            <div style={{ color: C.gray }}>📈 {idx.action_verdict.commentary.can_slim}</div>
          )}
          {idx.action_verdict.commentary.sepa && (
            <div style={{ color: C.gray }}>🎯 {idx.action_verdict.commentary.sepa}</div>
          )}
          {idx.action_verdict.commentary.derivation && (
            <div
              style={{
                color: C.text,
                fontWeight: 600,
                marginTop: 4,
                paddingTop: 4,
                borderTop: `1px dashed ${C.border}`,
              }}
            >
              🏷 {idx.action_verdict.commentary.derivation}
            </div>
          )}
        </div>
      )}

      {/* 진입가 / 손절가 mini-grid (Elliott entry + core stop) */}
      <div className="grid grid-cols-2 gap-2 mb-2">
        {/* 🎯 진입가 */}
        <div
          className="rounded px-2 py-1"
          style={{
            backgroundColor: C.bgAlt,
            border: `1px solid ${entryActionable ? C.green + "66" : C.border}`,
          }}
          title={entry.rationale || ""}
        >
          <div
            className="uppercase tracking-wide"
            style={{ fontSize: 10, color: C.gray }}
          >
            🎯 진입가
          </div>
          {entryActionable && entry.primary != null && isFinite(entry.primary) ? (
            <>
              <div
                className="font-mono font-bold"
                style={{ fontSize: 13, color: C.green }}
              >
                {fmtPrice(entry.primary, sym)}
              </div>
              <div style={{ fontSize: 10, color: C.gray, lineHeight: 1.3 }}>
                {cleanEntryLabel(entry.primary_label)}
              </div>
              {entry.zone_low != null &&
                entry.zone_high != null &&
                isFinite(entry.zone_low) &&
                isFinite(entry.zone_high) && (
                  <div className="font-mono" style={{ fontSize: 10, color: C.gray }}>
                    {fmtPrice(entry.zone_low, sym)}–{fmtPrice(entry.zone_high, sym)}
                  </div>
                )}
            </>
          ) : (
            <div style={{ fontSize: 13, color: C.gray, lineHeight: 1.3 }}>
              {entry.primary_label || "관망"}
            </div>
          )}
        </div>

        {/* 🌊 손절가 */}
        <div
          className="rounded px-2 py-1"
          style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}
          title={stop.rationale || ""}
        >
          <div
            className="uppercase tracking-wide"
            style={{ fontSize: 10, color: C.gray }}
          >
            🌊 손절가 <span style={{ textTransform: "none", fontStyle: "italic" }}>· 엘리엇 손절 도구</span>
          </div>
          {stop.price != null && isFinite(stop.price) ? (
            <div
              className="font-mono font-bold"
              style={{
                fontSize: 13,
                color: entryActionable ? stopPctColor(stop.pct) : C.yellow,
              }}
            >
              {fmtPrice(stop.price, sym)}
              {stop.pct != null && isFinite(stop.pct) && (
                <span className="ml-1" style={{ fontSize: 11 }}>
                  ({fmtPct(stop.pct)})
                </span>
              )}
            </div>
          ) : (
            <div style={{ fontSize: 12, color: C.gray, lineHeight: 1.3 }}>
              손절 계산 불가
            </div>
          )}
        </div>
      </div>

      {/* 📐 CAN SLIM (William O'Neil) — only present on the buy-Final-List ETF panel */}
      {idx.can_slim && (
        <div
          style={{
            borderTop: `1px dashed ${C.border}`,
            paddingTop: 8,
            marginTop: 4,
            marginBottom: 8,
          }}
        >
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="font-bold" style={{ fontSize: 11, color: C.claret }}>
              📐 CAN SLIM (William O'Neil)
            </span>
            <span style={{ fontSize: 9, color: C.gray, fontStyle: "italic" }}>진입 실행 도구</span>
            {idx.can_slim.verdict_label && (
              <span
                style={{
                  fontSize: 11,
                  fontWeight: "bold",
                  color: VERDICT_C[idx.can_slim.verdict_color] || C.gray,
                  backgroundColor: (VERDICT_C[idx.can_slim.verdict_color] || C.gray) + "18",
                  border: `1px solid ${VERDICT_C[idx.can_slim.verdict_color] || C.gray}66`,
                  borderRadius: 4,
                  padding: "2px 8px",
                  whiteSpace: "nowrap",
                }}
              >
                {idx.can_slim.verdict_label}
              </span>
            )}
            {(idx.can_slim.base_pattern || idx.can_slim.base_quality) && (
              <span style={{ fontSize: 11, color: C.gray }}>
                {BASE_PATTERN_LABEL[idx.can_slim.base_pattern as string] || idx.can_slim.base_pattern || "—"}
                {idx.can_slim.base_quality ? ` · Quality ${idx.can_slim.base_quality}` : ""}
              </span>
            )}
          </div>

          {/* 3-column entry-tier mini-grid (aggressive / primary / conservative) */}
          {(idx.can_slim.aggressive || idx.can_slim.primary || idx.can_slim.conservative) && (
            <div className="grid grid-cols-3 gap-2 mb-2">
              {idx.can_slim.aggressive && (
                <div
                  className="rounded px-2 py-1"
                  style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}
                  title={idx.can_slim.aggressive.rationale || ""}
                >
                  <div className="uppercase tracking-wide" style={{ fontSize: 10, color: C.gray }}>
                    즉시
                  </div>
                  <div className="font-mono font-bold" style={{ fontSize: 13, color: C.text }}>
                    {fmtPrice(idx.can_slim.aggressive.price, sym)}
                  </div>
                </div>
              )}
              {idx.can_slim.primary && (
                <div
                  className="rounded px-2 py-1"
                  style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}
                  title={idx.can_slim.primary.rationale || ""}
                >
                  <div className="flex items-center gap-1 uppercase tracking-wide" style={{ fontSize: 10, color: C.gray }}>
                    PRIMARY
                    {idx.can_slim.primary.status === "await_breakout" && (
                      <span
                        className="rounded px-1"
                        style={{ fontSize: 9, color: C.yellow, backgroundColor: C.yellow + "18", border: `1px solid ${C.yellow}66` }}
                      >
                        돌파대기
                      </span>
                    )}
                  </div>
                  <div
                    className="font-mono font-bold"
                    style={{
                      fontSize: 13,
                      color: idx.can_slim.primary.status === "actionable" ? C.green : C.text,
                    }}
                  >
                    {fmtPrice(idx.can_slim.primary.price, sym)}
                  </div>
                </div>
              )}
              {idx.can_slim.conservative && (
                <div
                  className="rounded px-2 py-1"
                  style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}` }}
                  title={idx.can_slim.conservative.rationale || ""}
                >
                  <div className="uppercase tracking-wide" style={{ fontSize: 10, color: C.gray }}>
                    CONSERVATIVE
                  </div>
                  <div className="font-mono font-bold" style={{ fontSize: 13, color: C.text }}>
                    {fmtPrice(idx.can_slim.conservative.price, sym)}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* O'Neil stop / R-R / volume confirmation chips */}
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span
              className="rounded px-2 py-1 font-mono"
              style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, fontSize: 11, color: C.red }}
              title="O'Neil 7% cut-loss stop"
            >
              O'Neil 손절 {fmtPrice(idx.can_slim.oneil_stop, sym)}
            </span>
            <span
              className="rounded px-2 py-1 font-mono"
              style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, fontSize: 11, color: C.text }}
              title="Reward/Risk ratio"
            >
              R/R {idx.can_slim.rr_ratio != null && isFinite(idx.can_slim.rr_ratio) ? idx.can_slim.rr_ratio.toFixed(2) : "—"}
            </span>
            <span
              className="rounded px-2 py-1 font-mono"
              style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, fontSize: 11, color: C.text }}
              title="거래량 비율 (평균 대비) / 확인 여부"
            >
              거래량 {idx.can_slim.volume_ratio != null && isFinite(idx.can_slim.volume_ratio) ? `${idx.can_slim.volume_ratio.toFixed(2)}x` : "—"}{" "}
              <span style={{ color: idx.can_slim.volume_confirmed ? C.green : C.red }}>
                {idx.can_slim.volume_confirmed ? "✓" : "✗"}
              </span>
            </span>
          </div>

          {/* Korean commentary */}
          {idx.can_slim.commentary && (
            <div style={{ fontSize: 12, color: C.text, lineHeight: 1.45 }}>
              {idx.can_slim.commentary}
            </div>
          )}
        </div>
      )}

      {/* 📈 Minervini SEPA — Trend Template + VCP + Stage */}
      {idx.sepa && idx.sepa.sepa_score != null && (
        <div
          style={{
            borderTop: `1px dashed ${C.border}`,
            paddingTop: 8,
            marginTop: 4,
            marginBottom: 8,
          }}
        >
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className="font-bold" style={{ fontSize: 11, color: C.blue }}>
              📈 Minervini SEPA
            </span>
            <span style={{ fontSize: 9, color: C.gray, fontStyle: "italic" }}>과열/스테이지 체크</span>
            {idx.sepa.verdict_label && (
              <span
                style={{
                  fontSize: 11,
                  fontWeight: "bold",
                  color: VERDICT_C[idx.sepa.verdict_color] || C.gray,
                  backgroundColor: (VERDICT_C[idx.sepa.verdict_color] || C.gray) + "18",
                  border: `1px solid ${VERDICT_C[idx.sepa.verdict_color] || C.gray}66`,
                  borderRadius: 4,
                  padding: "2px 8px",
                  whiteSpace: "nowrap",
                }}
              >
                {idx.sepa.verdict_label}
              </span>
            )}
            <span style={{ fontSize: 11, color: C.gray }}>
              {idx.sepa.stage} · SEPA {Math.round(idx.sepa.sepa_score)}
            </span>
          </div>

          {/* Trend Template criteria chips (✓ pass / ✗ fail / – unknown) */}
          {idx.sepa.trend_template?.criteria && (
            <div className="flex items-center gap-1 flex-wrap mb-1">
              {idx.sepa.trend_template.criteria.map((c: any, i: number) => {
                const col =
                  c.pass === true ? C.green : c.pass === false ? C.red : C.gray;
                const mark = c.pass === true ? "✓" : c.pass === false ? "✗" : "–";
                return (
                  <span
                    key={i}
                    title={c.label}
                    style={{
                      fontSize: 10,
                      color: col,
                      backgroundColor: col + "14",
                      border: `1px solid ${col}44`,
                      borderRadius: 3,
                      padding: "1px 5px",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {mark} {c.label}
                  </span>
                );
              })}
            </div>
          )}

          {/* VCP + template pass-count */}
          <div className="flex items-center gap-2 flex-wrap">
            {idx.sepa.vcp && (
              <span
                className="rounded px-2 py-1"
                style={{
                  backgroundColor: C.bgAlt,
                  border: `1px solid ${C.border}`,
                  fontSize: 11,
                  color: VERDICT_C[idx.sepa.vcp.color] || C.gray,
                }}
                title="Volatility Contraction Pattern"
              >
                VCP {idx.sepa.vcp.status} · {idx.sepa.vcp.label}
              </span>
            )}
            {idx.sepa.trend_template && (
              <span style={{ fontSize: 10, color: C.gray }}>
                추세템플릿 {idx.sepa.trend_template.n_pass}/{idx.sepa.trend_template.n_known} 충족
              </span>
            )}
          </div>
        </div>
      )}

      {/* Close line + labeled pivots overlay */}
      {hasSeries ? (
        <Plot
          data={[
            {
              type: "scatter",
              mode: "lines",
              x: dates,
              y: close,
              line: { color: C.blue, width: 2 },
              hovertemplate: "%{x}<br>%{y:.2f}<extra></extra>",
            },
            {
              type: "scatter",
              mode: "markers+text",
              x: labeled.map((p) => p.date),
              y: labeled.map((p) => p.price),
              text: labeled.map((p) => p.label),
              textposition: "top center" as const,
              textfont: { size: 11, color: C.text },
              marker: {
                size: 9,
                color: labeled.map((p) => pivColor(p.label)),
                line: { width: 1, color: C.panel },
              },
              hovertemplate: "%{text}: %{x}<br>%{y:.2f}<extra></extra>",
            },
          ]}
          layout={{
            ...DARK_LAYOUT,
            height: 240,
            showlegend: false,
            margin: { t: 24, b: 28, l: 40, r: 12 },
            title: { text: idx.ticker, font: { size: 13, color: C.gray } },
          }}
          config={{ displayModeBar: false }}
          style={{ width: "100%" }}
        />
      ) : (
        <div
          className="rounded flex items-center justify-center text-[12px]"
          style={{ height: 240, backgroundColor: C.bgAlt, color: C.gray, border: `1px solid ${C.border}` }}
        >
          가격 데이터 없음
        </div>
      )}

      {/* Korean interpretation */}
      {idx.interpretation && (
        <div className="mt-1 text-[12px]" style={{ color: C.text, lineHeight: 1.45 }}>
          {idx.interpretation}
        </div>
      )}
    </div>
  );
}
