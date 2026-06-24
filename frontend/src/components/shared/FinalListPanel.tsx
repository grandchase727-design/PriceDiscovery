/**
 * FinalListPanel — Buy/Sell Final List synthesized from all 4 signal layers.
 *
 * Renders at the bottom of the dashboard. Tabs between BUY and SELL.
 */
import { useEffect, useState } from "react";
import {
  fetchFinalList,
  type FinalListItem,
  type FinalListResponse,
  type ActivePositionItem,
  type ExitPendingItem,
} from "../../api/client";
import { ColDefToggle } from "./ColDefToggle";

// FT (Financial Times) palette — light pink paper, near-black text
const C = {
  bg: "#FFF1E5", bgAlt: "#FBF2E9", border: "#E6D9CE",
  text: "#33302E", gray: "#66605C",
  green: "#0A7D3F", amber: "#B85C00", red: "#CC0000",
  cyan: "#0D7680", purple: "#7D5BA6",
};

const URGENCY_EMOJI: Record<string, string> = {
  URGENT: "🚨", NORMAL: "▶", PATIENT: "⏸",
};

const ACTION_META: Record<string, { color: string; label: string }> = {
  EXECUTE_TODAY:  { color: C.green,  label: "오늘 실행 ✓" },
  CLOSE_TODAY:    { color: C.red,    label: "오늘 청산 ⚠" },
  WATCH_TOMORROW: { color: C.amber,  label: "내일 재확인 ⏳" },
  ALREADY_HELD:   { color: C.cyan,   label: "이미 보유 ▶" },
  OBSERVE:        { color: C.gray,   label: "관찰만" },
};

const HORIZON_EMOJI: Record<string, string> = {
  tactical: "🚀", core: "⚓", strategic: "🌐",
};

function Stars({ n }: { n: number }) {
  return (
    <span style={{ color: n >= 3 ? C.green : n >= 2 ? C.amber : C.gray, fontWeight: "bold" }}>
      {"★".repeat(n)}{"☆".repeat(3 - n)}
    </span>
  );
}

function Pill({ children, color, bg }: { children: React.ReactNode; color: string; bg?: string }) {
  return (
    <span style={{
      color, backgroundColor: bg || color + "22",
      border: `1px solid ${color}66`,
      borderRadius: 3, padding: "1px 6px", fontSize: 11, fontWeight: "bold",
      whiteSpace: "nowrap",
    }}>{children}</span>
  );
}

// ─────────────────────────────────────────────────────────────────
// 포지션 상태 해석 — State machine 상태를 사용자 친화적 한국어 commentary로 변환
//
// State machine (position_state.py):
//   PROSPECTING  → 관찰 중 (BUY_NOW 2일 연속 시 진입, SKIP 3일 시 탈락)
//   ENTERED      → 오늘 진입 (1일 transient → HOLDING)
//   HOLDING      → 보유 중 (sticky; SKIP 2일 또는 WAIT 5일 시 청산)
//   EXIT_PENDING → 청산 후보 (1일 transient → EXITED)
//   EXITED       → 청산 완료 (terminal)
//   DROPPED      → 후보 탈락 (PROSPECTING SKIP 3일)
// ─────────────────────────────────────────────────────────────────
function interpretPosition(r: any): { emoji: string; color: string; headline: string; detail: string } {
  const state = (r.state || "").toUpperCase();
  const sigDays = r.signal_days ?? 0;
  const daysHeld = r.days_held ?? 0;
  const urgency = (r.urgency || "NORMAL").toUpperCase();
  const action = (r.action || "").toUpperCase();
  const promotedShort = !!r.promoted_from_short;

  // Urgency 부가 설명
  const urgencyNote =
    urgency === "URGENT"  ? "긴급 — 즉시 대응 필요" :
    urgency === "PATIENT" ? "여유 — 서두를 필요 없음" :
    "보통 — 평상 모니터링";

  switch (state) {
    case "PROSPECTING": {
      const remain = Math.max(0, 2 - sigDays);
      return {
        emoji: "🔍",
        color: C.amber,
        headline: "관찰 중 (미진입)",
        detail: sigDays >= 2
          ? "BUY_NOW 2일 연속 충족 — 곧 진입 전환 예정."
          : remain === 1
            ? `BUY_NOW ${sigDays}/2일 — 1일 더 매수 신호 지속 시 진입. ${urgencyNote}.`
            : `아직 매수 신호 미확정 (${sigDays}/2일). 내일 재확인 권장. ${urgencyNote}.`,
      };
    }
    case "ENTERED":
      return {
        emoji: "✅",
        color: C.green,
        headline: "오늘 신규 진입",
        detail: "매수 체결 — 신규 포지션 개시. 익일 '보유 중(HOLDING)'으로 전환.",
      };
    case "HOLDING": {
      const inToday = r.in_today_buy_picks || r.in_today_picks;
      return {
        emoji: "📈",
        color: C.cyan,
        headline: `보유 중 (${daysHeld}일째)`,
        detail: inToday
          ? `${daysHeld}일 보유 — 오늘도 매수 신호 유지, 추세 지속. ${urgencyNote}.`
          : `${daysHeld}일 보유 — 오늘은 신규 매수 신호 없으나 보유 유지 (sticky). 신호 약화 모니터링.`,
      };
    }
    case "EXIT_PENDING":
      return {
        emoji: "⚠️",
        color: C.red,
        headline: "청산 후보",
        detail: promotedShort
          ? `보유 중 SHORT 신호 감지 (regime-flip) — 청산 권장. ${urgencyNote}.`
          : `청산 신호 감지 (SKIP 누적 또는 추세 약화). 청산 카운트다운 진행. ${urgencyNote}.`,
      };
    case "EXITED":
      return {
        emoji: "🔚",
        color: C.gray,
        headline: "청산 완료",
        detail: "포지션 종료 — 더 이상 보유 안 함. 재진입은 신규 base 패턴 확인 후 권장 (관찰만).",
      };
    case "DROPPED":
      return {
        emoji: "✗",
        color: C.gray,
        headline: "후보 탈락",
        detail: "관찰 중 매도(SKIP) 신호 3일 누적 — watchlist에서 제외됨.",
      };
    default: {
      // 신규 후보 (state machine 미진입) — buy_list NEW 등
      // 보유-인지 재정렬 정보가 있으면 분산 기여도를 함께 표시
      const ha = r.ha_rationale;
      const divW = r.ha_diversification_weight;
      const corrPen = r.ha_correlation_penalty;
      const isDiversifier = (divW ?? 1) >= 0.9 && (corrPen ?? 1) >= 0.9;
      const isCrowded = (divW ?? 1) <= 0.5 || (corrPen ?? 1) <= 0.55;
      const baseDetail = action === "EXECUTE_TODAY"
        ? "오늘 voting 통과 — 신규 매수 후보."
        : "신규 후보 — 아직 포지션 미보유.";
      return {
        emoji: isDiversifier ? "🟢" : isCrowded ? "🟡" : "🟢",
        color: isDiversifier ? C.green : isCrowded ? C.amber : C.gray,
        headline: isDiversifier ? "신규 후보 (분산 ✓)"
                : isCrowded ? "신규 후보 (집중 ⚠)"
                : (state || "신규 후보"),
        detail: ha ? `${baseDetail} ${ha}` : baseDetail,
      };
    }
  }
}

// Tier definitions — used to inject section header rows between star groups
const TIER_META: Record<number, { label: string; sublabel: string; color: string; bg: string }> = {
  3: {
    label: "✅ 3-Agent UNANIMOUS / MAJORITY_CLEAN",
    sublabel: "PM + Trading + Risk 모두 APPROVE (or 2 APPROVE + 1 CAUTION) — 가장 합의된 후보",
    color: "#0A7D3F",
    bg: "rgba(22,199,132,0.10)",
  },
  2: {
    label: "✓ MAJORITY_DISSENT or SOLO_CLEAN",
    sublabel: "2 APPROVE + 1 REJECT (불일치) 또는 1 APPROVE + 2 CAUTION (단독 청신호)",
    color: "#0D7680",
    bg: "rgba(34,184,207,0.10)",
  },
  1: {
    label: "⚠ SOLO_DISSENT / ALL_CAUTION",
    sublabel: "단 1개 APPROVE + 1 REJECT 또는 모든 agent CAUTION — 약한 신호, 신중히",
    color: "#B85C00",
    bg: "rgba(241,176,62,0.08)",
  },
};

function TierHeaderRow({ stars, count, color, bg, label, sublabel, colSpan }:
  { stars: number; count: number; color: string; bg: string; label: string; sublabel: string; colSpan: number }) {
  return (
    <tr style={{ backgroundColor: bg, borderTop: `2px solid ${color}`, borderBottom: `1px solid ${color}40` }}>
      <td colSpan={colSpan} className="px-2 py-1.5">
        <div className="flex items-center gap-2">
          <span style={{ color, fontSize: 16, fontWeight: "bold" }}>
            {"★".repeat(stars)}{"☆".repeat(3 - stars)}
          </span>
          <span style={{ color, fontSize: 13, fontWeight: "bold" }}>{label}</span>
          <span style={{ color: C.gray, fontSize: 12 }}>({count}개)</span>
          <span style={{ marginLeft: "auto", color: C.gray, fontSize: 11, fontStyle: "italic" }}>
            {sublabel}
          </span>
        </div>
      </td>
    </tr>
  );
}

function FinalListTable({ items, mode, categoryCommentary, typeFilter = "ALL" }: {
  items: FinalListItem[];
  mode: "buy" | "sell";
  categoryCommentary?: {
    entered?:      { commentary: string; cached?: boolean; n_items?: number };
    exit_pending?: { commentary: string; cached?: boolean; n_items?: number };
    holding?:      { commentary: string; cached?: boolean; n_items?: number };
    new?:          { commentary: string; cached?: boolean; n_items?: number };
  };
  typeFilter?: "ALL" | "STOCK" | "ETF";
}) {
  if (items.length === 0) {
    return (
      <div className="rounded p-4 text-center text-[14px]" style={{
        backgroundColor: C.bgAlt, color: C.gray, border: `1px solid ${C.border}`,
      }}>
        조건을 만족하는 종목이 없습니다 — 내일 swarm 실행 후 다시 확인하세요
      </div>
    );
  }

  // Group counts per tier (for header row "X개" label)
  const tierCounts: Record<number, number> = { 1: 0, 2: 0, 3: 0 };
  items.forEach((r) => { tierCounts[r.stars] = (tierCounts[r.stars] || 0) + 1; });

  const NCOLS = 19;   // + 🎭 Debate Transcript

  // Derive asset type from bucket name
  const assetType = (bucket: string): "Stock" | "ETF" | "—" => {
    const b = (bucket || "").toLowerCase();
    if (b.includes("stocks")) return "Stock";
    if (b.includes("etfs")) return "ETF";
    return "—";
  };

  // Format pct return; null → "—"
  const fmtPct = (v: number | null | undefined): string => {
    if (v == null) return "—";
    const pct = v * 100;
    return `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`;
  };
  // Color for pct return (green positive, red negative, gray null)
  const pctColor = (v: number | null | undefined): string => {
    if (v == null) return C.gray;
    if (v > 0) return C.green;
    if (v < 0) return C.red;
    return C.gray;
  };
  // Compact return cell renderer
  const RetCell = ({ v }: { v: number | null | undefined }) => (
    <td className="text-right px-1 py-1 font-mono whitespace-nowrap"
        style={{ color: pctColor(v), fontSize: 11, fontWeight: "bold", minWidth: 50 }}>
      {fmtPct(v)}
    </td>
  );

  return (
    <div className="overflow-auto rounded" style={{
      maxHeight: 600, backgroundColor: C.bgAlt, border: `1px solid ${C.border}`,
    }}>
      <table className="w-full text-[12px] border-collapse">
        <thead className="sticky top-0" style={{ backgroundColor: C.bgAlt, zIndex: 1 }}>
          <tr style={{ borderBottom: `1px solid ${C.border}` }}>
            <th className="text-center px-1.5 py-1.5" style={{ color: C.gray, minWidth: 40 }}>★</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray, minWidth: 90 }}>Ticker</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray }}>Name</th>
            <th className="text-center px-1.5 py-1.5" style={{ color: C.gray, minWidth: 55 }}>Type</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray, minWidth: 80 }}>Horizon</th>
            <th className="text-center px-1.5 py-1.5" style={{ color: C.gray, minWidth: 50 }}>Comp</th>
            <th className="text-center px-1.5 py-1.5" style={{ color: C.gray, minWidth: 110 }}>📂 분류</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray, minWidth: 110 }}>📊 State</th>
            <th className="text-center px-1.5 py-1.5" style={{ color: C.gray, minWidth: 110 }}>Action</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray, minWidth: 200 }}
                title="포지션 상태 해석 — State machine 상태(PROSPECTING/HOLDING/EXIT_PENDING/EXITED 등)를 사용자 친화적 한국어 설명으로 변환">
              🧭 포지션
            </th>
            <th className="text-center px-1.5 py-1.5" style={{ color: C.gray, minWidth: 130 }}
                title="3-tier 진입가 (CAN SLIM + Elliott + SMA50): AGGRESSIVE 현재가 / PRIMARY CAN SLIM pivot / CONSERVATIVE SMA50 pullback. O'Neil 7% cut-loss 자동 적용.">
              🎯 진입가
            </th>
            <th className="text-center px-1.5 py-1.5" style={{ color: C.gray, minWidth: 110 }}
                title="Elliott Wave 기반 손절 가격 (horizon 별 — tactical: W4 저점, core: W1 상단, strategic: W2 저점/W1 상단). CAN SLIM 7% 절대 cap 적용.">
              🌊 손절가
            </th>
            <th className="text-center px-1.5 py-1.5" style={{ color: C.gray, minWidth: 110 }}>🗳 3-Agent Votes</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray, minWidth: 320 }}>🎭 Debate Transcript</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray }}>💬 PM Agent Reason</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray }}>🎯 Trader Agent Reason</th>
            <th className="text-left px-1.5 py-1.5" style={{ color: C.gray }}>🛡 Risk Agent Reason</th>
            <th className="text-right px-1 py-1.5" style={{ color: C.gray, minWidth: 50 }}>5d</th>
            <th className="text-right px-1 py-1.5" style={{ color: C.gray, minWidth: 55 }}>1mo</th>
            <th className="text-right px-1 py-1.5" style={{ color: C.gray, minWidth: 55 }}>3mo</th>
            <th className="text-right px-1 py-1.5" style={{ color: C.gray, minWidth: 55 }}>6mo</th>
            <th className="text-right px-1 py-1.5" style={{ color: C.gray, minWidth: 55 }}>1y</th>
          </tr>
        </thead>
        <tbody>
          {(() => {
            const rows: React.ReactNode[] = [];
            // Group by CATEGORY now (not stars), since merged list contains active+exit+new
            const CAT_META: Record<string, { label: string; sublabel: string; color: string; bg: string }> = {
              ENTERED:      { label: "✓ 오늘 진입 (ENTERED)",       sublabel: "Phase 5.6에서 2일 BUY_NOW 확정 → 즉시 매수 실행 대상",
                              color: C.green, bg: C.green + "10" },
              EXIT_PENDING: { label: "⚠ 청산 후보 (EXIT_PENDING)",  sublabel: "State-Driven (SKIP×2/WAIT×5) 또는 Regime-Flip (보유 LONG + 오늘 SHORT 시그널) → 청산 검토",
                              color: C.amber, bg: C.amber + "10" },
              HOLDING:      { label: "🔵 보유 중 (HOLDING)",          sublabel: "이미 진입한 포지션 — 오늘 picks 여부와 무관하게 표시",
                              color: C.cyan,  bg: C.cyan + "10" },
              NEW:          { label: "🟢 신규 후보 (NEW)",            sublabel: "오늘 3-Agent voting 통과한 새로운 매수 후보",
                              color: "#0A7D3F", bg: "rgba(22,199,132,0.08)" },
            };
            // ── Split items by asset type FIRST, then by category ──
            // Top-level groups: 📈 STOCKS section + 📦 ETFs section
            // Apply Type filter: ALL shows both, STOCK shows only stocks, ETF shows only ETFs
            const stockItems = (items as any[]).filter((r) => assetType(r.bucket) === "Stock");
            const etfItems   = (items as any[]).filter((r) => assetType(r.bucket) === "ETF");
            const otherItems = (items as any[]).filter((r) => assetType(r.bucket) === "—");

            const allGroups: Array<{
              key: string; emoji: string; label: string; color: string; bg: string; items: any[];
            }> = [
              { key: "STOCK", emoji: "📈", label: "STOCKS — 개별 종목",
                color: C.cyan, bg: C.cyan + "08", items: stockItems },
              { key: "ETF", emoji: "📦", label: "ETFs — 상장지수펀드",
                color: C.purple, bg: C.purple + "08", items: etfItems },
            ];
            if (otherItems.length > 0 && typeFilter === "ALL") {
              allGroups.push({ key: "OTHER", emoji: "—", label: "기타",
                color: C.gray, bg: C.bgAlt, items: otherItems });
            }

            // Apply Type toggle filter
            const ASSET_GROUPS = allGroups.filter((g) => {
              if (typeFilter === "ALL") return true;
              return g.key === typeFilter;
            });

            ASSET_GROUPS.forEach((asset, ai) => {
              if (asset.items.length === 0) return;
              // ── Asset-type top-level header (e.g. "📈 STOCKS (N)") ──
              rows.push(
                <tr key={`asset-hdr-${asset.key}`}
                    style={{ backgroundColor: asset.color + "20",
                             borderTop: `3px double ${asset.color}`,
                             borderBottom: `2px solid ${asset.color}60` }}>
                  <td colSpan={NCOLS} className="px-2 py-2">
                    <div className="flex items-center gap-3">
                      <span style={{ color: asset.color, fontSize: 16, fontWeight: "bold" }}>
                        {asset.emoji} {asset.label}
                      </span>
                      <span style={{ color: C.text, fontSize: 13, fontWeight: "bold" }}>
                        ({asset.items.length}개 / cap 20)
                      </span>
                      {ai === 0 && (
                        <span style={{ marginLeft: "auto", color: C.gray, fontSize: 11, fontStyle: "italic" }}>
                          asset-type 분리: stock 20 + ETF 20 = 총 40종목 cap
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              );

              // ── Within each asset-type, group by category ──
              let prevCategory: string | null = null;
              asset.items.forEach((r: any, i: number) => {
                const cat = r.category || "NEW";
                if (cat !== prevCategory) {
                  const meta = CAT_META[cat];
                  if (meta) {
                    const count = asset.items.filter((x: any) => (x.category || "NEW") === cat).length;
                    rows.push(
                      <tr key={`hdr-${asset.key}-${cat}-${i}`}
                          style={{ backgroundColor: meta.bg,
                                   borderTop: `1.5px solid ${meta.color}80`,
                                   borderBottom: `1px solid ${meta.color}40` }}>
                        <td colSpan={NCOLS} className="px-3 py-1">
                          <div className="flex items-center gap-2">
                            <span style={{ color: asset.color, fontSize: 11, fontWeight: "bold" }}>
                              {asset.emoji}
                            </span>
                            <span style={{ color: meta.color, fontSize: 12, fontWeight: "bold" }}>
                              {meta.label}
                            </span>
                            <span style={{ color: C.gray, fontSize: 11 }}>({count}개)</span>
                            <span style={{ marginLeft: "auto", color: C.gray, fontSize: 10, fontStyle: "italic" }}>
                              {meta.sublabel}
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  }
                  prevCategory = cat;
                }
              const action = ACTION_META[r.action] || ACTION_META.OBSERVE;
              const rowBg =
                r.action === "EXECUTE_TODAY" ? (mode === "buy" ? C.green + "10" : C.red + "10") :
                r.action === "CLOSE_TODAY"   ? C.red   + "10" :
                r.action === "WATCH_TOMORROW"? C.amber + "0a" :
                "transparent";
              rows.push(
                <tr key={`row-${asset.key}-${r.ticker}-${r.horizon || "?"}-${i}`} style={{
                  borderBottom: `1px solid ${C.border}40`,
                  backgroundColor: rowBg,
                }}>
                  <td className="text-center px-1.5 py-1"><Stars n={r.stars} /></td>
                  <td className="px-1.5 py-1">
                    <div className="font-mono font-bold" style={{ color: C.text }}>{r.ticker}</div>
                    <div className="text-[11px]" style={{ color: C.gray }}>{r.sector}</div>
                  </td>
                  <td className="px-1.5 py-1" style={{ color: C.text, maxWidth: 200 }}>
                    {r.name}
                  </td>
                  <td className="text-center px-1.5 py-1">
                    {(() => {
                      const t = assetType(r.bucket);
                      const color = t === "Stock" ? C.cyan : t === "ETF" ? C.purple : C.gray;
                      const emoji = t === "Stock" ? "📈" : t === "ETF" ? "📦" : "—";
                      return <Pill color={color}>{emoji} {t}</Pill>;
                    })()}
                  </td>
                  <td className="px-1.5 py-1">
                    {(() => {
                      const allH = (r as any).all_horizons as string[] | undefined;
                      if (allH && allH.length > 1) {
                        const letters = allH.map((h) => h[0].toUpperCase()).join("·");
                        return (
                          <span title={`복수 horizon에 동시 등장: ${allH.join(", ")} (대표 = ${r.horizon})`}>
                            <Pill color={C.cyan}>⏱ {letters} ×{allH.length}</Pill>
                          </span>
                        );
                      }
                      return (
                        <Pill color={r.horizon === "tactical" ? C.amber : r.horizon === "core" ? C.purple : C.cyan}>
                          {HORIZON_EMOJI[r.horizon] || "?"} {r.horizon}
                        </Pill>
                      );
                    })()}
                  </td>
                  <td className="text-center px-1.5 py-1 font-mono" style={{ color: C.text }}>
                    {r.composite}
                  </td>
                  {/* 📂 분류 — Active Holdings / Exit Pending / New */}
                  <td className="text-center px-1.5 py-1">
                    {(() => {
                      const cat = (r as any).category || "NEW";
                      const dh = (r as any).days_held ?? 0;
                      const inToday = (r as any).in_today_picks;
                      const catMeta: Record<string, { emoji: string; label: string; color: string; bg: string }> = {
                        ENTERED:      { emoji: "✓", label: "오늘 진입",   color: C.green, bg: C.green + "25" },
                        HOLDING:      { emoji: "🔵", label: "보유 중",     color: C.cyan,  bg: C.cyan  + "20" },
                        EXIT_PENDING: { emoji: "⚠", label: "청산 후보",   color: C.amber, bg: C.amber + "25" },
                        NEW:          { emoji: "🟢", label: "신규 후보",   color: C.gray,  bg: "transparent" },
                      };
                      const m = catMeta[cat] || catMeta.NEW;
                      const isRegimeFlip = cat === "EXIT_PENDING" && (r as any).promoted_from_short;
                      const ssH = (r as any).short_signal_horizon;
                      const ssS = (r as any).short_signal_stars;
                      return (
                        <div className="flex flex-col items-center gap-0.5">
                          <span style={{
                            color: m.color, backgroundColor: m.bg,
                            border: `1px solid ${m.color}80`,
                            padding: "1px 6px", borderRadius: 3, fontSize: 11, fontWeight: "bold",
                            whiteSpace: "nowrap",
                          }}>
                            {m.emoji} {m.label}
                          </span>
                          {isRegimeFlip && (
                            <span title={(r as any).short_signal_reason || "보유 LONG + 오늘 PM SHORT 시그널"}
                                  style={{
                                    color: C.red, backgroundColor: C.red + "20",
                                    border: `1px solid ${C.red}80`,
                                    padding: "1px 4px", borderRadius: 3, fontSize: 10, fontWeight: "bold",
                                    whiteSpace: "nowrap",
                                  }}>
                              🔻 Regime-Flip{ssH ? ` (${ssH})` : ""}{ssS ? ` ★${ssS}` : ""}
                            </span>
                          )}
                          {dh > 0 && (
                            <span style={{ color: C.text, fontSize: 11, fontWeight: "bold" }}>
                              Day {dh}
                              {cat === "HOLDING" && (
                                <span style={{ color: inToday ? C.green : C.amber, marginLeft: 3, fontSize: 10 }}>
                                  {inToday ? "✓오늘picks" : "×미선정"}
                                </span>
                              )}
                            </span>
                          )}
                        </div>
                      );
                    })()}
                  </td>
                  <td className="px-1.5 py-1" style={{ color: C.text, fontSize: 11 }}>
                    {r.state}
                    {/*
                      signal_days = system이 BUY_NOW 신호를 연속으로 발행한 일수.
                      PROSPECTING(진입 전 확인 2일) + ENTERED + HOLDING 모두 포함하므로
                      days_held와 다를 수 있음. 사용자 혼동 방지를 위해 다음만 표시:
                        - PROSPECTING / EXIT_PENDING: signal_days 의미 있음 (진입/청산 카운트다운)
                        - HOLDING / ENTERED: days_held가 우선 — signal_days는 tooltip으로만
                    */}
                    {r.signal_days > 0 && (r.state === "PROSPECTING" || r.state === "EXIT_PENDING") && (
                      <span style={{ color: C.gray, marginLeft: 4 }}
                            title="해당 신호가 연속으로 발생한 일수">
                        ({r.signal_days}d)
                      </span>
                    )}
                    {r.signal_days > 0 && (r.state === "HOLDING" || r.state === "ENTERED") && (
                      <span style={{ color: C.gray + "80", marginLeft: 4, fontSize: 10, fontStyle: "italic" }}
                            title={`BUY_NOW 신호 연속 ${r.signal_days}일째 (PROSPECTING 진입 확인 2일 포함, days_held와 별개)`}>
                        sig {r.signal_days}d
                      </span>
                    )}
                  </td>
                  <td className="text-center px-1.5 py-1">
                    <Pill color={action.color}>{action.label}</Pill>
                    <div style={{ marginTop: 2, fontSize: 11, color: C.gray }}>
                      {URGENCY_EMOJI[r.urgency] || ""} {r.urgency}
                    </div>
                  </td>
                  {/* 🧭 포지션 상태 해석 */}
                  <td className="px-1.5 py-1" style={{ maxWidth: 240, minWidth: 200 }}>
                    {(() => {
                      const p = interpretPosition(r);
                      return (
                        <div style={{ lineHeight: 1.35 }}>
                          <div style={{
                            display: "inline-block", fontSize: 11, fontWeight: "bold",
                            color: p.color, backgroundColor: p.color + "18",
                            border: `1px solid ${p.color}55`,
                            borderRadius: 3, padding: "1px 5px", marginBottom: 2,
                          }}>
                            {p.emoji} {p.headline}
                          </div>
                          <div style={{ fontSize: 11, color: C.text, opacity: 0.85 }}>
                            {p.detail}
                          </div>
                        </div>
                      );
                    })()}
                  </td>
                  {/* 🎯 3-Tier Entry Prices (CAN SLIM + Elliott + SMA50) */}
                  <td className="text-center px-1.5 py-1"
                      title={[
                        r.entry_aggressive_rationale && `🟢 AGGRESSIVE: ${r.entry_aggressive_rationale}`,
                        r.entry_primary_rationale && `🟠 PRIMARY: ${r.entry_primary_rationale}`,
                        r.entry_conservative_rationale && `🔵 CONSERVATIVE: ${r.entry_conservative_rationale}`,
                        r.entry_oneil_cut_loss && `\nO'Neil cut-loss: ${r.currency_symbol || "$"}${r.entry_oneil_cut_loss}`,
                        r.entry_rr_ratio && `R/R: ${r.entry_rr_ratio}`,
                      ].filter(Boolean).join("\n\n") || "진입가 데이터 없음"}>
                    {(r.entry_aggressive || r.entry_primary || r.entry_conservative || r.entry_primary_status === "extended") ? (
                      <div style={{ lineHeight: 1.2 }}>
                        {(() => {
                          const sym = r.currency_symbol || "$";
                          const ccy = r.currency || "USD";
                          const isWhole = ccy === "KRW" || ccy === "JPY";
                          const fmt = (v?: number | null) =>
                            v == null ? "—" :
                            isWhole ? `${sym}${Math.round(v).toLocaleString()}` :
                            `${sym}${v.toFixed(2)}`;
                          // EXTENDED warning — Primary 자리에 표시
                          const isExtended = r.entry_primary_status === "extended";
                          return (
                            <>
                              {r.entry_aggressive != null && (
                                <div style={{ fontSize: 12, color: C.green, fontWeight: "bold" }}>
                                  🟢 {fmt(r.entry_aggressive)}
                                </div>
                              )}
                              {isExtended ? (
                                <div style={{ fontSize: 11, color: C.red, fontWeight: "bold",
                                              backgroundColor: C.red + "15",
                                              border: `1px solid ${C.red}50`,
                                              padding: "1px 4px", borderRadius: 3, margin: "1px 0" }}
                                     title={r.entry_primary_rationale || ""}>
                                  ⚠ EXTENDED
                                </div>
                              ) : (
                                r.entry_primary != null && (
                                  <div style={{ fontSize: 12, color: C.amber, fontWeight: "bold" }}>
                                    🟠 {fmt(r.entry_primary)}
                                    {r.entry_primary_status === "buy_zone" && (
                                      <span style={{ fontSize: 10, color: C.green, marginLeft: 2 }}>BZ</span>
                                    )}
                                    {r.entry_primary_status === "await_breakout" && (
                                      <span style={{ fontSize: 10, color: C.gray, marginLeft: 2 }}>↑BO</span>
                                    )}
                                  </div>
                                )
                              )}
                              {r.entry_conservative != null && (
                                <div style={{ fontSize: 12, color: C.cyan }}>
                                  🔵 {fmt(r.entry_conservative)}
                                </div>
                              )}
                              {r.entry_base_pattern && (
                                <div style={{ fontSize: 10, color: C.gray, marginTop: 2 }}>
                                  {r.entry_base_pattern === "cup_with_handle" ? "Cup/Handle" :
                                   r.entry_base_pattern === "flat_base" ? "Flat Base" :
                                   r.entry_base_pattern === "double_bottom" ? "Dbl Btm" : "—"}
                                  {r.entry_base_quality && ` ${r.entry_base_quality}`}
                                  {r.entry_volume_confirmed ? " ✓vol" : r.entry_volume_ratio ? ` ${r.entry_volume_ratio}x` : ""}
                                </div>
                              )}
                              {r.entry_rr_ratio != null && (
                                <div style={{ fontSize: 10, color: r.entry_rr_ratio >= 2 ? C.green : C.gray }}>
                                  R/R {r.entry_rr_ratio.toFixed(1)}
                                </div>
                              )}
                            </>
                          );
                        })()}
                      </div>
                    ) : (
                      <span className="text-[11px]" style={{ color: C.gray }}>
                        {r.entry_composite_tier === "SKIP" ? "SKIP" : "—"}
                      </span>
                    )}
                  </td>
                  {/* 🌊 Elliott Wave Stop Loss */}
                  <td className="text-center px-1.5 py-1"
                      title={r.stop_rationale ? `${r.stop_rationale}\n\n파동: ${r.stop_wave_guess || "—"} · 유형: ${r.stop_type || "—"} · 통화: ${r.currency || "USD"}` : "데이터 없음"}>
                    {r.stop_price != null ? (
                      <div>
                        <div className="font-mono font-bold" style={{
                          fontSize: 13,
                          color: (r.stop_pct ?? 0) >= -3 ? C.red
                               : (r.stop_pct ?? 0) >= -8 ? C.amber
                               : C.green,
                        }}>
                          {(() => {
                            const sym = r.currency_symbol || "$";
                            const ccy = r.currency || "USD";
                            // For non-decimal currencies (KRW, JPY), no decimal places
                            const isWhole = ccy === "KRW" || ccy === "JPY";
                            return `${sym}${isWhole ? Math.round(r.stop_price).toLocaleString() : r.stop_price.toFixed(2)}`;
                          })()}
                        </div>
                        <div className="text-[11px] font-mono" style={{ color: C.gray }}>
                          {r.stop_pct != null ? `${r.stop_pct > 0 ? "+" : ""}${r.stop_pct.toFixed(1)}%` : "—"}
                        </div>
                        <div className="text-[10px]" style={{
                          color: r.stop_type === "W4_TIGHT" ? C.amber
                               : r.stop_type === "W1_PRIMARY" ? C.cyan
                               : r.stop_type === "W2_INVALID" ? C.red
                               : r.stop_type === "MECHANICAL" ? C.gray
                               : C.text,
                        }}>
                          {r.stop_type === "W4_TIGHT"   ? "W4 TIGHT"
                          : r.stop_type === "W1_PRIMARY" ? "W1 PRIMARY"
                          : r.stop_type === "W2_INVALID" ? "W2 INVALID"
                          : r.stop_type === "SWING_LOW"  ? "SWING"
                          : r.stop_type === "MECHANICAL" ? "MECH"
                          : "—"}
                        </div>
                      </div>
                    ) : (
                      <span className="text-[11px]" style={{ color: C.gray }}>—</span>
                    )}
                  </td>
                  {/* 3-Agent Votes */}
                  <td className="px-1.5 py-1 text-[11px]">
                    {/* HELD_NO_EVAL: 보유 중이나 오늘 재평가 안 됨 — 빈칸 대신 명시 */}
                    {!r.votes && (r as any).tier === "HELD_NO_EVAL" && (
                      <div className="text-[10px] px-1 py-0.5 rounded text-center"
                           style={{ color: C.gray, backgroundColor: C.gray + "15",
                                    border: `1px solid ${C.gray}40` }}
                           title="보유 포지션 — 오늘 PM swarm이 재평가하지 않음 (신규 신호 발생 시에만 재토론)">
                        보유·미평가
                      </div>
                    )}
                    {r.votes && (
                      <div className="flex flex-col gap-0.5"
                           title={`Risk score: ${r.risk_score ?? "?"} · ${r.risk_reason || ""}`}>
                        {(["pm", "trading", "risk"] as const).map((agent) => {
                          const v = (r.votes as any)[agent];
                          const color = v === "APPROVE" ? C.green : v === "REJECT" ? C.red : C.amber;
                          const emoji = v === "APPROVE" ? "✓" : v === "REJECT" ? "✗" : "○";
                          const label = agent === "pm" ? "PM" : agent === "trading" ? "Trd" : "Rsk";
                          return (
                            <div key={agent} className="flex items-center gap-1 whitespace-nowrap"
                                 style={{ fontSize: 10 }}>
                              <span style={{ color: C.gray, width: 22 }}>{label}</span>
                              <span style={{ color, fontWeight: "bold" }}>{emoji} {v}</span>
                            </div>
                          );
                        })}
                        {r.consensus && (
                          <div className="text-[10px] mt-0.5 px-1 py-0.5 rounded text-center font-bold"
                               style={{
                                 color: r.consensus === "UNANIMOUS" ? C.green
                                      : r.consensus.includes("MAJORITY") ? C.cyan
                                      : C.amber,
                                 backgroundColor:
                                   r.consensus === "UNANIMOUS" ? C.green + "20"
                                   : r.consensus.includes("MAJORITY") ? C.cyan + "20"
                                   : C.amber + "20",
                               }}>
                            {r.consensus.replace("_", " ")}
                          </div>
                        )}
                      </div>
                    )}
                    {/* Legacy cross-check (small badges below) */}
                    <div className="mt-1 flex gap-0.5 flex-wrap">
                      {r.in_proxy_latest && <Pill color={C.cyan}>📊</Pill>}
                      {r.in_top_alpha    && <Pill color={C.green}>🏆</Pill>}
                      {r.in_worst_alpha  && <Pill color={C.red}>📉</Pill>}
                    </div>
                  </td>
                  {/* 🎭 Debate Synthesis Transcript */}
                  <td className="px-1.5 py-1" style={{ color: C.text, lineHeight: 1.4, maxWidth: 360, minWidth: 280 }}>
                    {(r as any).debate_transcript ? (
                      <>
                        <div style={{ color: C.text, fontSize: 12 }}>
                          {(r as any).debate_transcript}
                        </div>
                        <div className="text-[11px] mt-1 pt-1 flex items-center gap-2"
                             style={{ color: C.gray, borderTop: `1px dashed ${C.border}` }}>
                          {(r as any).key_factor && (
                            <span style={{ color: C.cyan }}>핵심: <span style={{color: C.text, fontWeight: "bold"}}>{(r as any).key_factor}</span></span>
                          )}
                          {(r as any).final_decision && (
                            <span style={{
                              color: (r as any).final_decision === "INCLUDE" ? C.green
                                   : (r as any).final_decision === "INCLUDE_REDUCED_SIZE" ? C.amber
                                   : C.gray,
                              fontWeight: "bold",
                            }}>
                              → {(r as any).final_decision}
                            </span>
                          )}
                        </div>
                      </>
                    ) : <span style={{ color: C.gray }}>—</span>}
                  </td>
                  {/* PM Agent reason commentary */}
                  <td className="px-1.5 py-1" style={{ color: C.text, lineHeight: 1.4, maxWidth: 320, minWidth: 240 }}>
                    {r.rationale || "—"}
                  </td>
                  {/* Trader Agent reason + entry trigger + exit trigger */}
                  <td className="px-1.5 py-1" style={{ color: C.text, lineHeight: 1.4, maxWidth: 320, minWidth: 240 }}>
                    {/* Primary: Trader Agent rationale (WHY this timing decision) */}
                    <div style={{ color: C.text }}>
                      {r.trading_rationale || "—"}
                    </div>
                    {/* Secondary: Trader Agent entry trigger (specific trigger condition) */}
                    {r.entry_trigger && r.entry_trigger !== r.trading_rationale && (
                      <div className="text-[11px] mt-1 pt-1"
                           style={{ color: C.cyan, borderTop: `1px dashed ${C.border}` }}>
                        ⏱ trigger: {r.entry_trigger}
                      </div>
                    )}
                    {mode === "buy" && r.exit_triggers && r.exit_triggers.length > 0 && (
                      <div className="text-[11px] mt-0.5" style={{ color: C.gray }}>
                        🛑 exit: {r.exit_triggers[0].type} · {r.exit_triggers[0].action}
                      </div>
                    )}
                  </td>
                  {/* Risk Agent reason — deterministic risk evaluation */}
                  <td className="px-1.5 py-1" style={{ color: C.text, lineHeight: 1.4, maxWidth: 280, minWidth: 220 }}>
                    <div style={{ color: C.text, fontSize: 12 }}>
                      {r.risk_reason || "—"}
                    </div>
                    {r.risk_score != null && (
                      <div className="text-[11px] mt-1 pt-1 flex items-center gap-1.5"
                           style={{ color: C.gray, borderTop: `1px dashed ${C.border}` }}>
                        <span>Risk score:</span>
                        <span style={{
                          color: r.risk_score <= 35 ? C.green : r.risk_score >= 55 ? C.red : C.amber,
                          fontWeight: "bold",
                        }}>
                          {r.risk_score.toFixed(0)}/100
                        </span>
                      </div>
                    )}
                  </td>
                  <RetCell v={r.ret_5d} />
                  <RetCell v={r.ret_1mo} />
                  <RetCell v={r.ret_3mo} />
                  <RetCell v={r.ret_6mo} />
                  <RetCell v={r.ret_1y} />
                </tr>
              );
              });  // close asset.items.forEach
            });  // close ASSET_GROUPS.forEach
            return rows;
          })()}
        </tbody>
      </table>
    </div>
  );
}


interface FinalListPanelProps {
  dataVersion?: number;  // bumps when parent scan/swarm/backtest completes — triggers refetch
  scanning?: boolean;    // true while Run Live Scan pipeline is in progress — poll every 30s
}

export default function FinalListPanel({ dataVersion = 0, scanning = false }: FinalListPanelProps) {
  const [data, setData] = useState<FinalListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const tab: "buy" | "sell" = "buy";
  const setTab = (_: "buy" | "sell") => {}; // sell tab deprecated — SHORT signals fold into EXIT_PENDING
  void setTab;

  // Type filter: ALL (both stock + ETF) | STOCK (📈 only) | ETF (📦 only)
  const [typeFilter, setTypeFilter] = useState<"ALL" | "STOCK" | "ETF">("ALL");

  // Commentary tab: UNIFIED (12,000자 single) | SPLIT (📋 공통 + 📈 Stock + 📦 ETF)
  const [commentaryTab, setCommentaryTab] = useState<"UNIFIED" | "COMMON" | "STOCK" | "ETF">("UNIFIED");

  const refresh = async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await fetchFinalList();
      setData(r);
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  // Re-fetch on mount AND whenever parent bumps dataVersion (e.g. Run Live Scan finished).
  useEffect(() => { refresh(); }, [dataVersion]);

  // While Run Live Scan is in progress, poll swarm_generated_at every 30s so the panel
  // refreshes as soon as the new swarm cache lands (rather than waiting for the parent's
  // final dataVersion bump). Stops automatically when `scanning` flips to false.
  useEffect(() => {
    if (!scanning) return;
    let lastGen = data?.metadata?.swarm_generated_at || "";
    const id = setInterval(async () => {
      try {
        const r = await fetchFinalList();
        const newGen = r.metadata?.swarm_generated_at || "";
        if (newGen && newGen !== lastGen) {
          lastGen = newGen;
          setData(r);   // surface new data immediately
        }
      } catch { /* ignore transient errors */ }
    }, 30000);
    return () => clearInterval(id);
  }, [scanning, data?.metadata?.swarm_generated_at]);

  if (loading && !data) return (
    <div className="px-3 py-2 text-[12px]" style={{ color: C.gray }}>Loading Final List…</div>
  );
  if (err) return (
    <div className="px-3 py-2 text-[12px]" style={{ color: C.red }}>Error: {err}</div>
  );
  if (!data) return null;

  // Merge buy_list + active_positions (LONG) + exit_pending into unified list with category
  const mergedItems: FinalListItem[] = (() => {
    const map = new Map<string, FinalListItem>();
    const key = (t: string, h: string) => `${t}::${h}`;

    // 1. Buy list (today's voted candidates) — base
    for (const r of data.buy_list) {
      map.set(key(r.ticker, r.horizon), { ...r, category: "NEW" } as any);
    }

    // 2. Active LONG positions — override category to HOLDING/ENTERED if present
    // active_positions only ever tracks LONG (SHORT positions are not persisted to state).
    // bucket may be "?", "long_stocks", "long_etfs", or empty (when PM didn't pick today).
    // Treat any non-"short_*" bucket as LONG so positions held off-cycle still surface.
    for (const ap of data.active_positions || []) {
      const k = key(ap.ticker, ap.horizon);
      const bucket = ap.bucket || "";
      const isLong = !bucket.startsWith("short");  // includes ?, long_*, ""
      if (!isLong) continue;
      const cat = ap.state === "ENTERED" ? "ENTERED" : "HOLDING";
      if (map.has(k)) {
        const existing: any = map.get(k);
        existing.category = cat;
        existing.days_held = ap.days_held;
        existing.persistence_days = ap.persistence_days;
        existing.in_today_picks = true;
      } else {
        // Active position NOT in today's buy list → synthesize entry.
        // Spread ...ap FIRST so API-annotated entry_*/stop_*/currency fields carry through,
        // then override display-specific fields.
        map.set(k, {
          ...ap,
          stars: 0, n_validations: 0,
          signal_days: 0, urgency: "NORMAL",
          action: "ALREADY_HELD",
          entry_trigger: "", rationale: ap.rationale || "",
          trading_rationale: ap.trading_rationale || "",
          consensus: "—",
          category: cat,
          in_today_picks: false,
        } as any);
      }
    }

    // 3. Exit pending → override category (includes regime-flip promoted items)
    for (const ep of (data.exit_pending || []) as any[]) {
      const k = key(ep.ticker, ep.horizon);
      if (map.has(k)) {
        const existing: any = map.get(k);
        existing.category = "EXIT_PENDING";
        existing.days_held = ep.days_held;
        existing.action = "CLOSE_TODAY";
        existing.rationale = ep.exit_reason || existing.rationale;
        existing.promoted_from_short = !!ep.promoted_from_short;
        existing.short_signal_stars = ep.short_signal_stars;
        existing.short_signal_action = ep.short_signal_action;
        existing.short_signal_horizon = ep.short_signal_horizon;
        existing.short_signal_reason = ep.short_signal_reason;
      } else {
        // Spread ...ep FIRST so API-annotated entry_*/stop_*/currency + risk fields carry through.
        map.set(k, {
          ...ep,
          bucket: ep.bucket || "(exiting)",
          stars: ep.short_signal_stars || 0, n_validations: 0,
          state: "EXIT_PENDING", signal_days: 0, urgency: "URGENT",
          action: "CLOSE_TODAY",
          entry_trigger: "", rationale: ep.exit_reason || ep.rationale || "",
          trading_rationale: ep.short_signal_reason || ep.trading_rationale || "",
          consensus: ep.promoted_from_short ? "REGIME_FLIP" : "EXITING",
          category: "EXIT_PENDING",
          in_today_picks: false,
          promoted_from_short: !!ep.promoted_from_short,
        } as any);
      }
    }

    // ── Phase 2: collapse rows by ticker (same ticker across horizons → one row) ──
    // Same ticker can appear in tactical/core/strategic — show as single row with
    // combined horizon badge; pick representative by category priority.
    const CAT_PRIORITY: Record<string, number> = {
      EXIT_PENDING: 0, ENTERED: 1, HOLDING: 2, NEW: 3,
    };
    const byTicker = new Map<string, any>();
    const horizonsByTicker = new Map<string, Set<string>>();
    for (const row of Array.from(map.values()) as any[]) {
      const t = row.ticker;
      // Track all horizons this ticker spans
      if (!horizonsByTicker.has(t)) horizonsByTicker.set(t, new Set());
      horizonsByTicker.get(t)!.add(row.horizon);

      const existing = byTicker.get(t);
      if (!existing) {
        byTicker.set(t, row);
        continue;
      }
      // Replace if new row has higher-priority category
      const pNew = CAT_PRIORITY[row.category || "NEW"] ?? 9;
      const pOld = CAT_PRIORITY[existing.category || "NEW"] ?? 9;
      if (pNew < pOld) {
        byTicker.set(t, row);
      } else if (pNew === pOld) {
        // Same category — prefer higher composite / more stars / more days_held
        const scoreNew = (row.stars || 0) * 100 + (row.composite || 0) + (row.days_held || 0) * 0.1;
        const scoreOld = (existing.stars || 0) * 100 + (existing.composite || 0) + (existing.days_held || 0) * 0.1;
        if (scoreNew > scoreOld) byTicker.set(t, row);
      }
    }
    // Attach collapsed horizon list to each representative row
    for (const [t, row] of byTicker.entries()) {
      const hSet = horizonsByTicker.get(t) || new Set();
      const H_ORDER: Record<string, number> = { tactical: 0, core: 1, strategic: 2 };
      (row as any).all_horizons = Array.from(hSet).sort(
        (a, b) => (H_ORDER[a] ?? 9) - (H_ORDER[b] ?? 9)
      );
    }

    // ── Turnover cap: keep only tickers whitelisted by backend (top 5 per category) ──
    // Backend computes top-5 per category by quality score (stars × 100 + composite + days_held × 0.1
    // + regime-flip bonus). Frontend respects that whitelist to keep displays consistent with
    // commentary input.
    let allRows = Array.from(byTicker.values()) as any[];
    const cappedMap = data.capped_tickers_by_category || {};
    if (cappedMap && (cappedMap.ENTERED || cappedMap.NEW || cappedMap.HOLDING || cappedMap.EXIT_PENDING)) {
      const allowByCat: Record<string, Set<string>> = {
        ENTERED:      new Set(cappedMap.ENTERED || []),
        NEW:          new Set(cappedMap.NEW || []),
        HOLDING:      new Set(cappedMap.HOLDING || []),
        EXIT_PENDING: new Set(cappedMap.EXIT_PENDING || []),
      };
      allRows = allRows.filter((r) => {
        const cat = r.category || "NEW";
        const allow = allowByCat[cat];
        return !allow || allow.size === 0 || allow.has(r.ticker);
      });
    }

    // Sort: ENTERED first → EXIT_PENDING → HOLDING (today picked) → HOLDING (not today) → NEW
    const CAT_ORDER: Record<string, number> = {
      ENTERED: 0, EXIT_PENDING: 1, HOLDING: 2, NEW: 3,
    };
    return allRows.sort((a: any, b: any) => {
      const ca = CAT_ORDER[a.category || "NEW"] ?? 9;
      const cb = CAT_ORDER[b.category || "NEW"] ?? 9;
      if (ca !== cb) return ca - cb;
      // Within same category: in_today_picks first, then by days_held desc
      if ((a.in_today_picks ?? true) !== (b.in_today_picks ?? true)) {
        return (b.in_today_picks ?? true) ? 1 : -1;
      }
      return (b.days_held ?? 0) - (a.days_held ?? 0);
    });
  })();

  const buyN = mergedItems.filter((r: any) => r.category !== "EXIT_PENDING").length;
  // Pre-filter items by Type toggle BEFORE passing down to FinalListTable.
  // This guarantees the filter is applied even if downstream components have stale closures.
  const items = mergedItems.filter((r: any) => {
    if (typeFilter === "ALL") return true;
    const b = (r.bucket || "").toLowerCase();
    if (typeFilter === "STOCK") return b.includes("stocks");
    if (typeFilter === "ETF")   return b.includes("etfs");
    return true;
  });
  void tab; // tab forced to "buy" — sell list deprecated
  const meta = data.metadata;

  // Summary counts
  const counts = items.reduce((acc, r) => {
    acc[r.action] = (acc[r.action] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  // Count promoted (regime-flip) exit_pending so we can surface in header
  const promotedExitN = (data.exit_pending || []).filter((r: any) => r.promoted_from_short).length;

  return (
    <div className="mt-6 mb-4 px-3 py-3 rounded" style={{
      backgroundColor: C.bg,
      border: `2px solid ${C.green}80`,
    }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3 pb-2 border-b" style={{ borderColor: C.border }}>
        <div>
          <div className="text-[16px] font-bold flex items-center gap-2" style={{ color: C.green }}>
            🟢 매수 Final List (통합)
            {promotedExitN > 0 && (
              <span style={{
                color: C.red, fontSize: "12px",
                padding: "1px 6px", borderRadius: "10px",
                border: `1px solid ${C.red}80`, backgroundColor: C.red + "15",
              }}>
                ⚠ {promotedExitN} regime-flip 청산
              </span>
            )}
          </div>
          <div className="text-[11px] mt-0.5" style={{ color: C.gray }}>
            4-Filter Pipeline: PM Swarm ∩ Trading Signal (BUY_NOW) ∩ Position State + Backtest cross-check
            · 매도 신호 = EXIT_PENDING 카테고리로 통합 (보유 LONG + SHORT 시그널 자동 승격)
            · <span style={{ color: C.amber }}>턴오버 관리: 전체 stock 20개 + ETF 20개 (총 40종목 cap, quality 점수 상위)</span>
            {meta.swarm_generated_at && <> · Swarm @ {meta.swarm_generated_at}</>}
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="px-3 py-1.5 rounded text-[13px] shrink-0"
                style={{
                  backgroundColor: C.green + "30",
                  color: C.green,
                  border: `1px solid ${C.green}80`,
                  fontWeight: "bold",
                }}>
            🟢 매수 ({buyN}) · 보유 ({data.active_positions?.length || 0}) · 청산 ({data.exit_pending?.length || 0})
          </span>

          {/* ── Type Filter Toggle: ALL / STOCK / ETF ── */}
          {/* IMPORTANT: counts use mergedItems (unfiltered) — `items` is already type-filtered */}
          {(() => {
            const stockCount = (mergedItems as any[]).filter((r) => {
              const b = (r.bucket || "").toLowerCase();
              return b.includes("stocks");
            }).length;
            const etfCount = (mergedItems as any[]).filter((r) => {
              const b = (r.bucket || "").toLowerCase();
              return b.includes("etfs");
            }).length;
            const TYPES: Array<{ key: "ALL" | "STOCK" | "ETF"; emoji: string; label: string; color: string; count: number }> = [
              { key: "ALL",   emoji: "🔁", label: "All",   color: C.gray,   count: stockCount + etfCount },
              { key: "STOCK", emoji: "📈", label: "Stock", color: C.cyan,   count: stockCount },
              { key: "ETF",   emoji: "📦", label: "ETF",   color: C.purple, count: etfCount },
            ];
            return (
              <div className="flex items-center rounded shrink-0"
                   style={{ border: `1px solid ${C.border}`, backgroundColor: C.bgAlt }}
                   title="Type 필터: 종목 종류별 표시 (Stock / ETF / All)">
                <span className="px-2 py-1.5 text-[12px] font-bold"
                      style={{ color: C.gray, borderRight: `1px solid ${C.border}` }}>
                  Type
                </span>
                {TYPES.map((t) => (
                  <button
                    key={t.key}
                    onClick={() => setTypeFilter(t.key)}
                    className="px-2 py-1.5 text-[12px] font-bold transition-colors"
                    style={{
                      backgroundColor: typeFilter === t.key ? t.color + "30" : "transparent",
                      color: typeFilter === t.key ? t.color : C.gray,
                      borderRight: t.key !== "ETF" ? `1px solid ${C.border}` : undefined,
                      cursor: "pointer",
                    }}>
                    {t.emoji} {t.label} ({t.count})
                  </button>
                ))}
              </div>
            );
          })()}

          <button onClick={refresh}
                  disabled={loading}
                  title="Final List 단순 새로고침 (캐시 사용 — 시스템 실행은 사이드바 Run Live Scan 사용)"
                  className="px-2 py-1 rounded text-[12px]"
                  style={{
                    color: C.text, backgroundColor: C.bgAlt,
                    border: `1px solid ${C.border}`,
                    cursor: loading ? "wait" : "pointer",
                  }}>
            {loading ? "⟳" : "🔄"}
          </button>
        </div>
      </div>

      {/* Column descriptions toggle */}
      <ColDefToggle title="📖 컬럼 설명 (클릭하여 펼치기/숨기기)" defs={[
        { col: "★ Stars",         desc: "3-Agent Voting (PM + Trading + Risk) 합의 점수: ★★★ UNANIMOUS / MAJORITY_CLEAN · ★★ MAJORITY_DISSENT / SOLO_CLEAN · ★ SOLO_DISSENT / ALL_CAUTION" },
        { col: "Ticker",          desc: "종목 코드 + Sector 표시" },
        { col: "Name",            desc: "종목명" },
        { col: "Type",            desc: "📈 Stock / 📦 ETF — bucket에서 도출" },
        { col: "Horizon",         desc: "🚀 Tactical (5d) / ⚓ Core (21d) / 🌐 Strategic (63d) — PM Agent가 picks한 투자 기간" },
        { col: "Comp",            desc: "Composite Score (0-100). 0.30·TCS + 0.25·TFS_resid + 0.30·RSS_hybrid + 0.15·URS − 0.10·max(0, OER−40)" },
        { col: "📂 분류",          desc: "보유 상태 분류: ✓ 오늘 진입 (ENTERED, 즉시 매수) · 🔵 보유 중 (HOLDING, 진입한 지 N일 — '✓오늘picks' or '×미선정' 표시) · ⚠ 청산 후보 (EXIT_PENDING) · 🟢 신규 후보 (NEW, 오늘 처음 voting 통과)" },
        { col: "📊 State",        desc: "Phase 5.6 Position State Machine: PROSPECTING (관찰) → ENTERED (확정) → HOLDING (보유) → EXIT_PENDING (청산 대기) → EXITED (완료). Hysteresis로 2일 confirmation 필요" },
        { col: "Action",          desc: "오늘 실행 (EXECUTE_TODAY: 🆕 NEW BUY alert) / 내일 재확인 (WATCH_TOMORROW: 1일째) / 이미 보유 (ALREADY_HELD) / 관찰만 (OBSERVE)" },
        { col: "🗳 3-Agent Votes", desc: "PM Agent (composite + classification 기반) · Trading Agent (entry_signal: BUY_NOW=APPROVE/WAIT=CAUTION/SKIP=REJECT) · Risk Agent (overheating+volatility+liquidity+concentration+drawdown 5-차원 점수). 각 ✓ APPROVE / ○ CAUTION / ✗ REJECT" },
        { col: "💬 PM Reason",    desc: "PM Agent의 한국어 commentary — 거시/섹터/스토리 관점에서 선정 이유" },
        { col: "🎯 Trader Reason", desc: "Trading Agent의 영어 commentary — timing/execution 관점에서 진입 이유 + entry trigger + exit trigger" },
        { col: "🛡 Risk Reason",  desc: "Risk Manager의 한국어 평가 — 5-차원 risk 점수 + 주요 위험 요인 + 사이즈 조정 권고" },
        { col: "5d ~ 1y",         desc: "Trailing total returns: 5 trading days · 21d (1mo) · 63d (3mo) · 126d (6mo) · 252d (1y). 가격 cache에 충분 history 부족 시 '—'" },
      ]} />

      {/* Action breakdown */}
      <div className="flex items-center gap-2 mb-3 flex-wrap text-[11px]">
        {Object.entries(ACTION_META).map(([k, v]) => {
          const n = counts[k] || 0;
          if (n === 0) return null;
          return (
            <span key={k} style={{
              color: v.color, backgroundColor: v.color + "1a",
              border: `1px solid ${v.color}50`,
              padding: "2px 6px", borderRadius: 3, fontWeight: "bold",
            }}>{v.label}: {n}</span>
          );
        })}
        <span style={{ marginLeft: "auto", color: C.gray }}>
          📊 cross-checks: {meta.n_proxy_long_stocks} proxy stocks · {meta.n_top_stocks} top α · {meta.n_positions_tracked} positions tracked
        </span>
      </div>

      {/* Executive Commentary — 3-Agent voting 기반 선정 근거 + 향후 전망 (1000자) */}
      {data.commentary && (
        <div className="mb-3 p-3 rounded"
             style={{
               backgroundColor: tab === "buy" ? C.green + "0a" : C.red + "0a",
               border: `1.5px solid ${tab === "buy" ? C.green + "60" : C.red + "60"}`,
             }}>
          <div className="flex items-center justify-between mb-2 pb-1.5"
               style={{ borderBottom: `1px solid ${C.green}30` }}>
            <div className="text-[14px] font-bold flex items-center gap-2 flex-wrap"
                 style={{ color: C.green }}>
              📋 Executive Commentary —
              {/* 4-tab selector: UNIFIED / COMMON / STOCK / ETF */}
              {([
                { key: "UNIFIED", emoji: "🌐", label: "통합 (12,000자)",         color: C.green },
                { key: "COMMON",  emoji: "📋", label: "공통 거시 (1,500자)",       color: C.cyan },
                { key: "STOCK",   emoji: "📈", label: "Stock Deep-Dive (5,500자)",color: C.cyan },
                { key: "ETF",     emoji: "📦", label: "ETF Deep-Dive (5,500자)",  color: C.purple },
              ] as const).map((t) => (
                <button
                  key={t.key}
                  onClick={() => setCommentaryTab(t.key)}
                  className="px-2 py-0.5 rounded text-[12px] transition-colors"
                  style={{
                    backgroundColor: commentaryTab === t.key ? t.color + "30" : "transparent",
                    color: commentaryTab === t.key ? t.color : C.gray,
                    border: `1px solid ${commentaryTab === t.key ? t.color + "80" : C.border}`,
                    fontWeight: commentaryTab === t.key ? "bold" : "normal",
                    cursor: "pointer",
                  }}>
                  {t.emoji} {t.label}
                </button>
              ))}
            </div>
            <div className="text-[11px]" style={{ color: C.gray }}>
              {(() => {
                const c: any = data.commentary;
                const fieldMap: Record<string, { txt: string; cached: boolean; stale: boolean; pending: boolean }> = {
                  UNIFIED: {
                    txt: c.unified_commentary || c.buy_commentary || "",
                    cached: !!(c.unified_cached ?? c.buy_cached),
                    stale: !!c.stale, pending: !!c.pending,
                  },
                  COMMON: {
                    txt: c.common_macro || "",
                    cached: !!c.common_macro_cached, stale: !!c.common_macro_stale, pending: !!c.common_macro_pending,
                  },
                  STOCK: {
                    txt: c.stock_split || "",
                    cached: !!c.stock_split_cached, stale: !!c.stock_split_stale, pending: !!c.stock_split_pending,
                  },
                  ETF: {
                    txt: c.etf_split || "",
                    cached: !!c.etf_split_cached, stale: !!c.etf_split_stale, pending: !!c.etf_split_pending,
                  },
                };
                const cur = fieldMap[commentaryTab];
                if (cur.pending) {
                  return <span style={{ color: C.amber, fontWeight: "bold" }}>⏳ 생성 중 (3-8분 후 새로고침)</span>;
                }
                if (cur.stale) {
                  return <span style={{ color: C.cyan }}>♻ 이전 결과 (갱신 중)</span>;
                }
                return cur.cached ? "(cached)" : "(generated)";
              })()}
              {" "}· {data.commentary.generated_at?.slice(0, 16) || ""}
              {" "}· {(() => {
                const c: any = data.commentary;
                const lenMap: Record<string, number> = {
                  UNIFIED: (c.unified_commentary || c.buy_commentary || "").length,
                  COMMON:  (c.common_macro || "").length,
                  STOCK:   (c.stock_split || "").length,
                  ETF:     (c.etf_split || "").length,
                };
                return lenMap[commentaryTab].toLocaleString();
              })()}자
            </div>
          </div>
          <div className="text-[13px]"
               style={{
                 color: C.text, lineHeight: 1.75,
                 whiteSpace: "pre-wrap",
                 fontFamily: "-apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif",
                 maxHeight: "1000px",
                 overflowY: "auto",
                 padding: "8px 12px",
                 backgroundColor: C.bg,
                 borderRadius: 4,
                 border: `1px solid ${C.green}30`,
               }}>
            {(() => {
              const c: any = data.commentary;
              const txtMap: Record<string, string> = {
                UNIFIED: c.unified_commentary || c.buy_commentary || "(통합 commentary 생성 중... 6-12분 소요)",
                COMMON:  c.common_macro || "(공통 거시 commentary 생성 중... 거시 + 과거 유사 구간 + 시나리오 1,500자, 3-5분 소요)",
                STOCK:   c.stock_split || "(Stock Deep-Dive 생성 중... 개별 종목 펀더멘털·실적·catalyst 5,500자, 5-8분 소요)",
                ETF:     c.etf_split || "(ETF Deep-Dive 생성 중... 섹터·region·factor 5,500자, 5-8분 소요)",
              };
              return txtMap[commentaryTab];
            })()}
          </div>
          <div className="text-[11px] mt-2 pt-1.5"
               style={{ color: C.gray, borderTop: `1px dashed ${C.border}` }}>
            ⓘ {commentaryTab === "UNIFIED" && "통합 commentary — 11개 섹션 (카테고리 + 호라이즌 + 🕰 과거 유사 + Top-3 + 액션)"}
            {commentaryTab === "COMMON"  && "공통 거시 — Stock+ETF 분석의 공통 기반 (3 섹션: 거시 / 🕰 과거 유사 / 시나리오)"}
            {commentaryTab === "STOCK"   && "Stock Deep-Dive — 개별 종목 펀더멘털·실적·catalyst (7 섹션, ETF 제외)"}
            {commentaryTab === "ETF"     && "ETF Deep-Dive — 섹터 로테이션·region·style·factor (7 섹션, 개별 stock 제외)"}
            · {items.length}종목 분석 · 턴오버 cap 적용 (stock 20 + ETF 20 = 40종목)
          </div>
        </div>
      )}

      <FinalListTable items={items} mode={tab} categoryCommentary={data.category_commentary} typeFilter={typeFilter} />

      <div className="mt-2 text-[11px]" style={{ color: C.gray }}>
        <span style={{ color: C.text, fontWeight: "bold" }}>★ Stars:</span>{" "}
        <span style={{ color: C.green }}>★★★</span> proxy + α top 모두 일치 (최고 신뢰도) ·{" "}
        <span style={{ color: C.amber }}>★★</span> 둘 중 하나 일치 ·{" "}
        <span style={{ color: C.gray }}>★</span> PM signal만 ·{" "}
        Action: <span style={{ color: C.green }}>오늘 실행</span> (Phase 5.6 ENTERED) →{" "}
        <span style={{ color: C.amber }}>내일 재확인</span> (1일 BUY_NOW) →{" "}
        <span style={{ color: C.cyan }}>이미 보유</span> (HOLDING)
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Active Positions Section — currently HELD (HOLDING/ENTERED) regardless of today's voting
// ─────────────────────────────────────────────────────────────────────
function ActivePositionsSection({ items }: { items: ActivePositionItem[] }) {
  // Group by ticker (collapse multiple horizons of same ticker)
  const grouped: Record<string, ActivePositionItem[]> = {};
  for (const r of items) {
    if (!grouped[r.ticker]) grouped[r.ticker] = [];
    grouped[r.ticker].push(r);
  }
  const tickers = Object.keys(grouped).sort((a, b) => {
    // Sort by max days_held desc
    const ma = Math.max(...grouped[a].map(r => r.days_held || 0));
    const mb = Math.max(...grouped[b].map(r => r.days_held || 0));
    return mb - ma;
  });

  return (
    <div className="mb-3 p-2 rounded" style={{
      backgroundColor: C.cyan + "0a",
      border: `1.5px solid ${C.cyan}50`,
    }}>
      <div className="text-[13px] font-bold mb-2 flex items-center gap-2"
           style={{ color: C.cyan }}>
        🔵 보유 중인 포지션 (Active Holdings) — {tickers.length}종목 {items.length}개 horizon
        <span className="text-[11px]" style={{ color: C.gray, fontWeight: "normal" }}>
          · Phase 5.6 HOLDING/ENTERED · 오늘 picks에 없어도 표시
        </span>
      </div>

      <div className="overflow-auto rounded" style={{
        maxHeight: 360, backgroundColor: C.bgAlt, border: `1px solid ${C.border}`,
      }}>
        <table className="w-full text-[12px] border-collapse">
          <thead className="sticky top-0" style={{ backgroundColor: C.bgAlt, zIndex: 1 }}>
            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
              <th className="text-left px-1.5 py-1" style={{ color: C.gray }}>Ticker</th>
              <th className="text-left px-1.5 py-1" style={{ color: C.gray }}>Name</th>
              <th className="text-left px-1.5 py-1" style={{ color: C.gray, minWidth: 70 }}>Horizons</th>
              <th className="text-center px-1.5 py-1" style={{ color: C.gray, minWidth: 70 }}>Days Held</th>
              <th className="text-center px-1.5 py-1" style={{ color: C.gray, minWidth: 60 }}>오늘 picks</th>
              <th className="text-center px-1.5 py-1" style={{ color: C.gray, minWidth: 60 }}>Risk</th>
              <th className="text-right px-1 py-1" style={{ color: C.gray, minWidth: 50 }}>5d</th>
              <th className="text-right px-1 py-1" style={{ color: C.gray, minWidth: 50 }}>1mo</th>
              <th className="text-right px-1 py-1" style={{ color: C.gray, minWidth: 50 }}>3mo</th>
              <th className="text-right px-1 py-1" style={{ color: C.gray, minWidth: 50 }}>6mo</th>
              <th className="text-left px-1.5 py-1" style={{ color: C.gray }}>현재 상황</th>
            </tr>
          </thead>
          <tbody>
            {tickers.map((t) => {
              const recs = grouped[t];
              const first = recs[0];
              const horizons = recs.map(r => r.horizon).join(", ");
              const maxDaysHeld = Math.max(...recs.map(r => r.days_held || 0));
              const inToday = recs.some(r => r.in_today_buy_picks || r.in_today_sell_picks);
              const fmtPct = (v: number | null | undefined) =>
                v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
              const pctColor = (v: number | null | undefined) =>
                v == null ? C.gray : v > 0 ? C.green : v < 0 ? C.red : C.gray;
              return (
                <tr key={t} style={{ borderBottom: `1px solid ${C.border}40` }}>
                  <td className="px-1.5 py-1 font-mono font-bold" style={{ color: C.text }}>{t}</td>
                  <td className="px-1.5 py-1" style={{ color: C.text }}>{first.name?.slice(0, 25)}</td>
                  <td className="px-1.5 py-1" style={{ color: C.gray, fontSize: 11 }}>{horizons}</td>
                  <td className="text-center px-1.5 py-1 font-mono font-bold"
                      style={{ color: maxDaysHeld >= 5 ? C.green : C.cyan }}>
                    Day {maxDaysHeld}
                  </td>
                  <td className="text-center px-1.5 py-1" style={{ fontSize: 11 }}>
                    {inToday ? (
                      <span style={{ color: C.green, fontWeight: "bold" }}>✓ Yes</span>
                    ) : (
                      <span style={{ color: C.amber }}>× No</span>
                    )}
                  </td>
                  <td className="text-center px-1.5 py-1 font-mono font-bold"
                      style={{ color: (first.risk_score || 0) <= 35 ? C.green : (first.risk_score || 0) >= 55 ? C.red : C.amber }}>
                    {first.risk_score?.toFixed(0) ?? "—"}
                  </td>
                  <td className="text-right px-1 py-1 font-mono" style={{ color: pctColor(first.ret_5d), fontSize: 11 }}>{fmtPct(first.ret_5d)}</td>
                  <td className="text-right px-1 py-1 font-mono" style={{ color: pctColor(first.ret_1mo), fontSize: 11 }}>{fmtPct(first.ret_1mo)}</td>
                  <td className="text-right px-1 py-1 font-mono" style={{ color: pctColor(first.ret_3mo), fontSize: 11 }}>{fmtPct(first.ret_3mo)}</td>
                  <td className="text-right px-1 py-1 font-mono" style={{ color: pctColor(first.ret_6mo), fontSize: 11 }}>{fmtPct(first.ret_6mo)}</td>
                  <td className="px-1.5 py-1" style={{ color: C.text, fontSize: 11, maxWidth: 280 }}>
                    {first.in_today_buy_picks || first.in_today_sell_picks
                      ? <span style={{ color: C.green }}>✓ 오늘도 voting 통과 (안정적)</span>
                      : <span style={{ color: C.amber }}>오늘 picks 미포함 — 보유 유지 검토 필요</span>}
                    {first.last_alert && (
                      <div className="text-[10px] mt-0.5" style={{ color: C.red }}>{first.last_alert}</div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="text-[11px] mt-1" style={{ color: C.gray }}>
        ⓘ Day N = 진입 후 보유 일수 · "오늘 picks ✓ Yes" = PM Agent가 오늘도 재선정 (안정 신호) · "× No" = 오늘 미선정 (보유 유지 또는 청산 검토)
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Exit Pending Section — positions in EXIT_PENDING state
// ─────────────────────────────────────────────────────────────────────
function ExitPendingSection({ items }: { items: ExitPendingItem[] }) {
  return (
    <div className="mb-3 p-2 rounded" style={{
      backgroundColor: C.amber + "0a",
      border: `1.5px solid ${C.amber}80`,
    }}>
      <div className="text-[13px] font-bold mb-2" style={{ color: C.amber }}>
        ⚠ 청산 후보 (Exit Pending) — {items.length}종목
        <span className="text-[11px] ml-2" style={{ color: C.gray, fontWeight: "normal" }}>
          · Phase 5.6 EXIT_PENDING · 오늘 청산 검토 필요
        </span>
      </div>
      <div className="overflow-auto rounded" style={{
        maxHeight: 200, backgroundColor: C.bgAlt, border: `1px solid ${C.border}`,
      }}>
        <table className="w-full text-[12px] border-collapse">
          <thead style={{ backgroundColor: C.bgAlt }}>
            <tr style={{ borderBottom: `1px solid ${C.border}` }}>
              <th className="text-left px-1.5 py-1" style={{ color: C.gray }}>Ticker</th>
              <th className="text-left px-1.5 py-1" style={{ color: C.gray }}>Name</th>
              <th className="text-left px-1.5 py-1" style={{ color: C.gray }}>Horizon</th>
              <th className="text-center px-1.5 py-1" style={{ color: C.gray }}>Days Held</th>
              <th className="text-right px-1 py-1" style={{ color: C.gray }}>5d</th>
              <th className="text-right px-1 py-1" style={{ color: C.gray }}>1mo</th>
              <th className="text-left px-1.5 py-1" style={{ color: C.gray }}>청산 사유</th>
            </tr>
          </thead>
          <tbody>
            {items.map((r, i) => {
              const fmtPct = (v: number | null | undefined) =>
                v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
              const pctColor = (v: number | null | undefined) =>
                v == null ? C.gray : v > 0 ? C.green : v < 0 ? C.red : C.gray;
              return (
                <tr key={i} style={{ borderBottom: `1px solid ${C.border}40` }}>
                  <td className="px-1.5 py-1 font-mono font-bold" style={{ color: C.text }}>{r.ticker}</td>
                  <td className="px-1.5 py-1" style={{ color: C.text }}>{r.name?.slice(0, 25)}</td>
                  <td className="px-1.5 py-1" style={{ color: C.gray, fontSize: 11 }}>{r.horizon}</td>
                  <td className="text-center px-1.5 py-1 font-mono" style={{ color: C.amber }}>Day {r.days_held}</td>
                  <td className="text-right px-1 py-1 font-mono" style={{ color: pctColor(r.ret_5d), fontSize: 11 }}>{fmtPct(r.ret_5d)}</td>
                  <td className="text-right px-1 py-1 font-mono" style={{ color: pctColor(r.ret_1mo), fontSize: 11 }}>{fmtPct(r.ret_1mo)}</td>
                  <td className="px-1.5 py-1" style={{ color: C.text, fontSize: 11, maxWidth: 320 }}>{r.exit_reason}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
