/**
 * FinalListEtfWavePanel — Elliott-wave phase analysis for every ticker
 * (stocks AND ETFs) in the live buy Final List.
 *
 * Fetches /api/final-list/elliott-waves on mount AND whenever the period toggle
 * (YTD / 1개월) changes (which reads the current buy_list live server-side) and
 * renders one <IndexCard> per ticker, using the exact same card shape as the
 * 4-core-indices ElliottWavePanel.
 *
 * Rendered immediately ABOVE the Portfolio panel on every tab (tab 0 via
 * MarketCommentaryTab, tabs 1-7 via App.tsx). Collapsible, default OPEN.
 */
import { useEffect, useState } from "react";
import { fetchFinalListEtfWaves } from "../../api/client";
import { C } from "../../styles/theme";
import { IndexCard } from "./ElliottWaveCard";
import StrategyIntegratedSummary from "./StrategyIntegratedSummary";

type Period = "ytd" | "1m";
const PERIOD_OPTIONS: { key: Period; label: string }[] = [
  { key: "ytd", label: "YTD" },
  { key: "1m", label: "1개월" },
];

export default function FinalListEtfWavePanel() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState(true); // default OPEN
  const [assetFilter, setAssetFilter] = useState<"all" | "stock" | "etf">("all");
  const [period, setPeriod] = useState<Period>("ytd");

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setErr(null);
    fetchFinalListEtfWaves(period)
      .then((r) => {
        if (alive) setData(r);
      })
      .catch((e: any) => {
        if (alive) setErr(e?.message || String(e));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
  }, [period]);

  const indices: any[] = Array.isArray(data?.indices) ? data.indices : [];
  const nStocks: number = data?.n_stocks ?? 0;
  const nEtfs: number = data?.n_etfs ?? indices.length;
  const filteredIndices =
    assetFilter === "all"
      ? indices
      : indices.filter(
          (idx: any) =>
            String(idx.asset_type || "").toLowerCase() === assetFilter
        );

  return (
    <div
      className="mt-6 mb-4 px-3 py-3 rounded"
      style={{ backgroundColor: C.bg, border: `2px solid ${C.blue}55` }}
    >
      {/* Header row: title + period toggle + collapse chevron */}
      <div className="flex items-center gap-2 mb-3">
        <div>
          <div className="text-[15px] font-bold" style={{ color: C.blue }}>
            🌊 매매전략
          </div>
          <div className="text-[12px]" style={{ color: C.gray }}>
            매수 Final List 전종목 (주식 {nStocks} · ETF {nEtfs}) · 파동 국면 + 진입/손절 참고
            {data?.period_label ? ` · 분석기간 ${data.period_label}` : ""}
            {data?.as_of ? ` · as_of ${data.as_of}` : ""}
          </div>
        </div>
        <div className="ml-auto flex items-center gap-1">
          {PERIOD_OPTIONS.map((opt) => {
            const active = period === opt.key;
            return (
              <button
                key={opt.key}
                type="button"
                onClick={() => setPeriod(opt.key)}
                className="rounded px-2 py-1 text-[12px] font-semibold"
                style={{
                  backgroundColor: active ? C.blue : C.bgAlt,
                  color: active ? "#fff" : C.gray,
                  border: `1px solid ${active ? C.blue : C.border}`,
                }}
              >
                {opt.label}
              </button>
            );
          })}
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="rounded px-2 py-1 text-[13px]"
            style={{ backgroundColor: C.bgAlt, border: `1px solid ${C.border}`, color: C.gray }}
            aria-expanded={open}
          >
            {open ? "▾" : "▸"}
          </button>
        </div>
      </div>

      {loading && !data && (
        <div className="px-3 py-2 text-[12px]" style={{ color: C.gray }}>
          매수 Final List 파동 분석 중… (최초 1회 ~10초)
        </div>
      )}
      {err && !data && (
        <div className="px-3 py-2 text-[12px]" style={{ color: C.red }}>
          Error: {err}
        </div>
      )}
      {!loading && !err && indices.length === 0 && (
        <div className="px-3 py-2 text-[12px]" style={{ color: C.gray }}>
          매수 Final List가 비어 있습니다.
        </div>
      )}

      {open && indices.length > 0 && (
        <>
          {/* 3전략 통합 요약 + 합치도 히트맵 — 종목 카드 바로 위 */}
          <StrategyIntegratedSummary indices={indices} />

          {/* Stock / ETF filter toggle */}
          <div className="flex items-center gap-2 mb-3">
            {(
              [
                { key: "all", label: `전체 ${indices.length}` },
                { key: "stock", label: `Stock ${nStocks}` },
                { key: "etf", label: `ETF ${nEtfs}` },
              ] as const
            ).map((btn) => {
              const active = assetFilter === btn.key;
              return (
                <button
                  key={btn.key}
                  type="button"
                  onClick={() => setAssetFilter(btn.key)}
                  className="rounded px-3 py-1 text-[12px] font-semibold"
                  style={{
                    backgroundColor: active ? C.blue : C.bgAlt,
                    color: active ? "#fff" : C.gray,
                    border: `1px solid ${active ? C.blue : C.border}`,
                  }}
                >
                  {btn.label}
                </button>
              );
            })}
          </div>

          {filteredIndices.length === 0 ? (
            <div className="px-3 py-2 text-[12px]" style={{ color: C.gray }}>
              해당 자산군 종목이 없습니다.
            </div>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {filteredIndices.map((idx: any) => (
                <IndexCard key={idx.ticker} idx={idx} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
