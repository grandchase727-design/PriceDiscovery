/**
 * ElliottWavePanel — Elliott-Wave phase labeler for the 25 broad/sector/leveraged
 * tickers (MSCI ACWI · S&P500 · Nasdaq-100 · Russell-2000 · GICS 11 섹터 · 레버리지 ETF).
 *
 * Renders immediately ABOVE the 매수 Final List on every tab. Fetches
 * /api/elliott-wave-indices on mount AND whenever the period toggle (YTD / 1개월)
 * changes, drawing a 2×2 grid of close-price charts, each overlaid with the
 * labeled swing pivots (1..5 / A..C) plus a colored phase badge + Korean
 * interpretation.
 */
import { useEffect, useState } from "react";
import { fetchElliottWaveIndices } from "../../api/client";
import { C } from "../../styles/theme";
import { IndexCard } from "./ElliottWaveCard";

type Period = "ytd" | "1m";
const PERIOD_OPTIONS: { key: Period; label: string }[] = [
  { key: "ytd", label: "YTD" },
  { key: "1m", label: "1개월" },
];

// A group as consumed by the panel: header meta + its own index list.
interface WaveGroup {
  key: string;
  label: string;
  emoji: string;
  default_open: boolean;
  indices: any[];
}

// Normalize the API payload into an ordered WaveGroup[].
// New contract: data.groups (ordered broad/sector/leveraged).
// Backward-compat: if data.groups is absent but data.indices exists, wrap the
// flat list into a single "broad" group so old backends keep rendering.
function toGroups(data: any): WaveGroup[] {
  if (data?.groups && Array.isArray(data.groups) && data.groups.length > 0) {
    return data.groups.map((g: any) => ({
      key: String(g.key ?? ""),
      label: String(g.label ?? ""),
      emoji: String(g.emoji ?? ""),
      default_open: g.default_open === true,
      indices: Array.isArray(g.indices) ? g.indices : [],
    }));
  }
  if (data?.indices && Array.isArray(data.indices) && data.indices.length > 0) {
    return [
      {
        key: "broad",
        label: "광역 지수",
        emoji: "📊",
        default_open: true,
        indices: data.indices,
      },
    ];
  }
  return [];
}

export default function ElliottWavePanel() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  // Per-group open/closed state, keyed by group.key.
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [period, setPeriod] = useState<Period>("ytd");

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setErr(null);
    fetchElliottWaveIndices(period)
      .then((r) => {
        if (!alive) return;
        setData(r);
        // Seed open state from each group's default_open (broad open, rest collapsed).
        const seed: Record<string, boolean> = {};
        toGroups(r).forEach((g) => {
          seed[g.key] = g.default_open;
        });
        setOpen(seed);
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

  const groups = toGroups(data);
  const totalIndices = groups.reduce((n, g) => n + g.indices.length, 0);

  const toggle = (key: string) =>
    setOpen((prev) => ({ ...prev, [key]: !prev[key] }));

  return (
    <div
      className="mt-6 mb-4 px-3 py-3 rounded"
      style={{ backgroundColor: C.bg, border: `2px solid ${C.blue}55` }}
    >
      <div className="flex items-center gap-2 mb-3">
        <div>
          <div className="text-[15px] font-bold" style={{ color: C.blue }}>
            🌊 시장 매매전략
          </div>
          <div className="text-[12px]" style={{ color: C.gray }}>
            광역 지수 · GICS 11 섹터 · 레버리지 ETF — 엘리엇 파동 국면 + CAN SLIM 진입/손절
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
        </div>
      </div>

      {loading && !data && (
        <div className="px-3 py-2 text-[12px]" style={{ color: C.gray }}>
          25개 지수·ETF 파동 분석 중… (최초 1회 ~20초 소요)
        </div>
      )}
      {err && !data && (
        <div className="px-3 py-2 text-[12px]" style={{ color: C.red }}>
          Error: {err}
        </div>
      )}
      {!loading && !err && totalIndices === 0 && (
        <div className="px-3 py-2 text-[12px]" style={{ color: C.gray }}>
          파동 데이터가 없습니다.
        </div>
      )}

      {groups.length > 0 && (
        <div className="flex flex-col gap-3">
          {groups.map((g) => {
            const isOpen = open[g.key] === true;
            return (
              <div key={g.key}>
                {/* Section-bar header (clickable) */}
                <button
                  type="button"
                  onClick={() => toggle(g.key)}
                  className="w-full flex items-center gap-2 rounded px-3 py-2 text-left"
                  style={{
                    backgroundColor: C.bgAlt,
                    border: `1px solid ${C.border}`,
                  }}
                  aria-expanded={isOpen}
                >
                  <span className="text-[13px] font-bold" style={{ color: C.text }}>
                    {g.emoji} {g.label}
                  </span>
                  <span className="text-[12px]" style={{ color: C.gray }}>
                    ({g.indices.length}개)
                  </span>
                  <span className="ml-auto text-[12px]" style={{ color: C.gray }}>
                    {isOpen ? "▾" : "▸"}
                  </span>
                </button>

                {/* Cards grid (only when open) */}
                {isOpen && g.indices.length > 0 && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-3">
                    {g.indices.map((idx: any) => (
                      <IndexCard key={idx.ticker} idx={idx} />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
