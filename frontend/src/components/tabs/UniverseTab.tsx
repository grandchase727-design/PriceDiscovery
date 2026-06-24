import { useEffect, useMemo, useState } from "react";
import { fetchUniverse, type FilterParams } from "../../api/client";
import { DataTable } from "../shared/DataTable";
import { ColDefToggle } from "../shared/ColDefToggle";
import type { ColumnDef } from "@tanstack/react-table";

function pctCell(info: any) {
  const v = info.getValue();
  if (v == null || v === "" || isNaN(Number(v))) return <span className="text-[#857F7A]">-</span>;
  const n = Number(v);
  const color = n > 0 ? "text-[#0A7D3F]" : n < 0 ? "text-[#CC0000]" : "text-[#66605C]";
  return <span className={color}>{n.toFixed(2)}%</span>;
}

const baseColumns: ColumnDef<any, any>[] = [
  { accessorKey: "ticker", header: "Ticker", cell: (info) => <span className="font-semibold text-[#0D7680]">{info.getValue()}</span> },
  { accessorKey: "name", header: "Name", cell: (info) => <span className="max-w-[200px] truncate block" title={info.getValue()}>{info.getValue()}</span> },
  { accessorKey: "category", header: "Sector" },
  { accessorKey: "theme", header: "Theme", cell: (info) => { const v = info.getValue(); return v === "-" ? <span className="text-[#857F7A]">-</span> : v; } },
  { accessorKey: "mktcap_B", header: "Mkt Cap ($B)", cell: (info) => {
    const v = info.getValue();
    if (v == null || Number(v) === 0) return <span className="text-[#857F7A]">-</span>;
    const n = Number(v);
    return <span className="text-[#33302E]">{n >= 1 ? n.toFixed(1) : n.toFixed(2)}</span>;
  }},
  { accessorKey: "ret_1d", header: "1D", cell: pctCell },
  { accessorKey: "ret_5d", header: "5D", cell: pctCell },
  { accessorKey: "ret_21d", header: "1M", cell: pctCell },
  { accessorKey: "ret_63d", header: "3M", cell: pctCell },
  { accessorKey: "ret_126d", header: "6M", cell: pctCell },
  { accessorKey: "ret_ytd", header: "YTD", cell: pctCell },
  { accessorKey: "ret_252d", header: "1Y", cell: pctCell },
  { accessorKey: "ret_3y_ann", header: "3Y/A", cell: pctCell },
  { accessorKey: "ret_5y_ann", header: "5Y/A", cell: pctCell },
  { accessorKey: "vol_3y_ann", header: "Vol 3Y/A", cell: pctCell },
];

// Stock view: drop "Theme" only if useless (otherwise keep both)
const stockColumns: ColumnDef<any, any>[] = baseColumns;
// ETF view: drop "Theme" column since ETFs typically have no theme
const etfColumns: ColumnDef<any, any>[] = baseColumns.filter(
  (c: any) => c.accessorKey !== "theme",
);

type AssetView = "all" | "stock" | "etf";

export function UniverseTab({ filters }: { filters: FilterParams }) {
  const [data, setData] = useState<any>(null);
  const [view, setView] = useState<AssetView>("stock");

  useEffect(() => {
    fetchUniverse(filters).then(setData);
  }, [filters]);

  const { stockRows, etfRows, allRows } = useMemo(() => {
    const all = (data?.rows || []) as any[];
    const stocks: any[] = [];
    const etfs:   any[] = [];
    for (const r of all) {
      const t = (r.asset_type || "").toString().toLowerCase();
      if (t === "stock") stocks.push(r);
      else if (t === "etf") etfs.push(r);
    }
    return { stockRows: stocks, etfRows: etfs, allRows: all };
  }, [data]);

  if (!data) return <div className="text-[#857F7A]">Loading...</div>;

  const counts = { all: allRows.length, stock: stockRows.length, etf: etfRows.length };

  const TABS: { key: AssetView; label: string; emoji: string; color: string; count: number }[] = [
    { key: "stock", label: "Stocks",    emoji: "📈", color: "cyan",   count: counts.stock },
    { key: "etf",   label: "ETFs",      emoji: "📦", color: "purple", count: counts.etf   },
    { key: "all",   label: "All (Stocks + ETFs)", emoji: "📊", color: "gray",   count: counts.all   },
  ];

  const currentRows  = view === "stock" ? stockRows : view === "etf" ? etfRows : allRows;
  const currentCols  = view === "etf"   ? etfColumns : stockColumns;
  const viewLabel    = view === "stock" ? "Stocks" : view === "etf" ? "ETFs" : "All";

  return (
    <div>
      <div className="flex items-baseline justify-between mb-3">
        <h2 className="text-[20px] font-bold text-[#33302E]">Universe — {viewLabel}</h2>
        <div className="text-[12px] text-[#857F7A]">
          전체 {counts.all}종목 = Stocks {counts.stock} + ETFs {counts.etf}
        </div>
      </div>

      {/* Sub-tab toggle: Stocks / ETFs / All */}
      <div className="flex items-center gap-2 mb-3">
        {TABS.map((tab) => {
          const active = view === tab.key;
          const colorMap: Record<string, string> = {
            cyan:   active ? "bg-[#0D7680]/25 text-[#0D7680] border-[#0F5499]" : "text-[#0F5499]/60 border-[#9CC3D5]/40",
            purple: active ? "bg-[#EFE9F5]/25 text-[#7D5BA6] border-[#C9B8DC]" : "text-[#7D5BA6]/60 border-[#C9B8DC]/40",
            gray:   active ? "bg-[#CCC1B7]/25 text-[#33302E] border-[#CCC1B7]" : "text-[#857F7A]/60 border-[#E6D9CE]",
          };
          return (
            <button key={tab.key}
                    onClick={() => setView(tab.key)}
                    className={`px-4 py-1.5 rounded text-[14px] border transition-colors ${colorMap[tab.color]}`}
                    style={{ fontWeight: active ? "bold" : "normal" }}>
              {tab.emoji} {tab.label} <span className="ml-1 opacity-60">({tab.count})</span>
            </button>
          );
        })}
      </div>

      <ColDefToggle defs={[
        { col: "Ticker", desc: "종목 코드" },
        { col: "Name", desc: "종목명" },
        { col: "Sector", desc: "통합 섹터 분류 (Technology / Healthcare / Financials / ... — 주식과 sector ETF는 같은 버킷에 통합)" },
        { col: "Theme", desc: "개별주 테마 (Stocks 보기에서만 표시; ETFs는 숨김)" },
        { col: "Mkt Cap ($B)", desc: "시가총액 (USD, Billions). 한국 종목은 USD/KRW 환산" },
        { col: "1D~1Y", desc: "기간별 수익률 (%): 1일, 5일, 1개월(21일), 3개월, 6개월, YTD(연초부터), 1년" },
        { col: "3Y/A · 5Y/A", desc: "3년/5년 연환산 수익률 (%)" },
        { col: "Vol 3Y/A", desc: "3년 연환산 변동성 (%). 일일 수익률 표준편차 × √252" },
      ]} />

      <DataTable data={currentRows} columns={currentCols} maxHeight="calc(100vh - 200px)" />
    </div>
  );
}
