import { useEffect, useState } from "react";
import { fetchUniverse, type FilterParams } from "../../api/client";
import { DataTable } from "../shared/DataTable";
import { ColDefToggle } from "../shared/ColDefToggle";
import type { ColumnDef } from "@tanstack/react-table";

function pctCell(info: any) {
  const v = info.getValue();
  if (v == null || v === "" || isNaN(Number(v))) return <span className="text-gray-600">-</span>;
  const n = Number(v);
  const color = n > 0 ? "text-green-400" : n < 0 ? "text-red-400" : "text-gray-400";
  return <span className={color}>{n.toFixed(2)}%</span>;
}

const columns: ColumnDef<any, any>[] = [
  { accessorKey: "ticker", header: "Ticker", cell: (info) => <span className="font-semibold text-cyan-300">{info.getValue()}</span> },
  { accessorKey: "name", header: "Name", cell: (info) => <span className="max-w-[200px] truncate block" title={info.getValue()}>{info.getValue()}</span> },
  { accessorKey: "category", header: "Category" },
  { accessorKey: "theme", header: "Theme", cell: (info) => { const v = info.getValue(); return v === "-" ? <span className="text-gray-600">-</span> : v; } },
  { accessorKey: "mktcap_B", header: "Mkt Cap ($B)", cell: (info) => {
    const v = info.getValue();
    if (v == null || Number(v) === 0) return <span className="text-gray-600">-</span>;
    const n = Number(v);
    return <span className="text-gray-300">{n >= 1 ? n.toFixed(1) : n.toFixed(2)}</span>;
  }},
  { accessorKey: "ret_1d", header: "1D", cell: pctCell },
  { accessorKey: "ret_5d", header: "5D", cell: pctCell },
  { accessorKey: "ret_21d", header: "1M", cell: pctCell },
  { accessorKey: "ret_63d", header: "3M", cell: pctCell },
  { accessorKey: "ret_126d", header: "6M", cell: pctCell },
  { accessorKey: "ret_252d", header: "1Y", cell: pctCell },
  { accessorKey: "ret_3y_ann", header: "3Y/A", cell: pctCell },
  { accessorKey: "ret_5y_ann", header: "5Y/A", cell: pctCell },
  { accessorKey: "vol_3y_ann", header: "Vol 3Y/A", cell: pctCell },
];

export function UniverseTab({ filters }: { filters: FilterParams }) {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    fetchUniverse(filters).then(setData);
  }, [filters]);

  if (!data) return <div className="text-gray-500">Loading...</div>;

  return (
    <div>
      <h2 className="text-lg font-bold text-gray-200 mb-3">Universe</h2>
      <ColDefToggle defs={[
        { col: "Ticker", desc: "종목 코드" },
        { col: "Name", desc: "종목명" },
        { col: "Category", desc: "ETF/주식 카테고리" },
        { col: "Theme", desc: "개별주 테마 (ETF는 '-')" },
        { col: "Mkt Cap ($B)", desc: "시가총액 (USD, Billions). 한국 종목은 USD/KRW 환산" },
        { col: "1D~1Y", desc: "기간별 수익률 (%): 1일, 5일, 1개월(21일), 3개월, 6개월, 1년" },
        { col: "3Y/A · 5Y/A", desc: "3년/5년 연환산 수익률 (%)" },
        { col: "Vol 3Y/A", desc: "3년 연환산 변동성 (%). 일일 수익률 표준편차 × √252" },
      ]} />
      <DataTable data={data.rows || []} columns={columns} maxHeight="calc(100vh - 160px)" />
    </div>
  );
}
