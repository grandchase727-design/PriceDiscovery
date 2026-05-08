import { CLASS_COLORS } from "../../styles/theme";
import type { ColumnDef } from "@tanstack/react-table";

function numCol(key: string, header: string, fmt = 1): ColumnDef<any, any> {
  return {
    accessorKey: key, header,
    cell: (info) => {
      const v = info.getValue();
      return v != null ? Number(v).toFixed(fmt) : "-";
    },
  };
}

function pctCol(key: string, header: string): ColumnDef<any, any> {
  return {
    accessorKey: key, header,
    cell: (info) => {
      const v = Number(info.getValue());
      const color = v > 0 ? "text-green-400" : v < 0 ? "text-red-400" : "";
      return <span className={color}>{v.toFixed(2)}%</span>;
    },
  };
}

function signalCol(key: string, header: string, isLong: boolean): ColumnDef<any, any> {
  return {
    accessorKey: key, header,
    cell: (info) => {
      const v = Number(info.getValue());
      if (v == null || isNaN(v)) return <span className="text-gray-600">-</span>;
      const color = isLong
        ? (v >= 70 ? "text-green-400 font-bold" : v >= 50 ? "text-green-400/70" : v >= 30 ? "text-gray-400" : "text-gray-600")
        : (v >= 70 ? "text-red-400 font-bold" : v >= 50 ? "text-red-400/70" : v >= 30 ? "text-gray-400" : "text-gray-600");
      return <span className={`${color} text-[10px]`}>{v.toFixed(0)}</span>;
    },
  };
}

export const detailColumns: ColumnDef<any, any>[] = [
  { accessorKey: "ticker", header: "Ticker", cell: (i) => <span className="font-mono font-bold">{i.getValue()}</span> },
  { accessorKey: "name", header: "Name", cell: (i) => <span className="text-gray-400 truncate max-w-[160px] block">{String(i.getValue()).slice(0, 25)}</span> },
  { accessorKey: "sector", header: "Sector", cell: (i) => <span className="text-cyan-400/70 text-[10px]">{String(i.getValue())}</span> },
  { accessorKey: "category", header: "Category", cell: (i) => <span className="text-gray-500 text-[10px]">{String(i.getValue()).replace("STK_", "")}</span> },
  { accessorKey: "theme", header: "Theme", cell: (i) => {
    const v = String(i.getValue());
    return v === "-" ? <span className="text-gray-700">-</span> : <span className="text-gray-400 text-[10px]">{v}</span>;
  }},
  numCol("composite", "Comp"),
  numCol("tcs", "TCS", 0), numCol("tfs", "TFS", 0), numCol("oer", "OER", 0), numCol("rss", "RSS", 0),
  {
    accessorKey: "classification", header: "Class",
    cell: (info) => {
      const cls = String(info.getValue());
      const color = CLASS_COLORS[cls] || "#6b7280";
      return <span style={{ color }} className="text-[10px] font-medium">{cls.slice(0, 16)}</span>;
    },
  },
  { accessorKey: "eligible", header: "Elg", cell: (i) => i.getValue() ? "✅" : "" },
  { accessorKey: "rejection", header: "Rejection", cell: (i) => <span className="text-gray-500 text-[10px]">{String(i.getValue())}</span> },
  numCol("rsi", "RSI"), numCol("trend_age", "Age", 0),
  signalCol("oneil_long", "Long", true), signalCol("oneil_short", "Short", false),
  {
    accessorKey: "event_flag", header: "Risk",
    cell: (info) => {
      const v = info.getValue();
      const reasons = String(info.row.original?.event_reasons || "");
      if (v) return <span className="text-yellow-400 font-semibold text-[9px]" title={reasons}>⚡ EVENT</span>;
      return <span className="text-gray-700 text-[9px]">—</span>;
    },
  },
  numCol("structural_q", "SQ", 0),
  numCol("val_prob", "Val%"), numCol("val_persist", "Pst", 0),
  pctCol("ret_1w", "1W"), pctCol("ret_1m", "1M"), pctCol("ret_3m", "3M"),
];
