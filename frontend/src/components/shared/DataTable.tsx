import { useState, useMemo } from "react";
import {
  useReactTable, getCoreRowModel, getSortedRowModel, getFilteredRowModel,
  flexRender, type ColumnDef, type SortingState,
} from "@tanstack/react-table";

interface Props<T> {
  data: T[];
  columns: ColumnDef<T, any>[];
  maxHeight?: string;
}

export function DataTable<T extends Record<string, any>>({ data, columns, maxHeight = "600px" }: Props<T>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");

  const table = useReactTable({
    data, columns, state: { sorting, globalFilter },
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
  });

  return (
    <div>
      <input
        className="mb-2 px-3 py-1.5 bg-[#F2E5D7] border border-[#E6D9CE] rounded text-[16px] text-[#33302E] w-64"
        placeholder="Search..."
        value={globalFilter}
        onChange={(e) => setGlobalFilter(e.target.value)}
      />
      <div className="overflow-auto border border-[#E6D9CE] rounded" style={{ maxHeight }}>
        <table className="w-full text-[14px]">
          <thead className="sticky top-0 bg-[#F2E5D7] z-10">
            {table.getHeaderGroups().map((hg) => (
              <tr key={hg.id}>
                {hg.headers.map((h) => (
                  <th key={h.id}
                    className="px-2 py-1.5 text-left text-[#66605C] cursor-pointer select-none whitespace-nowrap border-b border-[#E6D9CE]"
                    onClick={h.column.getToggleSortingHandler()}>
                    {flexRender(h.column.columnDef.header, h.getContext())}
                    {{ asc: " ▲", desc: " ▼" }[h.column.getIsSorted() as string] ?? ""}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="border-b border-[#E6D9CE]/50 hover:bg-[#F2E5D7]/50">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-2 py-1 whitespace-nowrap">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="text-[14px] text-[#857F7A] mt-1">{table.getRowModel().rows.length} rows</div>
    </div>
  );
}
