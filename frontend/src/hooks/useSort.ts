import { useState, useMemo } from "react";

export type SortDirection = "asc" | "desc";
export interface SortState {
  key: string;
  dir: SortDirection;
}

export type Accessor<T> = (row: T) => any;
export type AccessorMap<T> = Record<string, Accessor<T>>;

/**
 * useSort — sortable column hook.
 *
 * Usage:
 *   const accessors = { ticker: r => r.ticker, comp: r => r.composite, ... };
 *   const { sorted, sortKey, sortDir, onSort, indicator } = useSort(rows, accessors, { key: "comp", dir: "desc" });
 */
export function useSort<T>(
  rows: T[],
  accessors: AccessorMap<T>,
  initial?: SortState
) {
  const [sort, setSort] = useState<SortState | null>(initial ?? null);

  const sorted = useMemo(() => {
    if (!sort || !accessors[sort.key]) return rows;
    const fn = accessors[sort.key];
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const va = fn(a);
      const vb = fn(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "number" && typeof vb === "number") return (va - vb) * dir;
      return String(va).localeCompare(String(vb)) * dir;
    });
  }, [rows, sort, accessors]);

  const onSort = (key: string) => {
    setSort((prev) => {
      if (!prev || prev.key !== key) {
        // First click: descending for numeric-friendly UX
        return { key, dir: "desc" };
      }
      if (prev.dir === "desc") return { key, dir: "asc" };
      return null; // third click clears
    });
  };

  const indicator = (key: string) => {
    if (!sort || sort.key !== key) return "";
    return sort.dir === "desc" ? " \u25BC" : " \u25B2";
  };

  return { sorted, sortKey: sort?.key, sortDir: sort?.dir, onSort, indicator };
}
