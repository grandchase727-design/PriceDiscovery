import { useState } from "react";

interface ColDef { col: string; desc: string }

export function ColDefToggle({ title, defs }: { title?: string; defs: ColDef[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-gray-800 rounded-lg overflow-hidden mb-3">
      <button className="w-full px-4 py-2 text-left text-xs font-semibold bg-[#111827] hover:bg-[#1f2937] flex justify-between"
        onClick={() => setOpen(!open)}>
        <span>{title || "Column Definitions"}</span>
        <span className="text-gray-500">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="p-3 bg-[#0d1117]">
          <table className="w-full text-[10px] border-collapse">
            <thead><tr className="border-b border-gray-700">
              <th className="text-left py-1 px-2 text-gray-500 w-28">Column</th>
              <th className="text-left py-1 px-2 text-gray-500">Description</th>
            </tr></thead>
            <tbody>
              {defs.map((d) => (
                <tr key={d.col} className="border-b border-gray-800/50">
                  <td className="py-0.5 px-2 font-mono text-cyan-400 whitespace-nowrap">{d.col}</td>
                  <td className="py-0.5 px-2 text-gray-400">{d.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
