import { useState } from "react";

interface ColDef { col: string; desc: string }

export function ColDefToggle({ title, defs }: { title?: string; defs: ColDef[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border border-[#E6D9CE] rounded-lg overflow-hidden mb-3">
      <button className="w-full px-4 py-2 text-left text-[14px] font-semibold bg-[#FFFFFF] hover:bg-[#F2E5D7] flex justify-between"
        onClick={() => setOpen(!open)}>
        <span>{title || "Column Definitions"}</span>
        <span className="text-[#857F7A]">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="p-3 bg-[#FBEEE3]">
          <table className="w-full text-[12px] border-collapse">
            <thead><tr className="border-b border-[#E6D9CE]">
              <th className="text-left py-1 px-2 text-[#857F7A] w-28">Column</th>
              <th className="text-left py-1 px-2 text-[#857F7A]">Description</th>
            </tr></thead>
            <tbody>
              {defs.map((d) => (
                <tr key={d.col} className="border-b border-[#E6D9CE]/50">
                  <td className="py-0.5 px-2 font-mono text-[#0F5499] whitespace-nowrap">{d.col}</td>
                  <td className="py-0.5 px-2 text-[#66605C]">{d.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
