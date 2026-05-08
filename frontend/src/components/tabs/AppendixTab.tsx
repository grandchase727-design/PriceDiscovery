import { useState } from "react";
import { type FilterParams } from "../../api/client";
import { UniverseTab } from "./UniverseTab";
import { DescriptionTab } from "./DescriptionTab";
import { EfficacyTab } from "./EfficacyTab";
import { ReferenceTab } from "./ReferenceTab";
import { ClassificationTab } from "./ClassificationTab";

const SUBS = ["Universe", "Classification", "Description", "Efficacy", "Reference"] as const;

export function AppendixTab({ filters }: { filters: FilterParams }) {
  const [sub, setSub] = useState(0);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-1 border-b border-gray-800">
        {SUBS.map((label, i) => (
          <button
            key={label}
            onClick={() => setSub(i)}
            className={`px-4 py-2 text-sm border-b-2 transition-colors ${
              sub === i
                ? "border-cyan-400 text-cyan-400"
                : "border-transparent text-gray-500 hover:text-gray-300"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {sub === 0 && <UniverseTab filters={filters} />}
      {sub === 1 && <ClassificationTab />}
      {sub === 2 && <DescriptionTab />}
      {sub === 3 && <EfficacyTab />}
      {sub === 4 && <ReferenceTab />}
    </div>
  );
}
