import { useState } from "react";
import { type FilterParams } from "../../api/client";
import { FactorEfficacyTab } from "./FactorEfficacyTab";
import { MarketRegimeTab } from "./MarketRegimeTab";
import { C } from "../../styles/theme";

const SUBS = ["Factor Efficacy", "Market Regime"] as const;

export function MarketEnvironmentTab({ filters }: { filters: FilterParams }) {
  const [sub, setSub] = useState(0);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-1 border-b border-[#E6D9CE]">
        {SUBS.map((label, i) => (
          <button
            key={label}
            onClick={() => setSub(i)}
            className={`px-4 py-2 text-[16px] border-b-2 transition-colors ${
              sub === i
                ? "border-[#0F5499] text-[#0F5499]"
                : "border-transparent text-[#857F7A] hover:text-[#33302E]"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {sub === 0 && <FactorEfficacyTab />}
      {sub === 1 && <MarketRegimeTab filters={filters} />}
    </div>
  );
}
