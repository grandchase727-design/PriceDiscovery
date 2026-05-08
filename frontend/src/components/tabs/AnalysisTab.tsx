import { type FilterParams } from "../../api/client";
import { SectorRotationTab } from "./SectorRotationTab";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function AnalysisTab({ filters: _filters }: { filters: FilterParams }) {
  return <SectorRotationTab />;
}
