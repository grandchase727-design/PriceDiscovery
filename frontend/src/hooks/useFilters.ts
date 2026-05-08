import { useState } from "react";
import type { FilterParams } from "../api/client";

export function useFilters(allCategories: string[], allClassifications: string[]) {
  const [categories, setCategories] = useState<string[]>(allCategories);
  const [classifications, setClassifications] = useState<string[]>(allClassifications);
  const [eligibleOnly, setEligibleOnly] = useState(false);
  const [compRange, setCompRange] = useState<[number, number]>([0, 100]);

  const params: FilterParams = {
    categories: categories.length === allCategories.length ? undefined : categories,
    classifications: classifications.length === allClassifications.length ? undefined : classifications,
    eligible_only: eligibleOnly,
    comp_min: compRange[0],
    comp_max: compRange[1],
  };

  return {
    categories, setCategories,
    classifications, setClassifications,
    eligibleOnly, setEligibleOnly,
    compRange, setCompRange,
    params,
  };
}
