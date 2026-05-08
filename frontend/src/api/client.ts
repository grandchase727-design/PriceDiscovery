import axios from "axios";

const api = axios.create({ baseURL: "/api" });

export interface FilterParams {
  categories?: string[];
  sectors?: string[];
  subthemes?: string[];
  classifications?: string[];
  eligible_only?: boolean;
  comp_min?: number;
  comp_max?: number;
}

function toQuery(f: FilterParams): URLSearchParams {
  const p = new URLSearchParams();
  f.categories?.forEach((c) => p.append("categories", c));
  f.sectors?.forEach((c) => p.append("sectors", c));
  f.subthemes?.forEach((c) => p.append("subthemes", c));
  f.classifications?.forEach((c) => p.append("classifications", c));
  if (f.eligible_only) p.set("eligible_only", "true");
  if (f.comp_min !== undefined) p.set("comp_min", String(f.comp_min));
  if (f.comp_max !== undefined) p.set("comp_max", String(f.comp_max));
  return p;
}

export const fetchMeta = () => api.get("/meta").then((r) => r.data);
export const fetchOverview = (f: FilterParams) => api.get("/overview", { params: toQuery(f) }).then((r) => r.data);
export const fetchTable = (f: FilterParams) => api.get("/table", { params: toQuery(f) }).then((r) => r.data);
export const fetchUniverse = (f: FilterParams) => api.get("/universe", { params: toQuery(f) }).then((r) => r.data);
export const fetchMarketRegime = (f: FilterParams) => api.get("/market-regime", { params: toQuery(f) }).then((r) => r.data);
export const fetchWeeklyHeatmap = () => api.get("/weekly-heatmap").then((r) => r.data);
export const fetchEffectiveness = () => api.get("/effectiveness").then((r) => r.data);
export const fetchReport = (f: FilterParams) => api.get("/report", { params: toQuery(f) }).then((r) => r.data);

// Live scan
export const startScan = (opts?: { lookback_years?: number; use_realtime?: boolean; include_stocks?: boolean }) =>
  api.post("/scan", null, { params: opts }).then((r) => r.data);
export const fetchScanStatus = () => api.get("/scan/status").then((r) => r.data);
export const reloadCache = () => api.post("/reload").then((r) => r.data);
export const fetchPreMomentum = () => api.get("/pre-momentum").then((r) => r.data);

// ─── Unified classification (Phase Y — GICS / cap-tier / country) ───
export const fetchClassificationMeta = () =>
  api.get("/classification").then((r) => r.data);
export const fetchClassificationValidation = () =>
  api.get("/classification/validation").then((r) => r.data);

// ─── ML-rescored endpoints (mirror /api/* but using ML-optimized Composite weights) ───
export const fetchMlMeta = () => api.get("/ml/meta").then((r) => r.data);
export const fetchTableML = (f: FilterParams) =>
  api.get("/ml/table", { params: toQuery(f) }).then((r) => r.data);
export const fetchPreMomentumML = () => api.get("/ml/pre-momentum").then((r) => r.data);
export const fetchClassificationHistoryML = () =>
  api.get("/ml/classification-history").then((r) => r.data);
export const fetchFactorEfficacy = () => api.get("/factor-efficacy").then((r) => r.data);
export const fetchSectorRotation = () => api.get("/sector-rotation").then((r) => r.data);
export type SectorRotationSignalMode = "momentum_12_1m" | "composite_live" | "ml_momentum_blend" | "ml_lightgbm";

export const fetchSectorRotationBacktest = (
  lookback_years = 5,
  top_n = 3,
  turnover_bp = 30,
  signal_mode: SectorRotationSignalMode = "momentum_12_1m",
  vol_target_pct = 0,
  vol_lookback_months = 6,
  max_leverage = 1.0,
) =>
  api.get("/sector-rotation/backtest", {
    params: {
      lookback_years, top_n, turnover_bp, signal_mode,
      vol_target_pct, vol_lookback_months, max_leverage,
    },
  }).then((r) => r.data);
export const fetchClassificationHistory = () => api.get("/classification-history").then((r) => r.data);
export const fetchClassificationHistoryBySector = () => api.get("/classification-history-by-sector").then((r) => r.data);
export const fetchAIPrediction = () => api.get("/ai-prediction").then((r) => r.data);
export const fetchAIPerformance = () => api.get("/ai-performance").then((r) => r.data);
export const fetchAIBenchmarks = () => api.get("/ai-benchmarks").then((r) => r.data);
export const fetchAIWinRatio = () => api.get("/ai-winratio").then((r) => r.data);
