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
export const fetchValidation = () => api.get("/validation").then((r) => r.data);
export const fetchQuantStrategies = () => api.get("/quant-strategies").then((r) => r.data);
export const fetchNewPDValidation = () => api.get("/new-pd/validation").then((r) => r.data);
export const fetchNewPDv2Validation = () => api.get("/new-pd-v2/validation").then((r) => r.data);

// ─── ML-rescored endpoints (mirror /api/* but using ML-optimized Composite weights) ───
export const fetchMlMeta = () => api.get("/ml/meta").then((r) => r.data);
export const fetchTableML = (f: FilterParams) =>
  api.get("/ml/table", { params: toQuery(f) }).then((r) => r.data);
export const fetchPreMomentumML = () => api.get("/ml/pre-momentum").then((r) => r.data);
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

// ─── Multi-Agent (Tier-A) ConvictionDebate — Top 5 stocks + Top 5 ETFs ───
export interface SpecialistRoundOpinion {
  rating: "BUY" | "HOLD" | "SELL" | "STRONG_BUY" | "AVOID" | string;
  confidence: number;
  key_points?: string[];
  biggest_risk?: string;
  biggest_opportunity?: string;
  raw_text?: string;
}
export interface MultiAgentRoundSnapshot {
  round: number;
  fundamental: SpecialistRoundOpinion;
  sentiment:   SpecialistRoundOpinion;
  valuation:   SpecialistRoundOpinion;
}
export interface MultiAgentSynthesis {
  rating: "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "AVOID" | string;
  position_modifier: number;
  sizing_recommendation: string;
  reasoning: string;
}
export interface MultiAgentVerdict {
  ticker: string;
  asset_type: "stock" | "etf" | string;
  side: "long" | "short" | string;
  group: "momentum" | "pre_momentum" | string;
  tier: "A" | string;
  synthesis_neutral: MultiAgentSynthesis;
  synthesis_averse:  MultiAgentSynthesis;
  rounds: MultiAgentRoundSnapshot[];
  converged_round: number;
  disagreement: {
    rating_axis?: number;
    specialist_dispersion?: number;
    type?: string;
  };
  composite_at_time: number | null;
  classification_at_time: string | null;
  generated_at: string;
}
export interface MultiAgentDebateResponse {
  last_update: string | null;
  stale_minutes: number | null;
  n_verdicts: number;
  n_momentum?: number;
  n_pre_momentum?: number;
  verdicts: MultiAgentVerdict[];
  error?: string;
}
export const fetchConvictionDebateMulti = (): Promise<MultiAgentDebateResponse> =>
  api.get("/conviction-debate/multi").then((r) => r.data);

export interface RefreshQueueResponse {
  queue_file?: string;
  n_queued: number;
  instructions?: string;
  error?: string;
}
export const refreshDebateQueue = (): Promise<RefreshQueueResponse> =>
  api.post("/conviction-debate/refresh-queue").then((r) => r.data);

export interface AutoFillStartResponse {
  status: "started" | "already_running" | "no_claude_cli" | "no_queue" | "empty_queue" | string;
  n_total?: number;
  hint?: string;
}
export interface AutoFillStatus {
  running: boolean;
  started_at: string;
  finished_at: string;
  n_total: number;
  n_completed: number;
  n_failed: number;
  n_persisted: number;
  last_error: string;
  current: string;
  errors: string[];
}
export const startAutoFill = (): Promise<AutoFillStartResponse> =>
  api.post("/conviction-debate/auto-fill").then((r) => r.data);
export const fetchAutoFillStatus = (): Promise<AutoFillStatus> =>
  api.get("/conviction-debate/auto-fill/status").then((r) => r.data);

// ─── Market Leaders 6-Agent Swarm ───
export interface SwarmPhase1Verdict {
  agent: string;
  rating: string;
  confidence: number;
  narrative: string;
  key_signals: string[];
  biggest_risk: string;
  biggest_opportunity: string;
  websearch_queries: string[];
}
export interface SwarmPhase2Coherence {
  coherent: boolean;
  dominant_signal: string;
  contested_areas: string[];
  confidence_weighted_winner?: string;
  reasoning: string;
}
export interface SwarmSynthesis {
  regime_tag: string;
  confidence: number;
  narrative: string;
  historical_analog: string;
  watch_triggers: string[];
  cross_panel_coherence_score: number;
  key_risks: string[];
}
export interface SwarmPick {
  ticker: string;
  name: string;
  composite: number;
  sector?: string;
  rationale: string;
}
export interface SwarmSectorScore {
  sector: string;
  score: number;
  rationale: string;
}
export interface SwarmThemeScore {
  theme: string;
  score: number;
  rationale: string;
}
export interface SwarmActionOutput {
  long_stocks: SwarmPick[];
  long_etfs: SwarmPick[];
  short_stocks: SwarmPick[];
  short_etfs: SwarmPick[];
  sector_scores: SwarmSectorScore[];
  top_themes: SwarmThemeScore[];
  bottom_themes: SwarmThemeScore[];
  _error?: string;
}

// ─── Phase 5.5 — Trading Timing (Pattern T2) ───
export type EntrySignal = "BUY_NOW" | "WAIT" | "SKIP" | string;
export type TradingUrgency = "URGENT" | "NORMAL" | "PATIENT" | string;
export interface ExitTrigger {
  condition: string;
  action: string;
  type: "TAKE_PROFIT" | "STOP_LOSS" | "OVEREXT" | "REGIME_FLIP" | string;
}
export interface PickTiming {
  entry_signal: EntrySignal;
  entry_trigger: string;
  exit_triggers: ExitTrigger[];
  urgency: TradingUrgency;
  rationale: string;
}

// ─── Buy/Sell Final List ───
export interface FinalListItem {
  ticker: string;
  name: string;
  sector: string;
  composite: number;
  horizon: "tactical" | "core" | "strategic" | string;
  bucket: string;
  stars: number;
  in_proxy_latest?: boolean;
  in_top_alpha?: boolean;
  in_worst_alpha?: boolean;
  state: string;
  signal_days: number;
  urgency: string;
  action: "EXECUTE_TODAY" | "WATCH_TOMORROW" | "ALREADY_HELD" | "OBSERVE" | "CLOSE_TODAY" | string;
  entry_trigger: string;
  rationale: string;                  // PM Agent's reason commentary
  trading_rationale?: string;         // Trading Agent's WHY commentary
  // Debate Synthesis (Option A — Phase 5.6a)
  tier?: string;                       // UNANIMOUS / MAJORITY_CLEAN / MAJORITY_DISSENT / SOLO / EXCLUDED
  debate_transcript?: string;          // 3-4 sentence Korean discussion summary
  key_factor?: string;                  // 결정 핵심 요인
  final_decision?: string;             // INCLUDE / INCLUDE_REDUCED_SIZE / EXCLUDE / WATCH
  risk_key?: string;                   // Risk Agent's key risk word
  exit_triggers?: { type: string; condition: string; action: string }[];
  is_exit?: boolean;
  // Elliott Wave-based stop loss (Phase Z addition)
  stop_price?: number | null;          // target stop-loss price (in `currency`)
  stop_pct?: number | null;            // %change vs current price (e.g. -2.9)
  stop_type?: string;                  // W4_TIGHT / W1_PRIMARY / W2_INVALID / SWING_LOW / MECHANICAL / UNAVAILABLE
  stop_rationale?: string;             // Korean explanation of why this stop
  stop_wave_guess?: string;            // WAVE_5_ACTIVE / WAVE_3_EARLY / WAVE_1_NEW / AMBIGUOUS
  stop_computed_at?: string;           // ISO timestamp
  // Currency for stop_price + current_price (NEW: handles Korean ETFs in KRW, etc.)
  currency?: string;                   // USD / KRW / JPY / EUR / ...
  currency_symbol?: string;            // $ / ₩ / ¥ / € / ...
  current_price?: number | null;       // current market price (in `currency`)
  // 3-tier Entry prices (CAN SLIM + Elliott + SMA50)
  entry_aggressive?: number | null;        // current price (comp ≥ 75)
  entry_primary?: number | null;            // CAN SLIM pivot point
  entry_conservative?: number | null;       // SMA50 pullback
  entry_aggressive_rationale?: string;
  entry_primary_rationale?: string;
  entry_conservative_rationale?: string;
  entry_primary_status?: string;            // "actionable" / "buy_zone" / "await_breakout" / "extended" / "elliott_fallback"
  entry_base_pattern?: string;              // cup_with_handle / flat_base / double_bottom
  entry_base_quality?: string;              // A / B / C
  entry_volume_confirmed?: boolean;
  entry_volume_ratio?: number;
  entry_oneil_cut_loss?: number;            // -7% from primary entry
  entry_rr_ratio?: number;                  // Risk/Reward ratio
  entry_composite_tier?: string;            // ALL / PRIMARY_AND_CONSERVATIVE / CONSERVATIVE_ONLY / SKIP
  entry_pyramid_layers?: Array<{
    layer: number; price: number; trigger_pct: number;
    position_size_pct: number; condition: string;
  }>;
  entry_skip_reason?: string;
  // Holdings-aware re-ranking (방식 A+B: 섹터 cap + 상관관계)
  ha_diversification_weight?: number;   // 0.3-1.0 (섹터 집중도 기반)
  ha_correlation_penalty?: number;      // 0.5-1.0 (보유 종목과 상관관계)
  ha_max_corr?: number | null;          // 보유 종목 중 최대 상관계수
  ha_max_corr_ticker?: string;          // 최대 상관 보유 종목
  ha_adjusted_score?: number;           // 보유-인지 최종 점수
  ha_base_score?: number;               // 보유 무시 기본 점수
  ha_rationale?: string;                // 한국어 설명
  // Unified category (merged from active_positions/exit_pending/buy_list)
  category?: "NEW" | "HOLDING" | "ENTERED" | "EXIT_PENDING" | string;
  days_held?: number;
  persistence_days?: number;
  in_today_picks?: boolean;
  // Trailing total returns (from price cache; null when history insufficient)
  ret_5d?:  number | null;
  ret_1mo?: number | null;
  ret_3mo?: number | null;
  ret_6mo?: number | null;
  ret_1y?:  number | null;
  // 3-Agent Voting (Option C)
  votes?: {
    pm:      "APPROVE" | "CAUTION" | "REJECT" | string;
    trading: "APPROVE" | "CAUTION" | "REJECT" | string;
    risk:    "APPROVE" | "CAUTION" | "REJECT" | string;
  };
  consensus?: "UNANIMOUS" | "MAJORITY_CLEAN" | "MAJORITY_DISSENT"
              | "SOLO_CLEAN" | "SOLO_DISSENT" | "ALL_CAUTION" | "REJECTED" | string;
  n_approve?: number;
  n_reject?:  number;
  n_caution?: number;
  risk_score?: number;
  risk_reason?: string;
}
export interface FinalListCommentary {
  unified_commentary?: string;   // 12,000-char unified commentary
  unified_cached?:     boolean;
  // Option B: split commentaries (3 separate LLM outputs)
  common_macro?:        string;  // ~1,500자 거시 + 과거 유사 + 시나리오 (Stock+ETF 공통)
  common_macro_cached?: boolean;
  common_macro_stale?:  boolean;
  common_macro_pending?: boolean;
  stock_split?:         string;  // ~5,500자 Stock deep-dive
  stock_split_cached?:  boolean;
  stock_split_stale?:   boolean;
  stock_split_pending?: boolean;
  etf_split?:           string;  // ~5,500자 ETF deep-dive
  etf_split_cached?:    boolean;
  etf_split_stale?:     boolean;
  etf_split_pending?:   boolean;
  // Legacy / backward compat
  buy_commentary:  string;
  sell_commentary: string;
  buy_cached?:  boolean;
  sell_cached?: boolean;
  generated_at?: string;
  pending?: boolean;
  stale?: boolean;
  error?: string;
}
export interface CategoryCommentary {
  entered?:      { commentary: string; cached?: boolean; n_items?: number };
  exit_pending?: { commentary: string; cached?: boolean; n_items?: number };
  holding?:      { commentary: string; cached?: boolean; n_items?: number };
  new?:          { commentary: string; cached?: boolean; n_items?: number };
}
export interface ActivePositionItem {
  ticker: string;
  name: string;
  sector: string;
  composite: number;
  horizon: string;
  bucket: string;
  state: string;
  entered_date?: string;
  first_seen?: string;
  days_held: number;
  persistence_days: number;
  in_today_buy_picks?: boolean;
  in_today_sell_picks?: boolean;
  current_signal?: string;
  rationale?: string;
  trading_rationale?: string;
  risk_score?: number;
  risk_vote?: string;
  risk_reason?: string;
  last_alert?: string | null;
  ret_5d?:  number | null;
  ret_1mo?: number | null;
  ret_3mo?: number | null;
  ret_6mo?: number | null;
  ret_1y?:  number | null;
}
export interface ExitPendingItem {
  ticker: string;
  name: string;
  sector: string;
  composite: number;
  horizon: string;
  state: string;
  entered_date?: string;
  days_held: number;
  persistence_days: number;
  exit_reason?: string;
  last_alert?: string | null;
  ret_5d?:  number | null;
  ret_1mo?: number | null;
  ret_3mo?: number | null;
}
export interface FinalListResponse {
  buy_list: FinalListItem[];
  sell_list: FinalListItem[];
  active_positions?: ActivePositionItem[];
  exit_pending?: ExitPendingItem[];
  commentary?: FinalListCommentary;
  category_commentary?: CategoryCommentary;
  // Turnover-cap whitelist: only these tickers are surfaced per category (top-5 each).
  capped_tickers_by_category?: {
    ENTERED?:      string[];
    EXIT_PENDING?: string[];
    HOLDING?:      string[];
    NEW?:          string[];
  };
  items_by_category?: Record<string, any[]>;
  metadata: {
    proxy_cohort_date?: string;
    n_proxy_long_stocks?: number;
    n_proxy_long_etfs?: number;
    n_top_stocks?: number;
    n_top_etfs?: number;
    n_positions_tracked?: number;
    swarm_generated_at?: string;
  };
}
export const fetchFinalList = (): Promise<FinalListResponse> =>
  api.get("/final-list").then((r) => r.data);

export const fetchValidatedExtraTimeline = (): Promise<TradingLifecyclesCompact> =>
  api.get("/validated-extra-timeline").then((r) => r.data);

// ─── Phase 5.6 — Position State (Stateful Hysteresis) ───
export type PositionStateValue =
  | "PROSPECTING" | "ENTERED" | "HOLDING"
  | "EXIT_PENDING" | "EXITED" | "DROPPED" | string;
export interface PickPositionState {
  state: PositionStateValue;
  entered_date?: string | null;
  days_in_state: number;
  consecutive_signal_days: { BUY_NOW: number; WAIT: number; SKIP: number };
  alert?: string | null;
  last_signal?: string;
}

// ─── Phase 5 — PM Agent ───
export type SwarmChangeType = "SAME" | "PROMOTED" | "DEMOTED" | "NEW";
export interface SwarmPMPick extends SwarmPick {
  change_type: SwarmChangeType | string;
  change_reason?: string;
  timing?: PickTiming;                  // Pattern T2: attached by Phase 5.5 OR per-ticker R1/R2
  position_state?: PickPositionState;   // Phase 5.6: stateful hysteresis
  // Option C — Per-Ticker Debate fields
  debate_synthesis?: {
    tier?: string | null;
    stars?: number | null;
    final_decision?: string | null;
    debate_transcript?: string;
    key_factor?: string;
    _failed?: boolean;
    _failure_reason?: string;
  };
  risk_verdict?: {
    vote?: string | null;
    key_risk?: string;
    rationale?: string;
    _failed?: boolean;
    _failure_reason?: string;
  };
  // Phase 2 portfolio composition output
  composition_decision?: "INCLUDE" | "INCLUDE_HALF" | "EXCLUDED_BY_CAP" | "EXCLUDED_BY_BUDGET" | "EXCLUDE" | "WATCH" | string;
  final_size?: number;                  // 1.0, 0.5, or 0.0
  _excluded_reason?: string;
  _failed_agents?: { trading?: boolean; risk?: boolean; debate?: boolean };
  _votes_overridden?: string;
  _votes_fallback?: string;
  // Per-ticker debate transcript (debug)
  _pt_debate?: PerTickerDebateTranscript;
}
export interface SwarmPhase4Drop {
  bucket: "long_stocks" | "long_etfs" | "short_stocks" | "short_etfs" | string;
  ticker: string;
  reason: string;
}
export interface SwarmHedgePair {
  long: string;
  short: string;
  sector?: string;
  horizon?: "tactical" | "core" | "strategic" | string;
  rationale: string;
}
export interface SwarmRiskBudget {
  sector: string;
  allocation_pct: number;
  rationale: string;
}
export interface SwarmHorizonPicks {
  long_stocks: SwarmPMPick[];
  long_etfs: SwarmPMPick[];
  short_stocks: SwarmPMPick[];
  short_etfs: SwarmPMPick[];
}
export interface SwarmHorizons {
  tactical: SwarmHorizonPicks;   // 5d  — short-term tactical
  core: SwarmHorizonPicks;       // 21d — primary horizon, includes Phase 4 diff
  strategic: SwarmHorizonPicks;  // 63d — multi-month strategic
}
export interface SwarmIterationRound {
  round: number;
  n_tickers: number;
  delta: number;          // Δ vs prev round (0-1)
  n_objections: number;
  objections?: Record<string, Array<{ ticker: string; bucket: string; composite: number; issues: string[] }>>;
  // Sequential memory (Fix 3)
  kept_tickers?: string[];
  added_tickers?: string[];
  removed_tickers?: string[];
  // Pin tracking (Fix 4)
  n_pinned?: number;
  n_newly_pinned?: number;
  n_rejected_pool?: number;
}
export interface SwarmIteration {
  history: SwarmIterationRound[];
  converged: boolean;
  converged_at_round: number;
  max_rounds: number;
  convergence_threshold: number;
}

// ── Option C: Per-Ticker Debate types ──
export interface PerTickerDebateRound1 {
  trading?: { entry_signal?: string; entry_trigger?: string; urgency?: string; rationale?: string };
  risk?:    { vote?: string; key_risk?: string; rationale?: string };
  critic?:  { assessment?: string; challenge?: string; revise_needed?: boolean };
  _failed?: boolean;
  _failure_reason?: string;
}
export interface PerTickerDebateRound2 {
  revised_rationale?: string;
  trading?: { entry_signal?: string; urgency?: string; rationale?: string };
  risk?:    { vote?: string; key_risk?: string; rationale?: string };
  drop_pick?: boolean;
}
export interface PerTickerDebateRound3 {
  tier?: string;
  stars?: number;
  final_decision?: string;
  debate_transcript?: string;
  key_factor?: string;
}
export interface PerTickerDebateTranscript {
  round1?: PerTickerDebateRound1;
  round2?: PerTickerDebateRound2;
  round3?: PerTickerDebateRound3;
}
export interface PerTickerDebateSummary {
  tier_dist: Record<string, number>;
  trading_dist: Record<string, number>;
  risk_dist: Record<string, number>;
  n_failed: number;
  n_total: number;
}
export interface PortfolioCompositionMeta {
  regime_tag?: string;
  adaptive_budget_per_horizon?: number;
  max_sector_weight?: number;
  totals?: {
    n_include?: number;
    n_include_half?: number;
    n_excluded_debate?: number;
    n_excluded_cap?: number;
    n_excluded_budget?: number;
    n_watch?: number;
  };
  sector_distribution?: Record<string, number>;
  horizon_distribution?: Record<string, number>;
  warnings?: string[];
  _error?: string;
}
export interface PortfolioCompositionSummary {
  regime_tag?: string;
  adaptive_budget?: number;
  active_picks?: number;
  excluded_total?: number;
  warnings?: string[];
  sector_top3?: [string, number][];
}

export interface SwarmPMOutput {
  pm_commentary: string;
  portfolio_thesis: string;
  horizons: SwarmHorizons;
  // Legacy flat fields (backward compat — populated only if backend didn't return horizons)
  long_stocks?: SwarmPMPick[];
  long_etfs?: SwarmPMPick[];
  short_stocks?: SwarmPMPick[];
  short_etfs?: SwarmPMPick[];
  phase4_drops: SwarmPhase4Drop[];
  hedge_pairs: SwarmHedgePair[];
  risk_budget: SwarmRiskBudget[];
  // Option 2 — Iterative Swarm (5-round convergent)
  iteration?: SwarmIteration;
  // Option C — Per-Ticker Debate
  per_ticker_debate_summary?: PerTickerDebateSummary;
  portfolio_composition?: PortfolioCompositionMeta;
  portfolio_composition_summary?: PortfolioCompositionSummary;
  // Option C Phase 5 — Pareto Front Tracker
  pareto_summary?: {
    n_unique_tickers?: number;
    n_observations?: number;
    stable_tickers?: number;
    max_rounds_seen?: number;
    avg_pareto_dims?: {
      composite?: number;
      neg_n_failed?: number;
      conviction?: number;
      risk_score?: number;
    };
    _error?: string;
  };
  _error?: string;
}
export interface SwarmResult {
  generated_at: string;
  snapshot: {
    as_of: string;
    total_tickers: number;
    regime_tag_deterministic: string;
    cd_gap: number;
    gv_gap: number;
    oer_avg: number;
  };
  phase1: {
    macro_analyst?: SwarmPhase1Verdict;
    cross_asset_analyst?: SwarmPhase1Verdict;
    sector_theme_analyst?: SwarmPhase1Verdict;
    flow_momentum_analyst?: SwarmPhase1Verdict;
    news_narrative_analyst?: SwarmPhase1Verdict;
  };
  phase1_errors: Record<string, string>;
  phase2: SwarmPhase2Coherence;
  synthesis_neutral: SwarmSynthesis;
  synthesis_averse: SwarmSynthesis;
  phase4_action?: SwarmActionOutput;
  phase5_pm?: SwarmPMOutput;
}
export interface SwarmStartResponse {
  status: "started" | "already_running" | "cached" | "no_claude_cli" | "import_error" | string;
  generated_at?: string;
  ttl_hours?: number;
  error?: string;
}
export interface SwarmStatus {
  running: boolean;
  started_at: string;
  finished_at: string;
  phase: string;
  current: string;
  events: { t: string; phase: string; agent: string; status: string }[];
  last_error: string;
}
export interface SwarmResultResponse {
  available: boolean;
  fresh: boolean;
  result?: SwarmResult;
  error?: string;
}
export const startMarketLeadersSwarm = (force = false): Promise<SwarmStartResponse> =>
  api.post("/market-leaders/swarm", null, { params: { force } }).then((r) => r.data);
export const fetchSwarmStatus = (): Promise<SwarmStatus> =>
  api.get("/market-leaders/swarm/status").then((r) => r.data);
export const fetchSwarmResult = (): Promise<SwarmResultResponse> =>
  api.get("/market-leaders/swarm/result").then((r) => r.data);

// ─── PM Agent Backtest (deterministic proxy) ───
export interface BacktestForwardDist {
  n: number;
  hit_rate?: number;
  mean?: number;
  median?: number;
  std?: number;
  p10?: number; p25?: number; p75?: number; p90?: number;
  min?: number; max?: number;
}
export interface BacktestAlphaStats {
  n: number;
  mean_alpha?: number;
  median_alpha?: number;
  win_rate?: number;
  avg_win?: number;
  avg_loss?: number;
  win_loss_ratio?: number;
  profit_factor?: number;
  t_stat?: number;
}
export interface BacktestQuintile {
  n: number;
  rank_range?: string;
  hit_rate?: number;
  mean_ret?: number;
  mean_alpha?: number;
}
export interface BacktestRankAnalysis {
  Q1?: BacktestQuintile; Q2?: BacktestQuintile; Q3?: BacktestQuintile;
  Q4?: BacktestQuintile; Q5?: BacktestQuintile;
  _ic?: number;
  _monotonicity?: number;
}
export interface BacktestBucketMetric {
  n_total: number;
  forward_return_dist: BacktestForwardDist;
  alpha_stats: BacktestAlphaStats;
  rank_quintile: BacktestRankAnalysis;
  by_sector: Record<string, BacktestAlphaStats>;
  by_classification: Record<string, BacktestAlphaStats>;
}
export interface TradingMetricsBucket {
  n_picks: number;
  entry_signal_edge: Record<string, {
    n: number; mean_return?: number; mean_alpha?: number;
    hit_rate?: number; days_to_trigger_mean?: number;
  }>;
  exit_trigger_effectiveness: Record<string, {
    n: number; fire_pct: number; mean_return: number;
    win_rate: number; delta_vs_hold_mean?: number;
    avg_days_held?: number;
  }>;
  trade_lifecycle: {
    n: number;
    managed_mean_return?: number;
    buyhold_mean_return?: number;
    delta_alpha?: number;
    managed_win_rate?: number;
    buyhold_win_rate?: number;
    managed_sharpe?: number;
    buyhold_sharpe?: number;
    managed_max_dd?: number;
    buyhold_max_dd?: number;
    trading_helped_pct?: number;
  };
  urgency_calibration: Record<string, {
    n: number;
    pct_moved_3pct_in_3d?: number;
    avg_days_to_trigger?: number;
  }>;
}
export interface LifecycleRecord {
  t: string;                          // ticker
  n?: string;                         // ticker display name
  d: string;                          // entry/cohort date (YYYY-MM-DD)
  sig: string;                        // entry signal
  mst?: string | null;                // managed state (EXITED / NEVER_ENTERED / etc)
  mex?: string | null;                // managed exit type
  mr?: number | null;                 // managed realized return
  mdh?: number | null;                // managed days held
  mdt?: number | null;                // managed days to trigger
  bh?: number | null;                 // buy-and-hold return @ full horizon (null if in-flight)
  bhp?: number | null;                // buy-and-hold partial return @ latest price (MTM)
  bhd?: number | null;                // trading days elapsed since entry
}
export interface TradingLifecyclesCompact {
  tactical:  Record<string, LifecycleRecord[]>;
  core:      Record<string, LifecycleRecord[]>;
  strategic: Record<string, LifecycleRecord[]>;
}
export interface BacktestResult {
  as_of_run: string;
  year: number;
  end_date: string;
  n_picks: number;
  n_evaluations: number;
  weekly_summary: { week: number; date: string; picks: Record<string, number> }[];
  horizon_metrics: {
    h5d: Record<string, BacktestBucketMetric>;
    h21d: Record<string, BacktestBucketMetric>;
    h63d: Record<string, BacktestBucketMetric>;
  };
  trading_metrics?: {
    tactical:  TradingMetricsBucket;
    core:      TradingMetricsBucket;
    strategic: TradingMetricsBucket;
  };
  trading_lifecycles_compact?: TradingLifecyclesCompact;
}
export interface BacktestResponse {
  available: boolean;
  result?: BacktestResult;
  error?: string;
}
export const fetchBacktestResults = (): Promise<BacktestResponse> =>
  api.get("/backtest/results").then((r) => r.data);
export const startBacktestRun = (): Promise<{status: string}> =>
  api.post("/backtest/run").then((r) => r.data);
export const fetchBacktestStatus = (): Promise<{running: boolean; started_at?: string; finished_at?: string; last_error?: string}> =>
  api.get("/backtest/status").then((r) => r.data);

// ─── Backtest ticker drilldown ───
export interface TickerAppearance {
  entry_date: string;
  bucket: string;
  side: string;
  rank: number | null;
  proxy_score: number | null;
  classification: string | null;
  ret_5d: number | null; ret_21d: number | null; ret_63d: number | null;
  bench_5d: number | null; bench_21d: number | null; bench_63d: number | null;
  alpha_5d: number | null; alpha_21d: number | null; alpha_63d: number | null;
  hit_5d: number | null; hit_21d: number | null; hit_63d: number | null;
}
export interface TickerDetail {
  ticker: string;
  name: string;
  sector: string;
  asset_type: string;
  n_appearances: number;
  avg_rank: number | null;
  buckets: string[];
  sides: string[];
  mean_alpha_5d?: number | null;
  mean_alpha_21d?: number | null;
  mean_alpha_63d?: number | null;
  win_rate_5d?: number | null;
  win_rate_21d?: number | null;
  win_rate_63d?: number | null;
  mean_ret_5d?: number | null;
  mean_ret_21d?: number | null;
  mean_ret_63d?: number | null;
  appearances: TickerAppearance[];
}
export interface TickerDrilldownResponse {
  available: boolean;
  ticker?: string;
  data?: TickerDetail;
  as_of_run?: string;
  error?: string;
}
export const fetchTickerDrilldown = (ticker: string): Promise<TickerDrilldownResponse> =>
  api.get(`/backtest/ticker/${encodeURIComponent(ticker)}`).then((r) => r.data);

export interface BucketRanking {
  ticker: string;
  name: string;
  sector: string;
  n: number;
  mean_alpha_21d: number;
  win_rate_21d: number;
  avg_rank: number;
}
export interface BacktestRankings {
  long_stocks: { top: BucketRanking[]; worst: BucketRanking[] };
  long_etfs: { top: BucketRanking[]; worst: BucketRanking[] };
  short_stocks: { top: BucketRanking[]; worst: BucketRanking[] };
  short_etfs: { top: BucketRanking[]; worst: BucketRanking[] };
}
export const fetchBacktestRankings = (): Promise<{available: boolean; rankings?: BacktestRankings}> =>
  api.get("/backtest/rankings").then((r) => r.data);

// ─── PM History (forward collection) ───
export interface PMHistorySummary {
  n_snapshots: number;
  first_date?: string | null;
  last_date?: string | null;
  unique_tickers: number;
  top_persistent_tickers: { ticker: string; count: number; pct: number }[];
  error?: string;
}
export const fetchPMHistorySummary = (): Promise<PMHistorySummary> =>
  api.get("/pm-history/summary").then((r) => r.data);
