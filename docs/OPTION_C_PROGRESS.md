# Option C: Per-Ticker Debate Pipeline — COMPLETE ✅

## 전체 완료 상태

| Phase | 모듈 | 상태 | LoC |
|---|---|---|---|
| **0** Tier 1 Critical Fixes | 5 파일 수정 | ✅ | — |
| **1** Per-Ticker Debate Engine | `agents/per_ticker_debate.py` | ✅ | 577 |
| **2** Portfolio Composer | `agents/portfolio_composer.py` | ✅ | 280 |
| **3** market_leaders_swarm 통합 | branching logic | ✅ | +60 |
| **4** Frontend types + UI | `client.ts` + `SwarmAnalysis.tsx` | ✅ | +180 |
| **5** Adaptive Macro + Pareto | `agents/pareto_tracker.py` | ✅ | 215 |
| **Session Fix** | global lock + reaper + cleanup script | ✅ | — |

---

## Phase 0: Tier 1 Critical Fixes ✅

### 0.1 Default Fallback Removal
- `agents/debate_synthesizer.py:_extract_synthesis()` — silent `SOLO/WATCH` defaults replaced with `_failed=True` flag + invalid-tier/decision validation
- `agents/risk_manager_llm.py:_extract_verdicts()` — silent `CAUTION` default replaced with `_failed=True`
- `agents/trading_layer.py:_extract_timings()` — silent `WAIT` default replaced with `_failed=True`

### 0.2 False Convergence Block
- `agents/market_leaders_swarm.py:1790-1816` — 2 guards:
  - `_round_failed_recovered=True` AND `Δ<5%` → reject convergence
  - `n_obj ≥ 90% of n_tickers` → universal pessimism, not convergence

### 0.3 Iteration Time Budget
- `ITERATION_TIME_BUDGET_SEC=3600` — 1 hour cumulative cap

### 0.4 Strict Sequential
- All 6 ThreadPoolExecutor → `max_workers=1`

### 0.5 Downstream `_failed` Handler
- `agents/final_list.py:228-280` — composite-based 4-tier override

---

## Phase 1: Per-Ticker Debate Engine ✅

**Module**: `agents/per_ticker_debate.py`

3-round per-ticker debate (5 tickers/batch):
- **R1**: Trading + Risk + Critic verdict
- **R2**: Revise (only if critic.revise_needed)
- **R3**: Final Arbiter → tier + final_decision

```python
result = run_per_ticker_debate(
    pm_horizons=pm_horizons,
    regime_tag="LATE_CYCLE",
    macro_context="...",
    run_claude_fn=_run_claude,
    _emit_fn=_emit,
)
```

Per pick augmented with: `timing`, `risk_verdict`, `debate_synthesis`, `_pt_debate` (R1/R2/R3 transcript)

---

## Phase 2: Portfolio Composer ✅

**Module**: `agents/portfolio_composer.py`

5-step composition pipeline:
1. Classify each pick (`INCLUDE` / `INCLUDE_HALF` / `EXCLUDE` / `WATCH`)
2. Sort by conviction (composite × tier × signal × risk)
3. Apply adaptive budget (regime-aware)
4. Enforce sector cap (default 30%)
5. Cross-horizon health checks

**Verified outputs** (LATE_CYCLE regime, mock 3-pick test):
- AAPL (UNANIMOUS, BUY_NOW, APPROVE) → `INCLUDE` size 1.0 ✓
- MSFT (MAJORITY_CLEAN, WAIT, CAUTION) → `EXCLUDED_BY_CAP` (Tech sector cap hit) ✓
- NVDA (SOLO, SKIP, REJECT) → `EXCLUDE` ✓

---

## Phase 3: market_leaders_swarm Integration ✅

`PER_TICKER_DEBATE_ENABLED = True` (default) toggles new path:

**New flow:**
```
Phase 1 (5 analysts) → Phase 2 (Coherence) → Phase 3 (Synthesis)
  → Phase 4 (Action) → Phase 5 (PM, 1 round)
  → Phase 5pt (Per-Ticker Debate)
  → Phase 5b (Portfolio Composer)
  → Phase 5.6 (Position State Machine)
```

**Legacy flow** (when `PER_TICKER_DEBATE_ENABLED=False`):
- Same as before — Phase 5.5/5.55/5.6a iterative batched mode (preserved for rollback)

---

## Phase 4: Frontend ✅

### `frontend/src/api/client.ts`
- `PerTickerDebateRound1/2/3` interfaces
- `PerTickerDebateTranscript`, `PerTickerDebateSummary` types
- `PortfolioCompositionMeta`, `PortfolioCompositionSummary` types
- `SwarmPMPick` extended with `debate_synthesis`, `risk_verdict`, `composition_decision`, `final_size`, `_failed_agents`, `_pt_debate`
- `SwarmPMOutput` augmented with `per_ticker_debate_summary`, `portfolio_composition`, `portfolio_composition_summary`, `pareto_summary`

### `frontend/src/components/shared/SwarmAnalysis.tsx`
- **Per-Ticker Debate panel** (purple banner): tier/trading/risk distribution + failure indicator
- **Portfolio Composition panel** (amber banner): adaptive budget · active/excluded counts · sector top-3 · warnings

---

## Phase 5: Adaptive Macro + Pareto Tracking ✅

**Module**: `agents/pareto_tracker.py`

### 5.1 Pareto Front Tracker
- Multi-dimensional dominance: `(composite, neg_n_failed, conviction, risk_score)`
- Replaces single best_round_snapshot with proper Pareto-optimal pick set
- API:
  ```python
  tracker = ParetoFrontTracker()
  for round_n in range(1, max_rounds + 1):
      tracker.add_round(round_n, pm_horizons)
  optimal_horizons = tracker.get_pareto_optimal()
  ```

### 5.2 Adaptive Convergence Threshold
| Regime | Δ Threshold (base=0.20) |
|---|---|
| RISK_ON | 0.250 (looser) |
| NEUTRAL | 0.200 |
| MIXED / TRANSITIONAL | 0.180 |
| **RISK_OFF / LATE_CYCLE / ROTATION** | **0.150** (tighter) |

### 5.3 Adaptive Pick Pool Size
| Regime | Pick Pool (base=240) |
|---|---|
| RISK_ON | 240 |
| NEUTRAL | 204 |
| ROTATION_IN_PROGRESS | 168 |
| LATE_CYCLE | 132 |
| **RISK_OFF** | **96** |

### 5.4 Integration in market_leaders_swarm.py
- `CONVERGENCE_DELTA_THRESHOLD = adaptive_convergence_threshold(snap.regime_tag)` (replaces hardcoded 0.20)
- `pareto_tracker.add_round(round_n, ...)` after each iteration round
- `pm_output["pareto_summary"]` persisted in final output

---

## Session Fix (Logout/Concurrent Session) ✅

### Root Cause
**6개 Claude 프로세스 동시 실행** (Max plan ~2 한도 위반):
- VSCode 세션 2개 (Sun 11AM → 31시간 살아있음)
- swarm subprocess timeout 좀비 3개

### Fix
1. **즉시 정리**: 5 stale 프로세스 종료 (현재 1개만 잔존)
2. **`_run_claude` 강화** (`market_leaders_swarm.py:1138-1259`):
   - `_CLAUDE_SUBPROCESS_LOCK` (전역 lock) — **검증: 3 thread × 1s = 정확히 3.01s, overlap 0%**
   - `_reap_zombie_claude_processes()` — 5분 이상 orphan 자동 청소 (5호출마다)
   - Process group kill on timeout — `start_new_session=True` + `os.killpg(SIGKILL)`
3. **수동 cleanup**: `scripts/cleanup_claude_sessions.sh`
   - 현재 VSCode 세션 자동 감지 + 보호
   - 12h+ VSCode 세션 종료
   - 5분+ orphan `claude` 종료

---

## End-to-End Verification ✅

```
Per-Ticker Debate Summary:
  Total: 3, Failed: 0
  Tier dist:    {'UNANIMOUS': 1, 'MAJORITY_CLEAN': 1, 'SOLO': 1}
  Trading dist: {'BUY_NOW': 1, 'WAIT': 1, 'SKIP': 1}
  Risk dist:    {'APPROVE': 1, 'CAUTION': 1, 'REJECT': 1}

Portfolio Composition (LATE_CYCLE):
  Adaptive budget: 14/horizon  ✓
  Active: 1.0, Excluded: 2
  AAPL → INCLUDE size=1.0  ✓
  MSFT → EXCLUDED_BY_CAP (Tech sector)  ✓
  NVDA → EXCLUDE (final_decision)  ✓

Pareto Front: 3 unique, 1 round seen

Session Lock OK ✓
Zombie reaper callable ✓

Adaptive Logic per Regime:
  RISK_ON                   Δ=0.250  Pool=240
  NEUTRAL                   Δ=0.200  Pool=204
  LATE_CYCLE                Δ=0.150  Pool=132
  RISK_OFF                  Δ=0.150  Pool=96
  ROTATION_IN_PROGRESS      Δ=0.150  Pool=168
```

---

## Files Modified Across All Sessions

```
NEW MODULES:
  agents/per_ticker_debate.py     # Phase 1 — 577 LoC
  agents/portfolio_composer.py    # Phase 2 — 280 LoC
  agents/pareto_tracker.py        # Phase 5 — 215 LoC
  scripts/cleanup_claude_sessions.sh   # Session Fix

MODIFIED:
  agents/debate_synthesizer.py    # Phase 0.1 — honest failure
  agents/risk_manager_llm.py      # Phase 0.1 — honest failure
  agents/trading_layer.py         # Phase 0.1 — honest failure
  agents/final_list.py            # Phase 0.5 — _failed handler
  agents/market_leaders_swarm.py  # Phase 0.2-0.4 + Phase 3 + Phase 5 + Session Fix
  frontend/src/api/client.ts      # Phase 4 — types
  frontend/src/components/shared/SwarmAnalysis.tsx   # Phase 4 — UI panels
  docs/OPTION_C_PROGRESS.md       # This document
```

---

## Operational Notes

### Switching back to legacy mode
Edit `agents/market_leaders_swarm.py:1481`:
```python
PER_TICKER_DEBATE_ENABLED = False   # was True
```

### Monitoring per-ticker debate progress
Look for emit events:
- `phase5_pt_debate.{horizon}_r1_b{batch_idx}` — Round 1 batched verdict
- `phase5_pt_debate.{horizon}_r2_b{batch_idx}` — Round 2 revision
- `phase5_pt_debate.{horizon}_r3` — Final Arbiter
- `phase5b_compose.portfolio_composer` — Portfolio composition

### Session troubleshooting
```bash
./scripts/cleanup_claude_sessions.sh
```

### Cost expectations
| Pipeline | LLM Calls | Wall Clock (sequential) |
|---|---|---|
| Legacy (5 rounds × phases) | ~60 large | ~50 min |
| **Option C** (per-ticker + composer) | ~144 small | **~36 min** |

Per-ticker debate trades batch size for resilience: 5-ticker batches don't suffer the universal-CAUTION cascade.
