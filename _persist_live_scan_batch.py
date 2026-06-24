# -*- coding: utf-8 -*-
"""Persist 20 normalized verdicts from live-scan dispatch.

Each entry hand-extracted from sub-agent JSON output (schemas varied widely;
we collapsed to the canonical shape used by .multi_agent_debate_cache.json).
"""
import json, time
from pathlib import Path

CACHE = Path(".multi_agent_debate_cache.json")


def make(ticker, asset, side, group, comp, cls_str,
         f_rating, f_conf, s_rating, s_conf, v_rating, v_conf,
         n_rating, n_mod, n_size, n_reason,
         a_rating, a_mod, a_size, a_reason,
         dis_type, axis=1, disp=0.15, gen="2026-05-31T00:00:00"):
    return {
        "ticker": ticker, "tier": "A",
        "asset_type": asset, "side": side, "group": group,
        "rounds": [{
            "round_num": 1,
            "fundamental": {"persona":"fundamental","rating":f_rating,"confidence":f_conf,"key_points":[],"biggest_risk":"","biggest_opportunity":"","raw_text":"[archived from in-session R1]","narrative_summary":"","critique":""},
            "sentiment":   {"persona":"sentiment","rating":s_rating,"confidence":s_conf,"key_points":[],"biggest_risk":"","biggest_opportunity":"","raw_text":"[archived from in-session R1]","narrative_summary":"","critique":""},
            "valuation":   {"persona":"valuation","rating":v_rating,"confidence":v_conf,"key_points":[],"biggest_risk":"","biggest_opportunity":"","raw_text":"[archived from in-session R1]","narrative_summary":"","critique":""},
            "notes": ""
        }],
        "synthesis_neutral": {"risk_mode":"neutral","rating":n_rating,"position_modifier":n_mod,"sizing_recommendation":n_size,"reasoning":n_reason,"raw_text":"[rule-based neutral synthesis]"},
        "synthesis_averse":  {"risk_mode":"averse","rating":a_rating,"position_modifier":a_mod,"sizing_recommendation":a_size,"reasoning":a_reason,"raw_text":"[rule-based averse synthesis]"},
        "converged_round": 1,
        "disagreement": {"rating_axis":axis,"specialist_dispersion":disp,"type":dis_type},
        "composite_at_time": comp,
        "classification_at_time": cls_str,
        "generated_at": gen,
    }


VERDICTS = [
    make("BUD","stock","long","momentum",75,"🟢 CONTINUATION",
         "BUY",0.72,"BUY",0.65,"BUY",0.60,
         "BUY",3,"Add to full Momentum sleeve weight; harvest +3 modifier given OER 12 headroom.",
         "BUD prints a clean Momentum LONG profile: Composite 75 with CONTINUATION classification, perfect SMA alignment (TCS 99/100), and OER only 12 — exceptional headroom for an entrenched trend. All three lanes converge BUY: fundamentals (EM premiumization + deleveraging), sentiment (boycott overhang fading without euphoria), and valuation (~10x EV/EBITDA discount to global staples). TFS 27/35 indicates trend is maturing rather than exhausting, leaving room for continuation over the weekly hold horizon. Peer Mom LONGs (SAN 85, MRK 78, BNS 77) score higher but BUD's OER asymmetry justifies an above-baseline modifier.",
         "BUY",1,"Initiate or hold at base Momentum weight with a small +1 tilt; avoid stacking with other staples names.",
         "Even under risk-averse framing BUD warrants a long: low OER (12) + CONTINUATION + cross-lane consensus is a high-quality momentum setup. However TFS still only moderate (27/35) and RSS mid-60s means it is not the strongest horse in today's bucket. GLP-1 alcohol-demand tail risk and EM FX whipsaw justify trimming the modifier to +1.",
         "CONSENSUS_BUY", 0, 0.08),

    make("TOELY","stock","long","momentum",73,"🟢 CONTINUATION",
         "BUY",0.78,"BUY",0.72,"HOLD",0.58,
         "BUY",3,"Size at +3 modifier in core momentum sleeve; weekly review on parent FY26 guidance.",
         "TOELY is a Tier-A momentum long with structural EUV-track monopoly into N2/GAA transition and HBM layer-count tailwind. Composite 73 + CONTINUATION + RSS_long 82 confirms leadership; OER 35 leaves room before overheating. TFS_long=0 reflects late-cycle slope-flattening within an intact trend — same diagnosis as KLAC — not a topping signal. Fundamental momentum (FY26 EPS revisions +8%, JPY tailwind, Rapidus/HBM4 capex) supports continued upside.",
         "HOLD",1,"Size at 60-70% of neutral target. Late-cycle extension signature (TCS_long=100/TFS_long=0) means upside requires fresh catalyst.",
         "Risk-averse stance acknowledges TOELY momentum strength but flags TFS_long=0 + valuation premium + WFE cycle peak debate. Late-cycle extension signature means upside requires fresh catalyst rather than pure trend continuation. Valuation (FY26 P/E 26x, PEG 1.3x) prices in no-recession WFE scenario.",
         "ENTRY_TIMING", 1, 0.18),

    make("CP","stock","long","momentum",74,"🟢 CONTINUATION",
         "BUY",0.72,"BUY",0.78,"HOLD",0.60,
         "BUY",2,"Standard long allocation with modest upsize given clean technical setup.",
         "Two of three specialists land at BUY with high conviction, and the dissenting valuation view is HOLD rather than SELL — disagreement is about magnitude of upside, not direction. The Composite-74 / OER-27 / TCS-100 profile is the cleanest momentum signature available: trend intact, not overheated, with accelerating relative strength. KCS franchise quality provides a fundamental anchor that justifies riding the momentum without fear of vertical reversal. Premium valuation caps the position modifier at +2.",
         "BUY",0,"Standard size with tight trailing stop; trim on any TCS_long break below 70.",
         "Risk-averse lens still arrives at BUY because the technical setup is genuinely high-quality and the franchise is non-replicable, but sizes neutrally. Valuation premium and TFS_long at 20 (mature trend) mean the asymmetric upside is muted versus earlier-cycle names. USMCA review 2026 is a binary overhang that can re-rate the sector quickly.",
         "ENTRY_TIMING", 1, 0.18),

    make("CMC","stock","long","momentum",71,"🟢 CONTINUATION",
         "BUY",0.72,"BUY",0.68,"HOLD",0.58,
         "BUY",2,"Standard Momentum LONG with +2 modifier; trim if RSS_short breaks below 70 or OER pushes above 55.",
         "Two of three agents BUY with the fundamental and sentiment lenses both supportive — IIJA/tariff fundamental tailwind aligns with textbook CONTINUATION tape (TCS 100/100, OER 33, RSS_short 81). Valuation HOLD is a yellow flag but not a stop sign: CMC is fairly valued, not stretched, and capital return provides a ~5% floor. The weak TFS residual is the main technical caveat — momentum is mature, not fresh — so position sizing should be measured rather than aggressive.",
         "HOLD",1,"Lean long without conviction commitment; cut on RSS_short < 65 or any classification downgrade.",
         "Risk-averse lens weights the valuation HOLD and the weak TFS residual (22/35) more heavily. CMC has re-rated meaningfully off 2022-23 lows, sits at peer-median multiples on potentially peak-ish EBITDA, and the technical setup — while bullish — is showing the signature of a mature trend rather than a fresh thrust. Steel is cyclical; rebar spread mean-reversion is a real downside path.",
         "ENTRY_TIMING", 1, 0.18),

    make("HSBC","stock","long","momentum",74,"🟢 CONTINUATION",
         "BUY",0.74,"BUY",0.70,"BUY",0.68,
         "BUY",3,"Core-momentum 3.5% sized — full Pre-Mom + Momentum allocation given clean tape + capital return floor.",
         "HSBC is a high-quality momentum LONG with the rare combination of (i) Composite 74 / CONTINUATION classification (ii) cool OER 35 — no exhaustion (iii) ~10-11% combined shareholder yield providing a hard floor (iv) Asia wealth structural growth driver decoupling from NIM. Three-specialist consensus 68-72 with no material disagreement. TFS_long=0 anomaly is benign — reflects low-vol compounder regime, not deteriorating trend formation.",
         "BUY",1,"2.0% size vs 3.5% neutral; scale in thirds — 1/3 now, 1/3 on 2% pullback, 1/3 on Composite >75 with OER <45.",
         "Risk-averse synthesis: same constructive read but acknowledges (1) banks are macro-correlated, (2) TCS pinned at 100 + TFS_long=0 means trend is mature not forming, (3) re-rating cap at ~1.3x TBV limits upside asymmetry. Size at 2.0% to capture carry/momentum combo while preserving capital.",
         "CONSENSUS_BUY", 0, 0.05),

    make("XME","etf","long","momentum",74,"🟢 CONTINUATION",
         "BUY",0.72,"BUY",0.70,"BUY",0.76,
         "BUY",4,"Initiate full 4.5% at current levels; add 0.5% on any pullback to 20-day SMA.",
         "XME is a high-conviction Momentum LONG. The dual-timeframe TFS positivity (46/50) is the rarest and most predictive signal in this snapshot — it indicates the metals & mining basket is in active formation across both short and long horizons simultaneously. Composite 74 with OER only 20 means the technical setup has runway before mean-reversion math activates. Equal-weight construction maximizes operating leverage capture across NUE/STLD/CLF/X/FCX. Cross-signal corroboration from CMC and FCX (both in Pre-Mom LONG cohort) reinforces bottom-up read.",
         "BUY",2,"Initiate 2.0% immediately, add 1.0% on confirmation of TFS_short holding above 40 at next weekly close.",
         "Risk-averse synthesis still arrives at BUY because quant evidence too strong to fade. However, TCS at 90/85 implies trend is mature, sentiment shifted to consensus-long making XME first-exit vehicle on negative tariff headline, and macro tailwind stack is policy-dependent. Equal-weight construction also means CLF and X carry same weight as NUE/STLD.",
         "CONSENSUS_BUY", 0, 0.06),

    make("QUAL","etf","long","momentum",72,"🟢 CONTINUATION",
         "BUY",0.68,"BUY",0.64,"HOLD",0.58,
         "BUY",2,"Maintain a slight overweight as part of balanced Momentum LONG basket; do not add at OER 39.",
         "QUAL is the cleanest expression of late-cycle defensive-quality leadership with a still-intact trend (TCS 100/100), positive AUM creation flow, and elite-ROE mega-caps whose earnings revisions remain net positive. CONTINUATION + OER below penalty trigger means trade is still mechanically clean. However, OER 39 sits one tick below 40 penalty floor and valuation percentile is 94th — bull case wins on tape + flow + classification, but valuation percentile caps conviction. Position +2 reflects core-overweight, NOT max-conviction add.",
         "HOLD",0,"Standard baseline weight; no fresh adds at OER 39. Trim only if Composite breaks below 60 or RSS_short rolls below 50.",
         "Risk-averse view weights valuation percentile (94th) and OER proximity to penalty (39 vs 40) more heavily. Bull case is real but late-stage on valuation; if AAPL/NVDA hit single-session drawdown >5% the basket suffers concentrated. Baseline weight, no chase.",
         "ENTRY_TIMING", 1, 0.15),

    make("COPX","etf","long","momentum",68,"🟢 CONTINUATION",
         "BUY",0.72,"BUY",0.65,"HOLD",0.55,
         "BUY",2,"Half-size starter (+2 modifier) with stop below recent consolidation low.",
         "COPX is a textbook re-formation setup inside a structural copper supercycle. Composite 68 with CONTINUATION, OER only 12, URS 10 (no crowded behavioral footprint) is a rare 'stealth strength' configuration. TFS_long 50 > TFS_short 21 confirms long-horizon trend intact while short-term formation re-ignites after pause. Top holdings (FCX in Pre-Mom LONG today, BHP, Southern Copper) provide diversified exposure to copper deficit (AI power, electrification, ore grade decline). FCX overlap with Pre-Mom LONG provides bottom-up confirmation.",
         "HOLD",1,"Quarter-size tactical position; treat as momentum trade not thematic conviction hold.",
         "Single-commodity ETF concentration + TFS_short 21 weakness + country risk (Chile/Peru/Panama/Zambia) + 2-3x copper beta make this a tactical exposure, not core. Hold ≤ weekly horizon does not match multi-year supercycle thesis. Size smaller, defined stop, treat as momentum.",
         "ENTRY_TIMING", 1, 0.18),

    make("MGK","etf","long","momentum",67,"🟢 CONTINUATION",
         "HOLD",0.55,"BUY",0.60,"HOLD",0.50,
         "HOLD",-1,"Trim to 60-70% of standard ETF allocation; do not add fresh size at OER 40 with TFS_long 0.",
         "MGK is structurally bullish but late-cycle. Technical tape (Composite 67, CONTINUATION, RSS 68/71) earns bullish bias, but OER 40 floor, TFS_long 0, URS 28 cluster collectively say 'crowded mature trend, not a fresh setup.' Concentration in 8 names = effectively Mag7 + AVGO levered bet. Fundamental and macro agents agree this is fine wrapper but poor entry — same edge as MTUM with Mag7 gap risk on top.",
         "HOLD",-2,"Half-weight or rotate to alternative ETF (QQQ unhedged Mag7 exposure better priced).",
         "Risk-averse view: OER 40 at penalty floor + TFS_long 0 + URS 28 = three weak signals dominate pristine TCS. Single-name (AAPL/NVDA/MSFT) gap risk concentrated in basket. Rotate to alternative expressions.",
         "CONSENSUS_HOLD", 1, 0.10),

    make("VIG","etf","long","momentum",66,"🟢 CONTINUATION",
         "BUY",0.70,"BUY",0.80,"BUY",0.70,
         "BUY",2,"Hold full target weight or add 1pt to baseline; ideal as defensive-quality momentum anchor.",
         "Triple-bullish alignment with technical Composite 66 + CONTINUATION + OER only 27 providing clean trend exposure, fundamental quality-dividend-growth factor offering structural premium, and anomalously high URS 69 confirming institutional behavioral accumulation. Only yellow flag is weak TFS (33/20) indicating mature rather than fresh trend — argues for holding/adding rather than aggressive new entry. URS 69 + dividend factor passive bid + late-cycle defensive rotation = stacked support.",
         "BUY",1,"Standard ETF weight + small +1 tilt; trim if URS falls below 50 or AAPL/MSFT show top-holding drawdown.",
         "Risk-averse still BUY given URS 69 anomaly + triple consensus + OER cool. However mature TFS (33/20) + top-holding concentration in mega-cap tech limits asymmetric upside. Size at +1 with tighter stops.",
         "CONSENSUS_BUY", 0, 0.08),

    make("FCX","stock","long","pre_momentum",66,"🔵 RECOVERY",
         "BUY",0.78,"BUY",0.72,"BUY",0.65,
         "BUY",2,"+2 modifier; size moderately given COPX parent ETF also in Mom LONG (avoid double-exposure).",
         "FCX는 구조적 구리 supercycle deficit + Grasberg/Cerro Verde 운영 레버리지 + RECOVERY classification에서 pre-momentum→momentum transition 진입한 다중 confirmation 종목. TCS 단·장 모두 강세, OER 20 저점으로 추가 상승 runway, COPX parent ETF Mom LONG으로 sector confirmation. TFS_long 30 미흡은 단점이나 가격 발견 초기 단계 특성으로 weekly~bi-weekly 단위 추세 강화 기대. COPX ETF 동시 보유 시 single-name double-exposure 위험으로 +3 풀 modifier 대신 +2로 제한.",
         "HOLD",-1,"COPX 동시 보유 포트폴리오라면 FCX 노출 일부 축소; 독립 포지션이면 hold 정도.",
         "Sell 측 케이스는 약함. Bear agent들도 STRONG_SELL이 아닌 HOLD 수준 — Composite 66 + RECOVERY + OER 20 조합은 명백한 매도 신호가 아님. 다만 TFS_long 30 미완성 + URS 14 underreaction edge 부재 + RSS momentum gap 없음은 분할진입 시사. COPX/FCX 동반 노출 시 축소 권고.",
         "CONSENSUS_BUY", 0, 0.08),

    make("CL","stock","long","pre_momentum",58,"🔵 RECOVERY",
         "BUY",0.70,"BUY",0.65,"BUY",0.55,
         "BUY",2,"+2 modifier — Pre-Mom RECOVERY with clean technical re-engagement + Aristocrat quality floor. Pair with higher-torque cohort names.",
         "CL is a legitimate Pre-Mom RECOVERY setup with clean technical re-engagement (TCS 100/92, OER 24) and quality fundamental base (Hill's, LATAM volume recovery, Dividend Aristocrat), but scores B+ rather than A on cohort because TFS_short 8 and URS 24 indicate near-term formation and behavioral edge are both muted. Trade works as defensive ballast inside Pre-Mom LONG basket — pair-friendly with higher-torque names like FCX/GE/ARGX. Weekly to bi-weekly hold appropriate given slower-developing pattern.",
         "HOLD",0,"Pass or minimal exposure; CL is lowest-torque name in cohort, opportunity cost vs FCX/GE/ARGX.",
         "Risk-averse passes: TFS_short 8 reveals stalled near-term setup, URS 24 = no behavioral edge, RSS_long 39 = structural laggard, Composite 58 is marginal pass not high-conviction. Cohort opportunity cost is brutal — FCX/GE/ARGX offer 3-5x torque on weekly hold.",
         "CONSENSUS_BUY", 0, 0.10),

    make("GE","stock","long","pre_momentum",60,"🔵 RECOVERY",
         "BUY",0.78,"BUY",0.72,"HOLD",0.60,
         "BUY",2,"+2 modifier (not +3) to respect TechBear/FundBear caveats on weak long-horizon stack and premium valuation.",
         "Four bullish agents (Fund/Tech/Macro/Catalyst), two cautious-hold (FundBear on valuation, TechBear on weak long-horizon signals). Core agreement: short-term setup + macro + catalyst stack support weekly~bi-weekly long. RECOVERY-with-cool-OER + leadership-RSS profile historically converts to CONTINUATION over weekly~bi-weekly horizons. Modifier capped at +2 (not +3) to respect weak long-horizon stack (TCS_long 46, TFS_long 30, URS 31) and premium valuation. Re-evaluate on TCS_long crossing 55 or RSS_long crossing 50.",
         "HOLD",0,"Wait for TFS_long lift before adding; tactical bounce risk on weak structural base.",
         "Risk-averse weights weak long-horizon stack (TCS_long 46 / TFS_long 30 / URS 31 / RSS_long 39 all sub-50). This is short-term RECOVERY bounce inside structurally unproven long trend — gives one strong week then chops. Without TFS_long lifting >50 the trade is tactical pop, not Pre-Mom compounder.",
         "ENTRY_TIMING", 1, 0.18),

    make("DB","stock","long","pre_momentum",65,"🔵 RECOVERY",
         "BUY",0.78,"BUY",0.62,"HOLD",0.55,
         "BUY",2,"+2 modifier respecting +3 ceiling while leaving headroom if TCS_long confirms.",
         "Bull case dominates on Pre-Momentum-specific signals (URS 74 behavioral accumulation + TFS_long > TFS_short re-forming + OER 27 unstressed) which are precisely what Pre-Mom framework rewards. Bear's chronic-laggard critique (RSS_long 35, TCS_long 45) is valid as TIMING caveat rather than thesis killer — those metrics should be low at Pre-Mom entry point. Neutral's 55-60% hit rate anchor keeps from over-sizing. Lean long with disciplined cap.",
         "HOLD",1,"Quarter-size only; wait for TCS_long >55 confirmation before scaling.",
         "Risk-averse weights lagging confirmation (TCS_long 45, RSS_long 35, sub-10% structural ROE). RECOVERY can fail back to FADING if macro cracks. European banks face NII compression as ECB cuts. Postbank litigation tail + CRE exposure. Quarter-size until structural confirmation.",
         "ENTRY_TIMING", 1, 0.17),

    make("ARGX","stock","long","pre_momentum",62,"🔵 RECOVERY",
         "BUY",0.78,"BUY",0.75,"BUY",0.72,
         "BUY",3,"Full Pre-Mom LONG sizing (1.0x standard slot). Clean OER 12 supports adding on minor pullback to rising 20-SMA.",
         "ARGX presents the canonical Pre-Momentum LONG setup: Composite 62 + RECOVERY classification + exceptionally clean OER 12 (no overhead supply) + dense forward catalyst stack (Q2 earnings, multiple Phase 3 readouts, sub-Q launch ramp) on a platform franchise (Vyvgart/FcRn) with structural revenue and margin expansion runway. TFS softness (35/20) is only quant blemish but reads as stored energy in low-OER recovery context. Macro regime favors defensive-growth healthcare. Lone bear (concentration + FcRn competition) raises valid longer-tail risks but they fall outside bi-weekly hold horizon.",
         "BUY",2,"75% of full sizing; trim only if competitive Phase 3 (JNJ nipocalimab) reads positive in head-to-head indication.",
         "Risk-averse still BUY given setup quality + Vyvgart franchise + OER 12 cleanliness. Single-asset concentration (>90% rev from Vyvgart) + FcRn competitive entrants (JNJ/Immunovant) are real longer-tail risks but outside weekly~bi-weekly window. Size 75% to preserve flexibility.",
         "CONSENSUS_BUY", 0, 0.08),

    make("ITA","etf","long","pre_momentum",62,"🔵 FORMATION",
         "BUY",0.65,"BUY",0.70,"BUY",0.72,
         "BUY",3,"Full Pre-Mom long allocation; size up vs typical Pre-Mom candidate given FORMATION + STRONG agreement + low OER.",
         "Four-of-five bullish (Macro/Technical/Sentiment/Regime) with Fundamental Skeptic at neutral — not bearish, just not adding edge. FORMATION classification + low OER + accelerating short-horizon RSS is highest-quality Pre-Mom long setup the framework produces. Sister-ETF divergence (ITA bid vs SHLD Pre-Mom SHORT) confirms quality-within-theme rotation rather than blanket defense exhaustion. Multi-year backlog visibility (~1.7-2x revenue), bipartisan budget support, constructive sell-side revisions create momentum + fundamentals confluence. Modifier cap +3 binds.",
         "BUY",1,"Half-size starter; wait for TFS_long confirmation above 50 in next 4 weeks before scaling.",
         "Risk-averse trims: valuation on primes is full (18-22x fwd P/E vs 14-16x historical), TFS_long only 30 means long-horizon trend unconfirmed, URS 27 leaves limited underreaction fuel. Ukraine ceasefire / CR-driven award delay / single prime margin miss could break FORMATION pattern. Half-size.",
         "CONSENSUS_BUY", 0, 0.08),

    make("RSP","etf","long","pre_momentum",51,"🔵 RECOVERY",
         "BUY",0.72,"BUY",0.68,"HOLD",0.58,
         "BUY",2,"+2 modifier, capped below +3 ceiling because Composite 51 < 55 Eligibility Gate and breadth-widening catalyst not yet visible.",
         "RSP screens as structurally clean Pre-Momentum LONG candidate with embedded macro-regime option on breadth widening. Quant view dominates (TCS 100/100 + OER 15 is statistically rare and high-quality), Macro view supports (equal-weight = natural vehicle for any Mag7-fatigue rotation), Behavioral view tempers (URS 17 + TFS_long 0 means no behavioral/formation tailwind). Legitimate PROVISIONAL LONG worthy of +2 modifier, capped because Composite 51 below Eligibility Gate.",
         "HOLD",0,"Watch only; wait for Composite >55 or RSS_short >60 before initiating.",
         "Risk-averse: Composite 51 < 55 Eligibility Gate fails main momentum filter. TFS_long=0 is red flag not artifact. URS 17 (bottom decile) = no underreaction tailwind. 1-2 week hold mismatched with breadth-rotation thesis (typically resolves over 1-3 months).",
         "ENTRY_TIMING", 1, 0.20),

    make("UNG","etf","long","pre_momentum",48,"🔵 RECOVERY",
         "HOLD",0.55,"BUY",0.65,"HOLD",0.50,
         "HOLD",1,"Minimal +1 tactical sizing only; Composite 48 below Eligibility Gate + structural contango warrant restraint.",
         "Pre-Mom setup is real (short-horizon momentum + RECOVERY classification + low OER), but UNG's structural contango drag and broken long-horizon trend mean this is strictly a 2-6 week tactical trade. Composite 48 places it in Pre-Mom watchlist (not Momentum-confirmed). Size small, scale only on confirmation (Composite breaking 55+ with TCS_long inflecting up). Time-stop at 4 weeks regardless of P/L to avoid contango bleed.",
         "HOLD",0,"Pass; structural contango + 1.06% ER + broken long-horizon trend make this unattractive for risk-averse mandates.",
         "Risk-averse passes: futures wrapper structural ~30-50bp/month contango + 1.06% ER + every long-horizon dimension broken (TCS_long 21, RSS_long 22, TFS_long 30). Counter-trend bounce inside multi-year downtrend, not regime change. False starts in natgas are the norm.",
         "ENTRY_TIMING", 1, 0.22),

    make("DASH","stock","short","momentum",20,"🔴 CYCLE_PEAK",
         "SELL",0.80,"SELL",0.85,"AVOID",0.62,
         "SELL",-3,"Maintain core short, do not press at washed-out Composite 20 / OER 15. Add to short on relief bounces toward Composite 30-40 retest.",
         "Bear thesis is structurally validated by 4 of 5 agents with strong fundamental, technical, and macro alignment. DASH exhibits the full post-darling unwind pattern: Composite 20, TCS/TFS zeroed, CYCLE_PEAK classification, RSS deeply negative, and revision cycle turning down. Cover discipline is the key risk: OER 15 + elevated short interest creates squeeze setup, so press shorts on rallies (Composite 30-40 retests, SMA20 reclaim failures) rather than chase weakness at washed-out levels. Hard cover trigger on classification exit from CYCLE_PEAK or Composite >35 sustained.",
         "AVOID",-1,"Reduce short exposure 25-30% given asymmetric squeeze risk; re-establish on bounce to Composite 35+.",
         "Cover-risk respect: at Composite 20 / OER 15 short trade is mature not fresh. Quality metrics (positive FCF, net cash, improving contribution margin) provide structural floor. Elevated short interest creates squeeze risk on positive catalyst. Post-darling unwinds bottom 6-9 months after CYCLE_PEAK entry.",
         "EXIT_TIMING", 2, 0.28),

    make("BILI","stock","short","momentum",19,"⬇️ DOWNTREND",
         "SELL",0.70,"SELL",0.80,"SELL",0.60,
         "SELL",-3,"Maintain core short, do not press at washed-out levels. Add on relief bounce toward Composite 30-35.",
         "BILI presents one of cleanest Momentum SHORT signals in universe: Composite 19, DOWNTREND classification, all TCS signals at floor, RSS in bottom decile both horizons. Peer KWEB confirms group weakness while NTES Pre-Mom LONG status shows capital rotating AWAY from BILI within China internet sector. Fundamentally, GAAP profitability narrative is fragile — gaming hit-decay and ad ARPU compression remain unresolved. -3 reflects Risk Manager cover-risk concern (-4 aggressive view tempered).",
         "AVOID",-2,"Only initiate small (-2) on clean +6-10% counter-rally that fails technically. Don't chase weakness from depressed levels.",
         "Risk-averse view: late to trade. Shorting at Composite 19 with OER 0 means initiating after trend fully expressed — marginal seller largely exhausted, asymmetry skews toward COVER risk. China ADR borrow costs + squeeze tail (policy headlines) make small-size carry costly on weekly hold.",
         "EXIT_TIMING", 1, 0.22),
]


def main() -> None:
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    for v in VERDICTS:
        cache[v["ticker"]] = v
    cache["_meta"] = {
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_verdicts": len([k for k in cache if not k.startswith("_")]),
        "tier": "A",
    }
    CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(VERDICTS)} live-scan verdicts; cache now has {cache['_meta']['n_verdicts']} verdicts")


if __name__ == "__main__":
    main()
