# -*- coding: utf-8 -*-
"""refresh_from_queue.py — In-session helper for Live Scan cache refresh.

Workflow:
1. Dashboard's Live Scan button (after scan completes) calls
   POST /api/conviction-debate/refresh-queue → server writes
   .debate_refresh_queue.json with uncached live picks.

2. In a Claude Code session, the user runs:
       python3 agents/refresh_from_queue.py --print-prompts
   This script reads the queue and prints ready-to-dispatch sub-agent prompts.

3. The user (or the assistant) dispatches the printed sub-agents in parallel
   using the Agent tool. After all return, the user runs:
       python3 agents/refresh_from_queue.py --persist <verdicts.json>
   to write the persisted verdicts into .multi_agent_debate_cache.json
   and clear the queue.

All LLM work is plan-billed via sub-agent invocations — server stays LLM-free.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle as _pickle
import time
from pathlib import Path

QUEUE_PATH = Path(".debate_refresh_queue.json")
CACHE_PATH = Path(".multi_agent_debate_cache.json")
FUND_CACHE_PATH = Path(".fundamentals_cache.pkl")


# ─────────────────────────────────────────────────────────────────────
# Fundamentals block — extract per-ticker raw fundamental data from
# fundamentals_cache.pkl so that BOTH Fundamental specialist (intrinsic
# value, EPS growth) AND Valuation specialist (multiples vs fair value)
# can ground their reasoning in concrete numbers instead of guessing.
# ─────────────────────────────────────────────────────────────────────

_FUND_CACHE: dict | None = None


def _load_fund_cache() -> dict:
    global _FUND_CACHE
    if _FUND_CACHE is not None:
        return _FUND_CACHE
    try:
        if not FUND_CACHE_PATH.exists():
            _FUND_CACHE = {}
            return _FUND_CACHE
        data = _pickle.load(open(FUND_CACHE_PATH, "rb"))
        _FUND_CACHE = data.get("tickers", {}) if isinstance(data, dict) else {}
    except Exception:
        _FUND_CACHE = {}
    return _FUND_CACHE


def _fnum(v, fmt: str = "{:.1f}", default: str = "—") -> str:
    if v is None:
        return default
    try:
        f = float(v)
        if not math.isfinite(f):
            return default
        return fmt.format(f)
    except Exception:
        return default


def _fpct(v, default: str = "—") -> str:
    """v expected as decimal (0.10 → +10.0%)."""
    if v is None:
        return default
    try:
        f = float(v)
        if not math.isfinite(f):
            return default
        return f"{f * 100:+.1f}%"
    except Exception:
        return default


def build_fundamentals_block(ticker: str) -> str:
    """Format the per-ticker fundamental data into a prompt-friendly block.

    Returns '' if the ticker is not in fundamentals_cache or has no usable data.
    """
    cache = _load_fund_cache()
    rec = cache.get(ticker)
    if not rec or not isinstance(rec, dict):
        return ""

    info = rec.get("info") or {}
    fh_metrics = rec.get("finnhub_metrics") or {}
    fh_derived = rec.get("finnhub_derived") or {}
    estimates = rec.get("estimates") or {}
    revisions = rec.get("revisions") or {}
    recs = rec.get("recommendations") or {}
    pts = rec.get("price_targets") or {}
    asset_type = rec.get("asset_type", "Stock")

    lines: list[str] = []

    if asset_type == "ETF":
        ttm_pe = info.get("trailing_pe")
        er = info.get("expense_ratio")
        if ttm_pe is None and er is None:
            return ""
        lines.append("## FUNDAMENTALS (ETF — basket aggregate)")
        if ttm_pe is not None:
            lines.append(f"- Basket trailing P/E (weighted): {_fnum(ttm_pe)}")
        if er is not None:
            lines.append(f"- Expense ratio: {_fpct(er)}")
        return "\n".join(lines)

    # ── STOCK ──
    lines.append("## FUNDAMENTALS (raw — .fundamentals_cache.pkl)")
    lines.append("(Fundamental specialist: primary use. Valuation specialist: may cite multiples as price-vs-fair-value evidence.)")

    # Valuation multiples
    fpe = info.get("forward_pe") or fh_metrics.get("peNormalizedAnnual")
    tpe = info.get("trailing_pe") or fh_metrics.get("peTTM")
    pb = info.get("price_to_book") or fh_metrics.get("pbQuarterly")
    peg = info.get("peg")
    val_parts = []
    if fpe is not None: val_parts.append(f"fwd P/E {_fnum(fpe)}")
    if tpe is not None: val_parts.append(f"TTM P/E {_fnum(tpe)}")
    if peg is not None: val_parts.append(f"PEG {_fnum(peg, '{:.2f}')}")
    if pb is not None:  val_parts.append(f"P/B {_fnum(pb, '{:.2f}')}")
    if val_parts:
        lines.append(f"- Valuation multiples: {' · '.join(val_parts)}")

    # Quality (TTM)
    def _scale(fh_val):
        try:
            return float(fh_val) / 100.0 if fh_val is not None else None
        except Exception:
            return None
    gm = info.get("gross_margin") or _scale(fh_metrics.get("grossMarginTTM"))
    om = info.get("operating_margin") or _scale(fh_metrics.get("operatingMarginTTM"))
    pm = info.get("profit_margin")
    roe = info.get("roe") or _scale(fh_metrics.get("roeTTM"))
    q_parts = []
    if gm is not None and gm != 0: q_parts.append(f"gross margin {_fpct(gm)}")
    if om is not None and om != 0: q_parts.append(f"op margin {_fpct(om)}")
    if pm is not None and pm != 0: q_parts.append(f"profit margin {_fpct(pm)}")
    if roe is not None and roe != 0: q_parts.append(f"ROE {_fpct(roe)}")
    if q_parts:
        lines.append(f"- Quality (TTM): {' · '.join(q_parts)}")

    # Forward estimates (consensus EPS + growth)
    y0 = estimates.get("0y") or {}
    y1 = estimates.get("+1y") or {}
    q_next = estimates.get("+1q") or {}
    e_parts = []
    if y0.get("eps_avg") is not None:
        e_parts.append(f"FY0 EPS ${_fnum(y0.get('eps_avg'), '{:.2f}')} (growth {_fpct(y0.get('growth'))})")
    if y1.get("eps_avg") is not None:
        e_parts.append(f"FY1 EPS ${_fnum(y1.get('eps_avg'), '{:.2f}')} (growth {_fpct(y1.get('growth'))}, rev_growth {_fpct(y1.get('rev_growth'))})")
    if q_next.get("eps_avg") is not None:
        e_parts.append(f"+1Q EPS ${_fnum(q_next.get('eps_avg'), '{:.2f}')} (growth {_fpct(q_next.get('growth'))})")
    if e_parts:
        n_a = y1.get("n_analysts") or y0.get("n_analysts")
        n_lbl = f" [n={int(n_a)}]" if n_a else ""
        lines.append(f"- Forward estimates{n_lbl}: {' · '.join(e_parts)}")

    # EPS revisions (30d) — LEADING signal
    rn = revisions.get("net_30d")
    rr = revisions.get("ratio_30d")
    if rn is not None:
        rr_str = f", {rr * 100:.0f}% up" if rr is not None else ""
        lines.append(f"- EPS revisions (30d): net {int(rn):+d}{rr_str}")

    # Analyst recommendation consensus
    if recs.get("total"):
        lines.append(
            f"- Analyst recs: {recs.get('strong_buy', 0)} SB / "
            f"{recs.get('buy', 0)} B / {recs.get('hold', 0)} H / "
            f"{recs.get('sell', 0)} S / {recs.get('strong_sell', 0)} SS "
            f"(n={recs.get('total')}, bullish {_fpct(recs.get('bullish_ratio'))})"
        )

    # Price target
    if pts.get("mean") and pts.get("upside_pct") is not None:
        n_pt = pts.get("n_analysts", "?")
        lines.append(
            f"- Analyst PT: mean ${_fnum(pts.get('mean'), '{:.2f}')} "
            f"(upside {pts.get('upside_pct'):+.1f}%, n={n_pt})"
        )

    # Finnhub-derived leading signals
    fh_parts = []
    if fh_derived.get("bullish_change_3m") is not None:
        fh_parts.append(f"analyst bullish-change 3M {_fpct(fh_derived['bullish_change_3m'])}")
    if fh_derived.get("eps_beat_rate") is not None:
        fh_parts.append(f"EPS beat rate {_fpct(fh_derived['eps_beat_rate'])}")
    if fh_derived.get("eps_surprise_avg") is not None:
        fh_parts.append(f"avg EPS surprise {fh_derived['eps_surprise_avg']:+.2f}")
    if fh_parts:
        lines.append(f"- Leading signals: {' · '.join(fh_parts)}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Prompt template (group/side aware)
# ─────────────────────────────────────────────────────────────────────

def _hold_horizon(group: str) -> str:
    return "weekly~bi-weekly" if group == "pre_momentum" else "≤ weekly"


def _modifier_range(side: str, group: str) -> str:
    if side == "long":
        return "-5..+5" if group == "momentum" else "-5..+3 (Pre-Mom cap +3)"
    return "-5..0"


def build_prompt(item: dict) -> str:
    ticker = item.get("ticker", "?")
    group = item.get("group", "momentum")
    side  = item.get("side", "long")
    asset = item.get("asset_type", "stock")
    name  = item.get("name", "")
    cls_str = item.get("classification", "")
    comp = item.get("composite", 0) or 0
    oer  = item.get("oer", 0) or 0
    tcs_s, tcs_l = item.get("tcs_short", 0) or 0, item.get("tcs_long", 0) or 0
    tfs_s, tfs_l = item.get("tfs_short", 0) or 0, item.get("tfs_long", 0) or 0
    rss_s, rss_l = item.get("rss_short", 0) or 0, item.get("rss_long", 0) or 0
    urs = item.get("urs", 0) or 0

    side_label = side.upper()
    group_label = "Momentum" if group == "momentum" else "Pre-Momentum"
    fund_block = build_fundamentals_block(ticker)
    fund_section = f"\n\n{fund_block}" if fund_block else ""

    return f"""Multi-Agent ConvictionDebate Round 1 + dual synthesis for **{ticker} ({name})** in the **{group_label} {side_label}** bucket. Hold {_hold_horizon(group)}. Modifier {_modifier_range(side, group)}.

## QUANT SNAPSHOT (Valuation specialist: primary technical evidence)
- {ticker} ({name})
- Composite: {comp:.0f} | Classification: {cls_str}
- OER: {oer:.0f} | TCS_short {tcs_s:.0f} / TCS_long {tcs_l:.0f}
- TFS_short {tfs_s:.0f} / TFS_long {tfs_l:.0f}
- RSS_short {rss_s:.0f} / RSS_long {rss_l:.0f} | URS {urs:.0f}{fund_section}

## TASK
Mental Round 1 Fund/Sent/Val with strict lane discipline → Dual Synthesis (neutral + averse).
Return strict JSON in a ```json fence using this exact schema (no extra fields):
```json
{{"ticker":"{ticker}","asset_type":"{asset}","side":"{side}","group":"{group}","tier":"A",
"rounds":[{{"round_num":1,
"fundamental":{{"persona":"fundamental","rating":"STRONG_BUY|BUY|HOLD|SELL|AVOID","confidence":0.0-1.0,"key_points":["..."],"biggest_risk":"...","biggest_opportunity":"...","raw_text":"1-2 sent","narrative_summary":"","critique":""}},
"sentiment":{{"persona":"sentiment","rating":"...","confidence":0.0-1.0,"key_points":["..."],"biggest_risk":"...","biggest_opportunity":"...","raw_text":"1-2 sent","narrative_summary":"","critique":""}},
"valuation":{{"persona":"valuation","rating":"...","confidence":0.0-1.0,"key_points":["..."],"biggest_risk":"...","biggest_opportunity":"...","raw_text":"1-2 sent","narrative_summary":"","critique":""}},
"notes":""}}],
"synthesis_neutral":{{"risk_mode":"neutral","rating":"...","position_modifier":N,"sizing_recommendation":"1-2 sent","reasoning":"4-6 sent with {group_label} ({_hold_horizon(group)}) framing","raw_text":"[rule-based neutral synthesis]"}},
"synthesis_averse":{{"risk_mode":"averse","rating":"...","position_modifier":N,"sizing_recommendation":"1-2 sent","reasoning":"3-5 sent","raw_text":"[rule-based averse synthesis]"}},
"converged_round":1,
"disagreement":{{"rating_axis":0-4,"specialist_dispersion":0.0-0.5,"type":"CONSENSUS_BUY|CONSENSUS_HOLD|CONSENSUS_SELL|ENTRY_TIMING|EXIT_TIMING|POLAR_SPLIT|WIDE_SPAN"}},
"composite_at_time":{comp:.0f},"classification_at_time":"{cls_str}","generated_at":"{time.strftime('%Y-%m-%dT%H:%M:%S')}"}}
```
Use WebSearch sparingly (≤2 queries). Return ONLY the fenced ```json block."""


# ─────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────

def cmd_print_prompts() -> None:
    if not QUEUE_PATH.exists():
        print("[refresh] no queue file — run Live Scan in dashboard first")
        return
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    items = queue.get("tickers") or []
    print(f"[refresh] {len(items)} tickers queued (queued_at={queue.get('queued_at')})")
    print()
    for i, item in enumerate(items, 1):
        print(f"━━━━━━━━━━━━━━━━━━━━ ({i}/{len(items)}) {item['ticker']} ━━━━━━━━━━━━━━━━━━━━")
        print(build_prompt(item))
        print()


def cmd_persist(verdicts_json_path: str) -> None:
    p = Path(verdicts_json_path)
    if not p.exists():
        raise SystemExit(f"verdicts file not found: {verdicts_json_path}")
    verdicts = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(verdicts, list):
        raise SystemExit("verdicts file must be a JSON array")

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    for v in verdicts:
        if not isinstance(v, dict) or "ticker" not in v:
            continue
        cache[v["ticker"]] = v
    cache["_meta"] = {
        "last_update": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_verdicts": len([k for k in cache if not k.startswith("_")]),
        "tier": "A",
    }
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[refresh] persisted {len(verdicts)} verdicts; cache now {cache['_meta']['n_verdicts']}")

    # Clear queue
    if QUEUE_PATH.exists():
        QUEUE_PATH.unlink()
        print(f"[refresh] cleared queue {QUEUE_PATH}")


def cmd_show_queue() -> None:
    if not QUEUE_PATH.exists():
        print("[refresh] no queue file")
        return
    queue = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
    print(json.dumps(queue, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Refresh debate cache from queue")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("show", help="Show current queue contents")
    sub.add_parser("print-prompts", help="Print sub-agent prompts ready for dispatch")
    persist = sub.add_parser("persist", help="Persist verdicts JSON to cache and clear queue")
    persist.add_argument("verdicts_json")
    ap.add_argument("--print-prompts", action="store_true",
                    help="Shortcut for `print-prompts` subcommand")
    ap.add_argument("--persist", type=str, default=None,
                    help="Shortcut for `persist <file>` subcommand")
    args = ap.parse_args()

    if args.print_prompts or args.cmd == "print-prompts":
        cmd_print_prompts()
    elif args.persist or args.cmd == "persist":
        cmd_persist(args.persist or args.verdicts_json)
    else:
        cmd_show_queue()


if __name__ == "__main__":
    main()
