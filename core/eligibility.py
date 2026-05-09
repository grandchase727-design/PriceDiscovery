"""Portfolio eligibility gate (Layer 5).

`evaluate_eligible` decides whether a single ticker analysis passes the
classification + composite + ADV filters. The QVR (4th condition) is applied
separately in api.py because fundamentals live in a different cache and are
loaded after the technical scan.
"""
from __future__ import annotations

from config.scoring import ADV_MIN_USD, ELIGIBLE_COMPOSITE


def evaluate_eligible(analysis, adv_usd, min_adv=ADV_MIN_USD, comp_threshold=ELIGIBLE_COMPOSITE):
    """Portfolio eligibility 평가.
    부적격 클래스: DOWNTREND, EXHAUSTING, FADING, COUNTER_RALLY, CYCLE_PEAK, WEAKENING.
      - WEAKENING (DOWN, FLAT): 단기 약세 + 장기 횡보 → 매수 진입 위험 (#8 fix).
      - OVEREXTENDED는 위험 신호이나 차익실현/관망용 — 부적격은 아니나 CLASS_RANK=1.
    LAGGING_CATCHUP은 적격 (URS 기반 catch-up 매수 후보)."""
    cls = analysis['classification']
    comp = analysis['composite']
    reasons = []
    if cls == "⬇️ DOWNTREND":
        reasons.append("Downtrend")
    if cls == "🟤 EXHAUSTING":
        reasons.append("Exhausting")
    if cls == "🟤 FADING":
        reasons.append("Fading")
    if cls == "🟣 COUNTER_RALLY":
        reasons.append("CounterRally")
    if cls == "🔴 CYCLE_PEAK":
        reasons.append("CyclePeak")
    if cls == "⚠️ WEAKENING":
        reasons.append("Weakening")
    if comp < comp_threshold:
        reasons.append("LowScore")
    if adv_usd < min_adv:
        reasons.append(f"Liq({adv_usd/1e6:.1f}M)")
    eligible = len(reasons) == 0
    return eligible, "/".join(reasons) if reasons else "None"
