"""
Phase 1 Simulation — Volatility-Adjusted Buffer 적용 시 분류 변화 추정

목적: 전체 스캔 없이 현재 cache의 results 데이터를 사용해
      새 classify() 로직(adaptive buffer)으로 재분류했을 때의 변화를 즉시 시뮬레이션.

주의: results dict는 sma20_dist 등 raw 필드를 포함하지 않을 수 있음.
      필드가 없는 경우 시뮬레이션 불가 (현재 데이터의 한계).
"""

import os
import sys
import pickle
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def simulate_phase1(cache_path: str = ".scan_cache.pkl"):
    """현재 cache + classify() 새 로직으로 분류 변화 시뮬레이션."""

    if not os.path.isabs(cache_path):
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_path)

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    results = cache.get("results", [])
    if not results:
        print("ERROR: results not found in cache")
        return

    # raw 필드 보유 여부 확인
    sample = results[0]
    needed = ['sma20_dist', 'sma20_slope', 'sma50_dist', 'sma50_sma200_spread',
              'sma50_slope', 'realized_vol']
    missing = [f for f in needed if f not in sample]
    if missing:
        print(f"⚠️  Missing raw fields in results: {missing}")
        print(f"    이는 현재 cache가 Phase 1 변경 적용 이전의 결과임을 의미합니다.")
        print(f"    다음 스캔(Run Live Scan)을 실행하면 새 raw 필드가 저장되고 시뮬레이션 가능합니다.")
        print()
        print(f"    현재 분류 분포 (Phase 0):")
        cls_dist = Counter(r.get('classification', '') for r in results)
        for cls, n in cls_dist.most_common():
            print(f"      {cls:25s}: {n}")
        return

    # ── 시뮬레이션 ──
    def reclassify(r, mode):
        """간소화된 base 분류만 시뮬 (override 제외)."""
        sma20_d = r.get('sma20_dist', 0)
        sma20_s = r.get('sma20_slope', 0)
        sma50_d = r.get('sma50_dist', 0)
        sma50_s = r.get('sma50_slope', 0)
        spread = r.get('sma50_sma200_spread', 0)
        rv = r.get('realized_vol', 0.02)

        if mode == "adaptive":
            daily_sigma = rv / np.sqrt(252) if rv > 1 else rv * 100
            short_buf = max(0.3, 0.4 * daily_sigma * np.sqrt(20))
            long_buf  = max(0.7, 0.6 * daily_sigma * np.sqrt(60))
            short_buf = min(short_buf, 3.0)
            long_buf  = min(long_buf, 6.0)
        else:
            short_buf = 0.5
            long_buf  = 1.0

        # short
        if sma20_d > short_buf and sma20_s > 0:
            short_dir = "UP"
        elif sma20_d < -short_buf and sma20_s < 0:
            short_dir = "DOWN"
        else:
            short_dir = "FLAT"

        # long
        if sma50_d > long_buf and (spread > 0 or sma50_s > 0):
            long_dir = "UP"
        elif sma50_d < -long_buf and (spread < 0 or sma50_s < 0):
            long_dir = "DOWN"
        else:
            long_dir = "FLAT"

        MATRIX = {
            ("UP",   "UP"):   "🟢 CONTINUATION",
            ("UP",   "FLAT"): "🔵 RECOVERY",
            ("UP",   "DOWN"): "🟣 COUNTER_RALLY",
            ("FLAT", "UP"):   "🟡 CONSOLIDATION",
            ("FLAT", "FLAT"): "🟠 NEUTRAL",
            ("FLAT", "DOWN"): "🟤 FADING",
            ("DOWN", "UP"):   "🔶 PULLBACK",
            ("DOWN", "FLAT"): "⚠️ WEAKENING",
            ("DOWN", "DOWN"): "⬇️ DOWNTREND",
        }
        return MATRIX[(short_dir, long_dir)], short_dir, long_dir, short_buf, long_buf

    print("=" * 80)
    print("  Phase 1 Simulation — Volatility-Adjusted Buffer")
    print("=" * 80)
    print(f"  Cache: {cache_path}")
    print(f"  Results count: {len(results)}")
    print()

    # 시뮬레이션 실행
    transitions = defaultdict(lambda: defaultdict(int))
    fixed_dist = Counter()
    adaptive_dist = Counter()

    sample_changes = []

    for r in results:
        cls_fixed, _, _, sb_f, lb_f = reclassify(r, "fixed")
        cls_adapt, _, _, sb_a, lb_a = reclassify(r, "adaptive")

        fixed_dist[cls_fixed] += 1
        adaptive_dist[cls_adapt] += 1
        transitions[cls_fixed][cls_adapt] += 1

        if cls_fixed != cls_adapt and len(sample_changes) < 10:
            sample_changes.append({
                'ticker': r.get('ticker', ''),
                'rv': r.get('realized_vol', 0),
                'sma20_d': r.get('sma20_dist', 0),
                'sma50_d': r.get('sma50_dist', 0),
                'short_buf': round(sb_a, 2),
                'long_buf': round(lb_a, 2),
                'fixed': cls_fixed,
                'adaptive': cls_adapt,
            })

    print(f"{'Class':25s} {'Phase 0 (fixed)':>15s} {'Phase 1 (adaptive)':>20s} {'Diff':>8s}")
    print("-" * 80)
    all_classes = sorted(set(fixed_dist.keys()) | set(adaptive_dist.keys()),
                         key=lambda c: -(fixed_dist.get(c, 0) + adaptive_dist.get(c, 0)))
    for cls in all_classes:
        f = fixed_dist[cls]
        a = adaptive_dist[cls]
        diff = a - f
        sign = '+' if diff > 0 else ''
        print(f"{cls:25s} {f:>15d} {a:>20d} {sign}{diff:>7d}")

    n_changed = sum(transitions[f][a] for f in transitions for a in transitions[f] if f != a)
    print()
    print(f"Total reclassified: {n_changed} / {len(results)} ({n_changed/len(results)*100:.1f}%)")

    if sample_changes:
        print()
        print("Sample reclassifications:")
        print(f"  {'Ticker':>10} {'RV':>6} {'sma20%':>7} {'sma50%':>7} {'shortBuf':>8} {'longBuf':>7}  {'Fixed':22s} → {'Adaptive':22s}")
        for s in sample_changes:
            print(f"  {s['ticker']:>10} {s['rv']:>6.2f} {s['sma20_d']:>7.2f} {s['sma50_d']:>7.2f} "
                  f"{s['short_buf']:>8.2f} {s['long_buf']:>7.2f}  "
                  f"{s['fixed']:22s} → {s['adaptive']:22s}")

    print()
    print("Top transitions:")
    for f, to_dict in sorted(transitions.items(), key=lambda x: -sum(v for to, v in x[1].items() if to != x[0])):
        for a, n in sorted(to_dict.items(), key=lambda x: -x[1]):
            if a == f or n < 3:
                continue
            print(f"  {f:25s} → {a:25s} : {n}")

    print()
    print("=" * 80)
    print("  Note: 이는 base classification만 시뮬 (override 제외).")
    print("        실제 effect는 다음 Run Live Scan에서 완전히 반영됩니다.")
    print("=" * 80)


if __name__ == "__main__":
    simulate_phase1()
