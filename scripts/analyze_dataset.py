#!/usr/bin/env python3
"""Dataset stats: categories, outcomes, market-baseline calibration, splits.

Usage: python scripts/analyze_dataset.py [data/manifest.jsonl]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "manifest.jsonl"
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if not rows:
        print("no rows")
        return

    n = len(rows)
    print(f"rows: {n}")
    print(f"splits: {dict(Counter(r.get('split') for r in rows))}")
    print(f"categories: {dict(Counter(r.get('category') for r in rows).most_common())}")
    print(f"outcomes: {dict(Counter(r.get('outcome') for r in rows))}")
    print(f"cutoff_policy: {dict(Counter(r.get('cutoff_policy') for r in rows))}")

    res_dates = sorted(str(r.get("resolution_date") or "") for r in rows)
    print(f"resolution dates: {res_dates[0][:10]} .. {res_dates[-1][:10]}")

    # Market baseline: Brier of price_at_cutoff vs outcome, overall and per split.
    def brier(rs) -> float | None:
        pairs = [(float(r["price_at_cutoff"]), int(r["outcome"]))
                 for r in rs if r.get("price_at_cutoff") is not None]
        if not pairs:
            return None
        return sum((p - y) ** 2 for p, y in pairs) / len(pairs)

    print(f"\nmarket baseline Brier (price_at_cutoff): {brier(rows):.4f}")
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r.get("split") == split]
        b = brier(rs)
        if b is not None:
            print(f"  {split:5s}: {b:.4f}  (n={len(rs)})")

    # Calibration of the crowd price in 10 buckets.
    buckets: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for r in rows:
        p = r.get("price_at_cutoff")
        if p is None:
            continue
        buckets[min(9, int(float(p) * 10))].append((float(p), int(r["outcome"])))
    print("\ncrowd calibration (bucket: mean_price -> freq_yes, n):")
    for b in sorted(buckets):
        pairs = buckets[b]
        mp = sum(p for p, _ in pairs) / len(pairs)
        fy = sum(y for _, y in pairs) / len(pairs)
        print(f"  [{b/10:.1f}-{(b+1)/10:.1f}): {mp:.3f} -> {fy:.3f}  (n={len(pairs)})")

    # Volume distribution.
    vols = sorted(float(r.get("volume") or 0) for r in rows)
    print(f"\nvolume: min={vols[0]:.0f} p50={vols[n//2]:.0f} p90={vols[int(n*0.9)]:.0f} max={vols[-1]:.0f}")


if __name__ == "__main__":
    main()
