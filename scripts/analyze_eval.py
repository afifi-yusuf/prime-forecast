#!/usr/bin/env python3
"""Publication-grade analysis of prime eval results.jsonl files.

Computes, per condition (Turtel et al. 2025 protocol):
  - Brier score with 95% Wald CI (no-submit rollouts scored as 0.25-penalty
    soft-Brier is NOT applied; missing probs use (0.5-y)^2 like a uniform
    fallback, reported separately)
  - ECE over 10 equal-mass bins (bootstrap 95% CI)
  - Crowd (market) Brier on the same questions
  - Simulated 1-share trading P&L vs price_at_cutoff with $0.01 fee
  - Submit rate, |p - crowd| distribution

Usage:
  python scripts/analyze_eval.py LABEL=path/to/results.jsonl [LABEL2=path2 ...]
"""

from __future__ import annotations

import json
import math
import random
import sys


def load(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def get_info(r: dict) -> dict:
    info = r.get("info")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            info = {}
    return info or {}


def extract(rows: list[dict]) -> list[dict]:
    """One record per rollout: p (or None), y, crowd."""
    out = []
    for r in rows:
        m = r.get("metrics") or {}
        info = get_info(r)
        p = m.get("predicted_prob")
        p = None if p is None or p < 0 else float(p)
        y_raw = info.get("outcome")
        if y_raw is None:
            # reconstruct from brier when info is stripped
            br = m.get("brier_score")
            if p is None or br is None:
                continue
            y = 1 if abs((p - 1) ** 2 - br) < abs(p**2 - br) else 0
        else:
            y = 1 if int(y_raw) == 1 else 0
        crowd = info.get("price_at_cutoff")
        if crowd is None and m.get("market_brier", -1) >= 0:
            mb = m["market_brier"]
            crowd = 1 - math.sqrt(mb) if y == 1 else math.sqrt(mb)
        out.append({"p": p, "y": y, "crowd": None if crowd is None else float(crowd)})
    return out


def wald_ci(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    half = 1.96 * math.sqrt(var / n)
    return mean, mean - half, mean + half


def ece(recs: list[dict], bins: int = 10) -> float:
    scored = sorted((r for r in recs if r["p"] is not None), key=lambda r: r["p"])
    n = len(scored)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(bins):
        chunk = scored[b * n // bins:(b + 1) * n // bins]
        if not chunk:
            continue
        conf = sum(r["p"] for r in chunk) / len(chunk)
        freq = sum(r["y"] for r in chunk) / len(chunk)
        total += (len(chunk) / n) * abs(conf - freq)
    return total


def ece_bootstrap_ci(recs: list[dict], iters: int = 1000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    scored = [r for r in recs if r["p"] is not None]
    if not scored:
        return float("nan"), float("nan")
    vals = sorted(ece([rng.choice(scored) for _ in scored]) for _ in range(iters))
    return vals[int(0.025 * iters)], vals[int(0.975 * iters)]


def trading_pnl(recs: list[dict], fee: float = 0.01) -> tuple[float, int]:
    """1-share Polymarket bet per question where |p - crowd| > fee."""
    pnl, n = 0.0, 0
    for r in recs:
        p, y, c = r["p"], r["y"], r["crowd"]
        if p is None or c is None or abs(p - c) <= fee:
            continue
        n += 1
        if p > c:   # buy YES at c
            pnl += (1.0 - c if y == 1 else -c) - fee
        else:       # buy NO at 1-c
            pnl += (c if y == 0 else -(1.0 - c)) - fee
    return pnl, n


def analyze(label: str, path: str) -> None:
    recs = extract(load(path))
    n = len(recs)
    submitted = [r for r in recs if r["p"] is not None]
    briers = [(r["p"] - r["y"]) ** 2 for r in submitted]
    briers_all = [((r["p"] if r["p"] is not None else 0.5) - r["y"]) ** 2 for r in recs]
    crowd = [(r["crowd"] - r["y"]) ** 2 for r in recs if r["crowd"] is not None]
    mean, lo, hi = wald_ci(briers) if briers else (float("nan"),) * 3
    mean_all, lo_all, hi_all = wald_ci(briers_all)
    e = ece(recs)
    elo, ehi = ece_bootstrap_ci(recs)
    pnl, bets = trading_pnl(recs)
    dists = sorted(abs(r["p"] - r["crowd"]) for r in submitted if r["crowd"] is not None)
    med_d = dists[len(dists) // 2] if dists else float("nan")

    print(f"\n== {label}  (n={n} rollouts, {len(submitted)} submitted, "
          f"submit rate {len(submitted)/n:.0%})")
    print(f"  Brier (submitted only):     {mean:.4f}  [{lo:.4f}, {hi:.4f}]")
    print(f"  Brier (0.5 fallback, all):  {mean_all:.4f}  [{lo_all:.4f}, {hi_all:.4f}]")
    if crowd:
        cm, cl, ch = wald_ci(crowd)
        print(f"  Crowd Brier (same rows):    {cm:.4f}  [{cl:.4f}, {ch:.4f}]")
    print(f"  ECE (10 equal-mass bins):   {e:.4f}  [{elo:.4f}, {ehi:.4f}]")
    print(f"  Trading P&L (1 share, 1c fee): ${pnl:+.2f} over {bets} bets "
          f"(${pnl/bets:+.4f}/bet)" if bets else "  Trading: no bets")
    print(f"  median |p - crowd|:         {med_d:.3f}")


def main() -> None:
    for arg in sys.argv[1:]:
        label, _, path = arg.partition("=")
        analyze(label, path)


if __name__ == "__main__":
    main()
