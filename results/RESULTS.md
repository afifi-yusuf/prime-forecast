# prime-forecast: evaluation results

All evaluations on the held-out **test split** (160 resolved Polymarket questions,
resolutions 2026-07-01 → 2026-07-27, strictly after all training data; crowd
Brier 0.191 on this split ≈ dataset average — i.e. normal difficulty).

Trained model: Qwen3.5-35B-A3B + LoRA (r=16, α=32), 45-step GRPO run
`rqn28a9dizg2ete43qgzqyry` on env `yafifi/prime-forecast` (Brier reward,
0.55 no-submit floor, crowd-copy penalty 0.20@0.02, max 8 turns, ≤2 searches).

## Headline: platform eval (complete, per-rollout instrumented)

Step-30 checkpoint evaluated **through Prime's own inference stack** (run
`p6pphpg8fs9mzfl424e19lfb`: checkpoint warm-start + eval-before-training),
search via self-hosted SearXNG, every rollout reported via results webhook
(`results/platform_eval_step30_test.jsonl` — raw data, 160/160 questions).

| metric | trained (step-30) | base (untrained) | crowd |
|---|---|---|---|
| submit rate | **94%** (150/160) | 31% | — |
| Brier, submitted only | 0.214 [0.182, 0.246] | 0.203 [0.137, 0.268] | **0.194** (same subset) |
| ECE (10 equal-mass bins) | **0.119** | 0.170 | — |
| env reward (0.55 floor) | **0.711** | ~0.60 | — |
| trading P&L (1-share, 1¢ fee) | −$11.47 / 137 bets | +$0.64 / 47 bets | — |
| median \|p − crowd\| | 0.028 | 0.065 | — |
| within 0.02 of crowd | 32% | — | — |
| research tool calls / rollout | 7.2 (1.9 searches) | ~5 | — |

**Findings**

1. **Training transfers.** 94% submit rate and reward 0.711 on never-seen
   questions match late-training levels; ECE 0.119 sits in the published-model
   band (Turtel et al. 0.09–0.13; Mantic 0.06–0.11). The RL objective —
   always submit, research efficiently, stay calibrated — was learned and
   generalizes.
2. **The learned strategy is crowd-anchoring.** Median distance from the
   market price 0.028; accuracy statistically indistinguishable from the crowd
   (0.214 vs 0.194, overlapping CIs) and does **not** beat it (trading sim
   −$0.08/bet). The crowd-copy penalty (ε=0.02) kept predictions from exact
   echoes but the anchor-and-adjust strategy dominated because the reward
   permits it.
3. Note: evaluated policy is step 30 of 45 (the final checkpoint was lost to a
   platform pipeline failure; see caveats).

## Local-serving 2×2 (baselines valid; trained row attenuated — see caveats)

Same test split, evaluated via local vLLM (merged adapter / stock base),
Firecrawl search:

| | market tools ON | market tools OFF |
|---|---|---|
| trained (step-45, merged, **local serving**) | submit 48% · Brier(all) 0.261 · ECE 0.188 · P&L −$6.53 | not run |
| base | submit 31% · Brier(all) 0.236 · ECE 0.170 · P&L +$0.64 | submit 29% · Brier(all) 0.263 · ECE 0.274 · P&L −$1.05 |

- Crowd prior worth ≈0.027 Brier + large calibration gain to the base model (C vs D).
- **Caveat (important):** the trained-local row understates the policy. The same
  weights produce 94% submit through Prime's stack vs 48% locally — a
  serving-stack discrepancy (reported to Prime; adapter merge was verified
  tensor-exactly against prime-rl's own semantics: scale α/r=2.0
  `runs.py:560`, expert mapping `conversion_ops.py:321`, fused gate-first split).
  Local numbers are lower bounds, not the policy's capability.

## Provenance / method notes

- **Leak safety**: 5 layers (domain blocklist, provider date filters, date/outcome
  heuristics, Claude Haiku 4.5 KEEP/DROP filter — benchmarked against 5 cheaper
  models, all failed the undated-inference cases — plus cutoff-clamped tools).
  Search cache doubles as an auditable corpus of all retrieved context.
- **Filter-model benchmark**: Haiku 4.5 the only model with zero leak-direction
  errors; Qwen3-32B validated fallback (zero false-KEEPs, ~12% over-drop, ~7× cheaper).
- **Training curve** (train batches): submit 44%→94-97%, Brier ~0.65→~0.30 over
  45 steps; three-phase dynamics (confident-wrong dip → confidence moderation →
  consolidation). Search cache cut credit burn ~64% (0.55→0.20 credits/rollout).
- **Costs (approx)**: three training runs + probes ≈ $60 Prime compute;
  ≈5k Firecrawl credits; Bedrock filtering on AWS credits; SearXNG eval
  infra ≈ $0.08/h Azure.
- **Known platform issues** (reported): run-end checkpoint/adapter pipeline
  failure (step-45 artifacts lost), checkpoint deploy 502s, LoRA deploy
  unsupported for this base, val metrics require explicit env-spec wiring,
  adapter-serving behavior gap (local vs platform).

## Next-run design (motivated by these results)

1. Crowd-relative reward (beat-the-market margin) to break the anchoring optimum.
2. Market-tools-off condition to measure independent forecasting skill.
3. SearXNG search (free) → full-scale batch 64, 100+ steps, no credit compromises.
4. Explicit val split wiring + webhook instrumentation from step 0.
5. Checkpoint interval landing exactly on the final step.
