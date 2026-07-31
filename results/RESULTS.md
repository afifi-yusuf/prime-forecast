# prime-forecast: evaluation results

> **v2 run results added below** (see "v2 main run"). Narrative:
> `research-arc.md`. Reward evolution: `../docs/reward-design.md`.

## v2 main run (env 0.1.15, run `mjkvxreh4rna7trg1621r9kt`)

Single epoch over dataset v4 (33 steps × 64 × 8; 2,113 train questions),
graded anti-anchoring ramp (w=0.15, ε=0.08), SearXNG search, max_turns 10,
≤3 searches. Both evals platform-served on the v4 test split (n=265, crowd
Brier 0.189); raw per-rollout data: `v2_run_webhook.jsonl.gz`, split pinned
as `test_split_v4.jsonl`.

| | base (step-0 eval) | trained v2 (step-33 eval) |
|---|---|---|
| submit rate | 64% | **100%** |
| soft-Brier (0.5-imputed) | 0.215 [0.194, 0.235] | **0.211 [0.188, 0.233]** |
| median \|p − crowd\| | 0.050 | 0.080 (= ramp ε) |
| within 0.02 of crowd | 16% | **4%** |

## Frontier panel — all policies in the SAME agentic harness

Identical protocol for every row: v4 test split (n=265), max_turns 10, ≤3
searches, market tools on, SearXNG search, leak filter active. Frontier models
served via LiteLLM→Bedrock (Anthropic, AWS credits) and →Vertex (Google, GCP
credits); our models platform-served on Prime.

| policy | soft-Brier | ECE | submit | trading P&L/bet | median \|p−crowd\| |
|---|---|---|---|---|---|
| Claude Opus 4.5 | **0.186** [.161,.211] | **0.053** | 98% | **+$0.042** | 0.035 |
| Claude Sonnet 4.5 | 0.191 [.164,.217] | 0.063 | 100% | +$0.010 | 0.045 |
| Gemini 3.1 Pro | 0.197 [.169,.224] | **0.062** | 94% | +$0.008 | 0.042 |
| Gemini 3.6 Flash | 0.196 [.169,.222] | 0.085 | 99% | −$0.015 | 0.040 |
| **trained 35B (v2)** | 0.211 [.188,.233] | 0.119 | 100% | — | 0.080 |
| base 35B | 0.215 [.194,.235] | 0.170 | 64% | — | 0.050 |
| **crowd (market)** | **0.189** | — | — | — | — |

**Finding 7 (accuracy ceiling is the scaffold; capability shows up as
calibration).** Five policies spanning ~3B–frontier scale land in a 0.191–0.215
Brier band with fully overlapping CIs, and *none* beats the market. Calibration,
by contrast, separates monotonically with model capability: Sonnet 0.063 ≈
Gemini Pro 0.062 < Flash 0.085 < trained-35B 0.119 < base-35B 0.170. RL training
moves the open 35B a third of the way up that ladder (0.170→0.119) without
touching the accuracy ceiling. Only the two strongest frontier models turn
(marginally) positive simulated trading profit.

Raw data: `opus45_harness_eval.jsonl`, `sonnet45_harness_eval.jsonl`, `gemini36flash_harness_eval.jsonl`,
`gemini31pro_harness_eval.jsonl`.

**Protocol caveats (frontier rows).** Bedrock/Vertex tool-schema validators
required a proxy-side compatibility layer: Optional-union flattening,
`additionalProperties:false`, a compacted belief schema, belief updates only on
`submit`, and a 7-of-13 core toolset (dropped tools saw <5% usage by any model).
Our models ran the full 13-tool protocol. **Validity note:** an early "Gemini
Flash" run was discovered to be Sonnet 4.5 — `prime eval`'s config file silently
overrides its `-m` flag; results were retracted and re-run with per-model
configs (`configs/eval/frontier-<model>.toml`). Model identity was verified from
output-directory naming and proxy call logs.

### The complete 2×2 (v4 test, n=265, all platform-served)

| soft-Brier (ECE) | tools ON | tools OFF |
|---|---|---|
| trained v2 | **0.211** [.188,.233] (ECE **0.119**) | **0.250** [.224,.277] (ECE **0.127**) |
| base | 0.215 [.194,.235] (ECE 0.170) | 0.254 [.228,.280] (ECE 0.185) |
| crowd | 0.189 | — |

Submit rates: trained 100%/97%, base 64%/84%. Raw data:
`v2_toolsoff_eval.jsonl`, `base_toolsoff_eval.jsonl` (filter against
`test_split_v4.jsonl`). Quadrant findings: (i) tools worth ~0.04 Brier to
both models; training worth ~0.004 (n.s.) — accuracy is scaffold-bound,
extending Mantic's scaffold-dominates-weights result to trained retrieval;
(ii) training's consistent effect is calibration: ECE −30% in both columns,
surviving crowd removal; (iii) behavior: coverage, independence, efficiency.

Headline findings: cliff-camping eliminated (v1: 0.028 median / 32% inside
0.02) but the policy re-anchors at the new boundary ε; accuracy statistically
unchanged while coverage rises 64%→100% and crowd-distance triples —
internalized skill without headline Brier movement. Training curve: 100%
submit by step 15; several late batches beat their batch crowd (e.g. step 19:
0.170 vs 0.194). Platform artifact loss recurred: step-33 checkpoint never
uploaded, step-33 adapter stuck UPLOADING (final "Latest" adapter READY);
step-22 checkpoint is the last warm-startable artifact.

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

**Cross-paper primary metric** (soft-Brier, all 160 questions, non-answers
imputed at 0.5 — identical to Turtel et al.'s 0.25-penalty convention):
**trained 0.216 [0.186, 0.247] · base 0.236 · crowd 0.191.**
Submit rate below is an environment-internal compliance diagnostic (analogous
to Turtel's unparseable-output rate), not a headline skill metric.
Reference: raw v3 test split archived as `results/test_split_v3.jsonl` —
analyses of `platform_eval_step30_test.jsonl` must filter against it (the
live `data/` splits have since been rebuilt to v4).

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
