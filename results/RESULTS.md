# prime-forecast: evaluation results

> **v3 no-market run added** (run `slgbosbzsn4dzt2706bux0qx`): trained and
> evaluated without market tools. Base 0.261 → trained 0.245 (paired +0.016,
> t=1.25 n.s.), ECE 0.160→0.110, submit 74→98%; median prediction parked at
> the dataset base rate (0.35 vs 0.355) with extremes collapsing 10%→2% —
> pre-registered base-rate-herding prediction confirmed (Finding 10, anchor
> ladder). Raw: `v3_run_webhook.jsonl.gz`. Narrative:
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

"Market tools OFF" removes only the Polymarket price/history/search tools;
web search (≤3), URL reads, Wikipedia and time-series tools remain in BOTH
columns. The ablated variable is access to the crowd price, not research.

| soft-Brier (ECE) | market tools ON | market tools OFF |
|---|---|---|
| trained v2 | **0.211** [.188,.233] (ECE **0.119**) | **0.250** [.224,.277] (ECE **0.127**) |
| base | 0.215 [.194,.235] (ECE 0.170) | 0.254 [.228,.280] (ECE 0.185) |
| Sonnet 4.5 | 0.191 [.164,.217] (ECE 0.063) | 0.246 [.216,.276] (ECE 0.130) |
| **trained v3 (no-mkt trained)** | — | **0.245 [.221,.268] (ECE 0.110)** |
| Gemini 3.6 Flash | 0.196 [.169,.222] (ECE 0.085) | 0.259 [.226,.293] (ECE 0.157) |
| Gemini 3.1 Pro | 0.197 [.169,.224] (ECE 0.062) | 0.272 [.234,.309] (ECE 0.220, submit 89%) |
| crowd | 0.189 | — |

**Anchor-worth (paired within-model, same questions, p<0.001 all rows):**

| policy | anchor-worth [95% CI] |
|---|---|
| Gemini 3.1 Pro | +0.075 [+0.040, +0.110] |
| Gemini 3.6 Flash | +0.063 [+0.033, +0.094] |
| Sonnet 4.5 | +0.058 [+0.031, +0.085] |
| trained 35B | +0.039 [+0.016, +0.063] |
| base 35B | +0.039 [+0.016, +0.063] |

Frontier models lean MORE on the crowd than the 35Bs — the frontier's
in-harness advantage is substantially better anchor exploitation. Without
the market, Gemini Pro's ECE (0.220) is worse than the untrained 35B's, and
frontier accuracy (0.246-0.272) collapses into/below the 35B band
(0.250-0.254). Raw: `gemini36flash_notools_eval.jsonl`,
`gemini31pro_notools_eval.jsonl`.

Submit rates: trained 100%/97%, base 64%/84%. Raw data:
`v2_toolsoff_eval.jsonl`, `base_toolsoff_eval.jsonl` (filter against
`test_split_v4.jsonl`). Quadrant findings: (i) the market anchor is worth ~0.04 Brier to
both models (research tools held constant); training worth ~0.004 (n.s.) — accuracy is scaffold-bound,
extending Mantic's scaffold-dominates-weights result to trained retrieval;
(ii) training's consistent effect is calibration: ECE −30% in both columns,
surviving crowd removal; (iii) behavior: coverage, independence, efficiency.

**Finding 9 (the market anchor explains "beats frontier" claims — and yields
an honest parity result).** Sonnet 4.5 without market tools scores 0.246
(ECE 0.130): the anchor is worth 0.055 to a frontier model, and our trained
35B WITH the anchor (0.211) "beats" Sonnet without it — reproducing the
Turtel/Mantic claim shape (their frontier baselines had frozen context and
no market price) by scaffold asymmetry alone. In the matched no-market condition
ALL pairwise accuracy differences are n.s. at n=265 (bootstrap CIs span 0;
even base~Sonnet) — no accuracy claim is available in this cell for anyone.
Calibration point estimates put trained (0.127) at Sonnet parity (0.130)
with base worse, but per-cell ECE CIs (±0.07) make no single pairwise diff
significant; the calibration claim rests on the cross-eval pattern (trained
ECE < base ECE in all four independent evals), not this cell alone. NOTE:
ECE is imputation-sensitive (base submits 84% here; 0.5-imputed 0.143 vs
submitted-only 0.185) — paper must pin one convention. Raw:
`sonnet45_notools_eval.jsonl`.

Headline findings: cliff-camping eliminated (v1: 0.028 median / 32% inside
0.02) but the policy re-anchors at the new boundary ε; accuracy statistically
unchanged while coverage rises 64%→100% and crowd-distance triples —
internalized skill without headline Brier movement. Training curve: 100%
submit by step 15; several late batches beat their batch crowd (e.g. step 19:
0.170 vs 0.194). Platform artifact loss recurred: step-33 checkpoint never
uploaded, step-33 adapter stuck UPLOADING (final "Latest" adapter READY);
step-22 checkpoint is the last warm-startable artifact.

Model: Qwen3.5-35B-A3B + LoRA (r=16, α=32). v4 test split: 265 resolved
Polymarket questions, resolutions 2026-07-01 onward, strictly after all
training data.

## Archive: v1 run (superseded by v2; kept for findings it grounds)

v1: 45-step GRPO, 9 epochs over 1,280 questions, binary crowd-copy cliff
(0.20 @ eps=0.02), <=2 searches. Final checkpoint lost to the platform
upload bug; step-30 evaluated. v3 test split (n=160), raw data
`platform_eval_step30_test.jsonl` (filter against `test_split_v3.jsonl`).

Numbers still cited by the paper:

- **Anchoring (Finding 1):** median |p-crowd| 0.028, 32% of predictions
  within 0.02 of the market -- the policy camped just outside the penalty
  cliff. Trading P&L -$0.08/bet. Soft-Brier 0.216 [0.186, 0.247] vs base
  0.236 vs crowd 0.191; submit 94% vs base 31%; ECE 0.119.
- **Serving gap (Finding 2):** same merged weights produced 48% submit
  locally vs 94% platform-served (merge verified tensor-exact against
  prime-rl semantics: scale alpha/r=2.0, GATE_DOWN_UP expert mapping).
  Local evals understate trained policies; all headline evals therefore run
  through the platform stack (webhook methodology). Local-serving 2x2 data
  retained in git history only.

## Provenance / method notes

- **Leak safety**: 5 layers (domain blocklist, provider date filters,
  date/outcome heuristics, Claude Haiku 4.5 KEEP/DROP filter -- benchmarked
  against 5 cheaper models, all failed the undated-inference cases -- plus
  cutoff-clamped tools). Search cache doubles as an auditable corpus of all
  retrieved context.
- **Filter-model benchmark**: Haiku 4.5 the only model with zero
  leak-direction errors; Qwen3-32B validated fallback (zero false-KEEPs,
  ~12% over-drop, ~7x cheaper).
- **Known platform issues** (reported): run-end checkpoint/adapter pipeline
  failures (v1 step-45 and v2 step-33 artifacts both lost; v2 tools-off
  cell therefore uses the step-22 checkpoint, likely conservative),
  checkpoint deploy 502s, LoRA deploy unsupported for this base,
  platform-vs-local serving gap.
- **Costs (approx, whole project)**: ~$120 Prime compute across three
  training runs + probes + evals; ~6k Firecrawl credits; Bedrock leak
  filtering + Anthropic panel rows on AWS credits; Gemini rows on GCP
  credits via Vertex; SearXNG + webhook on Azure (~$0.08/h).
