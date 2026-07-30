# Research arc: what outcome-based RLVR learns when given the market

Chronological narrative of the project's experiments and findings, for the
paper. Companion docs: `docs/reward-design.md` (reward evolution),
`RESULTS.md` (metric tables), `docs/prime-support-ticket.md` (platform issues).

## Question

Both Turtel et al. (2025) and Mantic (2026) freeze the research context before
RL training and name in-the-loop retrieval as the consequential extension.
We fold context acquisition into the RL loop: a multi-turn agent researches
resolved Polymarket questions with live cutoff-safe tools (web search, page
reads, time-series, Wikipedia-as-of-date, and crowd-price tools) and is
rewarded on Brier score against the real outcome.

## Experiment 1 — v1 run: pure Brier + binary anti-copy cliff

45 GRPO steps, 9 epochs over 1,280 questions, fixed 5–14d horizons.
Training metrics transformed (submit 44%→97%, train-batch Brier →0.30).
Held-out platform eval of the trained policy:

- Submit 94%, ECE 0.119 (published-model band), soft-Brier 0.216 vs crowd 0.191.
- **Finding 1 (anchoring):** the policy's dominant strategy was to read the
  crowd price and sit at median |p−crowd| = 0.028 — just outside the 0.02
  binary penalty cliff (32% of predictions inside 0.02). Crowd-matching, not
  crowd-beating; simulated trading −$0.08/bet.
- **Finding 2 (serving gap):** locally-served exports of the same weights
  reproduced only a fraction of the behavior (48% vs 94% submit) despite
  tensor-exact merge verification against prime-rl's own semantics; evals must
  run through the training platform's serving (webhook methodology below).

## Methodology contribution — complete evals through the training platform

The platform stores only 16 rollout samples per step. We added a results
webhook to the environment: every scored rollout POSTs its complete record
(question id, p, outcome, crowd price, tool usage) to an endpoint we control.
Combined with checkpoint-warm-start probe runs that evaluate before training
(`skip_first_step=false`), this yields complete held-out evals through the
exact serving stack that trained the policy — at ~$3–5 per condition.

## Experiment 2 — v2 run: dataset + reward redesign

Changes, each motivated by a measured v1 failure:

1. **Dataset v4**: 2,642 questions / 1,377 unique events; cutoffs drawn
   uniformly within each market's lifetime (Turtel's protocol) instead of a
   fixed grid — continuous 5–130d horizon spread; caps of ≤5 per question
   template and ≤4 markets per event (correlated siblings).
2. **Single epoch** (33 steps × batch 64 ≈ one pass over 2,113 train
   questions), matching both papers' no-repetition regime; v1's 9 epochs were
   a plausible co-driver of anchoring.
3. **Graded anti-anchoring ramp** replacing the cliff: penalty
   w·(1−dist/ε) inside ε (w=0.15, ε=0.08). Design note: GRPO group-relative
   advantages cancel question-constant reward terms, so "beat the crowd"
   bonuses are gradient no-ops; only within-group-varying terms (like each
   rollout's crowd distance) can shape strategy.

**Results (held-out, n=265, platform-served, baseline = untrained base
through identical harness):**

| | base | trained v2 | crowd |
|---|---|---|---|
| submit | 64% | 100% | — |
| soft-Brier | 0.215 | 0.211 [0.188–0.233] | 0.189 |
| median \|p−crowd\| | 0.050 | 0.080 | — |
| within 0.02 | 16% | 4% | — |

- **Finding 3 (boundary relocation):** the ramp destroyed cliff-camping
  (0.028 → 0.080 median, 32% → 4% inside 0.02) — and the policy re-anchored
  precisely at the new boundary ε=0.08, confirming the pre-registered
  hypothesis: penalty-boundary-seeking is robust under GRPO; shaping moves
  the anchor rather than removing it.
- **Finding 4 (skill under the surface):** accuracy is statistically
  unchanged (0.211 vs 0.215, n.s.) — but the trained model achieves it while
  answering 100% of questions (vs a self-selected 64%) and positioned 3×
  further from the near-optimal crowd. Holding accuracy constant while
  surrendering the crowd prior implies internalized skill that headline
  Brier cannot show at n=265 (CI ±0.023 vs ~0.026 total base-to-crowd
  headroom).
- Neither model beats the crowd (0.189) — consistent with both reference
  papers, whose trained models also do not beat market prices.

## Experiment 3 — the harness × training quadrant (standalone skill)

Both remaining cells measured platform-served on the same 265 test questions
(base tools-off and trained tools-off probes):

| soft-Brier (ECE) | tools ON | tools OFF |
|---|---|---|
| trained v2 | 0.211 (0.119) | 0.250 (0.127) |
| base | 0.215 (0.170) | 0.254 (0.185) |

- **Finding 5 (accuracy is scaffold-bound):** the agentic harness is worth
  ~0.04 Brier to either model; the weight updates are worth ~0.004 (not
  significant). The hypothesized Turtel-shaped standalone-Brier gain did not
  materialize: even with retrieval folded into the RL loop, accuracy gains
  come from the scaffold, not the weights — extending Mantic's
  scaffold-dominates-weights decomposition to the trained-retrieval setting.
- **Finding 6 (training buys calibration):** ECE falls ~30% in both columns
  (0.170→0.119 tools-on; 0.185→0.127 tools-off). The calibration gain
  survives complete removal of the crowd anchor — internalized, not borrowed.
  Together with the behavioral transformations (coverage 64→100%, 3×
  crowd-independence, turn efficiency), this is what outcome-based RLVR
  actually teaches at this scale.

Revised thesis: *in-the-loop retrieval RLVR primarily teaches calibration and
task behavior; forecasting accuracy remains scaffold-bound and crowd-bounded.*

### The training curve, difficulty-adjusted

The raw v2 reward curve looks like noise (0.60–0.88, no visible trend) —
because under single-epoch training every batch is 64 never-seen questions and
per-batch reward is dominated by question difficulty: crowd-attainable reward
(1 − batch crowd Brier, logged by the env each step) itself swings 0.66–0.94,
and raw policy reward correlates with it at r = +0.58. A smoothly rising raw
curve in this regime would indicate re-fitting repeated data, not learning.

Subtracting crowd-attainable reward per step recovers the signal
(`figures/v2_reward_difficulty_adjusted.png`, data
`v2_run_platform_metrics.json`): the gap to the crowd narrows from ~0.145
(first 5 steps) to ~0.059 (last 5), trend +0.0037/step, t = 4.5. Roughly half
the closure is the submit-rate rise (0.59→1.00 by step 15, ending 0.55-floor
penalties); after saturation the gap continues narrowing (~0.076→0.052, trend
+0.0021/step, t = 1.5 — suggestive, not significant, consistent with the flat
held-out Brier result).

## Experiment 4 — frontier panel through the same harness

Neither reference paper tested frontier models *agentically* (both fed them
frozen research context). We ran Claude Sonnet 4.5, Gemini 3.1 Pro and Gemini
3.6 Flash through the identical loop, tools and leak filter as our own models.

| policy | soft-Brier | ECE |
|---|---|---|
| Sonnet 4.5 | 0.191 | 0.063 |
| Gemini 3.1 Pro | 0.197 | 0.062 |
| Gemini 3.6 Flash | 0.196 | 0.085 |
| trained 35B | 0.211 | 0.119 |
| base 35B | 0.215 | 0.170 |
| crowd | 0.189 | — |

- **Finding 7 (two axes):** accuracy is scaffold-bound — five policies from 3B
  active params to frontier scale sit in one 0.19–0.22 band with overlapping
  CIs and none beats the market — while **calibration tracks capability
  monotonically** (0.062 → 0.170 across the ladder). The harness equalizes
  *what you can find out*; the weights determine *how honestly you report it*.
- Our RL run advances the open 35B roughly a third of the calibration gap
  toward frontier models at ~1/10th inference cost, and closes the coverage gap
  entirely (64%→100% submit, matching or exceeding every frontier row).
- **Finding 8 (matched-subset, controlling for self-selection):** on the 169
  questions both base and trained answered, paired Brier difference is
  −0.001 ± 0.020 — statistically identical. Training's headline effect is
  therefore coverage + calibration, with resolution flat to within ±0.01
  (n=265 can only detect effects ≥0.02; ~1,400 questions would be needed for
  0.01).

## How this compares to the genre's publishable claims

Turtel: large Brier gain over a weak 14B base + parity with o1; no
market-beating. Mantic: large Metaculus-score gain over a weak gpt-oss base +
marginal frontier parity; no market-beating. Our base-with-tools is already
near-crowd, compressing the visible headroom — the genre-shaped claims
therefore live in two remaining evals:

1. **Tools-off** (crowd price hidden): base scored ~0.26 in this condition
   (v3 eval); if trained v2 holds near 0.21, that is the Turtel-shaped
   "large gain over base" claim, measured cleanly.
2. **Frontier panel through the same agentic harness** (Bedrock): the
   Mantic-shaped parity claim — and novel, since neither paper tested
   frontier models agentically.

## Costs (approximate, whole project)

~$120 Prime compute (3 training runs, pilots, probes, evals), ~6k Firecrawl
credits, Bedrock leak-filtering on AWS credits, ~$2 Azure (SearXNG VM +
webhook). Search for v2 was free (self-hosted SearXNG); its cache doubles as
a publishable corpus of all retrieved context.
