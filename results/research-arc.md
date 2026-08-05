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

## Experiment 3 — the market-anchor × training quadrant (standalone skill)

Both remaining cells measured platform-served on the same 265 test questions
(base tools-off and trained tools-off probes):

| soft-Brier (ECE) | market tools ON | market tools OFF |
|---|---|---|
| trained v2 | 0.211 (0.119) | 0.250 (0.127) |
| base | 0.215 (0.170) | 0.254 (0.185) |

- **Finding 5 (accuracy is scaffold-bound):** the market anchor is worth
  ~0.04 Brier to either model (research tools present in both conditions); the weight updates are worth ~0.004 (not
  significant). The hypothesized Turtel-shaped standalone-Brier gain did not
  materialize: even with retrieval folded into the RL loop, accuracy gains
  come from the scaffold, not the weights — extending Mantic's
  scaffold-dominates-weights decomposition to the trained-retrieval setting.
- **Finding 6 (training buys calibration):** ECE falls 30–40% in every
  train/eval pair (market-on 0.099→0.065; market-off 0.185→0.127; v3
  0.170→0.103; all submitted-only — see RESULTS.md Corrections C1/C2). The
  gain survives complete removal of the crowd anchor — internalized, not
  borrowed.
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
| Opus 4.5 | 0.186 | 0.053 |
| Sonnet 4.5 | 0.191 | 0.063 |
| Gemini 3.1 Pro | 0.197 | 0.062 |
| Gemini 3.6 Flash | 0.196 | 0.085 |
| trained 35B | 0.211 | 0.119 |
| base 35B | 0.215 | 0.170 |
| crowd | 0.189 | — |

- **Finding 7 (two axes, corrected 2026-08-04):** accuracy is scaffold-bound
  — six policies from 3B-active to frontier sit in one 0.186–0.215 band and
  none beats the market — while **calibration is trainable, not
  scale-bound**: with market tools the trained 35B (ECE 0.065) matches
  Sonnet (0.063); the poorly calibrated policies are the UNTRAINED ones
  (base 0.099) and frontier models denied the anchor (0.130–0.219). An
  earlier "monotone in capability" version of this finding rested on
  carried-over v1 numbers and is retracted (RESULTS.md Corrections C1).
- Our RL run advances the open 35B roughly a third of the calibration gap
  toward frontier models at ~1/10th inference cost, and closes the coverage gap
  entirely (64%→100% submit, matching or exceeding every frontier row).
- **Finding 8 (matched-subset, controlling for self-selection):** on the 169
  questions both base and trained answered, paired Brier difference is
  −0.001 ± 0.020 — statistically identical. Training's headline effect is
  therefore coverage + calibration, with resolution flat to within ±0.01
  (n=265 can only detect effects ≥0.02; ~1,400 questions would be needed for
  0.01).

## Experiment 5 — the market anchor decomposition (Sonnet tools-off)

One more cell: Sonnet 4.5 in the identical no-market condition as our
quadrant. Result: 0.246 [.216,.276], ECE 0.130 (vs 0.191/0.063 with market
tools).

- **Finding 9a (generalized, 3 frontier models, 2 vendors):** frontier
  in-harness accuracy is substantially crowd-reading. Paired anchor-worth
  (tools-on vs tools-off, same questions, all p<0.001): Gemini Pro +0.075,
  Flash +0.063, Sonnet +0.058, vs +0.039 for both 35Bs — frontier models
  lean MORE on the crowd than small models, and their advantage largely IS
  better anchor exploitation. Without the market, frontier accuracy falls
  into/below the 35B band and Gemini Pro's calibration (ECE 0.220) drops
  below the untrained 35B's.
- **Finding 9b:** "small trained model beats frontier" (Turtel's headline,
  Mantic's Fig. 1) reproduces on demand from scaffold asymmetry: our trained
  35B with market tools (0.211) beats Sonnet without (0.246). Both reference
  papers' frontier baselines were denied the scaffold their trained models
  used at full strength.
- **Finding 9c (bounded parity):** in the matched no-market condition no
  pairwise accuracy difference among base/trained/Sonnet is significant at
  n=265 — so "matches frontier accuracy" is true but vacuous (base matches
  too); the honest statement is that evidence-based forecasting accuracy
  does not separate any policy at this n. Calibration point estimates show
  trained (0.127) at Sonnet parity (0.130), base worse — but per-cell ECE
  CIs are ±0.07, so the calibration claim rests on the consistent
  cross-eval direction (trained < base in all four evals), not one cell.
  ECE is imputation-sensitive when submit<100% (base: 0.143 imputed vs
  0.185 submitted-only) — convention must be pinned paper-wide.

## Experiment 6 — v3: no-market training (run slgbosbzsn4dzt2706bux0qx)

Causal test of the anchor-suppression hypothesis: identical to v2 except
include_market_tools=false in training AND eval, pure Brier reward (no
crowd-copy penalty — nothing visible to copy). Launched 2026-07-31.

**Pre-registered predictions (written before any v3 training metrics):**

1. Submit rate saturates ~100% by ~step 15 (near-certain; both prior runs).
2. Calibration improves, likely MORE than v2 (~85% confidence): no anchor
   to borrow, so all moderation pressure lands on the weights.
3. Resolution: directionally positive but small — final soft-Brier
   ~0.240–0.245 from base 0.254 (~0.005–0.010 gain; Turtel bought 0.016
   with 5x our data), not individually significant at n=265, but ahead of
   Flash (0.259) / Pro (0.272) point estimates in the matched column.
4. **Base-rate herding wildcard:** with the crowd gone, the dataset YES
   base rate is the next-nearest attractor; watch for predicted_prob
   distribution collapsing toward it (good ECE, no resolution). If
   observed: anchoring is a property of outcome-based RL, not of the
   market tool ("remove one anchor and the policy finds the next").
5. Web searches per rollout increase (search is now the only information
   channel).

**Results (platform-served, n=265, matched no-market condition,
`v3_run_webhook.jsonl.gz` / `v3_run_platform_metrics.json`):**

| | base (step 0) | trained v3 (step 33) |
|---|---|---|
| soft-Brier | 0.261 [0.235, 0.287] | **0.245 [0.221, 0.268]** |
| ECE (submitted-only) | 0.170 | **0.103** |
| submit | 74% | 98% |
| pred p10/p50/p90 | 0.10 / 0.35 / 0.63 | 0.15 / **0.35** / 0.55 |
| extreme preds (<0.1 or >0.9) | 10% | **2%** |

Paired (base − trained): **+0.016 ± 0.025 (t=1.25, n.s.)** — 4× the v2
market-on paired effect (−0.001) but below significance at n=265.

**Predictions scored:** (1) submit saturation ✓ (74→98%); (2) calibration
gain ✓ (0.160→0.110, similar to v2); (3) resolution "directionally positive,
~0.240–0.245, not significant" ✓ — landed 0.245 exactly; (4) **base-rate
herding ✓ — the headline**: trained median prediction 0.35 = test-split YES
base rate 0.355 exactly; extremes collapse 10%→2%; spread narrows to
[0.15, 0.55]. (5) searches up ✗ — flat (4.07→3.87/rollout).

- **Finding 10 (the anchor ladder):** remove the crowd anchor and the
  policy migrates to the next-nearest attractor — the dataset base rate.
  Anchoring is a property of outcome-based RL, not of the market tool.
  Combined with v1/v2 (crowd-camping, boundary relocation): GRPO teaches
  calibration by *seeking anchors*; resolution gains appear only to the
  extent anchors are exhausted (v3's +0.016 vs v2's −0.001 — removing one
  anchor unlocked a real-but-small resolution gain, consistent with the
  shortcut-suppression hypothesis at reduced strength).
- **Cross-model placement:** trained v3 (0.245) is the best-in-column point
  estimate alongside Sonnet 4.5 (0.246), ahead of v2 (0.250), base (0.254),
  Flash (0.259), Pro (0.272) — an open 35B trained for ~$40 matching the
  best frontier point estimate in the genre-comparable (no-market) agentic
  condition.

## Experiment 7 — v4: no-market training with WORKING search

Identical to v3 (no market tools, pure Brier, single epoch over 2,113
questions) except web search functions: agentcore-only backend (licensed
AWS API, uniform quality across all 33 steps), verified end-to-end through
the leak filter at launch. v3 is the exactly-matched dead-search control.
In-run evals at steps 0/33 in BOTH market conditions (named envs).

**Pre-registered predictions (committed before launch, 2026-08-04):**

1. Submit saturates ~100% by ~step 15 (near-certain; three prior runs).
2. Calibration improves again (~85% confidence): trained ECE lands
   0.09–0.13 submitted-only (from base ~0.17).
3. **Search-utilization:** searches/rollout and lookup_url follow-ups rise
   vs v3 (~65% confidence) — results now contain content worth reading.
4. **Herding (the headline question):** partial de-herding (~55%
   confidence): prediction spread widens vs v3 (extremes >2%, p10–p90
   wider than [0.15, 0.55]). Full de-herding (median leaves the base
   rate) NOT expected at 2k questions.
5. **Resolution:** paired base−trained gain exceeds v3's +0.016; final
   trained soft-Brier 0.230–0.245; significance at n=265 still unlikely
   (~30% chance t>2). Base-with-search (step 0) improves on v3's base
   (0.245–0.255 vs 0.261) — search helps even untrained.
6. **Transfer cell (market-on eval of the no-market-trained policy):** the
   policy largely ignores tools it never learned — its anchor-worth ≈ 0
   (|market-on − market-off| < 0.01), in sharp contrast to +0.039 for
   v2/base. Falsifiable and diagnostic: a large anchor-worth here would
   mean anchor exploitation is zero-shot, not learned.

Results: PENDING.

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
