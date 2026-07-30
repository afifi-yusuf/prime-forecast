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
