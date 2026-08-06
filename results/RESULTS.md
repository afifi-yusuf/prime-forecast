# prime-forecast: results and findings

Narrative version: `research-arc.md`. Reward evolution: `../docs/reward-design.md`.
Paper framing: `../docs/paper-framing.md`. Eval/figure plan: `../docs/paper-eval-plan.md`.

## Scope statement (read first)

A trace-level audit (2026-08-01) found that **93–100% of web searches returned
zero results across every headline evaluation and both the v2 and v3 training
runs**: the self-hosted SearXNG backend's upstream engines had IP-banned the
VM, while SearXNG returned HTTP 200 with empty result lists — silent to every
health check. Every result below is therefore measured in a **reduced harness:
parametric knowledge + market tools (when on) + structured-data tools
(Wikipedia, yfinance, FRED, Polymarket data) — but no live web search.**
All comparisons are internally valid (every policy faced the same condition);
claims about "agentic web research" are NOT supported by these data. A
working, ban-proof search stack (AgentCore API + fallback chain, verified
end-to-end 2026-08-02) exists for the planned v4 run — retrieval-in-the-loop
training with functioning search — pending compute credits. v3 then serves as
v4's exactly-matched web-search-off control.

## Findings index

1. **Anchoring** (v1): the policy's dominant strategy was crowd-copying —
   median |p−crowd| = 0.028, camped just outside the 0.02 penalty cliff.
2. **Serving gap** (v1): identical weights, 94% vs 48% submit across serving
   stacks → all headline evals run through the training platform (webhook
   methodology).
3. **Boundary relocation** (v2): replacing the cliff with a ramp (ε=0.08)
   moved the camp to exactly 0.080. Shaping relocates anchors, never removes
   them.
4. **Coverage without accuracy loss** (v2): submit 64%→100% on never-seen
   questions with accuracy held flat.
5. **Accuracy is scaffold-bound**: the market anchor is worth ~0.04–0.075
   Brier to every policy; two training runs moved resolution ~0.004 (n.s.).
6. **Training buys calibration**: ECE falls 30–40% in every train/eval pair
   (v2: 0.099→0.065; v3: 0.170→0.103; market-off: 0.185→0.127) — consistent
   direction across all evals (per-cell CIs are wide; the claim is the
   pattern, not one cell).
7. **Two axes** (frontier panel, as corrected — see Corrections C1): accuracy
   converges to one crowd-bounded band (0.186–0.215, nobody beats 0.189)
   across ~100× model scale; calibration is TRAINABLE, not scale-bound —
   with market tools the trained 35B (0.065) sits at Sonnet level (0.063);
   without them the trained models (0.103/0.127) top every frontier point
   estimate. Untrained models are the calibration outliers, not small ones.
8. **Matched-subset null**: on questions both base and trained answered,
   paired ΔBrier = −0.001 ± 0.020 — resolution flat within ±0.02 at n=265.
9. **Anchor decomposition**: paired anchor-worth (same model, same questions,
   market tools on/off) — Gemini Pro +0.075, Flash +0.063, Sonnet +0.058,
   both 35Bs +0.039, all p<0.001. Frontier models lean on the crowd MORE
   than small models; "small trained model beats frontier" reproduces on
   demand by toggling who sees the market.
10a. **Search de-herds but does not pay** (v4, pre-registered 4.5/6): with
    WORKING search (agentcore, leak-fixed), training dissolved the base-rate
    anchor (median 0.35→0.40, extremes 2%→12%) and produced search economy
    (3.4→2.25/rollout) and evidence-based reasoning — but the endpoint
    (0.2545 [.234,.275]) does not beat the web-search-off control v3 (0.2446),
    and search-worth ≈ 0 ± 0.02 for base and trained alike. The second
    information channel measured; only the crowd anchor ever paid.
10. **The anchor ladder** (v3, pre-registered): with no market tools at all,
    the trained policy parked its median prediction at 0.35 — the dataset
    YES base rate is 0.355 — with extreme predictions collapsing 10%→2%.
    Three runs, three anchors (crowd → penalty boundary → base rate):
    outcome-based RL buys calibration by seeking the cheapest statistical
    regularity available; resolution moves only as anchors are exhausted
    (v3 paired gain +0.016 vs v2's −0.001, still n.s. at n=265).

## Headline tables

All on the v4 test split (n=265 resolved Polymarket questions, resolutions
2026-07-01+, strictly after all training data; crowd soft-Brier 0.189).
Soft-Brier = 0.5-imputed non-answers (≡ Turtel's 0.25-penalty convention);
ECE = 10 equal-mass bins; our models platform-served via webhook capture;
frontier models via LiteLLM→Bedrock/Vertex in the identical harness.

### Frontier panel (market tools ON)

| policy | soft-Brier | ECE | submit | trading P&L/bet | med \|p−crowd\| |
|---|---|---|---|---|---|
| Claude Opus 4.5 | **0.186** [.161,.211] | **0.053** | 98% | **+$0.042** | 0.035 |
| Claude Sonnet 4.5 | 0.191 [.164,.217] | 0.063 | 100% | +$0.010 | 0.045 |
| Gemini 3.6 Flash | 0.196 [.169,.222] | 0.085 | 99% | −$0.015 | 0.040 |
| Gemini 3.1 Pro | 0.197 [.169,.224] | 0.062 | 94% | +$0.008 | 0.042 |
| trained 35B (v2) | 0.211 [.188,.233] | 0.065 | 100% | — | 0.080 |
| base 35B | 0.215 [.194,.235] | 0.099 | 64% | — | 0.050 |
| **crowd (market)** | **0.189** | 0.058 | — | — | — |

Opus nominally beats the crowd; paired (crowd − Opus) = +0.003 ± 0.008
(t=0.77, n.s.). No policy separates from the market at n=265.

### Market-anchor × training quadrant

"Market tools OFF" removes only the Polymarket price/history/search tools;
all research tools remain. The ablated variable is the crowd anchor.

| soft-Brier (ECE) | market ON | market OFF |
|---|---|---|
| trained v2 (step-22†) | **0.211** (0.065) | 0.250 (0.127) |
| base | 0.215 (0.099) | 0.254 (0.185) |
| Sonnet 4.5 | 0.191 (0.063) | 0.246 (0.130) |
| Gemini 3.6 Flash | 0.196 (0.085) | 0.259 (0.157) |
| Gemini 3.1 Pro | 0.197 (0.062) | 0.272 (0.219, submit 89%) |

† market-ON v2 number is the true final step-33 policy (in-run eval);
market-OFF cells use the step-22 checkpoint (step-33 artifact lost to a
recurring platform upload bug — conservative direction).
ECE convention PINNED paper-wide (2026-08-04): submitted-only, 10
equal-mass bins (matches both reference papers; submit rate reported
separately). See Corrections C2.

### Anchor-worth (paired within-model, same questions; all p<0.001)

| policy | anchor-worth [95% CI] |
|---|---|
| Gemini 3.1 Pro | +0.075 [+0.040, +0.110] |
| Gemini 3.6 Flash | +0.063 [+0.033, +0.094] |
| Sonnet 4.5 | +0.058 [+0.031, +0.085] |
| trained 35B | +0.039 [+0.016, +0.063] |
| base 35B | +0.039 [+0.016, +0.063] |

Without the market, frontier accuracy falls into/below the 35B band and
Gemini Pro's calibration (0.220) drops below the untrained 35B's. The
genre's "small trained model beats frontier" headline is manufactured by
scaffold asymmetry: our trained-with-market 0.211 "beats" Sonnet-without
0.246.

### v3 — no-market training (run `slgbosbzsn4dzt2706bux0qx`)

Identical to v2 except include_market_tools=false in training AND eval,
pure Brier reward. Predictions pre-registered before any training metric
existed (git-timestamped, `research-arc.md` Experiment 6): 4 of 5 confirmed.

| | base (step 0) | trained v3 (step 33) |
|---|---|---|
| soft-Brier | 0.261 [.235,.287] | **0.245 [.221,.268]** |
| ECE (submitted-only) | 0.170 | **0.103** |
| submit | 74% | 98% |
| pred p10/p50/p90 | 0.10/0.35/0.63 | 0.15/**0.35**/0.55 |
| extreme preds (<.1 or >.9) | 10% | **2%** |

Paired (base − trained) = **+0.016 ± 0.025 (t=1.25, n.s.)** — 4× the v2
effect, below significance at n=265. Median prediction = dataset base rate
(0.355): base-rate herding as pre-registered (Finding 10). Searches/rollout
flat (~4→3.9): the policy herded rather than searched — interpretation
scoped by the web-search-off regime (see search audit) (searching WAS futile; v4 separates
RL-nature from starvation). In the matched no-market column, trained v3
(0.245) is the best point estimate alongside Sonnet (0.246), ahead of v2
(0.250), base (0.254), Flash (0.259), Pro (0.272).

### Training dynamics (v2; figure `figures/v2_reward_difficulty_adjusted.png`)

Raw single-epoch reward curves are difficulty noise: batch crowd-attainable
reward (1 − batch crowd Brier, logged per step) spans 0.66–0.94 and raw
reward tracks it at r=+0.58. Difficulty-adjusted, the gap to the crowd
narrows 0.145→0.059 (+0.0037/step, t=4.5); roughly half is the submit-rate
rise; post-saturation trend +0.0021/step (t=1.5, suggestive). Neither
reference paper difficulty-adjusts its curves.

## Frontier search-worth (web-search-on arm, all cells complete 2026-08-06)

Same harness/env 0.1.18 as v4; AgentCore search; leak filter active.
Search-worth = working-search minus web-search-off soft-Brier (positive =
search HURT). Pro market-on is an n=231 partial (overnight API stall,
documented).

| policy | market-ON: off-arm → on-arm | market-OFF: off-arm → on-arm |
|---|---|---|
| Opus 4.5 | 0.186 → 0.208 (+0.022) | (not measured) → 0.256 |
| Sonnet 4.5 | 0.191 → 0.217 (+0.026) | 0.246 → 0.273 (+0.027) |
| Gemini 3.6 Flash | 0.196 → 0.189 (−0.007) | 0.259 → 0.286 (+0.027) |
| Gemini 3.1 Pro | 0.197 → 0.207* (+0.010) | 0.272 → 0.278 (+0.006) |
| **trained 35B** (v3→v4) | — | 0.245 → 0.254 (+0.009 n.s.) |

**Finding 12 (search degrades frontier forecasters):** 7 of 8 frontier
search-worth cells are NEGATIVE-value (search hurt); the sole exception
(Flash market-on, −0.007) kept the tightest crowd anchor of any policy
(med |p−crowd| 0.040) — search helps only when subordinated to the anchor.
Mechanism: retrieved public news is stale relative to an efficient price;
evidence-driven deviation from the crowd loses (Sonnet P&L +$0.010→−$0.018).
The search-trained 35B is the only policy that does not degrade, and tops
the fair-fight column (search-on/no-market: 0.254 vs Opus 0.256, Sonnet
0.273, Pro 0.278, Flash 0.286). Training with retrieval in the loop taught
evidence discipline that frontier scale does not confer zero-shot. Raw:
`*_searchon_*_eval.jsonl`.

## Search-health audit (the scope-defining measurement)

Zero-result rate of web_search by eval (strict per-query trace parse):

| eval | searches | zero-result |
|---|---|---|
| Gemini Pro (both conditions) | 776/784 | **100%** |
| Gemini Flash (both) | 722/758 | **99–100%** |
| Opus 4.5 | 772 | **96%** |
| Sonnet 4.5 (both) | 739/780 | **93–96%** |
| v2 training (sampled steps) | — | **100%** |
| v3 training (sampled steps) | — | **100%** |

Root cause: upstream engines (Google/DDG/Brave/Startpage) IP-banned the
scraping VM under training query volume; SearXNG returns HTTP 200 + empty
list when all engines are suspended. Detection required reading rollout
traces; status-code health checks passed throughout. Remediations now in
place: result-count health checks, never-cache/serve empty results, backend
fallback chain, licensed-API workhorse (Bedrock AgentCore WebSearch,
verified end-to-end incl. leak filter), errors degrade to empty results
(an erroring search tool cost 97%→54% submit in discarded probes by burning
agent turns on retries).

## Corrections (append-only)

- **C1 (2026-08-04).** The frontier-panel and quadrant tables previously
  reported trained-v2 ECE 0.119 and base ECE 0.170. Those were v1-era
  values (step-30 eval, n=160 v3 split) carried over erroneously when the
  v2 panel table was assembled. Correct v2 values (submitted-only, v4
  split): trained 0.065, base 0.099. Discovered by independent
  re-derivation while generating figures. Consequence: Finding 7's
  "calibration monotone in capability" claim was an artifact of the wrong
  numbers and is retracted; the corrected claim is "calibration is
  trainable" (see Findings 6–7). Finding 6 strengthens.
- **C2 (2026-08-04).** ECE convention pinned to submitted-only paper-wide.
  Previously v3 numbers were quoted 0.5-imputed (0.160/0.110); restated
  submitted-only as 0.170/0.103. No conclusion changes.

## Discarded data (recorded for completeness)

- Four re-eval probe runs (2026-08-01) ran on degraded API-fallback search
  after `prime train stop` commands failed silently; agent behavior was
  poisoned (submit 54–88%, search-retry turn burn). ~$25. Not used.
- An early "Gemini Flash" panel row was actually Sonnet 4.5 — `prime eval`'s
  config file silently overrides its `-m` flag. Retracted, re-run with
  per-model configs; identity verified via output dirs + proxy logs.

## Validity notes

- Frontier rows ran a 7-of-13 core toolset with flattened schemas (Bedrock/
  Vertex validator constraints); our models ran the full 13 tools.
- v1/v2/v3 final checkpoints (steps 45/33/33) all lost to a recurring
  platform upload bug (support ticket filed); step-22 checkpoints used
  where warm-starts were required — conservative for trained rows.
- Local serving understates trained policies (Finding 2); all trained-model
  numbers here are platform-served.
- Trained models are search-naive (trained under dead search): any future
  search-on evaluation of v2/v3 measures scaffold transfer, not trained
  retrieval. v4 (train WITH search) is the designed fix.
- **Selection-criteria audit (2026-08-04, EQP D28-style):** three dataset
  design elements are hindsight-conditioned (not computable at cutoff t):
  (a) resolved-only membership — voided/unresolved markets excluded using
  post-cutoff information (partially mitigated: candidates are fetched by
  scheduled end-date windows, which ARE t-computable); (b) the lifetime-
  volume filter (>=5000) and volume-ranked category caps use total volume
  including post-cutoff trading; (c) cutoff placement is sampled relative
  to the realized resolution date, not the scheduled close (Turtel uses
  scheduled close). None of these leaks outcome direction into context,
  and all reported comparisons are internal to the same dataset — but the
  question distribution itself is defined with post-t information, which
  scopes external validity. (a) is genre-universal; (c) is fixable in any
  future build by sampling cutoffs against scheduled close.

## Archive: v1 run (superseded; grounds Findings 1–2)

45-step GRPO, 9 epochs over 1,280 questions, binary cliff penalty
(0.20 @ 0.02), ≤2 searches (Firecrawl era — search then functional).
Step-30 platform eval (v3 test split, n=160, `test_split_v3.jsonl`):
soft-Brier 0.216 [0.186,0.247] vs base 0.236 vs crowd 0.191; submit 94% vs
31%; ECE 0.119; median |p−crowd| 0.028 with 32% within 0.02 (Finding 1);
trading −$0.08/bet. Serving gap: same merged weights 48% submit locally vs
94% platform (Finding 2; merge verified tensor-exact).

## Provenance / costs

- **Leak safety**: 5 layers — domain blocklist, provider date filters,
  heuristics, Claude Haiku 4.5 KEEP/DROP judge (only model with zero
  leak-direction errors in a 6-model benchmark; Qwen3-32B validated
  fallback), cutoff-clamped tools. Search cache doubles as an auditable
  corpus.
- **Raw data** (this directory): `v2_run_webhook.jsonl.gz`,
  `v3_run_webhook.jsonl.gz`, `v2_run_platform_metrics.json`,
  `v3_run_platform_metrics.json`, `*_harness_eval.jsonl`,
  `*_notools_eval.jsonl`, `platform_eval_step30_test.jsonl`, pinned splits
  `test_split_v3.jsonl` / `test_split_v4.jsonl`.
- **Costs**: ~$145 Prime (3 training runs + probes + evals; incl. ~$25
  discarded probes), ~6k Firecrawl credits, Bedrock filtering + Anthropic
  panel on AWS credits, Gemini panel via Vertex on GCP credits, SearXNG +
  webhook VM ~$0.08/h Azure. v2 run $41.47; v3 run ~$20.
