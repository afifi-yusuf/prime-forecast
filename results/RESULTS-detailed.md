# prime-forecast: results and findings

Narrative: `research-arc.md` · Reward evolution: `../docs/reward-design.md` ·
Framing: `../docs/paper-framing.md` · Run map: `../docs/run-registry.md`

## Abstract

**Agentic RLVR lifts an open base model to frontier parity at
evidence-based forecasting**: after ~$300 of training, our 35B-A3B (3B
active) holds the best point estimate in both no-market evaluation
columns — statistically tied with Claude Opus 4.5 and Sonnet 4.5 — with
frontier-level calibration and a robustness to retrieval that no frontier
model exhibits, at roughly 1/100th the inference cost. Decomposing where
forecasting performance comes from explains the result: the market's own
price dominates every policy's score (withholding it costs 0.039–0.075
Brier, paired p<0.001, most for frontier models; nobody beats the crowd);
web search never pays (1 of 13 ablation contrasts positive — retrieved
news is stale relative to an efficient price); and outcome-based RL buys
calibration and behavior through anchor-seeking (crowd → penalty boundary
→ base rate, pre-registered), with retrieval-in-the-loop training
replacing anchor-camping by evidence-weighing. Half these findings were
invisible to aggregate metrics and surfaced only through trace-level
audits, pre-registration, and append-only corrections — practices we
argue are mandatory for agentic RL evaluation.

## Scope statement (read first)

The project has two evaluation arms. The **search-off arm** originated as an
undetected infrastructure outage: a trace-level audit (2026-08-01) found
93–100% of web searches returned empty results across all early headline
evaluations and the v2/v3 training runs (upstream engines had IP-banned the
scraping VM; HTTP status stayed 200 — silent to every health check). Those
measurements are internally valid (every policy faced identical conditions:
parametric knowledge + structured-data tools + market tools when granted)
and are retained, fully disclosed, as the search-off arm. The **search-on
arm** (env 0.1.18+, licensed-API retrieval, verified per-trace) was then
measured for all policies, completing a two-arm design. Trained models in
the search-off arm are search-naive by construction; v4/v5 trained with
retrieval in the loop.

## The five training runs (what each version is)

All runs fine-tune the same base model (Qwen3.5-35B-A3B, 3B active
params) with LoRA GRPO on Prime Intellect hosted training. v2–v5 share
one recipe — a single epoch over the same 2,113-question train split
(33 steps × batch 64 × 8 rollouts/question), max 10 turns, ≤3 web
searches — and differ **only** in which information channels the agent
gets during training and in the anti-copy reward term (see "Reward
functions" below). The versions are the cells of a 2×2 design: market
tools (can the agent read the Polymarket price?) × search (does web
retrieval actually work?).

| ver | run id | market tools | search | why it exists |
|---|---|---|---|---|
| v1 | `rqn28a9d` | on | Firecrawl era | pilot: 9 epochs × 1,280 questions, cliff penalty. Discovered crowd-anchoring + the serving gap. Archived — superseded dataset and recipe; grounds Findings 1–2 only |
| v2 | `mjkvxreh` | **on** | off (outage) | main market-on run; ramp penalty. Best trained market-on endpoint (0.211). Grounds boundary-relocation (Finding 3) |
| v3 | `slgbosbz` | **off** | off (outage) | control: remove the crowd anchor entirely. Shows RL migrates to the base rate (anchor ladder) |
| v4 | `r64dayfb`+`basziye6` | **off** | **on** (verified) | first run with working retrieval in the training loop (env 0.1.18 leak-fix); tests whether search substitutes for the anchor. Continuation = credit-exhaustion warm-start, same recipe |
| v5 | `zvxexo35` | **on** | **on** | completes the 2×2: both channels live. Tests whether search adds anything on top of the anchor (it didn't: 0.224 vs v2's 0.211) |

"Search off" always means the outage condition disclosed in the scope
statement above — tools present but returning empty — not tools removed.
The n=593 expansion runs (August 2026) are continuation re-runs of v4
and v5 from their step-22 checkpoints, evaluated in-run on the pinned
593-question union split; they produce sibling endpoints, not the
original weights (step-33 checkpoints were unrecoverable platform-side).

## Findings index

1. **Anchoring** (v1): dominant learned strategy is crowd-copying — median
   |p−crowd| 0.028, camped outside the 0.02 penalty cliff.
2. **Serving gap** (v1): identical weights, 94% vs 48% submit across
   serving stacks → all headline evals platform-served (webhook method).
3. **Boundary relocation** (v2, pre-registered): replace cliff with ramp
   (ε=0.08) → camp moves to exactly 0.080. Shaping relocates anchors.
4. **Coverage without accuracy loss** (v2): submit 64→100%, accuracy flat.
5. **Accuracy is scaffold-bound**: the market anchor is worth 0.039–0.075
   Brier to every policy; training moved resolution ≤ +0.016 (n.s.).
6. **Training buys calibration**: ECE −30–40% in every train/eval pair
   (0.099→0.065; 0.185→0.127; 0.170→0.103); consistent direction, per-cell
   CIs wide. *n=593 revision:* does not replicate in the search-trained
   siblings — v5-sibling market-on ECE 0.098 ≈ untrained base 0.099;
   v4-sibling market-off 0.180 ≈ base 0.185. Scope the claim to the
   search-off runs (v2/v3) pending the base probe at n=593 (Finding 15).
7. **Two axes** (frontier panel, corrected C1): accuracy converges to one
   crowd-bounded band (0.186–0.215; crowd 0.189 unbeaten); calibration is
   TRAINABLE, not scale-bound — trained 35B ECE 0.065 ≈ Sonnet 0.063.
8. **Matched-subset null**: paired base-vs-trained resolution −0.001 ±
   0.020 — flat within ±0.02.
9. **Anchor decomposition** (paired, all p<0.001): anchor-worth Gemini Pro
   +0.075, Flash +0.063, Sonnet +0.058, both 35Bs +0.039. Frontier leans
   on the crowd hardest; "small model beats frontier" reproduces in either
   direction by toggling who sees the price.
10. **The anchor ladder** (v3, pre-registered 4/5): market hidden → policy
    herds to the dataset base rate (median 0.35 vs rate 0.355; extremes
    10→2%). Three runs, three anchors: crowd → boundary → base rate.
11. **Search de-herds but does not pay** (v4, pre-registered 4.5/6):
    retrieval-in-loop training dissolves the base-rate anchor (median
    0.40, extremes 12%, search economy 3.4→2.25/rollout) yet the endpoint
    (0.254) does not beat the search-off control v3 (0.245).
12. **Search degrades frontier forecasters** (8-cell campaign): 7 of 8
    frontier search-worth cells negative-value (+0.006 to +0.027); the
    sole exception (Flash market-on, −0.007) kept the tightest crowd
    anchor measured. Paired stats: individually significant
    for Sonnet (+0.025, t=2.4) and Opus (+0.022, t=2.4) market-on;
    12/13 contrasts same direction (sign test p≈0.002). Mechanism:
    retrieved news is stale relative to an efficient price; informed
    deviation loses (Sonnet P&L +$0.010 → −$0.018/bet).
13. **Search never pays, trained or frontier** (v5 completes the 2×2;
    pre-registered 2/6): thirteen search contrasts, one improvement,
    twelve harms/nulls. v5 (market+search, near-pure Brier — residual
    0.02-radius anti-copy ramp, C3) neither camped
    (median |p−crowd| 0.125) nor herded: it learned evidence-weighing and
    adaptive channel arbitration (searches 1.53 with anchor, 2.35
    without) — the richest behavior and the worst trained market-on Brier
    (0.224 vs search-off v2's 0.211). Behavior and score decoupled.
    *n=593 revision:* the v2-vs-v5 ordering is retired (gap 0.004 on
    matched questions, inside sibling noise — Finding 15); "search hurts
    the trained endpoint" softens to "search adds nothing."
14. **Trained anchor-worth is real and monotone in anchor exposure**
    (n=593 expansion): paired off−on is +0.020 ± 0.018 (t=2.13) for the
    v4 sibling (never trained with market tools) and **+0.045 ± 0.019
    (t=4.61)** for the v5 sibling (trained with them) — versus +0.058 to
    +0.075 for frontier. The n=265 transfer null was underpowered; every
    policy measured profits from seeing the price, and training with the
    anchor deepens the dependence.
15. **Sibling variance and instrument nonstationarity** (n=593
    expansion): continuation re-runs from the same step-22 checkpoint
    drift ±0.015 soft-Brier and up to +0.05 ECE at the endpoint;
    platform "step 33" evals are actually served by a mixture of policy
    versions 31–33 (orchestrator overlaps eval with final training —
    logged warning, presumably true of all prior in-run endpoints).
    Market-on cells replicate across siblings (v5: 0.2198 vs 0.224);
    market-off cells shifted +0.015–0.020 in BOTH siblings on the same
    questions — consistent with retrieval nonstationarity (the
    search-dependent condition inherits the retrieval stack's
    week-to-week state; the anchored condition doesn't). Consequence:
    endpoint differences under ~0.015 must not be ordered.

## Grand summary (all cells)

**Training 2×2 (endpoint soft-Brier, matched eval condition):**

| trained with… | search OFF | search ON |
|---|---|---|
| market ON | v2 **0.211** | v5 **0.224** |
| market OFF | v3 **0.245** | v4 **0.254** |

Untrained base: 0.215 / ~0.258 (market-on), 0.254–0.261 / ~0.261–0.269
(market-off). Search never improved a trained endpoint.

**n=593 expansion (2026-08-07; continuation-sibling endpoints, in-run
dual-condition evals, condition-labeled webhook rows, env 0.1.20):**

| sibling endpoint | market-ON | market-OFF |
|---|---|---|
| v5-sibling (`lm45xhpr`) | **0.2198** (.098) | 0.2651 (.190) |
| v4-sibling (`gjhkz8xq`) | 0.2483 (.157) | 0.2679 (.180) |
| crowd (same 593) | 0.1948 | — |

Single-cell 95% CIs ±0.018–0.020. Submit rates 95.6–99.8%. Old-265 vs
new-328 subsets agree within 0.001–0.012 (no composition drift). Paired
anchor-worth: v4-sibling +0.0196 ± 0.0180 (t=2.13), v5-sibling +0.0453 ±
0.0192 (t=4.61). Uncertain-band (price 0.30–0.70, n=236): crowd 0.242,
v5-on 0.259, v4-on 0.274, v4-off 0.282, v5-off 0.283. Adaptive search
arbitration replicates in v5 (1.40 searches/rollout with anchor vs 2.25
without); the v5 sibling anchors tighter than the original (median
|p−crowd| 0.050 vs 0.125). Raw data:
`results/v{4,5}expand_webhook_raw.jsonl.gz` + platform metrics.

**Search-OFF arm** (soft-Brier, ECE submitted-only; each column ranked):

| market-ON | market-OFF |
|---|---|
| Opus 4.5 **0.186** (.053) | trained v3 **0.245** (.103) |
| crowd 0.189 (.058) | Sonnet 0.246 (.130) |
| Sonnet 0.191 (.063) | trained v2† 0.250 (.127) |
| Flash 0.196 (.085) | base 0.254 (.185) |
| Pro 0.197 (.062) | Flash 0.259 (.157) |
| trained v2 0.211 (.065) | Pro 0.272 (.219) |
| base 0.215 (.099) | |

**Search-ON arm** (each column ranked):

| market-ON | market-OFF |
|---|---|
| Flash **0.189** (.067) | trained v5 **0.252** |
| Pro 0.207* (.102) | trained v4 0.254 (.128) |
| Opus 0.208 (.082) | Opus 0.256 (.136) |
| Sonnet 0.217 (.100) | base ~0.261–0.269 |
| trained v5 0.224 | Sonnet 0.273 (.176) |
| trained v4 ~0.254 (transfer) | Pro 0.278 (.215) |
| base ~0.258 | Flash 0.286 (.205) |

† step-22 checkpoint · * n=231 partial (documented stall) · crowd = 0.189
on the same rows in every cell; no policy beats it. Point-estimate order;
statistically the columns resolve into tiers, not ranks (n=265).

**Paired effects (same questions, within-model):**

| effect | size | grade |
|---|---|---|
| anchor-worth, frontier | +0.058 to +0.075 | p<0.001 each |
| anchor-worth, 35Bs | +0.039 | p<0.001 |
| search-worth (13 contrasts) | −0.007 once; +0.006…+0.027 (or ~0) 12× | 12/13 direction |
| training → resolution | ~0.000 matched; ≤+0.016 | bound |
| training → calibration | −30–40% ECE every run | direction |
| anchor ladder | 0.028 → 0.080 → base rate → 0.125 | pre-registered ×3 |

CIs (stated once, not per cell): single-cell soft-Brier 95% CIs are
±0.019–0.034 at n=265; paired anchor-worth CIs are ±0.024–0.035 (all
excluding zero); the matched-subset resolution bound is ±0.020; per-run
CIs appear in the per-run sections' prose.

Table notation: the bold/plain leading number in each cell is soft-Brier
(0.5-imputed for non-submissions); the parenthesized number is
**submitted-only ECE** (10 equal-mass bins, convention C2). Cells without
parentheses are ones where the ECE was not computed/archived.

## Reward functions (per run)

Common core, every run: a rollout that submits a final probability `p` on
a question with outcome `y ∈ {0,1}` earns `r = 1 − (p − y)²`
(positive-shifted Brier, range 0–1). A rollout that never submits earns a
flat **0.55** — deliberately below the 0.75 floor guaranteed by always
answering 0.5, so finishing always dominates stalling. GRPO advantages
are group-relative per question, so reward terms constant within a
question's rollout group (e.g. the crowd's own Brier) cancel out of the
gradient; only within-group-varying terms — like each rollout's distance
to the price — shape learning.

| run | anti-copy term on top of `1−(p−y)²` | fired (of submitted train rollouts) |
|---|---|---|
| v1 | binary cliff: −0.20 flat if \|p−crowd\| ≤ 0.02 | (v1-era logs not archived) |
| v2 | graded ramp: −0.15·(1−d/0.08) for d = \|p−crowd\| < 0.08 | 51% (1,263/2,470) |
| v3 | residual default ramp: −0.20·(1−d/0.02) for d < 0.02 (C3) | 5.0% (128/2,539) |
| v4 | residual ramp, as v3 (C3) | 5.5% main (117/2,128); 4.6% continuation (62/1,338) |
| v5 | residual ramp, as v3 (C3) | 11.2% (318/2,840) |

The v3/v4/v5 residual term was **not** a design choice: those configs
were intended as pure Brier, but never overrode the environment's default
`crowd_copy_penalty = 0.20, eps = 0.02`, so a narrow anti-copy ramp
stayed active (correction C3). Incidence measured from training webhook
rows where `reward ≠ 1 − brier`. All evaluation tables in this document
are unaffected: reported soft-Brier/ECE are computed from the logged
`brier` field, never from `reward`.

## Per-run detail

### v1 (run `rqn28a9d`) — archive; grounds Findings 1–2

45 steps, 9 epochs × 1,280 questions, cliff penalty, Firecrawl-era search.
Step-30 platform eval (v3 split, n=160): soft-Brier 0.216 [.186,.247] vs
base 0.236, crowd 0.191; submit 94% vs 31%; median |p−crowd| 0.028, 32%
within 0.02; trading −$0.08/bet. Serving gap measured here (94%/48%).

### v2 (run `mjkvxreh`) — single epoch, ramp penalty, market-on

33×64×8 over 2,113 questions (dataset v4: uniform-in-lifetime cutoffs,
template ≤5 / event ≤4 caps). Endpoint (step-33 in-run eval, n=265):
0.211 [.188,.233] vs base 0.215 [.194,.235]; submit 64→100%; median
|p−crowd| 0.080 = ramp ε (Finding 3); within-0.02 mass 32→4%.
Difficulty-adjusted training curve: gap to crowd-attainable 0.145→0.059
(t=4.5), raw curve is batch-difficulty noise (r=+0.58 with attainable;
figure `figures/v2_reward_difficulty_adjusted.png`).

### v3 (run `slgbosbz`) — market-off control, search-off

Identical frame, no market tools, Brier + residual 0.02-radius anti-copy
ramp (C3; fired 5.0%). Endpoint: 0.245 [.221,.268]
vs base 0.261 [.235,.287] (paired +0.016 ± 0.025, t=1.25); ECE 0.170→
0.103; submit 74→98%; median prediction 0.35 = base rate 0.355; extremes
10→2% (Finding 10, pre-registered).

### v4 (runs `r64dayfb`+`basziye6`) — market-off, search-ON

First attempt (`xzmqo4zh`) aborted at step ~2: trace audit caught reward
leakage (AgentCore verbose dates unparsed → undated → hard drop bypassed;
a rollout retrieved a resolved macro figure for reward 0.998). Fixed in
env 0.1.18; replay of the leaked query post-fix returns zero results.
Main run credit-exhausted at step 29 (no step-29 checkpoint); continued
via step-22 warm-start to a true step-33 endpoint. Endpoint: 0.254
[.234,.275] pooled; ECE 0.128; submit 99.6%; median prediction 0.40,
extremes 12% (de-herding, Finding 11); search economy 3.4→2.25/rollout;
transfer cell (market-on, never trained): score-level anchor-worth ≈ 0
yet per-question |p_on−p_off| median 0.120 — zero-shot engagement without
profit.

### v5 (run `zvxexo35`) — market-on + search-ON, near-pure Brier (C3)

The final 2×2 cell. Endpoint: market-on 0.224, market-off 0.252; searches
1.53 (anchor available) vs 2.35 (not); market calls 1.71/rollout; median
|p−crowd| 0.125 — no camping, though the residual 0.02-radius ramp (C3)
still disincentivized exact copying, so this is "no re-anchoring outside
a narrow penalized zone," not "none under pure reward"; predictions wide (7%
extremes). Pre-registration 2/6: submit ✓, transfer-degradation ✓;
anchoring-returns ✗, search-collapse ✗ (became adaptive), endpoint band ✗
(worse than v2), Flash-pattern ✗. Two-factor caveat: v5 differs from v2
in penalty AND search.

### Frontier panels (all 16 cells; local harness via Bedrock/Vertex)

Search-off arm measured 2026-07-30–31; search-on arm 2026-08-05–06 (env
0.1.18, AgentCore). Values in the grand summary. Search-worth per model
(paired within-model): Sonnet +0.026/+0.027 (on/off), Opus +0.022/(off
n/a→0.256), Flash −0.007/+0.027, Pro +0.010/+0.006*. Calibration doubles
without the anchor for every frontier model (e.g. Pro .062→.219).

## Exploratory: uncertain-question subset (pre-declared band, computed 2026-08-07)

Motivation: Turtel et al.'s Fig. 3 shows skill concentrates where the
market is uncertain. Subset rule (t-computable, no selection leakage):
price_at_cutoff in [0.30, 0.70]; n=104 of 265. Evidence-based (market-off)
cells, soft-Brier: crowd 0.232 · trained v3 0.276 · Opus(on-arm) 0.281 ·
Sonnet 0.293/0.295 · Flash 0.319/0.321 · Pro 0.336.

Paired (frontier − trained v3; + = trained better): vs Pro +0.059
(t=2.18); vs Flash +0.045 (t=1.72); vs Sonnet +0.016 (n.s.); vs Opus
+0.005 (tie, cross-arm). Crowd remains significantly ahead of trained
(t=2.05) even here. GRADE: exploratory, theory-motivated; the vs-Pro t
does not survive 4-way multiple-comparison correction. Effect sizes are
3–50× the full-set gaps, confirming discrimination concentrates on
uncertain questions; the n=593 extension pre-registers this subset
analysis as confirmatory.

## Corrections (append-only)

- **C1 (2026-08-04).** Panel ECEs for trained/base were v1-era values
  carried over (0.119/0.170); corrected to 0.065/0.099. "Calibration
  monotone in capability" retracted → "calibration is trainable."
  Discovered by re-derivation during figure generation.
- **C2 (2026-08-04).** ECE convention pinned: submitted-only, 10
  equal-mass bins, paper-wide. v3 restated 0.170/0.103.
- **C3 (2026-08-07).** v3/v4/v5 were documented as "pure Brier"; in fact
  the environment's default anti-copy ramp (max −0.20 within 0.02 of the
  crowd price) remained active — those configs never overrode
  `crowd_copy_penalty`. Incidence among submitted training rollouts:
  v3 5.0%, v4 4.6–5.5%, v5 11.2% (v2's ramp was by design and fired 51%).
  All eval tables unaffected (they use the logged `brier` field, not
  `reward`). v5's "no re-anchoring under pure reward" is restated as "no
  re-anchoring outside a residual 0.02-radius penalized zone"; the v5
  pre-registration premise ("penalty removed") was partially false.
  Discovered 2026-08-07 while documenting per-run reward functions.

## Discarded data

- Four re-eval probes (2026-08-01) on degraded fallback search after
  `prime train stop` invocations failed (CLI flag misuse, diagnosed
  later): agent behavior poisoned (submit 54–88%). ~$25. Unused.
- Early "Gemini Flash" panel row was Sonnet (config silently overrides
  `-m`); retracted, re-run with per-model configs.
- v4 first-attempt training rollouts (leak-contaminated reward window).

## Audits & validity notes

- **Search-health audit**: strict per-trace zero-result rates — search-off
  era 93–100% empty everywhere; search-on era 16–58% empty (legitimate
  filter drops). Detection required reading rollouts; every status-code
  check passed throughout. Remediations: result-count health checks,
  never-cache/serve empties, error→empty degradation (erroring tools cost
  97→54% submit in probes), licensed-API backend.
- **Selection-criteria audit (D28-style)**: three hindsight-conditioned
  dataset elements — resolved-only membership (partially mitigated by
  scheduled-end-date windows), lifetime-volume filter/caps, cutoff
  placement relative to realized resolution. No outcome-direction leakage
  into context; scopes external validity.
- Frontier rows ran a 7-of-13 toolset (provider schema limits); ours ran
  all 13.
- **Serving-gap addendum (2026-08-07):** the v4 step-33 adapter deployed on
  the platform's own inference product passed behavioral vitals (submit
  100%, searches 2.1) but failed score validation against the training-
  stack reference: soft-Brier 0.2945 vs 0.2521, ECE 0.218 vs 0.128,
  median |p−crowd| 0.210 vs 0.125. Third serving stack measured, third
  distinct behavior (train-cluster faithful; local vLLM severely degraded;
  provider inference calibration-degraded). Measured policy quality is a
  property of the policy×serving-stack pair; endpoint re-evaluation via
  deployments is ruled out (raw: `v4step33_deployed_validation_eval.jsonl`).
- Final-step artifacts: v1/v2/v3 step-33 checkpoints lost to a recurring
  upload bug; v4c step-33 adapter survived (first ever). All adapters
  archived locally (`artifacts/adapters/MANIFEST.json`).
- Trained search-off models are search-naive; v5-vs-v2 differs in two
  factors; Pro search-on market-on is n=231; base search-on cells are
  step-1 evals.
- Retrospective data: day-level ordering only (intra-day sequence
  unrecoverable; cf. EQP).

## Provenance / costs

Leak safety: 5 layers (blocklist, provider date filters, heuristics,
Claude Haiku 4.5 judge — sole zero-leak-error model of six benchmarked;
Qwen3-32B fallback — plus cutoff clamps + verbose-date parsing). Search
cache doubles as auditable corpus. Raw data in this directory:
`v2/v3/v4/v5_run_webhook.jsonl.gz`, platform metrics JSONs, all
`*_eval.jsonl`, pinned splits v3/v4/v5. Costs: ~$300 Prime total across
5 runs + probes + evals (incl. ~$30 discarded); frontier panels + filter
on AWS/GCP credits; search infra ~$0.08/h Azure (decommission pending).

## Base probe at n=593 (2026-08-07, run `f1az1iyl`, env 0.1.20)

Untrained Qwen3.5-35B-A3B, dual-condition eval on the full 593 split,
policy v0 (no version mixing — clean instrument), search-on. Archived:
`results/base593_webhook_raw.jsonl.gz`.

| condition | soft-Brier | submit | ECE |
|---|---|---|---|
| market-on | 0.2575 | 52.4% | 0.168 |
| market-off | 0.2639 | 79.3% | 0.184 |

Paired anchor-worth (off−on): +0.0064 ± 0.0186 (t=0.68, null) — the
base barely converts anchor access into score, unlike the trained
policies (v4-sibling +0.020, v5-sibling +0.045): anchor-worth is
LEARNED, completing the monotone exposure ladder (Finding 14).

Same-week, same-instrument comparisons at n=593 (search-on):

- **Evidence-based column (market-off): base 0.2639 ≈ v5-sibling 0.2651
  ≈ v4-sibling 0.2679.** The training gain in this column is zero at
  sibling resolution; training's real effect here is coverage
  (submit 79%→99%+) and behavior, not headline accuracy. The marquee
  claim must be stated as parity-with-Opus + robustness + coverage, NOT
  as "training lifted accuracy over base" in this column.
- **Anchored column (market-on): base 0.2575 vs v5-sibling 0.2198** —
  training pays +0.036 where the anchor can be exploited; and coverage
  52%→99.7%.
- Base ECEs (0.168/0.184) ≈ trained sibling ECEs (0.157–0.190): no
  calibration gain from training in the search-on regime (consistent
  with Finding 6's n=593 revision).
