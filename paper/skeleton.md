# Who's Really Forecasting? Decomposing Agentic RLVR Forecasters into Crowd, Scaffold, and Weights

> Working skeleton (2026-08-04). Every claim carries a tag:
> **[LARGE]** large point-estimate effect · **[BOUND]** powered bound/null ·
> **[DIR]** consistent direction across evals, per-cell underpowered ·
> **[SCOPE]** validity statement. Numbers → `../results/RESULTS.md`
> (authoritative); figures → `../results/figures/`.

## Abstract (skeleton)

Prior work trains LLM forecasters on frozen research context and reports
that small trained models beat frontier baselines. We build an environment
where context acquisition lives inside the RL loop and — critically — the
prediction market's own price is a tool that can be granted or withheld,
for our trained models and for five frontier models run through the
identical agentic harness. This knob yields a decomposition no prior setup
could measure. (1) The crowd term dominates: withholding the price costs
every policy 0.039–0.075 Brier (paired, p<0.001), more for frontier models
than small ones. (2) "Small trained model beats frontier" reproduces in
either direction by toggling the price — prior headlines of this shape are
scaffold artifacts. (3) Across two training runs, outcome-based RL moved
resolution by ≤0.02 (bounded) while consistently transforming behavior:
coverage 64→100%, calibration −30–40%, reaching frontier-level ECE.
(4) Pre-registered: with the market hidden, the policy re-anchors on the
dataset base rate (median prediction 0.35 vs base rate 0.355) — three runs,
three anchors (crowd → penalty boundary → base rate). Outcome-based RLVR
teaches calibration by seeking anchors; forecasting skill is what remains
when anchors run out. Closer: *the market price functions as a supervision
leak — it flows into every policy that can see it, flattens capability
differences, and substitutes for learning itself; we measure that flow.*

## 1. Introduction

- Hook: the genre's claim shape ("our trained X beats frontier Y") and why
  it is unidentified without an anchor-controlled baseline. [Fig placement:
  none; Table 1 teaser optional]
- Our design: agentic harness, market price as a controllable tool, six
  policies (2 ours, 4 frontier) under one protocol; retrieval in the RL
  loop (the extension both Turtel and Mantic name as future work).
- Contributions list (the four abstract points + methodology bullets:
  platform-stack evals, difficulty-adjusted curves, auditable leak
  pipeline, pre-registration).
- **[SCOPE]** paragraph, early and plain: a trace audit found the web-search
  channel dead (93–100% empty) across all headline evals and training —
  results characterize the reduced no-live-search harness; the working-
  search run (v4) is designed, staged, and future work. Frame as the
  no-search arm of a two-arm design.

## 2. Related work

- Turtel et al. (outcome-RL, 10k→110k questions, frontier baseline = o1 on
  frozen headlines, no market price in context). Mantic (frozen research
  phase, Metaculus score, no crowd baseline, no CIs, N unstated). Map: what
  each measures vs what our knob isolates.
- Halawi et al. (retrieval scaffolds), EQP audit (independent corroboration:
  a price-delta "world model" loses to a 2-parameter price-only baseline;
  69% of its "relevant" articles postdated the cutoff — external validation
  of the crowd-dominance and leakage concerns; cite for the "what does a
  score license" framing).
- Prediction-market efficiency literature (one paragraph: why crowd ≈
  optimal is expected).

## 3. Environment

- 3.1 Task and tools (13 tools; market tools enumerated; belief protocol).
- 3.2 Leak safety: 5 layers; filter-model benchmark (Haiku 4.5 zero
  leak-direction errors); cutoff-clamped tools; published search corpus.
- 3.3 Dataset v4: 2,642 questions / 1,377 events; uniform-in-lifetime
  cutoffs (Turtel protocol); caps; temporal split; test n=265, crowd Brier
  0.189, YES base rate 0.355.
- 3.4 Reward: Brier + no-submit floor 0.55; the anti-anchoring penalty arc
  (cliff → ramp) told in §6; GRPO note: group-relative advantages cancel
  question-constant terms — market-relative bonuses are gradient no-ops
  (design lemma, also kills profit-shaped rewards).

## 4. Methodology contributions

- 4.1 Platform-stack evaluation via results webhook (the 94%/48% serving
  gap **[LARGE]**; complete per-rollout capture at ~$3–5/condition).
- 4.2 Difficulty-adjusted training curves: raw single-epoch reward is
  batch-difficulty noise (r=+0.58 with crowd-attainable); adjusted gap to
  crowd narrows 0.145→0.059, t=4.5 **[LARGE]**; post-saturation +0.0021/step
  t=1.5 **[DIR]**. [Fig: v2_reward_difficulty_adjusted]
- 4.3 Pre-registration + append-only corrections (point to repo ledger:
  C1 ECE carryover, the retracted Gemini row, discarded probes).

## 5. Experimental setup

Runs table: v1 (9 epochs, cliff), v2 (single epoch, ramp), v3 (single
epoch, no market tools, pure Brier — pre-registered). Frontier protocol:
LiteLLM→Bedrock/Vertex, 7-of-13 toolset caveat, per-model configs.
Metrics: soft-Brier (0.5-imputed ≡ Turtel), ECE submitted-only 10
equal-mass bins, paired tests throughout. **[SCOPE]** search-audit table
(93–100% empty) + remediation summary.

## 6. Results

- 6.1 **The band and the ladder** [Fig f1]. Accuracy: six policies, one
  band, nobody beats the crowd **[BOUND]** (Opus−crowd +0.003±0.008).
  Calibration: trainable — trained 35B 0.065 ≈ Sonnet 0.063; untrained and
  anchor-denied policies are the outliers **[DIR]**. [Fig f2 reliability]
- 6.2 **Anchor decomposition** [Fig f5; Table anchor-worth]. Paired
  anchor-worth +0.039…+0.075, all p<0.001 **[LARGE]**; frontier leans more
  than 35Bs; manufactured-headline demonstration (0.211-with beats
  0.246-without) **[LARGE]**.
- 6.3 **What RL changes** — coverage 64→100% **[LARGE]**; matched-subset
  resolution null −0.001±0.020 **[BOUND]**; ECE −30–40% everywhere
  **[DIR]**; crowd-independence 3×.
- 6.4 **Reward shaping moves anchors** [Fig f7]. Cliff-camping at 0.028 →
  ramp-camping at exactly 0.080 **[LARGE]**, pre-registered hypothesis.
- 6.5 **The anchor ladder** [Fig f8]. v3: median prediction = base rate
  (0.35 vs 0.355), extremes 10→2% **[LARGE]**, pre-registered 4/5;
  resolution +0.016±0.025 **[DIR]** (4× v2's effect); trained v3 tops the
  no-market column alongside Sonnet. Interpretation scoped: dead search
  means herding-vs-research is resolved by v4, not this run.

## 7. Discussion

- The anchor ladder as the unifying mechanism (crowd → boundary → base
  rate); calibration as anchor-seeking; skill as anchor exhaustion — and
  the reading of prior work's data-scale gains it suggests.
- The supervision-leak framing of the market price.
- What the genre should report: anchor-controlled baselines, search-health
  audits, difficulty-adjusted curves, powered nulls.

## 8. Limitations **[SCOPE]**

- n=265 ceiling: which claims are bounds vs directions (be explicit).
- No-live-search regime; trained models are search-naive; v4 staged
  (verified stack, config committed) — the two-arm completion.
- 2.1k-question training regime (prior accuracy gains needed ≥10k).
- Retrospective data, publish-date granularity (cite EQP's unrecoverability
  result; forward-recorded data as the fix).
- Single seed per condition (cost); step-22 checkpoint cells.

## 9. Future work

v4 (train with working search) + event-necessity perturbations on retrieved
context (EQP-style EMPTY/WDATE/IRRLEN operators) as the acceptance test for
"learned to research"; distill-then-RL from Sonnet traces; forward-recorded
dataset (EQP campaign); Ensemble-7 and n≈1,200 eval when funded.

## Figures/tables checklist

- [x] f1 panel (Brier+ECE, CIs, crowd lines)
- [x] f2 reliability (base/trained/Opus)
- [x] f5 anchor-worth pairs
- [x] f7 boundary relocation histogram
- [x] f8 base-rate herding
- [x] difficulty-adjusted curve
- [ ] Table 1: panel (from RESULTS.md)
- [ ] Table 2: quadrant + anchor-worth
- [ ] Table 3: v3 + pre-registration scorecard
- [ ] search-audit mini-table
- deliberately omitted: trading figures (no trading claims at n=265)
