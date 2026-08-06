# 2. Related Work

**Outcome-based RL for forecasting.** Turtel et al. train
DeepSeek-R1-Distill-14B with outcome-based RL on 10k–110k Polymarket
questions, reporting soft-Brier parity with o1 (0.190 vs 0.202, paired
n = 1,265) and improved calibration; their statistical reporting — paired
Wald intervals, bootstrap ECE, significance tests — sets the genre's
standard, and we adopt their metric conventions (soft-Brier imputation,
ten-bin equal-mass ECE) for comparability. Mantic fine-tune gpt-oss-120b
on ~10k synthetic questions and report frontier-level Metaculus scores.
Both freeze research context before training — headlines in the prompt,
or a pre-generated research phase shared across models — and both name
retrieval inside the training loop as the consequential extension; our
environment implements it. Both also evaluate frontier baselines without
the trained system's scaffold: o1 receives frozen headlines and no market
price; Mantic's panel receives frozen research. Our anchor-controlled
design shows such comparisons can be inverted at will (§6.2), and our
agentic frontier panel — the same tools, filters, and turn budget for
every policy — is to our knowledge the first in this domain. Neither
prior work reports search-channel health, an audit our results suggest
is not optional (§6.5, §7). Mantic reports no confidence intervals or
sample size; where our claims are similarly under-powered we label them
as bounds rather than findings.

**LLM forecasting without RL.** Halawi et al. show that
retrieval-augmented scaffolds bring LLM forecasters near crowd accuracy,
foreshadowing our scaffold-dominance results; ForecastBench tracks
frontier models against superforecasters on an ongoing benchmark. Schoenegger
et al. and related work document LLM ensembles approaching human crowd
performance. Our contribution to this line is decompositional: we
measure *which* scaffold components carry the performance (the market
price: +0.039–0.075; web search: ≈0 or negative) rather than the
scaffold's aggregate worth.

**Evaluation validity for forecasting systems.** The Evidence
Qualification Protocol (EQP) audits a published event-forecasting system
and finds its headline subset hindsight-conditioned (69% of attributed
articles postdate the forecast cutoff) and its skill negative against a
two-parameter price-only baseline — independent corroboration, by audit
rather than intervention, of our central finding that price-copying
dominates apparent skill. We share its position that a score licenses
only what its measurement conditions support, adopt kindred practices
(pre-registration, append-only corrections, trace-level audits), and
appropriated its perturbation-operator design for our staged
event-necessity evaluations. Its documentation of unrecoverable intra-day
ordering in archived market data bounds our leak-safety claims (§8).

**Prediction-market efficiency.** A long literature documents that
prediction-market prices aggregate dispersed information efficiently
[Wolfers & Zitzewitz; Berg et al.], implying that public news is largely
priced in at retrieval time. Our frontier search results (§6.5) are an
agentic restatement of this classical point: models that deviate from the
price on the strength of retrieved news systematically lose to it —
including in per-bet trading simulations. To our knowledge this is the
first measurement of that mechanism operating *inside* LLM agents.

**Reward hacking and shortcut learning.** Our anchor-ladder findings
(§6.3–6.4) join a broad literature on specification gaming in RL: the
policy optimizes the written reward's cheapest path — a penalty
boundary, a base rate — rather than the intended skill. The forecasting
setting is distinctive in that its dominant shortcut (the crowd price)
is also a near-optimal answer, making the gaming *quantitatively
rational* and detectable only by ablation. The reward-leakage incident
we document (§7) — retrieval of a resolved outcome scoring near-perfect
reward — is to our knowledge the first reported case of temporal leakage
functioning as a live reward channel in agentic RL, rather than as a
dataset-contamination concern.
