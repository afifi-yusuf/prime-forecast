# 7. Discussion

**The anchor ladder as a unifying mechanism.** Our three completed
training runs are most economically described as a single phenomenon
observed at three rungs. Outcome-based reward on a proper scoring rule is
most cheaply harvested by locating a statistical regularity and moderating
toward it: the market price when visible (run 1: median distance 0.028,
camped outside the penalty cliff), the penalty boundary itself when the
cliff becomes a ramp (run 2: median distance exactly ε = 0.080,
pre-registered), and the dataset base rate when the market is hidden
(run 3: median prediction 0.35 against a base rate of 0.355,
pre-registered). Anchoring plus moderation *is* calibration — which
explains why calibration improved 30–40% in every run while resolution
moved at most +0.016 — and each anchor is cheaper than genuine research.
On this reading, "learning to forecast" is what remains after the anchors
are exhausted, and the order-of-magnitude larger training sets of prior
work [Turtel; Mantic] may function less as more signal than as longer
ladders: enough reward pressure to exhaust the shortcuts. This is a
testable account: it predicts that resolution gains in any outcome-based
RLVR forecaster will be preceded by measurable de-anchoring, and that
reward shaping alone cannot produce them (our run 2 shows shaping merely
relocates the camp).

**The market price is a supervision leak, and news is stale.** The
anchor-worth table quantifies what prior comparisons left implicit:
access to the crowd's belief is worth +0.039 to +0.075 Brier — several
times any training effect we measured, and *largest for the strongest
models*. The frontier search results then supply the complementary
surprise: giving models real retrieval made them worse in seven of eight
cells, because retrieved public news is already priced in; deviating from
an efficient market on the strength of stale evidence is trading against
better-informed counterparties, and the trading simulations show exactly
that (Sonnet: +$0.010/bet blind, −$0.018/bet informed). The one positive
cell — Gemini Flash with market access — kept the tightest crowd anchor
of any policy measured: search paid only where it was subordinated to the
price. Together these results dissolve the genre's central comparison.
"Small trained model beats frontier" is reproducible in either direction
by choosing who sees the market: our trained model with the anchor
"beats" Sonnet without it, and Sonnet with the anchor "beats" our model
without. Until the information channels are controlled, such comparisons
measure scaffold allocation, not capability.

**What training bought that scale did not.** The affirmative result is
robustness. The search-trained policy is the only one measured whose
performance survives functioning retrieval unchanged, and it ends
statistically tied with Opus 4.5 atop the evidence-based column at
roughly one-hundredth the inference cost. We read this as evidence that
*evidence discipline* — when to search, how much to trust what returns,
when to fall back on priors — is learnable by RL but not conferred
zero-shot by scale: the frontier models over-trust what they read, while
the trained policy, whose reward history included thousands of
encounters with filtered, partial, and empty search results, treats
retrieval as one noisy signal among several. The v4 traces make the
learned skill legible: bounded arithmetic arguments over retrieved
figures, turn-economical search budgets, and query phrasing adapted to
the leak filter's constraints.

**Measurement is the field's binding constraint.** Three of our findings
exist only because we read rollouts rather than dashboards: the silent
search outage (93–100% empty results across an entire evaluation campaign
returning HTTP 200), the reward-leakage channel through unparsed
publication dates (a policy briefly earning near-perfect reward by
retrieving a resolved outcome — invisible in every aggregate metric), and
the serving gap (identical weights, 94% vs 48% task compliance across
inference stacks). Each is a way an agentic RL result can be confidently
wrong, and none is specific to our setup. We suspect analogues exist
unnoticed in published work, and we adopt — and argue the field should
adopt — the disciplines that caught them: trace-level audits as part of
any evaluation, per-rollout instrumentation through the serving stack
that trained the policy, difficulty adjustment for single-epoch training
curves, pre-registered predictions, and corrections kept as first-class
records rather than silent edits.

**Limits of the account.** Our training regime is small (2,113 questions,
single epoch, one seed per condition), our test set moderate (n = 265;
key claims are therefore paired effects, explicit bounds, or replicated
directions), and the retrospective data admits only day-level temporal
ordering — no leak filter can recover intra-day sequence, an
unrecoverable property of archived market data [cf. EQP]. The fourth
training-condition cell (market and search jointly available) was
designed and pre-registered but blocked by a platform model deprecation
before launch; its predictions are timestamped in the repository. These
bounds define the regime our conclusions inhabit: what outcome-based
RLVR teaches *below* the data scale where prior work reports accuracy
gains — and what the instruments must catch before any such gain can be
believed.
