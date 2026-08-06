# 6. Results

All evaluations use the held-out test split (n = 265 resolved Polymarket
questions whose resolutions strictly postdate all training data; crowd
soft-Brier 0.189, YES base rate 0.355). Soft-Brier imputes 0.5 for
non-answers, matching Turtel et al.'s penalty convention; ECE is computed
over submitted forecasts with ten equal-mass bins, with submission rates
reported separately. Our policies are served through the training
platform's own inference stack and captured per-rollout via the results
webhook (§4.1); frontier models run the identical environment through an
API proxy. Unless stated otherwise, comparisons are paired over the same
question set. All results are measured in the reduced no-live-search
harness (§5.3); we return to the consequences of this scope in §6.5 and §8.

## 6.1 Accuracy converges to a crowd-bounded band; calibration is trainable

Figure 1 (left) shows soft-Brier for six policies spanning roughly two
orders of magnitude in scale — an untrained and an RL-trained open 35B
(3B active parameters), Gemini 3.6 Flash, Gemini 3.1 Pro, Claude Sonnet
4.5, and Claude Opus 4.5 — in the identical harness with market tools
available. Every policy lands in a single band between 0.186 and 0.215
with mutually overlapping confidence intervals, and none beats the market
price itself (0.189). The strongest model, Opus 4.5, is the only policy
whose point estimate edges past the crowd, and the paired difference is
indistinguishable from zero (crowd − Opus = +0.003 ± 0.008, t = 0.77).
Whatever separates a 3B-active open model from the strongest frontier
model available, it is not, in this environment, the ability to out-predict
the crowd.

Calibration behaves differently (Figure 1, right; reliability curves in
Figure 2). The best-calibrated policies are Opus 4.5 (ECE 0.053), the
market itself (0.058), Gemini Pro (0.062), Sonnet (0.063) — and the
RL-trained 35B (0.065), which its own untrained base trails at 0.099.
Training moved a mid-size open model into the frontier calibration
cluster at roughly one-tenth of frontier inference cost. We stress the
grade of this claim: per-cell ECE intervals are wide at n = 265, and the
evidence is a consistent direction across every train/eval pair we ran
(0.099→0.065 with market tools; 0.185→0.127 without; 0.170→0.103 in the
no-market-trained run of §6.4) rather than any single significant cell.
An earlier version of this analysis reported calibration as monotone in
model capability; that pattern rested on erroneously carried-over numbers
and is retracted in the repository's correction ledger (C1). The corrected
pattern is starker: calibration tracks training and anchor access, not
scale.

## 6.2 The anchor decomposition

The market price is a tool we can withhold. Doing so, per policy, on the
same questions yields each policy's *anchor-worth* — the paired Brier cost
of losing the crowd (Figure 3, Table 2): Gemini Pro +0.075 [0.040, 0.110],
Flash +0.063 [0.033, 0.094], Sonnet +0.058 [0.031, 0.085], and +0.039
[0.016, 0.063] for both the trained and untrained 35B. Every difference is
individually significant (p < 0.001, paired); these are the largest and
most robust effects in the paper — two to twenty times larger than any
training effect we measured.

Two readings follow. First, frontier models depend on the crowd *more*
than small models do: their in-harness advantage is substantially superior
anchor exploitation, and removing the anchor collapses frontier accuracy
into — and partly below — the 35B band, while Gemini Pro's calibration
(0.219) falls below even the untrained 35B's no-market value (0.185).
Second, the decomposition explains the genre's headline claim shape.
Turtel et al.'s 14B "beats o1" and Mantic's 120B "beats GPT-5" both
compare a scaffolded trained model against frontier baselines denied the
scaffold. We can reproduce that comparison in either direction at will:
our trained 35B with market tools (0.211) "beats" Sonnet 4.5 without them
(0.246). The comparison is unidentified until the anchor is controlled —
which is precisely the control our design adds.

## 6.3 What outcome-based RL actually changes

Across two training runs with different reward shapes, resolution — the
ability to distinguish which questions resolve YES — did not detectably
move. On the subset of questions that both the base and trained v2 policy
answered, the paired Brier difference is −0.001 ± 0.020: flat to within
±0.02. The headline soft-Brier difference (0.215 → 0.211) is driven
entirely by questions the base declined and the trained policy answered.
That points to what training *did* change, consistently and by large
margins: task behavior. Coverage rose from 64% to 100% of questions with
accuracy held flat — the trained policy answers the hard residual the
base self-selects away from, at no headline cost. Median distance from
the crowd tripled (0.028 → 0.080), and calibration improved by 30–40% in
every condition (§6.1). Outcome-based RLVR, at this data scale, is a
behavior-and-calibration trainer, not a resolution trainer.

## 6.4 Reward shaping relocates anchors; it does not remove them

The v1 run used a hard penalty for predictions within 0.02 of the market
price. The learned policy camped at a median distance of 0.028 — just
outside the penalty — with 32% of predictions inside 0.02 (Figure 4). For
v2 we replaced the cliff with a linear ramp extending to ε = 0.08 and
pre-registered the hypothesis that the policy would re-anchor at the new
boundary. It did, exactly: median distance 0.080, with the within-0.02
mass collapsing to 4%. A design lemma explains the robustness of this
behavior under GRPO: group-relative advantage estimation cancels any
reward term that is constant within a question's rollout group, so
"beat the crowd" bonuses are gradient no-ops, and only within-group-varying
shaping — like each rollout's distance to the price — reaches the
gradient. The policy optimizes the geometry of the penalty, not the
intention behind it.

## 6.5 The anchor ladder: remove the crowd and RL finds the next anchor

The v3 run deleted the market tools from training and evaluation entirely,
leaving a pure Brier reward. Before any training metric existed we
committed five predictions to the repository (verifiable by commit
ordering); four held. Submission saturated (74% → 98%). Calibration
improved more than in v2 (0.170 → 0.103). Resolution moved further than
ever before — paired +0.016 ± 0.025, four times the v2 effect — but
remained short of significance at n = 265. And the headline prediction
confirmed: the trained policy's median prediction settled at 0.35 against
a dataset YES base rate of 0.355, with extreme predictions (<0.1 or >0.9)
collapsing from 10% to 2% (Figure 5). Denied the crowd, the policy
migrated to the next-cheapest statistical regularity: the base rate.

Across three runs the pattern is a ladder: the crowd price (v1), the
penalty boundary (v2), the base rate (v3) — each the nearest reward-safe
anchor available under that run's constraints. We propose this as the
unifying mechanism for what outcome-based RLVR teaches in this regime:
proper-scoring-rule reward is most cheaply harvested by anchoring plus
moderation, which *is* calibration; genuine resolution gains begin only
where anchors are exhausted, consistent with the small unlocked gain when
we removed the strongest anchor, and with prior work's observation that
accuracy gains demanded an order of magnitude more training data than we
deploy. One scope note applies (§8): in the no-live-search regime,
research was largely futile, so v3 alone cannot distinguish "RL prefers
anchors to research" from "RL anchors when research is unavailable." The
staged v4 run — identical to v3 with a verified search-on retrieval stack —
separates these hypotheses, with v3 as its exactly-matched control.

The final placement is nonetheless notable: in the no-market condition,
the v3-trained 35B (0.245) is the best point estimate in the column,
alongside Sonnet 4.5 (0.246) and ahead of both Geminis — an open model,
one epoch, ~$20 of compute, matching the strongest frontier model we
could afford to test in the fair version of the comparison.
