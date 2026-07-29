# Reward design: evolution and lessons

Chronological record of every reward change, its motivation, and the observed
behavioral consequence. All rewards operate on the submitted probability
`p = P(YES)` against the resolved outcome `y ∈ {0,1}`, trained with GRPO
(group-relative advantages, 8 rollouts per question).

## Base objective

```
r = 1 − (p − y)²        (positive-shifted Brier, strictly proper)
```

Same choice as Turtel et al. 2025 and Mantic 2026; both report the log score
destabilizing training (unbounded ⇒ high-variance gradients), so bounded Brier
is the genre standard. Submitted probabilities are clamped to [0.05, 0.95] —
never 0/1, bounding worst-case reward and discouraging degenerate certainty.

## Change 1 — no-submit floor: 0.75 → 0.55 (pre-v1, 9B smoke era)

**Original:** a missing/failed submission scored 0.75 — numerically equal to
always predicting 0.5. **Observed failure:** this matched the mean reward of
mediocre submissions, so stalling to max-turns was a *safe* strategy; smoke
runs showed low submit rates with no gradient pressure to finish.

**Fix:** floor lowered to **0.55**, strictly below the always-0.5 reward
(0.75), so any calibrated submission beats silence in expectation.

**Observed dynamics under 0.55 (35B, both the pilot and the 45-step run):**
a three-phase curve. (1) Early *confident-wrong trap*: the model submits
overconfident extremes (0.9+), scores 0.1–0.3 — i.e. *below* the floor — and
GRPO groups reward silence over wrong submissions, dropping submit rates to
~20–30%. (2) *Escape via confidence moderation*: the policy discovers hedged
mid-range submissions beat the floor; submit rate climbs (steps ~21–27 of the
main run). (3) *Consolidation*: 84–97% submit on train batches; 94% on
held-out (platform eval). Lesson: the floor's value sits in a narrow band —
too high teaches stalling, too low would punish honest abstention on genuinely
unresolvable questions; 0.55 empirically threads it, at the cost of a
transient mid-training dip that should not be mistaken for divergence.

## Change 2 — BLF protocol bonus: shipped OFF (weight 0.0)

An auxiliary shaped bonus (fraction of: explicit submit call, ≥1 research
call, ≥1 belief update, belief moved off 0.5, non-empty evidence lists).
**Observed failure in smoke runs:** it paid for *form* rather than outcome —
encouraging a degenerate `web_search → submit` two-step that satisfied the
checklist without research depth. Retained in the codebase at weight 0 for
ablations only. Lesson: process-shaping on top of a proper scoring rule
invited Goodharting faster than it improved calibration.

## Change 3 — binary crowd-copy penalty (v1 main run)

With cutoff-safe Polymarket crowd tools enabled (price-at-cutoff, history),
GRPO can farm reward by echoing the market price. v1 countermeasure:

```
if |p − crowd| ≤ ε:  r ← max(0, r − w)      (ε = 0.02, w = 0.20)
```

**Observed result (held-out platform eval of the v1 policy):** the policy
learned to anchor *just outside* the cliff — median |p − crowd| = 0.028,
32% of predictions inside 0.02, submitted-Brier 0.214 vs crowd 0.194
(statistically indistinguishable), simulated trading −$0.08/bet against the
market. The penalty prevented exact echoes but a threshold is a location, and
the policy simply moved next to it. Anchor-and-adjust remained the optimal
strategy under the reward as written.

## Change 4 — graded anti-anchoring ramp (v2, env 0.1.14)

```
if |p − crowd| < ε:  r ← max(0, r − w·(1 − |p − crowd|/ε))
```

with v2 run settings ε = 0.08, w = 0.15. Full penalty at an exact echo,
decaying linearly to zero at ε — continuous, so there is no boundary to camp
beside; within the ramp every unit of genuine deviation from the crowd is
paid at constant marginal rate.

**Design constraint that shaped this choice (GRPO-specific):** advantages are
normalized *within* a question's rollout group, so any reward term constant
across a group cancels out of the gradient. "Reward beating the crowd's
Brier" (r + λ·(crowd_brier − brier)) is therefore a no-op: crowd_brier is
question-constant and the residual is the Brier term already present. Only
terms that *vary between rollouts of the same question* — like each rollout's
distance from the crowd — can shape the learned strategy. This rules out the
whole family of question-level relative-performance bonuses and is, we
believe, an underappreciated constraint on reward design for group-relative
RL.

## Training-mechanics adjuncts (reward-adjacent)

- **Zero-advantage pre-batch filter** (prime-rl): drops rollout groups whose
  rewards are all identical (no gradient signal). Under crowd-anchoring
  collapse, whole groups converge to near-identical rewards; the filter both
  saves compute and serves as a live collapse detector (drop-rate fell from
  ~0.43–0.73 early to ~0.20 as groups differentiated).
- **Probability clamp [0.05, 0.95]** applied at submit parsing.
- **Metrics vs reward separation:** `market_brier` (crowd baseline) is logged
  as an eval metric but deliberately excluded from reward (see GRPO
  cancellation above; it would also leak the crowd's calibration into
  training pressure in uncontrolled ways).

## Open questions for the paper

1. Does the graded ramp break anchoring, or does the policy re-anchor at ε?
   (v2 run answers this; if it camps at 0.08, the next step is a convex or
   unbounded-decay penalty.)
2. Abstention economics: 0.55 flat floor treats all no-submits equally; a
   calibrated-abstention reward (e.g. proper scoring with an explicit
   "no forecast" outside option) is principled but changes the task
   definition relative to the comparison papers.
3. Whether anti-anchoring shaping costs accuracy on questions where the crowd
   is simply right — the tools-off eval condition bounds how much independent
   skill exists to substitute.
