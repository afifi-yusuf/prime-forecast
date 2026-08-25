# Paper framing: the anatomy paper

The frame: **an anatomy, not a victory lap.** Every RLVR-forecasting paper so
far is "we trained X and it beat Y." Ours is the first with the
instrumentation to answer *what these systems are actually made of* — how
much is scaffold, how much is crowd, how much is weights, and what RL
specifically changes. Under this frame every result, including the nulls, is
load-bearing evidence rather than a miss.

## Narrative shape (mirrors evidence, not chronology)

1. **Setup / hook.** Prior work trains forecasters on frozen research context
   and evaluates against frontier models denied any scaffold. We build the
   first environment where everyone — trained 35B and five frontier models —
   forecasts through the identical agentic loop, retrieval inside the RL
   loop, and the market price is a *tool that can be granted or withheld*.
   That knob is the paper's scalpel.

2. **Result A — the decomposition (strongest table).** Performance
   decomposes into crowd + scaffold + weights; the crowd term dominates.
   Anchor-worth: frontier +0.058–0.075, 35Bs +0.039, all paired p<0.001.
   Punchline: frontier models lean on the crowd MORE than small models;
   without the market Gemini Pro falls below an untrained 35B on
   calibration.

3. **Result B — the genre's headlines are scaffold artifacts.** We reproduce
   "small trained model beats frontier" on demand in either direction by
   toggling who sees the market (0.211-with beats 0.246-without). Framed
   respectfully: the comparison is unidentified without an anchor-controlled
   baseline, which we provide.

4. **Result C — what RL actually changes.** Across four independent evals:
   coverage 64→100%, crowd-independence 3×, calibration direction (ECE down
   everywhere) — resolution flat (paired −0.001 ± 0.020). Reward-shaping
   micro-findings (cliff-camping, boundary relocation, GRPO cancellation) as
   mechanism: the policy optimizes the reward you wrote, not the skill you
   meant.

5. **Result D — v3 resolves the causal ambiguity** (either way):
   - anchor removed → resolution learning appears: **shortcut suppression**
     headline (the market price functions as a supervision leak that
     substitutes for learning);
   - still flat → the null is real, data-scale-bound, stated with ±CI.
   Write both paragraph variants now; drop one in when v3 lands.

6. **Methods as contributions** (brief, explicit): platform-stack evals via
   results webhook, difficulty-adjusted training
   curves (raw curves uninterpretable under single-epoch RLVR),
   5-layer auditable leak pipeline + released search corpus.

## Title direction

Question-form to match the anatomy frame, e.g.:
*"Who's really forecasting? Decomposing agentic RLVR forecasters into
crowd, scaffold, and weights."*

## Framing disciplines

- Every claim tagged as **large-effect / powered-bound / consistent-
  direction** — never blurred (lesson of the Finding 9c overclaim-retract).
- Limitations as **scope, not apology**: "2k-question regime; both prior
  papers needed ≥10k for accuracy gains — we characterize what RL teaches
  below that threshold."
- The crowd is the **environment's structure, not an adversary**: nobody
  beats it, and that is a measurement, not a shortfall.

## Abstract closer (candidate)

*In agentic forecasting, the market price functions as a supervision leak —
it flows into every policy that can see it, flattens capability
differences, and may substitute for learning itself; we measure that flow
for the first time.*
