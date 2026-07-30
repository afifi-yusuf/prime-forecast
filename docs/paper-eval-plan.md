# What the paper needs: eval + figure plan vs Turtel and Mantic

Derived from full reads of Turtel et al. (arXiv 2505.17989 v4) and Mantic
(ICML 2026). Their published bar, what we already meet, and what remains.

## The two papers' evaluation bars

| dimension | Turtel | Mantic | us today |
|---|---|---|---|
| test set | 1,265 Polymarket, temporal split, public on HF | Metaculus Q2'25, **N never stated** | 265 Polymarket, temporal split |
| accuracy metric | soft-Brier (0.25 non-answer) | Metaculus baseline score (log) | soft-Brier (0.5-imputed ≡ theirs) ✓ |
| calibration | ECE, 10 equal-mass bins | ECE, bins unspecified | ECE 10 equal-mass ✓ |
| statistics | paired Wald CIs, bootstrap ECE, sig. stars | **none at all** | Wald + bootstrap ✓ (n small) |
| crowd baseline | Polymarket price (contemporaneous) | **none** | ✓ |
| frontier baseline | o1 (text-only prompt) | 5 frontier LLMs, frozen shared research | 4 frontier models **agentically** ✓ |
| trading sim | 1-share, +1¢ fee, edge-ranked, 3 stopping rules | none | same mechanics ✓ (partial reporting) |
| ablations | 4 RL algorithms; 10k vs 110k scale; guardrails | scaffold on/off | scaffold×training 2×2 ✓; reward-design arc ✓ |
| training curve | none | test-score vs step, no error bars | difficulty-adjusted curve ✓ (novel) |
| leakage control | o3 judge + 3 rules | "backtest-compliant", unaudited | 5-layer + benchmarked filter + published corpus ✓ |

Headline-claim shapes: Turtel = "matches o1, better ECE than frontier, >10%
trading ROI, doesn't beat market." Mantic = "RL lifts base past frontier on
benchmark score, ECE −17%, scaffold dominates." Both rest on a *weak base
model with visible headroom*; our base-with-tools is already near-crowd, so
our claims are the two-axis result (accuracy scaffold-bound, calibration
capability-bound) + coverage — already documented in research-arc.md.

## Evals still needed (ranked)

1. **Expanded test set, n≈1,200** (candidates cache has 2,103 in-window; cap
   crypto). This is THE gap: Turtel's n=1,265 with paired stars is the genre's
   statistical bar; our n=265 cannot support "Δ vs crowd n.s." claims at their
   standard, and powers the ±0.01-resolution null. Rows in priority order:
   trained v2, base (Prime, ~$40 credits), Sonnet 4.5, one Gemini (API
   credits). Produce a Turtel-style Table 1: soft-Brier | ECE | Δ vs base |
   Δ vs crowd with paired two-sided stars.
2. **Ensemble-7 (median of 7 samples) for trained v2** — Turtel's headline row
   is ensembled; reviewers will ask. 7× rollouts on (a subset of) the test
   set through the platform. Optional but cheap insurance at n=265.
3. **Metaculus baseline score as secondary metric** — free to compute from
   existing per-question data; makes our table directly readable against
   Mantic's Fig. 1 axis.
4. **(done, running) Opus 4.5 row** — completes the frontier ladder.
5. **gpt-oss-120b row (~$2)** — Mantic's base model, agentically: direct
   cross-paper bridge.

## Figures to generate

From EXISTING data (no new evals, no credits):
- **F1. Frontier-panel bar chart with 95% CIs** (Mantic Fig. 1 analog; they
  have no error bars — ours do). Soft-Brier per policy + crowd line; ECE
  panel alongside.
- **F2. Reliability diagrams** (neither paper plots them): trained vs base vs
  Sonnet, 10 equal-mass bins — makes the ECE story visual and is our central
  claim.
- **F3. Trading simulation** (Turtel Fig. 2 analog): cumulative profit vs
  edge-ranked trades per policy, with Edge>ECE and Edge>0 markers; bar panel
  of totals under the 3 stopping rules.
- **F4. Win-rate edge by market-price bucket** (Turtel Fig. 3 analog): where
  does any policy have edge vs the crowd; expect flat ≈0 — an honest negative
  mirroring their 40–60% bucket result.
- **F5. 2×2 quadrant bar figure** (scaffold × training, both metrics) — our
  Mantic-Fig.4-shaped scaffold ablation, with CIs.
- **F6. Difficulty-adjusted training curve** — done
  (`results/figures/v2_reward_difficulty_adjusted.png`); methodological
  contribution (raw curves uninterpretable under single-epoch RLVR; neither
  paper difficulty-adjusts).
- **F7. Anchoring-behavior histogram**: |p − crowd| distributions v1 (ε=0.02
  cliff) vs v2 (ε=0.08 ramp), boundary relocation — supports the
  reward-design findings; no analog in either paper.
- **F8. Category breakdown** (neither paper has one): Brier by question
  category, trained vs crowd.

Redo at n≈1,200 after eval #1: F1–F4 (F2/F3/F4 gain the most from n).

## Differentiators to keep explicit in the writing

- Retrieval **in the RL loop** — the exact extension both papers name as next
  step (Mantic's plateau discussion; Turtel's frozen-headline prompts).
- Frontier models run **agentically**, not on frozen context (novel panel).
- Auditable leak pipeline + released search corpus (both papers are weak here;
  Mantic's is unaudited, Turtel's judge is single-model o3).
- Full statistical reporting Mantic lacks; matched-subset/self-selection
  analysis neither has.
- Cost transparency: whole project ≈ $120 Prime + credits; Turtel used
  8×H100×3 days/run.

## Claim-safety notes (what NOT to claim)

- No trading-ROI headline unless it survives n≈1,200 with CIs (our n=265 P&L
  is noise: −$0.015 to +$0.010/bet).
- "Beats frontier" is not available to us (unlike Mantic); our claim is
  parity-of-band + calibration ladder + coverage, which the panel supports.
- Ensemble our model before comparing against any ensembled baseline row.
