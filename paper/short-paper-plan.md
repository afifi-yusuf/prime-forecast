# Short paper (4 pages) — plan

Target: IAEval @ NeurIPS 2026 short-paper track (NeurIPS 2026 template,
anonymized, references + appendix excluded from the 4-page limit —
verify exact policy against the CFP before layout). Deadline Aug 29
AoE; internal target Aug 27 for the short paper.

## Thesis (one sentence)

We contribute an open agentic forecasting dataset/environment in which
context is acquired by the agent rather than pre-collected, and show
that single-epoch outcome-based RL takes an open 35B (3B active) to
parity with Claude Opus 4.5 at evidence-based forecasting at ~1/100th
the inference cost.

## Contributions (the three the intro names)

1. **Agentic dataset + environment** (open-sourced on the Prime hub):
   2,100+ train / ~600 held-out resolved Polymarket questions;
   cutoff-filtered live tool use; strict temporal eligibility (all
   outcomes postdate base-model knowledge; test strictly postdates
   train); continuously renewable with no context-snapshot step.
2. **An identical-harness agentic evaluation**: trained, base, and four
   frontier models in the same tools/filters/turn budget;
   platform-served policies with per-rollout capture; trace-level
   audits as part of the method.
3. **Result**: RL-trained Qwen3.5-35B-A3B matches Opus 4.5 in the
   evidence-based column (search on, market price withheld), with
   30–40% ECE reduction and coverage 64%→~100% at flat accuracy; on
   uncertain questions (price 0.30–0.70) it tops every frontier point
   estimate.

## What is IN (and at what claim strength)

- Evidence-based column table + f10 bar chart (headline).
- Behavioral gains: f12 (ECE deltas + coverage) — strongest replicated
  training effect; one reliability curve if space (f2, else appendix).
- Uncertain-band divergence (f11) — labeled exploratory, pre-declared.
- One sentence of anchor logic to justify withholding the price
  ("with the price visible every policy converges toward the crowd and
  none beats it — the comparison measures price-copying, not
  forecasting"), pointing to appendix table.
- One observational sentence on retrieval robustness: "frontier models
  degraded in 7 of 8 cells when live search was enabled; the trained
  policy was unchanged" — no mechanism claims in the main text.
- CIs RESTORED for every headline number (paper policy: intervals in
  text/table notes; the no-CI rule applies only to the repo results
  page). Parity stated as "statistically indistinguishable" WITH the
  paired interval.
- One trace vignette IF space (the Messi rollout, 3 sentences, boxed) —
  else appendix.

## What is OUT (main text) → appendix or cut

- Search-worth campaign, trading simulations, efficient-market
  mechanism → appendix ("Why we withhold the market price" +
  search-worth table); avoids the three attack surfaces (retrieval
  quality, filter artifact, cross-campaign timing).
- Anchor-worth decomposition, anchor ladder, boundary relocation,
  base-rate herding → appendix (one table + f5/f7/f8).
- Search-off arm and outage history → OMITTED entirely (report only
  search-on-arm cells in the main text; search-off numbers appear only
  in the appendix anchor table, labeled as an earlier no-retrieval
  condition). This removes the two-arm scope apparatus from the paper.
- Serving gap / reward leakage / audit war stories → one sentence in
  §3 ("trace-level audits caught and corrected instrument failures;
  ledger in the repo") + appendix paragraph.
- n=593 expansion, sibling variance, C1–C3 ledger → repo/appendix
  pointer only.
- Reward-shaping details beyond the base formula → appendix.

## Page budget (4.0 pages)

- §1 Intro + contributions ......................... 0.75 pp
  (forecasting as RLVR domain; prior work pre-collects context; the
  three contributions; headline numbers in prose)
- §2 The environment & dataset ..................... 1.00 pp
  (agentic-context framing; tools in one compact paragraph; leak
  filtering in one paragraph; data collection filters + composition in
  one paragraph + small table; reward formula + why Brier in 3 lines)
- §3 Evaluation setup .............................. 0.50 pp
  (identical harness; platform serving + webhook capture; why the
  market price is withheld; metric conventions incl. CIs)
- §4 Results ....................................... 1.25 pp
  (f10 + parity table with CIs; f12 behavioral gains; f11 uncertain
  band, exploratory; retrieval-robustness sentence)
- §5 Related work + discussion ..................... 0.50 pp
  (Turtel/Mantic — frozen context, metric conventions adopted; Halawi
  reconciliation: "retrieval gets you to the crowd, not past it" in
  one line; limitations: sample size, one seed, day-level ordering)
- References (not counted) + Appendix A–D (not counted):
  A. market-visible results + anchor decomposition
  B. search-worth table + trading sims
  C. trace vignettes (Messi; discipline examples)
  D. audits, corrections ledger, reproducibility (env hub link, run
     IDs, per-rollout archives)

## Figures/tables (main text max: 2 figures + 1 table)

- Fig 1 = f10 (evidence-based ranking). Fig 2 = f12 (behavioral gains).
- Table 1 = evidence-based column with soft-Brier [CI], ECE, submit%.
- f11 becomes Fig 3 only if layout allows; else appendix.

## Numbers policy

- Main text uses the original-campaign numbers (0.252/0.254/0.256 …)
  consistently with RESULTS.md; the n=593 sibling replication is an
  appendix robustness note ("continuation re-runs replicate the
  market-on cell within 0.004; evidence-based cells shift ≤0.015,
  within seed variance").
- Every comparative claim gets its paired interval or is labeled a
  point-estimate tier. Uncertain-band stays "exploratory,
  pre-declared".

## Writing tasks (order)

1. Verify CFP short-paper policy (page count, appendix rules,
   anonymization, non-archival) — 15 min, blocks layout decisions.
2. Abstract (~120 words) from the RESULTS.md main-claim block.
3. §2 from RESULTS.md harness/data/leak/reward sections (compress 3:1).
4. §4 from RESULTS.md headline sections; restore CIs from
   RESULTS-detailed.md.
5. §1, §5 fresh (reuse paper/sections/01_intro.md + 02_related.md
   fragments; both pre-date v5 — update counts and drop search-worth
   emphasis).
6. Appendices by lift-and-trim from RESULTS-detailed.md.
7. Anonymization pass: strip run IDs/usernames; anonymous env link
   (anonymous.4open.science mirror) — the Prime hub link deanonymizes
   (`yafifi`), so appendix D's hub link goes in the camera-ready only.
8. LaTeX assembly in NeurIPS 2026 template; figures re-exported at
   column width.

## Risks / open items

- CFP may cap appendices or disallow them for shorts → fallback: cut
  f11 and the vignette, keep appendix A only.
- "Matches Opus" with n=265 CIs is a tie claim — phrase as parity/tier,
  never superiority; superiority language only inside the uncertain
  band and only as point estimates.
- If reviewers ask for search-worth evidence anyway, the appendix
  carries it; main text never depends on it.
- Aug 20 freeze rule still governs which expansion numbers may enter
  (currently: appendix robustness note only).
