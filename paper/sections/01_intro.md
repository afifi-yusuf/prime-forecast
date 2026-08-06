# 1. Introduction

Large language models are increasingly trained and evaluated as
*forecasters*: agents that read the world and output calibrated
probabilities about future events. Recent work reports striking headline
results — a 14B model trained with outcome-based reinforcement learning
"matching o1" on real prediction-market questions [Turtel et al.], a 120B
model "surpassing frontier models" on a forecasting benchmark [Mantic].
These results share a template: train a mid-size model with reinforcement
learning against real outcomes, then compare it to frontier models. They
also share two structural omissions. The frontier baselines are never
given the trained system's scaffold — retrieval, tools, or the market's
own price — and the information channels that dominate forecasting scores
are never measured in isolation.

This paper measures them. We built an agentic RLVR environment on
resolved Polymarket questions in which a multi-turn agent researches each
question with live, cutoff-safe tools and is rewarded on Brier score
against the realized outcome. Two design choices make the environment an
instrument rather than a leaderboard. First, retrieval lives *inside* the
training loop — the extension both prior works name as the consequential
next step. Second, every information channel is a switch: the market's
price history is a tool that can be granted or withheld, and web search
can be enabled or disabled, for our trained models and for five frontier
models run through the *identical* harness — to our knowledge the first
agentic evaluation of frontier models in this domain.

Toggling those switches across four training runs and twenty-two
evaluation cells yields a decomposition of what forecasting scores
actually measure, with results that reframe the genre's headlines:

**The crowd term dominates, and frontier models lean on it hardest.**
Withholding the market price costs every policy 0.039–0.075 Brier
(paired, p<0.001) — more for frontier models than for our 35B. With the
anchor removed, Gemini 2.x-class calibration collapses below that of an
*untrained* open model. "Small trained model beats frontier" reproduces
in either direction by choosing who sees the price: our trained model
with market access "beats" Claude Sonnet 4.5 without it, and vice versa.
The genre's comparison is unidentified until the anchor is controlled.

**Web search — the channel assumed to drive agentic forecasting — is
worth approximately nothing, and frequently less.** In eight frontier
search-ablation cells, functioning retrieval *degraded* performance in seven
(+0.006 to +0.027 Brier): retrieved public news is stale relative to an
efficient market price, and evidence-driven deviation from the crowd
loses. The sole exception kept the tightest crowd anchor of any policy
measured — search helps only when subordinated to the price.

**What outcome-based RL actually teaches is calibration and behavior,
purchased by anchor-seeking.** Across our runs, training saturated task
compliance (submit 64%→100%), improved calibration 30–40% in every
condition, and moved resolution by at most +0.016 (n.s.). Mechanistic
probes explain why: the policy camps at whatever anchor the reward
landscape offers — the crowd price (run 1), the exact edge of an
anti-copying penalty (run 2, pre-registered), and, when the market is
hidden entirely, the dataset base rate (run 3, pre-registered: median
prediction 0.35 against a base rate of 0.355). Remove one anchor and RL
finds the next; genuine research is the expensive strategy that
single-epoch reward does not pay for.

**Training with retrieval in the loop buys robustness that scale does
not.** Our search-trained 35B is the only policy measured that does not
degrade when given functioning retrieval, and it tops the "fair fight" column
— no market access, search-on retrieval — at 0.254, statistically tied with
Claude Opus 4.5 (0.256) and ahead of Sonnet, Gemini Pro, and Flash, at
roughly one-hundredth of frontier inference cost.

A second thread of contributions is methodological, and we believe as
consequential as the findings. All headline evaluations run through the
training platform's own serving stack via a per-rollout results webhook,
after we measured a 94%-vs-48% behavioral gap between platform and local
serving of identical weights. Training curves are reported
difficulty-adjusted, after showing that raw single-epoch reward curves
are batch-difficulty noise (r=0.58). Predictions for two runs were
pre-registered by commit before training began (9.5 of 12 confirmed).
And a trace-level audit — reading rollouts, not dashboards — twice caught
silent failures that every aggregate metric missed: a search backend
returning empty results at 93–100% for an entire evaluation campaign
while reporting HTTP 200, and a reward-leakage channel through unparsed
publication dates that briefly let a policy earn near-perfect reward by
retrieving a resolved outcome. Both are documented with corrections in an
append-only ledger; we argue this discipline, borrowed from the
measurement-validity literature, should be standard for agentic RL
evaluation.

We release the environment, all per-rollout evaluation data, trained
adapters, the retrieved-context corpus, and the correction ledger.
