# prime-forecast

**Agentic RLVR forecasting on resolved Polymarket questions.** A
multi-turn tool-use agent researches real, resolved prediction-market
questions — web search, financial/economic time series, optionally the
market's own price — under strict temporal leak filtering, and is
trained with LoRA GRPO on Brier-score rewards via Prime Intellect
Hosted Training.

**The environment is live on the Prime Intellect hub:
[`yafifi/prime-forecast`](https://app.primeintellect.ai/dashboard/environments/yafifi/prime-forecast)** —
evaluate any model in it, or train against it.

**Headline result**: outcome-based RL takes an open Qwen3.5-35B-A3B to
parity with Claude Opus 4.5 at evidence-based forecasting (working
search, market price withheld) at roughly 1/100th the inference cost,
with 30–40% calibration gains and coverage saturating at ~100%. Full
results, figures, and per-rollout archives: **[results/RESULTS.md](results/RESULTS.md)**.

Unlike prior RLVR forecasting work that freezes research context before
training (headlines in the prompt, or a shared pre-generated research
phase), context here is acquired *by the agent at rollout time* through
cutoff-filtered tools — research is part of the measured skill, and the
dataset is continuously renewable from newly resolved markets.

Builds on:
- **Outcome-based RLVR for forecasting** — [Turtel et al. 2025](https://arxiv.org/abs/2505.17989)
- **BLF iterative tool-use harness with a structured belief state** — [arXiv:2604.18576](https://arxiv.org/abs/2604.18576)
- **verifiers + prime-rl** — Prime Intellect's environment/training stack

## Layout

```
environments/prime_forecast/   # verifiers environment package (published to the hub)
  prime_forecast/data/         # packaged dataset splits (train/test JSONL)
scripts/build_dataset.py       # Polymarket Gamma/CLOB -> JSONL splits
scripts/analyze_dataset.py     # dataset stats (categories, outcomes, calibration)
scripts/make_figures.py        # regenerates all figures from results/ archives
configs/eval/                  # prime eval configs (frontier panel, ablations)
configs/rl/                    # Hosted Training configs (the five runs + expansions)
results/                       # RESULTS.md + per-rollout archives + figures
docs/                          # run registry, eval plans, framing notes
tests/                         # cutoff, leak filter, reward, submit-parsing tests
```

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e environments/prime_forecast pytest pytest-asyncio
uv tool install prime

# 1. Build a fresh dataset (public Polymarket APIs, no key needed)
.venv/bin/python scripts/build_dataset.py --target-total 4000 --install

# 2. Run tests
.venv/bin/python -m pytest tests/ -q

# 3. Secrets: search backend + Bedrock leak filter
cp secrets.env.example secrets.env   # then fill in your keys (see Web search below)

# 4. Local eval of any model
prime eval run prime-forecast -m <model> -n 10

# 5. Hosted Training (see configs/rl/ for the runs used in the paper)
prime train configs/rl/prime-forecast-35b-v2.toml
```

## The environment

Up to 10 turns; tools: `web_search` (≤3/rollout, leak-filtered),
`lookup_url`, `fetch_ts_yfinance` / `fetch_fred_series` /
`fetch_ts_dbnomics` (cutoff-truncated series), `fetch_wikipedia_toc` /
`fetch_wikipedia_section`, `analyze_trend`, optional Polymarket crowd
tools (`include_market_tools`), and `submit`. The system prompt
maintains an explicit belief state — every tool call carries an updated
probability and evidence lists, so each forecast is an auditable update
trajectory.

## Reward

`reward = 1 - (p - y)^2` (positive-shifted Brier; strictly proper).
Missing submission scores **0.55** (below always-0.5) so stalling loses
to mediocre submits. Optional anti-copy penalty for market-visible
training (`crowd_copy_penalty`, `crowd_copy_eps`) deducts reward near
the crowd price. `market_brier` is logged as an eval metric.

## Web search

Backends via `PF_SEARCH_BACKEND`: `agentcore` (AWS gateway; used for
the paper's search-on runs), `firecrawl`, `tavily`, `brave`, `exa`,
`google_cse`, `searxng` — with a comma-separated fallback chain.
Results are cached post-leak-filter (`PF_SEARCH_CACHE_DIR`): repeat
queries cost zero credits and the cache is a publishable corpus of
every (query, cutoff) → filtered-context pair the agent saw. Empty
result sets are never cached, and result-count health checks guard
against silent backend failures (`scripts/check_search.sh`).

## Leakage controls

Temporal eligibility (all questions resolve after the base model's
release window; test resolutions strictly postdate all training data),
prediction-market domain blocklist, heuristic post-cutoff date/fact
filters, Bedrock Haiku LLM KEEP/DROP filter, cutoff-clamped
time-series/Wikipedia tools, and temporal train/test splits. Residual
bound: archived market data supports day-level ordering only.
