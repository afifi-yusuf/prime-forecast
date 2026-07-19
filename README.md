# prime-forecast

Agentic forecasting RLVR on [Prime Intellect Lab](https://docs.primeintellect.ai/): a multi-turn tool-use agent researches resolved Polymarket questions (context acquisition folded into the RL loop — no pre-baked context in the prompt) and is trained with LoRA GRPO on Brier-score rewards via Hosted Training.

Combines:
- **Outcome-based RLVR for forecasting** — [Turtel et al. 2025](https://arxiv.org/abs/2505.17989)
- **BLF iterative tool-use harness with a structured belief state** — [arXiv:2604.18576](https://arxiv.org/abs/2604.18576)
- **verifiers + prime-rl** — Prime Intellect's environment/training stack

Ported and adapted from the `haruspex` research codebase (TRL/vLLM custom loop → verifiers `StatefulToolEnv`).

## Layout

```
environments/prime_forecast/   # verifiers environment package (see its README)
scripts/build_dataset.py       # Polymarket Gamma/CLOB -> JSONL train/val/test
scripts/analyze_dataset.py     # dataset stats (categories, outcomes, calibration)
data/                          # built dataset splits (JSONL)
configs/eval/                  # prime eval configs
configs/rl/                    # Hosted Training configs (9B smoke, 35B main)
tests/                         # cutoff, leak filter, reward, submit-parsing tests
```

## Quickstart

```bash
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e environments/prime_forecast pytest pytest-asyncio
uv tool install prime

# 1. Build the dataset (public Polymarket APIs, no key needed)
.venv/bin/python scripts/build_dataset.py --target-total 4000 --install

# 2. Run tests
.venv/bin/python -m pytest tests/ -q

# 3. Search + secrets (AgentCore Web Search recommended; Exa optional)
cp secrets.env.example secrets.env
# Create AgentCore Gateway (us-east-1; needs IAM keys, not Bedrock bearer alone):
.venv/bin/python scripts/setup_agentcore_search.py
# Paste AGENTCORE_GATEWAY_URL into secrets.env and set PF_SEARCH_BACKEND=agentcore

# 4. Local eval (AWS IAM for AgentCore + Bedrock leak filter)
prime eval run prime-forecast -m openai/gpt-4.1-mini -n 10

# 5. Smoke train, then the main run (Hosted Training, private beta)
prime train submit configs/rl/prime-forecast-9b.toml
prime train submit configs/rl/prime-forecast-35b.toml

# 6. Deploy the adapter
prime train download <job-id> --output checkpoints/
prime deployments create <adapter-id>
```

## Web search

Backends via `PF_SEARCH_BACKEND`: **`tavily`** (free tier; good for local eval), **`agentcore`** (AWS, us-east-1, ~$7/1k; needs IAM gateway), **`exa`**. RL configs set `max_web_searches=2` to cap spend.


## Reward

`reward = 1 - (p - y)^2` (positive-shifted Brier). Missing submission scores **0.55** (below always-0.5) so stalling loses to mediocre submits. Optional BLF protocol bonus defaults to **off** (`protocol_bonus_weight=0`). Cutoff-safe Polymarket crowd tools are on by default (`include_market_tools=true`); `market_brier` remains an eval metric.

## Leakage controls

Temporal eligibility (markets resolving after the Qwen3.5 release window), Exa hard date filters, prediction-market domain blocklist, heuristic post-cutoff date/outcome filters, Bedrock Haiku LLM leak filter, cutoff-clamped time-series/Wikipedia tools, and temporal train/val/test splits.
