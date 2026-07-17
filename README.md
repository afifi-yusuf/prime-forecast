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

# 3. Local eval against a cheap model (needs EXA_API_KEY; AWS creds for leak filter)
cp secrets.env.example secrets.env  # fill in
prime eval run prime-forecast -m openai/gpt-4.1-mini -n 10

# 4. Smoke train, then the main run (Hosted Training, private beta)
prime train submit configs/rl/prime-forecast-9b.toml
prime train submit configs/rl/prime-forecast-35b.toml

# 5. Deploy the adapter
prime train download <job-id> --output checkpoints/
prime deployments create <adapter-id>
```

## Reward

`reward = 1 - (p - y)^2` (positive-shifted Brier; 0.75 for an always-0.5 baseline, 0 on missing submission) + 0.05 × BLF protocol bonus. `market_brier` (crowd price at cutoff) is tracked as an eval-only baseline — the agent cannot see the crowd price unless `include_market_tools=true`.

## Leakage controls

Temporal eligibility (markets resolving after the Qwen3.5 release window), Exa hard date filters, prediction-market domain blocklist, heuristic post-cutoff date/outcome filters, Bedrock Haiku LLM leak filter, cutoff-clamped time-series/Wikipedia tools, and temporal train/val/test splits.
