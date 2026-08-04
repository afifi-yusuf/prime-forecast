# Frontier-model comparison panel (planned)

Both reference papers benchmark against closed-source frontier models —
Turtel et al. vs OpenAI o1 + market prices; Mantic vs Claude Opus 4, GPT-5,
Gemini 3 Pro, Grok 4 (all given identical frozen research context). Ours is
stronger in one dimension: frontier models run through the **full agentic
loop** (same tools, search, leak filter, turn budget) rather than pre-baked
context — testing agentic research-and-forecast, which neither paper did.

## Protocol

- Test split (160 questions), same env args as the platform eval:
  `split=test, max_turns=8, max_web_searches=2, include_market_tools=true`
- Search: SearXNG (free; start VM first: `az vm start -g pf-eval-rg -n pf-searx`)
- Each frontier model via its native API serving (Prime inference provider);
  our trained model's row comes from the platform eval (results/RESULTS.md)
- Metrics via `scripts/analyze_eval.py`: Brier + 95% CI, ECE, trading P&L,
  |p − crowd| distribution, submit rate
- Report alongside: crowd baseline (0.194 submitted-subset / 0.191 all),
  base Qwen3.5-35B (row C), trained model (platform eval)

## Panel (est. inference cost for 160 questions, ~30k in / 2k out tokens per rollout)

| model | prime inference id | est. cost |
|---|---|---|
| Claude Opus 4.5 | anthropic/claude-opus-4.5 | ~$30 |
| Claude Sonnet 5 | anthropic/claude-sonnet-5 | ~$12 |
| Gemini 3 Pro | google/gemini-3.1-pro-preview | ~$14 |
| DeepSeek V4 Pro | deepseek/deepseek-v4-pro | ~$12 |
| GPT-5 class | (confirm availability in `prime inference models`) | ~$15 |

Suggested order: Sonnet 5 + Gemini 3 Pro first (validate plumbing, see the bar),
then Opus/GPT-5/DeepSeek if the comparison earns its budget.

## Command template

```bash
az vm start -g pf-eval-rg -n pf-searx   # if deallocated
set -a && source secrets.env && set +a
export PF_SEARCH_BACKEND=searxng SEARXNG_URL=http://20.110.86.90:8080
prime eval run configs/eval/test-conditions.toml -m anthropic/claude-sonnet-5 \
  --provider prime --plain --disable-tui
.venv/bin/python scripts/analyze_eval.py "sonnet-5"=<results.jsonl>
```

Fairness notes for the paper: identical env/tools/turn budget for all models;
native serving per model; temperature = provider defaults; our model evaluated
through the training platform (one-sentence methods note on serving asymmetry).
