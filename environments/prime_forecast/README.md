# prime-forecast

Agentic forecasting RLVR environment for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) / Prime Intellect Hosted Training.

A multi-turn agent receives a resolved Polymarket binary question with a historical cutoff date, researches it with live cutoff-safe tools (Exa web search, yfinance, FRED, DBnomics, Wikipedia-as-of-cutoff, trend analysis), maintains a BLF-style structured belief state, and submits `P(YES)`. Reward is positive-shifted Brier against the real outcome:

```
reward = 1 - (p - y)^2        (0 if the agent never submits)
```

plus a small (weight 0.05) protocol-adherence bonus. The crowd price at cutoff is kept as an eval-only baseline metric (`market_brier`) and, by default, is **not** available to the agent (`include_market_tools=False`) — crowd-price copying collapses GRPO advantage groups.

## Leak safety

1. Exa `endPublishedDate <= cutoff` + client-side publish-date drop
2. Prediction-market domain blocklist (polymarket, kalshi, manifold, metaculus, predictit)
3. Heuristic post-cutoff date/outcome-language filter on all retrieved text
4. Bedrock LLM leak filter (Claude Haiku by default) with per-result KEEP/DROP votes
5. All time-series/Wikipedia tools clamp end dates to the cutoff; model-supplied end dates can only tighten, never extend

## Required Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `EXA_API_KEY` | yes | live web search |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | recommended | Bedrock LLM leak filter + summarizer |
| `FRED_API_KEY` | optional | `fetch_fred_series` tool |
| `PF_LLM_FILTER` | optional | force LLM filter on/off (`1`/`0`); auto-detects AWS creds by default |

## Environment args

| Arg | Default | Description |
|---|---|---|
| `dataset_path` | packaged data | JSONL file or dir with `{split}.jsonl` |
| `split` / `eval_split` | `train` / `val` | dataset splits |
| `num_examples` / `num_eval_examples` | `-1` | truncate splits |
| `max_turns` | `8` | BLF T_max |
| `include_market_tools` | `false` | expose crowd-price tools |
| `protocol_bonus_weight` | `0.05` | BLF shaping weight |

## Dataset

Built from the Polymarket Gamma + CLOB APIs by `scripts/build_dataset.py` (repo root): resolved binary YES/NO markets, volume/price filters, per-market cutoff 7 days before resolution, temporal train/val/test splits. Run with `--install` to copy splits into the package data directory.
