# prime-forecast

Agentic forecasting RLVR environment for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) / Prime Intellect Hosted Training.

A multi-turn agent receives a resolved Polymarket binary question with a historical cutoff date, researches it with live cutoff-safe tools (Firecrawl/Brave/Tavily/AgentCore/Exa web search, yfinance, FRED, DBnomics, Wikipedia-as-of-cutoff, trend analysis), maintains a BLF-style structured belief state, and submits `P(YES)`. Reward is positive-shifted Brier against the real outcome:

```
reward = 1 - (p - y)^2        (0.55 soft no-submit)
```

Missing submission scores 0.55 (below always-0.5) so stalling is not a safe reward; optional protocol bonus defaults to off. Cutoff-safe Polymarket crowd tools (`include_market_tools=True` by default) expose price-at-cutoff, price history, and dataset volume; `market_brier` remains an eval metric. Use the zero-advantage pre-batch filter to drop collapsed crowd-copy groups.

## Leak safety

1. Provider date filter (Exa `endPublishedDate`) + client-side publish-date drop
2. Prediction-market domain blocklist (polymarket, kalshi, manifold, metaculus, predictit)
3. Heuristic post-cutoff date/outcome-language filter on all retrieved text
4. Bedrock LLM leak filter (Claude Haiku by default) with per-result KEEP/DROP votes
5. All time-series/Wikipedia tools clamp end dates to the cutoff; model-supplied end dates can only tighten, never extend

## Required Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `PF_SEARCH_BACKEND` | recommended | `firecrawl` \| `brave` \| `tavily` \| `agentcore` \| `exa` \| `none` |
| `FIRECRAWL_API_KEY` | for firecrawl | Firecrawl Search API (tbs date-range cutoff; 1 credit/search) |
| `BRAVE_API_KEY` | for brave | Brave Search API (cheap; leak filter supplies cutoff) |
| `TAVILY_API_KEY` | for tavily | Tavily Search (free tier; local eval / small smokes) |
| `AGENTCORE_GATEWAY_URL` | for agentcore | MCP gateway URL from `scripts/setup_agentcore_search.py` |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` / `AWS_PROFILE` | for agentcore | SigV4 for Gateway MCP (`us-east-1`) + Bedrock leak filter |
| `AWS_BEARER_TOKEN_BEDROCK` | optional | Bedrock Runtime only (leak filter); **not** enough for AgentCore |
| `EXA_API_KEY` | for exa | Exa web search |
| `FRED_API_KEY` | optional | `fetch_fred_series` tool |
| `PF_LLM_FILTER` | optional | force LLM filter on/off (`1`/`0`); auto-detects AWS creds by default |

## Environment args

| Arg | Default | Description |
|---|---|---|
| `dataset_path` | packaged data | JSONL file or dir with `{split}.jsonl` |
| `split` / `eval_split` | `train` / `val` | dataset splits |
| `num_examples` / `num_eval_examples` | `-1` | truncate splits |
| `max_turns` | `8` | BLF T_max |
| `max_web_searches` | `2` | cap live `web_search` calls per rollout |
| `include_market_tools` | `true` | expose cutoff-safe Polymarket price/history/volume tools |
| `crowd_copy_penalty` | `0.20` | subtract from reward when `|p - crowd| ≤ crowd_copy_eps` |
| `crowd_copy_eps` | `0.02` | absolute distance to `price_at_cutoff` treated as a copy |
| `protocol_bonus_weight` | `0.0` | optional BLF shaping (off by default) |

## Dataset

Built from the Polymarket Gamma + CLOB APIs by `scripts/build_dataset.py` (repo root): resolved binary YES/NO markets, volume/price filters, per-market cutoff 7 days before resolution, temporal train/val/test splits. Run with `--install` to copy splits into the package data directory.
