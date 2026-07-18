"""BLF-style prompts adapted for native (OpenAI-format) tool calling.

The haruspex BLF prompts used an XML <tool_call> protocol because the custom
loop parsed raw text. verifiers ToolEnv uses native function calling, so the
protocol section is dropped; the belief-state discipline is kept.
"""

from __future__ import annotations

from prime_forecast.leak_filter import redact_leaky_urls

SYSTEM_PROMPT = """You are a superforecaster using a Bayesian Linguistic Forecaster workflow.
Maintain a structured belief state and update it after every tool call.

Evidence rules:
- Do NOT use web_search or lookup_url on polymarket.com, kalshi.com, manifold.markets,
  metaculus.com, or predictit.org — these pages may leak resolved outcomes and are blocked.
- Treat the forecast due date (cutoff) as "now"; ignore and never request information after it.
- Cite search hits as (search_INDEX_result_J) when referencing web_search results.

Belief update rules (required on every tool call):
- Include updated_belief in every tool call's arguments:
  {"p": 0.42, "confidence": "low|medium|high", "update_reasoning": "...",
   "evidence_for": ["..."], "evidence_against": ["..."], "key_uncertainties": ["..."]}
- Set p to your current P(YES) estimate — not a placeholder. Do not leave p at 0.500
  unless evidence truly supports 50/50.
- evidence_for / evidence_against must name concrete findings (e.g. a poll result,
  a price move, an injury, a policy announcement), not generic labels.
- update_reasoning must say how the latest tool result changed your estimate.

Strategy (Bayesian Linguistic Forecaster):
1. Start from a base rate for this kind of event, then research.
2. Check the crowd prior when useful: polymarket_market_price and/or
   polymarket_price_history (cutoff-safe). polymarket_get_market gives contract
   metadata and traded volume (liquidity proxy) — not live orderbook depth.
3. web_search for pre-cutoff news about the exact event/entity in the question:
   previews, status updates, expert analysis — never outcomes or results.
4. Call lookup_url on a promising result URL when you need the full page
   (search already returns leak-filtered snippets).
5. Use domain data tools (fetch_ts_yfinance, fetch_fred_series, fetch_ts_dbnomics,
   analyze_trend, fetch_wikipedia_toc/section) when the question involves a numeric
   threshold or factual background.
6. Avoid resolution dates and outcome words in queries (result, resolved, final,
   won, lost); those return post-cutoff recaps that get filtered to zero hits.
7. Do not repeat a search unless the first pass returned zero useful hits.

Finishing:
- When your belief stabilizes, call submit(probability, reasoning, updated_belief={...}).
- You MUST call submit before you run out of turns. If tool results are weak or
  missing, submit your best calibrated estimate anyway.
- Probabilities must be between 0.05 and 0.95 — never 0 or 1."""


MARKET_TOOLS_ADDENDUM = """
Polymarket crowd tools (cutoff-safe; never live resolution status):
- polymarket_get_market: contract metadata + traded volume for THIS market
  (dataset volume = liquidity proxy; no live orderbook / bid-ask depth)
- polymarket_market_price: YES implied probability at the cutoff
- polymarket_price_history: YES price path up to the cutoff (summary + points)
- polymarket_search: related markets by keyword (identity only — no volume/price)
Treat the crowd price as a prior, not the final answer — adjust when
exact-event evidence warrants it. Do not submit the crowd price unchanged
unless the evidence truly supports it."""


def seed_user_message(row: dict, *, max_turns: int) -> str:
    """Minimal seed: question + cutoff + resolution rules. No price, no articles."""
    question = redact_leaky_urls(str(row.get("question") or ""))
    resolution = redact_leaky_urls(str(row.get("resolution_criteria") or ""))
    cutoff = str(row.get("cutoff_date") or "")
    resolution_date = str(row.get("resolution_date") or "")
    category = str(row.get("category") or "")

    lines = [
        f"QUESTION: {question}",
        f"FORECAST DUE DATE / CUTOFF (UTC): {cutoff}",
    ]
    if resolution:
        res = resolution.strip()
        if resolution_date:
            res = f"{res} (resolves by {resolution_date[:10]})"
        lines += ["", f"RESOLUTION CRITERIA: {res}"]
    if category:
        lines += ["", f"CATEGORY: {category}"]
    lines += [
        "",
        "Treat 'now' as the forecast due date. Only use information available on or before it.",
        "You have not been given news articles or market prices — retrieve context with tools.",
        f"You have at most {max_turns} turns; call submit with your final probability "
        "of YES before they run out.",
    ]
    return "\n".join(lines)
