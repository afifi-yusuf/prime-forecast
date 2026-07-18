"""ForecastEnv — multi-turn agentic forecasting with Brier RLVR.

The agent researches a resolved Polymarket question via live, cutoff-safe
tools and submits P(YES). Reward is positive-shifted Brier vs the outcome:
    reward = 1 - (p - y)^2      (0.75 soft no-submit = always-0.5 baseline)

Crowd-price tools (cutoff-safe CLOB history + dataset volume) are on by default.
Treat the crowd as a prior; zero-advantage groups are handled by the
zero_advantage pre-batch filter. Live Gamma payloads are never returned —
they contain resolution/winner fields.
"""

import json
import re
from importlib import resources
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_forecast.belief import BeliefState, BeliefUpdate, SearchResult, SearchStore
from prime_forecast.cutoff import find_dates_after, parse_ts
from prime_forecast.leak_filter import (
    domain_blocked,
    filter_page_content,
    sanitize_search_query,
)
from prime_forecast.leak_filter import summarize_results as blf_summarize
from prime_forecast import datatools, polymarket, prompts, search

MIN_PROB = 0.05
MAX_PROB = 0.95
# Soft no-submit: same reward as always predicting 0.5 under Brier.
# Hard 0 forced models to spam submit ~0.5 (smoke-run collapse).
NO_SUBMIT_REWARD = 0.75

_PROB_TAG_RE = re.compile(r"<probability>\s*([01]?\.?\d+)\s*</probability>", re.IGNORECASE)


def _clamp_prob(value) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= p <= 1.0):
        return None
    return max(MIN_PROB, min(MAX_PROB, p))


def _message_text(msg) -> str:
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, list):
        content = " ".join(str(getattr(c, "text", c)) for c in content)
    return str(content or "")


def extract_probability(state: vf.State) -> float | None:
    """Submitted probability, falling back to a <probability> tag in the last message."""
    p = state.get("submitted_prob")
    if p is not None:
        return _clamp_prob(p)
    completion = state.get("completion") or []
    for msg in reversed(list(completion)):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "assistant":
            continue
        m = _PROB_TAG_RE.search(_message_text(msg))
        if m:
            return _clamp_prob(m.group(1))
        break
    return None


def _parse_info(info) -> dict:
    if isinstance(info, str):
        try:
            return json.loads(info)
        except json.JSONDecodeError:
            return {}
    return dict(info or {})


# ------------------------------------------------------------------ rewards

async def forecast_reward(state: vf.State, info) -> float:
    """Main reward: positive-shifted Brier; soft no-submit (= always-0.5).

    r = 1 - (p - y)^2 when a probability is produced; NO_SUBMIT_REWARD (0.75)
    when missing. Matches the always-0.5 baseline so RL is not forced to spam
    mid-probability submits (Mantic / Turtel use proper scores on valid p̂).
    """
    p = extract_probability(state)
    if p is None:
        return NO_SUBMIT_REWARD
    meta = _parse_info(info)
    y = 1.0 if int(meta.get("outcome", 0)) == 1 else 0.0
    return 1.0 - (p - y) ** 2


async def protocol_bonus(state: vf.State) -> float:
    """Optional BLF protocol shaping (off by default; weight 0.0).

    Kept for ablations. Smoke runs showed it encouraged web_search→submit
    collapse without improving calibration.
    """
    belief: BeliefState | None = state.get("belief")
    checks = [
        bool(state.get("submitted")),                       # explicit submit call
        int(state.get("research_tool_calls", 0)) >= 1,       # actually researched
        belief is not None and belief.step >= 1,             # belief updated at least once
        belief is not None and abs(belief.p - 0.5) > 0.001,  # moved off the placeholder
        belief is not None and bool(
            belief.evidence_for or belief.evidence_against or belief.key_uncertainties
        ),
    ]
    return sum(checks) / len(checks)


async def brier_score(state: vf.State, info) -> float:
    """Metric: raw Brier ((p-y)^2); 1.0 when no probability was produced."""
    p = extract_probability(state)
    meta = _parse_info(info)
    y = 1.0 if int(meta.get("outcome", 0)) == 1 else 0.0
    if p is None:
        return 1.0
    return (p - y) ** 2


async def market_brier(info) -> float:
    """Metric: crowd baseline Brier from price_at_cutoff (-1 when missing)."""
    meta = _parse_info(info)
    price = meta.get("price_at_cutoff")
    if price is None:
        return -1.0
    y = 1.0 if int(meta.get("outcome", 0)) == 1 else 0.0
    return (float(price) - y) ** 2


async def submitted(state: vf.State) -> float:
    """Metric: 1.0 when the agent explicitly called submit."""
    return 1.0 if state.get("submitted") else 0.0


async def predicted_prob(state: vf.State) -> float:
    """Metric: the submitted probability (-1 when missing)."""
    p = extract_probability(state)
    return p if p is not None else -1.0


# -------------------------------------------------------------- environment

_RESEARCH_TOOLS = frozenset({
    "web_search", "lookup_url", "fetch_ts_yfinance",
    "fetch_fred_series", "fetch_ts_dbnomics", "fetch_wikipedia_toc",
    "fetch_wikipedia_section", "analyze_trend", "polymarket_get_market",
    "polymarket_search", "polymarket_market_price", "polymarket_price_history",
})


class ForecastEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        *,
        include_market_tools: bool = True,
        max_turns: int = 8,
        max_web_searches: int = 2,
        **kwargs,
    ):
        super().__init__(tools=None, max_turns=max_turns, **kwargs)
        self.include_market_tools = include_market_tools
        self.max_web_searches = max(0, int(max_web_searches))

        self.add_tool(self.web_search, args_to_skip=["state"])
        self.add_tool(self.lookup_url, args_to_skip=["state"])
        self.add_tool(self.fetch_ts_yfinance, args_to_skip=["state"])
        self.add_tool(self.fetch_fred_series, args_to_skip=["state"])
        self.add_tool(self.fetch_ts_dbnomics, args_to_skip=["state"])
        self.add_tool(self.fetch_wikipedia_toc, args_to_skip=["state"])
        self.add_tool(self.fetch_wikipedia_section, args_to_skip=["state"])
        self.add_tool(self.analyze_trend, args_to_skip=["state"])
        if include_market_tools:
            self.add_tool(self.polymarket_get_market, args_to_skip=["state"])
            self.add_tool(self.polymarket_search, args_to_skip=["state"])
            self.add_tool(self.polymarket_market_price, args_to_skip=["state"])
            self.add_tool(self.polymarket_price_history, args_to_skip=["state"])
        self.add_tool(self.submit, args_to_skip=["state"])

    # -- rollout lifecycle ---------------------------------------------------

    async def setup_state(self, state: vf.State) -> vf.State:
        meta = _parse_info(state.get("info"))
        # ctx is what tools may see. It deliberately EXCLUDES the outcome.
        state["ctx"] = {
            "question": meta.get("question", ""),
            "resolution_criteria": meta.get("resolution_criteria", ""),
            "resolution_date": meta.get("resolution_date"),
            "cutoff_date": meta.get("cutoff_date", ""),
            "market_id": meta.get("market_id"),
            "slug": meta.get("slug"),
            "token_id": meta.get("token_id"),
            "category": meta.get("category"),
            "price_at_cutoff": meta.get("price_at_cutoff"),
            "volume": meta.get("volume"),
        }
        state["belief"] = BeliefState()
        state["store"] = SearchStore()
        state["submitted"] = False
        state["submitted_prob"] = None
        state["research_tool_calls"] = 0
        state["web_search_calls"] = 0
        await super().setup_state(state)
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        # Merge the BLF belief update centrally, then inject hidden state.
        belief: BeliefState = state.get("belief") or BeliefState()
        update = tool_args.pop("updated_belief", None)
        if isinstance(update, str):
            try:
                update = json.loads(update)
            except json.JSONDecodeError:
                update = None
        if isinstance(update, dict):
            belief.merge_update(update)
            belief.compact_if_needed()
        state["belief"] = belief
        if tool_name in _RESEARCH_TOOLS:
            state["research_tool_calls"] = int(state.get("research_tool_calls", 0)) + 1
        tool_args["state"] = state
        return tool_args

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs) -> vf.Messages:
        tool_messages = await super().env_response(messages, state, **kwargs)
        if state.get("submitted"):
            # submit() ended the episode: return the ack as the final env
            # response so the rollout terminates without another model turn.
            state["final_env_response"] = tool_messages
        return tool_messages

    @vf.stop(priority=50)
    async def forecast_submitted(self, state: vf.State) -> bool:
        return bool(state.get("submitted"))

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _belief_payload(state: vf.State) -> dict:
        belief: BeliefState = state.get("belief") or BeliefState()
        return belief.to_dict()

    # -- research tools ----------------------------------------------------

    async def web_search(
        self, query: str, num_results: int = 10,
        updated_belief: BeliefUpdate | None = None, state: dict | None = None,
    ) -> str:
        """Search the pre-cutoff web. Post-cutoff results are filtered automatically.

        Backend is selected by PF_SEARCH_BACKEND (brave | tavily | exa | agentcore).
        Each rollout is limited to max_web_searches calls to control API spend.

        Args:
            query: Search query. Avoid outcome words (result, resolved, final, won, lost).
            num_results: Number of results to return (max 20; AgentCore caps at 25).
            updated_belief: Your updated belief state (p, confidence, update_reasoning, evidence_for, evidence_against, key_uncertainties).

        Returns:
            JSON with search_index, result snippets, and your current belief state.
        """
        ctx = state["ctx"]
        used = int(state.get("web_search_calls", 0))
        if used >= self.max_web_searches:
            return json.dumps({
                "error": (
                    f"web_search budget exhausted ({self.max_web_searches} per rollout). "
                    "Call lookup_url on promising hit URLs, use data/market tools, or submit."
                ),
                "web_search_calls": used,
                "max_web_searches": self.max_web_searches,
                "belief": self._belief_payload(state),
            })
        q = sanitize_search_query(str(query or ""))
        if not q:
            return json.dumps({"error": "query empty after sanitization"})
        try:
            raw, parsed, _debug = await search.web_search(
                q, cutoff_date=ctx["cutoff_date"],
                num_results=int(num_results), question=ctx["question"],
            )
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": str(e)})

        state["web_search_calls"] = used + 1
        results = [
            SearchResult(index=i, title=p["title"], url=p["url"],
                         snippet=p.get("snippet", ""), body=p.get("body", ""))
            for i, p in enumerate(parsed)
        ]
        store: SearchStore = state["store"]
        search_index = store.add_search(q, results)

        belief: BeliefState = state["belief"]
        if q not in belief.searches_tried:
            belief.searches_tried.append(q)

        remaining = self.max_web_searches - state["web_search_calls"]
        return json.dumps({
            "search_index": search_index,
            "query": q,
            "backend": search.search_backend(),
            "num_results": len(results),
            "results": [r.to_snippet_dict() for r in results],
            "web_search_calls": state["web_search_calls"],
            "web_searches_remaining": remaining,
            "belief": self._belief_payload(state),
            "hint": "Call lookup_url on a promising result URL to read the full page.",
        }, default=str)

    async def lookup_url(
        self, url: str, updated_belief: BeliefUpdate | None = None, state: dict | None = None,
    ) -> str:
        """Fetch and summarize a specific URL (prediction-market sites are blocked).

        Args:
            url: The URL to fetch. Must not be a prediction-market page.
            updated_belief: Your updated belief state.

        Returns:
            JSON with a leak-filtered summary of the page.
        """
        ctx = state["ctx"]
        u = str(url or "").strip()
        if not u:
            return json.dumps({"error": "url required"})
        if domain_blocked(u):
            return json.dumps(
                {"error": "URL blocked: prediction market pages may leak resolved outcomes"})
        try:
            text = await search.fetch_url_text(u)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"fetch failed: {e}"})

        cutoff_dt = parse_ts(ctx["cutoff_date"])
        if cutoff_dt is not None:
            leaks = find_dates_after(text, cutoff_dt)
            if leaks:
                return json.dumps(
                    {"error": f"page rejected: contains dates after cutoff ({leaks[0]})"})

        filtered, flog = await filter_page_content(
            text, ctx["cutoff_date"], question=ctx["question"], url=u,
        )
        if not filtered:
            return json.dumps({"error": f"page rejected by leak filter ({flog.get('mode')})"})

        summary, _debug = await blf_summarize(
            f"{u}\n{filtered}",
            question=ctx["question"],
            cutoff_date=ctx["cutoff_date"],
            resolution_criteria=ctx.get("resolution_criteria") or "",
        )
        return json.dumps({"url": u, "summary": summary,
                           "belief": self._belief_payload(state)}, default=str)

    async def fetch_ts_yfinance(
        self, ticker: str, end_date: str | None = None, state: dict | None = None,
    ) -> str:
        """Fetch daily closing prices for a ticker, up to the cutoff date.

        Args:
            ticker: Ticker symbol (e.g. SPY, BTC-USD, ^GSPC).
            end_date: Optional end date (YYYY-MM-DD); clamped to the cutoff.

        Returns:
            JSON with sampled daily closes up to the cutoff.
        """
        ctx = state["ctx"]
        t = str(ticker or "").strip()
        if not t:
            return json.dumps({"error": "ticker required"})
        return await datatools.fetch_yfinance(t, ctx["cutoff_date"], end_date=end_date)

    async def fetch_fred_series(
        self, series_id: str, end_date: str | None = None, limit: int = 24,
        state: dict | None = None,
    ) -> str:
        """Fetch a FRED macro series (UNRATE, CPIAUCSL, FEDFUNDS, DGS10, ...) up to the cutoff.

        Args:
            series_id: FRED series id (e.g. UNRATE, CPIAUCSL, FEDFUNDS).
            end_date: Optional end date (YYYY-MM-DD); clamped to the cutoff.
            limit: Maximum rows to display.

        Returns:
            JSON with series metadata and observations up to the cutoff.
        """
        ctx = state["ctx"]
        sid = str(series_id or "").strip()
        if not sid:
            return json.dumps({"error": "series_id required (e.g. UNRATE, CPIAUCSL, FEDFUNDS)"})
        return await datatools.fetch_fred(
            sid, ctx["cutoff_date"], end_date=end_date, limit=int(limit))

    async def fetch_ts_dbnomics(
        self, url: str, end_date: str | None = None, state: dict | None = None,
    ) -> str:
        """Fetch an international macro series from DBnomics, up to the cutoff.

        Args:
            url: db.nomics.world provider/dataset/series path or URL.
            end_date: Optional end date (YYYY-MM-DD); clamped to the cutoff.

        Returns:
            JSON with observations up to the cutoff.
        """
        ctx = state["ctx"]
        u = str(url or "").strip()
        if not u:
            return json.dumps({"error": "url required (db.nomics.world provider/dataset/series)"})
        return await datatools.fetch_dbnomics(u, ctx["cutoff_date"], end_date=end_date)

    async def fetch_wikipedia_toc(
        self, url: str, end_date: str | None = None, state: dict | None = None,
    ) -> str:
        """Get the table of contents of a Wikipedia article as of the cutoff date.

        Args:
            url: Wikipedia article URL or title.
            end_date: Optional end date (YYYY-MM-DD); clamped to the cutoff.

        Returns:
            JSON with the article's section headings at the pre-cutoff revision.
        """
        ctx = state["ctx"]
        u = str(url or "").strip()
        if not u:
            return json.dumps({"error": "url required (Wikipedia article URL or title)"})
        return await datatools.fetch_wikipedia_toc(u, ctx["cutoff_date"], end_date=end_date)

    async def fetch_wikipedia_section(
        self, url: str, section: str, end_date: str | None = None,
        state: dict | None = None,
    ) -> str:
        """Read one section of a Wikipedia article as of the cutoff date.

        Args:
            url: Wikipedia article URL or title.
            section: Section heading from fetch_wikipedia_toc, or 'introduction'.
            end_date: Optional end date (YYYY-MM-DD); clamped to the cutoff.

        Returns:
            JSON with the section text at the pre-cutoff revision.
        """
        ctx = state["ctx"]
        u = str(url or "").strip()
        sec = str(section or "").strip()
        if not u:
            return json.dumps({"error": "url required"})
        if not sec:
            return json.dumps({"error": "section required (heading name or 'introduction')"})
        return await datatools.fetch_wikipedia_section(
            u, sec, ctx["cutoff_date"], end_date=end_date)

    async def analyze_trend(
        self, source: str, comparison_value: float, resolution_date: str,
        ticker: str | None = None, series_id: str | None = None,
        url: str | None = None, end_date: str | None = None,
        state: dict | None = None,
    ) -> str:
        """Estimate P(value > threshold) at a future date via linear + seasonal trend.

        Args:
            source: Data source — yfinance, fred, or dbnomics.
            comparison_value: The numeric threshold to compare against.
            resolution_date: Date the question resolves (YYYY-MM-DD).
            ticker: Ticker (required when source=yfinance).
            series_id: FRED series id (required when source=fred).
            url: DBnomics series path (required when source=dbnomics).
            end_date: Optional end date (YYYY-MM-DD); clamped to the cutoff.

        Returns:
            JSON with linear/seasonal/combined exceedance probabilities.
        """
        ctx = state["ctx"]
        src = str(source or "").strip()
        if not src:
            return json.dumps({"error": "source required (yfinance, fred, or dbnomics)"})
        fetch_args = {k: v for k, v in
                      (("ticker", ticker), ("series_id", series_id), ("url", url)) if v}
        return await datatools.analyze_trend(
            src, comparison_value, str(resolution_date or ""), ctx["cutoff_date"],
            end_date=end_date, **fetch_args,
        )

    # -- optional crowd-prior tools -----------------------------------------

    async def polymarket_get_market(
        self, updated_belief: BeliefUpdate | None = None, state: dict | None = None,
    ) -> str:
        """Get cutoff-safe metadata for this episode's market (id, volume, criteria).

        Args:
            updated_belief: Your updated belief state.

        Returns:
            JSON with static contract identity + dataset volume (no live resolution).
        """
        return await polymarket.get_market_metadata(state["ctx"])

    async def polymarket_search(
        self, query: str, limit: int = 5,
        updated_belief: BeliefUpdate | None = None, state: dict | None = None,
    ) -> str:
        """Search related Polymarket markets by keyword (Gamma API).

        Args:
            query: Keywords to search for.
            limit: Maximum results.
            updated_belief: Your updated belief state.

        Returns:
            JSON list of related markets (identity only).
        """
        return await polymarket.search_markets(str(query or ""), state["ctx"], limit=int(limit))

    async def polymarket_market_price(
        self, updated_belief: BeliefUpdate | None = None, state: dict | None = None,
    ) -> str:
        """Get this market's YES price at the cutoff (CLOB history endTs, never live mid).

        Args:
            updated_belief: Your updated belief state.

        Returns:
            JSON with yes_price_at_cutoff (the crowd's implied probability).
        """
        return await polymarket.market_price(state["ctx"])

    async def polymarket_price_history(
        self, interval: str = "1d", fidelity: int = 60,
        updated_belief: BeliefUpdate | None = None, state: dict | None = None,
    ) -> str:
        """Get this market's YES price history up to the cutoff.

        Args:
            interval: Lookback window — 1h, 6h, 1d, 1w, or 1m.
            fidelity: Sampling fidelity in minutes.
            updated_belief: Your updated belief state.

        Returns:
            JSON with [{t, p}] points up to the cutoff.
        """
        return await polymarket.price_history(
            state["ctx"], interval=str(interval), fidelity=int(fidelity))

    # -- terminal tool -------------------------------------------------------

    async def submit(
        self, probability: float, reasoning: str = "",
        updated_belief: BeliefUpdate | None = None, state: dict | None = None,
    ) -> str:
        """Submit your final probability that the question resolves YES. Ends the episode.

        Args:
            probability: Your final P(YES), between 0.05 and 0.95.
            reasoning: Brief justification for the estimate.
            updated_belief: Your final belief state.

        Returns:
            JSON acknowledgment.
        """
        p = _clamp_prob(probability)
        if p is None:
            return json.dumps({
                "error": "probability must be a number between 0 and 1; episode continues"})
        state["submitted"] = True
        state["submitted_prob"] = p
        state["submit_reasoning"] = str(reasoning or "")
        return json.dumps({"status": "submitted", "probability": p})


# ------------------------------------------------------------ load_environment

def _load_rows(dataset_path: str | None, split: str) -> list[dict]:
    """Load JSONL rows from an explicit path or the packaged data files."""
    import os

    filename = f"{split}.jsonl"
    if dataset_path:
        path = dataset_path
        if os.path.isdir(path):
            path = os.path.join(path, filename)
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    pkg_data = resources.files("prime_forecast").joinpath("data").joinpath(filename)
    if not pkg_data.is_file():
        raise FileNotFoundError(
            f"No packaged dataset at prime_forecast/data/{filename}. "
            "Build one with scripts/build_dataset.py --install, or pass dataset_path."
        )
    with pkg_data.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _row_to_example(row: dict, *, max_turns: int) -> dict:
    info = {
        "outcome": int(row["outcome"]),
        "price_at_cutoff": row.get("price_at_cutoff"),
        "cutoff_date": row.get("cutoff_date"),
        "resolution_date": row.get("resolution_date"),
        "market_id": row.get("market_id"),
        "slug": row.get("slug"),
        "token_id": row.get("token_id"),
        "category": row.get("category"),
        "question": row.get("question"),
        "resolution_criteria": row.get("resolution_criteria"),
        "volume": row.get("volume"),
    }
    return {
        "prompt": [{"role": "user", "content": prompts.seed_user_message(row, max_turns=max_turns)}],
        "info": json.dumps(info, default=str),
    }


def _ensure_search_keys() -> None:
    """Require credentials for the active search backend."""
    import os

    backend = search.search_backend()
    if backend == "brave":
        vf.ensure_keys(["BRAVE_API_KEY"])
    elif backend == "tavily":
        vf.ensure_keys(["TAVILY_API_KEY"])
    elif backend == "exa":
        vf.ensure_keys(["EXA_API_KEY"])
    elif backend == "agentcore":
        if not search.agentcore_gateway_url():
            raise ValueError(
                "PF_SEARCH_BACKEND=agentcore requires AGENTCORE_GATEWAY_URL "
                "or AGENTCORE_GATEWAY_ID (run scripts/setup_agentcore_search.py)"
            )
        # SigV4 needs an IAM credential chain. Bedrock bearer alone is not enough.
        has_iam = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("AWS_PROFILE")
            or os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            or os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE")
        )
        if not has_iam:
            raise ValueError(
                "AgentCore Gateway requires IAM SigV4 credentials "
                "(AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE). "
                "AWS_BEARER_TOKEN_BEDROCK is only for Bedrock Runtime (leak filter)."
            )
    elif backend == "none":
        # Allowed for offline unit tests that inject dataset_rows and mock tools.
        pass


def load_environment(
    dataset_path: str | None = None,
    split: str = "train",
    eval_split: str = "val",
    num_examples: int = -1,
    num_eval_examples: int = -1,
    max_turns: int = 8,
    max_web_searches: int = 2,
    include_market_tools: bool = True,
    protocol_bonus_weight: float = 0.0,
    dataset_rows: list[dict] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """Entry point for prime eval / prime-rl / Hosted Training.

    Args:
        dataset_path: Optional path to a JSONL file or a directory containing
            {split}.jsonl files. Defaults to the data packaged with the wheel.
        split: Training split name (train).
        eval_split: Evaluation split name (val or test).
        num_examples: Truncate the training split (-1 = all).
        num_eval_examples: Truncate the eval split (-1 = all).
        max_turns: Maximum agent turns per rollout (BLF T_max).
        max_web_searches: Cap live web_search calls per rollout (cost control).
        include_market_tools: Expose Polymarket crowd-price/history tools (on by default).
        protocol_bonus_weight: Weight of optional BLF shaping (default 0 = off).
        dataset_rows: Inline rows (tests only) — bypasses file loading.
    """
    if dataset_rows is None:
        _ensure_search_keys()

    if dataset_rows is not None:
        train_rows, eval_rows = dataset_rows, dataset_rows
    else:
        train_rows = _load_rows(dataset_path, split)
        try:
            eval_rows = _load_rows(dataset_path, eval_split)
        except FileNotFoundError:
            eval_rows = []

    if num_examples > 0:
        train_rows = train_rows[:num_examples]
    if num_eval_examples > 0:
        eval_rows = eval_rows[:num_eval_examples]

    train_ds = Dataset.from_list([_row_to_example(r, max_turns=max_turns) for r in train_rows])
    eval_ds = (
        Dataset.from_list([_row_to_example(r, max_turns=max_turns) for r in eval_rows])
        if eval_rows else None
    )

    rubric = vf.Rubric(
        funcs=[forecast_reward, protocol_bonus],
        weights=[1.0, protocol_bonus_weight],
    )
    rubric.add_metric(brier_score)
    rubric.add_metric(market_brier)
    rubric.add_metric(submitted)
    rubric.add_metric(predicted_prob)

    system_prompt = prompts.SYSTEM_PROMPT
    if include_market_tools:
        system_prompt = system_prompt + prompts.MARKET_TOOLS_ADDENDUM

    return ForecastEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
        system_prompt=system_prompt,
        max_turns=max_turns,
        max_web_searches=max_web_searches,
        include_market_tools=include_market_tools,
        **kwargs,
    )
