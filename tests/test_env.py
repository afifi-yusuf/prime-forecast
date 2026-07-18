import json

import pytest

from prime_forecast import load_environment
from prime_forecast.belief import BeliefState, SearchStore
from prime_forecast.env import ForecastEnv


def _make_env(sample_row, **kwargs) -> ForecastEnv:
    return load_environment(dataset_rows=[sample_row], **kwargs)


async def _make_state(env: ForecastEnv, sample_row) -> dict:
    info = env.dataset[0]["info"]
    state = {"info": info, "trajectory": []}
    await env.setup_state(state)
    return state


def test_market_tools_included_by_default(sample_row):
    env = _make_env(sample_row)
    names = [t.name for t in env.tool_defs]
    assert "web_search" in names and "submit" in names
    assert "lookup_url" in names
    assert "summarize_results" not in names
    assert "polymarket_market_price" in names
    assert "polymarket_price_history" in names
    assert "polymarket_get_market" in names

    env2 = _make_env(sample_row, include_market_tools=False)
    names2 = [t.name for t in env2.tool_defs]
    assert not any(n.startswith("polymarket") for n in names2)


def test_hidden_state_arg_not_in_schemas(sample_row):
    env = _make_env(sample_row)
    for tool_def in env.tool_defs:
        props = tool_def.parameters.get("properties", {})
        assert "state" not in props, f"{tool_def.name} leaks hidden arg"


@pytest.mark.asyncio
async def test_setup_state_excludes_outcome_from_ctx(sample_row):
    env = _make_env(sample_row)
    state = await _make_state(env, sample_row)
    assert "outcome" not in state["ctx"]
    assert state["ctx"]["cutoff_date"] == sample_row["cutoff_date"]
    assert isinstance(state["belief"], BeliefState)
    assert isinstance(state["store"], SearchStore)


@pytest.mark.asyncio
async def test_submit_sets_state_and_clamps(sample_row):
    env = _make_env(sample_row)
    state = await _make_state(env, sample_row)
    out = json.loads(await env.submit(probability=0.99, reasoning="sure", state=state))
    assert out["status"] == "submitted"
    assert out["probability"] == 0.95
    assert state["submitted"] is True
    assert state["submitted_prob"] == 0.95
    assert await env.forecast_submitted(state) is True


@pytest.mark.asyncio
async def test_submit_rejects_garbage_probability(sample_row):
    env = _make_env(sample_row)
    state = await _make_state(env, sample_row)
    out = json.loads(await env.submit(probability="high", state=state))
    assert "error" in out
    assert state["submitted"] is False

    out2 = json.loads(await env.submit(probability=7.0, state=state))
    assert "error" in out2
    assert state["submitted"] is False


@pytest.mark.asyncio
async def test_update_tool_args_injects_state_and_merges_belief(sample_row):
    env = _make_env(sample_row)
    state = await _make_state(env, sample_row)
    args = env.update_tool_args(
        "web_search",
        {"query": "x", "updated_belief": {"p": 0.7, "update_reasoning": "prior news"}},
        [],
        state,
    )
    assert args["state"] is state
    assert "updated_belief" not in args
    assert state["belief"].p == 0.7
    assert state["belief"].step == 1
    assert state["research_tool_calls"] == 1


@pytest.mark.asyncio
async def test_update_tool_args_handles_json_string_belief(sample_row):
    env = _make_env(sample_row)
    state = await _make_state(env, sample_row)
    env.update_tool_args(
        "submit", {"probability": 0.6, "updated_belief": '{"p": 0.61}'}, [], state)
    assert state["belief"].p == 0.61


@pytest.mark.asyncio
async def test_data_tools_clamp_end_date_to_cutoff(sample_row, monkeypatch):
    """A model-supplied end_date past the cutoff must be clamped."""
    from prime_forecast import datatools

    captured = {}

    async def fake_rows(ticker, cutoff_date, *, end_date=None):
        captured["end"] = end_date
        return [{"date": "2026-04-30", "value": 1.0}], None

    monkeypatch.setattr(datatools, "fetch_yfinance_rows", fake_rows)
    env = _make_env(sample_row)
    state = await _make_state(env, sample_row)
    out = json.loads(await env.fetch_ts_yfinance(
        ticker="SPY", end_date="2026-12-31", state=state))
    # datatools.fetch_yfinance clamps internally; verify via the reported end_date.
    assert out["end_date"] == "2026-05-01"


@pytest.mark.asyncio
async def test_lookup_url_blocks_market_domains(sample_row):
    env = _make_env(sample_row)
    state = await _make_state(env, sample_row)
    out = json.loads(await env.lookup_url(
        url="https://polymarket.com/event/will-x-happen", state=state))
    assert "blocked" in out["error"]


@pytest.mark.asyncio
async def test_web_search_requires_query(sample_row):
    env = _make_env(sample_row)
    state = await _make_state(env, sample_row)
    out = json.loads(await env.web_search(query="site:polymarket.com", state=state))
    assert "error" in out


def test_prompt_contains_no_outcome_or_price(sample_row):
    env = _make_env(sample_row)
    prompt = env.dataset[0]["prompt"]
    text = json.dumps(prompt)
    assert "0.437" not in text  # price_at_cutoff never in prompt
    assert sample_row["question"] in text
    # The seed (user) message is built from the row: no outcome/price fields.
    user_msgs = [m for m in prompt if m["role"] == "user"]
    assert len(user_msgs) == 1
    seed = user_msgs[0]["content"].lower()
    assert "outcome" not in seed
    assert "price" not in seed or "market prices — retrieve" in seed
