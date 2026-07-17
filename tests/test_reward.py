import json

import pytest

from prime_forecast.belief import BeliefState
from prime_forecast.env import (
    MAX_PROB,
    MIN_PROB,
    NO_SUBMIT_REWARD,
    brier_score,
    extract_probability,
    forecast_reward,
    market_brier,
    protocol_bonus,
)


def _info(outcome: int, price: float | None = 0.4) -> str:
    return json.dumps({"outcome": outcome, "price_at_cutoff": price})


def _state(prob=None, **extra) -> dict:
    s = {"submitted_prob": prob, "submitted": prob is not None, "completion": []}
    s.update(extra)
    return s


@pytest.mark.asyncio
async def test_forecast_reward_perfect_and_worst():
    assert await forecast_reward(_state(0.95), _info(1)) == pytest.approx(1 - 0.05**2)
    assert await forecast_reward(_state(0.05), _info(1)) == pytest.approx(1 - 0.95**2)
    assert await forecast_reward(_state(0.5), _info(0)) == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_forecast_reward_soft_no_submit():
    # Soft no-submit matches always-0.5 so RL is not forced to spam mid-probs.
    assert await forecast_reward(_state(None), _info(1)) == pytest.approx(NO_SUBMIT_REWARD)
    assert await forecast_reward(_state(None), _info(0)) == pytest.approx(NO_SUBMIT_REWARD)
    assert NO_SUBMIT_REWARD == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_reward_band_always_half():
    r = await forecast_reward(_state(0.5), _info(1))
    assert r == pytest.approx(0.75)
    # Sharp correct beats soft no-submit; sharp wrong loses.
    assert await forecast_reward(_state(0.9), _info(1)) > NO_SUBMIT_REWARD
    assert await forecast_reward(_state(0.1), _info(1)) < NO_SUBMIT_REWARD


@pytest.mark.asyncio
async def test_brier_metric():
    assert await brier_score(_state(0.7), _info(1)) == pytest.approx(0.09)
    assert await brier_score(_state(None), _info(1)) == 1.0


@pytest.mark.asyncio
async def test_market_brier_metric():
    assert await market_brier(_info(1, price=0.4)) == pytest.approx(0.36)
    assert await market_brier(_info(0, price=0.4)) == pytest.approx(0.16)
    assert await market_brier(_info(1, price=None)) == -1.0


def test_extract_probability_clamps():
    assert extract_probability(_state(0.999)) == MAX_PROB
    assert extract_probability(_state(0.001)) == MIN_PROB
    assert extract_probability(_state(0.42)) == 0.42


def test_extract_probability_tag_fallback():
    state = {
        "submitted_prob": None,
        "completion": [
            {"role": "assistant", "content": "My final answer: <probability>0.62</probability>"},
        ],
    }
    assert extract_probability(state) == pytest.approx(0.62)


def test_extract_probability_none_when_absent():
    state = {"submitted_prob": None,
             "completion": [{"role": "assistant", "content": "no clue"}]}
    assert extract_probability(state) is None


@pytest.mark.asyncio
async def test_protocol_bonus_components():
    empty = _state(None)
    empty["belief"] = BeliefState()
    empty["research_tool_calls"] = 0
    assert await protocol_bonus(empty) == 0.0

    belief = BeliefState()
    belief.merge_update({"p": 0.7, "evidence_for": ["poll at 65%"],
                         "update_reasoning": "poll moved me up"})
    full = _state(0.7)
    full["belief"] = belief
    full["research_tool_calls"] = 3
    assert await protocol_bonus(full) == 1.0
