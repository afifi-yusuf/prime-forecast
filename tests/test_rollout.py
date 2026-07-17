"""End-to-end rollout: scripted model calls web_search (mocked Exa) then submit.

Validates the full StatefulToolEnv loop offline: native tool dispatch, hidden
state injection, belief merging, the submit stop condition, and rubric scoring.
"""

import json

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

import verifiers as vf
from verifiers.clients import OpenAIChatCompletionsClient

from prime_forecast import load_environment
from prime_forecast.env import forecast_reward, protocol_bonus


def _completion(message: ChatCompletionMessage) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[Choice(finish_reason="stop", index=0, message=message)],
        created=0,
        model="scripted",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=10, prompt_tokens=100, total_tokens=110),
    )


def _tool_call_msg(name: str, args: dict, call_id: str) -> ChatCompletionMessage:
    return ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[ChatCompletionMessageToolCall(
            id=call_id, type="function",
            function=Function(name=name, arguments=json.dumps(args)),
        )],
    )


class ScriptedOpenAI:
    """Minimal fake openai.AsyncOpenAI that replays a fixed turn script.

    verifiers' OpenAIChatCompletionsClient issues raw `client.post(...)`
    calls expecting an httpx.Response containing ChatCompletion JSON.
    """

    base_url = "http://scripted.local/v1"

    def __init__(self, turns: list[ChatCompletionMessage]):
        self._turns = list(turns)

    async def post(self, path, *, body=None, cast_to=None, options=None):
        import httpx

        completion = _completion(self._turns.pop(0))
        return httpx.Response(
            status_code=200,
            content=completion.model_dump_json().encode(),
            request=httpx.Request("POST", self.base_url + path),
        )

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_full_rollout_and_scoring(sample_row, monkeypatch):
    from prime_forecast import search

    async def fake_exa(query, *, cutoff_date, num_results=10, question=""):
        parsed = [{
            "title": "Pre-cutoff preview",
            "url": "https://example.com/preview",
            "snippet": "X looks likely per analysts",
            "body": "X looks likely per analysts",
            "published": "2026-04-20",
        }]
        return "raw", parsed, {}

    monkeypatch.setattr(search, "search_exa", fake_exa)

    env = load_environment(dataset_rows=[sample_row])

    turns = [
        _tool_call_msg("web_search", {
            "query": "will X happen preview",
            "updated_belief": {"p": 0.65, "confidence": "medium",
                               "update_reasoning": "analyst previews lean yes",
                               "evidence_for": ["analyst preview favors X"]},
        }, "call_1"),
        _tool_call_msg("submit", {
            "probability": 0.8,
            "reasoning": "strong pre-cutoff signal",
            "updated_belief": {"p": 0.8},
        }, "call_2"),
    ]
    client = OpenAIChatCompletionsClient(ScriptedOpenAI(turns))

    row = env.dataset[0]
    rollout_input = {
        "prompt": row["prompt"],
        "example_id": 0,
        "answer": "",
        "info": row["info"],
    }
    state = await env.rollout(rollout_input, client, model="scripted")

    assert state.get("error") is None, state.get("error")
    assert state["submitted"] is True
    assert state["submitted_prob"] == 0.8
    assert state["stop_condition"] == "forecast_submitted"
    assert state["belief"].p == 0.8
    assert state["belief"].step == 2
    assert "analyst preview favors X" in state["belief"].evidence_for

    # Tool result reached the conversation and was leak-filtered JSON.
    tool_msgs = [m for m in state["completion"] if getattr(m, "role", None) == "tool"]
    assert len(tool_msgs) == 2
    ws_payload = json.loads(tool_msgs[0].content)
    assert ws_payload["num_results"] == 1
    assert ws_payload["belief"]["p"] == 0.65

    # Rubric math: outcome=1, p=0.8 -> reward 0.96; full protocol bonus.
    reward = await forecast_reward(state, row["info"])
    assert reward == pytest.approx(1 - 0.04)
    assert await protocol_bonus(state) == 1.0


@pytest.mark.asyncio
async def test_rollout_no_submit_scores_zero(sample_row):
    env = load_environment(dataset_rows=[sample_row])
    turns = [ChatCompletionMessage(role="assistant", content="I refuse to use tools.")]
    client = OpenAIChatCompletionsClient(ScriptedOpenAI(turns))

    row = env.dataset[0]
    state = await env.rollout({
        "prompt": row["prompt"], "example_id": 0, "answer": "",
        "info": row["info"],
    }, client, model="scripted")

    assert state["submitted"] is False
    assert await forecast_reward(state, row["info"]) == 0.0


@pytest.mark.asyncio
async def test_rollout_probability_tag_fallback(sample_row):
    env = load_environment(dataset_rows=[sample_row])
    turns = [ChatCompletionMessage(
        role="assistant",
        content="Based on priors alone: <probability>0.7</probability>",
    )]
    client = OpenAIChatCompletionsClient(ScriptedOpenAI(turns))

    row = env.dataset[0]
    state = await env.rollout({
        "prompt": row["prompt"], "example_id": 0, "answer": "",
        "info": row["info"],
    }, client, model="scripted")

    # No explicit submit, but the tag is honored: reward = 1 - (0.7-1)^2.
    assert await forecast_reward(state, row["info"]) == pytest.approx(1 - 0.09)
