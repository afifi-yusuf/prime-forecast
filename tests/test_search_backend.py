"""Unit tests for search backend selection + AgentCore payload parsing."""

from __future__ import annotations

import json

import pytest

from prime_forecast import search


def test_search_backend_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("PF_SEARCH_BACKEND", "agentcore")
    monkeypatch.setenv("EXA_API_KEY", "x")
    assert search.search_backend() == "agentcore"


def test_search_backend_auto_brave(monkeypatch):
    monkeypatch.delenv("PF_SEARCH_BACKEND", raising=False)
    monkeypatch.delenv("AGENTCORE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("AGENTCORE_GATEWAY_ID", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "BSA-x")
    assert search.search_backend() == "brave"


def test_search_backend_auto_tavily(monkeypatch):
    monkeypatch.delenv("PF_SEARCH_BACKEND", raising=False)
    monkeypatch.delenv("AGENTCORE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("AGENTCORE_GATEWAY_ID", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    assert search.search_backend() == "tavily"


def test_search_backend_auto_exa(monkeypatch):
    monkeypatch.delenv("PF_SEARCH_BACKEND", raising=False)
    monkeypatch.delenv("AGENTCORE_GATEWAY_URL", raising=False)
    monkeypatch.delenv("AGENTCORE_GATEWAY_ID", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("EXA_API_KEY", "x")
    assert search.search_backend() == "exa"


def test_agentcore_gateway_url_from_id(monkeypatch):
    monkeypatch.delenv("AGENTCORE_GATEWAY_URL", raising=False)
    monkeypatch.setenv("AGENTCORE_GATEWAY_ID", "abc123")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    assert search.agentcore_gateway_url().endswith(
        "gateway-abc123.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
    )


def test_parse_agentcore_payload_json_results():
    payload = {
        "id": "x",
        "results": [
            {
                "title": "Fed holds rates",
                "url": "https://example.com/a",
                "publishedDate": "2026-04-01",
                "text": "The Fed held rates steady.",
            }
        ],
    }
    items = search._parse_agentcore_payload([
        {"type": "text", "text": json.dumps(payload)},
    ])
    assert len(items) == 1
    assert items[0]["url"] == "https://example.com/a"


@pytest.mark.asyncio
async def test_web_search_budget(monkeypatch, sample_row):
    from prime_forecast.env import ForecastEnv, load_environment

    calls = {"n": 0}

    async def fake_search(query, *, cutoff_date, num_results=10, question=""):
        calls["n"] += 1
        return "", [], {"mode": "test"}

    monkeypatch.setattr(search, "web_search", fake_search)
    monkeypatch.setenv("PF_SEARCH_BACKEND", "none")

    env = load_environment(dataset_rows=[sample_row], max_web_searches=1)
    assert isinstance(env, ForecastEnv)
    state = {"info": sample_row}
    await env.setup_state(state)

    out1 = json.loads(await env.web_search("q1", updated_belief=None, state=state))
    assert "error" not in out1
    assert out1["web_searches_remaining"] == 0
    assert calls["n"] == 1

    out2 = json.loads(await env.web_search("q2", updated_belief=None, state=state))
    assert "budget exhausted" in out2["error"]
    assert calls["n"] == 1
