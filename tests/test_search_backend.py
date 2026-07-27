"""Unit tests for search backend selection + AgentCore payload parsing."""

from __future__ import annotations

import json

import pytest

from prime_forecast import search


def test_search_backend_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("PF_SEARCH_BACKEND", "agentcore")
    monkeypatch.setenv("EXA_API_KEY", "x")
    assert search.search_backend() == "agentcore"


def _clear_backend_env(monkeypatch, *keep: str) -> None:
    for var in ("PF_SEARCH_BACKEND", "AGENTCORE_GATEWAY_URL", "AGENTCORE_GATEWAY_ID",
                "FIRECRAWL_API_KEY", "BRAVE_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY"):
        if var not in keep:
            monkeypatch.delenv(var, raising=False)


def test_search_backend_auto_firecrawl(monkeypatch):
    _clear_backend_env(monkeypatch, "FIRECRAWL_API_KEY")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-x")
    assert search.search_backend() == "firecrawl"


def test_search_backend_auto_brave(monkeypatch):
    _clear_backend_env(monkeypatch, "BRAVE_API_KEY")
    monkeypatch.setenv("BRAVE_API_KEY", "BSA-x")
    assert search.search_backend() == "brave"


def test_search_backend_auto_tavily(monkeypatch):
    _clear_backend_env(monkeypatch, "TAVILY_API_KEY")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    assert search.search_backend() == "tavily"


def test_search_backend_auto_exa(monkeypatch):
    _clear_backend_env(monkeypatch, "EXA_API_KEY")
    monkeypatch.setenv("EXA_API_KEY", "x")
    assert search.search_backend() == "exa"


def test_firecrawl_web_results_v2_shape():
    data = {"success": True, "data": {"web": [
        {"title": "Fed holds rates", "url": "https://example.com/a",
         "description": "The Fed held rates steady."},
        "not-a-dict",
    ]}}
    items = search._firecrawl_web_results(data)
    assert len(items) == 1
    assert items[0]["url"] == "https://example.com/a"


def test_firecrawl_web_results_v1_shape():
    data = {"success": True, "data": [
        {"title": "A", "url": "https://example.com/a", "description": "x"},
        {"title": "B", "url": "https://example.com/b", "description": "y"},
    ]}
    assert len(search._firecrawl_web_results(data)) == 2


def test_firecrawl_web_results_bad_shapes():
    assert search._firecrawl_web_results({}) == []
    assert search._firecrawl_web_results({"data": None}) == []
    assert search._firecrawl_web_results([1, 2]) == []


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
