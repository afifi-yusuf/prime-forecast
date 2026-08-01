"""Web search backends with publish-date filtering + leak filter.

Backends (PF_SEARCH_BACKEND):
  firecrawl  — Firecrawl Search API (tbs custom date range cutoff)
  brave      — Brave Search API (cheap; no native end-date — leak filter applies)
  tavily     — Tavily Search API (free tier; end_date cutoff)
  exa        — Exa API (supports endPublishedDate)
  agentcore  — Amazon Bedrock AgentCore Web Search via Gateway MCP
  none       — disabled (returns empty / error)

Env:
  PF_SEARCH_BACKEND          — firecrawl | brave | tavily | exa | agentcore | none (auto if unset)
  FIRECRAWL_API_KEY          — required for firecrawl
  BRAVE_API_KEY              — required for brave
  TAVILY_API_KEY             — required for tavily
  EXA_API_KEY                — required for exa
  AGENTCORE_GATEWAY_URL      — MCP endpoint, e.g.
      https://gateway-XXXX.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp
  AGENTCORE_GATEWAY_ID       — optional; used to derive URL if URL unset
  AWS_REGION / AWS_DEFAULT_REGION — default us-east-1 (AgentCore Web Search region)
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_PROFILE — IAM for Gateway (SigV4)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

import httpx

from prime_forecast.cutoff import parse_ts
from prime_forecast.leak_filter import (
    RESULTS_SEPARATOR,
    domain_blocked,
    filter_results,
    sanitize_search_query,
)

_EXA_URL = "https://api.exa.ai/search"
_TAVILY_URL = "https://api.tavily.com/search"
_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
_FIRECRAWL_URL = "https://api.firecrawl.dev/v2/search"
_MAX_TEXT_CHARS = 2000
_AGENTCORE_TOOL_NAMES = ("WebSearch", "WebSearchTool", "web_search", "web-search")
_BACKENDS = frozenset({"firecrawl", "brave", "tavily", "exa", "agentcore", "searxng", "none"})


def search_backend() -> str:
    """Resolve active search backend from env.

    PF_SEARCH_BACKEND may be a comma-separated fallback chain
    (e.g. "searxng,brave,tavily"): each backend is tried in order and the
    first non-empty result set wins. Empty results count as failure —
    scraping backends can silently return [] when upstream engines
    rate-limit or CAPTCHA the caller's IP (observed: 93-100% empty across
    an entire eval campaign while HTTP status stayed 200).
    """
    raw = os.environ.get("PF_SEARCH_BACKEND", "").strip().lower()
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts and all(p in _BACKENDS for p in parts):
            return raw
    if raw in _BACKENDS:
        return raw
    if os.environ.get("AGENTCORE_GATEWAY_URL") or os.environ.get("AGENTCORE_GATEWAY_ID"):
        return "agentcore"
    if os.environ.get("FIRECRAWL_API_KEY", "").strip():
        return "firecrawl"
    if os.environ.get("BRAVE_API_KEY", "").strip():
        return "brave"
    if os.environ.get("TAVILY_API_KEY", "").strip():
        return "tavily"
    if os.environ.get("EXA_API_KEY", "").strip():
        return "exa"
    if os.environ.get("SEARXNG_URL", "").strip():
        return "searxng"
    return "none"


def agentcore_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def agentcore_gateway_url() -> str:
    url = os.environ.get("AGENTCORE_GATEWAY_URL", "").strip()
    if url:
        return url.rstrip("/")
    gw_id = os.environ.get("AGENTCORE_GATEWAY_ID", "").strip()
    if not gw_id:
        return ""
    region = agentcore_region()
    return f"https://gateway-{gw_id}.gateway.bedrock-agentcore.{region}.amazonaws.com/mcp"


def _age_flag(published: str | None, cutoff_date: str) -> str:
    if not published:
        return "  [no known date]"
    dt = parse_ts(published)
    cutoff_dt = parse_ts(str(cutoff_date)[:10])
    if dt is None or cutoff_dt is None:
        return "  [?? unparseable date — check manually]"
    if dt > cutoff_dt:
        return f"  [!! AFTER CUTOFF {str(cutoff_date)[:10]}]"
    return f"  [ok, before {str(cutoff_date)[:10]}]"


def _sanitize_query(query: str, *, max_chars: int = 400) -> str:
    q = sanitize_search_query(query)
    if len(q) > max_chars:
        q = q[:max_chars].rsplit(" ", 1)[0]
    return q


def _build_from_items(
    items: list[dict],
    *,
    cutoff_date: str,
) -> tuple[str, list[dict]]:
    """Convert provider-normalized items into (raw_blocks, parsed)."""
    blocks: list[str] = []
    parsed: list[dict] = []
    cutoff_str = str(cutoff_date)[:10]
    cutoff_dt = parse_ts(cutoff_str)

    for item in items:
        url = item.get("url", "") or ""
        if url and domain_blocked(url):
            continue
        title = item.get("title", "") or ""
        pub = item.get("publishedDate") or item.get("published_date") or item.get("published") or ""
        if cutoff_dt is not None and pub:
            d = parse_ts(pub)
            if d is not None and d > cutoff_dt:
                continue
        text = (item.get("text") or item.get("snippet") or item.get("body") or "")[:_MAX_TEXT_CHARS]
        flag = _age_flag(pub, cutoff_str)
        block = f"{title}\n{url}\nPublished: {pub}{flag}\n{text}"
        blocks.append(block)
        parsed.append({
            "title": title,
            "url": url,
            "snippet": text[:400],
            "body": text,
            "published": pub,
        })
    return RESULTS_SEPARATOR.join(blocks), parsed


async def _apply_leak_filter(
    raw: str,
    parsed: list[dict],
    *,
    cutoff_date: str,
    question: str,
) -> tuple[str, list[dict], dict]:
    filtered, filter_debug = await filter_results(raw, cutoff_date, question=question)
    if filtered != raw:
        kept_urls = set()
        for part in filtered.split(RESULTS_SEPARATOR):
            for line in part.splitlines():
                if line.startswith("http"):
                    kept_urls.add(line.strip())
                    break
        if kept_urls:
            parsed = [p for p in parsed if p.get("url") in kept_urls]
        else:
            # Keep undated/title-only hits that survived filter but have no URL line
            parsed = [] if kept_urls == set() and filtered.strip() == "" else [
                p for p in parsed if (p.get("url") in kept_urls) or not p.get("url")
            ]
            if not filtered.strip():
                parsed = []
        raw = filtered
    return raw, parsed, filter_debug


# ------------------------------------------------------------------ Brave

async def search_brave(
    query: str,
    *,
    cutoff_date: str,
    num_results: int = 10,
    question: str = "",
) -> tuple[str, list[dict], dict]:
    """Brave Web Search API. No native publish-date cutoff — leak filter applies."""
    api_key = os.environ.get("BRAVE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("BRAVE_API_KEY not set (required for PF_SEARCH_BACKEND=brave)")

    q = _sanitize_query(query, max_chars=400)
    if not q:
        raise ValueError("query empty after sanitization")
    k = max(1, min(int(num_results), 20))

    params = {
        "q": q,
        "count": k,
        "search_lang": "en",
        "country": "us",
        "text_decorations": "0",
        "spellcheck": "1",
        "extra_snippets": "true",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            _BRAVE_URL,
            params=params,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )
        r.raise_for_status()
        data = r.json()

    items = []
    for item in (data.get("web") or {}).get("results") or []:
        snippets = item.get("extra_snippets") or []
        extra = " ".join(str(s) for s in snippets if s)
        desc = item.get("description") or item.get("snippet") or ""
        text = (desc + (" " + extra if extra else "")).strip()
        items.append({
            "title": item.get("title", "") or "",
            "url": item.get("url", "") or "",
            "publishedDate": (
                item.get("page_age")
                or item.get("age")
                or item.get("published")
                or ""
            ),
            "text": text[:_MAX_TEXT_CHARS],
        })
    raw, parsed = _build_from_items(items, cutoff_date=cutoff_date)
    return await _apply_leak_filter(raw, parsed, cutoff_date=cutoff_date, question=question)


# -------------------------------------------------------------- Firecrawl

def _firecrawl_web_results(data: Any) -> list[dict]:
    """Normalize v1 (data: [...]) and v2 (data: {web: [...]}) response shapes."""
    payload = data.get("data") if isinstance(data, dict) else None
    if isinstance(payload, dict):
        payload = payload.get("web") or []
    if not isinstance(payload, list):
        return []
    return [x for x in payload if isinstance(x, dict)]


async def search_firecrawl(
    query: str,
    *,
    cutoff_date: str,
    num_results: int = 10,
    question: str = "",
) -> tuple[str, list[dict], dict]:
    """Firecrawl Search API. tbs custom date range caps indexing at the cutoff.

    Search-only (no scrapeOptions) to keep per-call credit cost at 1; the agent
    reads full pages via lookup_url. Publish dates are often absent from search
    hits, so the tbs provider filter + leak filter carry the cutoff.
    """
    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY not set (required for PF_SEARCH_BACKEND=firecrawl)")

    q = _sanitize_query(query, max_chars=400)
    if not q:
        raise ValueError("query empty after sanitization")
    k = max(1, min(int(num_results), 20))

    payload: dict = {"query": q, "limit": k, "sources": ["web"]}
    cutoff_dt = parse_ts(str(cutoff_date)[:10]) if cutoff_date else None
    if cutoff_dt is not None:
        # Google-style custom date range: only pages dated on or before the cutoff.
        payload["tbs"] = f"cdr:1,cd_max:{cutoff_dt.month}/{cutoff_dt.day}/{cutoff_dt.year}"

    # Concurrent rollouts hit Firecrawl's per-minute rate limit; back off on
    # 429/5xx instead of surfacing an error rollout with zero context.
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(4):
            r = await client.post(_FIRECRAWL_URL, json=payload, headers=headers)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == 3:
                    r.raise_for_status()
                retry_after = r.headers.get("Retry-After")
                try:
                    delay = min(float(retry_after), 30.0) if retry_after else 2.0 * 2**attempt
                except ValueError:
                    delay = 2.0 * 2**attempt
                await asyncio.sleep(delay)
                continue
            r.raise_for_status()
            data = r.json()
            break

    items = []
    for item in _firecrawl_web_results(data):
        meta = item.get("metadata") or {}
        items.append({
            "title": item.get("title") or meta.get("title") or "",
            "url": item.get("url") or meta.get("sourceURL") or "",
            "publishedDate": (
                item.get("publishedDate")
                or meta.get("publishedDate")
                or meta.get("publishedTime")
                or meta.get("article:published_time")
                or ""
            ),
            "text": (
                item.get("description")
                or item.get("markdown")
                or meta.get("description")
                or ""
            )[:_MAX_TEXT_CHARS],
        })
    raw, parsed = _build_from_items(items, cutoff_date=cutoff_date)
    return await _apply_leak_filter(raw, parsed, cutoff_date=cutoff_date, question=question)


# ----------------------------------------------------------------- Tavily

async def search_tavily(
    query: str,
    *,
    cutoff_date: str,
    num_results: int = 10,
    question: str = "",
) -> tuple[str, list[dict], dict]:
    """Tavily Search API. Uses end_date for hard publish cutoff when available."""
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set (required for PF_SEARCH_BACKEND=tavily)")

    q = _sanitize_query(query, max_chars=400)
    if not q:
        raise ValueError("query empty after sanitization")
    k = max(1, min(int(num_results), 20))

    payload: dict = {
        "api_key": api_key,
        "query": q,
        "max_results": k,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""
    if cutoff_str:
        # Tavily rejects results published after end_date (YYYY-MM-DD).
        payload["end_date"] = cutoff_str

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(_TAVILY_URL, json=payload)
        r.raise_for_status()
        data = r.json()

    items = []
    for item in data.get("results", []) or []:
        items.append({
            "title": item.get("title", "") or "",
            "url": item.get("url", "") or "",
            "publishedDate": (
                item.get("published_date")
                or item.get("publishedDate")
                or ""
            ),
            "text": (item.get("content") or item.get("raw_content") or "")[:_MAX_TEXT_CHARS],
        })
    raw, parsed = _build_from_items(items, cutoff_date=cutoff_date)
    return await _apply_leak_filter(raw, parsed, cutoff_date=cutoff_date, question=question)


# ------------------------------------------------------------------- Exa

async def search_exa(
    query: str,
    *,
    cutoff_date: str,
    num_results: int = 10,
    question: str = "",
) -> tuple[str, list[dict], dict]:
    """Returns (raw_filtered_text, parsed_results, filter_debug)."""
    api_key = os.environ.get("EXA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("EXA_API_KEY not set (required for PF_SEARCH_BACKEND=exa)")

    q = _sanitize_query(query, max_chars=400)
    if not q:
        raise ValueError("query empty after sanitization")
    k = max(1, min(int(num_results), 20))

    payload: dict = {
        "query": q,
        "numResults": k,
        "type": "auto",
        "contents": {"text": {"maxCharacters": _MAX_TEXT_CHARS}},
    }
    if cutoff_date:
        payload["endPublishedDate"] = str(cutoff_date)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            _EXA_URL,
            json=payload,
            headers={"x-api-key": api_key, "Accept": "application/json"},
        )
        r.raise_for_status()
        data = r.json()

    items = []
    for item in data.get("results", []):
        items.append({
            "title": item.get("title", "") or "",
            "url": item.get("url", "") or "",
            "publishedDate": item.get("publishedDate") or item.get("published_date") or "",
            "text": (item.get("text") or item.get("snippet") or "")[:_MAX_TEXT_CHARS],
        })
    raw, parsed = _build_from_items(items, cutoff_date=cutoff_date)
    return await _apply_leak_filter(raw, parsed, cutoff_date=cutoff_date, question=question)


# ------------------------------------------------------------- AgentCore

def _parse_agentcore_payload(content_blocks: Any) -> list[dict]:
    """Extract result dicts from MCP tools/call content."""
    if content_blocks is None:
        return []
    texts: list[str] = []
    if isinstance(content_blocks, str):
        texts = [content_blocks]
    elif isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    texts.append(str(block["text"]))
                elif "text" in block:
                    texts.append(str(block["text"]))
            else:
                # mcp SDK TextContent
                t = getattr(block, "text", None)
                if t:
                    texts.append(str(t))
    elif isinstance(content_blocks, dict) and content_blocks.get("text"):
        texts = [str(content_blocks["text"])]

    items: list[dict] = []
    for text in texts:
        text = text.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: treat as a single snippet
            items.append({"title": "", "url": "", "text": text[:_MAX_TEXT_CHARS], "publishedDate": ""})
            continue
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            for r in data["results"]:
                if isinstance(r, dict):
                    items.append(r)
        elif isinstance(data, list):
            items.extend(r for r in data if isinstance(r, dict))
        elif isinstance(data, dict):
            items.append(data)
    return items


async def _agentcore_call_tool(query: str, max_results: int) -> list[dict]:
    """Invoke WebSearch on the AgentCore Gateway via SigV4 MCP."""
    from mcp import ClientSession
    from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client

    endpoint = agentcore_gateway_url()
    if not endpoint:
        raise ValueError(
            "AGENTCORE_GATEWAY_URL or AGENTCORE_GATEWAY_ID required for "
            "PF_SEARCH_BACKEND=agentcore"
        )
    region = agentcore_region()

    async with aws_iam_streamablehttp_client(
        endpoint=endpoint,
        aws_service="bedrock-agentcore",
        aws_region=region,
        timeout=45.0,
    ) as streams:
        read, write, _get_session_id = streams
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in (tools.tools or [])]
            name = next((n for n in _AGENTCORE_TOOL_NAMES if n in tool_names), None)
            if name is None:
                # fuzzy: any tool with 'search' in the name
                name = next((n for n in tool_names if "search" in n.lower()), None)
            if name is None:
                raise RuntimeError(
                    f"No WebSearch tool on AgentCore gateway; available={tool_names}"
                )
            result = await session.call_tool(
                name,
                arguments={"query": query, "maxResults": int(max_results)},
            )
            if getattr(result, "isError", False):
                err_txt = ""
                for block in result.content or []:
                    err_txt += getattr(block, "text", "") or str(block)
                raise RuntimeError(f"AgentCore WebSearch error: {err_txt[:500]}")
            return _parse_agentcore_payload(result.content)


async def search_agentcore(
    query: str,
    *,
    cutoff_date: str,
    num_results: int = 10,
    question: str = "",
) -> tuple[str, list[dict], dict]:
    """Amazon Bedrock AgentCore Web Search (us-east-1 Gateway MCP)."""
    # AgentCore query max is 200 chars
    q = _sanitize_query(query, max_chars=200)
    if not q:
        raise ValueError("query empty after sanitization")
    k = max(1, min(int(num_results), 25))

    items = await _agentcore_call_tool(q, k)
    raw, parsed = _build_from_items(items, cutoff_date=cutoff_date)
    return await _apply_leak_filter(raw, parsed, cutoff_date=cutoff_date, question=question)


# ------------------------------------------------------------------ cache

def _cache_enabled() -> bool:
    return os.environ.get("PF_SEARCH_CACHE", "1").strip().lower() not in ("0", "false", "no")


def _cache_dir() -> str:
    import tempfile
    d = os.environ.get("PF_SEARCH_CACHE_DIR", "").strip()
    if not d:
        d = os.path.join(tempfile.gettempdir(), "prime_forecast_search_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_key(backend: str, query: str, cutoff_date: str) -> str:
    import hashlib
    q = re.sub(r"\s+", " ", sanitize_search_query(query).lower()).strip()
    return hashlib.sha1(f"{backend}|{str(cutoff_date)[:10]}|{q}".encode()).hexdigest()


_MEM_CACHE: dict[str, tuple[str, list[dict]]] = {}


def _cache_get(key: str) -> tuple[str, list[dict]] | None:
    if key in _MEM_CACHE:
        return _MEM_CACHE[key]
    path = os.path.join(_cache_dir(), f"{key}.json")
    try:
        with open(path) as f:
            data = json.load(f)
        hit = (data["raw"], data["parsed"])
        _MEM_CACHE[key] = hit
        return hit
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def _cache_put(key: str, raw: str, parsed: list[dict], *, query: str, cutoff_date: str) -> None:
    _MEM_CACHE[key] = (raw, parsed)
    path = os.path.join(_cache_dir(), f"{key}.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            # query/cutoff kept alongside the results so the cache dir doubles
            # as a publishable corpus of everything the agent was shown.
            json.dump({"query": query, "cutoff_date": str(cutoff_date),
                       "raw": raw, "parsed": parsed}, f)
        os.replace(tmp, path)
    except OSError:
        pass


# ---------------------------------------------------------------- SearXNG

async def search_searxng(
    query: str,
    *,
    cutoff_date: str,
    num_results: int = 10,
    question: str = "",
) -> tuple[str, list[dict], dict]:
    """Self-hosted SearXNG metasearch (SEARXNG_URL). No native date filter —
    the leak-filter stack carries the cutoff, as with Brave. Instance must
    allow format=json in its settings."""
    base = os.environ.get("SEARXNG_URL", "").strip().rstrip("/")
    if not base:
        raise ValueError("SEARXNG_URL not set (required for PF_SEARCH_BACKEND=searxng)")

    q = _sanitize_query(query, max_chars=400)
    if not q:
        raise ValueError("query empty after sanitization")
    k = max(1, min(int(num_results), 20))

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(f"{base}/search", params={
            "q": q, "format": "json", "language": "en", "safesearch": 0,
        })
        r.raise_for_status()
        data = r.json()

    items = []
    for item in (data.get("results") or [])[: k * 2]:
        items.append({
            "title": item.get("title", "") or "",
            "url": item.get("url", "") or "",
            "publishedDate": item.get("publishedDate") or "",
            "text": (item.get("content") or "")[:_MAX_TEXT_CHARS],
        })
    raw, parsed = _build_from_items(items[: k + 5], cutoff_date=cutoff_date)
    return await _apply_leak_filter(raw, parsed, cutoff_date=cutoff_date, question=question)


# -------------------------------------------------------------- dispatcher

async def web_search(
    query: str,
    *,
    cutoff_date: str,
    num_results: int = 10,
    question: str = "",
) -> tuple[str, list[dict], dict]:
    """Dispatch to the configured search backend.

    Results are cached post-leak-filter, keyed on (backend, cutoff, normalized
    query): a hit costs zero search credits AND zero Bedrock filter calls, and
    GRPO groups re-searching the same phrasing get identical context. Disable
    with PF_SEARCH_CACHE=0; corpus lives in PF_SEARCH_CACHE_DIR.
    """
    backend = search_backend()
    if backend == "none":
        raise ValueError(
            "No search backend configured. Set PF_SEARCH_BACKEND=firecrawl|brave|tavily|exa|agentcore "
            "and the matching credentials."
        )
    if _cache_enabled():
        key = _cache_key(backend, query, cutoff_date)
        hit = _cache_get(key)
        if hit is not None:
            raw, parsed = hit
            if parsed:  # never serve a cached empty result set
                return raw, parsed, {"mode": "cache", "cache": "hit"}

    chain = [b.strip() for b in backend.split(",") if b.strip() and b.strip() != "none"] or [backend]
    impls = {
        "firecrawl": search_firecrawl,
        "brave": search_brave,
        "tavily": search_tavily,
        "agentcore": search_agentcore,
        "exa": search_exa,
        "searxng": search_searxng,
    }
    raw, parsed, debug = "", [], {}
    attempts = []
    for b in chain:
        try:
            raw, parsed, debug = await impls[b](
                query, cutoff_date=cutoff_date, num_results=num_results, question=question,
            )
        except Exception as e:
            attempts.append(f"{b}:error:{type(e).__name__}")
            if b == chain[-1] and not parsed:
                raise
            continue
        if parsed:
            attempts.append(f"{b}:ok")
            debug = {**debug, "backend_used": b}
            break
        attempts.append(f"{b}:empty")
    debug = {**debug, "attempts": attempts}

    if _cache_enabled() and parsed:  # never cache empties: transient upstream
        _cache_put(key, raw, parsed, query=query, cutoff_date=cutoff_date)  # bans would poison the corpus
        debug = {**debug, "cache": "miss"}
    return raw, parsed, debug


async def fetch_url_text(url: str, *, max_chars: int = 8000) -> str:
    """Fetch a URL and strip HTML tags naively. Raises on HTTP errors."""
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "PrimeForecast/0.1 (research bot)"})
        r.raise_for_status()
        text = r.text

    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]
