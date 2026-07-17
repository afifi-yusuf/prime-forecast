"""Exa web search with publish-date filtering (BLF layer 1), async port.

Live web_search backend. Uses endPublishedDate=cutoff plus client-side date
drop, then the BLF layer-2 LLM filter (leak_filter.filter_results).
"""

from __future__ import annotations

import os

import httpx

from prime_forecast.cutoff import parse_ts
from prime_forecast.leak_filter import (
    RESULTS_SEPARATOR,
    domain_blocked,
    filter_results,
    sanitize_search_query,
)

_EXA_URL = "https://api.exa.ai/search"
_MAX_TEXT_CHARS = 2000


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
        raise ValueError("EXA_API_KEY not set (required for live web_search)")

    q = sanitize_search_query(query)
    q = q[:400].rsplit(" ", 1)[0] if len(q) > 400 else q
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

    blocks = []
    parsed = []
    cutoff_str = str(cutoff_date)[:10]
    cutoff_dt = parse_ts(cutoff_str)

    for item in data.get("results", []):
        url = item.get("url", "") or ""
        if domain_blocked(url):
            continue
        title = item.get("title", "") or ""
        pub = item.get("publishedDate") or item.get("published_date") or ""
        if cutoff_dt is not None and pub:
            d = parse_ts(pub)
            if d is not None and d > cutoff_dt:
                continue
        text = (item.get("text") or item.get("snippet") or "")[:_MAX_TEXT_CHARS]
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

    raw = RESULTS_SEPARATOR.join(blocks)
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
            parsed = []
        raw = filtered
    return raw, parsed, filter_debug


async def fetch_url_text(url: str, *, max_chars: int = 8000) -> str:
    """Fetch a URL and strip HTML tags naively. Raises on HTTP errors."""
    import re

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        r = await client.get(url, headers={"User-Agent": "PrimeForecast/0.1 (research bot)"})
        r.raise_for_status()
        text = r.text

    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:max_chars]
