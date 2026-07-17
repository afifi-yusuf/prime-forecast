"""Polymarket read-only API tools (Gamma + CLOB), async port from haruspex.

Public endpoints per https://docs.polymarket.com/api-reference/introduction
  Gamma: https://gamma-api.polymarket.com  — market metadata, search
  CLOB:  https://clob.polymarket.com       — price history (no auth for reads)

LEAKAGE: never return live midpoint/price for historical cutoffs. Always use
CLOB /prices-history with endTs = cutoff. Metadata responses are stripped to
static contract identity (Gamma payloads contain resolution/winner fields).

These tools are DISABLED by default in the environment (crowd-prior copying
collapsed GRPO groups in haruspex runs); enable via include_market_tools=True.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from prime_forecast.cutoff import parse_ts

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

_HEADERS = {"Accept": "application/json", "User-Agent": "PrimeForecast/0.1"}


async def _http_get(url: str, params: dict | None = None, timeout: float = 25.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params or {}, headers=_HEADERS)
        r.raise_for_status()
        return r.json()


def _parse_json_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [val]
    return [val]


def _text_score(a: str, b: str) -> float:
    wa = set(re.findall(r"\w+", (a or "").lower()))
    wb = set(re.findall(r"\w+", (b or "").lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _gamma_rows(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "markets", "events", "results"):
            v = data.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [data]
    return []


async def _gamma_search(query: str, limit: int = 5) -> list[dict]:
    try:
        data = await _http_get(f"{GAMMA_BASE}/public-search",
                               params={"q": query, "limit_per_type": limit})
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    for key in ("events", "markets"):
        for item in data.get(key) or []:
            if isinstance(item, dict):
                out.append({**item, "_gamma_kind": key[:-1]})
    return out


def _normalize_history_points(raw: Any, cutoff_date: str) -> list[dict]:
    """Extract [{t, p}] and filter to <= cutoff."""
    if isinstance(raw, dict):
        pts = raw.get("history") or raw.get("points") or raw.get("prices") or []
    elif isinstance(raw, list):
        pts = raw
    else:
        pts = []
    cutoff = parse_ts(cutoff_date)
    out = []
    for pt in pts:
        if not isinstance(pt, dict):
            continue
        ts = pt.get("t") or pt.get("time") or pt.get("timestamp")
        price = pt.get("p") if "p" in pt else pt.get("price")
        if price is None:
            continue
        try:
            p = float(price)
        except (TypeError, ValueError):
            continue
        d = parse_ts(ts)
        if cutoff is not None and d is not None and d > cutoff:
            continue
        out.append({"t": ts, "p": p})
    return out


def _cutoff_unix(cutoff_date: str) -> int | None:
    dt = parse_ts(cutoff_date)
    return int(dt.timestamp()) if dt else None


async def get_market_metadata(ctx: dict) -> str:
    """Cutoff-safe metadata for the episode's own market (dataset fields only).

    We never expose the raw Gamma payload live — it contains resolution status
    and winner fields. The dataset row already holds everything static.
    """
    return json.dumps({
        "market_id": ctx.get("market_id"),
        "question": ctx.get("question"),
        "resolution_criteria": ctx.get("resolution_criteria"),
        "resolution_date": ctx.get("resolution_date"),
        "slug": ctx.get("slug"),
        "token_id": ctx.get("token_id"),
        "cutoff_date": ctx.get("cutoff_date"),
        "category": ctx.get("category"),
        "source": "dataset_manifest",
    }, default=str)


async def search_markets(query: str, ctx: dict, limit: int = 5) -> str:
    results = []
    for item in (await _gamma_search(query, limit=limit))[:limit]:
        results.append({
            "question": item.get("title") or item.get("question"),
            "slug": item.get("slug"),
            "market_id": item.get("id") or item.get("conditionId"),
            "kind": item.get("_gamma_kind"),
        })
    return json.dumps({"query": query, "results": results, "source": "gamma_search"}, default=str)


async def market_price(ctx: dict) -> str:
    """YES implied probability at cutoff — never the live orderbook mid."""
    cutoff_date = str(ctx.get("cutoff_date") or "")
    tok = ctx.get("token_id")
    end_ts = _cutoff_unix(cutoff_date)
    if tok and end_ts is not None:
        try:
            raw = await _http_get(f"{CLOB_BASE}/prices-history", params={
                "market": str(tok),
                "startTs": max(0, end_ts - 7 * 86400),
                "endTs": end_ts,
                "fidelity": 60,
            })
            pts = _normalize_history_points(raw, cutoff_date)
            if pts:
                p = float(pts[-1]["p"])
                return json.dumps({
                    "yes_price_at_cutoff": p,
                    "implied_probability": p,
                    "as_of": cutoff_date,
                    "source": "clob_prices_history",
                }, default=str)
        except Exception as e:  # noqa: BLE001
            return json.dumps({
                "error": f"clob price at cutoff failed: {e}",
                "fallback_price": ctx.get("price_at_cutoff"),
            }, default=str)

    return json.dumps({
        "yes_price_at_cutoff": ctx.get("price_at_cutoff"),
        "as_of": cutoff_date,
        "source": "dataset_fallback",
    }, default=str)


async def price_history(ctx: dict, interval: str = "1d", fidelity: int = 60) -> str:
    cutoff_date = str(ctx.get("cutoff_date") or "")
    tok = ctx.get("token_id")
    end_ts = _cutoff_unix(cutoff_date)
    if not tok:
        return json.dumps({"error": "no token_id for this market"})

    params: dict = {"market": str(tok), "fidelity": fidelity}
    if end_ts is not None:
        window = {"1h": 3600, "6h": 6 * 3600, "1d": 86400, "1w": 7 * 86400,
                  "1m": 30 * 86400}.get(interval, 30 * 86400)
        params["startTs"] = max(0, end_ts - window)
        params["endTs"] = end_ts
    else:
        params["interval"] = interval

    try:
        raw = await _http_get(f"{CLOB_BASE}/prices-history", params=params)
        pts = _normalize_history_points(raw, cutoff_date)
        return json.dumps({
            "interval": interval,
            "points": pts,
            "cutoff_date": cutoff_date,
            "source": "clob_prices_history",
        }, default=str)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"clob prices-history failed: {e}"})
