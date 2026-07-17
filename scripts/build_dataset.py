#!/usr/bin/env python3
"""Build the prime-forecast dataset from Polymarket Gamma + CLOB APIs.

One row = one forecasting episode: a resolved binary YES/NO market with a
historical cutoff. Context is acquired at rollout time through the tool
harness — rows carry no articles or prompts.

Ported from haruspex scripts/build_polymarket_manifest.py with the Gamma
offset-pagination cap (~2100 rows) fixed: events are harvested in sliding
end-date windows (end_date_min/end_date_max), each window paginated with
offsets that stay under the cap; windows that still overflow are subdivided.

Filters (defaults):
  - closed/resolved binary YES/NO markets
  - cutoff 7 days before close/resolution (midpoint fallback for short markets)
  - cutoff price in [0.10, 0.90], volume >= 5000
  - temporal eligibility: resolution after the Qwen3.5 release window
  - sports/pop-culture excluded unless --include-sports/--include-popculture
  - temporal train/val/test split by resolution date (no cross-time leakage)

Example:
  python scripts/build_dataset.py --target-total 4000 --out-dir data --install
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environments" / "prime_forecast"))

from prime_forecast.cutoff import parse_ts  # noqa: E402

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
PKG_DATA_DIR = ROOT / "environments" / "prime_forecast" / "prime_forecast" / "data"

SPORT_TERMS = {
    "nba", "nfl", "mlb", "nhl", "ufc", "mma", "soccer", "football", "basketball",
    "baseball", "hockey", "tennis", "golf", "fifa", "premier league",
    "champions league", "laliga", "serie a", "bundesliga", "cricket", "wnba",
    "ncaa", "nascar", "formula 1", "f1", "wrestlemania", "super bowl",
}

POPCULTURE_TERMS = {
    "grammy", "oscars", "academy awards", "billboard", "box office", "netflix",
    "taylor swift", "drake", "kendrick", "celebrity", "movie", "album",
    "song", "youtube", "twitter", "x post", "instagram",
}

CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("politics_policy", (
        "election", "trump", "biden", "congress", "senate", "house", "supreme court",
        "scotus", "president", "governor", "mayor", "tariff", "bill", "law",
        "nomination", "cabinet", "fed chair", "state of the union", "government",
    )),
    ("crypto_finance", (
        "bitcoin", "btc", "ethereum", "eth", "solana", "xrp", "doge", "cardano",
        "ada", "bnb", "binance", "chainlink", "link", "litecoin", "ltc", "sui",
        "avalanche", "avax", "polkadot", "dot", "hyperliquid", "hype", "crypto",
        "etf", "coinbase", "stablecoin", "stock", "nasdaq", "s&p", "dow",
        "market cap",
    )),
    ("weather_climate", (
        "temperature", "highest temperature", "lowest temperature", "weather",
        "rain", "snow", "hurricane", "storm", "wind", "forecast",
    )),
    ("earth_science", (
        "earthquake", "magnitude", "volcano", "wildfire",
    )),
    ("macro_economics", (
        "fed", "fomc", "interest rate", "cpi", "inflation", "jobs report", "payroll",
        "unemployment", "gdp", "recession", "treasury", "oil", "wti", "gold",
        "silver", "mortgage", "bank", "ecb", "boe",
    )),
    ("ai_tech", (
        "ai", "openai", "anthropic", "google", "gemini", "gpt", "claude",
        "llama", "qwen", "nvidia", "apple", "microsoft", "tesla", "model",
        "benchmark", "chip", "semiconductor",
    )),
    ("geopolitics", (
        "ukraine", "russia", "china", "taiwan", "israel", "iran", "gaza",
        "ceasefire", "war", "nato", "missile", "sanction", "houthi", "north korea",
    )),
]


@dataclass
class Candidate:
    row: dict[str, Any]
    sort_key: tuple[float, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prime-forecast dataset from Polymarket")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--target-total", type=int, default=4000)
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--max-offset", type=int, default=2000,
                   help="Gamma offset pagination cap per window (hard API limit ~2100).")
    p.add_argument("--window-days", type=float, default=3.0,
                   help="Initial end-date window width; subdivided when a window overflows.")
    p.add_argument("--eligible-after", default="2026-03-02",
                   help="Only markets RESOLVING after this date (Qwen3.5 small-series release). "
                        "Empty disables.")
    p.add_argument("--eligible-before", default="",
                   help="Only markets resolving before this date (default: now).")
    p.add_argument("--horizon-days", type=float, default=7.0)
    p.add_argument("--min-before-resolution-hours", type=float, default=24.0)
    p.add_argument("--min-after-start-hours", type=float, default=24.0)
    p.add_argument("--price-window-days", type=float, default=7.0)
    p.add_argument("--min-price", type=float, default=0.10)
    p.add_argument("--max-price", type=float, default=0.90)
    p.add_argument("--min-volume", type=float, default=5000.0)
    p.add_argument("--include-sports", action="store_true")
    p.add_argument("--include-popculture", action="store_true")
    p.add_argument("--max-sports-frac", type=float, default=0.10)
    p.add_argument("--max-category-frac", type=float, default=0.35)
    p.add_argument("--train-frac", type=float, default=0.80)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--fetch-prices", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel workers for window harvest and price fetches.")
    p.add_argument("--sleep", type=float, default=0.02)
    p.add_argument("--install", action="store_true",
                   help="Copy slim splits into the environment package data dir.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


_client = httpx.Client(
    timeout=30.0,
    headers={"Accept": "application/json", "User-Agent": "PrimeForecastDataset/0.1"},
)


def http_get(url: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = _client.get(url, params=params or {})
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            raise RuntimeError(f"HTTP {e.response.status_code} for {e.request.url}: {body}") from e
        except httpx.HTTPError as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} retries: {last_err}")


def json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return [s]
    return [value]


def first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = d.get(key)
        if value not in (None, "", []):
            return value
    return None


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def gamma_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("data", "events", "markets", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [data]
    return []


def _fetch_window(start: datetime, end: datetime, args: argparse.Namespace) -> tuple[list[dict], bool]:
    """Fetch all closed events whose end date falls in [start, end).

    Returns (rows, overflowed). overflowed=True means the window hit the
    offset cap and should be subdivided.
    """
    rows: list[dict] = []
    offset = 0
    while offset <= args.max_offset:
        params = {
            "closed": "true",
            "limit": args.page_size,
            "offset": offset,
            "order": "endDate",
            "ascending": "true",
            "end_date_min": iso(start),
            "end_date_max": iso(end),
        }
        try:
            data = http_get(f"{GAMMA_BASE}/events", params=params)
        except RuntimeError as e:
            if "offset" in str(e).lower():
                return rows, True
            raise
        page = gamma_rows(data)
        if not page:
            return rows, False
        rows.extend(page)
        if len(page) < args.page_size:
            return rows, False
        offset += args.page_size
        time.sleep(args.sleep)
    return rows, True


def _harvest_window(start: datetime, end: datetime, width: float, args: argparse.Namespace) -> list[dict]:
    """Fetch one window, subdividing on offset overflow."""
    out: list[dict] = []
    sub = [(start, end, width)]
    while sub:
        s, e, w = sub.pop()
        rows, overflowed = _fetch_window(s, e, args)
        if overflowed and w > 0.25:
            mid = s + (e - s) / 2
            print(f"  window {iso(s)}..{iso(e)} overflowed; subdividing")
            sub.append((mid, e, w / 2))
            sub.append((s, mid, w / 2))
            continue
        out.extend(rows)
    return out


def iter_events(args: argparse.Namespace):
    """Sliding end-date windows across the eligibility range (keyset-free), in parallel."""
    lo = parse_ts(args.eligible_after) or datetime(2026, 3, 2, tzinfo=timezone.utc)
    hi = parse_ts(args.eligible_before) if args.eligible_before else datetime.now(timezone.utc)

    windows: list[tuple[datetime, datetime]] = []
    cur = lo
    while cur < hi:
        nxt = min(cur + timedelta(days=args.window_days), hi)
        windows.append((cur, nxt))
        cur = nxt

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_harvest_window, s, e, args.window_days, args): (s, e)
                   for s, e in windows}
        for fut in as_completed(futures):
            done += 1
            if done % 10 == 0:
                print(f"windows done: {done}/{len(windows)}")
            yield from fut.result()


def extract_markets(event: dict[str, Any]) -> list[dict[str, Any]]:
    markets = event.get("markets")
    if isinstance(markets, list) and markets:
        return [m for m in markets if isinstance(m, dict)]
    return [event]


def text_blob(event: dict[str, Any], market: dict[str, Any]) -> str:
    pieces = []
    for obj in (event, market):
        for key in (
            "title", "question", "description", "resolutionSource", "rules",
            "category", "subcategory", "slug", "seriesSlug",
        ):
            if obj.get(key):
                pieces.append(str(obj[key]))
        for tag in json_list(obj.get("tags")):
            if isinstance(tag, dict):
                pieces.extend(str(tag.get(k, "")) for k in ("label", "name", "slug"))
            else:
                pieces.append(str(tag))
    return " ".join(p for p in pieces if p).lower()


def is_sports(blob: str) -> bool:
    if " vs. " in blob or " vs " in blob or re.search(r"\bfc\b", blob):
        return True
    return any(term in blob for term in SPORT_TERMS)


def is_popculture(blob: str) -> bool:
    return any(term in blob for term in POPCULTURE_TERMS)


def classify(blob: str) -> str:
    if is_sports(blob):
        return "sports"
    if is_popculture(blob):
        return "popculture"
    for category, terms in CATEGORY_RULES:
        if any(term in blob for term in terms):
            return category
    return "other"


def binary_yes_no(market: dict[str, Any]) -> tuple[int, int, list[str], list[str]] | None:
    outcomes = [str(x).strip() for x in json_list(market.get("outcomes"))]
    token_ids = [str(x).strip() for x in json_list(
        market.get("clobTokenIds") or market.get("clob_token_ids") or market.get("token_ids")
    )]
    if len(outcomes) != 2 or len(token_ids) < 2:
        return None
    lowered = [x.lower() for x in outcomes]
    if "yes" not in lowered or "no" not in lowered:
        return None
    yes_i = lowered.index("yes")
    no_i = lowered.index("no")
    return yes_i, no_i, outcomes, token_ids


def infer_binary_outcome(market: dict[str, Any], yes_i: int, no_i: int) -> int | None:
    for key in ("resolvedOutcome", "resolved_outcome", "winningOutcome", "winner", "outcome"):
        value = market.get(key)
        if isinstance(value, str):
            s = value.strip().lower()
            if s == "yes":
                return 1
            if s == "no":
                return 0
    prices = [as_float(x) for x in json_list(market.get("outcomePrices") or market.get("outcome_prices"))]
    if len(prices) >= 2 and prices[yes_i] is not None and prices[no_i] is not None:
        yes_p = float(prices[yes_i])
        no_p = float(prices[no_i])
        if yes_p >= 0.98 and no_p <= 0.02:
            return 1
        if no_p >= 0.98 and yes_p <= 0.02:
            return 0
    return None


def market_volume(market: dict[str, Any], event: dict[str, Any]) -> float:
    value = first_present(market, ("volume", "volumeNum", "volume_num", "volumeClob", "volume24hr"))
    if value is None:
        value = first_present(event, ("volume", "volumeNum", "volume_num", "volumeClob"))
    return as_float(value) or 0.0


def choose_cutoff(
    created_at: datetime | None,
    resolution_at: datetime | None,
    *,
    horizon_days: float,
    min_after_start_hours: float,
    min_before_resolution_hours: float,
) -> tuple[datetime | None, str]:
    if resolution_at is None:
        return None, "missing_resolution_date"

    latest = resolution_at - timedelta(hours=min_before_resolution_hours)
    if created_at is None:
        candidate = resolution_at - timedelta(days=horizon_days)
        if candidate > latest:
            candidate = latest
        return candidate if candidate < resolution_at else None, "horizon_no_start"

    earliest = created_at + timedelta(hours=min_after_start_hours)
    if earliest >= latest:
        return None, "too_short"

    candidate = resolution_at - timedelta(days=horizon_days)
    if earliest <= candidate <= latest:
        return candidate, "horizon"

    # Shorter market: use midpoint of the valid interval instead of a very-late cutoff.
    midpoint = earliest + (latest - earliest) / 2
    return midpoint, "midpoint_fallback"


def fetch_price_history(token_id: str, cutoff: datetime, window_days: float, sleep: float) -> list[dict[str, Any]]:
    start = int((cutoff - timedelta(days=window_days)).timestamp())
    end = int(cutoff.timestamp())
    data = http_get(f"{CLOB_BASE}/prices-history", params={
        "market": token_id,
        "startTs": start,
        "endTs": end,
        "fidelity": 60,
    })
    time.sleep(sleep)
    raw = data.get("history") if isinstance(data, dict) else data
    out: list[dict[str, Any]] = []
    for pt in raw or []:
        if not isinstance(pt, dict):
            continue
        t = pt.get("t") or pt.get("time") or pt.get("timestamp")
        p = as_float(pt.get("p") if "p" in pt else pt.get("price"))
        dt = parse_ts(t)
        if p is None or dt is None or dt > cutoff:
            continue
        out.append({"t": iso(dt), "p": p})
    out.sort(key=lambda x: str(x.get("t") or ""))
    return out


def build_row(
    event: dict[str, Any],
    market: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str]:
    binary = binary_yes_no(market)
    if not binary:
        return None, "not_binary_yes_no"
    yes_i, no_i, outcomes, token_ids = binary
    outcome = infer_binary_outcome(market, yes_i, no_i)
    if outcome is None:
        return None, "missing_outcome"

    blob = text_blob(event, market)
    category = classify(blob)
    if category == "sports" and not args.include_sports:
        return None, "sports_excluded"
    if category == "popculture" and not args.include_popculture:
        return None, "popculture_excluded"

    volume = market_volume(market, event)
    if volume < args.min_volume:
        return None, "low_volume"

    created_at = parse_ts(first_present(market, ("createdAt", "created_at", "startDate", "start_date")))
    if created_at is None:
        created_at = parse_ts(first_present(event, ("createdAt", "created_at", "startDate", "start_date")))
    resolution_at = parse_ts(first_present(market, (
        "closedTime", "closed_time", "resolvedAt", "resolved_at", "endDate", "end_date", "endDateIso",
    )))
    if resolution_at is None:
        resolution_at = parse_ts(first_present(event, (
            "closedTime", "closed_time", "resolvedAt", "resolved_at", "endDate", "end_date", "endDateIso",
        )))

    eligible_after = parse_ts(args.eligible_after) if args.eligible_after else None
    if eligible_after is not None:
        if resolution_at is None or resolution_at < eligible_after:
            return None, "pre_eligible"

    cutoff, cutoff_policy = choose_cutoff(
        created_at,
        resolution_at,
        horizon_days=args.horizon_days,
        min_after_start_hours=args.min_after_start_hours,
        min_before_resolution_hours=args.min_before_resolution_hours,
    )
    if cutoff is None:
        return None, cutoff_policy

    market_id = str(first_present(market, ("id", "conditionId", "condition_id")) or "")
    condition_id = str(first_present(market, ("conditionId", "condition_id")) or "")
    slug = str(first_present(market, ("slug",)) or first_present(event, ("slug",)) or market_id)
    event_id = str(first_present(event, ("id", "ticker", "slug")) or slug)
    question = str(first_present(market, ("question", "title", "groupItemTitle"))
                   or first_present(event, ("title", "question")) or "")
    resolution_criteria = str(first_present(market, (
        "description", "resolutionCriteria", "resolution_criteria", "rules", "resolutionSource",
    )) or first_present(event, ("description", "resolutionCriteria", "rules", "resolutionSource")) or "")
    if not question.strip() or not resolution_criteria.strip():
        return None, "missing_question_or_resolution"

    row = {
        "example_id": f"pm-{market_id or condition_id or slug}",
        "market_id": market_id or condition_id or slug,
        "event_id": event_id,
        "slug": slug,
        "event_slug": event.get("slug"),
        "question": question,
        "resolution_criteria": resolution_criteria,
        "cutoff_date": iso(cutoff),
        "resolution_date": iso(resolution_at),
        "created_at": iso(created_at),
        "outcome": int(outcome),
        "market_type": "binary",
        "outcomes": outcomes,
        "token_ids": token_ids,
        "token_id": token_ids[yes_i],
        "condition_id": condition_id,
        "category": category,
        "volume": volume,
        "price_at_cutoff": None,
        "price_history": [],
        "cutoff_policy": cutoff_policy,
        "source": "polymarket_gamma_clob",
    }
    return row, "accepted"


def attach_price(row: dict[str, Any], args: argparse.Namespace) -> str:
    """Fetch YES price history up to cutoff; returns 'accepted' or a reject reason."""
    cutoff = parse_ts(row["cutoff_date"])
    try:
        price_history = fetch_price_history(
            row["token_id"], cutoff, args.price_window_days, args.sleep)
    except Exception as e:  # noqa: BLE001
        return f"price_fetch_failed:{type(e).__name__}"
    if not price_history:
        return "missing_price_history"
    price_at_cutoff = as_float(price_history[-1].get("p"))
    if price_at_cutoff is None:
        return "missing_price_at_cutoff"
    if not (args.min_price <= price_at_cutoff <= args.max_price):
        return "extreme_price"
    row["price_history"] = price_history
    row["price_at_cutoff"] = price_at_cutoff
    return "accepted"


class BalancedSelector:
    """Incremental category-capped selection (highest-volume first)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.selected: list[dict[str, Any]] = []
        self.counts: Counter[str] = Counter()
        self.sports_cap = int(args.target_total * args.max_sports_frac)
        self.category_cap = max(1, int(args.target_total * args.max_category_frac))

    def wants(self, row: dict[str, Any]) -> bool:
        cat = row["category"]
        if cat == "sports" and self.counts[cat] >= self.sports_cap:
            return False
        return self.counts[cat] < self.category_cap

    def add(self, row: dict[str, Any]) -> None:
        self.selected.append(row)
        self.counts[row["category"]] += 1

    @property
    def full(self) -> bool:
        return len(self.selected) >= self.args.target_total


def temporal_split(rows: list[dict[str, Any]], train_frac: float, val_frac: float) -> None:
    """Split by resolution date: oldest -> train, newest -> test."""
    rows.sort(key=lambda r: str(r.get("resolution_date") or ""))
    n = len(rows)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    for i, row in enumerate(rows):
        if i < n_train:
            row["split"] = "train"
        elif i < n_train + n_val:
            row["split"] = "val"
        else:
            row["split"] = "test"


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, slim: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            out = {k: v for k, v in row.items() if k != "price_history"} if slim else row
            f.write(json.dumps(out, default=str, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    candidates: list[Candidate] = []
    reject_counts: Counter[str] = Counter()
    seen_market_ids: set[str] = set()
    scanned_events = 0
    scanned_markets = 0

    for event in iter_events(args):
        scanned_events += 1
        for market in extract_markets(event):
            scanned_markets += 1
            row, reason = build_row(event, market, args)
            if row is None:
                reject_counts[reason] += 1
                continue
            key = str(row["market_id"])
            if key in seen_market_ids:
                reject_counts["duplicate_market"] += 1
                continue
            seen_market_ids.add(key)
            volume = as_float(row.get("volume")) or 0.0
            candidates.append(Candidate(row=row, sort_key=(math.log1p(volume), str(row["cutoff_date"]))))

        if scanned_events % 200 == 0:
            print(
                f"scanned_events={scanned_events} scanned_markets={scanned_markets} "
                f"candidates={len(candidates)} rejects={sum(reject_counts.values())}"
            )

    # Price pass: fetch CLOB history for the best candidates until target met.
    candidates.sort(key=lambda c: c.sort_key, reverse=True)
    selector = BalancedSelector(args)
    chunk_size = max(50, args.workers * 25)
    idx = 0
    if not args.fetch_prices:
        for cand in candidates:
            if selector.full:
                break
            if selector.wants(cand.row):
                selector.add(cand.row)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while idx < len(candidates) and not selector.full:
                chunk = [c.row for c in candidates[idx: idx + chunk_size] if selector.wants(c.row)]
                idx += chunk_size
                futures = {pool.submit(attach_price, row, args): row for row in chunk}
                accepted_rows = []
                for fut in as_completed(futures):
                    row = futures[fut]
                    reason = fut.result()
                    if reason == "accepted":
                        accepted_rows.append(row)
                    else:
                        reject_counts[reason] += 1
                # Preserve volume ordering within the chunk.
                accepted_rows.sort(key=lambda r: -(as_float(r.get("volume")) or 0.0))
                for row in accepted_rows:
                    if selector.full:
                        break
                    if selector.wants(row):
                        selector.add(row)
                print(f"price pass: {min(idx, len(candidates))}/{len(candidates)} candidates, "
                      f"selected={len(selector.selected)}")

    selected = selector.selected
    temporal_split(selected, args.train_frac, args.val_frac)

    out_dir = ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    summary = {
        "created_at": iso(datetime.now(timezone.utc)),
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "scanned_events": scanned_events,
        "scanned_markets": scanned_markets,
        "candidates": len(candidates),
        "selected": len(selected),
        "categories": dict(Counter(r["category"] for r in selected)),
        "outcomes": dict(Counter(r["outcome"] for r in selected)),
        "splits": dict(Counter(r["split"] for r in selected)),
        "cutoff_policy": dict(Counter(r["cutoff_policy"] for r in selected)),
        "rejects": dict(reject_counts),
    }
    print(json.dumps(summary, indent=2, default=str))
    if args.dry_run:
        return

    write_jsonl(out_dir / "manifest.jsonl", selected)
    for split in ("train", "val", "test"):
        rows = [r for r in selected if r["split"] == split]
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {len(selected)} rows to {out_dir}")

    if args.install:
        PKG_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            rows = [r for r in selected if r["split"] == split]
            write_jsonl(PKG_DATA_DIR / f"{split}.jsonl", rows, slim=True)
        print(f"Installed slim splits into {PKG_DATA_DIR}")


if __name__ == "__main__":
    main()
