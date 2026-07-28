"""Layered leak filtering for retrieved content (ported from haruspex BLF).

Layers, applied in order:
  1. Domain blocklist — prediction-market pages are never fetched.
  2. Heuristic date filter — publish dates / post-cutoff facts (cutoff.py).
  3. Bedrock LLM filter — KEEP/DROP per result (Claude Haiku by default).
The summarizer (also Bedrock) re-filters its input and scrubs its output.
"""

from __future__ import annotations

import re

from prime_forecast.bedrock import chat, llm_enabled
from prime_forecast.cutoff import (
    find_all_dates_after,
    parse_ts,
    published_date_in_text,
    scrub_lines_after_cutoff,
    text_has_post_cutoff_facts,
)

RESULTS_SEPARATOR = "\n----------------------------------------------------\n"

BLOCKED_DOMAINS = (
    "polymarket.com", "manifold.markets", "metaculus.com",
    "kalshi.com", "predictit.org",
    # Mirrors/aggregators that republish resolved market pages with stale
    # publish dates (found leaking in live audits).
    "copilot.markets", "lines.com",
)

# Any URL that mentions a prediction-market brand or odds-page path is treated
# as blocked, regardless of host — mirrors resurface market pages under
# arbitrary domains (e.g. polymarket.copilot.markets).
_BLOCKED_URL_TOKENS = (
    "polymarket", "kalshi", "manifold", "metaculus", "predictit",
    "prediction-market", "prediction_market", "predictionmarket",
)

# Content-level guard: prediction-market odds pages leak resolved outcomes
# even when their metadata predates the cutoff. Restricted to market-brand /
# resolution markers — generic "% chance" phrasing is legitimate content
# (e.g. weather forecasts) and must NOT trip this.
_MARKET_CONTENT_RE = re.compile(
    r"(resolution verdict|market (?:resolved|settled)|trading odds"
    r"|polymarket|kalshi|manifold\.markets|metaculus|predictit)",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)
_SITE_OP_RE = re.compile(r"\bsite:[^\s]+", re.IGNORECASE)

_AGE_FLAG_RE = re.compile(
    r"  \[(?:ok, before \d{4}-\d{2}-\d{2}"
    r"|no known date"
    r"|\?\? unparseable date — check manually"
    r"|!{2} AFTER CUTOFF \d{4}-\d{2}-\d{2})\]"
)

_FILTER_PROMPT = """You are a strict information-cutoff filter. The knowledge cutoff is {cutoff} (inclusive).

For each numbered search result, reply one line: number, colon, KEEP or DROP, brief reason.

Rules (apply in order):
1. Published date AFTER {cutoff} → DROP (includes previews published later).
2. Post-cutoff outcomes, results, scores, resolutions, final reports → DROP.
3. Facts about events dated AFTER {cutoff} → DROP.
4. NO publish date shown? Estimate one from the content's reference frame:
   "what to expect" previews are written DAYS before the event they preview;
   reactions and wrap-ups are written after it. If the estimated publish date
   falls after {cutoff} → DROP. General analysis whose latest referenced fact
   predates {cutoff} may be kept.
5. Pre-cutoff news, previews PUBLISHED ON OR BEFORE {cutoff} → KEEP.
   (A preview of a future event is KEEP only if it was published by the cutoff.)
6. When unsure → DROP. Dropping a good result is cheap; keeping a leaky one is not.

OUTPUT: Reply ONLY with numbered lines (no preamble). Example:
1: KEEP
2: DROP

Forecast question: {question}

Results:
{numbered}"""


def domain_blocked(url: str) -> bool:
    u = (url or "").lower()
    if any(d in u for d in BLOCKED_DOMAINS):
        return True
    return any(t in u for t in _BLOCKED_URL_TOKENS)


def market_content_leak(text: str) -> bool:
    """True when retrieved content looks like a prediction-market odds page."""
    return bool(text and _MARKET_CONTENT_RE.search(text))


def redact_leaky_urls(text: str) -> str:
    """Replace prediction-market URLs in question/background (may encode outcomes)."""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        url = m.group(0)
        return "[URL redacted — market pages may leak outcomes]" if domain_blocked(url) else url

    return _URL_RE.sub(_repl, text)


def sanitize_search_query(query: str) -> str:
    """Strip site: operators (they can target blocked domains)."""
    q = str(query).strip()
    for dom in BLOCKED_DOMAINS:
        q = re.sub(rf"\bsite:{re.escape(dom)}[^\s]*", "", q, flags=re.IGNORECASE)
    q = _SITE_OP_RE.sub("", q)
    return re.sub(r"\s+", " ", q).strip()


def strip_age_flags(text: str) -> str:
    return _AGE_FLAG_RE.sub("", text)


def _strict_undated_drop(block: str, cutoff_str: str) -> bool:
    """Heuristic-only fallback rule (LLM filter unavailable): a result with no
    parseable publish date that mentions post-cutoff dates is presumed leaky
    (e.g. FOMC minutes pages published after the meeting carry no date in Exa).
    Mirrors the LLM prompt's "when unsure -> DROP"."""
    cutoff_dt = parse_ts(cutoff_str)
    if cutoff_dt is None:
        return False
    if published_date_in_text(block) is not None:
        return False  # dated results are handled by the standard heuristic
    return bool(find_all_dates_after(block, cutoff_dt))


def _parse_filter_decisions(text: str) -> dict[int, str]:
    """Parse KEEP/DROP; if a line wavers, the last token wins."""
    decisions: dict[int, str] = {}
    for line in text.splitlines():
        m = re.match(r"\[?(\d+)\]?\s*:", line.strip())
        if not m:
            continue
        tokens = re.findall(r"\b(KEEP|DROP)\b", line, re.IGNORECASE)
        if tokens:
            decisions[int(m.group(1))] = tokens[-1].upper()
    return decisions


async def filter_results(raw: str, cutoff_date: str, *, question: str = "") -> tuple[str, dict]:
    """Return (filtered_raw, debug). Heuristic pre-filter, then LLM on survivors.

    Bedrock auth/API failures propagate — callers (web_search) fail rather than
    silently weakening the leak filter.
    """
    debug: dict = {"role": "filter", "mode": "none", "decisions": {}, "heuristic_dropped": 0}
    if not raw or not raw.strip():
        return "", debug
    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""
    results = [r.strip() for r in raw.split(RESULTS_SEPARATOR) if r.strip()]
    if not results:
        return "", debug

    decisions: dict[int, str] = {}
    survivors: list[tuple[int, str]] = []
    for i, block in enumerate(results, 1):
        if text_has_post_cutoff_facts(block, cutoff_str) or market_content_leak(block):
            decisions[i] = "DROP"
        else:
            survivors.append((i, block))

    debug["heuristic_dropped"] = len(results) - len(survivors)

    if not survivors:
        debug.update(mode="heuristic", decisions=decisions)
        return "", debug

    if not llm_enabled() or not cutoff_str:
        kept = []
        for i, block in survivors:
            if _strict_undated_drop(block, cutoff_str):
                decisions[i] = "DROP"
                debug["heuristic_dropped"] += 1
            else:
                decisions[i] = "KEEP"
                kept.append(block)
        debug.update(mode="heuristic", decisions=decisions)
        return RESULTS_SEPARATOR.join(kept), debug

    cleaned = [strip_age_flags(block) for _, block in survivors]
    renumbered = "\n\n".join(f"[{j}]\n{block}" for j, block in enumerate(cleaned, 1))
    idx_map = {j: orig_i for j, (orig_i, _) in enumerate(survivors, 1)}
    prompt = _FILTER_PROMPT.format(
        cutoff=cutoff_str, question=question[:300], numbered=renumbered,
    )
    text, inp, out = await chat(prompt, max_tokens=len(survivors) * 60 + 200, role="filter")
    llm_dec = _parse_filter_decisions(text)

    kept_blocks: list[str] = []
    for j, (_, block) in enumerate(survivors, 1):
        orig_i = idx_map[j]
        vote = llm_dec.get(j, "DROP")  # default DROP when LLM silent
        decisions[orig_i] = vote
        if vote != "DROP":
            kept_blocks.append(block)

    debug.update(mode="llm", decisions=decisions, bedrock_reply=text,
                 input_tokens=inp, output_tokens=out)
    return RESULTS_SEPARATOR.join(kept_blocks), debug


async def filter_page_content(
    text: str,
    cutoff_date: str,
    *,
    question: str = "",
    url: str = "",
) -> tuple[str, dict]:
    """Single-page leak filter (lookup_url). Returns (filtered_text, debug)."""
    debug: dict = {"role": "filter_page", "mode": "none", "decisions": {}}
    if not text or not text.strip():
        return "", debug
    cutoff_str = str(cutoff_date)[:10] if cutoff_date else ""

    if text_has_post_cutoff_facts(text, cutoff_str) or market_content_leak(text):
        debug.update(mode="heuristic", decisions={1: "DROP"})
        return "", debug

    if not llm_enabled() or not cutoff_str:
        debug.update(mode="heuristic", decisions={1: "KEEP"})
        return text, debug

    prompt = _FILTER_PROMPT.format(
        cutoff=cutoff_str,
        question=question[:300],
        numbered=f"[1]\nURL: {url[:200]}\n{text[:8000]}",
    )
    resp, inp, out = await chat(prompt, max_tokens=80, role="filter")
    tokens = re.findall(r"\b(KEEP|DROP)\b", resp, re.IGNORECASE)
    decision = tokens[-1].upper() if tokens else "DROP"
    debug.update(mode="llm", decisions={1: decision}, bedrock_reply=resp,
                 input_tokens=inp, output_tokens=out)
    if decision == "DROP":
        return "", debug
    return text, debug


async def summarize_results(
    raw: str,
    *,
    question: str = "",
    cutoff_date: str = "",
    resolution_criteria: str = "",
    max_chars: int = 4000,
) -> tuple[str, dict]:
    """Summarize filtered search results. Re-filters input; scrubs output lines."""
    debug: dict = {"role": "summarize", "mode": "none"}
    if not raw or not raw.strip():
        return "No results to summarize.", debug

    filtered, fdebug = await filter_results(raw, cutoff_date, question=question)
    debug["prefilter"] = fdebug
    if not filtered or not filtered.strip():
        return "No pre-cutoff search results to summarize.", debug

    if not llm_enabled():
        parts = [p.strip()[:600] for p in filtered.split(RESULTS_SEPARATOR) if p.strip()]
        text = scrub_lines_after_cutoff("\n\n---\n\n".join(parts), cutoff_date)[:max_chars]
        debug.update(mode="passthrough")
        return text, debug

    ctx = ""
    if resolution_criteria:
        ctx = f"Resolution criteria: {resolution_criteria[:500]}\n\n"
    cutoff_str = str(cutoff_date)[:10]
    prompt = (
        f"You summarize web search results for a forecaster. Strict cutoff: {cutoff_str} (inclusive).\n"
        f"Question: {question[:400]}\n{ctx}"
        f"Only include facts knowable on or before {cutoff_str}. "
        "Do NOT include post-cutoff outcomes, results, scores, or resolutions.\n"
        "Do NOT cite odds or previews published after the cutoff.\n"
        "Bullet key points with source URLs. Cite as (search_INDEX_result_J) when indices appear.\n\n"
        f"Results:\n{filtered[:12000]}"
    )
    text, inp, out = await chat(prompt, max_tokens=800, role="summarize")
    summary = scrub_lines_after_cutoff(text, cutoff_date)
    if not summary.strip():
        summary = "No pre-cutoff facts could be extracted from the search results."
    summary = summary[:max_chars]
    debug.update(mode="llm", input_tokens=inp, output_tokens=out)
    return summary, debug
