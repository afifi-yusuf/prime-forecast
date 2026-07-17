import pytest

from prime_forecast.leak_filter import (
    RESULTS_SEPARATOR,
    domain_blocked,
    filter_page_content,
    filter_results,
    market_content_leak,
    redact_leaky_urls,
    sanitize_search_query,
    summarize_results,
)

CUTOFF = "2026-05-01"
QUESTION = "Will X happen by June 2026?"


def _block(title: str, pub: str, body: str) -> str:
    return f"{title}\nhttps://example.com/{title.lower()}\nPublished: {pub}\n{body}"


def test_domain_blocklist():
    assert domain_blocked("https://polymarket.com/event/foo")
    assert domain_blocked("https://www.kalshi.com/markets/x")
    assert domain_blocked("https://manifold.markets/q")
    assert domain_blocked("https://metaculus.com/questions/123")
    assert not domain_blocked("https://reuters.com/article")


def test_domain_blocklist_catches_mirrors():
    # Mirrors republish resolved market pages under other hosts (live-audit find)
    assert domain_blocked("https://polymarket.copilot.markets/event/foo")
    assert domain_blocked("https://www.lines.com/prediction-markets/politics/x")
    assert domain_blocked("https://somehost.io/polymarket-odds/foo")


def test_market_content_leak_detects_odds_pages():
    assert market_content_leak("Resolution Verdict\nNO Market Resolved\nVolume $69.7K")
    assert market_content_leak("Trading Odds & Predictions | Polymarket")
    assert not market_content_leak("The Fed left rates unchanged in its April meeting.")
    # Legitimate forecast phrasing must survive (weather questions need it)
    assert not market_content_leak("Tomorrow there is a 70% chance of rain in London.")


@pytest.mark.asyncio
async def test_filter_results_drops_market_content_blocks():
    raw = RESULTS_SEPARATOR.join([
        _block("News", "2026-04-20", "pre-cutoff analysis of the situation"),
        _block("Odds", "2026-04-25", "Resolution Verdict NO Market Resolved"),
    ])
    filtered, debug = await filter_results(raw, CUTOFF, question=QUESTION)
    assert "News" in filtered
    assert "Odds" not in filtered


@pytest.mark.asyncio
async def test_filter_results_heuristic_drops_undated_post_cutoff():
    # No Published line + post-cutoff dates => presumed leaky in heuristic mode
    raw = "FOMC Minutes, June 16-17, 2026\nhttps://example.gov/minutes\nThe committee met on 2026-06-16."
    filtered, _ = await filter_results(raw, CUTOFF, question=QUESTION)
    assert filtered == ""


def test_redact_leaky_urls():
    text = "See https://polymarket.com/event/will-x and https://reuters.com/a"
    out = redact_leaky_urls(text)
    assert "polymarket.com" not in out
    assert "reuters.com" in out


def test_sanitize_search_query_strips_site_operators():
    q = sanitize_search_query("X outcome site:polymarket.com news site:reuters.com")
    assert "site:" not in q
    assert "news" in q


@pytest.mark.asyncio
async def test_filter_results_heuristic_drops_post_cutoff():
    raw = RESULTS_SEPARATOR.join([
        _block("Preview", "2026-04-20", "pre-cutoff preview text"),
        _block("Recap", "2026-05-10", "they won 3-1 final score"),
    ])
    filtered, debug = await filter_results(raw, CUTOFF, question=QUESTION)
    assert "Preview" in filtered
    assert "Recap" not in filtered
    assert debug["heuristic_dropped"] == 1


@pytest.mark.asyncio
async def test_filter_results_empty_input():
    filtered, _ = await filter_results("", CUTOFF)
    assert filtered == ""


@pytest.mark.asyncio
async def test_filter_page_content_drops_leaky_page():
    page = "Published: 2026-06-01\nThe market resolved YES after the event."
    filtered, debug = await filter_page_content(page, CUTOFF, question=QUESTION)
    assert filtered == ""
    assert debug["decisions"] == {1: "DROP"}


@pytest.mark.asyncio
async def test_filter_page_content_keeps_clean_page():
    page = "Published: 2026-04-01\nBackground about the topic, nothing after cutoff."
    filtered, _ = await filter_page_content(page, CUTOFF, question=QUESTION)
    assert filtered == page


@pytest.mark.asyncio
async def test_summarize_results_scrubs_post_cutoff_lines():
    raw = _block("Mixed", "2026-04-01",
                 "good fact from 2026-04-15\nleaky fact from 2026-05-20 they lost")
    summary, _ = await summarize_results(
        raw, question=QUESTION, cutoff_date=CUTOFF)
    assert "2026-05-20" not in summary
