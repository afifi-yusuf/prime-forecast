from datetime import datetime, timezone

from prime_forecast.cutoff import (
    clamp_end_date,
    find_all_dates_after,
    find_dates_after,
    parse_ts,
    scrub_lines_after_cutoff,
    text_has_post_cutoff_facts,
)

CUTOFF = "2026-05-01"
CUTOFF_DT = datetime(2026, 5, 1, tzinfo=timezone.utc)


def test_parse_ts_formats():
    assert parse_ts("2026-05-01") == CUTOFF_DT
    assert parse_ts("2026-05-01T12:30:00Z").hour == 12
    assert parse_ts(1774915200).year == 2026  # unix seconds
    assert parse_ts(1774915200000).year == 2026  # unix millis
    assert parse_ts("garbage") is None
    assert parse_ts(None) is None
    assert parse_ts("") is None


def test_find_dates_after_iso():
    text = "Event on 2026-04-30 was fine but 2026-05-02 is the future."
    hits = find_dates_after(text, CUTOFF_DT)
    assert hits == ["2026-05-02"]


def test_find_prose_dates_after():
    text = "Announced May 15, 2026 and also 3 June 2026; earlier was April 1, 2026."
    hits = find_all_dates_after(text, CUTOFF_DT)
    assert "May 15, 2026" in hits
    assert "3 June 2026" in hits
    assert all("April" not in h for h in hits)


def test_post_cutoff_publish_date_drops():
    block = "Some Title\nhttps://example.com\nPublished: 2026-05-10\npreview text"
    assert text_has_post_cutoff_facts(block, CUTOFF)


def test_pre_cutoff_publish_date_kept():
    block = "Some Title\nhttps://example.com\nPublished: 2026-04-20\npreview text"
    assert not text_has_post_cutoff_facts(block, CUTOFF)


def test_post_cutoff_outcome_language_drops():
    block = (
        "Match report\nhttps://example.com\nPublished: 2026-04-01\n"
        "Final score on 2026-05-03: they won 2-1."
    )
    assert text_has_post_cutoff_facts(block, CUTOFF)


def test_future_date_without_outcome_language_kept():
    # A pre-cutoff preview may legitimately mention a future event date.
    block = (
        "Preview\nhttps://example.com\nPublished: 2026-04-20\n"
        "The summit is scheduled for 2026-05-20 in Geneva."
    )
    assert not text_has_post_cutoff_facts(block, CUTOFF)


def test_scrub_lines_after_cutoff():
    text = "line ok 2026-04-01\nleaky line 2026-06-01\nanother ok line"
    out = scrub_lines_after_cutoff(text, CUTOFF)
    assert "leaky" not in out
    assert "another ok line" in out


def test_clamp_end_date_never_extends():
    assert clamp_end_date("2026-06-15", CUTOFF) == "2026-05-01"
    assert clamp_end_date("2026-04-15", CUTOFF) == "2026-04-15"
    assert clamp_end_date(None, CUTOFF) == "2026-05-01"
