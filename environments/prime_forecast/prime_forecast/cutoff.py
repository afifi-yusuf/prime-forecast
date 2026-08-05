"""Cutoff parsing + date-leak detection (ported from haruspex agent/cutoff.py).

Cutoff enforcement is the whole reason this environment is leak-safe. Ground
truth (outcome, post-cutoff prices) must never reach the agent context. These
helpers parse heterogeneous timestamps and scan tool observations for any date
that falls after the episode cutoff.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?")

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# Prose / EU-style dates common in news snippets.
PROSE_DATE_RES = (
    re.compile(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\s+(\d{4})\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(\d{1,2})[\./](\d{1,2})[\./](\d{4})\b"),
)

PUBLISHED_LINE_RE = re.compile(r"^Published:\s*(.+?)(?:\s+\[(?:ok|!!|\?\?).*)?$", re.MULTILINE)

# "FT" (full-time) must stay case-sensitive: lowercase "ft" is feet and shows
# up constantly on weather pages ("Elev 210 ft").
OUTCOME_LANGUAGE_RE = re.compile(
    r"\b(final score|full time|(?-i:FT)|won \d+\s*[-–]\s*\d+|lost \d+\s*[-–]\s*\d+|"
    r"defeated|were beaten|knocked off|comeback win|match report|boxscore|"
    r"resolved (?:yes|no)|market (?:resolved|settled))\b",
    re.IGNORECASE,
)


def parse_ts(value) -> datetime | None:
    """Parse a timestamp into an aware UTC datetime. Tolerant of date-only,
    ISO, trailing 'Z', and unix seconds/millis. Returns None if unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e12:  # milliseconds
            ts /= 1000.0
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    s = str(value).strip()
    if not s:
        return None
    s2 = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s2)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    m = ISO_DATE_RE.search(s)
    if m:
        y, mo, d, hh, mm, ss = m.groups()
        try:
            return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0),
                            int(ss or 0), tzinfo=timezone.utc)
        except ValueError:
            return None
    # Verbose formats, e.g. AgentCore's "02:01PM, Friday, July 31 2026, PDT"
    # or "July 31, 2026". Unparsed dates make results look undated, which
    # bypasses the hard post-cutoff drop (observed leak in v4 step-1 traces).
    m = _VERBOSE_DATE_RE.search(s)
    if m:
        month, day, year = m.group(1), int(m.group(2)), int(m.group(3))
        mo = _MONTHS.get(month.lower()[:3])
        if mo:
            try:
                return datetime(year, mo, day, tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_VERBOSE_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
    re.IGNORECASE,
)


def find_dates_after(text: str, cutoff: datetime) -> list[str]:
    """Return ISO date substrings in `text` strictly after `cutoff`."""
    if not text or cutoff is None:
        return []
    out = []
    for m in ISO_DATE_RE.finditer(text):
        dt = parse_ts(m.group(0))
        if dt is not None and dt > cutoff:
            out.append(m.group(0))
    return out


def _month_num(name: str) -> int | None:
    return _MONTHS.get(name.lower())


def _dt_from_prose_match(m: re.Match) -> datetime | None:
    g = m.groups()
    try:
        if m.re is PROSE_DATE_RES[0]:
            mo, day, year = _month_num(g[0]), int(g[1]), int(g[2])
        elif m.re is PROSE_DATE_RES[1]:
            day, mo, year = int(g[0]), _month_num(g[1]), int(g[2])
        else:
            day, mo, year = int(g[0]), int(g[1]), int(g[2])
        if not mo:
            return None
        return datetime(year, mo, day, tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def find_prose_dates_after(text: str, cutoff: datetime) -> list[str]:
    """Return prose date substrings strictly after `cutoff`."""
    if not text or cutoff is None:
        return []
    out: list[str] = []
    for pat in PROSE_DATE_RES:
        for m in pat.finditer(text):
            dt = _dt_from_prose_match(m)
            if dt is not None and dt > cutoff:
                out.append(m.group(0))
    return out


def find_all_dates_after(text: str, cutoff: datetime) -> list[str]:
    return find_dates_after(text, cutoff) + find_prose_dates_after(text, cutoff)


def published_date_in_text(text: str) -> datetime | None:
    m = PUBLISHED_LINE_RE.search(text or "")
    if not m:
        return None
    return parse_ts(m.group(1).strip())


def text_has_post_cutoff_facts(text: str, cutoff_date: str) -> bool:
    """True when publish date or post-cutoff factual content should be dropped."""
    cutoff_dt = parse_ts(str(cutoff_date)[:10])
    if cutoff_dt is None or not text:
        return False
    pub = published_date_in_text(text)
    if pub is not None and pub > cutoff_dt:
        return True
    post_dates = find_all_dates_after(text, cutoff_dt)
    if not post_dates:
        return False
    if OUTCOME_LANGUAGE_RE.search(text):
        return True
    if re.search(r"\b(won|lost|beat|defeated|scored|score|result|goals?|resolved)\b", text, re.IGNORECASE):
        return True
    return False


def scrub_lines_after_cutoff(text: str, cutoff_date: str) -> str:
    """Drop lines that mention any date strictly after cutoff (summaries/obs scrub)."""
    cutoff_dt = parse_ts(str(cutoff_date)[:10])
    if cutoff_dt is None or not text:
        return text
    kept = []
    for line in text.splitlines():
        if find_all_dates_after(line, cutoff_dt):
            continue
        kept.append(line)
    out = "\n".join(kept).strip()
    return out or "No pre-cutoff facts could be extracted from the sources."


def clamp_end_date(end_date: str | None, cutoff_date: str) -> str:
    """Clamp a model-supplied end date to the episode cutoff (YYYY-MM-DD)."""
    env_cutoff = str(cutoff_date)[:10]
    requested = str(end_date or cutoff_date)[:10]
    return env_cutoff if requested > env_cutoff else requested
