"""Cutoff-safe data tools: yfinance, FRED, DBnomics, Wikipedia, trend analysis.

Async ports of the haruspex BLF data tools. All series are truncated at the
episode cutoff; model-supplied end dates are clamped (never extended).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import urllib.parse
from datetime import datetime, timedelta

import httpx

from prime_forecast.cutoff import clamp_end_date

_FRED_BASE = "https://api.stlouisfed.org/fred"
_DBNOMICS_API = "https://api.db.nomics.world/v22/series"
_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_MAX_CHARS = 20000
_HEADERS = {"Accept": "application/json", "User-Agent": "PrimeForecast/0.1 (cutoff-safe research)"}


def default_start(end: str, days: int = 730) -> str:
    end_dt = datetime.strptime(end[:10], "%Y-%m-%d")
    return (end_dt - timedelta(days=days)).strftime("%Y-%m-%d")


def subsample_rows(rows: list[dict], max_rows: int = 30) -> list[dict]:
    """Return up to max_rows with recent + sampled history."""
    if len(rows) <= max_rows:
        return rows
    n_recent = max_rows // 2
    n_hist = max_rows - n_recent
    recent = rows[-n_recent:]
    older = rows[:-n_recent]
    if len(older) <= n_hist:
        return older + recent
    step = max(1, len(older) // n_hist)
    sampled = [older[i] for i in range(0, len(older), step)][:n_hist]
    return sampled + recent


async def _http_get_json(url: str, params: dict, timeout: float = 30.0) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params, headers=_HEADERS)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------- yfinance

def _yf_download_sync(ticker: str, start: str, end: str) -> list[dict]:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return []
    # yfinance >= 0.2.40 returns MultiIndex columns even for a single ticker
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    series = df["Close"] if "Close" in df.columns else df.iloc[:, -1]
    return [
        {"date": str(idx)[:10], "value": float(val)}
        for idx, val in series.items()
    ]


async def fetch_yfinance_rows(
    ticker: str, cutoff_date: str, *, end_date: str | None = None,
) -> tuple[list[dict], str | None]:
    end = clamp_end_date(end_date, cutoff_date)
    start = default_start(end, days=730)
    try:
        rows = await asyncio.to_thread(_yf_download_sync, ticker, start, end)
    except Exception as e:  # noqa: BLE001
        return [], f"yfinance failed: {e}"
    return rows, None


async def fetch_yfinance(ticker: str, cutoff_date: str, *, end_date: str | None = None) -> str:
    rows, err = await fetch_yfinance_rows(ticker, cutoff_date, end_date=end_date)
    if err:
        return json.dumps({"error": err})
    end = clamp_end_date(end_date, cutoff_date)
    if not rows:
        return json.dumps({"ticker": ticker, "rows": [], "note": "no data up to cutoff"})
    return json.dumps({
        "ticker": ticker,
        "end_date": end,
        "rows": subsample_rows(rows),
        "num_points": len(rows),
    }, default=str)


# -------------------------------------------------------------------- FRED

async def fetch_fred_rows(
    series_id: str, cutoff_date: str, *, end_date: str | None = None,
) -> tuple[list[dict], str | None]:
    import os

    api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not api_key:
        return [], "FRED_API_KEY not set (free key at fred.stlouisfed.org)"
    sid = str(series_id).strip().upper()
    if not sid:
        return [], "series_id required"

    end = clamp_end_date(end_date, cutoff_date)
    start = default_start(end, days=730)
    try:
        obs = await _http_get_json(f"{_FRED_BASE}/series/observations", {
            "series_id": sid,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "asc",
        })
    except Exception as e:  # noqa: BLE001
        return [], f"fred request failed: {e}"

    rows = []
    for item in obs.get("observations", []):
        val = item.get("value")
        if val in (None, ".", ""):
            continue
        try:
            rows.append({"date": item.get("date"), "value": float(val)})
        except (TypeError, ValueError):
            continue
    return rows, None


async def fetch_fred(
    series_id: str, cutoff_date: str, *, end_date: str | None = None, limit: int = 24,
) -> str:
    import os

    sid = str(series_id).strip().upper()
    if not sid:
        return json.dumps({"error": "series_id required (e.g. UNRATE, CPIAUCSL, FEDFUNDS)"})
    end = clamp_end_date(end_date, cutoff_date)
    rows, err = await fetch_fred_rows(sid, cutoff_date, end_date=end)
    if err:
        return json.dumps({"error": err, "series_id": sid})

    meta: dict = {}
    api_key = os.environ.get("FRED_API_KEY", "").strip()
    try:
        meta = (await _http_get_json(f"{_FRED_BASE}/series", {
            "series_id": sid, "api_key": api_key, "file_type": "json",
        })).get("seriess", [{}])[0]
    except Exception:  # noqa: BLE001
        meta = {}

    display = subsample_rows(rows, max_rows=max(1, min(int(limit), 100)))
    return json.dumps({
        "series_id": sid,
        "title": meta.get("title"),
        "units": meta.get("units"),
        "frequency": meta.get("frequency"),
        "observation_end": end,
        "rows": display,
        "num_points": len(rows),
        "source": "fred_api",
    }, default=str)


# ---------------------------------------------------------------- DBnomics

def _parse_dbnomics_path(url_or_path: str) -> tuple[str, str, str] | None:
    s = str(url_or_path).strip()
    if "db.nomics.world/" in s:
        s = s.split("db.nomics.world/", 1)[1]
    s = s.rstrip("/")
    if "/" in s:
        parts = s.split("/", 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    if "_" in s:
        parts = s.split("_", 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
    return None


async def fetch_dbnomics_rows(
    url: str, cutoff_date: str, *, end_date: str | None = None,
) -> tuple[list[dict], str | None]:
    parsed = _parse_dbnomics_path(url)
    if not parsed:
        return [], "invalid dbnomics URL or path (need provider/dataset/series)"
    provider, dataset, series = parsed
    end = clamp_end_date(end_date, cutoff_date)
    start = default_start(end, days=730)

    api_url = f"{_DBNOMICS_API}/{provider}/{dataset}/{series}"
    try:
        data = await _http_get_json(api_url, {"observations": "1", "limit": 5000})
    except Exception as e:  # noqa: BLE001
        return [], f"dbnomics API failed: {e}"

    series_doc = data.get("series", {}) if isinstance(data, dict) else {}
    obs = series_doc.get("observations", {}) if isinstance(series_doc, dict) else {}
    periods = obs.get("period", []) if isinstance(obs, dict) else []
    values = obs.get("value", []) if isinstance(obs, dict) else []

    rows = []
    for period, val in zip(periods, values):
        if val in (None, ".", ""):
            continue
        d = str(period)[:10]
        if d > end or d < start:
            continue
        try:
            rows.append({"date": d, "value": float(val)})
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda x: x["date"])
    return rows, None


async def fetch_dbnomics(url: str, cutoff_date: str, *, end_date: str | None = None) -> str:
    rows, err = await fetch_dbnomics_rows(url, cutoff_date, end_date=end_date)
    if err:
        return json.dumps({"error": err, "url": url})
    end = clamp_end_date(end_date, cutoff_date)
    return json.dumps({
        "url": url,
        "end_date": end,
        "rows": subsample_rows(rows),
        "num_points": len(rows),
        "source": "dbnomics_api",
    }, default=str)


# --------------------------------------------------------------- Wikipedia

def _title_from_url(url: str) -> str:
    m = re.search(r"/wiki/([^#?]+)", url)
    if m:
        return urllib.parse.unquote(m.group(1)).replace("_", " ")
    return url.strip()


async def _wiki_revision(url: str, end_date: str) -> tuple[str, int | None, str, str]:
    title = _title_from_url(url)
    rev_resp = await _http_get_json(_WIKI_API, {
        "action": "query", "titles": title, "prop": "revisions",
        "rvprop": "ids|timestamp", "rvstart": f"{end_date}T23:59:59Z",
        "rvlimit": 1, "rvdir": "older", "format": "json",
    })
    pages = rev_resp.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))
    revisions = page.get("revisions", [])
    if not revisions:
        return title, None, "", ""
    revid = revisions[0]["revid"]
    rev_ts = revisions[0]["timestamp"]
    parse_resp = await _http_get_json(_WIKI_API, {
        "action": "parse", "oldid": revid, "prop": "wikitext", "format": "json",
    }, timeout=60.0)
    wikitext = parse_resp.get("parse", {}).get("wikitext", {}).get("*", "")
    return title, revid, rev_ts, wikitext


def _strip_markup(wikitext: str) -> str:
    text = re.sub(r"\{\{[^{}]*\}\}", "", wikitext)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _extract_toc(wikitext: str) -> str:
    headings = re.findall(r"^(={2,})\s*(.+?)\s*\1\s*$", wikitext, re.MULTILINE)
    lines = []
    for eq, heading in headings:
        indent = " " * (len(eq) - 2)
        lines.append(f"{indent}{heading}")
    return "\n".join(lines)


def _wiki_header(title: str, revid: int, rev_ts: str, end_date: str) -> str:
    return (
        f"Wikipedia: '{title}'\n"
        f"Revision: {revid} ({rev_ts[:10]})\n"
        f"(Content as of {rev_ts[:10]}, cutoff {end_date})\n\n"
    )


async def fetch_wikipedia_toc(url: str, cutoff_date: str, *, end_date: str | None = None) -> str:
    end = clamp_end_date(end_date, cutoff_date)
    try:
        title, revid, rev_ts, wikitext = await _wiki_revision(url, end)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"wikipedia fetch failed: {e}", "url": url})
    if revid is None:
        return json.dumps({"error": f"no revision at or before {end}", "title": title})

    hdr = _wiki_header(title, revid, rev_ts, end)
    toc = _extract_toc(wikitext)
    if not toc:
        body = _strip_markup(wikitext)[:_WIKI_MAX_CHARS]
        return json.dumps({"title": title, "revision": revid, "content": hdr + body}, default=str)
    return json.dumps({
        "title": title,
        "revision": revid,
        "revision_date": rev_ts[:10],
        "toc": toc,
        "hint": "Call fetch_wikipedia_section with a section name from toc.",
        "header": hdr,
    }, default=str)


async def fetch_wikipedia_section(
    url: str, section: str, cutoff_date: str, *, end_date: str | None = None,
) -> str:
    end = clamp_end_date(end_date, cutoff_date)
    try:
        title, revid, rev_ts, wikitext = await _wiki_revision(url, end)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"wikipedia fetch failed: {e}", "url": url})
    if revid is None:
        return json.dumps({"error": f"no revision at or before {end}", "title": title})

    hdr = _wiki_header(title, revid, rev_ts, end)
    sec = str(section).strip()
    if sec.lower() == "introduction":
        first = re.search(r"^==[^=]", wikitext, re.MULTILINE)
        chunk = wikitext[: first.start()] if first else wikitext
    else:
        pattern = re.compile(r"(={2,})\s*" + re.escape(sec) + r"\s*\1", re.IGNORECASE)
        m = pattern.search(wikitext)
        if not m:
            toc = _extract_toc(wikitext)
            return json.dumps({
                "error": f"section '{section}' not found",
                "available_sections": toc,
            })
        level = len(m.group(1))
        nxt = re.compile(r"={" + str(level) + r",}[^=]", re.MULTILINE)
        m2 = nxt.search(wikitext, m.end())
        chunk = wikitext[m.start(): m2.start() if m2 else len(wikitext)]

    body = _strip_markup(chunk)[:_WIKI_MAX_CHARS]
    return json.dumps({
        "title": title,
        "section": section,
        "revision": revid,
        "content": hdr + body,
    }, default=str)


# ----------------------------------------------------------- trend analysis

def _norm_sf(x: float, loc: float, scale: float) -> float:
    if scale <= 0:
        scale = max(abs(loc) * 0.05, 1e-6)
    z = (x - loc) / scale
    return 0.5 * math.erfc(z / math.sqrt(2))


async def _fetch_rows_for_trend(
    source: str, args: dict, cutoff_date: str, end: str,
) -> tuple[list[dict], str | None]:
    src = source.lower()
    if src == "yfinance":
        ticker = str(args.get("ticker", "")).strip()
        if not ticker:
            return [], "ticker required for yfinance"
        return await fetch_yfinance_rows(ticker, cutoff_date, end_date=end)
    if src == "fred":
        sid = str(args.get("series_id", args.get("series", ""))).strip()
        if not sid:
            return [], "series_id required for fred"
        return await fetch_fred_rows(sid, cutoff_date, end_date=end)
    if src == "dbnomics":
        url = str(args.get("url", "")).strip()
        if not url:
            return [], "url required for dbnomics"
        return await fetch_dbnomics_rows(url, cutoff_date, end_date=end)
    return [], f"unknown source: {source}"


async def analyze_trend(
    source: str,
    comparison_value: float,
    resolution_date: str,
    cutoff_date: str,
    *,
    end_date: str | None = None,
    **fetch_args,
) -> str:
    """Estimate P(value > comparison_value) at resolution_date."""
    end = clamp_end_date(end_date, cutoff_date)
    res_date = str(resolution_date)[:10]
    try:
        comp = float(comparison_value)
    except (TypeError, ValueError):
        return json.dumps({"error": "comparison_value must be a number"})

    rows, err = await _fetch_rows_for_trend(source, fetch_args, cutoff_date, end)
    if err:
        return json.dumps({"error": err})
    if len(rows) < 3:
        return json.dumps({"error": "insufficient data for trend analysis (need >= 3 points)"})

    def _analyze_sync() -> str:
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(rows).rename(columns={"date": "ds", "value": "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["ds", "y"]).sort_values("ds")
        if len(df) < 3:
            return json.dumps({"error": "insufficient data for trend analysis (need >= 3 points)"})

        src = source.lower()
        window = {"yfinance": 60, "fred": 30, "dbnomics": min(len(df), 365)}.get(src, 30)
        recent = df.iloc[-min(len(df), window):].copy()

        t0 = recent["ds"].min()
        x = (recent["ds"] - t0).dt.days.values.astype(float)
        y = recent["y"].values.astype(float)
        coeffs = np.polyfit(x, y, 1)
        residuals = y - np.polyval(coeffs, x)
        std = float(residuals.std()) if len(residuals) > 1 else max(abs(float(y.mean())) * 0.05, 1e-6)
        if std <= 0:
            std = max(abs(float(y.mean())) * 0.05, 1e-6)

        x_rd = float((pd.to_datetime(res_date) - t0).days)
        predicted = float(np.polyval(coeffs, x_rd))
        prob_raw = _norm_sf(comp, predicted, std)

        shrink = {"yfinance": 0.1, "fred": 0.5, "dbnomics": 0.4}.get(src, 0.7)
        prob_linear = max(0.05, min(0.95, shrink * prob_raw + (1 - shrink) * 0.5))

        # Same period in prior years
        rd = pd.to_datetime(res_date)
        same_vals = []
        for off in range(1, 6):
            try:
                target = rd.replace(year=rd.year - off)
            except ValueError:
                continue
            mask = (df["ds"] >= target - pd.Timedelta(days=7)) & (df["ds"] <= target + pd.Timedelta(days=7))
            window_data = df[mask]
            if window_data.empty:
                continue
            idx = (window_data["ds"] - target).abs().idxmin()
            same_vals.append(float(window_data.loc[idx, "y"]))

        prob_seasonal = None
        if same_vals:
            n_ex = sum(1 for v in same_vals if v > comp)
            prob_seasonal = max(0.05, min(0.95, (n_ex + 1) / (len(same_vals) + 2)))

        if prob_seasonal is not None and src == "dbnomics":
            prob_combined = 0.7 * prob_seasonal + 0.3 * prob_linear
        elif prob_seasonal is not None:
            prob_combined = 0.5 * prob_seasonal + 0.5 * prob_linear
        else:
            prob_combined = prob_linear
        prob_combined = max(0.05, min(0.95, prob_combined))

        last_row = recent.iloc[-1]
        return json.dumps({
            "source": src,
            "comparison_value": comp,
            "resolution_date": res_date,
            "end_date": end,
            "last_value": float(last_row["y"]),
            "last_date": str(last_row["ds"])[:10],
            "predicted_at_resolution": predicted,
            "p_exceed_linear": round(prob_linear, 4),
            "p_exceed_seasonal": round(prob_seasonal, 4) if prob_seasonal is not None else None,
            "p_exceed_combined": round(prob_combined, 4),
            "slope_per_day": float(coeffs[0]),
            "analysis": (
                f"Combined estimate P(value > {comp:.4g}) at {res_date}: {prob_combined:.2f}. "
                f"Last value {float(last_row['y']):.4g} on {str(last_row['ds'])[:10]}."
            ),
        }, default=str)

    return await asyncio.to_thread(_analyze_sync)
