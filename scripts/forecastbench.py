#!/usr/bin/env python3
"""Run a prime-forecast policy on ForecastBench tournament rounds (https://www.forecastbench.org).

The policy is whatever an OpenAI-compatible endpoint serves (e.g. vLLM with Qwen3.5-35B-A3B plus the
DYOR LoRA adapter). Every forecast is one episode of this repo's environment: same tools, search
budget, domain blocklist and leak filter as the paper.

    set -a && source secrets.env && set +a
    python scripts/forecastbench.py fetch  --due 2026-08-16                       # question + resolution sets
    python scripts/forecastbench.py build  --due 2026-08-16                       # inspect episodes
    python scripts/forecastbench.py run    --due 2026-08-16 --mode historical --sample 300 \
        --base-url http://localhost:8000/v1 --model dyor-v4                       # rehearsal on a past round
    python scripts/forecastbench.py run    --due <today> --mode live --model dyor-v4   # a real round (24 h window)
    python scripts/forecastbench.py submit --due <due> --org "Future Principle" --model-name "DYOR-35B-A3B" --variants anchored
    python scripts/forecastbench.py score  --due 2026-08-16

Episodes:
  market question    one episode (final outcome). The crowd value (freeze_datetime_value) is NOT shown;
                     the policy never sees a market price. Only the `anchored` submission variant blends
                     it in afterwards, outside the policy.
  dataset question   one episode per resolution date, with {forecast_due_date} / {resolution_date} filled.
                     The reference value is part of the question, so it IS shown.

Modes:
  historical  cutoff = the round's due date (00:00 UTC). Search is date-filtered and the LLM leak filter
              runs when Bedrock credentials are present. Use for rehearsals on past, resolved rounds.
  live        cutoff = now; the LLM leak filter and the date heuristics are off (nothing after now exists,
              and a future date is a scheduled event). The prediction-market domain/content filter stays on.
              Use for real submissions.

Submission variants (from the same rollouts, no extra compute):
  pure        the policy's own probability everywhere
  anchored    market questions: logit blend w*logit(crowd) + (1-w)*logit(policy); dataset questions unchanged

Everything is written under outputs/forecastbench/ (gitignored); `run` appends and resumes. Each run keeps
rollouts.jsonl (one line per rollout: probability, tokens) and traces.jsonl.gz (the full transcripts).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import math
import os
import random
import re
import tempfile
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "forecastbench"
RAW = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets"
MARKET_SOURCES = {"kalshi", "manifold", "metaculus", "polymarket", "infer"}


# ------------------------------------------------------------------ question sets

def set_path(due: str) -> Path:
    return OUT / "question_sets" / f"{due}-llm.json"


def resolutions_path(due: str) -> Path:
    return OUT / "resolution_sets" / f"{due}_resolution_set.json"


def load_set(due: str) -> dict:
    path = set_path(due)
    if not path.exists():
        raise SystemExit(f"{path} missing; run `fetch --due {due}` first")
    return json.loads(path.read_text())


def resolution_dates(q: dict) -> list[str]:
    rd = q.get("resolution_dates")
    if isinstance(rd, list):
        return rd
    if isinstance(rd, str) and rd.startswith("["):
        return json.loads(rd.replace("'", '"'))
    return []


def is_market(q: dict) -> bool:
    return q["source"] in MARKET_SOURCES or not resolution_dates(q)


def key(source: str, qid: str, rdate: str | None) -> str:
    return f"fb:{source}:{qid}:{rdate or 'final'}"


def _na(v) -> bool:
    return v in (None, "", "N/A", "nan")


def _float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def episodes(qset: dict) -> list[dict]:
    """One episode per required forecast."""
    due = qset["forecast_due_date"]
    out = []
    for q in qset["questions"]:
        rules = [q.get("market_info_resolution_criteria") if not _na(q.get("market_info_resolution_criteria"))
                 else q.get("resolution_criteria") or ""]
        if not _na(q.get("background")):
            rules.append(f"Background: {q['background']}")
        if not _na(q.get("source_intro")):
            rules.append(f"Context: {q['source_intro']}")
        if is_market(q):
            deadline = q.get("market_info_close_datetime")
            out.append({"key": key(q["source"], q["id"], None), "fb_id": q["id"], "source": q["source"],
                        "resolution_date": None, "question": q["question"], "rules": "\n\n".join(rules),
                        "deadline": deadline if not _na(deadline) else q.get("freeze_datetime"),
                        "crowd": _float(q.get("freeze_datetime_value"))})
            continue
        ref = q.get("freeze_datetime_value")
        if not _na(ref):
            rules.append(f"Reference value at {q.get('freeze_datetime')}: {ref}. "
                         f"{q.get('freeze_datetime_value_explanation') or ''}")
        for rd in resolution_dates(q):
            text = q["question"].replace("{forecast_due_date}", due).replace("{resolution_date}", rd)
            out.append({"key": key(q["source"], q["id"], rd), "fb_id": q["id"], "source": q["source"],
                        "resolution_date": rd, "question": text, "rules": "\n\n".join(rules),
                        "deadline": f"{rd}T23:59:59Z", "crowd": None})
    return out


# --notes: per-source guidance appended to dataset questions' resolution text. Each targets a failure seen in
# rehearsal traces (2026-08-16 and held-out rounds); adopt only if it beats post-processing on held-out episodes.
NOTES = {
    "acled": "Forecasting note: this question asks about a CHANGE versus the stated baseline average. Evidence "
             "that the activity is ongoing is already reflected in that baseline and is not evidence of an "
             "increase. Historically about 12% of these questions resolve YES.",
    "yfinance": "Forecasting note: over this horizon, a stock's move is close to a random walk. Stay near 50% "
                "unless there is a specific, scheduled catalyst before the resolution date.",
}


def env_row(e: dict, cutoff: str, notes: bool = False) -> dict:
    """A row in this repo's dataset schema. Outcome is a placeholder; scoring uses the resolution set."""
    rules = e["rules"]
    if notes and e["resolution_date"] is not None and e["source"] in NOTES:
        rules = f"{rules}\n\n{NOTES[e['source']]}"
    return {"example_id": e["key"], "market_id": e["key"], "question": e["question"],
            "resolution_criteria": rules, "cutoff_date": cutoff, "resolution_date": e["deadline"],
            "category": e["source"], "outcome": 0, "price_at_cutoff": None, "price_history": [],
            "token_id": None, "slug": e["fb_id"], "volume": None}


def stratified(eps: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_src = defaultdict(list)
    for e in eps:
        by_src[e["source"]].append(e)
    per = max(1, n // max(1, len(by_src)))
    picked = []
    for _, items in sorted(by_src.items()):
        rng.shuffle(items)
        picked += items[:per]
    return picked


def run_dir(due: str, name: str) -> Path:
    d = OUT / "runs" / due / name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ rollouts

def _plain(x):
    """verifiers messages mix dicts and pydantic models; make them plain JSON (reasoning_content included)."""
    if hasattr(x, "model_dump"):
        x = x.model_dump(exclude_none=True)
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def rollouts(rows: list[dict], args, mode: str) -> list[dict]:
    """Evaluate rows in this repo's environment against an OpenAI-compatible endpoint."""
    import verifiers as vf

    from prime_forecast import load_environment

    if mode == "live":  # cutoff = now: nothing after it exists, and future dates are scheduled events
        os.environ["PF_LLM_FILTER"] = "0"
        os.environ["PF_DATE_FILTER"] = "0"
    else:  # a past cutoff always keeps the date filter
        os.environ["PF_DATE_FILTER"] = "1"
    os.environ.setdefault(args.api_key_var, "EMPTY")
    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "fb.jsonl", "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in rows) + "\n")
        env = load_environment(dataset_path=tmp, split="fb", eval_split="fb", max_turns=args.max_turns,
                               max_web_searches=args.max_web_searches, include_market_tools=False,
                               crowd_copy_penalty=0.0)
        client = vf.ClientConfig(api_base_url=args.base_url, api_key_var=args.api_key_var)
        results = asyncio.run(env.evaluate(
            client=client, model=args.model,
            sampling_args={"temperature": args.temperature, "max_tokens": args.max_tokens},
            rollouts_per_example=args.rollouts, max_concurrent=args.max_concurrent, max_retries=1))
    out, seen = [], Counter()
    for o in results["outputs"]:
        info = o.get("info") or {}
        info = json.loads(info) if isinstance(info, str) else info
        k = info.get("market_id")
        p = (o.get("metrics") or {}).get("predicted_prob")
        usage = o.get("token_usage") or {}
        out.append({"key": k, "rollout_idx": seen[k], "p": float(p) if p is not None and p >= 0 else None,
                    "tokens_in": usage.get("input_tokens") or usage.get("prompt_tokens"),
                    "tokens_out": usage.get("output_tokens") or usage.get("completion_tokens"),
                    "trace": {"completion": _plain(o.get("completion")), "metrics": o.get("metrics"),
                              "stop_condition": o.get("stop_condition"), "error": o.get("error")}})
        seen[k] += 1
    return out


# ------------------------------------------------------------------ commands

def cmd_fetch(args) -> None:
    for url, path in ((f"{RAW}/question_sets/{args.due}-llm.json", set_path(args.due)),
                      (f"{RAW}/resolution_sets/{args.due}_resolution_set.json", resolutions_path(args.due))):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, path)
            print(f"saved {path.relative_to(ROOT)}")
        except Exception as e:  # resolution sets appear only once questions start resolving
            print(f"not available yet: {url} ({e})")


def cmd_build(args) -> None:
    qset = load_set(args.due)
    eps = episodes(qset)
    kinds = Counter("market" if e["resolution_date"] is None else "dataset" for e in eps)
    print(json.dumps({"question_set": qset["question_set"], "questions": len(qset["questions"]),
                      "episodes": len(eps), "by_type": dict(kinds),
                      "by_source": dict(Counter(e["source"] for e in eps))}, indent=2))


def cmd_run(args) -> None:
    qset = load_set(args.due)
    eps = episodes(qset)
    if args.resolved_only:  # rehearsals: spend compute only on episodes the resolution set can score
        eps = [e for e in eps if e["key"] in resolved_truth(args.due)]
    if args.keys:  # an explicit episode list, one key per line (e.g. a hand-picked trace study)
        want = {l.strip() for l in open(args.keys) if l.strip()}
        eps = [e for e in eps if e["key"] in want]
    if args.sample:
        eps = stratified(eps, args.sample, args.seed)
    path = run_dir(args.due, args.name) / "rollouts.jsonl"
    done = {json.loads(l)["key"] for l in open(path)} if path.exists() else set()
    todo = [e for e in eps if e["key"] not in done]
    cutoff = (f"{qset['forecast_due_date']}T00:00:00Z" if args.mode == "historical"
              else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    print(f"{len(todo)} episodes to run ({len(done)} done), mode={args.mode}, cutoff={cutoff}, model={args.model}")
    for attempt in range(args.retries + 1):
        if attempt:  # coverage pass: rerun episodes with no submitted forecast in any rollout so far
            have = set(predictions(args.due, args.name))
            todo = [e for e in eps if e["key"] not in have]
            if not todo:
                break
            print(f"retry {attempt}: {len(todo)} episodes without a forecast")
        for i in range(0, len(todo), args.batch):
            batch = todo[i: i + args.batch]
            if args.stub:  # pipeline test without a model; misses ~5% so the retry pass is exercised
                rng = random.Random(f"{attempt}:{i}")
                res = [{"key": e["key"], "rollout_idx": 0, "p": None if rng.random() < 0.05 else rng.random(),
                        "tokens_in": 0, "tokens_out": 0} for e in batch]
            else:
                res = rollouts([env_row(e, cutoff, args.notes) for e in batch], args, args.mode)
            with open(path, "a") as fh, gzip.open(path.with_name("traces.jsonl.gz"), "at") as tf:
                for r in res:  # full transcripts kept apart: rollouts.jsonl stays small and fast to rescore
                    trace = r.pop("trace", None)
                    fh.write(json.dumps({**r, "cutoff": cutoff, "model": args.model}) + "\n")
                    if trace is not None:
                        tf.write(json.dumps({"key": r["key"], "rollout_idx": r["rollout_idx"], **trace}, default=str) + "\n")
            print(f"  batch {i // args.batch + 1}: {sum(r['p'] is not None for r in res)}/{len(res)} submitted")
    have = set(predictions(args.due, args.name))
    print(f"coverage: {sum(e['key'] in have for e in eps)}/{len(eps)} episodes have a forecast")


def predictions(due: str, name: str) -> dict[str, float]:
    acc = defaultdict(list)
    for line in open(run_dir(due, name) / "rollouts.jsonl"):
        r = json.loads(line)
        if r["p"] is not None:
            acc[r["key"]].append(r["p"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as f:
        return json.loads(f.read())


def _live_price(e: dict) -> float | None:
    """Current YES price for a market question from its platform's public API (None if unavailable).
    Metaculus needs an API token, so its questions keep ForecastBench's freeze value."""
    src, qid = e["source"], e["fb_id"]
    if src == "polymarket":
        for closed in ("false", "true"):
            ms = _get_json("https://gamma-api.polymarket.com/markets?" + urllib.parse.urlencode({"condition_ids": qid, "closed": closed}))
            if ms:
                outs, prices = json.loads(ms[0]["outcomes"]), json.loads(ms[0]["outcomePrices"])
                return float(prices[[o.lower() for o in outs].index("yes")]) if "yes" in [o.lower() for o in outs] else float(prices[0])
    elif src == "manifold":
        m = _get_json(f"https://api.manifold.markets/v0/market/{qid}")
        return float(m["probability"]) if m.get("probability") is not None else None
    elif src == "kalshi":
        m = _get_json(f"https://api.elections.kalshi.com/trade-api/v2/markets/{qid}")["market"]
        bid, ask = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
        if 0 < bid <= ask < 1 and ask - bid <= 0.10:
            return (bid + ask) / 2
        return float(m["last_price_dollars"]) if m.get("last_price_dollars") else None
    return None


def live_crowd(eps: list[dict], path: Path) -> dict[str, float]:
    """Fetch (once; cached at path) current prices for every market episode; the submission's anchor uses them."""
    if path.exists():
        return json.loads(path.read_text())["prices"]
    from concurrent.futures import ThreadPoolExecutor

    def one(e):
        try:
            return e["key"], _live_price(e)
        except Exception:  # noqa: BLE001  any API failure falls back to the freeze value
            return e["key"], None
    with ThreadPoolExecutor(16) as ex:
        got = dict(ex.map(one, [e for e in eps if e["resolution_date"] is None]))
    prices = {k: v for k, v in got.items() if v is not None}
    path.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "prices": prices}, indent=1))
    by = Counter(k.split(":")[1] for k in prices); tot = Counter(k.split(":")[1] for k in got)
    print("live crowd prices:", {s: f"{by[s]}/{tot[s]}" for s in sorted(tot)}, "(others keep the freeze value)")
    return prices


def _temp_climatology(obs: dict, due, rd) -> float | None:
    """P(temp on rd > temp on due) from the station's own history: past 30 years, same calendar days +/-3, pairwise."""
    from datetime import date, timedelta
    wins = n = 0
    for y in range(due.year - 30, due.year):
        for dd in range(-3, 4):
            try:
                a = obs.get(date(y, due.month, due.day) + timedelta(days=dd))
                b = obs.get(date(y + (rd.year - due.year), rd.month, rd.day) + timedelta(days=dd))
            except ValueError:
                continue
            if a is not None and b is not None:
                wins += b > a
                n += 1
    return (wins + 0.5) / (n + 1) if n >= 20 else None


def climatology(eps: list[dict], due: str, path: Path) -> dict[str, float]:
    """DBnomics (Meteo-France station temperature) episodes: climatological probability, data before due only.
    Cached at path. Rehearsals: climatology alone scored 0.144-0.285 Brier over 4 rounds and beat DYOR on both
    rounds with DYOR forecasts (0.186 vs 0.205, 0.294 vs 0.347)."""
    if path.exists():
        return json.loads(path.read_text())
    from datetime import date
    due_d = date.fromisoformat(due)
    out, series = {}, {}
    for e in eps:
        if e["source"] != "dbnomics" or e["resolution_date"] is None:
            continue
        sid = e["fb_id"]
        if sid not in series:
            try:
                prov, ds, code = sid.split("_", 2)
                doc = _get_json(f"https://api.db.nomics.world/v22/series/{prov}/{ds}/{code}?observations=1&format=json")
                doc = doc["series"]["docs"][0]
                series[sid] = {date.fromisoformat(t[:10]): float(v) for t, v in zip(doc["period"], doc["value"])
                               if v not in (None, "NA") and date.fromisoformat(t[:10]) < due_d}
            except Exception:  # noqa: BLE001  no climatology for this station: the model's forecast stands
                series[sid] = {}
        c = _temp_climatology(series[sid], due_d, date.fromisoformat(e["resolution_date"])) if series[sid] else None
        if c is not None:
            out[e["key"]] = c
    path.write_text(json.dumps(out, indent=1))
    print(f"climatology: {len(out)} DBnomics episodes from {sum(bool(v) for v in series.values())}/{len(series)} stations")
    return out


def pooled(due: str, name: str, pool: list[str] | None, first_weight: dict[str, float] | None = None) -> dict[str, float]:
    """Average each episode's forecast across runs (each run's own mean first); a run missing an episode is skipped.
    first_weight: per-source weight on the --name run (the rest is split evenly over the pool), e.g. wikipedia=0.25."""
    runs = [predictions(due, n) for n in [name, *(pool or [])]]
    out = {}
    for k in set().union(*runs):
        have = [i for i, r in enumerate(runs) if k in r]
        w1 = (first_weight or {}).get(k.split(":")[1])
        if w1 is None or len(have) < 2 or 0 not in have:
            out[k] = sum(runs[i][k] for i in have) / len(have)
        else:
            rest = [i for i in have if i != 0]
            out[k] = w1 * runs[0][k] + (1 - w1) * sum(runs[i][k] for i in rest) / len(rest)
    return out


def parse_priors(items: list[str] | None) -> dict[str, tuple[float, float | None]]:
    """SOURCE=RATE or SOURCE=RATE:WEIGHT (model weight; default --prior-weight)."""
    out = {}
    for i in items or []:
        src, v = i.split("=")
        rate, _, w = v.partition(":")
        out[src] = (float(rate), float(w) if w else None)
    return out


def with_prior(e: dict, p: float | None, priors: dict, w: float) -> float | None:
    """Dataset questions from a listed source: blend toward that source's rate (a base rate, or 0.5 for noise)."""
    if p is None or e["resolution_date"] is None or e["source"] not in priors:
        return p
    rate, wsrc = priors[e["source"]]
    w = w if wsrc is None else wsrc
    return w * p + (1 - w) * rate


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def variant(e: dict, p: float | None, name: str, w: float) -> float | None:
    if name == "anchored" and e["resolution_date"] is None and e["crowd"] is not None:
        return e["crowd"] if p is None else 1 / (1 + math.exp(-(w * _logit(e["crowd"]) + (1 - w) * _logit(p))))
    return p


def cmd_submit(args) -> None:
    qset = load_set(args.due)
    eps = episodes(qset)
    preds = pooled(args.due, args.name, args.pool, {s: float(w) for s, w in (x.split("=") for x in args.pool_weight or [])})
    priors = parse_priors(args.source_prior)
    if args.climatology is not None:  # DBnomics: blend the model with station climatology (model weight W)
        clim = climatology(eps, qset["forecast_due_date"], run_dir(args.due, args.name) / "climatology.json")
        preds = {k: (args.climatology * p + (1 - args.climatology) * clim[k]) if k in clim else p for k, p in preds.items()}
    if args.live_crowd:  # fresher anchor than the freeze value (10 days old in most rounds)
        live = live_crowd(eps, run_dir(args.due, args.name) / "live_crowd.json")
        eps = [{**e, "crowd": live.get(e["key"], e["crowd"])} for e in eps]
    totals = Counter("market" if e["resolution_date"] is None else "dataset" for e in eps)
    for n, var in enumerate(args.variants, start=args.number):
        forecasts, missing = [], Counter()
        for e in eps:
            p = variant(e, with_prior(e, preds.get(e["key"]), priors, args.prior_weight), var, args.anchor_weight)
            if p is None:
                missing["market" if e["resolution_date"] is None else "dataset"] += 1
                continue
            forecasts.append({"id": e["fb_id"], "source": e["source"], "forecast": round(min(max(p, 0.0), 1.0), 4),
                              "resolution_date": e["resolution_date"], "reasoning": None})
        coverage = {k: round(1 - missing[k] / totals[k], 3) for k in totals}
        model = f"{args.model_name} ({var})" if len(args.variants) > 1 else args.model_name
        doc = {"organization": args.org, "model": model, "model_organization": args.model_org or args.org,
               "question_set": qset["question_set"], "forecasts": forecasts}
        org_slug = re.sub(r"[^A-Za-z0-9]+", "", args.org)  # file name: <due>.<org>.<N>.json, no spaces
        path = run_dir(args.due, args.name) / f"{qset['forecast_due_date']}.{org_slug}.{n}.json"
        path.write_text(json.dumps(doc, indent=1))
        ok = all(v >= 0.95 for v in coverage.values())
        print(f"{var:9s} -> {path.relative_to(ROOT)}: {len(forecasts)} forecasts, coverage {coverage} "
              f"{'OK' if ok else 'BELOW 95% (missing forecasts are imputed as 0.5)'}")


def resolved_truth(due: str) -> dict[str, float]:
    res = json.loads(resolutions_path(due).read_text())["resolutions"]
    return {key(r["source"], r["id"], None if r["source"] in MARKET_SOURCES else r["resolution_date"]):
            float(r["resolved_to"]) for r in res if r.get("resolved")}


def cmd_score(args) -> None:
    qset = load_set(args.due)
    eps = {e["key"]: e for e in episodes(qset)}
    truth = resolved_truth(args.due)
    preds = pooled(args.due, args.name, args.pool)
    priors = parse_priors(args.source_prior)
    rows = [{"source": eps[k]["source"], "kind": "market" if eps[k]["resolution_date"] is None else "dataset",
             "y": y, "policy": preds[k], "anchored": variant(eps[k], preds[k], "anchored", args.anchor_weight),
             "submitted": variant(eps[k], with_prior(eps[k], preds[k], priors, args.prior_weight), "anchored",
                                  args.anchor_weight),
             "crowd": eps[k]["crowd"]} for k, y in truth.items() if k in preds and k in eps]

    def brier(xs, field):
        xs = [x for x in xs if x[field] is not None]
        return round(sum((x[field] - x["y"]) ** 2 for x in xs) / len(xs), 4) if xs else None

    groups = [("overall", rows)] + [(k, [x for x in rows if x["kind"] == k]) for k in ("market", "dataset")]
    groups += [(s, [x for x in rows if x["source"] == s]) for s in sorted({x["source"] for x in rows})]
    print(json.dumps({g: {"n": len(xs), "policy": brier(xs, "policy"), "anchored": brier(xs, "anchored"),
                          "anchored+prior": brier(xs, "submitted"), "crowd": brier(xs, "crowd"), "always_0.5": 0.25}
                      for g, xs in groups}, indent=1))
    print("Brier, lower is better. The leaderboard reports a difficulty-adjusted Brier Index instead.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "build", "run", "submit", "score"):
        p = sub.add_parser(name)
        p.add_argument("--due", required=True, help="round's forecast due date, e.g. 2026-08-16")
        p.add_argument("--name", default="dyor-v4", help="run name under outputs/forecastbench/runs/<due>/")
        p.add_argument("--anchor-weight", type=float, default=0.8, help="crowd weight in the anchored variant")
        if name in ("submit", "score"):
            p.add_argument("--pool", nargs="*", default=None, help="more run names to average with --name, per episode")
            p.add_argument("--pool-weight", nargs="*", default=None,
                           help="SOURCE=W: weight on the --name run for that source when pooling (default equal)")
            p.add_argument("--source-prior", nargs="*", default=None,
                           help="SOURCE=RATE[:WEIGHT]: blend that source's dataset forecasts toward RATE with model weight WEIGHT")
            p.add_argument("--prior-weight", type=float, default=0.5, help="model weight when a source prior applies")
        if name == "run":
            p.add_argument("--mode", choices=["historical", "live"], required=True)
            p.add_argument("--base-url", default="http://localhost:8000/v1")
            p.add_argument("--api-key-var", default="VLLM_API_KEY")
            p.add_argument("--model", default="dyor-v4", help="served model name")
            p.add_argument("--rollouts", type=int, default=1)
            p.add_argument("--sample", type=int, default=0, help="stratified subsample size (0 = all)")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--batch", type=int, default=64)
            p.add_argument("--max-turns", type=int, default=10)
            p.add_argument("--max-web-searches", type=int, default=3)
            p.add_argument("--max-concurrent", type=int, default=16)
            p.add_argument("--temperature", type=float, default=1.0)
            p.add_argument("--max-tokens", type=int, default=12288)
            p.add_argument("--notes", action="store_true", help="append per-source forecasting notes (NOTES)")
            p.add_argument("--keys", default=None, help="file of episode keys to run (one per line)")
            p.add_argument("--resolved-only", action="store_true", help="only episodes already in the resolution set")
            p.add_argument("--retries", type=int, default=2, help="coverage passes over episodes with no forecast")
            p.add_argument("--stub", action="store_true", help="random forecasts; tests the pipeline without a model")
        if name == "submit":
            p.add_argument("--org", required=True)
            p.add_argument("--model-name", required=True, help='permanent leaderboard name, e.g. "DYOR-35B-A3B"')
            p.add_argument("--model-org", default=None)
            p.add_argument("--variants", nargs="+", default=["pure", "anchored"], choices=["pure", "anchored"])
            p.add_argument("--climatology", type=float, default=None, metavar="W",
                           help="DBnomics temperature questions: W*model + (1-W)*station climatology")
            p.add_argument("--live-crowd", action="store_true",
                           help="anchor market questions on current platform prices (cached in the run dir)")
            p.add_argument("--number", type=int, default=1, help="N in <due>.<org>.<N>.json for the first variant")
    args = ap.parse_args()
    {"fetch": cmd_fetch, "build": cmd_build, "run": cmd_run, "submit": cmd_submit, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
