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
    python scripts/forecastbench.py submit --due <due> --org <org> --model-name "DYOR-35B-A3B (v4)"
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
  live        cutoff = now; the leak filter is off (nothing after now exists). Use for real submissions.

Submission variants (from the same rollouts, no extra compute):
  pure        the policy's own probability everywhere
  anchored    market questions: logit blend w*logit(crowd) + (1-w)*logit(policy); dataset questions unchanged

Everything is written under outputs/forecastbench/ (gitignored); `run` appends and resumes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import tempfile
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


def env_row(e: dict, cutoff: str) -> dict:
    """A row in this repo's dataset schema. Outcome is a placeholder; scoring uses the resolution set."""
    return {"example_id": e["key"], "market_id": e["key"], "question": e["question"],
            "resolution_criteria": e["rules"], "cutoff_date": cutoff, "resolution_date": e["deadline"],
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

def rollouts(rows: list[dict], args, mode: str) -> list[dict]:
    """Evaluate rows in this repo's environment against an OpenAI-compatible endpoint."""
    import verifiers as vf

    from prime_forecast import load_environment

    if mode == "live":
        os.environ["PF_LLM_FILTER"] = "0"
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
                    "tokens_out": usage.get("output_tokens") or usage.get("completion_tokens")})
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
    if args.sample:
        eps = stratified(eps, args.sample, args.seed)
    path = run_dir(args.due, args.name) / "rollouts.jsonl"
    done = {json.loads(l)["key"] for l in open(path)} if path.exists() else set()
    todo = [e for e in eps if e["key"] not in done]
    cutoff = (f"{qset['forecast_due_date']}T00:00:00Z" if args.mode == "historical"
              else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    print(f"{len(todo)} episodes to run ({len(done)} done), mode={args.mode}, cutoff={cutoff}, model={args.model}")
    for i in range(0, len(todo), args.batch):
        batch = todo[i: i + args.batch]
        if args.stub:  # pipeline test without a model
            rng = random.Random(i)
            res = [{"key": e["key"], "rollout_idx": 0, "p": rng.random(), "tokens_in": 0, "tokens_out": 0}
                   for e in batch]
        else:
            res = rollouts([env_row(e, cutoff) for e in batch], args, args.mode)
        with open(path, "a") as fh:
            for r in res:
                fh.write(json.dumps({**r, "cutoff": cutoff, "model": args.model}) + "\n")
        print(f"  batch {i // args.batch + 1}: {sum(r['p'] is not None for r in res)}/{len(res)} submitted")


def predictions(due: str, name: str) -> dict[str, float]:
    acc = defaultdict(list)
    for line in open(run_dir(due, name) / "rollouts.jsonl"):
        r = json.loads(line)
        if r["p"] is not None:
            acc[r["key"]].append(r["p"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


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
    preds = predictions(args.due, args.name)
    totals = Counter("market" if e["resolution_date"] is None else "dataset" for e in eps)
    for n, var in enumerate(args.variants, start=1):
        forecasts, missing = [], Counter()
        for e in eps:
            p = variant(e, preds.get(e["key"]), var, args.anchor_weight)
            if p is None:
                missing["market" if e["resolution_date"] is None else "dataset"] += 1
                continue
            forecasts.append({"id": e["fb_id"], "source": e["source"], "forecast": round(min(max(p, 0.0), 1.0), 4),
                              "resolution_date": e["resolution_date"], "reasoning": None})
        coverage = {k: round(1 - missing[k] / totals[k], 3) for k in totals}
        model = f"{args.model_name} ({var})" if len(args.variants) > 1 else args.model_name
        doc = {"organization": args.org, "model": model, "model_organization": args.model_org or args.org,
               "question_set": qset["question_set"], "forecasts": forecasts}
        path = run_dir(args.due, args.name) / f"{qset['forecast_due_date']}.{args.org}.{n}.json"
        path.write_text(json.dumps(doc, indent=1))
        ok = all(v >= 0.95 for v in coverage.values())
        print(f"{var:9s} -> {path.relative_to(ROOT)}: {len(forecasts)} forecasts, coverage {coverage} "
              f"{'OK' if ok else 'BELOW 95% (missing forecasts are imputed as 0.5)'}")


def cmd_score(args) -> None:
    qset = load_set(args.due)
    eps = {e["key"]: e for e in episodes(qset)}
    res = json.loads(resolutions_path(args.due).read_text())["resolutions"]
    truth = {key(r["source"], r["id"], None if r["source"] in MARKET_SOURCES else r["resolution_date"]):
             float(r["resolved_to"]) for r in res if r.get("resolved")}
    preds = predictions(args.due, args.name)
    rows = [{"source": eps[k]["source"], "kind": "market" if eps[k]["resolution_date"] is None else "dataset",
             "y": y, "policy": preds[k], "anchored": variant(eps[k], preds[k], "anchored", args.anchor_weight),
             "crowd": eps[k]["crowd"]} for k, y in truth.items() if k in preds and k in eps]

    def brier(xs, field):
        xs = [x for x in xs if x[field] is not None]
        return round(sum((x[field] - x["y"]) ** 2 for x in xs) / len(xs), 4) if xs else None

    groups = [("overall", rows)] + [(k, [x for x in rows if x["kind"] == k]) for k in ("market", "dataset")]
    groups += [(s, [x for x in rows if x["source"] == s]) for s in sorted({x["source"] for x in rows})]
    print(json.dumps({g: {"n": len(xs), "policy": brier(xs, "policy"), "anchored": brier(xs, "anchored"),
                          "crowd": brier(xs, "crowd"), "always_0.5": 0.25} for g, xs in groups}, indent=1))
    print("Brier, lower is better. The leaderboard reports a difficulty-adjusted Brier Index instead.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "build", "run", "submit", "score"):
        p = sub.add_parser(name)
        p.add_argument("--due", required=True, help="round's forecast due date, e.g. 2026-08-16")
        p.add_argument("--name", default="dyor-v4", help="run name under outputs/forecastbench/runs/<due>/")
        p.add_argument("--anchor-weight", type=float, default=0.8, help="crowd weight in the anchored variant")
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
            p.add_argument("--stub", action="store_true", help="random forecasts; tests the pipeline without a model")
        if name == "submit":
            p.add_argument("--org", required=True)
            p.add_argument("--model-name", required=True, help='permanent leaderboard name, e.g. "DYOR-35B-A3B (v4)"')
            p.add_argument("--model-org", default=None)
            p.add_argument("--variants", nargs="+", default=["pure", "anchored"], choices=["pure", "anchored"])
    args = ap.parse_args()
    {"fetch": cmd_fetch, "build": cmd_build, "run": cmd_run, "submit": cmd_submit, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
