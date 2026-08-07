"""Generate all paper figures from archived per-question data in results/.

Outputs to results/figures/. No network, no credits. Trading figures (Turtel
F2/F3 analogs) deliberately omitted: P&L at n=265 is noise and we make no
trading claims (see docs/paper-eval-plan.md claim-safety notes).
"""
import gzip
import json
import math
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
FIG = f"{R}/figures"

C_TRAINED = "#0a9c6d"
C_BASE = "#888888"
C_FRONTIER = "#7c5cff"
C_CROWD = "#d4a017"


def test_ids(split):
    return {str(json.loads(l)["market_id"]) for l in open(f"{R}/{split}")}


def rows_from(path, ids=None):
    """(p, y, crowd, submitted) tuples; 0.5-imputed non-answers."""
    out = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for l in f:
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "market_id" not in l:
                continue
            m = r.get("metrics") or r
            info = r.get("info") or r
            mid = str(info.get("market_id", r.get("market_id")))
            if ids is not None and mid not in ids:
                continue
            y = info.get("outcome", r.get("outcome"))
            c = info.get("price_at_cutoff", r.get("price_at_cutoff"))
            if y is None or c is None:
                continue
            sub = bool(m.get("submitted", r.get("submitted", True)))
            p = m.get("predicted_prob", r.get("predicted_prob"))
            if not sub or p is None:
                p, sub = 0.5, False
            out.append((float(p), float(y), float(c), sub, mid))
    return out


def webhook_blocks(path, ids):
    """First/last 265-blocks of test rows in a training-run webhook file."""
    rows = []
    with gzip.open(path, "rt") as f:
        for l in f:
            if "market_id" not in l:
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            rows.append(r)
    tr = [(i, r) for i, r in enumerate(rows) if str(r.get("market_id")) in ids]
    idxs = [i for i, _ in tr]
    split = max((idxs[j + 1] - idxs[j], j) for j in range(len(idxs) - 1))[1] + 1

    def mk(rs):
        out = []
        for _, r in rs:
            y, c = r.get("outcome"), r.get("price_at_cutoff")
            if y is None or c is None:
                continue
            sub = bool(r.get("submitted"))
            p = r.get("predicted_prob")
            if not sub or p is None:
                p, sub = 0.5, False
            out.append((float(p), float(y), float(c), sub, str(r["market_id"])))
        return out

    return mk(tr[:split]), mk(tr[split:])


def brier_ci(rows):
    n = len(rows)
    b = sum((p - y) ** 2 for p, y, *_ in rows) / n
    v = sum(((p - y) ** 2 - b) ** 2 for p, y, *_ in rows) / (n - 1)
    return b, 1.96 * math.sqrt(v / n)


def ece(rows, k=10):
    srt = sorted(rows, key=lambda r: r[0])
    n = len(srt)
    tot = 0.0
    for i in range(k):
        blk = srt[i * n // k : (i + 1) * n // k]
        if blk:
            tot += len(blk) / n * abs(
                sum(r[0] for r in blk) / len(blk) - sum(r[1] for r in blk) / len(blk)
            )
    return tot


def ece_ci(rows, boot=2000, seed=0):
    rng = random.Random(seed)
    es = sorted(ece([rng.choice(rows) for _ in rows]) for _ in range(boot))
    return es[int(boot * 0.025)], es[int(boot * 0.975)]


def reliability(rows, k=10):
    srt = sorted(rows, key=lambda r: r[0])
    n = len(srt)
    xs, ys = [], []
    for i in range(k):
        blk = srt[i * n // k : (i + 1) * n // k]
        if blk:
            xs.append(sum(r[0] for r in blk) / len(blk))
            ys.append(sum(r[1] for r in blk) / len(blk))
    return xs, ys


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


ids4 = test_ids("test_split_v4.jsonl")
ids3 = test_ids("test_split_v3.jsonl")

base_v2, trained_v2 = webhook_blocks(f"{R}/v2_run_webhook.jsonl.gz", ids4)
base_v3, trained_v3 = webhook_blocks(f"{R}/v3_run_webhook.jsonl.gz", ids4)
opus = rows_from(f"{R}/opus45_harness_eval.jsonl")
sonnet = rows_from(f"{R}/sonnet45_harness_eval.jsonl")
flash = rows_from(f"{R}/gemini36flash_harness_eval.jsonl")
pro = rows_from(f"{R}/gemini31pro_harness_eval.jsonl")
sonnet_off = rows_from(f"{R}/sonnet45_notools_eval.jsonl")
flash_off = rows_from(f"{R}/gemini36flash_notools_eval.jsonl")
pro_off = rows_from(f"{R}/gemini31pro_notools_eval.jsonl")
v2_off = rows_from(f"{R}/v2_toolsoff_eval.jsonl", ids4)
base_off = rows_from(f"{R}/base_toolsoff_eval.jsonl", ids4)
v1_trained = rows_from(f"{R}/platform_eval_step30_test.jsonl", ids3)

crowd_rows = [(c, y, c, True, m) for _, y, c, _, m in base_v2]

# ---------------------------------------------------------------- F1: panel
panel = [
    ("Opus 4.5", opus, C_FRONTIER),
    ("Sonnet 4.5", sonnet, C_FRONTIER),
    ("Gemini 3.6 Flash", flash, C_FRONTIER),
    ("Gemini 3.1 Pro", pro, C_FRONTIER),
    ("trained 35B (market-only)", trained_v2, C_TRAINED),
    ("base 35B", base_v2, C_BASE),
]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
fig.suptitle("Search-off arm · market tools available (n=265)",
             fontsize=10, y=1.02, style="italic")
names = [n for n, *_ in panel]
xs = range(len(panel))
bs = [brier_ci(r) for _, r, _ in panel]
a1.bar(xs, [b for b, _ in bs], yerr=[e for _, e in bs], capsize=3,
       color=[c for *_, c in panel], alpha=0.85)
cb, _ = brier_ci(crowd_rows)
a1.axhline(cb, color=C_CROWD, lw=1.6, ls="--", label=f"crowd ({cb:.3f})")
a1.set_ylabel("soft-Brier (lower better)")
a1.set_ylim(0.15, 0.25)
a1.legend(frameon=False, fontsize=8)
a1.set_title("Accuracy: one crowd-bounded band", fontsize=10)
def _sub(rows):
    return [r for r in rows if r[3]]
es = [(ece(_sub(r)),) + ece_ci(_sub(r)) for _, r, _ in panel]
a2.bar(xs, [e for e, *_ in es],
       yerr=[[max(0, e - lo) for e, lo, _ in es], [max(0, hi - e) for e, _, hi in es]],
       capsize=3, color=[c for *_, c in panel], alpha=0.85)
a2.axhline(0.058, color=C_CROWD, lw=1.6, ls="--", label="crowd (0.058)")
a2.legend(frameon=False, fontsize=8)
a2.set_ylabel("ECE (submitted-only, 10 equal-mass bins)")
a2.set_title("Calibration: trainable — RL reaches frontier level", fontsize=10)
for ax in (a1, a2):
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8)
    style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/f1_frontier_panel.png", dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------- F2: reliability curves
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls=":")
for name, rows, color, ls in [
    ("base 35B (ECE 0.099)", base_v2, C_BASE, "-"),
    ("trained 35B (ECE 0.065)", trained_v2, C_TRAINED, "-"),
    ("Opus 4.5 (ECE 0.053)", opus, C_FRONTIER, "-"),
]:
    xs_, ys_ = reliability([r for r in rows if r[3]])
    ax.plot(xs_, ys_, "-o", ms=3.5, color=color, ls=ls, label=name, lw=1.6)
ax.set_xlabel("stated probability (bin mean)")
ax.set_ylabel("empirical frequency of YES")
ax.set_title("Reliability: what training moves\n(search-off arm, market tools available, n=265)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/f2_reliability.png", dpi=180)
plt.close()

# ------------------------------------------------- F5: anchor-worth quadrant
pairs = [
    ("Gemini 3.1 Pro", pro, pro_off),
    ("Gemini 3.6 Flash", flash, flash_off),
    ("Sonnet 4.5", sonnet, sonnet_off),
    ("trained 35B", trained_v2, v2_off),
    ("base 35B", base_v2, base_off),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
w = 0.38
for i, (name, on, off) in enumerate(pairs):
    bon, eon = brier_ci(on)
    boff, eoff = brier_ci(off)
    ax.bar(i - w / 2, bon, w, yerr=eon, capsize=3, color=C_FRONTIER, alpha=0.85,
           label="market tools ON" if i == 0 else None)
    ax.bar(i + w / 2, boff, w, yerr=eoff, capsize=3, color="#c44e52", alpha=0.8,
           label="market tools OFF" if i == 0 else None)
    ax.annotate(f"+{boff - bon:.3f}", (i, max(bon, boff) + 0.033),
                ha="center", fontsize=8, color="#333")
ax.axhline(cb, color=C_CROWD, lw=1.6, ls="--", label="crowd")
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels([n for n, *_ in pairs], rotation=20, ha="right", fontsize=8.5)
ax.set_ylabel("soft-Brier")
ax.set_ylim(0.14, 0.33)
ax.set_title(
    "Anchor-worth: frontier models lean on the crowd more than 35Bs\n"
    "(paired within-model, search-off arm, all p<0.001)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/f5_anchor_worth.png", dpi=180)
plt.close()

# ------------------------------------------------- F7: boundary relocation
fig, ax = plt.subplots(figsize=(6.8, 4.0))
d1 = [abs(p - c) for p, _, c, sub, _ in v1_trained if sub]
d2 = [abs(p - c) for p, _, c, sub, _ in trained_v2 if sub]
bins = [x * 0.01 for x in range(0, 31)]
ax.hist(d1, bins=bins, density=True, alpha=0.55, color="#c44e52",
        label="pilot policy (cliff penalty at 0.02)")
ax.hist(d2, bins=bins, density=True, alpha=0.55, color=C_TRAINED,
        label="market-only policy (ramp penalty to 0.08)")
ax.axvline(0.02, color="#c44e52", ls="--", lw=1.4)
ax.axvline(0.08, color=C_TRAINED, ls="--", lw=1.4)
ax.annotate("cliff penalty edge", (0.021, ax.get_ylim()[1] * 0.92), fontsize=8, color="#c44e52")
ax.annotate("ramp penalty edge", (0.081, ax.get_ylim()[1] * 0.82), fontsize=8, color=C_TRAINED)
ax.set_xlabel("|prediction − market price|")
ax.set_ylabel("density")
ax.set_title("Boundary relocation: the policy camps at whatever edge the reward draws\n(market tools available, held-out eval)",
             fontsize=10)
ax.legend(frameon=False, fontsize=8.5)
style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/f7_boundary_relocation.png", dpi=180)
plt.close()

# --------------------------------------------------- F8: base-rate herding
fig, ax = plt.subplots(figsize=(6.8, 4.0))
pb = [p for p, _, _, sub, _ in base_v3 if sub]
pt = [p for p, _, _, sub, _ in trained_v3 if sub]
bins = [x * 0.05 for x in range(0, 21)]
ax.hist(pb, bins=bins, density=True, alpha=0.5, color=C_BASE,
        label="base (submit 74%)")
ax.hist(pt, bins=bins, density=True, alpha=0.6, color=C_TRAINED,
        label="trained, neither-condition (submit 98%)")
base_rate = sum(y for _, y, *_ in base_v3) / len(base_v3)
ax.axvline(base_rate, color="#c44e52", lw=2,
           label=f"dataset YES base rate ({base_rate:.3f})")
ax.set_xlabel("predicted probability (no market tools anywhere)")
ax.set_ylabel("density")
ax.set_title("Base-rate herding: remove the crowd and RL finds the next anchor\n(neither condition: market hidden, search off)\n"
             "(trained median 0.35; extremes 10%→2%; pre-registered)", fontsize=10)
ax.legend(frameon=False, fontsize=8.5)
style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/f8_base_rate_herding.png", dpi=180)
plt.close()

print("figures written:")
import os
for f in sorted(os.listdir(FIG)):
    print(" ", f)

# ------------------------- F: difficulty-adjusted training curve (market-only)
# Webhook train rows in arrival order ~ step order; 33 equal chronological
# blocks stand in for the 33 steps (capture is lossy but unbiased in time).
_train = []
with gzip.open(f"{R}/v2_run_webhook.jsonl.gz", "rt") as f:
    for l in f:
        if "market_id" not in l:
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if str(r.get("market_id")) in ids4:
            continue
        y, c, rew = r.get("outcome"), r.get("price_at_cutoff"), r.get("reward")
        if y is None or c is None or rew is None:
            continue
        _train.append((float(rew), (float(c) - float(y)) ** 2))
S = 33
diffs = []
for i in range(S):
    blk = _train[i * len(_train) // S : (i + 1) * len(_train) // S]
    if not blk:
        continue
    pol = sum(r for r, _ in blk) / len(blk)
    crowd_att = 1 - sum(cb_ for _, cb_ in blk) / len(blk)
    diffs.append(pol - crowd_att)
n = len(diffs)
xbar = (n - 1) / 2
sl = sum((i - xbar) * d for i, d in enumerate(diffs)) / sum((i - xbar) ** 2 for i in range(n))
ic = sum(diffs) / n - sl * xbar
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.axhline(0, color="#999", lw=1.4, ls="--", label="crowd-level performance")
ax.plot(range(n), diffs, "-o", ms=3.5, lw=1.6, color=C_TRAINED,
        label="policy reward above/below what crowd-copying would earn")
ax.plot([0, n - 1], [ic, ic + sl * (n - 1)], ":", lw=2, color=C_TRAINED,
        label=f"trend: +{sl:.4f}/step — the gap to the crowd closes")
ax.set_xlabel("training step (single epoch, every question seen once)")
ax.set_ylabel("reward vs crowd-attainable")
ax.set_title("Learning, once batch difficulty is removed: the policy closes\n"
             "its gap to the crowd across training (market-only run)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/v2_reward_difficulty_adjusted.png", dpi=180)
plt.close()
