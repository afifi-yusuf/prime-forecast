"""Regenerate the figures used in the paper (results/figures/f10, f12, f14*).

The values below are the headline cells reported in results/RESULTS.md and
are derived from the archived per-rollout records in results/:

  evidence-based full set (n=265, soft-Brier, 0.5 imputed for non-answers):
    trained Qwen3.5-35B-A3B  v4 endpoint, pooled 2 rollouts/question
                             (v4_continue_webhook.jsonl.gz)          0.254
    Claude Opus 4.5          opus45_searchon_marketoff_eval.jsonl     0.256
    Claude Sonnet 4.5        sonnet45_searchon_marketoff_eval.jsonl   0.273
    Gemini 3.1 Pro           gemini31pro_searchon_marketoff_eval.jsonl 0.278
    Gemini 3.6 Flash         gemini36flash_searchon_marketoff_eval.jsonl 0.286
  uncertain subset (price_at_cutoff in [0.30, 0.70], n=104), same files:
    0.274 / 0.281 / 0.295 / 0.316 / 0.319 (crowd 0.232)
  ECE (submitted-only, 10 equal-mass bins) and submission rate, same files.
  base vs trained ECE 0.185 -> 0.128 and coverage 79% -> 99.6% are the
  step-0 and endpoint test-market-off evals of the v4 run
  (v4_main_platform_metrics.json, v4_continue_platform_metrics.json).

Run from the repository root:  python scripts/make_paper_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "results/figures"
BLUE, ORANGE = "#4C72B0", "#DD8452"
GREEN, GRAY = "#009B77", "#9a9a9a"


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.22)


# ----------------------------------------------- F10: evidence-based column
models = ["Gemini\n3.6 Flash", "Gemini\n3.1 Pro", "Claude\nSonnet 4.5",
          "Claude\nOpus 4.5", "Qwen3.5-35B\n(trained, ours)"]
full_ = [0.286, 0.278, 0.273, 0.256, 0.254]
unc = [0.319, 0.316, 0.295, 0.281, 0.274]
x = np.arange(5)
w = 0.38
fig = plt.figure(figsize=(7.6, 3.3), dpi=200)
gs = fig.add_gridspec(2, 1, height_ratios=[8, 1], hspace=0.08)
axT = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1], sharex=axT)
for ax in (axT, axB):
    ax.bar(x - w / 2, full_, w, color=BLUE, label="full test set (n=265)")
    ax.bar(x + w / 2, unc, w, color=ORANGE, label="uncertain subset (n=104)")
axT.set_ylim(0.24, 0.34)
axB.set_ylim(0, 0.013)
for xi, v in zip(x, full_):
    axT.text(xi - w / 2, v + 0.0025, f"{v:.3f}", ha="center", fontsize=8.2)
for xi, v in zip(x, unc):
    axT.text(xi + w / 2, v + 0.0025, f"{v:.3f}", ha="center", fontsize=8.2)
axT.spines["bottom"].set_visible(False)
axB.spines["top"].set_visible(False)
axT.tick_params(bottom=False, labelbottom=False)
axB.set_yticks([0])
axB.set_yticklabels(["0"])
axB.set_xticks(x)
axB.set_xticklabels(models, fontsize=9)
axB.get_xticklabels()[-1].set_fontweight("bold")
d = 0.012
axT.plot((-d, +d), (-d, +d), transform=axT.transAxes, color="k", clip_on=False, lw=1)
axB.plot((-d, +d), (1 - 8 * d, 1 + 8 * d), transform=axB.transAxes, color="k",
         clip_on=False, lw=1)
axT.set_ylabel("soft-Brier (lower is better)")
axT.legend(fontsize=8.5, loc="upper right", frameon=False)
for a in (axT, axB):
    style(a)
plt.savefig(f"{FIG}/f10_evidence_based_column.png", bbox_inches="tight")
plt.close()

# ------------------------------------------ F14: ECE and submission rate
models14 = ["Qwen3.5-35B\n(trained, ours)", "Claude\nOpus 4.5", "Claude\nSonnet 4.5",
            "Gemini\n3.1 Pro", "Gemini\n3.6 Flash"]
ece = [0.128, 0.136, 0.176, 0.215, 0.205]
sub = [99.6, 99.6, 99.2, 94.0, 100.0]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.6), dpi=200)
ax = axes[0]
ax.bar(x, ece, 0.55, color=BLUE)
for xi, v in zip(x, ece):
    ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8.2)
ax.set_ylim(0, 0.24)
ax.set_ylabel("ECE (lower is better)", fontsize=9.5)
ax = axes[1]
ax.bar(x, sub, 0.55, color=BLUE)
for xi, v in zip(x, sub):
    ax.text(xi, v + 2.5, f"{v:.1f}".rstrip("0").rstrip(".") + "%", ha="center", fontsize=8.2)
ax.set_ylim(0, 112)
ax.set_ylabel("submission rate (%)", fontsize=9.5)
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(models14, fontsize=7.8)
    ax.get_xticklabels()[0].set_fontweight("bold")
    style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/f14_ece_submit.png", bbox_inches="tight")
plt.close()

# ECE-only variant (descending, ours rightmost)
models14b = ["Gemini\n3.1 Pro", "Gemini\n3.6 Flash", "Claude\nSonnet 4.5",
             "Claude\nOpus 4.5", "Qwen3.5-35B\n(trained, ours)"]
ece_b = [0.215, 0.205, 0.176, 0.136, 0.128]
fig, ax = plt.subplots(figsize=(6.0, 2.0), dpi=200)
ax.bar(x, ece_b, 0.55, color=BLUE)
for xi, v in zip(x, ece_b):
    ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, 0.245)
ax.set_ylabel("ECE (lower is better)", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(models14b, fontsize=8.5)
ax.get_xticklabels()[-1].set_fontweight("bold")
style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/f14_ece_only.png", bbox_inches="tight")
plt.close()

# --------------------------------------- F12: base vs trained behavior
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.3), dpi=200)
ax = axes[0]
vals = [0.185, 0.128]
ax.bar([0, 1], vals, color=[GRAY, GREEN], width=0.55)
for x0, v in zip([0, 1], vals):
    ax.text(x0, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
ax.text(0.5, 0.185, "$-31\\%$", ha="center", fontsize=11, color=GREEN, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["base", "trained"], fontsize=10)
ax.set_ylim(0, 0.212)
ax.set_ylabel("ECE (lower is better)")
ax = axes[1]
vals = [79, 99.6]
ax.bar([0, 1], vals, color=[GRAY, GREEN], width=0.55)
for x0, v, lbl in zip([0, 1], vals, ["79%", "99.6%"]):
    ax.text(x0, v + 2, lbl, ha="center", fontsize=10)
ax.text(0.5, 108, "$+26\\%$", ha="center", fontsize=11, color=GREEN, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["base", "trained"], fontsize=10)
ax.set_ylim(0, 119)
ax.set_ylabel("questions answered (%)")
for ax in axes:
    style(ax)
plt.tight_layout()
plt.savefig(f"{FIG}/f12_rlvr_behavioral_gains.png", bbox_inches="tight")
plt.close()

print("wrote f10_evidence_based_column, f14_ece_submit, f14_ece_only, f12_rlvr_behavioral_gains")
