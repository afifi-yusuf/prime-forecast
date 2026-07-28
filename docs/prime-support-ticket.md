# Support ticket: served LoRA adapter does not reproduce training-time behavior (Qwen3.5-35B-A3B, Hosted Training)

**Account:** yafifi · **Run:** `rqn28a9dizg2ete43qgzqyry` (Qwen/Qwen3.5-35B-A3B, LoRA r=16 α=32, env `yafifi/prime-forecast@0.1.12`, 45 steps, completed 2026-07-28)

## Summary

Adapters exported from this run, merged into the base model with prime-rl's exact
semantics and served via stock vLLM, reproduce only a fraction of the behavior the
same policy displayed during training rollouts. We have exhausted client-side
explanations and believe the difference lives in the Hosted Training inference
stack. We'd like to know what differs, and ideally a supported way to reproduce
training-time inference behavior outside the platform.

## Observed gap (same weights, same questions, same env code)

| | during training (steps 40–44, train batches) | our serving (merged adapter, same train questions) |
|---|---|---|
| submit-tool usage | 72–97% of rollouts | 29% (n=24) / 48% on held-out (n=160) |
| behavior | fast, decisive research→submit | base-model-like extended researching |

Base-model control through our identical stack: 29–31% submit — i.e. the adapter
moves behavior +17pts on held-out (real effect) but nowhere near training-time.

## Client-side checks already performed (all passed)

1. **Adapter artifacts are genuine**: `Latest`, step-40, and step-20 adapters all
   downloaded via `/rft/adapters/{id}/download`; LoRA-B magnitudes grow
   monotonically over training (3.64e-4 → 4.04e-4 → 4.06e-4), two independent
   export paths (interval vs run-end) agree.
2. **Merge semantics match prime-rl exactly**:
   - scaling = alpha/rank = 2.0 (`trainer/runs.py:560`, `lora/base.py` docstring)
   - expert mapping w1/w2/w3 = gate/down/up (`models/conversion_ops.py:321`,
     `GATE_DOWN_UP`), fused `gate_up_proj` split dim-1 gate-first (matches
     transformers `Qwen3_5MoeExperts.chunk(2, dim=-1)`)
   - B@A orientation per `multi_moe.py` grouped-mm math
   - merge verified tensor-level: (merged − base) ≡ 2.0·B@A to bf16 precision
3. **Serving checks**: vLLM 0.26, `--tool-call-parser qwen3_xml` (calls parse
   correctly), `--reasoning-parser qwen3`, sampling from the model's
   generation_config (temp 1.0, top_p 0.95, top_k 20), max_tokens 12288/turn,
   same verifiers env version as training (0.1.12), same max_turns.

## Questions

1. Does the Hosted Training inference server (prime-rl patched vLLM) construct
   multi-turn rollout contexts differently from stock vLLM chat completions —
   e.g. retaining prior-turn reasoning content, custom chat template, or
   template kwargs?
2. What sampling parameters does the orchestrator send during training rollouts
   (temperature/top_p/top_k), and do they differ from generation_config?
3. Is there a supported way to run inference with a trained adapter that
   reproduces training-time behavior — e.g. LoRA deployment support for
   Qwen/Qwen3.5-35B-A3B (currently rejected: "Base model is not currently
   available for LoRA deployment"), or checkpoint deployment (currently
   returning HTTP 502 for READY checkpoints, e.g. `fn68s0t1zoi7mkmdg5fo4nm0`)?

## Secondary issues from the same run

- Step-40 checkpoint upload hung in `UPLOADING` for hours, then vanished;
  the documented automatic final checkpoint never appeared. `keep_cloud`
  default may also have contributed. Net effect: no end-of-run checkpoint exists.
- Checkpoint deployment API has returned HTTP 502 for ~24h.
- `prime deployments create <adapter>` rejects this base model for LoRA serving.
- Val metrics were configured (`[val] num_examples=16 interval=10/5`) but never
  appeared in `/rft` metrics; if val requires additional env-spec wiring, the
  config docs don't mention it.
