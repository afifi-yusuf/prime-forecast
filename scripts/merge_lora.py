#!/usr/bin/env python3
"""Merge a per-expert LoRA adapter into a Qwen3.5 MoE checkpoint, one shard at a time.

The adapter trained with prime-rl (e.g. DYOR v4 on Qwen/Qwen3.5-35B-A3B) stores one LoRA per expert
(`experts.<e>.{gate,up,down}_proj`) plus the full-attention q/k/v/o projections. The base checkpoint
stores each layer's experts fused: `experts.gate_up_proj` [E, 2I, H] (gate rows first, then up, as in
the transformers forward `gate, up = linear(x, W[e]).chunk(2, dim=-1)`) and `experts.down_proj` [E, H, I].
Merging writes W += (alpha / r) * B @ A into those slices, so the result serves as a plain checkpoint
(vLLM `--quantization fp8` fits it on one 48 GB GPU, which a bf16 base plus runtime LoRA does not).

    python scripts/merge_lora.py --base <hf snapshot dir> --adapter <adapter dir> --out <dir>

Memory: one base shard (~5 GB) plus the adapter (~2.5 GB) at a time. Tensor names, shapes, dtypes and
shard layout are unchanged, so the base index and configs are copied as-is. The run fails, and removes
its output, if a delta's shape mismatches its base slice or any adapter group is left unused.
Untouched: the vision tower, the MTP head (not adapted; do not use it for speculative decoding).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

KEY = re.compile(r"^(?:base_model\.model\.)?(?P<mod>.+?)(?:\.experts\.(?P<e>\d+))?\.(?P<proj>\w+_proj)\.lora_(?P<ab>[AB])\.weight$")


def load_adapter(adapter: Path) -> tuple[dict, float]:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("use_dora"):
        raise SystemExit("rsLoRA / DoRA adapters are not supported")
    scale = cfg["lora_alpha"] / cfg["r"]
    # (module, proj) -> {"A": {expert|None: tensor}, "B": {...}}
    lora: dict = defaultdict(lambda: {"A": {}, "B": {}})
    for k, t in load_file(str(adapter / "adapter_model.safetensors")).items():
        m = KEY.match(k)
        if not m:
            raise SystemExit(f"unrecognised adapter key: {k}")
        e = int(m["e"]) if m["e"] is not None else None
        lora[(m["mod"], m["proj"], e is not None)][m["ab"]][e] = t
    return lora, scale


def delta(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    return (B.float() @ A.float()) * scale


def stacked(entry: dict, n_experts: int, scale: float) -> torch.Tensor:
    """[E, out, in] delta for an expert projection; every expert must be present."""
    missing = set(range(n_experts)) - set(entry["A"]) | set(range(n_experts)) - set(entry["B"])
    if missing:
        raise SystemExit(f"adapter lacks experts {sorted(missing)[:5]}...")
    A = torch.stack([entry["A"][e] for e in range(n_experts)]).float()
    B = torch.stack([entry["B"][e] for e in range(n_experts)]).float()
    return torch.bmm(B, A) * scale


def merge_tensor(name: str, W: torch.Tensor, lora: dict, scale: float, used: set) -> torch.Tensor:
    if name.endswith(".experts.gate_up_proj"):
        mod = name[: -len(".experts.gate_up_proj")]  # "...layers.<n>.mlp", as the adapter keys parse
        g, u = (mod, "gate_proj", True), (mod, "up_proj", True)
        if g not in lora and u not in lora:
            return W
        E, two_i, _ = W.shape
        out = W.float()
        for k, rows in ((g, slice(0, two_i // 2)), (u, slice(two_i // 2, two_i))):
            if k in lora:
                d = stacked(lora[k], E, scale)
                if d.shape != out[:, rows].shape:
                    raise SystemExit(f"{name} {k[1]}: delta {tuple(d.shape)} vs slice {tuple(out[:, rows].shape)}")
                out[:, rows] += d
                used.add(k)
        return out.to(W.dtype)
    if name.endswith(".experts.down_proj"):
        k = (name[: -len(".experts.down_proj")], "down_proj", True)
        if k not in lora:
            return W
        d = stacked(lora[k], W.shape[0], scale)
        if d.shape != W.shape:
            raise SystemExit(f"{name}: delta {tuple(d.shape)} vs weight {tuple(W.shape)}")
        used.add(k)
        return (W.float() + d).to(W.dtype)
    m = re.match(r"^(.+)\.(\w+_proj)\.weight$", name)
    if m and (k := (m[1], m[2], False)) in lora:
        d = delta(lora[k]["A"][None], lora[k]["B"][None], scale)
        if d.shape != W.shape:
            raise SystemExit(f"{name}: delta {tuple(d.shape)} vs weight {tuple(W.shape)}")
        used.add(k)
        return (W.float() + d).to(W.dtype)
    return W


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="local HF snapshot dir of the base model")
    ap.add_argument("--adapter", type=Path, required=True, help="dir with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    lora, scale = load_adapter(args.adapter)
    print(f"adapter: {len(lora)} target groups, scale alpha/r = {scale}")
    args.out.mkdir(parents=True, exist_ok=True)
    for f in args.base.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, args.out / f.name)

    used: set = set()
    shards = sorted(args.base.glob("*.safetensors"))
    for i, shard in enumerate(shards, 1):
        tensors, meta = {}, None
        with safe_open(str(shard), framework="pt") as fh:
            meta = fh.metadata()
            for name in fh.keys():
                tensors[name] = merge_tensor(name, fh.get_tensor(name), lora, scale, used)
        save_file(tensors, str(args.out / shard.name), metadata=meta or {"format": "pt"})
        print(f"[{i}/{len(shards)}] {shard.name}: groups merged so far {len(used)}/{len(lora)}")

    unused = set(lora) - used
    if unused:
        shutil.rmtree(args.out)
        raise SystemExit(f"{len(unused)} adapter groups matched no base tensor, e.g. {sorted(unused)[:3]}; output removed")
    (args.out / "MERGED_FROM.json").write_text(json.dumps(
        {"base": str(args.base), "adapter": str(args.adapter), "scale": scale, "groups": len(used)}, indent=2))
    print(f"merged {len(used)} groups into {args.out}")


if __name__ == "__main__":
    main()
