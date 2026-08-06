# Prime training-run registry (identify runs by ID prefix in the UI)

Status legend: KEEP = results used in the paper · DISCARD = data unused ·
CONTROL = comparison policy. UI shows first ~8 chars of each ID.

## The five research runs (KEEP)

| ID (prefix) | what it is | status | role |
|---|---|---|---|
| `rqn28a9d` | **v1 main** — 45 steps, 9 epochs, cliff penalty, Firecrawl era | COMPLETED | archive: Findings 1–2 |
| `mjkvxreh` | **v2 main** ("1eopch") — single epoch, ramp penalty, market on | COMPLETED | Findings 3–8 |
| `slgbosbz` | **v3** — no market tools, pure Brier (web-search-off control) | COMPLETED | Finding 10, v4 control |
| `r64dayfb` | **v4 main** — no market, working search; credits ran out step 29 | STOPPED | training telemetry + step-0/base evals |
| `basziye6` | **v4 continuation** — warm-start step-22 → step 33 + final evals | COMPLETED | Finding 11, v4 endpoint |
| `zvxexo35` | **v5** — market ON + working search, pure Brier | RUNNING | the final 2×2 cell |

## Probes & eval runs (KEEP — data in results/)

| ID | what |
|---|---|
| `p6pphpg8` | v1 step-30 platform-eval probe (webhook methodology debut) |
| `y6g24dpz` | v2-era eval probe (Jul 30) |

## Discarded (data unused — safe to ignore/delete in UI)

| ID | why discarded |
|---|---|
| `yp3x5007`, `f2r21n5y` | FAILED at config validation (duplicate eval-env names) — $0 |
| `kh57chok`, `gj5gavyb` | probes on stale-secret Brave-only search — poisoned behavior |
| `thhpjdvh`, `y9rbpcg5` | probes on degraded chain search — poisoned behavior |
| `xzmqo4zh` | v4 first attempt — ABORTED step ~2 (date-parse leak, fixed in 0.1.18) |
| `fxzqz3lq` | v2-era continuation stopped by user as unnecessary |

## Pre-summary-era runs (early pilots/smokes — all superseded)

`giaaj00b`, `xttjaf7v`, `u6qkyxxs`, `w0hmwkb3`, `m483fh70`, `km5kefn9`,
`pwuis6at`, `em54vh0x`, `o30dztza`, `mbdffm2t`, `lcqdqt0j` — 9B smoke tests
and 35B pilots from before the v1 naming scheme. No paper data.

## Checkpoints that matter

- `mo3l86ms` — v4-main step 22 (the v4 warm-start parent)
- `vjv9wojk` — v2 step 22 (used for market-off eval cells)
- v2/v3/v4c step-33 checkpoints: lost to the platform upload bug (adapters
  survived for v4c step 33; see artifacts/adapters/MANIFEST.json)
