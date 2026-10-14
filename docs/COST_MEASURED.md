# Measured protection cost (Kaggle T4, 2026-10-15) vs the 48 h plan

All numbers are measured, from `results/J01`, `results/S04t`, `results/S04u` (4–70 images; per-image GPU seconds, 512 px, K = 4 purifier draws, checkpointing on). "x plan" compares with `docs/V2_PLAN_48H.md` §1. Smoke runs: never report protection quality from them.

## Finding 1: the planner's cost model was 2–6x too optimistic for v2

S04t (VAE in fp32, as the code originally ran it), 10 iterations, 4 images, s per iteration: v2_cheap 8.7 (plan 1.35), v2_main 11.3 (3.0), v2_antipure 15.9 (3.75), v2_final 14.5 (4.75), PhotoGuard 0.27, Mist 0.55 (official 100 iterations ≈ 55 s, matches plan). Cost is linear in iterations (5-iteration S04 runs gave the same s/iter), so it is not warmup.

## Finding 2: fp16 autocast for the VAE (`runtime.vae_dtype_protect: autocast`) is ~2x faster and behaves like fp32

S04u, same 4 images, 10 iterations, same seeds:

| method | fp32 s/img | autocast s/img | speed-up | PSNR fp32 vs autocast (per image) | non-finite draws | zero-gradient fraction |
|---|---|---|---|---|---|---|
| v2_main | 135.1 | 65.7 | 2.05x | 43.00/43.02, 43.50/43.47, 42.96/42.93, 45.42/45.39 | 0 | 0.000 |
| v2_cheap | 86.9 (S04t) | 36.4 | 2.4x | – | 0 | 0.000 |
| v2_final | 145.0 (S04t) | 98.7 | 1.47x | – | 0 | 0.000 |

`L_vis` goes from the same starting value to nearly the same final value (v2_main, 4 images: 2.139→11.44 vs 11.57, 3.359→9.931 vs 10.11, 2.601→12.74 vs 12.81, 4.439→11.32 vs 11.11). Only 4 images × 10 iterations: this shows the gradients are usable, not that final protection is identical. Without gradient checkpointing, v2_main runs out of memory on a 15 GB T4 (S04u), so checkpointing stays on.

## Projected cost at 40 iterations (autocast), per image

v1 46 s (1.8x plan) · v2_cheap 146 s (2.7x) · v2_main 263 s (2.2x) · v2_final 395 s (2.1x) · v2_antipure ≈ 433 s (2.9x; estimated from S04t scaled by the v2_final ratio, not measured with autocast) · PhotoGuard (1000 iters) ≈ 270 s · Mist 55 s. Ablations (abl_*) and eps* are not measured; assumed 250 s and 395 s.

## What this does to the plan (protection only, GPU-hours summed over both GPUs)

| item | images | plan | now |
|---|---|---|---|
| v1 | 120 | 0.8 | 1.5 |
| v2_cheap | 100 | 1.5 | 4.1 |
| v2_main | 120 | 4.0 | 8.8 |
| v2_antipure | 100 | 4.2 | 11.9 |
| v2_final (+ Optuna separate) | 120 | 6.3 | 13.2 |
| 4 ablations (assumed 250 s) | 400 | 10.8 | 27.8 |
| eps 2/6/8 | 300 | 15.8 | 32.9 |
| adaptive-attacker pool (K=2) | 120 | 3.3 | 6.7 |
| PhotoGuard + Mist | 240 | 7.0 | 11.0 |
| **total** | | **≈54** | **≈118** |

Edits are unchanged from the plan (≈77 GPU-h, unmeasured until S05 passes). Total ≈195 GPU-h instead of 131. If Kaggle bills wall-clock (≈98 session-hours of 150) this is tight but possible; if it bills per GPU (195 of 150) it does not fit. Applying fallback cuts "K=2 for ablations and eps sweep" saves ≈30 GPU-h of protection.
