# Phase 0 report: reproducible base

Date: 2026-10-14 · Code: branch `v2/phase-0` (tags `v1-unpatched` = f8e8329, `v1.0` = first commit with the three patches) · Runs: J00 (environment probe) and J01 (v1 reproduction), Kaggle acct1, 2026-10-14, bundle B0 = `v2/phase-0` @ 50dd3b4 (clean tree); J01b (second person) = not run (optional)

STATUS: **done. The GPU reproduction J01 ran on 70 images and passes the proposed gate.** Numbers below are copied from the files generated in the run (`results/J01/results/J01/{budget_check.md, tables/invisibility.md, compare_v1.md, summary.json, gpu_time.json}`), not computed by hand.

## Result first (plain language)

- The three runtime patches from the v1 notebook are now ordinary commits, and v1 runs through one command (`scripts/run_protect.py --method v1`). Nothing changes v1's algorithm.
- **Budget confirmed: v1's "8/255" is 4/255 on the standard 0-255 scale.** In the reproduced protected PNGs the largest change of any pixel value is exactly 4 gray levels, in all 70 images. `budget.eps_255 = 4`, `budget.eps_confirmed: true`.
- **v1's invisibility numbers reproduce.** PSNR 41.99 dB (v1: 41.91), SSIM 0.9754 (v1: 0.9750), LPIPS 0.2314 (v1: 0.2316), n = 70; every paired difference is far inside the accepted tolerance. 0 of 70 images break PSNR >= 38 or SSIM >= 0.95.
- Cost: 45.6 GPU-s per image on a T4 (40 iterations), about 1.8x the planner's estimate of 25 s; 29 min wall for 70 images on 2 GPUs.

## Budget finding

| Source | n images | max abs gray-level change | share of values at max | inferred eps (standard) |
|---|---|---|---|---|
| v1 original run, tensors in [-1,1] (`stage1_per_image_metrics.csv`, column Linf) | 70 | 0.0313725 on [-1,1] (= 8/255) | n/a | 4/255 by conversion (Linf*255/2) |
| v1 original protected PNGs | not available (the zip has no images) | – | – | – |
| Reproduced v1 run J01, saved PNGs vs the JPEG q75 clean inputs (`budget_check.md`) | 70 | **4** (in all 70 images; no value above 4) | 9.35% of all pixel values | **4/255** |

Conclusion: confirmed. `budget.eps_255 = 4` and `eps_confirmed: true` (run J01) are set in `config/config.yaml`. All v2 budgets are on the standard scale from here on.
Distribution of |gray-level change| over all pixel values (J01): 0: 11,468,029; 1: 18,754,991; 2: 11,998,407; 3: 7,683,111; 4: 5,145,702; >= 5: 0.
Step-size finding: v1 used `alpha = 1/255` on [-1,1], which is **0.5/255 on the standard scale**, not 1/255 as the handoff table says. `methods.v1` uses `step_255: 0.5`. The v2 default step is a Phase 3 decision.

## Reproduction of v1 Phase 1 (invisibility)

Reference = v1's original stored results (70 images, `data/v1_ref/stage1_summary_metrics.csv`): PSNR 41.911 ± 0.440 dB, SSIM 0.97496 ± 0.00480, LPIPS(VGG) 0.23156 ± 0.06304, Linf 0.0313725; 0 violations of PSNR >= 38 and SSIM >= 0.95.

| Metric | v1 original run | Reproduced, v1 convention: mean ± std [95% CI] | Reproduced, published PNG: mean ± std [95% CI] | Violations |
|---|---|---|---|---|
| PSNR (dB) | 41.911 ± 0.440 | 41.986 ± 0.416 [41.890, 42.082] | 42.171 ± 0.436 [42.070, 42.272] | 0 / 0 of 70 below 38 |
| SSIM (windowed) | 0.97496 ± 0.00480 | 0.9754 ± 0.0054 [0.9741, 0.9766] | 0.9756 ± 0.0054 [0.9744, 0.9769] | 0 / 0 of 70 below 0.95 |
| LPIPS VGG (clean vs protected) | 0.23156 ± 0.06304 | 0.2314 ± 0.0637 [0.2168, 0.2463] | 0.2305 ± 0.0635 [0.2160, 0.2454] | n/a |
| Linf ([-1,1]) | 8/255 | 0.0313725 = 8/255 in all 70 | n/a (PNG max change = 4/255) | n/a |

n = 70, 1 protection seed per image, 95% bootstrap CI over images in brackets. "v1 convention" = tensor -> uint8 by truncation, as v1 measured; "published PNG" = the rounded image that would actually be posted.

Paired comparison with v1's per-image CSV (pairing by original file name, `compare_v1.md`), J01 minus v1:

| Metric | mean J01 | mean v1 | mean difference [95% CI] | tolerance | within |
|---|---|---|---|---|---|
| PSNR (dB) | 41.9863 | 41.9111 | +0.0752 [+0.0021, +0.1507] | ±0.5 | yes |
| SSIM | 0.9754 | 0.9750 | +0.0004 [-0.0002, +0.0010] | ±0.005 | yes |
| LPIPS | 0.2314 | 0.2316 | -0.0002 [-0.0042, +0.0039] | ±0.03 | yes |

The PSNR difference is statistically above zero but tiny (0.075 dB, 15% of the tolerance); the likely cause is the different random draws (per-image seeds instead of v1's global seed 42, see below), not a change in the method (unverified).
Second-person run agreement (`compare_runs.py --a J01 --b J01b`): not run. J01b is optional (only needed for a "reproduced by a second person" claim); without it the report must not make that claim.
Gate with the accepted tolerance (mean PSNR within ±0.5 dB, SSIM ±0.005, LPIPS ±0.03; Linf exactly 8/255; PNG max |diff| = 4): **PASS** on every item. The user accepted this tolerance on 2026-10-14.

## Differences from the original v1 run (and their expected effect)

- Same 70 images and the same weights (alpha 0.2699475469647097, beta 0.9023875567667267, gamma 60.52658429696642, Optuna trial 23), read from v1's own files.
- Clean inputs: 512 px JPEG at PIL default quality 75, replicated exactly as v1 did it (`scripts/prepare_v1_data.py`).
- Seeds: v1 used one global seed 42 and shuffled image order; v2 derives a seed per image (sha256 of base seed, purpose and image id) and processes images in sorted order. Bit-exact reproduction is therefore impossible; reproduction is statistical, and the comparison is per image only in distribution.
- GPU kernels are non-deterministic; library versions differ from the original v1 session. J00 recorded the Kaggle image (torch 2.11.0+cu128, diffusers 0.40.0, transformers 5.16.1, kornia 0.8.3, Python 3.13.15, CUDA 12.8); `requirements.txt` is now pinned to it.
- The v1 notebook's runtime patches are now source code. The code in this repo contains the exact patched text, so the old notebook's idempotent patch cell skips them.
- Two housekeeping changes with no effect on the algorithm: `DiffShieldLoss` can take model ids/components as arguments (defaults unchanged); the unused `CLIPProcessor` load is wrapped so a processor-loading difference in a newer `transformers` cannot stop the run.

## What was done

- Patches as commits: `same_on_batch=True` in EoT, `loss_fn` injection in `PGDOptimizer`, windowed-SSIM structure loss subtracted in `forward` (see `git log`; tag `v1.0`).
- Layout: `src/{utils,protect,eval,baselines}`; v1 code kept at its import paths; `src/eval.py` moved to `src/legacy_eval.py` with a lazy compat shim in `src/eval/__init__.py`.
- `config/config.yaml` (+ `config/parts/*.yaml` merge), run logging (resolved config, git hash or bundle info, library versions, environment, seed written into every run folder), per-image seeds.
- Scripts: `run_protect.py` (folder in -> folder out, checkpoint/resume, `--shard`, GPU-time log), `merge_shards.py`, `budget_check.py`, `prepare_v1_data.py`, `make_tables.py`, `compare_runs.py`, `check_env.py`, `gpu_ledger.py`, `env_probe.py`, `merge_requirements.py`, `run_job.py`.
- Kaggle: `kaggle/build_bundle.py` (single code zip), `kaggle/job_runner.ipynb` (set `JOB_IDS` only), `kaggle/jobs.yaml` (J00 environment probe, J01 and J01b reproduction), `kaggle/README.md`.
- CPU tests under `tests/` (no model weights downloaded; also run inside J01 on Kaggle before the job: 32 passed): config, budget conversions, seeding, run log, v1 patches (markers, paired EoT, SSIM loss, loss sign, injection), tiny-model PGD, metrics against an oracle copy of the v1 function, bootstrap statistics, budget check, J01 pipeline end to end with tiny models (including sharding, merge and resume), import isolation of Half A and Half B. Result: 32 passed (see the final reply for the exact run).

## GPU time used

| Job | GPUs | session time | GPU-seconds | per image |
|---|---|---|---|---|
| J00 env probe | 0 (ran inside the GPU session) | 10 s | 0 | – |
| J01 smoke (5 images) | – | not run separately | – | – |
| J01 full (70 images, 40 iters) | 2xT4 | 1756 s (29.3 min; protect step 1680 s) | 3189 (0.89 GPU-h) | 45.6 ± 1.2 s; peak memory 7.2 GB |

## Decisions taken

- `methods.v1` budget: eps 4/255 and step 0.5/255 on the standard scale (equal to 8/255 and 1/255 on the tensors), enforced by the v1 adapter.
- KMeans re-selection was not re-implemented: v1's manifests (`data/v1_ref/`) give the exact 70 and 4 file names (`data/splits/v1_eval70.txt`, `v1_calib.txt`).
- Metrics are reported in two conventions: v1 (tensor -> uint8 by truncation) and published PNG (rounded).
- `requirements.txt` pinned (`==`) to the J00 freeze of the Kaggle image (insightface 0.7.3 and onnxruntime-gpu are not in the image; onnxruntime-gpu stays unpinned until S05 shows a build that loads with CUDA 12.8).
- Measured v1 cost 45.6 GPU-s/image (1.14 s per PGD iteration) vs the plan's estimate 25 s (0.55 s/iteration): 1.8x. This alone does not trigger fallback rule 3 (that uses the S04/S05 smoke timings of the v2 methods), but S08/S15, which re-protect v1, will take longer than planned.
- Fixed `scripts/gpu_ledger.py`: it counted the merged `gpu_time.json` plus both shard files, i.e. J01 twice (6377 s instead of 3189 s).

- **Step size (integration):** `methods.v1` keeps v1's true step 0.5/255 (standard scale). Every v2 preset (`v2_cheap`, `v2_main`, `v2_antipure`, `v2_final_default`, `abl_*`) sets `budget.step_255: 1.0` explicitly, as in the handoff; `eps2/6/8` inherit it. The v2 loss weights start from v1's exact Optuna values (alpha 0.2699475469647097, beta 0.9023875567667267, gamma 60.52658429696642; eta = alpha) instead of the rounded 0.27/0.90/60.5.
- **v2 method presets** are resolved inside `src/protect/v2_registry.py`; `run_protect` writes the resolved preset to `resolved_method_cfg*.json` and logs its hash and budget in the CSV.

## Open questions for the user

1. ~~Accept the proposed reproduction tolerance?~~ Accepted by the user on 2026-10-14 (±0.5 dB PSNR, ±0.005 SSIM, ±0.03 LPIPS).
2. Enable "Pin to original environment" in the Kaggle notebooks so the image cannot drift during the 48 h? (`requirements.txt` is pinned; `check_env.py` flags drift but only stops the job for torch/torchvision/diffusers/transformers/kornia.)

## Artifacts

`results/<run_id>/{config.json, git_hash.txt, versions.json, env.json, seed.txt, command*.txt, metrics.csv, summary.json, gpu_time.json, budget_check.json, compare_v1.json, tables/}`. Images stay on Kaggle / local disk, never in git.

Checkpoint: **Phase 0 STOP gate passed** (budget 4/255 confirmed; v1 invisibility reproduced within the accepted tolerance, n = 70). Phase 0 is closed.
