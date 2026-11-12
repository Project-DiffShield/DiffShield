# Step 1 plan: representative tuning photos and a joint re-tune (alpha, beta, gamma, eta, DiffPure-style cleaner, denoiser range)

Branch `v2/tuning` (worktree `C:\Users\karth\Documents\DiffShield-wt\tuning`, cut from `v2/release` 53adae2).
Deadline for the whole project: Sunday 2026-11-15 12:00. Binding rules: never invent numbers, mark unverified
things, no face images in commits or deliverables, no dataset redistribution, no "Co-Authored-By: Claude" trailer.

Local test interpreter: `python` (3.13, torch 2.14 CPU, numpy). Run `python -m pytest -q tests/<file>` from the worktree.

## 0. Why

* Current final method `v2_tgt_zero_mid`: alpha 0.55 was one of 3 declared values; beta 0.9024 and gamma 60.53 are
  inherited from the prototype's Optuna run on 4 photos. Weak justification.
* DiffPure is only in evaluation; the one-step SD purifier (`SDOneStepPurifier`, `sd_onestep`) exists but is unused
  by the final method.
* The denoiser term L_den (`eta`) is 0 in the final method, and its timestep range is tied to `sd_onestep`'s
  t_min/t_max (50-300) while strength-0.55 editing noises to about t = 550.
* Old tuning photos (20 val) were picked from a small pool; we want a defensible "representative of the dataset" set.

## 1. Representative photo selection (Part A)

### 1.1 Pool
All 30,000 CelebAMask-HQ images MINUS test (100), messy (20), ae_pool (120), v1_calib (4) (v1_eval70 is inside test),
MINUS images with no ArcFace face, MINUS any image whose ArcFace cosine >= 0.5 to any test / messy / ae_pool / v1_calib
image (same-person leakage gate; threshold 0.5 is the project's existing UNVERIFIED value). Old val/train are allowed.

### 1.2 Features
* Selection space: CLIP ViT-L/14 image embeddings (`models.clip_eval`, `openai/clip-vit-large-patch14`), L2-normalised.
* Independent check space (NOT used for selection): the 40 binary attributes (standardised) + pose (yaw, pitch, roll,
  standardised) from `CelebAMask-HQ-attribute-anno.txt` and `CelebAMask-HQ-pose-anno.txt`.
* Identity: ArcFace buffalo_l embeddings (same pack as evaluation), for the leakage gate and for one-photo-per-identity
  inside the selected sets.

### 1.3 Algorithm: MMD-critic (Kim, Khanna, Koyejo, NeurIPS 2016)
* Kernel: RBF k(a,b) = exp(-||a-b||^2 / (2 s^2)), s from the median heuristic on 5,000 random pool images (seed 2026).
* Prototypes = greedy maximisation of the MMD-critic objective
  J(S) = (2/(n|S|)) sum_{i in S, j in P} k(i,j) - (1/|S|^2) sum_{i,j in S} k(i,j)
  (equivalently greedy minimisation of MMD^2(S, P)); colmeans mean_j k(i,j) are computed once in chunks.
  A candidate is skipped if its ArcFace cosine >= 0.5 to an already selected photo.
* Criticisms = greedy maximisation of |witness(x)| = |mean_j k(x,j) - mean_{s in S} k(x,s)| with the paper's log-det
  diversity regulariser, over pool images not in S, identity-gated against everything selected.
* Sets (defaults, configurable):
  * `tune_search` = 28 prototypes on P.
  * `tune_confirm` = 8 prototypes from an independent greedy run on P minus the search identities, plus 4 criticisms of
    the search set (the faces the search set represents worst). 12 photos.
* Deterministic given the inputs (no randomness except the bandwidth subsample and the random baselines, seed 2026).

### 1.4 Evidence (written to `selection_report.json` and `selection_report.md`, numbers only, no images)
For each of `tune_search`, `tune_confirm` (prototype part and whole), and the old `val` list for context:
1. MMD^2 to P in CLIP space vs 1,000 random same-size subsets of P: our value, random mean / 5th / 50th / 95th
   percentile, rank-based p = (1 + #{random <= ours}) / 1001. (Expected to be extreme: it is the optimised space.)
2. The same in the independent attribute+pose space (the fair check; median-heuristic bandwidth there).
3. Coverage: mean and max over P of the CLIP cosine distance to the nearest selected photo, vs random.
4. Attribute marginals: pool share vs set share for all 40 attributes; pose mean/std vs pool.
Commit the lists + meta (sha256) + report BEFORE any tuning trial (pre-registration).

### 1.5 Code
* `scripts/embed_pool.py` (Kaggle GPU): reads the CelebA-HQ image dir (same auto-discovery as `make_splits.py`
  `find_celebahq`), writes `pool_embeddings.npz` = names, clip [N,768] float16, arcface [N,512] float16 (NaN row if no
  face), has_face [N] bool, attr [N,40] int8, attr_names, pose [N,3] float32 (NaN if missing). Batched, resumable in
  chunks (`--chunk 2000`, partial files merged), `--limit N` for smoke. Uses insightface buffalo_l the same way as
  `src/eval/faces.py` / `make_splits.make_face_cache` (reuse, do not reimplement scoring rules).
* `scripts/select_tuning_photos.py` (CPU, local or Kaggle): reads the npz + split lists, builds P, runs MMD-critic,
  writes `data/splits/tune_search.txt`, `data/splits/tune_confirm.txt` (CelebA-HQ names, e.g. `1234.jpg`, one per line,
  same convention as `val.txt`), `data/splits/tune_meta.json` (args, pool size, exclusions counts, bandwidths, order of
  selection, which confirm photos are criticisms, sha256 of both lists, npz sha256), and the reports into `--report_dir`.
  Refuses to overwrite existing lists without `--force`.
* `tests/test_select_tuning.py`: synthetic clusters -> prototypes hit every cluster; exclusions and identity gate
  respected; criticisms come from the under-covered cluster; determinism; random-baseline stats shape; refuses overwrite.

## 2. Tuning code (Part B)

### 2.1 Decoupled denoiser timestep range
* New config key `loss.den_t_range: [lo, hi]` (default null = old behaviour: use `sd_onestep` t_min/t_max).
* `PGDParams.t_range` comes from `loss.den_t_range` when set.
* `l_den` reuses the one-step purifier's aux (eps, noise) ONLY when the aux timestep lies inside the denoiser range;
  otherwise it does its own forward with its own t in `t_range` on the purified image. Simplest correct rule: reuse
  aux only if `den_t_range` is null or equal to the sd_onestep range; else always fresh. Log the reuse count in info.
* Tests in `tests/test_protect_pgd.py` style (tiny fake models via existing fakes): fresh pass used when ranges differ,
  aux reused when equal, t sampled within range.

### 2.2 New base preset `v2_next` (config/parts/protect.yaml; do NOT edit existing presets)
Target-zero L_vis like `v2_tgt_zero_mid`, `budget.step_255 1.0`, `eot.k_draws 4`, ensemble off, antipure off,
`purifiers_train [jpeg, downup, vae_roundtrip, sd_onestep]`, `purifier_train_probs` derived from a single knob
`p_onestep` (jpeg : downup : vae = 0.35 : 0.35 : 0.30 scaled to 1 - p_onestep). Default values = current final
(alpha 0.55, beta 0.9023875567667267, gamma 60.52658429696642, eta 0, p_onestep 0.0, den_t_range null) so that
`v2_next` with defaults reproduces `v2_tgt_zero_mid` exactly (test this: resolved loss/purifier settings equal,
sd_onestep prob 0). Register it wherever `v2_registry.PRESETS` / `_CHAIN` need it.

### 2.3 `scripts/tune_optuna.py` (keep old behaviour as the default; new behaviour behind flags)
* `--base-method NAME` (default `v2_final`): the preset built and overridden (override goes into `methods.<NAME>`).
* `--image-list PATH` (alias of `--val-list`); `--n-images 0` = all names in the list.
* `--space {v1final, v2next}` (default `v1final` = current `suggest`). `v2next`:
  * alpha log-uniform [0.1, 2.0]; beta uniform [0, 1.5]; gamma log-uniform [20, 150];
  * `den_on` categorical {False, True}; if True eta log-uniform [0.01, 1.0] and `den_t` categorical
    {"50-300", "300-700"}; else eta 0;
  * `p_onestep` categorical {0.0, 0.15, 0.3}; antipure off; lambda_id 0.
  * `params_to_override` maps these to loss.*, purifier_train_probs (via p_onestep), loss.den_t_range.
  * Anchor trial enqueued first = the current final (alpha 0.55, beta 0.9024, gamma 60.53, den_on False, p_onestep 0).
* `--constraint {mean, per_photo}` (default `mean`). `per_photo`: constraints = [38 - min PSNR, 0.95 - min SSIM] over
  the photos of the trial (every photo must pass). Record mean and min in user attrs as today.
* `--purifiers` / `--editors` / `--edit-seeds` CLI overrides of DEFAULT_SPEC (editors as `name` or `name:strength`).
* Confirm stage: `--confirm-list PATH` (photos for confirmation; default = same list as before), uses per-photo
  constraint when `--constraint per_photo`, writes per-photo PSNR/SSIM into confirm.json, and applies the pre-registered
  decision rule (section 3) writing `decision.json` + `final_candidate.yaml` under `methods.v2_next_tuned`.
* `tests/test_protect_tune.py`: extend with the stub scorer (`--stub-score`) and `PROTECTOR_COMPONENTS` fakes: v2next
  space produces valid overrides, per-photo constraint math, anchor enqueued, decision rule picks/rejects correctly.

## 3. Tuning protocol (pre-registered; goes in `docs/TUNING_PROTOCOL.md` and is committed before launch)
* Search photos: `tune_search` (28). Confirm photos: `tune_confirm` (12). Test photos never used.
* Search objective: mean PSR (threshold 0.1005, the existing 95th-percentile impostor value) over editors
  {sd15_img2img s=0.55, ipadapter_faceid} x cleanups {none, jpeg_q75, downup_0.5, vae_rt_sd15, diffpure_t0.2}, edit seed 1.
  Proxy protection: 20 iterations, K = 2.
* Held out from both training and tuning (reported at test only): jpeg_q_other, downup_other, restore_swin2sr,
  adaptive AE. DiffPure is NO LONGER held out (one-step version in training, diffpure_t0.2 in the objective); reports
  must say so.
* Constraint: every photo PSNR >= 38 dB and SSIM >= 0.95 (per-photo, not mean).
* Sampler: Optuna TPE, seed 2026, 8 random startup trials, constant liar, budget 30 trials (anchor = trial 0).
* Confirm: top 3 feasible trials + anchor, full settings (40 iterations, K = 4), on `tune_confirm`, edit seeds 1-3.
* Decision rule: the best confirmed candidate replaces the current final method only if (a) zero per-photo
  invisibility violations on the confirm photos and (b) its confirm PSR >= the anchor's confirm PSR. Otherwise the
  current final method stays and the report says the re-tune did not beat it.
* Then one run of the winner on the 100 test photos with the full Half B (separate step, not in Step 1).

## 4. Kaggle jobs (kaggle/jobs.yaml; follow the existing schema and macros exactly)
* `T01` embed + select: accelerator gpu_t4 (one GPU is enough), CelebAMask-HQ input, runs `embed_pool.py` then
  `select_tuning_photos.py`; outputs npz (derived features, keep private), lists, meta, reports. smoke_params: `--limit 600`
  with smaller set sizes. Estimated time: unknown until smoke (planner guess 30-60 min, UNVERIFIED).
* `T02` search: gpu_t4x2, `workers: true`, resumable via RESUME_FROM like the existing tune job, prep macro for the
  `tune_search` split, the section-3 flags, `--psr-threshold 0.1005`, `--base-method v2_next --space v2next
  --constraint per_photo`. smoke_params: 2 trials, 2 photos, 3 iterations.
* `T03` confirm: gpu_t4x2, prep for `tune_confirm`, restores the study from RESUME_FROM, `--confirm-top 3`.
* `config/parts/data.yaml`: add `splits_files.tune_search` / `tune_confirm` / `tune_meta`. `scripts/prepare_data.py`
  must accept these split names (add to SPLITS or accept any listed split; no messy-style degradation for them).
* `python scripts/run_job.py --jobs T01 T02 T03 --dry_run --work tmp` must expand cleanly (smoke and full);
  `tests/test_jobs_expand_nonempty.py` must pass.
* Rebuild the bundle with `python kaggle/build_bundle.py` only after the lists are committed.

## 5. Execution order and owners
1. Agent A (Sonnet): Part A code + tests (files: `scripts/embed_pool.py`, `scripts/select_tuning_photos.py`,
   `tests/test_select_tuning.py`). Touches nothing else.
2. Agent B (Sonnet), in parallel: Part B code + tests (files: `src/protect/pgd.py`, `src/protect/losses.py`,
   `src/protect/v2_registry.py`, `config/parts/protect.yaml`, `scripts/tune_optuna.py`, `tests/test_protect_*.py`).
3. Agent C (Sonnet), after A and B: jobs.yaml T01-T03, data.yaml, prepare_data.py split names, `docs/TUNING_PROTOCOL.md`
   (section 3 verbatim, plus the selection method of section 1), full `python -m pytest -q`, dry runs.
4. Reviewer (Opus, coordinator): review diffs, run tests, commit on `v2/tuning` (no Claude trailer), build bundle.
5. User: Kaggle T01 (smoke then full) -> coordinator checks report, commits lists -> rebuild bundle -> T02 smoke ->
   T02 full on as many accounts/GPUs as available -> T03.
