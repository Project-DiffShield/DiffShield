# Tuning protocol (pre-registered)

Status: written and committed BEFORE the tuning photo lists exist and before any trial runs. Anything marked UNVERIFIED is an
assumption, not a measurement. No numbers in this file are results.

Scope: a joint re-tune of the v2 protection method (loss weights alpha, beta, gamma, the denoiser term eta and its timestep
range, and the probability of the one-step DiffPure-style purifier in training). Base preset `v2_next` (its defaults reproduce the
current final method `v2_tgt_zero_mid`). Test photos are never used.

## 1. Which photos are used for tuning, and how they are chosen

### 1.1 Candidate pool
All 30,000 CelebAMask-HQ images, minus: the test split (100), the messy split (20), the adaptive-attacker pool (120), the v1
calibration photos (4) (the 70 v1 evaluation photos are inside test); minus images where ArcFace finds no face; minus any image
whose ArcFace cosine similarity is >= 0.5 to any test / messy / ae_pool / v1_calib image (same-person leakage gate; the 0.5
threshold is the project's existing value and is UNVERIFIED). Old val and train photos are allowed in the pool.

### 1.2 Features
* Selection space: CLIP ViT-L/14 image embeddings (`openai/clip-vit-large-patch14`, the model used by the evaluation), L2-normalised.
* Independent check space (NOT used for selection): the 40 binary CelebAMask-HQ attributes (standardised) plus head pose
  (yaw, pitch, roll; standardised) from the dataset annotation files.
* Identity: ArcFace buffalo_l embeddings (same pack as the evaluation), for the leakage gate and to keep one photo per identity
  within the selected sets.

### 1.3 Algorithm: MMD-critic (Kim, Khanna and Koyejo, NeurIPS 2016)
* RBF kernel k(a,b) = exp(-||a-b||^2 / (2 s^2)); s from the median heuristic on 5,000 random pool images (seed 2026).
* Prototypes: greedy maximisation of the MMD-critic objective
  J(S) = (2/(n|S|)) sum_{i in S, j in P} k(i,j) - (1/|S|^2) sum_{i,j in S} k(i,j),
  equivalent to greedy minimisation of MMD^2(S, P). A candidate is skipped if its ArcFace cosine is >= 0.5 to an already selected photo.
* Criticisms: greedy maximisation of the absolute witness function |mean_j k(x,j) - mean_{s in S} k(x,s)| with the paper's
  log-det diversity regulariser, over pool images not in S and identity-gated against everything selected; these are the faces
  the search set represents worst.
* Sets (defaults, configurable; the actual values are recorded in `tune_meta.json`):
  * `tune_search`: 28 prototypes of the pool.
  * `tune_confirm`: 8 prototypes from an independent greedy run on the pool minus the search identities, plus 4 criticisms of
    the search set (12 photos).
* Deterministic given the inputs; the only randomness (bandwidth subsample, random baselines) uses seed 2026.

### 1.4 Evidence reported with the lists (`selection_report.json` / `.md`; numbers only, no images)
For `tune_search`, `tune_confirm` (prototype part and whole) and, for context, the old `val` list:
1. MMD^2 to the pool in CLIP space against 1,000 random same-size subsets: our value, random mean / 5th / 50th / 95th percentile,
   rank-based p = (1 + #{random <= ours}) / 1001. This is expected to be extreme because CLIP space is the optimised space.
2. The same in the independent attribute + pose space (the fair check; median-heuristic bandwidth there).
3. Coverage: mean and max over the pool of the CLIP cosine distance to the nearest selected photo, against random subsets.
4. Attribute marginals (pool share against set share for all 40 attributes) and pose mean / std against the pool.

The lists, `tune_meta.json` (arguments, pool size, exclusion counts, bandwidths, selection order, which confirm photos are
criticisms, sha256 of both lists and of the embedding file) and the report are committed before any tuning trial.

## 2. Tuning protocol
* Search photos: `tune_search` (28). Confirm photos: `tune_confirm` (12). Test photos never used.
* Search objective: mean PSR (threshold 0.1005, the existing 95th-percentile impostor value) over editors
  {sd15_img2img at strength 0.55, ipadapter_faceid} x cleanups {none, jpeg_q75, downup_0.5, vae_rt_sd15, diffpure_t0.2}, edit seed 1.
  Proxy protection during the search: 20 iterations, K = 2.
* Search space (`--space v2next`): alpha log-uniform [0.1, 2.0]; beta uniform [0, 1.5]; gamma log-uniform [20, 150]; denoiser term`r`n  off, or on with eta log-uniform [0.01, 100] (wide on purpose: the size of L_den next to L_vis is not measured, UNVERIFIED) and timestep range {50-300, 0-600} (0-600 = the noise levels a strength-0.55 edit visits); one-step purifier probability in
  {0.0, 0.15, 0.3}; anti-purify off; identity weight 0. Anchor trial 0 = the current final method (alpha 0.55, beta 0.9024,
  gamma 60.53, denoiser off, one-step probability 0).
* Held out from both training and tuning (reported at test only): jpeg_q_other, downup_other, restore_swin2sr, adaptive AE.
  DiffPure is NO LONGER held out (a one-step version may be used in training and diffpure_t0.2 is in the objective); reports must say so.
* Constraint: every photo PSNR >= 38 dB and SSIM >= 0.95 (per photo, not mean).
* Sampler: Optuna TPE, seed 2026, 8 random startup trials, constant liar, budget 30 trials (anchor = trial 0).
* Confirm: the top 3 feasible non-anchor trials, plus the best feasible trial that has BOTH the one-step cleaner (probability > 0) and the denoiser term on (if it is not already among them), plus the anchor; full settings (40 iterations, K = 4), on `tune_confirm`, edit seeds 1-3. The extra candidate gives a direct measured answer to "did DiffPure-in-training and the denoiser attack help?".
* Decision rule: the best confirmed candidate replaces the current final method only if (a) there are zero per-photo
  invisibility violations on the confirm photos and (b) its confirm PSR >= the anchor's confirm PSR. Otherwise the current final
  method stays and the report says the re-tune did not beat it.
* Afterwards, one run of the winner on the 100 test photos with the full evaluation (separate step, not part of the tuning). Final adoption rule, written before any trial: the winner is adopted as the final method only if its test PSR beats the current method's on the same 100 photos with a paired bootstrap 95% interval whose lower bound is above 0; otherwise the current method stays, and the re-tune is reported as tried and not adopted. (The confirm set has only 12 photos, so the confirm rule alone would accept a chance winner about half the time when two methods are truly equal.)

## 3. Planned jobs and time estimates (UNVERIFIED)
T01 (embed + select, one T4; planner guess 30-60 min), T02 (search, 2 x T4), T03 (confirm, 2 x T4). All times are planner guesses
until a smoke run has been timed; none has been measured yet.

## 4. What we will and will not claim
* The tuning photos are a documented, dataset-representative sample chosen by a stated algorithm with a stated seed, with the
  evidence in section 1.4. We do not claim they are representative of arbitrary real-world faces.
* MMD in CLIP space is the quantity that was optimised, so a small value there is not independent evidence. The attribute + pose
  MMD and the coverage statistics are the fair checks; we report all of them and say which is which.
* DiffPure is no longer a held-out purifier: it enters the training mix (one-step version) and the search objective
  (diffpure_t0.2). Any DiffPure robustness claim therefore refers to a seen purifier. Unseen-purifier claims rest on
  jpeg_q_other, downup_other, restore_swin2sr and the adaptive AE only.
* The ArcFace same-person threshold of 0.5 (leakage gate and within-set identity de-duplication) is UNVERIFIED; it may let a few
  same-person pairs through or reject some different people.
* The PSR threshold 0.1005 is inherited from the existing calibration; it is not re-checked here.
* The tuned candidate is adopted only under the pre-registered decision rule. If it does not beat the anchor, we report exactly that.
* With 28 search photos and 30 trials the search is small; the confirm set guards against overfitting to the search photos, and the
  final test run is the only unbiased estimate. We make no claim beyond what the test run shows.
* No face images or embeddings are published; only file-name lists and numeric summaries.
