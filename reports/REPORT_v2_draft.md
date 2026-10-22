# DiffShield v2: protecting face photos from diffusion-model edits (draft)

Status: FINAL DRAFT 2026-10-22 (all planned sessions that were run have been analysed; the experiments marked "not run" were cut for time). Every number below is copied from the generated tables (`results/tables/`, `results/tables/final_readout.md`) or from the run logs named in brackets; nothing is estimated. Items in **[PENDING ...]** are experiments still running or not yet analysed. Face images are CelebA-HQ derivatives, used for research only; none are published here.

## 1. Summary (fill the last bullets after S26-S28)

- **Goal.** Add an almost invisible perturbation to a face photo so that edits of the protected photo by a diffusion editor no longer look like the same person, and the protection survives cleanup ("purification") of the photo.
- **What was wrong with v1** (the starting system): "8/255" was a measurement on [-1,1] tensors, i.e. 4/255 on the normal scale (confirmed on real PNGs: maximum change exactly 4 gray levels in 70 of 70 photos); no noise control; no identity metric; no test against cleanups; one editor, one prompt, one seed, 15 images; runtime string patches.
- **What we built.** A reproducible protection and evaluation pipeline (one config, per-image seeds, logged environment; 377 CPU tests), an evaluation on 100 held-out photos (+20 harder "messy" photos) with 5 editors, 13 cleanups, 3 seeds, ArcFace identity scoring, paired bootstrap confidence intervals, noise controls, PhotoGuard and Mist baselines, an epsilon sweep, ablations and an adaptive-attacker test.
- **Measured results (SD1.5 img2img, strength 0.55, 100 test photos; PSR = share of edits with no detected face or ArcFace similarity below the 95th percentile of impostor similarity, threshold 0.101):**
  - Clean photos: PSR 13.9%. Uniform noise at the same budget: 14.9% (+1.0 points over clean, CI [-1.0, +3.1]); Gaussian noise matched to v1's PSNR: 14.7%. **Noise alone does nothing.**
  - **v1: 24.0%** (+10.1 points over clean, [+7.2, +13.0]). v2_main: 24.3%, no better than v1 (+0.3 [-1.8, +2.6]).
  - The encoder-shift term accounts for the whole effect (ablation `vis_only` 25.3%); the denoiser-only term gives no protection (12.3%) and the anti-purification terms give small, borderline gains at a visibility cost.
  - The effect survives held-out DiffPure (net +10 to +16 points over clean photos at t* = 0.05-0.3) and is reduced by JPEG and resizing (v1: net +4.7 under JPEG 75, +0.0 under down-up 0.5; v2_main restores +5.7 under down-up).
  - **v2_final** (tuned, antipure on): 29.4%, +5.4 points over v1 [+2.4, +8.6], but it fails the invisibility rule on 29 of 100 photos (SSIM below 0.95).
  - **Target-steered variants** (new; sections 3 and 5.8), 100 test photos: `v2_tgt_zero_mid` reaches **PSR 49.7%** (net +35.8 points over clean [+31.4, +40.4]; **+25.7 over v1 [+21.8, +29.8]**) with **0 of 100 photos violating the invisibility rule** (PSNR 40.91, min 39.75; SSIM 0.9728, min 0.9615). `v2_tgt_zero` (weaker): 41.8%, +17.8 over v1 [+14.1, +21.7], 0 violations, PSNR 42.16. `v2_tgt_zero_hi` (strongest): 52.8%, +28.8 over v1 [+23.8, +33.8], 4 of 100 photos below SSIM 0.95. The gain survives cleanups: net over clean photos +24 to +48 points under the VAE round trip and DiffPure 0.1-0.3; weaker under JPEG (net +5 to +16) and down-up (+5 to +9). The weights were not tuned on test, but which variant is called "best" is read off the test table (post hoc, among three declared variants; all three are reported).
  - **Existing tools** (same photos, budget 4/255): PhotoGuard 81.0% and Mist 63.1% PSR, but both break the invisibility rule on every photo (PSNR 37.2 / 38.0 dB, SSIM 0.906 / 0.923); under JPEG and down-up cleanups `v2_tgt_zero_mid` matches or beats them at invisible levels (section 5.8).
  - **An adaptive attacker removes the protection** (a weak autoencoder purifier trained on 120 other photos: v1 24.0% -> 11.0%, v2_final 29.4% -> 10.7%, clean 9.3%).
  - **No protection** against the identity-conditioned editor (IP-Adapter FaceID, 0.0-0.3%), the inpainting editor (0%), SDXL (about 2%) or at strength 0.3 (0%); at strength 0.75 clean edits already destroy identity (84%).

## 2. Threat model (state this on page one)

- **Defender.** Owns the photo, runs the protection offline with white-box access to the Stable Diffusion 1.5 VAE (and, for some variants, the UNet). Budget: L-infinity at most 4/255 on the 0-255 scale unless stated (sweep 2, 4, 6, 8).
- **Attacker (non-adaptive, headline).** Uses one of five editors (SD1.5 img2img at strengths 0.3, 0.55, 0.75; SDXL img2img; InstructPix2Pix; SD inpainting with a mask that keeps the face; IP-Adapter FaceID), optionally after one of 13 cleanups (JPEG q90/75/50 and another JPEG setting, down-up 0.5 and another ratio, VAE round trip, SD-based DiffPure-style at t* = 0.05/0.1/0.2/0.3, Swin2SR restoration). The attacker does not know the protection method.
- **Attacker (adaptive, stress test).** Trains a small autoencoder on clean/protected pairs of other photos and uses it as the cleanup.
- **Out of scope / not claimed.** Editors and cleanups not listed; GrIDPure (model file unavailable) and IMPRESS (not run); InstantID; human perception of identity (no human study was run). If an attacker masks the face itself in inpainting, the masked pixels are discarded and noise there cannot protect them.

## 3. Method

v1: PGD for 40 iterations with EoT over random JPEG and crops; loss = alpha L_vis - beta L_sem - gamma L_str (L_vis: distance of the VAE latent from the clean latent, maximised; L_sem: CLIP "a potted plant" term; L_str: 1 - SSIM, keeps the photo unchanged).

v2 (Half A, protection): the same PGD, but each step draws K = 4 cleanups from a training mixture (JPEG, down-up, VAE round trip, one-step SD purifier), scores the cleaned photo, and keeps the visibility penalty on the published photo. Optional terms: denoiser error `L_den`; anti-purification (patch-DCT high frequency and error-grows-with-t); SDXL-VAE ensemble; identity loss `L_id` (implemented, switched off: lambda_id = 0 in every preset, never evaluated). Presets: `v2_cheap`, `v2_main`, `v2_antipure`, `v2_final` (Optuna-tuned), ablations, `eps2/6/8`.

**Target-steered variants** (added 2026-10-20): the L_vis term only pushes the latent away from the clean latent, with no direction. The new term pulls the latent of the cleaned photo toward a fixed destructive target (PhotoGuard's zero latent, or a gray image), inside the same cleanup loop: `v2_tgt_zero` (alpha 0.27), `v2_tgt_zero_mid` (0.55), `v2_tgt_zero_hi` (1.0), `v2_tgt_gray`.

Half B (evaluation): held-out cleanups and editors; ArcFace (insightface buffalo_l) identity similarity; PSR; edit-LPIPS (headline setting only); CLIP-I; invisibility metrics. Half A and Half B share no components.

## 4. Experimental setup

- Data: CelebA-HQ (Kaggle `ipythonx/celebamaskhq`); frozen lists in `data/splits`: test 100 (all 70 of v1's evaluation photos + 30 new), messy 20 (pose, blur, lighting, occlusion, pre-compression), val 20 (all tuning and screening), train 20, adaptive-attacker pool 120. v1's calibration photos are excluded. The 20 validation photos were used for Optuna and for screening the target-steered variants; the test photos were never used for tuning.
- Edits: DPM-Solver++ 30 steps, 3 seeds, prompts rotated per photo (all 3 prompts for the headline setting), saved as JPEG q90.
- Metrics and statistics: as in `reports/PHASE_0.md` and `docs/EVAL_PLAN_FINAL.md` section B. Unit of analysis = the photo; 95% bootstrap CI over photos (2,000 resamples); all comparisons paired on the same photos.
- Compute: Kaggle T4 x2; all runs are logged with config, seed, git hash and library versions.

## 5. Results (test photos unless stated)

### 5.1 Phase 0: v1 reproduced and the budget corrected
n = 70: PSNR 41.986 +- 0.416 [41.890, 42.082], SSIM 0.9754 +- 0.0054, LPIPS 0.2314 +- 0.0637; v1's own stored numbers 41.911, 0.9750, 0.2316; paired differences +0.075 dB, +0.0004, -0.0002, all inside the accepted tolerance (+-0.5 dB, +-0.005, +-0.03). Max pixel change 4 gray levels = 4/255. A second-person reproduction was not run.

### 5.2 Invisibility (published PNG; rule: PSNR >= 38 dB and SSIM >= 0.95 on every photo)

| method | n | PSNR mean (min) | SSIM mean (min) | photos < 38 dB | photos SSIM < 0.95 |
|---|---|---|---|---|---|
| v1 | 100 | 42.42 (41.17) | 0.9763 (0.9587) | 0 | 0 |
| v2_main | 100 | 41.72 (40.34) | 0.9784 (0.9707) | 0 | 0 |
| v2_cheap | 100 | 41.58 (40.28) | 0.9790 (0.9694) | 0 | 0 |
| v2_antipure | 100 | 41.18 (38.91) | 0.9700 (0.9235) | 0 | 13 |
| **v2_final** | 100 | 40.46 (38.76) | 0.9549 (0.9197) | 0 | **29** |
| v2_tgt_zero | 100 | 42.16 (40.58) | 0.9822 (0.9728) | 0 | 0 |
| v2_tgt_zero_mid | 100 | 40.91 (39.75) | 0.9728 (0.9615) | 0 | 0 |
| v2_tgt_zero_hi | 100 | 40.05 (39.14) | 0.9628 (0.9459) | 0 | 4 |
| eps2 | 100 | 45.34 (44.86) | 0.9836 (0.9628) | 0 | 0 |
| PhotoGuard (official 1000 it., our eps 4/255) | 100 | 37.15 (36.83) | 0.9061 (0.8123) | **100** | **100** |
| Mist (official 100 it., our eps 4/255) | 100 | 38.00 (37.47) | 0.9229 (0.8275) | **50** | **100** |
| **eps8** | 100 | 35.98 (33.33) | 0.8958 (0.7724) | **98** | **96** |
| uniform noise | 100 | 40.80 (40.72) | 0.9518 (0.9257) | 0 | 43 |
| Gaussian noise (PSNR-matched) | 100 | 41.91 (41.89) | 0.9623 (0.9337) | 0 | 3 |
| abl_vis_only | 100 | 41.73 (40.32) | 0.9781 (0.9697) | 0 | 0 |
| abl_den_only | 100 | 50.76 (48.79) | 0.9951 (0.9900) | 0 | 0 |

The Optuna search checked the PSNR and SSIM limits on the average over 20 photos, not per photo; that is why `v2_final` slipped through with per-photo violations. PhotoGuard and Mist change at most 4 gray levels like everything else, but they use the whole budget on nearly every pixel and have no structure penalty, so they visibly lower PSNR and SSIM.

### 5.3 Noise control (E1)
PSR 13.9% (clean), 14.9% (uniform noise, +1.0 [-1.0, +3.1]), 14.7% (Gaussian, +0.8 [-1.2, +2.9]), 24.0% (v1). v1 minus uniform noise: +9.1 points [+6.4, +12.0]; v2_main minus Gaussian noise: +9.7 [+6.8, +12.3]. The effect is not just noise.

### 5.4 Protection under cleanup (E3, E4, E5)
Net protection = PSR of the method minus PSR of clean photos under the same cleanup (paired, 100 photos). Full table: `results/tables/final_readout.md`.

| cleanup | clean PSR | v1 net | v2_main net | v2_cheap net | v2_final net | tgt_zero net | tgt_zero_mid net | tgt_zero_hi net |
|---|---|---|---|---|---|---|---|---|
| none | 0.139 | +0.101* | +0.104* | +0.086* | +0.156* | +0.279* | +0.358* | +0.389* |
| JPEG q75 | 0.103 | +0.047* | +0.080* | +0.063* | +0.060* | +0.107* | +0.130* | +0.163* |
| JPEG q50 | 0.130 | +0.037* | +0.050* | +0.043* | +0.023 | +0.050* | +0.073* | +0.083* |
| down-up 0.5 | 0.097 | +0.000 | +0.057* | +0.047* | +0.047* | +0.047* | +0.093* | +0.093* |
| VAE round trip | 0.123 | +0.097* | +0.127* | +0.090* | +0.150* | +0.310* | +0.400* | +0.410* |
| DiffPure 0.1 | 0.133 | +0.130* | +0.147* | +0.133* | +0.163* | +0.310* | +0.383* | +0.477* |
| DiffPure 0.2 | 0.217 | +0.100* | +0.117* | +0.097* | +0.133* | +0.260* | +0.353* | +0.383* |
| DiffPure 0.3 | 0.260 | +0.100* | +0.103* | +0.087* | +0.153* | +0.237* | +0.340* | +0.403* |

(* = 95% CI excludes 0.) Reading: the effect survives DiffPure at every strength tried; JPEG and resizing shrink it; v1 loses it entirely under down-up 0.5 and the cleanup-aware variants keep +4.7 to +5.7 points there. Paired v2_main minus v1 is supported only under down-up 0.5 (+5.7 [+3.0, +8.7]) and down-up (other) (+3.0 [+0.3, +6.3]). v2_final minus v1: +5.4 [+2.4, +8.7] without cleanup, +4.7 under down-up 0.5, +5.3 under the VAE round trip, +5.3 [+0.3, +10.7] under DiffPure 0.3; within noise elsewhere.
Gates (as defined in the plan): E3 "barely drops" by the raw-PSR rule (this rule ignores that cleanups also change clean photos, which is why we report net protection); **E4 FAIL** (the doubling rule fails: v1 already survives most cleanups); **E5 PASS** (v2_main PSR under DiffPure 0.1 is 0.280 vs 0.5 x 0.243); E6 GO on 1 of 3 purifiers; invisibility FAIL for v2_final.

### 5.5 Ablations (E7), SD1.5 s0.55, no cleanup (n = 100)
`vis_only` 25.3% [19.7, 31.0] (+1.3 vs v1 [-2.4, +5.1]); `den_only` 12.3% [8.0, 17.0] (-11.7 vs v1 [-16.4, -7.3]; the perturbation is almost invisible, PSNR 50.8 dB, and does not protect); v2_main (both) 24.3%. Conclusion: the encoder-shift term does the work; the denoiser term adds nothing. **Not run:** the `no_sem` and `no_pur` ablations (S14), so the question whether the CLIP term helps and whether the cleanups in the loop matter for the encoder-shift effect is unanswered.

### 5.6 Anti-purification terms (E6)
`v2_antipure` minus v1: +4.7 [0.0, +8.7] under the VAE round trip and +5.0 [0.0, +10.0] under DiffPure 0.1 (borderline CIs), nothing elsewhere; 13 of 100 photos violate SSIM 0.95. Not recommended without a stronger structure penalty.

### 5.7 Epsilon sweep (E9) and the adaptive attacker (E8)
Budget is the main lever: PSR 17.3% at 2/255 (45.3 dB), 29.4% at 4/255 (`v2_final`, 40.5 dB), 37.7% at 8/255 (36.0 dB, fails invisibility on 96 of 100 photos); under JPEG 75 the gain flattens (about 17%, 16% and 21%, read from the figure; exact values in `results/tables/e9.md`). Figure `results/figures/psr_vs_psnr.png`. **Not run:** 6/255 (S23); the sweep has 2, 4 and 8.
Adaptive attacker (autoencoder purifier, 120 other photos, 35,496 steps, validation PSNR of its output vs the clean photo 45.1 dB): PSR under it: clean 9.3%, v1 11.0% (net +1.7 [-1.0, +4.7]), `v2_final` 10.7% (net +1.3 [-1.0, +3.7]). The protection is removed. The attacker is weak by construction (a lower bound on the damage).

### 5.8 Target-steered variants (new)

Headline figure: `results/figures/pareto_psr_vs_psnr.png` (PSR against PSNR for every method on the 100 test photos; hollow = breaks the per-photo invisibility rule).
**Screening on the 20 validation photos** (3 seeds; clean 8.3%, v1 15.0%): 

| method | PSNR mean (min) | SSIM mean (min) | photos failing the rule | PSR none | vs v1 | PSR JPEG 75 | vs v1 | PSR DiffPure 0.1 | vs v1 |
|---|---|---|---|---|---|---|---|---|---|
| v2_cheap (control) | 41.65 (40.62) | 0.9789 (0.9710) | 0 | 11.7% | -3.3 | 13.3% | +0.0 | 16.7% | -3.3 |
| v2_tgt_zero | 42.07 (40.62) | 0.9819 (0.9728) | 0 | 35.0% | +20.0* | 13.3% | +0.0 | 41.7% | +21.7* |
| v2_tgt_zero_hi | 39.99 (39.16) | 0.9617 (0.9358) | 1 of 20 (SSIM) | 46.7% | +31.7* | 26.7% | +13.3* | 58.3% | +38.3* |
| v2_tgt_gray | 41.94 (40.38) | 0.9825 (0.9744) | 0 | 35.0% | +20.0* | 15.0% | +1.7 | 28.3% | +8.3 |
| PhotoGuard (1000 it.) | 37.14 (36.91) | 0.9040 (0.8235) | 20 of 20 | 73.3% | +58.3* | 15.0% | +1.7 | 63.3% | +43.3* |
| Mist | 37.97 (37.57) | 0.9206 (0.8421) | 20 of 20 | 60.0% | +45.0* | 11.7% | -1.7 | 56.7% | +36.7* |

n = 20 was a screening sample (wide intervals, `results/S24`, `results/S25`). The control `v2_cheap` (same cleanups, no target term) does not gain, so the gain comes from the target term.

**Test photos (n = 100, 3 seeds; S26, S27, S28).** SD1.5, strength 0.55. Cell = PSR; in brackets the paired difference to **v1** under the same cleanup, 95% CI over photos (* = excludes 0). Full table with net protection over clean photos: `results/tables/final_readout.md`.

| method | none | JPEG 75 | JPEG 50 | down-up 0.5 | VAE round trip | DiffPure 0.1 | DiffPure 0.2 | DiffPure 0.3 |
|---|---|---|---|---|---|---|---|---|
| clean | 0.139 | 0.103 | 0.130 | 0.097 | 0.123 | 0.133 | 0.217 | 0.260 |
| v1 | 0.240 | 0.150 | 0.167 | 0.097 | 0.220 | 0.263 | 0.317 | 0.360 |
| v2_tgt_zero | 0.418 (+0.178 [+0.141,+0.217]*) | 0.210 (+0.060*) | 0.180 (+0.013) | 0.143 (+0.047*) | 0.433 (+0.213*) | 0.443 (+0.180*) | 0.477 (+0.160*) | 0.497 (+0.137*) |
| **v2_tgt_zero_mid** | **0.497 (+0.257 [+0.218,+0.298]*)** | 0.233 (+0.083*) | 0.203 (+0.037*) | 0.190 (+0.093*) | 0.523 (+0.303*) | 0.517 (+0.253*) | 0.570 (+0.253*) | 0.600 (+0.240*) |
| v2_tgt_zero_hi | 0.528 (+0.288 [+0.238,+0.338]*) | 0.267 (+0.117*) | 0.213 (+0.047*) | 0.190 (+0.093*) | 0.533 (+0.313*) | 0.610 (+0.347*) | 0.600 (+0.283*) | 0.663 (+0.303*) |

Reading:
- The effect is about **twice v1's** without cleanup (0.497 vs 0.240), under down-up 0.5 (0.190 vs 0.097), under DiffPure 0.1 (0.517 vs 0.263) and about 2.4 times under the VAE round trip (0.523 vs 0.220); under JPEG 75 it is 1.6 times (0.233 vs 0.150) and under JPEG 50 about 1.2 times. JPEG remains the weakest point.
- `v2_tgt_zero_mid` is the strongest variant that satisfies the invisibility rule on all 100 test photos; `v2_tgt_zero_hi` is stronger but 4 photos fall below SSIM 0.95. The monotone order (zero < mid < hi) with the target weight is the dose-response we expected.
- Other editors (no cleanup): SD1.5 strength 0.75: PSR 0.910 / 0.900 / 0.897 vs clean 0.840 (+0.070, +0.060, +0.057, all CIs exclude 0); IP-Adapter FaceID: 0.007 / 0.020 / 0.020 vs clean 0.000 (tiny: +0.020 [+0.003, +0.043] for `mid`); SD1.5 strength 0.3: 0.000 / 0.000 / 0.007. The identity-conditioned and masked editors are still not meaningfully protected. InstructPix2Pix, inpainting and SDXL were not run for these variants.
- **Not tested:** the adaptive attacker (section 5.7) was run for v1 and `v2_final` only; we have no evidence that the target-steered variants survive it, and the earlier result suggests they may not.
- **Comparison with existing tools (E2), 100 test photos, run at our budget 4/255:** PhotoGuard reaches **PSR 81.0%** (net +67.1 over clean [+62.6, +71.7]) and Mist **63.1%** (net +49.2 [+44.3, +54.2]), more than any of our variants, **but at the cost of visibility:** PhotoGuard fails the invisibility rule on 100 of 100 photos (PSNR 37.15, SSIM 0.906, worst SSIM 0.812) and Mist on 100 of 100 by SSIM and 50 of 100 by PSNR (38.00, SSIM 0.923). Under cleanups their advantage shrinks or reverses: under JPEG 75 PhotoGuard 18.3% and Mist 21.0% against 23.3% for `v2_tgt_zero_mid` (paired mid minus PhotoGuard +5.0 [+1.3, +9.0]; minus Mist +2.3 [-1.3, +6.0]); under JPEG 50 mid is ahead of both (+3.7, +4.7); under down-up 0.5 mid is ahead (+5.3 and +6.3, both CIs exclude 0). Under the VAE round trip and DiffPure 0.1 the existing tools stay ahead (PhotoGuard 59.0% and 67.3% against mid's 52.3% and 51.7%; paired mid minus PhotoGuard -6.7 [-12.0, -1.0] and -15.7 [-21.7, -9.0]). So: stronger raw protection from the visible tools; at invisible levels, our variants keep more of their protection under JPEG and resizing. Other editors: PhotoGuard reaches 71.7% on InstructPix2Pix (clean 18.0%, net +53.7 [+44.3, +62.3]; v1 and Mist do not), 95.7% at SD1.5 strength 0.75, 17.0% at strength 0.3, 2.7% on FaceID and 0% on inpainting; Mist: 4.7% at strength 0.3, 4.3% on InstructPix2Pix, 3.7% on FaceID. Caveat: both are our re-implementations checked against the official code and run at our budget (4/255); at their own larger budgets they would be stronger and more visible.

## 6. Failure cases (see `results/figures/failure_gallery.png`, `before_after.png`, `failure_cases.csv`; the images contain real faces and stay local)
- Strength 0.3: no edit is protected (0%); strength 0.75: identity is already destroyed for clean photos (84%), so no gain can be measured.
- Identity-conditioned and masked editors: FaceID 0.0-0.3%, inpainting 0%, SDXL 2%.
- InstructPix2Pix: protected edits are *less* often counted as protected than clean ones (v1 9.0% vs clean 18.0%; v2_final 3.3%). Part of this is a face-detection effect (clean edits lose the face more often, which counts as protected); we have not separated it.
- JPEG q50 and down-up cleanups erase most of the gain for v1 and `v2_final`.
- Adaptive attacker: removes the protection.
- Visual check: in the before/after panel the protected edits often keep the person's hair, pose and facial structure; some `v2_final` edits show texture artifacts in hair and background. A recognition score below a threshold is not the same as a person no longer being recognisable. No human study was run.

## 7. Limitations (all real)
DPM-Solver++ sampler for all editors (not v1's); the DiffPure-style cleanup is built on SD1.5, not the original DiffPure; GrIDPure unavailable, IMPRESS and InstantID not run; 40 PGD iterations, K = 4; Optuna 16 trials on 10 validation photos with a proxy objective and a 20-photo confirm, where the proxy's best trial (0.10) did not hold up (0.05) and the tuned configuration is not separable from the hand-set one (0.067 vs 0.042); the training identity loss was not evaluated; a single CLIP model; same-person de-duplication used ArcFace similarity >= 0.5 (unverified threshold) and the splitter's face detector missed 38 of 100 test faces (fixed for evaluation by padding the image; de-duplication of the test list was not repeated); the adaptive attacker is weak and was not run against the target-steered variants; the choice of the "best" target-steered variant is read from the test table (post hoc among three declared variants); PhotoGuard and Mist are our re-implementations (checked against the official code) run at our budget 4/255 on all 100 test photos (they were also screened on the 20 validation photos); edit-LPIPS was computed for 513 pairs only (CPU cost) and is not reported; v1's Phase 2 outputs were lost (its 0.46 LPIPS can only be quoted); the PSR threshold comes from ArcFace similarity between different people and is the project's operational definition of "not the same person".

## 8. Reproducibility
Config `config/config.yaml` + `config/parts/*.yaml`; bundles built from commits (hash printed in every Kaggle log); results under `results/` (CSV/JSON/logs committed, images never); pipeline: `scripts/ingest_results.py`, `scripts/make_tables.py`, `scripts/make_figures.py`, `scripts/readout_final.py`; 377 CPU tests; `requirements.txt` pinned to the Kaggle image of 2026-10-14. See `README.md`.
