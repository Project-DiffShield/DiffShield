# DiffShield v2: report draft (skeleton)

Status: **skeleton with placeholders.** Nothing in this file is a result. Placeholders (double-brace `table:`, `num:` and `gate:` tags) are
filled by `python scripts/make_tables.py --render docs/REPORT_DRAFT.md --tables_dir results/tables --out reports/rendered` from the tables and
`numbers.json` generated from the results CSVs; a placeholder with no data renders as `PENDING (...)`, never as an invented value.
Text outside placeholders is method description and limitations decided before any results existed (sources: docs/HANDOFF_v2.txt,
docs/V2_PLAN.md, docs/V2_PLAN_48H.md).

Data status of the rendered copy: {{num:synthetic}}

## Abstract (write last)
Claim template: under the stated threat model (white-box SD encoders, the listed editors and purifiers, epsilon = 4/255 on the standard
0-255 scale), DiffShield lowers the ArcFace identity retention of edits by **{{num:claim.A}}** versus a uniform-noise control and
**{{num:claim.B}}** over v1, including after held-out purifiers (section 5.5); an adaptive attacker reduces the protection success rate to
**{{num:claim.C}}**. Write nothing stronger than the tables show; if a gate failed, say so here.

## 1. Threat model (page one)
**Who and what is protected.** A person posts a face photo. An attacker feeds it to a diffusion editor to make an edit of that person. The
defender adds a small invisible perturbation first so that edits no longer look like the same person. Defensive, privacy-protecting use.

**Defender budget.** L-infinity epsilon = 4/255 on the standard 0-255 scale (8/255 on [-1,1] tensors, which is what v1 reported). Check on
real pixels: {{num:budget.confirmed|PENDING (Phase 0 budget check, J01)}}. Required invisibility: PSNR >= 38 dB, SSIM >= 0.95 on every
image, human 2AFC accuracy not far above 50% (E10).

**Attacker knowledge** (two modes, never mixed in one table):
| Mode | Attacker knows | Used for |
|---|---|---|
| Non-adaptive (headline) | nothing about DiffShield; picks an editor and optionally a purifier | E0-E7, E9 |
| Adaptive (stress test) | the method; trains an autoencoder on (clean, protected) pairs of images disjoint from the test identities and uses it as purifier | E8 |

**Defender knowledge (white-box on the surrogate).** Protection is computed with the public SD1.5 VAE and UNet (plus the SDXL VAE in the
v2_final ensemble). Editors that share these components are attacked white-box; the others are transfer targets. This is an assumption, not a
worst-case guarantee.

**Editors** (DPM-Solver++ 30 steps, 3 edit seeds): SD1.5 img2img (strength 0.3/0.55/0.75; headline 0.55 with 3 prompts), SDXL img2img (subset
of methods), InstructPix2Pix, SD inpainting with a keep-face mask, IP-Adapter FaceID. Not tested: InstantID.

**Purifiers.** In the protection loop: JPEG, down-up resize, SD VAE round trip, one-step SD purifier. Held out (evaluation only, real codecs and
separately loaded models): JPEG q90/q75/q50 and a different JPEG, 0.5x and a different down-up, VAE round trip, DiffPure-style noise-denoise at
t* = 0.05/0.1/0.2/0.3, Swin2SR restoration, GrIDPure ({{num:gridpure.status|PENDING (smoke job)}}). DiffPure shares the SD1.5 UNet with the
one-step training purifier, so it is gray-box rather than fully held out.

**Metric.** ArcFace similarity between source and edit (insightface buffalo_l, evaluation only). PSR = share of edits with no face (counted as
success, also reported as no-face rate) or similarity below a threshold = 95th percentile of the similarity between clean-photo edits and OTHER
people's photos ({{num:threshold.pooled|PENDING (E0)}} pooled; per editor in table notes). Seeds are averaged within image, then mean over
images with a 95% bootstrap CI (10,000 resamples); differences are paired with v1 or the noise control on the same images.

**Known blind spot.** If an attacker masks the face itself in inpainting, those pixels are discarded before the model sees them, so noise there
cannot protect it. Our inpainting editor uses a keep-face mask (it repaints background, clothes, hair), the setting where protection can work.

## 2. Method
v1 (frozen): PGD + EoT (random JPEG, random resized crop); loss alpha*L_vis - beta*L_sem - gamma*L_str (VAE latent shift, CLIP B/32 similarity to
"a potted plant", windowed SSIM); epsilon 4/255, step 0.5/255, 40 iterations; alpha 0.2699, beta 0.9024, gamma 60.53. Reproduction:
{{num:phase0.summary|PENDING (PHASE_0.md)}}.

v2: two halves that never share components. Half A (protection): each PGD step draws a purifier from a training mixture (JPEG, down-up, VAE round
trip, one-step SD purifier) and averages gradients over K = 4 draws; objective alpha*L_vis + eta*L_den - beta*L_sem - gamma*L_str with L_den the
UNet noise-prediction error at a random timestep in [50, 300], L_sem kept only if the ablation shows it helps, L_str on the PUBLISHED image.
Variants: v2_cheap, v2_main (+ one-step purifier + L_den), v2_antipure (+ high-frequency and error-growth terms), v2_final (tuned on validation
images only; SD1.5 + SDXL VAE ensemble). Half B (attack and score): saved protected PNG -> held-out purifier -> editors -> ArcFace, CLIP-I, LPIPS.

Tuning: Optuna on validation images only, {{num:optuna.trials|PENDING (Phase 6)}} trials on a proxy objective (PSR under none/jpeg_q75/diffpure_0.1,
10 images, 20 iterations, K = 2, 1 seed), top candidates confirmed on all 20 validation images; constraints PSNR >= 38, SSIM >= 0.95.
Controls and baselines: uniform noise at the same budget and Gaussian noise at v1's PSNR (E1); PhotoGuard-style encoder attack and a Mist-style
fused loss (E2; our reimplementations, not the official tools).

## 3. Setup
Test 100 identities (one CelebA-HQ image each) + 20 messy photos; train 20, validation 20 (lists in data/splits/); identity separation by ArcFace
de-duplication (same person if similarity >= 0.5, unverified threshold) unless identity labels were supplied; 1 protection seed per image; 3 edit
seeds; 3 prompts for SD1.5 s0.55, else 1 rotated prompt; edit LPIPS on JPEG q90 versions of both edits. What actually ran:
{{table:coverage}}

## 4. Invisibility
{{table:invisibility_methods}}
{{gate:INV}}
Human 2AFC (E10): {{num:human.task_a|PENDING (E10 not run)}} ({{num:human.pilot|PENDING}}; raters {{num:human.n_raters|PENDING}}).

## 5. Results
5.1 E0 {{table:e0}}
5.2 E1 {{table:e1}}
5.3 E2 {{table:e2}} and E3 {{table:e3}}  Decision (Phase 5): {{gate:E3}}
5.4 E4 {{table:e4}}  Gate: {{gate:E4}}
5.5 E5, the headline {{table:headline}} {{table:e5}}  Gate: {{gate:E5}}  Claim numbers: {{table:claims}}
5.6 E6 {{table:e6}}  Go/no-go: {{gate:E6}}
5.7 E7 {{table:e7}} {{table:e7_editors}} {{table:e7_paired}}
5.8 E8 adaptive attacker {{table:e8}}
5.9 E9 epsilon sweep {{table:e9}} (figure results/figures/psr_vs_psnr.png)
5.10 Per editor and messy photos {{table:per_editor}} {{table:messy}}
5.11 E10 {{table:e10}}
All gates: {{table:gates}}

## 6. Failure cases
List every failure with numbers: images where v2_final edits still match the source (results/figures/failure_cases.csv; the gallery contains
CelebA-HQ photos and stays local), weak editors (FaceID reads identity through ArcFace, not the VAE), collapsing purifiers (5.5), missed gates
(5.4-5.6), negative ablation results (5.7). {{num:failures.summary|PENDING (write after the tables exist)}}

## 7. Limitations (decided before results; all scale-downs of docs/V2_PLAN_48H.md section 2)
1. Scale-downs: 16 Optuna trials on a 10-image proxy; PGD 40 iterations and K = 4 (K = 2 for Optuna and the adaptive pool), no 100-iteration or
   K = 8 run; 1 protection seed per image; 3 prompts only for SD1.5 s0.55; strengths 0.3/0.55/0.75 only for nine headline methods; SDXL only for
   clean, v1, PhotoGuard, v2_final; purifier grid not complete for every pair (see coverage).
2. Sampler changed: DPM-Solver++ 30 steps for all editors; v1's original evaluation used the library default (PNDM 50). v1 was re-evaluated under
   the same sampler; v1's original Phase 2 LPIPS (0.46, 15 pairs, one prompt) is not comparable.
3. PhotoGuard (encoder attack only) and Mist (fused loss, procedural target pattern) are reimplementations with unverified hyper-parameters.
4. GrIDPure: {{num:gridpure.status|PENDING}}; IMPRESS not run; DiffPure is gray-box (shares the SD1.5 UNet).
5. The adaptive attacker is weak by construction (about 1 GPU-hour, 120 disjoint images, 128 px patches): E8 is a lower bound on the damage.
6. Identity splits are approximate (ArcFace >= 0.5 as same person, unverified) unless labels were supplied; the PSR threshold assumes one image per
   identity.
7. Edit LPIPS is computed after JPEG q90 on both edits (same for every method), later, on CPU.
8. Single CLIP encoder in L_sem; L_sem may have no direct path into SD1.5 img2img (E7 answers it); CLIP-I uses ViT-L/14.
9. A differentiable face model in the loss (if enabled) differs from the evaluation ArcFace; the evaluation model's licence is non-commercial.
10. v1's step is 0.5/255 on the standard scale (handoff table said 1/255); v2 defaults are tuned, not copied.
11. Human study: {{num:human.pilot|PENDING}}; below 10 raters it is a pilot with the actual n.
12. GPU kernels are non-deterministic; reproduction is statistical; library versions are in versions.json of every run.
13. CelebA-HQ is research-use only; no identifiable image is published or committed; only CSV/JSON results are in git.

## 8. What we claim and do not claim
Claim (only if the tables support it): under the stated threat model (white-box SD encoders, listed editors and purifiers, epsilon = 4/255),
ArcFace identity retention of edits drops by A points versus a noise control and B points over v1, including after the held-out purifiers of 5.5;
an adaptive attacker reduces this to C (E8).
Do not claim: survival of any scenario; immunity to purification; protection against editors, purifiers or attacks not tested (InstantID, IMPRESS,
stronger adaptive attackers, face-masking inpainting); that the measured PhotoGuard/Mist are the official tools; anything from the demo.

## 9. Reproducibility
python scripts/ingest_results.py --results results
python scripts/compute_edit_lpips.py --combined results/combined --workers 4
python scripts/ingest_results.py --results results            # re-applies the LPIPS cache
python scripts/make_tables.py  --combined results/combined --out results/tables
python scripts/make_figures.py --combined results/combined --out results/figures
python scripts/make_tables.py  --render docs/REPORT_DRAFT.md --tables_dir results/tables --out reports/rendered
