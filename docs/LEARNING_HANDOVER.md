# DiffShield: learning handover (for a chat whose job is to TEACH the project)

Paste this whole file into a new chat and say: "Teach me this project; quiz me; answer questions from this document and say when something is not covered." Written 2026-10-14 night. Everything here comes from the repo docs and the Phase 0 / S03 / S04 results so far. Things marked **(unverified)** have not been confirmed by a real run. **No v2 result numbers exist yet**; never invent any.

The learner (Siddhartha Gopala) is a final-year student, new to diffusion models. Explain in short plain language, define each term the first time, lead with the main idea, use small analogies, ask at most one or two questions at a time.

---

## 1. The problem and the idea (one paragraph)

People post face photos online. Anyone can feed a photo to a diffusion-model editor (for example Stable Diffusion img2img: "take this photo and turn it into the same person in a suit") and make a fake. DiffShield is a **defensive** tool: before posting, you add an almost invisible noise pattern to the photo. The pattern is found by an optimizer so that, when someone edits the protected photo, the result **no longer looks like the same person**. The protected photo itself must look identical to the original to a human.

Key vocabulary
- **Adversarial perturbation**: a tiny, carefully chosen change to the pixels that confuses a model.
- **L∞ budget (epsilon, ε)**: the largest allowed change of any single pixel value. ε = 4/255 means no pixel moves by more than 4 brightness levels out of 255.
- **PGD (projected gradient descent)**: repeat many times: compute how to change the noise to increase a goal, take a small step in the sign of that direction, then clip the noise back inside the ε limit.
- **Diffusion model / Stable Diffusion (SD)**: generates images by starting from noise and removing it step by step. SD works in a compressed "latent" space made by a **VAE** (an encoder that shrinks an image into latents and a decoder that expands them back). A **UNet** is the network that predicts the noise to remove.
- **img2img editing**: encode the photo with the VAE, add noise to a chosen strength, then denoise guided by a text prompt. Low strength keeps the photo, high strength changes more.
- **Purification (purifier)**: a cleanup an attacker applies before editing to wash the noise off: JPEG compression, shrink then enlarge, VAE encode-decode, "DiffPure" (add noise then denoise with a diffusion model), restoration networks.
- **EoT (Expectation over Transformation)**: while optimizing, randomly apply cleanups/transforms and average the gradients, so the noise survives them.
- **ArcFace**: a face-recognition network. It turns a face into a vector; cosine similarity between two vectors says how much they look like the same person.
- **PSR (Protection Success Rate)**: share of edits whose ArcFace similarity to the source falls below a threshold (the headline number). Threshold = 95th percentile of similarity between different people, measured on clean-image edits (experiment E0). If no face is found in an edit, it counts as protected, and the no-face rate is reported separately.
- **PSNR / SSIM / LPIPS**: PSNR and SSIM measure how similar two images are (higher = more similar). LPIPS measures perceptual difference (lower = more similar). "Invisibility" = protected vs original must stay PSNR ≥ 38 and SSIM ≥ 0.95. Edit-LPIPS = LPIPS between the edit of the clean photo and the edit of the protected photo (how much the edit changed).
- **Bootstrap 95% CI**: a confidence range from resampling images; every reported number needs n, mean ± CI, 3 edit seeds, and paired comparison against v1 on the same images.

## 2. v1 (what existed) and what is wrong with it

v1 = two notebooks (Phase 1 build/tune protection, Phase 2 test on SD1.5 img2img) plus the repo Project-DiffShield/DiffShield, patched at runtime with string replaces (`same_on_batch=True` in `src/eot.py`, loss injection in `src/optimizer.py`, SSIM structure loss in `src/losses.py`).

v1 algorithm: PGD for 40 iterations; EoT with Kornia RandomJPEG + RandomResizedCrop. Loss `L = α·L_vis − β·L_sem − γ·L_str`:
- `L_vis`: MSE between SD1.5 VAE latent of protected and original image, **maximized** (push the encoder's view of the photo away).
- `L_sem`: 1 − cosine(CLIP image embedding, text embedding of "a potted plant"), **minimized** (make the photo "look like a potted plant" to CLIP).
- `L_str`: 1 − windowed SSIM between original and protected, **minimized** (keep it looking unchanged).
Weights tuned with Optuna (30 trials): α = 0.2699475469647097, β = 0.9023875567667267, γ = 60.52658429696642. Calibration set from KMeans on CelebA-HQ attributes (4 images: 5319, 1715, 6385, 8932); 70 disjoint evaluation images.

v1's reported results: PSNR 41.911 ± 0.440, SSIM 0.97496 ± 0.0048, LPIPS(clean, protected) 0.2316 ± 0.063 (n = 70); in Phase 2, edit-LPIPS about 0.46 on 15 image pairs (SD1.5 img2img, strength 0.55, guidance 7.5, seed 2026, one prompt). The Phase 2 outputs were lost (the zip was empty), so 0.46 can only be quoted, not recomputed.

What is wrong with v1 (what v2 fixes)
1. "Survives any scenario" cannot be proven; it was never tested against purification.
2. No random-noise control: part of the effect may be just noise.
3. No identity metric: LPIPS measures change, not whether the person is still recognizable.
4. One editor, one prompt, one strength, one seed, 15 images.
5. Optuna rewarded visible change (worst-case LPIPS), not protection.
6. Attacks mainly one phase (the VAE encoder). The denoiser (UNet) is untouched. `L_sem` uses the CLIP **image** encoder, which SD1.5 img2img never runs, so it may do nothing **(unverified; one ablation answers it)**.
7. One encoder, no ensemble, no ablation of loss terms.
8. Runtime string patches, unlogged seeds, hard to reproduce; clean images only.

## 3. Phase 0 findings (real, from Kaggle runs J00 + J01, 2026-10-14)

- **The budget is 4/255, not 8/255.** v1's "8/255" was measured on tensors scaled to [-1, 1]; on the normal 0–255 scale that is 4 gray levels. Confirmed on real PNGs: max change = 4 in all 70 images. All v2 budgets are on the 0–255 scale (`eps_model = eps_255 × 2/255`).
- **v1's step size** was 0.5/255 on the normal scale (not 1/255 as the brief says). v2 presets use 1.0/255.
- **v1 reproduces** with the rebuilt code (n = 70): PSNR 41.986 ± 0.416, SSIM 0.9754 ± 0.0054, LPIPS 0.2314 ± 0.0637; paired differences vs v1's own CSV: +0.075 dB, +0.0004, −0.0002. The user accepted the tolerance ±0.5 dB, ±0.005 SSIM, ±0.03 LPIPS. Not bit-exact (per-image seeds instead of one global seed; GPU non-determinism).
- v1 cost on a T4: 45.6 GPU-seconds per image (1.14 s per PGD iteration), 1.8× the planner's estimate.
- Software pinned to Kaggle's image: Python 3.13.15, torch 2.11.0+cu128, diffusers 0.40.0, transformers 5.16.1, kornia 0.8.3.
- Clean input convention in v1: 512 px JPEG at PIL default quality 75. v1's PSNR cast to uint8 by truncation; v2 reports both that and the rounded published PNG.

## 4. v2: the target system

Two halves that never share components.

**Half A, protection loop (offline).** For each photo x, repeat PGD: draw a purifier P from a training mixture (EoT), compute the scores on P(x + δ): latent shift `L_vis`, denoiser error `L_den`, identity loss `L_id` (ArcFace through the VAE reconstruction; Phase 6), semantic term `L_sem` (kept only if ablation shows it helps), and visibility penalty `L_str` computed on x + δ itself (the published image, not the purified one). Step δ by sign gradient, clip to the ε ball, repeat.

`δ* = argmax_{‖δ‖∞ ≤ ε} E_{P,ξ}[ α·L_vis + η·L_den − β·L_sem ] − γ·L_str`

- `L_den`: noise-prediction error of the SD UNet on the processed image at a random timestep, maximized.
- Training purifiers (inside the loop, differentiable): JPEG at random quality, bicubic down-up at random scale, SD VAE round trip, and a **one-step SD purifier** (VAE-encode, add noise at a random timestep in [50, 300], predict noise with the UNet, recover a one-step estimate, decode).
- Optional **anti-purification** terms (AntiPure idea): raise high-frequency content of the one-step reconstruction; keep the reconstruction error growing with t.

**Half B, attack-and-score harness.** Takes a protected image, applies **held-out** purifiers (never used in training: JPEG/down-up with different parameters, multi-step DiffPure at t* = 0.05/0.1/0.2/0.3, Swin2SR restoration, GrIDPure if it runs), edits with five editors (SD1.5 img2img at strengths 0.3/0.55/0.75, SDXL img2img, InstructPix2Pix, SD inpainting with face-keeping masks, IP-Adapter FaceID as an identity-conditioned editor), and scores with ArcFace, PSR, CLIP-I, edit-LPIPS.

Two evaluation modes, never mixed: **non-adaptive** (attacker does not know the method; headline) and **adaptive** (attacker trains a small autoencoder on clean/protected pairs of other images, the LightShed idea, to remove the noise; E8; honest stress test).

Method names in the code: `v1, v2_cheap` (cheap purifiers only), `v2_main` (+ one-step purifier + `L_den`), `v2_antipure`, `v2_final` (Optuna-tuned), ablations `abl_vis_only, abl_den_only, abl_no_sem, abl_no_pur`, ε sweep `eps2, eps6, eps8`, controls `noise_uniform, noise_gauss_psnr`, baselines `photoguard, mist`.

Baselines: noise controls (uniform noise at the same L∞ budget; Gaussian noise matched to the same PSNR); **PhotoGuard** (encoder attack, official 1000 iterations) and **Mist** (reimplementations checked against the official code, run at OUR ε = 4/255). Reimplementations, not the official packages.

## 5. Experiments E0–E10 and gates

| ID | What | Question |
|---|---|---|
| E0 | clean images through all editors | reference edits and the PSR threshold |
| E1 | random noise at same budget | how much is just noise? |
| E2 | PhotoGuard, Mist | compare with existing tools |
| E3 | v1 through each purifier | how badly does purification break v1? |
| E4 | v2 with cheap purifiers | fixes JPEG, down-up, VAE round trip? |
| E5 | v2 with one-step SD purifier, tested on held-out DiffPure/GrIDPure | transfers to unseen purifiers? |
| E6 | anti-purification terms | more robust at higher t*? |
| E7 | ablations | which loss term / phase matters (encoder only vs denoiser only vs both vs both without `L_sem`) |
| E8 | adaptive attacker | what breaks it? |
| E9 | ε sweep 2, 4, 6, 8 | invisibility vs robustness |
| E10 | human study (15 raters, 20 pairs; pilot if fewer than 10 raters) | is it invisible, is identity gone? |

Gates (suggestions; if missed, report honestly, never quietly change): after E3, if v1 barely degrades under purification, skip anti-purification; after E4, PSR under JPEG/down-up/VAE round trip at least doubles vs v1 at PSNR ≥ 38; after E5, PSR under held-out DiffPure t* = 0.1 reaches at least half of the no-purifier PSR. Invisibility: PSNR ≥ 38, SSIM ≥ 0.95, human 2AFC not far above 50%.

Expected outcome (a prediction, not a result): cheap purifiers largely fixed; partial transfer to low-strength DiffPure; weak against strong purification, restoration purifiers and adaptive attackers. Partial but measured results count as a good outcome.

What may be claimed: "Under a stated threat model (white-box SD encoders, listed editors and purifiers, ε = X/255), DiffShield lowers ArcFace identity retention of edits by A points vs a noise control and B points vs v1, including after [held-out purifiers]; an adaptive attacker reduces this to C." Not claimable: survives any scenario, immunity to purification, protection against untested editors. Inpainting limit: if an attacker masks the face itself, those pixels are discarded, so noise there cannot protect it.

## 6. Data and splits

CelebA-HQ via Kaggle `ipythonx/celebamaskhq` (no identity labels, so same-person de-duplication uses ArcFace cosine ≥ 0.5, threshold **unverified**). Frozen lists in `data/splits/` (made by session S03, committed): test 100 (includes all 70 v1 evaluation images, so paired comparison with v1 works) + 30 new; messy 20 (pose, blur, lighting, occlusion, with JPEG/resize degradation); train 20; val 20; adaptive-attacker pool 120; v1 calibration images excluded. 40 images had no detectable face for the identity check (limitation). Never tune on test.

Edit settings: DPM-Solver++ 30 steps, 3 seeds, prompts rotated per image except the headline SD1.5 strength-0.55 setting (all 3 prompts); edits saved as JPEG q90; edit-LPIPS computed afterwards on CPU.

## 7. Code architecture (repo: `C:\Users\karth\Documents\DiffShield`, integration worktree `...\DiffShield-wt\integration`, branch `v2/integration`; nothing pushed)

- `config/config.yaml` + `config/parts/*.yaml` deep-merged; every run writes resolved config, git hash, library versions, seed.
- `src/protect/`: `registry.py` (build a protector by name), `v1.py` (v1 adapter), `models, purifiers, losses, pgd, antipure, ensemble, face_loss, v2_registry` (Half A). Frozen interface: `Protector.protect(x[1,3,H,W] in [-1,1], seed) -> (tensor, dict)`.
- `src/eval/`: `editors, purify_attacks, faces, clip_metric, masks, edit_runner, analysis, metrics, stats, schema, adaptive_ae` (Half B). CSV schema: one row per (image, seed, editor, purifier, method).
- `src/baselines/`: noise controls, PhotoGuard, Mist.
- `scripts/`: `run_protect, run_job, evaluate, tune_optuna, make_splits, prepare_data, ingest_results, compute_edit_lpips, make_tables, make_figures, make_human_study, analyze_human_study, demo, compare_runs, budget_check, gpu_ledger, smoke_report`, and others.
- `kaggle/`: `jobs.yaml` (job and session definitions), `job_runner.ipynb` (human sets `JOB_IDS`, optional `PARAMS`, `RESUME_FROM`), `build_bundle.py` (builds the code zip uploaded as a Kaggle dataset).
- Tests: CPU-only with tiny stand-in models (297 passed, 1 skipped at last run). Real SD/ArcFace/CLIP weights are exercised only on Kaggle.
- Decision made earlier: planning with Opus (high effort), implementation by Sonnet subagents in separate git worktrees (protect, eval, baselines, analysis), coordinator merges.

## 8. The 48-hour plan and compute

Deadline about 16:30 on 2026-10-16. No local GPU; all GPU work on Kaggle (T4 x2), 5 accounts with about 30 GPU-hours/week each (acct1, acct2 are the user's; acct3–5 teammates, launched 09:00–23:00). Sessions S01–S18 (see `docs/V2_PLAN_48H.md` §5 and `docs/KAGGLE_LAUNCH.md`): smoke tests (S04 protection, S05 editing/scoring), then overnight S06–S09 (Optuna tuning + v2_final protection, clean-edit reference, v1 edits, v2_main edits), then day-2 sessions for the other methods, ablations, adaptive attacker, ε sweep. Outputs come back as `<JOB>_outputs.zip`, unzipped under `results/` (images are git-ignored: identifiable faces, research use only, never commit).

Scale-downs (must be listed as limitations): Optuna 16 trials × 10 val images with a proxy objective instead of 30–50 × 20; 40 PGD iterations and K = 4 purifier draws (no 100 iterations / K = 8); DPM-Solver++ sampler instead of v1's default; SDXL only for some methods; IMPRESS not run; InstantID not run; single CLIP (no ensemble); weak adaptive attacker (120 images, 128 px patches, 1 GPU-hour); not every method × purifier pair measured; GrIDPure dropped if it won't run after about 1 hour of effort.

## 9. Where things stand (2026-10-14, about 00:00 IST)

Done and verified: Phase 0 (J00, J01); splits (S03); S04 protection smoke: 12 of 14 methods ran, PhotoGuard crashed because the protector did not expose its device (fixed, regression test added; Mist therefore not yet run).

Open: smoke timings for the v2 methods looked 3–4× slower than planned (v2_main about 50–75 s per image at only 5 iterations vs a plan of about 120 s for 40 iterations), but with only 2–3 images per GPU the warmup may explain much of it. New job **S04t** (4 images × 10 iterations) measures the real cost, plus PhotoGuard and Mist. S05 (editors/purifiers/scoring smoke, first real use of insightface, SDXL, IP-Adapter, GrIDPure) is running. Unknown: whether Kaggle bills a 2-GPU session as 1 or 2 quota hours per hour. Fallback rules cut work if timings exceed estimates by more than 1.5×. No v2 protection-quality or protection-success numbers exist yet.

## 10. Honesty rules (binding)

Never invent a number, citation or result; label unverified things "unverified". No "RSAS" method. Report negative results. Every number: n, mean ± 95% bootstrap CI over images, 3 seeds, paired vs v1. Never tune on test. No deletions (move to `_to_delete/`). Never ask for or store passwords or tokens. Don't redistribute datasets or commit identifiable images. Don't push without asking.

## 11. Sources named in the brief (look up the official versions before citing)

DiffPure; GrIDPure (Zhao et al.); IMPRESS; DiffAttack; AntiPure; LightShed (USENIX Security 2025); Liu et al. diffusion purification benchmark; FastProtect (Ahn et al., CVPR 2025); PhotoGuard (MadryLab/photoguard, checked at commit 686bea7); Mist (psyker-team/mist, checked at commit 99f5f3c). Do not cite details from memory.

## 12. Suggested way to teach this (for the new chat)

1. Diffusion editing and why a photo can be attacked (VAE, UNet, img2img). 2. v1: the three loss terms and PGD with EoT. 3. Why v1's claims were weak (the 8 problems above). 4. Budget and metrics (what 4/255, PSNR, SSIM, LPIPS, ArcFace, PSR mean). 5. Purifiers and why EoT with purifiers helps. 6. v2 architecture (Half A vs Half B), the new losses, held-out purifiers. 7. Experiments E0–E10 and what each would prove. 8. How the pipeline runs on Kaggle. 9. Limitations and honest claims. After each step ask the learner one short check question.
