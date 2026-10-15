# DiffShield v2: the complete guide (for a reader who is new to all of this)

Written 2026-10-15 (early morning, IST). Paste this whole file into another chat and say: "Explain this project to me section by section, check my understanding, and answer from this document. Say clearly when something is not in it."
Everything below is from the repo documents, code, and the real Kaggle runs so far. Anything marked **(unverified)** has not been confirmed by a real run. **No v2 protection-quality results exist yet** (the big runs are in progress). Nothing here may be quoted as a result unless it is in section 9 and marked measured.

---

## 1. The problem in plain words

People post face photos online. Anyone can give such a photo to an AI image editor and ask for a changed version: "the same person wearing a suit", "at a beach", "as an oil painting". These editors are **diffusion models** (Stable Diffusion and similar). The result can be a convincing fake of a real person.

DiffShield is a **defensive** tool. Before you post a photo, it adds a tiny, almost invisible pattern of noise to the pixels. The pattern is found by an optimizer so that **when someone feeds the protected photo to an editor, the output no longer looks like the same person**. To a human the protected photo looks identical to the original. It protects people's own photos; it does not create fakes.

Two goals pull against each other:
- **Invisible:** the protected photo must look the same (measured by PSNR, SSIM, LPIPS).
- **Effective:** the edits of the protected photo must stop resembling the person (measured by a face-recognition network).

A stronger pattern is more effective but more visible. The "budget" ε (below) caps how visible it may be.

## 2. Background you need (short glossary)

- **Pixel values and "/255":** an image pixel channel is a number 0–255. "ε = 4/255" means: no pixel value may change by more than 4 steps. Some code uses the range [-1, 1] instead of 0–255; on that scale the same change is "8/255". **This mix-up is the first thing we found wrong with v1 (section 3).**
- **Diffusion model:** generates an image by starting from random noise and removing noise step by step, guided by a text prompt.
- **Latent space and the VAE:** Stable Diffusion does not work on pixels directly. A **VAE** has an *encoder* that squeezes an image into a small grid of numbers (the "latent") and a *decoder* that expands it back. A **UNet** is the network that predicts, at each step, which noise to remove.
- **img2img editing:** encode the photo with the VAE, add noise up to a chosen *strength* (low = keep the photo, high = change a lot), then denoise guided by the prompt. Other editors: **InstructPix2Pix** (follows instructions such as "make him wear a suit"), **inpainting** (repaint only a masked region), **SDXL** (a bigger Stable Diffusion), **IP-Adapter FaceID** (an editor that is told whose face to keep, using a face embedding).
- **Adversarial perturbation:** a tiny change to an input that makes an AI model behave wrongly. DiffShield attacks the *editor's* encoder so that its view of the photo is wrong.
- **PGD (projected gradient descent):** the optimizer. Repeat N times: ask "which tiny change to the noise increases my goal the most?" (the gradient), take a small step in that direction (we use only the *sign* of the gradient), then clip the noise back inside the ±ε limit.
- **EoT (Expectation over Transformation):** while optimizing, randomly apply cleanups to the image and average the gradients, so the noise survives those cleanups.
- **Purification / purifier:** a cleanup an attacker applies before editing, to wash the noise off: JPEG compression, shrink-then-enlarge, VAE encode-then-decode, **DiffPure** (add some noise, then denoise with a diffusion model), restoration networks (Swin2SR), GrIDPure, IMPRESS.
- **ArcFace:** a face-recognition network. It turns a face into a vector; the cosine similarity of two vectors says how much they look like the same person (higher = more similar). We use it through the `insightface` package.
- **PSR (Protection Success Rate):** the headline number: the share of edits whose ArcFace similarity to the source person falls below a threshold. The threshold is the 95th percentile of the similarity between *different* people, measured on edits of clean photos (experiment E0). If no face is found in an edit, that counts as protection success, and the no-face rate is reported separately.
- **PSNR, SSIM, LPIPS:** PSNR and SSIM measure how alike two images are (higher = more alike); LPIPS measures perceptual difference (lower = more alike). Invisibility rule: protected vs original must keep **PSNR ≥ 38 and SSIM ≥ 0.95**. "Edit-LPIPS" = LPIPS between the edit of the clean photo and the edit of the protected photo.
- **Bootstrap 95% confidence interval (CI):** a range from resampling the images; it says how sure we are about a mean. Every reported number needs n, mean ± CI, and 3 edit seeds.
- **Paired comparison:** comparing two methods on the *same* images, which removes image-to-image noise.
- **Ablation:** switching one component off to see whether it mattered.
- **Optuna:** a tool that searches for good hyperparameters (the weights of the loss terms) by trying many settings.

## 3. v1: what existed, and what was wrong with it

### 3.1 How v1 worked
Two notebooks plus the repo `Project-DiffShield/DiffShield`, which the notebook patched at run time with string replacements. Algorithm:
- PGD for 40 iterations, step 1/255 on the [-1,1] scale, EoT with Kornia's random JPEG and random resized crop.
- Loss `L = α·L_vis − β·L_sem − γ·L_str` (the optimizer maximizes L_vis and minimizes the other two):
  - `L_vis`: how far the SD1.5 VAE latent of the protected image is from that of the original (maximized). Idea: the editor starts from the latent, so a wrong latent means a wrong edit.
  - `L_sem`: 1 − cosine similarity between the CLIP image embedding of the protected photo and the text embedding of "a potted plant" (minimized). Idea: make the photo "look like a plant" to CLIP.
  - `L_str`: 1 − SSIM between original and protected (minimized): keeps the photo looking unchanged.
- Weights tuned by Optuna (30 trials): α = 0.2699475469647097, β = 0.9023875567667267, γ = 60.52658429696642.
- Data: CelebA-HQ faces; 4 calibration images (5319, 1715, 6385, 8932) from KMeans on face attributes; 70 disjoint evaluation images.
- v1's reported results: PSNR 41.911 ± 0.440, SSIM 0.97496 ± 0.0048, LPIPS(clean, protected) 0.2316 ± 0.063 (n = 70). In Phase 2, edit-LPIPS about 0.46 on 15 pairs (SD1.5 img2img, strength 0.55, one prompt, one seed). Those Phase 2 outputs were lost (an empty zip), so 0.46 can only be quoted, not recomputed.

### 3.2 What was wrong (and what v2 does about each)
| # | Problem in v1 | Why it matters | v2 fix |
|---|---|---|---|
| 1 | "Survives any scenario" cannot be proven; never tested against purification | an attacker can simply clean the noise off | train against purifiers; test on purifiers never used in training |
| 2 | No random-noise control | part of the effect might be just noise | uniform and Gaussian noise controls at the same budget (E1) |
| 3 | No identity metric: LPIPS measures *change*, not "is it still the same person" | a photo can change a lot and still be recognizable | ArcFace identity similarity and PSR |
| 4 | One editor, one prompt, one strength, one seed, 15 images | one lucky configuration proves nothing | 5 editors, several strengths and prompts, 3 seeds, 100 test images |
| 5 | Optuna rewarded visible change (worst-case LPIPS), not protection | tunes the wrong thing | Optuna maximizes PSR under a purifier mixture, with invisibility constraints |
| 6 | Attacks mostly one stage (the VAE encoder); the denoiser (UNet) untouched; `L_sem` uses the CLIP *image* encoder, which Stable Diffusion img2img never runs, so it may do nothing **(unverified, an ablation answers it)** | wasted effort | add a denoiser loss; ablate each term |
| 7 | One encoder, no ensemble, no loss ablation | cannot say what matters | optional SDXL-VAE ensemble; ablations E7 |
| 8 | Runtime string patches, unlogged seeds, hard to reproduce, clean images only | nobody can repeat it | patches became real commits; one config file; every run logs config, seed, git hash, library versions; tests |
| 9 | The budget was mis-labelled | numbers could not be compared fairly | all budgets on the 0–255 scale |

### 3.3 First finding (real, measured in Phase 0)
v1's "8/255" was on the [-1,1] scale. On the normal scale the real budget is **4/255**: in the reproduced protected PNGs the largest change of any pixel value is exactly 4, in all 70 images. v1's real step size was 0.5/255 (normal scale), not the 1/255 the brief assumed.

## 4. The ideas for improving it

1. **Measure first.** Build the evaluation harness, the noise control and the baselines *before* adding anything new, so any improvement can be measured.
2. **Put the attacker's cleanups inside the optimization** (EoT over purifiers): cheap ones (JPEG, down-up scaling, VAE round trip) plus a **one-step diffusion purifier** (a cheap stand-in for DiffPure).
3. **Attack the denoiser too** (`L_den`): maximize the UNet's noise-prediction error on the purified image, not only the VAE latent shift.
4. **Anti-purification terms** (AntiPure idea): push the one-step reconstruction toward high-frequency content, and keep the reconstruction error growing with the noise level.
5. **Encoder ensemble** (SD1.5 VAE + SDXL VAE) so the noise transfers across encoders.
6. **Identity loss** (`L_id`, optional): minimize ArcFace similarity between the source face and the face recovered through the VAE.
7. **Tune for protection, not visible change**, on held-out validation images.
8. **Test honestly:** held-out purifiers and editors, an **adaptive attacker** that knows the method and trains a small network to remove it (the LightShed idea), an ε sweep, ablations, a small human study, and a list of failure cases.
9. **Compare with existing tools:** PhotoGuard and Mist, plus the noise controls.

## 5. The new architecture (v2) and how it works

### 5.1 Two halves that never share components
**Half A, the protection loop (offline, makes the protected photo).** For one photo x:
1. Start with a random small noise δ inside the ±ε ball.
2. Repeat 40 times:
   - Draw K = 4 samples; for each, pick a purifier P from a training mixture (JPEG, down-up, VAE round trip, one-step diffusion purifier) and apply it to the current published image x + δ.
   - Score the purified image: `L_vis` (latent shift), `L_den` (denoiser error), `−β·L_sem` (kept only if the ablation shows it helps), optional anti-purification and ensemble terms. Score `L_str` (visibility) on x + δ itself, the *published* image, not the purified one.
   - Average the K gradients; step δ by the sign of the gradient; clip δ to ±ε and to valid pixel range.
3. Round to the 8-bit grid, so the saved PNG differs from the original by at most ε gray levels.

In one line: `δ* = argmax over ‖δ‖∞ ≤ ε of  E over purifiers P and randomness [ α·L_vis + η·L_den − β·L_sem ]  −  γ·L_str`.

**Half B, the attack-and-score harness (measures every claim).** Takes a protected image, applies **held-out** purifiers (never used in training), edits it with several editors, and scores with ArcFace identity retention, PSR, CLIP-I, edit-LPIPS, plus the invisibility numbers. Reasoning: if the attacker's cleanups are the ones we trained against, the test proves nothing; held-out ones show whether the noise generalizes.

**Training purifiers (Half A):** JPEG (random quality), down-up (random scale), SD VAE round trip, one-step SD purifier (add noise at a random timestep in [50, 300], predict noise with the frozen UNet, recover the one-step estimate, decode). One down-up setting is held out as an early check.
**Held-out purifiers (Half B only):** JPEG q90/75/50 and another JPEG setting, down-up 0.5× and another ratio, VAE round trip, DiffPure at t* = 0.05/0.1/0.2/0.3 (multi-step), Swin2SR restoration. GrIDPure was planned but its model file is unavailable (section 8); it will be reported as a limitation. IMPRESS is not run.

**Editors (Half B):** SD1.5 img2img (strengths 0.3, 0.55, 0.75), SDXL img2img, InstructPix2Pix, SD inpainting with masks that keep the face, IP-Adapter FaceID. All use a DPM-Solver++ 30-step sampler, 3 seeds, prompts rotated per image except the headline setting (SD1.5, strength 0.55) which uses all 3 prompts. Edits are saved as JPEG q90; edit-LPIPS is computed afterwards.

### 5.2 Methods (names in the code)
`v1` (the reference), `v2_cheap` (cheap purifiers in the loop), `v2_main` (+ one-step purifier + `L_den`), `v2_antipure` (+ anti-purification terms), `v2_final` (Optuna-tuned, with the SDXL-VAE ensemble), ablations `abl_vis_only`, `abl_den_only`, `abl_no_sem` (both phases without `L_sem`), `abl_no_pur` (no purifier in the loop; `v2_main` plays "both"), ε sweep `eps2`, `eps6`, `eps8` (ε = 4 is `v2_final`), controls `noise_uniform`, `noise_gauss_psnr`, baselines `photoguard`, `mist`. PhotoGuard (official 1000 iterations) and Mist are re-implementations checked against the official code (MadryLab/photoguard@686bea7, psyker-team/mist@99f5f3c), run at **our** ε = 4/255; Mist's official target image is downloaded at run time.

### 5.3 Experiments
| ID | What | Question |
|---|---|---|
| E0 | clean photos through all editors | reference edits; the PSR threshold |
| E1 | random noise at the same budget | how much is just noise? |
| E2 | PhotoGuard and Mist | how does it compare with existing tools? |
| E3 | v1 through each purifier | how badly does purification break v1? |
| E4 | v2_cheap | does it fix JPEG, down-up, VAE round trip? |
| E5 | v2_main, tested on held-out DiffPure etc. | does it transfer to unseen purifiers? |
| E6 | v2_antipure | more robust at stronger purification? |
| E7 | ablations | which loss term / which phase matters? |
| E8 | adaptive attacker | what breaks it? |
| E9 | ε = 2, 4, 6, 8 | invisibility vs robustness |
| E10 | human study (target 15 raters × 20 pairs; reported as a pilot if fewer than 10) | is it invisible; is the identity gone? |

Gates (suggestions; if missed, report honestly and ask; never change quietly): after E4, PSR under JPEG/down-up/VAE round trip at least doubles vs v1 at PSNR ≥ 38; after E5, PSR under held-out DiffPure t* = 0.1 reaches at least half of the no-purifier PSR; invisibility PSNR ≥ 38, SSIM ≥ 0.95 and human 2AFC not far above 50%.

### 5.4 What may be claimed
"Under a stated threat model (white-box SD encoders, listed editors and purifiers, ε = X/255), DiffShield lowers ArcFace identity retention of edits by A points vs a noise control and B points vs v1, including after [held-out purifiers]; an adaptive attacker reduces this to C." Not claimable: survives any scenario, immunity to purification, protection against editors or attacks not tested. Inpainting limit: if an attacker masks the face itself, those pixels are discarded, so noise there cannot protect them. Expected outcome (a prediction, not a result): cheap purifiers largely fixed, partial transfer to low-strength DiffPure, weak protection against strong purification, restoration-based purifiers and adaptive attackers. Partial but measured results count as a good outcome.

## 6. How it is implemented

### 6.1 Repository (`C:\Users\karth\Documents\DiffShield`; the main working branch is `v2/integration` in the worktree `...\DiffShield-wt\integration`; nothing has been pushed)
- `config/config.yaml` + `config/parts/*.yaml`: every hyperparameter, seed and path; merged in sorted order; `--set a.b=value` overrides.
- `src/protect/`: Half A (`registry`, `v1` adapter, `models`, `losses`, `purifiers`, `pgd`, `antipure`, `ensemble`, `face_loss`, `v2_registry`). Frozen interface: `Protector.protect(x[1,3,H,W] in [-1,1], seed) -> (image, info)`.
- `src/eval/`: Half B (`editors`, `purify_attacks`, `faces`, `clip_metric`, `masks`, `edit_runner`, `analysis`, `metrics`, `stats`, `adaptive_ae`).
- `src/baselines/`: noise controls, PhotoGuard, Mist.
- `scripts/`: `run_protect`, `evaluate`, `tune_optuna`, `make_splits`, `prepare_data`, `run_job`, `ingest_results`, `compute_edit_lpips`, `make_tables`, `make_figures`, `make_human_study`, `analyze_human_study`, `demo`, `compare_runs`, `budget_check`, `smoke_report`, and more.
- `kaggle/`: `jobs.yaml` (every session defined as a list of steps), `job_runner.ipynb` (the human sets `JOB_IDS`), `build_bundle.py` (zips the code for upload).
- `tests/`: 305 CPU tests with tiny stand-in models. Real weights are exercised only on Kaggle.
- `data/splits/`: fixed photo lists (committed). Results go under `results/` (images are git-ignored: identifiable faces, research use only).

### 6.2 Compute
No GPU on the local PC; all GPU work runs on **Kaggle** (2 × Tesla T4, 5 accounts; acct1 and acct2 are the user's, acct3–5 are teammates and launch from 09:00). Quota is billed by **wall-clock** time (checked: acct1 showed about 1.7 h used after about 1.7 h of 2-GPU sessions). The code is packaged as a zip uploaded as a private Kaggle dataset; each session is the same notebook with a different `JOB_IDS`. Sessions are committed ("Save & Run All") so they run unattended for up to 12 h; the runner stops starting new work after 11 h and can resume with `RESUME_FROM`. Outputs return as `<SESSION>_outputs.zip`.

### 6.3 Data
CelebA-HQ via the Kaggle dataset `ipythonx/celebamaskhq`. Fixed lists: **test 100** (all 70 of v1's evaluation images + 30 new, so paired comparison with v1 works), **messy 20** (pose, blur, lighting, occlusion, pre-compressed), **train 20**, **val 20**, **adaptive-attacker pool 120**; v1's 4 calibration images are excluded. No identity labels exist, so "same person" de-duplication uses ArcFace similarity ≥ 0.5 (**threshold unverified**; and the earlier detector missed faces in 38 of the 100 test photos, so de-duplication there was incomplete; to be rechecked with the corrected detector).

### 6.4 Reliability rules built in
Never invent a number; every run logs its config, seed, git hash and library versions; per-image seeds derive from a hash (so runs are repeatable and resumable); work on copies and never delete; no credentials in code; smoke-test everything on 5 images before a long run.

## 7. The 48-hour plan and where we are

Deadline: **16:30 on 2026-10-16** (48 h from 16:30 on 2026-10-14). The full v2 plan (phases 0–8, normally about 10 weeks of work) was compressed into 48 hours by running everything in parallel on 5 Kaggle accounts.

Sessions (each = one notebook run): S01–S05 smoke/validation; **S06** Optuna tuning (3 h cap) then protect `v2_final`; **S07** clean-photo reference edits and PSR thresholds; **S08** v1 protection and edits; **S09** `v2_main`; **S10** `v2_cheap`; **S11** and **S14** ablations and noise controls; **S12** `v2_antipure`; **S13** PhotoGuard and Mist; **S15** adaptive attacker; **S16** `v2_final` edits; **S17**, **S18** ε sweep. Launch sheets: `docs/KAGGLE_LAUNCH.md`, `docs/LAUNCH_DAY2.md`.

Scale-downs decided (each must appear as a limitation in the report): ablations and the ε sweep use the **first 50 test photos** (paired with the main methods on the same 50), not 100; Optuna 16 trials × 10 validation photos with a proxy objective (not 30–50 × 20), possibly fewer trials within its 3 h cap; 40 PGD iterations and K = 4 (no 100 iterations or K = 8); DPM-Solver++ sampler for all editors (not v1's default); SDXL only for some methods; GrIDPure unavailable; IMPRESS and InstantID not run; single CLIP (no CLIP ensemble); the adaptive attacker is weak (120 photos, 128 px patches, about 1 GPU-hour of training), so it gives a lower bound on the damage; not every method × purifier pair is measured.

Current status (2026-10-15 early morning IST):
- Done: Phase 0 (budget confirmed, v1 reproduced); photo lists frozen; all validation sessions passed after fixes.
- **Running now:** S06 (restarted 05:23 IST, ends about 15:20 IST), S07, S08, S09 (end between about 10:00 and 15:00 IST).
- Next: acct3–5 launch S10–S14 from 09:00 IST; after S06 ends, launch S15, S16, S17, S18 (about 15:30 IST, finishing about 22:30 IST). Then: download outputs, compute edit-LPIPS locally, ingest, make tables and figures, human study (needs v2_final edits), failure cases, report, demo script, tag `v2.0-final` (ask before pushing anything).

## 8. What we learned while validating (all real, from Kaggle runs; useful for the report's honest-limitations section)

Measured costs (T4):
- v1: 45.6 GPU-seconds per image (plan: 25).
- v2 initially cost 2–6× the planner's estimate, because the VAE ran in full precision. Running the VAE in fp16 "autocast" (the option `runtime.vae_dtype_protect`, now the default) made it about 1.5–2.4× faster; on 4 images × 10 iterations the loss values matched full precision closely, with no NaN or zero gradients. At 40 iterations: `v2_cheap` about 146 s, `v2_main` about 263 s, `v2_final` about 395 s per image (S07 later showed `v2_antipure` at about 250–270 s, faster than estimated).
- Edits: SD1.5 about 3.3 s, InstructPix2Pix about 7.5 s, inpainting about 5.5 s, FaceID about 3.4 s, SDXL about 18.7 s (single seed, 5 images; 1.3–1.7× the plan).
- Whole plan: about 195 GPU-hours (instead of 131); fits in the quota because billing is wall-clock.

Bugs found and fixed (each now has a regression test):
1. PhotoGuard/Mist crashed because they did not tell the runner which GPU they were on.
2. CLIP scoring crashed under the newer `transformers` (returns a wrapper object, not a tensor).
3. One missing GrIDPure file aborted the whole job; now unavailable purifiers are skipped and logged.
4. **Face detection missed 38 of the 100 test photos** (tightly cropped faces); fixed by padding the image before detection (5 of 5 detected afterwards). This would otherwise have silently corrupted every identity number.
5. The GPU build of `onnxruntime` needed CUDA 13; pinned to a CUDA-12 build (version 1.22.0), now running on the GPU.
6. Two GPU processes downloading the same face model raced; one loaded a half-unpacked model and produced garbage embeddings; now a file lock plus a check.
7. IP-Adapter FaceID could not load because of an old preinstalled `torchao`; removed in the setup step.
8. One failing editor aborted all later editors; now failures are isolated and recorded.
9. `--limit N` picked different images in the protection and evaluation scripts (sorted vs list order); both now sort identically.
10. Two Optuna workers racing to create the database killed one worker; study creation now retries.

## 9. Measured results so far (Phase 0 only; n = 70 images, v1 method)
Reproduction of v1's invisibility on the 70 original images (published-PNG convention in brackets): PSNR 41.986 ± 0.416 (42.171), SSIM 0.9754 ± 0.0054 (0.9756), LPIPS 0.2314 ± 0.0637 (0.2305); paired differences vs v1's own numbers: +0.075 dB, +0.0004, −0.0002, all inside the accepted tolerance (±0.5 dB, ±0.005, ±0.03). 0 of 70 images below PSNR 38 or SSIM 0.95. Max pixel change exactly 4 gray levels. Not claimable: that "a second person reproduced it" (the optional second run was not done).

**There are no v2 protection-success numbers yet.** Smoke-test values (5 images, few iterations) must never be reported.

## 10. Honesty rules (binding on anyone who continues)
Never invent a number, citation or result; label unverified things; no "RSAS" method; report negative results; every number needs n, mean ± 95% CI, 3 seeds and a paired comparison; never tune on the test photos; no deletions; never ask for or store passwords or tokens; do not redistribute CelebA-HQ or commit identifiable images; do not push without asking.

## 11. Sources named in the project brief (look up the official versions before citing)
DiffPure; GrIDPure (Zhao et al.); IMPRESS; DiffAttack; AntiPure; LightShed (USENIX Security 2025); Liu et al. diffusion purification benchmark; FastProtect (Ahn et al., CVPR 2025); PhotoGuard; Mist. Do not cite details from memory.

## 12. Where to read more in the repo
`docs/HANDOFF_v2.txt` (the original team brief), `docs/V2_PLAN_48H.md` (the compressed plan), `docs/V2_PLAN.md` (module designs), `docs/COST_MEASURED.md` (measured costs), `docs/KAGGLE_LAUNCH.md` and `docs/LAUNCH_DAY2.md` (what to launch), `reports/PHASE_0.md` (Phase 0 numbers), `docs/REPORT_DRAFT.md` (report skeleton), `docs/LEARNING_HANDOVER.md` (a shorter teaching brief).
