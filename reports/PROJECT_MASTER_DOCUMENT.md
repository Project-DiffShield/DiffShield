# Comprehensive Project Master Document: Technical Details, Implementation, and Experimental Results

**Project:** DiffShield v2: protecting face photos from diffusion-model editing
**Purpose of this document:** a self-contained technical record of the whole project (problem, objectives, background, architecture, algorithms, implementation, parameters, data, experiments, every numerical result, analysis, limitations, and open items), written so that another person or AI with no access to the working conversation can write an accurate academic paper from it. It is **not** the paper.
**Date assembled:** 2026-10-25. **Source of truth:** the project repository (branch `v2/release` of `Project-DiffShield/DiffShield`, local path `C:\Users\karth\Documents\DiffShield-wt\integration`), its generated result tables under `results/tables/`, and the working-conversation record summarised in Section 21. Sections were drafted by separate reading passes over the code and result files and then merged; each section states its own sources.

## How to read this document and its labels
- **IMPLEMENTED AND RUN**: code exists and results from it are reported. **IMPLEMENTED BUT NOT RUN**: code exists, no reported result. **PROPOSED ONLY**: appears in a plan, no code. **DROPPED**: planned and deliberately not done.
- **SUPPORTED BY DATA / HYPOTHESIS / NOT TESTED** (Section 22) separate measured results from interpretation.
- "UNKNOWN / not recorded" means the project files do not contain the information; nothing was guessed.
- Every number is copied from a file whose path is given next to it. Where two files disagree, both values and the discrepancy are given.
- Face images are never reproduced; figures that contain faces are described only.
- No citation in this document is verified. Prior work is named only as the project files name it, with "citation details not verified".

## Register of known inconsistencies and corrections (read before using any number)
These were found while assembling this document. They affect other deliverables already produced (report, slides, website) and are the first things a paper author should resolve.

| # | Issue | Details | Where it appears |
|---|---|---|---|
| 1 | **Twelve cleanups evaluated, not thirteen** | The configured list has 13 names but GrIDPure's checkpoint was unavailable and its jobs were skipped. Not all twelve are strictly unseen: JPEG (q90/75/50/65), down-up 0.5 and the VAE round trip overlap the training mixture; strictly unseen are down-up 0.375 (Lanczos) and the multi-step DiffPure-style cleanups (which share SD1.5 weights with the one-step training purifier). | `reports/FINAL_REPORT.md`, the slide deck, the website say "13 held-out cleanups". Section 15.4 gives the exact list. |
| 2 | **`v2_tgt_zero_mid` was not screened on validation photos** | Screened on the 20 validation photos: `v2_tgt_zero`, `v2_tgt_zero_hi`, `v2_tgt_gray` (plus the control `v2_cheap`). `mid` went straight to the test run with the other two declared variants. | `FINAL_REPORT.md` section 5 implies all three were screened. `config/parts/protect.yaml` states the truth. |
| 3 | **Bootstrap resamples** | Generated tables (`results/tables/*.md`, `config/config.yaml`): 10,000. `scripts/readout_final.py` (source of most numbers in `FINAL_REPORT.md` 7.3 to 7.9) and `docs/EVAL_PLAN_FINAL.md`: 2,000. Interval endpoints can differ in the third decimal (e.g. mid vs v1: upper end +0.298 vs +0.299). | Section 15.7 |
| 4 | **Optuna photo counts** | The search used the first 10 validation photos; the confirm step used all 20 (the 10 are inside the 20). `FINAL_REPORT.md` 7.1 says "20 photos" for the constraint check. The PSR threshold used inside tuning was not recorded. | Section 18 |
| 5 | **Test-count figures** | The last full test run in the working conversation was 377 passed, 8 skipped (also stated in `README.md` and `FINAL_REPORT.md`). Files in the repo record older counts (32; 296+1 skipped; 297+1 skipped; `docs/PROJECT_GUIDE.md` says 305). Pytest counts parametrised cases, so the number of test functions (223 in 40 files at assembly time) is smaller. Re-run `python -m pytest -q tests` before quoting a number. | Section 13 |
| 6 | **PhotoGuard and Mist are re-implementations** | Draft text calls them "official"; the tables label them re-implementations (checked against the official code at the commits cited in `config/parts/baselines.yaml`) run at our budget 4/255, not their published budgets. | Section 15.9 |
| 7 | **Edit-LPIPS** | Not reported. The local cache holds 512 pairs (the report says 513); 22,560 further rows have values computed on Kaggle for a subset of methods only. | Section 16.6 |
| 8 | **Invisibility numbers** | The `png_ssim` (published PNG) and the `inv_ssim` in the edit rows use different SSIM implementations; the reported invisibility tables use the published-PNG values. | Section 9, 16.4 |
| 9 | **Face-containing files on the public GitHub master** | `results/figures/before_after.pdf` and `results/figures/failure_gallery.pdf` contain CelebA-HQ faces and are on `origin/master` (found 2026-10-25; not yet removed). Git history was rewritten on 2026-10-23 to remove 84 face-derived JPEGs; the old-to-new hash table is `reports/HASH_MAP.md` (the backup tag name carries 20261022). | Section 21 |
| 10 | **Step size** | v1 uses a 0.5/255 step; every v2 preset uses 1.0/255. No ablation isolates this difference. | Section 9, 11 |
| 11 | **Unrun items** | `J01b` (second-person reproduction), `S14`, `S23` (eps 6), `S20`, `E10`, `no_sem` and `no_pur` ablations, GrIDPure, IMPRESS, InstantID, a human study, an adaptive attacker against the target-steered variants. | Section 25 |

## Table of contents
- 1. Executive overview
- 2. Problem statement and motivation
- 3. Objectives and success criteria
- 4. Threat model and scope
- 5. Theoretical and technical background
- 6. DiffShield v1: baseline method, its results, and the problems found
- 7. The v2 plan as written versus what was actually carried out
- 8. System architecture
- 9. Algorithms
- 10. Implementation details
- 11. Parameters and configuration
- 12. Tools, libraries, models and environment
- 13. Testing and software quality
- 14. Datasets and data splits
- 15. Experimental protocol
- 16. Metric definitions
- 17. Experiment catalogue
- 18. Hyperparameter tuning (Optuna, S06)
- 19. Compute, cost and reproducibility
- 20. Complete numerical results
- 21. Project evolution
- 22. Analysis and interpretation of results
- 23. Contributions
- 24. Limitations, risks to validity and uncertainties
- 25. What was NOT implemented or NOT run
- 26. Future work
- 27. Claims a paper can and cannot make
- 28. Artifact map and glossary
- 29. Open questions and missing information

---

# DiffShield v2 - Project Master Document, PART 1

**Scope of this part:** sections 1 to 7 (overview, problem, objectives, threat model, background, DiffShield v1, v2 plan versus what was done). Other parts of the master document cover the v2 method internals, experiments in full, results tables, and the conference-paper guidance.

**Source of truth:** the repository `C:\Users\karth\Documents\DiffShield-wt\integration` (branch `v2/release`; last commit seen: `22d3f8c`, 2026-10-23). All paths below are relative to that folder unless they start with a drive letter. Facts that exist only in the working conversation are taken from `HISTORY_FACTS.md` (scratchpad) and are marked "(conversation)".

**Reading conventions used throughout this part**

- Status labels: **IMPLEMENTED AND RUN** (code exists and result files exist), **IMPLEMENTED BUT NOT RUN** (code exists, no result rows), **PROPOSED ONLY** (written in a plan, no code), **DROPPED** (planned and deliberately not done, or replaced).
- Every number is copied from a file; the file is named next to it. "n" is the number of photos unless said otherwise; "CI" is the 95% bootstrap interval over photos; "3 seeds" means 3 edit seeds averaged within a photo before bootstrapping.
- Prior-work names (PhotoGuard, Mist, DiffPure, ...) are written exactly as the project files write them. Any reference is marked **"citation details not verified; look up before citing"**. No author, venue or year is given here unless a project file gives it, and then the file is named.
- Where two project files disagree, the disagreement is stated (see the "Discrepancies noticed" list at the end of section 7).

---

## 1. Executive overview

### 1.1 What the project is

DiffShield is a **defensive research project** (Python, PyTorch, Hugging Face diffusers; all GPU work on Kaggle T4 x2). A person who posts a face photo can protect it before posting: the tool adds an almost invisible, L-infinity-bounded perturbation, found by projected gradient descent (PGD), to the pixels. The aim is that when someone later feeds the *protected* photo to a diffusion-model editor (Stable Diffusion image-to-image, InstructPix2Pix, inpainting, an identity-conditioned editor), the *edit* no longer resembles the person, while the protected photo itself looks identical to the original. Source: `README.md`, `reports/FINAL_REPORT.md` section 1.1. It protects people's own photos; it is not meant to create fakes.

The project exists in two generations:

- **v1** (a student course project, the "Tri-Modal" PGD method; repo `Project-DiffShield/DiffShield`): a single loss with three terms, one editor, 15 evaluation edits, no controls. Evidence: `docs/HANDOFF_v2.txt`, `docs/PROJECT_GUIDE.md` section 3, `main.pdf` (the v1 project-review slide deck, title "DiffShield: A Tri-Modal Adversarial Framework for Robust Immunization Against Deepfake Generation").
- **v2** (the work documented here, Oct 2026): a rebuilt, reproducible code base plus a *fair evaluation harness*, plus new protection variants that train against "purification" (cleanup) steps, plus a late-added **target-steered** variant.

### 1.2 What was built (all in the repository; counts as stated by the project)

- A config-driven code base (`config/config.yaml` + `config/parts/*.yaml`), per-image hashed seeds, run logging (config, git hash, library versions), resumable and sharded jobs, and CPU tests (reported as **377 passed, 8 skipped** in `reports/FINAL_REPORT.md` Appendix A and `README.md`; the test count was not re-run for this document).
- **Half A (protection):** `src/protect/` - v1 adapter, a PGD loop with Expectation over Transformation (EoT) over a mixture of differentiable "purifiers" (JPEG, down-up resize, VAE round trip, one-step Stable Diffusion purifier), a denoiser-error term, anti-purification terms, an SDXL-VAE ensemble, an optional identity term, and the target-steered term.
- **Half B (attack-and-score harness):** `src/eval/` - five editors, twelve evaluated cleanups (a thirteenth, GrIDPure, was coded but its model file was unavailable), ArcFace face scoring (insightface `buffalo_l`), CLIP-I, LPIPS, protection success rate (PSR) with a threshold derived from clean edits, paired bootstrap statistics, gates, tables and figures.
- Baselines: uniform and Gaussian noise controls, and re-implementations of PhotoGuard and Mist (`src/baselines/`), checked against the official code by the project (see `config/parts/baselines.yaml` for the commit hashes the project recorded).
- An adaptive attacker (a small autoencoder trained to undo the protection; `src/eval/adaptive_ae.py`).
- About 30 Kaggle sessions (`kaggle/jobs.yaml`), spread over several accounts (`reports/FINAL_REPORT.md` section 4.5).

### 1.3 Headline findings (all from `reports/FINAL_REPORT.md`; CI = 95% bootstrap over photos; n = 100 test CelebA-HQ photos; 3 edit seeds; SD1.5 image-to-image, strength 0.55, no cleanup unless stated)

| Finding | Number | Where it comes from |
|---|---|---|
| v1's "8/255" budget was on [-1,1] tensors; the real budget is **4/255** | max pixel change exactly 4 in 70/70 reproduced images | `reports/PHASE_0.md` (budget table, J01) |
| v1's invisibility numbers reproduce | PSNR 41.986 +- 0.416, SSIM 0.9754 +- 0.0054, LPIPS 0.2314 +- 0.0637 (n = 70) | `reports/PHASE_0.md` |
| Noise alone does nothing | PSR clean 13.9%; uniform noise 14.9% (+1.0 over clean [-1.0, +3.1]); PSNR-matched Gaussian 14.7% (+0.8 [-1.2, +2.9]) | `reports/FINAL_REPORT.md` 7.2 |
| v1's protection effect is real but modest | PSR 24.0% (+10.1 points over clean [+7.2, +13.0]) | same |
| The original v2 idea (cleanups in the loop, denoiser term) adds almost nothing over v1 | `v2_main` minus v1 = +0.3 points [-1.8, +2.6] with no cleanup | `reports/FINAL_REPORT.md` 7.4 |
| The Optuna-tuned `v2_final` is better than v1 but breaks the invisibility rule | +5.4 points over v1 [+2.4, +8.7]; SSIM < 0.95 on 29/100 photos | `reports/FINAL_REPORT.md` 7.1, 7.4 |
| The target-steered variant `v2_tgt_zero_mid` roughly doubles v1 while every photo stays invisible | PSR 49.7% vs v1 24.0%; paired +25.7 points [+21.8, +29.8] (readout table gives +0.257 [+0.218, +0.299]); 0/100 photos violate PSNR >= 38 / SSIM >= 0.95 | `reports/FINAL_REPORT.md` 7.9; `results/tables/final_readout.md` |
| PhotoGuard and Mist (our re-implementations, run at our 4/255) protect more without cleanup but are visibly altered | PSR 81.0% and 63.1%; PSNR 37.15 / 38.00, SSIM 0.906 / 0.923; fail the per-photo rule on 100/100 (PhotoGuard) and 100/100 SSIM, 50/100 PSNR (Mist) | `reports/FINAL_REPORT.md` 7.1, 7.8 |
| Under JPEG and resizing the target-steered variant is ahead of the visible tools | JPEG75: 23.3% vs PhotoGuard 18.3%, paired +5.0 [+1.3, +9.0]; down-up 0.5: 19.0% vs 13.7%, +5.3 [+2.7, +8.3] | `reports/FINAL_REPORT.md` 7.8 |
| Under the VAE round trip and DiffPure-style 0.1 the visible tools stay ahead | mid minus PhotoGuard: -6.7 and -15.7 points | same |
| An adaptive autoencoder attacker removes the protection of v1 and `v2_final` | PSR 9.3% clean, 11.0% v1 (net +1.7 [-1.0, +4.7]), 10.7% `v2_final` (net +1.3 [-1.0, +3.7]); attacker is weak by construction; **not run** against the target-steered variants | `reports/FINAL_REPORT.md` 7.6 |
| Protection does not extend to FaceID, inpainting, SDXL, or strength 0.3 | PSR about 0 for all methods except the PhotoGuard/Mist cells shown in the report | `reports/FINAL_REPORT.md` 7.7 |
| No human study was run | - | `results/tables/e10.md`, `reports/FINAL_REPORT.md` section 10 |

The report's own one-sentence verdict (`reports/FINAL_REPORT.md` section 9): v2 as originally designed (cleanups and extra terms) added little over v1; adding a target-steered term produced a measurable, supported improvement (about twice v1's protection at invisible levels for SD1.5 image-to-image editing), with clear limits (JPEG, adaptive attackers, FaceID/inpainting/SDXL, human perception untested).

**Honest caveats the project itself states** (`reports/FINAL_REPORT.md` sections 7.9, 7.10, 10): the choice of `mid` as the best target variant was read from the test table (post hoc among three variants declared in advance); PSR is an ArcFace-based operational definition and the protected edits are often *degraded or stylised* rather than turned into a different believable person; many comparisons are not corrected for multiplicity; results are for CelebA-HQ portraits only.

---

## 2. Problem statement and motivation

**The harm.** Anyone who can download a face photo can pass it to a diffusion-model editor with a prompt such as "the same person in a suit" or "at a beach" and obtain a realistic altered picture. Image-to-image editors, instruction-following editors (InstructPix2Pix), inpainting editors and identity-conditioned editors (IP-Adapter FaceID) all make this cheap. Source for the framing: `docs/PROJECT_GUIDE.md` section 1, `reports/FINAL_REPORT.md` section 1.1.

**The defensive idea ("immunisation").** Because the editor's first step is to *encode* the photo into a latent representation with a VAE encoder (see section 5), a small change to the pixels that moves that latent a long way can derail everything downstream. The change is optimised offline by the photo owner using white-box access to the open Stable Diffusion weights. The perturbation is bounded (L-infinity <= epsilon) so the photo looks unchanged.

**Why v1 was not enough** (details in section 6): its claims were not backed by controls or by an identity metric; its stated budget was mislabelled; it was never tested against the obvious counter-move, *purification* (an attacker cleans the noise off before editing: JPEG, resizing, VAE round trip, diffusion-based denoising, restoration networks); it was evaluated on one editor, one prompt, one strength, one seed and 15 edits; and it was hard to reproduce (runtime string patches of library files, unlogged seeds). Source: `docs/HANDOFF_v2.txt` ("What is wrong with v1"), `reports/FINAL_REPORT.md` section 3.2.

**Research questions that follow** (as the project phrased them, `docs/HANDOFF_v2.txt` experiment table and `docs/EVAL_PLAN_FINAL.md` section C):

1. Is the protection effect real, i.e. distinguishable from random noise of the same size? (E1)
2. How does it compare with existing public tools (PhotoGuard, Mist)? (E2)
3. How badly does purification break v1? (E3) Can purification-aware training (cheap purifiers, a one-step diffusion purifier, a denoiser-error term, anti-purification terms) fix it? (E4 to E7)
4. Does it survive an adaptive attacker who knows the method? (E8)
5. What is the trade-off between invisibility and robustness as epsilon grows? (E9)
6. Is it really invisible to people, and is the identity really gone for people? (E10, **not run**)

**Why the evaluation design is part of the contribution.** The brief insists that "without a control, an identity metric and a purifier baseline nobody can tell whether a change helped" (`docs/HANDOFF_v2.txt`, operating rule 1). The project therefore built the measurement harness *before* new loss terms ("measure before you build") and kept it separated from the protection code ("two halves that never share components"; enforced by an import-isolation test, `tests/test_isolation.py`, described in `docs/V2_PLAN.md` section 3.9).

---

## 3. Objectives and success criteria

### 3.1 The final report's goals G1 to G6 (verbatim content of `reports/FINAL_REPORT.md` section 1.2, with the verdicts of its section 9)

| # | Objective | What would count as success (as written) | Verdict in `FINAL_REPORT.md` section 9 |
|---|---|---|---|
| G1 | **Invisibility**: the protected photo looks identical to the original | PSNR >= 38 dB and SSIM >= 0.95 on **every** photo; maximum pixel change = the stated budget | "achieved" for the recommended variants: v1, `v2_main`, `v2_cheap`, `v2_tgt_zero`, `v2_tgt_zero_mid` pass on 100/100; `v2_final`, antipure, eps = 8, PhotoGuard, Mist do not |
| G2 | **Protection**: edits of the protected photo stop looking like the person | higher PSR than clean photos, than a random-noise control and than v1, with 95% CIs that exclude zero | "achieved in one regime; modest elsewhere": v1 +10 points, `v2_tgt_zero_mid` +36 points over clean (about 2x v1), SD1.5 image-to-image at moderate strength |
| G3 | **Robustness to cleanup**: protection survives an attacker cleaning the noise first | protection remains after JPEG, resizing, VAE round trip, DiffPure-style denoising, restoration networks, none of which the method was trained on (held-out) | "partial": survives DiffPure and the VAE round trip; reduced by JPEG and resizing; removed by an adaptive attacker (tested on v1 and `v2_final`) |
| G4 | **Generality**: works across editors and edit strengths | measured per editor (SD1.5, SDXL, InstructPix2Pix, inpainting, IP-Adapter FaceID) and per strength | "not achieved": does not protect FaceID, inpainting, SDXL or strength 0.3 |
| G5 | **Honest, fair evaluation** | controls, paired comparisons, fixed train/validation/test lists, held-out cleanups, adaptive-attacker stress test, comparison with PhotoGuard and Mist, every number with n, CI and seeds | "achieved" |
| G6 | **Reproducibility** | one config, logged seeds/versions/git hash, tests, one command per analysis step | "achieved" (one config, per-image seeds, logged environments, 377 tests, hash-stamped bundles) |

### 3.2 The original v2 plan's objectives (from `docs/HANDOFF_v2.txt`, the "brief", and `docs/V2_PLAN.md`)

**"What done looks like"** (HANDOFF_v2.txt, Mission section): (1) a versioned repo that reproduces every number with one command per step; (2) result tables and figures for experiments E0 to E10 with confidence intervals; (3) a short threat-model-first report section and an honest list of failure cases; (4) a small demo script that protects an image and shows before/after edits.

**Phase goals and "done when" conditions** (HANDOFF_v2.txt "Work plan by phase"):

- Phase 0: reproducible base; another person reproduces v1's Phase 1 numbers (PSNR about 41.9, SSIM about 0.975); budget check (expected max pixel change 4).
- Phase 1: evaluation harness: 100 CelebA-HQ test identities + 20 messy + 20 train + 20 validation; five editors (SD1.5 img2img at strengths 0.3/0.55/0.75, SDXL img2img, InstructPix2Pix, SD inpainting with keep-face masks, one identity-conditioned editor); three prompts, three seeds; ArcFace similarity and PSR, LPIPS, CLIP-I, PSNR/SSIM/LPIPS of the protected image; bootstrap CIs and paired differences; clean edits cached.
- Phase 2: baselines and controls: uniform noise at the same L-infinity budget and Gaussian noise matched to the same PSNR; v1 under purification (JPEG q90/75/50, down-up 0.5, SD VAE round trip, DiffPure-style t* = 0.05/0.1/0.2, GrIDPure if feasible); PhotoGuard and Mist. Decision rule: if v1 barely drops under purification, skip Phase 5.
- Phase 3: cheap purifiers in the loop (Kornia JPEG, bicubic down-up, SD VAE round trip), K = 4 draws (8 if memory allows), one down-up variant held out.
- Phase 4: one-step SD purifier (probability about 0.3) and the denoiser-error term `L_den`.
- Phase 5 (optional): anti-purification terms.
- Phase 6: encoder ensemble (SD1.5 VAE + SDXL VAE), identity term `L_id`, Optuna re-tuning (30 to 50 trials x 20 validation images, objective = mean PSR under the purifier mixture with invisibility constraints), randomisation of the perturbation.
- Phase 7: final experiments E0 to E10, epsilon sweep 2/4/6/8, adaptive attacker (LightShed idea), small human study (15 raters, 20 image pairs), phase ablation (four variants).
- Phase 8: write-up (threat model on page one), demo script, README that reproduces the main table.

**Pre-set gates** (HANDOFF_v2.txt "Pass and fail gates"; "suggestions - if one is missed, report and ask, do not quietly change a gate"): after E3, if v1 does not degrade under purification skip Phase 5; after E4, PSR under JPEG/down-up/VAE round trip at least doubles against v1 with PSNR >= 38; after E5, PSR under held-out DiffPure t* = 0.1 reaches at least half of the no-purifier PSR; every experiment >= 100 identities, 3 seeds, mean +- 95% CI, paired test vs v1; invisibility PSNR >= 38, SSIM >= 0.95 and human 2AFC accuracy "not far above 50%".

**Claim template** (HANDOFF_v2.txt, "What to claim"): "Under a stated threat model (white-box SD encoders, listed editors and purifiers, epsilon = X/255), DiffShield lowers ArcFace identity retention of edits by A points versus a noise control and B points over v1, including after [held-out purifiers]; an adaptive attacker reduces this to C." Explicitly **not** to be claimed: survival of any scenario, immunity to purification, protection against untested editors or attacks.

**Operating rules** that act as success criteria for process (HANDOFF_v2.txt): never invent a number; every number needs n, mean +- 95% bootstrap CI over images, 3 seeds, paired vs v1; fixed splits, never tune on test; one config; no runtime patching; results to CSV, tables and figures by script; no deletions; no credentials; research use of the dataset only.

### 3.3 Where the two lists differ

| Topic | Original plan (HANDOFF_v2 / V2_PLAN) | Final report goals (G1-G6) |
|---|---|---|
| Structure | Phases 0-8, experiments E0-E10, gates E3/E4/E5/E6, invisibility gate | Six goals; the experiments and gates are reported under them |
| Invisibility criterion | PSNR >= 38, SSIM >= 0.95 **and** human 2AFC accuracy not far above 50% | G1 uses PSNR/SSIM only; the human part is not in G1 (no human study was run) |
| Protection target | PSR must rise relative to v1 *after purification* (E4: at least doubles; E5: at least half of unpurified under DiffPure 0.1) | G2 asks for higher PSR than clean, noise and v1 with CIs excluding zero (the "doubling" gate is reported separately as the E4 gate, which **FAILED**: `results/tables/gate_E4.md`) |
| Comparison with other tools | Phase 2 task (E2) | folded into G5 |
| Generality | implicit in "five editors" | explicit goal G4 (the only goal "not achieved") |
| Reproducibility | operating rules + "done looks like" item 1 | explicit goal G6 |
| Demo and README | Phase 8 item | not a numbered goal (CPU wiring dry-run exists: `results/demo_dry/`; a GPU demo run is **not recorded** in the files) |
| Human study (E10) | in plan, forms and analysis scripts to be prepared | not a goal; report states "not run" |
| New method | none planned beyond purifier terms, `L_den`, anti-purification, ensemble, `L_id`, tuning | the **target-steered term** (2026-10-20) is the main new idea in the report and has no counterpart in the plan |

The user later stated the intent behind the work in the conversation: the goal was not merely to show v2 beats v1 but that "v2 should genuinely work" (conversation). This is why the target-steered diagnosis and the audit section were added after the first full results showed `v2_main` roughly equal to v1.

---

## 4. Threat model and scope

Sources: `reports/FINAL_REPORT.md` section 2, `README.md` (threat model paragraph), `docs/HANDOFF_v2.txt` (Evaluation spec, Risks), `docs/V2_PLAN.md` section 6 (R10, R17, R22).

### 4.1 Defender (the photo owner)

- Runs the protection **offline**, before posting.
- Has **white-box** access to the open Stable Diffusion 1.5 VAE and, for some variants, the UNet; the CLIP B/32 model is also used by the `L_sem` term.
- **Budget:** L-infinity <= epsilon on the *standard 0-255 pixel scale*. Default epsilon = 4/255. Sweep: 2, 4, 6, 8 (6 was not run). After the budget finding, every budget is given on the 0-255 scale; code converts to the [-1,1] tensor scale with `eps_model = eps_255 * 2 / 255` (`config/config.yaml` header, `src/utils/budget.py`).
- The published image is a uint8 PNG; the loop quantises so the saved image differs from the original by at most `floor(eps_255)` gray levels (`src/protect/pgd.py` docstring and `quantize` handling).
- Invisibility requirement: PSNR >= 38 dB and SSIM >= 0.95 per photo (`config/config.yaml: metrics.invisibility_min`).

### 4.2 Non-adaptive attacker (the headline mode)

The attacker does **not know** the protection method. He or she chooses an editor and optionally a cleanup (purifier) first.

- **Editors** (`config/parts/eval.yaml`): SD1.5 image-to-image (strengths 0.3, 0.55, 0.75; the headline is 0.55), SDXL image-to-image (512 px input upsampled to 1024), InstructPix2Pix, SD inpainting with a keep-face mask (the repainted region is background/cloth/hat/hair; face parts are kept), IP-Adapter FaceID (SD1.5 variant, adapter scale 0.6). All use a 30-step DPM-Solver++ sampler, guidance 7.5, 3 edit seeds, and three prompts (all three for the SD1.5 s0.55 headline cell, one rotated prompt per photo elsewhere). Edits are saved as JPEG q90.
- **Cleanups actually evaluated** (12, `results/combined/edits.csv` purifier column): JPEG q90 / q75 / q50 and a fourth JPEG setting (`jpeg_other`: quality 65, 4:4:4); down-up 0.5 (bicubic) and `downup_other` (scale 0.375, Lanczos); SD1.5 VAE round trip; SD-based DiffPure-style noise-then-denoise at strengths 0.05 / 0.1 / 0.2 / 0.3; Swin2SR restoration (down-scale 0.5 then a x2 super-resolution model). The final report says "13 cleanups"; the thirteenth, **GrIDPure**, never ran (no rows in `results/combined/edits.csv`; model file unavailable, `reports/FINAL_REPORT.md` section 2).
- This is the setting for the PSR numbers in section 1.3.

### 4.3 Adaptive attacker (stress test, reported separately)

Knows the method and trains a small autoencoder on clean/protected pairs of *other* photos (the 120-photo `ae_pool` list, disjoint from test/train/validation/messy/v1-calibration), then uses it as the cleanup (`src/eval/adaptive_ae.py`; ~1 GPU-hour, 128 px patches; the report gives "35,496 steps, output PSNR vs the clean photo 45.1 dB"). The project calls it a LightShed-style idea but states that it is **not** a LightShed implementation (`src/eval/adaptive_ae.py` docstring). It is weak by construction, so the measured drop is a lower bound on what an adaptive attacker can do. It was run against clean, v1 and `v2_final` only.

### 4.4 Out of scope, not covered, or not claimed

- Survival in any scenario not listed above; "immunity to purification"; protection against editors/attacks that were not tested.
- **Human perception of identity or of the noise** (no human study).
- Face-masking inpainting: if the attacker masks the *face itself*, those pixels are discarded before the model sees them, so noise there cannot protect them (`docs/HANDOFF_v2.txt`, `reports/FINAL_REPORT.md` section 2). The evaluated inpainting editor keeps the face, so the measured PSR is about 0 for every method.
- GrIDPure (model file unavailable), IMPRESS (not run; no reference in `src/` or `scripts/`), InstantID (not run), PhotoGuard's diffusion-attack variant (only the encoder attack was run; `src/baselines/photoguard.py` docstring).
- The adaptive attacker against the target-steered variants (not run).
- Domain: CelebA-HQ portraits only; the dataset is used for research and no face image appears in the report (images are never committed; two face-containing PDFs were still on the public master branch when last checked - conversation).
- Compute: 40 PGD iterations and K = 4 EoT draws (no 100 iterations or K = 8).

---

## 5. Theoretical and technical background

Notation: x is a photo in [-1,1]^(3xHxW) (H = W = 512 here), delta is the perturbation, epsilon the budget, E(.) the SD VAE encoder, D(.) the decoder. Items flagged **"general background, not verified from project files"** are standard knowledge written from memory; the project files do not define them. Literature items are mentioned only where a project file names them; their citation details are **not verified; look up before citing**.

### 5.1 Diffusion models and latent diffusion

*General background, not verified from project files, except where a file is named.* A diffusion model generates data by learning to reverse a gradual noising process. In the forward process a clean sample x0 is turned into x_t = sqrt(alpha_bar_t) x0 + sqrt(1 - alpha_bar_t) eps, with eps ~ N(0, I) and a noise schedule alpha_bar_t that decreases with the timestep t (t runs from 0 to about 1000 in Stable Diffusion). A neural network (for Stable Diffusion, a UNet) is trained to predict the added noise eps from (x_t, t, condition); sampling starts from noise and repeatedly removes predicted noise, with a sampler/scheduler deciding the step rule. Text conditioning is injected through cross-attention using text embeddings; classifier-free guidance mixes conditional and unconditional predictions with a guidance scale (7.5 in this project's edits, `config/parts/eval.yaml: guidance_scale`).

A **latent diffusion model** (Stable Diffusion) runs this process not on pixels but on a compressed latent produced by a pretrained autoencoder (a KL-regularised VAE): 512 x 512 x 3 pixels become a 64 x 64 x 4 latent. The UNet operates on latents; the VAE decoder converts the final latent back to pixels. The project's code uses the standard diffusers components (`AutoencoderKL`, a UNet, a CLIP text encoder) and the model id `stable-diffusion-v1-5/stable-diffusion-v1-5` for v2 (the original id `runwayml/stable-diffusion-v1-5` is kept only for the v1 reproduction; `config/config.yaml` models block). The project notes that UNet inputs must be the latent multiplied by the VAE's scaling factor (0.18215 for SD1.5, 0.13025 for SDXL according to `docs/V2_PLAN.md` R1; the code reads it from `vae.config`), while the v1-style `L_vis` uses the *unscaled* latent mean, to keep the Optuna-tuned weight `alpha` on its original scale (`src/protect/losses.py`, `docs/V2_PLAN.md` F8).

Samplers used here: DPM-Solver++ (multistep, 2nd order) with 30 steps for every editor (`config/parts/eval.yaml`); v1's own Phase 2 used the default scheduler and steps (`docs/V2_PLAN.md` F9). The project lists "DPM-Solver++ for all editors, not v1's" as a limitation.

### 5.2 The SD1.5 VAE and why latent-space attacks work

*General background, not verified from project files, except as marked.* The VAE encoder maps an image to the mean and variance of a Gaussian over latents. The project's losses use the posterior **mean** `vae.encode(x).latent_dist.mean` (differentiable). Editors sample from the posterior (img2img) or use the mode (InstructPix2Pix) - `docs/V2_PLAN.md` R4 notes the SD posterior std is small so mean is close to sample, and says any gap would show up in the measurements.

Why this is attackable: the encoder is a deep convolutional network, and like any such network it is not locally smooth - a perturbation of a few gray levels, chosen by gradient ascent, can move the latent far more than a random perturbation of the same size. In image-to-image editing the *latent of the input photo* is the starting point of the denoising trajectory (section 5.3), so a latent that has been pushed to a wrong place (v1's `L_vis`, "away from the clean latent") or pulled to a degenerate place (the target-steered term, "toward a fixed target latent") can make the output degrade or lose the person. The project's own words: "the editor starts from the latent, so a wrong latent means a wrong edit" (`docs/PROJECT_GUIDE.md` section 3.1). That this works at all for *identity* is an empirical result of this project (PSR rises above clean), not a theorem. The project's mechanism explanation for why a fixed target helps ("a target latent gives the optimiser a consistent objective") is explicitly labelled a **hypothesis, not tested** (`reports/FINAL_REPORT.md` section 8, point 3).

The attack transfers only as far as the editor uses the same encoder: it is expected not to help against IP-Adapter FaceID (the identity comes from a face-embedding network on the photo, not the VAE path), against inpainting that discards the face pixels, or against SDXL (a different VAE). The final report states these as hypotheses consistent with the measurements, "not individually tested" (section 8, point 6).

### 5.3 Image-to-image editing and "strength"

*General background, not verified from project files, except the values used.* In Stable Diffusion image-to-image, the input photo is encoded to a latent z0, noised forward to timestep t_start = strength x T (strength in [0,1]), and then denoised from t_start down to 0 under the text prompt. Low strength keeps most of the original (edit stays close to the photo); high strength destroys most of it (the result is mostly governed by the prompt). The project found:

- At strength 0.3 the PSR is 0 for clean photos and for every method except PhotoGuard (0.170) and Mist (0.047) - the edit keeps almost everything, so the identity is retained (`reports/FINAL_REPORT.md` 7.7).
- At strength 0.75 the clean edits already destroy identity (PSR 84.0% clean), so improvements cannot be measured there (saturation).
- Strength 0.55 is the informative regime and the headline setting.
InstructPix2Pix (instruction following, no strength parameter; image guidance 1.5), SD inpainting (strength 1.0 on the masked region) and IP-Adapter FaceID (image-to-image at 0.55 with a face-embedding adapter, scale 0.6) are the other editors (`config/parts/eval.yaml`).

### 5.4 PGD and L-infinity budgets

*General background, not verified from project files, except the project's implementation.* Projected gradient descent (PGD, here gradient *ascent* on an objective) searches for a perturbation delta with ||delta||_inf <= epsilon: start from a random point inside the box; repeat: compute the gradient of the objective with respect to delta; take a step of size `step` in the direction of its **sign** (so each pixel moves by the same amount, the "sign gradient" update); project (clip) delta back into [-epsilon, epsilon] and also clip x + delta to the valid image range. L-infinity means *no single pixel value may change by more than epsilon*; "4/255" means at most 4 gray levels out of 255.

Project specifics:

- v1: 40 iterations, step 1/255 on the [-1,1] scale (= 0.5/255 on the standard scale), uniform random start over the full ball, a single EoT draw per iteration (`src/optimizer.py`, `docs/V2_PLAN.md` F1, F5).
- v2: 40 iterations, step 1.0/255 (standard scale) in every v2 preset, K = 4 EoT draws, random start `uniform_full`, per-draw `autograd.grad` with an fp16-safe loss scale, final result rounded onto the uint8 grid (`src/protect/pgd.py`, `config/parts/protect.yaml`). The change from step 0.5 to 1.0 is recorded in `reports/PHASE_0.md` "Decisions taken"; it is a difference between v1 and v2 that is not isolated by an ablation.
- Unit mix-up lesson: the [-1,1] scale doubles every number (4/255 on [0,255] = 8/255 on [-1,1]); this is the root of the v1 budget mislabel (section 6).

### 5.5 Expectation over Transformation (EoT)

*General background, not verified from project files, except as marked.* If the attacker will apply a random transformation T (compression, crop, resize...) to the protected image, optimising the loss at a single fixed transform lets the perturbation overfit to it. EoT optimises the *expected* loss over a distribution of transformations by averaging gradients over sampled transformations at each step. The v1 slide deck (`main.pdf`, reference list) lists "Synthesizing Robust Adversarial Examples" (ICML 2018) as the EoT reference; **citation details not verified; look up before citing**.

In the project: v1 uses Kornia `RandomJPEG` (quality 50 to 100, p = 0.5) and `RandomResizedCrop` (scale 0.8 to 1.0, p = 0.5) with `same_on_batch=True`, one random draw per iteration; the "paired" mode applies the *same* random parameters to the clean and the poisoned image by concatenating them on the batch axis (`src/eot.py`). v2 replaces the random spatial/JPEG augmentation by a **mixture of purifiers** drawn per EoT sample (K = 4 draws per iteration): the gradient of the objective evaluated on P(x + delta) is averaged over draws, P being drawn from `purifier_train_probs` (`src/protect/purifiers.py`, `pgd.py`). Spatial EoT is switched off in v2 by default (`eot.spatial.enabled: false`) because the clean latent `z_clean` is computed once and is only valid for geometry-preserving purifiers (`docs/V2_PLAN.md` R3).

### 5.6 Purification (cleanup) methods

A *purifier* is a cleanup an attacker applies to the protected photo before editing, hoping to erase the perturbation. The project distinguishes **training purifiers** (differentiable, used inside Half A) from **held-out/evaluation purifiers** (real codecs and pipelines, used only in Half B).

| Purifier | What it does (general background, not verified from project files) | How the project implements it |
|---|---|---|
| JPEG | Lossy compression: block DCT and quantisation removes high-frequency detail, which is where many perturbations live | Training: Kornia differentiable JPEG, quality 50 to 95 (`protect.yaml purifier_train_params.jpeg`; the project notes it approximates libjpeg rounding, `docs/V2_PLAN.md` R7). Evaluation: real PIL JPEG at q90/75/50 and `jpeg_other` (q65, 4:4:4) |
| Down-up resize | Shrink and re-enlarge, discarding fine detail | Training: random scale 0.5 to 0.9, bilinear/bicubic, antialias; a disjoint `downup_heldout` (scale 0.30 to 0.45) exists but is excluded from training. Evaluation: PIL `downup_0.5` (bicubic) and `downup_other` (scale 0.375, Lanczos) |
| VAE round trip | Encode with the SD VAE and decode again; the autoencoder reconstructs a clean-looking image and may remove perturbations it cannot represent | Training: `VAERoundTrip` (SD1.5, latent mean). Evaluation: `vae_rt` |
| DiffPure-style | Add a moderate amount of diffusion noise and denoise with a diffusion model, so the noise swamps the perturbation and the model reconstructs a natural image. The method name "DiffPure" comes from `docs/HANDOFF_v2.txt` sources list ("DiffPure, diffusion models for adversarial purification"); **citation details not verified; look up before citing** | Evaluation: SD1.5 image-to-image with an empty prompt, strength t* = 0.05/0.1/0.2/0.3, 50 steps, guidance 1.0 (`eval.yaml purifiers.diffpure_*`). The report calls it "DiffPure-style, SD1.5-based, not the original DiffPure" (`FINAL_REPORT.md` section 10). It shares the SD1.5 UNet with the training surrogate, so it is "gray-box, not truly held-out" (`docs/V2_PLAN.md` R17) |
| One-step SD purifier (training surrogate for DiffPure) | Encode, add noise at one random timestep t in [50, 300], predict the noise with the frozen UNet (empty prompt), recover the one-step estimate of z0, decode | `SDOneStepPurifier` (`src/protect/purifiers.py`), mixture probability 0.3 in `v2_main` etc. |
| Swin2SR restoration | A super-resolution/restoration network applied after downscaling, trying to reconstruct a "clean" high-resolution image | Evaluation `restore_swin2sr`: down-scale 0.5 then model `caidas/swin2SR-classical-sr-x2-64` (`eval.yaml`) |
| GrIDPure | Another purification method named in the brief ("GrIDPure (Zhao et al.)"); **citation details not verified; look up before citing** | Code exists (`src/eval/gridpure.py`, official repo + checkpoint required) but the model file was unavailable; **no results** |
| IMPRESS | Named in the brief as a purification method; **citation details not verified** | **PROPOSED ONLY** (no code) |

### 5.7 Anti-purification ideas

The brief names "AntiPure" as a source for terms that make a perturbation survive purification (`docs/HANDOFF_v2.txt` Phase 5 and sources; **citation details not verified; look up before citing**). The project implements two terms in `src/protect/antipure.py`, both maximised and both evaluated on the one-step purifier's reconstruction:

- `l_hf`: mean squared high-frequency energy of the 8x8 patch DCT of the reconstruction, divided by the same quantity on the clean image (so about 1 at start). Rationale: push the purified output to carry high-frequency artifacts.
- `l_mono`: reconstruction error versus the clean image at two timesteps sharing the same noise, `0.5*(e_lo+e_hi) - relu(e_lo - e_hi)`, i.e. large error that does not shrink as t grows. The file states: this is **the project's own reading** of "keep the reconstruction error growing with t"; "the AntiPure paper's exact formulation was not checked; unverified".

Result: gain on 1 of 3 DiffPure strengths (t* = 0.3: +0.07 [+0.02, +0.11]), no gain within the CI at 0.1 and 0.2 (`results/tables/gate_E6.md`), at a visibility cost (13/100 photos below SSIM 0.95).

### 5.8 Target-steered latent attacks (PhotoGuard and Mist, as the project documents them)

*The following describes the project's documentation of the two tools; the original papers were not read for this document.*

- **PhotoGuard** (project docstring: "Salman et al. 2023, 'Raising the Cost of Malicious AI-Powered Image Editing'"; the v1 deck lists it as ICML 2023): the **encoder attack** runs PGD so that the VAE latent of the protected image moves to a target - in the official img2img demo the loss is the norm of the latent mean, i.e. a **zero-latent target**; in the inpainting demo the target is the latent of a target image. The project's checked official defaults (commit `686bea7...`): epsilon 0.06 on [-1,1] (= 7.65/255 standard), step 0.02 (img2img), 1000 iterations, step decayed linearly to step/100. The project's re-implementation runs at **our** epsilon = 4/255 with 1000 iterations and step 2.55/255, zero-latent target (`src/baselines/photoguard.py`, `config/parts/baselines.yaml`). Only the encoder attack was run (not the diffusion attack). **Citation details not verified; look up before citing.**
- **Mist** (project docstring: "Liang & Wu 2023, 'Mist: Towards Improved Adversarial Examples for Diffusion Models'"): a fused loss with a *textural* term (distance between the latent of the image and the latent of a fixed target pattern image, `MIST.png`) and a *semantic* term (the diffusion denoising error at random t, prompt "a painting"), weight ratio 1e4 on the semantic term, 100 steps, step 1/255. The re-implementation runs at our epsilon 4/255 with the official target image fetched at run time (not vendored because the repo is GPL-3.0). Known deviations are listed in the file (posterior mean instead of sample, no block splitting, etc.). **Citation details not verified.**

**The project's new idea** (`reports/FINAL_REPORT.md` 4.2, `src/protect/losses.py: TargetVis`): v1's `L_vis` has no direction (it only pushes the latent away from the clean latent). The target term instead *pulls the latent of the cleaned photo toward a fixed destructive target* **inside the purifier-EoT loop**:

value = - w_tgt * MSE( enc(P(x + delta)), z_target ) + w_dev * MSE( enc(P(x + delta)), z_clean ), maximised with weight alpha; `z_target = 0` ("zero") or the latent of a mid-gray image ("gray"); `w_tgt = 1`, `w_dev = 0` in all presets. Variants: `v2_tgt_zero` (alpha = 0.27), `v2_tgt_zero_mid` (alpha = 0.55), `v2_tgt_zero_hi` (alpha = 1.0), `v2_tgt_gray` (alpha = 0.27, validation only). All use the cheap purifier mixture (JPEG, down-up, VAE round trip), `beta = 0.9024` (the CLIP term stays on), `eta = 0`, K = 4, step 1.0/255 (`config/parts/protect.yaml`, resolved config in `results/S28/.../resolved_method_cfg*.json`).

### 5.9 Face recognition: ArcFace, insightface, and the impostor-threshold PSR

*General background, not verified from project files, except as marked.* ArcFace is a face-recognition network trained with an angular-margin loss so that embeddings of the same person cluster by **cosine similarity**. The project uses it through the `insightface` package, model pack `buffalo_l`, detection size 640 x 640, with the input image **padded by 40%** (`pad_frac: 0.4`) because the detector missed 38 of the 100 test faces on tight crops (`config/parts/eval.yaml`, `docs/PROJECT_GUIDE.md` section 8 bug 4). Evaluation uses the embedding of the largest detected face.

**PSR (protection success rate).** An edit is a *protection success* ("hit") if **no face is detected** in it, **or** the ArcFace cosine similarity between the edit's face and the source photo's face is below a threshold (`src/eval/edit_runner.py: psr_hit`; `docs/V2_PLAN.md` section 4.1). The threshold is the **95th percentile of the similarity between different people**: ArcFace similarity between source photo i and the *clean* edit of a different photo j (i != j), computed per editor from the E0 clean edits (`results/combined/psr_thresholds.json`: SD1.5 0.1005, ip2p 0.1078, FaceID 0.1079, inpaint 0.1097, SDXL 0.1037, pooled 0.1034; SD1.5 uses 147,609 pairs, pooled 261,162). Interpretation: an edit counts as "no longer the same person" if it is less similar to the source than 95% of impostor pairs. Counting "no face" as a success is a design choice (handoff); the no-face rate is always reported next to PSR (0.3% for clean and v1 edits, 1.4 to 2.6% for target variants, 8.4% PhotoGuard, 8.2% Mist - `FINAL_REPORT.md` 7.10).

**Limits the project states** (`FINAL_REPORT.md` 6.2, 7.10): the threshold is operational; even clean edits at strength 0.55 already have low mean similarity (0.222) so the 0.101 bar is stringent and numbers are relative to clean edits (hence "net protection" = PSR(method) - PSR(clean) under the same cleanup); an ArcFace score is not human recognition; same-person de-duplication of the photo lists used an unverified ArcFace similarity >= 0.5 and the splitter's detector missed faces, so the impostor pairs may contain same-person pairs (`docs/V2_PLAN.md` R19).

**Identity loss `L_id` (IMPLEMENTED BUT NOT RUN as a lever).** `src/protect/face_loss.py` implements an identity term using a *different* network from the evaluation ArcFace (default backend `facenet`, pretrained `vggface2`, per `config/parts/protect.yaml`) to avoid "grading on the training network" (`docs/V2_PLAN.md` R16). `lambda_id = 0` in every preset and every reported run; no result exists.

### 5.10 Image quality metrics: PSNR, SSIM, LPIPS

*General background, not verified from project files, except the project's conventions.*

- **PSNR** = 20 log10(255 / sqrt(MSE)) between two 8-bit images (higher means more alike); a constant difference of 4 gray levels gives 36.12 dB (test in `docs/V2_PLAN.md` section 3.9). The project requires >= 38 dB.
- **SSIM** is a local structural-similarity score in [-1,1]. The project's evaluation SSIM follows v1: `skimage.metrics.structural_similarity(channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)`; the in-loop visibility penalty uses an 11x11 Gaussian window with sigma 1.5, constants 0.01^2 and 0.03^2 (`src/losses.py`, `src/protect/losses.py`). Required >= 0.95.
- **LPIPS** is a learned perceptual distance; the project uses `lpips.LPIPS(net='vgg')` on [-1,1] tensors (lower means more alike). Use 1: LPIPS(clean, protected) as a secondary invisibility measure (v1: 0.2314 +- 0.0637). Use 2: "edit-LPIPS" between the edit of the clean photo and the edit of the protected photo. **Edit-LPIPS was not reported at scale** (only 513 pairs were computed, not reported - `FINAL_REPORT.md` 6.2, 10).
- Two conventions are reported for the v1 reproduction: "v1 convention" (tensor to uint8 by truncation, as v1's code did) and "published PNG" (rounded image that would actually be posted) (`reports/PHASE_0.md`).
- CLIP-I (CLIP image-image similarity between source and edit) is computed with CLIP ViT-L/14, deliberately a different model from the CLIP B/32 used in `L_sem`, so the metric cannot be "gamed" by the loss (`docs/V2_PLAN.md` R11; `config/config.yaml models.clip_eval`). It is a secondary metric (`results/tables/clip_i.md`).

### 5.11 Bootstrap confidence intervals and paired comparisons

*General background, not verified from project files, except the project's procedure.* A **percentile bootstrap** resamples the units of analysis (here, photos) with replacement many times, recomputes the statistic (mean), and takes the 2.5th and 97.5th percentiles as the 95% interval. A **paired** comparison computes the per-photo difference between two methods on the *same* photos and bootstraps that difference, which removes between-photo variability; a difference counts as "supported" only if its CI excludes 0.

Project procedure (`docs/EVAL_PLAN_FINAL.md` section B; `src/eval/stats.py`; `config/config.yaml stats`): unit = photo; seeds and prompts are averaged within a photo first; 10,000 resamples with seed 2026 are stated in `config/config.yaml` and in the generated tables ("percentile-bootstrap CI over images (10000 resamples, seed 2026)"); `docs/EVAL_PLAN_FINAL.md` says "2,000 resamples" - **a discrepancy; the tables, which are generated, say 10,000**. Tables also give a two-sided Wilcoxon signed-rank p, explicitly **not corrected for multiple comparisons**. Intervals are never shown as two overlapping CIs; the paired difference with its own CI is the quantity compared.

### 5.12 Other components named in project files (for completeness)

- **CLIP B/32** (`openai/clip-vit-base-patch32`): used in v1 and in v2's `L_sem` term - "1 minus cosine similarity between the image embedding and the text embedding of 'a potted plant'", minimised. The project flags that SD1.5 image-to-image never runs a CLIP *image* encoder, so `L_sem` "may do nothing (unverified)" (`docs/HANDOFF_v2.txt` flaw 8, `docs/V2_PLAN.md` R11). The `no_sem` ablation that would answer this was **not run**.
- **Optuna**: hyperparameter search used in v1 (30 trials) and v2 (16 trials; section 7).
- **Grad-CAM** (`src/gradcam.py`): a v1 explainability utility, "not used by v2" (`docs/V2_PLAN.md` section 1.1).
- **IP-Adapter** (FaceID variant, model id `h94/IP-Adapter-FaceID`, weight `ip-adapter-faceid_sd15.bin`): the identity-conditioned editor (`config/config.yaml`). **InstantID**: considered, not run. **Citation details for IP-Adapter, ArcFace, Stable Diffusion, SDXL, InstructPix2Pix, CelebA-HQ, Kornia, LPIPS are not given in the project files beyond model ids; look up before citing.**

---

## 6. DiffShield v1: baseline method, its results, and the problems found

Sources: `docs/HANDOFF_v2.txt` (Background and current state), `docs/V2_PLAN.md` section 1.2 (findings F1-F12), `docs/PROJECT_GUIDE.md` section 3, `reports/PHASE_0.md`, `reports/FINAL_REPORT.md` section 3, `data/v1_ref/*`, and the code `src/losses.py`, `src/optimizer.py`, `src/eot.py`, `src/data.py`, `src/legacy_eval.py`, `src/gradcam.py`, `src/protect/v1.py`. The v1 slide deck `main.pdf` (project review, dated 2026-03-26 in its title slide; five student authors are listed there) gives the original proposal.

### 6.1 The original proposal (from `main.pdf`)

The deck proposes a "Tri-Modal" adversarial immunisation: **visual collapse** (maximise MSE in the VAE latent), **semantic erasure** (move the CLIP image embedding toward an unrelated concept such as "a potted plant") and **structural sabotage** (a differentiable *Canny* edge term meant to disrupt ControlNet geometric conditioning), under an EoT loop with differentiable JPEG/crop, L-infinity epsilon = 8/255, with Grad-CAM cross-attention heatmaps as "XAI evidence". Its hypotheses and deliverables were ControlNet-robustness and ASR (attack success rate via ArcFace) against PhotoGuard and Anti-DreamBooth; its literature table lists PhotoGuard, AdvI2I, Anti-DreamBooth, DiffAM, TUAP (all **citation details not verified; look up before citing**; the slide gives venues and years that this document does not repeat). In the implemented v1 code the Canny term was replaced by a windowed-SSIM structure-preservation term (below), so the implemented v1 is **not** the Canny/ControlNet design of the deck; the ControlNet evaluator survives only as legacy code (`src/legacy_eval.py`: SD1.5 + ControlNet-canny pipeline, 20 steps).

### 6.2 The v1 algorithm as implemented

- **Optimisation:** PGD gradient ascent on the "Tri-Modal" loss, 40 iterations, `epsilon = 8/255` and `alpha (step) = 1/255` *on [-1,1] tensors*, uniform random start in the full epsilon ball, clipped each step, final L-infinity assert (`src/optimizer.py`). One EoT draw per iteration.
- **EoT layer** (`src/eot.py`): `RandomJPEG(quality (50, 100), p = 0.5, same_on_batch=True)` and `RandomResizedCrop(512, scale (0.8, 1.0), p = 0.5, same_on_batch=True)`; `forward_paired` applies the same random parameters to the clean and poisoned images so losses measure the perturbation rather than augmentation misalignment. (`same_on_batch=True` was originally a runtime string patch applied by the Phase 1 notebook; it is now a real commit - `docs/V2_PLAN.md` section 3.2.)
- **Loss** (`src/losses.py: DiffShieldLoss.forward`): `total = alpha * L_vis - beta * L_sem - gamma * L_str`, maximised by PGD, where
  - `L_vis` = MSE between `vae.encode(x_aug).latent_dist.mean` and `vae.encode(x_poisoned_aug).latent_dist.mean` (SD1.5 VAE, fp32, **unscaled** mean, no 0.18215 factor); maximised;
  - `L_sem` = 1 - cosine(CLIP-B/32 image embedding of the poisoned image [resized to 224, CLIP-normalised], projected text embedding of "a potted plant"); minimised;
  - `L_str` = 1 - windowed SSIM(x_aug, x_poisoned_aug), 11x11 Gaussian, sigma 1.5, on [0,1]; minimised via the subtracted term. This replaced an earlier buggy Canny term ("It replaced an earlier Canny-based term that was buggy", `docs/HANDOFF_v2.txt`; `docs/V2_PLAN.md` F-notes).
- **Weights** (Optuna trial 23, `data/v1_ref/optimal_hyperparameters.json`): alpha = 0.2699475469647097, beta = 0.9023875567667267, gamma = 60.52658429696642 (the brief rounds these to 0.27, 0.90, 60.5; best objective value 0.12650258839130402). The base multipliers that Optuna scaled are in `data/v1_ref/base_multipliers.json` (alpha_base 0.5416, beta_base 1.2569, gamma_base 28.186; raw losses visual 1.846, semantic 0.796, structural 0.0355).
- **Optuna objective** (brief): 30 trials maximising the worst-case LPIPS(clean, protected) subject to PSNR >= 38 and SSIM >= 0.95. That is, the search *rewards visible change* in the protected image rather than protection of identity.
- **Data selection:** calibration set = KMeans centroids on the 40 binary CelebAMask-HQ attributes (k from a `kneed` elbow on k = 1..20; k = 4: images 5319, 1715, 6385, 8932 - `data/splits/v1_calib.txt`); evaluation set = 70 disjoint KMeans centroids (`data/splits/v1_eval70.txt`, `data/v1_ref/evaluation_70_manifest.csv`). Clean images were produced by resizing CelebA-HQ 1024 -> 512 with PIL bicubic and saving under the original `.jpg` name, i.e. **JPEG at PIL default quality 75** (`docs/V2_PLAN.md` F2).
- **Inputs/outputs:** images in [-1,1], 512 x 512 (`src/data.py`: Resize((512,512)), ToTensor, Normalize(0.5, 0.5)); output tensors saved by `torchvision.utils.save_image`.

### 6.3 v1's reported results (n = 70 for invisibility, 15 for edits)

| Test | Result | Source |
|---|---|---|
| Invisibility, 70 images | PSNR 41.911 +- 0.440, SSIM 0.97496 +- 0.0048, LPIPS(clean, protected) 0.2316 +- 0.063; Linf 0.0313725 = 8/255 on [-1,1]; 0 violations of PSNR >= 38 and SSIM >= 0.95 | `data/v1_ref/stage1_summary_metrics.csv` (this file also records 6 "violations" of an LPIPS >= 0.15 target, labelled "High=Good") |
| Edit disruption, SD1.5 img2img, 15 pairs | LPIPS between clean-edit and protected-edit about 0.46 (range 0.38 to 0.53), strength 0.55, guidance 7.5, seed 2026, one prompt ("a highly detailed portrait, cyberpunk style, neon lighting, digital art masterpiece") | `docs/HANDOFF_v2.txt`; `docs/V2_PLAN.md` F9. Mean 0.4601 per `V2_PLAN` F9. **The Phase 2 outputs were lost (an empty zip), so 0.46 can only be quoted, not recomputed** (`docs/PROJECT_GUIDE.md` 3.1) |
| README numbers (PSNR 38.64, SSIM 0.9989) | Do not match; they belong to older Canny-era trial notebooks; superseded and unverified; **do not quote** | `docs/V2_PLAN.md` F11 |

### 6.4 Reproduction of v1 in v2 (Phase 0; **IMPLEMENTED AND RUN**, Kaggle job J01, bundle B0, 2026-10-14)

Source: `reports/PHASE_0.md`; run folder `results/J01/`. n = 70, one protection seed per image, 40 iterations; CIs are 95% bootstrap over images.

| Metric | v1 original | Reproduced (v1 convention): mean +- std [95% CI] | Reproduced (published PNG) | Violations |
|---|---|---|---|---|
| PSNR (dB) | 41.911 +- 0.440 | 41.986 +- 0.416 [41.890, 42.082] | 42.171 +- 0.436 [42.070, 42.272] | 0/70 below 38 |
| SSIM | 0.97496 +- 0.00480 | 0.9754 +- 0.0054 [0.9741, 0.9766] | 0.9756 +- 0.0054 [0.9744, 0.9769] | 0/70 below 0.95 |
| LPIPS (VGG) | 0.23156 +- 0.06304 | 0.2314 +- 0.0637 [0.2168, 0.2463] | 0.2305 +- 0.0635 [0.2160, 0.2454] | n/a |
| Linf ([-1,1]) | 8/255 | 8/255 in all 70 | PNG max change = 4 gray levels | n/a |

Paired differences to v1's own per-image file (matched by original file name): PSNR +0.0752 [+0.0021, +0.1507] dB; SSIM +0.0004 [-0.0002, +0.0010]; LPIPS -0.0002 [-0.0042, +0.0039]; all within the accepted tolerance (+-0.5 dB, +-0.005, +-0.03; accepted by the user on 2026-10-14, `reports/PHASE_0.md`). The PSNR difference is statistically above zero but tiny; its cause (different random draws) is **unverified**. The distribution of |pixel change| over all values (J01): 0: 11,468,029; 1: 18,754,991; 2: 11,998,407; 3: 7,683,111; 4: 5,145,702; >= 5: 0. Cost: 45.6 +- 1.2 GPU-seconds per image on a T4 (peak memory 7.2 GB), versus the planner's 25 s. **The optional second-person reproduction (J01b) was not run, so the project does not claim a second person reproduced it.**

### 6.5 v1 re-measured inside the v2 harness (100 test photos, 3 seeds)

From `reports/FINAL_REPORT.md` 7.1, 7.2, 7.3, 7.7, 7.6 (copied; see there for full tables):

- Invisibility: PSNR 42.42 (min 41.17), SSIM 0.9763 (min 0.9587), 0 photos below either limit (n = 100).
- PSR, SD1.5 s0.55, no cleanup: clean 13.9%, v1 24.0% (+10.1 [+7.2, +13.0]); uniform noise 14.9%.
- Net protection over clean under cleanups: JPEG75 +0.047*, JPEG50 +0.037*, down-up 0.5 **+0.000 (no gain)**, VAE round trip +0.097*, DiffPure 0.1/0.2/0.3 +0.130* / +0.100* / +0.100* (* = CI excludes 0).
- Other editors: v1 PSR is 0.000 for strength 0.3, FaceID and inpainting; **InstructPix2Pix: 0.090 versus 0.180 for clean (-0.090 [-0.147, -0.040]) - protected edits are *less* protected than clean**, partly a face-detection effect not separated; SDXL 0.020.
- Adaptive autoencoder attacker: 11.0% (net +1.7 [-1.0, +4.7]) - protection removed.
- E3 gate (decision rule for Phase 5): "BARELY DROPS" under the raw-PSR rule (mean relative drop over six core cleanups = 0.16, `results/tables/gates.md`); the report notes this rule ignores that cleanups also change clean photos, hence net protection.

### 6.6 Every flaw found in v1 and the fix chosen

**A. Flaws listed in the original brief** (`docs/HANDOFF_v2.txt`) and the final report's table (`FINAL_REPORT.md` 3.2):

| # | Flaw | Consequence | v2 response (status) |
|---|---|---|---|
| 1 | "Survives any scenario" cannot be proven; never tested against purification | an attacker can clean the noise off | cleanups inside the optimisation (Phases 3-5) and 12 held-out cleanups in evaluation (**RUN**) |
| 2 | No random-noise control | part of the effect could be noise | uniform and PSNR-matched Gaussian controls (E1; **RUN**: noise adds +1.0 and +0.8 points over clean, CIs include 0) |
| 3 | No identity metric (LPIPS measures change, not who it is) | "disrupted" does not mean "no longer the person" | ArcFace similarity and PSR (**RUN**) |
| 4 | One editor, one prompt, one strength, one seed, 15 edits | one lucky setting proves nothing | 5 editors, strengths 0.3/0.55/0.75, 3 prompts for the headline, 3 seeds, 100 test photos (**RUN**; SDXL for clean, v1, PhotoGuard, `v2_final`; InstructPix2Pix/inpainting not for every variant) |
| 5 | Optuna rewarded visible change (worst-case LPIPS), not protection | tunes the wrong thing | Optuna maximises PSR on validation photos, with invisibility constraints (**RUN**: 16 trials, proxy objective; section 7) |
| 6 | Attacks one stage (the VAE encoder); `L_sem` may do nothing; no ablations | cannot say what matters | denoiser term, anti-purification terms, ablations (**partly RUN**: `abl_vis_only`, `abl_den_only` run; `abl_no_sem`, `abl_no_pur` **not run**) |
| 7 | One encoder, no ensemble | transfer unknown | SDXL-VAE ensemble in `v2_final` (**RUN**, not isolated by an ablation) |
| 8 | Runtime string patches, unlogged seeds, hard to reproduce, clean images only | nobody can repeat it | patches became commits (tags `v1-unpatched`, `v1.0`); one config; per-image hashed seeds; run logging (**RUN**) |
| 9 | Budget mislabelled | numbers not comparable | all budgets on the 0-255 scale (**RUN**; confirmed 4/255 in J01) |

**B. Findings from reading the v1 code and notebooks** (`docs/V2_PLAN.md` section 1.2, "[FINDING]" items):

- **F1 budget:** `PGDOptimizer(epsilon=8/255, alpha=1/255)` on [-1,1] means epsilon = 4/255 and step = 0.5/255 on the standard scale; the brief's "step 1/255, same as v1" read the tensor value as standard.
- **F2 clean inputs were JPEG** (PIL default q75, 4:2:0 subsampling), so "clean" images were JPEG-decoded pixels; v1 reproduction replicates this, v2 uses lossless PNG (`data.clean_format: png`).
- **F3 metric conversion:** v1 converts tensors with `((x+1)*127.5).astype(np.uint8)` (truncation) while the saved PNG rounds; both conventions are reported.
- **F4** SSIM/LPIPS definitions (windowed Gaussian SSIM; LPIPS VGG on [-1,1]).
- **F5 randomness:** global `SEED=42`, a single EoT draw, full-epsilon uniform start, shuffled data loader; output names follow shuffled order, mapping only in `stage1_per_image_metrics.csv`. Bit-exact reproduction is impossible; reproduction is statistical.
- **F6 selection:** KMeans/elbow selection is deterministic given the attribute file and sklearn version; the notebook had no stored outputs, so the 70 names and weights were recovered from v1's saved manifests.
- **F7 `L_str` placement:** v1 computes `L_vis` and `L_str` on the *augmented pair*, not on the published image x + delta; v2 computes `L_str` on the published image.
- **F8 `L_vis` scale:** unscaled latent mean; alpha tuned at that scale; v2 keeps it for `L_vis`, uses scaled latents for the UNet.
- **F9 Phase 2 setup:** one prompt, strength 0.55, guidance 7.5, seed 2026, default scheduler/steps, 15 pairs, `runwayml/stable-diffusion-v1-5` fp16.
- **F10 model id:** `runwayml/stable-diffusion-v1-5` was taken down in 2024 but still resolved on Kaggle (likely a redirect; unverified); v2 uses the `stable-diffusion-v1-5/...` mirrors, v1 adapter falls back to the mirror if the old id cannot be loaded (commit `50dd3b4`).
- **F11 README numbers** are from older Canny-era notebooks and unverified.
- **F12** the Phase 1 notebook's patches are idempotent (marker strings), so committed patches keep the old notebook runnable.

**C. Findings from measurement in v2:**

- The budget is 4/255, not 8/255 (J01).
- v1's effect is real but modest (+10.1 points over clean) and **vanishes under down-up resizing** and under an adaptive autoencoder; it is also worse than clean on InstructPix2Pix; it gives nothing at strength 0.3, for FaceID or inpainting.
- Phase 2 outputs and the original protected images were not preserved, so some v1 claims can only be quoted.

### 6.7 What the v1 code base became

`src/eot.py`, `src/optimizer.py`, `src/losses.py`, `src/data.py`, `src/gradcam.py` stay at their import paths, patched and frozen (bug fixes only); `src/eval.py` was moved to `src/legacy_eval.py` with a lazy compatibility shim; the thin adapter `src/protect/v1.py` makes v1 usable through the same `Protector.protect(x, seed)` interface as v2 methods, enforcing epsilon = 8/255 and step = 1/255 on [-1,1] for the `v1` preset (`docs/V2_PLAN.md` sections 1.3, 3.5). Two housekeeping changes with no effect on the algorithm: model ids/components can be injected into `DiffShieldLoss`, and the unused `CLIPProcessor` load is wrapped in try/except (`reports/PHASE_0.md`).

---

## 7. The v2 plan as written versus what was actually carried out

### 7.1 Planning documents and how they relate

1. `docs/HANDOFF_v2.txt` - the team brief (phases 0-8, experiments E0-E10, gates). Authoritative; where the plan disagrees, the brief wins except for factual corrections.
2. `docs/V2_PLAN.md` (written 2026-10-14) - module designs, interfaces, CSV schema, Phase 0 in detail, risks R1-R22, questions to ask.
3. `docs/V2_PLAN_48H.md` - compresses the plan into 48 hours by running everything in parallel on 5 Kaggle accounts; cost model (estimates, later replaced by measurements in `docs/COST_MEASURED.md`), scale-downs, tiers and a session matrix S01-S18.
4. `docs/EVAL_PLAN_FINAL.md` (written 2026-10-20) - status and full evaluation protocol for the remaining time (deadline Thursday 2026-10-22 06:00 IST), wave-1 launch table, integrity checks.
5. `reports/PHASE_1.md` to `PHASE_8.md` - **unrendered templates with placeholders** (e.g. `{{num:psr.v1.none}}`, "PENDING"); only `PHASE_0.md` is a finished phase report. Everything that would have gone into them is in `reports/FINAL_REPORT.md` and `results/tables/` instead.

The overall timeline: the plan asked for about ten weeks of sequential work, compressed to a 48-hour plan (T+0 = 2026-10-14 16:30, deadline T+48 = 2026-10-16 16:30), which was then extended in practice (the evaluation plan of 2026-10-20 and the final report of 2026-10-22; sessions S19 to S28 were added after the 48-hour plan - see `kaggle/jobs.yaml` and `docs/EVAL_PLAN_FINAL.md`).

### 7.2 Component and experiment status table

Legend: **R** = implemented and run (code + result files); **IN** = implemented, not run; **P** = proposed only; **D** = dropped/replaced. "Evidence" gives the file that proves the status (I checked `results/combined/edits.csv`, `results/tables/coverage.md`, and the source tree).

**Infrastructure and Phase 0-2**

| Planned component / experiment | Status | Evidence and notes |
|---|---|---|
| Patches as commits; layout; tags `v1-unpatched`, `v1.0` | R | `git tag` shows both tags; `reports/PHASE_0.md` |
| `config/config.yaml` + parts, run logging, per-image seeds, resumable/sharded jobs | R | `src/utils/`, `kaggle/jobs.yaml`, run folders with `config.json` etc. |
| Budget check and v1 reproduction (n = 70) | R | `results/J01`, `reports/PHASE_0.md`; second-person run J01b **IN** (defined in `kaggle/jobs.yaml`, not run) |
| `requirements.txt` pinned to the Kaggle image | R | `reports/PHASE_0.md` (J00 freeze) |
| Splits: test 100 (incl. all 70 v1 photos), messy 20, train 20, val 20, AE pool 120; v1's 4 calibration photos excluded | R | `data/splits/*.txt`, `splits_meta.json`; identity de-dup by ArcFace >= 0.5 (**threshold unverified**); `splits_meta.json` records `n_noface_forced: 40` (its exact meaning is not documented in the files I read) |
| Identity labels (`identity_CelebA.txt`) | D | not supplied; replaced by ArcFace de-duplication |
| Five editors (SD1.5 s0.3/0.55/0.75, SDXL, InstructPix2Pix, SD inpaint keep-face, IP-Adapter FaceID) | R | `results/tables/coverage.md`; SDXL only for clean, v1, PhotoGuard, `v2_final` (plan: same four; `v2_main`, ablations: no SDXL) |
| InstantID | P | `docs/V2_PLAN.md` R10 chose FaceID; InstantID "only if time allows" - not run |
| ArcFace, PSR threshold from E0 clean edits (95th percentile, per editor + pooled) | R | `results/combined/psr_thresholds.json` |
| CLIP-I | R | `results/tables/clip_i.md`; one CLIP model (ViT-L/14) |
| Edit-LPIPS (clean-edit vs protected-edit) | IN, partial | script `scripts/compute_edit_lpips.py`; 513 of about 5,300 headline pairs computed and **not reported** (CPU too slow, process crash - conversation; `FINAL_REPORT.md` 10) |
| Bootstrap CIs, paired tests, gates, tables, figures | R | `src/eval/analysis.py`, `results/tables/`, `results/figures/` |
| E0 clean edits | R | S07; coverage shows clean x 14 settings (12 evaluated cleanups + adaptive_ae + none) |
| E1 noise controls (uniform, PSNR-matched Gaussian) | R | S22, `results/tables/e1.md`; edits for SD1.5 (three strengths), InstructPix2Pix, inpaint, FaceID; **SDXL: not run** |
| E2 PhotoGuard and Mist (re-implementations at our 4/255) | R | S13 (PhotoGuard), S21 (Mist), `results/tables/e2.md`; each with 5 cleanups on SD1.5 s0.55 |
| E3 v1 under cleanups | R | S08, S19; 12 evaluated cleanups (GrIDPure missing) |
| GrIDPure | IN | `src/eval/gridpure.py`; model file unavailable; zero rows; an unavailable purifier is skipped and logged |
| IMPRESS | P | not in code; listed in the brief as "if feasible" |
| Swin2SR restoration purifier | R | `restore_swin2sr` rows exist |
| A "truly different-model" held-out purifier (pixel-space DiffPure with ADM checkpoint, V2_PLAN R17) | P | the DiffPure-style cleanup shares the SD1.5 UNet |

**Phases 3-6 (protection variants)**

| Planned component / experiment | Status | Evidence and notes |
|---|---|---|
| Phase 3 `v2_cheap` (JPEG, down-up, VAE round trip in the EoT loop, K = 4, one down-up setting held out) | R | S10; `results/tables/gate_E4.md`; held-out `downup_heldout` defined but excluded (evaluation uses real `downup_other`) |
| **E4 gate** (PSR at least doubles vs v1 under JPEG/down-up/VAE round trip) | R, **FAIL** | ratios 1.11 (JPEG75), 1.04 (JPEG50), 1.48 (down-up 0.5), 0.97 (VAE round trip) (`gate_E4.md`) |
| Phase 4 `v2_main` (+ one-step SD purifier at share 0.3, + `L_den`) | R | S09 (protect), S19 (cleanup tier); PSNR >= 40.3, SSIM >= 0.97 on all 100 test photos |
| **E5 gate** (PSR under DiffPure 0.1 >= half the unpurified PSR) | R, **PASS (robust)** | `gate_E5.md`; v2_main 0.280 vs half of 0.243 (my observation from `final_readout.md`: v1 would pass by the same arithmetic, 0.263 under DiffPure 0.1 vs 0.240 unpurified, so the gate does not separate v2 from v1) |
| Phase 5 anti-purification (`v2_antipure`) | R | S12, S07/S20 (validation single-term ablation); **E6 gate: GO on 1 of 3** (t* = 0.3 only); SSIM < 0.95 on 13/100 photos |
| Phase 5 option "3 to 5 step DDIM purification in the last PGD iterations" | P | plan only; no code found |
| Phase 6 SDXL-VAE ensemble | R, no ablation | part of `v2_final` (resolved config: `protect.ensemble.enabled: true`) |
| Phase 6 CLIP ensemble | D | plan: only if `L_sem` helps; single CLIP; `no_sem` ablation not run |
| Phase 6 `L_id` (identity loss, separate recogniser) | IN | `src/protect/face_loss.py`; `lambda_id = 0` everywhere; never evaluated |
| Phase 6 per-term loss normalisation | IN | `TermScales`, `loss.normalize_terms`; `false` in every resolved config seen |
| Phase 6 Optuna re-tuning | R, scaled down | plan 30-50 trials x 20 validation photos; run **16 trials x 10 validation photos**, proxy objective (20 iterations, K = 2, 1 seed, SD1.5 s0.55 + FaceID, cleanups {none, JPEG75, DiffPure0.1}), top 3 confirmed on all 20 validation photos at 20 iterations (`kaggle/jobs.yaml` S06 params: `optuna_trials: 16, optuna_images: 10, optuna_iters: 20, confirm_images: 20`; S06; `reports/optuna_trials_S06.csv`, `optuna_confirm_S06.json`). Result: trial 5 selected (`config/parts/final.yaml`: alpha 0.728, eta 0.0578, beta 0.668, gamma 62.05, antipure on, w_hf = w_mono = 0.931); confirm PSR 0.0667 (trial 5), 0.05 (trials 10, 11), hand-set 0.0417, i.e. tuned vs hand-set not separable; proxy-best trial 10 (0.10) fell to 0.05 on confirm (`FINAL_REPORT.md` 10) |
| Per-photo invisibility constraint in tuning | P / missing | the search checked invisibility on the *mean* over its photos, so `v2_final` breaks SSIM >= 0.95 on 29/100 test photos |
| Randomised perturbation (random start per image) | R | `random_start: uniform_full`, per-image seeds |
| `v2_final` (tuned) | R | S06 (protect), S16 (edits); 100 test + 20 messy photos |

**Phase 7 experiments and ablations**

| Planned component / experiment | Status | Evidence and notes |
|---|---|---|
| E7 ablation arm: encoder loss only (`abl_vis_only`) | R | S11, `results/tables/e7.md`: PSR 0.25 [0.20, 0.31] (none) |
| E7 ablation arm: denoiser loss only (`abl_den_only`) | R | PSR 0.12 [0.08, 0.17]; perturbation almost invisible (PSNR 50.76), so a weak test of the denoiser idea (`FINAL_REPORT.md` 7.5) |
| E7 arm: both without `L_sem` (`abl_no_sem`) | IN, **not run** | `results/tables/e7.md`: "not run"; job S14 not run |
| E7 arm: no purifier in the loop (`abl_no_pur`) | IN, **not run** | same |
| E7 "both" arm (`v2_main`) | R | PSR 0.24 [0.20, 0.29] |
| Note on ablation size | scaled | `docs/PROJECT_GUIDE.md` planned ablations and epsilon sweep on the first 50 test photos; the coverage table shows all ablation and eps cells with **100** photos (the plan's scale-down was not needed) |
| E8 adaptive attacker | R, weak | S15; `results/tables/e8.md`; clean/v1/`v2_final`; SD1.5 and FaceID; 120-photo pool, ~1 GPU-hour |
| E8 against the target-steered variants | IN/not run | stated as the most important unknown (`FINAL_REPORT.md` 11) |
| E9 epsilon sweep | R, partial | eps2 (S17) and eps8 (S18) run on 100 photos; **eps6 (S23) not run**; eps4 = `v2_final`; cells {none, JPEG75, DiffPure0.1} + FaceID none |
| E10 human study (15 raters, 20 pairs, 2AFC + same-person) | IN | `scripts/make_human_study.py`, `scripts/analyze_human_study.py` exist; `results/tables/e10.md`: Task A and B "not run" |
| Messy 20 photos | R | S-sessions; coverage shows messy cells for 8 methods with SD1.5 and FaceID |
| Phase 8 demo script | IN / partly R | `scripts/demo.py`; `results/demo_dry/` is a CPU wiring dry run; a real GPU demo output is not recorded in the files |
| README reproduces the main table from the saved config | R (as documented) | `README.md` "Reproduce every table"; `scripts/finalize_all.py`; whether an independent person re-ran it is **not recorded** |
| Tag `v2.0-final` | IN | planned (`V2_PLAN.md`, `EVAL_PLAN_FINAL.md` E7); `git tag` in this repo shows only `v1-unpatched`, `v1.0` and a backup tag; no `v2.0-final` tag was found here (may have been created elsewhere - UNKNOWN / not recorded) |
| Per-phase reports PHASE_1..8 | D (replaced) | templates only; superseded by `FINAL_REPORT.md` |

**Added after the original plan (not in the brief)**

| Component | Status | Evidence |
|---|---|---|
| Target-steered term `TargetVis` and variants `v2_tgt_zero`, `v2_tgt_zero_mid`, `v2_tgt_zero_hi`, `v2_tgt_gray` | R | code `src/protect/losses.py`; presets in `config/parts/protect.yaml`; S24/S25 (validation screening) and S26-S28 (100 test photos). `v2_tgt_gray` was screened on validation only and not run on test. `v2_tgt_zero_mid` was declared as an interpolation and was **not screened on validation** (preset comment) |
| Control for the target term (`v2_cheap`, same cleanups, no target term) | R | validation: no gain (11.7% vs clean 8.3%) |
| Screening on 20 validation photos (S24/S25) | R | my recomputation from `results/combined/edits.csv` reproduces the report's validation values (clean 0.083, v1 0.150, v2_cheap 0.117, tgt_zero 0.350, tgt_zero_hi 0.467, tgt_gray 0.350 with no cleanup) |
| Audit of what PSR measures (no-face share, similarity distributions, per-photo, visual check) | R | `FINAL_REPORT.md` 7.10; `results/figures/audit_panel.png`, `sim_distributions.png` |
| Pareto figure (protection vs invisibility), failure gallery | R | `results/figures/` (some figures contain faces and must stay local) |
| Presentation deck (14 slides), FINAL_REPORT (md/pdf), hash map after history rewrite | R | `reports/` |

### 7.3 Where the actual run deviated from the 48-hour plan (and the stated reasons)

- **Sampler:** DPM-Solver++ 30 steps for all editors (`config.yaml` still shows `edit.num_inference_steps: 50`, but `config/parts/eval.yaml` overrides to 30). The plan listed it as a decided scale-down.
- **Prompts:** 3 for the headline cell; one rotated prompt per photo elsewhere (as planned).
- **PGD:** 40 iterations, K = 4 (as planned); no 100 iterations / K = 8.
- **Optuna:** 16 x 10 proxy instead of 30-50 x 20 (as planned in the 48-hour plan); the plan's "cap 3 h" was exceeded - S06 hit its 12-hour timeout because the confirm step took longer than estimated (conversation); the confirmation JSON shows `t_total_s` of about 3,700 to 4,200 s per confirmed trial.
- **Costs:** the planner's estimates were 2 to 6 times too low for v2 (`docs/COST_MEASURED.md`): measured per-image GPU seconds at 40 iterations (autocast): v1 46, `v2_cheap` 146, `v2_main` 263, `v2_final` 395 (plan 25 / 54 / 120 / 190). Using fp16 autocast for the VAE gave a 1.5x to 2.4x speed-up (`COST_MEASURED.md`).
- **Cleanup tiers:** the purifier-tier evaluation jobs initially expanded to zero jobs because of a default `--purifier none`; this was a silent failure fixed with `purifier: all`, re-runs S19/S20, and a regression test (`FINAL_REPORT.md` Appendix B item 11).
- **Sessions not run:** S14 (`abl_no_sem`, `abl_no_pur`), S23 (epsilon 6). (S14's noise-control portion was moved to S22 and run.)
- **Quota and accounts:** the plan assumed 5 accounts with chained overnight sessions; the weekly-quota figure was never confirmed in the files (`docs/EVAL_PLAN_FINAL.md`: "UNVERIFIED, ask").
- **Not in the plan but done:** the target-steered family (above), the audit, extra figures.

### 7.4 Discrepancies noticed between project files (for the person writing the paper)

1. **Number of cleanups:** `FINAL_REPORT.md` sections 2 and 6 say "13 cleanups"; the 13th (GrIDPure) never ran. Twelve were evaluated (plus the adaptive autoencoder, which is a separate mode).
2. **Bootstrap resamples:** `docs/EVAL_PLAN_FINAL.md` says 2,000; `config/config.yaml` and all generated tables say 10,000 (seed 2026).
3. **Paired CI for `mid` vs v1:** `FINAL_REPORT.md` 7.9 prints [+0.218, +0.298]; `results/tables/final_readout.md` prints [+0.218, +0.299]. Same point estimate +0.257.
4. **`v2_final` vs v1:** `FINAL_REPORT.md` 7.4 gives +5.4 [+2.4, +8.7]; `results/tables/claims.md` gives +0.05 [+0.02, +0.09] (rounded to two decimals).
5. **Screening:** `FINAL_REPORT.md` section 5 says the three declared variants were screened on validation, but `config/parts/protect.yaml` states that `v2_tgt_zero_mid` was **not** screened on validation (the screened ones were `zero`, `zero_hi`, `gray`).
6. **Test counts:** `docs/PROJECT_GUIDE.md` says 305 CPU tests; `README.md` and `FINAL_REPORT.md` say 377 (the test suite grew).
7. **Planned vs actual ablation size:** `docs/PROJECT_GUIDE.md` says ablations and the epsilon sweep use the first 50 test photos; the coverage table shows 100.
8. **Step size:** the brief says "step 1/255 standard scale, same as v1"; v1's real step is 0.5/255. The `v1` preset uses 0.5; all v2 presets use 1.0. This difference between v1 and v2 is not isolated by any ablation.
9. **Phase reports:** `PHASE_1.md` to `PHASE_8.md` are unrendered templates (not results).
10. **Sample-size of the optional second reproduction:** J01b was never run; do not say "reproduced by a second person".

*(End of Part 1.)*


---

## 8. System architecture

Source of truth: repository `DiffShield-wt/integration`, branch `v2/release` (paths below are relative to that repository root). Status labels used throughout: IMPLEMENTED AND RUN, IMPLEMENTED BUT NOT RUN, PROPOSED ONLY. "Run" means it was executed in a real (non-smoke) Kaggle session whose results are reported in `reports/FINAL_REPORT.md`; the evidence for "run" in this part comes from that report and from `kaggle/jobs.yaml`, not from re-executing code.

### 8.1 The two halves

The design rule (stated in `docs/PROJECT_GUIDE.md` section 5.1 and enforced by `tests/test_isolation.py`) is that the code that makes a protected photo and the code that attacks and scores it share no model code.

* **Half A, protection (offline, makes the published photo).** Package `src/protect/` (v2) plus the frozen v1 code (`src/losses.py`, `src/optimizer.py`, `src/eot.py`, wrapped by `src/protect/v1.py`). Input: one clean 512x512 RGB photo in [-1,1]. Output: a protected photo on the 8-bit grid whose gray levels differ from the clean photo by at most `floor(eps_255)` (eps = 4/255 in all reported runs except the E9 sweep, which also uses 2/255 and 8/255). Uses differentiable "training purifiers" (`src/protect/purifiers.py`).
* **Half B, attack-and-score harness (measures every claim).** Package `src/eval/`. Takes the saved PNG of a protected photo, optionally applies a REAL (non-differentiable) held-out cleanup, edits it with a generative editor, and scores identity retention, protection success rate (PSR), CLIP-I, edit-LPIPS and invisibility. It never imports `src.protect`.
* **Baselines** (`src/baselines/`): noise controls, PhotoGuard and Mist re-implementations. They import only `src.utils` (test `test_baselines_only_utils`). They produce protected photos, so they sit on the Half A side of the data flow but share no code with it.

Isolation test (code): `tests/test_isolation.py` asserts by AST scan that no file in `src/eval` imports `src.protect`, none in `src/protect` imports `src.eval`, and `src/baselines` imports none of `src.eval`, `src.protect`, `src.losses`, `src.optimizer`, `src.eot`. Evaluation-side purifiers are separate implementations from the training-side ones (e.g. `src/eval/purify_attacks.py` JpegPurifier = PIL libjpeg vs `src/protect/purifiers.py` DiffJPEG = kornia differentiable codec). Status: IMPLEMENTED AND RUN.

### 8.2 End-to-end data flow

1. **Data and splits** (`scripts/make_splits.py`, `scripts/prepare_data.py`, `scripts/prepare_v1_data.py`, lists in `data/splits/*.txt`, config `data:` block in `config/config.yaml`): CelebA-HQ, image_size 512. Splits per `config/config.yaml` / `reports/FINAL_REPORT.md`: test 100 (includes v1's 70 evaluation photos), messy 20, validation 20, train 20, adaptive-attacker pool `ae_pool` 120. The 'messy' split gets a pre-compression in `prepare_data`.
2. **Protect** (`scripts/run_protect.py`): for each photo in the split (sorted by stem, `src/utils/io.py: list_images`), `load_tensor_v1` (Resize 512x512, ToTensor, Normalize(0.5,0.5), range [-1,1]), seed `derive_seed(cfg["seed"], "protect", image_id)` (`src/utils/seeding.py`; sha256 of `"2026|protect|<id>"` mod 2^31), `protector.protect(x, seed)`, `save_png_v1` (torchvision `save_image` after (x+1)/2), then per-photo invisibility metrics (`src/eval/metrics.py: v1_quality_metrics`, `png_quality_metrics`) appended to `metrics.csv`. `--shard K/N`, `--resume`, `--checkpoint_every` give checkpoint/resume per image. The v2 protector additionally writes `info/<image_id>.json` (per-iteration loss history).
3. **Method dispatch** (`src/protect/registry.py: build_protector`): `v1` -> `src/protect/v1.py`; names starting `v2_`, `abl_`, `eps` -> `src/protect/v2_registry.py: build_v2_protector`; `noise_*`, `photoguard`, `mist` -> `src/baselines/registry.py: build_baseline`. All satisfy `protect(x[1,3,H,W] in [-1,1], seed) -> (tensor, info dict)`.
4. **Evaluate** (`scripts/evaluate.py` -> `src/eval/edit_runner.py`): for every (method, photo, purifier, editor, strength, prompt, 3 edit seeds) of a *tier* (tiers H, M, P, A, SCREEN, TGT, AP, E9, E8, SMOKE defined in `config/parts/eval.yaml: eval.tiers`): load protected PNG, apply held-out purifier (cached PNG), edit, obtain the edit of the CLEAN photo from the shared E0 cache, score, append one CSV row (columns `ROW_COLUMNS` = `src/eval/schema.py: EDIT_COLUMNS`).
5. **Ingest and analyse** (`scripts/ingest_results.py` -> `results/combined/{edits,protect}.csv`, `psr_thresholds.json`; `src/eval/analysis.py`; `scripts/make_tables.py`, `scripts/readout_final.py`, `scripts/make_figures.py`, `scripts/make_pareto_figure.py`): de-duplicate by key, compute PSR thresholds, per-image aggregation, bootstrap CIs, paired differences, tables, figures.
6. **Orchestration on Kaggle** (`kaggle/jobs.yaml`, `scripts/run_job.py`, `kaggle/job_runner.ipynb`, `kaggle/build_bundle.py`): a code bundle zip is uploaded as a Kaggle dataset; the notebook sets `JOB_IDS` and runs `run_job.py`, which reads job definitions (macros `setup`, `prep`, `protect`, `evaluate`) and executes steps sequentially, one process per GPU for "sharded" steps (T4 x2), resumable via DONE markers and `--resume_from`. Each job zips outputs; `scripts/gpu_ledger.py` tallies GPU seconds.
7. **Tuning** (`scripts/tune_optuna.py`): Optuna TPE over (alpha, eta, beta, gamma, antipure on/off, shared w_hf=w_mono) with a proxy PSR objective computed by `src/eval/edit_runner.py: score_protected` on 10 validation photos; the top trials were confirmed on 20 validation photos; the result is written as `config/parts/final.yaml`. Status: IMPLEMENTED AND RUN (S06, 12 h session timeout hit; see HISTORY notes).

### 8.3 Diagram

```
                        config/config.yaml  +  config/parts/{protect,eval,baselines,data,final}.yaml
                         (deep-merged, sorted; src/utils/config.py: load_config; method preset; --set overrides)
                                         |
        data/splits/*.txt  -->  scripts/prepare_data.py (512 px PNGs)
                                         |
   +-------------------- HALF A: make the published photo -------------------+
   | scripts/run_protect.py                                                  |
   |   seed = derive_seed(2026,"protect",image_id)                           |
   |   src/protect/registry.build_protector                                  |
   |     v1            -> src/protect/v1.py -> src/{optimizer,losses,eot}.py  |
   |     v2_*/abl_*/eps*-> src/protect/v2_registry.py                         |
   |                        pgd.py  (PGD+EoT)                                 |
   |                        purifiers.py (jpeg,downup,vae_rt,sd_onestep)     |
   |                        losses.py (L_vis,L_den,L_str,TargetVis)          |
   |                        antipure.py, ensemble.py, face_loss.py, models.py |
   |     noise_*, photoguard, mist -> src/baselines/*   (utils only)          |
   +--> results/<run>/images/<id>.png   metrics.csv (PSNR,SSIM,LPIPS,Linf)   |
                                         |
   +--------------------- HALF B: attack and score ---------------------------+
   | scripts/evaluate.py -> src/eval/edit_runner.py (EditRunner)             |
   |   protected PNG --purify_attacks.py (real JPEG/down-up/VAE/DiffPure/    |
   |                    Swin2SR/GrIDPure*/adaptive_ae)--> editors.py          |
   |   (SD1.5 img2img, SDXL img2img, InstructPix2Pix, SD inpaint(mask.py),   |
   |    IP-Adapter FaceID) --> faces.py (ArcFace buffalo_l), clip_metric.py   |
   |   E0 clean-edit cache (shared by all methods) ----> edit_lpips, PSR thr |
   +--> metrics.csv (one row per edit)                                       |
                                         |
   scripts/ingest_results.py -> results/combined/{edits,protect}.csv, psr_thresholds.json
   src/eval/analysis.py (bootstrap, paired, gates) -> scripts/make_tables.py / readout_final.py / make_figures.py
                                         |
                        reports/FINAL_REPORT.md (+pdf), results/tables/*

   Orchestration: kaggle/jobs.yaml + scripts/run_job.py + kaggle/job_runner.ipynb (+ build_bundle.py)
   Tuning: scripts/tune_optuna.py (uses edit_runner.score_protected) -> config/parts/final.yaml
   Adaptive attacker: scripts/train_adaptive_ae.py -> src/eval/adaptive_ae.py (a Half B purifier)
   (* GrIDPure unavailable in the runs: model file missing; jobs skipped and logged)
```

### 8.4 Module dependency description

* `src/utils/` (config.py, seeding.py, budget.py, io.py, runlog.py, tiny_models.py): leaf utilities used by both halves. `config.py: load_config(path, method, overrides)` deep-merges `config/config.yaml` with `config/parts/*.yaml` in sorted order, then `methods[method]` preset, then `--set a.b.c=val` overrides (parsed with `yaml.safe_load`), then `validate_config` (checks `budget.scale == standard_0_255`, `0 < eps_255 <= 16`, `0 < step_255 <= eps_255`, `iters >= 1`). Because `final.yaml` sorts before `protect.yaml`, `v2_final` is layered by `v2_registry.resolve_cfg`: `methods.v2_final_default` (hand-set start, `protect.yaml`) < `methods.v2_final` (tuned, `final.yaml`) < `methods.<epsN>`.
* `src/protect/models.py`: lazy frozen SD1.5 VAE+UNet bundle `SDModels`, SDXL VAE loader, `ClipSem`, and generator-aware random helpers. Default model ids: `stable-diffusion-v1-5/stable-diffusion-v1-5`, `madebyollin/sdxl-vae-fp16-fix`, `openai/clip-vit-base-patch32` (`models.py: DEFAULT_IDS`). Revisions are not pinned in config (`revision: null`).
* `src/protect/purifiers.py` <- `models.py`; `losses.py` <- `models.py`; `antipure.py` <- `models.py`, `purifiers.py`; `ensemble.py` <- `models.py`; `face_loss.py` standalone (lazy facenet-pytorch); `pgd.py` <- `losses.py`, `models.py`, `purifiers.py`; `v2_registry.py` <- `pgd.py` (+ lazy others); `v1.py` <- legacy `src/` v1 code + `src/utils`.
* `src/eval/`: `edit_runner.py` (runner, tiers, rows, PSR helpers, `score_protected`) <- `editors.py`, `purify_attacks.py`, `faces.py`, `clip_metric.py`, `masks.py`, `adaptive_ae.py` (plug-in loaded lazily by name), `analysis.py` (imports `edit_runner.impostor_sims`), `metrics.py` and `stats.py` (invisibility metrics and bootstrap; `metrics.py` is also used by `scripts/run_protect.py`).
* Scripts are thin CLIs over the packages: `run_protect.py`, `evaluate.py`, `merge_shards.py`, `ingest_results.py`, `make_tables.py`, `readout_final.py`, `make_figures.py`, `tune_optuna.py`, `train_adaptive_ae.py`, `run_job.py`. Human-study scripts `make_human_study.py` / `analyze_human_study.py` exist: IMPLEMENTED BUT NOT RUN (no human study was run).

### 8.5 Component status table

| Component | Code | Status |
|---|---|---|
| v1 protector (frozen reference) | `src/protect/v1.py`, `src/{losses,optimizer,eot}.py` | IMPLEMENTED AND RUN (Phase 0 reproduction J01, then as comparison method) |
| v2_cheap, v2_main, v2_final, abl_vis_only, abl_den_only, v2_antipure | `src/protect/*`, presets in `config/parts/protect.yaml` | IMPLEMENTED AND RUN (FINAL_REPORT tables list v2_cheap, v2_main, v2_final, antipure, ablations vis-only and den-only) |
| abl_no_sem, abl_no_pur | presets exist | IMPLEMENTED BUT NOT RUN (FINAL_REPORT limitations: "`no_sem` / `no_pur` ablations ... not run") |
| eps2, eps6, eps8 | presets | eps2 and eps8 IMPLEMENTED AND RUN (E9 table, FINAL_REPORT 7.6); eps6 (S23) IMPLEMENTED BUT NOT RUN ("eps 6 was not run") |
| v2_tgt_zero, v2_tgt_zero_mid, v2_tgt_zero_hi | `losses.TargetVis`, presets | IMPLEMENTED AND RUN (S24/S25 screening on validation, S26-S28 on test) |
| v2_tgt_gray | preset | screened on validation only (SCREEN tier lists it); not in the test TGT tier. IMPLEMENTED AND RUN on validation only |
| Identity loss L_id | `src/protect/face_loss.py`, wired in `v2_registry._build` when `lambda_id > 0` | IMPLEMENTED BUT NOT RUN (`lambda_id = 0` in every preset; FINAL_REPORT: "never evaluated") |
| Spatial EoT (`eot.spatial`) | config key only, no code reads it | PROPOSED ONLY (grep finds no use in `src/` or `scripts/`) |
| Loss-term normalisation (`loss.normalize_terms`, `TermScales`) | `losses.py`, `pgd.py: params_from_cfg` | IMPLEMENTED BUT NOT RUN (`normalize_terms: false` in `config/config.yaml`) |
| noise_uniform, noise_gauss_psnr, photoguard, mist | `src/baselines/*` | IMPLEMENTED AND RUN (S13, S21, S22) |
| Editors: SD1.5 img2img, SDXL img2img, IP2P, SD inpaint, IP-Adapter FaceID | `src/eval/editors.py` | IMPLEMENTED AND RUN |
| Held-out purifiers: jpeg_q90/75/50/other, downup_0.5/other, vae_rt, diffpure 0.05/0.1/0.2/0.3, restore_swin2sr | `src/eval/purify_attacks.py` | IMPLEMENTED AND RUN (13-purifier grid in tier P) |
| gridpure | `src/eval/gridpure.py` | IMPLEMENTED BUT NOT RUN (model file unavailable; unavailable purifiers are skipped and logged by `EditRunner._purify_stage`) |
| adaptive_ae | `src/eval/adaptive_ae.py`, `scripts/train_adaptive_ae.py` | IMPLEMENTED AND RUN against clean, v1, v2_final (S15); NOT run against target-steered variants |
| Optuna tuning | `scripts/tune_optuna.py` | IMPLEMENTED AND RUN (S06) |
| Human study | `scripts/make_human_study.py`, `scripts/analyze_human_study.py` | IMPLEMENTED BUT NOT RUN |
| Edit-LPIPS at scale | `scripts/compute_edit_lpips.py`, `edit_runner.recompute_lpips` | IMPLEMENTED; only a partial set (513 pairs per HISTORY_FACTS) computed and NOT reported |

### 8.6 Documentation-versus-code disagreements noticed

* Editor step count: `config/config.yaml` has `edit.num_inference_steps: 50` and `edit.strengths`, but the editors read `eval.num_inference_steps` (30) from `config/parts/eval.yaml` (`editors.py: _PipeEditor.__init__`: `self.ev.get("num_inference_steps", 30)`). The effective sampler is DPM-Solver++ (multistep, order 2), 30 steps (`set_dpmpp`). The `edit:` block of `config/config.yaml` is not read by `src/eval`.
* `config/config.yaml: loss.l_str_on: published` and `loss.l_vis_latent: unscaled_mean` and `loss.ssim` are documentation flags; no code reads them. The behavior they describe (L_str on the published image, unscaled latent mean, 11x11 sigma 1.5 window) is hard-coded in `src/protect/losses.py` / `pgd.py`.
* `config/config.yaml: purifiers_heldout` is not read; `src/protect/purifiers.py: build_heldout` exists but nothing in `src/` or `scripts/` calls it.
* `config/parts/final.yaml` header calls itself "candidate for config/parts/final.yaml (trial_5; confirmed on 20 val images ...)". It is the file that is merged, so it is the `v2_final` that was run if the bundle used it (S15/S16 note that "the tuned config is in the bundle").
* Docstring of `losses.l_den` says it uses "the noised latent of the purified image"; in code, when the drawn purifier is the one-step purifier the reused UNet pass (`aux`) is on the PUBLISHED image's latent (the one-step purifier's input), not on its output (see section 9.4).

## 9. Algorithms

Notation. Images are tensors in [-1,1] (range 2); `x` = clean photo (snapped to the uint8 grid first), `delta` = perturbation, `x_pub = clip(x + delta, -1, 1)` = the PUBLISHED image, `P_k` = a purifier drawn at random for EoT draw k. Budgets are given on the standard 0-255 scale and converted by `eps = eps_255 * 2/255` (`src/utils/budget.py: to_model_scale`; `pgd.py: pgd_protect`). All formulas below are read from the code at the cited function; where a number is a config value, the config key is cited.

### 9.1 Effective hyper-parameters of the protection presets

Root defaults `config/config.yaml`: `budget.eps_255 = 4`, `budget.step_255 = 0.5`, `budget.iters = 40`, `budget.random_start = uniform_full`, `eot.k_draws = 4`, `runtime.loss_scale = 1024`, `runtime.unet_dtype = fp16`, `runtime.vae_dtype_protect = autocast`, `runtime.grad_checkpointing = true`, `purifier_train_params.sd_onestep = {t_min 50, t_max 300}`, `protect.quantize = true` (`config/parts/protect.yaml`). The eps = 4/255 value is on the standard scale; v1's code constants 8/255 and 1/255 on [-1,1] equal 4/255 and 0.5/255 standard (`config/config.yaml` comment "confirmed by J01 budget check"; `src/protect/v1.py` enforces `eps == 8/255` and `step == 1/255` on the model scale for method `v1`).

`alpha = 0.2699475469647097`, `beta = 0.9023875567667267`, `gamma = 60.52658429696642` are described as the "v1 Optuna best trial" in the comment in `config/config.yaml: loss`. Step size is 1.0 (standard scale) in every v2 preset (`protect.yaml: methods.*.budget.step_255`); v1 uses 0.5.

| Preset | Cleanup mixture (probabilities) | alpha | eta | beta | gamma | extra |
|---|---|---|---|---|---|---|
| v1 (`config.yaml: methods.v1`) | none (own EoT: random-resized-crop + RandomJPEG) | 0.2699 | 0 | 0.9024 | 60.527 | K=1, step 0.5, L_str on augmented pair |
| v2_cheap | jpeg .35, downup .35, vae_roundtrip .30 | 0.2699 | 0 | 0.9024 | 60.527 | K=4 |
| v2_main | jpeg .25, downup .25, vae_roundtrip .20, sd_onestep .30 | 0.2699 | 0.2699 | 0.9024 | 60.527 | K=4 |
| v2_antipure | as v2_main | 0.2699 | 0.2699 | 0.9024 | 60.527 | anti_purify on, w_hf = w_mono = 0.5 |
| abl_vis_only | as v2_main | 0.2699 | 0 | 0 | 60.527 | |
| abl_den_only | as v2_main | 0 | 0.2699 | 0 | 60.527 | |
| abl_no_sem | as v2_main | 0.2699 | 0.2699 | 0 | 60.527 | |
| abl_no_pur | none (identity) | 0.2699 | 0.2699 | 0.9024 | 60.527 | `purifiers_train: []` |
| v2_final_default (hand-set) | as v2_main | 0.2699 | 0.2699 | 0.9024 | 60.527 | anti_purify off, SDXL-VAE ensemble ON |
| v2_final (`config/parts/final.yaml`, layered over default) | as v2_main | 0.7281249466629797 | 0.05776266977639212 | 0.6676014641453782 | 62.04711204188804 | anti_purify on, w_hf = w_mono = 0.9310312738392569, ensemble ON |
| v2_tgt_zero | jpeg .35, downup .35, vae_roundtrip .30 | 0.2699 | 0 | 0.9024 | 60.527 | `loss.target = {kind zero, w_tgt 1.0, w_dev 0.0}` |
| v2_tgt_zero_mid | as v2_tgt_zero | 0.55 | 0 | 0.9024 | 60.527 | same target |
| v2_tgt_zero_hi | as v2_tgt_zero | 1.0 | 0 | 0.9024 | 60.527 | same target |
| v2_tgt_gray | as v2_tgt_zero | 0.2699 | 0 | 0.9024 | 60.527 | `target.kind = gray` |
| eps2 / eps6 / eps8 | as v2_final | as v2_final | | | | only `budget.eps_255` = 2 / 6 / 8 |

(`lambda_id = 0` in every preset. `v2_tgt_zero_mid` is commented in `protect.yaml` as a "declared interpolation between v2_tgt_zero (alpha 0.27) and v2_tgt_zero_hi (alpha 1.0); not screened on val".)

Cleanup mixture draw: `purifiers.py: PurifierMixture`. `purifier_train_exclude: [downup_heldout]` removes the held-out down-up variant (scale 0.30-0.45, bicubic) from any draw. Probabilities are renormalised over the surviving names (`self.probs = p / sum(p)`). Draw: `idx = torch.multinomial(probs, 1, generator=gen)`.

Which "vis" branch is built (`v2_registry.py: V2Protector._build`): if `alpha != 0` and `protect.ensemble.enabled` -> `EnsembleVis` (SD1.5 VAE + `sdxl_vae_fp16`, equal normalised weights 0.5/0.5 unless `protect.ensemble.weights` is set); elif `loss.target` present -> `TargetVis`; else `SingleVis`. The branches are exclusive in that order (an ensemble would override a target; no preset combines them). `ClipSem` is built only if `beta != 0`; `IdentityLoss` only if `lambda_id != 0`; `AntiPure` only if `loss.anti_purify.enabled` (it needs the one-step purifier, building one if the mixture lacks it).

### 9.2 The PGD protection loop with K draws (v2) - `src/protect/pgd.py: pgd_protect`

Maximisation problem (docstrings of `pgd.py` and `losses.py`):

```
max over ||delta||_inf <= eps :
   -gamma * L_str(x, x_pub)  -  lambda_id * cos_id(x_pub)
 + E_{P ~ mixture, randomness}[ alpha * L_vis(P(x_pub)) + eta * L_den(P(x_pub)) - beta * L_sem(P(x_pub)) + w_hf * hf + w_mono * mono ]
```
(`L_vis`, `L_den`, `hf`, `mono` are maximised; `L_sem` = 1 - cos and `L_str` = 1 - SSIM are minimised; `cos_id` is minimised. When `TargetVis` or `EnsembleVis` is active they replace `L_vis`.)

Algorithm, per image, with seed `s = derive_seed(2026, "protect", image_id)`:

1. `gen = torch.Generator(device=<cuda|cpu>)`, `manual_seed(s & 0x7FFF...)` (`models.py: make_generator`). EVERY random draw below (random start, purifier choice, JPEG quality, down-up scale and mode, timesteps, noise) uses this one generator. The same (image, seed) gives the same output on the same device/backend (GPU kernels may be non-deterministic; the docstring cites risk R15).
2. `x0` = x converted to the uint8 grid: `u8 = round((x+1)*127.5)` clamped to [0,255], `x0 = u8 * 2/255 - 1` (only if `protect.quantize`, which is true).
3. Precompute once: `z_state = vis.prepare(x0)` (clean latent(s), or target latents), `ap_state = antipure.prepare(x0)` if anti-purification is on, and `id_loss.prepare(x0)` if `lambda_id != 0`.
4. Random start (`random_start`, default `uniform_full`): `delta = U(-1,1) * eps` per element (`uniform_half` multiplies by 0.5; `zero` gives zeros), then `delta = clip(x0 + delta, -1, 1) - x0`.
5. For `t = 1..iters` (40):
   a. `g_tot = 0`. Helper `acc(loss)` computes `g = autograd.grad(loss * loss_scale, delta)[0]`; if `g` has a non-finite entry it counts one `nonfinite_draws` and returns zeros; otherwise returns `g / loss_scale`. (`loss_scale` = 1024 guards fp16 underflow; the sign is scale invariant.)
   b. Published-image term, once per iteration: `x_pub = clip(x0 + delta, -1, 1)`, `v_str = 1 - SSIM(x0, x_pub)`; `g_tot += acc(-gamma * v_str)`. If `lambda_id != 0`: `g_tot += acc(-lambda_id * cos_id(x_pub))`.
   c. For `k = 1..K` (K = 4): `x_pub = clip(x0 + delta, -1, 1)`; `(xp, aux, name) = mixture.sample(x_pub, gen)`; build `total` = sum of the weighted terms of Section 9.4 evaluated on `xp`; `g_tot += acc(total / K)` (a separate backward per draw, so peak memory is that of one draw). If `total` is None or does not require grad, nothing is added.
   d. Update: `delta <- clip( delta + step * sign(g_tot), -eps, eps )` then `delta <- clip(x0 + delta, -1, 1) - x0`, with `step = step_255 * 2/255`. The step size is constant (no decay). This is sign-gradient ascent with L_inf projection.
   e. Logged per iteration: mean of each term over draws, fraction of exactly-zero gradient pixels (`zero_grad_frac_mean`), purifier counts.
6. Finalisation (`quantize=True`): `E = floor(eps_255 + 1e-6)`; `d_u8 = round(delta / (2/255))` clamped to `[-E, E]`; `x_adv = (x0_u8 + d_u8)` clamped to [0,255], mapped back to [-1,1]. So the returned photo is on the 8-bit grid and `|x_adv_u8 - x0_u8| <= E` (for eps 4: at most 4 gray levels). `info` records `linf_u8`, `psnr` (`losses.psnr_uint8`), `ssim` (`losses.ssim` on the uint8-rounded result), loss history, time and peak memory.
7. `scripts/run_protect.py` saves with torchvision `save_image((x+1)/2)`; budget verification is `scripts/budget_check.py` (J01 found `max |gray-level change| = 4` for 70/70 v1 PNGs per the `config/config.yaml` comment).

Differences from v1 (`src/optimizer.py: PGDOptimizer.optimize`, `src/losses.py: DiffShieldLoss`): v1 uses one draw per iteration of a paired EoT (`src/eot.py: EoTLayer.forward_paired`: kornia `RandomResizedCrop(512, scale=(0.8,1.0), p=0.5)` then `RandomJPEG(quality=(50,100), p=0.5)`, same random parameters applied to the clean and the poisoned image concatenated on the batch axis). Its objective is `alpha*L_vis - beta*L_sem - gamma*L_str` with `L_vis = MSE(E(aug poison).mean, E(aug clean).mean)` (clean latent recomputed on the augmented clean image), `L_sem = mean(1 - cos(CLIP_img(aug poison), text("a potted plant")))`, and `L_str = 1 - SSIM(aug clean, aug poison)` computed on the AUGMENTED pair, not on the published image. PGD step is `1/255` on [-1,1]; the result is `clamp(x + delta)` with no 8-bit rounding inside the optimiser (rounding happens on PNG save); random start `delta.uniform_(-eps, eps)`; seeding uses global `seed_everything(seed)` (`src/protect/v1.py`).

Status: v1 and v2 loops IMPLEMENTED AND RUN.

### 9.3 Differentiable training cleanups - `src/protect/purifiers.py`

All are `TrainPurifier.forward(x, gen) -> x` with `x` in [-1,1], differentiable w.r.t. pixels. After every draw `PurifierMixture.sample` applies `ste_clamp(xp) = xp + (clip(xp,-1,1) - xp).detach()` (clamp in the forward pass, identity gradient).

* **JPEG** (`DiffJPEG`, name `jpeg`; config `purifier_train_params.jpeg = {impl kornia_diff, quality [50, 95]}`): `q ~ Uniform(50, 95)` (continuous, one value per batch element, `g_uniform`); `x01 = (x+1)/2`; `y01 = kornia.enhance.jpeg_codec_differentiable(clamp(x01,0,1), q)`; output `2*y01 - 1`. Kornia's codec only approximates libjpeg rounding (code comment, R7). Fallback `impl: pil_bpda` (also selected automatically if kornia cannot be imported): forward runs the real PIL codec at `quality = int(round(q))`, backward is the identity (`x01 + (pil(x01) - x01).detach()`). The source I read does not record whether the Kaggle runs used kornia or the fallback (the preset requests `kornia_diff`).
* **Down-up** (`DownUp`, name `downup`; `scale [0.5, 0.9]`, `modes [bilinear, bicubic]`, `antialias true`): `s ~ Uniform(0.5, 0.9)`; mode drawn uniformly from the two; `h_s = max(8, round(H*s))`, `w_s = max(8, round(W*s))`; `down = interpolate(x, (h_s, w_s), mode, align_corners=False, antialias=True)`; `up = interpolate(down, (H, W), mode, align_corners=False)` (no antialias on the way up); then `ste_clamp`. The held-out check variant `downup_heldout` (scale 0.30-0.45, bicubic) is defined and excluded from the mixture; `build_heldout` is never called (see 8.6).
* **VAE round trip** (`VAERoundTrip`, name `vae_roundtrip`): `decode(encode_mean(x, scaled=False), scaled=False)`; the deterministic posterior mean of the SD1.5 VAE, no latent sampling; the VAE runs under fp16 autocast when `runtime.vae_dtype_protect = autocast` on CUDA (weights stay fp32).
* **One-step SD purifier** (`SDOneStepPurifier`, name `sd_onestep`; `t_min 50, t_max 300`, empty prompt): frozen SD1.5 VAE + UNet and the real null-prompt text embedding:
  1. `t ~ UniformInteger[50, 300]` inclusive (`g_randint`), `noise ~ N(0, I)` of shape `(B, 4, H/8, W/8)`, both from `gen`.
  2. `z0 = scaling_factor * posterior_mean(x)` (`encode_mean(x, scaled=True)`; the factor is read from `vae.config.scaling_factor`, never hard-coded).
  3. `abar = alphas_cumprod[t]` (scheduler of the checkpoint; fallback scaled-linear schedule beta 0.00085..0.012, 1000 steps); `zt = sqrt(abar) z0 + sqrt(1 - abar) noise`.
  4. `eps = UNet(zt, t, null_cond)` (fp32 in/out around an fp16 UNet).
  5. `z0_hat = (zt - sqrt(1 - abar) eps) / sqrt(abar)`; `rec = decode(z0_hat, scaled=True)`.
  6. Returns `(rec, aux)` with `aux = {t, noise, eps, z0, zt, ab, rec}`, reused by L_den and the anti-purification terms.

The mixture is therefore: JPEG, down-up, VAE round trip, and (only in v2_main, v2_antipure, v2_final and the ablations) the one-step SD purifier. Status: IMPLEMENTED AND RUN.

### 9.4 Loss terms exactly as coded - `src/protect/losses.py`, `ensemble.py`, `antipure.py`, `face_loss.py`

Per draw, `pgd.py` builds `total = sum_i w_i * S(name_i, value_i)` (`S` = per-term scale, identity unless `loss.normalize_terms` is true, which it is not). A term is added only if its weight is non-zero (`if p.alpha`, `if p.eta`, `if p.beta and sem is not None`).

| Term | Weight in `total` (ascent) | Value | Where |
|---|---|---|---|
| L_vis (SingleVis) | `+alpha` | `MSE( enc_mean(xp), z_clean )`; `enc_mean` = SD1.5 VAE posterior mean, UNSCALED; `z_clean = enc_mean(x0)` computed once; MSE = mean over all latent elements (`F.mse_loss`) | `losses.l_vis`, `SingleVis` |
| L_vis (EnsembleVis) | `+alpha` | `sum_n w_n * MSE( enc_n(xp), enc_n(x0) )`, networks `{sd15, sdxl_vae_fp16}`, `w_n` normalised to sum 1 (equal by default), unscaled posterior means, SDXL VAE run in the same autocast context | `ensemble.EnsembleVis` |
| L_vis (TargetVis) | `+alpha` | `-w_tgt * MSE(enc_mean(xp), z_t) + w_dev * MSE(enc_mean(xp), z_clean)`, with `z_t = 0` (kind `zero`) or `z_t = enc_mean(zeros_like(x0))` (kind `gray`: the image whose value is 0 in [-1,1], mid-gray). All presets use `w_tgt = 1.0`, `w_dev = 0.0`, so the term is `-MSE(enc(xp), z_t)`; maximising it pulls the latent of the purified image TOWARD the target | `losses.TargetVis` |
| L_den | `+eta` | if the drawn purifier was `sd_onestep`: `MSE(aux.eps, aux.noise)` (reuses that draw's UNet pass; this pass is on the noised latent of the purifier's INPUT x_pub, not of its output). Otherwise: `z = enc_mean(xp, scaled=True)`, `t ~ UInt[50, 300]` (from `purifier_train_params.sd_onestep.t_min/t_max`), `n ~ N(0,I)`, `zt = sqrt(abar_t) z + sqrt(1-abar_t) n`, value `MSE(UNet(zt, t, null), n)` (one extra UNet pass) | `losses.l_den` |
| L_sem | `-beta` | `mean( 1 - cos( CLIP_img(xp), CLIP_text("a potted plant") ) )`; CLIP ViT-B/32 (`openai/clip-vit-base-patch32`); image preprocessing: `(xp+1)/2`, bicubic resize to 224x224, normalise with mean (0.48145466, 0.4578275, 0.40821073) and std (0.26862954, 0.26130258, 0.27577711), use the projected `image_embeds`; text from `loss.sem_target_text` | `models.ClipSem.sem_loss` |
| L_str | `-gamma` (outside the draws, once per iteration) | `1 - mean SSIM(x0, x_pub)` on the PUBLISHED image: both mapped to [0,1]; 11x11 Gaussian window, sigma 1.5, per channel (`groups=C`), zero padding 5; `C1 = 0.01^2`, `C2 = 0.03^2`; `smap = ((2 mu_x mu_y + C1)(2 s_xy + C2)) / ((mu_x^2 + mu_y^2 + C1)(s_x + s_y + C2))`; mean over all positions; fp32 | `losses.ssim`, `l_str` |
| cos_id (L_id) | `-lambda_id` (outside the draws) | `mean cos( emb(x0), emb( VAE_decode(VAE_encode_mean(x_pub)) ) )`; embedder FaceNet InceptionResnetV1 (vggface2, facenet-pytorch), 160 px bilinear crop of a fixed bbox (default central 60 percent: `(0.2W, 0.2H, 0.8W, 0.8H)`), L2-normalised; minimised. Deliberately a different network from the evaluation ArcFace | `face_loss.IdentityLoss` |
| hf (anti-purify) | `+w_hf` | only on draws where `sd_onestep` was drawn: `hf_energy(aux.rec) / hf_energy(x0)` | `antipure.AntiPure.terms` |
| mono (anti-purify) | `+w_mono` | only on one-step draws; see below | same |

Anti-purification details (`antipure.py`; IMPLEMENTED AND RUN inside v2_antipure and v2_final):
* `hf_energy(x, patch=8, hf_cut=8)`: luma `g = (0.299 R + 0.587 G + 0.114 B + 1)/2` (x in [-1,1]); crop to a multiple of 8; non-overlapping 8x8 blocks; orthonormal 2-D DCT-II `coef = D B D^T` (first row of D divided by sqrt 2); mask `(u + v) >= 8`; value `= mean over blocks of sum(coef^2 * mask) / mask.sum()`. `hf_ref = hf_energy(x0)` is clamped at 1e-8 and constant; the term is the ratio `hf(rec) / hf_ref` (about 1 at start), MAXIMISED.
* `mono`: `mid = (t_min + t_max)//2 = 175`. Let `t1` be the draw's timestep and `rec1` its reconstruction. If `max(t1) <= mid`: `t2 ~ UInt[176, 300]`, `rec_lo = rec1`, `rec_hi = reconstruct(x_pub, t2, same noise)`; else `t2 ~ UInt[50, 175]`, `rec_lo = reconstruct(x_pub, t2, same noise)`, `rec_hi = rec1`. `e_lo = MSE(rec_lo, x0)`, `e_hi = MSE(rec_hi, x0)`. Value `= 0.5 (e_lo + e_hi) - relu(e_lo - e_hi)`, MAXIMISED ("large error that does not shrink as t grows"). The code comments say this is OUR reading of AntiPure and that the paper's exact formulation was not checked (unverified). Cost: one extra one-step pass per one-step draw.

Total ascent objective as in the `losses.py` module docstring, extended: `alpha*L_vis + eta*L_den - beta*L_sem - gamma*L_str - lambda_id*cos_id + w_hf*hf + w_mono*mono`.

Observation from the code, not a measurement: because `L_str` carries weight about 60 and is differentiated on x_pub every iteration, while L_den enters with a small weight and only via the UNet error, a preset with only L_den active (`abl_den_only`) can end with a small perturbation; FINAL_REPORT 7.5 reports PSNR 50.8 dB for that ablation (a report number, not recomputed here).

### 9.5 Baseline and control protectors (Half A side, `src/baselines/`)

All return a tensor on the uint8 grid with `|u8(x_pub) - u8(x)| <= floor(eps_255)` (`noise_control.py: quantize_to_budget`), eps 4/255 (our budget, not the originals').

* **noise_uniform** (`UniformNoiseProtector`, `uniform_linf`): `n ~ U(-1,1) * E` gray levels (E = floor(eps_255) = 4), `u_adv = round(u0 + n)` clipped to `[u0-E, u0+E]` and [0,255]. CPU generator seeded with the per-image seed. Variant `sign` adds +/-E. IMPLEMENTED AND RUN (S22 / E1).
* **noise_gauss_psnr** (`GaussianPSNRProtector`, `gaussian_matched_psnr`): `u_adv = round(u0 + sigma * N(0,I))` clipped to [0,255]; sigma found per image by bisection (tolerance 0.02 dB, max 60 iterations) so that PSNR(u8 clean, u8 noisy) equals a target; target = per-image PSNR of v1's protected photo from `data/v1_ref/stage1_per_image_metrics.csv` (`baselines.noise_gauss_psnr.psnr_csv`, set in `kaggle/jobs.yaml`), else the constant 41.911 dB. Its L_inf is NOT bounded by eps. IMPLEMENTED AND RUN.
* **PhotoGuard encoder attack** (`photoguard.py: EncoderAttackProtector`; our re-implementation checked against MadryLab/photoguard@686bea7): `z_t = 0` (target `zero_latent`); random start `U(-eps, eps)` then projection; for i in 0..999: `loss = || enc_mean(x_adv) - z_t ||_2` (unsquared L2 norm of the unscaled SD1.5 VAE posterior mean, fp16 VAE with `loss_scale` 1024), `x_adv <- project( x_adv - s_i * sign(grad loss), x_ref, eps )` (gradient DESCENT), `s_i = step - (step - step/100) * i / iters` (linear decay to step/100), `step = 2.55/255*2` model scale, then `quantize_to_budget`. Parameters from `config/parts/baselines.yaml: photoguard` (iters 1000, step_255 2.55). IMPLEMENTED AND RUN (S13).
* **Mist** (`mist.py: MistProtector`; re-implementation checked against psyker-team/mist@99f5f3c): ascent on `obj = w_sem * L_sem - w_tex * L_tex` with `w_sem = 1e4`, `w_tex = 1`; `z = sf * enc_mean(x_adv)` (`sf` = VAE scaling factor); `L_sem = MSE( UNet(sqrt(abar_t) z + sqrt(1-abar_t) n, t, cond("a painting")), n )`, `t ~ randint[0, 1000)`, one noise draw (`n_noise = 1`); `L_tex = sum( ((enc_mean(x_adv) - z_target) * sf)^2 )` (sum, not mean); `x_adv <- project( x_adv + s * sign(grad obj), x_ref, eps )`, `s = 1/255` standard (no decay), 100 iterations; target `z_target` = latent of the official `MIST.png` target image when `mist.target_image` is set (documented to be fetched at run time, not vendored), else a procedural stand-in pattern (`make_texture_target`; must be reported as a deviation). Documented deviations in the module docstring: posterior MEAN used instead of a sample; `L_sem` shares the latent instead of encoding a random crop separately; no 512 resize/block splitting/mask. IMPLEMENTED AND RUN (S21).

### 9.6 Evaluation-side editors (Half B) - `src/eval/editors.py`, `config/parts/eval.yaml`

Common: diffusers pipelines in fp16, no safety checker, DPM-Solver++ (multistep, order 2) scheduler (`set_dpmpp`), `num_inference_steps = 30`, `guidance_scale = 7.5`, one `torch.Generator(device).manual_seed(s)` per edit seed, no negative prompt, seeds batched up to `max_batch`. Edit seeds: `gen_seed(image_id, k) = derive_seed(2026, "edit", image_id, k)` for `k in eval.seeds = [1, 2, 3]`; the CLEAN photo and the protected photo use the same generator seeds (`edit_runner.py: EditRunner.gen_seed`).

| Editor | Pipeline and settings | Prompts / strength |
|---|---|---|
| `sd15_img2img` | `StableDiffusionImg2ImgPipeline`, SD1.5 | `img2img` prompts; strengths 0.3, 0.55, 0.75 in the headline tier, default 0.55 |
| `sdxl_img2img` | `StableDiffusionXLImg2ImgPipeline`; input bicubically upsampled 512 -> 1024; fp16-fix VAE | strength 0.55 |
| `ip2p` | `StableDiffusionInstructPix2PixPipeline` (`timbrooks/instruct-pix2pix`), `image_guidance_scale = 1.5`, no `strength` | `ip2p` instruction prompts |
| `sd_inpaint` | `StableDiffusionInpaintPipeline`, `stable-diffusion-v1-5/stable-diffusion-inpainting`, strength 1.0, keep-face mask (CelebAMask-HQ parts skin/nose/eyes/brows/ears/mouth/lips/glasses/neck, dilated 8 px; repaint = background, cloth, hair, hat; bbox-ellipse fallback), no paste-back of the original | `img2img` prompts |
| `ipadapter_faceid` | SD1.5 img2img + IP-Adapter FaceID (`h94/IP-Adapter-FaceID`, `ip-adapter-faceid_sd15.bin`), `ip_adapter_scale = 0.6`, strength 0.55; identity condition = ArcFace embedding (512-d) of the editor's input image (zeros if no face found), negative embedding zeros; the FaceID LoRA is not loaded | `img2img` prompts |

Prompts (`eval.prompts`): img2img = "a person in a suit", "a person at a beach", "an oil painting of a person"; ip2p = "put the person in a suit", "put the person on a beach", "turn it into an oil painting". Prompt assignment (`expand_tier`): for the headline setting (editor `sd15_img2img`, strength 0.55) cells marked `all_if_headline` use all 3 prompts; every other cell uses prompt index `image_index mod 3`. So a headline cell has up to 9 edits per photo (3 prompts x 3 seeds), other cells 3.

Status: all five editors IMPLEMENTED AND RUN (reports exist for each; SDXL is run only for clean, v1, photoguard, v2_final per tier H in `config/parts/eval.yaml`; the target-steered variants were not run on IP2P, inpainting or SDXL per the report).

### 9.7 Evaluation-side cleanups (the attacker's "purifiers") - `src/eval/purify_attacks.py`

Real codecs/models applied to the saved protected PNG, cached as PNG, seed `derive_seed(2026, "purify", image_id, purifier)`:

* `jpeg_q90 / q75 / q50`: PIL libjpeg round trip at that quality (default 4:2:0 subsampling); `jpeg_other`: quality 65, subsampling 0 (4:4:4).
* `downup_0.5`: PIL `resize` to round(0.5*size) with BICUBIC then back with BICUBIC; `downup_other`: scale 0.375, LANCZOS.
* `vae_rt`: a separately loaded SD1.5 VAE (fp16 on CUDA), reflect-pad to a multiple of 8, `decode(encode(x).latent_dist.mean)`, clamp, round, crop.
* `diffpure_0.05 / 0.1 / 0.2 / 0.3`: `StableDiffusionImg2ImgPipeline` (SD1.5), DDIM scheduler, `prompt=""`, `strength = t*`, `num_inference_steps = 50`, `guidance_scale = 1.0` (no CFG). This is an SD-based "DiffPure-style" noise-then-denoise, NOT the original DiffPure; the output also passes through the SD VAE (docstring). Number of denoising steps actually executed is set by diffusers (general diffusers behavior: roughly `steps * strength`; not verified in this repository).
* `restore_swin2sr`: bicubic down by 0.5, then `caidas/swin2SR-classical-sr-x2-64` x2 super-resolution, resize to original if needed.
* `gridpure`: tiled GrIDPure (256 px windows, stride 128, 10 pure steps, 20 iterations, gamma 0.1) needing an external model file; NOT available in the runs (IMPLEMENTED BUT NOT RUN; jobs skipped with a log line).
* `adaptive_ae`: see 9.10.

The 13-purifier grid used in tier P is `eval.tiers.P` with group `purifiers_13` = jpeg_q90, jpeg_q75, jpeg_q50, jpeg_other, downup_0.5, downup_other, vae_rt, diffpure_0.05/0.1/0.2/0.3, restore_swin2sr, gridpure.

### 9.8 Metrics, PSR, net protection and bootstrap

**Identity score.** `faces.py: FaceScorer` runs insightface `buffalo_l` (detection + ArcFace `w600k_r50`) on the image padded by `pad_frac = 0.4 * max(H, W)` on every side with constant value 127 (S03/S05 found the detector missed 38 of 100 tightly cropped test faces unpadded); bbox/keypoints shifted back; the largest face by bbox area is used; embedding L2-normalised. `arcface_sim = cos(emb(edit), emb(source photo))` (`edit_runner.EditRunner._process`); NaN if either has no face. `face_detected = int(emb is not None)`.

**PSR hit for one edit** (`edit_runner.psr_hit`; vectorised in `analysis.Analysis._per_image`):
`hit = 1 if no face detected in the edit (or sim is NaN) else 1[ arcface_sim < tau_editor ]`. Counting "no face" as protection is the project's stated policy (`config.yaml: metrics.no_face_counts_as_success`), with the no-face rate always reported next to PSR. (In `analysis._per_image` a row with a detected face but NaN similarity or missing threshold yields NaN, not a hit.)

**Threshold tau** (`edit_runner.compute_psr_thresholds`, `analysis.thresholds_from_embeddings`): from the E0 reference (method `clean`, purifier `none`, i.e. editing clean photos), for each editor: pool, over all E0 clean-edit embeddings (all prompts, strengths and seeds of that editor), the cosine similarity between each clean edit of photo i and the SOURCE embedding of every other photo j != i that had a detected face; `tau_editor = np.percentile(pooled sims, 95)` (`eval.psr.percentile`); a pooled threshold over all editors is also stored. It assumes one image per identity in the split (risk R19 in the code). The threshold is the same for every purifier and method (computed once from the unpurified clean edits). FINAL_REPORT quotes tau = 0.101 for SD1.5 (report number).

**Per-image PSR and cell PSR** (`analysis.Analysis._per_image`, `cell`): for a cell (split, method, purifier, editor, strength): average `hit` over the edit seeds and prompts of each photo first (so the unit is the photo), then `PSR = mean over photos`. `no_face`, `sim` (mean ArcFace similarity), `edit_lpips`, `clip_i` are aggregated the same way.

**Net protection** (`scripts/readout_final.py: cell`; defined in the FINAL_REPORT metrics table): `net(method, cleanup) = PSR(method, cleanup) - PSR(clean, cleanup)`, paired per photo under the SAME cleanup, editor and strength: `d_i = PSR_i(method) - PSR_i(clean)`, reported as `mean(d_i)` with a percentile bootstrap CI over photos. Purpose: cleanups (DiffPure especially) change clean photos too, so raw PSR under cleanup overstates protection. "Paired difference to v1" is the same construction with v1 as the reference.

**Bootstrap** (`src/eval/stats.py: bootstrap_ci`; `analysis.boot_ci`): finite values only; `idx = rng.integers(0, n, size=(n_boot, n))` with `rng = default_rng(2026)`; `means = x[idx].mean(axis=1)`; CI = percentiles 2.5 and 97.5 of `means` (percentile bootstrap); config `stats: {n_boot: 10000, ci: 0.95, unit: image, bootstrap_seed: 2026}`. `analysis.paired_stats` builds `d = a - b` aligned by `image_id` (inner join, drop NaN), reports `mean_diff`, bootstrap CI of the mean, two-sided Wilcoxon signed-rank p (`scipy.stats.wilcoxon`, zeros dropped, `zero_method="wilcox"`, uncorrected for multiplicity), and shares of d > 0 and d < 0. `analysis.paired_ratio` bootstraps mean(a)/mean(b) jointly. `analysis.binom_ci` gives Clopper-Pearson intervals. DISAGREEMENT: `scripts/readout_final.py: boot_ci` (used for the final readout table cells and brackets) uses its own bootstrap with `n = 2000` resamples via `rng.choice(x, size=(2000, n), replace=True)` and `default_rng(2026)`, not the 10,000 of `stats:`/`analysis.py`. Which function produced a given number in `reports/FINAL_REPORT.md` is not recorded in the code; the reader should treat numbers from `final_readout.md` as 2000-resample intervals and numbers from `make_tables.py`/`analysis.py` as 10,000-resample intervals.

**Invisibility.** Per photo, protected PNG vs clean (`src/eval/metrics.py`, run in `run_protect.py`): `PSNR = 20 log10(255 / sqrt(MSE))` on uint8 arrays; `SSIM = skimage.metrics.structural_similarity(a, b, channel_axis=-1, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)`; LPIPS (VGG) on [-1,1] tensors; `png_*` columns decode the saved PNGs (what an attacker receives), `v1_*` columns replicate v1 (tensor-to-uint8 by truncation). Rule applied to every photo: PSNR >= 38 dB and SSIM >= 0.95 (`config.yaml: metrics.invisibility_min`). DISAGREEMENT: the `inv_psnr / inv_ssim / inv_lpips` columns inside the edit rows (`edit_runner.ssim_pil`) use skimage's default SSIM (uniform 7x7 window, `data_range=255`), not the Gaussian-window SSIM above, so `inv_ssim` and `png_ssim` differ in definition; the per-photo invisibility counts in the report come from `protect.csv` (`png_*`) per `readout_final.py`.

**Other per-edit metrics.** CLIP-I = cosine of CLIP ViT-L/14 image embeddings (`openai/clip-vit-large-patch14`, deliberately different from the protection loss's B/32) between the source photo and the edit (`clip_metric.ClipScorer`). Edit-LPIPS = LPIPS(VGG) between the clean-photo edit and the protected-photo edit with identical editor/prompt/strength/seed, both JPEG q90 round-tripped (`edit_runner._process`); computed in-job or later (`--stage lpips`); NOT reported at scale (see 8.5).

### 9.9 Tuning objective (Optuna; context for `v2_final`)

`scripts/tune_optuna.py`: TPE over alpha, eta (log-uniform 0.05-1), beta in [0, 1.5], gamma (log-uniform 20-150), antipure on/off and one shared weight for `w_hf = w_mono` in [0.1, 1.0]; constraints mean PSNR >= 38 and mean SSIM >= 0.95 of the trial's protected images (means over the search photos, not per photo: the reason `v2_final` violates the per-photo SSIM rule on 29 of 100 test photos per FINAL_REPORT); proxy objective (per the script docstring plan) = mean PSR over editors {SD1.5 img2img 0.55, IP-Adapter FaceID} x purifiers {none, jpeg_q75, diffpure_0.1} with 1 edit seed, 10 validation photos, 20 PGD iterations and K=2 draws; trial 0 = the hand-set default; top 3 confirmed on 20 validation photos with the final settings. FINAL_REPORT states the proxy's best trial (0.10) fell to 0.05 on the 20-photo confirm, and tuned versus hand-set is not separable (0.067 vs 0.042). Status: IMPLEMENTED AND RUN (S06).

### 9.10 The adaptive-attacker autoencoder (E8) - `src/eval/adaptive_ae.py`, `scripts/train_adaptive_ae.py`

Threat model: the attacker knows the protection method, protects its own pool of 120 other photos (`ae_pool`, disjoint from test/messy/train/val/calibration), and trains a network `f` on (protected, clean) pairs to remove the perturbation, then applies `f` to the victim's published photo before editing. Deliberately weak (docstring): small network, 120 images, about 1 GPU hour; it is not the LightShed implementation (only the idea).

* **Network** `AdaptiveAE(base_channels=32, depth=3)`: residual U-Net, `out = x + outc(decoder(x))`. Channels `[32, 64, 128, 256]`. Encoder: `inc` = two (conv3x3, GroupNorm, SiLU) layers 3->32; each of 3 down stages = stride-2 conv3x3 then a two-conv block. Decoder: nearest-neighbour upsample to the skip size, concatenate skip, two-conv block. Final conv3x3 32->3 is zero-initialised, so `f` starts as the identity. Input is reflect-padded to a multiple of `2**depth = 8` and cropped back (fully convolutional, any size).
* **Training pairs** (S15): protected images = `v2_final` applied to `ae_pool` with `eot.k_draws = 2` (override in `kaggle/jobs.yaml: S15`), paired with the clean pool images (so the attacker saw only `v2_final`-style perturbations; v1 and clean were evaluated with the same network). 10 percent of pairs held out for validation (`val_fraction 0.1`, `np.random.RandomState(seed).permutation`).
* **Optimisation** (`train_ae`): random aligned 128x128 patches from the uint8 pairs mapped to [-1,1], random horizontal flip (p 0.5) and rot90 k in {0..3}; loss `L1(f(x_prot), x_clean)` (+ `lpips_weight * LPIPS` with weight 0, so L1 only); AdamW lr 2e-4, weight decay 0; batch 16; gradient-norm clip 1.0; wall-clock budget `minutes = 60` (S15 `ae_minutes: 60`); checkpoint every 500 steps; seed 2026. Reported by FINAL_REPORT: 35,496 steps; PSNR of AE output vs clean photo 45.1 dB.
* **Use as a cleanup**: `AdaptiveAEPurifier.apply`: whole 512x512 protected image -> `[-1,1]` -> `f` -> clamp -> round to uint8 (deterministic, seed unused). Evaluation: tier `E8` (clean, v1, v2_final; purifier `adaptive_ae`; SD1.5 img2img 0.55 and FaceID).
* **Reported outcome** (FINAL_REPORT 7.6, report numbers): PSR under the AE 9.3 percent clean, 11.0 percent v1 (net +1.7 [-1.0, +4.7]), 10.7 percent v2_final (net +1.3 [-1.0, +3.7]); i.e. the protection is removed. NOT run against the target-steered variants (v2_tgt_*), which are the recommended methods; this is the main unresolved caveat in the report. Status: IMPLEMENTED AND RUN for v1 and v2_final; IMPLEMENTED BUT NOT RUN for v2_tgt_*.


---


Source of truth: repository `C:\Users\karth\Documents\DiffShield-wt\integration` (branch v2/release). All paths below are relative to that repo root. Status labels: IMPLEMENTED AND RUN (executed on Kaggle GPU with real weights, or at least in the sessions listed), IMPLEMENTED BUT NOT RUN (code and CPU tests exist, never executed with real models or never produced reported numbers), PROPOSED ONLY (named in config/docs, no code consumer).

## 10. Implementation details

### 10.1 Architecture in one paragraph
The code is split in two halves that must not import each other (enforced by `tests/test_isolation.py`): Half A = protection (`src/protect/*`, `src/baselines/*` which only import `src/utils`), Half B = attack-and-score evaluation (`src/eval/*`). Shared helpers are in `src/utils/*`. Everything is driven by `config/config.yaml` + `config/parts/*.yaml` (deep-merged, files in sorted order, then a method preset, then `--set a.b.c=value` overrides; `src/utils/config.py::load_config`). Jobs run on Kaggle through `kaggle/job_runner.ipynb` -> `scripts/run_job.py` -> `kaggle/jobs.yaml`. A protector has one frozen interface: `protect(x [1,3,H,W] in [-1,1], seed) -> (tensor, info_dict)`.

### 10.2 Module table: `src/protect/*` (Half A)

| File | Purpose | Key classes / functions | Important parameters (source) | Status |
|---|---|---|---|---|
| `src/protect/registry.py` | Method name -> Protector. Lazy imports. `v1` -> v1 adapter; names starting `v2_`, `abl_`, `eps` -> v2 registry; `noise_*`, `photoguard`, `mist` -> baselines registry; otherwise `KeyError`. | `build_protector(name, cfg, device, backend)`, `_is_v2`, `_is_baseline` | `runtime.backend` (`real`/`tiny`) | RUN |
| `src/protect/v1.py` | Adapter that exposes the frozen v1 algorithm (`src/optimizer.py` PGD + `src/losses.py` DiffShieldLoss) through the Protector interface. Refuses to run preset `v1` unless eps = 8/255 and step = 1/255 on [-1,1] (= 4 and 0.5 on the standard scale) unless `v1_strict=false`. Falls back from `runwayml/stable-diffusion-v1-5` to the `sd15` mirror id if the first cannot be loaded (records `model_id_used`). | `V1Protector`, `build_v1` | `budget.eps_255`, `budget.step_255`, `budget.random_start` (must be `uniform_full`), `loss.alpha/beta/gamma`, `loss.sem_target_text` ("a potted plant"), `models.sd15_v1`, `models.clip_sem` (`src/protect/v1.py`, `config/config.yaml`) | RUN (J01, S08 etc.) |
| `src/protect/v2_registry.py` | Resolves a v2 preset to a config and builds the protector lazily (models load on first `protect` call). See 10.6. | `PRESETS` tuple, `_CHAIN`, `deep_merge`, `resolve_cfg`, `V2Protector` (`_build`, `protect`, `unload`), `build_v2_protector` | `PRESETS = (v2_cheap, v2_main, v2_antipure, v2_final, abl_vis_only, abl_den_only, abl_no_sem, abl_no_pur, eps2, eps6, eps8, v2_tgt_zero, v2_tgt_zero_mid, v2_tgt_zero_hi, v2_tgt_gray)` | RUN |
| `src/protect/pgd.py` | The v2 protection loop: sign-PGD with Expectation-over-Transformation (EoT) over randomly drawn differentiable cleanup operators ("purifiers"). One backward pass per EoT draw (memory of one draw). Quantises the input to the uint8 grid and returns an output on the uint8 grid so the saved PNG respects the budget exactly. | `PGDParams` (dataclass), `params_from_cfg`, `pgd_protect`, `to_u8`, `from_u8` | `eps_255` (default 4), `step_255` (0.5 dataclass default; presets set 1.0), `iters` 40, `k_draws` 4, `random_start` uniform_full, `alpha, beta, gamma, eta, lambda_id, w_hf, w_mono`, `loss_scale` (from `runtime.loss_scale`, 1024), `t_range` (50,300) from `purifier_train_params.sd_onestep`, `quantize` (from `protect.quantize`, true), `loss_norm` (only if `loss.normalize_terms`) (`src/protect/pgd.py::params_from_cfg`) | RUN |
| `src/protect/losses.py` | Loss terms. PGD ascends on `total = alpha*L_vis + eta*L_den - beta*L_sem - gamma*L_str - lambda_id*cos_id [+ anti-purify terms]`. | `l_vis` (MSE of unscaled VAE latent means, purified protected vs clean), `l_den` (MSE between true noise and UNet noise prediction at a random timestep, empty prompt), `l_sem` (1 - cos to the target text embedding, CLIP B/32), `ssim`, `l_str` (1 - SSIM(x, x_pub), 11x11 Gaussian window sigma 1.5), `psnr_uint8`, `TermScales`/`scales_from_infos` (optional per-term normalisation; `normalize_terms: false` so unused), `SingleVis`, `TargetVis` | `loss.ssim: {window: 11, sigma: 1.5}`; `TargetVis(kind in {zero, gray}, w_tgt, w_dev)` | RUN (TargetVis RUN in S25-S28) |
| `src/protect/models.py` | Frozen model bundle. Lazy imports. `SDModels` holds VAE (fp32 weights, optional fp16 autocast), UNet (fp16 by default), alphas_cumprod, null-prompt condition. Scaling factor read from `vae.config`, never hard-coded. | `SDModels` (`encode_mean`, `decode`, `unet_eps`, `alpha_bar`, `set_grad_checkpointing`), `load_sd15`, `load_sdxl_vae`, `ClipSem`, `load_clip_sem`, cached `get_sd15/get_sdxl_vae/get_clip_sem`, `make_generator`, `g_randn_like`, `g_uniform`, `g_randint`, `sd15_alphas_cumprod(n=1000, beta_start=0.00085, beta_end=0.012)` | `runtime.unet_dtype` fp16, `runtime.vae_dtype_protect` autocast, `runtime.grad_checkpointing` true, `models.*` ids (`src/protect/models.py`, `config/config.yaml`) | RUN |
| `src/protect/purifiers.py` | Differentiable training purifiers and the mixture sampler. Interface `TrainPurifier.forward(x, gen) -> x`. | `Identity`, `DiffJPEG` (kornia `jpeg_codec_differentiable`; alternative `impl: pil_bpda` = real PIL codec forward, straight-through backward), `DownUp` (random scale and mode, antialias), `VAERoundTrip` (encode mean -> decode), `SDOneStepPurifier` (noise to t in [50,300], one UNet step, reconstruct; also returns `aux` reused by L_den and anti-purify terms), `PurifierMixture` (`sample`, probabilities renormalised over the active names, excludes `downup_heldout`), `build_purifier`, `build_mixture`, `build_heldout` | `purifiers_train`, `purifier_train_probs`, `purifier_train_params.*` (jpeg quality [50,95], downup scale [0.5,0.9] bilinear/bicubic antialias, sd_onestep t_min 50 t_max 300 empty prompt), `purifier_train_exclude: [downup_heldout]`, `downup_heldout` scale [0.30,0.45] (`config/config.yaml`, `config/parts/protect.yaml`) | RUN (heldout variant: IMPLEMENTED, used only in CPU tests) |
| `src/protect/antipure.py` | Anti-purification terms (AntiPure idea), both MAXIMISED, only evaluated on EoT draws where the one-step purifier was drawn. `l_hf`: 8x8 patch DCT, mean squared high-frequency coefficient energy, normalised by the clean image's value (dimensionless ratio ~1 at start). `l_mono`: error-grows-with-t term (one extra one-step pass per such draw). | `dct_matrix`, `hf_energy(patch=8, hf_cut=8)`, `AntiPure` (`prepare`, `terms`) | `loss.anti_purify: {enabled, w_hf, w_mono}` | RUN (v2_antipure S07/S12, v2_final) |
| `src/protect/ensemble.py` | Encoder ensemble for L_vis: average of the unscaled latent-mean MSE over SD1.5 VAE plus extra VAEs (SDXL VAE fp16-fix). | `EnsembleVis` | `protect.ensemble: {enabled, vaes: [sdxl_vae_fp16], weights: {}}` (`config/parts/protect.yaml`); enabled only in `v2_final_default` (and hence v2_final, eps2/6/8) | RUN |
| `src/protect/face_loss.py` | L_id: identity loss through a differentiable face embedder (facenet-pytorch InceptionResnetV1 vggface2, deliberately different from the evaluation ArcFace). Loaded only when `lambda_id > 0`. | `FaceEmbedder`, `load_face_embedder`, `IdentityLoss`, `_crop_resize` | `face: {backend: facenet, pretrained: vggface2, bbox: null}` (`config/parts/protect.yaml`) | IMPLEMENTED BUT NOT RUN (`lambda_id = 0` in every preset; facenet-pytorch not in requirements) |
| `src/optimizer.py`, `src/losses.py`, `src/eot.py`, `src/data.py`, `src/gradcam.py` | Original v1 code (PGD, DiffShieldLoss, EoT layer with JPEG/crop, dataset, Grad-CAM). `src/losses.py` and `src/eot.py` carry the three v1 runtime patches as real code (paired EoT halves, SSIM structure loss, loss sign), checked by `tests/test_v1_patches.py`. | `PGDOptimizer`, `DiffShieldLoss`, `EoTLayer`, `CelebAHQDataset`, `GradCAMDiffusion` | v1 EoT: `eot.v1: {jpeg_quality: [50,100], crop_scale: [0.8,1.0], p: 0.5, same_on_batch: true}` | v1 pieces RUN; Grad-CAM not used by v2 |
| `src/legacy_eval.py` | The v1 ControlNet evaluator moved here, with a lazy shim in `src/eval/__init__.py` (`__getattr__`) for old imports. | `Evaluator`, `calculate_psnr` | - | legacy, not used in v2 numbers |

### 10.3 Module table: `src/baselines/*`

| File | Purpose | Key classes / functions | Important parameters (source: `config/parts/baselines.yaml`) | Status |
|---|---|---|---|---|
| `src/baselines/registry.py` | `build_baseline(name, cfg, device)` for `noise_uniform`, `noise_gauss_psnr`, `photoguard`, `mist`; minimal self-contained SD component loader (VAE, plus UNet+text encoder for Mist). | `build_baseline`, `canonical`, `resolve_params`, `load_sd_components`, `vae_mean_encode_fn`, `unet_eps_fn`, `_load_target_image` | `baselines._common: {eps_255: null (= budget.eps_255), dtype: auto}` | RUN |
| `src/baselines/noise_control.py` | Noise controls and shared budget helpers (all baselines return outputs exactly on the uint8 grid with max change <= floor(eps_255)). | `uniform_linf`, `sign_linf`, `gaussian_matched_psnr` (bisection on sigma to hit a target PSNR, tol 0.02), `UniformNoiseProtector`, `GaussianPSNRProtector`, `quantize_to_budget`, `linf_u8`, `psnr_u8`, `load_psnr_map` | `noise_uniform.variant: uniform`; `noise_gauss_psnr: target_psnr 41.911 (fallback), psnr_csv (S22 sets data/v1_ref/stage1_per_image_metrics.csv for per-image PSNR matching), psnr_key_col Original_Filename, psnr_val_col PSNR_dB` | RUN (S22) |
| `src/baselines/photoguard.py` | PhotoGuard encoder attack, re-implementation checked against official MadryLab/photoguard@686bea75c786cb46c88fc396a0cd0ee3d7d28c2e (notebooks demo_simple_attack_img2img cell 7). Labelled "reimplementation". | `EncoderAttackProtector` (`target_latent`, `protect`), `step_size_at`, `project` | `iters 1000, step_255 2.55, step_decay true (linear to step/100), target zero_latent, loss_scale 1024`; eps = OUR budget 4/255 (official 0.06 on [-1,1] not adopted); diffusion-attack variant out of scope | RUN (S13, S24) |
| `src/baselines/mist.py` | Mist fused semantic+textural loss, re-implementation checked against psyker-team/mist@99f5f3c455022bef77ef25ba440a4289ea472d25. Official target image `MIST.png` is fetched at run time (S21 step puts it in `{out}/assets/MIST.png` and passes `baselines.mist.target_image`); if absent, a procedural stand-in texture is used and must be reported as a deviation. | `MistProtector` (`semantic_loss`, `textural_loss`, `protect`), `make_texture_target` | `iters 100, step_255 1.0, w_sem 10000, w_tex 1.0, t_min 0, t_max 1000, n_noise 1, prompt "a painting", loss_scale 1024, grad_checkpointing true`; eps = OUR budget (official 16 not adopted) | RUN (S21, S24) |

### 10.4 Module table: `src/eval/*` (Half B) and `src/utils/*`

| File | Purpose | Key classes / functions | Important parameters (source) | Status |
|---|---|---|---|---|
| `src/eval/edit_runner.py` | Core edit-and-score loop for a tier. Row key `(method, image_id, seed, editor, prompt_id, strength, purifier)`; resume by row key; progress and `gpu_time.json` checkpointed every `eval.checkpoint_every` (50) rows; clean-edit cache, source-embedding cache. Heavy dependencies enter through backends so the whole loop runs on CPU with fakes. | `Job`, `expand_tier`, `count_rows`, `ResultsWriter`, `EmbCache`, `SourceStore`, `CleanEditCache`, `derive_seed`, `jpeg_roundtrip`, `load_eval_config` | `config/parts/eval.yaml` (all `eval.*` keys, tiers) | RUN |
| `src/eval/editors.py` | The attacker's editors, DPM-Solver++ multistep scheduler for all, fp16. | `SD15Img2Img`, `SDXLImg2Img`, `InstructPix2Pix`, `SDInpaint` (keep-face mask), `IPAdapterFaceID` (run as img2img with identity injected via IP-Adapter), `build_editor`, `set_dpmpp` | `eval.num_inference_steps 30`, `guidance_scale 7.5`, `ip2p_image_guidance 1.5`, per-editor settings in `eval.editors` (see 11.3) | RUN (all five editors) |
| `src/eval/purify_attacks.py` | Held-out, real-codec test-time cleanups. Contract `EvalPurifier.apply(PIL, seed) -> PIL`. Names: none, jpeg_q90/q75/q50/other, downup_0.5/other, vae_rt, diffpure_0.05/0.1/0.2/0.3, restore_swin2sr, gridpure, adaptive_ae. Unavailable purifiers raise `NotAvailable` and are skipped and logged. | `NonePurifier`, `JpegPurifier`, `DownUpPurifier`, `VaeRoundTrip`, `DiffPurePurifier` (SD1.5 img2img with empty prompt at strength 0.05-0.3, 50 steps, guidance 1.0), `Swin2SRRestore`, `canonical_name` | `eval.purifiers.*`, `eval.purifier_aliases` | RUN except gridpure |
| `src/eval/gridpure.py` | GrIDPure (Zhao et al., CVPR 2024) re-implementation from the official README (UNVERIFIED against cloned code). Needs `256x256_diffusion_uncond.pt` and the GrIDPure repo. | `get_crop_boxes`, `pack_corners`, `grid_pure_pass`, `gridpure`, `GuidedDiffusionPatchFn`, `GridPurePurifier` | window 256, stride 128, pure_steps 10, iters 20, gamma 0.1, `checkpoint: null`, `repo_path: null` | IMPLEMENTED BUT NOT RUN (model file missing; purifier skipped; GrIDPure unavailable) |
| `src/eval/adaptive_ae.py` | E8 adaptive attacker: small conv U-Net autoencoder trained on (protected -> clean) pairs of the 120-image `ae_pool` split, 128 px patches, applied fully convolutionally. | `AdaptiveAE`, `PatchSampler`, `train_ae`, `load_model`, `AdaptiveAEPurifier`, `build_purifier` | `adaptive_ae: {weights_path results/adaptive_ae/ae_final.pt, base_channels 32, depth 3, patch 128, train: {minutes 60, batch_size 16, lr 0.0002, lpips_weight 0.0, val_fraction 0.1}}` | RUN (S15, against clean, v1, v2_final only) |
| `src/eval/faces.py` | ArcFace scoring through insightface `buffalo_l`. Pads the image (`pad_frac` 0.4) before detection and maps boxes back (the detector missed 38 of 100 test faces otherwise). File lock around pack init; refuses a pack without both detection and recognition; error on missing embedding; CUDA provider first, CPU fallback; `onnxruntime.preload_dlls()` when present. | `FaceScorer` (`load`, `analyze`, `embed`, `similarity`), `choose_providers`, `cosine`, `cosine_matrix` | `eval.models.arcface: {pack buffalo_l, det_size [640,640], pad_frac 0.4}` | RUN |
| `src/eval/clip_metric.py` | CLIP-I (source vs edit) with CLIP ViT-L/14, deliberately different from the loss's B/32; LPIPS(VGG) edit distance on JPEG-q90 round-tripped clean-edit vs protected-edit. Uses vision_model + visual_projection (transformers 5 return type). | `ClipScorer`, `LpipsScorer`, `edit_lpips` | `eval.models.clip_eval`, `eval.models.lpips_net: vgg` | CLIP-I RUN; edit-LPIPS IMPLEMENTED, run on Kaggle was moved to local CPU (`scripts/compute_edit_lpips.py`), only 513 pairs computed, NOT reported |
| `src/eval/masks.py` | Keep-face inpainting masks (255 = repaint). From CelebAMask-HQ parsing annotations (keep skin, nose, eyes, brows, ears, mouth, lips, glasses, ear ring, neck, dilated 8 px) or a face-bbox ellipse fallback. | `MaskProvider`, `repaint_mask_from_parsing`, `repaint_mask_from_bbox`, `ensure_face_unmasked` | `eval.masks: {keep_parts [...], dilate_px 8}` | RUN |
| `src/eval/metrics.py` | Invisibility metrics: v1 convention (tensor -> uint8 by truncation, replica of the v1 function) and png convention (decode saved files); LPIPS(VGG). | `psnr_u8`, `ssim_u8`, `max_abs_diff_u8`, `diff_histogram_u8`, `LPIPSScorer(net="vgg")`, `v1_quality_metrics`, `png_quality_metrics` | - | RUN |
| `src/eval/stats.py`, `src/eval/analysis.py` | Bootstrap CIs over images (10000 resamples, 95%, seed 2026), paired differences, Wilcoxon signed-rank (uncorrected), binomial CI, PSR thresholds (95th percentile of ArcFace similarity between different people), gate logic, "not run" cells never fabricated. | `bootstrap_ci`, `summarize`, `paired_bootstrap_diff`, `boot_ci`, `paired_stats`, `paired_ratio`, `binom_ci`, `thresholds_from_embeddings`, `Analysis`, `Spec` | `stats: {n_boot 10000, ci 0.95, unit image, bootstrap_seed 2026}`, `metrics.psr_impostor_percentile 95`, `psr_threshold_scope per_editor` | RUN |
| `src/eval/schema.py` | Column constants for results CSVs (`PROTECT_COLUMNS`, edit columns). | constants | - | RUN |
| `src/utils/config.py` | Config loader, overrides, validation, hash. | `load_config`, `deep_merge`, `apply_overrides`, `cfg_get`, `validate_config`, `config_hash` | validation: budget.scale == standard_0_255, 0 < eps_255 <= 16, 0 < step_255 <= eps_255, iters >= 1, positive mass in `purifier_train_probs` | RUN |
| `src/utils/budget.py` | Budget conversions (standard 0-255 <-> [-1,1]: `eps_model = eps_255*2/255`), uint8 conversions (round vs v1 truncation). | `to_model_scale`, `to_standard_scale`, `tensor_to_u8_round`, `tensor_to_u8_trunc_v1` | - | RUN |
| `src/utils/seeding.py` | Deterministic seeds via sha256 (never Python `hash()`). | `derive_seed(base, *parts)`, `seed_everything`, `torch_generator` | base seed 2026; per-image protect seed = `derive_seed(seed, "protect", image_id)` (`scripts/run_protect.py`) | RUN |
| `src/utils/runlog.py` | Run logging (see 10.7). | `start_run`, `finish_run`, `RunContext`, `GpuTimer`, `collect_env`, `collect_versions`, `get_git_info`, `make_run_id` | - | RUN |
| `src/utils/io.py` | Image listing (sorted), sharding, v1-compatible load/save. | `list_images`, `shard_items`, `parse_shard`, `load_tensor_v1`, `save_png_v1`, `load_u8` | - | RUN |
| `src/utils/tiny_models.py` | Offline random-init tiny stand-ins for CPU tests and `--backend tiny`. | `tiny_vae`, `tiny_clip_vision/text`, `tiny_target_embedding`, `tiny_v1_loss` | - | used by tests |

### 10.5 Module table: `scripts/*`, `kaggle/*`

| File | Purpose | Notes / parameters | Status |
|---|---|---|---|
| `scripts/run_protect.py` | Protect a folder of images with checkpoint/resume and sharding; writes protected PNGs, `metrics.csv` (per-image PSNR/SSIM/LPIPS in both v1 and png conventions, GPU seconds, peak memory), `info/<id>.json` loss histories, `resolved_method_cfg*.json`. | args: `--method`, `--input_dir`, `--list`, `--limit`, `--run_id`, `--shard K/N`, `--resume`, `--resume_from`, `--checkpoint_every` (5), `--backend real\|tiny`, `--device`, `--set a.b=c`; `--limit` sorts ids, same as evaluation | RUN |
| `scripts/merge_shards.py` | Merge `metrics.shard*.csv`, summaries, gpu_time into single files; `--also` merges the same run id from other accounts. | | RUN |
| `scripts/evaluate.py` | Half B CLI: edit + score for a tier / method / split; `--jobs_file`, `--count_only`, sharded; `--purifier all` for tiers P/A/E9/E8. | tiers in `config/parts/eval.yaml`: H, M, P, SCREEN, TGT, AP, A, E9, E8, SMOKE | RUN |
| `scripts/run_job.py` | Runs sessions from `kaggle/jobs.yaml` sequentially: macros, steps with `sharded` (one process per GPU, `CUDA_VISIBLE_DEVICES=g`, `--shard K/N`), `workers` (Optuna), `optional`, `soft`, `when`, `timeout_min`, `builtin` steps (pip_install, resolve_splits, make_config, set_var). DONE marker, `kaggle_session.json`, zip of outputs; resumable (`--resume_from`); `--dry_run`. | | RUN |
| `scripts/tune_optuna.py` | Optuna TPE tuning of v2_final on the VALIDATION split (S06). Searches alpha [0.05,1] log, eta [0.05,1] log, beta [0,1.5], gamma [20,150] log, antipure on/off and one shared weight in [0.1,1] (w_hf = w_mono). Not searched: eps, lambda_id, K, iterations. Constraint: mean PSNR >= 38 and mean SSIM >= 0.95 over the search photos (mean, not per photo). Defaults: 16 trials, 10 images, 20 iters, K=2, 1 edit seed, SQLite storage, hand-set default enqueued as trial 0, confirmation of the top 3 on 20 val images. Objective: mean PSR over editors (SD1.5 s0.55, FaceID) x purifiers {none, jpeg_q75, diffpure_0.1}. Writes `final_candidate.yaml` which became `config/parts/final.yaml`. | RUN (S06; hit 12 h timeout; trials in `reports/optuna_trials_S06.csv`, `reports/optuna_confirm_S06.json`) |
| `scripts/prepare_data.py`, `scripts/make_splits.py`, `scripts/make_smoke_splits.py`, `scripts/prepare_v1_data.py` | Build frozen splits from CelebAMask-HQ (test 100, messy 20, train 20, val 20, ae_pool 120; v1 calibration images excluded; ArcFace de-duplication), materialise 512 px PNGs (messy gets degradation), prepare v1 evaluation inputs (JPEG). Smoke splits are throw-away. | RUN (S03 for splits; smoke splits smoke only) |
| `scripts/budget_check.py`, `scripts/compare_runs.py`, `scripts/check_env.py`, `scripts/env_probe.py`, `scripts/gpu_ledger.py`, `scripts/merge_requirements.py`, `scripts/smoke_report.py` | Phase 0 budget measurement (4/255 confirmed), paired comparison against v1 reference CSV, pin checker (ignores `+local` part), Kaggle environment probe (J00), GPU-hours ledger, requirements merge, smoke cost report. | RUN |
| `scripts/ingest_results.py`, `scripts/make_tables.py`, `scripts/make_figures.py`, `scripts/make_pareto_figure.py`, `scripts/readout_final.py`, `scripts/finalize_all.py` | Combine Kaggle zips into validated de-duplicated CSVs (smoke sessions excluded, safe unzip), generate tables (Markdown, CSV, long CSV, LaTeX) and figures from CSVs, final readout, one-command chain. | RUN |
| `scripts/train_adaptive_ae.py` | Train the E8 autoencoder (60 min budget, resumable checkpoints). | RUN (S15) |
| `scripts/compute_edit_lpips.py` | Fill the edit-LPIPS column locally (LPIPS VGG on JPEG q90). | partial, not reported |
| `scripts/demo.py` | Protect one photo and show editing; `--cpu-dry-run` tests wiring with fake models. | CPU dry run tested (`tests/test_analysis_demo.py`); not part of reported numbers |
| `scripts/make_human_study.py`, `scripts/analyze_human_study.py` | Offline HTML form for a human study with separate answer key; analysis of rater CSVs. | IMPLEMENTED BUT NOT RUN with humans (no human study was done; tested on synthetic answers) |
| `kaggle/jobs.yaml` | Job and session definitions (J00, J01, J01b, S01-S28 incl. S04t/S04u/S05). Macros: `setup`, `prep`, `protect`, `evaluate`. | see 11.4 for which preset ran where | RUN |
| `kaggle/job_runner.ipynb` | The only notebook the user edits: `JOB_IDS`, `PARAMS`, `RESUME_FROM`, `JOB_SHARD`, `FORCE`; finds the newest code bundle under `/kaggle/input`, unpacks to `/kaggle/tmp/DiffShield`, runs `scripts/run_job.py`, prints summary. | RUN |
| `kaggle/build_bundle.py` | Builds `kaggle/dist/diffshield-code-<hash7>.zip` with `src scripts config tests requirements.d data/splits data/v1_ref`, `requirements*.txt`, `pytest.ini`, `kaggle/jobs.yaml`, `kaggle/README.md`, `kaggle/job_runner.ipynb`, `BUNDLE_INFO.json`; never images/results/.git/.venv; refuses a dirty tree without `--allow_dirty`. | RUN (bundle history B0 to B9) |
| `kaggle/README.md` | Runbook (account setup, datasets, accelerator GPU T4 x2, internet ON, resume, sharding across accounts). | |

### 10.6 How a method preset is resolved (config -> protector)
1. `scripts/run_protect.py` calls `load_config(config.yaml, method=<name>, overrides=[..., "runtime.backend=real"])` (`src/utils/config.py`): load `config/config.yaml`; deep-merge every `config/parts/*.yaml` in sorted order (baselines, data, eval, final, protect); then merge `cfg["methods"][<name>]` over the root. Special cases: `v2_final` falls back to `methods.v2_final_default` if no tuned `methods.v2_final` exists; `noise_*`, `photoguard`, `mist` have no preset and read their `baselines:` block; any other missing preset raises `KeyError`. `--set a.b.c=value` overrides (YAML-parsed) are applied last; `validate_config` checks budget sanity.
2. `build_protector(name, cfg, device)` (`src/protect/registry.py`) dispatches by name prefix.
3. `V2Protector.__init__` calls `resolve_cfg(name, cfg)` (`src/protect/v2_registry.py`): re-reads `config/parts/protect.yaml` for the `methods` and top-level keys (so a minimal cfg also works), deep-merges, and layers presets by the chain `_CHAIN`: `v2_final` = [`v2_final_default`, `v2_final`]; `eps2/eps6/eps8` = [`v2_final_default`, `v2_final`, `eps<N>`]; every other preset is a one-element chain. Reason: `final.yaml` (tuned) sorts BEFORE `protect.yaml`, so `methods.v2_final` cannot be defined in protect.yaml without being overwritten; hence the hand-set start is named `v2_final_default`, and the tuned values from `final.yaml` are layered above it. Later layers win. `eps*` therefore carry the tuned v2_final loss with only `budget.eps_255` changed. The resolved config is saved per run as `resolved_method_cfg.shard{K}of{N}.json` (e.g. `results/S04/S04/results/prot_eps2_test/resolved_method_cfg.shard0of2.json`).
4. `params_from_cfg` turns the resolved config into `PGDParams`. On the first `protect` call `_build` loads SD1.5 (cached process-wide in `models._CACHE`), builds the purifier mixture (`build_mixture`), chooses the L_vis object: `EnsembleVis` if `alpha > 0` and `protect.ensemble.enabled`; else `TargetVis` if `loss.target` is set; else `SingleVis`. It loads CLIP B/32 only if `beta != 0`, the face embedder only if `lambda_id != 0`, and `AntiPure` only if `loss.anti_purify.enabled` (building a one-step purifier if it is not in the mixture).
5. `pgd_protect` runs the loop (10.7).

### 10.7 Runtime features
- **fp16 autocast VAE (protection).** `runtime.vae_dtype_protect: autocast` makes the VAE forward/backward run under `torch.autocast("cuda", dtype=float16)` while weights stay fp32 (`SDModels._vae_ctx`, `src/protect/models.py`). Config comment: S04u measured about 2x faster with the same loss trajectory as fp32 (4 images, 10 iterations); S04t had measured the fp32 VAE at 3-6x the planned cost on a T4. UNet in fp16 (`runtime.unet_dtype`). No-op on CPU (`tests/test_protect_fakes.py::test_vae_autocast_flag_is_noop_on_cpu`).
- **fp16 safety.** The scalar loss is multiplied by `runtime.loss_scale` (1024) before `autograd.grad` and divided after; sign-PGD is scale-invariant (`tests/test_protect_pgd.py::test_loss_scale_does_not_change_the_result`). Non-finite per-draw gradients are skipped and counted (`nonfinite_draws` in info); the fraction of exactly-zero gradient pixels is logged.
- **Gradient checkpointing.** `runtime.grad_checkpointing: true` calls `enable_gradient_checkpointing()` on UNet and VAE and keeps them in `train()` mode (needed for checkpointing to be active in some diffusers versions; SD1.5 UNet/VAE have no dropout/batch-norm). `tests/test_protect_diffusers_api.py::test_checkpointing_does_not_change_outputs` checks equal outputs. Also `baselines.mist.grad_checkpointing: true`.
- **One backward per EoT draw** (`k_draws` 4 by default) so peak memory is that of one draw; published-image terms (L_str, L_id) are differentiated once per iteration.
- **Quantised output.** Input snapped to uint8 grid; final delta rounded to integer gray levels within +-floor(eps_255); saved PNG respects the budget exactly.
- **Determinism.** One `torch.Generator` per (image, seed) for random start, purifier choice, JPEG quality, timesteps, noise; per-image seed `derive_seed(2026, "protect", image_id)`. GPU kernels may still be non-deterministic.
- **Resumability.** Protection: `scripts/run_protect.py` skips images that are in `metrics*.csv` and have a PNG, writes `progress*.json` every `checkpoint_every` (5) images, `--resume_from` copies a previous run directory. Evaluation: row-key resume, checkpoint every 50 rows. Jobs: DONE marker, `--resume_from`. Optuna: SQLite `load_if_exists`, stale trials retried once. `start_run` refuses a run directory whose config hash differs.
- **2-GPU sharding.** `sharded: true` steps in `kaggle/jobs.yaml` start one process per GPU with `CUDA_VISIBLE_DEVICES=g` and `--shard g/N` (T4 x2 -> `shard0of2`, `shard1of2`); items are split by `shard_items`; shard CSVs/summaries are merged by `scripts/merge_shards.py`. Job-level sharding across Kaggle accounts: `--job_shard K/N`. Optuna runs two worker processes (`workers: true`) on one SQLite study.
- **Logging.** `RunContext` (`src/utils/runlog.py`) writes per run: resolved `config.json`/`config.yaml`, `git_hash.txt` (+dirty flag), `versions.json` (library versions), `env.json` (+nvidia-smi), `seed.txt`, `command.shard*.txt`, `logs/run.shard*.log`, `metrics.shard*.csv`, `progress.shard*.json`, `gpu_time.shard*.json`, `summary.shard*.json`; session-level `kaggle_session.json` and `results/gpu_ledger.csv`. Per-image `info/<id>.json` holds loss-term histories, purifier draw counts, PSNR/SSIM, GPU seconds, peak memory.
- **Failure isolation.** One editor failing does not abort the others (`failed_editors.json`); unavailable purifiers are skipped and logged; optional/soft steps in jobs.

### 10.8 Implemented-but-unused config keys (do not describe as results)
- `eot.spatial: {enabled: false, crop_scale [0.8,1.0], p 0.5}`: no consumer found in `src/protect` or `src/baselines` (grep for `spatial`/`crop_scale` finds none) -> PROPOSED ONLY. (`eot.v1` crop/JPEG belongs to the v1 EoT layer `src/eot.py`.)
- `loss.normalize_terms: false` + `loss_norm`: per-term scaling is implemented (`TermScales`) but off in every preset.
- `loss.lambda_id`: 0.0 everywhere; L_id never evaluated.
- `eval.purifiers.gridpure`, `purifiers_heldout` entries `diffpure_t0.1..0.3, gridpure`: gridpure unavailable.
- `models.ipadapter_faceid` etc. are used by evaluation only.

## 11. Parameters and configuration

Merge order at load time: `config/config.yaml`, then `config/parts/*.yaml` in sorted order (baselines, data, eval, final, protect), then the method preset (`methods.<name>`), then `--set` overrides. All budgets are on the STANDARD 0-255 scale; tensors are in [-1,1] so `eps_model = eps_255*2/255`. The six files below are reproduced verbatim (copied by script from the repo).

### 11.1 Verbatim configuration files

#### `config/config.yaml`

```yaml
# DiffShield v2: single source of truth for hyperparameters, seeds and paths.
# ALL perturbation budgets are on the STANDARD 0-255 scale. Tensors in code are in [-1, 1],
# so code converts with eps_model = eps_255 * 2 / 255 (src/utils/budget.py). Never hard-code 8/255.
project: diffshield
config_version: 1
seed: 2026                     # base seed; per-item seeds = derive_seed(seed, purpose, item_id) (sha256)

paths:
  results_root: results
  data_root: data
  splits_dir: data/splits
  cache_dir: results/cache
  celebahq_root: null          # dir containing CelebA-HQ-img/ and CelebAMask-HQ-attribute-anno.txt
                               # null on Kaggle => auto-discover under /kaggle/input (same logic as v1)

models:                        # ids UNVERIFIED until env probe / first load; then pin `revision` to the commit sha
  sd15:          {id: "stable-diffusion-v1-5/stable-diffusion-v1-5", revision: null}
  sd15_v1:       {id: "runwayml/stable-diffusion-v1-5", revision: null}      # id used by v1; reproduction only
  sd15_inpaint:  {id: "stable-diffusion-v1-5/stable-diffusion-inpainting", revision: null}
  sdxl:          {id: "stabilityai/stable-diffusion-xl-base-1.0", revision: null}
  sdxl_vae_fp16: {id: "madebyollin/sdxl-vae-fp16-fix", revision: null}
  ip2p:          {id: "timbrooks/instruct-pix2pix", revision: null}
  ipadapter_faceid: {id: "h94/IP-Adapter-FaceID", weight: "ip-adapter-faceid_sd15.bin", revision: null}
  clip_sem:      {id: "openai/clip-vit-base-patch32", revision: null}     # L_sem (v1)
  clip_eval:     {id: "openai/clip-vit-large-patch14", revision: null}    # CLIP-I metric; deliberately != clip_sem
  lpips_net: vgg
  arcface_eval:  {pack: buffalo_l, det_size: [640, 640]}                  # insightface; evaluation only

budget:
  scale: standard_0_255
  eps_255: 4                   # confirmed by J01 budget check: v1 code 8/255 on [-1,1] == 4/255 standard
  eps_confirmed: true          # J01 (2026-10-14): 70/70 PNGs max |gray-level change| = 4, Linf = 8/255 on [-1,1]
  step_255: 0.5                # v1 actual (alpha=1/255 on [-1,1]). Handoff table said 1; see PHASE_0.md finding F1
  iters: 40
  random_start: uniform_full   # uniform_full = U(-eps,eps) (v1) | uniform_half (handoff ref code) | zero
  eps_sweep_255: [2, 4, 6, 8]  # E9

loss:
  alpha: 0.2699475469647097    # v1 Optuna best trial (data/v1_ref/optimal_hyperparameters.json); handoff rounds to 0.27
  beta: 0.9023875567667267
  gamma: 60.52658429696642
  eta: 0.27                    # L_den weight; starts = alpha; retuned in Phase 6
  lambda_id: 0.0               # L_id weight; enabled in Phase 6
  sem_target_text: "a potted plant"
  l_vis_latent: unscaled_mean  # MSE on vae.encode(x).latent_dist.mean, no scaling_factor (v1 scale; alpha tuned on it)
  l_str_on: published          # v2: 1-SSIM(x, x+delta). v1 preset overrides to augmented_pair
  ssim: {window: 11, sigma: 1.5}
  normalize_terms: false       # Phase 6
  anti_purify: {enabled: false, w_hf: 0.5, w_mono: 0.5}   # Phase 5 (AntiPure idea)

eot:
  k_draws: 4                   # 8 if memory allows
  spatial: {enabled: false, crop_scale: [0.8, 1.0], p: 0.5}   # if enabled: same crop on clean, z_clean recomputed per draw
  v1: {jpeg_quality: [50, 100], crop_scale: [0.8, 1.0], p: 0.5, same_on_batch: true}

purifiers_train: [jpeg, downup, vae_roundtrip, sd_onestep]
purifier_train_probs: {jpeg: 0.25, downup: 0.25, vae_roundtrip: 0.2, sd_onestep: 0.3}   # Phase 3 renormalizes without sd_onestep
purifier_train_params:
  jpeg:          {impl: kornia_diff, quality: [50, 95]}
  downup:        {scale: [0.5, 0.9], modes: [bilinear, bicubic], antialias: true}
  vae_roundtrip: {model: sd15, latent: mean}
  sd_onestep:    {model: sd15, t_min: 50, t_max: 300, prompt: ""}
purifiers_heldout: [jpeg_q_other, downup_other, diffpure_t0.1, diffpure_t0.2, diffpure_t0.3, gridpure]
purifier_eval:                 # Half B only; real codecs (PIL), never the differentiable approximations
  none: {}
  jpeg_q90: {impl: pil, quality: 90}
  jpeg_q75: {impl: pil, quality: 75}
  jpeg_q50: {impl: pil, quality: 50}
  jpeg_q_other: {impl: pil, quality: 65, subsampling: "4:4:4"}
  downup_0.5:   {impl: pil, scale: 0.5, resample: bicubic}
  downup_other: {impl: pil, scale: 0.375, resample: lanczos}
  vae_rt_sd15:  {model: sd15}
  diffpure_t0.05: {impl: sd_img2img_empty_prompt, strength: 0.05, steps: 50, guidance: 1.0}
  diffpure_t0.1:  {impl: sd_img2img_empty_prompt, strength: 0.1,  steps: 50, guidance: 1.0}
  diffpure_t0.2:  {impl: sd_img2img_empty_prompt, strength: 0.2,  steps: 50, guidance: 1.0}
  diffpure_t0.3:  {impl: sd_img2img_empty_prompt, strength: 0.3,  steps: 50, guidance: 1.0}
  gridpure: {impl: official, feasible: unverified}
  restore_swin2sr: {impl: downup_then_sr, scale: 0.5, model: "caidas/swin2SR-classical-sr-x2-64"}

editors: [sd15_img2img, sdxl_img2img, ip2p, sd_inpaint, ipadapter_faceid]   # identity editor: IP-Adapter FaceID (see Section 6)
edit:
  strengths: [0.3, 0.55, 0.75]          # sd15_img2img sweeps these; others use default_strength
  default_strength: 0.55
  seeds: [1, 2, 3]
  guidance_scale: 7.5
  num_inference_steps: 50
  prompts:
    img2img: ["a person in a suit", "a person at a beach", "an oil painting of a person"]
    ip2p:    ["put the person in a suit", "put the person on a beach", "turn it into an oil painting"]
  ip2p_image_guidance: 1.5
  sdxl_resolution: 1024
  v1_phase2: {prompt: "a highly detailed portrait, cyberpunk style, neon lighting, digital art masterpiece",
              strength: 0.55, guidance: 7.5, seed: 2026}

data:
  image_size: 512
  clean_format: png             # v2: lossless. v1 preset: v1_jpeg (PIL default q75)
  n_train: 20
  n_val: 20
  n_test: 100
  n_messy: 20
  split_seed: 2026
  v1_selection: {kmeans_random_state: 42, n_init: 10, k_range: [1, 20], k_eval: 70, fallback_k_calib: 4}

metrics:
  identity: arcface
  psr_threshold: "from clean-edit distribution"   # 95th pct of ArcFace sim(source_i, clean_edit_j), i != j
  psr_impostor_percentile: 95
  psr_threshold_scope: per_editor
  no_face_counts_as_success: true                  # and report no-face rate separately
  invisibility_min: {psnr: 38.0, ssim: 0.95}

stats: {n_boot: 10000, ci: 0.95, unit: image, bootstrap_seed: 2026}

runtime:
  device: auto                  # cuda if available else cpu
  unet_dtype: fp16
  vae_dtype_protect: autocast   # fp32 | autocast. autocast = fp16 VAE fwd/bwd; S04u: ~2x faster, same loss trajectory as fp32 (4 images, 10 iters)
  grad_checkpointing: true
  loss_scale: 1024.0            # multiply loss before autograd.grad (fp16 underflow guard; sign() is scale-invariant)
  checkpoint_every: 5           # images
  cudnn_benchmark: false

methods:                        # presets deep-merged over the root by load_config(method=...)
  v1:
    budget: {eps_255: 4, step_255: 0.5, iters: 40, random_start: uniform_full}
    eot: {k_draws: 1}
    loss: {alpha: 0.2699475469647097, beta: 0.9023875567667267, gamma: 60.52658429696642, eta: 0.0, lambda_id: 0.0, l_str_on: augmented_pair}
    purifiers_train: []
    model_key: sd15_v1
    data: {clean_format: v1_jpeg}
  # v2_cheap (Phase 3), v2_onestep (Phase 4), v2_full (Phase 6) are added later, never edited after their gate.

```

#### `config/parts/baselines.yaml`

```yaml
# Baselines (E1 noise controls, E2 PhotoGuard / Mist) and E8 adaptive autoencoder purifier. Owner: agent A4.
# Loaded by load_config via the config/parts/*.yaml deep-merge. eps_255 = null -> budget.eps_255 (standard 0-255 scale).
# PhotoGuard / Mist stay labelled "reimplementation", but hyper-parameters were checked against the official code
# (details in src/baselines/{photoguard,mist}.py). eps is OUR budget (budget.eps_255), never the official one.
baselines:
  _common:
    eps_255: null
    dtype: auto                     # fp16 on cuda, fp32 on cpu
  noise_uniform:
    variant: uniform                # uniform | sign (sign = +/-eps, extra control)
  noise_gauss_psnr:
    target_psnr: 41.911             # v1 mean PSNR over the 70 eval images (V2_PLAN_48H section 0); fallback only
    psnr_csv: null                  # per-image target: CSV with the protected-image PSNRs to match
    psnr_key_col: Original_Filename # v1 stage1_per_image_metrics.csv columns (key stripped to the image stem)
    psnr_val_col: PSNR_dB
  photoguard:                       # encoder attack only; the diffusion attack variant is OUT of scope
    # source_verified: https://github.com/MadryLab/photoguard@686bea75c786cb46c88fc396a0cd0ee3d7d28c2e
    #   notebooks/demo_simple_attack_img2img.ipynb cell 7 / demo_simple_attack_inpainting.ipynb cell 7
    # official defaults (NOT adopted for eps): eps=0.06 on [-1,1] = 7.65/255 standard; step 0.02 (img2img) or 0.01
    #   (inpainting) = 2.55 / 1.275 /255; iters=1000; linear step decay to step/100; loss = latent_dist.mean.norm()
    #   (img2img) or (mean - target_latent).norm() (inpainting, target = a target image or zeros).
    iters: 1000
    step_255: 2.55                  # official img2img demo 0.02 on [-1,1]
    step_decay: true                # linear decay step -> step/100 (verified, official pgd())
    target: zero_latent             # zero_latent (img2img demo) | image (+ target_image path) | gray (ours, non-official)
    target_image: null              # official asset: MadryLab/photoguard assets/ (not vendored)
    loss_scale: 1024.0              # only used when the VAE runs in fp16
  mist:                             # REIMPLEMENTATION of the fused semantic + textural loss
    # source_verified: https://github.com/psyker-team/mist@99f5f3c455022bef77ef25ba440a4289ea472d25 (mist_v3.py, mist_utils.py)
    # official defaults (NOT adopted for eps): epsilon=16 (/255, final budget (eps+1)/255), steps=100, alpha=1,
    #   mode=2 (fused), --rate 1 -> fused weight 10**(1+3)=1e4 on the semantic loss, input 512, seed 23.
    iters: 100
    step_255: 1.0
    w_sem: 10000.0                  # official rate
    w_tex: 1.0
    t_min: 0
    t_max: 1000
    n_noise: 1
    prompt: "a painting"            # official style default ("a photo" with --object)
    # Official target MIST.png (repo root, GPL-3.0 repo) is NOT vendored. To use it:
    #   curl -L -o assets/MIST.png https://raw.githubusercontent.com/psyker-team/mist/99f5f3c455022bef77ef25ba440a4289ea472d25/MIST.png
    # then set target_image: assets/MIST.png. null -> procedural stand-in (must be reported as a deviation).
    target_image: null
    target_image_url: https://raw.githubusercontent.com/psyker-team/mist/99f5f3c455022bef77ef25ba440a4289ea472d25/MIST.png
    loss_scale: 1024.0              # GPU TODO: w_sem=1e4 with fp16 VAE backward may overflow; lower to 1 if grads are inf/NaN
    grad_checkpointing: true

# E8 adaptive attacker (LightShed-style autoencoder). Weights are produced by scripts/train_adaptive_ae.py (job J42).
# src/eval/adaptive_ae.py::build_purifier(cfg, device) reads this block (eval.purifiers.adaptive_ae, if set, overrides it).
adaptive_ae:
  weights_path: results/adaptive_ae/ae_final.pt
  base_channels: 32
  depth: 3                          # number of 2x downsamplings (image side padded to a multiple of 2**depth)
  patch: 128
  device: auto
  train:
    minutes: 60                     # wall-clock budget (~1 GPU-hour)
    batch_size: 16
    lr: 0.0002
    lpips_weight: 0.0               # 0 = L1 only; >0 needs the `lpips` package
    val_fraction: 0.1
    log_every: 50
    ckpt_every: 500

```

#### `config/parts/data.yaml`

```yaml
# Dataset / split settings. Owner: agent A4. Deep-merged over config/config.yaml `data:` (keys there: image_size, n_train, ...).
data:
  n_test: 100
  n_messy: 20
  n_train: 20
  n_val: 20
  n_ae_pool: 120                    # adaptive-attacker (E8) training pool; disjoint from every other split
  split_seed: 2026
  splits_files:                     # written by scripts/make_splits.py (frozen; never edited by hand)
    test: data/splits/test.txt
    messy: data/splits/messy.txt
    train: data/splits/train.txt
    val: data/splits/val.txt
    ae_pool: data/splits/ae_pool.txt
    meta: data/splits/splits_meta.json
  v1_calib_file: data/splits/v1_calib.txt        # 5319, 1715, 6385, 8932 - excluded from every v2 split
  v1_eval70_file: data/splits/v1_eval70.txt      # ALL 70 kept in test (paired v1 comparison) + 30 new
  identity:
    mode: auto                      # auto | labels | arcface | none. CelebAMask-HQ Kaggle copy has NO identity_CelebA.txt -> arcface
    arcface_same_person_cosine: 0.5 # UNVERIFIED threshold
  messy:
    quota: {blur: 5, lighting: 5, occlusion: 4, pose: 6}
    pose_top_fraction: 0.10         # top 10% by max(|yaw|,|pitch|) of CelebAMask-HQ-pose-anno.txt (rank based, units unverified)
    degrade: {scale: [0.60, 0.85], jpeg_quality: [50, 70], resample: bicubic}   # applied by scripts/prepare_data.py
  prepared:
    format: png                     # lossless 512 px (v1 inputs were JPEG q75)
    resample: bicubic

```

#### `config/parts/eval.yaml`

```yaml
# Half B (attack-and-score harness) config. Owned by agent A3. Loaded by load_config via
# config/parts/*.yaml deep-merge. Everything lives under the `eval:` key so it cannot collide with
# config/config.yaml. Model ids marked "unverified" were NOT loaded in this workspace (no GPU / no
# downloads); the J03b smoke job is the first real load. Pin `revision` after the smoke job.
eval:
  seeds: [1, 2, 3]                  # edit seeds; protection uses its own per-image seed
  num_inference_steps: 30           # DPM-Solver++ (multistep, 2nd order) for ALL editors (48h plan section 2)
  guidance_scale: 7.5
  ip2p_image_guidance: 1.5
  dtype: float16
  image_size: 512
  checkpoint_every: 50              # edits (rows) between progress.json / gpu_time.json writes
  cache_purified: true              # purified PNGs reused across editors of one run

  headline:                         # all 3 prompts here, 1 rotated prompt (image_index mod 3) elsewhere
    editor: sd15_img2img
    strength: 0.55

  save_edits:
    all: true                       # coordinator request: save EVERY edit (JPEG q90, small). false => policy below
    sample_first_n: 10              # policy mode: first N images of the split (all methods)
    headline_methods: [clean, v1, v2_final]
    headline_purifiers: [none]      # headline methods are saved for these purifiers only

  psr:
    percentile: 95                  # 95th pct of ArcFace sim between DIFFERENT people (clean edits vs other sources)
    scope: per_editor               # also write a pooled threshold

  masks:                            # keep-face inpainting masks (src/eval/masks.py); 255 = repaint
    keep_parts: [skin, nose, l_eye, r_eye, l_brow, r_brow, l_ear, r_ear, mouth, u_lip, l_lip, eye_g, ear_r, neck, neck_l]
    dilate_px: 8                    # repainted: background, cloth, hat, hair. Messy / non-CelebA ids: face-bbox ellipse fallback

  edit_images:                      # every edit is saved as JPEG so edit-LPIPS can be recomputed later (stage lpips)
    format: jpg
    quality: 90                     # edit_lpips is defined on JPEG-q90 round-tripped images (clean AND protected edit)

  prompts:
    img2img: ["a person in a suit", "a person at a beach", "an oil painting of a person"]
    ip2p:    ["put the person in a suit", "put the person on a beach", "turn it into an oil painting"]

  models:                           # ids: see 'verification' notes
    sd15:           {id: "stable-diffusion-v1-5/stable-diffusion-v1-5", revision: null}      # id seen in current diffusers docs; weights load unverified here
    sd15_inpaint:   {id: "stable-diffusion-v1-5/stable-diffusion-inpainting", revision: null} # unverified (mirror of runwayml repo)
    sdxl:           {id: "stabilityai/stable-diffusion-xl-base-1.0", revision: null}          # unverified here
    sdxl_vae_fp16:  {id: "madebyollin/sdxl-vae-fp16-fix", revision: null}                     # unverified here
    ip2p:           {id: "timbrooks/instruct-pix2pix", revision: null}                        # unverified here
    ipadapter_faceid: {id: "h94/IP-Adapter-FaceID", weight_name: "ip-adapter-faceid_sd15.bin", revision: null}  # repo + file name seen in diffusers docs (verified by reading docs, not by loading)
    swin2sr:        {id: "caidas/swin2SR-classical-sr-x2-64", revision: null}                 # unverified
    clip_eval:      {id: "openai/clip-vit-large-patch14", revision: null}                     # CLIP-I metric
    lpips_net: vgg
    arcface: {pack: buffalo_l, det_size: [640, 640], pad_frac: 0.4}                                          # insightface; pack downloads on first use (unverified on Kaggle)

  editors:
    sd15_img2img:     {kind: sd15_img2img, model: sd15, prompts: img2img, default_strength: 0.55, max_batch: 3}
    sdxl_img2img:     {kind: sdxl_img2img, model: sdxl, prompts: img2img, default_strength: 0.55, max_batch: 1,
                       resolution: 1024, cpu_offload: false}
    ip2p:             {kind: ip2p, model: ip2p, prompts: ip2p, default_strength: null, max_batch: 3}   # no strength param
    sd_inpaint:       {kind: sd_inpaint, model: sd15_inpaint, prompts: img2img, default_strength: 1.0, max_batch: 3,
                       needs_mask: true}
    ipadapter_faceid: {kind: ipadapter_faceid, model: ipadapter_faceid, base_model: sd15, prompts: img2img,
                       default_strength: 0.55, max_batch: 1, ip_adapter_scale: 0.6,
                       scheduler: dpmpp   # diffusers docs recommend DDIM or Euler for FaceID; switch to `ddim` if J03b shows artefacts
                       }

  # Purifier params (code defaults used if a key is missing). Names are canonical; aliases map the older config.yaml names.
  purifier_aliases:
    jpeg_q_other: jpeg_other
    vae_rt_sd15: vae_rt
    diffpure_t0.05: diffpure_0.05
    diffpure_t0.1: diffpure_0.1
    diffpure_t0.2: diffpure_0.2
    diffpure_t0.3: diffpure_0.3
  purifiers:
    jpeg_q90:   {quality: 90}
    jpeg_q75:   {quality: 75}
    jpeg_q50:   {quality: 50}
    jpeg_other: {quality: 65, subsampling: 0}          # 4:4:4
    downup_0.5:   {scale: 0.5, resample: bicubic}
    downup_other: {scale: 0.375, resample: lanczos}
    vae_rt:     {model: sd15, latent: mean}
    diffpure_0.05: {model: sd15, strength: 0.05, steps: 50, guidance: 1.0}
    diffpure_0.1:  {model: sd15, strength: 0.1,  steps: 50, guidance: 1.0}
    diffpure_0.2:  {model: sd15, strength: 0.2,  steps: 50, guidance: 1.0}
    diffpure_0.3:  {model: sd15, strength: 0.3,  steps: 50, guidance: 1.0}
    restore_swin2sr: {down_scale: 0.5, model: swin2sr}
    gridpure:                                           # from the paper/official repo description (see src/eval/gridpure.py)
      window: 256
      stride: 128
      pure_steps: 10                                    # official example --pure_steps
      iters: 20                                         # official example --pure_iter_num
      gamma: 0.1
      repo_path: null                                   # clone of github.com/ZhengyueZhao/GrIDPure (guided_diffusion package)
      checkpoint: null                                  # path to 256x256_diffusion_uncond.pt (guided-diffusion); null => NotAvailable
      pure_model_url: "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
    adaptive_ae: {}                                     # lazy: src/eval/adaptive_ae.py must expose build_purifier(cfg, device)

  # ---- Tiers (48h plan section 3). A rule = (methods x purifiers x cells). `methods` accept fnmatch patterns.
  # cell: {editor, strengths (null => editor default), prompts: all_if_headline | rotate}
  _groups:
    purifiers_13: &p13 [jpeg_q90, jpeg_q75, jpeg_q50, jpeg_other, downup_0.5, downup_other, vae_rt,
                        diffpure_0.05, diffpure_0.1, diffpure_0.2, diffpure_0.3, restore_swin2sr, gridpure]
    purifiers_5:  &p5  [jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1]
    purifiers_faceid: &pf [jpeg_q75, vae_rt, diffpure_0.1, diffpure_0.2]
    purifiers_ip2p: &pi [jpeg_q75, diffpure_0.1]
    purifiers_abl: &pa [none, jpeg_q50, vae_rt, diffpure_0.1, diffpure_0.2]
    methods_h: &mh [clean, noise_uniform, noise_gauss_psnr, photoguard, mist, v1, v2_cheap, v2_main, v2_final]
    cell_sd15_headline: &c_sd15 {editor: sd15_img2img, strengths: [0.3, 0.55, 0.75], prompts: all_if_headline}
    cell_sd15_055_rot: &c_sd15r {editor: sd15_img2img, strengths: [0.55], prompts: rotate}
    cell_faceid: &c_fid {editor: ipadapter_faceid, strengths: null, prompts: rotate}
    cell_ip2p: &c_ip2p {editor: ip2p, strengths: null, prompts: rotate}
    cell_inpaint: &c_inp {editor: sd_inpaint, strengths: null, prompts: rotate}
    cell_sdxl: &c_sdxl {editor: sdxl_img2img, strengths: [0.55], prompts: rotate}

  tiers:
    H:                                                  # no purifier, test split
      - {splits: [test], methods: *mh, purifiers: [none], cells: [*c_sd15, *c_ip2p, *c_inp, *c_fid]}
      - {splits: [test], methods: [clean, v1, photoguard, v2_final], purifiers: [none], cells: [*c_sdxl]}
    M:                                                  # messy 20: SD1.5 rotated + FaceID
      - {splits: [messy], methods: [clean, noise_uniform, noise_gauss_psnr, photoguard, mist, v1, v2_main, v2_final],
         purifiers: [none], cells: [*c_sd15r, *c_fid]}
    P:                                                  # purifier grid (test split)
      - {splits: [test], methods: [clean, v1, v2_cheap, v2_main, v2_antipure, v2_final], purifiers: *p13, cells: [*c_sd15r]}
      - {splits: [test], methods: [photoguard, mist], purifiers: *p5, cells: [*c_sd15r]}
      - {splits: [test], methods: [clean, v1, v2_main, v2_final], purifiers: *pf, cells: [*c_fid]}
      - {splits: [test], methods: [v1, v2_final], purifiers: *pi, cells: [*c_ip2p]}
      - {splits: [test], methods: [v2_antipure], purifiers: [none], cells: [*c_fid]}      # E6 "FaceID none"
    SCREEN:                                             # S24 / S25: method screening on the VAL split (20 photos); never test
      - {splits: [val], methods: [clean, v1, v2_cheap, v2_main, photoguard, mist, v2_tgt_zero, v2_tgt_zero_hi, v2_tgt_gray],
         purifiers: [none, jpeg_q75, diffpure_0.1], cells: [*c_sd15r]}
    TGT:                                                # S26-S28: target-steered family on the TEST split (clean / v1 / v2_main / v2_cheap already have these cells)
      - {splits: [test], methods: [v2_tgt_zero, v2_tgt_zero_mid, v2_tgt_zero_hi], purifiers: [none], cells: [*c_sd15, *c_fid]}
      - {splits: [test], methods: [v2_tgt_zero, v2_tgt_zero_mid, v2_tgt_zero_hi],
         purifiers: [jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1, diffpure_0.2, diffpure_0.3], cells: [*c_sd15r]}
    AP:                                                 # S07 / S20: single-term antipure ablation on the VAL split (selection of the antipure terms; never test)
      - {splits: [val], methods: [clean, v2_antipure], purifiers: [jpeg_q75, vae_rt, diffpure_0.1], cells: [*c_sd15r]}
    A:                                                  # phase ablation (v2_main is the "both" arm)
      - {splits: [test], methods: [abl_vis_only, abl_den_only, abl_no_sem, abl_no_pur], purifiers: *pa, cells: [*c_sd15r]}
      - {splits: [test], methods: [abl_vis_only, abl_den_only, abl_no_sem, abl_no_pur], purifiers: [none],
         cells: [*c_fid, *c_ip2p, *c_inp]}
    E9:                                                 # epsilon sweep of v2_final (method names eps2/eps6/eps8; unverified naming)
      - {splits: [test], methods: ["eps*", "*_eps*"], purifiers: [none, jpeg_q75, diffpure_0.1], cells: [*c_sd15r]}
      - {splits: [test], methods: ["eps*", "*_eps*"], purifiers: [none], cells: [*c_fid]}
    E8:                                                 # adaptive autoencoder attacker (A4 module)
      - {splits: [test], methods: [clean, v1, v2_final], purifiers: [adaptive_ae], cells: [*c_sd15r, *c_fid]}
    SMOKE:                                              # J03b: every editor and a few purifiers, 5 images
      - {splits: [test, val, messy], methods: ["*"], purifiers: [none, jpeg_q75, downup_0.5, vae_rt, diffpure_0.1],
         cells: [*c_sd15r, *c_ip2p, *c_inp, *c_fid, *c_sdxl]}

```

#### `config/parts/final.yaml`

```yaml
# candidate for config/parts/final.yaml (trial_5; confirmed on 20 val images; hand-set psr=0.041666666666666664)
methods:
  v2_final:
    loss:
      alpha: 0.7281249466629797
      eta: 0.05776266977639212
      beta: 0.6676014641453782
      gamma: 62.04711204188804
      lambda_id: 0.0
      anti_purify:
        enabled: true
        w_hf: 0.9310312738392569
        w_mono: 0.9310312738392569

```

#### `config/parts/protect.yaml`

```yaml
# Half A (v2 protection loop) config part. Owned by agent A2. Deep-merged over config/config.yaml by load_config
# (parts are merged in sorted file order, so DO NOT define `methods.v2_final` here: the tuned
# `config/parts/final.yaml` sorts BEFORE this file and would be overwritten. The hand-set
# starting point is `methods.v2_final_default`; src/protect/v2_registry.py layers
# v2_final_default < methods.v2_final (tuned, from final.yaml, optional) < epsN).
#
# All budgets on the STANDARD 0-255 scale. Numbers below are the handoff's / v1's starting values,
# not tuned results. Root keys used by the code (defined in config/config.yaml; defaults in code
# otherwise): models.*, budget.*, loss.*, eot.k_draws, runtime.*, purifier_train_params.

protect:
  quantize: true                 # snap input to uint8 grid, return uint8-grid output (|diff| <= floor(eps_255) in saved PNG)
  ensemble:
    enabled: false               # average L_vis over SD1.5 VAE + the VAEs listed (Phase 6)
    vaes: [sdxl_vae_fp16]        # keys into models.*; run in fp32 here
    weights: {}                  # optional per-VAE weights, e.g. {sd15: 1.0, sdxl_vae_fp16: 1.0}

face:                            # L_id (loss.lambda_id > 0 only). Different network from the eval ArcFace on purpose (R16).
  backend: facenet               # facenet (facenet-pytorch InceptionResnetV1, vggface2) | onnx2torch
  pretrained: vggface2
  bbox: null                     # (x0, y0, x1, y1) of the face on the clean image; null = central 60% crop

# training purifier details; merged with the root dict of the same name
purifier_train_params:
  jpeg:            {impl: kornia_diff, quality: [50, 95]}      # impl: pil_bpda = real PIL codec + straight-through backward
  downup:          {scale: [0.5, 0.9], modes: [bilinear, bicubic], antialias: true}
  downup_heldout:  {scale: [0.30, 0.45], modes: [bicubic], antialias: true}   # disjoint from `downup`; never drawn in training
  vae_roundtrip:   {model: sd15, latent: mean}
  sd_onestep:      {model: sd15, t_min: 50, t_max: 300, prompt: ""}
purifier_train_exclude: [downup_heldout]

methods:
  # Phase 3: cheap purifiers in the loop, no UNet term (cost estimate 54 s/img on T4: planner estimate, not measured)
  v2_cheap:
    budget: {step_255: 1.0}   # handoff value; v1 preset keeps its true 0.5
    purifiers_train: [jpeg, downup, vae_roundtrip]
    purifier_train_probs: {jpeg: 0.35, downup: 0.35, vae_roundtrip: 0.3}
    loss: {alpha: 0.2699475469647097, beta: 0.9023875567667267, gamma: 60.52658429696642, eta: 0.0, lambda_id: 0.0}
    eot: {k_draws: 4}
  # Phase 4: + one-step SD purifier (share 0.3) and L_den (eta starts = alpha)
  v2_main:
    budget: {step_255: 1.0}   # handoff value; v1 preset keeps its true 0.5
    purifiers_train: &main_purifiers [jpeg, downup, vae_roundtrip, sd_onestep]
    purifier_train_probs: &main_probs {jpeg: 0.25, downup: 0.25, vae_roundtrip: 0.2, sd_onestep: 0.3}
    loss: {alpha: 0.2699475469647097, beta: 0.9023875567667267, gamma: 60.52658429696642, eta: 0.2699475469647097, lambda_id: 0.0}
    eot: {k_draws: 4}
  # Phase 5: + anti-purification terms (patch-DCT high-frequency + error-grows-with-t), weights 0.5 / 0.5
  v2_antipure:
    budget: {step_255: 1.0}   # handoff value; v1 preset keeps its true 0.5
    purifiers_train: *main_purifiers
    purifier_train_probs: *main_probs
    loss:
      alpha: 0.2699475469647097
      beta: 0.9023875567667267
      gamma: 60.52658429696642
      eta: 0.2699475469647097
      lambda_id: 0.0
      anti_purify: {enabled: true, w_hf: 0.5, w_mono: 0.5}
    eot: {k_draws: 4}
  # E7 ablations (phase ablation). INTERPRETATION: vis_only / den_only drop L_sem too (beta 0) so that
  # phases are isolated; abl_no_sem = both phases without L_sem; v2_main is "both". Compare vis_only /
  # den_only / abl_no_sem to each other, and abl_no_sem vs v2_main for L_sem.
  abl_vis_only:
    budget: {step_255: 1.0}   # handoff value; v1 preset keeps its true 0.5
    purifiers_train: *main_purifiers
    purifier_train_probs: *main_probs
    loss: {alpha: 0.2699475469647097, beta: 0.0, gamma: 60.52658429696642, eta: 0.0, lambda_id: 0.0}
    eot: {k_draws: 4}
  abl_den_only:
    budget: {step_255: 1.0}   # handoff value; v1 preset keeps its true 0.5
    purifiers_train: *main_purifiers
    purifier_train_probs: *main_probs
    loss: {alpha: 0.0, beta: 0.0, gamma: 60.52658429696642, eta: 0.2699475469647097, lambda_id: 0.0}
    eot: {k_draws: 4}
  abl_no_sem:
    budget: {step_255: 1.0}   # handoff value; v1 preset keeps its true 0.5
    purifiers_train: *main_purifiers
    purifier_train_probs: *main_probs
    loss: {alpha: 0.2699475469647097, beta: 0.0, gamma: 60.52658429696642, eta: 0.2699475469647097, lambda_id: 0.0}
    eot: {k_draws: 4}
  abl_no_pur:
    budget: {step_255: 1.0}   # handoff value; v1 preset keeps its true 0.5                    # v2_main losses, no purifier in the loop (identity); L_den uses its own UNet pass
    purifiers_train: []
    loss: {alpha: 0.2699475469647097, beta: 0.9023875567667267, gamma: 60.52658429696642, eta: 0.2699475469647097, lambda_id: 0.0}
    eot: {k_draws: 4}
  # Target-steered variants (2026-10-20): v2_cheap's cleanups in the loop (jpeg, down-up, VAE round trip; no UNet term, cheap)
  # but L_vis is replaced by "pull the latent toward a destructive target" (PhotoGuard / Mist idea) instead of "move away
  # from the clean latent". Screened on the VALIDATION photos only (S24 / S25); never tuned on test.
  v2_tgt_zero:
    budget: {step_255: 1.0}
    purifiers_train: [jpeg, downup, vae_roundtrip]
    purifier_train_probs: {jpeg: 0.35, downup: 0.35, vae_roundtrip: 0.3}
    loss: {alpha: 0.2699475469647097, beta: 0.9023875567667267, gamma: 60.52658429696642, eta: 0.0, lambda_id: 0.0,
           target: {kind: zero, w_tgt: 1.0, w_dev: 0.0}}
    eot: {k_draws: 4}
  v2_tgt_zero_hi:
    budget: {step_255: 1.0}
    purifiers_train: [jpeg, downup, vae_roundtrip]
    purifier_train_probs: {jpeg: 0.35, downup: 0.35, vae_roundtrip: 0.3}
    loss: {alpha: 1.0, beta: 0.9023875567667267, gamma: 60.52658429696642, eta: 0.0, lambda_id: 0.0,
           target: {kind: zero, w_tgt: 1.0, w_dev: 0.0}}
    eot: {k_draws: 4}
  v2_tgt_zero_mid:                       # declared interpolation between v2_tgt_zero (alpha 0.27) and v2_tgt_zero_hi (alpha 1.0); not screened on val
    budget: {step_255: 1.0}
    purifiers_train: [jpeg, downup, vae_roundtrip]
    purifier_train_probs: {jpeg: 0.35, downup: 0.35, vae_roundtrip: 0.3}
    loss: {alpha: 0.55, beta: 0.9023875567667267, gamma: 60.52658429696642, eta: 0.0, lambda_id: 0.0,
           target: {kind: zero, w_tgt: 1.0, w_dev: 0.0}}
    eot: {k_draws: 4}
  v2_tgt_gray:
    budget: {step_255: 1.0}
    purifiers_train: [jpeg, downup, vae_roundtrip]
    purifier_train_probs: {jpeg: 0.35, downup: 0.35, vae_roundtrip: 0.3}
    loss: {alpha: 0.2699475469647097, beta: 0.9023875567667267, gamma: 60.52658429696642, eta: 0.0, lambda_id: 0.0,
           target: {kind: gray, w_tgt: 1.0, w_dev: 0.0}}
    eot: {k_draws: 4}
  # Hand-set starting point for v2_final (before Optuna). Tuned values arrive as methods.v2_final in
  # config/parts/final.yaml and are layered on top of this by the registry.
  v2_final_default:
    budget: {step_255: 1.0}   # handoff value; v1 preset keeps its true 0.5
    purifiers_train: *main_purifiers
    purifier_train_probs: *main_probs
    loss:
      alpha: 0.2699475469647097
      beta: 0.9023875567667267
      gamma: 60.52658429696642
      eta: 0.2699475469647097
      lambda_id: 0.0
      anti_purify: {enabled: false, w_hf: 0.5, w_mono: 0.5}
    eot: {k_draws: 4}
    protect:
      ensemble: {enabled: true, vaes: [sdxl_vae_fp16]}
  # E9 eps sweep: variants of v2_final (only the budget differs)
  eps2: {budget: {eps_255: 2}}
  eps6: {budget: {eps_255: 6}}
  eps8: {budget: {eps_255: 8}}

```

### 11.2 Global defaults that apply to every method unless a preset overrides them (from `config/config.yaml`)
| Parameter | Value | Where |
|---|---|---|
| Base seed | 2026 (per-item seeds by sha256 `derive_seed`) | `seed` |
| Budget eps | 4 (/255, standard scale; confirmed by J01: 70/70 PNGs max change 4 gray levels, Linf 8/255 on [-1,1]) | `budget.eps_255` |
| Step | 0.5 (/255) at root; v2 presets set 1.0 | `budget.step_255`, presets |
| PGD iterations | 40 | `budget.iters` |
| Random start | `uniform_full` = U(-eps, eps) (v1 behaviour) | `budget.random_start` |
| EoT draws per iteration (K) | 4 (v1 preset: 1) | `eot.k_draws` |
| Image size | 512 | `data.image_size` |
| Clean format | PNG lossless (v1: `v1_jpeg`, PIL default q75) | `data.clean_format` |
| Loss weights at root | alpha 0.2699475469647097, beta 0.9023875567667267, gamma 60.52658429696642 (v1 Optuna best, from `data/v1_ref/optimal_hyperparameters.json`), eta 0.27, lambda_id 0.0 | `loss.*` |
| L_sem target text | "a potted plant" | `loss.sem_target_text` |
| L_vis latent | `unscaled_mean` (VAE posterior mean, no scaling factor) | `loss.l_vis_latent` |
| L_str | `published` = 1 - SSIM(x, x + delta) (v1 preset: `augmented_pair`) | `loss.l_str_on` |
| Anti-purify at root | disabled, w_hf 0.5, w_mono 0.5 | `loss.anti_purify` |
| Quantisation of output | true | `protect.quantize` (`config/parts/protect.yaml`) |
| Invisibility gate | PSNR >= 38, SSIM >= 0.95 | `metrics.invisibility_min` |
| Runtime | `unet_dtype fp16`, `vae_dtype_protect autocast`, `grad_checkpointing true`, `loss_scale 1024`, `checkpoint_every 5`, `cudnn_benchmark false` | `runtime.*` |

### 11.3 Evaluation configuration summary (from `config/parts/eval.yaml`; full text in 11.1)
| Item | Value |
|---|---|
| Edit seeds | [1, 2, 3] (`eval.seeds`) |
| Sampler / steps / guidance | DPM-Solver++ multistep, 30 steps (`eval.num_inference_steps`; the root `edit.num_inference_steps: 50` block is an older default, the `eval` value is what the editors read), guidance 7.5, IP2P image guidance 1.5, fp16 |
| Headline cell | SD1.5 img2img, strength 0.55; all 3 prompts on headline cells, one rotated prompt (image_index mod 3) elsewhere |
| Strength sweep (headline tier) | [0.3, 0.55, 0.75] (SD1.5 img2img) |
| Prompts img2img | "a person in a suit", "a person at a beach", "an oil painting of a person" |
| Prompts IP2P | "put the person in a suit", "put the person on a beach", "turn it into an oil painting" |
| Editors | `sd15_img2img` (`stable-diffusion-v1-5/stable-diffusion-v1-5`, max_batch 3), `sdxl_img2img` (`stabilityai/stable-diffusion-xl-base-1.0`, resolution 1024, max_batch 1, VAE `madebyollin/sdxl-vae-fp16-fix`), `ip2p` (`timbrooks/instruct-pix2pix`, no strength), `sd_inpaint` (`stable-diffusion-v1-5/stable-diffusion-inpainting`, strength 1.0, keep-face mask), `ipadapter_faceid` (`h94/IP-Adapter-FaceID`, `ip-adapter-faceid_sd15.bin`, base sd15, ip_adapter_scale 0.6, img2img strength 0.55, dpmpp scheduler, max_batch 1) |
| Protection success rate (PSR) | An edit counts as protected if ArcFace similarity to the source is below the 95th percentile of ArcFace similarity between different people (clean edits vs other sources), per editor; no face detected counts as success, and the no-face rate is reported separately |
| Saved edits | all edits saved as JPEG q90 (`eval.save_edits`, `eval.edit_images`) |
| Tiers | H (no purifier, test), M (messy 20), P (purifier grid: 13 purifiers for SD1.5; 5 for photoguard/mist; 4 for FaceID; 2 for IP2P), SCREEN (val), TGT, AP, A (ablation), E9 (eps sweep), E8 (adaptive AE), SMOKE |
| Splits | test 100, messy 20, train 20, val 20, ae_pool 120 (`config/parts/data.yaml`); v1 calibration images (5319, 1715, 6385, 8932) excluded; all 70 v1 evaluation images kept in test for the paired v1 comparison + 30 new |

### 11.4 Method preset table (effective values and splits actually run)
"Splits run" is taken from `kaggle/jobs.yaml` plus the presence of output folders under `results/`. Numbers are in other parts of the master document. All v2 presets use K = 4 EoT draws, 40 iterations, random start uniform_full, quantised output, step 1.0 (/255) unless stated. Config source: `config/parts/protect.yaml` (`methods:`) and `config/parts/final.yaml`.

| Preset | Effective values (eps 4/255 unless stated) | Training purifier mixture | L_vis form | Splits run (sessions) | Status |
|---|---|---|---|---|---|
| `v1` | step 0.5, K 1, alpha 0.26995, beta 0.90239, gamma 60.5266, eta 0, lambda_id 0, L_str = `augmented_pair`, model `sd15_v1` (`runwayml/stable-diffusion-v1-5`), clean format `v1_jpeg`, own v1 EoT (JPEG q 50-100, crop 0.8-1.0, p 0.5) (`config/config.yaml` `methods.v1`) | none (`purifiers_train: []`) | SD1.5 VAE (v1 code) | v1_eval70 (J01, J01b; 70 images; Phase 0); test and messy (S08); val (S24); test again for E8 (S15) | IMPLEMENTED AND RUN |
| `v2_cheap` | step 1.0, alpha 0.26995, beta 0.90239, gamma 60.5266, eta 0, lambda_id 0 | jpeg 0.35, downup 0.35, vae_roundtrip 0.30 (no UNet term) | SingleVis (push latent away from clean) | test (S10); val (S25, control of target-steered variants) | IMPLEMENTED AND RUN |
| `v2_main` | as v2_cheap but eta = 0.26995 (= alpha) | jpeg 0.25, downup 0.25, vae_roundtrip 0.20, sd_onestep 0.30 | SingleVis | test and messy (S09); val (S24) | IMPLEMENTED AND RUN |
| `v2_antipure` | as v2_main + anti_purify enabled, w_hf 0.5, w_mono 0.5 | as v2_main | SingleVis | val with single-term variants (S07: `--set` w_mono=0 gives hf only, w_hf=0 gives mono only, plus both; an S20 rerun is defined in jobs.yaml but there is no S20 folder in repo `results/`); test (S12) | IMPLEMENTED AND RUN |
| `v2_final` | step 1.0; layers: `v2_final_default` (main mix, alpha 0.26995, beta 0.90239, gamma 60.5266, eta 0.26995, anti_purify disabled 0.5/0.5, `protect.ensemble` enabled with `sdxl_vae_fp16`) overridden by `final.yaml` (Optuna trial 5): alpha 0.7281249466629797, eta 0.05776266977639212, beta 0.6676014641453782, gamma 62.04711204188804, lambda_id 0.0, anti_purify enabled with w_hf = w_mono = 0.9310312738392569 | as v2_main | EnsembleVis (SD1.5 VAE + SDXL VAE fp16-fix, equal weights) | tuned on val (S06: 16 trials x 10 images, 20 iters, K 2; confirmed on 20 val images); test (S06, S16); messy (S06 timed out, redone in S16); ae_pool with K=2 (S15) | IMPLEMENTED AND RUN |
| `eps2` | v2_final values with `budget.eps_255 = 2` (step stays 1.0) | as v2_final | EnsembleVis | test (S17) | IMPLEMENTED AND RUN |
| `eps6` | v2_final values with eps 6 | as v2_final | EnsembleVis | test (S23) defined in jobs.yaml; no results folder | IMPLEMENTED BUT NOT RUN |
| `eps8` | v2_final values with eps 8 | as v2_final | EnsembleVis | test (S18) | IMPLEMENTED AND RUN |
| `abl_vis_only` | alpha 0.26995, beta 0, eta 0, gamma 60.5266 | main mix | SingleVis | test (S11) | IMPLEMENTED AND RUN |
| `abl_den_only` | alpha 0, beta 0, eta 0.26995, gamma 60.5266 | main mix | none (alpha 0) | test (S11) | IMPLEMENTED AND RUN |
| `abl_no_sem` | alpha 0.26995, beta 0, eta 0.26995, gamma 60.5266 | main mix | SingleVis | test defined (S14); no results folder (smoke only in S04) | IMPLEMENTED BUT NOT RUN (at scale) |
| `abl_no_pur` | v2_main losses (alpha, beta, eta 0.26995, gamma) | empty list = identity (L_den uses its own UNet pass) | SingleVis | test defined (S14); smoke only | IMPLEMENTED BUT NOT RUN (at scale) |
| `v2_tgt_zero` | step 1.0, alpha 0.26995, beta 0.90239, gamma 60.5266, eta 0, `target: {kind: zero, w_tgt 1.0, w_dev 0.0}`; no ensemble | jpeg 0.35, downup 0.35, vae_roundtrip 0.30 | TargetVis (pull latent toward the zero latent, PhotoGuard zero-latent idea) | val (S25); test (S27) | IMPLEMENTED AND RUN |
| `v2_tgt_zero_mid` | as v2_tgt_zero with alpha 0.55 (declared interpolation between 0.27 and 1.0; not screened on val) | same | TargetVis zero | test (S28) | IMPLEMENTED AND RUN |
| `v2_tgt_zero_hi` | as v2_tgt_zero with alpha 1.0 | same | TargetVis zero | val (S25); test (S26) | IMPLEMENTED AND RUN |
| `v2_tgt_gray` | as v2_tgt_zero with `target.kind: gray` (latent of a mid-gray image) | same | TargetVis gray | val (S25); not run on test | RUN on val only |
| `noise_uniform` | uniform noise within +-4/255 on the uint8 grid (`variant: uniform`; a `sign` variant exists, unused) | n/a | n/a | test and messy (S22) | IMPLEMENTED AND RUN |
| `noise_gauss_psnr` | Gaussian noise, sigma by bisection to match the per-image PSNR of v1's protected images (`data/v1_ref/stage1_per_image_metrics.csv`; fallback target 41.911 dB) | n/a | n/a | test and messy (S22) | IMPLEMENTED AND RUN |
| `photoguard` | encoder attack, eps 4/255 (our budget), step 2.55/255 with linear decay to step/100, 1000 iterations, zero-latent target; re-implementation | n/a | n/a | test and messy (S13); val (S24) | IMPLEMENTED AND RUN |
| `mist` | fused semantic+textural loss, eps 4/255, step 1.0/255, 100 iterations, w_sem 10000, w_tex 1, t in [0,1000), prompt "a painting", official MIST.png target fetched at run time; re-implementation | n/a | n/a | test and messy (S21); val (S24) | IMPLEMENTED AND RUN |
| `clean` | no protection (reference for clean edits and E0 thresholds) | n/a | n/a | test, messy, val (S07, S10, S24, S25) | IMPLEMENTED AND RUN |

Notes on the table:
- `v2_final` is layered as described in 10.6; its Optuna-tuned values were chosen on the 20 validation photos, never on test. The Optuna constraints (PSNR >= 38, SSIM >= 0.95) were checked on the mean over the search photos, so `v2_final` violates SSIM >= 0.95 on 29 of 100 test photos (known limitation, from the project history).
- `abl_vis_only` and `abl_den_only` also drop L_sem (beta 0) so the two phases are isolated; compare them to each other, and `abl_no_sem` against `v2_main` for L_sem (comment in `config/parts/protect.yaml`).
- The choice of `v2_tgt_zero_mid` as the best target variant was made after seeing test results (post hoc among three variants declared in advance); `v2_tgt_gray` was only screened on val.
- Smoke runs (S03, S04, S04t, S04u, S05, S05b, S05c; 4 or 5 photos, few iterations) ran many presets for timing and wiring only; those numbers are never reported.
- The v2 tgt presets, noise controls, photoguard and mist read different loss blocks; only `v2_*`/`abl_*`/`eps*` go through `src/protect/v2_registry.py`.


## 12. Tools, libraries, models and environment

### 12.1 Hardware and platform (recorded by job J00, `results/J00/env_probe/env.json`)
| Item | Value |
|---|---|
| Platform | Kaggle notebook (batch), Linux-6.18.48+-x86_64-with-glibc2.39 |
| Docker image | `gcr.io/kaggle-gpu-images/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f` |
| Python | 3.13.15 |
| GPU | 2 x Tesla T4, 15360 MiB each (driver 580.178.04), compute capability sm_75 |
| CUDA / cuDNN | CUDA 12.8 (torch built `cu128`), cuDNN 9.19 (`91900`) |
| Session limits (project history, not from a file) | free-tier Kaggle GPU quota is limited; sessions capped near 12 h; the team used about 5 free accounts; comments in `kaggle/jobs.yaml` refer to an "11 h wall guard" when sizing sessions; S06 additionally hit its 12 h timeout (history) |
| Local dev machine | Windows 11, Python 3.13, CPU-only torch, no GPU; used only for tests, analysis, report building (`docs/SESSION_HANDOFF.md`, `requirements-dev.txt`) |
| Datasets | CelebAMask-HQ (Kaggle dataset `ipythonx/celebamaskhq`, attached as input, auto-discovered under `/kaggle/input`); the copy has no `identity_CelebA.txt`, so identity de-duplication uses ArcFace (`data.identity.mode: auto` -> arcface, same-person cosine 0.5 UNVERIFIED); images are not redistributed |

### 12.2 Pinned Python packages (`requirements.txt`, pinned to the J00 image; J01 ran v1 end to end with exactly these)
Provided by the Kaggle image (checked by `scripts/check_env.py`, not reinstalled): torch==2.11.0 (reported `2.11.0+cu128`), torchvision==0.26.0 (`0.26.0+cu128`), numpy==2.1.3, pillow==12.3.0, safetensors==0.8.0, scipy==1.16.3, matplotlib==3.10.0.

Installed by the job runner (core): diffusers==0.40.0, transformers==5.16.1, accelerate==1.14.0, huggingface_hub==1.29.0, kornia==0.8.3, lpips==0.1.4, kneed==0.8.6, scikit-image==0.25.2, scikit-learn==1.6.1, pandas==2.3.3, pyyaml==6.0.3, tqdm==4.67.3, optuna==5.0.0.

Installed only for jobs with `extras: [eval]` (face scoring): insightface==0.7.3 (not in the image; compiles Cython at install time), onnxruntime-gpu==1.22.0 (CUDA 12 build; the unpinned latest wheel needed CUDA 13 `libcublasLt.so.13` and silently fell back to CPU in S05), opencv-python-headless==4.14.0.94.

Optional, never auto-installed: facenet-pytorch (`pip install --no-deps`, only for L_id) and onnx2torch.

Also: the job runner uninstalls `torchao` (the image ships 0.10.0, which diffusers/peft reject when loading IP-Adapter FaceID) (`scripts/run_job.py`). Image-probe values in `env.json`: lpips, insightface, onnxruntime-gpu and kneed were "not-installed" in the base image.

Local CPU dev (`requirements-dev.txt`, lower bounds only): torch>=2.6, torchvision>=0.21, kornia>=0.7.0, diffusers>=0.28, transformers>=4.40, numpy>=1.26, pillow>=10, pyyaml>=6, scikit-image>=0.22, scikit-learn>=1.4, kneed>=0.8, pandas>=2.1, lpips==0.1.4, pytest>=8, tqdm. Per-agent floors are in `requirements.d/{protect,eval,baselines}.txt` (e.g. diffusers>=0.25 tested with 0.41.0 locally, transformers tested with 5.19.0, optuna>=3.5 tested with 5.0.0). Local test runs therefore used newer diffusers/transformers than the Kaggle pins.

### 12.3 Models and checkpoints (ids from `config/config.yaml` and `config/parts/eval.yaml`; `revision: null` everywhere, i.e. not pinned to a commit in the config)
| Role | ID / pack | Used by |
|---|---|---|
| SD 1.5 (VAE + UNet in protection; img2img editor; DiffPure; VAE round trip) | `stable-diffusion-v1-5/stable-diffusion-v1-5` (`models.sd15`) | v2 protection, editors, purifiers |
| SD 1.5, v1 id | `runwayml/stable-diffusion-v1-5` (`models.sd15_v1`); J00 probe: both ids resolve to commit `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` and the VAE weights have the identical sha256 `a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c` (`_vae_sha256_identical_between_sd15_ids: true`). `src/protect/v1.py` falls back to the `sd15` id if the runwayml repo cannot be loaded. | v1 reproduction |
| SD 1.5 inpainting | `stable-diffusion-v1-5/stable-diffusion-inpainting` | editor `sd_inpaint` |
| SDXL base | `stabilityai/stable-diffusion-xl-base-1.0` | editor `sdxl_img2img` |
| SDXL VAE (fp16-safe) | `madebyollin/sdxl-vae-fp16-fix` | encoder ensemble in v2_final; SDXL editor |
| InstructPix2Pix | `timbrooks/instruct-pix2pix` | editor `ip2p` |
| IP-Adapter FaceID | `h94/IP-Adapter-FaceID`, weight `ip-adapter-faceid_sd15.bin` | editor `ipadapter_faceid` (face embedding comes from insightface) |
| CLIP for L_sem (protection) | `openai/clip-vit-base-patch32` (J00 probe commit `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`) | v1 and v2 L_sem |
| CLIP for the CLIP-I metric (evaluation) | `openai/clip-vit-large-patch14` (J00 probe commit `32bd64288804d66eefd0ccbe215aa642df71cc41`); deliberately different from the loss CLIP so L_sem cannot game the metric | `src/eval/clip_metric.py` |
| LPIPS backbone | `vgg` (`lpips_net: vgg`, package lpips==0.1.4; downloads torchvision VGG weights) | invisibility LPIPS and edit LPIPS |
| Face recognition (evaluation) | insightface pack `buffalo_l`, detection size 640x640, image padded by 0.4 per side; ArcFace embedding | PSR |
| Face embedder for L_id (not used) | facenet-pytorch InceptionResnetV1, pretrained `vggface2` | `src/protect/face_loss.py`, lambda_id = 0 |
| Restoration purifier | `caidas/swin2SR-classical-sr-x2-64` | `restore_swin2sr` (half-size down-up then super-resolution) |
| GrIDPure | guided-diffusion `256x256_diffusion_uncond.pt` from `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt` + repo github.com/ZhengyueZhao/GrIDPure | NOT available / not run |
| Mist target image | `MIST.png` from psyker-team/mist@99f5f3c455022bef77ef25ba440a4289ea472d25 (fetched at run time, not vendored) | `mist` baseline |
| Official code checked against | MadryLab/photoguard@686bea75c786cb46c88fc396a0cd0ee3d7d28c2e; psyker-team/mist@99f5f3c455022bef77ef25ba440a4289ea472d25 | baselines (re-implementations, never called "official") |
| Adaptive attacker | own small conv U-Net autoencoder (`base_channels 32`, `depth 3`), trained in S15 | E8 |

Status note: in `config/config.yaml` the comment says model ids are "UNVERIFIED until env probe / first load"; the J00 probe verified only that the SD1.5, CLIP B/32 and CLIP L/14 ids resolve and recorded their commits/hashes (`results/J00/env_probe/hf_models.json`); the other ids were confirmed by being loaded in later Kaggle sessions, not by a separate probe file.

### 12.4 Other tooling
- Optuna 5.0.0 TPE sampler with SQLite storage (two worker processes) for tuning.
- Kornia 0.8.3 `jpeg_codec_differentiable` for the differentiable JPEG training purifier; PIL for real JPEG test-time purifiers.
- scikit-image SSIM and a Gaussian-window SSIM (`src/protect/losses.py`) inside the loss; scikit-learn KMeans + `kneed` for the v1 image selection (`data.v1_selection`: kmeans_random_state 42, n_init 10, k_range [1,20], k_eval 70, fallback_k_calib 4).
- pandas / numpy / scipy for analysis; matplotlib for figures (Okabe-Ito palette); bootstrap 10000 resamples, 95% CI, seed 2026.
- Kaggle runner: `kaggle/job_runner.ipynb` + code bundle built by `kaggle/build_bundle.py`; no tokens or passwords anywhere (Hugging Face downloads anonymous).

## 13. Testing and software quality

### 13.1 Recorded results
- Phase 0: 32 tests passed (also run inside job J01 on Kaggle before the job: `results/J01/logs/tests.log` records "32 passed in 57.36s"; `reports/PHASE_0.md`).
- Integration worktree handoff (`docs/SESSION_HANDOFF.md`): "296 passed, 1 skipped" (one template test failed and was fixed in commit f63becc).
- Later handover (`docs/LEARNING_HANDOVER.md`): "297 passed, 1 skipped at last run".
- Counting test functions in `tests/` at this commit gives 223 `def test_` functions in 40 test files (the pass counts are higher because of parametrised tests). I did not re-run the suite for this document; the pass counts above are the ones recorded in the repo. The one skip is `tests/test_eval_purify.py` (`pytest.skip("adaptive_ae present")`, a plug-in-absence test that is skipped once the module exists).
- Test configuration `pytest.ini`: `testpaths = tests`; markers `slow` (needs downloaded weights) and `gpu` (needs CUDA); FutureWarning and DeprecationWarning ignored. Rule adopted during the project: tests must pass before a Kaggle bundle is built (after a bundle was built before a failing test result was read); the bundle also contains `tests/` and J01 ran them on Kaggle.

### 13.2 CPU stand-in design
No real SD, CLIP or ArcFace weights are used by any test. Two mechanisms: (1) tiny random-initialised stand-ins with the same call signatures as the real classes (`tests/test_protect_fakes.py`: `TinyVAE` with stride-8 4-channel latent and `config.scaling_factor`, `TinyUNet`, `TinySem`, `TinyFace`, `make_models`, `base_cfg`, `make_image`; `src/utils/tiny_models.py` for `--backend tiny` which builds tiny diffusers/CLIP modules offline); (2) fake backends injected into the evaluation loop (fake editors, face scorer, CLIP, LPIPS in `tests/test_eval_runner.py`; `components=` injection in `V2Protector` for models, CLIP, extra VAEs, face embedder). `tests/test_protect_diffusers_api.py` uses REAL diffusers classes with tiny random weights to check API compatibility and that checkpointing does not change outputs (skipped if diffusers missing). Consequence: tests prove wiring, shapes, budgets, determinism, gradient flow, resume and statistics; they do NOT prove protection quality, and real weights are exercised only on Kaggle (and the first real run was the S04/S05 smoke sessions).

### 13.3 Test files and what each checks
| File | n tests (functions) | What it checks |
|---|---|---|
| `tests/conftest.py` | fixtures | adds repo to path; `synth_dir` fixture of 3 synthetic 64 px JPEGs |
| `tests/test_config.py` | 4 | real config loads and v1 preset; override parsing; parts merge in sorted order; stable hash and validation |
| `tests/test_budget.py` | 2 | budget scale conversions; round vs truncation uint8 |
| `tests/test_seeding.py` | 2 | `derive_seed` properties and across processes |
| `tests/test_runlog.py` | 3 | run files written, conflicting config refused; bundle-info fallback; shard files separate |
| `tests/test_metrics.py` | 3 | PSNR for constant difference 4; v1 metrics equal an oracle copy of the v1 function; png metrics |
| `tests/test_stats.py` | 1 | bootstrap CI properties |
| `tests/test_v1_patches.py` | 7 | v1 patch markers; paired EoT halves identical; structure loss zero and growing, close to skimage; loss sign; loss injection; legacy eval shim lazy |
| `tests/test_v1_pgd_tiny.py` | 2 | v1 PGD with tiny model: bounds, determinism, non-zero gradients |
| `tests/test_protect_fakes.py` | 2 | fakes match diffusers signatures; VAE autocast flag is no-op on CPU |
| `tests/test_protect_pgd.py` | 19 | projection respects eps in uint8 terms; budget saturates; PNG-saved budget; clamp at range edges; determinism by seed; info contents and all terms logged; ascent increases L_vis when it is the only term; loss scale invariance; no-purifier and zero-weight paths; anti-purify gradients; and further loop checks |
| `tests/test_protect_purifiers.py` | 9 | gradient flows through every purifier; JPEG quality random; down-up loses detail and held-out range disjoint; one-step purifier aux and scaling factor; t range inclusive; mixture renormalises/excludes/empty; PIL-BPDA forward identity and straight-through backward; STE clamp |
| `tests/test_protect_registry.py` | 8 | part file defines all presets; every preset constructs and resolves without loading models; preset contents; tuned final layer overrides default and flows to eps variants; unknown preset; minimal-cfg dispatch; every preset runs with injected components; ablation term selection reaches the loop |
| `tests/test_protect_tune.py` | 3 | Optuna search resume and constraints; confirm stage; params-to-override and constraints |
| `tests/test_protect_diffusers_api.py` | 2 | real diffusers tiny modules: gradient flow, checkpointing equivalence |
| `tests/test_run_protect_smoke.py` | 3 | run_protect smoke with resume and shards; `--resume_from`; info sidecar |
| `tests/test_baselines_noise.py` | 8 | uniform budget exact in uint8; determinism; sign noise; interface; budget override; Gaussian PSNR matching; per-image target; missing CSV raises |
| `tests/test_baselines_attacks.py` | 8 | PhotoGuard and Mist budget, gradient flow, progress, determinism, texture target, semantic term |
| `tests/test_baselines_device.py` | 1 | S04 regression: PhotoGuard/Mist expose `.device` |
| `tests/test_baselines_ae.py` | 7 | adaptive AE shapes, patch sampler, training smoke, error reduction and checkpoint round trip, missing weights is NotAvailable, pair listing |
| `tests/test_eval_runner.py` | 18 | edit runner end to end with fakes: rows, schema, cache, prompt rotation, resume, checkpoints, paired seeds give zero LPIPS, no-face handling, inpaint mask, edit-saving policy, two shards equal unsharded, header mismatch refused, impostor threshold math, jobs-file sequencing, unavailable purifier skipped, etc. |
| `tests/test_eval_tiers.py` | 12 | tier expansion counts (H, P, M, ablation patterns, eps patterns), prompt rotation, sharding partitions, row keys, schema columns, derive_seed |
| `tests/test_eval_purify.py` | 9 | cheap purifiers deterministic; JPEG order and PIL equality; down-up loses high frequencies; aliases; registry names; model purifiers construct lazily; adaptive AE plug-in contract (one skip) |
| `tests/test_eval_gridpure.py` | 5 | crop boxes, corner packing, identity patch function, windows blend, NotAvailable behaviour |
| `tests/test_eval_masks.py` | 4 | bbox mask keeps face; ensure face unmasked; parsing mask; provider cache/fallback |
| `tests/test_eval_clip_shape.py` | 1 | CLIP-I embedding independent of the return type of the vision tower (transformers 4 vs 5 regression) |
| `tests/test_eval_analysis.py` | 1 | loading NPZ sources skips foreign layouts |
| `tests/test_faces_padding.py` | 2 | S05 regression: padding and mapping boxes back; missing embedding is an error, not NaN |
| `tests/test_isolation.py` | 3 | Half B never imports Half A and vice versa; baselines import only `src.utils` |
| `tests/test_integration_wiring.py` | 7 | every method name dispatches through the registry with the real merged config; v1 preset keeps true step; baselines; unknown name raises; adaptive AE builder signature; final.yaml layer picked up; target presets dispatch |
| `tests/test_data_splits.py` | 9 | splits disjoint/deterministic; seed changes selection; ArcFace de-duplication gate; messy strata and pose rule; leak check; frozen split writing |
| `tests/test_data_prepare.py` | 5 | messy degradation parameters; PNG 512 lossless idempotent; end to end without cache |
| `tests/test_limit_order_consistency.py` | 1 | S05 regression: `--limit` picks the same images in protection and evaluation |
| `tests/test_job_sessions.py` | 5 | every session in `kaggle/jobs.yaml` resolves in dry run; step names unique; scripts exist; dependent sessions declare `prev_root`; a mini session runs end to end with fake backends |
| `tests/test_jobs_expand_nonempty.py` | 1 | S07/S08 regression: every eval job expands to at least one job and says which purifiers it wants |
| `tests/test_scripts.py` | 3 | budget_check pairing; prepare_v1_data from manifest; J01 pipeline end to end with tiny models (sharding, merge, resume) |
| `tests/test_analysis_stats.py` | 18 | bootstrap CI determinism/coverage; Wilcoxon; paired alignment by image id; threshold = 95th percentile of impostor similarities; PSR rules (no face = success); missing threshold gives NaN; gates |
| `tests/test_analysis_tables.py` | 12 | every table written in four formats; headline content and "not run" banner; numbers and gates JSON; E9 mapping; missing data is "not run", never fabricated; figures on synthetic data |
| `tests/test_analysis_ingest.py` | 7 | ingest validates and de-duplicates; coverage; thresholds from NPZ; idempotent re-ingest; safe unzip (zip-slip blocked); fills NaN edit LPIPS resumably; smoke sessions excluded |
| `tests/test_analysis_human.py` | 5 | human-study forms offline with no key inside; balanced key; analysis on synthetic answers (chance level, known accuracy, wrong key) |
| `tests/test_analysis_demo.py` | 1 | `demo.py --cpu-dry-run` wiring |

### 13.4 Software-quality practices visible in the repo
- Regression tests named after real bugs: S04 device, S05 face padding / insightface NaN embedding / `--limit` order / unavailable purifier, S07-S08 empty purifier tier expansion, transformers-5 CLIP output type, duplicate step names in jobs.
- Resolved configuration, git hash (with dirty flag), library versions, environment and seed are written for every run; results CSV schema is fixed (`src/eval/schema.py`) and a header mismatch is refused on resume.
- Frozen interfaces (Protector, Editor, EvalPurifier, FaceScorer) and import isolation between protection and evaluation.
- Budget is validated end to end: configs on the standard 0-255 scale, quantised outputs, `scripts/budget_check.py` measured the saved PNGs (4 gray levels).
- Known gaps: no test uses real weights; GPU behaviour (fp16, checkpointing speed, CUDA providers) is only observed on Kaggle; local CPU runs used newer diffusers/transformers than the Kaggle pins.


---

> **Part 3 of the DiffShield v2 Master Document.** Source of truth: the repository at `C:\Users\karth\Documents\DiffShield-wt\integration` (branch `v2/release`); every path below is relative to that repo root unless it starts with a drive letter. Status labels used throughout: **IMPLEMENTED AND RUN**, **IMPLEMENTED BUT NOT RUN**, **PROPOSED ONLY**. A value I could not find in any file is written "UNKNOWN / not recorded". No face image is included or needed.

---

## 14. Datasets and data splits

### 14.1 Source dataset (IMPLEMENTED AND RUN)

- **Dataset:** CelebA-HQ face photographs, taken from the **CelebAMask-HQ** distribution. `data/splits/splits_meta.json` records `n_images_total: 30000` and the attribute file `/kaggle/input/datasets/ipythonx/celebamaskhq/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt`. In every Kaggle notebook the dataset was attached as the Kaggle dataset **`ipythonx/celebamaskhq`** (named in `docs/KAGGLE_LAUNCH.md` step 3, `kaggle/job_runner.ipynb`, and `results/S06/S06/data/prepare_meta.json`, which also gives `celebahq_img_dir = .../CelebAMask-HQ/CelebA-HQ-img` and `parsing_dir = .../CelebAMask-HQ/CelebAMask-HQ-mask-anno`).
- **Original resolution:** the source JPEGs are 1024 px (`scripts/prepare_data.py` docstring: "open the 1024 px CelebA-HQ jpg"). The project's own dataset-acquisition story beyond "attached the Kaggle dataset `ipythonx/celebamaskhq`" is UNKNOWN / not recorded (no download script, no checksum of the original archive).
- **Files used from the distribution:** `CelebA-HQ-img/*.jpg`, `CelebAMask-HQ-attribute-anno.txt` (40 binary attributes), `CelebAMask-HQ-pose-anno.txt` (yaw/pitch/roll; used for the pose stratum of the messy split; units unverified, so only ranks are used), and `CelebAMask-HQ-mask-anno` (face-part parsing masks, used for the keep-face inpainting masks). The Kaggle copy has **no** `identity_CelebA.txt` (comment in `config/parts/data.yaml`), so true identity labels were unavailable and the identity gate fell back to ArcFace similarity (`splits_meta.json`: `identity.mode = "arcface"`).
- **Image preparation** (`scripts/prepare_data.py`, `config/parts/data.yaml: prepared`): open the 1024 px JPEG, resize to **512x512 with PIL BICUBIC**, save as **lossless PNG**. (v1 re-saved its inputs as JPEG at PIL default quality 75; v2 does not, so all v1 numbers were re-run on the PNGs for the paired comparisons. The Phase 0 reproduction J01 is the exception: it used v1's JPEG-q75 inputs on purpose, `reports/PHASE_0.md`.)
- **Copyright / redistribution rule** (binding project rules): `docs/HANDOFF_v2.txt` item 14: "Datasets (CelebA-HQ and others) are for research use. Do not redistribute them or publish identifiable images." `README.md` "Data and privacy": CelebA-HQ images are research-use only; protected, purified and edited images, human-study forms and answer keys stay local and git-ignored; only CSV/JSON/txt results and file-name lists are committed. `.gitignore` enforces `results/**/images/`, `results/**/edits/`, `results/cache/`, `data/images/`, `*.zip`. Consequence for a paper: **no CelebA-HQ face (original, protected, edited) may be reproduced in the paper figures as-is; only file-name lists and numbers are shareable.** Note: two face-containing PDFs (`results/figures/before_after.pdf`, `failure_gallery.pdf`) were found on the public `master` branch on 2026-10-25 and had not been removed at that time (project history facts, not verifiable from the working tree).

### 14.2 Frozen split lists (IMPLEMENTED AND RUN; produced by Kaggle job S03, `scripts/make_splits.py`, seed 2026)

| list file (`data/splits/`) | role | n | sha256 of the LF-joined text (from `splits_meta.json`, re-verified locally) |
|---|---|---|---|
| `test.txt` | all headline numbers | **100** | `7dd3db3ffba1bd2e...` |
| `messy.txt` | "harder conditions" photos, reported separately | **20** | `e03bae8f4d549939...` |
| `val.txt` | all tuning and screening (Optuna, S24/S25) | **20** | `fe56b9221ae0d71e...` |
| `train.txt` | reserved; no result in this repo uses it | **20** | `c2fc64e43ac3d80a...` |
| `ae_pool.txt` | training pool of the E8 adaptive attacker | **120** | `480b9842dc4e7577...` |
| `v1_calib.txt` | v1's 4 calibration photos, **excluded from every v2 split** | 4 | n/a |
| `v1_eval70.txt` | v1's 70 evaluation photos | 70 | n/a |

Full hashes are in `data/splits/splits_meta.json` (`sha256` key). I verified locally: the five lists are mutually disjoint (280 unique file names = 100+20+20+20+120), none contains a calibration photo, and the first 70 lines of `test.txt` equal `v1_eval70.txt` (same names, same order). The split files are frozen: `make_splits.py` refuses to overwrite a differing file without `--force`.

### 14.3 How each list was chosen (code: `scripts/make_splits.py`, design in its docstring)

Common machinery: attribute-diverse selection like v1. **KMeans** (`k-means++`, `n_init=10`, `random_state = split seed + offset`) on the 40 binary CelebA attributes; within each cluster the member closest to the centroid that **passes the identity gate** is taken (up to 60 tries per cluster; shortfalls are filled by globally nearest-to-centroid candidates). Order of selection: test (extra 30), then messy, then val, train, ae_pool, each time excluding everything already used. KMeans seeds: test `2026+1`, messy `2026+2`, val `2026+3`, train `2026+4`, ae_pool `2026+5`.

- **test (100)** = all **70 v1 evaluation photos** (kept verbatim, in v1 order, so v1 vs v2 can be compared on the same photos) + **30 new** attribute-diverse photos (KMeans with k = 30). The 4 v1 calibration photos are excluded from everything.
- **messy (20)** = quota by stratum: blur 5, lighting 5, occlusion 4, pose 6 (`splits_meta.json: messy.quota`, with the 20 records and stratum labels). Rules: *blur* = CelebA "Blurry" attribute first, then low Laplacian variance (thumbnail 128 px, lowest 7% quantile); *lighting* = extreme mean luma (below 7% or above 93% quantile) or low contrast (luma std below 7% quantile) on 128 px thumbnails of an 800-image sample; *occlusion* = `Wearing_Hat` or `Eyeglasses`; *pose* = top 10% of the pool by max(|yaw|,|pitch|) from `CelebAMask-HQ-pose-anno.txt` (rank-based, `pose_source: pose_anno`). Recorded yaw/pitch of the six pose photos range from about 35 to 53 degrees in |yaw|.
- **Messy pre-degradation** (`scripts/prepare_data.py`, `config/parts/data.yaml: messy.degrade`): for the messy split only, after the 512 px resize: bicubic down to scale `s in [0.60, 0.85]`, JPEG quality `q in [50, 70]`, decode, bicubic up to 512, save PNG. Parameters are drawn per image from `sha256(seed|messy|file)` and recorded in `messy_degradation.json` (e.g. `4334.jpg`: scale 0.83, q 51). So the messy "source photo" that all messy numbers are measured against is the already-degraded PNG.
- **val (20), train (20), ae_pool (120)** = plain KMeans attribute-diverse picks (k = 20, 20, 120) from the remaining pool, identity-gated.

### 14.4 De-duplication and its caveats (read before claiming "one image per person")

- **Goal:** one image per person across all splits, so the PSR "impostor" threshold (Section 15.5) is computed between different people and no test person appears in tuning data.
- **What actually ran:** identity mode `arcface` (labels unavailable). Two images count as the same person if ArcFace cosine >= **0.5**, insightface `buffalo_l`. `config/parts/data.yaml` and `splits_meta.json` both mark this threshold **`threshold_verified: false`** (an unverified choice).
- **Caveat 1, detector misses.** The splitter used the *un-padded* detector. For the 74 forced-in v1 photos (70 eval + 4 calibration) `splits_meta.json` records `n_noface_forced: 40`: the code (`ArcFaceIdentity.register`) increments that counter exactly when a registered image has no detectable face, so those images could not be compared for identity. The project later found the detector misses **38 of the 100 test faces** on tightly cropped photos and fixed *evaluation* by padding the image (`arcface.pad_frac = 0.4`, `src/eval/faces.py`). **The test list was not re-de-duplicated with the corrected detector** (`reports/FINAL_REPORT.md` section 10). New photos (the 30 extra test, messy, val, train, ae_pool) are only accepted if a face *was* found (`conflicts()` returns True for no-face images), so the gate was effective for them.
- **Caveat 2, recorded clash counts are conditional.** `v1_calib_identity_clashes_among_themselves = 0` and `v1_eval70_identity_clashes = 0` in `splits_meta.json` only count pairs where both images had an embedding.
- **Caveat 3, same-person assumption in the threshold.** `compute_psr_thresholds` "assumes one image per identity in the split" (docstring). With 100 test photos and an unverified 0.5 gate this is an assumption, not a verified property. Direction of the possible error: any hidden same-person pair would put a genuine-identity similarity into the impostor pool and push the 95th-percentile threshold up (so PSR would be slightly easier to reach). I did not measure the size of this effect: UNKNOWN / not recorded.
- Messy photos are *not* guaranteed disjoint from test by the same detector problem, but the file-level disjointness is verified above.

### 14.5 v1's photo sets (provenance, IMPLEMENTED AND RUN as reference data)

- v1 chose photos with a two-stage unsupervised stratification (`README.md` legacy section; `config/config.yaml: data.v1_selection`: KMeans `random_state 42`, `n_init 10`, elbow search `k in [1,20]`, `k_eval 70`, `fallback_k_calib 4`): the **elbow method** picked `K_calib` archetypes that were *removed from the pool and used only for hyper-parameter tuning*, then the remaining images were clustered to give **K = 70 evaluation centroids**.
- The exact file names were not re-derived by v2; they come from v1's own manifests copied to `data/v1_ref/`: `evaluation_70_manifest.csv` (70 rows, `Eval_Image_ID face_001..`, `Original_Index`, `Filename`) and `calibration_subset_manifest.csv`. **The 4 calibration photos: `5319.jpg, 1715.jpg, 6385.jpg, 8932.jpg`** (`data/splits/v1_calib.txt`).
- v1's stored results: `data/v1_ref/stage1_per_image_metrics.csv`, `stage1_summary_metrics.csv` (70 photos: PSNR 41.911 ± 0.440 dB, SSIM 0.97496 ± 0.00480, LPIPS(VGG) 0.23156 ± 0.06304, per `reports/PHASE_0.md`), and `optimal_hyperparameters.json` (Optuna best trial number 23, objective 0.1265, alpha 0.2699475469647097, beta 0.9023875567667267, gamma 60.52658429696642). `docs/HANDOFF_v2.txt` line 45 states the v1 Optuna had 30 trials; the number of trials is not stored in the v1 json.
- v1's Phase 2 (edit test) used 15 photo pairs, one prompt, one seed (README legacy note); its outputs were lost, so only its LPIPS (0.46) can be quoted (`reports/FINAL_REPORT.md` section 10). `config/config.yaml: edit.v1_phase2` keeps its prompt ("a highly detailed portrait, cyberpunk style, neon lighting, digital art masterpiece", strength 0.55, guidance 7.5, seed 2026) but no session in `kaggle/jobs.yaml` runs it: IMPLEMENTED (config only) BUT NOT RUN.

### 14.6 Prompts used (`config/parts/eval.yaml: eval.prompts`; IMPLEMENTED AND RUN)

| editor family | prompt 0 | prompt 1 | prompt 2 |
|---|---|---|---|
| text-to-image-conditioned (SD1.5, SDXL, inpainting, FaceID) | "a person in a suit" | "a person at a beach" | "an oil painting of a person" |
| InstructPix2Pix | "put the person in a suit" | "put the person on a beach" | "turn it into an oil painting" |

No negative prompt is used. Prompt assignment: see 15.3.

### 14.7 Other data
- **Adaptive-attacker pool:** the 120 `ae_pool` photos, protected by `v2_final` (K=2 draws), used with their clean versions to train the autoencoder (Section 15.8).
- **Mist target image:** the official `MIST.png` is fetched at run time from the pinned commit `99f5f3c455022bef77ef25ba440a4289ea472d25` of the Mist repository (GPL-3.0) and is never vendored (`kaggle/jobs.yaml` S21, S24).
- **Human-study data:** none (E10 not run).

---

## 15. Experimental protocol

### 15.1 Threat model as operationalised (all in `config/*.yaml`)
- Budget: L-infinity **eps = 4/255 on the standard 0-255 scale** (`budget.eps_255: 4`, confirmed in J01: max |gray-level change| = 4 in 70/70 PNGs; v1's "8/255" was measured on [-1,1] tensors = 4/255). Sweep presets `eps2`, `eps6`, `eps8`.
- Protection settings common to the v2 family (`config/config.yaml`, `config/parts/protect.yaml`): 40 PGD iterations (`budget.iters`), step 1.0/255 for v2 presets (v1: 0.5/255), K = 4 augmentation draws (`eot.k_draws`; v1: 1), VAE in fp16 autocast during protection (`runtime.vae_dtype_protect: autocast`, ~2x faster, `docs/COST_MEASURED.md`), gradient checkpointing on. The protection internals belong to another part of this document; only what the evaluation depends on is listed here.
- Attack side: the *non-adaptive* attacker picks an editor and optionally a cleanup, without knowing the protection (headline). The *adaptive* attacker (E8) knows the method and trains a cleanup network.

### 15.2 Editors and their settings (`config/parts/eval.yaml`, `src/eval/editors.py`; all IMPLEMENTED AND RUN)

Common: diffusers pipelines, fp16, **DPM-Solver++ multistep, 2nd order (`algorithm_type="dpmsolver++", solver_order=2`) for every editor**, **30 inference steps** (`eval.num_inference_steps: 30`), **guidance scale 7.5**, safety checker disabled, no negative prompt, edits generated at `image_size 512`. (Note: `config/config.yaml` still contains an `edit:` block with `num_inference_steps: 50`; it is **not** what the evaluation uses, the `eval:` block overrides. Report 30.) `max_batch` = number of seeds generated at once.

| editor key | pipeline / weights (model id) | settings | notes |
|---|---|---|---|
| `sd15_img2img` | `StableDiffusionImg2ImgPipeline`, `stable-diffusion-v1-5/stable-diffusion-v1-5` | strength **0.3 / 0.55 / 0.75** (0.55 = headline, default); `max_batch 3` | the headline editor. Same SD1.5 VAE family the protection attacks |
| `sdxl_img2img` | `StableDiffusionXLImg2ImgPipeline`, `stabilityai/stable-diffusion-xl-base-1.0` + `madebyollin/sdxl-vae-fp16-fix` | strength 0.55; input upsampled bicubic **512 -> 1024**; `max_batch 1` | SDXL edit outputs are at 1024 px; run only for clean, v1, photoguard, v2_final |
| `ip2p` | `StableDiffusionInstructPix2PixPipeline`, `timbrooks/instruct-pix2pix` | guidance 7.5, **image guidance 1.5**, no strength parameter | many edits contain no detectable face (about 18% for clean, `results/tables/e0.md`) |
| `sd_inpaint` | `StableDiffusionInpaintPipeline`, `stable-diffusion-v1-5/stable-diffusion-inpainting` | strength **1.0**, mask = keep-face | no pixel composite of the original is pasted back (docstring) |
| `ipadapter_faceid` | SD1.5 img2img + IP-Adapter FaceID: adapter `h94/IP-Adapter-FaceID`, `ip-adapter-faceid_sd15.bin`, base `sd15` | strength 0.55, **ip_adapter_scale 0.6**, `max_batch 1`; the FaceID LoRA is **not** loaded | identity prompt = ArcFace embedding of the *input* photo computed with the **same insightface `buffalo_l` scorer used for evaluation**; no face -> zero embedding. Run as img2img (so the VAE path stays attackable) |

Keep-face inpainting mask (`src/eval/masks.py`, `eval.masks`): 255 = repaint (background, cloth, hat, hair), 0 = keep. Keep parts from the CelebAMask-HQ parsing annotation: skin, nose, l_eye, r_eye, l_brow, r_brow, l_ear, r_ear, mouth, u_lip, l_lip, eye_g, ear_r, neck, neck_l, **dilated by 8 px**; for ids without parsing masks (messy / non-CelebA) a face-bbox ellipse fallback is used; `ensure_face_unmasked` is applied on top. The threat model therefore assumes the attacker never masks the face itself (if the face were masked its pixels would be discarded and no pixel perturbation could protect it).

Model revisions: all `revision: null` in `config/parts/eval.yaml` (not pinned to a commit; weights were pulled from the Hugging Face hub at run time). The eval config comments mark several ids as "unverified" because they were authored without GPU access; they were loaded for real in the Kaggle sessions (S05 onward), but the exact hub commits used are UNKNOWN / not recorded except via `results/J00/env_probe` (J00 recorded HF model hashes at 2026-10-14).

### 15.3 Seeds, prompt rotation, saving (`src/eval/edit_runner.py`)
- **Edit seeds:** `eval.seeds: [1, 2, 3]` (3 per cell). The row's `seed` column stores k in {1,2,3}; the real generator seed is `derive_seed(2026, "edit", image_id, k)` = sha256 of `"2026|edit|<image_id>|<k>"` reduced to 31 bits (`src/utils/seeding.py`). It does **not** depend on method, purifier, editor or strength, so the clean edit and the protected edit of the same photo share the same generator seed (paired noise).
- **Purifier seed:** `derive_seed(2026, "purify", image_id, purifier)`. **Protection seed per photo:** `derive_seed(cfg.seed, "protect", image_id)` (`scripts/run_protect.py`), `cfg.seed = 2026`. Image processing order is the **sorted** list of ids, in both protection and evaluation (so `--limit N` picks the same photos).
- **Prompt rotation:** in the *headline cell* (editor `sd15_img2img` at strength 0.55) **all three prompts** are run per photo (9 edits per photo = 3 prompts x 3 seeds). In every other cell **one prompt per photo**, `prompt index = (position of the photo in the sorted split list) mod 3` (3 edits per photo). Implemented in `expand_tier` (`cells[].prompts: all_if_headline | rotate`).
- **Saving:** `eval.save_edits.all: true` and `eval.edit_images: {format: jpg, quality: 90}`: every edit is saved as **JPEG quality 90**. **ArcFace and CLIP-I are computed on the in-memory (not yet JPEG-compressed) edit**, and the E0 clean-edit embeddings used for the PSR threshold are also from the in-memory edits (`_process`: `fr = self.b.face.analyze(e_img)` before `_save_jpeg`). Only edit-LPIPS is defined on JPEG-q90 round-tripped images.
- **Unit of analysis:** the photo. Seeds and prompts are averaged within a photo first (`Analysis._per_image`), so n = 100 photos for test cells (20 for messy and val cells).

### 15.4 The cleanups (called "purifiers" in code) (`src/eval/purify_attacks.py`, `eval.purifiers`)

All are real codecs/models applied to the saved 512 px protected PNG (never the differentiable training approximations). The list `purifiers_13` in `eval.yaml` has 13 names:

| name | exact definition |
|---|---|
| `jpeg_q90`, `jpeg_q75`, `jpeg_q50` | PIL JPEG round trip, quality 90 / 75 / 50, PIL default subsampling (4:2:0) |
| `jpeg_other` | PIL JPEG quality **65**, subsampling 0 (4:4:4) |
| `downup_0.5` | bicubic resize to 0.5x, back to 512 px with the same filter |
| `downup_other` | **Lanczos** resize to **0.375x**, back with Lanczos |
| `vae_rt` | SD1.5 VAE encode -> decode (posterior **mean**), separately loaded VAE instance (fp16 on cuda), reflect-padded to multiples of 8 |
| `diffpure_0.05/0.1/0.2/0.3` | "DiffPure-style" with **SD1.5**: diffusers img2img with empty prompt, DDIM scheduler, **50 steps**, guidance 1.0, `strength` = t* in {0.05, 0.1, 0.2, 0.3}; output also passes the SD VAE. Not the original (ImageNet) DiffPure |
| `restore_swin2sr` | bicubic down 0.5x then **Swin2SR x2** (`caidas/swin2SR-classical-sr-x2-64`) |
| `gridpure` | GrIDPure (window 256, stride 128, 10 pure steps, 20 iterations, gamma 0.1): **NOT AVAILABLE**: the 256x256 guided-diffusion checkpoint was not installed; its jobs are skipped and logged ("gridpure NOT AVAILABLE ... SKIPPING 50 jobs", `results/S10/S10/logs/*.log`). So **12 of the 13 cleanups produced results** (IMPLEMENTED BUT NOT RUN for GrIDPure) |
| `adaptive_ae` | only for E8 (Section 15.8); not one of the 13 |

**Important caveat on the word "held-out".** `reports/FINAL_REPORT.md` calls the 13 cleanups "held-out". Strictly, the config's own list of held-out cleanups is only `purifiers_heldout: [jpeg_q_other, downup_other, diffpure_t0.1, diffpure_t0.2, diffpure_t0.3, gridpure]` (`config/config.yaml`). The training mixture (`purifier_train_params`) contains differentiable JPEG with quality 50-95, down-up with scale 0.5-0.9, an SD1.5 VAE round trip and (for `v2_main`/`v2_antipure`/`v2_final`) a one-step SD1.5 purifier. Therefore `jpeg_q90/75/50`, `jpeg_other` (quality 65 lies inside 50-95; only the codec and subsampling differ), `downup_0.5` (edge of the training scale range) and **`vae_rt` (the same SD1.5 VAE the attack targets)** are *not* held-out in the strict sense; they differ from training only by using the real PIL codec / a separately loaded VAE instance. `downup_other` (scale 0.375, outside the training range 0.5-0.9; a "held-out" down-up range 0.30-0.45 exists in `protect.yaml`) and the multi-step SD1.5-based DiffPure variants are genuinely not in the training mixture, though DiffPure shares the SD1.5 weights with the one-step training purifier. A paper must not describe all 13 as unseen.

### 15.5 PSR threshold computation (E0, IMPLEMENTED AND RUN in S07)

1. **E0 = clean (unprotected) photos edited by every editor** (tier H, M, P for method `clean`; S07, with the clean purifier tier redone in S10 because S07's expanded to zero jobs).
2. For each editor, take every clean edit of a test photo and compute the ArcFace cosine similarity between the edit and the **source embedding of every *other* test photo** ("impostor" pairs, same image excluded).
3. **Threshold = the 95th percentile (numpy percentile, linear interpolation) of those impostor similarities**, computed **per editor** (all prompts, strengths and seeds of that editor pooled) and also **pooled** over editors (`thresholds_from_embeddings` in `src/eval/analysis.py`, restricted to the test split ids; `scope: per_editor` is what all analyses use; the pooled value is reported but not used for PSR).
4. Values (`results/combined/psr_thresholds.json`; identical to the S07 job output `results/S07/S07/results/thresholds/psr_thresholds.json`):

| editor | threshold | impostor pairs |
|---|---|---|
| `sd15_img2img` (all three strengths pooled) | **0.10054539651520272** | 147,609 |
| `sdxl_img2img` | 0.10367028692086117 | 29,700 |
| `ip2p` | 0.10775226496545169 | 24,453 |
| `sd_inpaint` | 0.10970453486167729 | 29,700 |
| `ipadapter_faceid` | 0.1078936712001695 | 29,700 |
| pooled (all editors) | 0.1034364047631 | 261,162 |

   (`n_sources_with_face: 100`.) The pair counts are consistent with 99 other photos per edit: sd15 = 1,491 edits (1,500 minus 9 without a face), faceid/sdxl/inpaint = 300 edits each, ip2p = 247 edits (53 of 300 have no face). The `e0.md` table rounds the SD1.5 value to 0.101.
5. Interpretation: ArcFace cosine of ~0.10 is what two *different* people look like; the threshold is a deliberately strict "no longer recognisable as the same person" bar. The threshold is determined from clean edits of the **same 100 test photos** it is later applied to (no separate threshold-setting set). This is an operational definition, validated against no human judgement (no human study).
6. The threshold was **not** used inside tuning: Optuna derived its own threshold (Section 18.4).

### 15.6 Evaluation tiers (`eval.tiers` in `config/parts/eval.yaml`; these are what the runner expands)

Notation: `H cells` = {SD1.5 at strengths 0.3, 0.55 (all 3 prompts), 0.75 (rotated prompt); InstructPix2Pix; inpainting; FaceID} on the test split; `rot` = one rotated prompt per photo; 3 seeds everywhere. `methods_h` = {clean, noise_uniform, noise_gauss_psnr, photoguard, mist, v1, v2_cheap, v2_main, v2_final}. Edits per (method, photo): SD1.5 H = 9 (0.55 x 3 prompts) + 3 + 3 = 15 edits; each other cell 3 edits; so a full H method on 100 photos = 1,500 SD1.5 rows + 300 each for ip2p, inpaint, FaceID = 2,400 rows (matches `results/combined/coverage.csv`: `noise_uniform` and `noise_gauss_psnr` have 2,400 rows each; the SDXL cell adds 300 more for clean, v1, photoguard, v2_final).

| tier | split | methods | cleanups | cells | what it is for | status |
|---|---|---|---|---|---|---|
| **H** | test | `methods_h` (9 methods) x {no cleanup}; plus {clean, v1, photoguard, v2_final} on SDXL (strength 0.55, rot) | none | SD1.5 {0.3, 0.55, 0.75}, ip2p, inpaint, FaceID (+SDXL for the 4 methods) | headline: no cleanup, all editors | RUN |
| **M** | messy | clean, noise_uniform, noise_gauss_psnr, photoguard, mist, v1, v2_main, v2_final | none | SD1.5 0.55 rot + FaceID rot | robustness to messy photos | RUN |
| **P** | test | clean, v1, v2_cheap, v2_main, v2_antipure, v2_final x the 13 cleanups (SD1.5 0.55 rot); photoguard, mist x 5 cleanups {jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1}; clean, v1, v2_main, v2_final x {jpeg_q75, vae_rt, diffpure_0.1, diffpure_0.2} on FaceID; v1, v2_final x {jpeg_q75, diffpure_0.1} on ip2p; v2_antipure x none on FaceID | 13 / 5 / 4 / 2 | see left | cleanup robustness (the decision tables) | RUN except GrIDPure (not available) |
| **A** | test | abl_vis_only, abl_den_only, abl_no_sem, abl_no_pur | none, jpeg_q50, vae_rt, diffpure_0.1, diffpure_0.2 (SD1.5 0.55 rot); none on FaceID, ip2p, inpaint | per left | loss-term ablation | RUN for vis_only and den_only; **no_sem and no_pur NOT RUN (S14)** |
| **E9** | test | methods matching `eps*`, `*_eps*` | none, jpeg_q75, diffpure_0.1 (SD1.5 0.55 rot); none on FaceID | | epsilon sweep | RUN for eps2 (S17) and eps8 (S18); **eps6 NOT RUN (S23)**; eps = 4 is `v2_final` |
| **E8** | test | clean, v1, v2_final | `adaptive_ae` | SD1.5 0.55 rot + FaceID rot | adaptive attacker | RUN (S15) |
| **SMOKE** | test, val, messy (5 photos) | all (`*`) | none, jpeg_q75, downup_0.5, vae_rt, diffpure_0.1 | every editor incl. SDXL | pipeline check (J03b/S05) | RUN as smoke only, its steps failed partly (see 17.2); **numbers never reported** |
| **AP** | val | clean, v2_antipure | jpeg_q75, vae_rt, diffpure_0.1 (SD1.5 0.55 rot) | | choose between anti-purification terms on validation photos (S07/S20) | IMPLEMENTED BUT NOT RUN to completion: S07's AP jobs expanded to zero jobs; the S20 rerun has no result folder |
| **SCREEN** | val | clean, v1, v2_cheap, v2_main, photoguard, mist, v2_tgt_zero, v2_tgt_zero_hi, v2_tgt_gray | none, jpeg_q75, diffpure_0.1 (SD1.5 0.55 rot) | | screen methods on validation photos only, never on test (S24, S25) | RUN |
| **TGT** | test | v2_tgt_zero, v2_tgt_zero_mid, v2_tgt_zero_hi | none (SD1.5 {0.3, 0.55, 0.75} + FaceID rot); and {jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1, diffpure_0.2, diffpure_0.3} (SD1.5 0.55 rot) | | the target-steered family on the test photos (S26-S28); clean, v1, v2_main and v2_cheap reuse their existing cells | RUN |

Check on row counts: `results/combined/coverage.csv` gives e.g. `test/clean` 8,100 rows, `test/v1` 8,700, `test/v2_final` 8,700, `test/v2_tgt_zero_mid` 3,900, `test/photoguard` 4,200, `test/mist` 3,900, noise methods 2,400 each. The combined edit table has **76,980 rows** (180 duplicate keys dropped, 0 conflicts; `results/combined/ingest_report.json`). Planned cells missing from the result set: 26 (`coverage_vs_plan.csv`): 16 cells of `abl_no_sem`/`abl_no_pur` (tier A, 8 each), 4 E9 cells listed under the pattern name (eps6 not run), and 6 GrIDPure cells (one per method of tier P).

### 15.7 Statistics (code: `src/eval/analysis.py`, `src/eval/stats.py`; `config/config.yaml: stats`)
- **Per-photo value first**: for each (cell, photo): PSR = mean over that photo's edits (seeds x prompts) of the hit indicator; ArcFace similarity and CLIP-I = mean over edits; no-face = mean of face-not-found indicator.
- **Confidence interval**: 95% **percentile bootstrap over photos**, **10,000 resamples**, `numpy.random.default_rng(2026)`, resample indices cached per (n, n_boot, seed) (`boot_ci`; identical to `bootstrap_ci` in `stats.py`). Config: `stats: {n_boot: 10000, ci: 0.95, unit: image, bootstrap_seed: 2026}`. **Exception:** `scripts/readout_final.py` (which produced `results/tables/final_readout.md`, the source of most numbers in `reports/FINAL_REPORT.md` sections 7.3-7.9) uses **2,000 resamples** with percentiles 2.5/97.5 from a single shared `default_rng(2026)` (so its CIs depend on call order). `docs/EVAL_PLAN_FINAL.md` also says 2,000. The generated tables (`results/tables/*.md`) say 10,000. Both are percentile bootstraps over photos; the two procedures give slightly different interval endpoints.
- **Pairing**: every comparison between methods is **paired on the same photos** and the same cell (same editor, strength, cleanup): the per-photo difference is bootstrapped (`paired_stats`, inner join on `image_id`); the quantity reported is the paired difference with its CI, never two separate CIs. A two-sided **Wilcoxon signed-rank p** (zeros dropped) is computed alongside but is **uncorrected for multiplicity**; claims in the report rest on "the paired CI excludes 0". Paired ratios of means (`paired_ratio`) use a joint image bootstrap. Clopper-Pearson (`binom_ci`) exists for proportions.
- **Net protection** = PSR(method) - PSR(clean photos) under the same cleanup/editor/strength, paired by photo (see 16.2).
- **Multiple comparisons:** many cells are compared and **no multiplicity correction is applied** (stated in `FINAL_REPORT.md` section 7.10 and 10).
- Smoke-test numbers are never reported (project rule); validation photos are never mixed into test statistics.

### 15.8 The adaptive attacker E8 (IMPLEMENTED AND RUN, S15)
`src/eval/adaptive_ae.py`, `scripts/train_adaptive_ae.py`, `config/parts/baselines.yaml: adaptive_ae`. A small conv U-Net autoencoder (`base_channels 32`, `depth 3`) trained on (protected -> clean) pairs of the 120 `ae_pool` photos, protected with `v2_final` at K=2 (`methods.v2_final.eot.k_draws=2`), on random 128 px patches, batch 16, lr 2e-4, **L1 loss only** (`lpips_weight 0`), 60 minute wall-clock budget; training log `results/S15/S15/adaptive_ae/ae_final_train_log.json`: **35,496 steps**, `elapsed_s 3600.07`, validation (n = 8 held-out pool photos) PSNR protected-vs-clean 40.84 dB, AE-output-vs-clean 45.12 dB. It is then applied fully-convolutionally to the full 512 px protected test photos before editing (clean, v1 and v2_final photos; `adaptive_ae` cleanup). The code states it is a weak attacker by design and not the LightShed implementation. It was trained against `v2_final` photos only, and **not run against the target-steered variants** (NOT RUN).

### 15.9 Baselines and controls (IMPLEMENTED AND RUN; re-implementations)
`config/parts/baselines.yaml`: `noise_uniform` (uniform noise at eps = 4/255), `noise_gauss_psnr` (Gaussian noise whose per-photo PSNR is matched to v1's protected PSNR from `data/v1_ref/stage1_per_image_metrics.csv`), `photoguard` (encoder attack, 1000 iterations, step 2.55/255 with linear decay to step/100, target = zero latent, **our** eps 4/255 not the official ~7.65/255), `mist` (fused semantic + textural loss, 100 iterations, step 1/255, `w_sem 10000`, prompt "a painting", official `MIST.png` target fetched at run time, our eps 4/255 not the official 16). Hyper-parameters were checked against the official code at the commits cited in the yaml; they are our re-implementations. Measured protection cost: PhotoGuard ~232 s/photo, Mist ~58 s/photo (Section 19.3).

---

## 16. Metric definitions

All metrics are computed per edit and then averaged over seeds/prompts within a photo (unit = photo, Section 15.7).

### 16.1 Identity: ArcFace similarity (`src/eval/faces.py`, `src/eval/edit_runner.py::_process`)
- **Model:** insightface **`buffalo_l`** pack (`allowed_modules: detection, recognition`, `det_size (640, 640)`), onnxruntime GPU (pinned 1.22.0) with CPU fallback. Embedding: 512-d L2-normalised ArcFace (`normed_embedding`) of the **largest detected face** (by bbox area).
- **Padding:** before detection the image is padded on every side by `pad_frac = 0.4` of `max(H, W)` (about 205 px for a 512 px image) with constant value **127** (`np.pad`), then bbox/keypoints are shifted back. Reason: without padding the detector missed 38 of 100 tightly cropped test faces (S03/S05 finding); edits showed the same miss.
- **Similarity:** `arcface_sim = cos(embedding of the edit, embedding of the SOURCE photo)` where the source is the clean prepared 512 px PNG of that photo (for messy photos: the degraded messy PNG). NaN when no face is found in the edit or in the source. The *edit* is the in-memory model output (not JPEG-compressed), and the *source embedding* is cached. Higher = more like the same person.
- `face_detected` = 1 if an embedding was obtained from the edit. `fr2_sim` (second recogniser) is always NaN: **no second face recogniser was used**, so identity is judged by the same network family for FaceID conditioning and scoring (a possible optimistic/pessimistic bias for the FaceID editor that was not measured).

### 16.2 PSR, net protection, paired difference, no-face rate, per-photo share

- **Hit (per edit)** (`analysis.py::Analysis._per_image`): hit = **1 if no face was detected in the edit**; otherwise hit = 1 if `arcface_sim < threshold(editor)` else 0 (NaN similarity with a detected face, e.g. source without face, is excluded). `threshold(editor)` = per-editor E0 threshold of Section 15.5.
- **PSR (protection success rate)** of a cell = mean over photos of the per-photo mean hit. (`scripts/readout_final.py` computes the same thing as `(sim < thr) | sim is NaN`.)
- **No-face rate** = mean over photos of the per-photo share of edits with `face_detected == 0`. Always reported next to PSR because no-face edits count as protected. The audit in `FINAL_REPORT.md` 7.10 also recomputes PSR **excluding no-face edits**.
- **Net protection** = PSR(method) minus PSR(`clean` photos) in the **same cell** (same editor, strength and cleanup), paired per photo; reported with the paired bootstrap CI (`readout_final.py::cell`). Needed because cleanups (DiffPure especially) change clean photos too.
- **Paired difference** (vs v1, vs the noise control, ablation vs `v2_main`, purified vs unpurified): per-photo difference of PSR, bootstrap CI over photos; "supported" = CI excludes 0.
- **Per-photo fully-protected share**: the share of photos whose per-photo PSR equals 1, i.e. protected for **every** seed and prompt of the headline cell (9 edits). Values reported in `FINAL_REPORT.md` 7.10: 0.0% clean, 2.0% v1, 5.0% `v2_tgt_zero_mid`, 35.0% PhotoGuard. I found **no script in the repository that computes this share** (it was an ad-hoc audit computation); the definition above is inferred from the report text: treat as definition "as reported", code location UNKNOWN / not recorded.
- **Mean ArcFace similarity** (`sim`) is the threshold-free companion measure, averaged over edits where a face was found.

### 16.3 CLIP-I (`src/eval/clip_metric.py`)
Cosine similarity between CLIP image embeddings of the **source photo** and the **edit** (both converted to RGB; fp16 on GPU; `CLIPProcessor` default preprocessing; embedding = `visual_projection(vision_model(...).pooler_output)`, L2-normalised). Model: **`openai/clip-vit-large-patch14`** (`eval.models.clip_eval`), deliberately different from the `clip-vit-base-patch32` used in the protection loss `L_sem`, so the loss cannot game the metric. A secondary, similarity-to-source measure (higher = edit closer to the original image content), not an identity measure. Table: `results/tables/clip_i.md`.

### 16.4 Invisibility: PSNR, SSIM, LPIPS and the rule

- **PSNR** = `20 log10(255 / sqrt(MSE))` over all pixels/channels of the uint8 clean PNG vs the uint8 protected PNG (`src/eval/metrics.py::psnr_u8`).
- **SSIM** (the one used for the protect table `protect.csv: png_ssim`, `metrics.py::ssim_u8`): scikit-image `structural_similarity`, `channel_axis=-1`, `data_range=255`, **Gaussian weights, sigma 1.5, `use_sample_covariance=False`**. NOTE: the `inv_ssim` stored on each *edit row* is computed by `edit_runner.ssim_pil` with scikit-image **defaults** (uniform 7x7 window), so the two SSIMs differ slightly; the invisibility tables use the protect-table values (`invisibility_by_method` prefers `protect.csv`, edit-row `inv_*` only as fallback). The SSIM inside the protection loss and the Optuna constraint (`src/protect/losses`, 11x11 Gaussian window, sigma 1.5, [0,1] RGB) is a third implementation (`tune_optuna.py` docstring warns it may differ from the harness).
- **LPIPS** (protected vs clean): `lpips` VGG (`eval.models.lpips_net: vgg`), inputs scaled to [-1,1].
- **Two conventions** are written (`metrics.py` docstring): `v1` (tensor -> uint8 by truncation, exact replica of v1) and `png` (the rounded uint8 image that would actually be posted). All v2 invisibility tables use the **published-PNG** convention.
- **Maximum pixel change** `png_maxabs_u8` must equal the stated eps (4 for v1/v2 methods; confirmed `4` in `final_readout.md` for every listed method).
- **The invisibility rule (G1):** per photo **PSNR >= 38 dB and SSIM >= 0.95**. Reported as mean (min) and as the *number of photos violating*, because a mean can hide violations (`FINAL_REPORT.md` 7.1: `v2_final` mean SSIM 0.9549 but 29/100 photos below 0.95; PhotoGuard 100/100 below 38 dB and below 0.95; Mist 50/100 below 38 dB and 100/100 below 0.95). The rule is the project's choice (`config.yaml: metrics.invisibility_min`); no human study confirms invisibility.

### 16.5 Gates (code: `analysis.py::gate_e3/e4/e5/e6/gate_invisibility`, parameters in `GateParams`; tables `results/tables/gate_*.md`, `gates.md`)
The gate *values* (2x, one half) come from the project brief (`docs/HANDOFF_v2.txt`); the E3 cut-offs and the purifier sets are analyst choices (the code says so); all can be changed with `--gate_param key=value`. A failed gate is reported, not redefined.

| gate | definition in code | verdict recorded in `results/tables/gates.md` |
|---|---|---|
| **E3** (does cleanup break v1?) | For v1 (SD1.5, 0.55), over the core cleanups {jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1, diffpure_0.2}: mean **relative PSR drop** `1 - PSR(cleanup)/PSR(none)`. >= 0.5 "DEGRADES (continue with Phase 5)"; < 0.2 "BARELY DROPS (recommend skipping Phase 5)"; between: "AMBIGUOUS". Needs >= 3 core cleanups available | **BARELY DROPS** (raw-PSR rule; ignores that cleanups also change clean photos, hence the net-protection tables) |
| **E4** (do cheap cleanups in the loop help?) | For `v2_cheap` vs v1 under {jpeg_q75, jpeg_q50, downup_0.5, vae_rt}: point estimate `PSR(v2_cheap) >= 2 x PSR(v1)` in each, and `v2_cheap` mean PSNR >= 38. Status "DEGENERATE" if v1 PSR >= 0.5; "trivial" if < 0.02. Any FAIL, or PSNR mean < 38 -> FAIL | **FAIL** (v2_cheap mean PSNR 41.58 dB over 100 photos; 0 below 38) |
| **E5** (does the one-step purifier transfer?) | `v2_main`: `PSR(diffpure_0.1) >= 0.5 x PSR(no cleanup)` (SD1.5 0.55, same photos); "robust" if the paired-ratio CI lower bound also >= 0.5; informational rows for diffpure_0.2/0.3, restore_swin2sr, gridpure and for `v2_final` | **PASS (robust)** |
| **E6** (anti-purification terms) | `v2_antipure` vs `v2_main`, paired PSR difference under diffpure_0.1/0.2/0.3: GO if the 95% CI lower bound > 0 for at least one | **GO (gain on 1 of 3 purifiers)** |
| **INV** (invisibility) | per method on test: PASS if mean PSNR >= 38, mean SSIM >= 0.95 and zero photos below either; "FAIL" if a mean is below; else "MARGINAL" (mean ok but some photo violates). Notes human 2AFC accuracy if available | **FAIL/MARGINAL** (and "human 2AFC result not available (E10 not run)") |

### 16.6 Edit-LPIPS and why it is not reported
- **Definition** (`clip_metric.edit_lpips`, `edit_runner._process`): LPIPS-VGG distance between the **clean-photo edit** and the **protected-photo edit**, same editor / prompt / strength / generator seed, **both JPEG-q90 round-tripped**; large = the protected edit differs a lot from what the attacker would have got from the unprotected photo. Rows for `clean` + cleanup measure the cleanup's own effect.
- **Status: not reported anywhere in the report because the coverage is incomplete and uneven.** `results/combined/ingest_report.json`: 512 pairs were computed locally on CPU (`edit_lpips_cache.csv`; the project's text says "513", the file has 512 data rows plus a header), 53,908 rows are missing. In addition 22,560 rows carry a value computed *inline on Kaggle* (23,072 of 76,980 rows have a value in total), but only for method subsets (v1, v2_cheap, v2_main, clean-under-cleanup on the test split; mist, photoguard, v2_cheap/main/tgt_* only on the 20 validation photos). The headline test cells of `v2_final`, the three target-steered variants, PhotoGuard, Mist and the noise controls have **no** edit-LPIPS, so no complete method comparison exists. The CPU cost (~4 s per pair, `EVAL_PLAN_FINAL.md`) made the local pass impractical, a concurrent VGG weight download crashed one local run (`BrokenProcessPool`), and the pass was paused. `reports/FINAL_REPORT.md` says "edit-LPIPS only for 513 pairs"; the inline Kaggle values are an additional, partial source that report does not mention (documentation gap, not a data error). The table `results/tables/edit_lpips.md` shows the partial cells (e.g. v1 InstructPix2Pix 0.40 [0.38, 0.42]; v2_cheap SD1.5 s0.55 0.35 [0.34, 0.36]) and `n/a` elsewhere. **A paper should not report edit-LPIPS comparisons.**

---

## 17. Experiment catalogue

Experiment definitions come from `docs/EVAL_PLAN_FINAL.md` section C and the table builders in `scripts/make_tables.py`. Folder names are under `results/`; session sub-folders are `results/<S>/<S>/...`. "Wall h" is `kaggle_session.json: session_seconds`/3600 on a 2xT4 session (`n_gpus 2`); GPU-hours are about twice that. GPU-time cost per image is only recorded in `docs/COST_MEASURED.md` (probe sessions) and in the per-run `gpu_time.json`/`protect.csv`.

### 17.1 The experiments E0-E10

| exp | question | methods x cleanups x editors | split / n | sessions | status | tables using it |
|---|---|---|---|---|---|---|
| E0 | reference edits of clean photos; PSR thresholds | `clean` x {none; 12 cleanups on SD1.5} x all 5 editors | test 100 (+ messy 20) | S07 (none, messy, thresholds), S10 (clean cleanup tier) | RUN | `e0`, thresholds json, `final_readout` |
| E1 | how much is just noise? | `noise_uniform`, `noise_gauss_psnr` x none x SD1.5 (3 strengths), ip2p, inpaint, FaceID | test 100 + messy 20 | S22 | RUN | `e1`, `paired_vs_noise`, `headline*` |
| E2 | vs existing tools | `photoguard`, `mist` x {none; 5 cleanups} | test 100 + messy 20 (+ val 20 screening) | S13 (PhotoGuard), S21 (Mist), S24 (val) | RUN (re-implementations at our eps) | `e2`, `final_readout` |
| E3 | does cleanup break v1? | `v1` x 12 cleanups (+FaceID, ip2p subsets) | test 100 | S08 (none, messy; its cleanup tier was empty), S19 (cleanup tier) | RUN | `e3`, `gate_E3`, `final_readout` |
| E4 | cheap cleanups in the loop | `v2_cheap` x 12 cleanups | test 100 | S10 | RUN | `e4`, `gate_E4` |
| E5 | one-step purifier + denoiser loss | `v2_main` x 12 cleanups (+FaceID, subsets) | test 100 | S09 (none, messy; cleanup tier empty), S19 | RUN | `e5`, `gate_E5` |
| E6 | anti-purification terms | `v2_antipure` x 12 cleanups + FaceID none | test 100 | S12 | RUN | `e6`, `gate_E6` |
| E7 | which loss term/phase matters | `abl_vis_only`, `abl_den_only` (+ `v2_main` as "both"); `abl_no_sem`, `abl_no_pur` | test 100 | S11 (vis_only, den_only) | RUN for 2 of 4 ablations; **`abl_no_sem`, `abl_no_pur` NOT RUN (S14)** | `e7`, `e7_editors`, `e7_paired` |
| Final | tuned config `v2_final` | `v2_final` x H, P, M, SDXL | test 100 + messy 20 | S06 (tune + protect test), S16 (edits + protect messy) | RUN | `headline*`, `e5` (info), `final_readout`, all |
| E8 | adaptive attacker | clean, v1, v2_final x `adaptive_ae` x SD1.5, FaceID | test 100 | S15 | RUN (not for target-steered variants: NOT RUN) | `e8` |
| E9 | epsilon sweep | `eps2` (S17), `eps8` (S18), eps4 = `v2_final`; `eps6` | test 100 | S17, S18; **S23 (eps6) NOT RUN** | RUN for 3 of 4 budgets | `e9` |
| E10 | human study | clean vs protected pairs | - | - | **NOT RUN** (code: `make_human_study.py`, `analyze_human_study.py` IMPLEMENTED BUT NOT RUN) | `e10` shows "not run" |
| (new) screening | target-steered variants on validation photos | clean, v1, v2_cheap, v2_main, photoguard, mist, v2_tgt_zero, v2_tgt_zero_hi, v2_tgt_gray x {none, jpeg_q75, diffpure_0.1} SD1.5 0.55 | **val 20** | S24, S25 | RUN | `FINAL_REPORT` 7.9 table 1 |
| (new) TGT | target-steered family on test | `v2_tgt_zero`, `_mid`, `_hi` x H cells + 7 cleanups | test 100 | S26 (hi), S27 (zero), S28 (mid) | RUN | `final_readout`, `FINAL_REPORT` 7.3, 7.9 |
| Phase 0 | reproduce v1 Phase 1 | `v1`, eps 4/255, 40 iters | 70 v1 photos (JPEG q75 inputs) | J01 (= S01) | RUN | `reports/PHASE_0.md` |

Planned-but-not-run summary: GrIDPure cleanup (checkpoint not available), `abl_no_sem`/`abl_no_pur` (S14), eps = 6 (S23), E10 human study, AP single-term ablation on validation (S07 AP jobs empty; S20 rerun absent), J01b/S02 (second-person reproduction), adaptive attacker against the target-steered variants, IMPRESS and InstantID (never implemented).

### 17.2 Kaggle sessions (`kaggle/jobs.yaml`, `kaggle_session.json`)

Hash = git commit printed in the job log (`git_hash.txt`), OLD hashes from before the 2026-10-23 history rewrite (map in `reports/HASH_MAP.md`). "SMOKE" = throw-away validation: numbers never reported, and ingest/finalize skip S03/S04*/S05*/J00.

| session | purpose | method(s) / split / n photos | editors, cleanups | wall h | status | hash | result folder (under `results/`) |
|---|---|---|---|---|---|---|---|
| J00 | environment probe (pip freeze, GPU, HF model hashes) | - | - | 0.003 (10 s) | RUN | - | `J00/` |
| J01 (= S01) | Phase 0: reproduce v1 Phase 1 invisibility, confirm eps | `v1`, 70 photos (v1_eval70), 40 iters | - | 0.49 (protect step 1680 s; 3189 GPU-s) | RUN | 50dd3b4 | `J01/` |
| J01b (= S02) | second-person reproduction | - | - | - | **NOT RUN** | - | none |
| S03 | build splits (ArcFace dedup), caches (CPU) | 5 lists | - | 0.86 | RUN (data-prep; produced `data/splits`; excluded from ingest) | - | `S03/` |
| S04 | **SMOKE** protection of every method, 5 photos x 5 iters | all methods | - | 0.49 | FAILED (step `protect_prot_photoguard_test`) | caac5a1 | `S04/` |
| S04t | timing probe: 4 photos x 10 iters, slow methods + PhotoGuard + Mist | v2_cheap, v2_main, v2_antipure, v2_final, photoguard, mist | - | 0.33 | RUN as probe (cost only; `COST_MEASURED.md`) | 8f535c1 | `S04t/` |
| S04u | fp16-autocast VAE speed/quality probe (4 photos x 10 iters) | v2_main fp32 vs autocast, v2_cheap, v2_final | - | 0.22 | partly FAILED (the no-checkpointing run, expected out-of-memory); cost numbers used | 7c55cb6 | `S04u/` |
| S05, S05b, S05c | **SMOKE** edit/cleanup pipeline, all editors and cleanups, GrIDPure attempt, LPIPS/threshold stages (5 photos, 1 seed) | noise_uniform etc. | all editors | 0.08 / 0.11 / 0.49 | FAILED steps (`smoke_tiers`, ...); bugs fixed afterwards; S05c's early commit contained 84 face JPEGs (later removed from history) | caac5a1 / 7c55cb6 / 58d6c2c | `S05/`, `S05b/`, `S05c/` |
| S06 | Optuna (16 trials, 10 val photos, 20 iters) + confirm (top 3 + hand-set, 20 val photos) + protect `v2_final` on test (+ messy, incomplete) | `v2_final`, val 20 / test 100 | proxy: SD1.5 0.55 + FaceID; none/jpeg_q75/diffpure_0.1 | no `kaggle_session.json` (hit the 12 h limit); log spans ~11.9 h: tune 8,855 s, confirm 15,674 s, protect test 18,185 s | RUN but **timed out** during messy protection (messy redone in S16) | 64b0815 | `S06/` (+ `optuna/`) |
| S07 | single-term antipure ablation on val (protect only), E0 clean edits H, M, thresholds | `clean`; `v2_antipure` val x3 (hf-only, mono-only, both) | all editors; AP eval empty | 4.95 | RUN; AP and clean-P eval jobs expanded to zero jobs (bug) | 9e1128d | `S07/` |
| S08 | protect `v1` (test+messy), E3 edits H, M | `v1`, test 100 + messy 20 | H, M cells | 2.98 | RUN; cleanup tier empty (bug) | 9e1128d | `S08/` |
| S09 | protect `v2_main`, E5 edits H, M | `v2_main`, test 100 + messy 20 | H, M cells | 5.95 | RUN; cleanup tier empty (bug) | 9e1128d | `S09/` |
| S10 | protect `v2_cheap`, E4 edits (H, P) + E0 clean cleanup tier | `v2_cheap`, `clean`, test 100 | H, P cells | 7.51 | RUN | b72e5ff | `S10/` |
| S11 | protect `abl_vis_only`, `abl_den_only`; tier A | test 100 | A cells | 7.58 | RUN | b72e5ff | `S11/` |
| S12 | protect `v2_antipure`; E6 edits (tier P incl. FaceID none) | test 100 | P cells | 6.10 | RUN | b72e5ff | `S12/` |
| S13 | protect PhotoGuard (test + messy), E2 edits H, M, P | `photoguard`, test 100 + messy 20 | H, M, P (5 cleanups) | 6.99 (first launch accidentally on CPU, relaunched on GPU) | RUN | b72e5ff | `S13/` |
| S14 | protect `abl_no_sem`, `abl_no_pur`; tier A | test 100 | - | - | **NOT RUN** | - | none |
| S15 | E8: protect AE pool (`v2_final`, K=2) + v1 test, train AE (60 min), edits | clean, v1, v2_final; ae_pool 120, test 100 | SD1.5, FaceID x `adaptive_ae` | 5.81 | RUN | b72e5ff | `S15/` |
| S16 | `v2_final` edits from S06 protected photos: H (4 editors), SDXL, P, M; re-protect messy with tuned config | `v2_final`, test 100 + messy 20 | H, P, M, SDXL | 5.63 (an earlier failed attempt is under `results/_to_delete/`) | RUN (second attempt) | b72e5ff | `S16/` |
| S17 | E9 eps = 2/255 | `eps2`, test 100 | E9 cells | 6.23 | RUN | b72e5ff | `S17/` |
| S18 | E9 eps = 8/255 | `eps8`, test 100 | E9 cells | 5.91 | RUN | b72e5ff | `S18/` |
| S19 | cleanup tier (P) for v1 and v2_main from S08/S09 protected photos (rerun of empty jobs) | `v1`, `v2_main`, test 100 | P cells | 5.46 | RUN | b72e5ff | `S19/` |
| S20 | val antipure single-term ablation from S07 photos (rerun) | `v2_antipure`, val 20 | AP | - | **NOT RUN** (no result folder; no val `v2_antipure` rows in `coverage.csv`) | - | none |
| S21 | protect Mist (test + messy), E2 edits | `mist`, test 100 + messy 20 | H, M, P (5 cleanups) | 3.30 (first launch on CPU, relaunched) | RUN | b72e5ff | `S21/` |
| S22 | noise controls, E1 edits | `noise_uniform`, `noise_gauss_psnr`, test 100 + messy 20 | H, M | 2.72 | RUN | b72e5ff | `S22/` |
| S23 | E9 eps = 6/255 | `eps6` | E9 | - | **NOT RUN** | - | none |
| S24 | screening of reference methods on **val 20** | clean, v1, v2_main, photoguard, mist | SD1.5 0.55 x {none, jpeg_q75, diffpure_0.1} | 2.08 | RUN | 801245d | `S24/` |
| S25 | screening of target-steered variants vs control on **val 20** | v2_cheap (control), v2_tgt_zero, v2_tgt_zero_hi, v2_tgt_gray | same | 1.90 | RUN | 801245d | `S25/` |
| S26 | `v2_tgt_zero_hi` on test | test 100 | TGT cells | 3.32 | RUN | 0b36407 | `S26/` |
| S27 | `v2_tgt_zero` on test | test 100 | TGT cells | 3.26 | RUN | 0b36407 | `S27/` |
| S28 | `v2_tgt_zero_mid` on test | test 100 | TGT cells | 3.24 | RUN | 0b36407 | `S28/` |

Wall-clock sum of the sessions with a `kaggle_session.json` is 94.0 h (91.42 h excluding J00/S03/S04*/S05*); S06 adds about 11.9 h. Failed/relaunched sessions that left no `kaggle_session.json` in this repo (earlier S15, S16, S13, S21 launches) are not included: their time is UNKNOWN / not recorded. The combined result set (`results/combined/`): 76,980 edit rows (243 cells) and 2,150 protection rows, ingested from all non-smoke sessions by `scripts/ingest_results.py`. The target-steered `v2_tgt_zero_mid` was *declared* as an interpolation between `v2_tgt_zero` (alpha 0.27) and `_hi` (alpha 1.0) and was **not screened on validation photos** (comment in `protect.yaml`); its selection as the "recommended" variant was made after seeing test results (post hoc among three declared variants).

---

## 18. Hyperparameter tuning (Optuna, S06)

Status: IMPLEMENTED AND RUN (S06, Kaggle 2xT4, hash 64b0815, 2026-10-14 23:54 to 2026-10-15 06:43 for tune + confirm). Files: `scripts/tune_optuna.py`, `config/parts/final.yaml`, `reports/optuna_trials_S06.csv`, `reports/optuna_confirm_S06.json` (also `results/S06/S06/optuna/{trials.csv,confirm.json,final_candidate.yaml}` and the SQLite study `v2_final.db`), figure `results/figures/optuna_history.png`.

### 18.1 Set-up (what actually ran: `results/S06/S06/logs/job.log`)
- **Study:** `optuna` 5.0.0, name `v2_final`, SQLite storage, **TPE sampler** `TPESampler(seed=2026, n_startup_trials=5, constant_liar=True, constraints_func=...)`, direction **maximise**. Two worker processes (one per GPU) shared the study; **16 trials** total (global budget counts complete + running + waiting).
- **Tuning set:** the **first 10 photos of `data/splits/val.txt`** (file order) for the search ("proxy set"); the **20 photos of `val.txt`** for the confirm stage. Test photos are never touched. (The 10 search photos are a subset of the 20 confirm photos, so the confirm is not a held-out re-test of the search.)
- **Per-trial cost reductions ("proxy"):** 20 PGD iterations (final protection uses 40), K = 2 augmentation draws (final uses 4), **1 edit seed**, first prompt only ("a person in a suit"), editors SD1.5 img2img at strength 0.55 + IP-Adapter FaceID, cleanups {none, jpeg_q75, diffpure_t0.1}.
- **Search space** (`suggest`): `alpha` log-uniform [0.05, 1.0]; `eta` log-uniform [0.05, 1.0]; `beta` uniform [0.0, 1.5] (0 switches `L_sem` off); `gamma` log-uniform [20, 150]; `antipure` categorical {False, True}; if True, one shared weight `ap_w` uniform [0.1, 1.0] used for both `w_hf` and `w_mono` (a project choice). **Not searched:** eps (fixed by the budget), `lambda_id` (kept 0), K, iterations. The hand-set default (alpha 0.27, eta 0.27, beta 0.90, gamma 60.5, antipure False) was enqueued as trial 0.
- **Objective:** mean PSR over (image, editor, cleanup) cells = `score_protected(...)["psr"]` (an edit is a hit if no face or similarity < threshold). With 10 photos x 2 editors x 3 cleanups x 1 seed = 60 edits per trial, PSR can only take multiples of 1/60 = 0.0167. In the confirm stage 20 photos give 120 edits, multiples of 1/120.
- **Constraints** (Optuna `constraints_func`): `[38 - mean PSNR, 0.95 - mean SSIM] <= 0`, where the means are over the trial's 10 photos, from the protector's own PSNR/SSIM on the uint8-rounded images. **They are constraints on the average, not per photo** (see 18.5).
- **PSR threshold inside tuning:** the S06 command lines contain **no `--psr-threshold`** (verified in `job.log`), so `score_protected` derived the threshold from the impostor distribution of the clean edits of the *same* 10 (or 20) validation photos and the two editors pooled (`threshold=None` branch). The threshold actually used is UNKNOWN / not recorded (not stored in the trial attributes), and it differs from the test-time per-editor thresholds of Section 15.5. The code docstring itself calls this "noisy for small n".
- **Confirm stage** (`--confirm-top 3 --confirm-images 20 --confirm-iters 20`): re-run the top-3 *feasible* trials by objective and the hand-set default on all 20 validation photos, 20 iterations; `--confirm-k` was not given, so K came from the resolved `v2_final` config (4 per `config/parts/protect.yaml`; derived from the code, not printed in the log). A candidate is "feasible" if the mean PSNR >= 38 and mean SSIM >= 0.95; the best feasible candidate by confirm PSR is written to `final_candidate.yaml`.
- **Runtime:** tune step 8,855 s (about 2.5 h, 2 workers), confirm 15,674 s (about 4.35 h; the plan had assumed far less), `v2_final` protection of the test photos 18,185 s. The session was cut at its 12 h limit while protecting messy photos.

### 18.2 All 16 trials (`reports/optuna_trials_S06.csv`; all state COMPLETE, all feasible; PSR = objective on the 10-photo proxy set)

| trial | worker | PSR | alpha | eta | beta | gamma | antipure | ap_w | PSNR mean | SSIM mean | PSNR min | SSIM min | protect s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (hand-set) | w0 | 0.03333 | 0.27 | 0.27 | 0.9 | 60.5 | False | - | 42.848 | 0.97753 | 41.849 | 0.96565 | 853.6 |
| 1 | w1 | 0.06667 | 0.09646 | 0.17231 | 1.46495 | 23.9234 | True | 0.27833 | 42.222 | 0.97419 | 41.448 | 0.96260 | 1004.2 |
| 2 | w0 | 0.05000 | 0.09646 | 0.17231 | 1.46495 | 23.9234 | True | 0.27833 | 42.309 | 0.97497 | 41.593 | 0.96454 | 928.0 |
| 3 | w1 | 0.03333 | 0.76761 | 0.26624 | 1.17484 | 36.7888 | False | - | 40.645 | 0.95980 | 40.163 | 0.94507 | 792.5 |
| 4 | w0 | 0.06667 | 0.76761 | 0.26624 | 1.17484 | 36.7888 | False | - | 40.675 | 0.96014 | 40.171 | 0.94604 | 783.0 |
| 5 | w1 | 0.08333 | 0.72812 | 0.05776 | 0.66760 | 62.0471 | True | 0.93103 | 40.987 | 0.96312 | 40.528 | 0.94938 | 940.3 |
| 6 | w0 | 0.06667 | 0.28310 | 0.36344 | 0.18674 | 33.5848 | True | 0.10273 | 41.499 | 0.96806 | 40.795 | 0.95392 | 928.9 |
| 7 | w1 | 0.01667 | 0.19797 | 0.12674 | 0.26093 | 109.3732 | True | 0.99349 | 44.006 | 0.98324 | 42.964 | 0.97613 | 940.3 |
| 8 | w0 | 0.06667 | 0.57880 | 0.73487 | 0.90453 | 71.1852 | True | 0.99819 | 41.476 | 0.96774 | 40.909 | 0.95455 | 930.3 |
| 9 | w1 | 0.06667 | 0.10449 | 0.05032 | 0.74321 | 23.1705 | True | 0.67522 | 41.903 | 0.97144 | 40.825 | 0.95858 | 939.1 |
| 10 | w0 | **0.10000** | 0.92253 | 0.05998 | 0.28191 | 56.5016 | False | - | 40.906 | 0.96248 | 40.363 | 0.94812 | 781.6 |
| 11 | w1 | 0.08333 | 0.48438 | 0.09595 | 1.23735 | 74.2281 | True | 0.65783 | 41.809 | 0.97046 | 41.139 | 0.95693 | 940.3 |
| 12 | w0 | 0.05000 | 0.79884 | 0.06025 | 0.18155 | 57.7950 | False | - | 41.110 | 0.96443 | 40.440 | 0.94972 | 781.4 |
| 13 | w1 | 0.08333 | 0.81494 | 0.08964 | 0.57308 | 39.3281 | True | 0.61111 | 40.431 | 0.95718 | 40.132 | 0.94337 | 940.8 |
| 14 | w0 | 0.03333 | 0.50347 | 0.05699 | 0.54517 | 41.3134 | False | - | 41.279 | 0.96602 | 40.605 | 0.95124 | 781.3 |
| 15 | w1 | 0.05000 | 0.93349 | 0.11668 | 0.50180 | 86.7050 | False | - | 41.425 | 0.96740 | 40.765 | 0.95450 | 791.2 |

Per-cell PSR of every trial is in the CSV column `user_attrs_per` (e.g. trial 10: none|SD1.5 0.1, diffpure_0.1|SD1.5 0.1, all others 0.0; trial 5: none|SD1.5 0.3, diffpure_0.1|SD1.5 0.2, others 0.0). **In all 16 trials the FaceID cells and the jpeg_q75 cells are 0 or near 0 (FaceID exactly 0.0 in every trial), so the objective was effectively driven by SD1.5 under "none" and "diffpure_0.1".**

Observations (derived directly from the table): (i) trials 1/2 and 3/4 have **identical parameters** (the two workers each seeded their sampler with 2026 and so drew the same startup points; this explanation is an inference, not verified) and give PSR 0.0667 vs 0.0500 and 0.0333 vs 0.0667: a direct measure of evaluation noise, which is as large as the spread between most trials; (ii) PSR ranges 0.0167 to 0.1000 on 60 edits per trial, i.e. 1 to 6 hits; (iii) the best search trial (10, PSR 0.10) has antipure off.

### 18.3 Confirm results (`reports/optuna_confirm_S06.json`; 20 validation photos, 120 edits per candidate, 20 iterations)

| candidate | confirm PSR | none/SD1.5 | jpeg_q75/SD1.5 | diffpure_0.1/SD1.5 | any FaceID cell | PSNR mean (min) | SSIM mean (min) | protect s | antipure | alpha | eta | beta | gamma | ap_w |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trial_10 | 0.0500 | 0.15 | 0.00 | 0.15 | 0.0 | 40.508 (40.184) | 0.95460 (0.92231) | 3186.9 | False | 0.92253 | 0.05998 | 0.28191 | 56.5016 | - |
| **trial_5** | **0.06667** | 0.20 | 0.00 | 0.20 | 0.0 | 40.673 (39.762) | 0.95711 (0.92371) | 3854.6 | True | 0.72812 | 0.05776 | 0.66760 | 62.0471 | 0.93103 |
| trial_11 | 0.0500 | 0.15 | 0.00 | 0.15 | 0.0 | 41.371 (40.124) | 0.96499 (0.94096) | 3855.3 | True | 0.48438 | 0.09595 | 1.23735 | 74.2281 | 0.65783 |
| hand_set | 0.04167 | 0.15 | 0.05 | 0.05 | 0.0 | 42.151 (41.502) | 0.97168 (0.95831) | 3182.0 | False | 0.27 | 0.27 | 0.90 | 60.5 | - |

- **Selected:** trial_5 (best confirm PSR 0.06667 = 8 hits of 120) -> `config/parts/final.yaml` header: "candidate for config/parts/final.yaml (trial_5; confirmed on 20 val images; hand-set psr=0.0417)". Final tuned values (`methods.v2_final.loss`): `alpha 0.7281249466629797, eta 0.05776266977639212, beta 0.6676014641453782, gamma 62.04711204188804, lambda_id 0.0`, `anti_purify: {enabled: true, w_hf: 0.9310312738392569, w_mono: 0.9310312738392569}`. (The `v2_final` preset additionally has the SDXL-VAE ensemble enabled in `v2_final_default`.)
- Tuned vs hand-set: 8 vs 5 hits of 120 edits. No CI or paired test was computed for this pair; given the trial-to-trial noise in 18.2 they are **not separable** (`FINAL_REPORT.md` section 10 says the same).
- The proxy-best trial 10 (0.10) fell to 0.05 on the confirm set: a regression to the mean consistent with selection among noisy objective values.

### 18.4 Caveats on the tuning (for the paper's limitations)
1. **Proxy objective, tiny sample:** 16 trials x 10 photos x 1 seed x 1 prompt, 20 PGD iterations instead of 40 and K=2 instead of 4; PSR on 60 edits is quantised to 1/60.
2. **The objective is not what is reported:** the final method runs 40 iterations with K=4, evaluated on 100 test photos with 3 seeds and 3 prompts at the per-editor test thresholds. Tuning transfer was never checked beyond the 20-photo confirm.
3. **Different, small-sample PSR threshold** than at test time (Section 18.1); actual value not recorded.
4. **Confirm set overlaps the search set** (the 10 search photos are in the 20 confirm photos).
5. **Invisibility constraint checked on averages only:** the feasible set was defined by mean PSNR >= 38 and mean SSIM >= 0.95 over 10 photos. The confirm run already shows the selected candidate has a per-photo minimum SSIM of 0.9237 (< 0.95), and on the 100 test photos `v2_final` violates SSIM >= 0.95 on **29/100** photos (`FINAL_REPORT.md` 7.1; mean PSNR 40.46, min 38.76, mean SSIM 0.9549, min 0.9197). The hand-set default has min SSIM 0.9583 in the confirm run. So the Optuna procedure selected a configuration that fails the project's own per-photo invisibility rule.
6. **FaceID never contributed:** PSR 0.0 in every trial; the search therefore tuned for SD1.5 and diffpure_0.1 only.
7. **Not tuned:** eps, iterations, K, `lambda_id`, the target-steered family (its three variants were screened on validation photos without Optuna, S24/S25).
8. Study resumability: `RESUME_FROM`/`prepare_optuna_dir` restore an earlier study; a heartbeat/retry callback was configured (`RetryFailedTrialCallback(max_retry=1)`, deprecated in Optuna 4.9); a creation-race between the two workers' SQLite initialisation was patched with retries after S06 found it.

---

## 19. Compute, cost and reproducibility

### 19.1 Pipeline (IMPLEMENTED AND RUN)
- **No local GPU.** All GPU work ran on **Kaggle notebooks (2x Tesla T4)**, internet on, from a single private code bundle. The project used about five free Kaggle accounts (per `docs/LEARNING_HANDOVER.md`: "5 accounts with about 30 GPU-hours/week each"; quota reported as billed by wall-clock, `docs/KAGGLE_LAUNCH.md`). Session cap about 12 h (S06 was cut at it).
- **Bundle:** `python kaggle/build_bundle.py` zips `src, scripts, config, tests, requirements.d, data/splits, data/v1_ref, requirements*.txt, pytest.ini, kaggle/{jobs.yaml,README.md,job_runner.ipynb}` plus `BUNDLE_INFO.json` (git hash) into `kaggle/dist/diffshield-code-<hash7>.zip`. It never includes images, results, `.git`, weights or archives; it refuses a dirty tree unless `--allow_dirty`. Rule adopted after an incident: tests must pass before building a bundle (a bundle built before reading a failing test was rebuilt). The zip is uploaded as a private Kaggle dataset (`diffshield-code`, versions v3 to v7 appear in the docs).
- **Notebook:** `kaggle/job_runner.ipynb`; the human edits one cell: `JOB_IDS` (list, run in order), `PARAMS` (dict -> `--set k=v`; e.g. `{"smoke": true}` for 5 photos x 5 iterations x 1 seed, or `{"prev_root": "/kaggle/input/<S06 output>"}`), `RESUME_FROM`, `JOB_SHARD` ("0/2" = half of every job on this account), `FORCE`. It calls `python scripts/run_job.py --jobs <ids> --work /kaggle/working [...]`.
- **Job runner** (`scripts/run_job.py`, `kaggle/jobs.yaml`): a job is a list of steps (macros `setup`, `prep`, `protect`, `evaluate`); `sharded: true` steps run one process per GPU (`CUDA_VISIBLE_DEVICES`, `--shard K/N`); `workers: true` for Optuna; every step logs `rc` and seconds; a `DONE` marker and `kaggle_session.json` (`ok`, `failed_steps`, wall seconds, params) are written; outputs are zipped as `<JOB>_outputs.zip`.
- **Resumability:** protection checkpoints every 5 photos; evaluation writes one CSV row per (method, image, seed, editor, prompt, strength, cleanup) and skips rows already present (`ResultsWriter.done`); a dead/timed-out session is relaunched with its output added as an input and `RESUME_FROM` set; finished jobs (DONE) are skipped unless `FORCE`.
- **Failure isolation:** one failing editor does not abort the others (`failed_editors.json`); unavailable cleanups (GrIDPure) are skipped and logged, not silently dropped; `soft`/`optional` steps do not fail the job.

### 19.2 Versions and environment
`requirements.txt` is pinned to the Kaggle image recorded by J00 (2026-10-14; image `gcr.io/kaggle-gpu-images/python@sha256:2757e0c7...`, Python 3.13.15, CUDA 12.8, cuDNN 9.19, 2x T4): `torch 2.11.0`, `torchvision 0.26.0`, `numpy 2.1.3`, `diffusers 0.40.0`, `transformers 5.16.1`, `accelerate 1.14.0`, `huggingface_hub 1.29.0`, `kornia 0.8.3`, `lpips 0.1.4`, `scikit-image 0.25.2`, `scikit-learn 1.6.1`, `optuna 5.0.0`, eval extras `insightface 0.7.3`, `onnxruntime-gpu 1.22.0` (the unpinned latest wheel needed CUDA 13 and fell back to CPU), `opencv-python-headless 4.14.0.94`. `scripts/check_env.py` flags drift. FaceID loading needed removing a preinstalled `torchao` (project history). Every run folder holds `config.json`, `git_hash.txt`, `versions.json`, `env.json`, `seed.txt`, `command*.txt` (`src/utils/runlog.py`), so each result is traceable to code and library versions.

### 19.3 Costs actually recorded
From `docs/COST_MEASURED.md` (probe sessions S04t/S04u/J01 on T4; 4-70 photos, 512 px, K=4; projected to 40 iterations with fp16-autocast VAE): v1 46 s/photo, v2_cheap 146 s, v2_main 263 s, v2_final 395 s, v2_antipure ~433 s (estimated, not measured with autocast), PhotoGuard (1000 iterations) ~270 s, Mist 55 s; the plan had underestimated v2 cost by 2-6x. fp16 autocast gave a 2.05x speed-up for `v2_main` (135.1 -> 65.7 s/photo) with PSNR within 0.03 dB and the same trajectory of `L_vis` on 4 photos x 10 iterations (does not prove final protection is identical). Without gradient checkpointing `v2_main` runs out of memory on a 15 GB T4.

Measured in the real runs (my own aggregation of `gpu_seconds` in `results/combined/protect.csv`, sum of per-photo GPU seconds): per-photo mean protection seconds and total GPU-hours: v1 ~47-48 s (test: 48.3; 1.34 h), v2_cheap 115.7 s (3.21 h), v2_tgt_zero 117.6 s / `_mid` 116.0 s / `_hi` 119.8 s (~3.2-3.3 h each), abl_vis_only 169.6 s, abl_den_only 162.1 s, v2_main 259.2 s (7.20 h), v2_antipure 313.5 s (8.71 h), v2_final 359.9 s (10.0 h, test), eps2 391.7 s, eps8 384.4 s (about 10.7-10.9 h each), PhotoGuard 232.3 s (6.45 h), Mist 57.7 s (1.6 h), noise controls ~0, `v2_final` on the 120-photo AE pool at K=2 193.8 s (6.46 h). **Total protection about 103 GPU-hours (2,150 protection rows).** Edit time: the sum of `gpu_seconds_edit` over all 76,980 combined edit rows is about 70.2 h (SD1.5 42.2 h, FaceID 9.8 h, InstructPix2Pix 8.5 h, SDXL 5.3 h, inpainting 4.4 h) and cleanup time about 2.6 h; these are per-shard process clocks. Wall-clock of sessions: Section 17.2 (94.0 h recorded + about 11.9 h for S06).

### 19.4 How to rerun (commands from `README.md`, `kaggle/README.md`, `docs/KAGGLE_LAUNCH.md`)
1. **Tests (CPU):** `python -m pytest -q tests` (377 pass, 8 skipped per `FINAL_REPORT.md`; fake/tiny models, no weights).
2. **Build and upload bundle:** `python kaggle/build_bundle.py` -> upload `kaggle/dist/diffshield-code-<hash>.zip` as a private Kaggle dataset; in a notebook import `kaggle/job_runner.ipynb`, attach the code dataset and `ipythonx/celebamaskhq` (and, where the launch table says so, an earlier session's output), accelerator GPU T4 x2, internet on.
3. **Run sessions:** set `JOB_IDS = ["S06"]` (etc.) and Run All / Save & Run All. Preview locally: `python scripts/run_job.py --jobs J01 --dry_run --work tmp`. Smoke first: `PARAMS = {"smoke": true}`. Order/inputs per session: `docs/KAGGLE_LAUNCH.md`, `docs/EVAL_PLAN_FINAL.md` section G (note that several rows of `KAGGLE_LAUNCH.md` were superseded by the later fixes S19/S20 and the S24-S28 additions in `jobs.yaml`, which is the source of truth).
4. **Ingest and analyse locally** (one command): `python scripts/finalize_all.py` (unpack new `S<n>_outputs.zip` from Downloads, then `ingest_results.py`, `make_tables.py`, `readout_final.py`, `make_figures.py`); then `python scripts/make_pareto_figure.py`. Step by step: `python scripts/ingest_results.py --results results`; `python scripts/readout_final.py --split test`; `python scripts/make_tables.py --combined results/combined --out results/tables`; `python scripts/make_figures.py --combined results/combined --out results/figures`; optional `python scripts/compute_edit_lpips.py --combined results/combined --workers 4`.
5. **Single protected photo demo:** `python scripts/demo.py --image photo.jpg --out demo_out` (`--cpu-dry-run` checks wiring only).
6. **Re-derive splits:** `python scripts/make_splits.py --celebahq_root <CelebAMask-HQ> --out_dir <dir> --identity_mode auto` (will refuse to overwrite the frozen lists in `data/splits` without `--force`).

### 19.5 Reproducibility properties and limits
- **Determinism:** all seeds are sha256-derived (`derive_seed`, never Python `hash()`): protection `(2026, "protect", image_id)`, edits `(2026, "edit", image_id, k)`, cleanups `(2026, "purify", image_id, name)`, bootstrap seed 2026, split seed 2026, Optuna sampler seed 2026. GPU kernels are non-deterministic, so rerun results are statistical, not bit-exact (Phase 0 reproduced v1 within PSNR +0.075 dB / SSIM +0.0004 / LPIPS -0.0002 of its stored numbers; accepted tolerance ±0.5 dB / ±0.005 / ±0.03; v1 had used one global seed 42, v2 uses per-image seeds). The combine step recorded **0 conflicts** among 180 duplicate edit keys (a conflict is a duplicate key whose ArcFace similarity differs by more than 1e-4), i.e. the duplicated rows agreed.
- **Hashes:** each Kaggle log prints the git hash of the bundle it ran: `9e1128d` (S07-S09), `64b0815` (S06), `b72e5ff` (S10-S13, S15-S19, S21, S22), `801245d` (S24, S25), `0b36407` (S26-S28), `50dd3b4` (J01). These are **pre-rewrite** hashes; the history was rewritten on 2026-10-23 to remove 84 face-derived smoke-test JPEGs and the old-to-new map is in `reports/HASH_MAP.md` (several early hashes map to themselves, later ones change). Code, configs and result CSVs are otherwise identical.
- **Not pinned:** Hugging Face model revisions (`revision: null`), the original CelebAMask-HQ archive checksum, the insightface `buffalo_l` download version, and the official `MIST.png` content hash (the commit is pinned in the URL).
- **Sessions needing earlier outputs:** S15/S16 read the S06 protected photos (`prev_root`); S19 reads S08/S09; S20 would read S07. Folder layouts are in `jobs.yaml`.
- **Known data-hygiene items** relevant to anyone redistributing the repo: protected/edited images are never committed; 84 early smoke JPEGs were purged by history rewrite; the two face-containing figure PDFs on the public `master` branch (Section 14.1) are an open issue.

### 19.6 Open uncertainties noted in this part
1. Exact dataset acquisition/licence text for CelebA-HQ and a checksum: not in the repo.
2. The Optuna in-trial PSR threshold value: not recorded.
3. Whether the 38 missed test faces are all among the 70 v1 photos is consistent with the code (new photos are rejected without a face) but not recorded as a breakdown.
4. Per-photo fully-protected share: definition inferred; no script found.
5. Edit-LPIPS: report says "513 pairs", the cache file holds 512 pairs (plus 22,560 inline Kaggle values for partial method sets).
6. Bootstrap resamples: 10,000 (code, tables) vs 2,000 (`readout_final.py`, plan text); the report's main numbers mix both sources.
7. GPU cost of failed/relaunched sessions and the true S06 end time: not recorded.


---

## 20. Complete numerical results

**Scope and conventions of this part.** Every number below is copied from a file in the repository `C:\Users\karth\Documents\DiffShield-wt\integration` (branch v2/release): the generated tables in `results/tables/`, `reports/FINAL_REPORT.md`, `reports/REPORT_v2_draft.md`, `reports/PHASE_0.md`, `reports/optuna_*`, `results/combined/psr_thresholds.json`, and the Adaptive-AE training log. Paths are relative to that repository. Where a number comes from somewhere else (the one ad hoc audit script run during the session), that is said explicitly. Nothing is interpreted here; interpretation belongs to other parts. A neutral "how to read" line is given for each table.

**Global facts that apply to every generated table unless stated otherwise (they are the footnote printed under each table).**
- Test split = 100 CelebA-HQ photos (includes all 70 of v1's evaluation photos). Messy split = 20 photos. Validation split = 20 photos (all tuning and screening). AE pool = 120 photos (adaptive-attacker training only). Unit of analysis = the photo.
- Edits: 3 edit seeds per photo (seeds 1,2,3), prompts rotated per photo; saved as JPEG q90. Seeds and prompts are averaged within each photo first.
- Interval = 95% percentile-bootstrap over photos, 10000 resamples, seed 2026 (this is the footnote in `results/tables/*.md`). Note: `reports/REPORT_v2_draft.md` section 4 says "2,000 resamples"; the tables say 10000. The tables are the later, generated source.
- Paired comparisons A-B: mean per-image difference with paired image-bootstrap CI and two-sided Wilcoxon signed-rank p, **not corrected for multiple comparisons**. PSR A-B > 0 means A protects more; sim A-B < 0 means A removes more identity.
- PSR (protection success rate) = share of edits with no detected face, or ArcFace cosine similarity to the source below a per-editor threshold. Threshold = 95th percentile of impostor (different-person) similarity measured on E0 clean edits.
- Default setting: SD1.5 img2img, strength 0.55, no cleanup ("none"), test split.
- Row labels in the older generated tables map as: "v2 cheap purifiers (E4)" = `v2_cheap`; "v2 + one-step SD purifier + L_den (E5)" = `v2_main`; "v2 + anti-purification (E6)" = `v2_antipure` (only in some cells); "v2 final" = `v2_final`; "uniform noise (E1)" = `noise_uniform`; "Gaussian noise, PSNR-matched (E1)" = `noise_gauss_psnr`; "PhotoGuard encoder attack (E2, reimpl.)" = `photoguard`; "Mist-style (E2, reimpl.)" = `mist`; "clean (E0)" = `clean`. The `v2_tgt_*` methods (target-steered) appear only in `final_readout.md`, `invisibility_methods.md`, `gates.md`, `coverage.md`, and FINAL_REPORT 7.x, not in the older headline tables.
- The 13 held-out cleanups are: jpeg_q90, jpeg_q75, jpeg_q50, jpeg_other, downup_0.5, downup_other, vae_rt, diffpure_0.05, diffpure_0.1, diffpure_0.2, diffpure_0.3, restore_swin2sr, gridpure. **gridpure was never run** (model file unavailable; the column reads "not run" everywhere), so 12 cleanups have data. `adaptive_ae` is a separate (adaptive) mode, not one of the 13.
- Coverage fact needed to read the cleanup columns (from `results/tables/coverage.md`, see 20.0 below): for SD1.5 s0.55 the "none" cell has 900 edit rows per method (100 photos x 3 seeds x 3 prompts); the cleanup cells have 300 rows (100 photos x 3 seeds x 1 rotated prompt). The footers of E4/E5/E6 print "prompts per image: up to 1", the other tables print "up to 3". So within one row of a headline table the "none" column and the cleanup columns can be based on different numbers of edits per photo (inferred from the row counts; the project files do not comment on it).

**Files NOT reported, and why.**
- Smoke-test numbers (sessions S03, S04, S04t/u, S05, S05b, S05c; also `results/demo_dry`, `results/tables_partial`): they were a throw-away 5-photo validation of the pipeline and are excluded by the project's rule; the result combiner had once mixed them in and was fixed (FINAL_REPORT Appendix B, item 12). Nothing from them appears here.
- Edit-LPIPS (`results/tables/edit_lpips.*`): only 513 clean-edit/protected-edit pairs were computed (the job crashed on a concurrent VGG weight download and was too slow on CPU), so the table is mostly "n/a" and is **not reported** (FINAL_REPORT 6.2 and section 10). The few cells that do exist in `edit_lpips.md` (v2_cheap SD1.5 s0.3 0.35 [0.34,0.35]; s0.55 0.35 [0.34,0.36]; s0.75 0.32 [0.31,0.34]; InstructPix2Pix 0.38; inpaint 0.27; FaceID 0.34; v1 InstructPix2Pix 0.40 and FaceID 0.34) are listed here only so the reader knows they exist; they are partial, cover 513 pairs in total, and must not be cited.
- Human study (E10, 2AFC "which image has noise" and "same person?"): not run (`results/tables/e10.md` reads "not run" for both tasks).

---

### 20.0 Coverage map (summary of `results/tables/coverage.md`, also `results/combined/coverage.md`)

The table has one row per (split, method, purifier, editor, strengths) cell with columns n_images, n_rows, seeds, n_prompts, face_rate (share of edits in which ArcFace found a face). Summary:

- **Test split, SD1.5 img2img s0.55**: every cell has 100 images, seeds 1,2,3, 3 prompts. Cells with "none" cleanup and strengths 0.3/0.55/0.75 have 1500 rows (900 at s0.55 plus 300 each at s0.3 and s0.75) for: clean, mist, noise_gauss_psnr, noise_uniform, photoguard, v1, v2_cheap, v2_final, v2_main, v2_tgt_zero, v2_tgt_zero_hi, v2_tgt_zero_mid. All other cells have 300 rows (one rotated prompt).
- **Methods with test-split rows**: abl_den_only, abl_vis_only, clean, eps2, eps8, mist, noise_gauss_psnr, noise_uniform, photoguard, v1, v2_antipure, v2_cheap, v2_final, v2_main, v2_tgt_zero, v2_tgt_zero_hi, v2_tgt_zero_mid. **No rows** for `abl_no_sem`, `abl_no_pur`, `eps6`, `v2_tgt_gray` on test (v2_tgt_gray exists only on validation).
- **Editors with test rows**: sd15_img2img (s0.3, 0.55, 0.75), ipadapter_faceid (0.55), ip2p, sd_inpaint (strength 1), sdxl_img2img (0.55). SDXL only for clean, v1, photoguard, v2_final. InstructPix2Pix and inpaint for clean, v1, mist, photoguard, noise_*, v2_cheap, v2_main, v2_final, abl_*; not for v2_tgt_*. FaceID for clean, v1, v2_main, v2_antipure, v2_final, v2_tgt_*, mist, photoguard, noise_*, eps2, eps8, abl_*.
- **Purifiers with test rows** (SD1.5 s0.55): none, jpeg_q90, jpeg_q75, jpeg_q50, jpeg_other, downup_0.5, downup_other, vae_rt, diffpure_0.05/0.1/0.2/0.3, restore_swin2sr, adaptive_ae. Coverage differs by method (for example `v2_tgt_*` only have jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1/0.2/0.3; `mist`/`photoguard` only jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1; `v2_antipure` has no "none" on SD1.5 s0.55 in the cells listed, only FaceID "none" plus the purified SD1.5 cells; `adaptive_ae` only for clean, v1, v2_final, on SD1.5 and FaceID).
- **Messy split** (20 photos x 3 seeds x 3 prompts, 60 rows per cell): clean, mist, noise_gauss_psnr, noise_uniform, photoguard, v1, v2_final, v2_main; editors SD1.5 s0.55 and FaceID; no cleanup.
- **Validation split** (20 photos, 60 rows per cell; SD1.5 s0.55): methods clean, mist, photoguard, v1, v2_cheap, v2_main, v2_tgt_gray, v2_tgt_zero, v2_tgt_zero_hi, over purifiers none, jpeg_q75, diffpure_0.1. (`val | clean` also has these three; `mist/photoguard/v1/v2_cheap/v2_main/v2_tgt_*` as listed.)
- face_rate (share of edits with a detected face) ranges 0.71 to 1.00; the lowest values are `photoguard` on InstructPix2Pix (0.71), `photoguard` SD1.5 3-strength (0.90), `mist` SD1.5 3-strength (0.93), `clean` InstructPix2Pix (0.82), `abl_den_only` ip2p (0.85).

---

### 20.1 Reference / clean-edit statistics and PSR thresholds (E0)

**Source:** `results/tables/e0.md` (also `.csv`, `.tex`, `e0_long.csv`) and `results/combined/psr_thresholds.json` (copy `psr_thresholds.from_job.json` has the same values). **Split:** test. **n:** 100 photos per cell, 3 edit seeds, up to 3 prompts per photo. **Methods:** clean photos only (no protection, no cleanup). **CI:** yes, 95% bootstrap over photos. **What a cell is:** the editor applied to the unprotected photo; ArcFace sim = mean cosine similarity of the clean edit to the source; PSR of clean edits = share of clean edits that would be counted "protected" (a sanity check, should be small); no-face rate = share of edits where no face is detected; CLIP-I = CLIP ViT-L/14 image similarity between source and edit; PSR threshold = the per-editor threshold used everywhere else.

| editor | n images | ArcFace sim (clean edit vs source) | PSR of clean edits | no-face rate | CLIP-I | PSR threshold |
|---|---|---|---|---|---|---|
| SD1.5 s0.3 | 100 | 0.522 [0.508, 0.537] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.819 [0.806, 0.832] | 0.101 |
| SD1.5 s0.55 | 100 | 0.223 [0.208, 0.237] | 0.14 [0.10, 0.18] | 0.00 [0.00, 0.01] | 0.650 [0.638, 0.663] | 0.101 |
| SD1.5 s0.75 | 100 | 0.039 [0.029, 0.048] | 0.84 [0.79, 0.89] | 0.02 [0.00, 0.04] | 0.528 [0.513, 0.544] | 0.101 |
| SDXL s0.55 | 100 | 0.337 [0.319, 0.355] | 0.01 [0.00, 0.03] | 0.00 [0.00, 0.00] | 0.736 [0.722, 0.751] | 0.104 |
| InstructPix2Pix | 100 | 0.835 [0.799, 0.867] | 0.18 [0.12, 0.25] | 0.18 [0.12, 0.24] | 0.744 [0.712, 0.774] | 0.108 |
| SD inpaint (keep-face) | 100 | 0.959 [0.956, 0.961] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.823 [0.810, 0.836] | 0.110 |
| IP-Adapter FaceID | 100 | 0.701 [0.683, 0.718] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.752 [0.733, 0.770] | 0.108 |

**`results/combined/psr_thresholds.json` (exact values).** percentile 95.0; computed by `src.eval.analysis.thresholds_from_embeddings`; restricted to test-split image ids; n_sources_with_face = 100; n_source_embeddings_merged = 130; n_edit_embeddings_merged = 2880.

| editor | threshold (full precision) | n impostor pairs |
|---|---|---|
| sd15_img2img | 0.10054539651520272 | 147609 |
| sdxl_img2img | 0.10367028692086117 | 29700 |
| ip2p | 0.10775226496545169 | 24453 |
| sd_inpaint | 0.10970453486167729 | 29700 |
| ipadapter_faceid | 0.1078936712001695 | 29700 |
| pooled | 0.1034364047631 | 261162 |

How to read: the threshold is the similarity that only 5% of different-person pairs exceed; an edit with similarity below it counts as "no longer the same person". FINAL_REPORT 6.2 quotes the SD1.5 threshold as 0.1005; REPORT_v2_draft quotes 0.101. All three agree. The tables print the pooled threshold as 0.103 (the exact value is 0.1034).

---

### 20.2 v1 reproduction (Phase 0)

**Source:** `reports/PHASE_0.md` (which states that its numbers were copied from `results/J01/results/J01/{budget_check.md, tables/invisibility.md, compare_v1.md, summary.json, gpu_time.json}`; the per-run subfolder of J01 is present in the repository but the underlying files were not re-opened for this part). **Split:** the 70 v1 evaluation photos (`data/splits/v1_eval70.txt`). **Run:** Kaggle job J01, 2026-10-14, 1 protection seed per image, 40 PGD iterations, bundle B0 (`v2/phase-0` @ 50dd3b4), 2xT4. **CI:** 95% bootstrap over images. The second-person run J01b was **not run**, so no "reproduced by a second person" claim is allowed.

Budget finding:

| source | n images | max abs gray-level change | share of values at max | inferred eps (standard 0-255 scale) |
|---|---|---|---|---|
| v1 original run, tensors in [-1,1] (`stage1_per_image_metrics.csv`, column Linf) | 70 | 0.0313725 on [-1,1] (= 8/255) | n/a | 4/255 by conversion (Linf*255/2) |
| v1 original protected PNGs | not available | – | – | – |
| Reproduced v1 run J01, saved PNGs vs the JPEG q75 clean inputs (`budget_check.md`) | 70 | 4 (in all 70 images; no value above 4) | 9.35% of all pixel values | 4/255 |

Distribution of |gray-level change| over all pixel values (J01): 0: 11,468,029; 1: 18,754,991; 2: 11,998,407; 3: 7,683,111; 4: 5,145,702; >= 5: 0. Step size: v1 used alpha = 1/255 on [-1,1], i.e. 0.5/255 on the standard scale. v1 weights (Optuna trial 23 of v1): alpha 0.2699475469647097, beta 0.9023875567667267, gamma 60.52658429696642.

Invisibility reproduction (reference = v1's stored results, `data/v1_ref/stage1_summary_metrics.csv`):

| metric | v1 original run | reproduced, v1 convention: mean ± std [95% CI] | reproduced, published PNG: mean ± std [95% CI] | violations |
|---|---|---|---|---|
| PSNR (dB) | 41.911 ± 0.440 | 41.986 ± 0.416 [41.890, 42.082] | 42.171 ± 0.436 [42.070, 42.272] | 0 / 0 of 70 below 38 |
| SSIM (windowed) | 0.97496 ± 0.00480 | 0.9754 ± 0.0054 [0.9741, 0.9766] | 0.9756 ± 0.0054 [0.9744, 0.9769] | 0 / 0 of 70 below 0.95 |
| LPIPS VGG (clean vs protected) | 0.23156 ± 0.06304 | 0.2314 ± 0.0637 [0.2168, 0.2463] | 0.2305 ± 0.0635 [0.2160, 0.2454] | n/a |
| Linf ([-1,1]) | 8/255 | 0.0313725 = 8/255 in all 70 | n/a (PNG max change = 4/255) | n/a |

"v1 convention" = tensor to uint8 by truncation (as v1 measured); "published PNG" = rounded image that would be posted.

Paired comparison with v1's per-image CSV (pairing by file name, `compare_v1.md`), J01 minus v1:

| metric | mean J01 | mean v1 | mean difference [95% CI] | tolerance (user-accepted 2026-10-14) | within |
|---|---|---|---|---|---|
| PSNR (dB) | 41.9863 | 41.9111 | +0.0752 [+0.0021, +0.1507] | ±0.5 | yes |
| SSIM | 0.9754 | 0.9750 | +0.0004 [-0.0002, +0.0010] | ±0.005 | yes |
| LPIPS | 0.2314 | 0.2316 | -0.0002 [-0.0042, +0.0039] | ±0.03 | yes |

Gate: PASS on every item. The PSNR difference is statistically above zero but 15% of the tolerance; PHASE_0 attributes it, marked "unverified", to different random draws (per-image seeds instead of v1's global seed 42). Cost: 45.6 ± 1.2 GPU-seconds per image on a T4 (40 iterations; 1.14 s per PGD iteration), peak memory 7.2 GB, 3189 GPU-seconds (0.89 GPU-h) in total, 29.3 min wall (protect step 1680 s) for 70 images on 2xT4; the plan had estimated 25 s per image (1.8x less). Environment recorded by J00: torch 2.11.0+cu128, diffusers 0.40.0, transformers 5.16.1, kornia 0.8.3, Python 3.13.15, CUDA 12.8. CPU tests inside J01: 32 passed (the full suite later reached 377 passed, 8 skipped, per FINAL_REPORT App. A).

Cross-check: `results/tables/invisibility_methods.md` row `v1 | v1_eval70 | 70`: PSNR 42.17 [42.07, 42.27], min 41.12, SSIM 0.9756 [0.9743, 0.9769], 0 below 38 dB, 0 below 0.95, LPIPS 0.231 [0.215, 0.245]. This agrees with the "published PNG" column above. On the 100 test photos v1 has PSNR 42.42, SSIM 0.9763 (see 20.3).

v1 also had a stored LPIPS of 0.46 from its Phase 2 whose outputs were lost; FINAL_REPORT section 10 says it "can only be quoted".

---

### 20.3 Invisibility of every method

**Source:** `results/tables/invisibility_methods.md` (+`.csv`, `_long.csv` with min SSIM and CIs, `.tex`); the same data are in FINAL_REPORT 7.1, `results/tables/final_readout.md` (last block) and `results/tables/gate_INV.md`. **Split:** test unless the split column says otherwise. **What a cell is:** protected photo (published uint8 PNG convention) compared with the clean photo. Rule from the project: PSNR >= 38 dB **and** SSIM >= 0.95 on **every** photo. "below 38 dB" and "below 0.95" are **per-photo rule-violation counts out of n**. **CI:** yes (95% bootstrap over photos) for PSNR, SSIM, LPIPS. Budget eps = 4/255 unless stated. Source of values: the protection runs' `protect.csv` (not the edit rows).

| method | split | n | eps (x/255) | PSNR dB [95% CI] | min PSNR | photos below 38 dB | SSIM [95% CI] | min SSIM (from `_long.csv`) | photos below 0.95 | LPIPS(VGG) [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| abl_den_only | test | 100 | 4 | 50.76 [50.68, 50.82] | 48.79 | 0 | 0.9951 [0.9949, 0.9954] | 0.9900 | 0 | 0.024 [0.022, 0.027] |
| abl_vis_only | test | 100 | 4 | 41.73 [41.62, 41.85] | 40.32 | 0 | 0.9781 [0.9773, 0.9789] | 0.9697 | 0 | 0.242 [0.232, 0.253] |
| eps2 | test | 100 | 2 | 45.34 [45.29, 45.39] | 44.86 | 0 | 0.9836 [0.9828, 0.9843] | 0.9628 | 0 | 0.160 [0.149, 0.172] |
| eps8 | test | 100 | 8 | 35.98 [35.74, 36.20] | 33.33 | 98 | 0.8958 [0.8864, 0.9045] | 0.7724 | 96 | 0.408 [0.392, 0.424] |
| mist | messy | 20 | 4 | 38.05 [37.93, 38.17] | 37.43 | 6 | 0.9216 [0.9166, 0.9262] | n/a | 20 | 0.476 [0.455, 0.495] |
| mist | test | 100 | 4 | 38.00 [37.95, 38.05] | 37.47 | 50 | 0.9229 [0.9197, 0.9255] | 0.8275 | 100 | 0.439 [0.429, 0.448] |
| mist | val | 20 | 4 | 37.97 [37.86, 38.09] | 37.57 | 13 | 0.9206 [0.9109, 0.9272] | n/a | 20 | 0.428 [0.413, 0.444] |
| noise_gauss_psnr | messy | 20 | 4 | 41.91 [41.91, 41.92] | 41.89 | 0 | 0.9600 [0.9572, 0.9628] | n/a | 1 | 0.141 [0.123, 0.160] |
| noise_gauss_psnr | test | 100 | 4 | 41.91 [41.91, 41.91] | 41.89 | 0 | 0.9623 [0.9609, 0.9636] | 0.9337 | 3 | 0.083 [0.077, 0.090] |
| noise_uniform | messy | 20 | 4 | 40.82 [40.76, 40.90] | 40.73 | 0 | 0.9492 [0.9457, 0.9527] | n/a | 11 | 0.154 [0.136, 0.172] |
| noise_uniform | test | 100 | 4 | 40.80 [40.77, 40.83] | 40.72 | 0 | 0.9518 [0.9501, 0.9534] | 0.9257 | 43 | 0.096 [0.088, 0.103] |
| photoguard | messy | 20 | 4 | 37.24 [37.15, 37.33] | 36.85 | 20 | 0.9052 [0.8989, 0.9111] | n/a | 20 | 0.497 [0.475, 0.517] |
| photoguard | test | 100 | 4 | 37.15 [37.12, 37.19] | 36.83 | 100 | 0.9061 [0.9027, 0.9092] | 0.8123 | 100 | 0.467 [0.457, 0.476] |
| photoguard | val | 20 | 4 | 37.14 [37.04, 37.26] | 36.91 | 19 | 0.9040 [0.8938, 0.9112] | n/a | 20 | 0.456 [0.438, 0.475] |
| v1 | messy | 20 | 4 | 42.03 [41.85, 42.21] | 40.96 | 0 | 0.9703 [0.9678, 0.9727] | n/a | 0 | 0.271 [0.243, 0.299] |
| v1 | test | 100 | 4 | 42.42 [42.33, 42.51] | 41.17 | 0 | 0.9763 [0.9753, 0.9773] | 0.9587 | 0 | 0.239 [0.226, 0.251] |
| v1 | v1_eval70 | 70 | 4 | 42.17 [42.07, 42.27] | 41.12 | 0 | 0.9756 [0.9743, 0.9769] | 0.9575 | 0 | 0.231 [0.215, 0.245] |
| v1 | val | 20 | 4 | 42.42 [42.23, 42.68] | 41.85 | 0 | 0.9760 [0.9741, 0.9779] | n/a | 0 | 0.236 [0.214, 0.260] |
| v2_antipure | test | 100 | 4 | 41.18 [40.99, 41.36] | 38.91 | 0 | 0.9700 [0.9669, 0.9728] | 0.9235 | 13 | 0.270 [0.256, 0.285] |
| v2_antipure | val | 20 | 4 | 41.39 [41.02, 41.78] | 39.61 | 0 | 0.9735 [0.9694, 0.9770] | n/a | 1 | 0.259 [0.233, 0.286] |
| v2_cheap | test | 100 | 4 | 41.58 [41.45, 41.70] | 40.28 | 0 | 0.9790 [0.9783, 0.9797] | 0.9694 | 0 | 0.220 [0.211, 0.230] |
| v2_cheap | val | 20 | 4 | 41.65 [41.36, 41.99] | 40.62 | 0 | 0.9789 [0.9772, 0.9806] | n/a | 0 | 0.212 [0.194, 0.232] |
| v2_final | ae_pool | 120 | 4 | 40.91 [40.82, 41.01] | 40.02 | 0 | 0.9612 [0.9594, 0.9630] | 0.9346 | 19 | 0.259 [0.247, 0.271] |
| v2_final | messy | 20 | 4 | 39.72 [39.44, 39.98] | 38.61 | 0 | 0.9431 [0.9363, 0.9493] | n/a | 14 | 0.371 [0.339, 0.401] |
| v2_final | test | 100 | 4 | 40.46 [40.35, 40.57] | 38.76 | 0 | 0.9549 [0.9522, 0.9574] | 0.9197 | 29 | 0.289 [0.275, 0.304] |
| v2_main | messy | 20 | 4 | 41.63 [41.38, 41.88] | 40.24 | 0 | 0.9749 [0.9729, 0.9770] | n/a | 0 | 0.274 [0.247, 0.301] |
| v2_main | test | 100 | 4 | 41.72 [41.60, 41.85] | 40.34 | 0 | 0.9784 [0.9777, 0.9791] | 0.9707 | 0 | 0.239 [0.229, 0.250] |
| v2_main | val | 20 | 4 | 41.62 [41.40, 41.89] | 40.98 | 0 | 0.9775 [0.9758, 0.9791] | n/a | 0 | 0.237 [0.217, 0.257] |
| v2_tgt_gray | val | 20 | 4 | 41.94 [41.63, 42.27] | 40.38 | 0 | 0.9825 [0.9809, 0.9841] | n/a | 0 | 0.268 [0.250, 0.285] |
| v2_tgt_zero | test | 100 | 4 | 42.16 [42.03, 42.29] | 40.58 | 0 | 0.9822 [0.9814, 0.9831] | 0.9728 | 0 | 0.264 [0.251, 0.277] |
| v2_tgt_zero | val | 20 | 4 | 42.07 [41.79, 42.37] | 40.62 | 0 | 0.9819 [0.9800, 0.9837] | n/a | 0 | 0.257 [0.234, 0.278] |
| v2_tgt_zero_hi | test | 100 | 4 | 40.05 [39.97, 40.13] | 39.14 | 0 | 0.9628 [0.9615, 0.9641] | 0.9459 | 4 | 0.360 [0.349, 0.371] |
| v2_tgt_zero_hi | val | 20 | 4 | 39.99 [39.79, 40.19] | 39.16 | 0 | 0.9617 [0.9579, 0.9651] | n/a | 1 | 0.352 [0.335, 0.370] |
| v2_tgt_zero_mid | test | 100 | 4 | 40.91 [40.81, 41.01] | 39.75 | 0 | 0.9728 [0.9718, 0.9739] | 0.9615 | 0 | 0.326 [0.315, 0.338] |

("n/a" in the min-SSIM column = the `_long.csv` row for that split was not inspected here; the min SSIM for the test, v1_eval70 and ae_pool rows was read from `invisibility_methods_long.csv`.) The `v2_tgt_zero_mid` val row does not exist (mid was never run on validation; only on test).

The `final_readout.md` copy of the invisibility block (same test photos; it adds the max pixel change) reads: v1 42.42 (41.17) / 0.9763 (0.9587) / 0 / 0 / max change 4; v2_main 41.72 (40.34) / 0.9784 (0.9707) / 0 / 0 / 4; v2_tgt_zero 42.16 (40.58) / 0.9822 (0.9728) / 0 / 0 / 4; v2_tgt_zero_mid 40.91 (39.75) / 0.9728 (0.9615) / 0 / 0 / 4; v2_tgt_zero_hi 40.05 (39.14) / 0.9628 (0.9459) / 0 / 4 / 4; photoguard 37.15 (36.83) / 0.9061 (0.8123) / 100 / 100 / 4; mist 38.00 (37.47) / 0.9229 (0.8275) / 50 / 100 / 4 (columns: PSNR mean (min) / SSIM mean (min) / photos < 38 dB / photos SSIM < 0.95 / max pixel change).

**FINAL_REPORT 7.1** gives the same numbers in a different order, plus these rows not in the generated invisibility table's printed columns: v2_cheap SSIM min 0.9694; v2_antipure PSNR min 38.91, SSIM min 0.9235; v2_final PSNR min 38.76, SSIM min 0.9197; eps8 SSIM min 0.7724; uniform noise SSIM min 0.9257; Gaussian noise SSIM min 0.9337. They agree with `_long.csv`.

**Gate INV (`results/tables/gate_INV.md`), verdict FAIL/MARGINAL**, rule "PSNR >= 38 and SSIM >= 0.95 (per image); human 2AFC accuracy not far above 50%". Status column: PASS / MARGINAL / FAIL as printed by the project. Mean values rounded as printed.

| method | n | psnr | ssim | n_below_psnr | n_below_ssim | status |
|---|---|---|---|---|---|---|
| abl_den_only | 100 | 50.76 | 1.00 | 0 | 0 | PASS |
| abl_vis_only | 100 | 41.73 | 0.98 | 0 | 0 | PASS |
| eps2 | 100 | 45.34 | 0.98 | 0 | 0 | PASS |
| eps8 | 100 | 35.98 | 0.90 | 98 | 96 | FAIL |
| mist | 100 | 38.00 | 0.92 | 50 | 100 | FAIL |
| noise_gauss_psnr | 100 | 41.91 | 0.96 | 0 | 3 | MARGINAL |
| noise_uniform | 100 | 40.80 | 0.95 | 0 | 43 | MARGINAL |
| photoguard | 100 | 37.15 | 0.91 | 100 | 100 | FAIL |
| v1 | 100 | 42.42 | 0.98 | 0 | 0 | PASS |
| v2_antipure | 100 | 41.18 | 0.97 | 0 | 13 | MARGINAL |
| v2_cheap | 100 | 41.58 | 0.98 | 0 | 0 | PASS |
| v2_final | 100 | 40.46 | 0.95 | 0 | 29 | MARGINAL |
| v2_main | 100 | 41.72 | 0.98 | 0 | 0 | PASS |
| v2_tgt_zero | 100 | 42.16 | 0.98 | 0 | 0 | PASS |
| v2_tgt_zero_hi | 100 | 40.05 | 0.96 | 0 | 4 | MARGINAL |
| v2_tgt_zero_mid | 100 | 40.91 | 0.97 | 0 | 0 | PASS |

"human 2AFC result not available (E10 not run)".

Note on the invisibility source: `invisibility_methods_long.csv` rows carry a note "source: protect.csv, run prot_<method>_test". The `headline_sim_long.csv`/`claims_long.csv` files carry a second set of invisibility values computed from the edit rows (columns inv_psnr, inv_ssim, inv_lpips) which differ slightly from the protect.csv values for some methods (for example noise_uniform inv_ssim 0.9542 versus 0.9518 in `invisibility_methods.md`; v2_final 0.9555 versus 0.9549). The project's printed tables and report use the protect.csv values; the difference was seen but not explained in the files.

**Linked facts.** The Optuna search only checked the PSNR/SSIM limits on the *mean* over its photos, which is why `v2_final` has 29/100 per-photo SSIM violations (FINAL_REPORT 7.1 says "its 20 photos", but see the discrepancy list at the end of this part). PhotoGuard and Mist also change at most 4 gray levels (max pixel change 4 in the final_readout block) but use the whole budget on nearly every pixel with no structure penalty.

---

### 20.4 Noise controls and paired difference vs noise (E1)

**Source:** `results/tables/e1.md` (noise vs clean), `results/tables/paired_vs_noise.md` (methods vs the two noise controls). **Split:** test. **n:** 100 photos, 3 seeds, up to 3 prompts. **Methods:** `noise_uniform` (uniform random noise, same L-infinity budget 4/255, PSNR 40.80) and `noise_gauss_psnr` (Gaussian noise matched to v1's PSNR, PSNR 41.91; its maximum change can exceed the budget by design). No cleanup. **CI:** yes. **Cell meaning:** PSR A / PSR B are the two methods' PSR on the same photos; "PSR A-B [CI]" is the paired mean difference; "p (PSR)" Wilcoxon; "sim A-B" difference in mean ArcFace similarity (negative = A removes more identity); "images A>B" = printed share/column as in the file (its definition is not stated in the footnote; the values behave like the share of photos for which A's per-photo PSR is larger than B's).

**E1: noise vs clean (all editors), `e1.md`.**

| comparison (A vs B) | purifier | editor | pairs | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|---|---|
| noise_uniform vs clean | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.008 [-0.011, -0.004] | <0.001 | 0.00 |
| noise_uniform vs clean | none | SD1.5 s0.55 | 100 | 0.15 | 0.14 | +0.01 [-0.01, +0.03] | 0.362 | -0.010 [-0.013, -0.006] | <0.001 | 0.23 |
| noise_uniform vs clean | none | SD1.5 s0.75 | 100 | 0.83 | 0.84 | -0.01 [-0.05, +0.03] | 0.770 | -0.001 [-0.007, +0.005] | 0.970 | 0.15 |
| noise_uniform vs clean | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| noise_uniform vs clean | none | InstructPix2Pix | 100 | 0.08 | 0.18 | -0.10 [-0.16, -0.06] | <0.001 | +0.048 [+0.022, +0.075] | <0.001 | 0.03 |
| noise_uniform vs clean | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.002 [-0.003, -0.001] | <0.001 | 0.00 |
| noise_uniform vs clean | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.01] | 1.000 | -0.003 [-0.006, +0.001] | 0.291 | 0.01 |
| noise_gauss_psnr vs clean | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.009 [-0.013, -0.006] | <0.001 | 0.00 |
| noise_gauss_psnr vs clean | none | SD1.5 s0.55 | 100 | 0.15 | 0.14 | +0.01 [-0.01, +0.03] | 0.560 | -0.008 [-0.012, -0.005] | <0.001 | 0.19 |
| noise_gauss_psnr vs clean | none | SD1.5 s0.75 | 100 | 0.83 | 0.84 | -0.01 [-0.05, +0.03] | 0.550 | -0.002 [-0.008, +0.003] | 0.762 | 0.14 |
| noise_gauss_psnr vs clean | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| noise_gauss_psnr vs clean | none | InstructPix2Pix | 100 | 0.09 | 0.18 | -0.09 [-0.14, -0.05] | <0.001 | +0.049 [+0.023, +0.076] | <0.001 | 0.03 |
| noise_gauss_psnr vs clean | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.001 [-0.002, -0.000] | 0.001 | 0.00 |
| noise_gauss_psnr vs clean | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.002 [-0.005, +0.001] | 0.789 | 0.00 |

Noise PSR rows are also in `numbers.json` and `headline.md`: clean 0.14 [0.10, 0.18], uniform noise 0.15 [0.11, 0.19], Gaussian noise 0.15 [0.11, 0.19], mean ArcFace similarity 0.22 / 0.21 / 0.21 (headline_sim.md; numbers.json: clean 0.223 [0.208, 0.237], noise_uniform 0.213, noise_gauss_psnr 0.214).

**FINAL_REPORT 7.2 version (3-decimal PSR, differences in points):** clean photos 13.9%; uniform noise 14.9% (+1.0 over clean [-1.0, +3.1]); Gaussian noise PSNR-matched 14.7% (+0.8 [-1.2, +2.9]); v1 24.0% (+10.1 over clean [+7.2, +13.0]); v1 minus uniform noise +9.1 points [+6.4, +12.0]. (The REPORT_v2_draft also gives v2_main minus Gaussian noise +9.7 [+6.8, +12.3].) The 3-decimal 14.9/14.7 come from FINAL_REPORT; the table above gives 0.15 for both because it rounds to 2 decimals.

**Paired differences vs noise, `paired_vs_noise.md` (no purifier). SD1.5 s0.55 rows and all other editors. Rows against `noise_uniform` first; rows against `noise_gauss_psnr` are given for SD1.5 s0.55 and are near-identical for the other editors (full rows in the file).**

| comparison (A vs B) | editor | pairs | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|---|
| v1 vs noise_uniform | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.043 [-0.051, -0.035] | <0.001 | 0.00 |
| v1 vs noise_uniform | SD1.5 s0.55 | 100 | 0.24 | 0.15 | +0.09 [+0.06, +0.12] | <0.001 | -0.036 [-0.042, -0.030] | <0.001 | 0.51 |
| v1 vs noise_uniform | SD1.5 s0.75 | 100 | 0.85 | 0.83 | +0.02 [-0.03, +0.07] | 0.405 | -0.002 [-0.009, +0.006] | 0.559 | 0.19 |
| v1 vs noise_uniform | InstructPix2Pix | 100 | 0.09 | 0.08 | +0.01 [-0.02, +0.04] | 0.532 | -0.081 [-0.111, -0.050] | <0.001 | 0.12 |
| v1 vs noise_uniform | SD inpaint | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.012 [-0.014, -0.011] | <0.001 | 0.00 |
| v1 vs noise_uniform | IP-Adapter FaceID | 100 | 0.00 | 0.00 | -0.00 [-0.01, +0.00] | 1.000 | -0.039 [-0.047, -0.031] | <0.001 | 0.00 |
| v1 vs noise_gauss_psnr | SD1.5 s0.55 | 100 | 0.24 | 0.15 | +0.09 [+0.06, +0.12] | <0.001 | -0.037 [-0.043, -0.031] | <0.001 | 0.53 |
| photoguard vs noise_uniform | SD1.5 s0.3 | 100 | 0.17 | 0.00 | +0.17 [+0.12, +0.22] | <0.001 | -0.317 [-0.332, -0.302] | <0.001 | 0.33 |
| photoguard vs noise_uniform | SD1.5 s0.55 | 100 | 0.81 | 0.15 | +0.66 [+0.61, +0.71] | <0.001 | -0.169 [-0.181, -0.157] | <0.001 | 0.99 |
| photoguard vs noise_uniform | SD1.5 s0.75 | 100 | 0.96 | 0.83 | +0.12 [+0.08, +0.17] | <0.001 | -0.035 [-0.046, -0.024] | <0.001 | 0.34 |
| photoguard vs noise_uniform | InstructPix2Pix | 100 | 0.72 | 0.08 | +0.64 [+0.56, +0.72] | <0.001 | -0.628 [-0.697, -0.555] | <0.001 | 0.80 |
| photoguard vs noise_uniform | SD inpaint | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.066 [-0.071, -0.061] | <0.001 | 0.00 |
| photoguard vs noise_uniform | IP-Adapter FaceID | 100 | 0.03 | 0.00 | +0.02 [+0.00, +0.05] | 0.125 | -0.197 [-0.223, -0.173] | <0.001 | 0.04 |
| photoguard vs noise_gauss_psnr | SD1.5 s0.55 | 100 | 0.81 | 0.15 | +0.66 [+0.62, +0.71] | <0.001 | -0.170 [-0.183, -0.158] | <0.001 | 0.99 |
| mist vs noise_uniform | SD1.5 s0.3 | 100 | 0.05 | 0.00 | +0.05 [+0.02, +0.08] | 0.002 | -0.227 [-0.243, -0.212] | <0.001 | 0.10 |
| mist vs noise_uniform | SD1.5 s0.55 | 100 | 0.63 | 0.15 | +0.48 [+0.43, +0.53] | <0.001 | -0.130 [-0.141, -0.120] | <0.001 | 0.99 |
| mist vs noise_uniform | SD1.5 s0.75 | 100 | 0.94 | 0.83 | +0.11 [+0.06, +0.16] | <0.001 | -0.028 [-0.038, -0.018] | <0.001 | 0.29 |
| mist vs noise_uniform | InstructPix2Pix | 100 | 0.04 | 0.08 | -0.03 [-0.07, +0.00] | 0.087 | -0.074 [-0.104, -0.042] | <0.001 | 0.06 |
| mist vs noise_uniform | SD inpaint | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.097 [-0.104, -0.090] | <0.001 | 0.00 |
| mist vs noise_uniform | IP-Adapter FaceID | 100 | 0.04 | 0.00 | +0.03 [+0.01, +0.06] | 0.008 | -0.189 [-0.215, -0.164] | <0.001 | 0.08 |
| mist vs noise_gauss_psnr | SD1.5 s0.55 | 100 | 0.63 | 0.15 | +0.48 [+0.43, +0.53] | <0.001 | -0.132 [-0.142, -0.121] | <0.001 | 0.98 |
| v2_cheap vs noise_uniform | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.038 [-0.046, -0.030] | <0.001 | 0.00 |
| v2_cheap vs noise_uniform | SD1.5 s0.55 | 100 | 0.22 | 0.15 | +0.08 [+0.05, +0.10] | <0.001 | -0.028 [-0.034, -0.023] | <0.001 | 0.49 |
| v2_cheap vs noise_uniform | SD1.5 s0.75 | 100 | 0.85 | 0.83 | +0.01 [-0.04, +0.06] | 0.838 | -0.003 [-0.012, +0.005] | 0.570 | 0.20 |
| v2_cheap vs noise_uniform | InstructPix2Pix | 100 | 0.15 | 0.08 | +0.07 [+0.03, +0.12] | 0.001 | -0.168 [-0.208, -0.129] | <0.001 | 0.18 |
| v2_cheap vs noise_uniform | SD inpaint | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.013 [-0.014, -0.011] | <0.001 | 0.00 |
| v2_cheap vs noise_uniform | IP-Adapter FaceID | 100 | 0.00 | 0.00 | -0.00 [-0.01, +0.00] | 1.000 | -0.054 [-0.063, -0.046] | <0.001 | 0.00 |
| v2_cheap vs noise_gauss_psnr | SD1.5 s0.55 | 100 | 0.22 | 0.15 | +0.08 [+0.05, +0.11] | <0.001 | -0.030 [-0.036, -0.024] | <0.001 | 0.45 |
| v2_main vs noise_uniform | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.052 [-0.061, -0.043] | <0.001 | 0.00 |
| v2_main vs noise_uniform | SD1.5 s0.55 | 100 | 0.24 | 0.15 | +0.09 [+0.07, +0.12] | <0.001 | -0.035 [-0.042, -0.028] | <0.001 | 0.48 |
| v2_main vs noise_uniform | SD1.5 s0.75 | 100 | 0.88 | 0.83 | +0.05 [+0.01, +0.09] | 0.075 | -0.007 [-0.015, +0.000] | 0.069 | 0.23 |
| v2_main vs noise_uniform | InstructPix2Pix | 100 | 0.12 | 0.08 | +0.04 [+0.00, +0.08] | 0.037 | -0.158 [-0.197, -0.120] | <0.001 | 0.15 |
| v2_main vs noise_uniform | SD inpaint | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.014 [-0.016, -0.013] | <0.001 | 0.00 |
| v2_main vs noise_uniform | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.063 [-0.075, -0.052] | <0.001 | 0.00 |
| v2_main vs noise_gauss_psnr | SD1.5 s0.55 | 100 | 0.24 | 0.15 | +0.10 [+0.07, +0.13] | <0.001 | -0.036 [-0.043, -0.029] | <0.001 | 0.49 |
| v2_antipure vs noise_uniform | IP-Adapter FaceID (only editor with data) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.055 [-0.065, -0.045] | <0.001 | 0.00 |
| v2_final vs noise_uniform | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.060 [-0.068, -0.052] | <0.001 | 0.00 |
| v2_final vs noise_uniform | SD1.5 s0.55 | 100 | 0.29 | 0.15 | +0.15 [+0.11, +0.18] | <0.001 | -0.055 [-0.062, -0.049] | <0.001 | 0.69 |
| v2_final vs noise_uniform | SD1.5 s0.75 | 100 | 0.87 | 0.83 | +0.04 [-0.01, +0.09] | 0.102 | -0.015 [-0.024, -0.007] | 0.003 | 0.23 |
| v2_final vs noise_uniform | InstructPix2Pix | 100 | 0.03 | 0.08 | -0.04 [-0.08, -0.01] | 0.009 | -0.019 [-0.046, +0.010] | 0.026 | 0.03 |
| v2_final vs noise_uniform | SD inpaint | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.019 [-0.021, -0.017] | <0.001 | 0.00 |
| v2_final vs noise_uniform | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.046 [-0.054, -0.039] | <0.001 | 0.00 |
| v2_final vs noise_gauss_psnr | SD1.5 s0.55 | 100 | 0.29 | 0.15 | +0.15 [+0.12, +0.18] | <0.001 | -0.057 [-0.064, -0.050] | <0.001 | 0.68 |

SDXL rows are "not run" for the noise controls (no SDXL noise run), so every comparison against noise is "not run" for SDXL. `v2_antipure` has no SD1.5/ip2p/inpaint rows. The remaining `noise_gauss_psnr` rows (other editors) are in `paired_vs_noise.md`. Observed quirk to note: for InstructPix2Pix the p (sim) of `v2_final vs noise_uniform` is 0.026 although its CI [-0.046, +0.010] includes zero (Wilcoxon p versus bootstrap CI of the mean; both printed as in the file).

---

### 20.5 Headline PSR / similarity tables

**Source:** `results/tables/headline.md` (7 purifier columns), `headline_full.md` (all purifiers), `headline_sim.md` (mean ArcFace similarity), `e1.md`, `e2.md`. **Split:** test. **n:** 100 photos per cell, 3 seeds, 3 prompts (none) / 1 rotated prompt (cleanups; see the coverage note above). **Setting:** SD1.5 img2img s0.55. **CI:** yes. **Cell meaning:** PSR of the row's method under the column's cleanup applied before editing; "not run" = no rows. "Non-adaptive mode. Held-out purifiers: jpeg_other, downup_other, diffpure_*, restore_swin2sr, gridpure. PhotoGuard and Mist are our reimplementations (not the official code)."

**headline.md: PSR by method and purifier.**

| method | none | jpeg_q75 | jpeg_q50 | downup_0.5 | vae_rt | diffpure_0.1 | diffpure_0.2 |
|---|---|---|---|---|---|---|---|
| clean (E0) | 0.14 [0.10, 0.18] | 0.10 [0.07, 0.14] | 0.13 [0.09, 0.18] | 0.10 [0.06, 0.14] | 0.12 [0.08, 0.17] | 0.13 [0.09, 0.18] | 0.22 [0.16, 0.27] |
| uniform noise (E1) | 0.15 [0.11, 0.19] | not run | not run | not run | not run | not run | not run |
| Gaussian noise, PSNR-matched (E1) | 0.15 [0.11, 0.19] | not run | not run | not run | not run | not run | not run |
| PhotoGuard encoder attack (E2, reimpl.) | 0.81 [0.77, 0.85] | 0.18 [0.13, 0.24] | 0.17 [0.12, 0.21] | 0.14 [0.09, 0.19] | 0.59 [0.52, 0.66] | 0.67 [0.61, 0.74] | not run |
| Mist-style (E2, reimpl.) | 0.63 [0.58, 0.68] | 0.21 [0.16, 0.27] | 0.16 [0.11, 0.21] | 0.13 [0.08, 0.17] | 0.59 [0.53, 0.65] | 0.64 [0.58, 0.70] | not run |
| v1 (E3) | 0.24 [0.19, 0.29] | 0.15 [0.10, 0.20] | 0.17 [0.12, 0.22] | 0.10 [0.06, 0.14] | 0.22 [0.17, 0.28] | 0.26 [0.21, 0.32] | 0.32 [0.25, 0.38] |
| v2 cheap purifiers (E4) | 0.22 [0.18, 0.27] | 0.17 [0.12, 0.22] | 0.17 [0.13, 0.23] | 0.14 [0.10, 0.19] | 0.21 [0.16, 0.27] | 0.27 [0.21, 0.33] | 0.31 [0.25, 0.38] |
| v2 + one-step SD purifier + L_den (E5) | 0.24 [0.20, 0.29] | 0.18 [0.13, 0.24] | 0.18 [0.13, 0.23] | 0.15 [0.10, 0.21] | 0.25 [0.19, 0.32] | 0.28 [0.22, 0.34] | 0.33 [0.27, 0.40] |
| v2 + anti-purification (E6) | not run | 0.16 [0.11, 0.21] | 0.16 [0.11, 0.21] | 0.10 [0.07, 0.14] | 0.27 [0.21, 0.33] | 0.31 [0.25, 0.37] | 0.35 [0.28, 0.41] |
| v2 final | 0.29 [0.24, 0.35] | 0.16 [0.12, 0.21] | 0.15 [0.11, 0.20] | 0.14 [0.10, 0.19] | 0.27 [0.21, 0.34] | 0.30 [0.23, 0.36] | 0.35 [0.28, 0.42] |

**headline_full.md: PSR by method and ALL purifiers** (columns: none, jpeg_q90, jpeg_q75, jpeg_q50, jpeg_other, downup_0.5, downup_other, vae_rt, diffpure_0.05, diffpure_0.1, diffpure_0.2, diffpure_0.3, restore_swin2sr, gridpure). gridpure is "not run" for every row. Noise rows only have "none".

| method | none | jpeg_q90 | jpeg_q75 | jpeg_q50 | jpeg_other | downup_0.5 | downup_other |
|---|---|---|---|---|---|---|---|
| clean (E0) | 0.14 [0.10, 0.18] | 0.11 [0.07, 0.16] | 0.10 [0.07, 0.14] | 0.13 [0.09, 0.18] | 0.12 [0.08, 0.16] | 0.10 [0.06, 0.14] | 0.09 [0.05, 0.12] |
| PhotoGuard (reimpl.) | 0.81 [0.77, 0.85] | not run | 0.18 [0.13, 0.24] | 0.17 [0.12, 0.21] | not run | 0.14 [0.09, 0.19] | not run |
| Mist-style (reimpl.) | 0.63 [0.58, 0.68] | not run | 0.21 [0.16, 0.27] | 0.16 [0.11, 0.21] | not run | 0.13 [0.08, 0.17] | not run |
| v1 (E3) | 0.24 [0.19, 0.29] | 0.22 [0.16, 0.27] | 0.15 [0.10, 0.20] | 0.17 [0.12, 0.22] | 0.15 [0.11, 0.20] | 0.10 [0.06, 0.14] | 0.09 [0.06, 0.13] |
| v2_cheap (E4) | 0.22 [0.18, 0.27] | 0.20 [0.15, 0.26] | 0.17 [0.12, 0.22] | 0.17 [0.13, 0.23] | 0.17 [0.12, 0.22] | 0.14 [0.10, 0.19] | 0.11 [0.07, 0.16] |
| v2_main (E5) | 0.24 [0.20, 0.29] | 0.22 [0.17, 0.28] | 0.18 [0.13, 0.24] | 0.18 [0.13, 0.23] | 0.18 [0.13, 0.24] | 0.15 [0.10, 0.21] | 0.12 [0.08, 0.17] |
| v2_antipure (E6) | not run | 0.21 [0.16, 0.27] | 0.16 [0.11, 0.21] | 0.16 [0.11, 0.21] | 0.17 [0.13, 0.22] | 0.10 [0.07, 0.14] | 0.11 [0.07, 0.15] |
| v2_final | 0.29 [0.24, 0.35] | 0.20 [0.15, 0.25] | 0.16 [0.12, 0.21] | 0.15 [0.11, 0.20] | 0.17 [0.12, 0.21] | 0.14 [0.10, 0.19] | 0.11 [0.07, 0.15] |

| method | vae_rt | diffpure_0.05 | diffpure_0.1 | diffpure_0.2 | diffpure_0.3 | restore_swin2sr | gridpure |
|---|---|---|---|---|---|---|---|
| clean (E0) | 0.12 [0.08, 0.17] | 0.12 [0.08, 0.17] | 0.13 [0.09, 0.18] | 0.22 [0.16, 0.27] | 0.26 [0.21, 0.32] | 0.10 [0.06, 0.14] | not run |
| PhotoGuard (reimpl.) | 0.59 [0.52, 0.66] | not run | 0.67 [0.61, 0.74] | not run | not run | not run | not run |
| Mist-style (reimpl.) | 0.59 [0.53, 0.65] | not run | 0.64 [0.58, 0.70] | not run | not run | not run | not run |
| v1 (E3) | 0.22 [0.17, 0.28] | 0.24 [0.18, 0.30] | 0.26 [0.21, 0.32] | 0.32 [0.25, 0.38] | 0.36 [0.30, 0.42] | 0.15 [0.11, 0.20] | not run |
| v2_cheap (E4) | 0.21 [0.16, 0.27] | 0.26 [0.19, 0.32] | 0.27 [0.21, 0.33] | 0.31 [0.25, 0.38] | 0.35 [0.28, 0.41] | 0.17 [0.12, 0.23] | not run |
| v2_main (E5) | 0.25 [0.19, 0.32] | 0.26 [0.20, 0.33] | 0.28 [0.22, 0.34] | 0.33 [0.27, 0.40] | 0.36 [0.30, 0.43] | 0.18 [0.13, 0.24] | not run |
| v2_antipure (E6) | 0.27 [0.21, 0.33] | 0.30 [0.24, 0.36] | 0.31 [0.25, 0.37] | 0.35 [0.28, 0.41] | 0.43 [0.37, 0.49] | 0.16 [0.12, 0.21] | not run |
| v2_final | 0.27 [0.21, 0.34] | 0.27 [0.21, 0.33] | 0.30 [0.23, 0.36] | 0.35 [0.28, 0.42] | 0.41 [0.35, 0.48] | 0.16 [0.12, 0.21] | not run |

(The file also has the two noise rows with "not run" in all purified columns. The original method labels in the file are "v2 cheap purifiers (E4)" etc., shortened here.)

**headline_sim.md: mean ArcFace similarity of the edit to the source (lower = identity removed); edits without a detected face are excluded from the mean (they count as successes in PSR).**

| method | none | jpeg_q75 | jpeg_q50 | downup_0.5 | vae_rt | diffpure_0.1 | diffpure_0.2 |
|---|---|---|---|---|---|---|---|
| clean (E0) | 0.22 [0.21, 0.24] | 0.23 [0.21, 0.24] | 0.22 [0.21, 0.24] | 0.24 [0.23, 0.26] | 0.23 [0.21, 0.24] | 0.21 [0.20, 0.23] | 0.19 [0.18, 0.21] |
| uniform noise (E1) | 0.21 [0.20, 0.23] | not run | not run | not run | not run | not run | not run |
| Gaussian noise, PSNR-matched (E1) | 0.21 [0.20, 0.23] | not run | not run | not run | not run | not run | not run |
| PhotoGuard (reimpl.) | 0.04 [0.04, 0.05] | 0.20 [0.18, 0.21] | 0.20 [0.19, 0.22] | 0.22 [0.21, 0.24] | 0.09 [0.07, 0.10] | 0.07 [0.06, 0.08] | not run |
| Mist-style (reimpl.) | 0.08 [0.07, 0.09] | 0.19 [0.18, 0.21] | 0.21 [0.19, 0.22] | 0.22 [0.21, 0.24] | 0.09 [0.08, 0.10] | 0.08 [0.07, 0.09] | not run |
| v1 (E3) | 0.18 [0.16, 0.19] | 0.21 [0.19, 0.22] | 0.21 [0.19, 0.22] | 0.23 [0.22, 0.25] | 0.18 [0.16, 0.19] | 0.17 [0.15, 0.18] | 0.15 [0.14, 0.16] |
| v2_cheap (E4) | 0.18 [0.17, 0.20] | 0.20 [0.19, 0.22] | 0.21 [0.19, 0.23] | 0.23 [0.21, 0.24] | 0.18 [0.17, 0.20] | 0.17 [0.16, 0.19] | 0.16 [0.15, 0.18] |
| v2_main (E5) | 0.18 [0.17, 0.19] | 0.20 [0.18, 0.22] | 0.21 [0.19, 0.22] | 0.22 [0.21, 0.24] | 0.18 [0.16, 0.19] | 0.16 [0.15, 0.18] | 0.15 [0.14, 0.17] |
| v2_antipure (E6) | not run | 0.21 [0.19, 0.23] | 0.21 [0.20, 0.23] | 0.24 [0.22, 0.25] | 0.16 [0.15, 0.18] | 0.15 [0.14, 0.17] | 0.14 [0.13, 0.16] |
| v2_final | 0.16 [0.15, 0.17] | 0.21 [0.19, 0.22] | 0.21 [0.19, 0.23] | 0.23 [0.22, 0.25] | 0.16 [0.15, 0.17] | 0.15 [0.14, 0.17] | 0.14 [0.13, 0.15] |

`results/tables/numbers.json` is a flat lookup (keys like `psr.<method>.<purifier>`, `psr_pt`, `sim`, `n`) of exactly these values with 3 decimals for similarity (for example `sim.clean.none` 0.223 [0.208, 0.237]; `sim.photoguard.none` 0.044 [0.037, 0.052]; `sim.mist.none` 0.083 [0.073, 0.092]; `sim.v1.none` 0.177 [0.164, 0.190]; `psr.v2_*` rows are also there). It is a number source for the templates, not a separate result.

**E2 (PhotoGuard and Mist vs noise control and v1), `e2.md`.** Same layout as 20.4; test, 100 photos, no purifier; PhotoGuard and Mist are our re-implementations at our budget 4/255.

| comparison (A vs B) | editor | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|
| photoguard vs noise_uniform | SD1.5 s0.3 | 0.17 | 0.00 | +0.17 [+0.12, +0.22] | <0.001 | -0.317 [-0.332, -0.302] | <0.001 | 0.33 |
| photoguard vs noise_uniform | SD1.5 s0.55 | 0.81 | 0.15 | +0.66 [+0.61, +0.71] | <0.001 | -0.169 [-0.181, -0.157] | <0.001 | 0.99 |
| photoguard vs noise_uniform | SD1.5 s0.75 | 0.96 | 0.83 | +0.12 [+0.08, +0.17] | <0.001 | -0.035 [-0.046, -0.024] | <0.001 | 0.34 |
| photoguard vs noise_uniform | SDXL s0.55 | not run | not run | not run |  | not run |  |  |
| photoguard vs noise_uniform | InstructPix2Pix | 0.72 | 0.08 | +0.64 [+0.56, +0.72] | <0.001 | -0.628 [-0.697, -0.555] | <0.001 | 0.80 |
| photoguard vs noise_uniform | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.066 [-0.071, -0.061] | <0.001 | 0.00 |
| photoguard vs noise_uniform | IP-Adapter FaceID | 0.03 | 0.00 | +0.02 [+0.00, +0.05] | 0.125 | -0.197 [-0.223, -0.173] | <0.001 | 0.04 |
| mist vs noise_uniform | SD1.5 s0.3 | 0.05 | 0.00 | +0.05 [+0.02, +0.08] | 0.002 | -0.227 [-0.243, -0.212] | <0.001 | 0.10 |
| mist vs noise_uniform | SD1.5 s0.55 | 0.63 | 0.15 | +0.48 [+0.43, +0.53] | <0.001 | -0.130 [-0.141, -0.120] | <0.001 | 0.99 |
| mist vs noise_uniform | SD1.5 s0.75 | 0.94 | 0.83 | +0.11 [+0.06, +0.16] | <0.001 | -0.028 [-0.038, -0.018] | <0.001 | 0.29 |
| mist vs noise_uniform | SDXL s0.55 | not run | not run | not run |  | not run |  |  |
| mist vs noise_uniform | InstructPix2Pix | 0.04 | 0.08 | -0.03 [-0.07, +0.00] | 0.087 | -0.074 [-0.104, -0.042] | <0.001 | 0.06 |
| mist vs noise_uniform | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.097 [-0.104, -0.090] | <0.001 | 0.00 |
| mist vs noise_uniform | IP-Adapter FaceID | 0.04 | 0.00 | +0.03 [+0.01, +0.06] | 0.008 | -0.189 [-0.215, -0.164] | <0.001 | 0.08 |
| v1 vs photoguard | SD1.5 s0.3 | 0.00 | 0.17 | -0.17 [-0.22, -0.12] | <0.001 | +0.274 [+0.259, +0.289] | <0.001 | 0.00 |
| v1 vs photoguard | SD1.5 s0.55 | 0.24 | 0.81 | -0.57 [-0.62, -0.52] | <0.001 | +0.133 [+0.122, +0.144] | <0.001 | 0.00 |
| v1 vs photoguard | SD1.5 s0.75 | 0.85 | 0.96 | -0.10 [-0.15, -0.06] | <0.001 | +0.032 [+0.023, +0.042] | <0.001 | 0.07 |
| v1 vs photoguard | SDXL s0.55 | 0.02 | 0.03 | -0.01 [-0.03, +0.02] | 0.781 | +0.022 [+0.016, +0.027] | <0.001 | 0.04 |
| v1 vs photoguard | InstructPix2Pix | 0.09 | 0.72 | -0.63 [-0.71, -0.54] | <0.001 | +0.547 [+0.467, +0.625] | <0.001 | 0.04 |
| v1 vs photoguard | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.054 [+0.049, +0.058] | <0.001 | 0.00 |
| v1 vs photoguard | IP-Adapter FaceID | 0.00 | 0.03 | -0.03 [-0.05, -0.01] | 0.062 | +0.158 [+0.136, +0.182] | <0.001 | 0.00 |
| v1 vs mist | SD1.5 s0.3 | 0.00 | 0.05 | -0.05 [-0.08, -0.02] | 0.002 | +0.185 [+0.171, +0.199] | <0.001 | 0.00 |
| v1 vs mist | SD1.5 s0.55 | 0.24 | 0.63 | -0.39 [-0.44, -0.34] | <0.001 | +0.095 [+0.085, +0.104] | <0.001 | 0.02 |
| v1 vs mist | SD1.5 s0.75 | 0.85 | 0.94 | -0.09 [-0.14, -0.04] | 0.001 | +0.026 [+0.017, +0.035] | <0.001 | 0.08 |
| v1 vs mist | SDXL s0.55 | not run | not run | not run |  | not run |  |  |
| v1 vs mist | InstructPix2Pix | 0.09 | 0.04 | +0.05 [+0.01, +0.09] | 0.037 | -0.007 [-0.034, +0.018] | 0.613 | 0.14 |
| v1 vs mist | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.085 [+0.078, +0.091] | <0.001 | 0.00 |
| v1 vs mist | IP-Adapter FaceID | 0.00 | 0.04 | -0.04 [-0.06, -0.01] | 0.008 | +0.150 [+0.127, +0.175] | <0.001 | 0.00 |

Quirk: `v1 vs photoguard` on FaceID prints CI [-0.05, -0.01] (excludes 0) with p = 0.062 (Wilcoxon), and `photoguard vs noise_gauss_psnr` on FaceID p = 0.062 as well; the file prints both and does not comment.

**Messy split (`messy.md`, 20 photos, one rotated prompt per photo, no cleanup, intervals wide).**

| method | SD1.5 s0.55 | IP-Adapter FaceID |
|---|---|---|
| clean (E0) | 0.15 [0.07, 0.25] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| uniform noise (E1) | 0.17 [0.08, 0.25] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| Gaussian noise, PSNR-matched (E1) | 0.17 [0.07, 0.27] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| PhotoGuard (reimpl.) | 0.73 [0.60, 0.85] (n=20) | 0.03 [0.00, 0.08] (n=20) |
| Mist-style (reimpl.) | 0.62 [0.47, 0.75] (n=20) | 0.05 [0.00, 0.10] (n=20) |
| v1 (E3) | 0.32 [0.18, 0.45] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| v2_main (E5) | 0.28 [0.15, 0.43] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| v2 final | 0.43 [0.28, 0.58] (n=20) | 0.02 [0.00, 0.05] (n=20) |

(Messy photos: pose, blur, lighting, occlusion, pre-compression. The coverage map lists 3 prompts x 3 seeds = 60 rows per cell; the `messy.md` footnote says "one rotated prompt per image". Both are printed in the project files.)

---

### 20.6 Robustness to the 13 held-out cleanups, net protection, and the gates

#### 20.6.1 `results/tables/final_readout.md` (generated by `scripts/readout_final.py`)

**Split:** test. **n:** 100 photos per method (printed "(n=100)"). **Editor:** SD1.5 img2img s0.55 in the first two tables; other editors in the third. **Seeds:** 3. **Cell meaning (first table):** PSR (mean over photos); in brackets the *net protection* = paired difference to the **clean** photos under the *same* cleanup, with 95% CI over photos; "*" = CI excludes 0. **Second table:** paired difference of PSR to v1 under the same cleanup with CI. "-" = not run. Columns: no cleanup, JPEG q75, JPEG q50, down-up 0.5, VAE round trip, DiffPure t*=0.1/0.2/0.3.

**Table 1: PSR and net protection over clean photos under the same cleanup.**

| method | none | jpeg_q75 | jpeg_q50 | downup_0.5 | vae_rt | diffpure_0.1 | diffpure_0.2 | diffpure_0.3 |
|---|---|---|---|---|---|---|---|---|
| clean (n=100) | 0.139 | 0.103 | 0.130 | 0.097 | 0.123 | 0.133 | 0.217 | 0.260 |
| v1 (n=100) | 0.240 (+0.101 [+0.072,+0.130]*) | 0.150 (+0.047 [+0.010,+0.087]*) | 0.167 (+0.037 [+0.010,+0.067]*) | 0.097 (+0.000 [-0.023,+0.027]) | 0.220 (+0.097 [+0.053,+0.143]*) | 0.263 (+0.130 [+0.083,+0.180]*) | 0.317 (+0.100 [+0.050,+0.150]*) | 0.360 (+0.100 [+0.047,+0.153]*) |
| v2_main (n=100) | 0.243 (+0.104 [+0.076,+0.134]*) | 0.183 (+0.080 [+0.037,+0.127]*) | 0.180 (+0.050 [+0.020,+0.083]*) | 0.153 (+0.057 [+0.023,+0.093]*) | 0.250 (+0.127 [+0.080,+0.177]*) | 0.280 (+0.147 [+0.093,+0.197]*) | 0.333 (+0.117 [+0.067,+0.167]*) | 0.363 (+0.103 [+0.050,+0.160]*) |
| v2_tgt_zero (n=100) | 0.418 (+0.279 [+0.232,+0.326]*) | 0.210 (+0.107 [+0.067,+0.153]*) | 0.180 (+0.050 [+0.017,+0.087]*) | 0.143 (+0.047 [+0.013,+0.083]*) | 0.433 (+0.310 [+0.240,+0.380]*) | 0.443 (+0.310 [+0.250,+0.377]*) | 0.477 (+0.260 [+0.197,+0.323]*) | 0.497 (+0.237 [+0.170,+0.303]*) |
| v2_tgt_zero_mid (n=100) | 0.497 (+0.358 [+0.311,+0.403]*) | 0.233 (+0.130 [+0.083,+0.180]*) | 0.203 (+0.073 [+0.033,+0.117]*) | 0.190 (+0.093 [+0.053,+0.137]*) | 0.523 (+0.400 [+0.327,+0.473]*) | 0.517 (+0.383 [+0.320,+0.447]*) | 0.570 (+0.353 [+0.283,+0.420]*) | 0.600 (+0.340 [+0.270,+0.410]*) |
| v2_tgt_zero_hi (n=100) | 0.528 (+0.389 [+0.338,+0.443]*) | 0.267 (+0.163 [+0.113,+0.213]*) | 0.213 (+0.083 [+0.047,+0.120]*) | 0.190 (+0.093 [+0.057,+0.133]*) | 0.533 (+0.410 [+0.340,+0.477]*) | 0.610 (+0.477 [+0.407,+0.540]*) | 0.600 (+0.383 [+0.313,+0.457]*) | 0.663 (+0.403 [+0.340,+0.470]*) |
| photoguard (n=100) | 0.810 (+0.671 [+0.626,+0.717]*) | 0.183 (+0.080 [+0.040,+0.123]*) | 0.167 (+0.037 [+0.010,+0.063]*) | 0.137 (+0.040 [+0.010,+0.073]*) | 0.590 (+0.467 [+0.390,+0.543]*) | 0.673 (+0.540 [+0.470,+0.610]*) | - | - |
| mist (n=100) | 0.631 (+0.492 [+0.443,+0.542]*) | 0.210 (+0.107 [+0.070,+0.147]*) | 0.157 (+0.027 [-0.003,+0.057]) | 0.127 (+0.030 [-0.000,+0.060]) | 0.590 (+0.467 [+0.400,+0.530]*) | 0.640 (+0.507 [+0.437,+0.573]*) | - | - |

`final_readout.md` has no `v2_cheap` or `v2_final` rows. Their net-protection rows are only in FINAL_REPORT 7.3 and REPORT_v2_draft 5.4 (below).

**FINAL_REPORT 7.3 / draft 5.4: net protection over clean photos (adds v2_cheap and v2_final).** Cell = PSR(method) minus PSR(clean) under the same cleanup, paired; * = CI excludes 0. Source of these two extra columns: FINAL_REPORT.md only (the underlying CIs are not printed there).

| cleanup | clean PSR | v1 | v2_main | v2_cheap | v2_final | tgt_zero | tgt_zero_mid | tgt_zero_hi |
|---|---|---|---|---|---|---|---|---|
| none | 0.139 | +0.101* | +0.104* | +0.086* | +0.156* | +0.279* | +0.358* | +0.389* |
| JPEG q75 | 0.103 | +0.047* | +0.080* | +0.063* | +0.060* | +0.107* | +0.130* | +0.163* |
| JPEG q50 | 0.130 | +0.037* | +0.050* | +0.043* | +0.023 | +0.050* | +0.073* | +0.083* |
| down-up 0.5 | 0.097 | +0.000 | +0.057* | +0.047* | +0.047* | +0.047* | +0.093* | +0.093* |
| VAE round trip | 0.123 | +0.097* | +0.127* | +0.090* | +0.150* | +0.310* | +0.400* | +0.410* |
| DiffPure 0.1 | 0.133 | +0.130* | +0.147* | +0.133* | +0.163* | +0.310* | +0.383* | +0.477* |
| DiffPure 0.2 | 0.217 | +0.100* | +0.117* | +0.097* | +0.133* | +0.260* | +0.353* | +0.383* |
| DiffPure 0.3 | 0.260 | +0.100* | +0.103* | +0.087* | +0.153* | +0.237* | +0.340* | +0.403* |

These agree with Table 1 for the shared columns.

**Table 2: paired difference of PSR to v1 (same photos, same cleanup), `final_readout.md`.**

| method | none | jpeg_q75 | jpeg_q50 | downup_0.5 | vae_rt | diffpure_0.1 | diffpure_0.2 | diffpure_0.3 |
|---|---|---|---|---|---|---|---|---|
| v2_main | +0.003 [-0.018,+0.024] | +0.033 [-0.003,+0.070] | +0.013 [-0.017,+0.043] | +0.057 [+0.027,+0.087]* | +0.030 [-0.007,+0.070] | +0.017 [-0.027,+0.060] | +0.017 [-0.030,+0.067] | +0.003 [-0.043,+0.050] |
| v2_tgt_zero | +0.178 [+0.141,+0.217]* | +0.060 [+0.023,+0.103]* | +0.013 [-0.020,+0.047] | +0.047 [+0.013,+0.080]* | +0.213 [+0.160,+0.267]* | +0.180 [+0.120,+0.243]* | +0.160 [+0.107,+0.217]* | +0.137 [+0.077,+0.200]* |
| v2_tgt_zero_mid | +0.257 [+0.218,+0.299]* | +0.083 [+0.037,+0.130]* | +0.037 [+0.000,+0.073]* | +0.093 [+0.053,+0.137]* | +0.303 [+0.243,+0.363]* | +0.253 [+0.190,+0.317]* | +0.253 [+0.187,+0.320]* | +0.240 [+0.173,+0.303]* |
| v2_tgt_zero_hi | +0.288 [+0.240,+0.340]* | +0.117 [+0.077,+0.160]* | +0.047 [+0.010,+0.083]* | +0.093 [+0.050,+0.137]* | +0.313 [+0.250,+0.377]* | +0.347 [+0.280,+0.413]* | +0.283 [+0.223,+0.347]* | +0.303 [+0.240,+0.363]* |
| photoguard | +0.570 [+0.524,+0.619]* | +0.033 [-0.003,+0.073] | +0.000 [-0.023,+0.023] | +0.040 [+0.010,+0.073]* | +0.370 [+0.303,+0.437]* | +0.410 [+0.347,+0.477]* | - | - |
| mist | +0.391 [+0.348,+0.438]* | +0.060 [+0.027,+0.093]* | -0.010 [-0.043,+0.023] | +0.030 [-0.000,+0.063] | +0.370 [+0.310,+0.430]* | +0.377 [+0.313,+0.437]* | - | - |

Rounding inconsistency to be aware of: the `mid` JPEG q50 interval prints "[+0.000,+0.073]*" (lower bound rounds to 0.000 but is starred); the Mist rows print "[-0.000,+0.060]" without a star.

**Table 3: other editors and strengths, no cleanup (PSR; net over clean in brackets), `final_readout.md`.**

| method | SD1.5 s0.3 | SD1.5 s0.75 | IP-Adapter FaceID | InstructPix2Pix | inpainting | SDXL |
|---|---|---|---|---|---|---|
| clean | 0.000 | 0.840 | 0.000 | 0.180 | 0.000 | 0.013 |
| v1 | 0.000 (+0.000 [+0.000,+0.000]) | 0.853 (+0.013 [-0.027,+0.053]) | 0.000 (+0.000 [+0.000,+0.000]) | 0.090 (-0.090 [-0.147,-0.040]*) | 0.000 (+0.000 [+0.000,+0.000]) | 0.020 (+0.007 [-0.007,+0.020]) |
| v2_main | 0.000 (+0.000 [+0.000,+0.000]) | 0.883 (+0.043 [-0.000,+0.090]) | 0.003 (+0.003 [+0.000,+0.010]) | 0.117 (-0.063 [-0.127,-0.007]*) | 0.000 (+0.000 [+0.000,+0.000]) | - |
| v2_tgt_zero | 0.000 (+0.000 [+0.000,+0.000]) | 0.910 (+0.070 [+0.027,+0.113]*) | 0.007 (+0.007 [+0.000,+0.020]) | - | - | - |
| v2_tgt_zero_mid | 0.000 (+0.000 [+0.000,+0.000]) | 0.900 (+0.060 [+0.010,+0.113]*) | 0.020 (+0.020 [+0.003,+0.043]*) | - | - | - |
| v2_tgt_zero_hi | 0.007 (+0.007 [+0.000,+0.017]) | 0.897 (+0.057 [+0.007,+0.107]*) | 0.020 (+0.020 [+0.000,+0.047]) | - | - | - |
| photoguard | 0.170 (+0.170 [+0.120,+0.227]*) | 0.957 (+0.117 [+0.070,+0.163]*) | 0.027 (+0.027 [+0.007,+0.053]*) | 0.717 (+0.537 [+0.443,+0.623]*) | 0.000 (+0.000 [+0.000,+0.000]) | 0.027 (+0.013 [-0.000,+0.030]) |
| mist | 0.047 (+0.047 [+0.020,+0.077]*) | 0.940 (+0.100 [+0.053,+0.147]*) | 0.037 (+0.037 [+0.013,+0.067]*) | 0.043 (-0.137 [-0.200,-0.070]*) | 0.000 (+0.000 [+0.000,+0.000]) | - |

The fourth block of `final_readout.md` is the invisibility table already reproduced in 20.3.

Mist and PhotoGuard cleanup columns beyond these (jpeg_q90, jpeg_other, ...) are "not run".

#### 20.6.2 Gates E3, E4, E5, E6, INV (`results/tables/gates.md`, `gates.csv/json`, `gate_E3.md` ... `gate_INV.md`, and E3-E6 detail tables `e3.md`..`e6.md`)

Project statement: "reported honestly; a missed gate is reported, not changed". Gate values are the handoff's suggestions; the E3 cut-offs and purifier sets are analyst choices (stated in the rule column, changeable with `--gate_param`). "DEGENERATE" = the doubling rule is meaningless.

| gate | what | verdict | rule (operationalisation) | notes |
|---|---|---|---|---|
| E3 | v1 under purification (decision rule for Phase 5) | **BARELY DROPS** (recommend skipping Phase 5; ask the user) | mean relative PSR drop of v1 over jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1, diffpure_0.2 (SD1.5 s0.55): >= 50% collapses -> continue with Phase 5; < 20% barely drops -> recommend skipping Phase 5 and ask the user; in between -> ask the user. The cut-offs are analyst proposals (the handoff gives none). | mean relative PSR drop over 6 core purifiers = **0.16** |
| E4 | v2_cheap vs v1 under cheap purifiers | **FAIL** | PSR(v2_cheap) >= 2 x PSR(v1) under jpeg_q75, jpeg_q50, downup_0.5, vae_rt (SD1.5 s0.55) with PSNR >= 38. Point estimate decides; the paired ratio CI is shown. Degenerate if v1 PSR >= 0.5, trivial if < 0.02. | v2_cheap PSNR mean 41.58 dB over 100 images; 0 below 38 |
| E5 | v2_main: held-out DiffPure t*=0.1 vs no purifier | **PASS (robust)** | PSR under diffpure_0.1 >= 0.5 x PSR without purifier (SD1.5 s0.55, same images). Point estimate decides; "robust" if the paired ratio CI lower bound also >= the fraction. | |
| E6 | v2_antipure vs v2_main (anti-purification terms) | **GO (gain on 1 of 3 purifiers)** | go if the paired PSR gain under higher-t* DiffPure has a 95% CI above 0; "no gain (within CI)" otherwise (handoff Phase 5: drop it if gains are within the CI). | |
| INV | Invisibility | **FAIL/MARGINAL** (see rows in 20.3) | PSNR >= 38 and SSIM >= 0.95 (per image); human 2AFC accuracy not far above 50%. | human 2AFC result not available (E10 not run) |

**E3 detail (`e3.md` and `gate_E3.md`), v1 under purification, SD1.5 s0.55, n = 100 pairs.** "in decision" = whether the purifier is among the 6 core purifiers of the rule. drop = PSR(purified) - PSR(no purifier), paired; relative drop = drop/PSR(none), negative = PSR went up. v1 PSR without purifier = 0.24 in every row. Last column = PSR of *clean* photos under the same purifier (control).

| purifier | v1 PSR (purified) [95% CI] | drop [95% CI] | p | relative drop | in decision | clean + purifier PSR (control) |
|---|---|---|---|---|---|---|
| jpeg_q90 | 0.22 [0.16, 0.27] | -0.02 [-0.06, +0.01] | 0.089 | 0.10 | no | 0.11 [0.07, 0.16] |
| jpeg_q75 | 0.15 [0.10, 0.20] | -0.09 [-0.13, -0.05] | <0.001 | 0.37 | yes | 0.10 [0.07, 0.14] |
| jpeg_q50 | 0.17 [0.12, 0.22] | -0.07 [-0.11, -0.03] | <0.001 | 0.31 | yes | 0.13 [0.09, 0.18] |
| jpeg_other | 0.15 [0.11, 0.20] | -0.09 [-0.13, -0.05] | <0.001 | 0.36 | no | 0.12 [0.08, 0.16] |
| downup_0.5 | 0.10 [0.06, 0.14] | -0.14 [-0.18, -0.11] | <0.001 | 0.60 | yes | 0.10 [0.06, 0.14] |
| downup_other | 0.09 [0.06, 0.13] | -0.15 [-0.18, -0.11] | <0.001 | 0.61 | no | 0.09 [0.05, 0.12] |
| vae_rt | 0.22 [0.17, 0.28] | -0.02 [-0.05, +0.02] | 0.108 | 0.08 | yes | 0.12 [0.08, 0.17] |
| diffpure_0.05 | 0.24 [0.18, 0.30] | -0.00 [-0.04, +0.04] | 0.709 | 0.01 | no | 0.12 [0.08, 0.17] |
| diffpure_0.1 | 0.26 [0.21, 0.32] | +0.02 [-0.02, +0.06] | 0.514 | -0.10 | yes | 0.13 [0.09, 0.18] |
| diffpure_0.2 | 0.32 [0.25, 0.38] | +0.08 [+0.03, +0.12] | 0.006 | -0.32 | yes | 0.22 [0.16, 0.27] |
| diffpure_0.3 | 0.36 [0.30, 0.42] | +0.12 [+0.08, +0.16] | <0.001 | -0.50 | no | 0.26 [0.21, 0.32] |
| restore_swin2sr | 0.15 [0.11, 0.20] | -0.09 [-0.12, -0.05] | <0.001 | 0.36 | no | 0.10 [0.06, 0.14] |

Mean relative drop over the 6 core purifiers = 0.16 (checked: 0.37, 0.31, 0.60, 0.08, -0.10, -0.32 average 0.157).

**E4 detail (`e4.md`, `gate_E4.md`), v2_cheap vs v1, gate FAIL.** n = 100; ratio = PSR(v2_cheap)/PSR(v1) with paired CI; gate needs ratio >= 2 on the four primary purifiers.

| purifier | v1 PSR | v2_cheap PSR | diff [95% CI] | p | ratio [95% CI] | gate |
|---|---|---|---|---|---|---|
| jpeg_q75 | 0.15 | 0.17 | +0.02 [-0.02, +0.05] | 0.404 | 1.11 [0.89, 1.38] | FAIL |
| jpeg_q50 | 0.17 | 0.17 | +0.01 [-0.03, +0.04] | 0.665 | 1.04 [0.85, 1.29] | FAIL |
| downup_0.5 | 0.10 | 0.14 | +0.05 [+0.01, +0.08] | 0.009 | 1.48 [1.13, 1.96] | FAIL |
| vae_rt | 0.22 | 0.21 | -0.01 [-0.05, +0.03] | 0.914 | 0.97 [0.81, 1.16] | FAIL |
| jpeg_other (held-out, info) | 0.15 | 0.17 | +0.02 [-0.01, +0.05] | 0.346 | 1.11 [0.92, 1.37] | FAIL |
| downup_other (held-out, info) | 0.09 | 0.11 | +0.02 [-0.01, +0.05] | 0.216 | 1.21 [0.89, 1.67] | FAIL |

**E5 detail (`e5.md`, `gate_E5.md`), gate PASS (robust).** Held-out purifiers; diffpure_* share the SD1.5 UNet with the one-step training purifier (gray-box); gridpure/restore_swin2sr do not. Ratio = PSR(purified)/PSR(none) with paired CI. Gate = ratio of diffpure_0.1 >= 0.5 (primary cell is v2_main diffpure_0.1 only; the v2_final row is flagged non-primary in `gate_E5.md`).

| method | purifier | PSR (no purifier) | PSR (purified) | ratio purified/none [95% CI] | minus v1 under same purifier [95% CI] | p | gate |
|---|---|---|---|---|---|---|---|
| v2_main | diffpure_0.1 | 0.24 | 0.28 | 1.15 [1.00, 1.32] | +0.02 [-0.02, +0.06] | 0.125 | PASS (robust) |
| v2_main | diffpure_0.2 | 0.24 | 0.33 | 1.37 [1.18, 1.61] | +0.02 [-0.03, +0.06] | 0.547 | info |
| v2_main | diffpure_0.3 | 0.24 | 0.36 | 1.49 [1.29, 1.74] | +0.00 [-0.04, +0.05] | 0.885 | info |
| v2_main | gridpure | not run | not run | not run | not run |  | not run |
| v2_main | restore_swin2sr | 0.24 | 0.18 | 0.74 [0.59, 0.88] | +0.03 [-0.00, +0.06] | 0.065 | info |
| v2_final | diffpure_0.1 | 0.29 | 0.30 | 1.01 [0.86, 1.16] | +0.03 [-0.02, +0.09] | 0.154 | PASS (robust) |
| v2_final | diffpure_0.2 | 0.29 | 0.35 | 1.19 [1.04, 1.35] | +0.03 [-0.02, +0.09] | 0.203 | info |
| v2_final | diffpure_0.3 | 0.29 | 0.41 | 1.40 [1.24, 1.59] | +0.05 [+0.00, +0.11] | 0.039 | info |
| v2_final | gridpure | not run | not run | not run | not run |  | not run |
| v2_final | restore_swin2sr | 0.29 | 0.16 | 0.55 [0.44, 0.67] | +0.01 [-0.02, +0.04] | 0.682 | info |

**E6 detail (`e6.md`, `gate_E6.md`), v2_antipure vs v2_main, GO on 1 of 3.**

| purifier | v2_main PSR | v2_antipure PSR | diff [95% CI] | p | verdict |
|---|---|---|---|---|---|
| diffpure_0.1 | 0.28 | 0.31 | +0.03 [-0.02, +0.09] | 0.369 | no gain (within CI) |
| diffpure_0.2 | 0.33 | 0.35 | +0.01 [-0.03, +0.06] | 0.625 | no gain (within CI) |
| diffpure_0.3 | 0.36 | 0.43 | +0.07 [+0.02, +0.11] | 0.014 | gain |

FINAL_REPORT 7.3 summary of the gates: "E3 'v1 barely drops' by the raw-PSR rule (this rule ignores that cleanups also change clean photos, hence net protection above); E4 fails ...; E5 passes (v2_main PSR under DiffPure 0.1 is 0.280 against half of 0.243); E6 'go' on 1 of 3 cleanups; invisibility fails for `v2_final`."

---

### 20.7 Paired differences vs v1 (`results/tables/paired_vs_v1.md`)

**Source:** `paired_vs_v1.md` (+csv/tex/`_long.csv`). **Split:** test, no purifier, 100 pairs per row, 3 seeds, up to 3 prompts. **CI:** yes. **Cell meaning:** A minus v1 on the same photos; PSR A-B > 0 = A protects more than v1; sim A-B < 0 = A lowers ArcFace similarity more than v1. `v2_antipure` only has FaceID data here; SDXL only exists for v2_final and photoguard. The "images A>B" column is printed as in the file.

| comparison | editor | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|
| v2_cheap vs v1 | SD1.5 s0.3 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.005 [-0.002, +0.012] | 0.265 | 0.00 |
| v2_cheap vs v1 | SD1.5 s0.55 | 0.22 | 0.24 | -0.02 [-0.04, +0.01] | 0.275 | +0.007 [+0.003, +0.012] | 0.007 | 0.22 |
| v2_cheap vs v1 | SD1.5 s0.75 | 0.85 | 0.85 | -0.01 [-0.05, +0.04] | 0.483 | -0.001 [-0.009, +0.006] | 0.652 | 0.14 |
| v2_cheap vs v1 | InstructPix2Pix | 0.15 | 0.09 | +0.06 [+0.02, +0.10] | 0.003 | -0.078 [-0.105, -0.052] | <0.001 | 0.17 |
| v2_cheap vs v1 | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.000 [-0.002, +0.001] | 0.545 | 0.00 |
| v2_cheap vs v1 | IP-Adapter FaceID | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.016 [-0.022, -0.010] | <0.001 | 0.00 |
| v2_main vs v1 | SD1.5 s0.3 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.009 [-0.016, -0.002] | 0.011 | 0.00 |
| v2_main vs v1 | SD1.5 s0.55 | 0.24 | 0.24 | +0.00 [-0.02, +0.02] | 0.966 | +0.001 [-0.004, +0.005] | 0.749 | 0.26 |
| v2_main vs v1 | SD1.5 s0.75 | 0.88 | 0.85 | +0.03 [-0.01, +0.07] | 0.283 | -0.005 [-0.011, +0.000] | 0.094 | 0.19 |
| v2_main vs v1 | InstructPix2Pix | 0.12 | 0.09 | +0.03 [-0.00, +0.05] | 0.038 | -0.074 [-0.100, -0.048] | <0.001 | 0.12 |
| v2_main vs v1 | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.002 [-0.004, -0.000] | 0.005 | 0.00 |
| v2_main vs v1 | IP-Adapter FaceID | 0.00 | 0.00 | +0.00 [+0.00, +0.01] | 1.000 | -0.025 [-0.033, -0.017] | <0.001 | 0.01 |
| v2_antipure vs v1 | IP-Adapter FaceID | 0.00 | 0.00 | +0.00 [+0.00, +0.01] | 1.000 | -0.016 [-0.023, -0.009] | <0.001 | 0.01 |
| v2_final vs v1 | SD1.5 s0.3 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.017 [-0.025, -0.008] | <0.001 | 0.00 |
| v2_final vs v1 | SD1.5 s0.55 | 0.29 | 0.24 | +0.05 [+0.02, +0.09] | 0.008 | -0.019 [-0.025, -0.014] | <0.001 | 0.45 |
| v2_final vs v1 | SD1.5 s0.75 | 0.87 | 0.85 | +0.02 [-0.02, +0.06] | 0.307 | -0.014 [-0.022, -0.007] | 0.001 | 0.16 |
| v2_final vs v1 | SDXL s0.55 | 0.02 | 0.02 | +0.00 [-0.01, +0.01] | 1.000 | -0.003 [-0.007, +0.002] | 0.307 | 0.02 |
| v2_final vs v1 | InstructPix2Pix | 0.03 | 0.09 | -0.06 [-0.09, -0.03] | 0.002 | +0.062 [+0.039, +0.088] | <0.001 | 0.02 |
| v2_final vs v1 | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.007 [-0.009, -0.005] | <0.001 | 0.00 |
| v2_final vs v1 | IP-Adapter FaceID | 0.00 | 0.00 | +0.00 [+0.00, +0.01] | 1.000 | -0.008 [-0.013, -0.002] | <0.001 | 0.01 |
| photoguard vs v1 | SD1.5 s0.3 | 0.17 | 0.00 | +0.17 [+0.12, +0.22] | <0.001 | -0.274 [-0.289, -0.259] | <0.001 | 0.33 |
| photoguard vs v1 | SD1.5 s0.55 | 0.81 | 0.24 | +0.57 [+0.52, +0.62] | <0.001 | -0.133 [-0.144, -0.122] | <0.001 | 0.97 |
| photoguard vs v1 | SD1.5 s0.75 | 0.96 | 0.85 | +0.10 [+0.06, +0.15] | <0.001 | -0.032 [-0.042, -0.023] | <0.001 | 0.29 |
| photoguard vs v1 | SDXL s0.55 | 0.03 | 0.02 | +0.01 [-0.02, +0.03] | 0.781 | -0.022 [-0.027, -0.016] | <0.001 | 0.05 |
| photoguard vs v1 | InstructPix2Pix | 0.72 | 0.09 | +0.63 [+0.54, +0.71] | <0.001 | -0.547 [-0.625, -0.467] | <0.001 | 0.80 |
| photoguard vs v1 | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.054 [-0.058, -0.049] | <0.001 | 0.00 |
| photoguard vs v1 | IP-Adapter FaceID | 0.03 | 0.00 | +0.03 [+0.01, +0.05] | 0.062 | -0.158 [-0.182, -0.136] | <0.001 | 0.05 |
| mist vs v1 | SD1.5 s0.3 | 0.05 | 0.00 | +0.05 [+0.02, +0.08] | 0.002 | -0.185 [-0.199, -0.171] | <0.001 | 0.10 |
| mist vs v1 | SD1.5 s0.55 | 0.63 | 0.24 | +0.39 [+0.34, +0.44] | <0.001 | -0.095 [-0.104, -0.085] | <0.001 | 0.91 |
| mist vs v1 | SD1.5 s0.75 | 0.94 | 0.85 | +0.09 [+0.04, +0.14] | 0.001 | -0.026 [-0.035, -0.017] | <0.001 | 0.23 |
| mist vs v1 | InstructPix2Pix | 0.04 | 0.09 | -0.05 [-0.09, -0.01] | 0.037 | +0.007 [-0.018, +0.034] | 0.613 | 0.03 |
| mist vs v1 | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.085 [-0.091, -0.078] | <0.001 | 0.00 |
| mist vs v1 | IP-Adapter FaceID | 0.04 | 0.00 | +0.04 [+0.01, +0.06] | 0.008 | -0.150 [-0.175, -0.127] | <0.001 | 0.08 |

Not-run rows (all "not run"): v2_cheap/v2_main/mist on SDXL; v2_antipure on all editors except FaceID. The target-steered variants have no row in this table; their paired differences to v1 under cleanups are in `final_readout.md` Table 2 (20.6.1).

**Claim-sentence numbers (`results/tables/claims.md`; see 20.15).** v2_final vs v1: ArcFace-sim reduction 1.9 points [1.4, 2.5], p < 0.001, PSR gain +0.05 [+0.02, +0.09] (none); consistent with the v2_final vs v1 row above (sim -0.019, PSR +0.05 [+0.02, +0.09]).

**FINAL_REPORT 7.4 text version.** v2_main − v1: +0.3 points [−1.8, +2.6] without cleanup (no gain); supported only under down-up 0.5 (+5.7 [+3.0, +8.7]) and down-up (other) (+3.0 [+0.3, +6.3]). v2_final − v1: +5.4 [+2.4, +8.7] without cleanup, +4.7 under down-up 0.5, +5.3 under the VAE round trip, +5.3 [+0.3, +10.7] under DiffPure 0.3, "within noise elsewhere". REPORT_v2_draft §1 gives v2_final − v1 as "+5.4 points ... [+2.4, +8.6]" (upper bound 8.6 vs 8.7 in the report body). **Disagreement flagged:** these report-level intervals (v2_main − v1 [−1.8, +2.6]) differ slightly from the generated `final_readout.md` Table 2 (`+0.003 [-0.018,+0.024]`, i.e. [−1.8, +2.4]); the draft states a bootstrap with 2000 resamples while the generated tables use 10000, which is the likely source. Report both when citing; prefer the generated tables.

---

### 20.8 Ablations E7

**Source:** `results/tables/e7.md`, `e7_editors.md`, `e7_paired.md` (+csv/tex/long). **Split:** test, 100 photos, 3 seeds, SD1.5 s0.55 (e7.md) and editors (e7_editors.md, no cleanup). **Methods:** `abl_vis_only` ("L_vis only") and `abl_den_only` ("L_den only") both also drop L_sem (beta = 0) to isolate the phases; `v2_main` = both phases with L_sem ("v2 + one-step SD purifier + L_den (E5)" row). **`abl_no_sem` and `abl_no_pur` were NOT run** (no rows; planned session S14 not run). **CI:** yes.

**e7.md: PSR by variant and purifier.**

| method | none | jpeg_q50 | vae_rt | diffpure_0.1 | diffpure_0.2 |
|---|---|---|---|---|---|
| L_vis only | 0.25 [0.20, 0.31] | 0.19 [0.14, 0.25] | 0.26 [0.20, 0.32] | 0.30 [0.24, 0.37] | 0.31 [0.25, 0.38] |
| L_den only | 0.12 [0.08, 0.17] | 0.12 [0.08, 0.16] | 0.12 [0.08, 0.16] | 0.14 [0.10, 0.19] | 0.21 [0.16, 0.26] |
| both, no L_sem | not run | not run | not run | not run | not run |
| both, no purifier in loop | not run | not run | not run | not run | not run |
| v2_main (v2 + one-step SD purifier + L_den, E5) | 0.24 [0.20, 0.29] | 0.18 [0.13, 0.23] | 0.25 [0.19, 0.32] | 0.28 [0.22, 0.34] | 0.33 [0.27, 0.40] |

**e7_editors.md: PSR by variant and editor, no purifier.**

| method | SD1.5 s0.55 | InstructPix2Pix | SD inpaint (keep-face) | IP-Adapter FaceID |
|---|---|---|---|---|
| L_vis only | 0.25 [0.20, 0.31] | 0.13 [0.08, 0.18] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| L_den only | 0.12 [0.08, 0.17] | 0.15 [0.10, 0.21] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| both, no L_sem | not run | not run | not run | not run |
| both, no purifier in loop | not run | not run | not run | not run |
| v2_main (E5) | 0.24 [0.20, 0.29] | 0.12 [0.07, 0.17] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |

Invisibility of the ablation perturbations (20.3): `abl_den_only` PSNR 50.76 dB, SSIM 0.9951 (almost invisible: the optimiser left a tiny perturbation); `abl_vis_only` PSNR 41.73, SSIM 0.9781, both with 0 rule violations.

**e7_paired.md: paired differences between ablation variants (test, 100 pairs per row).**

| comparison (A vs B) | purifier | editor | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|---|
| abl_den_only vs abl_vis_only | none | SD1.5 s0.55 | 0.12 | 0.25 | -0.13 [-0.19, -0.08] | <0.001 | +0.051 [+0.040, +0.063] | <0.001 | 0.07 |
| abl_den_only vs abl_vis_only | none | InstructPix2Pix | 0.15 | 0.13 | +0.02 [-0.03, +0.08] | 0.542 | +0.140 [+0.095, +0.184] | <0.001 | 0.17 |
| abl_den_only vs abl_vis_only | none | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.016 [+0.014, +0.017] | <0.001 | 0.00 |
| abl_den_only vs abl_vis_only | none | IP-Adapter FaceID | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.065 [+0.055, +0.077] | <0.001 | 0.00 |
| abl_den_only vs abl_vis_only | jpeg_q50 | SD1.5 s0.55 | 0.12 | 0.19 | -0.07 [-0.11, -0.03] | <0.001 | +0.017 [+0.010, +0.024] | <0.001 | 0.04 |
| abl_den_only vs abl_vis_only | vae_rt | SD1.5 s0.55 | 0.12 | 0.26 | -0.14 [-0.19, -0.09] | <0.001 | +0.054 [+0.043, +0.067] | <0.001 | 0.04 |
| abl_den_only vs abl_vis_only | diffpure_0.1 | SD1.5 s0.55 | 0.14 | 0.30 | -0.16 [-0.21, -0.10] | <0.001 | +0.049 [+0.040, +0.059] | <0.001 | 0.07 |
| abl_den_only vs abl_vis_only | diffpure_0.2 | SD1.5 s0.55 | 0.21 | 0.31 | -0.11 [-0.16, -0.05] | <0.001 | +0.041 [+0.032, +0.051] | <0.001 | 0.11 |
| v2_main vs abl_vis_only | none | SD1.5 s0.55 | 0.24 | 0.25 | -0.01 [-0.05, +0.03] | 0.806 | +0.006 [-0.002, +0.014] | 0.063 | 0.29 |
| v2_main vs abl_vis_only | none | InstructPix2Pix | 0.12 | 0.13 | -0.01 [-0.04, +0.02] | 0.361 | +0.015 [-0.006, +0.036] | 0.318 | 0.06 |
| v2_main vs abl_vis_only | none | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.000 [-0.001, +0.001] | 0.797 | 0.00 |
| v2_main vs abl_vis_only | none | IP-Adapter FaceID | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.003 [-0.003, +0.009] | 0.934 | 0.00 |
| v2_main vs abl_vis_only | jpeg_q50 | SD1.5 s0.55 | 0.18 | 0.19 | -0.01 [-0.05, +0.03] | 0.508 | +0.005 [-0.000, +0.010] | 0.029 | 0.11 |
| v2_main vs abl_vis_only | vae_rt | SD1.5 s0.55 | 0.25 | 0.26 | -0.01 [-0.05, +0.03] | 0.829 | +0.007 [+0.001, +0.013] | 0.021 | 0.14 |
| v2_main vs abl_vis_only | diffpure_0.1 | SD1.5 s0.55 | 0.28 | 0.30 | -0.02 [-0.07, +0.02] | 0.389 | +0.006 [-0.001, +0.012] | 0.126 | 0.11 |
| v2_main vs abl_vis_only | diffpure_0.2 | SD1.5 s0.55 | 0.33 | 0.31 | +0.02 [-0.02, +0.06] | 0.581 | +0.005 [-0.000, +0.011] | 0.115 | 0.15 |
| v2_main vs abl_den_only | none | SD1.5 s0.55 | 0.24 | 0.12 | +0.12 [+0.08, +0.16] | <0.001 | -0.045 [-0.055, -0.034] | <0.001 | 0.56 |
| v2_main vs abl_den_only | none | InstructPix2Pix | 0.12 | 0.15 | -0.03 [-0.09, +0.02] | 0.274 | -0.121 [-0.168, -0.075] | <0.001 | 0.12 |
| v2_main vs abl_den_only | none | SD inpaint | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.016 [-0.017, -0.014] | <0.001 | 0.00 |
| v2_main vs abl_den_only | none | IP-Adapter FaceID | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.063 [-0.075, -0.051] | <0.001 | 0.00 |
| v2_main vs abl_den_only | jpeg_q50 | SD1.5 s0.55 | 0.18 | 0.12 | +0.06 [+0.04, +0.09] | <0.001 | -0.012 [-0.018, -0.005] | <0.001 | 0.21 |
| v2_main vs abl_den_only | vae_rt | SD1.5 s0.55 | 0.25 | 0.12 | +0.13 [+0.08, +0.18] | <0.001 | -0.048 [-0.060, -0.036] | <0.001 | 0.30 |
| v2_main vs abl_den_only | diffpure_0.1 | SD1.5 s0.55 | 0.28 | 0.14 | +0.14 [+0.09, +0.19] | <0.001 | -0.042 [-0.051, -0.033] | <0.001 | 0.38 |
| v2_main vs abl_den_only | diffpure_0.2 | SD1.5 s0.55 | 0.33 | 0.21 | +0.13 [+0.08, +0.17] | <0.001 | -0.036 [-0.046, -0.028] | <0.001 | 0.37 |

All rows `v2_main vs abl_no_sem` and `abl_no_sem vs abl_no_pur` (eight each) are "not run" (0 pairs).

FINAL_REPORT 7.5 / REPORT_v2_draft 5.5 text: `vis_only` 25.3% [19.7, 31.0] (+1.3 vs v1 [−2.4, +5.1], not significant); `den_only` 12.3% [8.0, 17.0] (−11.7 vs v1 [−16.4, −7.3]); v2_main ("both") 24.3%. `v2_antipure − v1` +4.7 [0.0, +8.7] under the VAE round trip and +5.0 [0.0, +10.0] under DiffPure 0.1 (borderline), 13/100 photos below SSIM 0.95.

---

### 20.9 Budget sweep (E9), adaptive attacker (E8), human study (E10)

**E9, `results/tables/e9.md`.** Source: protect.csv invisibility plus edit rows. **Test, 100 photos, 3 seeds.** Epsilon on the 0-255 scale. The eps=4 row is `v2_final` (so the sweep varies budget on the *tuned, antipure-on* config; eps2 and eps8 are the same config at a different budget). Editor SD1.5 s0.55 unless "FaceID". `eps6` was **not run** (session S23 not run). Cell = PSR with 95% CI; PSNR with CI; SSIM mean.

| eps (x/255) | method | PSNR dB [95% CI] | SSIM | PSR SD1.5 none | PSR SD1.5 jpeg_q75 | PSR SD1.5 diffpure_0.1 | PSR FaceID none |
|---|---|---|---|---|---|---|---|
| 2 | eps2 | 45.34 [45.29, 45.39] | 0.984 | 0.17 [0.12, 0.23] | 0.17 [0.12, 0.22] | 0.20 [0.15, 0.26] | 0.00 [0.00, 0.00] |
| 4 | v2_final | 40.46 [40.35, 40.57] | 0.955 | 0.29 [0.24, 0.35] | 0.16 [0.12, 0.21] | 0.30 [0.23, 0.36] | 0.00 [0.00, 0.01] |
| 8 | eps8 | 35.98 [35.74, 36.20] | 0.896 | 0.38 [0.31, 0.45] | 0.21 [0.16, 0.27] | 0.40 [0.34, 0.46] | 0.00 [0.00, 0.01] |

Per-photo invisibility at the three budgets (20.3): eps2 0/100 violations; v2_final 29/100 (SSIM); eps8 98/100 (PSNR) and 96/100 (SSIM). The REPORT_v2_draft 5.7 also gives point values "PSR 17.3% at 2/255, 29.4% at 4/255, 37.7% at 8/255" (3-digit).

**E8, adaptive attacker, `results/tables/e8.md`.** The attacker is a convolutional autoencoder used as a purifier ("LightShed-style learned perturbation remover"), applied before the editor. Editors SD1.5 s0.55 and IP-Adapter FaceID; test, 100 photos; "adaptive mode, reported separately from the non-adaptive headline". Cell: PSR with no purifier vs with the adaptive AE; change = paired difference (adaptive minus none). Last column = mean ArcFace similarity under the adaptive AE.

| method | editor | PSR (no purifier) | PSR (adaptive AE) | change [95% CI] | p | ArcFace sim (adaptive AE) |
|---|---|---|---|---|---|---|
| clean (E0) | SD1.5 s0.55 | 0.14 [0.10, 0.18] | 0.09 [0.06, 0.14] | -0.05 [-0.08, -0.01] | 0.005 | 0.232 [0.217, 0.248] |
| clean (E0) | IP-Adapter FaceID | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 1.000 | 0.703 [0.684, 0.720] |
| v1 (E3) | SD1.5 s0.55 | 0.24 [0.19, 0.29] | 0.11 [0.07, 0.15] | -0.13 [-0.17, -0.09] | <0.001 | 0.232 [0.216, 0.248] |
| v1 (E3) | IP-Adapter FaceID | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] | 1.000 | 0.699 [0.681, 0.716] |
| v2 final | SD1.5 s0.55 | 0.29 [0.24, 0.35] | 0.11 [0.07, 0.15] | -0.19 [-0.24, -0.14] | <0.001 | 0.229 [0.214, 0.244] |
| v2 final | IP-Adapter FaceID | 0.00 [0.00, 0.01] | 0.00 [0.00, 0.01] | +0.00 [+0.00, +0.00] | 1.000 | 0.701 [0.683, 0.718] |

Table footnote: "The attacker is weak by construction (120 disjoint images, 128 px patches, ~1 GPU-hour): the drop is a LOWER BOUND on the damage an adaptive attacker can do." (The printed change for v2 final, -0.19, is a paired mean of unrounded values; the rounded PSRs 0.29 and 0.11 differ by 0.18.) The adaptive attacker was **not run** against `v2_tgt_*`.

FINAL_REPORT 7.6 text version (3 decimals, net over clean in brackets): PSR under the adaptive AE = 9.3% for clean photos, 11.0% for v1 (net +1.7 [−1.0, +4.7]), 10.7% for v2_final (net +1.3 [−1.0, +3.7]).

**Autoencoder training details (`results/S15/S15/adaptive_ae/ae_final_train_log.json`, config `config/parts/baselines.yaml` `adaptive_ae:` block, script `scripts/train_adaptive_ae.py`, job definition in `kaggle/jobs.yaml`, session S15).**
- Training pairs: 120 photos of the `ae_pool` split (disjoint from test, messy, train, validation, calibration), each as (clean, protected) where the protected images were produced by **`v2_final`** (job step `prot_v2_final_ae_pool`, with `methods.v2_final.eot.k_draws=2`). The log lists 108 training stems and 12 validation stems (val fraction 0.1). Consequence: the AE was trained on `v2_final` perturbations only; its effect on `v1` photos (and clean photos) is a transfer. (The files do not state this consequence; it follows from the job definition.)
- Architecture: autoencoder with `base_channels` 32, `depth` 3 (three 2x downsamplings), 3 input channels. Loss: L1 only (`lpips_weight` 0). Patch 128 px, batch size 16, learning rate 0.0002, seed 2026.
- Budget: 60 minutes wall-clock (log: 3601.8 s wall, 3600.07 s elapsed), **35496 steps**. The first logged L1 loss is 0.0148 at step 1; the last logged loss is 0.00790 at step 35450 (710 logged points).
- Validation: the log's `val` block reports PSNR of protected vs clean 40.84 dB and of the AE output vs clean **45.12 dB** (the FINAL_REPORT and draft quote "45.1 dB"; the draft calls it "validation PSNR of its output vs the clean photo"). The `val` block prints `n: 8`, whereas `meta.val_stems` has 12 entries (unexplained inconsistency in the log; possibly a minimum-size filter).
- Weights file `ae_final.pt` and checkpoint are in `results/S15/S15/adaptive_ae/`.

**E10 human study (`results/tables/e10.md`).** Task A (2AFC: which image has noise): **not run**. Task B (same person?): **not run**. Scripts exist (`scripts/make_human_study.py`, `scripts/analyze_human_study.py`); no data collected. No claim about human perception is made anywhere.

---

### 20.10 Other editors and strengths; per prompt; CLIP-I

**`results/tables/per_editor.md`: PSR by method and editor, no purifier, test, 100 photos each, 3 seeds.** Cell = PSR with CI. Footer: "FaceID reads identity through ArcFace, not the VAE; low PSR there for VAE-based attacks is expected. SDXL ran for clean, v1, PhotoGuard and v2_final only."

| method | SD1.5 s0.3 | SD1.5 s0.55 | SD1.5 s0.75 | SDXL s0.55 | InstructPix2Pix | SD inpaint (keep-face) | IP-Adapter FaceID |
|---|---|---|---|---|---|---|---|
| clean (E0) | 0.00 [0.00, 0.00] | 0.14 [0.10, 0.18] | 0.84 [0.79, 0.89] | 0.01 [0.00, 0.03] | 0.18 [0.12, 0.25] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| uniform noise (E1) | 0.00 [0.00, 0.00] | 0.15 [0.11, 0.19] | 0.83 [0.79, 0.88] | not run | 0.08 [0.04, 0.12] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| Gaussian noise, PSNR-matched (E1) | 0.00 [0.00, 0.00] | 0.15 [0.11, 0.19] | 0.83 [0.78, 0.88] | not run | 0.09 [0.05, 0.13] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| PhotoGuard (reimpl.) | 0.17 [0.12, 0.22] | 0.81 [0.77, 0.85] | 0.96 [0.93, 0.98] | 0.03 [0.01, 0.05] | 0.72 [0.64, 0.79] | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.05] |
| Mist-style (reimpl.) | 0.05 [0.02, 0.08] | 0.63 [0.58, 0.68] | 0.94 [0.91, 0.96] | not run | 0.04 [0.02, 0.07] | 0.00 [0.00, 0.00] | 0.04 [0.01, 0.06] |
| v1 (E3) | 0.00 [0.00, 0.00] | 0.24 [0.19, 0.29] | 0.85 [0.80, 0.90] | 0.02 [0.00, 0.04] | 0.09 [0.05, 0.13] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| v2_cheap (E4) | 0.00 [0.00, 0.00] | 0.22 [0.18, 0.27] | 0.85 [0.80, 0.89] | not run | 0.15 [0.10, 0.21] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| v2_main (E5) | 0.00 [0.00, 0.00] | 0.24 [0.20, 0.29] | 0.88 [0.84, 0.93] | not run | 0.12 [0.07, 0.17] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| v2_antipure (E6) | not run | not run | not run | not run | not run | not run | 0.00 [0.00, 0.01] |
| v2 final | 0.00 [0.00, 0.00] | 0.29 [0.24, 0.35] | 0.87 [0.83, 0.92] | 0.02 [0.00, 0.04] | 0.03 [0.01, 0.06] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |

The target-steered variants for these editors are in `final_readout.md` Table 3 (20.6.1). FINAL_REPORT 7.7 gives v2_final as 0.000 / 0.873 / 0.003 / 0.033 (−0.147*) / 0.000 / 0.020 for SD1.5 s0.3 / s0.75 / FaceID / InstructPix2Pix / inpaint / SDXL, vs v1 0.000 / 0.853 / 0.000 / 0.090 (−0.090*) / 0.000 / 0.020. The table above prints these to 2 decimals.

**`results/tables/per_prompt.md`: PSR by prompt, SD1.5 s0.55, no purifier** (the headline cell has all 3 prompts per photo; test, 100 photos). p0 = "a person in a suit", p1 = "a person at a beach", p2 = "an oil painting of a person".

| method | p0 | p1 | p2 |
|---|---|---|---|
| clean (E0) | 0.08 [0.05, 0.12] | 0.16 [0.11, 0.22] | 0.17 [0.12, 0.23] |
| uniform noise (E1) | 0.09 [0.05, 0.13] | 0.16 [0.11, 0.22] | 0.20 [0.15, 0.25] |
| Gaussian noise, PSNR-matched (E1) | 0.09 [0.05, 0.13] | 0.17 [0.11, 0.22] | 0.18 [0.13, 0.24] |
| PhotoGuard (reimpl.) | 0.82 [0.77, 0.86] | 0.83 [0.78, 0.88] | 0.78 [0.72, 0.83] |
| Mist-style (reimpl.) | 0.63 [0.57, 0.70] | 0.63 [0.57, 0.69] | 0.63 [0.56, 0.70] |
| v1 (E3) | 0.21 [0.15, 0.26] | 0.26 [0.20, 0.32] | 0.25 [0.20, 0.31] |
| v2_cheap (E4) | 0.17 [0.12, 0.22] | 0.21 [0.16, 0.27] | 0.29 [0.23, 0.35] |
| v2_main (E5) | 0.19 [0.14, 0.25] | 0.24 [0.19, 0.30] | 0.30 [0.24, 0.35] |
| v2_antipure (E6) | not run | not run | not run |
| v2 final | 0.26 [0.19, 0.32] | 0.32 [0.26, 0.38] | 0.31 [0.25, 0.37] |

**Messy split:** see 20.5 (`messy.md`).

**`results/tables/clip_i.md`: CLIP-I (ViT-L/14) similarity between the source and the edit, no purifier** (secondary metric; "the CLIP model differs from the one in L_sem on purpose"). Test, 100 photos.

| method | SD1.5 s0.3 | SD1.5 s0.55 | SD1.5 s0.75 | SDXL s0.55 | InstructPix2Pix | SD inpaint (keep-face) | IP-Adapter FaceID |
|---|---|---|---|---|---|---|---|
| clean (E0) | 0.82 [0.81, 0.83] | 0.65 [0.64, 0.66] | 0.53 [0.51, 0.54] | 0.74 [0.72, 0.75] | 0.74 [0.71, 0.77] | 0.82 [0.81, 0.84] | 0.75 [0.73, 0.77] |
| uniform noise (E1) | 0.81 [0.80, 0.83] | 0.64 [0.63, 0.66] | 0.52 [0.50, 0.53] | not run | 0.80 [0.78, 0.83] | 0.82 [0.81, 0.84] | 0.75 [0.73, 0.77] |
| Gaussian noise, PSNR-matched (E1) | 0.81 [0.80, 0.82] | 0.65 [0.63, 0.66] | 0.52 [0.51, 0.54] | not run | 0.79 [0.77, 0.82] | 0.82 [0.81, 0.84] | 0.75 [0.74, 0.77] |
| PhotoGuard (reimpl.) | 0.57 [0.55, 0.58] | 0.50 [0.49, 0.51] | 0.46 [0.45, 0.47] | 0.71 [0.69, 0.72] | 0.50 [0.48, 0.52] | 0.76 [0.75, 0.78] | 0.62 [0.60, 0.64] |
| Mist-style (reimpl.) | 0.66 [0.64, 0.67] | 0.53 [0.52, 0.54] | 0.48 [0.47, 0.50] | not run | 0.76 [0.74, 0.77] | 0.77 [0.75, 0.78] | 0.64 [0.62, 0.66] |
| v1 (E3) | 0.76 [0.74, 0.77] | 0.61 [0.59, 0.62] | 0.51 [0.50, 0.53] | 0.72 [0.71, 0.74] | 0.72 [0.70, 0.74] | 0.80 [0.79, 0.81] | 0.72 [0.70, 0.74] |
| v2_cheap (E4) | 0.76 [0.75, 0.78] | 0.62 [0.60, 0.63] | 0.52 [0.50, 0.53] | not run | 0.68 [0.66, 0.70] | 0.79 [0.78, 0.81] | 0.72 [0.70, 0.73] |
| v2_main (E5) | 0.74 [0.73, 0.76] | 0.61 [0.59, 0.62] | 0.52 [0.50, 0.53] | not run | 0.69 [0.67, 0.71] | 0.79 [0.78, 0.80] | 0.71 [0.69, 0.73] |
| v2_antipure (E6) | not run | not run | not run | not run | not run | not run | 0.71 [0.69, 0.73] |
| v2 final | 0.76 [0.75, 0.78] | 0.59 [0.58, 0.61] | 0.51 [0.49, 0.52] | 0.72 [0.70, 0.73] | 0.77 [0.76, 0.79] | 0.80 [0.79, 0.82] | 0.71 [0.69, 0.73] |

How to read: lower CLIP-I = the edit moved further from the source in CLIP image space. It is not an identity measure.

**Edit-LPIPS**: not reported; see the top of this part.

---

### 20.11 Comparison with PhotoGuard and Mist

**Source:** `e2.md` (paired comparisons; 20.5), `final_readout.md` (Tables 1-3, invisibility block; 20.6), FINAL_REPORT 7.8. **Split:** test, 100 photos, 3 seeds, SD1.5 s0.55 unless stated. **Important qualifier printed in every table footer:** "PhotoGuard and Mist are our reimplementations of the encoder/semantic+textural attacks at our budget" (4/255, the same budget as ours). Mist uses its official target image fetched at run time (HISTORY_FACTS). The older REPORT_v2_draft labels them "(official 1000 it., our eps 4/255)" and "(official 100 it., our eps 4/255)" and says they were "checked against the official code"; FINAL_REPORT 4.4 says baselines are "(official ...)" in part. The generated tables use "reimpl.". Treat them as re-implementations.

**FINAL_REPORT 7.8 table (PSR, 3 decimals).**

| | PSR none | JPEG 75 | JPEG 50 | down-up 0.5 | VAE round trip | DiffPure 0.1 | per-photo invisibility |
|---|---|---|---|---|---|---|---|
| PhotoGuard | 0.810 | 0.183 | 0.167 | 0.137 | 0.590 | 0.673 | fails on 100/100 photos |
| Mist | 0.631 | 0.210 | 0.157 | 0.127 | 0.590 | 0.640 | fails on 100/100 (SSIM), 50/100 (PSNR) |
| v2_tgt_zero_mid | 0.497 | 0.233 | 0.203 | 0.190 | 0.523 | 0.517 | passes on 100/100 |

These equal the corresponding cells of `final_readout.md` Table 1.

**Net protection over clean photos (from `final_readout.md` Table 1, with CI):** PhotoGuard none +0.671 [+0.626,+0.717], JPEG75 +0.080 [+0.040,+0.123], JPEG50 +0.037 [+0.010,+0.063], down-up +0.040 [+0.010,+0.073], VAE +0.467 [+0.390,+0.543], DiffPure0.1 +0.540 [+0.470,+0.610]. Mist none +0.492 [+0.443,+0.542], JPEG75 +0.107 [+0.070,+0.147], JPEG50 +0.027 [-0.003,+0.057], down-up +0.030 [-0.000,+0.060], VAE +0.467 [+0.400,+0.530], DiffPure0.1 +0.507 [+0.437,+0.573]. Versus v1 (Table 2): PhotoGuard +0.570 [+0.524,+0.619] none; Mist +0.391 [+0.348,+0.438] none.

**Mean ArcFace similarity of the edit (headline_sim.md):** PhotoGuard none 0.04 [0.04, 0.05], Mist 0.08 [0.07, 0.09], v1 0.18, v2_final 0.16, clean 0.22. FINAL_REPORT 7.10 (3 decimals, from the audit) gives 0.045 (PhotoGuard), 0.085 (Mist).

**Paired differences of v2_tgt_zero_mid against PhotoGuard and Mist** (FINAL_REPORT 7.8 and REPORT_v2_draft 5.8; **the CIs for these specific pairs are not in any generated table file**, they come from an ad hoc computation described in the reports). The point differences agree with the net-protection columns above (for instance mid 0.130 minus PhotoGuard 0.080 = 0.050 under JPEG 75): mid minus PhotoGuard: JPEG 75 +5.0 [+1.3, +9.0]; JPEG 50 +3.7 [+0.3, +7.0]; down-up 0.5 +5.3 [+2.7, +8.3]; VAE round trip −6.7 [−12.0, −1.0]; DiffPure 0.1 −15.7 [−21.7, −9.0] (percentage points). Mist: mid minus Mist JPEG 75 +2.3 [−1.3, +6.0]; JPEG 50 +4.7; down-up 0.5 +6.3 (CIs for the last two: "both CIs exclude 0", not printed).

**Other editors (final_readout Table 3):** PhotoGuard InstructPix2Pix 0.717 (+0.537 [+0.443,+0.623]*), SD1.5 s0.75 0.957, s0.3 0.170, FaceID 0.027, inpainting 0.000, SDXL 0.027; Mist InstructPix2Pix 0.043 (−0.137 [−0.200,−0.070]*), s0.75 0.940, s0.3 0.047, FaceID 0.037, inpainting 0.000.

**Validation-split screening of both tools** (20 photos): see 20.12.

**Invisibility (20.3):** PhotoGuard PSNR 37.15, SSIM 0.9061, LPIPS 0.467; Mist PSNR 38.00, SSIM 0.9229, LPIPS 0.439.

---

### 20.12 Target-steered variants: validation screening (S24/S25) and test results (S26-S28)

Variants: `v2_tgt_zero` (target weight alpha 0.27), `v2_tgt_zero_mid` (0.55), `v2_tgt_zero_hi` (1.0), `v2_tgt_gray` (gray-image target, validation only); control `v2_cheap` (same cleanups, no target term). Names and weights from REPORT_v2_draft §3.

**Validation screening (S24, S25; `results/S24`, `results/S25`). Split: validation, 20 photos, 3 seeds, 3 prompts (60 edit rows per cell), SD1.5 s0.55. No generated table in `results/tables/` covers this; the numbers come from REPORT_v2_draft §5.8 and FINAL_REPORT 7.9 and were cross-checked here by recomputing the point PSR values from `results/combined/edits.csv` (all matched).** Baselines on validation: clean 8.3%, v1 15.0%. "vs v1" = paired PSR difference, * = CI excludes 0 (the CI bounds are not printed in the reports for these rows). CIs on 20 photos are wide.

| method | PSNR mean (min) | SSIM mean (min) | photos failing the rule | PSR none | vs v1 | PSR JPEG 75 | vs v1 | PSR DiffPure 0.1 | vs v1 |
|---|---|---|---|---|---|---|---|---|---|
| v2_cheap (control) | 41.65 (40.62) | 0.9789 (0.9710) | 0 | 11.7% | -3.3 | 13.3% | +0.0 | 16.7% | -3.3 |
| v2_tgt_zero | 42.07 (40.62) | 0.9819 (0.9728) | 0 | 35.0% | +20.0* | 13.3% | +0.0 | 41.7% | +21.7* |
| v2_tgt_zero_hi | 39.99 (39.16) | 0.9617 (0.9358) | 1 of 20 (SSIM) | 46.7% | +31.7* | 26.7% | +13.3* | 58.3% | +38.3* |
| v2_tgt_gray | 41.94 (40.38) | 0.9825 (0.9744) | 0 | 35.0% | +20.0* | 15.0% | +1.7 | 28.3% | +8.3 |
| PhotoGuard | 37.14 (36.91) | 0.9040 (0.8235) | 20 of 20 | 73.3% | +58.3* | 15.0% | +1.7 | 63.3% | +43.3* |
| Mist | 37.97 (37.57) | 0.9206 (0.8421) | 20 of 20 | 60.0% | +45.0* | 11.7% | -1.7 | 56.7% | +36.7* |

My recomputation (point values only, `results/combined/edits.csv`, split val, SD1.5 s0.55, mean over photos of the per-photo share of hit edits): none / jpeg_q75 / diffpure_0.1: clean 0.0833 / 0.0833 / 0.1000; v1 0.1500 / 0.1333 / 0.2000; v2_cheap 0.1167 / 0.1333 / 0.1667; v2_main 0.2000 / 0.1167 / 0.2500; v2_tgt_zero 0.3500 / 0.1333 / 0.4167; v2_tgt_zero_hi 0.4667 / 0.2667 / 0.5833; v2_tgt_gray 0.3500 / 0.1500 / 0.2833; photoguard 0.7333 / 0.1500 / 0.6333; mist 0.6000 / 0.1167 / 0.5667. They match the table above (the JPEG-75 clean value 8.3% is the same as the no-cleanup 8.3%; the clean diffpure_0.1 value is 10.0%). Note that the validation `invisibility_methods.md` rows give the same PSNR means (v2_tgt_zero 42.07, v2_tgt_zero_hi 39.99, v2_tgt_gray 41.94, v2_cheap 41.65, photoguard 37.14, mist 37.97). The validation `v2_tgt_zero_hi` failing photo: n_below_ssim = 1 (table in 20.3). Wording in FINAL_REPORT 7.9: "The control does not gain, so the gain comes from the target term."

**Test results (S26, S27, S28; test split, 100 photos, 3 seeds, SD1.5 s0.55).** Source: `final_readout.md` Tables 1-2 (20.6.1, CIs given) and FINAL_REPORT 7.9 (3 decimals). Cell = PSR; in brackets paired difference to v1 under the same cleanup:

| method | none | JPEG 75 | down-up 0.5 | VAE round trip | DiffPure 0.1 | DiffPure 0.3 | photos failing the invisibility rule |
|---|---|---|---|---|---|---|---|
| clean | 0.139 | 0.103 | 0.097 | 0.123 | 0.133 | 0.260 | n/a |
| v1 | 0.240 | 0.150 | 0.097 | 0.220 | 0.263 | 0.360 | 0 |
| v2_tgt_zero | 0.418 (+0.178 [+0.141, +0.217]*) | 0.210 (+0.060*) | 0.143 (+0.047*) | 0.433 (+0.213*) | 0.443 (+0.180*) | 0.497 (+0.137*) | 0 |
| v2_tgt_zero_mid | 0.497 (+0.257 [+0.218, +0.298]*) | 0.233 (+0.083*) | 0.190 (+0.093*) | 0.523 (+0.303*) | 0.517 (+0.253*) | 0.600 (+0.240*) | 0 |
| v2_tgt_zero_hi | 0.528 (+0.288 [+0.238, +0.338]*) | 0.267 (+0.117*) | 0.190 (+0.093*) | 0.533 (+0.313*) | 0.610 (+0.347*) | 0.663 (+0.303*) | 4 |

(FINAL_REPORT prints mid none as [+0.218, +0.298] and `final_readout.md` as [+0.218,+0.299]; hi none [+0.238, +0.338] versus [+0.240,+0.340]. Small differences between the two reports and the generated file; the generated file is later.) Also in REPORT_v2_draft 5.8, the extra columns for JPEG 50 and DiffPure 0.2: tgt_zero JPEG 50 0.180 (+0.013), DiffPure 0.2 0.477 (+0.160*); mid JPEG 50 0.203 (+0.037*), DiffPure 0.2 0.570 (+0.253*); hi JPEG 50 0.213 (+0.047*), DiffPure 0.2 0.600 (+0.283*); these equal Table 1/2 of `final_readout.md`.

Ratios stated in FINAL_REPORT 7.9 for `mid` relative to v1: about 2x without cleanup (0.497 vs 0.240), under down-up (0.190 vs 0.097) and under DiffPure 0.1 (0.517 vs 0.263); about 2.4x under the VAE round trip (0.523 vs 0.220); 1.6x under JPEG 75; 1.2x under JPEG 50 (the weakest point). "Protection grows with the target weight (zero < mid < hi)". Per-photo invisibility on test: zero 0/100, mid 0/100, hi 4/100 (SSIM < 0.95; min SSIM 0.9459). The adaptive attacker was not run on these variants. The weights were not tuned on test photos, but **naming `mid` as the best variant that stays invisible was read off the test table** (post hoc among three variants declared in advance; all three are shown).

Other editors and strengths for `v2_tgt_*` (no cleanup): `final_readout.md` Table 3 (20.6.1); InstructPix2Pix, inpainting and SDXL were not run for these variants.

---

### 20.13 Audit numbers (FINAL_REPORT 7.10; REPORT_v2_draft §6)

**Where computed.** There is no script for this in the repository (`scripts/` and `src/eval/` do not contain it). The numbers were produced by an inline Python snippet executed during the working session (a command with description "Audit how much of PSR comes from no-face edits versus low-similarity edits"), reading `results/combined/edits.csv`, filtering split = test, editor = sd15_img2img, purifier = none, strength = 0.55, and thresholding `arcface_sim` at the SD1.5 threshold 0.10054539651520272 from `results/combined/psr_thresholds.json`. The snippet's printed output is reproduced below (it is the same data as FINAL_REPORT 7.10). Unit: edit rows, 900 per method (100 photos x 3 seeds x 3 prompts); "no-face" = `arcface_sim` missing; "face&below" = a face was found and the similarity was below the threshold; PSR = no-face + face&below. The two per-photo/face-only blocks reproduce the figures quoted in the report. This is a point-estimate audit without confidence intervals. The visual-reading part (d) is qualitative, from `results/figures/audit_panel.png` and the other face figures (not reproduced).

| method | rows | PSR | no-face | face&below | mean sim (face found) | median sim |
|---|---|---|---|---|---|---|
| clean | 900 | 0.139 | 0.003 | 0.136 | 0.222 | 0.222 |
| noise_uniform | 900 | 0.149 | 0.002 | 0.147 | 0.213 | 0.210 |
| v1 | 900 | 0.240 | 0.003 | 0.237 | 0.177 | 0.167 |
| v2_main | 900 | 0.243 | 0.004 | 0.239 | 0.178 | 0.175 |
| v2_final | 900 | 0.294 | 0.004 | 0.290 | 0.158 | 0.157 |
| v2_tgt_zero | 900 | 0.418 | 0.014 | 0.403 | 0.129 | 0.124 |
| v2_tgt_zero_mid | 900 | 0.497 | 0.021 | 0.476 | 0.109 | 0.103 |
| v2_tgt_zero_hi | 900 | 0.528 | 0.026 | 0.502 | 0.098 | 0.098 |
| photoguard | 900 | 0.810 | 0.084 | 0.726 | 0.045 | 0.041 |
| mist | 900 | 0.631 | 0.082 | 0.549 | 0.085 | 0.076 |

PSR using only edits where a face was found (drops the no-face rows):

| method | PSR (face-found edits only) | n edits |
|---|---|---|
| clean | 0.136 | 897 |
| v1 | 0.237 | 897 |
| v2_tgt_zero | 0.409 | 887 |
| v2_tgt_zero_mid | 0.486 | 881 |
| v2_tgt_zero_hi | 0.515 | 877 |
| photoguard | 0.792 | 824 |
| mist | 0.598 | 826 |

Fraction of photos whose edits are protected in **every** seed and prompt (all 9 edits of a photo are "no face or below threshold"; 100 photos):

| method | share of fully protected photos |
|---|---|
| clean | 0.000 |
| v1 | 0.020 |
| v2_tgt_zero_mid | 0.050 |
| photoguard | 0.350 |

FINAL_REPORT 7.10 text (same numbers, percent): no-face edits 0.3% (clean), 0.3% (v1), 1.4-2.6% (target variants), 8.4% (PhotoGuard), 8.2% (Mist); with those edits removed, PSR is 13.6% / 23.7% / 40.9% / 48.6% / 51.5% / 79.2% / 59.8% (clean, v1, tgt_zero, tgt_zero_mid, tgt_zero_hi, PhotoGuard, Mist); mean ArcFace similarity of edits (face found) 0.222 (clean) → 0.177 (v1) → 0.158 (v2_final) → 0.129 / 0.109 / 0.098 (target zero / mid / hi) → 0.085 (Mist) → 0.045 (PhotoGuard); per-photo fully protected: 0.0% (clean), 2.0% (v1), 5.0% (mid), 35.0% (PhotoGuard). Qualitative statements from the same section: protected edits of the target-steered variants and PhotoGuard are often degraded or restyled (cartoon, sketch or sepia looks, texture artefacts, sometimes a different-looking face), not a believable different person; v1's edits look almost like clean edits; some protected edits are still recognisable to a person. "No human study was run."

Related number from E0 (20.1): clean SD1.5 s0.55 edits have mean ArcFace similarity 0.223 [0.208, 0.237] (the audit's 0.222 uses the per-edit mean over edits with a face, not the per-photo mean).

The per-edit no-face rate for other editors is in `e0.md` (InstructPix2Pix clean edits: no-face rate 0.18 [0.12, 0.24]) and as `face_rate` in the coverage map (for example `photoguard` ip2p 0.71, `v2_final` ip2p none 0.97, `v1` ip2p none 0.92, `clean` ip2p none 0.82).

---

### 20.14 Optuna results (S06)

**Sources:** `reports/optuna_trials_S06.csv` (all 16 trials), `reports/optuna_confirm_S06.json` (confirm step), `config/parts/final.yaml` (chosen config), `scripts/tune_optuna.py` docstring (design). **Design (from the script):** Phase 6 protection-aware tuning of `v2_final`. 16 trials x **10 validation images**, proxy objective (20 PGD iterations, K = 2 EoT draws, 1 edit seed, editors SD1.5 img2img s0.55 + IP-Adapter FaceID, purifiers {none, jpeg_q75, diffpure_t0.1}); objective = **mean PSR over the 6 editor x purifier cells** (higher is better); constraints (TPE `constraints_func`): **mean** PSNR >= 38 and **mean** SSIM >= 0.95 of the protected uint8 images vs the clean images (SSIM there is an 11x11 Gaussian-window SSIM on [0,1] RGB, which may differ slightly from the harness's skimage SSIM). The top 3 feasible trials plus the hand-set default (trial 0) are re-run on **all 20 validation images** with the final protection settings (confirm step). Searched: alpha, eta (log-uniform 0.05-1), beta = L_sem weight in [0, 1.5], gamma (log-uniform 20-150), antipure on/off and one shared anti-purification weight in [0.1, 1.0]. Not searched: eps, lambda_id (0), K, iterations. Two workers (w0, w1) shared an SQLite study (a creation race had to be fixed). Session S06 hit its 12-hour timeout; per FINAL_REPORT it ended with 16 trials (matching the CSV). No test image or identity was used.

**All 16 trials (`optuna_trials_S06.csv`).** value = mean PSR over the 6 cells on 10 photos (multiples of 1/60 because 6 cells x 10 photos); PSNR/SSIM are means over the 10 photos, min = worst photo; "constraints" are the two signed slack values printed by the study (negative = feasible); all 16 trials have `feasible = True` and `state = COMPLETE`. alpha/eta/beta/gamma are loss weights; antipure = anti-purification terms on; ap_w = shared w_hf = w_mono weight (empty when antipure is off). Time = `t_protect_s` / `t_total_s` in seconds. Trial 0 is the hand-set default (alpha 0.27, eta 0.27, beta 0.9, gamma 60.5, antipure off).

| trial | value | alpha | eta | beta | gamma | antipure | ap_w | PSNR mean (min) | SSIM mean (min) | worker | t_protect_s / t_total_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.03333 | 0.27 | 0.27 | 0.9 | 60.5 | False | – | 42.848 (41.849) | 0.97753 (0.96565) | w0 | 853.58 / 1106.72 |
| 1 | 0.06667 | 0.09646 | 0.17231 | 1.46495 | 23.9234 | True | 0.27833 | 42.222 (41.448) | 0.97419 (0.96260) | w1 | 1004.24 / 1255.01 |
| 2 | 0.05 | 0.09646 | 0.17231 | 1.46495 | 23.9234 | True | 0.27833 | 42.309 (41.593) | 0.97497 (0.96454) | w0 | 928.04 / 1110.04 |
| 3 | 0.03333 | 0.76761 | 0.26624 | 1.17484 | 36.7888 | False | – | 40.645 (40.163) | 0.95980 (0.94507) | w1 | 792.48 / 977.98 |
| 4 | 0.06667 | 0.76761 | 0.26624 | 1.17484 | 36.7888 | False | – | 40.675 (40.171) | 0.96014 (0.94604) | w0 | 783.04 / 965.43 |
| 5 | 0.08333 | 0.72812 | 0.05776 | 0.66760 | 62.0471 | True | 0.93103 | 40.987 (40.528) | 0.96312 (0.94938) | w1 | 940.30 / 1125.99 |
| 6 | 0.06667 | 0.28310 | 0.36344 | 0.18674 | 33.5848 | True | 0.10273 | 41.499 (40.795) | 0.96806 (0.95392) | w0 | 928.92 / 1110.83 |
| 7 | 0.01667 | 0.19797 | 0.12674 | 0.26093 | 109.3732 | True | 0.99349 | 44.006 (42.964) | 0.98324 (0.97613) | w1 | 940.35 / 1125.89 |
| 8 | 0.06667 | 0.57880 | 0.73487 | 0.90453 | 71.1852 | True | 0.99819 | 41.476 (40.909) | 0.96774 (0.95455) | w0 | 930.31 / 1111.83 |
| 9 | 0.06667 | 0.10449 | 0.05032 | 0.74321 | 23.1705 | True | 0.67522 | 41.903 (40.825) | 0.97144 (0.95858) | w1 | 939.15 / 1124.52 |
| 10 | 0.1 | 0.92253 | 0.05998 | 0.28191 | 56.5016 | False | – | 40.906 (40.363) | 0.96248 (0.94812) | w0 | 781.61 / 963.53 |
| 11 | 0.08333 | 0.48438 | 0.09595 | 1.23735 | 74.2281 | True | 0.65783 | 41.809 (41.139) | 0.97046 (0.95693) | w1 | 940.27 / 1125.82 |
| 12 | 0.05 | 0.79884 | 0.06025 | 0.18155 | 57.7950 | False | – | 41.110 (40.440) | 0.96443 (0.94972) | w0 | 781.35 / 963.22 |
| 13 | 0.08333 | 0.81494 | 0.08964 | 0.57308 | 39.3281 | True | 0.61111 | 40.431 (40.132) | 0.95718 (0.94337) | w1 | 940.75 / 1125.96 |
| 14 | 0.03333 | 0.50347 | 0.05699 | 0.54517 | 41.3134 | False | – | 41.279 (40.605) | 0.96602 (0.95124) | w0 | 781.34 / 963.35 |
| 15 | 0.05 | 0.93349 | 0.11668 | 0.50180 | 86.7050 | False | – | 41.425 (40.765) | 0.96740 (0.95450) | w1 | 791.22 / 976.22 |

(CSV rounding to 5 decimals here. Trials 1 and 2 have the same parameters (the second was enqueued/duplicated by the study) and gave values 0.0667 and 0.05; trials 3 and 4 likewise 0.0333 and 0.0667. These duplicates reflect proxy noise: same parameters, different values. The CSV contains the per-cell breakdown in `user_attrs_per`; for trial 5: none|sd15 0.3, none|faceid 0.0, jpeg_q75|sd15 0.0, jpeg_q75|faceid 0.0, diffpure_0.1|sd15 0.2, diffpure_0.1|faceid 0.0.) Trial datetimes: study start 2026-10-14 23:54:49, last completion 2026-10-15 02:22:15 (the whole 16-trial search ran in about 2.5 hours of wall time; the session-level 12-hour timeout was hit later, in the confirm step, per the history notes). Best proxy value: trial 10 (0.1). All trials are "feasible" under the mean-based constraint, even though several have a worst-photo SSIM below 0.95 (min SSIM below 0.95 for trials 3, 4, 5, 10, 12, 13: 0.9451, 0.9460, 0.9494, 0.9481, 0.9497, 0.9434).

**Confirm step (`optuna_confirm_S06.json`): top 3 feasible trials + hand-set, re-run on 20 validation images with final settings.** psr = mean over the 6 cells.

| candidate | PSR (mean of 6 cells) | none/SD1.5 | none/FaceID | jpeg_q75/SD1.5 | jpeg_q75/FaceID | diffpure_0.1/SD1.5 | diffpure_0.1/FaceID | PSNR mean (min) | SSIM mean (min) | params | t_protect_s / t_total_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| trial_10 | 0.05 | 0.15 | 0.0 | 0.0 | 0.0 | 0.15 | 0.0 | 40.508 (40.184) | 0.95461 (0.92231) | alpha 0.92253, eta 0.05998, beta 0.28191, gamma 56.5016, antipure False | 3186.91 / 3675.39 |
| **trial_5** | **0.06667** | 0.2 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 | 40.673 (39.762) | 0.95711 (0.92371) | alpha 0.72812, eta 0.05776, beta 0.66760, gamma 62.0471, antipure True, ap_w 0.93103 | 3854.64 / 4222.03 |
| trial_11 | 0.05 | 0.15 | 0.0 | 0.0 | 0.0 | 0.15 | 0.0 | 41.371 (40.124) | 0.96499 (0.94096) | alpha 0.48438, eta 0.09595, beta 1.23735, gamma 74.2281, antipure True, ap_w 0.65783 | 3855.35 / 4222.35 |
| hand_set | 0.04167 | 0.15 | 0.0 | 0.05 | 0.0 | 0.05 | 0.0 | 42.151 (41.502) | 0.97168 (0.95831) | alpha 0.27, eta 0.27, beta 0.9, gamma 60.5, antipure False, ap_w 0.5 | 3182.00 / 3549.09 |

All four are marked `feasible: true`. **Chosen:** trial 5, written as the candidate for `config/parts/final.yaml` ("trial_5; confirmed on 20 val images; hand-set psr=0.041666666666666664"); final.yaml: alpha 0.7281249466629797, eta 0.05776266977639212, beta 0.6676014641453782, gamma 62.04711204188804, lambda_id 0.0, anti_purify enabled with w_hf = w_mono = 0.9310312738392569. This is `v2_final` (antipure on, SDXL ensemble per FINAL_REPORT 4.4).

Statements in the project's own words (FINAL_REPORT section 10): "Optuna on 10 photos with a proxy objective, where the proxy's best trial (0.10) fell to 0.05 on the 20-photo confirm and tuned vs hand-set is not separable (0.067 vs 0.042)." Figure: `results/figures/optuna_history.png` (20.17). Note that in the confirm step trial 5's mean PSNR/SSIM satisfy the mean rule but its worst-photo values do not (SSIM min 0.9237), consistent with the later finding that `v2_final` breaks SSIM >= 0.95 on 29/100 test photos.

---

### 20.15 Claims table (`results/tables/claims.md`, `claims.csv`, `claims_long.csv`)

The project's own table is **titled "Claim numbers: v2_final vs noise controls and vs v1 (SD1.5 s0.55)"**. It has **no status/verdict column**: it only lists numbers for a template sentence. Test split, 100 pairs per row, SD1.5 s0.55, 3 seeds, one sentence template: "Under the stated threat model, DiffShield lowers ArcFace identity retention of edits by A points versus a noise control and B points over v1, including after [held-out purifiers]; an adaptive attacker reduces PSR to C." Definitions: "ArcFace sim reduction = mean over images of sim(reference edit) - sim(v2_final edit), in points (x100)"; the paired differences use the usual paired image-bootstrap CI and two-sided Wilcoxon p, not corrected for multiple comparisons. For each row the "reference" is the named comparison.

| purifier | v2_final vs | pairs | ArcFace sim reduction, points [95% CI] | p | PSR gain [95% CI] |
|---|---|---|---|---|---|
| none | noise_uniform | 100 | 5.5 [4.9, 6.2] | <0.001 | +0.15 [+0.11, +0.18] |
| none | noise_gauss_psnr | 100 | 5.7 [5.0, 6.4] | <0.001 | +0.15 [+0.12, +0.18] |
| none | v1 | 100 | 1.9 [1.4, 2.5] | <0.001 | +0.05 [+0.02, +0.09] |
| jpeg_other | v1 | 100 | -0.1 [-0.6, 0.4] | 0.910 | +0.01 [-0.02, +0.05] |
| downup_other | v1 | 100 | -0.0 [-0.5, 0.5] | 0.940 | +0.02 [-0.01, +0.04] |
| diffpure_0.1 | v1 | 100 | 1.3 [0.6, 2.0] | <0.001 | +0.03 [-0.02, +0.09] |
| diffpure_0.2 | v1 | 100 | 1.0 [0.3, 1.7] | 0.005 | +0.03 [-0.02, +0.09] |
| diffpure_0.3 | v1 | 100 | 1.4 [0.7, 2.1] | <0.001 | +0.05 [+0.00, +0.11] |

Reading (neutral): the template's value C (the adaptive-attacker PSR) is not in this table; it is in `e8.md` (v2 final PSR under adaptive AE 0.11 [0.07, 0.15]; FINAL_REPORT 10.7%). The table covers `v2_final` only; the target-steered variants have no row in `claims.md`. Cross-check with 20.4/20.7: v2_final vs noise_uniform PSR +0.15 [+0.11, +0.18], sim -0.055; vs v1 PSR +0.05 [+0.02, +0.09], sim -0.019 (these agree to rounding with 5.5 and 1.9 points).

**The project's goal-level verdict table (FINAL_REPORT section 9, not generated; this is the closest thing to claim status).**

| Goal | Outcome (project's wording) | Verdict (project's wording) |
|---|---|---|
| G1 invisibility | v1, v2_main, v2_cheap, `tgt_zero` and `tgt_zero_mid` pass the per-photo rule on all 100 photos; `v2_final`, antipure, eps = 8, PhotoGuard, Mist do not | achieved for the recommended variants |
| G2 protection | clearly above clean and noise; v1 +10 points, `tgt_zero_mid` +36 points (about 2x v1), all with intervals excluding 0, for SD1.5 image-to-image at moderate strength | achieved in one regime; modest elsewhere |
| G3 robustness to cleanup | survives DiffPure and the VAE round trip; reduced by JPEG and resizing; removed by an adaptive attacker (tested on v1 and `v2_final`) | partial |
| G4 generality | does not protect FaceID, inpainting, SDXL or strength 0.3 | not achieved |
| G5 fair evaluation | controls, pairing, fixed splits, held-out cleanups, adaptive test, existing-tool comparison, intervals everywhere | achieved |
| G6 reproducibility | one config, per-image seeds, logged environments, 377 tests, hash-stamped bundles, one command per analysis step | achieved |

One-sentence verdict in FINAL_REPORT section 9: v2 as originally designed (cleanups and extra terms) added little over v1; adding a target-steered term produced a measurable, supported improvement (about twice v1's protection at invisible levels for SD1.5 image-to-image editing), with clear limits (JPEG, adaptive attackers, FaceID/inpainting/SDXL, human perception untested).

---

### 20.16 Number index: the most citable numbers

All SD1.5 img2img s0.55, test split, 100 photos, 3 seeds, no cleanup, unless the row says otherwise. "net" = minus clean photos under the same cleanup.

| # | quantity | value | source file |
|---|---|---|---|
| 1 | SD1.5 PSR threshold (95th pct impostor sim) | 0.10054539651520272 (0.101) | `results/combined/psr_thresholds.json`; `e0.md` |
| 2 | Pooled PSR threshold | 0.1034364047631 (0.103) | `results/combined/psr_thresholds.json` |
| 3 | Clean-edit ArcFace sim, SD1.5 s0.3 / s0.55 / s0.75 | 0.522 / 0.223 / 0.039 | `e0.md` |
| 4 | Clean-edit PSR, s0.55 / s0.75 | 0.14 [0.10, 0.18] / 0.84 [0.79, 0.89] | `e0.md` |
| 5 | v1 budget confirmed | 4/255 (max pixel change 4 in 70/70 photos) | `reports/PHASE_0.md` |
| 6 | v1 reproduction PSNR (published PNG), n = 70 | 42.171 ± 0.436 [42.070, 42.272] dB (v1 original 41.911) | `reports/PHASE_0.md` |
| 7 | v1 cost | 45.6 ± 1.2 GPU-s per image (T4) | `reports/PHASE_0.md` |
| 8 | PSR clean photos | 0.139 (0.14 [0.10, 0.18]) | `final_readout.md`; `headline.md` |
| 9 | PSR uniform noise / Gaussian noise (4/255) | 0.15 [0.11, 0.19] / 0.15 [0.11, 0.19]; vs clean +0.01 [-0.01, +0.03] | `headline.md`; `e1.md` |
| 10 | PSR v1 | 0.240 (0.24 [0.19, 0.29]); net +0.101 [+0.072, +0.130] | `final_readout.md` |
| 11 | v1 minus uniform noise (PSR) | +0.09 [+0.06, +0.12], p < 0.001 | `paired_vs_noise.md` |
| 12 | PSR v2_main | 0.243 (net +0.104 [+0.076, +0.134]); minus v1 +0.003 [-0.018, +0.024] | `final_readout.md` |
| 13 | PSR v2_final | 0.29 [0.24, 0.35] (0.294 in the audit); minus v1 +0.05 [+0.02, +0.09] | `headline.md`; `paired_vs_v1.md`; `claims.md` |
| 14 | PSR v2_tgt_zero / mid / hi | 0.418 / 0.497 / 0.528 | `final_readout.md` |
| 15 | mid minus v1 (PSR) | +0.257 [+0.218, +0.299] | `final_readout.md` |
| 16 | mid net over clean | +0.358 [+0.311, +0.403] | `final_readout.md` |
| 17 | mid vs v1 under JPEG75 / down-up 0.5 / VAE / DiffPure0.1 | 0.233 vs 0.150 / 0.190 vs 0.097 / 0.523 vs 0.220 / 0.517 vs 0.263 | `final_readout.md` |
| 18 | mid under JPEG50 (weakest point) | 0.203 (+0.037 [+0.000, +0.073] vs v1) | `final_readout.md` |
| 19 | PhotoGuard PSR none / JPEG75 / DiffPure0.1 | 0.810 / 0.183 / 0.673 | `final_readout.md` |
| 20 | Mist PSR none / JPEG75 / DiffPure0.1 | 0.631 / 0.210 / 0.640 | `final_readout.md` |
| 21 | PhotoGuard invisibility | PSNR 37.15, SSIM 0.9061; 100/100 photos break PSNR and SSIM rule | `invisibility_methods.md` |
| 22 | Mist invisibility | PSNR 38.00, SSIM 0.9229; 50/100 break PSNR, 100/100 break SSIM | `invisibility_methods.md` |
| 23 | v1 invisibility | PSNR 42.42 (min 41.17), SSIM 0.9763 (min 0.9587), 0 violations | `invisibility_methods.md`; `final_readout.md` |
| 24 | v2_tgt_zero_mid invisibility | PSNR 40.91 (min 39.75), SSIM 0.9728 (min 0.9615), 0 violations | `invisibility_methods.md` |
| 25 | v2_tgt_zero_hi invisibility | PSNR 40.05, SSIM 0.9628, 4/100 below SSIM 0.95 | `invisibility_methods.md` |
| 26 | v2_final invisibility | PSNR 40.46 (min 38.76), SSIM 0.9549 (min 0.9197), 29/100 below SSIM 0.95 | `invisibility_methods.md`; `gate_INV.md` |
| 27 | eps8 invisibility | PSNR 35.98, SSIM 0.8958; 98/100 below 38 dB, 96/100 below 0.95 | `invisibility_methods.md` |
| 28 | Gate E3 | BARELY DROPS; mean relative drop 0.16 | `gates.md`; `gate_E3.md` |
| 29 | Gate E4 | FAIL; ratios 1.11 / 1.04 / 1.48 / 0.97 (need >= 2) | `gate_E4.md` |
| 30 | Gate E5 | PASS (robust); v2_main 0.28 vs 0.24, ratio 1.15 [1.00, 1.32] | `gate_E5.md` |
| 31 | Gate E6 | GO on 1 of 3 (diffpure_0.3 +0.07 [+0.02, +0.11]) | `gate_E6.md` |
| 32 | Ablation vis only / den only | 0.25 [0.20, 0.31] / 0.12 [0.08, 0.17] (den-only PSNR 50.76) | `e7.md`; `invisibility_methods.md` |
| 33 | Budget sweep PSR eps 2 / 4 / 8 | 0.17 / 0.29 / 0.38 (PSNR 45.34 / 40.46 / 35.98) | `e9.md` |
| 34 | Adaptive AE: PSR clean / v1 / v2_final | 0.09 [0.06, 0.14] / 0.11 [0.07, 0.15] / 0.11 [0.07, 0.15] (FINAL_REPORT: 9.3 / 11.0 / 10.7%) | `e8.md`; `FINAL_REPORT.md` 7.6 |
| 35 | Adaptive AE training | 120 pairs, 35496 steps, 60 min, AE-vs-clean PSNR 45.12 dB | `results/S15/S15/adaptive_ae/ae_final_train_log.json` |
| 36 | Other editors (v1/v2_final): FaceID, inpaint, SDXL, s0.3 | all ~0.00 to 0.02 | `per_editor.md`; `final_readout.md` |
| 37 | InstructPix2Pix PSR clean / v1 / v2_final / PhotoGuard | 0.18 / 0.09 / 0.03 / 0.72 | `per_editor.md` |
| 38 | Audit: no-face share clean / mid / PhotoGuard | 0.003 / 0.021 / 0.084 | FINAL_REPORT 7.10 (inline audit output, 20.13) |
| 39 | Audit: fully protected photos clean / v1 / mid / PhotoGuard | 0.000 / 0.020 / 0.050 / 0.350 | FINAL_REPORT 7.10 (20.13) |
| 40 | Audit: mean ArcFace sim of edits clean / v1 / mid / PhotoGuard | 0.222 / 0.177 / 0.109 / 0.045 | FINAL_REPORT 7.10 (20.13) |
| 41 | Optuna: trials; best proxy value; confirm chosen | 16 trials; 0.1 (trial 10); trial 5 (0.0667 vs hand-set 0.0417) | `reports/optuna_trials_S06.csv`; `optuna_confirm_S06.json` |
| 42 | Validation screening mid-family: none PSR (tgt_zero / tgt_zero_hi / control v2_cheap) | 0.350 / 0.467 / 0.117 (clean 0.083, v1 0.150; n = 20) | REPORT_v2_draft 5.8 (recomputed in 20.12) |

---

### 20.17 Figures in `results/figures/`

The folder holds PNG and PDF versions of each figure (generated by `scripts/make_figures.py` and `scripts/make_pareto_figure.py`; `figures_manifest.json` records them, marking each `synthetic: false` and listing `human_study` as skipped because E10 was not run). The `audit_panel.png` is not in the manifest (it was made by an inline command, see 20.13). Okabe-Ito colour-blind-safe palette; hollow markers mark methods that violate the per-photo invisibility rule.

| file(s) | what it plots | faces? |
|---|---|---|
| `pareto_psr_vs_psnr.png/.pdf` | Headline: PSR (no cleanup, SD1.5 s0.55) against PSNR of the protected photo for every method on the 100 test photos; hollow marker = breaks the rule on at least one photo; error bars = 95% bootstrap CI over photos | no |
| `psr_vs_psnr.png/.pdf` | E9 epsilon sweep: PSR against PSNR for eps 2/4/8 | no |
| `psr_by_purifier.png/.pdf` | PSR by method and purifier (PLOT_PURS: none, jpeg_q75, jpeg_q50, downup_0.5, vae_rt, diffpure_0.1, diffpure_0.2) | no |
| `per_editor.png/.pdf` | PSR by method and editor/strength | no |
| `ablation.png/.pdf` | E7: PSR of the ablation variants by purifier | no |
| `sim_distributions.png/.pdf` | Distributions of the ArcFace similarity of edits by method | no (plots only) |
| `optuna_history.png/.pdf` | Optuna trial history; infeasible trials marked | no |
| `before_after.png/.pdf` | Source photo, clean edit and protected edits side by side for example CelebA-HQ test photos | **yes (real face images; describe only, do not reproduce)**. The history notes say `before_after.pdf` and `failure_gallery.pdf` are still on the public master branch (found 2026-10-25, not yet removed). |
| `failure_gallery.png/.pdf` | Gallery of failure cases (protected edits that remain similar to the source) for selected methods | **yes (real face images)** |
| `failure_cases.csv` | Not a figure: 12 data rows (method, image_id, sim, psr, no_face, clean_edit_sim, sim_drop, threshold); the CelebA-HQ image ids of the failure cases plotted in the gallery | no pixels (ids only) |
| `audit_panel.png` | 6 test photos (first six common ids) x 5 columns: source, clean edit, v1 edit, `v2_tgt_zero_mid` edit, PhotoGuard edit (SD1.5 s0.55, seed 1, prompt rotation per file); used for the visual part of the audit | **yes (real face images)** |

Not produced: `human_study` figure (E10 not run). FINAL_REPORT 7.10 lists the face-free figures as `pareto_psr_vs_psnr.png`, `psr_vs_psnr.png`, `psr_by_purifier.png`, `per_editor.png`, `ablation.png`, `sim_distributions.png`, `optuna_history.png`. Face-containing images stay local by the project's rules; the face-derived edit images themselves are never committed.

---

### 20.18 Discrepancies and open uncertainties found while assembling this part

1. **Bootstrap resamples**: tables say 10000; REPORT_v2_draft §4 says 2000. Small interval differences follow (for example v2_main − v1: FINAL_REPORT 7.4 [−1.8, +2.6] versus `final_readout.md` [−0.018, +0.024]; v2_final − v1 upper bound 8.6 in the draft summary, 8.7 elsewhere; mid − v1 none [+0.218, +0.298] versus [+0.218, +0.299]). Prefer the generated tables.
2. **Optuna photo count**: FINAL_REPORT 7.1 and the draft say the limits were checked "on the average over its 20 photos", but the search itself used 10 validation photos (FINAL_REPORT 5 and 10, script docstring, trial values are multiples of 1/60); the 20-photo set is the confirm step. Both statements refer to the mean-not-per-photo issue.
3. **Prompts per image** footers: E4/E5/E6 say "up to 1", other tables "up to 3". The row counts in `coverage.md` (900 rows for test SD1.5 s0.55 "none" cells, 300 for cleanup cells) suggest the "none" cells use all 3 prompts per photo and the cleanup cells one rotated prompt (inference; not stated in the files). The messy footnote says "one rotated prompt per image" while its coverage rows (60 per cell = 20 photos x 3 seeds) fit one prompt per photo.
4. **PhotoGuard/Mist wording**: draft says "official 1000 it." and "official 100 it."; generated tables say "reimpl."; HISTORY_FACTS says re-implementations checked against official code. Cited as re-implementations.
5. **Validation screening has no generated table**; numbers are from the reports (point values re-verified from `edits.csv`; CIs for those rows not available).
6. **Audit script not in the repo**; numbers exist in FINAL_REPORT 7.10 and in the session's printed output (20.13). No CIs.
7. **mid − PhotoGuard/Mist paired CIs** (FINAL_REPORT 7.8) are not in any generated table file.
8. **Adaptive AE**: trained on `v2_final` protected images (job definition), so the v1 and clean-photo results are a transfer; the log's `val.n = 8` versus 12 validation stems is unexplained; FINAL_REPORT says "output PSNR vs the clean photo 45.1 dB" (matches 45.124).
9. **`final_readout.md` has no `v2_cheap`/`v2_final` rows**; their net-protection rows exist only in FINAL_REPORT 7.3.
10. **Two invisibility sources** (protect.csv versus edit-rows `inv_*` columns) differ slightly (20.3).
11. **Rounding quirks** in the generated text: stars on intervals that print a bound of +0.000; `p` and CI sometimes disagree in sign of significance (Wilcoxon versus bootstrap CI of the mean); the "images A>B" column is not defined in the footnote.
12. **Claims table has no status column**; the closest "status" statements are the gates (20.6.2) and FINAL_REPORT section 9 (20.15).
13. **Missing planned cells**: `abl_no_sem`, `abl_no_pur`, `eps6`, `gridpure`, SDXL for noise/mist/v2_cheap/v2_main, adaptive attacker on target variants, human study, edit-LPIPS at scale. `results/combined/coverage_vs_plan.csv` lists planned-but-missing cells (not opened for this part).
14. The Phase-0 source files under `results/J01/results/J01/` were not individually opened; the PHASE_0.md copy is used.


---

## 21. Project evolution

Sources: `git log` of `C:\Users\karth\Documents\DiffShield-wt\integration` (branch `v2/release`, 130 commits by `cootot`, 5 by Varun Raj V, 4 each by Destroyer795 and NarainBK, 2 by venkatesh21bit per `git shortlog -sn --all`), `reports/PHASE_0.md`, `reports/FINAL_REPORT.md`, `docs/EVAL_PLAN_FINAL.md`, `docs/SESSION_HANDOFF.md`, `docs/V2_PLAN.md`, `kaggle/jobs.yaml`, and `HISTORY_FACTS.md`. Dates are git author dates; the machine clock jumped several times during the work, so only git dates are used. Times of day are not used.

### 21.1 Who committed what (measurable facts only; roles are NOT recorded)
| git author | commits (all branches) | what the log shows |
|---|---|---|
| cootot | 130 | every commit from 2026-08-16 (notebook "trial 1/2/3", 2026-08-16 to 08-19) and every v2 commit (2026-10-14 to 2026-10-23) |
| Varun Raj V | 5 | notebooks 2026-07-29 (`fyp-diffshield abg.ipynb`), 2026-08-05 (hyper-parameter range sweep), 2026-08-09 (target concept optimisation and SSIM preservation), 2026-08-15 (validation pipeline calibration), and README init 2026-09-06 |
| Destroyer795 | 4 | "9th and 10th try", "11th try" (08-07), "12th try" (08-12), "13th try" (08-14) "to optimize weights" |
| NarainBK | 4 | `phase-1.ipynb` family (07-24, 07-28, 08-01) and removal of redundant ipynb files (09-06) |
| venkatesh21bit | 2 | `2b4fc71` "implementation of PGD pipeline with fixed parameters" (2026-07-18, first commit in the log), `25d1fd1` "main.py updated" (07-21) |

`docs/SESSION_HANDOFF.md` states that the user is a "final-year student, team of 5" and names a person; `DiffShield-site/data/team.js` lists the five git authors with empty `role` fields. No file records who did what. A paper author must obtain roles and affiliations from the team (see section 29).

### 21.2 Phases of the project
**Phase A, v1 (2026-07-18 to 2026-09-06).** PGD pipeline with fixed parameters, then notebooks for phase 1 evaluation, weight-optimisation "tries" 9 to 13, hyper-parameter range sweep, target-concept (potted plant) and SSIM preservation, validation calibration, trials 1 to 3. A v1 paper PDF sits in the repository root (`main.pdf`, `main (3).pdf`); it is the v1 write-up, not v2 work. Method of v1 (from `reports/FINAL_REPORT.md` section 3): PGD 40 iterations, EoT over random transforms, loss `alpha*L_vis - beta*L_sem - gamma*L_str`, Optuna weights alpha 0.2699, beta 0.9024, gamma 60.53.

**Phase B, v2 planning (2026-10-14).** Commits `11e4da3` (v2 handoff brief, implementation plan, v1 reference notebooks), `9d70a9a` (48-hour plan), `0f03da2` (plan remapped to 5 accounts). The plan (`docs/HANDOFF_v2.txt`, `docs/V2_PLAN.md`) prescribed phases 0 to 8 and experiments E0 to E10; `reports/FINAL_REPORT.md` section 5 says the original brief was for about ten weeks of sequential work, compressed to a few days.

**Phase C, v2 build (2026-10-14).** Three v1 runtime patches turned into commits (`36c9155`, `a27d744`, `03ea8a4`, `8d0e6e2`; tag `v1.0`; tag `v1-unpatched` = `f8e8329`), repo re-layout (`4e05435`), then parallel worktree branches `v2/protect`, `v2/eval`, `v2/baselines`, `v2/analysis` merged into an integration branch (`87f40bc`, `b8e8688`, `5f493a4`, `ef0ebe4`, `1900d32`). Phase 0 on Kaggle (J00, J01): budget confirmed 4/255, v1 reproduced on 70 photos within the tolerance the user accepted (`caac5a1`).

**Phase D, smoke sessions and engineering fixes (2026-10-14 to 10-15).** S03 (splits), S04/S04t/S04u (protection smoke and speed probes), S05 rounds 1 to 3 (edit and purifier smoke). Smoke numbers are never reported. Section 21.4 lists the bugs found.

**Phase E, planned full-scale runs (2026-10-15 to 10-21).** S06 (Optuna), S07 (clean reference edits and thresholds), S08 (v1), S09 (v2_main), S10 to S12, S15 to S19 (see `kaggle/jobs.yaml`), S20 reruns. By `docs/EVAL_PLAN_FINAL.md` (written 2026-10-20) the measured v1-vs-clean signal was PSR 13.9% to 24.0%, and "everything about v2 is still unmeasured".

**Phase F, diagnosis and pivot to target-steered variants (2026-10-20 to 10-22).** After `v2_main` showed PSR 24.3% (equal to v1, +0.3 [-1.8, +2.6]) and the ablations showed that the encoder-shift term carries the effect, the target-steered term was designed (`FINAL_REPORT.md` sections 4.2, 8.3): `L_vis` has no direction, so a fixed destructive target latent (zero latent, or latent of a gray image) is pulled toward inside the cleanup loop (the idea of PhotoGuard and Mist). Commit `6b40e51` (2026-10-20) added `v2_tgt_zero`, `_zero_hi`, `_gray`, a SCREEN tier on the 20 validation photos and sessions S24/S25. The variant `v2_tgt_zero_mid` is named in the later commit `42cb3fe` (2026-10-21) together with the TGT tier and S26 to S28. The screening used validation photos only; the three declared variants were then run on the 100 test photos (`2079157`, 2026-10-22).

**Phase G, comparisons, audit, deliverables (2026-10-22 to 10-23).** PhotoGuard (S13) and Mist (S21) at 100 test photos (`e286b74`); audit paragraph (`4833a46`); `FINAL_REPORT.md/.pdf` (`50c2197`, `f1e7774`); hash map after history rewrite (`94ffbe3`); 14-slide deck (`22d3f8c`, last commit in the log, 2026-10-23). A static website exists in the separate folder `C:\Users\karth\Documents\DiffShield-site` (its `README.md` says every number on the site must appear in `FINAL_REPORT.md`).

### 21.3 Dated timeline
| date (git) | commit(s) | event |
|---|---|---|
| 2026-07-18 | `2b4fc71` | first commit: PGD pipeline with fixed parameters (v1), by venkatesh21bit |
| 2026-07-21 | `25d1fd1` | `main.py` updated |
| 2026-07-24 to 08-01 | `e947a94`, `0f94558`, `90743af` | phase-1 notebooks added (NarainBK) |
| 2026-07-29 to 08-15 | Varun Raj V notebooks | range sweep, target concept + SSIM preservation, calibration |
| 2026-08-02 to 08-14 | Destroyer795 commits | tries 9 to 13 to optimise weights |
| 2026-08-16 to 08-19 | `d9d153a`, `632565b`, `231f1ef` | trials 1 to 3 notebooks (cootot) |
| 2026-09-06 | `20b4c91`, `d6601bc` | notebook clean-up; README initialised |
| 2026-10-14 | `11e4da3`..`0f03da2` | v2 handoff, 48 h plan, 5-account remap |
| 2026-10-14 | `36c9155`..`8d0e6e2`, `4e05435` | v1 runtime patches as commits; v2 layout |
| 2026-10-14 | `2e6c7b8`, `e36f258`, `f82539b`, ... | v2 protection loop, editors, purifiers, baselines, split builder, Optuna script, analysis layer (parallel branches merged) |
| 2026-10-14 | `add50c2`, `50dd3b4`, `d25c712`, `caac5a1` | Kaggle runner; Phase 0 report; budget 4/255 confirmed; v1 reproduced (J01, n=70); tolerance accepted |
| 2026-10-14 | `0e0ad49`, `8f535c1` | S03 splits frozen; PhotoGuard/Mist device bug fixed after S04 |
| 2026-10-15 | `78418ba`, `174cbdf` | fp16-autocast VAE as protection default after speed probe S04t/S04u |
| 2026-10-15 | `9aa5598`, `16df7a7`, `7e52cdf` | S05 fixes: CLIP vision tower, skip-unavailable purifiers, onnxruntime pin, face-detector padding, failure isolation, torchao removal, insightface lock |
| 2026-10-15 | `dd7f02b`, `a0cb961` | Optuna race fix; tuned `v2_final` (trial 5) from S06 |
| 2026-10-15 | `81198d4`, `9ae16ee` | purifier tiers expanded to zero jobs fixed; face-derived smoke images untracked |
| 2026-10-15 | `c52a125`, `8cca06d` | ingest excludes smoke; S13/S21 split; S16 single messy step; S09 results |
| 2026-10-20 | `0a722d3`, `bd76ce9`, `465b66e` | full-scale jobs (100 photos) planned; final evaluation plan; S06 results |
| 2026-10-20 | `6babd18`, `1b57f54`, `97ba09e` | S15 path fix; S19, S18, S22 results |
| 2026-10-20 | `6b40e51` | target-steered variants introduced (see pivot) |
| 2026-10-21 | `1f48d22`, `4a47e92`, `42cb3fe` | S11/S12/S15/S17, S10/S16 results; S24/S25 screening results; target variants on test configured |
| 2026-10-21 | `cbb3454`, `faabd8f` | `readout_final.py`, report draft with measured numbers |
| 2026-10-22 | `2079157`, `1c830af`, `e763b4a` | S26 to S28 results; `finalize_all.py`; README status |
| 2026-10-22 | `f298f5c`, `e286b74`, `b354f9c`, `4833a46` | Pareto figure; S13 and S21 results; audit paragraph |
| 2026-10-22 | `50c2197`, `f1e7774` | `FINAL_REPORT.md` and PDF |
| 2026-10-23 | `94ffbe3`, `22d3f8c` | hash map after history rewrite; presentation deck |

Note: `FINAL_REPORT.md` says 2026-10-23 for the hash map and the HASH_MAP file says the rewrite happened "on 2026-10-23" while its backup tag is named `backup-before-rewrite-20261022`; the exact rewrite date is therefore ambiguous (git dates of the rewritten commits are author dates and may be preserved).

### 21.4 Decisions and the reasons recorded in files
| decision | recorded reason | source |
|---|---|---|
| Budget fixed to 4/255 on the 0 to 255 scale | v1 "8/255" was measured on [-1,1] tensors; real PNGs show max change 4 in 70/70 photos | `reports/PHASE_0.md` |
| Accept reproduction tolerance PSNR +-0.5, SSIM +-0.005, LPIPS +-0.03 | user accepted on 2026-10-14; observed differences +0.075 dB, +0.0004, -0.0002 | `reports/PHASE_0.md` |
| 5 Kaggle accounts, T4 x2 sessions, resumable job runner | no local GPU | `FINAL_REPORT.md` 4.5 |
| fp16 autocast VAE in the loop | about 2x faster; "loss trajectories matched fp32 on a check" | `FINAL_REPORT.md` 4.2, commit `174cbdf` |
| Tune only on validation photos; never on test | operating rule | `HISTORY_FACTS.md`, `EVAL_PLAN_FINAL.md` |
| Target-steered term | diagnosis that `L_vis` has no direction; ablation: encoder-shift term does the work | `FINAL_REPORT.md` 8.2 and 8.3 |
| Report net protection over clean photos under same cleanup | cleanups also change clean photos | `FINAL_REPORT.md` 6.2 |
| Evaluation pads images before face detection | detector missed 38 of 100 test faces | `FINAL_REPORT.md` App. B |
| Skip S14 (no_sem, no_pur) and S23 (eps 6), adaptive attacker on target variants, human study, edit-LPIPS at scale | recorded as "cut for time" | `REPORT_v2_draft.md` header, `FINAL_REPORT.md` 10 |
| Rewrite history to remove 84 face-derived JPEGs | privacy; trees otherwise identical | `reports/HASH_MAP.md` |

### 21.5 Engineering problems and fixes (all documented)
`FINAL_REPORT.md` Appendix B lists 13; `HISTORY_FACTS.md` adds more. Fixed: PhotoGuard/Mist device mismatch (crash); CLIP output type crash under newer transformers; missing GrIDPure file aborting a job (now skipped and logged); face detector missing 38/100 test faces (padding); onnxruntime CUDA mismatch (pinned 1.22.0); insightface unpack race giving NaN embeddings (file lock, completeness check); FaceID load failure from preinstalled torchao; one failing editor aborting all (isolation); `--limit N` selecting different photos in protection and evaluation (sorted ids); Optuna SQLite race; purifier-tier jobs silently expanding to zero (default `--purifier none`; test added); result combiner mixing smoke rows; wrong input paths and CPU launches (S15, S16, S13, S21 per HISTORY_FACTS); a bundle built before a test result was read (rule: tests pass before bundles); edit-LPIPS CPU process-pool crash (only 513 pairs computed; NOT reported); S06 hit its 12-hour timeout; Optuna checked invisibility on the mean not per photo, so `v2_final` violates SSIM >= 0.95 on 29/100 test photos.

## 22. Analysis and interpretation of results

Tags: **SUPPORTED BY DATA** (a measured result with a CI in the files), **HYPOTHESIS** (a mechanism offered but not tested), **NOT TESTED**. All numbers are from `reports/FINAL_REPORT.md` section 7 and 8, `reports/REPORT_v2_draft.md`, and `results/tables/` (SD1.5 img2img strength 0.55, 100 test photos, 3 seeds, PSR threshold 0.1005 to 0.101, unless stated).

| # | statement | tag | evidence / note |
|---|---|---|---|
| 1 | Random noise at the same budget does nothing: clean 13.9%, uniform 14.9% (+1.0 [-1.0, +3.1]), Gaussian 14.7% (+0.8 [-1.2, +2.9]) | SUPPORTED BY DATA | FINAL_REPORT 7.2 |
| 2 | v1 raises PSR to 24.0% (+10.1 [+7.2, +13.0]) over clean | SUPPORTED BY DATA | 7.2 |
| 3 | `v2_main` (cleanups in loop, one-step purifier, denoiser term) is no better than v1 without cleanup (24.3%, +0.3 [-1.8, +2.6]); supported gain only under down-up 0.5 (+5.7 [+3.0, +8.7]) | SUPPORTED BY DATA | 7.4 |
| 4 | In the ablations the encoder-shift term carries the effect (`vis_only` 25.3%, `den_only` 12.3%) | SUPPORTED BY DATA for this implementation; `den_only` left a perturbation of PSNR 50.8 dB so it is a weak test of the denoiser idea | 7.5 |
| 5 | The target-steered term roughly doubles PSR: `v2_tgt_zero_mid` 49.7% vs v1 24.0% (+25.7 [+21.8, +29.8]), 0/100 invisibility violations; dose-response zero 41.8 < mid 49.7 < hi 52.8 | SUPPORTED BY DATA | 7.9 |
| 6 | The control `v2_cheap` (same cleanups, no target term) does not gain on validation (11.7% vs v1 15.0%, 20 photos), so the gain is attributable to the target term | SUPPORTED BY DATA (n=20 validation, plus consistent test-set ordering) | 7.9 |
| 7 | Mechanism: a target latent gives a consistent objective so the editor output is forced toward degraded content | HYPOTHESIS | stated in 8.3 as "not tested here" |
| 8 | PhotoGuard (81.0%) and Mist (63.1%) score higher raw PSR than `mid` (49.7%) without cleanup | SUPPORTED BY DATA (our re-implementations at 4/255) | 7.8 |
| 9 | They fail the per-photo invisibility rule: PhotoGuard PSNR 37.15 / SSIM 0.906, fails 100/100; Mist PSNR 38.00 / SSIM 0.923, SSIM fails 100/100, PSNR fails 50/100 | SUPPORTED BY DATA | 7.1 |
| 10 | Reason: they spend the full 4/255 on nearly every pixel with no structure penalty | HYPOTHESIS (report states it as explanation; per-pixel distribution not shown in report text) | 7.1 |
| 11 | Under JPEG 75, JPEG 50, down-up 0.5, `mid` is ahead of PhotoGuard (paired +5.0 [+1.3, +9.0]; +3.7 [+0.3, +7.0]; +5.3 [+2.7, +8.3]) | SUPPORTED BY DATA | 7.8 |
| 12 | Under VAE round trip and DiffPure 0.1 the visible tools stay ahead (mid minus PhotoGuard -6.7 and -15.7) | SUPPORTED BY DATA | 7.8 |
| 13 | At their own larger official budgets PhotoGuard/Mist would be stronger and more visible | NOT TESTED (caveat in report) | 7.8 |
| 14 | JPEG and resizing are the weakest points: net protection of `mid` +7.3 pts (JPEG 50) vs +35.8 (none); v1 net 0.000 under down-up 0.5 | SUPPORTED BY DATA | 7.3 |
| 15 | Why JPEG/resize hurt: low-pass filtering deletes high-frequency perturbation | HYPOTHESIS | 8.5 |
| 16 | DiffPure-style (SD1.5-based) cleanup does not remove the effect (net +10 to +48 points depending on the method) | SUPPORTED BY DATA for this SD1.5-based cleanup only; "plausible because diffusion cleanup removes noise of a different kind" is a HYPOTHESIS | 7.3, 8.5 |
| 17 | Nothing meaningfully protects IP-Adapter FaceID (PSR 0.000 clean, v1; `mid` 0.020), inpainting (0.000), SDXL (clean 0.013, v1 0.020), or SD1.5 strength 0.3 (0.000) | SUPPORTED BY DATA | 7.7 |
| 18 | Why: FaceID takes identity from a face-embedding network, not the VAE path; inpainting masks the face so noise on it is discarded; SDXL uses a different VAE | HYPOTHESIS (consistent with design; not individually tested; `mid` not run on inpainting/SDXL/IP2P) | 8.6 |
| 19 | At strength 0.75 clean edits already destroy identity (PSR 0.840), so gains cannot be measured | SUPPORTED BY DATA | 7.7 |
| 20 | InstructPix2Pix: PhotoGuard 71.7%, but v1 and v2_final show lower PSR than clean (0.090, 0.033 vs 0.180) | SUPPORTED BY DATA as numbers; the explanation "partly a face-detection effect" is a HYPOTHESIS not separated | 7.7 |
| 21 | No-face audit: no-face share 0.3% (clean, v1), 1.4 to 2.6% (target variants), 8.4% (PhotoGuard), 8.2% (Mist). With those edits removed PSR is 13.6 / 23.7 / 40.9 / 48.6 / 51.5 / 79.2 / 59.8 (clean, v1, zero, mid, hi, PhotoGuard, Mist). So the gain of target variants is not driven by missing faces; PhotoGuard/Mist are partly | SUPPORTED BY DATA | 7.10 |
| 22 | Mean ArcFace similarity of edits falls smoothly: 0.222 clean, 0.177 v1, 0.158 v2_final, 0.129/0.109/0.098 (zero/mid/hi), 0.085 Mist, 0.045 PhotoGuard | SUPPORTED BY DATA | 7.10 |
| 23 | Per-photo protection is probabilistic: share of photos protected on every seed and prompt: 0.0% clean, 2.0% v1, 5.0% `mid`, 35.0% PhotoGuard | SUPPORTED BY DATA | 7.10 |
| 24 | Target variants and PhotoGuard protect by degrading/restyling the edit (cartoon, sketch, sepia, artefacts), not by producing a believable different person; some protected edits remain recognisable to a person | OBSERVATION from a visual look by the authors (not quantified, no human study); treat as qualitative | 7.10 |
| 25 | Adaptive autoencoder attacker brings v1 (11.0%) and v2_final (10.7%) to the clean level (9.3%): net +1.7 [-1.0, +4.7], +1.3 [-1.0, +3.7] | SUPPORTED BY DATA for this weak attacker (120 photos, 35,496 steps) | 7.6 |
| 26 | Target-steered variants survive adaptive attackers | NOT TESTED | 8.7 |
| 27 | Budget is the main lever: PSR none 0.17 / 0.29 / 0.38 at eps 2 / 4 / 8 (different methods at each eps: eps2, v2_final, eps8); eps 8 breaks invisibility (98 and 96 photos) | SUPPORTED BY DATA (eps 6 not run; eps-sweep methods differ from the target variants) | 7.6 |
| 28 | Post hoc selection of `mid`: it was named best-while-invisible after reading the test table, among three variants declared in advance (zero, mid, hi; all shown). Its screening on validation did not include `mid` (validation screened zero, zero_hi, gray) | the post hoc status is an OBSERVATION from the report; the mid-vs-others ranking is therefore SUPPORTED BY DATA only as "one of three declared", not as an unbiased selection | 7.9 |
| 29 | Even clean edits at strength 0.55 already have low ArcFace similarity (0.222), so threshold 0.101 is stringent and all numbers are relative to clean | SUPPORTED BY DATA | 7.10 |
| 30 | `v2_final` (Optuna-tuned) beats v1 by +5.4 [+2.4, +8.7] but fails SSIM >= 0.95 on 29/100 photos | SUPPORTED BY DATA | 7.1, 7.4 |
| 31 | Tuned vs hand-set not separable on validation (0.067 vs 0.042); proxy best 0.10 fell to 0.05 at confirm | SUPPORTED BY DATA (small n=20); Optuna proxy used 10 photos, 16 trials | FINAL_REPORT 10 |
| 32 | Protection works against human viewers / reduces recognition by people | NOT TESTED (no human study) | E10 table: "not run" |
| 33 | Protection generalises to other face datasets or non-portraits | NOT TESTED | CelebA-HQ only |

Per-photo probabilistic protection (row 23) matters for interpretation: a PSR of 49.7% means about half of edits (over photos, seeds, prompts) are below threshold, not that half of photos are safe.

## 23. Contributions

Included only if implemented AND evaluated (evidence in files). Novelty is a claim the authors may or may not be able to defend; no related-work survey was performed in the project files, so none of the novelty statements has been verified against the literature.

| # | contribution | implemented | evaluated | evidence | novelty status |
|---|---|---|---|---|---|
| C1 | Fair attack-and-score evaluation harness: two non-sharing halves (protection vs harness), 5 editors, 13 held-out cleanups, ArcFace PSR with threshold from clean-edit impostor distribution, bootstrap CIs over photos, paired comparisons, noise controls, gates, per-photo invisibility rule | yes (`src/eval/`, `scripts/`) | used for all results | FINAL_REPORT 4.3, 6 | methodological; novelty NOT verified |
| C2 | Corrections to v1: budget is 4/255 not 8/255; reproduction within tolerance (n=70); v1's effect over noise control +10.1 points; v1 loses effect under down-up 0.5 | yes | yes | PHASE_0, FINAL_REPORT 3, 7.2, 7.3 | internal corrections, not a novelty claim |
| C3 | Net-protection reporting (PSR minus clean PSR under the same cleanup) and per-photo (not mean) invisibility constraint check | yes | yes | FINAL_REPORT 6.2, 7.1 | methodological; novelty NOT verified |
| C4 | Training-time cleanup mixture (JPEG, down-up, VAE round trip, one-step SD purifier) in PGD+EoT (`v2_cheap`, `v2_main`) | yes | yes | 7.3 to 7.5 | measured: no clear gain over v1 beyond resizing. A negative result |
| C5 | Target-steered term inside the cleanup loop (zero or gray latent target), variants zero/mid/hi/gray | yes | yes (zero/mid/hi on 100 test photos; gray only on validation) | 7.9 | a combination of PhotoGuard-style target with cleanup-aware EoT; the authors may claim it as a combination, but related-work novelty (e.g. other work doing target attacks under purification) was not verified |
| C6 | Head-to-head with PhotoGuard and Mist at equal budget with per-photo invisibility, showing a protection/invisibility/robustness trade-off | yes (re-implementations checked against official code, `source_verified` notes) | yes | 7.8, Pareto figure | caveat: re-implementations, equal 4/255 budget |
| C7 | Audit methodology: no-face share, similarity distribution, per-photo protected share, visual reading | yes | yes | 7.10 | |
| C8 | Honest negative results: adaptive attacker removes v1/v2_final protection; FaceID/inpainting/SDXL unprotected; ablations | yes | yes | 7.6, 7.7 | |
| C9 | Reproducible Kaggle job pipeline, per-image seeds, logged environment, 377 CPU tests (8 skipped), hash-stamped bundles | yes | tests pass per report | Appendix A | engineering |

Implemented but NOT evaluated (so not contributions): identity loss `L_id` (`lambda_id` 0 everywhere), human study scripts (`make_human_study.py`, `analyze_human_study.py`), GrIDPure approximation (not run), `abl_no_sem`, `abl_no_pur`, `eps6`, SDXL-VAE ensemble as a separate ablation (only inside `v2_final`), `v2_tgt_gray` on test.

## 24. Limitations, risks to validity and uncertainties

Each item was checked against `FINAL_REPORT.md` section 5, 10, App. B, or a code/config file.

**Scale-downs and settings**
1. 40 PGD iterations and K = 4 draws per step (brief suggested 100 / 8).
2. DPM-Solver++ 30-step sampler for every editor, not the sampler of v1 (the sampler differences vs v1 are not enumerated in the files).
3. Optuna: 16 trials on 10 validation photos with a proxy objective; 20-photo confirm; tuned vs hand-set not separable.
4. One CLIP model.
5. Edits are saved as JPEG q90; protected-photo evaluation uses PNG; inputs for v1 reproduction were JPEG q75 (PHASE_0).

**Cleanups and attackers**
6. "DiffPure-style" is an SD1.5-based noise-then-denoise, not the original DiffPure.
7. GrIDPure model unavailable (skipped); IMPRESS and InstantID not run.
8. Adaptive attacker is weak (autoencoder, 120 photos, about 1 GPU-hour) and was run only against v1 and `v2_final`, not the target-steered variants.
9. Cleanup-in-loop training mixture overlaps in kind with held-out cleanups (JPEG, down-up, VAE are of the same family; one held-out down-up setting); held-out DiffPure is the genuinely different one.

**Baselines**
10. PhotoGuard and Mist are re-implementations (checked against MadryLab/photoguard@686bea7 and psyker-team/mist@99f5f3c per SESSION_HANDOFF), run at our 4/255 budget, not their official budgets; Mist's target image is fetched at run time.

**Data and measurement**
11. CelebA-HQ portraits only.
12. De-duplication of identities used ArcFace cosine >= 0.5, an unverified threshold; the splitter's detector missed 38 of 100 test faces; the test list was not re-de-duplicated afterward.
13. PSR is defined with one recogniser only (ArcFace, insightface `buffalo_l`). The identity loss `L_id` (which uses a different network, facenet) is switched off in every preset, so the scorer and the training objective never share a network; but the plan's second recognition model for evaluation was NOT implemented in the evaluation code, so identity is judged by a single network.
14. PSR counts "no face" as protected; the no-face rate is reported (audit).
15. Identity loss `L_id` implemented, never evaluated.
16. Edit-LPIPS missing (only 513 pairs, not reported); LPIPS of the protected photo is reported.
17. No human study (neither noise visibility nor recognisability). ArcFace similarity is not human perception.
18. v1's Phase 2 outputs were lost (user's zip had 0 bytes); v1's LPIPS 0.46 can only be quoted.

**Statistics and selection**
19. Choice of `mid` is post hoc among three variants declared in advance.
20. Many comparisons without multiplicity correction.
21. Target-steered screening used 20 validation photos (n=20 CIs are wide).
22. Seeds: 3 edit seeds; one protection seed per photo (PHASE_0: "1 protection seed per image").
23. `v2_final` violates SSIM >= 0.95 on 29/100 photos (Optuna checked mean).
24. Missing ablations `no_sem`, `no_pur`; eps 6 not run; S14, S23 not run.
25. Net protection under cleanup depends on the clean-baseline PSR, which varies with cleanup (0.097 to 0.260).
26. Run hashes quoted in logs are old hashes; mapping in `reports/HASH_MAP.md`.
27. The Kaggle environment is pinned to the J00 image (torch 2.11.0+cu128, diffusers 0.40.0, transformers 5.16.1); results may not reproduce on other versions; GPU kernels are non-deterministic.

**Privacy / repository hygiene**
28. History was rewritten to remove 84 face-derived JPEGs; per `HISTORY_FACTS.md`, two face-containing figure PDFs (`results/figures/before_after.pdf`, `failure_gallery.pdf`) are still tracked (`git ls-files` confirms they are in the repository) and had not been removed as of 2026-10-25.

## 25. What was NOT implemented or NOT run

| component | where proposed | status | evidence |
|---|---|---|---|
| Human study E10 (2AFC noise visibility, same-person) | HANDOFF_v2, V2_PLAN Phase 7, EVAL_PLAN_FINAL | scripts written, study not run | `results/tables/e10.md` "not run" |
| Adaptive attacker vs target-steered variants | EVAL_PLAN E8 | not run | FINAL_REPORT 7.6, 8.7 |
| S14 (`abl_no_sem`, `abl_no_pur`) | EVAL_PLAN E7 | not run | FINAL_REPORT 7.5 |
| S23 (eps = 6) | EVAL_PLAN E9 | not run | FINAL_REPORT 7.6 |
| GrIDPure purifier | V2_PLAN Phase 2 | model file unavailable; skipped | FINAL_REPORT 2, App. B |
| IMPRESS baseline | proposal | not run | FINAL_REPORT 2, 10 |
| InstantID editor | proposal | not run | FINAL_REPORT 2, 10 |
| Identity loss `L_id` in use | V2_PLAN Phase 6 | implemented (`src/protect/face_loss.py`, facenet backend), `lambda_id: 0.0` everywhere, never evaluated | config/parts/*.yaml |
| Second face recogniser for evaluation (facenet-pytorch) | V2_PLAN Phase 6 | not implemented in `src/eval` (grep shows facenet only in protect/face_loss) | repository grep |
| Edit-LPIPS at scale | EVAL_PLAN E section | 513 of about 5,300 pairs; not reported | EVAL_PLAN A, FINAL_REPORT 6.2 |
| `v2_tgt_zero_mid` on InstructPix2Pix, inpainting, SDXL | not planned in S26 to S28 | "not run" cells in table 7.7 | FINAL_REPORT 7.7 |
| `v2_tgt_gray` on test photos | screened on validation only | not run on test | FINAL_REPORT 7.9 |
| Second-person reproduction J01b | PHASE_0 | not run; no "reproduced by a second person" claim allowed | PHASE_0 |
| Per-photo-constrained re-tuning, more iterations | future work | not done | FINAL_REPORT 11 |
| `git tag v2.0-final` | EVAL_PLAN E.7 | tags present: `v1-unpatched`, `v1.0`, `backup-before-rewrite-20261022`; no `v2.0-final` | `git tag` |
| Original DiffPure | -- | replaced by SD1.5-based style | FINAL_REPORT 10 |
| Other datasets / non-CelebA-HQ images | -- | not run | FINAL_REPORT 10 |
| Official-budget PhotoGuard/Mist runs | -- | not run | FINAL_REPORT 7.8 |

## 26. Future work

From `FINAL_REPORT.md` section 11 (in the authors' stated order of value) plus items implied by sections 7 to 10. Each is a proposal, not a result.
1. Run the adaptive attacker against `v2_tgt_zero_mid`: the most important unknown, since the weak attacker erased v1 and `v2_final` protection.
2. Human study: whether noise is visible and whether the person is recognisable in the degraded edits; ArcFace PSR is not a human measure.
3. Improve JPEG robustness (stronger cleanup mixture): JPEG 50 is the weakest point (+7.3 net for `mid`).
4. SDXL / encoder-ensemble targets: SDXL is unprotected because it uses a different VAE.
5. Face-embedding term for FaceID, evaluated with a second recogniser to avoid grading on the training network: FaceID PSR is about 0.
6. Re-tune with per-photo invisibility constraints, more iterations and draws: `v2_final` violates SSIM on 29% of photos.
7. Run `no_sem`, `no_pur`, eps 6: to complete the ablation and budget curves.
8. Replace post hoc choice of `mid` by a rule fixed on validation photos: to remove selection bias.
9. Evaluate at official PhotoGuard/Mist budgets and with original DiffPure/GrIDPure/IMPRESS/InstantID: to fix baseline fairness caveats (implied, not in the report's list).
10. Complete edit-LPIPS and a multiple-comparison correction: secondary metric and statistical validity (implied).
11. Add other datasets and non-portrait images (implied by the CelebA-HQ-only limitation).

## 27. Claims a paper can and cannot make

| claim | evidence strength | required caveat |
|---|---|---|
| Random noise at equal budget gives no protection | strong (CIs include 0, n=100) | SD1.5 img2img s0.55, ArcFace PSR |
| v1's "8/255" is 4/255 | strong (max change 4 in 70/70 PNGs) | v1's own scale mislabelling only |
| v1 yields +10.1 pts PSR over clean | strong for this setting | modest; strength dependent |
| Training against cleanups alone adds little over v1 | moderate | our implementation, 40 iterations, K=4 |
| Target-steered `mid` about doubles v1 PSR (49.7% vs 24.0%) at 0/100 invisibility violations | strong within setting (paired CI [+21.8, +29.8]) | SD1.5 img2img s0.55, ArcFace PSR, CelebA-HQ, post hoc choice of `mid` among 3 declared variants |
| Gain attributable to the target term | moderate to strong (control `v2_cheap`, dose-response) | control verified on 20 validation photos; mechanism untested |
| `mid` outperforms PhotoGuard under JPEG and resize | moderate (paired CIs exclude 0) | our re-implementations at 4/255; PhotoGuard stronger without cleanup and under VAE/DiffPure |
| PhotoGuard/Mist are visibly worse in invisibility | strong (PSNR/SSIM per photo) | at 4/255 in our re-implementation |
| Protection survives held-out SD1.5-based DiffPure | moderate | not original DiffPure |
| Protection fully robust | NOT supported | JPEG/resize reduce it; adaptive attacker erased v1 and v2_final protection |
| Protection works on FaceID, inpainting, SDXL | NOT supported (PSR about 0) | |
| Edits become a different believable person | NOT supported | edits are often degraded/restyled; qualitative look only |
| Protection works against human recognition | NOT supported | no human study |
| `mid` survives an adaptive attacker | NOT supported | not tested |
| Results generalise beyond CelebA-HQ | NOT supported | one dataset |
| First / state of the art | NOT verifiable | no systematic related-work check |

**Statements to avoid:** "robust to purification" without qualification; "outperforms PhotoGuard/Mist" without the invisibility and cleanup qualifiers; "defeats deepfakes"; "reproduced by a second person"; any number from smoke sessions (S03, S04*, S05*) or the 513 edit-LPIPS pairs; "tuned beats hand-set"; "identity loss improves"; claims about GrIDPure, IMPRESS, InstantID; claims about human perception; "8/255" for v1; "ArcFace-independent" evaluation; claims that `mid` was selected without test-set knowledge.

## 28. Artifact map and glossary

### 28.1 Repository (`C:\Users\karth\Documents\DiffShield-wt\integration`, branch `v2/release`)
| path | purpose |
|---|---|
| `README.md` | v2 status and reproduction steps (v1 legacy section flagged stale) |
| `config/config.yaml`, `config/parts/*.yaml` | all hyper-parameters, seeds, paths, method presets, eval editors/cleanups/tiers; `final.yaml` = tuned `v2_final` (Optuna trial 5) |
| `src/protect/` | Half A: models, purifiers, losses, PGD+EoT, anti-purification, ensemble, `face_loss.py`, `v1.py` adapter, `v2_registry.py` |
| `src/eval/` | Half B: editors, purifiers (`purify_attacks.py`), faces, CLIP, masks, `edit_runner.py`, analysis, stats, adaptive autoencoder |
| `src/baselines/` | noise controls, PhotoGuard, Mist |
| `src/utils/` | config, budget conversion, seeding, run log, IO |
| `src/*.py` (data, eot, losses, optimizer, gradcam, legacy_eval) | v1 code kept at its import paths |
| `scripts/` | `run_protect`, `evaluate`, `tune_optuna`, `run_job`, `ingest_results`, `make_tables`, `make_figures`, `readout_final`, `make_pareto_figure`, `finalize_all`, human-study and demo scripts |
| `kaggle/` | `jobs.yaml` (J00, J01, S01 to S28), `job_runner.ipynb`, `build_bundle.py` |
| `data/splits/`, `data/v1_ref/` | frozen photo lists; v1 reference CSV/JSON copies |
| `tests/` | CPU tests (377 pass, 8 skipped per report) |
| `results/` | CSV/JSON/logs per session (J00, J01, S03 to S28), `combined/`, `tables/`, `figures/` (images never committed apart from figure PDFs listed above) |
| `reports/` | `FINAL_REPORT.md/.pdf`, `REPORT_v2_draft.md`, `PHASE_0..8.md`, `HASH_MAP.md`, `DiffShield_v2_presentation.pptx` (14 slides), Optuna S06 files |
| `docs/` | `V2_PLAN.md`, `V2_PLAN_48H.md`, `HANDOFF_v2.txt`, `EVAL_PLAN_FINAL.md`, `COST_MEASURED.md`, `KAGGLE_LAUNCH.md`, `PROJECT_GUIDE.md`, `LEARNING_HANDOVER.md`, `SESSION_HANDOFF.md`, `REPORT_DRAFT.md` |
| `notebooks/`, `main.py`, `verify_immunization.py`, `test_diffshield.py`, `main.pdf`, `main (3).pdf` | v1 material (the PDFs are the v1 write-up, "DiffShield: A Tri-Modal Adversarial Framework for Robust Immunization Against Deepfakes" per HISTORY_FACTS) |

### 28.2 Deliverables
Report: `reports/FINAL_REPORT.md` and `reports/FINAL_REPORT.pdf` (a copy `DiffShield_v2_final_report.pdf` is in the site's `assets/downloads/`). Deck: `reports/DiffShield_v2_presentation.pptx` (copy in the site downloads). Website: `C:\Users\karth\Documents\DiffShield-site` (static HTML: index, method, results, progress, team, docs, demo pages; `data/team.js` roles empty). Tables: `results/tables/`. Headline figure: `results/figures/pareto_psr_vs_psnr.png`. Optuna: `reports/optuna_trials_S06.csv`, `reports/optuna_confirm_S06.json`.

### 28.3 Glossary
- **PSR**, protection success rate: share of edits with no detected face or ArcFace similarity to the source below a threshold (95th percentile of impostor similarity on clean edits; 0.1005 for SD1.5, pooled 0.103).
- **Net protection**: PSR(method) minus PSR(clean) under the same cleanup, paired over photos.
- **PGD**: projected gradient descent. **EoT**: expectation over transformation (averaging gradients over random transforms/cleanups).
- **L_inf / eps**: maximum per-pixel change; 4/255 on the 0 to 255 scale (v1's "8/255" on [-1,1]).
- **VAE**: variational autoencoder (Stable Diffusion's image encoder/decoder). **UNet**: SD denoiser. **SD1.5 / SDXL**: Stable Diffusion models.
- **L_vis, L_sem, L_str, L_den, L_id**: latent-shift, CLIP-semantic ("a potted plant"), structure (1 - SSIM), denoiser-error, identity losses. **alpha, beta, gamma, eta**: their weights.
- **Target-steered term (TargetVis)**: `-w*MSE(enc(P(x+d)), z_target)` with zero or gray-image latent.
- **PSNR**: peak signal-to-noise ratio (dB); **SSIM**: structural similarity; **LPIPS**: learned perceptual image patch similarity; **CLIP-I**: CLIP image-embedding similarity of edit to source; **ArcFace**: face-recognition embedding (insightface `buffalo_l`).
- **Invisibility rule**: PSNR >= 38 dB and SSIM >= 0.95 on every photo.
- **Purification / cleanup**: attacker preprocessing (JPEG q90/75/50, down-up resize, VAE round trip, **DiffPure-style** SD1.5 noise-then-denoise at t* 0.05 to 0.3, Swin2SR).
- **Anti-purification (AntiPure)**: patch-DCT high-frequency and error-grows-with-timestep terms.
- **Editors**: SD1.5 img2img (strengths 0.3, 0.55, 0.75), SDXL img2img, InstructPix2Pix (IP2P), SD inpainting with keep-face mask, IP-Adapter FaceID.
- **Method names**: `v1`; `v2_cheap` (cleanups in loop, no UNet term; also the control); `v2_main` (+ one-step purifier + denoiser term); `v2_antipure`; `v2_final` (Optuna-tuned, antipure, SDXL ensemble); `abl_*`; `eps2/6/8`; `v2_tgt_zero`, `_mid`, `_hi`, `_gray`; baselines `noise_uniform`, `noise_gauss_psnr`, `photoguard`, `mist`.
- **E0 to E10**: E0 clean reference edits and thresholds; E1 noise controls; E2 vs PhotoGuard/Mist; E3 v1 under cleanups; E4 cheap cleanups in loop; E5 one-step purifier; E6 anti-purification; E7 ablations; E8 adaptive attacker; E9 eps sweep; E10 human study (`docs/EVAL_PLAN_FINAL.md`).
- **S-numbers / J-numbers**: Kaggle sessions defined in `kaggle/jobs.yaml` (J00 environment probe, J01 v1 reproduction, S03 splits, S04/S05 smoke, S06 Optuna, S07 to S28 experiments).
- **Tiers**: H headline (SD1.5 s0.55 etc.), M messy, P purifier, A ablation, AP antipure, SCREEN (validation screening), TGT (target-steered).
- **Splits**: test 100 (includes v1's 70), messy 20, validation 20, train 20, adaptive-attacker pool 120.
- **Gates**: pre-set pass/fail rules from the brief. **Bundle (B0 to B9)**: code zip uploaded as Kaggle dataset. **Optuna**: hyper-parameter search library. **2AFC**: two-alternative forced choice.

## 29. Open questions and missing information

The project files do not contain, so a paper author must obtain:
1. Author names, affiliations, roles and contributions of the five git authors (roles empty in `team.js`; `SESSION_HANDOFF.md` mentions a final-year student and team of 5 but not roles); institution, course/advisor, funding.
2. Real names behind GitHub handles (cootot, Destroyer795, NarainBK, Varun Raj V, venkatesh21bit).
3. All citations and bibliographic details: v1 paper's reference list (`main.pdf`), AdvDM, PhotoGuard, Mist, AntiPure, DiffPure, GrIDPure, IMPRESS, InstantID, IP-Adapter, InstructPix2Pix, DPM-Solver++, CelebA-HQ/CelebAMask-HQ, insightface/ArcFace, LPIPS, CLIP, Optuna. Related-work coverage was never systematically checked.
4. Ethics: IRB or ethics review (none recorded), dataset licence compliance for CelebA-HQ/CelebAMask-HQ (Kaggle `ipythonx/celebamaskhq`), Mist is GPL (not vendored), policy for face images in publications.
5. Human study: not run; no rater data, protocol or approval.
6. Exact sampler differences between v1's editing setup and the v2 harness (DPM-Solver++ 30 steps vs v1's); exact v1 prompts and settings are in `data/v1_ref` and notebooks but were not summarised.
7. Exact Hugging Face model ids/revisions of every model (partly in `config/parts/eval.yaml` and J00 `env_probe`; the runwayml id fallback to an sd15 mirror in the v1 adapter).
8. Verification of the 0.5 ArcFace de-duplication threshold and re-de-duplication of the test list with the corrected detector.
9. Whether the five editors' prompts (three prompts) are in the paper's required form; exact prompt texts are in config.
10. GPU-hour accounting across sessions (`results/gpu_ledger.csv` exists; Kaggle billing of T4 x2 unknown per SESSION_HANDOFF), and the exact number of sessions ("about 30" per report).
11. Edit-LPIPS full results; second-recogniser results; adaptive attacker on target variants; official-budget baselines.
12. Why S14 and S23 were dropped is recorded only as "cut for time".
13. Whether the recorded Optuna details (search space, seeds, proxy photos, constraints) are sufficient to reproduce `config/parts/final.yaml` exactly; the PSR threshold used inside the tuning runs was not recorded.
14. Exact rewrite date of the git history (22 or 23 Oct 2026, ambiguous) and the status of the two face-containing figure PDFs still tracked.
15. Public release plan for code, splits, and results, and any disclosure/dual-use statement for a defensive tool whose bypass (adaptive attacker) is documented.
