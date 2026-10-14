# DiffShield v2: 48-hour execution plan

Replaces the phase-by-phase timing of `docs/V2_PLAN.md`; module designs, interfaces, CSV schema (one row per (image, seed, editor, purifier, method)) and the technical-risk list in `V2_PLAN.md` still hold. T+0 = 2026-10-14 ~16:30 local. Honesty rules unchanged (never invent numbers; 95% bootstrap CI over images, 3 edit seeds, paired vs v1; never tune on test; no deletions; work on copies; no credentials; smoke first). STOP gates become **report checkpoints**: each gate's numbers are computed and written honestly in `reports/PHASE_n.md` but do not block the next job.

All per-image / per-edit costs below are planner **estimates, not measurements**; the smoke jobs (J03a/b) replace them.

## 0. Facts confirmed from v1 artifacts (`data/v1/phase1_zip/kaggle/working/metrics/`)

- PSNR 41.911 ± 0.440, SSIM 0.97496 ± 0.0048, LPIPS 0.2316 ± 0.063, Linf = 0.0313725 = 8/255 on [-1,1] for all 70 images → **budget = 4/255 standard scale** (set `eps_confirmed: true` after reproduced PNGs show max |Δ| = 4 gray levels).
- Exact weights: α = 0.2699475469647097, β = 0.9023875567667267, γ = 60.52658429696642 (Optuna trial 23) → `methods.v1`.
- Calibration set: k = 4 (5319, 1715, 6385, 8932). Eval list: `evaluation_70_manifest.csv`.
- `stage1_per_image_metrics.csv` is keyed by `Original_Filename`; output names were shuffled (`protected_face_001` = `1600.jpg`), so the Phase 0 check pairs by original filename (`compare_runs.py --v1_csv`).
- v1 protected images and Phase 2 outputs are not available. v1's Phase 2 LPIPS 0.46 can only be quoted from the handoff.

## 1. Cost model (T4, fp16 autocast unless noted; estimates)

| Op (one 512 px image, batch 1) | T4 s |
|---|---|
| VAE encoder fwd+bwd (fp16) | 0.15 |
| VAE decoder fwd+bwd (fp16, checkpointing) | 0.35 |
| UNet fwd+bwd (fp16, checkpointing, no CFG) | 0.30 |
| CLIP B/32 fwd+bwd / SSIM / Kornia JPEG | 0.02 / 0.01 / 0.03 |
| v1 PGD iteration (fp32) | 0.55 |
| SD1.5 sampling step with CFG (fp16) | 0.13 |

Edits (DPM-Solver++ 30 steps, 3 seeds batched), s/edit: SD1.5 img2img s0.3/0.55/0.75 = 1.6/2.5/3.2; InstructPix2Pix 4.5; SD inpaint 4.0; IP-Adapter FaceID 4.0; SDXL img2img 1024 s0.55 = 20. Scoring +0.1 s/edit.

Protection s/image: v1 25; PhotoGuard 30; Mist 60; noise ≈0 (CPU); v2_cheap 54; v2_main 120; v2_antipure 150; abl_vis_only 83, abl_den_only 115, abl_no_sem 115, abl_no_pur 75; v2_final 190 (K=2: 100).

Per-method edit cost at n=100, 3 seeds: Tier H 2.25 GPU-h (+1.67 SDXL); Tier P 3.3 GPU-h (FaceID subset 1.33, IP2P subset 0.75); Tier A 2.1 GPU-h per ablation.

Always choose **GPU T4 x2**; P100 fallback only (newer torch may not support it; unverified).

## 2. Scale-downs (decided)

| Item | Handoff | 48 h plan | Limitation to state |
|---|---|---|---|
| Test identities | 100 + 20 messy | kept | none |
| Edit seeds | 3 | 3 everywhere | protection uses 1 seed/image (per-image sha256 seed) |
| Prompts | 3/editor | 3 for headline SD1.5 s0.55; elsewhere 1 prompt/image rotated (index mod 3) | per-prompt breakdown only for SD1.5 s0.55 |
| Strengths | 0.3/0.55/0.75 | all 3 for 9 headline methods; 0.55 elsewhere | |
| Sampler | v1 default (PNDM 50) | DPM-Solver++ 30 for all editors | not v1's sampler; v1 re-evaluated under same sampler |
| Editors | 5 | all 5, IP-Adapter FaceID; SDXL only for clean, v1, PhotoGuard, v2_final | InstantID not run; SDXL partial |
| Method × purifier grid | full | tiered (§3) | not every pair measured |
| GrIDPure | if feasible | official code tried in smoke; dropped after 1 h if not running | documented |
| IMPRESS | if feasible | not run | limitation |
| PGD iters / EoT K | 40(→100) / 4(8) | 40 / 4; K=2 for Optuna and adaptive-attacker pool | no 100-iter / K=8 |
| Optuna | 30–50 × 20 val | 16 trials × 10 val, proxy objective (20 iters, K=2, 1 seed, SD1.5 s0.55 + FaceID, purifiers {none, jpeg_q75, diffpure_0.1}); top 3 confirmed on all 20 val | weights may be under-tuned |
| Phase gates 3–6 | sequential + retry | all variants built in parallel; gates computed after the fact; v2_final chosen only on val (antipure on/off and β∈[0,1.5] are Optuna params) | no "retry once" |
| CLIP ensemble | if L_sem helps | not done; single CLIP, β may go to 0 | limitation |
| Adaptive attacker E8 | AE on pairs | 120 disjoint images, 128 px patches, 1 GPU-h training | weak attacker → lower bound on damage |
| Human study E10 | 15 × 20 | materials at T+18; if <10 raters by T+40 report as pilot with actual n | |
| Identity labels | — | ArcFace de-dup (same person if sim ≥ 0.5, unverified threshold) unless `identity_CelebA.txt` supplied | splits approximate by identity |

Fallback cuts if smoke timings exceed estimates by >1.5×, in order: SDXL → 1 seed; drop Tier P FaceID subset; ablation purifiers 5→3 (none, jpeg_q50, diffpure_0.1); ε-sweep purifiers → {none, diffpure_0.1}; K=2 for ablations.

## 3. Methods and evaluation tiers

Methods: clean (E0); noise_uniform, noise_gauss_psnr (E1); photoguard, mist (E2); v1 (E3); v2_cheap (E4); v2_main = one-step purifier + L_den (E5); v2_antipure (E6); abl_vis_only / abl_den_only / abl_no_sem / abl_no_pur + v2_main as "both" (E7); v2_final (Phase 6); eps2/eps6/eps8 of v2_final (E9); adaptive AE purifier (E8).

- **Tier H (no purifier):** {clean, noise×2, photoguard, mist, v1, v2_cheap, v2_main, v2_final} × all editors × all strengths (SDXL subset). Messy 20: SD1.5 rotated + FaceID for {clean, noise, photoguard, mist, v1, v2_main, v2_final}.
- **Tier P (13 purifiers:** jpeg q90/q75/q50, jpeg_other, downup_0.5, downup_other, vae_rt, diffpure 0.05/0.1/0.2/0.3, restore_swin2sr, gridpure**) × SD1.5 s0.55 rotated:** {clean, v1, v2_cheap, v2_main, v2_antipure, v2_final} all 13; photoguard & mist 5 (jpeg75, jpeg50, downup0.5, vae_rt, diffpure0.1); FaceID under {jpeg75, vae_rt, diffpure0.1, diffpure0.2} for {clean, v1, v2_main, v2_final}; IP2P under {jpeg75, diffpure0.1} for {v1, v2_final}.
- **Tier A:** 4 ablations × {none, jpeg50, vae_rt, diffpure0.1, diffpure0.2} with SD1.5, plus FaceID, IP2P, inpaint with no purifier.
- **E9:** eps 2/6/8 × {none, jpeg75, diffpure0.1} SD1.5 + FaceID no purifier.
- **E8:** AE purifier on {clean, v1, v2_final} × {SD1.5 rotated, FaceID}.

Edit jobs score in-job (ArcFace via insightface, CLIP-I, edit LPIPS vs shared E0 clean edits). Edit images saved only as samples (first 10 images) plus headline clean/v1/v2_final edits for figures and human study; everything else CSV-only.

## 4. Coding waves and agent split (one owner per file; separate worktree/branch per agent; coordinator merges)

**Interface contract (frozen):**
- `Protector.protect(x: Tensor[1,3,H,W] in [-1,1], seed: int) -> (Tensor, dict)`
- `src/protect/registry.py` (A1) dispatches: `v1` → `src/protect/v1.py`; `v2_*`, `abl_*`, `eps*` → `src/protect/v2_registry.build_v2_protector(name, cfg, device)` (A2, lazy import); `noise_*`, `photoguard`, `mist` → `src/baselines/registry.build_baseline(name, cfg, device)` (A4, lazy import)
- `TrainPurifier(nn.Module).forward(x, gen) -> x` (differentiable)
- `EvalPurifier.apply(img: PIL.Image, seed: int) -> PIL.Image`
- `Editor.load(device)`, `Editor.edit_batch(img, prompt, strength, seeds) -> list[PIL.Image]`, `Editor.unload()`
- `FaceScorer.embed(img) -> (emb | None, det_score, bbox)`
- CSV schema per `V2_PLAN.md` §4.1

**Config without conflicts:** `load_config` deep-merges `config/config.yaml` with `config/parts/*.yaml` (sorted). Each agent owns only its `config/parts/<area>.yaml`. Extra deps go in `requirements.d/<agent>.txt`; merged into `requirements.txt` at integration.

| Agent | Owns (create/modify only these) |
|---|---|
| **A1** Phase 0 | everything in `V2_PLAN.md` §3, plus `config/parts` loading, `src/protect/registry.py`, `scripts/run_job.py` + `kaggle/job_runner.ipynb` (human sets `JOB_ID`; reads `kaggle/jobs.yaml`), `requirements.d` merge |
| **A2** Half A protection | `src/protect/{models,losses,purifiers,pgd,antipure,ensemble,face_loss,v2_registry}.py`, `config/parts/protect.yaml` (all v2/abl/eps presets), `scripts/tune_optuna.py` (2 workers, shared SQLite), `tests/test_protect_*.py` |
| **A3** Half B harness | `src/eval/{editors,purify_attacks,gridpure,faces,clip_metric,masks,edit_runner}.py`, `scripts/evaluate.py` (stage edit+score, tiers, resume by row key), `config/parts/eval.yaml`, `tests/test_eval_*.py` (fake pipelines) |
| **A4** baselines/data/adaptive | `src/baselines/{registry,noise_control,photoguard,mist}.py`, `src/eval/adaptive_ae.py`, `scripts/{make_splits,prepare_data,train_adaptive_ae}.py`, `config/parts/{baselines,data}.yaml`, `tests/test_baselines_*.py`, `tests/test_data_*.py` |
| **A5** analysis/report (after Phase 0 merge) | `scripts/make_tables.py`, `scripts/make_figures.py`, `src/eval/analysis.py`, `scripts/{make_human_study,analyze_human_study,demo}.py`, `reports/PHASE_*.md`, `README.md`, `kaggle/jobs.yaml` content |

Bundles: B0 (Phase 0), B1 (integrated, T+8), B2 (smoke fixes, T+9.3), B3 (tuned `config/parts/final.yaml`, T+12.7).

## 5. Kaggle job matrix (each account 2 slots s1/s2, T4 x2; wall ≈ GPU-h/2 + 0.3 h)

Cross-account inputs: Kaggle datasets `diffshield-code` (bundles) and `diffshield-e0` (clean edits), owned by account 1 with accounts 2–6 as collaborators. All jobs checkpoint every 5 protected images / 50 edits, resume via `--resume_from`, write `gpu_time.json`.

| Job | Acct/slot | T+h | Experiment | Inputs | GPU-h |
|---|---|---|---|---|---|
| J00 | 1/s2 CPU | 1→1.3 | env probe (pip freeze, model hashes) | — | 0 |
| J01 | 1/s1 | 4→4.6 | Phase 0 v1 reproduction, 70 imgs (JPEG q75 inputs) | B0, CelebAMask-HQ | 0.5 |
| J01b | 2/s1 | 5→5.6 | same, second person | same | 0.5 |
| J02 | 2/s2 CPU | 5→6 | splits (test100, messy20, train20, val20, AE pool 120, excl. 4 calib), ArcFace dedup, source embeddings, inpaint masks | CelebAMask-HQ, B0 | 0 |
| J03a | 1/s1 | 8→8.5 | protection smoke, all methods, 5 imgs, 5 iters | B1 | 0.3 |
| J03b | 1/s2 | 8→8.6 | edit/purifier smoke, 5 imgs, 1 seed, all editors/purifiers, GrIDPure attempt | B1 | 0.4 |
| J10a/b | 1/s1+s2 | 9.5→12.2 | E0 clean edits (H + SDXL + P + subsets + messy), 2 shards → publish `diffshield-e0` | B2 | 9.4 |
| J11 | 2/s1 | 9.5→11.7 | protect v1, PhotoGuard, Mist (test+messy); noise (CPU) | B2 | 3.8 |
| J12a/b | 3/s1+s2 | 9.5→11.2 | protect v2_cheap (100), v2_main (120) | B2 | 5.5 |
| J13 | 4/s1 | 9.5→13.0 | protect v2_antipure, abl_no_pur | B2 | 6.3 |
| J14 | 4/s2 | 9.5→14.2 | protect abl_vis_only, abl_den_only, abl_no_sem | B2 | 8.7 |
| J15 | 5/s1 | 9.5→12.2 | Phase 6 Optuna on val → `config/parts/final.yaml` (B3) | B2 | 4.7 |
| J16 | 5/s2 | 9.5→11.0 | Phase 5 single-term antipure ablation on val | B2 | 2.3 |
| J20a/b | 2/s1+s2 | 12.5→15.2 | E3 v1 edits (H + SDXL + P + subsets + messy) | J11, e0 | 9.4 |
| J23 | 3/s1 | 12.5→15.6 | E4 v2_cheap edits H + P | J12, e0 | 5.6 |
| J24 | 3/s2 | 12.5→16.3 | E5 v2_main edits H + P + FaceID subset + messy | J12, e0 | 7.0 |
| J30a/b | 5/s1+s2 | 12.8→14.7 | protect v2_final test+messy | B3 | 6.3 |
| J31a | 6/s1 | 12.8→16.3 | E9 ε=2 protect + edit | B3, e0 | 6.3 |
| J31b | 6/s2 | 12.8→16.3 | E9 ε=6 | B3, e0 | 6.3 |
| J25 | 4/s1 | 13.1→15.2 | E6 v2_antipure edits P + FaceID none | J13, e0 | 3.6 |
| J26a | 4/s2 | 14.3→16.7 | E7 edits abl_vis_only, abl_den_only | J14, e0 | 4.2 |
| J40a/b | 5/s1+s2 | 14.8→17.5 | v2_final edits H + SDXL + P + subsets + messy | J30, e0 | 9.4 |
| J21 | 2/s1 | 15.3→18.0 | E1 noise-control edits H + messy | J11, e0 | 4.7 |
| J22 | 2/s2 | 15.3→19.9 | E2 PhotoGuard + Mist edits | J11, e0 | 8.5 |
| J26b | 4/s1 | 15.3→17.7 | E7 edits abl_no_pur, abl_no_sem | J13+J14, e0 | 4.2 |
| J31c | 6/s1 | 16.4→19.9 | E9 ε=8 | B3, e0 | 6.3 |
| J42 | 5/s1 | 17.6→20.9 | E8: protect 120 pool imgs (v2_final K=2), train AE, apply to clean/v1/v2_final, edit | J30, e0 | 5.9 |
| Buffer | 6/s2, 3/*, 1/* | 16→28 | re-runs / fallback | | ≤30 |

Totals ≈ 130 GPU-h ≈ 73 of 180 session-h (41%). Per account ≈ 7 / 15 / 9 / 13 / 15 / 11 session-h. Last planned GPU job ends ≈ T+21; every job < 5 h wall.

## 6. Timeline

| T+h | Coding | Kaggle | Analysis / report |
|---|---|---|---|
| 0–1 | freeze interfaces; spawn A2–A4; A1 continues Phase 0 | J00 | |
| 1–6 | A1 finishes Phase 0 → B0; A2–A4 build with fakes/tiny models | J01, J01b, J02 | PHASE_0 checkpoint |
| 6–8 | integration, CPU tests → B1 | | A5 starts tables/figures |
| 8–9.3 | smoke fixes → B2 | J03a/b | measured costs → fallback cuts if needed |
| 9.5–12.7 | on-call; tuned config → B3 | wave 2: J10–J16 | PHASE_1 from E0 |
| 12.5–21 | on-call | waves 3–4: J20–J42 | PHASE_2..6 checkpoints as CSVs land |
| 18 | human-study materials ready | | students run study T+18→T+40 |
| 21–28 | re-runs | buffer | PHASE_7 tables/figures, failure-case gallery |
| 28–42 | demo (T+30–34); `git tag v2.0-final` (T+34) | | report: threat model, method, results w/ CIs, failures, limitations |
| 42–48 | README reproduction, package splits/configs | | final review |
