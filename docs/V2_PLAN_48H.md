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

## 5. Kaggle session matrix (5 accounts, chained sessions)

**Clock.** T+0 = 16:30 day 1 (2026-10-14); deadline T+48 = 16:30 day 3. Acct1 and Acct2 (coordinating user) launch any hour. Acct3–5 launch only 09:00–23:00, but a committed run ("Save Version → Run All") continues unattended up to 12 h. All sessions GPU T4 x2.

**Every session:** `kaggle/job_runner.ipynb` takes `JOBS=[...]` and runs them sequentially; work sharded across the 2 GPUs; checkpoint every 5 protected images / 50 edits; `--resume_from`; `gpu_time.json` per job. **Wall guard:** skip a job whose estimate would push the session past 11 h (goes to buffer).

**Dependencies.** Edit LPIPS is NOT computed in-run: all edits saved as JPEG q90; edit LPIPS vs E0 clean edits computed later (rule 6). Each session needs only the code bundle, `ipythonx/celebamaskhq`, and its own account's earlier outputs. Bundles: B0 Phase 0 (~20:00 d1), B1 full code (00:30 d2), B2 smoke fixes (01:50 d2), B3 = B2 + `config/parts/final.yaml` (~09:40 d2). S15/S16 on acct1 read the tuned config and protected set directly from S06 output (`--config_from_run`).

| Session | Acct/slot | Launch → end (local) | T+ | Jobs in order | Inputs | GPU-h |
|---|---|---|---|---|---|---|
| S01 | 1/s1 | d1 20:30 → 21:05 | 4.0→4.6 | Phase 0 v1 reproduction, 70 images | B0 | 0.5 |
| S02 | 3/s1 | d1 21:30 → 22:05 | 5.0→5.6 | Phase 0 reproduction by second person | B0 | 0.5 |
| S03 | 2 CPU | d1 21:00 or d2 00:30 | 4.5/8.0 | splits (test100, messy20, train20, val20, AE pool 120; excl. v1 calib), ArcFace dedup, source embeddings, inpaint masks | B0 + A4 | 0 |
| S04 | 1/s1 | d2 00:30 → 01:00 | 8.0→8.5 | protection smoke, all methods, 5 imgs, 5 iters | B1 | 0.3 |
| S05 | 1/s2 | d2 00:30 → 01:10 | 8.0→8.7 | edit/purifier smoke, 5 imgs, 1 seed, all editors/purifiers, GrIDPure attempt | B1 | 0.4 |
| S06 | 1/s1 | d2 02:00 → ~09:20 | 9.5→16.8 | Optuna on val (cap 3 h) → protect v2_final test+messy → v2_final Tier H edits (no SDXL) | B2 | 13.25 |
| S07 | 1/s2 | d2 02:00 → ~08:15 | 9.5→15.75 | Phase 5 antipure single-term ablation on val → E0 all clean edits | B2 | 11.7 |
| S08 | 2/s1 | d2 02:00 → ~07:25 | 9.5→14.9 | protect v1 test+messy → E3 edits (H + SDXL + P + subsets + messy) | B2 | 10.2 |
| S09 | 2/s2 | d2 02:00 → ~07:50 | 9.5→15.3 | protect v2_main → E5 edits (H + P + FaceID subset + messy) | B2 | 11.0 |
| S10 | 3/s1 | d2 09:00 → ~12:55 | 16.5→20.4 | protect v2_cheap → E4 edits H + P | B2 | 7.1 |
| S11 | 3/s2 | d2 09:00 → ~14:10 | 16.5→21.7 | protect abl_vis_only + abl_den_only → E7a edits | B2 | 9.7 |
| S12 | 4/s2 | d2 09:00 → ~13:10 | 16.5→20.7 | protect v2_antipure → E6 edits | B2 | 7.8 |
| S13 | 5/s1 | d2 09:00 → ~15:05 | 16.5→22.6 | protect PhotoGuard + Mist → E2 edits | B2 | 11.5 |
| S14 | 5/s2 | d2 09:00 → ~16:25 | 16.5→23.9 | protect abl_no_sem + abl_no_pur → E7b edits → noise controls (CPU) → E1 edits | B2 | 14.2 |
| S15 | 1/s1 | d2 09:30 → ~13:10 | 17.0→20.7 | E8: protect AE pool with v2_final (K=2) + re-protect v1 on test → train AE (1 h) → apply → edits | B2 + S06 out | 6.7 |
| S16 | 1/s2 | d2 09:30 → ~13:25 | 17.0→20.9 | v2_final remainder: SDXL + Tier P + FaceID-P + IP2P-P + messy | B2 + S06 out | 7.15 |
| S17 | 4/s1 | d2 09:45 → ~16:20 | 17.25→23.8 | E9 ε=2 then ε=6 (protect + edits) | B3 | 12.6 |
| S18 | 2/s1 | d2 09:45 → ~13:15 | 17.25→20.75 | E9 ε=8 | B3 | 6.3 |
| Buffer | 2/s2 from d2 08:00; 3/s1 from 13:00; 4/s2 from 13:15; 1/* from 13:30 | until d3 ~10:00 | ≤41.5 | re-runs, wall-guard leftovers, late GrIDPure, J50 LPIPS fallback | | ≤20 |

Totals ≈ 131 GPU-h ≈ 71 session-h of 150 (47%). Per account: acct1 ≈22.7, acct2 ≈14.7, acct3 ≈9.6, acct4 ≈10.8, acct5 ≈13.5 session-h. Latest planned GPU job: S14 ≈16:25 d2. Critical path: B2 (01:50) → S06 (09:20) → B3 (09:40) → S17 (16:20).

Coverage: E0 S07; E1 S14; E2 S13; E3 S08; E4 S10; E5 S09; E6 S12; E7 S11+S14 (+ v2_main from S09 as "both"); E8 S15; E9 S17+S18 (ε=4 = v2_final from S06+S16); E10 human study.

### Fallback rules
1. Optuna capped at 3 h wall inside S06; use best trial so far; report trials completed.
2. Wall guard: if S06 projects > 11 h after smoke, move its Tier H edits into S16; guard-skipped jobs re-queue in buffer.
3. Smoke > 1.5× estimate (known by 01:10 d2): apply §2 cut order before the 02:00 launches: SDXL 1 seed → drop FaceID-P → ablation purifiers 5→3 → ε-sweep purifiers {none, diffpure0.1} → K=2 for ablations.
4. Quota check (unverified how Kaggle charges T4 x2): after S04/S05 compare quota used vs wall time; if 1 wall-h costs 2 quota-h the plan needs ≈141/150 → apply cuts 1–5 immediately; buffer only for failures.
5. Launch windows: acct3–5 latest launch 22:30 d2; missed 09:00 launch moves to next buffer slot. Acct1–2 last re-run launch 04:00 d3. No new experiments after 23:00 d2.
6. LPIPS pass: `scripts/compute_edit_lpips.py` locally on CPU (4 workers, ≈0.2 s/pair, ≈71k pairs, ≈4 h) as outputs are downloaded; fallback Kaggle job J50 (~0.5 GPU-h). Report states both images were JPEG q90 before LPIPS (same for every method).
7. Data moves between accounts only via the user's local download; v1 protected images for E8 regenerated inside S15 with the same per-image seeds.

## 6. Timeline (local clock; T+0 = 16:30 d1)

| Local time | T+ | Coding | Kaggle | Analysis / humans |
|---|---|---|---|---|
| d1 16:30–20:00 | 0–3.5 | A1 Phase 0; A2/A3/A4; A4 make_splits first | J00 env probe (acct1 CPU) | coordinator writes `kaggle/jobs.yaml` |
| d1 20:00–22:30 | 3.5–6 | B0 (~20:00) | S01 20:30; S03 21:00 if ready; S02 (acct3) 21:30 | PHASE_0 draft from S01 |
| d1 22:30–d2 00:30 | 6–8 | integration, CPU tests → B1 | (S03 at 00:30 if not earlier) | PHASE_0 final (budget confirmed?) |
| d2 00:30–01:50 | 8–9.3 | smoke fixes → B2 | S04, S05 (acct1) | measured costs → fallback rules 3–4 |
| **d2 02:00** | 9.5 | on call | **launch S06, S07 (acct1), S08, S09 (acct2)** | |
| d2 02:00–09:00 | 9.5–16.5 | hotfixes → B2.x | running | A5 tables from S01–S05 |
| d2 07:25–08:15 | 15–15.75 | | S07–S09 finish; download | E3/E5 identity numbers; PSR thresholds from E0 |
| **d2 09:00** | 16.5 | | **acct3–5 launch S10–S14** | |
| d2 09:20–09:45 | 16.8–17.25 | commit final.yaml → B3 | S06 ends → S15, S16 (acct1) 09:30; S17 (acct4), S18 (acct2) 09:45 | PHASE_6 checkpoint |
| d2 09:45–12:00 | 17.25–19.5 | A5 human-study materials | | study forms ready by 12:00; sessions d2 12:00–22:00 and d3 09:00–12:00 |
| d2 12:55–16:25 | 20.4–23.9 | | sessions finish (S14 last ~16:25) | rolling download → LPIPS pass → tables; PHASE_1/2 ~14:00; PHASE_3/4/5 ~16:00 |
| d2 16:30–23:00 | 24–30.5 | fixes; demo.py | buffer re-runs (acct3–5 by 22:30) | PHASE_7 tables/figures, failure gallery |
| d2 23:00–d3 09:00 | 30.5–40.5 | `git tag v2.0-final` | acct1/2 re-runs only (by 04:00) | report drafting |
| d3 09:00–13:00 | 40.5–44.5 | README reproduction, package splits/configs | done | human-study analysis ~13:00; final figures |
| d3 13:00–16:30 | 44.5–48 | | | final review, PHASE_8, demo check |

**User actions:** d1 ~17:30 J00 · 20:00 upload B0 · 20:30 S01 · 21:30 teammate S02 · d2 00:30 upload B1 + S04/S05 (+S03) · 01:50 upload B2 · 02:00 launch S06–S09 · 07:30–08:30 download S07–S09 · 09:00 teammates S10–S14 · 09:20 download S06 best params · 09:40 upload B3 · 09:30/09:45 S15/S16, S18, S17 · afternoon downloads.
