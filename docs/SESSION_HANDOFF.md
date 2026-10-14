# DiffShield v2 — session handoff (read this first)

Written 2026-10-14 ~19:15 local (IST) by the previous Claude Code session, for a new Claude session on another account.
The user is **Siddhartha Gopala** (final-year student, team of 5). The project is a defensive privacy tool: an invisible,
L∞-bounded adversarial perturbation (PGD) on face photos so that diffusion-model edits of the protected photo no longer
preserve the person's identity. v2 adds robustness to "purification", a fair evaluation harness and baselines.

**Deadline: ~16:30 on 2026-10-16** (48 h from 16:30 on 2026-10-14). The user wants the whole v2 plan implemented, not a subset.

## 1. Read these, in this order

1. `docs/HANDOFF_v2.txt` — the original brief from the team (authoritative: phases 0–8, operating rules, eval spec,
   experiments E0–E10, gates, what to claim). Its operating rules still apply (see §8 below).
2. `docs/V2_PLAN_48H.md` — the compressed 48-hour execution plan **currently in force** (scale-downs §2, tiers §3,
   agent/file ownership §4, Kaggle session matrix S01–S18 §5, fallback rules, timeline §6).
3. `docs/V2_PLAN.md` — the original detailed plan (module designs, CSV schema §4.1, risk list §6). Timing is superseded by the 48 h plan.
4. In the integration worktree: `docs/KAGGLE_LAUNCH.md` (per-session launch table), `kaggle/README.md` (runbook),
   `reports/PHASE_0.md`, `docs/REPORT_DRAFT.md`, `README.md`.

## 2. Where the code is (git)

Repo: `C:\Users\karth\Documents\DiffShield` (origin github.com/Project-DiffShield/DiffShield; **nothing has been pushed**, never push without asking).
Work was split across git worktrees; **all of it is merged into `v2/integration`. Continue there.**

| Worktree | Branch | State |
|---|---|---|
| `C:\Users\karth\Documents\DiffShield-wt\integration` | `v2/integration` @ 0180bcb | **Main line. Everything merged. Do all new work here.** |
| `C:\Users\karth\Documents\DiffShield` | `v2/phase-0` @ 50dd3b4 (+ this file) | Phase 0 only; B0 was built from it. Holds `.venv` and `data/v1/`. |
| `...\DiffShield-wt\protect` | `v2/protect` | merged (Half A: v2 protection loop) |
| `...\DiffShield-wt\eval` | `v2/eval` | merged (Half B: editors, purifiers, scoring) |
| `...\DiffShield-wt\baselines` | `v2/baselines` | merged (noise, PhotoGuard, Mist, splits, adaptive AE) |
| `...\DiffShield-wt\analysis` | `v2/analysis` | merged (tables, figures, human study, demo, report) |

`master` = untouched v1 history (tag `v1-unpatched` = f8e8329; tag `v1.0` = v1 with the three runtime patches as real commits).

**Python env:** `C:\Users\karth\Documents\DiffShield\.venv\Scripts\python.exe` (Python 3.13, torch CPU, kornia, diffusers,
transformers, optuna, sklearn, skimage, pandas, lpips, matplotlib, pytest). No GPU on this PC — all GPU work runs on Kaggle.
Run tests from the integration worktree: `& C:\Users\karth\Documents\DiffShield\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider`.
Last full run: **296 passed, 1 skipped** (one template test failed, then fixed in f63becc). All tests are CPU-only with tiny
stand-in models — **no real SD/ArcFace/CLIP weights have ever been loaded by this code yet.** The Kaggle smoke sessions S04/S05 are the first real test.

### Code layout (integration)
- `config/config.yaml` + `config/parts/*.yaml` (deep-merged, sorted; `--set a.b=v` overrides). `config/parts/final.yaml`
  will hold the Optuna-tuned v2_final (from S06); until then `v2_final` falls back to `methods.v2_final_default`.
- `src/protect/` — `registry.py` (build_protector by name), `v1.py` (v1 adapter), A2's `models, purifiers, losses, pgd,
  antipure, ensemble, face_loss, v2_registry`. Method names: `v1, v2_cheap, v2_main, v2_antipure, v2_final, abl_vis_only,
  abl_den_only, abl_no_sem, abl_no_pur, eps2, eps6, eps8, noise_uniform, noise_gauss_psnr, photoguard, mist`.
- `src/eval/` — `editors` (SD1.5 img2img, SDXL img2img, InstructPix2Pix, SD inpaint with keep-face masks, IP-Adapter FaceID as img2img),
  `purify_attacks` (jpeg q90/75/50/other, downup, vae_rt, diffpure 0.05–0.3, swin2sr, gridpure, adaptive_ae), `faces` (insightface ArcFace),
  `clip_metric`, `masks`, `edit_runner` (tiers, resume, CSV), `analysis` (bootstrap CIs, PSR, paired tests, gates), `metrics`, `stats`, `schema`, `adaptive_ae`.
- `src/baselines/` — noise controls, PhotoGuard and Mist **reimplementations, checked against the official code**
  (MadryLab/photoguard@686bea7, psyker-team/mist@99f5f3c), run at OUR eps (4/255), official eps recorded in yaml.
- `scripts/` — `run_protect, merge_shards, budget_check, prepare_v1_data, make_splits, prepare_data, evaluate, tune_optuna,
  train_adaptive_ae, run_job, ingest_results, compute_edit_lpips, make_tables, make_figures, make_human_study,
  analyze_human_study, demo, compare_runs, env_probe, check_env, gpu_ledger, smoke_report, merge_requirements`.
- `kaggle/` — `jobs.yaml` (J00, J01, J01b, S01–S18), `job_runner.ipynb` (human sets `JOB_IDS` and optional `PARAMS`,
  `RESUME_FROM`), `build_bundle.py` (builds `kaggle/dist/diffshield-code-<hash>.zip` from a clean commit).
- `data/splits/v1_eval70.txt`, `v1_calib.txt` (committed). `data/v1_ref/` = small v1 CSV/JSON copies (committed).
  `data/v1/` = user's extracted v1 zip (git-ignored, read-only).

## 3. Key facts and decisions already made

- **Budget:** v1's "8/255" was on [-1,1] tensors = **4/255 on the standard scale** (v1 CSV: Linf 0.0313725 for all 70 images).
  All v2 budgets are on the 0–255 scale; `eps_model = eps_255*2/255`. To be confirmed on real PNGs by J01's budget check.
- **Step size:** v1 actually used 0.5/255 standard (not the handoff's 1/255). `methods.v1` keeps 0.5; every v2 preset sets `step_255: 1.0`.
- **v1 exact weights:** α 0.2699475469647097, β 0.9023875567667267, γ 60.52658429696642 (Optuna trial 23). v1 results:
  PSNR 41.911 ± 0.440, SSIM 0.97496 ± 0.0048, LPIPS 0.2316 ± 0.063 (n=70). v1 inputs were JPEG q75; v1 PSNR used truncating uint8 cast; seed 42; shuffled output names → pair by `Original_Filename`.
- v1 protected images and v1 Phase 2 outputs are **not available** (user's Phase 2 zip was 0 bytes). v1's Phase 2 LPIPS 0.46 can only be quoted.
- **Dataset:** Kaggle `ipythonx/celebamaskhq` (CelebA-HQ-img, attribute anno, HQ↔CelebA mapping, pose anno, parsing masks;
  **no identity file** → identity de-dup by ArcFace cosine ≥ 0.5, unverified threshold).
- **Splits:** test 100 = all 70 v1 eval images + 30 new (enables paired v1 comparison); messy 20 (pose/blur/lighting/occlusion + JPEG/resize degradation);
  train 20, val 20, ae_pool 120; v1 calibration images (5319, 1715, 6385, 8932) excluded. Produced on Kaggle by **S03**, then must be committed into `data/splits/` before B2.
- Edits: DPM-Solver++ 30 steps, 3 seeds, prompts rotated except headline SD1.5 s0.55 (all 3). Edits saved as JPEG q90; edit-LPIPS computed later on CPU (`compute_edit_lpips.py`).
- PSR threshold = 95th percentile of impostor ArcFace similarity from E0 clean edits. No-face counts as protection success; no-face rate reported separately.
- Scale-downs and every limitation to report are listed in `docs/V2_PLAN_48H.md` §2 and `docs/REPORT_DRAFT.md` §7.
- PhotoGuard now uses the official 1000 iterations (≈5× costlier than planned → S13 longer, still < 12 h). Mist's official target image is downloaded at runtime in S13 (GPL repo; not vendored).
- Unverified items to watch in smoke runs: several HF model ids, IP-Adapter FaceID with DPM-Solver (switch `eval.editors.ipadapter_faceid.scheduler: ddim` if artefacts),
  GrIDPure (needs guided-diffusion checkpoint; drop after ~1 h effort if it won't run), fp16 gradient underflow / Mist 1e4 weight overflow (`loss_scale`), T4 memory with K=4.
  Whether Kaggle bills a "T4 x2" session as 1 or 2 GPU-h per hour is unknown — check after S04/S05 (fallback rule 4).

## 4. Kaggle setup and accounts

- **5 accounts**, ~30 GPU-h/week each. **Acct1 and Acct2 are the user's own** (launchable any time, incl. overnight). Acct3–5 are teammates: launch only ~09:00–23:00
  (a committed "Save Version → Run All" keeps running unattended up to 12 h).
- Private dataset **`diffshield-code`** (owned by the user, teammates added as collaborators) holds the code zip. Each session attaches `diffshield-code` + `ipythonx/celebamaskhq`,
  GPU T4 x2, Internet ON. Notebook = `kaggle/job_runner.ipynb` (imported per account). The notebook picks the newest bundle in the dataset.
- Outputs come back as `<JOB>_outputs.zip`; the user downloads them to `C:\Users\karth\Downloads` and tells you. Unzip under `results/` in the integration worktree
  (git-ignored; they contain identifiable images — never commit them).

### Bundles
| Bundle | Built from | Path | Status |
|---|---|---|---|
| B0 | v2/phase-0 @ 50dd3b4 | `Documents\DiffShield\kaggle\dist\diffshield-code-50dd3b4.zip` | uploaded; J00+J01 (Phase 0) running/ran on acct1 |
| B1 | v2/integration @ 8b95e54 | `Documents\DiffShield\kaggle\dist\B1\` (zip + `job_runner.ipynb` + `KAGGLE_LAUNCH.md`) | ready; user asked to upload as new dataset version |
| B2 | — | to build | needs: S03 splits committed into `data/splits/`, fixes from S04/S05 smoke reports. Commits after 8b95e54 (analysis merge, PhotoGuard/Mist alignment, Mist runtime fetch) are also only in B2. |
| B3 | — | to build | B2 + `config/parts/final.yaml` from S06 Optuna |

Build: in the integration worktree, commit everything, then `python kaggle/build_bundle.py` (refuses a dirty tree).

## 5. Current status (at handoff, ~19:15 day 1)

- User is running **Phase 0 (B0, `JOB_IDS=["J00","J01"]`) on acct1** and just asked how to view outputs. Told them: viewer page → Output section (download the zips) / Logs; or editor right sidebar → Output → ⋮ → Download.
  Expected in log: budget check max pixel change **4**, PSNR ≈ 41.9, SSIM ≈ 0.975.
- Asked the user to also: upload B1 as a new `diffshield-code` version; on acct2 run **S03** (CPU) and **S05** (GPU); on acct1 run **S04** after Phase 0.
  Optional: a teammate runs `JOB_IDS=["J01b"]` (independent second-person reproduction, ~0.5 GPU-h; only for the report's reproducibility claim).
- Nothing from Kaggle has been received yet. **No real numbers exist. Never invent any.**

## 6. What to do next (in order)

1. **Phase 0 outputs arrive** → unzip `J00_outputs.zip`/`J01_outputs.zip` under `results/`; check budget_check (max |Δ| = 4?), run
   `scripts/compare_runs.py --a results/<J01 run> --v1_csv data/v1_ref/stage1_per_image_metrics.csv`; fill `<PENDING>` in `reports/PHASE_0.md`; set `budget.eps_confirmed: true` if confirmed;
   pin `requirements.txt` from `J00/env_probe/pip_freeze.txt`. Report to the user in plain language (lead with the result). Ask the user to accept the reproduction tolerance (±0.5 dB, ±0.005 SSIM, ±0.03 LPIPS) if numbers differ.
2. **S03 outputs** → copy `splits_out/*.txt` into `data/splits/`, commit.
3. **S04/S05 smoke reports** (`smoke_report.md`: s/image, s/iteration, peak memory, failures) → fix every failure; compare timings with `docs/V2_PLAN_48H.md` §1; apply fallback rules 3–4 if >1.5× or if quota bills 2×/hour.
4. Build **B2**, give the user launch steps for **S06–S09 at ~02:00 day 2** (acct1: S06, S07; acct2: S08, S09). Then S10–S14 at 09:00 (acct3–5), S15/S16 (acct1, `PARAMS={"prev_root": "/kaggle/input/<S06 output>"}`), and after S06 → commit `final.yaml` → **B3** → S17 (acct4), S18 (acct2). See `docs/V2_PLAN_48H.md` §5–6 and `docs/KAGGLE_LAUNCH.md`.
5. As outputs land: `scripts/ingest_results.py --results results --unzip` → `compute_edit_lpips.py --workers 4` → re-ingest → `make_tables.py` → `make_figures.py` → render `reports/PHASE_*.md` and `docs/REPORT_DRAFT.md` (commands in README).
6. Human study (E10): `make_human_study.py` once v2_final edits exist (~day 2 noon); students run it; `analyze_human_study.py` (pilot if < 10 raters).
7. Day 2 evening–day 3: failure cases, limitations, report write-up, `scripts/demo.py`, README reproduction, `git tag v2.0-final` (ask before pushing anything).

## 7. How the previous session worked (the user asked for this)

- Planning with **Opus 5.5 (high effort)**, implementation with **Sonnet 5.5 subagents (high effort)**, each in its own git worktree with a file-ownership table, coordinator merges and runs the full test suite.
- The user is new to diffusion models: report in short plain language, define terms, lead with the result, ask at most 1–2 questions at a time.

## 8. Operating rules (from the handoff; still binding)

- Never invent a number, citation or result; mark unverified things "unverified". Do not use any "RSAS" method.
- Every reported number: n, mean ± 95% bootstrap CI over images, 3 seeds, paired vs v1. Never tune on test.
- No deletions (move to `_to_delete/` and tell the user). Work on copies; never modify `data/v1/`, `notebooks/v1_reference/` or the user's originals in Downloads.
- Never type, store or ask for passwords/API tokens; the user signs in to Kaggle/HF themselves.
- Don't redistribute datasets or commit identifiable images.
- Commit often; commit messages end with the co-author line used so far (`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; use your own model's line). Don't push without asking.
- STOP gates are report checkpoints in the 48 h plan: report the gate numbers honestly, don't silently change a gate, ask the user if a gate fails badly.
