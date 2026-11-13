# Kaggle launch table (sessions S01-S18)

Every session is the same notebook, `kaggle/job_runner.ipynb`; only the first code cell changes. All sessions use
**GPU T4 x2**, Internet **ON**, and need no logins or tokens. GPU-h and durations are the planner's estimates from
`docs/V2_PLAN_48H.md` section 5, not measurements; S04/S05 produce the first real timings (`smoke_report.md`).

## One-time per account
1. Datasets -> New Dataset -> upload `kaggle/dist/diffshield-code-<hash>.zip` (private, name `diffshield-code`). Kaggle unzips it. A new bundle = New Version of the same dataset.
2. New notebook -> File -> Import Notebook -> `kaggle/job_runner.ipynb`. Accelerator GPU T4 x2, Internet ON.
3. Add Data (right panel): `diffshield-code` and `ipythonx/celebamaskhq` (CelebAMask-HQ). Add a previous session's *Notebook Output* when the table says so.

## Launch
Edit the first code cell (`JOB_IDS`, `PARAMS`), press **Save Version -> Save & Run All (Commit)** so the run continues without you.
When it ends: Output tab -> download `<SESSION>_outputs.zip` (it contains `<SESSION>/results/...` with CSVs, protected PNGs,
edits as JPEG q90, logs, `gpu_time.json`) and unzip it into the repo `results/` folder. Never commit images.

Dead or timed-out session: add its output as an input and relaunch with `RESUME_FROM = "/kaggle/input/<that output>"`
(finished jobs, images and edit rows are skipped).

Smoke mode for any session: `PARAMS = {"smoke": True}` (5 images, 5 iterations, 1 edit seed, tiny Optuna/AE budgets).
Smoke runs use throw-away smoke splits if the bundle has no frozen `data/splits/test.txt` yet. Never report numbers from smoke runs.

## Order and table

| Session | Acct | JOB_IDS / PARAMS | Attach as inputs | Needs bundle | Est. GPU-h | Main outputs (in `<S>/results/`) |
|---|---|---|---|---|---|---|
| S01 | 1 | `["S01"]` (= J01) | code, celebamaskhq | B0+ | 0.5 | `J01/` reproduction of v1 Phase 1, `budget_check.json`, `compare_v1.md` |
| S02 | 3 | `["S02"]` (= J01b) | code, celebamaskhq | B0+ | 0.5 | same, second person |
| S03 | 2 (CPU) | `["S03"]`, accelerator None | code, celebamaskhq | B0+ | 0 | `splits_out/{test,messy,train,val,ae_pool}.txt`, `splits_meta.json` -> **commit into `data/splits/` for B2** |
| S04 | 1 | `["S04"]` | code, celebamaskhq | B1 | 0.3 | `prot_<method>_test/` for 14 methods, `smoke_report.md` (s/image, peak memory) |
| S05 | 1 | `["S05"]` | code, celebamaskhq | B1 | 0.4 | `smoke_*` edit runs for all editors/purifiers, GrIDPure attempt, LPIPS and threshold stages, `smoke_report.md` |
| S06 | 1 | `["S06"]` | code, celebamaskhq | B2 | 13 | `optuna/` (trials.csv, `v2_final/final_candidate.yaml`), `prot_v2_final_{test,messy}/` (the Tier H edits moved to S16) |
| S07 | 1 | `["S07"]` | code, celebamaskhq | B2 | 12 | `eval_ap_*_val` (antipure single-term ablation), `e0_clean_{H,M,P}/`, `S07/cache/clean_edits` (needed for the LPIPS pass), `thresholds/psr_thresholds.json` |
| S08 | 2 | `["S08"]` | code, celebamaskhq | B2 | 10 | `prot_v1_{test,messy}/`, `eval_v1_{H,M,P}/` |
| S09 | 2 | `["S09"]` | code, celebamaskhq | B2 | 11 | `prot_v2_main_*/`, `eval_v2_main_{H,M,P}/` |
| S10 | 3 | `["S10"]` | code, celebamaskhq | B2 | 7 | `prot_v2_cheap_test/`, `eval_v2_cheap_{H,P}/` |
| S11 | 3 | `["S11"]` | code, celebamaskhq | B2 | 10 | `prot_abl_{vis,den}_only_test/`, `eval_abl_vis_den_A/` |
| S12 | 4 | `["S12"]` | code, celebamaskhq | B2 | 8 | `prot_v2_antipure_test/`, `eval_v2_antipure_P/` |
| S13 | 5 | `["S13"]` | code, celebamaskhq | B2 | 12 | `prot_{photoguard,mist}_{test,messy}/`, `eval_pg_mist_{H,M,P}/` |
| S14 | 5 | `["S14"]` | code, celebamaskhq | B2 | 14 | `prot_abl_{no_sem,no_pur}_test/`, `eval_abl_nosem_nopur_A/`, `prot_noise_*`, `eval_noise_{H,M}/` |
| S15 | 1 | `["S15"]`, `PARAMS={"prev_root": "/kaggle/input/<S06 output>"}` | code, celebamaskhq, **S06 output** | B2 | 7 | `adaptive_ae/ae_final.pt`, `prot_v2_final_ae_pool/`, `prot_v1_test/`, `eval_ae_{clean,v1,v2_final}/` |
| S16 | 1 | `["S16"]`, `PARAMS={"prev_root": "/kaggle/input/<S06 output>"}` | code, celebamaskhq, **S06 output** | B2 | 7 | `eval_v2_final_{H,P,M,H_sdxl}/` (reads the S06 protected images; protects nothing) |
| S17 | 4 | `["S17"]` | code, celebamaskhq | **B3** (has tuned `config/parts/final.yaml`) | 13 | `prot_eps{2,6}_test/`, `eval_eps{2,6}_E9/` |
| S18 | 2 | `["S18"]` | code, celebamaskhq | **B3** | 6 | `prot_eps8_test/`, `eval_eps8_E9/` |

Notes
- S15/S16 take the tuned config from the S06 output (`<prev_root>/S06/optuna/v2_final/final_candidate.yaml`). If B3 is already
  in the dataset and has `config/parts/final.yaml`, add `"final_yaml": ""` to PARAMS to use the bundle copy instead.
- `prev_root` is the path under `/kaggle/input` printed by the first cells of the notebook (look at the "inputs attached" list).
- S17/S18 without B3: pass `"final_yaml": "/kaggle/input/<S06 output>/S06/optuna/v2_final/final_candidate.yaml"`.
- Several sessions can be chained in one run: `JOB_IDS = ["S10", "S12"]` (outputs in separate folders; a failure in one does not stop the next).
- A session that failed a step is not marked DONE; rerun the same cell (or with `RESUME_FROM`) and only unfinished work repeats.
- After downloads: edit LPIPS is computed on the edits that were saved as JPEG q90 (`python scripts/evaluate.py --stage lpips --out <eval dir> --clean_edits_dir <S07>/results/.../cache/clean_edits`), PSR thresholds come from S07 `thresholds/psr_thresholds.json`.

## Changes after the S04t/S04u/S05 smoke tests (2026-10-15)
- Measured v2 protection cost is 2-3x the plan (`docs/COST_MEASURED.md`); the VAE now runs under fp16 autocast by default (`runtime.vae_dtype_protect: autocast`, ~2x faster).
- Pre-agreed fallback cut: the four ablations (S11, S14) and the epsilon sweep (S17, S18) are protected and edited on the **first 50 test images** (`--limit 50`) instead of 100, with the same K=4. Compare them with v2_main / v2_final restricted to the same 50 images (paired). Limitation to report: n = 50 for E7 and E9.
- S06 no longer runs the Tier H edits of v2_final (S16 does), so S06 stays under the 11 h wall guard.
- Quota is billed by wall-clock time (acct1: 1h43m used while about 1.6-1.7 h of 2-GPU sessions had run).

## Step 1 re-tune sessions T01 / T02 / T03 (docs/STEP1_PLAN.md, docs/TUNING_PROTOCOL.md)
Run in order. Time estimates are planner guesses (UNVERIFIED) until a smoke run has been timed. Smoke: `PARAMS = {"smoke": True}`.

| Job | Accelerator | JOB_IDS / PARAMS | Attach as inputs | Smoke | Outputs to download (`<JOB>/...`) |
|---|---|---|---|---|---|
| T01 | GPU T4 (one GPU is enough) | `["T01"]` | code, celebamaskhq | first 600 images, small set sizes (6 + 2 + 2); its lists are throw-away | `splits/{tune_search,tune_confirm}.txt`, `splits/tune_meta.json`, `reports/selection_report.{json,md}`; `pool/pool_embeddings.npz` is derived features (keep private) |
| T02 | GPU T4 x2 | `["T02"]` | code, celebamaskhq, optionally the previous T02 output | 2 trials, 2 photos, 3 iterations | `optuna/v2_next.db`, `optuna/v2_next/` (trials, per-trial results) |
| T03 | GPU T4 x2 | `["T03"]` | code, celebamaskhq, **the T02 output** | confirm-top 1, 2 photos, 3 iterations | `optuna/v2_next/confirm.json`, `decision.json`, and only if accepted `final_candidate.yaml` |

- T01 does not need the tune lists. After the full T01 the coordinator checks `selection_report.md`, copies the lists and `tune_meta.json` into `data/splits/`, commits them with `docs/TUNING_PROTOCOL.md` BEFORE any trial, then rebuilds the bundle.
- T02 and T03 read `data/splits/tune_search.txt` / `tune_confirm.txt` from the bundle, so they fail (also in smoke mode) if the bundle was built before the lists were committed.
- Resume: T01 restores finished embedding chunks and T02 restores the Optuna study from `RESUME_FROM = "/kaggle/input/<earlier output>"` (a T02 session ended by its 10 h timeout guard is resumed this way). T03 needs `RESUME_FROM` = the finished T02 output (it reads the study from there); candidates already in `confirm.json` are skipped.
- Do not run two T02 sessions on copies of the same study DB unless you merge them deliberately; the job does not support merging.

## Final-run sessions T04 / T05 (tuned `v2_next_tuned` vs the old final `v2_tgt_zero_mid`)
Only after T03 accepted a winner: the coordinator copies T03's `final_candidate.yaml` to `config/parts/next_tuned.yaml` and rebuilds the code bundle. Both jobs use tier rows on the 100 TEST photos. Smoke: `PARAMS = {"smoke": True}` (never report smoke numbers).

| Job | Accelerator | JOB_IDS / PARAMS | Attach as inputs | RESUME_FROM | Outputs (`<JOB>/results/`) |
|---|---|---|---|---|---|
| T04 | GPU T4 x2, Internet ON | `["T04"]` | code (bundle WITH `config/parts/next_tuned.yaml`), celebamaskhq | only to resume a dead T04: `"/kaggle/input/<earlier T04 output>"` | `prot_v2_next_tuned_test/`, `eval_v2_next_tuned_NEXT/` (= the TGT rows), `eval_v2_next_tuned_HELD/` |
| T05 | GPU T4 x2, Internet ON | `["T05"]`, `PARAMS = {"old_prot": "/kaggle/input/<S28 output dataset>"}` | code, celebamaskhq, **the S28 output as a Kaggle dataset** | only to resume a dead T05 (its own earlier output), NOT for the S28 images | `prot_v2_tgt_zero_mid_test/` (restored copy), `eval_v2_tgt_zero_mid_HELD/`, `eval_clean_HELD/` |

- T04 guard: the first step after setup (`guard_next_tuned`) loads the merged config like the runner does (`load_config(..., method="v2_next_tuned")`) and aborts the job before any GPU work if `methods.v2_next_tuned` is missing or empty. Without it the registry would run plain `v2_next` (= the old final method) under the new name. It also runs in smoke mode. T04 writes the NEXT and HELD rows of `v2_next_tuned`; T05 writes the HELD rows of the old method and of `clean`.
- T05 does not re-protect. Why not `RESUME_FROM`: the runner looks for `<RESUME_FROM>/<JOB_ID>/results/<run_id>`, and the S28 output is laid out as `S28/results/prot_v2_tgt_zero_mid_test`; a plain RESUME_FROM would only work if the dataset folder were renamed `T05`, and the resumed protect step would then also re-check the config hash against today's code. Instead the step `restore_old_protected` searches `old_prot` recursively for the folder `prot_v2_tgt_zero_mid_test`, copies it (images, shard metrics CSVs, config) into `T05/results/`, aborts if fewer than 100 PNGs (5 in smoke) are present, and the HELD edits then read `images/<id>.png` from there. Upload: zip or folder of the S28 output (`S28_outputs.zip` unzipped is fine; only `S28/results/prot_v2_tgt_zero_mid_test/images` is needed, about 100 PNGs), dataset name e.g. `diffshield-s28-prot`. If the S28 images are unavailable, fall back: add a protect macro step for `v2_tgt_zero_mid` to T05 (cost below).
- GPU time. Measured earlier: protection 116 s per photo on one T4, about 98 min for 100 photos on 2 T4; the TGT tier is 1950 rows at a mean 2.6 s per edit (about 85 min of summed edit time, about 42 min wall on 2 GPUs: UNVERIFIED arithmetic, assumes even sharding and no model-load overhead). T04 total: about 98 min protect plus the NEXT rows plus HELD rows. HELD cost (pixel DDPM purifier at t*=100/200, `docs/DIFFPURE_PIXEL_EVAL.md` planner estimate 5-10 s per photo and purifier, unmeasured): UNVERIFIED, so T04 is expected at roughly 3 to 4 h and T05 at under 1 h (both UNVERIFIED). Re-protecting the old method in T05 would add about 98 min.
- The notebook is run with `Save & Run All`; finished rows are skipped on a rerun of the same session (per-row resume).
