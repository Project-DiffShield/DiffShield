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
| S06 | 1 | `["S06"]` | code, celebamaskhq | B2 | 13 | `optuna/` (trials.csv, `v2_final/final_candidate.yaml`), `prot_v2_final_{test,messy}/`, `eval_v2_final_H/` |
| S07 | 1 | `["S07"]` | code, celebamaskhq | B2 | 12 | `eval_ap_*_val` (antipure single-term ablation), `e0_clean_{H,M,P}/`, `S07/cache/clean_edits` (needed for the LPIPS pass), `thresholds/psr_thresholds.json` |
| S08 | 2 | `["S08"]` | code, celebamaskhq | B2 | 10 | `prot_v1_{test,messy}/`, `eval_v1_{H,M,P}/` |
| S09 | 2 | `["S09"]` | code, celebamaskhq | B2 | 11 | `prot_v2_main_*/`, `eval_v2_main_{H,M,P}/` |
| S10 | 3 | `["S10"]` | code, celebamaskhq | B2 | 7 | `prot_v2_cheap_test/`, `eval_v2_cheap_{H,P}/` |
| S11 | 3 | `["S11"]` | code, celebamaskhq | B2 | 10 | `prot_abl_{vis,den}_only_test/`, `eval_abl_vis_den_A/` |
| S12 | 4 | `["S12"]` | code, celebamaskhq | B2 | 8 | `prot_v2_antipure_test/`, `eval_v2_antipure_P/` |
| S13 | 5 | `["S13"]` | code, celebamaskhq | B2 | 12 | `prot_{photoguard,mist}_{test,messy}/`, `eval_pg_mist_{H,M,P}/` |
| S14 | 5 | `["S14"]` | code, celebamaskhq | B2 | 14 | `prot_abl_{no_sem,no_pur}_test/`, `eval_abl_nosem_nopur_A/`, `prot_noise_*`, `eval_noise_{H,M}/` |
| S15 | 1 | `["S15"]`, `PARAMS={"prev_root": "/kaggle/input/<S06 output>"}` | code, celebamaskhq, **S06 output** | B2 | 7 | `adaptive_ae/ae_final.pt`, `prot_v2_final_ae_pool/`, `prot_v1_test/`, `eval_ae_{clean,v1,v2_final}/` |
| S16 | 1 | `["S16"]`, `PARAMS={"prev_root": "/kaggle/input/<S06 output>"}` | code, celebamaskhq, **S06 output** | B2 | 7 | `eval_v2_final_{P,M,H_sdxl}/` (reads the S06 protected images; protects nothing) |
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
