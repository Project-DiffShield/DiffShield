# Launch sheet for the teammate accounts (acct3, acct4, acct5) - day 2, from 09:00

Bundle: the dataset `diffshield-code-v3` (git `64b0815`, from `kaggle/dist/B2c`; shared by the coordinator). Do NOT use any older dataset.
Every notebook: import `kaggle/job_runner.ipynb`; inputs = the code dataset + `ipythonx/celebamaskhq`; accelerator GPU T4 x2; Internet ON;
`PARAMS = {}`, `RESUME_FROM = None`; **Save & Run All** (so it keeps running when the laptop is closed). One notebook per session, two sessions can run at once per account.

| Account | Notebook | `JOB_IDS` | What it does | Est. wall time | Output zip |
|---|---|---|---|---|---|
| acct3 | A | `["S10"]` | v2_cheap protection (100 photos) + edits (E4) | ~5 h | `S10_outputs.zip` |
| acct3 | B | `["S11"]` | abl_vis_only + abl_den_only on the first 50 test photos + edits (E7a) | ~6 h | `S11_outputs.zip` |
| acct4 | A | `["S12"]` | v2_antipure protection (100 photos) + edits (E6) | ~8 h | `S12_outputs.zip` |
| acct5 | A | `["S13"]` | PhotoGuard (1000 iterations) + Mist + edits (E2); downloads the official Mist target image at runtime | ~7.5 h | `S13_outputs.zip` |
| acct5 | B | `["S14"]` | abl_no_sem + abl_no_pur (first 50 test photos) + edits (E7b), noise controls + edits (E1) | ~7.5 h | `S14_outputs.zip` |

Estimates come from `docs/COST_MEASURED.md` and the S05 edit timings (edits are unmeasured at full scale; treat as +-30%).

## Check 10 minutes after each launch (Logs tab of the running version; open it in its own browser tab)
1. `"git_hash"` starts with `64b0815`.
2. After the data-preparation step you see protection lines `[k/n] <id> psnr=.. ssim=.. gpu_s=..` (expected `gpu_s` per image at 40 iterations: v2_cheap ~150, v2_antipure ~430, abl_* ~250, PhotoGuard ~270, Mist ~55).
3. Later, in the edit stage: `[faces] insightface buffalo_l providers=['CUDAExecutionProvider', ...]` and NO line containing `EDITOR FAILED` or `pack incomplete`.
If a line is wrong, copy the log into the chat with the coordinator right away. Do not edit the notebook or the dataset while a session is running (that restarts it).

## When a session finishes
Notebook -> Output -> download `<SESSION>_outputs.zip` -> give it to the coordinator (who unzips it under `results/` of the repo). Never upload or share the images publicly (CelebA-HQ research use only).

## Limits
Latest launch on a teammate account: 22:30 on day 2. A committed run continues up to 12 h; the job stops starting new work after 11 h and can be resumed (`RESUME_FROM = "/kaggle/input/<name of the earlier output>"` with that output added as an input).
