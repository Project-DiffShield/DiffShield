> SUPERSEDED on 2026-10-20 by `docs/EVAL_PLAN_FINAL.md` section G (new deadline, n = 100 jobs, sessions S22/S23 added). Dataset `diffshield-code-v7`, git hash `b72e5ff`.

# Final launch sheet (written 2026-10-15 ~21:45 IST; deadline 16:30 on 2026-10-16)

Use ONLY the dataset `diffshield-code-v7` (bundle `kaggle/dist/B7/diffshield-code-b72e5ff.zip`, git hash `b72e5ff`, tuned `config/parts/final.yaml` included). Older datasets (v2..v5) contain jobs that
silently write empty purifier results or lack the tuned config. Every notebook: import `kaggle/job_runner.ipynb`; inputs = `diffshield-code-v7` +
`ipythonx/celebamaskhq` (+ the extra inputs named below); accelerator GPU T4 x2; Internet ON; `RESUME_FROM = None`; **Save & Run All**.
In the first log lines check `git_hash` equals the one the coordinator announced. Two GPU sessions at a time per account.
No session may be launched after ~22:30 (it must end by ~09:00 to leave the day for analysis). Sessions stop starting new work after 11 h.

| Acct | Notebook | `JOB_IDS` / `PARAMS` | Extra inputs | What | Wall h |
|---|---|---|---|---|---|
| acct1 | A | `["S16"]`, `{"prev_root": "/kaggle/input/<S06 output>"}` | acct1's S06 notebook output | v2_final edits (H, P, M, SDXL); re-protects the 20 messy photos | ~6.5 |
| acct1 | B | `["S15"]`, `{}` | none | adaptive attacker E8 (uses the tuned config from the bundle) | ~5 |
| acct2 | A | `["S19"]`, `{"prev_root_s08": "/kaggle/input/<S08 output>", "prev_root_s09": "/kaggle/input/<S09 output>"}` | acct2's S08 and S09 outputs | purifier tier for v1 and v2_main | ~5.5 |
| acct2 | B | `["S18"]`, `{}` | none | epsilon 8, first 50 photos | ~3.3 |
| acct3 | A | `["S10"]`, `{}` | none | v2_cheap + clean purifier tier | ~7.5 |
| acct3 | B | `["S11"]`, `{}` | none | ablations vis_only, den_only (first 50 photos) | ~6 |
| acct4 | A | `["S12"]`, `{}` | none | v2_antipure | ~7 |
| acct4 | B | `["S17"]`, `{}` | none | epsilon 2 and 6, first 50 photos | ~6.5 |
| acct5 | A | `["S13"]`, `{}` | none | PhotoGuard | ~6.5 |
| acct5 | B | `["S14"]`, `{}` | none | ablations no_sem, no_pur (first 50 photos) + noise controls | ~7.5 |
| acct3 (after S11) | C | `["S21"]`, `{}` | none | Mist | ~3 |

Priority if quota or time runs short (drop from the bottom): S19, S16, S10, S14, S11, S17, S18, S12, S15, S13, S21.
When a session ends: Output -> download `<SESSION>_outputs.zip` into Downloads and tell the coordinator. Never publish or share the images.
Per-session 10-minute check: `git_hash` correct; `[faces] ... providers=['CUDAExecutionProvider'...`; no `EDITOR FAILED` / `pack incomplete`; no `exited 1` within the first minute of any step.
