# Running DiffShield jobs on Kaggle

You never edit code on Kaggle. You upload one zip once, then set `JOB_IDS` in one notebook and press Run All.

## One-time setup (per Kaggle account)

1. Get the code zip `diffshield-code-<hash>.zip` (built on the dev machine with `python kaggle/build_bundle.py`; the file is in `kaggle/dist/`, which is git-ignored).
2. Kaggle -> Datasets -> New Dataset -> upload the zip. Name it `diffshield-code`, keep it **Private**. Kaggle unzips it automatically. (Account 1 owns it and adds the other accounts as collaborators, or each account uploads its own copy.)
   - New code version later: open the dataset -> New Version -> upload the new zip.
3. Kaggle -> Code -> New Notebook -> File -> Import Notebook -> upload `kaggle/job_runner.ipynb` (from the repo).
4. In the notebook right panel:
   - **Add Data**: the `diffshield-code` dataset; for J01 also the dataset **ipythonx/celebamaskhq** (CelebAMask-HQ; search "celebamaskhq").
   - **Accelerator**: J00 -> None (CPU is enough). J01 -> **GPU T4 x2** (use P100 only as a fallback; newer torch builds may not support it, unverified).
   - **Internet**: ON (models are downloaded from Hugging Face without any login: nothing to enter).
   - Optional, after the first successful run: Settings -> Environment -> "Pin to original environment" so the library versions do not drift.

## Running session S01 (J00 then J01)

In the first code cell set

```python
JOB_IDS = ["J00", "J01"]    # a list: run in order in one session
PARAMS  = {}                # {"n_images": 5} for a quick smoke test of J01
```

then **Run All** (or Save Version -> Save & Run All). J00 needs no GPU, so for J00 alone you can use a CPU session (saves quota). If you want both in one GPU session, J00 costs about a minute.

Smoke test first (recommended, ~10 min): `JOB_IDS = ["J01"]`, `PARAMS = {"n_images": 5}`. Check that it finishes and that the last cells print `ok`. Then run the full one (`PARAMS = {}`, 70 images).

Outputs appear in the notebook Output tab as `J00_outputs.zip`, `J01_outputs.zip` and `job_runner_summary.json`. Download them and tell the coordinator / put them under `results/` (unzipped) in the repo folder. They contain protected images (CelebA-HQ derivatives): research use only, never commit or publish them.

## If a session dies or hits the time limit

Run the same notebook again with the previous output added as an input (Add Data -> your earlier notebook output) and set
`RESUME_FROM = "/kaggle/input/<name-of-that-output>"`. Finished images are skipped (checked per image); finished jobs (`DONE` marker) are skipped.

## Spreading one job over several accounts

`JOB_SHARD = "0/2"` on one account and `"1/2"` on another processes disjoint halves (the combined sharding with the two GPUs is automatic). Merge afterwards with `python scripts/merge_shards.py --run_dir <dir> --also <other dir>`.

## Job definitions

`kaggle/jobs.yaml` (schema in the docstring of `scripts/run_job.py`). Locally, preview any job without running it:
`python scripts/run_job.py --jobs J01 --dry_run --work tmp`.

## Rules

- No tokens, passwords or API keys anywhere. Hugging Face downloads are anonymous.
- Do not commit anything from `results/**/images`, datasets or zips.
- Check `gpu_time.json` in each output: the GPU ledger (`python scripts/gpu_ledger.py`) is built from these.
