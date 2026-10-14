# Phase 5 report: anti-purification terms (E6) and the single-term ablation (template)

Template: render with `python scripts/make_tables.py --render reports/PHASE_5.md --tables_dir results/tables --out reports/rendered`.
Placeholders without data render as `PENDING (...)`; never type numbers by hand.

Date: `<PENDING>` | Code: `<git hash / bundle>` | Runs: v2_antipure protect + edits (S12), anti-pure single-term ablation on validation (S07): `<run ids PENDING>`
Data status: {{num:synthetic}}

## Result first (plain language)
- v2_antipure versus v2_main under DiffPure t*=0.1 / 0.2 (SD1.5 s0.55): v2_main {{num:psr.v2_main.diffpure_0.1}} / {{num:psr.v2_main.diffpure_0.2}}; v2_antipure {{num:psr.v2_antipure.diffpure_0.1}} / {{num:psr.v2_antipure.diffpure_0.2}}.
- Go/no-go on Tier 3 (handoff: drop it if the gains are within the CI): **{{gate:E6}}**
- Interpretation (write after reading the tables): `<PENDING>`

## E6 table
{{table:e6}}

## Where the validation-set single-term ablation of each anti-purification term is recorded
`<PENDING: path of the S07 validation output; each term alone, with n images, CI>`; the test-split tables above are the confirmation.

## What was done / GPU time / decisions
`<PENDING: jobs run, results/gpu_ledger.csv, go/no-go decision and how v2_final was configured (antipure on/off is an Optuna parameter chosen on validation only)>`

## Open questions for the user (max 2)
1. `<PENDING>`
2. `<PENDING>`

Checkpoint: a negative result is reported with numbers.
