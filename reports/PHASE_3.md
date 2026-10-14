# Phase 3 report: cheap purifiers in the loop (E4) (template)

Template: render with `python scripts/make_tables.py --render reports/PHASE_3.md --tables_dir results/tables --out reports/rendered`.
Placeholders without data render as `PENDING (...)`; never type numbers by hand.

Date: `<PENDING>` | Code: `<git hash / bundle>` | Runs: v2_cheap protect + edits (S10): `<run ids PENDING>`
Data status: {{num:synthetic}}

## Result first (plain language)
- PSR under JPEG q75 / down-up 0.5 / VAE round trip, v1 versus v2_cheap (SD1.5 s0.55): see the table below (v1: {{num:psr.v1.jpeg_q75}} / {{num:psr.v1.downup_0.5}} / {{num:psr.v1.vae_rt}}; v2_cheap: {{num:psr.v2_cheap.jpeg_q75}} / {{num:psr.v2_cheap.downup_0.5}} / {{num:psr.v2_cheap.vae_rt}}).
- Invisibility of v2_cheap: PSNR {{num:inv.v2_cheap.psnr}}, SSIM {{num:inv.v2_cheap.ssim}} (requirement: PSNR >= 38, SSIM >= 0.95).
- E4 gate (PSR at least doubles versus v1 under JPEG, down-up and VAE round trip, with PSNR >= 38): **{{gate:E4}}**
- Interpretation (write after reading the tables): `<PENDING>`

## E4 table
{{table:e4}}

## Gate details
Gate values are the handoff's suggestions. If the gate is DEGENERATE (v1 PSR already above 50%, or about zero) the doubling rule is meaningless (V2_PLAN R20): report the absolute numbers and ask the user; do not reinterpret silently.
{{table:gates}}

## Invisibility
{{table:invisibility_methods}}

## Paired differences versus v1 (no purifier, per editor)
{{table:paired_vs_v1}}

## What was done / GPU time / decisions
`<PENDING: jobs run, results/gpu_ledger.csv, decisions, missed gates stated plainly>`

## Open questions for the user (max 2)
1. `<PENDING>`
2. `<PENDING>`

Checkpoint: gate numbers reported as measured, a missed gate is reported, not changed.
