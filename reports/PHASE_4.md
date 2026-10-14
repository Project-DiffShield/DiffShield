# Phase 4 report: one-step SD purifier and denoiser loss against held-out purifiers (E5) (template)

Template: render with `python scripts/make_tables.py --render reports/PHASE_4.md --tables_dir results/tables --out reports/rendered`.
Placeholders without data render as `PENDING (...)`; never type numbers by hand.

Date: `<PENDING>` | Code: `<git hash / bundle>` | Runs: v2_main protect + edits (S09): `<run ids PENDING>`
Data status: {{num:synthetic}}

## Result first (plain language)
- v2_main PSR without purifier and under DiffPure t*=0.1 / 0.2 (SD1.5 s0.55): {{num:psr.v2_main.none}} / {{num:psr.v2_main.diffpure_0.1}} / {{num:psr.v2_main.diffpure_0.2}}; v1 under DiffPure t*=0.1: {{num:psr.v1.diffpure_0.1}}.
- E5 gate (PSR under held-out DiffPure t*=0.1 reaches at least half of the no-purifier PSR): **{{gate:E5}}**
- Caveat to state: DiffPure at several t* shares the SD1.5 UNet with the one-step training purifier (gray-box, V2_PLAN R17); GrIDPure: {{num:gridpure.status}}.
- Interpretation (write after reading the tables): `<PENDING>`

## E5 table
{{table:e5}}

## Headline table (PSR by method and purifier, SD1.5 s0.55)
{{table:headline}}

## All purifiers
{{table:headline_full}}

## Handoff note
If held-out DiffPure PSR is not clearly above Phase 3's at equal PSNR, the handoff says to raise the one-step purifier's share and retry once. The 48-hour plan builds all variants in parallel and has no retry: state this as a limitation if the gate is missed.

## What was done / GPU time / decisions
`<PENDING: jobs run, results/gpu_ledger.csv, decisions>`

## Open questions for the user (max 2)
1. `<PENDING>`
2. `<PENDING>`

Checkpoint: gate numbers reported as measured, a missed gate is reported, not changed.
