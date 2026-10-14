# Phase 2 report: baselines, controls and v1 under purification (template)

Template: render with `python scripts/make_tables.py --render reports/PHASE_2.md --tables_dir results/tables --out reports/rendered`.
Placeholders without data render as `PENDING (...)`; never type numbers by hand.

Date: `<PENDING>` | Code: `<git hash / bundle>` | Runs: noise controls (S14), PhotoGuard/Mist (S13), v1 purifier grid (S08): `<run ids PENDING>`
Data status: {{num:synthetic}}

## Result first (plain language)
- Noise control (uniform, same budget) PSR: {{num:psr.noise_uniform.none}}; Gaussian at matched PSNR: {{num:psr.noise_gauss_psnr.none}}; v1: {{num:psr.v1.none}}.
- v1 after JPEG q75 / down-up 0.5 / VAE round trip / DiffPure t*=0.1 (SD1.5 s0.55): {{num:psr.v1.jpeg_q75}} / {{num:psr.v1.downup_0.5}} / {{num:psr.v1.vae_rt}} / {{num:psr.v1.diffpure_0.1}}.
- E3 decision rule (continue with Phase 5 or skip): **{{gate:E3}}**
- Interpretation (write after reading the tables): `<PENDING>`

## E1: how much of the effect is just noise?
{{table:e1}}

## E2: PhotoGuard and Mist (our reimplementations) versus controls and v1
{{table:e2}}

## E3: v1 under each purifier
{{table:e3}}

## Paired differences versus the noise controls (no purifier)
{{table:paired_vs_noise}}

## Gate (E3 decision rule)
The cut-offs are analyst proposals stated with the verdict (the handoff gives none); change them with `--gate_param e3_collapse_rel=... e3_barely_rel=...` and re-render.
Handoff rule: if v1 does not degrade under purification, skip Phase 5 and spend the time on the ensemble and the study; if it collapses, continue as planned.
{{table:gates}}

## What was done
`<PENDING: noise controls, PhotoGuard/Mist runs (and whether they ran), purifier grid actually run per method (see coverage)>`

## GPU time used
`<PENDING: results/gpu_ledger.csv>`

## Decisions taken
`<PENDING>`

## Open questions for the user (max 2)
1. `<PENDING>` (ask only if the E3 verdict is AMBIGUOUS)
2. `<PENDING>`

Checkpoint: numbers reported as measured; later jobs do not wait for approval (48-hour plan).
