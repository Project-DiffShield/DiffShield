# Phase 7 report: final experiments E0-E10, full results and every failure case (template)

Template: render with `python scripts/make_tables.py --render reports/PHASE_7.md --tables_dir results/tables --out reports/rendered`.
Placeholders without data render as `PENDING (...)`; never type numbers by hand. Run the pipeline in `README.md` (section "Reproduce every table") first.

Date: `<PENDING>` | Code: tag `v2.0-final` (`<git hash>`) | Runs: `<all job ids PENDING>`
Data status: {{num:synthetic}}

## Result first (plain language)
- Claim numbers (the handoff sentence): ArcFace identity retention of v2_final edits is lowered by **{{num:claim.A}}** versus the uniform-noise control and **{{num:claim.B}}** versus v1 (SD1.5 s0.55, no purifier, paired over the same images); adaptive attacker: PSR **{{num:claim.C}}**.
- Gates: E3 {{gate:E3}}; E4 {{gate:E4}}; E5 {{gate:E5}}; E6 {{gate:E6}}; invisibility {{gate:INV}}.
- Human study: {{num:human.task_a}} ({{num:human.pilot}}, raters {{num:human.n_raters}}).
- Interpretation (write after reading the tables, including everything that did not work): `<PENDING>`

## Claim table
{{table:claims}}

## Headline: PSR by method and purifier
{{table:headline}}

## Gates
{{table:gates}}

## E7 ablations
{{table:e7}}

{{table:e7_editors}}

{{table:e7_paired}}

## E8 adaptive attacker
{{table:e8}}

## E9 epsilon sweep
{{table:e9}}

Figure: `results/figures/psr_vs_psnr.png`.

## Per editor, per prompt, messy photos
{{table:per_editor}}

{{table:per_prompt}}

{{table:messy}}

## E10 human study
{{table:e10}}

## Invisibility of every method
{{table:invisibility_methods}}

## Failure cases (write all of them)
- Images where v2_final edits still match the person: `results/figures/failure_cases.csv` (gallery `failure_gallery.png` contains CelebA-HQ photos: local only).
- Editors/purifiers where protection is weak or collapses: `<PENDING: list from the tables, with numbers>`
- Negative results (ablations, E6, anything that did not help): `<PENDING>`
- Cells that did not run or have fewer images than planned (see coverage): {{table:coverage}}

## Figures produced
`results/figures/`: psr_vs_psnr, psr_by_purifier, per_editor, ablation, sim_distributions, before_after, failure_gallery, optuna_history, human_study (those whose inputs exist; see `figures_manifest.json`).

## GPU time / decisions / open questions
`<PENDING>`

Checkpoint: full results and every failure case reported; no gate was changed after the fact.
