# Phase 1 report: evaluation harness and v1 identity retention (template)

Template. Fill the numbers by rendering: `python scripts/make_tables.py --render reports/PHASE_1.md --tables_dir results/tables --out reports/rendered`
(after `scripts/ingest_results.py`, `scripts/compute_edit_lpips.py`, `scripts/make_tables.py`). A placeholder without data renders as `PENDING (...)`;
never type a number by hand. Sentences below are neutral: write the interpretation only after the tables exist.

Date: `<PENDING>` | Code: `<git hash / bundle>` | Runs: E0 (S07), v1 protect + edits (S08), harness smoke (S05): `<run ids PENDING>`
Data status: {{num:synthetic}}

## Result first (plain language)
- v1 protection success rate (SD1.5 img2img s0.55, no purifier, test split): {{num:psr.v1.none}} (images: {{num:n.v1.none}}); clean images: {{num:psr.clean.none}}.
- PSR threshold (95th percentile of ArcFace similarity between edits of clean photos and other people): pooled {{num:threshold.pooled}}.
- Mean ArcFace similarity to the source, v1 edits: {{num:sim.v1.none}}; clean edits: {{num:sim.clean.none}}.
- Interpretation (write after reading the tables): `<PENDING>`

## E0: reference edit distribution and threshold
{{table:e0}}

## v1 identity retention per editor (no purifier)
{{table:per_editor}}

## Coverage (what ran)
{{table:coverage}}

## What was done
- Harness (A3): `scripts/evaluate.py` edit + score in one job, 5 editors, 13 purifiers, tiers H/M/P/A/E9/E8, resume by row key, JPEG q90 edits.
- Analysis (A5): `scripts/ingest_results.py`, `src/eval/analysis.py`, `scripts/make_tables.py`, `scripts/compute_edit_lpips.py`.
- Edit LPIPS was computed locally after the runs on JPEG q90 versions of both edits (fallback rule 6): rows still without LPIPS: `<PENDING: see results/combined/ingest_report.json>`.

## GPU time used
`<PENDING: fill from results/gpu_ledger.csv, never estimates>`

## Decisions taken
`<PENDING>` (e.g. PSR threshold scope: per editor, pooled threshold also reported).

## Open questions for the user (max 2)
1. `<PENDING>`
2. `<PENDING>`

Checkpoint: Phase 1 numbers written honestly; the next job does not wait for approval (48-hour plan).
