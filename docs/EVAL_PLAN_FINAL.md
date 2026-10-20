# DiffShield v2: status and full evaluation plan (written Tue 2026-10-20 ~15:30 IST; deadline Thu 2026-10-22 06:00 IST, about 38 h)

Rules that still bind everything below: never invent a number; every number needs n, mean +/- 95% bootstrap CI over images, 3 edit seeds, and a paired comparison on the same images; never tune on the test photos; smoke-test numbers are never reported; unverified things are labelled; nothing is pushed or published without the user's say-so; face images are never committed.

## A. Status (from the files on this PC; sessions that ran after 2026-10-16 00:25 are unknown to me)

Done and verified (outputs unpacked under `results/`, checked, ingested):
| Piece | Evidence |
|---|---|
| Phase 0: budget = 4/255, v1 reproduced (PSNR 41.986, SSIM 0.9754, LPIPS 0.2314, n = 70) | `results/J01`, `reports/PHASE_0.md` |
| Frozen photo lists: test 100 (incl. all 70 v1 photos), messy 20, train 20, val 20, AE pool 120 | `data/splits`, S03 |
| E0 clean reference edits, headline tier (5 editors x 100 photos x 3 seeds) + messy (SD1.5, FaceID) | S07: 2820 rows |
| PSR threshold = 95th percentile of impostor ArcFace similarity: pooled 0.103 (per editor 0.100-0.110), 261,162 pairs | `results/S07/.../thresholds` |
| E3 v1: protection (100 + 20 photos, exactly 4/255) + headline/messy edits | S08: 2820 rows |
| E5 v2_main: protection (100 + 20) + headline/messy edits; PSNR >= 40.3, SSIM >= 0.97, 0 of 200 below 38 | S09: 2520 rows |
| Optuna on val (16 trials, 20-photo confirm) -> tuned v2_final config (trial 5, antipure ON); v2_final protected for the 100 test photos | S06; `config/parts/final.yaml`, `reports/optuna_*S06*` |
| Pipeline code: ingest, tables, figures, human-study scripts, demo; 350 CPU tests passing | repo |

NOT done (no result files exist for any of these):
| Missing piece | Experiment | Session |
|---|---|---|
| Purifier tier (13 purifiers) for clean, v1, v2_main | E0-P, E3, E5 | S10 (clean P), S19 (v1 P, v2_main P) |
| v2_final edits (headline, purifiers, messy, SDXL) | Phase 6 / headline | S16 |
| v2_cheap | E4 | S10 |
| v2_antipure | E6 | S12 |
| Noise controls | E1 | S22 |
| PhotoGuard, Mist | E2 | S13, S21 |
| Ablations vis_only / den_only / no_sem / no_pur | E7 | S11, S14 |
| Epsilon sweep 2 / 6 / 8 | E9 | S17, S23, S18 |
| Adaptive attacker | E8 | S15 |
| Human study | E10 | after S16 |
| Edit-LPIPS (513 of ~5,300 pairs done; very slow on CPU) | secondary metric | local, headline setting only |
| Tables, figures, report text, demo run, README reproduction, final tag | Phase 8 | local, after ingest |

First measured signal (preliminary, SD1.5 strength 0.55, no purifier, 100 photos, v1 vs clean): PSR 13.9% -> 24.0%, paired difference +10.1 points [7.3, 13.0]; ArcFace similarity -0.045 [-0.053, -0.039]. At strength 0.3 neither reaches PSR above 0%; at 0.75 clean edits already reach 84%; InstructPix2Pix shows v1 with LOWER PSR than clean (confounded by how often a face is found). So the effect of v1 is real but modest, and the strength setting dominates the reading. Everything about v2 is still unmeasured.

Known limitations to state (all real): v1 Phase 2 outputs were lost (0.46 can only be quoted); DPM-Solver++ sampler for all editors; DiffPure is the SD1.5-based "DiffPure-style" version; GrIDPure unavailable; IMPRESS and InstantID not run; 40 PGD iterations and K = 4; training identity loss `L_id` implemented but not used (lambda_id = 0); Optuna 16 trials x 10 photos with a proxy objective (its best proxy trial did not hold up on 20 confirm photos; tuned vs hand-set is within noise); single CLIP; same-person de-duplication used ArcFace similarity >= 0.5 (unverified threshold) and the splitter's detector missed 38 of 100 test faces, so de-duplication there must be re-checked with the corrected detector; adaptive attacker is weak (120 photos, 1 GPU-hour); edit-LPIPS only for the headline setting.

## B. What is evaluated, and how (the protocol)

Unit of analysis: the photo (n = 100 test photos; 20 messy photos reported separately). For each photo, edits of 3 seeds (and 3 prompts for the headline setting) are averaged FIRST, so confidence intervals are over photos, not over edits.

Metrics per edit: ArcFace cosine similarity to the source (insightface buffalo_l, face found after padding the image); "protected" = similarity below the editor-pooled threshold from E0 OR no face found (no-face rate is always reported next to PSR); edit-LPIPS (clean edit vs protected edit, both JPEG q90) for the headline setting; CLIP-I (secondary); invisibility PSNR, SSIM, LPIPS of the protected image (published PNG convention; must stay PSNR >= 38 and SSIM >= 0.95).

Statistics: 95% bootstrap CI (2,000 resamples) over photos; every method comparison is paired on the same photos (v2 vs v1, v2 vs the noise control, ablation vs v2_main, purified vs unpurified); report the paired difference with its CI, never two separate CIs side by side; a difference counts as "supported" only if its CI excludes 0. Multiple comparisons: say so and do not claim more than the CIs allow. Strength 0.3 / 0.55 / 0.75 are reported separately (do not average them).

Two modes, never mixed: non-adaptive (headline) and adaptive (E8, a stress test).

## C. Experiments, sessions and the exact outputs

| Exp | Question | Methods x purifiers x editors | Session(s) | Main table / figure |
|---|---|---|---|---|
| E0 | reference edits, PSR threshold | clean x (none; 13 purifiers on SD1.5) x all | S07 (done), S10 (clean P) | `thresholds`, E0 table |
| E1 | how much is just noise? | noise_uniform, noise_gauss_psnr x none x all | S22 | PSR vs noise, paired |
| E2 | vs existing tools | photoguard, mist x (none; 5 purifiers) | S13, S21 | PSR table, vs v1 |
| E3 | does purification break v1? | v1 x 13 purifiers (SD1.5) + FaceID/IP2P subsets | S08 (none, done), S19 (P) | PSR by purifier (the decision table) |
| E4 | do cheap purifiers in the loop help? | v2_cheap x 13 purifiers | S10 | gate: PSR under JPEG / down-up / VAE round trip at least doubles vs v1 at PSNR >= 38 |
| E5 | does the one-step purifier transfer? | v2_main x 13 purifiers incl. held-out DiffPure 0.1-0.3 | S09 (none, done), S19 (P) | gate: PSR under DiffPure 0.1 at least half of the no-purifier PSR |
| E6 | anti-purification terms | v2_antipure x 13 purifiers + FaceID none | S12 | vs v2_main, paired |
| E7 | which term / phase matters | abl_vis_only, abl_den_only, abl_no_sem, abl_no_pur, and v2_main as "both" x {none, jpeg_q50, vae_rt, diffpure_0.1, 0.2} + FaceID/IP2P/inpaint none | S11, S14 | ablation bars, paired vs v2_main |
| Final | tuned config | v2_final x all tiers | S06 (protect, done), S16 (edits) | headline claim |
| E8 | adaptive attacker | clean, v1, v2_final x adaptive_ae x (SD1.5, FaceID) | S15 | PSR drop vs non-adaptive |
| E9 | epsilon sweep | eps2, eps6, eps8 (+ eps4 = v2_final) x {none, jpeg75, diffpure0.1} SD1.5 + FaceID none | S17, S23, S18 | PSR vs PSNR, trade-off curve |
| E10 | human study | clean vs v2_final pairs | after S16 | 2AFC accuracy; identity question |

Decision rules after the data arrives (state the outcome honestly either way):
1. If v1's PSR barely drops under purification (E3), the purifier work did not matter; say so. If it collapses, E4/E5 are the main result.
2. Gates are reported as pass/fail with the numbers and CIs. A failed gate is a finding, not something to hide or re-define.
3. The headline claim template (fill with measured values only): "Under the stated threat model (white-box SD1.5 encoder, listed editors and purifiers, eps = 4/255), v2_final changes the protection success rate from X% (noise control) / Y% (v1) to Z% (paired difference +A points [CI]), after purifiers P1..Pk: ..., adaptive attacker: ...".
4. Strength 0.75 is saturated for clean photos (identity is already destroyed); the informative regime is strength 0.55 and the identity-conditioned FaceID editor. Say this in the results.

## D. Integrity checks before any number is trusted (do every time a zip arrives)
1. `kaggle_session.json` has `ok: true`, and `logs/job.log` has no `rc=1`; look for `failed_editors.json` and `NOT AVAILABLE` lines.
2. Row counts match the plan: for each (method, editor, purifier) 100 photos x 3 seeds (x 3 prompts for the headline). `coverage_vs_plan.csv` lists every missing cell; zero rows in a tier is a bug, not a result.
3. Face-found rate per cell (expect >= 99% for SD1.5/SDXL/FaceID/inpaint; 8-18% no-face for InstructPix2Pix); NaN counts.
4. Invisibility per method: max pixel change = eps, PSNR >= 38, SSIM >= 0.95 (report violations).
5. Git hash in every session equals the announced bundle; smoke sessions (S03, S04*, S05*, J00) are excluded by the ingest script.
6. No duplicate keys with different values (ingest reports conflicts).

## E. Local pipeline, in order (run on this PC; commands in README section "Reproduction")
1. `python scripts/ingest_results.py --results results` (never deletes; excludes smoke sessions).
2. `python scripts/compute_edit_lpips.py ... --workers 4`: headline setting only (SD1.5 strength 0.55), CPU about 4 s per pair; everything else shows "not computed".
3. `python scripts/make_tables.py --combined results/combined --out results/tables` -> E0-E10 tables, gates, claim numbers (`numbers.json`).
4. `python scripts/make_figures.py ...` -> PSR vs PSNR, bars by purifier / editor / ablation, ArcFace distributions, before/after panels (uses local copies of the images, never committed), failure gallery, Optuna history.
5. Failure cases: list the photos where v2_final does NOT protect (PSR failures per editor and purifier), show 6-10 panels, and explain the pattern (pose, glasses, blur, strength 0.3, face not found).
6. Human study (E10): `python scripts/make_human_study.py` from the S16 edits -> forms for the students; `analyze_human_study.py` when answers arrive (report as a pilot with the actual n if fewer than 10 raters).
7. Render `reports/PHASE_*.md`, fill `docs/REPORT_DRAFT.md`, `scripts/demo.py` on one photo, README reproduction check from the saved config, `git tag v2.0-final` (ask before pushing anything).

## F. Schedule (IST)
| When | What |
|---|---|
| Tue 20 Oct 15:30-17:30 | create dataset `diffshield-code-v7` from `kaggle/dist/B7/diffshield-code-b72e5ff.zip` (git hash `b72e5ff`), share with acct3-5, launch wave 1 (table in G). Nothing may launch after ~18:00 (it must end by ~05:00) |
| Tue 20 Oct night | wave 1 runs (sessions end 23:00-04:00). Locally: nothing blocks; prepare report text and figures code |
| Wed 21 Oct 06:00-10:00 | download wave 1 zips as they finish; ingest; integrity checks (D); first full tables; list gaps and failures |
| Wed 21 Oct 10:00-18:00 | wave 2 = re-runs and gaps only (each <= 8 h); human study sessions; failure-case analysis; report drafting |
| Wed 21 Oct 18:00 | compute freeze. Final ingest, final LPIPS, final tables and figures |
| Wed 21 Oct 18:00 - Thu 22 Oct 04:00 | report write-up, README reproduction, demo, tag, package, final review |
| Thu 22 Oct 04:00-06:00 | buffer only |

## G. Wave 1 launch table (all with dataset v7, `celebamaskhq`, GPU T4 x2, Internet ON, Save & Run All; check the announced git hash)
| Acct | A | B | Notes |
|---|---|---|---|
| acct1 | `["S16"]` + S06 output as input, `{"prev_root": ...}` | `["S15"]` | S16 ~6.5 h; S15 ~5 h |
| acct2 | `["S19"]` + S08 and S09 outputs as inputs, `{"prev_root_s08": ..., "prev_root_s09": ...}` | `["S18"]` | S19 ~5.5 h; S18 ~6.3 h |
| acct3 | `["S10"]` | `["S11"]` | S10 ~7.5 h; S11 ~8.5 h |
| acct4 | `["S12"]` | `["S17"]` | S12 ~7 h; S17 ~6.3 h |
| acct5 | `["S13"]` | `["S14"]` | S13 ~6.5 h; S14 ~8.5 h |
| second wave on whichever slot frees first | `["S23"]` (eps 6, ~6.3 h), `["S22"]` (noise, ~3 h), `["S21"]` (Mist, ~3 h) | | |
Total about 82 session-hours; if the weekly quota has been reset to 30 h per account (UNVERIFIED, ask), this fits with room. If a quota is short, drop in this order: S21, S15, S13, S12, S18, S23, S17, S11, S14, S22, S10, S16, S19.

## H. Contingencies
- A session fails: send me its log; re-run with `RESUME_FROM` (finished images are skipped). Do not edit notebooks or datasets while a session runs (it restarts).
- A session would pass 11 h: it stops starting new work and leaves a partial result; the partial rows are kept; report n honestly.
- Time runs out: freeze at the Wed 18:00 compute freeze and report what exists; the minimum set that still supports the main claim is E0-E5 (S10, S19, S16) plus E1 (S22).
- A claim does not hold: write it as a negative result with the numbers.
