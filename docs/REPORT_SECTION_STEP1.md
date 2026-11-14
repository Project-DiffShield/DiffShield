# Report text: systematic re-tuning and what else we tried (draft, 2026-11-14)

Draft text for docs/REPORT_DRAFT.md: new subsections 5.12-5.15 plus additions to sections 6, 7 and 8. Every number below
is measured unless marked PENDING (F1/W1/W2/X9 not yet run) or UNVERIFIED. T04, T05 and X6-X8 are filled in. Sources of the numbers: `reports/step1_selection/`,
`reports/step1_confirm/`, `reports/exploration/` (explore_compare.md, similarity.md). Wording rule: our evaluation
"DiffPure" is **DiffPure-style (SD1.5 img2img, empty prompt)**; the pixel-space cleaner of T05 is **DiffPure-style,
pixel-space face DDPM** (`google/ddpm-celebahq-256`), not the official DiffPure checkpoint (its download returned 403).

---

## 5.12 Re-choosing the loss weights with a pre-registered procedure

**Why.** The weights of the previous final method (alpha 0.55, beta 0.9024, gamma 60.53) were partly inherited from an earlier
prototype's search and partly chosen among three values after seeing results. We re-chose them, together with two new
components, with a procedure fixed before any trial ran (docs/TUNING_PROTOCOL.md).

**Tuning photos.** 28 search and 12 confirm photos were chosen by MMD-critic (Kim, Khanna & Koyejo, NeurIPS 2016) on CLIP
embeddings of the whole usable pool (28,974 photos; only test and messy photos excluded): greedy prototypes that minimise the
maximum mean discrepancy to the pool, plus 4 criticisms (photos the prototypes explain worst) in the confirm set. Against 1,000
random subsets of the same size, the search set is more representative than random (p = 0.001 in CLIP space, the space it was
optimised in; p = 0.008 in an independent attribute + pose space). The confirm set is representative in CLIP space (p = 0.003)
but only weakly in attribute + pose space (p = 0.20); we report this rather than re-draw. No two chosen photos are the same
person (largest pairwise ArcFace similarity 0.451). The selection is reproducible from the saved embeddings with one command
(`scripts/reproduce_tuning_selection.py`, prints MATCH).

**Search.** Optuna TPE (seed 2026, 8 random start-up trials, constant liar, 2 parallel workers), 25 trials on a cheap proxy
(20 PGD iterations, K = 2 purifier draws, 1 edit seed) over: alpha, beta, gamma; the denoiser term on/off with weight eta and
timestep range; and the share `p_onestep` of the one-step DiffPure-style cleaner in the training mixture. Objective: mean PSR over
2 editors (SD1.5 img2img at strength 0.55, IP-Adapter FaceID) x 5 cleanups (none, JPEG q75, down-up 0.5, VAE round trip,
DiffPure-style t = 0.2). Constraint, per photo: PSNR >= 38 dB and SSIM >= 0.95. Trial 0 was the previous final method.
Result: 18 of 25 trials met the constraint; the best feasible trial reached proxy PSR 0.232 against 0.204 for the previous method.
Four pairs of trials had identical settings (both workers drew the same start-up points; a seeding flaw on our side) and differed
by up to 0.021 PSR, which is the size of the search's noise. Several of the highest-PSR trials violated the constraint by a small
margin (minimum SSIM 0.93-0.94), i.e. part of the apparent gain was visibility. Strong denoiser weights (eta 1.8 and 89) gave
the worst trials of all (PSR 0.075-0.121); the best trials kept eta near the bottom of its range (about 0.01).

**Confirmation.** The top 3 feasible trials and the previous method were re-run at full settings (40 iterations, K = 4) on the
12 confirm photos with 3 edit seeds. PSR: previous method 0.211, trial 14 0.211, trial 17 0.203 (the best search trial),
trial 10 0.169; all with zero invisibility violations. The pre-registered rule ("zero violations and PSR >= the previous
method") accepted trial 14 on an exact tie. We read this as **no evidence of improvement**: the hand-set weights were already
about as good as the search could find. (In hindsight the rule should have required a strict improvement.)

**Final candidate.** Trial 14 (alpha 0.188, beta 0.546, gamma 21.6, denoiser eta 0.012 on timesteps 50-300, one-step
DiffPure-style cleaner in 30% of training draws) is evaluated on the 100 test photos against the previous method with a paired
bootstrap (`scripts/paired_compare.py`, adopt only if the 95% interval of the PSR difference is above 0):
**verdict KEEP**. Headline cell (SD1.5 img2img, no cleanup, strength 0.55): PSR 0.497 (previous) vs 0.512 (trial 14),
difference +0.016 [-0.009, +0.040] (10,000 paired draws over the 100 photos, seed 2026). Mean ArcFace similarity on the same
cell moved slightly in the right direction, -0.006 [-0.009, -0.003] (secondary measure, not part of the rule): a real but very
small effect. Other cells: no cell improved with an interval above 0; down-up resizing got worse (PSR 0.190 -> 0.137,
-0.053 [-0.083, -0.027]); IP-Adapter FaceID stays near 0 (0.020 -> 0.013). Both methods are equally invisible on the test
photos (mean PSNR 40.9 vs 41.0 dB, lowest SSIM 0.962 vs 0.959, no photo below the limit). The claim is therefore **"on par
with the previous method"**, not "better" (`reports/step1_final/T04_paired_main.md`).

**Does training with the cleaner help against cleaners?** The previous method did not see a cleaner in training, trial 14
saw the one-step SD1.5 DiffPure-style cleaner in 30% of draws. On the DiffPure-style (SD1.5 img2img, empty prompt) evaluation
cleanups trial 14 is ahead at t = 0.1 and 0.3 (+0.033 and +0.030) but neither interval excludes 0; similarity is slightly lower at
t = 0.1 and 0.2 (-0.005, -0.008). On the held-out DiffPure-style, pixel-space face DDPM cleaner it was never trained on, it is
slightly worse (t = 200: PSR -0.013 [-0.027, -0.003]; `reports/step1_final/T04_paired_held.md`). Training against one cleaner
did not transfer to a different one.

## 5.13 What else we tried for a larger gain (exploration, 28 search photos)

Each variant changes one thing relative to the previous method; all were protected at full settings and evaluated on the same
2 editors x 5 cleanups x 3 seeds (840 edits per variant). Exploration only: these are tuning photos, not test results.

| Variant | Change | PSR all cells | Diff to baseline [95% CI] | PSR IP-Adapter | Photos violating PSNR/SSIM limit |
|---|---|---|---|---|---|
| X0 | none (baseline) | 0.186 | - | 0.000 | 0 of 28 |
| X1 | budget 8/255 | 0.225 | +0.039 [+0.020, +0.057] | 0.002 | 12 of 28 |
| X2 | 100 PGD iterations | 0.190 | +0.005 [-0.013, +0.020] | 0.000 | 0 of 28 |
| X3 | identity loss on the photo (FaceNet), weight 2 | 0.168 | -0.018 [-0.032, -0.004] | 0.000 | 0 of 28 |
| X4 | same, weight 10 | 0.156 | -0.030 [-0.045, -0.015] | 0.000 | 1 of 28 |
| X5 | identity loss after the VAE round trip, weight 2 | 0.181 | -0.005 [-0.027, +0.015] | 0.000 | 0 of 28 |
| X8 | final candidate (trial 14), 4/255 | 0.189 | +0.004 [-0.018, +0.023] | 0.000 | 0 of 28 |
| X7 | final candidate, 6/255 | 0.213 | +0.027 [+0.001, +0.051] | 0.000 | 1 of 28 |
| X6 | final candidate, 8/255 | 0.213 | +0.027 [+0.002, +0.050] | 0.002 | 12 of 28 |

Findings: (i) more iterations do not help, the 40-iteration attack has converged; (ii) a larger budget helps, but only by
becoming visible (mean PSNR 40.9 -> 38.1 dB, 12 of 28 photos below the limit), so it is a trade-off, not a better method;
(iii) the identity loss lowers PSR on SD1.5 and leaves IP-Adapter FaceID PSR at 0 (explained in 5.14); (iv) at 4/255 the final
candidate equals the baseline (X8 vs X0, +0.004), as on the test photos; raising its budget to 6/255 gains +0.027, and 8/255
gains nothing more (the same +0.027) while making 12 of 28 photos visibly worse.

## 5.14 Why the identity loss did not raise PSR

The identity loss did what it was asked: the surrogate FaceNet's cosine between the protected and the original face fell from
1.00 to -0.28 (X3), -0.42 (X4) and -0.65 (X5) on average, i.e. FaceNet no longer recognises the person. The question is how much
of that reaches the face model that matters. Mean ArcFace similarity between edit and source (lower = less like the person;
`reports/exploration/similarity.md`):

| Editor | X0 baseline | X1 (8/255) | X3 | X4 | X5 |
|---|---|---|---|---|---|
| IP-Adapter FaceID | 0.595 | 0.576 (-0.019 [-0.026, -0.012]) | 0.556 (-0.039 [-0.051, -0.028]) | 0.555 (-0.039 [-0.051, -0.028]) | 0.551 (-0.044 [-0.055, -0.034]) |
| SD1.5 img2img | 0.131 | 0.118 | 0.137 (+0.006) | 0.139 (+0.008) | 0.132 (+0.000) |

So the identity loss **does** weaken identity under IP-Adapter, about twice as much as doubling the budget, and the effect is
clearly outside noise. It is invisible in PSR because IP-Adapter edits stay far above the threshold: the lowest similarity of any
X3-X5 edit is 0.168, against a threshold of 0.1005. Three reasons, in order of how sure we are:

1. **The editor reads identity through ArcFace, we attacked FaceNet.** IP-Adapter FaceID conditions generation on the
   insightface ArcFace embedding of the input photo (our editor uses the same `buffalo_l` model that scores the edits). An
   adversarial perturbation found against one face network transfers only partly to another; FaceNet's similarity went negative
   while ArcFace's barely moved. Increasing the weight from 2 to 10 changed nothing on ArcFace (-0.039 both), so the limit is
   transfer, not optimisation strength. Published face-cloaking tools address this with ensembles of face models and input
   augmentation (e.g. LowKey, Fawkes); we used one surrogate. (Measured: the numbers above. Mechanism: our interpretation.)
2. **Geometry mismatch.** The loss uses a fixed central 60% crop resized to 160 px; ArcFace detects, aligns and resizes the face to
   112 px. A 4/255 pattern optimised for one framing is resampled by the other. (UNVERIFIED; not measured separately.)
3. **Shared budget.** The identity term competes for the same 4/255 perturbation as the latent attack; SD1.5 protection fell
   slightly (similarity +0.006 to +0.008, PSR -0.018 to -0.030), so the budget it took was not free.

Not tried (time): an identity loss on ArcFace itself. For IP-Adapter this is the editor's own encoder, so it is the right white-box
threat model, but the same network also scores the edits, so such a result must be checked with a second, independent face
recogniser (see 7, item 15).

## 5.15 Budget: where 4/255 sits

There is no accepted universal invisibility bound; image-protection work mostly uses larger budgets than ours:

| Work | Budget | Status of the figure |
|---|---|---|
| Mist (Liang et al., ICML 2023) | L-inf 16/255 (default) | read from the released code |
| Glaze (Shan et al., USENIX Security 2023) | perceptual, LPIPS < 0.05-0.07 | paper / authors' follow-up |
| Anti-DreamBooth (Van Le et al., ICCV 2023) | L-inf about 0.03-0.05, with a budget ablation | secondary summary; check the paper before citing |
| PhotoGuard (Salman et al., ICML 2023) | L-inf, value to be checked | UNVERIFIED |
| MetaCloak-JPEG (2026) | L-inf 8/255 | paper |

We keep 4/255 (fixed before results) as the headline and report the trade-off as a curve, PSR and PSNR at 4, 6 and 8/255 for the
final candidate on the same 28 search photos (`reports/exploration/explore_compare.md`):

| Budget | PSR all cells | PSR SD1.5 | Diff to X0 [95% CI] | Mean / min PSNR (dB) | Min SSIM | Photos below limit |
|---|---|---|---|---|---|---|
| 4/255 (X8) | 0.189 | 0.379 | +0.004 [-0.018, +0.023] | 40.88 / 39.63 | 0.960 | 0 of 28 |
| 6/255 (X7) | 0.213 | 0.426 | +0.027 [+0.001, +0.051] | 39.00 / 36.85 | 0.954 | 1 of 28 |
| 8/255 (X6) | 0.213 | 0.424 | +0.027 [+0.002, +0.050] | 38.12 / 35.19 | 0.951 | 12 of 28 |

The curve flattens after 6/255: 8/255 costs almost a further 1 dB of PSNR and 11 more photos below the limit for no extra PSR.
6/255 is the only point with a gain whose interval excludes 0 at near-acceptable visibility (one photo below the limit), and the
interval's lower end is close to 0. With the previous weights, 8/255 gave +0.039 (X1) with the same 12 of 28 photos below the
limit. These are tuning photos; none of the larger budgets was run on the test photos.

---

## Additions to section 6 (failure cases)
- IP-Adapter FaceID: PSR 0 for every method and variant; edits keep ArcFace similarity about 0.55-0.60 (threshold 0.1005).
  Protection that targets the SD1.5 VAE and UNet does not reach an editor that takes identity from a face-recognition embedding.
  On the 100 test photos the previous method reaches 0.020 and trial 14 0.013.
- A held-out DiffPure-style, pixel-space face DDPM cleaner (`google/ddpm-celebahq-256`, applied at 256 px) removes most of the
  protection (T05, previous method, 100 test photos). PSR difference protected minus unprotected, paired by photo: after resizing
  to 256 px and back alone (the control), +0.090 [+0.050, +0.130], so protection survives the resize; after the cleaner at
  t = 100, -0.003 [-0.027, +0.020], i.e. nothing left; at t = 200, +0.023 [+0.007, +0.043]. The resize does not explain the loss;
  the diffusion step does. The cleaner also changes unprotected faces on its own (PSR about 0.17 on clean photos). Trial 14, trained with a different (SD1.5, latent) cleaner, is no better here
  (5.12).

## Additions to section 7 (limitations)
14. Tuning and exploration used 28 + 12 photos; one cell of the confirm PSR moves in steps of about 0.03, so gains below that
    size are not detectable there. The final claim uses the 100 test photos.
15. A single face recogniser (insightface `buffalo_l`) scores all edits, and it is also the identity encoder of the IP-Adapter
    editor. The second recogniser planned in the design (`fr2_sim`) was not computed in any run (the column is empty).
    Addressed by: job F1 (`scripts/score_fr2.py`), which scores every saved edit post hoc with FaceNet (vggface2) after MTCNN
    detection, a recogniser and detector independent of insightface, with its own calibrated threshold and an fr2-PSR table. It is
    not independent for X3-X5, which were trained against that FaceNet. Jobs W1/W2 additionally evaluate the final comparison on
    SDXL img2img, InstructPix2Pix and SD inpainting. Results of F1/W1/W2 are not in this section until the jobs have run.
16. PSR is a yes/no count at a strict threshold; it hides graded changes in identity (5.14). We therefore also report mean
    ArcFace similarity with paired intervals.
    Addressed by: `scripts/paired_compare.py` now also reports, as secondary numbers next to the PSR rule (which is unchanged), the
    paired bootstrap of the difference in mean ArcFace similarity (and mean fr2 similarity when F1 has filled it), per headline cell
    and per editor x cleanup.
17. The exploration runs have no unprotected (clean) control on the 28 search photos; the clean rows exist for the test photos only.
    Addressed by: job X9 (tier EXPL, method `clean`, same cells and cleanups as X0-X8 on the same 28 photos); `explore_compare.py` and
    `explore_similarity.py` list it as label C (never as a base and not in the difference to X0).

## Addition to section 8 (claims)
Do not claim: protection against IP-Adapter FaceID or any editor that conditions on a face-recognition embedding; that the
re-tuned weights improve on the previous ones (T04: KEEP, +0.016 [-0.009, +0.040]); that training with the one-step cleaner
improves robustness to cleaners (no interval above 0 on any cleaner cell); that a larger budget is a better method (it is a
visibility trade-off, measured on 28 tuning photos only). May claim: the re-tuned method with the cleaner and the denoiser term
is on par with the previous final method on the 100 test photos at the same invisibility.
