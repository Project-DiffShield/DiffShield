# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 1 file(s), 3900 rows
- new: 2 file(s), 3900 rows
- clean: 1 file(s), 8100 rows

## VERDICT: KEEP  (lower bound -0.0089 <= 0)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ipadapter_faceid | none | 0.55 | 100 | 0.020 | 0.013 | -0.007 [-0.020, +0.007] | KEEP | 0.000 | 0.020 | 0.013 | -0.007 [-0.020, +0.007] | - |
| sd15_img2img | diffpure_0.1 | 0.55 | 100 | 0.517 | 0.550 | +0.033 [-0.010, +0.077] | KEEP | 0.133 | 0.383 | 0.417 | +0.033 [-0.010, +0.077] | - |
| sd15_img2img | diffpure_0.2 | 0.55 | 100 | 0.570 | 0.567 | -0.003 [-0.040, +0.033] | KEEP | 0.217 | 0.353 | 0.350 | -0.003 [-0.040, +0.033] | - |
| sd15_img2img | diffpure_0.3 | 0.55 | 100 | 0.600 | 0.630 | +0.030 [-0.013, +0.073] | KEEP | 0.260 | 0.340 | 0.370 | +0.030 [-0.013, +0.073] | - |
| sd15_img2img | downup_0.5 | 0.55 | 100 | 0.190 | 0.137 | -0.053 [-0.083, -0.027] | KEEP | 0.097 | 0.093 | 0.040 | -0.053 [-0.083, -0.027] | - |
| sd15_img2img | jpeg_q50 | 0.55 | 100 | 0.203 | 0.190 | -0.013 [-0.047, +0.020] | KEEP | 0.130 | 0.073 | 0.060 | -0.013 [-0.047, +0.020] | - |
| sd15_img2img | jpeg_q75 | 0.55 | 100 | 0.233 | 0.227 | -0.007 [-0.043, +0.030] | KEEP | 0.103 | 0.130 | 0.123 | -0.007 [-0.043, +0.030] | - |
| sd15_img2img | none | 0.3 | 100 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | KEEP | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | - |
| sd15_img2img | none (HEADLINE) | 0.55 | 100 | 0.497 | 0.512 | +0.016 [-0.009, +0.040] | KEEP | 0.139 | 0.358 | 0.373 | +0.016 [-0.009, +0.040] | - |
| sd15_img2img | none | 0.75 | 100 | 0.900 | 0.910 | +0.010 [-0.023, +0.040] | KEEP | 0.840 | 0.060 | 0.070 | +0.010 [-0.023, +0.040] | - |
| sd15_img2img | vae_rt | 0.55 | 100 | 0.523 | 0.527 | +0.003 [-0.037, +0.043] | KEEP | 0.123 | 0.400 | 0.403 | +0.003 [-0.037, +0.043] | - |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: arcface_sim: -0.006 [-0.009, -0.003] (n=100)

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ipadapter_faceid | none | 0.55 | 0.611 | 0.606 | -0.004 [-0.011, +0.003] | n/a |
| sd15_img2img | diffpure_0.1 | 0.55 | 0.100 | 0.094 | -0.005 [-0.010, -0.000] | n/a |
| sd15_img2img | diffpure_0.2 | 0.55 | 0.094 | 0.086 | -0.008 [-0.014, -0.002] | n/a |
| sd15_img2img | diffpure_0.3 | 0.55 | 0.078 | 0.075 | -0.002 [-0.008, +0.003] | n/a |
| sd15_img2img | downup_0.5 | 0.55 | 0.197 | 0.208 | +0.010 [+0.007, +0.014] | n/a |
| sd15_img2img | jpeg_q50 | 0.55 | 0.187 | 0.192 | +0.005 [+0.001, +0.010] | n/a |
| sd15_img2img | jpeg_q75 | 0.55 | 0.174 | 0.176 | +0.002 [-0.003, +0.006] | n/a |
| sd15_img2img | none | 0.3 | 0.387 | 0.381 | -0.006 [-0.012, -0.000] | n/a |
| sd15_img2img | none | 0.55 | 0.108 | 0.102 | -0.006 [-0.009, -0.003] | n/a |
| sd15_img2img | none | 0.75 | 0.017 | 0.015 | -0.002 [-0.008, +0.005] | n/a |
| sd15_img2img | vae_rt | 0.55 | 0.104 | 0.103 | -0.000 [-0.005, +0.005] | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ipadapter_faceid | none | -0.004 [-0.011, +0.003] | n/a |
| sd15_img2img | diffpure_0.1 | -0.005 [-0.010, -0.000] | n/a |
| sd15_img2img | diffpure_0.2 | -0.008 [-0.014, -0.002] | n/a |
| sd15_img2img | diffpure_0.3 | -0.002 [-0.008, +0.003] | n/a |
| sd15_img2img | downup_0.5 | +0.010 [+0.007, +0.014] | n/a |
| sd15_img2img | jpeg_q50 | +0.005 [+0.001, +0.010] | n/a |
| sd15_img2img | jpeg_q75 | +0.002 [-0.003, +0.006] | n/a |
| sd15_img2img | none | -0.005 [-0.008, -0.003] | n/a |
| sd15_img2img | vae_rt | -0.000 [-0.005, +0.005] | n/a |
