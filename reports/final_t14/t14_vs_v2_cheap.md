# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 1 file(s), 6000 rows
- new: 2 file(s), 3900 rows
- clean: 1 file(s), 8100 rows

## VERDICT: ADOPT  (lower bound +0.2444 > 0)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ip2p | none |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | none | 0.55 | 100 | 0.000 | 0.013 | +0.013 [+0.000, +0.030] | KEEP | 0.000 | 0.000 | 0.013 | +0.013 [+0.000, +0.030] | - |
| sd15_img2img | diffpure_0.05 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | diffpure_0.1 | 0.55 | 100 | 0.267 | 0.550 | +0.283 [+0.220, +0.347] | ADOPT | 0.133 | 0.133 | 0.417 | +0.283 [+0.220, +0.347] | - |
| sd15_img2img | diffpure_0.2 | 0.55 | 100 | 0.313 | 0.567 | +0.253 [+0.180, +0.327] | ADOPT | 0.217 | 0.097 | 0.350 | +0.253 [+0.180, +0.327] | - |
| sd15_img2img | diffpure_0.3 | 0.55 | 100 | 0.347 | 0.630 | +0.283 [+0.217, +0.353] | ADOPT | 0.260 | 0.087 | 0.370 | +0.283 [+0.217, +0.353] | - |
| sd15_img2img | downup_0.5 | 0.55 | 100 | 0.143 | 0.137 | -0.007 [-0.040, +0.027] | KEEP | 0.097 | 0.047 | 0.040 | -0.007 [-0.040, +0.027] | - |
| sd15_img2img | downup_other | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | jpeg_other | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | jpeg_q50 | 0.55 | 100 | 0.173 | 0.190 | +0.017 [-0.020, +0.053] | KEEP | 0.130 | 0.043 | 0.060 | +0.017 [-0.020, +0.053] | - |
| sd15_img2img | jpeg_q75 | 0.55 | 100 | 0.167 | 0.227 | +0.060 [+0.013, +0.107] | ADOPT | 0.103 | 0.063 | 0.123 | +0.060 [+0.013, +0.107] | - |
| sd15_img2img | jpeg_q90 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | none | 0.3 | 100 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | KEEP | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | - |
| sd15_img2img | none (HEADLINE) | 0.55 | 100 | 0.224 | 0.512 | +0.288 [+0.244, +0.333] | ADOPT | 0.139 | 0.086 | 0.373 | +0.288 [+0.244, +0.333] | - |
| sd15_img2img | none | 0.75 | 100 | 0.847 | 0.910 | +0.063 [+0.017, +0.110] | ADOPT | 0.840 | 0.007 | 0.070 | +0.063 [+0.017, +0.110] | - |
| sd15_img2img | restore_swin2sr | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | vae_rt | 0.55 | 100 | 0.213 | 0.527 | +0.313 [+0.247, +0.383] | ADOPT | 0.123 | 0.090 | 0.403 | +0.313 [+0.247, +0.383] | - |
| sd_inpaint | none | 1 | - | - | - | - | n/a | - | - | - | - | cell missing in new |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: arcface_sim: -0.083 [-0.090, -0.075] (n=100)

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ip2p | none |  | n/a | n/a | n/a | n/a |
| ipadapter_faceid | none | 0.55 | 0.644 | 0.606 | -0.038 [-0.051, -0.025] | n/a |
| sd15_img2img | diffpure_0.05 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.1 | 0.55 | 0.173 | 0.094 | -0.078 [-0.090, -0.066] | n/a |
| sd15_img2img | diffpure_0.2 | 0.55 | 0.161 | 0.086 | -0.076 [-0.089, -0.063] | n/a |
| sd15_img2img | diffpure_0.3 | 0.55 | 0.142 | 0.075 | -0.067 [-0.078, -0.056] | n/a |
| sd15_img2img | downup_0.5 | 0.55 | 0.228 | 0.208 | -0.020 [-0.030, -0.010] | n/a |
| sd15_img2img | downup_other | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_other | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_q50 | 0.55 | 0.210 | 0.192 | -0.017 [-0.025, -0.011] | n/a |
| sd15_img2img | jpeg_q75 | 0.55 | 0.203 | 0.176 | -0.027 [-0.035, -0.020] | n/a |
| sd15_img2img | jpeg_q90 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | none | 0.3 | 0.477 | 0.381 | -0.096 [-0.104, -0.087] | n/a |
| sd15_img2img | none | 0.55 | 0.184 | 0.102 | -0.083 [-0.090, -0.075] | n/a |
| sd15_img2img | none | 0.75 | 0.034 | 0.015 | -0.018 [-0.027, -0.010] | n/a |
| sd15_img2img | restore_swin2sr | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | vae_rt | 0.55 | 0.184 | 0.103 | -0.081 [-0.094, -0.068] | n/a |
| sd_inpaint | none | 1 | n/a | n/a | n/a | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ipadapter_faceid | none | -0.038 [-0.051, -0.025] | n/a |
| sd15_img2img | diffpure_0.1 | -0.078 [-0.090, -0.066] | n/a |
| sd15_img2img | diffpure_0.2 | -0.076 [-0.089, -0.063] | n/a |
| sd15_img2img | diffpure_0.3 | -0.067 [-0.078, -0.056] | n/a |
| sd15_img2img | downup_0.5 | -0.020 [-0.030, -0.010] | n/a |
| sd15_img2img | jpeg_q50 | -0.017 [-0.025, -0.011] | n/a |
| sd15_img2img | jpeg_q75 | -0.027 [-0.035, -0.020] | n/a |
| sd15_img2img | none | -0.071 [-0.077, -0.066] | n/a |
| sd15_img2img | vae_rt | -0.081 [-0.094, -0.068] | n/a |
