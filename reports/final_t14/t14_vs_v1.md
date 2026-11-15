# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 1 file(s), 8700 rows
- new: 2 file(s), 3900 rows
- clean: 1 file(s), 8100 rows

## VERDICT: ADOPT  (lower bound +0.2300 > 0)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ip2p | diffpure_0.1 |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ip2p | jpeg_q75 |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ip2p | none |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | adaptive_ae | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | diffpure_0.1 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | diffpure_0.2 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | jpeg_q75 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | none | 0.55 | 100 | 0.000 | 0.013 | +0.013 [+0.000, +0.030] | KEEP | 0.000 | 0.000 | 0.013 | +0.013 [+0.000, +0.030] | - |
| ipadapter_faceid | vae_rt | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | adaptive_ae | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | diffpure_0.05 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | diffpure_0.1 | 0.55 | 100 | 0.263 | 0.550 | +0.287 [+0.227, +0.347] | ADOPT | 0.133 | 0.130 | 0.417 | +0.287 [+0.227, +0.347] | - |
| sd15_img2img | diffpure_0.2 | 0.55 | 100 | 0.317 | 0.567 | +0.250 [+0.187, +0.313] | ADOPT | 0.217 | 0.100 | 0.350 | +0.250 [+0.187, +0.313] | - |
| sd15_img2img | diffpure_0.3 | 0.55 | 100 | 0.360 | 0.630 | +0.270 [+0.210, +0.330] | ADOPT | 0.260 | 0.100 | 0.370 | +0.270 [+0.210, +0.330] | - |
| sd15_img2img | downup_0.5 | 0.55 | 100 | 0.097 | 0.137 | +0.040 [+0.007, +0.073] | ADOPT | 0.097 | 0.000 | 0.040 | +0.040 [+0.007, +0.073] | - |
| sd15_img2img | downup_other | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | jpeg_other | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | jpeg_q50 | 0.55 | 100 | 0.167 | 0.190 | +0.023 [-0.007, +0.057] | KEEP | 0.130 | 0.037 | 0.060 | +0.023 [-0.007, +0.057] | - |
| sd15_img2img | jpeg_q75 | 0.55 | 100 | 0.150 | 0.227 | +0.077 [+0.033, +0.120] | ADOPT | 0.103 | 0.047 | 0.123 | +0.077 [+0.033, +0.120] | - |
| sd15_img2img | jpeg_q90 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | none | 0.3 | 100 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | KEEP | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | - |
| sd15_img2img | none (HEADLINE) | 0.55 | 100 | 0.240 | 0.512 | +0.272 [+0.230, +0.317] | ADOPT | 0.139 | 0.101 | 0.373 | +0.272 [+0.230, +0.317] | - |
| sd15_img2img | none | 0.75 | 100 | 0.853 | 0.910 | +0.057 [+0.007, +0.110] | ADOPT | 0.840 | 0.013 | 0.070 | +0.057 [+0.007, +0.110] | - |
| sd15_img2img | restore_swin2sr | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | vae_rt | 0.55 | 100 | 0.220 | 0.527 | +0.307 [+0.247, +0.370] | ADOPT | 0.123 | 0.097 | 0.403 | +0.307 [+0.247, +0.370] | - |
| sd_inpaint | none | 1 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sdxl_img2img | none | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: arcface_sim: -0.075 [-0.083, -0.067] (n=100)

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ip2p | diffpure_0.1 |  | n/a | n/a | n/a | n/a |
| ip2p | jpeg_q75 |  | n/a | n/a | n/a | n/a |
| ip2p | none |  | n/a | n/a | n/a | n/a |
| ipadapter_faceid | adaptive_ae | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | diffpure_0.1 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | diffpure_0.2 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | jpeg_q75 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | none | 0.55 | 0.660 | 0.606 | -0.054 [-0.067, -0.042] | n/a |
| ipadapter_faceid | vae_rt | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | adaptive_ae | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.05 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.1 | 0.55 | 0.166 | 0.094 | -0.072 [-0.082, -0.062] | n/a |
| sd15_img2img | diffpure_0.2 | 0.55 | 0.151 | 0.086 | -0.065 [-0.076, -0.055] | n/a |
| sd15_img2img | diffpure_0.3 | 0.55 | 0.136 | 0.075 | -0.060 [-0.070, -0.051] | n/a |
| sd15_img2img | downup_0.5 | 0.55 | 0.232 | 0.208 | -0.024 [-0.031, -0.017] | n/a |
| sd15_img2img | downup_other | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_other | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_q50 | 0.55 | 0.209 | 0.192 | -0.017 [-0.022, -0.012] | n/a |
| sd15_img2img | jpeg_q75 | 0.55 | 0.206 | 0.176 | -0.030 [-0.037, -0.023] | n/a |
| sd15_img2img | jpeg_q90 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | none | 0.3 | 0.472 | 0.381 | -0.091 [-0.101, -0.081] | n/a |
| sd15_img2img | none | 0.55 | 0.177 | 0.102 | -0.075 [-0.083, -0.067] | n/a |
| sd15_img2img | none | 0.75 | 0.036 | 0.015 | -0.021 [-0.029, -0.012] | n/a |
| sd15_img2img | restore_swin2sr | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | vae_rt | 0.55 | 0.178 | 0.103 | -0.075 [-0.086, -0.064] | n/a |
| sd_inpaint | none | 1 | n/a | n/a | n/a | n/a |
| sdxl_img2img | none | 0.55 | n/a | n/a | n/a | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ipadapter_faceid | none | -0.054 [-0.067, -0.042] | n/a |
| sd15_img2img | diffpure_0.1 | -0.072 [-0.082, -0.062] | n/a |
| sd15_img2img | diffpure_0.2 | -0.065 [-0.076, -0.055] | n/a |
| sd15_img2img | diffpure_0.3 | -0.060 [-0.070, -0.051] | n/a |
| sd15_img2img | downup_0.5 | -0.024 [-0.031, -0.017] | n/a |
| sd15_img2img | jpeg_q50 | -0.017 [-0.022, -0.012] | n/a |
| sd15_img2img | jpeg_q75 | -0.030 [-0.037, -0.023] | n/a |
| sd15_img2img | none | -0.067 [-0.072, -0.061] | n/a |
| sd15_img2img | vae_rt | -0.075 [-0.086, -0.064] | n/a |
