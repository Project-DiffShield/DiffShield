# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 1 file(s), 8100 rows
- new: 2 file(s), 3900 rows

## VERDICT: ADOPT  (lower bound +0.3267 > 0)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ip2p | none |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | adaptive_ae | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | diffpure_0.1 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | diffpure_0.2 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | jpeg_q75 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | none | 0.55 | 100 | 0.000 | 0.013 | +0.013 [+0.000, +0.030] | KEEP | n/a | n/a | n/a | n/a | - |
| ipadapter_faceid | vae_rt | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | adaptive_ae | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | diffpure_0.05 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | diffpure_0.1 | 0.55 | 100 | 0.133 | 0.550 | +0.417 [+0.350, +0.483] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd15_img2img | diffpure_0.2 | 0.55 | 100 | 0.217 | 0.567 | +0.350 [+0.280, +0.420] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd15_img2img | diffpure_0.3 | 0.55 | 100 | 0.260 | 0.630 | +0.370 [+0.300, +0.440] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd15_img2img | downup_0.5 | 0.55 | 100 | 0.097 | 0.137 | +0.040 [+0.010, +0.070] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd15_img2img | downup_other | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | jpeg_other | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | jpeg_q50 | 0.55 | 100 | 0.130 | 0.190 | +0.060 [+0.027, +0.093] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd15_img2img | jpeg_q75 | 0.55 | 100 | 0.103 | 0.227 | +0.123 [+0.077, +0.170] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd15_img2img | jpeg_q90 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | none | 0.3 | 100 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | KEEP | n/a | n/a | n/a | n/a | - |
| sd15_img2img | none (HEADLINE) | 0.55 | 100 | 0.139 | 0.512 | +0.373 [+0.327, +0.422] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd15_img2img | none | 0.75 | 100 | 0.840 | 0.910 | +0.070 [+0.020, +0.120] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd15_img2img | restore_swin2sr | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | vae_rt | 0.55 | 100 | 0.123 | 0.527 | +0.403 [+0.340, +0.470] | ADOPT | n/a | n/a | n/a | n/a | - |
| sd_inpaint | none | 1 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sdxl_img2img | none | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: arcface_sim: -0.121 [-0.131, -0.110] (n=100)

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ip2p | none |  | n/a | n/a | n/a | n/a |
| ipadapter_faceid | adaptive_ae | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | diffpure_0.1 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | diffpure_0.2 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | jpeg_q75 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | none | 0.55 | 0.701 | 0.606 | -0.095 [-0.110, -0.081] | n/a |
| ipadapter_faceid | vae_rt | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | adaptive_ae | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.05 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.1 | 0.55 | 0.210 | 0.094 | -0.116 [-0.129, -0.104] | n/a |
| sd15_img2img | diffpure_0.2 | 0.55 | 0.194 | 0.086 | -0.108 [-0.123, -0.094] | n/a |
| sd15_img2img | diffpure_0.3 | 0.55 | 0.172 | 0.075 | -0.096 [-0.108, -0.085] | n/a |
| sd15_img2img | downup_0.5 | 0.55 | 0.243 | 0.208 | -0.035 [-0.041, -0.028] | n/a |
| sd15_img2img | downup_other | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_other | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_q50 | 0.55 | 0.222 | 0.192 | -0.030 [-0.035, -0.024] | n/a |
| sd15_img2img | jpeg_q75 | 0.55 | 0.227 | 0.176 | -0.051 [-0.058, -0.043] | n/a |
| sd15_img2img | jpeg_q90 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | none | 0.3 | 0.522 | 0.381 | -0.141 [-0.151, -0.132] | n/a |
| sd15_img2img | none | 0.55 | 0.223 | 0.102 | -0.121 [-0.131, -0.110] | n/a |
| sd15_img2img | none | 0.75 | 0.038 | 0.015 | -0.023 [-0.033, -0.014] | n/a |
| sd15_img2img | restore_swin2sr | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | vae_rt | 0.55 | 0.228 | 0.103 | -0.124 [-0.137, -0.111] | n/a |
| sd_inpaint | none | 1 | n/a | n/a | n/a | n/a |
| sdxl_img2img | none | 0.55 | n/a | n/a | n/a | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ipadapter_faceid | none | -0.095 [-0.110, -0.081] | n/a |
| sd15_img2img | diffpure_0.1 | -0.116 [-0.129, -0.104] | n/a |
| sd15_img2img | diffpure_0.2 | -0.108 [-0.123, -0.094] | n/a |
| sd15_img2img | diffpure_0.3 | -0.096 [-0.108, -0.085] | n/a |
| sd15_img2img | downup_0.5 | -0.035 [-0.041, -0.028] | n/a |
| sd15_img2img | jpeg_q50 | -0.030 [-0.035, -0.024] | n/a |
| sd15_img2img | jpeg_q75 | -0.051 [-0.058, -0.043] | n/a |
| sd15_img2img | none | -0.104 [-0.112, -0.097] | n/a |
| sd15_img2img | vae_rt | -0.124 [-0.137, -0.111] | n/a |
