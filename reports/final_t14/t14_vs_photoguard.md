# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 1 file(s), 4200 rows
- new: 2 file(s), 3900 rows
- clean: 1 file(s), 8100 rows

## VERDICT: KEEP  (lower bound -0.3422 <= 0)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ip2p | none |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | none | 0.55 | 100 | 0.027 | 0.013 | -0.013 [-0.040, +0.007] | KEEP | 0.000 | 0.027 | 0.013 | -0.013 [-0.040, +0.007] | - |
| sd15_img2img | diffpure_0.1 | 0.55 | 100 | 0.673 | 0.550 | -0.123 [-0.187, -0.060] | KEEP | 0.133 | 0.540 | 0.417 | -0.123 [-0.187, -0.060] | - |
| sd15_img2img | diffpure_0.2 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | diffpure_0.3 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | downup_0.5 | 0.55 | 100 | 0.137 | 0.137 | +0.000 [-0.023, +0.023] | KEEP | 0.097 | 0.040 | 0.040 | +0.000 [-0.023, +0.023] | - |
| sd15_img2img | jpeg_q50 | 0.55 | 100 | 0.167 | 0.190 | +0.023 [-0.007, +0.053] | KEEP | 0.130 | 0.037 | 0.060 | +0.023 [-0.007, +0.053] | - |
| sd15_img2img | jpeg_q75 | 0.55 | 100 | 0.183 | 0.227 | +0.043 [+0.007, +0.080] | ADOPT | 0.103 | 0.080 | 0.123 | +0.043 [+0.007, +0.080] | - |
| sd15_img2img | none | 0.3 | 100 | 0.170 | 0.000 | -0.170 [-0.223, -0.120] | KEEP | 0.000 | 0.170 | 0.000 | -0.170 [-0.223, -0.120] | - |
| sd15_img2img | none (HEADLINE) | 0.55 | 100 | 0.810 | 0.512 | -0.298 [-0.342, -0.252] | KEEP | 0.139 | 0.671 | 0.373 | -0.298 [-0.342, -0.252] | - |
| sd15_img2img | none | 0.75 | 100 | 0.957 | 0.910 | -0.047 [-0.083, -0.010] | KEEP | 0.840 | 0.117 | 0.070 | -0.047 [-0.083, -0.010] | - |
| sd15_img2img | vae_rt | 0.55 | 100 | 0.590 | 0.527 | -0.063 [-0.113, -0.010] | KEEP | 0.123 | 0.467 | 0.403 | -0.063 [-0.113, -0.010] | - |
| sd_inpaint | none | 1 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sdxl_img2img | none | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: arcface_sim: +0.058 [+0.050, +0.066] (n=100)

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ip2p | none |  | n/a | n/a | n/a | n/a |
| ipadapter_faceid | none | 0.55 | 0.502 | 0.606 | +0.105 [+0.086, +0.125] | n/a |
| sd15_img2img | diffpure_0.1 | 0.55 | 0.069 | 0.094 | +0.026 [+0.015, +0.036] | n/a |
| sd15_img2img | diffpure_0.2 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.3 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | downup_0.5 | 0.55 | 0.223 | 0.208 | -0.015 [-0.020, -0.010] | n/a |
| sd15_img2img | jpeg_q50 | 0.55 | 0.202 | 0.192 | -0.010 [-0.015, -0.005] | n/a |
| sd15_img2img | jpeg_q75 | 0.55 | 0.195 | 0.176 | -0.019 [-0.025, -0.013] | n/a |
| sd15_img2img | none | 0.3 | 0.198 | 0.381 | +0.183 [+0.170, +0.196] | n/a |
| sd15_img2img | none | 0.55 | 0.044 | 0.102 | +0.058 [+0.050, +0.066] | n/a |
| sd15_img2img | none | 0.75 | 0.005 | 0.017 | +0.012 [+0.003, +0.021] | n/a |
| sd15_img2img | vae_rt | 0.55 | 0.086 | 0.103 | +0.017 [+0.009, +0.025] | n/a |
| sd_inpaint | none | 1 | n/a | n/a | n/a | n/a |
| sdxl_img2img | none | 0.55 | n/a | n/a | n/a | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ipadapter_faceid | none | +0.105 [+0.086, +0.125] | n/a |
| sd15_img2img | diffpure_0.1 | +0.026 [+0.015, +0.036] | n/a |
| sd15_img2img | downup_0.5 | -0.015 [-0.020, -0.010] | n/a |
| sd15_img2img | jpeg_q50 | -0.010 [-0.015, -0.005] | n/a |
| sd15_img2img | jpeg_q75 | -0.019 [-0.025, -0.013] | n/a |
| sd15_img2img | none | +0.072 [+0.065, +0.078] | n/a |
| sd15_img2img | vae_rt | +0.017 [+0.009, +0.025] | n/a |
