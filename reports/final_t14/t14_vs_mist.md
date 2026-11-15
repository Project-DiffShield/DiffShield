# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 1 file(s), 3900 rows
- new: 2 file(s), 3900 rows
- clean: 1 file(s), 8100 rows

## VERDICT: KEEP  (lower bound -0.1533 <= 0)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ip2p | none |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | none | 0.55 | 100 | 0.037 | 0.013 | -0.023 [-0.047, -0.003] | KEEP | 0.000 | 0.037 | 0.013 | -0.023 [-0.047, -0.003] | - |
| sd15_img2img | diffpure_0.1 | 0.55 | 100 | 0.640 | 0.550 | -0.090 [-0.140, -0.040] | KEEP | 0.133 | 0.507 | 0.417 | -0.090 [-0.140, -0.040] | - |
| sd15_img2img | diffpure_0.2 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | diffpure_0.3 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | downup_0.5 | 0.55 | 100 | 0.127 | 0.137 | +0.010 [-0.013, +0.033] | KEEP | 0.097 | 0.030 | 0.040 | +0.010 [-0.013, +0.033] | - |
| sd15_img2img | jpeg_q50 | 0.55 | 100 | 0.157 | 0.190 | +0.033 [+0.000, +0.067] | ADOPT | 0.130 | 0.027 | 0.060 | +0.033 [+0.000, +0.067] | - |
| sd15_img2img | jpeg_q75 | 0.55 | 100 | 0.210 | 0.227 | +0.017 [-0.020, +0.050] | KEEP | 0.103 | 0.107 | 0.123 | +0.017 [-0.020, +0.050] | - |
| sd15_img2img | none | 0.3 | 100 | 0.047 | 0.000 | -0.047 [-0.077, -0.020] | KEEP | 0.000 | 0.047 | 0.000 | -0.047 [-0.077, -0.020] | - |
| sd15_img2img | none (HEADLINE) | 0.55 | 100 | 0.631 | 0.512 | -0.119 [-0.153, -0.084] | KEEP | 0.139 | 0.492 | 0.373 | -0.119 [-0.153, -0.084] | - |
| sd15_img2img | none | 0.75 | 100 | 0.940 | 0.910 | -0.030 [-0.067, +0.007] | KEEP | 0.840 | 0.100 | 0.070 | -0.030 [-0.067, +0.007] | - |
| sd15_img2img | vae_rt | 0.55 | 100 | 0.590 | 0.527 | -0.063 [-0.120, -0.007] | KEEP | 0.123 | 0.467 | 0.403 | -0.063 [-0.120, -0.007] | - |
| sd_inpaint | none | 1 | - | - | - | - | n/a | - | - | - | - | cell missing in new |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: arcface_sim: +0.019 [+0.014, +0.025] (n=100)

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ip2p | none |  | n/a | n/a | n/a | n/a |
| ipadapter_faceid | none | 0.55 | 0.510 | 0.606 | +0.097 [+0.078, +0.116] | n/a |
| sd15_img2img | diffpure_0.1 | 0.55 | 0.080 | 0.094 | +0.014 [+0.006, +0.023] | n/a |
| sd15_img2img | diffpure_0.2 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.3 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | downup_0.5 | 0.55 | 0.222 | 0.208 | -0.014 [-0.019, -0.010] | n/a |
| sd15_img2img | jpeg_q50 | 0.55 | 0.205 | 0.192 | -0.013 [-0.018, -0.008] | n/a |
| sd15_img2img | jpeg_q75 | 0.55 | 0.191 | 0.176 | -0.015 [-0.021, -0.009] | n/a |
| sd15_img2img | none | 0.3 | 0.287 | 0.381 | +0.094 [+0.082, +0.106] | n/a |
| sd15_img2img | none | 0.55 | 0.083 | 0.102 | +0.019 [+0.014, +0.025] | n/a |
| sd15_img2img | none | 0.75 | 0.009 | 0.015 | +0.005 [-0.002, +0.013] | n/a |
| sd15_img2img | vae_rt | 0.55 | 0.093 | 0.103 | +0.010 [+0.001, +0.019] | n/a |
| sd_inpaint | none | 1 | n/a | n/a | n/a | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ipadapter_faceid | none | +0.097 [+0.078, +0.116] | n/a |
| sd15_img2img | diffpure_0.1 | +0.014 [+0.006, +0.023] | n/a |
| sd15_img2img | downup_0.5 | -0.014 [-0.019, -0.010] | n/a |
| sd15_img2img | jpeg_q50 | -0.013 [-0.018, -0.008] | n/a |
| sd15_img2img | jpeg_q75 | -0.015 [-0.021, -0.009] | n/a |
| sd15_img2img | none | +0.030 [+0.026, +0.035] | n/a |
| sd15_img2img | vae_rt | +0.010 [+0.001, +0.019] | n/a |
