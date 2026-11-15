# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 1 file(s), 7200 rows
- new: 2 file(s), 3900 rows
- clean: 1 file(s), 8100 rows

## VERDICT: ADOPT  (lower bound +0.2300 > 0)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ip2p | none |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | diffpure_0.1 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | diffpure_0.2 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | jpeg_q75 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | none | 0.55 | 100 | 0.003 | 0.013 | +0.010 [+0.000, +0.027] | KEEP | 0.000 | 0.003 | 0.013 | +0.010 [+0.000, +0.027] | - |
| ipadapter_faceid | vae_rt | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | diffpure_0.05 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | diffpure_0.1 | 0.55 | 100 | 0.280 | 0.550 | +0.270 [+0.210, +0.333] | ADOPT | 0.133 | 0.147 | 0.417 | +0.270 [+0.210, +0.333] | - |
| sd15_img2img | diffpure_0.2 | 0.55 | 100 | 0.333 | 0.567 | +0.233 [+0.170, +0.300] | ADOPT | 0.217 | 0.117 | 0.350 | +0.233 [+0.170, +0.300] | - |
| sd15_img2img | diffpure_0.3 | 0.55 | 100 | 0.363 | 0.630 | +0.267 [+0.207, +0.327] | ADOPT | 0.260 | 0.103 | 0.370 | +0.267 [+0.207, +0.327] | - |
| sd15_img2img | downup_0.5 | 0.55 | 100 | 0.153 | 0.137 | -0.017 [-0.047, +0.013] | KEEP | 0.097 | 0.057 | 0.040 | -0.017 [-0.047, +0.013] | - |
| sd15_img2img | downup_other | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | jpeg_other | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | jpeg_q50 | 0.55 | 100 | 0.180 | 0.190 | +0.010 [-0.027, +0.050] | KEEP | 0.130 | 0.050 | 0.060 | +0.010 [-0.027, +0.050] | - |
| sd15_img2img | jpeg_q75 | 0.55 | 100 | 0.183 | 0.227 | +0.043 [-0.000, +0.090] | KEEP | 0.103 | 0.080 | 0.123 | +0.043 [-0.000, +0.090] | - |
| sd15_img2img | jpeg_q90 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | none | 0.3 | 100 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | KEEP | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | - |
| sd15_img2img | none (HEADLINE) | 0.55 | 100 | 0.243 | 0.512 | +0.269 [+0.230, +0.309] | ADOPT | 0.139 | 0.104 | 0.373 | +0.269 [+0.230, +0.309] | - |
| sd15_img2img | none | 0.75 | 100 | 0.883 | 0.910 | +0.027 [-0.023, +0.080] | KEEP | 0.840 | 0.043 | 0.070 | +0.027 [-0.023, +0.080] | - |
| sd15_img2img | restore_swin2sr | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| sd15_img2img | vae_rt | 0.55 | 100 | 0.250 | 0.527 | +0.277 [+0.213, +0.340] | ADOPT | 0.123 | 0.127 | 0.403 | +0.277 [+0.213, +0.340] | - |
| sd_inpaint | none | 1 | - | - | - | - | n/a | - | - | - | - | cell missing in new |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: arcface_sim: -0.076 [-0.084, -0.069] (n=100)

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ip2p | none |  | n/a | n/a | n/a | n/a |
| ipadapter_faceid | diffpure_0.1 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | diffpure_0.2 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | jpeg_q75 | 0.55 | n/a | n/a | n/a | n/a |
| ipadapter_faceid | none | 0.55 | 0.635 | 0.606 | -0.029 [-0.041, -0.017] | n/a |
| ipadapter_faceid | vae_rt | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.05 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.1 | 0.55 | 0.164 | 0.095 | -0.069 [-0.081, -0.058] | n/a |
| sd15_img2img | diffpure_0.2 | 0.55 | 0.153 | 0.086 | -0.067 [-0.081, -0.054] | n/a |
| sd15_img2img | diffpure_0.3 | 0.55 | 0.136 | 0.075 | -0.061 [-0.071, -0.051] | n/a |
| sd15_img2img | downup_0.5 | 0.55 | 0.222 | 0.208 | -0.015 [-0.022, -0.007] | n/a |
| sd15_img2img | downup_other | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_other | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_q50 | 0.55 | 0.208 | 0.192 | -0.016 [-0.023, -0.009] | n/a |
| sd15_img2img | jpeg_q75 | 0.55 | 0.201 | 0.176 | -0.025 [-0.032, -0.018] | n/a |
| sd15_img2img | jpeg_q90 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | none | 0.3 | 0.463 | 0.381 | -0.081 [-0.090, -0.074] | n/a |
| sd15_img2img | none | 0.55 | 0.178 | 0.102 | -0.076 [-0.084, -0.069] | n/a |
| sd15_img2img | none | 0.75 | 0.030 | 0.015 | -0.015 [-0.023, -0.006] | n/a |
| sd15_img2img | restore_swin2sr | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | vae_rt | 0.55 | 0.177 | 0.103 | -0.074 [-0.086, -0.062] | n/a |
| sd_inpaint | none | 1 | n/a | n/a | n/a | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ipadapter_faceid | none | -0.029 [-0.041, -0.017] | n/a |
| sd15_img2img | diffpure_0.1 | -0.069 [-0.081, -0.058] | n/a |
| sd15_img2img | diffpure_0.2 | -0.067 [-0.081, -0.054] | n/a |
| sd15_img2img | diffpure_0.3 | -0.061 [-0.071, -0.051] | n/a |
| sd15_img2img | downup_0.5 | -0.015 [-0.022, -0.007] | n/a |
| sd15_img2img | jpeg_q50 | -0.016 [-0.023, -0.009] | n/a |
| sd15_img2img | jpeg_q75 | -0.025 [-0.032, -0.018] | n/a |
| sd15_img2img | none | -0.064 [-0.069, -0.059] | n/a |
| sd15_img2img | vae_rt | -0.074 [-0.086, -0.062] | n/a |
