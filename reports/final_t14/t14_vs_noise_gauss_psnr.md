# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 1 file(s), 2400 rows
- new: 2 file(s), 3900 rows
- clean: 1 file(s), 8100 rows

## VERDICT: ADOPT  (lower bound +0.3200 > 0)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ip2p | none |  | - | - | - | - | n/a | - | - | - | - | cell missing in new |
| ipadapter_faceid | none | 0.55 | 100 | 0.000 | 0.013 | +0.013 [+0.000, +0.030] | KEEP | 0.000 | 0.000 | 0.013 | +0.013 [+0.000, +0.030] | - |
| sd15_img2img | diffpure_0.1 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | diffpure_0.2 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | diffpure_0.3 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | downup_0.5 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | jpeg_q50 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | jpeg_q75 | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd15_img2img | none | 0.3 | 100 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | KEEP | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | - |
| sd15_img2img | none (HEADLINE) | 0.55 | 100 | 0.147 | 0.512 | +0.366 [+0.320, +0.412] | ADOPT | 0.139 | 0.008 | 0.373 | +0.366 [+0.320, +0.412] | - |
| sd15_img2img | none | 0.75 | 100 | 0.830 | 0.910 | +0.080 [+0.030, +0.133] | ADOPT | 0.840 | -0.010 | 0.070 | +0.080 [+0.030, +0.133] | - |
| sd15_img2img | vae_rt | 0.55 | - | - | - | - | n/a | - | - | - | - | cell missing in old |
| sd_inpaint | none | 1 | - | - | - | - | n/a | - | - | - | - | cell missing in new |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: arcface_sim: -0.112 [-0.122, -0.103] (n=100)

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ip2p | none |  | n/a | n/a | n/a | n/a |
| ipadapter_faceid | none | 0.55 | 0.699 | 0.606 | -0.093 [-0.108, -0.079] | n/a |
| sd15_img2img | diffpure_0.1 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.2 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | diffpure_0.3 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | downup_0.5 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_q50 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | jpeg_q75 | 0.55 | n/a | n/a | n/a | n/a |
| sd15_img2img | none | 0.3 | 0.513 | 0.381 | -0.132 [-0.142, -0.122] | n/a |
| sd15_img2img | none | 0.55 | 0.214 | 0.102 | -0.112 [-0.122, -0.103] | n/a |
| sd15_img2img | none | 0.75 | 0.037 | 0.015 | -0.021 [-0.031, -0.012] | n/a |
| sd15_img2img | vae_rt | 0.55 | n/a | n/a | n/a | n/a |
| sd_inpaint | none | 1 | n/a | n/a | n/a | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ipadapter_faceid | none | -0.093 [-0.108, -0.079] | n/a |
| sd15_img2img | none | -0.097 [-0.104, -0.090] | n/a |
