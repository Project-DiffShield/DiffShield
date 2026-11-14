# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 2 file(s), 900 rows
- new: 2 file(s), 900 rows

## VERDICT: INVALID  (headline cell ('sd15_img2img', 'none', '0.55') not present in both methods)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sd15_img2img | diffpure_ddpm256_t100 | 0.55 | 100 | 0.163 | 0.150 | -0.013 [-0.033, +0.003] | KEEP | n/a | n/a | n/a | n/a | - |
| sd15_img2img | diffpure_ddpm256_t200 | 0.55 | 100 | 0.197 | 0.183 | -0.013 [-0.027, -0.003] | KEEP | n/a | n/a | n/a | n/a | - |
| sd15_img2img | downup_256 | 0.55 | 100 | 0.190 | 0.137 | -0.053 [-0.083, -0.027] | KEEP | n/a | n/a | n/a | n/a | - |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: n/a

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| sd15_img2img | diffpure_ddpm256_t100 | 0.55 | 0.205 | 0.204 | -0.001 [-0.002, +0.001] | n/a |
| sd15_img2img | diffpure_ddpm256_t200 | 0.55 | 0.189 | 0.190 | +0.001 [-0.000, +0.002] | n/a |
| sd15_img2img | downup_256 | 0.55 | 0.197 | 0.208 | +0.010 [+0.007, +0.014] | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| sd15_img2img | diffpure_ddpm256_t100 | -0.001 [-0.002, +0.001] | n/a |
| sd15_img2img | diffpure_ddpm256_t200 | +0.001 [-0.000, +0.002] | n/a |
| sd15_img2img | downup_256 | +0.010 [+0.007, +0.014] | n/a |
