# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 2 file(s), 300 rows
- new: 2 file(s), 300 rows
- clean: 2 file(s), 300 rows

## VERDICT: INVALID  (headline cell ('sd15_img2img', 'none', '0.55') not present in both methods)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sdxl_img2img | none | 0.55 | 100 | 0.020 | 0.023 | +0.003 [-0.007, +0.013] | KEEP | 0.013 | 0.007 | 0.010 | +0.003 [-0.007, +0.013] | - |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: n/a

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| sdxl_img2img | none | 0.55 | 0.320 | 0.321 | +0.001 [-0.002, +0.004] | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| sdxl_img2img | none | +0.001 [-0.002, +0.004] | n/a |
