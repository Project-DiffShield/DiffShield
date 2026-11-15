# Paired PSR comparison (new minus old)

Headline cell: sd15_img2img x cleanup none x strength 0.55
Bootstrap: {'n_boot': 10000, 'ci': 0.95, 'seed': 2026, 'unit': 'photo, paired'}

- old: 2 file(s), 600 rows
- new: 2 file(s), 600 rows
- clean: 2 file(s), 600 rows

## VERDICT: INVALID  (headline cell ('sd15_img2img', 'none', '0.55') not present in both methods)

ADOPT iff paired-bootstrap lower bound of PSR_new - PSR_old on the headline cell > 0

## Cells (per-cleanup verdicts are informational only; the decision uses the headline cell)

| editor | cleanup | strength | n | PSR old | PSR new | diff new-old [95% CI] | info | clean PSR | net old | net new | net-gain diff | issues |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ip2p | none |  | 100 | 0.020 | 0.017 | -0.003 [-0.010, +0.000] | KEEP | 0.180 | -0.160 | -0.163 | -0.003 [-0.010, +0.000] | - |
| sd_inpaint | none | 1 | 100 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | KEEP | 0.000 | 0.000 | 0.000 | +0.000 [+0.000, +0.000] | - |

## Secondary: mean identity similarity, paired difference new - old (negative = better)

SECONDARY (does not enter the ADOPT/KEEP rule): paired bootstrap of the difference in mean similarity new - old (negative = better protection); per-photo means over seeds/prompts, no-face edits ignored; fr2_sim only where the column has values

Headline cell: n/a

| editor | cleanup | strength | arcface_sim old | arcface_sim new | diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|---|---|---|
| ip2p | none |  | 0.868 | 0.861 | -0.006 [-0.017, +0.004] | n/a |
| sd_inpaint | none | 1 | 0.934 | 0.931 | -0.002 [-0.004, -0.000] | n/a |

Pooled over strengths, per (editor, purifier):

| editor | purifier | arcface_sim diff [95% CI] | fr2_sim diff [95% CI] |
|---|---|---|---|
| ip2p | none | -0.006 [-0.017, +0.004] | n/a |
| sd_inpaint | none | -0.002 [-0.004, -0.000] | n/a |
