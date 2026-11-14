# Exploration comparison (X0..X5)

EXPLORATION on the 28 tune_search photos. NOT a test-set claim; adopting any variant needs a fresh 100-photo test run (docs/EXPLORATION.md).

PSR = share of edits whose ArcFace similarity to the source is below 0.1005 (or no face found); higher is better. Seeds/prompts are averaged within a photo; a cell is editor x cleanup.

Difference to X0: paired by photo, all-cell mean per photo, photos resampled with replacement (2000 draws, seed 2026), percentile 95% interval.

**X1 (eps8) changes the threat / visibility trade-off (a larger perturbation budget is visibly stronger noise); it cannot be compared with the other variants at equal invisibility.** Compare it by its own PSNR/SSIM columns, not as a free gain.

## Summary

| variant | name | photos | PSR all-cell | PSR sd15 | PSR ipadapter | diff to X0 [95% CI] | mean PSNR | mean SSIM | min PSNR | min SSIM | photos violating PSNR>=38 and SSIM>=0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C | clean (unprotected control) | 28 | 0.045 | 0.090 | 0.000 | - | n/a | n/a | n/a | n/a | n/a |
| X0 | base (v2_next) | 28 | 0.186 | 0.371 | 0.000 | - | 40.86 | 0.9714 | 39.62 | 0.9560 | 0 of 28 |
| X1 | eps8 | 28 | 0.225 | 0.448 | 0.002 | +0.039 [+0.020, +0.057] (n=28) | 38.11 | 0.9648 | 35.23 | 0.9468 | 12 of 28 |
| X2 | it100 | 28 | 0.190 | 0.381 | 0.000 | +0.005 [-0.013, +0.020] (n=28) | 40.69 | 0.9718 | 39.86 | 0.9577 | 0 of 28 |
| X3 | id_direct_a | 28 | 0.168 | 0.336 | 0.000 | -0.018 [-0.032, -0.004] (n=28) | 40.14 | 0.9683 | 39.20 | 0.9534 | 0 of 28 |
| X4 | id_direct_b | 28 | 0.156 | 0.312 | 0.000 | -0.030 [-0.045, -0.015] (n=28) | 39.66 | 0.9646 | 38.89 | 0.9493 | 1 of 28 |
| X5 | id_recon | 28 | 0.181 | 0.362 | 0.000 | -0.005 [-0.027, +0.015] (n=28) | 40.48 | 0.9659 | 39.51 | 0.9513 | 0 of 28 |
| X6 | eps8_tuned | 28 | 0.213 | 0.424 | 0.002 | +0.027 [+0.002, +0.050] (n=28) | 38.12 | 0.9636 | 35.19 | 0.9505 | 12 of 28 |
| X7 | eps6_tuned | 28 | 0.213 | 0.426 | 0.000 | +0.027 [+0.001, +0.051] (n=28) | 39.00 | 0.9655 | 36.85 | 0.9536 | 1 of 28 |
| X8 | eps4_tuned | 28 | 0.189 | 0.379 | 0.000 | +0.004 [-0.018, +0.023] (n=28) | 40.88 | 0.9703 | 39.63 | 0.9595 | 0 of 28 |

## PSR per variant x editor x cleanup

| variant | ipadapter_faceid / diffpure_0.2 | ipadapter_faceid / downup_0.5 | ipadapter_faceid / jpeg_q75 | ipadapter_faceid / none | ipadapter_faceid / vae_rt | sd15_img2img / diffpure_0.2 | sd15_img2img / downup_0.5 | sd15_img2img / jpeg_q75 | sd15_img2img / none | sd15_img2img / vae_rt |
|---|---|---|---|---|---|---|---|---|---|---|
| C | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.119 | 0.095 | 0.048 | 0.071 | 0.119 |
| X0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.548 | 0.167 | 0.214 | 0.417 | 0.512 |
| X1 | 0.012 | 0.000 | 0.000 | 0.000 | 0.000 | 0.619 | 0.238 | 0.250 | 0.560 | 0.571 |
| X2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.512 | 0.179 | 0.262 | 0.464 | 0.488 |
| X3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.488 | 0.190 | 0.190 | 0.405 | 0.405 |
| X4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.440 | 0.202 | 0.179 | 0.381 | 0.357 |
| X5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.524 | 0.167 | 0.202 | 0.488 | 0.429 |
| X6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | 0.667 | 0.131 | 0.262 | 0.500 | 0.560 |
| X7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.655 | 0.155 | 0.250 | 0.500 | 0.571 |
| X8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.583 | 0.131 | 0.214 | 0.452 | 0.512 |
