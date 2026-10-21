**INV: Invisibility**  verdict: **FAIL/MARGINAL (see rows)**

Rule: PSNR >= 38 and SSIM >= 0.95 (per image); human 2AFC accuracy not far above 50%.

| method | n | psnr | ssim | n_below_psnr | n_below_ssim | status |
|---|---|---|---|---|---|---|
| abl_den_only | 100 | 50.76 | 1.00 | 0 | 0 | PASS |
| abl_vis_only | 100 | 41.73 | 0.98 | 0 | 0 | PASS |
| eps2 | 100 | 45.34 | 0.98 | 0 | 0 | PASS |
| eps8 | 100 | 35.98 | 0.90 | 98 | 96 | FAIL |
| noise_gauss_psnr | 100 | 41.91 | 0.96 | 0 | 3 | MARGINAL |
| noise_uniform | 100 | 40.80 | 0.95 | 0 | 43 | MARGINAL |
| v1 | 100 | 42.42 | 0.98 | 0 | 0 | PASS |
| v2_antipure | 100 | 41.18 | 0.97 | 0 | 13 | MARGINAL |
| v2_cheap | 100 | 41.58 | 0.98 | 0 | 0 | PASS |
| v2_final | 100 | 40.46 | 0.95 | 0 | 29 | MARGINAL |
| v2_main | 100 | 41.72 | 0.98 | 0 | 0 | PASS |

- human 2AFC result not available (E10 not run)
