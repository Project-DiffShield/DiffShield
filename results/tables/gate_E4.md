**E4: v2_cheap vs v1 under cheap purifiers**  verdict: **FAIL**

Rule: PSR(v2_cheap) >= 2 x PSR(v1) under jpeg_q75,jpeg_q50,downup_0.5,vae_rt (SD1.5 s0.55) with PSNR >= 38. Point estimate decides; the paired ratio CI is shown. Degenerate if v1 PSR >= 0.5 (doubling impossible), trivial if < 0.02.

| purifier | primary | psr_new | psr_v1 | ratio | ratio_lo | ratio_hi | diff | diff_lo | diff_hi | p | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jpeg_q75 | True | 0.17 | 0.15 | 1.11 | 0.89 | 1.38 | 0.02 | -0.02 | 0.05 | 0.404 | FAIL |
| jpeg_q50 | True | 0.17 | 0.17 | 1.04 | 0.85 | 1.29 | 0.01 | -0.03 | 0.04 | 0.665 | FAIL |
| downup_0.5 | True | 0.14 | 0.10 | 1.48 | 1.13 | 1.96 | 0.05 | 0.01 | 0.08 | 0.009 | FAIL |
| vae_rt | True | 0.21 | 0.22 | 0.97 | 0.81 | 1.16 | -0.01 | -0.05 | 0.03 | 0.914 | FAIL |
| jpeg_other | False | 0.17 | 0.15 | 1.11 | 0.92 | 1.37 | 0.02 | -0.01 | 0.05 | 0.346 | FAIL |
| downup_other | False | 0.11 | 0.09 | 1.21 | 0.89 | 1.67 | 0.02 | -0.01 | 0.05 | 0.216 | FAIL |

- v2_cheap PSNR mean 41.58 dB over 100 images; 0 below 38
