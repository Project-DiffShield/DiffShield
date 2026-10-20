**E4: v2_cheap vs v1 under cheap purifiers**  verdict: **NOT RUN**

Rule: PSR(v2_cheap) >= 2 x PSR(v1) under jpeg_q75,jpeg_q50,downup_0.5,vae_rt (SD1.5 s0.55) with PSNR >= 38. Point estimate decides; the paired ratio CI is shown. Degenerate if v1 PSR >= 0.5 (doubling impossible), trivial if < 0.02.

| purifier | primary | psr_new | psr_v1 | ratio | ratio_lo | ratio_hi | diff | diff_lo | diff_hi | p | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jpeg_q75 | True | None | None | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not run |
| jpeg_q50 | True | None | None | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not run |
| downup_0.5 | True | None | None | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not run |
| vae_rt | True | None | None | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not run |
| jpeg_other | False | None | None | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not run |
| downup_other | False | None | None | n/a | n/a | n/a | n/a | n/a | n/a | n/a | not run |

- invisibility of v2_cheap not available (no inv_psnr)
