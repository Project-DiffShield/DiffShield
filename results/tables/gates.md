**Gate evaluation (reported honestly; a missed gate is reported, not changed)**

| gate | what | verdict | rule (operationalisation) | notes |
|---|---|---|---|---|
| E3 | v1 under purification (decision rule for Phase 5) | BARELY DROPS (recommend skipping Phase 5; ask the user) | mean relative PSR drop of v1 over jpeg_q75,jpeg_q50,downup_0.5,vae_rt,diffpure_0.1,diffpure_0.2 (SD1.5 s0.55): >= 50% collapses -> continue with Phase 5; < 20% barely drops -> recommend skipping Phase 5 and ask the user; in between -> ask the user. The cut-offs are analyst proposals (the handoff gives none). | mean relative PSR drop over 6 core purifiers = 0.16 |
| E4 | v2_cheap vs v1 under cheap purifiers | FAIL | PSR(v2_cheap) >= 2 x PSR(v1) under jpeg_q75,jpeg_q50,downup_0.5,vae_rt (SD1.5 s0.55) with PSNR >= 38. Point estimate decides; the paired ratio CI is shown. Degenerate if v1 PSR >= 0.5 (doubling impossible), trivial if < 0.02. | v2_cheap PSNR mean 41.58 dB over 100 images; 0 below 38 |
| E5 | v2_main: held-out DiffPure t*=0.1 vs no purifier | PASS (robust) | PSR under diffpure_0.1 >= 0.5 x PSR without purifier (SD1.5 s0.55, same images). Point estimate decides; 'robust' if the paired ratio CI lower bound also >= the fraction. |  |
| E6 | v2_antipure vs v2_main (anti-purification terms) | GO (gain on 1 of 3 purifiers) | go if the paired PSR gain under higher-t* DiffPure has a 95% CI above 0; 'no gain (within CI)' otherwise (handoff Phase 5: drop it if gains are within the CI). |  |
| INV | Invisibility | FAIL/MARGINAL (see rows) | PSNR >= 38 and SSIM >= 0.95 (per image); human 2AFC accuracy not far above 50%. | human 2AFC result not available (E10 not run) |

Gate values are the handoff's suggestions. E3 cut-offs and purifier sets are analyst choices stated in the rule column; change them with --gate_param. 'DEGENERATE' = the doubling rule is meaningless (V2_PLAN R20): ask the user.
