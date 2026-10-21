**E3: v1 under purification (decision rule for Phase 5)**  verdict: **BARELY DROPS (recommend skipping Phase 5; ask the user)**

Rule: mean relative PSR drop of v1 over jpeg_q75,jpeg_q50,downup_0.5,vae_rt,diffpure_0.1,diffpure_0.2 (SD1.5 s0.55): >= 50% collapses -> continue with Phase 5; < 20% barely drops -> recommend skipping Phase 5 and ask the user; in between -> ask the user. The cut-offs are analyst proposals (the handoff gives none).

| purifier | psr_none | psr | psr_lo | psr_hi | diff | diff_lo | diff_hi | p | rel_drop | n_pairs | core |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jpeg_q90 | 0.24 | 0.22 | 0.16 | 0.27 | -0.02 | -0.06 | 0.01 | 0.089 | 0.10 | 100 | False |
| jpeg_q75 | 0.24 | 0.15 | 0.10 | 0.20 | -0.09 | -0.13 | -0.05 | <0.001 | 0.37 | 100 | True |
| jpeg_q50 | 0.24 | 0.17 | 0.12 | 0.22 | -0.07 | -0.11 | -0.03 | <0.001 | 0.31 | 100 | True |
| jpeg_other | 0.24 | 0.15 | 0.11 | 0.20 | -0.09 | -0.13 | -0.05 | <0.001 | 0.36 | 100 | False |
| downup_0.5 | 0.24 | 0.10 | 0.06 | 0.14 | -0.14 | -0.18 | -0.11 | <0.001 | 0.60 | 100 | True |
| downup_other | 0.24 | 0.09 | 0.06 | 0.13 | -0.15 | -0.18 | -0.11 | <0.001 | 0.61 | 100 | False |
| vae_rt | 0.24 | 0.22 | 0.17 | 0.28 | -0.02 | -0.05 | 0.02 | 0.108 | 0.08 | 100 | True |
| diffpure_0.05 | 0.24 | 0.24 | 0.18 | 0.30 | -0.00 | -0.04 | 0.04 | 0.709 | 0.01 | 100 | False |
| diffpure_0.1 | 0.24 | 0.26 | 0.21 | 0.32 | 0.02 | -0.02 | 0.06 | 0.514 | -0.10 | 100 | True |
| diffpure_0.2 | 0.24 | 0.32 | 0.25 | 0.38 | 0.08 | 0.03 | 0.12 | 0.006 | -0.32 | 100 | True |
| diffpure_0.3 | 0.24 | 0.36 | 0.30 | 0.42 | 0.12 | 0.08 | 0.16 | <0.001 | -0.50 | 100 | False |
| restore_swin2sr | 0.24 | 0.15 | 0.11 | 0.20 | -0.09 | -0.12 | -0.05 | <0.001 | 0.36 | 100 | False |

- mean relative PSR drop over 6 core purifiers = 0.16
