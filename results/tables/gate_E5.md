**E5: v2_main: held-out DiffPure t*=0.1 vs no purifier**  verdict: **PASS (robust)**

Rule: PSR under diffpure_0.1 >= 0.5 x PSR without purifier (SD1.5 s0.55, same images). Point estimate decides; 'robust' if the paired ratio CI lower bound also >= the fraction.

| method | purifier | psr_none | psr | ratio | ratio_lo | ratio_hi | status | primary |
|---|---|---|---|---|---|---|---|---|
| v2_main | diffpure_0.1 | 0.24 | 0.28 | 1.15 | 1.00 | 1.32 | PASS (robust) | True |
| v2_main | diffpure_0.2 | 0.24 | 0.33 | 1.37 | 1.18 | 1.61 | info | False |
| v2_main | diffpure_0.3 | 0.24 | 0.36 | 1.49 | 1.29 | 1.74 | info | False |
| v2_main | gridpure | None | None | n/a | n/a | n/a | not run | False |
| v2_main | restore_swin2sr | 0.24 | 0.18 | 0.74 | 0.59 | 0.88 | info | False |
| v2_final | diffpure_0.1 | 0.29 | 0.30 | 1.01 | 0.86 | 1.16 | PASS (robust) | False |
| v2_final | diffpure_0.2 | 0.29 | 0.35 | 1.19 | 1.04 | 1.35 | info | False |
| v2_final | diffpure_0.3 | 0.29 | 0.41 | 1.40 | 1.24 | 1.59 | info | False |
| v2_final | gridpure | None | None | n/a | n/a | n/a | not run | False |
| v2_final | restore_swin2sr | 0.29 | 0.16 | 0.55 | 0.44 | 0.67 | info | False |

