**E5: v2_main: held-out DiffPure t*=0.1 vs no purifier**  verdict: **NOT RUN**

Rule: PSR under diffpure_0.1 >= 0.5 x PSR without purifier (SD1.5 s0.55, same images). Point estimate decides; 'robust' if the paired ratio CI lower bound also >= the fraction.

| method | purifier | psr_none | psr | ratio | ratio_lo | ratio_hi | status | primary |
|---|---|---|---|---|---|---|---|---|
| v2_main | diffpure_0.1 | None | None | n/a | n/a | n/a | not run | True |
| v2_main | diffpure_0.2 | None | None | n/a | n/a | n/a | not run | False |
| v2_main | diffpure_0.3 | None | None | n/a | n/a | n/a | not run | False |
| v2_main | gridpure | None | None | n/a | n/a | n/a | not run | False |
| v2_main | restore_swin2sr | None | None | n/a | n/a | n/a | not run | False |
| v2_final | diffpure_0.1 | None | None | n/a | n/a | n/a | not run | False |
| v2_final | diffpure_0.2 | None | None | n/a | n/a | n/a | not run | False |
| v2_final | diffpure_0.3 | None | None | n/a | n/a | n/a | not run | False |
| v2_final | gridpure | None | None | n/a | n/a | n/a | not run | False |
| v2_final | restore_swin2sr | None | None | n/a | n/a | n/a | not run | False |

