**E6: v2_antipure vs v2_main (anti-purification terms)**  verdict: **GO (gain on 1 of 3 purifiers)**

Rule: go if the paired PSR gain under higher-t* DiffPure has a 95% CI above 0; 'no gain (within CI)' otherwise (handoff Phase 5: drop it if gains are within the CI).

| purifier | psr_new | psr_ref | diff | diff_lo | diff_hi | p | status |
|---|---|---|---|---|---|---|---|
| diffpure_0.1 | 0.31 | 0.28 | 0.03 | -0.02 | 0.09 | 0.369 | no gain (within CI) |
| diffpure_0.2 | 0.35 | 0.33 | 0.01 | -0.03 | 0.06 | 0.625 | no gain (within CI) |
| diffpure_0.3 | 0.43 | 0.36 | 0.07 | 0.02 | 0.11 | 0.014 | gain |

