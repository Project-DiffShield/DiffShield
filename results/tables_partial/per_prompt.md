**PSR by prompt, SD1.5 s0.55, no purifier (headline cell has all 3 prompts per image)**

| method | p0 | p1 | p2 |
|---|---|---|---|
| clean (E0) | 0.08 [0.05, 0.12] | 0.16 [0.11, 0.22] | 0.17 [0.12, 0.23] |
| uniform noise (E1) | not run | not run | not run |
| Gaussian noise, PSNR-matched (E1) | not run | not run | not run |
| PhotoGuard encoder attack (E2, reimpl.) | not run | not run | not run |
| Mist-style (E2, reimpl.) | not run | not run | not run |
| v1 (E3) | 0.21 [0.15, 0.26] | 0.26 [0.20, 0.32] | 0.25 [0.20, 0.31] |
| v2 cheap purifiers (E4) | not run | not run | not run |
| v2 + one-step SD purifier + L_den (E5) | 0.19 [0.14, 0.25] | 0.24 [0.19, 0.30] | 0.30 [0.24, 0.35] |
| v2 + anti-purification (E6) | not run | not run | not run |
| v2 final | not run | not run | not run |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 1. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). p0 'a person in a suit', p1 'a person at a beach', p2 'an oil painting of a person'.
