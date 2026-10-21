**Mean ArcFace similarity of the edit to the source (lower = identity removed), SD1.5 s0.55 (test split)**

| method | none | jpeg_q75 | jpeg_q50 | downup_0.5 | vae_rt | diffpure_0.1 | diffpure_0.2 |
|---|---|---|---|---|---|---|---|
| clean (E0) | 0.22 [0.21, 0.24] | 0.23 [0.21, 0.24] | 0.22 [0.21, 0.24] | 0.24 [0.23, 0.26] | 0.23 [0.21, 0.24] | 0.21 [0.20, 0.23] | 0.19 [0.18, 0.21] |
| uniform noise (E1) | 0.21 [0.20, 0.23] | not run | not run | not run | not run | not run | not run |
| Gaussian noise, PSNR-matched (E1) | 0.21 [0.20, 0.23] | not run | not run | not run | not run | not run | not run |
| PhotoGuard encoder attack (E2, reimpl.) | not run | not run | not run | not run | not run | not run | not run |
| Mist-style (E2, reimpl.) | not run | not run | not run | not run | not run | not run | not run |
| v1 (E3) | 0.18 [0.16, 0.19] | 0.21 [0.19, 0.22] | 0.21 [0.19, 0.22] | 0.23 [0.22, 0.25] | 0.18 [0.16, 0.19] | 0.17 [0.15, 0.18] | 0.15 [0.14, 0.16] |
| v2 cheap purifiers (E4) | 0.18 [0.17, 0.20] | 0.20 [0.19, 0.22] | 0.21 [0.19, 0.23] | 0.23 [0.21, 0.24] | 0.18 [0.17, 0.20] | 0.17 [0.16, 0.19] | 0.16 [0.15, 0.18] |
| v2 + one-step SD purifier + L_den (E5) | 0.18 [0.17, 0.19] | 0.20 [0.18, 0.22] | 0.21 [0.19, 0.22] | 0.22 [0.21, 0.24] | 0.18 [0.16, 0.19] | 0.16 [0.15, 0.18] | 0.15 [0.14, 0.17] |
| v2 + anti-purification (E6) | not run | 0.21 [0.19, 0.23] | 0.21 [0.20, 0.23] | 0.24 [0.22, 0.25] | 0.16 [0.15, 0.18] | 0.15 [0.14, 0.17] | 0.14 [0.13, 0.16] |
| v2 final | 0.16 [0.15, 0.17] | 0.21 [0.19, 0.22] | 0.21 [0.19, 0.23] | 0.23 [0.22, 0.25] | 0.16 [0.15, 0.17] | 0.15 [0.14, 0.17] | 0.14 [0.13, 0.15] |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 3. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). ArcFace cosine similarity: 1 = same face embedding. Mean over images of the per-image mean over seeds and prompts; edits without a detected face are excluded from the mean (they count as successes in PSR).
