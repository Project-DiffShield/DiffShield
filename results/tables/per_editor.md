**PSR by method and editor, no purifier (test split)**

| method | SD1.5 s0.3 | SD1.5 s0.55 | SD1.5 s0.75 | SDXL s0.55 | InstructPix2Pix | SD inpaint (keep-face) | IP-Adapter FaceID |
|---|---|---|---|---|---|---|---|
| clean (E0) | 0.00 [0.00, 0.00] | 0.14 [0.10, 0.18] | 0.84 [0.79, 0.89] | 0.01 [0.00, 0.03] | 0.18 [0.12, 0.25] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| uniform noise (E1) | 0.00 [0.00, 0.00] | 0.15 [0.11, 0.19] | 0.83 [0.79, 0.88] | not run | 0.08 [0.04, 0.12] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| Gaussian noise, PSNR-matched (E1) | 0.00 [0.00, 0.00] | 0.15 [0.11, 0.19] | 0.83 [0.78, 0.88] | not run | 0.09 [0.05, 0.13] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| PhotoGuard encoder attack (E2, reimpl.) | 0.17 [0.12, 0.22] | 0.81 [0.77, 0.85] | 0.96 [0.93, 0.98] | 0.03 [0.01, 0.05] | 0.72 [0.64, 0.79] | 0.00 [0.00, 0.00] | 0.03 [0.01, 0.05] |
| Mist-style (E2, reimpl.) | 0.05 [0.02, 0.08] | 0.63 [0.58, 0.68] | 0.94 [0.91, 0.96] | not run | 0.04 [0.02, 0.07] | 0.00 [0.00, 0.00] | 0.04 [0.01, 0.06] |
| v1 (E3) | 0.00 [0.00, 0.00] | 0.24 [0.19, 0.29] | 0.85 [0.80, 0.90] | 0.02 [0.00, 0.04] | 0.09 [0.05, 0.13] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| v2 cheap purifiers (E4) | 0.00 [0.00, 0.00] | 0.22 [0.18, 0.27] | 0.85 [0.80, 0.89] | not run | 0.15 [0.10, 0.21] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| v2 + one-step SD purifier + L_den (E5) | 0.00 [0.00, 0.00] | 0.24 [0.20, 0.29] | 0.88 [0.84, 0.93] | not run | 0.12 [0.07, 0.17] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| v2 + anti-purification (E6) | not run | not run | not run | not run | not run | not run | 0.00 [0.00, 0.01] |
| v2 final | 0.00 [0.00, 0.00] | 0.29 [0.24, 0.35] | 0.87 [0.83, 0.92] | 0.02 [0.00, 0.04] | 0.03 [0.01, 0.06] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 3. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). FaceID reads identity through ArcFace, not the VAE; low PSR there for VAE-based attacks is expected (V2_PLAN R10). SDXL ran for clean, v1, PhotoGuard and v2_final only.
