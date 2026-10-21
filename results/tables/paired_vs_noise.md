**Paired differences vs the noise controls (no purifier, per editor)**

| comparison (A vs B) | purifier | editor | pairs | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 vs noise_uniform | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.043 [-0.051, -0.035] | <0.001 | 0.00 |
| v1 vs noise_uniform | none | SD1.5 s0.55 | 100 | 0.24 | 0.15 | +0.09 [+0.06, +0.12] | <0.001 | -0.036 [-0.042, -0.030] | <0.001 | 0.51 |
| v1 vs noise_uniform | none | SD1.5 s0.75 | 100 | 0.85 | 0.83 | +0.02 [-0.03, +0.07] | 0.405 | -0.002 [-0.009, +0.006] | 0.559 | 0.19 |
| v1 vs noise_uniform | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_uniform | none | InstructPix2Pix | 100 | 0.09 | 0.08 | +0.01 [-0.02, +0.04] | 0.532 | -0.081 [-0.111, -0.050] | <0.001 | 0.12 |
| v1 vs noise_uniform | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.012 [-0.014, -0.011] | <0.001 | 0.00 |
| v1 vs noise_uniform | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | -0.00 [-0.01, +0.00] | 1.000 | -0.039 [-0.047, -0.031] | <0.001 | 0.00 |
| v1 vs noise_gauss_psnr | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.041 [-0.049, -0.033] | <0.001 | 0.00 |
| v1 vs noise_gauss_psnr | none | SD1.5 s0.55 | 100 | 0.24 | 0.15 | +0.09 [+0.06, +0.12] | <0.001 | -0.037 [-0.043, -0.031] | <0.001 | 0.53 |
| v1 vs noise_gauss_psnr | none | SD1.5 s0.75 | 100 | 0.85 | 0.83 | +0.02 [-0.02, +0.07] | 0.270 | -0.001 [-0.008, +0.006] | 0.810 | 0.18 |
| v1 vs noise_gauss_psnr | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_gauss_psnr | none | InstructPix2Pix | 100 | 0.09 | 0.09 | +0.00 [-0.03, +0.03] | 1.000 | -0.074 [-0.107, -0.042] | <0.001 | 0.11 |
| v1 vs noise_gauss_psnr | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.013 [-0.015, -0.012] | <0.001 | 0.00 |
| v1 vs noise_gauss_psnr | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.039 [-0.048, -0.032] | <0.001 | 0.00 |
| v2_cheap vs noise_uniform | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.038 [-0.046, -0.030] | <0.001 | 0.00 |
| v2_cheap vs noise_uniform | none | SD1.5 s0.55 | 100 | 0.22 | 0.15 | +0.08 [+0.05, +0.10] | <0.001 | -0.028 [-0.034, -0.023] | <0.001 | 0.49 |
| v2_cheap vs noise_uniform | none | SD1.5 s0.75 | 100 | 0.85 | 0.83 | +0.01 [-0.04, +0.06] | 0.838 | -0.003 [-0.012, +0.005] | 0.570 | 0.20 |
| v2_cheap vs noise_uniform | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_cheap vs noise_uniform | none | InstructPix2Pix | 100 | 0.15 | 0.08 | +0.07 [+0.03, +0.12] | 0.001 | -0.168 [-0.208, -0.129] | <0.001 | 0.18 |
| v2_cheap vs noise_uniform | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.013 [-0.014, -0.011] | <0.001 | 0.00 |
| v2_cheap vs noise_uniform | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | -0.00 [-0.01, +0.00] | 1.000 | -0.054 [-0.063, -0.046] | <0.001 | 0.00 |
| v2_cheap vs noise_gauss_psnr | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.036 [-0.045, -0.028] | <0.001 | 0.00 |
| v2_cheap vs noise_gauss_psnr | none | SD1.5 s0.55 | 100 | 0.22 | 0.15 | +0.08 [+0.05, +0.11] | <0.001 | -0.030 [-0.036, -0.024] | <0.001 | 0.45 |
| v2_cheap vs noise_gauss_psnr | none | SD1.5 s0.75 | 100 | 0.85 | 0.83 | +0.02 [-0.03, +0.06] | 0.810 | -0.002 [-0.010, +0.006] | 0.628 | 0.20 |
| v2_cheap vs noise_gauss_psnr | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_cheap vs noise_gauss_psnr | none | InstructPix2Pix | 100 | 0.15 | 0.09 | +0.06 [+0.03, +0.10] | 0.002 | -0.165 [-0.205, -0.125] | <0.001 | 0.18 |
| v2_cheap vs noise_gauss_psnr | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.014 [-0.015, -0.012] | <0.001 | 0.00 |
| v2_cheap vs noise_gauss_psnr | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.055 [-0.064, -0.047] | <0.001 | 0.00 |
| v2_main vs noise_uniform | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.052 [-0.061, -0.043] | <0.001 | 0.00 |
| v2_main vs noise_uniform | none | SD1.5 s0.55 | 100 | 0.24 | 0.15 | +0.09 [+0.07, +0.12] | <0.001 | -0.035 [-0.042, -0.028] | <0.001 | 0.48 |
| v2_main vs noise_uniform | none | SD1.5 s0.75 | 100 | 0.88 | 0.83 | +0.05 [+0.01, +0.09] | 0.075 | -0.007 [-0.015, +0.000] | 0.069 | 0.23 |
| v2_main vs noise_uniform | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_uniform | none | InstructPix2Pix | 100 | 0.12 | 0.08 | +0.04 [+0.00, +0.08] | 0.037 | -0.158 [-0.197, -0.120] | <0.001 | 0.15 |
| v2_main vs noise_uniform | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.014 [-0.016, -0.013] | <0.001 | 0.00 |
| v2_main vs noise_uniform | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.063 [-0.075, -0.052] | <0.001 | 0.00 |
| v2_main vs noise_gauss_psnr | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.050 [-0.060, -0.041] | <0.001 | 0.00 |
| v2_main vs noise_gauss_psnr | none | SD1.5 s0.55 | 100 | 0.24 | 0.15 | +0.10 [+0.07, +0.13] | <0.001 | -0.036 [-0.043, -0.029] | <0.001 | 0.49 |
| v2_main vs noise_gauss_psnr | none | SD1.5 s0.75 | 100 | 0.88 | 0.83 | +0.05 [+0.01, +0.10] | 0.025 | -0.006 [-0.013, +0.001] | 0.088 | 0.25 |
| v2_main vs noise_gauss_psnr | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_gauss_psnr | none | InstructPix2Pix | 100 | 0.12 | 0.09 | +0.03 [-0.01, +0.07] | 0.088 | -0.154 [-0.194, -0.113] | <0.001 | 0.15 |
| v2_main vs noise_gauss_psnr | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.015 [-0.017, -0.014] | <0.001 | 0.00 |
| v2_main vs noise_gauss_psnr | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.01] | 1.000 | -0.064 [-0.076, -0.053] | <0.001 | 0.01 |
| v2_antipure vs noise_uniform | none | SD1.5 s0.3 | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_uniform | none | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_uniform | none | SD1.5 s0.75 | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_uniform | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_uniform | none | InstructPix2Pix | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_uniform | none | SD inpaint (keep-face) | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_uniform | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.055 [-0.065, -0.045] | <0.001 | 0.00 |
| v2_antipure vs noise_gauss_psnr | none | SD1.5 s0.3 | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_gauss_psnr | none | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_gauss_psnr | none | SD1.5 s0.75 | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_gauss_psnr | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_gauss_psnr | none | InstructPix2Pix | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_gauss_psnr | none | SD inpaint (keep-face) | 0 | not run | not run | not run |  | not run |  |  |
| v2_antipure vs noise_gauss_psnr | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.01] | 1.000 | -0.056 [-0.066, -0.046] | <0.001 | 0.01 |
| v2_final vs noise_uniform | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.060 [-0.068, -0.052] | <0.001 | 0.00 |
| v2_final vs noise_uniform | none | SD1.5 s0.55 | 100 | 0.29 | 0.15 | +0.15 [+0.11, +0.18] | <0.001 | -0.055 [-0.062, -0.049] | <0.001 | 0.69 |
| v2_final vs noise_uniform | none | SD1.5 s0.75 | 100 | 0.87 | 0.83 | +0.04 [-0.01, +0.09] | 0.102 | -0.015 [-0.024, -0.007] | 0.003 | 0.23 |
| v2_final vs noise_uniform | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_final vs noise_uniform | none | InstructPix2Pix | 100 | 0.03 | 0.08 | -0.04 [-0.08, -0.01] | 0.009 | -0.019 [-0.046, +0.010] | 0.026 | 0.03 |
| v2_final vs noise_uniform | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.019 [-0.021, -0.017] | <0.001 | 0.00 |
| v2_final vs noise_uniform | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.046 [-0.054, -0.039] | <0.001 | 0.00 |
| v2_final vs noise_gauss_psnr | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.058 [-0.066, -0.049] | <0.001 | 0.00 |
| v2_final vs noise_gauss_psnr | none | SD1.5 s0.55 | 100 | 0.29 | 0.15 | +0.15 [+0.12, +0.18] | <0.001 | -0.057 [-0.064, -0.050] | <0.001 | 0.68 |
| v2_final vs noise_gauss_psnr | none | SD1.5 s0.75 | 100 | 0.87 | 0.83 | +0.04 [-0.01, +0.09] | 0.061 | -0.015 [-0.024, -0.007] | 0.006 | 0.23 |
| v2_final vs noise_gauss_psnr | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_final vs noise_gauss_psnr | none | InstructPix2Pix | 100 | 0.03 | 0.09 | -0.05 [-0.09, -0.02] | 0.004 | -0.013 [-0.042, +0.018] | 0.082 | 0.03 |
| v2_final vs noise_gauss_psnr | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.020 [-0.022, -0.018] | <0.001 | 0.00 |
| v2_final vs noise_gauss_psnr | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.01] | 1.000 | -0.047 [-0.055, -0.040] | <0.001 | 0.01 |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 3. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). Paired differences A-B on the same images (aligned by image_id): mean per-image difference with paired image-bootstrap CI and two-sided Wilcoxon signed-rank p, NOT corrected for the number of comparisons in this table. PSR A-B > 0 means A protects more; sim A-B < 0 means A removes more identity (ArcFace sim, 1 = same person).
