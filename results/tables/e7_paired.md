**E7: paired differences between ablation variants**

| comparison (A vs B) | purifier | editor | pairs | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|---|---|
| abl_den_only vs abl_vis_only | none | SD1.5 s0.55 | 100 | 0.12 | 0.25 | -0.13 [-0.19, -0.08] | <0.001 | +0.051 [+0.040, +0.063] | <0.001 | 0.07 |
| abl_den_only vs abl_vis_only | none | InstructPix2Pix | 100 | 0.15 | 0.13 | +0.02 [-0.03, +0.08] | 0.542 | +0.140 [+0.095, +0.184] | <0.001 | 0.17 |
| abl_den_only vs abl_vis_only | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.016 [+0.014, +0.017] | <0.001 | 0.00 |
| abl_den_only vs abl_vis_only | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.065 [+0.055, +0.077] | <0.001 | 0.00 |
| abl_den_only vs abl_vis_only | jpeg_q50 | SD1.5 s0.55 | 100 | 0.12 | 0.19 | -0.07 [-0.11, -0.03] | <0.001 | +0.017 [+0.010, +0.024] | <0.001 | 0.04 |
| abl_den_only vs abl_vis_only | vae_rt | SD1.5 s0.55 | 100 | 0.12 | 0.26 | -0.14 [-0.19, -0.09] | <0.001 | +0.054 [+0.043, +0.067] | <0.001 | 0.04 |
| abl_den_only vs abl_vis_only | diffpure_0.1 | SD1.5 s0.55 | 100 | 0.14 | 0.30 | -0.16 [-0.21, -0.10] | <0.001 | +0.049 [+0.040, +0.059] | <0.001 | 0.07 |
| abl_den_only vs abl_vis_only | diffpure_0.2 | SD1.5 s0.55 | 100 | 0.21 | 0.31 | -0.11 [-0.16, -0.05] | <0.001 | +0.041 [+0.032, +0.051] | <0.001 | 0.11 |
| v2_main vs abl_vis_only | none | SD1.5 s0.55 | 100 | 0.24 | 0.25 | -0.01 [-0.05, +0.03] | 0.806 | +0.006 [-0.002, +0.014] | 0.063 | 0.29 |
| v2_main vs abl_vis_only | none | InstructPix2Pix | 100 | 0.12 | 0.13 | -0.01 [-0.04, +0.02] | 0.361 | +0.015 [-0.006, +0.036] | 0.318 | 0.06 |
| v2_main vs abl_vis_only | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.000 [-0.001, +0.001] | 0.797 | 0.00 |
| v2_main vs abl_vis_only | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | +0.003 [-0.003, +0.009] | 0.934 | 0.00 |
| v2_main vs abl_vis_only | jpeg_q50 | SD1.5 s0.55 | 100 | 0.18 | 0.19 | -0.01 [-0.05, +0.03] | 0.508 | +0.005 [-0.000, +0.010] | 0.029 | 0.11 |
| v2_main vs abl_vis_only | vae_rt | SD1.5 s0.55 | 100 | 0.25 | 0.26 | -0.01 [-0.05, +0.03] | 0.829 | +0.007 [+0.001, +0.013] | 0.021 | 0.14 |
| v2_main vs abl_vis_only | diffpure_0.1 | SD1.5 s0.55 | 100 | 0.28 | 0.30 | -0.02 [-0.07, +0.02] | 0.389 | +0.006 [-0.001, +0.012] | 0.126 | 0.11 |
| v2_main vs abl_vis_only | diffpure_0.2 | SD1.5 s0.55 | 100 | 0.33 | 0.31 | +0.02 [-0.02, +0.06] | 0.581 | +0.005 [-0.000, +0.011] | 0.115 | 0.15 |
| v2_main vs abl_den_only | none | SD1.5 s0.55 | 100 | 0.24 | 0.12 | +0.12 [+0.08, +0.16] | <0.001 | -0.045 [-0.055, -0.034] | <0.001 | 0.56 |
| v2_main vs abl_den_only | none | InstructPix2Pix | 100 | 0.12 | 0.15 | -0.03 [-0.09, +0.02] | 0.274 | -0.121 [-0.168, -0.075] | <0.001 | 0.12 |
| v2_main vs abl_den_only | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.016 [-0.017, -0.014] | <0.001 | 0.00 |
| v2_main vs abl_den_only | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.063 [-0.075, -0.051] | <0.001 | 0.00 |
| v2_main vs abl_den_only | jpeg_q50 | SD1.5 s0.55 | 100 | 0.18 | 0.12 | +0.06 [+0.04, +0.09] | <0.001 | -0.012 [-0.018, -0.005] | <0.001 | 0.21 |
| v2_main vs abl_den_only | vae_rt | SD1.5 s0.55 | 100 | 0.25 | 0.12 | +0.13 [+0.08, +0.18] | <0.001 | -0.048 [-0.060, -0.036] | <0.001 | 0.30 |
| v2_main vs abl_den_only | diffpure_0.1 | SD1.5 s0.55 | 100 | 0.28 | 0.14 | +0.14 [+0.09, +0.19] | <0.001 | -0.042 [-0.051, -0.033] | <0.001 | 0.38 |
| v2_main vs abl_den_only | diffpure_0.2 | SD1.5 s0.55 | 100 | 0.33 | 0.21 | +0.13 [+0.08, +0.17] | <0.001 | -0.036 [-0.046, -0.028] | <0.001 | 0.37 |
| v2_main vs abl_no_sem | none | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs abl_no_sem | none | InstructPix2Pix | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs abl_no_sem | none | SD inpaint (keep-face) | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs abl_no_sem | none | IP-Adapter FaceID | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs abl_no_sem | jpeg_q50 | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs abl_no_sem | vae_rt | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs abl_no_sem | diffpure_0.1 | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs abl_no_sem | diffpure_0.2 | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| abl_no_sem vs abl_no_pur | none | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| abl_no_sem vs abl_no_pur | none | InstructPix2Pix | 0 | not run | not run | not run |  | not run |  |  |
| abl_no_sem vs abl_no_pur | none | SD inpaint (keep-face) | 0 | not run | not run | not run |  | not run |  |  |
| abl_no_sem vs abl_no_pur | none | IP-Adapter FaceID | 0 | not run | not run | not run |  | not run |  |  |
| abl_no_sem vs abl_no_pur | jpeg_q50 | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| abl_no_sem vs abl_no_pur | vae_rt | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| abl_no_sem vs abl_no_pur | diffpure_0.1 | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| abl_no_sem vs abl_no_pur | diffpure_0.2 | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 3. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). Paired differences A-B on the same images (aligned by image_id): mean per-image difference with paired image-bootstrap CI and two-sided Wilcoxon signed-rank p, NOT corrected for the number of comparisons in this table. PSR A-B > 0 means A protects more; sim A-B < 0 means A removes more identity (ArcFace sim, 1 = same person).
