**Paired differences vs v1 (no purifier, per editor)**

| comparison (A vs B) | purifier | editor | pairs | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|---|---|
| v2_main vs v1 | none | SD1.5 s0.3 | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.009 [-0.016, -0.002] | 0.011 | 0.00 |
| v2_main vs v1 | none | SD1.5 s0.55 | 100 | 0.24 | 0.24 | +0.00 [-0.02, +0.02] | 0.966 | +0.001 [-0.004, +0.005] | 0.749 | 0.26 |
| v2_main vs v1 | none | SD1.5 s0.75 | 100 | 0.88 | 0.85 | +0.03 [-0.01, +0.07] | 0.283 | -0.005 [-0.011, +0.000] | 0.094 | 0.19 |
| v2_main vs v1 | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs v1 | none | InstructPix2Pix | 100 | 0.12 | 0.09 | +0.03 [-0.00, +0.05] | 0.038 | -0.074 [-0.100, -0.048] | <0.001 | 0.12 |
| v2_main vs v1 | none | SD inpaint (keep-face) | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | 1.000 | -0.002 [-0.004, -0.000] | 0.005 | 0.00 |
| v2_main vs v1 | none | IP-Adapter FaceID | 100 | 0.00 | 0.00 | +0.00 [+0.00, +0.01] | 1.000 | -0.025 [-0.033, -0.017] | <0.001 | 0.01 |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 3. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). Paired differences A-B on the same images (aligned by image_id): mean per-image difference with paired image-bootstrap CI and two-sided Wilcoxon signed-rank p, NOT corrected for the number of comparisons in this table. PSR A-B > 0 means A protects more; sim A-B < 0 means A removes more identity (ArcFace sim, 1 = same person).
