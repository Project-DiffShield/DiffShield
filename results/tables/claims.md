**Claim numbers: v2_final vs noise controls and vs v1 (SD1.5 s0.55)**

| purifier | v2_final vs | pairs | ArcFace sim reduction, points [95% CI] | p | PSR gain [95% CI] |
|---|---|---|---|---|---|
| none | noise_uniform | 100 | 5.5 [4.9, 6.2] | <0.001 | +0.15 [+0.11, +0.18] |
| none | noise_gauss_psnr | 100 | 5.7 [5.0, 6.4] | <0.001 | +0.15 [+0.12, +0.18] |
| none | v1 | 100 | 1.9 [1.4, 2.5] | <0.001 | +0.05 [+0.02, +0.09] |
| jpeg_other | v1 | 100 | -0.1 [-0.6, 0.4] | 0.910 | +0.01 [-0.02, +0.05] |
| downup_other | v1 | 100 | -0.0 [-0.5, 0.5] | 0.940 | +0.02 [-0.01, +0.04] |
| diffpure_0.1 | v1 | 100 | 1.3 [0.6, 2.0] | <0.001 | +0.03 [-0.02, +0.09] |
| diffpure_0.2 | v1 | 100 | 1.0 [0.3, 1.7] | 0.005 | +0.03 [-0.02, +0.09] |
| diffpure_0.3 | v1 | 100 | 1.4 [0.7, 2.1] | <0.001 | +0.05 [+0.00, +0.11] |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 3. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). ArcFace sim reduction = mean over images of sim(reference edit) - sim(v2_final edit), in points (x100). Sentence template: 'Under the stated threat model, DiffShield lowers ArcFace identity retention of edits by A points versus a noise control and B points over v1, including after [held-out purifiers]; an adaptive attacker reduces PSR to C.' Paired differences A-B on the same images (aligned by image_id): mean per-image difference with paired image-bootstrap CI and two-sided Wilcoxon signed-rank p, NOT corrected for the number of comparisons in this table. PSR A-B > 0 means A protects more; sim A-B < 0 means A removes more identity (ArcFace sim, 1 = same person).
