**Paired differences vs the noise controls (no purifier, per editor)**

| comparison (A vs B) | purifier | editor | pairs | PSR A | PSR B | PSR A-B [95% CI] | p (PSR) | sim A-B [95% CI] | p (sim) | images A>B |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 vs noise_uniform | none | SD1.5 s0.3 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_uniform | none | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_uniform | none | SD1.5 s0.75 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_uniform | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_uniform | none | InstructPix2Pix | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_uniform | none | SD inpaint (keep-face) | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_uniform | none | IP-Adapter FaceID | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_gauss_psnr | none | SD1.5 s0.3 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_gauss_psnr | none | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_gauss_psnr | none | SD1.5 s0.75 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_gauss_psnr | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_gauss_psnr | none | InstructPix2Pix | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_gauss_psnr | none | SD inpaint (keep-face) | 0 | not run | not run | not run |  | not run |  |  |
| v1 vs noise_gauss_psnr | none | IP-Adapter FaceID | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_uniform | none | SD1.5 s0.3 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_uniform | none | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_uniform | none | SD1.5 s0.75 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_uniform | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_uniform | none | InstructPix2Pix | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_uniform | none | SD inpaint (keep-face) | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_uniform | none | IP-Adapter FaceID | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_gauss_psnr | none | SD1.5 s0.3 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_gauss_psnr | none | SD1.5 s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_gauss_psnr | none | SD1.5 s0.75 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_gauss_psnr | none | SDXL s0.55 | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_gauss_psnr | none | InstructPix2Pix | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_gauss_psnr | none | SD inpaint (keep-face) | 0 | not run | not run | not run |  | not run |  |  |
| v2_main vs noise_gauss_psnr | none | IP-Adapter FaceID | 0 | not run | not run | not run |  | not run |  |  |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 3. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). Paired differences A-B on the same images (aligned by image_id): mean per-image difference with paired image-bootstrap CI and two-sided Wilcoxon signed-rank p, NOT corrected for the number of comparisons in this table. PSR A-B > 0 means A protects more; sim A-B < 0 means A removes more identity (ArcFace sim, 1 = same person).
