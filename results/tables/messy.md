**PSR on the 20 messy photos (pose / lighting / pre-compression), no purifier**

| method | SD1.5 s0.55 | IP-Adapter FaceID |
|---|---|---|
| clean (E0) | 0.15 [0.07, 0.25] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| uniform noise (E1) | 0.17 [0.08, 0.25] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| Gaussian noise, PSNR-matched (E1) | 0.17 [0.07, 0.27] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| PhotoGuard encoder attack (E2, reimpl.) | 0.73 [0.60, 0.85] (n=20) | 0.03 [0.00, 0.08] (n=20) |
| Mist-style (E2, reimpl.) | 0.62 [0.47, 0.75] (n=20) | 0.05 [0.00, 0.10] (n=20) |
| v1 (E3) | 0.32 [0.18, 0.45] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| v2 + one-step SD purifier + L_den (E5) | 0.28 [0.15, 0.43] (n=20) | 0.00 [0.00, 0.00] (n=20) |
| v2 final | 0.43 [0.28, 0.58] (n=20) | 0.02 [0.00, 0.05] (n=20) |

n images per cell: 20; edit seeds per image: 3; prompts per image: up to 1. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...). Messy photos: one rotated prompt per image, 20 images only, so intervals are wide.
