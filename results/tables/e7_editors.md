**E7 ablation: PSR by variant and editor, no purifier**

| method | SD1.5 s0.55 | InstructPix2Pix | SD inpaint (keep-face) | IP-Adapter FaceID |
|---|---|---|---|---|
| L_vis only | 0.25 [0.20, 0.31] | 0.13 [0.08, 0.18] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| L_den only | 0.12 [0.08, 0.17] | 0.15 [0.10, 0.21] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |
| both, no L_sem | not run | not run | not run | not run |
| both, no purifier in loop | not run | not run | not run | not run |
| v2 + one-step SD purifier + L_den (E5) | 0.24 [0.20, 0.29] | 0.12 [0.07, 0.17] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.01] |

n images per cell: 100; edit seeds per image: 3; prompts per image: up to 3. Mean with 95% percentile-bootstrap CI over images (10000 resamples, seed 2026); seeds and prompts are averaged within each image first. PSR = share of edits with no detected face or ArcFace sim to the source below the per-editor threshold (95th percentile of impostor sim of the E0 clean edits): ip2p 0.108, ipadapter_faceid 0.108, sd15_img2img 0.101, sd_inpaint 0.110, sdxl_img2img 0.104, pooled 0.103. 'not run' = no rows for the cell. A cell with fewer images than the largest cell shows (n=...).
