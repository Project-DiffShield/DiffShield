**Invisibility of protected images per method (PSNR, SSIM, LPIPS)**

| method | split | n images | eps (x/255) | PSNR dB [95% CI] | min PSNR | below 38 dB | SSIM [95% CI] | below 0.95 | LPIPS(VGG) [95% CI] | source |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 | messy | 20 | 4 | 42.03 [41.85, 42.21] | 40.96 | 0 | 0.9703 [0.9678, 0.9727] | 0 | 0.271 [0.243, 0.299] | protect.csv |
| v1 | test | 100 | 4 | 42.42 [42.33, 42.51] | 41.17 | 0 | 0.9763 [0.9753, 0.9772] | 0 | 0.239 [0.226, 0.252] | protect.csv |
| v1 | v1_eval70 | 70 | 4 | 42.17 [42.07, 42.27] | 41.12 | 0 | 0.9756 [0.9743, 0.9769] | 0 | 0.231 [0.215, 0.245] | protect.csv |
| v2_antipure | val | 20 | 4 | 41.39 [41.02, 41.78] | 39.61 | 0 | 0.9735 [0.9694, 0.9770] | 1 | 0.259 [0.233, 0.286] | protect.csv |
| v2_final | messy | 2 | 4 | 39.30 [38.42, 40.19] | 38.42 | 0 | 0.9294 [0.9023, 0.9565] | 1 | 0.386 [0.303, 0.470] | protect.csv |
| v2_final | test | 100 | 4 | 40.46 [40.35, 40.57] | 38.76 | 0 | 0.9549 [0.9522, 0.9574] | 29 | 0.289 [0.275, 0.304] | protect.csv |
| v2_main | messy | 20 | 4 | 41.63 [41.38, 41.88] | 40.24 | 0 | 0.9749 [0.9729, 0.9770] | 0 | 0.274 [0.247, 0.301] | protect.csv |
| v2_main | test | 100 | 4 | 41.72 [41.60, 41.85] | 40.34 | 0 | 0.9784 [0.9777, 0.9791] | 0 | 0.239 [0.229, 0.250] | protect.csv |

Protected vs clean image, published uint8 PNG convention. Requirement: PSNR >= 38 dB and SSIM >= 0.95 on every image. n images and CI as in the cell; 'source' says whether the numbers come from the protection run (protect.csv) or the edit rows.
