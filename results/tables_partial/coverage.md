**Coverage: cells with rows, images and seeds**

| split | method | purifier | editor | strengths | n_images | n_rows | seeds | n_prompts | face_rate |
|---|---|---|---|---|---|---|---|---|---|
| messy | clean | none | ipadapter_faceid | 0.55 | 20 | 60 | 1,2,3 | 3 | 1.00 |
| messy | clean | none | sd15_img2img | 0.55 | 20 | 60 | 1,2,3 | 3 | 1.00 |
| messy | v1 | none | ipadapter_faceid | 0.55 | 20 | 60 | 1,2,3 | 3 | 1.00 |
| messy | v1 | none | sd15_img2img | 0.55 | 20 | 60 | 1,2,3 | 3 | 1.00 |
| messy | v2_main | none | ipadapter_faceid | 0.55 | 20 | 60 | 1,2,3 | 3 | 1.00 |
| messy | v2_main | none | sd15_img2img | 0.55 | 20 | 60 | 1,2,3 | 3 | 0.98 |
| test | clean | none | ip2p | - | 100 | 300 | 1,2,3 | 3 | 0.82 |
| test | clean | none | ipadapter_faceid | 0.55 | 100 | 300 | 1,2,3 | 3 | 1.00 |
| test | clean | none | sd15_img2img | 0.3,0.55,0.75 | 100 | 1500 | 1,2,3 | 3 | 0.99 |
| test | clean | none | sd_inpaint | 1 | 100 | 300 | 1,2,3 | 3 | 1.00 |
| test | clean | none | sdxl_img2img | 0.55 | 100 | 300 | 1,2,3 | 3 | 1.00 |
| test | v1 | none | ip2p | - | 100 | 300 | 1,2,3 | 3 | 0.92 |
| test | v1 | none | ipadapter_faceid | 0.55 | 100 | 300 | 1,2,3 | 3 | 1.00 |
| test | v1 | none | sd15_img2img | 0.3,0.55,0.75 | 100 | 1500 | 1,2,3 | 3 | 0.99 |
| test | v1 | none | sd_inpaint | 1 | 100 | 300 | 1,2,3 | 3 | 1.00 |
| test | v1 | none | sdxl_img2img | 0.55 | 100 | 300 | 1,2,3 | 3 | 1.00 |
| test | v2_main | none | ip2p | - | 100 | 300 | 1,2,3 | 3 | 0.92 |
| test | v2_main | none | ipadapter_faceid | 0.55 | 100 | 300 | 1,2,3 | 3 | 1.00 |
| test | v2_main | none | sd15_img2img | 0.3,0.55,0.75 | 100 | 1500 | 1,2,3 | 3 | 0.99 |
| test | v2_main | none | sd_inpaint | 1 | 100 | 300 | 1,2,3 | 3 | 1.00 |

Rows per (split, method, purifier, editor) cell. See results/combined/coverage_vs_plan.csv for planned-but-missing cells.
