# Coverage of combined results

Rows after de-duplication: 8160 (dropped 0 duplicates, 0 duplicate keys with differing ArcFace sim).

Cells: 20; methods: clean, v1, v2_main; splits: messy, test.

| split | method | purifier | editor | images | seeds | strengths | prompts | face rate | LPIPS filled |
|---|---|---|---|---|---|---|---|---|---|
| messy | clean | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | clean | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v1 | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v1 | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v2_main | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v2_main | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 0.98 | 0.00 |
| test | clean | none | ip2p | 100 | 1,2,3 | - | 3 | 0.82 | 0.00 |
| test | clean | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | clean | none | sd15_img2img | 100 | 1,2,3 | 0.3,0.55,0.75 | 3 | 0.99 | 0.00 |
| test | clean | none | sd_inpaint | 100 | 1,2,3 | 1 | 3 | 1.00 | 0.00 |
| test | clean | none | sdxl_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | v1 | none | ip2p | 100 | 1,2,3 | - | 3 | 0.92 | 0.00 |
| test | v1 | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | v1 | none | sd15_img2img | 100 | 1,2,3 | 0.3,0.55,0.75 | 3 | 0.99 | 0.00 |
| test | v1 | none | sd_inpaint | 100 | 1,2,3 | 1 | 3 | 1.00 | 0.00 |
| test | v1 | none | sdxl_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | v2_main | none | ip2p | 100 | 1,2,3 | - | 3 | 0.92 | 0.00 |
| test | v2_main | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | v2_main | none | sd15_img2img | 100 | 1,2,3 | 0.3,0.55,0.75 | 3 | 0.99 | 0.00 |
| test | v2_main | none | sd_inpaint | 100 | 1,2,3 | 1 | 3 | 1.00 | 0.00 |
