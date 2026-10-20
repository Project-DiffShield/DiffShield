# Coverage of combined results

Rows after de-duplication: 24720 (dropped 0 duplicates, 0 duplicate keys with differing ArcFace sim).

Cells: 72; methods: clean, eps8, noise_gauss_psnr, noise_uniform, v1, v2_final, v2_main; splits: messy, test.

| split | method | purifier | editor | images | seeds | strengths | prompts | face rate | LPIPS filled |
|---|---|---|---|---|---|---|---|---|---|
| messy | clean | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | clean | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | noise_gauss_psnr | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | noise_gauss_psnr | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | noise_uniform | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | noise_uniform | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v1 | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.65 |
| messy | v1 | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v2_final | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v2_final | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v2_main | none | ipadapter_faceid | 20 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| messy | v2_main | none | sd15_img2img | 20 | 1,2,3 | 0.55 | 3 | 0.98 | 0.00 |
| test | clean | none | ip2p | 100 | 1,2,3 | - | 3 | 0.82 | 0.00 |
| test | clean | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | clean | none | sd15_img2img | 100 | 1,2,3 | 0.3,0.55,0.75 | 3 | 0.99 | 0.00 |
| test | clean | none | sd_inpaint | 100 | 1,2,3 | 1 | 3 | 1.00 | 0.00 |
| test | clean | none | sdxl_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | eps8 | diffpure_0.1 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 0.00 |
| test | eps8 | jpeg_q75 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | eps8 | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | eps8 | none | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | noise_gauss_psnr | none | ip2p | 100 | 1,2,3 | - | 3 | 0.92 | 0.00 |
| test | noise_gauss_psnr | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | noise_gauss_psnr | none | sd15_img2img | 100 | 1,2,3 | 0.3,0.55,0.75 | 3 | 0.99 | 0.00 |
| test | noise_gauss_psnr | none | sd_inpaint | 100 | 1,2,3 | 1 | 3 | 1.00 | 0.00 |
| test | noise_uniform | none | ip2p | 100 | 1,2,3 | - | 3 | 0.93 | 0.00 |
| test | noise_uniform | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | noise_uniform | none | sd15_img2img | 100 | 1,2,3 | 0.3,0.55,0.75 | 3 | 0.99 | 0.00 |
| test | noise_uniform | none | sd_inpaint | 100 | 1,2,3 | 1 | 3 | 1.00 | 0.00 |
| test | v1 | diffpure_0.05 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 1.00 |
| test | v1 | diffpure_0.1 | ip2p | 100 | 1,2,3 | - | 3 | 0.93 | 1.00 |
| test | v1 | diffpure_0.1 | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | diffpure_0.1 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | diffpure_0.2 | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | diffpure_0.2 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | diffpure_0.3 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | downup_0.5 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | downup_other | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | jpeg_other | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | jpeg_q50 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | jpeg_q75 | ip2p | 100 | 1,2,3 | - | 3 | 0.90 | 1.00 |
| test | v1 | jpeg_q75 | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | jpeg_q75 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | jpeg_q90 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | none | ip2p | 100 | 1,2,3 | - | 3 | 0.92 | 1.00 |
| test | v1 | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.58 |
| test | v1 | none | sd15_img2img | 100 | 1,2,3 | 0.3,0.55,0.75 | 3 | 0.99 | 0.00 |
| test | v1 | none | sd_inpaint | 100 | 1,2,3 | 1 | 3 | 1.00 | 0.00 |
| test | v1 | none | sdxl_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | v1 | restore_swin2sr | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | vae_rt | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v1 | vae_rt | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | diffpure_0.05 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 1.00 |
| test | v2_main | diffpure_0.1 | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | diffpure_0.1 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 1.00 |
| test | v2_main | diffpure_0.2 | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | diffpure_0.2 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 1.00 |
| test | v2_main | diffpure_0.3 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | downup_0.5 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 1.00 |
| test | v2_main | downup_other | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | jpeg_other | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | jpeg_q50 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 1.00 |
| test | v2_main | jpeg_q75 | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | jpeg_q75 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | jpeg_q90 | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | none | ip2p | 100 | 1,2,3 | - | 3 | 0.92 | 0.00 |
| test | v2_main | none | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 0.00 |
| test | v2_main | none | sd15_img2img | 100 | 1,2,3 | 0.3,0.55,0.75 | 3 | 0.99 | 0.00 |
| test | v2_main | none | sd_inpaint | 100 | 1,2,3 | 1 | 3 | 1.00 | 0.00 |
| test | v2_main | restore_swin2sr | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 1.00 | 1.00 |
| test | v2_main | vae_rt | ipadapter_faceid | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 1.00 |
| test | v2_main | vae_rt | sd15_img2img | 100 | 1,2,3 | 0.55 | 3 | 0.99 | 1.00 |
