# Identity similarity of the exploration variants (continuous, complements PSR)

ArcFace (buffalo_l) similarity between edit and source; lower = less like the person. PSR counts an edit as protected only below 0.1005. Exploration on the 28 tune_search photos, not a test-set claim. Difference to X0: paired by photo, 2000 bootstrap draws, seed 2026, percentile 95% interval.

| variant | name | editor | mean sim | diff to X0 [95% CI] | lowest edit sim | surrogate FaceNet cos first -> last iter |
|---|---|---|---|---|---|---|
| C | clean (unprotected control) | ipadapter_faceid | 0.670 | +0.075 [+0.062, +0.090] (n=28) | 0.340 | - |
| X0 | base (v2_next) | ipadapter_faceid | 0.595 | - | 0.170 | - |
| X1 | eps8 | ipadapter_faceid | 0.576 | -0.019 [-0.026, -0.012] (n=28) | 0.083 | - |
| X2 | it100 | ipadapter_faceid | 0.595 | +0.001 [-0.004, +0.006] (n=28) | 0.174 | - |
| X3 | id_direct_a | ipadapter_faceid | 0.556 | -0.039 [-0.051, -0.027] (n=28) | 0.169 | 1.00 -> -0.28 (n=28) |
| X4 | id_direct_b | ipadapter_faceid | 0.555 | -0.039 [-0.051, -0.028] (n=28) | 0.185 | 1.00 -> -0.42 (n=28) |
| X5 | id_recon | ipadapter_faceid | 0.551 | -0.044 [-0.055, -0.033] (n=28) | 0.168 | 0.99 -> -0.65 (n=28) |
| X6 | eps8_tuned | ipadapter_faceid | 0.576 | -0.019 [-0.031, -0.009] (n=28) | 0.137 | - |
| X7 | eps6_tuned | ipadapter_faceid | 0.584 | -0.010 [-0.021, -0.000] (n=28) | 0.115 | - |
| X8 | eps4_tuned | ipadapter_faceid | 0.594 | -0.001 [-0.010, +0.008] (n=28) | 0.185 | - |
| C | clean (unprotected control) | sd15_img2img | 0.212 | +0.081 [+0.066, +0.097] (n=28) | -0.020 | - |
| X0 | base (v2_next) | sd15_img2img | 0.131 | - | -0.124 | - |
| X1 | eps8 | sd15_img2img | 0.118 | -0.014 [-0.019, -0.008] (n=28) | -0.117 | - |
| X2 | it100 | sd15_img2img | 0.132 | +0.001 [-0.004, +0.006] (n=28) | -0.073 | - |
| X3 | id_direct_a | sd15_img2img | 0.137 | +0.006 [+0.002, +0.010] (n=28) | -0.098 | 1.00 -> -0.28 (n=28) |
| X4 | id_direct_b | sd15_img2img | 0.139 | +0.008 [+0.003, +0.012] (n=28) | -0.080 | 1.00 -> -0.42 (n=28) |
| X5 | id_recon | sd15_img2img | 0.132 | +0.000 [-0.005, +0.005] (n=28) | -0.103 | 0.99 -> -0.65 (n=28) |
| X6 | eps8_tuned | sd15_img2img | 0.122 | -0.009 [-0.015, -0.003] (n=28) | -0.108 | - |
| X7 | eps6_tuned | sd15_img2img | 0.126 | -0.006 [-0.012, +0.000] (n=28) | -0.119 | - |
| X8 | eps4_tuned | sd15_img2img | 0.135 | +0.004 [-0.000, +0.009] (n=28) | -0.082 | - |
