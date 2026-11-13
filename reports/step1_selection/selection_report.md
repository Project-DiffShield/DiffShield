# Tuning photo selection report

Numbers only. MMD^2 is to the pool P (lower = more representative). p = (1 + #{random <= ours}) / (n_random + 1).
CLIP space is the optimised space (an extreme p is expected); attribute+pose space is the independent check.

Pool: {"n_npz": 30000, "n_no_face": 2, "n_excluded_names_in_npz": 244, "n_excluded_names_missing_from_npz": 0, "n_identity_leak_removed": 1840, "n_pool": 27914}
Bandwidths: CLIP 0.98145, attribute+pose 8.8229
Random subsets: 1000 (MMD), 200 (coverage). Seed 2026.

## MMD^2 vs random same-size subsets

| set | n | space | ours | rand mean | rand p05 | rand p50 | rand p95 | p |
|---|---|---|---|---|---|---|---|---|
| tune_search | 28 | clip | 0.0045483 | 0.01387 | 0.011024 | 0.013667 | 0.017577 | 0.000999 |
| tune_search | 28 | attr_pose | 0.0090289 | 0.014581 | 0.0098623 | 0.013827 | 0.021516 | 0.014985 |
| tune_confirm_prototypes | 8 | clip | 0.020422 | 0.048403 | 0.038291 | 0.04756 | 0.061276 | 0.000999 |
| tune_confirm_prototypes | 8 | attr_pose | 0.02625 | 0.050512 | 0.033478 | 0.048398 | 0.074506 | 0.000999 |
| tune_confirm | 12 | clip | 0.019947 | 0.03213 | 0.02572 | 0.031512 | 0.040682 | 0.000999 |
| tune_confirm | 12 | attr_pose | 0.022005 | 0.033868 | 0.022885 | 0.03234 | 0.050367 | 0.031968 |
| old_val | 20 | clip | 0.023166 | 0.019509 | 0.015473 | 0.019079 | 0.024758 | 0.89311 |
| old_val | 20 | attr_pose | 0.02584 | 0.020167 | 0.013857 | 0.019186 | 0.03017 | 0.88012 |

## Coverage: CLIP cosine distance from each pool image to the nearest selected photo

| set | stat | ours | rand mean | rand p05 | rand p50 | rand p95 | p |
|---|---|---|---|---|---|---|---|
| tune_search | mean_dist | 0.31044 | 0.32157 | 0.31441 | 0.32191 | 0.32869 | 0.0099502 |
| tune_search | max_dist | 0.62273 | 0.62994 | 0.60118 | 0.62814 | 0.65937 | 0.32836 |
| tune_confirm_prototypes | mean_dist | 0.33417 | 0.36681 | 0.3508 | 0.3657 | 0.38338 | 0.0049751 |
| tune_confirm_prototypes | max_dist | 0.64048 | 0.677 | 0.64358 | 0.67855 | 0.7079 | 0.034826 |
| tune_confirm | mean_dist | 0.33135 | 0.3518 | 0.34081 | 0.35124 | 0.36589 | 0.0049751 |
| tune_confirm | max_dist | 0.60949 | 0.66079 | 0.63016 | 0.66171 | 0.68914 | 0.0049751 |
| old_val | mean_dist | 0.33578 | 0.33327 | 0.32495 | 0.33268 | 0.342 | 0.68159 |
| old_val | max_dist | 0.65907 | 0.64651 | 0.61792 | 0.64621 | 0.6765 | 0.76119 |

## Attribute shares (pool vs set)

| attribute | pool | tune_search | tune_confirm_prototypes | tune_confirm | old_val |
|---|---|---|---|---|---|
| 5_o_Clock_Shadow | 0.151 | 0.107 | 0.125 | 0.167 | 0.15 |
| Arched_Eyebrows | 0.361 | 0.286 | 0.25 | 0.417 | 0.45 |
| Attractive | 0.568 | 0.536 | 0.625 | 0.583 | 0.6 |
| Bags_Under_Eyes | 0.29 | 0.357 | 0.25 | 0.333 | 0.3 |
| Bald | 0.024 | 0 | 0 | 0 | 0 |
| Bangs | 0.185 | 0.214 | 0.25 | 0.25 | 0.05 |
| Big_Lips | 0.368 | 0.214 | 0.375 | 0.417 | 0.2 |
| Big_Nose | 0.327 | 0.286 | 0.25 | 0.25 | 0.25 |
| Black_Hair | 0.223 | 0.214 | 0.125 | 0.167 | 0.2 |
| Blond_Hair | 0.17 | 0.107 | 0.25 | 0.167 | 0.05 |
| Blurry | 0.0038 | 0 | 0 | 0 | 0 |
| Brown_Hair | 0.228 | 0.179 | 0.25 | 0.167 | 0.1 |
| Bushy_Eyebrows | 0.194 | 0.25 | 0 | 0.167 | 0.1 |
| Chubby | 0.072 | 0.0714 | 0.125 | 0.167 | 0.1 |
| Double_Chin | 0.06 | 0.0714 | 0 | 0 | 0.1 |
| Eyeglasses | 0.0511 | 0.0714 | 0.125 | 0.0833 | 0 |
| Goatee | 0.0774 | 0.0714 | 0 | 0 | 0.1 |
| Gray_Hair | 0.041 | 0.0357 | 0 | 0 | 0 |
| Heavy_Makeup | 0.449 | 0.393 | 0.5 | 0.417 | 0.55 |
| High_Cheekbones | 0.462 | 0.393 | 0.5 | 0.417 | 0.55 |
| Male | 0.375 | 0.357 | 0.375 | 0.5 | 0.4 |
| Mouth_Slightly_Open | 0.471 | 0.5 | 0.5 | 0.5 | 0.55 |
| Mustache | 0.0585 | 0.107 | 0 | 0 | 0.05 |
| Narrow_Eyes | 0.12 | 0.107 | 0.125 | 0.0833 | 0 |
| No_Beard | 0.809 | 0.893 | 0.875 | 0.917 | 0.75 |
| Oval_Face | 0.21 | 0.107 | 0.25 | 0.333 | 0.15 |
| Pale_Skin | 0.052 | 0.107 | 0 | 0 | 0 |
| Pointy_Nose | 0.318 | 0.25 | 0.25 | 0.333 | 0.1 |
| Receding_Hairline | 0.0838 | 0.107 | 0 | 0 | 0 |
| Rosy_Cheeks | 0.111 | 0.0714 | 0 | 0 | 0.05 |
| Sideburns | 0.0824 | 0.107 | 0 | 0 | 0.05 |
| Smiling | 0.469 | 0.429 | 0.375 | 0.333 | 0.55 |
| Straight_Hair | 0.216 | 0.143 | 0.375 | 0.333 | 0 |
| Wavy_Hair | 0.358 | 0.429 | 0.375 | 0.333 | 0.3 |
| Wearing_Earrings | 0.262 | 0.286 | 0.125 | 0.0833 | 0.2 |
| Wearing_Hat | 0.0362 | 0 | 0 | 0.0833 | 0 |
| Wearing_Lipstick | 0.555 | 0.571 | 0.5 | 0.417 | 0.55 |
| Wearing_Necklace | 0.169 | 0.25 | 0.125 | 0.0833 | 0 |
| Wearing_Necktie | 0.073 | 0.0714 | 0 | 0 | 0 |
| Young | 0.774 | 0.786 | 0.625 | 0.667 | 0.8 |

## Pose (mean / std; raw annotation units, unverified)

| axis | pool | tune_search | tune_confirm_prototypes | tune_confirm | old_val |
|---|---|---|---|---|---|
| yaw | 2.03 / 15.7 | 0.512 / 13.8 | 6.91 / 13.6 | 6.01 / 12.3 | 4.7 / 12.7 |
| pitch | -7.03 / 8.18 | -6.55 / 6.22 | -9.98 / 7.97 | -8.94 / 7.39 | -9.43 / 7.16 |
| roll | -0.12 / 2.58 | -0.0112 / 2.23 | -0.592 / 1.49 | -0.413 / 1.29 | 0.0133 / 2.01 |

Max pairwise ArcFace cosine among all selected photos: 0.19475
Confirm criticisms: 14935.jpg, 10259.jpg, 9445.jpg, 24986.jpg

