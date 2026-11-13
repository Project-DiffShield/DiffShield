# Tuning photo selection report

Numbers only. MMD^2 is to the pool P (lower = more representative). p = (1 + #{random <= ours}) / (n_random + 1).
CLIP space is the optimised space (an extreme p is expected); attribute+pose space is the independent check.

Pool: {"n_npz": 30000, "n_no_face": 2, "n_excluded_names_in_npz": 120, "n_excluded_names_missing_from_npz": 0, "n_identity_leak_removed": 904, "n_pool": 28974}
Bandwidths: CLIP 0.97891, attribute+pose 8.7658
Random subsets: 1000 (MMD), 200 (coverage). Seed 2026.

## MMD^2 vs random same-size subsets

| set | n | space | ours | rand mean | rand p05 | rand p50 | rand p95 | p |
|---|---|---|---|---|---|---|---|---|
| tune_search | 28 | clip | 0.0044238 | 0.013868 | 0.010834 | 0.013616 | 0.017651 | 0.000999 |
| tune_search | 28 | attr_pose | 0.0082324 | 0.014531 | 0.0097884 | 0.01406 | 0.020965 | 0.007992 |
| tune_confirm_prototypes | 8 | clip | 0.020462 | 0.048761 | 0.038315 | 0.04786 | 0.062344 | 0.000999 |
| tune_confirm_prototypes | 8 | attr_pose | 0.026582 | 0.051113 | 0.034728 | 0.048627 | 0.077683 | 0.000999 |
| tune_confirm | 12 | clip | 0.022328 | 0.032638 | 0.026129 | 0.032136 | 0.040894 | 0.002997 |
| tune_confirm | 12 | attr_pose | 0.02762 | 0.034748 | 0.023002 | 0.033442 | 0.050558 | 0.1988 |
| old_val | 20 | clip | 0.023039 | 0.019353 | 0.015426 | 0.018943 | 0.024916 | 0.8981 |
| old_val | 20 | attr_pose | 0.025935 | 0.020785 | 0.014036 | 0.019596 | 0.031846 | 0.85215 |

## Coverage: CLIP cosine distance from each pool image to the nearest selected photo

| set | stat | ours | rand mean | rand p05 | rand p50 | rand p95 | p |
|---|---|---|---|---|---|---|---|
| tune_search | mean_dist | 0.30644 | 0.32002 | 0.31371 | 0.31962 | 0.3277 | 0.0049751 |
| tune_search | max_dist | 0.62273 | 0.63488 | 0.61048 | 0.63515 | 0.6617 | 0.26368 |
| tune_confirm_prototypes | mean_dist | 0.33334 | 0.3672 | 0.35173 | 0.36678 | 0.38459 | 0.0049751 |
| tune_confirm_prototypes | max_dist | 0.64048 | 0.67599 | 0.6477 | 0.67525 | 0.70506 | 0.014925 |
| tune_confirm | mean_dist | 0.32993 | 0.35117 | 0.33967 | 0.35059 | 0.36322 | 0.0049751 |
| tune_confirm | max_dist | 0.60949 | 0.66132 | 0.63297 | 0.66076 | 0.69009 | 0.0049751 |
| old_val | mean_dist | 0.33472 | 0.33248 | 0.32355 | 0.33182 | 0.34371 | 0.66169 |
| old_val | max_dist | 0.65907 | 0.64364 | 0.61758 | 0.64376 | 0.67101 | 0.81592 |

## Attribute shares (pool vs set)

| attribute | pool | tune_search | tune_confirm_prototypes | tune_confirm | old_val |
|---|---|---|---|---|---|
| 5_o_Clock_Shadow | 0.15 | 0.107 | 0.125 | 0.0833 | 0.15 |
| Arched_Eyebrows | 0.365 | 0.357 | 0.25 | 0.25 | 0.45 |
| Attractive | 0.572 | 0.607 | 0.625 | 0.583 | 0.6 |
| Bags_Under_Eyes | 0.288 | 0.321 | 0.25 | 0.333 | 0.3 |
| Bald | 0.0237 | 0 | 0 | 0 | 0 |
| Bangs | 0.184 | 0.25 | 0.25 | 0.333 | 0.05 |
| Big_Lips | 0.367 | 0.321 | 0.375 | 0.417 | 0.2 |
| Big_Nose | 0.326 | 0.321 | 0.25 | 0.25 | 0.25 |
| Black_Hair | 0.221 | 0.25 | 0.125 | 0.167 | 0.2 |
| Blond_Hair | 0.17 | 0.107 | 0.25 | 0.167 | 0.05 |
| Blurry | 0.00369 | 0 | 0 | 0 | 0 |
| Brown_Hair | 0.231 | 0.286 | 0.25 | 0.25 | 0.1 |
| Bushy_Eyebrows | 0.191 | 0.286 | 0 | 0.0833 | 0.1 |
| Chubby | 0.0713 | 0.0357 | 0.125 | 0.167 | 0.1 |
| Double_Chin | 0.0599 | 0.0357 | 0 | 0 | 0.1 |
| Eyeglasses | 0.0499 | 0.0714 | 0.125 | 0.0833 | 0 |
| Goatee | 0.077 | 0.0714 | 0 | 0 | 0.1 |
| Gray_Hair | 0.0409 | 0.0714 | 0 | 0.0833 | 0 |
| Heavy_Makeup | 0.455 | 0.429 | 0.5 | 0.333 | 0.55 |
| High_Cheekbones | 0.462 | 0.393 | 0.5 | 0.417 | 0.55 |
| Male | 0.371 | 0.357 | 0.375 | 0.5 | 0.4 |
| Mouth_Slightly_Open | 0.472 | 0.5 | 0.5 | 0.417 | 0.55 |
| Mustache | 0.058 | 0.0714 | 0 | 0 | 0.05 |
| Narrow_Eyes | 0.119 | 0.143 | 0.125 | 0.0833 | 0 |
| No_Beard | 0.81 | 0.857 | 0.875 | 0.917 | 0.75 |
| Oval_Face | 0.208 | 0.179 | 0.25 | 0.333 | 0.15 |
| Pale_Skin | 0.0518 | 0.0357 | 0 | 0.167 | 0 |
| Pointy_Nose | 0.318 | 0.214 | 0.25 | 0.167 | 0.1 |
| Receding_Hairline | 0.0841 | 0.0714 | 0 | 0.0833 | 0 |
| Rosy_Cheeks | 0.113 | 0.107 | 0 | 0 | 0.05 |
| Sideburns | 0.0818 | 0.107 | 0 | 0 | 0.05 |
| Smiling | 0.47 | 0.393 | 0.375 | 0.333 | 0.55 |
| Straight_Hair | 0.216 | 0.107 | 0.375 | 0.333 | 0 |
| Wavy_Hair | 0.358 | 0.429 | 0.375 | 0.25 | 0.3 |
| Wearing_Earrings | 0.264 | 0.179 | 0.125 | 0.0833 | 0.2 |
| Wearing_Hat | 0.0357 | 0 | 0 | 0 | 0 |
| Wearing_Lipstick | 0.56 | 0.607 | 0.5 | 0.333 | 0.55 |
| Wearing_Necklace | 0.17 | 0.214 | 0.125 | 0.0833 | 0 |
| Wearing_Necktie | 0.0724 | 0.0714 | 0 | 0 | 0 |
| Young | 0.776 | 0.786 | 0.625 | 0.583 | 0.8 |

## Pose (mean / std; raw annotation units, unverified)

| axis | pool | tune_search | tune_confirm_prototypes | tune_confirm | old_val |
|---|---|---|---|---|---|
| yaw | 2.02 / 15.6 | 1.64 / 13.6 | 6.91 / 13.6 | 6.94 / 11.8 | 4.7 / 12.7 |
| pitch | -7.04 / 8.16 | -8.18 / 6.27 | -9.98 / 7.97 | -8.09 / 7.61 | -9.43 / 7.16 |
| roll | -0.117 / 2.58 | 0.0133 / 1.98 | -0.592 / 1.49 | -0.302 / 1.75 | 0.0133 / 2.01 |

Max pairwise ArcFace cosine among all selected photos: 0.45138
Confirm criticisms: 18185.jpg, 14889.jpg, 8471.jpg, 9445.jpg

