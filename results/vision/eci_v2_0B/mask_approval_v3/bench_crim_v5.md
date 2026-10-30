## crim13 (384 held-out frames, 24 videos; canvas 640, h 27.02 px; background margin 1.5 h = 40.5 px + 6 h disk)

chosen arm: **T**; pre-registered PASS vs N2 (pooled): **True**

| metric | N2 pooled bg | N2 per-clip bg | T (mice probe) | B (bench probe) |
|---|---|---|---|---|
| all_kp | 0.758 | 0.755 | 0.785 | 0.831 |
| kp/int_BC_point | 0.094 | 0.089 | 0.448 | 0.411 |
| kp/int_BC_point_tol1 | 0.385 | 0.365 | 0.812 | 0.734 |
| kp/int_C57_point | 0.995 | 0.995 | 0.927 | 0.979 |
| kp/int_C57_point_tol1 | 1.000 | 0.995 | 0.984 | 1.000 |
| kp/res_point | 0.971 | 0.969 | 0.883 | 0.966 |
| kp/res_point_tol1 | 0.984 | 0.987 | 0.982 | 0.990 |
| miss/all | 0.161 | 0.167 | 0.060 | 0.072 |
| miss/intruder_BC | 0.615 | 0.635 | 0.188 | 0.266 |
| miss/intruder_C57 | 0.000 | 0.005 | 0.016 | 0.000 |
| miss/resident | 0.016 | 0.013 | 0.018 | 0.010 |
| bg_tok | 59.362 | 25.232 | 2.656 | 7.232 |
| iso_bg_comp | 5.570 | 3.349 | 0.831 | 1.247 |
| tok | 203.068 | 141.760 | 58.880 | 89.336 |

store (union t-2..t+2 at 5 fps, fp16 1024-d, 725,094 frames): T 114.5 tok/frame = 0.170 TB (dilated 1 token 0.388 TB); B 169.6 tok/frame = 0.252 TB (dilated 1 token 0.511 TB); all patches 2.376 TB

