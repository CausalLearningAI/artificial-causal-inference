## crim13 KG (384 held-out frames; CIRCULAR: built and scored on the same track point)

adopt KG (pre-registered): **n/a (circular)**; window frames falling back to B: 1 / 1920

| metric | T (mice probe) | B (bench probe) | KG (keypoint-guided B) |
|---|---|---|---|
| held_kp | 0.785 | 0.831 | 1.000 |
| kp/int_BC_point | 0.448 | 0.411 | 1.000 |
| kp/int_BC_point_tol1 | 0.812 | 0.734 | 1.000 |
| kp/int_C57_point | 0.927 | 0.979 | 1.000 |
| kp/int_C57_point_tol1 | 0.984 | 1.000 | 1.000 |
| kp/res_point | 0.883 | 0.966 | 1.000 |
| kp/res_point_tol1 | 0.982 | 0.990 | 1.000 |
| miss_held | 0.060 | 0.072 | 0.000 |
| miss/all | 0.060 | 0.072 | 0.000 |
| miss/intruder_BC | 0.188 | 0.266 | 0.000 |
| miss/intruder_C57 | 0.016 | 0.000 | 0.000 |
| miss/resident | 0.018 | 0.010 | 0.000 |
| bg_tok | 2.656 | 7.232 | 0.049 |
| iso_bg_comp | 0.831 | 1.247 | 0.000 |
| tok | 58.880 | 89.336 | 78.190 |

store (union t-2..t+2, 5 fps, fp16 1024-d): T 114.5 tok/frame = 0.170 TB; B 169.6 tok/frame = 0.252 TB; KG 137.4 tok/frame = 0.204 TB

