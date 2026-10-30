## calms21 (456 held-out frames, 19 videos; canvas 1024, h 40.58 px; background margin 1.5 h = 60.9 px)

chosen arm: **T**; pre-registered PASS vs N2 (pooled): **True**

| metric | N2 pooled bg | N2 per-clip bg | T (mice probe) | B (bench probe) |
|---|---|---|---|---|
| all_kp | 0.641 | 0.619 | 0.693 | 0.825 |
| kp/int_ear_l | 0.248 | 0.208 | 0.559 | 0.636 |
| kp/int_ear_r | 0.232 | 0.200 | 0.518 | 0.675 |
| kp/int_hip_l | 0.408 | 0.346 | 0.487 | 0.713 |
| kp/int_hip_r | 0.375 | 0.340 | 0.434 | 0.728 |
| kp/int_neck | 0.325 | 0.261 | 0.717 | 0.651 |
| kp/int_nose | 0.228 | 0.193 | 0.410 | 0.513 |
| kp/int_tail_base | 0.386 | 0.360 | 0.680 | 0.737 |
| kp/res_ear_l | 0.950 | 0.936 | 0.752 | 0.993 |
| kp/res_ear_r | 0.943 | 0.932 | 0.730 | 0.993 |
| kp/res_hip_l | 0.987 | 0.982 | 0.943 | 0.989 |
| kp/res_hip_r | 0.993 | 0.991 | 0.906 | 0.978 |
| kp/res_neck | 0.987 | 0.987 | 0.936 | 0.998 |
| kp/res_nose | 0.956 | 0.965 | 0.673 | 0.958 |
| kp/res_tail_base | 0.958 | 0.961 | 0.954 | 0.989 |
| miss/all | 0.117 | 0.150 | 0.011 | 0.010 |
| miss/intruder_white | 0.235 | 0.300 | 0.022 | 0.020 |
| miss/resident | 0.000 | 0.000 | 0.000 | 0.000 |
| bg_tok | 75.757 | 50.487 | 3.026 | 6.539 |
| iso_bg_comp | 16.173 | 14.189 | 0.671 | 2.000 |
| tok | 269.265 | 229.498 | 109.737 | 156.360 |

store (union t-2..t+2 at 5 fps, fp16 1024-d, 128,343 frames): T 223.2 tok/frame = 0.059 TB (dilated 1 token 0.111 TB); B 290.9 tok/frame = 0.076 TB (dilated 1 token 0.137 TB); all patches 1.077 TB

