## calms21 KG (456 held-out frames)

adopt KG (pre-registered): **True**; window frames falling back to B: 0 / 2280

| metric | T (mice probe) | B (bench probe) | KG (keypoint-guided B) |
|---|---|---|---|
| held_kp | 0.607 | 0.795 | 0.848 |
| held/int_ear_l | 0.559 | 0.636 | 0.838 |
| held/int_ear_r | 0.518 | 0.675 | 0.868 |
| held/int_nose | 0.410 | 0.513 | 0.445 |
| held/res_ear_l | 0.752 | 0.993 | 0.996 |
| held/res_ear_r | 0.730 | 0.993 | 0.996 |
| held/res_nose | 0.673 | 0.958 | 0.943 |
| miss_held | 0.129 | 0.066 | 0.014 |
| miss_held/intruder_white | 0.193 | 0.132 | 0.029 |
| miss_held/resident | 0.066 | 0.000 | 0.000 |
| bg_tok | 3.026 | 6.539 | 0.276 |
| iso_bg_comp | 0.671 | 2.000 | 0.000 |
| tok | 109.737 | 156.360 | 178.193 |

store (union t-2..t+2, 5 fps, fp16 1024-d): T 223.2 tok/frame = 0.059 TB; B 290.9 tok/frame = 0.076 TB; KG 293.1 tok/frame = 0.077 TB

