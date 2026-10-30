## mabe22 (960 held-out frames, 120 videos; canvas 512, h 10.35 px; background margin 1.5 h = 15.5 px)

chosen arm: **T**; pre-registered PASS vs N2 (pooled): **False**

| metric | N2 pooled bg | N2 per-clip bg | T (mice probe) | B (bench probe) |
|---|---|---|---|---|
| all_kp | 0.888 | 0.534 | 0.326 | 0.888 |
| kp/center | 1.000 | 0.626 | 0.477 | 0.995 |
| kp/ear_l | 0.945 | 0.589 | 0.327 | 0.938 |
| kp/ear_r | 0.952 | 0.600 | 0.312 | 0.922 |
| kp/neck | 0.999 | 0.636 | 0.405 | 0.993 |
| kp/nose | 0.852 | 0.515 | 0.259 | 0.855 |
| kp/paw_fl | 0.980 | 0.618 | 0.365 | 0.983 |
| kp/paw_fr | 0.974 | 0.615 | 0.348 | 0.976 |
| kp/paw_hl | 0.984 | 0.617 | 0.374 | 0.967 |
| kp/paw_hr | 0.983 | 0.600 | 0.357 | 0.966 |
| kp/tail_base | 0.944 | 0.578 | 0.337 | 0.925 |
| kp/tail_mid | 0.707 | 0.312 | 0.269 | 0.728 |
| kp/tail_tip | 0.343 | 0.106 | 0.087 | 0.412 |
| miss/all | 0.000 | 0.245 | 0.263 | 0.000 |
| miss/mouse | 0.000 | 0.245 | 0.263 | 0.000 |
| bg_tok | 36.459 | 0.315 | 0.165 | 12.722 |
| iso_bg_comp | 1.938 | 0.030 | 0.126 | 4.644 |
| tok | 66.850 | 15.539 | 8.367 | 41.167 |

store (union t-2..t+2 at 5 fps, fp16 1024-d, 786,212 frames): T 15.8 tok/frame = 0.025 TB (dilated 1 token 0.092 TB); B 58.7 tok/frame = 0.095 TB (dilated 1 token 0.283 TB); all patches 1.649 TB

