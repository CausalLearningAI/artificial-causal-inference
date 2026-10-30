## mabe22 KG (960 held-out frames)

adopt KG (pre-registered): **True**; window frames falling back to B: 391 / 4800

| metric | T (mice probe) | B (bench probe) | KG (keypoint-guided B) |
|---|---|---|---|
| held_kp | 0.300 | 0.861 | 0.858 |
| held/ear_l | 0.327 | 0.938 | 0.942 |
| held/ear_r | 0.312 | 0.922 | 0.931 |
| held/nose | 0.259 | 0.855 | 0.848 |
| held/paw_fl | 0.365 | 0.983 | 0.983 |
| held/paw_fr | 0.348 | 0.976 | 0.976 |
| held/paw_hl | 0.374 | 0.967 | 0.976 |
| held/paw_hr | 0.357 | 0.966 | 0.975 |
| held/tail_mid | 0.269 | 0.728 | 0.716 |
| held/tail_tip | 0.087 | 0.412 | 0.378 |
| miss_held | 0.300 | 0.000 | 0.000 |
| bg_tok | 0.165 | 12.722 | 0.529 |
| iso_bg_comp | 0.126 | 4.644 | 0.000 |
| tok | 8.367 | 41.167 | 27.909 |

store (union t-2..t+2, 5 fps, fp16 1024-d): T 15.8 tok/frame = 0.025 TB; B 58.7 tok/frame = 0.095 TB; KG 42.1 tok/frame = 0.068 TB

