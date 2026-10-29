# mask3 / mask3b vs frozen N2 (pre-registered in scripts/eci/v2/mask3_val.py; mask3b = post-hoc amendment 1)

## Lab (120 random frames per series, rng 0; ghost area over ALL videos)

| series | tok/frame N2 | mask3 | mask3b | added / removed mask3 | added / removed mask3b | leak A N2 / m3 / m3b | leak B m3 / m3b | ghost area median / p90 / n>2% (N2 bg) | after fix |
|---|---|---|---|---|---|---|---|---|---|
| mice_v1 | 243.4 | 287.4 | 252.4 | 0.190 / 0.009 | 0.046 / 0.009 | 0.000 / 0.079 / 0.000 | 0.000 / 0.000 | 0.0082 / 0.0245 / 86 of 432 | 0.0000 max |
| mice_v2 | 241.1 | 283.9 | 250.2 | 0.182 / 0.005 | 0.043 / 0.005 | 0.000 / 0.079 / 0.000 | 0.000 / 0.000 | 0.0122 / 0.0243 / 60 of 216 | 0.0000 max |
| ants_v2 | 38.4 | 38.7 | 38.4 | 0.007 / 0.000 | 0.000 / 0.000 | 0.000 / 0.002 / 0.000 | 0.000 / 0.000 | 0.0000 / 0.0001 / 0 of 44 | 0.0000 max |
| ants_v3 | 47.2 | 48.7 | 47.2 | 0.034 / 0.002 | 0.002 / 0.001 | 0.000 / 0.010 / 0.000 | 0.000 / 0.000 | 0.0010 / 0.0036 / 0 of 107 | 0.0000 max |
| ants_v4 | 42.4 | 43.4 | 42.5 | 0.026 / 0.002 | 0.001 / 0.000 | 0.000 / 0.004 / 0.000 | 0.000 / 0.000 | 0.0012 / 0.0045 / 0 of 81 | 0.0000 max |
| frogs_juv | 27.4 | 27.4 | 27.4 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 | 0.0000 / 0.0006 / 0 of 35 | 0.0000 max |
| frogs_tad | 23.4 | 23.4 | 23.4 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 | 0.0000 / 0.0002 / 0 of 118 | 0.0000 max |

## Benchmarks (validate.py frames; N2 reproduces validate/*_final.json exactly: max abs diff 0)

| set | metric | N2 | mask3 | mask3b |
|---|---|---|---|---|
| mab | tail_seg_tok | 0.311 | 0.440 | 0.343 |
| mab | tail_tip_tok | 0.095 | 0.167 | 0.109 |
| mab | tail_base_tok | 0.589 | 0.675 | 0.622 |
| mab | body_tok | 0.591 | 0.630 | 0.599 |
| mab | leak | 0.028 | 0.028 | 0.026 |
| mab | tokens_per_frame | 41.323 | 49.005 | 43.076 |
| mab | ghost area mean (validation packs) | 0.0025 | 0.0000 | 0.0000 |
| cal | pale_missed | 0.300 | 0.099 | 0.283 |
| cal | pale_tok | 0.273 | 0.577 | 0.266 |
| cal | tail_base_tok_int | 0.360 | 0.724 | 0.386 |
| cal | tail_base_tok_res | 0.961 | 0.996 | 0.993 |
| cal | body_tok | 0.966 | 0.989 | 0.984 |
| cal | leak | 0.221 | 0.543 | 0.227 |
| cal | tokens_per_frame | 229.498 | 784.787 | 240.759 |
| cal | ghost area mean (validation packs) | 0.0028 | 0.0000 | 0.0000 |
| crim | pale_tok | 0.089 | 0.219 | 0.094 |
| crim | pale_tok_tol1 | 0.365 | 0.661 | 0.432 |
| crim | res_tok | 0.969 | 0.979 | 0.971 |
| crim | c57_tok | 0.995 | 0.995 | 0.995 |
| crim | leak | 0.178 | 0.295 | 0.181 |
| crim | tokens_per_frame | 141.766 | 254.914 | 151.182 |
| crim | ghost area mean (validation packs) | 0.0089 | 0.0000 | 0.0000 |

## Mice tail eye check (Claude, 15 metric frames per series, tail counted if >= half its visible length is in kept tokens)

| series | visible tails | N2 | mask3 | mask3b | off-animal |
|---|---|---|---|---|---|
| v1 | 34 | 25 | 27 | 27 | isolated tokens in 3 frames (all arms); mask3: wall blocks next to a corner huddle in 2 frames + bedding halo in 3; mask3b: none seen |
| v2 | 30 | 24 | 25 | 24 | isolated token in 1 frame (all arms); dark-bedding blob in 1 frame (all arms, larger in mask3 / mask3b); mask3 bedding halo in 3 frames of a ghost video; mask3b: none seen |

Hard case wt_ash1l_m_1_S_H k6001 (pale tail over an old huddle spot): N2 no, mask3 yes (plus a bedding blob at the tip), mask3b no (tail change vs the fixed background ~42 grey levels, min with the median -> only 47% of tail pixels > T_LO 35).
