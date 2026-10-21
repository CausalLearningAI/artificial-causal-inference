# ECI v2 discoveries (number of selected neurons; Bonferroni, primary window; per prefix)

NES = Neural Effect Search (src/eci/nes.py runners); NESS = src/eci/dines.py (atlas_v37_methods.py). Input sets: `<sae>` = 0.6 s smoothed + fnorm firing threshold (pre-specified); `<sae>_smooth` = 0.6 s smoothed, no threshold (sensitivity; keeps the behaviour alignment, results/vision/eci_v2/*/align.md).

## mice|v2_mice_c0_w2048_s7

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | A_het_1to2: 128:2 / 512:2 / 2048:2; A_het_2to3: 128:2 / 512:2 / 2048:2; A_het_4to5: 128:2 / 512:2 / 2048:2; A_het_5to6: 128:2 / 512:2 / 2048:2; A_wt_1to2: 128:2 / 512:2 / 2048:2; A_wt_2to3: 128:2 / 512:2 / 2048:2; A_wt_4to5: 128:2 / 512:2 / 2048:2; A_wt_5to6: 128:2 / 512:2 / 2048:2; B_stage1: 128:0 / 512:0 / 2048:0; B_stage2: 128:0 / 512:0 / 2048:0; B_stage3: 128:0 / 512:0 / 2048:0; B_stage4: 128:0 / 512:0 / 2048:0; B_stage5: 128:0 / 512:0 / 2048:0; B_stage6: 128:1 / 512:0 / 2048:0 |
| NESS_mean | A_het_1to2: 128:3 / 256:1 / 2048:30; A_het_2to3: 128:0 / 256:2 / 2048:0; A_het_4to5: 128:0 / 256:0 / 2048:0; A_het_5to6: 128:0 / 256:0 / 2048:0; A_wt_1to2: 128:5 / 256:0 / 2048:30; A_wt_2to3: 128:0 / 256:0 / 2048:0; A_wt_4to5: 128:0 / 256:0 / 2048:0; A_wt_5to6: 128:0 / 256:3 / 2048:2; B_stage1: 128:0 / 256:1 / 2048:0; B_stage2: 128:0 / 256:0 / 2048:0; B_stage3: 128:1 / 256:0 / 2048:0; B_stage4: 128:0 / 256:0 / 2048:0; B_stage5: 128:0 / 256:0 / 2048:0; B_stage6: 128:1 / 256:1 / 2048:0 |
| NESS_mean_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:0; C_5to6: 128:0 / 256:0 / 2048:0 |
| NES_mean_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:0; C_5to6: 128:0 / 256:0 / 2048:0 |
| NES_bouts | A_het_1to2: 128:7 / 512:7 / 2048:7; A_het_2to3: 128:7 / 512:7 / 2048:7; A_het_4to5: 128:7 / 512:7 / 2048:7; A_het_5to6: 128:7 / 512:7 / 2048:7; A_wt_1to2: 128:7 / 512:7 / 2048:7; A_wt_2to3: 128:7 / 512:7 / 2048:7; A_wt_4to5: 128:7 / 512:7 / 2048:7; A_wt_5to6: 128:7 / 512:7 / 2048:7; B_stage1: 128:0 / 512:4 / 2048:0; B_stage2: 128:0 / 512:1 / 2048:0; B_stage3: 128:3 / 512:0 / 2048:0; B_stage4: 128:0 / 512:0 / 2048:0; B_stage5: 128:1 / 512:0 / 2048:4; B_stage6: 128:3 / 512:0 / 2048:2 |
| NESS_bouts | A_het_1to2: 128:2 / 256:0 / 2048:30; A_het_2to3: 128:0 / 256:4 / 2048:0; A_het_4to5: 128:0 / 256:0 / 2048:0; A_het_5to6: 128:1 / 256:0 / 2048:0; A_wt_1to2: 128:3 / 256:0 / 2048:30; A_wt_2to3: 128:0 / 256:0 / 2048:0; A_wt_4to5: 128:0 / 256:0 / 2048:0; A_wt_5to6: 128:2 / 256:5 / 2048:8; B_stage1: 128:0 / 256:1 / 2048:0; B_stage2: 128:0 / 256:0 / 2048:0; B_stage3: 128:1 / 256:1 / 2048:0; B_stage4: 128:0 / 256:0 / 2048:0; B_stage5: 128:0 / 256:0 / 2048:1; B_stage6: 128:1 / 256:1 / 2048:1 |
| NESS_bouts_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:1 / 2048:0; C_5to6: 128:0 / 256:0 / 2048:0 |
| NES_bouts_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:0; C_5to6: 128:0 / 256:0 / 2048:0 |
| NES_latency | A_het_1to2: 128:10 / 512:7 / 2048:5; A_het_2to3: 128:6 / 512:5 / 2048:5; A_het_4to5: 128:5 / 512:8 / 2048:5; A_het_5to6: 128:4 / 512:5 / 2048:5; A_wt_1to2: 128:6 / 512:8 / 2048:5; A_wt_2to3: 128:5 / 512:6 / 2048:5; A_wt_4to5: 128:5 / 512:5 / 2048:5; A_wt_5to6: 128:9 / 512:5 / 2048:5; B_stage1: 128:0 / 512:4 / 2048:0; B_stage2: 128:0 / 512:0 / 2048:1; B_stage3: 128:0 / 512:0 / 2048:1; B_stage4: 128:0 / 512:0 / 2048:0; B_stage5: 128:0 / 512:0 / 2048:0; B_stage6: 128:0 / 512:0 / 2048:4 |
| NESS_latency | A_het_1to2: 128:0 / 256:0 / 2048:0; A_het_2to3: 128:0 / 256:0 / 2048:0; A_het_4to5: 128:0 / 256:0 / 2048:0; A_het_5to6: 128:0 / 256:0 / 2048:0; A_wt_1to2: 128:2 / 256:1 / 2048:0; A_wt_2to3: 128:0 / 256:0 / 2048:0; A_wt_4to5: 128:0 / 256:0 / 2048:0; A_wt_5to6: 128:0 / 256:0 / 2048:1; B_stage1: 128:0 / 256:1 / 2048:0; B_stage2: 128:0 / 256:0 / 2048:0; B_stage3: 128:0 / 256:0 / 2048:1; B_stage4: 128:0 / 256:0 / 2048:1; B_stage5: 128:0 / 256:0 / 2048:0; B_stage6: 128:0 / 256:0 / 2048:1 |
| NESS_latency_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:1; C_5to6: 128:0 / 256:0 / 2048:0 |
| NES_latency_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:1; C_5to6: 128:0 / 256:0 / 2048:0 |

## mice|v2_mice_c0_w2048_s7_smooth

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | A_het_1to2: 128:2 / 512:2 / 2048:2; A_het_2to3: 128:2 / 512:2 / 2048:2; A_het_4to5: 128:2 / 512:2 / 2048:2; A_het_5to6: 128:1 / 512:2 / 2048:2; A_wt_1to2: 128:2 / 512:2 / 2048:2; A_wt_2to3: 128:2 / 512:2 / 2048:2; A_wt_4to5: 128:2 / 512:2 / 2048:2; A_wt_5to6: 128:2 / 512:2 / 2048:2; B_stage1: 128:2 / 512:1 / 2048:1; B_stage2: 128:1 / 512:1 / 2048:0; B_stage3: 128:1 / 512:1 / 2048:0; B_stage4: 128:1 / 512:0 / 2048:0; B_stage5: 128:1 / 512:0 / 2048:1; B_stage6: 128:1 / 512:1 / 2048:1 |
| NESS_mean | A_het_1to2: 128:0 / 256:1 / 2048:30; A_het_2to3: 128:11 / 256:5 / 2048:1; A_het_4to5: 128:0 / 256:0 / 2048:0; A_het_5to6: 128:0 / 256:0 / 2048:0; A_wt_1to2: 128:0 / 256:0 / 2048:30; A_wt_2to3: 128:0 / 256:0 / 2048:0; A_wt_4to5: 128:0 / 256:0 / 2048:0; A_wt_5to6: 128:2 / 256:3 / 2048:8; B_stage1: 128:1 / 256:1 / 2048:1; B_stage2: 128:1 / 256:1 / 2048:0; B_stage3: 128:1 / 256:1 / 2048:1; B_stage4: 128:1 / 256:0 / 2048:0; B_stage5: 128:1 / 256:0 / 2048:1; B_stage6: 128:2 / 256:2 / 2048:1 |
| NESS_mean_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:1; C_5to6: 128:0 / 256:0 / 2048:0 |
| NES_mean_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:1; C_5to6: 128:0 / 256:0 / 2048:0 |
| NES_bouts | A_het_1to2: 128:7 / 512:7 / 2048:7; A_het_2to3: 128:7 / 512:7 / 2048:7; A_het_4to5: 128:7 / 512:7 / 2048:7; A_het_5to6: 128:7 / 512:7 / 2048:7; A_wt_1to2: 128:7 / 512:7 / 2048:7; A_wt_2to3: 128:7 / 512:7 / 2048:7; A_wt_4to5: 128:7 / 512:7 / 2048:7; A_wt_5to6: 128:7 / 512:7 / 2048:7; B_stage1: 128:2 / 512:6 / 2048:6; B_stage2: 128:2 / 512:1 / 2048:0; B_stage3: 128:2 / 512:1 / 2048:0; B_stage4: 128:1 / 512:2 / 2048:0; B_stage5: 128:1 / 512:4 / 2048:5; B_stage6: 128:4 / 512:5 / 2048:6 |
| NESS_bouts | A_het_1to2: 128:0 / 256:30 / 2048:1; A_het_2to3: 128:8 / 256:7 / 2048:0; A_het_4to5: 128:1 / 256:0 / 2048:0; A_het_5to6: 128:0 / 256:2 / 2048:0; A_wt_1to2: 128:0 / 256:30 / 2048:30; A_wt_2to3: 128:1 / 256:0 / 2048:0; A_wt_4to5: 128:0 / 256:0 / 2048:0; A_wt_5to6: 128:8 / 256:12 / 2048:15; B_stage1: 128:2 / 256:1 / 2048:1; B_stage2: 128:1 / 256:1 / 2048:0; B_stage3: 128:1 / 256:0 / 2048:0; B_stage4: 128:0 / 256:0 / 2048:1; B_stage5: 128:0 / 256:0 / 2048:1; B_stage6: 128:1 / 256:1 / 2048:1 |
| NESS_bouts_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:1 / 256:1 / 2048:0; C_5to6: 128:1 / 256:1 / 2048:0 |
| NES_bouts_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:0; C_5to6: 128:1 / 256:1 / 2048:0 |
| NES_latency | A_het_1to2: 128:8 / 512:7 / 2048:9; A_het_2to3: 128:6 / 512:5 / 2048:5; A_het_4to5: 128:7 / 512:5 / 2048:5; A_het_5to6: 128:4 / 512:7 / 2048:7; A_wt_1to2: 128:6 / 512:6 / 2048:5; A_wt_2to3: 128:6 / 512:5 / 2048:6; A_wt_4to5: 128:5 / 512:5 / 2048:5; A_wt_5to6: 128:4 / 512:6 / 2048:5; B_stage1: 128:0 / 512:0 / 2048:1; B_stage2: 128:1 / 512:0 / 2048:0; B_stage3: 128:2 / 512:1 / 2048:1; B_stage4: 128:0 / 512:1 / 2048:1; B_stage5: 128:2 / 512:2 / 2048:5; B_stage6: 128:0 / 512:0 / 2048:2 |
| NESS_latency | A_het_1to2: 128:0 / 256:0 / 2048:0; A_het_2to3: 128:0 / 256:0 / 2048:0; A_het_4to5: 128:0 / 256:0 / 2048:0; A_het_5to6: 128:0 / 256:0 / 2048:0; A_wt_1to2: 128:0 / 256:0 / 2048:0; A_wt_2to3: 128:0 / 256:0 / 2048:0; A_wt_4to5: 128:0 / 256:0 / 2048:0; A_wt_5to6: 128:1 / 256:0 / 2048:0; B_stage1: 128:0 / 256:0 / 2048:0; B_stage2: 128:0 / 256:0 / 2048:0; B_stage3: 128:0 / 256:0 / 2048:0; B_stage4: 128:0 / 256:0 / 2048:0; B_stage5: 128:1 / 256:0 / 2048:1; B_stage6: 128:0 / 256:0 / 2048:0 |
| NESS_latency_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:1; C_5to6: 128:0 / 256:0 / 2048:0 |
| NES_latency_familyC | C_1to2: 128:0 / 256:0 / 2048:0; C_2to3: 128:0 / 256:0 / 2048:0; C_4to5: 128:0 / 256:0 / 2048:1; C_5to6: 128:0 / 256:0 / 2048:0 |

## ants_eci_r4v2p|v2_ants_c0_w2048_s6

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | v2_1_vs_2: 128:0 / 512:0 / 2048:0 |
| NESS_mean | v2_1_vs_2: 128:0 / 256:0 / 2048:0 |
| NES_bouts | v2_1_vs_2: 128:0 / 512:0 / 2048:0 |
| NESS_bouts | v2_1_vs_2: 128:0 / 256:0 / 2048:0 |
| NES_latency | v2_1_vs_2: 128:0 / 512:0 / 2048:0 |
| NESS_latency | v2_1_vs_2: 128:0 / 256:0 / 2048:0 |

## ants_eci_r4v2p|v2_ants_c0_w2048_s6_smooth

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | v2_1_vs_2: 128:0 / 512:0 / 2048:0 |
| NESS_mean | v2_1_vs_2: 128:0 / 256:0 / 2048:0 |
| NES_bouts | v2_1_vs_2: 128:0 / 512:0 / 2048:0 |
| NESS_bouts | v2_1_vs_2: 128:0 / 256:0 / 2048:0 |
| NES_latency | v2_1_vs_2: 128:0 / 512:0 / 2048:0 |
| NESS_latency | v2_1_vs_2: 128:0 / 256:0 / 2048:1 |

## ants_eci_r4v3p|v2_ants_c0_w2048_s6

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | v3_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NES_mean_v3_2_vs_8_noday | v3_2_vs_6: 128:0 / 512:0 / 2048:0; v3_2_vs_8: 128:0 / 512:2 / 2048:7 |
| NESS_mean | v3_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NESS_mean_v3_2_vs_8_noday | v3_2_vs_8: 128:0 / 256:0 / 2048:0 |
| NES_bouts | v3_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NES_bouts_v3_2_vs_8_noday | v3_2_vs_6: 128:0 / 512:0 / 2048:0; v3_2_vs_8: 128:0 / 512:7 / 2048:39 |
| NESS_bouts | v3_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NESS_bouts_v3_2_vs_8_noday | v3_2_vs_8: 128:0 / 256:0 / 2048:1 |
| NES_latency | v3_2_vs_6: 128:0 / 512:3 / 2048:0 |
| NES_latency_v3_2_vs_8_noday | v3_2_vs_6: 128:0 / 512:0 / 2048:0; v3_2_vs_8: 128:0 / 512:4 / 2048:4 |
| NESS_latency | v3_2_vs_6: 128:0 / 256:1 / 2048:0 |
| NESS_latency_v3_2_vs_8_noday | v3_2_vs_8: 128:0 / 256:0 / 2048:1 |

## ants_eci_r4v3p|v2_ants_c0_w2048_s6_smooth

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | v3_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NES_mean_v3_2_vs_8_noday | v3_2_vs_6: 128:0 / 512:0 / 2048:0; v3_2_vs_8: 128:1 / 512:2 / 2048:3 |
| NESS_mean | v3_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NESS_mean_v3_2_vs_8_noday | v3_2_vs_8: 128:1 / 256:1 / 2048:1 |
| NES_bouts | v3_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NES_bouts_v3_2_vs_8_noday | v3_2_vs_6: 128:0 / 512:0 / 2048:0; v3_2_vs_8: 128:8 / 512:11 / 2048:17 |
| NESS_bouts | v3_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NESS_bouts_v3_2_vs_8_noday | v3_2_vs_8: 128:1 / 256:1 / 2048:1 |
| NES_latency | v3_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NES_latency_v3_2_vs_8_noday | v3_2_vs_6: 128:0 / 512:0 / 2048:0; v3_2_vs_8: 128:0 / 512:4 / 2048:3 |
| NESS_latency | v3_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NESS_latency_v3_2_vs_8_noday | v3_2_vs_8: 128:0 / 256:0 / 2048:0 |

## ants_eci_r4v4p|v2_ants_c0_w2048_s6

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | infested_vs_healthy_t2: 128:0 / 512:1 / 2048:1; infested_vs_healthy_t2_6: 128:2 / 512:1 / 2048:1; infested_vs_healthy_t6: 128:1 / 512:1 / 2048:1; v4_2_vs_6: 128:0 / 512:0 / 2048:0; v4_infested_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NESS_mean | infested_vs_healthy_t2: 128:0 / 256:0 / 2048:1; infested_vs_healthy_t2_6: 128:0 / 256:0 / 2048:0; infested_vs_healthy_t6: 128:0 / 256:0 / 2048:0; v4_2_vs_6: 128:0 / 256:0 / 2048:0; v4_infested_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NES_bouts | infested_vs_healthy_t2: 128:9 / 512:6 / 2048:7; infested_vs_healthy_t2_6: 128:13 / 512:10 / 2048:11; infested_vs_healthy_t6: 128:11 / 512:6 / 2048:6; v4_2_vs_6: 128:0 / 512:0 / 2048:0; v4_infested_2_vs_6: 128:1 / 512:0 / 2048:0 |
| NESS_bouts | infested_vs_healthy_t2: 128:0 / 256:0 / 2048:0; infested_vs_healthy_t2_6: 128:1 / 256:0 / 2048:0; infested_vs_healthy_t6: 128:0 / 256:0 / 2048:0; v4_2_vs_6: 128:0 / 256:0 / 2048:0; v4_infested_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NES_latency | infested_vs_healthy_t2: 128:0 / 512:4 / 2048:4; infested_vs_healthy_t2_6: 128:10 / 512:8 / 2048:4; infested_vs_healthy_t6: 128:4 / 512:4 / 2048:4; v4_2_vs_6: 128:0 / 512:3 / 2048:1; v4_infested_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NESS_latency | infested_vs_healthy_t2: 128:0 / 256:0 / 2048:0; infested_vs_healthy_t2_6: 128:1 / 256:1 / 2048:1; infested_vs_healthy_t6: 128:0 / 256:0 / 2048:0; v4_2_vs_6: 128:0 / 256:0 / 2048:1; v4_infested_2_vs_6: 128:0 / 256:0 / 2048:0 |

## ants_eci_r4v4p|v2_ants_c0_w2048_s6_smooth

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | infested_vs_healthy_t2: 128:1 / 512:1 / 2048:1; infested_vs_healthy_t2_6: 128:2 / 512:1 / 2048:1; infested_vs_healthy_t6: 128:1 / 512:1 / 2048:1; v4_2_vs_6: 128:0 / 512:0 / 2048:0; v4_infested_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NESS_mean | infested_vs_healthy_t2: 128:0 / 256:0 / 2048:0; infested_vs_healthy_t2_6: 128:0 / 256:0 / 2048:0; infested_vs_healthy_t6: 128:0 / 256:0 / 2048:0; v4_2_vs_6: 128:0 / 256:0 / 2048:0; v4_infested_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NES_bouts | infested_vs_healthy_t2: 128:9 / 512:6 / 2048:6; infested_vs_healthy_t2_6: 128:13 / 512:11 / 2048:9; infested_vs_healthy_t6: 128:9 / 512:6 / 2048:6; v4_2_vs_6: 128:0 / 512:0 / 2048:0; v4_infested_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NESS_bouts | infested_vs_healthy_t2: 128:0 / 256:0 / 2048:0; infested_vs_healthy_t2_6: 128:0 / 256:0 / 2048:0; infested_vs_healthy_t6: 128:0 / 256:0 / 2048:0; v4_2_vs_6: 128:0 / 256:0 / 2048:0; v4_infested_2_vs_6: 128:0 / 256:0 / 2048:0 |
| NES_latency | infested_vs_healthy_t2: 128:0 / 512:0 / 2048:1; infested_vs_healthy_t2_6: 128:5 / 512:3 / 2048:5; infested_vs_healthy_t6: 128:1 / 512:1 / 2048:2; v4_2_vs_6: 128:0 / 512:0 / 2048:0; v4_infested_2_vs_6: 128:0 / 512:0 / 2048:0 |
| NESS_latency | infested_vs_healthy_t2: 128:1 / 256:0 / 2048:1; infested_vs_healthy_t2_6: 128:0 / 256:1 / 2048:1; infested_vs_healthy_t6: 128:0 / 256:0 / 2048:0; v4_2_vs_6: 128:0 / 256:0 / 2048:0; v4_infested_2_vs_6: 128:1 / 256:0 / 2048:0 |

## frogs|v2_frogs_c0_w2048_s6

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | En1_vs_WT: 128:0 / 512:0 / 2048:0; FoxP1_vs_WT: 128:1 / 512:1 / 2048:1 |
| NESS_mean | En1_vs_WT: 128:0 / 256:1 / 2048:0; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |
| NES_bouts | En1_vs_WT: 128:1 / 512:0 / 2048:0; FoxP1_vs_WT: 128:6 / 512:6 / 2048:6 |
| NESS_bouts | En1_vs_WT: 128:0 / 256:0 / 2048:0; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |
| NES_latency | En1_vs_WT: 128:3 / 512:2 / 2048:4; FoxP1_vs_WT: 128:4 / 512:4 / 2048:4 |
| NESS_latency | En1_vs_WT: 128:0 / 256:1 / 2048:1; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |

## frogs|v2_frogs_c0_w2048_s6_smooth

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | En1_vs_WT: 128:1 / 512:1 / 2048:0; FoxP1_vs_WT: 128:1 / 512:1 / 2048:1 |
| NESS_mean | En1_vs_WT: 128:0 / 256:0 / 2048:0; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |
| NES_bouts | En1_vs_WT: 128:2 / 512:5 / 2048:2; FoxP1_vs_WT: 128:6 / 512:6 / 2048:7 |
| NESS_bouts | En1_vs_WT: 128:0 / 256:0 / 2048:0; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |
| NES_latency | En1_vs_WT: 128:1 / 512:3 / 2048:2; FoxP1_vs_WT: 128:4 / 512:4 / 2048:4 |
| NESS_latency | En1_vs_WT: 128:1 / 256:0 / 2048:1; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |

## frogs_tadpole|v2_frogs_c0_w2048_s6

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | En1_vs_WT: 128:3 / 512:2 / 2048:2; FoxP1_vs_WT: 128:0 / 512:0 / 2048:1 |
| NESS_mean | En1_vs_WT: 128:0 / 256:0 / 2048:1; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |
| NES_bouts | En1_vs_WT: 128:18 / 512:14 / 2048:14; FoxP1_vs_WT: 128:0 / 512:0 / 2048:11 |
| NESS_bouts | En1_vs_WT: 128:1 / 256:0 / 2048:1; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |
| NES_latency | En1_vs_WT: 128:8 / 512:8 / 2048:9; FoxP1_vs_WT: 128:0 / 512:1 / 2048:3 |
| NESS_latency | En1_vs_WT: 128:0 / 256:0 / 2048:0; FoxP1_vs_WT: 128:0 / 256:0 / 2048:1 |

## frogs_tadpole|v2_frogs_c0_w2048_s6_smooth

| method / aggregation | analysis: counts per prefix |
|---|---|
| NES_mean | En1_vs_WT: 128:2 / 512:3 / 2048:2; FoxP1_vs_WT: 128:1 / 512:0 / 2048:0 |
| NESS_mean | En1_vs_WT: 128:0 / 256:0 / 2048:0; FoxP1_vs_WT: 128:1 / 256:1 / 2048:0 |
| NES_bouts | En1_vs_WT: 128:14 / 512:18 / 2048:13; FoxP1_vs_WT: 128:4 / 512:1 / 2048:1 |
| NESS_bouts | En1_vs_WT: 128:0 / 256:0 / 2048:0; FoxP1_vs_WT: 128:1 / 256:1 / 2048:0 |
| NES_latency | En1_vs_WT: 128:10 / 512:9 / 2048:10; FoxP1_vs_WT: 128:0 / 512:1 / 2048:1 |
| NESS_latency | En1_vs_WT: 128:0 / 256:0 / 2048:0; FoxP1_vs_WT: 128:0 / 256:0 / 2048:0 |

