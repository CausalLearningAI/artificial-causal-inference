# CalMS21: DINOv3-L block 20 token grid 32 / 48 / 64 / 80 (pre-registered, scripts/eci/resolution/calms21_res.sh)

primary AUROC = mean over attack, investigation, mount of the cross-fitted best-neuron AUROC (AUROC-picked neuron), split seeds 0 and 1 averaged; one value per SAE seed (0-5). Same frames, same mask cells (pass A, 32 grid), same recipe (BatchTopK 4096, k 16, 6000 steps, Delta 2 s). D3L-32 and D3L-64 are the existing arms of eci_res.sh (their per-seed eval JSONs, not rerun). D3L-80 (1280 px) is UPSAMPLED beyond the native 1024 canvas: it tests token density, not image detail. Body length 160.1 native px.

| arm | grid | input px | patches / body | per-seed primary AUROC | mean ± SE (sd) primary AUROC | cf AUROC attack / investigation / mount | encoder tokens / frame (t and t-60) | SAE tokens / frame (after mask) | encode frames / GPU-s |
|---|---|---|---|---|---|---|---|---|---|
| D3L-32 (d3l_512_L20) | 32 | 512 | 5.0 | 0.878, 0.846, 0.861, 0.834, 0.852, 0.834 | 0.8509 ± 0.0069 (sd 0.0169) | 0.855 / 0.802 / 0.896 | 2,058 | 75 | 46.7 |
| D3L-48 (d3l_768_L20) | 48 | 768 | 7.5 | 0.860, 0.888, 0.917, 0.882, 0.866, 0.887 | 0.8834 ± 0.0082 (sd 0.0200) | 0.879 / 0.817 / 0.954 | 4,618 | 168 | 22.2 |
| D3L-64 (d3l_1024_L20) | 64 | 1024 | 10.0 | 0.840, 0.897, 0.902, 0.892, 0.877, 0.880 | 0.8813 ± 0.0091 (sd 0.0223) | 0.872 / 0.830 / 0.942 | 8,202 | 300 | 10.7 |
| D3L-80 (d3l_1280_L20) | 80 | 1280 (upsampled) | 12.5 | 0.907, 0.904, 0.903, 0.900, 0.881, 0.908 | 0.9006 ± 0.0041 (sd 0.0102) | 0.891 / 0.840 / 0.970 | 12,810 | 468 | 6.1 |

## (e1) adjacent grids, WIN = delta primary AUROC >= +0.010 and Welch 95% CI excludes 0 (LOSES symmetric)

| contrast | delta primary AUROC [95% CI] | df | verdict |
|---|---|---|---|
| D3L-48 vs D3L-32 | +0.0325 [+0.0086, +0.0564] | 9.7 | WIN |
| D3L-64 vs D3L-48 | -0.0021 [-0.0294, +0.0253] | 9.9 | NO DIFFERENCE |
| D3L-80 vs D3L-64 | +0.0193 [-0.0044, +0.0429] | 7.0 | NO DIFFERENCE |

## (e2) grid choice: best-mean grid = D3L-80; choice = smallest grid not LOSING to it = **D3L-48 (48 x 48)**

| arm | delta primary AUROC vs best [95% CI] | verdict |
|---|---|---|
| D3L-32 | -0.0497 [-0.0682, -0.0313] | LOSES |
| D3L-48 | -0.0172 [-0.0387, +0.0042] | NO DIFFERENCE |
| D3L-64 | -0.0193 [-0.0429, +0.0044] | NO DIFFERENCE |
| D3L-80 | - | BEST |

## jobs (GPU-hours = elapsed x 1 GPU, sacct)

| job | name | state | GPU-h | node |
|---|---|---|---|---|
| 910747 | res_calms_enc_train | COMPLETED | 0.82 | gpu286 |
| 910748 | res_calms_enc_eval | COMPLETED | 0.68 | gpu279 |
| 910749 | res_calms_tr_d3l_768_L20 | COMPLETED | 0.14 | gpu279 |
| 910750 | res_calms_tr_d3l_1280_L20 | COMPLETED | 0.19 | gpu279 |

total GPU-h (all listed jobs): 1.84
