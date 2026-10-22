| neurons | score | v1 | M0-v2 | N2-v2 | M0 - v1 (95% t-CI) | N2 - v1 (95% t-CI) | N2 better? | N2 - M0 (Welch) |
|---|---|---|---|---|---|---|---|---|
| first256 | BOX-F R~ | 0.3368 | 0.4542 ± 0.2857 (n=20) | 0.3431 ± 0.0894 (n=20) | +0.1174 [-0.0163, +0.2511] | +0.0063 [-0.0356, +0.0481] | False | -0.1111 [-0.2497, +0.0275] |
| first256 | TAIL-T share | 0.9150 | 0.6027 ± 0.1442 (n=20) | 0.9000 ± 0.0378 (n=20) | -0.3123 [-0.3798, -0.2448] | -0.0150 [-0.0327, +0.0027] | False | +0.2973 [+0.2281, +0.3665] |
| first256 | TAIL-MAB share | 0.9381 | 0.5707 ± 0.1566 (n=20) | 0.9528 ± 0.0204 (n=20) | -0.3674 [-0.4407, -0.2942] | +0.0147 [+0.0051, +0.0243] | False | +0.3821 [+0.3084, +0.4559] |
| first256 | BOX-T share | 0.6086 | 0.4454 ± 0.1791 (n=20) | 0.5613 ± 0.1616 (n=20) | -0.1631 [-0.2469, -0.0793] | -0.0473 [-0.1229, +0.0283] | False | +0.1158 [+0.0066, +0.2251] |
| first256 | TAIL-F R~ | 0.1027 | 0.0981 ± 0.0048 (n=20) | 0.2405 ± 0.0149 (n=20) | -0.0046 [-0.0068, -0.0023] | +0.1378 [+0.1308, +0.1448] | True | +0.1424 [+0.1352, +0.1497] |
| first256 | TAIL2-F R~ (2+ tails) | 0.0047 | 0.0047 ± 0.0072 (n=20) | 0.0018 ± 0.0019 (n=20) | -0.0001 [-0.0034, +0.0033] | -0.0030 [-0.0039, -0.0021] | False | -0.0029 [-0.0063, +0.0005] |
| all | BOX-F R~ | 0.5388 | 0.5038 ± 0.2441 (n=20) | 0.4546 ± 0.0897 (n=20) | -0.0350 [-0.1492, +0.0793] | -0.0842 [-0.1262, -0.0422] | False | -0.0493 [-0.1693, +0.0707] |
| all | TAIL-T share | 0.8340 | 0.7228 ± 0.0479 (n=20) | 0.8802 ± 0.0771 (n=20) | -0.1112 [-0.1337, -0.0888] | +0.0462 [+0.0101, +0.0823] | False | +0.1574 [+0.1161, +0.1988] |
| all | TAIL-MAB share | 0.9756 | 0.7186 ± 0.0869 (n=20) | 0.9579 ± 0.0234 (n=20) | -0.2570 [-0.2976, -0.2163] | -0.0176 [-0.0286, -0.0067] | False | +0.2393 [+0.1975, +0.2811] |
| all | BOX-T share | 0.8684 | 0.8772 ± 0.0523 (n=20) | 0.8481 ± 0.1284 (n=20) | +0.0088 [-0.0156, +0.0333] | -0.0204 [-0.0805, +0.0398] | False | -0.0292 [-0.0930, +0.0347] |
| all | TAIL-F R~ | 0.1326 | 0.0996 ± 0.0040 (n=20) | 0.2408 ± 0.0135 (n=20) | -0.0330 [-0.0349, -0.0312] | +0.1082 [+0.1018, +0.1145] | True | +0.1412 [+0.1347, +0.1477] |
| all | TAIL2-F R~ (2+ tails) | 0.0047 | 0.0047 ± 0.0072 (n=20) | 0.0018 ± 0.0019 (n=20) | -0.0001 [-0.0034, +0.0033] | -0.0030 [-0.0039, -0.0021] | False | -0.0029 [-0.0063, +0.0005] |

VERDICT: {"TAIL (N2 better than v1)": false, "BOX (N2 better than v1)": false, "reported: TAIL (M0 better than v1)": false, "reported: BOX (M0 better than v1)": false}

```
{
 "tail_neuron_exists_share_of_seeds (held-out appendage share > 0.5, first 256)": {
  "M0": [
   0.65,
   0.4893604849295929,
   20
  ],
  "N2": [
   1.0,
   0.0,
   20
  ],
  "v1": 1.0
 },
 "targets": {
  "n_frames": 43200,
  "base_rate": {
   "box": 0.3083564814814815,
   "tail_vis": 0.20006944444444444,
   "tail2": 0.9989583333333333
  },
  "box_zone_radius_px_1024_median": 163.625,
  "body_tokens_in_zone_per_frame_mean": 11.002037037037036
 },
 "v1_checks": {
  "tokens_per_frame": 148.8796064814815,
  "check_vs_deployed_codes_max": {
   "max_abs_diff": 1.64453125,
   "mean_abs_diff_per_frame_max": 0.00043418601853773,
   "share_frames_diff_gt_0.01": 0.0033564814814814816
  },
  "mask_jaccard_vs_M0_32grid": 0.3821724875261957,
  "mask_jaccard_vs_N2_32grid": 0.5300839567708153,
  "check_nose_nose_R": 0.05279733906000321,
  "check_nose_tail_R": 0.09001276564908495
 },
 "v2_identity_checks": {
  "M0": [
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ],
  "N2": [
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0,
   0.0
  ]
 },
 "examples": {
  "v1": {
   "tail": {
    "neuron": 75,
    "sheet": "results/vision/eci_v2_mask2/discovery/sheets/v1_tail_n75.jpg",
    "n_frames": 24,
    "frames_box_target_share": 0.5
   },
   "box": {
    "neuron": 192,
    "sheet": "results/vision/eci_v2_mask2/discovery/sheets/v1_box_n192.jpg",
    "n_frames": 24,
    "frames_box_target_share": 0.7916666666666666
   }
  },
  "M0": {
   "tail": {
    "neuron": 48,
    "sheet": "results/vision/eci_v2_mask2/discovery/sheets/M0_s70_tail_n48.jpg",
    "n_frames": 24,
    "peak_token_app_frac_mean": 0.23697916666666666,
    "peak_token_in_zone_share": 0.0,
    "frames_box_target_share": 0.25
   },
   "box": {
    "neuron": 110,
    "sheet": "results/vision/eci_v2_mask2/discovery/sheets/M0_s70_box_n110.jpg",
    "n_frames": 24,
    "peak_token_app_frac_mean": 0.14225260416666666,
    "peak_token_in_zone_share": 0.9583333333333334,
    "frames_box_target_share": 1.0
   }
  },
  "N2": {
   "tail": {
    "neuron": 201,
    "sheet": "results/vision/eci_v2_mask2/discovery/sheets/N2_s71_tail_n201.jpg",
    "n_frames": 24,
    "peak_token_app_frac_mean": 0.4840494791666667,
    "peak_token_in_zone_share": 0.0,
    "frames_box_target_share": 0.2916666666666667
   },
   "box": {
    "neuron": 65,
    "sheet": "results/vision/eci_v2_mask2/discovery/sheets/N2_s71_box_n65.jpg",
    "n_frames": 24,
    "peak_token_app_frac_mean": 0.166015625,
    "peak_token_in_zone_share": 0.125,
    "frames_box_target_share": 0.5833333333333334
   }
  }
 },
 "n_seeds": {
  "M0": 20,
  "N2": 20
 },
 "n_seeds_mab": {
  "M0": 20,
  "N2": 20
 }
}
```