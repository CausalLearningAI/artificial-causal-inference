# ECI v0 / v1 / v2 at a glance (labelled-behaviour audit, 2026-10-26)

**This is an audit with metrics chosen earlier. Nothing new is pre-registered.** Primary metric = AUROC, secondary = nAP (both from results/vision/eci_v2_prob/decision.json). The scorer is scripts/eci/v2_eval/eval.py; it imports the v2_prob metric code unchanged.

Metrics, one line each:
- **AUROC (primary)**: pick the single neuron with the highest AUROC on half A of the videos (mice: half of the pools), then measure that neuron's AUROC on half B. Average over 6 splits (the atlas split plus 5 random ones) x 2 directions. 0.5 = chance.
- **nAP (secondary)**: (average precision - base rate) / (1 - base rate), cross-fitted the same way, with the neuron picked by nAP. 0 = chance. It depends on prevalence, so compare it only within a row.
- Best-threshold F1 and R~ (legacy) are in table.json only.

Models:
- **v0** = the original paper recipe (mean-pooled DINOv2-S token, 2 fps).
- **v1 deployed** = matryoshka_btk_1024_k16_*_s0. v1 mean ± sd is over 11 dictionaries (s0 plus retrained s1-10).
- **v2** = V2F_N2. Mice: seeds 900-919 (n=20). Ants: seeds 900-932 (n=33). Deployment seed = s900. The V2F decide step picks one seed for both domains: the highest mice held-out FVE among 900-902.
- **v2 - v1 mean** = Welch 95% CI over the seeds.

Frames:
- **mice**: the v2_prob Tier B probe windows, 144 videos x 600 contiguous 5 fps frames.
- **ants**: every labelled frame on the 1 fps full1 grid, 10 min x 212 videos (v2 44, v3 107, v4 40, v4_infested 21).
- All models are scored on the same frames in each row. One exception: mice v0 has codes only on its 2 fps subset of those frames. On that subset, v1 mean AUROC = nn 0.706 and v2 mean AUROC = nn 0.803, the same as on all frames.

Readout:
- v1 = raw frame max (its deployed readout).
- v2 = 0.6 s smoothed frame max on mice. On ants it is the raw frame max, because frames are 1 s apart and cannot be smoothed.
- v0 = raw.
- Mice sensitivity, with the same readout for both models: if v1 is also smoothed, v1 mean AUROC rises to nn 0.716 / np 0.689 / nose_nose 0.696 / nt 0.725. If v2 is left raw, v2 mean AUROC falls to 0.768 / 0.708 / 0.726 / 0.696. With both raw, v2 - v1 on nt is +0.003 AUROC; with both smoothed it is +0.016 AUROC. So part of v2's mice nt lead comes from the smoothing.

A row is included if it has **>= 20 bouts and >= 3 positive videos in EACH half of the atlas split**, counted on its scored frames. This rule was given by the coordinator; no such rule existed in code before (see coverage.md).

## AUROC (primary)

| domain | series | behaviour | scored frames (positive) | bouts half A / B | v0 AUROC | v1 deployed AUROC | v1 AUROC mean ± sd (n) | v2 AUROC mean ± sd (n) | v2 deploy s900 AUROC | v2 - v1 mean AUROC [Welch 95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| mice | v1 | nn | 86,400 (608) | 132 / 100 | 0.513 | 0.725 | 0.704 ± 0.048 (11) | 0.801 ± 0.051 (20) | 0.813 | +0.097 [+0.059, +0.135] |
| mice | v1 | np | 86,400 (1,248) | 205 / 107 | 0.487 | 0.610 | 0.661 ± 0.057 (11) | 0.752 ± 0.040 (20) | 0.757 | +0.091 [+0.050, +0.132] |
| mice | v1 | nose_nose | 86,400 (1,852) | 334 / 207 | 0.532 | 0.681 | 0.670 ± 0.055 (11) | 0.761 ± 0.048 (20) | 0.779 | +0.091 [+0.049, +0.132] |
| mice | v1 | nt | 86,400 (728) | 67 / 54 | 0.531 | 0.747 | 0.693 ± 0.039 (11) | 0.741 ± 0.024 (20) | 0.743 | +0.048 [+0.021, +0.075] |
| ants | v2 | Y2F | 26,400 (3,542) | 59 / 61 | 0.516 | 0.861 | 0.882 ± 0.031 (11) | 0.851 ± 0.025 (33) | 0.843 | -0.030 [-0.053, -0.008] |
| ants | v2 | B2F | 26,400 (2,989) | 45 / 55 | 0.511 | 0.882 | 0.871 ± 0.040 (11) | 0.851 ± 0.016 (33) | 0.855 | -0.020 [-0.047, +0.007] |
| ants | v2 | groom_any | 26,400 (5,904) | 81 / 99 | 0.538 | 0.900 | 0.911 ± 0.036 (11) | 0.882 ± 0.016 (33) | 0.877 | -0.029 [-0.054, -0.004] |
| ants | v3 | Y2F | 64,200 (12,606) | 178 / 173 | 0.507 | 0.846 | 0.837 ± 0.039 (11) | 0.817 ± 0.010 (33) | 0.819 | -0.020 [-0.046, +0.007] |
| ants | v3 | B2F | 64,200 (12,203) | 148 / 169 | 0.551 | 0.857 | 0.833 ± 0.043 (11) | 0.833 ± 0.009 (33) | 0.837 | +0.000 [-0.029, +0.029] |
| ants | v3 | groom_any | 64,200 (21,173) | 241 / 275 | 0.525 | 0.889 | 0.871 ± 0.050 (11) | 0.866 ± 0.011 (33) | 0.873 | -0.004 [-0.038, +0.030] |
| ants | v4 | Y2F | 24,000 (7,542) | 114 / 141 | 0.618 | 0.811 | 0.791 ± 0.039 (11) | 0.760 ± 0.017 (33) | 0.769 | -0.031 [-0.057, -0.004] |
| ants | v4 | B2F | 24,000 (7,317) | 84 / 128 | 0.517 | 0.795 | 0.770 ± 0.035 (11) | 0.740 ± 0.016 (33) | 0.733 | -0.030 [-0.054, -0.006] |
| ants | v4 | groom_any | 24,000 (11,719) | 142 / 152 | 0.588 | 0.844 | 0.820 ± 0.039 (11) | 0.803 ± 0.012 (33) | 0.789 | -0.017 [-0.043, +0.010] |
| ants | v4_infested | Y2F | 12,600 (2,789) | 29 / 36 | 0.521 | 0.866 | 0.853 ± 0.046 (11) | 0.829 ± 0.021 (33) | 0.838 | -0.024 [-0.055, +0.008] |
| ants | v4_infested | B2F | 12,600 (2,380) | 23 / 32 | 0.616 | 0.846 | 0.821 ± 0.048 (11) | 0.796 ± 0.021 (33) | 0.801 | -0.025 [-0.057, +0.008] |
| ants | v4_infested | groom_any | 12,600 (4,317) | 42 / 45 | 0.586 | 0.895 | 0.883 ± 0.042 (11) | 0.885 ± 0.019 (33) | 0.897 | +0.002 [-0.027, +0.031] |
| **mice (mean of 4 rows)** | | | | | 0.516 | 0.691 | 0.682 ± 0.036 (11) | 0.764 ± 0.031 (20) | 0.773 | +0.082 [+0.055, +0.109] |
| **ants (mean of 12 rows)** | | | | | 0.549 | 0.858 | 0.845 ± 0.038 (11) | 0.826 ± 0.010 (33) | 0.828 | -0.019 [-0.044, +0.007] |

## nAP (secondary)

| domain | series | behaviour | scored frames (positive) | bouts half A / B | v0 nAP | v1 deployed nAP | v1 nAP mean ± sd (n) | v2 nAP mean ± sd (n) | v2 deploy s900 nAP | v2 - v1 mean nAP [Welch 95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| mice | v1 | nn | 86,400 (608) | 132 / 100 | 0.004 | 0.008 | 0.012 ± 0.014 (11) | 0.051 ± 0.038 (20) | 0.055 | +0.039 [+0.019, +0.058] |
| mice | v1 | np | 86,400 (1,248) | 205 / 107 | 0.001 | 0.002 | 0.009 ± 0.012 (11) | 0.063 ± 0.036 (20) | 0.093 | +0.054 [+0.036, +0.072] |
| mice | v1 | nose_nose | 86,400 (1,852) | 334 / 207 | 0.001 | 0.022 | 0.029 ± 0.022 (11) | 0.120 ± 0.057 (20) | 0.153 | +0.091 [+0.062, +0.120] |
| mice | v1 | nt | 86,400 (728) | 67 / 54 | 0.002 | 0.037 | 0.014 ± 0.009 (11) | 0.020 ± 0.016 (20) | 0.035 | +0.006 [-0.003, +0.016] |
| ants | v2 | Y2F | 26,400 (3,542) | 59 / 61 | 0.010 | 0.471 | 0.482 ± 0.064 (11) | 0.389 ± 0.070 (33) | 0.381 | -0.093 [-0.141, -0.046] |
| ants | v2 | B2F | 26,400 (2,989) | 45 / 55 | 0.014 | 0.435 | 0.396 ± 0.074 (11) | 0.381 ± 0.024 (33) | 0.398 | -0.015 [-0.065, +0.036] |
| ants | v2 | groom_any | 26,400 (5,904) | 81 / 99 | 0.025 | 0.698 | 0.703 ± 0.081 (11) | 0.633 ± 0.035 (33) | 0.612 | -0.070 [-0.125, -0.015] |
| ants | v3 | Y2F | 64,200 (12,606) | 178 / 173 | 0.009 | 0.438 | 0.431 ± 0.063 (11) | 0.356 ± 0.025 (33) | 0.339 | -0.076 [-0.118, -0.033] |
| ants | v3 | B2F | 64,200 (12,203) | 148 / 169 | 0.026 | 0.474 | 0.421 ± 0.075 (11) | 0.394 ± 0.015 (33) | 0.383 | -0.027 [-0.078, +0.024] |
| ants | v3 | groom_any | 64,200 (21,173) | 241 / 275 | 0.033 | 0.667 | 0.634 ± 0.091 (11) | 0.616 ± 0.017 (33) | 0.617 | -0.018 [-0.079, +0.043] |
| ants | v4 | Y2F | 24,000 (7,542) | 114 / 141 | 0.061 | 0.473 | 0.432 ± 0.094 (11) | 0.327 ± 0.046 (33) | 0.385 | -0.105 [-0.169, -0.040] |
| ants | v4 | B2F | 24,000 (7,317) | 84 / 128 | 0.033 | 0.398 | 0.380 ± 0.069 (11) | 0.287 ± 0.051 (33) | 0.288 | -0.093 [-0.142, -0.045] |
| ants | v4 | groom_any | 24,000 (11,719) | 142 / 152 | 0.129 | 0.615 | 0.569 ± 0.079 (11) | 0.489 ± 0.042 (33) | 0.516 | -0.080 [-0.134, -0.026] |
| ants | v4_infested | Y2F | 12,600 (2,789) | 29 / 36 | 0.057 | 0.533 | 0.517 ± 0.088 (11) | 0.429 ± 0.040 (33) | 0.383 | -0.088 [-0.148, -0.028] |
| ants | v4_infested | B2F | 12,600 (2,380) | 23 / 32 | 0.109 | 0.407 | 0.379 ± 0.084 (11) | 0.327 ± 0.068 (33) | 0.376 | -0.052 [-0.112, +0.008] |
| ants | v4_infested | groom_any | 12,600 (4,317) | 42 / 45 | 0.151 | 0.711 | 0.686 ± 0.085 (11) | 0.707 ± 0.033 (33) | 0.719 | +0.021 [-0.036, +0.079] |
| **mice (mean of 4 rows)** | | | | | 0.002 | 0.017 | 0.016 ± 0.011 (11) | 0.064 ± 0.031 (20) | 0.084 | +0.048 [+0.032, +0.063] |
| **ants (mean of 12 rows)** | | | | | 0.055 | 0.527 | 0.503 ± 0.071 (11) | 0.445 ± 0.021 (33) | 0.450 | -0.058 [-0.106, -0.010] |

## Rows below the eligibility rule (scored, NOT for conclusions; few bouts / videos)

AUROC:

| domain | series | behaviour | scored frames (positive) | bouts half A / B | v0 AUROC | v1 deployed AUROC | v1 AUROC mean ± sd (n) | v2 AUROC mean ± sd (n) | v2 deploy s900 AUROC | v2 - v1 mean AUROC [Welch 95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| ants | v3 | FOL | 64,200 (213) | 1 / 2 | 0.487 | 0.726 | 0.776 ± 0.044 (11) | 0.737 ± 0.034 (33) | 0.756 | -0.039 [-0.070, -0.008] |
| ants | v3 | YOL | 64,200 (1,754) | 13 / 10 | 0.533 | 0.862 | 0.861 ± 0.025 (11) | 0.864 ± 0.022 (33) | 0.846 | +0.002 [-0.015, +0.020] |
| ants | v3 | BOL | 64,200 (420) | 2 / 13 | 0.570 | 0.589 | 0.597 ± 0.028 (11) | 0.671 ± 0.025 (33) | 0.686 | +0.075 [+0.054, +0.095] |
| ants | v3 | onlid_any | 64,200 (2,174) | 15 / 23 | 0.497 | 0.890 | 0.913 ± 0.015 (11) | 0.936 ± 0.005 (33) | 0.937 | +0.023 [+0.013, +0.033] |
| ants | v3 | lid_any3 | 64,200 (2,376) | 16 / 23 | 0.491 | 0.895 | 0.916 ± 0.014 (11) | 0.941 ± 0.006 (33) | 0.943 | +0.025 [+0.015, +0.035] |
| ants | v4 | BOL | 24,000 (189) | 2 / 0 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4 | onlid_any | 24,000 (189) | 2 / 0 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4 | lid_any3 | 24,000 (189) | 2 / 0 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4_infested | BOL | 12,600 (61) | 0 / 1 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4_infested | onlid_any | 12,600 (61) | 0 / 1 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4_infested | lid_any3 | 12,600 (61) | 0 / 1 | - | - | - ± - (0) | - ± - (0) | - | - |

nAP:

| domain | series | behaviour | scored frames (positive) | bouts half A / B | v0 nAP | v1 deployed nAP | v1 nAP mean ± sd (n) | v2 nAP mean ± sd (n) | v2 deploy s900 nAP | v2 - v1 mean nAP [Welch 95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| ants | v3 | FOL | 64,200 (213) | 1 / 2 | 0.000 | 0.008 | 0.006 ± 0.005 (11) | 0.004 ± 0.001 (33) | 0.005 | -0.002 [-0.006, +0.001] |
| ants | v3 | YOL | 64,200 (1,754) | 13 / 10 | 0.054 | 0.565 | 0.632 ± 0.052 (11) | 0.468 ± 0.063 (33) | 0.415 | -0.163 [-0.203, -0.123] |
| ants | v3 | BOL | 64,200 (420) | 2 / 13 | 0.000 | 0.004 | 0.004 ± 0.004 (11) | 0.002 ± 0.006 (33) | 0.001 | -0.002 [-0.005, +0.001] |
| ants | v3 | onlid_any | 64,200 (2,174) | 15 / 23 | 0.030 | 0.623 | 0.699 ± 0.043 (11) | 0.724 ± 0.030 (33) | 0.634 | +0.025 [-0.005, +0.055] |
| ants | v3 | lid_any3 | 64,200 (2,376) | 16 / 23 | 0.025 | 0.784 | 0.772 ± 0.036 (11) | 0.853 ± 0.005 (33) | 0.856 | +0.081 [+0.057, +0.105] |
| ants | v4 | BOL | 24,000 (189) | 2 / 0 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4 | onlid_any | 24,000 (189) | 2 / 0 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4 | lid_any3 | 24,000 (189) | 2 / 0 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4_infested | BOL | 12,600 (61) | 0 / 1 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4_infested | onlid_any | 12,600 (61) | 0 / 1 | - | - | - ± - (0) | - ± - (0) | - | - |
| ants | v4_infested | lid_any3 | 12,600 (61) | 0 / 1 | - | - | - ± - (0) | - ± - (0) | - | - |

ants v4 / v4_infested on-lid rows have 1-2 bouts in a single half, so no metric can be computed for them.

## Checks

- Scorer selftest: on the v2_prob Tier B frames and codes, eval.py reproduces the v2_prob A3 cross-fitted Rt / nAP / AUROC / F1max for v1_s0, v1_s3 and V2F_N2_s900, both domains. Largest difference: 0.0 (results/vision/eci_v2_eval/selftest/).
- Leakage: halves are made of whole units (ants = videos, mice = pools = observation_id without its last two fields). Smoothing happens inside each video window. The neuron is picked on half A only and scored on half B, in both directions. Neuron eligibility (active on >= 0.2% of frames) uses no labels.
- Same frames for every model: one frame table per domain (eci_codes/v2_eval/<dom>/frames.parquet). Ants v1 codes were re-encoded with v1's own token pipeline on these frames. They match the stored deployed codes_max (fire agreement 99.999% eci30, 99.997% c1v4). Scoring the stored codes directly gives the same AUROC on every row. Ants V2F_N2 codes match an independent encode of the same frames from v2_prob (fire agreement 1.000, 11,700 shared frames).
- Label-shuffle null: label vectors were permuted across the videos of each series, 3 permutations, for v0, v1_s0 and V2F_N2_s900 (table.json checks.null_*). For v1 and v2 on eligible rows, mean null AUROC is 0.47-0.52 and mean null R~ is between -0.03 and +0.03. For v0, the mice nt null AUROC is 0.573 (range 0.483-0.664). This is above v0's real nt AUROC of 0.531, so v0's mice numbers are at chance; v0's ants null is 0.47-0.52 AUROC.
- Reproduction of earlier tables: mice v1_s0 AUROC on the probe frames with both models smoothed = nn|np 0.699 and nt 0.785. These equal results/vision/eci_v2_prob/table.md A3 (v1_dep 0.699 / 0.785). Ants v1 deployed AUROC on the 1 fps frames is v2 groom 0.900 and v3 groom 0.889, close to the atlas align values (results/vision/eci_atlas_v37/align, v1 prefix 256, 1 fps frames offset by 0.4 s: 0.902 / 0.889).
- Compute: 0.76 GPU-h (job 1120613: V2F_N2 readout of the full1 tokens plus the v1 token encode). Everything else ran on CPU.
