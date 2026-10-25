# ECI v2 attribution: tight mask (V2F_M0 s901) vs thin-part mask (V2F_N2 s900), mice v1

2592 frames; tokens {"strict_body": 334790, "tail": 129708, "M0_kept": 378661, "N2_kept": 629336, "M0_kept_with_fillT_px": 43871}

## Q2 context leak: token probe on strict body tokens (own pixels identical in orig / tail-occluded)

| target | metric | orig | tail occluded | control occlusion (same shape on bedding) | orig re-encoded alone (numerical floor) | drop orig - tail occl. |
|---|---|---|---|---|---|---|
| K1 tail within 2 tokens (base rate 0.498) | AUROC | 0.725 | 0.730 | 0.725 | 0.725 | -0.005 [-0.007, -0.003] |
| K2 log1p tail px nearby | R^2 | 0.202 | 0.210 | 0.202 | 0.202 | -0.008 |
| K3 direction to tail (n 166790) | R^2 | 0.320 | 0.343 | 0.320 | 0.320 | -0.023 |

K1 leak share of the probe score: -0.016; tail vs control occlusion AUROC -0.005 [-0.007, -0.003]. Q2 context carries tail: False; mostly context: False

| token distance to nearest tail patch | 1 | 2 | 3 | 4-6 | >=7 |
|---|---|---|---|---|---|
| n strict body tokens | 103631 | 122938 | 69224 | 36582 | 2415 |
| median rel. change of z, tail occluded | 0.173 | 0.117 | 0.100 | 0.091 | 0.072 |
| median rel. change of z, control occlusion | 0.019 | 0.020 | 0.020 | 0.019 | 0.018 |
| median rel. change of z, numerical floor | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| share M0 SAE active set changed, tail occluded | 0.970 | 0.933 | 0.903 | 0.879 | 0.797 |
| share M0 SAE active set changed, control occlusion | 0.430 | 0.440 | 0.430 | 0.418 | 0.387 |
| share M0 SAE active set changed, numerical floor | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Frame-level probes (5-fold by pool)

F3 base rate 1.000

| features | F1 log1p tail px R^2 | F2 n tails R^2 | F3 tail visible AUROC | F1 R^2 tail occluded |
|---|---|---|---|---|
| strict_body_mean | 0.414 | 0.390 | - | 0.282 (mean pred 8.43 -> 8.29) |
| tail_tokens_mean | 0.454 | 0.523 | - | - |
| M0_kept_mean | 0.380 | 0.396 | - | - |
| N2_kept_mean | 0.548 | 0.636 | - | - |
| M0_frame_codes | 0.324 | 0.330 | - | - |
| N2_frame_codes | 0.513 | 0.573 | - | - |

## Q3 occlusion attribution (TEST frames, top 30 per neuron; medians)

| group | neuron | sel. score | a | tail share D_T/(D_T+D_B) | D_T/a | D_Tout/a | D_B/a | D_C/a (control) | |noise|/a | heat-map tail mass | flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N2_tail | 1779 | 1.000 | 10.453 | 1.000 | 0.011 | 0.008 | 0.000 | 0.003 | 0.000 | 1.000 | faithful True |
| N2_tail | 1671 | 1.000 | 5.344 | 0.607 | 0.054 | 0.051 | 0.028 | 0.005 | 0.000 | 1.000 | faithful False |
| N2_tail | 1367 | 1.000 | 8.375 | 1.000 | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 1.000 | faithful True |
| N2_tail | 1317 | 1.000 | 9.539 | 0.273 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | faithful False |
| N2_tail | 981 | 1.000 | 6.832 | 1.000 | 0.024 | 0.026 | 0.000 | 0.004 | 0.000 | 1.000 | faithful True |
| M0_token | 49 | 0.539 | 18.688 | 0.118 | 0.060 | 0.050 | 0.482 | 0.003 | 0.000 | 0.039 | unseen-tail False, misleading False |
| M0_token | 97 | 0.538 | 28.633 | 0.294 | 0.002 | 0.001 | 0.024 | 0.000 | 0.000 | 0.122 | unseen-tail False, misleading False |
| M0_token | 55 | 0.533 | 15.590 | 0.064 | 0.042 | 0.047 | 0.750 | 0.001 | 0.000 | 0.000 | unseen-tail False, misleading False |
| M0_token | 123 | 0.525 | 12.414 | 0.106 | 0.045 | 0.004 | 0.394 | 0.000 | 0.000 | 0.019 | unseen-tail False, misleading False |
| M0_token | 10 | 0.520 | 9.922 | 0.060 | 0.038 | 0.018 | 0.643 | 0.002 | 0.000 | 0.028 | unseen-tail False, misleading False |
| M0_frame | 55 | 0.373 | 15.590 | 0.064 | 0.042 | 0.047 | 0.750 | 0.001 | 0.000 | 0.000 | unseen-tail False, misleading False |
| M0_frame | 88 | 0.332 | 27.070 | 0.308 | 0.383 | 0.260 | 0.879 | 0.000 | 0.000 | 0.008 | unseen-tail False, misleading False |
| M0_frame | 2 | 0.332 | 11.820 | 0.053 | 0.052 | 0.027 | 1.000 | 0.000 | 0.000 | 0.000 | unseen-tail False, misleading False |
| M0_frame | 80 | 0.324 | 26.852 | 0.000 | 0.000 | 0.000 | 0.951 | 0.000 | 0.000 | 0.000 | unseen-tail False, misleading False |
| M0_frame | 36 | 0.317 | 14.594 | 0.015 | 0.015 | 0.005 | 0.798 | 0.000 | 0.000 | 0.000 | unseen-tail False, misleading False |
| M0_random (20) | median of medians | - | 4.364 | 0.039 | 0.041 | 0.001 | 1.000 | 0.001 | 0.000 | 0.000 | |
| N2_random (20) | median of medians | - | 4.413 | 0.110 | 0.054 | 0.033 | 1.000 | 0.001 | 0.000 | 0.000 | |
