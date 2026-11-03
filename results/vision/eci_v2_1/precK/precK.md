## mice

identity: top-1% precision recomputed vs stored, max abs diff 0.00e+00 over 92 (model, cell) values; weights-vs-direct precK diff 0.00e+00

| cell | eval frames | base rate | held-out frames/direction (median) | 1% of held-out | dirs with held-out < 1000 |
|---|---|---|---|---|---|
| nose-nose-mutual (nn) | 86400 | 0.0070 | 43200 | 432 | 0/12 |
| nose-nose-passive (np) | 86400 | 0.0144 | 43200 | 432 | 0/12 |
| nose-nose (nose_nose) | 86400 | 0.0214 | 43200 | 432 | 0/12 |
| nose-tail (nt) | 86400 | 0.0084 | 43200 | 432 | 0/12 |

### gate (mean over 4 cells; levels = seed median)

| arm | precision@100 | precision@1000 | top-1% precision |
|---|---|---|---|
| v1 | 0.070 | 0.051 | 0.061 |
| S | 0.096 | 0.070 | 0.090 |
| V | 0.123 | 0.100 | 0.124 |

| contrast | d precision@100 [95% CI pools/videos+seeds] | d precision@1000 [95% CI] |
|---|---|---|
| V - v1 | +0.057 [+0.019, +0.100] | +0.049 [+0.028, +0.074] |
| S - v1 | +0.028 [-0.005, +0.069] | +0.016 [+0.002, +0.034] |
| V - S | +0.029 [-0.015, +0.067] | +0.033 [+0.015, +0.053] |

## ants

identity: top-1% precision recomputed vs stored, max abs diff 0.00e+00 over 414 (model, cell) values; weights-vs-direct precK diff 0.00e+00

| cell | eval frames | base rate | held-out frames/direction (median) | 1% of held-out | dirs with held-out < 1000 |
|---|---|---|---|---|---|
| v2|Y2F | 26400 | 0.1342 | 13200 | 132 | 0/12 |
| v2|B2F | 26400 | 0.1132 | 13200 | 132 | 0/12 |
| v2|groom_any | 26400 | 0.2236 | 13200 | 132 | 0/12 |
| v3|Y2F | 64200 | 0.1964 | 32100 | 321 | 0/12 |
| v3|B2F | 64200 | 0.1901 | 32100 | 321 | 0/12 |
| v3|groom_any | 64200 | 0.3298 | 32100 | 321 | 0/12 |
| v4|Y2F | 24000 | 0.3142 | 12000 | 120 | 0/12 |
| v4|B2F | 24000 | 0.3049 | 12000 | 120 | 0/12 |
| v4|groom_any | 24000 | 0.4883 | 12000 | 120 | 0/12 |
| v4_infested|Y2F | 12600 | 0.2213 | 6300 | 63 | 0/12 |
| v4_infested|B2F | 12600 | 0.1889 | 6300 | 63 | 0/12 |
| v4_infested|groom_any | 12600 | 0.3426 | 6300 | 63 | 0/12 |

### gate (mean over 12 cells; levels = seed median)

| arm | precision@100 | precision@1000 | top-1% precision |
|---|---|---|---|
| v1 | 0.881 | 0.736 | 0.876 |
| S | 0.768 | 0.690 | 0.758 |
| V | 0.741 | 0.676 | 0.730 |

| contrast | d precision@100 [95% CI pools/videos+seeds] | d precision@1000 [95% CI] |
|---|---|---|
| V - v1 | -0.104 [-0.145, -0.048] | -0.037 [-0.067, -0.008] |
| S - v1 | -0.080 [-0.118, -0.018] | -0.024 [-0.053, +0.007] |
| V - S | -0.024 [-0.057, +0.005] | -0.013 [-0.029, +0.004] |

### focal_single (mean over 4 cells; levels = seed median)

| arm | precision@100 | precision@1000 | top-1% precision |
|---|---|---|---|
| v1 | 0.230 | 0.470 | 0.273 |
| S | 0.418 | 0.559 | 0.434 |
| V | 0.444 | 0.566 | 0.452 |

| contrast | d precision@100 [95% CI pools/videos+seeds] | d precision@1000 [95% CI] |
|---|---|---|
| V - v1 | +0.183 [+0.070, +0.262] | +0.092 [+0.050, +0.127] |
| S - v1 | +0.134 [+0.033, +0.213] | +0.078 [+0.039, +0.114] |
| V - S | +0.049 [-0.009, +0.112] | +0.014 [-0.009, +0.038] |

### focal_double (mean over 2 cells; levels = seed median)

| arm | precision@100 | precision@1000 | top-1% precision |
|---|---|---|---|
| v1 | 0.807 | 0.587 | 0.764 |
| S | 0.622 | 0.510 | 0.592 |
| V | 0.603 | 0.526 | 0.602 |

| contrast | d precision@100 [95% CI pools/videos+seeds] | d precision@1000 [95% CI] |
|---|---|---|
| V - v1 | -0.149 [-0.253, -0.028] | -0.036 [-0.102, +0.039] |
| S - v1 | -0.133 [-0.237, -0.018] | -0.043 [-0.108, +0.025] |
| V - S | -0.017 [-0.116, +0.091] | +0.007 [-0.059, +0.072] |

### focal_or (mean over 4 cells; levels = seed median)

| arm | precision@100 | precision@1000 | top-1% precision |
|---|---|---|---|
| v1 | 0.944 | 0.885 | 0.941 |
| S | 0.893 | 0.856 | 0.883 |
| V | 0.880 | 0.854 | 0.876 |

| contrast | d precision@100 [95% CI pools/videos+seeds] | d precision@1000 [95% CI] |
|---|---|---|
| V - v1 | -0.052 [-0.085, -0.009] | -0.013 [-0.041, +0.015] |
| S - v1 | -0.046 [-0.083, -0.000] | -0.011 [-0.037, +0.014] |
| V - S | -0.006 [-0.039, +0.025] | -0.001 [-0.019, +0.014] |
