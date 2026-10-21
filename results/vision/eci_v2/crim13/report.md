# ECI v2 (frozen) on CRIM13 - validation check

Deployed SAE v2_crim13_c0_w2048_s6 (held-out FVE 0.8319). R~ = normalised R-precision (P - pi) / (1 - pi) of the R-precision-picked neuron, picked on one released half and scored on the other, mean of both directions; AUROC of the AUROC-picked neuron, same cross-fit. Weighted (inverse-probability) eval frames of the CRIM13 alignment plan. v2 cells: mean ± sd over the 3 SAE seeds (smoothed 0.6 s readout). Stored arms: per-fold SAEs, unsmoothed frame max, mean over their seeds.

| behaviour | base rate (train / test half, weighted) | v2 R~ | v2 R~ deployed | v2 R~ unsmoothed | v2 AUROC | B0 R~ | B2 R~ | C1 R~ | previous best R~ (arm) |
|---|---|---|---|---|---|---|---|---|---|
| approach | 0.0344 / 0.0337 | 0.119 ± 0.020 | 0.142 | 0.115 | 0.767 | 0.128 | 0.120 | 0.142 | 0.147 (B3) |
| attack | 0.0349 / 0.0321 | 0.141 ± 0.005 | 0.147 | 0.107 | 0.770 | 0.167 | 0.129 | 0.140 | 0.171 (B7) |
| copulation | 0.0401 / 0.0446 | 0.296 ± 0.091 | 0.355 | 0.243 | 0.776 | 0.253 | 0.294 | 0.207 | 0.297 (B7) |
| chase | 0.0070 / 0.0125 | 0.204 ± 0.007 | 0.199 | 0.162 | 0.881 | 0.209 | 0.174 | 0.166 | 0.209 (B0) |
| circle | 0.0036 / 0.0029 | 0.004 ± 0.005 | 0.008 | 0.005 | 0.674 | 0.029 | 0.057 | 0.030 | 0.093 (C4) |
| drink | 0.0040 / 0.0021 | 0.064 ± 0.014 | 0.065 | 0.018 | 0.695 | 0.052 | 0.071 | 0.030 | 0.100 (B4) |
| eat | 0.0190 / 0.0140 | 0.029 ± 0.014 | 0.025 | 0.006 | 0.637 | 0.010 | 0.006 | 0.016 | 0.025 (C4) |
| clean | 0.0757 / 0.0723 | 0.347 ± 0.004 | 0.346 | 0.280 | 0.802 | 0.321 | 0.303 | 0.296 | 0.344 (B7) |
| sniff | 0.1377 / 0.1514 | 0.365 ± 0.017 | 0.368 | 0.360 | 0.831 | 0.342 | 0.375 | 0.304 | 0.375 (B6) |
| up | 0.0393 / 0.0380 | 0.222 ± 0.006 | 0.229 | 0.211 | 0.667 | 0.227 | 0.278 | 0.230 | 0.297 (B7) |
| walk_away | 0.0343 / 0.0334 | 0.127 ± 0.017 | 0.128 | 0.101 | 0.726 | 0.085 | 0.087 | 0.092 | 0.104 (C3) |
| **mean over 11 behaviours** | | 0.174 ± 0.010 | 0.183 | 0.146 | | 0.166 | 0.172 | 0.150 | 0.178 (B7) |
| **mean over 7 pair behaviours** | | 0.179 ± 0.013 | 0.192 | 0.156 | | 0.173 | 0.177 | 0.154 | 0.178 (B7) |

## SAE health

| seed | held-out FVE | L0 | dead at end of training | neurons active on >= 1 eval frame | neurons with weighted rate >= 1% |
|---|---|---|---|---|---|
| 6 (deployed) | 0.8319 | 16.11 | 0 | 2048 | 1862 |
| 7 | 0.8309 | 16.11 | 0 | 2048 | 1824 |
| 8 | 0.8304 | 16.09 | 0 | 2048 | 1835 |

## tight mask (eval frames, released track points; never used to build the mask)

kept tokens per eval frame 45.4; eval frames with 0 kept 162; own-cell recall: resident 0.933, BC (pale) intruder 0.045, C57 (dark) intruder 0.811. Old dilated dark mask (all plan frames): resident 0.976, BC 0.441, C57 0.879.
