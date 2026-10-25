# Behaviour-neuron suite (label-free) - per representation and domain

Capacity = number of DISTINCT behaviour neurons (pass all six checks, de-duplicated at Pearson r >= 0.5 or firing Jaccard >= 0.5). Mean ± sd over seeds (v1 5, v2dep 3, V2F 5). Welch 95% CI vs v1.

| domain | arm (n seeds) | W | eligible | behaviour neurons | capacity full | capacity @256 | N_eff | share | capacity full - v1 (Welch) | @256 - v1 (Welch) |
|---|---|---|---|---|---|---|---|---|---|---|
| mice | v1 (5) | 1024 | 1021.8 ± 1.5 | 216.8 ± 5.8 | 132.8 ± 4.1 | 13.8 ± 2.9 | 33.6 ± 0.8 | 0.212 |  |  |
| mice | v2dep (3) | 2048 | 2020.0 ± 3.5 | 281.3 ± 3.8 | 162.0 ± 2.0 | 24.7 ± 4.9 | 71.2 ± 3.2 | 0.137 | +29.2 [+23.9, +34.5] | +10.9 [+0.6, +21.1] |
| mice | V2F_M0 (5) | 2048 | 2015.4 ± 6.9 | 276.0 ± 13.2 | 151.8 ± 9.7 | 21.6 ± 3.4 | 69.1 ± 2.6 | 0.135 | +19.0 [+7.1, +30.9] | +7.8 [+3.2, +12.4] |
| mice | V2F_N2 (5) | 2048 | 2026.6 ± 1.5 | 458.2 ± 10.3 | 258.6 ± 3.4 | 16.0 ± 2.0 | 62.9 ± 1.0 | 0.224 | +125.8 [+120.3, +131.3] | +2.2 [-1.6, +6.0] |
| ants | v1 (5) | 1024 | 829.4 ± 17.0 | 60.0 ± 3.0 | 31.4 ± 3.0 | 9.6 ± 2.2 | 31.6 ± 0.7 | 0.059 |  |  |
| ants | v2dep (3) | 2048 | 1971.7 ± 3.8 | 175.3 ± 6.4 | 109.7 ± 2.3 | 29.7 ± 5.5 | 72.3 ± 1.7 | 0.086 | +78.3 [+73.5, +83.0] | +20.1 [+7.8, +32.4] |
| ants | V2F_M0 (5) | 2048 | 1962.8 ± 5.4 | 170.2 ± 6.3 | 108.6 ± 10.9 | 38.2 ± 7.7 | 70.3 ± 1.8 | 0.083 | +77.2 [+63.8, +90.6] | +28.6 [+19.2, +38.0] |
| ants | V2F_N2 (5) | 2048 | 1996.8 ± 6.5 | 169.4 ± 6.5 | 101.0 ± 6.0 | 31.4 ± 5.3 | 65.9 ± 2.4 | 0.083 | +69.6 [+62.2, +77.0] | +21.8 [+15.3, +28.3] |
| frogs | v2dep (3) | 2048 | 1942.7 ± 6.1 | 332.7 ± 5.1 | 233.3 ± 10.0 | 55.0 ± 6.0 | 69.6 ± 1.7 | 0.162 |  |  |

## Median score per check over eligible neurons (mean over seeds) and pass rate among eligible

| domain | arm | off-animal share (C1, <= 0.5) | eta2 between videos (C2, <= 0.5) | tau s (C3, 0.2-120) | conditions R2cv (C4, <= 0.5) | reproducibility r (C5, >= 0.5) | invariance min r (C6, >= 0.5) | pass rates C1-C6 |
|---|---|---|---|---|---|---|---|---|
| mice | v1 | 0.367 | 0.272 | 1.81 | -0.259 | 0.689 | 0.520 | 0.55 / 0.88 / 1.00 / 0.99 / 0.80 / 0.53 |
| mice | v2dep | 0.000 | 0.164 | 0.99 | -0.264 | 0.521 | 0.188 | 1.00 / 0.93 / 0.95 / 1.00 / 0.54 / 0.23 |
| mice | V2F_M0 | 0.000 | 0.163 | 0.97 | -0.267 | 0.511 | 0.184 | 1.00 / 0.93 / 0.95 / 1.00 / 0.52 / 0.23 |
| mice | V2F_N2 | 0.000 | 0.198 | 1.18 | -0.238 | 0.581 | 0.317 | 1.00 / 0.92 / 0.97 / 1.00 / 0.65 / 0.34 |
| ants | v1 | 0.619 | 0.115 | 1.29 | -0.177 | 0.480 | 0.394 | 0.44 / 0.98 / 0.94 / 0.99 / 0.47 / 0.34 |
| ants | v2dep | 0.000 | 0.069 | 0.77 | -0.242 | 0.399 | 0.113 | 1.00 / 0.98 / 0.90 / 1.00 / 0.32 / 0.16 |
| ants | V2F_M0 | 0.000 | 0.064 | 0.71 | -0.237 | 0.390 | 0.108 | 1.00 / 0.97 / 0.88 / 1.00 / 0.32 / 0.16 |
| ants | V2F_N2 | 0.000 | 0.069 | 0.69 | -0.222 | 0.400 | 0.110 | 1.00 / 0.98 / 0.86 / 1.00 / 0.33 / 0.16 |
| frogs | v2dep | 0.000 | 0.158 | 1.73 | -0.017 | 0.639 | 0.271 | 1.00 / 0.91 / 0.90 / 0.98 / 0.73 / 0.28 |

## NES prefilter: candidates (behaviour neurons inside the prefix), mean over seeds; deployment seed in brackets; Bonferroni per-test alpha = 0.05 / m

| domain | arm | 128 | 256 | 512 | W | alpha @256 today -> filtered (deployment seed) |
|---|---|---|---|---|---|---|
| mice | v1 | 33.6 [34] | 77.6 [80] | 146.2 [144] | 216.8 [224] | 1.95e-04 -> 6.25e-04 |
| mice | v2dep | 61.0 [65] | 93.3 [102] | 161.3 [170] | 281.3 [284] | 1.95e-04 -> 4.90e-04 |
| mice | V2F_M0 | 64.8 [65] | 114.2 [110] | 165.2 [160] | 276.0 [279] | 1.95e-04 -> 4.55e-04 |
| mice | V2F_N2 | 58.2 [55] | 124.8 [114] | 224.2 [218] | 458.2 [450] | 1.95e-04 -> 4.39e-04 |
| ants | v1 | 27.6 [29] | 37.4 [37] | 50.8 [53] | 60.0 [61] | 1.95e-04 -> 1.35e-03 |
| ants | v2dep | 63.7 [60] | 83.3 [80] | 123.7 [123] | 175.3 [178] | 1.95e-04 -> 6.25e-04 |
| ants | V2F_M0 | 63.6 [63] | 93.4 [94] | 121.0 [116] | 170.2 [178] | 1.95e-04 -> 5.32e-04 |
| ants | V2F_N2 | 61.8 [58] | 89.4 [95] | 116.6 [120] | 169.4 [173] | 1.95e-04 -> 5.26e-04 |
| frogs | v2dep | 36.3 [38] | 80.7 [83] | 171.3 [171] | 332.7 [327] | 1.95e-04 -> 6.02e-04 |

## Validation (labels used only here)

- mice: labelled behaviour neurons passing all checks 0.64 (full) / 0.72 (@256); eligible 1.00; per check c1 1.00, c2 1.00, c3 1.00, c4 1.00, c5 0.69, c6 0.78; enrichment AUROC (alignment of behaviour vs other eligible neurons) 0.517 [0.496, 0.538]; validated: False
- ants: labelled behaviour neurons passing all checks 0.85 (full) / 0.74 (@256); eligible 1.00; per check c1 0.93, c2 1.00, c3 1.00, c4 1.00, c5 1.00, c6 0.93; enrichment AUROC (alignment of behaviour vs other eligible neurons) 0.580 [0.568, 0.593]; validated: True
- SUITE VALIDATED (pre-registered rule): False

| domain | arm | enrichment AUROC | Spearman(#checks, alignment) | best R~ among behaviour neurons | best R~ overall |
|---|---|---|---|---|---|
| mice | v1 | 0.493 | 0.021 | 0.063 | 0.073 |
| mice | v2dep | 0.488 | 0.018 | 0.134 | 0.153 |
| mice | V2F_M0 | 0.496 | 0.021 | 0.150 | 0.150 |
| mice | V2F_N2 | 0.580 | 0.147 | 0.079 | 0.117 |
| ants | v1 | 0.589 | 0.079 | 0.702 | 0.788 |
| ants | v2dep | 0.576 | 0.078 | 0.863 | 0.863 |
| ants | V2F_M0 | 0.594 | 0.100 | 0.852 | 0.852 |
| ants | V2F_N2 | 0.560 | 0.069 | 0.732 | 0.860 |
