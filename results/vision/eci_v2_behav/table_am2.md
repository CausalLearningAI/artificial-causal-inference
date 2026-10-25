# Behaviour-neuron suite (label-free) - per representation and domain

Capacity = number of DISTINCT behaviour neurons (pass all six checks, de-duplicated at Pearson r >= 0.5 or firing Jaccard >= 0.5). Mean ± sd over seeds (v1 5, v2dep 3, V2F 5). Welch 95% CI vs v1.

| domain | arm (n seeds) | W | eligible | behaviour neurons | capacity full | capacity @256 | N_eff | share | capacity full - v1 (Welch) | @256 - v1 (Welch) |
|---|---|---|---|---|---|---|---|---|---|---|
| mice | v1 (5) | 1024 | 1021.8 ± 1.5 | 199.2 ± 5.0 | 124.4 ± 3.1 | 13.4 ± 3.0 | 33.8 ± 0.9 | 0.195 |  |  |
| mice | v2dep (3) | 2048 | 2020.0 ± 3.5 | 192.3 ± 5.5 | 131.3 ± 4.7 | 19.3 ± 3.1 | 59.1 ± 2.6 | 0.094 | +6.9 [-2.7, +16.5] | +5.9 [-0.1, +11.9] |
| mice | V2F_M0 (5) | 2048 | 2015.4 ± 6.9 | 189.2 ± 9.0 | 123.8 ± 5.2 | 18.8 ± 2.5 | 56.3 ± 3.1 | 0.092 | -0.6 [-7.1, +5.9] | +5.4 [+1.4, +9.4] |
| mice | V2F_N2 (5) | 2048 | 2026.6 ± 1.5 | 323.2 ± 7.6 | 211.4 ± 7.0 | 11.8 ± 1.5 | 51.6 ± 1.1 | 0.158 | +87.0 [+78.5, +95.5] | -1.6 [-5.2, +2.0] |
| ants | v1 (5) | 1024 | 829.4 ± 17.0 | 42.4 ± 3.9 | 23.6 ± 2.4 | 8.2 ± 2.4 | 26.0 ± 1.4 | 0.041 |  |  |
| ants | v2dep (3) | 2048 | 1971.7 ± 3.8 | 122.7 ± 0.6 | 79.3 ± 2.1 | 23.7 ± 6.0 | 64.3 ± 0.9 | 0.060 | +55.7 [+51.6, +59.9] | +15.5 [+2.0, +28.9] |
| ants | V2F_M0 (5) | 2048 | 1962.8 ± 5.4 | 118.6 ± 3.7 | 78.0 ± 6.3 | 34.0 ± 6.2 | 62.7 ± 2.2 | 0.058 | +54.4 [+46.7, +62.1] | +25.8 [+18.2, +33.4] |
| ants | V2F_N2 (5) | 2048 | 1996.8 ± 6.5 | 108.4 ± 6.9 | 65.6 ± 7.0 | 27.6 ± 6.0 | 59.0 ± 3.1 | 0.053 | +42.0 [+33.5, +50.5] | +19.4 [+12.1, +26.7] |
| frogs | v2dep (3) | 2048 | 1942.7 ± 6.1 | 141.7 ± 8.3 | 122.3 ± 9.3 | 35.0 ± 2.6 | 42.5 ± 3.8 | 0.069 |  |  |

## Median score per check over eligible neurons (mean over seeds) and pass rate among eligible

| domain | arm | off-animal share (C1, <= 0.5) | eta2 between videos (C2, <= 0.5) | tau s (C3, 0.2-120) | conditions R2cv (C4, <= 0.5) | reproducibility r (C5, >= 0.5) | invariance min r (C6, >= 0.5) | pass rates C1-C6 |
|---|---|---|---|---|---|---|---|---|
| mice | v1 | 0.367 | 0.272 | 1.81 | -0.259 | 0.689 | 0.520 | 0.55 / 0.88 / 1.00 / 0.99 / 0.80 / 0.53 / 0.71 / 0.95 |
| mice | v2dep | 0.000 | 0.164 | 0.99 | -0.264 | 0.521 | 0.188 | 1.00 / 0.93 / 0.95 / 1.00 / 0.54 / 0.23 / 0.69 / 0.97 |
| mice | V2F_M0 | 0.000 | 0.163 | 0.97 | -0.267 | 0.511 | 0.184 | 1.00 / 0.93 / 0.95 / 1.00 / 0.52 / 0.23 / 0.69 / 0.97 |
| mice | V2F_N2 | 0.000 | 0.198 | 1.18 | -0.238 | 0.581 | 0.317 | 1.00 / 0.92 / 0.97 / 0.99 / 0.65 / 0.34 / 0.72 / 0.96 |
| ants | v1 | 0.619 | 0.115 | 1.29 | -0.177 | 0.480 | 0.394 | 0.44 / 0.98 / 0.94 / 0.99 / 0.47 / 0.34 / 0.75 / 0.96 |
| ants | v2dep | 0.000 | 0.069 | 0.77 | -0.242 | 0.399 | 0.113 | 1.00 / 0.98 / 0.90 / 1.00 / 0.32 / 0.16 / 0.75 / 0.99 |
| ants | V2F_M0 | 0.000 | 0.064 | 0.71 | -0.237 | 0.390 | 0.108 | 1.00 / 0.97 / 0.88 / 1.00 / 0.32 / 0.16 / 0.80 / 0.99 |
| ants | V2F_N2 | 0.000 | 0.069 | 0.69 | -0.222 | 0.400 | 0.110 | 1.00 / 0.98 / 0.86 / 1.00 / 0.33 / 0.16 / 0.81 / 0.98 |
| frogs | v2dep | 0.000 | 0.158 | 1.73 | -0.017 | 0.639 | 0.271 | 1.00 / 0.91 / 0.90 / 0.98 / 0.73 / 0.28 / 0.39 / 1.00 |

## Amendment 2 checks: median over eligible neurons (mean over seeds) and pass rate

| domain | arm | place TV (C7, <= 0.5) | on-share (C8, <= 0.5) | conditions R2cv, more videos (C4b, <= 0.5) | C4b videos | pass C4b / C7 / C8 |
|---|---|---|---|---|---|---|
| mice | v1 | 0.364 | 0.113 | -0.022 | 192 | 0.99 / 0.71 / 0.95 |
| mice | v2dep | 0.348 | 0.064 | -0.038 | 192 | 1.00 / 0.69 / 0.97 |
| mice | V2F_M0 | 0.346 | 0.066 | -0.035 | 192 | 1.00 / 0.69 / 0.97 |
| mice | V2F_N2 | 0.339 | 0.094 | -0.002 | 192 | 0.99 / 0.72 / 0.96 |
| ants | v1 | 0.291 | 0.065 | -0.077 | 175 | 0.99 / 0.75 / 0.96 |
| ants | v2dep | 0.293 | 0.024 | -0.145 | 175 | 1.00 / 0.75 / 0.99 |
| ants | V2F_M0 | 0.290 | 0.025 | -0.143 | 175 | 1.00 / 0.80 / 0.99 |
| ants | V2F_N2 | 0.279 | 0.031 | -0.130 | 175 | 1.00 / 0.81 / 0.98 |
| frogs | v2dep | 0.528 | 0.010 | -0.013 | 72 | 0.98 / 0.39 / 1.00 |

## NES prefilter: candidates (behaviour neurons inside the prefix), mean over seeds; deployment seed in brackets; Bonferroni per-test alpha = 0.05 / m

| domain | arm | 128 | 256 | 512 | W | alpha @256 today -> filtered (deployment seed) |
|---|---|---|---|---|---|---|
| mice | v1 | 24.8 [24] | 68.2 [69] | 133.6 [131] | 199.2 [203] | 1.95e-04 -> 7.25e-04 |
| mice | v2dep | 30.3 [31] | 54.0 [57] | 110.0 [114] | 192.3 [195] | 1.95e-04 -> 8.77e-04 |
| mice | V2F_M0 | 33.8 [29] | 75.4 [69] | 116.6 [109] | 189.2 [187] | 1.95e-04 -> 7.25e-04 |
| mice | V2F_N2 | 19.8 [19] | 75.4 [71] | 159.4 [164] | 323.2 [324] | 1.95e-04 -> 7.04e-04 |
| ants | v1 | 17.8 [20] | 26.6 [27] | 36.4 [40] | 42.4 [47] | 1.95e-04 -> 1.85e-03 |
| ants | v2dep | 50.3 [46] | 63.7 [59] | 92.7 [90] | 122.7 [123] | 1.95e-04 -> 8.47e-04 |
| ants | V2F_M0 | 49.4 [49] | 72.6 [73] | 90.6 [90] | 118.6 [125] | 1.95e-04 -> 6.85e-04 |
| ants | V2F_N2 | 44.8 [38] | 66.6 [67] | 82.0 [82] | 108.4 [113] | 1.95e-04 -> 7.46e-04 |
| frogs | v2dep | 20.0 [20] | 44.7 [50] | 93.0 [88] | 141.7 [139] | 1.95e-04 -> 1.00e-03 |

## Validation (labels used only here)

- mice: labelled behaviour neurons passing all checks 0.64 (full) / 0.64 (@256); eligible 1.00; per check c1 1.00, c2 1.00, c3 1.00, c4b 1.00, c5 0.69, c6 0.78, c7 1.00, c8 1.00; enrichment AUROC (alignment of behaviour vs other eligible neurons) 0.517 [0.495, 0.539]; validated: False
- ants: labelled behaviour neurons passing all checks 0.85 (full) / 0.72 (@256); eligible 1.00; per check c1 0.93, c2 1.00, c3 1.00, c4b 1.00, c5 1.00, c6 0.93, c7 1.00, c8 1.00; enrichment AUROC (alignment of behaviour vs other eligible neurons) 0.654 [0.638, 0.669]; validated: True
- SUITE VALIDATED (pre-registered rule): False

| domain | arm | enrichment AUROC | Spearman(#checks, alignment) | best R~ among behaviour neurons | best R~ overall |
|---|---|---|---|---|---|
| mice | v1 | 0.491 | 0.048 | 0.063 | 0.073 |
| mice | v2dep | 0.484 | 0.135 | 0.134 | 0.153 |
| mice | V2F_M0 | 0.496 | 0.153 | 0.150 | 0.150 |
| mice | V2F_N2 | 0.583 | 0.213 | 0.079 | 0.117 |
| ants | v1 | 0.639 | 0.133 | 0.702 | 0.788 |
| ants | v2dep | 0.661 | 0.153 | 0.863 | 0.863 |
| ants | V2F_M0 | 0.680 | 0.164 | 0.852 | 0.852 |
| ants | V2F_N2 | 0.638 | 0.127 | 0.732 | 0.860 |
