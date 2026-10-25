# Post-hoc additions (amendment 2026-10-25 ~03:00; see attr.py docstring)

## P1 appendage neurons by the mask2_test rule (share > 0.5, >= 20 SEL frames); attribution on TEST frames

| arm | neuron | SEL firing frames | appendage share | a | tail share | D_T/a | D_Tout/a | D_B/a | D_C/a | heat-map tail mass | argmax at border |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M0 | 100 | 191 | 0.666 | 36.19 | 0.877 | 0.019 | 0.011 | 0.009 | 0.002 | 1.000 | 0.63 |
| M0 | 1272 | 121 | 0.526 | 8.00 | 1.000 | 0.013 | 0.020 | 0.000 | 0.002 | 0.711 | 0.00 |
| M0 | 275 | 119 | 0.517 | 13.11 | 0.664 | 0.125 | 0.051 | 0.090 | 0.001 | 0.699 | 0.00 |
| M0 | 166 | 116 | 0.501 | 17.05 | 0.203 | 0.000 | 0.011 | 0.003 | 0.000 | 0.738 | 0.00 |
| M0 | 801 | 109 | 0.505 | 6.96 | 0.354 | 0.237 | 0.079 | 0.363 | 0.000 | 0.542 | 0.00 |
| N2 | 119 | 1258 | 0.767 | 27.32 | 0.898 | 0.509 | 0.458 | 0.055 | 0.000 | 0.772 | 0.00 |
| N2 | 57 | 1255 | 0.532 | 22.74 | 0.331 | 0.353 | 0.354 | 0.783 | 0.000 | 0.616 | 0.00 |
| N2 | 75 | 1166 | 0.833 | 23.00 | 0.808 | 0.381 | 0.379 | 0.085 | 0.001 | 0.868 | 0.00 |
| N2 | 56 | 1156 | 0.611 | 18.15 | 0.401 | 0.101 | 0.106 | 0.119 | 0.003 | 0.532 | 0.00 |
| N2 | 64 | 1155 | 0.903 | 14.97 | 0.836 | 0.492 | 0.497 | 0.107 | 0.003 | 0.970 | 0.00 |
| M0 | 88 (pre-reg. pick) | - | - | 27.07 | 0.308 | 0.383 | 0.260 | 0.879 | 0.000 | 0.008 | 0.00 |

| set | n neurons | median tail share | share with tail share >= 0.5 | median D_T/a | median D_Tout/a | median D_B/a | median D_C/a | median heat tail mass | N2: heat faithful share |
|---|---|---|---|---|---|---|---|---|---|
| M0 appendage neurons, all | 13 | 0.664 | 0.692 | 0.118 | 0.020 | 0.031 | 0.000 | 0.738 | - |
| M0 appendage neurons, argmax not at border | 12 | 0.659 | 0.667 | 0.121 | 0.022 | 0.036 | 0.000 | 0.725 | - |
| N2 appendage neurons, all | 204 | 0.803 | 0.848 | 0.383 | 0.371 | 0.053 | 0.001 | 1.000 | 0.583 |
| N2 appendage neurons, argmax not at border | 157 | 0.819 | 0.892 | 0.497 | 0.493 | 0.094 | 0.000 | 0.974 | 0.631 |

## P2 pair probe: orig vs occluded version of the same strict body token (AUROC, 5-fold by pool)

| pair | K1 = 1 (tail within 2 tokens) | K1 = 0 |
|---|---|---|
| orig vs tail occluded | 0.973 | 0.960 |
| orig vs control occlusion | 0.654 | 0.661 |
