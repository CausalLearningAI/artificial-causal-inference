## mice

| arm | gate PASS_final P(on) (median [range] of eligible cells) | gate cont | mean AUROC P(on) | mean AUROC cont | mean top-1% prec P(on) | mean nAP P(on) / cont | mean F1max P(on) / cont | prox strat AUROC (ants) | dyn share @256 | FVE |
|---|---|---|---|---|---|---|---|---|---|---|
| v1 (11 seeds, mean) | - | - | 0.682 | - | 0.060 | 0.016 | - | - | 0.510 | - |
| v2.0 PASS B (old store, N2) | 4 / 4 | 4 / 4 | 0.778 | 0.771 | 0.082 | 0.042 / 0.095 | - | - | 0.554 (5 fps seeds) | - |
| S (3 seeds) | [4, 4, 4] | [4, 4, 4] | 0.737 [0.724, 0.740] | 0.741 [0.703, 0.746] | 0.095 [0.092, 0.106] | 0.035 [0.031, 0.036] / 0.038 [0.031, 0.051] | 0.095 [0.087, 0.106] / 0.106 [0.096, 0.123] | - | 0.492 [0.473, 0.500] | 0.874 [0.873, 0.874] |
| A (3 seeds) | [2, 2, 3] | [3, 2, 3] | 0.707 [0.691, 0.713] | 0.725 [0.698, 0.730] | 0.053 [0.048, 0.057] | 0.017 [0.013, 0.023] / 0.014 [0.011, 0.025] | 0.069 [0.060, 0.089] / 0.066 [0.062, 0.082] | - | 0.785 [0.781, 0.801] | 0.838 [0.838, 0.838] |
| B (3 seeds) | [4, 4, 4] | [4, 3, 4] | 0.786 [0.780, 0.791] | 0.753 [0.748, 0.755] | 0.116 [0.110, 0.130] | 0.040 [0.038, 0.045] / 0.051 [0.049, 0.061] | 0.108 [0.106, 0.129] / 0.132 [0.123, 0.145] | - | 0.773 [0.773, 0.788] | 0.828 [0.828, 0.828] |
| V (3 seeds) | [4, 4, 4] | [4, 4, 4] | 0.812 [0.782, 0.816] | 0.801 [0.761, 0.809] | 0.137 [0.112, 0.160] | 0.054 [0.043, 0.066] / 0.077 [0.060, 0.090] | 0.138 [0.106, 0.141] / 0.164 [0.129, 0.171] | - | 0.613 [0.609, 0.629] | 0.851 [0.850, 0.851] |

| cell | v1 AUROC | v1 top-1% | S PASS_final P(on) (seeds) / AUROC median | A PASS_final P(on) (seeds) / AUROC median | B PASS_final P(on) (seeds) / AUROC median | V PASS_final P(on) (seeds) / AUROC median |
|---|---|---|---|---|---|---|
| v1|nn | 0.704 | 0.043 | PPP / 0.764 | FPP / 0.696 | PPP / 0.844 | PPP / 0.859 |
| v1|nose_nose | 0.670 | 0.097 | PPP / 0.745 | FFP / 0.694 | PPP / 0.788 | PPP / 0.826 |
| v1|np | 0.661 | 0.055 | PPP / 0.712 | PFF / 0.661 | PPP / 0.722 | PPP / 0.802 |
| v1|nt | 0.693 | 0.046 | PPP / 0.715 | PPP / 0.748 | PPP / 0.782 | PPP / 0.769 |

Gate = AMENDMENT 5 rule of results/vision/eci_v2_0B/gate/gate.md vs v1 (11 seeds); P = PASS_final per seed (1001 / 1002 / 1003).
Gate thresholds (top-1% precision >= max(min(3 base, base + 0.5 (1 - base)), v1 - 0.05)): nn 0.021, nt 0.025, np 0.043, nose_nose 0.064.
Best neuron (most-picked AUROC neuron, P(on)) nn / nt per seed: S 250/250, 57/57, 917/917 (never distinct); A 200/1033, 1181/1762, 445/445;
B 791/791, 1414/553, 874/874; V 222/222, 436/270, 1472/1802; v2.0 PASS B 803/1402.
Prep (mice, mask v5 store): 16.0M train M_t tokens (of 18.4M sampled), ~195 M_t tokens / frame; U1 / U2 keep 86.3% / 86.1% of the
change / deviation variance; block scales s1 3.53, s2 2.87; median ||dx|| / ||x|| (arm V) = 0.236.
Capacity: not computed (needs mask-v5 tokens of transformed frames; stated in the pre-registration).
Mice-only pick by the pre-registered rule: S, B, V tie at 4/4 cells (median); highest median mean AUROC (P(on)) = V 0.812
(B 0.786, S 0.737); V passes as many cells as S and has a higher dynamic share @256 (0.613 vs 0.492) -> V.
The combined (mice + ants) pick follows when ants scoring completes.
