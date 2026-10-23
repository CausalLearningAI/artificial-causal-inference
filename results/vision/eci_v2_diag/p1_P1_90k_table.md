| criterion | cell | prefix | v1 | P1_90k mean ± sd (n) | P1_90k - v1 (95% t-CI) | pass | P1_90k - M0 (Welch) |
|---|---|---|---|---|---|---|---|
| A distinct | nn vs nt best neuron | full | - | 1.00 of seeds | deploy s500: {'nose_nose': 439, 'nose_tail': 271} distinct=True | False | |
| A distinct | nn vs nt best neuron | 256 | - | 0.75 of seeds | deploy s500: {'nose_nose': 79, 'nose_tail': 23} distinct=True | False | |
| A distinct | nn vs nt best neuron | 128 | - | 1.00 of seeds | deploy s500: {'nose_nose': 79, 'nose_tail': 23} distinct=True | (reported) | |
| B R_atlas | mice nose_nose | full | 0.0548 | 0.1652 ± 0.0154 (20) | +0.1104 [+0.1032, +0.1176] | True | +0.0209 [+0.0112, +0.0305] |
| B R_atlas | mice nose_nose | 256 | 0.0173 | 0.0605 ± 0.0727 (20) | +0.0433 [+0.0093, +0.0773] | True | +0.0055 [-0.0359, +0.0469] |
| B R_atlas | mice nose_tail | full | 0.1029 | 0.0871 ± 0.0115 (20) | -0.0158 [-0.0211, -0.0104] | False | -0.0085 [-0.0199, +0.0029] |
| B R_atlas | mice nose_tail | 256 | -0.0047 | 0.0319 ± 0.0294 (20) | +0.0366 [+0.0228, +0.0504] | True | -0.0123 [-0.0340, +0.0094] |
| B R_atlas | ants v2 groom_any | full | 0.6373 | 0.6125 ± 0.0482 (33) | -0.0248 [-0.0419, -0.0077] | False | -0.0106 [-0.0319, +0.0108] |
| B R_atlas | ants v2 groom_any | 256 | 0.6373 | 0.6125 ± 0.0482 (33) | -0.0248 [-0.0419, -0.0077] | False | -0.0106 [-0.0319, +0.0108] |
| B R_atlas | ants v3 groom_any | full | 0.6004 | 0.5654 ± 0.0122 (33) | -0.0351 [-0.0394, -0.0307] | False | -0.0024 [-0.0104, +0.0057] |
| B R_atlas | ants v3 groom_any | 256 | 0.6004 | 0.5654 ± 0.0122 (33) | -0.0351 [-0.0394, -0.0307] | False | -0.0024 [-0.0104, +0.0057] |
| B R_atlas | ants v3 on_lid_any | full | 0.7726 | 0.8120 ± 0.0103 (33) | +0.0394 [+0.0357, +0.0430] | True | -0.0019 [-0.0092, +0.0055] |
| B R_atlas | ants v3 on_lid_any | 256 | 0.6373 | 0.4670 ± 0.3820 (33) | -0.1704 [-0.3058, -0.0349] | False | +0.1201 [-0.0672, +0.3073] |
| B R_atlas | mice nn_mutual | full | 0.0138 | 0.0769 ± 0.0174 (20) | +0.0631 [+0.0549, +0.0712] | (reported: True) | +0.0136 [+0.0004, +0.0268] |
| B R_atlas | mice nn_mutual | 256 | 0.0091 | 0.0247 ± 0.0395 (20) | +0.0157 [-0.0028, +0.0341] | (reported: True) | +0.0035 [-0.0190, +0.0259] |
| B R_atlas | mice np_directional | full | 0.0080 | 0.1237 ± 0.0167 (20) | +0.1157 [+0.1079, +0.1235] | (reported: True) | +0.0164 [+0.0055, +0.0273] |
| B R_atlas | mice np_directional | 256 | 0.0036 | 0.0429 ± 0.0546 (20) | +0.0392 [+0.0137, +0.0648] | (reported: True) | +0.0064 [-0.0266, +0.0395] |

A PASS: False
B PASS (5 decision cells x full/256): False
background share <= 0.01: True {"mice": {"n_seeds": 5, "share_background_mean": 0.0, "n_background": [0, 0, 0, 0, 0], "n_app_mean": 19.6, "n_odor_mean": 10.8}}