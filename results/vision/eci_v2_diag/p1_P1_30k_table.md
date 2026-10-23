| criterion | cell | prefix | v1 | P1_30k mean ± sd (n) | P1_30k - v1 (95% t-CI) | pass | P1_30k - M0 (Welch) |
|---|---|---|---|---|---|---|---|
| A distinct | nn vs nt best neuron | full | - | 0.95 of seeds | deploy s502: {'nose_nose': 317, 'nose_tail': 219} distinct=True | False | |
| A distinct | nn vs nt best neuron | 256 | - | 0.65 of seeds | deploy s502: {'nose_nose': 198, 'nose_tail': 219} distinct=True | False | |
| A distinct | nn vs nt best neuron | 128 | - | 0.95 of seeds | deploy s502: {'nose_nose': 50, 'nose_tail': 2} distinct=True | (reported) | |
| B R_atlas | mice nose_nose | full | 0.0548 | 0.1558 ± 0.0133 (20) | +0.1011 [+0.0948, +0.1073] | True | +0.0115 [+0.0025, +0.0205] |
| B R_atlas | mice nose_nose | 256 | 0.0173 | 0.0737 ± 0.0724 (20) | +0.0564 [+0.0225, +0.0903] | True | +0.0186 [-0.0227, +0.0599] |
| B R_atlas | mice nose_tail | full | 0.1029 | 0.0947 ± 0.0187 (20) | -0.0082 [-0.0170, +0.0005] | False | -0.0009 [-0.0141, +0.0122] |
| B R_atlas | mice nose_tail | 256 | -0.0047 | 0.0426 ± 0.0338 (20) | +0.0473 [+0.0315, +0.0631] | True | -0.0016 [-0.0245, +0.0213] |
| B R_atlas | ants v2 groom_any | full | 0.6373 | 0.6003 ± 0.0481 (33) | -0.0370 [-0.0540, -0.0199] | False | -0.0227 [-0.0440, -0.0014] |
| B R_atlas | ants v2 groom_any | 256 | 0.6373 | 0.6003 ± 0.0481 (33) | -0.0370 [-0.0540, -0.0199] | False | -0.0227 [-0.0440, -0.0014] |
| B R_atlas | ants v3 groom_any | full | 0.6004 | 0.5678 ± 0.0133 (33) | -0.0326 [-0.0373, -0.0279] | False | +0.0001 [-0.0081, +0.0083] |
| B R_atlas | ants v3 groom_any | 256 | 0.6004 | 0.5678 ± 0.0133 (33) | -0.0326 [-0.0373, -0.0279] | False | +0.0001 [-0.0081, +0.0083] |
| B R_atlas | ants v3 on_lid_any | full | 0.7726 | 0.8132 ± 0.0080 (33) | +0.0406 [+0.0377, +0.0434] | True | -0.0007 [-0.0077, +0.0063] |
| B R_atlas | ants v3 on_lid_any | 256 | 0.6373 | 0.2791 ± 0.3567 (33) | -0.3582 [-0.4847, -0.2317] | False | -0.0678 [-0.2489, +0.1134] |
| B R_atlas | mice nn_mutual | full | 0.0138 | 0.0654 ± 0.0199 (20) | +0.0516 [+0.0423, +0.0609] | (reported: True) | +0.0021 [-0.0118, +0.0160] |
| B R_atlas | mice nn_mutual | 256 | 0.0091 | 0.0322 ± 0.0348 (20) | +0.0231 [+0.0068, +0.0394] | (reported: True) | +0.0109 [-0.0098, +0.0317] |
| B R_atlas | mice np_directional | full | 0.0080 | 0.1186 ± 0.0122 (20) | +0.1107 [+0.1049, +0.1164] | (reported: True) | +0.0114 [+0.0017, +0.0210] |
| B R_atlas | mice np_directional | 256 | 0.0036 | 0.0547 ± 0.0571 (20) | +0.0511 [+0.0244, +0.0778] | (reported: True) | +0.0183 [-0.0156, +0.0522] |

A PASS: False
B PASS (5 decision cells x full/256): False
background share <= 0.01: True {"mice": {"n_seeds": 5, "share_background_mean": 0.0, "n_background": [0, 0, 0, 0, 0], "n_app_mean": 22.6, "n_odor_mean": 13.4}}