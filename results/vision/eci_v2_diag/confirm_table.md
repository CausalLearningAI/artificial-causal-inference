| criterion | cell | prefix | v1 | R_comb mean ± sd (n) | R_comb - v1 (95% t-CI) | pass |
|---|---|---|---|---|---|---|
| A distinct | nn vs nt best neuron | full | - | 1.00 of seeds | deploy s402: {'nose_nose': 290, 'nose_tail': 77} distinct=True | True |
| A distinct | nn vs nt best neuron | 256 | - | 0.90 of seeds | deploy s402: {'nose_nose': 14, 'nose_tail': 77} distinct=True | True |
| A distinct | nn vs nt best neuron | 128 | - | 0.95 of seeds | deploy s402: {'nose_nose': 14, 'nose_tail': 77} distinct=True | (reported) |
| B R_atlas | mice nose_nose | full | 0.0548 | 0.1099 ± 0.0164 (20) | +0.0551 [+0.0474, +0.0628] | True |
| B R_atlas | mice nose_nose | 256 | 0.0173 | 0.0562 ± 0.0472 (20) | +0.0390 [+0.0169, +0.0610] | True |
| B R_atlas | mice nose_tail | full | 0.1029 | 0.1005 ± 0.0135 (20) | -0.0024 [-0.0087, +0.0039] | True |
| B R_atlas | mice nose_tail | 256 | -0.0047 | 0.0884 ± 0.0253 (20) | +0.0930 [+0.0812, +0.1049] | True |
| B R_atlas | ants v2 groom_any | full | 0.6373 | 0.5919 ± 0.0505 (33) | -0.0454 [-0.0633, -0.0275] | False |
| B R_atlas | ants v2 groom_any | 256 | 0.6373 | 0.5919 ± 0.0505 (33) | -0.0454 [-0.0633, -0.0275] | False |
| B R_atlas | ants v3 groom_any | full | 0.6004 | 0.5649 ± 0.0234 (33) | -0.0355 [-0.0438, -0.0272] | False |
| B R_atlas | ants v3 groom_any | 256 | 0.6004 | 0.5649 ± 0.0234 (33) | -0.0355 [-0.0438, -0.0272] | False |
| B R_atlas | ants v3 on_lid_any | full | 0.7726 | 0.8022 ± 0.0118 (33) | +0.0296 [+0.0254, +0.0338] | True |
| B R_atlas | ants v3 on_lid_any | 256 | 0.6373 | 0.4935 ± 0.3676 (33) | -0.1438 [-0.2742, -0.0135] | False |
| B R_atlas | mice nn_mutual | full | 0.0138 | 0.0687 ± 0.0283 (20) | +0.0549 [+0.0416, +0.0682] | (reported: True) |
| B R_atlas | mice nn_mutual | 256 | 0.0091 | 0.0262 ± 0.0392 (20) | +0.0171 [-0.0013, +0.0355] | (reported: True) |
| B R_atlas | mice np_directional | full | 0.0080 | 0.0511 ± 0.0252 (20) | +0.0431 [+0.0313, +0.0548] | (reported: True) |
| B R_atlas | mice np_directional | 256 | 0.0036 | 0.0364 ± 0.0326 (20) | +0.0328 [+0.0175, +0.0480] | (reported: True) |

A PASS: True
B PASS (5 decision cells x full/256): False
background share <= 0.01 (seeds 400-404): True {"mice": {"n_seeds": 5, "share_background_mean": 0.0, "n_background": [0, 0, 0, 0, 0], "n_app_mean": 4.8, "n_odor_mean": 3.6}, "ants": {"n_seeds": 5, "share_background_mean": 0.0, "n_background": [0, 0, 0, 0, 0], "n_app_mean": 1.2, "n_odor_mean": 0.0}}