| criterion | cell | prefix | v1 | V2F_M0 mean ± sd (n) | V2F_M0 - v1 (95% t-CI) | pass | V2F_M0 - M0 (Welch) |
|---|---|---|---|---|---|---|---|
| A distinct | nn vs nt best neuron | full | - | 0.90 of seeds | deploy s901: {'nose_nose': 147, 'nose_tail': 152} distinct=False | False | |
| A distinct | nn vs nt best neuron | 256 | - | 0.65 of seeds | deploy s901: {'nose_nose': 147, 'nose_tail': 152} distinct=False | False | |
| A distinct | nn vs nt best neuron | 128 | - | 1.00 of seeds | deploy s901: {'nose_nose': 73, 'nose_tail': 51} distinct=True | (reported) | |
| B R_atlas | mice nose_nose | full | 0.0548 | 0.1523 ± 0.0172 (20) | +0.0975 [+0.0895, +0.1055] | True | +0.0080 [-0.0023, +0.0182] |
| B R_atlas | mice nose_nose | 256 | 0.0173 | 0.0754 ± 0.0636 (20) | +0.0582 [+0.0284, +0.0879] | True | +0.0203 [-0.0178, +0.0585] |
| B R_atlas | mice nose_tail | full | 0.1029 | 0.1041 ± 0.0223 (20) | +0.0012 [-0.0093, +0.0117] | True | +0.0085 [-0.0057, +0.0228] |
| B R_atlas | mice nose_tail | 256 | -0.0047 | 0.0683 ± 0.0328 (20) | +0.0730 [+0.0576, +0.0884] | True | +0.0241 [+0.0015, +0.0467] |
| B R_atlas | ants v2 groom_any | full | 0.6373 | 0.5930 ± 0.0477 (33) | -0.0443 [-0.0612, -0.0274] | False | -0.0300 [-0.0513, -0.0088] |
| B R_atlas | ants v2 groom_any | 256 | 0.6373 | 0.5930 ± 0.0477 (33) | -0.0443 [-0.0612, -0.0274] | False | -0.0300 [-0.0513, -0.0088] |
| B R_atlas | ants v3 groom_any | full | 0.6004 | 0.5636 ± 0.0151 (33) | -0.0369 [-0.0422, -0.0315] | False | -0.0042 [-0.0127, +0.0044] |
| B R_atlas | ants v3 groom_any | 256 | 0.6004 | 0.5636 ± 0.0151 (33) | -0.0369 [-0.0422, -0.0315] | False | -0.0042 [-0.0127, +0.0044] |
| B R_atlas | ants v3 on_lid_any | full | 0.7726 | 0.8110 ± 0.0131 (33) | +0.0384 [+0.0338, +0.0431] | True | -0.0028 [-0.0107, +0.0050] |
| B R_atlas | ants v3 on_lid_any | 256 | 0.6373 | 0.5887 ± 0.3245 (33) | -0.0486 [-0.1637, +0.0665] | False | +0.2418 [+0.0681, +0.4156] |
| B R_atlas | mice nn_mutual | full | 0.0138 | 0.0636 ± 0.0240 (20) | +0.0498 [+0.0386, +0.0610] | (reported: True) | +0.0003 [-0.0148, +0.0155] |
| B R_atlas | mice nn_mutual | 256 | 0.0091 | 0.0254 ± 0.0331 (20) | +0.0164 [+0.0009, +0.0319] | (reported: True) | +0.0042 [-0.0160, +0.0243] |
| B R_atlas | mice np_directional | full | 0.0080 | 0.1181 ± 0.0110 (20) | +0.1101 [+0.1050, +0.1153] | (reported: True) | +0.0108 [+0.0014, +0.0202] |
| B R_atlas | mice np_directional | 256 | 0.0036 | 0.0594 ± 0.0495 (20) | +0.0558 [+0.0326, +0.0789] | (reported: True) | +0.0230 [-0.0084, +0.0543] |

A PASS: False
B PASS (5 decision cells x full/256): False
background share <= 0.01: True {"mice": {"n_seeds": 5, "share_background_mean": 0.0, "n_background": [0, 0, 0, 0, 0], "n_app_mean": 22.6, "n_odor_mean": 13.8}, "ants": {"n_seeds": 5, "share_background_mean": 0.0, "n_background": [0, 0, 0, 0, 0], "n_app_mean": 18.4, "n_odor_mean": 0.0}}