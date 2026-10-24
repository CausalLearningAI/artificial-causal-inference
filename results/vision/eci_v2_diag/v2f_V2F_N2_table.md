| criterion | cell | prefix | v1 | V2F_N2 mean ± sd (n) | V2F_N2 - v1 (95% t-CI) | pass | V2F_N2 - M0 (Welch) |
|---|---|---|---|---|---|---|---|
| A distinct | nn vs nt best neuron | full | - | 0.90 of seeds | deploy s900: {'nose_nose': 624, 'nose_tail': 485} distinct=True | True | |
| A distinct | nn vs nt best neuron | 256 | - | 1.00 of seeds | deploy s900: {'nose_nose': 152, 'nose_tail': 51} distinct=True | True | |
| A distinct | nn vs nt best neuron | 128 | - | 1.00 of seeds | deploy s900: {'nose_nose': 112, 'nose_tail': 51} distinct=True | (reported) | |
| B R_atlas | mice nose_nose | full | 0.0548 | 0.1279 ± 0.0415 (20) | +0.0732 [+0.0538, +0.0926] | True | -0.0164 [-0.0367, +0.0040] |
| B R_atlas | mice nose_nose | 256 | 0.0173 | 0.0132 ± 0.0087 (20) | -0.0040 [-0.0081, +0.0001] | True | -0.0419 [-0.0679, -0.0158] |
| B R_atlas | mice nose_tail | full | 0.1029 | 0.0728 ± 0.0244 (20) | -0.0301 [-0.0415, -0.0186] | False | -0.0227 [-0.0377, -0.0078] |
| B R_atlas | mice nose_tail | 256 | -0.0047 | 0.0083 ± 0.0115 (20) | +0.0130 [+0.0076, +0.0183] | True | -0.0359 [-0.0542, -0.0177] |
| B R_atlas | ants v2 groom_any | full | 0.6373 | 0.5921 ± 0.0345 (33) | -0.0452 [-0.0575, -0.0330] | False | -0.0309 [-0.0488, -0.0131] |
| B R_atlas | ants v2 groom_any | 256 | 0.6373 | 0.5921 ± 0.0345 (33) | -0.0452 [-0.0575, -0.0330] | False | -0.0309 [-0.0488, -0.0131] |
| B R_atlas | ants v3 groom_any | full | 0.6004 | 0.5700 ± 0.0125 (33) | -0.0305 [-0.0349, -0.0261] | False | +0.0022 [-0.0058, +0.0103] |
| B R_atlas | ants v3 groom_any | 256 | 0.6004 | 0.5700 ± 0.0125 (33) | -0.0305 [-0.0349, -0.0261] | False | +0.0022 [-0.0058, +0.0103] |
| B R_atlas | ants v3 on_lid_any | full | 0.7726 | 0.8138 ± 0.0053 (33) | +0.0412 [+0.0394, +0.0431] | True | -0.0000 [-0.0068, +0.0067] |
| B R_atlas | ants v3 on_lid_any | 256 | 0.6373 | 0.5182 ± 0.3598 (33) | -0.1192 [-0.2468, +0.0084] | False | +0.1713 [-0.0106, +0.3531] |
| B R_atlas | mice nn_mutual | full | 0.0138 | 0.0470 ± 0.0488 (20) | +0.0331 [+0.0103, +0.0560] | (reported: True) | -0.0163 [-0.0411, +0.0084] |
| B R_atlas | mice nn_mutual | 256 | 0.0091 | -0.0008 ± 0.0049 (20) | -0.0099 [-0.0122, -0.0076] | (reported: False) | -0.0221 [-0.0361, -0.0080] |
| B R_atlas | mice np_directional | full | 0.0080 | 0.0892 ± 0.0342 (20) | +0.0812 [+0.0652, +0.0972] | (reported: True) | -0.0181 [-0.0357, -0.0006] |
| B R_atlas | mice np_directional | 256 | 0.0036 | 0.0084 ± 0.0099 (20) | +0.0047 [+0.0001, +0.0094] | (reported: True) | -0.0281 [-0.0511, -0.0050] |

A PASS: True
B PASS (5 decision cells x full/256): False
background share <= 0.01: True {"mice": {"n_seeds": 5, "share_background_mean": 0.0, "n_background": [0, 0, 0, 0, 0], "n_app_mean": 203.6, "n_odor_mean": 14.2}, "ants": {"n_seeds": 5, "share_background_mean": 0.0, "n_background": [0, 0, 0, 0, 0], "n_app_mean": 170.6, "n_odor_mean": 0.0}}