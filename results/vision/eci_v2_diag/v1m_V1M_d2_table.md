| criterion | cell | prefix | v1 | V1M_d2 mean ± sd (n) | V1M_d2 - v1 (95% t-CI) | pass | V1M_d2 - M0 (Welch) |
|---|---|---|---|---|---|---|---|
| A distinct | nn vs nt best neuron | full | - | 0.65 of seeds | deploy s600: {'nose_nose': 1666, 'nose_tail': 1666} distinct=False | False | |
| A distinct | nn vs nt best neuron | 256 | - | 1.00 of seeds | deploy s600: {'nose_nose': 210, 'nose_tail': 24} distinct=True | False | |
| A distinct | nn vs nt best neuron | 128 | - | 0.80 of seeds | deploy s600: {'nose_nose': 94, 'nose_tail': 24} distinct=True | (reported) | |
| B R_atlas | mice nose_nose | full | 0.0548 | 0.0678 ± 0.0591 (20) | +0.0130 [-0.0146, +0.0407] | False | -0.0765 [-0.1048, -0.0482] |
| B R_atlas | mice nose_nose | 256 | 0.0173 | 0.0014 ± 0.0065 (20) | -0.0158 [-0.0189, -0.0128] | False | -0.0536 [-0.0796, -0.0277] |
| B R_atlas | mice nose_tail | full | 0.1029 | 0.0723 ± 0.0400 (20) | -0.0306 [-0.0494, -0.0119] | False | -0.0233 [-0.0442, -0.0024] |
| B R_atlas | mice nose_tail | 256 | -0.0047 | 0.0060 ± 0.0115 (20) | +0.0107 [+0.0053, +0.0161] | True | -0.0382 [-0.0565, -0.0200] |
| B R_atlas | ants v2 groom_any | full | 0.6373 | 0.5413 ± 0.0774 (33) | -0.0960 [-0.1235, -0.0686] | False | -0.0818 [-0.1119, -0.0516] |
| B R_atlas | ants v2 groom_any | 256 | 0.6373 | 0.4940 ± 0.1184 (33) | -0.1433 [-0.1853, -0.1013] | False | -0.1290 [-0.1728, -0.0853] |
| B R_atlas | ants v3 groom_any | full | 0.6004 | 0.5211 ± 0.0463 (33) | -0.0793 [-0.0957, -0.0629] | False | -0.0466 [-0.0642, -0.0290] |
| B R_atlas | ants v3 groom_any | 256 | 0.6004 | 0.4698 ± 0.0877 (33) | -0.1307 [-0.1618, -0.0996] | False | -0.0980 [-0.1297, -0.0662] |
| B R_atlas | ants v3 on_lid_any | full | 0.7726 | 0.8153 ± 0.0060 (33) | +0.0427 [+0.0406, +0.0448] | True | +0.0015 [-0.0053, +0.0083] |
| B R_atlas | ants v3 on_lid_any | 256 | 0.6373 | 0.4664 ± 0.3514 (33) | -0.1709 [-0.2955, -0.0463] | False | +0.1195 [-0.0604, +0.2994] |
| B R_atlas | mice nn_mutual | full | 0.0138 | 0.0190 ± 0.0337 (20) | +0.0052 [-0.0106, +0.0209] | (reported: False) | -0.0443 [-0.0629, -0.0257] |
| B R_atlas | mice nn_mutual | 256 | 0.0091 | 0.0049 ± 0.0081 (20) | -0.0042 [-0.0080, -0.0004] | (reported: True) | -0.0164 [-0.0307, -0.0021] |
| B R_atlas | mice np_directional | full | 0.0080 | 0.0309 ± 0.0334 (20) | +0.0229 [+0.0073, +0.0386] | (reported: True) | -0.0764 [-0.0936, -0.0591] |
| B R_atlas | mice np_directional | 256 | 0.0036 | 0.0010 ± 0.0064 (20) | -0.0027 [-0.0057, +0.0003] | (reported: True) | -0.0355 [-0.0583, -0.0127] |

A PASS: False
B PASS (5 decision cells x full/256): False
background share <= 0.01: False {"mice": {"n_seeds": 5, "share_background_mean": 0.4999029188033982, "n_background": [996, 1047, 1035, 1007, 1027], "n_app_mean": 327.4, "n_odor_mean": 39.0}, "ants": {"n_seeds": 5, "share_background_mean": 0.6463130254561099, "n_background": [1297, 1333, 1345, 1316, 1324], "n_app_mean": 214.4, "n_odor_mean": 0.0}}