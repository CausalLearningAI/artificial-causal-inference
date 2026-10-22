# ants: behaviour alignment of v2_ants_c0_w2048_s6

R~ = cross-fitted normalised R-precision (best neuron on one half of the units, scored on the other, 10 random splits); AUROC = cf AUROC. v2_raw = frame max codes; v2_nes = the NES input (0.6 s smoothed, fnorm thresholded).

## v2 (132,000 frames, 44 units)

| model | set | groom_any R~ / AUROC | groom_yellow R~ / AUROC | groom_blue R~ / AUROC |
|---|---|---|---|---|
| v2_raw (W 2048) | all | 0.629 / 0.882 | 0.508 / 0.883 | 0.389 / 0.859 |
| v2_nes (W 2048) | all | 0.075 / 0.550 | 0.053 / 0.528 | 0.159 / 0.586 |
| v2_smooth (W 2048) | all | 0.673 / 0.900 | 0.531 / 0.896 | 0.403 / 0.871 |
| v2_prob (W 2048) | all | 0.618 / 0.901 | 0.358 / 0.859 | 0.414 / 0.878 |
| v2_map (W 2048) | all | 0.431 / 0.833 | 0.249 / 0.794 | 0.355 / 0.810 |
| v1_online (W 1024) | all | 0.632 / 0.901 | 0.432 / 0.861 | 0.423 / 0.883 |
| r4_d2 (W 512) | all | 0.724 / 0.949 | 0.474 / 0.911 | 0.418 / 0.901 |

## v3 (321,000 frames, 107 units)

| model | set | groom_any R~ / AUROC | groom_yellow R~ / AUROC | groom_blue R~ / AUROC | on_lid_any R~ / AUROC | on_lid_yellow R~ / AUROC | on_lid_blue R~ / AUROC | on_lid_focal R~ / AUROC |
|---|---|---|---|---|---|---|---|---|
| v2_raw (W 2048) | all | 0.576 / 0.874 | 0.387 / 0.819 | 0.382 / 0.837 | 0.854 / 0.952 | 0.547 / 0.942 | 0.009 / 0.676 | nan / nan |
| v2_nes (W 2048) | all | 0.204 / 0.635 | 0.237 / 0.631 | 0.256 / 0.641 | 0.863 / 0.950 | 0.524 / 0.941 | -0.001 / 0.720 | nan / nan |
| v2_smooth (W 2048) | all | 0.594 / 0.885 | 0.400 / 0.828 | 0.394 / 0.846 | 0.863 / 0.956 | 0.551 / 0.944 | 0.009 / 0.722 | nan / nan |
| v2_prob (W 2048) | all | 0.594 / 0.885 | 0.404 / 0.835 | 0.399 / 0.846 | 0.617 / 0.917 | 0.506 / 0.907 | 0.011 / 0.686 | nan / nan |
| v2_map (W 2048) | all | 0.328 / 0.785 | 0.255 / 0.740 | 0.345 / 0.747 | 0.570 / 0.920 | 0.487 / 0.912 | 0.009 / 0.675 | nan / nan |
| v1_online (W 1024) | all | 0.600 / 0.888 | 0.407 / 0.844 | 0.436 / 0.856 | 0.813 / 0.904 | 0.587 / 0.917 | 0.004 / 0.648 | nan / nan |

