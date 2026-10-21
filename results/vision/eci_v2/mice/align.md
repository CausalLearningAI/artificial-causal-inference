# mice: behaviour alignment of v2_mice_c0_w2048_s7

R~ = cross-fitted normalised R-precision (best neuron on one half of the units, scored on the other, 10 random splits); AUROC = cf AUROC. v2_raw = frame max codes; v2_nes = the NES input (0.6 s smoothed, fnorm thresholded).

## v1 (864,000 frames, 24 units)

| model | set | nose_nose R~ / AUROC | nose_tail R~ / AUROC | nn_mutual R~ / AUROC | np_directional R~ / AUROC |
|---|---|---|---|---|---|
| v2_raw (W 2048) | all | 0.174 / 0.784 | 0.073 / 0.733 | 0.102 / 0.759 | 0.127 / 0.793 |
| v2_raw (W 2048) | consistent | 0.186 / 0.781 | 0.067 / 0.720 | 0.110 / 0.754 | 0.138 / 0.793 |
| v2_nes (W 2048) | all | 0.093 / 0.545 | 0.072 / 0.573 | 0.088 / 0.512 | 0.095 / 0.538 |
| v2_nes (W 2048) | consistent | 0.093 / 0.535 | 0.064 / 0.567 | 0.090 / 0.511 | 0.095 / 0.533 |
| v2_smooth (W 2048) | all | 0.202 / 0.812 | 0.078 / 0.761 | 0.105 / 0.786 | 0.158 / 0.822 |
| v2_smooth (W 2048) | consistent | 0.214 / 0.809 | 0.073 / 0.746 | 0.116 / 0.780 | 0.171 / 0.821 |
| v2_prob (W 2048) | all | 0.090 / 0.702 | 0.038 / 0.662 | 0.002 / 0.685 | 0.089 / 0.709 |
| v2_prob (W 2048) | consistent | 0.083 / 0.701 | 0.030 / 0.644 | 0.002 / 0.644 | 0.083 / 0.709 |
| v1_online (W 1024) | all | 0.070 / 0.687 | 0.108 / 0.765 | 0.045 / 0.723 | 0.034 / 0.666 |
| v1_online (W 1024) | consistent | 0.074 / 0.690 | 0.105 / 0.748 | 0.049 / 0.725 | 0.036 / 0.668 |
| r4_d2 (W 512) | all | 0.038 / 0.651 | 0.019 / 0.676 | 0.018 / 0.709 | 0.035 / 0.655 |
| r4_d2 (W 512) | consistent | 0.034 / 0.652 | 0.017 / 0.708 | 0.021 / 0.718 | 0.030 / 0.655 |

