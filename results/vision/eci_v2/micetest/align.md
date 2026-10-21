# micetest: behaviour alignment of v2_mice_c0_w2048_s7

R~ = cross-fitted normalised R-precision (best neuron on one half of the units, scored on the other, 10 random splits); AUROC = cf AUROC. v2_raw = frame max codes; v2_nes = the NES input (0.6 s smoothed, fnorm thresholded).

## v1 (54,000 frames, 6 units)

| model | set | nose_nose R~ / AUROC | nose_tail R~ / AUROC | nn_mutual R~ / AUROC | np_directional R~ / AUROC |
|---|---|---|---|---|---|
| v2_raw (W 2048) | all | 0.213 / 0.816 | 0.053 / 0.574 | 0.067 / 0.795 | 0.146 / 0.819 |
| v2_raw (W 2048) | consistent | 0.213 / 0.816 | 0.053 / 0.574 | 0.067 / 0.795 | 0.146 / 0.819 |
| v2_nes (W 2048) | all | 0.242 / 0.604 | 0.060 / 0.545 | 0.083 / 0.538 | 0.217 / 0.627 |
| v2_nes (W 2048) | consistent | 0.242 / 0.604 | 0.060 / 0.545 | 0.083 / 0.538 | 0.217 / 0.627 |
| v1_online (W 1024) | all | 0.051 / 0.667 | 0.044 / 0.645 | -0.002 / 0.761 | 0.028 / 0.601 |
| v1_online (W 1024) | consistent | 0.051 / 0.667 | 0.044 / 0.645 | -0.002 / 0.761 | 0.028 / 0.601 |

