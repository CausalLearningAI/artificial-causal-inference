# Step 0 diagnosis (in-sample, all 24 pools, stored SAEs, no retraining)

Per-neuron measures on all 86,400 mice eval frames (readout: V = HMM P(on), v1 = raw code). Discrimination AUROC = nose-tail (nt) vs nose-nose (nose_nose) among frames with exactly one of the two labels (> 0.5 prefers nose-tail). This is IN-SAMPLE (no held-out split): it describes what exists in each dictionary, not how well a selection rule generalises. The "SG" columns in diag/*.json were computed before rd41_3 was moved to "unknown" (12 SG pools there; 11 in the scored evaluation).

| model | best nose-nose neuron (Matryoshka level): AUROC, discrimination AUROC | best nose-tail neuron (level): AUROC, discrimination AUROC | same neuron? | best balanced-AUROC pick nose-nose / nose-tail | # neurons with nose-tail AUROC > 0.7 and discrimination > 0.6 | # with nose-nose AUROC > 0.7 and discrimination < 0.4 |
|---|---|---|---|---|---|---|
| V_s1001 | #222 (L256): 0.830, 0.44 | #759 (L2048): 0.793, 0.76 | no | #222 / #453 | 53 | 0 |
| V_s1002 | #436 (L512): 0.773, 0.47 | #270 (L512): 0.810, 0.79 | no | #1436 / #270 | 47 | 2 |
| V_s1003 | #1472 (L2048): 0.827, 0.34 | #1802 (L2048): 0.812, 0.80 | no | #1472 / #1802 | 44 | 2 |
| V_s1004 | #262 (L512): 0.758, 0.49 | #449 (L512): 0.801, 0.79 | no | #1991 / #449 | 44 | 1 |
| V_s1005 | #492 (L512): 0.795, 0.42 | #317 (L512): 0.793, 0.77 | no | #492 / #317 | 44 | 0 |
| V_s1006 | #404 (L512): 0.825, 0.33 | #437 (L512): 0.795, 0.78 | no | #404 / #437 | 38 | 2 |
| v1_s0 | #611 (L2048): 0.680, 0.44 | #163 (L256): 0.762, 0.69 | no | #611 / #163 | 2 | 0 |
| v1_s1 | #682 (L2048): 0.659, 0.51 | #431 (L512): 0.751, 0.71 | no | #477 / #431 | 1 | 0 |
| v1_s2 | #594 (L2048): 0.640, 0.47 | #475 (L512): 0.691, 0.58 | no | #38 / #399 | 0 | 0 |
| v1_s3 | #451 (L512): 0.667, 0.44 | #460 (L512): 0.724, 0.59 | no | #451 / #15 | 0 | 0 |
| v1_s4 | #282 (L512): 0.700, 0.38 | #240 (L256): 0.736, 0.59 | no | #282 / #504 | 1 | 1 |
| v1_s5 | #376 (L512): 0.659, 0.58 | #376 (L512): 0.729, 0.58 | yes | #42 / #507 | 1 | 0 |
| v1_s6 | #157 (L256): 0.650, 0.59 | #157 (L256): 0.730, 0.59 | yes | #222 / #157 | 0 | 0 |
| v1_s7 | #361 (L512): 0.802, 0.39 | #361 (L512): 0.720, 0.39 | yes | #361 / #22 | 0 | 1 |
| v1_s8 | #819 (L2048): 0.705, 0.43 | #439 (L512): 0.705, 0.66 | no | #819 / #439 | 1 | 0 |
| v1_s9 | #545 (L2048): 0.699, 0.46 | #897 (L2048): 0.695, 0.62 | no | #259 / #296 | 0 | 0 |
| v1_s10 | #922 (L2048): 0.662, 0.51 | #851 (L2048): 0.731, 0.59 | no | #111 / #414 | 1 | 0 |

Findings:
- V: in 6 of 6 seeds the best nose-tail neuron on all the data is a DIFFERENT neuron from the best nose-nose neuron, and it is nose-tail-specific (discrimination AUROC 0.76-0.80). 38-53 neurons per V seed have nose-tail AUROC > 0.7 with discrimination > 0.6. So the representation already holds a distinct nose-tail neuron.
- The "same neuron" result of the v2.1 extended report comes from the cross-fitted selection: on half the pools the broad nose-nose neuron (e.g. #222, nose-tail AUROC 0.777, 4th best) sometimes wins the nose-tail AUROC. The plain AUROC barely penalises firing on the sibling behaviour, which is only ~1-2% of frames.
- Nose-nose is the weak side: the best nose-nose neurons are only mildly nose-nose-specific (discrimination 0.33-0.49; #222 0.44). Only 0-2 neurons per V seed have nose-nose AUROC > 0.7 with discrimination < 0.4.
- No evidence of hidden Matryoshka "children": #222 (level 256) has at most one neuron with decoder cosine > 0.3 (#921, nose-tail AUROC 0.59). The specific nose-tail neurons sit at levels 512 / 2048 and are unrelated in decoder direction (cosine with the nose-nose top neuron mostly < 0.1).
- v1 (in-sample): distinct best neurons in 8/11 seeds; its nose-tail neurons are less specific (discrimination 0.58-0.72) and its nose-nose neurons not specific (0.38-0.59).
