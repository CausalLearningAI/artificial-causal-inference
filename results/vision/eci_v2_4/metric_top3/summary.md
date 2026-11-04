# precision@top3/video: ECI v2 alignment metric (replaces precision@100)

**Definition.** For each eligible held-out video: the 3 frames with the highest frame-max code of the selected neuron (raw code, BEFORE the HMM; ties -> earliest frame). precision@top3/video = labelled-positive frames among them / (3 x eligible videos). Eligible = held-out (measurement) video, annotated and valid for the behaviour, with >= 1 positive frame in the scored frame set; videos with no positive frame are excluded (counts below). Chance level = mean over eligible videos of the per-video positive-frame rate (expected precision of 3 random frames per video).

Neuron selection, selection / measurement sets and frame sets are UNCHANGED (stored v2.3 tags). Numbers: mean over seeds [95% CI, bootstrap over held-out pools (mice) / held-out videos within series x treatment (ants) and seeds, 1,000 replicates, the same replicate weights and seed draws as precision@100 of the same arm and behaviour].

Note: the existing precision@100 ranks v2 models (V, Vall) by the HMM P(on) readout (ties by raw code), v1 by its raw code. The headline precision@top3/video uses the raw code for every model as specified; the column "ranked by the precision@100 readout" gives the P(on)-ranked variant (identical for v1).

Models: v1s0 = deployed v1 seed 0; V = v2 current recipe, seeds 1001-1006; Vall = v2.3 (V trained on all tokens), seeds 2001-2003; v1 = v1 seeds 0-10. Rules: auc = best AUROC on the selection set (primary); bal = best balanced AUROC; hyb = nose-nose by auc, nose-tail by bal (post hoc).

## Mice, ext frames (864,000 = 15-min atlas window + first 15 min of the habituation videos)

Select on the 11 SG pools, measure on the 12 non-SG pools (432,000 held-out frames).

### nose-nose (nose_nose)

Eligible held-out videos: 72 of 72 (0 excluded, no positive frame); frames scored per model seed: 216 (= 3 x 72); invalid (unlabelled) held-out frames: 0.

| model \| rule | precision@top3/video [95% CI] | chance level (random 3 frames / video) [95% CI] | precision@top3/video, ranked by the precision@100 readout [95% CI] | AUROC [95% CI] | precision@100 [95% CI] | precision@top3/video per seed |
|---|---|---|---|---|---|---|
| V|auc | 0.270 [0.195, 0.351] | 0.0157 [0.0131, 0.0182] | 0.269 [0.194, 0.350] | 0.805 [0.781, 0.828] | 0.323 [0.235, 0.435] | 0.269, 0.278, 0.306, 0.245, 0.329, 0.194 |
| V|bal | 0.231 [0.157, 0.313] | 0.0157 [0.0131, 0.0182] | 0.230 [0.154, 0.313] | 0.792 [0.757, 0.826] | 0.275 [0.177, 0.385] | 0.269, 0.139, 0.306, 0.148, 0.329, 0.194 |
| V|hyb | 0.270 [0.191, 0.348] | 0.0157 [0.0131, 0.0182] | 0.269 [0.191, 0.347] | 0.805 [0.782, 0.828] | 0.323 [0.238, 0.422] | 0.269, 0.278, 0.306, 0.245, 0.329, 0.194 |
| Vall|auc | 0.236 [0.139, 0.332] | 0.0157 [0.0131, 0.0182] | 0.235 [0.139, 0.329] | 0.769 [0.735, 0.801] | 0.327 [0.180, 0.467] | 0.306, 0.259, 0.144 |
| Vall|bal | 0.168 [0.069, 0.296] | 0.0157 [0.0131, 0.0182] | 0.167 [0.066, 0.292] | 0.733 [0.683, 0.791] | 0.213 [0.060, 0.397] | 0.306, 0.130, 0.069 |
| Vall|hyb | 0.236 [0.148, 0.338] | 0.0157 [0.0131, 0.0182] | 0.235 [0.148, 0.338] | 0.769 [0.737, 0.802] | 0.327 [0.190, 0.470] | 0.306, 0.259, 0.144 |
| v1s0|auc | 0.074 [0.042, 0.120] | 0.0157 [0.0131, 0.0182] | 0.074 [0.042, 0.120] | 0.691 [0.678, 0.709] | 0.100 [0.040, 0.160] | 0.074 |
| v1s0|bal | 0.074 [0.042, 0.120] | 0.0157 [0.0131, 0.0182] | 0.074 [0.042, 0.120] | 0.691 [0.678, 0.709] | 0.100 [0.040, 0.160] | 0.074 |
| v1s0|hyb | 0.074 [0.042, 0.120] | 0.0157 [0.0131, 0.0182] | 0.074 [0.042, 0.120] | 0.691 [0.678, 0.709] | 0.100 [0.040, 0.160] | 0.074 |

### nose-tail (nt)

Eligible held-out videos: 63 of 72 (9 excluded, no positive frame); frames scored per model seed: 189 (= 3 x 63); invalid (unlabelled) held-out frames: 0.

| model \| rule | precision@top3/video [95% CI] | chance level (random 3 frames / video) [95% CI] | precision@top3/video, ranked by the precision@100 readout [95% CI] | AUROC [95% CI] | precision@100 [95% CI] | precision@top3/video per seed |
|---|---|---|---|---|---|---|
| V|auc | 0.086 [0.035, 0.159] | 0.0103 [0.0058, 0.0172] | 0.086 [0.036, 0.159] | 0.791 [0.742, 0.826] | 0.043 [0.017, 0.085] | 0.032, 0.132, 0.085, 0.069, 0.032, 0.164 |
| V|bal | 0.120 [0.061, 0.201] | 0.0103 [0.0058, 0.0172] | 0.123 [0.063, 0.207] | 0.798 [0.749, 0.833] | 0.038 [0.007, 0.107] | 0.085, 0.132, 0.153, 0.153, 0.032, 0.164 |
| V|hyb | 0.120 [0.061, 0.205] | 0.0103 [0.0058, 0.0172] | 0.123 [0.064, 0.208] | 0.798 [0.751, 0.832] | 0.038 [0.007, 0.105] | 0.085, 0.132, 0.153, 0.153, 0.032, 0.164 |
| Vall|auc | 0.132 [0.053, 0.224] | 0.0103 [0.0058, 0.0172] | 0.136 [0.055, 0.231] | 0.793 [0.752, 0.830] | 0.037 [0.003, 0.123] | 0.085, 0.159, 0.153 |
| Vall|bal | 0.152 [0.091, 0.230] | 0.0103 [0.0058, 0.0172] | 0.157 [0.097, 0.235] | 0.811 [0.773, 0.836] | 0.043 [0.007, 0.130] | 0.143, 0.159, 0.153 |
| Vall|hyb | 0.152 [0.092, 0.232] | 0.0103 [0.0058, 0.0172] | 0.157 [0.097, 0.234] | 0.811 [0.773, 0.837] | 0.043 [0.003, 0.127] | 0.143, 0.159, 0.153 |
| v1s0|auc | 0.127 [0.046, 0.227] | 0.0103 [0.0058, 0.0172] | 0.127 [0.046, 0.227] | 0.777 [0.711, 0.816] | 0.210 [0.040, 0.360] | 0.127 |
| v1s0|bal | 0.127 [0.046, 0.227] | 0.0103 [0.0058, 0.0172] | 0.127 [0.046, 0.227] | 0.777 [0.711, 0.816] | 0.210 [0.040, 0.360] | 0.127 |
| v1s0|hyb | 0.127 [0.046, 0.227] | 0.0103 [0.0058, 0.0172] | 0.127 [0.046, 0.227] | 0.777 [0.711, 0.816] | 0.210 [0.040, 0.360] | 0.127 |

## Mice, v1 seeds 0-10 on the 648,000 atlas-window frames only

v1 seeds 1-10 have codes on the 648,000 atlas-window frames only (no ext codes), so this arm is scored on the atlas window, NOT the ext frame set; not directly comparable with the table above. Selection recomputed with the v2.2 rule code (v1 seed 0 picks checked against the stored v2.2 full-window tag).

### nose-nose (nose_nose)

Eligible held-out videos: 66 of 72 (6 excluded, no positive frame); frames scored per model seed: 198 (= 3 x 66); invalid (unlabelled) held-out frames: 0.

| model \| rule | precision@top3/video [95% CI] | chance level (random 3 frames / video) [95% CI] | precision@top3/video, ranked by the precision@100 readout [95% CI] | AUROC [95% CI] | precision@100 [95% CI] | precision@top3/video per seed |
|---|---|---|---|---|---|---|
| v1s0-10|auc | 0.084 [0.051, 0.126] | 0.0154 [0.0124, 0.0181] | 0.084 [0.051, 0.126] | 0.673 [0.643, 0.708] | 0.103 [0.056, 0.140] | 0.066, 0.076, 0.051, 0.051, 0.045, 0.146, 0.035, 0.131, 0.045, 0.091, 0.182 |
| v1s0-10|bal | 0.049 [0.026, 0.082] | 0.0154 [0.0124, 0.0181] | 0.049 [0.026, 0.082] | 0.617 [0.563, 0.674] | 0.046 [0.015, 0.091] | 0.066, 0.020, 0.045, 0.010, 0.045, 0.066, 0.030, 0.131, 0.045, 0.045, 0.030 |
| v1s0-10|hyb | 0.084 [0.053, 0.124] | 0.0154 [0.0124, 0.0181] | 0.084 [0.053, 0.124] | 0.673 [0.641, 0.708] | 0.103 [0.061, 0.138] | 0.066, 0.076, 0.051, 0.051, 0.045, 0.146, 0.035, 0.131, 0.045, 0.091, 0.182 |

### nose-tail (nt)

Eligible held-out videos: 57 of 72 (15 excluded, no positive frame); frames scored per model seed: 171 (= 3 x 57); invalid (unlabelled) held-out frames: 0.

| model \| rule | precision@top3/video [95% CI] | chance level (random 3 frames / video) [95% CI] | precision@top3/video, ranked by the precision@100 readout [95% CI] | AUROC [95% CI] | precision@100 [95% CI] | precision@top3/video per seed |
|---|---|---|---|---|---|---|
| v1s0-10|auc | 0.092 [0.038, 0.161] | 0.0105 [0.0060, 0.0167] | 0.092 [0.038, 0.161] | 0.733 [0.665, 0.771] | 0.141 [0.043, 0.250] | 0.123, 0.082, 0.076, 0.058, 0.187, 0.088, 0.023, 0.152, 0.018, 0.135, 0.070 |
| v1s0-10|bal | 0.098 [0.047, 0.165] | 0.0105 [0.0060, 0.0167] | 0.098 [0.047, 0.165] | 0.707 [0.657, 0.746] | 0.134 [0.045, 0.237] | 0.123, 0.099, 0.006, 0.064, 0.187, 0.088, 0.035, 0.152, 0.117, 0.135, 0.070 |
| v1s0-10|hyb | 0.098 [0.046, 0.169] | 0.0105 [0.0060, 0.0167] | 0.098 [0.046, 0.169] | 0.707 [0.654, 0.746] | 0.134 [0.047, 0.239] | 0.123, 0.099, 0.006, 0.064, 0.187, 0.088, 0.035, 0.152, 0.117, 0.135, 0.070 |

## Ants, v3 + v4 (147 videos, first 10 min at 1 fps), select on half A, measure on half B (PRIMARY)

### grooming-to-focal (any)

Eligible held-out videos: 66 of 73 (7 excluded, no positive frame); frames scored per model seed: 198. Rule auc.

| model \| rule | precision@top3/video [95% CI] | chance level (random 3 frames / video) [95% CI] | precision@top3/video, ranked by the precision@100 readout [95% CI] | AUROC [95% CI] | precision@100 [95% CI] | precision@top3/video per seed |
|---|---|---|---|---|---|---|
| v1 \| auc | 0.801 [0.732, 0.872] | 0.4259 [0.3705, 0.4864] | 0.801 [0.732, 0.872] | 0.852 [0.821, 0.882] | 0.955 [0.885, 0.988] | 0.788, 0.833, 0.803, 0.793, 0.818, 0.778, 0.823, 0.833, 0.808, 0.773, 0.758 |
| v1s0 \| auc | 0.788 [0.710, 0.870] | 0.4259 [0.3705, 0.4864] | 0.788 [0.710, 0.870] | 0.875 [0.857, 0.894] | 0.940 [0.850, 1.000] | 0.788 |
| V \| auc | 0.746 [0.666, 0.828] | 0.4259 [0.3705, 0.4864] | 0.790 [0.711, 0.864] | 0.861 [0.841, 0.882] | 0.863 [0.750, 0.942] | 0.773, 0.753, 0.763, 0.737, 0.732, 0.717 |
| Vall \| auc | 0.763 [0.690, 0.843] | 0.4259 [0.3705, 0.4864] | 0.810 [0.736, 0.881] | 0.860 [0.840, 0.883] | 0.897 [0.803, 0.960] | 0.778, 0.732, 0.778 |

## Ants, v3 + v4 (147 videos, first 10 min at 1 fps), select on half B, measure on half A

### grooming-to-focal (any)

Eligible held-out videos: 69 of 74 (5 excluded, no positive frame); frames scored per model seed: 207. Rule auc.

| model \| rule | precision@top3/video [95% CI] | chance level (random 3 frames / video) [95% CI] | precision@top3/video, ranked by the precision@100 readout [95% CI] | AUROC [95% CI] | precision@100 [95% CI] | precision@top3/video per seed |
|---|---|---|---|---|---|---|
| v1 \| auc | 0.810 [0.729, 0.879] | 0.3871 [0.3369, 0.4406] | 0.810 [0.729, 0.879] | 0.872 [0.841, 0.900] | 0.954 [0.894, 0.982] | 0.821, 0.710, 0.841, 0.836, 0.855, 0.802, 0.850, 0.845, 0.821, 0.758, 0.773 |
| v1s0 \| auc | 0.821 [0.731, 0.899] | 0.3871 [0.3369, 0.4406] | 0.821 [0.731, 0.899] | 0.888 [0.861, 0.910] | 0.940 [0.870, 0.990] | 0.821 |
| V \| auc | 0.778 [0.706, 0.850] | 0.3871 [0.3369, 0.4406] | 0.828 [0.760, 0.889] | 0.874 [0.848, 0.898] | 0.790 [0.655, 0.918] | 0.700, 0.787, 0.778, 0.812, 0.787, 0.802 |
| Vall \| auc | 0.804 [0.727, 0.868] | 0.3871 [0.3369, 0.4406] | 0.833 [0.758, 0.897] | 0.873 [0.844, 0.898] | 0.793 [0.637, 0.920] | 0.792, 0.816, 0.802 |

## Checks

1. Identity: precision@100 (mean and both CI ends) recomputed through this script's bootstrap vs the stored reports: mice ext max abs diff 0.00e+00 over 18 arm x behaviour cells; ants max abs diff 0.00e+00 over 8 cells.
2. Hand check (mice ext, V seed 1001, rule auc, nose-nose; top 3 frames of the first 3 eligible held-out videos by raw code):

```
{
 "model": "V_s1001",
 "rule": "auc",
 "behaviour": "nose_nose",
 "neuron": 222,
 "videos": [
  {
   "video": "het_ash1l_f_1_F_H",
   "positive_frames": 149,
   "valid_frames": 9000,
   "top3 (frame k, raw code, readout, label)": [
    [
     7632,
     10.0156,
     1.0,
     false
    ],
    [
     771,
     9.6719,
     1.0,
     false
    ],
    [
     6518,
     9.6562,
     1.0,
     false
    ]
   ]
  },
  {
   "video": "het_ash1l_f_1_F_O",
   "positive_frames": 89,
   "valid_frames": 4500,
   "top3 (frame k, raw code, readout, label)": [
    [
     1841,
     10.5,
     1.0,
     false
    ],
    [
     1184,
     9.6641,
     1.0,
     true
    ],
    [
     2044,
     9.0,
     1.0,
     false
    ]
   ]
  },
  {
   "video": "het_ash1l_f_1_F_P",
   "positive_frames": 82,
   "valid_frames": 4500,
   "top3 (frame k, raw code, readout, label)": [
    [
     836,
     11.9297,
     1.0,
     true
    ],
    [
     1451,
     10.25,
     1.0,
     false
    ],
    [
     2402,
     9.2578,
     1.0,
     true
    ]
   ]
  }
 ],
 "hits": 3,
 "precision_on_these_3_videos": 0.3333333333333333,
 "stored_hits_raw": [
  0,
  1,
  2
 ]
}
```

Ants hand check (V seed 1001, A -> B, grooming-to-focal (any)):

```
{
 "model": "V_s1001",
 "dir": "A2B",
 "behaviour": "groom_any",
 "neuron": 20,
 "videos": [
  {
   "video": "3_10_6",
   "positive_frames": 447,
   "valid_frames": 600,
   "top3 (frame k, raw code, readout, label)": [
    [
     2647,
     14.6562,
     1.0,
     true
    ],
    [
     697,
     14.1328,
     1.0,
     true
    ],
    [
     592,
     13.1094,
     1.0,
     true
    ]
   ]
  },
  {
   "video": "3_10_9",
   "positive_frames": 20,
   "valid_frames": 600,
   "top3 (frame k, raw code, readout, label)": [
    [
     957,
     12.1406,
     1.0,
     true
    ],
    [
     2292,
     11.6094,
     1.0,
     false
    ],
    [
     1567,
     11.5703,
     0.9983,
     false
    ]
   ]
  },
  {
   "video": "3_11_2",
   "positive_frames": 82,
   "valid_frames": 600,
   "top3 (frame k, raw code, readout, label)": [
    [
     17,
     13.9219,
     0.9629,
     false
    ],
    [
     1672,
     13.0625,
     0.9999,
     false
    ],
    [
     2942,
     10.1641,
     0.8686,
     true
    ]
   ]
  }
 ],
 "hits": 5,
 "precision_on_these_3_videos": 0.5555555555555556,
 "stored_hits_raw": [
  3,
  1,
  1
 ]
}
```

3. Top-100 frames: labelled positives among the stored tag top 100 vs this script's own ranking (same readout, ties earliest frame instead of random): see top3.json (top100_check).

v1 seed 0 on the 648,000 atlas frames, picks [stored v2.2 full tag, recomputed]: {"auc": [{"nose_nose": 611, "nt": 414}, {"nose_nose": 611, "nt": 414}], "bal": [{"nose_nose": 611, "nt": 414}, {"nose_nose": 611, "nt": 414}], "hyb": [{"nose_nose": 611, "nt": 414}, {"nose_nose": 611, "nt": 414}]}

