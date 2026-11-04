# ECI v1 vs v2 on known behaviours, pooled per domain (extended evaluation)

Date: 2026-11-03. Post hoc, reported view (not a pre-registered gate). Everything here comes from existing saved codes. There was no retraining and no re-encoding of the scored frames.

- **v2** is arm V: DINOv3-L/16 block 20, mask v5, input x + (x_{t+1} − x_{t−1})/2, Matryoshka SAE with 2048 latents, per-neuron 3-state HMM giving P(on), frame max over patches. Seeds 1001-1006.
- **v1** is the deployed v1 recipe. Seeds v1_s0 to v1_s10.

## Summary

### Mice

All 24 annotated pools are pooled (144 two-minute windows, 86,400 frames at 5 fps).

**v2 is clearly better at finding the nose behaviours. It is no better at measuring how much time each video contains.**

Clearly better (95% CI excludes 0):
- **AUROC, all 4 behaviours:**
  - nose-tail +0.065;
  - nose-nose +0.126;
  - nose-nose-mutual +0.148;
  - nose-nose-passive +0.110.
- **Precision@100:**
  - nose-nose +0.107 (0.098 → 0.210);
  - nose-nose-mutual +0.059;
  - nose-nose-passive +0.069.
- **Per-frame F1, all 4 behaviours.** The gains are small (+0.015 to +0.029), and the F1 levels stay very low (0.05 to 0.11).
- **Raw per-video time error (MAE of the summed P(on)) for nose-tail:** 0.238 → 0.040 of video length.

On par (CI includes 0):
- precision@100 for nose-tail;
- raw time MAE for the three nose-nose behaviours;
- calibrated time MAE for all four;
- Pearson correlation between predicted and true per-video time for all four.

Not useful for time, for either model:
- Raw on-time overshoots the true time by 15-24% of the video. The true time is only 1-2% of the video.
- After a one-scalar calibration, MAE is no better than a constant-rate guess that uses no neuron at all.
- Per-video correlation is weak: Pearson r 0.17-0.36, Spearman rho 0.26-0.43.

### Ants

The v2, v3, v4 and v4_infested series are pooled: 212 ten-minute videos at 1 frame per second, 127,200 frames.

**v2 is on par on ranking (AUROC) and on time. It is clearly worse at the very top of the ranking. The exception is focal single, where v2 is better.**

Clearly worse (CI excludes 0):
- **Precision@100:**
  - grooming-to-focal (any) −0.102;
  - Y2F −0.185;
  - B2F −0.210;
  - focal double −0.207.
- **F1 for Y2F:** −0.035.

Clearly better, all for focal single:
- F1 +0.030;
- raw time MAE −0.019 of video length;
- calibrated time MAE −0.019 of video length, about −11 s per video.

On par:
- AUROC for all 5 behaviours. The CIs span about ±0.03.
- Precision@100 for focal single: +0.143 [−0.021, +0.307].
- Calibrated time MAE and the time correlations for the other 4 behaviours.
- Grooming-to-focal (any), calibrated MAE: −0.016 [−0.031, +0.001].

For both models, per-video time tracks the truth well here:
- Pearson r is 0.66 to 0.81.
- Calibrated MAE is clearly below the constant-rate reference. For example, grooming-to-focal (any) is about 0.12-0.13 against 0.21.

### Interpretability

The top frames are in `sheet_mice.png` and `sheet_ants.png`, and my reading of them is below.

**Ants:** both models' grooming neurons are interpretable, as "two or more ants in body contact".
- v2's neuron #20 marks the exact contact point between ants. v1's neuron #90 marks the whole ant cluster.
- Neither separates Y2F from B2F.
- v2's focal-double neuron (#157) is **not** interpretable. Its peak patch often sits off the ants, near the dish wall.

**Mice:** both models' neurons are interpretable, as "mice in head-to-body contact or huddling".
- v2's neuron #222 is more tightly located at the point where the heads meet. v1's neuron #611 spreads over the bodies.
- Neither is specific to a nose behaviour. Of the 12 top frames, only 0-3 carry the label.
- Even so, 3/12 is about 12 times the nose-nose base rate of 2.1%.

## Prevalence and per-video time (bias, MAE)

Per-video numbers are averaged over the held-out videos of the cell; "± X" after a bias or MAE is the standard error (SE) over videos (for v2 the median of the per-seed SE). v2 = median over the 6 seeds with [min, max] seed range; v1 = median over its 11 seeds (HMM P(on) fitted to the raw code, as in the rest of this report). Bias = mean over videos of (predicted time - true time); MAE = mean over videos of |predicted time - true time|; predicted time = sum over frames of P(on). "raw" = unscaled sum P(on); "calibrated" = sum P(on) times one scale factor fitted on the selection half. Video length: mice 120 s (600 frames at 5 fps), ants 600 s (600 frames at 1 fps). Frac = fraction of video length.

### mice

**Ground truth (identical for v1 and v2).** Overall = pooled fraction of frames with the label; per-video = mean ± SE over videos of the per-video fraction; time = true seconds of the behaviour per video, mean ± SE.

| behaviour | videos | overall frame fraction | per-video frame fraction, mean ± SE | true time per video, s, mean ± SE |
|---|---|---|---|---|
| nose-nose | 144 | 0.0214 | 0.0214 ± 0.0027 | 2.6 ± 0.3 |
| nose-tail | 144 | 0.0084 | 0.0084 ± 0.0019 | 1.0 ± 0.2 |
| nose-nose-mutual | 144 | 0.0070 | 0.0070 ± 0.0011 | 0.8 ± 0.1 |
| nose-nose-passive | 144 | 0.0144 | 0.0144 ± 0.0023 | 1.7 ± 0.3 |

**Per-video total-time bias and MAE** (held-out videos, mean over videos ± SE).

| behaviour | model | raw bias, s | raw bias, frac | raw MAE, s | raw MAE, frac | calibrated bias, s | calibrated bias, frac | calibrated MAE, s | calibrated MAE, frac |
|---|---|---|---|---|---|---|---|---|---|
| nose-nose | v1 | +17.8 ± 1.4 [+11.3, +26.6] | +0.148 ± 0.011 [+0.094, +0.222] | +18.1 ± 1.3 [+11.7, +27.1] | +0.151 ± 0.011 [+0.097, +0.226] | +0.05 ± 0.34 [-0.14, +0.33] | +0.0004 ± 0.0028 [-0.0012, +0.0028] | +2.5 ± 0.3 [+2.2, +2.8] | +0.021 ± 0.002 [+0.018, +0.023] |
| nose-nose | v2 | +21.5 ± 1.3 [+10.8, +30.3] | +0.179 ± 0.011 [+0.090, +0.252] | +21.6 ± 1.3 [+11.0, +30.3] | +0.180 ± 0.011 [+0.091, +0.252] | -0.02 ± 0.32 [-0.41, +0.03] | -0.0002 ± 0.0027 [-0.0034, +0.0003] | +2.3 ± 0.3 [+2.2, +2.5] | +0.019 ± 0.002 [+0.019, +0.021] |
| nose-tail | v1 | +28.6 ± 1.5 [+14.2, +41.1] | +0.238 ± 0.013 [+0.118, +0.343] | +28.6 ± 1.5 [+14.3, +41.1] | +0.238 ± 0.013 [+0.119, +0.343] | +0.00 ± 0.22 [-0.09, +0.12] | +0.0000 ± 0.0018 [-0.0008, +0.0010] | +1.3 ± 0.2 [+1.3, +1.5] | +0.011 ± 0.002 [+0.011, +0.012] |
| nose-tail | v2 | +4.2 ± 0.6 [+3.3, +13.5] | +0.035 ± 0.005 [+0.027, +0.112] | +4.8 ± 0.6 [+3.9, +13.7] | +0.040 ± 0.005 [+0.032, +0.115] | +0.14 ± 0.24 [+0.02, +0.41] | +0.0011 ± 0.0020 [+0.0002, +0.0034] | +1.5 ± 0.2 [+1.3, +1.8] | +0.012 ± 0.002 [+0.011, +0.015] |
| nose-nose-mutual | v1 | +25.0 ± 1.6 [+16.3, +35.5] | +0.208 ± 0.013 [+0.136, +0.296] | +25.1 ± 1.6 [+16.4, +35.6] | +0.210 ± 0.013 [+0.136, +0.297] | -0.01 ± 0.13 [-0.05, +0.05] | -0.0001 ± 0.0011 [-0.0004, +0.0004] | +0.9 ± 0.1 [+0.8, +0.9] | +0.007 ± 0.001 [+0.007, +0.008] |
| nose-nose-mutual | v2 | +24.6 ± 1.4 [+12.6, +32.0] | +0.205 ± 0.012 [+0.105, +0.267] | +24.6 ± 1.4 [+12.6, +32.0] | +0.205 ± 0.012 [+0.105, +0.267] | -0.02 ± 0.13 [-0.02, -0.01] | -0.0001 ± 0.0011 [-0.0002, -0.0001] | +0.8 ± 0.1 [+0.8, +0.9] | +0.007 ± 0.001 [+0.007, +0.008] |
| nose-nose-passive | v1 | +21.5 ± 1.3 [+11.7, +34.7] | +0.179 ± 0.011 [+0.098, +0.289] | +21.7 ± 1.3 [+11.8, +34.8] | +0.181 ± 0.011 [+0.099, +0.290] | +0.01 ± 0.28 [-0.12, +0.20] | +0.0001 ± 0.0024 [-0.0010, +0.0016] | +2.0 ± 0.2 [+1.8, +2.1] | +0.017 ± 0.002 [+0.015, +0.018] |
| nose-nose-passive | v2 | +17.5 ± 1.1 [+3.3, +31.1] | +0.146 ± 0.009 [+0.028, +0.259] | +17.6 ± 1.1 [+4.6, +31.1] | +0.147 ± 0.009 [+0.039, +0.259] | +0.01 ± 0.27 [-0.25, +0.26] | +0.0001 ± 0.0023 [-0.0021, +0.0022] | +1.9 ± 0.2 [+1.9, +2.3] | +0.016 ± 0.002 [+0.016, +0.019] |

### ants

**Ground truth (identical for v1 and v2).** Overall = pooled fraction of frames with the label; per-video = mean ± SE over videos of the per-video fraction; time = true seconds of the behaviour per video, mean ± SE.

| behaviour | videos | overall frame fraction | per-video frame fraction, mean ± SE | true time per video, s, mean ± SE |
|---|---|---|---|---|
| grooming-to-focal (any) | 212 | 0.3389 | 0.3389 ± 0.0172 | 203.4 ± 10.3 |
| Y2F | 212 | 0.2082 | 0.2082 ± 0.0142 | 124.9 ± 8.5 |
| B2F | 212 | 0.1957 | 0.1957 ± 0.0133 | 117.4 ± 8.0 |
| focal single | 212 | 0.2740 | 0.2740 ± 0.0128 | 164.4 ± 7.7 |
| focal double | 147 | 0.0768 | 0.0768 ± 0.0096 | 46.1 ± 5.8 |

**Per-video total-time bias and MAE** (held-out videos, mean over videos ± SE).

| behaviour | model | raw bias, s | raw bias, frac | raw MAE, s | raw MAE, frac | calibrated bias, s | calibrated bias, frac | calibrated MAE, s | calibrated MAE, frac |
|---|---|---|---|---|---|---|---|---|---|
| grooming-to-focal (any) | v1 | -65.8 ± 6.2 [-81.0, -33.1] | -0.110 ± 0.010 [-0.135, -0.055] | +88.1 ± 4.8 [+82.8, +104.0] | +0.147 ± 0.008 [+0.138, +0.173] | +0.19 ± 7.24 [-3.82, +0.44] | +0.0003 ± 0.0121 [-0.0064, +0.0007] | +78.1 ± 5.0 [+62.9, +100.5] | +0.130 ± 0.008 [+0.105, +0.167] |
| grooming-to-focal (any) | v2 | -55.8 ± 6.2 [-62.4, -47.0] | -0.093 ± 0.010 [-0.104, -0.078] | +84.0 ± 4.5 [+75.6, +88.0] | +0.140 ± 0.008 [+0.126, +0.147] | +0.22 ± 6.90 [+0.08, +0.29] | +0.0004 ± 0.0115 [+0.0001, +0.0005] | +70.1 ± 4.9 [+60.7, +73.4] | +0.117 ± 0.008 [+0.101, +0.122] |
| Y2F | v1 | +10.2 ± 6.2 [-15.5, +45.4] | +0.017 ± 0.010 [-0.026, +0.076] | +67.6 ± 4.5 [+58.8, +86.4] | +0.113 ± 0.007 [+0.098, +0.144] | +0.10 ± 6.11 [-0.09, +1.09] | +0.0002 ± 0.0102 [-0.0002, +0.0018] | +65.9 ± 4.2 [+59.0, +72.1] | +0.110 ± 0.007 [+0.098, +0.120] |
| Y2F | v2 | +24.1 ± 6.7 [-20.4, +31.5] | +0.040 ± 0.011 [-0.034, +0.052] | +72.0 ± 4.7 [+69.1, +82.6] | +0.120 ± 0.008 [+0.115, +0.138] | +0.14 ± 6.49 [-1.92, +0.26] | +0.0002 ± 0.0108 [-0.0032, +0.0004] | +70.5 ± 4.3 [+66.4, +75.4] | +0.118 ± 0.007 [+0.111, +0.126] |
| B2F | v1 | +17.7 ± 6.5 [-6.9, +52.9] | +0.030 ± 0.011 [-0.011, +0.088] | +67.5 ± 4.7 [+61.8, +88.7] | +0.113 ± 0.008 [+0.103, +0.148] | +0.05 ± 6.13 [-2.95, +0.16] | +0.0001 ± 0.0102 [-0.0049, +0.0003] | +63.0 ± 4.3 [+60.5, +77.0] | +0.105 ± 0.007 [+0.101, +0.128] |
| B2F | v2 | +30.1 ± 6.5 [+15.0, +39.0] | +0.050 ± 0.011 [+0.025, +0.065] | +71.0 ± 4.7 [+69.3, +74.2] | +0.118 ± 0.008 [+0.115, +0.124] | -0.02 ± 6.02 [-0.87, +1.02] | -0.0000 ± 0.0100 [-0.0015, +0.0017] | +65.1 ± 4.0 [+64.3, +66.0] | +0.108 ± 0.007 [+0.107, +0.110] |
| focal single | v1 | -23.6 ± 6.5 [-41.4, +5.9] | -0.039 ± 0.011 [-0.069, +0.010] | +74.7 ± 4.5 [+69.1, +89.8] | +0.124 ± 0.007 [+0.115, +0.150] | +0.25 ± 7.44 [-0.31, +0.48] | +0.0004 ± 0.0124 [-0.0005, +0.0008] | +80.0 ± 5.0 [+67.9, +90.4] | +0.133 ± 0.008 [+0.113, +0.151] |
| focal single | v2 | -16.9 ± 6.1 [-23.5, -8.0] | -0.028 ± 0.010 [-0.039, -0.013] | +66.1 ± 4.2 [+58.3, +68.3] | +0.110 ± 0.007 [+0.097, +0.114] | +0.29 ± 6.58 [+0.11, +0.35] | +0.0005 ± 0.0110 [+0.0002, +0.0006] | +69.2 ± 4.6 [+58.7, +72.2] | +0.115 ± 0.008 [+0.098, +0.120] |
| focal double | v1 | +112.8 ± 7.4 [+49.0, +119.7] | +0.188 ± 0.012 [+0.082, +0.200] | +112.8 ± 7.4 [+54.9, +123.1] | +0.188 ± 0.012 [+0.092, +0.205] | -0.02 ± 4.22 [-0.61, +0.09] | -0.0000 ± 0.0070 [-0.0010, +0.0001] | +35.4 ± 3.0 [+34.7, +37.3] | +0.059 ± 0.005 [+0.058, +0.062] |
| focal double | v2 | +72.3 ± 6.9 [+27.5, +155.4] | +0.121 ± 0.012 [+0.046, +0.259] | +78.3 ± 6.4 [+38.2, +155.4] | +0.131 ± 0.011 [+0.064, +0.259] | -0.00 ± 4.40 [-0.06, +0.04] | -0.0000 ± 0.0073 [-0.0001, +0.0001] | +35.5 ± 3.2 [+29.9, +38.0] | +0.059 ± 0.005 [+0.050, +0.063] |


## How the numbers were made

### Neuron selection

This uses the ladder's cross-fitted rule (`v21_ladder.score2`):
- **Splits:** there are 6 unit splits. In each, the units are cut into two halves, giving 12 directions.
- **Units:**
  - mice: units are pools, 24 of them;
  - ants: units are videos.
- **Pick and evaluate:** in each direction, the neuron with the best AUROC on one half is picked. It is then evaluated on the other half.
- **Averaging:**
  - AUROC, precision@100, F1 and correlations are averaged over the 12 directions.
  - Per-video time errors are first averaged per video, over the 6 directions in which that video is held out. They are then averaged over videos.
- **Pooling:**
  - Mice: the four cells of the single series v1 already pool all pools.
  - Ants: the pooled frame table of `v21_precK.pooled_frame` is used. Focal double is eligible only in **v3 and v4**: 147 videos, 88,200 frames.
- The selected neurons per seed are listed in the "Selected neurons per seed" section of `tables.md`.

### Readouts

- **v2:** P(on) from the HMM, refitted exactly as the ladder does it. Identity checks are below.
- **v1:** the ladder ranks v1 by its raw frame-max code, and this report keeps that for AUROC and precision@100.
  - v1 has no P(on) of its own. For F1 and the time metrics, I fitted the **same 3-state HMM** (`src.eci.v2.hmm.fit`, K=3, label-free) to each selected v1 neuron's raw code, over all eval frames, with one sequence per video. These P(on) values are computed here and are not part of the stored v1 outputs.
  - For ants this HMM runs on the 1 fps eval grid. The v2 HMM runs at 5 fps.
  - As a side note, v1's native hard rule "code > 0" gives lower F1 than v1's HMM P(on) > 0.5 in 8 of 9 cells. The exception is focal single, 0.541 against 0.508. The values are in report.json under `F1fire`.

### Metric definitions

- **Ground-truth rate:** the fraction of eval frames with the label, overall and per video (mean ± standard error over videos).
- **Per-frame F1:** the hard prediction is P(on) > 0.5, scored on held-out frames.
- **Precision@100:** the top 100 held-out frames by the ladder readout.
  - This is identical by construction to `precK` (mice) and `precK/pooled` (ants).
  - Recomputed here, the identity check passed.
- **Per-video total time:** bias = predicted − true on-time per held-out video, and MAE = |bias|.
  - Units are seconds (mice 0.2 s per frame, ants 1 s per frame) and fraction of video length.
  - The scored "video" is the eval window: mice 120 s, ants 600 s.
  - **Raw:** the sum of P(on) per video (main tables), and the sum of hard on-frames (`tables.md`).
  - **Calibrated:** one scalar c per neuron and direction, fitted on the **selection half only** so that total predicted time equals total true time there. It is then applied to the held-out videos.
  - **Reference:** a constant-rate predictor, which is the selection half's behaviour rate × video length, with no neuron.
  - **Correlation:** Pearson r and Spearman rho of predicted vs true per-video totals across held-out videos, computed per direction and then averaged.
    - Calibration does not change these correlations.
    - They are the most relevant time number for causal contrasts.

### Uncertainty

- **Levels:** seed median [min, max] over 11 v1 seeds and 6 v2 seeds.
- **v2 − v1 differences:** the mean over seeds, with a 95% CI from a bootstrap that resamples units (pools or videos) and seeds.
  - F1, time MAE and Pearson r: 1000 replicates, computed here.
  - AUROC and precision@100: taken from the existing bootstrap arrays (2000 replicates). For mice these are ladder `boot_mice.npz` and precK `boot_mice.npz`; for ants, precK `pooled_ants.json`.
- **Caveat on the calibrated-MAE CI for mice nose-tail.** The upper end, +1.032, comes from the calibration itself. One v2 seed (V_s1004) uses a rarely-on neuron. In a few percent of replicates, resampling removes the pools where it fires on the selection half, and the calibration scalar explodes. That is a real fragility of one-scalar calibration for rare behaviours, not a coding error. That seed's median replicate is normal: 0.018.

### Identity checks (fail loud)

- **AUROC and precision@100** recomputed per (model, behaviour) against the stored ladder / precK values:
  - mice: max abs diff 5.6e-17 over 136 values;
  - ants: max abs diff 2.2e-16 over 170 values.
- **Per-patch maps in the figures:**
  - The recomputed max over tokens matches the stored frame code within fp16 precision.
  - Mice: max abs diff 0.0033 (v1) and 0.0036 (v2).
  - Ants: 0.0053 (v1) and 0.0078 (v2).
  - The ants v1 tokens were re-encoded on CPU from the jpgs, using DINOv2 and the foreground mask.

## Mice

### Ground truth

| behaviour | videos (pools) | frames | rate overall | per-video rate, mean ± SE | true time per video, s, mean ± SE (window 120 s) |
|---|---|---|---|---|---|
| nose-tail | 144 (24) | 86400 | 0.0084 | 0.0084 ± 0.0019 | 1.0 ± 0.2 |
| nose-nose | 144 (24) | 86400 | 0.0214 | 0.0214 ± 0.0027 | 2.6 ± 0.3 |
| nose-nose-mutual | 144 (24) | 86400 | 0.0070 | 0.0070 ± 0.0011 | 0.8 ± 0.1 |
| nose-nose-passive | 144 (24) | 86400 | 0.0144 | 0.0144 ± 0.0023 | 1.7 ± 0.3 |

### Main table

Levels are seed median [min, max]; the v2 − v1 rows give the difference with its [95% CI].

| behaviour | model | AUROC | precision@100 | per-frame F1 | raw time MAE, sum P(on), frac of video | calibrated time MAE, frac (s) | Pearson r, per-video time |
|---|---|---|---|---|---|---|---|
| nose-tail | v1 | 0.698 [0.628, 0.747] | 0.046 [0.026, 0.102] | 0.041 [0.030, 0.049] | 0.238 [0.119, 0.343] | 0.011 (1.3 s) | 0.203 [0.104, 0.268] |
| nose-tail | v2 | 0.757 [0.746, 0.772] | 0.047 [0.032, 0.057] | 0.072 [0.056, 0.075] | 0.040 [0.032, 0.115] | 0.012 (1.5 s) | 0.171 [0.100, 0.229] |
| nose-tail | v2 − v1 | **+0.065 [+0.017, +0.117]** | −0.007 [−0.034, +0.025] | **+0.029 [+0.011, +0.056]** | **−0.190 [−0.236, −0.143]** | +0.001 [−0.000, +1.032] | −0.031 [−0.112, +0.102] |
| nose-nose | v1 | 0.653 [0.595, 0.803] | 0.098 [0.043, 0.258] | 0.092 [0.065, 0.127] | 0.151 [0.097, 0.226] | 0.021 (2.5 s) | 0.232 [0.120, 0.381] |
| nose-nose | v2 | 0.809 [0.733, 0.829] | 0.210 [0.147, 0.328] | 0.112 [0.086, 0.179] | 0.180 [0.091, 0.252] | 0.019 (2.3 s) | 0.263 [0.192, 0.340] |
| nose-nose | v2 − v1 | **+0.126 [+0.068, +0.182]** | **+0.107 [+0.031, +0.186]** | **+0.029 [+0.001, +0.061]** | +0.011 [−0.039, +0.064] | −0.002 [−0.005, +0.000] | +0.032 [−0.066, +0.140] |
| nose-nose-mutual | v1 | 0.697 [0.660, 0.822] | 0.046 [0.027, 0.153] | 0.033 [0.026, 0.048] | 0.210 [0.136, 0.297] | 0.007 (0.9 s) | 0.311 [0.263, 0.373] |
| nose-nose-mutual | v2 | 0.851 [0.839, 0.865] | 0.114 [0.083, 0.157] | 0.046 [0.041, 0.076] | 0.205 [0.105, 0.267] | 0.007 (0.8 s) | 0.358 [0.168, 0.391] |
| nose-nose-mutual | v2 − v1 | **+0.148 [+0.090, +0.200]** | **+0.059 [+0.013, +0.110]** | **+0.015 [+0.003, +0.032]** | −0.007 [−0.058, +0.040] | −0.000 [−0.001, +0.000] | +0.021 [−0.099, +0.134] |
| nose-nose-passive | v1 | 0.636 [0.610, 0.793] | 0.050 [0.039, 0.163] | 0.057 [0.046, 0.100] | 0.181 [0.099, 0.290] | 0.017 (2.0 s) | 0.162 [0.130, 0.299] |
| nose-nose-passive | v2 | 0.771 [0.729, 0.809] | 0.138 [0.077, 0.200] | 0.077 [0.066, 0.119] | 0.147 [0.039, 0.259] | 0.016 (1.9 s) | 0.169 [0.129, 0.240] |
| nose-nose-passive | v2 − v1 | **+0.110 [+0.053, +0.167]** | **+0.069 [+0.010, +0.134]** | **+0.021 [+0.001, +0.045]** | −0.029 [−0.092, +0.038] | +0.000 [−0.002, +0.002] | −0.009 [−0.104, +0.078] |

**Constant-rate reference, calibrated-MAE scale (fraction of video):**
- nose-tail 0.0120;
- nose-nose 0.0210;
- nose-nose-mutual 0.0080;
- nose-nose-passive 0.0166.

Both models' calibrated time MAE sits at or just under this reference.

**Raw time bias, sum P(on), seed median:**
- v1 overshoots by +18 to +29 s per 120 s window.
- v2 overshoots by +4 s (nose-tail) to +25 s (nose-nose-mutual).
- The true times are 1-3 s.

**Spearman rho of per-video time, seed median, v1 / v2:**
- nose-tail 0.27 / 0.36;
- nose-nose 0.31 / 0.40;
- nose-nose-mutual 0.40 / 0.43;
- nose-nose-passive 0.26 / 0.26.

## Ants

### Ground truth

| behaviour | series pooled | videos | frames | rate overall | per-video rate, mean ± SE | true time per video, s, mean ± SE (video 600 s) |
|---|---|---|---|---|---|---|
| grooming-to-focal (any) | v2, v3, v4, v4_infested | 212 | 127200 | 0.3389 | 0.3389 ± 0.0172 | 203.4 ± 10.3 |
| Y2F | v2, v3, v4, v4_infested | 212 | 127200 | 0.2082 | 0.2082 ± 0.0142 | 124.9 ± 8.5 |
| B2F | v2, v3, v4, v4_infested | 212 | 127200 | 0.1957 | 0.1957 ± 0.0133 | 117.4 ± 8.0 |
| focal single (one ant grooms the focal) | v2, v3, v4, v4_infested | 212 | 127200 | 0.2740 | 0.2740 ± 0.0128 | 164.4 ± 7.7 |
| focal double (two ants groom the focal) | **v3, v4 only** (eligible only there) | 147 | 88200 | 0.0768 | 0.0768 ± 0.0096 | 46.1 ± 5.8 |

### Main table

Same layout as the mice table: seed median [min, max], and v2 − v1 [95% CI].

| behaviour | model | AUROC | precision@100 | per-frame F1 | raw time MAE, sum P(on), frac of video | calibrated time MAE, frac (s) | Pearson r, per-video time |
|---|---|---|---|---|---|---|---|
| grooming-to-focal (any) | v1 | 0.890 [0.793, 0.919] | 0.952 [0.916, 0.985] | 0.660 [0.572, 0.691] | 0.147 [0.138, 0.173] | 0.130 (78 s) | 0.811 [0.660, 0.842] |
| grooming-to-focal (any) | v2 | 0.879 [0.872, 0.890] | 0.848 [0.824, 0.892] | 0.664 [0.648, 0.673] | 0.140 [0.126, 0.147] | 0.117 (70 s) | 0.798 [0.786, 0.826] |
| grooming-to-focal (any) | v2 − v1 | +0.005 [−0.020, +0.034] | **−0.102 [−0.164, −0.040]** | +0.010 [−0.018, +0.039] | −0.011 [−0.024, +0.001] | −0.016 [−0.031, +0.001] | +0.013 [−0.036, +0.059] |
| Y2F | v1 | 0.850 [0.777, 0.885] | 0.908 [0.635, 0.954] | 0.582 [0.534, 0.617] | 0.113 [0.098, 0.144] | 0.110 (66 s) | 0.710 [0.638, 0.752] |
| Y2F | v2 | 0.831 [0.827, 0.844] | 0.692 [0.570, 0.754] | 0.551 [0.507, 0.562] | 0.120 [0.115, 0.138] | 0.118 (71 s) | 0.668 [0.630, 0.701] |
| Y2F | v2 − v1 | −0.009 [−0.030, +0.014] | **−0.185 [−0.271, −0.085]** | **−0.035 [−0.061, −0.007]** | +0.008 [−0.004, +0.021] | +0.008 [−0.002, +0.018] | −0.034 [−0.084, +0.013] |
| B2F | v1 | 0.845 [0.766, 0.873] | 0.921 [0.720, 0.952] | 0.559 [0.483, 0.595] | 0.113 [0.103, 0.148] | 0.105 (63 s) | 0.687 [0.555, 0.702] |
| B2F | v2 | 0.836 [0.832, 0.845] | 0.657 [0.619, 0.712] | 0.548 [0.538, 0.556] | 0.118 [0.115, 0.124] | 0.108 (65 s) | 0.670 [0.664, 0.678] |
| B2F | v2 − v1 | +0.002 [−0.020, +0.026] | **−0.210 [−0.307, −0.102]** | −0.007 [−0.036, +0.022] | +0.002 [−0.011, +0.013] | +0.000 [−0.009, +0.009] | +0.007 [−0.049, +0.063] |
| focal single | v1 | 0.806 [0.738, 0.833] | 0.109 [0.055, 0.636] | 0.508 [0.467, 0.563] | 0.124 [0.115, 0.150] | 0.133 (80 s) | 0.675 [0.573, 0.714] |
| focal single | v2 | 0.812 [0.800, 0.822] | 0.323 [0.290, 0.584] | 0.541 [0.520, 0.555] | 0.110 [0.097, 0.114] | 0.115 (69 s) | 0.702 [0.687, 0.745] |
| focal single | v2 − v1 | +0.014 [−0.008, +0.035] | +0.143 [−0.021, +0.307] | **+0.030 [+0.006, +0.055]** | **−0.019 [−0.031, −0.008]** | **−0.019 [−0.032, −0.007]** | +0.043 [−0.009, +0.089] |
| focal double (v3, v4) | v1 | 0.925 [0.841, 0.947] | 0.825 [0.537, 0.897] | 0.440 [0.366, 0.520] | 0.188 [0.092, 0.205] | 0.059 (35 s) | 0.715 [0.644, 0.737] |
| focal double (v3, v4) | v2 | 0.920 [0.907, 0.942] | 0.527 [0.442, 0.735] | 0.456 [0.363, 0.554] | 0.131 [0.064, 0.259] | 0.059 (36 s) | 0.659 [0.622, 0.773] |
| focal double (v3, v4) | v2 − v1 | +0.011 [−0.013, +0.035] | **−0.207 [−0.345, −0.042]** | +0.013 [−0.045, +0.079] | −0.012 [−0.075, +0.050] | −0.002 [−0.008, +0.005] | −0.029 [−0.106, +0.042] |

**Constant-rate reference, calibrated-MAE scale (fraction of video):**
- grooming-to-focal (any) 0.209;
- Y2F 0.175;
- B2F 0.159;
- focal single 0.155;
- focal double 0.093.

**Raw time bias, sum P(on), seed median, v1 / v2:**
- grooming-to-focal (any): −66 s / −56 s. Both under-predict.
- Y2F: +10 / +24 s.
- B2F: +18 / +30 s.
- focal single: −24 / −17 s.
- focal double: +113 / +72 s.

**Spearman rho of per-video time:** 0.66 to 0.83 for both models.

The ants focal "or" label is the same as grooming-to-focal (any), the `groom_any` cell.

## What the selected neurons are

- **The same neuron is selected for every behaviour.**
  - Ants: within a seed, both models usually pick the **same** neuron for grooming-to-focal (any), Y2F, B2F and focal single. In 9 of 11 v1 seeds it is also the same for focal double.
  - v2 picks a **different**, dedicated neuron for focal double in every seed.
  - Mice: v2 picks one neuron for nose-nose, nose-nose-mutual and nose-nose-passive in 4 of 6 seeds, stable in 12 of 12 directions. v1's picks are less stable across directions.
  - So no model has a neuron that separates Y2F from B2F, or the nose-nose sub-types.
- **Figures** (one neuron per behaviour = the modal pick of the seed; v1 = v1_s0, v2 = V_s1001; frames drawn from both halves, so they are illustrations only):
  - `sheet_mice.png`
  - `sheet_ants.png`
  - Each frame's metadata, including its P(on), label and code, is in `sheet_<dom>.json`.
  - Top frames are chosen with at most 2 per video, at least 5 s apart (mice) or 10 s apart (ants).
  - Each frame is cropped around the max-activation patch (cyan) with the neuron's per-patch code as a heat overlay.
  - The v1 image is the 512 px jpg. The v2 image is the source frame decoded at the token store's geometry. Mice frames are turned by the odour corner.
  - In the mice v2 frames P(on) is saturated at 1.00, so ties are broken by the raw code.
- **Number of top-12 frames carrying the true label, v1 / v2:**
  - mice: nose-tail 3 / 0, nose-nose 1 / 3, nose-nose-mutual 0 / 2, nose-nose-passive 1 / 1;
  - ants: any 11 / 10, Y2F 9 / 9, B2F 11 / 7, focal single 2 / 4, focal double 9 / 3.

## Files

- `scripts/eci/v2/v21_extended.py`: per-seed selection, metrics and identity check (`tag`), plus the report (`report`).
- `scripts/eci/v2/v21_extended_sheet.py`: the figures.
- `scripts/eci/v2/v21_extended.sh` and `scripts/eci/v2/v21_extended_sheet.sh`: the Slurm jobs. All are CPU jobs (`defaultp`).
- `report.json`: every number, per seed.
- `tables.md`: the full tables, including hard on-frames, bias rows and the selected neurons per seed.
- The per-seed pickles are in `tags/` and are not committed.
- The selected columns (raw code and P(on)) are in `/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_1/extended/`.
