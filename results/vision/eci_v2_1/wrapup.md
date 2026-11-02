# ECI v2 wrap-up: the v1 -> v2 evidence ladder (2026-11-02)

Pre-registration: AMENDMENT 4 in `scripts/eci/v2/v21_train.py` (commit 03dbf92, before any fresh-seed SAE, focal score,
top-10 code of W/B/SN2 or bootstrap number). Code: `scripts/eci/v2/v21_ladder.py`, `v21_cells.py`. Full tables:
`ladder/ladder.md` (frame-max readout), `ladder/ladder_top10.md` (top-10 readout), `ladder/cells.md` (per cell).
CalMS21 check: `bench/cal_report.md` (commits f03776f prereg, 2044a9e results).

## Bottom line

- **Mice: v2 is clearly better than v1.** Mean AUROC P(on) rises 0.674 -> 0.812 (V; v1 = median of 11 seeds).
  - Almost all of the gain comes from the DINOv3 native-resolution step.
  - Temporal context (V) adds more on top of the static arm.
  - Mask v5 HURTS the static arm on mice.
- **Ants: v2 is not better than v1 on any step.**
  - Mean AUROC is flat everywhere.
  - Top-1% precision is lower than v1 under the frame-max readout.
  - The top-10 readout closes most of that precision gap, but not all of it.
- **Pre-registered confirmation on fresh seeds:**
  - V beats S on mice in direction and in top-1% precision.
  - On mean AUROC P(on), the CI that resamples videos and seeds includes 0, so C1 is NOT confirmed.
  - V is significantly better than v1 on mice.
  - On ants, non-inferiority to v1 (C4) is NOT demonstrated: the CIs are too wide.
- **The pre-registered frozen-arm rule picks W.** Gate passes under top-10, median over the fresh seeds, mice + ants:
  W 10, V 9, S 10. But W fails its own confirmation tests. The rule's "temporal helps" condition is therefore NOT met
  (see the recipe section below).

## Mice ladder

Pre-registered P(on) readout. Values are the median over seeds. Δ = difference of the seed means, with the 95% CI
from 2000 bootstrap replicates that resample pools AND seeds.

| rung | mean AUROC (max) | top-1% prec (max) | mean AUROC (top10) | top-1% prec (top10) | gate passes / seed (max) | distinct nn/nt |
|---|---|---|---|---|---|---|
| R0 v1 (11 seeds) | 0.674 | 0.061 | 0.674 | 0.061 | - | 5/11 |
| R1 PB (v2.0, 1 model) | 0.786 | 0.167 | (max only) | | [4] | 1/1 |
| R1 SN2 (DINOv3 static, N2 mask) | 0.768 | 0.082 | 0.755 | 0.071 | [3,4,4] | 2/3 |
| R2 S (mask v5) | 0.737 | 0.095 | 0.737 | 0.062 | [4,4,4] | 0/3 |
| R3 V | 0.812 | 0.137 | 0.801 | 0.128 | [4,4,4] | 2/3 |
| R3 W | 0.752 | 0.079 | 0.740 | 0.062 | [4,3,4] | 3/3 |
| ref B | 0.786 | 0.116 | 0.783 | 0.074 | [4,4,4] | 1/3 |
| S fresh (1004-6) | 0.738 | 0.063 | 0.725 | 0.047 | [4,4,3] | 3/3 |
| V fresh | 0.781 | 0.117 | 0.784 | 0.116 | [4,4,4] | 3/3 |
| W fresh | 0.746 | 0.078 | 0.746 | 0.087 | [4,4,4] | 2/3 |

| contrast (mice, frame max) | Δ mean AUROC [CI] | Δ top-1% prec [CI] | Δ nAP [CI] |
|---|---|---|---|
| SN2 − v1 (DINOv3 native res) | +0.085 [+0.058, +0.111] | +0.026 [+0.001, +0.059] | +0.021 [+0.006, +0.040] |
| S − SN2 (mask v5) | **−0.034 [−0.057, −0.011]** | +0.011 [−0.018, +0.037] | −0.003 [−0.020, +0.012] |
| V − S (temporal) | +0.070 [+0.044, +0.097] | +0.039 [+0.011, +0.074] | +0.020 [+0.001, +0.039] |
| W − S | +0.015 [−0.025, +0.052] | +0.002 [−0.039, +0.057] | +0.002 [−0.021, +0.030] |
| V − v1 | +0.121 [+0.082, +0.158] | +0.076 [+0.042, +0.116] | +0.038 [+0.020, +0.061] |
| fresh V − fresh S (C1) | +0.030 [−0.018, +0.071] (pools only: [+0.014, +0.048]) | +0.041 [+0.007, +0.074] | +0.019 [+0.003, +0.035] |
| fresh W − fresh S (C2) | −0.010 [−0.063, +0.043] | +0.010 [−0.019, +0.034] | +0.005 [−0.008, +0.018] |
| fresh V − v1 (C3) | +0.104 [+0.067, +0.140] | +0.052 [+0.024, +0.088] | +0.025 [+0.009, +0.042] |
| fresh W − v1 (C3) | +0.063 [+0.021, +0.106] | +0.020 [+0.001, +0.043] | +0.011 [−0.000, +0.021] |
| top10: fresh V − fresh S | +0.041 [−0.014, +0.086] | +0.046 [+0.001, +0.091] | +0.020 [+0.001, +0.042] |

## Ants ladder

Gate cells are Y2F / B2F / groom_any × series v2, v3, v4 and v4_infested. Units = videos, resampled within series.

| rung | mean AUROC (max) | top-1% prec (max) | mean AUROC (top10) | top-1% prec (top10) | gate passes / seed (max; top10) | focal single / double / or AUROC (max) |
|---|---|---|---|---|---|---|
| R0 v1 (11 seeds) | 0.858 | 0.876 | = | = | - | 0.796 / 0.912 / 0.882 |
| R1 PB | 0.839 | 0.785 | (max only) | | [4] | 0.801 / 0.924 / 0.865 |
| R1 SN2 | 0.837 | 0.745 | 0.836 | 0.787 | [4,7,2]; [6,11,4] | 0.807 / 0.912 / 0.871 |
| R2 S | 0.837 | 0.760 | 0.835 | 0.790 | [3,2,6]; [5,6,7] | 0.805 / 0.897 / 0.870 |
| R3 V | 0.848 | 0.724 | 0.842 | 0.822 | [3,2,2]; [11,8,6] | 0.813 / 0.898 / 0.882 |
| R3 W | 0.844 | 0.770 | 0.842 | 0.788 | [11,4,6]; [8,6,6] | 0.807 / 0.927 / 0.872 |
| ref B | 0.826 | 0.688 | 0.829 | 0.707 | [1,6,1]; [0,11,0] | 0.804 / 0.870 / 0.866 |
| S fresh | 0.835 | 0.755 | 0.828 | 0.807 | [2,8,3]; [6,8,7] | 0.804 / 0.879 / 0.870 |
| V fresh | 0.834 | 0.736 | 0.830 | 0.786 | [5,3,3]; [8,5,5] | 0.806 / 0.910 / 0.870 |
| W fresh | 0.833 | 0.719 | 0.825 | 0.761 | [3,4,1]; [7,7,3] | 0.801 / 0.876 / 0.866 |

(The v1 median of 11 seeds is shown: mean AUROC 0.858, top-1% precision 0.876. The earlier reports quoted the v1 seed
mean, 0.845 / 0.840. All Δ below are differences of seed means.)

| contrast (ants) | Δ mean AUROC max [CI] | Δ top-1% prec max [CI] | Δ mean AUROC top10 [CI] | Δ top-1% prec top10 [CI] |
|---|---|---|---|---|
| SN2 − v1 | −0.009 [−0.036, +0.019] | −0.087 [−0.140, −0.016] | −0.011 [−0.038, +0.016] | −0.044 [−0.103, +0.022] |
| S − SN2 (mask v5) | +0.002 [−0.014, +0.018] | +0.010 [−0.039, +0.058] | +0.002 [−0.012, +0.017] | −0.002 [−0.047, +0.052] |
| V − S | +0.005 [−0.005, +0.014] | −0.028 [−0.063, +0.003] | +0.003 [−0.007, +0.014] | +0.020 [−0.020, +0.045] |
| W − S | +0.005 [−0.002, +0.012] | +0.017 [−0.029, +0.066] | +0.006 [−0.003, +0.014] | +0.003 [−0.028, +0.039] |
| V − v1 | −0.002 [−0.026, +0.022] | −0.105 [−0.145, −0.044] | −0.006 [−0.030, +0.017] | −0.027 [−0.068, +0.029] |
| fresh V − fresh S | −0.002 [−0.013, +0.010] | −0.018 [−0.066, +0.032] | −0.003 [−0.015, +0.011] | −0.003 [−0.035, +0.035] |
| fresh V − v1 (C4: lower bound ≥ −0.01?) | −0.009 [−0.033, +0.015] NO | −0.093 [−0.137, −0.031] | −0.015 [−0.038, +0.010] NO | −0.040 [−0.084, +0.017] |
| fresh W − v1 (C4) | −0.020 [−0.048, +0.006] NO | −0.129 [−0.169, −0.070] | −0.019 [−0.044, +0.005] NO | −0.110 [−0.192, −0.029] |
| fresh W − fresh S | −0.013 [−0.030, +0.002] | −0.054 [−0.097, −0.014] | −0.007 [−0.023, +0.008] | −0.073 [−0.146, −0.009] |

**Focal grooming cells** (reported cells, not gate cells; frame-max readout, Δ vs v1 with videos+seeds CI):
- **single** (4 series):
  - Every v2 rung has a much higher top-1% precision than v1: v1 0.273; SN2 0.474, Δ +0.160 [+0.056, +0.242];
    V 0.457, Δ +0.199 [+0.086, +0.290].
  - AUROC +0.01 to +0.02, not significant.
  - Under top-10 the precision gain shrinks to +0.03 to +0.10.
- **double** (v3, v4 only):
  - AUROC is equal to v1 (v1 0.912; W 0.927, V 0.898). W − V +0.035 [+0.006, +0.066].
  - Top-1% precision is lower than v1: V −0.159 [−0.286, −0.037].
- **or** (= groom_any): AUROC and top-1% precision are equal to v1 within CI on every step. Top-1% precision is −0.03
  to −0.05 under max, not significant.

**Per cell** (`ladder/cells.md`):
- Under max, the ants top-1% precision loss vs v1 is spread across the Y2F / B2F cells of all series. It is largest
  on v3|Y2F (v1 0.804, V 0.583) and v4|B2F (0.829 vs 0.597).
- Top-10 recovers most of v3 (V v3|Y2F 0.737, v3|B2F 0.841) and v4_infested.
- v4 Y2F / B2F stay below v1 in AUROC for every arm (v1 0.810 / 0.789; V 0.770 / 0.748). No step helps v4.

## Pre-registered outcomes (AMENDMENT 4)

- **R, readout:** top-10 is ADOPTED for all token-code arms. Over the 9 fresh models (median):
  - ants mean top-1% precision P(on), top10 − max = +0.047 (rule: ≥ +0.03);
  - mice mean AUROC P(on), top10 − max = −0.009 (rule: ≥ −0.02).
  - v1 and PB stay at frame max (they have stored frame-max codes only).
- **C1** (mice, fresh V − S, mean AUROC > 0): NOT confirmed. +0.030 [−0.018, +0.071]; under top-10 +0.041
  [−0.014, +0.086]. The direction holds, and it is significant on top-1% precision and nAP.
- **C2** (W − S): NOT confirmed (−0.010 [−0.063, +0.043]).
- **C3** (mice V, W > v1): CONFIRMED (+0.104, +0.063; both CIs exclude 0).
- **C4** (ants non-inferior, lower bound ≥ −0.01): NOT demonstrated for V or W under either readout.
- **C5** (ants top-1% precision vs v1, max): V −0.093, W −0.129 (both significant).
- **Frozen arm (rule):** W. Gate passes under top-10, median over fresh seeds of mice + ants: W 10, V 9; S also 10.
  "Temporal helps" is NOT supported by the rule's own criterion, because W fails C2 and C4.

## The story, tested

1. **Step 1 (v1 works) holds.**
   - Mice: v1 is weak (mean AUROC P(on) 0.674, top-1% precision 0.061).
   - Ants: v1 is strong (0.858 / 0.876, median of 11 seeds) and is the bar every v2 arm struggles to match.
2. **Step 2 (DINOv3 native resolution + 5 fps) is supported on mice and NOT on ants.**
   - Mice: SN2 − v1 is +0.085 mean AUROC [+0.058, +0.111].
   - Ants: mean AUROC is unchanged (−0.009 [−0.036, +0.019]) and top-1% precision falls (−0.087 [−0.140, −0.016]).
   - DINOv2 vs DINOv3 is confounded with resolution, mask and frame rate here. The earlier size-matched CalMS21 test
     found DINOv3-L = DINOv2-L at the same 64 grid (+0.017 primary AUROC [−0.006, +0.041]), while the finer grid
     helped (D3L 64 vs 32: +0.031 [+0.005, +0.056]). So "resolution / grid", not "DINOv3", is the likely mice driver.
3. **Step 3a (the better mask, v5) is NOT supported.**
   - Mice: it lowers static alignment (S − SN2 −0.034 [−0.057, −0.011]).
   - Ants: it does nothing (+0.002 [−0.014, +0.018]).
   - The mice gap test (AMENDMENT 3) refuted the "lost bridge patches" explanation, so the cause is open.
   - SN2 misses 14-20% of the real N2 tokens on mice, which makes this contrast conservative, if anything.
4. **Step 3b (temporal context) is supported for V on mice, in three independent looks:**
   - seeds 1001-1003: +0.070 [+0.044, +0.097];
   - fresh seeds: +0.030, CI with seeds [−0.018, +0.071], pools only [+0.014, +0.048]; top-1% precision +0.041
     [+0.007, +0.074];
   - CalMS21 (untuned benchmark): +0.016 [+0.007, +0.026].
   - It is NOT supported on ants (fresh V − S −0.002 [−0.013, +0.010]).
   - W is never better than S on mice, and is worse on ants top-1% precision in fresh seeds (−0.054 [−0.097, −0.014]).
5. **The readout matters as much as the architecture on ants.** Top-10 instead of frame max brings V's ants top-1%
   precision from 0.724 to 0.822 (V − v1: −0.105 -> −0.027 [−0.068, +0.029]). It costs ~0.01 mean AUROC on mice.
6. **The new focal-grooming labels favour v2 for "single groomer" frames.** Top-1% precision is +0.13 to +0.20 vs v1
   (significant). They favour v1 for "double" frames on precision, and are neutral on "or".

## Recommended frozen v2 recipe

- **What the pre-registered rule selects:** W, mask v5, top-10 readout. W tops gate passes by one cell over V (10 vs 9;
  S also 10). But W does not pass any of its own confirmation tests. On fresh seeds it is the weakest arm on ants
  (top-1% precision top10 0.761) and no better than S on mice.
- **My recommendation, which needs the user's sign-off because it deviates from the pre-registered rule:**
  - **input arm V (x + centred 1-frame difference); mask v5; readout P(on) on the top-10 token aggregation; DINOv3-L/16
    block 20, native resolution, 5 fps.**
  - Evidence for V:
    - best mice alignment in every look (seeds 1001-1003, fresh seeds, CalMS21);
    - 4/4 mice gate cells on all 6 seeds;
    - ants within CI of S and v1 on mean AUROC under top-10;
    - highest ants top-1% precision under top-10 on seeds 1001-1003 (0.822).
  - Cost of V: ants fresh-seed top-10 precision is no better than S (0.786 vs 0.807, Δ −0.003 [−0.035, +0.035]).
- **The conservative alternative is S + top-10.** It is static and simplest, and it ties W on gate passes. It is
  defensible if the user wants no temporal claim.
- **Open question on the mask:** mask v5 vs N2 favours N2 on mice (static arms only). A V-on-N2 arm does not exist.

## Limitations

- **The bootstrap holds the neuron pick per cross-fit direction fixed,** so the CIs omit selection variance. Three
  seeds per rung make the seed-resampled CIs crude: v1 has 11 seeds and a wide spread, which dominates the ants CIs.
- **v1 and PB stay at frame max.** Every top-10 comparison against them mixes readout and model.
- **The top-10 readout was chosen post hoc** (after D2). It was confirmed only on fresh seeds of the SAME labels.
  CalMS21 did not test it.
- **The arm candidates (V, W) were chosen on the same labels;** the fresh seeds control for seed luck, not label
  overfitting. CalMS21 is the only untuned check, and only for S / V / W under frame max. Attack has 5 positive
  test videos.
- **SN2 is an approximation of the old-style mask:** it misses 14-20% of the real N2 tokens on mice. PB is a single
  model trained on a different sample (old store).
- **DINOv2 vs DINOv3 is not isolated tonight.** It is confounded with resolution, mask and fps; the CalMS21
  size-matched test is the only evidence.
- **Focal double** is eligible only in v3 and v4. All focal cells are reported, not gate cells.
- **Capacity** (distinct behaviour neurons under transformed frames) was not computed. Distinct nn/nt is reported
  per seed only.
