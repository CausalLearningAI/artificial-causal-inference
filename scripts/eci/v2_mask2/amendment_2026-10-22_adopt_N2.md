# Amendment 2026-10-22: carry arm N2 to step 2 (user decision), release criteria

Pre-registration amended: `scripts/eci/v2_mask2/mask2_test.py` (docstring, commits f74169c, 02786b9).
Written on 2026-10-22, after the step-1 results (commit 6c66194, `results/vision/eci_v2_mask2/validate/step1_final.json`)
and BEFORE any step-2 score: no step-2 SAE had been trained and no `score/`, `neurons/` or `decision.json` file
of step 2 existed when this file was committed.

## What changes

1. **Step-1 gate waived for N2 (user decision, 2026-10-22).** Step 1 failed for both arms on two
   criteria: (1b) CRIM13 pale-intruder token recall +0.057 (rule +0.10), and (1d) keypoint-region leak 0.18 on CRIM13 /
   0.22 on CalMS21 (rule <= 0.05). N2 passed (1a) (MABe22 tail-segment token recall +0.204, rule +0.20), (1b) on CalMS21
   (+0.197) and (1c). The user adopts N2 as the step-2 arm anyway. The reasons:
   - (i) **CRIM13 pale mouse.** On CRIM13 the albino (BC) intruder sits on light bedding and has the same grey level as
     the floor. No pixel-based cue can detect it, so this criterion cannot be met by any pixel-cue arm. It does not
     separate good from bad cues.
   - (ii) **The absolute leak cap (<= 0.05) was invalid.** Leak was measured against a keypoint-hull "animal region"
     (capsules around keypoints) that misses real body edges. M0 itself scores 0.15 (CalMS21) and 0.18 (CRIM13) on that
     measure, so the cap fails the current mask too. Visual checks of the lab videos show 0 off-animal tokens. The
     independent lab leak measure (kept tokens with < 5% of their pixels in the frozen cue or the raw change map) is
     0.000 for N2 in mice, ants v2 and ants v3 (`results/vision/eci_v2_mask2/lab_coverage.json`).
   So criterion (i) of the decision rule ("step 1 passed") becomes "N2 carried by the user decision of 2026-10-22".
   The step-1 numbers stay reported unchanged.

2. **Release criteria made explicit (user's list, 2026-10-22).** These are added to the pre-registered rule; none
   are removed. N2 is released over M0 iff ALL of:
   - (ii) pre-registered: mice primary R~ (mean of nose-nose and nose-tail R~) and ants primary R~ (mean over v2 / v3
     of groom_any R~): the Welch 95% CI lower bound of mean(N2) - mean(M0) is >= -0.01 (20 seeds mice, 33 ants).
   - (ii-user) added: the same non-inferiority test (Welch 95% CI lower bound >= -0.01) separately for mice nose_nose
     R~, mice nose_tail R~ and ants v2 groom_any R~. Ants v3 on_lid_any R~ is reported with its CI and does not
     decide.
   - (iii) unchanged: lab leak <= 0.05 in mice, ants v2 and ants v3 (the lab measure above, NOT the keypoint-hull leak).
     The background-neuron share is <= 0.01 in mice and in ants (mean over all seeds). Background neuron = the mask test 1
     definition (`results/vision/eci_v2_mask/decision.json`, `scripts/eci/v2_mask/mask_test.py`): for each neuron and
     each selection-half frame where it fires, take its argmax token. The neuron is a background neuron iff that token
     is off-animal (< 5% of its pixels in frozen cue | raw change) in > 50% of >= 20 firing frames. The count per arm is
     reported too.
   - (iv) unchanged: >= 5 appendage neurons per domain (mean over seeds) in mice and ants, plus the visual check of
     the deployment seed's top-3 sheets (reported per sheet).

   **Power note, written before scoring.** The seed counts were sized for the primary scores. From mask test 1's
   6-seed standard deviations, the expected CI half-widths at 20 / 33 seeds are: mice nose_nose about 0.020, mice
   nose_tail about 0.010, ants v2 groom_any about 0.017. So the per-label tests can pass only if N2's observed mean
   is at least about +0.010 (nose_nose) or +0.007 (ants v2 groom_any) above M0's. A per-label failure with a point
   estimate near 0 means "not resolved", not "worse". It is still reported as a fail.

3. **Everything else is unchanged:** frames, splits, labels, SAE recipe, seeds (mice 70-89, ants 70-102, the same seeds
   for both arms), the deployment seed rule (highest held-out FVE among 70-72), the token stores already on BeeGFS,
   and the background, appendage and odour neuron definitions.

The discovery comparison (v1 vs M0-v2 vs N2-v2: tail and odour-box neurons) is pre-registered on its own, in
`scripts/eci/v2_mask2/discovery.py`, committed before any of its scores.
