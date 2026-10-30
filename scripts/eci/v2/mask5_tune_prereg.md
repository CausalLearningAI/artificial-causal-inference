# Benchmark masks: per-benchmark tuned arm B and the final pick — pre-registration

Written 2026-10-30, after the user's decision (keypoints when they help, at minimum as a gate; otherwise the lab-domain
recipe, i.e. arm B with thresholds tuned per benchmark on a dev split). Written before any tuned threshold or any test
number of the tuned arms existed. Code: `scripts/eci/v2/mask5_tune.py`. Outputs:
`results/vision/eci_v2_0B/mask_approval_v3/tune_*`. No store is encoded.

Known before writing this (not hidden): the untuned results of `mask5_bench_prereg.md` and `mask5_kpg_prereg.md` on
ALL held-out frames (MABe22 / CRIM13 KG results; CalMS21 KG was still running). The dev / test split below re-uses
those same frames, so the test half has been seen once in aggregate under the untuned settings.

## Dev / test split (by video, rng 7)
The mask5_bench held-out frames: CalMS21 19 videos -> 10 dev / 9 test; MABe22 120 clips -> 60 / 60; CRIM13 24 videos
-> 6 BC + 6 C57 dev / 6 BC + 6 C57 test.

## Tuned arm B ("Bt")
Grid on dev: t in {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}, t_core in {0.5, 0.8, 0.9, 0.95, 0.98}, 3 x 3 smoothing
{off, on}. The N2 thin-token OR of the lab recipe is NOT in the grid: the saved benchmark data has no N2 appendage
tokens (stated deviation). Selection on dev: maximise keypoint F1 = harmonic mean of (all-keypoint token recall,
region precision = 1 - background tokens / kept tokens) among settings with per-animal miss rate <= N2-pooled dev
miss rate; if none qualifies, maximise keypoint F1.

## Keypoint-guided arm on Bt ("KGt")
KG of `mask5_kpg_prereg.md` (same build parts, gate, fill, component rule, fallback to Bt where build points are
missing) applied on Bt. Implementation change for speed only: the gate dilation uses a Euclidean distance transform
(distance <= 0.8 L) instead of an elliptic structuring element.

## Test metrics (test half, once)
Arms T, B (0.9 / 0.8), Bt, KG (on B), KGt: all-keypoint recall, held-out-part recall (MABe22 / CalMS21 parts not used
to build KG), per-animal miss rate (all parts and held-out parts), background tok/frame, isolated background
components / frame, tok/frame, keypoint F1; store projection (union t-2..t+2) for Bt and KGt.

## Final pick per benchmark
- MABe22, CalMS21: KGt iff on test KGt held-out recall >= Bt - 0.005 AND KGt held-out miss <= Bt AND KGt background
  tok/frame < Bt; else Bt. Frames whose build keypoints are missing use Bt.
- CRIM13: KGt (the user's minimum: keypoints at least gate the bodies), with the statement that CRIM13 has no
  non-circular check; frames with fewer than 2 track points use Bt.
