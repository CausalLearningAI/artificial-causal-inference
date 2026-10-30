# Keypoint-guided mask on the benchmark probe (arm KG) — pre-registration

Written 2026-10-30, before any KG mask was built or scored. Code: `scripts/eci/v2/mask5_kpg.py`. Inputs: the arm-B
probabilities already saved for the held-out frames of `scripts/eci/v2/mask5_bench.py`
(/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5_bench/<set>/eval_*.npz) and the benchmark keypoints / tracks.
Outputs: `results/vision/eci_v2_0B/mask_approval_v3/kpg_*`. No store is encoded.

## Build / held-out keypoint split (the mask is built ONLY from the build parts)
- MABe22: build = neck, center, tail_base; held-out = nose, ear_l, ear_r, paw_fl, paw_fr, paw_hl, paw_hr,
  tail_mid, tail_tip.
- CalMS21: build = neck, hip_l, hip_r, tail_base; held-out = nose, ear_l, ear_r.
- CRIM13: one track point per mouse, so no split is possible. KG is built from the point and every CRIM13 metric is
  CIRCULAR (reported, labelled as such, not used as evidence).

## KG mask (per frame; B = mask5 arm B keep, t 0.9, t_core 0.8, no smoothing, no N2 OR)
Per animal, L = |neck - tail_base| (canvas px); u = unit vector neck -> tail_base.
1. Gate: G = union over animals of the convex hull of {build points, tail_base + 1.0 L u} dilated by 0.8 L.
   CRIM13: disk of radius 1.25 x 131 px x (640 / native width) around each track point (131 px = the measured median
   body major axis, bench inventory bodylen).
2. Fill: F = tokens covered >= 25% by capsules of radius 0.75 h along the build skeleton (MABe22 neck-center-tail_base;
   CalMS21 neck-mid(hips)-tail_base and hip_l-hip_r); CRIM13 a disk of radius h around the point
   (h = validate.py body half-width of the set, mask_approval_v2/bench_*.json; MABe22 x 512 / 896).
3. KG = (B AND G) OR F, then 8-connected components without a token of F are dropped.
4. Fallback: a frame where any animal lacks a build point (or, CRIM13, has fewer than 2 track points) gets KG = B.

## Metrics (held-out frames of mask5_bench: CalMS21 456, MABe22 960, CRIM13 384; arms T, B, KG)
- held-out-part recall: share of HELD-OUT keypoints whose token is kept (per part and over all held-out parts);
- per-animal miss rate: share of (frame, animal) with no held-out keypoint in a kept token;
- background tokens / frame and isolated background components / frame: as mask5_bench (kept tokens with no pixel of
  the all-keypoint hull dilated by 1.5 h; CalMS21 + tail-base disk; CRIM13 6 h disk). Note: the hull uses held-out
  points too, so it is not built by KG;
- tokens / frame; store projection (union t-2..t+2 at 5 fps, keypoints of each window frame, fallback where missing).

## Decision rule (per benchmark, MABe22 and CalMS21 only)
KG is adopted over B iff KG held-out recall >= B held-out recall - 0.005 AND KG animal miss <= B AND KG background
tok/frame < B. CRIM13 gets no decision (circular).
