# Mask v5 on the mouse benchmarks (MABe22, CalMS21, CRIM13) — pre-registration

Written 2026-10-30, before any benchmark probe was trained or any held-out frame scored. Code:
`scripts/eci/v2/mask5_bench.py`. Outputs: `results/vision/eci_v2_0B/mask_approval_v3/bench_*`. No store is encoded.
SAM 2.1 is used OFFLINE only, to label the training frames of arm B. Scoring uses benchmark keypoints only.

## Canvas (= the planned benchmark store encode, scripts/eci/v2_temporal/temporal.py BENCHES)
MABe22 native 224 bicubic to 512 (32 x 32 tokens); CalMS21 1024 padded canvas (64 x 64); CRIM13 640 canvas (40 x 40).
Tokens: src/eci/v2/encode.Encoder (DINOv3-L/16 block 20, rev ea8dc28, bf16 autocast, fp16 tokens).

## Arms
- T (transfer): the approved mice probe as it is (`/mnt/beegfs/locatgrp/shared/rcadei/eci_mask5/mice/probe.npz`),
  approved mice post-processing: keep = p >= 0.9, no 3x3 smoothing, no N2 thin-token OR, 8-connected components with
  no token p >= 0.8 dropped.
- B (per benchmark): the same logistic probe recipe (scripts/eci/v2/mask5_val.fit_probe, same negative subsampling)
  trained on 120 frames of the benchmark's TRAINING split, labels = SAM 2.1 large full-frame B_l_full masks prompted
  by N2 (src/eci/v2/mask4, rule B) passed through the N2 token rule. Same post-processing and thresholds as T (no
  tuning: there is no dev split). Training frames: CalMS21 30 train videos x 4 frames; MABe22 60 mouse_userTrain clips
  x 2 frames; CRIM13 40 train videos (20 BC + 20 C57 intruder) x 3 frames (rng 0).
- N2 (reference): src/eci/v2/mask2 r25t with the POOLED per-setup background
  (/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_bench_bg/<bench>/pool_g*.npy, the v20_bench_store grouping; clips
  outside a setup keep their per-clip background). N2 with the per-clip background is reported too; for CalMS21 /
  CRIM13 it must reproduce the earlier validate.py N2 numbers (identity check). MABe22 N2 is computed at the store
  side 512 (h x 512 / 896), as N2s in v20_bench_store.
- The body half-width h is the earlier validate.py value per set (mask_approval_v2/bench_*.json).

## Held-out frames (disjoint from all B training clips: different official split)
validate.py's frames: CalMS21 all 19 task-1 TEST videos x 24 frames; MABe22 120 mouse_submission clips x 8 frames;
CRIM13 24 TEST videos (12 BC + 12 C57 intruder) x 16 frames.

## Metrics (per arm, frame-averaged; token = 16 px patch)
- keypoint recall per body part = share of keypoints whose token is kept (CalMS21 7 parts x resident / white
  intruder; MABe22 12 parts incl. paws, tail base / mid / tip; CRIM13 one track point per mouse, strict and with a
  1-token tolerance); all-keypoint recall = over all keypoints.
- per-animal miss rate = share of (frame, animal) with no keypoint in a kept token (CRIM13: no kept token within 1
  token of the track point). Reported per animal type (CalMS21 white intruder separately).
- background tokens / frame = kept tokens containing no pixel of the animal region; region = union over animals of
  the keypoint convex hull dilated by a margin of 1.5 h; CalMS21 adds a disk at the tail base of radius nose-to-tail
  base distance (tails are not keypointed); CRIM13 (one point per mouse) a disk of radius 6 h around each track point.
- isolated background components / frame = 8-connected components of kept tokens that are all background.
- tokens / frame.

## Decision rules (per benchmark, held-out frames)
1. Arm choice: B is chosen iff B all-keypoint recall > T AND B per-animal miss rate <= T AND B background tok/frame
   <= T + max(1, 0.1 x T). Otherwise T.
2. Pass / fail of the chosen arm vs N2 (pooled): PASS iff all-keypoint recall > N2 AND per-animal miss rate <= N2
   AND background tok/frame <= N2.

## Also reported
Store projection for the chosen arm: tokens in the union of the masks at t-2..t+2 (5 fps steps: 6 source frames for
MABe22 / CalMS21, 5 for CRIM13) around every held-out frame, x 1024 x 2 bytes x the 5 fps frame count of the whole
benchmark (bench_store/size.json); also with that union dilated by 1 token. Approval sheets bench_<name>.png.
