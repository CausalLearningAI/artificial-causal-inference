# Mask v5 (cheap, no SAM at inference) — pre-registration

Written 2026-10-30, before any test-split number exists. Code: `scripts/eci/v2/mask5_val.py`. Outputs:
`results/vision/eci_v2_0B/mask_approval_v3/`. No store is encoded.

## Split (fixed now, `mask_approval_v3/split.json`)
The mask-v4 gold set (20 frames per domain; SAM 2.1 large masks from hand prompts + 410 hand thin points), split per
domain with rng 4 into 10 dev / 10 test frames. Tuning (thresholds, post-processing choices, route choice) uses dev
only. Test is scored once, at the end.

## Candidate route (b): per-domain linear probe on the store's own tokens
- Features: DINOv3-L/16 block-20 tokens (src/eci/v2/encode.Encoder, the exact encode of the planned store).
- Training labels: offline SAM pseudo-labels (mask4 B_l_full, the N2 token rule on its pixel mask) on frames from the
  mask3 metric sample (`mask3_val.sample_frames`, rng 0) of videos that contain NO test gold frame; gold frames
  themselves are excluded too.
- Model: L2-regularised logistic regression per domain (standardised tokens).
- Inference: p = probe probability; p_s = 3 x 3 mean of p; keep = p_s >= t; optional OR with N2 thin tokens
  (patches with >= 16 N2 appendage pixels) 8-adjacent to keep; 8-connected components of keep without any token
  p >= t_core are dropped.
- Tuned on dev per domain over t in {0.2, 0.3, 0.4, 0.5, 0.6}, t_core in {0.5, 0.8, 0.9, 0.95}, N2-thin-OR in {off, on}:
  maximise dev token F1 among settings with dev thin recall >= N2 and dev background tokens/frame <= N2; if none
  qualifies, maximise dev token F1.
- Route (a) (tuned classical) is tried only if time permits; if it is, it is tuned on dev the same way, and the route
  per domain is the one with the better dev token F1 under the same constraints.

## Metrics (test split, per domain; token = 16 px patch; gold tokens = N2 token rule on the gold pixel mask)
token F1 (recall / precision); thin-part recall (share of hand thin points whose token is kept); background tokens
per frame (kept tokens with < 5% gold-animal pixels); isolated background components per frame (8-connected
components of kept tokens with no token >= 5% gold-animal pixels). Compared: new vs N2 vs B_s_crop (SAM reference).

## Decision rule (per domain, on test)
PASS iff new token F1 > N2 AND new thin recall > N2 AND new background tokens/frame <= N2.

## Also reported
Extra per-frame cost of the new mask; on 5-frame windows (t-2..t+2) around the gold and sheet frames: the 5-frame
union of the new mask / new mask at t (token ratio), the share of the new mask at t inside the previously proposed
store S (union of N2 dilated by 1 token), and the projected store size (union tokens/frame x 1024 x 2 bytes x frames
at 5 fps: mice 3.89 M, ants 1.82 M, frogs 2.76 M).

## POST-HOC AMENDMENT 1 (2026-10-30 ~00:25, AFTER the ants test numbers of the pre-registered grid were seen)
Ants failed: no dev setting met background <= N2 (the grid stopped at t = 0.6 and always smoothed). Amendment: a
second, extended dev grid t in {0.5, 0.6, 0.7, 0.8, 0.9}, t_core in {0.8, 0.9, 0.95, 0.98}, N2-thin-OR {off, on},
3x3 smoothing {on, off}, same dev selection rule, scored on the same test split. Reported separately and labelled
post hoc ("ext"); the pre-registered results stand as the primary verdict.
