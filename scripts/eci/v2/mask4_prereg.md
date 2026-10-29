# Mask v4 bake-off (SAM 2.1 prompted by the N2 cue) — pre-registration

Written 2026-10-29, before any gold-set metric was computed. Code: `src/eci/v2/mask4.py` (candidate),
`scripts/eci/v2/mask4_val.py` (capture, arms, gold, metrics, bench, sheets). Outputs:
`results/vision/eci_v2_0B/mask_approval_v2/`. No DINOv3 token is encoded or stored.

## Candidates
All candidates take the same input (the frame at the encoder input side S, canonical orientation, exactly as the frozen
v2.0 N2 path produces it, via `mask3_val.series_pair`, identity-asserted against `v20.N2Series.n2`).

| name | what |
|---|---|
| N2 | frozen v2.0 mask (`mask2.py`, r25t token rule) — the reference |
| mask3b | `mask3.keep3b` (post-hoc amendment 1 of the mask3 test) |
| A_s_full / A_l_full | SAM 2.1 small / large, image predictor, whole frame, prompts = N2 cue components (box + body points) |
| A_s_crop / A_l_crop | the same, one SAM pass per prompt object on a zoomed square crop |
| A_s_video | SAM 2.1 small video predictor: prompts on the first frame of a 10-frame clip (2 s at 5 fps) ending at the scored frame, propagated to it (= the worst case of re-prompting every 10 frames) |
| B_* | hybrid: A_* ∪ N2 appendage pixels whose connected component touches the SAM mask; components without a core pixel dropped |

Parameters of `mask4.py` (MIN_BODY_H2 0.5, PAD_H 0.5, PTS_AREA_H2 8, MAX_PTS 4, MAX_GROW 3, MIN_COVER 0.5, CROP_MIN_H 4,
CROP_PAD_H 1) are the first values written; development looked only at one hard-case frame (ants v2 b6 k1500, a smoke
test) and changed nothing. They are frozen at the commit of this file. No arm output on a gold frame was viewed before
this commit. Every candidate uses the N2 token rule (≥ 25% mask pixels OR ≥ 16 appendage pixels of its own mask).

## Gold set (lab)
Frames: per series the old approval sheet's random frames plus further frames from the old metric sample
(`mask3_val.sample_frames`, rng 0) drawn with rng 2: mice v1 10, v2 10; ants v2 7, v3 7, v4 6; frogs juv 10, tad 10
(= 20 per domain). Hard cases are not gold frames.

Gold prompts as built (`gold/build_prompts.py`, `gold/thin_manual.json`, `gold/prompts.json`): ants / frogs = one
positive point at the deepest body pixel of each N2 prompt object (the location seed) + every hand-placed thin point
as a further positive point of the nearest object, NO box (the arms use the box); mice = every mouse, its body
point(s) and tail points placed by hand (N2 seeds unused: they missed whole resting mice in 5 of 20 frames and seeded
wall shadows in 3). Hand corrections after viewing every gold mask (`manual` entries): 3 off-animal thin points
removed (ants v2 b1_2580, e2_2826, v3 3_6_3_1661), one inverted mask re-picked (v3 3_25_5_2044), box prompts added to
4 ants v4 frames and 1 tadpole with background speckle, a negative point + moved tail tip (tadpole FoxP1_170_6) and
eye points (tadpole WT_178_5). Tadpole En1_300_6_16028 lies under the dish wall: hand body point only (no thin).

Gold masks: SAM 2.1 LARGE on per-animal zoomed crops with prompts placed BY HAND (Claude, from viewing the frame:
positive points on each body and on visible thin parts — tail, legs, antennae, limbs; negative points on background
that SAM wrongly includes), multimask output chosen by hand, every gold mask viewed (frame + overlay + zoom) and
re-prompted until it matches the animal as judged by eye. Where SAM cannot draw a thin part, the thin part is
recorded only as hand-placed THIN POINTS (below). Thin points: hand-placed points on visible thin parts (mouse tail:
mid and tip; ant: antenna tips and leg tips that are visible; frog: limb tips / tadpole tail tip), independent of any
mask. Known bias: the gold pixel masks come from SAM (large, hand prompts), so pixel IoU / token metrics favour SAM
arms; the thin-point metric does not depend on SAM.

## Metrics (per domain = pooled over its series' gold frames; and per series)
Token rule for gold tokens: the N2 rule applied to the gold pixel mask.
1. pixel IoU vs gold (pixels pooled over frames).
2. token recall = |kept ∩ gold tokens| / |gold tokens|; token precision = |kept ∩ gold tokens| / |kept|; token F1.
3. thin-part recall (token) = share of thin points whose 16-px token is kept; (pixel) = share of thin points with a
   mask pixel within 3 px.
4. background patches kept = kept tokens with < 5% gold-animal pixels, per frame (mean) and as a share of kept tokens.
5. tokens per frame (mean).

Benchmarks (`scripts/eci/v2_mask2/validate.py` frames, keypoints, `score_frame` unchanged; video arm not run there):
MABe22 tail_seg_tok, tail_tip_tok, tail_base_tok, body_tok, leak, tokens/frame; CalMS21 pale_tok, pale_missed,
tail_base_tok_int / res, body_tok, leak; CRIM13 pale_tok, pale_tok_tol1, res_tok, c57_tok, leak. N2 must reproduce
`results/vision/eci_v2_mask2/validate/<set>_final.json`.

## Decision rule
Per domain (mice, ants, frogs) and overall (mean over the 3 domains): the WINNER is the candidate (A_* / B_*) with the
best overall token F1 (mean of the three domain F1s). It is ADOPTED (proposed to the user) only if, in every one of
the three domains, it beats N2 on BOTH thin-part recall (token) AND background patches kept per frame (strictly
better). Margins are reported as differences to N2 with a frame-bootstrap 95% interval (1000 resamples of gold
frames, seed 0). A tie or a loss in any domain → not adopted, reported as such. The user approves the final choice
from the sheets.

## Device
The L40S / A100 / A40 queue estimated a start 1.5 days out, so gold masks and all arm masks used for the metrics are
computed on the debug partition (RTX 2080 Ti, fp16 autocast; bf16 is not supported there). Throughput is reported
from an L40S / A100 / A40 job (bf16) when it runs, together with the token agreement of its masks with the fp16 masks
on the same frames; if it has not run by the report, the 2080 Ti numbers are reported and labelled as such.

## Cost
Throughput (s / frame for the mask alone, decode excluded, median over the captured frames, GPU name recorded) per
arm and domain; GPU-h for every analysed video at 5 fps: mice 3.89 M, ants 1.82 M, frogs 2.76 M frames.
