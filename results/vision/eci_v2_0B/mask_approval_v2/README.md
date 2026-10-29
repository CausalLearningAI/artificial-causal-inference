# Mask v4 bake-off: SAM 2.1 prompted by the N2 cue vs N2 / mask3b

Pre-registration (committed before any gold metric): `scripts/eci/v2/mask4_prereg.md`. Code: `src/eci/v2/mask4.py`,
`scripts/eci/v2/mask4_val.py`. Full tables: `table.md` (gold set), `bench_{mab,crim,cal}.json` (keypoint benchmarks).
No DINOv3 token was encoded.

## The candidate algorithm (one rule for every domain; only per-domain number = body half-width h)
1. Prompts: connected components of the N2 cue pixels that contain a body (opening by a disk of radius h/2) and a core
   pixel. Per component: a box (padded by h/2) + 1-4 positive points spread over the body.
2. SAM 2.1 (small or large) gives 3 candidate masks per component; keep the most confident one whose area is
   <= 3x the cue component and which covers >= 50% of its body; otherwise fall back to the cue component.
   `full` = one SAM pass on the whole frame, `crop` = one pass per component on a zoomed crop,
   `video` = SAM 2 video predictor, prompted on the first frame of a 10-frame (2 s) clip and propagated.
3. Components of the union with no core pixel are dropped (no isolated background).
4. Hybrid `B_*`: add N2 appendage pixels whose component touches the SAM mask.
5. Token rule unchanged from N2 (>= 25% mask pixels OR >= 16 appendage pixels in a 16 x 16 patch).

## Gold set
20 frames per domain (the old sheet's random frames + further metric-sample frames, rng 2). Gold masks: SAM 2.1
large with prompts placed by hand (Claude), every mask viewed and re-prompted where wrong (`gold/prompts.json`,
`gold/thin_manual.json`). Thin points (410 in total: ant legs/antennae, mouse tails, frog limbs, tadpole tails) were
placed by hand from the images, independently of any mask.
**Bias**: gold pixel masks are SAM outputs, so IoU / token F1 favour the SAM arms. Thin-point recall and the
keypoint benchmarks do not depend on SAM.

## Result (per domain, gold set; token metrics use the 16-px token rule)

| domain | arm | pixel IoU | token recall | token precision | token F1 | thin recall (token) | background tok / frame | tok / frame |
|---|---|---|---|---|---|---|---|---|
| mice | N2 | 0.660 | 0.907 | 0.716 | 0.800 | 0.526 | 55.4 | 242.0 |
| mice | mask3b | 0.654 | 0.938 | 0.698 | 0.801 | 0.619 | 61.9 | 256.5 |
| mice | A_s_crop | 0.820 | 0.904 | 0.884 | 0.894 | 0.474 | 17.4 | 195.4 |
| mice | A_l_full | 0.821 | 0.905 | 0.893 | 0.899 | 0.557 | 15.2 | 193.7 |
| mice | B_l_full | 0.811 | 0.917 | 0.869 | 0.892 | 0.588 | 20.2 | 201.7 |
| ants | N2 (= mask3b) | 0.681 | 0.804 | 0.917 | 0.857 | 0.434 | 1.55 | 42.0 |
| ants | A_s_crop | 0.775 | 0.846 | 0.965 | 0.902 | 0.533 | 0.85 | 41.9 |
| ants | A_l_full | 0.746 | 0.835 | 0.941 | 0.885 | 0.467 | 1.00 | 42.4 |
| ants | B_l_full | 0.747 | 0.849 | 0.941 | 0.893 | 0.512 | 1.05 | 43.2 |
| frogs | N2 (= mask3b) | 0.656 | 0.873 | 0.779 | 0.823 | 0.696 | 4.60 | 25.6 |
| frogs | A_s_crop | 0.737 | 0.893 | 0.877 | 0.885 | 0.826 | 2.50 | 23.2 |
| frogs | A_l_full | 0.736 | 0.873 | 0.886 | 0.880 | 0.783 | 2.10 | 22.5 |
| frogs | B_l_full | 0.700 | 0.879 | 0.832 | 0.855 | 0.783 | 3.60 | 24.1 |

**Pre-registered decision: NOT adopted.** The best-F1 candidate, A_s_crop (mean token F1 0.894 vs N2 0.827), loses
mouse-tail recall (0.474 vs 0.526).

Post hoc (not the pre-registered decision): 4 candidates beat N2 on both thin recall and background in every domain:
A_l_full (F1 0.888), B_s_crop (0.887), B_l_full (0.880), A_s_video (0.875). Only **B_l_full** has frame-bootstrap
95% intervals that exclude zero on all three criteria in all three domains (thin recall diff: mice [+0.009, +0.12],
ants [+0.023, +0.127], frogs [+0.026, +0.171]; background tokens/frame diff: mice [-48.2, -24.2], ants [-0.9, -0.2],
frogs [-1.65, -0.4]).

## Chosen variant (coordinator's deadline criterion, 2026-10-29 ~21:40): B_s_crop
Criterion stated before choosing: within the hybrid family B, best mean token F1 subject to thin recall (token) >= N2
in every domain, throughput as tie-break. B_s_crop (F1 0.887; thin recall mice 0.536 / ants 0.574 / frogs 0.841 vs
N2 0.526 / 0.434 / 0.696) beats B_l_full (0.880), B_s_video (0.870); B_l_crop and B_s_full fail (mice thin recall
0.515). B_s_crop bootstrap 95% intervals vs N2: token F1 mice [+0.073, +0.113], ants [+0.046, +0.068], frogs
[+0.008, +0.056]; thin recall mice [-0.041, +0.067] (a tie), ants [+0.081, +0.194], frogs [+0.054, +0.255];
background tokens/frame mice [-46.7, -22.4], ants [-1.05, -0.15], frogs [-1.2, +0.45] (a tie).

Keypoint benchmarks (validate.py frames; N2 reproduces validate/*_final.json: MABe22 and CalMS21 exactly, CRIM13 up to
2 tokens over 384 frames on the RTX 2080 Ti):

| set | metric | N2 | B_s_crop | A_l_full | B_l_full |
|---|---|---|---|---|---|
| MABe22 | tail segment tok | 0.311 | 0.306 | 0.356 | 0.364 |
| MABe22 | tail tip tok | 0.095 | 0.105 | 0.132 | 0.135 |
| MABe22 | tail base tok | 0.589 | 0.541 | 0.598 | 0.607 |
| MABe22 | body tok | 0.591 | 0.580 | 0.607 | 0.611 |
| MABe22 | leak / tok per frame | 0.028 / 41.3 | 0.030 / 38.1 | 0.024 / 41.8 | 0.025 / 42.5 |
| CalMS21 | white intruder missed | 0.300 | not run | 0.570 | 0.522 |
| CalMS21 | pale tok / leak / tok per frame | 0.273 / 0.221 / 229.5 | not run | 0.290 / 0.028 / 111.6 | 0.313 / 0.034 / 127.6 |
| CRIM13 | albino tok / tol1 | 0.089 / 0.365 | 0.229 / 0.312 | 0.208 / 0.286 | 0.208 / 0.318 |
| CRIM13 | resident tok / leak / tok per frame | 0.969 / 0.178 / 141.8 | 0.958 / 0.103 / 99.5 | 0.958 / 0.102 / 86.6 | 0.958 / 0.109 / 99.3 |

Temporal union over t-2..t+2 at 5 fps (B_s_crop, 10 videos x 3 windows per domain, `tempunion_<dom>.json`):
|union| / |frame t| tokens (ratio of sums) mice 1.53, ants 1.56, frogs 1.52 (N2: 1.43 / 1.53 / 1.48).

Cost of B_s_crop, end to end per frame (N2 cue + prompts + SAM 2.1 small crops + assembly + token rule; decode and
background-pack loading excluded), RTX 2080 Ti fp16, one frame per call (`tempunion_<dom>.json` breakdown):
mice 0.501 s (2.0 frames/s; SAM 0.354, prompts 0.096, N2 0.029), ants 0.255 s (3.9 frames/s), frogs 0.140 s
(7.1 frames/s). GPU-h at 5 fps: mice 541 (3.89 M frames), ants 129 (1.82 M), frogs 107 (2.76 M) = 777 GPU-h on
2080 Ti. No L40S / A100 / A40 timing (that queue never started the job); no batching across frames was tried.

## Sheets
`mice.png`, `ants.png`, `frogs.png`: frame | N2 (blue) | B_s_crop (green) | thin-part zoom centred on the densest
appendage region (green = kept only by B_s_crop, red = kept only by N2, yellow = B_s_crop pixel outline).
Rows = the old approval sheet's random frames + hard cases (huddles, pale tail, odour box, largest ghost, dish rim).
