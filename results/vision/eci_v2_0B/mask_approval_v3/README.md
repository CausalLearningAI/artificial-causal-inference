# Mask v5: cheap per-domain patch mask, no SAM at inference

Pre-registration: `scripts/eci/v2/mask5_prereg.md` (commit d5eeea2, before any test number) plus POST-HOC AMENDMENT 1
(extended dev grid, written after the pre-registered ants test failed). Code: `scripts/eci/v2/mask5_val.py`.
Nothing was encoded into a store.

## Route (b), the only route tried (route (a), tuned classical with the MAD fix, was not attempted: no time)
Same algorithm for every domain; parameters per domain.
1. DINOv3-L/16 block-20 tokens of the frame (the planned store's own encode).
2. Linear logistic probe per domain (standardised tokens, L2 1e-3, LBFGS). Training labels: the SAM 2.1 B_l_full
   mask (offline only), N2 token rule, on 120 frames per domain from the mask3 metric sample, from videos with no
   test frame and excluding all gold frames.
3. keep = probe probability >= t (optional 3x3 mean smoothing, optional OR with attached N2 thin tokens); then
   8-connected components with no token of probability >= t_core are dropped.
4. Chosen on the 10 dev gold frames per domain; scored once on the 10 test gold frames.

| domain | pre-registered grid: chosen setting | extended grid (post hoc): chosen setting |
|---|---|---|
| mice | t 0.6, t_core 0.5, smoothing on, N2-thin OR on | t 0.9, t_core 0.8, smoothing off, OR off |
| ants | t 0.5, t_core 0.5, smoothing on, OR on | t 0.7, t_core 0.8, smoothing off, OR off |
| frogs | t 0.6, t_core 0.5, smoothing on, OR on | t 0.9, t_core 0.8, smoothing off, OR off |

The extended choice for mice and frogs sits on the grid edge (t = 0.9), so a higher threshold might do better still (not tried).

## Test split (10 frames per domain; token = 16 px patch; background = kept token with < 5% gold pixels)

| domain | arm | token F1 | token recall | token precision | thin-part recall | background tok / frame | isolated bg components / frame | tok / frame |
|---|---|---|---|---|---|---|---|---|
| mice | N2 | 0.778 | 0.904 | 0.683 | 0.489 | 66.9 | 0.9 | 256.6 |
| mice | new, pre-registered | 0.849 | 0.923 | 0.786 | 0.574 | 37.3 | 0.0 | 227.6 |
| mice | new, extended (post hoc) | 0.921 | 0.924 | 0.918 | 0.574 | 10.1 | 0.6 | 195.2 |
| mice | B_s_crop (SAM ref.) | 0.887 | 0.916 | 0.860 | 0.511 | 23.1 | 0.1 | 206.5 |
| ants | N2 | 0.867 | 0.818 | 0.921 | 0.479 | 0.8 | 0.0 | 41.6 |
| ants | new, pre-registered | 0.834 | 0.868 | 0.802 | 0.547 | 6.1 | 0.0 | 50.6 |
| ants | new, extended (post hoc) | 0.911 | 0.949 | 0.876 | 0.761 | 2.6 | 0.0 | 50.7 |
| ants | B_s_crop (SAM ref.) | 0.924 | 0.874 | 0.981 | 0.581 | 0.2 | 0.0 | 41.7 |
| frogs | N2 | 0.773 | 0.910 | 0.672 | 0.621 | 7.8 | 0.6 | 27.1 |
| frogs | new, pre-registered | 0.859 | 0.870 | 0.849 | 0.621 | 2.4 | 0.0 | 20.5 |
| frogs | new, extended (post hoc) | 0.884 | 0.915 | 0.855 | 0.793 | 2.1 | 0.0 | 21.4 |
| frogs | B_s_crop (SAM ref.) | 0.827 | 0.955 | 0.729 | 0.862 | 6.6 | 0.1 | 26.2 |

Thin points on test: mice 47, ants 117, frogs 29.

**Pre-registered rule** (token F1 > N2 AND thin recall > N2 AND background tok/frame <= N2):
- pre-registered grid: mice PASS, ants FAIL (F1 and background), frogs FAIL (thin recall tie 0.621 = 0.621)
- extended grid (post hoc): mice PASS, frogs PASS, ants FAIL (background 2.6 vs N2 0.8 tok/frame)

## Store numbers (5-frame windows t-2..t+2 around all gold + sheet frames; extended setting)

| domain | 5-frame union / mask at t (tokens) | mask at t inside S (union of N2 dilated by 1 token) | union tok / frame | store at 5 fps (fp16, 1024-d) |
|---|---|---|---|---|
| mice | 1.28 | 97.9% | 247.7 | 1.97 TB (3.89 M frames) |
| ants | 1.33 | 99.9% | 68.2 | 0.25 TB (1.82 M) |
| frogs | 1.52 | 99.8% | 36.0 | 0.20 TB (2.76 M) |
| total | | | | 2.43 TB |

Extra per-frame cost: the probe is one 1024-d dot product per token on tokens the encode computes anyway;
measured 0.07-0.13 s per frame in numpy on the CPU (about 4096 x 1024 multiply-adds for mice). On the GPU it is
negligible next to the DINOv3 encode (0.36-0.83 s/frame here on an RTX 2080 Ti). Encoding ran in fp32 or bf16 on the
RTX 2080 Ti (the L40S queue was 5 weeks out); the planned store encodes bf16 on L40S, so the probe should be checked
on a few store frames before use.

## Caveats
- Training labels and the gold pixel masks both come from SAM 2.1 (B_l_full offline / large with hand prompts):
  token F1 is biased towards SAM-like masks. Thin-point recall uses hand points and is independent.
- 10 test frames per domain; no bootstrap intervals computed (no time).
- The extended grid was chosen after seeing the ants pre-registered test result (post hoc).
- Ants: the probe keeps a few more background tokens next to the ants than N2 (2.6 vs 0.8 per frame) and fails.

## Sheets
`mice.png`, `ants.png`, `frogs.png`: frame | N2 (blue) | new (green, extended setting) with the 16 px patch grid |
thin-part zoom (green = new only, red = N2 only, white = both). Rows: the old approval sheets' random frames + hard
cases (huddles, pale tail, odour box, largest ghost, dish rim). Some rows are dev or training-video frames.

# Mask v5 on the mouse benchmarks (MABe22, CalMS21, CRIM13)

Pre-registration: `scripts/eci/v2/mask5_bench_prereg.md` (commit f936a05, before any benchmark probe or score).
Code: `scripts/eci/v2/mask5_bench.py` (+ `.sh`). Per-benchmark tables: `bench_{mab,cal,crim}_v5.md` / `.json`.
Arms: T = approved mice probe as is (t 0.9, t_core 0.8, no smoothing, no N2 OR); B = same recipe trained on 120
SAM-labelled frames (B_l_full, offline) of the benchmark's TRAINING split, same thresholds; N2 with the pooled
per-setup background (reference) and with the per-clip background. Held-out = validate.py test frames; scoring with
keypoints only. Ran on debug_gpu (RTX 2080 Ti, fp32 encode; the -p gpu queue start estimate was the next morning).
N2 per-clip reproduces validate.py exactly (max |diff| CalMS21 5.6e-17, CRIM13 0.0).

| bench (frames) | arm | all-keypoint recall | animal miss rate | white / BC intruder miss | background tok / frame | isolated bg comp / frame | tok / frame |
|---|---|---|---|---|---|---|---|
| MABe22 (960) | N2 pooled | 0.888 | 0.000 | - | 36.5 | 1.94 | 66.9 |
| | N2 per-clip | 0.534 | 0.245 | - | 0.3 | 0.03 | 15.5 |
| | T | 0.326 | 0.263 | - | 0.2 | 0.13 | 8.4 |
| | B | 0.888 | 0.000 | - | 12.7 | 4.64 | 41.2 |
| CalMS21 (456) | N2 pooled | 0.641 | 0.117 | 0.235 | 75.8 | 16.17 | 269.3 |
| | N2 per-clip | 0.619 | 0.150 | 0.300 | 50.5 | 14.19 | 229.5 |
| | T | 0.693 | 0.011 | 0.022 | 3.0 | 0.67 | 109.7 |
| | B | 0.825 | 0.010 | 0.020 | 6.5 | 2.00 | 156.4 |
| CRIM13 (384) | N2 pooled | 0.758 | 0.161 | 0.615 | 59.4 | 5.57 | 203.1 |
| | N2 per-clip | 0.755 | 0.167 | 0.635 | 25.2 | 3.35 | 141.8 |
| | T | 0.785 | 0.060 | 0.188 | 2.7 | 0.83 | 58.9 |
| | B | 0.831 | 0.072 | 0.266 | 7.2 | 1.25 | 89.3 |

Background = kept token with no pixel of (keypoint convex hull dilated by 1.5 h) (CalMS21 + tail-base disk; CRIM13 a
6 h disk per track point). CRIM13 miss = no kept token within 1 token of the track point.

Pre-registered verdict: chosen arm T on all three (B always fails the background guard B <= T + max(1, 0.1 T)).
PASS vs pooled N2: CalMS21 PASS, CRIM13 PASS, MABe22 FAIL (T misses 26% of mice). Post hoc (not the pre-registered
rule): on MABe22 B equals pooled N2 on recall (0.8884 vs 0.8885) and miss (0.000) with a third of its background.

Store projection (union t-2..t+2, 5 fps, fp16 1024-d): T MABe22 0.025 TB, CalMS21 0.059 TB, CRIM13 0.170 TB;
B 0.095 / 0.076 / 0.252 TB (dilated by 1 token: T 0.092 / 0.111 / 0.388, B 0.283 / 0.137 / 0.511 TB).
Sheets: `bench_mabe22.png`, `bench_calms21.png`, `bench_crim13.png` (frame with keypoints | N2 pooled | T | B |
zoom of the chosen arm vs N2 on MABe22 tail tip / CalMS21 white intruder nose / CRIM13 intruder point; 6 random rows
+ the 2 worst frames of the chosen arm).
