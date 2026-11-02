# Ants v1-gap: verdict against the pre-registered rules (scripts/eci/v2/v21_v1gap_prereg.md)

All rungs are scored on the same 127,200 ants eval frames, with the same cells, splits and scorer, using the raw frame
max (except SN2, which uses HMM P(on)). Metrics are averaged over the 12 gate cells. Each delta is a difference of seed
means, with a 95% CI that resamples videos within series AND seeds (2000 replicates, the same replicates as the ladder).
Tokens: 6.22M v1-mask tokens (48.9 per frame). v1 identity check: fire agreement 0.9997 with the stored v1_s0 codes.
SAE training ran on gpu238 (A100).

Correction: in report.md (first version) the contrast-table header named the columns in the wrong order. The numbers
were right; the header was fixed in the same commit. Columns are AUROC | nAP | top-1%.

## Chain (top-1% precision is where v2 loses)
| step | what changes | d mean AUROC [CI] | d top-1% prec [CI] |
|---|---|---|---|
| v1 -> D2r1 | same v1 tokens; v1-sized SAE (1024, 8,380 steps) re-trained by us on the eval frames | -0.014 [-0.054, +0.031] | -0.060 [-0.165, +0.028] |
| D2r1 -> D2r2 | SAE recipe r1 -> r2 (2048 latents, 30k steps), DINOv2 tokens | -0.004 [-0.046, +0.032] | -0.019 [-0.104, +0.084] |
| D2r1 -> D3r1 | encoder DINOv2-B -> DINOv3-L blk 20 (same 32 grid, same mask), recipe r1 | +0.000 [-0.045, +0.032] | +0.028 [-0.068, +0.140] |
| D2r2 -> D3r2 | encoder, recipe r2 | +0.003 [-0.030, +0.040] | -0.013 [-0.060, +0.042] |
| D3r1 -> D3r2 | SAE recipe r1 -> r2, DINOv3 tokens | -0.001 [-0.036, +0.033] | -0.060 [-0.113, +0.009] |
| D3r2 -> SN2raw | native grid (48-52) + N2 mask + v2.1 5 fps sample | -0.003 [-0.039, +0.034] | -0.008 [-0.069, +0.057] |
| SN2raw -> SN2 | readout raw max -> HMM P(on) | +0.009 [-0.010, +0.029] | +0.013 [-0.042, +0.079] |
| total SN2raw - v1 | | -0.018 [-0.043, +0.007] | **-0.101 [-0.155, -0.030]** |
| total SN2 - v1 | | -0.009 [-0.034, +0.018] | **-0.087 [-0.144, -0.009]** |

Levels, top-1% precision, seed mean (seeds):
- v1: 0.840 (0.746-0.914, 11 seeds)
- D2r1: 0.780 (0.853 / 0.659 / 0.828)
- D2r2: 0.761
- D3r1: 0.808 (0.826 / 0.838 / 0.761)
- D3r2: 0.748
- SN2raw: 0.740
- SN2: 0.753

Against v1 on top-1% precision, with the CI resampling videos + seeds:
- Significantly below v1: D2r2 -0.079 [-0.129, -0.029], D3r2 -0.092 [-0.144, -0.026], SN2raw -0.101.
- Not significantly below v1: D3r1 -0.032 [-0.089, +0.030] and D2r1 -0.060 [-0.165, +0.028].

## Rules
1. **Control.** D2r1 - v1 includes 0 on both metrics, so the rule is passed. Its point estimate is, however, the largest
   single step (-0.060 top-1%, about 60% of the total gap). The CI is wide because of one bad seed (0.659). v1's
   advantage is NOT reliably reproduced when v1's own tokens and v1's SAE size are re-trained on the same frames.
2. **Step that explains the loss** (CI excludes 0): NONE.
3. **Encoder:** NOT blamed (D3 - D2 is +0.028 under r1 and -0.013 under r2; both CIs include 0).
   **Native grid + N2 mask + sample:** NOT blamed (-0.008 [-0.069, +0.057]).
4. **512 px fix:** NOT supported.
   - D3r2 is not non-inferior to v1 on mean AUROC (lower bound -0.053 < -0.01).
   - Its top-1% precision is below v1 (-0.092, CI excludes 0).
   - D3r2 - SN2raw on top-1% precision is +0.008 [-0.057, +0.069], not > 0.
   - Encoding ants on the v1 grid does not recover v1.
5. **Applies:** no single pipeline factor is attributable at this sample size. On mean AUROC, v1's advantage is within
   its seed spread.

## Post hoc (not pre-registered; flagged)
- The only consistent direction is the SAE recipe.
  - r2 (2048 latents, 30k steps) costs top-1% precision under both encoders: -0.019 (DINOv2) and -0.060 [-0.113, +0.009]
    (DINOv3).
  - Its units-only CI (seeds held fixed) excludes 0 for DINOv3: [-0.078, -0.038].
  - Every r2 rung is significantly below v1 on top-1% precision. The one DINOv3 rung with r1 (D3r1) is not.
- The rest of the gap sits in "v1 vs a faithful re-train of v1" (-0.060 point estimate). That points to v1's
  particular seeds and training details (v2 + v3-only sample, training-time threshold), not to any v2 design choice.
- A smaller / shorter SAE for ants (r1-like, 1024 latents) is the one candidate fix. It needs fresh seeds and the mice
  check before adoption. The mice gain came from resolution, not the recipe, so it would need to be re-checked there.
