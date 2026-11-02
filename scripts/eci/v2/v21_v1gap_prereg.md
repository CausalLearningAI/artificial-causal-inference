# Pre-registration: why does ECI v2 not beat v1 on ants? (2026-11-02)

Written and committed BEFORE any score of the new models below exists (the SN2 raw-readout numbers quoted in H1 were
already stored in `results/vision/eci_v2_1/ants/SN2_s*.json` 'raw' and are not new). Code: `scripts/eci/v2/v21_v1gap.py`.

## Known (not re-derived)
Ants ladder (`results/vision/eci_v2_1/ladder/`): SN2 - v1 mean AUROC P(on) -0.009 [-0.036, +0.019], top-1% precision
-0.087 [-0.140, -0.016]. v1 11 seeds: mean AUROC seed mean 0.845 (sd 0.038, range 0.780-0.887), top-1% precision seed
mean 0.840 (sd 0.064). SN2 3 seeds: 0.836 (sd 0.017) / 0.753 (sd 0.050). P(random v1 seed > random SN2 seed): AUROC 0.61,
top-1% precision 0.88. v1 is scored with RAW frame max, v2 with HMM P(on).

## Difference inventory (v1 vs SN2, ants)
| factor | v1 | SN2 |
|---|---|---|
| encoder / layer | DINOv2-B/14, last layer (12/12), 768-d | DINOv3-L/16, block 20/24 + final LN, 1024-d |
| input | 512 px JPG -> 448 px, 32 x 32 grid | native video frame, side 768 / 832 / 816 (v2 / v3 / v4), 48-52 grid |
| patches per ant body length | 2.7 (v2) / 2.9 (v3) | ~4.0 / ~4.8 |
| mask | v1 FgBackgrounds rule 'ants' (DINOv2-token background distance + dark pixels), ~49 tokens / frame | N2 thin-part pixel mask, ~42-44 tokens / frame |
| SAE | Matryoshka BatchTopK 1024, prefixes 128-1024, k 16, 5 epochs = 8,380 steps x 4096, training-time threshold | 2048, prefixes 128/256/512/2048, k 16, 30k x 4096 steps, theta* calibrated on held-out videos |
| training data | 138.6k frames at 1 fps of v2 + v3 only (6.9M tokens), 10% videos held out | 16M tokens from 25-frame windows at 5 fps of v2 / v3 / v4 training-role videos (no v4_infested), video-count weighted (v3 46%, v4 35%, v2 19%) |
| readout | raw frame max | HMM P(on) of the 5 fps frame max |
| eval frames / labels / cells / splits / scorer | identical (127,200 labelled 1 fps frames, eval.py; v1 codes recomputed on the same frames, identity-checked) | identical |

## Hypotheses and tests (ranked by cost)
- **H0 noise (no compute).** Reported descriptively (above): on mean AUROC the v1 advantage is inside v1's own seed spread;
  on top-1% precision it is not (SN2 below 10 of 11 v1 seeds).
- **H1 readout asymmetry (no compute).** If v1 only wins because of the readout, SN2 under v1's raw frame max should be
  >= v1. Measured from stored JSONs, raw SN2 mean AUROC 0.827, top-1% 0.740 (seed means): the HMM HELPS v2, so the readout
  does not explain the gap. The new SN2raw rung puts this into the bootstrap.
- **H2 decomposition chain (GPU ~2 h + CPU).** All new rungs: same frames, scorer, splits, cells; raw frame max readout.
  - `D2r1` = v1 tokens (DINOv2-B 448, v1 mask) of the eval frames, SAE r1 (v1-sized: 1024, 8,380 steps), trained on these
    tokens (90% videos; 5% calib, 5% eval; all scored). Control: should reproduce v1.
  - `D2r2` = the same tokens, the v2 SAE recipe r2.
  - `D3r1`, `D3r2` = DINOv3-L block 20 on the SAME 512 px JPG = the same 32 x 32 grid, the SAME v1 mask positions.
  - `SN2raw` = the ladder's SN2 models under raw frame max.
  - Steps: recipe = D2r2 - D2r1 (and D3r2 - D3r1); **encoder at a fixed grid and mask** = D3r1 - D2r1 and D3r2 - D2r2;
    **native resolution + N2 mask + v2.1 5 fps sample** = SN2raw - D3r2; readout = SN2 - SN2raw.
  - Seeds 1-3 each. Bootstrap = the ladder's (videos within series + seeds, 2000 replicates, identical replicates to
    `boot_ants.npz`; neuron pick fixed). Groups: all 12 gate cells, each series, Y2F+B2F, focal single / double.

## Decision rules (metrics named: mean AUROC and top-1% precision over the 12 gate cells, raw frame max)
1. **Control.** If D2r1 - v1 has a CI excluding 0 on either metric, the SAE training sample / recipe details are
   themselves a factor and are reported as such before attributing the rest.
2. **A step EXPLAINS the loss** iff its delta is negative with the CI excluding 0 on top-1% precision or mean AUROC.
   Its share = step delta / (SN2raw - v1) on that metric. "Strong explanation" = one step (or encoder + resolution
   together) with share >= 0.5 and CI excluding 0.
3. **Encoder** is blamed iff D3 - D2 is significantly negative under BOTH recipes. **Resolution / native grid
   (+ N2 mask + sample)** is blamed iff SN2raw - D3r2 is significantly negative. These two are not separated from the
   N2 mask or the 5 fps sample by this design; that is stated, and the mask is not tuned here.
4. **Fix.** If D3r2 is non-inferior to v1 on mean AUROC (lower CI bound >= -0.01) AND its top-1% precision delta vs v1
   has a CI including 0 or above, AND D3r2 - SN2raw on top-1% precision is > 0 with CI excluding 0, the proposed ants fix
   is: encode ants at the 512 px side (32 x 32 grid, ~2.8 patches per ant) instead of native. Otherwise no fix is
   claimed from this experiment.
5. If no step is significant, the conclusion is that the v1 - v2 difference on ants is not attributable to any single
   pipeline factor at this sample size, and that v1's advantage is within its seed spread on AUROC.
