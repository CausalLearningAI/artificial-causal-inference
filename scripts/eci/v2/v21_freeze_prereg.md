# v2.1 freeze: one clean mask + one frame readout for arm V (pre-registration)

Written 2026-11-02 ~19:10, AFTER the label-free b-token characterisation below and BEFORE any SAE of this file was
trained and before any score. Code: `scripts/eci/v2/v21_freeze.py`. Results: `results/vision/eci_v2_1/freeze/`.

## Why (measured, label-free: `freeze/btok_<dom>.json`, every 5th eval frame)

b tokens (mask v5 but not N2) sit on the edge of the mask v5 blobs; a tokens (v5 and N2) mostly do not.

| | mice a | mice b | ants a | ants b |
|---|---|---|---|---|
| tokens per frame | 188.4 | 15.7 | 42.8 | 13.1 |
| share with < 4 of 4 neighbours in mask v5 (boundary) | 0.47 | 0.83 | 0.59 | 0.98 |
| share removed by a one-patch opening (thin parts) | 0.13 | 0.55 | 0.13 | 0.52 |
| median probe (uint8) | 255 | 249 | 255 | 243 |

N2 keeps a patch only if >= 25% of its pixels are animal (background difference), so b = "edge and thin patches the
probe calls animal but that are mostly background": the mixed-token hypothesis.

Share of a / b kept by single rules (inside mask v5):

| rule | mice a | mice b | ants a | ants b | tokens / frame mice, ants |
|---|---|---|---|---|---|
| E: one-patch erosion (all 4 neighbours in mask v5) | 0.53 | 0.17 | 0.41 | 0.02 | 102, 18 |
| P253: probe >= 253/255 | 0.84 | 0.31 | 0.87 | 0.24 | 163, 40 |
| P254 | 0.79 | 0.23 | 0.83 | 0.18 | 153, 38 |
| O: one-patch opening | 0.87 | 0.45 | 0.87 | 0.48 | 170, 43 |

Probe threshold rule fixed here: the highest uint8 threshold that keeps >= 80% of the a tokens in BOTH domains -> P253.

## Arms and seeds

- Masks: **M** = mask v5 (M_t, current), **E** = M_t eroded by one patch (4-neighbour), **P253** = M_t tokens with
  probe >= 253/255. Same token store, the same position rule for the motion term (V = x + (x_{t+1} - x_{t-1}) / 2 at
  the same position, v21_train edge rule).
- Arms S (static x) and V on each mask. Training sample: the v21_train window plan rule (rng 0) with the token rate of
  that mask, 16M / 1.5M / 1.5M tokens (rng 1); same SAE recipe (Matryoshka BatchTopK 2048, k16). Training and readout
  use the same token set.
- Stage 1 (selection): seeds **1004-1006**, all 3 masks x {S, V} (M: the existing SAEs, codes recomputed with an
  identity check vs the stored max / mean / top10 codes, fail loud).
- Stage 2 (confirmation): fresh seeds **1007-1009**, arm V on all 3 masks trained up front; ONLY the frozen recipe and
  V@M + max are scored, and only if the frozen recipe differs from V@M + max.
- The existing 1001-1003 M models are not rescored here (their max / top10 numbers are in the ladder and maskdiag).

## Readouts (frame pooling of the token codes, applied to EVERY arm including v1)

max; mean; top10 = mean of the 10 largest token codes (fewer tokens -> their mean); maxmean = 0.5 (max + mean).
v2: then the per-neuron 3-state HMM P(on) (fitted label-free on the scoring spans, as v21_train). v1: raw pooled code
(its deployed readout), recomputed for s0-10 on v1's own tokens of the same frames; identity check of the max readout vs
the stored v1 codes (fail loud). Every v2 - v1 comparison uses the SAME readout on both sides.

## Metrics, cells, CI

Primary: mean AUROC P(on) over the gate cells (mice nn, nt, np, nose_nose; ants Y2F, B2F, groom_any per series).
Also: top-1% precision, nAP, gate passes (AMENDMENT 5 PASS_final, with the stored v1 frame-max bar as in AMENDMENT 5,
and also with v1 under the same readout), ants focal cells single / double / or. 95% CI = the AMENDMENT 4 B paired
bootstrap (pools / videos + seeds resampled, 2000 replicates; picked neurons held fixed). "Effect" = CI excludes 0;
"non-inferior" = lower CI bound >= -0.01.

## Decision rules (stage 1, seeds 1004-1006)

1. **Mask** (arm V, readout max). Mask X in {E, P253} replaces M iff mice mean AUROC V@X - V@M has an effect (> 0)
   AND ants V@X - V@M is non-inferior. If both qualify: the higher mean over the two domains of mean AUROC (median of
   seeds). Otherwise keep M. (S@X - S@M reported, not used.)
2. **Readout** (arm V on the chosen mask). Readout r in {mean, top10, maxmean} replaces max iff r - max on mean AUROC
   has an effect (> 0) in at least one domain AND is non-inferior in the other. If several qualify: the highest mean
   over the two domains of mean AUROC (median of seeds). Otherwise keep max.
3. Frozen recipe = V + chosen mask + chosen readout.

## Confirmation (stage 2, seeds 1007-1009; only if the recipe is not V@M + max)

CONFIRMED iff, recipe - V@M+max on mean AUROC: every domain that had an effect in stage 1 (for the mask or readout
change) again has an effect > 0, and the other domain is non-inferior. If not confirmed, freeze V@M + max.

## Against v1 (always reported, same readout on both sides)

The frozen recipe (on its stage-2 seeds if run, and on 1004-1006) - v1: mice mean AUROC > 0 with an effect (C3) and ants
non-inferior (C4). Reported with top-1% precision and nAP. If nothing beats M_t + max, this is said plainly.

## Stated limits

- Picked neurons held fixed in the bootstrap (CI omits selection variance).
- dyn share @256 is not computed (no probe-frame codes in this run).
- E keeps only 18 tokens per frame on ants (41% of a): an aggressive rule there.
