# Pre-registration: does v1's smaller SAE close the ants top-1% precision gap? (2026-11-03)

Written and committed BEFORE any small-SAE model was trained or scored. Code: `scripts/eci/v2/v21_small.py`.
Requested by the user (approved run). The lever was found POST HOC in the ants v1-gap chain
(`results/vision/eci_v2_1/v1gap/report.md`): SAE recipe step under DINOv3, D3r2 - D3r1, top-1% precision -0.060
[-0.113, +0.009]; D3r1 (v1's recipe on DINOv3) - v1 top-1% precision -0.032 [-0.089, +0.030]. This test asks whether the
same recipe change helps the deployed v2.1 pipeline (native resolution, mask v5, HMM readout).

## Arms (only the SAE changes)
- `S_small` (key `Ssm`) = arm S of `v21_train.py` (static x on the mask-v5 tokens M_t), `V_small` (key `Vsm`) = arm V
  (x + (x_{t+1} - x_{t-1}) / 2 on M_t).
- SAE = v1's recipe, identical to rung D3r1 of the v1-gap chain except the input normalisation:
  Matryoshka BatchTopK, **1024 latents, prefixes 128 / 256 / 512 / 1024, k 16, 8,380 steps x 4096 tokens**, lr 5e-4,
  warmup 500. Choice of prefixes: v1's own (not 64/128/256/1024), so the arm is exactly the recipe the v1-gap lever
  measured; the smallest Matryoshka level stays at 128 like v1 and v2. k 16 = v1's k (and v2's).
- Unchanged from S / V: tokens (`eci_tokens/v2_full/<dom>/v2.1_m5`, flag bit0), training sample (16M train, 1.5M
  calibration, 1.5M evaluation tokens, the stored `<role>_{B,C}.npy`), per-1024-block TokenNorm, theta* calibration,
  threshold-mode encoding.
- 8,380 x 4096 = 34.3M token draws = 2.1 passes over the 16M-token sample (v1: 5 epochs of 6.9M tokens). Stated, not
  tuned.
- Seeds 1004, 1005, 1006, both domains (12 SAEs).

## Readouts and scoring
- Primary: HMM P(on) of the per-frame max over M_t token codes (exactly the ladder, `v21_ladder.py`).
- Secondary: HMM P(on) of the top10 frame aggregation (mean of the 10 largest token codes). v1 stays at its raw frame
  max in both (stored codes only), as in the ladder.
- Same eval frames, cells, splits, scorer (`v21_ladder.score2`) and bootstrap (AMENDMENT 4 B: units = ants videos within
  series / mice pools, plus seeds; 2000 replicates; the SAME unit replicates as the stored ladder `boot_<dom>.npz`;
  neuron pick fixed at its full-data pick).
- Comparators: S and V on seeds 1004-1006 (current recipe, stored ladder rows), v1 (11 seeds, stored ladder rows).

## Identity checks (fail loud)
1. Scorer: V_s1004 re-scored through this file's cols step must equal the stored score JSON
   (`results/vision/eci_v2_1/<dom>/V_s1004.json`) and the ladder's cols JSON to 1e-6 (every gate / focal cell).
2. Bootstrap: V_s1004 replicates must equal the stored ladder bootstrap rows (same unit weights).
3. Code path (recorded, not fatal): V_s1004 frame codes recomputed by this file's codes step vs the stored codes.

## Metrics (domain mean over the eligible gate cells)
Mean AUROC P(on), top-1% precision P(on), nAP P(on). Ants gate cells: Y2F / B2F / groom_any of v2, v3, v4,
v4_infested. Mice gate cells: nn, np, nose_nose, nt. Ants focal grooming cells reported (single = Y2F XOR B2F, double =
Y2F AND B2F, or = Y2F OR B2F), plus every per-cell value.

## Decision rule (primary readout, CI = 95% bootstrap over videos/pools + seeds)
ADOPT the small recipe for v2.1 iff ALL of
- C1: ants mean top-1% precision, V_small - V: CI excludes 0 on the positive side (lower bound > 0);
- C2: ants mean AUROC, V_small - V: lower bound >= -0.01 (non-inferior);
- C3: mice mean AUROC, V_small - V: lower bound >= -0.01 (non-inferior).
Otherwise the current recipe stays. Reported but not part of the rule: S_small - S (the same three tests on the static
arm), V_small - v1 and S_small - v1 on every metric, the focal cells, the top10 readout, SAE eval FVE / L0 / dead latents.
If C1 holds but C2 or C3 fails, the result is reported as a trade-off, not an adoption.
