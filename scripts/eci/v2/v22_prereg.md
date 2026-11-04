# ECI v2.2 pre-registration: distinct, specific nose-nose and nose-tail neurons

Written 2026-11-04 (overnight lead), BEFORE any held-out score of any variant below existed. What had been seen before
writing: the v2.1 extended report (V picks one neuron, e.g. #222 in V_s1001, for nose-nose and for nose-tail), and the
Step 0 IN-SAMPLE per-neuron diagnosis of the stored V (seeds 1001-1006) and v1 (s0-s10) SAEs on all 24 pools
(`results/vision/eci_v2_2/diag/`, summarised in `results/vision/eci_v2_2/step0.md`). No variant SAE had been trained to
completion, no Step 1 readout code existed, and nothing had been scored with SG selection on held-out pools.

## Goal

A v2 whose selected nose-nose (nose_nose = nose-nose-mutual nn OR nose-nose-passive np) neuron and nose-tail (nt) neuron
are different, each specific to its behaviour, with top frames that visibly show it. Ants grooming-to-focal (any) is a
non-regression check.

## Labels and annotators (counted before scoring)

- Annotator per video: `data/mice/v1/experiment.csv` (`observation_id` = eval frame-table `obs`; one annotator per
  pool x odour).
- **rd41_3 (= het_kmt5b_m_3)**: the csv says SG, but the annotator is recorded as unknown upstream (blank in the lab's
  tracking sheet). It is treated as UNKNOWN: left out of the SG selection set and out of the non-SG evaluation set.
- **SG** (selection): 11 pools, 66 videos, 39,600 frames.
  - nose_nose: 1,030 frames, 314 bouts, 56 videos.
  - nt: 345 frames, 82 bouts, 32 videos, 10 pools.
  - Exclusive frames (for the specificity measure): nt only 337, nose_nose only 1,022, both 8.
- **non-SG** (evaluation (a)): 12 pools (annotators SD, MB, MF, CP, MR), 72 videos, 43,200 frames.
  - nose_nose: 718 frames, 188 bouts. nt: 370 frames but only **34 bouts in 20 videos, 10 pools** (nt evaluation is
    noisy; stated up front).
- Sufficiency rule for SG selection (fixed here): a behaviour is selected on SG only if SG has >= 50 bouts in >= 8
  pools; otherwise fall back to all annotators for that behaviour, flagged. Both behaviours pass (nose_nose 314 bouts /
  11 pools, nt 82 bouts / 10 pools), so **no fallback**.

## Measures

- **AUROC**: behaviour vs every other frame, on held-out frames, ties 0.5.
- **Discrimination AUROC** (specificity): AUROC of the neuron for nt (positive) vs nose_nose (negative), among frames
  carrying exactly one of the two labels. A nose-tail neuron should be > 0.5, a nose-nose neuron < 0.5.
- **Balanced AUROC** (selection criterion of the new rule) for behaviour b with sibling s:
  0.5 x [AUROC(b vs frames with neither label) + discrimination AUROC(b vs s)]. It weights the two kinds of negative
  frames (background, the confusable sibling behaviour) equally instead of by frequency (the sibling is ~1-2% of frames,
  so the plain AUROC barely penalises firing on it).
- Readout for ranking/selection: v2-type models = per-neuron 3-state HMM P(on) (as V); v1 = raw frame code (as before).
  F1 and time use P(on) (v1: the same HMM on the picked neuron's code). Precision@100 ranks by the readout, ties broken
  by the raw code, then a fixed random draw.
- Time per video (held-out, 120-s windows): bias and MAE of sum P(on) vs true on-time (seconds and fraction of the
  video), raw and calibrated (one scalar = total true / total sum P(on) on the selection frames), Pearson r; plus the
  constant-rate reference.

## Selection rules

Eligibility (all rules, label-free): the readout is > 0 on >= 0.2% of all eval frames.

- `auc` (current rule): per behaviour, the neuron with the best AUROC on the selection frames.
- `bal` (new primary rule): per behaviour, the neuron with the best balanced AUROC on the selection frames. Nothing forces
  the two picks to differ (a neuron's discrimination AUROCs for the two behaviours sum to 1, so one neuron rarely tops
  both, but it can).
- `balpair` (reported): the pair (i != j) maximising balanced AUROC nose_nose(i) + nt(j).

## Evaluation designs

- **(a) primary, `sg2non`**: select on all SG frames, evaluate on the 12 non-SG pools.
- **(b) secondary, `sgcf`**: within the 11 SG pools, the 6 unit splits (atlas + rnd0-4 of eval.py) x 2 halves = 12
  directions; select on one half, evaluate on the other, mean over directions.
- Identity: V and v1 under the old rule (all 24 pools, 12 directions, `auc`) must reproduce the v21 extended picks and
  numbers (AUROC, per-video sum P(on); fail loud at 1e-4).
- **Bootstrap** (1000 replicates, rng 20261104): resample held-out pools (within the held-out pool set) and each arm's
  seeds with replacement; picks and calibration scalars held fixed at their full-data values. Difference = mean over
  seeds; 95% CI = 2.5 / 97.5 percentiles.

## Arms

Reference rows: v1 (s0-s10) under `auc` and `bal`; **V** (seeds 1001-1006, frame max) under `auc` (the current recipe)
and under `bal`.

Candidate variants (each scored under `bal`, the primary rule):

Readout-only, on the stored V SAEs (seeds 1001-1006), no retraining (Step 1):
1. **V+bal**: V, frame max, the new rule only. Tests whether the selection rule alone is the fix.
2. **V@box3**: per-patch code map smoothed by a 3x3 box mean (zeros outside the mask, / 9), then frame max. A neuron must
   fire over a neighbourhood, not one patch.
3. **V@top10**: mean of the 10 largest token codes per frame (stored codes). Same idea, without spatial structure.
4. **V@coarse**: the 2x2-average-pooled tokens (32-px effective patches) encoded with the existing V SAE, frame max.
   Out of the SAE's training distribution; tests whether coarser tokens alone carry the behaviour.
5. **V@ms**: max over native and pooled tokens of the frame (multi-scale readout of the existing SAE).

Retrained SAEs (Step 2; seeds 1004-1006; mice first, ants only for the adopted variant):
6. **VP2** (pooled-token SAE): train and read out on 2x2-pooled tokens only. v1's patches held ~4x more body per patch on
   mice; this gives v2 the same token scale while keeping DINOv3 and the native-resolution encoding.
7. **VMS** (scale augmentation): train on native + 2x2-pooled tokens together (all of both); read out with `ms` (max
   over native and pooled tokens). Token-space analogue of multi-scale crop augmentation (grid flips / rotations are not
   valid: DINO features are not equivariant).
8. **VW8** (anti-absorption by width): 8192 latents, prefixes 128/256/512/2048/8192 = V's four levels plus a deeper
   level of 6,144 latents. If a broad "heads in contact" parent absorbs the nose-tail feature, a deeper Matryoshka level
   gives the specific children room to split off (feature splitting with width); everything else as V.

Not run (reasons): AuxK is already in the trainer (all arms); a firing-frequency penalty and decoder orthogonality add
new hyper-parameters with no prior setting on this data, so one night does not allow tuning them honestly.

## Adoption criteria (per variant; mice with SG-based selection, design (a), rule `bal`)

- **C1** distinct: in >= 2 of 3 seeds (stored-V readout variants with 6 seeds: >= 4 of 6) the selected nose-nose and
  nose-tail neurons differ.
- **C2** specific: held-out discrimination AUROC (nt vs nose_nose), median over seeds: nose-tail neuron > 0.6 AND
  nose-nose neuron < 0.4.
- **C3** non-inferior detection: AUROC of nose-nose and of nose-tail, variant (`bal`) - V (`auc`, the current recipe,
  same SG selection and held-out pools): 95% CI lower bound >= -0.02, both behaviours.
- **C4** ants non-regression: grooming-to-focal (any) AUROC and per-video time Pearson r, variant - V (both under the
  v21 extended pipeline and rule, pooled ants series): lower CI bound >= -0.02. Run only for variants passing C1-C3
  (readout-only variants: on the stored ants V SAEs; retrained: train on ants first).
- **ADOPT** iff C1-C4. Tie-break between adopted variants: mean of precision@100 for nose-nose and nose-tail (design
  (a)).
- C1-C3 are also reported for design (b) (secondary; not part of the decision).

## Qualitative check

Top-12 frame sheets (re-using `v21_extended_sheet.py`) for v1, V and every adopted variant (if none is adopted: the
best-scoring variant under the tie-break, labelled "not adopted"), for nose-nose, nose-tail and ants grooming-to-focal
(any). I look at them and describe what the top frames show.

## AMENDMENT 1 (2026-11-04 ~05:10, after all pre-registered arms were scored; before any number of the seeds below existed)

The pre-registered variants all failed C3. Two POST HOC options were explored on the same seeds and look best: `hyb`
(nose-nose by the AUROC rule, nose-tail by the balanced rule) and the mean-of-top-10-tokens readout. Because they were
chosen after seeing the scores, they are confirmed here on **fresh V seeds 1007-1009** (same recipe as V, trained during
v2.1, never scored in v2.2). Same design (a), metrics, bootstrap and criteria C1-C3. Arms:
`Vf|auc` (frame max, current rule = reference), `Vf|hyb`, `Vf@top10|hyb`, `Vf@top10|bal`. The top10 codes are
recomputed on CPU here (`v22_encode.py --readouts max,top10`, the v21_train top10 definition). CONFIRMED iff an arm
passes C1-C3 against `Vf|auc`; reported either way, and nothing here changes the pre-registered verdict.
