# v2.1 mask diagnosis: pre-registration

Written 2026-11-02 ~12:20, BEFORE any of the arms or readouts below was trained, coded or scored.
Code: `scripts/eci/v2/v21_maskdiag.py`. Results: `results/vision/eci_v2_1/maskdiag/`.

## Question

On mice, static x on mask v5 (arm S, mean AUROC P(on) 0.737) aligns worse with the behaviour labels than static x on the
old N2 mask restricted to U_t (arm SN2, 0.768): S - SN2 = -0.034 [-0.057, -0.011] (commit 22ea512). On ants S and SN2
are close. Why?

## Token categories (inside the stored U_t of a frame)

a = in M_t and N2; b = M_t only (v5 only); c = N2 only; d = neither (U_t only: v5 of the frames t-2..t+2).
S = a+b. SN2 = a+c.

## Measurements

1. Token statistics (no model), per eval frame, mice and ants: counts of a / b / c / d; the chessboard distance of the c
   and d tokens to the nearest M_t token; "bridge" c tokens (within 4 patches of two distinct M_t components of >= 4
   tokens); the probe probability of c and b tokens; all of these on positive vs negative frames of every label.
2. Readout vs training (H3, mice, no retraining): SAEs S s1001-1003 read out on the N2 tokens (key SxN2); SAEs SN2
   s1001-1003 read out on the M_t tokens (key SN2xM). Identity check: the same code recomputes the stored S and SN2 frame
   codes (fail loud if max abs diff > 1e-2).
   Argmax-token category (a / b / c / d) of each SAE's own best neurons (ladder picks) on every eval frame, reported on
   positive frames and on that neuron's top-1% frames.
3. Token-set arms (H1 / H2), static x, same SAE recipe, seeds 1001-1003, readout over the same token set:
   SA = a (mode A); SMN = a+b+c (mode MN); SU = a+b+c+d = all of U_t (mode U).
   Training sample: the exact S window plan (v21_train cmd_prep, rng 0), 16M / 1.5M / 1.5M tokens drawn with rng 1.
   Mice and ants.
4. Scoring: v21_train score (P(on) HMM readout, frame max), v21_ladder cols (identity check vs the score json), and the
   AMENDMENT 4 B paired bootstrap (pools / videos + seeds resampled, 2000 replicates) in a separate ladder dir. The S, SN2,
   PB and V rows are the existing ladder entries.

## Primary metric and decision rules

The metric is mean AUROC P(on) over the eligible gate cells. "Effect" means the 95% CI (pools + seeds) excludes 0.

- **H3 readout vs training (mice).** "Training" effect: SN2xM - S and SN2 - SxN2. "Readout" effect: SxN2 - S and
  SN2 - SN2xM.
  - Attributed to TRAINING if both training contrasts are > 0 with an effect, and neither readout contrast is.
  - Attributed to READOUT if the reverse holds.
  - BOTH if all four are > 0 with an effect.
  - Otherwise MIXED / UNRESOLVED.
- **H2 c tokens (N2-only context) carry the gain.** SUPPORTED if SMN - S > 0 (effect) and SMN - SN2 is not < 0 (effect).
  REFUTED if SMN - S has no effect.
- **H1 temporal dilation (d tokens).** SUPPORTED if SU - SMN > 0 (effect), or if SU - S > 0 (effect) and
  SU - SMN >= 0. REFUTED if SU - SMN < 0 (effect).
- **b tokens hurt (v5-only thin parts and edges).** SUPPORTED if SA - S > 0 (effect).
- **H4 thin parts.** N2 has LOWER thin-part recall than v5 on mice (0.489 vs 0.574), so missing thin parts in v5 cannot
  by themselves explain SN2 > S. It is tested only through the SA - S contrast and the argmax categories.
- **Recommendation for the frozen v2 token set.** Among S, SMN and SU, pick the arm with the highest mice mean AUROC
  P(on) whose mice contrast vs S is > 0 (effect), provided ants mean AUROC vs S is non-inferior (lower CI bound >= -0.01).
  If none qualifies, keep M_t.
  - The motion arm V is NOT retrained on the new token sets in this diagnosis. Whether the gain carries over to V is
    stated as untested.

## Stated limits

- The SN2 SAEs were trained on their own window plan (N2-based tokens per frame), not the S plan. The new arms use the S
  plan.
- Picked neurons are held fixed in the bootstrap, so the CI omits selection variance (as in AMENDMENT 4).
- Every number is P(on) with the frame-max readout.
