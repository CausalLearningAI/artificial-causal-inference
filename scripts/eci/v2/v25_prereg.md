# ECI v2.5 pre-registration: native-resolution SAEs, a proper temporal-context test, video edges excluded

Written 2026-11-05 (overnight run), BEFORE any v2.5 SAE was trained and before any v2.5 score existed. Seen before
writing: all v2.3 / v2.4 results (`results/vision/eci_v2_3/`, `results/vision/eci_v2_4/summary.md`, `LOG.md`,
`metric_top3/summary.md`), including the v2.4 video-edge bug (motion arms put many top frames on the first / last 2
frames of a video, where the old code used a doubled one-sided difference).

## Why (user requirements)

- **A.** Mice need a SEPARATE nose-tail neuron (distinct from the nose-nose neuron) whose top activations look like
  nose-to-rear contact.
- **B.** Show that temporal context helps (not a detector of static appearance). Re-test summing a motion term at
  NATIVE resolution (no pooling), after fixing the edge bug.
- **Edge rule (user)**: exclude the frames without full temporal context (the first and last L frames of every store
  video) and use ALL other frames. Applied to training AND to the scored frame set, for ALL models, including the V and
  v1 baselines, which are rescored on the same edge-excluded frame set.

## Input (code: `scripts/eci/v2/v25_train.py`)

- Token store `eci_tokens/v2_full/<dom>/v2.1_m5`: DINOv3-L/16 block 20, native grid (mice 64 x 64, ants 48 x 48), 5 fps,
  mask v5 (M_t = flag bit0); frame t stores U_t = union of M_{t-2..t+2}, so every M_t position is stored at t-2..t+2.
- **Lag L = 2 frames = +-0.4 s at 5 fps** (the largest symmetric lag the store holds; no re-encoding).
- **Edge rule**: frame t is used iff 2 <= t < T - 2 (T = frames of the store video). Every used token has x_{t-2} and
  x_{t+2} at its position (asserted, fail loud). No one-sided difference is ever formed.
- x_t = the M_t token; Delta_t = fp16(float32(x_{t+2}) - float32(x_{t-2})) at the same position.
- **alpha** (per domain, label-free, fixed before training): alpha = sqrt(E||x - mean x||^2 / E||Delta||^2) on a fixed
  sample of whole random windows (rng 6, >= 2M used tokens) -> `results/vision/eci_v2_5/<dom>_count.json`.

## Arms (each: seeds 2501, 2502, 2503; mice and ants)

| arm | SAE input | d_in |
|---|---|---|
| **N0** (static control) | x_t | 1024 |
| **W1** | x_t + alpha Delta_t | 1024 |
| **Wh** (lighter sum) | x_t + (alpha / 2) Delta_t | 1024 |
| **C1** | [x_t, alpha Delta_t] | 2048 |

Incumbent **V** (seeds 1001-1006, x + (x_{t+1} - x_{t-1}) / 2, trained on a subset in v2.1) is not retrained; its stored
codes are rescored on the edge-excluded frame set. **v1** = seed 0 (mice: ext frames; ants: seeds 0-10 as before, seed 0
headline).

SAE = V's recipe (src/eci/sae: Matryoshka BatchTopK, 2048 latents, k = 16, prefixes 128/256/512/2048, AuxK, Adam lr 5e-4,
500 warm-up steps, cosine to 10%, batch 4096, TokenNorm = ONE scalar scale over the whole input (also for C1, so alpha
keeps its meaning), geometric-median b_dec). Same lr / batch as Vall: the step counts below are >= 50k in both domains.

Training set: every used (edge-excluded) M_t token of every store video, full length, no train / test split (label-free):
mice 648 videos (v1 432 + v2 216); ants the 232 store videos (v2 all, v3 t = 2/6/8, v4 + v4_infested t = 2/6; asserted).
**Length**: passes = max(1, ceil(50,000 x 4096 / N_used)), steps = floor(passes x N_used / 4096). Stream: videos cut
into windows of 100 centre frames inside [2, T - 2); windows in a random order per pass (rng 20261105 + pass), chunks of
256 windows, each chunk permuted per seed, tail carried. ONE job per domain trains all 12 SAEs (4 arms x 3 seeds) from
one store read; within a seed all arms see the same batches (paired). theta* / FVE on two label-free samples of 1.5M
used tokens (whole random windows, rng 7 / 8; training tokens too, stated). Health: eval FVE, L0, dead latents at the
end of training, never-firing fraction on the eval sample. GPU: gpu100, v2.5 total <= 8 H100-hours (logged).

## Codes and readouts

Frame max over the M_t tokens of a frame of encode(norm(input)) (threshold mode), computed from the store on CPU.
P(on) = per-neuron 3-state HMM (`v22_full.pon_fast`), label-free, one sequence per video over the edge-excluded scored
frames (mice: all edge-excluded ext frames of the video; ants: the 5 fps span of each scored video). The V baselines'
P(on) is recomputed the same way on the edge-excluded frames (like for like).

## Frame sets

- **Mice**: ext frames (864,000; v2.3) minus the edge frames (first / last 2 frames of each store video). Select on
  annotator SG's 11 pools, measure on the 12 non-SG pools (rd41_3 unknown annotator excluded). Behaviours: nose-nose
  (code nose_nose = nose-nose-mutual OR nose-nose-passive) and nose-tail (code nt). The frames and positives removed by the
  edge rule are reported.
- **Ants**: v3 + v4 annotated videos (147; first 10 min at 1 fps), halves A / B of `results/vision/eci_v2_3/ants/split.json`;
  select on A, measure on B (PRIMARY), and swapped. Behaviour grooming-to-focal (any). Rule: best AUROC on the selection
  half. (If no scored ants frame is an edge frame, the stored v2.3 V / v1 tags are reused unchanged; asserted.)

## Selection rules (mice)

- **auc**: per behaviour the best-AUROC eligible neuron on the selection set (as before).
- **dist** (pre-registered for requirement A): nose-nose = the best-AUROC neuron; nose-tail = the best nose-tail AUROC
  neuron among eligible neurons != the nose-nose pick AND with discrimination AUROC (nose-tail vs nose-nose frames,
  exclusive frames) >= 0.6 on the selection set. Reported: how often the constraint binds (the auc pick differs).
- Eligibility (all rules): readout > 0 on >= 0.2% of the scored frames.

## Metrics (held-out)

AUROC [95% CI] and difference vs V; **precision@top3/video** (per eligible held-out video = annotated, valid, >= 1
positive frame: the 3 frames with the highest RAW frame-max code, ties -> earliest; precision = labelled / (3 x eligible
videos)) with chance (= mean per-video positive rate); ground-truth frame prevalence; distinct picks (nose-nose !=
nose-tail per seed); discrimination AUROC; per-video time bias and MAE (mean over videos +- SE); Pearson r of per-video
time; precision@100 (appendix). Edge-artifact check: fraction of each selected neuron's top-3/video frames lying in the
first or last 2 s (10 frames) of a store video. Bootstrap as v2.3 / v2.4 (mice: held-out pools + seeds; ants: held-out
videos within series x treatment + seeds; 1,000 replicates). Arm - N0 and arm - arm contrasts among the v2.5 arms are
PAIRED (same seed index, same pool / video replicate); contrasts vs V / v1 resample seeds independently.

## Pre-registered decision

Reference: V with rule auc on the edge-excluded frame set (V|auc). Evaluated per arm and mice rule (auc, dist).

- **A passes** (arm, rule) iff: distinct picks 3/3 seeds; nose-tail precision@top3/video >= V's (point estimate) AND
  nose-tail AUROC lower 95% CI of (arm - V) >= -0.02; nose-nose AUROC lower CI of (arm - V) >= -0.03 AND nose-nose
  precision@top3/video lower CI of (arm - V) >= -0.08; AND the best-seed nose-tail sheet (best seed = highest held-out
  nose-tail precision@top3/video, ties -> lower seed; 12 top held-out frames, at most 1 per video, ranked by raw
  frame-max code) shows >= 3/12 labelled frames and plausible nose-to-rear contact on visual inspection (described).
- **B passes** (motion arm X in W1 / Wh / C1, mice rule r) iff at least one of {mice nose-tail AUROC, mice nose-tail
  precision@top3/video, ants grooming-to-focal (any) AUROC, ants precision@top3/video} has (X - N0, paired seeds, same
  rule) lower 95% CI > 0, AND every AUROC metric of that set that is not the winning one has (X - N0) lower CI >= -0.02.
  All motion - N0 contrasts are reported regardless.
- **Ants non-inferiority**: ants AUROC (A -> B) lower CI of (arm - V) >= -0.02.
- **Deploy candidate** = the (arm, rule) passing A, B and ants non-inferiority with the best nose-tail
  precision@top3/video; ties -> the simpler arm (W1 / Wh before C1).

## Escalation ladder (only if nothing passes; each later step = a short amendment committed BEFORE scoring it)

1. Readout-only on the trained arms. Pre-registered NOW (computed with the main tags, consulted only if no (arm, rule)
   passes): rule **dist2048** = dist with the nose-tail pick restricted to latents of the finest Matryoshka level only
   (index >= 512). Box / top10-mean readouts would need a later amendment.
2. Multi-lag concat [x, alpha Delta_{L=1}, alpha Delta_{L=2}] (within the store's +-2 frames).
3. A wider SAE (4096 / 8192 latents, extra Matryoshka level) on the best motion arm.

## Checks (fail loud)

- Featcheck: loader features for 50 windows (40 random + 10 at the edge boundary: first used frames 2-4 / last used
  frames T-5..T-3) == an independent per-token recomputation (memmap reader, python dict lookups), bit-exact, all four
  arms; the training windows cover exactly the frames [2, T - 2) of every video; GPU == numpy features on the first 16
  windows of the training run.
- No edge frame in training (asserted per window) or scoring (asserted in the codes step; edge rows dropped beforehand
  and counted).
- Codes: a direct SAE forward pass on sample frames == the assembled codes.
- Scorer identity: V (s1001-1006) rescored with the v2.5 code on the ORIGINAL ext frame set reproduces v2.4 / v2.3 V
  (rule auc: nose-nose AUROC 0.805, nose-tail 0.791; precision@top3/video 0.270 / 0.086; ants A -> B AUROC 0.861).
- SAE health per arm / seed: FVE, dead fraction.

## Figures (required)

For the deploy candidate (or the closest arm) and v1 seed 0 (mice neurons #611 nose-nose / #414 nose-tail; ants: v1 s0's
grooming pick), per behaviour: 12 top held-out frames of the selected neuron (at most 1 per video, ranked by raw frame
code), the neuron's per-patch activation map overlaid, the ground-truth label marked, neuron id / arm / seed in the title;
same crop size and frame count for v1 and v2. `results/vision/eci_v2_5/figures/`.

## Measured before training (appended before any SAE training started)

Measured 2026-11-05 01:30-01:33 (jobs 2024596-2024606, CPU), BEFORE any v2.5 SAE training started (nothing scored):

| domain | videos | frames used (of all) | used M_t tokens (removed by edge rule) | alpha | E||x - mean x||^2 | E||Delta||^2 | passes | steps x 4096 |
|---|---|---|---|---|---|---|---|---|
| mice | 648 | 3,885,408 (3,888,000) | 747,672,867 (477,314) | 1.12266 | 476.33 | 377.93 | 1 | 182,537 |
| ants | 232 | 1,823,072 (1,824,000) | 104,700,084 (53,950) | 1.18189 | 357.54 | 255.96 | 2 | 51,123 |

alpha sample: 2.0M used tokens of whole random windows (rng 6). Wh uses alpha / 2 (mice 0.56133, ants 0.59094).
Featcheck (`<dom>_featcheck.json`): bit-exact, max abs diff 0, 50 windows (10 at the edge boundary), all four arms;
mice 28,979 tokens, ants 8,187; window coverage == exactly frames [2, T - 2) of every video (both domains).
Scored frames (`eci_codes/v2_5/<dom>/codes/rows.parquet`): mice 864,000 ext frames -> 863,424 (576 edge frames removed =
4 per video x 144 videos); ants 440,412 5 fps span frames of the 147 v3 + v4 videos, 0 edge frames (the spans are
k = 2..2997), so the ants scored frame set (1 fps, k = 2, 7, ..., 2997) is unchanged and the v2.3 V / v1 ants tags are
reused. Seeds 2501-2503. Jobs: one gpu100 job per domain, 12 SAEs each.
