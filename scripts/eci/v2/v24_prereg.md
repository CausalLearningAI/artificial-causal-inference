# ECI v2.4 pre-registration: SAEs on average-pooled DINOv3 tokens with weighted motion

Written 2026-11-04 ~16:05, BEFORE any v2.4 SAE was trained and before any v2.4 score existed. Seen before writing: the
v2.3 results (`results/vision/eci_v2_3/ext/summary.md`, `ants/summary.md`), the v2.2 VP2 result (2x2-pooled SAE trained
on 6.65M tokens, old 2-min windows: nose-nose AUROC 0.658 vs V 0.830, nose-tail 0.750 vs 0.777), the precision@top3/video
numbers of V / Vall / v1 (`results/vision/eci_v2_4/metric_top3/summary.md`), and the throughput of a 2-video pooling
smoke test (no SAE, no score).

## Why

User request (v2.4): average-pool the DINOv3 token grid BEFORE the SAE, to approach v1's patch size (v1 = DINOv2-base,
448 px, 32 x 32 grid; mice native v2 grid = 64 x 64 at 1024 px, so 2x2 pooling gives exactly v1's 32 x 32 grid), 2x2
and a stronger 4x4; temporal context (speed) on by default, as a PROPERLY WEIGHTED sum (the linear probe found that V's
unweighted x + (x_{t+1} - x_{t-1}) / 2 adds nothing; the user doubts this is a fair test of motion).

## Input (code: `scripts/eci/v2/v24_pool.py`, `v24_train.py`)

Token store: v2.1 store (`eci_tokens/v2_full/<dom>/v2.1_m5`): U_t = union of M_s for s in t-2..t+2, so every M_t
position is stored at frames t-2..t+2 inside a video. The largest symmetric lag the store supports is therefore
**L = 2 frames = +-0.4 s at 5 fps** (no re-encoding, no larger lag).

- **Pooling rule (= VP2, `v22_encode.pool2`)**: a pooled cell of frame t exists iff >= 1 M_t token (mask v5, flag
  bit0) lies in its P x P block of the native grid (G -> ceil(G / P); mice G = 64 -> 32 (P = 2) / 16 (P = 4); ants
  G = 48 -> 24 / 12; all grids divide evenly, so no partial border blocks occur; the code handles them as pool2 does:
  mean over the tokens the block has). Cell value = mean over the block's M_t tokens; n = their number (reported:
  native tokens per pooled token).
- x_t = pooled token at t. **Delta_t** = mean over the SAME M_t positions of (x_{t+2} - x_{t-2}) (the same pooling
  applied to the neighbour frames before differencing). Edge rule (first / last 2 frames of a video):
  2 (x_{t+2} - x_t) or 2 (x_t - x_{t-2}) when only one neighbour exists (same scale as the central difference), 0 when
  none.
- **alpha**: one global scalar per domain and pooling size P, fixed BEFORE training, label-free, over ALL pooled tokens
  (a superset of a sample): alpha = sqrt(E||x - mean(x)||^2 / E||Delta||^2), so alpha Delta has the same energy as the
  centred appearance part. Logged in `results/vision/eci_v2_4/<dom>_pool.json`. P2W and P2C use the same alpha (P = 2);
  P4W uses the alpha of the 4x4-pooled tokens (the same rule on its own tokens).
- Arithmetic: float64 sums of the fp16 tokens (exact), mean -> float32 -> fp16 stored; arm features in float32, fp16 to
  the SAE.

## Arms (each: seeds 2401, 2402, 2403; mice and ants)

| arm | tokens | SAE input | d_in |
|---|---|---|---|
| **P2W** | 2x2 pooled | x_t + alpha Delta_t | 1024 |
| **P4W** | 4x4 pooled | x_t + alpha Delta_t | 1024 |
| **P2N** (control: no motion) | 2x2 pooled | x_t | 1024 |
| **P2C** (secondary) | 2x2 pooled | [x_t, alpha Delta_t] | 2048 |

SAE = V's recipe (src/eci/sae: Matryoshka BatchTopK, 2048 latents, k = 16, prefixes 128/256/512/2048, AuxK, Adam
lr 5e-4, 500 warm-up steps, cosine to 10%, batch 4096, TokenNorm, geometric-median b_dec). TokenNorm uses ONE scalar
scale for the whole input (also for P2C), so alpha keeps its meaning. lr / batch unchanged (the pooled token counts do
not force a change: >= 50k steps of 4096 in every arm).

Training set: every pooled token of every store video, full length, no train / test video split (SAE training is
label-free): mice = 648 videos (v1 432 + v2 216); ants = the 232 store videos (v2 44, v3 107 t = 2/6/8, v4 40 +
v4_infested 41 t = 2/6; asserted = the v2.3 AMENDMENT 1 rule).

**Training length rule**: passes = max(1, ceil(50,000 x 4096 / N_pooled)), steps = floor(passes x N_pooled / 4096)
(>= 1 full pass and >= ~50k steps). Stream: the pooled store rows cut into segments of 8,192 rows, segments in a fresh
random order per pass (rng 20261104 + pass), chunks of 320 segments, each chunk permuted per seed, tail carried. All
arms of one P train in one job on one stream; within a seed, every arm sees the same batches (paired). The measured
counts, alpha, passes and steps are appended below before training starts.

theta* / FVE: two fixed label-free samples of 1.5M pooled tokens (random segments, rng 7 / 8); they are training tokens
too (stated). SAE health reported per arm / seed: eval FVE, L0, dead latents at the end of training (no firing in the
last 2M tokens) and the fraction of latents never firing on the eval sample.

GPU: `-p gpu100`, hard cap 5 H100-hours for v2.4 in total (logged).

## Codes and readouts

Frame max over the pooled cells of a frame of encode(norm(input)) (threshold mode), computed from the pooled store.
P(on) = per-neuron 3-state HMM (v22_full.pon_fast), label-free, one sequence per video (mice: all ext frames of the video;
ants: the 5 fps span of every ants eval video), exactly as v2.3.

## Evaluation (as v2.3; only the arms and the headline metric change)

- **Mice**: ext frames = **864,000** (648,000 atlas window + first 15 min of the 48 H videos; v2.3 count, re-asserted).
  Select on annotator SG's 11 pools, measure on the 12 non-SG pools (432,000 frames; rd41_3 unknown annotator excluded).
  Behaviours: nose-nose (nose_nose = nose-nose-mutual nn OR nose-nose-passive np) and nose-tail (nt). Rules: **auc
  (primary)**, bal (reported), hyb (exploratory). Eligibility: readout > 0 on >= 0.2% of the frames.
- **Ants**: v3 + v4 annotated videos (147; first 10 min at 1 fps), halves A / B of `results/vision/eci_v2_3/ants/split.json`,
  select on A, measure on B (PRIMARY) and swapped. Behaviour grooming-to-focal (any). Rule: best AUROC on the selection
  half.
- **Baselines** (existing caches / tags reused, not recomputed): V (seeds 1001-1006), Vall (2001-2003), v1 seed 0 on
  the ext frames; v1 seeds 0-10 only where available (mice: atlas frames only, reported from `metric_top3`, different
  frame set, stated; ants: s0-s10).
- **Metrics** (held-out): AUROC [95% CI] and arm - V; **precision@top3/video** (headline, replaces precision@100): per
  eligible held-out video (annotated, valid, >= 1 positive frame) the 3 frames with the highest RAW frame-max code of the
  selected neuron (ties -> earliest frame); precision = labelled frames / (3 x eligible videos); chance = mean per-video
  positive rate (functions of `scripts/eci/v2/v24_top3.py`, reused); ground-truth frame prevalence; distinct picks
  (nose-nose neuron != nose-tail neuron per seed); discrimination AUROC nose-tail vs nose-nose; per-video time bias and
  MAE (mean over videos +- SE); Pearson r of per-video time; F1 (P(on) > 0.5); precision@100 kept as a secondary column.
- **Bootstrap**: as v2.3 (mice: held-out pools + seeds; ants: held-out videos within series x treatment + seeds; 1,000
  replicates); differences = mean over seeds, CI = 2.5 / 97.5 percentiles of the replicate differences.

## Adoption criteria (per arm, vs V|auc; mice measured on non-SG, ext frames, rule auc)

- **B1** nose-nose AUROC: lower 95% CI of (arm - V) >= -0.02.
- **B2** nose-tail precision@top3/video: arm - V > 0 with lower 95% CI > 0, AND nose-tail AUROC: lower 95% CI of
  (arm - V) >= -0.02.
- **B3** distinct nose-nose vs nose-tail picks in all 3 seeds.
- **B4** ants grooming-to-focal (any), A -> B: AUROC lower 95% CI of (arm - V) >= -0.02.
- ADOPT an arm iff B1-B4 all hold. If several pass: higher mean of nose-nose and nose-tail precision@top3/video.

**Secondary question (not a criterion): does weighted motion help?** P2W - P2N on every metric with 95% CI (paired
seeds 2401-2403, same batches), mice and ants. Also reported: P2C - P2W, P4W - P2W, arm - Vall, arm - v1s0.

## Checks (fail loud)

- Pooling identity: 3 random frames + 2 video-edge frames per domain and P: stored x / Delta == an independent per-cell
  float64 python loop over the native store (bit-exact); cell set == `v22_encode.pool2` (VP2's rule) on the same frame;
  max |pool2 float32 mean - stored x| reported.
- Featcheck: loader features (stored rows + arm feature function) for 50 windows (40 random + 10 at video edges) ==
  an independent recomputation from the native store, bit-exact, arms W / N / C; and GPU == numpy features on the first
  16 segments of every training run.
- Codes: a direct SAE forward pass on sample frames == the assembled codes.
- Scorer identity: V (s1001-1006) re-tagged and re-reported with the v2.4 scoring code reproduces the V AUROCs of
  `results/vision/eci_v2_3/ext/summary.md` (3 decimals), and V's precision@top3/video equals `metric_top3`.
- Top-frame sheets for the best seed of each arm (highest mean of nose-nose / nose-tail precision@top3/video; ants: the
  grooming precision@top3/video): mice nose-nose and nose-tail neurons, ants grooming neuron; count of labelled frames
  among the 12 shown, looked at and described.

## Measured before training (appended before any SAE training started)

Measured 2026-11-04 16:18-16:25 (pool jobs 2006961 / 2006962, alpha / check jobs 2008422 / 2008502), BEFORE any v2.4
SAE training job started (ants training started 16:23:48 after this table was computed; nothing scored):

| domain | P | pooled tokens | native M_t tokens per pooled token | alpha | E||x - mean x||^2 | E||Delta||^2 | passes | steps x 4096 |
|---|---|---|---|---|---|---|---|---|
| mice | 2x2 | 269,779,550 | 2.773 | 1.2798 | 398.4 | 243.3 | 1 | 65,864 |
| mice | 4x4 | 109,859,600 | 6.810 | 1.3740 | 348.8 | 184.7 | 2 | 53,642 |
| ants | 2x2 | 42,679,289 | 2.454 | 1.2437 | 283.5 | 183.3 | 5 | 52,098 |
| ants | 4x4 | 19,125,944 | 5.477 | 1.2919 | 238.4 | 142.9 | 11 | 51,363 |

Pooled cells cover every M_t token of the store (mice 748,150,181, ants 104,754,034; asserted). Lag L = 2 frames (0.4 s).
Checks passed: pooling identity bit-exact (5 frames per domain and P incl. 2 video-edge frames; cell set == pool2;
pool2's float32 mean cast to fp16 == stored x), featcheck bit-exact (50 windows incl. 10 video edges, arms W / N / C,
both domains, both P). Seeds 2401-2403. Jobs: one per (domain, P, arm group): P2 W+N (6 SAEs), P2 C (3), P4 W (3).
