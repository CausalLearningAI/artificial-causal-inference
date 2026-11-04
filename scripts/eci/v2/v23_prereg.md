# ECI v2.3 pre-registration: the V SAE trained on all tokens of all videos

Written 2026-11-04 ~12:45, BEFORE any v2.3 SAE was trained and before any v2.3 score existed. Seen before writing: the
token counts and read benchmarks below, the ext frame counts below, and the v2.2 results
(`results/vision/eci_v2_2/full/summary_full.md`).

## Why

The current v2 SAE (arm V) was trained on a fixed sample of 16M tokens, cycled ~7.7 times (30k steps x 4096):
mice 16M of the ~18.4M training-role tokens drawn from windows (2.1% of the 748M mice M_t tokens of all videos); ants 16M. v1 trained on all
its foreground tokens (67.6M, 5 epochs). SAE training is label-free, so no video needs to be held out: v2.3 trains on
every M_t token of every store video.

## Data (measured, `results/vision/eci_v2_3/<dom>_tokens.json`)

| domain | videos | frames (5 fps) | M_t tokens (mask v5, flag bit0) | per frame | U rows read (store) | V tokens fp16 |
|---|---|---|---|---|---|---|
| mice (v1 432 + v2 216 videos) | 648 | 3,888,000 | 748,150,181 | 192.4 | 1,048,111,383 (1.40 x M) | 1.53 TB |
| ants (store: v2 44, v3 107, v4 81) | 232 | 1,824,000 | 104,754,034 | 57.4 | 155,833,906 (1.49 x M) | 0.21 TB |

ANTS ARE ON HOLD (coordinator, 2026-11-04 12:30): the store holds DINOv3 tokens for only 232 of the 489 Exp2-4 videos
(v2 44/45, v3 107/212, v4 40/113, v4_infested 41/119; v5 and vA not encoded). The ants pipeline reads whatever video list
the store manifest holds; ants training starts only on the coordinator's go, and its counts will be re-measured then.

## How the tokens are fed (no copy of the store)

- Input = V's input x_t + (x_{t+1} - x_{t-1}) / 2 on the M_t tokens, V's edge rule (one-sided difference at a video's
  first / last frame), float32 arithmetic, fp16 to the SAE.
- Videos are cut into windows of 100 frames (mice 38,880 windows); the windows are put in one random order (rng
  20261104) and grouped into chunks of 256 windows (~4.9M tokens, ~152 chunks for mice). Each chunk is permuted per seed
  and consumed in batches of 4096; the tail of a chunk (< 4096 tokens) is carried into the next chunk. One pass = every
  M_t token exactly once (except the final tail, < 4096 tokens).
- Measured (mice, 16 threads, random windows, CPU node): raw store reads 12.4 GB/s (128 windows; part may come from
  the BeeGFS client cache), but V features built in numpy only 0.21M tokens/s (0.02M/s on one thread). V needs
  ~0.46M tokens/s per SAE (30k steps in 261-270 s on an L40S = 8.8-9.0 ms/step). So the loader threads only read raw rows
  and compute the neighbour index; the features are built on the GPU (same float32 ops; checked exactly equal to the
  numpy features on the first 16 windows of every run, fail loud; the numpy features equal v22_full.feats_v exactly,
  `<dom>_featcheck.json`).
- The 3 seeds of a domain are trained in ONE job on ONE stream (one store read for all seeds, ~2.1 TB for mice). Per seed:
  own init, TokenNorm and geometric-median b_dec (from 250k tokens of the first chunk), own batch permutation in every
  chunk. Shared: the window order. Stated as a difference from V, whose seeds also differed in sampling order.
- GPU utilisation, steps/s, tokens/s and loader wait are logged every 2,000 steps. A GPU below 50% utilisation is a
  bug to fix, not to wait out.

## Training budget

- Main arm **Vall**: one pass over all mice M_t tokens: **182,653 steps x 4096** (= floor(748,150,181 / 4096)), 1.00
  epoch. Same recipe as V otherwise: Matryoshka BatchTopK 2048 latents, k = 16, prefixes 128/256/512/2048, AuxK as in
  V (trainer defaults), Adam lr 5e-4, 500 warm-up steps, cosine decay over the 182,653 steps.
- Seeds **2001, 2002, 2003** (mice first; ants later, same seeds).
- Projection: 182,653 steps x 3 SAEs x ~9 ms = ~82 min on an L40S-class GPU; less on an H100. A short benchmark
  (3,000 steps, same job) records the measured step time first.
- GPU: `-p gpu` was estimated to start 2026-11-20 (> 3 h), so the SAE trainings use `-p gpu100` (allowed, <= 12
  H100-hours in total). Encodes never use gpu100 (codes run on CPU).
- Fallback, only if one pass is impractical (projected > 4 h for the 3 seeds or > 12 H100-hours): train on a uniform
  temporal subsample = the M_t tokens of every 5th frame (t mod 5 = 0, 1 fps-equivalent, 20% of tokens), steps =
  N_used / 4096. Not expected to be needed.
- theta* (calibration) and FVE (reconstruction): the v2.1 calib / eval V token samples (1.5M tokens each,
  `eci_codes/v2_1/<dom>/data/{calib,eval}_{B,C}.npy`), i.e. exactly the tokens V was calibrated / evaluated on. Their
  videos are training videos here (overlap stated), so FVE is in-sample-video for Vall and out-of-video for V.

## Evaluation frames ('ext', user decision): every labelled mice frame

The atlas window (648,000 frames = v2.2 FULL WINDOW) + the labelled FIRST 15 min of the 48 habituation (_H) videos
(frames 0-4499, 216,000 frames) = **864,000 frames** (`results/vision/eci_v2_3/ext/frames.md`). Counted before scoring:

| frame set | pools | videos | frames | nose-nose (nose_nose) frames / bouts | nose-tail (nt) frames / bouts |
|---|---|---|---|---|---|
| ext, all | 24 | 144 | 864,000 | 18,269 / 5,205 | 9,032 / 1,600 |
| ext, SG (selection) | 11 | 66 | 396,000 | 10,263 / 3,039 | 5,071 / 1,031 |
| ext, non-SG (evaluation) | 12 | 72 | 432,000 | 6,783 / 1,864 | 3,652 / 510 |
| ext, unknown annotator (rd41_3) | 1 | 6 | 36,000 | 1,223 / 302 | 309 / 59 |
| of which first 15 min of H, non-SG | 12 | 24 | 108,000 | 2,215 / 586 | 956 / 179 |

(nose-nose = nose-nose-mutual (nn) OR nose-nose-passive (np); bouts counted inside each set.)

## Models scored on the ext frames

- **V**: seeds 1001-1006 (the stored v2.1 SAEs), re-encoded on all 864,000 frames.
- **Vall**: seeds 2001-2003.
- **v1 s0**: the deployed v1 codes (`dataset/mice/v1/eci/codes/matryoshka_btk_1024_k16_fg448al_s0/codes_max.npy`, the
  source of the Tier A extract used for v2.2's full window) on all 864,000 frames. v1 s1-s10: only if GPU job 1959004
  (v1 codes, -p gpu, est. start 2026-11-05) runs in time; otherwise not reported.
- Codes: frame max over the M_t tokens (top10 also stored for reference), one token read for all v2 SAEs (v22_full
  codes2, CPU array). P(on) = the 3-state HMM per neuron, label-free, ONE sequence per video over all its ext frames
  (H videos: 9,000 frames). Because the H sequences are longer, V's P(on) on the atlas frames differs from v2.2's (stated).

## Scoring (the v2.2 full pipeline, `V22_FRAMES=ext`, no forked code)

- Selection on the SG pools (11; rd41_3 unknown, excluded), evaluation on the 12 non-SG pools (design (a), PRIMARY);
  design (b) (cross-fitted within SG) reported.
- Rules: **auc (primary: the current V rule)**, bal (reported), hyb (reported, post hoc in v2.2).
- Metrics: prevalence, AUROC, precision@100, F1 (P(on) > 0.5), discrimination AUROC (nose-tail vs nose-nose), per-video
  time bias / MAE (mean over videos +- SE; raw and calibrated) with the constant-rate reference, Pearson r of per-video
  time. Bootstrap over held-out pools and seeds, 1,000 replicates (v22_eval report).
- Contrasts: Vall|auc - V|auc (PRIMARY), Vall|bal - V|bal, Vall|hyb - V|hyb, Vall|auc - v1s0|auc, V|auc - v1s0|auc.
- Ants: grooming-to-focal (any) (cell pooled|groom_any) with v22_ants.py (12 cross-fitted directions over videos,
  every labelled frame, video bootstrap within series): AUROC and Pearson r of per-video time; Vall - V.

## Adoption criteria (Vall vs V, all must hold)

- A1 mice nose-nose AUROC, Vall|auc - V|auc: lower 95% CI >= -0.02.
- A2 mice nose-tail AUROC, Vall|auc - V|auc: lower 95% CI >= -0.02.
- A3 distinct picks: under rule auc, the nose-nose and nose-tail neurons differ in >= 2 of the 3 Vall seeds.
- A4 ants grooming-to-focal (any): AUROC and per-video-time Pearson r, Vall - V: lower 95% CI >= -0.02 (pending the ants
  go; until then the verdict is "mice criteria only").
- Primary interest (reported, not a criterion): does nose-tail improve - AUROC, precision@100, discrimination AUROC, and
  whether the top-12 held-out frames of the Vall nose-tail neuron show real nose-to-tail contact (sheets).

## Identity checks (fail loud)

- Streamed V features == v22_full.feats_v (exact; random windows + video-edge windows) and GPU == numpy features in
  every training run.
- ext frame table restricted to the atlas window == the v2.2 full frame table (order, rows, labels; asserted).
- ext codes of V s1001-1006 on the 648,000 atlas frames == the v2.2 full-window codes (bit-identical expected: same CPU
  code path; otherwise only threshold flips), and the v2.2 full report (V22_FRAMES=full) re-run after the edits to the
  v22 scripts reproduces `results/vision/eci_v2_2/full/summary_full.md` values.
- New Vall codes on a sample of frames == a direct SAE forward pass (frame max of encode(norm(x)) over the frame's
  tokens).
- Sheets: top-12 held-out frames of Vall (seed 2001) for nose-nose and nose-tail, rule auc (and bal), read and described.

## AMENDMENT 1 (2026-11-04 ~13:05, user decision relayed by the coordinator; BEFORE any ants v2.3 SAE or any v2.3 score)

Supersedes the ants hold above and the ants parts of "Models scored", "Scoring" and A4.

**Ants training set (asserted in code, `v23_train.ants_rule_videos`)**: the valid videos used downstream, at full
length: all valid v2 (valid == 1: 44 of 45), v3 treatments 2, 6, 8 (107), v4 treatments 2, 6 (40), v4_infested treatments
2, 6 (41) = **232 videos**, exactly the 232 videos of the v2.1 store (the store keeps v4_infested under 'v4'); the code
asserts store set == rule set. All of them are training videos, no calib / eval role split. All M_t tokens: measured
**104,754,034** (1,824,000 frames, 57.4 per frame; the coordinator's "~99M" was an estimate) -> one pass = **25,574 steps**
x 4096, same recipe, seeds 2001-2003, one job on gpu100 (same <= 12 H100-hour cap). theta* / FVE: the v2.1 ants calib /
eval V samples (1.5M each; overlap with training, stated).

**Ants alignment design**
- Frames: the eval.py ants frame table (the human-labelled first 10 min of each video, 1 fps, 600 frames per video),
  ONLY series v3 and v4: **147 videos** (v3 107: treatments 2 / 6 / 8 = 38 / 34 / 35; v4 40: treatments 2 / 6 = 20 / 20),
  88,200 frames. v2 and v4_infested are excluded from neuron selection and scoring.
- Split by video into halves A and B, stratified by series x treatment: within each of the 5 strata the videos (sorted
  by name) are permuted with numpy default_rng(**20261105**); positions 0, 2, 4, ... -> A, 1, 3, 5, ... -> B (an odd
  stratum gives A one more video). PRIMARY direction: select on A, measure on B. Also reported: select on B, measure on A.
- Behaviours: **grooming-to-focal (any)** (= Y2F OR B2F; focus), Y2F, B2F, focal single (Y2F XOR B2F), focal double
  (Y2F AND B2F).
- Rule: per behaviour, the eligible neuron (readout > 0 on >= 0.2% of the 88,200 frames) with the best AUROC on the
  selection half. Readouts as for mice: v2-type models (V s1001-1006, Vall s2001-2003) = per-neuron 3-state HMM P(on)
  fitted label-free on the 5 fps spans of every ants eval video (all series; one sequence per video, the fast HMM of
  v22_full checked == src/eci/v2/hmm.py), read at the scored frames; v1 (s0-s10, its stored codes on these frames) = raw
  frame code for ranking, the same HMM on the picked neuron's 1 fps code (one sequence per video) for F1 / time.
  v1 s0 alone is reported too.
- Metrics on the held-out half: prevalence, AUROC, precision@100, F1 (P(on) > 0.5), per-video time bias / MAE (raw and
  calibrated by total true / total sum P(on) on the selection half) with the constant-rate reference, Pearson r of
  per-video time. Bootstrap: held-out videos resampled within series x treatment, and seeds; 1,000 replicates.
- Contrasts: Vall - V, Vall - v1, V - v1, Vall - v1s0.
- **A4 (replaces the earlier A4)**: grooming-to-focal (any), primary direction (A -> B): Vall - V AUROC lower 95% CI >=
  -0.02 AND Pearson r of per-video time lower 95% CI >= -0.02.
