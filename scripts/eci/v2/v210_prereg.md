# ECI v2.10 pre-registration: the PRED-contact recipe (PREDk, k = 16) on the validation benchmarks + a frogs dictionary

Written 2026-11-26, BEFORE any v2.10 SAE was trained, any v2.10 code was extracted and any v2.10 score existed.
Seen before writing: everything up to v2.9 (lab results: PREDk clearly better than v1 on mice, level on ants), the v2.1
CalMS21 benchmark check (`results/vision/eci_v2_1/bench/cal_report.md`, recipe V mean AUROC P(on) 0.955 median of 3 seeds).

## Question
On each validation benchmark, is the new recipe (PREDk, k = 16) at least as good as the November-frozen recipe V
(2048-latent Matryoshka, input x + d1, no temporal prediction) on behaviour alignment of single neurons?

## What exists (recon) and what is compared
- Token store: `eci_tokens/v2_full/<set>/v2.1_m5` (DINOv3-L/16 block 20, 5 fps, mask v5 = flag bit0 M_t; benchmark masks =
  the approved keypoint-guided mask-v5 variants). Sets used: `cal` (CalMS21, 89 videos), `mab` (MABe22 MOUSE triplets,
  2,612 clips = userTrain + submission), `crim` (CRIM13, 236 videos), `frogs` (153 videos: 35 juveniles + 118 tadpoles).
- **MABe22 beetles are NOT in the v2_full store** (only mouse clips were encoded), so they are not scored here.
- **Recipe V baseline:** only CalMS21 was ever scored with recipe V on this store (v2.1 bench, `v21_bench.py`, seeds
  2001-2003; SAEs and codes on BeeGFS `eci_codes/v2_1/bench_cal`). Those SAEs and codes are reused UNCHANGED.
  MABe22 and CRIM13 have no recipe-V numbers on this store, so recipe V is trained on them now with the SAME code path
  (`v21_bench.py` prep / train, copied with the benchmark as a parameter), seeds 2001-2003.
- **v1 has never been run on these benchmarks** (v1 dictionaries are lab-domain only; different encoder and store).
  v1 is therefore NOT compared on the benchmarks; this is reported, not filled in.

## Data per benchmark (labels are read only for scoring)
- SAE pool = label-free training videos; scored videos = the benchmark's labelled evaluation videos.
- **CalMS21:** pool = the 70 task-1 train videos, split exactly as v2.1 (`v21_bench.video_lists`, rng 0: 58 train /
  6 calib / 6 eval). Scored = the 19 task-1 test videos. Behaviours attack, investigation, mount. Label at source frame
  6k (offset 0). Cross-fit halves = `scripts/benchmark/calms21/eci_train.halves(E, s, labels)`, s = 0..4, both directions
  (10 directions) = v2.1 unchanged.
- **MABe22 (mouse triplets):** pool = the userTrain clips in the store; scored = the submission clips in the store
  (clips in `mouse_frame_number_map.npy`). Pool split: sorted clips permuted with rng 0, 10% calib, 10% eval, 80% train
  (rounded). Behaviours = `eci_common.PRIMARY` (oral_oral_contact, oral_genital_contact, oral_ear_contact, approaches,
  huddles). Label index = 6k - 1 (offset.json: video frame = label index + 1); frames with no label index (k = 0) are not
  scored; positive = label value 1. Halves = `scripts/benchmark/mabe22/eci_train.halves(E, s)` (clip level), s = 0..4,
  both directions.
- **CRIM13:** pool = the released TRAIN-half videos (103), split as MABe22 (rng 0, 10% / 10% / 80%). Scored = all 236
  videos. Behaviours = `align_common.SCORED` (11: approach, attack, copulation, chase, circle, drink, eat, clean, sniff,
  up, walk_away); pair subset `align_common.PAIR` reported as a sub-mean. Label at source frame 5k (offset 0); frames
  whose label is -1 (uncovered) are not scored; one-vs-rest ('other' and 'human' are negatives). Halves = the released
  train / test halves, both directions (= every earlier CRIM13 check).
- **Scored frame set:** ALL 5 fps store frames of the scored videos that carry a label (not a weighted sample as in the
  October benchmark studies). The exact counts vs all annotated source frames are reported in tables.md.
- **Frogs:** no frame labels. SAE pool = all 153 store videos (juveniles + tadpoles, one dictionary per species as on
  the lab sets). Health only + codes for later NES.

## Arms
- **V (baseline)** = v2.1 bench recipe: Matryoshka BatchTopK 2048 latents, prefixes 128/256/512/2048, k 16, 30,000 steps,
  batch 4096, input x + (x_{t+1} - x_{t-1}) / 2, <= 8M train / 1M calib / 1M eval tokens of the pool roles.
- **PREDk16 (new)** = the v2.9 training path, recipe unchanged: input x_t, 4096 latents, prefixes 128/256/512/2048/4096,
  k 16, second decoder predicting x_{t+2}, loss L0 + 1 x L1, per-token weight W_C on contact-proxy tokens, AuxK,
  Adam 5e-4, 500 warm-up, cosine, batch 4096, TokenNorm, window stream (windows of 100 centre frames inside the edge rule
  2 <= t < T - 2, chunks of 256 windows, per-seed batch permutation, all seeds see the same batches).
  Seeds 3001, 3002, 3003, one job per set.
  - Training videos: the pool's TRAIN role only (CalMS21 58, MABe22 / CRIM13 80% of the pool), i.e. the same videos as
    V's training tokens; frogs: all videos.
  - W_C measured label-free per set by the v2.7 prep rule (proxy tokens carry 20% of the weight; cap 30; 2M-token sample
    of windows in rng-6 order; proxy identity check on 20 windows must give 0 mismatches). If a set has NO proxy token
    in the sample, W_C = 1 (PREDk = PREDc) and this is reported.
  - **Steps rule** (the ants rule extended to small sets): one pass over all train-role M_t tokens if that pass has
    >= 25,561 steps (the ants v2.7 / v2.9 run); otherwise the smallest whole number of passes P reaching >= 25,561 steps.
    steps = floor(P x N / 4096). Pass p > 0 uses the window order of rng 20261105 + p (pass 0 = the v2.7 order).
    The LR schedule length = steps.
  - theta* calibrated on 1.5M training-window tokens (rng 7) as v2.9; health FVE / L0 also measured on V's held-out
    'eval' role tokens (the same tokens as V's eval FVE) so the two recipes are compared on identical held-out tokens.
- Code changes to the v2.9 path: only (a) the video list (pool train role), (b) multi-pass chunk order, (c) W_C from
  this run's prep, (d) output paths. **Identity check first:** the new training script with mice, k 16, seed 2701,
  5,001 steps must reproduce the stored v2.7 PREDk_s2701 history at steps 0 and 5,000 exactly (loss, L0, L1, aux,
  n_dead, fve, l0). Any difference stops the chain (training jobs depend on it).

## Codes and readouts (identical for V and PREDk)
- Frame code = max over the frame's M_t tokens of the theta* SAE code (`v21_bench.cmd_codes`), every 5 fps frame of every
  scored video (frames without tokens = 0). PREDk encodes x_t only (static input).
- Readouts: raw frame max, and P(on) = the per-neuron 3-state HMM fitted label-free on the scored spans (as v2.1).

## Metrics (per dictionary, behaviour, direction; mean over directions)
- Eligible neurons (as v2.1): readout > 0 on >= 0.2% of the scored frames.
- **AUROC** (P(on)) of the neuron with the best AUROC on the selection half, measured on the other half. Chance 0.5.
- **AP** (P(on)) of the neuron with the best AP on the selection half, measured on the other half. Chance = prevalence of
  the held-out half; the overall prevalence is printed next to it.
- **precision@top3/video**: for the AP-picked neuron, on each held-out video with >= 1 positive frame, the 3 frames with
  the highest RAW code (earliest first on ties); fraction positive, averaged over those videos. Chance = mean per-video
  prevalence over the same videos.
- Secondary (reported, not decided on): the same three on the raw readout; v2.1's nAP and top-1% precision (kept so the
  stored CalMS21 V numbers can be re-derived exactly: the re-scored V must equal `cal_scores.json` on AUROC / nAP / top-1%
  for both readouts, else the scorer is wrong and nothing is reported).

## Decision (per benchmark, per behaviour, per primary metric; applied literally)
- best_P = max over the 3 PREDk16 dictionaries; best_V = max over the 3 V dictionaries; spread = the larger of the two
  arms' seed ranges (max - min over 3 seeds) for that behaviour and metric.
- PASS if best_P >= best_V - spread. A behaviour is "at least as good" if all three primary metrics PASS.
- Reported per behaviour with both numbers side by side, plus per-benchmark counts and the mean over behaviours.
- The benchmark answer: "new recipe at least as good as V on <benchmark>" iff every behaviour of that benchmark is
  "at least as good". Anything else computed later is labelled POST HOC.

## Frogs
- Health per dictionary: eval FVE (rng-8 1.5M training-window tokens, as v2.9), L0 per token, dead latents (train) and
  never-firing fraction (eval). Codes: frame max over M_t tokens, every frame of all 153 videos, for all 3 seeds.

## Budget and paths
- GPU partition gpu100 (= v29_train.sh); total <= 12 GPU-hours; steps/s and GPU utilisation logged.
- Big files: `/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_10/`; results `results/vision/eci_v2_10/`.
- Code: `scripts/eci/v2/v210_train.py` (prep / train / identity), `scripts/eci/v2/v210_bench.py` (V prep / train,
  codes, score, report), job wrappers `v210_gpu.sh`, `v210_cpu.sh`, chain `v210_submit.sh`.
