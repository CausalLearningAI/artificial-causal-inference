# ECI v2.4 log: SAEs on average-pooled DINOv3 tokens with weighted motion

Goal (user): average-pool the DINOv3 token grid before the SAE (2x2 and 4x4), motion on by default as a properly
weighted sum. Pre-registration: `scripts/eci/v2/v24_prereg.md` (commits 716156c, d32ee12, both before any training /
score). Code: `scripts/eci/v2/v24_pool.py` (pooled store, checks, alpha), `v24_train.py` (trainer, codes, featcheck),
`v24_eval.py` (cache / tag / prep / report / fwdcheck / sheets; reuses v22_core, v22_eval, v22_full, v23_ants, v24_top3
by patching output paths only), `v24_summary.py`. Big files: `/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_4/`
(pool/, sae/, codes/, cache/).

## Timeline (2026-11-04)
- 15:56 read v2.3 / v2.2 code. Store check: U_t = union of M_{t-2..t+2} -> largest symmetric lag L = 2 frames (0.4 s).
- 15:59 pooling smoke test: first numpy version 7.7k native tokens/s (too slow) -> rewritten as exact float64 sparse
  sums (every fp16 is a multiple of 2^-24, so float64 sums are exact and order-independent); P4 sums = sums of P2 sums.
- 16:03 pooled store, CPU arrays: mice 216 tasks (2006961), ants 116 tasks (2006962), 2 CPUs each; ~4k native
  tokens/s per task under contention; longest task 1,169 s. Every M_t token covered (asserted: mice 748,150,181,
  ants 104,754,034).
- 16:05 pre-registration committed (716156c).
- 16:18-16:25 alpha / identity jobs (2008422, 2008502): see `<dom>_pool.json`, `<dom>_poolcheck.json`,
  `<dom>_P<P>_featcheck.json`. Pooling identity bit-exact on 5 frames per domain and P (incl. first / last frame of a
  video); cell set == v22_encode.pool2 (VP2's rule); pool2's float32 mean cast to fp16 == stored x (max float32 diff
  0.004-0.008 = fp16 rounding). Featcheck bit-exact (50 windows incl. 10 video edges, arms W / N / C).
  alpha: mice P2 1.2798, P4 1.3740; ants P2 1.2437, P4 1.2919 (all > 1: at lag +-2 the centred motion energy is
  0.61 (mice P2) of the centred appearance energy).
- 16:23 training: ants (2008496-8), 16:30 mice (2008516-8), gpu100 H100s. GPU == numpy features bit-exact on the first
  16 segments of every run. Measured values appended to the pre-registration (d32ee12) before any score.
- 16:34 all 6 jobs loader-bound on BeeGFS (0.3-0.4 GB/s each, GPU util ~0-2%); projected total > 5 H100-hours. Ants
  jobs cancelled after 10.9 min each (0.54 H100-hours lost) and restarted (2008536-8) with a host-RAM segment cache
  (V24_RAM=1: passes >= 2 read from RAM; same stream, same order, same steps; only the read path differs). Ants then ran
  GPU-bound (60-77 steps/s, GPU util ~80%). Mice stayed loader-bound (1 or 2 passes, cache not useful there).
- A CPU stopper job (2008687) cancelled each training job as soon as all its checkpoints were saved; codes then ran on
  CPU arrays (24 tasks) instead of on the H100 (saves H100 time; same code path, `v24_train.codes`).
- 17:13 all 24 SAEs trained (commit b91c6e2). H100 time: 3 x 10.85 (cancelled) + 47.0 + 45.3 + 37.7 (mice) +
  31.4 + 27.6 + 21.8 (ants) min = **243 min = 4.06 H100-hours** (cap 5).
- Scorer identity (before any v2.4 model was scored): V s1001-1006 re-tagged with the v2.4 code reproduces the v2.3
  ext AUROCs exactly (nose-nose 0.8053, nose-tail 0.7906 rule auc; bal / hyb also equal; precision@100 equal) and
  precision@top3/video equals metric_top3 (nose-nose 0.270, nose-tail 0.086). Ants: V / Vall / v1 / v1s0 AUROC equal
  to v2.3 (A2B V 0.8612) and top3 equal to metric_top3 (v1 0.801, v1s0 0.788).
- 16:56-17:13 codes on CPU arrays (24 tasks per domain / P group); per model: merge, HMM cache (mice ~15-25 min, ants
  ~25 min), tag, prep, fwdcheck (`v24_post.sh`). Direct forward pass vs assembled codes on 200 frames: max |diff|
  0 for most models, 0.0039 (= 1 fp16 step at that magnitude; CPU batch-size dependent matmul rounding) for ants P4W
  s2402 and P2C s2402 (log lines; `codes/fwdcheck_<dom>_<key>.json` holds the LAST domain run per key only: a file-name
  bug, the logs keep both).
- 17:38 ants report, 18:12 mice report (`ants/report.json`, `mice/report.json`, `summary.md`).

## Results (rule auc, mice design (a) on ext frames, ants A -> B)
- **No arm adopted.** B1 fails for every arm (nose-nose AUROC arm - V lower CI: P2W -0.138, P4W -0.195, P2N -0.046,
  P2C -0.121); B2 fails for every arm (nose-tail precision@top3/video arm - V: P2W -0.036, P4W -0.068, P2N +0.033
  [lower CI -0.056], P2C +0.056 [-0.019, +0.134]); B3: P4W 3/3, P2C 3/3, P2W 2/3, P2N 1/3; B4 fails for every arm
  (ants grooming AUROC arm - V lower CI: P2W -0.050, P4W -0.183, P2N -0.048, P2C -0.021).
- Motion question, P2W - P2N (same pooling, paired seeds): mice nose-nose AUROC -0.056 [-0.126, +0.004], nose-tail
  AUROC -0.040 [-0.113, +0.018], nose-tail precision@top3/video -0.069 [-0.148, +0.007]; ants grooming AUROC -0.007
  [-0.028, +0.015], precision@top3/video -0.086 [-0.154, -0.020]. Weighted-sum motion does not help; it tends to hurt.
  Concatenation behaves differently: P2C - P2W nose-tail AUROC +0.058 [+0.003, +0.132], top3 +0.092 [+0.014, +0.177];
  ants AUROC +0.029 [+0.009, +0.052]; P2C - P2N ants AUROC +0.022 [+0.001, +0.047], mice nose-tail AUROC +0.019
  [-0.016, +0.063], but mice nose-nose AUROC -0.085 [-0.123, -0.052].
- 4x4 pooling is clearly worse everywhere (P4W - P2W mice nose-nose AUROC -0.099, nose-tail -0.103, ants -0.107).
- Video-edge artifact: with the 2x one-sided Delta at a video's first / last 2 frames, motion arms put more top-3 frames
  on video edges (nose-tail: P2W 38/648, P4W 40/648, P2N 0/648, V 4/1296; `mice/top3_video_edge_frames.json`).
- Sheets (`sheets/`; best seed per arm = highest mean nose-nose / nose-tail precision@top3/video, ants = highest
  grooming precision@top3/video; top 12 held-out frames ranked by P(on), <= 2 per video), looked at:
  - mice P2W s2403: nose-nose #807 4/12 labelled (heads close in 2-3 mouse clusters); nose-tail #91 0/12: single or
    loosely grouped moving mice, several shown frames are video-edge frames (k0, k4498, k4499, k8999).
  - mice P2N s2403: one neuron (#1902) picked for both behaviours; nose-nose 2/12, nose-tail 1/12; fires where one
    mouse's head touches another's flank / back (body contact, mostly unlabelled).
  - mice P2C s2401: nose-nose #316 1/12 (mice huddled / side by side); nose-tail #656 3/12, the most plausible sheet:
    one mouse's head at another's rear or flank.
  - mice P4W s2403: nose-nose #295 1/12, nose-tail #42 0/12; the max 64-px cell usually sits on bedding next to a mouse.
  - ants P2N s2402 #107 11/12 (three ants clustered in contact around the focal); P2C s2402 #71 9/12 (same kind of
    cluster); P2W s2403 #97 6/12 (ants close or approaching, several without grooming); P4W s2402 #400 6/12 (12 x 12
    grid; max cell on the arena floor beside the ants).
