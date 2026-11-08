# ECI v2.6 log: static-input SAEs, time only through the loss

Pre-registration: `scripts/eci/v2/v26_prereg.md` (commit 190aa07, before any training or score). Code: `v26_train.py`
(kmeans, featcheck, rows, train), `v26_eval.py` (= v25_eval / v25_train codes with v2.6 paths + extra contrasts),
`v26_sheet.py` (= v25_sheet). Big files: `/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_6/`.

## Timeline (2026-11-05)
- ~09:30 budget estimate: 5 arms x 3 seeds = ~22 "4096-latent SAE units" per domain; at the v2.5 measured speed
  (mice 6 units 1 pass = 96 min) one mice pass would need ~6 H100-hours alone -> pre-registered mice 50,000 steps
  (~0.27 pass), ants 1 pass (25,561 steps); est. ~2.6 H100-hours total.
- 12:13-12:16 featcheck (x_t, x_{t+2}) bit-exact both domains; rows / counts copied from v2.5; RW k-means: weights mice
  0.728-2.076, ants 0.764-1.606, none clipped, effective sample size 98.6% (nearly balanced clusters -> mild RW).
- 12:2x prereg + measured values committed 190aa07. GPU smoke test queued behind a full gpu100 queue (72 running,
  70 pending) -> cancelled; the real jobs submitted directly (mice 2070874, ants 2070875) with the codes / merge / per-model
  chains queued behind them (`v26_score_submit.sh`). V / v1 s0 mice caches linked and tags / prep copied from v2.5.
- 14:23 ants 2070875 started on gpu100 first; the A100 copy 2090788 (-p gpu, submitted 14:13 as a fallback) cancelled before it started; dependency chain unchanged.
- 15:47-16:03 ants codes task 29 stalled on node eta338 (1 video in 37 min vs ~9 min per task elsewhere): rerun as 2104168 on eta411 (323 s, same code / same part files), straggler cancelled, merge dependency rewired to afterany(array)+afterok(rerun).
- 15:13 / 15:36 ants (2070875, 53.6 min) and mice (2070874, 92.3 min) training done -> 2.43 H100-hours for v2.6.
  mu: mice 1.309, ants 1.226. Ants N0x s2603: 2,251 / 8,192 latents dead at the end, 36% never fire on the eval sample.
- 17:10 ants report (7627f68). Mice N0wc / PRED / SMOOTH / RW post chains done overnight; the three mice N0x chains ran
  OUT OF MEMORY (120G; 8,192-latent float32 HMM on 863k frames) -> resubmitted 2026-11-06 09:56 with 360G (2166649-51).
- 2026-11-06 09:57 mice report without N0x (627d34a); v2-vs-v1 table on the atlas subset (647,520 frames; per-model cell
  jobs, then combine).
- 10:00-10:05 sheets `figures/sheet_{mice,ants}_v1_N0wc_PRED.png` (v1 s0 | N0w-c s2602 rule dist | PRED s2601 rule dist;
  best seeds by held-out nose-tail / grooming precision@top3/video). Looked at:
  mice nose-nose: v1 #611 0/12 (bodies of clustered mice); N0w-c #1352 6/12 (heads together, side-by-side huddles as false
  positives); PRED #287 2/12 (any head-to-body contact point, not face-to-face).
  mice nose-tail: v1 #414 2/12 (broad on bodies); N0w-c #2269 6/12 (along a mouse's back / flank near tails, mixed);
  PRED #2785 10/12: one mouse's snout at another mouse's tail base / rear, heat on the contact point -- the first clearly
  recognisable nose-tail neuron. Ants grooming-to-focal (any): v1 #90 10/12, N0w-c #35 11/12, PRED #40 10/12 (clusters
  of ants around the focal; v2 maps localised).
- 2026-11-06 11:07 N0x mice chains done (360G, ~70 min each; forward check <= 0.0039). Complete mice report (all 5 arms)
  rerun -> decisions recomputed. CAUTION: adding the N0x contrasts shifts the shared bootstrap random stream (rng 20261107
  consumed in pair order), and PRED|auc's nose-nose AUROC lower CI vs N0w-c moved from -0.041 (report without N0x) to
  -0.017 (complete report), across the pre-registered -0.02 threshold. The complete report is the pre-registered analysis:
  PRED|auc PASSES temporal B (nose-tail AUROC +0.061 [+0.026, +0.093], nose-tail precision@top3/video +0.277 [+0.161,
  +0.376] vs N0w-c; non-inferior vs N0w-c and V) -> deploy pick PRED (rule auc). PRED|dist still fails (nose-nose AUROC LCI
  -0.041). The decision sits on a Monte-Carlo-sensitive boundary; v2.7 (fresh seeds) is the confirmation.
