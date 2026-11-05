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
