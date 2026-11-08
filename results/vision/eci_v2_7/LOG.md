# ECI v2.7 log: confirm PRED + Matryoshka emergence arms

Pre-registration: `scripts/eci/v2/v27_prereg.md` (19539ef, before training / scoring). Code: `v27_train.py` (prep, proxydiag,
train), `v27_eval.py` (v2.5 scoring code redirected to v2.7; rule dist2 added to v25_eval.select), `v27_levels.py`,
`v27_summary.py`. Big files: `/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_7/`.

## Timeline (2026-11-08)
- 11:27 v2.6 finished first: N0x added to the mice report (07fa067); complete-report decision PRED|auc passes temporal B
  (nose-nose AUROC LCI vs N0w-c -0.017; -0.041 before N0x: bootstrap-stream sensitive), logged in v2.6 LOG (fef7f72).
- 11:36 prep (CPU): contact proxy fraction mice 2.34% -> W_C 10.43, ants 4.75% -> 5.01; proxy identity 0 mismatches;
  PCA-64 hard weights mild (1-99% 0.48-2.20, none clipped, ESS 89%). Committed with the prereg (19539ef).
- 11:37 training mice 2401193 (gpu267), ants 2401194 (gpu277); 12 SAEs each; GPU == numpy (x_t, x_t+2) bit-exact.
  lambda_Delta (label-free warm-up): mice 0.4592879241742148, ants 0.46387905890672143. Mice 11.1 steps/s, ETA ~177 min.
- 11:38 baseline tags (V s1001-1006, v1 s0, PRED26 = v2.6 PRED s2601-2603) with rule dist2: first submitted by mistake with
  V25_EDGE=0 (original frame set) -> cancelled within a minute, resubmitted in edge mode.
- 11:50 rule dist2 finds no eligible nose-nose-specific neuron for v1 s0 (all eligible neurons have discrimination AUROC > 0.4) -> select falls back to the dist pick for that model (logged; v1 is compared under rule auc only).
- 11:55 proxy diagnostic (mice, diagnostic only): frames with >= 1 proxy token: nose-nose labelled 77.1% vs not 67.3%;
  nose-tail 75.2% vs 67.5%; mean proxy tokens 5.8 vs 4.8 -> weak enrichment (1.2x).
- 12:06 interim mice report (baselines + PRED26 only): v2.6 PRED rescored with rule dist2 (readout only): nose-nose AUROC
  0.675 [0.605, 0.738] vs 0.815 under dist (-0.140 [-0.207, -0.071]); nose-nose precision@top3/video 0.091 vs 0.164;
  dist2 picks 3602 (L5), 2545 (L5), 1023 (L4). Forcing nose-nose specificity on the v2.6 PRED dictionary costs a lot.
- 12:34 ants training done (2401194, ~57 min). Eval FVE PRED-c 0.859, PRED-delta 0.866, PRED-contact 0.858, PRED-hard
  0.859; dead latents <= 10 / 4096.
- 14:49 ants report (first attempt on eta393 crawled, cancelled after 30 min; rerun took 33 min: the report re-reads the
  v1 s0-s10 ants code files from NFS once per direction, ~70 GB -- heavy NFS load, to be cached next time).
  A -> B grooming-to-focal (any) AUROC: PRED-c 0.861 (- V -0.001 [-0.016, +0.016]), PRED-delta 0.852, PRED-contact
  0.858, PRED-hard 0.867; vs PRED-c (paired): delta -0.008 [-0.023, +0.007], contact -0.002 [-0.020, +0.017], hard
  +0.006 [-0.009, +0.020]; precision@top3/video PRED-c 0.800, contact 0.808, hard 0.811 (V 0.746, v1 s0 0.788).
