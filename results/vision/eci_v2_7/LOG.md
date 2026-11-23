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

## Timeline (2026-11-23): final mice report + pre-registered decisions
- 16:52 (11-08) all 12 mice v2.7 post chains (cache / tag / prep / fwdcheck) had finished; the final mice report was never built.
- 09:40 mice report, all 22 models (V s1001-1006, v1 s0, PRED26 s2601-2603, PREDc/d/k/h s2701-2703): CPU job 3065948,
  41 s, same invocation as the interim (`v27_cpu.sh v27_eval.py report --dom mice`). Interim kept as `mice/report_interim.json`.
  Held-out = 12 non-SG pools, 72 videos, 431,712 edge-excluded frames; prevalence nose-nose 1.57%, nose-tail 0.85%.
- Level curves (`v27_levels.py`, jobs 3065958-3065959, ~9 min each) for PRED-c and the best emergence arm under rule dist
  (PRED-hard, by v27_summary `best_emergence`) -> `levels/PREDc.json`, `levels/PREDh.json`.
- `v27_summary.py` -> `summary.json`, `tables.md`.
- **PRED NOT confirmed** (PRED-c vs V|auc). Rule dist: nose-tail precision@top3/video +0.116 [-0.032, +0.306] (LCI <= 0, FAIL),
  nose-tail AUROC +0.044 [+0.003, +0.087] (pass), nose-nose AUROC -0.029 [-0.054, -0.001] (LCI < -0.02, FAIL), ants AUROC
  LCI -0.016 (pass), distinct 3/3 (pass). Rule dist2: nose-tail precision@top3 LCI -0.032 (FAIL), nose-tail AUROC LCI
  +0.001 (pass), nose-nose AUROC -0.163 [-0.226, -0.090] (FAIL), ants pass, distinct 3/3.
- **No emergence arm replaces PRED-c** (no arm non-inferior under any rule). Closest: PRED-delta|dist fails only on ants
  AUROC (LCI -0.0228 vs -0.02); it would otherwise qualify via nose-nose precision@top3/video gain +0.144 [+0.059, +0.221];
  nose-tail picks are never in an earlier level in >= 2/3 seeds (PRED-contact|dist: 1/3).
- **Deploy pick: none** (per the pre-registered rule).
- PRED-c fresh seeds reproduce the v2.6 PRED nose-tail effect only partly: nose-tail precision@top3/video (rule dist) 0.203
  [0.060, 0.398] vs PRED26 0.369; nose-tail AUROC 0.835 vs 0.857; nose-nose AUROC 0.777 vs 0.815.
- Not done: top-frame sheets (no v2.7 sheet wrapper exists; needs new code + frame re-encoding), v2-vs-v1 atlas table
  (no v2.7 port of v26_v1.py).
