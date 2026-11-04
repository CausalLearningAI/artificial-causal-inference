# ECI v2.2 overnight log (distinct nose-nose / nose-tail neurons)

Goal: a v2 whose selected nose-nose (nose_nose) neuron and nose-tail (nt) neuron differ and whose top frames show each
behaviour; v2 activations less localized. Mice selection on annotator SG labels only. Ants = non-regression.
Outputs: results/vision/eci_v2_2/, scripts/eci/v2/v22_*, big files /mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_2/.

## Timeline
- 02:38 start. Read v21 extended/freeze/small code. Facts: mice eval = 144 windows x 600 frames (86,400), 24 pools
  (eval unit = pool). Annotator per (pool, odour) in data/mice/v1/experiment.csv (observation_id = frame-table obs).
  SG = 12 pools (72 videos). rd11_2 (MB/CP) and rd14 (MR/MB) split by odour. AuxK already in the SAE trainer.
  Token store: eci_tokens/v2_full/mice/v2.1_m5, grid G = 64 (idx = row*64 + col), ~195 M_t tokens / frame on mice.
- 02:50 code: scripts/eci/v2/v22_core.py (frames+annotator, readout cache, AUROC kernels, selection rules auc/bal/balpair,
  Step-0 diag), v22_encode.py (2x2-pooled prep on CPU; eval codes max/box3/coarse/ms on CPU or GPU), v22_eval.py (tags +
  bootstrap report), v22_train.py (variants VP2 / VMS / VW8), v22_run.sh (job wrapper), v22_step0.sh.
- Jobs: Step0 cache+diag array 1908236 (17 models, defaultp); prep P2 mice 1908294 (defaultp CPU);
  codes for stored V_s1001-1006 1908330 (defaultp CPU; first submitted as GPU 1908295, cancelled: est. start 21:34).
- GPU queue estimate on -p gpu was ~19 h -> SAE trainings go to gpu100 (allowed, <= 15 H100-h). VW8 seeds 1004-1006:
  1908516-1908518 (gpu100).
- 03:05 Step 0 done (step0.md): V already has distinct, specific nose-tail neurons in 6/6 seeds (in-sample); the "same
  neuron" in v2.1 came from the cross-fitted AUROC selection. Nose-nose neurons are only mildly specific.
  rd41_3 treated as unknown annotator (csv says SG; user says unrecorded): SG = 11 pools.
- prereg scripts/eci/v2/v22_prereg.md committed before any held-out score.
- 03:05 jobs: tags (V s1001-6, v1 s0-10, V@top10) array 1910055; ants V tags 1910241 (v22_ants.py, C4 reference);
  VW8 trained (gpu100, ~6.5 min each, FVE 0.854 s1004); VW8 codes (CPU, max readout) 1910093/5/7;
  VP2 / VMS trainings (gpu100, after prep) 1910480-1910485.
- 03:18 tags V / v1 done; identity vs v21 extended: picks identical for all 17 models, max abs diff 0 (V) / 2e-6 (v1,
  HMM refit) over 1730 values each. First report (V, v1 arms): with SG selection on all 11 SG pools the CURRENT rule
  already gives distinct picks in 6/6 V seeds (nt disc 0.72, nose_nose disc 0.41). V|bal: C1 6/6, C2 yes (0.80/0.33),
  C3 NO (nose_nose AUROC -0.066 [-0.146, +0.013] vs V|auc).
- 03:25 prep P2 done (6.65M pooled train tokens, 2.77 native tokens per pooled token). Added POST HOC rule 'hyb'
  (nose_nose by auc, nt by bal) - exploratory only. Jobs: VW8 tags 1911091; VP2/VMS codes 1911096-1911106 (even ids)
  and tags (odd ids); V readout tags (box3/coarse/ms) 1911xxx (after codes 1908330).
- 03:45 codes on CPU vs stored GPU codes (V s1001-1006 native frame max): differences are threshold flips only (theta* = 2.23): 1e-7 of entries differ by > 0.5 (one side 0, other ~theta*), 3.7e-6 by > 0.01 (V_s1001, 20k frames checked); max abs diff 2.23-2.24 reported in results/vision/eci_v2_2/codes/mice_Vstored.json.
- 04:15 pre-registered arms scored (except VW8): every variant passes C1 (distinct) and all V-readout variants + VMS pass
  C2, but ALL fail C3: under 'bal' the nose-nose pick is a more specific but weaker detector (nose-nose AUROC -0.064 to
  -0.135 vs V|auc). VP2 (pooled-token SAE) is worse everywhere (nose-nose AUROC 0.658, v1-like).
  Ants V tags (1910241) done: identity vs extended exact (AUROC diff 0, picks identical).
- 04:20 POST HOC exploratory rules added (flagged in tables): 'hyb' (nose_nose by auc, nt by bal) and 'p100' (best
  precision@100 on SG). Re-tag of all models with them: 1912507 (47 keys); VW8 re-tag after its first tags.
- 04:45 post hoc rules scored: none passes C1-C3 either (V|hyb fails C2 by nose-nose disc 0.41; VMS|auc passes C1/C2 but C3 nose-nose lower CI -0.117). p100 rule overfits (AUROC -0.11 to -0.28). Ants P2 prep 1913044 (for an informative VMS ants run; prereg does not require it).
- ants VMS chain (informative C4): train 1913256/59/62 (gpu100) -> codes 1913257/60/63 -> ants tags 1913258/61/64
- 04:55 ALL mice arms scored. VERDICT: no pre-registered variant adopted (all fail C3 on nose-nose AUROC; C1 all yes; C2
  yes for V readouts, VMS, VW8). Best tie-break among non-adopted = V@top10|bal (mean p@100 0.179; nose-tail p@100 mean
  0.212, +0.120 [+0.013, +0.278] vs V|auc). Post hoc 'hyb' on V: nose-nose unchanged, nose-tail AUROC +0.021, fails C2
  only by nose-nose disc 0.41. Sheets read: V's specific nose-tail neurons are TAIL (tail-base) detectors; nose-nose
  specific picks are huddles at the food hopper. summary_table.md = compact table (v22_final.py).
- Remaining: ants VMS chain (informative C4 only) 1913256-64.
- 05:10 AMENDMENT 1 (prereg file): confirm post hoc hyb / top10 on fresh V seeds 1007-1009. codes 1915058/60/63, tags 1915059/61/64. Arms Vf|* in arms.json.
- 05:45 AMENDMENT 1 result (fresh V seeds 1007-1009): Vf|hyb C1 3/3, C2 yes (0.83 / 0.28), C3 NO (CIs too wide with 3
  seeds: nose-nose +0.000 [-0.073, +0.075], nose-tail +0.036 [-0.137, +0.228]) -> NOT confirmed by the criterion.
  Replicated: top10 readout raises nose-tail precision@100, Vf@top10|hyb - Vf|auc +0.157 [+0.007, +0.360]
  (first seeds +0.120 [+0.013, +0.278]).

## FULL WINDOW (AMENDMENT 2: re-score on the 15-min atlas window, 648,000 frames, instead of the 2-min windows)
- 10:50 error found: every mice number since v2.1 used the 2-min Tier B probe windows (144 x 600 = 86,400 frames, ~13% of
  the annotated window). Code: scripts/eci/v2/v22_full.py (+ V22_FRAMES=full|fullsub flag in v22_core / v22_eval; the
  default 'old' path is unchanged). Outputs results/vision/eci_v2_2/full/, big files eci_codes/v2_2/full/.
- 11:00 frame table (job 1956371): 144 videos x 4,500 frames (H videos: frames >= 4500) = 648,000 = Tier A rows exactly;
  old 86,400 frames an ordered subset, labels identical (asserted). full non-SG: nose-nose 4,568 frames / 1,278 bouts,
  nose-tail 2,696 / 331 bouts (2-min: 718 / 188 and 370 / 34). Window check: the FIRST 15 min of the 48 H videos are
  labelled too (216,000 frames; 1,427 nose-nose and 542 nose-tail bouts) and are outside the atlas window (not scored;
  flagged). full/frames.md.
- AMENDMENT 2 committed (e7ef64b) before any full-window score; push failed (GitHub internal server error), retrying.
- 11:03 fast HMM check (1957606): max abs diff of P(on) vs the stored V_s1001 cache 3.0e-8 over 2048 neurons x 86,400
  frames (full/hmmcheck.json). Not faster per neuron than src/eci/v2/hmm.py (1.9 vs 1.8 CPU-s / neuron); parallel over
  32 threads.
- GPU queue: estimated start 2026-11-05 16:35 or later for any -p gpu request -> v2 codes moved to CPU (array 1960601,
  72 tasks x 8 CPUs, 2 videos each). Measured: 4,316 tokens/s per task with 9 SAEs (max + top10) -> ~450 s per task,
  projected codes wall ~10 min. v1 codes (GPU array 1959004, 16 tasks) still pending in the GPU queue.
- fullsub tags (old caches, new code path) 1959792: 29 models, ~40 s each.
- 11:11 PIPELINE IDENTITY PASSED (full/identity_pipeline.json): V22_FRAMES=fullsub (new table builder, new AUROC kernel,
  fast HMM) on the old frames with the old caches vs results/vision/eci_v2_2/report.json, 14 arms x 2 designs x 644
  metric values per seed: picks identical, AUROC max abs diff 0, every v2 level value diff 0, v1 values <= 3.3e-11
  (HMM on picked v1 neurons), contrast point differences <= 3.0e-12.
