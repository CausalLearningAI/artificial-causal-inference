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
