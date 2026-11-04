# ECI linear-probe ceiling - LOG

Question: how much behaviour information is linearly available in the raw encoder tokens, the full SAE code, and the
single best SAE neuron (the ECI readout)? Does the SAE throw behaviour signal away, and is DINOv3 better than DINOv2
when the whole representation is used?

Code: `scripts/eci/probe/{feats.py, probe.py, report.py, job.sh}`. Features (float16, per frame):
`/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/probe/<dom>/parts*`; predictions `.../probe/preds`; bootstrap `.../probe/boot`.

## 2026-11-04 12:20 - frame sets (counts: `frames.json`)

Mice: every labelled frame of the 144 annotated videos at 5 fps = **864,000 frames** (24 pools x 6 videos):
- the 15-min atlas window: 648,000 frames (identical row set to `eci_codes/v2_2/full/frames.parquet`, asserted);
- the labelled first 15 min of the 48 habituation (`_H`) videos: 216,000 frames.
- Annotator split: SG 11 pools (396,000 frames), non-SG 12 pools (432,000), unknown 1 pool (rd41_3 = het_kmt5b_m_3,
  36,000 frames; in neither the training nor the test set of either direction).
- Positive frames (all 864,000): nose-nose (nose_nose) 18,269; nose-nose-mutual (nn) 6,516; nose-nose-passive (np) 11,772;
  nose-tail (nt) 9,032.

Mice 1 fps common subset = **172,800 frames** (exactly 1 in 5 of every video): the frames held by the v1 DINOv2 token
store `dataset/mice/v1/eci/train_tokens/dinov2_base_l-1_fg448al_fps1` (stride 5, with a PER-VIDEO offset - first
assumed k % 5 == 4, which is wrong; caught by an assertion, grid now read from the store rows). The raw DINOv2 tokens
(and so v1 SAE codes of them) exist ONLY on this subset; a 5 fps DINOv2 encode needs a GPU (estimated start > 2 h, not
done). Positives on it: nose-nose 3,666; nn 1,294; np 2,378; nt 1,796. Same frame set as the 2026-10-08 patch diagnostic
(`results/vision/eci_repr_diag/mice/a`, 172,800 frames, 144 videos).

Ants: every labelled frame on the eval.py 1 fps grid (k % 5 == 2) = **127,200 frames**, 212 videos (v2 44, v3 107,
v4 40, v4_infested 21; 600 frames each). Positives: grooming-to-focal (any) 43,113; Y2F 26,479; B2F 24,889; focal
single 34,858; focal double 8,255 (scored on v3 + v4 only: 88,200 frames, 6,776 positives).

## Representations
- d2: DINOv2-base last layer, v1's fg448al mask (mice: the stored fp16 tokens; ants: re-encoded on CPU with eval.py's
  exact v1 token pipeline - no GPU slot within 2 h). 768-d.
- v1c: v1 SAE s0 (1024 latents, threshold mode) on those tokens.
- d3: DINOv3-L/16 block 20, native res, mask v5 M_t tokens (store v2.1_m5). 1024-d.
- V: V input x_t + (x_{t+1} - x_{t-1}) / 2 (v22_full.feats_v). 1024-d.
- V_s1001-1003: v2 SAE arm V codes of the V input (2048 latents, threshold mode).
- Pooling over the frame's masked tokens: mean, max; "mm" = mean||max concatenated.

## Probe
Standardised features, L2 logistic regression, C in {1e-4 .. 1} by 3-fold grouped inner CV (AUROC) on the training
units only. Mice: train on the 1 fps frames of the training pools for EVERY column (so all columns see identical
training frames; DINOv2 exists only there); test on the 1 fps frames of the test pools (common set, every column) and
additionally on every 5 fps test frame for the v2-store columns. Ants: 5 video folds stratified by series.
Best single neuron: highest training-frame AUROC among the SAE neurons (frame max = ECI readout), raw code as score.

## Earlier related result
`results/vision/eci_repr_diag/mice/a/results.json` (2026-10-08; same 172,800 frames, 5 folds grouped by pool, NOT the
SG / non-SG split): mean-pooled DINOv2 linear probe nose-nose AUROC 0.823, AP 0.133; nose-tail AUROC 0.842, AP 0.081;
max-pooled 0.765 / 0.803. Supervised attention pooling (MIL) reached nose-tail AUROC 0.919, nose-nose 0.886.

## Jobs
- 2002097 frames (first version, wrong grid1); 2002248 frames (fixed).
- 2002119 mice3 (48 tasks; 28-43 failed on eta380: no /localhome -> 2002286); 2002120 mice1 (failed: grid1 bug) ->
  2002249 (8-12 failed on eta380 -> 2002281); 2002121 ants3 (8); 2002122 ants1 (32, CPU encode).
- job.sh now falls back to $TMPDIR on nodes without /localhome.
