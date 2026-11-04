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

## 12:45 - timing (measured) and projected wall time
- Pooled features: mice v2 store, 48 CPU tasks x 16 cores, every video read once, d3 / V / 3 V seeds' codes (mean + max)
  in one pass: ~10 min for 864,000 frames. Mice v1 (stored 1 fps DINOv2 tokens, v1 SAE codes): 16 tasks, < 1 min each.
  Ants v2 store: 8 tasks, ~6 min.
- Ants DINOv2 tokens do not exist on the eval grid; GPU start estimate was > 2 h, so they are re-encoded ON CPU
  (32 tasks, ~1.7 frames/s per task, ~40 min; started 12:22, the longest single step).
- Probes: sklearn lbfgs logistic regression trained on the 1 fps frames of the training units only (79-86k frames for
  mice, ~44k for ants), never on 864k x 2048; 5 C values x 3 inner folds. Mice: 72 tasks, all done in ~5 min.
- Projected: ants probes + bootstrap + tables done by ~13:20, i.e. ~1 h after the start (12:20).

## 12:40 - ants design change (user decision via coordinator)
Ants now: v3 + v4 only (147 videos; v2 and v4_infested excluded), first 10 min at 1 fps (88,200 frames), half A / half B
by video stratified by series x treatment (`ants_split.csv`, rng 20261104: A 73 videos, B 74). A2B = train A test B
(primary), B2A = swapped. Replaces the 5-fold CV. SAEs probed: v1 deployed s0 (`dataset/ants/eci/sae/
matryoshka_btk_1024_k16_antsfg_s0`) and the current v2 arm V seeds 1001-1003 (`eci_codes/v2_1/ants/sae/V_s100x`);
the v2 SAE is being retrained elsewhere.

## 14:35 - done (all jobs completed; tables.md, summary.json, checks.json, ants_split.csv)
Jobs: features 2002119/2002286 (mice3), 2002249/2002281 (mice1), 2002121 (ants3), 2002122 (ants1, CPU encode, 44 min);
probes 2002993 (mice, 72), 2003101 (ants, 90); bootstrap 2003127 (mice), 2003147 (ants); checks 2003148; table 2003149.

Sanity checks:
- Label-permuted probes (whole label sequences swapped between training videos of equal length): mice AUROC 0.51-0.57
  (sg2non, 1 fps), ants 0.44-0.53 (A2B). The mice residual above 0.5 is plausible because the swap keeps video type
  (habituation videos only swap with habituation videos), which carries prevalence.
- V_s1001 frame max from this pass vs the v2.2 full cache: fire agreement 1.0, max abs diff 0.016 (fp16 rounding).
- v1 SAE frame max: from the stored fp16 1 fps tokens vs stored deployed codes, fire agreement 0.9997 (mice);
  ants CPU re-encode vs eval.py GPU codes 0.9997; 1.2% of entries differ by > 0.01 (encoder fp32 CPU vs GPU rounding
  before the fp16 token cast).
- Best neuron vs results/vision/eci_v2_2/full (V|auc, sg2non, 324,000 non-SG atlas frames): the v2.2 P(on) AUROC is
  reproduced exactly from the cache (0.836 / 0.788 / 0.836 nose-nose, 0.828 / 0.822 / 0.803 nose-tail). On the raw
  frame max (the readout the probe table uses) the same neurons score 0.035-0.17 lower (e.g. s1001 nose-nose 0.802).
  My raw-max pick (selected on SG 1 fps frames incl. habituation first 15 min) equals the v2.2 pick in 4/6 cases.
- Caveat found: mean-pooled SAE-code probes give precision@100 = 0 for 2/3 v2 seeds on mice nose-nose although AUROC
  is 0.83-0.85: all top-100 frames come from one video (wt_kdm6b_m_1_F_H), a video-level outlier the linear probe
  extrapolates on. AUROC / AP are not driven by it.

Retrained v2 SAEs (eci_codes/v2_3/*/sae/Vall_s2001-2003) exist but were NOT probed (not waited for).
