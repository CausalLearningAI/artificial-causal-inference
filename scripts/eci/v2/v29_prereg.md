# ECI v2.9 pre-registration: confirm rule sgap on fresh PRED-contact dictionaries, and vary the sparsity k

Written 2026-11-25 ~07:30, BEFORE any v2.9 SAE was trained or scored. Seen before writing: all v2.5-v2.7 results and the
v2.8 audit (`results/vision/eci_v2_8/audit/`, commits 42d36f4, 2f981fd).

What the audit found (POST HOC, the reason for this run):
- The deliverable is ONE sparse-autoencoder dictionary, so the choice is per dictionary, not on seed means.
- Rule "sgap" (written after the held-out numbers were seen) finds sparse, specific nose-nose and nose-tail neurons in
  the v2.7 PRED-contact arm (PREDk) seeds 2702 and 2703 (held-out nose-nose AP 0.190 / 0.230, precision@top3/video
  0.556 / 0.528). Seed 2701 has no such nose-nose neuron (held-out nose-nose AP 0.039). It needs confirmation on fresh seeds.
- Choosing a dictionary by its selection-pool (annotator SG) AP sum predicts the held-out AP sum (Spearman 0.86, 33 dictionaries).
- The BatchTopK sparsity k (16) was never varied in v2.

## Data and scoring frames (= v2.7)
- Mice: store v2.1_m5 (DINOv3-L/16 block 20, native grid, 5 fps, mask v5 tokens), edge rule 2 <= t < T-2. Scored frames =
  the full edge-excluded ext frame set, 863,424 frames. Selection = annotator SG's 11 pools (431,712 frames).
  Held-out = the 12 non-SG pools, 72 videos, 431,712 frames (nose-nose prevalence 1.57%, nose-tail 0.85%).
- Ants: v3 + v4, 1 fps, halves A / B (v2.3 split), A -> B (primary) and B -> A.
- Codes: frame code = max over the frame's mask tokens of the thresholded SAE code (v2.5 `cmd_codes`); P(on) = the v2.5
  3-state HMM readout per video (`v25_eval.cmd_cache`). Same code paths as v2.7 (`v27_eval.py`), redirected to v2.9.

## Arms (all PRED-contact = v2.7 PREDk; only k changes)
- PREDk recipe, unchanged: input x_t (1024), 4096 latents, Matryoshka prefixes 128/256/512/2048/4096, second decoder D1
  predicting x_{t+2} from the same code, loss = L0 + 1 x L1 (both Matryoshka prefix sums), per-token weight W_C on
  contact-proxy tokens (the v2.7 values, NOT re-measured: mice W_C = 10.43, ants 5.01), AuxK (k_aux 512, coefficient 1/32,
  dead after 2M tokens), Adam lr 5e-4, 500 warm-up steps, cosine to 10%, batch 4096, TokenNorm, the v2.5/v2.6 pass-0
  window order, per-seed batch permutation. Mice 120,000 steps (0.66 pass), ants 25,561 steps (1 pass).
- How k enters (checked in `v27_train.py` / `src/eci/sae.py`): (1) training BatchTopK keeps the batch x k largest
  post-ReLU activations over ALL 4096 latents (no per-level quota); (2) the running training threshold is the EMA of the
  smallest kept value; (3) the inference threshold theta* is calibrated after training so that the mean active latents per
  token on 1.5M held-out training tokens equals k (`calibrate_threshold`, default k = sae.k). The Matryoshka prefixes, the
  loss, AuxK (k_aux = 512 fixed) and the dead-latent rule do not depend on k. So changing k changes only (1)-(3).
- Arms and seeds (one mice job and one ants job, 12 SAEs each):
  - **K16** (k = 16 = v2.7 PREDk): seeds 2901-2906 (6 fresh).
  - **K8** (k = 8): seeds 2901-2903.
  - **K32** (k = 32): seeds 2901-2903.
  - Seeds 2901-2903 of K8 / K16 / K32 have the same initial weights and see the same batches (paired).
- Code: `scripts/eci/v2/v29_train.py` (the v2.7 training loop with a per-SAE k; arm names PREDk8 / PREDk16 / PREDk32).
- Identity check before the long run: `v29_train.py` with k = 16, seed 2701, mice, 5,001 steps must reproduce the stored
  v2.7 PREDk_s2701 training history (loss at step 0 and step 5,000; bit-identical expected, any difference reported).

## Rule sgap (frozen = `v28_audit_combine.py` section 6b, plus a firing cap)
Per dictionary, statistics on the SG frames only (labels of held-out pools never used):
- Eligible = the v2.5 rule (P(on) > 0 on >= 0.2% of all 863,424 frames; label-free).
- SG AUROC and SG AP per neuron and behaviour on the P(on) readout (AP: frames sorted by P(on), raw code as tie-break);
  SG discrimination AUROC = nose-tail-only frames (positive) vs nose-nose-only frames.
- Candidates per behaviour = top-20 eligible neurons by SG AUROC UNION top-20 eligible by SG AP.
  (The audit code also added the auc / dist / dist2 picks and the held-out-best-AUROC neuron to the candidate set; v2.9
  drops those extras. Recomputed from the audit pickles: the strict candidate set gives the same picks for PREDk s2701,
  s2702 and s2703: #359 / #3776, #4027 / #2063, #3593 / #3009.)
- **Firing cap (new):** raw firing rate < 10%. Raw firing rate of a neuron = fraction of the 863,424 edge-excluded mice
  frames whose raw frame code (the stored fp16 max-over-tokens code) is > 0. Label-free, computed on all frames (same
  frames as the eligibility rule). For the audit picks this cap changes nothing: PREDk s2702 nose-nose #4027 fires on
  5.64% of frames, nose-tail #2063 on 0.63%; PREDk s2703 #3593 on 1.45%, #3009 on 3.15% (v2.8 final.json), and removing
  candidates cannot change an argmax that survives the cap.
- Nose-nose pick = the candidate with the best SG nose-nose AP among nose-nose candidates with SG discrimination AUROC
  <= 0.4 and raw firing < 10%.
- Nose-tail pick = the candidate with the best SG nose-tail AP among nose-tail candidates != the nose-nose pick with SG
  discrimination AUROC >= 0.6 and raw firing < 10%.
- If either candidate set is empty, the dictionary has no sgap pair (counted as not accepted; SG AP sum = 0 for medians).
- **Dictionary score** = SG (nose-nose AP + nose-tail AP) of its sgap pair (SG pools only).

## Acceptance criteria (held-out = 12 non-SG pools, 72 videos, 431,712 frames)
A mice dictionary is ACCEPTED (numeric part) if its sgap pair meets all of:
- nose-nose neuron: raw firing < 10% of frames; held-out AP lower 95% CI >= 0.10; held-out precision@top3/video >= 0.35;
  held-out AUROC >= 0.75.
- nose-tail neuron: a different neuron; raw firing < 10%; held-out AP lower 95% CI >= 0.10; held-out precision@top3/video
  >= 0.30.
- top-3-frame overlap between the two neurons < 10%.
- Definitions: AUROC / AP on the P(on) readout (AP raw-code tie-break) over all held-out frames; 95% CI = percentile
  bootstrap over the 72 held-out videos (1,000 resamples, rng 20261125); precision@top3/video = raw-code ranking, the 3
  highest valid frames per video (earliest first on ties), averaged over videos with >= 1 positive (the official v2.5
  metric); top-3 overlap = in each held-out video, the 3 frames with the highest raw code (all frames) for each neuron;
  overlap = |nose-nose top frames AND nose-tail top frames| / |nose-nose top frames| (as `v28_audit_final.py`).
- Not numeric (made later by another agent, not part of this run's decisions): top-frame sheets >= 8/12 frames labelled
  by eye, each neuron.
- Ants (per dictionary): pick per direction = best selection-half grooming-to-focal (any) AUROC (v2.7 rule); accepted if
  held-out grooming-to-focal (any) AUROC >= 0.85 AND precision@top3/video >= 0.75, in BOTH directions (A -> B and B -> A).

## Questions and decisions (applied literally)
- **Q1 confirmation:** number of the 6 fresh K16 dictionaries (s2901-2906) whose sgap pair meets the numeric criteria;
  reported next to the old PREDk s2701-2703 scored by the same v2.9 code (audit: 2 of 3).
- **Q2 k:** per arm on seeds 2901-2903: fraction of dictionaries accepted and median SG AP sum. k changes from 16 only if
  K8 or K32 has a STRICTLY higher accepted fraction AND a higher median SG AP sum than K16 (K16 on seeds 2901-2903 only).
  If both qualify: the higher accepted fraction, then the higher median SG AP sum.
- **Q3 deliverable candidate (mice):** among all PREDk dictionaries of the winning k (k = 16: K16 s2901-2906 AND the v2.7
  PREDk s2701-2703; otherwise that arm's s2901-2903) that meet the numeric criteria, the one with the maximum SG AP sum.
  Its held-out numbers are reported. If none meets the criteria: no deliverable (the max-SG-AP-sum dictionary is reported
  as not accepted). For fresh seeds the held-out pools are not used to pick neurons or to rank dictionaries; they enter
  only through the pass/fail acceptance gate.
- **Ants deliverable:** the same arm and seed's ants dictionary if it meets the ants criteria; else the ants dictionary of
  the winning k with the best selection-half (A) grooming-to-focal (any) AUROC of its A -> B pick (reported with pass/fail).

## Reported (per dictionary: 15 new + the 3 old PREDk; v1 s0 and V s1001-1006 on the same frames from existing files)
- sgap picks: neuron ids, Matryoshka level (1-5 = latents 1-128, 129-256, 257-512, 513-2048, 2049-4096), raw firing rate,
  SG and held-out AUROC, AP [video CI], precision@top3/video and precision@top10/video (+ chance = mean per-video
  prevalence over videos with >= 1 positive), top-3 overlap, per-video time bias and MAE (sum of P(on) x 0.2 s minus true
  time; mean over held-out videos +- SE), ground-truth prevalence.
- The official v2.7 rules auc and dist (and dist2) on the same dictionaries for continuity (v2.5 report machinery).
- Training health: final eval FVE, L0 per token, dead latents (train) and never-firing fraction (eval) per SAE.
- Ants: per dictionary and direction, grooming-to-focal as single / double / or (any): held-out AUROC [video CI],
  precision@top3/video [CI], AP.
- Anything else is labelled POST HOC.

## Budget
- GPU: partition gpu100 (H100), as v2.7 training; total <= 10 H100-hours (expected ~5: mice ~3.7 h, ants ~1.1 h,
  identity check ~0.2 h). If the queue blocks > 1 hour, the next suitable GPU partition is used and reported.
- Big files: `/mnt/beegfs/locatgrp/shared/rcadei/eci_codes/v2_9/`.
