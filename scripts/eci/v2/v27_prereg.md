# ECI v2.7 pre-registration: confirm PRED, and make behaviour neurons emerge earlier in the Matryoshka hierarchy

Written 2026-11-08 ~11:40, BEFORE any v2.7 SAE was trained or scored. Seen before writing: all v2.6 results.
- PRED (second decoder predicting x_{t+2}) has the best nose-tail neuron so far.
  - Mice nose-tail precision@top3/video 0.369 vs V 0.086; nose-tail AUROC 0.857.
  - Sheet: 10/12 labelled, snout at another mouse's rear.
  - It passes temporal B under rule auc in the complete v2.6 report. The nose-nose AUROC lower CI vs N0w-c was −0.017;
    that number is sensitive to the bootstrap stream (−0.041 without N0x).
- Its nose-nose pick is non-specific: 2/12 labelled, fires on any head-to-body contact.
- In all 3 seeds, the PRED nose-tail pick sits in Matryoshka level 5 (latents 2049-4096).

## Data, recipe, pairing (= v2.6)
- Store v2.1_m5, DINOv3-L/16 block 20, native grid, 5 fps, mask v5 M_t tokens.
- Edge rule: frames 2 ≤ t < T−2. Lag L = 2.
- Input of every arm: x_t only.
- SAE: 4096 latents, Matryoshka prefixes 128/256/512/2048/4096, k = 16, AuxK, Adam lr 5e-4, 500 warm-up steps, cosine to
  10%, batch 4096, TokenNorm with one scalar.
- Seeds 2701-2703.
- Stream: the v2.5/v2.6 pass-0 window order. All arms of a seed see the SAME batches (paired).
- One job per domain trains all 4 arms × 3 seeds from one store read.
- **Training length (5 H100-hour budget):** mice 120,000 steps (about 0.66 pass of the 747.7M edge-excluded tokens; v2.6
  PRED had 50,000 steps); ants 25,561 steps = 1 pass. Same for all arms.
- Code: `scripts/eci/v2/v27_train.py`. Scoring: the v2.5 / v2.6 code via `v27_eval.py`.

## Arms (all PRED-type: D0 reconstructs x_t, D1 decodes a target from the SAME code z_t)
Loss = L0 + λ·L1, both Matryoshka prefix sums.

| arm | D1 target | λ | per-token weight w (multiplies L0 and L1; AuxK unweighted) |
|---|---|---|---|
| **PRED-c** (PREDc) | x_{t+2} | 1 | 1 (= the v2.6 PRED recipe; confirmation and paired baseline) |
| **PRED-Δ** (PREDd) | norm(x_{t+2}) − norm(x_t), b_dec1 = 0 | λ_Δ (label-free, below) | 1 |
| **PRED-contact** (PREDk) | x_{t+2} | 1 | W_C on contact-proxy tokens, 1 elsewhere |
| **PRED-hard** (PREDh) | x_{t+2} | 1 | min(10, e(x)/mean e), e = squared residual of raw x_t after a fixed top-64 PCA |

- **λ_Δ (label-free):** measured in the training job before the PRED-Δ SAEs exist.
  - A throwaway PRED-Δ SAE (seed 2700, λ = 1) trains for 2,000 steps on the first chunk.
  - λ_Δ = mean L0 / mean L1 over steps 1,500-1,999, so the Δ term equals the x_t term at the end of the warm-up.
  - Logged in `results/vision/eci_v2_7/train/<dom>_lam_delta.json`.
- **Contact proxy (label-free, from the stored mask v5 of frame t on the native grid, 8-connectivity):** proxy = A or B.
  - A, "neck": a mask token removed by a 3×3 erosion whose 5×5 neighbourhood contains ≥ 2 distinct components of the
    eroded mask (a narrow bridge between two body blobs).
  - B, "near other": a mask token whose 5×5 neighbourhood contains a token of a different, un-eroded mask component (two
    animals within 2 patches).
  - Proxy token fraction p is measured on the rng-6 sample (2M tokens).
  - W_C = 0.2(1−p) / (0.8p), so proxy tokens carry 20% of the total weight, capped at 30 (if the cap binds, the realised
    share is reported).
  - Identity check: the vectorised proxy == an independent per-token loop on 20 windows.
  - DIAGNOSTIC ONLY (computed after the weights are fixed, never used for tuning): the proxy's rate on labelled
    nose-nose / nose-tail frames vs other mice frames.
- **Hard weights:** PCA (top 64 eigenvectors of the covariance of 500k tokens of the rng-6 sample, rng 27). The
  normaliser mean e is taken over the 2M sample. Clipped at 10, not renormalised. Distribution reported.

## Selection rules (mice)
- **auc** and **dist**: as v2.6.
- **dist2** (new):
  - Nose-tail = the dist pick: best nose-tail AUROC among eligible neurons ≠ the best nose-nose-AUROC neuron, with
    discrimination AUROC (nose-tail vs nose-nose frames) ≥ 0.6.
  - Nose-nose = best nose-nose AUROC among eligible neurons ≠ that nose-tail pick, with discrimination AUROC
    (nose-nose vs nose-tail) ≥ 0.6, i.e. the nose-tail-positive discrimination AUROC ≤ 0.4.
- dist2 is applied to all v2.7 arms. As a readout-only rescore it is also applied to the stored v2.6 PRED models
  (seeds 2601-2603; same caches, new tags).

Mice: select on SG's 11 pools, measure on the 12 non-SG pools, edge-excluded ext frames (863,424). Ants: v3 + v4, A → B
(primary) and B → A, best AUROC.

## Reported per arm × rule
- Matryoshka level of every pick per seed (levels 1-5 = latents 1-128, 129-256, 257-512, 513-2048, 2049-4096).
- AUROC and precision@top3/video (with chance), distinct picks, discrimination AUROC, per-video MAE.
- Contrasts vs PRED-c (paired), vs V and vs v1 s0.
- The v2-vs-v1 table for the deploy pick (mice atlas subset, v1 seeds 0-10, as v2.6).
- **Level curve** for the best arm: for m = 1..5, the held-out AUROC of the best selection-set neuron restricted to
  levels 1..m. Behaviours: mice nose-nose, mice nose-tail (rule auc within the level set, SG → non-SG), ants grooming
  (A → B).

## Pre-registered decisions
- **PRED confirmed** if PRED-c under rule dist OR dist2, vs V|auc, meets all of:
  - nose-tail precision@top3/video lower 95% CI > 0;
  - nose-tail AUROC lower CI ≥ −0.02;
  - nose-nose AUROC lower CI ≥ −0.02;
  - ants AUROC (A → B) lower CI ≥ −0.02;
  - distinct picks 3/3.
- **An emergence arm (PRED-Δ / PRED-contact / PRED-hard) replaces PRED-c** if, under the same rule, BOTH:
  - it is non-inferior to PRED-c (paired): AUROC lower CI ≥ −0.02 for mice nose-nose, mice nose-tail and ants, and
    nose-tail precision@top3/video lower CI ≥ −0.05;
  - AND at least one of:
    - its nose-tail pick sits in an earlier Matryoshka level than PRED-c's in ≥ 2/3 seeds (seed-paired);
    - nose-nose precision@top3/video gain with lower CI > 0.
- **Deploy pick:**
  - the replacing arm (if several, the higher nose-tail precision@top3/video);
  - else PRED-c if confirmed (rule: dist2 if it passes, else dist);
  - else none (reported).

## Figures
- Side-by-side sheets v1 s0 | PRED-c | best emergence arm (highest mice nose-tail precision@top3/video under rule dist),
  for mice nose-nose, mice nose-tail and ants grooming-to-focal (any).
- Plus the v2.6 PRED rescored under dist2 (mice nose-nose).
- Format: 12 held-out frames, at most 1 per video, raw-code ranking; best seed = highest held-out nose-tail
  precision@top3/video (ants: grooming); described in words.

## Measured before training (11:36, CPU jobs 2401142-2401143; nothing trained or scored)
- Contact proxy token fraction: mice 0.02341 -> W_C = 10.43; ants 0.04751 -> W_C = 5.01 (cap 30 not binding; proxy share
  of the weight = 20.0%). Proxy identity (vectorised == per-token loop): 0 mismatches on 20 windows (mice 11,393 tokens,
  ants 3,394).
- Hard weights: PCA-64 explains 86.9% (mice) / 87.7% (ants) of the token variance; normalised residual weight quantiles
  (1/10/50/90/99%) mice 0.48 / 0.64 / 0.94 / 1.41 / 2.19, ants 0.53 / 0.67 / 0.94 / 1.38 / 2.20; none clipped at 10;
  effective sample size 89.2% (mice) / 89.7% (ants). The hard reweighting is MILD (stated before any result).
- lambda_Delta: measured in the training job (rule above).
